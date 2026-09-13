"""Non-blocking native JTC handle tracking, subordinate to the FR5 motion owner.

This is not an admission service, controller loop, or evidence writer. The caller
admits the exact timed trajectory and retains returned native events. A queue pop,
goal acceptance, and controller completion are deliberately different facts.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from tools.fr5_data_factory import ContractError


@dataclass
class _Submission:
    revision: str
    goal: object
    response: object
    handle: object = None
    result: object = None
    cancel_response: object = None
    cancel_requested: bool = False
    cancel_observed: bool = False
    feedback: object = None


class ArmStream:
    """One attempt's native rolling trajectory handles; never sends ServoJ.

    All methods are called by the same existing motion-owner thread. ROS futures
    are observed without waits. Feedback callbacks retain one latest native
    sample per owned goal; they never write files, advance policy queues or send.
    Only one replacement can be pending. A predecessor remains owned until its
    actual native result arrives, even after the successor has been accepted.
    """

    def __init__(self, client, *, deadline, clock):
        if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                or not math.isfinite(deadline) or deadline <= clock()):
            raise ContractError("ROS_EXEC_RESULT_TIMEOUT")
        self._client, self._clock, self.deadline = client, clock, deadline
        self._current = self._pending = self._retiring = None
        self._fenced = False
        self._submitting = False

    @property
    def owns_goals(self):
        return any(item is not None for item in (self._current, self._pending, self._retiring))

    @property
    def current_revision(self):
        return self._current.revision if self._current else None

    @property
    def fenced(self):
        return self._fenced

    def submit(self, goal, *, revision, dispatch_guard):
        """Submit a whole admitted FJT goal without waiting for any waypoint.

        The mandatory guard belongs to PickupExecutor and checks the exact goal,
        predecessor, current authority/Scene/hardware and original attempt lease.
        It must raise on rejection. This method does not grant those authorities.
        """
        if self._submitting:
            raise ContractError("ROS_EXEC_ACTIVE")
        self._submitting = True
        try:
            self._submit(goal, revision=revision, dispatch_guard=dispatch_guard)
        finally:
            self._submitting = False

    def _submit(self, goal, *, revision, dispatch_guard):
        if self._fenced or self._clock() >= self.deadline:
            raise ContractError("ROS_EXEC_CANCELLED")
        if self._pending is not None or self._retiring is not None:
            raise ContractError("ROS_EXEC_GOAL_PENDING")
        if (not isinstance(revision, str) or not revision
                or revision == self.current_revision or not callable(dispatch_guard)):
            raise ContractError("ROS_EXEC_DISPATCH_GUARD")
        retained = copy.deepcopy(goal)
        checked = copy.deepcopy(retained)
        predecessor = self.current_revision
        dispatch_guard(checked, predecessor)
        if checked != retained:
            raise ContractError("LEARNED_SERIALIZED_ACTION_MISMATCH")
        if self.current_revision != predecessor:
            raise ContractError("ROS_EXEC_ACTIVE")
        if self._fenced or self._clock() >= self.deadline:
            raise ContractError("ROS_EXEC_CANCELLED")
        # Own even a synchronously failed/ambiguous send; no implicit retry follows.
        pending = _Submission(revision, retained, None)
        self._pending = pending
        try:
            pending.response = self._client.send_goal_async(copy.deepcopy(retained),
                feedback_callback=lambda message: self._receive_feedback(pending, message))
        except Exception:
            self._fenced = True
            raise

    def _receive_feedback(self, item, message):
        if not any(item is owned for owned in (self._current, self._pending, self._retiring)):
            return
        if item.handle is not None and list(message.goal_id.uuid) != list(item.handle.goal_id.uuid):
            return
        count = 1 if item.feedback is None else item.feedback[2] + 1
        item.feedback = (copy.deepcopy(message), self._clock(), count)

    def cancel(self):
        """Fence new updates; request cancellation once handles become known.

        Neither this call nor cancellation acknowledgement proves a physical stop.
        poll() retains native terminal status, including success that raced cancel.
        """
        self._fenced = True

    def poll(self):
        """Return newly observed native facts; no disk I/O or blocking waits.

        The owner must retain these events outside the high-rate controller path.
        Missing/failed futures remain owned and fence further submissions rather
        than authorizing a replacement controller or a new attempt.
        """
        events = []
        failure = None
        if self._clock() >= self.deadline:
            self._fenced = True
        pending = self._pending
        try:
            if (pending is not None and pending.handle is None
                    and pending.response is not None and pending.response.done()):
                handle = pending.response.result()
                if getattr(handle, "accepted", None) is not True and getattr(handle, "accepted", None) is not False:
                    raise ContractError("ROS_EXEC_GOAL_RESPONSE_INVALID")
                if handle.accepted is False:
                    events.append({"revision": pending.revision, "event": "REJECTED"})
                    self._pending = None
                else:
                    pending.handle = handle
                    events.append({"revision": pending.revision, "event": "ACCEPTED"})
                    pending.result = handle.get_result_async()
                    self._retiring = self._current
                    self._current, self._pending = pending, None
        except Exception as exc:
            self._fenced, failure = True, exc
        for slot in ("_retiring", "_current", "_pending"):
            item = getattr(self, slot)
            if item is None:
                continue
            try:
                if item.feedback is not None and item.handle is not None:
                    message, received, count = item.feedback
                    item.feedback = None
                    if list(message.goal_id.uuid) == list(item.handle.goal_id.uuid):
                        from rosidl_runtime_py.convert import message_to_ordereddict
                        events.append({"revision": item.revision, "event": "FEEDBACK",
                                       "goal_id": list(message.goal_id.uuid),
                                       "received_monotonic_s": received,
                                       "samples_since_poll": count,
                                       "feedback": message_to_ordereddict(message.feedback)})
            except Exception as exc:
                # Diagnostic projection is not actuator authority. Missing
                # feedback cannot advance a queue, but must not cancel motion.
                events.append({"revision": item.revision, "event": "FEEDBACK_UNAVAILABLE",
                               "error_type": type(exc).__name__})
            try:
                if (item.cancel_response is not None and not item.cancel_observed
                        and item.cancel_response.done()):
                    item.cancel_observed = True
                    response = item.cancel_response.result()
                    identifier = list(item.handle.goal_id.uuid)
                    accepted = response.return_code == 0 and any(
                        list(info.goal_id.uuid) == identifier for info in response.goals_canceling)
                    events.append({"revision": item.revision, "event": "CANCEL_RESPONSE",
                                   "return_code": response.return_code, "goal_id": identifier,
                                   "accepted": accepted})
            except Exception as exc:
                self._fenced, failure = True, failure if failure is not None else exc
            try:
                if item.result is not None and item.result.done():
                    result = item.result.result()
                    if type(getattr(result, "status", None)) is not int or result.status not in (4, 5, 6):
                        raise ContractError("ROS_EXEC_RESULT_FAILED")
                    events.append({"revision": item.revision, "event": "TERMINAL",
                                   "result_status": result.status,
                                   "error_code": result.result.error_code,
                                   **({"error_string": result.result.error_string}
                                      if hasattr(result.result, "error_string") else {})})
                    if slot == "_current" and (result.status != 4 or result.result.error_code != 0):
                        self._fenced = True
                    setattr(self, slot, None)
                    continue
            except Exception as exc:
                self._fenced, failure = True, failure if failure is not None else exc
            try:
                if self._fenced and item.handle is not None and not item.cancel_requested:
                    item.cancel_requested = True
                    item.cancel_response = item.handle.cancel_goal_async()
            except Exception as exc:
                self._fenced, failure = True, failure if failure is not None else exc
        if failure is not None:
            # Preserve facts already consumed in this poll alongside the original
            # error, just as the existing execution owner retains primary faults.
            failure.arm_stream_events = events
            raise failure
        return events
