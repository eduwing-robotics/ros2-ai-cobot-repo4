"""One logical native revision across the disjoint ARM and GRIPPER JTC clients.

This subordinate port owns native action handles only. It neither schedules policy
rows nor grants physical authority; the existing motion owner supplies the one
pair-wide dispatch guard and retains every returned native fact.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from tools.data_factory.motion.arm_stream import ArmStream
from tools.fr5_data_factory import ContractError


@dataclass
class _Pair:
    revision: str
    predecessor: str | None
    arm_accepted: bool = False
    gripper_accepted: bool = False
    arm_done: bool = False
    gripper_done: bool = False
    arm_succeeded: bool = False
    gripper_succeeded: bool = False
    rejected: bool = False
    acceptance_emitted: bool = False


class _AttemptClient:
    """Count entry into the real client call without interpreting its outcome."""

    def __init__(self, client, attempts, actuator):
        self._client, self._attempts, self._actuator = client, attempts, actuator

    def send_goal_async(self, *args, **kwargs):
        self._attempts[self._actuator] += 1
        return self._client.send_goal_async(*args, **kwargs)


class ActuatorStream:
    """One lifecycle owner for a logical revision sent to two native clients."""

    def __init__(self, arm_client, gripper_client, *, deadline, clock):
        if (
            not callable(clock)
            or isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise ContractError("ROS_EXEC_RESULT_TIMEOUT")
        self._clock, self.deadline = clock, deadline
        self._submission_attempts = {"arm": 0, "gripper": 0}
        self._arm = ArmStream(
            _AttemptClient(arm_client, self._submission_attempts, "arm"),
            deadline=deadline, clock=clock,
        )
        self._gripper = ArmStream(
            _AttemptClient(gripper_client, self._submission_attempts, "gripper"),
            deadline=deadline, clock=clock,
        )
        self._pairs: dict[str, _Pair] = {}
        self._seen_revisions: set[str] = set()
        self._current_revision = self._pending_revision = None
        self._fenced = False
        self._submitting = False

    @property
    def owns_goals(self):
        return self._arm.owns_goals or self._gripper.owns_goals

    @property
    def current_revision(self):
        return self._current_revision

    @property
    def submission_attempts(self):
        """Native send-call entries, not transmission or acceptance claims."""
        return dict(self._submission_attempts)

    @property
    def fenced(self):
        return self._fenced or self._arm.fenced or self._gripper.fenced

    def submit(self, arm_goal, gripper_goal, *, revision, dispatch_guard):
        """Guard and submit one exact logical pair without waiting for acceptance."""
        if self._submitting:
            raise ContractError("ROS_EXEC_ACTIVE")
        self._submitting = True
        try:
            self._submit(
                arm_goal, gripper_goal, revision=revision,
                dispatch_guard=dispatch_guard,
            )
        finally:
            self._submitting = False

    def _submit(self, arm_goal, gripper_goal, *, revision, dispatch_guard):
        if self.fenced or self._clock() >= self.deadline:
            raise ContractError("ROS_EXEC_CANCELLED")
        if self._pending_revision is not None:
            raise ContractError("ROS_EXEC_GOAL_PENDING")
        if (
            not isinstance(revision, str)
            or not revision
            or revision in self._seen_revisions
            or not callable(dispatch_guard)
        ):
            raise ContractError("ROS_EXEC_DISPATCH_GUARD")
        current_pair = self._pairs.get(self._current_revision)
        actual_predecessors = {
            "arm": self._arm.current_revision,
            "gripper": self._gripper.current_revision,
        }
        # Both native helpers must be ready before either client sees a goal. A
        # positively completed child has no native predecessor even while its
        # logical sibling still owns the current pair revision.
        def child_ready(actuator, stream):
            actual = actual_predecessors[actuator]
            completed = (
                current_pair is not None
                and getattr(current_pair, f"{actuator}_succeeded")
            )
            return (
                actual == self._current_revision
                or completed and actual is None
            ) and stream._pending is None and stream._retiring is None

        if (
            not child_ready("arm", self._arm)
            or not child_ready("gripper", self._gripper)
        ):
            raise ContractError("ROS_EXEC_GOAL_PENDING")

        retained = copy.deepcopy({"arm": arm_goal, "gripper": gripper_goal})
        checked = copy.deepcopy(retained)
        predecessor = self._current_revision
        dispatch_guard(checked, predecessor)
        if checked != retained:
            raise ContractError("LEARNED_SERIALIZED_ACTION_MISMATCH")
        if self._current_revision != predecessor:
            raise ContractError("ROS_EXEC_ACTIVE")
        if self.fenced or self._clock() >= self.deadline:
            raise ContractError("ROS_EXEC_CANCELLED")

        pair = _Pair(revision=revision, predecessor=predecessor)
        self._pairs[revision] = pair
        self._seen_revisions.add(revision)
        self._pending_revision = revision

        def child_guard(candidate, child_predecessor, *, actuator):
            if (
                child_predecessor != actual_predecessors[actuator]
                or candidate != retained[actuator]
            ):
                raise ContractError("LEARNED_SERIALIZED_ACTION_MISMATCH")

        def submit_child(actuator, stream, goal):
            stream.submit(
                goal, revision=revision,
                dispatch_guard=lambda candidate, prior: child_guard(
                    candidate, prior, actuator=actuator,
                ),
            )

        try:
            submit_child("arm", self._arm, retained["arm"])
            submit_child("gripper", self._gripper, retained["gripper"])
        except BaseException:
            self.cancel()
            raise

    def cancel(self):
        """Fence the logical stream and every known or ambiguous child goal."""
        self._fenced = True
        self._arm.cancel()
        self._gripper.cancel()

    def _record(self, actuator, event):
        pair = self._pairs.get(event.get("revision"))
        if pair is None:
            raise ContractError("ROS_EXEC_ACTIVE")
        kind = event.get("event")
        if kind == "ACCEPTED":
            setattr(pair, f"{actuator.lower()}_accepted", True)
        elif kind == "REJECTED":
            setattr(pair, f"{actuator.lower()}_done", True)
            pair.rejected = True
            self.cancel()
        elif kind == "TERMINAL":
            setattr(pair, f"{actuator.lower()}_done", True)
            if event.get("result_status") == 4 and event.get("error_code") == 0:
                setattr(pair, f"{actuator.lower()}_succeeded", True)

    def _poll_child(self, actuator, stream, events):
        try:
            native = stream.poll()
        except BaseException as exc:
            for event in getattr(exc, "arm_stream_events", []):
                self._record(actuator, event)
                events.append({"actuator": actuator, **event})
            return exc
        for event in native:
            self._record(actuator, event)
            events.append({"actuator": actuator, **event})
        return None

    def _logical_events(self):
        events = []
        for revision, pair in tuple(self._pairs.items()):
            if (
                pair.arm_accepted
                and pair.gripper_accepted
                and not pair.rejected
                and not pair.acceptance_emitted
            ):
                pair.acceptance_emitted = True
                self._pending_revision = None
                self._current_revision = revision
                events.append({
                    "revision": revision,
                    "event": "PAIR_ACCEPTED",
                    "predecessor": pair.predecessor,
                })
            if pair.arm_done and pair.gripper_done:
                if self._current_revision == revision:
                    self._current_revision = None
                if self._pending_revision == revision:
                    self._pending_revision = None
                del self._pairs[revision]
        return events

    def poll(self):
        """Return tagged native facts and pair acceptance without blocking."""
        events = []
        failures = []
        completed = []
        for actuator, stream in (("arm", self._arm), ("gripper", self._gripper)):
            failure = self._poll_child(actuator, stream, events)
            if failure is None:
                completed.append((actuator, stream))
            else:
                failures.append(failure)

        newly_fenced = self.fenced or bool(failures)
        if newly_fenced:
            self.cancel()
            # A sibling polled before the fault may only now know it must cancel.
            for actuator, stream in completed:
                failure = self._poll_child(actuator, stream, events)
                if failure is not None:
                    failures.append(failure)

        events.extend(self._logical_events())
        if failures:
            primary = failures[0]
            for secondary in failures[1:]:
                primary.add_note(f"ACTUATOR_STREAM_SECONDARY: {secondary!r}")
            primary.actuator_stream_events = events
            raise primary
        return events
