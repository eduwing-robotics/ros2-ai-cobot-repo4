"""Bounded reference-progress reduction for one native actuator revision.

This tracks the prefix whose ARM and gripper reference timelines have both
crossed their policy-row endpoints.  It is neither physical adoption evidence
nor a scheduler, completion gate, or execution authority.
"""

from bisect import bisect_right
from collections.abc import Mapping
import math

from tools.fr5_data_factory import ContractError


_ACTUATORS = ("arm", "gripper")
_JOINT_NAMES = {
    "arm": ("j1", "j2", "j3", "j4", "j5", "j6"),
    "gripper": ("finger_right_joint",),
}
_NANOSECONDS = 1_000_000_000


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


class ReferenceProgress:
    """Reduce latest native facts to a cumulative full-7D reference prefix."""

    __slots__ = (
        "revision", "_action_times_ns", "_accepted", "_elapsed_ns",
        "_failed", "_goal_ids", "_selected_count",
    )

    def __init__(self, revision, point_times_ns):
        if not isinstance(revision, str) or not revision:
            raise ContractError("LEARNED_REFERENCE_PROGRESS")
        if not isinstance(point_times_ns, (list, tuple)) or len(point_times_ns) < 2:
            raise ContractError("LEARNED_REFERENCE_PROGRESS")
        times = tuple(point_times_ns)
        if (
            times[0] != 0
            or any(type(value) is not int or not 0 <= value < 2**31 * _NANOSECONDS
                   for value in times)
            or any(current <= previous for previous, current in zip(times, times[1:]))
        ):
            raise ContractError("LEARNED_REFERENCE_PROGRESS")
        self.revision = revision
        self._action_times_ns = times[1:]
        self._accepted = False
        self._elapsed_ns = {actuator: None for actuator in _ACTUATORS}
        self._failed = set()
        self._goal_ids = {actuator: None for actuator in _ACTUATORS}
        self._selected_count = 0

    @property
    def selected_count(self):
        """Number of policy-row endpoints crossed by both native references."""

        return self._selected_count

    def observe(self, event):
        """Consume one ActuatorStream event and return cumulative progress."""

        if not isinstance(event, Mapping) or event.get("revision") != self.revision:
            return self._selected_count
        kind = event.get("event")
        if kind == "PAIR_ACCEPTED":
            predecessor = event.get("predecessor")
            if "predecessor" in event and (
                predecessor is None or isinstance(predecessor, str) and predecessor
            ):
                self._accepted = True
        else:
            actuator = event.get("actuator")
            if actuator not in _ACTUATORS:
                return self._selected_count
            if kind == "FEEDBACK":
                self._observe_feedback(actuator, event)
            elif kind == "REJECTED":
                self._failed.add(actuator)
            elif kind == "TERMINAL":
                self._observe_terminal(actuator, event)
        self._update()
        return self._selected_count

    def _observe_feedback(self, actuator, event):
        goal_id = event.get("goal_id")
        received = event.get("received_monotonic_s")
        samples = event.get("samples_since_poll")
        feedback = event.get("feedback")
        names = feedback.get("joint_names") if isinstance(feedback, Mapping) else None
        desired = feedback.get("desired") if isinstance(feedback, Mapping) else None
        positions = desired.get("positions") if isinstance(desired, Mapping) else None
        if (
            not isinstance(goal_id, (list, tuple))
            or len(goal_id) != 16
            or any(type(value) is not int or not 0 <= value <= 255 for value in goal_id)
            or not _finite_number(received)
            or type(samples) is not int
            or samples < 1
            or not isinstance(feedback, Mapping)
            or not isinstance(names, (list, tuple))
            or tuple(names) != _JOINT_NAMES[actuator]
            or not isinstance(desired, Mapping)
            or not isinstance(positions, (list, tuple))
            or len(positions) != len(names)
            or any(not _finite_number(value) for value in positions)
        ):
            return
        stamp = desired.get("time_from_start")
        if not isinstance(stamp, Mapping):
            return
        sec, nanosec = stamp.get("sec"), stamp.get("nanosec")
        if (
            type(sec) is not int
            or type(nanosec) is not int
            or not -(2**31) <= sec < 2**31
            or not 0 <= nanosec < _NANOSECONDS
        ):
            return
        identifier = tuple(goal_id)
        if self._goal_ids[actuator] not in (None, identifier):
            return
        elapsed = sec * _NANOSECONDS + nanosec
        previous = self._elapsed_ns[actuator]
        if previous is not None and elapsed < previous:
            return
        self._goal_ids[actuator] = identifier
        self._elapsed_ns[actuator] = elapsed

    def _observe_terminal(self, actuator, event):
        status, code = event.get("result_status"), event.get("error_code")
        if type(status) is not int or type(code) is not int or status != 4 or code != 0:
            self._failed.add(actuator)
            return
        if actuator not in self._failed:
            self._elapsed_ns[actuator] = self._action_times_ns[-1]

    def _update(self):
        if not self._accepted or self._failed or any(
            self._elapsed_ns[actuator] is None for actuator in _ACTUATORS
        ):
            return
        crossed = bisect_right(
            self._action_times_ns,
            min(self._elapsed_ns.values()),
        )
        self._selected_count = max(self._selected_count, crossed)


__all__ = ["ReferenceProgress"]
