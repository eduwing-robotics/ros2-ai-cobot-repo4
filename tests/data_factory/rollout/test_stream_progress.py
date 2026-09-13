"""Pure native-event reduction; no ROS node, model, camera, or device."""

import unittest

from tools.data_factory.rollout.stream_progress import ReferenceProgress
from tools.fr5_data_factory import ContractError


PERIOD = 50_000_000
TIMES = [0, PERIOD, 2 * PERIOD, 3 * PERIOD, 4 * PERIOD]


def accepted(revision="r1"):
    return {"revision": revision, "event": "PAIR_ACCEPTED", "predecessor": None}


def feedback(actuator, elapsed_ns, *, revision="r1", goal=1, samples=1):
    sec, nanosec = divmod(elapsed_ns, 1_000_000_000)
    names = [f"j{i}" for i in range(1, 7)] if actuator == "arm" else ["finger_right_joint"]
    return {
        "actuator": actuator,
        "revision": revision,
        "event": "FEEDBACK",
        "goal_id": [0] * 15 + [goal],
        "received_monotonic_s": 10.0,
        "samples_since_poll": samples,
        "feedback": {
            "header": {"stamp": {"sec": 42, "nanosec": 0}, "frame_id": ""},
            "joint_names": names,
            "desired": {
                "positions": [0.0] * len(names),
                "velocities": [], "accelerations": [], "effort": [],
                "time_from_start": {"sec": sec, "nanosec": nanosec},
            },
            "actual": {"positions": [0.0] * len(names)},
            "error": {"positions": [0.0] * len(names)},
        },
    }


def terminal(actuator, *, revision="r1", status=4, code=0):
    return {
        "actuator": actuator, "revision": revision, "event": "TERMINAL",
        "result_status": status, "error_code": code,
    }


class ReferenceProgressTest(unittest.TestCase):
    def test_requires_native_anchor_and_strict_integer_point_times(self):
        for revision, times in (("", TIMES), ("r1", [PERIOD]), ("r1", [1, PERIOD]),
                                ("r1", [0, PERIOD, PERIOD]), ("r1", [0, True])):
            with self.subTest(revision=revision, times=times), self.assertRaisesRegex(
                ContractError, "LEARNED_REFERENCE_PROGRESS"
            ):
                ReferenceProgress(revision, times)

    def test_actual_event_order_and_endpoint_crossing_do_not_count_interval_selection(self):
        progress = ReferenceProgress("r1", TIMES)
        # ActuatorStream can publish child feedback before its trailing logical
        # PAIR_ACCEPTED event. Future-header and zero elapsed select no full 7D row.
        progress.observe(feedback("arm", -PERIOD))
        progress.observe(feedback("gripper", -PERIOD))
        self.assertEqual(progress.observe({"revision": "r1", "event": "PAIR_ACCEPTED"}), 0)
        self.assertEqual(progress.observe(accepted()), 0)
        self.assertEqual(progress.observe(feedback("arm", 0)), 0)
        self.assertEqual(progress.observe(feedback("gripper", 0)), 0)
        # Gripper NONE already points at row 0, but ARM has not crossed row 0's
        # endpoint; this is not yet a consumed full-7D reference row.
        self.assertEqual(progress.observe(feedback("gripper", PERIOD)), 0)
        self.assertEqual(progress.observe(feedback("arm", PERIOD - 1)), 0)
        self.assertEqual(progress.observe(feedback("arm", PERIOD)), 1)

    def test_delayed_coalesced_and_asymmetric_feedback_catches_up_cumulatively(self):
        progress = ReferenceProgress("r1", TIMES)
        progress.observe(accepted())
        progress.observe(feedback("arm", 4 * PERIOD, samples=17))
        self.assertEqual(progress.observe(feedback("gripper", 2 * PERIOD, samples=9)), 2)
        self.assertEqual(progress.observe(feedback("gripper", 4 * PERIOD, samples=8)), 4)

    def test_speed_pause_on_either_channel_holds_only_the_progress_watermark(self):
        progress = ReferenceProgress("r1", TIMES)
        progress.observe(accepted())
        progress.observe(feedback("arm", 2 * PERIOD))
        self.assertEqual(progress.observe(feedback("gripper", 2 * PERIOD)), 2)
        self.assertEqual(progress.observe(feedback("gripper", 3 * PERIOD)), 2)
        self.assertEqual(progress.observe(feedback("arm", 2 * PERIOD)), 2)
        self.assertEqual(progress.observe(feedback("arm", 3 * PERIOD)), 3)

    def test_unrelated_malformed_changed_goal_and_regressing_feedback_cannot_advance(self):
        progress = ReferenceProgress("r1", TIMES)
        progress.observe(accepted())
        progress.observe(feedback("arm", 2 * PERIOD))
        self.assertEqual(progress.observe(feedback("gripper", 2 * PERIOD)), 2)
        self.assertEqual(progress.observe(feedback("arm", 4 * PERIOD, revision="other")), 2)
        malformed = feedback("arm", 4 * PERIOD)
        malformed["feedback"]["desired"]["time_from_start"]["nanosec"] = 1_000_000_000
        self.assertEqual(progress.observe(malformed), 2)
        malformed = feedback("arm", 4 * PERIOD)
        malformed["received_monotonic_s"] = 10**400
        self.assertEqual(progress.observe(malformed), 2)
        for names in (None, ["finger_right_joint"], ["j2", "j1", "j3", "j4", "j5", "j6"]):
            malformed = feedback("arm", 4 * PERIOD)
            if names is None:
                malformed["feedback"].pop("joint_names")
            else:
                malformed["feedback"]["joint_names"] = names
            self.assertEqual(progress.observe(malformed), 2)
        self.assertEqual(progress.observe(feedback("arm", 4 * PERIOD, goal=2)), 2)
        self.assertEqual(progress.observe(feedback("arm", PERIOD)), 2)
        self.assertEqual(progress.observe(feedback("gripper", 4 * PERIOD)), 2)

    def test_only_positive_terminals_reconcile_their_actuator(self):
        progress = ReferenceProgress("r1", TIMES)
        progress.observe(terminal("arm"))
        progress.observe(feedback("gripper", 2 * PERIOD))
        self.assertEqual(progress.observe(accepted()), 2)
        self.assertEqual(progress.observe(terminal("gripper", status=5)), 2)
        self.assertEqual(progress.observe(terminal("gripper")), 2)

        complete = ReferenceProgress("r1", TIMES)
        complete.observe(terminal("arm"))
        complete.observe(terminal("gripper"))
        self.assertEqual(complete.observe(accepted()), 4)

        rejected = ReferenceProgress("r1", TIMES)
        rejected.observe({"actuator": "arm", "revision": "r1", "event": "REJECTED"})
        rejected.observe(terminal("gripper"))
        rejected.observe(terminal("arm"))
        self.assertEqual(rejected.observe(accepted()), 0)

    def test_retains_only_latest_scalar_facts_under_many_events(self):
        progress = ReferenceProgress("r1", TIMES)
        progress.observe(accepted())
        for elapsed in range(1_000):
            progress.observe(feedback("arm", elapsed))
            progress.observe(feedback("gripper", elapsed))
        self.assertFalse(hasattr(progress, "__dict__"))
        self.assertEqual(len(progress._elapsed_ns), 2)
        self.assertEqual(len(progress._goal_ids), 2)
        self.assertEqual(progress.selected_count, 0)


if __name__ == "__main__":
    unittest.main()
