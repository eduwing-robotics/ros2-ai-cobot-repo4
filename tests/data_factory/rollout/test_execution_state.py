"""Pure selected-reference checks; no device, sampler or policy load."""
import copy
import unittest

from tools.data_factory.rollout.execution_state import (
    JOINTS, check_reference_anchor, reference_rows,
)
from tools.fr5_data_factory import ContractError


class ReferenceRowsTest(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "period_s": .1, "velocity_scaling": .1,
            "robot_description": '<robot name="test">' + ''.join(
                f'<joint name="{name}" type="{kind}"><limit lower="{low}" '
                f'upper="{high}" velocity="{speed}"/></joint>'
                for name, kind, low, high, speed in [
                    *[(name, "revolute", -3, 3, 10) for name in JOINTS[:6]],
                    (JOINTS[6], "prismatic", 0, .021, .001),
                ]) + '</robot>',
        }
        self.anchor = [0.] * 6 + [.01]

    def test_preserves_detached_rows_and_native_gripper_step(self):
        actions = [[.05] * 6 + [.021], [.1] * 6 + [.005]]
        original = copy.deepcopy(actions)
        rows = reference_rows(self.policy, self.anchor, actions)
        self.assertEqual(rows, [self.anchor, *original])
        rows[0][0] = 2.
        rows[1][6] = 0.
        self.assertEqual(self.anchor[0], 0.)
        self.assertEqual(actions, original)

    def test_arm_velocity_and_position_rejection_without_retiming(self):
        for actions, code in (
            ([[.1001] + [0.] * 5 + [.01]], "LEARNED_VELOCITY_LIMIT"),
            ([[0.] * 6 + [.022]], "LEARNED_JOINT_LIMIT"),
            ([], "LEARNED_ACTION_7D"),
            ([[float("nan")] + [0.] * 5 + [.01]], "LEARNED_ACTION_7D"),
        ):
            with self.subTest(code=code):
                with self.assertRaisesRegex(ContractError, code):
                    reference_rows(self.policy, self.anchor, actions)

    def test_invalid_period_or_scaling_cannot_create_reference_rows(self):
        for key, value in (("period_s", 0), ("period_s", float("inf")),
                           ("velocity_scaling", 0), ("velocity_scaling", 1.1)):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ContractError):
                    reference_rows({**self.policy, key: value}, self.anchor, [self.anchor])


class ReferenceAnchorTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "planning": {"goal_tolerances": {"joint_rad": .01}},
            "steps": [{"phase": "GRIPPER_CLOSE", "limits": {
                "completion_tolerance_m": .0002}}],
        }
        self.anchor = [0.] * 6 + [.01]
        self.prepared = {"gripper_controller": {"reference_position_m": .012}}

    def test_unchanged_reference_feedback_gap_is_not_a_completion_barrier(self):
        check_reference_anchor(self.source, self.anchor, self.anchor,
                               copy.deepcopy(self.prepared), prepared_observed=self.prepared)

    def test_actual_anchor_or_reference_movement_invalidates_only_candidate(self):
        for channel in ("arm", "feedback", "reference"):
            with self.subTest(channel=channel):
                current, observed = self.anchor[:], copy.deepcopy(self.prepared)
                if channel == "arm":
                    current[0] = .02
                elif channel == "feedback":
                    current[6] += .001
                else:
                    observed["gripper_controller"]["reference_position_m"] += .001
                with self.assertRaisesRegex(ContractError, "START_STATE_MISMATCH"):
                    check_reference_anchor(self.source, self.anchor, current, observed,
                                           prepared_observed=self.prepared)
        self.assertEqual(self.anchor, [0.] * 6 + [.01])
        self.assertEqual(self.prepared["gripper_controller"]["reference_position_m"], .012)
