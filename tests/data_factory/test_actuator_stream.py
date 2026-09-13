"""Composite ARM/GRIPPER native action seam; no ROS node or hardware calls."""
from concurrent.futures import Future
from itertools import count
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from tools.data_factory.motion.actuator_stream import ActuatorStream
from tools.fr5_data_factory import ContractError


def goal(joints, offset=0.):
    value = FollowJointTrajectory.Goal()
    value.trajectory.joint_names = list(joints)
    point = JointTrajectoryPoint(positions=[offset] * len(joints))
    point.time_from_start.sec = 1
    value.trajectory.points.append(point)
    return value


_identifiers = count(1)


def handle(*, accepted=True):
    return SimpleNamespace(
        accepted=accepted,
        goal_id=SimpleNamespace(uuid=[0] * 15 + [next(_identifiers)]),
        get_result_async=Mock(return_value=Future()),
        cancel_goal_async=Mock(return_value=Future()),
    )


def finish(item, status=4, code=0):
    item.get_result_async.return_value.set_result(SimpleNamespace(
        status=status, result=SimpleNamespace(error_code=code),
    ))


class ActuatorStreamTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.arm_client, self.gripper_client = Mock(), Mock()
        self.arm_response, self.gripper_response = Future(), Future()
        self.arm_client.send_goal_async.return_value = self.arm_response
        self.gripper_client.send_goal_async.return_value = self.gripper_response
        self.stream = ActuatorStream(
            self.arm_client, self.gripper_client,
            deadline=20., clock=lambda: self.now,
        )
        self.arm_goal = goal([f"j{i}" for i in range(1, 7)])
        self.gripper_goal = goal(["finger_right_joint"])

    def submit(self, revision="first", guard=None):
        guard = guard or Mock()
        self.stream.submit(
            self.arm_goal, self.gripper_goal,
            revision=revision, dispatch_guard=guard,
        )
        return guard

    def accept(self):
        arm, gripper = handle(), handle()
        self.arm_response.set_result(arm)
        self.gripper_response.set_result(gripper)
        return arm, gripper

    def test_pair_guard_runs_once_before_both_exact_deepcopied_sends(self):
        observed = []
        def guard(pair, predecessor):
            observed.append((pair, predecessor))
        self.submit(guard=guard)
        self.assertEqual(len(observed), 1)
        self.assertIsNone(observed[0][1])
        self.assertEqual(observed[0][0], {
            "arm": self.arm_goal, "gripper": self.gripper_goal,
        })
        self.arm_goal.trajectory.points[0].positions[0] = 9.
        self.gripper_goal.trajectory.points[0].positions[0] = 9.
        self.assertEqual(
            self.arm_client.send_goal_async.call_args.args[0].trajectory.points[0].positions[0],
            0.,
        )
        self.assertEqual(
            self.gripper_client.send_goal_async.call_args.args[0].trajectory.points[0].positions[0],
            0.,
        )
        self.assertEqual(self.stream.submission_attempts, {"arm": 1, "gripper": 1})

    def test_guard_rejection_or_pair_mutation_sends_neither_actuator(self):
        for guard in (
            Mock(side_effect=ContractError("SCENE_CHANGED")),
            lambda pair, _prior: pair["gripper"].trajectory.points.clear(),
        ):
            with self.subTest(guard=guard), self.assertRaises(ContractError):
                self.stream.submit(
                    self.arm_goal, self.gripper_goal,
                    revision="first", dispatch_guard=guard,
                )
        self.arm_client.send_goal_async.assert_not_called()
        self.gripper_client.send_goal_async.assert_not_called()
        self.assertFalse(self.stream.owns_goals)
        self.assertEqual(self.stream.submission_attempts, {"arm": 0, "gripper": 0})

    def test_logical_acceptance_requires_both_native_acceptances(self):
        self.submit()
        arm, gripper = handle(), handle()
        self.arm_response.set_result(arm)
        self.assertEqual(self.stream.poll(), [
            {"actuator": "arm", "revision": "first", "event": "ACCEPTED"},
        ])
        self.assertIsNone(self.stream.current_revision)
        self.gripper_response.set_result(gripper)
        self.assertEqual(self.stream.poll(), [
            {"actuator": "gripper", "revision": "first", "event": "ACCEPTED"},
            {"revision": "first", "event": "PAIR_ACCEPTED", "predecessor": None},
        ])
        self.assertEqual(self.stream.current_revision, "first")
        self.assertTrue(self.stream.owns_goals)

    def test_one_rejection_fences_pair_and_cancels_accepted_sibling(self):
        self.submit()
        arm = handle()
        self.arm_response.set_result(arm)
        self.gripper_response.set_result(handle(accepted=False))
        events = self.stream.poll()
        self.assertEqual(
            [(item["actuator"], item["event"]) for item in events],
            [("arm", "ACCEPTED"), ("gripper", "REJECTED")],
        )
        self.assertNotIn("PAIR_ACCEPTED", [item["event"] for item in events])
        self.assertTrue(self.stream.fenced)
        self.assertIsNone(self.stream.current_revision)
        arm.cancel_goal_async.assert_called_once()

    def test_second_send_error_retains_ambiguous_pair_and_cancels_known_arm(self):
        error = OSError("gripper send outcome unknown")
        self.gripper_client.send_goal_async.side_effect = error
        with self.assertRaises(OSError) as raised:
            self.submit()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.stream.submission_attempts, {"arm": 1, "gripper": 1})
        self.assertTrue(self.stream.fenced)
        self.assertTrue(self.stream.owns_goals)
        arm = handle()
        self.arm_response.set_result(arm)
        self.assertEqual(self.stream.poll(), [
            {"actuator": "arm", "revision": "first", "event": "ACCEPTED"},
        ])
        arm.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.owns_goals)

    def test_first_ambiguous_send_counts_only_the_entered_arm_boundary(self):
        error = OSError("arm send outcome unknown")
        self.arm_client.send_goal_async.side_effect = error
        with self.assertRaises(OSError) as raised:
            self.submit()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.stream.submission_attempts, {"arm": 1, "gripper": 0})
        self.gripper_client.send_goal_async.assert_not_called()
        self.assertTrue(self.stream.owns_goals)
        self.assertTrue(self.stream.fenced)
        self.assertEqual(self.stream.poll(), [])
        self.assertTrue(self.stream.owns_goals)

    def test_cancel_during_first_native_send_prevents_gripper_and_owns_late_arm(self):
        def cancel_during_send(*_args, **_kwargs):
            self.stream.cancel()
            return self.arm_response
        self.arm_client.send_goal_async.side_effect = cancel_during_send
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCELLED"):
            self.submit()
        self.assertEqual(self.stream.submission_attempts, {"arm": 1, "gripper": 0})
        self.gripper_client.send_goal_async.assert_not_called()
        self.assertTrue(self.stream.fenced)
        self.assertTrue(self.stream.owns_goals)

        arm = handle()
        self.arm_response.set_result(arm)
        self.assertEqual(self.stream.poll(), [
            {"actuator": "arm", "revision": "first", "event": "ACCEPTED"},
        ])
        arm.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.owns_goals)
        self.assertFalse(arm.get_result_async.return_value.done())

    def test_cancel_fences_both_handles_without_inventing_completion(self):
        self.submit()
        arm, gripper = self.accept()
        self.stream.poll()
        self.stream.cancel()
        self.assertEqual(self.stream.poll(), [])
        arm.cancel_goal_async.assert_called_once()
        gripper.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.owns_goals)
        finish(arm, status=5)
        finish(gripper, status=5)
        events = self.stream.poll()
        self.assertEqual(
            [(item["actuator"], item["event"]) for item in events],
            [("arm", "TERMINAL"), ("gripper", "TERMINAL")],
        )
        self.assertFalse(self.stream.owns_goals)
        self.assertIsNone(self.stream.current_revision)

    def test_replacement_has_one_logical_predecessor_and_waits_for_retirements(self):
        self.submit()
        first_arm, first_gripper = self.accept()
        self.stream.poll()
        second_arm_response, second_gripper_response = Future(), Future()
        self.arm_client.send_goal_async.return_value = second_arm_response
        self.gripper_client.send_goal_async.return_value = second_gripper_response
        guard = self.submit(revision="second")
        guard.assert_called_once()
        self.assertEqual(guard.call_args.args[1], "first")
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_GOAL_PENDING"):
            self.stream.submit(
                self.arm_goal, self.gripper_goal,
                revision="third", dispatch_guard=Mock(),
            )
        self.assertEqual(self.arm_client.send_goal_async.call_count, 2)
        second_arm, second_gripper = handle(), handle()
        second_arm_response.set_result(second_arm)
        second_gripper_response.set_result(second_gripper)
        self.stream.poll()
        self.assertEqual(self.stream.current_revision, "second")
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_GOAL_PENDING"):
            self.stream.submit(
                self.arm_goal, self.gripper_goal,
                revision="third", dispatch_guard=Mock(),
            )
        self.assertEqual(self.arm_client.send_goal_async.call_count, 2)
        finish(first_arm, status=6, code=-1)
        finish(first_gripper, status=6, code=-1)
        self.stream.poll()
        self.assertEqual(self.stream.current_revision, "second")

    def test_replacement_does_not_wait_for_both_positive_child_terminals(self):
        for completed_actuator in ("arm", "gripper"):
            with self.subTest(completed_actuator=completed_actuator):
                self.setUp()
                self.submit()
                first_arm, first_gripper = self.accept()
                self.stream.poll()
                completed = first_arm if completed_actuator == "arm" else first_gripper
                finish(completed)
                self.assertEqual(self.stream.poll()[0]["actuator"], completed_actuator)
                self.assertEqual(self.stream.current_revision, "first")

                arm_response, gripper_response = Future(), Future()
                self.arm_client.send_goal_async.return_value = arm_response
                self.gripper_client.send_goal_async.return_value = gripper_response
                guard = self.submit(revision="second")
                guard.assert_called_once()
                self.assertEqual(guard.call_args.args[1], "first")
                self.assertEqual(
                    self.stream.submission_attempts, {"arm": 2, "gripper": 2},
                )

                second_arm, second_gripper = handle(), handle()
                arm_response.set_result(second_arm)
                gripper_response.set_result(second_gripper)
                events = self.stream.poll()
                self.assertEqual(
                    [event["event"] for event in events].count("PAIR_ACCEPTED"), 1,
                )
                self.assertEqual(self.stream.current_revision, "second")

    def test_deadline_fences_and_cancels_both_native_handles(self):
        self.submit()
        arm, gripper = self.accept()
        self.stream.poll()
        self.now = 20.
        self.assertEqual(self.stream.poll(), [])
        self.assertTrue(self.stream.fenced)
        arm.cancel_goal_async.assert_called_once()
        gripper.cancel_goal_async.assert_called_once()
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCELLED"):
            self.stream.submit(
                self.arm_goal, self.gripper_goal,
                revision="second", dispatch_guard=Mock(),
            )

    def test_native_result_error_is_preserved_and_cancels_its_sibling(self):
        self.submit()
        arm, gripper = self.accept()
        self.stream.poll()
        error = OSError("arm result channel failed")
        arm.get_result_async.return_value.set_exception(error)
        with self.assertRaises(OSError) as raised:
            self.stream.poll()
        self.assertIs(raised.exception, error)
        self.assertEqual(error.actuator_stream_events, [])
        self.assertTrue(self.stream.fenced)
        arm.cancel_goal_async.assert_called_once()
        gripper.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.owns_goals)


if __name__ == "__main__":
    unittest.main()
