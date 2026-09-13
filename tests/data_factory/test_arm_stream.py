"""Native FJT message/future seam; no ROS node, model or hardware calls."""
from concurrent.futures import Future
import copy
import gc
from itertools import count
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from unique_identifier_msgs.msg import UUID

from tools.data_factory.motion.arm_stream import ArmStream
from tools.data_factory.motion.moveit_transport import RosMoveItTransport
from tools.fr5_data_factory import ContractError


def done(value):
    future = Future()
    future.set_result(value)
    return future


def goal(offset=0.):
    value = FollowJointTrajectory.Goal()
    value.trajectory.joint_names = [f"j{i}" for i in range(1, 7)]
    for i in range(50):
        point = JointTrajectoryPoint(positions=[offset + .001 * i] * 6)
        point.time_from_start.sec = i // 30
        point.time_from_start.nanosec = round((i % 30) / 30 * 1e9)
        value.trajectory.points.append(point)
    return value


_identifiers = count(1)


def handle():
    return SimpleNamespace(accepted=True, get_result_async=Mock(return_value=Future()),
                           goal_id=SimpleNamespace(uuid=[0] * 15 + [next(_identifiers)]),
                           cancel_goal_async=Mock(return_value=Future()))


def finish(item, status=4, code=0):
    item.get_result_async.return_value.set_result(SimpleNamespace(
        status=status, result=SimpleNamespace(error_code=code)))


def feedback(item, nanosec=100_000_000):
    message = FollowJointTrajectory.Impl.FeedbackMessage()
    message.goal_id = UUID(uuid=list(item.goal_id.uuid))
    message.feedback.joint_names = [f"j{i}" for i in range(1, 7)]
    message.feedback.header.stamp.sec = 42
    message.feedback.desired = JointTrajectoryPoint(positions=[.1] * 6)
    message.feedback.desired.time_from_start.nanosec = nanosec
    message.feedback.actual = JointTrajectoryPoint(positions=[.09] * 6)
    return message


class ArmStreamTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.client = Mock()
        self.response = Future()
        self.client.send_goal_async.return_value = self.response
        self.stream = ArmStream(self.client, deadline=20., clock=lambda: self.now)
        self.guard = Mock()

    def initial(self):
        self.stream.submit(goal(), revision="first", dispatch_guard=self.guard)
        first = handle()
        self.response.set_result(first)
        self.assertEqual(self.stream.poll(), [{"revision": "first", "event": "ACCEPTED"}])
        return first

    def replacement(self):
        response = Future()
        self.client.send_goal_async.return_value = response
        self.stream.submit(goal(.1), revision="second", dispatch_guard=self.guard)
        return response

    def test_full_trajectory_once_no_waypoint_or_result_barrier(self):
        value = goal()
        self.stream.submit(value, revision="first", dispatch_guard=self.guard)
        sent = self.client.send_goal_async.call_args.args[0]
        self.assertEqual(sent, value)
        self.assertEqual(len(sent.trajectory.points), 50)
        value.trajectory.points[0].positions[0] = 2.
        self.assertEqual(sent.trajectory.points[0].positions[0], 0.)
        self.assertEqual(self.stream.poll(), [])
        self.assertTrue(self.stream.owns_goals)
        self.assertIsNone(self.stream.current_revision)
        self.client.send_goal_async.assert_called_once()

    def test_goal_feedback_is_bound_to_native_identity_and_preserves_source_time(self):
        first = self.initial()
        callback = self.client.send_goal_async.call_args.kwargs["feedback_callback"]
        sample = feedback(first)
        callback(sample)
        sample.feedback.desired.positions[0] = 9.
        self.now = 11.
        events = self.stream.poll()
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event["revision"], event["event"]), ("first", "FEEDBACK"))
        self.assertEqual(event["goal_id"], list(first.goal_id.uuid))
        self.assertEqual(event["received_monotonic_s"], 10.)
        self.assertEqual(event["feedback"]["header"]["stamp"]["sec"], 42)
        self.assertEqual(event["feedback"]["desired"]["positions"][0], .1)
        self.assertEqual(event["feedback"]["actual"]["positions"][0], .09)
        self.assertEqual(event["feedback"]["desired"]["time_from_start"]["nanosec"], 100_000_000)
        self.assertEqual(self.stream.poll(), [])
        self.assertFalse(first.get_result_async.return_value.done())
        self.client.send_goal_async.assert_called_once()

    def test_feedback_before_acceptance_is_retained_but_not_published_unbound(self):
        self.stream.submit(goal(), revision="first", dispatch_guard=self.guard)
        callback = self.client.send_goal_async.call_args.kwargs["feedback_callback"]
        first = handle()
        callback(feedback(first))
        self.assertEqual(self.stream.poll(), [])
        self.response.set_result(first)
        self.assertEqual([e["event"] for e in self.stream.poll()], ["ACCEPTED", "FEEDBACK"])
        callback(feedback(handle()))
        self.assertEqual(self.stream.poll(), [])
        self.assertFalse(self.stream.fenced)

    def test_feedback_is_latest_only_and_cannot_alias_replacement_or_revive_terminal(self):
        first = self.initial()
        old_callback = self.client.send_goal_async.call_args.kwargs["feedback_callback"]
        response = self.replacement()
        new_callback = self.client.send_goal_async.call_args.kwargs["feedback_callback"]
        second = handle()
        response.set_result(second)
        self.stream.poll()
        for n in range(1, 1001):
            old_callback(feedback(first, n))
        new_callback(feedback(second, 7))
        finish(first, 6, -1)
        events = self.stream.poll()
        self.assertEqual([(e["revision"], e["event"]) for e in events],
                         [("first", "FEEDBACK"), ("first", "TERMINAL"), ("second", "FEEDBACK")])
        self.assertEqual(events[0]["samples_since_poll"], 1000)
        self.assertEqual(events[0]["feedback"]["desired"]["time_from_start"]["nanosec"], 1000)
        self.assertEqual(events[2]["samples_since_poll"], 1)
        old_callback(feedback(first, 2000))
        self.assertEqual(self.stream.poll(), [])
        self.assertEqual(self.stream.current_revision, "second")

    def test_feedback_does_not_dispatch_or_hide_cancellation_and_native_terminal(self):
        first = self.initial()
        callback = self.client.send_goal_async.call_args.kwargs["feedback_callback"]
        self.stream.cancel()
        callback(feedback(first))
        self.assertEqual(self.stream.poll()[0]["event"], "FEEDBACK")
        first.cancel_goal_async.assert_called_once()
        callback(feedback(first, 200_000_000))
        finish(first, 5)
        self.assertEqual([e["event"] for e in self.stream.poll()], ["FEEDBACK", "TERMINAL"])
        self.assertFalse(self.stream.owns_goals)
        self.client.send_goal_async.assert_called_once()

    def test_feedback_projection_error_does_not_cancel_motion_or_hide_native_result(self):
        first = self.initial()
        callback = self.client.send_goal_async.call_args.kwargs["feedback_callback"]
        response = self.replacement()
        second = handle()
        response.set_result(second)
        self.stream.poll()
        callback(feedback(first))
        finish(first, 6, -1)
        error = ValueError("projection failed")
        with patch("rosidl_runtime_py.convert.message_to_ordereddict", side_effect=error):
            events = self.stream.poll()
        self.assertEqual(events, [
            {"revision": "first", "event": "FEEDBACK_UNAVAILABLE", "error_type": "ValueError"},
            {"revision": "first", "event": "TERMINAL", "result_status": 6, "error_code": -1}])
        self.assertFalse(self.stream.fenced)
        second.cancel_goal_async.assert_not_called()

    def test_native_orphan_callback_does_not_retain_rejected_attempt_payloads(self):
        callbacks, submissions = [], []
        for i in range(3):
            response = Future()
            self.client.send_goal_async.return_value = response
            self.stream.submit(goal(), revision=str(i), dispatch_guard=self.guard)
            callbacks.append(self.client.send_goal_async.call_args.kwargs["feedback_callback"])
            submissions.append(weakref.ref(self.stream._pending))
            response.set_result(SimpleNamespace(accepted=False))
            self.assertEqual(self.stream.poll()[0]["event"], "REJECTED")
        owner = weakref.ref(self.stream)
        self.stream = None
        gc.collect()
        self.assertIsNone(owner())
        self.assertTrue(all(ref() is None for ref in submissions))
        for callback in callbacks:
            callback(feedback(handle()))  # A late native callback is harmless.

    def test_replacement_does_not_invent_predecessor_terminal(self):
        first = self.initial()
        response = self.replacement()
        self.assertEqual(self.stream.current_revision, "first")
        self.assertEqual(self.stream.poll(), [])
        second = handle()
        response.set_result(second)
        self.assertEqual(self.stream.poll(), [{"revision": "second", "event": "ACCEPTED"}])
        self.assertEqual(self.stream.current_revision, "second")
        self.assertFalse(first.get_result_async.return_value.done())
        first.cancel_goal_async.assert_not_called()
        finish(first, 6, -1)  # Preserve native ABORTED; do not fabricate CANCELED.
        self.assertEqual(self.stream.poll(), [{"revision": "first", "event": "TERMINAL",
                                              "result_status": 6, "error_code": -1}])
        self.assertTrue(self.stream.owns_goals)
        finish(second)
        self.assertEqual(self.stream.poll()[0]["revision"], "second")
        self.assertFalse(self.stream.owns_goals)

    def test_rejected_candidate_keeps_predecessor_owned_and_running(self):
        first = self.initial()
        response = self.replacement()
        response.set_result(SimpleNamespace(accepted=False))
        self.assertEqual(self.stream.poll(), [{"revision": "second", "event": "REJECTED"}])
        self.assertEqual(self.stream.current_revision, "first")
        first.cancel_goal_async.assert_not_called()

    def test_guard_rejection_or_mutation_sends_nothing_and_keeps_current(self):
        first = self.initial()
        for guard in (Mock(side_effect=ContractError("SCENE_CHANGED")),
                      lambda g, p: g.trajectory.points.clear()):
            with self.subTest(guard=guard), self.assertRaises(ContractError):
                self.stream.submit(goal(.1), revision="second", dispatch_guard=guard)
        self.client.send_goal_async.assert_called_once()
        self.assertEqual(self.stream.current_revision, "first")
        first.cancel_goal_async.assert_not_called()

    def test_reentrant_guard_cannot_overwrite_an_untracked_native_goal(self):
        def guard(g, predecessor):
            with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
                self.stream.submit(goal(.1), revision="nested", dispatch_guard=self.guard)
        self.stream.submit(goal(), revision="outer", dispatch_guard=guard)
        self.client.send_goal_async.assert_called_once()
        item = handle()
        self.response.set_result(item)
        self.assertEqual(self.stream.poll(), [{"revision": "outer", "event": "ACCEPTED"}])
        self.stream.cancel()
        self.stream.poll()
        item.cancel_goal_async.assert_called_once()

    def test_guard_failure_releases_only_the_unsent_submission_reservation(self):
        def guard(g, predecessor):
            self.stream.submit(goal(), revision="nested", dispatch_guard=self.guard)
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
            self.stream.submit(goal(), revision="outer", dispatch_guard=guard)
        self.client.send_goal_async.assert_not_called()
        self.assertFalse(self.stream.owns_goals)
        self.stream.submit(goal(), revision="retry_preparation", dispatch_guard=self.guard)
        self.client.send_goal_async.assert_called_once()

    def test_cancel_pending_acceptance_fences_updates_and_waits_for_native_result(self):
        first = self.initial()
        response = self.replacement()
        self.stream.cancel()
        self.assertEqual(self.stream.poll(), [])
        first.cancel_goal_async.assert_called_once()
        second = handle()
        response.set_result(second)
        self.assertEqual(self.stream.poll(), [{"revision": "second", "event": "ACCEPTED"}])
        second.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.owns_goals)
        self.assertTrue(self.stream.fenced)
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCELLED"):
            self.stream.submit(goal(), revision="third", dispatch_guard=self.guard)
        finish(first, 5)
        finish(second, 4)  # Success racing cancel remains success, not stop proof.
        events = self.stream.poll()
        self.assertEqual([e["result_status"] for e in events], [5, 4])
        self.assertFalse(self.stream.owns_goals)

    def test_original_deadline_includes_guard_time_and_is_not_renewed(self):
        first = self.initial()
        def delayed_guard(g, p):
            self.now = 20.
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCELLED"):
            self.stream.submit(goal(), revision="second", dispatch_guard=delayed_guard)
        self.client.send_goal_async.assert_called_once()
        self.stream.poll()
        first.cancel_goal_async.assert_called_once()
        self.assertEqual(self.stream.deadline, 20.)

    def test_ambiguous_send_is_owned_fenced_and_never_retried(self):
        error = OSError("transport failed")
        self.client.send_goal_async.side_effect = error
        with self.assertRaises(OSError) as raised:
            self.stream.submit(goal(), revision="first", dispatch_guard=self.guard)
        self.assertIs(raised.exception, error)
        self.assertTrue(self.stream.owns_goals)
        self.assertTrue(self.stream.fenced)
        self.assertEqual(self.stream.poll(), [])
        self.client.send_goal_async.assert_called_once()

    def test_pending_or_unresolved_predecessor_prevents_unbounded_handle_growth(self):
        first = self.initial()
        response = self.replacement()
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_GOAL_PENDING"):
            self.stream.submit(goal(), revision="third", dispatch_guard=self.guard)
        response.set_result(handle())
        self.stream.poll()
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_GOAL_PENDING"):
            self.stream.submit(goal(), revision="third", dispatch_guard=self.guard)
        self.assertFalse(first.get_result_async.return_value.done())
        self.assertEqual(self.client.send_goal_async.call_count, 2)

    def test_pending_response_error_does_not_prevent_cancelling_owned_predecessor(self):
        first = self.initial()
        response = self.replacement()
        error = OSError("acceptance response lost")
        response.set_exception(error)
        with self.assertRaises(OSError) as raised:
            self.stream.poll()
        self.assertIs(raised.exception, error)
        first.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.fenced)
        self.assertTrue(self.stream.owns_goals)

    def test_result_channel_error_preserves_acceptance_and_cancels_both_known_handles(self):
        first = self.initial()
        response = self.replacement()
        second = handle()
        error = OSError("result subscription failed")
        second.get_result_async.side_effect = error
        response.set_result(second)
        with self.assertRaises(OSError) as raised:
            self.stream.poll()
        self.assertIs(raised.exception, error)
        self.assertEqual(error.arm_stream_events, [{"revision": "second", "event": "ACCEPTED"}])
        first.cancel_goal_async.assert_called_once()
        second.cancel_goal_async.assert_called_once()
        self.assertTrue(self.stream.owns_goals)
        self.assertEqual(self.stream.poll(), [])
        second.get_result_async.assert_called_once()

    def test_cancel_send_failure_is_not_retried_or_treated_as_stop(self):
        first = self.initial()
        error = OSError("cancel response lost")
        first.cancel_goal_async.side_effect = error
        self.stream.cancel()
        with self.assertRaises(OSError) as raised:
            self.stream.poll()
        self.assertIs(raised.exception, error)
        self.assertEqual(self.stream.poll(), [])
        self.assertTrue(self.stream.owns_goals)
        first.cancel_goal_async.assert_called_once()
        finish(first, 5)
        self.assertEqual(self.stream.poll()[0]["result_status"], 5)
        self.assertFalse(self.stream.owns_goals)

    def test_active_controller_failure_fences_following_updates(self):
        first = self.initial()
        finish(first, 6, -4)
        event = self.stream.poll()[0]
        self.assertEqual((event["result_status"], event["error_code"]), (6, -4))
        self.assertTrue(self.stream.fenced)
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCELLED"):
            self.stream.submit(goal(), revision="second", dispatch_guard=self.guard)

    def test_native_controller_explanation_survives_without_relabeling_status(self):
        first = self.initial()
        first.get_result_async.return_value.set_result(SimpleNamespace(status=6,
            result=FollowJointTrajectory.Result(error_code=-1, error_string="preempted by successor")))
        event = self.stream.poll()[0]
        self.assertEqual(event["error_string"], "preempted by successor")
        self.assertEqual(event["result_status"], 6)

    def test_cancel_acknowledgement_matches_exact_goal_and_is_not_stop_evidence(self):
        for matching, code in ((True, 0), (False, 0), (True, 1)):
            with self.subTest(matching=matching, code=code):
                self.setUp()
                first = self.initial()
                self.stream.cancel()
                self.stream.poll()
                identifier = first.goal_id if matching else SimpleNamespace(uuid=[255] * 16)
                first.cancel_goal_async.return_value.set_result(SimpleNamespace(
                    return_code=code, goals_canceling=[SimpleNamespace(goal_id=identifier)]))
                events = self.stream.poll()
                self.assertEqual(events, [{"revision": "first", "event": "CANCEL_RESPONSE",
                    "return_code": code, "goal_id": list(first.goal_id.uuid),
                    "accepted": matching and code == 0}])
                self.assertTrue(self.stream.owns_goals)
                self.assertFalse(first.get_result_async.return_value.done())
                self.assertEqual(self.stream.poll(), [])

    def test_missing_native_preemption_result_remains_unresolved_not_fabricated(self):
        # Installed JTC4.40.1 can lose the old result. A mock supplying ABORTED
        # models the upstream fixed controller; it does not qualify this binary.
        first = self.initial()
        response = self.replacement()
        second = handle()
        response.set_result(second)
        self.stream.poll()
        finish(second)
        self.assertEqual(self.stream.poll()[0]["revision"], "second")
        self.assertTrue(self.stream.owns_goals)
        self.assertFalse(first.get_result_async.return_value.done())
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_GOAL_PENDING"):
            self.stream.submit(goal(), revision="third", dispatch_guard=self.guard)

    def test_ready_cancel_response_is_retained_even_when_terminal_is_already_ready(self):
        first = self.initial()
        self.stream.cancel()
        self.stream.poll()
        first.cancel_goal_async.return_value.set_result(SimpleNamespace(
            return_code=0, goals_canceling=[SimpleNamespace(goal_id=first.goal_id)]))
        finish(first, 5)
        self.assertEqual([e["event"] for e in self.stream.poll()], ["CANCEL_RESPONSE", "TERMINAL"])

    def test_failed_cancel_response_does_not_hide_actual_terminal_result(self):
        first = self.initial()
        self.stream.cancel()
        self.stream.poll()
        error = OSError("cancel acknowledgement lost")
        first.cancel_goal_async.return_value.set_exception(error)
        finish(first, 5)
        with self.assertRaises(OSError) as raised:
            self.stream.poll()
        self.assertIs(raised.exception, error)
        self.assertEqual(error.arm_stream_events, [{"revision": "first", "event": "TERMINAL",
            "result_status": 5, "error_code": 0}])
        self.assertFalse(self.stream.owns_goals)
        self.assertTrue(self.stream.fenced)
        self.assertEqual(self.stream.poll(), [])


class TransportActuatorStreamTest(unittest.TestCase):
    def setUp(self):
        self.transport = object.__new__(RosMoveItTransport)
        self.transport._active = None
        self.transport._execution_locked = False
        self.transport._execute_goal_count = 0
        self.transport._gripper_goal_count = 0
        self.transport._clock = lambda: 10.
        self.transport.node = object()
        self.transport._FollowJointTrajectory = FollowJointTrajectory
        self.transport._JointTrajectoryPoint = JointTrajectoryPoint
        self.client = Mock()
        self.response = Future()
        self.client.send_goal_async.return_value = self.response
        self.gripper_response = Future()
        self.transport.gripper = Mock()
        self.transport.gripper.send_goal_async.return_value = self.gripper_response
        self.transport._ActionClient = Mock(return_value=self.client)
        self.transport._rclpy = SimpleNamespace(spin_once=Mock())

    @staticmethod
    def goals():
        arm = goal()
        arm.trajectory.header.stamp.sec = 42
        gripper = FollowJointTrajectory.Goal()
        gripper.trajectory.header = copy.deepcopy(arm.trajectory.header)
        gripper.trajectory.joint_names = ["finger_right_joint"]
        point = JointTrajectoryPoint(positions=[.012])
        point.time_from_start.nanosec = 100_000_000
        gripper.trajectory.points = [point]
        return arm, gripper

    def test_same_transport_slot_excludes_legacy_and_reuses_gripper_client(self):
        self.transport.open_learned_actuator_stream(deadline=20.)
        self.transport._ActionClient.assert_called_once_with(
            self.transport.node, FollowJointTrajectory, "/fairino5_controller/follow_joint_trajectory")
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()
        self.assertTrue(self.transport.owns_active_goal)
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
            self.transport.start_phase({})
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
            self.transport.open_learned_actuator_stream(deadline=20.)
        arm, gripper = self.goals()
        guard = Mock()
        self.transport.submit_learned_actuator_revision(arm, gripper, revision="one", dispatch_guard=guard)
        self.assertEqual(self.client.send_goal_async.call_args.args[0], arm)
        self.assertEqual(self.transport.gripper.send_goal_async.call_args.args[0], gripper)
        guard.assert_called_once_with({"arm": arm, "gripper": gripper}, None)
        self.assertEqual(self.transport._execute_goal_count, 1)
        self.assertEqual(self.transport._gripper_goal_count, 1)
        self.assertEqual(self.transport.poll_learned_actuator_stream(), [])

    def test_builder_preserves_all_rows_and_shared_native_time_without_sending(self):
        initial = [0.] * 6 + [.021]
        actions = [[.001 * i + j / 10 for j in range(6)] + [.012 + i * .0001]
                   for i in range(50)]
        original = copy.deepcopy(actions)
        arm, gripper = self.transport.build_learned_actuator_goals(
            initial, actions, period_s=1 / 30, start_time_ns=42_123_456_789)
        self.assertEqual(arm.trajectory.header, gripper.trajectory.header)
        self.assertEqual(arm.trajectory.header.stamp.sec, 42)
        self.assertEqual(arm.trajectory.header.stamp.nanosec, 123_456_789)
        self.assertEqual(arm.trajectory.joint_names, [f"j{i}" for i in range(1, 7)])
        self.assertEqual(gripper.trajectory.joint_names, ["finger_right_joint"])
        self.assertEqual(len(arm.trajectory.points), 51)
        self.assertEqual(len(gripper.trajectory.points), 51)
        for index, (a, g) in enumerate(zip(arm.trajectory.points, gripper.trajectory.points)):
            expected = initial if index == 0 else original[index - 1]
            self.assertEqual(list(a.positions) + list(g.positions), expected)
            self.assertEqual(a.time_from_start, g.time_from_start)
            self.assertEqual(a.time_from_start.sec * 10**9 + a.time_from_start.nanosec,
                             round(index / 30 * 10**9))
            self.assertFalse(a.velocities or a.accelerations or g.velocities or g.accelerations)
        initial[0] = actions[0][0] = 9.
        self.assertEqual(arm.trajectory.points[0].positions[0], 0.)
        self.assertEqual(arm.trajectory.points[1].positions[0], original[0][0])
        self.transport._ActionClient.assert_not_called()
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()
        self.assertIsNone(self.transport._active)
        # The pure builder's exact output is accepted by the existing pair port;
        # admission remains the owner's responsibility, not builder authority.
        self.transport.open_learned_actuator_stream(deadline=20.)
        guard = Mock()
        self.transport.submit_learned_actuator_revision(arm, gripper, revision="built", dispatch_guard=guard)
        guard.assert_called_once_with({"arm": arm, "gripper": gripper}, None)
        self.client.send_goal_async.assert_called_once()
        self.transport.gripper.send_goal_async.assert_called_once()

    def test_builder_rejects_unrepresentable_inputs_without_effects(self):
        base = dict(initial_state=[0.] * 7, actions=[[0.] * 7],
                    period_s=1 / 30, start_time_ns=42_000_000_000)
        cases = [
            {"start_time_ns": value} for value in (0, -1, True, 42., 2**31 * 10**9)
        ] + [{"period_s": value} for value in
             (0, -1, True, "0.03", float("nan"), float("inf"), 1e-12, 2**31, 10**400)]
        cases += [{"actions": []}, {"initial_state": [0.] * 6}, {"actions": [[0.] * 8]}]
        for bad in (True, "0", float("nan"), float("inf"), 10**400):
            cases.extend(({"initial_state": [0.] * 6 + [bad]}, {"actions": [[bad] + [0.] * 6]}))
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ContractError, "ROS_EXEC_STEP"):
                    self.transport.build_learned_actuator_goals(**(base | changes))
        self.transport._ActionClient.assert_not_called()
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()
        self.assertIsNone(self.transport._active)

    def test_built_pair_uses_native_sampler_without_skipping_first_gripper_row(self):
        import pathlib
        import subprocess
        import tempfile

        initial = [0.] * 6 + [.021]
        actions = [[.001 * i] * 6 + [g] for i, g in enumerate((.012, .018, .014), 1)]
        arm, gripper = self.transport.build_learned_actuator_goals(
            initial, actions, period_s=.1, start_time_ns=10_200_000_000)
        # Reuse the native sampler fixture: it splits these same 7D knots back
        # into ARM spline and gripper NONE, with JTC's next-update sample time.
        knots = [(a.time_from_start.sec * 10**9 + a.time_from_start.nanosec,
                  *a.positions, *g.positions)
                 for a, g in zip(arm.trajectory.points, gripper.trajectory.points)]
        fixture = pathlib.Path(__file__).parent / "rollout" / "native_sampling_fixture.cpp"
        ros = pathlib.Path("/opt/ros/jazzy")
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "sampling"
            compiled = subprocess.run([
                "g++", "-std=c++17", *["-I" + str(p) for p in (ros / "include").iterdir() if p.is_dir()],
                str(fixture), "-L" + str(ros / "lib"), "-Wl,-rpath," + str(ros / "lib"),
                "-ljoint_trajectory_controller", "-lrclcpp", "-lrcutils", "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            def sample(points):
                data = str(len(points)) + "\n" + "\n".join(" ".join(map(str, p)) for p in points) + "\n"
                result = subprocess.run([str(binary), "mixed", "0", "0", "0", ".021"],
                                        input=data, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return [[float(v) for v in line.split()] for line in result.stdout.splitlines()]
            sampled = sample(knots)
            for index, tick in enumerate((20, 30, 40)):
                self.assertEqual(sampled[tick][-1], actions[index][-1])
            self.assertAlmostEqual(sampled[20][3], .0001)
            self.assertEqual(sampled[-1][3:], actions[-1])
            no_anchor = sample(knots[1:])
            self.assertEqual(no_anchor[20][-1], initial[-1])
            self.assertNotIn(actions[0][-1], [row[-1] for row in no_anchor])
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()

    def test_generic_cancel_never_reports_async_fence_as_completed_stop(self):
        self.transport.open_learned_actuator_stream(deadline=20.)
        self.assertEqual(self.transport.learned_actuator_stream_status(),
                         {"active": True, "fenced": False, "owns_goals": False})
        self.transport.submit_learned_actuator_revision(*self.goals(), revision="one", dispatch_guard=Mock())
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCEL_UNCERTAIN"):
            self.transport.cancel_active(.1)
        self.assertEqual(self.transport.learned_actuator_stream_status(),
                         {"active": True, "fenced": True, "owns_goals": True})
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
            self.transport.close_learned_actuator_stream()
        item, gripper = handle(), handle()
        self.response.set_result(item)
        self.gripper_response.set_result(gripper)
        self.transport.poll_learned_actuator_stream()
        item.cancel_goal_async.assert_called_once()
        gripper.cancel_goal_async.assert_called_once()
        self.assertTrue(self.transport.owns_active_goal)
        finish(item, 5)
        events = self.transport.poll_learned_actuator_stream()
        self.assertTrue(any(e["event"] == "TERMINAL" and e["actuator"] == "arm" and e["result_status"] == 5 for e in events))
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
            self.transport.close_learned_actuator_stream()
        finish(gripper, 5)
        self.transport.poll_learned_actuator_stream()
        self.assertEqual(self.transport.learned_actuator_stream_status(),
                         {"active": True, "fenced": True, "owns_goals": False})
        self.transport.close_learned_actuator_stream()
        self.assertEqual(self.transport.learned_actuator_stream_status(),
                         {"active": False, "fenced": False, "owns_goals": False})
        self.assertFalse(self.transport.owns_active_goal)
        self.assertTrue(self.transport._execution_locked)

    def test_native_pair_feedback_advances_reference_prefix_without_terminal_wait(self):
        from tools.data_factory.rollout.stream_progress import ReferenceProgress

        arm, gripper = self.transport.build_learned_actuator_goals(
            [0.] * 6 + [.021], [[i * .001] * 6 + [.012] for i in range(1, 5)],
            period_s=.1, start_time_ns=42_000_000_000)
        times = [p.time_from_start.sec * 10**9 + p.time_from_start.nanosec
                 for p in arm.trajectory.points]
        progress = ReferenceProgress("one", times)
        self.transport.open_learned_actuator_stream(deadline=20.)
        self.transport.submit_learned_actuator_revision(arm, gripper, revision="one", dispatch_guard=Mock())
        first, second = handle(), handle()
        self.response.set_result(first)
        self.gripper_response.set_result(second)
        def push(actuator, elapsed_ns):
            item = first if actuator == "arm" else second
            message = feedback(item, elapsed_ns)
            client = self.client if actuator == "arm" else self.transport.gripper
            if actuator == "gripper":
                message.feedback.joint_names = ["finger_right_joint"]
                message.feedback.desired.positions = [.012]
                message.feedback.actual.positions = [.011]
            client.send_goal_async.call_args.kwargs["feedback_callback"](message)
        def consume():
            import json

            events = self.transport.poll_learned_actuator_stream()
            self.assertEqual(json.loads(json.dumps(events)), events)
            for event in events:
                progress.observe(event)
            return events
        push("arm", 100_000_000)
        push("gripper", 50_000_000)
        self.assertIn("PAIR_ACCEPTED", [event["event"] for event in consume()])
        self.assertEqual(progress.selected_count, 0)
        push("arm", 300_000_000)
        push("gripper", 200_000_000)
        self.assertEqual([e["event"] for e in consume()], ["FEEDBACK", "FEEDBACK"])
        self.assertEqual(progress.selected_count, 2)
        self.assertFalse(first.get_result_async.return_value.done())
        self.assertFalse(second.get_result_async.return_value.done())
        self.client.send_goal_async.assert_called_once()
        self.transport.gripper.send_goal_async.assert_called_once()
        first.cancel_goal_async.assert_not_called()
        second.cancel_goal_async.assert_not_called()
        finish(first)
        consume()
        self.assertEqual(progress.selected_count, 2)
        finish(second)
        consume()
        self.assertEqual(progress.selected_count, 4)

    def test_native_collision_queries_actual_mixed_references_without_motion(self):
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity
        from sensor_msgs.msg import JointState
        from tools.fr5_data_factory import canonical_digest

        self.transport._GetStateValidity = GetStateValidity
        self.transport._RobotState = RobotState
        self.transport._JointState = JointState
        self.transport._service = Mock(return_value=SimpleNamespace(valid=True, constraint_result=[]))
        plan = {"frames": {"planning_group": "arm"}, "initial_joint_state": [0.] * 6, "steps": []}
        pair = self.transport.build_learned_actuator_goals(
            [0.] * 6 + [.021], [[.001] * 6 + [.012], [.002] * 6 + [.018]],
            period_s=.1, start_time_ns=42_000_000_000)
        original = copy.deepcopy(pair)
        report = self.transport.check_learned_actuator_collision(plan, *pair)
        self.assertEqual(pair, original)
        self.assertEqual(report["plan_digest"], canonical_digest(plan))
        self.assertEqual(report["sample_count"], 13)
        self.assertFalse(report["physical_tracking_qualified"])
        samples = report["samples"]
        self.assertEqual(samples[0]["finger_right_joint_m"], .021)
        self.assertEqual(samples[1]["joints_rad"], [0.] * 6)
        self.assertEqual(samples[1]["finger_right_joint_m"], .012)
        self.assertAlmostEqual(samples[2]["joints_rad"][0], .0002)
        self.assertEqual(samples[2]["finger_right_joint_m"], .012)
        self.assertEqual(samples[-1]["joints_rad"], [.002] * 6)
        self.assertEqual(samples[-1]["finger_right_joint_m"], .018)
        for call in self.transport._service.call_args_list:
            request = call.args[2]
            self.assertEqual(request.group_name, "")  # Whole robot, including gripper.
            self.assertEqual(list(request.robot_state.joint_state.name),
                             [f"j{i}" for i in range(1, 7)] + ["finger_right_joint"])
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()
        self.assertIsNone(self.transport._active)

        # A unified 7D interpolation would check .0192 here and miss this
        # native next-point gripper collision at the same ARM reference.
        def validity(_kind, _name, request, _code):
            q = request.robot_state.joint_state.position
            return SimpleNamespace(valid=not (abs(q[0] - .0002) < 1e-12 and q[-1] == .012),
                                   constraint_result=[])
        self.transport._service.side_effect = validity
        with self.assertRaisesRegex(ContractError, "COLLISION_DETECTED"):
            self.transport.check_learned_actuator_collision(plan, *pair)
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()

        def mutate_caller(*_args):
            pair[0].trajectory.points[-1].positions[0] = .9
            plan["frames"]["planning_group"] = "changed"
            return SimpleNamespace(valid=True, constraint_result=[])
        self.transport._service.side_effect = mutate_caller
        detached = self.transport.check_learned_actuator_collision(plan, *pair)
        self.assertEqual(detached, report)
        self.assertEqual(pair[0].trajectory.points[-1].positions[0], .9)

        # Header is part of the exact queried pair, not interchangeable timing.
        self.transport._service.side_effect = None
        later = copy.deepcopy(original)
        for item in later:
            item.trajectory.header.stamp.sec += 1
        self.assertNotEqual(self.transport.check_learned_actuator_collision(plan, *later)["native_pair_digest"],
                            report["native_pair_digest"])

    def test_native_collision_rejects_unsupported_timing_or_derivatives_before_query(self):
        pair = self.transport.build_learned_actuator_goals(
            [0.] * 6 + [.021], [[.001] * 6 + [.012]], period_s=.1, start_time_ns=42_000_000_000)
        self.transport._service = Mock()
        for change in (
            lambda a, g: setattr(a.trajectory.points[1], "velocities", [0.] * 6),
            lambda a, g: setattr(g.trajectory.points[1], "accelerations", [0.]),
            lambda a, g: setattr(g.trajectory.points[1].time_from_start, "nanosec", 200_000_000),
            lambda a, g: g.trajectory.points.pop(),
            lambda a, g: setattr(a.trajectory.points[0].time_from_start, "nanosec", 1),
            lambda a, g: setattr(g.trajectory.header.stamp, "sec", 43),
        ):
            with self.subTest(change=change):
                arm, gripper = copy.deepcopy(pair)
                change(arm, gripper)
                with self.assertRaises(ContractError):
                    self.transport.check_learned_actuator_collision({}, arm, gripper)
        self.transport._service.assert_not_called()
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()

    def test_native_send_failure_preserves_actual_attempt_counts(self):
        for failed_actuator, expected in (("arm", (1, 0)), ("gripper", (1, 1))):
            with self.subTest(actuator=failed_actuator):
                self.setUp()
                self.transport.open_learned_actuator_stream(deadline=20.)
                client = self.client if failed_actuator == "arm" else self.transport.gripper
                error = OSError("native send outcome unknown")
                client.send_goal_async.side_effect = error
                with self.assertRaises(OSError) as raised:
                    self.transport.submit_learned_actuator_revision(
                        *self.goals(), revision="one", dispatch_guard=Mock())
                self.assertIs(raised.exception, error)
                self.assertEqual((self.transport._execute_goal_count,
                                  self.transport._gripper_goal_count), expected)
                self.assertTrue(self.transport._active.fenced)
                self.assertTrue(self.transport.owns_active_goal)

    def test_rejected_guard_does_not_increment_native_attempt_counts(self):
        self.transport.open_learned_actuator_stream(deadline=20.)
        with self.assertRaisesRegex(ContractError, "SCENE_CHANGED"):
            self.transport.submit_learned_actuator_revision(*self.goals(), revision="one",
                dispatch_guard=Mock(side_effect=ContractError("SCENE_CHANGED")))
        self.assertEqual((self.transport._execute_goal_count,
                          self.transport._gripper_goal_count), (0, 0))
        self.assertFalse(self.transport._active.owns_goals)
        self.transport.fence_learned_actuator_stream()
        self.transport.close_learned_actuator_stream()
        self.assertFalse(self.transport.owns_active_goal)

    def test_stream_cannot_replace_an_existing_collection_motion_owner(self):
        existing = object()
        self.transport._active = existing
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
            self.transport.open_learned_actuator_stream(deadline=20.)
        self.assertIs(self.transport._active, existing)
        self.transport._ActionClient.assert_not_called()

    def test_no_open_stream_or_wrong_native_joint_contract_cannot_send(self):
        with self.assertRaises(ContractError):
            self.transport.submit_learned_actuator_revision(*self.goals(), revision="one", dispatch_guard=Mock())
        self.transport.open_learned_actuator_stream(deadline=20.)
        for actuator in (0, 1):
            with self.subTest(actuator=actuator):
                goals = self.goals()
                goals[actuator].trajectory.joint_names.append("wrong_joint")
                with self.assertRaisesRegex(ContractError, "ROS_EXEC_STEP"):
                    self.transport.submit_learned_actuator_revision(*goals, revision="one", dispatch_guard=Mock())
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()

    def test_pair_timing_and_structure_reject_before_either_send(self):
        self.transport.open_learned_actuator_stream(deadline=20.)
        def wrong_stamp(arm, gripper):
            gripper.trajectory.header.stamp.nanosec = 1
        def zero_stamp(arm, gripper):
            arm.trajectory.header.stamp.sec = gripper.trajectory.header.stamp.sec = 0
        def dimension(arm, gripper):
            gripper.trajectory.points[0].positions = [.01, .02]
        def nonfinite(arm, gripper):
            arm.trajectory.points[0].positions[0] = float("nan")
        def repeated_time(arm, gripper):
            arm.trajectory.points[1].time_from_start = copy.deepcopy(arm.trajectory.points[0].time_from_start)
        def derivative_dimension(arm, gripper):
            gripper.trajectory.points[0].velocities = [.1, .2]
        for change in (wrong_stamp, zero_stamp, dimension, nonfinite, repeated_time, derivative_dimension):
            with self.subTest(change=change.__name__):
                goals = self.goals()
                change(*goals)
                guard = Mock()
                with self.assertRaisesRegex(ContractError, "ROS_EXEC_STEP"):
                    self.transport.submit_learned_actuator_revision(*goals, revision="one", dispatch_guard=guard)
                guard.assert_not_called()
        self.client.send_goal_async.assert_not_called()
        self.transport.gripper.send_goal_async.assert_not_called()


if __name__ == "__main__":
    unittest.main()
