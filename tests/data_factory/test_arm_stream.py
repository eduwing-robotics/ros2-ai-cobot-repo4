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

    def test_generic_cancel_never_reports_async_fence_as_completed_stop(self):
        self.transport.open_learned_actuator_stream(deadline=20.)
        self.transport.submit_learned_actuator_revision(*self.goals(), revision="one", dispatch_guard=Mock())
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCEL_UNCERTAIN"):
            self.transport.cancel_active(.1)
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
        self.transport.close_learned_actuator_stream()
        self.assertFalse(self.transport.owns_active_goal)
        self.assertTrue(self.transport._execution_locked)

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
