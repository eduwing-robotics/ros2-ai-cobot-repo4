import base64
import copy
import math
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import MoveItErrorCodes, RobotTrajectory
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from tools.fr5_data_factory import ContractError
from tools.data_factory.motion.moveit_transport import RosMoveItTransport, _ActivePhase


class TestExecutionTransport(unittest.TestCase):
    @staticmethod
    def endpoint_fixture():
        from datetime import datetime, timezone
        from tests.data_factory.test_motion import snapshot
        from tests.data_factory.rollout.test_finite_plan import XML
        from tools.data_factory.rollout.gripper_evidence import SNAPSHOT_FIELDS, CALENDAR
        clock = [10.]
        def observed(receipt, position, sequence=2):
            value = snapshot([position] * 6)
            wire = dict.fromkeys(SNAPSHOT_FIELDS, 0.)
            wire.update(version=5., valid=1., arm_resumed=1., incarnation_0=1., incarnation_1=2.,
                        incarnation_2=3., incarnation_3=4., producer_sequence=float(sequence), connection_epoch=1.,
                        current_max_age_s=.1, host_clock_tolerance_s=.001, raw_reference_m=.01, feedback_m=.01,
                        sample_system_s=receipt, sample_steady_s=receipt, host_receive_steady_s=receipt,
                        source_progress_steady_s=receipt)
            stamp = datetime.fromtimestamp(receipt, timezone.utc)
            wire.update(zip(CALENDAR, [stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute,
                                       stamp.second, stamp.microsecond // 1000]))
            wire.update({f"arm_j{i}_rad": position for i in range(1, 7)})
            value["gripper_controller"]["hardware_execution"] = {"wire": wire, "received_steady_s": receipt,
                "clock_binding": {"schema_version": "fr5.gripper_temporal_policy.v2", "incarnation": [1, 2, 3, 4],
                                  "connection_epoch": 1, "configuration_epoch": 0, "max_age_s": .1,
                                  "host_clock_tolerance_s": .001}}
            return value
        segment = {"type": "ARM", "phase": "LEARNED_CHUNK", "learned_proposal": {"robot_description": XML},
                   "start_joint_state": [0.] * 6, "final_joint_state": [.1] * 6, "joint_tolerance_rad": .01,
                   "max_joint_state_age_s": .1, "gripper_position_m": .01,
                   "acceptable_feedback_m": {"min": .009, "max": .011}}
        result = SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED,
                                 result=SimpleNamespace(error_code=SimpleNamespace(val=MoveItErrorCodes.SUCCESS)))
        active = _ActivePhase("LEARNED_CHUNK", "ARM", 10.05, goal_handle=mock.Mock(),
            result_future=SimpleNamespace(done=lambda: True, result=lambda: result), arm_segment=segment,
            start_observation={"captured_at_s": 9.98, "captured_monotonic_s": 9.98,
                               "snapshot": observed(9.98, 0., 1)})
        transport = object.__new__(RosMoveItTransport)
        transport._active, transport._execution_locked = active, False
        transport._clock = lambda: clock[0]
        transport._rclpy, transport.node = SimpleNamespace(spin_once=lambda *a, **k: None), object()
        transport._goal_succeeded, transport._goal_canceled, transport._goal_aborted = 4, 5, 6
        transport._moveit_success, transport._gripper_success = MoveItErrorCodes.SUCCESS, 0
        return transport, active, clock, observed

    def test_native_arm_endpoint_wait_keeps_original_result_deadline_and_exact_sample(self):
        transport, active, clock, observed = self.endpoint_fixture()
        old, later = observed(9.99, 0.), observed(10.01, .1, 3)
        transport.snapshot = mock.Mock(side_effect=[old, later])
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", side_effect=lambda: clock[0]):
            self.assertIsNone(transport.poll_active())
            self.assertIs(transport._active, active)
            result = copy.deepcopy(active.action_terminal_observation)
            self.assertEqual(transport.terminal_verification()["status"], "PENDING")
            self.assertEqual(transport.terminal_verification()["checked_observation"]["snapshot"], old)
            clock[0] = 10.02
            self.assertIs(transport.poll_active(), active)
        self.assertIsNone(transport._active)
        self.assertEqual(active.deadline, 10.05)
        self.assertEqual(active.action_terminal_observation, result)
        self.assertEqual(active.terminal_observation["snapshot"], later)
        self.assertEqual(active.terminal_observation_status, "CONFIRMED")
        later["joint_positions"][0] = 2.
        self.assertEqual(active.terminal_observation["snapshot"]["joint_positions"][0], .1)
        active.goal_handle.cancel_goal_async.assert_not_called()

    def test_native_arm_valid_endpoint_needs_no_post_observed_result_receipt(self):
        transport, active, clock, observed = self.endpoint_fixture()
        transport.snapshot = mock.Mock(return_value=observed(9.99, .1))
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=clock[0]):
            self.assertIs(transport.poll_active(), active)
        self.assertLess(active.terminal_observation["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]["host_receive_steady_s"],
                        active.action_terminal_observation["observed_monotonic_s"])

    def test_native_arm_mismatch_expires_without_replacing_last_witness_or_canceled_result(self):
        transport, active, clock, observed = self.endpoint_fixture()
        transport.snapshot = mock.Mock(side_effect=[observed(9.99, 0.), observed(10.02, .05, 3)])
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", side_effect=lambda: clock[0]):
            self.assertIsNone(transport.poll_active())
            clock[0] = 10.03
            self.assertIsNone(transport.poll_active())  # Even later delivery is not acquisition causality.
            retained = transport.terminal_verification()
            clock[0] = 10.06
            with self.assertRaisesRegex(ContractError, "LEARNED_HARDWARE_COMPLETION_TIMEOUT"):
                transport.poll_active()
        self.assertEqual(transport.terminal_verification(), retained)
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCEL_NOT_CANCELED"):
            transport.cancel_active(.1)
        active.goal_handle.cancel_goal_async.assert_not_called()
        self.assertEqual(transport.terminal_verification(), retained)
        self.assertEqual(transport.poll_terminal_evidence()["result_status"], GoalStatus.STATUS_SUCCEEDED)
        self.assertEqual(transport.snapshot.call_count, 2)

    def test_native_arm_off_target_never_masks_hardware_gripper_or_identity_failure(self):
        mutations = [
            ("LEARNED_HARDWARE_INVALID", lambda s: s["gripper_controller"]["hardware_execution"]["wire"].update(device_main_error=1.)),
            ("LEARNED_HARDWARE_UNRESOLVED", lambda s: s["gripper_controller"]["hardware_execution"]["wire"].update(stopped=1.)),
            ("GRIPPER_FEEDBACK_OUT_OF_RANGE", lambda s: s["gripper_controller"].update(reference_position_m=.02)),
            ("LEARNED_HARDWARE_INCARNATION", lambda s: s["gripper_controller"]["hardware_execution"]["wire"].update(incarnation_0=9.)),
            ("LEARNED_HARDWARE_SUPERSEDED", lambda s: s["gripper_controller"]["hardware_execution"]["wire"].update(generation=1.)),
            ("LEARNED_HARDWARE_SCHEMA", lambda s: s["gripper_controller"]["hardware_execution"]["wire"].pop("device_main_error")),
            ("LEARNED_HARDWARE_STALE", lambda s: s["gripper_controller"]["hardware_execution"]["wire"].update(source_progress_steady_s=9.8)),
        ]
        for code, mutate in mutations:
            with self.subTest(code=code):
                transport, active, clock, observed = self.endpoint_fixture()
                sample = observed(9.99, 0.)
                mutate(sample)
                transport.snapshot = mock.Mock(return_value=sample)
                with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=clock[0]):
                    with self.assertRaisesRegex(ContractError, code):
                        transport.poll_active()
                retained = transport.terminal_verification()
                self.assertEqual(retained["status"], "REJECTED")
                self.assertEqual(retained["code"], code)
                self.assertEqual(retained["checked_observation"]["snapshot"], sample)

    def test_gripper_completion_returns_exact_checked_witness_for_delayed_executor(self):
        import threading
        from datetime import datetime, timezone
        from tools.data_factory.motion.pickup_executor import PickupExecutor
        from tools.data_factory.rollout.gripper_evidence import FIELDS, CALENDAR
        transport, active, clock, observed = self.endpoint_fixture()
        segment = copy.deepcopy(active.arm_segment)
        segment.update(type="GRIPPER", start_joint_state=[.1] * 6)
        active.type, active.arm_segment, active.held_segment = "GRIPPER", None, segment
        active.result_future = SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(
            status=GoalStatus.STATUS_SUCCEEDED, result=SimpleNamespace(error_code=0)))
        def legacy(sample, generation):
            hw = sample["gripper_controller"]["hardware_execution"]
            wire = {key: hw["wire"][key] for key in FIELDS}
            wire.update(version=1., generation=float(generation), completed_generation=float(generation),
                        completion_reason=float(generation), command_started_system_s=9.98 if generation else 0.)
            if generation:
                stamp = datetime.fromtimestamp(9.987, timezone.utc)
                wire.update(zip(("completion_" + key for key in CALENDAR), [stamp.year, stamp.month, stamp.day,
                    stamp.hour, stamp.minute, stamp.second, stamp.microsecond // 1000]))
            hw.update(wire=wire, clock_binding={"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                "calendar_to_system_offset_s": 0., "uncertainty_s": .001, "system_anchor_s": 9.,
                "steady_anchor_s": 9., "valid_until_system_s": 100.})
            return sample
        active.start_observation["snapshot"] = legacy(observed(9.98, .1, 1), 0)
        transport.snapshot = mock.Mock(return_value=legacy(observed(9.99, .1), 1))
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=10.):
            self.assertIs(transport.poll_active(), active)
        witness = copy.deepcopy(active.terminal_observation)
        self.assertEqual(transport.snapshot.call_count, 1)
        clock[0] = 10.5  # Historical completion is not invalidated by consumer delay.
        transport.poll_active = mock.Mock(return_value=active)
        transport.snapshot = mock.Mock(side_effect=AssertionError("confirmed witness must not be resampled"))
        executor = PickupExecutor(transport=transport, source_clock=lambda: clock[0], monotonic_clock=lambda: clock[0])
        executor._emit_phase_event = mock.Mock()
        executor._start_current_step = mock.Mock()
        run = {"state": "EXECUTING", "digest": "synthetic", "cancel_event": threading.Event(),
               "plan": {"learned_proposal": segment["learned_proposal"], "steps": [{"phase": "LEARNED_CHUNK",
                   "held_target_segments": [segment], "pause_after": "LEARNED_CHUNK_COMPLETE"}],
                   "planning": {"max_joint_state_age_s": .1}, "execution_timeouts_s": {"semantic_verdict": 1.}},
               "execution": {"step_index": 0, "lease_deadline": 11., "terminal_phases": [], "active": True,
                             "learned_start_observation": active.start_observation}}
        executor.runs = {"synthetic": run}
        with mock.patch("tools.data_factory.rollout.finite_plan.execution_step", side_effect=lambda s, p: s):
            executor._tick()
        self.assertEqual(run["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(run["execution"]["learned_segments"][0]["terminal_observation"], witness)
        transport.snapshot.assert_not_called()
        executor._start_current_step.assert_not_called()

    def test_executor_pending_timeout_retains_witness_before_fault_snapshot_and_cancel(self):
        import threading
        from tools.data_factory.motion.pickup_executor import PickupExecutor
        transport, active, clock, observed = self.endpoint_fixture()
        pending, fault = observed(9.99, 0.), observed(10.06, .1, 3)
        transport.snapshot = mock.Mock(side_effect=[pending, fault])
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=10.):
            self.assertIsNone(transport.poll_active())
        retained = transport.terminal_verification()
        clock[0] = 10.06
        executor = PickupExecutor(transport=transport, source_clock=lambda: clock[0], monotonic_clock=lambda: clock[0])
        executor._emit_phase_event = mock.Mock()
        run = {"state": "EXECUTING", "digest": "synthetic", "cancel_event": threading.Event(),
               "plan": {"run_id": "synthetic", "scene_binding": {}, "planning": {"max_joint_state_age_s": .1},
                        "execution_timeouts_s": {"cancel": .1}},
               "execution": {"step_index": 0, "lease_deadline": 11., "active": True}}
        executor.runs = {"synthetic": run}
        executor._tick()
        self.assertEqual(run["failure_code"], "LEARNED_HARDWARE_COMPLETION_TIMEOUT")
        self.assertEqual(run["execution"]["cancel_error"], "ROS_EXEC_CANCEL_NOT_CANCELED")
        self.assertEqual(run["execution"]["snapshot"], fault)
        expected = {"plan_digest": "synthetic", "segment_index": 0, **retained}
        self.assertEqual(run["execution"]["terminal_verification"], expected)
        self.assertEqual(executor._execution_data(run)["terminal_verification"], expected)
        self.assertTrue(run["cancel_event"].is_set())
        self.assertEqual(transport.poll_terminal_evidence()["result_status"], GoalStatus.STATUS_SUCCEEDED)
        self.assertEqual(run["execution"]["terminal_verification"], expected)

    def test_executor_cancellation_or_expired_lease_cannot_advance_confirmed_or_pending_result(self):
        import threading
        from tools.data_factory.motion.pickup_executor import PickupExecutor
        for canceled in (False, True):
            with self.subTest(canceled=canceled):
                transport, active, clock, observed = self.endpoint_fixture()
                transport.snapshot = mock.Mock(return_value=observed(9.99, .1 if canceled else 0.))
                with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=10.):
                    self.assertIs(transport.poll_active(), active if canceled else None)
                witness = copy.deepcopy(active.terminal_observation)
                clock[0] = 10.02
                executor = PickupExecutor(transport=transport, source_clock=lambda: clock[0], monotonic_clock=lambda: clock[0])
                executor._emit_phase_event = mock.Mock()
                executor._start_current_step = mock.Mock(side_effect=AssertionError("no later goal"))
                run = {"state": "EXECUTING", "digest": "synthetic", "cancel_event": threading.Event(),
                       "plan": {"run_id": "synthetic", "scene_binding": {}, "planning": {"max_joint_state_age_s": .1},
                                "execution_timeouts_s": {"cancel": .1}},
                       "execution": {"step_index": 0, "lease_deadline": 11. if canceled else 10.01, "active": True}}
                executor.runs = {"synthetic": run}
                def concurrent_cancel():
                    run["state"] = "BLOCKED"
                    run["cancel_event"].set()
                    return active
                transport.poll_active = mock.Mock(side_effect=concurrent_cancel if canceled else AssertionError("expired lease must not poll"))
                executor._tick()
                self.assertEqual(run["state"], "BLOCKED")
                self.assertEqual(run["execution"]["terminal_verification"]["checked_observation"], witness)
                self.assertEqual(run["execution"]["terminal_verification"]["status"], "CONFIRMED" if canceled else "PENDING")
                if not canceled:
                    self.assertEqual(run["failure_code"], "HEARTBEAT_TIMEOUT")
                    transport.poll_active.assert_not_called()
                executor._start_current_step.assert_not_called()
    def test_initial_contact_readback_failure_cleans_only_its_world_object(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneComponents
        from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
        from shape_msgs.msg import SolidPrimitive
        for fault in (None, "initial_read_timeout", "remove_timeout", "remove_false", "partial", "extra_world", "attachment", "floor_changed", "verify_timeout"):
            with self.subTest(fault=fault):
                transport = object.__new__(RosMoveItTransport)
                transport._CollisionObject, transport._SolidPrimitive, transport._Pose = CollisionObject, SolidPrimitive, Pose
                transport._GetPlanningScene, transport._PlanningSceneComponents = GetPlanningScene, PlanningSceneComponents
                transport._ApplyPlanningScene, transport._PlanningScene = ApplyPlanningScene, PlanningScene
                source = {"planning_scene":{"frame_id":"base_link",
                    "floor":{"id":"floor", "dimensions_m":[2.,2.,.05], "surface_z_m":0.},
                    "wall":{"id":"wall", "dimensions_m":[2.,.05,2.], "near_face_y_m":-.3}}}
                plan = {"steps":[{"held_target_segments":[{"type":"ARM"}]}],
                        "learned_source_program":source, "scene_binding":{"object_instance_id":"cube"}}
                context = {"dimensions_m":[.024]*3, "datum":{"translation_m":[.2,.5,.01],
                           "rotation_columns":[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}}
                scene, calls = PlanningScene(), []
                def service(kind, endpoint, request, code):
                    calls.append((endpoint, copy.deepcopy(request)))
                    if endpoint == "/apply_planning_scene":
                        objects = request.scene.world.collision_objects
                        if len(calls) == 1:
                            scene.world.collision_objects = copy.deepcopy(objects)
                            if fault == "extra_world": scene.world.collision_objects.append(transport._collision_object("foreign", [.1]*3, [0.]*3, "base_link"))
                            if fault == "attachment": scene.robot_state.attached_collision_objects = [AttachedCollisionObject(link_name="foreign", object=CollisionObject(id="foreign"))]
                            if fault == "floor_changed": scene.world.collision_objects[0].pose.position.z += .1
                        else:
                            self.assertEqual([(obj.id, obj.operation) for obj in objects], [("cube", CollisionObject.REMOVE)])
                            self.assertEqual(list(request.scene.robot_state.attached_collision_objects), [])
                            if fault == "remove_timeout": raise ContractError("SYNTHETIC_REMOVE_TIMEOUT")
                            if fault == "remove_false": return SimpleNamespace(success=False)
                            if fault != "partial": scene.world.collision_objects = [obj for obj in scene.world.collision_objects if obj.id != "cube"]
                        return SimpleNamespace(success=True)
                    if len(calls) == 2:
                        if fault == "initial_read_timeout": raise ContractError("SYNTHETIC_INITIAL_READ_TIMEOUT")
                        wrong = copy.deepcopy(scene)
                        next(obj for obj in wrong.world.collision_objects if obj.id == "cube").pose.position.x += .001
                        return SimpleNamespace(scene=wrong)
                    if fault == "verify_timeout": raise ContractError("SYNTHETIC_VERIFY_TIMEOUT")
                    return SimpleNamespace(scene=copy.deepcopy(scene))
                transport._service = service
                with mock.patch("tools.data_factory.motion.contact_transition.prepare", return_value=context):
                    result = transport.prepare_contact_transition(plan, {}, first=True, deadline_s=100.)
                self.assertEqual(result["status"], "UNAVAILABLE")
                self.assertEqual(result["code"], "SYNTHETIC_INITIAL_READ_TIMEOUT" if fault == "initial_read_timeout" else "PLANNING_SCENE_MISMATCH")
                cleanup = result["collision_world_cleanup"]
                self.assertEqual(cleanup["scope"], "COLLISION_WORLD_ONLY")
                self.assertEqual(cleanup["object_id"], "cube")
                self.assertEqual(cleanup["status"], "REMOVAL_CONFIRMED" if fault in (None, "initial_read_timeout") else "UNCONFIRMED")
                self.assertFalse(hasattr(transport, "_contact_world_object"))
                self.assertEqual(sum(endpoint == "/apply_planning_scene" for endpoint, _ in calls), 2)
                self.assertTrue({"floor", "wall"} <= {obj.id for obj in scene.world.collision_objects})
                if fault == "extra_world": self.assertIn("foreign", [obj.id for obj in scene.world.collision_objects])
                if fault == "attachment": self.assertEqual(len(scene.robot_state.attached_collision_objects), 1)

    def test_mechanical_scene_readback_allows_only_pose_roundoff(self):
        from geometry_msgs.msg import Pose
        from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject, CollisionObject,
                                     PlanningScene, PlanningSceneComponents)
        from moveit_msgs.srv import GetPlanningScene
        from shape_msgs.msg import Mesh, SolidPrimitive
        transport = object.__new__(RosMoveItTransport)
        transport._CollisionObject, transport._SolidPrimitive, transport._Pose = CollisionObject, SolidPrimitive, Pose
        transport._GetPlanningScene, transport._PlanningSceneComponents = GetPlanningScene, PlanningSceneComponents
        source = {"planning_scene":{"frame_id":"base_link",
            "floor":{"id":"table_floor_conservative_r003", "dimensions_m":[2.,2.,.05], "surface_z_m":-.015647},
            "wall":{"id":"home_back_wall_candidate", "dimensions_m":[2.,.05,2.], "near_face_y_m":-.3}}}
        cube = transport._collision_object("production-object-cf77147bbba918083d03", [.024]*3,
            [.2395916155781882, .5848593226212451, -.004652], "base_link")
        # Exact r6 request/readback values: one double-precision ulp plus signed zero.
        cube.primitive_poses[0].orientation.x = -0.
        cube.primitive_poses[0].orientation.z = .30808118803675266
        cube.primitive_poses[0].orientation.w = .9513600693627325
        for held in (False, True):
            for change in ("roundtrip", "sign", "position_roundoff", "position", "rotation", "nonunit", "nan", "inf",
                           "shape", "dimensions", "frame", "id", "operation", "pose_count", "mesh", "subframe", "stamp",
                           "extra_world", "missing_world", "duplicate_world", "extra_attachment", "touch_links", "link", "acm", "default_acm"):
                if not held and change in {"touch_links", "link"}:
                    continue
                with self.subTest(held=held, change=change):
                    expected = copy.deepcopy(cube)
                    if held: expected.header.frame_id = "gripper_link"
                    attached = AttachedCollisionObject(link_name="gripper_link", object=expected,
                        touch_links=["finger_tip_left_link", "finger_tip_right_link"]) if held else None
                    scene = PlanningScene()
                    scene.world.collision_objects = transport._planning_scene_objects(source["planning_scene"])
                    if held:
                        scene.robot_state.attached_collision_objects = [copy.deepcopy(attached)]
                        actual = scene.robot_state.attached_collision_objects[0].object
                    else:
                        scene.world.collision_objects.append(copy.deepcopy(expected))
                        actual = scene.world.collision_objects[-1]
                    actual.primitive_poses[0].orientation.x = 0.
                    actual.primitive_poses[0].orientation.w = .9513600693627324
                    if change == "sign":
                        for orientation in (actual.pose.orientation, actual.primitive_poses[0].orientation):
                            for key in ("x", "y", "z", "w"): setattr(orientation, key, -getattr(orientation, key))
                    if change == "position_roundoff": actual.pose.position.x += 1e-12
                    if change == "position": actual.pose.position.x += 2e-9
                    if change == "rotation": actual.primitive_poses[0].orientation.x += 2e-9
                    if change == "nonunit": actual.primitive_poses[0].orientation.w *= 2
                    if change == "nan": actual.pose.position.x = math.nan
                    if change == "inf": actual.primitive_poses[0].orientation.w = math.inf
                    if change == "shape": actual.primitives[0].type = SolidPrimitive.SPHERE
                    if change == "dimensions": actual.primitives[0].dimensions[0] += 1e-12
                    if change == "frame": actual.header.frame_id = "other_frame"
                    if change == "id": actual.id = "other_object"
                    if change == "operation": actual.operation = CollisionObject.REMOVE
                    if change == "pose_count": actual.primitive_poses.append(Pose())
                    if change == "mesh": actual.meshes.append(Mesh())
                    if change == "subframe": actual.subframe_names.append("extra")
                    if change == "stamp": actual.header.stamp.sec = 1
                    if change == "extra_world": scene.world.collision_objects.append(transport._collision_object("extra", [.1]*3, [0.]*3, "base_link"))
                    if change == "missing_world": scene.world.collision_objects.pop(0)
                    if change == "duplicate_world": scene.world.collision_objects.append(copy.deepcopy(scene.world.collision_objects[0]))
                    if change == "extra_attachment": scene.robot_state.attached_collision_objects.append(AttachedCollisionObject(object=copy.deepcopy(expected)))
                    if change == "touch_links": scene.robot_state.attached_collision_objects[0].touch_links.append("wrist3_link")
                    if change == "link": scene.robot_state.attached_collision_objects[0].link_name = "wrist3_link"
                    if change == "acm":
                        scene.allowed_collision_matrix.entry_names = [cube.id]
                        scene.allowed_collision_matrix.entry_values = [AllowedCollisionEntry(enabled=[True])]
                    if change == "default_acm":
                        scene.allowed_collision_matrix.default_entry_names = [cube.id]
                        scene.allowed_collision_matrix.default_entry_values = [True]
                    scene.world.collision_objects.reverse()
                    untouched = copy.deepcopy(scene)
                    transport._service = mock.Mock(return_value=SimpleNamespace(scene=scene))
                    if change in {"roundtrip", "sign", "position_roundoff"}:
                        self.assertTrue(transport._read_mechanical_scene(source, attached=attached,
                            released=None if held else expected).startswith("sha256:"))
                    else:
                        with self.assertRaises(ContractError):
                            transport._read_mechanical_scene(source, attached=attached, released=None if held else expected)
                    # NaN is unequal to itself; ROS serialization padding is
                    # not deterministic. The full message representation is.
                    self.assertEqual(str(scene), str(untouched))

    def test_native_v5_live_snapshot_requests_latest_state_not_history(self):
        policy = dict(schema_version="fr5.gripper_temporal_policy.v2",
            incarnation=[1, 2, 3, 4], connection_epoch=1, configuration_epoch=0,
            max_age_s=.1, host_clock_tolerance_s=.001)
        topics = {"/dynamic_joint_states", "/joint_states",
            "/fairino5_controller/controller_state", "/gripper_controller/controller_state"}
        for configured, supplied, expected in ((True, policy, 1), (False, policy, 10), (True, None, 10)):
            with self.subTest(configured=configured, policy=supplied):
                subscriptions = {}
                def subscribe(kind, topic, callback, qos):
                    subscriptions[topic] = qos
                    return object()
                node = SimpleNamespace(create_subscription=subscribe)
                with mock.patch("rclpy.action.ActionClient"):
                    RosMoveItTransport(node, gripper_temporal_policy=supplied,
                        allow_clock_configuration=configured)
                self.assertEqual({topic: subscriptions[topic] for topic in topics},
                                 dict.fromkeys(topics, expected))
                from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
                qos = QoSProfile(depth=expected)
                self.assertEqual(qos.history, HistoryPolicy.KEEP_LAST)
                self.assertEqual(qos.reliability, ReliabilityPolicy.RELIABLE)
                self.assertEqual(qos.durability, DurabilityPolicy.VOLATILE)
                self.assertEqual(subscriptions["/robot_description"].durability,
                                 DurabilityPolicy.TRANSIENT_LOCAL)

    def test_native_snapshot_drains_old_source_samples_within_original_budget(self):
        from tools.data_factory.rollout.finite_plan import check_execution_start
        description = (
            "<robot><ros2_control><hardware>"
            "<plugin>fairino_hardware/FairinoHardwareInterface</plugin>"
            "<param name='gripper_velocity'>20</param><param name='gripper_force'>20</param>"
            "<param name='gripper_settle_time_ms'>500</param>"
            "</hardware><joint name='finger_right_joint'/></ros2_control></robot>"
        )
        model = '<robot>' + ''.join(
            f'<joint name="{name}" type="{kind}"><limit lower="-3" upper="3" velocity="1"/></joint>'
            for name, kind in [(f'j{i}', 'revolute') for i in range(1, 7)]
            + [('finger_right_joint', 'prismatic')]) + '</robot>'
        step = {"max_joint_state_age_s": .1, "joint_tolerance_rad": .001,
                "gripper_tolerance_m": .001, "start_joint_state": [0.] * 6,
                "learned_proposal": {"robot_description": model, "initial_state": [0.] * 6 + [.01]}}
        for mode in ("queued_then_current", "paused_joint", "paused_arm", "paused_gripper",
                     "future_source", "expires_during_snapshot", "legacy"):
            with self.subTest(mode=mode):
                clock, spins = [10.], []
                transport = object.__new__(RosMoveItTransport)
                transport._clock = lambda: clock[0]
                transport.graph_timeout_s = .2
                transport.preflight_timeout_s = 1.
                transport._initial_snapshot_complete = True
                transport._robot_description = description
                transport._robot_description_client = None
                transport._native_clock_configured_age = None if mode == "legacy" else .1
                # Isolate ROS source acquisition: native SDK delivery/command
                # validation has its own tests and is not hardware-qualified here.
                transport._native_current_ready = lambda _: True
                kind = ["control_msgs/msg/JointTrajectoryControllerState"]
                def topics():
                    if mode == "expires_during_snapshot":
                        clock[0] += .11
                    return [("/fairino5_controller/controller_state", kind),
                            ("/gripper_controller/controller_state", kind)]
                transport.node = SimpleNamespace(count_publishers=lambda _: 1,
                    get_topic_names_and_types=topics)

                def publish(stamp_s):
                    joint = JointState(name=[f"j{i}" for i in range(1, 7)], position=[0.] * 6)
                    arm = JointTrajectoryControllerState(joint_names=["j1"], speed_scaling_factor=1.)
                    gripper = JointTrajectoryControllerState(joint_names=["finger_right_joint"], speed_scaling_factor=1.)
                    for name, message, position, callback in (
                            ("joint", joint, None, transport._on_joint_state),
                            ("arm", arm, 0., transport._on_arm_controller_state),
                            ("gripper", gripper, .01, transport._on_gripper_controller_state)):
                        source_s = (9. if mode == "paused_" + name else clock[0] - .01) if mode.startswith("paused_") else stamp_s
                        message.header.stamp.sec = int(source_s)
                        message.header.stamp.nanosec = round((source_s - int(source_s)) * 1e9)
                        if position is not None:
                            message.reference.positions = [position]
                            message.feedback.positions = [position]
                        callback(deserialize_message(serialize_message(message), type(message)))

                def spin(*_, timeout_sec):
                    spins.append(timeout_sec)
                    clock[0] += timeout_sec
                    publish(clock[0] - .01 if mode == "queued_then_current" and len(spins) >= 3
                            else 11. if mode == "future_source" else 9.)

                transport._rclpy = SimpleNamespace(spin_once=spin)
                publish(11. if mode == "future_source" else 9.99 if mode == "expires_during_snapshot" else 9.)
                with mock.patch("tools.data_factory.motion.moveit_transport.time.monotonic", side_effect=lambda: clock[0]), \
                     mock.patch("tools.data_factory.motion.moveit_transport.time.time", side_effect=lambda: clock[0]):
                    if mode.startswith("paused_") or mode in ("future_source", "expires_during_snapshot"):
                        with self.assertRaisesRegex(ContractError, "LEARNED_STALE_STATE"):
                            transport.snapshot(.1)
                        self.assertLessEqual(clock[0], 10.2)
                        continue
                    observed = transport.snapshot(.1)
                    evidence = {"snapshot": observed, "captured_at_s": clock[0]}
                    if mode == "legacy":
                        self.assertEqual(spins, [])
                        with self.assertRaisesRegex(ContractError, "LEARNED_STALE_STATE"):
                            check_execution_start(step, evidence, clock[0], steady_now=clock[0])
                    else:
                        self.assertEqual(len(spins), 3)
                        self.assertEqual(observed["joint_state_stamp_ns"], round((clock[0] - .01) * 1e9))
                        with mock.patch("tools.data_factory.rollout.gripper_evidence.check_hardware", return_value={"version": 5}):
                            self.assertEqual(check_execution_start(step, evidence, clock[0], steady_now=clock[0]),
                                             [0.] * 6 + [.01])

    def test_initial_snapshot_discovery_budget_does_not_relax_live_freshness(self):
        description = (
            "<robot><ros2_control><hardware>"
            "<plugin>fairino_hardware/FairinoHardwareInterface</plugin>"
            "<param name='gripper_velocity'>20</param>"
            "<param name='gripper_force'>20</param>"
            "<param name='gripper_settle_time_ms'>500</param>"
            "</hardware><joint name='finger_right_joint'/>"
            "</ros2_control></robot>"
        )
        controller_type = "control_msgs/msg/JointTrajectoryControllerState"
        for mode in ("late_description", "missing_description", "invalid_description"):
            with self.subTest(mode=mode):
                clock = [0.0]
                transport = object.__new__(RosMoveItTransport)
                transport.graph_timeout_s = 1.0
                transport.preflight_timeout_s = 5.0
                transport._initial_snapshot_complete = False
                transport._clock = lambda: clock[0]
                transport._robot_description = None
                transport._robot_description_client = None
                transport._joint_state_received_at = None
                transport._arm_controller_received_at = None
                transport._gripper_controller_received_at = None
                transport.node = SimpleNamespace(
                    count_publishers=lambda _: 1,
                    get_topic_names_and_types=lambda: [
                        ("/fairino5_controller/controller_state", [controller_type]),
                        ("/gripper_controller/controller_state", [controller_type]),
                    ],
                )

                def spin(*_, timeout_sec):
                    clock[0] += timeout_sec
                    joint_state = JointState(
                        name=["j1", "j2", "j3", "j4", "j5", "j6"],
                        position=[0.0] * 6,
                    )
                    joint_state.header.stamp.sec = 1234
                    joint_state.header.stamp.nanosec = 5678
                    transport._on_joint_state(joint_state)
                    for callback, name in (
                        (transport._on_arm_controller_state, "j1"),
                        (transport._on_gripper_controller_state, "finger_right_joint"),
                    ):
                        message = JointTrajectoryControllerState()
                        message.joint_names = [name]
                        message.reference.positions = [0.0]
                        message.feedback.positions = [0.0]
                        message.speed_scaling_factor = 1.0
                        callback(message)
                    if clock[0] >= 1.5 and mode != "missing_description":
                        transport._on_robot_description(SimpleNamespace(
                            data=description if mode == "late_description" else "<robot/>",
                        ))

                transport._rclpy = SimpleNamespace(spin_once=spin)
                with mock.patch(
                    "tools.data_factory.motion.moveit_transport.time.monotonic",
                    side_effect=lambda: clock[0],
                ):
                    if mode != "late_description":
                        with self.assertRaises(ContractError) as caught:
                            transport.snapshot(0.1)
                        self.assertEqual(caught.exception.code, "ROS_GRIPPER_SETTINGS_UNVERIFIED")
                        self.assertFalse(transport._initial_snapshot_complete)
                        self.assertLessEqual(clock[0], 5.0)
                        continue
                    snapshot = transport.snapshot(0.1)
                    self.assertEqual(snapshot["joint_state_stamp_ns"], 1234000005678)
                    self.assertEqual(snapshot["gripper_settings"]["velocity_percent"], 20)
                    self.assertTrue(transport._initial_snapshot_complete)
                    self.assertLess(clock[0], 1.6)  # No unconditional five-second sleep.
                    clock[0] += 1.0
                    stale_started = clock[0]
                    transport._rclpy.spin_once = (
                        lambda *_, timeout_sec: clock.__setitem__(0, clock[0] + timeout_sec)
                    )
                    with self.assertRaisesRegex(ContractError, "ROS_JOINT_STATE_STALE"):
                        transport.snapshot(0.1)
                    self.assertAlmostEqual(clock[0] - stale_started, 1.0)

    def test_cancel_race_keeps_non_cancel_terminal_result_pollable(self):
        class Future:
            def __init__(self, value):
                self.value = value

            def done(self):
                return True

            def result(self):
                return self.value

        class Handle:
            def __init__(self, result):
                self.result = result

            def cancel_goal_async(self):
                return Future(SimpleNamespace(goals_canceling=[object()]))

        for status in (
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_ABORTED,
        ):
            with self.subTest(status=status):
                result = Future(SimpleNamespace(status=status))
                active = SimpleNamespace(
                    phase="SAFE_POSE_PTP", type="ARM",
                    goal_handle=Handle(result), result_future=result,
                )
                transport = object.__new__(RosMoveItTransport)
                transport._active = active
                transport._execution_locked = False
                transport._goal_succeeded = GoalStatus.STATUS_SUCCEEDED
                transport._goal_canceled = GoalStatus.STATUS_CANCELED
                transport._goal_aborted = GoalStatus.STATUS_ABORTED
                transport.node = object()
                transport._rclpy = SimpleNamespace(
                    spin_until_future_complete=lambda *_args, **_kwargs: None,
                    spin_once=lambda *_args, **_kwargs: None,
                )

                if status == GoalStatus.STATUS_CANCELED:
                    self.assertIs(transport.cancel_active(1.0), active)
                    self.assertFalse(transport.owns_active_goal)
                    continue
                with self.assertRaisesRegex(
                    ContractError, "ROS_EXEC_CANCEL_NOT_CANCELED",
                ):
                    transport.cancel_active(1.0)
                self.assertTrue(transport.owns_active_goal)
                self.assertEqual(
                    transport.poll_terminal_evidence()["result_status"], status,
                )
                self.assertFalse(transport.owns_active_goal)

    def test_robot_description_parameter_retries_transient_future_failure(self):
        description = (
            "<robot><ros2_control><hardware>"
            "<plugin>fairino_hardware/FairinoHardwareInterface</plugin>"
            "<param name='gripper_velocity'>20</param>"
            "<param name='gripper_force'>50</param>"
            "<param name='gripper_settle_time_ms'>500</param>"
            "</hardware><joint name='finger_right_joint'/>"
            "</ros2_control></robot>"
        )

        class Future:
            def __init__(self, result): self.result_value = result
            def done(self): return True
            def result(self):
                if isinstance(self.result_value, Exception):
                    raise self.result_value
                return self.result_value

        response = SimpleNamespace(values=[SimpleNamespace(
            type=4, string_value=description,
        )])
        for transient_error in (RuntimeError("graph changed"), TimeoutError()):
            with self.subTest(error=type(transient_error).__name__):
                results = iter((transient_error, response))
                client = SimpleNamespace(
                    wait_for_services=lambda **_: True,
                    get_parameters=mock.Mock(
                        side_effect=lambda _: Future(next(results))
                    ),
                )
                transport = object.__new__(RosMoveItTransport)
                transport.node = object()
                transport._robot_description = None
                transport._robot_description_client = client
                transport._parameter_string = 4
                transport._rclpy = SimpleNamespace(
                    spin_once=lambda *args, **kwargs: None,
                )

                transport._load_robot_description_parameter(
                    time.monotonic() + 1.0,
                )

                self.assertEqual(transport._robot_description, description)
                self.assertEqual(client.get_parameters.call_count, 2)

    def test_execute_cancel_and_snapshot_contract(self):
        class Future:
            def __init__(self, value, done=True): self.value, self.complete = value, done
            def done(self): return self.complete
            def result(self): return self.value

        class Handle:
            accepted = True
            def __init__(self, result, canceling=True): self.result, self.canceled, self.canceling = result, 0, canceling
            def get_result_async(self): return self.result
            def cancel_goal_async(self):
                self.canceled += 1
                return Future(SimpleNamespace(goals_canceling=[object()] if self.canceling else []))

        class Client:
            def __init__(self): self.goals, self.handle, self.send_done = [], None, True
            def wait_for_server(self, timeout_sec): return True
            def send_goal_async(self, goal): self.goals.append(goal); return Future(self.handle, self.send_done)

        class Node:
            def __init__(self): self.callbacks = {}
            def create_subscription(self, kind, topic, callback, depth):
                self.callbacks[topic] = callback
                return object()
            def count_publishers(self, topic): return 1
            def get_topic_names_and_types(self):
                kind = ["control_msgs/msg/JointTrajectoryControllerState"]
                return [("/fairino5_controller/controller_state", kind), ("/gripper_controller/controller_state", kind)]

        clock = [10.0]
        clients = {}
        def client_factory(node, kind, topic):
            del node, kind
            clients[topic] = Client()
            return clients[topic]
        node = Node()
        with mock.patch("rclpy.action.ActionClient", side_effect=client_factory):
            transport = RosMoveItTransport(node, clock=lambda: clock[0])
        transport._rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *args, **kwargs: None,
            spin_once=lambda *args, **kwargs: None,
        )

        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ["j1"]
        trajectory.joint_trajectory.points = [JointTrajectoryPoint(positions=[1.0])]
        arm = {"phase": "ARM", "type": "ARM", "trajectory_b64": base64.b64encode(serialize_message(trajectory)).decode(), "limits": {"execution_timeout_s": 2.0}}
        arm_result = transport._ExecuteTrajectory.Result()
        arm_result.error_code.val = MoveItErrorCodes.SUCCESS
        clients["/execute_trajectory"].handle = Handle(Future(SimpleNamespace(status=GoalStatus.STATUS_SUCCEEDED, result=arm_result)))
        active = transport.start_phase(arm)
        sent = clients["/execute_trajectory"].goals[-1]
        self.assertEqual(deserialize_message(serialize_message(sent.trajectory), RobotTrajectory).joint_trajectory.joint_names, ["j1"])
        self.assertEqual(transport.poll_active(), active)

        # Exercise the actual CDR goal decoder, not just a delayed compiler stub.
        decoded = transport._deserialize_message
        deadline = clock[0] + .05
        count = len(clients["/execute_trajectory"].goals)
        def slow_decode(*args):
            value = decoded(*args)
            clock[0] += .1
            return value
        def original_deadline():
            if clock[0] >= deadline:
                raise ContractError("HEARTBEAT_TIMEOUT")
        with mock.patch.object(transport, "_deserialize_message", side_effect=slow_decode):
            with self.assertRaisesRegex(ContractError, "HEARTBEAT_TIMEOUT"):
                transport.start_phase(arm, dispatch_guard=original_deadline)
        self.assertEqual(len(clients["/execute_trajectory"].goals), count)
        self.assertIsNone(transport._active)
        self.assertFalse(transport._execution_locked)
        with self.assertRaisesRegex(ContractError, "ROS_EXEC_DISPATCH_GUARD"):
            transport.start_phase(arm, dispatch_guard=1)
        self.assertEqual(len(clients["/execute_trajectory"].goals), count)
        clock[0] = 10.0

        gripper_goal = transport._FollowJointTrajectory.Goal()
        gripper = {"phase": "GRIPPER", "type": "GRIPPER", "trajectory_b64": base64.b64encode(serialize_message(gripper_goal)).decode(), "limits": {"execution_timeout_s": 2.0}}
        canceled_result = Future(SimpleNamespace(status=GoalStatus.STATUS_CANCELED, result=SimpleNamespace(error_code=0)))
        clients["/gripper_controller/follow_joint_trajectory"].handle = Handle(canceled_result)
        active = transport.start_phase(gripper)
        self.assertEqual(deserialize_message(serialize_message(clients["/gripper_controller/follow_joint_trajectory"].goals[-1]), FollowJointTrajectory.Goal).trajectory.joint_names, [])
        self.assertEqual(transport.cancel_active(1.0), active)
        self.assertIsNone(transport._active)
        with self.assertRaises(ContractError) as caught:
            transport.start_phase(gripper)
        self.assertEqual(caught.exception.code, "ROS_EXEC_ACTIVE")

        mapping = {"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                   "calendar_to_system_offset_s": 0., "uncertainty_s": .001,
                   "system_anchor_s": 10., "steady_anchor_s": 10., "valid_until_system_s": 20.}
        with mock.patch("rclpy.action.ActionClient", side_effect=client_factory):
            transport = RosMoveItTransport(node, clock=lambda: clock[0])
        transport._rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *args, **kwargs: None,
            spin_once=lambda *args, **kwargs: None,
        )
        with self.assertRaises(ContractError) as caught:
            transport.start_phase({**arm, "trajectory_b64": "?"})
        self.assertEqual(caught.exception.code, "ROS_EXEC_B64")

        clients["/execute_trajectory"].handle = Handle(Future(SimpleNamespace()))
        clients["/execute_trajectory"].send_done = False
        with self.assertRaises(ContractError) as caught:
            transport.start_phase(arm)
        self.assertEqual(caught.exception.code, "ROS_EXEC_GOAL_TIMEOUT")
        goal_count = len(clients["/execute_trajectory"].goals)
        with self.assertRaises(ContractError) as caught:
            transport.start_phase(arm)
        self.assertEqual(caught.exception.code, "ROS_EXEC_ACTIVE")
        self.assertEqual(len(clients["/execute_trajectory"].goals), goal_count)

        with mock.patch("rclpy.action.ActionClient", side_effect=client_factory):
            transport = RosMoveItTransport(node, clock=lambda: clock[0])
        transport._rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *args, **kwargs: None,
            spin_once=lambda *args, **kwargs: None,
        )
        clients["/execute_trajectory"].handle = None
        with self.assertRaises(ContractError) as caught:
            transport.start_phase(arm)
        self.assertEqual(caught.exception.code, "ROS_EXEC_GOAL_RESPONSE_INVALID")
        goal_count = len(clients["/execute_trajectory"].goals)
        with self.assertRaises(ContractError) as caught:
            transport.start_phase(arm)
        self.assertEqual(caught.exception.code, "ROS_EXEC_ACTIVE")
        self.assertEqual(len(clients["/execute_trajectory"].goals), goal_count)

        with mock.patch("rclpy.action.ActionClient", side_effect=client_factory):
            transport = RosMoveItTransport(node, clock=lambda: clock[0], gripper_source_clock=mapping)
        transport._rclpy = SimpleNamespace(
            spin_until_future_complete=lambda *args, **kwargs: None,
            spin_once=lambda *args, **kwargs: None,
        )
        clients["/gripper_controller/follow_joint_trajectory"].handle = Handle(Future(SimpleNamespace(status=GoalStatus.STATUS_ABORTED, result=SimpleNamespace(error_code=0))), canceling=False)
        transport.start_phase(gripper)
        with self.assertRaises(ContractError) as caught:
            transport.cancel_active(1.0)
        self.assertEqual(caught.exception.code, "ROS_EXEC_CANCEL_REJECTED")
        self.assertIsNotNone(transport._active)
        with self.assertRaises(ContractError):
            transport.poll_active()
        with self.assertRaises(ContractError) as caught:
            transport.start_phase(gripper)
        self.assertEqual(caught.exception.code, "ROS_EXEC_ACTIVE")

        from builtin_interfaces.msg import Duration
        from control_msgs.msg import JointTrajectoryControllerState
        arm_state = JointTrajectoryControllerState(joint_names=["j1"], speed_scaling_factor=.5)
        gripper_state = JointTrajectoryControllerState(joint_names=["finger_right_joint"], speed_scaling_factor=1.)
        for message, position in ((arm_state, 0.), (gripper_state, .01)):
            message.header.stamp.sec = 10
            message.reference.positions = [position]
            message.feedback.positions = [position]
            message.reference.time_from_start = Duration(sec=-1, nanosec=800000000)
            message.feedback.time_from_start = Duration(sec=-1, nanosec=900000000)
        arm_state.output.positions = [-.05]
        from sensor_msgs.msg import JointState
        joint_state = JointState(name=["finger", "j6", "j5", "j4", "j3", "j2", "j1"], position=[0., 6., 5., 4., 3., 2., 1.])
        joint_state.header.stamp.sec, joint_state.header.stamp.nanosec = 10, 123
        def populate(*_, **__):
            node.callbacks["/joint_states"](deserialize_message(serialize_message(joint_state), JointState))
            node.callbacks["/fairino5_controller/controller_state"](deserialize_message(serialize_message(arm_state), JointTrajectoryControllerState))
            node.callbacks["/gripper_controller/controller_state"](deserialize_message(serialize_message(gripper_state), JointTrajectoryControllerState))
            node.callbacks["/robot_description"](SimpleNamespace(data="<robot><ros2_control><hardware><plugin>fairino_hardware/FairinoHardwareInterface</plugin><param name='gripper_velocity'>20</param><param name='gripper_force'>50</param><param name='gripper_settle_time_ms'>500</param></hardware><joint name='finger_right_joint'/></ros2_control></robot>"))
        transport._rclpy = SimpleNamespace(spin_until_future_complete=lambda *args, **kwargs: None, spin_once=populate)
        snapshot = transport.snapshot(1.0)
        self.assertEqual(snapshot["joint_positions"], [1., 2., 3., 4., 5., 6.])
        self.assertEqual(snapshot["joint_state_stamp_ns"], 10_000_000_123)
        transport._joint_state.header.stamp.nanosec = 10**9
        with self.assertRaisesRegex(ContractError, "ROS_JOINT_STATE"):
            transport.snapshot(1.)
        transport._joint_state.header.stamp.nanosec = 123
        self.assertEqual((snapshot["arm_controller"]["ready"], snapshot["arm_controller"]["speed_scaling"]), (True, 0.5))
        self.assertEqual((snapshot["gripper_controller"]["reference_position_m"], snapshot["gripper_controller"]["feedback_position_m"]), (0.01, 0.01))
        self.assertEqual(snapshot["gripper_settings"]["velocity_percent"], 20)
        retained = snapshot["arm_controller"]["sample"]
        self.assertEqual((retained["ros_stamp_ns"], retained["reference_elapsed_ns"], retained["feedback_elapsed_ns"]),
                         (10_000_000_000, -200_000_000, -100_000_000))
        self.assertEqual((retained["reference_positions"], retained["reported_output_positions"]), ([0.], [-.05]))
        self.assertEqual(snapshot["gripper_controller"]["sample"]["reported_output_positions"], [])
        # A later publication may retain an old command output in native JTC.
        # Capture its literal report; never fill an empty output from reference.
        arm_state.header.stamp.nanosec = 1
        populate()
        self.assertEqual(transport.snapshot(1.)["arm_controller"]["sample"]["reported_output_positions"], [-.05])
        self.assertEqual(retained["ros_stamp_ns"], 10_000_000_000)
        arm_state.header.stamp.nanosec = 1_000_000_000
        populate()
        with self.assertRaisesRegex(ContractError, "ROS_CONTROLLER_SAMPLE"):
            transport.snapshot(1.)
        arm_state.header.stamp.nanosec = 0
        populate()

        # Actual constructor subscription, callback and snapshot consume the wire.
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from tools.data_factory.rollout.gripper_evidence import FIELDS, RESOURCE
        raw = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
            interface_names=list(FIELDS), values=[0.] * len(FIELDS))])
        node.callbacks["/dynamic_joint_states"](deserialize_message(serialize_message(raw), DynamicJointState))
        hardware = transport.snapshot(1.)["gripper_controller"]["hardware_execution"]
        self.assertEqual(hardware["clock_binding"], mapping)
        self.assertEqual(hardware["received_steady_s"], clock[0])
        self.assertEqual(hardware["wire"], dict.fromkeys(FIELDS, 0.))
        # Decoding conveys the record; held admission, not snapshot presence, checks validity.
        transport._robot_description = None
        transport._robot_description_client = SimpleNamespace(
            wait_for_services=lambda **_: True,
            get_parameters=lambda _: Future(SimpleNamespace(values=[SimpleNamespace(
                type=4,
                string_value="<robot><ros2_control><hardware><plugin>fairino_hardware/FairinoHardwareInterface</plugin><param name='gripper_velocity'>20</param><param name='gripper_open_velocity'>10</param><param name='gripper_force'>50</param><param name='gripper_open_force'>45</param><param name='gripper_settle_time_ms'>500</param></hardware><joint name='finger_right_joint'/></ros2_control></robot>",
            )])),
        )
        transport._rclpy.spin_once = lambda *args, **kwargs: None
        self.assertEqual(transport.snapshot(1.0)["gripper_settings"]["force_percent"], 50)
        self.assertEqual(
            transport.snapshot(1.0)["gripper_settings"]["open_force_percent"], 45,
        )
        self.assertEqual(
            transport.snapshot(1.0)["gripper_settings"]["open_velocity_percent"],
            10,
        )

        transport._robot_description = None
        transport._robot_description_client = SimpleNamespace(
            wait_for_services=lambda **_: True,
            get_parameters=lambda _: Future(None, done=False),
        )
        transport._rclpy.spin_once = lambda *args, **kwargs: node.callbacks[
            "/robot_description"
        ](SimpleNamespace(data=(
            "<robot><ros2_control><hardware>"
            "<plugin>fairino_hardware/FairinoHardwareInterface</plugin>"
            "<param name='gripper_velocity'>20</param>"
            "<param name='gripper_force'>50</param>"
            "<param name='gripper_settle_time_ms'>500</param>"
            "</hardware><joint name='finger_right_joint'/>"
            "</ros2_control></robot>"
        )))
        self.assertEqual(
            transport.snapshot(1.0)["gripper_settings"]["settle_time_ms"], 500,
        )

        clock[0] = 12.0
        transport.graph_timeout_s = 0.0
        transport._rclpy.spin_once = lambda *args, **kwargs: None
        with self.assertRaisesRegex(ContractError, "ROS_JOINT_STATE_STALE"):
            transport.snapshot(1.0)
        transport._joint_state_received_at = transport._arm_controller_received_at = transport._gripper_controller_received_at = None
        with self.assertRaisesRegex(ContractError, "ROS_JOINT_STATE_STALE"):
            transport.snapshot(1.0)


if __name__ == "__main__":
    unittest.main()

class NativeClockPreparationTest(unittest.TestCase):
    def test_causal_bootstrap_binds_identity_not_legacy_readiness_and_keeps_plan_only_read_only(self):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from tools.data_factory.rollout.gripper_evidence import FIELDS, CAUSAL_FIELDS, RESOURCE, native_temporal_parameter
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        policy = {"schema_version": "fr5.gripper_temporal_policy.v1", "incarnation": [1, 2, 3, 4],
                  "max_age_s": .3, "host_clock_tolerance_s": .001}
        for mode in ("live", "legacy_identity", "plan_only", "wrong_incarnation", "wrong_age", "readback_mismatch"):
            with self.subTest(mode=mode):
                t = object.__new__(RosMoveItTransport)
                t._gripper_source_clock = policy
                t._allow_clock_configuration = mode != "plan_only"
                t._native_clock_configured_age = None
                t.preflight_timeout_s = t.graph_timeout_s = .1
                t.node = SimpleNamespace()
                names = FIELDS if mode == "legacy_identity" else CAUSAL_FIELDS
                wire = dict.fromkeys(names, 0.)
                wire.update(version=1. if mode == "legacy_identity" else 3., incarnation_0=99. if mode == "wrong_incarnation" else 1.,
                            incarnation_1=2., incarnation_2=3., incarnation_3=4.)
                t._gripper_hardware_state = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
                    interface_names=list(names), values=[wire[k] for k in names])])
                t._gripper_hardware_received_at = 10.
                configured = []
                def set_parameters(parameters):
                    configured.append((parameters[0].name, parameters[0].value))
                    return SimpleNamespace(result=SimpleNamespace(successful=True))
                def get_parameters(names):
                    self.assertEqual(names, ["gripper_temporal_policy_v1"])
                    return SimpleNamespace(values=[SimpleNamespace(double_array_value=
                        [] if mode == "readback_mismatch" else configured[-1][1])])
                client = SimpleNamespace(wait_for_services=lambda **_: True,
                    set_parameters_atomically=set_parameters, get_parameters=get_parameters)
                t._AsyncParameterClient = lambda *_: client
                t._wait = lambda result, *_: result
                if mode in ("wrong_incarnation", "wrong_age", "readback_mismatch"):
                    with self.assertRaises(ContractError):
                        t._prepare_native_clock(.2 if mode == "wrong_age" else .3)
                else:
                    t._prepare_native_clock(.3)
                self.assertEqual(len(configured), int(mode in ("live", "legacy_identity", "readback_mismatch")))
                if configured:
                    self.assertEqual(configured[0], ("gripper_temporal_policy_v1", native_temporal_parameter(policy)))
                if mode in ("live", "legacy_identity"):
                    t._prepare_native_clock(.3)
                    self.assertEqual(len(configured), 1)
                # Identity plus parameter readback is not a fresh controller certificate.
                t._clock = lambda: 10.
                self.assertFalse(t._native_current_ready(.3))

    def test_only_live_preparation_sets_and_reads_back_same_incarnation(self):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from tools.data_factory.rollout.gripper_evidence import LIVE_FIELDS, RESOURCE
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        binding = {"schema_version":"fr5.gripper_source_clock.v1", "incarnation":[1,2,3,4],
                   "calendar_to_system_offset_s":0., "uncertainty_s":.001,
                   "system_anchor_s":10., "steady_anchor_s":10., "valid_until_system_s":20.}
        for mode in ("live", "plan_only", "wrong_incarnation", "legacy", "readback_mismatch"):
            with self.subTest(mode=mode):
                t = object.__new__(RosMoveItTransport)
                t._gripper_source_clock=binding
                t._allow_clock_configuration=mode!="plan_only"
                t._native_clock_configured_age=None
                t.preflight_timeout_s=t.graph_timeout_s=.1
                t.node=SimpleNamespace(get_name=lambda: "fr5_pickup_plan_only" if mode=="plan_only" else "fr5_pickup_live")
                w=dict.fromkeys(LIVE_FIELDS,0.)
                w.update(version=1. if mode=="legacy" else 2., incarnation_0=99. if mode=="wrong_incarnation" else 1.,
                         incarnation_1=2.,incarnation_2=3.,incarnation_3=4.)
                t._gripper_hardware_state=DynamicJointState(joint_names=[RESOURCE],interface_values=[InterfaceValue(
                    interface_names=list(LIVE_FIELDS),values=[w[k] for k in LIVE_FIELDS])])
                t._gripper_hardware_received_at=10.
                configured=[]
                def set_parameters(parameters):
                    configured.append(parameters[0].value)
                    return SimpleNamespace(result=SimpleNamespace(successful=True))
                client=SimpleNamespace(wait_for_services=lambda **_:True, set_parameters_atomically=set_parameters,
                    get_parameters=lambda _:SimpleNamespace(values=[SimpleNamespace(double_array_value=
                        [] if mode=="readback_mismatch" else configured[-1])]))
                t._AsyncParameterClient=lambda node,name:client
                t._wait=lambda result,*_:result
                if mode in ("wrong_incarnation","legacy","readback_mismatch"):
                    with self.assertRaises(ContractError):t._prepare_native_clock(.3)
                else:t._prepare_native_clock(.3)
                self.assertEqual(len(configured),int(mode in ("live","readback_mismatch")))
                if mode=="live":
                    t._prepare_native_clock(.3)
                    self.assertEqual(len(configured),1)


class TestExecutorActuatorStreamDrain(unittest.TestCase):
    class Transport:
        def __init__(self, *, terminal_ready=None, fail_first_poll=False, feedback_while_pending=False):
            self.active = True
            self.fenced = False
            self.owns_goals = True
            self.terminal_ready = terminal_ready
            self.fail_first_poll = fail_first_poll
            self.feedback_while_pending = feedback_while_pending
            self.status_error = False
            self.fence_calls = 0
            self.poll_calls = 0
            self.close_calls = 0
            self.cancel_calls = 0

        @property
        def owns_active_goal(self):
            return self.active

        def learned_actuator_stream_status(self):
            if self.status_error:
                raise ContractError("STREAM_STATUS_UNAVAILABLE")
            return {
                "active": self.active,
                "fenced": self.fenced if self.active else False,
                "owns_goals": self.owns_goals if self.active else False,
            }

        def fence_learned_actuator_stream(self):
            self.fence_calls += 1
            self.fenced = True

        def poll_learned_actuator_stream(self):
            self.poll_calls += 1
            if self.fail_first_poll and self.poll_calls == 1:
                error = ContractError("NATIVE_PRIMARY")
                error.actuator_stream_events = [{
                    "actuator": "arm", "revision": "rev-1", "event": "CANCEL_RESPONSE",
                    "return_code": 0, "goal_id": [1], "accepted": True,
                }]
                raise error
            if self.terminal_ready is not None and not self.terminal_ready.is_set():
                if self.feedback_while_pending:
                    return [{
                        "actuator": "arm", "revision": "rev-1", "event": "FEEDBACK",
                        "goal_id": [1], "received_monotonic_s": self.poll_calls,
                        "samples_since_poll": 2, "feedback": {"desired": {"time_from_start": {"sec": 0}}},
                    }]
                return []
            self.owns_goals = False
            return [{
                "actuator": "gripper", "revision": "rev-1", "event": "TERMINAL",
                "result_status": 4, "error_code": 0,
            }]

        def close_learned_actuator_stream(self):
            if not self.fenced or self.owns_goals:
                raise ContractError("ROS_EXEC_ACTIVE")
            self.close_calls += 1
            self.active = False

        def cancel_active(self, _timeout):
            self.cancel_calls += 1
            raise AssertionError("stream fault must use the pair fence")

        def snapshot(self, _age):
            return {}

    @staticmethod
    def executor_run(transport):
        import threading
        from tools.data_factory.motion.pickup_executor import PickupExecutor

        executor = PickupExecutor(
            transport=transport, source_clock=lambda: 10., monotonic_clock=lambda: 10.,
        )
        run = {
            "state": "EXECUTING", "digest": "sha256:" + "1" * 64,
            "cancel_event": threading.Event(),
            "plan": {
                "run_id": "stream-run", "steps": [],
                "scene_binding": {"object_instance_id": "cube"},
                "planning": {"max_joint_state_age_s": .1},
                "execution_timeouts_s": {"cancel": .1},
            },
            "execution": {
                "step_index": 0, "lease_deadline": 20., "active": True,
                "motion_dispatch_attempted": True, "terminal_phases": [],
            },
        }
        executor.runs = {"stream-run": run}
        return executor, run

    def test_fault_fences_once_and_drains_exact_events_without_replacing_primary(self):
        transport = self.Transport(fail_first_poll=True)
        executor, run = self.executor_run(transport)

        executor._fault(run, "PRIMARY_FAULT")
        self.assertEqual((run["state"], run["failure_code"]), ("BLOCKED", "PRIMARY_FAULT"))
        self.assertTrue(run["execution"]["active"])
        self.assertEqual(run["execution"]["cancel_error"], "ROS_EXEC_CANCEL_UNCERTAIN")
        self.assertEqual((transport.fence_calls, transport.cancel_calls, transport.close_calls), (1, 0, 0))
        self.assertFalse(executor.close())
        executor._fault(run, "LATER_FAULT")
        self.assertEqual((run["failure_code"], transport.fence_calls), ("PRIMARY_FAULT", 1))

        executor.tick()
        self.assertEqual(run["failure_code"], "PRIMARY_FAULT")
        self.assertEqual(run["execution"]["actuator_stream_events"][0]["event"], "CANCEL_RESPONSE")
        self.assertEqual(run["execution"]["actuator_stream_drain_error"], "NATIVE_PRIMARY")
        self.assertTrue(run["execution"]["active"])

        executor.tick()
        self.assertEqual([event["event"] for event in run["execution"]["actuator_stream_events"]],
                         ["CANCEL_RESPONSE", "TERMINAL"])
        self.assertEqual((transport.fence_calls, transport.close_calls), (1, 1))
        self.assertFalse(run["execution"]["active"])
        self.assertEqual(run["execution"]["actuator_stream_drain"]["status"], "NATIVE_HANDLES_TERMINAL")
        self.assertNotIn("physical_stop", executor._execution_data(run)["actuator_stream_drain"])
        self.assertTrue(executor.close())

    def test_jsonl_eof_waits_for_owned_stream_terminal_facts(self):
        import io
        import threading
        import time
        from tools.data_factory.motion.pickup_executor import run_jsonl

        terminal_ready = threading.Event()
        transport = self.Transport(terminal_ready=terminal_ready)
        executor, run = self.executor_run(transport)
        output, result = io.StringIO(), []
        worker = threading.Thread(target=lambda: result.append(run_jsonl(io.StringIO(), output, executor)))
        worker.start()
        time.sleep(.08)
        self.assertTrue(worker.is_alive())
        self.assertEqual((transport.fence_calls, transport.close_calls), (1, 0))
        self.assertEqual(run["failure_code"], "INPUT_EOF")

        terminal_ready.set()
        worker.join(1.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [False])
        self.assertEqual((transport.fence_calls, transport.close_calls), (1, 1))
        response = __import__("json").loads(output.getvalue())
        self.assertEqual((response["state"], response["data"]["actuator_stream_events"][0]["event"]),
                         ("BLOCKED", "TERMINAL"))

    def test_missing_stream_status_never_releases_ambiguous_owner(self):
        transport = self.Transport()
        executor, run = self.executor_run(transport)
        executor._fault(run, "PRIMARY_FAULT")
        transport.active = False
        transport.owns_goals = False

        executor.tick()
        self.assertTrue(run["execution"]["active"])
        self.assertTrue(run["execution"]["actuator_stream_stop_pending"])
        self.assertEqual(run["execution"]["actuator_stream_drain"]["status"], "OWNERSHIP_UNAVAILABLE")
        self.assertEqual(transport.close_calls, 0)
        self.assertFalse(executor.close())

        class MissingStatus:
            def poll_learned_actuator_stream(self):
                return []

            def close_learned_actuator_stream(self):
                raise AssertionError("missing ownership status cannot authorize close")

        executor.transport = MissingStatus()
        executor.tick()
        self.assertTrue(run["execution"]["active"])
        self.assertTrue(run["execution"]["actuator_stream_stop_pending"])
        self.assertEqual(run["execution"]["actuator_stream_drain"]["status"], "OWNERSHIP_UNAVAILABLE")

    def test_fault_status_error_preserves_ambiguous_owner_without_generic_cancel(self):
        transport = self.Transport()
        transport.status_error = True
        executor, run = self.executor_run(transport)

        executor._fault(run, "PRIMARY_FAULT")
        self.assertEqual((run["failure_code"], run["execution"]["active"]), ("PRIMARY_FAULT", True))
        self.assertTrue(run["execution"]["actuator_stream_stop_pending"])
        self.assertEqual(run["execution"]["actuator_stream_drain"]["status"], "OWNERSHIP_UNAVAILABLE")
        self.assertEqual((transport.fence_calls, transport.cancel_calls), (1, 0))

    def test_blocked_pending_stream_eof_still_waits_for_terminal_facts(self):
        import io
        import threading
        import time
        from tools.data_factory.motion.pickup_executor import run_jsonl

        terminal_ready = threading.Event()
        transport = self.Transport(terminal_ready=terminal_ready)
        executor, run = self.executor_run(transport)
        executor._fault(run, "PRIMARY_FAULT")
        output, result = io.StringIO(), []
        worker = threading.Thread(target=lambda: result.append(run_jsonl(io.StringIO(), output, executor)))
        worker.start()
        time.sleep(.08)
        self.assertTrue(worker.is_alive())
        terminal_ready.set()
        worker.join(1.)
        self.assertEqual((worker.is_alive(), result), (False, [False]))
        self.assertEqual(__import__("json").loads(output.getvalue())["data"]["actuator_stream_events"][0]["event"],
                         "TERMINAL")

    def test_after_line_tick_fault_cannot_exit_with_owned_stream(self):
        import io
        import threading
        import time
        from tools.data_factory.motion.pickup_executor import run_jsonl

        terminal_ready = threading.Event()
        transport = self.Transport(terminal_ready=terminal_ready)
        executor, run = self.executor_run(transport)
        run["execution"]["lease_deadline"] = 5.
        output, result = io.StringIO(), []
        worker = threading.Thread(target=lambda: result.append(run_jsonl(io.StringIO("{}\n"), output, executor)))
        worker.start()
        time.sleep(.08)
        self.assertTrue(worker.is_alive())
        self.assertEqual(run["failure_code"], "HEARTBEAT_TIMEOUT")
        terminal_ready.set()
        worker.join(1.)
        self.assertEqual((worker.is_alive(), result), (False, [False]))
        self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_pending_feedback_is_bounded_to_latest_identity_with_coalesced_counts(self):
        import threading

        terminal_ready = threading.Event()
        transport = self.Transport(terminal_ready=terminal_ready, feedback_while_pending=True)
        executor, run = self.executor_run(transport)
        executor._fault(run, "PRIMARY_FAULT")
        for _ in range(20):
            executor.tick()
        feedback = run["execution"]["actuator_stream_feedback"]
        self.assertEqual(len(feedback), 1)
        self.assertEqual((feedback[0]["coalesced_event_count"], feedback[0]["coalesced_sample_count"]),
                         (20, 40))
        self.assertEqual(feedback[0]["latest_event"]["received_monotonic_s"], 20)
        self.assertNotIn("FEEDBACK", [event["event"] for event in run["execution"].get("actuator_stream_events", [])])
        terminal_ready.set()
        executor.tick()
        self.assertEqual(run["execution"]["actuator_stream_events"][0]["event"], "TERMINAL")
