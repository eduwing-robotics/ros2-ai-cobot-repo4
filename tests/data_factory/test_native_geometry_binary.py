"""Installed CPU geometry binary, actual CDR; no ROS initialization or devices."""
import copy
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
import unittest

from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject, CollisionObject,
                             ContactInformation, LinkPadding, LinkScale, PlanningScene, RobotState)
from moveit_msgs.srv import GetStateValidity
from rclpy.serialization import deserialize_message, serialize_message
from shape_msgs.msg import SolidPrimitive


JOINTS = [f"j{i}" for i in range(1, 7)] + ["finger_right_joint"]
XML = '<robot name="cpu"><link name="base_link"/>'
for index in range(1, 7):
    parent = "base_link" if index == 1 else f"link{index-1}"
    child = "gripper_link" if index == 6 else f"link{index}"
    XML += (f'<link name="{child}"/><joint name="j{index}" type="revolute"><parent link="{parent}"/>'
            f'<child link="{child}"/><origin xyz="0 0 {0.2 if index == 6 else 0}"/><axis xyz="0 0 1"/>'
            '<limit lower="-3" upper="3" velocity="1" effort="20"/></joint>')
XML += ('<link name="finger_tip_right_link"><collision><geometry><box size="0.02 0.02 0.02"/></geometry></collision></link>'
        '<joint name="finger_right_joint" type="prismatic"><parent link="gripper_link"/><child link="finger_tip_right_link"/>'
        '<axis xyz="1 0 0"/><limit lower="0" upper="0.021" effort="20" velocity="0.1"/></joint></robot>')
SRDF = '<robot name="cpu"/>'


def cdr(message):
    return serialize_message(message).hex()


def state():
    result = RobotState(is_diff=True)
    result.joint_state.name = JOINTS[:]
    result.joint_state.position = [0.] * 7
    return result


def box(name, position, dimensions=(.024, .024, .024), frame="base_link"):
    result = CollisionObject(id=name, operation=CollisionObject.ADD)
    result.header.frame_id = frame
    result.pose.orientation.w = 1.
    result.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(dimensions))]
    p = Pose()
    p.position.x, p.position.y, p.position.z = position
    p.orientation.w = 1.
    result.primitive_poses = [p]
    return result


def scene():
    result = PlanningScene(is_diff=False, robot_model_name="cpu")
    result.robot_state = state()
    result.robot_state.is_diff = False
    result.world.collision_objects = [box("floor", (0., 0., -.02), (2., 2., .04)),
                                    box("wall", (0., -.5, 0.), (2., .05, 2.)),
                                    box("source", (.3, 0., .012))]
    return result


class NativeGeometryBinaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("FR5_NATIVE_GEOMETRY_BINARY")
        if configured:
            cls.binary = Path(configured)
        else:
            try:
                from ament_index_python.packages import get_package_prefix
                cls.binary = Path(get_package_prefix("fr5_motion_geometry")) / "lib/fr5_motion_geometry/fr5_native_geometry"
            except LookupError:
                raise unittest.SkipTest("Build fr5_motion_geometry or set FR5_NATIVE_GEOMETRY_BINARY to its installed executable")
        if not cls.binary.is_file():
            raise AssertionError(f"missing configured geometry binary: {cls.binary}")

    def setUp(self):
        self.stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen([str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1)
        self.sequence = 0
        self.addCleanup(self.close)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
            self.fail("geometry process failed to exit on EOF")
        finally:
            self.process.stdout.close()
            self.stderr.close()

    def exchange(self, request, *, raw=False):
        self.process.stdin.write((request if raw else json.dumps(request, allow_nan=False)) + "\n")
        self.process.stdin.flush()
        self.assertTrue(select.select([self.process.stdout], [], [], 5)[0], "geometry response timed out")
        line = self.process.stdout.readline()
        self.assertTrue(line, f"geometry exited: {self.process.poll()}")
        response = json.loads(line)
        if not raw:
            self.assertEqual(response["id"], request["id"])
            self.assertEqual(response["op"], request["op"])
        return response

    def init(self, value=None):
        self.sequence += 1
        return self.exchange({"op": "init", "id": self.sequence, "urdf": XML, "srdf": SRDF,
                              "scene_cdr_hex": cdr(value if value is not None else scene())})

    def variant(self, *, hypothesis="source", worlds=(), states=None):
        return {"hypothesis": hypothesis, "world_objects_cdr_hex": [cdr(v) for v in worlds],
                "states_cdr_hex": [cdr(v) for v in (states if states is not None else [state()])]}

    def query(self, variants=None):
        self.sequence += 1
        return self.exchange({"op": "query", "id": self.sequence,
                              "variants": variants if variants is not None else [self.variant()]})

    def sample(self, result, variant=0, sample=0):
        self.assertTrue(result["ok"], result)
        value = result["variants"][variant]["samples"][sample]
        self.assertEqual(set(value), {"response_cdr_hex", "saturated", "gripper_pose"})
        self.assertFalse(value["saturated"])
        return deserialize_message(bytes.fromhex(value["response_cdr_hex"]), GetStateValidity.Response)

    def test_private_released_world_removes_only_artificial_floor_contact(self):
        self.assertTrue(self.init()["ok"])
        released = box("released", (0., 0., .011))  # 1 mm conservative floor overlap.
        attached = state()
        attached.attached_collision_objects = [AttachedCollisionObject(link_name="base_link", object=released)]
        old = self.sample(self.query([self.variant(hypothesis="released", states=[attached])]))
        self.assertFalse(old.valid)
        self.assertTrue(any({c.contact_body_1, c.contact_body_2} == {"released", "floor"} for c in old.contacts))
        result = self.query([self.variant(), self.variant(hypothesis="carried"),
                             self.variant(hypothesis="released", worlds=[released])])
        for index in range(3):
            self.assertTrue(self.sample(result, variant=index).valid)
        # World additions are request-local, not silently retained or replacing facts.
        self.assertTrue(self.sample(self.query([self.variant(worlds=[released])])).valid)
        self.assertTrue(self.sample(self.query()).valid)

    def test_world_cube_top_contact_is_native_and_body_types_are_ros_values(self):
        self.assertTrue(self.init()["ok"])
        top = box("released", (0., 0., .179))
        result = self.query([self.variant(hypothesis="released", worlds=[top])])
        response = self.sample(result)
        self.assertFalse(response.valid)
        contacts = [c for c in response.contacts if {c.contact_body_1, c.contact_body_2} == {"released", "finger_tip_right_link"}]
        self.assertTrue(contacts)
        for contact in contacts:
            self.assertEqual(contact.header.frame_id, "base_link")
            self.assertEqual({(contact.contact_body_1, contact.body_type_1), (contact.contact_body_2, contact.body_type_2)},
                             {("released", ContactInformation.WORLD_OBJECT), ("finger_tip_right_link", ContactInformation.ROBOT_LINK)})
            self.assertAlmostEqual(abs(contact.normal.z), 1.)
            self.assertAlmostEqual(contact.depth, .001)
        self.assertEqual(result["variants"][0]["samples"][0]["gripper_pose"]["translation_m"], [0., 0., .2])
        self.assertTrue(self.sample(self.query()).valid)

    def test_existing_attachment_cannot_be_removed_or_replaced(self):
        initial = scene()
        held = box("held", (.3, 0., 0.), frame="gripper_link")
        initial.robot_state.attached_collision_objects = [AttachedCollisionObject(link_name="gripper_link", object=held)]
        initial.world.collision_objects.append(box("obstacle", (.3, 0., .2)))
        self.assertTrue(self.init(initial)["ok"])
        response = self.sample(self.query())
        self.assertTrue(any({c.contact_body_1, c.contact_body_2} == {"held", "obstacle"} for c in response.contacts))
        for operation in (CollisionObject.ADD, CollisionObject.REMOVE, CollisionObject.MOVE, CollisionObject.APPEND):
            changed = state()
            body = copy.deepcopy(initial.robot_state.attached_collision_objects[0])
            body.object.operation = operation
            changed.attached_collision_objects = [body]
            result = self.query([self.variant(states=[changed])])
            self.assertFalse(result["ok"], (operation, result))
            self.assertNotIn("variants", result)
        self.assertFalse(self.sample(self.query()).valid)

    def test_samples_do_not_inherit_prior_sample_attachments(self):
        self.assertTrue(self.init()["ok"])
        first = state()
        first.attached_collision_objects = [AttachedCollisionObject(link_name="base_link", object=box("held", (0., 0., .011)))]
        result = self.query([self.variant(hypothesis="carried", states=[first, state()])])
        self.assertFalse(self.sample(result, sample=0).valid)
        self.assertTrue(self.sample(result, sample=1).valid)

    def test_init_rejects_malformed_full_scene_without_acquiring_owner(self):
        for change in ("diff", "missing_joints", "shape", "acm", "padding", "subframes"):
            initial = scene()
            if change == "diff": initial.is_diff = True
            elif change == "missing_joints": initial.robot_state.joint_state.name = []; initial.robot_state.joint_state.position = []
            elif change == "shape": initial.world.collision_objects[0].primitives[0].dimensions[0] = -1.
            elif change == "acm": initial.allowed_collision_matrix.entry_names = ["floor"]
            elif change == "padding": initial.link_padding = [LinkPadding(link_name="unknown", padding=.01)]
            else: initial.world.collision_objects[0].subframe_names = ["missing_pose"]
            with self.subTest(change=change):
                response = self.init(initial)
                self.assertFalse(response["ok"], response)
        request = {"op": "init", "id": 50, "urdf": XML.replace('size="0.02 0.02 0.02"', 'size="-1 0.02 0.02"'),
                   "srdf": SRDF, "scene_cdr_hex": cdr(scene())}
        self.assertEqual(self.exchange(request)["error"], "MODEL_SHAPES")
        self.assertTrue(self.init()["ok"])

    def test_query_errors_do_not_modify_baseline_or_publish_partial_variants(self):
        self.assertTrue(self.init()["ok"])
        for change in ("replace", "remove", "bad_frame", "nan", "shape", "state_diff", "unknown_joint", "missing_joint", "attachment_frame"):
            variant = self.variant()
            if change in ("replace", "remove", "bad_frame", "nan", "shape"):
                item = box("floor" if change == "replace" else "additional", (1., 0., 0.))
                if change == "remove": item.operation = CollisionObject.REMOVE
                if change == "bad_frame": item.header.frame_id = "unknown"
                if change == "nan": item.primitive_poses[0].position.x = float("nan")
                if change == "shape": item.primitives[0].dimensions = []
                variant["world_objects_cdr_hex"] = [cdr(item)]
            else:
                item = state()
                if change == "state_diff": item.is_diff = False
                elif change == "unknown_joint": item.joint_state.name[0] = "unknown"
                elif change == "missing_joint": item.joint_state.name.pop(); item.joint_state.position.pop()
                else: item.attached_collision_objects = [AttachedCollisionObject(link_name="gripper_link", object=box("held", (0., 0., 0.)))]
                variant["states_cdr_hex"] = [cdr(item)]
            result = self.query([self.variant(hypothesis="carried"), variant])
            with self.subTest(change=change):
                self.assertFalse(result["ok"], result)
                self.assertNotIn("variants", result)
                self.assertTrue(self.sample(self.query()).valid)

    def test_full_scene_acm_padding_and_scaling_are_preserved(self):
        initial = scene()
        initial.world.collision_objects.append(box("touch", (0., 0., .2)))
        initial.allowed_collision_matrix.entry_names = ["touch", "finger_tip_right_link"]
        initial.allowed_collision_matrix.entry_values = [AllowedCollisionEntry(enabled=[False, True]), AllowedCollisionEntry(enabled=[True, False])]
        initial.link_padding = [LinkPadding(link_name="finger_tip_right_link", padding=.01)]
        initial.link_scale = [LinkScale(link_name="finger_tip_right_link", scale=1.1)]
        self.assertTrue(self.init(initial)["ok"])
        self.assertTrue(self.sample(self.query()).valid)  # Full-scene ACM retained.
        near = box("padding_probe", (.029, 0., .2), (.02, .02, .02))
        response = self.sample(self.query([self.variant(worlds=[near])]))
        self.assertFalse(response.valid)  # Actual padded/scaled robot geometry.
        self.assertTrue(any("padding_probe" in (c.contact_body_1, c.contact_body_2) for c in response.contacts))

    def test_full_scene_fixed_frame_is_preserved_for_private_world_geometry(self):
        initial = scene()
        transform = TransformStamped(child_frame_id="base_link")
        transform.header.frame_id = "fixture_frame"
        transform.transform.rotation.w = 1.
        transform.transform.translation.z = .179
        initial.fixed_frame_transforms = [transform]
        self.assertTrue(self.init(initial)["ok"])
        response = self.sample(self.query([self.variant(worlds=[box("top", (0., 0., 0.), frame="fixture_frame")])]))
        self.assertFalse(response.valid)
        self.assertTrue(any({c.contact_body_1, c.contact_body_2} == {"top", "finger_tip_right_link"} for c in response.contacts))

    def test_malformed_cdr_and_json_are_rejected_and_init_is_single_use(self):
        self.assertFalse(self.query()["ok"])
        malformed = {"op": "init", "id": 20, "urdf": XML, "srdf": SRDF, "scene_cdr_hex": "00010000"}
        self.assertFalse(self.exchange(malformed)["ok"])
        self.assertTrue(self.init()["ok"])
        self.assertEqual(self.init()["error"], "ALREADY_INITIALIZED")
        for payload in ("zz", "0", "00010000", cdr(state()) + "00"):
            variant = self.variant()
            variant["states_cdr_hex"] = [payload]
            self.assertFalse(self.query([variant])["ok"])
        for raw in ('{"op":"query","op":"init","id":1}', '{', '[]', '{"op":"query","id":true,"variants":[]}'):
            self.assertFalse(self.exchange(raw, raw=True)["ok"])
        self.assertTrue(self.sample(self.query()).valid)

    def test_unknown_keys_duplicate_hypotheses_and_oversize_line_are_rejected(self):
        self.assertTrue(self.init()["ok"])
        variant = self.variant()
        variant["allowed_collision_matrix"] = {}
        self.assertFalse(self.query([variant])["ok"])
        self.assertFalse(self.query([self.variant(), self.variant()])["ok"])
        result = self.exchange(" " * (16 * 1024 * 1024 + 1), raw=True)
        self.assertEqual(result, {"op": "", "id": 0, "ok": False, "error": "PAYLOAD_LIMIT"})
        self.assertTrue(self.sample(self.query()).valid)

    def test_actual_adapter_initializes_queries_and_closes(self):
        from tools.data_factory.motion.native_geometry import NativeGeometry
        from tools.fr5_data_factory import canonical_digest
        plan = {"fixture": "native-geometry-ipc-only"}
        context = {"schema_version": "data_factory.request_contact_geometry.v1", "status": "PROSPECTIVE",
                   "physical_success": False, "plan_digest": canonical_digest(plan),
                   "source_object_id": "source", "planning_frame": "base_link"}
        context["geometry_digest"] = canonical_digest(context)
        adapter = NativeGeometry(urdf=XML, srdf=SRDF, scene=scene(), context=context, plan=plan,
                                 deadline=time.monotonic()+5., command=[str(self.binary)])
        try:
            def ready(**kwargs):
                deadline = time.monotonic()+5.
                while time.monotonic() < deadline:
                    result = adapter.poll(**kwargs)
                    if result is not None:
                        return result
                    time.sleep(.001)
                self.fail("adapter query did not finish")
            self.assertEqual(ready()["status"], "READY")
            samples = (("source", tuple((index*.001, 0., 0., 0., 0., 0., .01) for index in range(7))),)
            binding = canonical_digest(samples)
            adapter.submit(samples, binding=binding, deadline=time.monotonic()+5.)
            result = ready(binding=binding)
            self.assertEqual((result["status"], result["binding"]), ("CHECKED", binding))
            self.assertEqual(len(result["variants"][0]["samples"]), 7)
            self.assertTrue(all(item["allowed"] for item in result["variants"][0]["samples"]))
            self.assertIsNone(adapter.poll())
        finally:
            deadline = time.monotonic()+5.
            while not adapter.close():
                if time.monotonic() >= deadline:
                    self.fail("adapter failed to close its actual CPU process")
                time.sleep(.001)
        self.assertIsNotNone(adapter._process.poll())

    def test_actual_native_query_binds_the_pair_consumed_by_mock_actuators(self):
        from concurrent.futures import Future
        from types import SimpleNamespace
        from unittest.mock import Mock
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        from tools.data_factory.motion.native_geometry import NativeGeometry
        from tools.fr5_data_factory import canonical_digest

        plan = {"fixture": "native-pair-query-only"}
        context = {"schema_version": "data_factory.request_contact_geometry.v1", "status": "PROSPECTIVE",
                   "physical_success": False, "plan_digest": canonical_digest(plan),
                   "source_object_id": "source", "planning_frame": "base_link"}
        context["geometry_digest"] = canonical_digest(context)
        deadline = time.monotonic() + 5.
        adapter = NativeGeometry(urdf=XML, srdf=SRDF, scene=scene(), context=context, plan=plan,
                                 deadline=deadline, command=[str(self.binary)])
        transport = object.__new__(RosMoveItTransport)
        transport._active, transport._execution_locked = None, False
        transport._clock = time.monotonic
        transport._execute_goal_count = transport._gripper_goal_count = 0
        transport._FollowJointTrajectory, transport._JointTrajectoryPoint = FollowJointTrajectory, JointTrajectoryPoint
        arm_client = Mock()
        arm_client.send_goal_async.return_value = Future()
        transport.gripper = Mock()
        transport.gripper.send_goal_async.return_value = Future()
        transport._ActionClient = Mock(return_value=arm_client)
        transport.node = object()
        transport._rclpy = SimpleNamespace(spin_once=Mock())
        transport.open_learned_actuator_stream(deadline=deadline)
        try:
            def wait_for(poll):
                while time.monotonic() < deadline:
                    result = poll()
                    if result is not None:
                        return result
                    # Test driver only; production owner never waits here.
                    time.sleep(.001)
                self.fail("native query did not finish")
            transport._native_geometry_ready = wait_for(adapter.poll)
            transport._native_geometry = adapter
            transport._native_geometry_context, transport._native_geometry_plan = context, plan
            transport._native_geometry_deadline = deadline
            pair = transport.build_learned_actuator_goals([0.] * 6 + [.021],
                [[.001] * 6 + [.012], [.002] * 6 + [.018]], period_s=.1)
            scene_binding = {"scene_state_digest": canonical_digest("fixture-scene"), "revision": 1}
            binding = transport.prepare_learned_revision_geometry(*pair, revision="cpu-revision",
                scene_binding=scene_binding, assignments=(("source", tuple(range(13))),), deadline=deadline)
            checked = wait_for(lambda: transport.poll_learned_revision_geometry(binding=binding))
            self.assertTrue(checked["allowed"])
            self.assertEqual(checked["sample_count"], 13)
            arm_client.send_goal_async.assert_not_called()
            guard = Mock()  # Test-only: no current hardware/Scene authority is asserted.
            transport.submit_checked_learned_revision(binding=binding, start_time_ns=42_000_000_000,
                scene_binding=scene_binding, dispatch_guard=guard)
            for item in pair: item.trajectory.header.stamp.sec = 42
            self.assertEqual(arm_client.send_goal_async.call_args.args[0], pair[0])
            self.assertEqual(transport.gripper.send_goal_async.call_args.args[0], pair[1])
            self.assertEqual(transport.poll_learned_actuator_stream(), [])
            self.assertEqual((transport._execute_goal_count, transport._gripper_goal_count), (1, 1))
        finally:
            limit = time.monotonic()+5.
            while not adapter.close():
                if time.monotonic() >= limit:
                    self.fail("CPU geometry child failed to close")
                time.sleep(.001)

    def test_native_default_revision_derives_intent_and_queries_expanded_geometry_without_sends(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from tools.data_factory.motion.contact_transition import TIPS
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        from tools.data_factory.motion.native_geometry import NativeGeometry
        from tools.fr5_data_factory import canonical_digest

        # Same small native model, now with both explicitly modeled moving jaws.
        xml = XML.replace('<axis xyz="1 0 0"/>', '<origin xyz=".005 0 0"/><axis xyz="1 0 0"/>')
        xml = xml.replace('</robot>', '<link name="finger_tip_left_link"><collision><geometry>'
            '<box size="0.02 0.02 0.02"/></geometry></collision></link>'
            '<joint name="finger_left_joint" type="prismatic"><parent link="gripper_link"/>'
            '<child link="finger_tip_left_link"/><origin xyz="-.005 0 0"/><axis xyz="-1 0 0"/>'
            '<limit lower="0" upper="0.021" effort="20" velocity="0.1"/>'
            '<mimic joint="finger_right_joint" multiplier="1" offset="0"/></joint></robot>')
        initial = scene()
        initial.world.collision_objects[-1] = box("source", (0., .004, .2))
        original_scene = copy.deepcopy(initial)
        plan = {"fixture": "native-derived-intent-only"}
        datum = {"translation_m": [0., .004, .2], "rotation_columns": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
        context = {"schema_version": "data_factory.request_contact_geometry.v1", "status": "PROSPECTIVE",
            "physical_success": False, "plan_digest": canonical_digest(plan), "source_object_id": "source",
            "planning_frame": "base_link", "source_datum": datum, "released_datum": copy.deepcopy(datum),
            "source_dimensions_m": [.024] * 3, "open_m": .021, "release_m": .0126,
            "jaw_midplane_m": 0., "closed_gap_bound_m": .01436, "orientation_tolerance_rad": .1,
            "fingertip_boxes": [{"center": [.005, 0., 0.], "dimensions": [.02] * 3},
                                 {"center": [-.005, 0., 0.], "dimensions": [.02] * 3}], "proxies": {}}
        for hypothesis, link, touch, position in (("carried", "gripper_link", TIPS, [0., 0., 0.]),
                                                  ("released", "base_link", [], datum["translation_m"])):
            context["proxies"][hypothesis] = {"object_id": "source::prospective:" + hypothesis,
                "link_name": link, "touch_links": touch[:], "translation_m": position[:],
                "rotation_xyzw": [0., 0., 0., 1.], "dimensions_m": [.024] * 3}
        context["geometry_digest"] = canonical_digest(context)
        original_context = copy.deepcopy(context)
        deadline = time.monotonic() + 5.
        adapter = NativeGeometry(urdf=xml, srdf=SRDF, scene=initial, context=context, plan=plan,
                                 deadline=deadline, command=[str(self.binary)])
        transport = object.__new__(RosMoveItTransport)
        transport._active, transport._execution_locked = None, False
        transport._clock = time.monotonic
        transport._execute_goal_count = transport._gripper_goal_count = 0
        transport._FollowJointTrajectory, transport._JointTrajectoryPoint = FollowJointTrajectory, JointTrajectoryPoint
        arm, gripper = Mock(), Mock()
        transport._ActionClient, transport.gripper = Mock(return_value=arm), gripper
        transport.node = object()
        transport._rclpy = SimpleNamespace(spin_once=Mock())
        transport.open_learned_actuator_stream(deadline=deadline)
        try:
            def wait_for(poll):
                while time.monotonic() < deadline:
                    result = poll()
                    if result is not None:
                        return result
                    time.sleep(.001)  # Bounded test harness, not owner code.
                self.fail("native derived geometry did not finish")

            transport._native_geometry_ready = wait_for(adapter.poll)
            transport._native_geometry = adapter
            transport._native_geometry_context, transport._native_geometry_plan = context, plan
            transport._native_geometry_deadline = deadline
            calls, exchange = [], adapter._exchange

            def record(request):
                calls.append(copy.deepcopy(request))
                return exchange(request)

            adapter._exchange = record
            pair = transport.build_learned_actuator_goals([0.] * 6 + [.021],
                [[.001] * 6 + [.01176], [.002] * 6 + [.021]], period_s=.1)
            scene_binding = {"scene_state_digest": canonical_digest("fixture-scene"), "revision": 1}
            binding = transport.prepare_learned_revision_geometry(*pair, revision="derived-cpu-revision",
                scene_binding=scene_binding, deadline=deadline)  # Actual omitted-assignment path.
            checked = wait_for(lambda: transport.poll_learned_revision_geometry(binding=binding))
            self.assertTrue(checked["allowed"], checked)
            self.assertEqual(checked["sample_count"], 13)
            self.assertEqual(checked["reference_intent"]["intent"]["geometry_digest"], context["geometry_digest"])
            self.assertAlmostEqual(checked["reference_intent"]["evidence"]["candidate_relations"][0]["source_to_gripper"]["translation_m"][2], 0.)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["variants"][0]["hypothesis"], "source")
            carried = next(v for v in calls[1]["variants"] if v["hypothesis"] == "carried")
            body = deserialize_message(bytes.fromhex(carried["states_cdr_hex"][0]), RobotState).attached_collision_objects[0]
            self.assertGreater(body.object.primitive_poses[0].position.y, .003)
            self.assertGreaterEqual(body.object.primitives[0].dimensions[1], .024)
            self.assertTrue(any(v["hypothesis"] == "released" and v["world_objects_cdr_hex"] for v in calls[1]["variants"]))
            self.assertEqual(initial, original_scene)  # CDR padding itself is not canonical.
            self.assertEqual(adapter._context, original_context)
            retained = copy.deepcopy(checked)
            checked["reference_intent"]["intent"]["carried_envelope"]["translation_m"][1] = 99.
            self.assertEqual(transport.poll_learned_revision_geometry(binding=binding), retained)
            self.assertEqual(len(calls), 2)
            arm.send_goal_async.assert_not_called()
            gripper.send_goal_async.assert_not_called()
            self.assertEqual((transport._execute_goal_count, transport._gripper_goal_count), (0, 0))
        finally:
            limit = time.monotonic() + 5.
            while not adapter.close():
                if time.monotonic() >= limit:
                    self.fail("native derived geometry child failed to close")
                time.sleep(.001)


if __name__ == "__main__":
    unittest.main()
