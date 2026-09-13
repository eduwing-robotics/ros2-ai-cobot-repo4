"""Qualified request-local models only: no node, service, or physical outcome."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from moveit_msgs.msg import ContactInformation
from rclpy.serialization import deserialize_message, serialize_message
from moveit_msgs.srv import GetStateValidity

from tools.data_factory.run_job import resolve_inputs
from tools.data_factory.motion.contact_transition import (
    TIPS, bind_native_request_geometry, bind_request_contacts, intended_request_contact,
    prepare_request_geometry, request_geometry,
)
from tools.fr5_data_factory import ContractError, TASK_CONTRACTS, canonical_digest, compose_rigid_transform


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/data_factory"


def inputs():
    def load(path):
        return json.loads(path.read_text())
    model = ROOT / "src/fairino_description/urdf/fairino5_v6_gripper_opening_r001.urdf"
    base = load(CONFIG / "jobs/center-live-24mm-20260903-r002.job.json")
    base.update(task="pick_place", instruction="pick up the 24 mm wooden cube and place it at the destination",
                episode_intent=TASK_CONTRACTS["pick_place"]["episode_intent"])
    jobs = []
    for side, cell_id in (("a", "place-a-yaw0-r003"), ("b", "place-b-yaw0-r001")):
        cell = load(CONFIG / "cells" / (cell_id + ".json"))
        job = {**base, "place_id": cell["place_id"], "cell_calibration_id": cell_id,
               "sheet_manifest_digest": cell["yaw0_manifest_digest"]}
        sheet = str(CONFIG / "workspace_sheets" / (cell_id + "_yaw0_sheet.json"))
        qualification = CONFIG / "motion_qualifications" / (
            f"fr5-place-{side}-wood-cube-24mm-r001-demonstration-rhythm-r001-opening-coordinate-r001.json")
        jobs.append(dict(job=job, selected_sheet=sheet, yaw0_sheet=sheet, motion_qualification=str(qualification)))
    preset = load(CONFIG / "motion_presets/demonstration-rhythm-r001.json")
    payload = {**jobs[0], "destination": jobs[1], "config_root": str(CONFIG), "urdf": str(model),
        "home_candidate": str(CONFIG / "home_candidates/fr5-lab-a-tcp-r002-home-r002-opening-coordinate.json"),
        "expected_robot_system_id": base["robot_system_id"], "run_id": "cpu-contact-geometry",
        "motion_preset": {"id": preset["motion_preset_id"], "digest": canonical_digest(preset)}}
    def binding(_validated, release, _run_id):
        return {"object_instance_id": "cube", "revision": 1, "scene_state_digest": canonical_digest("scene"),
                "release_slot": {"pose": release}}
    validated, source, scene = resolve_inputs(payload, scene_binding_call=binding)
    plan = {"schema_version": "data_factory.native_learned_task_plan.v1", "run_id": payload["run_id"],
            "source_program": source, "resolved_job_digest": validated["resolved_job_digest"],
            "scene_binding": scene, "policy": {"robot_description": model.read_text()}}
    obj = {"state": "ON_SURFACE", "object_profile_id": base["object_profile_id"],
           "pose": {k: jobs[0]["job"][k] for k in ("place_id", "x_mm", "y_mm", "yaw_deg")}}
    return SimpleNamespace(_robot_description=model.read_text(), contact_config_root=CONFIG), plan, obj


class RequestContactGeometryTest(unittest.TestCase):
    def setUp(self):
        self.transport, self.plan, self.obj = inputs()
        self.context = prepare_request_geometry(self.transport, self.plan, self.obj)

    def test_normal_qualified_source_destination_and_detached_ros_requests(self):
        original = copy.deepcopy((self.plan, self.obj, self.context))
        for hypothesis in ("source", "carried", "released"):
            request = request_geometry(self.context, self.plan, [0.] * 6, .012, hypothesis=hypothesis)
            self.assertEqual(deserialize_message(serialize_message(request), GetStateValidity.Request), request)
            self.assertTrue(request.robot_state.is_diff)
            self.assertEqual(request.group_name, "")
            self.assertEqual(len(request.robot_state.joint_state.position), 7)
            attached = request.robot_state.attached_collision_objects
            self.assertEqual(len(attached), hypothesis != "source")
            if attached:
                body = attached[0]
                self.assertEqual(body.object.header.frame_id, body.link_name)
                self.assertEqual(body.link_name, "gripper_link" if hypothesis == "carried" else "base_link")
                self.assertEqual(body.touch_links, TIPS if hypothesis == "carried" else [])
                self.assertEqual(len(body.object.primitives), 1)
                self.assertEqual(len(body.object.primitive_poses), 1)
                self.assertEqual(body.object.pose.orientation.w, 1.)
                body.object.primitives[0].dimensions[0] = 9.
        self.assertEqual((self.plan, self.obj, self.context), original)
        self.assertFalse(self.context["physical_success"])
        self.assertEqual(self.context["status"], "PROSPECTIVE")
        self.assertNotEqual(self.context["proxies"]["released"]["translation_m"], self.context["source_datum"]["translation_m"])
        # No transport API is available: preparing/building cannot query or apply a Scene.
        self.assertEqual(set(vars(self.transport)), {"_robot_description", "contact_config_root"})

    def test_scene_destination_and_model_mismatch_cannot_acquire_context(self):
        for change in ("source", "destination", "model", "floor_id"):
            plan, obj = copy.deepcopy((self.plan, self.obj))
            if change == "source":
                obj["pose"]["x_mm"] += 1
            elif change == "destination":
                plan["scene_binding"]["release_slot"]["pose"]["x_mm"] += 1
            elif change == "model":
                plan["policy"]["robot_description"] += " "
            else:
                plan["scene_binding"]["object_instance_id"] = plan["source_program"]["planning_scene"]["floor"]["id"]
            with self.subTest(change=change), self.assertRaises(ContractError):
                prepare_request_geometry(self.transport, plan, obj)

    def test_native_batch_keeps_released_proxy_in_private_world(self):
        build = bind_native_request_geometry(self.context, self.plan)
        state, world = build("released", [0.] * 6, .012)
        self.assertTrue(state.is_diff)
        self.assertEqual(state.attached_collision_objects, [])
        self.assertEqual(len(world), 1)
        self.assertEqual(world[0].id, self.context["proxies"]["released"]["object_id"])
        self.assertEqual(world[0].header.frame_id, "base_link")
        old = copy.deepcopy(world)
        self.context["proxies"]["released"]["dimensions_m"][0] = 2.
        self.plan["policy"]["robot_description"] = "changed"
        world[0].id = "caller-mutated"
        self.assertEqual(build("released", [0.] * 6, .012)[1], old)
        state, world = build("carried", [0.] * 6, .012)
        self.assertEqual(world, [])
        self.assertEqual(state.attached_collision_objects[0].link_name, "gripper_link")
        self.assertEqual(state.attached_collision_objects[0].touch_links, TIPS)
        with self.assertRaisesRegex(ContractError, "CONTACT_REQUEST_BINDING"):
            build("success", [0.] * 6, .012)

    def test_full_scene_private_binding_keeps_other_obstacles_and_original_bytes(self):
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive
        from moveit_msgs.msg import CollisionObject, PlanningScene, AttachedCollisionObject
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        native = object.__new__(RosMoveItTransport)
        native._Pose, native._SolidPrimitive, native._CollisionObject = Pose, SolidPrimitive, CollisionObject
        captured = PlanningScene(is_diff=False, robot_model_name="fairino5_v6_robot")
        foreign = native._collision_object("other-obstacle", [.1] * 3, [1., 1., 1.], "base_link")
        captured.world.collision_objects = [foreign]
        original = serialize_message(captured)
        bound = native._bound_native_geometry_scene(captured, self.plan, self.context)
        self.assertEqual(serialize_message(captured), original)
        self.assertEqual(bound.world.collision_objects[0], foreign)
        self.assertEqual(len(bound.world.collision_objects), 4)
        # Native pose roundoff does not change the qualified geometry identity.
        bound.world.collision_objects[-1].primitive_poses[0].position.x += 1e-16
        self.assertEqual(native._bound_native_geometry_scene(bound, self.plan, self.context), bound)
        changed = copy.deepcopy(bound)
        changed.world.collision_objects[-1].primitive_poses[0].position.x += .01
        with self.assertRaisesRegex(ContractError, "CONTACT_SCENE_BINDING"):
            native._bound_native_geometry_scene(changed, self.plan, self.context)
        changed = copy.deepcopy(captured)
        changed.robot_state.attached_collision_objects = [AttachedCollisionObject(
            link_name="gripper_link", object=bound.world.collision_objects[-1])]
        with self.assertRaisesRegex(ContractError, "CONTACT_SCENE_BINDING"):
            native._bound_native_geometry_scene(changed, self.plan, self.context)
        changed = copy.deepcopy(captured)
        changed.allowed_collision_matrix.default_entry_names = ["cube"]
        changed.allowed_collision_matrix.default_entry_values = [True]
        with self.assertRaisesRegex(ContractError, "CONTACT_COLLISION_POLICY"):
            native._bound_native_geometry_scene(changed, self.plan, self.context)

    def test_malformed_attachment_or_changed_context_never_reaches_native_conversion(self):
        for key, value in (("dimensions_m", []), ("dimensions_m", [-.024, .024, .024]),
                           ("rotation_xyzw", [0., 0., 0., 0.]), ("translation_m", [float("inf"), 0., 0.]),
                           ("link_name", "unknown"), ("touch_links", ["wrist3_link"])):
            context = copy.deepcopy(self.context)
            context["proxies"]["carried"][key] = value
            # Even a re-digested malformed geometry cannot silently disappear
            # inside MoveIt's logging-only message conversion failure path.
            if key != "translation_m":
                context["geometry_digest"] = canonical_digest({k: v for k, v in context.items() if k != "geometry_digest"})
            with self.subTest(key=key), self.assertRaises((ContractError, ValueError)):
                request_geometry(context, self.plan, [0.] * 6, .012, hypothesis="carried")

    def contact(self, first, first_type, second, second_type):
        item = ContactInformation(contact_body_1=first, body_type_1=first_type,
                                  contact_body_2=second, body_type_2=second_type)
        item.header.frame_id = "base_link"
        item.normal.x = 1.
        return item

    def test_only_exact_duplicate_representation_is_a_bookkeeping_exemption(self):
        C = ContactInformation
        proxy = self.context["proxies"]["carried"]["object_id"]
        for name, kind, expected in (("cube", C.WORLD_OBJECT, True), ("floor", C.WORLD_OBJECT, False),
                                     ("wall", C.WORLD_OBJECT, False), ("other-cube", C.WORLD_OBJECT, False),
                                     ("wrist3_link", C.ROBOT_LINK, False), ("cube", C.ROBOT_LINK, False)):
            contact = self.contact(name, kind, proxy, C.ROBOT_ATTACHED)
            self.assertEqual(intended_request_contact(self.context, self.plan, "carried", contact), expected)
        contact = self.contact("cube", C.WORLD_OBJECT, proxy, C.ROBOT_ATTACHED)
        self.assertFalse(intended_request_contact(self.context, self.plan, "released", contact))
        contact.header.frame_id = "unknown"
        self.assertFalse(intended_request_contact(self.context, self.plan, "carried", contact))

    def test_batch_classifier_binds_owned_inputs_once_without_repeated_plan_hashes(self):
        from tools.data_factory.motion import contact_transition
        C = ContactInformation
        contact = self.contact("cube", C.WORLD_OBJECT,
                               self.context["proxies"]["carried"]["object_id"], C.ROBOT_ATTACHED)
        with mock.patch.object(contact_transition, "canonical_digest", wraps=canonical_digest) as digest:
            classify = bind_request_contacts(self.context, self.plan)
            binding_calls = digest.call_count
            self.assertGreater(binding_calls, 0)
            self.context["source_object_id"] = "changed"
            self.plan["policy"]["robot_description"] = "changed"
            for _ in range(10):
                self.assertTrue(classify("carried", contact))
                self.assertFalse(classify("released", contact))
            self.assertEqual(digest.call_count, binding_calls)
            with self.assertRaisesRegex(ContractError, "CONTACT_REQUEST_BINDING"):
                classify("detected-success", contact)
        with self.assertRaisesRegex(ContractError, "CONTACT_REQUEST_BINDING"):
            bind_request_contacts(self.context, self.plan)

    def test_top_pressing_is_not_intended_side_jaw_contact(self):
        C = ContactInformation
        source = self.plan["source_program"]
        tool = next(s["target"]["base_tool"] for s in source["steps"] if s["phase"] == "FINAL_APPROACH_LIN")
        mount = {"translation_m": [0., 0., .109], "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
        gripper = compose_rigid_transform(tool, mount)
        datum = self.context["source_datum"]
        # A point on the source cube's side, 2 mm below its top, within the
        # qualified fingertip pad. Its normal is side-facing, not downward.
        point = compose_rigid_transform(datum, {"translation_m": [-.012, 0., .010],
            "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]})["translation_m"]
        contact = self.contact("cube", C.WORLD_OBJECT, "finger_tip_left_link", C.ROBOT_LINK)
        contact.position.x, contact.position.y, contact.position.z = point
        contact.normal.x, contact.normal.y, contact.normal.z = datum["rotation_columns"][0]
        contact.depth = .003
        args = dict(gripper_pose=gripper, gripper_m=.01176)
        classify = bind_request_contacts(self.context, self.plan)
        self.assertTrue(intended_request_contact(self.context, self.plan, "source", contact, **args))
        self.assertTrue(classify("source", contact, **args))
        self.assertFalse(intended_request_contact(self.context, self.plan, "source", contact))
        self.assertFalse(classify("source", contact))
        contact.normal.x, contact.normal.y, contact.normal.z = datum["rotation_columns"][2]
        self.assertFalse(intended_request_contact(self.context, self.plan, "source", contact, **args))
        self.assertFalse(classify("source", contact, **args))
        contact.normal.x, contact.normal.y, contact.normal.z = datum["rotation_columns"][0]
        contact.position.z += .2
        self.assertFalse(intended_request_contact(self.context, self.plan, "source", contact, **args))
        self.assertFalse(classify("source", contact, **args))

    def test_released_cube_does_not_hide_top_collision_behind_native_touch_links(self):
        from tools.fr5_data_factory import inverse_rigid_transform
        C = ContactInformation
        request = request_geometry(self.context, self.plan, [0.] * 6, .012, hypothesis="released")
        self.assertEqual(request.robot_state.attached_collision_objects[0].touch_links, [])
        source = self.plan["source_program"]
        tool = next(s["target"]["base_tool"] for s in source["steps"] if s["phase"] == "FINAL_APPROACH_LIN")
        mount = {"translation_m": [0., 0., .109], "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
        relation = compose_rigid_transform(inverse_rigid_transform(self.context["source_datum"]),
                                           compose_rigid_transform(tool, mount))
        datum = self.context["released_datum"]
        gripper = compose_rigid_transform(datum, relation)
        point = compose_rigid_transform(datum, {"translation_m": [-.012, 0., .010],
            "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]})["translation_m"]
        contact = self.contact(self.context["proxies"]["released"]["object_id"], C.ROBOT_ATTACHED,
                               "finger_tip_left_link", C.ROBOT_LINK)
        contact.position.x, contact.position.y, contact.position.z = point
        contact.normal.x, contact.normal.y, contact.normal.z = datum["rotation_columns"][0]
        contact.depth = .003
        args = dict(gripper_pose=gripper, gripper_m=.01176)
        self.assertTrue(intended_request_contact(self.context, self.plan, "released", contact, **args))
        native = bind_request_contacts(self.context, self.plan, released_world=True)
        self.assertFalse(native("released", contact, **args))
        contact.body_type_1 = C.WORLD_OBJECT
        self.assertTrue(native("released", contact, **args))
        self.assertFalse(intended_request_contact(self.context, self.plan, "released", contact, **args))
        contact.normal.x, contact.normal.y, contact.normal.z = datum["rotation_columns"][2]
        self.assertFalse(intended_request_contact(self.context, self.plan, "released", contact, **args))
        self.assertFalse(native("released", contact, **args))
        changed = copy.deepcopy(self.context)
        changed["proxies"]["released"]["touch_links"] = TIPS[:]
        changed["geometry_digest"] = canonical_digest({k: v for k, v in changed.items() if k != "geometry_digest"})
        with self.assertRaisesRegex(ContractError, "CONTACT_REQUEST_GEOMETRY"):
            request_geometry(changed, self.plan, [0.] * 6, .012, hypothesis="released")

    def test_installed_moveit_conversion_preserves_world_and_checks_added_geometry(self):
        import subprocess
        import tempfile
        # Same RobotState conversion/checkCollision boundary as installed
        # GetStateValidity, but a tiny in-memory model: no ROS initialization.
        source = r'''
#include <iostream>
#include <iterator>
#include <cstring>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/conversions.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <rclcpp/serialization.hpp>
int main() {
  std::string xml = "<robot name='cpu'><link name='base_link'/>";
  std::string parent = "base_link";
  for (int i=1; i<=6; ++i) {
    std::string child = i==6 ? "gripper_link" : "link"+std::to_string(i);
    xml += "<link name='"+child+"'/><joint name='j"+std::to_string(i)+"' type='continuous'>"
      "<parent link='"+parent+"'/><child link='"+child+"'/><axis xyz='0 0 1'/>"
      "<limit effort='20' velocity='1'/></joint>";
    parent=child;
  }
  xml += R"(<link name="finger"/>
    <joint name="finger_right_joint" type="prismatic"><parent link="gripper_link"/><child link="finger"/>
      <axis xyz="1 0 0"/><limit lower="0" upper="0.021" effort="20" velocity="0.05"/></joint>
  </robot>)";
  auto urdf = urdf::parseURDF(xml);
  auto srdf = std::make_shared<srdf::Model>();
  if (!srdf->initString(*urdf, "<robot name='cpu'/>")) return 2;
  auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  scene->getCurrentStateNonConst().setToDefaultValues();
  std::string bytes((std::istreambuf_iterator<char>(std::cin)), {});
  rclcpp::SerializedMessage serialized(bytes.size());
  auto &raw = serialized.get_rcl_serialized_message();
  std::memcpy(raw.buffer, bytes.data(), bytes.size()); raw.buffer_length = bytes.size();
  moveit_msgs::msg::RobotState message;
  rclcpp::Serialization<moveit_msgs::msg::RobotState>().deserialize_message(&serialized, &message);
  // An existing attachment is preserved by is_diff=true, not cleared as a
  // hidden side effect of checking the prospective body.
  auto retained_message = message;
  retained_message.attached_collision_objects[0].object.id = "existing";
  moveit::core::robotStateMsgToRobotState(retained_message, scene->getCurrentStateNonConst());
  moveit::core::RobotState state = scene->getCurrentState();
  if (!moveit::core::robotStateMsgToRobotState(message, state)) return 3;
  const auto *body = state.getAttachedBody("cube::prospective:carried");
  if (!body || body->getShapes().size()!=1 || body->getAttachedLinkName()!="gripper_link" || !state.hasAttachedBody("existing")) return 4;
  // Retained source geometry overlaps its request-local carried hypothesis.
  // Native FCL must report that artificial pair, not silently erase the world.
  scene->getWorldNonConst()->addToObject("cube", body->getShapes()[0], body->getGlobalCollisionBodyTransforms()[0]);
  collision_detection::CollisionRequest request; request.contacts=true; request.max_contacts=10;
  collision_detection::CollisionResult result;
  scene->checkCollision(request, result, state);
  if (!result.collision || !scene->getWorld()->hasObject("cube") || scene->getCurrentState().hasAttachedBody(body->getName())) return 5;
  std::cout << "REQUEST_ATTACHMENT_PRESENT SOURCE_WORLD_RETAINED DUPLICATE_CONTACT_CHECKED\n";
}
'''
        ros = Path("/opt/ros/jazzy")
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / "request_geometry")
            command = ["g++", "-std=c++17", "-x", "c++", "-", "-o", binary,
                "-I/usr/include/eigen3", *["-I" + str(p) for p in (ros / "include").iterdir() if p.is_dir()],
                "-L" + str(ros / "lib"), "-Wl,-rpath," + str(ros / "lib"),
                "-lmoveit_planning_scene", "-lmoveit_robot_model", "-lmoveit_robot_state",
                "-lmoveit_collision_detection", "-lsrdfdom", "-lurdfdom_model", "-lrclcpp",
                "-lmoveit_msgs__rosidl_typesupport_cpp", "-lrcutils"]
            compiled = subprocess.run(command, input=source, text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            request = request_geometry(self.context, self.plan, [0.] * 6, .012, hypothesis="carried")
            replay = subprocess.run([binary], input=serialize_message(request.robot_state), capture_output=True)
            self.assertEqual(replay.returncode, 0, replay.stderr.decode())
            self.assertIn(b"REQUEST_ATTACHMENT_PRESENT SOURCE_WORLD_RETAINED DUPLICATE_CONTACT_CHECKED", replay.stdout)


if __name__ == "__main__":
    unittest.main()
