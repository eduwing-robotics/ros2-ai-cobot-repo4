"""ROS Jazzy plan-only transport for the FR5 pickup executor."""
from __future__ import annotations

import base64
import copy
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from tools.fr5_data_factory import ContractError, canonical_digest
from .arm_stream import ArmStream


JOINT_ORDER = ["j1", "j2", "j3", "j4", "j5", "j6"]
ACTION_TYPES = {
    "/move_action": "moveit_msgs/action/MoveGroup",
    "/execute_trajectory": "moveit_msgs/action/ExecuteTrajectory",
    "/gripper_controller/follow_joint_trajectory":
        "control_msgs/action/FollowJointTrajectory",
}


def validate_controller_sample(controller):
    """Validate retained JTC diagnostics, not goal identity or hardware freshness."""
    if not isinstance(controller, dict):
        raise ContractError("ROS_CONTROLLER_SAMPLE")
    if "sample" not in controller:  # Older/native-independent snapshots lack it.
        return
    sample = controller["sample"]
    fields = {"joint_names", "ros_stamp_ns", "reference_elapsed_ns", "feedback_elapsed_ns",
              "reference_positions", "feedback_positions", "reported_output_positions"}
    if not isinstance(sample, dict) or set(sample) != fields:
        raise ContractError("ROS_CONTROLLER_SAMPLE")
    names = sample["joint_names"]
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names)
            or len(names) != len(set(names))):
        raise ContractError("ROS_CONTROLLER_SAMPLE")
    limit = 2**31 * 10**9
    for key in ("ros_stamp_ns", "reference_elapsed_ns", "feedback_elapsed_ns"):
        value = sample[key]
        if type(value) is not int or not (-limit if key != "ros_stamp_ns" else 0) <= value < limit:
            raise ContractError("ROS_CONTROLLER_SAMPLE")
    for key in ("reference_positions", "feedback_positions", "reported_output_positions"):
        values = sample[key]
        if (not isinstance(values, list)
                or len(values) not in ({0, len(names)} if key == "reported_output_positions" else {len(names)})):
            raise ContractError("ROS_CONTROLLER_SAMPLE")
        try:
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ContractError("ROS_CONTROLLER_SAMPLE")
        except OverflowError as exc:
            raise ContractError("ROS_CONTROLLER_SAMPLE") from exc
    if "reference_position_m" in controller:
        if (names != ["finger_right_joint"]
                or sample["reference_positions"] != [controller["reference_position_m"]]
                or sample["feedback_positions"] != [controller.get("feedback_position_m")]):
            raise ContractError("ROS_CONTROLLER_SAMPLE_BINDING")


@dataclass(slots=True)
class _ActivePhase:
    phase: str
    type: str
    deadline: float
    goal_future: object | None = None
    goal_handle: object | None = None
    result_future: object | None = None
    held_segment: dict | None = None
    arm_segment: dict | None = None
    start_observation: dict | None = None
    action_succeeded: bool = False
    action_terminal_observation: dict | None = None
    terminal_observation: dict | None = None
    terminal_observation_status: str = "PENDING"
    terminal_observation_code: str | None = None


def _rotation_quaternion(columns):
    """Convert column-major rotation axes to a deterministic xyzw quaternion."""
    matrix = [[columns[column][row] for column in range(3)] for row in range(3)]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
        w = scale / 4
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        if index == 0:
            scale = math.sqrt(1 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
            x, y, z, w = (
                scale / 4,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[2][1] - matrix[1][2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
            x, y, z, w = (
                (matrix[0][1] + matrix[1][0]) / scale,
                scale / 4,
                (matrix[1][2] + matrix[2][1]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
            )
        else:
            scale = math.sqrt(1 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
            x, y, z, w = (
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                scale / 4,
                (matrix[1][0] - matrix[0][1]) / scale,
            )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    values = [x / norm, y / norm, z / norm, w / norm]
    if values[3] < 0:
        values = [-value for value in values]
    return values


class RosMoveItTransport:
    """Build plan-only MoveGroup requests and serialized gripper goals."""

    def capture_scene_illumination(self, plan):
        """Measure the mapped UP frame through the existing observation owner.

        This is a source-timed measurement, not a light-state certificate or a
        continuous monitor. The terminal consumer applies its brightness rule.
        """
        import cv2
        import numpy as np
        from tools.data_factory.rollout.finite_plan import check_freshness
        from tools.data_factory.readiness import RECORDER_READINESS_CONTRACT
        try:
            proposal = plan["learned_proposal"]
            inputs = proposal["runtime_inputs"]
            camera = inputs["camera_mapping"]["observation.images." + RECORDER_READINESS_CONTRACT["scene_camera"]]
            slot = camera.removeprefix("observation.images.")
            if camera not in {"observation.images.camera1", "observation.images.camera2"}:
                raise ValueError("scene camera mapping")
            topic = inputs["camera_topics"][slot]
            age = proposal["max_observation_age_s"]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ContractError("MECHANICAL_ILLUMINATION_UNAVAILABLE") from exc
        observation = self.capture_policy_observation(inputs["camera_topics"], age)
        # capture_policy_observation already validates ROS encoding and shape.
        frame = observation[camera]
        rgb = np.frombuffer(bytes.fromhex(frame["data_hex"]), dtype=np.uint8).reshape(frame["shape"])
        brightness = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())
        check_freshness({"source_timestamps_s": observation["source_timestamps_s"],
                         "max_observation_age_s": age}, time.time())
        stamp = observation["source_timestamps_s"][slot]
        return {"kind": "CURRENT_REQUIRED_ILLUMINATION", "camera_topic": topic,
                "brightness_mean": brightness, "source_timestamp_s": stamp,
                "valid_until_s": stamp + age}

    def prepare_contact_transition(self, plan, scene_object, *, first, deadline_s):
        from .contact_transition import prepare
        if not first:
            return {"status": "UNAVAILABLE", "code": "CONTACT_PROSPECTIVE_START_REQUIRED"}
        segments = plan["steps"][0].get("held_target_segments", [])
        if not segments or any(step["type"] not in {"ARM", "GRIPPER"} for step in segments):
            return {"status": "UNAVAILABLE", "code": "CONTACT_PREFIX_UNSUPPORTED"}
        try:
            context = prepare(self, plan, scene_object)
            context["deadline_s"] = deadline_s
        except ContractError as exc:
            if exc.code == "CONTACT_PROFILE_UNAVAILABLE":
                return {"status": "UNAVAILABLE", "code": exc.code}
            raise
        except (KeyError, AttributeError, ValueError, TypeError, ET.ParseError) as exc:
            raise ContractError("CONTACT_GEOMETRY_UNAVAILABLE") from exc
        source = plan["learned_source_program"]
        obj = self._collision_object(plan["scene_binding"]["object_instance_id"], context["dimensions_m"],
                                     context["datum"]["translation_m"], source["planning_scene"]["frame_id"])
        q = _rotation_quaternion(context["datum"]["rotation_columns"])
        obj.primitive_poses[0].orientation.x, obj.primitive_poses[0].orientation.y, obj.primitive_poses[0].orientation.z, obj.primitive_poses[0].orientation.w = q
        request = self._ApplyPlanningScene.Request()
        request.scene = self._PlanningScene(is_diff=True)
        request.scene.world.collision_objects = [*self._planning_scene_objects(source["planning_scene"]), obj]
        response = self._service(self._ApplyPlanningScene, "/apply_planning_scene", request, "PLANNING_SCENE_APPLY")
        if response.success is not True:
            raise ContractError("PLANNING_SCENE_APPLY")
        try:
            self._read_mechanical_scene(source, released=obj)
        except ContractError as exc:
            # Only this initial prepare owns the just-added world object. No
            # motion was submitted; removing it does not restore physical Scene.
            cleanup = {"status": "UNCONFIRMED", "scope": "COLLISION_WORLD_ONLY", "object_id": obj.id}
            try:
                removed = self._CollisionObject(id=obj.id, operation=self._CollisionObject.REMOVE)
                removed.header.frame_id = obj.header.frame_id
                cleanup_request = self._ApplyPlanningScene.Request()
                cleanup_request.scene = self._PlanningScene(is_diff=True)
                cleanup_request.scene.world.collision_objects = [removed]
                cleanup_response = self._service(self._ApplyPlanningScene, "/apply_planning_scene", cleanup_request, "PLANNING_SCENE_APPLY")
                if cleanup_response.success is not True:
                    raise ContractError("PLANNING_SCENE_APPLY")
                digest = self._read_mechanical_scene(source)
                cleanup.update(status="REMOVAL_CONFIRMED", planning_scene_readback_digest=digest)
            except Exception as cleanup_error:
                cleanup["code"] = cleanup_error.code if isinstance(cleanup_error, ContractError) else "CONTACT_CLEANUP_UNCONFIRMED"
            return {"status": "UNAVAILABLE", "code": exc.code, "collision_world_cleanup": cleanup}
        self._contact_world_object = obj
        return context

    def contact_fk(self, source, snapshot, link):
        from moveit_msgs.srv import GetPositionFK
        from tools.fr5_data_factory import validate_rigid_transform
        query = GetPositionFK.Request()
        query.header.frame_id = source["frames"]["planning_frame"]
        query.fk_link_names = [link]
        query.robot_state = self._RobotState(joint_state=self._JointState(
            name=[*JOINT_ORDER, "finger_right_joint"],
            position=[*snapshot["joint_positions"], snapshot["gripper_controller"]["feedback_position_m"]]))
        result = self._service(GetPositionFK, "/compute_fk", query, "MECHANICAL_FK")
        if (result.error_code.val != self._moveit_success or list(result.fk_link_names) != [link]
                or len(result.pose_stamped) != 1 or result.pose_stamped[0].header.frame_id != query.header.frame_id):
            raise ContractError("MECHANICAL_FK")
        p = result.pose_stamped[0].pose
        x,y,z,w = p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w
        if abs(x*x+y*y+z*z+w*w-1) > 1e-9:
            raise ContractError("MECHANICAL_FK")
        return validate_rigid_transform({"translation_m": [p.position.x,p.position.y,p.position.z],
            "rotation_columns": [[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)],
                                 [2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)],
                                 [2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)]]}, "MECHANICAL_FK")

    def check_contact_segment(self, plan, step, snapshot, context, *, closing):
        if context["close"] is not None:
            return self.check_mechanical_step(plan, step, snapshot)
        self._read_mechanical_scene(plan["learned_source_program"], released=self._contact_world_object)
        checked = {**plan, "initial_joint_state": snapshot["joint_positions"], "steps": [step]}
        if closing:
            checked["contact_touch_object"] = plan["scene_binding"]["object_instance_id"]
        return self._check_plan_collision(checked, snapshot["gripper_controller"]["feedback_position_m"])

    def mechanical_contact_context(self, plan, scene_object, snapshot, *, prospective=None):
        """Report the native producer's available evidence without inventing grasp.

        Native stable closure and its calibrated feedback range establish the
        accepted contact criterion. They do not locate the cube along the tips
        after arbitrary learned motion; preserve that distinction at the caller.
        """
        if prospective is not None and prospective.get("status") == "PROSPECTIVE":
            from .contact_transition import consume
            return consume(self, plan, scene_object, snapshot, prospective)
        diagnostic = {"status": "BLOCKED_UNAVAILABLE", "source_program_digest": canonical_digest(plan["learned_source_program"]),
            "scene_binding": copy.deepcopy(plan["scene_binding"]), "snapshot_digest": canonical_digest(snapshot),
            "scene_object_digest": canonical_digest(scene_object), "physical_success": False,
            "missing_measurements": ["QUALIFIED_OBJECT_TOOL_RELATION_OR_CARRIED_ENVELOPE",
                "QUALIFIED_CONTACT_TRANSITION_AND_APERTURE_MODEL", "CALIBRATED_CLOSURE_PLATEAU", "CURRENT_REQUIRED_ILLUMINATION"],
            "code": "MECHANICAL_CONTACT_UNAVAILABLE"}
        from tools.data_factory.motion.mechanical_terminal import closure_plateau
        try:
            diagnostic["closure_contact"] = closure_plateau(plan, snapshot, time.time(), self._clock())
            diagnostic["missing_measurements"].remove("CALIBRATED_CLOSURE_PLATEAU")
            diagnostic["relation_status"] = "UNOBSERVED_AFTER_LEARNED_MOTION"
            # This is the source Scene's initial pose, not a measured terminal
            # pose. No nominal datum-to-tool transform is attached here.
            diagnostic["initial_scene_pose"] = copy.deepcopy(scene_object.get("pose"))
        except ContractError as exc:
            diagnostic["closure_contact"] = {"status": "UNAVAILABLE", "code": exc.code}
        try:
            diagnostic["illumination"] = self.capture_scene_illumination(plan)
            diagnostic["missing_measurements"].remove("CURRENT_REQUIRED_ILLUMINATION")
        except ContractError as exc:
            diagnostic["illumination"] = {"kind": "UNAVAILABLE", "code": exc.code}
        description = getattr(self, "_robot_description", None)
        if isinstance(description,str):
            try:
                root = ET.fromstring(description)
                finger = [root.find(f"joint[@name='finger_{side}_joint']") for side in ("right","left")]
                origins = [float(j.find("origin").attrib["xyz"].split()[0]) for j in finger]
                axes = [float(j.find("axis").attrib["xyz"].split()[0]) for j in finger]
                widths = [float(root.find(f"link[@name='finger_tip_{side}_link']/collision/geometry/box").attrib["size"].split()[0]) for side in ("right","left")]
                opened = next(s for s in plan["learned_source_program"]["steps"] if s["phase"]=="GRIPPER_OPEN")["gripper_position_m"]
                gap = origins[0]-origins[1]+opened*(axes[0]-axes[1])-sum(widths)/2
                diagnostic["modeled_full_open_inner_gap_m"] = gap
                diagnostic["model_opening_direction"] = "INWARD" if axes[0]-axes[1]<0 else "OUTWARD"
                if axes[0]-axes[1] <= 0:
                    diagnostic["model_contact_code"] = "MODELED_APERTURE_OPPOSES_OPEN_COMMAND"
            except (ET.ParseError, AttributeError, KeyError, ValueError, IndexError, StopIteration):
                diagnostic["model_geometry"] = "UNSUPPORTED"
        return diagnostic

    def prepare_mechanical_terminal(self, source, contact):
        """Keep floor/wall checks and add exactly one supported carried object.

        Applying a MoveIt attachment changes collision accounting only. This
        method never upgrades the contact producer's physical evidence.
        """
        from moveit_msgs.msg import AttachedCollisionObject
        required = {"object_id", "link_name", "touch_links", "dimensions_m", "translation_m", "rotation_xyzw"}
        obj = contact.get("carried_object")
        if not isinstance(obj, dict) or set(obj) != required:
            raise ContractError("MECHANICAL_CONTACT_GEOMETRY")
        links = {"finger_tip_right_link", "finger_tip_left_link"}
        if (obj["link_name"] != "gripper_link" or set(obj["touch_links"]) != links
                or not isinstance(obj["object_id"], str) or not obj["object_id"]):
            raise ContractError("MECHANICAL_CONTACT_LINKS")
        for key, size in (("dimensions_m", 3), ("translation_m", 3), ("rotation_xyzw", 4)):
            values = obj[key]
            if (not isinstance(values, list) or len(values) != size
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                    or key == "dimensions_m" and any(v <= 0 for v in values)):
                raise ContractError("MECHANICAL_CONTACT_GEOMETRY")
        if abs(sum(v*v for v in obj["rotation_xyzw"]) - 1) > 1e-9:
            raise ContractError("MECHANICAL_CONTACT_GEOMETRY")
        if obj["object_id"] in {source["planning_scene"][key]["id"] for key in ("floor", "wall")}:
            raise ContractError("MECHANICAL_CONTACT_GEOMETRY")
        held = self._collision_object(obj["object_id"], obj["dimensions_m"], obj["translation_m"], obj["link_name"])
        held.primitive_poses[0].orientation.x, held.primitive_poses[0].orientation.y, held.primitive_poses[0].orientation.z, held.primitive_poses[0].orientation.w = obj["rotation_xyzw"]
        attached = AttachedCollisionObject(link_name=obj["link_name"], object=held, touch_links=sorted(links))
        removed = self._CollisionObject(id=obj["object_id"], operation=self._CollisionObject.REMOVE)
        removed.header.frame_id = source["planning_scene"]["frame_id"]
        request = self._ApplyPlanningScene.Request()
        request.scene = self._PlanningScene(is_diff=True)
        request.scene.world.collision_objects = [*self._planning_scene_objects(source["planning_scene"]), removed]
        request.scene.robot_state.is_diff = True
        request.scene.robot_state.attached_collision_objects = [attached]
        response = self._service(self._ApplyPlanningScene, "/apply_planning_scene", request, "PLANNING_SCENE_APPLY")
        if getattr(response, "success", False) is not True:
            raise ContractError("PLANNING_SCENE_APPLY")
        readback=self._read_mechanical_scene(source, attached)
        self._mechanical_scene_expected=(copy.deepcopy(source),attached,None)
        return readback

    def check_mechanical_step(self, terminal, step, snapshot):
        source,attached,released=self._mechanical_scene_expected
        self._read_mechanical_scene(source,attached,released)
        return self._check_plan_collision({**terminal,"initial_joint_state":snapshot["joint_positions"],
            "steps":[step],"mechanical_terminal":True},snapshot["gripper_controller"]["feedback_position_m"])

    @staticmethod
    def _same_collision_object_readback(actual, expected):
        """Allow only pose roundoff; all remaining ROS message fields stay exact."""
        if len(actual.primitive_poses) != len(expected.primitive_poses):
            return False
        for a, b in zip([actual.pose, *actual.primitive_poses], [expected.pose, *expected.primitive_poses]):
            position_a = [a.position.x, a.position.y, a.position.z]
            position_b = [b.position.x, b.position.y, b.position.z]
            qa = [a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w]
            qb = [b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w]
            # Same 1e-9 numerical convention as _apply_and_readback_scene,
            # not a physical collision/position tolerance. Unit q and -q
            # represent the same rotation; never normalize an invalid input.
            if (any(not math.isfinite(v) for v in [*position_a, *position_b, *qa, *qb])
                    or any(abs(x-y) > 1e-9 for x,y in zip(position_a, position_b))
                    or any(abs(sum(v*v for v in q)-1) > 1e-9 for q in (qa, qb))
                    or not any(all(abs(x-sign*y) <= 1e-9 for x,y in zip(qa, qb)) for sign in (1, -1))):
                return False
        compared = copy.deepcopy(actual)
        compared.pose = copy.deepcopy(expected.pose)
        compared.primitive_poses = copy.deepcopy(expected.primitive_poses)
        return compared == expected

    def _read_mechanical_scene(self, source, attached=None, released=None):
        query = self._GetPlanningScene.Request()
        query.components = self._PlanningSceneComponents(components=(
            self._PlanningSceneComponents.WORLD_OBJECT_GEOMETRY | self._PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | self._PlanningSceneComponents.ALLOWED_COLLISION_MATRIX))
        result = self._service(self._GetPlanningScene, "/get_planning_scene", query, "PLANNING_SCENE_READ")
        scene = result.scene
        matrix = scene.allowed_collision_matrix
        object_id = released.id if released is not None else attached.object.id if attached is not None else None
        if (any(matrix.default_entry_values)
                or object_id in matrix.entry_names and any(matrix.entry_values[list(matrix.entry_names).index(object_id)].enabled)):
            raise ContractError("CONTACT_COLLISION_POLICY")
        expected = self._planning_scene_objects(source["planning_scene"])
        if released is not None:
            expected.append(released)
        expected_attached = [] if attached is None else [attached]
        if (len(scene.world.collision_objects) != len(expected)
                or {obj.id for obj in scene.world.collision_objects} != {obj.id for obj in expected}
                or len(scene.robot_state.attached_collision_objects) != len(expected_attached)):
            raise ContractError("PLANNING_SCENE_MISMATCH")
        for actual, obj in zip(scene.robot_state.attached_collision_objects, expected_attached):
            if not self._same_collision_object_readback(actual.object, obj.object):
                raise ContractError("PLANNING_SCENE_MISMATCH")
            compared = copy.deepcopy(actual)
            compared.object = copy.deepcopy(obj.object)
            if compared != obj:
                raise ContractError("PLANNING_SCENE_MISMATCH")
        for obj in expected:
            actual = next(item for item in scene.world.collision_objects if item.id == obj.id)
            if not self._same_collision_object_readback(actual, obj):
                raise ContractError("PLANNING_SCENE_MISMATCH")
        return canonical_digest({"world":[str(obj) for obj in sorted(scene.world.collision_objects,key=lambda obj:obj.id)],
                                 "attached":[str(obj) for obj in scene.robot_state.attached_collision_objects]})

    def check_mechanical_terminal(self, terminal, source):
        """Collision/no-motion checks accept a measured held start, never open-only."""
        before = terminal["initial_snapshot"]
        # Post-release trajectories were planned against a request-local detach
        # diff. The actual scene stays attached until full-open completion;
        # dispatch checks those trajectories against the actual detached scene.
        plan = {**source, "initial_joint_state": before["joint_positions"],
                "steps": terminal["steps"][:3], "mechanical_terminal": True}
        collision=self._check_plan_collision(plan, before["gripper_controller"]["feedback_position_m"])
        current = self.snapshot(source["planning"]["max_joint_state_age_s"])
        tolerance = source["planning"]["goal_tolerances"]["joint_rad"]
        opened = next(s for s in source["steps"] if s["phase"] == "GRIPPER_OPEN")
        if (any(abs(a-b) > tolerance for a,b in zip(current["joint_positions"], before["joint_positions"]))
                or any(abs(current["gripper_controller"][key] - before["gripper_controller"][key])
                       > opened["limits"]["completion_tolerance_m"] for key in ("feedback_position_m", "reference_position_m"))):
            raise ContractError("PLAN_ONLY_MOVED_ROBOT")
        return {"collision_report":collision,"before_snapshot_digest":canonical_digest(before),
                "after_snapshot_digest":canonical_digest(current),"status":"PASS"}

    def mechanical_future_scene(self, source, contact, joints):
        """Request-local detached geometry for retreat planning; never apply it."""
        from moveit_msgs.msg import AttachedCollisionObject
        from tools.fr5_data_factory import compose_rigid_transform
        obj = contact["carried_object"]
        opened = next(s["gripper_position_m"] for s in source["steps"] if s["phase"] == "GRIPPER_OPEN")
        base = self.contact_fk(source, {"joint_positions": joints,
            "gripper_controller": {"feedback_position_m": opened}}, obj["link_name"])
        x,y,z,w = obj["rotation_xyzw"]
        relation = {"translation_m": obj["translation_m"], "rotation_columns":
            [[1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)],
             [2*(x*y-z*w),1-2*(x*x+z*z),2*(y*z+x*w)],
             [2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)]]}
        pose = compose_rigid_transform(base, relation)
        released = self._collision_object(obj["object_id"], obj["dimensions_m"], pose["translation_m"], source["frames"]["planning_frame"])
        released.primitive_poses[0].orientation.x, released.primitive_poses[0].orientation.y, released.primitive_poses[0].orientation.z, released.primitive_poses[0].orientation.w = _rotation_quaternion(pose["rotation_columns"])
        diff = self._PlanningScene(is_diff=True)
        diff.robot_state.is_diff = True
        diff.robot_state.joint_state = self._JointState(name=["finger_right_joint"], position=[opened])
        detach = AttachedCollisionObject(link_name=obj["link_name"])
        detach.object.id, detach.object.operation = obj["object_id"], self._CollisionObject.REMOVE
        diff.robot_state.attached_collision_objects = [detach]
        diff.world.collision_objects = [released]
        return diff

    def complete_mechanical_terminal(self, terminal):
        """Read final commanded state; no learned/physical landing success claim."""
        current = self.snapshot(terminal["planning"]["max_joint_state_age_s"])
        safe = terminal["steps"][-1]["final_joint_state"]
        opened = next(step for step in terminal["steps"] if step["phase"] == "GRIPPER_OPEN")
        tolerance = terminal["planning"]["goal_tolerances"]["joint_rad"]
        if (current["arm_controller"]["ready"] is not True or current["gripper_controller"]["ready"] is not True
                or any(abs(a-b) > tolerance for a,b in zip(current["joint_positions"], safe))
                or any(abs(current["gripper_controller"][key] - opened["gripper_position_m"])
                       > opened["limits"]["completion_tolerance_m"] for key in ("feedback_position_m", "reference_position_m"))):
            raise ContractError("POST_RESET_SAFE_SNAPSHOT")
        return {"status": "EXPECTED_RESET", "snapshot_digest": canonical_digest(current),
                "semantic_success": "NOT_MEASURED", "physical_landing": "NOT_OBSERVED"}

    def mechanical_release_readback(self, terminal):
        """Detach one object at measured FK; this is an expected world pose."""
        from moveit_msgs.msg import AttachedCollisionObject
        from moveit_msgs.srv import GetPositionFK
        snapshot = self.snapshot(terminal["planning"]["max_joint_state_age_s"])
        obj = terminal["contact_evidence"]["carried_object"]
        query = GetPositionFK.Request()
        query.header.frame_id = terminal["planning_scene"]["frame_id"]
        query.fk_link_names = [obj["link_name"]]
        query.robot_state = self._RobotState(joint_state=self._JointState(
            name=[*JOINT_ORDER, "finger_right_joint"],
            position=[*snapshot["joint_positions"], snapshot["gripper_controller"]["feedback_position_m"]]))
        result = self._service(GetPositionFK, "/compute_fk", query, "MECHANICAL_FK")
        if (result.error_code.val != self._moveit_success or list(result.fk_link_names) != [obj["link_name"]]
                or len(result.pose_stamped) != 1 or result.pose_stamped[0].header.frame_id != query.header.frame_id):
            raise ContractError("MECHANICAL_FK")
        pose = result.pose_stamped[0].pose
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        def multiply(a,b):
            x,y,z,w=a; X,Y,Z,W=b
            return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]
        if any(not math.isfinite(v) for v in [*q, pose.position.x, pose.position.y, pose.position.z]) or abs(sum(v*v for v in q)-1)>1e-9:
            raise ContractError("MECHANICAL_FK")
        rotated = multiply(multiply(q, [*obj["translation_m"],0.]),[-q[0],-q[1],-q[2],q[3]])[:3]
        translation = [a+b for a,b in zip([pose.position.x,pose.position.y,pose.position.z],rotated)]
        orientation = multiply(q,obj["rotation_xyzw"])
        released = self._collision_object(obj["object_id"],obj["dimensions_m"],translation,query.header.frame_id)
        released.primitive_poses[0].orientation.x, released.primitive_poses[0].orientation.y, released.primitive_poses[0].orientation.z, released.primitive_poses[0].orientation.w = orientation
        detach = AttachedCollisionObject(link_name=obj["link_name"])
        detach.object.id = obj["object_id"]
        detach.object.operation = self._CollisionObject.REMOVE
        request = self._ApplyPlanningScene.Request()
        request.scene = self._PlanningScene(is_diff=True)
        request.scene.robot_state.is_diff = True
        request.scene.robot_state.attached_collision_objects = [detach]
        request.scene.world.collision_objects = [released]
        response = self._service(self._ApplyPlanningScene, "/apply_planning_scene", request, "PLANNING_SCENE_APPLY")
        if getattr(response,"success",False) is not True:
            raise ContractError("PLANNING_SCENE_APPLY")
        self._read_mechanical_scene(terminal,released=released)
        self._mechanical_scene_expected=(copy.deepcopy(terminal),None,released)
        return {"semantics":"MODEL_BASED_EXPECTATION", "physical_landing":"NOT_OBSERVED",
                "object_id":obj["object_id"], "translation_m":translation, "rotation_xyzw":orientation,
                "snapshot_digest":canonical_digest(snapshot)}

    def __init__(
        self, node, *, graph_timeout_s=1.0, preflight_timeout_s=5.0,
        clock=time.monotonic, gripper_source_clock=None, gripper_temporal_policy=None, allow_clock_configuration=False,
    ):
        try:
            import rclpy
            from action_msgs.msg import GoalStatus
            from builtin_interfaces.msg import Duration
            from control_msgs.action import FollowJointTrajectory
            from control_msgs.msg import JointTolerance, JointTrajectoryControllerState, DynamicJointState
            from geometry_msgs.msg import Pose
            from moveit_msgs.action import ExecuteTrajectory, MoveGroup
            from moveit_msgs.msg import (
                CollisionObject,
                Constraints,
                JointConstraint,
                OrientationConstraint,
                PositionConstraint,
                RobotState,
                RobotTrajectory,
                PlanningScene,
                PlanningSceneComponents,
                MoveItErrorCodes,
            )
            from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
            from rclpy.action import ActionClient, get_action_names_and_types
            from rclpy.parameter_client import AsyncParameterClient
            from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
            from rcl_interfaces.msg import ParameterType
            from rclpy.serialization import deserialize_message, serialize_message
            from sensor_msgs.msg import JointState
            from shape_msgs.msg import SolidPrimitive
            from std_msgs.msg import String
            from trajectory_msgs.msg import JointTrajectoryPoint
        except ImportError as exc:
            raise ContractError("ROS_JAZZY_UNAVAILABLE", str(exc)) from exc

        self.node = node
        self.graph_timeout_s = graph_timeout_s
        self.preflight_timeout_s = preflight_timeout_s
        self._clock = clock
        self._rclpy = rclpy
        self._get_action_names_and_types = get_action_names_and_types
        self._ActionClient = ActionClient
        self._serialize_message = serialize_message
        self._deserialize_message = deserialize_message
        self._Duration = Duration
        self._FollowJointTrajectory = FollowJointTrajectory
        self._ExecuteTrajectory = ExecuteTrajectory
        self._RobotTrajectory = RobotTrajectory
        self._CollisionObject = CollisionObject
        self._PlanningScene = PlanningScene
        self._PlanningSceneComponents = PlanningSceneComponents
        self._ApplyPlanningScene = ApplyPlanningScene
        self._GetPlanningScene = GetPlanningScene
        self._GetStateValidity = GetStateValidity
        self._goal_succeeded = GoalStatus.STATUS_SUCCEEDED
        self._goal_canceled = GoalStatus.STATUS_CANCELED
        self._goal_aborted = GoalStatus.STATUS_ABORTED
        self._moveit_success = MoveItErrorCodes.SUCCESS
        self._gripper_success = FollowJointTrajectory.Result.SUCCESSFUL
        self._parameter_string = ParameterType.PARAMETER_STRING
        self._JointTolerance = JointTolerance
        self._Pose = Pose
        self._MoveGroup = MoveGroup
        self._Constraints = Constraints
        self._JointConstraint = JointConstraint
        self._OrientationConstraint = OrientationConstraint
        self._PositionConstraint = PositionConstraint
        self._RobotState = RobotState
        self._JointState = JointState
        self._JointTrajectoryControllerState = JointTrajectoryControllerState
        self._SolidPrimitive = SolidPrimitive
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self.move_group = ActionClient(node, MoveGroup, "/move_action")
        self.execute_trajectory = ActionClient(
            node, ExecuteTrajectory, "/execute_trajectory"
        )
        self.gripper = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self._joint_state = None
        self._joint_state_received_at = None
        self._arm_controller_state = None
        self._arm_controller_received_at = None
        self._gripper_hardware_state = None
        self._gripper_hardware_received_at = None
        from tools.data_factory.rollout.gripper_evidence import validate_clock_binding
        if gripper_source_clock is not None and gripper_temporal_policy is not None:
            raise ContractError("LEARNED_RUN_INPUTS")
        self._gripper_source_clock = (validate_clock_binding(gripper_source_clock)
                                      if gripper_source_clock is not None else None)
        if gripper_temporal_policy is not None:
            from tools.data_factory.rollout.gripper_evidence import validate_temporal_policy
            self._gripper_source_clock = validate_temporal_policy(gripper_temporal_policy)
        self._allow_clock_configuration = allow_clock_configuration is True
        self._native_clock_configured_age = None
        self._AsyncParameterClient = AsyncParameterClient
        self._gripper_controller_state = None
        self._gripper_controller_received_at = None
        self._robot_description = None
        self._initial_snapshot_complete = False
        self._active = None
        self._execution_locked = False
        self._execute_goal_count = 0
        self._gripper_goal_count = 0
        self._service_clients = {}
        self._joint_state_subscription = None
        self._arm_controller_subscription = None
        self._gripper_controller_subscription = None
        self._robot_description_subscription = None
        self._robot_description_client = None
        if hasattr(node, "create_subscription"):
            # Native v5 publishes current state plus latched command/fault proof.
            # This getter is not a recorder: queued history after inference can
            # spend nearly the whole admission budget before a callback runs.
            state_depth = 1 if (
                self._allow_clock_configuration and self._gripper_source_clock is not None
                and self._gripper_source_clock["schema_version"] == "fr5.gripper_temporal_policy.v2"
            ) else 10
            # Observe identity before qualification; subscription grants no authority.
            self._gripper_hardware_subscription = node.create_subscription(
                DynamicJointState, "/dynamic_joint_states", self._on_gripper_hardware_state, state_depth)
            self._joint_state_subscription = node.create_subscription(
                JointState, "/joint_states", self._on_joint_state, state_depth
            )
            self._arm_controller_subscription = node.create_subscription(
                JointTrajectoryControllerState,
                "/fairino5_controller/controller_state",
                self._on_arm_controller_state,
                state_depth,
            )
            self._gripper_controller_subscription = node.create_subscription(
                JointTrajectoryControllerState,
                "/gripper_controller/controller_state",
                self._on_gripper_controller_state,
                state_depth,
            )
            self._robot_description_subscription = node.create_subscription(
                String,
                "/robot_description",
                self._on_robot_description,
                QoSProfile(
                    depth=1,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    reliability=ReliabilityPolicy.RELIABLE,
                ),
            )
        if hasattr(node, "create_client"):
            self._robot_description_client = AsyncParameterClient(
                node, "/robot_state_publisher"
            )

    def _on_gripper_hardware_state(self, message):
        self._gripper_hardware_state = message
        self._gripper_hardware_received_at = self._clock()

    def _gripper_hardware_evidence(self):
        if (getattr(self, "_gripper_hardware_state", None) is None
                or getattr(self, "_gripper_source_clock", None) is None):
            return None
        from tools.data_factory.rollout.gripper_evidence import decode_dynamic_state
        return decode_dynamic_state(self._gripper_hardware_state, self._gripper_source_clock,
                                    self._gripper_hardware_received_at)

    def _prepare_native_clock(self, max_age_s):
        """Only the existing LIVE child may configure the opt-in hardware node."""
        if (getattr(self, "_gripper_source_clock", None) is None or not getattr(self, "_allow_clock_configuration", False)
                or self._native_clock_configured_age == max_age_s):
            return
        if self._native_clock_configured_age is not None:
            raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
        from rclpy.parameter import Parameter
        from tools.data_factory.rollout.gripper_evidence import native_clock_parameter, decode_dynamic_state, identity
        causal = self._gripper_source_clock["schema_version"] in ("fr5.gripper_temporal_policy.v1", "fr5.gripper_temporal_policy.v2")
        successor = self._gripper_source_clock["schema_version"] == "fr5.gripper_temporal_policy.v2"
        if causal and self._gripper_source_clock["max_age_s"] != max_age_s:
            raise ContractError("LEARNED_HARDWARE_TEMPORAL_POLICY")
        deadline = time.monotonic() + self.preflight_timeout_s
        observed = None
        while time.monotonic() < deadline:
            if self._gripper_hardware_state is not None:
                observed = decode_dynamic_state(self._gripper_hardware_state, self._gripper_source_clock,
                                                self._gripper_hardware_received_at)
                if observed["wire"]["version"] in (1, 2, 3, 4, 5):
                    break
            self._rclpy.spin_once(self.node, timeout_sec=.01)
        if observed is None:
            raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
        if (successor and observed["wire"]["version"] != 5) or (not successor and observed["wire"]["version"] == 5):
            raise ContractError("LEARNED_HARDWARE_INVALID")
        if ((not causal and observed["wire"]["version"] != 2) or observed["wire"]["stopped"] or observed["wire"]["error"]
                or identity(observed["wire"]) != self._gripper_source_clock["incarnation"]):
            raise ContractError("LEARNED_HARDWARE_INCARNATION")
        client = self._AsyncParameterClient(self.node, "/fr5system")
        if not client.wait_for_services(timeout_sec=self.graph_timeout_s):
            raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
        parameter_name = "gripper_source_clock_v1"
        if causal:
            from tools.data_factory.rollout.gripper_evidence import native_temporal_parameter
            expected = native_temporal_parameter(self._gripper_source_clock)
            parameter_name = "gripper_temporal_policy_v2" if successor else "gripper_temporal_policy_v1"
        else:
            expected = native_clock_parameter(self._gripper_source_clock, max_age_s)
        result = self._wait(client.set_parameters_atomically([Parameter(parameter_name, value=expected)]),
                            self.graph_timeout_s, "LEARNED_HARDWARE_CLOCK_BINDING")
        if not result.result.successful:
            raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
        readback = self._wait(client.get_parameters([parameter_name]),
                              self.graph_timeout_s, "LEARNED_HARDWARE_CLOCK_BINDING")
        if len(readback.values) != 1 or list(readback.values[0].double_array_value) != expected:
            raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
        observed = decode_dynamic_state(self._gripper_hardware_state, self._gripper_source_clock,
                                        self._gripper_hardware_received_at)
        if identity(observed["wire"]) != self._gripper_source_clock["incarnation"]:
            raise ContractError("LEARNED_HARDWARE_INCARNATION")
        self._native_clock_configured_age = max_age_s

    def _native_current_ready(self, max_age_s):
        from tools.data_factory.rollout.gripper_evidence import check_current_bracket
        try:
            hw = self._gripper_hardware_evidence()
            if self._gripper_source_clock["schema_version"] == "fr5.gripper_temporal_policy.v2":
                from tools.data_factory.rollout.gripper_evidence import check_delivery
                check_delivery(hw["wire"], self._gripper_source_clock, self._clock(), max_age_s)
                return True
            check_current_bracket(hw["wire"], time.time(), self._clock(), max_age_s)
            versions = (3, 4) if self._gripper_source_clock["schema_version"] == "fr5.gripper_temporal_policy.v1" else (2,)
            return hw["wire"]["version"] in versions and hw["wire"]["valid"] == 1
        except (ContractError, TypeError, KeyError):
            return False

    def _on_joint_state(self, message):
        self._joint_state = message
        self._joint_state_received_at = self._clock()

    def _on_arm_controller_state(self, message):
        self._arm_controller_state = message
        self._arm_controller_received_at = self._clock()

    def _on_gripper_controller_state(self, message):
        self._gripper_controller_state = message
        self._gripper_controller_received_at = self._clock()

    def _on_robot_description(self, message):
        self._robot_description = message.data

    def _service(self, service_type, endpoint, request, code):
        if not hasattr(self.node, "create_client"):
            raise ContractError(code)
        client = self._service_clients.get(endpoint)
        if client is None:
            client = self.node.create_client(service_type, endpoint)
            self._service_clients[endpoint] = client
        try:
            if not client.wait_for_service(timeout_sec=self.graph_timeout_s):
                raise ContractError(code)
            return self._wait(client.call_async(request), self.graph_timeout_s, code)
        except ContractError:
            raise
        except RuntimeError as exc:
            raise ContractError(code, str(exc)) from exc

    def _collision_object(self, identifier, dimensions, position, frame_id):
        if (
            not isinstance(identifier, str) or not identifier
            or not isinstance(dimensions, list) or len(dimensions) != 3
            or not isinstance(position, list) or len(position) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in dimensions)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in position)
        ):
            raise ContractError("PLANNING_SCENE_SCHEMA")
        item = self._CollisionObject()
        item.header.frame_id = frame_id
        item.id = identifier
        item.pose.position.x, item.pose.position.y, item.pose.position.z = map(float, position)
        item.pose.orientation.w = 1.0
        primitive = self._SolidPrimitive(type=self._SolidPrimitive.BOX, dimensions=list(map(float, dimensions)))
        pose = self._Pose()
        pose.orientation.w = 1.0
        item.primitives, item.primitive_poses = [primitive], [pose]
        item.operation = self._CollisionObject.ADD
        return item

    def _planning_scene_objects(self, spec):
        if not isinstance(spec, dict) or set(spec) != {"frame_id", "floor", "wall"}:
            raise ContractError("PLANNING_SCENE_SCHEMA")
        frame_id, floor, wall = spec["frame_id"], spec["floor"], spec["wall"]
        if not isinstance(frame_id, str) or not frame_id:
            raise ContractError("PLANNING_SCENE_SCHEMA")
        if not isinstance(floor, dict) or not isinstance(wall, dict):
            raise ContractError("PLANNING_SCENE_SCHEMA")
        try:
            floor_position = [0.0, 0.0, float(floor["surface_z_m"]) - float(floor["dimensions_m"][2]) / 2]
            wall_position = [0.0, float(wall["near_face_y_m"]) - float(wall["dimensions_m"][1]) / 2, 0.0]
            return [
                self._collision_object(floor["id"], floor["dimensions_m"], floor_position, frame_id),
                self._collision_object(wall["id"], wall["dimensions_m"], wall_position, frame_id),
            ]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ContractError("PLANNING_SCENE_SCHEMA", str(exc)) from exc

    def _apply_and_readback_scene(self, spec, expected_digest):
        if canonical_digest(spec) != expected_digest:
            raise ContractError("PLANNING_SCENE_BINDING")
        expected = self._planning_scene_objects(spec)
        request = self._ApplyPlanningScene.Request()
        request.scene = self._PlanningScene(is_diff=True)
        request.scene.world.collision_objects = expected
        response = self._service(self._ApplyPlanningScene, "/apply_planning_scene", request, "PLANNING_SCENE_APPLY")
        if not getattr(response, "success", False):
            raise ContractError("PLANNING_SCENE_APPLY")
        query = self._GetPlanningScene.Request()
        query.components = self._PlanningSceneComponents(components=self._PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        response = self._service(self._GetPlanningScene, "/get_planning_scene", query, "PLANNING_SCENE_READ")
        observed = getattr(getattr(getattr(response, "scene", None), "world", None), "collision_objects", None)
        by_id = {item.id: item for item in observed or []}
        if set(by_id) != {item.id for item in expected}:
            raise ContractError("PLANNING_SCENE_MISMATCH")
        readback = []
        for item in expected:
            actual = by_id[item.id]
            if (
                actual.header.frame_id != item.header.frame_id
                or len(actual.primitives) != 1 or len(actual.primitive_poses) != 1
                or actual.primitives[0].type != item.primitives[0].type
                or list(actual.primitives[0].dimensions) != list(item.primitives[0].dimensions)
                or any(abs(a - b) > 1e-9 for a, b in zip(
                    (actual.pose.position.x, actual.pose.position.y, actual.pose.position.z),
                    (item.pose.position.x, item.pose.position.y, item.pose.position.z),
                ))
                or any(abs(a - b) > 1e-9 for a, b in zip(
                    (actual.pose.orientation.x, actual.pose.orientation.y, actual.pose.orientation.z, actual.pose.orientation.w),
                    (item.pose.orientation.x, item.pose.orientation.y, item.pose.orientation.z, item.pose.orientation.w),
                ))
                or any(abs(a - b) > 1e-9 for a, b in zip(
                    (actual.primitive_poses[0].position.x, actual.primitive_poses[0].position.y, actual.primitive_poses[0].position.z),
                    (item.primitive_poses[0].position.x, item.primitive_poses[0].position.y, item.primitive_poses[0].position.z),
                ))
                or any(abs(a - b) > 1e-9 for a, b in zip(
                    (actual.primitive_poses[0].orientation.x, actual.primitive_poses[0].orientation.y, actual.primitive_poses[0].orientation.z, actual.primitive_poses[0].orientation.w),
                    (item.primitive_poses[0].orientation.x, item.primitive_poses[0].orientation.y, item.primitive_poses[0].orientation.z, item.primitive_poses[0].orientation.w),
                ))
            ):
                raise ContractError("PLANNING_SCENE_MISMATCH")
            readback.append({
                "id": item.id,
                "frame_id": item.header.frame_id,
                "primitive_type": int(item.primitives[0].type),
                "dimensions_m": list(item.primitives[0].dimensions),
                "pose_position_m": [item.pose.position.x, item.pose.position.y, item.pose.position.z],
                "pose_orientation_xyzw": [item.pose.orientation.x, item.pose.orientation.y, item.pose.orientation.z, item.pose.orientation.w],
                "primitive_pose_position_m": [item.primitive_poses[0].position.x, item.primitive_poses[0].position.y, item.primitive_poses[0].position.z],
                "primitive_pose_orientation_xyzw": [item.primitive_poses[0].orientation.x, item.primitive_poses[0].orientation.y, item.primitive_poses[0].orientation.z, item.primitive_poses[0].orientation.w],
            })
        return readback

    def _compiled_execution_goal(self, compiled_step):
        if not isinstance(compiled_step, dict):
            raise ContractError("ROS_EXEC_STEP")
        phase = compiled_step.get("phase")
        step_type = compiled_step.get("type")
        encoded = compiled_step.get("trajectory_b64")
        limits = compiled_step.get("limits")
        if (
            not isinstance(phase, str)
            or not phase
            or step_type not in {"ARM", "GRIPPER"}
            or not isinstance(encoded, str)
            or not isinstance(limits, dict)
        ):
            raise ContractError("ROS_EXEC_STEP")
        timeout = limits.get("execution_timeout_s")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ContractError("ROS_EXEC_STEP")
        try:
            serialized = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ContractError("ROS_EXEC_B64", str(exc)) from exc
        if not serialized:
            raise ContractError("ROS_EXEC_B64")
        try:
            if step_type == "ARM":
                goal = self._ExecuteTrajectory.Goal()
                goal.trajectory = self._deserialize_message(
                    serialized, self._RobotTrajectory
                )
                client = self.execute_trajectory
            else:
                goal = self._deserialize_message(serialized, self._FollowJointTrajectory.Goal)
                client = self.gripper
        except RuntimeError as exc:
            raise ContractError("ROS_EXEC_DESERIALIZATION", str(exc)) from exc
        if phase == "LEARNED_CHUNK":
            from tools.data_factory.rollout.finite_plan import execution_step
            execution_step(compiled_step, compiled_step["learned_proposal"])
            if "action_range" not in compiled_step and step_type != "ARM":
                raise ContractError("LEARNED_SERIALIZED_ACTION_MISMATCH")
            expected = (self.build_learned_segment(compiled_step["learned_proposal"], compiled_step)
                        if "action_range" in compiled_step else self.build_learned_trajectory(compiled_step.get("learned_proposal")))
            # CDR alignment padding is not a trajectory field and can vary on
            # reserialization. Compare every decoded field; send the approved one.
            actual_message = goal.trajectory if step_type == "ARM" else goal
            expected_message = self._deserialize_message(expected, self._RobotTrajectory if step_type == "ARM" else self._FollowJointTrajectory.Goal)
            if actual_message != expected_message:
                raise ContractError("LEARNED_SERIALIZED_ACTION_MISMATCH")
        return phase, step_type, goal, client, float(timeout)

    def start_phase(
        self, compiled_step, *, cancel_event=None, cancel_timeout_s=None, start_observation=None, dispatch_guard=None,
    ):
        """Start one approved serialized action and retain its sole active handle."""
        if self._execution_locked or self._active is not None:
            raise ContractError("ROS_EXEC_ACTIVE")
        if dispatch_guard is not None and not callable(dispatch_guard):
            raise ContractError("ROS_EXEC_DISPATCH_GUARD")
        phase, step_type, goal, client, timeout = self._compiled_execution_goal(compiled_step)
        if cancel_event is not None and (
            not callable(getattr(cancel_event, "is_set", None))
            or isinstance(cancel_timeout_s, bool)
            or not isinstance(cancel_timeout_s, (int, float))
            or not math.isfinite(cancel_timeout_s)
            or cancel_timeout_s <= 0
        ):
            raise ContractError("ROS_EXEC_CANCEL_TIMEOUT")
        if phase == "LEARNED_CHUNK":
            from tools.data_factory.rollout.finite_plan import check_execution_start
            check_execution_start(compiled_step, start_observation, time.time(), steady_now=self._clock())
        active = _ActivePhase(phase, step_type, self._clock() + timeout)
        if phase == "LEARNED_CHUNK" and step_type == "GRIPPER" and "action_range" in compiled_step:
            active.held_segment = copy.deepcopy(compiled_step)
            active.start_observation = copy.deepcopy(start_observation)
        elif (phase == "LEARNED_CHUNK" and step_type == "ARM" and "action_range" in compiled_step
              and compiled_step["learned_proposal"].get("runtime_inputs", {}).get("hardware_wire_version") == 5):
            active.arm_segment = copy.deepcopy(compiled_step)
            active.start_observation = copy.deepcopy(start_observation)
        self._active = active
        self._execution_locked = True
        if cancel_event is not None and cancel_event.is_set():
            self._active = None
            self._execution_locked = False
            raise ContractError("ROS_EXEC_CANCELLED")
        try:
            # Compilation and evidence validation can consume the caller's
            # remaining lease. Check its original deadline at the actual send.
            if dispatch_guard is not None:
                dispatch_guard()
            sent = client.send_goal_async(goal)
        except ContractError:
            self._active = None
            self._execution_locked = False
            raise
        except Exception as exc:
            self._active = None
            self._execution_locked = False
            raise ContractError("ROS_EXEC_GOAL_FAILED", str(exc)) from exc
        active.goal_future = sent
        if step_type == "ARM":
            self._execute_goal_count += 1
        else:
            self._gripper_goal_count += 1
        try:
            handle = self._wait(sent, self.graph_timeout_s, "ROS_EXEC_GOAL_TIMEOUT")
            accepted = getattr(handle, "accepted", None)
            if accepted is False:
                self._active = None
                self._execution_locked = False
                if cancel_event is not None and cancel_event.is_set():
                    raise ContractError("ROS_EXEC_CANCELLED")
                raise ContractError("ROS_EXEC_REJECTED")
            if accepted is not True:
                raise ContractError("ROS_EXEC_GOAL_RESPONSE_INVALID")
            active.goal_handle = handle
            active.result_future = handle.get_result_async()
        except ContractError as exc:
            if exc.code == "ROS_EXEC_CANCELLED":
                raise
            if cancel_event is None or not cancel_event.is_set():
                raise
            try:
                self.cancel_active(float(cancel_timeout_s))
            except ContractError as cancel_exc:
                raise ContractError("ROS_EXEC_CANCEL_UNCERTAIN") from cancel_exc
            raise ContractError("ROS_EXEC_CANCELLED") from exc
        except RuntimeError as exc:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    self.cancel_active(float(cancel_timeout_s))
                except ContractError as cancel_exc:
                    raise ContractError("ROS_EXEC_CANCEL_UNCERTAIN") from cancel_exc
                raise ContractError("ROS_EXEC_CANCELLED") from exc
            raise ContractError("ROS_EXEC_RESULT_FAILED", str(exc)) from exc
        if cancel_event is not None and cancel_event.is_set():
            try:
                self.cancel_active(float(cancel_timeout_s))
            except ContractError as exc:
                raise ContractError("ROS_EXEC_CANCEL_UNCERTAIN") from exc
            raise ContractError("ROS_EXEC_CANCELLED")
        self._execution_locked = False
        return active

    def open_learned_arm_stream(self, *, deadline):
        """Reserve this transport's sole motion slot for native rolling JTC goals.

        No goal is sent here. The task owner must admit each exact trajectory
        through submit_learned_arm_trajectory's mandatory dispatch guard. Legacy
        MoveIt/gripper phases cannot run concurrently through this transport.
        """
        if self._execution_locked or self._active is not None:
            raise ContractError("ROS_EXEC_ACTIVE")
        client = getattr(self, "_learned_arm_client", None)
        if client is None:
            client = self._ActionClient(self.node, self._FollowJointTrajectory,
                                       "/fairino5_controller/follow_joint_trajectory")
            self._learned_arm_client = client
        stream = ArmStream(client, deadline=deadline, clock=self._clock)
        self._active = stream

    def submit_learned_arm_trajectory(self, goal, *, revision, dispatch_guard):
        """Send a whole native goal; never create row-level execution phases."""
        if self._execution_locked or not isinstance(self._active, ArmStream):
            raise ContractError("ROS_EXEC_ACTIVE")
        if (not isinstance(goal, self._FollowJointTrajectory.Goal)
                or list(goal.trajectory.joint_names) != JOINT_ORDER
                or not goal.trajectory.points):
            raise ContractError("ROS_EXEC_STEP")
        self._active.submit(goal, revision=revision, dispatch_guard=dispatch_guard)
        self._execute_goal_count += 1

    def poll_learned_arm_stream(self):
        if not isinstance(self._active, ArmStream):
            raise ContractError("ROS_EXEC_NO_ACTIVE")
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        return self._active.poll()

    def close_learned_arm_stream(self):
        """Release only a fenced stream whose native handles are all terminal."""
        if not isinstance(self._active, ArmStream):
            raise ContractError("ROS_EXEC_NO_ACTIVE")
        if self._active.owns_goals or not self._active.fenced:
            raise ContractError("ROS_EXEC_ACTIVE")
        self._active = None

    def fence_learned_arm_stream(self):
        """Request cancellation without representing it as completed stopping."""
        if not isinstance(self._active, ArmStream):
            raise ContractError("ROS_EXEC_NO_ACTIVE")
        self._active.cancel()

    def poll_active(self):
        """Return None while active, or a successful phase handle when terminal."""
        active = getattr(self, "_active", None)
        if active is None:
            raise ContractError("ROS_EXEC_NO_ACTIVE")
        if isinstance(active, ArmStream):
            raise ContractError("ROS_EXEC_STREAM_OWNER_REQUIRED")
        if active.result_future is None:
            raise ContractError("ROS_EXEC_GOAL_PENDING")
        if self._clock() > active.deadline:
            raise ContractError("LEARNED_HARDWARE_COMPLETION_TIMEOUT" if active.action_succeeded
                                and (active.held_segment is not None or active.arm_segment is not None) else "ROS_EXEC_RESULT_TIMEOUT")
        try:
            if not active.result_future.done():
                self._rclpy.spin_once(self.node, timeout_sec=0.0)
            if not active.result_future.done():
                return None
            result = active.result_future.result()
        except RuntimeError as exc:
            raise ContractError("ROS_EXEC_RESULT_FAILED", str(exc)) from exc
        if result.status != self._goal_succeeded:
            self._active = None
            self._execution_locked = True
            raise ContractError("ROS_EXEC_FAILED")
        if active.type == "ARM":
            succeeded = result.result.error_code.val == self._moveit_success
        else:
            succeeded = result.result.error_code == self._gripper_success
        if not succeeded:
            self._active = None
            self._execution_locked = True
            raise ContractError("ROS_EXEC_FAILED")
        if active.held_segment is not None or active.arm_segment is not None:
            if not active.action_succeeded:
                active.action_terminal_observation = {
                    "result_status": result.status,
                    "error_code": result.result.error_code.val if active.type == "ARM" else result.result.error_code,
                    "observed_at_s": time.time(), "observed_monotonic_s": self._clock()}
            active.action_succeeded = True
            if self._execution_locked:
                raise ContractError("ROS_EXEC_CANCELLED")
            try:
                completed = (self._arm_endpoint_confirmed(active) if active.arm_segment is not None
                             else self._held_hardware_completed(active))
            except ContractError as exc:
                active.terminal_observation_status = "REJECTED"
                active.terminal_observation_code = exc.code
                raise
            if self._execution_locked:
                raise ContractError("ROS_EXEC_CANCELLED")
            if not completed:
                return None
            active.terminal_observation_status = "CONFIRMED"
            active.terminal_observation_code = None
        self._active = None
        return active

    def terminal_verification(self):
        """Retain the last checked witness separately from any fault snapshot."""
        active = getattr(self, "_active", None)
        if active is None or isinstance(active, ArmStream) or active.action_terminal_observation is None:
            return None
        return copy.deepcopy({"phase": active.phase, "type": active.type,
            "deadline_monotonic_s": active.deadline, "status": active.terminal_observation_status,
            "code": active.terminal_observation_code, "action_terminal": active.action_terminal_observation,
            "checked_observation": active.terminal_observation})

    def _arm_endpoint_confirmed(self, active):
        """Action success plus endpoint confirmation under the original budget.

        v5 proves coherent delivery, not physical acquisition after the action
        result. An otherwise valid off-target sample is pending, even if it was
        delivered later. A valid endpoint needs no extra receipt-time barrier.
        """
        from tools.data_factory.rollout.finite_plan import check_segment_observation
        from tools.data_factory.rollout.gripper_evidence import check_transition
        segment = active.arm_segment
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        snapshot = self.snapshot(segment["max_joint_state_age_s"])
        now, steady = time.time(), self._clock()
        evidence = {"captured_at_s": now, "captured_monotonic_s": steady,
                    "snapshot": copy.deepcopy(snapshot), "action_terminal": copy.deepcopy(active.action_terminal_observation)}
        active.terminal_observation = evidence
        if steady > active.deadline:
            raise ContractError("LEARNED_HARDWARE_COMPLETION_TIMEOUT")
        check_transition(active.start_observation, evidence, command=False)
        try:
            check_segment_observation(segment, evidence, now, steady_now=steady, terminal=True)
        except ContractError as exc:
            # The shared checker validates all hard hardware/reference errors
            # before this endpoint-only condition. Never turn those into waiting.
            if exc.code != "LEARNED_TERMINAL_STATE":
                raise
            active.terminal_observation_status = "PENDING"
            active.terminal_observation_code = exc.code
            return False
        return True

    def _held_hardware_completed(self, active):
        """Observe one retained command; never resend, renew its deadline or replan."""
        from tools.data_factory.rollout.finite_plan import check_segment_observation, _number
        from tools.data_factory.rollout.gripper_evidence import check_transition
        segment = active.held_segment
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        snapshot = self.snapshot(segment["max_joint_state_age_s"])
        now, steady = time.time(), self._clock()
        if steady > active.deadline:
            raise ContractError("LEARNED_HARDWARE_COMPLETION_TIMEOUT")
        evidence = {"captured_at_s": now, "captured_monotonic_s": steady, "snapshot": copy.deepcopy(snapshot),
                    "action_terminal": copy.deepcopy(active.action_terminal_observation)}
        active.terminal_observation = evidence
        try:
            check_segment_observation(segment, evidence, now, steady_now=steady, allow_pending=True)
        except ContractError as exc:
            if exc.code == "START_STATE_MISMATCH":
                raise ContractError("LEARNED_TERMINAL_STATE") from exc
            raise
        check_transition(active.start_observation, evidence, command=True, allow_queued=True)
        wire = snapshot["gripper_controller"]["hardware_execution"]["wire"]
        reference = _number(snapshot["gripper_controller"]["reference_position_m"], "GRIPPER_FEEDBACK_OUT_OF_RANGE")
        if any(abs(value - segment["gripper_position_m"]) > 1e-9 for value in (reference, wire["raw_reference_m"])):
            raise ContractError("GRIPPER_FEEDBACK_OUT_OF_RANGE")
        if wire["completed_generation"] != wire["generation"]:
            return False
        check_segment_observation(segment, evidence, now, steady_now=steady, terminal=True)
        return True

    def cancel_active(self, cancel_timeout_s):
        """Cancel the active action, retaining non-cancel results for evidence."""
        active = getattr(self, "_active", None)
        if active is None:
            raise ContractError("ROS_EXEC_NO_ACTIVE")
        if (
            isinstance(cancel_timeout_s, bool)
            or not isinstance(cancel_timeout_s, (int, float))
            or not math.isfinite(cancel_timeout_s)
            or cancel_timeout_s <= 0
        ):
            raise ContractError("ROS_EXEC_CANCEL_TIMEOUT")
        self._execution_locked = True
        if isinstance(active, ArmStream):
            # The stream's owner polls native cancellation/terminal evidence;
            # never translate a fence or cancel acknowledgement into a stop.
            active.cancel()
            raise ContractError("ROS_EXEC_CANCEL_UNCERTAIN")
        if getattr(active, "action_succeeded", False):
            # The action is already terminal. A cancel request fences the handoff but
            # cannot manufacture CANCELED or stop an in-flight hardware RPC.
            # Keep the actual terminal result for poll_terminal_evidence.
            raise ContractError("ROS_EXEC_CANCEL_NOT_CANCELED")
        try:
            if active.goal_handle is None:
                if active.goal_future is None:
                    raise ContractError("ROS_EXEC_CANCEL_ACK_TIMEOUT")
                handle = self._wait(
                    active.goal_future, float(cancel_timeout_s),
                    "ROS_EXEC_CANCEL_ACK_TIMEOUT",
                )
                accepted = getattr(handle, "accepted", None)
                if accepted is False:
                    self._active = None
                    return active
                if accepted is not True:
                    raise ContractError("ROS_EXEC_GOAL_RESPONSE_INVALID")
                active.goal_handle = handle
                active.result_future = handle.get_result_async()
            canceled = active.goal_handle.cancel_goal_async()
            response = self._wait(canceled, float(cancel_timeout_s), "ROS_EXEC_CANCEL_ACK_TIMEOUT")
            if not response.goals_canceling:
                raise ContractError("ROS_EXEC_CANCEL_REJECTED")
            result = self._wait(
                active.result_future, float(cancel_timeout_s), "ROS_EXEC_CANCEL_RESULT_TIMEOUT"
            )
        except ContractError:
            raise
        except RuntimeError as exc:
            raise ContractError("ROS_EXEC_CANCEL_FAILED", str(exc)) from exc
        if result.status != self._goal_canceled:
            raise ContractError("ROS_EXEC_CANCEL_NOT_CANCELED")
        self._active = None
        return active

    @property
    def owns_active_goal(self):
        """Report ownership of an action, native handoff or unresolved cancellation."""
        return getattr(self, "_active", None) is not None

    def poll_terminal_evidence(self):
        """Observe an uncertain action without retrying or cancelling it again."""
        active = getattr(self, "_active", None)
        if active is None:
            raise ContractError("ROS_EXEC_NO_ACTIVE")
        if isinstance(active, ArmStream):
            raise ContractError("ROS_EXEC_STREAM_OWNER_REQUIRED")
        if getattr(active, "action_succeeded", False) and not self._execution_locked:
            # A normal hardware handoff is still owned; diagnostic polling must
            # not release it early. After cancellation, report the actual JTC result.
            return None
        try:
            if active.goal_handle is None:
                if active.goal_future is None:
                    return None
                if not active.goal_future.done():
                    self._rclpy.spin_once(self.node, timeout_sec=0.0)
                if not active.goal_future.done():
                    return None
                handle = active.goal_future.result()
                accepted = getattr(handle, "accepted", None)
                if accepted is False:
                    self._active = None
                    return {
                        "schema_version": "data_factory.ros_action_terminal_evidence.v1",
                        "terminal": True,
                        "phase": active.phase,
                        "type": active.type,
                        "goal_acceptance": "REJECTED",
                        "result_status": None,
                    }
                if accepted is not True:
                    return None
                active.goal_handle = handle
                active.result_future = handle.get_result_async()
            if active.result_future is None:
                return None
            if not active.result_future.done():
                self._rclpy.spin_once(self.node, timeout_sec=0.0)
            if not active.result_future.done():
                return None
            result = active.result_future.result()
            status = getattr(result, "status", None)
            if type(status) is not int or status not in {
                self._goal_succeeded, self._goal_canceled, self._goal_aborted,
            }:
                return None
        except Exception:
            return None
        self._active = None
        return {
            "schema_version": "data_factory.ros_action_terminal_evidence.v1",
            "terminal": True,
            "phase": active.phase,
            "type": active.type,
            "goal_acceptance": "ACCEPTED",
            "result_status": status,
        }

    def _fresh(self, received_at, max_age_s, code):
        if received_at is None:
            raise ContractError(code)
        age = self._clock() - received_at
        if age < 0 or age > max_age_s:
            raise ContractError(code)
        return age

    def capture_policy_observation(self, camera_topics, max_age_s):
        """Read one bounded observation on the sole transport node; send no goals.

        Wire images use hex only for JSON transport. Original source stamps are
        retained; a new callback cannot renew a paused publisher's timestamp.
        """
        from sensor_msgs.msg import Image
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.validate_full_topic_name import validate_full_topic_name
        from rclpy.exceptions import InvalidTopicNameException
        from tools.ros_image import image_message_to_rgb
        if (not isinstance(camera_topics, dict) or set(camera_topics) != {"camera1", "camera2"}
                or any(not isinstance(t, str) or not t.startswith("/") or t == "/"
                       for t in camera_topics.values())
                or len(set(camera_topics.values())) != 2):
            raise ContractError("LEARNED_CAMERA_TOPICS")
        try:
            for topic in camera_topics.values():
                validate_full_topic_name(topic)
        except InvalidTopicNameException as exc:
            raise ContractError("LEARNED_CAMERA_TOPICS") from exc
        if (isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float))
                or not math.isfinite(max_age_s) or not 0 < max_age_s <= 5):
            raise ContractError("LEARNED_SOURCE_CLOCK")
        if self._execution_locked or self._active is not None:
            raise ContractError("ROS_EXEC_ACTIVE")
        if self.node.get_parameter("use_sim_time").value is not False:
            raise ContractError("LEARNED_SOURCE_CLOCK")
        started, started_system = self._clock(), time.time()
        previous_state = self._joint_state
        deadline = time.monotonic() + self.graph_timeout_s
        frames, subscriptions = {}, []
        def receive(name, message):
            frames[name] = (message, self._clock())
        try:
            for name, topic in camera_topics.items():
                subscriptions.append(self.node.create_subscription(
                    Image, topic, lambda message, name=name: receive(name, message), qos_profile_sensor_data))
            while time.monotonic() < deadline:
                self._rclpy.spin_once(self.node, timeout_sec=max(0., min(.05, deadline - time.monotonic())))
                if (len(frames) == 2 and self._joint_state_received_at is not None
                        and self._joint_state is not previous_state
                        and self._joint_state_received_at >= started):
                    break
            if (len(frames) != 2 or self._joint_state_received_at is None
                    or self._joint_state is previous_state
                    or self._joint_state_received_at < started):
                raise ContractError("LEARNED_OBSERVATION_UNAVAILABLE")
            samples = {"state": (self._joint_state, self._joint_state_received_at), **frames}
            now, steady = time.time(), self._clock()
            if steady < started or abs((now - started_system) - (steady - started)) > max_age_s:
                raise ContractError("LEARNED_SOURCE_CLOCK")
            stamps = {}
            for name, (message, received) in samples.items():
                stamp = message.header.stamp
                if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                    raise ContractError("LEARNED_SOURCE_CLOCK")
                source = stamp.sec + stamp.nanosec / 1e9
                if not 0 <= now - source <= max_age_s or not 0 <= steady - received <= max_age_s:
                    raise ContractError("LEARNED_STALE_OBSERVATION")
                stamps[name] = source
            from tools.data_factory.rollout.finite_plan import JOINTS
            names, positions = list(samples["state"][0].name), list(samples["state"][0].position)
            if (len(names) != len(set(names)) or len(names) != len(positions)
                    or not set(JOINTS).issubset(names)
                    or any(not math.isfinite(value) for value in positions)):
                raise ContractError("ROS_JOINT_STATE")
            by_name = dict(zip(names, positions))
            observation = {"source_clock": "SYSTEM_TIME", "source_timestamps_s": stamps,
                           "observation.state": [by_name[name] for name in JOINTS]}
            for name, (message, _) in frames.items():
                try:
                    rgb = image_message_to_rgb(message)
                except (ValueError, TypeError) as exc:
                    raise ContractError("LEARNED_IMAGE", str(exc)) from exc
                observation[f"observation.images.{name}"] = {
                    "dtype": "uint8", "color_space": "RGB", "shape": list(rgb.shape),
                    "data_hex": rgb.tobytes().hex()}
            # Conversion time also consumes the source freshness budget.
            from tools.data_factory.rollout.finite_plan import check_freshness
            finished, finished_steady = time.time(), self._clock()
            check_freshness({"source_timestamps_s": stamps, "max_observation_age_s": max_age_s}, finished)
            if abs((finished - started_system) - (finished_steady - started)) > max_age_s:
                raise ContractError("LEARNED_SOURCE_CLOCK")
            if any(not 0 <= self._clock() - received <= max_age_s for _, received in samples.values()):
                raise ContractError("LEARNED_STALE_OBSERVATION")
            if self.node.get_parameter("use_sim_time").value is not False:
                raise ContractError("LEARNED_SOURCE_CLOCK")
            if self._execution_locked or self._active is not None:
                raise ContractError("ROS_EXEC_ACTIVE")
            return observation
        finally:
            for subscription in subscriptions:
                self.node.destroy_subscription(subscription)

    def _source_samples_fresh(self, max_age_s):
        """A just-dispatched DDS callback does not renew its source timestamp."""
        now = time.time()
        for message in (self._joint_state, self._arm_controller_state, self._gripper_controller_state):
            if message is None:
                return False
            stamp = message.header.stamp
            if (type(stamp.sec) is not int or type(stamp.nanosec) is not int
                    or not 0 <= stamp.sec < 2**31 or not 0 <= stamp.nanosec < 10**9
                    or not 0 <= now - (stamp.sec * 1_000_000_000 + stamp.nanosec) / 1e9 <= max_age_s):
                return False
        return True

    def snapshot(self, max_age_s):
        """Return a fresh, complete observation for execution safety checks."""
        if (
            isinstance(max_age_s, bool)
            or not isinstance(max_age_s, (int, float))
            or not math.isfinite(max_age_s)
            or max_age_s < 0
        ):
            raise ContractError("ROS_SNAPSHOT_AGE")
        max_age_s = float(max_age_s)
        self._prepare_native_clock(max_age_s)
        # The configured native LIVE path already requires SYSTEM_TIME stamps
        # at learned admission. Drain queued old-source samples within the same
        # acquisition deadline; legacy unconfigured readers retain their contract.
        native = getattr(self, "_native_clock_configured_age", None) is not None
        # New DDS participants need discovery time, not a relaxed sample age.
        # After one complete snapshot, retain the short live observation budget.
        timeout = (
            self.graph_timeout_s if self._initial_snapshot_complete
            else self.preflight_timeout_s
        )
        deadline = time.monotonic() + timeout
        self._load_robot_description_parameter(deadline)
        while (
            self._joint_state_received_at is None
            or self._arm_controller_received_at is None
            or self._gripper_controller_received_at is None
            or self._robot_description is None
            or (native and (not self._native_current_ready(max_age_s) or not self._source_samples_fresh(max_age_s)))
            or any(
                self._clock() - received_at > max_age_s
                for received_at in (
                    self._joint_state_received_at,
                    self._arm_controller_received_at,
                    self._gripper_controller_received_at,
                )
            )
        ) and time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())))
        joint_age = self._fresh(self._joint_state_received_at, max_age_s, "ROS_JOINT_STATE_STALE")
        arm_age = self._fresh(self._arm_controller_received_at, max_age_s, "ROS_ARM_CONTROLLER_STALE")
        gripper_age = self._fresh(self._gripper_controller_received_at, max_age_s, "ROS_GRIPPER_CONTROLLER_STALE")
        names = list(self._joint_state.name)
        positions = list(self._joint_state.position)
        if len(names) != len(set(names)) or not set(JOINT_ORDER).issubset(names) or len(positions) != len(names):
            raise ContractError("ROS_JOINT_STATE")
        by_name = dict(zip(names, positions))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in by_name.values()):
            raise ContractError("ROS_JOINT_STATE")
        try:
            arm_publishers = self.node.count_publishers("/fairino5_controller/controller_state")
            gripper_publishers = self.node.count_publishers("/gripper_controller/controller_state")
            topics = dict(self.node.get_topic_names_and_types())
        except RuntimeError as exc:
            raise ContractError("ROS_GRAPH_FAILED", str(exc)) from exc
        controller_type = "control_msgs/msg/JointTrajectoryControllerState"
        arm_type = topics.get("/fairino5_controller/controller_state", [])
        gripper_type = topics.get("/gripper_controller/controller_state", [])
        arm_values = self._controller_values(self._arm_controller_state, "ROS_ARM_CONTROLLER_STATE")
        gripper_values = self._controller_values(
            self._gripper_controller_state, "ROS_GRIPPER_CONTROLLER_STATE"
        )
        if "finger_right_joint" not in gripper_values["reference"]:
            raise ContractError("ROS_GRIPPER_CONTROLLER_STATE")
        arm_speed = self._arm_controller_state.speed_scaling_factor
        if not isinstance(arm_speed, (int, float)) or not math.isfinite(arm_speed):
            raise ContractError("ROS_ARM_CONTROLLER_STATE")
        gripper_speed = self._gripper_controller_state.speed_scaling_factor
        if not isinstance(gripper_speed, (int, float)) or not math.isfinite(gripper_speed):
            raise ContractError("ROS_GRIPPER_CONTROLLER_STATE")
        hardware = self._gripper_hardware_evidence()
        stamp = self._joint_state.header.stamp
        if (type(stamp.sec) is not int or type(stamp.nanosec) is not int
                or not 0 <= stamp.sec < 2**31 or not 0 <= stamp.nanosec < 10**9):
            raise ContractError("ROS_JOINT_STATE")
        observation = {
            "joint_positions": [by_name[name] for name in JOINT_ORDER],
            "joint_state_stamp_ns": stamp.sec * 1_000_000_000 + stamp.nanosec,
            "joint_state_age_s": joint_age,
            "gripper_settings": self._gripper_settings(),
            "arm_controller": {
                "endpoint": "/fairino5_controller/controller_state",
                "type": controller_type if arm_type == [controller_type] else "|".join(sorted(arm_type)),
                "publisher_count": arm_publishers,
                "ready": arm_type == [controller_type] and arm_publishers > 0,
                "age_s": arm_age,
                "speed_scaling": float(arm_speed),
                "sample": arm_values["sample"],
            },
            "gripper_controller": {
                "endpoint": "/gripper_controller/controller_state",
                "type": controller_type if gripper_type == [controller_type] else "|".join(sorted(gripper_type)),
                "publisher_count": gripper_publishers,
                "ready": gripper_type == [controller_type] and gripper_publishers > 0,
                "age_s": gripper_age,
                "speed_scaling": float(gripper_speed),
                "reference_position_m": gripper_values["reference"]["finger_right_joint"],
                "feedback_position_m": gripper_values["feedback"]["finger_right_joint"],
                "sample": gripper_values["sample"],
                **({"hardware_execution": hardware} if hardware is not None else {}),
            },
        }
        if hardware is not None and hardware["wire"]["version"] == 5:
            from tools.data_factory.rollout.gripper_evidence import check_delivery
            wire = hardware["wire"]
            check_delivery(wire, hardware["clock_binding"], self._clock(), max_age_s)
            # Controller reference and lifecycle samples remain independently checked;
            # measured coordinates all come from this one native SDK publication.
            observation["joint_positions"] = [wire[f"arm_j{i}_rad"] for i in range(1, 7)]
            observation["joint_state_age_s"] = self._clock() - wire["host_receive_steady_s"]
            observation["gripper_controller"]["feedback_position_m"] = wire["feedback_m"]
        for key in ("arm_controller", "gripper_controller"):
            validate_controller_sample(observation[key])
        if native and not self._source_samples_fresh(max_age_s):
            raise ContractError("LEARNED_STALE_STATE")
        self._initial_snapshot_complete = True
        return observation

    def _load_robot_description_parameter(self, deadline):
        client = self._robot_description_client
        if self._robot_description is not None or client is None:
            return
        parameter_future = None
        parameter_finished = False
        while self._robot_description is None and time.monotonic() < deadline:
            if not parameter_finished and parameter_future is None:
                try:
                    if client.wait_for_services(timeout_sec=0.0):
                        parameter_future = client.get_parameters(["robot_description"])
                except (RuntimeError, TimeoutError):
                    pass
            if parameter_future is not None and parameter_future.done():
                parameter_finished = True
                try:
                    response = parameter_future.result()
                except (RuntimeError, TimeoutError):
                    response = None
                    parameter_finished = False
                parameter_future = None
                values = getattr(response, "values", None)
                candidate = (
                    values[0].string_value
                    if isinstance(values, (list, tuple))
                    and len(values) == 1
                    and values[0].type == self._parameter_string
                    and isinstance(values[0].string_value, str)
                    and values[0].string_value
                    else None
                )
                if candidate is not None:
                    try:
                        self._parse_gripper_settings(candidate)
                    except ContractError:
                        pass
                    else:
                        self._robot_description = candidate
                        return
            self._rclpy.spin_once(
                self.node,
                timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())),
            )
            if self._robot_description is not None:
                try:
                    self._parse_gripper_settings(self._robot_description)
                except ContractError:
                    self._robot_description = None
                else:
                    return

    def _gripper_settings(self):
        return self._parse_gripper_settings(self._robot_description)

    @staticmethod
    def _parse_gripper_settings(robot_description):
        try:
            root = ET.fromstring(robot_description)
        except (TypeError, ET.ParseError) as exc:
            raise ContractError("ROS_GRIPPER_SETTINGS_UNVERIFIED", str(exc)) from exc
        blocks = []
        for control in root.findall(".//ros2_control"):
            if control.find("./joint[@name='finger_right_joint']") is None:
                continue
            hardware = control.find("hardware")
            if hardware is not None:
                blocks.append(hardware)
        if len(blocks) != 1:
            raise ContractError("ROS_GRIPPER_SETTINGS_UNVERIFIED")
        plugin = (blocks[0].findtext("plugin") or "").strip()
        params = {
            item.get("name"): (item.text or "").strip()
            for item in blocks[0].findall("param")
        }
        try:
            settings = {
                "hardware_plugin": plugin,
                "velocity_percent": int(params["gripper_velocity"]),
                "open_velocity_percent": int(params.get(
                    "gripper_open_velocity", params["gripper_velocity"],
                )),
                "force_percent": int(params["gripper_force"]),
                "open_force_percent": int(params.get(
                    "gripper_open_force", params["gripper_force"],
                )),
                "settle_time_ms": int(params["gripper_settle_time_ms"]),
            }
        except (KeyError, ValueError) as exc:
            raise ContractError("ROS_GRIPPER_SETTINGS_UNVERIFIED", str(exc)) from exc
        if (
            plugin not in {
                "fairino_hardware/FairinoHardwareInterface",
                "mock_components/GenericSystem",
            }
            or not 1 <= settings["velocity_percent"] <= 100
            or not 1 <= settings["open_velocity_percent"] <= 100
            or not 1 <= settings["force_percent"] <= 100
            or not 1 <= settings["open_force_percent"] <= 100
            or not 50 <= settings["settle_time_ms"] <= 10000
        ):
            raise ContractError("ROS_GRIPPER_SETTINGS_UNVERIFIED")
        return settings

    @staticmethod
    def _controller_values(message, code):
        names = list(message.joint_names)
        if not names or len(names) != len(set(names)):
            raise ContractError(code)
        result = {}
        for label, point in (("reference", message.reference), ("feedback", message.feedback)):
            positions = list(point.positions)
            if len(positions) != len(names) or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in positions
            ):
                raise ContractError(code)
            result[label] = {name: float(value) for name, value in zip(names, positions)}
        def nanoseconds(stamp, *, signed=False):
            if (type(stamp.sec) is not int or type(stamp.nanosec) is not int
                    or not (-2**31 if signed else 0) <= stamp.sec < 2**31
                    or not 0 <= stamp.nanosec < 10**9):
                raise ContractError("ROS_CONTROLLER_SAMPLE")
            return stamp.sec * 10**9 + stamp.nanosec
        try:
            # ROS time is retained in its own domain, never relabeled SYSTEM_TIME.
            # JTC may retain an old output when reading command interfaces fails;
            # this is reported output, not a fresh hardware acknowledgement.
            result["sample"] = {
                "joint_names": names,
                "ros_stamp_ns": nanoseconds(message.header.stamp),
                "reference_elapsed_ns": nanoseconds(message.reference.time_from_start, signed=True),
                "feedback_elapsed_ns": nanoseconds(message.feedback.time_from_start, signed=True),
                "reference_positions": list(message.reference.positions),
                "feedback_positions": list(message.feedback.positions),
                "reported_output_positions": list(message.output.positions),
            }
            validate_controller_sample({"sample": result["sample"]})
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractError("ROS_CONTROLLER_SAMPLE") from exc
        return result

    def preflight(self):
        clients = {
            "/move_action": self.move_group,
            "/execute_trajectory": self.execute_trajectory,
            "/gripper_controller/follow_joint_trajectory": self.gripper,
        }
        deadline = time.monotonic() + self.preflight_timeout_s
        server_ready = {endpoint: False for endpoint in clients}
        while True:
            try:
                actions = dict(self._get_action_names_and_types(self.node))
                topics = dict(self.node.get_topic_names_and_types())
                server_ready = {
                    endpoint: bool(client.server_is_ready())
                    for endpoint, client in clients.items()
                }
            except RuntimeError as exc:
                raise ContractError("ROS_GRAPH_FAILED", str(exc)) from exc
            joint_sample_ready = self._joint_state is not None
            graph_ready = (
                all(actions.get(endpoint) == [kind] for endpoint, kind in ACTION_TYPES.items())
                and all(server_ready.values())
                and topics.get("/joint_states") == ["sensor_msgs/msg/JointState"]
                and joint_sample_ready
            )
            if graph_ready or time.monotonic() >= deadline:
                break
            self._rclpy.spin_once(
                self.node,
                timeout_sec=max(0.0, min(0.05, deadline - time.monotonic())),
            )

        facts = {}
        fact_keys = {
            "/move_action": "move_action",
            "/execute_trajectory": "execute_trajectory",
            "/gripper_controller/follow_joint_trajectory": "gripper",
        }
        for endpoint, expected_type in ACTION_TYPES.items():
            observed = actions.get(endpoint, [])
            exact_type = observed == [expected_type]
            facts[fact_keys[endpoint]] = {
                "endpoint": endpoint,
                "type": expected_type if exact_type else "|".join(sorted(observed)),
                "ready": exact_type and server_ready[endpoint],
            }

        joint_types = topics.get("/joint_states", [])
        exact_joint_type = joint_types == ["sensor_msgs/msg/JointState"]
        facts["joint_states"] = {
            "endpoint": "/joint_states",
            "type": (
                "sensor_msgs/msg/JointState"
                if exact_joint_type
                else "|".join(sorted(joint_types))
            ),
            "ready": exact_joint_type and joint_sample_ready,
        }
        facts["joint_order"] = list(JOINT_ORDER)
        return facts

    def _duration(self, seconds):
        nanoseconds = round(seconds * 1_000_000_000)
        return self._Duration(
            sec=nanoseconds // 1_000_000_000,
            nanosec=nanoseconds % 1_000_000_000,
        )

    def _wait(self, future, timeout_s, code):
        try:
            self._rclpy.spin_until_future_complete(
                self.node, future, timeout_sec=timeout_s
            )
            if not future.done():
                raise ContractError(code)
            return future.result()
        except ContractError:
            raise
        except (RuntimeError, TimeoutError) as exc:
            raise ContractError(code, str(exc)) from exc

    def _cancel_planning(self, goal_handle):
        try:
            future = goal_handle.cancel_goal_async()
        except RuntimeError as exc:
            raise ContractError("ROS_PLAN_CANCEL_FAILED", str(exc)) from exc
        response = self._wait(
            future, self.graph_timeout_s, "ROS_PLAN_CANCEL_TIMEOUT"
        )
        if not response.goals_canceling:
            raise ContractError("ROS_PLAN_CANCEL_REJECTED")

    def _check_plan_collision(self, plan, initial_gripper):
        request_type = self._GetStateValidity.Request
        gripper = float(initial_gripper)
        samples, failures = [], []
        feedback_bounds = None

        def check(label, joints):
            if len(joints) != len(JOINT_ORDER) or any(not math.isfinite(value) for value in joints):
                raise ContractError("COLLISION_STATE")
            positions = [gripper] if feedback_bounds is None else sorted({gripper, *feedback_bounds.values()})
            for position in positions:
                request = request_type()
                request.group_name = "" if "learned_proposal" in plan or plan.get("mechanical_terminal") else plan["frames"]["planning_group"]
                request.robot_state = self._RobotState(
                    joint_state=self._JointState(
                        name=[*JOINT_ORDER, "finger_right_joint"],
                        position=[*map(float, joints), float(position)],
                    )
                )
                if plan.get("mechanical_terminal"):
                    request.robot_state.is_diff = True
                response = self._service(self._GetStateValidity, "/check_state_validity", request, "COLLISION_SERVICE")
                constraints = list(getattr(response, "constraint_result", []))
                constraints_valid = all(c.result is True and math.isfinite(c.distance) and c.distance == 0 for c in constraints)
                valid = getattr(response, "valid", False) is True and constraints_valid
                if not valid and plan.get("contact_touch_object"):
                    from .contact_transition import TIPS
                    obj = plan["contact_touch_object"]
                    contacts = list(getattr(response, "contacts", []))
                    # MoveIt 2.12 state validation requests (world+collision
                    # links)^2 contacts. A saturated response is not complete
                    # evidence that all collisions are intended contacts.
                    model = ET.fromstring(self._robot_description)
                    limit = (3 + sum(bool(link.findall("collision")) for link in model.findall("link"))) ** 2
                    def intended(c):
                        bodies = {(c.contact_body_1, c.body_type_1), (c.contact_body_2, c.body_type_2)}
                        values = [c.depth, c.position.x,c.position.y,c.position.z,c.normal.x,c.normal.y,c.normal.z]
                        return (any(bodies == {(obj, c.WORLD_OBJECT), (tip, c.ROBOT_LINK)} for tip in TIPS)
                            and c.header.frame_id == plan["frames"]["planning_frame"]
                            and all(math.isfinite(v) for v in values) and c.depth >= 0
                            and sum(v*v for v in values[4:]) > 0)
                    valid = (constraints_valid and hasattr(response, "constraint_result")
                             and 0 < len(contacts) < limit and all(intended(c) for c in contacts))
                evidence = {"label": label, "joints_rad": list(map(float, joints)), "finger_right_joint_m": float(position), "valid": valid}
                if plan.get("contact_touch_object"):
                    evidence["native_valid"] = getattr(response, "valid", False) is True
                    evidence["contacts"] = [{"body_1": c.contact_body_1, "type_1": c.body_type_1,
                        "body_2": c.contact_body_2, "type_2": c.body_type_2, "depth_m": c.depth}
                        for c in getattr(response, "contacts", [])]
                samples.append(evidence)
                if not evidence["valid"]:
                    failures.append(evidence)

        check("initial", plan["initial_joint_state"])
        for step in (segment for outer in plan["steps"] for segment in outer.get("held_target_segments", [outer])):
            feedback_bounds = step.get("acceptable_feedback_m") if step["type"] == "ARM" else None
            if feedback_bounds is not None and "native_close_feedback" in step:
                calibrated = step["native_close_feedback"]["requirements"]["acceptable_feedback_m"]
                # Collision coverage includes either possible feedback envelope;
                # only the live checker can admit a completed native selection.
                feedback_bounds = {"min": min(feedback_bounds["min"], calibrated["min"]),
                                   "max": max(feedback_bounds["max"], calibrated["max"])}
            if step["type"] == "GRIPPER":
                target = step.get("gripper_position_m", plan["gripper_requirements"]["command_position_m"])
                targets = ([step["release_position_m"],target]
                           if plan.get("mechanical_terminal") and "release_position_m" in step else [target])
                for target in targets:
                    if "action_range" in step or plan.get("mechanical_terminal"):
                        start = gripper
                        for part in range(1, 5):
                            gripper = start + (target - start) * part / 5
                            check(f"{step['phase']}:gripper:{part}", step["final_joint_state"])
                    gripper = target
                    check(step["phase"], step["final_joint_state"])
                continue
            try:
                trajectory = self._deserialize_message(
                    base64.b64decode(step["trajectory_b64"], validate=True), self._RobotTrajectory
                ).joint_trajectory
                names, points = list(trajectory.joint_names), list(trajectory.points)
            except (ValueError, RuntimeError, TypeError) as exc:
                raise ContractError("COLLISION_TRAJECTORY", str(exc)) from exc
            learned = step["phase"] == "LEARNED_CHUNK" and "action_range" not in step
            expected_names = [*JOINT_ORDER, "finger_right_joint"] if learned else JOINT_ORDER
            if names != expected_names or not points:
                raise ContractError("COLLISION_TRAJECTORY")
            previous = [*step["start_joint_state"], gripper] if learned else step["start_joint_state"]
            previous_time = -1.0
            for index, point in enumerate(points):
                current = list(point.positions)
                seconds = point.time_from_start.sec + point.time_from_start.nanosec / 1e9
                if len(current) != len(expected_names) or seconds <= previous_time:
                    raise ContractError("COLLISION_TRAJECTORY")
                for part in range(1, 5):
                    ratio = part / 5
                    interpolated = [a + (b - a) * ratio for a, b in zip(previous, current)]
                    if learned:
                        gripper = interpolated[-1]
                    check(f"{step['phase']}:interp:{index}:{part}", interpolated[:6])
                if learned:
                    gripper = current[-1]
                check(f"{step['phase']}:point:{index}", current[:6])
                previous, previous_time = current, seconds
        report = {"schema_version": "data_factory.collision_report.v1", "plan_digest": canonical_digest(plan), "sample_count": len(samples), "samples": samples, "failure_count": len(failures), "all_valid": not failures}
        if failures:
            raise ContractError("COLLISION_DETECTED")
        return report

    def precommit_safety(self, plan, planning_scene, before_snapshot):
        """Prove scene/readback, serialized-plan collision, and plan-only no-motion."""
        if not isinstance(plan, dict) or not isinstance(before_snapshot, dict):
            raise ContractError("PRECOMMIT_SAFETY_SCHEMA")
        expected = plan["binding_digests"]["planning_scene_digest"]
        readback = self._apply_and_readback_scene(planning_scene, expected)
        try:
            before_gripper = before_snapshot["gripper_controller"]
            initial_gripper = float(before_gripper["feedback_position_m"])
            initial_reference = float(before_gripper["reference_position_m"])
            if "learned_proposal" in plan:
                open_target = float(plan["learned_proposal"]["initial_state"][-1])
                gripper_tolerance = float(plan["steps"][0]["gripper_tolerance_m"])
            else:
                open_step = next(step for step in plan["steps"] if step["phase"] == "GRIPPER_OPEN")
                open_target = float(open_step["gripper_position_m"])
                gripper_tolerance = float(open_step["limits"]["completion_tolerance_m"])
            if abs(initial_gripper - open_target) > gripper_tolerance or abs(initial_reference - open_target) > gripper_tolerance:
                raise ContractError("GRIPPER_INITIAL_NOT_OPEN")
            collision = self._check_plan_collision(plan, initial_gripper)
            after_snapshot = self.snapshot(plan["planning"]["max_joint_state_age_s"])
            before_joints, after_joints = before_snapshot["joint_positions"], after_snapshot["joint_positions"]
            joint_delta = max(abs(a - b) for a, b in zip(before_joints, after_joints))
            gripper_delta = abs(
                float(before_snapshot["gripper_controller"]["feedback_position_m"])
                - float(after_snapshot["gripper_controller"]["feedback_position_m"])
            )
            tolerance = float(plan["planning"]["goal_tolerances"]["joint_rad"])
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError, StopIteration) as exc:
            raise ContractError("PRECOMMIT_SAFETY_SCHEMA", str(exc)) from exc
        plan_digest = canonical_digest(plan)
        readback_payload = {
            "schema_version": "data_factory.planning_scene_readback.v1",
            "run_id": plan["run_id"],
            "plan_digest": plan_digest,
            "expected_planning_scene_digest": expected,
            "objects": readback,
        }
        no_motion = {
            "schema_version": "data_factory.plan_only_no_motion.v1",
            "run_id": plan["run_id"],
            "plan_digest": plan_digest,
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot,
            "max_joint_delta_rad": joint_delta,
            "gripper_delta_m": gripper_delta,
            "execute_goal_count": self._execute_goal_count,
            "gripper_goal_count": self._gripper_goal_count,
        }
        if joint_delta > tolerance or gripper_delta > gripper_tolerance or self._execute_goal_count or self._gripper_goal_count:
            raise ContractError("PLAN_ONLY_MOVED_ROBOT")
        safety = {
            "schema_version": "data_factory.precommit_safety.v1",
            "run_id": plan["run_id"],
            "approved_plan_digest": plan_digest,
            "scene_binding_digest": canonical_digest(plan["scene_binding"]),
            "expected_planning_scene_digest": expected,
            "planning_scene_readback_digest": canonical_digest(readback_payload),
            "collision_report_digest": canonical_digest(collision),
            "plan_only_no_motion_digest": canonical_digest(no_motion),
            "post_reset_safe_snapshot_digest": None,
            "status": "PENDING",
        }
        return {
            "precommit_safety": safety,
            "precommit_evidence": {
                "schema_version": "data_factory.precommit_evidence.v1",
                "run_id": plan["run_id"],
                "approved_plan_digest": plan_digest,
                "scene_binding_digest": canonical_digest(plan["scene_binding"]),
                "expected_planning_scene_digest": expected,
                "planning_scene_readback": readback_payload,
                "collision_report": collision,
                "plan_only_no_motion": no_motion,
            },
        }

    def precommit_joint_transition(
        self, *, serialized_trajectory, start_joint_state, final_joint_state,
        planning_scene, planning_scene_digest, planning_group,
        max_joint_state_age_s, joint_tolerance_rad, gripper_tolerance_m,
        before_snapshot,
    ):
        """Check one exact joint-target trajectory without issuing a goal."""
        numeric = (
            [
                *start_joint_state, *final_joint_state,
                max_joint_state_age_s, joint_tolerance_rad, gripper_tolerance_m,
            ]
            if isinstance(start_joint_state, list)
            and isinstance(final_joint_state, list)
            else []
        )
        if (
            not isinstance(serialized_trajectory, bytes)
            or not serialized_trajectory
            or not isinstance(start_joint_state, list)
            or not isinstance(final_joint_state, list)
            or len(start_joint_state) != len(JOINT_ORDER)
            or len(final_joint_state) != len(JOINT_ORDER)
            or not isinstance(planning_group, str)
            or not planning_group
            or not isinstance(before_snapshot, dict)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in numeric
            )
            or max_joint_state_age_s <= 0
            or joint_tolerance_rad <= 0
            or gripper_tolerance_m <= 0
        ):
            raise ContractError("JOINT_TRANSITION_PRECOMMIT")
        execute_goals_before = self._execute_goal_count
        gripper_goals_before = self._gripper_goal_count
        readback = self._apply_and_readback_scene(
            planning_scene, planning_scene_digest,
        )
        try:
            initial_gripper = float(
                before_snapshot["gripper_controller"]["feedback_position_m"]
            )
            plan = {
                "frames": {"planning_group": planning_group},
                "initial_joint_state": list(map(float, start_joint_state)),
                "steps": [{
                    "phase": "SAFE_POSE_PTP",
                    "type": "ARM",
                    "trajectory_b64": base64.b64encode(
                        serialized_trajectory
                    ).decode("ascii"),
                    "start_joint_state": list(map(float, start_joint_state)),
                    "final_joint_state": list(map(float, final_joint_state)),
                }],
            }
            collision = self._check_plan_collision(plan, initial_gripper)
            after_snapshot = self.snapshot(max_joint_state_age_s)
            joint_delta = max(abs(a - b) for a, b in zip(
                before_snapshot["joint_positions"],
                after_snapshot["joint_positions"],
            ))
            gripper_delta = abs(
                float(before_snapshot["gripper_controller"]["feedback_position_m"])
                - float(after_snapshot["gripper_controller"]["feedback_position_m"])
            )
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("JOINT_TRANSITION_PRECOMMIT", str(exc)) from exc
        if (
            joint_delta > joint_tolerance_rad
            or gripper_delta > gripper_tolerance_m
            or self._execute_goal_count != execute_goals_before
            or self._gripper_goal_count != gripper_goals_before
        ):
            raise ContractError("JOINT_TRANSITION_PLAN_ONLY_MOVED")
        evidence = {
            "schema_version": "data_factory.joint_transition_precommit.v1",
            "planning_scene_digest": planning_scene_digest,
            "planning_scene_readback": readback,
            "collision_report": collision,
            "before_snapshot": before_snapshot,
            "after_snapshot": after_snapshot,
            "max_joint_delta_rad": joint_delta,
            "gripper_delta_m": gripper_delta,
            "execute_goal_count": self._execute_goal_count,
            "gripper_goal_count": self._gripper_goal_count,
            "execute_goal_count_delta": self._execute_goal_count - execute_goals_before,
            "gripper_goal_count_delta": self._gripper_goal_count - gripper_goals_before,
        }
        evidence["evidence_digest"] = canonical_digest(evidence)
        return evidence

    def precommit_home_recovery(self, **kwargs):
        """Backward-compatible HOME evidence alias for the generic check."""
        try:
            evidence = self.precommit_joint_transition(**kwargs)
        except ContractError as exc:
            aliases = {
                "JOINT_TRANSITION_PRECOMMIT": "HOME_RECOVERY_PRECOMMIT",
                "JOINT_TRANSITION_PLAN_ONLY_MOVED":
                    "HOME_RECOVERY_PLAN_ONLY_MOVED",
            }
            if exc.code in aliases:
                raise ContractError(aliases[exc.code], str(exc)) from exc
            raise
        evidence["schema_version"] = "data_factory.home_recovery_precommit.v1"
        evidence.pop("evidence_digest")
        evidence["evidence_digest"] = canonical_digest(evidence)
        return evidence

    def plan_arm(
        self,
        phase,
        target,
        joint_target,
        limits,
        frames,
        planning,
        start_joint_state,
        *, planning_scene_diff=None,
    ):
        goal = self._move_group_goal(
            phase,
            target,
            joint_target,
            limits,
            frames,
            planning,
            start_joint_state,
        )
        if planning_scene_diff is not None:
            goal.planning_options.planning_scene_diff = copy.deepcopy(planning_scene_diff)
        try:
            send_future = self.move_group.send_goal_async(goal)
        except RuntimeError as exc:
            raise ContractError("ROS_PLAN_GOAL_FAILED", str(exc)) from exc
        handle = self._wait(
            send_future, self.graph_timeout_s, "ROS_PLAN_GOAL_TIMEOUT"
        )
        if handle is None or not handle.accepted:
            raise ContractError("ROS_PLAN_REJECTED")
        try:
            result = self._wait(
                handle.get_result_async(),
                limits["planning_timeout_s"] + self.graph_timeout_s,
                "ROS_PLAN_RESULT_TIMEOUT",
            )
        except ContractError as exc:
            if exc.code == "ROS_PLAN_RESULT_TIMEOUT":
                self._cancel_planning(handle)
            raise
        if (
            result.status != self._goal_succeeded
            or result.result.error_code.val != self._moveit_success
        ):
            raise ContractError("ROS_PLAN_FAILED")

        trajectory = result.result.planned_trajectory
        names = list(trajectory.joint_trajectory.joint_names)
        points = trajectory.joint_trajectory.points
        if len(names) != len(set(names)) or set(names) != set(JOINT_ORDER) or not points:
            raise ContractError("ROS_FINAL_JOINTS")
        positions = list(points[-1].positions)
        if len(positions) != len(names) or any(not math.isfinite(value) for value in positions):
            raise ContractError("ROS_FINAL_JOINTS")
        by_name = dict(zip(names, positions))
        serialized = self._serialize_message(trajectory)
        if not serialized:
            raise ContractError("ROS_PLAN_SERIALIZATION")
        return {
            "terminal_status": "SUCCEEDED",
            "moveit_success": True,
            "serialized_trajectory": serialized,
            "final_joint_state": [by_name[name] for name in JOINT_ORDER],
        }

    def build_learned_trajectory(self, proposal):
        """Serialize exact absolute 7D knots for the existing ExecuteTrajectory owner.

        No retiming, clipping, delta conversion or secondary gripper goal is used.
        Controller support for the combined trajectory needs physical qualification.
        """
        from tools.data_factory.rollout.finite_plan import validate_proposal, JOINTS, PROPOSAL_SCHEMA, trajectory_schema
        proposal = validate_proposal(proposal)
        if trajectory_schema(proposal) != PROPOSAL_SCHEMA:
            raise ContractError("LEARNED_HELD_SEGMENTS_REQUIRED")
        trajectory = self._RobotTrajectory()
        trajectory.joint_trajectory.joint_names = list(JOINTS)
        for index, row in enumerate([proposal["initial_state"], *proposal["actions"]]):
            point = self._JointTrajectoryPoint()
            point.positions = list(map(float, row))
            point.time_from_start = self._duration(index * proposal["period_s"])
            trajectory.joint_trajectory.points.append(point)
        return self._serialize_message(trajectory)

    def build_learned_segment(self, proposal, segment):
        """Serialize a frozen held target or six-joint slice, without rebasing."""
        from tools.data_factory.rollout.finite_plan import validate_proposal, HELD_PROPOSAL_SCHEMA, REFERENCE_PROPOSAL_SCHEMA, trajectory_schema
        p = validate_proposal(proposal)
        start, end = segment["action_range"]
        reference = trajectory_schema(p) == REFERENCE_PROPOSAL_SCHEMA
        if (trajectory_schema(p) not in {HELD_PROPOSAL_SCHEMA, REFERENCE_PROPOSAL_SCHEMA} or type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(p["actions"]) or start == len(p["actions"])
                or not (reference and segment["type"] == "ARM") and p["actions"][start][-1] != segment["gripper_position_m"]):
            raise ContractError("LEARNED_SEGMENT_BINDING")
        if segment["type"] == "GRIPPER":
            if start != end or segment["limits"] != segment["gripper_limits"]:
                raise ContractError("LEARNED_SEGMENT_BINDING")
            return self.build_gripper_goal("LEARNED_CHUNK", segment["gripper_position_m"], segment["limits"])
        if (segment["type"] != "ARM" or start == end
                or not reference and any(row[-1] != segment["gripper_position_m"] for row in p["actions"][start:end])):
            raise ContractError("LEARNED_SEGMENT_BINDING")
        if reference and (end != start + 1 or start > 0 and segment["gripper_position_m"] != p["actions"][start - 1][-1]):
            raise ContractError("LEARNED_SEGMENT_BINDING")
        trajectory = self._RobotTrajectory()
        trajectory.joint_trajectory.joint_names = list(JOINT_ORDER)
        initial = p["initial_state"] if start == 0 else p["actions"][start - 1]
        elapsed = 0.
        for index, row in enumerate([initial, *p["actions"][start:end]]):
            point = self._JointTrajectoryPoint()
            point.positions = list(map(float, row[:6]))
            if index:
                elapsed += (p["reference_timing"]["durations_s"][start + index - 1]
                            if "reference_timing" in p else p["period_s"])
            point.time_from_start = self._duration(elapsed)
            trajectory.joint_trajectory.points.append(point)
        return self._serialize_message(trajectory)

    def arm_trajectory_duration_s(self, serialized):
        """Return the approved trajectory's controller time horizon."""
        try:
            trajectory = self._deserialize_message(serialized, self._RobotTrajectory)
            points = trajectory.joint_trajectory.points
            end = points[-1].time_from_start
            duration = float(end.sec) + float(end.nanosec) / 1_000_000_000
        except (IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise ContractError("ROS_TRAJECTORY_DURATION", str(exc)) from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ContractError("ROS_TRAJECTORY_DURATION")
        return duration

    def _move_group_goal(
        self,
        phase,
        target,
        joint_target,
        limits,
        frames,
        planning,
        start_joint_state,
    ):
        goal = self._MoveGroup.Goal()
        request = goal.request
        request.pipeline_id = planning["pipeline_id"]
        request.planner_id = (
            planning["ptp_planner_id"]
            if phase.endswith("_PTP")
            else planning["lin_planner_id"]
        )
        request.group_name = frames["planning_group"]
        request.num_planning_attempts = 1
        request.allowed_planning_time = float(limits["planning_timeout_s"])
        request.max_velocity_scaling_factor = float(limits["velocity_scaling"])
        request.max_acceleration_scaling_factor = float(limits["acceleration_scaling"])
        request.start_state = self._RobotState(
            joint_state=self._JointState(
                name=list(JOINT_ORDER), position=list(map(float, start_joint_state))
            ),
            is_diff=True,
        )
        request.goal_constraints = [
            self._constraints(phase, target, joint_target, frames, planning)
        ]
        goal.planning_options.plan_only = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def _constraints(self, phase, target, joint_target, frames, planning):
        constraints = self._Constraints(name=phase)
        tolerances = planning["goal_tolerances"]
        if joint_target is not None:
            constraints.joint_constraints = [
                self._JointConstraint(
                    joint_name=name,
                    position=float(value),
                    tolerance_above=float(tolerances["joint_rad"]),
                    tolerance_below=float(tolerances["joint_rad"]),
                    weight=1.0,
                )
                for name, value in zip(JOINT_ORDER, joint_target)
            ]
            return constraints

        pose = target["base_tool"]
        primitive = self._SolidPrimitive(
            type=self._SolidPrimitive.SPHERE,
            dimensions=[float(tolerances["position_m"])],
        )
        region_pose = self._Pose()
        region_pose.position.x, region_pose.position.y, region_pose.position.z = (
            map(float, pose["translation_m"])
        )
        region_pose.orientation.w = 1.0
        position = self._PositionConstraint()
        position.header.frame_id = frames["planning_frame"]
        position.link_name = frames["tool_link"]
        position.constraint_region.primitives = [primitive]
        position.constraint_region.primitive_poses = [region_pose]
        position.weight = 1.0

        quaternion = _rotation_quaternion(pose["rotation_columns"])
        orientation = self._OrientationConstraint()
        orientation.header.frame_id = frames["planning_frame"]
        orientation.link_name = frames["tool_link"]
        (
            orientation.orientation.x,
            orientation.orientation.y,
            orientation.orientation.z,
            orientation.orientation.w,
        ) = quaternion
        orientation.absolute_x_axis_tolerance = float(tolerances["orientation_rad"])
        orientation.absolute_y_axis_tolerance = float(tolerances["orientation_rad"])
        orientation.absolute_z_axis_tolerance = float(tolerances["orientation_rad"])
        orientation.weight = 1.0
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    def build_gripper_goal(self, phase, position, limits):
        goal = self._FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["finger_right_joint"]
        endpoint = self._JointTrajectoryPoint()
        endpoint.positions = [float(position)]
        completion_check = self._JointTrajectoryPoint()
        completion_check.positions = [float(position)]
        completion_check.time_from_start = self._duration(limits["command_duration_s"])
        goal.trajectory.points = [endpoint, completion_check]
        goal.goal_tolerance = [
            self._JointTolerance(
                name="finger_right_joint",
                position=float(limits["completion_tolerance_m"]),
            )
        ]
        goal.goal_time_tolerance = self._duration(
            limits["execution_timeout_s"] - limits["command_duration_s"]
        )
        serialized = self._serialize_message(goal)
        if not serialized:
            raise ContractError("ROS_GRIPPER_SERIALIZATION")
        return serialized


__all__ = ["RosMoveItTransport"]
