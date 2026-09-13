"""Restricted prospective first contact, using the existing Scene and executor.

Collision checks establish a model expectation, never observed object immobility.
No attempt which started without this boundary can acquire it retrospectively.
"""
import copy
import hashlib
import math
import time
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.fr5_data_factory import (
    ContractError, canonical_digest, load_json_strict, _profile,
    compose_rigid_transform, inverse_rigid_transform, validate_rigid_transform,
    validate_cell_calibration_document, resolve_place_pose,
)

TIPS = ["finger_tip_left_link", "finger_tip_right_link"]


def _binding_inputs(plan):
    """Read the normal task or historical finite binding, without translating it."""
    if plan.get("schema_version") == "data_factory.native_learned_task_plan.v1":
        return plan["source_program"], plan["policy"]["robot_description"]
    return plan["learned_source_program"], plan.get("learned_proposal", {}).get("robot_description")


def lateral_envelope(relation, dimensions, mid, closed_gap):
    """Union initial projection with possible opposed-jaw lateral centering."""
    half = [sum(abs(relation["rotation_columns"][j][i]) * dimensions[j] / 2 for j in range(3)) for i in range(3)]
    closed_half = max(closed_gap / 2, half[0])
    low = min(relation["translation_m"][0] - half[0], mid - closed_half)
    high = max(relation["translation_m"][0] + half[0], mid + closed_half)
    return ({"translation_m": [(low + high) / 2, *relation["translation_m"][1:]],
             "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}, [high-low, 2*half[1], 2*half[2]])


def bound_document(root, folder, digest):
    for path in sorted((Path(root) / folder).glob("*.json")):
        value = load_json_strict(path)
        if canonical_digest(value) == digest:
            return path, value
    raise ContractError("CONTACT_PROFILE_UNAVAILABLE")


def bound_robot_description(transport, plan):
    """Bind source URDF geometry to native xacro output, not its serialization.

    ros2_control settings belong to the existing hardware/settings admission.
    Every link, joint, collision shape and other model element remains exact.
    """
    actual = transport._robot_description
    program, source = _binding_inputs(plan)
    expected_digest = program["binding_digests"]["robot_description_digest"]
    if plan.get("schema_version") == "data_factory.native_learned_task_plan.v1" and (
            not isinstance(source, str)
            or "sha256:" + hashlib.sha256(source.encode()).hexdigest() != expected_digest):
        raise ContractError("CONTACT_MODEL_BINDING")
    if "sha256:" + hashlib.sha256(actual.encode()).hexdigest() == expected_digest:
        return actual
    try:
        if "sha256:" + hashlib.sha256(source.encode()).hexdigest() != expected_digest:
            raise ContractError("CONTACT_MODEL_BINDING")
        deployed = ET.fromstring(actual)
        controls = deployed.findall("ros2_control")
        if len(controls) != 1 or controls[0].attrib != {"name": "FR5System", "type": "system"}:
            raise ContractError("CONTACT_MODEL_BINDING")
        deployed.remove(controls[0])
        if ET.canonicalize(ET.tostring(deployed, encoding="unicode"), strip_text=True) != ET.canonicalize(source, strip_text=True):
            raise ContractError("CONTACT_MODEL_BINDING")
    except (KeyError, AttributeError, TypeError, ET.ParseError) as exc:
        raise ContractError("CONTACT_MODEL_BINDING") from exc
    return actual


def prepare(transport, plan, scene_object):
    """Resolve only existing pinned qualified inputs before the first command."""
    source, _ = _binding_inputs(plan)
    pins = source["binding_digests"]
    root = getattr(transport, "contact_config_root", Path(__file__).resolve().parents[3] / "config/data_factory")
    _, qualification = bound_document(root, "motion_qualifications", pins["motion_qualification"])
    if (qualification.get("qualification_status") != "QUALIFIED"
            or qualification["robot_description_digest"] != pins["robot_description_digest"]
            or qualification["frames"] != source["frames"]
            or qualification["robot_system_id"] != source["robot_system_id"]
            or any(qualification["profile_digests"][key] != pins[key]
                   for key in ("object_profile", "grasp_profile", "robot_system", "cell_calibration"))):
        raise ContractError("CONTACT_QUALIFICATION_BINDING")
    profiles = {}
    for key, folder, schema in (("object_profile", "objects", "data_factory.object_profile.v2"),
                                ("grasp_profile", "grasps", "data_factory.grasp_profile.v2"),
                                ("robot_system", "robot_systems", "data_factory.robot_system.v1")):
        path, _ = bound_document(root, folder, pins[key])
        profiles[key] = _profile(root, folder, path.stem, key + "_id" if key != "robot_system" else "robot_system_id", schema)
    obj, grasp = profiles["object_profile"], profiles["grasp_profile"]
    requirements = copy.deepcopy(grasp["gripper_close"])
    if grasp["schema_version"] == "data_factory.grasp_profile.v3":
        requirements.update(open_velocity_percent=grasp["gripper_open"]["velocity_percent"],
                            open_force_percent=grasp["gripper_open"]["force_percent"])
    if (scene_object.get("state") != "ON_SURFACE" or scene_object.get("object_profile_id") != obj["object_profile_id"]
            or grasp.get("object_profile_digest") != pins["object_profile"]
            or grasp["grasp_geometry"]["datum_to_tcp_grasp"] != qualification["datum_to_tcp_grasp"]
            or requirements != source["gripper_requirements"]):
        raise ContractError("CONTACT_PROFILE_BINDING")
    _, cell = bound_document(root, "cells", pins["cell_calibration"])
    _, yaw0 = bound_document(root, "workspace_sheets", pins["yaw0_sheet"])
    calibration = validate_cell_calibration_document(cell, yaw0=yaw0, robot=profiles["robot_system"], required_status="QUALIFIED")
    pose = scene_object["pose"]
    if pose["place_id"] != cell["place_id"]:
        raise ContractError("CONTACT_SCENE_BINDING")
    resolved = resolve_place_pose(calibration["center"], calibration["x"], calibration["y"], calibration["z"],
                                  pose["yaw_deg"], pose["x_mm"], pose["y_mm"])
    datum = {"translation_m": resolved["position_base_m"], "rotation_columns": resolved["rotation_base_columns"]}
    target = next(s["target"] for s in source["steps"] if s["phase"] == "FINAL_APPROACH_LIN")
    expected = compose_rigid_transform(datum, qualification["datum_to_tcp_grasp"])
    expected_tool = compose_rigid_transform(expected, inverse_rigid_transform(qualification["tool_to_tcp"]))
    if (canonical_digest(expected) != canonical_digest(target["base_tcp"])
            or canonical_digest(expected_tool) != canonical_digest(target["base_tool"])):
        raise ContractError("CONTACT_SCENE_BINDING")
    xml = bound_robot_description(transport, plan)
    model = ET.fromstring(xml)
    dimensions = [v / 1000 for v in obj["dimensions_mm"]]
    opened = grasp["gripper_open"]["command_position_m"]
    rows = []
    for side in ("right", "left"):
        joint = model.find(f"joint[@name='finger_{side}_joint']")
        collision = model.find(f"link[@name='finger_tip_{side}_link']/collision")
        origin = [float(v) for v in joint.find("origin").attrib["xyz"].split()]
        offset = [float(v) for v in collision.find("origin").attrib["xyz"].split()]
        origin = [a+b for a,b in zip(origin, offset)]
        axis = [float(v) for v in joint.find("axis").attrib["xyz"].split()]
        size = [float(v) for v in collision.find("geometry/box").attrib["size"].split()]
        if (joint.find("parent").attrib["link"] != "gripper_link" or axis[1:] != [0., 0.]
                or any(not math.isfinite(v) for v in [*origin, *axis, *size]) or any(v <= 0 for v in size)
                or abs(axis[0]) != 1 or joint.attrib["type"] != "prismatic"
                or any(float(v) != 0 for element in (joint.find("origin"), collision.find("origin"))
                       for v in element.attrib.get("rpy", "0 0 0").split())
                or len(model.find(f"link[@name='finger_tip_{side}_link']").findall("collision")) != 1):
            raise ContractError("CONTACT_MODEL_GEOMETRY")
        rows.append((origin, axis, size))
    right, left = rows
    gap = lambda q: right[0][0] - left[0][0] + q * (right[1][0] - left[1][0]) - (right[2][0] + left[2][0]) / 2
    if (right[1][0] - left[1][0] <= 0 or gap(opened) <= dimensions[0]
            or gap(grasp["gripper_close"]["command_position_m"]) > dimensions[0]):
        raise ContractError("CONTACT_MODEL_GEOMETRY")
    return {"status": "PROSPECTIVE", "plan_digest": canonical_digest(plan), "source_program_digest": canonical_digest(source),
            "runtime_robot_description_digest": "sha256:" + hashlib.sha256(xml.encode()).hexdigest(),
            "capture_capability": "SOURCE_CONTACT_POSE_TRACKING_ONLY",
            "invariant_semantics": "SAMPLED_MODEL_NO_EARLIER_CONTACT",
            "scene_object": copy.deepcopy(scene_object), "scene_binding": copy.deepcopy(plan["scene_binding"]),
            "datum": datum, "dimensions_m": dimensions, "open_m": opened,
            "jaw_midplane_m": (right[0][0] + left[0][0]) / 2,
            "closed_gap_bound_m": max(gap(v) for v in grasp["gripper_close"]["acceptable_feedback_m"].values()),
            "fingertip_boxes": [{"center": r[0], "dimensions": r[2]} for r in rows],
            "qualification": qualification, "checked_segments": [], "close": None}


def prepare_request_geometry(transport, plan, scene_object):
    """Bind prospective source/carry/release models; never apply a Scene change.

    Hypotheses are geometry, not detected phases or evidence of successful grip.
    The qualified nominal relation/lateral envelope is not a bound on arbitrary
    slip. Existing task admission and subsequent outcome evaluation own that
    distinction; no stationary close or completion handshake is imposed here.
    """
    from tools.fr5_data_factory import validate_motion_program
    from .moveit_transport import _rotation_quaternion

    plan, scene_object = copy.deepcopy((plan, scene_object))
    source, _ = _binding_inputs(plan)
    source = validate_motion_program(source)
    if source["schema_version"] != "fr5.motion_program.v4":
        raise ContractError("CONTACT_DESTINATION_BINDING")
    context = prepare(transport, plan, scene_object)
    root = getattr(transport, "contact_config_root", Path(__file__).resolve().parents[3] / "config/data_factory")
    pins, destination = source["binding_digests"], source["destination_binding_digests"]
    _, grasp = bound_document(root, "grasps", pins["grasp_profile"])
    _, robot = bound_document(root, "robot_systems", pins["robot_system"])
    _, qualification = bound_document(root, "motion_qualifications", destination["motion_qualification"])
    if (any(destination[key] != pins[key] for key in ("object_profile", "grasp_profile", "robot_system", "robot_description_digest"))
            or qualification["qualification_status"] != "QUALIFIED"
            or qualification["robot_description_digest"] != pins["robot_description_digest"]
            or qualification["frames"] != source["frames"]
            or qualification["robot_system_id"] != source["robot_system_id"]
            or any(qualification["profile_digests"][key] != destination[key]
                   for key in ("object_profile", "grasp_profile", "robot_system", "cell_calibration"))
            or qualification["datum_to_tcp_grasp"] != context["qualification"]["datum_to_tcp_grasp"]
            or qualification["tool_to_tcp"] != context["qualification"]["tool_to_tcp"]):
        raise ContractError("CONTACT_DESTINATION_BINDING")
    _, cell = bound_document(root, "cells", destination["cell_calibration"])
    _, yaw0 = bound_document(root, "workspace_sheets", destination["yaw0_sheet"])
    calibration = validate_cell_calibration_document(cell, yaw0=yaw0, robot=robot, required_status="QUALIFIED")
    pose = plan["scene_binding"]["release_slot"]["pose"]
    if pose["place_id"] != cell["place_id"]:
        raise ContractError("CONTACT_DESTINATION_BINDING")
    resolved = resolve_place_pose(calibration["center"], calibration["x"], calibration["y"], calibration["z"],
                                  pose["yaw_deg"], pose["x_mm"], pose["y_mm"])
    datum = {"translation_m": resolved["position_base_m"], "rotation_columns": resolved["rotation_base_columns"]}
    expected = compose_rigid_transform(datum, qualification["datum_to_tcp_grasp"])
    clearance = grasp["grasp_geometry"]["release_clearance_mm"] / 1000
    expected["translation_m"] = [v + clearance * z for v, z in zip(expected["translation_m"], datum["rotation_columns"][2])]
    lower = next(step["target"] for step in source["steps"] if step["phase"] == "LOWER_LIN")
    if (canonical_digest(expected) != canonical_digest(lower["base_tcp"])
            or canonical_digest(compose_rigid_transform(expected, inverse_rigid_transform(qualification["tool_to_tcp"])))
            != canonical_digest(lower["base_tool"])):
        raise ContractError("CONTACT_DESTINATION_BINDING")

    # The qualified FR5 model has a fixed, translation-only wrist-to-gripper
    # mount. Resolve its actual origins; do not hardcode a nominal TCP offset or
    # ask live FK to manufacture a measured grasp relation.
    model = ET.fromstring(bound_robot_description(transport, plan))
    link, translation, visited = "gripper_link", [0., 0., 0.], set()
    while link != source["frames"]["tool_link"]:
        joints = [joint for joint in model.findall("joint") if joint.find("child").get("link") == link]
        if link in visited or len(joints) != 1 or joints[0].get("type") != "fixed":
            raise ContractError("CONTACT_MODEL_GEOMETRY")
        visited.add(link)
        joint = joints[0]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        if len(xyz) != 3 or any(not math.isfinite(v) for v in xyz) or any(float(v) != 0 for v in origin.get("rpy", "0 0 0").split()):
            raise ContractError("CONTACT_MODEL_GEOMETRY")
        translation = [a + b for a, b in zip(translation, xyz)]
        link = joint.find("parent").get("link")
    mount = {"translation_m": translation, "rotation_columns": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
    relation = compose_rigid_transform(inverse_rigid_transform(mount),
        compose_rigid_transform(qualification["tool_to_tcp"], inverse_rigid_transform(qualification["datum_to_tcp_grasp"])))
    envelope, dimensions = lateral_envelope(relation, context["dimensions_m"], context["jaw_midplane_m"], context["closed_gap_bound_m"])
    object_id = plan["scene_binding"]["object_instance_id"]
    frame = source["frames"]["planning_frame"]
    if (not isinstance(object_id, str) or not object_id
            or object_id in {source["planning_scene"][key]["id"] for key in ("floor", "wall")}
            or frame not in {item.get("name") for item in model.findall("link")}
            or frame in {joint.find("child").get("link") for joint in model.findall("joint")}):
        raise ContractError("CONTACT_MODEL_GEOMETRY")
    proxies = {}
    for hypothesis, parent, transform, size in (("carried", "gripper_link", envelope, dimensions),
                                              ("released", frame, datum, context["dimensions_m"])):
        proxies[hypothesis] = {"object_id": object_id + "::prospective:" + hypothesis,
            "link_name": parent, "dimensions_m": size, "translation_m": transform["translation_m"],
            "rotation_xyzw": _rotation_quaternion(transform["rotation_columns"]),
            "touch_links": TIPS[:] if hypothesis == "carried" else []}
    result = {"schema_version": "data_factory.request_contact_geometry.v1", "status": "PROSPECTIVE",
        "semantics": "QUALIFIED_NOMINAL_MODEL_EXPECTATION", "physical_success": False,
        "plan_digest": canonical_digest(plan), "source_program_digest": canonical_digest(source),
        "runtime_robot_description_digest": context["runtime_robot_description_digest"],
        "scene_binding": copy.deepcopy(plan["scene_binding"]), "scene_object": scene_object,
        "planning_frame": frame, "source_object_id": object_id, "proxies": proxies,
        "source_datum": context["datum"], "released_datum": datum,
        "source_dimensions_m": context["dimensions_m"],
        "fingertip_boxes": context["fingertip_boxes"], "open_m": context["open_m"],
        "orientation_tolerance_rad": source["planning"]["goal_tolerances"]["orientation_rad"]}
    result["geometry_digest"] = canonical_digest(result)
    return result


def _request_geometry_binding(context, plan, hypothesis):
    if (hypothesis not in {"source", "carried", "released"}
            or context.get("schema_version") != "data_factory.request_contact_geometry.v1"
            or context.get("status") != "PROSPECTIVE" or context.get("physical_success") is not False
            or context.get("plan_digest") != canonical_digest(plan)
            or context.get("geometry_digest") != canonical_digest({k: v for k, v in context.items() if k != "geometry_digest"})):
        raise ContractError("CONTACT_REQUEST_BINDING")


def request_geometry(context, plan, joints, gripper_m, *, hypothesis):
    """Build one additive /check_state_validity request; caller owns the query."""
    from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
    from moveit_msgs.srv import GetStateValidity
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    _request_geometry_binding(context, plan, hypothesis)
    values = [*joints, gripper_m]
    if len(joints) != 6 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ContractError("CONTACT_REQUEST_GEOMETRY")
    request = GetStateValidity.Request()
    request.group_name = ""
    request.robot_state.is_diff = True  # Preserve native existing attachments.
    request.robot_state.joint_state.name = ["j1", "j2", "j3", "j4", "j5", "j6", "finger_right_joint"]
    request.robot_state.joint_state.position = list(map(float, values))
    if hypothesis != "source":
        obj = context["proxies"][hypothesis]
        parent = "gripper_link" if hypothesis == "carried" else context["planning_frame"]
        touch_links = TIPS if hypothesis == "carried" else []
        if (obj["link_name"] != parent or obj["object_id"] != context["source_object_id"] + "::prospective:" + hypothesis
                or obj["touch_links"] != touch_links):
            raise ContractError("CONTACT_REQUEST_GEOMETRY")
        for key, length in (("dimensions_m", 3), ("translation_m", 3), ("rotation_xyzw", 4)):
            value = obj[key]
            if (not isinstance(value, list) or len(value) != length
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                    or key == "dimensions_m" and any(v <= 0 for v in value)):
                raise ContractError("CONTACT_REQUEST_GEOMETRY")
        if abs(sum(v * v for v in obj["rotation_xyzw"]) - 1) > 1e-9:
            raise ContractError("CONTACT_REQUEST_GEOMETRY")
        # A carried model permits its own gripping links. A released cube is
        # stationary geometry: native touch-links would hide top pressing as
        # well as legitimate side-jaw contact before our classifier can see it.
        attached = AttachedCollisionObject(link_name=parent, touch_links=touch_links[:])
        attached.object.id, attached.object.operation = obj["object_id"], CollisionObject.ADD
        attached.object.header.frame_id = parent  # No conversion fallback transform.
        attached.object.pose.orientation.w = 1.
        attached.object.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=obj["dimensions_m"])]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = obj["translation_m"]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = obj["rotation_xyzw"]
        attached.object.primitive_poses = [pose]
        request.robot_state.attached_collision_objects = [attached]
    return request


def intended_request_contact(context, plan, hypothesis, contact, *, gripper_pose=None, gripper_m=None):
    """Only inner-jaw side contact and artificial same-cube duplication are exempt.

    This classifies one contact, not a response. The caller must still reject
    incomplete/saturated contact results, constraints and every other collision.
    For fingertip contact the caller supplies native model FK at the exact
    queried joint/gripper state, not a measured object pose or a policy phase.
    """
    _request_geometry_binding(context, plan, hypothesis)
    source = context["source_object_id"]
    bodies = {(contact.contact_body_1, contact.body_type_1), (contact.contact_body_2, contact.body_type_2)}
    values = [contact.depth, contact.position.x, contact.position.y, contact.position.z,
              contact.normal.x, contact.normal.y, contact.normal.z]
    if (contact.header.frame_id != context["planning_frame"] or not all(math.isfinite(v) for v in values)
            or contact.depth < 0 or sum(v * v for v in values[4:]) <= 0):
        return False
    if hypothesis != "source":
        proxy = context["proxies"][hypothesis]["object_id"]
        if bodies == {(source, contact.WORLD_OBJECT), (proxy, contact.ROBOT_ATTACHED)}:
            return True
    object_body = (source, contact.WORLD_OBJECT)
    datum = context["source_datum"]
    if hypothesis == "released" and any(
            bodies == {(context["proxies"]["released"]["object_id"], contact.ROBOT_ATTACHED), (tip, contact.ROBOT_LINK)}
            for tip in TIPS):
        object_body = (context["proxies"]["released"]["object_id"], contact.ROBOT_ATTACHED)
        datum = context["released_datum"]
    tip = next((tip for tip in TIPS if bodies == {object_body, (tip, contact.ROBOT_LINK)}), None)
    if tip is None or gripper_pose is None or type(gripper_m) not in (int, float) or not 0 <= gripper_m <= context["open_m"]:
        return False
    try:
        gripper_pose = validate_rigid_transform(gripper_pose, "CONTACT_REQUEST_GEOMETRY")
    except ContractError:
        return False
    def project(frame, vector, *, point=False):
        if point:
            vector = [v - t for v, t in zip(vector, frame["translation_m"])]
        return [sum(a * b for a, b in zip(column, vector)) for column in frame["rotation_columns"]]
    point, normal = values[1:4], values[4:]
    source_point = project(datum, point, point=True)
    jaw_point = project(gripper_pose, point, point=True)
    normal_length = math.sqrt(sum(v * v for v in normal))
    minimum_lateral = math.cos(context["orientation_tolerance_rad"])
    if any(abs(project(frame, normal)[0]) / normal_length < minimum_lateral
           for frame in (datum, gripper_pose)):
        return False  # In particular, pressing on the cube's top is not jaw contact.
    if any(abs(v) > size / 2 + 1e-9 for v, size in zip(source_point, context["source_dimensions_m"])):
        return False
    right = tip == "finger_tip_right_link"
    box = context["fingertip_boxes"][0 if right else 1]
    center = [box["center"][0] + (gripper_m if right else -gripper_m), *box["center"][1:]]
    if any(abs(v - c) > size / 2 + 1e-9 for v, c, size in zip(jaw_point, center, box["dimensions"])):
        return False
    inner = center[0] + (-1 if right else 1) * box["dimensions"][0] / 2
    return abs(jaw_point[0] - inner) <= contact.depth + 1e-9


def before(transport, plan, step, observation, context):
    from tools.data_factory.rollout.gripper_evidence import check_transition
    if context["plan_digest"] != canonical_digest(plan):
        raise ContractError("CONTACT_PREFIX_BINDING")
    if "sha256:" + hashlib.sha256(transport._robot_description.encode()).hexdigest() != context["runtime_robot_description_digest"]:
        raise ContractError("CONTACT_MODEL_BINDING")
    prior = context["checked_segments"]
    if prior:
        check_transition(prior[-1]["terminal_observation"], observation, command=False)
    source = plan["learned_source_program"]
    snapshot = observation["snapshot"]
    if snapshot["gripper_controller"]["hardware_execution"]["wire"]["version"] not in (4, 5):
        raise ContractError("CONTACT_NATIVE_SELECTED_TUPLE_UNAVAILABLE")
    command = step["type"] == "GRIPPER"
    target = step["gripper_position_m"]
    endpoint = lambda value: math.floor(100 * value / context["open_m"] + .5)
    equivalent_target = 0 <= target <= context["open_m"] and endpoint(target) == endpoint(source["gripper_requirements"]["command_position_m"])
    is_close = command and equivalent_target and endpoint(target) <= round(snapshot["gripper_controller"]["feedback_position_m"] / context["open_m"] * 100)
    held = context["close"] is not None
    if held:
        if command and not equivalent_target:
            raise ContractError("CONTACT_HELD_EVOLUTION_UNSUPPORTED")
        from .mechanical_terminal import closure_plateau
        closure_plateau(plan, snapshot, observation["captured_at_s"], observation["captured_monotonic_s"],
                        endpoint=True, native_equivalence=True)
    if is_close and not held:
        # The close is stationary and at the source-bound contact pose. FK is
        # measured now, not inferred from the learned row's requested pose.
        tool = transport.contact_fk(source, snapshot, source["frames"]["tool_link"])
        expected = next(s["target"]["base_tool"] for s in source["steps"] if s["phase"] == "FINAL_APPROACH_LIN")
        tolerance = source["planning"]["goal_tolerances"]
        if any(abs(a-b) > tolerance["position_m"] for a,b in zip(tool["translation_m"], expected["translation_m"])):
            raise ContractError("CONTACT_CLOSE_POSE")
        relative = compose_rigid_transform(inverse_rigid_transform(expected), tool)
        angle = math.acos(max(-1., min(1., (sum(relative["rotation_columns"][i][i] for i in range(3))-1)/2)))
        if angle > tolerance["orientation_rad"]:
            raise ContractError("CONTACT_CLOSE_POSE")
        gripper = transport.contact_fk(source, snapshot, "gripper_link")
        relation = compose_rigid_transform(inverse_rigid_transform(gripper), context["datum"])
        half = [sum(abs(relation["rotation_columns"][j][i]) * context["dimensions_m"][j] / 2
                    for j in range(3)) for i in range(3)]
        if any(min(box["center"][axis] + box["dimensions"][axis]/2, relation["translation_m"][axis] + half[axis])
               <= max(box["center"][axis] - box["dimensions"][axis]/2, relation["translation_m"][axis] - half[axis])
               for box in context["fingertip_boxes"] for axis in (1, 2)):
            raise ContractError("CONTACT_CAPTURE_GEOMETRY")
    elif step["type"] not in {"ARM", "GRIPPER"}:
        raise ContractError("CONTACT_PREFIX_UNSUPPORTED")
    report = transport.check_contact_segment(plan, step, snapshot, context, closing=is_close)
    return {"plan_digest": canonical_digest(plan), "segment_digest": canonical_digest(step), "start_observation": copy.deepcopy(observation),
            "closing": is_close, "command": command, "held": held, "collision_report": report}


def completed(transport, plan, step, observation, context, pending):
    from tools.data_factory.rollout.gripper_evidence import check_transition
    check_transition(pending["start_observation"], observation, command=pending["command"])
    if pending["segment_digest"] != canonical_digest(step):
        raise ContractError("CONTACT_PREFIX_BINDING")
    record = {**pending, "terminal_observation": copy.deepcopy(observation)}
    if pending["command"]:
        from tools.data_factory.rollout.gripper_evidence import native_selected_command
        record["native_command"] = native_selected_command(observation["snapshot"]["gripper_controller"]["hardware_execution"]["wire"],
            plan["learned_source_program"]["gripper_requirements"], context["open_m"])
    if pending["closing"]:
        from .mechanical_terminal import closure_plateau
        snapshot = observation["snapshot"]
        evidence = closure_plateau(plan, snapshot, observation["captured_at_s"], observation["captured_monotonic_s"], endpoint=True, native_equivalence=True)
        start = pending["start_observation"]["snapshot"]["joint_positions"]
        tol = plan["planning"]["goal_tolerances"]["joint_rad"]
        if any(abs(a-b) > tol for a,b in zip(start, snapshot["joint_positions"])):
            raise ContractError("CONTACT_CLOSE_MOVED_ARM")
        if pending["held"]:
            context["close"]["completion"] = evidence
            context["checked_segments"].append(record)
            return
        gripper = transport.contact_fk(plan["learned_source_program"], snapshot, "gripper_link")
        relation = compose_rigid_transform(inverse_rigid_transform(gripper), context["datum"])
        # The initial pose need not survive symmetric closure laterally. Bound
        # its initial projection and the closing jaws' center interval together;
        # never turn centering into an observed exact object pose.
        envelope, dimensions = lateral_envelope(relation, context["dimensions_m"], context["jaw_midplane_m"], context["closed_gap_bound_m"])
        context["close"] = {"record": record, "completion": evidence, "relation": relation}
        context["close"].update(envelope=envelope, envelope_dimensions_m=dimensions)
    context["checked_segments"].append(record)
    if pending["closing"]:
        from .moveit_transport import _rotation_quaternion
        transport.prepare_mechanical_terminal(plan["learned_source_program"], {"carried_object": {
            "object_id": plan["scene_binding"]["object_instance_id"], "link_name": "gripper_link", "touch_links": TIPS[:],
            "dimensions_m": dimensions, "translation_m": envelope["translation_m"],
            "rotation_xyzw": _rotation_quaternion(envelope["rotation_columns"])}})


def consume(transport, plan, scene_object, snapshot, context):
    from tools.data_factory.rollout.gripper_evidence import check_transition
    from .mechanical_terminal import closure_plateau
    from .moveit_transport import _rotation_quaternion
    if (context["status"] != "PROSPECTIVE" or context["plan_digest"] != canonical_digest(plan)
            or context["scene_object"] != scene_object or context["close"] is None):
        raise ContractError("CONTACT_RELATION_UNAVAILABLE")
    now, steady = time.time(), transport._clock()
    current = {"snapshot": snapshot, "captured_at_s": now, "captured_monotonic_s": steady}
    from tools.data_factory.rollout.finite_plan import _execution_state
    _execution_state(plan["steps"][0], current, now)
    if "sha256:" + hashlib.sha256(transport._robot_description.encode()).hexdigest() != context["runtime_robot_description_digest"]:
        raise ContractError("CONTACT_MODEL_BINDING")
    close = context["close"]
    check_transition(context["checked_segments"][-1]["terminal_observation"], current, command=False)
    completion = closure_plateau(plan, snapshot, now, steady, endpoint=True, native_equivalence=True)
    if completion["generation"] != close["completion"]["generation"]:
        raise ContractError("CONTACT_GENERATION")
    before = context["checked_segments"][-1]["terminal_observation"]["snapshot"]
    if any(abs(a-b) > plan["planning"]["goal_tolerances"]["joint_rad"] for a,b in zip(before["joint_positions"], snapshot["joint_positions"])):
        raise ContractError("CONTACT_HELD_STATE_CHANGED")
    relation = validate_rigid_transform(close["envelope"], "CONTACT_RELATION")
    source = plan["learned_source_program"]
    return {"status": "AVAILABLE", "semantics": "MODEL_BASED_EXPECTATION", "physical_success": False,
            "source_program_digest": canonical_digest(source), "scene_binding": copy.deepcopy(plan["scene_binding"]),
            "snapshot_digest": canonical_digest(snapshot), "qualification_source_digest": source["binding_digests"]["motion_qualification"],
            "valid_until_s": context["deadline_s"],
            "illumination": transport.capture_scene_illumination(plan), "closure_contact": completion,
            "prospective_transition_digest": canonical_digest(context),
            "prospective_transition": copy.deepcopy(context),
            "relation_semantics": "OPPOSED_JAW_LATERAL_ENVELOPE", "initial_relation": copy.deepcopy(close["relation"]),
            "capture_capability": context["capture_capability"],
            "carried_object": {"object_id": plan["scene_binding"]["object_instance_id"], "link_name": "gripper_link",
                "touch_links": TIPS[:], "dimensions_m": close["envelope_dimensions_m"],
                "translation_m": relation["translation_m"], "rotation_xyzw": _rotation_quaternion(relation["rotation_columns"])}}
