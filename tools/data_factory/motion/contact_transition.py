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
    expected_digest = plan["learned_source_program"]["binding_digests"]["robot_description_digest"]
    if "sha256:" + hashlib.sha256(actual.encode()).hexdigest() == expected_digest:
        return actual
    try:
        source = plan["learned_proposal"]["robot_description"]
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
    source = plan["learned_source_program"]
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
