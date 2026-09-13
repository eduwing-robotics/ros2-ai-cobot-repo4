"""Current execution observations, independent of policy outputs or motion targets.

These checks consume retained native evidence and existing owner-selected bounds.
They neither acquire observations nor authorize dispatch, claim command completion,
or renew policy inputs. Command identity remains the execution owner's concern.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from functools import lru_cache

from tools.fr5_data_factory import ContractError
from tools.data_factory.learned_action_adapter import _action

JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "finger_right_joint"]


def _number(value, code):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(code)
        return float(value)
    except OverflowError as exc:
        raise ContractError(code) from exc


def _limits(xml):
    if not isinstance(xml, str):
        raise ContractError("LEARNED_URDF_LIMITS")
    return _parsed_limits(xml)


@lru_cache(maxsize=16)
def _parsed_limits(xml):
    try:
        root = ET.fromstring(xml)
        joints = {joint.get("name"): joint for joint in root.findall("joint")}
        result = []
        for name in JOINTS:
            joint = joints[name]
            if joint.get("type") != ("prismatic" if name == JOINTS[-1] else "revolute"):
                raise ValueError("joint type")
            limit = joint.find("limit")
            values = [float(limit.attrib[key]) for key in ("lower", "upper", "velocity")]
            if not all(math.isfinite(v) for v in values) or values[0] >= values[1] or values[2] <= 0:
                raise ValueError("limit")
            result.append(tuple(values))
        return tuple(result)
    except (ET.ParseError, KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ContractError("LEARNED_URDF_LIMITS") from exc


def controller_state(policy, evidence, now, *, max_age_s):
    """Shared historical state checks, without acquisition or hardware binding."""
    try:
        observed = evidence["snapshot"]
        elapsed = _number(now, "LEARNED_SOURCE_CLOCK") - _number(evidence["captured_at_s"], "LEARNED_SOURCE_CLOCK")
        ages = [_number(age, "LEARNED_STALE_STATE") for age in
                (observed["joint_state_age_s"], observed["arm_controller"]["age_s"], observed["gripper_controller"]["age_s"])]
        if elapsed < 0 or any(age < 0 or age + elapsed > max_age_s for age in ages):
            raise ContractError("LEARNED_STALE_STATE")
        if any(observed[key]["ready"] is not True for key in ("arm_controller", "gripper_controller")):
            raise ContractError("CONTROLLER_NOT_READY")
        # Positive scaling is not same-command completion or acknowledgement.
        from tools.data_factory.motion.moveit_transport import validate_controller_sample
        for key in ("arm_controller", "gripper_controller"):
            if _number(observed[key]["speed_scaling"], "LEARNED_STATE_SCHEMA") <= 0:
                raise ContractError("LEARNED_CONTROLLER_PAUSED")
            validate_controller_sample(observed[key])
        state = list(_action([*observed["joint_positions"], observed["gripper_controller"]["feedback_position_m"]]))
        if any(not low <= value <= high for value, (low, high, _) in zip(state, _limits(policy["robot_description"]))):
            raise ContractError("LEARNED_JOINT_LIMIT")
        return state
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("LEARNED_STATE_SCHEMA") from exc


def check_acquisition_stamps(evidence, now, *, max_age_s):
    """Require the original native joint/JTC source stamps, not fresh receipt."""
    try:
        observed = evidence["snapshot"]
        stamps = [observed["joint_state_stamp_ns"],
                  *[observed[k]["sample"]["ros_stamp_ns"] for k in ("arm_controller", "gripper_controller")]]
        for stamp in stamps:
            if type(stamp) is not int or not 0 <= now - stamp / 1e9 <= max_age_s:
                raise ContractError("LEARNED_STALE_STATE")
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("LEARNED_STATE_SCHEMA") from exc


def check_runtime_binding(policy, evidence, wire):
    """Compare current native hardware to the existing static policy binding."""
    try:
        inputs = policy.get("runtime_inputs")
        if inputs is not None and "hardware_wire_version" in inputs and wire["version"] != inputs["hardware_wire_version"]:
            raise ContractError("LEARNED_HARDWARE_INVALID")
        if inputs is not None and evidence["snapshot"]["gripper_controller"]["hardware_execution"]["clock_binding"] != inputs["clock_binding"]:
            raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("LEARNED_STATE_SCHEMA") from exc


def current_state(policy, evidence, now, *, steady_now, max_age_s, allow_pending=False):
    """Return a detached checked 7D state; no selected target or hold condition.

    ``policy`` is the validated static policy configuration, and ``max_age_s``
    comes from source planning limits. Pending uses only the existing hardware
    progress rule; the caller still owns command identity, grants and dispatch.
    """
    from .gripper_evidence import check_hardware
    state = controller_state(policy, evidence, now, max_age_s=max_age_s)
    check_acquisition_stamps(evidence, now, max_age_s=max_age_s)
    wire = check_hardware(evidence, now, steady_now, max_age_s, allow_pending=allow_pending)
    check_runtime_binding(policy, evidence, wire)
    return state
