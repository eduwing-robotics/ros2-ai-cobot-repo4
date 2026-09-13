"""Native held-command evidence, carried by the existing dynamic state broadcaster.

The controller calendar is not UTC. Archived v1/v2 evidence retains its supplied
clock mapping. Live v3 uses exact causal query brackets and owner-selected bounds;
this module neither estimates offsets nor approves execution.
"""
from __future__ import annotations

import copy
import datetime
import math

from tools.fr5_data_factory import ContractError

RESOURCE = "fr5_gripper_execution"
FIELDS = (
    "version", "incarnation_0", "incarnation_1", "incarnation_2", "incarnation_3",
    "generation", "active_generation", "completed_generation", "raw_reference_m",
    "sample_system_s", "sample_steady_s", "command_started_system_s",
    "year", "month", "day", "hour", "minute", "second", "millisecond", "frame",
    "feedback_m", "completion_reason", "pending", "rpc_active", "arm_resumed",
    "stopped", "error", "valid", "completion_year", "completion_month", "completion_day",
    "completion_hour", "completion_minute", "completion_second", "completion_millisecond",
)
CURRENT_FIELDS = (
    "current_valid", "query_before_controller_s", "query_after_controller_s",
    "query_before_system_s", "query_before_steady_s", "query_after_system_s", "query_after_steady_s", "current_max_age_s",
    "proof_valid", "proof_system_s", "proof_steady_s",
    "proof_clock_version", "proof_incarnation_0", "proof_incarnation_1", "proof_incarnation_2", "proof_incarnation_3",
    "proof_offset_s", "proof_uncertainty_s", "proof_system_anchor_s", "proof_steady_anchor_s", "proof_valid_until_s", "proof_max_age_s",
    "certificate_frame", "certificate_source_s", "certificate_sample_system_s", "certificate_sample_steady_s", "certificate_generation",
    "certificate_incarnation_0", "certificate_incarnation_1", "certificate_incarnation_2", "certificate_incarnation_3", "host_clock_tolerance_s",
)
LIVE_FIELDS = FIELDS + CURRENT_FIELDS

CAUSAL_FIELDS = LIVE_FIELDS + (
    "sample_position_percent", "sample_motion_done", "sample_fault", "ack_system_s", "ack_steady_s",
    "terminal_frame", "terminal_sample_system_s", "terminal_sample_steady_s", "terminal_position_percent", "terminal_motion_done", "terminal_fault",
    "terminal_query_before_controller_s", "terminal_query_after_controller_s", "terminal_query_before_system_s", "terminal_query_before_steady_s",
    "terminal_query_after_system_s", "terminal_query_after_steady_s", "terminal_max_age_s", "terminal_host_clock_tolerance_s",
    "terminal_reference_m", "terminal_generation", "terminal_incarnation_0", "terminal_incarnation_1", "terminal_incarnation_2", "terminal_incarnation_3",
)

# v4 appends the actual SDK selection. Archived causal v3 fields retain meaning.
SELECTED_FIELDS = CAUSAL_FIELDS + (
    "selected_valid", "selected_generation", "selected_upper_m", "selected_initial_percent",
    "selected_index", "selected_position", "selected_velocity", "selected_force", "selected_max_time",
    "selected_arg5", "selected_arg6", "selected_arg7", "selected_arg8", "selected_arg9",
)


# v5 separates coherent delivery from the original same-command terminal proof.
SNAPSHOT_FIELDS = SELECTED_FIELDS + (
    "host_receive_steady_s", "producer_sequence", "connection_epoch", "configuration_epoch", "source_progress_steady_s",
    "arm_j1_rad", "arm_j2_rad", "arm_j3_rad", "arm_j4_rad", "arm_j5_rad", "arm_j6_rad",
    "device_main_error", "device_sub_error", "terminal_connection_epoch", "terminal_configuration_epoch",
)


def native_selected_command(wire, required, upper, *, qualified_close=False):
    """Compare an acknowledged SDK selection, not two raw policy coordinates.

    Index 1 and max_time 30000 are the existing native configuration baseline;
    alternate installations cannot inherit this equivalence. Fixed arguments
    are the existing MoveGripper call, not new controller settings.
    """
    if wire.get("version") not in (4, 5) or wire.get("selected_valid") != 1:
        raise ContractError("CONTACT_NATIVE_SELECTED_TUPLE_UNAVAILABLE")
    target = wire["raw_reference_m"]
    if not 0 <= target <= upper or wire["selected_upper_m"] != upper:
        raise ContractError("CONTACT_NATIVE_UPPER_BINDING")
    endpoint = math.floor(100 * target / upper + .5)
    qualified = math.floor(100 * required["command_position_m"] / upper + .5)
    initial = integer(wire["selected_initial_percent"], 100)
    opening = endpoint > initial
    velocity = required.get("open_velocity_percent", required["velocity_percent"]) if opening else required["velocity_percent"]
    force = required.get("open_force_percent", required["force_percent"]) if opening else required["force_percent"]
    expected = [1, endpoint, velocity, force, 30000, 1, 0, 0, 0, 0]
    keys = ("index", "position", "velocity", "force", "max_time", "arg5", "arg6", "arg7", "arg8", "arg9")
    actual = [wire["selected_" + key] for key in keys]
    if (wire["selected_generation"] != wire["generation"] or wire["completed_generation"] != wire["generation"]
            or wire["ack_system_s"] < wire["command_started_system_s"] or wire["ack_steady_s"] <= 0):
        raise ContractError("CONTACT_NATIVE_GENERATION")
    if actual != expected or qualified_close and (endpoint != qualified or opening):
        raise ContractError("CONTACT_NATIVE_COMMAND_NOT_EQUIVALENT")
    return {"kind": "NATIVE_COMMAND_EQUIVALENCE" if qualified_close else "NATIVE_COMMAND_SELECTION", "raw_reference_m": target,
            "generation": wire["generation"], "selected_tuple": actual,
            "upper_position_m": upper, "initial_position_percent": wire["selected_initial_percent"],
            "qualified_reference_m": required["command_position_m"], "physical_success": False}


def native_close_equivalence(wire, required, upper):
    return native_selected_command(wire, required, upper, qualified_close=True)


def validate_temporal_policy(policy):
    """Live owner-selected bounds, not a calendar-offset estimate or approval."""
    try:
        successor = policy.get("schema_version") == "fr5.gripper_temporal_policy.v2"
        keys = {"schema_version", "incarnation", "max_age_s", "host_clock_tolerance_s"}
        if successor:
            keys |= {"connection_epoch", "configuration_epoch"}
            for key in ("connection_epoch", "configuration_epoch"):
                if integer(policy[key], 2**53-1) == 0 and key == "connection_epoch":
                    raise ValueError()
        if set(policy) != keys or policy["schema_version"] not in ("fr5.gripper_temporal_policy.v1", "fr5.gripper_temporal_policy.v2"):
            raise ValueError()
        ids = policy["incarnation"]
        if not isinstance(ids, list) or len(ids) != 4 or not any(ids):
            raise ValueError()
        for value in ids:
            integer(value, 2**32-1)
        if number(policy["max_age_s"]) <= 0 or number(policy["host_clock_tolerance_s"]) < 0:
            raise ValueError()
        return copy.deepcopy(policy)
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        raise ContractError("LEARNED_HARDWARE_TEMPORAL_POLICY") from exc


def native_temporal_parameter(policy):
    """Atomic gripper_temporal_policy_v1 double-array; explicitly configured by LIVE owner."""
    policy = validate_temporal_policy(policy)
    successor = policy["schema_version"] == "fr5.gripper_temporal_policy.v2"
    return [2. if successor else 1., *map(float, policy["incarnation"]), float(policy["max_age_s"]), float(policy["host_clock_tolerance_s"]),
            *([float(policy["connection_epoch"]), float(policy["configuration_epoch"])] if successor else [])]


def validate_evidence_binding(binding):
    if isinstance(binding, dict) and binding.get("schema_version") in ("fr5.gripper_temporal_policy.v1", "fr5.gripper_temporal_policy.v2"):
        return validate_temporal_policy(binding)
    return validate_clock_binding(binding)

CALENDAR = ("year", "month", "day", "hour", "minute", "second", "millisecond")
CLOCK_FIELDS = {"schema_version", "incarnation", "calendar_to_system_offset_s", "uncertainty_s",
                "system_anchor_s", "steady_anchor_s", "valid_until_system_s"}


def native_clock_parameter(binding, max_age_s):
    """Encode the existing measured binding for the hardware node's atomic parameter.

    This does not set a ROS parameter, measure a clock or authorize execution.
    The driver pins this value to its command; updates cannot renew that command.
    """
    binding = validate_clock_binding(binding)
    if number(max_age_s) <= 0:
        raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING")
    return [1., *map(float, binding["incarnation"]),
            *[float(binding[key]) for key in ("calendar_to_system_offset_s", "uncertainty_s",
              "system_anchor_s", "steady_anchor_s", "valid_until_system_s")], float(max_age_s)]


def number(value):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError()
        return value
    except (ValueError, OverflowError) as exc:
        raise ContractError("LEARNED_HARDWARE_SCHEMA") from exc


def integer(value, upper):
    value = number(value)
    if not 0 <= value <= upper or value != int(value):
        raise ContractError("LEARNED_HARDWARE_SCHEMA")
    return int(value)


def identity(wire):
    return [integer(wire[f"incarnation_{i}"], 2**32 - 1) for i in range(4)]


def validate_clock_binding(binding):
    try:
        if set(binding) != CLOCK_FIELDS or binding["schema_version"] != "fr5.gripper_source_clock.v1":
            raise ValueError()
        ids = binding["incarnation"]
        if not isinstance(ids, list) or len(ids) != 4 or not any(ids):
            raise ValueError()
        for value in ids:
            integer(value, 2**32 - 1)
        for field in CLOCK_FIELDS - {"schema_version", "incarnation"}:
            number(binding[field])
        if (binding["uncertainty_s"] < 0 or binding["steady_anchor_s"] < 0
                or binding["valid_until_system_s"] <= binding["system_anchor_s"]):
            raise ValueError()
        return copy.deepcopy(binding)
    except (TypeError, KeyError, ValueError, ContractError) as exc:
        raise ContractError("LEARNED_HARDWARE_CLOCK_BINDING") from exc


def decode_dynamic_state(message, binding, received_steady_s):
    """Decode actual control_msgs/DynamicJointState; header receipt is not source time."""
    try:
        names = list(message.joint_names)
        if len(names) != len(set(names)) or len(names) != len(message.interface_values):
            raise ValueError()
        values = message.interface_values[names.index(RESOURCE)]
        fields = list(values.interface_names)
        if len(fields) != len(set(fields)) or set(fields) not in (set(FIELDS), set(LIVE_FIELDS), set(CAUSAL_FIELDS), set(SELECTED_FIELDS), set(SNAPSHOT_FIELDS)) or len(fields) != len(values.values):
            raise ValueError()
        wire = dict(zip(fields, values.values))
        for value in wire.values():
            number(value)
        return {"wire": wire, "clock_binding": validate_evidence_binding(binding),
                "received_steady_s": number(received_steady_s)}
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ContractError("LEARNED_HARDWARE_SCHEMA") from exc


def calendar_s(wire, prefix=""):
    try:
        values = [integer(wire[prefix + key], 9999) for key in CALENDAR]
        if values[-1] > 999:
            raise ValueError()
        # UTC here is only a deterministic calendar encoding. The explicit measured
        # offset below supplies the relationship to this host's SYSTEM clock.
        return datetime.datetime(*values[:6], values[6] * 1000,
                                 tzinfo=datetime.timezone.utc).timestamp()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContractError("LEARNED_HARDWARE_SOURCE_CLOCK") from exc


def check_delivery(wire, policy, steady_now, max_age_s):
    """One SDK publication, original host receipt; no acquisition-age guarantee.

    Connection/config identity is frozen before approval. Native source progress
    uses the original receipt of the last strictly newer source stamp, never the
    getter or DDS delivery time. The age bound is supplied by the existing owner.
    """
    policy = validate_temporal_policy(policy)
    if (wire["version"] != 5 or policy["schema_version"] != "fr5.gripper_temporal_policy.v2"
            or wire["current_max_age_s"] != policy["max_age_s"] or policy["max_age_s"] != max_age_s
            or wire["host_clock_tolerance_s"] != policy["host_clock_tolerance_s"]):
        raise ContractError("LEARNED_HARDWARE_TEMPORAL_POLICY")
    for key in ("producer_sequence", "connection_epoch", "configuration_epoch"):
        if integer(wire[key], 2**53-1) == 0 and key != "configuration_epoch":
            raise ContractError("LEARNED_HARDWARE_INVALID")
    if any(wire[key] != policy[key] for key in ("connection_epoch", "configuration_epoch")):
        raise ContractError("LEARNED_HARDWARE_INCARNATION")
    if wire["valid"] != 1 or wire["stopped"] or wire["error"] or any(wire[k] for k in ("device_main_error", "device_sub_error", "sample_fault")):
        raise ContractError("LEARNED_HARDWARE_INVALID")
    receipt, progress = number(wire["host_receive_steady_s"]), number(wire["source_progress_steady_s"])
    if (receipt != wire["sample_steady_s"] or not 0 < progress <= receipt <= steady_now
            or steady_now-progress > max_age_s or steady_now-receipt > max_age_s):
        raise ContractError("LEARNED_HARDWARE_STALE")
    calendar_s(wire)
    for i in range(1, 7):
        number(wire[f"arm_j{i}_rad"])
    # Optional alignment is absent, not silently synthesized from receipt.
    if any(wire[key] for key in ("current_valid", *CURRENT_FIELDS[1:7], *CURRENT_FIELDS[22:31])):
        raise ContractError("LEARNED_HARDWARE_SCHEMA")


def check_current_bracket(wire, now, steady_now, max_age_s):
    """Retrospective causal HOST interval; never convert controller deltas to age."""
    w = wire
    source = calendar_s(w)
    s0, s1 = w["query_before_system_s"], w["query_after_system_s"]
    m0, m1 = w["query_before_steady_s"], w["query_after_steady_s"]
    age = w["current_max_age_s"]
    associated = (w["certificate_frame"] == w["frame"] and w["certificate_source_s"] == source
                  and w["certificate_sample_system_s"] == w["sample_system_s"]
                  and w["certificate_sample_steady_s"] == w["sample_steady_s"]
                  and w["certificate_generation"] == w["generation"]
                  and all(w[f"certificate_incarnation_{i}"] == w[f"incarnation_{i}"] for i in range(4)))
    if (not associated or w["host_clock_tolerance_s"] < 0 or w["current_valid"] != 1 or not 0 < age <= max_age_s
            or not w["query_before_controller_s"] < source-.001 < source+.001 < w["query_after_controller_s"]
            or not 0 <= m0 <= w["sample_steady_s"] <= m1 <= steady_now
            or not s0 <= w["sample_system_s"] <= s1 <= now
            or steady_now-m0 > age or now-s0 > age
            or abs((now-s0)-(steady_now-m0)) > w["host_clock_tolerance_s"]):
        raise ContractError("LEARNED_HARDWARE_STALE")


def check_causal_command_proof(w, now, steady_now, max_age_s, *, completion=False):
    """Validate frozen terminal evidence at decision time; renewal is CURRENT only."""
    at, mono = w["proof_system_s"], w["proof_steady_s"]
    ack, ack_mono = w["ack_system_s"], w["ack_steady_s"]
    source = calendar_s(w, "completion_")
    s0, m0 = w["terminal_query_before_system_s"], w["terminal_query_before_steady_s"]
    s1, m1 = w["terminal_query_after_system_s"], w["terminal_query_after_steady_s"]
    age, tolerance = w["terminal_max_age_s"], w["terminal_host_clock_tolerance_s"]
    integer(w["terminal_frame"], 255)
    integer(w["terminal_position_percent"], 100)
    integer(w["terminal_motion_done"], 1)
    # Legacy offset-proof slots are reserved zero in v3, never repurposed.
    if (any(w[key] != 0 for key in CURRENT_FIELDS[11:22])
            or w["proof_valid"] != 1 or w["terminal_fault"] != 0
            or w["terminal_generation"] != w["generation"] or w["terminal_reference_m"] != w["raw_reference_m"]
            or any(w[f"terminal_incarnation_{i}"] != w[f"incarnation_{i}"] for i in range(4))
            or not w["command_started_system_s"] <= ack <= s0 <= w["terminal_sample_system_s"] <= s1 <= at <= w["sample_system_s"]
            or not 0 < ack_mono <= m0 <= w["terminal_sample_steady_s"] <= m1 <= mono <= w["sample_steady_s"]
            or not w["terminal_query_before_controller_s"] < source-.001 < source+.001 < w["terminal_query_after_controller_s"]
            or source > calendar_s(w) or age != w["current_max_age_s"] or not 0 < age <= max_age_s
            or tolerance != w["host_clock_tolerance_s"] or at-s0 > age or mono-m0 > age
            or abs((at-s0)-(mono-m0)) > tolerance or abs((s0-ack)-(m0-ack_mono)) > tolerance
            or (w["completion_reason"] == 1 and w["terminal_motion_done"] != 1)
            or (completion and (now-s0 > max_age_s or steady_now-m0 > max_age_s))):
        raise ContractError("LEARNED_HARDWARE_COMPLETION")


def check_hardware(evidence, now, steady_now, max_age_s, *, completion=False, allow_pending=False):
    try:
        hw = evidence["snapshot"]["gripper_controller"]["hardware_execution"]
        if not isinstance(hw, dict) or set(hw) != {"wire", "clock_binding", "received_steady_s"}:
            raise ValueError()
        wire = hw["wire"]
        if not isinstance(wire, dict) or set(wire) != set(SNAPSHOT_FIELDS if wire.get("version") == 5 else SELECTED_FIELDS if wire.get("version") == 4 else CAUSAL_FIELDS if wire.get("version") == 3 else LIVE_FIELDS if wire.get("version") == 2 else FIELDS):
            raise ValueError()
        for value in wire.values():
            number(value)
        mapping = (validate_temporal_policy if wire.get("version") in (3, 4, 5) else validate_clock_binding)(hw["clock_binding"])
        if not any(identity(wire)) or identity(wire) != mapping["incarnation"]:
            raise ContractError("LEARNED_HARDWARE_INCARNATION")
        for key in ("generation", "active_generation", "completed_generation"):
            integer(wire[key], 2**53 - 1)
        integer(wire["frame"], 255)
        integer(wire["completion_reason"], 2)
        for key in ("pending", "rpc_active", "arm_resumed", "stopped", "valid"):
            integer(wire[key], 1)
        if wire["version"] not in (1, 2, 3, 4, 5) or wire["valid"] != 1:
            raise ContractError("LEARNED_HARDWARE_INVALID")
        if wire["stopped"] or wire["error"]:
            raise ContractError("LEARNED_HARDWARE_UNRESOLVED")
        unresolved = (wire["pending"] or wire["rpc_active"] or not wire["arm_resumed"]
                      or wire["active_generation"])
        # Pending progress is not completion and need not suppress the ARM
        # producer. Its native availability flag can be true between gripper
        # RPCs while the same gripper generation remains active.
        pending = (allow_pending and not completion and unresolved
                   and wire["generation"] > 0 and wire["completed_generation"] < wire["generation"]
                   and wire["active_generation"] in (0, wire["generation"])
                   and (wire["pending"] or wire["rpc_active"] or wire["active_generation"])
                   and (not wire["arm_resumed"] or (
                       not wire["pending"] and not wire["rpc_active"]
                       and wire["active_generation"] == wire["generation"])))
        if unresolved and not pending:
            raise ContractError("LEARNED_HARDWARE_UNRESOLVED")
        now, steady_now = number(now), number(steady_now)
        uncertainty = mapping.get("uncertainty_s", mapping.get("host_clock_tolerance_s"))
        if wire["version"] == 5:
            check_delivery(wire, mapping, steady_now, max_age_s)
            observed = evidence["snapshot"]
            if (observed["joint_positions"] != [wire[f"arm_j{i}_rad"] for i in range(1, 7)]
                    or observed["gripper_controller"]["feedback_position_m"] != wire["feedback_m"]):
                raise ContractError("LEARNED_HARDWARE_SAMPLE_BINDING")
        elif wire["version"] >= 2:
            check_current_bracket(wire, now, steady_now, max_age_s)
        elif (not mapping["system_anchor_s"] <= now <= mapping["valid_until_system_s"]
                or abs((now - mapping["system_anchor_s"]) - (steady_now - mapping["steady_anchor_s"])) > uncertainty):
            raise ContractError("LEARNED_HARDWARE_SOURCE_CLOCK")
        # Monotonic age independently rejects frozen SYSTEM/ROS time and newly
        # delivered old broadcaster messages. All clocks here are on one host.
        if not wire["sample_steady_s"] <= number(hw["received_steady_s"]) <= number(evidence["captured_monotonic_s"]) <= steady_now:
            raise ContractError("LEARNED_HARDWARE_STALE")
        for stamp in (hw["received_steady_s"], wire["sample_steady_s"], evidence["captured_monotonic_s"]):
            age = steady_now - number(stamp)
            if not 0 <= age <= max_age_s:
                raise ContractError("LEARNED_HARDWARE_STALE")
        if not 0 <= now - wire["sample_system_s"] <= max_age_s:
            raise ContractError("LEARNED_HARDWARE_STALE")
        if wire["version"] in (3, 4, 5):
            if (wire["current_max_age_s"] != mapping["max_age_s"]
                    or wire["host_clock_tolerance_s"] != mapping["host_clock_tolerance_s"]
                    or wire["sample_fault"] != 0):
                raise ContractError("LEARNED_HARDWARE_TEMPORAL_POLICY")
            integer(wire["sample_motion_done"], 1)
            integer(wire["sample_position_percent"], 100)
        source = calendar_s(wire) + mapping.get("calendar_to_system_offset_s", 0.)
        if wire["version"] == 1 and (source + uncertainty > now or now - (source - uncertainty) > max_age_s):
            raise ContractError("LEARNED_HARDWARE_STALE")
        if pending:
            if wire["active_generation"] == wire["generation"] and wire["completion_reason"] != 0:
                raise ContractError("LEARNED_HARDWARE_COMPLETION")
            # Only the transport's existing post-JTC handoff uses this branch.
            # Freshness, identity and hard faults above still apply; the caller
            # must bind generation/reference to its one retained command.
            return wire
        generation = wire["generation"]
        if generation == 0:
            if completion or wire["completed_generation"] or wire["completion_reason"]:
                raise ContractError("LEARNED_HARDWARE_COMPLETION")
        else:
            if wire["completed_generation"] != generation or wire["completion_reason"] not in (1, 2):
                raise ContractError("LEARNED_HARDWARE_COMPLETION")
            if wire["version"] in (3, 4, 5):
                check_causal_command_proof(wire, now, steady_now, max_age_s, completion=completion and wire["version"] != 5)
                if wire["version"] == 5 and any(wire["terminal_" + key] != wire[key] for key in ("connection_epoch", "configuration_epoch")):
                    raise ContractError("LEARNED_HARDWARE_INCARNATION")
                # Selected-command authority is consumed separately. An absent
                # tuple must not destroy otherwise valid diagnostic telemetry.
                return wire
            if wire["version"] == 2:
                # Revalidate the original proof at its recorded instant, not now.
                proof = validate_clock_binding({
                    "schema_version": "fr5.gripper_source_clock.v1",
                    "incarnation": [wire[f"proof_incarnation_{i}"] for i in range(4)],
                    "calendar_to_system_offset_s": wire["proof_offset_s"],
                    "uncertainty_s": wire["proof_uncertainty_s"],
                    "system_anchor_s": wire["proof_system_anchor_s"],
                    "steady_anchor_s": wire["proof_steady_anchor_s"],
                    "valid_until_system_s": wire["proof_valid_until_s"],
                })
                at, mono = wire["proof_system_s"], wire["proof_steady_s"]
                uncertainty = proof["uncertainty_s"]
                if (wire["proof_valid"] != 1 or wire["proof_clock_version"] != 1
                        or proof["incarnation"] != identity(wire) or proof != mapping
                        or not proof["system_anchor_s"] <= at <= proof["valid_until_system_s"]
                        or not 0 <= mono <= wire["sample_steady_s"]
                        or abs((at-proof["system_anchor_s"])-(mono-proof["steady_anchor_s"])) > uncertainty):
                    raise ContractError("LEARNED_HARDWARE_COMPLETION")
                finished = calendar_s(wire, "completion_") + proof["calendar_to_system_offset_s"]
                if (finished + uncertainty > at or wire["proof_max_age_s"] <= 0
                        or at-(finished-uncertainty) > wire["proof_max_age_s"]):
                    raise ContractError("LEARNED_HARDWARE_COMPLETION")
            else:
                finished = calendar_s(wire, "completion_") + mapping["calendar_to_system_offset_s"]
            if (finished - uncertainty <= wire["command_started_system_s"] or finished > source
                    or (completion and now - (finished - uncertainty) > max_age_s)):
                raise ContractError("LEARNED_HARDWARE_COMPLETION")
        return wire
    except ContractError:
        raise
    except (TypeError, KeyError, ValueError, OverflowError) as exc:
        raise ContractError("LEARNED_HARDWARE_SCHEMA") from exc


def check_transition(start, terminal, *, command, allow_queued=False):
    """Check association in both the sole executor and the canonical trace reader."""
    try:
        a = start["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        b = terminal["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        if a["version"] == 5 or b["version"] == 5:
            if (a["version"] != b["version"] or any(a[k] != b[k] for k in ("connection_epoch", "configuration_epoch"))):
                raise ContractError("LEARNED_HARDWARE_INCARNATION")
            if (b["producer_sequence"] < a["producer_sequence"] or calendar_s(b) < calendar_s(a)
                    or b["host_receive_steady_s"] < a["host_receive_steady_s"]
                    or (calendar_s(b) == calendar_s(a) and b["source_progress_steady_s"] != a["source_progress_steady_s"])):
                raise ContractError("LEARNED_HARDWARE_STALE")
        if identity(a) != identity(b):
            raise ContractError("LEARNED_HARDWARE_INCARNATION")
        if b["generation"] != a["generation"] + int(command):
            raise ContractError("LEARNED_HARDWARE_SUPERSEDED")
        if allow_queued and b["completed_generation"] < b["generation"] and b["completed_generation"] != a["generation"]:
            raise ContractError("LEARNED_HARDWARE_COMPLETION")
        queued = allow_queued and b["active_generation"] == 0 and (b["pending"] or b["rpc_active"])
        if command and not queued and b["command_started_system_s"] < start["captured_at_s"]:
            raise ContractError("LEARNED_HARDWARE_COMPLETION")
    except (TypeError, KeyError) as exc:
        raise ContractError("LEARNED_HARDWARE_SCHEMA") from exc
