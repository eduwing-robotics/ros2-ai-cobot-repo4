"""Static admission identity for one native continuous learned task.

The plan binds the task and its qualified inputs.  Policy rows, native goal
timing, queue progress and execution evidence belong to revision records owned
by the sole executor; they are deliberately absent here.
"""

import copy
import math

from tools.data_factory.rollout.finite_plan import _limits, _robot_description_digest
from tools.data_factory.rollout.task_authority import validate_runtime_inputs
from tools.data_factory.scene_state import validate_scene_binding
from tools.fr5_data_factory import ContractError, DIGEST, SAFE_ID, validate_motion_program


PLAN_SCHEMA = "data_factory.native_learned_task_plan.v1"
POLICY_SCHEMA = "data_factory.native_learned_policy.v1"


def _number(value, code):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(code)
        return float(value)
    except OverflowError as exc:
        raise ContractError(code) from exc


def _validate_policy(value, source):
    fields = {
        "schema_version", "checkpoint", "instruction", "robot_description",
        "velocity_scaling", "period_s", "max_observation_age_s",
    }
    if isinstance(value, dict) and "runtime_inputs" in value:
        fields.add("runtime_inputs")
    if not isinstance(value, dict) or set(value) != fields or value.get("schema_version") != POLICY_SCHEMA:
        raise ContractError("LEARNED_STREAM_POLICY_SCHEMA")
    policy = copy.deepcopy(value)

    checkpoint = policy["checkpoint"]
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"tree_digest", "training_receipt_digest", "runtime"}
        or not isinstance(checkpoint["runtime"], str)
        or checkpoint["runtime"] not in {"lerobot-0.6.1-native", "SYNTHETIC_TEST_ONLY"}
        or any(
            not isinstance(checkpoint[key], str) or not DIGEST.fullmatch(checkpoint[key])
            for key in ("tree_digest", "training_receipt_digest")
        )
    ):
        raise ContractError("LEARNED_CHECKPOINT_BINDING")
    if not isinstance(policy["instruction"], str) or not policy["instruction"].strip():
        raise ContractError("LEARNED_INSTRUCTION")

    _limits(policy["robot_description"])
    if _robot_description_digest(policy["robot_description"]) != source["binding_digests"]["robot_description_digest"]:
        raise ContractError("LEARNED_ROBOT_BINDING")
    period = _number(policy["period_s"], "LEARNED_HORIZON")
    age = _number(policy["max_observation_age_s"], "LEARNED_SOURCE_CLOCK")
    scaling = _number(policy["velocity_scaling"], "LEARNED_LIMITS")
    source_scaling = min(
        step["limits"]["velocity_scaling"]
        for step in source["steps"]
        if "velocity_scaling" in step["limits"]
    )
    if period < 1 / 30 or age != .3 or not 0 < scaling <= min(.1, source_scaling):
        raise ContractError("LEARNED_STREAM_POLICY_LIMITS")

    if "runtime_inputs" not in policy:
        return policy
    inputs = policy["runtime_inputs"]
    if (not isinstance(inputs, dict) or "warmup" not in inputs
            or "hardware_wire_version" not in inputs or "reference_mode" in inputs):
        raise ContractError("LEARNED_RUNTIME_INPUTS")
    policy["runtime_inputs"] = validate_runtime_inputs(
        inputs, period_s=period, instruction=policy["instruction"],
    )
    return policy


def validate_stream_plan(value):
    """Return a detached checked task identity; this grants no execution."""
    fields = {
        "schema_version", "run_id", "source_program", "resolved_job_digest",
        "scene_binding", "policy",
    }
    if not isinstance(value, dict) or set(value) != fields or value.get("schema_version") != PLAN_SCHEMA:
        raise ContractError("LEARNED_STREAM_PLAN_SCHEMA")
    if not isinstance(value["run_id"], str) or not SAFE_ID.fullmatch(value["run_id"]):
        raise ContractError("LEARNED_STREAM_PLAN_SCHEMA")
    source = validate_motion_program(copy.deepcopy(value["source_program"]))
    if source["schema_version"] != "fr5.motion_program.v4":
        raise ContractError("LEARNED_STREAM_SOURCE_PROGRAM")
    if value["resolved_job_digest"] != source["resolved_job_digest"]:
        raise ContractError("LEARNED_STREAM_SOURCE_PROGRAM")
    scene = validate_scene_binding(copy.deepcopy(value["scene_binding"]))
    policy = _validate_policy(value["policy"], source)
    return {
        "schema_version": PLAN_SCHEMA,
        "run_id": value["run_id"],
        "source_program": source,
        "resolved_job_digest": source["resolved_job_digest"],
        "scene_binding": scene,
        "policy": policy,
    }


def build_stream_plan(run_id, source_program, scene_binding, policy):
    """Build the immutable task-level identity before native inference starts."""
    return validate_stream_plan({
        "schema_version": PLAN_SCHEMA,
        "run_id": run_id,
        "source_program": copy.deepcopy(source_program),
        "resolved_job_digest": source_program.get("resolved_job_digest") if isinstance(source_program, dict) else None,
        "scene_binding": copy.deepcopy(scene_binding),
        "policy": copy.deepcopy(policy),
    })
