"""Explicit bounded learned-task authority; execution remains PickupExecutor's."""
import copy
import math

from tools.fr5_data_factory import ContractError, SAFE_ID, canonical_digest

SCOPE = "SCOPED_TASK_GRANT"
GENERATION_SCHEMA = "data_factory.learned_generation_context.v1"


def _number(value, code):
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError(code)
        return float(value)
    except OverflowError as exc:
        raise ContractError(code) from exc


def validate_runtime_inputs(value, *, period_s, instruction,
                            serialized_references=False, quantized_gripper=False):
    """Validate the shared native deployment inputs, without reading their paths."""
    inputs = copy.deepcopy(value)
    causal = isinstance(inputs, dict) and inputs.get("hardware_wire_version") in (3, 4, 5)
    hardware_key = "gripper_temporal_policy" if causal else "gripper_source_clock"
    if (not isinstance(inputs, dict)
            or set(inputs) - {"warmup", "hardware_wire_version", "reference_mode"}
            != {"checkpoint", "device", hardware_key, "clock_binding", "camera_topics", "camera_mapping", "fps"}
            or any(not isinstance(inputs[key], str) or not inputs[key] for key in ("checkpoint", hardware_key))
            or not isinstance(inputs["device"], str) or inputs["device"] not in {"cpu", "cuda"}
            or not isinstance(inputs["camera_topics"], dict) or set(inputs["camera_topics"]) != {"camera1", "camera2"}
            or any(not isinstance(topic, str) or not topic.startswith("/") for topic in inputs["camera_topics"].values())
            or not isinstance(inputs["camera_mapping"], dict) or len(inputs["camera_mapping"]) != 2
            or any(not isinstance(key, str) or not key.startswith("observation.images.") or not isinstance(target, str)
                   for key, target in inputs["camera_mapping"].items())
            or set(inputs["camera_mapping"].values()) != {"observation.images.camera1", "observation.images.camera2"}
            or abs(_number(inputs["fps"], "LEARNED_HORIZON") * _number(period_s, "LEARNED_HORIZON") - 1) > 1e-9):
        raise ContractError("LEARNED_RUNTIME_INPUTS")
    if "reference_mode" in inputs:
        mode = inputs["reference_mode"]
        if (not isinstance(mode, str)
                or mode not in {"serialized_retime", "serialized_percent_retime"}
                or not serialized_references
                or (mode == "serialized_percent_retime") != quantized_gripper):
            raise ContractError("LEARNED_RUNTIME_INPUTS")
    if "warmup" in inputs:
        warmup = inputs["warmup"]
        if (not isinstance(warmup, dict)
                or set(warmup) != {"input_kind", "image_shape", "instruction_digest", "device", "model_calls", "output_disposition", "rng_state_restored", "duration_s", "inference_duration_s"}
                or warmup["input_kind"] != "SYNTHETIC_ZERO_RGB_STATE"
                or not isinstance(warmup["image_shape"], list) or len(warmup["image_shape"]) != 3
                or any(type(size) is not int or size < 1 for size in warmup["image_shape"])
                or warmup["image_shape"][-1] != 3
                or warmup["instruction_digest"] != canonical_digest(instruction)
                or warmup["device"] != inputs["device"]
                or type(warmup["model_calls"]) is not int or warmup["model_calls"] != 1
                or warmup["output_disposition"] != "DISCARDED" or warmup["rng_state_restored"] is not True
                or not 0 <= _number(warmup["inference_duration_s"], "LEARNED_WARMUP_INPUT")
                <= _number(warmup["duration_s"], "LEARNED_WARMUP_INPUT")):
            raise ContractError("LEARNED_WARMUP_INPUT")
    if "hardware_wire_version" in inputs and (
            type(inputs["hardware_wire_version"]) is not int
            or inputs["hardware_wire_version"] not in (2, 3, 4, 5)):
        raise ContractError("LEARNED_HARDWARE_SCHEMA")
    if causal:
        from .gripper_evidence import validate_temporal_policy
        binding = validate_temporal_policy(inputs["clock_binding"])
        expected_schema = ("fr5.gripper_temporal_policy.v2"
                           if inputs["hardware_wire_version"] == 5
                           else "fr5.gripper_temporal_policy.v1")
        if binding["schema_version"] != expected_schema:
            raise ContractError("LEARNED_HARDWARE_SCHEMA")
    else:
        from .gripper_evidence import validate_clock_binding
        validate_clock_binding(inputs["clock_binding"])
    return inputs


def validate_generation_context(value):
    from tools.fr5_data_factory import DIGEST
    fields = {"schema_version", "generation_id", "run_id", "task_grant_digest",
              "scene_binding_digest", "predecessor_plan_digest", "opened_at_s",
              "opened_monotonic_s", "task_deadline_monotonic_s"}
    if not isinstance(value, dict) or set(value) != fields or value["schema_version"] != GENERATION_SCHEMA:
        raise ContractError("LEARNED_GENERATION_CONTEXT")
    for key in ("generation_id", "run_id"):
        if not isinstance(value[key], str) or not SAFE_ID.fullmatch(value[key]):
            raise ContractError("LEARNED_GENERATION_CONTEXT")
    for key in ("task_grant_digest", "scene_binding_digest", "predecessor_plan_digest"):
        if key == "predecessor_plan_digest" and value[key] is None:
            continue
        if not isinstance(value[key], str) or not DIGEST.fullmatch(value[key]):
            raise ContractError("LEARNED_GENERATION_CONTEXT")
    for key in ("opened_at_s", "opened_monotonic_s", "task_deadline_monotonic_s"):
        try:
            valid = type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ContractError("LEARNED_GENERATION_CONTEXT")
    if value["task_deadline_monotonic_s"] <= value["opened_monotonic_s"]:
        raise ContractError("TASK_DEADLINE_EXHAUSTED")
    return copy.deepcopy(value)


def check_generation(grant, proposal, *, run_id, scene, predecessor, now, steady):
    """Validate frozen provenance; the executor additionally requires its allocation.

    Hashes are exact context bindings, not authentication of arbitrary Python.
    No current-state, Scene or execution authority is granted by this check.
    """
    context = validate_generation_context(proposal["generation_context"])
    if (context["run_id"] != run_id or context["task_grant_digest"] != grant["grant_digest"]
            or context["scene_binding_digest"] != canonical_digest(scene)
            or context["predecessor_plan_digest"] != predecessor):
        raise ContractError("LEARNED_GENERATION_CONTEXT")
    if not context["opened_at_s"] <= proposal["inference_started_at_s"] <= proposal["inference_completed_at_s"] <= now:
        raise ContractError("LEARNED_GENERATION_TIME")
    if not context["opened_monotonic_s"] <= steady < context["task_deadline_monotonic_s"] or now >= grant["deadline_s"]:
        raise ContractError("TASK_DEADLINE_EXHAUSTED")
    if min(grant["deadline_s"] - now, context["task_deadline_monotonic_s"] - steady) <= grant["terminal_reserve_s"] + 5.:
        raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")


def task_scope(source, scene, proposal):
    """Bind processor bytes through the checkpoint tree, and all adaptation inputs."""
    inputs = copy.deepcopy(proposal.get("runtime_inputs"))
    if inputs is not None:
        inputs.pop("warmup", None)  # Diagnostic timing is not an adaptation setting.
    return {"source_program_digest": canonical_digest(source),
            "robot_cell_scene_object": copy.deepcopy(scene),
            "robot_system_id": source["robot_system_id"],
            "policy_and_processors": copy.deepcopy(proposal["checkpoint"]),
            "task": proposal["instruction"],
            "adaptation": {key: copy.deepcopy(proposal[key]) for key in
                           ("schema_version", "robot_description", "velocity_scaling", "period_s", "max_observation_age_s")},
            "runtime_inputs": inputs,
            "permitted_grasp_release": [copy.deepcopy(step) for step in source["steps"]
                                         if step["phase"] in {"GRIPPER_CLOSE", "GRIPPER_OPEN"}]}


def validate_grant(value):
    fields = {"schema_version", "grant_id", "issued_by", "run_id", "scope", "deadline_s",
              "terminal_reserve_s", "max_outputs", "revoked", "grant_digest"}
    if not isinstance(value, dict) or set(value) != fields or value["schema_version"] != "data_factory.learned_task_grant.v1":
        raise ContractError("TASK_GRANT_SCHEMA")
    if any(not isinstance(value[k], str) or not SAFE_ID.fullmatch(value[k]) for k in ("grant_id", "issued_by", "run_id")):
        raise ContractError("TASK_GRANT_SCHEMA")
    try:
        valid_times = all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] > 0
                          for k in ("deadline_s", "terminal_reserve_s"))
    except OverflowError:
        valid_times = False
    if not valid_times or type(value["max_outputs"]) is not int or not 1 <= value["max_outputs"] <= 100:
        raise ContractError("TASK_GRANT_BOUNDS")
    if type(value["revoked"]) is not bool or not isinstance(value["scope"], dict):
        raise ContractError("TASK_GRANT_SCHEMA")
    if canonical_digest({k: v for k, v in value.items() if k != "grant_digest"}) != value["grant_digest"]:
        raise ContractError("TASK_GRANT_DIGEST")
    return copy.deepcopy(value)


def check_grant(grant, plan, now):
    validate_grant(grant)
    if grant["revoked"]:
        raise ContractError("TASK_GRANT_REVOKED")
    if now >= grant["deadline_s"]:
        raise ContractError("TASK_DEADLINE_EXHAUSTED")
    if plan.get("schema_version") == "data_factory.native_learned_task_plan.v1":
        # The private plan was validated on admission. Its immutable policy
        # settings bind the grant without freezing an inference output or
        # compiling a finite row-by-row executable on every revision.
        try:
            expected = task_scope(plan["source_program"], plan["scene_binding"], plan["policy"])
            if grant["run_id"] != plan["run_id"] or grant["scope"] != expected:
                raise ContractError("TASK_GRANT_SCOPE")
        except (KeyError, TypeError) as exc:
            raise ContractError("TASK_GRANT_SCOPE") from exc
        return
    if (grant["run_id"] != plan["run_id"] or "learned_proposal" not in plan
            or grant["scope"] != task_scope(plan["learned_source_program"], plan["scene_binding"], plan["learned_proposal"])):
        raise ContractError("TASK_GRANT_SCOPE")


def admission(grant, plan, digest, now):
    check_grant(grant, plan, now)
    # This is an admission receipt, never a fabricated human decision.
    return {"approval_id": "task-" + digest.removeprefix("sha256:"), "approved_by": grant["issued_by"],
            "approval_scope": SCOPE, "approval_expiry": None, "run_id": plan["run_id"],
            "resolved_job_digest": plan["resolved_job_digest"], "plan_digest": digest,
            "task_grant": copy.deepcopy(grant)}


def check_runtime_source(inputs):
    """Check the bound runtime file in the policy owner, before fresh capture.

    NativeSmolVLA.prepare_inference owns complete checkpoint byte verification
    and the loaded tensors for this output. Neither check belongs in the motion
    child's command/tick loop: file I/O must not delay its stop/deadline owner.
    The file is not consulted again while consuming that immutable output.
    """
    from pathlib import Path
    from tools.fr5_data_factory import load_json_strict
    try:
        key = "gripper_temporal_policy" if "gripper_temporal_policy" in inputs else "gripper_source_clock"
        if load_json_strict(Path(inputs[key])) != inputs["clock_binding"]:
            raise ContractError("TASK_SOURCE_CHANGED")
    except ContractError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ContractError("TASK_SOURCE_CHANGED") from exc


def uses_scoped_generation(grant):
    from .finite_plan import SCOPED_SCHEMAS
    if grant is None:
        return False
    grant = validate_grant(grant)
    adaptation = grant["scope"].get("adaptation")
    return isinstance(adaptation, dict) and adaptation.get("schema_version") in SCOPED_SCHEMAS
