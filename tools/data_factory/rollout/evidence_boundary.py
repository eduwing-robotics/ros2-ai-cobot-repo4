"""Deterministic, read-only RolloutEvidencePacket.v1 producer."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from tools.data_factory.candidate_admission import validate_candidate_admission
from tools.data_factory.episode_ledger import (
    validate_episode_ledger,
    validate_episode_state,
)
from tools.data_factory.task_recipe import (
    EPISODE_INSTRUCTION_SCHEMA,
    TASK_IDS,
    validate_episode_instruction_binding,
)
from tools.fr5_data_factory import (
    ContractError,
    DIGEST,
    canonical_digest,
    load_json_strict,
    task_review_checklist_id,
)

PACKET_SCHEMA = "data_factory.rollout_evidence_packet.v1"
UNKNOWN = "UNKNOWN"


def build_proposal_diagnostic(candidate: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnose a preexecution candidate, never synthesize a terminal lifecycle.

    The sidecar consumer validates event identity/order. This owner independently
    replays the numerical validator and binds exact candidate/output payloads.
    A numerical block is not a task failure or evidence of a data deficit.
    """
    import hashlib
    import math
    import struct
    from .action_projection import project_gripper_position
    from .finite_plan import validate_proposal

    code = "ROLLOUT_PROPOSAL_DIAGNOSTIC_BINDING"
    try:
        c, result = copy.deepcopy(dict(candidate)), dict(validation)
        if (set(c) != {"schema_version", "chunk_sequence", "raw_chunk", "processed_chunk",
                       "observation_evidence", "proposal_candidate", "projection"}
                or c["schema_version"] != "data_factory.rollout_proposal_candidate.v1"
                or type(c["chunk_sequence"]) is not int or c["chunk_sequence"] < 1
                or set(result) != {"candidate_event_digest", "status", "code", "dispatch_scope", "dispatch"}
                or result["dispatch_scope"] != "PROPOSAL_BUILDER_ONLY"
                or result["dispatch"] != "NOT_ATTEMPTED"
                or result["status"] not in {"VALID", "BLOCKED", "ERROR"}
                or not isinstance(c["observation_evidence"], dict)
                or not isinstance(c["projection"], dict)):
            raise ContractError(code)
        _digest(result["candidate_event_digest"], code)
        for name in ("raw_chunk", "processed_chunk"):
            tensor = c[name]
            shape = tensor["shape"]
            sizes = {"<f2": 2, "<f4": 4, "<f8": 8, ">f2": 2, ">f4": 4, ">f8": 8}
            if (set(tensor) != {"dtype", "shape", "data_hex", "sha256"}
                    or tensor["dtype"] not in sizes or not isinstance(shape, list)
                    or len(shape) not in (2, 3) or shape[-1] != 7
                    or len(shape) == 3 and shape[0] != 1
                    or any(type(n) is not int or n < 1 for n in shape)
                    or not 1 <= shape[-2] <= 50):
                raise ContractError(code)
            raw = bytes.fromhex(tensor["data_hex"])
            if (len(raw) != math.prod(shape) * sizes[tensor["dtype"]]
                    or hashlib.sha256(raw).hexdigest() != tensor["sha256"]):
                raise ContractError(code)
            if name == "processed_chunk":
                scalar = {"2": "e", "4": "f", "8": "d"}[tensor["dtype"][-1]]
                processed_rows = list(struct.iter_unpack(tensor["dtype"][0] + scalar * 7, raw))
        proposal = c["proposal_candidate"]
        if proposal["proposal_digest"] != canonical_digest({k: v for k, v in proposal.items() if k != "proposal_digest"}):
            raise ContractError(code)
        stamps = c["observation_evidence"]["source_timestamps_s"]
        if (proposal["source_timestamps_s"] != {
                "state": stamps["state"], "camera1": stamps["up"], "camera2": stamps["wrist"]}
                or any(c[k]["shape"][-2] != len(proposal["actions"])
                       for k in ("raw_chunk", "processed_chunk"))
                or c["projection"]["projected_actions_digest"] != canonical_digest(proposal["actions"])):
            raise ContractError(code)
        # Hashes bind bytes, not their claimed meaning. Replay the same pure
        # gripper projection as the adapter; all six arm values stay exact.
        projection = c["projection"]
        projected_rows, deltas = [], []
        for row in processed_rows:
            if not all(math.isfinite(v) for v in row):
                raise ContractError(code)
            gripper = project_gripper_position(
                row[-1], upper_m=projection["gripper_upper_m"],
                projection_quanta=projection["gripper_projection_quanta"],
            )["projected_m"]
            projected_rows.append([*row[:6], gripper])
            deltas.append(abs(gripper - row[-1]))
        if (projected_rows != proposal["actions"] or projection != {
                "rows": len(projected_rows),
                "gripper_projection_rows": sum(delta > 0 for delta in deltas),
                "max_gripper_projection_delta_m": max(deltas),
                "gripper_upper_m": projection["gripper_upper_m"],
                "gripper_projection_quanta": projection["gripper_projection_quanta"],
                "projected_actions_digest": canonical_digest(projected_rows)}):
            raise ContractError(code)
        try:
            validate_proposal(proposal)
        except ContractError as rejected:
            expected = ("BLOCKED", str(rejected))
        else:
            expected = ("VALID", None)
        if result["status"] == "ERROR":
            if not isinstance(result["code"], str) or not result["code"]:
                raise ContractError(code)
        elif (result["status"], result["code"]) != expected:
            raise ContractError(code)
        value = {
            "schema_version": "data_factory.rollout_proposal_diagnostic.v1",
            "candidate_digest": canonical_digest(c),
            "chunk_sequence": c["chunk_sequence"],
            "checkpoint": proposal["checkpoint"],
            "observation_digest": proposal["observation_digest"],
            "proposal_candidate_digest": proposal["proposal_digest"],
            "stage": "PROPOSAL_VALIDATION", "validation_status": result["status"],
            "code": result["code"], "dispatch_scope": result["dispatch_scope"],
            "dispatch": result["dispatch"], "task_outcome": "NOT_EVALUATED",
            "data_deficit": UNKNOWN, "execution_authorized": False, "training_authorized": False,
        }
        value["diagnostic_digest"] = canonical_digest(value)
        return value
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ContractError(code) from exc

_PACKET_FIELDS = frozenset({
    "schema_version", "identity", "data_quality_analysis",
    "rollout_evidence_analysis", "limitations", "packet_digest",
})
_IDENTITY_FIELDS = frozenset({
    "task_id", "task_binding_digest", "instruction_binding_digest",
    "episode_ref_digest", "ledger_digest", "state_digest",
    "candidate_admission_digest",
})
_DQA_FIELDS = frozenset({
    "identity", "technical_status", "candidate_status", "trace", "checkpoint",
})
_REA_FIELDS = frozenset({
    "identity", "trace", "checkpoint", "policy_row", "clock", "purpose",
    "effectiveness", "execution", "promotion", "physical_verification",
    "curator_approval", "training_authorization",
})
_EVIDENCE_FIELDS = frozenset({"status", "digest", "path"})
_EVIDENCE_NAMES = ("trace", "checkpoint", "policy_row", "clock", "purpose")
_AUTHORITY_NAMES = (
    "effectiveness", "execution", "promotion", "physical_verification",
    "curator_approval", "training_authorization",
)
_LIMITATIONS = [
    "Trace, checkpoint, policy-row, clock, and purpose evidence have no "
    "production owner or schema and remain UNKNOWN.",
    "This packet grants no execution, effectiveness, promotion, physical "
    "verification, curator, or training authority; each remains UNKNOWN.",
]
_SOURCE_FILES = {
    "ledger": "episode_ledger.json",
    "state": "episode_ledger_state.json",
    "candidate_admission": "candidate_admission.json",
}


def _digest(value: object, code: str) -> str:
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        raise ContractError(code)
    return value


def _unknown_evidence() -> dict[str, Any]:
    return {"status": UNKNOWN, "digest": None, "path": None}


def _plan_instruction(ledger: Mapping[str, Any]) -> dict[str, Any]:
    reference = ledger["artifacts"]["plan"]
    try:
        plan = load_json_strict(Path(reference["artifact_path"]))
    except (OSError, ContractError) as exc:
        raise ContractError("ROLLOUT_EVIDENCE_PLAN") from exc
    if canonical_digest(plan) != reference["artifact_digest"]:
        raise ContractError("ROLLOUT_EVIDENCE_PLAN_DIGEST")
    try:
        instruction = validate_episode_instruction_binding(
            plan["episode_instruction_binding"],
        )
    except (KeyError, ContractError) as exc:
        raise ContractError("ROLLOUT_EVIDENCE_PLAN_INSTRUCTION") from exc
    if (
        instruction["schema_version"] != EPISODE_INSTRUCTION_SCHEMA
        or plan.get("episode_instruction_binding_digest")
        != instruction["binding_digest"]
    ):
        raise ContractError("ROLLOUT_EVIDENCE_PLAN_INSTRUCTION")
    return instruction


def _state_candidate(
    state: Mapping[str, Any], *, ledger: Mapping[str, Any], task_id: str,
) -> dict[str, Any] | None:
    reference = state["candidate"]
    if reference is None:
        return None
    try:
        candidate = load_json_strict(Path(reference["artifact_path"]))
    except (OSError, ContractError) as exc:
        raise ContractError("ROLLOUT_EVIDENCE_CANDIDATE") from exc
    try:
        candidate = validate_candidate_admission(candidate)
    except ContractError as exc:
        raise ContractError("ROLLOUT_EVIDENCE_CANDIDATE_BINDING") from exc
    review = state["review"]
    if (
        canonical_digest(candidate) != reference["artifact_digest"]
        or candidate["run_id"] != ledger["episode"]["run_id"]
        or candidate["operational_gate"] != "PASS"
        or candidate["checklist_id"] != task_review_checklist_id(task_id)
        or candidate["review_context_digest"]
        != ledger["admission"]["review_context_digest"]
        or any(
            candidate[field] != review[field]
            for field in ("semantic_status", "reviewed_by", "reviewed_at", "reason")
        )
    ):
        raise ContractError("ROLLOUT_EVIDENCE_CANDIDATE_BINDING")
    return candidate


def build_packet(
    *, ledger: Mapping[str, Any], state: Mapping[str, Any],
    task_binding: Mapping[str, Any] | None = None,
    episode_instruction: Mapping[str, Any] | None = None,
    candidate_admission: Mapping[str, Any] | None = None,
    trace: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    policy_row: Mapping[str, Any] | None = None,
    clock: Mapping[str, Any] | None = None,
    purpose: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate canonical owners and return a packet; never writes or executes."""
    if any(value is not None for value in (trace, checkpoint, policy_row, clock, purpose)):
        raise ContractError("ROLLOUT_EVIDENCE_UNOWNED_INPUT")

    checked_ledger = validate_episode_ledger(ledger)
    checked_state = validate_episode_state(state, ledger=checked_ledger)
    instruction = _plan_instruction(checked_ledger)
    binding = instruction["task_binding"]
    if (
        task_binding is not None and task_binding != binding
        or episode_instruction is not None and episode_instruction != instruction
    ):
        raise ContractError("ROLLOUT_EVIDENCE_PLAN_INSTRUCTION")
    state_candidate = _state_candidate(
        checked_state, ledger=checked_ledger, task_id=binding["task_id"],
    )
    if candidate_admission is not None:
        if (
            state_candidate is None
            or not isinstance(candidate_admission, Mapping)
            or dict(candidate_admission) != state_candidate
        ):
            raise ContractError("ROLLOUT_EVIDENCE_CANDIDATE_BINDING")

    candidate_digest = (
        None
        if checked_state["candidate"] is None
        else checked_state["candidate"]["artifact_digest"]
    )
    identity = {
        "task_id": binding["task_id"],
        "task_binding_digest": binding["binding_digest"],
        "instruction_binding_digest": instruction["binding_digest"],
        "episode_ref_digest": checked_ledger["episode"]["episode_ref_digest"],
        "ledger_digest": checked_ledger["ledger_digest"],
        "state_digest": checked_state["state_digest"],
        "candidate_admission_digest": candidate_digest,
    }
    dq = {
        "identity": copy.deepcopy(identity),
        "technical_status": checked_ledger["admission"]["technical_status"],
        "candidate_status": checked_state["review"]["semantic_status"],
        "trace": _unknown_evidence(),
        "checkpoint": _unknown_evidence(),
    }
    re = {
        "identity": copy.deepcopy(identity),
        **{name: _unknown_evidence() for name in _EVIDENCE_NAMES},
        **{name: UNKNOWN for name in _AUTHORITY_NAMES},
    }
    packet = {
        "schema_version": PACKET_SCHEMA,
        "identity": identity,
        "data_quality_analysis": dq,
        "rollout_evidence_analysis": re,
        "limitations": copy.deepcopy(_LIMITATIONS),
    }
    packet["packet_digest"] = canonical_digest(packet)
    return validate_packet(packet)


def _validate_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_IDENTITY")
    identity = copy.deepcopy(dict(value))
    if identity["task_id"] not in TASK_IDS:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_IDENTITY")
    for key in (
        "task_binding_digest", "instruction_binding_digest", "episode_ref_digest",
        "ledger_digest", "state_digest",
    ):
        _digest(identity[key], "ROLLOUT_EVIDENCE_PACKET_IDENTITY")
    if identity["candidate_admission_digest"] is not None:
        _digest(
            identity["candidate_admission_digest"],
            "ROLLOUT_EVIDENCE_PACKET_IDENTITY",
        )
    return identity


def _validate_unknown_evidence(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != _EVIDENCE_FIELDS
        or dict(value) != _unknown_evidence()
    ):
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_EVIDENCE")


def validate_packet(value: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on every semantic claim in the canonical JSON boundary."""
    if (
        not isinstance(value, Mapping)
        or set(value) != _PACKET_FIELDS
        or value["schema_version"] != PACKET_SCHEMA
    ):
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_FIELDS")
    expected = _digest(value["packet_digest"], "ROLLOUT_EVIDENCE_PACKET_DIGEST")
    if canonical_digest({
        key: item for key, item in value.items() if key != "packet_digest"
    }) != expected:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_DIGEST")

    identity = _validate_identity(value["identity"])
    dq = value["data_quality_analysis"]
    re = value["rollout_evidence_analysis"]
    if not isinstance(dq, Mapping) or set(dq) != _DQA_FIELDS:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_DQA")
    if not isinstance(re, Mapping) or set(re) != _REA_FIELDS:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_REA")
    if dq["identity"] != identity or re["identity"] != identity:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_IDENTITY")
    for name in ("trace", "checkpoint"):
        _validate_unknown_evidence(dq[name])
    for name in _EVIDENCE_NAMES:
        _validate_unknown_evidence(re[name])
    if any(re[name] != UNKNOWN for name in _AUTHORITY_NAMES):
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_AUTHORITY")

    technical = dq["technical_status"]
    candidate = dq["candidate_status"]
    candidate_digest = identity["candidate_admission_digest"]
    if technical not in {"PASS", "FAIL"}:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_DQA")
    if candidate_digest is None:
        if candidate != ("NOT_MEASURED" if technical == "PASS" else "NOT_AVAILABLE"):
            raise ContractError("ROLLOUT_EVIDENCE_PACKET_DQA")
    elif technical != "PASS" or candidate not in {"PENDING", "PASS", "FAIL", "UNCERTAIN"}:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_DQA")
    if value["limitations"] != _LIMITATIONS:
        raise ContractError("ROLLOUT_EVIDENCE_PACKET_LIMITATIONS")
    return copy.deepcopy(dict(value))


def _load_sources(root: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, filename in _SOURCE_FILES.items():
        path = root / filename
        if name == "candidate_admission" and not path.exists() and not path.is_symlink():
            values[name] = None
            continue
        try:
            resolved = path.resolve(strict=True)
            if path.is_symlink() or not resolved.is_file() or resolved.parent != root:
                raise ContractError("ROLLOUT_EVIDENCE_INPUT")
            values[name] = load_json_strict(resolved)
        except (OSError, ContractError) as exc:
            raise ContractError("ROLLOUT_EVIDENCE_INPUT") from exc
    return values


def inspect_directory(root: str | Path) -> dict[str, Any]:
    """Read canonical ledger files and return a packet; never mutate the directory."""
    source = Path(root)
    try:
        directory = source.resolve(strict=True)
    except OSError as exc:
        raise ContractError("ROLLOUT_EVIDENCE_INPUT_ROOT") from exc
    if (
        source.is_symlink() or not directory.is_dir()
        or source.absolute() != directory
    ):
        raise ContractError("ROLLOUT_EVIDENCE_INPUT_ROOT")
    values = _load_sources(directory)
    packet = build_packet(
        ledger=values["ledger"], state=values["state"],
        candidate_admission=values["candidate_admission"],
    )
    reference = values["state"]["candidate"]
    candidate_path = directory / _SOURCE_FILES["candidate_admission"]
    if (
        (values["candidate_admission"] is None) != (reference is None)
        or reference is not None
        and reference["artifact_path"] != str(candidate_path)
    ):
        raise ContractError("ROLLOUT_EVIDENCE_CANDIDATE_PATH")
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path)
    args = parser.parse_args(argv)
    try:
        packet = inspect_directory(args.input_directory)
    except (ContractError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def build_run_diagnostic(lifecycle_result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a finite learned run without inventing a committed episode/ledger.

    The existing OneJob result owns recorder disposition and executor evidence.
    This read-only diagnostic can target later collection/re-evaluation; neither
    controller completion nor human semantic PASS grants terminal/reset safety.
    """
    from tools.data_factory.rollout.finite_plan import validate_execution_trace, validate_execution_history
    try:
        result = copy.deepcopy(dict(lifecycle_result))
        plan = result["plan_envelope"]["plan"]
        if (canonical_digest(plan) != result["plan_digest"] or plan["run_id"] != result["run_id"]
                or result["recorder_state"] not in {"ABORTED", "FROZEN", "QUARANTINED_COMMIT"}
                or result["state"] not in {"ABORTED", "BLOCKED", "QUARANTINED_COMMIT"}):
            raise ContractError("ROLLOUT_RUN_DIAGNOSTIC_BINDING")
        trace = validate_execution_trace(plan, result["execution_evidence"]["learned_execution"])
        history = validate_execution_history(plan, result["execution_evidence"].get("learned_history", []))
        if trace["status"] not in {"COMPLETED", "FAILED"}:
            raise ContractError("ROLLOUT_RUN_DIAGNOSTIC_TERMINAL")
        recorder = result["recorder_evidence"]
        if not isinstance(recorder, dict) or recorder.get("state") != result["recorder_state"]:
            raise ContractError("ROLLOUT_RUN_DIAGNOSTIC_RECORDER")
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("ROLLOUT_RUN_DIAGNOSTIC_BINDING") from exc
    diagnostic = {
        "schema_version": "data_factory.rollout_run_diagnostic.v1",
        "run_id": result["run_id"], "plan_digest": result["plan_digest"],
        "proposal_digest": trace["proposal_digest"], "checkpoint": trace["checkpoint"],
        "execution_trace": trace, "lifecycle_result_digest": canonical_digest(result),
        "recorder_disposition": result["recorder_state"], "recorder_evidence_digest": canonical_digest(recorder),
        "disposition_reason": result["code"], "episode_ledger": None,
        "human_semantic_verdict": result.get("semantic_verdict"),
        "human_semantic_decision": copy.deepcopy(result["execution_evidence"].get("semantic_decision")),
        "task_effectiveness": UNKNOWN, "physical_qualification": UNKNOWN,
        "training_authorized": False, "online_policy_authorized": False,
    }
    if history:
        diagnostic["execution_history"] = history
    evidence = result["execution_evidence"]
    if "mechanical_terminal" in evidence:
        from tools.data_factory.motion.mechanical_terminal import validate_terminal_evidence
        diagnostic["mechanical_terminal"] = validate_terminal_evidence(evidence["mechanical_terminal"], plan)
    elif "mechanical_contact_diagnostic" in evidence:
        diagnostic["mechanical_contact_diagnostic"] = copy.deepcopy(evidence["mechanical_contact_diagnostic"])
    nested_handoff = "task_handoff" in evidence
    legacy_handoff = "task_handoff" in result
    if nested_handoff and legacy_handoff and evidence["task_handoff"] != result["task_handoff"]:
        raise ContractError("ROLLOUT_RUN_DIAGNOSTIC_HANDOFF")
    if nested_handoff or legacy_handoff:
        diagnostic["task_handoff"] = copy.deepcopy(
            evidence["task_handoff"] if nested_handoff else result["task_handoff"])
    if "generation_cleanup" in result:
        cleanup = result["generation_cleanup"]
        # One unconfirmed cleanup attempt is diagnostic metadata, not evidence
        # of physical stop or a replacement for the lifecycle's disposition.
        if (not isinstance(cleanup, dict) or set(cleanup) != {"status", "code"}
                or cleanup["status"] != "UNCONFIRMED"
                or not isinstance(cleanup["code"], str) or not 1 <= len(cleanup["code"]) <= 128):
            raise ContractError("ROLLOUT_RUN_DIAGNOSTIC_CLEANUP")
        diagnostic["generation_cleanup"] = cleanup
    diagnostic["diagnostic_digest"] = canonical_digest(diagnostic)
    return diagnostic
