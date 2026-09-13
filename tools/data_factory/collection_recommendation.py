"""Pure validation and operator-intent projection for collection advice."""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from tools.data_factory.candidate_admission import (
    SCHEMA_VERSION as _CANDIDATE_SCHEMA,
)
from tools.data_factory.campaign_authoring import (
    MANIFEST_SCHEMA_V2 as MANIFEST_SCHEMA,
    _base_counts,
    _candidate_slots,
    _slot_template,
    validate_campaign_compilation_receipt,
)
from tools.data_factory.episode_ledger import (
    EPISODE_LOCATOR_SCHEMA as LOCATOR_SCHEMA,
    SCHEMA_VERSION as LEDGER_SCHEMA,
    STATE_SCHEMA_VERSION as STATE_SCHEMA,
    validate_loaded_episode_evidence,
)
from tools.data_factory.quality.coverage_report import (
    REPORT_SCHEMA as DATA_QUALITY_SCHEMA,
    build_coverage_report,
    validate_coverage_report,
)
from tools.fr5_data_factory import (
    ContractError,
    DIGEST,
    RFC3339,
    SAFE_ID,
    canonical_digest,
)


SCHEMA_VERSION = "data_factory.collection_recommendation.v1"
SNAPSHOT_SCHEMA = "data_factory.collection_recommendation_input_snapshot.v1"
EPISODE_REF_SCHEMA = "data_factory.episode_ref.v1"
VIEW_SCHEMA = "data_factory.operator_session_view.v2"
INTENT_SCHEMA = "data_factory.operator_intent.v1"

RECOMMENDATION_FIELDS = frozenset({
    "schema_version", "recommendation_id", "input_snapshot", "claims",
    "suggested_draft_patches", "authority", "recommendation_digest",
})
SNAPSHOT_FIELDS = frozenset({
    "schema_version", "source_commit", "campaign", "episodes",
    "data_quality_analysis_ref", "rollout_evidence_analysis_ref",
    "snapshot_digest",
})
CAMPAIGN_FIELDS = frozenset({
    "schema_version", "manifest_id", "manifest_digest",
})
EPISODE_SNAPSHOT_FIELDS = frozenset({
    "manifest_order_index", "run_id", "episode_index", "dataset_id",
    "dataset_digest", "episode_ref", "locator", "ledger", "state",
    "candidate", "source_provenance_digest", "recording_quality_digest",
})
SCHEMA_DIGEST_FIELDS = frozenset({"schema_version", "digest"})
ANALYSIS_REF_FIELDS = frozenset({
    "availability", "schema_version", "analysis_id", "analysis_digest",
    "reason_codes",
})
CLAIM_FIELDS = frozenset({
    "claim_id", "class", "subject", "value", "evidence_refs",
    "basis_claim_ids", "reason_codes",
})
PATCH_FIELDS = frozenset({"change_id", "field", "value", "basis_claim_ids"})
AUTHORITY_FIELDS = frozenset({
    "recommendation", "dataset_mutation", "candidate_mutation",
    "ledger_mutation", "training_authorization", "motion_authority",
    "gate_bypass", "plan_compile", "campaign_authorization",
})
AUTHORITY = {
    "recommendation": "ADVISORY_ONLY",
    "dataset_mutation": False,
    "candidate_mutation": False,
    "ledger_mutation": False,
    "training_authorization": False,
    "motion_authority": False,
    "gate_bypass": False,
    "plan_compile": False,
    "campaign_authorization": False,
}
EVIDENCE_FIELDS = frozenset({
    "manifest_order_index", "ledger", "state", "candidate", "artifacts",
})
VIEW_FIELDS = frozenset({
    "schema_version", "session_id", "revision", "projection", "generated_at",
    "view_digest", "authority",
})
VIEW_AUTHORITY = {
    "browser": "INTENT_ONLY",
    "lifecycle_owner": "BACKEND",
    "human_identity": "NOT_AUTHENTICATED",
    "training_approval": "SEPARATE",
}
CLAIM_CLASSES = frozenset({"OBSERVED", "SUGGESTED", "UNKNOWN"})
CLAIM_SUBJECTS = frozenset({
    "person", "background", "robot", "coverage", "quality", "rollout", "semantic",
})
UNKNOWN_REASON_SUBJECTS = {
    "PERSON_LABELS_UNAVAILABLE": "person",
    "BACKGROUND_LABELS_UNAVAILABLE": "background",
    "ROBOT_VARIATION_UNMEASURED": "robot",
    "COVERAGE_NOT_MEASURED": "coverage",
    "DATA_QUALITY_ANALYSIS_UNAVAILABLE": "quality",
    "NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS": "rollout",
    "ROLLOUT_DATA_DEFICIT_UNPROVEN": "rollout",
    "SEMANTIC_PROOF_UNAVAILABLE": "semantic",
}
OBSERVED_VALUE_FIELDS = frozenset({"metric", "count"})
SUGGESTED_VALUES = frozenset({"COLLECT_MORE", "REDEMONSTRATE_CONDITION"})
PATCH_FIELDS_ALLOWLIST = frozenset({
    "requested_count", "repeat", "split", "selection",
    "state_space_design_factors", "campaign_selection",
})
_MISSING = object()


def _exact(value: object, fields: frozenset[str], code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ContractError(code)
    return copy.deepcopy(dict(value))


def _identifier(value: object, code: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise ContractError(code)
    return value


def _digest(value: object, code: str) -> str:
    if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
        raise ContractError(code)
    return value


def _count(value: object, code: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ContractError(code)
    return value


def _strings(
    value: object, code: str, *, nonempty: bool = False, identifiers: bool = True,
    normalize: bool = False,
) -> list[str]:
    if (
        not isinstance(value, list)
        or nonempty and not value
        or any(
            not isinstance(item, str)
            or not item
            or identifiers and SAFE_ID.fullmatch(item) is None
            for item in value
        )
        or len(value) != len(set(value))
        or not normalize and value != sorted(value)
    ):
        raise ContractError(code)
    return sorted(value)


def _self_digest(value: Mapping[str, Any], field: str, code: str) -> str:
    expected = _digest(value.get(field), code)
    if canonical_digest({key: item for key, item in value.items() if key != field}) != expected:
        raise ContractError(code)
    return expected


def _campaign(
    manifest: object, *, hypothesis: Mapping[str, Any], draft: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validate_campaign_compilation_receipt(
        receipt, draft=draft, manifest=manifest, hypothesis=hypothesis,
    )
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ContractError("COLLECTION_RECOMMENDATION_MANIFEST_SCHEMA")
    return copy.deepcopy(dict(manifest))


def _episode_snapshot(
    value: object, manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, str, int]]:
    evidence = _exact(
        value, EVIDENCE_FIELDS, "COLLECTION_RECOMMENDATION_EVIDENCE_FIELDS",
    )
    order_index = _count(
        evidence["manifest_order_index"],
        "COLLECTION_RECOMMENDATION_EVIDENCE_ORDER",
    )
    if order_index >= len(manifest["slots"]):
        raise ContractError("COLLECTION_RECOMMENDATION_EVIDENCE_ORDER")

    checked = validate_loaded_episode_evidence(
        ledger=evidence["ledger"], state=evidence["state"],
        candidate=evidence["candidate"], artifacts=evidence["artifacts"],
    )
    payloads = checked["artifacts"]
    if (
        payloads["manifest"] != manifest
        or payloads["intent"].get("order_index") != order_index
        or payloads["intent"].get("slot") != manifest["slots"][order_index]
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_MANIFEST_ARTIFACT_BINDING")
    if checked["admission"]["technical_status"] != "PASS":
        raise ContractError("COLLECTION_RECOMMENDATION_ADMISSION")

    ledger = checked["ledger"]
    dataset = checked["dataset"]
    ref = checked["episode_ref"]
    expected_dataset_digest = canonical_digest({
        "repo_id": dataset["repo_id"],
        "dataset_root": dataset["dataset_root"],
        "episode_ref": ref,
    })
    if (
        dataset["dataset_digest"] != expected_dataset_digest
        or dataset["dataset_id"] != f"dataset-{expected_dataset_digest[7:23]}"
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_DATASET_DIGEST")

    candidate = checked["candidate"]
    if candidate is None:
        raise ContractError("COLLECTION_RECOMMENDATION_CANDIDATE_BINDING")
    summary = {
        "manifest_order_index": order_index,
        "run_id": checked["run_id"],
        "episode_index": ref["episode_index"],
        "dataset_id": dataset["dataset_id"],
        "dataset_digest": dataset["dataset_digest"],
        "episode_ref": {
            "schema_version": EPISODE_REF_SCHEMA,
            "digest": ledger["episode"]["episode_ref_digest"],
        },
        "locator": {
            "schema_version": LOCATOR_SCHEMA,
            "digest": checked["locator"]["locator_digest"],
        },
        "ledger": {
            "schema_version": LEDGER_SCHEMA,
            "digest": ledger["ledger_digest"],
        },
        "state": {
            "schema_version": STATE_SCHEMA,
            "digest": checked["state"]["state_digest"],
        },
        "candidate": {
            "schema_version": _CANDIDATE_SCHEMA,
            "digest": canonical_digest(candidate),
        },
        "source_provenance_digest": checked["artifact_refs"][
            "source_provenance"
        ]["artifact_digest"],
        "recording_quality_digest": checked["artifact_refs"][
            "recording_quality"
        ]["artifact_digest"],
    }
    identity = (
        dataset["repo_id"], dataset["dataset_root"], ref["episode_index"],
    )
    return summary, identity


def _episode_summaries(
    values: object, manifest: Mapping[str, Any], *, normalize: bool,
) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ContractError("COLLECTION_RECOMMENDATION_EPISODES")
    checked = [_episode_snapshot(item, manifest) for item in values]
    episodes = [item[0] for item in checked]
    identities = [item[1] for item in checked]
    if normalize:
        episodes.sort(key=lambda item: item["manifest_order_index"])
    if [item["manifest_order_index"] for item in episodes] != list(range(len(episodes))):
        raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_ORDER")
    if len(identities) != len(set(identities)):
        raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_DUPLICATE")
    unique_fields = (
        "run_id", "dataset_digest", "source_provenance_digest",
        "recording_quality_digest",
    )
    if any(len({item[field] for item in episodes}) != len(episodes) for field in unique_fields):
        raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_DUPLICATE")
    for nested in ("episode_ref", "locator", "ledger", "state", "candidate"):
        if len({item[nested]["digest"] for item in episodes}) != len(episodes):
            raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_DUPLICATE")
    return episodes


def _analysis_ref(
    value: object, *, owner: str, artifact: object = _MISSING,
    normalize: bool,
) -> dict[str, Any]:
    ref = _exact(value, ANALYSIS_REF_FIELDS, "COLLECTION_RECOMMENDATION_ANALYSIS_REF_FIELDS")
    reasons = _strings(
        ref["reason_codes"], "COLLECTION_RECOMMENDATION_ANALYSIS_REASONS",
        normalize=normalize,
    )
    if ref["availability"] == "UNAVAILABLE":
        if (
            any(ref[field] is not None for field in ("schema_version", "analysis_id", "analysis_digest"))
            or not reasons
            or owner == "rollout"
            and "NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS" not in reasons
            or artifact is not _MISSING and artifact is not None
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_ANALYSIS_UNAVAILABLE")
        ref["reason_codes"] = reasons
        return ref
    if ref["availability"] != "AVAILABLE" or reasons:
        raise ContractError("COLLECTION_RECOMMENDATION_ANALYSIS_AVAILABILITY")
    schema = _identifier(ref["schema_version"], "COLLECTION_RECOMMENDATION_ANALYSIS_SCHEMA")
    analysis_id = _identifier(ref["analysis_id"], "COLLECTION_RECOMMENDATION_ANALYSIS_ID")
    expected_digest = _digest(ref["analysis_digest"], "COLLECTION_RECOMMENDATION_ANALYSIS_DIGEST")
    if owner == "rollout":
        if schema != "data_factory.rollout_run_diagnostic.v1":
            raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_OWNER")
        if artifact is not _MISSING:
            from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
            if not isinstance(artifact, Mapping):
                raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_LIFECYCLE_REQUIRED")
            diagnostic = build_run_diagnostic(artifact)
            if diagnostic["run_id"] != analysis_id or canonical_digest(diagnostic) != expected_digest:
                raise ContractError("COLLECTION_RECOMMENDATION_ANALYSIS_DIGEST")
        return ref
    if schema != DATA_QUALITY_SCHEMA:
        raise ContractError("COLLECTION_RECOMMENDATION_DATA_QUALITY_OWNER")
    if artifact is not _MISSING:
        if not isinstance(artifact, Mapping):
            raise ContractError("COLLECTION_RECOMMENDATION_ANALYSIS_ARTIFACT")
        report = validate_coverage_report(artifact)
        if (
            report["schema_version"] != schema
            or report["collection_profile_id"] != analysis_id
            or canonical_digest(report) != expected_digest
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_ANALYSIS_DIGEST")
    ref["reason_codes"] = []
    return ref


def _analysis_refs(
    data_quality_ref: object, rollout_ref: object, *,
    data_quality_analysis: object = _MISSING,
    rollout_evidence_analysis: object = _MISSING,
    normalize: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_quality = _analysis_ref(
        data_quality_ref, owner="data_quality", artifact=data_quality_analysis,
        normalize=normalize,
    )
    rollout = _analysis_ref(
        rollout_ref, owner="rollout", artifact=rollout_evidence_analysis,
        normalize=normalize,
    )
    return data_quality, rollout


def _known_evidence(snapshot: Mapping[str, Any]) -> set[str]:
    result = {snapshot["campaign"]["manifest_digest"]}
    for episode in snapshot["episodes"]:
        result.update({
            episode["dataset_digest"], episode["source_provenance_digest"],
            episode["recording_quality_digest"],
            *(episode[name]["digest"] for name in ("episode_ref", "locator", "ledger", "state", "candidate")),
        })
    for name in ("data_quality_analysis_ref", "rollout_evidence_analysis_ref"):
        if snapshot[name]["availability"] == "AVAILABLE":
            result.add(snapshot[name]["analysis_digest"])
    return result


def _claim_value(claim: Mapping[str, Any]) -> Any:
    if claim["class"] == "UNKNOWN":
        if claim["value"] is not None:
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_VALUE")
        return None
    if claim["class"] == "SUGGESTED":
        if (
            claim["subject"] != "coverage"
            or not isinstance(claim["value"], str)
            or claim["value"] not in SUGGESTED_VALUES
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_VALUE")
        return claim["value"]
    if claim["subject"] != "coverage":
        raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_VALUE")
    value = _exact(
        claim["value"], OBSERVED_VALUE_FIELDS,
        "COLLECTION_RECOMMENDATION_CLAIM_VALUE",
    )
    if (
        value["metric"] != "COLLECTED_EPISODE_COUNT"
        or type(value["count"]) is not int
        or value["count"] < 0
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_VALUE")
    return value


def _claims(
    values: object, snapshot: Mapping[str, Any], *, normalize: bool,
) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ContractError("COLLECTION_RECOMMENDATION_CLAIMS")
    known_evidence = _known_evidence(snapshot)
    result = []
    for raw in values:
        claim = _exact(raw, CLAIM_FIELDS, "COLLECTION_RECOMMENDATION_CLAIM_FIELDS")
        _identifier(claim["claim_id"], "COLLECTION_RECOMMENDATION_CLAIM_ID")
        if claim["class"] not in CLAIM_CLASSES or claim["subject"] not in CLAIM_SUBJECTS:
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_TYPE")
        evidence = _strings(
            claim["evidence_refs"], "COLLECTION_RECOMMENDATION_CLAIM_EVIDENCE",
            identifiers=False, normalize=normalize,
        )
        if any(_digest(item, "COLLECTION_RECOMMENDATION_CLAIM_EVIDENCE") not in known_evidence for item in evidence):
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_EVIDENCE")
        basis = _strings(
            claim["basis_claim_ids"], "COLLECTION_RECOMMENDATION_CLAIM_BASIS",
            normalize=normalize,
        )
        reasons = _strings(
            claim["reason_codes"], "COLLECTION_RECOMMENDATION_CLAIM_REASONS",
            normalize=normalize,
        )
        if (
            claim["class"] == "OBSERVED" and (not evidence or basis or reasons)
            or claim["class"] == "SUGGESTED"
            and (not basis or reasons != [
                "HUMAN_REVIEWED_CHUNK_FAILURE" if claim["value"] == "REDEMONSTRATE_CONDITION"
                else "COVERAGE_DEFICIT"
            ])
            or claim["class"] == "UNKNOWN" and (basis or not reasons)
            or claim["class"] == "UNKNOWN" and any(
                UNKNOWN_REASON_SUBJECTS.get(reason) != claim["subject"]
                for reason in reasons
            )
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_EPISTEMIC")
        if claim["class"] == "SUGGESTED" and claim["value"] == "REDEMONSTRATE_CONDITION" and (
            snapshot["rollout_evidence_analysis_ref"]["availability"] != "AVAILABLE"
            or snapshot["rollout_evidence_analysis_ref"]["analysis_digest"] not in evidence
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_EPISTEMIC")
        claim.update(
            value=_claim_value(claim), evidence_refs=evidence,
            basis_claim_ids=basis, reason_codes=reasons,
        )
        if (
            claim["class"] == "OBSERVED"
            and (
                claim["value"]["count"] != len(snapshot["episodes"])
                or snapshot["campaign"]["manifest_digest"] not in evidence
            )
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_VALUE")
        result.append(claim)
    if normalize:
        result.sort(key=lambda item: item["claim_id"])
    ids = [claim["claim_id"] for claim in result]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_DUPLICATE")
    by_id = {claim["claim_id"]: claim for claim in result}
    for claim in result:
        if any(
            basis_id == claim["claim_id"]
            or basis_id not in by_id
            or by_id[basis_id]["class"] != "OBSERVED"
            for basis_id in claim["basis_claim_ids"]
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_CLAIM_BASIS")
    required_unknowns = {
        "person": "PERSON_LABELS_UNAVAILABLE",
        "background": "BACKGROUND_LABELS_UNAVAILABLE",
        "robot": "ROBOT_VARIATION_UNMEASURED",
        "rollout": "NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS",
    }
    if snapshot["rollout_evidence_analysis_ref"]["availability"] == "AVAILABLE":
        required_unknowns["rollout"] = "ROLLOUT_DATA_DEFICIT_UNPROVEN"
    if snapshot["data_quality_analysis_ref"]["availability"] == "UNAVAILABLE":
        required_unknowns["quality"] = "DATA_QUALITY_ANALYSIS_UNAVAILABLE"
    elif any(claim["subject"] == "quality" for claim in result):
        raise ContractError("COLLECTION_RECOMMENDATION_NUISANCE_CLAIM")
    for subject, reason in required_unknowns.items():
        matches = [claim for claim in result if claim["subject"] == subject]
        if (
            len(matches) != 1
            or matches[0]["class"] != "UNKNOWN"
            or matches[0]["reason_codes"] != [reason]
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_NUISANCE_CLAIM")
    return result


def _patch_value(field: str, value: object) -> Any:
    if field == "campaign_selection":
        from tools.data_factory.campaign_operator import UPDATE_FIELDS
        value = _exact(value, UPDATE_FIELDS | {"source_hypothesis_digest"},
                       "COLLECTION_RECOMMENDATION_PATCH_VALUE")
        _digest(value["source_hypothesis_digest"], "COLLECTION_RECOMMENDATION_PATCH_VALUE")
        if (value["authoring_mode"] != "DIRECT_EDIT"
                or type(value["requested_count"]) is not int
                or not 1 <= value["requested_count"] <= 100
                or type(value["normalized_seed"]) is not int or value["normalized_seed"] < 0
                or any(not isinstance(value[key], list)
                       or any(not isinstance(item, str) or not SAFE_ID.fullmatch(item) for item in value[key])
                       or len(value[key]) != len(set(value[key])) for key in ("pinned", "excluded"))
                or not isinstance(value["direct_slots"], list)
                or len(value["direct_slots"]) != value["requested_count"]):
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
    elif field in {"requested_count", "repeat"}:
        if type(value) is not int or not 1 <= value <= 100:
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
    elif field == "split":
        if value not in {"TRAIN", "ID", "OOD"}:
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
    elif field == "selection":
        if not isinstance(value, Mapping) or len(value) != 1:
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
        axis, selected = next(iter(value.items()))
        _identifier(axis, "COLLECTION_RECOMMENDATION_PATCH_VALUE")
        _identifier(selected, "COLLECTION_RECOMMENDATION_PATCH_VALUE")
    else:
        factors = _exact(
            value, frozenset({"columns", "rows", "yaw_cdf_strata"}),
            "COLLECTION_RECOMMENDATION_PATCH_VALUE",
        )
        columns, rows, yaw = factors["columns"], factors["rows"], factors["yaw_cdf_strata"]
        if (
            type(columns) is not int or type(rows) is not int or type(yaw) is not int
            or not 1 <= columns <= 100 or not 1 <= rows <= 100
            or columns * rows > 100 or not 1 <= yaw <= columns * rows
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
        value = factors
    canonical_digest(value)
    return copy.deepcopy(value)


def _patches(
    values: object, claims: Sequence[Mapping[str, Any]], *, normalize: bool,
) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ContractError("COLLECTION_RECOMMENDATION_PATCHES")
    by_id = {claim["claim_id"]: claim for claim in claims}
    result = []
    for raw in values:
        patch = _exact(raw, PATCH_FIELDS, "COLLECTION_RECOMMENDATION_PATCH_FIELDS")
        _identifier(patch["change_id"], "COLLECTION_RECOMMENDATION_PATCH_ID")
        if patch["field"] not in PATCH_FIELDS_ALLOWLIST:
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_FIELD")
        basis = _strings(
            patch["basis_claim_ids"], "COLLECTION_RECOMMENDATION_PATCH_BASIS",
            nonempty=True, normalize=normalize,
        )
        if any(item not in by_id for item in basis):
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_BASIS")
        if any(
            by_id[item]["class"] == "UNKNOWN"
            for item in basis
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_CAUSAL")
        patch["value"] = _patch_value(patch["field"], patch["value"])
        patch["basis_claim_ids"] = basis
        result.append(patch)
    if normalize:
        result.sort(key=lambda item: item["change_id"])
    ids = [patch["change_id"] for patch in result]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ContractError("COLLECTION_RECOMMENDATION_PATCH_DUPLICATE")
    return result


def _authority(value: object) -> dict[str, Any]:
    authority = _exact(value, AUTHORITY_FIELDS, "COLLECTION_RECOMMENDATION_AUTHORITY_FIELDS")
    if authority["recommendation"] != "ADVISORY_ONLY" or any(
        type(authority[field]) is not bool or authority[field] is not False
        for field in AUTHORITY_FIELDS - {"recommendation"}
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_AUTHORITY")
    return authority


def build_collection_recommendation(
    *, recommendation_id: str, source_commit: str,
    campaign_manifest: Mapping[str, Any], campaign_hypothesis: Mapping[str, Any],
    campaign_draft: Mapping[str, Any],
    campaign_compilation_receipt: Mapping[str, Any],
    episode_evidence: Sequence[Mapping[str, Any]],
    data_quality_analysis_ref: Mapping[str, Any],
    rollout_evidence_analysis_ref: Mapping[str, Any], claims: Sequence[Mapping[str, Any]],
    suggested_draft_patches: Sequence[Mapping[str, Any]],
    data_quality_analysis: Mapping[str, Any] | None = None,
    rollout_evidence_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical recommendation from already-loaded evidence."""
    _identifier(recommendation_id, "COLLECTION_RECOMMENDATION_ID")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_SOURCE_COMMIT")
    manifest = _campaign(
        campaign_manifest, hypothesis=campaign_hypothesis,
        draft=campaign_draft, receipt=campaign_compilation_receipt,
    )
    episodes = _episode_summaries(episode_evidence, manifest, normalize=True)
    data_quality_ref, rollout_ref = _analysis_refs(
        data_quality_analysis_ref, rollout_evidence_analysis_ref,
        data_quality_analysis=data_quality_analysis,
        rollout_evidence_analysis=rollout_evidence_analysis,
        normalize=True,
    )
    _rollout_condition(campaign_hypothesis, rollout_evidence_analysis)
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA,
        "source_commit": source_commit,
        "campaign": {
            "schema_version": MANIFEST_SCHEMA,
            "manifest_id": manifest["manifest_id"],
            "manifest_digest": manifest["manifest_digest"],
        },
        "episodes": episodes,
        "data_quality_analysis_ref": data_quality_ref,
        "rollout_evidence_analysis_ref": rollout_ref,
    }
    snapshot["snapshot_digest"] = canonical_digest(snapshot)
    checked_claims = _claims(claims, snapshot, normalize=True)
    if any(c["value"] == "REDEMONSTRATE_CONDITION" for c in checked_claims) and not _human_failed_chunk(rollout_evidence_analysis):
        raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_REVIEW_REQUIRED")
    value = {
        "schema_version": SCHEMA_VERSION,
        "recommendation_id": recommendation_id,
        "input_snapshot": snapshot,
        "claims": checked_claims,
        "suggested_draft_patches": _patches(
            suggested_draft_patches, checked_claims, normalize=True,
        ),
        "authority": copy.deepcopy(AUTHORITY),
    }
    value["recommendation_digest"] = canonical_digest(value)
    return validate_collection_recommendation(value)


def _rollout_condition(hypothesis, lifecycle_result):
    """Join the owner's terminal trace to an already qualified exact condition.

    The native lifecycle is the input, never a caller-authored failure score or
    diagnostic wrapper. Its digest binds checkpoint, recorder and unknowns.
    """
    if lifecycle_result is None:
        return None
    from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
    from tools.fr5_data_factory import validate_motion_program
    diagnostic = build_run_diagnostic(lifecycle_result)
    if diagnostic["schema_version"] != "data_factory.rollout_run_diagnostic.v1":
        raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_OWNER")
    plan = lifecycle_result["plan_envelope"]["plan"]
    resolved = plan.get("resolved_job_digest")
    matches = [base for base in hypothesis["base_conditions"]
               if base["resolved_job_digest"] == resolved]
    if len(matches) != 1:
        raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_CONDITION_UNAVAILABLE")
    # The resolver owns condition meaning; matching coordinates alone cannot join
    # a different calibration, object, camera profile or task to this run.
    receipt = next(item for item in hypothesis["resolver_receipts"]
                   if item["resolved_job_digest"] == resolved)
    program = validate_motion_program(plan.get("learned_source_program"))
    if (program["resolved_job_digest"] != resolved
            or plan.get("binding_digests") != program["binding_digests"]
            or plan.get("robot_system_id") != receipt["normalized_job"]["robot_system_id"]
            or program["robot_system_id"] != plan["robot_system_id"]
            or any(program["binding_digests"].get(key) != value
                   for key, value in receipt["input_digests"].items())):
        raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_CONDITION_BINDING")
    return canonical_digest(matches[0]["coverage_condition"])


def _unobserved_selection(source: Mapping[str, Any], report: Mapping[str, Any],
                          condition_digest: str | None = None, *, redemonstrate=False) -> dict | None:
    """Choose explicit admitted slots; a count alone cannot target a condition."""
    hypothesis, draft = source["hypothesis"], source["draft"]
    counts = _base_counts(hypothesis)
    missing = {canonical_digest(cell["condition"]) for cell in report["cells"]
               if not cell["counts"]["collected"]}
    if condition_digest is not None:
        missing = {condition_digest} if redemonstrate else missing.intersection({condition_digest})
    bases = {base["base_condition_digest"]: base for base in hypothesis["base_conditions"]}
    candidates = sorted(_candidate_slots(hypothesis, 1, _slot_template(draft)),
                        key=lambda item: item["slot_id"])
    by_id = {item["slot_id"]: item for item in candidates}
    if not set(draft["pinned"]).issubset(by_id):
        by_id.update({item["slot_id"]: item for item in
                      _candidate_slots(hypothesis, draft["requested_count"], _slot_template(draft))})
    # Mandatory pinned conditions lead the sequence even when already observed.
    # They are constraints, not claimed coverage deficits; exclusions stay exact.
    if any(identifier not in by_id or counts[by_id[identifier]["base_condition_digest"]]["pending_review"]
           for identifier in draft["pinned"]):
        return None
    slots = [by_id[identifier] for identifier in sorted(draft["pinned"])]
    selected_conditions = {canonical_digest(bases[slot["base_condition_digest"]]["coverage_condition"])
                           for slot in slots}
    capacity = min(draft["requested_count"], 100)
    added = 0
    # Reuse the compiler owner's candidate enumeration and budget accounting.
    for slot in candidates:
        if len(slots) >= capacity:
            break
        if (counts[slot["base_condition_digest"]]["pending_review"]
                or slot["slot_id"] in draft["excluded"]):
            continue
        condition = canonical_digest(bases[slot["base_condition_digest"]]["coverage_condition"])
        if condition in missing and condition not in selected_conditions:
            slots.append(slot)
            selected_conditions.add(condition)
            added += 1
    if not added:
        return None
    return {
        "source_hypothesis_digest": hypothesis["hypothesis_digest"],
        "authoring_mode": "DIRECT_EDIT", "requested_count": len(slots),
        "normalized_seed": draft["normalized_seed"],
        "pinned": copy.deepcopy(draft["pinned"]), "excluded": copy.deepcopy(draft["excluded"]),
        "direct_slots": slots,
    }


def _human_failed_chunk(lifecycle_result):
    """Existing explicit human chunk review supports a hypothesis, not a cause.

    OneJob.semantic_verdict and PickupExecutor._semantic_verdict own this shape.
    Neither unreviewed controller faults nor numeric proxies substitute for it.
    """
    if lifecycle_result is None:
        return False
    evidence = lifecycle_result["execution_evidence"]
    decision = evidence.get("semantic_decision")
    return (
        lifecycle_result.get("code") == "SEMANTIC_FAIL"
        and lifecycle_result.get("semantic_verdict") == "FAIL"
        and evidence.get("semantic_verdict") == "FAIL"
        and evidence["learned_execution"]["status"] == "COMPLETED"
        and isinstance(decision, dict)
        and set(decision) == {"source", "decided_by", "decided_at", "review_scope"}
        and decision["source"] == "HUMAN"
        and decision["review_scope"] == "FINITE_LEARNED_CHUNK"
        and isinstance(decision["decided_by"], str) and SAFE_ID.fullmatch(decision["decided_by"]) is not None
        and isinstance(decision["decided_at"], str) and RFC3339.fullmatch(decision["decided_at"]) is not None
    )


def _completed_scoped_terminal(lifecycle_result, diagnostic):
    """Recognize archived scoped execution and recorded terminal bindings.

    This establishes a canonical lineage contract only. It does not independently
    qualify terminal safety, task effectiveness, physical landing, or a data deficit.
    """
    try:
        from tools.data_factory.rollout.task_authority import SCOPE, task_scope, validate_grant

        plan = lifecycle_result["plan_envelope"]["plan"]
        trace = diagnostic["execution_trace"]
        terminal = diagnostic["mechanical_terminal"]
        terminal_plan = terminal["plan"]
        retained = lifecycle_result["execution_evidence"]
        handoff = diagnostic["task_handoff"]
        authority = retained["task_authority"]
        grant = validate_grant(authority["grant"])
        admission = authority["admission"]
        reference_consumption = trace.get("reference_consumption")
        history = diagnostic.get("execution_history", [])
        completed_outputs = len(history) + 1
        expected_admission = {
            "approval_id": "task-" + lifecycle_result["plan_digest"].removeprefix("sha256:"),
            "approved_by": grant["issued_by"], "approval_scope": SCOPE,
            "approval_expiry": None, "run_id": lifecycle_result["run_id"],
            "resolved_job_digest": plan["resolved_job_digest"],
            "plan_digest": lifecycle_result["plan_digest"], "task_grant": grant,
        }
        return (
            lifecycle_result["code"] == "MECHANICAL_TERMINAL_COMPLETE"
            and lifecycle_result.get("semantic_verdict") in {None, "PENDING"}
            and lifecycle_result["execution_evidence"].get("semantic_decision") is None
            and trace["status"] == "COMPLETED" and trace["failure_code"] is None
            and trace["terminal_phases"] == ["LEARNED_CHUNK"]
            and bool(plan["learned_proposal"]["actions"])
            and (reference_consumption is None
                 or bool(reference_consumption["completed_row_indices"]))
            and grant["revoked"] is False and grant["run_id"] == lifecycle_result["run_id"]
            and grant["scope"] == task_scope(
                plan["learned_source_program"], plan["scene_binding"], plan["learned_proposal"])
            and admission == expected_admission
            and all(item["approval"].get("approval_scope") == SCOPE
                    and item["approval"].get("task_grant") == grant for item in history)
            and terminal["status"] == "COMPLETED"
            and terminal_plan["grant_digest"] == grant["grant_digest"]
            and terminal_plan["deadline_s"] == grant["deadline_s"]
            and terminal_plan["semantic_success"] == "NOT_MEASURED"
            and handoff == {
                "status": "PLANNED", "termination_reason": handoff["termination_reason"],
                "terminal_digest": terminal_plan["terminal_digest"],
                "deadline_s": grant["deadline_s"], "semantic_success": "NOT_MEASURED",
            }
            and handoff["termination_reason"] in {
                "TASK_OUTPUT_LIMIT_REACHED", "TASK_POLICY_BUDGET_EXHAUSTED",
            }
            and completed_outputs <= grant["max_outputs"]
            and (handoff["termination_reason"] != "TASK_OUTPUT_LIMIT_REACHED"
                 or completed_outputs == grant["max_outputs"])
            and diagnostic["task_effectiveness"] == "UNKNOWN"
            and diagnostic["physical_qualification"] == "UNKNOWN"
        )
    except (ContractError, KeyError, TypeError, ValueError):
        return False


def derive_collection_recommendation(
    *, compiled_authoring: Mapping[str, Any] | None = None,
    episode_evidence: Sequence[Mapping[str, Any]], source_commit: str,
    acquisition: Mapping[str, Any] | None = None,
    rollout_lifecycle_result: Mapping[str, Any] | None = None,
    rollout_preapproval_evidence: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure the retained domain and advise a bounded first coverage pass.

    Historical aggregate counts are deliberately not merged with these episodes:
    their overlap is unknown. No semantic, vision or rollout conclusion is inferred.
    """
    from tools.data_factory.campaign_operator import validate_compiled_authoring_evidence

    if acquisition is not None:
        return _derive_acquisition_recommendation(
            acquisition=acquisition, episode_evidence=episode_evidence, source_commit=source_commit,
            rollout_lifecycle_result=rollout_lifecycle_result,
            rollout_preapproval_evidence=rollout_preapproval_evidence,
        )
    source = validate_compiled_authoring_evidence(compiled_authoring)
    manifest, hypothesis = source["manifest"], source["hypothesis"]
    condition_digest = _rollout_condition(hypothesis, rollout_lifecycle_result)
    redemonstrate = _human_failed_chunk(rollout_lifecycle_result)
    rollout_ref = {
        "availability": "UNAVAILABLE", "schema_version": None, "analysis_id": None,
        "analysis_digest": None, "reason_codes": ["NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS"],
    }
    if rollout_lifecycle_result is not None:
        from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
        diagnostic = build_run_diagnostic(rollout_lifecycle_result)
        rollout_ref = {
            "availability": "AVAILABLE", "schema_version": diagnostic["schema_version"],
            "analysis_id": diagnostic["run_id"], "analysis_digest": canonical_digest(diagnostic),
            "reason_codes": [],
        }
    summaries = _episode_summaries(episode_evidence, manifest, normalize=True)
    bases = {item["base_condition_digest"]: item for item in hypothesis["base_conditions"]}
    jobs = {item["resolved_job_digest"]: item["normalized_job"] for item in hypothesis["resolver_receipts"]}
    episodes = []
    for evidence in sorted(episode_evidence, key=lambda item: item["manifest_order_index"]):
        ledger, state = evidence["ledger"], evidence["state"]
        semantic = state["review"]["semantic_status"]
        episodes.append({
            "episode_id": ledger["episode"]["run_id"],
            "condition": bases[ledger["bindings"]["base_condition_digest"]]["coverage_condition"],
            "admission_state": (
                "HUMAN_SEMANTIC_PASS" if semantic == "PASS"
                else "PENDING_REVIEW" if semantic == "PENDING"
                else "REJECTED"
            ),
            "evidence_digests": {
                "job_spec": canonical_digest(jobs[ledger["bindings"]["resolved_job_digest"]]),
                "technical_validator_result": ledger["artifacts"]["technical"]["artifact_digest"],
                "candidate_admission": canonical_digest(evidence["candidate"]),
            },
            "trajectory_continuity": {},
        })
    previous = hypothesis["coverage_report"]
    report = build_coverage_report(
        collection_profile_id=previous["collection_profile_id"],
        domain=[cell["condition"] for cell in previous["cells"]], episodes=episodes,
    )
    report_digest = canonical_digest(report)
    claims = [{
        "claim_id": "coverage-observed", "class": "OBSERVED", "subject": "coverage",
        "value": {"metric": "COLLECTED_EPISODE_COUNT", "count": len(summaries)},
        "evidence_refs": [manifest["manifest_digest"], report_digest],
        "basis_claim_ids": [], "reason_codes": [],
    }]
    for reason, subject in UNKNOWN_REASON_SUBJECTS.items():
        if subject == "rollout" and reason != (
            "ROLLOUT_DATA_DEFICIT_UNPROVEN" if rollout_lifecycle_result is not None
            else "NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS"
        ):
            continue
        if subject in {"person", "background", "robot", "rollout"} or (
            subject == "semantic" and any(
                item["state"]["review"]["semantic_status"] == "PENDING" for item in episode_evidence
            )
        ):
            claims.append({
                "claim_id": f"{subject}-unknown", "class": "UNKNOWN", "subject": subject,
                "value": None, "evidence_refs": (
                    [rollout_ref["analysis_digest"]] if subject == "rollout" and condition_digest else []
                ), "basis_claim_ids": [],
                "reason_codes": [reason],
            })
    # One attempt per unobserved qualified condition is a finite coverage proposal,
    # not an assertion of data sufficiency or a reason to repeat approved cells.
    qualified = {canonical_digest(base["coverage_condition"]) for base in bases.values()}
    missing = [cell for cell in report["cells"] if not cell["counts"]["collected"]
               and canonical_digest(cell["condition"]) in qualified
               and (condition_digest is None or canonical_digest(cell["condition"]) == condition_digest)]
    patches = []
    if redemonstrate or missing and (condition_digest is not None or any(
        cell["condition"] == report["suggest_next"] for cell in missing
    )):
        claims.append({
            "claim_id": "coverage-suggested", "class": "SUGGESTED", "subject": "coverage",
            "value": "REDEMONSTRATE_CONDITION" if redemonstrate else "COLLECT_MORE",
            "evidence_refs": [report_digest, rollout_ref["analysis_digest"]] if redemonstrate else [report_digest],
            "basis_claim_ids": ["coverage-observed"],
            "reason_codes": ["HUMAN_REVIEWED_CHUNK_FAILURE" if redemonstrate else "COVERAGE_DEFICIT"],
        })
        selection = _unobserved_selection(source, report, condition_digest, redemonstrate=redemonstrate)
        if selection is not None:
            patches.append({
                "change_id": "reviewed-chunk-redemonstration" if redemonstrate else "cover-unobserved-conditions",
                "field": "campaign_selection",
                "value": selection, "basis_claim_ids": ["coverage-suggested"],
            })
    recommendation = build_collection_recommendation(
        recommendation_id="collection-" + canonical_digest({
            "authoring": source["authoring_digest"], "episodes": summaries,
            "quality": report_digest, "source_commit": source_commit,
            **({"rollout": rollout_ref} if condition_digest is not None else {}),
        })[7:31],
        source_commit=source_commit, campaign_manifest=manifest,
        campaign_hypothesis=hypothesis, campaign_draft=source["draft"],
        campaign_compilation_receipt=source["compilation_receipt"],
        episode_evidence=episode_evidence, data_quality_analysis=report,
        data_quality_analysis_ref={
            "availability": "AVAILABLE", "schema_version": report["schema_version"],
            "analysis_id": report["collection_profile_id"], "analysis_digest": report_digest,
            "reason_codes": [],
        },
        rollout_evidence_analysis_ref=rollout_ref,
        rollout_evidence_analysis=rollout_lifecycle_result,
        claims=claims, suggested_draft_patches=patches,
    )
    return report, recommendation


def _snapshot(value: object) -> dict[str, Any]:
    snapshot = _exact(value, SNAPSHOT_FIELDS, "COLLECTION_RECOMMENDATION_SNAPSHOT_FIELDS")
    _self_digest(snapshot, "snapshot_digest", "COLLECTION_RECOMMENDATION_SNAPSHOT_DIGEST")
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA:
        raise ContractError("COLLECTION_RECOMMENDATION_SNAPSHOT_SCHEMA")
    source_commit = snapshot["source_commit"]
    if (
        not isinstance(source_commit, str) or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_SOURCE_COMMIT")
    campaign = _exact(snapshot["campaign"], CAMPAIGN_FIELDS, "COLLECTION_RECOMMENDATION_CAMPAIGN_FIELDS")
    if campaign["schema_version"] != MANIFEST_SCHEMA:
        raise ContractError("COLLECTION_RECOMMENDATION_CAMPAIGN_SCHEMA")
    _identifier(campaign["manifest_id"], "COLLECTION_RECOMMENDATION_MANIFEST_ID")
    _digest(campaign["manifest_digest"], "COLLECTION_RECOMMENDATION_MANIFEST_DIGEST")
    episodes = snapshot["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ContractError("COLLECTION_RECOMMENDATION_EPISODES")
    normalized_episodes = []
    for index, raw in enumerate(episodes):
        episode = _exact(raw, EPISODE_SNAPSHOT_FIELDS, "COLLECTION_RECOMMENDATION_EPISODE_SNAPSHOT_FIELDS")
        if episode["manifest_order_index"] != index:
            raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_ORDER")
        _identifier(episode["run_id"], "COLLECTION_RECOMMENDATION_RUN_ID")
        _count(episode["episode_index"], "COLLECTION_RECOMMENDATION_EPISODE_INDEX")
        _identifier(episode["dataset_id"], "COLLECTION_RECOMMENDATION_DATASET_ID")
        for field in ("dataset_digest", "source_provenance_digest", "recording_quality_digest"):
            _digest(episode[field], "COLLECTION_RECOMMENDATION_EPISODE_DIGEST")
        expected_schemas = {
            "episode_ref": EPISODE_REF_SCHEMA,
            "locator": LOCATOR_SCHEMA,
            "ledger": LEDGER_SCHEMA,
            "state": STATE_SCHEMA,
            "candidate": _CANDIDATE_SCHEMA,
        }
        for name, schema in expected_schemas.items():
            nested = _exact(episode[name], SCHEMA_DIGEST_FIELDS, "COLLECTION_RECOMMENDATION_EPISODE_REF")
            if nested["schema_version"] != schema:
                raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_REF")
            _digest(nested["digest"], "COLLECTION_RECOMMENDATION_EPISODE_DIGEST")
            episode[name] = nested
        normalized_episodes.append(episode)
    unique_fields = (
        "run_id", "dataset_digest", "source_provenance_digest",
        "recording_quality_digest",
    )
    if any(
        len({item[field] for item in normalized_episodes}) != len(normalized_episodes)
        for field in unique_fields
    ) or any(
        len({item[name]["digest"] for item in normalized_episodes})
        != len(normalized_episodes)
        for name in ("episode_ref", "locator", "ledger", "state", "candidate")
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_DUPLICATE")
    snapshot["campaign"], snapshot["episodes"] = campaign, normalized_episodes
    data_quality_ref, rollout_ref = _analysis_refs(
        snapshot["data_quality_analysis_ref"], snapshot["rollout_evidence_analysis_ref"],
        normalize=False,
    )
    snapshot["data_quality_analysis_ref"] = data_quality_ref
    snapshot["rollout_evidence_analysis_ref"] = rollout_ref
    return snapshot


def validate_collection_recommendation(
    value: object, *, campaign_manifest: Mapping[str, Any] | None = None,
    campaign_hypothesis: Mapping[str, Any] | None = None,
    campaign_draft: Mapping[str, Any] | None = None,
    campaign_compilation_receipt: Mapping[str, Any] | None = None,
    episode_evidence: Sequence[Mapping[str, Any]] | None = None,
    data_quality_analysis: Mapping[str, Any] | None = None,
    rollout_evidence_analysis: Mapping[str, Any] | None = None,
    rollout_preapproval_evidence: Mapping[str, Any] | None = None,
    acquisition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a self-digested value, optionally rejoining supplied evidence."""
    if isinstance(value, Mapping) and value.get("schema_version") == ACQUISITION_SCHEMA:
        if acquisition is None or episode_evidence is None:
            raise ContractError("COLLECTION_ACQUISITION_CURRENT_INPUT_REQUIRED")
        if (value.get("input_snapshot", {}).get("rollout_evidence_analysis_ref") is not None
                and rollout_evidence_analysis is None):
            raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_LIFECYCLE_REQUIRED")
        _report, expected = _derive_acquisition_recommendation(
            acquisition=acquisition, episode_evidence=episode_evidence,
            source_commit=value.get("input_snapshot", {}).get("source_commit"),
            rollout_lifecycle_result=rollout_evidence_analysis,
            rollout_preapproval_evidence=rollout_preapproval_evidence,
        )
        if value != expected:
            raise ContractError("COLLECTION_ACQUISITION_INPUT_CHANGED")
        return expected
    recommendation = _exact(value, RECOMMENDATION_FIELDS, "COLLECTION_RECOMMENDATION_FIELDS")
    _self_digest(
        recommendation, "recommendation_digest",
        "COLLECTION_RECOMMENDATION_DIGEST",
    )
    if recommendation["schema_version"] != SCHEMA_VERSION:
        raise ContractError("COLLECTION_RECOMMENDATION_SCHEMA")
    _identifier(recommendation["recommendation_id"], "COLLECTION_RECOMMENDATION_ID")
    snapshot = _snapshot(recommendation["input_snapshot"])
    claims = _claims(recommendation["claims"], snapshot, normalize=False)
    patches = _patches(
        recommendation["suggested_draft_patches"], claims, normalize=False,
    )
    authority = _authority(recommendation["authority"])
    recommendation.update(
        input_snapshot=snapshot, claims=claims,
        suggested_draft_patches=patches, authority=authority,
    )
    campaign_evidence = (
        campaign_manifest, campaign_hypothesis, campaign_draft,
        campaign_compilation_receipt, episode_evidence,
    )
    if any(item is None for item in campaign_evidence) and any(
        item is not None for item in campaign_evidence
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_EVIDENCE_REQUIRED")
    if all(item is not None for item in campaign_evidence):
        manifest = _campaign(
            campaign_manifest, hypothesis=campaign_hypothesis,
            draft=campaign_draft, receipt=campaign_compilation_receipt,
        )
        episodes = _episode_summaries(
            episode_evidence, manifest, normalize=False,
        )
        if snapshot["campaign"] != {
            "schema_version": MANIFEST_SCHEMA,
            "manifest_id": manifest["manifest_id"],
            "manifest_digest": manifest["manifest_digest"],
        } or snapshot["episodes"] != episodes:
            raise ContractError("COLLECTION_RECOMMENDATION_SNAPSHOT_BINDING")
        checked_refs = _analysis_refs(
            snapshot["data_quality_analysis_ref"],
            snapshot["rollout_evidence_analysis_ref"],
            data_quality_analysis=data_quality_analysis,
            rollout_evidence_analysis=rollout_evidence_analysis,
            normalize=False,
        )
        _rollout_condition(campaign_hypothesis, rollout_evidence_analysis)
        if any(c["value"] == "REDEMONSTRATE_CONDITION" for c in claims) and not _human_failed_chunk(rollout_evidence_analysis):
            raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_REVIEW_REQUIRED")
        if checked_refs != (
            snapshot["data_quality_analysis_ref"],
            snapshot["rollout_evidence_analysis_ref"],
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_ANALYSIS_BINDING")
    elif data_quality_analysis is not None or rollout_evidence_analysis is not None:
        raise ContractError("COLLECTION_RECOMMENDATION_EVIDENCE_REQUIRED")
    return recommendation


def project_update_draft_intent(
    recommendation: object, *, selected_change_id: str,
    operator_view: Mapping[str, Any], intent_id: str | None = None,
    data_quality_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one update_draft intent; never consume or apply it."""
    checked = validate_collection_recommendation(recommendation)
    _analysis_ref(
        checked["input_snapshot"]["data_quality_analysis_ref"],
        owner="data_quality", artifact=data_quality_analysis, normalize=False,
    )
    _analysis_ref(
        checked["input_snapshot"]["rollout_evidence_analysis_ref"],
        owner="rollout", artifact=None, normalize=False,
    )
    _identifier(selected_change_id, "COLLECTION_RECOMMENDATION_PATCH_SELECTION")
    selected = [
        patch for patch in checked["suggested_draft_patches"]
        if patch["change_id"] == selected_change_id
    ]
    if len(selected) != 1:
        raise ContractError("COLLECTION_RECOMMENDATION_PATCH_SELECTION")
    patch = selected[0]
    if patch["field"] == "campaign_selection":
        raise ContractError("COLLECTION_RECOMMENDATION_CAMPAIGN_OWNER_REQUIRED")
    view = _exact(operator_view, VIEW_FIELDS, "COLLECTION_RECOMMENDATION_VIEW_FIELDS")
    projection = view["projection"]
    if (
        view["schema_version"] != VIEW_SCHEMA
        or not isinstance(projection, Mapping)
        or projection.get("workflow_state") != "AUTHORING"
        or not isinstance(projection.get("available_ops"), list)
        or projection["available_ops"].count("update_draft") != 1
        or view["authority"] != VIEW_AUTHORITY
        or type(view["revision"]) is not int
        or view["revision"] < 0
        or not isinstance(view["generated_at"], str)
        or RFC3339.fullmatch(view["generated_at"]) is None
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_VIEW_STATE")
    session_id = _identifier(view["session_id"], "COLLECTION_RECOMMENDATION_VIEW_SESSION")
    expected_view_digest = canonical_digest({
        "session_id": session_id,
        "revision": view["revision"],
        "projection": projection,
    })
    if view["view_digest"] != expected_view_digest:
        raise ContractError("COLLECTION_RECOMMENDATION_VIEW_STALE")
    draft = projection.get("draft")
    if not isinstance(draft, Mapping):
        raise ContractError("COLLECTION_RECOMMENDATION_VIEW_DRAFT")
    draft_id = _identifier(draft.get("draft_id"), "COLLECTION_RECOMMENDATION_VIEW_DRAFT")
    _count(draft.get("revision"), "COLLECTION_RECOMMENDATION_VIEW_DRAFT")
    field = patch["field"]
    selection = draft.get("selection")
    if field in {"selection", "split"}:
        axis, selected_value = (
            next(iter(patch["value"].items()))
            if field == "selection" else ("split", patch["value"])
        )
        catalog = projection.get("catalog")
        axes = catalog.get("axes") if isinstance(catalog, Mapping) else None
        options = axes.get(axis) if isinstance(axes, Mapping) else None
        matches = [
            option for option in options or []
            if isinstance(option, Mapping) and option.get("id") == selected_value
        ] if isinstance(options, list) else []
        if (
            not isinstance(selection, Mapping)
            or not isinstance(selection.get(axis), str)
            or len(matches) != 1 or matches[0].get("available") is not True
        ):
            raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
    elif field in {"requested_count", "repeat"}:
        _count(
            draft.get(field), "COLLECTION_RECOMMENDATION_VIEW_DRAFT",
            positive=True,
        )
    elif (
        draft.get("authoring_mode") != "ASSISTED"
        or not isinstance(projection.get("sampling_provenance"), Mapping)
        or not isinstance(
            projection["sampling_provenance"].get(
                "state_space_design_profile"
            ),
            Mapping,
        )
    ):
        raise ContractError("COLLECTION_RECOMMENDATION_PATCH_VALUE")
    if intent_id is None:
        intent_id = "recommendation-" + canonical_digest({
            "recommendation_digest": checked["recommendation_digest"],
            "change_id": selected_change_id,
            "view_digest": view["view_digest"],
        })[7:31]
    _identifier(intent_id, "COLLECTION_RECOMMENDATION_INTENT_ID")
    return {
        "schema_version": INTENT_SCHEMA,
        "intent_id": intent_id,
        "session_id": session_id,
        "view_revision": view["revision"],
        "view_digest": view["view_digest"],
        "op": "update_draft",
        "payload": {
            "draft_id": draft_id,
            field: copy.deepcopy(patch["value"]),
        },
    }


def project_campaign_update_intent(
    recommendation: object, *, compiled_authoring: Mapping[str, Any],
    operator_view: Mapping[str, Any], data_quality_analysis: Mapping[str, Any],
    rollout_lifecycle_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project exact condition selection to the existing CampaignOperator CAS.

    This does not rewrite the Collection UI's observed object pose, compile a
    plan, create a campaign, or obtain physical/training authority.
    """
    from tools.data_factory.campaign_operator import (
        PROJECTION_SCHEMA, validate_compiled_authoring_evidence,
    )

    checked = validate_collection_recommendation(recommendation)
    source = validate_compiled_authoring_evidence(compiled_authoring)
    _analysis_ref(checked["input_snapshot"]["data_quality_analysis_ref"],
                  owner="data_quality", artifact=data_quality_analysis, normalize=False)
    _analysis_ref(checked["input_snapshot"]["rollout_evidence_analysis_ref"],
                  owner="rollout", artifact=rollout_lifecycle_result, normalize=False)
    condition_digest = _rollout_condition(source["hypothesis"], rollout_lifecycle_result)
    if any(c["value"] == "REDEMONSTRATE_CONDITION" for c in checked["claims"]) and not _human_failed_chunk(rollout_lifecycle_result):
        raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_REVIEW_REQUIRED")
    selection = _unobserved_selection(source, data_quality_analysis, condition_digest,
                                      redemonstrate=_human_failed_chunk(rollout_lifecycle_result))
    patches = [patch for patch in checked["suggested_draft_patches"]
               if patch["field"] == "campaign_selection"]
    if (source["manifest"]["manifest_digest"] != checked["input_snapshot"]["campaign"]["manifest_digest"]
            or selection is None or len(patches) != 1 or patches[0]["value"] != selection):
        raise ContractError("COLLECTION_RECOMMENDATION_SELECTION_BINDING")
    view = _exact(operator_view, VIEW_FIELDS, "COLLECTION_RECOMMENDATION_VIEW_FIELDS")
    projection = view["projection"]
    if (view["schema_version"] != VIEW_SCHEMA or view["authority"] != VIEW_AUTHORITY
            or not isinstance(projection, Mapping) or projection.get("schema_version") != PROJECTION_SCHEMA
            or not isinstance(projection.get("draft"), Mapping)
            or projection["draft"].get("source") != source["draft"]["source"]
            or type(view["revision"]) is not int or view["revision"] < 0
            or not isinstance(view["generated_at"], str)
            or RFC3339.fullmatch(view["generated_at"]) is None):
        raise ContractError("COLLECTION_RECOMMENDATION_CAMPAIGN_VIEW")
    _identifier(view["session_id"], "COLLECTION_RECOMMENDATION_VIEW_SESSION")
    if view["view_digest"] != canonical_digest({
        "session_id": view["session_id"], "revision": view["revision"], "projection": projection,
    }):
        raise ContractError("COLLECTION_RECOMMENDATION_VIEW_STALE")
    # A fresh CAS cannot make advice derived from an older selection current.
    # Ignore unrelated view revisions, but preserve every later selection edit.
    if any(projection["draft"].get(key) != source["draft"].get(key) for key in (
        "selector", "requested_count", "normalized_seed", "pinned", "excluded", "direct_slots",
    )):
        raise ContractError("COLLECTION_RECOMMENDATION_DRAFT_CHANGED")
    return {
        "schema_version": INTENT_SCHEMA,
        "intent_id": "recommendation-" + canonical_digest([
            checked["recommendation_digest"], view["view_digest"],
        ])[7:31],
        "session_id": view["session_id"], "view_revision": view["revision"],
        "view_digest": view["view_digest"], "op": "update_draft",
        "payload": {key: copy.deepcopy(value) for key, value in selection.items()
                    if key != "source_hypothesis_digest"},
    }


__all__ = [
    "AUTHORITY", "SCHEMA_VERSION", "SNAPSHOT_SCHEMA",
    "build_collection_recommendation", "project_update_draft_intent",
    "derive_collection_recommendation",
    "project_campaign_update_intent",
    "validate_collection_recommendation",
]


ACQUISITION_SCHEMA = "data_factory.collection_recommendation.v2"


def _rollout_transition(context, selected, plans, preapproval, source_job, program, motion_preset):
    """Join original resolver and language endpoints to the current native domain."""
    from datetime import datetime, timezone
    from tools.data_factory.operator.catalog import resolve_workspace_cycle_selections
    from tools.data_factory.quality.coverage_report import RESOLVED_INPUT_DIGEST_FIELDS
    from tools.data_factory.task_recipe import validate_episode_instruction_binding
    from tools.fr5_data_factory import normalize_job_spec

    if program["schema_version"] != "fr5.motion_program.v4":
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_TRANSITION_SOURCE_REQUIRED")
    receipt = _exact(preapproval.get("destination_resolved_inputs"), frozenset({
        "normalized_job", "input_digests", "resolved_job_digest",
    }), "COLLECTION_ACQUISITION_ROLLOUT_DESTINATION_REQUIRED")
    destination_job = normalize_job_spec(receipt["normalized_job"], now=datetime.min.replace(tzinfo=timezone.utc))
    inputs = _exact(receipt["input_digests"], frozenset(RESOLVED_INPUT_DIGEST_FIELDS),
                    "COLLECTION_ACQUISITION_ROLLOUT_DESTINATION_BINDING")
    if (destination_job != receipt["normalized_job"]
            or canonical_digest({"job": destination_job, "input_digests": inputs}) != receipt["resolved_job_digest"]
            or receipt["resolved_job_digest"] != program["destination_resolved_job_digest"]
            or any(program["destination_binding_digests"].get(k) != v for k, v in inputs.items())):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_DESTINATION_BINDING")
    instruction = validate_episode_instruction_binding(preapproval.get("episode_instruction_binding"))
    if (preapproval.get("episode_instruction_binding_digest") != instruction["binding_digest"]
            or instruction["task_binding"]["task_id"] != "pick_place"
            or instruction["object_profile_id"] != selected["object_id"]
            or instruction["object_profile_digest"] != program["binding_digests"]["object_profile"]
            or any(p["learned_proposal"]["instruction"] != instruction["instruction"] for p in plans)):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_INSTRUCTION_BINDING")
    for plan in plans:
        release = plan.get("scene_binding", {}).get("release_slot")
        if (not isinstance(release, Mapping)
                or release.get("pose") != instruction["task_binding"]["spatial_bindings"][1]["pose"]):
            raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_DESTINATION_BINDING")
    cycle = resolve_workspace_cycle_selections(context["catalog"], selected, 1, require_executable=False)
    for endpoint, job, bindings, selection in zip(
        instruction["task_binding"]["spatial_bindings"], (source_job, destination_job),
        (program["binding_digests"], program["destination_binding_digests"]), cycle,
    ):
        combo = next(c for c in context["catalog"]["combinations"]
                     if c["combination_digest"] == selection["combination_digest"])
        domains = [d for d in context["catalog"]["workspace_domains"] if
                   (d["workspace_id"], d["frame_id"], d["object_id"]) ==
                   (selection["workspace_id"], selection["frame_id"], selection["object_id"])]
        if len(domains) != 1:
            raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_ENDPOINT_BINDING")
        domain = domains[0]
        expected_motion = (combo["source_digests"]["motion"] if motion_preset is None else
                           motion_preset["qualifications"][selection["motion_id"]])
        expected_endpoint = {"workspace_id": selection["workspace_id"],
            "cell_calibration_id": selection["frame_id"],
            "cell_calibration_digest": combo["source_digests"]["cell"], "motion_recipe_digest": expected_motion}
        if (expected_endpoint not in program["endpoint_bindings"]
                or endpoint["workspace_id"] != job["place_id"] or endpoint["workspace_id"] != selection["workspace_id"]
                or endpoint["frame_id"] != job["cell_calibration_id"] or endpoint["frame_id"] != selection["frame_id"]
                or endpoint["pose"] != {k: job[k] for k in ("place_id", "yaw_deg", "x_mm", "y_mm")}
                or endpoint["sheet_digest"] != job["sheet_manifest_digest"]
                or endpoint["sheet_digest"] != bindings["selected_sheet"]
                or endpoint["family_digest"] != domain["a4_family_digest"]
                or endpoint["region_binding"] != {k: domain["coverage_region"][k] for k in endpoint["region_binding"]}
                or bindings["yaw0_sheet"] != domain["yaw0_manifest_digest"]
                or any(bindings[k] != combo["source_digests"][name] for k, name in {
                    "selected_sheet": "selected_sheet", "yaw0_sheet": "yaw0_sheet", "cell_calibration": "cell",
                    "object_profile": "object", "grasp_profile": "grasp", "collection_profile": "camera_profile",
                }.items())
                or bindings["motion_qualification"] != expected_motion
                or bindings.get("motion_preset") != (None if motion_preset is None else motion_preset["digest"])
                or "motion_preset_trial" in bindings
                or any(job[k] != source_job[k] for k in (
                    "task", "robot_system_id", "object_profile_id", "grasp_profile_id", "collection_profile_id",
                    "instruction", "episode_intent"))):
            raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_ENDPOINT_BINDING")
    if destination_job["yaw_deg"] != source_job["yaw_deg"]:
        # Native Collection records a yaw-preserving release, then optionally
        # repositions for the next source. That is not a rotated failed release.
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_DESTINATION_YAW_UNSUPPORTED")
    return {"destination": copy.deepcopy(instruction["task_binding"]["spatial_bindings"][1]["pose"]),
            "episode_instruction_binding": instruction,
            "destination_resolved_job_digest": receipt["resolved_job_digest"],
            "endpoint_bindings": copy.deepcopy(program["endpoint_bindings"])}


def _current_rollout_condition(context, selected, lifecycle_result, preapproval, motion_preset=None):
    """Bind a reviewed failure or completed scoped attempt to its original condition.

    Original resolver fields own source XY/yaw; the current scene owns present
    placement. A later native recovery does not rewrite the historical result.
    """
    from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
    from tools.data_factory.scene_state import validate_scene_binding
    from datetime import datetime, timezone
    from tools.data_factory.quality.coverage_report import RESOLVED_INPUT_DIGEST_FIELDS
    from tools.fr5_data_factory import normalize_job_spec, validate_motion_program

    diagnostic = build_run_diagnostic(lifecycle_result)
    if diagnostic["schema_version"] != "data_factory.rollout_run_diagnostic.v1":
        # Native retained lifecycle evidence is not yet a collection analysis.
        raise ContractError("COLLECTION_RECOMMENDATION_ROLLOUT_OWNER")
    plan = lifecycle_result["plan_envelope"]["plan"]
    program = validate_motion_program(plan.get("learned_source_program"))
    binding = validate_scene_binding(plan.get("scene_binding"))
    scene = context["scene_state"]
    if (binding["object_instance_id"] != context["object_instance_id"]
            or lifecycle_result.get("scene_binding") != binding):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_OBJECT_MISMATCH")
    if not isinstance(preapproval, Mapping) or preapproval.get("schema_version") != "data_factory.preapproval_evidence.v4":
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_SOURCE_REQUIRED")
    envelope = preapproval.get("plan_envelope")
    plans = [plan, *[item["plan_envelope"]["plan"] for item in diagnostic.get("execution_history", [])]]
    if (not isinstance(envelope, Mapping) or envelope.get("plan") not in plans
            or preapproval.get("plan_envelope_digest") != canonical_digest(envelope)
            or preapproval.get("plan_digest") != canonical_digest(envelope["plan"])
            or preapproval.get("run_id") != diagnostic["run_id"]):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_SOURCE_BINDING")
    original_binding = validate_scene_binding(envelope["plan"].get("scene_binding"))
    if original_binding["object_instance_id"] != binding["object_instance_id"]:
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_OBJECT_MISMATCH")
    receipt = _exact(preapproval.get("resolved_inputs"), frozenset({
        "normalized_job", "input_digests", "resolved_job_digest",
    }), "COLLECTION_ACQUISITION_ROLLOUT_SOURCE_REQUIRED")
    # Historical normalization is not renewed execution approval.
    job = normalize_job_spec(receipt["normalized_job"], now=datetime.min.replace(tzinfo=timezone.utc))
    inputs = _exact(receipt["input_digests"], frozenset(RESOLVED_INPUT_DIGEST_FIELDS),
                    "COLLECTION_ACQUISITION_ROLLOUT_SOURCE_BINDING")
    if (job != receipt["normalized_job"]
            or canonical_digest({"job": job, "input_digests": inputs}) != receipt["resolved_job_digest"]
            or receipt["resolved_job_digest"] != plan.get("resolved_job_digest")
            or preapproval.get("resolved_job_digest") != receipt["resolved_job_digest"]
            or any(program["binding_digests"].get(key) != value for key, value in inputs.items())):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_SOURCE_BINDING")
    if selected["task_id"] == "pick_place" and job["place_id"] != selected["workspace_id"]:
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_SOURCE_REPOSITION_REQUIRED")
    combination = next(item for item in context["catalog"]["combinations"]
                       if item["combination_digest"] == selected["combination_digest"])
    transition = (_rollout_transition(context, selected, plans, preapproval, job, program, motion_preset)
                  if selected["task_id"] == "pick_place" else {})
    required = {"cell_calibration": "cell", "object_profile": "object", "grasp_profile": "grasp",
                "collection_profile": "camera_profile", "motion_qualification": "motion"}
    expected_digests = dict(combination["source_digests"])
    if motion_preset is not None:
        expected_digests["motion"] = motion_preset["qualifications"][selected["motion_id"]]
    if (plan.get("resolved_job_digest") != program["resolved_job_digest"]
            or plan.get("binding_digests") != program["binding_digests"]
            or plan.get("robot_system_id") != scene["robot_system_id"]
            or program["robot_system_id"] != scene["robot_system_id"]
            or program["binding_digests"].get("motion_preset") != (
                None if motion_preset is None else motion_preset["digest"])
            or "motion_preset_trial" in program["binding_digests"]
            or any(program["binding_digests"][key] != expected_digests.get(name)
                   for key, name in required.items())):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_CONTEXT_MISMATCH")
    if (job["task"] != selected["task_id"] or job["robot_system_id"] != scene["robot_system_id"]
            or any(job[key] != selected[field] for key, field in {
                "place_id": "workspace_id", "cell_calibration_id": "frame_id",
                "object_profile_id": "object_id", "grasp_profile_id": "grasp_id",
                "collection_profile_id": "camera_profile_id",
            }.items())):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_TASK_MISMATCH")
    reviewed_failure = _human_failed_chunk(lifecycle_result)
    lineage_validation = (
        not reviewed_failure and _completed_scoped_terminal(lifecycle_result, diagnostic)
    )
    if not reviewed_failure and not lineage_validation:
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_REDEMONSTRATION_UNPROVEN")
    condition = {
        **transition,
        "resolved_job_digest": program["resolved_job_digest"], "task_id": selected["task_id"],
        "source": {key: job[key] for key in ("place_id", "yaw_deg", "x_mm", "y_mm")},
        "preapproval_evidence_digest": canonical_digest(preapproval),
        "original_scene_binding": original_binding, "source_plan_digest": preapproval["plan_digest"],
        "review_scope": "FINITE_LEARNED_CHUNK" if reviewed_failure else None,
        "suggestion": (
            "REDEMONSTRATE_CONDITION" if reviewed_failure
            else "VALIDATE_ORIGINAL_CONDITION"
        ),
        "reason_code": (
            "HUMAN_REVIEWED_CHUNK_FAILURE" if reviewed_failure
            else "COMPLETED_SCOPED_ATTEMPT_LINEAGE"
        ),
        "task_effectiveness": "UNKNOWN", "data_deficit": "UNKNOWN",
    }
    if lineage_validation:
        condition["purpose"] = "ORIGINAL_CONDITION_VALIDATION"
    return diagnostic, condition


def _derive_acquisition_recommendation(*, acquisition, episode_evidence, source_commit,
                                      rollout_lifecycle_result=None, rollout_preapproval_evidence=None):
    """Current-source coverage advice, independent of historical authoring.

    Native ledger/DQA own observations; native samplers own the finite poses.
    This does not attest device availability, motion qualification or utility.
    """
    from collections import Counter
    from tools.data_factory.collection_seed import derive_domain_seed, validate_campaign_seed
    from tools.data_factory.operator.catalog import (
        validate_operator_selection, project_assisted_poses, project_direct_poses,
        project_workspace_cycle_poses, resolve_workspace_cycle_selections,
        project_yaw_sample_bindings, project_state_space_cells,
        selected_motion_preset, validate_operator_pose, validate_yaw_preserving_transitions,
    )
    from tools.data_factory.scene_state import _validate as validate_scene

    required = {"catalog", "selection", "scene_state", "object_instance_id",
                "requested_count", "normalized_seed", "repeat"}
    if isinstance(acquisition, Mapping) and "motion_preset" in acquisition:
        required.add("motion_preset")
    context = _exact(acquisition, required, "COLLECTION_ACQUISITION_INPUT_FIELDS")
    context = {key: value for key, value in context.items() if key != "motion_preset" or value is not None}
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)):
        raise ContractError("COLLECTION_RECOMMENDATION_SOURCE_COMMIT")
    _identifier(context["object_instance_id"], "COLLECTION_ACQUISITION_SOURCE")
    catalog = context["catalog"]
    selected = validate_operator_selection(catalog, context["selection"], require_executable=False)
    task = selected["task_id"]
    if task not in {"pickup_e2e", "pick_place"}:
        raise ContractError("COLLECTION_ACQUISITION_TASK_UNSUPPORTED")
    count, repeat = context["requested_count"], context["repeat"]
    if any(type(value) is not int or not 1 <= value <= 100 for value in (count, repeat)):
        raise ContractError("COLLECTION_ACQUISITION_BUDGET")
    seed = validate_campaign_seed(context["normalized_seed"])
    scene = context["scene_state"]
    if not isinstance(scene, dict):
        raise ContractError("COLLECTION_ACQUISITION_SCENE")
    scene = validate_scene(scene, scene.get("robot_system_id"))
    instance = scene["objects"].get(context["object_instance_id"])
    if (rollout_lifecycle_result is not None and task == "pick_place"
            and instance is not None and instance["state"] == "ON_SURFACE"
            and instance["pose"]["place_id"] != selected["workspace_id"]):
        raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_SOURCE_REPOSITION_REQUIRED")
    if (instance is None or instance["state"] != "ON_SURFACE"
            or instance["object_profile_id"] != selected["object_id"]
            or instance["pose"]["place_id"] != selected["workspace_id"]):
        raise ContractError("COLLECTION_ACQUISITION_SOURCE")
    source_pose = instance["pose"]
    cycle = (resolve_workspace_cycle_selections(catalog, selected, count, require_executable=False)
             if task == "pick_place" else [selected] * count)
    preset = selected_motion_preset(catalog, context.get("motion_preset"))
    motion_preset = None
    if preset is not None:
        qualifications = {}
        for endpoint in cycle:
            qualified = preset["qualifications"].get(endpoint["motion_id"])
            if qualified is None:
                raise ContractError("MOTION_PRESET_QUALIFICATION_REQUIRED")
            qualifications[endpoint["motion_id"]] = qualified["digest"]
        motion_preset = {**context["motion_preset"], "qualifications": qualifications}
    rollout = None if rollout_lifecycle_result is None else _current_rollout_condition(
        context, selected, rollout_lifecycle_result, rollout_preapproval_evidence, motion_preset,
    )
    combinations = {item["combination_digest"]: item for item in catalog["combinations"]}
    endpoints = {item["workspace_id"]: combinations[item["combination_digest"]] for item in cycle}
    evidence_rows, reports, compatible, excluded = [], [], [], []
    groups, profile_ids, seen, seen_refs = {}, {}, set(), set()
    for evidence in episode_evidence:
        # Each run has its own immutable manifest. Never synthesize absent
        # compiled_authoring_evidence or require unrelated campaigns to match.
        manifest = evidence["artifacts"]["manifest"]
        summary, identity = _episode_snapshot(evidence, manifest)
        ledger, state = evidence["ledger"], evidence["state"]
        recording = (ledger["dataset"]["dataset_root"], ledger["dataset"]["repo_id"],
                     ledger["episode"]["episode_index"])
        if recording in seen or identity in seen_refs:
            raise ContractError("COLLECTION_RECOMMENDATION_EPISODE_DUPLICATE")
        seen.add(recording)
        seen_refs.add(identity)
        condition = evidence["artifacts"]["intent"]["base_condition"]["coverage_condition"]
        semantic = state["review"]["semantic_status"]
        row = {"episode_id": ledger["episode"]["run_id"], "condition": condition,
               "admission_state": "HUMAN_SEMANTIC_PASS" if semantic == "PASS" else
                   "PENDING_REVIEW" if semantic == "PENDING" else "REJECTED",
               "evidence_digests": {
                   "job_spec": ledger["bindings"]["resolved_job_digest"],
                   "technical_validator_result": ledger["artifacts"]["technical"]["artifact_digest"],
                   "candidate_admission": canonical_digest(evidence["candidate"])},
               "trajectory_continuity": {}}
        # Per-manifest DQA retains distinct task/profile domains and counts each
        # recording once. It measures observed conditions, not a recovered plan.
        groups.setdefault(manifest["manifest_digest"], []).append(row)
        retained_profile = evidence["artifacts"]["intent"]["fixed_contract"].get("feature_contract", {}).get("collection_profile_id")
        if retained_profile is None:
            retained_profile = next((item["camera_profile_id"] for item in combinations.values()
                                     if item["source_digests"]["camera_profile"] == condition["collection_profile_digest"]),
                                    "unidentified-" + condition["collection_profile_digest"][7:23])
        profile_ids[manifest["manifest_digest"]] = retained_profile
        reference = {"recording": {"dataset_root": recording[0], "repo_id": recording[1],
                                    "episode_index": recording[2]},
                     "manifest_digest": manifest["manifest_digest"], "episode": summary}
        evidence_rows.append(reference)
        endpoint = endpoints.get(condition["place_id"])
        reason = None
        if condition["task"] != task:
            reason = "DIFFERENT_TASK"
        elif (endpoint is None or condition["robot_system_id"] != scene["robot_system_id"]
              or condition["object_profile_id"] != selected["object_id"]
              or condition["grasp_profile_id"] != selected["grasp_id"]
              or condition["collection_profile_digest"] != endpoint["source_digests"]["camera_profile"]
              or condition["cell_calibration_id"] != endpoint["frame_id"]
              or condition["cell_calibration_digest"] != endpoint["source_digests"]["cell"]):
            reason = "INCOMPATIBLE_COLLECTION_DOMAIN"
        else:
            bindings = evidence["artifacts"]["staging_manifest"]["binding_digests"]
            if any(bindings[name + "_profile_digest"] != endpoint["source_digests"][name]
                   for name in ("object", "grasp")):
                reason = "INCOMPATIBLE_OBJECT_OR_GRASP"
        if reason is None and semantic != "PASS":
            reason = "SEMANTIC_PASS_UNAVAILABLE"
        if reason:
            excluded.append({"recording": reference["recording"], "reason_code": reason})
        else:
            compatible.append(row)
    if not compatible:
        raise ContractError("COLLECTION_ACQUISITION_NO_COMPATIBLE_EVIDENCE")
    for manifest_digest, rows in sorted(groups.items()):
        domain = {canonical_digest(row["condition"]): row["condition"] for row in rows}
        report = build_coverage_report(collection_profile_id=profile_ids[manifest_digest],
                                       domain=list(domain.values()), episodes=sorted(rows, key=lambda x:x["episode_id"]))
        reports.append({"manifest_digest": manifest_digest, "report": report,
                        "report_digest": canonical_digest(report)})
    compatible_domain = {canonical_digest(row["condition"]): row["condition"] for row in compatible}
    compatible_report = build_coverage_report(
        collection_profile_id=selected["camera_profile_id"],
        domain=list(compatible_domain.values()),
        episodes=sorted(compatible, key=lambda row: row["episode_id"]),
    )
    analysis = {"campaign_reports": reports, "compatible_coverage": compatible_report}
    spatial_seed, yaw_seed = derive_domain_seed(seed, "spatial"), derive_domain_seed(seed, "yaw")
    projector = project_workspace_cycle_poses if task == "pick_place" else project_assisted_poses
    poses = projector(catalog, selected, source_pose, count, repeat=repeat,
                      normalized_seed=spatial_seed, yaw_sampling_seed=yaw_seed)
    authoring_mode = "ASSISTED"
    if rollout is not None and task == "pick_place":
        target = rollout[1]
        # The cycle keeps current placement first. When recovery moved the
        # object, two native transitions return to the original source before
        # replaying the original directed pair; no Scene declaration is made.
        index = 0 if source_pose == target["source"] else 2
        if count <= index:
            raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_BUDGET_INSUFFICIENT")
        poses[index:index + 2] = [copy.deepcopy(target["source"]), copy.deepcopy(target["destination"])]
        poses = [validate_operator_pose(catalog, endpoint, pose) for endpoint, pose in zip(cycle, poses)]
        validate_yaw_preserving_transitions(catalog, cycle, poses)
        authoring_mode = "DIRECT_EDIT"
    elif rollout is not None and rollout[1]["source"] not in poses[:count]:
        # Recovery can change the current pose. Preserve that first condition,
        # then reuse native direct authoring for the original reviewed source;
        # neither a lucky random sample nor resetting scene history is needed.
        if count < 2:
            raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_BUDGET_INSUFFICIENT")
        direct = [rollout[1]["source"]]
        for pose in poses[1:]:
            if pose != source_pose and pose not in direct and len(direct) < count - 1:
                direct.append(pose)
        poses = project_direct_poses(catalog, selected, source_pose, direct, count)
        authoring_mode = "DIRECT_EDIT"
    if poses[0] != source_pose:
        raise ContractError("COLLECTION_ACQUISITION_SOURCE_NOT_PRESERVED")
    yaw_bindings = (project_yaw_sample_bindings(catalog, cycle, poses, yaw_seed, repeat=repeat)
                    if authoring_mode == "ASSISTED" else [None] * len(poses))
    cells = project_state_space_cells(catalog, cycle, poses)
    source_poses = poses[:count]
    conditions = [{"order_index": i, "source": source_poses[i],
                   "destination": poses[i + 1] if task == "pick_place" else None,
                   "state_space_cell": cells[i], "yaw_sample_binding": yaw_bindings[i]}
                  for i in range(count)]
    observed = {place: sum(cell["counts"]["human_semantic_pass"]
                           for cell in compatible_report["cells"] if cell["condition"]["place_id"] == place)
                for place in sorted(endpoints)}
    suggested = dict(sorted(Counter(p["place_id"] for p in source_poses).items()))
    snapshot = {"source_commit": source_commit,
                "implementation_verification": "CALLER_SUPPLIED_UNVERIFIED",
                "context_digest": canonical_digest(context),
                "scene_binding": {"scene_state_digest": canonical_digest(scene),
                                  "revision": scene["revision"],
                                  "object_instance_id": context["object_instance_id"]},
                "episodes": sorted(evidence_rows, key=lambda row: tuple(row["recording"].values())),
                "data_quality_analysis_digest": canonical_digest(analysis)}
    if motion_preset is not None:
        snapshot["motion_preset"] = motion_preset
    if rollout is not None:
        diagnostic, target = rollout
        target["condition_indices"] = [item["order_index"] for item in conditions if item["source"] == target["source"]
                                       and (task != "pick_place" or item["destination"] == target["destination"])]
        if not target["condition_indices"]:
            raise ContractError("COLLECTION_ACQUISITION_ROLLOUT_CONDITION_NOT_PROPOSED")
        snapshot.update(rollout_evidence_analysis_ref={
            "availability": "AVAILABLE", "schema_version": diagnostic["schema_version"],
            "analysis_id": diagnostic["run_id"], "analysis_digest": canonical_digest(diagnostic), "reason_codes": [],
        }, rollout_condition=target)
    snapshot["snapshot_digest"] = canonical_digest(snapshot)
    recommendation = {
        "schema_version": ACQUISITION_SCHEMA, "input_snapshot": snapshot,
        "selection": selected,
        "sampling": {"requested_count": count, "normalized_seed": seed, "repeat": repeat,
                     "authoring_mode": authoring_mode},
        "object_poses": poses, "conditions": conditions,
        "observed_semantic_pass_by_source": observed,
        "proposed_by_source": suggested,
        "excluded_evidence": sorted(excluded, key=lambda row: tuple(row["recording"].values())),
        "reason_codes": ["TASK_SEPARATED_SUCCESS_COVERAGE", "NATIVE_BALANCED_STATE_SPACE",
                         "CURRENT_SOURCE_PRESERVED", "CALLER_BUDGET"],
        "limitations": ["Coverage is a utility proxy, not a learned value ranking.",
                        "Native balanced sampling is not an optimized missing-condition selector; evidence determines compatibility and coverage reasons, not a fitted seed.",
                        "Different motion recipes/speeds remain historical evidence, not current qualification.",
                        "Source and destination continuity is planned, not observed robot execution.",
                        "Current camera, motion, scene and admission checks belong to Collection.",
                        "This advice never repartitions or authorizes training data."],
        "authority": copy.deepcopy(AUTHORITY),
    }
    if rollout is not None:
        target = rollout[1]
        failure_targeted = target.get("purpose") != "ORIGINAL_CONDITION_VALIDATION"
        recommendation["reason_codes"].extend(
            ["HUMAN_REVIEWED_CHUNK_FAILURE",
             "REDEMONSTRATE_ORIGINAL_TRANSITION" if task == "pick_place" else "REDEMONSTRATE_ORIGINAL_SOURCE"]
            if failure_targeted else
            ["COMPLETED_SCOPED_ATTEMPT_LINEAGE",
             "VALIDATE_ORIGINAL_TRANSITION" if task == "pick_place" else "VALIDATE_ORIGINAL_SOURCE"]
        )
        recommendation["limitations"].append(
            "The original condition is a human-reviewed chunk re-demonstration hypothesis; "
            "task effectiveness and causal data deficit remain unknown."
            if failure_targeted else
            "The original condition is proposed only to validate a completed scoped attempt's lineage; "
            "task effectiveness, physical landing and causal data deficit remain unknown."
        )
        if authoring_mode == "DIRECT_EDIT":
            recommendation["reason_codes"].remove("NATIVE_BALANCED_STATE_SPACE")
            recommendation["reason_codes"].append(
                "NATIVE_DIRECT_REDEMONSTRATION" if failure_targeted
                else "NATIVE_DIRECT_ORIGINAL_CONDITION_VALIDATION"
            )
            recommendation["limitations"][1] = (
                "Remaining sampled coverage is not an optimized missing-condition selector or fitted utility ranking."
            )
            recommendation["limitations"].append(
                "Native direct authoring preserves current placement and the original condition within budget. "
                "Intervening cycle transitions and any next-source reposition remain subject to native qualification. "
                "Direct conditions have no stochastic yaw binding."
            )
    recommendation["recommendation_digest"] = canonical_digest(recommendation)
    return analysis, recommendation
