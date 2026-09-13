from __future__ import annotations

import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import threading
from types import SimpleNamespace
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from tools.data_factory.collection_recommendation import (
    AUTHORITY,
    build_collection_recommendation,
    project_update_draft_intent,
    project_campaign_update_intent,
    validate_collection_recommendation,
    derive_collection_recommendation,
)
from tools.data_factory.collection_recommendation_io import recommend_stored_collection, main as recommend_main
from tools.data_factory.campaign_operator import (
    AUTHORING_EVIDENCE_SCHEMA, CampaignOperator, SIDE_EFFECT_COUNTERS,
)
from tools.data_factory import run_job
from tools.data_factory.campaign_authoring import (
    DRAFT_SCHEMA_V2,
    campaign_cell_id,
    compile_collection_campaign,
)
from tools.data_factory.operator.workflow.intents import INTENT_SCHEMA, OperatorIntentCore
from tools.data_factory.operator.workflow.application import (
    CollectionOperatorApplication,
)
from tools.data_factory.operator.catalog import load_operator_catalog
from tools.data_factory.operator.web import projection
from tools.fr5_data_factory import ContractError, canonical_digest
from .operator.fixtures import draft as campaign_draft, hypothesis, motion, payload


COMMIT = "f0f380979d24711acca22e8e53da1e7985e0d7ad"
NOW = datetime(2026, 9, 4, 5, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[2]


def digest(value: object) -> str:
    return canonical_digest(value)


def redigest(value: dict, field: str) -> dict:
    value[field] = digest({key: item for key, item in value.items() if key != field})
    return value


def unavailable(reason: str) -> dict:
    return {
        "availability": "UNAVAILABLE",
        "schema_version": None,
        "analysis_id": None,
        "analysis_digest": None,
        "reason_codes": [reason],
    }


class RecommendationFixture:
    def __init__(self, *, dataset_root="/dataset/test", evidence_root="/evidence") -> None:
        self.dataset_root = dataset_root
        self.evidence_root = evidence_root
        state_space_profile = redigest({
            "schema_version": "data_factory.state_space_design_profile.v1",
            "state_space_design_profile_id": "design-r1",
            "object_profile_id": "object-r1",
            "object_profile_digest": digest("object"),
            "grasp_profile_id": "grasp-r1",
            "grasp_profile_digest": digest("grasp"),
            "yaw_sampling_profile_id": "yaw-r1",
            "yaw_sampling_profile_digest": digest("yaw"),
            "spatial_strata": {"columns": 2, "rows": 1},
            "yaw_cdf_strata": 2,
            "assignment": "ROTATING_BALANCED_FRACTIONAL_FACTORIAL",
            "execution_order": "CONTIGUOUS_YAW_BLOCKS",
            "initial_source_policy": "CONDITION_ON_OBSERVED_SOURCE",
        }, "profile_digest")
        self.hypothesis = hypothesis()
        self.draft = campaign_draft(self.hypothesis, count=2)
        self.draft.update(
            schema_version=DRAFT_SCHEMA_V2,
            state_space_design_profile=state_space_profile,
        )
        self.manifest, self.receipt = compile_collection_campaign(
            self.draft, hypothesis=self.hypothesis,
        )
        self.evidence = [self.episode(0, 10), self.episode(1, 11)]
        manifest_digest = self.manifest["manifest_digest"]
        self.claims = [
            self.unknown("person-unknown", "person", "PERSON_LABELS_UNAVAILABLE"),
            self.unknown("background-unknown", "background", "BACKGROUND_LABELS_UNAVAILABLE"),
            self.unknown("robot-unknown", "robot", "ROBOT_VARIATION_UNMEASURED"),
            self.unknown(
                "quality-unknown", "quality",
                "DATA_QUALITY_ANALYSIS_UNAVAILABLE",
            ),
            self.unknown(
                "rollout-unknown", "rollout",
                "NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS",
            ),
            {
                "claim_id": "coverage-observed",
                "class": "OBSERVED",
                "subject": "coverage",
                "value": {
                    "metric": "COLLECTED_EPISODE_COUNT",
                    "count": 2,
                },
                "evidence_refs": [manifest_digest],
                "basis_claim_ids": [],
                "reason_codes": [],
            },
            {
                "claim_id": "collection-suggested",
                "class": "SUGGESTED",
                "subject": "coverage",
                "value": "COLLECT_MORE",
                "evidence_refs": [],
                "basis_claim_ids": ["coverage-observed"],
                "reason_codes": ["COVERAGE_DEFICIT"],
            },
        ]
        self.patches = [
            {
                "change_id": "increase-count",
                "field": "requested_count",
                "value": 3,
                "basis_claim_ids": ["collection-suggested"],
            },
            {
                "change_id": "use-ood-split",
                "field": "split",
                "value": "OOD",
                "basis_claim_ids": ["coverage-observed"],
            },
        ]
        self.data_quality_ref = unavailable("NO_DATA_QUALITY_ANALYSIS")
        self.rollout_ref = unavailable("NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS")

    @staticmethod
    def unknown(claim_id: str, subject: str, reason: str) -> dict:
        return {
            "claim_id": claim_id,
            "class": "UNKNOWN",
            "subject": subject,
            "value": None,
            "evidence_refs": [],
            "basis_claim_ids": [],
            "reason_codes": [reason],
        }

    def episode(self, order_index: int, episode_index: int) -> dict:
        slot = self.manifest["slots"][order_index]
        base_condition = copy.deepcopy(next(
            item for item in self.hypothesis["base_conditions"]
            if item["base_condition_digest"] == slot["base_condition_digest"]
        ))
        run_id = f"campaign-r1-e{order_index + 1}"
        transaction_id = f"{run_id}:episode-{episode_index:06d}"
        rows = 2
        resolved_job_digest = base_condition["resolved_job_digest"]
        collection_profile_digest = base_condition["coverage_condition"][
            "collection_profile_digest"
        ]
        run = {
            "schema_version": "data_factory.recorder_result.v1",
            "run_id": run_id,
            "transaction_id": transaction_id,
            "episode_index": episode_index,
            "state": "COMMITTED",
            "reason_code": "COMMITTED",
            "rows": rows,
            "detail": "",
        }
        staging = {
            "schema_version": "data_factory.staging_manifest.v1",
            "run_id": run_id,
            "dataset_root": self.dataset_root,
            "episode_index": episode_index,
            "staging_mode": "batch",
            "binding_digests": {
                "resolved_job_digest": resolved_job_digest,
                "selected_sheet_digest": digest(["sheet", order_index]),
                "yaw0_sheet_digest": digest(["yaw0-sheet", order_index]),
                "cell_calibration_digest": digest(["cell", order_index]),
                "robot_system_digest": digest(["robot", order_index]),
                "collection_profile_digest": collection_profile_digest,
                "object_profile_digest": digest(["object", order_index]),
                "grasp_profile_digest": digest(["grasp", order_index]),
            },
            "camera_staging_dirs": {
                "up": f"{self.dataset_root}/images/up/episode-{episode_index:06d}",
            },
            "begin_snapshot": {"total_episodes": episode_index},
        }
        episode_ref = {
            "schema_version": "data_factory.episode_ref.v1",
            "repo_id": "local/test-dataset",
            "episode_index": episode_index,
            "transaction_id": transaction_id,
            "resolved_job_digest": resolved_job_digest,
            "staging_manifest_digest": digest(staging),
        }
        episode_ref_digest = digest(episode_ref)
        dataset_digest = digest({
            "repo_id": "local/test-dataset",
            "dataset_root": self.dataset_root,
            "episode_ref": episode_ref,
        })
        locator = redigest({
            "schema_version": "data_factory.lerobot_v3_episode_locator.v1",
            "repo_id": "local/test-dataset",
            "episode_index": episode_index,
            "data": {
                "chunk_index": 0,
                "file_index": episode_index,
                "relative_path": f"data/chunk-000/file-{episode_index:03d}.parquet",
                "file_row_start": 0,
                "file_row_end_exclusive": rows,
            },
            "videos": [{
                "camera_key": "observation.images.up",
                "chunk_index": 0,
                "file_index": episode_index,
                "relative_path": f"videos/observation.images.up/chunk-000/file-{episode_index:03d}.mp4",
                "file_frame_start": 0,
                "file_frame_end_exclusive": rows,
                "timestamp_start_s": 0.0,
                "timestamp_end_s": 0.1,
            }],
        }, "locator_digest")
        provenance = [
            {
                "frame_index": index,
                "target_ros_s": float(episode_index * 10 + index),
            }
            for index in range(rows)
        ]
        quality = {
            "episode_index": episode_index,
            "frames": rows,
            "image_quality_warnings": [],
        }
        scene_state_digest = digest(["scene", order_index])
        intent = redigest({
            "schema_version": "data_factory.campaign_episode_context.v1",
            "run_id": run_id,
            "manifest_id": self.manifest["manifest_id"],
            "manifest_digest": self.manifest["manifest_digest"],
            "order_index": order_index,
            "slot": copy.deepcopy(slot),
            "slot_digest": digest(slot),
            "base_condition": base_condition,
            "robot_start_pose": {
                "robot_start_pose_id": slot["robot_start_pose_id"],
            },
            "fixed_contract": {
                "collection_profile_digest": collection_profile_digest,
            },
            "required_scene_digest": scene_state_digest,
        }, "intent_digest")
        runtime = redigest({
            "schema_version": "data_factory.test_only_episode_binding.v1",
            "session_id": "session-r1",
            "run_id": run_id,
            "resolved_job_digest": resolved_job_digest,
            "manifest_digest": self.manifest["manifest_digest"],
            "intent_digest": intent["intent_digest"],
            "slot_digest": digest(slot),
            "robot_start_pose_id": slot["robot_start_pose_id"],
            "split_group": slot["split_group"],
            "repeat_index": slot["repeat_index"],
            "state_initialization_digest": digest(["initial-scene", order_index]),
            "scene_observation_digest": None,
            "scene_state_digest": scene_state_digest,
            "root_binding_digest": digest(["root", order_index]),
            "start_binding_digest": digest(["start", order_index]),
            "place_alias": "place-a",
            "place_id": "PLACE_A",
            "yaw_deg": 0.0,
            "x_mm": float(order_index),
            "y_mm": 0.0,
            "budget_digests": {
                "manifest_budget_digest": digest(["manifest-budget", order_index]),
                "program_budget_digest": digest(["program-budget", order_index]),
                "planned_usage_digest": digest(["planned-usage", order_index]),
                "slot_budget_digest": digest(["slot-budget", order_index]),
            },
            "expires_at": "2026-09-04T06:00:00Z",
            "data_disposition": "TEST_ONLY",
            "authority": {
                "execution": "NONE",
                "human_approval": "NONE",
                "semantic_pass": "NONE",
                "training_approval": "NONE",
                "persistent_start_qualification": "NONE",
            },
        }, "binding_digest")
        plan = {
            "schema_version": "fr5.pickup_plan.v3",
            "run_id": run_id,
            "resolved_job_digest": resolved_job_digest,
        }
        plan_digest = digest(plan)
        common_safety = {
            "schema_version": "data_factory.precommit_safety.v1",
            "run_id": run_id,
            "approved_plan_digest": plan_digest,
            "scene_binding_digest": digest(["scene-binding", order_index]),
            "expected_planning_scene_digest": digest([
                "planning-scene", order_index,
            ]),
            "planning_scene_readback_digest": digest([
                "scene-readback", order_index,
            ]),
            "collision_report_digest": digest(["collision", order_index]),
            "plan_only_no_motion_digest": digest(["no-motion", order_index]),
        }
        plan_envelope = {
            "plan": plan,
            "precommit_safety": {
                **common_safety,
                "post_reset_safe_snapshot_digest": None,
                "status": "PENDING",
            },
            "precommit_evidence": {
                "schema_version": "data_factory.precommit_evidence.v1",
                "run_id": run_id,
                "approved_plan_digest": plan_digest,
            },
            "operator_summary": {"summary": "fixture"},
        }
        plan_artifact = {
            "schema_version": "data_factory.preapproval_evidence.v1",
            "run_id": run_id,
            "resolved_job_digest": resolved_job_digest,
            "plan_digest": plan_digest,
            "plan_envelope": plan_envelope,
            "plan_envelope_digest": digest(plan_envelope),
        }
        technical = {
            "schema_version": "data_factory.technical_validator_result.v1",
            "run_id": run_id,
            "resolved_job_digest": resolved_job_digest,
            "plan_digest": plan_artifact["plan_digest"],
            "dataset_root": self.dataset_root,
            "expected_fps": 15.0,
            "status": "PASS",
            "result_digest": digest(["technical", order_index]),
        }
        review_context_digest = digest({
            "run_id": run_id,
            "resolved_job_digest": resolved_job_digest,
            "plan_digest": plan_artifact["plan_digest"],
            "technical_validator_digest": digest(technical),
        })
        candidate = {
            "schema_version": "data_factory.candidate_admission.v1",
            "run_id": run_id,
            "operational_gate": "PASS",
            "operational_source": "HIL_PROXY",
            "checklist_id": "pickup-v2",
            "semantic_status": "PASS",
            "reviewed_by": "operator-r1",
            "reviewed_at": "2026-09-04T04:00:00Z",
            "reason": None,
            "review_context_digest": review_context_digest,
        }
        artifacts = {
            "episode": {
                "schema_version": "data_factory.storage_usage.v1",
                "run_id": run_id,
                "episode_ref": episode_ref,
                "dataset_filesystem": {
                    "path": self.dataset_root, "device": 1,
                    "total_bytes": 10_000,
                },
                "encoder_temp_filesystem": {
                    "path": "/tmp", "device": 1, "total_bytes": 10_000,
                },
                "dataset_bytes_before": 100,
                "dataset_bytes_after": 200,
                "dataset_delta_bytes": 100,
                "temporary_peak_bytes_by_filesystem": {"1": 20},
                "free_bytes_before": {"1": 9_000},
                "free_bytes_after": {"1": 8_800},
                "reference_scan_status": "NOT_AVAILABLE",
                "dataset_prunable": [],
            },
            "run": run,
            "staging_manifest": staging,
            "manifest": self.manifest,
            "intent": intent,
            "plan": plan_artifact,
            "technical": technical,
            "source_provenance": provenance,
            "recording_quality": quality,
            "execution": {
                "schema_version": "fr5.pickup_executor.response.v3",
                "mode": "PRE_LIVE",
                "op_id": f"execute-{order_index}",
                "op": "execute",
                "ok": True,
                "code": "COMPLETE",
                "run_id": run_id,
                "plan_digest": plan_artifact["plan_digest"],
                "state": "COMPLETED",
                "data": {
                    "precommit_safety": {
                        **common_safety,
                        "post_reset_safe_snapshot_digest": digest([
                            "safe-reset", order_index,
                        ]),
                        "status": "PASS",
                    },
                },
            },
            "runtime_binding": runtime,
        }
        refs = {
            name: {
                "artifact_path": (
                    f"{self.evidence_root}/{run_id}/episode-{episode_index:06d}.jsonl"
                    if name == "source_provenance" else
                    f"{self.evidence_root}/{run_id}/{name}.json"
                ),
                "artifact_digest": digest(value),
            }
            for name, value in artifacts.items()
        }
        ledger = redigest({
            "schema_version": "data_factory.episode_ledger.v1",
            "dataset": {
                "dataset_id": f"dataset-{dataset_digest[7:23]}",
                "repo_id": "local/test-dataset",
                "dataset_root": self.dataset_root,
                "dataset_digest": dataset_digest,
            },
            "episode": {
                "run_id": run_id,
                "episode_index": episode_index,
                "transaction_id": transaction_id,
                "episode_ref": episode_ref,
                "episode_ref_digest": episode_ref_digest,
                "lerobot_v3_locator": locator,
            },
            "bindings": {
                "resolved_job_digest": episode_ref["resolved_job_digest"],
                "manifest_digest": self.manifest["manifest_digest"],
                "intent_digest": intent["intent_digest"],
                "slot_digest": digest(slot),
                "base_condition_digest": slot["base_condition_digest"],
                "robot_start_pose_id": slot["robot_start_pose_id"],
                "scene_state_digest": runtime["scene_state_digest"],
                "root_binding_digest": runtime["root_binding_digest"],
                "start_binding_digest": runtime["start_binding_digest"],
                "collection_profile_digest": collection_profile_digest,
                "plan_digest": plan_artifact["plan_digest"],
            },
            "artifacts": refs,
            "admission": {
                "technical_status": "PASS",
                "review_context_digest": candidate["review_context_digest"],
                "training_status": "NOT_AUTHORIZED",
            },
        }, "ledger_digest")
        state = redigest({
            "schema_version": "data_factory.episode_ledger_state.v1",
            "ledger_digest": ledger["ledger_digest"],
            "episode_ref_digest": episode_ref_digest,
            "candidate": {
                "artifact_path": f"{self.evidence_root}/{run_id}/candidate_admission.json",
                "artifact_digest": digest(candidate),
            },
            "review": {
                "semantic_status": "PASS",
                "reviewed_by": "operator-r1",
                "reviewed_at": "2026-09-04T04:00:00Z",
                "reason": None,
                "training_status": "NOT_AUTHORIZED",
            },
            "retention": {
                "retention_state": "PRESERVE",
                "reclaim_state": "NOT_EVALUATED",
                "physical_deletion": "NOT_AUTHORIZED",
                "storage_layout": "SHARED_CHUNK",
            },
        }, "state_digest")
        return {
            "manifest_order_index": order_index,
            "ledger": ledger,
            "state": state,
            "candidate": candidate,
            "artifacts": artifacts,
        }

    @staticmethod
    def rebind_episode(evidence: dict) -> None:
        ledger = evidence["ledger"]
        artifacts = evidence["artifacts"]
        for name, payload in artifacts.items():
            ledger["artifacts"][name]["artifact_digest"] = digest(payload)
        technical = artifacts["technical"]
        ref = ledger["episode"]["episode_ref"]
        review_context_digest = digest({
            "run_id": ledger["episode"]["run_id"],
            "resolved_job_digest": ref["resolved_job_digest"],
            "plan_digest": ledger["bindings"]["plan_digest"],
            "technical_validator_digest": digest(technical),
        })
        ledger["admission"]["technical_status"] = technical.get("status")
        ledger["admission"]["review_context_digest"] = review_context_digest
        evidence["candidate"]["review_context_digest"] = review_context_digest
        state = evidence["state"]
        state["candidate"]["artifact_digest"] = digest(evidence["candidate"])
        state["review"] = {
            "semantic_status": evidence["candidate"]["semantic_status"],
            "reviewed_by": evidence["candidate"]["reviewed_by"],
            "reviewed_at": evidence["candidate"]["reviewed_at"],
            "reason": evidence["candidate"]["reason"],
            "training_status": "NOT_AUTHORIZED",
        }
        redigest(ledger, "ledger_digest")
        state["ledger_digest"] = ledger["ledger_digest"]
        state["episode_ref_digest"] = ledger["episode"]["episode_ref_digest"]
        redigest(state, "state_digest")

    def build(self, **changes) -> dict:
        arguments = {
            "recommendation_id": "recommendation-r1",
            "source_commit": COMMIT,
            "campaign_manifest": self.manifest,
            "campaign_hypothesis": self.hypothesis,
            "campaign_draft": self.draft,
            "campaign_compilation_receipt": self.receipt,
            "episode_evidence": self.evidence,
            "data_quality_analysis_ref": self.data_quality_ref,
            "rollout_evidence_analysis_ref": self.rollout_ref,
            "claims": self.claims,
            "suggested_draft_patches": self.patches,
        }
        arguments.update(changes)
        return build_collection_recommendation(**arguments)

    def authoring(self) -> dict:
        return redigest({
            "schema_version": AUTHORING_EVIDENCE_SCHEMA,
            "hypothesis": self.hypothesis, "draft": self.draft,
            "manifest": self.manifest, "compilation_receipt": self.receipt,
        }, "authoring_digest")

    def store(self) -> list[Path]:
        """Write synthetic owner artifacts, including dummy locator targets."""
        roots = []
        for evidence in self.evidence:
            ledger = evidence["ledger"]
            root = Path(self.evidence_root) / ledger["episode"]["run_id"]
            root.mkdir(parents=True)
            roots.append(root)
            for name, value in evidence["artifacts"].items():
                path = Path(ledger["artifacts"][name]["artifact_path"])
                if name in {"source_provenance", "recording_quality"}:
                    rows = value if name == "source_provenance" else [value]
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                else:
                    path.write_text(json.dumps(value))
            for name, value in {
                "episode_ledger.json": ledger, "episode_ledger_state.json": evidence["state"],
                "candidate_admission.json": evidence["candidate"],
                "compiled_authoring_evidence.json": self.authoring(),
            }.items():
                (root / name).write_text(json.dumps(value))
            locator = ledger["episode"]["lerobot_v3_locator"]
            for location in [locator["data"], *locator["videos"]]:
                path = Path(self.dataset_root) / location["relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic locator target; not a real dataset")
        return roots

    @staticmethod
    def view() -> dict:
        projection = {
            "workflow_state": "AUTHORING",
            "available_ops": ["update_draft"],
            "draft": {
                "draft_id": "draft-r1", "revision": 4,
                "authoring_mode": "ASSISTED", "requested_count": 2,
                "repeat": 1,
                "selection": {"task": "pickup_e2e", "split": "TRAIN"},
            },
            "catalog": {
                "axes": {
                    "task": [
                        {"id": "pickup_e2e", "available": True},
                        {"id": "pick_place", "available": False},
                    ],
                    "split": [
                        {"id": "TRAIN", "available": True},
                        {"id": "ID", "available": False},
                        {"id": "OOD", "available": False},
                    ],
                },
            },
            "sampling_provenance": {
                "state_space_design_profile": {"profile_digest": digest("design")},
            },
        }
        bound = {"session_id": "session-r1", "revision": 9, "projection": projection}
        return {
            "schema_version": "data_factory.operator_session_view.v2",
            **bound,
            "generated_at": "2026-09-04T05:00:00Z",
            "view_digest": digest(bound),
            "authority": {
                "browser": "INTENT_ONLY",
                "lifecycle_owner": "BACKEND",
                "human_identity": "NOT_AUTHENTICATED",
                "training_approval": "SEPARATE",
            },
        }


def learned_lifecycle(fixture, order=1, *, completed=False, reviewed=False):
    """CPU-only terminal trace fixture, using native proposal/program contracts.

    This is neither a physical execution observation nor synthetic failure input
    to recollection.py. The diagnostic owner validates this native result shape.
    """
    completed = completed or reviewed
    from tools.data_factory.learned_action_adapter import fake_rgb
    from tools.data_factory.rollout.finite_plan import FinitePolicyInference, JOINTS, compile_program
    from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
    xml = '<robot name="synthetic">' + ''.join(
        f'<joint name="{name}" type="{"prismatic" if i == 6 else "revolute"}">'
        f'<limit lower="{0 if i == 6 else -3}" upper="{.02 if i == 6 else 3}" velocity="10"/></joint>'
        for i, name in enumerate(JOINTS)) + '</robot>'
    checkpoint = {"tree_digest": digest("synthetic-weights"),
                  "training_receipt_digest": digest("synthetic-receipt"), "runtime": "SYNTHETIC_TEST_ONLY"}
    action = [.001] * 6 + [.0101]
    proposal = FinitePolicyInference(
        lambda _: [action], checkpoint, source_clock=lambda: 10., monotonic_clock=lambda: 1.,
    ).propose({
        "source_clock": "SYSTEM_TIME", "source_timestamps_s": dict.fromkeys(("state", "camera1", "camera2"), 10.),
        "observation.state": [0.] * 6 + [.01],
        "observation.images.camera1": fake_rgb(), "observation.images.camera2": fake_rgb(),
    }, instruction="synthetic pickup", robot_description=xml, period_s=.1)
    source = motion()
    source["resolved_job_digest"] = fixture.hypothesis["base_conditions"][order]["resolved_job_digest"]
    receipt = next(r for r in fixture.hypothesis["resolver_receipts"]
                   if r["resolved_job_digest"] == source["resolved_job_digest"])
    source["robot_system_id"] = receipt["normalized_job"]["robot_system_id"]
    source["binding_digests"].update(receipt["input_digests"])
    source["binding_digests"]["robot_description_digest"] = "sha256:" + hashlib.sha256(xml.encode()).hexdigest()
    program = compile_program(source, proposal)
    plan = {
        "schema_version": "fr5.pickup_plan.v3", "run_id": "synthetic-learned-run",
        "motion_program_digest": digest(program), "resolved_job_digest": source["resolved_job_digest"],
        "robot_system_id": source["robot_system_id"], "binding_digests": source["binding_digests"],
        "learned_source_program": source, "learned_proposal": proposal, "steps": program["steps"],
    }
    trace = redigest({
        "schema_version": "data_factory.finite_learned_execution.v1",
        "proposal_digest": proposal["proposal_digest"], "plan_digest": digest(plan), "checkpoint": checkpoint,
        "status": "COMPLETED" if completed else "FAILED",
        "failure_code": None if completed else "CONTROLLER_FAULT",
        "terminal_state": action if completed else None,
        "terminal_phases": ["LEARNED_CHUNK"] if completed else [],
        "task_effectiveness": "UNKNOWN", "scene_outcome": "UNKNOWN",
        "cell_ready": False, "online_policy_authorized": False,
    }, "trace_digest")
    result = {
        "ok": False, "code": "PRECOMMIT_SAFETY" if completed else "CONTROLLER_FAULT", "state": "ABORTED",
        "run_id": plan["run_id"], "plan_digest": digest(plan), "plan_envelope": {"plan": plan},
        "recorder_state": "ABORTED", "recorder_evidence": {"state": "ABORTED"},
        "execution_evidence": {"learned_execution": trace}, "semantic_verdict": "PENDING",
    }
    if reviewed:
        # Shape recorded by the native semantic_verdict API after chunk completion;
        # this explicit fixture is not a real human decision or task-success label.
        result.update(code="SEMANTIC_FAIL", semantic_verdict="FAIL")
        result["execution_evidence"].update(semantic_verdict="FAIL", semantic_decision={
            "source": "HUMAN", "decided_by": "synthetic-reviewer", "decided_at": "2026-09-04T05:00:00Z",
            "review_scope": "FINITE_LEARNED_CHUNK",
        })
    build_run_diagnostic(result)
    return result


def scoped_terminal_lifecycle(result):
    """Project the canonical contract shape; this is no physical qualification."""
    from tools.data_factory.motion.mechanical_terminal import PHASES
    from tools.data_factory.rollout.task_authority import SCOPE, task_scope

    result = copy.deepcopy(result)
    plan = result["plan_envelope"]["plan"]
    grant = {
        "schema_version": "data_factory.learned_task_grant.v1",
        "grant_id": "synthetic-task", "issued_by": "synthetic-authority",
        "run_id": result["run_id"],
        "scope": task_scope(plan["learned_source_program"], plan["scene_binding"], plan["learned_proposal"]),
        "deadline_s": 100., "terminal_reserve_s": 30., "max_outputs": 1,
        "revoked": False,
    }
    redigest(grant, "grant_digest")
    admission = {
        "approval_id": "task-" + result["plan_digest"].removeprefix("sha256:"),
        "approved_by": grant["issued_by"], "approval_scope": SCOPE,
        "approval_expiry": None, "run_id": result["run_id"],
        "resolved_job_digest": plan["resolved_job_digest"],
        "plan_digest": result["plan_digest"], "task_grant": copy.deepcopy(grant),
    }
    terminal_plan = redigest({
        "parent_plan_digest": digest(plan),
        "source_program_digest": digest(plan["learned_source_program"]),
        "grant_digest": grant["grant_digest"], "deadline_s": grant["deadline_s"],
        "control_source": "QUALIFIED_MECHANICAL_RELEASE",
        "recording_scope": "OUT_OF_DATASET", "semantic_success": "NOT_MEASURED",
        "steps": [{"phase": phase} for phase in PHASES],
    }, "terminal_digest")
    result.update(code="MECHANICAL_TERMINAL_COMPLETE", semantic_verdict="PENDING")
    result["execution_evidence"].pop("semantic_verdict", None)
    result["execution_evidence"].pop("semantic_decision", None)
    result["execution_evidence"].update(
        task_authority={"grant": grant, "admission": admission},
        mechanical_terminal={"plan": terminal_plan, "status": "COMPLETED", "terminal_phases": list(PHASES)},
        task_handoff={
            "status": "PLANNED", "termination_reason": "TASK_OUTPUT_LIMIT_REACHED",
            "terminal_digest": terminal_plan["terminal_digest"], "deadline_s": grant["deadline_s"],
            "semantic_success": "NOT_MEASURED",
        },
    )
    from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
    build_run_diagnostic(result)
    return result


class MechanicalDiagnosticConsumerTests(unittest.TestCase):
    def lifecycle(self, kind):
        from tools.data_factory.motion.mechanical_terminal import PHASES
        result = learned_lifecycle(RecommendationFixture(), order=0, reviewed=True)
        if kind == "contact":
            result["execution_evidence"]["mechanical_contact_diagnostic"] = {
                "status": "BLOCKED_UNAVAILABLE", "code": "MECHANICAL_CONTACT_UNAVAILABLE",
            }
        elif kind == "terminal":
            parent = result["plan_envelope"]["plan"]
            # Minimal validator-accepted serialization, not physical qualification.
            terminal = redigest({
                "parent_plan_digest": digest(parent),
                "source_program_digest": digest(parent["learned_source_program"]),
                "control_source": "QUALIFIED_MECHANICAL_RELEASE",
                "recording_scope": "OUT_OF_DATASET", "semantic_success": "NOT_MEASURED",
                "steps": [{"phase": phase} for phase in PHASES],
            }, "terminal_digest")
            result["execution_evidence"]["mechanical_terminal"] = {
                "plan": terminal, "status": "COMPLETED", "terminal_phases": list(PHASES),
            }
        elif kind == "handoff":
            result["execution_evidence"]["task_handoff"] = {"status": "BLOCKED_UNAVAILABLE"}
        elif kind == "legacy_handoff":
            result["task_handoff"] = {"status": "BLOCKED_UNAVAILABLE"}
        elif kind == "legacy_null_handoff":
            result["task_handoff"] = None
        return result

    def test_retained_mechanical_diagnostic_reaches_operator_and_collection(self):
        from tools.data_factory.collection_recommendation import _analysis_ref
        from tools.data_factory.operator.workflow.learned_run import LearnedRunApplication
        from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
        for kind in ("baseline", "contact", "terminal", "handoff", "legacy_handoff", "legacy_null_handoff"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lifecycle = self.lifecycle(kind)
                before = copy.deepcopy(lifecycle)
                request = payload("live")
                request.update(run_id=lifecycle["run_id"], run_root=str(root),
                    dataset_root=str(root / "unused-dataset"),
                    learned_checkpoint=str(root / "unused-checkpoint"),
                    gripper_source_clock="SYSTEM_TIME")
                request["job"]["operator_or_agent_id"] = "fixture-operator"
                (root / lifecycle["run_id"]).mkdir()
                def producer(value, *_args, **_kwargs):
                    return run_job._response(ok=False, code=lifecycle["code"],
                        state=lifecycle["state"], run_id=lifecycle["run_id"],
                        plan_digest=lifecycle["plan_digest"],
                        data=run_job.learned_run_diagnostic(lifecycle, payload=value))
                app = LearnedRunApplication(payload=request, operator_label="fixture-operator",
                                            run_live_call=producer)
                app._run()
                self.assertIsNone(app.error)
                self.assertEqual(app.lifecycle_result, lifecycle)
                self.assertEqual(lifecycle, before)
                saved_path = root / lifecycle["run_id"] / "learned_lifecycle_result.json"
                saved_bytes = saved_path.read_bytes()
                self.assertEqual(json.loads(saved_bytes), lifecycle)
                diagnostic = app.result["data"]
                self.assertEqual(diagnostic, build_run_diagnostic(lifecycle))
                self.assertEqual(diagnostic["diagnostic_digest"], digest({
                    k: v for k, v in diagnostic.items() if k != "diagnostic_digest"}))
                for field in ("mechanical_terminal", "mechanical_contact_diagnostic"):
                    self.assertEqual(diagnostic.get(field), lifecycle["execution_evidence"].get(field))
                self.assertEqual(diagnostic.get("task_handoff"),
                    lifecycle["execution_evidence"].get("task_handoff", lifecycle.get("task_handoff")))
                if kind == "legacy_null_handoff":
                    self.assertIn("task_handoff", diagnostic)
                ref = {"availability": "AVAILABLE", "schema_version": diagnostic["schema_version"],
                       "analysis_id": diagnostic["run_id"], "analysis_digest": digest(diagnostic),
                       "reason_codes": []}
                self.assertEqual(_analysis_ref(ref, owner="rollout", artifact=lifecycle,
                                               normalize=False), ref)
                changed = copy.deepcopy(lifecycle)
                changed["code"] = "CHANGED"
                with self.assertRaisesRegex(ContractError, "COLLECTION_RECOMMENDATION_ANALYSIS_DIGEST"):
                    _analysis_ref(ref, owner="rollout", artifact=changed, normalize=False)
                # Strict operator equality still rejects a changed retained source.
                saved_path.write_text(json.dumps(changed))
                response = copy.deepcopy(app.result)
                tampered = LearnedRunApplication(payload=request, operator_label="fixture-operator",
                    run_live_call=lambda *_args, **_kwargs: response)
                tampered._run()
                self.assertEqual(tampered.error, "LEARNED_WEB_RESULT_CHANGED")
                self.assertIsNone(tampered.lifecycle_result)
                self.assertEqual(diagnostic["task_effectiveness"], "UNKNOWN")
                self.assertEqual(diagnostic["physical_qualification"], "UNKNOWN")
                self.assertFalse(diagnostic["training_authorized"])
                self.assertFalse((root / "unused-dataset").exists())

    def test_malformed_terminal_rejected_by_canonical_builder_before_persistence(self):
        from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
        for kind in ("missing_plan", "changed_parent", "changed_digest"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                lifecycle = self.lifecycle("terminal")
                terminal = lifecycle["execution_evidence"]["mechanical_terminal"]
                if kind == "missing_plan":
                    terminal.pop("plan")
                elif kind == "changed_parent":
                    terminal["plan"]["parent_plan_digest"] = digest("foreign")
                    redigest(terminal["plan"], "terminal_digest")
                else:
                    terminal["plan"]["terminal_digest"] = digest("changed")
                with self.assertRaisesRegex(ContractError, "MECHANICAL_TERMINAL_BINDING"):
                    build_run_diagnostic(lifecycle)
                with self.assertRaisesRegex(ContractError, "MECHANICAL_TERMINAL_BINDING"):
                    run_job.learned_run_diagnostic(lifecycle,
                        payload={"run_id": lifecycle["run_id"], "run_root": directory})
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_conflicting_nested_and_legacy_handoff_is_rejected(self):
        from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
        lifecycle = self.lifecycle("handoff")
        lifecycle["task_handoff"] = {"status": "DIFFERENT"}
        with self.assertRaisesRegex(ContractError, "ROLLOUT_RUN_DIAGNOSTIC_HANDOFF"):
            build_run_diagnostic(lifecycle)


class RolloutRecommendationTests(unittest.TestCase):
    def test_terminal_diagnostic_reaches_native_draft_and_compile(self):
        for reviewed in (False, True):
            with self.subTest(reviewed=reviewed):
                self._native_draft(reviewed)

    def _native_draft(self, reviewed):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(dataset_root=str(root / "data"), evidence_root=str(root / "runs"))
            runs = fixture.store()
            order = 0 if reviewed else 1
            lifecycle = learned_lifecycle(fixture, order=order, reviewed=reviewed)
            path = root / "learned" / "learned_lifecycle_result.json"
            path.parent.mkdir()
            path.write_text(json.dumps(lifecycle))
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            arguments = dict(run_directories=runs[:1], source_commit=COMMIT,
                             rollout_lifecycle_path=path, output_root=root / "advice")
            result = recommend_stored_collection(**arguments)
            self.assertEqual(result["availability"], "AVAILABLE", result)
            self.assertEqual(result, recommend_stored_collection(**arguments))
            diagnostic = result["rollout_evidence_analysis"]
            self.assertEqual(diagnostic["checkpoint"], lifecycle["execution_evidence"]["learned_execution"]["checkpoint"])
            self.assertEqual(diagnostic["lifecycle_result_digest"], digest(lifecycle))
            self.assertEqual(diagnostic["task_effectiveness"], "UNKNOWN")
            self.assertEqual(diagnostic["physical_qualification"], "UNKNOWN")
            self.assertFalse(diagnostic["training_authorized"])
            advice = result["recommendation"]
            self.assertEqual(advice["authority"], AUTHORITY)
            suggestion = next(c for c in advice["claims"] if c["class"] == "SUGGESTED")
            self.assertEqual(suggestion["value"], "REDEMONSTRATE_CONDITION" if reviewed else "COLLECT_MORE")
            self.assertEqual(next(c for c in advice["claims"] if c["subject"] == "rollout")["reason_codes"],
                             ["ROLLOUT_DATA_DEFICIT_UNPROVEN"])
            factory = mock.Mock(side_effect=AssertionError("no collection effects"))
            application = CampaignOperator(
                session_id="rollout-recommendation", lifecycle_owner="TEST_OPERATOR",
                workspace={"identity": "SYNTHETIC"}, hypothesis=fixture.hypothesis, draft=fixture.draft,
                effect_scope="FAKE", lifecycle_action="AUTHOR_ONLY", data_disposition="TEST_ONLY",
                subsystems={"planner": {"readiness": "READY", "capability": "AUTHOR", "reason": "SYNTHETIC"}},
                expires_at="2099-01-01T00:00:00Z", initial_scene_digest=digest("scene"),
                scene_evidence_call=factory, side_effect_counter_call=lambda: dict.fromkeys(SIDE_EFFECT_COUNTERS, 0),
                fake_lifecycle_factory=factory, clock=lambda: NOW,
            )
            intent_args = dict(compiled_authoring=fixture.authoring(), operator_view=application.core.snapshot(),
                               data_quality_analysis=result["data_quality_analysis"])
            with self.assertRaisesRegex(ContractError, "ROLLOUT_LIFECYCLE_REQUIRED"):
                project_campaign_update_intent(advice, **intent_args)
            intent = project_campaign_update_intent(advice, **intent_args, rollout_lifecycle_result=lifecycle)
            changed = copy.deepcopy(lifecycle)
            changed["code"] = "OTHER_TERMINAL_REASON"
            with self.assertRaisesRegex(ContractError, "ANALYSIS_DIGEST"):
                project_campaign_update_intent(advice, **intent_args, rollout_lifecycle_result=changed)
            application.core.consume(intent)
            with self.assertRaisesRegex(ContractError, "STALE_VIEW"):
                application.core.consume({**intent, "intent_id": "replayed-rollout-advice"})
            view = application.core.snapshot()
            application.core.consume({**intent, "intent_id": "compile-rollout-target", "op": "compile_draft",
                                      "view_revision": view["revision"], "view_digest": view["view_digest"], "payload": {}})
            self.assertEqual([s["base_condition_digest"] for s in application.manifest["slots"]],
                             [fixture.hypothesis["base_conditions"][order]["base_condition_digest"]])
            factory.assert_not_called()
            self.assertFalse(any(application.projection()["side_effect_counters"].values()))
            self.assertTrue(all(p.read_bytes() == original for p, original in before.items()))
            self.assertEqual(set(p.name for p in Path(result["output_path"]).iterdir()),
                             {"coverage_report.json", "collection_recommendation.json", "rollout_diagnostic.json"})
            with mock.patch("sys.stdout") as stdout:
                self.assertEqual(recommend_main([
                    "--run-dir", str(runs[0]), "--source-commit", COMMIT,
                    "--rollout-lifecycle", str(path), "--expected-recommendation-digest", advice["recommendation_digest"],
                ]), 0)
            emitted = json.loads("".join(c.args[0] for c in stdout.write.call_args_list))
            self.assertEqual(emitted["recommendation"], advice)

    def test_controller_failure_or_completion_does_not_imply_data_deficit(self):
        fixture = RecommendationFixture()
        for completed in (False, True):
            with self.subTest(completed=completed):
                lifecycle = learned_lifecycle(fixture, order=0, completed=completed)
                _, advice = derive_collection_recommendation(
                    compiled_authoring=fixture.authoring(), episode_evidence=fixture.evidence,
                    source_commit=COMMIT, rollout_lifecycle_result=lifecycle,
                )
                self.assertEqual(advice["suggested_draft_patches"], [])
                self.assertFalse(any(c["class"] == "SUGGESTED" for c in advice["claims"]))
                self.assertEqual(next(c for c in advice["claims"] if c["subject"] == "rollout")["class"], "UNKNOWN")

    def test_proxy_absent_or_different_scope_review_cannot_author_redemonstration(self):
        fixture = RecommendationFixture()
        lifecycle = learned_lifecycle(fixture, order=0, reviewed=True)
        args = dict(compiled_authoring=fixture.authoring(), episode_evidence=fixture.evidence, source_commit=COMMIT)
        mutations = [
            lambda r: r["execution_evidence"]["semantic_decision"].update(source="HIL_PROXY"),
            lambda r: r["execution_evidence"].pop("semantic_decision"),
            lambda r: r["execution_evidence"]["semantic_decision"].update(review_scope="FULL_TASK"),
            lambda r: r["execution_evidence"].update(semantic_verdict="PASS"),
            lambda r: r.update(semantic_verdict="PASS"),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(lifecycle)
            mutate(changed)
            _, advice = derive_collection_recommendation(**args, rollout_lifecycle_result=changed)
            self.assertEqual(advice["suggested_draft_patches"], [])
        # Rewording a canonical coverage claim cannot invent a human review.
        unreviewed = learned_lifecycle(fixture)
        report, advice = derive_collection_recommendation(**args, rollout_lifecycle_result=unreviewed)
        claim = next(c for c in advice["claims"] if c["class"] == "SUGGESTED")
        claim.update(value="REDEMONSTRATE_CONDITION", reason_codes=["HUMAN_REVIEWED_CHUNK_FAILURE"],
                     evidence_refs=sorted([digest(report), advice["input_snapshot"]["rollout_evidence_analysis_ref"]["analysis_digest"]]))
        redigest(advice, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "ROLLOUT_REVIEW_REQUIRED"):
            validate_collection_recommendation(advice, campaign_manifest=fixture.manifest,
                campaign_hypothesis=fixture.hypothesis, campaign_draft=fixture.draft,
                campaign_compilation_receipt=fixture.receipt, episode_evidence=fixture.evidence,
                data_quality_analysis=report, rollout_evidence_analysis=unreviewed)

    def test_tampering_missing_condition_and_stale_source_do_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(dataset_root=str(root / "data"), evidence_root=str(root / "runs"))
            runs = fixture.store()
            original = learned_lifecycle(fixture)
            path = root / "lifecycle.json"
            def invoke(value, **extra):
                path.write_text(json.dumps(value))
                return recommend_stored_collection(run_directories=runs[:1], source_commit=COMMIT,
                    rollout_lifecycle_path=path, output_root=root / "advice", **extra)
            changes = [
                lambda r: r.update(run_id="different-run"),
                lambda r: r["execution_evidence"]["learned_execution"]["checkpoint"].update(tree_digest=digest("other")),
                lambda r: r["recorder_evidence"].update(state="FROZEN"),
                lambda r: r["plan_envelope"]["plan"].update(resolved_job_digest=digest("other-condition")),
                lambda r: r["execution_evidence"]["learned_execution"].update(task_effectiveness="PASS"),
            ]
            for mutate in changes:
                value = copy.deepcopy(original)
                mutate(value)
                self.assertEqual(invoke(value)["availability"], "UNAVAILABLE")
                self.assertFalse((root / "advice").exists())
            # Even a self-consistent trace cannot invent a qualified condition.
            value = copy.deepcopy(original)
            value["plan_envelope"]["plan"]["resolved_job_digest"] = digest("unknown-condition")
            value["plan_digest"] = digest(value["plan_envelope"]["plan"])
            value["execution_evidence"]["learned_execution"]["plan_digest"] = value["plan_digest"]
            redigest(value["execution_evidence"]["learned_execution"], "trace_digest")
            self.assertEqual(invoke(value)["reason_codes"], ["COLLECTION_RECOMMENDATION_ROLLOUT_CONDITION_UNAVAILABLE"])
            self.assertFalse((root / "advice").exists())
            from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
            self.assertEqual(invoke(build_run_diagnostic(original))["availability"], "UNAVAILABLE")
            self.assertFalse((root / "advice").exists())
            self.assertEqual(invoke(original, expected_recommendation_digest=digest("old-advice"))["availability"], "UNAVAILABLE")
            self.assertFalse((root / "advice").exists())

    def test_changed_native_sources_are_rechecked_before_publication(self):
        from tools.data_factory import collection_recommendation_io as io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(dataset_root=str(root / "data"), evidence_root=str(root / "runs"))
            runs = fixture.store()
            lifecycle = learned_lifecycle(fixture)
            path = root / "learned" / "learned_lifecycle_result.json"
            path.parent.mkdir()
            path.write_text(json.dumps(lifecycle))
            source_before = {p: p.read_bytes() for p in (root / "data").rglob("*") if p.is_file()}
            derive = io.derive_collection_recommendation
            def replace_after_read(**kwargs):
                result = derive(**kwargs)
                path.write_text(json.dumps({**lifecycle, "code": "REPLACED"}))
                return result
            with mock.patch.object(io, "derive_collection_recommendation", side_effect=replace_after_read):
                result = io.recommend_stored_collection(run_directories=runs[:1], source_commit=COMMIT,
                    rollout_lifecycle_path=path, output_root=root / "advice")
            self.assertEqual(result["reason_codes"], ["COLLECTION_RECOMMENDATION_ROLLOUT_INPUT_CHANGED"])
            self.assertFalse((root / "advice").exists())
            self.assertTrue(all(p.read_bytes() == value for p, value in source_before.items()))

    def test_qualified_condition_context_mismatch_and_acquisition_mode_reject(self):
        fixture = RecommendationFixture()
        lifecycle = learned_lifecycle(fixture)
        args = dict(compiled_authoring=fixture.authoring(), episode_evidence=fixture.evidence[:1], source_commit=COMMIT)
        with self.assertRaisesRegex(ContractError, "ACQUISITION_INPUT_FIELDS"):
            derive_collection_recommendation(**args, acquisition={}, rollout_lifecycle_result=lifecycle)
        for key in ("selected_sheet", "collection_profile", "cell_calibration", "object_profile"):
            with self.subTest(key=key):
                changed = copy.deepcopy(lifecycle)
                plan = changed["plan_envelope"]["plan"]
                plan["learned_source_program"]["binding_digests"][key] = digest("other-context")
                plan["binding_digests"][key] = digest("other-context")
                changed["plan_digest"] = digest(plan)
                changed["execution_evidence"]["learned_execution"]["plan_digest"] = digest(plan)
                redigest(changed["execution_evidence"]["learned_execution"], "trace_digest")
                with self.assertRaisesRegex(ContractError, "ROLLOUT_CONDITION_BINDING"):
                    derive_collection_recommendation(**args, rollout_lifecycle_result=changed)


class AcquisitionRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_operator_catalog(ROOT, device_ids=["usb-Generic_USB2.0_PC_CAMERA-video-index0"])

    def setUp(self):
        from tools.a4_place_yaw.region_layout import a4_printable_polygon
        from tools.data_factory.workspace_geometry import polygon_bounds, rotation_envelope
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = RecommendationFixture(dataset_root=str(self.root / "data"), evidence_root=str(self.root / "runs"))
        # Distinct canonical campaigns, no retained or reconstructed authoring.
        second = copy.deepcopy(self.fixture)
        second.draft["manifest_id"] = "second-production-shaped-campaign"
        second.manifest, second.receipt = compile_collection_campaign(second.draft, hypothesis=second.hypothesis)
        self.fixture.evidence[1] = second.episode(1, 11)
        self.runs = self.fixture.store()
        for run in self.runs:
            (run / "compiled_authoring_evidence.json").unlink()
        condition = self.fixture.evidence[0]["artifacts"]["intent"]["base_condition"]["coverage_condition"]
        self.lifecycle = learned_lifecycle(self.fixture, order=0, reviewed=True)
        plan = self.lifecycle["plan_envelope"]["plan"]
        job = copy.deepcopy(next(r["normalized_job"] for r in self.fixture.hypothesis["resolver_receipts"]
                                 if r["resolved_job_digest"] == plan["resolved_job_digest"]))
        catalog = copy.deepcopy(self.catalog)
        combo = copy.deepcopy(next(c for c in catalog["combinations"]
                                   if c["task_id"] == "pickup_e2e" and c["variant_id"] == "DIRECT"
                                   and "yaw_sampling_profile" not in c))
        domain = copy.deepcopy(next(d for d in catalog["workspace_domains"] if
            (d["workspace_id"], d["frame_id"], d["object_id"]) ==
            (combo["workspace_id"], combo["frame_id"], combo["object_id"])))
        # Synthetic identities in a complete native catalog/domain shape. Native
        # geometry, selection, ledger and diagnostic validators are not mocked.
        combo.update(workspace_id=condition["place_id"], frame_id=condition["cell_calibration_id"],
                     object_id=condition["object_profile_id"], grasp_id=condition["grasp_profile_id"],
                     camera_profile_id=job["collection_profile_id"])
        domain.update(workspace_id=combo["workspace_id"], frame_id=combo["frame_id"], object_id=combo["object_id"])
        region = domain["coverage_region"]
        region.update(physical_binding_status="NOT_CONFIGURED", layout_id=None, layout_digest=None,
                      region_id=None, polygon_local_xy_mm=a4_printable_polygon())
        x, y = rotation_envelope(*polygon_bounds(region["polygon_local_xy_mm"]))
        domain.update(x_mm={"minimum": x[0], "maximum": x[1]}, y_mm={"minimum": y[0], "maximum": y[1]})
        redigest(domain, "domain_digest")
        staging = self.fixture.evidence[0]["artifacts"]["staging_manifest"]["binding_digests"]
        combo["source_digests"].update(cell=condition["cell_calibration_digest"],
            camera_profile=condition["collection_profile_digest"], object=staging["object_profile_digest"],
            grasp=staging["grasp_profile_digest"], motion=plan["binding_digests"]["motion_qualification"])
        redigest(combo, "combination_digest")
        catalog.update(combinations=[combo], workspace_domains=[domain])
        for cell in catalog["axes"]["cell"]:
            cell["metadata"]["place_id"] = combo["workspace_id"]
        redigest(catalog, "catalog_digest")
        self.selection = {"schema_version": "data_factory.operator_selection.v1", "data_mode": "GENERAL_COLLECTION",
            "policy_id": "DETERMINISTIC_SPREAD", **{k: combo[k] for k in (
                "combination_digest", "workspace_id", "frame_id", "task_id", "object_id", "grasp_id", "cell_id",
                "start_pose_id", "motion_id", "variant_id", "camera_profile_id", "camera_device_id")}}
        pose = {k: condition[k] for k in ("place_id", "yaw_deg", "x_mm", "y_mm")}
        self.scene = {"schema_version": "data_factory.scene_state.v2", "robot_system_id": condition["robot_system_id"],
            "revision": 3, "updated_at": "2026-09-08T00:00:00Z", "slot_allocations": {}, "objects": {"cube": {
                "instance_id": "cube", "object_profile_id": condition["object_profile_id"], "state": "ON_SURFACE",
                "pose": pose, "source": "HUMAN", "updated_by": "synthetic-fixture", "updated_at": "2026-09-08T00:00:00Z"}}}
        binding = {"scene_state_digest": digest(self.scene), "revision": 3, "object_instance_id": "cube"}
        self.lifecycle["scene_binding"] = binding
        plan["scene_binding"] = copy.deepcopy(binding)
        for key, value in {"object_profile": staging["object_profile_digest"], "grasp_profile": staging["grasp_profile_digest"]}.items():
            plan["binding_digests"][key] = value
            plan["learned_source_program"]["binding_digests"][key] = value
        from tools.data_factory.quality.coverage_report import RESOLVED_INPUT_DIGEST_FIELDS
        inputs = {key: plan["binding_digests"][key] for key in RESOLVED_INPUT_DIGEST_FIELDS}
        resolved = digest({"job": job, "input_digests": inputs})
        plan["resolved_job_digest"] = plan["learned_source_program"]["resolved_job_digest"] = resolved
        self.rebind(self.lifecycle)
        self.preapproval = {"schema_version": "data_factory.preapproval_evidence.v4",
            "run_id": self.lifecycle["run_id"], "plan_digest": self.lifecycle["plan_digest"],
            "resolved_job_digest": resolved, "plan_envelope": copy.deepcopy(self.lifecycle["plan_envelope"]),
            "plan_envelope_digest": digest(self.lifecycle["plan_envelope"]),
            "resolved_inputs": {"normalized_job": job, "input_digests": inputs, "resolved_job_digest": resolved}}
        self.preapproval_path = self.root / "preapproval_evidence.json"
        self.preapproval_path.write_text(json.dumps(self.preapproval))
        # Native learned completion makes the scene UNKNOWN. Later recovery is
        # a new canonical revision; it must not overwrite the original binding.
        unknown = copy.deepcopy(self.scene)
        unknown["revision"] = 4
        unknown["objects"]["cube"].update(state="UNKNOWN", pose=None, source="ROBOT_ACTION")
        self.lifecycle["execution_evidence"]["scene_transition"] = {"scene_state": unknown, "scene_state_digest": digest(unknown)}
        self.scene["revision"] = 5
        self.scene_path = self.root / "scene.json"
        self.scene_path.write_text(json.dumps(self.scene))
        self.lifecycle_path = self.root / "learned_lifecycle_result.json"
        self.lifecycle_path.write_text(json.dumps(self.lifecycle))
        self.acquisition = {"catalog": catalog, "selection": self.selection, "scene_state_path": str(self.scene_path),
            "expected_scene_digest": digest(self.scene), "object_instance_id": "cube",
            "requested_count": 4, "normalized_seed": 87, "repeat": 1}

    @staticmethod
    def rebind(lifecycle):
        lifecycle["plan_digest"] = digest(lifecycle["plan_envelope"]["plan"])
        lifecycle["execution_evidence"]["learned_execution"]["plan_digest"] = lifecycle["plan_digest"]
        redigest(lifecycle["execution_evidence"]["learned_execution"], "trace_digest")

    def call(self, **changes):
        return recommend_stored_collection(**{ "run_directories": self.runs, "source_commit": COMMIT,
            "acquisition": self.acquisition, "rollout_lifecycle_path": self.lifecycle_path, **changes})

    def use_registered_motion_preset(self):
        # Reuse the actual catalog registration/digests, not a guessed policy or
        # a newly qualified profile. Only synthetic run fixtures are changed.
        catalog = self.acquisition["catalog"]
        preset = next(p for p in catalog["motion_presets"] if p["qualifications"])
        motion_id = next(iter(preset["qualifications"]))
        combo = catalog["combinations"][0]
        combo["motion_id"] = motion_id
        redigest(combo, "combination_digest")
        redigest(catalog, "catalog_digest")
        self.selection.update(motion_id=motion_id, combination_digest=combo["combination_digest"])
        self.acquisition["motion_preset"] = {key: preset[key] for key in ("id", "digest")}
        plan = self.lifecycle["plan_envelope"]["plan"]
        for bindings in (plan["binding_digests"], plan["learned_source_program"]["binding_digests"]):
            bindings.update(motion_preset=preset["digest"],
                motion_qualification=preset["qualifications"][motion_id]["digest"])
        self.rebind(self.lifecycle)
        self.preapproval.update(plan_envelope=copy.deepcopy(self.lifecycle["plan_envelope"]),
            plan_digest=self.lifecycle["plan_digest"], plan_envelope_digest=digest(self.lifecycle["plan_envelope"]))
        self.preapproval_path.write_text(json.dumps(self.preapproval))
        self.lifecycle_path.write_text(json.dumps(self.lifecycle))
        return preset

    def test_registered_preset_uses_qualified_digest_and_revalidates_exact_identity(self):
        from tools.data_factory.collection_recommendation_io import _acquisition_context, _load_run
        preset = self.use_registered_motion_preset()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.call()
        self.assertEqual(result["availability"], "AVAILABLE", result)
        advice = result["recommendation"]
        binding = advice["input_snapshot"]["motion_preset"]
        self.assertEqual(binding, {**self.acquisition["motion_preset"], "qualifications": {
            self.selection["motion_id"]: preset["qualifications"][self.selection["motion_id"]]["digest"]}})
        self.assertNotEqual(binding["qualifications"][self.selection["motion_id"]],
            self.acquisition["catalog"]["combinations"][0]["source_digests"]["motion"])
        args = dict(acquisition=_acquisition_context(self.acquisition),
            episode_evidence=[_load_run(r)[0] for r in self.runs],
            rollout_evidence_analysis=self.lifecycle, rollout_preapproval_evidence=self.preapproval)
        self.assertEqual(validate_collection_recommendation(advice, **args), advice)
        self.assertEqual(self.call(), result)
        changed = copy.deepcopy(advice)
        changed["input_snapshot"]["motion_preset"]["qualifications"][self.selection["motion_id"]] = digest("tampered")
        redigest(changed["input_snapshot"], "snapshot_digest")
        redigest(changed, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "INPUT_CHANGED"):
            validate_collection_recommendation(changed, **args)
        self.assertEqual(self.call(acquisition={k: v for k, v in self.acquisition.items() if k != "motion_preset"})[
            "reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_CONTEXT_MISMATCH"])
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_changed_unregistered_trial_only_and_mismatched_preset_fail_without_writes(self):
        preset = self.use_registered_motion_preset()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        trial = next(p for p in self.acquisition["catalog"]["motion_presets"] if not p["qualifications"])
        for binding, code in (
            ({"id": preset["id"], "digest": digest("changed")}, "MOTION_PRESET_BINDING"),
            ({"id": "missing-preset", "digest": preset["digest"]}, "MOTION_PRESET_BINDING"),
            ({key: trial[key] for key in ("id", "digest")}, "MOTION_PRESET_QUALIFICATION_REQUIRED"),
        ):
            with self.subTest(binding=binding):
                result = self.call(acquisition={**self.acquisition, "motion_preset": binding})
                self.assertEqual(result["availability"], "UNAVAILABLE", result)
                self.assertEqual(result["reason_codes"], [code])
                self.assertIsNone(result["output_path"])
        changed = copy.deepcopy(self.acquisition)
        p = next(p for p in changed["catalog"]["motion_presets"] if p["id"] == preset["id"])
        p["qualifications"][self.selection["motion_id"]]["digest"] = digest("different-qualified-motion")
        redigest(changed["catalog"], "catalog_digest")
        self.assertEqual(self.call(acquisition=changed)["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_CONTEXT_MISMATCH"])
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_omitted_and_null_preset_keep_legacy_advice_identical(self):
        from tools.data_factory.collection_recommendation_io import _acquisition_context, _load_run
        for lifecycle in (self.lifecycle_path, None):
            omitted = self.call(rollout_lifecycle_path=lifecycle)
            explicit = self.call(rollout_lifecycle_path=lifecycle, acquisition={**self.acquisition, "motion_preset": None})
            self.assertEqual(omitted["availability"], "AVAILABLE", omitted)
            self.assertEqual(explicit, omitted)
            self.assertNotIn("motion_preset", omitted["recommendation"]["input_snapshot"])
        context = _acquisition_context(self.acquisition)
        evidence = [_load_run(r)[0] for r in self.runs]
        self.assertEqual(derive_collection_recommendation(acquisition=context, episode_evidence=evidence, source_commit=COMMIT),
            derive_collection_recommendation(acquisition={**context, "motion_preset": None}, episode_evidence=evidence, source_commit=COMMIT))

    def test_original_program_must_bind_both_qualified_motion_and_nontrial_preset(self):
        from tools.fr5_data_factory import _motion_preset_trial_digest
        self.use_registered_motion_preset()
        original = copy.deepcopy(self.lifecycle)
        for mismatch in ("base-qualification", "missing-preset", "different-preset", "trial"):
            with self.subTest(mismatch=mismatch):
                value = copy.deepcopy(original)
                plan = value["plan_envelope"]["plan"]
                for bindings in (plan["binding_digests"], plan["learned_source_program"]["binding_digests"]):
                    if mismatch == "base-qualification":
                        bindings["motion_qualification"] = self.acquisition["catalog"]["combinations"][0]["source_digests"]["motion"]
                    elif mismatch == "missing-preset":
                        bindings.pop("motion_preset", None)
                    elif mismatch == "different-preset":
                        bindings["motion_preset"] = digest("other-preset")
                    else:
                        bindings["motion_preset_trial"] = _motion_preset_trial_digest(bindings)
                self.rebind(value)
                receipt = {**self.preapproval, "plan_envelope": value["plan_envelope"],
                    "plan_digest": value["plan_digest"], "plan_envelope_digest": digest(value["plan_envelope"])}
                self.lifecycle_path.write_text(json.dumps(value))
                self.preapproval_path.write_text(json.dumps(receipt))
                before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
                result = self.call()
                self.assertEqual(result["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_CONTEXT_MISMATCH"])
                self.assertIsNone(result["output_path"])
                self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_preset_change_invalidates_advice_even_without_rollout_input(self):
        self.use_registered_motion_preset()
        result = self.call(rollout_lifecycle_path=None)
        self.assertEqual(result["availability"], "AVAILABLE", result)
        previous = result["recommendation"]["recommendation_digest"]
        changed = self.call(rollout_lifecycle_path=None, acquisition={**self.acquisition, "motion_preset": None},
            expected_recommendation_digest=previous)
        self.assertEqual(changed["reason_codes"], ["COLLECTION_ACQUISITION_INPUT_CHANGED"])
        self.assertIsNone(changed["output_path"])

    def test_native_v2_mixed_history_current_target_and_revalidation(self):
        from tools.data_factory.collection_recommendation_io import _acquisition_context, _load_run
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.call()
        self.assertEqual(result["availability"], "AVAILABLE", result)
        advice = result["recommendation"]
        self.assertEqual(self.call(run_directories=self.runs[::-1]), result)
        self.assertEqual(advice["input_snapshot"]["rollout_condition"]["source"], advice["conditions"][0]["source"])
        self.assertEqual(advice["input_snapshot"]["rollout_condition"]["data_deficit"], "UNKNOWN")
        self.assertNotIn("purpose", advice["input_snapshot"]["rollout_condition"])
        self.assertEqual(advice["input_snapshot"]["rollout_condition"]["condition_indices"], [0])
        baseline = self.call(rollout_lifecycle_path=None)["recommendation"]
        self.assertEqual(advice["object_poses"], baseline["object_poses"])
        self.assertEqual(advice["sampling"], baseline["sampling"])
        self.assertEqual(advice["authority"], AUTHORITY)
        args = dict(acquisition=_acquisition_context(self.acquisition), episode_evidence=[_load_run(r)[0] for r in self.runs],
                    rollout_preapproval_evidence=self.preapproval)
        self.assertEqual(validate_collection_recommendation(advice, **args, rollout_evidence_analysis=self.lifecycle), advice)
        self.assertEqual(advice["input_snapshot"]["scene_binding"]["revision"], 5)
        self.assertEqual(advice["input_snapshot"]["rollout_condition"]["original_scene_binding"]["revision"], 3)
        with self.assertRaisesRegex(ContractError, "ROLLOUT_LIFECYCLE_REQUIRED"):
            validate_collection_recommendation(advice, **args)
        changed = copy.deepcopy(advice)
        changed["input_snapshot"]["rollout_condition"]["condition_indices"] = [1]
        redigest(changed["input_snapshot"], "snapshot_digest")
        redigest(changed, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "INPUT_CHANGED"):
            validate_collection_recommendation(changed, **args, rollout_evidence_analysis=self.lifecycle)
        for count in (1, 7):
            variant = self.call(acquisition={**self.acquisition, "requested_count": count})
            self.assertEqual(variant["availability"], "AVAILABLE", variant)
            self.assertEqual(len(variant["recommendation"]["conditions"]), count)
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))

    def test_scoped_terminal_requires_consumed_complete_and_linked_lineage(self):
        base = scoped_terminal_lifecycle(self.lifecycle)

        def reject(value):
            self.lifecycle_path.write_text(json.dumps(value))
            result = self.call()
            self.assertEqual(result["availability"], "UNAVAILABLE", result)
            self.assertIn("REDEMONSTRATION_UNPROVEN", result["reason_codes"][0])
            self.assertIsNone(result["recommendation"])

        def retrace(value, *, status, failure_code, terminal_state, terminal_phases):
            trace = value["execution_evidence"]["learned_execution"]
            trace.update(status=status, failure_code=failure_code,
                         terminal_state=terminal_state, terminal_phases=terminal_phases)
            redigest(trace, "trace_digest")

        zero_consumed = copy.deepcopy(base)
        retrace(zero_consumed, status="FAILED", failure_code="LEARNED_CANCELLED",
                terminal_state=None, terminal_phases=[])
        reject(zero_consumed)

        for terminal_status, phases in (
            ("PLANNED", []), ("EXECUTING", ["RECYCLE_APPROACH_PTP"]),
            ("FAILED", ["RECYCLE_APPROACH_PTP"]),
        ):
            with self.subTest(terminal_status=terminal_status):
                value = copy.deepcopy(base)
                value["execution_evidence"]["mechanical_terminal"].update(
                    status=terminal_status, terminal_phases=phases)
                reject(value)

        missing_terminal = copy.deepcopy(base)
        missing_terminal.update(code="MECHANICAL_TERMINAL_UNAVAILABLE")
        missing_terminal["execution_evidence"].pop("mechanical_terminal")
        missing_terminal["execution_evidence"]["task_handoff"] = {
            "schema_version": "data_factory.learned_task_handoff.v1",
            "termination_reason": "TASK_OUTPUT_LIMIT_REACHED", "status": "BLOCKED_UNAVAILABLE",
            "code": "MECHANICAL_TERMINAL_UNAVAILABLE", "run_id": missing_terminal["run_id"],
            "plan_digest": missing_terminal["plan_digest"],
            "grant_digest": missing_terminal["execution_evidence"]["task_authority"]["grant"]["grant_digest"],
            "deadline_s": missing_terminal["execution_evidence"]["task_authority"]["grant"]["deadline_s"],
            "required_dependencies": ["QUALIFIED_MECHANICAL_TERMINAL", "RECORDER_TERMINAL_RETENTION"],
            "semantic_success": "NOT_MEASURED",
        }
        reject(missing_terminal)

        bad_grant = copy.deepcopy(base)
        grant = bad_grant["execution_evidence"]["task_authority"]["grant"]
        grant["scope"]["task"] = "other-task"
        redigest(grant, "grant_digest")
        bad_grant["execution_evidence"]["task_authority"]["admission"]["task_grant"] = copy.deepcopy(grant)
        terminal_plan = bad_grant["execution_evidence"]["mechanical_terminal"]["plan"]
        terminal_plan["grant_digest"] = grant["grant_digest"]
        redigest(terminal_plan, "terminal_digest")
        bad_grant["execution_evidence"]["task_handoff"]["terminal_digest"] = terminal_plan["terminal_digest"]
        reject(bad_grant)

        bad_admission = copy.deepcopy(base)
        bad_admission["execution_evidence"]["task_authority"]["admission"]["approved_by"] = "other-authority"
        reject(bad_admission)

        bad_terminal_link = copy.deepcopy(base)
        terminal_plan = bad_terminal_link["execution_evidence"]["mechanical_terminal"]["plan"]
        terminal_plan["grant_digest"] = digest("other-grant")
        redigest(terminal_plan, "terminal_digest")
        bad_terminal_link["execution_evidence"]["task_handoff"]["terminal_digest"] = terminal_plan["terminal_digest"]
        reject(bad_terminal_link)

        bad_handoff_link = copy.deepcopy(base)
        bad_handoff_link["execution_evidence"]["task_handoff"]["terminal_digest"] = digest("other-terminal")
        reject(bad_handoff_link)

        def continued(scoped_history):
            from tools.data_factory.rollout.task_authority import admission
            previous = copy.deepcopy(self.lifecycle)
            previous["plan_envelope"]["precommit_safety"] = {
                "approved_plan_digest": previous["plan_digest"]}
            history = {"plan_envelope": previous["plan_envelope"], "state": "LEARNED_CHUNK_COMPLETE",
                "approval": {"approval_id": "synthetic-first-chunk", "approval_scope": "HUMAN_GATED",
                    "plan_digest": previous["plan_digest"], "run_id": previous["run_id"],
                    "resolved_job_digest": previous["plan_envelope"]["plan"]["resolved_job_digest"]},
                "execution_evidence": {
                    "learned_execution": previous["execution_evidence"]["learned_execution"]}}
            current = copy.deepcopy(self.lifecycle)
            current["plan_envelope"]["plan"]["learned_continuation"] = {
                "previous_plan_digest": previous["plan_digest"],
                "previous_trace_digest": previous["execution_evidence"]["learned_execution"]["trace_digest"],
                "previous_chunk_digest": digest(history), "chunk_index": 1}
            current["execution_evidence"]["learned_history"] = [history]
            self.rebind(current)
            value = scoped_terminal_lifecycle(current)
            if scoped_history:
                grant = value["execution_evidence"]["task_authority"]["grant"]
                old_plan = value["execution_evidence"]["learned_history"][0]["plan_envelope"]["plan"]
                value["execution_evidence"]["learned_history"][0]["approval"] = admission(
                    grant, old_plan, digest(old_plan), grant["deadline_s"] - 1)
                continuation = value["plan_envelope"]["plan"]["learned_continuation"]
                continuation["previous_chunk_digest"] = digest(
                    value["execution_evidence"]["learned_history"][0])
                self.rebind(value)
                plan = value["plan_envelope"]["plan"]
                value["execution_evidence"]["task_authority"]["admission"] = admission(
                    grant, plan, value["plan_digest"], grant["deadline_s"] - 1)
                terminal_plan = value["execution_evidence"]["mechanical_terminal"]["plan"]
                terminal_plan["parent_plan_digest"] = digest(plan)
                redigest(terminal_plan, "terminal_digest")
                value["execution_evidence"]["task_handoff"]["terminal_digest"] = terminal_plan["terminal_digest"]
                from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
                build_run_diagnostic(value)
            return value

        reject(continued(False))  # A human-gated prior chunk is not scoped lineage.
        reject(continued(True))   # Two scoped outputs cannot fit max_outputs == 1.

        unsafe_scene = copy.deepcopy(self.scene)
        unsafe_scene["revision"] += 1
        unsafe_scene["objects"]["cube"].update(state="UNKNOWN", pose=None, source="ROBOT_ACTION")
        self.lifecycle_path.write_text(json.dumps(base))
        self.scene_path.write_text(json.dumps(unsafe_scene))
        result = self.call(acquisition={**self.acquisition, "expected_scene_digest": digest(unsafe_scene)})
        self.assertEqual(result["availability"], "UNAVAILABLE", result)
        self.assertIsNone(result["recommendation"])

    def test_recovered_different_pose_uses_exact_native_direct_condition_within_budget(self):
        from tools.data_factory.operator.catalog import project_direct_poses
        from tools.data_factory.collection_recommendation_io import _acquisition_context, _load_run
        original_pose = copy.deepcopy(self.scene["objects"]["cube"]["pose"])
        self.scene["objects"]["cube"]["pose"]["x_mm"] += 1.0
        self.scene_path.write_text(json.dumps(self.scene))
        acquisition = {**self.acquisition, "expected_scene_digest": digest(self.scene)}
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        for count in (2, 5):
            with self.subTest(count=count):
                request = {**acquisition, "requested_count": count}
                result = self.call(acquisition=request)
                self.assertEqual(result["availability"], "AVAILABLE", result)
                advice = result["recommendation"]
                self.assertEqual(advice["sampling"], {"authoring_mode": "DIRECT_EDIT",
                    "requested_count": count, "normalized_seed": 87, "repeat": 1})
                poses = advice["object_poses"]
                self.assertEqual(len(poses), count)
                self.assertEqual(poses[0], self.scene["objects"]["cube"]["pose"])
                self.assertEqual(poses[1], original_pose)
                self.assertEqual(advice["input_snapshot"]["rollout_condition"]["condition_indices"], [1])
                self.assertEqual(poses, project_direct_poses(request["catalog"], self.selection, poses[0], poses[1:], count))
                self.assertTrue(all(c["yaw_sample_binding"] is None for c in advice["conditions"]))
                self.assertEqual(validate_collection_recommendation(advice,
                    acquisition=_acquisition_context(request), episode_evidence=[_load_run(r)[0] for r in self.runs],
                    rollout_evidence_analysis=self.lifecycle, rollout_preapproval_evidence=self.preapproval), advice)
                self.assertEqual(self.call(acquisition=request), result)
        unavailable = self.call(acquisition={**acquisition, "requested_count": 1})
        self.assertEqual(unavailable["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_BUDGET_INSUFFICIENT"])
        self.assertIsNone(unavailable["output_path"])
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))

    def test_original_receipt_survives_native_validated_continuation_history(self):
        first = copy.deepcopy(self.lifecycle)
        first["plan_envelope"]["precommit_safety"] = {"approved_plan_digest": first["plan_digest"]}
        history = {"plan_envelope": first["plan_envelope"], "state": "LEARNED_CHUNK_COMPLETE",
            "approval": {"approval_id": "synthetic-first-chunk", "approval_scope": "HUMAN_GATED",
                         "plan_digest": first["plan_digest"], "run_id": first["run_id"],
                         "resolved_job_digest": first["plan_envelope"]["plan"]["resolved_job_digest"]},
            "execution_evidence": {"learned_execution": first["execution_evidence"]["learned_execution"]}}
        current = copy.deepcopy(self.lifecycle)
        current["plan_envelope"]["plan"]["learned_continuation"] = {
            "previous_plan_digest": first["plan_digest"],
            "previous_trace_digest": first["execution_evidence"]["learned_execution"]["trace_digest"],
            "previous_chunk_digest": digest(history), "chunk_index": 1}
        current["execution_evidence"]["learned_history"] = [history]
        self.rebind(current)
        self.lifecycle_path.write_text(json.dumps(current))
        result = self.call()
        self.assertEqual(result["availability"], "AVAILABLE", result)
        target = result["recommendation"]["input_snapshot"]["rollout_condition"]
        self.assertEqual(target["source_plan_digest"], first["plan_digest"])
        self.assertNotEqual(target["source_plan_digest"], current["plan_digest"])
        current["execution_evidence"]["learned_history"][0]["approval"]["run_id"] = "different-run"
        self.lifecycle_path.write_text(json.dumps(current))
        self.assertEqual(self.call()["availability"], "UNAVAILABLE")

    def test_native_cli_consumes_original_files_without_publication(self):
        import io
        path = self.root / "acquisition.json"
        path.write_text(json.dumps(self.acquisition))
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        stdout = io.StringIO()
        arguments = ["--source-commit", COMMIT, "--acquisition-input", str(path),
                     "--rollout-lifecycle", str(self.lifecycle_path)]
        for run in self.runs:
            arguments.extend(["--run-dir", str(run)])
        with mock.patch("sys.stdout", stdout):
            self.assertEqual(recommend_main(arguments), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result, self.call())
        self.assertIsNone(result["output_path"])
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}, before)

    def test_original_source_receipt_is_required_and_bound_without_historical_reconstruction(self):
        self.preapproval_path.unlink()
        self.assertEqual(self.call()["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_SOURCE_REQUIRED"])
        self.assertFalse(self.preapproval_path.exists())
        for edit in (
            lambda r: r["resolved_inputs"]["normalized_job"].update(x_mm=55.0),
            lambda r: r["resolved_inputs"]["input_digests"].update(cell_calibration=digest("different-cell")),
            lambda r: r.update(run_id="different-run"),
            lambda r: r.update(plan_digest=digest("different-plan")),
            lambda r: r.update(plan_envelope_digest=digest("different-envelope")),
        ):
            value = copy.deepcopy(self.preapproval)
            edit(value)
            self.preapproval_path.write_text(json.dumps(value))
            before = self.preapproval_path.read_bytes()
            result = self.call()
            self.assertEqual(result["availability"], "UNAVAILABLE", result)
            self.assertEqual(result["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_SOURCE_BINDING"])
            self.assertIsNone(result["output_path"])
            self.assertEqual(self.preapproval_path.read_bytes(), before)

    def test_unjustified_or_mismatched_original_cannot_become_a_target(self):
        edits = [
            (lambda r: r["execution_evidence"]["semantic_decision"].update(source="HIL_PROXY"), "REDEMONSTRATION_UNPROVEN"),
            (lambda r: r["execution_evidence"].pop("semantic_decision"), "REDEMONSTRATION_UNPROVEN"),
            (lambda r: r.update(run_id="different-run"), "DIAGNOSTIC_BINDING"),
            (lambda r: r["recorder_evidence"].update(state="FROZEN"), "DIAGNOSTIC_BINDING"),
            (lambda r: r["plan_envelope"]["plan"]["scene_binding"].update(revision=4), "ROLLOUT_OBJECT_MISMATCH"),
            (lambda r: r["execution_evidence"]["learned_execution"]["checkpoint"].update(tree_digest=digest("other")), "DIAGNOSTIC_BINDING"),
        ]
        for edit, code in edits:
            with self.subTest(code=code):
                value = copy.deepcopy(self.lifecycle)
                edit(value)
                self.rebind(value)
                self.lifecycle_path.write_text(json.dumps(value))
                result = self.call()
                self.assertEqual(result["availability"], "UNAVAILABLE", result)
                self.assertIn(code, result["reason_codes"][0])
                self.assertIsNone(result["output_path"])
        self.lifecycle_path.write_text(json.dumps(self.lifecycle))
        baseline = self.call()["recommendation"]["recommendation_digest"]
        self.assertEqual(self.call(acquisition={**self.acquisition, "normalized_seed": 88},
            expected_recommendation_digest=baseline)["reason_codes"], ["COLLECTION_ACQUISITION_INPUT_CHANGED"])

    def test_controller_fault_task_and_current_profile_changes_remain_insufficient(self):
        value = copy.deepcopy(self.lifecycle)
        value.update(code="CONTROLLER_FAULT", semantic_verdict=None)
        value["execution_evidence"].pop("semantic_decision")
        value["execution_evidence"].pop("semantic_verdict")
        value["execution_evidence"]["learned_execution"].update(status="FAILED", failure_code="CONTROLLER_FAULT",
                                                               terminal_state=None, terminal_phases=[])
        self.rebind(value)
        self.lifecycle_path.write_text(json.dumps(value))
        self.assertEqual(self.call()["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_REDEMONSTRATION_UNPROVEN"])
        for key in ("cell_calibration", "object_profile", "grasp_profile", "collection_profile", "motion_qualification"):
            value = copy.deepcopy(self.lifecycle)
            plan = value["plan_envelope"]["plan"]
            plan["binding_digests"][key] = digest("different-context")
            plan["learned_source_program"]["binding_digests"][key] = digest("different-context")
            self.rebind(value)
            self.lifecycle_path.write_text(json.dumps(value))
            self.assertEqual(self.call()["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_SOURCE_BINDING"])
        value = copy.deepcopy(self.lifecycle)
        for step in value["plan_envelope"]["plan"]["learned_source_program"]["steps"]:
            if step.get("pause_after") == "SEMANTIC_VERDICT":
                del step["pause_after"]
            if step["phase"] == "RETREAT_LIN":
                step["pause_after"] = "SEMANTIC_VERDICT"
        self.rebind(value)
        self.lifecycle_path.write_text(json.dumps(value))
        self.assertEqual(self.call()["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_SOURCE_BINDING"])

    def test_v2_rechecks_lifecycle_and_scene_after_native_sampling(self):
        from tools.data_factory import collection_recommendation_io as io
        original = io.derive_collection_recommendation
        for target in ("lifecycle", "preapproval", "scene"):
            self.lifecycle_path.write_text(json.dumps(self.lifecycle))
            self.preapproval_path.write_text(json.dumps(self.preapproval))
            self.scene_path.write_text(json.dumps(self.scene))
            def changed(**kwargs):
                result = original(**kwargs)
                if target == "lifecycle":
                    self.lifecycle_path.write_text(json.dumps({**self.lifecycle, "code": "REPLACED"}))
                elif target == "preapproval":
                    self.preapproval_path.write_text(json.dumps({**self.preapproval, "run_id": "REPLACED"}))
                else:
                    self.scene_path.write_text(json.dumps({**self.scene, "revision": 4}))
                return result
            with mock.patch.object(io, "derive_collection_recommendation", side_effect=changed):
                result = self.call()
            self.assertEqual(result["availability"], "UNAVAILABLE")
            self.assertEqual(result["reason_codes"], ["COLLECTION_RECOMMENDATION_ROLLOUT_INPUT_CHANGED"
                if target != "scene" else "COLLECTION_ACQUISITION_SCENE_CHANGED"])
            self.assertIsNone(result["output_path"])


class CollectionRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_operator_catalog(
            ROOT, device_ids=["sampling-camera-a", "sampling-camera-b"],
        )

    def setUp(self) -> None:
        self.fixture = RecommendationFixture()

    def application(
        self, campaign_factory=None,
    ) -> CollectionOperatorApplication:
        combination = next(
            item for item in self.catalog["combinations"]
            if item["execution"]["TEST_COLLECTION"]["executable"]
            and item["task_id"] == "pickup_e2e"
            and item["object_id"] == "wood-cube-24mm-r001"
            and item["variant_id"] == "TWO_STAGE_ALIGN_V2"
        )
        selection = {
            "schema_version": "data_factory.operator_selection.v1",
            "combination_digest": combination["combination_digest"],
            "data_mode": "TEST_COLLECTION",
            **{
                field: combination[field]
                for field in (
                    "workspace_id", "frame_id", "task_id", "object_id",
                    "grasp_id", "cell_id", "start_pose_id", "motion_id",
                    "variant_id", "camera_profile_id", "camera_device_id",
                )
            },
            "policy_id": "DETERMINISTIC_SPREAD",
        }
        environment = {
            "schema_version": "data_factory.operator_environment.v1",
            "state": "READY",
            "observed_at": "2026-09-04T05:00:00Z",
            "components": {
                name: {
                    "state": "READY", "owner": f"owner-{name}",
                    "reason": "ATTACHED",
                }
                for name in ("robot", "controller", "gripper", "camera")
            },
        }
        return CollectionOperatorApplication(
            session_id="recommendation-application-r1",
            operator_label="local-operator", catalog=self.catalog,
            initial_selection=selection, projector=projection,
            environment_call=lambda: copy.deepcopy(environment),
            prepare_environment_call=lambda: copy.deepcopy(environment),
            campaign_factory=(
                campaign_factory if campaign_factory is not None else
                lambda *_args: self.fail("campaign was not requested")
            ),
            initial_environment=environment,
        )

    def test_canonical_manifest_order_is_deterministic_and_rejoinable(self) -> None:
        before = copy.deepcopy(self.fixture.__dict__)
        first = self.fixture.build()
        second = self.fixture.build(
            episode_evidence=list(reversed(self.fixture.evidence)),
            claims=list(reversed(self.fixture.claims)),
            suggested_draft_patches=list(reversed(self.fixture.patches)),
        )
        self.assertEqual(first, second)

        self.assertEqual(
            [episode["manifest_order_index"] for episode in first["input_snapshot"]["episodes"]],
            [0, 1],
        )
        self.assertEqual(
            first["recommendation_digest"],
            digest({key: value for key, value in first.items() if key != "recommendation_digest"}),
        )
        self.assertEqual(
            validate_collection_recommendation(
                first,
                campaign_manifest=self.fixture.manifest,
                campaign_hypothesis=self.fixture.hypothesis,
                campaign_draft=self.fixture.draft,
                campaign_compilation_receipt=self.fixture.receipt,
                episode_evidence=self.fixture.evidence,
            ),
            first,
        )
        with self.assertRaisesRegex(ContractError, "EPISODE_ORDER"):
            validate_collection_recommendation(
                first,
                campaign_manifest=self.fixture.manifest,
                campaign_hypothesis=self.fixture.hypothesis,
                campaign_draft=self.fixture.draft,
                campaign_compilation_receipt=self.fixture.receipt,
                episode_evidence=list(reversed(self.fixture.evidence)),
            )
        self.assertEqual(self.fixture.__dict__, before)

    def test_retained_recommendation_preserves_later_native_draft_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "data"), evidence_root=str(root / "runs"),
            )
            runs = fixture.store()
            result = recommend_stored_collection(run_directories=runs[:1], source_commit=COMMIT)
            recommendation = result["recommendation"]
            target = recommendation["suggested_draft_patches"][0]["value"]["direct_slots"][0]["slot_id"]
            edits = {
                "excluded": [target], "pinned": [fixture.manifest["slots"][0]["slot_id"]],
                "normalized_seed": fixture.draft["normalized_seed"] + 1, "requested_count": 1,
            }
            for field, value in edits.items():
                with self.subTest(field=field):
                    forbidden = mock.Mock(side_effect=AssertionError("no physical effects"))
                    app = CampaignOperator(
                        session_id="edited-campaign", lifecycle_owner="TEST_OPERATOR",
                        workspace={"identity":"SYNTHETIC"}, hypothesis=fixture.hypothesis,
                        draft=fixture.draft, effect_scope="FAKE", lifecycle_action="AUTHOR_ONLY",
                        data_disposition="TEST_ONLY",
                        subsystems={"planner":{"readiness":"READY", "capability":"AUTHOR", "reason":"SYNTHETIC"}},
                        expires_at="2099-01-01T00:00:00Z", initial_scene_digest=digest("scene"),
                        scene_evidence_call=forbidden, fake_lifecycle_factory=forbidden,
                        side_effect_counter_call=lambda: {name:0 for name in SIDE_EFFECT_COUNTERS},
                        clock=lambda: NOW,
                    )
                    view = app.core.snapshot()
                    payload = {key: copy.deepcopy(fixture.draft[key]) for key in (
                        "requested_count", "normalized_seed", "pinned", "excluded", "direct_slots",
                    )}
                    payload.update(authoring_mode="ASSISTED", **{field:value})
                    app.core.consume({
                        "schema_version":INTENT_SCHEMA, "intent_id":"current-user-edit",
                        "session_id":view["session_id"], "view_revision":view["revision"],
                        "view_digest":view["view_digest"], "op":"update_draft", "payload":payload,
                    })
                    current = app.core.snapshot()
                    with self.assertRaisesRegex(ContractError, "COLLECTION_RECOMMENDATION_DRAFT_CHANGED"):
                        project_campaign_update_intent(
                            recommendation, compiled_authoring=fixture.authoring(), operator_view=current,
                            data_quality_analysis=result["data_quality_analysis"],
                        )
                    self.assertEqual(app.core.snapshot(), current)
                    self.assertEqual(app.draft[field], value)
                    forbidden.assert_not_called()

    def test_native_stored_consumer_replay_invalidation_and_update_draft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "synthetic-data"), evidence_root=str(root / "runs"),
            )
            runs = fixture.store()
            before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            arguments = dict(run_directories=runs[:1], source_commit=COMMIT, output_root=root / "derived")
            first = recommend_stored_collection(**arguments)
            self.assertEqual(first["availability"], "AVAILABLE", first)
            self.assertEqual(first, recommend_stored_collection(**arguments))
            advice = first["recommendation"]
            patch = advice["suggested_draft_patches"][0]
            self.assertEqual(patch["field"], "campaign_selection")
            self.assertEqual(patch["value"]["requested_count"], 1)
            factory = mock.Mock(side_effect=AssertionError("collection must not start"))
            application = CampaignOperator(
                session_id="recommendation-campaign", lifecycle_owner="TEST_OPERATOR",
                workspace={"identity": "SYNTHETIC"}, hypothesis=fixture.hypothesis,
                draft=fixture.draft, effect_scope="FAKE", lifecycle_action="AUTHOR_ONLY",
                data_disposition="TEST_ONLY",
                subsystems={"planner": {"readiness": "READY", "capability": "AUTHOR", "reason": "SYNTHETIC"}},
                expires_at="2099-01-01T00:00:00Z",
                initial_scene_digest=digest("scene"), scene_evidence_call=factory,
                side_effect_counter_call=lambda: {name: 0 for name in SIDE_EFFECT_COUNTERS},
                fake_lifecycle_factory=factory, clock=lambda: NOW,
            )
            # Unrelated owner publication does not stale the recommendation.
            original_view = application.core.snapshot()
            application.core.transition(lambda: None)
            view = application.core.snapshot()
            self.assertGreater(view["revision"], original_view["revision"])
            intent = project_campaign_update_intent(
                advice, compiled_authoring=fixture.authoring(), operator_view=view,
                data_quality_analysis=first["data_quality_analysis"],
            )
            application.core.consume(intent)
            self.assertEqual(application.draft["requested_count"], 1)
            with self.assertRaisesRegex(ContractError, "STALE_VIEW"):
                application.core.consume({**intent, "intent_id": "stale-native-advice"})
            compile_view = application.core.snapshot()
            application.core.consume({
                **intent, "intent_id": "compile-selected-condition", "op": "compile_draft",
                "view_revision": compile_view["revision"], "view_digest": compile_view["view_digest"],
                "payload": {},
            })
            missing = {digest(cell["condition"]) for cell in first["data_quality_analysis"]["cells"]
                       if cell["counts"]["collected"] == 0}
            bases = {base["base_condition_digest"]: base for base in fixture.hypothesis["base_conditions"]}
            compiled_conditions = {digest(bases[slot["base_condition_digest"]]["coverage_condition"])
                                   for slot in application.manifest["slots"]}
            self.assertEqual(compiled_conditions, missing)
            self.assertNotEqual(application.manifest["slots"][0]["base_condition_digest"],
                                fixture.manifest["slots"][0]["base_condition_digest"])
            factory.assert_not_called()
            self.assertFalse(any(application.projection()["side_effect_counters"].values()))
            with self.assertRaisesRegex(ContractError, "CAMPAIGN_OWNER_REQUIRED"):
                project_update_draft_intent(
                    advice, selected_change_id=patch["change_id"], operator_view=view,
                    data_quality_analysis=first["data_quality_analysis"],
                )
            altered = copy.deepcopy(advice)
            altered["suggested_draft_patches"][0]["value"]["normalized_seed"] += 1
            redigest(altered, "recommendation_digest")
            with self.assertRaisesRegex(ContractError, "SELECTION_BINDING"):
                project_campaign_update_intent(
                    altered, compiled_authoring=fixture.authoring(), operator_view=view,
                    data_quality_analysis=first["data_quality_analysis"],
                )
            altered_view = copy.deepcopy(view)
            altered_view["projection"]["draft"]["source"]["hypothesis_digest"] = digest("other-source")
            with self.assertRaisesRegex(ContractError, "CAMPAIGN_VIEW"):
                project_campaign_update_intent(
                    advice, compiled_authoring=fixture.authoring(), operator_view=altered_view,
                    data_quality_analysis=first["data_quality_analysis"],
                )
            second = recommend_stored_collection(**{**arguments, "run_directories": runs})
            self.assertEqual(second["availability"], "AVAILABLE", second)
            self.assertNotEqual(first["output_path"], second["output_path"])
            self.assertEqual(second, recommend_stored_collection(**{**arguments, "run_directories": runs[::-1]}))
            self.assertNotEqual(advice["input_snapshot"]["snapshot_digest"], second["recommendation"]["input_snapshot"]["snapshot_digest"])
            for path, content in before.items():
                self.assertEqual(Path(path).read_bytes(), content)
            self.assertTrue(Path(first["output_path"]).is_dir())
            with mock.patch("builtins.print"):
                self.assertEqual(recommend_main([
                    "--run-dir", str(runs[0]), "--source-commit", COMMIT,
                    "--output-root", str(root / "derived"),
                ]), 0)

    def test_native_missing_mismatched_changed_sources_and_exclusive_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "synthetic-data"), evidence_root=str(root / "runs"),
            )
            runs = fixture.store()
            arguments = dict(run_directories=runs[:1], source_commit=COMMIT)
            for output in (runs[0], root / "synthetic-data", root):
                with self.subTest(output=output), self.assertRaisesRegex(ContractError, "OUTPUT_OVERLAP"):
                    recommend_stored_collection(**arguments, output_root=output)
            output = root / "derived"
            first = recommend_stored_collection(**arguments, output_root=output)
            path = Path(first["output_path"]) / "coverage_report.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ContractError, "OUTPUT_CONFLICT"):
                recommend_stored_collection(**arguments, output_root=output)
            authoring_path = runs[0] / "compiled_authoring_evidence.json"
            original = authoring_path.read_text()
            authoring_path.unlink()
            missing = recommend_stored_collection(**arguments)
            self.assertEqual(missing["reason_codes"], ["COLLECTION_RECOMMENDATION_AUTHORING_UNAVAILABLE"])
            self.assertIsNone(missing["recommendation"])
            altered = json.loads(original)
            altered["draft"]["normalized_seed"] += 1
            redigest(altered, "authoring_digest")
            authoring_path.write_text(json.dumps(altered))
            self.assertEqual(recommend_stored_collection(**arguments)["availability"], "UNAVAILABLE")
            authoring_path.write_text(original)
            other = copy.deepcopy(fixture)
            other.draft["manifest_id"] = "another-valid-campaign"
            other.manifest, other.receipt = compile_collection_campaign(other.draft, hypothesis=other.hypothesis)
            authoring_path.write_text(json.dumps(other.authoring()))
            self.assertEqual(recommend_stored_collection(**arguments)["reason_codes"],
                             ["COLLECTION_RECOMMENDATION_AUTHORING_MISMATCH"])
            authoring_path.write_text(original)
            technical = Path(fixture.evidence[0]["ledger"]["artifacts"]["technical"]["artifact_path"])
            technical.write_text("{}")
            changed = recommend_stored_collection(**arguments)
            self.assertEqual(changed["availability"], "UNAVAILABLE")
            self.assertIn("DIGEST", changed["reason_codes"][0])

    def test_native_concurrent_identical_publish_reuses_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "synthetic-data"), evidence_root=str(root / "runs"),
            )
            runs = fixture.store()
            start = threading.Barrier(8)

            def publish(_):
                start.wait(timeout=10)
                return recommend_stored_collection(
                    run_directories=runs[:1], source_commit=COMMIT,
                    output_root=root / "derived",
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(publish, range(8)))
            self.assertTrue(all(result == results[0] for result in results))
            self.assertEqual(results[0]["availability"], "AVAILABLE")
            self.assertEqual(len(list((root / "derived").iterdir())), 1)

    def test_native_failed_publication_does_not_expose_partial_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "synthetic-data"), evidence_root=str(root / "runs"),
            )
            runs = fixture.store()
            arguments = dict(run_directories=runs[:1], source_commit=COMMIT,
                             output_root=root / "derived")
            original = json.dump
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("interrupted publication")
                return original(*args, **kwargs)

            with mock.patch("tools.data_factory.collection_recommendation_io.json.dump", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "interrupted publication"):
                    recommend_stored_collection(**arguments)
            self.assertEqual(list((root / "derived").iterdir()), [])
            self.assertEqual(recommend_stored_collection(**arguments)["availability"], "AVAILABLE")

    def test_native_source_commit_is_explicitly_unverified_caller_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "synthetic-data"), evidence_root=str(root / "runs"),
            )
            result = recommend_stored_collection(run_directories=fixture.store()[:1], source_commit="0" * 40)
            self.assertEqual(result["implementation_provenance"], {
                "source_commit": "0" * 40, "verification": "CALLER_SUPPLIED_UNVERIFIED",
            })

    def test_postcommit_owner_retains_exact_authoring_for_native_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RecommendationFixture(
                dataset_root=str(root / "synthetic-data"), evidence_root=str(root / "runs"),
            )
            runs = fixture.store()
            evidence = fixture.evidence[0]
            artifacts = evidence["artifacts"]
            run_dir = runs[0]
            (run_dir / "compiled_authoring_evidence.json").unlink()
            for name, artifact in {
                "result.json": "run", "storage_usage.json": "episode",
                "staging_manifest.json": "staging_manifest", "preapproval_evidence.json": "plan",
                "technical_validator.json": "technical",
            }.items():
                (run_dir / name).write_text(json.dumps(artifacts[artifact]))
            metadata = Path(fixture.dataset_root) / "meta"
            (metadata / "source_provenance").mkdir(parents=True)
            (metadata / "source_provenance/episode-000010.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in artifacts["source_provenance"]),
            )
            (metadata / "recording_quality.jsonl").write_text(json.dumps(artifacts["recording_quality"]) + "\n")
            context = {
                "manifest": fixture.manifest, "intent": artifacts["intent"],
                "compiled_authoring": fixture.authoring(),
            }
            before = copy.deepcopy(context)
            payload = {
                "run_id": artifacts["run"]["run_id"], "run_root": fixture.evidence_root,
                "dataset_root": fixture.dataset_root,
            }
            # This fixture predates trajectory v2; isolate that unrelated runtime
            # input validator, while exercising real ledger/compiler/state owners.
            with mock.patch.object(run_job, "_validated_trajectory_binding", return_value=None):
                reference = run_job._write_episode_ledger(
                    payload, {}, {"repo_id": evidence["ledger"]["dataset"]["repo_id"]},
                    SimpleNamespace(execution_response=artifacts["execution"], plan_envelope=artifacts["plan"]["plan_envelope"]),
                    artifacts["episode"], artifacts["runtime_binding"], context,
                    trajectory_binding=None, episode_locator=evidence["ledger"]["episode"]["lerobot_v3_locator"],
                )
            self.assertEqual(json.loads((run_dir / "compiled_authoring_evidence.json").read_text()), before["compiled_authoring"])
            self.assertEqual(context, before)
            run_job.bind_candidate_episode_state(reference, run_dir / "candidate_admission.json")
            result = recommend_stored_collection(
                run_directories=[run_dir], source_commit=COMMIT, output_root=root / "derived",
            )
            self.assertEqual(result["availability"], "AVAILABLE", result)
            self.assertEqual(result["recommendation"]["suggested_draft_patches"][0]["value"]["requested_count"], 1)
            with self.assertRaisesRegex(ContractError, "AUTHORING_BINDING"):
                other = copy.deepcopy(context)
                other["manifest"] = copy.deepcopy(context["manifest"])
                other["manifest"]["manifest_id"] = "other-manifest"
                redigest(other["manifest"], "manifest_digest")
                other["intent"]["manifest_digest"] = other["manifest"]["manifest_digest"]
                redigest(other["intent"], "intent_digest")
                binding = {**artifacts["runtime_binding"],
                           "manifest_digest": other["manifest"]["manifest_digest"],
                           "intent_digest": other["intent"]["intent_digest"]}
                run_job._validate_episode_ledger_context(other, episode_binding=binding, run_id=payload["run_id"])
            legacy = {key: context[key] for key in ("manifest", "intent")}
            self.assertEqual(run_job._validate_episode_ledger_context(
                legacy, episode_binding=artifacts["runtime_binding"], run_id=payload["run_id"],
            ), legacy)

    def test_native_quality_pending_review_and_covered_domain_do_not_invent_deficits(self) -> None:
        fixture = self.fixture
        first_slot = {key: value for key, value in fixture.manifest["slots"][0].items() if key != "order_index"}
        other_base = next(base for base in fixture.hypothesis["base_conditions"]
                          if base["base_condition_digest"] != first_slot["base_condition_digest"])
        fixture.draft.update(selector="DIRECT_LIST", direct_slots=[first_slot, {
            **first_slot, "slot_id": campaign_cell_id(other_base["base_condition_digest"], "start-3", "OOD", 0),
            "robot_start_pose_id": "start-3",
            "base_condition_digest": other_base["base_condition_digest"], "split_group": "OOD",
        }])
        fixture.manifest, fixture.receipt = compile_collection_campaign(
            fixture.draft, hypothesis=fixture.hypothesis,
        )
        fixture.evidence = [fixture.episode(index, index + 10) for index in range(len(fixture.manifest["slots"]))]
        report, advice = derive_collection_recommendation(
            compiled_authoring=fixture.authoring(), episode_evidence=fixture.evidence,
            source_commit=COMMIT,
        )
        self.assertTrue(all(cell["counts"]["collected"] for cell in report["cells"]))
        self.assertEqual(advice["suggested_draft_patches"], [])
        self.assertFalse(any(claim["class"] == "SUGGESTED" for claim in advice["claims"]))
        pending = fixture.evidence[0]
        pending["candidate"].update(semantic_status="PENDING", reviewed_by=None, reviewed_at=None)
        fixture.rebind_episode(pending)
        report, advice = derive_collection_recommendation(
            compiled_authoring=fixture.authoring(), episode_evidence=[pending], source_commit=COMMIT,
        )
        self.assertEqual(sum(cell["counts"]["pending_review"] for cell in report["cells"]), 1)
        self.assertEqual(sum(cell["counts"]["human_semantic_pass"] for cell in report["cells"]), 0)
        self.assertEqual({claim["subject"] for claim in advice["claims"] if claim["class"] == "UNKNOWN"},
                         {"person", "background", "robot", "rollout", "semantic"})

    def test_native_slot_advice_does_not_discard_explicit_selection_constraints(self) -> None:
        fixture = self.fixture
        fixture.draft["pinned"] = [fixture.manifest["slots"][0]["slot_id"]]
        fixture.manifest, fixture.receipt = compile_collection_campaign(
            fixture.draft, hypothesis=fixture.hypothesis,
        )
        fixture.evidence = [fixture.episode(0, 10)]
        before = copy.deepcopy(fixture.__dict__)
        _, advice = derive_collection_recommendation(
            compiled_authoring=fixture.authoring(), episode_evidence=fixture.evidence,
            source_commit=COMMIT,
        )
        selection = advice["suggested_draft_patches"][0]["value"]
        self.assertEqual(selection["pinned"], fixture.draft["pinned"])
        self.assertEqual(selection["excluded"], fixture.draft["excluded"])
        self.assertEqual(selection["direct_slots"][0]["slot_id"], fixture.draft["pinned"][0])
        self.assertLessEqual(selection["requested_count"], fixture.draft["requested_count"])
        missing = selection["direct_slots"][1]
        self.assertNotEqual(missing["base_condition_digest"], fixture.manifest["slots"][0]["base_condition_digest"])
        self.assertEqual(fixture.__dict__, before)

    def test_received_digests_and_list_order_are_validated_before_parsing(self) -> None:
        recommendation = self.fixture.build()
        reordered = copy.deepcopy(recommendation)
        reordered["claims"].reverse()
        self.assertNotEqual(
            reordered["recommendation_digest"],
            digest({
                key: value for key, value in reordered.items()
                if key != "recommendation_digest"
            }),
        )
        with self.assertRaisesRegex(ContractError, "RECOMMENDATION_DIGEST"):
            validate_collection_recommendation(reordered)

        rehashed_claims = copy.deepcopy(reordered)
        redigest(rehashed_claims, "recommendation_digest")
        rehashed_patches = copy.deepcopy(recommendation)
        rehashed_patches["suggested_draft_patches"].reverse()
        redigest(rehashed_patches, "recommendation_digest")
        rehashed_ref = copy.deepcopy(recommendation)
        rehashed_ref["input_snapshot"]["data_quality_analysis_ref"][
            "reason_codes"
        ] = ["REPORT_NOT_SUPPLIED", "ANALYSIS_NOT_SUPPLIED"]
        redigest(rehashed_ref["input_snapshot"], "snapshot_digest")
        redigest(rehashed_ref, "recommendation_digest")
        rehashed_evidence = copy.deepcopy(recommendation)
        observed = next(
            claim for claim in rehashed_evidence["claims"]
            if claim["class"] == "OBSERVED"
        )
        observed["evidence_refs"] = sorted([
            observed["evidence_refs"][0],
            recommendation["input_snapshot"]["episodes"][0]["dataset_digest"],
        ], reverse=True)
        redigest(rehashed_evidence, "recommendation_digest")
        for label, changed in (
            ("claims", rehashed_claims),
            ("patches", rehashed_patches),
            ("analysis-ref", rehashed_ref),
            ("claim-evidence", rehashed_evidence),
        ):
            with self.subTest(label=label), self.assertRaises(ContractError):
                validate_collection_recommendation(changed)

    def test_canonical_campaign_and_loaded_intent_joins_reject_review_probes(self) -> None:
        for field in self.fixture.evidence[0]["ledger"]["bindings"]:
            evidence = copy.deepcopy(self.fixture.evidence)
            episode = evidence[0]
            episode["ledger"]["bindings"][field] = (
                "forged-intent"
                if field == "robot_start_pose_id"
                else digest("forged-intent" if field == "intent_digest" else field)
            )
            redigest(episode["ledger"], "ledger_digest")
            episode["state"]["ledger_digest"] = episode["ledger"][
                "ledger_digest"
            ]
            redigest(episode["state"], "state_digest")
            with self.subTest(binding=field), self.assertRaisesRegex(
                ContractError, "SOURCE_BINDING",
            ):
                self.fixture.build(episode_evidence=evidence)

        manifest = copy.deepcopy(self.fixture.manifest)
        first_slot, second_slot = manifest["slots"]
        for field in (
            "base_condition_digest", "robot_start_pose_id", "split_group",
            "repeat_index",
        ):
            second_slot[field] = first_slot[field]
        redigest(manifest, "manifest_digest")
        evidence = copy.deepcopy(self.fixture.evidence)
        for index, episode in enumerate(evidence):
            slot = manifest["slots"][index]
            episode["artifacts"]["manifest"] = manifest
            episode["ledger"]["artifacts"]["manifest"][
                "artifact_digest"
            ] = digest(manifest)
            episode["ledger"]["bindings"].update(
                manifest_digest=manifest["manifest_digest"],
                slot_digest=digest(slot),
                base_condition_digest=slot["base_condition_digest"],
                robot_start_pose_id=slot["robot_start_pose_id"],
            )
            redigest(episode["ledger"], "ledger_digest")
            episode["state"]["ledger_digest"] = episode["ledger"][
                "ledger_digest"
            ]
            redigest(episode["state"], "state_digest")
        claims = copy.deepcopy(self.fixture.claims)
        next(
            claim for claim in claims if claim["class"] == "OBSERVED"
        )["evidence_refs"] = [manifest["manifest_digest"]]
        with self.assertRaisesRegex(
            ContractError, "COLLECTION_MANIFEST_DISALLOWED_PAIR",
        ):
            self.fixture.build(
                campaign_manifest=manifest, episode_evidence=evidence,
                claims=claims,
            )

    def test_duplicate_gap_and_manifest_mismatch_fail_closed(self) -> None:
        duplicate = copy.deepcopy(self.fixture.evidence)
        duplicate[1]["manifest_order_index"] = 0
        gap = copy.deepcopy(self.fixture.evidence)
        gap[1]["manifest_order_index"] = 2
        mismatch = copy.deepcopy(self.fixture.evidence)
        mismatch[0]["manifest_order_index"] = 1
        mismatch[1]["manifest_order_index"] = 0
        for label, evidence in (("duplicate", duplicate), ("gap", gap), ("mismatch", mismatch)):
            with self.subTest(label=label), self.assertRaises(ContractError):
                self.fixture.build(episode_evidence=evidence)

        same_dataset_episode = [
            self.fixture.evidence[0], self.fixture.episode(1, 10),
        ]
        self.assertNotEqual(
            same_dataset_episode[0]["ledger"]["dataset"]["dataset_digest"],
            same_dataset_episode[1]["ledger"]["dataset"]["dataset_digest"],
        )
        with self.assertRaisesRegex(ContractError, "EPISODE_DUPLICATE"):
            self.fixture.build(episode_evidence=same_dataset_episode)

    def test_all_join_layers_are_digest_and_identity_bound(self) -> None:
        mutations = {
            "manifest": lambda manifest, evidence: manifest.update(manifest_id="forged"),
            "run": lambda manifest, evidence: evidence[0]["artifacts"]
            ["run"].update(run_id="forged-run"),
            "dataset": lambda manifest, evidence: evidence[0]["ledger"]
            ["dataset"].update(dataset_id="forged"),
            "episode_ref": lambda manifest, evidence: evidence[0]["ledger"]
            ["episode"]["episode_ref"].update(episode_index=99),
            "locator": lambda manifest, evidence: evidence[0]["ledger"]
            ["episode"]["lerobot_v3_locator"].update(repo_id="forged/repo"),
            "ledger": lambda manifest, evidence: evidence[0]["ledger"].update(ledger_digest=digest("forged")),
            "state": lambda manifest, evidence: evidence[0]["state"].update(ledger_digest=digest("forged")),
            "candidate": lambda manifest, evidence: evidence[0]["candidate"].update(run_id="forged-run"),
            "provenance": lambda manifest, evidence: evidence[0]["artifacts"]
            ["source_provenance"][0].update(frame_index=1),
            "recording_quality": lambda manifest, evidence: evidence[0]
            ["artifacts"]["recording_quality"].update(frames=3),
            "slot": lambda manifest, evidence: evidence[0]["ledger"]["bindings"].update(slot_digest=digest("forged")),
        }
        for label, mutate in mutations.items():
            manifest, evidence = copy.deepcopy((self.fixture.manifest, self.fixture.evidence))
            mutate(manifest, evidence)
            with self.subTest(label=label), self.assertRaises(ContractError):
                self.fixture.build(campaign_manifest=manifest, episode_evidence=evidence)

    def test_recomputed_invented_pass_evidence_fails_at_the_ledger_owner(self) -> None:
        cases = (
            (
                "technical-schema",
                lambda item: item["artifacts"]["technical"].update(
                    schema_version="invented.technical_pass.v1",
                ),
                "EPISODE_LEDGER_TECHNICAL",
            ),
            (
                "technical-field-drift",
                lambda item: item["artifacts"]["technical"].update(
                    invented_pass=True,
                ),
                "EPISODE_LEDGER_TECHNICAL_FIELDS",
            ),
            (
                "semantic-checklist",
                lambda item: item["candidate"].update(
                    checklist_id="invented-pass-v1",
                ),
                "EPISODE_LEDGER_CANDIDATE_BINDING",
            ),
        )
        for label, mutate, code in cases:
            evidence = copy.deepcopy(self.fixture.evidence)
            mutate(evidence[0])
            self.fixture.rebind_episode(evidence[0])
            with self.subTest(label=label), self.assertRaisesRegex(
                ContractError, code,
            ):
                self.fixture.build(episode_evidence=evidence)

    def test_malformed_candidate_enums_keep_the_ledger_error_code(self) -> None:
        for field in (
            "operational_gate", "operational_source", "checklist_id",
            "semantic_status",
        ):
            evidence = copy.deepcopy(self.fixture.evidence)
            evidence[0]["candidate"][field] = []
            self.fixture.rebind_episode(evidence[0])
            with self.subTest(field=field), self.assertRaisesRegex(
                ContractError, "EPISODE_LEDGER_CANDIDATE_BINDING",
            ):
                self.fixture.build(episode_evidence=evidence)

    def test_analysis_owners_availability_alias_and_physical_scope_are_strict(self) -> None:
        report = copy.deepcopy(self.fixture.hypothesis["coverage_report"])
        available_claims = [
            claim for claim in self.fixture.claims if claim["subject"] != "quality"
        ]
        data_quality_ref = {
            "availability": "AVAILABLE",
            "schema_version": report["schema_version"],
            "analysis_id": report["collection_profile_id"],
            "analysis_digest": digest(report),
            "reason_codes": [],
        }
        recommendation = self.fixture.build(
            data_quality_analysis_ref=data_quality_ref,
            data_quality_analysis=report,
            claims=available_claims,
        )
        self.assertEqual(
            validate_collection_recommendation(
                recommendation,
                campaign_manifest=self.fixture.manifest,
                campaign_hypothesis=self.fixture.hypothesis,
                campaign_draft=self.fixture.draft,
                campaign_compilation_receipt=self.fixture.receipt,
                episode_evidence=self.fixture.evidence,
                data_quality_analysis=report,
            ),
            recommendation,
        )
        with self.assertRaisesRegex(ContractError, "ANALYSIS_ARTIFACT"):
            project_update_draft_intent(
                recommendation, selected_change_id="increase-count",
                operator_view=self.fixture.view(),
            )
        before = copy.deepcopy((recommendation, report))
        projected = project_update_draft_intent(
            recommendation, selected_change_id="increase-count",
            operator_view=self.fixture.view(), data_quality_analysis=report,
        )
        self.assertEqual(projected["payload"]["requested_count"], 3)
        self.assertEqual((recommendation, report), before)
        forged_report = copy.deepcopy(report)
        forged_report["collection_profile_id"] = "forged-quality"
        with self.assertRaises(ContractError):
            project_update_draft_intent(
                recommendation, selected_change_id="increase-count",
                operator_view=self.fixture.view(),
                data_quality_analysis=forged_report,
            )
        with self.assertRaisesRegex(ContractError, "ANALYSIS_ARTIFACT"):
            self.fixture.build(data_quality_analysis_ref=data_quality_ref)

        malformed = {
            "schema_version": "data_factory.coverage_report.v1",
            "collection_profile_id": "quality-r1",
            "authority": "REPORT_ONLY",
            "not_a_coverage_report": True,
        }
        malformed_ref = {
            "availability": "AVAILABLE",
            "schema_version": malformed["schema_version"],
            "analysis_id": malformed["collection_profile_id"],
            "analysis_digest": digest(malformed),
            "reason_codes": [],
        }
        with self.assertRaisesRegex(ContractError, "COVERAGE_REPORT_SCHEMA"):
            self.fixture.build(
                data_quality_analysis_ref=malformed_ref,
                data_quality_analysis=malformed,
            )

        invented_rollout = {
            "schema_version": "data_factory.invented_rollout_blob.v1",
            "analysis_id": "rollout-r1",
            "evidence_scope": "PHYSICAL",
        }
        invented_ref = {
            "availability": "AVAILABLE",
            "schema_version": invented_rollout["schema_version"],
            "analysis_id": invented_rollout["analysis_id"],
            "analysis_digest": digest(invented_rollout),
            "reason_codes": [],
        }
        with self.assertRaisesRegex(ContractError, "ROLLOUT_OWNER"):
            self.fixture.build(
                rollout_evidence_analysis_ref=invented_ref,
                rollout_evidence_analysis=invented_rollout,
            )

        invalid_unavailable = unavailable("MISSING")
        invalid_unavailable["analysis_id"] = "not-null"
        with self.assertRaisesRegex(ContractError, "ANALYSIS_UNAVAILABLE"):
            self.fixture.build(data_quality_analysis_ref=invalid_unavailable)
        with self.assertRaisesRegex(ContractError, "ANALYSIS_UNAVAILABLE"):
            self.fixture.build(
                rollout_evidence_analysis_ref=unavailable("ROLLOUT_NOT_RUN"),
            )

    def test_data_quality_availability_and_unknown_claim_are_coupled(self) -> None:
        missing_claims = [
            claim for claim in self.fixture.claims if claim["subject"] != "quality"
        ]
        with self.assertRaisesRegex(ContractError, "NUISANCE_CLAIM"):
            self.fixture.build(claims=missing_claims)

        report = copy.deepcopy(self.fixture.hypothesis["coverage_report"])
        data_quality_ref = {
            "availability": "AVAILABLE",
            "schema_version": report["schema_version"],
            "analysis_id": report["collection_profile_id"],
            "analysis_digest": digest(report),
            "reason_codes": [],
        }
        with self.assertRaisesRegex(ContractError, "NUISANCE_CLAIM"):
            self.fixture.build(
                data_quality_analysis_ref=data_quality_ref,
                data_quality_analysis=report,
            )

        before = copy.deepcopy((missing_claims, data_quality_ref, report))
        recommendation = self.fixture.build(
            data_quality_analysis_ref=data_quality_ref,
            data_quality_analysis=report,
            claims=missing_claims,
        )
        self.assertEqual(
            recommendation["input_snapshot"]["data_quality_analysis_ref"],
            data_quality_ref,
        )
        self.assertEqual(
            validate_collection_recommendation(
                recommendation,
                campaign_manifest=self.fixture.manifest,
                campaign_hypothesis=self.fixture.hypothesis,
                campaign_draft=self.fixture.draft,
                campaign_compilation_receipt=self.fixture.receipt,
                episode_evidence=self.fixture.evidence,
                data_quality_analysis=report,
            ),
            recommendation,
        )
        self.assertEqual((missing_claims, data_quality_ref, report), before)

    def test_claim_epistemics_nuisances_and_causal_rollout_fail_closed(self) -> None:
        cases = {}
        observed = copy.deepcopy(self.fixture.claims)
        next(claim for claim in observed if claim["class"] == "OBSERVED")["evidence_refs"] = []
        cases["observed-without-evidence"] = observed
        suggested = copy.deepcopy(self.fixture.claims)
        next(claim for claim in suggested if claim["class"] == "SUGGESTED")["basis_claim_ids"] = []
        cases["suggested-without-basis"] = suggested
        unknown = copy.deepcopy(self.fixture.claims)
        next(claim for claim in unknown if claim["class"] == "UNKNOWN")["value"] = "KNOWN"
        cases["unknown-with-value"] = unknown
        person = copy.deepcopy(self.fixture.claims)
        person_claim = next(claim for claim in person if claim["subject"] == "person")
        person_claim.update(
            {"class": "OBSERVED", "value": True,
             "evidence_refs": [self.fixture.manifest["manifest_digest"]],
             "reason_codes": []},
        )
        cases["person-assertion"] = person
        missing = copy.deepcopy(self.fixture.claims)
        missing[:] = [claim for claim in missing if claim["subject"] != "background"]
        cases["missing-background"] = missing
        duplicate = copy.deepcopy(self.fixture.claims)
        extra = copy.deepcopy(next(claim for claim in duplicate if claim["subject"] == "robot"))
        extra["claim_id"] = "robot-unknown-two"
        duplicate.append(extra)
        cases["duplicate-robot"] = duplicate
        rollout = copy.deepcopy(self.fixture.claims)
        rollout_claim = next(claim for claim in rollout if claim["subject"] == "rollout")
        rollout_claim.update(
            {"class": "OBSERVED", "value": "SUCCESS",
             "evidence_refs": [self.fixture.manifest["manifest_digest"]],
             "reason_codes": []},
        )
        cases["unsupported-rollout"] = rollout
        causal = copy.deepcopy(self.fixture.claims)
        causal_claim = next(claim for claim in causal if claim["class"] == "SUGGESTED")
        causal_claim["value"] = {"outcome": "WILL_IMPROVE"}
        cases["causal-benefit"] = causal
        observed_causal = copy.deepcopy(self.fixture.claims)
        observed_claim = next(
            claim for claim in observed_causal if claim["class"] == "OBSERVED"
        )
        observed_claim["value"] = {"benefit": "GUARANTEED"}
        cases["unsupported-observed-benefit"] = observed_causal
        for label, claims in cases.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                self.fixture.build(claims=claims)

    def test_closed_claim_values_and_subject_bound_limitations_reject_review_probes(self) -> None:
        causal = copy.deepcopy(self.fixture.claims)
        next(
            claim for claim in causal if claim["class"] == "SUGGESTED"
        )["value"] = {
            "prediction": "guaranteed higher task-completion rate",
        }
        with self.assertRaisesRegex(ContractError, "CLAIM_VALUE"):
            self.fixture.build(claims=causal)

        misplaced = copy.deepcopy(self.fixture.claims)
        robot = next(claim for claim in misplaced if claim["subject"] == "robot")
        robot.update({
            "class": "OBSERVED",
            "value": {
                "metric": "COLLECTED_EPISODE_COUNT",
                "count": 1,
            },
            "evidence_refs": [self.fixture.manifest["manifest_digest"]],
            "reason_codes": [],
        })
        background = next(
            claim for claim in misplaced if claim["subject"] == "background"
        )
        background["reason_codes"].append("ROBOT_VARIATION_UNMEASURED")
        with self.assertRaises(ContractError):
            self.fixture.build(claims=misplaced)

    def test_authority_unknown_fields_and_patch_allowlist_fail_closed(self) -> None:
        recommendation = self.fixture.build()
        self.assertEqual(recommendation["authority"], AUTHORITY)
        extra = copy.deepcopy(recommendation)
        extra["compile_draft"] = False
        with self.assertRaisesRegex(ContractError, "_FIELDS"):
            validate_collection_recommendation(extra)
        authority = copy.deepcopy(recommendation)
        authority["authority"]["plan_compile"] = True
        redigest(authority, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "_AUTHORITY"):
            validate_collection_recommendation(authority)
        integer_false = copy.deepcopy(recommendation)
        integer_false["authority"]["gate_bypass"] = 0
        redigest(integer_false, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "_AUTHORITY"):
            validate_collection_recommendation(integer_false)
        unknown_patch = copy.deepcopy(self.fixture.patches)
        unknown_patch[0]["field"] = "compile_draft"
        with self.assertRaisesRegex(ContractError, "PATCH_FIELD"):
            self.fixture.build(suggested_draft_patches=unknown_patch)
        multiple_fields = copy.deepcopy(self.fixture.patches)
        multiple_fields[0]["second_field"] = 4
        with self.assertRaisesRegex(ContractError, "PATCH_FIELDS"):
            self.fixture.build(suggested_draft_patches=multiple_fields)
        duplicate_episode = copy.deepcopy(recommendation)
        repeated = copy.deepcopy(
            duplicate_episode["input_snapshot"]["episodes"][0],
        )
        repeated["manifest_order_index"] = 2
        duplicate_episode["input_snapshot"]["episodes"].append(repeated)
        redigest(duplicate_episode["input_snapshot"], "snapshot_digest")
        redigest(duplicate_episode, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "EPISODE_DUPLICATE"):
            validate_collection_recommendation(duplicate_episode)

    def test_projector_returns_one_existing_intent_without_applying(self) -> None:
        view = self.fixture.view()
        before = copy.deepcopy((self.fixture.__dict__, view))
        with (
            mock.patch("builtins.open", side_effect=AssertionError("I/O forbidden")),
            mock.patch("pathlib.Path.open", side_effect=AssertionError("I/O forbidden")),
            mock.patch("pathlib.Path.resolve", side_effect=AssertionError("I/O forbidden")),
        ):
            recommendation = self.fixture.build()
            checked = validate_collection_recommendation(
                recommendation,
                campaign_manifest=self.fixture.manifest,
                campaign_hypothesis=self.fixture.hypothesis,
                campaign_draft=self.fixture.draft,
                campaign_compilation_receipt=self.fixture.receipt,
                episode_evidence=self.fixture.evidence,
            )
            intent = project_update_draft_intent(
                recommendation,
                selected_change_id="increase-count",
                operator_view=view,
            )
        self.assertEqual(checked, recommendation)
        self.assertEqual(intent, {
            "schema_version": "data_factory.operator_intent.v1",
            "intent_id": intent["intent_id"],
            "session_id": "session-r1",
            "view_revision": 9,
            "view_digest": view["view_digest"],
            "op": "update_draft",
            "payload": {"draft_id": "draft-r1", "requested_count": 3},
        })
        self.assertEqual((self.fixture.__dict__, view), before)
        self.assertEqual(len(intent["payload"]), 2)
        self.assertNotIn("compile_draft", intent)
        self.assertNotIn("authorize_campaign", intent)

    def test_projector_rejects_stale_non_authoring_unavailable_and_multiple_selection(self) -> None:
        recommendation = self.fixture.build()
        stale = self.fixture.view()
        stale["revision"] += 1
        non_authoring = self.fixture.view()
        non_authoring["projection"]["workflow_state"] = "REVIEW_CAMPAIGN"
        self._redigest_view(non_authoring)
        unavailable_op = self.fixture.view()
        unavailable_op["projection"]["available_ops"] = ["compile_draft"]
        self._redigest_view(unavailable_op)
        cases = (
            ("stale", stale, "increase-count"),
            ("non-authoring", non_authoring, "increase-count"),
            ("unavailable", unavailable_op, "increase-count"),
            ("unknown", self.fixture.view(), "unknown-change"),
            ("multiple", self.fixture.view(), ["increase-count", "use-ood-split"]),
        )
        for label, view, selection in cases:
            with self.subTest(label=label), self.assertRaises(ContractError):
                project_update_draft_intent(
                    recommendation,
                    selected_change_id=selection,
                    operator_view=view,
                )

    def test_selection_patch_is_checked_against_the_fresh_view(self) -> None:
        patches = [{
            "change_id": "select-task",
            "field": "selection",
            "value": {"task": "pickup_e2e"},
            "basis_claim_ids": ["coverage-observed"],
        }]
        recommendation = self.fixture.build(suggested_draft_patches=patches)
        intent = project_update_draft_intent(
            recommendation,
            selected_change_id="select-task",
            operator_view=self.fixture.view(),
            intent_id="selection-intent-r1",
        )
        self.assertEqual(intent["payload"]["selection"], {"task": "pickup_e2e"})
        blocked = self.fixture.view()
        patches[0]["value"] = {"task": "pick_place"}
        recommendation = self.fixture.build(suggested_draft_patches=patches)
        with self.assertRaisesRegex(ContractError, "PATCH_VALUE"):
            project_update_draft_intent(
                recommendation,
                selected_change_id="select-task",
                operator_view=blocked,
            )

        split_patch = [{
            "change_id": "select-split",
            "field": "split",
            "value": "OOD",
            "basis_claim_ids": ["coverage-observed"],
        }]
        recommendation = self.fixture.build(
            suggested_draft_patches=split_patch,
        )
        with self.assertRaisesRegex(ContractError, "PATCH_VALUE"):
            project_update_draft_intent(
                recommendation, selected_change_id="select-split",
                operator_view=self.fixture.view(),
            )
        split_patch[0]["value"] = "TRAIN"
        recommendation = self.fixture.build(
            suggested_draft_patches=split_patch,
        )
        intent = project_update_draft_intent(
            recommendation, selected_change_id="select-split",
            operator_view=self.fixture.view(),
        )
        self.assertEqual(intent["payload"]["split"], "TRAIN")

    def test_each_patch_kind_round_trips_through_the_canonical_owner(self) -> None:
        cases = (
            ("requested_count", 4),
            ("repeat", 2),
            ("split", "TRAIN"),
            ("selection", {"task": "pickup_e2e"}),
            (
                "state_space_design_factors",
                {"columns": 4, "rows": 2, "yaw_cdf_strata": 2},
            ),
        )
        for index, (field, value) in enumerate(cases, 1):
            patch = {
                "change_id": f"owner-round-trip-{index}",
                "field": field,
                "value": value,
                "basis_claim_ids": ["coverage-observed"],
            }
            recommendation = self.fixture.build(
                suggested_draft_patches=[patch],
            )
            application = self.application()
            try:
                view = application.bridge_core.snapshot()
                before = copy.deepcopy((recommendation, view))
                with (
                    mock.patch(
                        "builtins.open",
                        side_effect=AssertionError("I/O forbidden"),
                    ),
                    mock.patch(
                        "pathlib.Path.open",
                        side_effect=AssertionError("I/O forbidden"),
                    ),
                    mock.patch(
                        "pathlib.Path.resolve",
                        side_effect=AssertionError("I/O forbidden"),
                    ),
                ):
                    intent = project_update_draft_intent(
                        recommendation,
                        selected_change_id=patch["change_id"],
                        operator_view=view,
                    )
                self.assertEqual((recommendation, view), before)
                result = application.bridge_core.consume(intent)
                self.assertEqual(result["result"]["outcome"], "DRAFT_UPDATED")
                if field == "selection":
                    self.assertEqual(application.selection["task_id"], "pickup_e2e")
                elif field == "state_space_design_factors":
                    profile = application.draft["state_space_design_profile"]
                    self.assertEqual(profile["spatial_strata"], {
                        "columns": 4, "rows": 2,
                    })
                    self.assertEqual(profile["yaw_cdf_strata"], 2)
                else:
                    self.assertEqual(application.draft[field], value)
            finally:
                application.close()

    def test_frozen_available_dqa_selected_change_round_trip_and_stale_rejection(self) -> None:
        report = copy.deepcopy(self.fixture.hypothesis["coverage_report"])
        data_quality_ref = {
            "availability": "AVAILABLE",
            "schema_version": report["schema_version"],
            "analysis_id": report["collection_profile_id"],
            "analysis_digest": digest(report),
            "reason_codes": [],
        }
        claims = [
            claim for claim in self.fixture.claims if claim["subject"] != "quality"
        ]
        rollout_ref = unavailable("NO_CANONICAL_PHYSICAL_ROLLOUT_ANALYSIS")
        patches = [
            {
                "change_id": "adjust-repeat",
                "field": "repeat",
                "value": 2,
                "basis_claim_ids": ["coverage-observed"],
            },
            self.fixture.patches[0],
        ]
        recommendation = self.fixture.build(
            data_quality_analysis_ref=data_quality_ref,
            data_quality_analysis=report,
            rollout_evidence_analysis_ref=rollout_ref,
            claims=claims,
            suggested_draft_patches=patches,
        )
        self.assertEqual(recommendation["authority"], AUTHORITY)
        self.assertEqual(
            [patch["change_id"] for patch in recommendation["suggested_draft_patches"]],
            ["adjust-repeat", "increase-count"],
        )
        self.assertEqual(
            recommendation["input_snapshot"]["data_quality_analysis_ref"],
            data_quality_ref,
        )
        self.assertEqual(
            recommendation["input_snapshot"]["rollout_evidence_analysis_ref"],
            rollout_ref,
        )
        self.assertEqual(
            [claim["claim_id"] for claim in claims],
            [
                claim["claim_id"] for claim in self.fixture.claims
                if claim["claim_id"] != "quality-unknown"
            ],
        )

        campaign_factory = mock.Mock(
            side_effect=AssertionError("campaign factory was requested"),
        )
        application = self.application(campaign_factory=campaign_factory)
        try:
            core = application.bridge_core
            update_draft_handler = mock.Mock(wraps=core.handlers["update_draft"])
            core.handlers["update_draft"] = update_draft_handler
            view = core.snapshot()
            before = copy.deepcopy((recommendation, report, view))
            with (
                mock.patch("builtins.open", side_effect=AssertionError("I/O forbidden")),
                mock.patch("pathlib.Path.open", side_effect=AssertionError("I/O forbidden")),
                mock.patch("pathlib.Path.resolve", side_effect=AssertionError("I/O forbidden")),
            ):
                intent = project_update_draft_intent(
                    recommendation,
                    selected_change_id="increase-count",
                    operator_view=view,
                    intent_id="frozen-dqa-update-r1",
                    data_quality_analysis=report,
                )
            self.assertEqual((recommendation, report, view), before)
            self.assertEqual(intent["payload"], {
                "draft_id": application.draft["draft_id"], "requested_count": 3,
            })

            draft_before = copy.deepcopy(application.draft)
            result = core.consume(intent)
            draft_after = copy.deepcopy(application.draft)
            self.assertEqual(result["result"]["outcome"], "DRAFT_UPDATED")
            expected_draft = copy.deepcopy(draft_before)
            expected_draft.update(
                requested_count=3, revision=draft_before["revision"] + 1,
            )
            self.assertEqual(draft_after, expected_draft)
            update_draft_handler.assert_called_once()
            updated_view = core.snapshot()
            self.assertEqual(updated_view["revision"], view["revision"] + 1)
            expected_projection = copy.deepcopy(view["projection"])
            expected_projection["draft"].update(
                requested_count=3, revision=draft_after["revision"],
            )
            self.assertEqual(updated_view["projection"], expected_projection)
            campaign_factory.assert_not_called()

            fresh_view = core.snapshot()
            stale = project_update_draft_intent(
                recommendation,
                selected_change_id="increase-count",
                operator_view=fresh_view,
                intent_id="frozen-dqa-stale-r1",
                data_quality_analysis=report,
            )
            core.transition(lambda: None)
            stale_before = copy.deepcopy((application.draft, core.snapshot()))
            with self.assertRaisesRegex(ContractError, "OPERATOR_INTENT_STALE_VIEW"):
                core.consume(stale)
            self.assertEqual(application.draft, stale_before[0])
            self.assertEqual(
                {key: value for key, value in core.snapshot().items()
                 if key != "generated_at"},
                {key: value for key, value in stale_before[1].items()
                 if key != "generated_at"},
            )
            update_draft_handler.assert_called_once()
            campaign_factory.assert_not_called()
        finally:
            application.close()

    def test_existing_compare_and_swap_rejects_a_once_fresh_intent_after_transition(self) -> None:
        state = {
            "workflow_state": "AUTHORING",
            "available_ops": ["update_draft"],
            "draft": {
                "draft_id": "draft-r1", "revision": 0,
                "requested_count": 2,
            },
        }
        core = OperatorIntentCore(
            session_id="session-r1",
            projection_call=lambda: state,
            handlers={"update_draft": lambda payload, _view: payload},
            clock=lambda: NOW,
        )
        old_view = core.snapshot()
        projected = project_update_draft_intent(
            self.fixture.build(),
            selected_change_id="increase-count",
            operator_view=old_view,
        )
        core.transition(lambda: state["draft"].update(revision=1))
        with self.assertRaisesRegex(ContractError, "OPERATOR_INTENT_STALE_VIEW"):
            core.consume(projected)

    @staticmethod
    def _redigest_view(view: dict) -> None:
        view["view_digest"] = digest({
            "session_id": view["session_id"],
            "revision": view["revision"],
            "projection": view["projection"],
        })


if __name__ == "__main__":
    unittest.main()
