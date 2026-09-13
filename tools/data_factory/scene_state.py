#!/usr/bin/env python3
"""Small runtime object-state store for episode-to-episode replanning."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.data_factory.cell_state import CellStateStore
from tools.data_factory_recovery import RecoveryError, decode_json_strict, write_json_atomic
from tools.fr5_data_factory import ContractArgumentParser, ContractError, DIGEST, RFC3339, SAFE_ID, canonical_digest


SCHEMA_V1 = "data_factory.scene_state.v1"
SCHEMA_VERSION = "data_factory.scene_state.v2"
SCENE_V1_KEYS = {"schema_version", "robot_system_id", "revision", "objects", "updated_at"}
SCENE_KEYS = SCENE_V1_KEYS | {"slot_allocations"}
OBJECT_KEYS = {"instance_id", "object_profile_id", "state", "pose", "source", "updated_by", "updated_at"}
POSE_KEYS = {"place_id", "yaw_deg", "x_mm", "y_mm"}
OBJECT_STATES = {"ON_SURFACE", "HELD", "UNKNOWN"}
SOURCES = {
    "HUMAN", "AI", "ROBOT_ACTION", "ROBOT_RELEASE",
    "ROBOT_RELEASE_PROXY", "PERCEPTION",
}
SCENE_BINDING_KEYS = {"scene_state_digest", "revision", "object_instance_id"}
RELEASE_SLOT_KEYS = {"slot_id", "robot_system_id", "pose", "object_profile_id", "exclusion_geometry_digest", "role"}
SOURCE_SLOT_KEYS = {"slot_id", "slot_digest", "allowed_run_id"}
SLOT_KEYS = {"state", "role", "allowed_run_id", "evidence_run_id", "evidence_plan_digest", "evidence_digest", "updated_at"}
SLOT_STATES = {"AVAILABLE", "RESERVED", "LANDED_FOR_NEXT_SOURCE", "CONSUMED_PENDING_REVIEW", "QUARANTINED"}
SLOT_ROLES = {"PICK_SOURCE", "RELEASE_DESTINATION", "DESTINATION_THEN_NEXT_SOURCE"}
RELEASE_EVIDENCE_KEYS = {
    "schema_version", "run_id", "plan_digest", "release_slot_id",
    "expected_scene_state_digest", "expected_scene_revision",
    "gripper_reference_m", "gripper_feedback_m", "terminal_phases",
    "post_retreat_snapshot_digest", "next_start_tolerance_rad", "human_verdict",
}
RELEASE_EVIDENCE_V2_KEYS = {
    "schema_version", "run_id", "plan_digest", "release_slot_id",
    "expected_scene_state_digest", "expected_scene_revision",
    "gripper_reference_m", "gripper_feedback_m", "terminal_phases",
    "post_retreat_snapshot_digest", "next_start_tolerance_rad",
    "release_outcome", "outcome_source", "decided_by", "decided_at",
}


def _id(value: object, code: str) -> str:
    if not isinstance(value, str) or value in {".", ".."} or not SAFE_ID.fullmatch(value):
        raise ContractError(code)
    return value


def _number(value: object, code: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(code)
    number = float(value)
    return int(number) if number.is_integer() else number


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not RFC3339.fullmatch(value):
        raise ContractError("SCENE_TIMESTAMP")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("SCENE_TIMESTAMP") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError("SCENE_TIMESTAMP")
    normalized = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if value != normalized:
        raise ContractError("SCENE_TIMESTAMP")
    return normalized


def _pose(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != POSE_KEYS:
        raise ContractError("SCENE_POSE")
    return {
        "place_id": _id(value["place_id"], "SCENE_POSE"),
        **{key: _number(value[key], "SCENE_POSE") for key in ("yaw_deg", "x_mm", "y_mm")},
    }


def release_slot(*, robot_system_id: str, pose: dict, object_profile_id: str, exclusion_geometry_digest: str, role: str = "RELEASE_DESTINATION") -> dict:
    """Build the one canonical same-slot recycle binding used by P4.5."""
    robot_system_id = _id(robot_system_id, "SCENE_SLOT")
    pose = _pose(pose)
    object_profile_id = _id(object_profile_id, "SCENE_SLOT")
    if not isinstance(exclusion_geometry_digest, str) or not DIGEST.fullmatch(exclusion_geometry_digest) or role not in SLOT_ROLES:
        raise ContractError("SCENE_SLOT")
    identity = {
        "robot_system_id": robot_system_id,
        **pose,
        "object_profile_id": object_profile_id,
        "exclusion_geometry_digest": exclusion_geometry_digest,
    }
    return {
        "slot_id": canonical_digest(identity),
        "robot_system_id": robot_system_id,
        "pose": pose,
        "object_profile_id": object_profile_id,
        "exclusion_geometry_digest": exclusion_geometry_digest,
        "role": role,
    }


def validate_release_slot(value: object, robot_system_id: str) -> dict:
    if not isinstance(value, dict) or set(value) != RELEASE_SLOT_KEYS:
        raise ContractError("SCENE_SLOT")
    normalized = release_slot(
        robot_system_id=robot_system_id,
        pose=value["pose"],
        object_profile_id=value["object_profile_id"],
        exclusion_geometry_digest=value["exclusion_geometry_digest"],
        role=value["role"],
    )
    if value["slot_id"] != normalized["slot_id"]:
        raise ContractError("SCENE_SLOT")
    return normalized


def _validate(value: object, robot_system_id: str) -> dict:
    if not isinstance(value, dict) or frozenset(value) not in {frozenset(SCENE_V1_KEYS), frozenset(SCENE_KEYS)}:
        raise ContractError("SCENE_SCHEMA")
    if value["schema_version"] not in {SCHEMA_V1, SCHEMA_VERSION} or value["robot_system_id"] != robot_system_id:
        raise ContractError("SCENE_SCHEMA")
    if (
        value["schema_version"] == SCHEMA_V1 and set(value) != SCENE_V1_KEYS
        or value["schema_version"] == SCHEMA_VERSION and set(value) != SCENE_KEYS
    ):
        raise ContractError("SCENE_SCHEMA")
    if type(value["revision"]) is not int or value["revision"] < 0 or not isinstance(value["objects"], dict):
        raise ContractError("SCENE_SCHEMA")
    _timestamp(value["updated_at"])
    for instance_id, item in value["objects"].items():
        if not isinstance(item, dict) or set(item) != OBJECT_KEYS or instance_id != item.get("instance_id"):
            raise ContractError("SCENE_OBJECT")
        _id(instance_id, "SCENE_OBJECT")
        _id(item["object_profile_id"], "SCENE_OBJECT")
        _id(item["updated_by"], "SCENE_OBJECT")
        if item["state"] not in OBJECT_STATES or item["source"] not in SOURCES:
            raise ContractError("SCENE_OBJECT")
        _timestamp(item["updated_at"])
        if item["state"] == "ON_SURFACE":
            _pose(item["pose"])
        elif item["pose"] is not None:
            raise ContractError("SCENE_POSE")
    slots = value.get("slot_allocations", {})
    if not isinstance(slots, dict):
        raise ContractError("SCENE_SLOT")
    for slot_id, item in slots.items():
        if not isinstance(slot_id, str) or not DIGEST.fullmatch(slot_id) or not isinstance(item, dict) or set(item) != SLOT_KEYS:
            raise ContractError("SCENE_SLOT")
        if item["state"] not in SLOT_STATES or item["role"] not in SLOT_ROLES:
            raise ContractError("SCENE_SLOT")
        for key in ("allowed_run_id", "evidence_run_id"):
            _id(item[key], "SCENE_SLOT")
        for key in ("evidence_plan_digest", "evidence_digest"):
            if not isinstance(item[key], str) or not DIGEST.fullmatch(item[key]):
                raise ContractError("SCENE_SLOT")
        _timestamp(item["updated_at"])
    return value


def validate_scene_binding(value: object) -> dict:
    if not isinstance(value, dict) or frozenset(value) not in {
        frozenset(SCENE_BINDING_KEYS), frozenset(SCENE_BINDING_KEYS | {"release_slot"}),
        frozenset(SCENE_BINDING_KEYS | {"release_slot", "allowed_next_run_id"}),
        frozenset(SCENE_BINDING_KEYS | {"release_slot", "source_slot"}),
        frozenset(SCENE_BINDING_KEYS | {
            "release_slot", "allowed_next_run_id", "source_slot",
        }),
    }:
        raise ContractError("SCENE_BINDING")
    if not isinstance(value["scene_state_digest"], str) or not DIGEST.fullmatch(value["scene_state_digest"]):
        raise ContractError("SCENE_BINDING")
    if type(value["revision"]) is not int or value["revision"] < 0:
        raise ContractError("SCENE_BINDING")
    _id(value["object_instance_id"], "SCENE_BINDING")
    result = dict(value)
    if "release_slot" in value:
        if not isinstance(value["release_slot"], dict):
            raise ContractError("SCENE_BINDING")
        result["release_slot"] = validate_release_slot(value["release_slot"], value["release_slot"].get("robot_system_id"))
    if "allowed_next_run_id" in value:
        result["allowed_next_run_id"] = _id(value["allowed_next_run_id"], "SCENE_SLOT_NEXT_RUN")
        if result["release_slot"]["role"] != "DESTINATION_THEN_NEXT_SOURCE":
            raise ContractError("SCENE_SLOT_NEXT_RUN")
    if "source_slot" in value:
        source = value["source_slot"]
        if not isinstance(source, dict) or set(source) != SOURCE_SLOT_KEYS:
            raise ContractError("SCENE_SLOT_NEXT_RUN")
        for key in ("slot_id", "slot_digest"):
            if not isinstance(source[key], str) or not DIGEST.fullmatch(source[key]):
                raise ContractError("SCENE_SLOT_NEXT_RUN")
        _id(source["allowed_run_id"], "SCENE_SLOT_NEXT_RUN")
        result["source_slot"] = dict(source)
    return result


class SceneStateStore:
    def __init__(self, root: Path | str, robot_system_id: str) -> None:
        self._cell = CellStateStore(root, robot_system_id)
        self.robot_system_id = self._cell.robot_system_id

    def _path(self, *, create: bool = False) -> Path:
        return self._cell.runtime_path("scene_state.json", create_robot=create)

    def recover_initial_contact_rejection(
        self, *, prior_confirmation: dict, lifecycle_result: dict,
        preapproval_evidence: dict, expected_cell: dict,
    ) -> dict:
        """Correct the initial CONTACT_PROFILE_UNAVAILABLE bookkeeping fault.

        Trusted retained executor evidence, not a public operator assertion.
        Caller must have ended the lifecycle owner and frozen its retained
        inputs; this is not recovery concurrent with a live executor or writer.
        The source-bound prepare failure precedes the first start_phase call;
        empty traces or zero gripper generations alone never establish this.
        This preserves an old observation, not a new human acknowledgement.
        """
        from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
        code = "SCENE_NO_DISPATCH_RECOVERY"
        failure = "CONTACT_PROFILE_UNAVAILABLE"
        prior_confirmation, result, preapproval, expected_cell = copy.deepcopy(
            (prior_confirmation, lifecycle_result, preapproval_evidence, expected_cell))
        try:
            diagnostic = build_run_diagnostic(result)
            plan = result["plan_envelope"]["plan"]
            evidence = result["execution_evidence"]
            trace = diagnostic["execution_trace"]
            binding = validate_scene_binding(plan["scene_binding"])
            prior = _validate(prior_confirmation["after_scene"]["scene_state"], self.robot_system_id)
            prior_cell = self._cell._validate(prior_confirmation["after_cell"])
            failed = _validate(evidence["scene_transition"]["scene_state"], self.robot_system_id)
            self._cell._validate(expected_cell)
            item = prior["objects"][binding["object_instance_id"]]
            initial = plan["steps"][0]["initial_hardware_binding"]
            hardware = evidence["snapshot"]["gripper_controller"]["hardware_execution"]
            wire = hardware["wire"]
            generation = plan["learned_proposal"]["generation_context"]
            if (result["code"] != failure or evidence["failure_code"] != failure
                    or result["state"] != "ABORTED" or result["executor_state"] != "BLOCKED"
                    or result.get("cancel_error") is not None or evidence.get("cancel_error") is not None
                    or evidence["step_index"] != 0 or evidence.get("learned_history", []) != []
                    or "mechanical_terminal" in evidence or "task_handoff" in evidence or "task_handoff" in result
                    or evidence["prospective_contact"] != {"status": "UNAVAILABLE", "code": failure}
                    or trace["status"] != "FAILED" or trace["failure_code"] != failure
                    or trace.get("segments", []) != [] or trace["terminal_phases"] != []
                    or trace["terminal_state"] is not None
                    or generation["predecessor_plan_digest"] is not None
                    or plan["robot_system_id"] != self.robot_system_id
                    or result["scene_binding"] != binding or "source_slot" in binding
                    or preapproval["run_id"] != result["run_id"]
                    or preapproval["plan_digest"] != result["plan_digest"]
                    or preapproval["plan_envelope"]["plan"] != plan
                    or preapproval["plan_envelope_digest"] != canonical_digest(preapproval["plan_envelope"])
                    or prior_confirmation["after_scene"]["scene_state_digest"] != canonical_digest(prior)
                    or binding["scene_state_digest"] != canonical_digest(prior)
                    or binding["revision"] != prior["revision"]
                    or item["state"] != "ON_SURFACE" or item["source"] != "HUMAN"
                    or not prior_cell["cell_ready"] or prior_cell["reason_code"] != "HUMAN_ACKNOWLEDGED"
                    or expected_cell["cell_ready"] or expected_cell["reason_code"] != failure
                    or expected_cell["run_id"] != result["run_id"]
                    or expected_cell["plan_digest"] != result["plan_digest"]
                    or evidence["scene_transition"]["scene_state_digest"] != canonical_digest(failed)
                    or wire["version"] != 5 or initial["generation"] != 0
                    or [wire[f"incarnation_{i}"] for i in range(4)] != initial["incarnation"]
                    or hardware["clock_binding"] != plan["learned_proposal"]["runtime_inputs"]["clock_binding"]
                    or any(wire[key] != 0 for key in ("generation", "active_generation", "completed_generation",
                           "selected_generation", "selected_valid", "error", "device_main_error", "device_sub_error"))):
                raise ContractError(code)
            # Only the executor's exact UNKNOWN write may have changed Scene.
            expected_failed = copy.deepcopy(prior)
            stamp = failed["updated_at"]
            expected_failed.update(revision=prior["revision"] + 1, updated_at=stamp)
            expected_failed["objects"][item["instance_id"]] = {
                **item, "state": "UNKNOWN", "pose": None, "source": "ROBOT_ACTION",
                "updated_by": "pickup-executor", "updated_at": stamp,
            }
            phase_path = Path(evidence["phase_events_path"])
            if (failed != expected_failed or phase_path.name != "phase_events.jsonl"
                    or phase_path.parent.name != result["run_id"] or phase_path.read_bytes() != b""):
                raise ContractError(code)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ContractError(code) from exc

        # Same Scene -> Cell lock order as update_object. Publish the receipt
        # before either correction, and readiness last: partial I/O fails closed.
        scene_fd = os.open(self._cell.runtime_path("scene_state.lock", create_robot=True), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        cell_fd = None
        try:
            self._flock(scene_fd, False)
            cell_fd = os.open(self._cell.runtime_path("state.lock", create_robot=True), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            self._flock(cell_fd, False)
            if self.read() != failed or self._cell.read() != expected_cell:
                raise ContractError("STATE_CHANGED")
            receipt_path = self._cell.runtime_path(f"no_dispatch_recovery-{result['run_id']}.json")
            if receipt_path.exists():
                raise ContractError("SCENE_RECOVERY_ALREADY_RETAINED")
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            restored = {**prior, "revision": failed["revision"] + 1, "updated_at": now}
            ready = {**prior_cell, "reason_code": "NO_DISPATCH_CONTINUITY", "updated_at": now,
                     "run_id": result["run_id"], "plan_digest": result["plan_digest"]}
            receipt = {"schema_version": "data_factory.no_dispatch_recovery.v1",
                       "basis": "INITIAL_CONTACT_PROFILE_REJECTION_BEFORE_START_PHASE",
                       "prior_confirmation_digest": canonical_digest(prior_confirmation),
                       "lifecycle_result_digest": diagnostic["lifecycle_result_digest"],
                       "preapproval_evidence_digest": canonical_digest(preapproval),
                       "before_scene": failed, "before_cell": expected_cell,
                       "after_scene": restored, "after_cell": ready,
                       "new_human_confirmation": False, "execution_authorized": False}
            receipt["receipt_digest"] = canonical_digest(receipt)
            write_json_atomic(receipt_path, receipt)
            write_json_atomic(self._path(create=True), _validate(restored, self.robot_system_id))
            write_json_atomic(self._cell.runtime_path("state.json", create_robot=True), self._cell._validate(ready))
            return receipt
        finally:
            if cell_fd is not None:
                os.close(cell_fd)
            os.close(scene_fd)

    def read(self) -> dict:
        path = self._path()
        if not path.exists():
            return {"schema_version": SCHEMA_V1, "robot_system_id": self.robot_system_id, "revision": 0, "objects": {}, "updated_at": "1970-01-01T00:00:00Z"}
        if not path.is_file() or path.is_symlink():
            raise ContractError("SCENE_PATH")
        try:
            return _validate(decode_json_strict(path.read_text(encoding="utf-8"), "SCENE_JSON", path), self.robot_system_id)
        except (OSError, RecoveryError) as exc:
            raise ContractError("SCENE_JSON", str(exc)) from exc

    def snapshot(self) -> dict:
        scene = self.read()
        return {"scene_state": scene, "scene_state_digest": canonical_digest(scene)}

    @staticmethod
    def _flock(descriptor, blocking):
        if type(blocking) is not bool:
            raise ContractError("SCENE_LOCK_MODE")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ContractError("SCENE_STATE_BUSY") from exc

    @contextmanager
    def locked_snapshot(self, expected_digest: str, *, blocking: bool = True):
        if not isinstance(expected_digest, str) or not DIGEST.fullmatch(expected_digest):
            raise ContractError("SCENE_BINDING")
        lock_path = self._cell.runtime_path("scene_state.lock", create_robot=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            self._flock(descriptor, blocking)
            snapshot = self.snapshot()
            if snapshot["scene_state_digest"] != expected_digest:
                raise ContractError("SCENE_STATE_CHANGED")
            yield snapshot
        finally:
            os.close(descriptor)

    def transition_release(
        self,
        *,
        instance_id: str,
        release_slot: dict,
        evidence: dict,
        updated_by: str,
        expected_digest: str,
        expected_revision: int,
        allowed_next_run_id: str | None = None,
        parent_cell_binding: dict | None = None,
        blocking: bool = True,
    ) -> dict:
        """Publish the physical object and its slot in one scene-v2 revision."""
        instance_id = _id(instance_id, "SCENE_OBJECT")
        updated_by = _id(updated_by, "SCENE_OBJECT")
        release_slot = validate_release_slot(release_slot, self.robot_system_id)
        if not isinstance(expected_digest, str) or not DIGEST.fullmatch(expected_digest) or type(expected_revision) is not int or expected_revision < 0:
            raise ContractError("SCENE_BINDING")
        if not isinstance(evidence, dict):
            raise ContractError("RELEASE_EVIDENCE")
        if parent_cell_binding is not None:
            if not isinstance(parent_cell_binding, dict) or set(parent_cell_binding) != {"run_id", "plan_digest", "source_slot_id", "source_slot_digest"}:
                raise ContractError("SCENE_PARENT_CELL")
            _id(parent_cell_binding["run_id"], "SCENE_PARENT_CELL")
            if parent_cell_binding["run_id"] == evidence.get("run_id") or any(
                not isinstance(parent_cell_binding[key], str) or not DIGEST.fullmatch(parent_cell_binding[key])
                for key in ("plan_digest", "source_slot_id", "source_slot_digest")
            ):
                raise ContractError("SCENE_PARENT_CELL")
        chain = release_slot["role"] == "DESTINATION_THEN_NEXT_SOURCE"
        if chain:
            allowed_next_run_id = _id(allowed_next_run_id, "SCENE_SLOT_NEXT_RUN")
            if allowed_next_run_id == evidence.get("run_id"):
                raise ContractError("SCENE_SLOT_NEXT_RUN")
        elif allowed_next_run_id is not None:
            raise ContractError("SCENE_SLOT_NEXT_RUN")
        expected_terminals = ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"]
        schema = evidence.get("schema_version")
        if schema == "data_factory.recycle_release_evidence.v1":
            if set(evidence) != RELEASE_EVIDENCE_KEYS:
                raise ContractError("RELEASE_EVIDENCE")
            outcome = evidence.get("human_verdict")
            outcome_source = "HUMAN_TTY"
        elif schema == "data_factory.recycle_release_evidence.v2":
            if set(evidence) != RELEASE_EVIDENCE_V2_KEYS:
                raise ContractError("RELEASE_EVIDENCE")
            outcome = evidence.get("release_outcome")
            outcome_source = evidence.get("outcome_source")
            if (
                outcome_source not in {
                    "HUMAN_TTY", "LOCAL_UI_BUTTON",
                    "CAMPAIGN_CONTROL_PROXY", "EXECUTOR_FAILURE",
                }
                or outcome not in {
                    "LANDED", "EXPECTED_LANDED", "OFF_SLOT", "UNCERTAIN",
                }
                or outcome_source == "CAMPAIGN_CONTROL_PROXY"
                and outcome != "EXPECTED_LANDED"
                or outcome_source == "EXECUTOR_FAILURE" and outcome != "UNCERTAIN"
                or outcome_source in {"HUMAN_TTY", "LOCAL_UI_BUTTON"}
                and outcome == "EXPECTED_LANDED"
            ):
                raise ContractError("RELEASE_EVIDENCE")
            _id(evidence.get("decided_by"), "RELEASE_EVIDENCE")
            _timestamp(evidence.get("decided_at"))
        else:
            raise ContractError("RELEASE_EVIDENCE")
        if outcome not in {"LANDED", "EXPECTED_LANDED", "OFF_SLOT", "UNCERTAIN"}:
            raise ContractError("RELEASE_EVIDENCE")
        landed = outcome in {"LANDED", "EXPECTED_LANDED"}
        if (
            evidence["release_slot_id"] != release_slot["slot_id"]
            or evidence["expected_scene_state_digest"] != expected_digest
            or evidence["expected_scene_revision"] != expected_revision
            or not isinstance(evidence["terminal_phases"], list)
            or evidence["terminal_phases"] != expected_terminals[:len(evidence["terminal_phases"])]
            or landed and evidence["terminal_phases"] != expected_terminals
        ):
            raise ContractError("RELEASE_EVIDENCE")
        _id(evidence["run_id"], "RELEASE_EVIDENCE")
        for key in ("plan_digest", "post_retreat_snapshot_digest"):
            if not isinstance(evidence[key], str) or not DIGEST.fullmatch(evidence[key]):
                raise ContractError("RELEASE_EVIDENCE")
        for key in ("gripper_reference_m", "gripper_feedback_m"):
            if evidence[key] is None and not landed:
                continue
            if _number(evidence[key], "RELEASE_EVIDENCE") < 0:
                raise ContractError("RELEASE_EVIDENCE")
        if _number(evidence["next_start_tolerance_rad"], "RELEASE_EVIDENCE") < 0:
            raise ContractError("RELEASE_EVIDENCE")

        lock_path = self._cell.runtime_path("scene_state.lock", create_robot=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        cell_descriptor = None
        try:
            self._flock(descriptor, blocking)
            current = self.read()
            if canonical_digest(current) != expected_digest or current["revision"] != expected_revision:
                raise ContractError("SCENE_STATE_CHANGED")
            item = current["objects"].get(instance_id)
            if not isinstance(item, dict) or item.get("object_profile_id") != release_slot["object_profile_id"]:
                raise ContractError("SCENE_OBJECT_NOT_READY")
            slots = dict(current.get("slot_allocations", {}))
            if parent_cell_binding is not None:
                cell_descriptor = os.open(self._cell.runtime_path("state.lock", create_robot=True),
                                          os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                self._flock(cell_descriptor, blocking)
                parent = self._cell.read()
                source = slots.get(parent_cell_binding["source_slot_id"])
                if (parent["cell_ready"] is not False
                        or (parent["run_id"], parent["plan_digest"]) != (parent_cell_binding["run_id"], parent_cell_binding["plan_digest"])
                        or not isinstance(source, dict)
                        or canonical_digest(source) != parent_cell_binding["source_slot_digest"]
                        or (source["state"], source["role"], source["allowed_run_id"])
                        != ("CONSUMED_PENDING_REVIEW", "DESTINATION_THEN_NEXT_SOURCE", evidence["run_id"])):
                    raise ContractError("SCENE_PARENT_CELL")
            prior_slot = slots.get(release_slot["slot_id"])
            if prior_slot is not None and prior_slot.get("state") != "AVAILABLE":
                consumed_here_by_this_run = (
                    prior_slot.get("state") == "CONSUMED_PENDING_REVIEW"
                    and prior_slot.get("role") == "DESTINATION_THEN_NEXT_SOURCE"
                    and prior_slot.get("allowed_run_id") == evidence["run_id"]
                )
                if not consumed_here_by_this_run:
                    raise ContractError("SCENE_SLOT_UNAVAILABLE")

            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            evidence_digest = canonical_digest(evidence)
            cell = self._cell.read()
            if landed and (parent_cell_binding is not None or (
                    cell["reason_code"] == "EXECUTION_IN_PROGRESS"
                    and (cell["run_id"], cell["plan_digest"]) == (evidence["run_id"], evidence["plan_digest"]))):
                # Only this execution's confirmed landing vacates its consumed
                # source. Review evidence remains immutable in the source run.
                for slot_id, allocation in slots.items():
                    if (slot_id != release_slot["slot_id"]
                            and (parent_cell_binding is None or slot_id == parent_cell_binding["source_slot_id"])
                            and allocation["state"] == "CONSUMED_PENDING_REVIEW"
                            and allocation["role"] == "DESTINATION_THEN_NEXT_SOURCE"
                            and allocation["allowed_run_id"] == evidence["run_id"]):
                        slots[slot_id] = {**allocation, "state": "AVAILABLE"}
            objects = dict(current["objects"])
            objects[instance_id] = {
                "instance_id": instance_id,
                "object_profile_id": release_slot["object_profile_id"],
                "state": "ON_SURFACE" if landed else "UNKNOWN",
                "pose": release_slot["pose"] if landed else None,
                "source": (
                    "ROBOT_RELEASE_PROXY"
                    if outcome_source == "CAMPAIGN_CONTROL_PROXY" and landed
                    else "ROBOT_RELEASE" if landed else "ROBOT_ACTION"
                ),
                "updated_by": updated_by,
                "updated_at": now,
            }
            slots[release_slot["slot_id"]] = {
                "state": "LANDED_FOR_NEXT_SOURCE" if landed and chain else "CONSUMED_PENDING_REVIEW" if landed else "QUARANTINED",
                "role": release_slot["role"],
                "allowed_run_id": allowed_next_run_id if chain else evidence["run_id"],
                "evidence_run_id": evidence["run_id"],
                "evidence_plan_digest": evidence["plan_digest"],
                "evidence_digest": evidence_digest,
                "updated_at": now,
            }
            scene = _validate({
                "schema_version": SCHEMA_VERSION,
                "robot_system_id": self.robot_system_id,
                "revision": current["revision"] + 1,
                "objects": objects,
                "slot_allocations": slots,
                "updated_at": now,
            }, self.robot_system_id)
            write_json_atomic(self._path(create=True), scene)
            return {
                "scene_state": scene,
                "scene_state_digest": canonical_digest(scene),
                "release_evidence_digest": evidence_digest,
            }
        finally:
            if cell_descriptor is not None:
                os.close(cell_descriptor)
            os.close(descriptor)

    def object_position(self, *, object_profile_id, dimensions_mm):
        """A release is an execution-derived expectation, never a visual measurement.

        Cell arming precedes motion. A later, different run therefore invalidates
        reuse even if an interrupted executor could not publish UNKNOWN to scene.
        Cell readiness itself remains the lifecycle owner's independent gate.
        """
        store = self
        robot_system_id = self.robot_system_id
        snapshot = store.snapshot()
        cell = self._cell.read()
        if store.snapshot() != snapshot:
            raise ContractError("SCENE_STATE_CHANGED")
        scene = snapshot["scene_state"]
        result = {
            "status": "MISSING", "reason": "SCENE_OBJECT_NOT_READY",
            "source": None, "pose": None, "object_instance_id": None,
            "scene_state_digest": snapshot["scene_state_digest"],
            "scene_revision": scene["revision"],
            "cell_state_digest": canonical_digest(cell),
            "slot_id": None, "slot_digest": None,
            "run_id": None, "plan_digest": None, "evidence_digest": None,
        }
        objects = [item for item in scene["objects"].values()
                   if item["object_profile_id"] == object_profile_id]
        if len(objects) > 1:
            result.update(status="BLOCKED", reason="SCENE_OBJECT_AMBIGUOUS")
        elif objects:
            item = objects[0]
            cell_precedes_position = datetime.fromisoformat(cell["updated_at"].replace("Z", "+00:00")) <= datetime.fromisoformat(item["updated_at"].replace("Z", "+00:00"))
            result.update(status="BLOCKED", reason="SCENE_OBJECT_NOT_READY",
                          source=item["source"], pose=item["pose"],
                          object_instance_id=item["instance_id"])
            if item["state"] == "ON_SURFACE" and item["source"] == "HUMAN":
                if cell_precedes_position or cell["cell_ready"]:
                    result.update(status="AVAILABLE", reason=None)
                else:
                    result["reason"] = "OBJECT_POSITION_NEWER_EXECUTION"
            elif item["state"] == "ON_SURFACE" and item["source"] in {
                "ROBOT_RELEASE", "ROBOT_RELEASE_PROXY",
            }:
                slot_id = release_slot(
                    robot_system_id=robot_system_id, pose=item["pose"],
                    object_profile_id=object_profile_id,
                    exclusion_geometry_digest=canonical_digest({
                        "shape": "BOX", "dimensions_mm": dimensions_mm,
                    }),
                )["slot_id"]
                slot = scene.get("slot_allocations", {}).get(slot_id)
                result["reason"] = "OBJECT_POSITION_RELEASE_LINEAGE"
                if slot is not None:
                    result.update(slot_id=slot_id, slot_digest=canonical_digest(slot),
                                  run_id=slot["evidence_run_id"],
                                  plan_digest=slot["evidence_plan_digest"],
                                  evidence_digest=slot["evidence_digest"])
                    if (slot["state"] in {"LANDED_FOR_NEXT_SOURCE", "CONSUMED_PENDING_REVIEW"}
                            and slot["updated_at"] == item["updated_at"]):
                        # A motion-only child leaves cell ownership with its
                        # parent. Confirmed release atomically vacates that
                        # parent's source, retaining the exact child binding.
                        # Reuse this durable relation after parent completion;
                        # READY alone or a run-name prefix is not evidence.
                        parent_sources = [source for source_id, source in scene["slot_allocations"].items()
                            if source_id != slot_id
                            and source["state"] == "AVAILABLE"
                            and source["role"] == "DESTINATION_THEN_NEXT_SOURCE"
                            and source["allowed_run_id"] == slot["evidence_run_id"]
                            and (source["evidence_run_id"], source["evidence_plan_digest"])
                            == (cell["run_id"], cell["plan_digest"])
                            and datetime.fromisoformat(source["updated_at"].replace("Z", "+00:00"))
                            < datetime.fromisoformat(slot["updated_at"].replace("Z", "+00:00"))]
                        if (cell_precedes_position
                                or (cell["run_id"], cell["plan_digest"])
                                == (slot["evidence_run_id"], slot["evidence_plan_digest"])
                                or cell["cell_ready"] and len(parent_sources) == 1):
                            result.update(status="AVAILABLE", reason=None)
                        else:
                            result["reason"] = "OBJECT_POSITION_NEWER_EXECUTION"
        result["binding_digest"] = canonical_digest(result)
        return result

    def rebind_landed_source(
        self, *, object_profile_id, dimensions_mm, run_id,
        expected_scene_digest, expected_slot_digest, expected_cell_digest,
    ):
        """Transfer an eligible landing to a fresh run, without granting approval.

        Match executor lock order (scene, then cell). A concurrent arming or
        source consumption must win or lose the same CAS, never disappear.
        """
        run_id = _id(run_id, "SCENE_SLOT_NEXT_RUN")
        for digest in (expected_scene_digest, expected_slot_digest, expected_cell_digest):
            if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
                raise ContractError("SCENE_BINDING")
        with self.locked_snapshot(expected_scene_digest) as snapshot:
            descriptor = os.open(self._cell.runtime_path("state.lock", create_robot=True),
                                 os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                cell = self._cell.read()
                if canonical_digest(cell) != expected_cell_digest:
                    raise ContractError("STATE_CHANGED")
                if cell["reason_code"] == "EXECUTION_IN_PROGRESS":
                    raise ContractError("OBJECT_POSITION_EXECUTION_IN_PROGRESS")
                position = self.object_position(object_profile_id=object_profile_id, dimensions_mm=dimensions_mm)
                if position["status"] != "AVAILABLE" or position["source"] not in {"ROBOT_RELEASE", "ROBOT_RELEASE_PROXY"}:
                    raise ContractError(position["reason"] or "OBJECT_POSITION_RELEASE_LINEAGE")
                if position["slot_digest"] != expected_slot_digest:
                    raise ContractError("SCENE_SLOT_CHANGED")
                scene = snapshot["scene_state"]
                slot_id = position["slot_id"]
                slot = scene["slot_allocations"][slot_id]
                if run_id == slot["evidence_run_id"]:
                    raise ContractError("SCENE_SLOT_NEXT_RUN")
                if slot["allowed_run_id"] == run_id and slot["state"] == "LANDED_FOR_NEXT_SOURCE":
                    return snapshot
                slots = dict(scene["slot_allocations"])
                # Preserve the release timestamp and evidence, not a new landing.
                slots[slot_id] = {**slot, "state": "LANDED_FOR_NEXT_SOURCE",
                                  "role": "DESTINATION_THEN_NEXT_SOURCE", "allowed_run_id": run_id}
                value = _validate({**scene, "revision": scene["revision"] + 1,
                                   "slot_allocations": slots,
                                   "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}, self.robot_system_id)
                write_json_atomic(self._path(create=True), value)
                return {"scene_state": value, "scene_state_digest": canonical_digest(value)}
            finally:
                os.close(descriptor)

    def consume_next_source(
        self,
        *,
        slot_id: str,
        run_id: str,
        expected_scene_digest: str,
        expected_slot_digest: str,
    ) -> dict:
        """Consume one chain landing for its exact next run using scene and slot CAS."""
        if not isinstance(slot_id, str) or not DIGEST.fullmatch(slot_id):
            raise ContractError("SCENE_SLOT_NEXT_RUN")
        run_id = _id(run_id, "SCENE_SLOT_NEXT_RUN")
        for value in (expected_scene_digest, expected_slot_digest):
            if not isinstance(value, str) or not DIGEST.fullmatch(value):
                raise ContractError("SCENE_SLOT_NEXT_RUN")
        lock_path = self._cell.runtime_path("scene_state.lock", create_robot=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            current = self.read()
            if canonical_digest(current) != expected_scene_digest:
                raise ContractError("SCENE_STATE_CHANGED")
            slot = current.get("slot_allocations", {}).get(slot_id)
            if not isinstance(slot, dict) or canonical_digest(slot) != expected_slot_digest:
                raise ContractError("SCENE_SLOT_CHANGED")
            if (
                slot["state"] != "LANDED_FOR_NEXT_SOURCE"
                or slot["role"] != "DESTINATION_THEN_NEXT_SOURCE"
                or slot["allowed_run_id"] != run_id
            ):
                raise ContractError("SCENE_SLOT_NEXT_RUN")
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            slots = dict(current["slot_allocations"])
            slots[slot_id] = {**slot, "state": "CONSUMED_PENDING_REVIEW", "updated_at": now}
            scene = _validate({
                **current, "revision": current["revision"] + 1,
                "slot_allocations": slots, "updated_at": now,
            }, self.robot_system_id)
            write_json_atomic(self._path(create=True), scene)
            return {"scene_state": scene, "scene_state_digest": canonical_digest(scene)}
        finally:
            os.close(descriptor)

    def update_object(
        self,
        *,
        instance_id: str,
        object_profile_id: str,
        state: str,
        source: str,
        updated_by: str,
        pose: dict | None = None,
        expected_revision: int | None = None,
        expected_cell_digest: str | None = None,
        blocking: bool = True,
    ) -> dict:
        instance_id = _id(instance_id, "SCENE_OBJECT")
        object_profile_id = _id(object_profile_id, "SCENE_OBJECT")
        updated_by = _id(updated_by, "SCENE_OBJECT")
        if state not in OBJECT_STATES or source not in SOURCES:
            raise ContractError("SCENE_OBJECT")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise ContractError("SCENE_REVISION")
        if expected_cell_digest is not None and (
            source != "HUMAN" or not isinstance(expected_cell_digest, str)
            or not DIGEST.fullmatch(expected_cell_digest)
        ):
            raise ContractError("SCENE_BINDING")
        if state == "ON_SURFACE":
            if not isinstance(pose, dict) or set(pose) != POSE_KEYS:
                raise ContractError("SCENE_POSE")
            pose = {
                "place_id": _id(pose["place_id"], "SCENE_POSE"),
                **{key: _number(pose[key], "SCENE_POSE") for key in ("yaw_deg", "x_mm", "y_mm")},
            }
        elif pose is not None:
            raise ContractError("SCENE_POSE")
        lock_path = self._cell.runtime_path("scene_state.lock", create_robot=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        cell_descriptor = None
        try:
            self._flock(descriptor, blocking)
            if expected_cell_digest is not None:
                cell_descriptor = os.open(self._cell.runtime_path("state.lock", create_robot=True),
                                          os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                self._flock(cell_descriptor, blocking)
                cell = self._cell.read()
                if canonical_digest(cell) != expected_cell_digest:
                    raise ContractError("STATE_CHANGED")
            current = self.read()
            if expected_revision is not None and current["revision"] != expected_revision:
                raise ContractError("SCENE_REVISION_CONFLICT")
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            objects = dict(current["objects"])
            objects[instance_id] = {
                "instance_id": instance_id,
                "object_profile_id": object_profile_id,
                "state": state,
                "pose": pose,
                "source": source,
                "updated_by": updated_by,
                "updated_at": now,
            }
            next_scene = {
                **current,
                "revision": current["revision"] + 1,
                "objects": objects,
                "updated_at": now,
            }
            if source == "HUMAN" and "slot_allocations" in current:
                next_scene["slot_allocations"] = {}
            scene = _validate(next_scene, self.robot_system_id)
            write_json_atomic(self._path(create=True), scene)
            return {"scene_state": scene, "scene_state_digest": canonical_digest(scene)}
        finally:
            if cell_descriptor is not None:
                os.close(cell_descriptor)
            os.close(descriptor)


def main(argv=None) -> int:
    parser = ContractArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="outputs/data_factory/cells")
    common.add_argument("--robot-system-id", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show", parents=[common])
    for name in ("set-surface", "set-held", "mark-unknown"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("--instance-id", required=True)
        command.add_argument("--object-profile-id", required=True)
        command.add_argument("--source", choices=sorted(SOURCES), required=True)
        command.add_argument("--updated-by", required=True)
        command.add_argument("--expect-revision", type=int)
        if name == "set-surface":
            command.add_argument("--place-id", required=True)
            command.add_argument("--yaw-deg", type=float, required=True)
            command.add_argument("--x-mm", type=float, required=True)
            command.add_argument("--y-mm", type=float, required=True)
    try:
        args = parser.parse_args(argv)
        store = SceneStateStore(args.root, args.robot_system_id)
        if args.command == "show":
            result = store.snapshot()
        else:
            pose = None
            if args.command == "set-surface":
                pose = {"place_id": args.place_id, "yaw_deg": args.yaw_deg, "x_mm": args.x_mm, "y_mm": args.y_mm}
            result = store.update_object(
                instance_id=args.instance_id,
                object_profile_id=args.object_profile_id,
                state={"set-surface": "ON_SURFACE", "set-held": "HELD", "mark-unknown": "UNKNOWN"}[args.command],
                source=args.source,
                updated_by=args.updated_by,
                pose=pose,
                expected_revision=args.expect_revision,
            )
    except ContractError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": str(exc)}}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"error": {"code": "SCENE_IO", "message": str(exc)}}, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
