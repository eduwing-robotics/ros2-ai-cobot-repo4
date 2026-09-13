#!/usr/bin/env python3
"""Compile qualification-bound FR5 pickup plans without sending robot goals."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.fr5_data_factory import (
    DIGEST,
    RFC3339,
    SAFE_ID,
    ContractError,
    canonical_digest,
    load_json_strict,
    validate_motion_program,
)
from tools.data_factory.scene_state import validate_scene_binding
from tools.data_factory.quality.phase_events import PhaseEventWriter


MODE = "PRE_LIVE"
LIVE_MODE = "LIVE"
MOTION_ONLY_MODE = "LIVE_MOTION_ONLY"
PHASES = (
    "PREGRASP_PTP",
    "APPROACH_STOP_LIN",
    "FINAL_APPROACH_LIN",
    "GRIPPER_CLOSE",
    "LIFT_LIN",
    "RECYCLE_APPROACH_PTP",
    "LOWER_LIN",
    "GRIPPER_OPEN",
    "RETREAT_LIN",
    "SAFE_POSE_PTP",
)
ARM_PHASES = frozenset(PHASES) - {"GRIPPER_CLOSE", "GRIPPER_OPEN"}
JOINT_ORDER = ["j1", "j2", "j3", "j4", "j5", "j6"]
COMMAND_FIELDS = {"schema_version", "op_id", "op", "payload"}
COMMAND_OPS = {"begin_generation", "admit_task", "task_boundary", "execute_terminal", "revoke_task", "prepare_next", "approve_next", "execute_next", "preflight", "capture_observation", "observe_policy", "plan", "approve", "execute", "heartbeat", "confirm", "grasp_verdict", "semantic_verdict", "release_verdict", "cancel", "status"}
ACTIVE_STATES = {"EXECUTING", "PRECONTACT_HUMAN", "GRASP_VERDICT", "SEMANTIC_VERDICT", "LEARNED_CHUNK_COMPLETE", "RELEASE_VERDICT"}
RECYCLE_PHASES = ("RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP")
EXECUTION_RESULT_MARGIN_S = 2.0
EXPECTED_GRAPH = {
    "move_action": ("/move_action", "moveit_msgs/action/MoveGroup"),
    "execute_trajectory": ("/execute_trajectory", "moveit_msgs/action/ExecuteTrajectory"),
    "gripper": (
        "/gripper_controller/follow_joint_trajectory",
        "control_msgs/action/FollowJointTrajectory",
    ),
    "joint_states": ("/joint_states", "sensor_msgs/msg/JointState"),
}


def _exact(value, fields, code):
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError(code)
    return value


_PRECOMMIT_SAFETY_FIELDS = {
    "schema_version", "run_id", "approved_plan_digest", "scene_binding_digest",
    "expected_planning_scene_digest", "planning_scene_readback_digest",
    "collision_report_digest", "plan_only_no_motion_digest",
    "post_reset_safe_snapshot_digest", "status",
}
_PRECOMMIT_EVIDENCE_FIELDS = {
    "schema_version", "run_id", "approved_plan_digest", "scene_binding_digest",
    "expected_planning_scene_digest", "planning_scene_readback", "collision_report",
    "plan_only_no_motion",
}


def _precommit_evidence(value, safety, *, run_id, plan_digest, scene_binding, planning_scene_digest):
    value = _exact(value, _PRECOMMIT_EVIDENCE_FIELDS, "PRECOMMIT_EVIDENCE_SCHEMA")
    if (
        value["schema_version"] != "data_factory.precommit_evidence.v1"
        or value["run_id"] != run_id
        or value["approved_plan_digest"] != plan_digest
        or value["scene_binding_digest"] != canonical_digest(scene_binding)
        or value["expected_planning_scene_digest"] != planning_scene_digest
    ):
        raise ContractError("PRECOMMIT_EVIDENCE_BINDING")
    readback, collision, no_motion = (
        value["planning_scene_readback"], value["collision_report"], value["plan_only_no_motion"],
    )
    if (
        not isinstance(readback, dict)
        or set(readback) != {"schema_version", "run_id", "plan_digest", "expected_planning_scene_digest", "objects"}
        or readback["schema_version"] != "data_factory.planning_scene_readback.v1"
        or readback["run_id"] != run_id
        or readback["plan_digest"] != plan_digest
        or readback["expected_planning_scene_digest"] != planning_scene_digest
        or not isinstance(readback["objects"], list)
        or not isinstance(collision, dict)
        or set(collision) != {"schema_version", "plan_digest", "sample_count", "samples", "failure_count", "all_valid"}
        or collision["schema_version"] != "data_factory.collision_report.v1"
        or collision["plan_digest"] != plan_digest
        or not isinstance(no_motion, dict)
        or set(no_motion) != {"schema_version", "run_id", "plan_digest", "before_snapshot", "after_snapshot", "max_joint_delta_rad", "gripper_delta_m", "execute_goal_count", "gripper_goal_count"}
        or no_motion["schema_version"] != "data_factory.plan_only_no_motion.v1"
        or no_motion["run_id"] != run_id
        or no_motion["plan_digest"] != plan_digest
        or canonical_digest(readback) != safety["planning_scene_readback_digest"]
        or canonical_digest(collision) != safety["collision_report_digest"]
        or canonical_digest(no_motion) != safety["plan_only_no_motion_digest"]
    ):
        raise ContractError("PRECOMMIT_EVIDENCE_BINDING")
    return copy.deepcopy(value)


def _gripper_settings(value):
    required = {
        "hardware_plugin", "velocity_percent", "force_percent",
        "settle_time_ms",
    }
    optional = {"open_velocity_percent", "open_force_percent"}
    if (
        not isinstance(value, dict)
        or not required <= set(value) <= required | optional
    ):
        raise ContractError("GRIPPER_SETTINGS_UNVERIFIED")
    value = dict(value)
    value.setdefault("open_velocity_percent", value["velocity_percent"])
    value.setdefault("open_force_percent", value["force_percent"])
    if (
        value["hardware_plugin"] not in {"fairino_hardware/FairinoHardwareInterface", "mock_components/GenericSystem"}
        or any(type(value[key]) is not int or not 1 <= value[key] <= 100 for key in ("velocity_percent", "open_velocity_percent", "force_percent", "open_force_percent"))
        or type(value["settle_time_ms"]) is not int
        or not 50 <= value["settle_time_ms"] <= 10000
    ):
        raise ContractError("GRIPPER_SETTINGS_UNVERIFIED")
    return dict(value)


def _joint_positions(value):
    if (
        not isinstance(value, list)
        or len(value) != len(JOINT_ORDER)
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            for item in value
        )
    ):
        raise ContractError("JOINTS")
    return [float(item) for item in value]


def _future_timestamp(value, now):
    if not isinstance(value, str) or not RFC3339.fullmatch(value):
        raise ContractError("APPROVAL_EXPIRY")
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("APPROVAL_EXPIRY") from exc
    if now.tzinfo is None:
        raise ContractError("NOW_TIMEZONE")
    if expiry <= now:
        raise ContractError("APPROVAL_EXPIRED")
    return expiry


def _response(
    *,
    op_id=None,
    op=None,
    code="OK",
    ok=False,
    run_id=None,
    plan_digest=None,
    state="IDLE",
    data=None,
    mode=MODE,
):
    return {
        "schema_version": "fr5.pickup_executor.response.v3",
        "mode": mode,
        "op_id": op_id,
        "op": op,
        "ok": ok,
        "code": code,
        "run_id": run_id,
        "plan_digest": plan_digest,
        "state": state,
        "data": data,
    }


class UnavailableTransport:
    def preflight(self):
        raise ContractError("OFFLINE_TRANSPORT_UNAVAILABLE")

    def plan_arm(self, *args):
        raise ContractError("OFFLINE_TRANSPORT_UNAVAILABLE")

    def snapshot(self, *args):
        raise ContractError("OFFLINE_TRANSPORT_UNAVAILABLE")

    def build_gripper_goal(self, *args):
        raise ContractError("OFFLINE_TRANSPORT_UNAVAILABLE")

    def precommit_safety(self, *args):
        raise ContractError("OFFLINE_TRANSPORT_UNAVAILABLE")


class PickupExecutor:
    """Compile and approve plans; real execution stays opt-in for tests only."""

    def __init__(
        self, transport=None, clock=None, monotonic_clock=None,
        cell_state_store=None, scene_state_store=None, execution_enabled=False,
        phase_events_root=None, event_clock=None,
        source_clock=None,
        motion_only_binding_digest=None,
        motion_only_parent_run_id=None,
        motion_only_parent_plan_digest=None,
        motion_only_preapproval_scope_digest=None,
        motion_only_expected_run_id=None,
        motion_only_expected_resolved_job_digest=None,
        motion_only_expected_program_digest=None,
        motion_only_expected_scene_digest=None,
        motion_only_expectation_digest=None,
    ):
        motion_only_values = (
            motion_only_binding_digest, motion_only_parent_run_id,
            motion_only_parent_plan_digest,
            motion_only_preapproval_scope_digest,
            motion_only_expected_run_id,
            motion_only_expected_resolved_job_digest,
            motion_only_expected_program_digest,
            motion_only_expected_scene_digest,
            motion_only_expectation_digest,
        )
        if any(value is not None for value in motion_only_values) != all(
            value is not None for value in motion_only_values
        ) or motion_only_binding_digest is not None and (
            not execution_enabled
            or not isinstance(motion_only_binding_digest, str)
            or DIGEST.fullmatch(motion_only_binding_digest) is None
            or not isinstance(motion_only_parent_run_id, str)
            or SAFE_ID.fullmatch(motion_only_parent_run_id) is None
            or not isinstance(motion_only_parent_plan_digest, str)
            or DIGEST.fullmatch(motion_only_parent_plan_digest) is None
            or not isinstance(motion_only_preapproval_scope_digest, str)
            or DIGEST.fullmatch(motion_only_preapproval_scope_digest) is None
            or not isinstance(motion_only_expected_run_id, str)
            or SAFE_ID.fullmatch(motion_only_expected_run_id) is None
            or any(
                not isinstance(value, str) or DIGEST.fullmatch(value) is None
                for value in (
                    motion_only_expected_resolved_job_digest,
                    motion_only_expected_program_digest,
                    motion_only_expected_scene_digest,
                    motion_only_expectation_digest,
                )
            )
        ):
            raise ContractError("MOTION_ONLY_BINDING")
        self.transport = transport or UnavailableTransport()
        self.source_clock = source_clock or time.time
        self._processing = threading.Lock()
        self._ticking = False
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.cell_state_store = cell_state_store
        self.scene_state_store = scene_state_store
        self.execution_enabled = execution_enabled
        self.motion_only_binding_digest = motion_only_binding_digest
        self.motion_only_parent_run_id = motion_only_parent_run_id
        self.motion_only_parent_plan_digest = motion_only_parent_plan_digest
        self.motion_only_preapproval_scope_digest = (
            motion_only_preapproval_scope_digest
        )
        self.motion_only_expected_run_id = motion_only_expected_run_id
        self.motion_only_expected_resolved_job_digest = (
            motion_only_expected_resolved_job_digest
        )
        self.motion_only_expected_program_digest = (
            motion_only_expected_program_digest
        )
        self.motion_only_expected_scene_digest = (
            motion_only_expected_scene_digest
        )
        self.motion_only_expectation_digest = motion_only_expectation_digest
        self.phase_events_root = Path(phase_events_root) if phase_events_root is not None else None
        self.event_clock = event_clock or (lambda: (time.time_ns(), "SYSTEM_TIME"))
        self.mode = (
            MOTION_ONLY_MODE if motion_only_binding_digest is not None
            else LIVE_MODE if execution_enabled else MODE
        )
        self.cache = {}
        self._cached_observation = None
        self._policy_observer = None
        self.runs = {}
        self._generation = None  # One pending output on this existing command owner.
        self._phase_event_writer = None

    def close(self):
        if self._generation is not None:
            self._generation["cancelled"] = True
        self._retire_observation_cache()
        observer_closed = self._close_policy_observer()
        if self._phase_event_writer is None:
            return observer_closed
        ok = self._phase_event_writer.close()
        if not ok:
            for run in self.runs.values():
                if "execution" in run:
                    run["execution"]["behavior_report_status"] = "BEHAVIOR_REPORT_UNAVAILABLE"
        return ok and observer_closed

    def _close_policy_observer(self):
        if self._policy_observer is None:
            return True
        try:
            self._policy_observer[1].close()
        except Exception:
            # Observation cleanup cannot prevent the actual motion stop owner.
            # Keep the handle so close() can retry and report incomplete cleanup.
            return False
        self._policy_observer = None
        return True

    def _emit_phase_event(self, run, event, step, action_status, evidence):
        if "mechanical_terminal" in run and step["phase"] in RECYCLE_PHASES:
            terminal=run["mechanical_terminal"]
            terminal.setdefault("events",[]).append({"terminal_digest":terminal["plan"]["terminal_digest"],
                "phase":step["phase"], "event":event, "action_status":action_status,
                "monotonic_time_s":self.monotonic_clock(), "evidence_digest":canonical_digest(evidence)})
            return  # These out-of-recording events do not describe learned rows.
        writer = self._phase_event_writer
        if writer is None:
            return
        execution = run["execution"]
        sequence = execution["phase_event_sequence"]
        execution["phase_event_sequence"] += 1
        try:
            event_ros_time_ns, ros_clock_type = self.event_clock()
            segments = run["plan"]["steps"][0].get("held_target_segments") if step["phase"] == "LEARNED_CHUNK" else None
            index = execution.get("learned_segment_index", 0) if segments else 0
            count = len(segments) if segments else 1
            record = {
                "schema_version": "data_factory.phase_event.v1",
                "run_id": run["plan"]["run_id"],
                "plan_digest": run["digest"],
                "sequence": sequence,
                "phase": step["phase"],
                "segment_index": None if event in {"HOLD_ENTERED", "DECISION_RECEIVED"} else index,
                "segment_count": None if event in {"HOLD_ENTERED", "DECISION_RECEIVED"} else count,
                "event": event,
                "event_ros_time_ns": event_ros_time_ns,
                "monotonic_time_ns": int(round(self.monotonic_clock() * 1_000_000_000)),
                "ros_clock_type": ros_clock_type,
                "event_source": (
                    "object_reposition_executor"
                    if self.motion_only_binding_digest is not None
                    else "pickup_executor"
                ),
                "action_status": action_status,
                "evidence_digest": canonical_digest(evidence),
            }
            emitted = writer.emit(record, plan=run["plan"]) if run.get("learned_history") else writer.emit(record)
            if not emitted:
                execution["behavior_report_status"] = "BEHAVIOR_REPORT_UNAVAILABLE"
        except (ContractError, KeyError, TypeError, ValueError, OverflowError):
            execution["behavior_report_status"] = "BEHAVIOR_REPORT_UNAVAILABLE"

    def process(self, request):
        if not self._processing.acquire(blocking=False):
            for run in list(self.runs.values()):
                if run.get("state") in ACTIVE_STATES:
                    self._fault(run, "REENTRANT_COMMAND")
            return _response(code="REENTRANT_COMMAND", mode=self.mode)
        try:
            return self._process(request)
        finally:
            self._processing.release()

    def _retire_observation_cache(self, code="LEARNED_OBSERVATION_RETIRED"):
        if self._cached_observation is None:
            return
        op_id, _ = self._cached_observation
        digest, response = self.cache[op_id]
        # Keep idempotency/conflict identity, never cache raw RGB for the task's
        # history or silently recapture when an old operation is retried.
        self.cache[op_id] = (digest, {**response, "ok": False, "code": code, "data": None})
        self._cached_observation = None

    def _process(self, request):
        try:
            request = _exact(request, COMMAND_FIELDS, "COMMAND_SCHEMA")
            op_id, op = request["op_id"], request["op"]
            if (
                request["schema_version"] != "fr5.pickup_executor.command.v4"
                or not isinstance(op_id, str)
                or not SAFE_ID.fullmatch(op_id)
                or op not in COMMAND_OPS
            ):
                raise ContractError("COMMAND_SCHEMA")
            request_digest = canonical_digest(request)
        except ContractError as exc:
            return _response(code=exc.code, mode=self.mode)

        # Input reads do not drive actuation/completion. Normal polling and
        # heartbeats retain the lease/stop owner's tick; reads check their scope.
        if op != "observe_policy":
            self.tick()
        previous = self.cache.get(op_id)
        if previous is not None:
            if previous[0] == request_digest:
                if op == "observe_policy":
                    try:
                        self._policy_observation_owner(request["payload"])
                    except ContractError as exc:
                        self._retire_observation_cache(exc.code)
                        self.cache[op_id] = (request_digest, {
                            **previous[1], "ok": False, "code": exc.code, "data": None})
                        return copy.deepcopy(self.cache[op_id][1])
                if self._cached_observation is not None and self._cached_observation[0] == op_id:
                    from tools.data_factory.rollout.finite_plan import check_freshness
                    age = request["payload"]["max_observation_age_s"]
                    try:
                        if not 0 <= self.monotonic_clock() - self._cached_observation[1] <= age:
                            raise ContractError("LEARNED_STALE_OBSERVATION")
                        check_freshness({"source_timestamps_s": previous[1]["data"]["observation"]["source_timestamps_s"],
                                         "max_observation_age_s": age}, self.source_clock())
                    except ContractError as exc:
                        self._retire_observation_cache(exc.code)
                return copy.deepcopy(self.cache[op_id][1])
            return _response(op_id=op_id, op=op, code="OP_ID_CONFLICT", mode=self.mode)

        # A distinct request ends the previous capture's retry window. Only its
        # small command receipt survives; the actual caller owns the RGB input.
        self._retire_observation_cache()
        try:
            result = getattr(self, f"_{op}")(request["payload"])
        except ContractError as exc:
            result = _response(code=exc.code)
        result["mode"] = self.mode
        result["op_id"], result["op"] = op_id, op
        snapshot = copy.deepcopy(result)
        data = snapshot.get("data")
        history = self.runs.get(snapshot.get("run_id"), {}).get("learned_history")
        if isinstance(data, dict) and "learned_history" in data and data["learned_history"] == history:
            # Private history entries are frozen when archived. Share that
            # immutable prefix across cached receipts; callers still receive
            # deep copies, avoiding a full history copy per heartbeat in cache.
            data["learned_history"] = history
        self.cache[op_id] = (request_digest, snapshot)
        if op in {"capture_observation", "observe_policy"} and snapshot["ok"] and "observation" in (snapshot.get("data") or {}):
            self._cached_observation = (op_id, self.monotonic_clock())
        return copy.deepcopy(snapshot)

    def _policy_observation_owner(self, payload):
        fields = {"run_id", "plan_digest", "lease_id", "camera_topics", "max_observation_age_s"}
        _exact(payload, fields, "LEARNED_OBSERVATION_SCHEMA")
        run = self._bound(payload)
        if (self.motion_only_binding_digest is not None or "task_grant" not in run
                or "mechanical_terminal" in run
                or run["state"] not in {"EXECUTING", "LEARNED_CHUNK_COMPLETE"}):
            raise ContractError("TASK_GRANT_STATE")
        self._check_task(run)
        execution = run["execution"]
        if payload["lease_id"] != execution["lease_id"]:
            raise ContractError("LEASE_BINDING")
        if run["cancel_event"].is_set():
            raise ContractError("LEARNED_CANCELLED")
        if self.monotonic_clock() >= execution["lease_deadline"]:
            raise ContractError("HEARTBEAT_TIMEOUT")
        return run

    def _observe_policy(self, payload):
        """Read the current task's source observer without waiting for motion.

        This does not advance references, admit an output, renew a lease or
        change Scene/evidence authority. Initial absence is a pending sample,
        not a physical failure or a fabricated fresh observation.
        """
        run = self._policy_observation_owner(payload)
        from tools.data_factory.rollout.finite_plan import _number, check_freshness
        proposal = run["plan"]["learned_proposal"]
        age = _number(payload["max_observation_age_s"], "LEARNED_SOURCE_CLOCK")
        if not 0 < age <= proposal["max_observation_age_s"]:
            raise ContractError("LEARNED_STALE_OBSERVATION")
        inputs = proposal.get("runtime_inputs")
        if inputs is None or payload["camera_topics"] != inputs["camera_topics"]:
            raise ContractError("LEARNED_CAMERA_MAPPING")
        binding = copy.deepcopy(payload)
        if self._policy_observer is not None and self._policy_observer[0] != binding:
            if not self._close_policy_observer():
                raise ContractError("LEARNED_OBSERVATION_CLOSE_UNCONFIRMED")
        if self._policy_observer is None:
            factory = getattr(self.transport, "policy_observation_stream", None)
            if not callable(factory):
                raise ContractError("LEARNED_OBSERVATION_UNAVAILABLE")
            self._policy_observer = (binding, factory(payload["camera_topics"], age))
        observer = self._policy_observer[1]
        observation = self.transport.poll_policy_observation(observer)
        self._policy_observation_owner(payload)  # Late conversion cannot revive authority.
        if observation is None:
            return _response(code="LEARNED_OBSERVATION_PENDING", ok=True,
                run_id=payload["run_id"], plan_digest=payload["plan_digest"], state=run["state"], data={})
        check_freshness({"source_timestamps_s": observation["source_timestamps_s"],
                         "max_observation_age_s": age}, self.source_clock())
        return _response(code="LEARNED_OBSERVATION", ok=True, run_id=payload["run_id"],
            plan_digest=payload["plan_digest"], state=run["state"], data={"observation": observation})

    def _capture_observation(self, payload):
        self._check_pending_generation()
        fields = {"camera_topics", "max_observation_age_s"}
        bound = isinstance(payload, dict) and "run_id" in payload
        _exact(payload, fields | ({"run_id", "plan_digest", "lease_id"} if bound else set()), "LEARNED_OBSERVATION_SCHEMA")
        run = self._bound(payload) if bound else None
        if self.motion_only_binding_digest is not None or self.runs and not bound:
            raise ContractError("ONE_JOB_ONLY")
        if bound:
            if run["state"] != "LEARNED_CHUNK_COMPLETE":
                raise ContractError("LEARNED_CHUNK_STATE")
            if payload["lease_id"] != run["execution"]["lease_id"]:
                raise ContractError("LEASE_BINDING")
            from tools.data_factory.rollout.finite_plan import _number
            proposal = run["plan"]["learned_proposal"]
            age = _number(payload["max_observation_age_s"], "LEARNED_SOURCE_CLOCK")
            if not 0 < age <= proposal["max_observation_age_s"]:
                raise ContractError("LEARNED_STALE_OBSERVATION")
            inputs = proposal.get("runtime_inputs")
            if inputs is not None and payload["camera_topics"] != inputs["camera_topics"]:
                raise ContractError("LEARNED_CAMERA_MAPPING")
            if run["execution"]["active"] or getattr(self.transport, "owns_active_goal", True):
                raise ContractError("ROS_EXEC_ACTIVE")
        capture = getattr(self.transport, "capture_policy_observation", None)
        if capture is None:
            raise ContractError("LEARNED_OBSERVATION_UNAVAILABLE")
        observation = capture(payload["camera_topics"], payload["max_observation_age_s"])
        self._check_pending_generation()
        if bound:
            # A late read cannot renew the lease or revive a cancelled attempt.
            self.tick()
            if run["state"] != "LEARNED_CHUNK_COMPLETE":
                raise ContractError(run.get("failure_code", "LEARNED_CHUNK_STATE"))
            if run["execution"]["active"] or getattr(self.transport, "owns_active_goal", True):
                raise ContractError("ROS_EXEC_ACTIVE")
            from tools.data_factory.rollout.finite_plan import check_freshness
            check_freshness({"source_timestamps_s": observation["source_timestamps_s"],
                             "max_observation_age_s": payload["max_observation_age_s"]}, self.source_clock())
            return _response(code="LEARNED_OBSERVATION", ok=True, run_id=payload["run_id"],
                plan_digest=payload["plan_digest"], state=run["state"],
                data={**self._execution_data(run), "observation": observation})
        return _response(code="LEARNED_OBSERVATION", ok=True, state="IDLE", data={"observation": observation})

    def _validated_preflight(self, motion_program):
        validate_motion_program(motion_program)
        facts = _exact(
            self.transport.preflight(),
            {*EXPECTED_GRAPH, "joint_order"},
            "PREFLIGHT_FACTS",
        )
        for key, (endpoint, type_name) in EXPECTED_GRAPH.items():
            expected = {"endpoint": endpoint, "type": type_name, "ready": True}
            if _exact(facts[key], set(expected), "PREFLIGHT_FACTS") != expected:
                raise ContractError("PREFLIGHT_ACTION_SURFACE_MISMATCH")
        if facts["joint_order"] != JOINT_ORDER:
            raise ContractError("PREFLIGHT_ACTION_SURFACE_MISMATCH")
        return {"action_graph_digest": canonical_digest(facts), **facts}

    def _preflight(self, payload):
        _exact(payload, {"motion_program"}, "PREFLIGHT_SCHEMA")
        if self.motion_only_binding_digest is not None:
            raise ContractError("MOTION_ONLY_CONTINUATION_BINDING")
        facts = self._validated_preflight(payload["motion_program"])
        return _response(code="PREFLIGHT_OK", ok=True, state="PREFLIGHT", data=facts)

    def _validate_motion_only_continuation(
        self, run_id, motion_program, scene_binding,
    ):
        if self.motion_only_binding_digest is None:
            return
        expectation = {
            "schema_version": "data_factory.motion_only_continuation.v1",
            "preapproval_scope_digest":
                self.motion_only_preapproval_scope_digest,
            "object_reposition_binding_digest":
                self.motion_only_binding_digest,
            "run_id": run_id,
            "resolved_job_digest": motion_program["resolved_job_digest"],
            "motion_program_digest": canonical_digest(motion_program),
            "scene_binding_digest": canonical_digest(scene_binding),
        }
        expectation["expectation_digest"] = canonical_digest(expectation)
        if (
            run_id != self.motion_only_expected_run_id
            or motion_program["resolved_job_digest"]
            != self.motion_only_expected_resolved_job_digest
            or expectation["motion_program_digest"]
            != self.motion_only_expected_program_digest
            or expectation["scene_binding_digest"]
            != self.motion_only_expected_scene_digest
            or expectation["expectation_digest"]
            != self.motion_only_expectation_digest
        ):
            raise ContractError("MOTION_ONLY_CONTINUATION_BINDING")

    def _check_pending_generation(self):
        owned = self._generation
        if owned is None or owned["consumed"]:
            return
        if owned["cancelled"]:
            raise ContractError("LEARNED_CANCELLED")
        if owned["revoked"]:
            raise ContractError("TASK_GRANT_REVOKED")
        context, grant = owned["context"], owned["grant"]
        if self.source_clock() >= grant["deadline_s"] or not context["opened_monotonic_s"] <= self.monotonic_clock() < context["task_deadline_monotonic_s"]:
            raise ContractError("TASK_DEADLINE_EXHAUSTED")

    def _begin_generation(self, payload):
        from uuid import uuid4
        from tools.data_factory.rollout.finite_plan import SCOPED_SCHEMAS
        from tools.data_factory.rollout.task_authority import GENERATION_SCHEMA, check_grant
        fields = {"run_id", "grant", "source_program", "scene_binding", "proposal_spec", "predecessor_plan_digest", "lease_id"}
        _exact(payload, fields, "LEARNED_GENERATION_CONTEXT")
        source = validate_motion_program(copy.deepcopy(payload["source_program"]))
        scene = validate_scene_binding(payload["scene_binding"])
        spec, grant = payload["proposal_spec"], payload["grant"]
        keys = {"schema_version", "checkpoint", "instruction", "robot_description", "velocity_scaling", "period_s", "max_observation_age_s"}
        if isinstance(spec, dict) and "runtime_inputs" in spec:
            keys.add("runtime_inputs")
        if not isinstance(spec, dict) or set(spec) != keys or spec["schema_version"] not in SCOPED_SCHEMAS or spec["max_observation_age_s"] != .3:
            raise ContractError("LEARNED_GENERATION_CONTEXT")
        plan = {"run_id": payload["run_id"], "learned_source_program": source, "scene_binding": scene, "learned_proposal": spec}
        now, steady = self.source_clock(), self.monotonic_clock()
        check_grant(grant, plan, now)
        predecessor = payload["predecessor_plan_digest"]
        if predecessor is None:
            if self.runs or self._generation is not None or payload["lease_id"] is not None:
                raise ContractError("LEARNED_GENERATION_REUSED")
            deadline = steady + (grant["deadline_s"] - now)
        else:
            run = self._bound({"run_id": payload["run_id"], "plan_digest": predecessor})
            self._chunk_boundary(run, payload["lease_id"])
            if grant != run.get("task_grant") or scene != run["plan"]["scene_binding"] or source != run["plan"]["learned_source_program"]:
                raise ContractError("TASK_GRANT_SCOPE")
            if "pending_chunk" in run or (self._generation is not None and not self._generation["consumed"]):
                raise ContractError("LEARNED_GENERATION_REUSED")
            if self._task_termination(run) is not None:
                raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
            deadline = run["task_deadline"]
        if min(grant["deadline_s"] - now, deadline - steady) <= grant["terminal_reserve_s"] + 5.:
            raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
        context = {"schema_version": GENERATION_SCHEMA, "generation_id": uuid4().hex,
                   "run_id": payload["run_id"], "task_grant_digest": grant["grant_digest"],
                   "scene_binding_digest": canonical_digest(scene), "predecessor_plan_digest": predecessor,
                   "opened_at_s": now, "opened_monotonic_s": steady, "task_deadline_monotonic_s": deadline}
        self._generation = {"context": context, "grant": copy.deepcopy(grant), "spec": copy.deepcopy(spec),
                            "cancelled": False, "revoked": False, "consumed": False}
        return _response(ok=True, code="LEARNED_GENERATION_OPENED", state="IDLE" if predecessor is None else run["state"],
                         run_id=payload["run_id"], plan_digest=predecessor, data={"generation_context": copy.deepcopy(context)})

    def _check_generation(self, payload, predecessor):
        from tools.data_factory.rollout.task_authority import check_generation, check_grant
        p = payload["motion_program"]["learned_proposal"]
        owned = self._generation
        if owned is None or p["generation_context"] != owned["context"] or owned["consumed"]:
            raise ContractError("LEARNED_GENERATION_CONTEXT")
        if owned["cancelled"]:
            raise ContractError("LEARNED_CANCELLED")
        if owned["revoked"]:
            raise ContractError("TASK_GRANT_REVOKED")
        if any(p.get(k) != v for k, v in owned["spec"].items()):
            raise ContractError("TASK_GRANT_SCOPE")
        plan = {"run_id": payload["run_id"], "scene_binding": payload["scene_binding"],
                "learned_source_program": payload["motion_program"]["source_program"], "learned_proposal": p}
        check_grant(owned["grant"], plan, self.source_clock())
        check_generation(owned["grant"], p, run_id=payload["run_id"], scene=payload["scene_binding"],
                         predecessor=predecessor, now=self.source_clock(), steady=self.monotonic_clock())

    def _plan(self, payload):
        _exact(payload, {"run_id", "motion_program", "scene_binding"}, "PLAN_SCHEMA")
        run_id = payload["run_id"]
        if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id):
            raise ContractError("PLAN_SCHEMA")
        if run_id in self.runs:
            raise ContractError("RUN_ID_REUSED")
        if self.runs:
            raise ContractError("ONE_JOB_ONLY")

        response = self._compile_plan(payload)
        if response["ok"]:
            if "generation_context" in payload["motion_program"].get("learned_proposal", {}):
                self._check_generation(payload, None)
            self.runs[run_id] = self._planned_record(response)
            if "generation_context" in response["data"]["plan"].get("learned_proposal", {}):
                self._generation["consumed"] = True
                self.runs[run_id].update(generation_grant=copy.deepcopy(self._generation["grant"]),
                                         task_deadline=self._generation["context"]["task_deadline_monotonic_s"])
        return response

    @staticmethod
    def _planned_record(response):
        envelope = response["data"]
        return {"plan": copy.deepcopy(envelope["plan"]), "digest": response["plan_digest"],
                "precommit_safety": copy.deepcopy(envelope["precommit_safety"]),
                "precommit_evidence": copy.deepcopy(envelope["precommit_evidence"]),
                "recycle_plan_digest": envelope["operator_summary"].get("recycle", {}).get("plan_digest"),
                "envelope": copy.deepcopy(envelope), "state": "PLANNED"}

    def _compile_plan(self, payload, *, chunk_binding=None):
        # Compilation may read/serialize/check collision but never replaces the
        # current run, approval, lease or execution owner.
        run_id = payload["run_id"]
        motion_program = validate_motion_program(copy.deepcopy(payload["motion_program"]))
        scene_binding = validate_scene_binding(payload["scene_binding"])
        self._validate_motion_only_continuation(
            run_id, motion_program, scene_binding,
        )
        proposal = motion_program.get("learned_proposal")
        if proposal is not None:
            from tools.data_factory.rollout.finite_plan import check_freshness
            if self.motion_only_binding_digest is not None:
                raise ContractError("LEARNED_SCENE_SCOPE")
            # Retain the normal resolver's slots in the exact plan/grant. Their
            # presence is not evidence that this finite output placed anything.
            if "release_slot" in scene_binding:
                from tools.data_factory.scene_state import validate_release_slot
                validate_release_slot(scene_binding["release_slot"], motion_program["robot_system_id"])
            checked_at = self.source_clock()
            try:
                if "generation_context" in proposal:
                    self._check_generation(payload, None if chunk_binding is None else chunk_binding["previous_plan_digest"])
                else:
                    check_freshness(proposal, checked_at)
            except ContractError as exc:
                if exc.code != "LEARNED_STALE_OBSERVATION" or chunk_binding is not None:
                    raise
                try:
                    return _response(code=exc.code, run_id=run_id, data={"planning_failure": {
                        "stage": "INITIAL_PLAN_ADMISSION",
                        "motion_program_digest": canonical_digest(motion_program),
                        "checked_at_s": checked_at,
                        "source_timestamps_s": copy.deepcopy(proposal["source_timestamps_s"]),
                        "inference_completed_at_s": proposal["inference_completed_at_s"],
                        "max_observation_age_s": proposal["max_observation_age_s"],
                    }})
                except Exception:
                    # Diagnostic construction must not replace the original rejection.
                    raise exc
        action_graph = self._validated_preflight(motion_program)
        observed = self.transport.snapshot(motion_program["planning"]["max_joint_state_age_s"])
        observed = _exact(
            observed,
            {"joint_positions", "joint_state_age_s", "gripper_settings", "arm_controller", "gripper_controller"}
            | ({"joint_state_stamp_ns"} if "joint_state_stamp_ns" in observed else set()),
            "SNAPSHOT_SCHEMA",
        )
        settings = _gripper_settings(observed["gripper_settings"])
        required = motion_program["gripper_requirements"]
        if (
            settings["velocity_percent"] != required["velocity_percent"]
            or settings["open_velocity_percent"]
            != required.get("open_velocity_percent", required["velocity_percent"])
            or settings["force_percent"] != required["force_percent"]
            or settings["open_force_percent"]
            != required.get("open_force_percent", required["force_percent"])
        ):
            raise ContractError("GRIPPER_SETTINGS_MISMATCH")
        for controller in ("arm_controller", "gripper_controller"):
            fields = {"endpoint", "type", "publisher_count", "ready", "age_s", "speed_scaling"}
            if "sample" in observed[controller]:
                from tools.data_factory.motion.moveit_transport import validate_controller_sample
                fields.add("sample")
                validate_controller_sample(observed[controller])
            if controller == "gripper_controller":
                fields |= {"reference_position_m", "feedback_position_m"}
                if "hardware_execution" in observed[controller]:
                    fields.add("hardware_execution")
            controller_state = _exact(
                observed[controller],
                fields,
                "SNAPSHOT_SCHEMA",
            )
            if controller_state["ready"] is not True:
                raise ContractError("CONTROLLER_NOT_READY")
        initial_state = _joint_positions(observed.get("joint_positions"))
        hardware_binding = None
        if proposal is not None and "hardware_execution" in observed["gripper_controller"]:
            from tools.data_factory.rollout.gripper_evidence import check_hardware, identity
            captured = {"captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock(), "snapshot": observed}
            wire = check_hardware(captured, captured["captured_at_s"], captured["captured_monotonic_s"],
                                  motion_program["planning"]["max_joint_state_age_s"])
            hardware_binding = {"incarnation": identity(wire), "generation": wire["generation"]}
        state = initial_state
        planned_steps = []
        for step in motion_program["steps"]:
            phase = step["phase"]
            planned_duration_s = None
            continuation = None
            if phase == "LEARNED_CHUNK":
                serialized_references = "reference_timing" in proposal
                tolerance = motion_program["planning"]["goal_tolerances"]["joint_rad"]
                gripper_tolerance = next(item["limits"]["completion_tolerance_m"] for item in motion_program["source_program"]["steps"] if item["phase"] == "GRIPPER_OPEN")
                if (any(abs(a - b) > tolerance for a, b in zip(state, proposal["initial_state"][:6]))
                        or abs(observed["gripper_controller"]["feedback_position_m"] - proposal["initial_state"][-1]) > gripper_tolerance
                        or not serialized_references and abs(observed["gripper_controller"]["reference_position_m"] - proposal["initial_state"][-1]) > gripper_tolerance):
                    raise ContractError("LEARNED_START_STATE")
                held_segments = []
                for segment in step.get("held_target_segments", []):
                    segment = copy.deepcopy(segment)
                    begin, end = segment["action_range"]
                    gripper = observed["gripper_controller"]
                    if serialized_references and begin == 0 and segment["type"] == "ARM":
                        from tools.data_factory.rollout.finite_plan import _limits
                        rate = _limits(proposal["robot_description"])[-1][2] * proposal["velocity_scaling"]
                        if abs(proposal["actions"][0][-1] - gripper["reference_position_m"]) > rate * proposal["reference_timing"]["durations_s"][0] + 1e-9:
                            raise ContractError("LEARNED_VELOCITY_LIMIT")
                        segment["gripper_position_m"] = gripper["reference_position_m"]
                        if proposal.get("runtime_inputs", {}).get("hardware_wire_version") in (4, 5):
                            from tools.data_factory.rollout.finite_plan import native_close_feedback
                            segment.pop("native_close_feedback", None)
                            segment.update(native_close_feedback(motion_program["source_program"], gripper["reference_position_m"]))
                    if (segment["type"] == "GRIPPER" and begin == 0
                            and abs(gripper["reference_position_m"] - segment["gripper_position_m"]) <= 1e-9
                            and segment["acceptable_feedback_m"]["min"] <= gripper["feedback_position_m"] <= segment["acceptable_feedback_m"]["max"]):
                        # Freeze the omission before exact-plan approval. The
                        # following arm segment rechecks this held reference.
                        continue
                    prior = proposal["initial_state"] if begin == 0 else proposal["actions"][begin - 1]
                    if serialized_references and segment["type"] == "GRIPPER":
                        prior = proposal["actions"][begin]
                    segment = {**copy.deepcopy(segment), "phase": phase,
                               "learned_proposal": copy.deepcopy(proposal),
                               "start_joint_state": list(prior[:6]),
                               "final_joint_state": list(proposal["actions"][end - 1][:6]) if segment["type"] == "ARM" else list(prior[:6]),
                               "limits": copy.deepcopy(segment["gripper_limits"] if segment["type"] == "GRIPPER" else step["limits"]),
                               "max_joint_state_age_s": motion_program["planning"]["max_joint_state_age_s"],
                               "joint_tolerance_rad": tolerance}
                    encoded = self.transport.build_learned_segment(proposal, segment)
                    if not isinstance(encoded, bytes) or not encoded:
                        raise ContractError("LEARNED_TRAJECTORY")
                    segment["trajectory_b64"] = base64.b64encode(encoded).decode("ascii")
                    if serialized_references:
                        segment.pop("learned_proposal")
                        segment["learned_proposal_digest"] = proposal["proposal_digest"]
                    held_segments.append(segment)
                serialized = (base64.b64decode(held_segments[0]["trajectory_b64"]) if held_segments
                              else self.transport.build_learned_trajectory(proposal))
                if not isinstance(serialized, bytes) or not serialized:
                    raise ContractError("LEARNED_TRAJECTORY")
                final_state = list(proposal["actions"][-1][:6])
                step_type = "ARM"
                planned_duration_s = len(proposal["actions"]) * proposal["period_s"]
                if serialized_references:
                    planned_duration_s = sum(proposal["reference_timing"]["durations_s"])
                planned_duration_s += sum(s["limits"]["command_duration_s"] for s in held_segments if s["type"] == "GRIPPER")
                if planned_duration_s + EXECUTION_RESULT_MARGIN_S > step["limits"]["execution_timeout_s"]:
                    raise ContractError("LEARNED_EXECUTION_TIMEOUT")
            elif phase in ARM_PHASES:
                result = _exact(
                    self.transport.plan_arm(
                        phase,
                        step.get("target"),
                        step.get("joint_positions_rad"),
                        step["limits"],
                        motion_program["frames"],
                        motion_program["planning"],
                        state,
                    ),
                    {
                        "terminal_status",
                        "moveit_success",
                        "serialized_trajectory",
                        "final_joint_state",
                    },
                    "PLAN_RESULT",
                )
                if (
                    result["terminal_status"] != "SUCCEEDED"
                    or result["moveit_success"] is not True
                    or not isinstance(result["serialized_trajectory"], bytes)
                    or not result["serialized_trajectory"]
                ):
                    raise ContractError("PLAN_NOT_COMPLETE")
                final_state = _joint_positions(result["final_joint_state"])
                serialized = result["serialized_trajectory"]
                step_type = "ARM"
                duration_reader = getattr(self.transport, "arm_trajectory_duration_s", None)
                if callable(duration_reader):
                    planned_duration_s = duration_reader(serialized)
                    if (
                        not isinstance(planned_duration_s, (int, float))
                        or isinstance(planned_duration_s, bool)
                        or not math.isfinite(planned_duration_s)
                        or planned_duration_s <= 0
                    ):
                        raise ContractError("PLAN_TRAJECTORY_DURATION")
                    if planned_duration_s + EXECUTION_RESULT_MARGIN_S > step["limits"]["execution_timeout_s"]:
                        return _response(
                            code="EXECUTION_TIMEOUT_INSUFFICIENT",
                            run_id=run_id,
                            data={"planning_failure": {
                                "phase": phase,
                                "motion_program_digest": canonical_digest(motion_program),
                                "planned_duration_s": planned_duration_s,
                                "execution_timeout_s": step["limits"]["execution_timeout_s"],
                                "result_margin_s": EXECUTION_RESULT_MARGIN_S,
                            }},
                        )
            else:
                if phase == "GRIPPER_OPEN" and "release_position_m" in step:
                    stage_limits = {
                        **step["limits"],
                        "command_duration_s": step["release_hold_s"],
                    }
                    serialized = self.transport.build_gripper_goal(
                        phase, step["release_position_m"], stage_limits,
                    )
                    continuation = self.transport.build_gripper_goal(
                        phase, step["gripper_position_m"], step["limits"],
                    )
                else:
                    serialized = self.transport.build_gripper_goal(
                        phase, step["gripper_position_m"], step["limits"],
                    )
                if not isinstance(serialized, bytes) or not serialized:
                    raise ContractError("GRIPPER_GOAL")
                if continuation is not None and (
                    not isinstance(continuation, bytes) or not continuation
                ):
                    raise ContractError("GRIPPER_GOAL")
                final_state = state
                step_type = "GRIPPER"

            compiled = {
                "phase": phase,
                "type": step_type,
                "trajectory_b64": base64.b64encode(serialized).decode("ascii"),
                "limits": step["limits"],
                "start_joint_state": state,
                "final_joint_state": final_state,
            }
            if phase == "LEARNED_CHUNK":
                compiled["learned_proposal"] = copy.deepcopy(proposal)
                compiled["gripper_tolerance_m"] = gripper_tolerance
                compiled["max_joint_state_age_s"] = motion_program["planning"]["max_joint_state_age_s"]
                compiled["joint_tolerance_rad"] = tolerance
                compiled["initial_hardware_binding"] = copy.deepcopy(hardware_binding)
                if held_segments:
                    held_segments[0]["initial_hardware_binding"] = copy.deepcopy(hardware_binding)
                    if serialized_references:
                        held_segments[0]["initial_gripper_reference_m"] = float(observed["gripper_controller"]["reference_position_m"])
                        held_segments[0]["gripper_tolerance_m"] = gripper_tolerance
                    compiled["held_target_segments"] = held_segments
            if planned_duration_s is not None:
                compiled["planned_duration_s"] = float(planned_duration_s)
            if step_type == "GRIPPER" and continuation is not None:
                compiled["continuation_trajectory_b64"] = base64.b64encode(
                    continuation
                ).decode("ascii")
            for key in (
                "target",
                "joint_positions_rad",
                "gripper_position_m",
                "release_position_m",
                "release_hold_s",
                "requires_confirmation",
                "pause_after",
            ):
                if key in step:
                    compiled[key] = step[key]
            planned_steps.append(compiled)
            state = final_state

        plan = {
            "schema_version": "fr5.pickup_plan.v3",
            "run_id": run_id,
            "scene_binding": scene_binding,
            "motion_program_digest": canonical_digest(motion_program),
            "action_graph": action_graph,
            "resolved_job_digest": motion_program["resolved_job_digest"],
            "binding_digests": motion_program["binding_digests"],
            "robot_system_id": motion_program["robot_system_id"],
            "frames": motion_program["frames"],
            "planning": motion_program["planning"],
            "execution_timeouts_s": motion_program["execution_timeouts_s"],
            "gripper_requirements": motion_program["gripper_requirements"],
            "active_gripper_settings": settings,
            "initial_joint_state": initial_state,
            "steps": planned_steps,
        }
        if proposal is not None:
            plan["learned_proposal"] = copy.deepcopy(proposal)
            plan["learned_source_program"] = copy.deepcopy(motion_program["source_program"])
            # This is a finite probe, never evidence of a reset or scene transition.
            plan["execution_kind"] = "FINITE_LEARNED_PROBE"
        if "robot_model_trial" in motion_program:
            if proposal is None:
                raise ContractError("LEARNED_ROBOT_MODEL_TRIAL_BINDING")
            plan["robot_model_trial"] = copy.deepcopy(
                motion_program["robot_model_trial"]
            )
        if chunk_binding is not None:
            plan["learned_continuation"] = copy.deepcopy(chunk_binding)
        plan_digest = canonical_digest(plan)
        precommit_response = self.transport.precommit_safety(
            copy.deepcopy(plan), motion_program["planning_scene"], copy.deepcopy(observed)
        )
        precommit_response = _exact(
            precommit_response, {"precommit_safety", "precommit_evidence"}, "PRECOMMIT_EVIDENCE_SCHEMA"
        )
        precommit = precommit_response["precommit_safety"]
        precommit = _exact(
            precommit,
            _PRECOMMIT_SAFETY_FIELDS,
            "PRECOMMIT_SAFETY_SCHEMA",
        )
        if (
            precommit["schema_version"] != "data_factory.precommit_safety.v1"
            or precommit["run_id"] != run_id
            or precommit["approved_plan_digest"] != plan_digest
            or precommit["scene_binding_digest"] != canonical_digest(scene_binding)
            or precommit["expected_planning_scene_digest"]
                != motion_program["binding_digests"]["planning_scene_digest"]
            or precommit["post_reset_safe_snapshot_digest"] is not None
            or precommit["status"] != "PENDING"
            or any(
                not isinstance(precommit[key], str) or not precommit[key].startswith("sha256:")
                for key in (
                    "planning_scene_readback_digest", "collision_report_digest",
                    "plan_only_no_motion_digest",
                )
            )
        ):
            raise ContractError("PRECOMMIT_SAFETY_BINDING")
        evidence = _precommit_evidence(
            precommit_response["precommit_evidence"], precommit,
            run_id=run_id, plan_digest=plan_digest, scene_binding=scene_binding,
            planning_scene_digest=motion_program["binding_digests"]["planning_scene_digest"],
        )
        semantic_steps = [
            step for step in planned_steps
            if step.get("pause_after") in {"SEMANTIC_VERDICT", "LEARNED_CHUNK_COMPLETE"}
        ]
        if len(semantic_steps) != 1:
            raise ContractError("MOTION_PROGRAM_MARKER")
        recording_boundary = semantic_steps[0]["phase"]
        has_precontact_hold = any(
            step.get("requires_confirmation") == "PRECONTACT_HUMAN"
            for step in planned_steps
        )
        operator_summary = {
            "path": [step["phase"] for step in planned_steps],
            "flow": {
                "continuous_through": (
                    "APPROACH_STOP_LIN"
                    if has_precontact_hold else recording_boundary
                ),
                "next_human_hold": (
                    "PRECONTACT_HUMAN"
                    if has_precontact_hold
                    else (
                        "POST_LIFT_SEMANTIC"
                        if recording_boundary == "LIFT_LIN"
                        else "POST_RETREAT_SEMANTIC"
                    )
                ),
            },
            "speed": {
                "max_velocity_scaling": max(
                step["limits"]["velocity_scaling"]
                for step in planned_steps if step["type"] == "ARM"
            ),
                "max_acceleration_scaling": max(
                step["limits"]["acceleration_scaling"]
                for step in planned_steps if step["type"] == "ARM"
            ),
            },
            "clearance": {
                "status": "COLLISION_CHECKED_NO_DISTANCE",
                "collision_report_digest": precommit["collision_report_digest"],
            },
        }
        if proposal is not None:
            operator_summary["flow"] = {"continuous_through": "LEARNED_CHUNK", "next_human_hold": "PRECONTACT_HUMAN"}
            from tools.data_factory.rollout.finite_plan import proposal_summary
            operator_summary["learned"] = proposal_summary(proposal)
        recycle_plan_digest = None
        if proposal is None and "release_slot" in scene_binding:
            recycle_steps = [step for step in planned_steps if step["phase"] in RECYCLE_PHASES]
            recycle_plan_digest = canonical_digest({
                "schema_version": "fr5.recycle_plan.v1",
                "scene_binding": scene_binding,
                "recording_boundary_after": recording_boundary,
                "steps": recycle_steps,
            })
            operator_summary["recycle"] = {
                "recording_boundary_after": recording_boundary,
                "path": list(RECYCLE_PHASES),
                "release_slot_id": scene_binding["release_slot"]["slot_id"],
                "release_target": copy.deepcopy(scene_binding["release_slot"]["pose"]),
                "safe_staging_joint_positions_rad": copy.deepcopy(planned_steps[-1]["final_joint_state"]),
                "plan_digest": recycle_plan_digest,
            }
        return _response(
            code="PLANNED",
            ok=True,
            run_id=run_id,
            plan_digest=plan_digest,
            state="PLANNED",
            data={
                "plan": copy.deepcopy(plan),
                "precommit_safety": copy.deepcopy(precommit),
                "precommit_evidence": copy.deepcopy(evidence),
                "operator_summary": operator_summary,
            },
        )

    def _chunk_boundary(self, run, lease_id):
        self.tick()
        self._check_chunk_boundary(run, lease_id)

    def _check_chunk_boundary(self, run, lease_id):
        self._check_task(run)
        # Pure checks are safe while the Scene lock is held; tick/fault is not.
        if run["state"] != "LEARNED_CHUNK_COMPLETE" or "learned_proposal" not in run["plan"]:
            raise ContractError(run.get("failure_code", "LEARNED_CHUNK_STATE"))
        if lease_id != run["execution"]["lease_id"]:
            raise ContractError("LEASE_BINDING")
        if run["execution"]["active"] or getattr(self.transport, "owns_active_goal", True):
            raise ContractError("ROS_EXEC_ACTIVE")
        if run["cancel_event"].is_set():
            raise ContractError("LEARNED_CANCELLED")
        now = self.monotonic_clock()
        if now >= run["execution"]["lease_deadline"]:
            raise ContractError("HEARTBEAT_TIMEOUT")
        if now > run["execution"]["wait_deadline"]:
            raise ContractError("LEARNED_CHUNK_TIMEOUT")

    def _prepare_next(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "lease_id", "motion_program"}, "LEARNED_NEXT_SCHEMA")
        self._chunk_boundary(run, payload["lease_id"])
        if "task_grant" in run and self._task_termination(run) is not None:
            raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
        run["continuation_requested"] = True
        if "pending_chunk" in run:
            raise ContractError("LEARNED_NEXT_PENDING")
        program = validate_motion_program(copy.deepcopy(payload["motion_program"]))
        proposal, previous = program.get("learned_proposal"), run["plan"]["learned_proposal"]
        if (proposal is None or program.get("source_program") != run["plan"]["learned_source_program"]
                or any(proposal.get(key) != previous.get(key) for key in
                       ("schema_version", "checkpoint", "instruction", "runtime_inputs", "robot_description", "velocity_scaling", "period_s", "max_observation_age_s"))
                or proposal["proposal_digest"] == previous["proposal_digest"]):
            raise ContractError("LEARNED_NEXT_BINDING")
        prepared_source, prepared_steady = self.source_clock(), self.monotonic_clock()
        prior_data = self._execution_data(run)
        prior_data.pop("learned_history", None)
        prior_data.pop("pending_chunk", None)
        previous_chunk = {"plan_envelope": copy.deepcopy(run["envelope"]), "approval": copy.deepcopy(run["approval"]),
                          "state": run["state"], "execution_evidence": prior_data}
        continuation = {"previous_plan_digest": run["digest"], "previous_trace_digest": prior_data["learned_execution"]["trace_digest"],
                        "previous_chunk_digest": canonical_digest(previous_chunk),
                        "chunk_index": len(run.get("learned_history", [])) + 1}
        response = self._compile_plan({"run_id": payload["run_id"], "motion_program": program,
                                       "scene_binding": run["plan"]["scene_binding"]}, chunk_binding=continuation)
        # A late/reentrant compile cannot publish a candidate or revive a stop.
        self._chunk_boundary(run, payload["lease_id"])
        if not response["ok"]:
            raise ContractError(response["code"])
        from tools.data_factory.rollout.finite_plan import check_freshness
        if "generation_context" in proposal:
            self._check_generation({"run_id": payload["run_id"], "motion_program": program,
                                    "scene_binding": run["plan"]["scene_binding"]}, run["digest"])
        else:
            check_freshness(proposal, self.source_clock())
        elapsed = self.monotonic_clock() - prepared_steady
        if elapsed < 0 or ("generation_context" not in proposal and max(prepared_source - stamp for stamp in proposal["source_timestamps_s"].values()) + elapsed > proposal["max_observation_age_s"]):
            raise ContractError("LEARNED_STALE_OBSERVATION")
        run["pending_chunk"] = {**self._planned_record(response), "previous_chunk": previous_chunk}
        if "generation_context" in proposal:
            self._generation["consumed"] = True
        return self._execution_response(run, payload["run_id"], run["digest"], "LEARNED_NEXT_PLANNED")

    def _check_task(self, run):
        grant = run.get("task_grant")
        if grant is None:
            return
        from tools.data_factory.rollout.task_authority import check_grant
        if run.get("task_revoked"):
            raise ContractError("TASK_GRANT_REVOKED")
        check_grant(grant, run["plan"], self.source_clock())
        proposal = run["plan"].get("learned_proposal", {})
        if "generation_context" in proposal and (self.source_clock() < proposal["inference_completed_at_s"]
                or self.monotonic_clock() < proposal["generation_context"]["opened_monotonic_s"]):
            raise ContractError("LEARNED_GENERATION_TIME")
        if self.monotonic_clock() >= run["task_deadline"]:
            raise ContractError("TASK_DEADLINE_EXHAUSTED")

    def _admit_task(self, payload):
        from tools.data_factory.rollout.task_authority import admission
        initial = "grant" in payload
        fields = {"run_id", "plan_digest", "grant"} if initial else {"run_id", "plan_digest", "lease_id", "candidate_plan_digest"}
        run = self._bound(_exact(payload, fields, "TASK_GRANT_SCHEMA"))
        if initial:
            if run["state"] != "PLANNED" or "task_grant" in run:
                raise ContractError("TASK_GRANT_STATE")
            grant = payload["grant"]
            if "generation_context" in run["plan"].get("learned_proposal", {}):
                if run.get("task_revoked"):
                    raise ContractError("TASK_GRANT_REVOKED")
                if grant != run.get("generation_grant"):
                    raise ContractError("TASK_GRANT_SCOPE")
                if self.monotonic_clock() >= run["task_deadline"]:
                    raise ContractError("TASK_DEADLINE_EXHAUSTED")
                proposal = run["plan"]["learned_proposal"]
                if (self.source_clock() < proposal["inference_completed_at_s"]
                        or self.monotonic_clock() < proposal["generation_context"]["opened_monotonic_s"]):
                    raise ContractError("LEARNED_GENERATION_TIME")
            receipt = admission(grant, run["plan"], run["digest"], self.source_clock())
            remaining = min(grant["deadline_s"] - self.source_clock(), run.get("task_deadline", math.inf) - self.monotonic_clock())
            if remaining <= grant["terminal_reserve_s"] + 5.:
                raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
            run.update(task_grant=copy.deepcopy(grant), task_deadline=run.get("task_deadline", self.monotonic_clock() + remaining))
            run["approval"], run["state"] = receipt, "APPROVED"
        else:
            self._chunk_boundary(run, payload["lease_id"])
            self._check_task(run)
            if self._task_termination(run) is not None:
                raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
            candidate = run.get("pending_chunk")
            if not candidate or candidate["digest"] != payload["candidate_plan_digest"] or candidate["state"] != "PLANNED":
                raise ContractError("LEARNED_NEXT_BINDING")
            candidate["approval"] = admission(run["task_grant"], candidate["plan"], candidate["digest"], self.source_clock())
            candidate["state"] = "APPROVED"
        return self._execution_response(run, payload["run_id"], run["digest"], "TASK_PLAN_ADMITTED") if not initial else _response(
            ok=True, code="TASK_PLAN_ADMITTED", state="APPROVED", run_id=payload["run_id"], plan_digest=run["digest"])

    def _task_policy_window(self, run):
        grant = run["task_grant"]
        remaining = min(grant["deadline_s"] - self.source_clock(), run["task_deadline"] - self.monotonic_clock())
        return remaining > grant["terminal_reserve_s"] + 5.

    def _task_termination(self, run):
        grant = run["task_grant"]
        if len(run.get("learned_history", [])) + 1 >= grant["max_outputs"]:
            return "TASK_OUTPUT_LIMIT_REACHED"
        if not self._task_policy_window(run):
            return "TASK_POLICY_BUDGET_EXHAUSTED"
        return None

    def _task_boundary(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "lease_id"}, "TASK_GRANT_SCHEMA")
        self._chunk_boundary(run, payload["lease_id"])
        if "task_grant" not in run:
            raise ContractError("TASK_GRANT_REQUIRED")
        if "mechanical_terminal" in run:
            raise ContractError("MECHANICAL_TERMINAL_BINDING")
        self._check_task(run)
        reason = self._task_termination(run)
        if reason is not None:
            capture = getattr(self.transport, "mechanical_contact_context", None)
            if callable(capture):
                observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
                if _gripper_settings(observed["gripper_settings"]) != run["plan"]["active_gripper_settings"]:
                    raise ContractError("GRIPPER_SETTINGS_MISMATCH")
                options = {}
                if run["execution"].get("prospective_contact", {}).get("status") == "PROSPECTIVE":
                    options["prospective"] = copy.deepcopy(run["execution"]["prospective_contact"])
                contact = capture(copy.deepcopy(run["plan"]), copy.deepcopy(run["execution"]["scene_object"]), observed, **options)
                run["mechanical_contact_diagnostic"] = copy.deepcopy(contact)
                if contact.get("status") == "AVAILABLE":
                    until=contact.get("valid_until_s")
                    if type(until) not in (int,float) or not math.isfinite(until) or self.source_clock() >= until:
                        raise ContractError("MECHANICAL_CONTACT_STALE")
                    from tools.data_factory.motion.mechanical_terminal import compile_terminal
                    terminal = compile_terminal(self.transport, plan=run["plan"], grant=run["task_grant"],
                        snapshot=observed, contact=contact,
                        remaining_s=min(run["task_deadline"] - self.monotonic_clock(),
                                        run["task_grant"]["deadline_s"] - self.source_clock()))
                    self._check_chunk_boundary(run, payload["lease_id"])
                    run["mechanical_terminal"] = {"plan": terminal, "status": "PLANNED", "terminal_phases": []}
                    run["task_handoff"] = {"status": "PLANNED", "termination_reason": reason,
                        "terminal_digest": terminal["terminal_digest"], "deadline_s": terminal["deadline_s"],
                        "semantic_success": "NOT_MEASURED"}
                    return self._execution_response(run, payload["run_id"], run["digest"], "MECHANICAL_TERMINAL_PLANNED")
            run["task_handoff"] = {"schema_version": "data_factory.learned_task_handoff.v1",
                "termination_reason": reason, "status": "BLOCKED_UNAVAILABLE",
                "code": "MECHANICAL_TERMINAL_UNAVAILABLE", "run_id": payload["run_id"],
                "plan_digest": run["digest"], "grant_digest": run["task_grant"]["grant_digest"],
                "deadline_s": run["task_grant"]["deadline_s"],
                "required_dependencies": ["QUALIFIED_MECHANICAL_TERMINAL", "RECORDER_TERMINAL_RETENTION"],
                "semantic_success": "NOT_MEASURED"}
            self._fault(run, "MECHANICAL_TERMINAL_UNAVAILABLE")
        return self._execution_response(run, payload["run_id"], run["digest"], "TASK_CONTINUE")

    def _execute_terminal(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "lease_id", "terminal_digest"}, "MECHANICAL_TERMINAL_SCHEMA")
        self._chunk_boundary(run, payload["lease_id"])
        terminal = run.get("mechanical_terminal")
        if (terminal is None or terminal["status"] != "PLANNED"
                or terminal["plan"]["terminal_digest"] != payload["terminal_digest"]):
            raise ContractError("MECHANICAL_TERMINAL_BINDING")
        self._check_task(run)
        if self.monotonic_clock() + sum(s["limits"]["execution_timeout_s"] for s in terminal["plan"]["steps"]) >= run["task_deadline"]:
            raise ContractError("MECHANICAL_TERMINAL_BUDGET")
        execution = run["execution"]
        execution.pop("reference_deadline", None)
        execution.pop("wait_deadline", None)
        execution["step_index"] = 0
        terminal["status"] = "EXECUTING"
        run["state"] = "EXECUTING"
        self._start_current_step(run)
        return self._execution_response(run, payload["run_id"], run["digest"], "MECHANICAL_TERMINAL_STARTED")

    @staticmethod
    def _active_steps(run):
        terminal = run.get("mechanical_terminal")
        return terminal["plan"]["steps"] if terminal and terminal["status"] != "PLANNED" else run["plan"]["steps"]

    def _capture_mechanical_illumination(self, run):
        from tools.data_factory.motion.mechanical_terminal import check_illumination
        sample = self.transport.capture_scene_illumination(run["plan"])
        # Retain the actual rejected/accepted sample for terminal diagnosis.
        run["mechanical_terminal"]["last_illumination"] = sample
        check_illumination(sample, run["plan"], self.source_clock())

    def _check_mechanical_dispatch(self, run, step):
        self._check_learned_dispatch(run)
        from tools.data_factory.motion.mechanical_terminal import check_illumination
        check_illumination(run["mechanical_terminal"].get("last_illumination"), run["plan"], self.source_clock())
        terminal = run["mechanical_terminal"]["plan"]
        contact = terminal["contact_evidence"]
        if self.source_clock() >= contact["valid_until_s"]:
            raise ContractError("MECHANICAL_CONTACT_STALE")
        cell=self.cell_state_store.read()
        if (cell.get("cell_ready") is not False or cell.get("run_id") != run["plan"]["run_id"]
                or cell.get("plan_digest") != run["digest"] or cell.get("reason_code") != "EXECUTION_IN_PROGRESS"):
            raise ContractError("STATE_CHANGED")
        observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
        if (observed["gripper_settings"] != terminal["initial_snapshot"]["gripper_settings"]
                or any(abs(a-b) > terminal["planning"]["goal_tolerances"]["joint_rad"]
                       for a,b in zip(observed["joint_positions"],step["start_joint_state"]))):
            raise ContractError("MECHANICAL_TERMINAL_STATE")

    def _verify_mechanical_phase(self, run, step, *, partial=False):
        from tools.data_factory.rollout.gripper_evidence import check_hardware, identity
        terminal=run["mechanical_terminal"]["plan"]
        observed=self.transport.snapshot(terminal["planning"]["max_joint_state_age_s"])
        evidence={"captured_at_s":self.source_clock(), "captured_monotonic_s":self.monotonic_clock(), "snapshot":observed}
        wire=check_hardware(evidence,self.source_clock(),self.monotonic_clock(),terminal["planning"]["max_joint_state_age_s"])
        initial=terminal["initial_snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        if identity(wire) != identity(initial):
            raise ContractError("LEARNED_HARDWARE_INCARNATION")
        target=step["release_position_m"] if partial else step.get("gripper_position_m")
        if (any(abs(a-b)>terminal["planning"]["goal_tolerances"]["joint_rad"] for a,b in zip(observed["joint_positions"],step["final_joint_state"]))
                or target is not None and any(abs(observed["gripper_controller"][key]-target)>step["limits"]["completion_tolerance_m"]
                    for key in ("feedback_position_m","reference_position_m"))):
            raise ContractError("MECHANICAL_TERMINAL_STATE")
        run["mechanical_terminal"]["last_observation"]=evidence

    def _revoke_task(self, payload):
        _exact(payload, {"run_id", "grant_digest"}, "TASK_GRANT_SCHEMA")
        run = self.runs.get(payload["run_id"])
        owned = self._generation
        if owned is not None and payload["run_id"] == owned["context"]["run_id"] and payload["grant_digest"] == owned["grant"]["grant_digest"]:
            owned["revoked"] = True
            if run is None:
                return _response(ok=True, code="TASK_GRANT_REVOKED", state="BLOCKED", run_id=payload["run_id"])
        if run is None:
            raise ContractError("TASK_GRANT_SCOPE")
        if run.get("task_grant", run.get("generation_grant", {})).get("grant_digest") != payload["grant_digest"]:
            raise ContractError("TASK_GRANT_SCOPE")
        run["task_revoked"] = True
        if "execution" in run:
            self._fault(run, "TASK_GRANT_REVOKED")
        else:
            run["state"] = "BLOCKED"
            run["failure_code"] = "TASK_GRANT_REVOKED"
        return _response(ok=True, code="TASK_GRANT_REVOKED", state=run["state"], run_id=payload["run_id"], plan_digest=run["digest"])

    def _approve_next(self, payload):
        fields = {"run_id", "plan_digest", "lease_id", "candidate_plan_digest", "approval"}
        run = self._execution_payload(payload, fields, "LEARNED_NEXT_SCHEMA")
        self._chunk_boundary(run, payload["lease_id"])
        candidate = run.get("pending_chunk")
        if not candidate or candidate["digest"] != payload["candidate_plan_digest"] or candidate["state"] != "PLANNED":
            raise ContractError("LEARNED_NEXT_BINDING")
        if "task_grant" in run:
            raise ContractError("TASK_GRANT_ADMISSION_REQUIRED")
        approval = _exact(payload["approval"], {"approval_id", "approved_by", "approval_expiry", "approval_scope"}, "APPROVAL_SCHEMA")
        if any(not isinstance(approval[k], str) or not SAFE_ID.fullmatch(approval[k]) for k in ("approval_id", "approved_by")):
            raise ContractError("APPROVAL_SCHEMA")
        if approval["approval_scope"] != "HUMAN_GATED":
            raise ContractError("LEARNED_HUMAN_APPROVAL_REQUIRED")
        if approval["approval_id"] in [run["approval"]["approval_id"], *[item["approval"]["approval_id"] for item in run.get("learned_history", [])]]:
            raise ContractError("LEARNED_APPROVAL_REUSED")
        _future_timestamp(approval["approval_expiry"], self.clock())
        candidate["approval"] = {**copy.deepcopy(approval), "run_id": payload["run_id"],
                                 "plan_digest": candidate["digest"], "resolved_job_digest": candidate["plan"]["resolved_job_digest"]}
        candidate["state"] = "APPROVED"
        return self._execution_response(run, payload["run_id"], run["digest"], "LEARNED_NEXT_APPROVED")

    def _execute_next(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "lease_id", "candidate_plan_digest"}, "LEARNED_NEXT_SCHEMA")
        self._chunk_boundary(run, payload["lease_id"])
        candidate = run.get("pending_chunk")
        if not candidate or candidate["digest"] != payload["candidate_plan_digest"] or candidate["state"] != "APPROVED":
            raise ContractError("LEARNED_NEXT_NOT_APPROVED")
        if candidate["plan"].get("robot_model_trial") is not None:
            raise ContractError("ROBOT_MODEL_TRIAL_EXECUTION_BLOCKED")
        if "task_grant" not in run:
            _future_timestamp(candidate["approval"]["approval_expiry"], self.clock())
        else:
            self._check_task(run)
            if self._task_termination(run) is not None:
                raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
        if not self.execution_enabled:
            raise ContractError("LIVE_EXECUTION_BLOCKED")
        plan, execution = candidate["plan"], run["execution"]
        if plan["learned_proposal"]["checkpoint"]["runtime"] == "SYNTHETIC_TEST_ONLY" and self.transport.__class__.__module__ == "tools.data_factory.motion.moveit_transport":
            raise ContractError("LEARNED_SYNTHETIC_RUNTIME")
        first = plan["steps"][0]
        first = first.get("held_target_segments", [first])[0]
        observed = self.transport.snapshot(plan["planning"]["max_joint_state_age_s"])
        from tools.data_factory.rollout.finite_plan import check_execution_start, execution_step
        captured = {"captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock(), "snapshot": observed}
        check_execution_start(execution_step(first, plan["learned_proposal"]), captured, self.source_clock(), steady_now=self.monotonic_clock())
        from tools.data_factory.rollout.gripper_evidence import check_transition
        terminal = (execution["learned_segments"][-1]["terminal_observation"]
                    if execution.get("learned_segments") else execution.get("learned_terminal_observation"))
        check_transition(terminal, captured, command=False)
        if _gripper_settings(observed["gripper_settings"]) != plan["active_gripper_settings"]:
            raise ContractError("GRIPPER_SETTINGS_MISMATCH")
        cell = self.cell_state_store.read() if self.cell_state_store is not None else {}
        if (cell.get("robot_system_id") != plan["robot_system_id"] or cell.get("cell_ready") is not False
                or cell.get("reason_code") != "EXECUTION_IN_PROGRESS" or cell.get("run_id") != payload["run_id"]
                or cell.get("plan_digest") != run["digest"]):
            raise ContractError("LEARNED_NEXT_CELL_BINDING")
        if self.scene_state_store is None:
            raise ContractError("SCENE_STATE_REQUIRED")
        try:
            with self.scene_state_store.locked_snapshot(execution["scene_state_digest"], blocking=False) as snapshot:
                if (snapshot["scene_state_digest"] != execution["scene_state_digest"]
                        or snapshot["scene_state"]["revision"] != execution["scene_revision"]
                        or snapshot["scene_state"]["objects"].get(plan["scene_binding"]["object_instance_id"]) != execution["scene_object"]):
                    raise ContractError("SCENE_STATE_CHANGED")
                self._check_chunk_boundary(run, payload["lease_id"])
                # Freeze all prior provenance, including its approval, into the
                # next exact plan's predecessor digest. Never nest earlier history.
                history = [*run.get("learned_history", []), candidate["previous_chunk"]]
                candidate = {key: value for key, value in candidate.items() if key != "previous_chunk"}
                contact = execution.get("prospective_contact")
                if "task_grant" in run:
                    from tools.data_factory.rollout.finite_plan import validate_execution_history
                    validate_execution_history(candidate["plan"], history)
                    if (not contact or contact.get("status") != "PROSPECTIVE"
                            or contact["plan_digest"] != canonical_digest(run["plan"])
                            or contact["scene_binding"] != candidate["plan"]["scene_binding"]
                            or contact["deadline_s"] != run["task_grant"]["deadline_s"]):
                        raise ContractError("CONTACT_CONTINUATION_BINDING")
                    contact = copy.deepcopy(contact)
                    contact["plan_digest"] = canonical_digest(candidate["plan"])
                try:
                    self.cell_state_store.mark_blocked("EXECUTION_IN_PROGRESS", payload["run_id"], candidate["digest"],
                                                       expected_state_digest=canonical_digest(cell), blocking=False)
                except ContractError:
                    raise
                except Exception as exc:
                    raise ContractError("CELL_STATE_ARMING_FAILED") from exc
                self._check_chunk_boundary(run, payload["lease_id"])
                next_execution = {key: copy.deepcopy(execution[key]) for key in
                                  ("lease_id", "lease_deadline", "scene_object", "scene_state_digest", "scene_revision", "phase_event_sequence")}
                for key in ("phase_events_path", "behavior_report_status"):
                    if key in execution:
                        next_execution[key] = execution[key]
                next_execution.update(step_index=0, grasp_verdict=None, semantic_verdict=None, release_verdict=None,
                                      snapshot=None, active=False, terminal_phases=[])
                if contact is not None:
                    next_execution["prospective_contact"] = contact
                task_authority = {key: run[key] for key in ("task_grant", "task_deadline", "task_revoked") if key in run}
                cancel = run["cancel_event"]
                run.clear()
                run.update(candidate, execution=next_execution, cancel_event=cancel, learned_history=history, state="EXECUTING", **task_authority)
        except ContractError:
            # Fault handling may write Scene state. The lock must be released
            # before the sole lease/stop owner processes an expired deadline.
            self.tick()
            raise
        self._start_current_step(run)
        return self._execution_response(run, payload["run_id"], run["digest"], "EXECUTING")

    def _approve(self, payload):
        _exact(
            payload,
            {
                "approval_id",
                "approved_by",
                "run_id",
                "resolved_job_digest",
                "plan_digest",
                "approval_expiry",
                "approval_scope",
            },
            "APPROVAL_SCHEMA",
        )
        if any(
            not isinstance(payload[key], str) or not SAFE_ID.fullmatch(payload[key])
            for key in ("approval_id", "approved_by")
        ):
            raise ContractError("APPROVAL_SCHEMA")
        if payload["approval_scope"] not in {"HUMAN_GATED", "HIL_NUMERIC_PROXY"}:
            raise ContractError("APPROVAL_SCOPE")
        run = self._bound(payload)
        if "generation_context" in run["plan"].get("learned_proposal", {}):
            raise ContractError("TASK_GRANT_ADMISSION_REQUIRED")
        if run["state"] != "PLANNED":
            raise ContractError("APPROVAL_STATE")
        if payload["resolved_job_digest"] != run["plan"]["resolved_job_digest"]:
            raise ContractError("APPROVAL_BINDING")
        _future_timestamp(payload["approval_expiry"], self.clock())
        run["state"] = "APPROVED"
        run["approval"] = dict(payload)
        return _response(
            code="APPROVED",
            ok=True,
            run_id=payload["run_id"],
            plan_digest=payload["plan_digest"],
            state="APPROVED",
        )

    def _execution_payload(self, payload, fields, code):
        _exact(payload, fields, code)
        if not isinstance(payload["run_id"], str) or not SAFE_ID.fullmatch(payload["run_id"]):
            raise ContractError(code)
        return self._bound(payload)

    def _execute(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "lease_id"}, "EXECUTE_SCHEMA")
        if not isinstance(payload["lease_id"], str) or not SAFE_ID.fullmatch(payload["lease_id"]):
            raise ContractError("EXECUTE_SCHEMA")
        if run["state"] != "APPROVED":
            raise ContractError("NOT_APPROVED")
        if run.get("precommit_safety", {}).get("status") != "PENDING":
            raise ContractError("PRECOMMIT_SAFETY_REQUIRED")
        if "task_grant" in run:
            self._check_task(run)
            if not self._task_policy_window(run):
                raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
        else:
            _future_timestamp(run["approval"]["approval_expiry"], self.clock())
        if run["plan"].get("robot_model_trial") is not None:
            return _response(
                code="ROBOT_MODEL_TRIAL_EXECUTION_BLOCKED",
                run_id=payload["run_id"],
                plan_digest=payload["plan_digest"],
                state="APPROVED",
            )
        proposal = run["plan"].get("learned_proposal")
        if proposal is not None:
            if run["approval"]["approval_scope"] not in {"HUMAN_GATED", "SCOPED_TASK_GRANT"}:
                raise ContractError("LEARNED_HUMAN_APPROVAL_REQUIRED")
            if proposal["checkpoint"]["runtime"] == "SYNTHETIC_TEST_ONLY" and self.transport.__class__.__module__ == "tools.data_factory.motion.moveit_transport":
                raise ContractError("LEARNED_SYNTHETIC_RUNTIME")
        if not self.execution_enabled:
            return _response(code="LIVE_EXECUTION_BLOCKED", run_id=payload["run_id"], plan_digest=payload["plan_digest"], state="APPROVED")
        try:
            observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
            settings = _gripper_settings(observed["gripper_settings"])
            if proposal is not None:
                from tools.data_factory.rollout.finite_plan import check_execution_start, execution_step
                first = run["plan"]["steps"][0]
                first = first.get("held_target_segments", [first])[0]
                captured = {"captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock(), "snapshot": observed}
                check_execution_start(execution_step(first, run["plan"]["learned_proposal"]), captured, self.source_clock(), steady_now=self.monotonic_clock())
        except ContractError as exc:
            return _response(code=exc.code, run_id=payload["run_id"], plan_digest=payload["plan_digest"], state="APPROVED")
        except (KeyError, TypeError):
            return _response(code="GRIPPER_SETTINGS_UNVERIFIED", run_id=payload["run_id"], plan_digest=payload["plan_digest"], state="APPROVED")
        if settings != run["plan"]["active_gripper_settings"] or settings["hardware_plugin"] != "fairino_hardware/FairinoHardwareInterface":
            return _response(code="GRIPPER_SETTINGS_MISMATCH", run_id=payload["run_id"], plan_digest=payload["plan_digest"], state="APPROVED")
        if self.cell_state_store is None:
            raise ContractError("CELL_NOT_READY")
        if self.scene_state_store is None:
            raise ContractError("SCENE_STATE_REQUIRED")
        try:
            cell = self.cell_state_store.read()
            if cell["robot_system_id"] != run["plan"]["robot_system_id"]:
                raise ContractError("CELL_NOT_READY")
            if self.motion_only_binding_digest is None:
                if not cell["cell_ready"]:
                    raise ContractError("CELL_NOT_READY")
            elif (
                cell.get("cell_ready") is not False
                or cell.get("run_id") != self.motion_only_parent_run_id
                or cell.get("plan_digest")
                != self.motion_only_parent_plan_digest
            ):
                raise ContractError("MOTION_ONLY_PARENT_CELL")
            binding = run["plan"]["scene_binding"]
            execution_scene_digest = binding["scene_state_digest"]
            execution_scene_revision = binding["revision"]
            source_slot = binding.get("source_slot")
            parent_cell_binding = None
            if source_slot is not None:
                if source_slot["allowed_run_id"] != run["plan"]["run_id"]:
                    raise ContractError("SCENE_SLOT_NEXT_RUN")
                consumed = self.scene_state_store.consume_next_source(
                    slot_id=source_slot["slot_id"], run_id=run["plan"]["run_id"],
                    expected_scene_digest=binding["scene_state_digest"], expected_slot_digest=source_slot["slot_digest"],
                )
                execution_scene_digest = consumed["scene_state_digest"]
                execution_scene_revision = consumed["scene_state"]["revision"]
                if self.motion_only_binding_digest is not None:
                    parent_cell_binding = {
                        "run_id": self.motion_only_parent_run_id,
                        "plan_digest": self.motion_only_parent_plan_digest,
                        "source_slot_id": source_slot["slot_id"],
                        "source_slot_digest": canonical_digest(consumed["scene_state"]["slot_allocations"][source_slot["slot_id"]]),
                    }
            scene_options = {"blocking": False} if proposal is not None else {}
            with self.scene_state_store.locked_snapshot(execution_scene_digest, **scene_options) as snapshot:
                scene = snapshot["scene_state"]
                item = scene["objects"].get(binding["object_instance_id"])
                if snapshot["scene_state_digest"] != execution_scene_digest or scene["revision"] != execution_scene_revision:
                    raise ContractError("SCENE_STATE_CHANGED")
                if not isinstance(item, dict) or item.get("state") != "ON_SURFACE":
                    raise ContractError("SCENE_OBJECT_NOT_READY")
                if self.motion_only_binding_digest is None:
                    try:
                        options = {"blocking": False} if proposal is not None else {}
                        self.cell_state_store.mark_blocked("EXECUTION_IN_PROGRESS", run["plan"]["run_id"], run["digest"], **options)
                    except Exception as exc:
                        if isinstance(exc, ContractError) and exc.code == "STATE_BUSY":
                            raise
                        raise ContractError("CELL_STATE_ARMING_FAILED") from exc
                run["execution"] = {"lease_id": payload["lease_id"], "lease_deadline": self.monotonic_clock() + run["plan"]["execution_timeouts_s"]["heartbeat_lease"], "step_index": 0, "grasp_verdict": None, "semantic_verdict": None, "release_verdict": None, "snapshot": None, "active": False, "scene_object": copy.deepcopy(item), "scene_state_digest": execution_scene_digest, "scene_revision": execution_scene_revision, "terminal_phases": [], "phase_event_sequence": 0}
                if parent_cell_binding is not None:
                    run["execution"]["parent_cell_binding"] = parent_cell_binding
                if self.phase_events_root is not None:
                    path = self.phase_events_root / run["plan"]["run_id"] / "phase_events.jsonl"
                    try:
                        event_plan = run["plan"] if "held_target_segments" in run["plan"]["steps"][0] else None
                        self._phase_event_writer = PhaseEventWriter(path, plan=event_plan)
                        run["execution"]["phase_events_path"] = str(path)
                        run["execution"]["behavior_report_status"] = "PENDING"
                    except Exception:
                        self._phase_event_writer = None
                        run["execution"]["behavior_report_status"] = "BEHAVIOR_REPORT_UNAVAILABLE"
                run["cancel_event"] = threading.Event()
                run["state"] = "EXECUTING"
                if proposal is None:
                    self._start_current_step(run)
            if proposal is not None:
                # Learned start owns its validation-through-send Scene lock.
                # Automatic grant admission must not nest it inside arming;
                # mutations after this release are checked by that same owner.
                self._start_current_step(run)
        except ContractError as exc:
            return _response(code=exc.code, run_id=payload["run_id"], plan_digest=payload["plan_digest"], state="APPROVED")
        except Exception:
            return _response(code="CELL_STATE_ARMING_FAILED", run_id=payload["run_id"], plan_digest=payload["plan_digest"], state="APPROVED")
        return self._execution_response(run, payload["run_id"], payload["plan_digest"], "EXECUTING")

    def _execution_response(self, run, run_id, plan_digest, success_code):
        if run["state"] == "BLOCKED":
            response = _response(code=run["failure_code"], run_id=run_id, plan_digest=plan_digest, state="BLOCKED", data=self._execution_data(run))
        elif run["state"] == "COMPLETED":
            response = _response(code="COMPLETE", ok=True, run_id=run_id, plan_digest=plan_digest, state="COMPLETED", data=self._execution_data(run))
        else:
            response = _response(code=success_code, ok=True, run_id=run_id, plan_digest=plan_digest, state=run["state"], data=self._execution_data(run))
        response["mode"] = self.mode
        return response

    def _execution_data(self, run):
        execution = run["execution"]
        if self._phase_event_writer is not None:
            if self._phase_event_writer.error_code:
                execution["behavior_report_status"] = "BEHAVIOR_REPORT_UNAVAILABLE"
            elif self._phase_event_writer.ready:
                execution["behavior_report_status"] = "AVAILABLE"
        data = {key: copy.deepcopy(execution.get(key)) for key in ("step_index", "grasp_verdict", "semantic_verdict", "release_verdict", "precontact_confirmation", "grasp_decision", "semantic_decision", "release_decision", "gripper_feedback_m", "gripper_reference_m", "post_lift_gripper_feedback_m", "release_evidence", "scene_transition", "snapshot", "snapshot_error", "cancel_error", "durable_blocked", "cell_state_error", "scene_state_error", "phase_events_path", "behavior_report_status") if key in execution}
        if "prospective_contact" in execution:
            data["prospective_contact"] = copy.deepcopy(execution["prospective_contact"])
        if "terminal_verification" in execution:
            data["terminal_verification"] = copy.deepcopy(execution["terminal_verification"])
        if run.get("recycle_plan_digest") is not None:
            data["recycle_plan_digest"] = run["recycle_plan_digest"]
        if "learned_proposal" in run["plan"]:
            proposal = run["plan"]["learned_proposal"]
            trace = {"schema_version": "data_factory.finite_learned_execution.v1",
                     "proposal_digest": proposal["proposal_digest"], "plan_digest": run["digest"],
                     "checkpoint": copy.deepcopy(proposal["checkpoint"]),
                     "status": "FAILED" if "failure_code" in run else "COMPLETED" if run["state"] == "COMPLETED" else "PENDING",
                     "failure_code": run.get("failure_code"),
                     "terminal_state": copy.deepcopy(execution.get("learned_terminal_snapshot")),
                     "terminal_phases": copy.deepcopy(execution["terminal_phases"]),
                     "task_effectiveness": "UNKNOWN", "scene_outcome": "UNKNOWN",
                     "cell_ready": False, "online_policy_authorized": False}
            if "held_target_segments" in run["plan"]["steps"][0]:
                trace["segments"] = copy.deepcopy(execution.get("learned_segments", []))
                if "reference_timing" in proposal:
                    from tools.data_factory.rollout.finite_plan import reference_consumption
                    trace["reference_consumption"] = reference_consumption(run["plan"], trace["segments"])
            else:
                trace["start_observation"] = copy.deepcopy(execution.get("learned_start_observation"))
            if "learned_terminal_observation" in execution:
                trace["terminal_observation"] = copy.deepcopy(execution["learned_terminal_observation"])
            trace["trace_digest"] = canonical_digest(trace)
            data["learned_execution"] = trace
        if "learned_history" in run:
            data["learned_history"] = copy.deepcopy(run["learned_history"])
        if "pending_chunk" in run:
            pending = run["pending_chunk"]
            data["pending_chunk"] = {"plan_digest": pending["digest"], "state": pending["state"],
                                     "plan_envelope": copy.deepcopy(pending["envelope"]),
                                     "previous_chunk": copy.deepcopy(pending["previous_chunk"])}
        if "task_grant" in run:
            data["task_authority"] = {"grant": copy.deepcopy(run["task_grant"]), "admission": copy.deepcopy(run["approval"])}
        if "task_handoff" in run:
            data["task_handoff"] = copy.deepcopy(run["task_handoff"])
        for key in ("mechanical_terminal", "mechanical_contact_diagnostic"):
            if key in run:
                data[key] = copy.deepcopy(run[key])
        data["precommit_safety"] = copy.deepcopy(run.get("precommit_safety"))
        if "failure_code" in run:
            data["failure_code"] = run["failure_code"]
        return data

    def _check_learned_dispatch(self, run, deadlines=None, *, require_contact=False):
        self._check_task(run)
        if "task_grant" in run and "mechanical_terminal" not in run and run["execution"].get("learned_segment_index", 0) == 0 and not self._task_policy_window(run):
            raise ContractError("TASK_POLICY_BUDGET_EXHAUSTED")
        execution = run["execution"]
        if require_contact and "task_grant" in run:
            context, pending = execution.get("prospective_contact", {}), execution.get("contact_pending", {})
            step = self._active_steps(run)[execution["step_index"]]
            step = step.get("held_target_segments", [step])[execution.get("learned_segment_index", 0)]
            if (context.get("status") != "PROSPECTIVE" or context.get("plan_digest") != canonical_digest(run["plan"])
                    or pending.get("segment_digest") != canonical_digest(step)):
                raise ContractError("CONTACT_COVERAGE_UNAVAILABLE")
        if run["cancel_event"].is_set():
            raise ContractError(execution.get("_scene_dispatch_fault", "LEARNED_CANCELLED"))
        if run["state"] != "EXECUTING":
            raise ContractError(run.get("failure_code", "LEARNED_CHUNK_STATE"))
        lease, confirmation = deadlines or self._learned_dispatch_deadlines(run)
        now = self.monotonic_clock()
        if now >= lease:
            raise ContractError("HEARTBEAT_TIMEOUT")
        if confirmation is not None and now > confirmation:
            raise ContractError("PRECONTACT_TIMEOUT")
        if now >= run["execution"].get("reference_deadline", math.inf):
            raise ContractError("LEARNED_REFERENCE_TIMEOUT")
        # Original policy inputs were checked at inference and plan admission.
        # Recorder preparation and command duration do not renew those inputs or
        # invalidate the admitted plan. The native send checks current state.

    @staticmethod
    def _learned_dispatch_deadlines(run):
        execution = run["execution"]
        return (execution["lease_deadline"], execution.get("wait_deadline")
                if execution.get("learned_segment_index", 0) == 0 else None)

    @contextmanager
    def _learned_dispatch_scene(self, run):
        if "learned_proposal" not in run["plan"]:
            yield
            return
        if self.scene_state_store is None:
            raise ContractError("SCENE_STATE_REQUIRED")
        execution = run["execution"]
        # The existing Scene lock orders foreign writes against validation AND
        # goal submission. Reentrant stop callbacks only fence until it releases.
        execution["_scene_dispatch_active"] = True
        try:
            with self.scene_state_store.locked_snapshot(execution["scene_state_digest"], blocking=False) as snapshot:
                scene = snapshot["scene_state"]
                if (snapshot["scene_state_digest"] != execution["scene_state_digest"]
                        or scene["revision"] != execution["scene_revision"]
                        or scene["objects"].get(run["plan"]["scene_binding"]["object_instance_id"]) != execution["scene_object"]):
                    raise ContractError("SCENE_STATE_CHANGED")
                self._check_learned_dispatch(run)
                yield
        finally:
            execution.pop("_scene_dispatch_active", None)

    def _start_current_step(self, run):
        if run["state"] != "EXECUTING" or ("cancel_event" in run and run["cancel_event"].is_set()):
            return run["state"]
        execution, steps = run["execution"], self._active_steps(run)
        if execution["step_index"] >= len(steps) and "mechanical_terminal" in run:
            terminal = run["mechanical_terminal"]
            try:
                terminal["completion_evidence"] = self.transport.complete_mechanical_terminal(terminal["plan"])
                binding = run["plan"]["scene_binding"]
                execution["scene_transition"] = self.scene_state_store.update_object(
                    instance_id=binding["object_instance_id"], object_profile_id=execution["scene_object"]["object_profile_id"],
                    state="UNKNOWN", source="ROBOT_ACTION", updated_by="pickup-executor",
                    expected_revision=execution["scene_revision"])
                terminal["status"] = "COMPLETED"
                run["state"] = "COMPLETED"
            except Exception as exc:
                self._fault(run, exc.code if isinstance(exc, ContractError) else "MECHANICAL_TERMINAL_READBACK")
            return run["state"]
        if execution["step_index"] >= len(steps) and "learned_proposal" in run["plan"]:
            # Controller completion and human review do not establish a safe reset.
            try:
                item = execution["scene_object"]
                binding = run["plan"]["scene_binding"]
                execution["scene_transition"] = self.scene_state_store.update_object(
                    instance_id=binding["object_instance_id"], object_profile_id=item["object_profile_id"],
                    state="UNKNOWN", source="ROBOT_ACTION", updated_by="pickup-executor",
                    expected_revision=execution["scene_revision"],
                )
            except Exception:
                self._fault(run, "LEARNED_SCENE_UNCERTAIN")
                return run["state"]
            run["state"] = "COMPLETED"
            return run["state"]
        if execution["step_index"] >= len(steps):
            try:
                observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
                joints = _joint_positions(observed["joint_positions"])
                tolerance = run["plan"]["planning"]["goal_tolerances"]["joint_rad"]
                safe = steps[-1]["final_joint_state"]
                gripper = observed["gripper_controller"]
                open_step = next(step for step in steps if step["phase"] == "GRIPPER_OPEN")
                open_target = open_step.get("gripper_position_m")
                open_tolerance = open_step["limits"]["completion_tolerance_m"]
                settings = _gripper_settings(observed["gripper_settings"])
                if (
                    not observed["arm_controller"]["ready"]
                    or not gripper["ready"]
                    or settings != run["plan"]["active_gripper_settings"]
                    or any(abs(actual - target) > tolerance for actual, target in zip(joints, safe))
                    or not isinstance(open_target, (int, float))
                    or abs(float(gripper["feedback_position_m"]) - open_target) > open_tolerance
                    or abs(float(gripper["reference_position_m"]) - open_target) > open_tolerance
                ):
                    raise ContractError("POST_RESET_SAFE_SNAPSHOT")
                safety = run["precommit_safety"]
                safety["post_reset_safe_snapshot_digest"] = canonical_digest({
                    "joint_positions": joints,
                    "gripper_feedback_position_m": float(gripper["feedback_position_m"]),
                    "gripper_reference_position_m": float(gripper["reference_position_m"]),
                })
                safety["status"] = "PASS"
                slot = run["plan"]["scene_binding"].get("release_slot")
                if slot is None:
                    run["state"] = "COMPLETED"
                else:
                    terminals = [phase for phase in execution["terminal_phases"] if phase in RECYCLE_PHASES]
                    if terminals != list(RECYCLE_PHASES):
                        raise ContractError("RECYCLE_TERMINAL_EVIDENCE")
                    execution["release_evidence"] = {
                        "schema_version": "data_factory.recycle_release_evidence.v2",
                        "run_id": run["plan"]["run_id"],
                        "plan_digest": run["digest"],
                        "release_slot_id": slot["slot_id"],
                        "expected_scene_state_digest": execution["scene_state_digest"],
                        "expected_scene_revision": execution["scene_revision"],
                        "gripper_reference_m": float(gripper["reference_position_m"]),
                        "gripper_feedback_m": float(gripper["feedback_position_m"]),
                        "terminal_phases": terminals,
                        "post_retreat_snapshot_digest": safety["post_reset_safe_snapshot_digest"],
                        "next_start_tolerance_rad": tolerance,
                        "release_outcome": None,
                        "outcome_source": None,
                        "decided_by": None,
                        "decided_at": None,
                    }
                    run["state"] = "RELEASE_VERDICT"
                    execution["wait_deadline"] = self.monotonic_clock() + run["plan"]["execution_timeouts_s"]["semantic_verdict"]
            except (ContractError, KeyError, TypeError, ValueError):
                self._fault(run, "POST_RESET_SAFE_SNAPSHOT")
            return run["state"]
        step = steps[execution["step_index"]]
        if "task_grant" not in run and step.get("requires_confirmation") == "PRECONTACT_HUMAN" and execution.get("confirmed_step") != execution["step_index"]:
            run["state"] = "PRECONTACT_HUMAN"
            execution["wait_deadline"] = self.monotonic_clock() + run["plan"]["execution_timeouts_s"]["precontact_confirmation"]
            self._emit_phase_event(run, "HOLD_ENTERED", step, None, {"hold": "PRECONTACT_HUMAN", "step": step})
            return "PRECONTACT_HUMAN"
        if "held_target_segments" in step:
            step = step["held_target_segments"][execution.get("learned_segment_index", 0)]
        if step["phase"] == "LEARNED_CHUNK" and "reference_timing" in run["plan"]["learned_proposal"]:
            execution.setdefault("reference_deadline", self.monotonic_clock() + run["plan"]["learned_proposal"]["reference_timing"]["wall_time_limit_s"])
        deadlines = self._learned_dispatch_deadlines(run) if step["phase"] == "LEARNED_CHUNK" else None
        try:
            with self._learned_dispatch_scene(run):
                observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
                if not observed["arm_controller"]["ready"] or not observed["gripper_controller"]["ready"]:
                    raise ContractError("CONTROLLER_NOT_READY")
                expected = step["start_joint_state"]
                actual = _joint_positions(observed["joint_positions"])
                tolerance = run["plan"]["planning"]["goal_tolerances"]["joint_rad"]
                if any(abs(a - b) > tolerance for a, b in zip(actual, expected)):
                    raise ContractError("START_STATE_MISMATCH")
                if "mechanical_terminal" in run:
                    self.transport.check_mechanical_step(run["mechanical_terminal"]["plan"],step,observed)
                if step["phase"] == "LEARNED_CHUNK":
                    from tools.data_factory.rollout.finite_plan import check_execution_start, execution_step
                    evidence = {"captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock(),
                                "snapshot": copy.deepcopy(observed)}
                    resolved_step = execution_step(step, run["plan"]["learned_proposal"])
                    check_execution_start(resolved_step, evidence, self.source_clock(), steady_now=self.monotonic_clock())
                    execution["learned_start_observation"] = evidence
                    if "task_grant" in run:
                        if not hasattr(self.transport, "prepare_contact_transition"):
                            raise ContractError("CONTACT_COVERAGE_UNAVAILABLE")
                        if "prospective_contact" not in execution:
                            execution["prospective_contact"] = self.transport.prepare_contact_transition(
                                run["plan"], execution["scene_object"],
                                first=not run.get("learned_history") and not execution.get("learned_segments"),
                                deadline_s=run["task_grant"]["deadline_s"])
                        context = execution["prospective_contact"]
                        if context.get("status") != "PROSPECTIVE":
                            raise ContractError(context.get("code", "CONTACT_COVERAGE_UNAVAILABLE"))
                        if context.get("status") == "PROSPECTIVE":
                            from tools.data_factory.motion.contact_transition import before
                            execution["contact_pending"] = before(self.transport, run["plan"], step, evidence, context)
                            # Native profile/FK/collision work can exceed the
                            # observation age budget. Read new controller data;
                            # never restamp the sample used for planning checks.
                            fresh = {"snapshot": self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"]),
                                     "captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock()}
                            from tools.data_factory.rollout.gripper_evidence import check_transition
                            check_transition(evidence, fresh, command=False)
                            check_execution_start(resolved_step, fresh, self.source_clock(), steady_now=self.monotonic_clock())
                            if any(abs(a-b) > tolerance for a,b in zip(observed["joint_positions"], fresh["snapshot"]["joint_positions"])):
                                raise ContractError("CONTACT_CHECK_MOVED_ARM")
                            if observed["gripper_controller"]["feedback_position_m"] != fresh["snapshot"]["gripper_controller"]["feedback_position_m"]:
                                raise ContractError("CONTACT_CHECK_MOVED_GRIPPER")
                            execution["contact_pending"]["checked_start_observation"] = evidence
                            execution["contact_pending"]["start_observation"] = fresh
                            evidence = fresh
                            execution["learned_start_observation"] = fresh
                    options = {"start_observation": evidence}
                    if "action_range" in step:
                        if execution.get("learned_segments"):
                            from tools.data_factory.rollout.gripper_evidence import check_transition
                            check_transition(execution["learned_segments"][-1]["terminal_observation"], evidence, command=False)
                if "mechanical_terminal" in run:
                    self._capture_mechanical_illumination(run)
                execution["active"] = True
                self._emit_phase_event(run, "DISPATCH_REQUESTED", step, "REQUESTED", {"step": step})
                if step["phase"] == "LEARNED_CHUNK":
                    self._check_learned_dispatch(run, deadlines, require_contact=True)
                    self.transport.start_phase(resolved_step, cancel_event=run["cancel_event"], cancel_timeout_s=run["plan"]["execution_timeouts_s"]["cancel"],
                                               dispatch_guard=lambda: self._check_learned_dispatch(run, deadlines, require_contact=True), **options)
                    if run["state"] != "EXECUTING" or run["cancel_event"].is_set():
                        return run["state"]
                else:
                    if "mechanical_terminal" in run:
                        self.transport.start_phase(step, cancel_event=run["cancel_event"],
                            cancel_timeout_s=run["plan"]["execution_timeouts_s"]["cancel"],
                            dispatch_guard=lambda: self._check_mechanical_dispatch(run, step))
                    else:
                        self.transport.start_phase(step)
                self._emit_phase_event(run, "GOAL_ACCEPTED", step, "ACCEPTED", {"accepted": True, "step": step})
        except Exception as exc:
            self._fault(run, exc.code if isinstance(exc, ContractError) else "SNAPSHOT_SCHEMA")
        finally:
            if "_scene_dispatch_fault" in execution:
                self._fault(run, execution["_scene_dispatch_fault"])
        return run["state"]

    def _fault(self, run, code):
        self._retire_observation_cache()
        if "cancel_event" in run:
            run["cancel_event"].set()
        execution = run["execution"]
        if execution.get("_scene_dispatch_active"):
            # Native start_phase observes this same cancellation event. Scene
            # writes and sole-owner cancellation finish after the flock releases.
            return execution.setdefault("_scene_dispatch_fault", code)
        code = execution.pop("_scene_dispatch_fault", code)
        if run["state"] == "BLOCKED":
            return run["failure_code"]
        run["failure_code"] = code
        getter = getattr(self.transport, "terminal_verification", None)
        if callable(getter):
            try:
                retained = getter()
                if isinstance(retained, dict):
                    execution["terminal_verification"] = {
                        "plan_digest": run["digest"], "segment_index": execution.get("learned_segment_index", 0),
                        **copy.deepcopy(retained)}
            except Exception:
                pass  # Diagnostic retention must not replace the primary fault.
        if "mechanical_terminal" in run:
            run["mechanical_terminal"].update(status="FAILED", failure_code=code)
        # Fence callbacks before waiting on the sole transport cancellation owner.
        run["state"] = "BLOCKED"
        execution = run["execution"]
        if execution.get("active") and getattr(self.transport, "owns_active_goal", True):
            try:
                self.transport.cancel_active(run["plan"]["execution_timeouts_s"]["cancel"])
                step = self._active_steps(run)[execution["step_index"]]
                if "held_target_segments" in step:
                    step = step["held_target_segments"][execution.get("learned_segment_index", 0)]
                self._emit_phase_event(run, "ACTION_TERMINAL", step, "CANCELLED", {"failure_code": code, "step": step, "terminal_status": "CANCELLED"})
            except Exception as exc:
                execution["cancel_error"] = exc.code if isinstance(exc, ContractError) else "CANCEL_FAILED"
        execution["active"] = False
        try:
            execution["snapshot"] = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
        except Exception as exc:
            execution["snapshot"] = None
            execution["snapshot_error"] = exc.code if isinstance(exc, ContractError) else "SNAPSHOT_FAILED"
        execution["durable_blocked"] = False
        try:
            if self.cell_state_store is not None:
                options = {"blocking": False} if "learned_proposal" in run["plan"] else {}
                if run.get("continuation_requested") or run.get("learned_history"):
                    cell = self.cell_state_store.read()
                    if (cell.get("run_id") != run["plan"]["run_id"] or cell.get("plan_digest") != run["digest"]
                            or cell.get("cell_ready") is not False or cell.get("reason_code") != "EXECUTION_IN_PROGRESS"):
                        raise ContractError("STATE_CHANGED")
                    options["expected_state_digest"] = canonical_digest(cell)
                self.cell_state_store.mark_blocked(code, run["plan"]["run_id"], run["digest"], **options)
                execution["durable_blocked"] = True
            else:
                execution["cell_state_error"] = "CELL_STATE_STORE_MISSING"
        except Exception as exc:
            execution["cell_state_error"] = exc.code if isinstance(exc, ContractError) else "CELL_STATE_WRITE_FAILED"
        try:
            item = execution.get("scene_object")
            binding = run["plan"]["scene_binding"]
            if self.scene_state_store is not None and isinstance(item, dict):
                scene_options = {"blocking": False} if "learned_proposal" in run["plan"] else {}
                slot = binding.get("release_slot")
                if slot is None or "learned_proposal" in run["plan"]:
                    execution["scene_transition"] = self.scene_state_store.update_object(
                        instance_id=binding["object_instance_id"],
                        object_profile_id=item["object_profile_id"],
                        state="UNKNOWN",
                        source="ROBOT_ACTION",
                        updated_by="pickup-executor",
                        expected_revision=execution.get("scene_revision", binding["revision"]), **scene_options,
                    )
                else:
                    snapshot = execution.get("snapshot")
                    gripper = snapshot.get("gripper_controller") if isinstance(snapshot, dict) else None
                    evidence = {
                        "schema_version": "data_factory.recycle_release_evidence.v2",
                        "run_id": run["plan"]["run_id"],
                        "plan_digest": run["digest"],
                        "release_slot_id": slot["slot_id"],
                        "expected_scene_state_digest": execution.get("scene_state_digest", binding["scene_state_digest"]),
                        "expected_scene_revision": execution.get("scene_revision", binding["revision"]),
                        "gripper_reference_m": gripper.get("reference_position_m") if isinstance(gripper, dict) else None,
                        "gripper_feedback_m": gripper.get("feedback_position_m") if isinstance(gripper, dict) else None,
                        "terminal_phases": [phase for phase in execution["terminal_phases"] if phase in RECYCLE_PHASES],
                        "post_retreat_snapshot_digest": canonical_digest(snapshot if isinstance(snapshot, dict) else {"status": "UNAVAILABLE", "failure_code": code}),
                        "next_start_tolerance_rad": run["plan"]["planning"]["goal_tolerances"]["joint_rad"],
                        "release_outcome": "UNCERTAIN",
                        "outcome_source": "EXECUTOR_FAILURE",
                        "decided_by": "pickup-executor",
                        "decided_at": self.clock().isoformat().replace("+00:00", "Z"),
                    }
                    execution["release_evidence"] = evidence
                    execution["scene_transition"] = self.scene_state_store.transition_release(
                        instance_id=binding["object_instance_id"],
                        release_slot=slot,
                        evidence=evidence,
                        updated_by="pickup-executor",
                        expected_digest=execution.get("scene_state_digest", binding["scene_state_digest"]),
                        expected_revision=execution.get("scene_revision", binding["revision"]),
                        allowed_next_run_id=binding.get("allowed_next_run_id"), **scene_options,
                    )
        except Exception as exc:
            execution["scene_state_error"] = exc.code if isinstance(exc, ContractError) else "SCENE_STATE_WRITE_FAILED"
        run["state"] = "BLOCKED"
        return code

    def _verified_gripper_feedback(self, run, required=None, observed=None):
        if observed is None:
            observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
        controller = observed["gripper_controller"]
        feedback = float(controller["feedback_position_m"])
        reference = float(controller["reference_position_m"])
        if required is None:
            required = run["plan"]["gripper_requirements"]
        if (
            not controller["ready"]
            or not math.isfinite(feedback)
            or not math.isfinite(reference)
            or abs(reference - required["command_position_m"]) > 1e-9
            or not required["acceptable_feedback_m"]["min"] <= feedback <= required["acceptable_feedback_m"]["max"]
        ):
            raise ContractError("GRIPPER_FEEDBACK_OUT_OF_RANGE")
        return feedback, reference

    def tick(self):
        if self._ticking:
            return
        self._ticking = True
        try:
            self._tick()
        finally:
            self._ticking = False
            if self._policy_observer is not None:
                binding, _ = self._policy_observer
                run = self.runs.get(binding["run_id"])
                if (run is None or run.get("digest") != binding["plan_digest"]
                        or run["state"] not in {"EXECUTING", "LEARNED_CHUNK_COMPLETE"}
                        or "mechanical_terminal" in run
                        or run.get("cancel_event") is not None and run["cancel_event"].is_set()):
                    self._retire_observation_cache()
                    self._close_policy_observer()

    def _tick(self):
        for run in self.runs.values():
            if run["state"] not in ACTIVE_STATES:
                continue
            try:
                self._check_task(run)
            except ContractError as exc:
                self._fault(run, exc.code)
                continue
            execution, now = run["execution"], self.monotonic_clock()
            if now >= execution["lease_deadline"]:
                self._fault(run, "HEARTBEAT_TIMEOUT")
            elif run["state"] == "EXECUTING" and now >= execution.get("reference_deadline", math.inf):
                self._fault(run, "LEARNED_REFERENCE_TIMEOUT")
            elif run["state"] in {"PRECONTACT_HUMAN", "GRASP_VERDICT", "SEMANTIC_VERDICT", "LEARNED_CHUNK_COMPLETE", "RELEASE_VERDICT"}:
                if now > execution["wait_deadline"]:
                    self._fault(run, {"PRECONTACT_HUMAN": "PRECONTACT_TIMEOUT", "GRASP_VERDICT": "GRASP_VERDICT_TIMEOUT", "SEMANTIC_VERDICT": "SEMANTIC_TIMEOUT", "LEARNED_CHUNK_COMPLETE": "LEARNED_CHUNK_TIMEOUT", "RELEASE_VERDICT": "RELEASE_VERDICT_TIMEOUT"}[run["state"]])
            else:
                try:
                    active = self.transport.poll_active()
                except Exception as exc:
                    code = exc.code if isinstance(exc, ContractError) else "ROS_EXEC_POLL_FAILED"
                    if code == "ROS_EXEC_RESULT_TIMEOUT":
                        step = self._active_steps(run)[execution["step_index"]]
                        if step["phase"] == "GRIPPER_OPEN":
                            code = "GRIPPER_OPEN_TIMEOUT"
                    self._fault(run, code)
                    continue
                if active is not None and getattr(active, "terminal_observation", None) is not None:
                    # Completion releases transport ownership. Preserve its
                    # checked record even if cancellation concurrently fenced
                    # executor advancement; a later fault snapshot is different.
                    execution["terminal_verification"] = {
                        "plan_digest": run["digest"], "segment_index": execution.get("learned_segment_index", 0),
                        "phase": active.phase, "type": active.type, "deadline_monotonic_s": active.deadline,
                        "status": active.terminal_observation_status, "code": active.terminal_observation_code,
                        "action_terminal": copy.deepcopy(active.action_terminal_observation),
                        "checked_observation": copy.deepcopy(active.terminal_observation)}
                if run["state"] != "EXECUTING":
                    continue
                if active is not None:
                    execution["active"] = False
                    completed_step = self._active_steps(run)[execution["step_index"]]
                    continuation = completed_step.get("continuation_trajectory_b64")
                    if "mechanical_terminal" in run:
                        try:
                            self._verify_mechanical_phase(run,completed_step,
                                partial=continuation is not None and not execution.get("continuation_dispatched"))
                        except Exception as exc:
                            self._fault(run,exc.code if isinstance(exc,ContractError) else "MECHANICAL_TERMINAL_STATE")
                            continue
                    if continuation is not None and not execution.get("continuation_dispatched"):
                        continued_step = {**completed_step, "trajectory_b64": continuation}
                        try:
                            if "mechanical_terminal" in run:
                                with self._learned_dispatch_scene(run):
                                    self._capture_mechanical_illumination(run)
                                    self.transport.start_phase(continued_step, cancel_event=run["cancel_event"],
                                        cancel_timeout_s=run["plan"]["execution_timeouts_s"]["cancel"],
                                        dispatch_guard=lambda: self._check_mechanical_dispatch(run, continued_step))
                            else:
                                self.transport.start_phase(continued_step)
                        except Exception as exc:
                            self._fault(
                                run,
                                exc.code if isinstance(exc, ContractError)
                                else "ROS_EXEC_GOAL_FAILED",
                            )
                            continue
                        execution["continuation_dispatched"] = True
                        execution["active"] = True
                        continue
                    execution.pop("continuation_dispatched", None)
                    if completed_step["phase"] == "GRIPPER_OPEN" and "mechanical_terminal" in run:
                        try:
                            run["mechanical_terminal"]["release_readback"] = self.transport.mechanical_release_readback(run["mechanical_terminal"]["plan"])
                        except Exception as exc:
                            self._fault(run, exc.code if isinstance(exc, ContractError) else "MECHANICAL_RELEASE_READBACK")
                            continue
                    if "held_target_segments" in completed_step:
                        index = execution.get("learned_segment_index", 0)
                        segment = completed_step["held_target_segments"][index]
                        terminal_observation = None
                        try:
                            from tools.data_factory.rollout.finite_plan import check_segment_observation, execution_step
                            terminal_observation = copy.deepcopy(getattr(active, "terminal_observation", None))
                            if terminal_observation is None:
                                # Legacy/non-native transports have no checked
                                # witness; native completion returns its exact one.
                                observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
                                terminal_observation = {"captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock(),
                                                        "snapshot": copy.deepcopy(observed)}
                            action_terminal = getattr(active, "action_terminal_observation", None)
                            if action_terminal is not None:
                                terminal_observation["action_terminal"] = copy.deepcopy(action_terminal)
                            # Revalidate the confirmed record at its checked
                            # instant. Next dispatch separately admits current
                            # state; consumer delay cannot age a past result.
                            terminal = check_segment_observation(execution_step(segment, run["plan"]["learned_proposal"]), terminal_observation,
                                                                 terminal_observation["captured_at_s"], terminal=True,
                                                                 steady_now=terminal_observation["captured_monotonic_s"])
                            from tools.data_factory.rollout.gripper_evidence import check_transition
                            check_transition(execution["learned_start_observation"], terminal_observation, command=segment["type"] == "GRIPPER")
                            if "contact_pending" in execution:
                                from tools.data_factory.motion.contact_transition import completed
                                completed(self.transport, run["plan"], segment, terminal_observation,
                                          execution["prospective_contact"], execution.pop("contact_pending"))
                            if run["state"] != "EXECUTING" or run["cancel_event"].is_set():
                                continue
                            # The shared live/trace checker above owns exact raw
                            # tracking and source-calibrated native completion.
                            execution.setdefault("learned_segments", []).append({
                                "segment_index": index, "segment_digest": canonical_digest(segment),
                                "start_observation": execution["learned_start_observation"],
                                "terminal_observation": terminal_observation})
                            execution["learned_terminal_snapshot"] = terminal
                        except Exception as exc:
                            if terminal_observation is not None:
                                execution["terminal_verification"] = {
                                    "plan_digest": run["digest"], "segment_index": index, "status": "REJECTED",
                                    "code": exc.code if isinstance(exc, ContractError) else "LEARNED_TERMINAL_STATE",
                                    "checked_observation": copy.deepcopy(terminal_observation)}
                            self._fault(run, exc.code if isinstance(exc, ContractError) else "LEARNED_TERMINAL_STATE")
                            continue
                        self._emit_phase_event(run, "ACTION_TERMINAL", segment, "SUCCEEDED", {"step": segment, "terminal_status": "SUCCEEDED"})
                        if index + 1 < len(completed_step["held_target_segments"]):
                            execution["learned_segment_index"] = index + 1
                            self._start_current_step(run)
                            continue
                    (run["mechanical_terminal"]["terminal_phases"] if "mechanical_terminal" in run else execution["terminal_phases"]).append(completed_step["phase"])
                    if "held_target_segments" not in completed_step:
                        self._emit_phase_event(run, "ACTION_TERMINAL", completed_step, "SUCCEEDED", {"step": completed_step, "terminal_status": "SUCCEEDED"})
                    if completed_step["phase"] in {"GRIPPER_CLOSE", "LIFT_LIN"}:
                        try:
                            feedback, reference = self._verified_gripper_feedback(run)
                            if completed_step["phase"] == "GRIPPER_CLOSE":
                                execution["gripper_feedback_m"] = feedback
                                execution["gripper_reference_m"] = reference
                            else:
                                execution["post_lift_gripper_feedback_m"] = feedback
                        except (ContractError, KeyError, TypeError, ValueError):
                            self._fault(run, "GRIPPER_FEEDBACK_OUT_OF_RANGE")
                            continue
                    if completed_step["phase"] == "LEARNED_CHUNK" and "held_target_segments" not in completed_step:
                        try:
                            observed = self.transport.snapshot(run["plan"]["planning"]["max_joint_state_age_s"])
                            joints = _joint_positions(observed["joint_positions"])
                            gripper = observed["gripper_controller"]
                            terminal = [*joints, float(gripper["feedback_position_m"])]
                            target = run["plan"]["learned_proposal"]["actions"][-1]
                            tolerance = run["plan"]["planning"]["goal_tolerances"]["joint_rad"]
                            if (not observed["arm_controller"]["ready"] or not gripper["ready"]
                                    or any(abs(a - b) > tolerance for a, b in zip(terminal[:6], target[:6]))
                                    or any(not math.isfinite(float(gripper[k])) or abs(float(gripper[k]) - target[-1]) > completed_step["gripper_tolerance_m"] for k in ("feedback_position_m", "reference_position_m"))):
                                raise ContractError("LEARNED_TERMINAL_STATE")
                            from tools.data_factory.rollout.finite_plan import _execution_state
                            from tools.data_factory.rollout.gripper_evidence import check_hardware, identity
                            captured = {"captured_at_s": self.source_clock(), "captured_monotonic_s": self.monotonic_clock(), "snapshot": copy.deepcopy(observed)}
                            _execution_state(completed_step, captured, self.source_clock())
                            wire = check_hardware(captured, self.source_clock(), self.monotonic_clock(), completed_step["max_joint_state_age_s"])
                            if identity(wire) != completed_step["initial_hardware_binding"]["incarnation"]:
                                raise ContractError("LEARNED_HARDWARE_INCARNATION")
                            if run["state"] != "EXECUTING" or run["cancel_event"].is_set():
                                continue
                            execution["learned_terminal_snapshot"] = terminal
                            execution["learned_terminal_observation"] = captured
                        except Exception as exc:
                            self._fault(run, exc.code if isinstance(exc, ContractError) else "LEARNED_TERMINAL_STATE")
                            continue
                    execution["step_index"] += 1
                    pause_after = self._active_steps(run)[execution["step_index"] - 1].get("pause_after")
                    if pause_after in {"GRASP_VERDICT", "SEMANTIC_VERDICT", "LEARNED_CHUNK_COMPLETE"}:
                        run["state"] = pause_after
                        execution["wait_deadline"] = now + run["plan"]["execution_timeouts_s"]["semantic_verdict" if pause_after == "LEARNED_CHUNK_COMPLETE" else pause_after.lower()]
                        self._emit_phase_event(run, "HOLD_ENTERED", completed_step, None, {"hold": pause_after, "step": completed_step})
                    else:
                        self._start_current_step(run)

    def _heartbeat(self, payload):
        if self.motion_only_binding_digest is None:
            run = self._execution_payload(
                payload,
                {"run_id", "plan_digest", "lease_id", "recorder_health"},
                "HEARTBEAT_SCHEMA",
            )
            health = _exact(
                payload["recorder_health"],
                {"writer_alive", "writer_error"}, "HEARTBEAT_SCHEMA",
            )
            alive, error, failure_code = (
                health["writer_alive"], health["writer_error"],
                "RECORDER_WRITER_FAULT",
            )
        else:
            run = self._execution_payload(
                payload,
                {"run_id", "plan_digest", "lease_id", "motion_owner_health"},
                "HEARTBEAT_SCHEMA",
            )
            health = _exact(
                payload["motion_owner_health"],
                {
                    "owner_alive", "owner_error", "recording_scope",
                    "object_reposition_binding_digest",
                },
                "HEARTBEAT_SCHEMA",
            )
            if (
                health.get("recording_scope") != "OUT_OF_DATASET"
                or health.get("object_reposition_binding_digest")
                != self.motion_only_binding_digest
            ):
                raise ContractError("MOTION_ONLY_BINDING")
            alive, error, failure_code = (
                health["owner_alive"], health["owner_error"],
                "MOTION_OWNER_FAULT",
            )
        if not isinstance(payload["lease_id"], str) or not SAFE_ID.fullmatch(payload["lease_id"]) or payload["lease_id"] != run.get("execution", {}).get("lease_id"):
            raise ContractError("LEASE_BINDING")
        if type(alive) is not bool or error is not None and not isinstance(error, str):
            raise ContractError("HEARTBEAT_SCHEMA")
        if run["state"] in {"BLOCKED", "COMPLETED"}:
            return self._execution_response(run, payload["run_id"], payload["plan_digest"], "HEARTBEAT_OK")
        if run["state"] not in ACTIVE_STATES:
            raise ContractError("EXECUTION_STATE")
        if not alive or error is not None:
            self._fault(run, failure_code)
            return self._execution_response(run, payload["run_id"], payload["plan_digest"], "HEARTBEAT_OK")
        run["execution"]["lease_deadline"] = self.monotonic_clock() + run["plan"]["execution_timeouts_s"]["heartbeat_lease"]
        return self._execution_response(run, payload["run_id"], payload["plan_digest"], "HEARTBEAT_OK")

    def _confirm(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "confirmed_by", "source"}, "CONFIRM_SCHEMA")
        if (
            payload["source"] not in {"HUMAN", "CAMPAIGN_AUTHORIZATION"}
            or payload["source"] == "CAMPAIGN_AUTHORIZATION"
            and run["approval"]["approval_scope"] != "HIL_NUMERIC_PROXY"
            or not isinstance(payload["confirmed_by"], str)
            or not SAFE_ID.fullmatch(payload["confirmed_by"])
        ):
            raise ContractError("CONFIRM_SCHEMA")
        if run["state"] != "PRECONTACT_HUMAN":
            raise ContractError("CONFIRM_STATE")
        run["execution"]["precontact_confirmation"] = {"source": payload["source"], "decided_by": payload["confirmed_by"], "decided_at": self.clock().isoformat().replace("+00:00", "Z")}
        self._emit_phase_event(run, "DECISION_RECEIVED", run["plan"]["steps"][run["execution"]["step_index"]], None, {"decision": "PRECONTACT_HUMAN", **run["execution"]["precontact_confirmation"]})
        run["execution"]["confirmed_step"] = run["execution"]["step_index"]
        run["state"] = "EXECUTING"
        self._start_current_step(run)
        return self._execution_response(run, payload["run_id"], payload["plan_digest"], "CONFIRMED")

    def _semantic_verdict(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "verdict", "decided_by", "source"}, "VERDICT_SCHEMA")
        if payload["source"] not in {"HUMAN", "HIL_PROXY"} or payload["verdict"] not in {"PASS", "FAIL"} or not isinstance(payload["decided_by"], str) or not SAFE_ID.fullmatch(payload["decided_by"]):
            raise ContractError("VERDICT_SCHEMA")
        if payload["source"] == "HIL_PROXY" and run["approval"]["approval_scope"] != "HIL_NUMERIC_PROXY":
            raise ContractError("VERDICT_SOURCE")
        if run["state"] not in {"SEMANTIC_VERDICT", "LEARNED_CHUNK_COMPLETE"}:
            raise ContractError("VERDICT_STATE")
        run["execution"]["semantic_verdict"] = payload["verdict"]
        run["execution"]["semantic_decision"] = {"source": payload["source"], "decided_by": payload["decided_by"], "decided_at": self.clock().isoformat().replace("+00:00", "Z")}
        if run["state"] == "LEARNED_CHUNK_COMPLETE":
            run["execution"]["semantic_decision"]["review_scope"] = "FINITE_LEARNED_CHUNK"
        self._emit_phase_event(run, "DECISION_RECEIVED", run["plan"]["steps"][run["execution"]["step_index"] - 1], None, {"decision": "SEMANTIC_VERDICT", "verdict": payload["verdict"], **run["execution"]["semantic_decision"]})
        run["state"] = "EXECUTING"
        self._start_current_step(run)
        return self._execution_response(run, payload["run_id"], payload["plan_digest"], "VERDICT_ACCEPTED")

    def _grasp_verdict(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "verdict", "decided_by", "source"}, "GRASP_VERDICT_SCHEMA")
        if payload["source"] not in {"HUMAN", "HIL_PROXY"} or payload["verdict"] not in {"PASS", "FAIL"} or not isinstance(payload["decided_by"], str) or not SAFE_ID.fullmatch(payload["decided_by"]):
            raise ContractError("GRASP_VERDICT_SCHEMA")
        if payload["source"] == "HIL_PROXY" and run["approval"]["approval_scope"] != "HIL_NUMERIC_PROXY":
            raise ContractError("GRASP_VERDICT_SOURCE")
        if run["state"] != "GRASP_VERDICT":
            raise ContractError("GRASP_VERDICT_STATE")
        run["execution"]["grasp_verdict"] = payload["verdict"]
        run["execution"]["grasp_decision"] = {"source": payload["source"], "decided_by": payload["decided_by"], "decided_at": self.clock().isoformat().replace("+00:00", "Z")}
        self._emit_phase_event(run, "DECISION_RECEIVED", run["plan"]["steps"][run["execution"]["step_index"] - 1], None, {"decision": "GRASP_VERDICT", "verdict": payload["verdict"], **run["execution"]["grasp_decision"]})
        if payload["verdict"] == "FAIL":
            self._fault(run, "GRASP_REJECTED")
        else:
            run["state"] = "EXECUTING"
            self._start_current_step(run)
        return self._execution_response(run, payload["run_id"], payload["plan_digest"], "GRASP_VERDICT_ACCEPTED")

    def _release_verdict(self, payload):
        run = self._execution_payload(payload, {"run_id", "plan_digest", "verdict", "decided_by", "source"}, "RELEASE_VERDICT_SCHEMA")
        if (
            payload["source"] not in {
                "HUMAN", "LOCAL_UI_BUTTON", "CAMPAIGN_CONTROL_PROXY",
            }
            or payload["verdict"] not in {"LANDED", "OFF_SLOT", "UNCERTAIN"}
            or payload["source"] == "CAMPAIGN_CONTROL_PROXY"
            and (
                payload["verdict"] != "LANDED"
                or run["approval"]["approval_scope"] != "HIL_NUMERIC_PROXY"
            )
            or not isinstance(payload["decided_by"], str)
            or not SAFE_ID.fullmatch(payload["decided_by"])
            or run["state"] != "RELEASE_VERDICT"
        ):
            raise ContractError("RELEASE_VERDICT_SCHEMA")
        execution = run["execution"]
        evidence = copy.deepcopy(execution.get("release_evidence"))
        if (
            not isinstance(evidence, dict)
            or evidence.get("schema_version")
            != "data_factory.recycle_release_evidence.v2"
        ):
            raise ContractError("RELEASE_EVIDENCE")
        evidence["release_outcome"] = (
            "EXPECTED_LANDED"
            if payload["source"] == "CAMPAIGN_CONTROL_PROXY"
            else payload["verdict"]
        )
        evidence["outcome_source"] = {
            "HUMAN": "HUMAN_TTY",
            "LOCAL_UI_BUTTON": "LOCAL_UI_BUTTON",
            "CAMPAIGN_CONTROL_PROXY": "CAMPAIGN_CONTROL_PROXY",
        }[payload["source"]]
        evidence["decided_by"] = payload["decided_by"]
        evidence["decided_at"] = self.clock().isoformat().replace("+00:00", "Z")
        execution["release_verdict"] = payload["verdict"]
        execution["release_decision"] = {
            "source": payload["source"],
            "decided_by": payload["decided_by"],
            "decided_at": self.clock().isoformat().replace("+00:00", "Z"),
        }
        try:
            transition = self.scene_state_store.transition_release(
                instance_id=run["plan"]["scene_binding"]["object_instance_id"],
                release_slot=run["plan"]["scene_binding"]["release_slot"],
                evidence=evidence,
                updated_by="pickup-executor",
                expected_digest=run["execution"].get("scene_state_digest", run["plan"]["scene_binding"]["scene_state_digest"]),
                expected_revision=run["execution"].get("scene_revision", run["plan"]["scene_binding"]["revision"]),
                allowed_next_run_id=run["plan"]["scene_binding"].get("allowed_next_run_id"),
                **({"parent_cell_binding": execution["parent_cell_binding"]}
                   if "parent_cell_binding" in execution else {}),
            )
            execution["release_evidence"] = evidence
            execution["scene_transition"] = transition
        except ContractError as exc:
            return self._block_release(run, exc.code)
        except Exception:
            return self._block_release(run, "SCENE_TRANSITION_WRITE_FAILED")
        if payload["verdict"] == "LANDED":
            run["state"] = "COMPLETED"
            return self._execution_response(run, payload["run_id"], payload["plan_digest"], "RELEASE_CONFIRMED")
        return self._block_release(run, "RELEASE_OFF_SLOT" if payload["verdict"] == "OFF_SLOT" else "RELEASE_UNCONFIRMED")

    def _block_release(self, run, code):
        run["failure_code"] = code
        execution = run["execution"]
        execution["durable_blocked"] = False
        try:
            if self.cell_state_store is None:
                raise ContractError("CELL_STATE_STORE_MISSING")
            self.cell_state_store.mark_blocked(code, run["plan"]["run_id"], run["digest"])
            execution["durable_blocked"] = True
        except Exception as exc:
            execution["cell_state_error"] = exc.code if isinstance(exc, ContractError) else "CELL_STATE_WRITE_FAILED"
        run["state"] = "BLOCKED"
        return self._execution_response(run, run["plan"]["run_id"], run["digest"], code)

    def _cancel(self, payload):
        if isinstance(payload, dict) and set(payload) == {"run_id", "generation_id"}:
            owned = self._generation
            if owned is None or any(payload[k] != owned["context"][k] for k in payload):
                raise ContractError("LEARNED_GENERATION_CONTEXT")
            owned["cancelled"] = True
            run = self.runs.get(payload["run_id"])
            if run is not None:
                if "execution" in run:
                    self._fault(run, "LEARNED_CANCELLED")
                else:
                    run.update(state="BLOCKED", failure_code="LEARNED_CANCELLED")
            return _response(ok=True, code="LEARNED_CANCELLED", state="BLOCKED", run_id=payload["run_id"])
        run = self._execution_payload(payload, {"run_id", "plan_digest", "lease_id"}, "CANCEL_SCHEMA")
        if not isinstance(payload["lease_id"], str) or not SAFE_ID.fullmatch(payload["lease_id"]) or payload["lease_id"] != run.get("execution", {}).get("lease_id"):
            raise ContractError("LEASE_BINDING")
        if run["state"] in {"BLOCKED", "COMPLETED"}:
            return self._execution_response(run, payload["run_id"], payload["plan_digest"], "CANCELLED_BY_OPERATOR")
        self._fault(run, "CANCELLED_BY_OPERATOR")
        return self._execution_response(run, payload["run_id"], payload["plan_digest"], "CANCELLED_BY_OPERATOR")

    def _status(self, payload):
        _exact(payload, {"run_id", "plan_digest"}, "STATUS_SCHEMA")
        run = self._bound(payload)
        data = self._execution_data(run) if "execution" in run else None
        return _response(
            code="STATUS",
            ok=True,
            run_id=payload["run_id"],
            plan_digest=payload["plan_digest"],
            state=run["state"], data=data,
        )

    def _bound(self, payload):
        try:
            run = self.runs[payload["run_id"]]
        except (KeyError, TypeError) as exc:
            raise ContractError("RUN_NOT_FOUND") from exc
        if payload["plan_digest"] != run["digest"]:
            raise ContractError("PLAN_DIGEST_MISMATCH")
        return run


def run_jsonl(input_stream, output_stream, executor):
    """Keep ticking while stdin is quiet so a lease cannot be bypassed."""
    events = queue.Queue()
    terminal_ok = None
    def read_lines():
        try:
            for line in input_stream:
                events.put(("line", line))
            events.put(("eof", None))
        except Exception:
            events.put(("error", None))
    threading.Thread(target=read_lines, daemon=True).start()
    def terminal(run):
        executor.close()
        result = executor._execution_response(run, run["plan"]["run_id"], run["digest"], "TERMINAL")
        output_stream.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        output_stream.flush()
        return result["state"] == "COMPLETED"
    while True:
        try:
            # The same owner polls opted-in short reference segments at the
            # frozen 100 Hz cadence. Legacy Collection keeps its existing tick.
            from tools.data_factory.rollout.finite_plan import REFERENCE_TICK_S
            timeout = REFERENCE_TICK_S if any(
                run["state"] == "EXECUTING" and "reference_deadline" in run.get("execution", {})
                for run in executor.runs.values()) else .05
            kind, value = events.get(timeout=timeout)
        except queue.Empty:
            executor.tick()
            continue
        if kind == "line":
            try:
                result = executor.process(load_json_strict(value))
            except ContractError as exc:
                result = _response(code=exc.code, mode=executor.mode)
            if result["state"] in {"BLOCKED", "COMPLETED"} and result.get("run_id") in executor.runs:
                executor.close()
                run = executor.runs[result["run_id"]]
                result["data"] = executor._execution_data(run)
            output_stream.write(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            output_stream.flush()
            if result["state"] in {"BLOCKED", "COMPLETED"}:
                terminal_ok = result["state"] == "COMPLETED"
                continue
            executor.tick()
            for run in executor.runs.values():
                if run["state"] == "BLOCKED":
                    return terminal(run)
            continue
        for run in executor.runs.values():
            if run["state"] in ACTIVE_STATES:
                executor._fault(run, "INPUT_READER_ERROR" if kind == "error" else "INPUT_EOF")
                return terminal(run)
        return terminal_ok if terminal_ok is not None else not any(run["state"] == "BLOCKED" for run in executor.runs.values())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory-jsonl", action="store_true")
    ros_mode = parser.add_mutually_exclusive_group()
    ros_mode.add_argument("--ros-plan-only", action="store_true")
    ros_mode.add_argument("--ros-live", action="store_true")
    parser.add_argument("--robot-system-id")
    parser.add_argument("--cell-state-root")
    parser.add_argument("--phase-events-root")
    hardware_binding = parser.add_mutually_exclusive_group()
    hardware_binding.add_argument("--gripper-source-clock", help="Read a measured same-incarnation clock binding for learned held targets")
    hardware_binding.add_argument("--gripper-temporal-policy", help="Read an explicit same-incarnation live causal-evidence policy")
    parser.add_argument("--motion-only-binding-digest")
    parser.add_argument("--motion-only-parent-run-id")
    parser.add_argument("--motion-only-parent-plan-digest")
    parser.add_argument("--motion-only-preapproval-scope-digest")
    parser.add_argument("--motion-only-expected-run-id")
    parser.add_argument("--motion-only-expected-resolved-job-digest")
    parser.add_argument("--motion-only-expected-program-digest")
    parser.add_argument("--motion-only-expected-scene-digest")
    parser.add_argument("--motion-only-expectation-digest")
    args = parser.parse_args(argv)
    if not args.factory_jsonl:
        parser.error("--factory-jsonl required")
    if args.ros_live:
        if not args.robot_system_id or not args.cell_state_root or not args.phase_events_root:
            parser.error("--ros-live requires --robot-system-id, --cell-state-root, and --phase-events-root")
        try:
            if __package__ in (None, ""):
                from tools.data_factory.cell_state import CellStateStore
                from tools.data_factory.scene_state import SceneStateStore
            else:
                from ..cell_state import CellStateStore
                from ..scene_state import SceneStateStore
            cell_state_store = CellStateStore(args.cell_state_root, args.robot_system_id)
            scene_state_store = SceneStateStore(args.cell_state_root, args.robot_system_id)
        except ContractError as exc:
            parser.error(exc.code)
    elif (
        args.robot_system_id or args.cell_state_root or args.phase_events_root
        or args.motion_only_binding_digest
        or args.motion_only_parent_run_id
        or args.motion_only_parent_plan_digest
        or args.motion_only_preapproval_scope_digest
        or args.motion_only_expected_run_id
        or args.motion_only_expected_resolved_job_digest
        or args.motion_only_expected_program_digest
        or args.motion_only_expected_scene_digest
        or args.motion_only_expectation_digest
    ):
        parser.error("--robot-system-id, --cell-state-root, and --phase-events-root require --ros-live")
    else:
        cell_state_store = None
        scene_state_store = None
    gripper_source_clock = None
    gripper_temporal_policy = None
    if args.gripper_source_clock or args.gripper_temporal_policy:
        if not (args.ros_live or args.ros_plan_only):
            parser.error("gripper evidence configuration requires a ROS transport")
        try:
            if args.gripper_temporal_policy:
                from tools.data_factory.rollout.gripper_evidence import validate_temporal_policy
                with open(args.gripper_temporal_policy, encoding="utf-8") as stream:
                    gripper_temporal_policy = validate_temporal_policy(json.load(stream))
            else:
                from tools.data_factory.rollout.gripper_evidence import validate_clock_binding
                with open(args.gripper_source_clock, encoding="utf-8") as stream:
                    gripper_source_clock = validate_clock_binding(json.load(stream))
        except (OSError, ValueError, ContractError) as exc:
            parser.error(str(exc))
    transport = None
    node = None
    rclpy = None
    if args.ros_plan_only or args.ros_live:
        os.environ["RCUTILS_LOGGING_USE_STDOUT"] = "0"
        try:
            import rclpy
            if __package__ in (None, ""):
                from tools.data_factory.motion.moveit_transport import RosMoveItTransport
            else:
                from .moveit_transport import RosMoveItTransport
            rclpy.init()
            node = rclpy.create_node(
                "fr5_object_reposition_live"
                if args.motion_only_binding_digest
                else "fr5_pickup_live" if args.ros_live
                else "fr5_pickup_plan_only"
            )
            transport = RosMoveItTransport(
                node, allow_clock_configuration=args.ros_live,
                **({"gripper_source_clock": gripper_source_clock}
                   if gripper_source_clock is not None else {}),
                **({"gripper_temporal_policy": gripper_temporal_policy}
                   if gripper_temporal_policy is not None else {}),
            )
        except (ContractError, ImportError, RuntimeError) as exc:
            print(
                json.dumps(
                    {"error": {"code": "ROS_LIVE_UNAVAILABLE" if args.ros_live else "ROS_PLAN_ONLY_UNAVAILABLE", "message": str(exc)}},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                if rclpy is not None and rclpy.ok():
                    rclpy.shutdown()
            return 2
    executor = PickupExecutor(
        transport,
        cell_state_store=cell_state_store,
        scene_state_store=scene_state_store,
        execution_enabled=args.ros_live,
        phase_events_root=args.phase_events_root,
        event_clock=(lambda: (node.get_clock().now().nanoseconds, "ROS_TIME")) if node is not None else None,
        motion_only_binding_digest=args.motion_only_binding_digest,
        motion_only_parent_run_id=args.motion_only_parent_run_id,
        motion_only_parent_plan_digest=args.motion_only_parent_plan_digest,
        motion_only_preapproval_scope_digest=(
            args.motion_only_preapproval_scope_digest
        ),
        motion_only_expected_run_id=args.motion_only_expected_run_id,
        motion_only_expected_resolved_job_digest=(
            args.motion_only_expected_resolved_job_digest
        ),
        motion_only_expected_program_digest=(
            args.motion_only_expected_program_digest
        ),
        motion_only_expected_scene_digest=(
            args.motion_only_expected_scene_digest
        ),
        motion_only_expectation_digest=args.motion_only_expectation_digest,
    )
    try:
        return 0 if run_jsonl(sys.stdin, sys.stdout, executor) else 2
    finally:
        try:
            executor.close()
        finally:
            try:
                if node is not None:
                    node.destroy_node()
            finally:
                if rclpy is not None and rclpy.ok():
                    rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
