"""Strict phase-event sidecar contract; it never controls robot motion."""
from __future__ import annotations

import json
import copy
import os
import queue
import threading
from pathlib import Path
from typing import Any, Mapping

from tools.fr5_data_factory import ContractError, DIGEST, SAFE_ID, canonical_digest, load_json_strict


SCHEMA_VERSION = "data_factory.phase_event.v1"
EVENTS = frozenset({"DISPATCH_REQUESTED", "GOAL_ACCEPTED", "ACTION_TERMINAL", "HOLD_ENTERED", "DECISION_RECEIVED"})
ACTION_EVENTS = frozenset({"DISPATCH_REQUESTED", "GOAL_ACCEPTED", "ACTION_TERMINAL"})
EVENT_KEYS = frozenset({"schema_version", "run_id", "plan_digest", "sequence", "phase", "segment_index", "segment_count", "event", "event_ros_time_ns", "monotonic_time_ns", "ros_clock_type", "event_source", "action_status", "evidence_digest"})
QUEUE_CAPACITY = 64
MAX_LINE_BYTES = 4096
MAX_TEXT_BYTES = 128


def _integer(value: Any, code: str, *, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContractError(code)
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ContractError(code)
    return value


def writer_resource_contract() -> dict[str, int]:
    return {"queue_capacity": QUEUE_CAPACITY, "max_line_bytes": MAX_LINE_BYTES, "max_text_bytes": MAX_TEXT_BYTES}


def validate_phase_event(value: Mapping[str, Any], *, plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return one exact, immutable-by-convention phase-event record."""
    if not isinstance(value, Mapping) or set(value) != EVENT_KEYS:
        raise ContractError("PHASE_EVENT_FIELDS")
    event = _text(value["event"], "PHASE_EVENT_EVENT")
    if event not in EVENTS:
        raise ContractError("PHASE_EVENT_EVENT")
    run_id = _text(value["run_id"], "PHASE_EVENT_RUN_ID")
    if not SAFE_ID.fullmatch(run_id):
        raise ContractError("PHASE_EVENT_RUN_ID")
    plan_digest = _text(value["plan_digest"], "PHASE_EVENT_PLAN_DIGEST")
    evidence_digest = _text(value["evidence_digest"], "PHASE_EVENT_EVIDENCE_DIGEST")
    if not DIGEST.fullmatch(plan_digest) or not DIGEST.fullmatch(evidence_digest):
        raise ContractError("PHASE_EVENT_DIGEST")
    index, count = value["segment_index"], value["segment_count"]
    if event in {"HOLD_ENTERED", "DECISION_RECEIVED"}:
        if index is not None or count is not None:
            raise ContractError("PHASE_EVENT_SEGMENT")
    else:
        _integer(index, "PHASE_EVENT_SEGMENT")
        _integer(count, "PHASE_EVENT_SEGMENT", minimum=1)
        if index >= count:
            raise ContractError("PHASE_EVENT_SEGMENT")
        if index != 0 or count != 1:
            if plan is None:
                raise ContractError("PHASE_EVENT_PLAN_REQUIRED")
            if value["phase"] != "LEARNED_CHUNK" or value["event_source"] != "pickup_executor":
                raise ContractError("PHASE_EVENT_SEGMENT")
    if plan is not None:
        if not isinstance(plan, Mapping) or plan_digest != canonical_digest(plan) or run_id != plan.get("run_id"):
            raise ContractError("PHASE_EVENT_PLAN_BINDING")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps or not isinstance(steps[0], Mapping):
            raise ContractError("PHASE_EVENT_PLAN_BINDING")
        held = steps[0].get("held_target_segments")
        if held is not None:
            from tools.data_factory.rollout.finite_plan import HELD_PROPOSAL_SCHEMA, REFERENCE_PROPOSAL_SCHEMA, validate_proposal, trajectory_schema
            proposal = validate_proposal(plan.get("learned_proposal"))
            if (trajectory_schema(proposal) not in {HELD_PROPOSAL_SCHEMA, REFERENCE_PROPOSAL_SCHEMA} or not isinstance(held, list)
                    or not 1 <= len(held) <= 2 * len(proposal["actions"])
                    or value["phase"] != "LEARNED_CHUNK" or value["event_source"] != "pickup_executor"):
                raise ContractError("PHASE_EVENT_PLAN_BINDING")
            if event in ACTION_EVENTS:
                if count != len(held):
                    raise ContractError("PHASE_EVENT_SEGMENT")
                step = held[index]
                expected = ({"step": step, "accepted": True} if event == "GOAL_ACCEPTED" else
                            {"step": step} if event == "DISPATCH_REQUESTED" else
                            {"step": step, "terminal_status": "SUCCEEDED"} if value["action_status"] == "SUCCEEDED" else None)
                if expected is not None and evidence_digest != canonical_digest(expected):
                    raise ContractError("PHASE_EVENT_SEGMENT_BINDING")
        elif event in ACTION_EVENTS and (index != 0 or count != 1):
            raise ContractError("PHASE_EVENT_PLAN_BINDING")
    status = value["action_status"]
    if event in ACTION_EVENTS:
        _text(status, "PHASE_EVENT_ACTION_STATUS")
    elif status is not None:
        raise ContractError("PHASE_EVENT_ACTION_STATUS")
    record = dict(value)
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError("PHASE_EVENT_SCHEMA")
    _integer(record["sequence"], "PHASE_EVENT_SEQUENCE", maximum=2**31 - 1)
    _text(record["phase"], "PHASE_EVENT_PHASE")
    _integer(record["event_ros_time_ns"], "PHASE_EVENT_ROS_TIME", minimum=1)
    _integer(record["monotonic_time_ns"], "PHASE_EVENT_MONOTONIC_TIME")
    _text(record["ros_clock_type"], "PHASE_EVENT_CLOCK")
    _text(record["event_source"], "PHASE_EVENT_SOURCE")
    return record


def validate_phase_plans(plans, *, plan=None):
    """Validate an explicit immutable lookup for one task's exact chunk plans."""
    if not isinstance(plans, Mapping) or not plans:
        raise ContractError("PHASE_EVENT_PLANS_BINDING")
    run_ids = set()
    for digest, value in plans.items():
        if (not isinstance(digest, str) or not DIGEST.fullmatch(digest)
                or not isinstance(value, Mapping) or canonical_digest(value) != digest
                or not isinstance(value.get("run_id"), str) or not SAFE_ID.fullmatch(value["run_id"])):
            raise ContractError("PHASE_EVENT_PLANS_BINDING")
        run_ids.add(value["run_id"])
    if len(run_ids) != 1 or plan is not None and plans.get(canonical_digest(plan)) != plan:
        raise ContractError("PHASE_EVENT_PLANS_BINDING")
    return plans


def validate_phase_event_sequence(events, *, plan=None, plans=None):
    """Validate segment identities once, before existing timing/row consumers."""
    if plans is not None:
        plans = validate_phase_plans(plans, plan=plan)
    records = []
    for event in events:
        selected = plan
        if plans is not None:
            digest = event.get("plan_digest") if isinstance(event, Mapping) else None
            selected = plans.get(digest) if isinstance(digest, str) else None
            if selected is None:
                raise ContractError("PHASE_EVENT_PLAN_BINDING")
        records.append(validate_phase_event(event, plan=selected))
    counts, seen, last_index = {}, set(), {}
    for event in records:
        if event["event"] not in ACTION_EVENTS:
            continue
        phase, count = (event["run_id"], event["plan_digest"], event["phase"]), event["segment_count"]
        if phase in counts and counts[phase] != count:
            raise ContractError("PHASE_EVENT_SEGMENT_COUNT")
        counts[phase] = count
        key = (phase, event["segment_index"], event["event"])
        if key in seen:
            raise ContractError("PHASE_EVENT_SEGMENT_DUPLICATE")
        previous = last_index.get(phase, -1)
        if not previous <= event["segment_index"] <= previous + 1:
            raise ContractError("PHASE_EVENT_SEGMENT_ORDER")
        last_index[phase] = event["segment_index"]
        seen.add(key)
    return records


def read_phase_events(path: str | Path, *, plan: Mapping[str, Any] | None = None, plans: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Read JSONL strictly; malformed sidecars are not silently repaired."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError("PHASE_EVENT_IO", str(exc)) from exc
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ContractError("PHASE_EVENT_JSONL", f"blank line {number}")
        try:
            value = load_json_strict(line)
            if plans is None:
                value = validate_phase_event(value, plan=plan)
            records.append(value)
        except ContractError as exc:
            raise ContractError(exc.code, f"line {number}: {exc}") from exc
    return validate_phase_event_sequence(records, plan=plan, plans=plans)


class PhaseEventWriter:
    """Best-effort sidecar writer: queue/disk failure only latches report failure."""

    def __init__(self, path: str | Path, *, capacity: int = QUEUE_CAPACITY, max_line_bytes: int = MAX_LINE_BYTES, plan: Mapping[str, Any] | None = None) -> None:
        if capacity < 1 or max_line_bytes < 1:
            raise ContractError("PHASE_EVENT_WRITER_CONFIG")
        self.path = Path(path)
        self._plan = copy.deepcopy(plan)
        self._queue: queue.Queue[tuple[bytes, bool]] = queue.Queue(maxsize=capacity)
        self._max_line_bytes = max_line_bytes
        self._error: str | None = None
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._write, name="phase-event-writer", daemon=True)
        self._thread.start()

    @property
    def error_code(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self.error_code is None

    def _latch(self, code: str) -> None:
        with self._lock:
            self._error = self._error or code

    def emit(self, event: Mapping[str, Any], *, flush: bool = False, plan: Mapping[str, Any] | None = None) -> bool:
        try:
            record = validate_phase_event(event, plan=self._plan if plan is None else plan)
            line = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        except (ContractError, TypeError, ValueError):
            self._latch("BEHAVIOR_REPORT_UNAVAILABLE")
            return False
        if len(line) > self._max_line_bytes or self.error_code or self._closed.is_set():
            self._latch("BEHAVIOR_REPORT_UNAVAILABLE")
            return False
        try:
            self._queue.put_nowait((line, flush or record["event"] in {"ACTION_TERMINAL", "HOLD_ENTERED"}))
            return True
        except queue.Full:
            self._latch("BEHAVIOR_REPORT_UNAVAILABLE")
            return False

    def close(self, *, timeout_s: float = 1.0) -> bool:
        self._closed.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            self._latch("BEHAVIOR_REPORT_UNAVAILABLE")
        return self.error_code is None

    def _write(self) -> None:
        try:
            parent = self.path.parent
            if not parent.is_dir() or any(path.is_symlink() for path in (parent, *parent.parents)):
                raise OSError("unsafe phase-event directory")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags, 0o600)
            with os.fdopen(descriptor, "wb") as file:
                self._ready.set()
                while not self._closed.is_set() or not self._queue.empty():
                    try:
                        item = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    line, flush = item
                    file.write(line)
                    if flush:
                        file.flush()
                        os.fsync(file.fileno())
                file.flush()
                os.fsync(file.fileno())
        except OSError:
            self._latch("BEHAVIOR_REPORT_UNAVAILABLE")
        finally:
            self._ready.set()
