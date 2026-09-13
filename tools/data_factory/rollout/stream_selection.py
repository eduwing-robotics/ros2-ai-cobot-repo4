"""Detached native-queue selection for one continuous learned revision.

Selection is policy output evidence, not dispatch, controller acceptance,
reference progress, or physical success.  Queue advancement remains conditional
on the original native snapshot and cumulative controller reference progress.
"""

import copy
import math

from tools.data_factory.rollout.action_projection import project_gripper_position
from tools.data_factory.rollout.execution_state import JOINTS, _limits
from tools.data_factory.rollout.finite_plan import _robot_description_digest
from tools.fr5_data_factory import ContractError, DIGEST, canonical_digest


SELECTION_SCHEMA = "data_factory.native_stream_selection.v1"
GRIPPER_PROJECTION_QUANTA = 2
UNITS = ["rad"] * 6 + ["m"]


def _rows(value, code):
    if not isinstance(value, list) or not value:
        raise ContractError(code)
    rows = []
    try:
        for row in value:
            if (not isinstance(row, list) or len(row) != 7
                    or any(isinstance(item, bool) or not isinstance(item, (int, float))
                           or not math.isfinite(item) for item in row)):
                raise ContractError(code)
            rows.append([float(item) for item in row])
    except OverflowError as exc:
        raise ContractError(code) from exc
    return rows


def _tensor_rows(value):
    from torch import Tensor

    if not isinstance(value, Tensor):
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
    detached = value.detach().cpu()
    if detached.ndim != 2 or detached.shape[1] != 7 or detached.shape[0] < 1:
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
    return _rows(detached.tolist(), "LEARNED_STREAM_SELECTION_SNAPSHOT")


def _observation(value):
    fields = {"observation_digest", "source_clock", "source_timestamps_s"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")
    stamps = value["source_timestamps_s"]
    try:
        valid = (
            isinstance(value["observation_digest"], str)
            and DIGEST.fullmatch(value["observation_digest"])
            and value["source_clock"] == "SYSTEM_TIME"
            and isinstance(stamps, dict)
            and set(stamps) == {"camera1", "camera2", "state"}
            and all(not isinstance(stamp, bool) and isinstance(stamp, (int, float))
                    and math.isfinite(stamp) for stamp in stamps.values())
        )
    except OverflowError as exc:
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE") from exc
    if not valid:
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")
    return {
        "observation_digest": value["observation_digest"],
        "source_clock": value["source_clock"],
        "source_timestamps_s": {
            name: float(stamps[name]) for name in ("camera1", "camera2", "state")
        },
    }


def _projection(rows, *, upper_m, projection_quanta):
    projected = []
    for row in rows:
        try:
            gripper = project_gripper_position(
                row[-1], upper_m=upper_m,
                projection_quanta=projection_quanta,
            )["projected_m"]
        except (OverflowError, TypeError, ValueError) as exc:
            raise ContractError("LEARNED_JOINT_LIMIT") from exc
        projected.append([*row[:6], gripper])
    return projected


def _generation_receipt(value, source, chunk, max_age_s):
    """Replay original generation qualification without a dispatch-time TTL."""
    code = "LEARNED_STREAM_GENERATION_BINDING"
    fields = {"schema_version", "chunk_id", "observation_digest", "source_clock",
              "source_timestamps_s", "sampling_started_at_s", "sampling_started_monotonic_s",
              "merge_completed_at_s", "merge_completed_monotonic_s", "max_observation_age_s",
              "eligible", "receipt_digest"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema_version"] != "data_factory.native_generation_eligibility.v1"
            or type(value["chunk_id"]) is not int or value["chunk_id"] != chunk
            or value["eligible"] is not True
            or any(value[key] != source[key] for key in source)
            or value["receipt_digest"] != canonical_digest({
                key: item for key, item in value.items() if key != "receipt_digest"})):
        raise ContractError(code)
    numbers = [value[key] for key in ("sampling_started_at_s", "sampling_started_monotonic_s",
               "merge_completed_at_s", "merge_completed_monotonic_s", "max_observation_age_s")]
    try:
        if any(type(number) not in (int, float) or not math.isfinite(number) for number in numbers):
            raise ContractError(code)
        started, steady_started, completed, steady_completed, bound = numbers
        if (bound <= 0 or completed < started or steady_completed < steady_started
                or max_age_s is not None and bound != max_age_s):
            raise ContractError(code)
        for stamp in source["source_timestamps_s"].values():
            if (stamp > started or completed - stamp > bound
                    or started - stamp + steady_completed - steady_started > bound):
                raise ContractError("LEARNED_STALE_OBSERVATION")
    except OverflowError as exc:
        raise ContractError(code) from exc


def validate_stream_selection(value, *, robot_description,
                              gripper_projection_quanta=GRIPPER_PROJECTION_QUANTA,
                              max_observation_age_s=None):
    """Validate detached JSON and return its exact controller action rows.

    An explicit age bound requires the original once-qualified generation
    receipt. It never evaluates the age of queued rows against the current time.
    """
    try:
        if max_observation_age_s is not None and (
                type(max_observation_age_s) not in (int, float)
                or not math.isfinite(max_observation_age_s) or max_observation_age_s <= 0):
            raise ContractError("LEARNED_STREAM_GENERATION_BINDING")
    except OverflowError as exc:
        raise ContractError("LEARNED_STREAM_GENERATION_BINDING") from exc
    fields = {
        "schema_version", "snapshot_generation", "snapshot_queue_index",
        "source_chunk", "source_row_indices", "source_observation",
        "joint_order", "units", "action_semantics", "raw_actions",
        "processed_actions", "controller_actions", "robot_description_digest",
        "gripper_projection", "selection_digest",
    }
    if (not isinstance(value, dict) or not fields <= set(value) <= fields | {"generation_eligibility"}
            or value.get("schema_version") != SELECTION_SCHEMA):
        raise ContractError("LEARNED_STREAM_SELECTION_SCHEMA")
    selected = copy.deepcopy(value)
    if selected["selection_digest"] != canonical_digest({
            key: item for key, item in selected.items() if key != "selection_digest"}):
        raise ContractError("LEARNED_STREAM_SELECTION_DIGEST")

    if (type(selected["snapshot_generation"]) is not int
            or selected["snapshot_generation"] < 1
            or type(selected["snapshot_queue_index"]) is not int
            or selected["snapshot_queue_index"] < 0
            or type(selected["source_chunk"]) is not int
            or selected["source_chunk"] < 1):
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")
    indices = selected["source_row_indices"]
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or indices != list(range(indices[0], indices[0] + len(indices)))):
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")
    selected["source_observation"] = _observation(selected["source_observation"])
    if max_observation_age_s is not None or "generation_eligibility" in selected:
        _generation_receipt(selected.get("generation_eligibility"),
                            selected["source_observation"], selected["source_chunk"],
                            max_observation_age_s)
    if (selected["joint_order"] != JOINTS or selected["units"] != UNITS
            or selected["action_semantics"] != "ABSOLUTE_JOINT_POSITION"):
        raise ContractError("LEARNED_ACTION_CONTRACT")

    raw = _rows(selected["raw_actions"], "LEARNED_ACTION_7D")
    processed = _rows(selected["processed_actions"], "LEARNED_ACTION_7D")
    controller = _rows(selected["controller_actions"], "LEARNED_ACTION_7D")
    if not len(indices) == len(raw) == len(processed) == len(controller):
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")

    limits = _limits(robot_description)
    digest = _robot_description_digest(robot_description)
    projection = selected["gripper_projection"]
    if (selected["robot_description_digest"] != digest
            or not isinstance(projection, dict)
            or set(projection) != {"upper_m", "projection_quanta"}
            or type(gripper_projection_quanta) is not int
            or gripper_projection_quanta != GRIPPER_PROJECTION_QUANTA
            or projection["projection_quanta"] != gripper_projection_quanta
            or limits[-1][0] != 0
            or projection["upper_m"] != limits[-1][1]):
        raise ContractError("LEARNED_STREAM_SELECTION_MODEL")
    expected = _projection(
        processed, upper_m=limits[-1][1],
        projection_quanta=gripper_projection_quanta,
    )
    if controller != expected or any(
            not low <= item <= high
            for row in controller for item, (low, high, _) in zip(row, limits)):
        raise ContractError("LEARNED_JOINT_LIMIT")

    selected["raw_actions"] = raw
    selected["processed_actions"] = processed
    selected["controller_actions"] = controller
    if selected["selection_digest"] != canonical_digest({
            key: item for key, item in selected.items() if key != "selection_digest"}):
        raise ContractError("LEARNED_STREAM_SELECTION_DIGEST")
    return selected


def select_stream_revision(snapshot, *, robot_description,
                           gripper_projection_quanta=GRIPPER_PROJECTION_QUANTA,
                           row_count=None):
    """Select a prefix of the first remaining native source chunk, without ACK."""
    from lerobot_strategy_fr5.acknowledged_queue import (
        ActionQueueSnapshot, ObservationProvenance, RawActionIndex,
    )

    if not isinstance(snapshot, ActionQueueSnapshot) or not snapshot.rows:
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
    first = snapshot.rows[0]
    if not isinstance(first, RawActionIndex):
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
    available = 0
    for row in snapshot.rows:
        if not isinstance(row, RawActionIndex):
            raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
        if row.chunk != first.chunk:
            break
        available += 1
    if row_count is None:
        row_count = available
    if type(row_count) is not int or not 1 <= row_count <= available:
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")

    rows = snapshot.rows[:row_count]
    if tuple(row.index for row in rows) != tuple(range(first.index, first.index + row_count)):
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")
    provenance = snapshot.observation_provenance[:row_count]
    if (len(provenance) != row_count or not isinstance(provenance[0], ObservationProvenance)
            or any(item != provenance[0] for item in provenance)):
        raise ContractError("LEARNED_STREAM_SELECTION_SOURCE")
    raw = _tensor_rows(snapshot.original_actions)
    processed = _tensor_rows(snapshot.processed_actions)
    if len(raw) != len(snapshot.rows) or len(processed) != len(snapshot.rows):
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")

    limits = _limits(robot_description)
    if limits[-1][0] != 0:
        raise ContractError("LEARNED_STREAM_SELECTION_MODEL")
    processed = processed[:row_count]
    observation = _observation({
        "observation_digest": provenance[0].observation_digest,
        "source_clock": provenance[0].source_clock,
        "source_timestamps_s": dict(provenance[0].source_timestamps_s),
    })
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "snapshot_generation": snapshot.generation,
        "snapshot_queue_index": snapshot.queue_index,
        "source_chunk": first.chunk,
        "source_row_indices": [row.index for row in rows],
        "source_observation": observation,
        "joint_order": list(JOINTS),
        "units": list(UNITS),
        "action_semantics": "ABSOLUTE_JOINT_POSITION",
        "raw_actions": raw[:row_count],
        "processed_actions": processed,
        "controller_actions": _projection(
            processed, upper_m=limits[-1][1],
            projection_quanta=gripper_projection_quanta,
        ),
        "robot_description_digest": _robot_description_digest(robot_description),
        "gripper_projection": {
            "upper_m": limits[-1][1],
            "projection_quanta": gripper_projection_quanta,
        },
    }
    receipts = snapshot.generation_eligibility
    if receipts:
        if len(receipts) != len(snapshot.rows):
            raise ContractError("LEARNED_STREAM_GENERATION_BINDING")
        selected_receipts = receipts[:row_count]
        if any(item is not None for item in selected_receipts):
            from lerobot_strategy_fr5.acknowledged_queue import GenerationEligibility
            receipt = selected_receipts[0]
            if (not isinstance(receipt, GenerationEligibility)
                    or any(item != receipt for item in selected_receipts)):
                raise ContractError("LEARNED_STREAM_GENERATION_BINDING")
            selection["generation_eligibility"] = receipt.to_dict()
    selection["selection_digest"] = canonical_digest(selection)
    return validate_stream_selection(
        selection, robot_description=robot_description,
        gripper_projection_quanta=gripper_projection_quanta,
    )


def acknowledge_stream_selection(queue, snapshot, selection, *, selected_count,
                                 acknowledged_count=0, robot_description,
                                 gripper_projection_quanta=GRIPPER_PROJECTION_QUANTA):
    """ACK new cumulative reference progress against the retained snapshot.

    ``selected_count`` is full-7D controller reference progress.  It is not
    action-goal acceptance, physical arrival, or task success.  The returned
    integer is the new cumulative acknowledged count.
    """
    from lerobot_strategy_fr5.acknowledged_queue import (
        AcknowledgedActionQueue, ActionQueueSnapshot,
    )

    if not isinstance(queue, AcknowledgedActionQueue) or not isinstance(snapshot, ActionQueueSnapshot):
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
    checked = validate_stream_selection(
        selection, robot_description=robot_description,
        gripper_projection_quanta=gripper_projection_quanta,
    )
    if (type(selected_count) is not int or type(acknowledged_count) is not int
            or not 0 <= acknowledged_count <= selected_count
            or selected_count > len(checked["source_row_indices"])):
        raise ContractError("LEARNED_STREAM_SELECTION_PROGRESS")
    expected = select_stream_revision(
        snapshot, robot_description=robot_description,
        gripper_projection_quanta=gripper_projection_quanta,
        row_count=len(checked["source_row_indices"]),
    )
    if checked != expected:
        raise ContractError("LEARNED_STREAM_SELECTION_SNAPSHOT")
    queue.advance_unchanged_prefix(
        snapshot, offset=acknowledged_count,
        count=selected_count - acknowledged_count,
    )
    return selected_count
