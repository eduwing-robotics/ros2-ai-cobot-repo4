"""Single FAKE/PHYSICAL collection-operator wiring root."""
from __future__ import annotations

import copy
import math
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.a4_place_yaw.region_layout import (
    a4_printable_polygon,
    make_red_blue_region_layout,
    workspace_region,
)
from tools.data_factory import run_job
from tools.data_factory.campaign_authoring import (
    DRAFT_SCHEMA_V2,
    campaign_cell_id,
)
from tools.data_factory.campaign_operator import CampaignOperator, SIDE_EFFECT_COUNTERS
from tools.data_factory.cell_state import CellStateStore
from tools.data_factory.collection_seed import (
    derive_domain_seed as _domain_seed,
    trajectory_sampling_binding,
    validate_campaign_seed,
)
from tools.data_factory.experiment_manifest import compile_fr5_hypothesis
from tools.data_factory.one_job import OneJob, TEST_ONLY_READINESS_CONTRACT
from tools.data_factory.motion.trajectory_variants import VARIANT_IDS
from tools.data_factory.motion.pose_snapshot import (
    JOINTS as SNAPSHOT_JOINTS,
    _validate_snapshot,
)
from tools.data_factory.motion.object_reposition import (
    build_object_reposition_binding,
    yaw_preserving_destination,
)
from tools.data_factory.state_space import (
    YAW_BINDING_SCHEMA,
    validate_state_space_design_profile,
    validate_yaw_sample_binding,
    validate_yaw_sampling_profile,
)
from tools.data_factory.operator.preview import (
    QA_WORKFLOW as FAKE_QA_WORKFLOW,
    TEST_OPERATOR,
    build_fake_operator_console,
    synthetic_fixture,
)
from tools.data_factory.operator.workflow.application import CollectionOperatorApplication
from tools.data_factory.operator.workflow.intents import CandidateReviewPort, OperatorIntentCore
from tools.data_factory.operator.catalog import (
    CATALOG_SCHEMA,
    SELECTION_SCHEMA_V2,
    SELECTION_SCHEMA,
    camera_binding_digest,
    load_operator_catalog,
    selected_motion_preset,
    project_assisted_poses,
    project_balanced_start_pose_ids,
    project_direct_poses,
    project_workspace_cycle_poses,
    project_yaw_sample_bindings,
    resolve_workspace_cycle_selections,
    validate_operator_pose,
    validate_operator_selection,
    validate_yaw_preserving_transitions,
)
from tools.data_factory.operator.registries.start_pose import (
    compile_start_pose_profile,
    list_start_pose_profiles,
    project_robot_start_pose_qualification,
    save_start_pose_profile,
)
from tools.data_factory.operator.registries.region import (
    validate_region_endpoint_authority,
)
from tools.data_factory.operator.setup.contracts import (
    build_camera_role_bindings,
    build_runtime_episode_binding,
    build_runtime_root_binding,
    build_runtime_start_binding,
    build_test_only_root_binding,
    build_test_only_runtime_episode_binding,
    build_test_only_start_binding,
    gripper_setup_projection,
    initialize_test_only_state_from_user_declaration,
    bind_test_only_physical_scene,
    normalize_camera_devices,
    qualified_table_plane_reference,
    reuse_camera_role_bindings,
    select_yaw0_print_profile,
    validate_camera_role_bindings,
    validate_print_measurements,
    validate_test_only_start_binding,
    write_camera_role_bindings,
)
from tools.data_factory.operator.setup.camera import (
    _camera_binding,
    _v2_camera_profiles,
    discover_camera_devices,
    discover_uvc_device_ids,
    resolve_camera_setup,
)
from tools.data_factory.operator.setup.physical import (
    build_physical_operator_environment,
    capture_gripper_setup_readback,
    capture_home_snapshot,
    normalize_gripper_after_operator_ready,
    passive_physical_gate,
)
from tools.data_factory.operator.web.bridge import LoopbackBridge
from tools.data_factory.operator.web import projection
from tools.data_factory.operator.workflow.campaign import (
    DEFAULT_GRIPPER_RETUNE,
    DEFAULT_HOME,
    DEFAULT_JOB,
    DEFAULT_MOTION,
    DEFAULT_PROFILE,
    DEFAULT_START_POSES,
    DEFAULT_TCP_MANIFEST,
    DEFAULT_URDF,
    DEFAULT_YAW0,
    OperatorConsole,
    _build_physical_campaign_contract,
    _campaign_camera_warmup,
    _derive_test_only_gripper_program,
    _home_start_pose,
    _validate_successful_object_reposition_result,
    _validate_test_only_gripper_retune,
)
from tools.data_factory.quality.coverage_report import build_coverage_report
from tools.data_factory.operator.registries.workspace import WorkspaceManager
from tools.data_factory.scene_state import SceneStateStore
from tools.data_factory.task_recipe import (
    compile_episode_instruction_binding,
    compile_task_binding,
    get_task_recipe,
    validate_region_binding,
)
from tools.data_factory.workspace_geometry import (
    point_in_convex_polygon,
    rotate_xy,
    safe_convex_polygon_for_yaws,
)
from tools.fr5_data_factory import (
    ContractError,
    DIGEST,
    SAFE_ID,
    canonical_digest,
    load_json_strict,
    load_motion_preset,
    normalize_yaw_deg,
    task_instruction,
)


ROOT = Path(__file__).resolve().parents[3]
MIN_CAMPAIGN_AUTHORIZATION_TTL = timedelta(hours=1)
CAMPAIGN_STARTUP_MARGIN = timedelta(minutes=15)
QUALIFIED_EPISODE_RUNTIME = timedelta(minutes=2)


def _campaign_authorization_ttl(requested_count: int) -> timedelta:
    if type(requested_count) is not int or not 1 <= requested_count <= 100:
        raise ContractError("PHYSICAL_CONSOLE_REQUESTED_COUNT")
    return max(
        MIN_CAMPAIGN_AUTHORIZATION_TTL,
        CAMPAIGN_STARTUP_MARGIN + QUALIFIED_EPISODE_RUNTIME * requested_count,
    )


def _repository_path(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    path = (
        (repository / path).resolve(strict=True)
        if not path.is_absolute()
        else path.resolve(strict=True)
    )
    try:
        path.relative_to(repository)
    except ValueError as exc:
        raise ContractError("PHYSICAL_CONSOLE_PATH") from exc
    return path


def _tcp_manifest_for_robot(repository: Path, robot_system_id: object) -> Path:
    if (
        not isinstance(robot_system_id, str)
        or SAFE_ID.fullmatch(robot_system_id) is None
    ):
        raise ContractError("TCP_MANIFEST_BINDING")
    robot = load_json_strict(
        repository / "config/data_factory/robot_systems" / f"{robot_system_id}.json"
    )
    candidates = [
        _repository_path(repository, DEFAULT_TCP_MANIFEST),
        *sorted((repository / "config/data_factory/tcp_candidates").glob("*.json")),
    ]
    matches = []
    for path in candidates:
        manifest = load_json_strict(path)
        candidate = manifest.get("tcp_candidate")
        digest = manifest.get("tcp_candidate_digest")
        if (
            manifest.get("robot_system_id") == robot_system_id
            and isinstance(candidate, Mapping)
            and canonical_digest(candidate) == digest == robot.get("tcp_digest")
        ):
            matches.append(path)
    if len(matches) != 1:
        raise ContractError("TCP_MANIFEST_BINDING")
    return matches[0]


def _runtime_gripper_settings(
    repository: Path, initial_job: Mapping[str, Any],
    gripper_retune: str | Path | None,
) -> dict[str, int]:
    """Resolve the object-scoped gripper settings before ROS starts."""
    grasp_matches = []
    for path in sorted(
        (repository / "config/data_factory/grasps").glob("*.json"),
        key=lambda value: str(value),
    ):
        grasp = load_json_strict(path)
        if grasp.get("grasp_profile_id") == initial_job.get("grasp_profile_id"):
            grasp_matches.append(grasp)
    if len(grasp_matches) != 1:
        raise ContractError("GRIPPER_PROFILE_BINDING")
    if gripper_retune is None:
        close = grasp_matches[0].get("gripper_close")
        opened = grasp_matches[0].get("gripper_open")
        if not isinstance(close, Mapping) or not isinstance(opened, Mapping):
            raise ContractError("GRIPPER_PROFILE_BINDING")
        settings = {
            "velocity_percent": close.get("velocity_percent"),
            "force_percent": close.get("force_percent"),
            "open_velocity_percent": opened.get("velocity_percent"),
            "open_force_percent": opened.get("force_percent"),
        }
        if any(type(value) is not int or not 1 <= value <= 100 for value in settings.values()):
            raise ContractError("GRIPPER_PROFILE_BINDING")
        return settings
    retune = load_json_strict(_repository_path(repository, gripper_retune))
    motion_matches = []
    for path in sorted(
        (repository / "config/data_factory/motion_qualifications").glob("*.json"),
        key=lambda value: str(value),
    ):
        motion = load_json_strict(path)
        if canonical_digest(motion) == retune.get("base_motion_qualification_digest"):
            motion_matches.append(motion)
    if (
        len(motion_matches) != 1
        or retune.get("object_profile_id") != initial_job.get("object_profile_id")
        or retune.get("grasp_profile_id") != initial_job.get("grasp_profile_id")
    ):
        raise ContractError("TEST_ONLY_GRIPPER_RETUNE_BINDING")
    _checked, settings = (
        _validate_test_only_gripper_retune(
            retune, grasp=grasp_matches[0],
            motion=motion_matches[0],
        )
    )
    return settings


def _resolve_physical_pose_domain(
    *, template_job: Mapping[str, Any], poses: Sequence[Mapping[str, Any]],
    operator_label: str, payload_template: Mapping[str, Any],
    sheet_manifest: Path,
    release_poses: Sequence[Mapping[str, Any]] | None = None,
    workspace_bindings: Mapping[str, Mapping[str, Any]] | None = None,
    resolver=None,
) -> list[dict[str, Any]]:
    """Resolve each exact pose with the ordinary JobSpec/motion input path."""
    if not isinstance(poses, (list, tuple)) or not poses:
        raise ContractError("PHYSICAL_CONSOLE_POSE_DOMAIN")
    if (
        release_poses is not None
        and (
            not isinstance(release_poses, (list, tuple))
            or len(release_poses) != len(poses)
        )
    ):
        raise ContractError("PHYSICAL_CONSOLE_POSE_DOMAIN")
    sheet_digest = canonical_digest(load_json_strict(sheet_manifest))
    if workspace_bindings is not None and (
        not isinstance(workspace_bindings, Mapping)
        or not workspace_bindings
        or any(
            not isinstance(place_id, str)
            or not SAFE_ID.fullmatch(place_id)
            or not isinstance(binding, Mapping)
            or set(binding) != {
                "frame_id", "selected_sheet", "yaw0_sheet",
                "motion_qualification", "region_binding",
            }
            or not isinstance(binding["region_binding"], Mapping)
            or not isinstance(binding["frame_id"], str)
            or not SAFE_ID.fullmatch(binding["frame_id"])
            for place_id, binding in workspace_bindings.items()
        )
    ):
        raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")

    def endpoint(place_id: str) -> dict[str, Any]:
        if workspace_bindings is None:
            if place_id != template_job.get("place_id"):
                raise ContractError("PHYSICAL_CONSOLE_EXACT_SCOPE")
            return {
                "frame_id": template_job["cell_calibration_id"],
                "selected_sheet": sheet_manifest,
                "yaw0_sheet": payload_template["yaw0_sheet"],
                "motion_qualification": payload_template[
                    "motion_qualification"
                ],
            }
        binding = workspace_bindings.get(place_id)
        if not isinstance(binding, Mapping):
            raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
        return copy.deepcopy(dict(binding))

    def endpoint_job(
        pose: Mapping[str, Any], binding: Mapping[str, Any], token: str,
    ) -> dict[str, Any]:
        job = copy.deepcopy(dict(template_job))
        job.update(
            job_id=f"physical-pose-{token}",
            operator_or_agent_id=operator_label,
            cell_calibration_id=binding["frame_id"],
            sheet_manifest_digest=canonical_digest(
                load_json_strict(binding["selected_sheet"]),
            ),
            **copy.deepcopy(dict(pose)),
        )
        return job

    result = []
    seen = set()
    resolver = run_job.resolve_inputs if resolver is None else resolver
    if not callable(resolver):
        raise ContractError("PHYSICAL_CONSOLE_RESOLVER")
    for index, pose in enumerate(poses):
        if not isinstance(pose, Mapping) or set(pose) != {
            "place_id", "yaw_deg", "x_mm", "y_mm",
        }:
            raise ContractError("PHYSICAL_CONSOLE_POSE_DOMAIN")
        token = canonical_digest(dict(pose)).removeprefix("sha256:")[:20]
        source_endpoint = endpoint(pose["place_id"])
        job = endpoint_job(pose, source_endpoint, token)
        if workspace_bindings is None:
            job["sheet_manifest_digest"] = sheet_digest
        candidate_payload = copy.deepcopy(dict(payload_template))
        candidate_payload.update(
            job=job,
            selected_sheet=str(source_endpoint["selected_sheet"]),
            yaw0_sheet=str(source_endpoint["yaw0_sheet"]),
            motion_qualification=str(source_endpoint["motion_qualification"]),
        )
        if release_poses is not None:
            release_pose = release_poses[index]
            if not isinstance(release_pose, Mapping) or set(release_pose) != {
                "place_id", "yaw_deg", "x_mm", "y_mm",
            }:
                raise ContractError("PHYSICAL_CONSOLE_POSE_DOMAIN")
            release_endpoint = endpoint(release_pose["place_id"])
            if release_pose["place_id"] == pose["place_id"]:
                candidate_payload.update(
                    recycle_yaw_deg=release_pose["yaw_deg"],
                    recycle_x_mm=release_pose["x_mm"],
                    recycle_y_mm=release_pose["y_mm"],
                )
            else:
                release_token = canonical_digest(
                    dict(release_pose),
                ).removeprefix("sha256:")[:20]
                candidate_payload["destination"] = {
                    "job": endpoint_job(
                        release_pose, release_endpoint, release_token,
                    ),
                    "selected_sheet": str(release_endpoint["selected_sheet"]),
                    "yaw0_sheet": str(release_endpoint["yaw0_sheet"]),
                    "motion_qualification": str(
                        release_endpoint["motion_qualification"]
                    ),
                }
        resolved, program, _binding = resolver(
            candidate_payload, scene_binding_call=lambda *_args: {},
        )
        key = resolved["resolved_job_digest"]
        if key in seen:
            continue
        if (
            program.get("schema_version") not in {
                "fr5.motion_program.v2", "fr5.motion_program.v4",
            }
            or len(program.get("steps", [])) != 10
        ):
            raise ContractError("PHYSICAL_CONSOLE_EXACT_SCOPE")
        seen.add(key)
        # The campaign resolver schema describes each source condition. The
        # runtime-only destination receipt is retained again at live resolution.
        source_resolved = copy.deepcopy(resolved)
        source_resolved.pop("destination_resolved_inputs", None)
        result.append(source_resolved)
    return result


def _validate_recorded_release_region(
    *, recorded_pose: Mapping[str, Any], target_yaw_deg: float,
    endpoint: Mapping[str, Any], resolved_destination: Mapping[str, Any],
) -> None:
    """Keep one physical release point safe before and after post-recording yaw."""
    try:
        binding = endpoint["region_binding"]
        if binding["physical_binding_status"] == "NOT_CONFIGURED":
            polygon = a4_printable_polygon()
        else:
            layout = make_red_blue_region_layout()
            region = workspace_region(layout, recorded_pose["place_id"])
            if binding != {
                "layout_id": layout["layout_id"],
                "layout_digest": layout["layout_digest"],
                "region_id": region["region_id"],
                "physical_binding_status": binding["physical_binding_status"],
            }:
                raise ValueError("region binding")
            polygon = region["polygon_local_xy_mm"]
        safe_polygon = safe_convex_polygon_for_yaws(
            polygon=polygon,
            object_size_xy_mm=resolved_destination["object_profile"][
                "dimensions_mm"
            ][:2],
            uncertainty_mm=resolved_destination["calibration"]["document"][
                "limits"
            ]["combined_error_bound_mm"],
            yaw_degs=(recorded_pose["yaw_deg"], target_yaw_deg),
        )
        sheet_xy = rotate_xy(
            (recorded_pose["x_mm"], recorded_pose["y_mm"]),
            recorded_pose["yaw_deg"],
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING") from exc
    if not point_in_convex_polygon(sheet_xy, safe_polygon):
        raise ContractError(
            "JOB_COORDINATE_BOUNDS",
            str((recorded_pose["x_mm"], recorded_pose["y_mm"])),
        )


def _product_fixture(
    pose_sequence: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expand the generic fixture into the product's bounded Place 1 domain."""
    baseline, template = synthetic_fixture()
    fixed = copy.deepcopy(baseline["fixed_contract"])
    documents = {
        "robot_system": {
            "schema_version": "data_factory.robot_system.v1",
            "robot_system_id": "fr5-r1", "qualification_status": "QUALIFIED",
            "base_frame": "base_link", "tcp_digest": canonical_digest("synthetic-tcp"),
        },
        "collection_profile": {
            "schema_version": "data_factory.collection_profile.v1",
            "collection_profile_id": "fr5-dual-rgb-30hz-v1",
            "qualification_status": "QUALIFIED",
        },
        "object_profile": {
            "schema_version": "data_factory.object_profile.v2",
            "object_profile_id": "object-r1", "qualification_status": "QUALIFIED",
            "description": "synthetic object", "dimensions_mm": [40, 30, 20],
            "datum": "center",
        },
        "grasp_profile": {
            "schema_version": "data_factory.grasp_profile.v2",
            "grasp_profile_id": "grasp-r1", "qualification_status": "QUALIFIED",
            "object_profile_id": "object-r1", "grasp_kind": "top_center",
        },
        "cell_calibration": {
            "schema_version": "data_factory.cell_calibration.v1",
            "calibration_id": "calibration-r1", "qualification_status": "QUALIFIED",
            "robot_system_id": "fr5-r1", "place_id": "PLACE_A",
        },
    }
    fixed["cell_calibration_digest"] = canonical_digest(documents["cell_calibration"])
    poses = (
        ({"place_id": "PLACE_A", "yaw_deg": yaw, "x_mm": x_mm, "y_mm": y_mm}
         for yaw, x_mm, y_mm in (
             (0, 0, 0), (90, 35, 0), (-180, 0, 20), (-90, -35, 0),
         ))
        if pose_sequence is None else pose_sequence
    )
    unique_poses: list[dict[str, Any]] = []
    for pose in poses:
        candidate = copy.deepcopy(dict(pose))
        if set(candidate) != {"place_id", "yaw_deg", "x_mm", "y_mm"}:
            raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
        candidate["yaw_deg"] = normalize_yaw_deg(candidate["yaw_deg"])
        if candidate["yaw_deg"].is_integer():
            candidate["yaw_deg"] = int(candidate["yaw_deg"])
        if candidate not in unique_poses:
            unique_poses.append(candidate)
    if not unique_poses:
        raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
    if pose_sequence is not None:
        for index in range(101):
            holdout = {
                "place_id": "PLACE_A", "yaw_deg": index + 0.5,
                "x_mm": 0, "y_mm": 0,
            }
            if holdout not in unique_poses:
                unique_poses.append(holdout)
                break
        else:
            raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
    conditions = [{
        "task_schema_version": "data_factory.job.v1", "task": fixed["task"],
        "robot_system_id": fixed["robot_system_id"], "place_id": "PLACE_A",
        "cell_calibration_id": fixed["cell_calibration_id"],
        "cell_calibration_digest": fixed["cell_calibration_digest"],
        "yaw_deg": pose["yaw_deg"], "x_mm": pose["x_mm"], "y_mm": pose["y_mm"],
        "object_profile_id": fixed["object_profile_id"],
        "grasp_profile_id": fixed["grasp_profile_id"],
        "motion_recipe_digest": fixed["motion_recipe_digest"],
        "collection_profile_digest": fixed["collection_profile_digest"],
    } for pose in unique_poses]
    report = build_coverage_report(
        collection_profile_id=documents["collection_profile"]["collection_profile_id"],
        domain=conditions, episodes=[],
    )
    resolvers = []
    base_qualifications = []
    source_job = baseline["resolver_receipts"][0]["normalized_job"]
    for index, condition in enumerate(conditions, 1):
        job = copy.deepcopy(source_job)
        selected_sheet = canonical_digest(["product-synthetic-sheet", index])
        job.update({
            "job_id": f"product-job-{index}", "place_id": "PLACE_A",
            "cell_calibration_id": fixed["cell_calibration_id"],
            "sheet_manifest_digest": selected_sheet,
            "yaw_deg": condition["yaw_deg"], "x_mm": condition["x_mm"],
            "y_mm": condition["y_mm"],
        })
        inputs = {
            "selected_sheet": selected_sheet,
            "yaw0_sheet": canonical_digest("product-synthetic-yaw0"),
            **{name: canonical_digest(document) for name, document in documents.items()},
        }
        resolver = {
            "normalized_job": job, "input_digests": inputs,
            "resolved_job_digest": canonical_digest({"job": job, "input_digests": inputs}),
            "robot": copy.deepcopy(documents["robot_system"]),
            "collection_profile": copy.deepcopy(documents["collection_profile"]),
            "calibration": {
                "center": [0.4, 0.0, 0.1], "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0],
                "document": copy.deepcopy(documents["cell_calibration"]),
            },
            "object_profile": copy.deepcopy(documents["object_profile"]),
            "grasp_profile": copy.deepcopy(documents["grasp_profile"]),
        }
        resolvers.append(resolver)
        qualification = {
            "schema_version": "data_factory.fr5_base_condition_qualification.v1",
            "source": "SYNTHETIC_TEST_ONLY", "qualification_status": "QUALIFIED",
            "coverage_report_digest": canonical_digest(report),
            "coverage_domain_digest": report["domain_digest"],
            "coverage_condition_digest": canonical_digest(condition),
            "resolver_result_digest": canonical_digest(resolver),
            "resolved_job_digest": resolver["resolved_job_digest"],
            "yaw_action_binding_digest": canonical_digest([
                "product-yaw", condition["yaw_deg"],
            ]),
            "dual_view_observability_digest": canonical_digest([
                "product-view", condition["yaw_deg"],
            ]),
        }
        qualification["qualification_digest"] = canonical_digest(qualification)
        base_qualifications.append(qualification)

    by_condition = {
        item["coverage_condition_digest"]: item for item in base_qualifications
    }
    base_qualifications = [
        by_condition[canonical_digest(cell["condition"])] for cell in report["cells"]
    ]
    pose_qualifications = copy.deepcopy(
        baseline["qualification_catalog"]["robot_start_pose_qualifications"]
    )
    fourth = copy.deepcopy(pose_qualifications[-1])
    fourth.update(
        robot_start_pose_id="start-4",
        target_rad={
            joint: 0.3 + index / 10
            for index, joint in enumerate(fourth["joint_order"])
        },
        home_candidate_digest=canonical_digest(["synthetic-home", "start-4"]),
    )
    fourth["qualification_digest"] = canonical_digest({
        key: value for key, value in fourth.items() if key != "qualification_digest"
    })
    pose_qualifications.append(fourth)
    pose_qualifications.sort(key=lambda item: item["robot_start_pose_id"])
    by_pose = {item["robot_start_pose_id"]: item for item in pose_qualifications}
    groups = (
        (["TRAIN"], ["TRAIN", "ID"], ["TRAIN"], ["OOD"])
        if pose_sequence is None else tuple(
            ["TRAIN", "ID"] if index == 0
            else ["OOD"] if index == len(conditions) - 1
            else ["TRAIN"]
            for index in range(len(conditions))
        )
    )
    allowed_pairs = [{
        "base_condition_qualification_digest": by_condition[
            canonical_digest(condition)
        ]["qualification_digest"],
        "robot_start_pose_qualification_digest": pose_qualifications[
            (index - 1) % len(pose_qualifications)
        ]["qualification_digest"],
        "split_groups": list(groups[index - 1]),
    } for index, condition in enumerate(conditions, 1)]
    allowed_pairs.sort(key=lambda item: (
        item["base_condition_qualification_digest"],
        item["robot_start_pose_qualification_digest"],
    ))
    qualification_catalog = {
        "schema_version": "data_factory.fr5_qualification_catalog.v1",
        "source": "SYNTHETIC_TEST_ONLY", "qualification_status": "QUALIFIED",
        "fixed_contract_digest": canonical_digest(fixed),
        "coverage_report_digest": canonical_digest(report),
        "coverage_domain_digest": report["domain_digest"],
        "resolver_result_digests": sorted(canonical_digest(item) for item in resolvers),
        "base_condition_qualifications": base_qualifications,
        "robot_start_pose_qualifications": pose_qualifications,
        "allowed_pairs": allowed_pairs,
    }
    qualification_catalog["catalog_digest"] = canonical_digest(qualification_catalog)
    hypothesis = compile_fr5_hypothesis(
        fixed_contract=fixed, coverage_report=report,
        resolver_results=resolvers, qualification_catalog=qualification_catalog,
    )
    template = copy.deepcopy(template)
    template["source"] = {
        "hypothesis_digest": hypothesis["hypothesis_digest"],
        "catalog_digest": hypothesis["qualification_catalog"]["catalog_digest"],
        "coverage_digest": canonical_digest(hypothesis["coverage_report"]),
    }
    return hypothesis, template


def _catalog(
    repository_root: str | Path, device_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = load_operator_catalog(repository_root, device_ids=[device_id])
    candidates = [
        item for item in catalog["combinations"]
        if item["workspace_id"] == "PLACE_A"
        and item["frame_id"] == "place-a-yaw0-r002"
        and item["task_id"] == "pickup_e2e"
        and item["cell_id"] == "PLACE_A-yaw0-CENTER"
        and item["variant_id"] == "DIRECT"
        and item["camera_profile_id"] == "fr5-up-rgb-30hz-v1"
        and item["execution"]["TEST_COLLECTION"]["executable"] is True
    ]
    if len(candidates) != 1:
        raise ContractError("PRODUCT_FAKE_CATALOG")
    combination = candidates[0]
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "combination_digest": combination["combination_digest"],
        "data_mode": "TEST_COLLECTION",
        **{
            field: combination[field]
            for field in (
                "workspace_id", "frame_id", "task_id", "object_id", "grasp_id",
                "cell_id", "start_pose_id", "motion_id", "variant_id",
                "camera_profile_id", "camera_device_id",
            )
        },
        "policy_id": "DETERMINISTIC_SPREAD",
    }
    return catalog, selection


def _source_draft(
    template: Mapping[str, Any], draft: Mapping[str, Any], campaign_id: str, *,
    hypothesis: Mapping[str, Any], pose_sequence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    count = draft["requested_count"]
    result = copy.deepcopy(dict(template))
    result.update(
        draft_id=draft["draft_id"], revision=draft["revision"],
        selector="BALANCED_INITIAL" if draft["authoring_mode"] == "ASSISTED" else "DIRECT_LIST",
        requested_count=count, normalized_seed=draft["normalized_seed"],
        pinned=copy.deepcopy(draft["pinned"]), excluded=copy.deepcopy(draft["excluded"]),
        direct_slots=[],
        manifest_id=f"{campaign_id}-manifest",
    )
    if draft.get("state_space_design_profile") is not None:
        result.update(
            schema_version=DRAFT_SCHEMA_V2,
            state_space_design_profile=copy.deepcopy(
                draft["state_space_design_profile"],
            ),
        )
    result["manifest_budget"].update({
        "max_physical_episodes": count, "max_rollout_trials": count,
        "max_hil_prompts": count, "max_reviews": count,
        "max_pending_reviews": count, "max_storage_bytes": 10_000 * count,
    })
    result["program_budget"].update({
        "max_rounds": max(1, count),
        "max_total_physical_episodes": count,
        "max_total_rollout_trials": count,
        "max_total_hil_prompts": count,
        "max_total_reviews": count,
        "max_pending_reviews": count,
        "max_total_storage_bytes": 10_000 * count,
    })
    bases = starts = allowed = None
    if pose_sequence is not None:
        if len(pose_sequence) != count:
            raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
        bases = {
            tuple(base["coverage_condition"][field] for field in (
                "place_id", "yaw_deg", "x_mm", "y_mm",
            )): base
            for base in hypothesis["base_conditions"]
        }
        starts = {
            item["qualification_digest"]: item
            for item in hypothesis["robot_start_poses"]
        }
        allowed = {
            item["base_condition_qualification_digest"]: item
            for item in hypothesis["qualification_catalog"]["allowed_pairs"]
            if "TRAIN" in item["split_groups"]
        }
    if result["selector"] == "BALANCED_INITIAL" and pose_sequence is not None:
        source_key = tuple(pose_sequence[0][field] for field in (
            "place_id", "yaw_deg", "x_mm", "y_mm",
        ))
        base = bases.get(source_key)
        pair = None if base is None else allowed.get(base["qualification_digest"])
        start = None if pair is None else starts.get(
            pair["robot_start_pose_qualification_digest"]
        )
        if base is None or pair is None or start is None:
            raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
        result["pinned"] = [campaign_cell_id(
            base["base_condition_digest"], start["robot_start_pose_id"], "TRAIN", 0,
        )]
    elif result["selector"] == "DIRECT_LIST":
        if pose_sequence is None:
            raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
        repeats: dict[tuple[str, str], int] = {}
        slots = []
        for pose in pose_sequence:
            key = tuple(pose[field] for field in (
                "place_id", "yaw_deg", "x_mm", "y_mm",
            ))
            base = bases.get(key)
            pair = None if base is None else allowed.get(base["qualification_digest"])
            start = None if pair is None else starts.get(
                pair["robot_start_pose_qualification_digest"]
            )
            if base is None or pair is None or start is None:
                raise ContractError("PRODUCT_FAKE_POSE_SEQUENCE")
            repeat_key = (base["base_condition_digest"], start["robot_start_pose_id"])
            repeat_index = repeats.get(repeat_key, 0)
            repeats[repeat_key] = repeat_index + 1
            slots.append({
                "slot_id": campaign_cell_id(
                    base["base_condition_digest"], start["robot_start_pose_id"],
                    "TRAIN", repeat_index,
                ),
                "base_condition_digest": base["base_condition_digest"],
                "robot_start_pose_id": start["robot_start_pose_id"],
                "split_group": "TRAIN", "repeat_index": repeat_index,
                "hil_prompts": 1, "reviews": 1, "pending_reviews": 0,
                "storage_bytes": max(1, result["manifest_budget"]["max_storage_bytes"] // count),
            })
        result["direct_slots"] = slots
    return result


def _bind_fake_episode_context(driver, intent, value) -> dict[str, Any]:
    context = copy.deepcopy(dict(value))
    roots = build_test_only_root_binding(
        driver.fixture_root,
        session_id=context["session_id"], run_id=intent["run_id"],
    )
    pose = intent["robot_start_pose"]
    joint_order = pose["joint_order"]
    target = [pose["target_rad"][joint] for joint in joint_order]
    start = {
        "scope": "MOTION_Q_SAFE_START", "data_disposition": "TEST_ONLY",
        "manifest_digest": intent["manifest_digest"],
        "slot_digest": canonical_digest(intent["slot"]),
        "robot_start_pose_id": pose["robot_start_pose_id"],
        "robot_start_pose_qualification_digest": pose["qualification_digest"],
        "motion_qualification_id": "synthetic-motion-r001",
        "motion_qualification_digest": canonical_digest([
            "SYNTHETIC_FAKE_MOTION", intent["run_id"],
        ]),
        "home_candidate_digest": pose["home_candidate_digest"],
        "joint_order": joint_order, "target_rad": target,
        "current_rad": target, "tolerance_rad": 0.01,
        "max_snapshot_age_s": 0.1,
        "snapshot_digest": canonical_digest([
            "SYNTHETIC_FAKE_START", intent["run_id"],
        ]),
        "status": "BOUND_TEST_ONLY",
        "authority": {
            "execution": "NONE", "human_approval": "NONE",
            "semantic_pass": "NONE", "training_approval": "NONE",
            "persistent_start_qualification": "NONE",
        },
    }
    start["binding_digest"] = canonical_digest(start)
    context.update(
        root_binding=roots,
        start_binding=validate_test_only_start_binding(
            start, manifest=driver.campaign_operator.manifest,
            hypothesis=driver.hypothesis, slot=intent["slot"],
        ),
    )
    context["context_digest"] = canonical_digest({
        key: item for key, item in context.items() if key != "context_digest"
    })
    return context


class ProductFakeOperator:
    """Own separate temporary recorder and workspace roots for one FAKE process."""

    def __init__(
        self, *, session_id: str = "product-fake-operator-r001",
        operator_label: str = TEST_OPERATOR, technical_status: str = "PASS",
        fault: str | None = None, clock=None,
    ):
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._temporary = tempfile.TemporaryDirectory(prefix="product-fake-operator-")
        self.fixture_root = str(Path(self._temporary.name).resolve(strict=True))
        self._workspace_temporary = tempfile.TemporaryDirectory(
            prefix="product-fake-workspace-",
        )
        workspace_root = Path(self._workspace_temporary.name).resolve(strict=True)
        self.workspace_root = str(workspace_root)
        self.workspace_candidate_root = str(workspace_root / "workspace_candidates")
        self.workspace_config_root = str(workspace_root / "config/data_factory")
        self._closed = False
        self._ready = False
        self._campaigns: list[OperatorConsole] = []
        try:
            repository = Path(__file__).resolve().parents[3]
            shutil.copytree(
                repository / "config/data_factory",
                Path(self.workspace_config_root),
            )
            device_id = "usb-Generic_USB2.0_PC_CAMERA-video-index0"
            hypothesis, template = _product_fixture()
            catalog, selection = _catalog(workspace_root, device_id)
            source_cell = load_json_strict(
                Path(self.workspace_config_root) / "cells/place-a-yaw0-r002.json",
            )
            tcp_path = (
                Path(self.workspace_config_root)
                / "test_only_physical/goal2-place1/tcp_candidate_manifest.json"
            )
            tcp = load_json_strict(tcp_path)
            rigid = {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_columns": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
            snapshots = tuple({
                "schema_version": "data_factory.pose_snapshot.v1",
                "frames": {"base": "base_link", "wrist": "wrist3_link"},
                "joint_positions_rad": {
                    name: 0.0 for name in ("j1", "j2", "j3", "j4", "j5", "j6")
                },
                "base_wrist": copy.deepcopy(rigid),
                "base_tcp": {
                    **copy.deepcopy(rigid),
                    "translation_m": point,
                    "candidate_status": "CANDIDATE",
                    "candidate_source_sha256": tcp["tcp_candidate_digest"],
                    "manifest_source_sha256": canonical_digest(tcp),
                },
                "joint_state_age_s": 0.05,
                "joint_stamp_ns": 1_000_000_000,
                "transform_stamp_ns": 1_000_000_000,
                "ros_sample_age_s": 0.05,
            } for point in (
                [1.0, 2.0, 3.0],
                [1.1285, 2.0, 3.0],
                [0.8715, 2.08, 3.0],
            ))
            snapshot_index = 0

            def workspace_manager_factory(display_name):
                return WorkspaceManager(
                    session_id=f"{session_id}-workspace",
                    candidate_root=self.workspace_candidate_root,
                    config_root=self.workspace_config_root,
                    display_name=display_name,
                )

            def workspace_snapshot():
                nonlocal snapshot_index
                result = copy.deepcopy(snapshots[snapshot_index % len(snapshots)])
                snapshot_index += 1
                return result

            def workspace_preview(manager, measurements):
                measured = validate_print_measurements(
                    source_scale_bar_mm=measurements["source_scale_bar_mm"],
                    final_scale_bar_mm=measurements["final_scale_bar_mm"],
                )
                return manager.preview_captured(
                    plane_reference=qualified_table_plane_reference(source_cell),
                    print_measurements=measured,
                    operator_or_agent_id=operator_label,
                    yaw0_sheet=select_yaw0_print_profile(
                        repository,
                        place_id=source_cell["place_id"],
                        source_scale_bar_mm=measured[
                            "source_scale_bar_measured_mm"
                        ],
                    ),
                    tcp_candidate_manifest=tcp_path,
                    tolerance_mm=1.0,
                )

            def reload_catalog():
                return load_operator_catalog(workspace_root, device_ids=[device_id])

            def environment():
                return {
                    "schema_version": "data_factory.operator_environment.v1",
                    "state": "READY" if self._ready else "SETUP_REQUIRED",
                    "observed_at": self.clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "components": {
                        name: {
                            "state": "READY" if self._ready else "MISSING",
                            "owner": TEST_OPERATOR if self._ready else None,
                            "reason": "SYNTHETIC_ATTACHED" if self._ready else "SYNTHETIC_NOT_PREPARED",
                        }
                        for name in ("robot", "controller", "gripper", "camera")
                    },
                }

            def prepare_environment():
                self._ready = True
                return environment()

            def campaign_factory(campaign_id, selected, draft):
                expected_policy = (
                    "DETERMINISTIC_SPREAD"
                    if draft["authoring_mode"] == "ASSISTED" else "DIRECT_SELECTION"
                )
                baseline_fields = (
                    "data_mode", "workspace_id", "frame_id", "task_id",
                    "object_id", "grasp_id", "start_pose_id", "motion_id",
                    "variant_id", "camera_profile_id", "camera_device_id",
                )
                if (
                    selected["data_mode"] != "TEST_COLLECTION"
                    or selected.get("policy_id") != expected_policy
                    or any(
                        selected.get(field) != selection[field]
                        for field in baseline_fields
                    )
                ):
                    raise ContractError("PRODUCT_FAKE_SELECTION")
                campaign_hypothesis = hypothesis
                campaign_template = template
                pose_sequence = None
                initial_pose = validate_operator_pose(
                    self.application.catalog, selected,
                    draft.get("current_object_pose"),
                )
                if draft["authoring_mode"] == "ASSISTED":
                    design_kwargs = (
                        {"state_space_design_profile": draft["state_space_design_profile"]}
                        if draft.get("state_space_design_profile") is not None else {}
                    )
                    pose_sequence = project_assisted_poses(
                        self.application.catalog, selected, initial_pose,
                        draft["requested_count"], repeat=draft["repeat"],
                        normalized_seed=_domain_seed(
                            draft["normalized_seed"], "spatial",
                        ),
                        yaw_sampling_seed=_domain_seed(
                            draft["normalized_seed"], "yaw",
                        ),
                        **design_kwargs,
                    )
                    campaign_hypothesis, campaign_template = _product_fixture(
                        pose_sequence,
                    )
                elif draft["authoring_mode"] == "DIRECT_EDIT":
                    requested = [
                        validate_operator_pose(
                            self.application.catalog, selected, pose,
                        )
                        for pose in draft.get("direct_poses", [])
                    ]
                    pose_sequence = project_direct_poses(
                        self.application.catalog, selected, initial_pose,
                        requested, draft["requested_count"],
                    )
                    campaign_hypothesis, campaign_template = _product_fixture(
                        pose_sequence,
                    )
                for base in campaign_hypothesis["base_conditions"]:
                    condition = base["coverage_condition"]
                    validate_operator_pose(
                        self.application.catalog, selected,
                        {key: condition[key] for key in (
                            "place_id", "yaw_deg", "x_mm", "y_mm",
                        )},
                    )
                source_draft = _source_draft(
                    campaign_template, draft, campaign_id,
                    hypothesis=campaign_hypothesis,
                    pose_sequence=pose_sequence,
                )
                first_run = len(self._campaigns) * 100
                holder = {}
                synthetic_candidates: dict[str, dict[str, Any]] = {}

                def review_candidate(
                    path, *, expected_file_digest,
                    expected_review_context_digest, checklist_id,
                    semantic_status, reviewed_by, reason,
                ):
                    key = str(Path(path))
                    candidate = synthetic_candidates.get(key)
                    if (
                        candidate is None
                        or candidate["file_digest"] != expected_file_digest
                        or candidate["review_context_digest"]
                        != expected_review_context_digest
                        or candidate["checklist_id"] != checklist_id
                        or candidate["semantic_status"] != "PENDING"
                    ):
                        raise ContractError("PRODUCT_FAKE_CANDIDATE_REVIEW")
                    candidate.update(
                        semantic_status=semantic_status,
                        reviewed_by=reviewed_by, reason=reason,
                    )
                    return {
                        "run_id": candidate["run_id"],
                        "semantic_status": semantic_status,
                        "reviewed_by": reviewed_by,
                    }

                def bind_candidate_state(reference, path):
                    candidate = synthetic_candidates.get(str(Path(path)))
                    if (
                        not isinstance(reference, Mapping)
                        or reference.get("schema_version")
                        != "data_factory.synthetic_episode_review_reference.v1"
                        or candidate is None
                        or reference.get("run_id") != candidate["run_id"]
                        or candidate["semantic_status"] == "PENDING"
                    ):
                        raise ContractError("PRODUCT_FAKE_CANDIDATE_STATE")
                    return {
                        **copy.deepcopy(dict(reference)),
                        "review_status": candidate["semantic_status"],
                        "retention_state": "PRESERVE",
                        "training_status": "NOT_AUTHORIZED",
                    }

                def episode(
                    intent, lifecycle, cancel_event, episode_context,
                    decision_provider, checkpoint_provider,
                ):
                    driver = holder["driver"]

                    def bind_reposition_preapproval(request):
                        bound = copy.deepcopy(dict(request))
                        bound["decision_binding"][
                            "object_reposition_preapproval"
                        ] = None
                        return decision_provider(bound)

                    outcome = driver.run_episode(
                        intent, lifecycle, cancel_event,
                        _bind_fake_episode_context(driver, intent, episode_context),
                        bind_reposition_preapproval, checkpoint_provider,
                    )
                    result = outcome.get("result")
                    technical = outcome.get("technical_evidence")
                    if (
                        not isinstance(result, Mapping)
                        or not isinstance(technical, Mapping)
                        or technical.get("status") != "PASS"
                    ):
                        return outcome
                    slots = driver.campaign_operator.manifest["slots"]
                    release_slot_value = (
                        slots[intent["order_index"] + 1]
                        if intent["order_index"] + 1 < len(slots)
                        else intent["slot"]
                    )
                    release_base = next(
                        item for item in campaign_hypothesis["base_conditions"]
                        if item["base_condition_digest"]
                        == release_slot_value["base_condition_digest"]
                    )
                    terminal_object_pose = {
                        key: release_base["coverage_condition"][key]
                        for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
                    }
                    candidate_path = str(
                        Path(self.fixture_root)
                        / "synthetic_candidate_reviews" / intent["run_id"]
                        / "candidate_admission.json"
                    )
                    review_context_digest = canonical_digest({
                        "source": "SYNTHETIC_TEST_ONLY",
                        "run_id": intent["run_id"],
                        "technical_evidence_digest": canonical_digest(technical),
                    })
                    candidate = {
                        "run_id": intent["run_id"],
                        "review_context_digest": review_context_digest,
                        "checklist_id": get_task_recipe(
                            selected["task_id"],
                        )["review_checklist_id"],
                        "semantic_status": "PENDING",
                        "reviewed_by": None,
                        "reason": None,
                    }
                    candidate["file_digest"] = canonical_digest(candidate)
                    synthetic_candidates[candidate_path] = candidate
                    ledger_reference = {
                        "schema_version": (
                            "data_factory.synthetic_episode_review_reference.v1"
                        ),
                        "run_id": intent["run_id"],
                        "path": str(Path(candidate_path).with_name(
                            "synthetic_episode_ledger.json"
                        )),
                        "state_path": str(Path(candidate_path).with_name(
                            "synthetic_episode_review_state.json"
                        )),
                        "review_status": "PENDING",
                        "retention_state": "PRESERVE",
                        "reclaim_state": "NOT_EVALUATED",
                        "training_status": "NOT_AUTHORIZED",
                    }
                    sealed = copy.deepcopy(dict(result))
                    sealed.update({
                        "terminal_object_pose": terminal_object_pose,
                        "episode_ledger": copy.deepcopy(ledger_reference),
                        "candidate_review_offer": {
                            "candidate_path": candidate_path,
                            "run_id": intent["run_id"],
                            "expected_file_digest": candidate["file_digest"],
                            "expected_review_context_digest": review_context_digest,
                            "checklist_id": candidate["checklist_id"],
                            "ledger_reference": copy.deepcopy(ledger_reference),
                        },
                    })
                    return {**copy.deepcopy(dict(outcome)), "result": sealed}

                def operator_factory(episode_call):
                    driver = build_fake_operator_console(
                        hypothesis=campaign_hypothesis, draft=source_draft,
                        fixture_root=self.fixture_root, session_id=campaign_id,
                        technical_status=technical_status, fault=fault,
                        clock=self.clock, adapter_only=True,
                        campaign_episode_call=episode_call,
                        operator_label=operator_label,
                        run_index=first_run,
                    )
                    holder["driver"] = driver
                    return driver.campaign_operator

                def projection():
                    driver = holder["driver"]
                    conditions = [
                        base["coverage_condition"]
                        for base in campaign_hypothesis["base_conditions"]
                    ]
                    return {
                        "setup": {
                            "host_status": "READY",
                            "operator_label": operator_label,
                            "subsystems": [{
                                "label": "fake", "status": "READY",
                                "detail": "process-local OneJob adapters",
                            }],
                        },
                        "fixed_lane": {
                            "workspace": selected["workspace_id"],
                            "task": selected["task_id"],
                        },
                        "draft": {
                            "draft_id": source_draft["draft_id"],
                            "cells": copy.deepcopy(conditions),
                        },
                        "capabilities": [{
                            "label": "synthetic TEST_ONLY",
                            "status": "FAKE_EXECUTABLE",
                        }],
                        "workspace_wizard": {"capability": "OFFLINE_ONLY"},
                        "effect_counts": copy.deepcopy(driver.counters),
                    }

                def terminal_response():
                    return holder["driver"].terminal_response()

                campaign = OperatorConsole(
                    session_id=campaign_id,
                    run_id=f"synthetic-run-{first_run}",
                    operator_label=operator_label,
                    campaign_operator_factory=operator_factory,
                    episode_call=episode,
                    projection_call=projection,
                    test_only_paths=self.fixture_root,
                    terminal_response_call=terminal_response,
                    candidate_review_port=CandidateReviewPort(
                        operator_label=operator_label,
                        review_call=review_candidate,
                    ),
                    candidate_state_bind_call=bind_candidate_state,
                    campaign_approval_once=True,
                    run_id_factory=lambda index, start=first_run: (
                        f"synthetic-run-{start + index}"
                    ),
                    prepare_timeout_s=2.0, close_timeout_s=3.0,
                    clock=self.clock,
                )
                self._campaigns.append(campaign)
                return campaign

            self.application = CollectionOperatorApplication(
                session_id=session_id, operator_label=operator_label,
                catalog=catalog, initial_selection=selection,
                projector=projection,
                environment_call=environment,
                prepare_environment_call=prepare_environment,
                campaign_factory=campaign_factory,
                workspace_manager_factory=workspace_manager_factory,
                workspace_snapshot_call=workspace_snapshot,
                workspace_preview_call=workspace_preview,
                catalog_reload_call=reload_catalog,
            )
        except Exception:
            self._workspace_temporary.cleanup()
            self._temporary.cleanup()
            raise

    @property
    def bridge_core(self) -> OperatorIntentCore:
        return self.application.bridge_core

    @property
    def campaigns(self) -> tuple[OperatorConsole, ...]:
        return tuple(self._campaigns)

    @property
    def current_campaign(self) -> OperatorConsole | None:
        return self._campaigns[-1] if self._campaigns else None

    def wait_for_campaign(self, timeout_s: float | None = None) -> dict[str, Any] | None:
        campaign = self.current_campaign
        return None if campaign is None else campaign.wait_for_episode(timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.application.close()
        finally:
            try:
                self._workspace_temporary.cleanup()
            finally:
                self._temporary.cleanup()


def build_product_fake_operator(**kwargs) -> ProductFakeOperator:
    return ProductFakeOperator(**kwargs)


__all__ = [
    "OperatorRuntime", "ProductFakeOperator", "build_operator_runtime",
    "build_product_fake_operator",
]


QA_WORKFLOW = (
    "환경 준비 결과에서 robot·gripper·camera의 측정 상태를 확인한다",
    "수집 계획에서 사용 가능한 상태공간과 횟수를 선택한다",
    "사람이 읽는 캠페인 요약을 한 번 확인하고 시작한다",
    "실행 중 E-stop과 cell을 감시하고 문제 있을 때 즉시 중단한다",
    "완료 결과와 남은 횟수를 확인한 뒤 같은 설정 또는 변경한 설정으로 계속한다",
)


class OperatorRuntime:
    """One idempotent owner for a constructed foreground operator runtime."""

    def __init__(self, *, bridge, announcement, startup_call=None, close_calls=()):
        self.bridge = bridge
        self.announcement = copy.deepcopy(dict(announcement))
        self.startup_call = startup_call
        self._close_calls = tuple(close_calls)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        seen = set()
        for close in self._close_calls:
            owner = getattr(close, "__self__", None)
            key = id(owner) if owner is not None else id(close)
            if key in seen:
                continue
            seen.add(key)
            while True:
                try:
                    close()
                    break
                except ContractError as exc:
                    if exc.code not in {
                        "OPERATOR_APPLICATION_START_OWNER_ACTIVE",
                        "OPERATOR_CONSOLE_START_OWNER_ACTIVE",
                    }:
                        raise
                    time.sleep(0.01)
        self._closed = True


def _load_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError("FAKE_CONSOLE_FIXTURE_ROOT")
    return load_json_strict(root / "hypothesis.json"), load_json_strict(root / "draft.json")


def build_fake_runtime(*, port: int = 4174, fixture_root: str | Path | None = None) -> OperatorRuntime:
    product = console = bridge = None
    try:
        if fixture_root is None:
            product = build_product_fake_operator()
            core = product.bridge_core
            root = Path(product.fixture_root)
        else:
            root = Path(fixture_root)
            hypothesis, draft = _load_fixture(root)
            console = build_fake_operator_console(
                hypothesis=hypothesis, draft=draft, fixture_root=root,
            )
            core = console.bridge_core
        bridge = LoopbackBridge(
            core=core, ui_root=ROOT / "operator-ui",
            host="127.0.0.1", port=port,
        )
        return OperatorRuntime(
            bridge=bridge,
            announcement={
                "status": "LISTENING", "url": bridge.origin,
                "effect_scope": "FAKE", "operator_identity": TEST_OPERATOR,
                "fixture_root": str(root), "product_flow": product is not None,
                "qa_workflow": FAKE_QA_WORKFLOW,
            },
            close_calls=tuple(
                close for close in (
                    getattr(console, "close", None),
                    getattr(product, "close", None),
                    bridge.server.server_close,
                ) if callable(close)
            ),
        )
    except Exception:
        if console is not None:
            console.close()
        if product is not None:
            product.close()
        if bridge is not None:
            bridge.server.server_close()
        raise


def build_physical_runtime(
    *, port: int = 4174, repository_root: str | Path = ROOT,
    session_id: str | None = None, operator_label: str = "local-operator",
    camera_device_id: str | None = None, job: str | Path = DEFAULT_JOB,
    gripper_retune: str | Path | None = None,
    data_mode: str = "GENERAL_COLLECTION",
    motion_preset: str | None = None,
    dataset_name: str = "fr5_smolvla_up_wrist_30hz",
    auto_prepare: bool = True,
    rollout_lifecycle: str | Path | None = None,
) -> OperatorRuntime:
    now = datetime.now(timezone.utc)
    if (
        data_mode not in {"TEST_COLLECTION", "GENERAL_COLLECTION"}
        or not isinstance(dataset_name, str)
        or SAFE_ID.fullmatch(dataset_name) is None
    ):
        raise ContractError("OPERATOR_RUNTIME_DATA_MODE")
    scope = "production" if data_mode == "GENERAL_COLLECTION" else "test-only"
    session_id = session_id or f"collection-{scope}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    application = bridge = None
    environment_holder: dict[str, Any] = {"active": None, "pending": None}
    try:
        repository = Path(repository_root).resolve(strict=True)
        camera_descriptors = discover_camera_devices()
        devices = [item["logical_id"] for item in camera_descriptors]
        catalog = load_operator_catalog(repository, device_ids=devices)
        initial_job = load_json_strict(_repository_path(repository, job))
        gripper_settings = _runtime_gripper_settings(
            repository, initial_job, gripper_retune,
        )
        requested_bindings = None
        if camera_device_id is not None:
            if camera_device_id not in devices:
                raise ContractError("PHYSICAL_CAMERA_BINDING_MISMATCH")
            profile = _v2_camera_profiles(repository).get(
                initial_job.get("collection_profile_id"),
            )
            if profile is None or len(profile["camera_roles"]) != 1:
                raise ContractError("OPERATOR_APPLICATION_COMPATIBLE_COMBINATION")
            requested_bindings = {device: "UNUSED" for device in devices}
            requested_bindings[camera_device_id] = profile["camera_roles"][0].upper()
        initial_camera_setup, initial_resolution = resolve_camera_setup(
            repository_root=repository, devices=camera_descriptors,
            preferred_profile_id=initial_job.get("collection_profile_id", ""),
            requested_bindings=requested_bindings,
        )
        selected_device = None
        if initial_resolution is not None:
            profile = initial_resolution["collection_profile"]
            if len(profile["camera_roles"]) == 1:
                selected_device = initial_resolution["role_bindings"]["bindings"][
                    profile["camera_roles"][0]
                ]["stable_device_id"]

        def blocked_environment(reason: str) -> dict[str, Any]:
            return {
                "schema_version": "data_factory.operator_environment.v1",
                "state": "BLOCKED",
                "observed_at": now.isoformat().replace("+00:00", "Z"),
                "components": {
                    name: {
                        "state": "MISSING" if name == "camera" else "BLOCKED",
                        "owner": None, "reason": reason,
                    }
                    for name in ("robot", "controller", "gripper", "camera")
                },
            }

        blocked = blocked_environment(
            initial_camera_setup["reason"] or "CAMERA_ROLE_BINDING_REQUIRED",
        )

        def select_camera_environment(
            profile: Mapping[str, Any] | None,
            camera_devices: Mapping[str, Mapping[str, str]],
        ) -> Mapping[str, Any]:
            current = environment_holder["active"] or environment_holder["pending"]
            if profile is None:
                if current is not None:
                    current.stop_cameras()
                    environment_holder["pending"] = current
                    return current.projection()
                return copy.deepcopy(blocked_environment(
                    "CAMERA_ROLE_BINDING_REQUIRED",
                ))
            if current is not None:
                projected = current.rebind_cameras(profile, camera_devices)
                environment_holder["pending"] = current
                return projected
            pending = build_physical_operator_environment(
                repository_root=repository, collection_profile=profile,
                camera_devices=camera_devices,
                gripper_readback_call=capture_gripper_setup_readback,
                gripper_maintenance_call=normalize_gripper_after_operator_ready,
                gripper_velocity_percent=gripper_settings["velocity_percent"],
                gripper_force_percent=gripper_settings["force_percent"],
                gripper_open_velocity_percent=gripper_settings[
                    "open_velocity_percent"
                ],
                gripper_open_force_percent=gripper_settings[
                    "open_force_percent"
                ],
            )
            environment_holder["pending"] = pending
            return pending.projection()

        def environment_call() -> Mapping[str, Any]:
            current = environment_holder["pending"] or environment_holder["active"]
            return copy.deepcopy(blocked) if current is None else current.liveness()

        def prepare_environment_call() -> Mapping[str, Any]:
            pending = environment_holder["pending"]
            if pending is None:
                return copy.deepcopy(blocked)
            environment_holder["active"] = pending
            return pending.prepare_environment()

        def prepare_environment_owner_call():
            pending = environment_holder["pending"]
            if pending is None:
                return (lambda: copy.deepcopy(blocked), lambda: None)
            environment_holder["active"] = pending

            def close_owned_environment() -> None:
                pending.stop()
                if environment_holder["pending"] is pending:
                    environment_holder["pending"] = None
                if environment_holder["active"] is pending:
                    environment_holder["active"] = None

            return pending.prepare_environment, close_owned_environment

        def home_recovery_prepare_call() -> Mapping[str, Any]:
            current = environment_holder["pending"] or environment_holder["active"]
            if current is None:
                raise ContractError("HOME_RECOVERY_ENVIRONMENT")
            return current.prepare_home_recovery()

        if initial_resolution is None:
            initial_environment = copy.deepcopy(blocked)
        else:
            camera_devices = {
                role: {
                    "kind": binding["device_kind"],
                    "stable_id": binding["stable_device_id"],
                    "capture_endpoint": binding["capture_endpoint"],
                }
                for role, binding in initial_resolution["role_bindings"]["bindings"].items()
            }
            initial_environment = select_camera_environment(
                initial_resolution["collection_profile"], camera_devices,
            )
        application, context = build_physical_operator_application(
            repository_root=repository, session_id=session_id,
            operator_label=operator_label,
            selected_camera_device_id=selected_device,
            discovery_call=discover_camera_devices,
            environment_call=environment_call,
            prepare_environment_call=prepare_environment_call,
            prepare_environment_owner_call=prepare_environment_owner_call,
            home_recovery_prepare_call=home_recovery_prepare_call,
            initial_environment=initial_environment, initial_catalog=catalog,
            initial_camera_devices=camera_descriptors, job_path=job,
            gripper_retune_path=gripper_retune,
            production_dataset_root=(
                repository / "datasets/fr5_episodes" / dataset_name
            ),
            initial_data_mode=data_mode,
            initial_motion_preset=motion_preset,
            rollout_lifecycle_path=rollout_lifecycle,
            camera_environment_call=select_camera_environment,
        )
        bridge = LoopbackBridge(
            core=application.bridge_core, ui_root=repository / "operator-ui",
            host="127.0.0.1", port=port,
            inspection_video_call=application.stored_reviews.video,
        )
        startup_call = None
        if initial_resolution is not None and auto_prepare:
            def startup_call():
                snapshot = application.bridge_core.snapshot()
                application.bridge_core.consume({
                    "schema_version": "data_factory.operator_intent.v1",
                    "intent_id": f"{session_id}-auto-prepare",
                    "session_id": snapshot["session_id"],
                    "view_revision": snapshot["revision"],
                    "view_digest": snapshot["view_digest"],
                    "op": "prepare_environment", "payload": {},
                })
        def close_environments() -> None:
            stopped = set()
            for environment in (
                environment_holder.get("pending"), environment_holder.get("active"),
            ):
                if environment is not None and id(environment) not in stopped:
                    stopped.add(id(environment))
                    environment.stop()
        return OperatorRuntime(
            bridge=bridge, startup_call=startup_call,
            announcement={
                "status": "LISTENING", "url": bridge.origin,
                "environment_state": initial_environment["state"],
                "qa_workflow": QA_WORKFLOW, **context,
            },
            close_calls=(
                application.close,
                close_environments,
                bridge.server.server_close,
            ),
        )
    except Exception:
        if application is not None:
            application.close()
        stopped = set()
        for environment in (
            environment_holder.get("pending"), environment_holder.get("active"),
        ):
            if environment is not None and id(environment) not in stopped:
                stopped.add(id(environment))
                environment.stop()
        if bridge is not None:
            bridge.server.server_close()
        raise


def build_operator_runtime(*, effect_scope: str = "FAKE", **kwargs) -> OperatorRuntime:
    if kwargs.get("rollout_lifecycle") is not None and effect_scope != "PHYSICAL":
        raise ContractError("COLLECTION_ROLLOUT_CONFIGURATION")
    learned_request = kwargs.pop("learned_request", None)
    if effect_scope == "LEARNED_RUN" or learned_request is not None:
        if (effect_scope != "LEARNED_RUN" or learned_request is None
                or any(kwargs.get(key) is not None for key in ("training_request", "training_output", "curator_run_id", "curator_run_root"))):
            raise ContractError("LEARNED_WEB_CONFIGURATION")
        from tools.data_factory.operator.workflow.learned_run import LearnedRunApplication
        application = LearnedRunApplication(payload=load_json_strict(Path(learned_request)),
            operator_label=kwargs.get("operator_label", "local-operator"), session_id=kwargs.get("session_id"),
            run_live_call=kwargs.get("run_live_call"),
            repository_root=kwargs.get("repository_root") or ROOT)
        bridge = LoopbackBridge(core=application.core, ui_root=ROOT / "operator-ui",
            port=kwargs.get("port", 4174), index_page="learned.html")
        return OperatorRuntime(bridge=bridge,
            announcement={"status": "LISTENING", "url": bridge.origin, "effect_scope": "LEARNED_RUN",
                          "run_id": application.payload["run_id"], "starts_motion": False},
            close_calls=(application.close, bridge.server.server_close))
    training_request = kwargs.pop("training_request", None)
    training_output = kwargs.pop("training_output", None)
    curator_run_id = kwargs.pop("curator_run_id", None)
    curator_run_root = kwargs.pop("curator_run_root", None)
    curator_paths = kwargs.pop("curator_paths", None)
    if (effect_scope == "CURATOR_REVIEW" or curator_run_id is not None
            or curator_run_root is not None or curator_paths is not None):
        if (effect_scope != "CURATOR_REVIEW" or not curator_run_id
                or training_request is not None or training_output is not None
                or curator_paths is not None and curator_run_root is not None):
            raise ContractError("CURATOR_REVIEW_CONFIGURATION")
        from dataclasses import replace
        from tools.data_factory.curator.workflow.application import DEFAULT_PATHS
        from tools.data_factory.operator.workflow.curator_review import CuratorReviewApplication

        paths = curator_paths or DEFAULT_PATHS
        if curator_run_root is not None:
            paths = replace(paths, run_root=Path(curator_run_root))
        application = CuratorReviewApplication(run_id=curator_run_id, paths=paths)
        application.bridge_core.snapshot()  # Fail before listening if the configured review is unavailable.
        bridge = LoopbackBridge(
            core=application.bridge_core, ui_root=ROOT / "operator-ui",
            port=kwargs.get("port", 4174), index_page="curator.html",
            review_video_call=application.review_video,
        )
        return OperatorRuntime(
            bridge=bridge, announcement={"status": "LISTENING", "url": bridge.origin,
                                        "effect_scope": "CURATOR_REVIEW", "run_id": curator_run_id,
                                        "starts_training": False},
            close_calls=(bridge.server.server_close,),
        )
    if effect_scope == "TRAINING_REVIEW" or training_request is not None or training_output is not None:
        if training_request is None or training_output is None or effect_scope != "TRAINING_REVIEW":
            raise ContractError("TRAINING_REVIEW_CONFIGURATION")
        from tools.data_factory.operator.workflow.training_review import TrainingReviewApplication

        application = TrainingReviewApplication(
            request=load_json_strict(Path(training_request)),
            output=Path(training_output),
            approved_by=kwargs.get("operator_label", "local-operator"),
        )
        bridge = LoopbackBridge(
            core=application.bridge_core, ui_root=ROOT / "operator-ui",
            port=kwargs.get("port", 4174), index_page="training.html",
        )
        return OperatorRuntime(
            bridge=bridge, announcement={"status": "LISTENING", "url": bridge.origin,
                                        "effect_scope": "TRAINING_REVIEW", "starts_training": False},
            close_calls=(application.close, bridge.server.server_close),
        )
    if effect_scope == "FAKE":
        return build_fake_runtime(
            port=kwargs.get("port", 4174),
            fixture_root=kwargs.get("fixture_root"),
        )
    if effect_scope == "PHYSICAL":
        kwargs.pop("fixture_root", None)
        return build_physical_runtime(**kwargs)
    raise ContractError("OPERATOR_EFFECT_SCOPE")


def build_physical_operator_console(
    *, repository_root: str | Path, session_id: str, run_id: str,
    operator_label: str, job_path: str | Path = DEFAULT_JOB,
    yaw0_sheet: str | Path = DEFAULT_YAW0,
    motion_qualification_path: str | Path = DEFAULT_MOTION,
    motion_preset: Mapping[str, str] | None = None,
    motion_preset_trial: bool = False,
    home_candidate_path: str | Path = DEFAULT_HOME,
    collection_profile_path: str | Path = DEFAULT_PROFILE,
    urdf_path: str | Path = DEFAULT_URDF,
    tcp_candidate_manifest: str | Path = DEFAULT_TCP_MANIFEST,
    gripper_retune_path: str | Path | None = DEFAULT_GRIPPER_RETUNE,
    job_binding: Mapping[str, str] | None = None,
    workspace_bindings: Mapping[str, Mapping[str, str]] | None = None,
    selected_camera_device_id: str | None = None,
    selected_camera_bindings: Mapping[str, str] | None = None,
    selected_camera_binding_digest: str | None = None,
    selected_camera_binding_set: Mapping[str, Any] | None = None,
    discovery_call: Callable[[], Sequence[object]] = discover_uvc_device_ids,
    activation_call: Callable[[], bool | Mapping[str, Any]] | None = None,
    snapshot_call: Callable[[], Mapping[str, Any]] | None = None,
    gripper_readback_call: Callable[[], Mapping[str, Any]] | None = None,
    gripper_maintenance_call: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    run_live_call: Callable[..., Mapping[str, Any]] = run_job.run_live,
    task_id: str | None = None,
    trajectory_variant_id: str = "DIRECT",
    requested_count: int = 1, normalized_seed: int = 0,
    candidate_poses: Sequence[Mapping[str, Any]] | None = None,
    direct_pose_sequence: Sequence[Mapping[str, Any]] | None = None,
    direct_yaw_sample_bindings: Sequence[Mapping[str, Any] | None] | None = None,
    yaw_sampling_profile: Mapping[str, Any] | None = None,
    state_space_design_profile: Mapping[str, Any] | None = None,
    direct_start_pose_ids: Sequence[str] | None = None,
    selected_start_pose_qualifications: Sequence[Mapping[str, Any]] | None = None,
    start_transition_call: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Mapping[str, Any]] | None = None,
    initial_object_pose: Mapping[str, Any] | None = None,
    initial_object_position: Mapping[str, Any] | None = None,
    data_disposition: str = "TEST_ONLY",
    dataset_root: str | Path | None = None,
    environment_prepared: bool = False,
    clock=None,
) -> tuple[OperatorConsole, dict[str, Any]]:
    """Compose one finite registered-workspace physical campaign without activation."""
    if type(motion_preset_trial) is not bool or motion_preset_trial and (
        data_disposition != "TEST_ONLY" or motion_preset is None
    ):
        raise ContractError("MOTION_PRESET_TRIAL_SCOPE")
    normalized_seed = validate_campaign_seed(normalized_seed)
    repository = Path(repository_root).resolve(strict=True)
    clock = clock or (lambda: datetime.now(timezone.utc))
    if (
        data_disposition not in {"TEST_ONLY", "PRODUCTION"}
        or trajectory_variant_id not in VARIANT_IDS
        or data_disposition == "TEST_ONLY" and dataset_root is not None
        or data_disposition == "PRODUCTION" and dataset_root is None
        or data_disposition == "PRODUCTION" and gripper_retune_path is not None
    ):
        raise ContractError("PHYSICAL_CONSOLE_DATA_DISPOSITION")
    paths = {
        "job": _repository_path(repository, job_path),
        "yaw0_sheet": _repository_path(repository, yaw0_sheet),
        "motion": _repository_path(repository, motion_qualification_path),
        "home": _repository_path(repository, home_candidate_path),
        "profile": _repository_path(repository, collection_profile_path),
        "urdf": _repository_path(repository, urdf_path),
        "tcp": _repository_path(repository, tcp_candidate_manifest),
    }
    if gripper_retune_path is not None:
        paths["gripper_retune"] = _repository_path(
            repository, gripper_retune_path,
        )
    template_job = load_json_strict(paths["job"])
    template_job["operator_or_agent_id"] = operator_label
    if job_binding is not None:
        expected_binding = {
            "place_id", "cell_calibration_id", "object_profile_id",
            "grasp_profile_id",
        }
        if (
            not isinstance(job_binding, Mapping)
            or set(job_binding) != expected_binding
            or any(
                not isinstance(value, str)
                or SAFE_ID.fullmatch(value) is None
                for value in job_binding.values()
            )
        ):
            raise ContractError("PHYSICAL_CONSOLE_JOB_BINDING")
        template_job.update(copy.deepcopy(dict(job_binding)))
    selected_task = template_job.get("task") if task_id is None else task_id
    recipe = get_task_recipe(selected_task)
    if selected_task != template_job.get("task") or job_binding is not None:
        object_profiles = []
        for path in sorted(
            (repository / "config/data_factory/objects").glob("*.json"),
            key=lambda value: str(value),
        ):
            profile = load_json_strict(path)
            if profile.get("object_profile_id") == template_job.get("object_profile_id"):
                object_profiles.append(profile)
        if len(object_profiles) != 1:
            raise ContractError("PHYSICAL_CONSOLE_TASK_PROFILE")
        template_job.update(
            task=selected_task,
            episode_intent=recipe["episode_intent"],
            instruction=task_instruction(
                selected_task, object_profiles[0]["description"],
            ),
        )
    configured_profile = load_json_strict(paths["profile"])
    if (
        configured_profile.get("schema_version")
        != "data_factory.collection_profile.v2"
        or not isinstance(configured_profile.get("camera_profile"), str)
        or not configured_profile["camera_profile"]
    ):
        raise ContractError("PHYSICAL_CONSOLE_COLLECTION_PROFILE")
    # TEST_ONLY derives the runtime JobSpec from the selected profile in memory;
    # the source job and its production digest remain unchanged on disk.
    template_job["collection_profile_id"] = configured_profile[
        "collection_profile_id"
    ]
    default_pose = {
        key: template_job[key] for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
    }
    initial_pose = copy.deepcopy(dict(initial_object_pose or default_pose))
    if set(initial_pose) != set(default_pose):
        raise ContractError("PHYSICAL_CONSOLE_SEQUENCE_ANCHOR")
    spatial_node_count = requested_count + int(template_job["task"] == "pick_place")
    if direct_pose_sequence is not None:
        if (
            not isinstance(direct_pose_sequence, (list, tuple))
            or len(direct_pose_sequence) != spatial_node_count
            or not direct_pose_sequence
            or dict(direct_pose_sequence[0]) != initial_pose
            or direct_start_pose_ids is not None
            and (
                not isinstance(direct_start_pose_ids, (list, tuple))
                or len(direct_start_pose_ids) != requested_count
            )
        ):
            raise ContractError("PHYSICAL_CONSOLE_SEQUENCE_ANCHOR")
        pose_domain = [copy.deepcopy(dict(item)) for item in direct_pose_sequence]
        if (
            direct_yaw_sample_bindings is not None
            and (
                not isinstance(direct_yaw_sample_bindings, (list, tuple))
                or len(direct_yaw_sample_bindings) != len(pose_domain)
                or any(
                    item is not None and not isinstance(item, Mapping)
                    for item in direct_yaw_sample_bindings
                )
            )
        ):
            raise ContractError("PHYSICAL_CONSOLE_YAW_BINDING")
    else:
        if (
            direct_start_pose_ids is not None
            or direct_yaw_sample_bindings is not None
            or template_job["task"] == "pick_place"
        ):
            raise ContractError("PHYSICAL_CONSOLE_DIRECT_SEQUENCE")
        pose_domain = [
            copy.deepcopy(dict(item))
            for item in (candidate_poses or [initial_pose])
        ]
        if initial_pose not in pose_domain:
            pose_domain.insert(0, initial_pose)
    if workspace_bindings is None:
        resolved_workspace_bindings = {
            template_job["place_id"]: {
                "frame_id": template_job["cell_calibration_id"],
                "selected_sheet": paths["yaw0_sheet"],
                "yaw0_sheet": paths["yaw0_sheet"],
                "motion_qualification": paths["motion"],
                "region_binding": validate_region_binding({
                    "layout_id": None, "layout_digest": None,
                    "region_id": None,
                    "physical_binding_status": "NOT_CONFIGURED",
                }),
            },
        }
    else:
        if not isinstance(workspace_bindings, Mapping) or not workspace_bindings:
            raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
        resolved_workspace_bindings = {}
        for place_id, binding in workspace_bindings.items():
            binding_fields = {
                "frame_id", "selected_sheet", "yaw0_sheet",
                "motion_qualification",
            }
            if (
                not isinstance(place_id, str) or not SAFE_ID.fullmatch(place_id)
                or not isinstance(binding, Mapping)
                or frozenset(binding) not in {
                    frozenset(binding_fields),
                    frozenset(binding_fields | {"region_binding"}),
                }
                or not isinstance(binding["frame_id"], str)
                or not SAFE_ID.fullmatch(binding["frame_id"])
            ):
                raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
            resolved_workspace_bindings[place_id] = {
                "frame_id": binding["frame_id"],
                "region_binding": validate_region_binding(
                    binding.get("region_binding", {
                        "layout_id": None, "layout_digest": None,
                        "region_id": None,
                        "physical_binding_status": "NOT_CONFIGURED",
                    }),
                ),
                **{
                    field: _repository_path(repository, binding[field])
                    for field in (
                        "selected_sheet", "yaw0_sheet",
                        "motion_qualification",
                    )
                },
            }
            validate_region_endpoint_authority(
                repository, place_id=place_id,
                frame_id=resolved_workspace_bindings[place_id]["frame_id"],
                region_binding=resolved_workspace_bindings[place_id][
                    "region_binding"
                ],
            )
    if (
        set(resolved_workspace_bindings)
        != {pose["place_id"] for pose in pose_domain}
        or resolved_workspace_bindings.get(
            template_job["place_id"], {},
        ).get("frame_id") != template_job["cell_calibration_id"]
    ):
        raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
    payload = {
        "mode": "live", "run_id": run_id, "job": template_job,
        "selected_sheet": str(paths["yaw0_sheet"]),
        "yaw0_sheet": str(paths["yaw0_sheet"]),
        "config_root": str(repository / "config/data_factory"),
        "motion_qualification": str(paths["motion"]),
        "home_candidate": str(paths["home"]), "urdf": str(paths["urdf"]),
        "expected_robot_system_id": template_job["robot_system_id"],
        "camera_profile": configured_profile["camera_profile"],
        run_job.TRAJECTORY_VARIANT_KEY: trajectory_variant_id,
        run_job.TRAJECTORY_SAMPLING_SEED_KEY: normalized_seed,
    }
    retune = (
        None if gripper_retune_path is None
        else load_json_strict(paths["gripper_retune"])
    )
    if motion_preset is not None:
        load_motion_preset(repository / "config/data_factory", motion_preset)
        payload["motion_preset"] = copy.deepcopy(dict(motion_preset))
    motion_qualification = load_json_strict(paths["motion"])
    endpoint_motion_qualifications = [
        load_json_strict(binding["motion_qualification"])
        for _place_id, binding in sorted(resolved_workspace_bindings.items())
    ]
    motion_qualification_by_cell = {
        item["cell_calibration_id"]: item
        for item in endpoint_motion_qualifications
    }

    def physical_resolver(value, *, scene_binding_call):
        resolved, program, binding = run_job.resolve_inputs(
            value, scene_binding_call=scene_binding_call,
            motion_preset_trial=motion_preset_trial,
        )
        if retune is not None:
            program = _derive_test_only_gripper_program(
                resolved, motion_qualification, program, retune,
            )
        return resolved, program, binding

    resolved_jobs = _resolve_physical_pose_domain(
        template_job=template_job, poses=pose_domain,
        operator_label=operator_label, payload_template=payload,
        sheet_manifest=paths["yaw0_sheet"],
        release_poses=(
            [
                yaw_preserving_destination(
                    source,
                    pose_domain[index + 1]
                    if index + 1 < len(pose_domain) else pose_domain[-2],
                )
                for index, source in enumerate(pose_domain)
            ]
            if template_job["task"] == "pick_place" else None
        ),
        workspace_bindings=resolved_workspace_bindings,
        resolver=physical_resolver,
    )
    resolved_by_pose = {
        tuple(item["normalized_job"][key] for key in (
            "place_id", "yaw_deg", "x_mm", "y_mm",
        )): item for item in resolved_jobs
    }
    initial_key = tuple(initial_pose[key] for key in (
        "place_id", "yaw_deg", "x_mm", "y_mm",
    ))
    resolved = resolved_by_pose.get(initial_key)
    if resolved is None:
        raise ContractError("PHYSICAL_CONSOLE_SEQUENCE_ANCHOR")
    direct_digests = None
    ordered_direct_resolved = None
    if direct_pose_sequence is not None:
        ordered_direct_resolved = []
        for pose in direct_pose_sequence:
            normalized_key = (
                pose["place_id"], normalize_yaw_deg(pose["yaw_deg"]),
                float(pose["x_mm"]), float(pose["y_mm"]),
            )
            matched = next((
                item for key, item in resolved_by_pose.items()
                if key[0] == normalized_key[0]
                and all(float(key[index]) == normalized_key[index] for index in range(1, 4))
            ), None)
            if matched is None:
                raise ContractError("PHYSICAL_CONSOLE_DIRECT_SEQUENCE")
            ordered_direct_resolved.append(matched)
        direct_digests = [
            item["resolved_job_digest"]
            for item in ordered_direct_resolved[:requested_count]
        ]
    task_bindings = None
    episode_instruction_bindings = None
    yaw_sample_bindings: list[dict[str, Any] | None] | None = None
    checked_yaw_profile = None
    checked_state_space_design_profile = None
    recorded_release_poses: list[dict[str, Any]] | None = None
    reposition_bindings: list[dict[str, Any] | None] | None = None
    if ordered_direct_resolved is not None:
        yaw_sample_bindings = (
            list(direct_yaw_sample_bindings)
            if direct_yaw_sample_bindings is not None
            else [None for _item in ordered_direct_resolved]
        )
        checked_yaw_profile = (
            None if yaw_sampling_profile is None
            else validate_yaw_sampling_profile(
                yaw_sampling_profile,
                object_profile=resolved["object_profile"],
                grasp_profile=resolved["grasp_profile"],
            )
        )
        if checked_yaw_profile is None and any(
            item is not None for item in yaw_sample_bindings
        ):
            raise ContractError("PHYSICAL_CONSOLE_YAW_PROFILE")
        if state_space_design_profile is not None:
            if checked_yaw_profile is None:
                raise ContractError("PHYSICAL_CONSOLE_STATE_SPACE_DESIGN")
            checked_state_space_design_profile = (
                validate_state_space_design_profile(
                    state_space_design_profile,
                    object_profile=resolved["object_profile"],
                    grasp_profile=resolved["grasp_profile"],
                    yaw_sampling_profile=checked_yaw_profile,
                )
            )
        for index, yaw_binding in enumerate(yaw_sample_bindings):
            if yaw_binding is None:
                continue
            slotted = yaw_binding.get("schema_version") == YAW_BINDING_SCHEMA
            if slotted and checked_state_space_design_profile is None:
                raise ContractError("PHYSICAL_CONSOLE_STATE_SPACE_DESIGN")
            yaw_sample_bindings[index] = validate_yaw_sample_binding(
                yaw_binding,
                profile=checked_yaw_profile,
                state_space_design_profile=(
                    checked_state_space_design_profile if slotted else None
                ),
            )

        def spatial_binding(
            item: Mapping[str, Any], role: str,
            pose_override: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            job = item["normalized_job"]
            endpoint = resolved_workspace_bindings[job["place_id"]]
            return {
                "role": role,
                "workspace_id": job["place_id"],
                "frame_id": job["cell_calibration_id"],
                "pose": copy.deepcopy(dict(pose_override)) if pose_override is not None else {
                    key: job[key]
                    for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
                },
                "sheet_digest": job["sheet_manifest_digest"],
                "family_digest": item["calibration"]["document"][
                    "a4_family_digest"
                ],
                "region_binding": copy.deepcopy(endpoint["region_binding"]),
            }

        task_bindings = []
        recorded_release_poses = []
        reposition_bindings = []
        for index, source in enumerate(
            ordered_direct_resolved[:requested_count],
        ):
            parent_run_id = (
                run_id if index == 0 else f"{run_id}-e{index + 1}"
            )
            next_run_id = (
                f"{run_id}-e{index + 2}"
                if index + 1 < requested_count else None
            )
            desired_release = (
                ordered_direct_resolved[index + 1]
                if index + 1 < len(ordered_direct_resolved) else source
            )
            source_pose = {
                key: source["normalized_job"][key]
                for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
            }
            desired_release_pose = {
                key: desired_release["normalized_job"][key]
                for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
            }
            recorded_release = (
                yaw_preserving_destination(source_pose, desired_release_pose)
                if template_job["task"] == "pick_place"
                else desired_release_pose
            )
            if template_job["task"] == "pick_place":
                _validate_recorded_release_region(
                    recorded_pose=recorded_release,
                    target_yaw_deg=desired_release_pose["yaw_deg"],
                    endpoint=resolved_workspace_bindings[
                        recorded_release["place_id"]
                    ],
                    resolved_destination=desired_release,
                )
            recorded_release_poses.append(recorded_release)
            reposition = None
            if template_job["task"] == "pickup_e2e":
                yaw_binding = yaw_sample_bindings[
                    index + 1 if index + 1 < len(yaw_sample_bindings) else index
                ]
                reposition = build_object_reposition_binding(
                    parent_run_id=parent_run_id,
                    continuation_run_id=parent_run_id,
                    next_run_id=next_run_id,
                    start_state="HELD_OBJECT",
                    source_pose=source_pose, target_pose=desired_release_pose,
                    object_profile=resolved["object_profile"],
                    grasp_profile=resolved["grasp_profile"],
                    yaw_sampling_profile=(
                        checked_yaw_profile if yaw_binding is not None else None
                    ),
                    yaw_sample_binding=yaw_binding,
                )
            elif (
                index + 1 < requested_count
                and not math.isclose(
                    float(recorded_release["yaw_deg"]),
                    float(desired_release_pose["yaw_deg"]),
                    rel_tol=0.0, abs_tol=1e-9,
                )
            ):
                yaw_binding = yaw_sample_bindings[index + 1]
                reposition = build_object_reposition_binding(
                    parent_run_id=parent_run_id,
                    continuation_run_id=f"{parent_run_id}-reposition",
                    next_run_id=next_run_id,
                    start_state="ON_SURFACE",
                    source_pose=recorded_release,
                    target_pose=desired_release_pose,
                    object_profile=resolved["object_profile"],
                    grasp_profile=resolved["grasp_profile"],
                    yaw_sampling_profile=(
                        checked_yaw_profile if yaw_binding is not None else None
                    ),
                    yaw_sample_binding=yaw_binding,
                )
            reposition_bindings.append(reposition)
            task_bindings.append(
                compile_task_binding(
                    template_job["task"],
                    source=spatial_binding(source, "SOURCE"),
                    **(
                        {
                            "destination": spatial_binding(
                                desired_release, "DESTINATION",
                                recorded_release,
                            ),
                        }
                        if template_job["task"] == "pick_place" else {
                            "next_source_reset": spatial_binding(
                                desired_release, "NEXT_SOURCE_RESET",
                            ),
                        }
                    ),
                )
            )
        episode_instruction_bindings = [
            compile_episode_instruction_binding(item, resolved["object_profile"])
            for item in task_bindings
        ]
    profile = resolved["collection_profile"]
    if canonical_digest(profile) != canonical_digest(configured_profile):
        raise ContractError("PHYSICAL_CONSOLE_COLLECTION_PROFILE")
    discovered_cameras = normalize_camera_devices(discovery_call())
    discovered_devices = [item["logical_id"] for item in discovered_cameras]
    if selected_camera_bindings is None:
        legacy_binding = _camera_binding(
            repository, profile, selected_device_id=selected_camera_device_id,
            discovery_call=lambda: discovered_devices,
        )
        role_device_ids = {
            legacy_binding["intended_role"]: legacy_binding["stable_device_id"],
        }
    else:
        role_device_ids = copy.deepcopy(dict(selected_camera_bindings))
        if (
            set(role_device_ids) != set(profile["camera_roles"])
            or any(discovered_devices.count(device) != 1 for device in role_device_ids.values())
            or len(set(role_device_ids.values())) != len(role_device_ids)
        ):
            raise ContractError("PHYSICAL_CAMERA_ROLE_BINDING_REQUIRED")
    role_map_digest = camera_binding_digest(profile, role_device_ids)
    if (
        selected_camera_binding_digest is not None
        and selected_camera_binding_digest != role_map_digest
    ):
        raise ContractError("PHYSICAL_CAMERA_BINDING_MISMATCH")
    assignments = {device: "UNUSED" for device in discovered_devices}
    assignments.update({device: role.upper() for role, device in role_device_ids.items()})
    if selected_camera_binding_set is not None:
        camera_binding_set = reuse_camera_role_bindings(
            selected_camera_binding_set,
            discovered_device_ids=discovered_cameras,
            collection_profile=profile,
        )
        if {
            role: binding["stable_device_id"]
            for role, binding in camera_binding_set["bindings"].items()
        } != role_device_ids:
            raise ContractError("PHYSICAL_CAMERA_BINDING_MISMATCH")
    elif selected_camera_bindings is None:
        camera_binding_set = {
            "schema_version": "data_factory.camera_role_bindings.v1",
            "collection_profile_id": profile["collection_profile_id"],
            "collection_profile_digest": canonical_digest(profile),
            "devices": {
                device: {
                    "kind": descriptor["kind"],
                    "capture_endpoint": descriptor["capture_endpoint"],
                }
                for device, descriptor in (
                    (item["logical_id"], item) for item in discovered_cameras
                )
            },
            "assignments": dict(sorted(assignments.items())),
            "bindings": {legacy_binding["intended_role"]: legacy_binding},
        }
        camera_binding_set["binding_digest"] = canonical_digest(camera_binding_set)
        camera_binding_set = validate_camera_role_bindings(camera_binding_set)
    else:
        camera_binding_set = build_camera_role_bindings(
            collection_profile=profile, discovered_device_ids=discovered_cameras,
            assignments=assignments,
        )
    camera_bindings = camera_binding_set["bindings"]
    read_gripper = gripper_readback_call or capture_gripper_setup_readback
    maintain_gripper = gripper_maintenance_call or normalize_gripper_after_operator_ready
    if type(environment_prepared) is not bool:
        raise ContractError("PHYSICAL_CONSOLE_ENVIRONMENT")
    physical_test_scene = data_disposition == "TEST_ONLY" and initial_object_position is not None
    first_roots = build_runtime_root_binding(
        repository, session_id=session_id, run_id=run_id,
        data_disposition=data_disposition, dataset_root=dataset_root,
        physical_scene=physical_test_scene,
    )
    job = resolved["normalized_job"]
    state_initialization = None
    if data_disposition == "TEST_ONLY" and not physical_test_scene:
        state_initialization = initialize_test_only_state_from_user_declaration(
            first_roots, repository_root=repository,
            robot_system_id=job["robot_system_id"],
            object_instance_id=(
                "test-object-"
                + canonical_digest({
                    "session_id": session_id,
                    "object_profile_id": job["object_profile_id"],
                }).removeprefix("sha256:")[:20]
            ),
            object_profile_id=job["object_profile_id"],
            place_id=job["place_id"], yaw_deg=job["yaw_deg"],
            x_mm=job["x_mm"], y_mm=job["y_mm"],
            declared_by=operator_label,
        )
        initial_scene_digest = state_initialization["scene_state_digest"]
    else:
        scene_store = SceneStateStore(first_roots["cell_root"], job["robot_system_id"])
        before = scene_store.snapshot()
        matching = [
            item for item in before["scene_state"]["objects"].values()
            if item.get("object_profile_id") == job["object_profile_id"]
        ]
        if len(matching) > 1:
            raise ContractError("SCENE_OBJECT_AMBIGUOUS")
        if physical_test_scene and not matching:
            raise ContractError("SCENE_OBJECT_NOT_READY")
        instance_id = (
            matching[0]["instance_id"] if matching else
            "production-object-" + canonical_digest({
                "robot_system_id": job["robot_system_id"],
                "object_profile_id": job["object_profile_id"],
            }).removeprefix("sha256:")[:20]
        )
        if matching:
            position = scene_store.object_position(
                object_profile_id=job["object_profile_id"],
                dimensions_mm=resolved["object_profile"]["dimensions_mm"],
            )
            if initial_object_position is not None and position != initial_object_position:
                raise ContractError("OBJECT_POSITION_CHANGED")
            if position["status"] != "AVAILABLE":
                raise ContractError(position["reason"])
            if position["pose"] != initial_pose:
                raise ContractError("OBJECT_POSITION_CHANGED")
            observed = (
                scene_store.rebind_landed_source(
                    object_profile_id=job["object_profile_id"],
                    dimensions_mm=resolved["object_profile"]["dimensions_mm"],
                    run_id=run_id,
                    expected_scene_digest=position["scene_state_digest"],
                    expected_slot_digest=position["slot_digest"],
                    expected_cell_digest=position["cell_state_digest"],
                ) if not physical_test_scene and position["source"] in {"ROBOT_RELEASE", "ROBOT_RELEASE_PROXY"}
                else before
            )
        else:
            if initial_object_position is not None and (
                initial_object_position["scene_state_digest"] != before["scene_state_digest"]
            ):
                raise ContractError("OBJECT_POSITION_CHANGED")
            observed = scene_store.update_object(
                instance_id=instance_id,
                object_profile_id=job["object_profile_id"], state="ON_SURFACE",
                pose={key: job[key] for key in ("place_id", "yaw_deg", "x_mm", "y_mm")},
                source="HUMAN", updated_by=operator_label,
                expected_revision=before["scene_state"]["revision"],
            )
        initial_scene_digest = observed["scene_state_digest"]
        if physical_test_scene:
            state_initialization = bind_test_only_physical_scene(
                first_roots, repository_root=repository,
                robot_system_id=job["robot_system_id"],
                object_profile_id=job["object_profile_id"],
                dimensions_mm=resolved["object_profile"]["dimensions_mm"],
                observed_by=operator_label,
            )
            if state_initialization["scene_state_digest"] != initial_scene_digest:
                raise ContractError("OBJECT_POSITION_CHANGED")
    home_candidate = load_json_strict(paths["home"])
    qualification_source = (
        "SYNTHETIC_TEST_ONLY"
        if data_disposition == "TEST_ONLY" else "QUALIFICATION_ARTIFACT"
    )
    live_start_qualifications: dict[str, dict[str, Any]] = {}
    campaign_start_qualifications = None
    if selected_start_pose_qualifications is not None:
        campaign_start_qualifications = []
        for source_value in selected_start_pose_qualifications:
            if not isinstance(source_value, Mapping):
                raise ContractError("PHYSICAL_CONSOLE_START_POSE")
            qualification = copy.deepcopy(dict(source_value))
            identifier = qualification.get("robot_start_pose_id")
            if (
                not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier)
                or identifier in live_start_qualifications
                or qualification.get("home_candidate_digest")
                != canonical_digest(home_candidate)
            ):
                raise ContractError("PHYSICAL_CONSOLE_START_POSE")
            if qualification.get("source") == "QUALIFICATION_ARTIFACT":
                live_start_qualifications[identifier] = copy.deepcopy(qualification)
            qualification["source"] = qualification_source
            qualification["qualification_digest"] = canonical_digest({
                key: value for key, value in qualification.items()
                if key != "qualification_digest"
            })
            campaign_start_qualifications.append(qualification)
        campaign_start_qualifications.sort(
            key=lambda value: value["robot_start_pose_id"],
        )
    hypothesis, draft = _build_physical_campaign_contract(
        resolver_results=resolved_jobs, motion_qualification=motion_qualification,
        motion_qualifications=endpoint_motion_qualifications,
        home_candidate=home_candidate,
        scene_digest=initial_scene_digest,
        draft_id=f"{session_id}-draft", manifest_id=f"{session_id}-manifest",
        requested_count=requested_count,
        normalized_seed=normalized_seed,
        anchor_resolved_job_digest=(
            None if direct_digests is not None else resolved["resolved_job_digest"]
        ),
        direct_resolved_job_digests=direct_digests,
        direct_start_pose_ids=direct_start_pose_ids,
        start_pose_qualifications=campaign_start_qualifications,
        test_only_gripper_retune_digest=(
            None if retune is None else retune["retune_digest"]
        ),
        qualification_source=qualification_source,
        motion_recipe=trajectory_variant_id,
        state_space_design_profile=checked_state_space_design_profile,
    )
    resolved_by_digest = {
        item["resolved_job_digest"]: item
        for item in hypothesis["resolver_receipts"]
    }
    campaign_run_ids = [
        run_id if index == 0 else f"{run_id}-e{index + 1}"
        for index in range(requested_count)
    ]
    if (
        len(set(campaign_run_ids)) != requested_count
        or any(len(value) > 128 or SAFE_ID.fullmatch(value) is None for value in campaign_run_ids)
    ):
        raise ContractError("PHYSICAL_CONSOLE_RUN_ID")

    def run_id_for(index: int) -> str:
        if type(index) is not int or not 0 <= index < len(campaign_run_ids):
            raise ContractError("PHYSICAL_CONSOLE_RUN_ID")
        return campaign_run_ids[index]

    def roots_for(active_run_id: str) -> dict[str, Any]:
        if active_run_id not in campaign_run_ids:
            raise ContractError("PHYSICAL_CONSOLE_RUN_ID")
        return build_runtime_root_binding(
            repository, session_id=session_id, run_id=active_run_id,
            data_disposition=data_disposition, dataset_root=dataset_root,
            physical_scene=physical_test_scene,
        )

    counters = {name: 0 for name in SIDE_EFFECT_COUNTERS}
    holder: dict[str, Any] = {}

    def campaign_camera_warmup(
        active_payload: Mapping[str, Any], active_profile: Mapping[str, Any],
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Measure once per exact camera binding; keep episode readiness fresh."""
        return _campaign_camera_warmup(
            cache=holder, transport=holder.get("camera_transport_evidence"),
            payload=active_payload, profile=active_profile,
            cancel=cancel_event, measure_call=run_job._camera_warmup,
        )

    def physical_gate_evidence() -> dict[str, Any]:
        try:
            value = activation_call() if activation_call is not None else None
            if activation_call is None:
                transports = {}
                for role in profile["camera_roles"]:
                    binding = camera_bindings[role]
                    transports[role] = passive_physical_gate(
                        camera_topic=profile["camera_topics"][role],
                        camera_node=(
                            f"/camera/{role}/color/uvc_{role}_camera"
                            if binding["device_kind"] == "UVC"
                            else f"/camera/{role}"
                        ),
                        device_kind=binding["device_kind"],
                        capture_endpoint=binding["capture_endpoint"],
                        discovered_device_id=binding["stable_device_id"],
                        discovery_call=lambda: discovered_cameras,
                    )
                value = {
                    "schema_version": (
                        f"data_factory.{data_disposition.lower()}_camera_transport_set.v1"
                    ),
                    "camera_binding_digest": role_map_digest,
                    "roles": transports,
                    "status": "PASSIVE_GRAPH_VERIFIED",
                }
                value["binding_digest"] = canonical_digest(value)
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("CAMPAIGN_OPERATOR_PHYSICAL_ACTIVATION_FAILED") from exc
        if value is True:
            evidence = {
                "schema_version": (
                    f"data_factory.{data_disposition.lower()}_camera_transport_set.v1"
                ),
                "camera_binding_digest": role_map_digest,
                "roles": {
                    role: {
                        "stable_device_id": binding["stable_device_id"],
                        "status": "INJECTED_TEST_GATE",
                    }
                    for role, binding in camera_bindings.items()
                },
                "status": "INJECTED_TEST_GATE",
            }
            evidence["binding_digest"] = canonical_digest(evidence)
            return evidence
        if not isinstance(value, Mapping):
            raise ContractError("CAMPAIGN_OPERATOR_PHYSICAL_ACTIVATION_FAILED")
        evidence = copy.deepcopy(dict(value))
        if (
            evidence.get("camera_binding_digest") != role_map_digest
            or set(evidence.get("roles", {})) != set(camera_bindings)
            or evidence.get("binding_digest")
            != canonical_digest({
                key: item for key, item in evidence.items()
                if key != "binding_digest"
            })
        ):
            raise ContractError("PHYSICAL_CAMERA_BINDING_MISMATCH")
        return evidence

    def refresh_gripper() -> dict[str, Any]:
        try:
            readback = read_gripper()
            projection = gripper_setup_projection(readback)
        except ContractError as exc:
            holder["gripper_setup_error"] = exc.code
            holder["gripper_readback"] = None
            projection = gripper_setup_projection(None)
        projection["maintenance_call_count"] = (
            holder.get("gripper_projection", {}).get("maintenance_call_count", 0)
        )
        holder["gripper_projection"] = copy.deepcopy(projection)
        if projection["state"] != "NOT_AVAILABLE":
            holder["gripper_setup_error"] = None
            holder["gripper_readback"] = copy.deepcopy(dict(readback))
        return copy.deepcopy(projection)

    initial_gripper = (
        {
            "state": "ATTACHED", "supported_action": "VERIFY_AT_RUN",
            "maintenance_call_count": 0,
        }
        if environment_prepared else refresh_gripper()
    )
    holder["gripper_projection"] = copy.deepcopy(initial_gripper)
    setup_request = None
    initial_block_code = None
    if initial_gripper["state"] == "MAINTENANCE_APPROVAL_REQUIRED":
        setup_binding_digest = canonical_digest({
            "schema_version": "data_factory.gripper_setup_binding.v1",
            "run_id": run_id,
            "readback_digest": initial_gripper["readback_digest"],
            "operation": initial_gripper["supported_action"],
            "gripper_index": 1, "authority": "SETUP_ONLY",
        })
        setup_request = {
            "schema_version": "data_factory.operator_checkpoint_request.v1",
            "kind": "GRIPPER_MAINTENANCE", "run_id": run_id,
            "plan_digest": setup_binding_digest,
            "prompt": (
                "Confirm the gripper is empty and physically clear before one "
                f"{data_disposition} open-normalization action."
            ),
            "choices": ["READY", "CANCEL"],
            "evidence": {
                "setup_only": True, "plan_exists": False,
                "gripper_index": 1,
                "operation": initial_gripper["supported_action"],
                "readback_digest": initial_gripper["readback_digest"],
                "empty_gripper": "OPERATOR_CONFIRM_REQUIRED",
                "finger_and_cell_clear": "OPERATOR_CONFIRM_REQUIRED",
                "authority": "SETUP_ONLY",
            },
            "timeout_s": None,
        }
    elif initial_gripper["state"] != "ATTACHED":
        initial_block_code = (
            "GRIPPER_BINDING_MISMATCH"
            if initial_gripper["state"] == "BLOCKED_BINDING"
            else holder.get("gripper_setup_error") or "GRIPPER_SETUP_NOT_AVAILABLE"
        )
    if (
        not environment_prepared
        and setup_request is None
        and initial_block_code is None
    ):
        try:
            holder["camera_transport_evidence"] = physical_gate_evidence()
        except ContractError as exc:
            initial_block_code = exc.code

    def scene_evidence(_run_id: str) -> dict[str, Any]:
        snapshot = SceneStateStore(
            first_roots["cell_root"], resolved["normalized_job"]["robot_system_id"],
        ).snapshot()
        value = {
            "schema_version": "data_factory.scene_freshness_evidence.v1",
            "scene_digest": snapshot["scene_state_digest"],
            "observed_at": clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        value["evidence_digest"] = canonical_digest(value)
        return value

    def unused_port(_request):
        raise ContractError("PHYSICAL_CONSOLE_PORT_NOT_ATTACHED")

    def fresh_one_job() -> OneJob:
        counters["physical_factory"] += 1
        return OneJob(
            unused_port, unused_port,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT,
            allow_synthetic_test_operator=data_disposition == "TEST_ONLY",
        )

    def resolve_gripper_setup(_decision: Mapping[str, Any]) -> dict[str, Any]:
        current = refresh_gripper()
        if current["state"] == "ATTACHED":
            holder["operator"].subsystems["gripper"] = {
                "readiness": "READY", "capability": "ATTACH",
                "reason": "FRESH_CONTROLLER_READBACK",
            }
            return current
        if (
            current["state"] != "MAINTENANCE_APPROVAL_REQUIRED"
            or current.get("readback_digest") != initial_gripper.get("readback_digest")
        ):
            raise ContractError("GRIPPER_MAINTENANCE_STALE")
        counters["gripper"] += 1
        result = maintain_gripper(copy.deepcopy(holder["gripper_readback"]))
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "NORMALIZED"
            or type(result.get("requires_graph_switch")) is not bool
        ):
            raise ContractError("GRIPPER_MAINTENANCE_ACTION")
        if result["requires_graph_switch"]:
            holder["gripper_projection"] = {
                **current, "state": "NORMAL_GRAPH_REQUIRED",
                "supported_action": "RESTART_FOREGROUND_NORMAL_GRAPH",
                "maintenance_call_count": 1,
            }
            raise ContractError("GRIPPER_NORMAL_GRAPH_REQUIRED")
        refreshed = refresh_gripper()
        if refreshed["state"] != "ATTACHED":
            raise ContractError("GRIPPER_MAINTENANCE_RECHECK")
        refreshed["maintenance_call_count"] = 1
        holder["gripper_projection"] = copy.deepcopy(refreshed)
        holder["operator"].subsystems["gripper"] = {
            "readiness": "READY", "capability": "ATTACH",
            "reason": "APPROVED_OPEN_NORMALIZATION",
        }
        return refreshed

    def activate() -> bool:
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="physical-runtime-gate",
        ) as executor:
            gripper_future = executor.submit(refresh_gripper)
            camera_future = executor.submit(physical_gate_evidence)
            gripper = gripper_future.result()
            camera_transport_evidence = camera_future.result()
        if gripper["state"] != "ATTACHED":
            raise ContractError("GRIPPER_SETUP_NOT_AVAILABLE")
        holder["camera_transport_evidence"] = camera_transport_evidence
        return True

    def prepare_campaign():
        nonlocal state_initialization
        if physical_test_scene:
            store = SceneStateStore(first_roots["cell_root"], job["robot_system_id"])
            current = store.object_position(
                object_profile_id=job["object_profile_id"],
                dimensions_mm=resolved["object_profile"]["dimensions_mm"],
            )
            if current != state_initialization["source_position"]:
                raise ContractError("OBJECT_POSITION_CHANGED")
            expected_digest = current["scene_state_digest"]
            if current["source"] in {"ROBOT_RELEASE", "ROBOT_RELEASE_PROXY"}:
                rebound = store.rebind_landed_source(
                    object_profile_id=job["object_profile_id"],
                    dimensions_mm=resolved["object_profile"]["dimensions_mm"], run_id=run_id,
                    expected_scene_digest=current["scene_state_digest"],
                    expected_slot_digest=current["slot_digest"],
                    expected_cell_digest=current["cell_state_digest"],
                )
                expected_digest = rebound["scene_state_digest"]
            refreshed = bind_test_only_physical_scene(
                first_roots, repository_root=repository,
                robot_system_id=job["robot_system_id"], object_profile_id=job["object_profile_id"],
                dimensions_mm=resolved["object_profile"]["dimensions_mm"], observed_by=operator_label,
            )
            if (refreshed["scene_state_digest"] != expected_digest
                    or refreshed["source_position"]["cell_state_digest"] != current["cell_state_digest"]):
                raise ContractError("OBJECT_POSITION_CHANGED")
            state_initialization = refreshed
            holder["operator"].initial_scene_digest = state_initialization["scene_state_digest"]

    def start_binding(
        _run_id: str, slot: Mapping[str, Any], cancel_event,
    ) -> dict[str, Any]:
        if cancel_event.is_set():
            raise ContractError("CAMPAIGN_SESSION_CANCELLED")
        base = next((
            item for item in hypothesis["base_conditions"]
            if item["base_condition_digest"] == slot.get("base_condition_digest")
        ), None)
        receipt = next((
            item for item in hypothesis["resolver_receipts"]
            if isinstance(base, Mapping)
            and item["resolver_result_digest"] == base["resolver_result_digest"]
        ), None)
        endpoint_motion = (
            motion_qualification_by_cell.get(
                receipt["normalized_job"]["cell_calibration_id"]
            )
            if isinstance(receipt, Mapping) else None
        )
        if not isinstance(endpoint_motion, Mapping):
            raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
        policy = {}
        if motion_preset is not None:
            endpoint = resolved_workspace_bindings[receipt["normalized_job"]["place_id"]]
            if canonical_digest(load_json_strict(endpoint["motion_qualification"])) != canonical_digest(endpoint_motion):
                raise ContractError("MOTION_PRESET_BINDING")
            selected_policy = load_motion_preset(repository / "config/data_factory", motion_preset)
            if not motion_preset_trial:
                policy["motion_preset"] = selected_policy
        home_snapshot = (
            snapshot_call() if snapshot_call is not None
            else capture_home_snapshot(tcp_candidate_manifest=paths["tcp"])
        )
        checked_home = _validate_snapshot(copy.deepcopy(dict(home_snapshot)))
        max_age = min(0.1, float(endpoint_motion["max_joint_state_age_s"]))
        if max(
            checked_home["joint_state_age_s"],
            checked_home["ros_sample_age_s"],
        ) > max_age:
            raise ContractError("TEST_ONLY_START_STALE")
        home_target = endpoint_motion["qualified_safe_joint_positions_rad"]
        home_tolerance = float(endpoint_motion["goal_tolerances"]["joint_rad"])
        if any(
            abs(checked_home["joint_positions_rad"][joint] - target)
            > home_tolerance
            for joint, target in zip(SNAPSHOT_JOINTS, home_target)
        ):
            raise ContractError("TEST_ONLY_START_OUTSIDE_HOME")
        if cancel_event.is_set():
            raise ContractError("CAMPAIGN_SESSION_CANCELLED")
        start_pose_id = slot.get("robot_start_pose_id")
        qualification = live_start_qualifications.get(start_pose_id)
        if qualification is not None:
            if start_transition_call is None:
                from tools.data_factory.motion.home_recovery import (
                    transition_to_start_live,
                )
                transition = transition_to_start_live(
                    motion_qualification=endpoint_motion,
                    robot_start_pose_qualification=qualification,
                    cancel_event=cancel_event,
                    **policy,
                )
            else:
                transition = start_transition_call(
                    endpoint_motion, qualification, cancel_event,
                )
            if (
                not isinstance(transition, Mapping)
                or transition.get("status") not in {
                    "AT_START", "ALREADY_AT_START",
                }
                or transition.get("robot_start_pose_id") != start_pose_id
            ):
                raise ContractError("PHYSICAL_CONSOLE_START_TRANSITION")
        if cancel_event.is_set():
            raise ContractError("CAMPAIGN_SESSION_CANCELLED")
        selected_snapshot = (
            snapshot_call() if snapshot_call is not None
            else capture_home_snapshot(tcp_candidate_manifest=paths["tcp"])
        ) if qualification is not None else checked_home
        binding = build_runtime_start_binding(
            data_disposition=data_disposition,
            manifest=holder["operator"].manifest, hypothesis=hypothesis,
            motion_qualification=endpoint_motion,
            home_candidate=home_candidate,
            current_snapshot=selected_snapshot, slot=slot,
        )
        if cancel_event.is_set():
            raise ContractError("CAMPAIGN_SESSION_CANCELLED")
        return binding

    def episode(
        intent, lifecycle, cancel_event, episode_context,
        decision_provider, checkpoint_provider,
    ):
        active_roots = episode_context["root_binding"]
        active_payload = copy.deepcopy(payload)
        active_resolved = resolved_by_digest.get(
            intent["base_condition"]["resolved_job_digest"],
        )
        if active_resolved is None:
            raise ContractError("PHYSICAL_CONSOLE_RESOLVED_JOB")
        active_job = active_resolved["normalized_job"]
        source_endpoint = resolved_workspace_bindings.get(
            active_job["place_id"],
        )
        if (
            not isinstance(source_endpoint, Mapping)
            or source_endpoint["frame_id"] != active_job["cell_calibration_id"]
        ):
            raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
        active_payload.update(
            job=copy.deepcopy(active_job),
            selected_sheet=str(source_endpoint["selected_sheet"]),
            yaw0_sheet=str(source_endpoint["yaw0_sheet"]),
            motion_qualification=str(source_endpoint["motion_qualification"]),
        )
        trajectory_design = trajectory_sampling_binding(
            normalized_seed, intent["slot"],
            holder["operator"].manifest["slots"],
        )
        active_payload.update(
            run_id=intent["run_id"], run_root=active_roots["run_root"],
            dataset_root=active_roots["dataset_root"],
            trajectory_sampling_seed=trajectory_design.pop("sampling_seed"),
            trajectory_sampling_design=trajectory_design,
        )
        order_index = intent["order_index"]
        next_slot = (
            holder["operator"].manifest["slots"][order_index + 1]
            if order_index + 1 < len(holder["operator"].manifest["slots"])
            else None
        )
        next_run_id = (
            run_id_for(order_index + 1)
            if next_slot is not None
            else None
        )
        release_resolved = active_resolved
        destination_slot = next_slot
        reposition_binding = (
            reposition_bindings[order_index]
            if reposition_bindings is not None else None
        )
        if active_job["task"] == "pick_place":
            if (
                ordered_direct_resolved is None
                or recorded_release_poses is None
            ):
                raise ContractError("TASK_DESTINATION_REQUIRED")
            release_resolved = ordered_direct_resolved[order_index + 1]
        elif destination_slot is not None:
            next_base = next(
                item for item in hypothesis["base_conditions"]
                if item["base_condition_digest"]
                == destination_slot["base_condition_digest"]
            )
            release_resolved = resolved_by_digest.get(
                next_base["resolved_job_digest"],
            )
            if release_resolved is None:
                raise ContractError("PHYSICAL_CONSOLE_RESOLVED_JOB")
        release_job = copy.deepcopy(release_resolved["normalized_job"])
        if active_job["task"] == "pick_place":
            release_job.update(recorded_release_poses[order_index])
            release_job["job_id"] = (
                f"{intent['run_id']}-recorded-destination"
            )
        if active_job["task"] == "pick_place" and all(
            active_job[key] == release_job[key]
            for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
        ):
            raise ContractError("TASK_BINDING_DISTINCT")
        release_endpoint = resolved_workspace_bindings.get(
            release_job["place_id"],
        )
        if (
            not isinstance(release_endpoint, Mapping)
            or release_endpoint["frame_id"]
            != release_job["cell_calibration_id"]
        ):
            raise ContractError("PHYSICAL_CONSOLE_WORKSPACE_BINDING")
        if release_job["place_id"] == active_job["place_id"]:
            active_payload.update(
                recycle_yaw_deg=release_job["yaw_deg"],
                recycle_x_mm=release_job["x_mm"],
                recycle_y_mm=release_job["y_mm"],
            )
            active_payload.pop("destination", None)
        else:
            for key in (*run_job.RECYCLE_COORD_KEYS, run_job.RECYCLE_YAW_KEY):
                active_payload.pop(key, None)
            active_payload["destination"] = {
                "job": copy.deepcopy(release_job),
                "selected_sheet": str(release_endpoint["selected_sheet"]),
                "yaw0_sheet": str(release_endpoint["yaw0_sheet"]),
                "motion_qualification": str(
                    release_endpoint["motion_qualification"]
                ),
            }
        reposition_source_payload = None
        if (
            reposition_binding is not None
            and reposition_binding["start_state"] == "ON_SURFACE"
        ):
            reposition_source_payload = copy.deepcopy(active_payload)
            reposition_source_payload.update(
                job=copy.deepcopy(release_resolved["normalized_job"]),
                selected_sheet=str(release_endpoint["selected_sheet"]),
                yaw0_sheet=str(release_endpoint["yaw0_sheet"]),
                motion_qualification=str(
                    release_endpoint["motion_qualification"]
                ),
            )
            reposition_source_payload["job"].update(
                reposition_binding["source_pose"],
                job_id=reposition_binding["continuation_run_id"],
            )

        def episode_resolver(value):
            return run_job.resolve_campaign_episode_inputs(
                value,
                release_role=(
                    "DESTINATION_THEN_NEXT_SOURCE"
                    if next_run_id is not None else "RELEASE_DESTINATION"
                ),
                next_run_id=(
                    reposition_binding["continuation_run_id"]
                    if reposition_binding is not None
                    and reposition_binding["start_state"] == "ON_SURFACE"
                    else next_run_id
                ),
                cell_root=active_roots["cell_root"],
                resolver=physical_resolver,
            )

        scene_source: dict[str, Any]
        if data_disposition == "TEST_ONLY" and order_index == 0:
            scene_source = {"state_initialization": state_initialization}
        else:
            live_resolved, _program, active_scene_binding = episode_resolver(active_payload)
            if canonical_digest(live_resolved) != active_resolved["resolver_result_digest"]:
                raise ContractError("PHYSICAL_CONSOLE_RESOLVED_JOB")
            scene_source = {
                "scene_binding": active_scene_binding,
                "scene_evidence": scene_evidence(intent["run_id"]),
                "observed_by": operator_label,
            }
        place_alias = "place1" if active_job["place_id"] == "PLACE_A" else active_job["place_id"]
        episode_binding = build_runtime_episode_binding(
            roots=active_roots, repository_root=repository,
            manifest=holder["operator"].manifest, hypothesis=hypothesis,
            intent=intent, start_binding=episode_context["start_binding"],
            resolved_job=active_resolved, place_alias=place_alias, clock=clock,
            **scene_source,
        )
        transport = holder.get("camera_transport_evidence")
        if not isinstance(transport, Mapping):
            raise ContractError("PHYSICAL_CAMERA_BINDING_MISMATCH")
        task_recipe = get_task_recipe(active_job["task"])
        current_yaw_sample = (
            None if yaw_sample_bindings is None
            else yaw_sample_bindings[order_index]
        )
        checklist = {
            "schema_version": "data_factory.site_checklist.v1",
            "place_alias": place_alias, "place_id": active_job["place_id"],
            "cell_calibration_id": active_job["cell_calibration_id"],
            "yaw_deg": active_job["yaw_deg"],
            "x_mm": active_job["x_mm"], "y_mm": active_job["y_mm"],
            "object_profile_id": active_job["object_profile_id"],
            "grasp_profile_id": active_job["grasp_profile_id"],
            "task": active_job["task"],
            "motion_recipe": intent["fixed_contract"]["motion_recipe"],
            "full_return_step_count": (
                len(task_recipe["recorded_phases"])
                + len(task_recipe["post_recording_phases"])
            ),
            "robot_start_pose_id": intent["slot"]["robot_start_pose_id"],
            "gripper_empty": "OPERATOR_CONFIRM_REQUIRED",
            "cell_clear": "OPERATOR_CONFIRM_REQUIRED",
            "estop_monitoring": "OPERATOR_CONFIRM_REQUIRED",
            "camera_state": "CONNECTED_UNPLACED",
            "camera_profile_id": profile["collection_profile_id"],
            "camera_transport_binding_digest": transport["binding_digest"],
            "episode_number": order_index + 1,
            "episode_limit": requested_count,
            "data_disposition": data_disposition,
            "object_reposition_binding": copy.deepcopy(reposition_binding),
            **(
                {"yaw_sample_binding": copy.deepcopy(current_yaw_sample)}
                if current_yaw_sample is not None else {}
            ),
            **(
                {
                    "task_binding": copy.deepcopy(task_bindings[order_index]),
                    "episode_instruction_binding": copy.deepcopy(
                        episode_instruction_bindings[order_index]
                    ),
                }
                if task_bindings is not None else {}
            ),
        }
        holder["last_live_response"] = None
        try:
            live = run_live_call(
                active_payload, cancel_event, holder["console"].publish_runtime,
                resolver=episode_resolver,
                one_job=lifecycle, decision_provider=decision_provider,
                checkpoint_provider=checkpoint_provider,
                approval_scope="HIL_NUMERIC_PROXY",
                runtime_root_binding=episode_context["root_binding"],
                runtime_episode_binding=episode_binding,
                runtime_start_binding=episode_context["start_binding"],
                episode_ledger_context={
                    "manifest": holder["operator"].manifest,
                    "intent": intent,
                    "compiled_authoring": holder["operator"].compiled_authoring_evidence(),
                },
                preapproval_checklist=checklist,
                episode_instruction_binding=(
                    None if episode_instruction_bindings is None
                    else episode_instruction_bindings[order_index]
                ),
                yaw_sample_binding=current_yaw_sample,
                yaw_sampling_profile=(
                    checked_yaw_profile
                    if current_yaw_sample is not None else None
                ),
                state_space_design_profile=(
                    checked_state_space_design_profile
                    if current_yaw_sample is not None
                    and current_yaw_sample.get("schema_version")
                    == YAW_BINDING_SCHEMA else None
                ),
                object_reposition_binding=reposition_binding,
                object_reposition_resolver=physical_resolver,
                object_reposition_source_payload=reposition_source_payload,
                campaign_authorization=holder["console"].campaign_authorization,
                dataset_validation_scope="INCREMENTAL",
                camera_warmup_call=campaign_camera_warmup,
                candidate_writer_enabled=data_disposition == "PRODUCTION",
                repository_root=repository,
                **({"motion_preset_trial": True} if motion_preset_trial else {}),
            )
        except Exception:
            holder.pop("camera_warmup_cache", None)
            raise
        if not isinstance(live, Mapping) or live.get("ok") is not True:
            holder.pop("camera_warmup_cache", None)
            holder["last_live_response"] = (
                copy.deepcopy(dict(live)) if isinstance(live, Mapping) else None
            )
            code = live.get("code") if isinstance(live, Mapping) else "PHYSICAL_CONSOLE_LIVE"
            raise ContractError(code if isinstance(code, str) else "PHYSICAL_CONSOLE_LIVE")
        data = live.get("data")
        if not isinstance(data, Mapping):
            raise ContractError("PHYSICAL_CONSOLE_LIVE_RESULT")
        reposition_result = data.get("object_reposition")
        if (
            reposition_binding is not None
            and reposition_binding["start_state"] == "ON_SURFACE"
        ):
            reposition_result = _validate_successful_object_reposition_result(
                reposition_result, reposition_binding,
                post_scene_digest=data.get("postcommit_scene_state_digest"),
                code="PHYSICAL_CONSOLE_REPOSITION_RESULT",
            )
            terminal_object_pose = copy.deepcopy(
                reposition_binding["target_pose"],
            )
        else:
            if reposition_result is not None:
                raise ContractError("PHYSICAL_CONSOLE_REPOSITION_RESULT")
            terminal_object_pose = {
                key: release_job[key]
                for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
            }
        validator = data.get("technical_validator")
        technical_digest = (
            validator.get("result_digest")
            if isinstance(validator, Mapping)
            and isinstance(validator.get("result_digest"), str)
            and DIGEST.fullmatch(validator["result_digest"])
            else canonical_digest(validator)
        )
        admission = run_job.write_candidate_admission(
            active_payload, active_resolved, validator,
            operational_source="HIL_PROXY",
        )
        candidate_path = (
            Path(active_payload["run_root"]) / intent["run_id"]
            / "candidate_admission.json"
        ).resolve(strict=True)
        ledger_reference = run_job.bind_candidate_episode_state(
            data.get("episode_ledger"), candidate_path,
        )
        technical = {
            "schema_version": "data_factory.seed_technical_result.v1",
            "intent_digest": intent["intent_digest"], "run_id": intent["run_id"],
            "manifest_digest": intent["manifest_digest"],
            "slot_id": intent["slot"]["slot_id"], "status": "PASS",
            "technical_result_digest": technical_digest,
            "post_scene_digest": data["postcommit_scene_state_digest"],
            "observed_at": clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        technical["evidence_digest"] = canonical_digest(technical)
        return {
            "result": {
                "technical_evidence": technical,
                "human_semantic": data.get("human_semantic_outcome", "NOT_MEASURED"),
                "terminal_object_pose": terminal_object_pose,
                **(
                    {"object_reposition": reposition_result}
                    if reposition_result is not None else {}
                ),
                "episode_ledger": copy.deepcopy(ledger_reference),
                "candidate_review_offer": {
                    "candidate_path": str(candidate_path),
                    "run_id": intent["run_id"],
                    "expected_file_digest": canonical_digest(admission),
                    "expected_review_context_digest": admission["review_context_digest"],
                    "checklist_id": admission["checklist_id"],
                    "ledger_reference": copy.deepcopy(ledger_reference),
                },
            },
            "technical_evidence": technical,
        }

    workspace_alias = (
        "place1"
        if resolved["normalized_job"]["place_id"] == "PLACE_A"
        else resolved["normalized_job"]["place_id"]
    )

    def operator_factory(episode_call) -> CampaignOperator:
        operator = CampaignOperator(
            session_id=session_id, lifecycle_owner=operator_label,
            operator_label=operator_label,
            workspace={
                "workspace_id": f"{workspace_alias}-{data_disposition.lower()}",
                "identity": (
                    f"{resolved['normalized_job']['place_id']}@"
                    f"{resolved['normalized_job']['cell_calibration_id']}"
                ),
            },
            hypothesis=hypothesis, draft=draft,
            effect_scope="PHYSICAL", lifecycle_action="LIVE_COLLECT",
            data_disposition=data_disposition,
            subsystems={
                "robot": {"readiness": "READY", "capability": "ATTACH", "reason": "PASSIVE_GATE_AT_RUN"},
                "gripper": {
                    "readiness": (
                        "READY" if initial_gripper["state"] == "ATTACHED"
                        else "NOT_AVAILABLE"
                    ),
                    "capability": "ATTACH",
                    "reason": initial_gripper["state"],
                },
                "camera": {"readiness": "READY", "capability": "CONNECTED_ASSIGNED", "reason": "STABLE_LOCAL_BINDING"},
            },
            expires_at=(clock() + _campaign_authorization_ttl(requested_count))
            .astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            initial_scene_digest=initial_scene_digest,
            scene_evidence_call=scene_evidence,
            side_effect_counter_call=lambda: copy.deepcopy(counters),
            fake_lifecycle_factory=lambda: (_ for _ in ()).throw(
                ContractError("PHYSICAL_CONSOLE_FAKE_FACTORY"),
            ),
            physical_activation_gate=activate,
            physical_lifecycle_factory=fresh_one_job,
            physical_live_call=episode_call,
            physical_root_binding_call=roots_for,
            physical_start_binding_call=start_binding,
            repository_root=repository, clock=clock,
        )
        holder["operator"] = operator
        return operator

    fixed_lane = {
        "workspace": {
            "display_name": (
                f"{workspace_alias} · {resolved['normalized_job']['place_id']} · "
                f"{data_disposition}"
            ),
            "place_id": resolved["normalized_job"]["place_id"],
            "revision": resolved["normalized_job"]["cell_calibration_id"],
            "bounds": (
                f"{len(resolved_jobs)} validated candidate poses · "
                f"{requested_count} episodes"
            ),
        },
        "object_id": resolved["normalized_job"]["object_profile_id"],
        "grasp_id": resolved["normalized_job"]["grasp_profile_id"],
        "task": {
            "id": resolved["normalized_job"]["task"],
            "capability": "PHYSICAL_EXECUTABLE",
        },
        "motion": {
            "id": hypothesis["fixed_contract"]["motion_recipe"],
            "capability": "PHYSICAL_EXECUTABLE",
        },
        "start_pose_id": hypothesis["robot_start_poses"][0]["robot_start_pose_id"],
        "camera_role": (
            f"{'+'.join(profile['camera_roles'])} · CONNECTED_ASSIGNED · "
            f"{data_disposition}"
        ),
        "profile_id": profile["collection_profile_id"],
    }
    full_open_m = float(motion_qualification["gripper_positions_m"]["open"])
    grasp_profile = resolved["grasp_profile"]
    close_profile = grasp_profile["gripper_close"]
    if retune is None:
        open_profile = grasp_profile["gripper_open"]
        tuning_id = grasp_profile["grasp_profile_id"]
        tuning_digest = canonical_digest(grasp_profile)
        tuning_status = "QUALIFIED_PROFILE"
        command_position_m = close_profile["command_position_m"]
        tuning_feedback = close_profile["acceptable_feedback_m"]
        gripper_settings = {
            "velocity_percent": close_profile["velocity_percent"],
            "force_percent": close_profile["force_percent"],
            "open_velocity_percent": open_profile["velocity_percent"],
            "open_force_percent": open_profile["force_percent"],
        }
    else:
        retune, gripper_settings = _validate_test_only_gripper_retune(
            retune, grasp=grasp_profile, motion=motion_qualification,
        )
        tuning_id = retune["retune_id"]
        tuning_digest = retune["retune_digest"]
        tuning_status = retune["status"]
        command_position_m = retune["command_position_m"]
        tuning_feedback = retune["acceptable_feedback_m"]
    base_velocity_percent = int(
        close_profile["velocity_percent"]
    )
    base_force_percent = int(
        close_profile["force_percent"]
    )
    gripper_tuning = {
        "retune_id": tuning_id,
        "retune_digest": tuning_digest,
        "status": tuning_status,
        "object_profile_id": grasp_profile["object_profile_id"],
        "grasp_profile_id": grasp_profile["grasp_profile_id"],
        "command_position_m": command_position_m,
        "command_percent": round(
            100 * float(command_position_m) / full_open_m, 2,
        ),
        "acceptable_feedback_m": copy.deepcopy(tuning_feedback),
        "acceptable_feedback_percent": {
            key: round(100 * float(value) / full_open_m, 2)
            for key, value in tuning_feedback.items()
        },
        **gripper_settings,
        "base_velocity_percent": base_velocity_percent,
        "base_force_percent": base_force_percent,
        "data_disposition": data_disposition,
        "production_authority": data_disposition == "PRODUCTION",
        "training_authority": False,
    }

    def projection() -> dict[str, Any]:
        gripper = holder["gripper_projection"]
        conditions = [
            item["coverage_condition"] for item in hypothesis["base_conditions"]
        ]
        return {
            "setup": {
                "host_status": (
                    "READY" if gripper["state"] == "ATTACHED"
                    else "READY_WITH_EXCEPTION"
                    if gripper["state"] == "MAINTENANCE_APPROVAL_REQUIRED"
                    else "BLOCKED"
                ),
                "operator_label": operator_label,
                "subsystems": [
                    {"label": "robot", "status": "ATTACH_ON_RUN", "detail": "existing foreground ROS graph only"},
                    {
                        "label": "gripper", "status": gripper["state"],
                        "detail": (
                            f"{gripper['supported_action']} · maintenance calls "
                            f"{gripper['maintenance_call_count']}"
                        ),
                    },
                    {
                        "label": "camera", "status": "CONNECTED_ASSIGNED",
                        "detail": ", ".join(
                            f"{role}: {device}"
                            for role, device in role_device_ids.items()
                        ),
                    },
                    {
                        "label": "data", "status": data_disposition,
                        "detail": (
                            str(first_roots["dataset_root"])
                            if data_disposition == "PRODUCTION"
                            else "isolated test root"
                        ),
                    },
                ],
            },
            "fixed_lane": copy.deepcopy(fixed_lane),
            "gripper_tuning": copy.deepcopy(gripper_tuning),
            "draft": {
                "draft_id": draft["draft_id"], "revision": draft["revision"],
                "authoring_mode": "DIRECT_EDIT", "selector": draft["selector"],
                "selector_version": "campaign-selector-v1", "budget": requested_count,
                "selected_count": requested_count, "blocked_count": 0,
                "estimated_minutes": requested_count * 2,
                "split_summary": f"TRAIN {requested_count}",
                "repeat_summary": f"x{requested_count}",
                "coverage_summary": f"{requested_count}/{requested_count} selected",
                "cells": [{
                    "cell_id": f"pose-{canonical_digest(condition)[7:27]}",
                    "x_mm": condition["x_mm"], "y_mm": condition["y_mm"],
                    "yaw_deg": condition["yaw_deg"], "split": "TRAIN",
                    "repeat": 1, "coverage_count": 0,
                    "selection_state": "SELECTED",
                    "eligibility_status": "ELIGIBLE",
                    "reason_codes": [
                        "CURRENT_SCENE_ANCHOR"
                        if condition == hypothesis["base_conditions"][0]["coverage_condition"]
                        else "REGISTERED_WORKSPACE_CANDIDATE"
                    ],
                } for condition in conditions],
            },
            "capabilities": [{
                "label": (
                    f"{resolved['normalized_job']['task']} · "
                    f"{hypothesis['fixed_contract']['motion_recipe']} · "
                    f"{len(profile['camera_roles'])}-camera {data_disposition}"
                ),
                "status": "PHYSICAL_EXECUTABLE",
                "reason_codes": ["REGISTERED_WORKSPACE_FINITE_CAMPAIGN"],
            }],
            "workspace_wizard": {
                "capability": "NOT_AVAILABLE",
                "plane_reference": {
                    "id": resolved["normalized_job"]["cell_calibration_id"],
                    "digest": resolved["input_digests"]["cell_calibration"],
                    "table_normal_base": resolved["calibration"]["z"],
                },
                "source_measurement_mm": None, "final_measurement_mm": None,
                "captures": {"CENTER": False, "X_REF": False, "Y_CHECK": False},
            },
            "effect_counts": copy.deepcopy(counters),
        }

    paths_text = " · ".join(
        f"{name}={first_roots[name]}"
        for name in ("run_root", "dataset_root", "cell_root")
    )
    console = OperatorConsole(
        session_id=session_id, run_id=run_id, operator_label=operator_label,
        campaign_operator_factory=operator_factory, episode_call=episode,
        projection_call=projection, test_only_paths=paths_text,
        terminal_response_call=lambda: holder.get("last_live_response"),
        candidate_state_observe_call=run_job.read_candidate_episode_state,
        candidate_review_port=CandidateReviewPort(
            operator_label=operator_label,
            review_call=lambda path, **kwargs: run_job.review_candidate_admission(
                path, clock=clock, **kwargs,
            ),
        ),
        gripper_setup_request=setup_request,
        gripper_setup_resolution_call=(
            resolve_gripper_setup if setup_request is not None else None
        ),
        initial_block_code=initial_block_code,
        task_bindings=task_bindings,
        episode_instruction_bindings=episode_instruction_bindings,
        object_reposition_bindings=reposition_bindings,
        yaw_sample_bindings=(
            None if yaw_sample_bindings is None
            else yaw_sample_bindings[:requested_count]
        ),
        campaign_approval_once=True,
        prepare_campaign_call=prepare_campaign if physical_test_scene else None,
        run_id_factory=run_id_for,
        prepare_timeout_s=8.0, close_timeout_s=5.0, clock=clock,
    )
    holder["console"] = console
    context = {
        "session_id": session_id, "run_id": run_id,
        "effect_scope": "PHYSICAL", "data_disposition": data_disposition,
        "camera_binding_set": camera_binding_set,
        "camera_binding_digest": role_map_digest,
        "camera_bindings": copy.deepcopy(role_device_ids),
        "roots": first_roots,
        "requested_count": requested_count,
        "resolved_job_digest": resolved["resolved_job_digest"],
        "hypothesis_digest": hypothesis["hypothesis_digest"],
        "feature_contract": copy.deepcopy(
            hypothesis["fixed_contract"]["feature_contract"]
        ),
        "motion_qualification_digest": canonical_digest(motion_qualification),
        "base_motion_qualification_digest": canonical_digest(
            motion_qualification,
        ),
        "gripper_tuning": copy.deepcopy(gripper_tuning),
        "gripper_setup": copy.deepcopy(holder["gripper_projection"]),
        "production_writers_enabled": data_disposition == "PRODUCTION",
    }
    return console, context


def build_physical_operator_application(
    *, repository_root: str | Path, session_id: str, operator_label: str,
    environment_call: Callable[[], Mapping[str, Any]],
    prepare_environment_call: Callable[[], Mapping[str, Any]],
    selected_camera_device_id: str | None = None,
    discovery_call: Callable[[], Sequence[object]] = discover_uvc_device_ids,
    activation_call: Callable[[], bool | Mapping[str, Any]] | None = None,
    snapshot_call: Callable[[], Mapping[str, Any]] | None = None,
    gripper_readback_call: Callable[[], Mapping[str, Any]] | None = None,
    gripper_maintenance_call: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    home_recovery_call: Callable[[], Mapping[str, Any]] | None = None,
    home_recovery_prepare_call: Callable[[], Mapping[str, Any]] | None = None,
    start_transition_call: Callable[[Mapping[str, Any], Mapping[str, Any], Any], Mapping[str, Any]] | None = None,
    run_live_call: Callable[..., Mapping[str, Any]] = run_job.run_live,
    production_campaign_factory: Callable[
        [str, dict[str, Any], dict[str, Any]], OperatorConsole
    ] | None = None,
    production_dataset_root: str | Path | None = None,
    initial_data_mode: str = "TEST_COLLECTION",
    initial_motion_preset: str | None = None,
    initial_environment: Mapping[str, Any] | None = None,
    initial_catalog: Mapping[str, Any] | None = None,
    initial_camera_devices: Sequence[object] | None = None,
    collection_evidence_call: Callable[[], Mapping[str, Any]] | None = None,
    rollout_lifecycle_path: str | Path | None = None,
    job_path: str | Path = DEFAULT_JOB,
    gripper_retune_path: str | Path | None = DEFAULT_GRIPPER_RETUNE,
    camera_environment_call: Callable[
        [Mapping[str, Any] | None, Mapping[str, Mapping[str, str]]], Mapping[str, Any]
    ] | None = None,
    prepare_environment_owner_call: Callable[
        [], tuple[Callable[[], Mapping[str, Any]], Callable[[], None]]
    ] | None = None,
    clock=None,
) -> tuple[CollectionOperatorApplication, dict[str, Any]]:
    """Compose the reusable app without creating campaign roots or run state."""
    if production_campaign_factory is not None and not callable(
        production_campaign_factory
    ):
        raise ContractError("OPERATOR_APPLICATION_PRODUCTION_FACTORY")
    if (
        initial_data_mode not in {"TEST_COLLECTION", "GENERAL_COLLECTION"}
        or production_campaign_factory is not None
        and production_dataset_root is not None
        or initial_data_mode == "GENERAL_COLLECTION"
        and production_campaign_factory is None
        and production_dataset_root is None
    ):
        raise ContractError("OPERATOR_APPLICATION_PRODUCTION_FACTORY")
    if (
        home_recovery_prepare_call is not None
        and not callable(home_recovery_prepare_call)
    ):
        raise ContractError("OPERATOR_APPLICATION_RECOVERY")
    repository = Path(repository_root).resolve(strict=True)
    if initial_catalog is None:
        camera_devices = normalize_camera_devices(discovery_call())
        devices = [item["logical_id"] for item in camera_devices]
        catalog = load_operator_catalog(repository, device_ids=devices)
    else:
        catalog = copy.deepcopy(dict(initial_catalog))
        machine = catalog.get("machine")
        devices = machine.get("camera_device_ids") if isinstance(machine, Mapping) else None
        if (
            catalog.get("schema_version") != CATALOG_SCHEMA
            or catalog.get("catalog_digest") != canonical_digest({
                key: value for key, value in catalog.items()
                if key != "catalog_digest"
            })
            or not isinstance(devices, list)
            or any(not isinstance(item, str) or not item for item in devices)
            or devices != sorted(set(devices))
        ):
            raise ContractError("OPERATOR_CATALOG_SCHEMA")
        camera_devices = normalize_camera_devices(
            devices if initial_camera_devices is None else initial_camera_devices,
        )
        if {item["logical_id"] for item in camera_devices} != set(devices):
            raise ContractError("OPERATOR_CATALOG_SCHEMA")
    initial_job_path = _repository_path(repository, job_path)
    initial_job = load_json_strict(initial_job_path)
    initial_job_source = str(initial_job_path.relative_to(repository))
    tcp_manifest_path = _tcp_manifest_for_robot(
        repository, initial_job.get("robot_system_id"),
    )

    def scope_catalog_to_active_job(value: Mapping[str, Any]) -> dict[str, Any]:
        """Keep historical profiles readable without exposing them as this job."""
        scoped = copy.deepcopy(dict(value))
        combinations = [
            item for item in scoped["combinations"]
            if item.get("sources", {}).get("job") == initial_job_source
            and item.get("object_id") == initial_job.get("object_profile_id")
            and item.get("grasp_id") == initial_job.get("grasp_profile_id")
        ]
        if not combinations:
            raise ContractError("OPERATOR_APPLICATION_COMPATIBLE_COMBINATION")
        axis_fields = {
            "workspace": "workspace_id", "frame": "frame_id",
            "task": "task_id", "object": "object_id", "grasp": "grasp_id",
            "cell": "cell_id", "start_pose": "start_pose_id",
            "motion": "motion_id", "variant": "variant_id",
            "camera_profile": "camera_profile_id",
        }
        for axis, field in axis_fields.items():
            identifiers = {item[field] for item in combinations}
            scoped["axes"][axis] = [
                option for option in scoped["axes"][axis]
                if option["id"] in identifiers
            ]
        endpoints = {
            (item["workspace_id"], item["frame_id"], item["object_id"])
            for item in combinations
        }
        scoped["workspace_domains"] = [
            domain for domain in scoped["workspace_domains"]
            if (
                domain["workspace_id"], domain["frame_id"],
                domain["object_id"],
            ) in endpoints
        ]
        scoped["combinations"] = sorted(
            combinations, key=lambda item: item["combination_digest"],
        )
        scoped["catalog_digest"] = canonical_digest({
            key: item for key, item in scoped.items()
            if key != "catalog_digest"
        })
        return scoped

    catalog = scope_catalog_to_active_job(catalog)
    requested = None
    if selected_camera_device_id is not None:
        if devices.count(selected_camera_device_id) != 1:
            raise ContractError("PHYSICAL_CAMERA_BINDING_MISMATCH")
        requested = {device: "UNUSED" for device in devices}
        preferred_profile = _v2_camera_profiles(repository).get(
            initial_job.get("collection_profile_id"),
        )
        if preferred_profile is None or len(preferred_profile["camera_roles"]) != 1:
            raise ContractError("OPERATOR_APPLICATION_COMPATIBLE_COMBINATION")
        requested[selected_camera_device_id] = preferred_profile["camera_roles"][0].upper()
    camera_setup, camera_resolution = resolve_camera_setup(
        repository_root=repository, devices=camera_devices,
        preferred_profile_id=initial_job.get("collection_profile_id", ""),
        requested_bindings=requested,
        persist_compatible_revision=requested is None,
    )
    selected_camera_bindings: dict[str, str] = {}
    selected_camera_binding_digest = None
    selected_profile = None
    if camera_resolution is not None:
        selected_profile = camera_resolution["collection_profile"]
        selected_camera_bindings = {
            role: binding["stable_device_id"]
            for role, binding in camera_resolution["role_bindings"]["bindings"].items()
        }
        selected_camera_binding_digest = camera_binding_digest(
            selected_profile, selected_camera_bindings,
        )
        selected_camera_device_id = (
            next(iter(selected_camera_bindings.values()))
            if len(selected_camera_bindings) == 1 else None
        )
    available_binding_digests = sorted({
        candidate["camera_binding_digest"] for candidate in catalog["combinations"]
        if candidate["camera_bindings"]
    })
    mapping_ready = (
        camera_setup["status"] == "READY"
        and selected_camera_binding_digest in available_binding_digests
    )
    preferred_binding_digests = sorted({
        candidate["camera_binding_digest"] for candidate in catalog["combinations"]
        if candidate["camera_bindings"]
        and candidate["camera_profile_id"] == initial_job.get("collection_profile_id")
    })
    internal_binding_digest = (
        selected_camera_binding_digest if mapping_ready
        else preferred_binding_digests[0] if preferred_binding_digests
        else available_binding_digests[0] if available_binding_digests
        else None
    )
    camera_state = {
        "bindings": selected_camera_bindings,
        "binding_digest": selected_camera_binding_digest,
        "profile": selected_profile,
        "device_id": selected_camera_device_id,
        "ready": mapping_ready,
        "binding_set": (
            None if camera_resolution is None
            else copy.deepcopy(camera_resolution["role_bindings"])
        ),
    }

    def bind_selected_camera(
        value: Mapping[str, Any], *, binding_digest: str | None,
    ) -> dict[str, Any]:
        bound = copy.deepcopy(dict(value))
        for candidate in bound["combinations"]:
            if candidate["camera_binding_digest"] != binding_digest:
                reason = (
                    "CAMERA_REBIND_REQUIRED" if binding_digest is not None
                    else camera_setup["reason"] or "CAMERA_ROLE_BINDING_REQUIRED"
                )
                for mode in ("TEST_COLLECTION", "GENERAL_COLLECTION"):
                    candidate["execution"][mode] = {
                        "executable": False, "reason": reason,
                    }
            if (
                production_campaign_factory is None
                and production_dataset_root is None
            ):
                candidate["execution"]["GENERAL_COLLECTION"] = {
                    "executable": False,
                    "reason": "GENERAL_CALLER_NOT_CONFIGURED",
                }
            candidate["combination_digest"] = canonical_digest({
                key: item for key, item in candidate.items()
                if key != "combination_digest"
            })
        bound["combinations"].sort(key=lambda item: item["combination_digest"])
        bound["catalog_digest"] = canonical_digest({
            key: item for key, item in bound.items() if key != "catalog_digest"
        })
        return bound

    catalog = bind_selected_camera(
        catalog, binding_digest=(
            camera_state["binding_digest"] if camera_state["ready"] else None
        ),
    )
    compatible = [
        combination for combination in catalog["combinations"]
        if (
            internal_binding_digest is None
            or combination["camera_binding_digest"] == internal_binding_digest
        )
    ]
    initial_cells = {
        option["id"]
        for option in catalog["axes"]["cell"]
        if all(
            option.get("metadata", {}).get(field) == initial_job.get(field)
            for field in ("place_id", "yaw_deg", "x_mm", "y_mm")
        )
    }
    initial = [
        item for item in compatible
        if item["cell_id"] in initial_cells
        and item["sources"]["job"] == initial_job_source
        and item["task_id"] == initial_job.get("task")
        and item["variant_id"] == "DIRECT"
        and (
            internal_binding_digest is not None
            or item["camera_profile_id"] == initial_job.get("collection_profile_id")
        )
        and (
            not mapping_ready
            or item["execution"][initial_data_mode]["executable"] is True
        )
    ]
    if len(initial) != 1:
        raise ContractError("OPERATOR_APPLICATION_COMPATIBLE_COMBINATION")
    combination = initial[0]
    selection = {
        "schema_version": SELECTION_SCHEMA_V2,
        "combination_digest": combination["combination_digest"],
        "data_mode": initial_data_mode,
        **{
            field: combination[field]
            for field in (
                "workspace_id", "frame_id", "task_id", "object_id", "grasp_id",
                "cell_id", "start_pose_id", "motion_id", "variant_id",
                "camera_profile_id", "camera_device_id",
            )
        },
        "camera_bindings": copy.deepcopy(combination["camera_bindings"]),
        "camera_binding_digest": combination["camera_binding_digest"],
        "policy_id": "DETERMINISTIC_SPREAD",
    }
    application_holder: dict[str, CollectionOperatorApplication] = {}

    def active_catalog() -> Mapping[str, Any]:
        application = application_holder.get("application")
        return catalog if application is None else application.catalog

    def active_selection() -> Mapping[str, Any]:
        application = application_holder.get("application")
        return selection if application is None else application.selection

    def active_combination() -> Mapping[str, Any]:
        selected = active_selection()
        matches = [
            item for item in active_catalog()["combinations"]
            if item["combination_digest"] == selected["combination_digest"]
        ]
        if len(matches) != 1:
            raise ContractError("OPERATOR_APPLICATION_COMPATIBLE_COMBINATION")
        return matches[0]

    def object_position_call(selected):
        chosen = next(item for item in active_catalog()["combinations"]
                      if item["combination_digest"] == selected["combination_digest"])
        profile = load_json_strict(_repository_path(repository, chosen["sources"]["object"]))
        position = SceneStateStore(
            repository / "outputs/data_factory/cells", initial_job["robot_system_id"],
        ).object_position(
            object_profile_id=selected["object_id"], dimensions_mm=profile["dimensions_mm"],
        )
        # Legacy isolated tests have no physical source to inherit. Once one
        # exists, disposition cannot hide its current or blocked physical facts.
        return None if selected["data_mode"] == "TEST_COLLECTION" and position["status"] == "MISSING" else position

    def object_position_declare_call(selected, pose, expected):
        current = object_position_call(selected)
        if current is None:
            return None
        if current["reason"] == "SCENE_OBJECT_AMBIGUOUS":
            raise ContractError(current["reason"])
        if current["binding_digest"] != expected["binding_digest"]:
            raise ContractError("OBJECT_POSITION_CHANGED")
        robot_id = initial_job["robot_system_id"]
        instance_id = current["object_instance_id"] or "production-object-" + canonical_digest({
            "robot_system_id": robot_id, "object_profile_id": selected["object_id"],
        }).removeprefix("sha256:")[:20]
        SceneStateStore(repository / "outputs/data_factory/cells", robot_id).update_object(
            instance_id=instance_id, object_profile_id=selected["object_id"],
            state="ON_SURFACE", source="HUMAN", updated_by=operator_label,
            pose=pose, expected_revision=current["scene_revision"],
            expected_cell_digest=current["cell_state_digest"],
        )
        return object_position_call(selected)

    def start_pose_domain(
        selected_ids: Sequence[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        source = active_combination()["sources"]
        motion = load_json_strict(_repository_path(repository, source["motion"]))
        home = load_json_strict(_repository_path(repository, source["start_pose"]))
        home_digest = canonical_digest(home)
        home_pose = _home_start_pose(
            motion, home, motion["robot_system_id"],
        )
        qualifications = {home_pose["robot_start_pose_id"]: home_pose}
        profiles = [{
            "start_pose_id": home_pose["robot_start_pose_id"],
            "display_name": "HOME",
            "status": "AVAILABLE",
        }]
        registry = repository / DEFAULT_START_POSES
        for profile in list_start_pose_profiles(registry):
            identifier = profile["start_pose_id"]
            if identifier in qualifications:
                raise ContractError("PHYSICAL_CONSOLE_START_POSE")
            compatible = (
                profile["robot_system_id"] == motion["robot_system_id"]
                and profile["recovery_home_digest"] == home_digest
            )
            available = (
                compatible
                and profile["qualification_status"] == "QUALIFIED"
                and profile["safety_status"] == "SAFE_FOR_MOTION"
            )
            status = (
                "AVAILABLE" if available else "CANDIDATE"
                if compatible and profile["qualification_status"] == "CANDIDATE"
                else "QUALIFICATION_REQUIRED"
            )
            projected = {
                "start_pose_id": identifier,
                "display_name": profile["display_name"],
                "status": status,
            }
            if not available:
                projected["reason"] = (
                    "START_POSE_REQUIRES_QUALIFICATION"
                    if compatible else "START_POSE_BINDING_MISMATCH"
                )
            else:
                qualifications[identifier] = project_robot_start_pose_qualification(
                    profile,
                )
            profiles.append(projected)
        profiles.sort(key=lambda value: (
            value["start_pose_id"] != home_pose["robot_start_pose_id"],
            value["start_pose_id"],
        ))
        chosen = (
            [home_pose["robot_start_pose_id"]]
            if selected_ids is None else list(selected_ids)
        )
        if (
            not chosen or len(chosen) != len(set(chosen))
            or any(identifier not in qualifications for identifier in chosen)
        ):
            raise ContractError("PHYSICAL_CONSOLE_START_POSE")
        return {
            "profiles": profiles,
            "selected_start_pose_ids": chosen,
        }, qualifications

    start_pose_setup, _initial_start_qualifications = start_pose_domain()

    def _camera_bindings_update(
        assignments: Mapping[str, str] | None,
    ) -> Mapping[str, Any]:
        nonlocal camera_setup, devices, camera_devices
        camera_devices = normalize_camera_devices(discovery_call())
        devices = [item["logical_id"] for item in camera_devices]
        next_setup, resolution = resolve_camera_setup(
            repository_root=repository, devices=camera_devices,
            preferred_profile_id=initial_job.get("collection_profile_id", ""),
            requested_bindings=assignments,
        )
        current_catalog = active_catalog()
        current_selection = copy.deepcopy(dict(active_selection()))
        if resolution is None:
            camera_setup = next_setup
            camera_state.update(
                bindings={}, binding_digest=None, profile=None,
                device_id=None, ready=False, binding_set=None,
            )
            environment = (
                camera_environment_call(None, {})
                if camera_environment_call is not None else environment_call()
            )
            return {
                "camera_setup": next_setup, "catalog": current_catalog,
                "selection": current_selection, "environment": environment,
            }
        profile = resolution["collection_profile"]
        role_map = {
            role: binding["stable_device_id"]
            for role, binding in resolution["role_bindings"]["bindings"].items()
        }
        binding_digest = camera_binding_digest(profile, role_map)
        raw_catalog = scope_catalog_to_active_job(
            load_operator_catalog(repository, device_ids=devices),
        )
        bound_catalog = bind_selected_camera(
            raw_catalog, binding_digest=binding_digest,
        )
        current_source = active_combination()["sources"]["job"]
        candidates = [
            item for item in bound_catalog["combinations"]
            if item["camera_profile_id"] == profile["collection_profile_id"]
            and item["camera_binding_digest"] == binding_digest
            and item["sources"]["job"] == current_source
            and all(
                item[field] == current_selection[field]
                for field in (
                    "workspace_id", "frame_id", "task_id", "object_id",
                    "grasp_id", "cell_id", "start_pose_id", "motion_id",
                    "variant_id",
                )
            )
        ]
        if len(candidates) != 1:
            next_setup = {
                **next_setup, "status": "READY",
                "reason": "COLLECTION_CAMERA_CALLER_NOT_AVAILABLE",
            }
            camera_setup = next_setup
            camera_state.update(
                bindings=role_map, binding_digest=binding_digest, profile=profile,
                device_id=(next(iter(role_map.values())) if len(role_map) == 1 else None),
                ready=True,
                binding_set=copy.deepcopy(resolution["role_bindings"]),
            )
            write_camera_role_bindings(
                resolution["role_bindings"], repository_root=repository,
            )
            camera_devices = {
                role: {
                    "kind": binding["device_kind"],
                    "stable_id": binding["stable_device_id"],
                    "capture_endpoint": binding["capture_endpoint"],
                }
                for role, binding in resolution["role_bindings"]["bindings"].items()
            }
            environment = (
                camera_environment_call(profile, camera_devices)
                if camera_environment_call is not None else environment_call()
            )
            return {
                "camera_setup": next_setup, "catalog": current_catalog,
                "selection": current_selection, "environment": environment,
            }
        candidate = candidates[0]
        next_selection = {
            **current_selection,
            "combination_digest": candidate["combination_digest"],
            "camera_profile_id": candidate["camera_profile_id"],
            "camera_device_id": candidate["camera_device_id"],
            "camera_bindings": copy.deepcopy(candidate["camera_bindings"]),
            "camera_binding_digest": candidate["camera_binding_digest"],
        }
        validate_operator_selection(bound_catalog, next_selection)
        write_camera_role_bindings(
            resolution["role_bindings"], repository_root=repository,
        )
        camera_setup = next_setup
        camera_state.update(
            bindings=role_map, binding_digest=binding_digest, profile=profile,
            device_id=(next(iter(role_map.values())) if len(role_map) == 1 else None),
            ready=True, binding_set=copy.deepcopy(resolution["role_bindings"]),
        )
        camera_devices = {
            role: {
                "kind": binding["device_kind"],
                "stable_id": binding["stable_device_id"],
                "capture_endpoint": binding["capture_endpoint"],
            }
            for role, binding in resolution["role_bindings"]["bindings"].items()
        }
        environment = (
            camera_environment_call(profile, camera_devices)
            if camera_environment_call is not None else environment_call()
        )
        return {
            "camera_setup": next_setup, "catalog": bound_catalog,
            "selection": next_selection, "environment": environment,
        }

    def camera_bindings_call(
        assignments: Mapping[str, str],
    ) -> Mapping[str, Any]:
        return _camera_bindings_update(assignments)

    def camera_refresh_call() -> Mapping[str, Any]:
        return _camera_bindings_update(None)

    def workspace_manager_factory(display_name: str) -> WorkspaceManager:
        return WorkspaceManager(
            session_id=f"{session_id}-workspace",
            candidate_root=repository / "outputs/data_factory/workspace_registration",
            config_root=repository / "config/data_factory",
            display_name=display_name,
        )

    def workspace_snapshot_call() -> Mapping[str, Any]:
        if snapshot_call is not None:
            return snapshot_call()
        return capture_home_snapshot(
            tcp_candidate_manifest=tcp_manifest_path,
        )

    def start_pose_capture_call(display_name: str) -> Mapping[str, Any]:
        source = active_combination()["sources"]
        motion = load_json_strict(_repository_path(repository, source["motion"]))
        home = load_json_strict(_repository_path(repository, source["start_pose"]))
        captured = workspace_snapshot_call()
        joints = captured.get("joint_positions_rad") if isinstance(captured, Mapping) else None
        ages = (
            captured.get("joint_state_age_s"), captured.get("ros_sample_age_s"),
        ) if isinstance(captured, Mapping) else (None, None)
        if (
            not isinstance(captured, Mapping)
            or captured.get("schema_version") != "data_factory.pose_snapshot.v1"
            or not isinstance(joints, Mapping)
            or set(joints) != {"j1", "j2", "j3", "j4", "j5", "j6"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in joints.values()
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= float(motion["max_joint_state_age_s"])
                for value in ages
            )
        ):
            raise ContractError("START_POSE_SNAPSHOT")
        now = clock().astimezone(timezone.utc)
        normalized_snapshot = {
            "schema_version": "data_factory.start_pose_joint_snapshot.v1",
            "source": "READ_ONLY_JOINT_STATE",
            "robot_system_id": motion["robot_system_id"],
            "joint_order": ["j1", "j2", "j3", "j4", "j5", "j6"],
            "joint_positions_rad": {
                joint: float(joints[joint])
                for joint in ("j1", "j2", "j3", "j4", "j5", "j6")
            },
            "captured_at": now.isoformat().replace("+00:00", "Z"),
        }
        slug = re.sub(r"[^a-z0-9]+", "-", display_name.lower()).strip("-")[:32]
        token = canonical_digest({
            "display_name": display_name,
            "snapshot": normalized_snapshot,
        }).removeprefix("sha256:")[:12]
        profile = compile_start_pose_profile(
            start_pose_id=f"start-{slug or 'pose'}-{token}",
            display_name=display_name,
            robot_system_id=motion["robot_system_id"],
            snapshot=normalized_snapshot,
            tolerance_rad={
                joint: float(motion["goal_tolerances"]["joint_rad"])
                for joint in ("j1", "j2", "j3", "j4", "j5", "j6")
            },
            recovery_home_digest=canonical_digest(home),
            qualification_status="CANDIDATE", safety_status="UNASSESSED",
            max_snapshot_age_s=0.5, now=now,
        )
        save_start_pose_profile(repository / DEFAULT_START_POSES, profile, now=now)
        application = application_holder.get("application")
        selected = (
            application.draft["selected_start_pose_ids"]
            if application is not None else start_pose_setup["selected_start_pose_ids"]
        )
        setup, _qualifications = start_pose_domain(selected)
        return setup

    def workspace_preview_call(
        manager: WorkspaceManager, measurements: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        sources = active_combination()["sources"]
        cell = load_json_strict(_repository_path(repository, sources["cell"]))
        measured = validate_print_measurements(
            source_scale_bar_mm=measurements["source_scale_bar_mm"],
            final_scale_bar_mm=measurements["final_scale_bar_mm"],
        )
        yaw0_sheet = select_yaw0_print_profile(
            repository,
            place_id=cell["place_id"],
            source_scale_bar_mm=measured["source_scale_bar_measured_mm"],
        )
        return manager.preview_captured(
            plane_reference=qualified_table_plane_reference(cell),
            print_measurements=measured,
            operator_or_agent_id=operator_label,
            yaw0_sheet=yaw0_sheet,
            tcp_candidate_manifest=tcp_manifest_path,
            tolerance_mm=1.0,
        )

    def catalog_reload_call() -> Mapping[str, Any]:
        return bind_selected_camera(
            scope_catalog_to_active_job(
                load_operator_catalog(repository, device_ids=devices),
            ),
            binding_digest=(
                camera_state["binding_digest"] if camera_state["ready"] else None
            ),
        )

    def stored_collection_evidence():
        from tools.data_factory.collection_recommendation_io import discover_stored_collection

        run_root = repository / "outputs/data_factory/runs"
        try:
            discovery = discover_stored_collection(run_root)
        except (ContractError, OSError) as exc:
            return {"run_directories": [], "scene_state_path": scene_state_path,
                    "discovery": {"availability": "UNAVAILABLE", "reason_codes": [
                        exc.code if isinstance(exc, ContractError) else "COLLECTION_RECOMMENDATION_SOURCE_IO"]}}
        return {"run_directories": discovery["run_directories"], "scene_state_path": scene_state_path,
                **({"rollout_lifecycle_path": str(rollout_lifecycle_path)}
                   if rollout_lifecycle_path is not None else {}),
                "discovery": {"availability": discovery["availability"],
                    "discovery_digest": discovery["discovery_digest"], "run_count": len(discovery["run_directories"]),
                    "excluded": [{"run_id": Path(item["run_directory"]).name, "reason_code": item["reason_code"]}
                                 for item in discovery["excluded"]]}}

    def preset_qualification(endpoint, preset, *, trial=False):
        if preset is None or trial:
            return endpoint["sources"]["motion"]
        qualified = preset["qualifications"].get(endpoint["motion_id"])
        if qualified is None:
            raise ContractError("MOTION_PRESET_QUALIFICATION_REQUIRED")
        path = _repository_path(repository, qualified["source"])
        if canonical_digest(load_json_strict(path)) != qualified["digest"]:
            raise ContractError("MOTION_PRESET_BINDING")
        return str(path)

    def selected_home_recovery_call() -> Mapping[str, Any]:
        if home_recovery_call is not None:
            return home_recovery_call()
        from tools.data_factory.motion.home_recovery import (
            recover_home_live,
            validate_home_recovery_qualification,
        )
        application = application_holder.get("application")
        binding = application.draft.get("motion_preset") if application is not None else None
        preset = selected_motion_preset(active_catalog(), binding)
        trial = (application is not None and application.selection["data_mode"] == "TEST_COLLECTION"
                 and preset is not None and any(endpoint["motion_id"] not in preset["qualifications"]
                                               for endpoint in application._workspace_cycle()))
        policy = {}
        if binding is not None:
            selected_policy = load_motion_preset(repository / "config/data_factory", binding)
            if not trial:
                policy["motion_preset"] = selected_policy
        motion = load_json_strict(
            _repository_path(repository, preset_qualification(active_combination(), preset, trial=trial)),
        )
        validate_home_recovery_qualification(motion, **policy)
        if home_recovery_prepare_call is not None:
            home_recovery_prepare_call()
        recovery = recover_home_live(
            motion_qualification=motion,
            **policy,
        )
        application = application_holder.get("application")
        if (
            application is not None
            and application.selection.get("data_mode") == "GENERAL_COLLECTION"
            and recovery.get("schema_version") == "data_factory.home_recovery.v1"
            and recovery.get("status") in {"HOME", "ALREADY_HOME"}
            and recovery.get("gripper_open") is True
            and recovery.get("arm_goal_count") in {0, 1}
        ):
            cell_store = CellStateStore(
                repository / "outputs/data_factory/cells",
                motion["robot_system_id"],
            )
            cell = cell_store.read()
            if (
                cell["cell_ready"] is False
                and cell["reason_code"] == "ROS_EXEC_FAILED"
            ):
                cell_store.acknowledge_ready(
                    operator_label,
                    expected_run_id=cell["run_id"],
                    expected_plan_digest=cell["plan_digest"],
                )
        return recovery

    def physical_pose_plan(
        selected: Mapping[str, Any], draft: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        current_catalog = active_catalog()
        anchor = validate_operator_pose(
            current_catalog, selected, draft.get("current_object_pose"),
        )
        count = draft["requested_count"]
        spatial_node_count = count + int(selected["task_id"] == "pick_place")
        route = (
            resolve_workspace_cycle_selections(
                current_catalog, selected, count,
            )
            if selected["task_id"] == "pick_place"
            else [copy.deepcopy(selected) for _index in range(spatial_node_count)]
        )
        if draft.get("authoring_mode") == "ASSISTED":
            spatial_seed = _domain_seed(draft["normalized_seed"], "spatial")
            design_kwargs = (
                {"state_space_design_profile": draft["state_space_design_profile"]}
                if draft.get("state_space_design_profile") is not None else {}
            )
            poses = (
                project_workspace_cycle_poses(
                    current_catalog, selected, anchor, count,
                    repeat=draft["repeat"],
                    normalized_seed=spatial_seed,
                    yaw_sampling_seed=_domain_seed(
                        draft["normalized_seed"], "yaw",
                    ),
                    **design_kwargs,
                )
                if selected["task_id"] == "pick_place"
                else project_assisted_poses(
                    current_catalog, selected, anchor, spatial_node_count,
                    repeat=draft["repeat"],
                    normalized_seed=spatial_seed,
                    yaw_sampling_seed=_domain_seed(
                        draft["normalized_seed"], "yaw",
                    ),
                    **design_kwargs,
                )
            )
            start_ids = project_balanced_start_pose_ids(
                draft["selected_start_pose_ids"], count,
                normalized_seed=_domain_seed(
                    draft["normalized_seed"], "start_pose",
                ),
            )
        else:
            pairs = copy.deepcopy(draft.get("direct_pairs") or [])
            if len(pairs) != spatial_node_count:
                raise ContractError("PHYSICAL_CONSOLE_DIRECT_SEQUENCE")
            poses = [
                validate_operator_pose(current_catalog, endpoint, {
                    key: pair[key]
                    for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
                })
                for pair, endpoint in zip(pairs, route)
            ]
            if selected["task_id"] == "pick_place":
                validate_yaw_preserving_transitions(
                    current_catalog, route, poses,
                )
            start_ids = [pair["start_pose_id"] for pair in pairs[:count]]
        _setup, qualifications = start_pose_domain(
            draft["selected_start_pose_ids"],
        )
        yaw_bindings = (
            project_yaw_sample_bindings(
                current_catalog, route, poses,
                _domain_seed(draft["normalized_seed"], "yaw"),
                repeat=draft["repeat"],
                **design_kwargs,
            )
            if draft.get("authoring_mode") == "ASSISTED" else
            [None for _pose in poses]
        )
        advice = draft.get("acquisition_recommendation")
        advice_selection = None if advice is None else copy.deepcopy(advice["selection"])
        if advice_selection is not None and advice["sampling"]["authoring_mode"] == "DIRECT_EDIT":
            advice_selection["policy_id"] = "DIRECT_SELECTION"
        if advice is not None and (
            advice_selection != selected or advice["object_poses"] != poses
            or [item["yaw_sample_binding"] for item in advice["conditions"]] != yaw_bindings[:count]
            or advice["sampling"] != {"authoring_mode": draft["authoring_mode"], **{
                key: draft[key] for key in ("requested_count", "normalized_seed", "repeat")}}
        ):
            raise ContractError("COLLECTION_ADVICE_COMPILER_MISMATCH")
        return {
            "direct_pose_sequence": poses,
            "direct_yaw_sample_bindings": yaw_bindings,
            "direct_start_pose_ids": start_ids,
            "selected_start_pose_qualifications": [
                qualifications[identifier]
                for identifier in sorted(set(start_ids))
            ],
        }, poses[0]

    def verified_camera_environment() -> dict[str, Any]:
        environment = environment_call()
        component = environment.get("components", {}).get("camera")
        binding_set = camera_state.get("binding_set")
        if (
            not isinstance(component, Mapping)
            or component.get("state") != "READY"
            or not isinstance(binding_set, Mapping)
        ):
            raise ContractError("PHYSICAL_CAMERA_TOPIC")
        checked = validate_camera_role_bindings(binding_set)
        evidence = {
            "schema_version": "data_factory.test_only_camera_transport_set.v1",
            "camera_binding_digest": camera_state["binding_digest"],
            "roles": {
                role: {
                    "device_kind": binding["device_kind"],
                    "stable_device_id": binding["stable_device_id"],
                    "capture_endpoint": binding["capture_endpoint"],
                    "status": "ENVIRONMENT_LIFECYCLE_VERIFIED",
                }
                for role, binding in checked["bindings"].items()
            },
            "status": "ENVIRONMENT_LIFECYCLE_VERIFIED",
        }
        evidence["binding_digest"] = canonical_digest(evidence)
        return evidence

    def campaign_factory(
        campaign_id: str, selected: dict[str, Any], draft: dict[str, Any],
    ) -> OperatorConsole:
        chosen = next((
            item for item in active_catalog()["combinations"]
            if item["combination_digest"] == selected.get("combination_digest")
        ), None)
        mode = selected.get("data_mode")
        execution = (
            chosen.get("execution", {}).get(mode)
            if isinstance(chosen, Mapping) else None
        )
        if (
            not isinstance(chosen, Mapping)
            or not isinstance(execution, Mapping)
            or execution.get("executable") is not True
            or mode not in {"TEST_COLLECTION", "GENERAL_COLLECTION"}
            or draft.get("draft_id") != f"{campaign_id}-draft"
            or type(draft.get("requested_count")) is not int
        ):
            raise ContractError("OPERATOR_APPLICATION_CAMPAIGN_FACTORY")
        if mode == "GENERAL_COLLECTION" and production_campaign_factory is not None:
            if draft.get("acquisition_recommendation") is not None:
                raise ContractError("COLLECTION_ADVICE_COMPILER_UNAVAILABLE")
            console = production_campaign_factory(
                campaign_id, copy.deepcopy(selected), copy.deepcopy(draft),
            )
            if (
                not isinstance(console, OperatorConsole)
                or console.campaign_operator.effect_scope != "PHYSICAL"
                or console.campaign_operator.data_disposition != "PRODUCTION"
            ):
                close = getattr(console, "close", None)
                if callable(close):
                    close()
                raise ContractError("OPERATOR_APPLICATION_PRODUCTION_FACTORY")
            return console
        if mode == "GENERAL_COLLECTION" and production_dataset_root is None:
            raise ContractError("OPERATOR_APPLICATION_PRODUCTION_FACTORY")
        source = chosen["sources"]
        pose_plan, campaign_initial_pose = physical_pose_plan(selected, draft)
        endpoint_selections = (
            resolve_workspace_cycle_selections(
                active_catalog(), selected, draft["requested_count"],
            )
            if selected["task_id"] == "pick_place" else [selected]
        )
        endpoint_combinations = []
        for endpoint in endpoint_selections:
            if any(
                item["workspace_id"] == endpoint["workspace_id"]
                for item in endpoint_combinations
            ):
                continue
            match = next((
                item for item in active_catalog()["combinations"]
                if item["combination_digest"]
                == endpoint["combination_digest"]
            ), None)
            if not isinstance(match, Mapping):
                raise ContractError("OPERATOR_APPLICATION_CAMPAIGN_FACTORY")
            endpoint_combinations.append(match)
        runtime_workspace_bindings = {}
        preset = selected_motion_preset(active_catalog(), draft.get("motion_preset"))
        trial = (mode == "TEST_COLLECTION" and preset is not None
                 and any(endpoint["motion_id"] not in preset["qualifications"] for endpoint in endpoint_combinations))
        for endpoint in endpoint_combinations:
            domains = [
                domain for domain in active_catalog()["workspace_domains"]
                if domain["workspace_id"] == endpoint["workspace_id"]
                and domain["frame_id"] == endpoint["frame_id"]
                and domain["object_id"] == endpoint["object_id"]
            ]
            if len(domains) != 1:
                raise ContractError("OPERATOR_APPLICATION_CAMPAIGN_FACTORY")
            region = domains[0]["coverage_region"]
            runtime_workspace_bindings[endpoint["workspace_id"]] = {
                "frame_id": endpoint["frame_id"],
                "selected_sheet": endpoint["sources"]["selected_sheet"],
                "yaw0_sheet": endpoint["sources"]["yaw0_sheet"],
                "motion_qualification": preset_qualification(endpoint, preset, trial=trial),
                "region_binding": {
                    key: copy.deepcopy(region[key]) for key in (
                        "layout_id", "layout_digest", "region_id",
                        "physical_binding_status",
                    )
                },
            }
        disposition = (
            "PRODUCTION" if mode == "GENERAL_COLLECTION" else "TEST_ONLY"
        )
        console, _context = build_physical_operator_console(
            repository_root=repository,
            session_id=campaign_id,
            run_id=f"{campaign_id}-run-1",
            operator_label=operator_label,
            job_path=source["job"],
            yaw0_sheet=source["yaw0_sheet"],
            motion_qualification_path=preset_qualification(chosen, preset, trial=trial),
            motion_preset=draft.get("motion_preset"),
            motion_preset_trial=trial,
            home_candidate_path=source["start_pose"],
            collection_profile_path=source["camera_profile"],
            tcp_candidate_manifest=tcp_manifest_path,
            selected_camera_device_id=selected["camera_device_id"],
            selected_camera_bindings=selected["camera_bindings"],
            selected_camera_binding_digest=selected["camera_binding_digest"],
            selected_camera_binding_set=camera_state["binding_set"],
            discovery_call=discovery_call,
            activation_call=(
                activation_call
                if activation_call is not None or camera_environment_call is None
                else verified_camera_environment
            ),
            snapshot_call=snapshot_call,
            gripper_readback_call=gripper_readback_call,
            gripper_maintenance_call=gripper_maintenance_call,
            gripper_retune_path=(
                None if disposition == "PRODUCTION" else gripper_retune_path
            ),
            job_binding={
                "place_id": chosen["workspace_id"],
                "cell_calibration_id": chosen["frame_id"],
                "object_profile_id": chosen["object_id"],
                "grasp_profile_id": chosen["grasp_id"],
            },
            workspace_bindings=runtime_workspace_bindings,
            run_live_call=run_live_call,
            task_id=selected["task_id"],
            trajectory_variant_id=selected["variant_id"],
            start_transition_call=start_transition_call,
            requested_count=draft["requested_count"],
            normalized_seed=draft["normalized_seed"],
            yaw_sampling_profile=chosen.get("yaw_sampling_profile"),
            state_space_design_profile=draft.get(
                "state_space_design_profile",
            ),
            initial_object_pose=campaign_initial_pose,
            initial_object_position=draft.get("object_position"),
            **pose_plan,
            data_disposition=disposition,
            dataset_root=(
                production_dataset_root if disposition == "PRODUCTION" else None
            ),
            environment_prepared=True,
            clock=clock,
        )
        return console

    from tools.data_factory.operator.workflow.stored_reviews import StoredCandidateReviews

    application = CollectionOperatorApplication(
        session_id=session_id,
        operator_label=operator_label,
        catalog=catalog,
        initial_selection=selection,
        projector=projection,
        environment_call=environment_call,
        prepare_environment_call=prepare_environment_call,
        prepare_environment_owner_call=prepare_environment_owner_call,
        campaign_factory=campaign_factory,
        workspace_manager_factory=workspace_manager_factory,
        workspace_snapshot_call=workspace_snapshot_call,
        workspace_preview_call=workspace_preview_call,
        catalog_reload_call=catalog_reload_call,
        home_recovery_call=selected_home_recovery_call,
        camera_setup=(camera_setup if camera_environment_call is not None else None),
        camera_bindings_call=(
            camera_bindings_call if camera_environment_call is not None else None
        ),
        camera_refresh_call=(
            camera_refresh_call if camera_environment_call is not None else None
        ),
        start_pose_setup=start_pose_setup,
        start_pose_capture_call=start_pose_capture_call,
        initial_environment=initial_environment,
        effect_scope="PHYSICAL",
        stored_reviews=StoredCandidateReviews(repository / "outputs/data_factory/runs",
                                             operator_label=operator_label, clock=clock,
                                             request_root=repository / "outputs/curator/requests"),
        object_position_call=object_position_call,
        object_position_declare_call=object_position_declare_call,
    )
    application_holder["application"] = application
    # Bind the installed source without scanning it at startup or during views.
    scene_state_path = str(repository / "outputs/data_factory/cells" / initial_job["robot_system_id"] / "scene_state.json")
    application.collection_evidence_call = collection_evidence_call or stored_collection_evidence
    application._collection_source = {"run_directories": [], "scene_state_path": scene_state_path}
    if initial_motion_preset is not None:
        matches = [item for item in catalog.get("motion_presets", []) if item["id"] == initial_motion_preset]
        if len(matches) != 1:
            application.close()
            raise ContractError("MOTION_PRESET_BINDING")
        application.draft["motion_preset"] = {key: matches[0][key] for key in ("id", "digest")}
    return application, {
        "session_id": session_id,
        "effect_scope": "PHYSICAL",
        "data_disposition": (
            "PRODUCTION"
            if initial_data_mode == "GENERAL_COLLECTION" else "TEST_ONLY"
        ),
        "catalog_digest": catalog["catalog_digest"],
        "combination_digest": combination["combination_digest"],
        "camera_device_id": (
            camera_state["device_id"] if camera_state["ready"] else None
        ),
        "camera_bindings": copy.deepcopy(
            camera_state["bindings"] if camera_state["ready"] else {}
        ),
        "camera_binding_digest": (
            camera_state["binding_digest"] if camera_state["ready"] else None
        ),
        "production_writers_enabled": (
            initial_data_mode == "GENERAL_COLLECTION"
        ),
    }
