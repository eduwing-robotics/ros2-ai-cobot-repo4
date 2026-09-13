import copy
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from .test_motion import SCENE, T, snapshot
from . import test_one_job as one_job_test
from . import test_episode_ledger as episode_ledger_test
from . import test_campaign_session as campaign_session_test
from tools.data_factory.motion.pickup_executor import PHASES, PickupExecutor
from tools.data_factory.motion.object_reposition import (
    build_object_reposition_binding,
    yaw_preserving_destination,
)
from tools.data_factory.scene_state import release_slot
from tools.data_factory.state_space import (
    YAW_BINDING_SCHEMA,
    bind_yaw_sample_to_state_space,
    sample_yaw_cdf_strata,
    validate_state_space_design_profile,
    validate_yaw_sampling_profile,
)
from tools.data_factory.one_job import JsonlProcess, run_one_job
from tools.data_factory import run_job
from tools.data_factory.task_recipe import (
    compile_episode_instruction_binding,
    compile_task_binding,
)
from tools.data_factory.campaign_authoring import compile_collection_campaign
from tools.data_factory.campaign_session import CampaignSession
from tools.data_factory.operator.preview import (
    _motion_program,
    make_fake_one_job,
    new_effect_counters,
)
from tools.data_factory.operator.setup.contracts import (
    build_test_only_episode_binding,
    build_test_only_root_binding,
    build_test_only_start_binding,
    initialize_test_only_state_from_user_declaration,
)
from .operator.fixtures import (
    JOB,
    PROFILE,
    campaign_draft,
    compatible_start_fixture,
    motion,
    payload,
    pose_snapshot,
    runtime_motion,
    runtime_validated,
)

def command(op_id="run-1", op="run", value=None):
    return {"schema_version": run_job.COMMAND_SCHEMA, "op_id": op_id, "op": op, "payload": payload() if value is None else value}


class Executor:
    def __init__(self):
        self.transport = T()
        self.value = PickupExecutor(self.transport)

    def __call__(self, request):
        return self.value.process(request)

    def request(self, request, _cancel=None):
        return self(request)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def close(self, timeout_s=None):
        return 0


class RunJobTest(unittest.TestCase):
    def test_learned_preapproval_retains_exact_original_resolver_inputs(self):
        validated = runtime_validated(job={**JOB, "x_mm": 12.0, "y_mm": -8.0, "yaw_deg": 25.0},
                                      input_digests={"cell_calibration": run_job.canonical_digest("source-calibration")})
        original = copy.deepcopy(validated)
        for mismatch in (None, "source", "plan", "destination", "destination_digest",
                         "destination_inputs", "destination_missing", "destination_redigest", "destination_valid"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                request = {"run_root": directory, "run_id": "run-1"}
                root = run_job._prepare_run_dir(request)
                resolved = copy.deepcopy(original)
                plan = {"learned_proposal": {},
                        "resolved_job_digest": resolved["resolved_job_digest"]}
                destination = None
                if mismatch and mismatch.startswith("destination"):
                    destination = {key: copy.deepcopy(resolved[key]) for key in (
                        "normalized_job", "input_digests", "resolved_job_digest",
                    )}
                    destination["normalized_job"].update(
                        job_id="independent-destination-job", place_id="PLACE_B",
                        cell_calibration_id="cal-b", x_mm=-15, y_mm=9, yaw_deg=-32,
                        operator_or_agent_id="destination-operator",
                    )
                    destination["input_digests"]["cell_calibration"] = run_job.canonical_digest("destination-calibration")
                    destination["resolved_job_digest"] = run_job.canonical_digest({
                        "job": destination["normalized_job"], "input_digests": destination["input_digests"],
                    })
                    resolved["destination_resolved_inputs"] = copy.deepcopy(destination)
                    plan["learned_source_program"] = {
                        "schema_version": "fr5.motion_program.v4",
                        "destination_resolved_job_digest": destination["resolved_job_digest"],
                        "destination_binding_digests": copy.deepcopy(destination["input_digests"]),
                    }
                    if mismatch in {"destination", "destination_redigest"}:
                        resolved["destination_resolved_inputs"]["normalized_job"]["x_mm"] += 1
                        if mismatch == "destination_redigest":
                            forged = resolved["destination_resolved_inputs"]
                            forged["resolved_job_digest"] = run_job.canonical_digest({
                                "job": forged["normalized_job"], "input_digests": forged["input_digests"],
                            })
                    elif mismatch == "destination_digest":
                        plan["learned_source_program"]["destination_resolved_job_digest"] = run_job.canonical_digest("other-destination")
                    elif mismatch == "destination_inputs":
                        plan["learned_source_program"]["destination_binding_digests"]["cell_calibration"] = run_job.canonical_digest("other-calibration")
                    elif mismatch == "destination_missing":
                        del resolved["destination_resolved_inputs"]
                if mismatch == "source":
                    resolved["normalized_job"]["x_mm"] += 1
                elif mismatch == "plan":
                    plan["resolved_job_digest"] = run_job.canonical_digest("other-source")
                digest = run_job.canonical_digest(plan)
                safety = {
                    "schema_version": "data_factory.precommit_safety.v1",
                    "run_id": "run-1", "approved_plan_digest": digest,
                    "scene_binding_digest": run_job.canonical_digest(SCENE),
                    "expected_planning_scene_digest": run_job.canonical_digest("scene"),
                    "post_reset_safe_snapshot_digest": None, "status": "PENDING",
                }
                precommit = {
                    "schema_version": "data_factory.precommit_evidence.v1",
                    **{key: safety[key] for key in (
                        "run_id", "approved_plan_digest", "scene_binding_digest",
                        "expected_planning_scene_digest",
                    )},
                }
                for name in ("planning_scene_readback", "collision_report", "plan_only_no_motion"):
                    precommit[name] = {"fixture": name}
                    safety[name + "_digest"] = run_job.canonical_digest(precommit[name])
                planned = {"plan_digest": digest, "plan_envelope": {
                    "plan": plan, "precommit_safety": safety,
                    "precommit_evidence": precommit, "operator_summary": {},
                }}
                if mismatch and mismatch != "destination_valid":
                    code = "PREAPPROVAL_DESTINATION_RESOLVED_INPUTS" if mismatch.startswith("destination") else "PREAPPROVAL_RESOLVED_INPUTS"
                    with self.assertRaisesRegex(run_job.ContractError, code):
                        run_job._write_preapproval_evidence(request, resolved, planned, None)
                    self.assertEqual(list(root.iterdir()), [])
                    continue
                evidence = run_job._write_preapproval_evidence(request, resolved, planned, None)
                receipt = {key: original[key] for key in (
                    "normalized_job", "input_digests", "resolved_job_digest",
                )}
                self.assertEqual(evidence["resolved_inputs"], receipt)
                if destination is not None:
                    self.assertEqual(evidence["destination_resolved_inputs"], destination)
                    resolved["destination_resolved_inputs"]["normalized_job"]["x_mm"] += 2
                    self.assertEqual(evidence["destination_resolved_inputs"], destination)
                self.assertEqual(evidence["plan_envelope"]["plan"], plan)
                self.assertIsNone(evidence["trajectory_variant_binding"])
                # A later in-memory/current-position update cannot rewrite the
                # original condition retained before approval and execution.
                resolved["normalized_job"]["x_mm"] += 2
                self.assertEqual(evidence["resolved_inputs"], receipt)
                self.assertEqual(run_job.load_json_strict(root / "preapproval_evidence.json"), evidence)

    def test_learned_terminal_retains_original_result_for_diagnostic_consumer(self):
        result = {"run_id": "run-1", "execution_evidence": {"learned_execution": {}}}
        diagnostic = {"lifecycle_result_digest": run_job.canonical_digest(result)}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
            return_value=diagnostic,
        ) as derive:
            request = {"run_root": directory, "run_id": "run-1"}
            root = run_job._prepare_run_dir(request)
            self.assertEqual(run_job.learned_run_diagnostic(result, payload=request), diagnostic)
            target = root / "learned_lifecycle_result.json"
            self.assertEqual(run_job.load_json_strict(target), result)
            before = (target.read_bytes(), target.stat().st_mtime_ns)
            self.assertEqual(run_job.learned_run_diagnostic(result, payload=request), diagnostic)
            self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)
            with self.assertRaisesRegex(run_job.ContractError, "LEARNED_RESULT_CONFLICT"):
                run_job.learned_run_diagnostic({**result, "code": "different"}, payload=request)
            self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)
            self.assertEqual(derive.call_args_list[0], mock.call(result))

    def test_learned_diagnostic_rejection_or_wrong_run_does_not_publish(self):
        result = {"run_id": "run-1", "execution_evidence": {"learned_execution": {}}}
        with tempfile.TemporaryDirectory() as directory:
            request = {"run_root": directory, "run_id": "run-1"}
            root = run_job._prepare_run_dir(request)
            with mock.patch("tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
                            side_effect=run_job.ContractError("INVALID_TRACE")):
                with self.assertRaisesRegex(run_job.ContractError, "INVALID_TRACE"):
                    run_job.learned_run_diagnostic(result, payload=request)
            self.assertEqual(list(root.iterdir()), [])
            with mock.patch("tools.data_factory.rollout.evidence_boundary.build_run_diagnostic", return_value={}):
                with self.assertRaisesRegex(run_job.ContractError, "LEARNED_RESULT_RUN"):
                    run_job.learned_run_diagnostic({**result, "run_id": "other"}, payload=request)
            self.assertIsNone(run_job.learned_run_diagnostic({}, payload=request))
            self.assertEqual(list(root.iterdir()), [])

    def test_native_stream_diagnostic_uses_existing_canonical_persistence_boundary(self):
        result = {"run_id": "run-1", "plan_envelope": {"plan": {
            "schema_version": "data_factory.native_learned_task_plan.v1"}}, "execution_evidence": {
            "native_stream": {"status": "NATIVE_HANDLES_TERMINAL", "task_outcome": "UNKNOWN"}}}
        diagnostic = {"lifecycle_result_digest": run_job.canonical_digest(result)}
        with tempfile.TemporaryDirectory() as directory:
            request = {"run_root": directory, "run_id": "run-1"}
            root = run_job._prepare_run_dir(request)
            target = root / "learned_lifecycle_result.json"
            with mock.patch("tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
                            side_effect=run_job.ContractError("ROLLOUT_RUN_DIAGNOSTIC_BINDING")) as derive:
                with self.assertRaisesRegex(run_job.ContractError, "ROLLOUT_RUN_DIAGNOSTIC_BINDING"):
                    run_job.learned_run_diagnostic(result, payload=request)
                derive.assert_called_once_with(result)
            self.assertFalse(target.exists())
            with mock.patch("tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
                            return_value=diagnostic) as derive:
                self.assertEqual(run_job.learned_run_diagnostic(result, payload=request), diagnostic)
                derive.assert_called_once_with(result)
            self.assertEqual(run_job.load_json_strict(target), result)
            self.assertNotIn("learned_execution", result["execution_evidence"])
            with mock.patch("tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
                            return_value=diagnostic) as derive:
                # A pre-execute rejection has no stream. The canonical builder
                # decides whether its retained lifecycle is diagnosable.
                rejected = {**result, "execution_evidence": None}
                self.assertEqual(run_job.learned_run_diagnostic(rejected), diagnostic)
                derive.assert_called_once_with(rejected)

    def test_failed_start_retains_canonical_lifecycle_without_inventing_dispatch(self):
        started = {
            "run_id": "run-1", "code": "LEARNED_STALE_STATE",
            "execution_evidence": {"learned_execution": {}},
            "readiness_failure_evidence": None,
        }
        diagnostic = {
            "schema_version": "data_factory.rollout_run_diagnostic.v1",
            "execution_trace": {
                "status": "FAILED", "failure_code": "LEARNED_STALE_STATE",
            },
            "task_effectiveness": "UNKNOWN", "physical_qualification": "UNKNOWN",
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
            return_value=diagnostic,
        ):
            request = {"run_root": directory, "run_id": "run-1"}
            run_job._prepare_run_dir(request)
            self.assertEqual(run_job._failed_start_data(started, request), diagnostic)
            self.assertEqual(
                run_job.load_json_strict(
                    Path(directory) / "run-1" / "learned_lifecycle_result.json"
                ),
                started,
            )

        readiness = {"code": "RECORDER_READINESS_TIMEOUT"}
        with mock.patch.object(run_job, "learned_run_diagnostic", return_value=None):
            self.assertEqual(
                run_job._failed_start_data(
                    {"readiness_failure_evidence": readiness}, {"run_id": "run-1"},
                ),
                {
                    "mode": "live", "readiness_failure": readiness,
                    "recorder_goal_count": 1, "execute_goal_count": 0,
                    "camera_semantic_authority": False, "training_authorized": False,
                },
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
            side_effect=run_job.ContractError("ROLLOUT_RUN_DIAGNOSTIC_BINDING"),
        ):
            request = {"run_root": directory, "run_id": "run-1"}
            run_job._prepare_run_dir(request)
            fallback = run_job._failed_start_data(started, request)
            self.assertNotIn("execute_goal_count", fallback)
            self.assertFalse(
                (Path(directory) / "run-1" / "learned_lifecycle_result.json").exists()
            )
            self.assertEqual(started["code"], "LEARNED_STALE_STATE")

    def test_native_load_mapping_or_cancel_failure_precedes_live_children_and_run_writes(self):
        from tools.data_factory.learned_action_adapter import NativeSmolVLA
        for failure in ("load", "mapping", "cancel", "warmup", "warmup_cancel"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                profile = copy.deepcopy(PROFILE)
                profile.update(camera_profile="up-wrist", camera_roles=["up", "wrist"],
                    camera_serials={"up": "up", "wrist": "wrist"}, camera_topics={"up": "/up", "wrist": "/wrist"})
                validated = runtime_validated(job={**JOB, "instruction": "synthetic probe"}, profile=profile)
                program = runtime_motion(validated)
                binding = {"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                    "calendar_to_system_offset_s": 0., "uncertainty_s": .001, "system_anchor_s": 9.,
                    "steady_anchor_s": 9., "valid_until_system_s": 100.}
                (root / "clock.json").write_text(json.dumps(binding))
                (root / "train_config.json").write_text(json.dumps({"rename_map": {"observation.images.side": "observation.images.camera2"}}))
                value = {**payload("live"), "camera_profile": "up-wrist", "learned_checkpoint": str(root), "gripper_source_clock": str(root / "clock.json")}
                cancel = threading.Event()
                if failure.startswith("warmup"):
                    (root / "train_config.json").write_text(json.dumps({"rename_map": {"observation.images.up": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}}))
                def discarded_warmup(**_):
                    if failure == "warmup_cancel":
                        cancel.set()
                        return {}
                    raise run_job.ContractError("LEARNED_WARMUP_FAILED")
                def load(*_, **__):
                    if failure == "load":
                        raise run_job.ContractError("LEARNED_CHECKPOINT_LOAD_FAILED")
                    if failure == "cancel": cancel.set()
                    return SimpleNamespace(policy_dir=root, warmup=discarded_warmup)
                cell = SimpleNamespace(read=lambda: {"robot_system_id": value["expected_robot_system_id"], "cell_ready": True})
                with mock.patch.object(NativeSmolVLA, "load", side_effect=load), \
                     mock.patch.object(run_job, "CellStateStore", return_value=cell), \
                     mock.patch.object(run_job, "SceneStateStore"), \
                     mock.patch.object(run_job, "_prepare_run_dir") as writes, \
                     mock.patch.object(run_job, "ResourceMonitor") as resources:
                    child, warmup = mock.Mock(), mock.Mock()
                    result = run_job.run_live(value, cancel, lambda _: None, resolver=lambda _: (validated, program, SCENE),
                        executor_factory=child, camera_warmup_call=warmup)
                self.assertEqual(result["code"], {"load": "LEARNED_CHECKPOINT_LOAD_FAILED", "mapping": "LEARNED_CAMERA_MAPPING", "cancel": "LEARNED_CANCELLED", "warmup": "LEARNED_WARMUP_FAILED", "warmup_cancel": "LEARNED_CANCELLED"}[failure])
                child.assert_not_called()
                warmup.assert_not_called()
                writes.assert_not_called()
                resources.assert_not_called()

    def test_explicit_learned_public_fields_reject_incompatible_requests_before_effects(self):
        base = payload("live")
        valid = {**base, "learned_checkpoint": "/synthetic/checkpoint", "gripper_source_clock": "/synthetic/clock.json"}
        self.assertEqual(run_job._run_payload(base), base)
        self.assertEqual(run_job._run_payload(valid), valid)
        causal = {**base, "learned_checkpoint": "/synthetic/checkpoint", "gripper_temporal_policy": "/synthetic/temporal.json"}
        self.assertEqual(run_job._run_payload(causal), causal)
        self.assertEqual(run_job._learned_options(causal), {"checkpoint": "/synthetic/checkpoint", "device": "cpu",
                         "gripper_temporal_policy": "/synthetic/temporal.json"})
        invalid = [{**base, "learned_device": "cuda"}, {**base, "learned_checkpoint": "/checkpoint"},
                   {**base, "gripper_source_clock": "/clock"}, {**valid, "learned_device": "auto"},
                   {**valid, "learned_checkpoint": ""}, {**valid, "gripper_temporal_policy": "/policy"},
                   {**base, "gripper_temporal_policy": "/policy"}, {**causal, "gripper_temporal_policy": ""}]
        for value in invalid:
            with self.subTest(value=value), mock.patch.object(run_job, "run_live") as live:
                session = run_job.RunSession()
                result = session.process(command(value=value))
                self.assertEqual(result["code"], "LEARNED_RUN_INPUTS")
                self.assertIsNone(session.worker)
                live.assert_not_called()
        for extra in ({"approval_scope": "HIL_NUMERIC_PROXY"}, {"campaign_authorization": {}}, {"object_reposition_binding": {}}):
            with mock.patch.object(run_job, "resolve_inputs") as resolver, mock.patch.object(run_job, "_live_executor") as child:
                result = run_job.run_live(valid, threading.Event(), lambda _: None, resolver=resolver, executor_factory=child, **extra)
                self.assertEqual(result["code"], "LEARNED_HUMAN_APPROVAL_REQUIRED")
                resolver.assert_not_called()
                child.assert_not_called()

    def test_normal_stream_grant_and_static_policy_are_distinct_from_finite_output(self):
        from tools.data_factory.rollout.stream_plan import POLICY_SCHEMA
        from tools.data_factory.rollout.finite_plan import PROPOSAL_SCHEMA

        def grant(schema):
            value = {
                "schema_version": "data_factory.learned_task_grant.v1",
                "grant_id": "normal-grant", "issued_by": "operator", "run_id": "run",
                "scope": {"adaptation": {"schema_version": schema}},
                "deadline_s": 100., "terminal_reserve_s": 10., "max_outputs": 1,
                "revoked": False,
            }
            value["grant_digest"] = run_job.canonical_digest(value)
            return value

        self.assertTrue(run_job._normal_stream_grant({"task_grant": grant(POLICY_SCHEMA)}))
        self.assertFalse(run_job._normal_stream_grant({"task_grant": grant(PROPOSAL_SCHEMA[:-1] + "2")}))
        with tempfile.TemporaryDirectory() as directory:
            robot = Path(directory) / "robot.urdf"
            robot.write_text("<robot name='retained'/>")
            source = {"steps": [
                {"limits": {"velocity_scaling": .1}},
                {"limits": {"velocity_scaling": .05}},
            ]}
            inputs = {"fps": 30, "warmup": {"output_disposition": "DISCARDED"}}
            checkpoint = {"tree_digest": run_job.canonical_digest("tree")}
            policy = run_job._native_stream_policy(
                SimpleNamespace(checkpoint=checkpoint), source,
                {"urdf": str(robot)}, inputs, instruction="pick and place",
            )
        self.assertEqual(policy, {
            "schema_version": POLICY_SCHEMA,
            "checkpoint": checkpoint,
            "instruction": "pick and place",
            "robot_description": "<robot name='retained'/>",
            "velocity_scaling": .05,
            "period_s": 1 / 30,
            "max_observation_age_s": .3,
            "runtime_inputs": inputs,
        })
        checkpoint["tree_digest"] = "changed"
        inputs["fps"] = 1
        self.assertNotEqual(policy["checkpoint"], checkpoint)
        self.assertEqual(policy["runtime_inputs"]["fps"], 30)

    def test_normal_stream_routes_static_plan_and_stops_before_legacy_preapproval_or_effects(self):
        from tools.data_factory.rollout.stream_plan import POLICY_SCHEMA

        validated = runtime_validated(job={
            **JOB, "instruction": "pick and place",
        })
        program = runtime_motion(validated, continuous=True)
        request = {
            **payload("live"),
            "learned_checkpoint": "/synthetic/checkpoint",
            "gripper_source_clock": "/synthetic/clock.json",
        }
        grant = {
            "schema_version": "data_factory.learned_task_grant.v1",
            "grant_id": "normal-grant", "issued_by": "operator",
            "run_id": request["run_id"],
            "scope": {"adaptation": {"schema_version": POLICY_SCHEMA}},
            "deadline_s": 100., "terminal_reserve_s": 10.,
            "max_outputs": 1, "revoked": False,
        }
        grant["grant_digest"] = run_job.canonical_digest(grant)
        request["task_grant"] = grant
        policy = {
            "schema_version": POLICY_SCHEMA,
            "checkpoint": {"tree_digest": run_job.canonical_digest("tree")},
            "instruction": validated["normalized_job"]["instruction"],
            "robot_description": "<robot name='retained'/>",
            "velocity_scaling": .1,
            "period_s": 1 / 30,
            "max_observation_age_s": .3,
            "runtime_inputs": {"fps": 30},
        }
        from tools.data_factory.rollout.task_authority import task_scope
        grant["scope"] = task_scope(program, SCENE, policy)
        grant["grant_digest"] = run_job.canonical_digest({
            key: value for key, value in grant.items()
            if key != "grant_digest"
        })
        planned = {
            "ok": True, "code": "PLANNED", "state": "PLANNED",
            "plan_digest": run_job.canonical_digest("normal-plan"),
        }

        class Job:
            state = "IDLE"
            readiness_contract = None

            def __init__(self):
                self.plan_stream = mock.Mock(return_value=planned)
                self.plan_only = mock.Mock(side_effect=AssertionError(
                    "normal task must not enter finite planning",
                ))

            def set_lifecycle_event_call(self, callback):
                self.lifecycle_event_call = callback

        class Child:
            def __init__(self):
                self.closed = False

            def close(self, **_kwargs):
                self.closed = True
                return 0

        job, child = Job(), Child()
        recorder = mock.Mock(side_effect=AssertionError(
            "static normal planning must not start a recorder",
        ))
        with tempfile.TemporaryDirectory() as directory:
            robot = Path(directory) / "robot.urdf"
            robot.write_text(policy["robot_description"])
            request["urdf"] = str(robot)
            with (
                mock.patch.object(
                    run_job, "_native_run_inputs",
                    return_value=(SimpleNamespace(checkpoint=policy["checkpoint"]), {"fps": 30}),
                ),
                mock.patch.object(
                    run_job, "CellStateStore",
                    return_value=SimpleNamespace(read=lambda: {
                        "robot_system_id": request["expected_robot_system_id"],
                        "cell_ready": True,
                    }),
                ),
                mock.patch.object(run_job, "SceneStateStore"),
                mock.patch.object(run_job, "_prepare_run_dir"),
                mock.patch.object(run_job, "_infer_native_program", side_effect=AssertionError(
                    "normal task must not infer a finite program before planning",
                )),
                mock.patch.object(run_job, "_operator_summary", side_effect=AssertionError(
                    "normal task has no legacy finite operator summary",
                )),
                mock.patch.object(run_job, "_write_preapproval_evidence", side_effect=AssertionError(
                    "normal task has no legacy finite preapproval envelope",
                )),
                mock.patch.object(run_job, "ResourceMonitor") as resources,
            ):
                result = run_job.run_live(
                    request, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, program, SCENE),
                    executor_factory=lambda *_: child,
                    recorder_factory=recorder,
                    camera_warmup_call=lambda *_: {}, one_job=job,
                )
        self.assertEqual(
            (result["ok"], result["code"], result["state"]),
            (False, "LEARNED_STREAM_COMMIT_UNAVAILABLE", "PLANNED"),
        )
        job.plan_stream.assert_called_once_with(
            request["run_id"], program, SCENE, policy,
        )
        job.plan_only.assert_not_called()
        recorder.assert_not_called()
        resources.assert_not_called()
        self.assertTrue(child.closed)

    def test_learned_cli_preserves_explicit_inputs_without_enabling_live_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            job_path = Path(directory) / "job.json"
            job_path.write_text(json.dumps(JOB))
            original = payload()
            argv = []
            for key, value in original.items():
                argv += ["--" + key.replace("_", "-"), str(job_path) if key == "job" else str(value)]
            argv += ["--learned-checkpoint", "/synthetic/checkpoint", "--gripper-source-clock", "/synthetic/clock.json", "--learned-device", "cuda"]
            parsed = run_job._human_payload(run_job._parser().parse_args(argv))
            self.assertEqual(parsed["mode"], "plan_only")
            self.assertEqual(parsed["learned_checkpoint"], "/synthetic/checkpoint")
            self.assertEqual(parsed["gripper_source_clock"], "/synthetic/clock.json")
            self.assertEqual(parsed["learned_device"], "cuda")
        with mock.patch.object(run_job, "JsonlProcess") as child:
            run_job._live_executor({**payload("live"), "learned_checkpoint": "checkpoint", "gripper_source_clock": "/synthetic/clock.json"}, 3.)
        self.assertEqual(child.call_args.args[0][-2:], ["--gripper-source-clock", "/synthetic/clock.json"])

    def test_learned_planner_forwards_the_existing_measured_clock_file(self):
        with mock.patch.object(run_job, "JsonlProcess") as child:
            run_job._executor(3., gripper_source_clock="/synthetic/measured-clock.json")
        self.assertEqual(child.call_args.args[0][-2:], ["--gripper-source-clock", "/synthetic/measured-clock.json"])
        self.assertIn("--ros-plan-only", child.call_args.args[0])
        self.assertNotIn("--ros-live", child.call_args.args[0])

    def test_temporal_policy_is_explicit_in_cli_and_both_executor_modes(self):
        args = run_job._parser().parse_args(["--gripper-temporal-policy", "/synthetic/temporal.json"])
        self.assertEqual(args.gripper_temporal_policy, "/synthetic/temporal.json")
        with mock.patch.object(run_job, "JsonlProcess") as child:
            run_job._executor(3., gripper_temporal_policy="/synthetic/temporal.json")
            self.assertIn("--ros-plan-only", child.call_args.args[0])
            self.assertEqual(child.call_args.args[0][-2:], ["--gripper-temporal-policy", "/synthetic/temporal.json"])
            run_job._live_executor({**payload("live"), "learned_checkpoint": "/checkpoint",
                                   "gripper_temporal_policy": "/synthetic/temporal.json"}, 3.)
            self.assertIn("--ros-live", child.call_args.args[0])
            self.assertEqual(child.call_args.args[0][-2:], ["--gripper-temporal-policy", "/synthetic/temporal.json"])
        with mock.patch.object(run_job, "JsonlProcess") as child, self.assertRaisesRegex(run_job.ContractError, "LEARNED_RUN_INPUTS"):
            run_job._executor(3., gripper_temporal_policy="/policy", gripper_source_clock="/clock")
        child.assert_not_called()

    def test_native_observation_failure_and_cancellation_close_the_same_child(self):
        from tools.data_factory.learned_action_adapter import NativeSmolVLA
        from tools.data_factory.rollout.finite_plan import FinitePolicyInference
        for failure in ("capture", "cancel_after_load", "cancel_after_capture", "malformed", "ambiguous"):
            with self.subTest(failure=failure):
                cancel = threading.Event()
                child = mock.Mock()
                factory = mock.Mock(return_value=child)
                def load(*_, **__):
                    if failure == "cancel_after_load":
                        cancel.set()
                    return SimpleNamespace(checkpoint={}, prepare_inference=lambda: nullcontext(None))
                def capture(*_):
                    if failure == "cancel_after_capture":
                        cancel.set()
                    if failure == "capture":
                        return {"ok": False, "code": "LEARNED_STALE_OBSERVATION"}
                    frame = {"dtype": "uint8", "color_space": "RGB", "shape": [1, 1, 3], "data_hex": "000000"}
                    value = {key: dict(frame) for key in ("observation.images.camera1", "observation.images.camera2")}
                    if failure == "malformed":
                        value["observation.images.camera1"]["data"] = "ambiguous"
                    return {"ok": True, "data": {"observation": value}}
                child.request.side_effect = capture
                with mock.patch.object(NativeSmolVLA, "load", side_effect=load) as loader, \
                        mock.patch.object(FinitePolicyInference, "propose") as infer:
                    result = run_job.run_learned_plan_only({"run_id": "capture"}, cancel, lambda _: None,
                        checkpoint="synthetic", camera_topics={"camera1": "/up", "camera2": "/wrist"},
                        observation={} if failure == "ambiguous" else None,
                        instruction="probe", period_s=.1,
                        resolver=lambda _: ({}, motion(), SCENE), executor_factory=factory)
                self.assertFalse(result["ok"], result)
                expected = {"capture": "LEARNED_STALE_OBSERVATION", "cancel_after_load": "LEARNED_CANCELLED",
                            "cancel_after_capture": "LEARNED_CANCELLED", "malformed": "LEARNED_OBSERVATION_SCHEMA",
                            "ambiguous": "LEARNED_OBSERVATION_SCHEMA"}[failure]
                self.assertEqual(result["code"], expected)
                infer.assert_not_called()
                if failure in {"cancel_after_load", "ambiguous"}:
                    factory.assert_not_called()
                    child.close.assert_not_called()
                else:
                    child.close.assert_called_once()
                    self.assertEqual(child.request.call_args[0][0]["op"], "capture_observation")
                if failure == "ambiguous":
                    loader.assert_not_called()

    def test_runtime_yaw_scope_rejects_redigested_out_of_design_cell(self):
        repository = Path(__file__).resolve().parents[2]
        object_profile = run_job.load_json_strict(
            repository / "config/data_factory/objects/wood-cube-24mm-r001.json",
        )
        grasp_profile = run_job.load_json_strict(
            repository / "config/data_factory/grasps/"
            "wood-cube-24mm-top-3p5mm-r001.json",
        )
        yaw_profile = validate_yaw_sampling_profile(
            run_job.load_json_strict(
                repository / "config/data_factory/yaw_sampling_profiles/"
                "wood-cube-24mm-top-r001.json",
            ),
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        design_profile = validate_state_space_design_profile(
            run_job.load_json_strict(
                repository / "config/data_factory/state_space_design_profiles/"
                "wood-cube-24mm-a4-cdf3-r001.json",
            ),
            object_profile=object_profile, grasp_profile=grasp_profile,
            yaw_sampling_profile=yaw_profile,
        )
        yaw_sample = next(
            item for item in sample_yaw_cdf_strata(
                yaw_profile, sampling_seed=17,
                sweep_identity={"sweep": "runtime-scope-r001"},
                strata_count=3, conditioned_yaw_deg=0.0,
            )
            if item["sample_origin"] == "CONDITIONED_SOURCE_ANCHOR"
        )
        binding = bind_yaw_sample_to_state_space(
            yaw_sample, state_space_design_profile=design_profile,
            spatial_cell_index=7, spatial_row=1, spatial_column=2,
        )
        validated = runtime_validated(
            job={
                **JOB, "yaw_deg": 0.0,
                "object_profile_id": object_profile["object_profile_id"],
                "grasp_profile_id": grasp_profile["grasp_profile_id"],
            },
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        self.assertEqual(
            run_job._validated_yaw_sample_scope(
                binding, yaw_profile, design_profile,
                validated=validated, bound_runtime=True,
                preapproval_checklist={"yaw_sample_binding": binding},
            ),
            binding,
        )
        self.assertEqual(
            run_job._validated_yaw_sample_scope(
                yaw_sample, yaw_profile, None,
                validated=validated, bound_runtime=True,
                preapproval_checklist={"yaw_sample_binding": yaw_sample},
            ),
            yaw_sample,
        )
        with self.assertRaisesRegex(
            run_job.ContractError, "YAW_SAMPLE_BINDING_SCOPE",
        ):
            run_job._validated_yaw_sample_scope(
                yaw_sample, yaw_profile, None,
                validated=validated, bound_runtime=True,
                preapproval_checklist={"yaw_sample_binding": yaw_sample},
                require_slotted=True,
            )

        forged = copy.deepcopy(binding)
        forged["spatial_cell_index"] = 999
        forged["binding_digest"] = run_job.canonical_digest({
            key: value for key, value in forged.items()
            if key != "binding_digest"
        })
        with self.assertRaisesRegex(
            run_job.ContractError, "YAW_SAMPLE_BINDING_SCOPE",
        ):
            run_job._validated_yaw_sample_scope(
                forged, yaw_profile, design_profile,
                validated=validated, bound_runtime=True,
                preapproval_checklist={"yaw_sample_binding": forged},
            )
        with self.assertRaisesRegex(
            run_job.ContractError, "YAW_SAMPLE_BINDING_SCOPE",
        ):
            run_job._validated_yaw_sample_scope(
                binding, yaw_profile, None,
                validated=validated, bound_runtime=True,
                preapproval_checklist={"yaw_sample_binding": binding},
            )

    def test_preapproval_evidence_durably_binds_current_yaw_sample(self):
        repository = Path(__file__).resolve().parents[2]
        object_profile = run_job.load_json_strict(
            repository / "config/data_factory/objects/wood-cube-24mm-r001.json",
        )
        grasp_profile = run_job.load_json_strict(
            repository / "config/data_factory/grasps/"
            "wood-cube-24mm-top-3p5mm-r001.json",
        )
        yaw_profile = validate_yaw_sampling_profile(
            run_job.load_json_strict(
                repository / "config/data_factory/yaw_sampling_profiles/"
                "wood-cube-24mm-top-r001.json",
            ),
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        design_profile = validate_state_space_design_profile(
            run_job.load_json_strict(
                repository / "config/data_factory/state_space_design_profiles/"
                "wood-cube-24mm-a4-cdf3-r001.json",
            ),
            object_profile=object_profile, grasp_profile=grasp_profile,
            yaw_sampling_profile=yaw_profile,
        )
        yaw_sample = next(
            item for item in sample_yaw_cdf_strata(
                yaw_profile, sampling_seed=(1 << 63) + 17,
                sweep_identity={"sweep": "preapproval-r001"},
                strata_count=3, conditioned_yaw_deg=0.0,
            )
            if item["sample_origin"] == "CONDITIONED_SOURCE_ANCHOR"
        )
        yaw_binding = bind_yaw_sample_to_state_space(
            yaw_sample, state_space_design_profile=design_profile,
            spatial_cell_index=7, spatial_row=1, spatial_column=2,
        )
        validated = runtime_validated(
            job={
                **JOB, "yaw_deg": 0.0,
                "object_profile_id": object_profile["object_profile_id"],
                "grasp_profile_id": grasp_profile["grasp_profile_id"],
            },
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        program = runtime_motion(validated)
        with tempfile.TemporaryDirectory() as directory:
            value = payload("live")
            value["run_root"] = directory
            (Path(directory) / value["run_id"]).mkdir()
            trajectory = run_job._trajectory_binding(value, validated, program)
            plan = {"motion_program_digest": run_job.canonical_digest(program)}
            plan_digest = run_job.canonical_digest(plan)
            readback = {"kind": "planning-scene"}
            collision = {"kind": "collision"}
            no_motion = {"kind": "no-motion"}
            safety = {
                "schema_version": "data_factory.precommit_safety.v1",
                "run_id": value["run_id"],
                "approved_plan_digest": plan_digest,
                "scene_binding_digest": run_job.canonical_digest(SCENE),
                "expected_planning_scene_digest": run_job.canonical_digest(
                    "expected-scene",
                ),
                "planning_scene_readback_digest": run_job.canonical_digest(
                    readback,
                ),
                "collision_report_digest": run_job.canonical_digest(collision),
                "plan_only_no_motion_digest": run_job.canonical_digest(no_motion),
                "post_reset_safe_snapshot_digest": None,
                "status": "PENDING",
            }
            evidence = run_job._write_preapproval_evidence(
                value, validated, {
                    "plan_digest": plan_digest,
                    "plan_envelope": {
                        "plan": plan,
                        "precommit_safety": safety,
                        "precommit_evidence": {
                            "schema_version":
                            "data_factory.precommit_evidence.v1",
                            "run_id": value["run_id"],
                            "approved_plan_digest": plan_digest,
                            "scene_binding_digest": safety[
                                "scene_binding_digest"
                            ],
                            "expected_planning_scene_digest": safety[
                                "expected_planning_scene_digest"
                            ],
                            "planning_scene_readback": readback,
                            "collision_report": collision,
                            "plan_only_no_motion": no_motion,
                        },
                        "operator_summary": {},
                    },
                }, trajectory, yaw_sample_binding=yaw_binding,
            )
            stored = run_job.load_json_strict(
                Path(directory) / value["run_id"]
                / "preapproval_evidence.json",
            )
        self.assertEqual(stored, evidence)
        self.assertNotIn("resolved_inputs", stored)
        self.assertEqual(
            stored["yaw_sample_binding"]["schema_version"],
            YAW_BINDING_SCHEMA,
        )
        self.assertEqual(
            (
                stored["schema_version"], stored["yaw_sample_binding"],
                stored["yaw_sample_binding_digest"],
            ),
            (
                "data_factory.preapproval_evidence.v4", yaw_binding,
                yaw_binding["binding_digest"],
            ),
        )

    def test_runtime_child_keeps_requests_cancel_aware(self):
        cancel = threading.Event()
        calls = []

        class Child:
            def request(self, request, token):
                if request["op"] == "cancel":
                    self_outer.assertIsNone(token)
                else:
                    self_outer.assertIs(token, cancel)
                    if token.is_set():
                        raise run_job.ContractError("JSONL_REQUEST_CANCELLED")
                calls.append(request["op"])
                if request["op"] == "execute":
                    cancel.set()
                return {"op": request["op"], "ack": True}

        self_outer = self
        child = Child()
        self.assertEqual(
            run_job._runtime_child_request(
                child, {"op": "execute"}, cancel,
            ),
            {"op": "execute", "ack": True},
        )
        with self.assertRaisesRegex(
            run_job.ContractError, "JSONL_REQUEST_CANCELLED",
        ):
            run_job._runtime_child_request(
                child, {"op": "heartbeat"}, cancel,
            )
        self.assertEqual(
            run_job._runtime_child_request(
                child, {"op": "cancel"}, cancel,
            ),
            {"op": "cancel", "ack": True},
        )
        self.assertEqual(calls, ["execute", "cancel"])

    def test_postcommit_overlap_is_validator_only_and_joins_both_results(self):
        validator_started = threading.Event()
        reposition_started = threading.Event()
        validator_finished = threading.Event()
        reposition_finished = threading.Event()

        def validate(dataset_root, payload_value, profile):
            self.assertEqual(dataset_root, "/dataset")
            self.assertEqual(payload_value, {"run_id": "episode-run"})
            self.assertEqual(profile, {"profile": "bound"})
            validator_started.set()
            self.assertTrue(reposition_started.wait(1))
            validator_finished.set()
            return {"lane": "validator"}

        def reposition():
            reposition_started.set()
            self.assertTrue(validator_started.wait(1))
            reposition_finished.set()
            return {"lane": "reposition"}

        technical, moved = run_job._postcommit_validate_and_reposition(
            validate, "/dataset", {"run_id": "episode-run"},
            {"profile": "bound"}, reposition,
        )
        self.assertEqual(
            (technical, moved),
            ({"lane": "validator"}, {"lane": "reposition"}),
        )
        self.assertTrue(validator_finished.is_set())
        self.assertTrue(reposition_finished.is_set())

        serial_calls = []
        technical, moved = run_job._postcommit_validate_and_reposition(
            lambda *_: serial_calls.append("validator") or {"lane": "validator"},
            "/dataset", {"run_id": "episode-run"}, {"profile": "bound"},
        )
        self.assertEqual((serial_calls, moved), (["validator"], None))

    def test_only_an_explicit_failed_plan_only_receipt_releases_committed_parent(self):
        failed = {
            "status": "FAIL", "plan_digest": None, "plan_artifact_digest": None,
            "execution_response": {
                "ok": False, "state": "BLOCKED", "executor_state": "IDLE",
                "plan_envelope": None, "execution_evidence": None, "cancel_error": None,
            },
        }
        self.assertTrue(run_job._reposition_failed_before_motion(failed))
        for response in (
            None, {}, {**failed["execution_response"], "executor_state": "EXECUTING"},
            {**failed["execution_response"], "cancel_error": "CANCEL_UNCONFIRMED"},
            {**failed["execution_response"], "execution_evidence": {}},
        ):
            with self.subTest(response=response):
                self.assertFalse(run_job._reposition_failed_before_motion({
                    **failed, "execution_response": response,
                }))
        self.assertFalse(run_job._reposition_failed_before_motion({
            **failed, "plan_digest": "sha256:" + "1" * 64,
        }))
        self.assertFalse(run_job._reposition_failed_before_motion(None))

    def test_committed_parent_review_handoff_keeps_raw_data_and_all_other_gates(self):
        release = {
            "schema_version": "data_factory.recycle_release_evidence.v2",
            "release_outcome": "EXPECTED_LANDED", "outcome_source": "CAMPAIGN_CONTROL_PROXY",
            "release_slot_id": "slot-a",
        }
        recorded = {
            "state": "AWAITING_CELL_READY", "recorder_state": "COMMITTED",
            "executor_state": "COMPLETED", "frozen_rows": 12, "rows_after_recycle": 12,
            "execution_evidence": {
                "release_evidence": release,
                "scene_transition": {
                    "scene_state_digest": run_job.canonical_digest("scene"),
                    "release_evidence_digest": run_job.canonical_digest(release),
                },
            },
        }
        technical = {"status": "PASS", "plan_digest": run_job.canonical_digest("plan")}
        validated = {"resolved_job_digest": run_job.canonical_digest("job"), "normalized_job": {"task": "pick_place"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "parent-run"
            run_dir.mkdir()
            raw = root / "raw-episode.bin"
            raw.write_bytes(b"immutable recording and provenance")
            payload_value = {"run_id": "parent-run", "run_root": directory}
            ledger = {"path": str(run_dir / "episode_ledger.json"), "technical_status": "PASS"}
            bound = {**ledger, "review_status": "PENDING", "training_status": "NOT_AUTHORIZED"}
            options = {
                "result": recorded, "summary": {"recycle": {"release_slot_id": "slot-a"}},
                "approval_scope": "HIL_NUMERIC_PROXY", "enabled": True,
                "reposition_error": "ROS_GRIPPER_SETTINGS_UNVERIFIED", "postcommit_error": None,
            }
            with mock.patch.object(run_job, "bind_candidate_episode_state", return_value=bound) as binder:
                reference, offer = run_job._committed_review_handoff(
                    payload_value, validated, technical, ledger, **options,
                )
            path = run_dir / "candidate_admission.json"
            original = path.read_bytes()
            candidate = json.loads(original)
            self.assertEqual((candidate["operational_gate"], candidate["operational_source"], candidate["semantic_status"]), ("PASS", "HIL_PROXY", "PENDING"))
            self.assertEqual(offer["expected_file_digest"], run_job.canonical_digest(candidate))
            self.assertEqual(offer["ledger_reference"], reference)
            binder.assert_called_once_with(ledger, path)
            self.assertEqual(raw.read_bytes(), b"immutable recording and provenance")
            with mock.patch.object(run_job, "write_candidate_admission") as writer:
                for override in (
                    {"enabled": False}, {"postcommit_error": "RESOURCE_EVIDENCE_ERROR"},
                    {"reposition_error": None}, {"summary": {}},
                ):
                    self.assertEqual(run_job._committed_review_handoff(
                        payload_value, validated, technical, ledger, **{**options, **override},
                    ), (ledger, None))
                self.assertEqual(run_job._committed_review_handoff(
                    payload_value, validated, {**technical, "status": "FAIL"}, ledger, **options,
                ), (ledger, None))
                for field, value in (("recorder_state", "FROZEN"), ("executor_state", "EXECUTING"), ("rows_after_recycle", 13)):
                    with self.subTest(field=field), self.assertRaises(run_job.ContractError):
                        run_job._committed_review_handoff(
                            payload_value, validated, technical, ledger,
                            **{**options, "result": {**recorded, field: value}},
                        )
                writer.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_reposition_cancel_after_planning_never_dispatches_motion(self):
        object_profile = {"object_profile_id": "object-r1"}
        grasp_profile = {
            "grasp_profile_id": "grasp-r1",
            "object_profile_id": "object-r1",
        }
        target = {
            "place_id": "place-a", "yaw_deg": 0.0,
            "x_mm": 1.0, "y_mm": 2.0,
        }
        target["yaw_deg"] = 15.0
        source = yaw_preserving_destination(
            {**target, "yaw_deg": 0.0}, target,
        )
        binding = build_object_reposition_binding(
            parent_run_id="episode-run",
            continuation_run_id="episode-run-reposition",
            next_run_id="episode-next", start_state="ON_SURFACE",
            source_pose=source, target_pose=target,
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        scope = {"scope_digest": "sha256:" + "1" * 64}
        authorization = {
            "envelope": {
                "effect_scope": "PHYSICAL",
                "lifecycle_action": "LIVE_COLLECT",
                "task": "pick_place", "robot_system_id": "fr5-lab-a",
                "object_profile_id": "object-r1",
                "grasp_profile_id": "grasp-r1",
            },
        }
        program = {"schema_version": "fixture.motion"}
        plan = {"run_id": binding["continuation_run_id"]}
        plan_digest = run_job.canonical_digest(plan)
        planned = {
            "ok": True, "plan_digest": plan_digest,
            "plan_envelope": {
                "plan": plan,
                "precommit_safety": {
                    "collision_report_digest": "sha256:" + "2" * 64,
                    "plan_only_no_motion_digest": "sha256:" + "3" * 64,
                },
            },
        }
        continuation = {
            "expectation_digest": "sha256:" + "4" * 64,
        }
        cancel = threading.Event()
        calls = []

        class Process:
            def close(self, **_kwargs):
                calls.append("close")

        class MotionOnlyJob:
            def __init__(self, *_args, **_kwargs):
                self.execution_evidence = None

            def plan_only(self, *_args):
                calls.append("plan")
                return copy.deepcopy(planned)

            def approve(self, *_args):
                calls.append("approve")
                raise AssertionError("cancelled reposition must not be approved")

        with tempfile.TemporaryDirectory() as directory:
            parent_payload = {
                "run_id": binding["parent_run_id"],
                "run_root": directory,
                "expected_robot_system_id": "fr5-lab-a",
            }
            Path(directory, binding["parent_run_id"]).mkdir()
            motion_payload = {
                **parent_payload, "run_id": binding["continuation_run_id"],
            }

            def publish(_event):
                calls.append("publish")
                cancel.set()

            with (
                mock.patch.object(
                    run_job, "_validate_object_reposition_preapproval",
                    return_value=scope,
                ),
                mock.patch.object(run_job, "_load", return_value=scope),
                mock.patch.object(
                    run_job, "validate_campaign_authorization",
                    return_value=authorization,
                ),
                mock.patch.object(
                    run_job, "resolve_object_reposition_inputs",
                    return_value=(
                        motion_payload,
                        {"resolved_job_digest": "sha256:" + "5" * 64},
                        program, {}, binding,
                    ),
                ),
                mock.patch.object(
                    run_job, "_object_reposition_continuation_expectation",
                    return_value=continuation,
                ),
                mock.patch.object(run_job, "_timeout_s", return_value=1.0),
                mock.patch.object(run_job, "OneJob", MotionOnlyJob),
            ):
                result = run_job.run_object_reposition(
                    parent_payload, binding, cancel, publish,
                    parent_plan_digest="sha256:" + "6" * 64,
                    operator_id="operator", cell_root=Path(directory) / "cells",
                    executor_factory=lambda *_args, **_kwargs: Process(),
                    campaign_authorization={"fixture": True},
                    data_disposition="TEST_ONLY", preapproval_scope=scope,
                )

        self.assertEqual((result["status"], result["code"]), ("FAIL", "CANCELLED"))
        self.assertEqual(calls, ["plan", "publish", "close"])

    def test_parent_plan_release_edge_exactly_binds_both_reposition_modes(self):
        object_profile = {"object_profile_id": "object-r1"}
        grasp_profile = {
            "grasp_profile_id": "grasp-r1", "object_profile_id": "object-r1",
        }
        source = {
            "place_id": "place-a", "yaw_deg": 0.0,
            "x_mm": 1.0, "y_mm": 2.0,
        }
        target = {
            "place_id": "place-b", "yaw_deg": 15.0,
            "x_mm": 3.0, "y_mm": 4.0,
        }
        held = build_object_reposition_binding(
            parent_run_id="episode-run", continuation_run_id="episode-run",
            next_run_id="episode-next", start_state="HELD_OBJECT",
            source_pose=source, target_pose=target,
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        validated = {"normalized_job": {**source, "task": "pickup_e2e"}}
        held_scene = {
            "release_slot": {
                "pose": target, "role": "DESTINATION_THEN_NEXT_SOURCE",
            },
            "allowed_next_run_id": "episode-next",
        }
        summary = {"recycle": {"release_target": target}}
        self.assertEqual(
            run_job._validate_parent_reposition_edge(
                held, validated=validated, scene_binding=held_scene,
                operator_summary=summary,
            ),
            held,
        )

        rotated = {**target, "yaw_deg": 15.0}
        landed = yaw_preserving_destination(
            {**target, "yaw_deg": 0.0}, rotated,
        )
        surface = build_object_reposition_binding(
            parent_run_id="episode-run",
            continuation_run_id="episode-run-reposition",
            next_run_id="episode-next", start_state="ON_SURFACE",
            source_pose=landed, target_pose=rotated,
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        surface_scene = {
            "release_slot": {
                "pose": landed, "role": "DESTINATION_THEN_NEXT_SOURCE",
            },
            "allowed_next_run_id": "episode-run-reposition",
        }
        self.assertEqual(
            run_job._validate_parent_reposition_edge(
                surface, validated=validated, scene_binding=surface_scene,
                operator_summary={"recycle": {"release_target": landed}},
            ),
            surface,
        )
        for forged_scene in (
            {**surface_scene, "allowed_next_run_id": "episode-next"},
            {
                **surface_scene,
                "release_slot": {
                    **surface_scene["release_slot"], "pose": rotated,
                },
            },
        ):
            with self.assertRaisesRegex(
                run_job.ContractError, "OBJECT_REPOSITION_PARENT_EDGE",
            ):
                run_job._validate_parent_reposition_edge(
                    surface, validated=validated, scene_binding=forged_scene,
                    operator_summary={"recycle": {"release_target": landed}},
                )

    def test_trajectory_variant_binds_object_dimensions_seed_and_target_yaw(self):
        approach_profile = run_job.load_json_strict(
            Path(__file__).resolve().parents[2]
            / "config/data_factory/approach_sampling_profiles/"
            "wood-cube-24mm-top-wrist-r001.json"
        )
        value = payload()
        value.update(
            trajectory_variant_id="TWO_STAGE_ALIGN_V2",
            trajectory_sampling_seed=23,
            trajectory_sampling_design={
                "sample_rank": 2, "design_size": 7,
                "design_digest": run_job.canonical_digest("design"),
            },
        )
        self.assertEqual(
            run_job._run_payload(value)["trajectory_sampling_seed"], 23,
        )
        for invalid in (
            {**payload(), "trajectory_variant_id": "TWO_STAGE_ALIGN_V2"},
            {**value, "trajectory_sampling_seed": -1},
            {**value, "trajectory_variant_id": "UNKNOWN"},
            {**value, "trajectory_sampling_design": {"sample_rank": 7, "design_size": 7, "design_digest": run_job.canonical_digest("design")}},
        ):
            with self.assertRaisesRegex(run_job.ContractError, "RUN_PAYLOAD"):
                run_job._run_payload(invalid)

        validated = {
            "normalized_job": {
                **JOB, "task": "pickup_e2e", "place_id": "PLACE_A",
                "yaw_deg": 45.0, "x_mm": 0.0, "y_mm": 0.0,
            },
            "object_profile": {"dimensions_mm": [24.0, 24.0, 24.0]},
        }
        source_program = motion()
        for frame in ("base_tcp", "base_tool"):
            source_program["steps"][0]["target"][frame]["translation_m"] = [
                0.0, 0.0, 0.1,
            ]
            source_program["steps"][1]["target"][frame]["translation_m"] = [
                0.0, 0.0, 0.02,
            ]
        with (
            mock.patch.object(
                run_job, "validate_job_spec", return_value=validated,
            ),
            mock.patch.object(run_job, "_load", return_value={}),
            mock.patch.object(
                run_job, "resolve_motion_program", return_value=source_program,
            ),
            mock.patch.object(
                run_job, "compile_execution_motion_program",
                wraps=run_job.compile_execution_motion_program,
            ) as compile_variant,
            mock.patch.object(
                run_job, "_load_approach_sampling_profile",
                return_value=approach_profile,
            ),
        ):
            resolved, compiled_program, _scene = run_job.resolve_inputs(
                value, scene_binding_call=lambda *_args: {},
            )
            binding = run_job._trajectory_binding(
                value, resolved, compiled_program,
            )
        compile_variant.assert_called_once_with(
            source_program,
            trajectory_variant_id="TWO_STAGE_ALIGN_V2",
            sampling_seed=23,
            target_yaw_deg=45.0,
            object_dimensions_mm=[24.0, 24.0, 24.0],
            approach_sampling_profile=approach_profile,
            sample_rank=2, design_size=7,
            design_digest=run_job.canonical_digest("design"),
        )
        self.assertEqual(
            (binding["trajectory_variant_id"], binding["sampling_seed"]),
            ("TWO_STAGE_ALIGN_V2", 23),
        )
        self.assertEqual(
            (binding["sample_rank"], binding["design_size"]), (2, 7),
        )
        self.assertEqual(
            run_job._validated_trajectory_binding(
                binding, payload=value, validated=validated,
                motion_program_digest=run_job.canonical_digest(compiled_program),
            ),
            binding,
        )
        forged = copy.deepcopy(binding)
        forged["sampling_seed"] += 1
        with self.assertRaisesRegex(
            run_job.ContractError, "TRAJECTORY_BINDING",
        ):
            run_job._validated_trajectory_binding(
                forged, payload=value, validated=validated,
                motion_program_digest=run_job.canonical_digest(compiled_program),
            )

    def test_postcommit_reposition_reuses_executor_without_recorder(self):
        repository = Path(__file__).resolve().parents[2]
        object_profile = run_job.load_json_strict(
            repository / "config/data_factory/objects/wood-cube-24mm-r001.json",
        )
        grasp_profile = run_job.load_json_strict(
            repository / "config/data_factory/grasps/"
            "wood-cube-24mm-top-3p5mm-r001.json",
        )
        yaw_profile = validate_yaw_sampling_profile(
            run_job.load_json_strict(
                repository / "config/data_factory/yaw_sampling_profiles/"
                "wood-cube-24mm-top-r001.json",
            ),
            object_profile=object_profile, grasp_profile=grasp_profile,
        )
        yaw_binding = sample_yaw_cdf_strata(
            yaw_profile, sampling_seed=9,
            sweep_identity={"slot": "next"}, strata_count=1,
        )[0]
        target_pose = {
            "place_id": "PLACE_A", "yaw_deg": 0.0,
            "x_mm": 4.0, "y_mm": -6.0,
        }
        target_pose = {
            **target_pose,
            "yaw_deg": yaw_binding["source_object_yaw_deg"],
        }
        source_pose = yaw_preserving_destination(
            {**target_pose, "yaw_deg": 0.0}, target_pose,
        )
        binding = build_object_reposition_binding(
            parent_run_id="episode-run",
            continuation_run_id="episode-run-reposition",
            next_run_id="episode-run-2", start_state="ON_SURFACE",
            source_pose=source_pose, target_pose=target_pose,
            object_profile=object_profile, grasp_profile=grasp_profile,
            yaw_sampling_profile=yaw_profile,
            yaw_sample_binding=yaw_binding,
        )
        parent_plan_digest = "sha256:" + "7" * 64
        source_slot = {
            "slot_id": "sha256:" + "1" * 64,
            "slot_digest": "sha256:" + "2" * 64,
            "allowed_run_id": binding["continuation_run_id"],
        }
        target_slot = release_slot(
            robot_system_id="fr5-lab-a", pose=target_pose,
            object_profile_id=object_profile["object_profile_id"],
            exclusion_geometry_digest="sha256:" + "e" * 64,
            role="DESTINATION_THEN_NEXT_SOURCE",
        )
        scene_binding = {
            **SCENE, "source_slot": source_slot,
            "release_slot": target_slot,
            "allowed_next_run_id": binding["next_run_id"],
        }
        program_template = motion(True)
        input_digests = {
            key: program_template["binding_digests"][key]
            for key in (
                "selected_sheet", "yaw0_sheet", "cell_calibration",
                "robot_system", "object_profile", "grasp_profile",
            )
        }
        family_digest = "sha256:" + "d" * 64
        validated = runtime_validated(job={
            **JOB, "job_id": binding["continuation_run_id"],
            "task": "pick_place", "place_id": source_pose["place_id"],
            "cell_calibration_id": "cell-a",
            "sheet_manifest_digest": input_digests["selected_sheet"],
            "yaw_deg": source_pose["yaw_deg"],
            "x_mm": source_pose["x_mm"], "y_mm": source_pose["y_mm"],
            "object_profile_id": object_profile["object_profile_id"],
            "grasp_profile_id": grasp_profile["grasp_profile_id"],
        }, input_digests=input_digests,
            object_profile=object_profile, grasp_profile=grasp_profile,
            calibration={"document": {"a4_family_digest": family_digest}},
        )
        program = runtime_motion(validated, continuous=True)

        class Transport(T):
            def __init__(self):
                super().__init__()
                self.position = [0.0] * 6
                self.gripper_position = 0.01
                self.started = []

            def snapshot(self, *_):
                return snapshot(
                    self.position, gripper_position=self.gripper_position,
                )

            def start_phase(self, step):
                self.started.append(step["phase"])
                self.position = list(step["final_joint_state"])
                if step["type"] == "GRIPPER":
                    self.gripper_position = step["gripper_position_m"]

            def poll_active(self):
                return object()

            def cancel_active(self, *_):
                return None

        class Store:
            def __init__(self):
                self.consumed = []
                self.transitions = []
                self.blocked = []

            def read(self):
                return {
                    "robot_system_id": "fr5-lab-a", "cell_ready": False,
                    "run_id": binding["parent_run_id"],
                    "plan_digest": parent_plan_digest,
                }

            def consume_next_source(self, **value):
                self.consumed.append(value)
                self.consumed_slot = {
                    "state": "CONSUMED_PENDING_REVIEW",
                    "allowed_run_id": value["run_id"],
                }
                return {
                    "scene_state_digest": "sha256:" + "9" * 64,
                    "scene_state": {
                        "revision": 2,
                        "slot_allocations": {value["slot_id"]: self.consumed_slot},
                    },
                }

            def locked_snapshot(self, digest):
                class Locked:
                    def __enter__(_self):
                        return {
                            "scene_state_digest": digest,
                            "scene_state": {
                                "revision": 2,
                                "objects": {"cube-1": {
                                    "object_profile_id": object_profile[
                                        "object_profile_id"
                                    ],
                                    "state": "ON_SURFACE",
                                }},
                            },
                        }

                    def __exit__(_self, *_):
                        return None

                return Locked()

            def transition_release(self, **value):
                self.transitions.append(value)
                return {
                    "scene_state_digest": "sha256:" + "8" * 64,
                    "release_evidence_digest": run_job.canonical_digest(
                        value["evidence"],
                    ),
                }

            def mark_blocked(self, *value):
                self.blocked.append(value)

        published = []
        with tempfile.TemporaryDirectory() as directory:
            parent_payload = payload("live")
            parent_payload["job"] = {
                **copy.deepcopy(parent_payload["job"]),
                "task": "pick_place", "cell_calibration_id": "cell-a",
                "object_profile_id": object_profile["object_profile_id"],
                "grasp_profile_id": grasp_profile["grasp_profile_id"],
                "operator_or_agent_id": "operator",
            }
            parent_payload.update(
                run_id=binding["parent_run_id"], run_root=directory,
                config_root=str(repository / "config/data_factory"),
            )
            run_job._prepare_run_dir(parent_payload)
            motion_payload = run_job._object_reposition_payload(
                parent_payload, binding,
                source_payload=copy.deepcopy(parent_payload),
            )
            current_slot = {
                "slot_id": "current-slot",
                "base_condition_digest": "sha256:" + "d" * 64,
                "robot_start_pose_id": "start-a", "order_index": 0,
            }
            next_slot = {
                "slot_id": "next-slot",
                "base_condition_digest": "sha256:" + "e" * 64,
                "robot_start_pose_id": "start-a", "order_index": 1,
            }
            manifest = {"slots": [current_slot, next_slot]}
            manifest["manifest_digest"] = run_job.canonical_digest(manifest)
            endpoint_bindings = [
                {
                    "workspace_id": binding["source_pose"]["place_id"],
                    "cell_calibration_id": "cell-a",
                    "cell_calibration_digest": input_digests[
                        "cell_calibration"
                    ],
                    "motion_recipe_digest": program["binding_digests"][
                        "motion_qualification"
                    ],
                },
                {
                    "workspace_id": "PLACE_B",
                    "cell_calibration_id": "cell-b",
                    "cell_calibration_digest": "sha256:" + "5" * 64,
                    "motion_recipe_digest": "sha256:" + "6" * 64,
                },
            ]
            endpoint_bindings.sort(
                key=lambda item: (
                    item["workspace_id"], item["cell_calibration_id"],
                ),
            )
            fixed_contract = {
                "schema_version": "data_factory.fr5_fixed_contract.v2",
                "endpoint_bindings": endpoint_bindings,
                "endpoint_bindings_digest": run_job.canonical_digest(
                    endpoint_bindings,
                ),
            }
            fixed_contract["motion_recipe_digest"] = fixed_contract[
                "endpoint_bindings_digest"
            ]
            intent = {
                "run_id": binding["parent_run_id"], "order_index": 0,
                "slot": current_slot,
                "slot_digest": run_job.canonical_digest(current_slot),
                "manifest_digest": manifest["manifest_digest"],
                "fixed_contract": fixed_contract,
            }
            intent["intent_digest"] = run_job.canonical_digest(intent)
            ledger_context = {"manifest": manifest, "intent": intent}
            episode_binding = {
                "run_id": binding["parent_run_id"],
                "manifest_digest": manifest["manifest_digest"],
                "intent_digest": intent["intent_digest"],
                "slot_digest": run_job.canonical_digest(current_slot),
                "binding_digest": run_job.canonical_digest(
                    "runtime-episode-binding",
                ),
            }
            region = {
                "layout_id": None, "layout_digest": None,
                "region_id": None,
                "physical_binding_status": "NOT_CONFIGURED",
            }
            destination = {
                "role": "DESTINATION",
                "workspace_id": binding["source_pose"]["place_id"],
                "frame_id": "cell-a",
                "pose": copy.deepcopy(binding["source_pose"]),
                "sheet_digest": input_digests["selected_sheet"],
                "family_digest": family_digest,
                "region_binding": region,
            }
            source = {
                **copy.deepcopy(destination), "role": "SOURCE",
                "pose": {
                    **copy.deepcopy(destination["pose"]),
                    "x_mm": destination["pose"]["x_mm"] + 20.0,
                },
            }
            episode_instruction = compile_episode_instruction_binding(
                compile_task_binding(
                    "pick_place", source=source, destination=destination,
                ),
                object_profile,
            )
            envelope = {
                "manifest_digest": manifest["manifest_digest"],
                "fixed_contract_digest": run_job.canonical_digest(
                    fixed_contract,
                ),
                "episode_count": 2,
                "slot_digests": [
                    run_job.canonical_digest(current_slot),
                    run_job.canonical_digest(next_slot),
                ],
                "effect_scope": "PHYSICAL",
                "lifecycle_action": "LIVE_COLLECT",
                "task": "pick_place", "robot_system_id": "fr5-lab-a",
                "object_profile_id": object_profile["object_profile_id"],
                "grasp_profile_id": grasp_profile["grasp_profile_id"],
                "collection_profile_digest": validated["input_digests"][
                    "collection_profile"
                ],
                "motion_qualification_digest": fixed_contract[
                    "motion_recipe_digest"
                ],
            }
            envelope["envelope_digest"] = run_job.canonical_digest(envelope)
            authorization = {
                "expires_at": "2099-01-01T00:00:00Z",
                "envelope_digest": envelope["envelope_digest"],
                "envelope": envelope,
            }
            authorization["authorization_digest"] = run_job.canonical_digest(
                authorization,
            )
            preapproval_evidence = {
                "plan_digest": parent_plan_digest,
                "object_reposition_binding_digest": binding["binding_digest"],
            }
            with mock.patch.object(
                run_job, "validate_campaign_authorization",
                return_value=authorization,
            ):
                preapproval_scope = (
                    run_job._write_object_reposition_preapproval(
                        parent_payload, binding,
                        parent_plan_digest=parent_plan_digest,
                        parent_preapproval_evidence=preapproval_evidence,
                        campaign_authorization={"fixture": True},
                        ledger_context=ledger_context,
                        episode_binding=episode_binding,
                        episode_instruction_binding=episode_instruction,
                        source_payload=copy.deepcopy(parent_payload),
                        resolver=lambda _value, **_: (
                            validated, program, {}
                        ),
                    )
                )
            self.assertEqual(
                (
                    preapproval_scope["campaign_authorization_digest"],
                    preapproval_scope["manifest_digest"],
                    preapproval_scope["intent_digest"],
                    preapproval_scope["next_slot_digest"],
                    preapproval_scope["next_slot_endpoint"]["target_pose"],
                ),
                (
                    authorization["authorization_digest"],
                    manifest["manifest_digest"], intent["intent_digest"],
                    run_job.canonical_digest(next_slot),
                    binding["target_pose"],
                ),
            )
            expectation = run_job._object_reposition_continuation_expectation(
                preapproval_scope, authorization=authorization,
                parent_plan_digest=parent_plan_digest, binding=binding,
                motion_payload=motion_payload, validated=validated,
                program=program, scene_binding=scene_binding,
            )
            forged_scope = copy.deepcopy(preapproval_scope)
            forged_scope["next_slot_endpoint"][
                "cell_calibration_digest"
            ] = "sha256:" + "f" * 64
            forged_scope["next_slot_endpoint_digest"] = (
                run_job.canonical_digest(forged_scope["next_slot_endpoint"])
            )
            forged_scope["scope_digest"] = run_job.canonical_digest({
                key: value for key, value in forged_scope.items()
                if key != "scope_digest"
            })
            with self.assertRaisesRegex(
                run_job.ContractError,
                "OBJECT_REPOSITION_CONTINUATION_BINDING",
            ):
                run_job._object_reposition_continuation_expectation(
                    forged_scope, authorization=authorization,
                    parent_plan_digest=parent_plan_digest, binding=binding,
                    motion_payload=motion_payload, validated=validated,
                    program=program, scene_binding=scene_binding,
                )
            store = Store()
            transport = Transport()
            node = PickupExecutor(
                transport, execution_enabled=True,
                cell_state_store=store, scene_state_store=store,
                clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
                motion_only_binding_digest=binding["binding_digest"],
                motion_only_parent_run_id=binding["parent_run_id"],
                motion_only_parent_plan_digest=parent_plan_digest,
                motion_only_preapproval_scope_digest=preapproval_scope[
                    "scope_digest"
                ],
                motion_only_expected_run_id=expectation["run_id"],
                motion_only_expected_resolved_job_digest=expectation[
                    "resolved_job_digest"
                ],
                motion_only_expected_program_digest=expectation[
                    "motion_program_digest"
                ],
                motion_only_expected_scene_digest=expectation[
                    "scene_binding_digest"
                ],
                motion_only_expectation_digest=expectation[
                    "expectation_digest"
                ],
            )

            class Process:
                def request(self, request, *_):
                    return node.process(request)

                def close(self, **_):
                    return None

            with (
                mock.patch.object(
                    run_job, "validate_campaign_authorization",
                    return_value=authorization,
                ),
                mock.patch.object(
                    run_job, "resolve_object_reposition_inputs",
                    return_value=(
                        motion_payload, validated, program,
                        scene_binding, binding,
                    ),
                ),
                mock.patch.object(run_job, "CellStateStore", return_value=store),
            ):
                result = run_job.run_object_reposition(
                    parent_payload, binding, threading.Event(), published.append,
                    parent_plan_digest=parent_plan_digest,
                    operator_id="operator", cell_root=Path(directory) / "cells",
                    executor_factory=lambda *_, **__: Process(),
                    campaign_authorization={"fixture": True},
                    data_disposition="TEST_ONLY",
                    preapproval_scope=preapproval_scope,
                    source_payload=copy.deepcopy(parent_payload),
                    clock=lambda: datetime(2026, 9, 3, tzinfo=timezone.utc),
                )
            plan_file = Path(directory) / "episode-run/object_reposition_plan.json"
            result_file = Path(directory) / "episode-run/object_reposition_result.json"
            self.assertTrue(plan_file.is_file())
            self.assertTrue(
                (Path(directory) / binding["continuation_run_id"]).is_dir()
            )
            self.assertEqual(
                json.loads(result_file.read_text())["status"], "PASS", result,
            )
            self.assertEqual(
                run_job._validate_object_reposition_result(
                    parent_payload, binding, preapproval_scope, result,
                ),
                result,
            )
            plan_artifact = json.loads(plan_file.read_text())
            self.assertEqual(
                (
                    plan_artifact["preapproval_scope_digest"],
                    plan_artifact["continuation_expectation_digest"],
                    plan_artifact["resolved_job_digest"],
                    plan_artifact["motion_program_digest"],
                    plan_artifact["scene_binding_digest"],
                    plan_artifact["plan_digest"],
                ),
                (
                    preapproval_scope["scope_digest"],
                    expectation["expectation_digest"],
                    validated["resolved_job_digest"],
                    run_job.canonical_digest(program),
                    run_job.canonical_digest(scene_binding),
                    result["plan_digest"],
                ),
            )
            forged_result = copy.deepcopy(result)
            forged_result.update(
                status="FAIL", code="EXECUTION_FAILED",
                scene_state_digest=None,
                execution_response={
                    "ok": False, "code": "EXECUTION_FAILED",
                    "state": "BLOCKED", "run_id": "other-run",
                    "plan_digest": result["plan_digest"], "data": None,
                },
            )
            forged_result["result_digest"] = run_job.canonical_digest({
                key: value for key, value in forged_result.items()
                if key != "result_digest"
            })
            run_job.write_json_atomic(result_file, forged_result)
            with self.assertRaisesRegex(
                run_job.ContractError, "OBJECT_REPOSITION_RESULT",
            ):
                run_job._validate_object_reposition_result(
                    parent_payload, binding, preapproval_scope, forged_result,
                )

        self.assertEqual((result["status"], result["code"]), ("PASS", "PASS"))
        self.assertEqual(
            (store.consumed[0]["run_id"], len(store.transitions), store.blocked),
            (binding["continuation_run_id"], 1, []),
        )
        self.assertEqual(store.transitions[0]["parent_cell_binding"], {
            "run_id": binding["parent_run_id"],
            "plan_digest": parent_plan_digest,
            "source_slot_id": source_slot["slot_id"],
            "source_slot_digest": run_job.canonical_digest(store.consumed_slot),
        })
        self.assertEqual(transport.started, list(PHASES))
        self.assertEqual(
            [event["code"] for event in published],
            ["OBJECT_REPOSITION_PLANNED", "OBJECT_REPOSITION_EXECUTING"],
        )
        self.assertEqual(
            published[0]["data"]["object_reposition_plan_digest"],
            result["plan_digest"],
        )
        self.assertEqual(
            (
                published[0]["data"][
                    "object_reposition_plan_artifact_digest"
                ],
                published[0]["data"][
                    "object_reposition_collision_report_digest"
                ],
                published[0]["data"][
                    "object_reposition_plan_only_no_motion_digest"
                ],
            ),
            (
                plan_artifact["artifact_digest"],
                plan_artifact["plan_envelope"]["precommit_safety"][
                    "collision_report_digest"
                ],
                plan_artifact["plan_envelope"]["precommit_safety"][
                    "plan_only_no_motion_digest"
                ],
            ),
        )

    def test_only_fresh_authorized_production_cell_is_initialized(self):
        missing = {
            "cell_ready": False, "reason_code": "STATE_MISSING",
        }
        ready = {
            "cell_ready": True, "reason_code": "HUMAN_ACKNOWLEDGED",
        }
        store = SimpleNamespace(
            read=lambda: missing,
            acknowledge_ready=mock.Mock(return_value=ready),
        )
        self.assertEqual(
            run_job._read_live_cell_state(
                store, data_disposition="PRODUCTION",
                campaign_authorization={"validated": True},
                operator_id="local-operator",
                run_id="new-run", scene_binding=None, scene_store=None,
            ),
            ready,
        )
        store.acknowledge_ready.assert_called_once_with("local-operator")

        for disposition, authorization, reason in (
            ("PRODUCTION", None, "STATE_MISSING"),
            ("TEST_ONLY", {"validated": True}, "STATE_MISSING"),
            ("PRODUCTION", {"validated": True}, "EXECUTION_IN_PROGRESS"),
        ):
            with self.subTest(
                disposition=disposition, authorization=authorization,
                reason=reason,
            ):
                blocked = {"cell_ready": False, "reason_code": reason}
                unchanged = SimpleNamespace(
                    read=lambda: blocked,
                    acknowledge_ready=mock.Mock(),
                )
                untouched_scene = SimpleNamespace(snapshot=mock.Mock())
                self.assertEqual(
                    run_job._read_live_cell_state(
                        unchanged, data_disposition=disposition,
                        campaign_authorization=authorization,
                        operator_id="local-operator",
                        run_id="new-run", scene_binding=None,
                        scene_store=untouched_scene,
                    ),
                    blocked,
                )
                unchanged.acknowledge_ready.assert_not_called()
                untouched_scene.snapshot.assert_not_called()

    def test_fresh_human_scene_supersedes_only_an_older_scene_slot_fault(self):
        scene = {
            "revision": 11,
            "updated_at": "2026-09-02T06:12:14.026093Z",
            "objects": {
                "cube-1": {
                    "state": "ON_SURFACE", "source": "HUMAN",
                    "updated_at": "2026-09-02T06:12:14.026093Z",
                },
            },
            "slot_allocations": {},
        }
        scene_digest = run_job.canonical_digest(scene)
        scene_snapshot = {
            "scene_state": scene, "scene_state_digest": scene_digest,
        }
        scene_binding = {
            "scene_state_digest": scene_digest,
            "revision": 11,
            "object_instance_id": "cube-1",
        }
        blocked = {
            "cell_ready": False, "reason_code": "SCENE_SLOT_UNAVAILABLE",
            "run_id": "old-run", "plan_digest": "sha256:" + "a" * 64,
            "updated_at": "2026-09-02T05:58:29.854942Z",
        }
        ready = {**blocked, "cell_ready": True, "reason_code": "HUMAN_ACKNOWLEDGED"}
        store = SimpleNamespace(
            read=lambda: blocked,
            acknowledge_ready=mock.Mock(return_value=ready),
        )

        self.assertEqual(
            run_job._read_live_cell_state(
                store, data_disposition="PRODUCTION",
                campaign_authorization={"validated": True},
                operator_id="local-operator", run_id="new-run",
                scene_binding=scene_binding,
                scene_store=SimpleNamespace(snapshot=lambda: scene_snapshot),
            ),
            ready,
        )
        store.acknowledge_ready.assert_called_once_with(
            "local-operator", expected_run_id="old-run",
            expected_plan_digest="sha256:" + "a" * 64,
        )

        for label, changes in (
            ("physical fault", {"cell": {**blocked, "reason_code": "PRECONTACT_TIMEOUT"}}),
            ("same run", {"run_id": "old-run"}),
            ("robot scene", {"instance": {"source": "ROBOT_RELEASE"}}),
            ("occupied slot", {"scene": {"slot_allocations": {"slot": {}}}}),
            ("stale scene", {"scene": {"updated_at": "2026-09-02T05:00:00Z"}, "instance": {"updated_at": "2026-09-02T05:00:00Z"}}),
            ("digest mismatch", {"binding": {"scene_state_digest": "sha256:" + "b" * 64}}),
        ):
            with self.subTest(label=label):
                candidate_cell = changes.get("cell", blocked)
                candidate_scene = copy.deepcopy(scene)
                candidate_scene.update(changes.get("scene", {}))
                candidate_scene["objects"]["cube-1"].update(
                    changes.get("instance", {}),
                )
                candidate_snapshot = {
                    "scene_state": candidate_scene,
                    "scene_state_digest": scene_digest,
                }
                candidate_binding = {
                    **scene_binding, **changes.get("binding", {}),
                }
                unchanged = SimpleNamespace(
                    read=lambda value=candidate_cell: value,
                    acknowledge_ready=mock.Mock(),
                )
                self.assertEqual(
                    run_job._read_live_cell_state(
                        unchanged, data_disposition="PRODUCTION",
                        campaign_authorization={"validated": True},
                        operator_id="local-operator",
                        run_id=changes.get("run_id", "new-run"),
                        scene_binding=candidate_binding,
                        scene_store=SimpleNamespace(
                            snapshot=lambda value=candidate_snapshot: value,
                        ),
                    ),
                    candidate_cell,
                )
                unchanged.acknowledge_ready.assert_not_called()

    def test_episode_instruction_scope_binds_source_destination_and_object(self):
        family_digest = run_job.canonical_digest("a4-family")
        object_profile = {
            "schema_version": "data_factory.object_profile.v2",
            "object_profile_id": "wood-cube-24mm-r001",
            "qualification_status": "QUALIFIED",
            "description": "24 mm wooden cube",
            "dimensions_mm": [24, 24, 24],
            "datum": "CENTER",
        }
        job = {
            "task": "pick_place", "robot_system_id": "fr5-lab-a",
            "operator_or_agent_id": "operator",
            "instruction": "pick up the 24 mm wooden cube and place it at the destination",
            "place_id": "PLACE_A", "cell_calibration_id": "place-a-yaw0-r003",
            "sheet_manifest_digest": run_job.canonical_digest("sheet-a"),
            "yaw_deg": 0.0, "x_mm": 10.0, "y_mm": -5.0,
            "object_profile_id": object_profile["object_profile_id"],
        }
        validated = runtime_validated(
            job=job, object_profile=object_profile,
            calibration={"document": {"a4_family_digest": family_digest}},
            input_digests={"object_profile": run_job.canonical_digest(object_profile)},
        )

        def endpoint(role, place_id, frame_id, x_mm, y_mm, region_id):
            return {
                "role": role, "workspace_id": place_id, "frame_id": frame_id,
                "pose": {
                    "place_id": place_id, "yaw_deg": 0.0,
                    "x_mm": x_mm, "y_mm": y_mm,
                },
                "sheet_digest": (
                    job["sheet_manifest_digest"]
                    if role == "SOURCE" else run_job.canonical_digest("sheet-b")
                ),
                "family_digest": family_digest,
                "region_binding": {
                    "layout_id": "layout-r1",
                    "layout_digest": run_job.canonical_digest("layout-r1"),
                    "region_id": region_id,
                    "physical_binding_status": "PREPARED_NOT_VERIFIED",
                },
            }

        task_binding = compile_task_binding(
            "pick_place",
            source=endpoint(
                "SOURCE", "PLACE_A", "place-a-yaw0-r003", 10.0, -5.0, "RED",
            ),
            destination=endpoint(
                "DESTINATION", "PLACE_B", "place-b-yaw0-r001", -20.0, 15.0,
                "BLUE",
            ),
        )
        instruction_binding = compile_episode_instruction_binding(
            task_binding, object_profile,
        )
        checklist = {
            "task_binding": task_binding,
            "episode_instruction_binding": instruction_binding,
        }
        scene = {
            **SCENE,
            "release_slot": {
                "pose": {
                    "place_id": "PLACE_B", "yaw_deg": 0.0,
                    "x_mm": -20.0, "y_mm": 15.0,
                },
            },
        }
        self.assertEqual(
            run_job._validate_episode_instruction_scope(
                instruction_binding, validated=validated,
                scene_binding=scene, preapproval_checklist=checklist,
                repository_root=Path(__file__).resolve().parents[2],
            ),
            instruction_binding,
        )

        # Scoped learned wording is descriptive lineage, not a synthetic
        # Collection campaign/checklist. Its exact grant and destination still bind.
        from tools.data_factory.rollout.finite_plan import SCOPED_SCHEMAS
        destination = task_binding["spatial_bindings"][1]
        scoped_validated = copy.deepcopy(validated)
        scoped_validated["destination_resolved_inputs"] = {"normalized_job": {
            **destination["pose"], "cell_calibration_id": destination["frame_id"],
            "sheet_manifest_digest": destination["sheet_digest"],
        }}
        grant = {"schema_version": "data_factory.learned_task_grant.v1", "grant_id": "g",
                 "issued_by": "operator", "run_id": "run", "deadline_s": 100.,
                 "terminal_reserve_s": 10., "max_outputs": 2, "revoked": False,
                 "scope": {"task": instruction_binding["instruction"],
                           "adaptation": {"schema_version": sorted(SCOPED_SCHEMAS)[0]}}}
        grant["grant_digest"] = run_job.canonical_digest(grant)
        def check_scoped(value=scoped_validated, authority=grant):
            return run_job._validate_episode_instruction_scope(
                instruction_binding, validated=value, scene_binding=scene,
                preapproval_checklist=None, repository_root=Path(__file__).resolve().parents[2],
                task_grant=authority)
        self.assertEqual(check_scoped(), instruction_binding)
        from tools.data_factory.rollout.stream_plan import POLICY_SCHEMA
        normal_grant = copy.deepcopy(grant)
        normal_grant["scope"]["adaptation"]["schema_version"] = POLICY_SCHEMA
        normal_grant["grant_digest"] = run_job.canonical_digest({
            key: value for key, value in normal_grant.items()
            if key != "grant_digest"
        })
        self.assertEqual(
            check_scoped(authority=normal_grant), instruction_binding,
        )
        for key in ("place_id", "cell_calibration_id", "sheet_manifest_digest", "x_mm"):
            with self.subTest(scoped_destination=key):
                changed = copy.deepcopy(scoped_validated)
                changed["destination_resolved_inputs"]["normalized_job"][key] = "wrong"
                with self.assertRaisesRegex(run_job.ContractError, "EPISODE_INSTRUCTION_SCOPE"):
                    check_scoped(changed)
        changed_grant = copy.deepcopy(grant)
        changed_grant["scope"]["task"] = "another instruction"
        changed_grant["grant_digest"] = run_job.canonical_digest({
            key: value for key, value in changed_grant.items() if key != "grant_digest"})
        with self.assertRaisesRegex(run_job.ContractError, "EPISODE_INSTRUCTION_SCOPE"):
            check_scoped(authority=changed_grant)
        with self.assertRaisesRegex(run_job.ContractError, "EPISODE_INSTRUCTION_SCOPE"):
            check_scoped(authority=None)

        for label, mutate in (
            (
                "source",
                lambda value: value["task_binding"]["spatial_bindings"][0]["pose"].update(x_mm=11.0),
            ),
            (
                "destination",
                lambda value: value["task_binding"]["spatial_bindings"][1]["pose"].update(y_mm=16.0),
            ),
            (
                "instruction",
                lambda value: value.update(instruction="move it somewhere"),
            ),
        ):
            with self.subTest(label=label):
                tampered = copy.deepcopy(instruction_binding)
                mutate(tampered)
                tampered["binding_digest"] = run_job.canonical_digest({
                    key: value for key, value in tampered.items()
                    if key != "binding_digest"
                })
                with self.assertRaises(run_job.ContractError):
                    run_job._validate_episode_instruction_scope(
                        tampered, validated=validated,
                        scene_binding=scene,
                        preapproval_checklist={
                            "task_binding": tampered["task_binding"],
                            "episode_instruction_binding": tampered,
                        },
                        repository_root=Path(__file__).resolve().parents[2],
                    )

    def test_test_only_terminal_projection_binds_readiness_and_keeps_semantics_separate(self):
        profile_digest = "sha256:" + "7" * 64
        readiness = {
            "schema_version": "data_factory.recorder_readiness_evidence.v1",
            "run_id": "runner-test",
            "transaction_id": "tx-r001",
            "episode_index": 0,
            "collection_profile_digest": profile_digest,
            "quality_contract_digest": run_job.canonical_digest(
                run_job.TEST_ONLY_READINESS_CONTRACT
            ),
            "observed_monotonic_ns": 1,
            "metrics": {"quality_accepted": True},
        }
        projected = run_job._test_only_terminal_projection(
            readiness,
            run_id="runner-test",
            collection_profile_digest=profile_digest,
            approval_scope="HIL_NUMERIC_PROXY",
            decision_source="LOCAL_UI_BUTTON",
            mechanical_proxy="MECHANICAL_GRASP_PROXY_PASS",
            human_semantic_outcome="NOT_MEASURED",
        )
        self.assertEqual(projected["recorder_readiness_digest"], run_job.canonical_digest(readiness))
        self.assertEqual(projected["human_semantic_outcome"], "NOT_MEASURED")
        self.assertFalse(projected["candidate_admission_written"])

        for field in ("collection_profile_digest", "quality_contract_digest"):
            with self.subTest(field=field):
                mismatched = dict(readiness)
                mismatched[field] = "sha256:" + "0" * 64
                with self.assertRaisesRegex(run_job.ContractError, "TEST_ONLY_READINESS_EVIDENCE"):
                    run_job._test_only_terminal_projection(
                        mismatched,
                        run_id="runner-test",
                        collection_profile_digest=profile_digest,
                        approval_scope="HIL_NUMERIC_PROXY",
                        decision_source="LOCAL_UI_BUTTON",
                        mechanical_proxy="MECHANICAL_GRASP_PROXY_PASS",
                        human_semantic_outcome="NOT_MEASURED",
                    )

        with self.assertRaisesRegex(run_job.ContractError, "TEST_ONLY_PROXY_EVIDENCE"):
            run_job._test_only_terminal_projection(
                readiness,
                run_id="runner-test",
                collection_profile_digest=profile_digest,
                approval_scope="HIL_NUMERIC_PROXY",
                decision_source="LOCAL_UI_BUTTON",
                mechanical_proxy=None,
                human_semantic_outcome="NOT_MEASURED",
            )
        with self.assertRaisesRegex(run_job.ContractError, "TEST_ONLY_HUMAN_SEMANTIC_EVIDENCE"):
            run_job._test_only_terminal_projection(
                readiness,
                run_id="runner-test",
                collection_profile_digest=profile_digest,
                approval_scope="HUMAN_GATED",
                decision_source="LOCAL_UI_BUTTON",
                mechanical_proxy=None,
                human_semantic_outcome="NOT_MEASURED",
            )

    def test_test_only_button_gate_binds_exact_plan_before_recorder_or_execute(self):
        validated = runtime_validated(job={
            **JOB, "operator_or_agent_id": "operator", "instruction": "pick up",
        })

        class Cell:
            def read(self):
                return {"robot_system_id": "fr5-lab-a", "cell_ready": True}

        class PlannedExecutor(Executor):
            def __init__(self, events):
                super().__init__()
                self.ops = []
                self.events = events

            def request(self, request, cancel=None):
                self.ops.append(request["op"])
                self.events.append(f"executor:{request['op']}")
                return super().request(request, cancel)

        def run(
            choice, *, stale=False, expired=False, start_mismatch=False,
            with_site=False, site_choice="READY", warmup_error=False,
        ):
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            root = Path(directory.name)
            live_payload = payload("live")
            live_payload.update(
                run_root=str(root / "runs"),
                dataset_root=str(root / "dataset"),
            )
            roots = {
                "session_id": "session-r001", "run_id": live_payload["run_id"],
                "data_disposition": "TEST_ONLY",
                "run_root": str((root / "runs").resolve()),
                "cell_root": str((root / "cells").resolve()),
                "dataset_root": str((root / "dataset").resolve()),
                "production_writers_enabled": False,
            }
            roots["binding_digest"] = run_job.canonical_digest(roots)
            episode_binding = {
                "binding_digest": run_job.canonical_digest("episode-binding"),
                "start_binding_digest": run_job.canonical_digest("start-binding"),
                "expires_at": "2000-01-01T00:00:00Z" if expired else "2099-01-01T00:00:00Z",
            }
            planned_start = {
                "start_binding_digest": episode_binding["start_binding_digest"],
                "evidence_digest": run_job.canonical_digest("planned-start"),
                "status": "PASS",
            }
            observed = []
            site_requests = []
            events = []

            def decide(request):
                events.append("decision")
                observed.append(request)
                if choice is None:
                    return None
                bound = {
                    "run_id": request["run_id"],
                    "plan_digest": request["plan_digest"],
                    "approval_scope": request["approval_scope"],
                    "decision_binding": request["decision_binding"],
                }
                return {
                    "choice": choice,
                    "run_id": request["run_id"],
                    "plan_digest": request["plan_digest"],
                    "approval_scope": request["approval_scope"],
                    "decision_binding_digest": (
                        "sha256:" + "0" * 64 if stale
                        else run_job.canonical_digest(bound)
                    ),
                    "decision_source": "LOCAL_UI_BUTTON",
                    "operator_label": "operator",
                }

            def checkpoint(request):
                events.append(f"checkpoint:{request['kind']}")
                site_requests.append(request)
                if site_choice is None:
                    return None
                bound = {
                    key: request[key]
                    for key in (
                        "kind", "run_id", "plan_digest", "prompt", "choices", "evidence",
                    )
                }
                return {
                    "kind": request["kind"], "choice": site_choice,
                    "run_id": request["run_id"], "plan_digest": request["plan_digest"],
                    "checkpoint_binding_digest": run_job.canonical_digest(bound),
                    "decision_source": "LOCAL_UI_BUTTON", "operator_label": "operator",
                }

            def warmup(*_):
                events.append("camera_warmup")
                if warmup_error:
                    raise run_job.ContractError("CAMERA_WARMUP_RATE")
                return {
                    "schema_version": "data_factory.camera_warmup.v1", "attempts": [],
                }

            executor = PlannedExecutor(events)
            recorder = mock.Mock()
            with (
                mock.patch.object(
                    run_job, "validate_test_only_root_binding", return_value=roots,
                ) as root_validator,
                mock.patch.object(
                    run_job, "validate_test_only_episode_binding",
                    return_value=episode_binding,
                ),
                mock.patch.object(
                    run_job, "validate_test_only_planned_start",
                    side_effect=(
                        run_job.ContractError("TEST_ONLY_PLANNED_START_MISMATCH")
                        if start_mismatch else None
                    ),
                    return_value=planned_start,
                ),
                mock.patch.object(run_job, "CellStateStore", return_value=Cell()),
                mock.patch.object(run_job, "SceneStateStore"),
            ):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, runtime_motion(validated), SCENE),
                    executor_factory=lambda *_: executor,
                    recorder_factory=recorder,
                    camera_warmup_call=warmup,
                    decision_provider=decide,
                    checkpoint_provider=checkpoint if with_site else None,
                    decision_timeout_s=0,
                    test_only_root_binding={"fixture": True},
                    test_only_episode_binding={"fixture": True},
                    test_only_start_binding={"fixture": True},
                    candidate_writer_enabled=False,
                    preapproval_checklist=(
                        {
                            "place_alias": "place1", "place_id": "PLACE_A",
                            "yaw_deg": 0, "x_mm": 0, "y_mm": 0,
                            "cube_at_target": "OPERATOR_CONFIRM_REQUIRED",
                            "gripper_empty": "OPERATOR_CONFIRM_REQUIRED",
                            "cell_clear": "OPERATOR_CONFIRM_REQUIRED",
                            "estop_monitored": "OPERATOR_CONFIRM_REQUIRED",
                        }
                        if with_site else None
                    ),
                    repository_root=root,
                )
            root_validator.assert_called_once_with(
                {"fixture": True}, repository_root=root,
            )
            return result, observed, executor.ops, recorder, events, site_requests

        for choice, code, state in (
            (None, "PAUSED_AWAITING_OPERATOR", "PLANNED"),
            ("REJECT", "PLAN_REJECTED", "CANCELLED"),
            ("CANCEL", "CANCELLED", "CANCELLED"),
        ):
            with self.subTest(choice=choice):
                result, observed, ops, recorder, events, _ = run(choice)
                self.assertEqual((result["code"], result["state"]), (code, state))
                self.assertEqual((result["data"]["recorder_goal_count"], result["data"]["execute_goal_count"]), (0, 0))
                self.assertEqual(ops, ["plan"])
                self.assertEqual(events[-1], "decision")
                self.assertCountEqual(
                    events[:-1], ["executor:plan", "camera_warmup"],
                )
                recorder.assert_not_called()
                self.assertEqual(observed[0]["decision_binding"]["data_disposition"], "TEST_ONLY")
                self.assertIsNotNone(observed[0]["decision_binding"]["root_binding_digest"])
                self.assertEqual(
                    observed[0]["decision_binding"]["episode_binding"]["binding_digest"],
                    run_job.canonical_digest("episode-binding"),
                )
                self.assertEqual(
                    observed[0]["decision_binding"]["start_binding_digest"],
                    run_job.canonical_digest("start-binding"),
                )
                self.assertEqual(
                    (
                        observed[0]["decision_binding"]["planned_start_evidence"]["status"],
                        observed[0]["decision_binding"]["planned_start_evidence"]["evidence_digest"],
                    ),
                    ("PASS", run_job.canonical_digest("planned-start")),
                )

        result, _, ops, recorder, _, _ = run("APPROVE", stale=True)
        self.assertEqual((result["code"], result["state"]), ("PLAN_DECISION_BINDING", "BLOCKED"))
        self.assertEqual(ops, ["plan"])
        recorder.assert_not_called()

        result, observed, ops, recorder, events, _ = run("APPROVE", expired=True)
        self.assertEqual((result["code"], result["state"]), ("TEST_ONLY_EPISODE_EXPIRED", "BLOCKED"))
        self.assertEqual((observed, ops), ([], []))
        self.assertEqual(events, [])
        recorder.assert_not_called()

        result, observed, ops, recorder, events, _ = run("APPROVE", start_mismatch=True)
        self.assertEqual(
            (result["code"], result["state"]),
            ("TEST_ONLY_PLANNED_START_MISMATCH", "BLOCKED"),
        )
        self.assertEqual((observed, ops), ([], ["plan"]))
        self.assertCountEqual(events, ["executor:plan", "camera_warmup"])
        recorder.assert_not_called()

        result, observed, ops, recorder, events, site_requests = run(
            None, with_site=True, site_choice="CANCEL",
        )
        self.assertEqual(
            (result["code"], result["state"]),
            ("PAUSED_AWAITING_OPERATOR", "PLANNED"),
            result,
        )
        self.assertEqual(result["data"]["measurement_outcome"], "NOT_MEASURED")
        self.assertEqual(observed, [])
        self.assertEqual(ops, ["plan"])
        self.assertEqual(events[-1], "checkpoint:PHYSICAL_SCENE_CONFIRMATION")
        self.assertCountEqual(
            events[:-1], ["executor:plan", "camera_warmup"],
        )
        self.assertEqual(site_requests[0]["kind"], "PHYSICAL_SCENE_CONFIRMATION")
        self.assertEqual(site_requests[0]["evidence"]["checklist"]["place_alias"], "place1")
        recorder.assert_not_called()

        result, observed, ops, recorder, events, site_requests = run(
            "REJECT", with_site=True,
        )
        self.assertEqual((result["code"], result["state"]), ("PLAN_REJECTED", "CANCELLED"))
        self.assertEqual(
            events[-2:],
            ["checkpoint:PHYSICAL_SCENE_CONFIRMATION", "decision"],
        )
        self.assertCountEqual(
            events[:-2], ["executor:plan", "camera_warmup"],
        )
        binding = observed[0]["decision_binding"]
        self.assertEqual(
            binding["operator_summary"]["flow"]["next_human_hold"],
            "PRECONTACT_HUMAN",
        )
        self.assertGreaterEqual(len(binding["operator_summary"]["path"]), 1)
        self.assertEqual(binding["preapproval_checklist"]["place_id"], "PLACE_A")
        self.assertEqual(
            binding["trajectory_variant_binding_digest"],
            binding["trajectory_variant_binding"]["binding_digest"],
        )
        self.assertEqual(
            binding["precommit_safety"]["approved_plan_digest"],
            observed[0]["plan_digest"],
        )
        self.assertIsNone(binding["yaw_sample_binding"])
        self.assertIsNone(binding["yaw_sample_binding_digest"])
        self.assertRegex(binding["plan_envelope_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(
            binding["preapproval_evidence_digest"],
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(
            binding["site_confirmation_digest"], run_job.canonical_digest({
                "kind": site_requests[0]["kind"], "choice": "READY",
                "run_id": site_requests[0]["run_id"],
                "plan_digest": site_requests[0]["plan_digest"],
                "checkpoint_binding_digest": run_job.canonical_digest({
                    key: site_requests[0][key]
                    for key in (
                        "kind", "run_id", "plan_digest", "prompt", "choices", "evidence",
                    )
                }),
                "decision_source": "LOCAL_UI_BUTTON", "operator_label": "operator",
            }),
        )
        recorder.assert_not_called()

        result, observed, ops, recorder, events, _ = run(
            "APPROVE", with_site=True, warmup_error=True,
        )
        self.assertEqual(
            (result["code"], result["state"], result["data"]["measurement_outcome"]),
            ("CAMERA_WARMUP_RATE", "BLOCKED", "FAIL"),
        )
        self.assertRegex(result["plan_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertIn("operator_summary", result["data"])
        self.assertRegex(
            result["data"]["preapproval_evidence_digest"],
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(
            result["data"]["test_only_episode_binding_digest"],
            run_job.canonical_digest("episode-binding"),
        )
        self.assertEqual(result["data"]["test_only_planned_start"], {
            "start_binding_digest": run_job.canonical_digest("start-binding"),
            "evidence_digest": run_job.canonical_digest("planned-start"),
            "status": "PASS",
        })
        self.assertEqual(observed, [])
        self.assertEqual(ops, ["plan"])
        self.assertCountEqual(events, ["executor:plan", "camera_warmup"])
        self.assertEqual(
            (result["data"]["recorder_goal_count"], result["data"]["execute_goal_count"]),
            (0, 0),
        )
        recorder.assert_not_called()

    def test_bound_production_uses_the_same_plan_gate_and_rejects_mixed_bindings(self):
        validated = runtime_validated(job={
            **JOB, "operator_or_agent_id": "operator", "instruction": "pick up",
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live_payload = payload("live")
            live_payload.update(
                run_root=str(root / "runs"), dataset_root=str(root / "dataset"),
            )
            roots = {
                "session_id": "session-r001", "run_id": live_payload["run_id"],
                "data_disposition": "PRODUCTION",
                "run_root": str((root / "runs").resolve()),
                "cell_root": str((root / "cells").resolve()),
                "dataset_root": str((root / "dataset").resolve()),
                "production_writers_enabled": True,
            }
            roots["binding_digest"] = run_job.canonical_digest(roots)
            episode = {
                "binding_digest": run_job.canonical_digest("production-episode"),
                "start_binding_digest": run_job.canonical_digest("production-start"),
                "manifest_digest": run_job.canonical_digest("manifest"),
                "intent_digest": run_job.canonical_digest("intent"),
                "slot_digest": run_job.canonical_digest("slot"),
                "data_disposition": "PRODUCTION",
                "expires_at": "2099-01-01T00:00:00Z",
            }
            ledger_context = {
                "manifest": {"manifest_digest": episode["manifest_digest"]},
                "intent": {
                    "intent_digest": episode["intent_digest"],
                    "slot": {"slot_id": "slot-1"},
                },
            }
            planned_start = {
                "start_binding_digest": episode["start_binding_digest"],
                "evidence_digest": run_job.canonical_digest("production-planned-start"),
                "status": "PASS",
            }
            decisions = []

            def reject(request):
                decisions.append(request)
                bound = {
                    "run_id": request["run_id"], "plan_digest": request["plan_digest"],
                    "approval_scope": request["approval_scope"],
                    "decision_binding": request["decision_binding"],
                }
                return {
                    "choice": "REJECT", "run_id": request["run_id"],
                    "plan_digest": request["plan_digest"],
                    "approval_scope": request["approval_scope"],
                    "decision_binding_digest": run_job.canonical_digest(bound),
                    "decision_source": "LOCAL_UI_BUTTON", "operator_label": "operator",
                }

            recorder = mock.Mock()
            cell = mock.Mock()
            cell.read.return_value = {
                "robot_system_id": live_payload["expected_robot_system_id"],
                "cell_ready": True,
            }
            with (
                mock.patch.object(run_job, "validate_runtime_root_binding", return_value=roots),
                mock.patch.object(run_job, "validate_runtime_episode_binding", return_value=episode),
                mock.patch.object(run_job, "validate_runtime_planned_start", return_value=planned_start),
                mock.patch.object(
                    run_job, "_validate_episode_ledger_context",
                    return_value=ledger_context,
                ),
                mock.patch.object(run_job, "CellStateStore", return_value=cell),
                mock.patch.object(run_job, "SceneStateStore"),
            ):
                no_ledger_resolver = mock.Mock()
                blocked = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=no_ledger_resolver, decision_provider=reject,
                    runtime_root_binding={"fixture": "production-root"},
                    runtime_episode_binding={"fixture": "production-episode"},
                    runtime_start_binding={"fixture": "production-start"},
                    repository_root=root,
                )
                self.assertEqual(
                    (blocked["code"], blocked["state"]),
                    ("PRODUCTION_RUN_BINDING", "BLOCKED"),
                )
                no_ledger_resolver.assert_not_called()
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, runtime_motion(validated), SCENE),
                    executor_factory=lambda *_: Executor(), recorder_factory=recorder,
                    camera_warmup_call=lambda *_: {
                        "schema_version": "data_factory.camera_warmup.v1", "attempts": [],
                    },
                    decision_provider=reject,
                    runtime_root_binding={"fixture": "production-root"},
                    runtime_episode_binding={"fixture": "production-episode"},
                    runtime_start_binding={"fixture": "production-start"},
                    episode_ledger_context={"fixture": "production-ledger"},
                    repository_root=root,
                )
            self.assertEqual((result["code"], result["state"]), ("PLAN_REJECTED", "CANCELLED"))
            recorder.assert_not_called()
            decision_binding = decisions[0]["decision_binding"]
            self.assertEqual(decision_binding["data_disposition"], "PRODUCTION")
            self.assertEqual(decision_binding["root_binding_digest"], roots["binding_digest"])
            self.assertEqual(decision_binding["planned_start_evidence"], planned_start)

            resolver = mock.Mock()
            mixed = run_job.run_live(
                live_payload, threading.Event(), lambda _: None,
                resolver=resolver, decision_provider=reject,
                test_only_root_binding={"fixture": True},
                runtime_root_binding={"fixture": True},
            )
            self.assertEqual(mixed["code"], "RUNTIME_BINDING_AMBIGUOUS")
            resolver.assert_not_called()

    def test_test_only_scope_rejects_before_resolver_or_filesystem_side_effect(self):
        resolver = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload.update(
                run_root=str(Path(directory) / "runs"),
                dataset_root=str(Path(directory) / "dataset"),
            )
            roots = {
                "run_id": live_payload["run_id"],
                "run_root": str(Path(live_payload["run_root"]).resolve()),
                "dataset_root": str(Path(live_payload["dataset_root"]).resolve()),
                "cell_root": str((Path(directory) / "cells").resolve()),
                "binding_digest": "sha256:" + "1" * 64,
            }
            with mock.patch.object(run_job, "validate_test_only_root_binding", return_value=roots):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=resolver,
                    decision_provider=lambda _: None,
                    test_only_root_binding={"fixture": True},
                )
            self.assertFalse(Path(live_payload["run_root"]).exists())
        self.assertEqual((result["code"], result["state"]), ("TEST_ONLY_RUN_BINDING", "BLOCKED"))
        resolver.assert_not_called()

    def test_runtime_collection_substitution_blocks_before_every_live_side_effect(self):
        for name in (
            "job_profile_id", "profile_document", "resolved_job_digest",
            "motion_resolved_job", "motion_profile_digest",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                validated = runtime_validated()
                program = runtime_motion(validated)
                if name == "job_profile_id":
                    validated["normalized_job"]["collection_profile_id"] = "other-profile"
                elif name == "profile_document":
                    validated["collection_profile"]["camera_topics"]["up"] = "/substituted"
                elif name == "resolved_job_digest":
                    validated["resolved_job_digest"] = run_job.canonical_digest("substituted")
                elif name == "motion_resolved_job":
                    program["resolved_job_digest"] = run_job.canonical_digest("substituted")
                else:
                    program["binding_digests"]["collection_profile"] = run_job.canonical_digest(
                        "substituted"
                    )
                live_payload = payload("live")
                live_payload["run_root"] = str(Path(directory) / "runs")
                live_payload["dataset_root"] = str(Path(directory) / "dataset")
                cell_store = mock.Mock()
                executor = mock.Mock()
                recorder = mock.Mock()
                warmup = mock.Mock()
                candidate = mock.Mock()
                with (
                    mock.patch.object(run_job, "CellStateStore", cell_store),
                    mock.patch.object(run_job, "write_candidate_admission", candidate),
                ):
                    result = run_job.run_live(
                        live_payload, threading.Event(), lambda _: None,
                        resolver=lambda _: (validated, program, SCENE),
                        executor_factory=executor, recorder_factory=recorder,
                        camera_warmup_call=warmup,
                    )
                self.assertEqual(
                    (result["code"], result["state"]),
                    ("COLLECTION_PROFILE_BINDING", "BLOCKED"),
                )
                self.assertEqual(list(Path(directory).iterdir()), [])
                for side_effect in (cell_store, executor, recorder, warmup, candidate):
                    side_effect.assert_not_called()

    def test_test_only_default_resolver_reads_only_the_bound_scene_root(self):
        validated = {
            "normalized_job": {
                **JOB, "place_id": "PLACE_A", "yaw_deg": 0,
                "x_mm": 0, "y_mm": 0, "object_profile_id": "wood-cube",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            live_payload = payload("live")
            live_payload.update(
                run_root=str(repository / "runs"),
                dataset_root=str(repository / "dataset"),
            )
            roots = {
                "run_id": live_payload["run_id"],
                "run_root": str(Path(live_payload["run_root"]).resolve()),
                "dataset_root": str(Path(live_payload["dataset_root"]).resolve()),
                "cell_root": str((repository / "isolated-cells").resolve()),
                "binding_digest": "sha256:" + "1" * 64,
            }
            with (
                mock.patch.object(run_job, "validate_test_only_root_binding", return_value=roots),
                mock.patch.object(run_job, "validate_job_spec", return_value=validated),
                mock.patch.object(run_job, "_load", return_value={}),
                mock.patch.object(run_job, "resolve_motion_program", return_value=motion()),
                mock.patch.object(
                    run_job, "_scene_binding",
                    side_effect=run_job.ContractError("ISOLATED_SCENE_SENTINEL"),
                ) as scene_binding,
            ):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    decision_provider=lambda _: None,
                    test_only_root_binding={"fixture": True},
                    test_only_episode_binding={"fixture": True},
                    test_only_start_binding={"fixture": True},
                    candidate_writer_enabled=False,
                    repository_root=repository,
                )
        self.assertEqual((result["code"], result["state"]), ("ISOLATED_SCENE_SENTINEL", "BLOCKED"))
        self.assertEqual(scene_binding.call_args.kwargs["root"], Path(roots["cell_root"]))
        self.assertNotEqual(scene_binding.call_args.kwargs["root"], run_job.ROOT / "outputs/data_factory/cells")

    def test_campaign_session_reaches_run_live_through_pure_test_only_ports(self):
        physical_profile = {
            **PROFILE, "collection_profile_id": "fr5-up-rgb-30hz-v1",
        }
        contract, motion_qualification, home_candidate = compatible_start_fixture(
            collection_profile=physical_profile,
        )
        source = campaign_draft(contract, count=1)
        manifest, receipt = compile_collection_campaign(source, hypothesis=contract)
        now = lambda: datetime.now(timezone.utc)
        trace, counters, checkpoint_requests = [], new_effect_counters(), []

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            roots = build_test_only_root_binding(
                repository, session_id="integrated-session", run_id="integrated-run",
            )
            slot = manifest["slots"][0]
            base = next(
                item for item in contract["base_conditions"]
                if item["base_condition_digest"] == slot["base_condition_digest"]
            )
            resolved = next(
                item for item in contract["resolver_receipts"]
                if item["resolver_result_digest"] == base["resolver_result_digest"]
            )
            normalized_job = resolved["normalized_job"]
            initialized = initialize_test_only_state_from_user_declaration(
                roots, repository_root=repository,
                robot_system_id=normalized_job["robot_system_id"],
                object_instance_id="synthetic-object",
                object_profile_id=normalized_job["object_profile_id"],
                place_id=normalized_job["place_id"], yaw_deg=normalized_job["yaw_deg"],
                x_mm=normalized_job["x_mm"], y_mm=normalized_job["y_mm"],
                declared_by="test-operator",
            )
            start_binding = build_test_only_start_binding(
                manifest=manifest, hypothesis=contract,
                motion_qualification=motion_qualification,
                home_candidate=home_candidate,
                current_snapshot=pose_snapshot(
                    motion_qualification["qualified_safe_joint_positions_rad"],
                ),
            )
            lifecycle = make_fake_one_job(trace=trace, counters=counters, clock=now)
            session = CampaignSession(
                session_id=roots["session_id"], source_draft=source,
                manifest=manifest, compilation_receipt=receipt,
                hypothesis=contract, lifecycle_owner="TEST_OPERATOR",
                expires_at="2099-01-01T00:00:00Z",
                initial_scene_digest=initialized["scene_state_digest"],
                effect_scope="PHYSICAL", lifecycle_action="LIVE_COLLECT",
                data_disposition="TEST_ONLY",
                fake_lifecycle_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("fake fallback must not be selected"),
                ),
                physical_lifecycle_factory=lambda: lifecycle,
                repository_root=repository, clock=now,
            )
            scene_evidence = {
                "schema_version": "data_factory.scene_freshness_evidence.v1",
                "scene_digest": initialized["scene_state_digest"],
                "observed_at": now().isoformat().replace("+00:00", "Z"),
            }
            scene_evidence["evidence_digest"] = run_job.canonical_digest(scene_evidence)

            class Cell:
                def __init__(self):
                    self.value = {
                        "robot_system_id": normalized_job["robot_system_id"],
                        "cell_ready": True, "reason_code": "TEST_OPERATOR_ACKNOWLEDGED",
                        "run_id": "NONE", "plan_digest": "sha256:" + "0" * 64,
                        "acknowledged_by": "TEST_OPERATOR",
                    }

                def read(self):
                    return dict(self.value)

                def mark_active(self, plan_digest):
                    self.value.update(
                        cell_ready=False, reason_code="EXECUTION_IN_PROGRESS",
                        run_id=roots["run_id"], plan_digest=plan_digest,
                        acknowledged_by="UNACKNOWLEDGED",
                    )

                def acknowledge_ready(self, operator, *, expected_run_id=None, expected_plan_digest=None):
                    if (expected_run_id, expected_plan_digest) != (
                        self.value["run_id"], self.value["plan_digest"],
                    ):
                        raise run_job.ContractError("STATE_CHANGED")
                    self.value.update(
                        cell_ready=True, reason_code="TEST_OPERATOR_ACKNOWLEDGED",
                        acknowledged_by=operator,
                    )
                    return self.read()

            cell = Cell()
            original_executor = lifecycle.executor_call
            original_recorder = lifecycle.recorder_call
            program_holder = {}

            class ExecutorPort:
                process = None

                def request(self, request, _cancel=None):
                    trace.append(f"run_live_executor:{request['op']}")
                    if request["op"] == "preflight":
                        return {"ok": True, "code": "PREFLIGHT_OK"}
                    response = original_executor(request)
                    if request["op"] == "plan" and response.get("ok"):
                        envelope = response["data"]
                        release = program_holder["scene_binding"]["release_slot"]
                        envelope["operator_summary"] = {
                            "path": [step["phase"] for step in program_holder["program"]["steps"]],
                            "flow": {
                                "continuous_through": "LIFT_LIN",
                                "next_human_hold": "POST_LIFT_SEMANTIC",
                            },
                            "speed": {
                                "max_velocity_scaling": 0.1,
                                "max_acceleration_scaling": 0.1,
                            },
                            "clearance": {
                                "status": "COLLISION_CHECKED_NO_DISTANCE",
                                "collision_report_digest": envelope["precommit_safety"]["collision_report_digest"],
                            },
                            "recycle": {
                                "recording_boundary_after": "LIFT_LIN",
                                "path": [
                                    "RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN",
                                    "RETREAT_LIN", "SAFE_POSE_PTP",
                                ],
                                "release_slot_id": release["slot_id"],
                                "release_target": normalized_job,
                                "safe_staging_joint_positions_rad": [0.0] * 6,
                                "plan_digest": run_job.canonical_digest("synthetic-recycle"),
                            },
                        }
                    if request["op"] == "execute" and response.get("ok"):
                        cell.mark_active(response["plan_digest"])
                    if request["op"] == "heartbeat" and response.get("state") == "COMPLETED":
                        release_evidence = {
                            "schema_version": "data_factory.recycle_release_evidence.v2",
                            "release_outcome": "LANDED",
                            "outcome_source": "LOCAL_UI_BUTTON",
                            "release_slot_id": program_holder["scene_binding"]["release_slot"]["slot_id"],
                        }
                        response["data"].update(
                            release_evidence=release_evidence,
                            scene_transition={
                                "scene_state_digest": run_job.canonical_digest("synthetic-post-scene"),
                                "release_evidence_digest": run_job.canonical_digest(release_evidence),
                            },
                        )
                    return response

                def close(self, **_):
                    return None

            class RecorderPort:
                process = None

                def __call__(self, request):
                    return original_recorder(request)

                def close(self, **_):
                    return None

            class Resource:
                def start(self):
                    return self

                def set_pid(self, *_):
                    return self

                def record_control_round_trip(self, _value):
                    return None

                def finish(self, *_args, **_kwargs):
                    return {"sampling": {"status": "AVAILABLE"}}

            def episode(intent, child, cancel_event, episode_context):
                self.assertIs(child, lifecycle)
                self.assertEqual(
                    episode_context["root_binding"]["binding_digest"],
                    roots["binding_digest"],
                )
                self.assertEqual(
                    episode_context["start_binding"]["binding_digest"],
                    start_binding["binding_digest"],
                )
                episode_binding = build_test_only_episode_binding(
                    roots=episode_context["root_binding"], repository_root=repository,
                    manifest=manifest, hypothesis=contract, intent=intent,
                    start_binding=episode_context["start_binding"],
                    state_initialization=initialized, resolved_job=resolved,
                    place_alias="place1",
                )
                live_payload = payload("live")
                live_payload.update(
                    run_id=roots["run_id"], job=normalized_job,
                    expected_robot_system_id=normalized_job["robot_system_id"],
                    camera_profile=physical_profile["camera_profile"],
                    run_root=roots["run_root"], dataset_root=roots["dataset_root"],
                )
                program_holder["program"] = _motion_program(
                    intent, contract, episode_context["start_binding"],
                )
                validated = {
                    **resolved, "collection_profile": copy.deepcopy(physical_profile),
                    "object_profile": {"dimensions_mm": [40, 30, 20]},
                }
                release_pose = {
                    key: normalized_job[key] for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
                }
                program_holder["scene_binding"] = run_job._scene_binding(
                    validated, release_pose,
                    roots["run_id"], root=Path(roots["cell_root"]),
                )

                def approve(request):
                    return {
                        "choice": "APPROVE", "run_id": request["run_id"],
                        "plan_digest": request["plan_digest"],
                        "approval_scope": request["approval_scope"],
                        "decision_binding_digest": run_job.canonical_digest({
                            "run_id": request["run_id"],
                            "plan_digest": request["plan_digest"],
                            "approval_scope": request["approval_scope"],
                            "decision_binding": request["decision_binding"],
                        }),
                        "decision_source": "LOCAL_UI_BUTTON",
                        "operator_label": normalized_job["operator_or_agent_id"],
                    }

                def checkpoint(request):
                    checkpoint_requests.append(copy.deepcopy(request))
                    bound = {
                        key: request[key]
                        for key in (
                            "kind", "run_id", "plan_digest", "prompt", "choices", "evidence",
                        )
                    }
                    return {
                        "kind": request["kind"], "choice": request["choices"][0],
                        "run_id": request["run_id"], "plan_digest": request["plan_digest"],
                        "checkpoint_binding_digest": run_job.canonical_digest(bound),
                        "decision_source": "LOCAL_UI_BUTTON",
                        "operator_label": normalized_job["operator_or_agent_id"],
                    }

                with (
                    mock.patch.object(run_job, "CellStateStore", return_value=cell) as cell_store,
                    mock.patch.object(run_job, "ResourceMonitor", return_value=Resource()),
                    mock.patch.object(run_job, "_write_storage_reference", return_value={"status": "SYNTHETIC"}),
                    mock.patch.object(run_job, "_write_resource_reference", return_value={"sampling": {"status": "AVAILABLE"}}),
                    mock.patch.object(run_job, "write_candidate_admission") as candidate_writer,
                ):
                    live_result = run_job.run_live(
                        live_payload, cancel_event, lambda _event: None,
                        resolver=lambda _payload: (
                            validated,
                            program_holder["program"], program_holder["scene_binding"],
                        ),
                        executor_factory=lambda *_: ExecutorPort(),
                        recorder_factory=lambda *_: RecorderPort(),
                        validator_call=lambda *_: {
                            "ok": True, "code": "PASS",
                            "result_digest": run_job.canonical_digest("synthetic-validator"),
                        },
                        tty_decision=lambda *_: (_ for _ in ()).throw(
                            AssertionError("browser checkpoint must replace the TTY")
                        ),
                        camera_warmup_call=lambda *_: {
                            "schema_version": "data_factory.camera_warmup.v1", "attempts": [],
                        },
                        one_job=child, decision_provider=approve,
                        checkpoint_provider=checkpoint,
                        approval_scope="HIL_NUMERIC_PROXY", decision_timeout_s=0,
                        test_only_root_binding=episode_context["root_binding"],
                        test_only_episode_binding=episode_binding,
                        test_only_start_binding=episode_context["start_binding"],
                        candidate_writer_enabled=False, repository_root=repository,
                    )
                cell_store.assert_called_once_with(
                    Path(roots["cell_root"]), normalized_job["robot_system_id"],
                )
                candidate_writer.assert_not_called()
                self.assertEqual(
                    (live_result["ok"], live_result["code"], live_result["state"]),
                    (True, "VALIDATED", "COMPLETE"),
                )
                technical = {
                    "schema_version": "data_factory.seed_technical_result.v1",
                    "intent_digest": intent["intent_digest"], "run_id": intent["run_id"],
                    "manifest_digest": intent["manifest_digest"],
                    "slot_id": intent["slot"]["slot_id"], "status": "PASS",
                    "technical_result_digest": live_result["data"]["technical_validator"]["result_digest"],
                    "post_scene_digest": live_result["data"]["postcommit_scene_state_digest"],
                    "observed_at": now().isoformat().replace("+00:00", "Z"),
                }
                technical["evidence_digest"] = run_job.canonical_digest(technical)
                return {"result": live_result, "technical_evidence": technical}

            result = session.run_next(
                run_id=roots["run_id"], scene_evidence=scene_evidence,
                episode_call=episode, roots=roots, start_binding=start_binding,
            )

        self.assertEqual(result["campaign"]["state"], "COMPLETE")
        self.assertEqual(result["result"]["data"]["data_disposition"], "TEST_ONLY")
        self.assertEqual(result["result"]["data"]["human_semantic_outcome"], "NOT_MEASURED")
        self.assertFalse(result["result"]["data"]["candidate_admission_written"])
        self.assertEqual(
            [(item["kind"], item["choices"]) for item in checkpoint_requests],
            [("RELEASE_VERDICT", ["LANDED", "OFF_SLOT", "UNCERTAIN"])],
        )
        self.assertTrue(all(counters[name] == 1 for name in (
            "fake_recorder_begin", "fake_recorder_readiness_status",
            "fake_recorder_freeze", "fake_recorder_commit",
        )))
        self.assertTrue(all(counters[name] == 0 for name in counters if name not in {
            "fake_recorder_begin", "fake_recorder_readiness_status",
            "fake_recorder_freeze", "fake_recorder_commit",
        }))

    def test_hil_numeric_proxy_uses_only_checked_gripper_range(self):
        program = motion()
        for state, key in (
            ("GRASP_VERDICT", "gripper_feedback_m"),
            ("SEMANTIC_VERDICT", "post_lift_gripper_feedback_m"),
        ):
            with self.subTest(state=state):
                evidence = {key: 0.011, "gripper_reference_m": 0.01}
                self.assertEqual(
                    run_job.hil_numeric_gripper_verdict(
                        state, evidence, program["gripper_requirements"],
                    ),
                    "PASS",
                )
                evidence[key] = 0.02
                self.assertEqual(
                    run_job.hil_numeric_gripper_verdict(
                        state, evidence, program["gripper_requirements"],
                    ),
                    "FAIL",
                )
        self.assertEqual(
            run_job.hil_numeric_gripper_verdict(
                "GRASP_VERDICT",
                {"gripper_feedback_m": 0.011, "gripper_reference_m": 0.011},
                program["gripper_requirements"],
            ),
            "FAIL",
        )

    def test_quality_rejected_recycle_reopens_only_the_exact_safe_cell(self):
        plan_digest = "sha256:" + "1" * 64
        slot_id = "sha256:" + "2" * 64
        scene_digest = "sha256:" + "3" * 64
        release = {
            "schema_version": "data_factory.recycle_release_evidence.v2",
            "release_outcome": "EXPECTED_LANDED",
            "outcome_source": "CAMPAIGN_CONTROL_PROXY",
            "release_slot_id": slot_id,
        }
        result = {
            "code": "QUALITY_REJECTED", "state": "ABORTED", "executor_state": "COMPLETED", "recorder_state": "ABORTED",
            "execution_evidence": {"release_evidence": release, "scene_transition": {
                "scene_state_digest": scene_digest, "release_evidence_digest": run_job.canonical_digest(release),
            }},
            "frozen_rows": 528, "rows_after_recycle": 528,
        }
        class Cell:
            def __init__(self): self.value = {"cell_ready": False, "run_id": "run", "plan_digest": plan_digest}; self.acks = 0
            def read(self): return dict(self.value)
            def acknowledge_ready(self, operator, *, expected_run_id, expected_plan_digest):
                self.assertions = (operator, expected_run_id, expected_plan_digest); self.acks += 1; self.value["cell_ready"] = True; return self.read()
        cell = Cell()
        observed = run_job._recover_quality_rejected_recycle(
            result, {"recycle": {"release_slot_id": slot_id}}, cell, "operator",
            {"run_id": "run"}, plan_digest,
        )
        self.assertEqual((observed[0], observed[1]["cell_ready"], cell.acks, cell.assertions), (scene_digest, True, 1, ("operator", "run", plan_digest)))
        with self.assertRaisesRegex(run_job.ContractError, "RECYCLE_EVIDENCE"):
            run_job._recover_quality_rejected_recycle(
                {**result, "rows_after_recycle": 529}, {"recycle": {"release_slot_id": slot_id}}, Cell(), "operator",
                {"run_id": "run"}, plan_digest,
            )

    def test_recycle_coordinates_are_an_exact_pair_and_reach_the_resolver(self):
        value = payload()
        value.update(recycle_x_mm=60, recycle_y_mm=-20)
        self.assertEqual(run_job._run_payload(value)["recycle_x_mm"], 60)
        for bad in (
            {**payload(), "recycle_x_mm": 60},
            {**payload(), "recycle_yaw_deg": 90},
            {**value, "recycle_y_mm": float("nan")},
            {**value, "recycle_x_mm": True},
        ):
            with self.assertRaisesRegex(run_job.ContractError, "RUN_PAYLOAD"):
                run_job._run_payload(bad)

        validated = {
            "normalized_job": {
                **JOB, "place_id": "PLACE_A", "yaw_deg": 0,
                "x_mm": -60, "y_mm": 20,
            },
            "object_profile": {"dimensions_mm": [25, 25, 25]},
            "calibration": {"document": {
                "limits": {"combined_error_bound_mm": 16},
            }},
        }
        coordinate_safety = {
            "object_dimensions_mm": [25, 25, 25],
            "uncertainty_mm": 16,
        }
        with (
            mock.patch.object(run_job, "validate_job_spec", return_value=validated),
            mock.patch.object(run_job, "_load", side_effect=lambda path, _: {"selected.json": {}, "motion.json": {}, "home.json": {}}[path]),
            mock.patch.object(run_job, "bounded_place_coordinate", return_value=(60, -20)) as bounded,
            mock.patch.object(run_job, "resolve_motion_program", return_value={}) as resolve,
        ):
            _, _, binding = run_job.resolve_inputs(value, scene_binding_call=lambda _, pose, _run_id: pose)
        bounded.assert_called_once_with({}, 60, -20, **coordinate_safety)
        self.assertEqual(binding, {"place_id": "PLACE_A", "yaw_deg": 0, "x_mm": 60, "y_mm": -20})
        self.assertEqual(resolve.call_args.kwargs["release_pose"], binding)

        pick_place_job = {
            **validated["normalized_job"], "task": "pick_place",
        }
        pick_place_validated = {
            **validated, "normalized_job": pick_place_job,
        }
        without_destination = payload()
        with (
            mock.patch.object(
                run_job, "validate_job_spec", return_value=pick_place_validated,
            ),
            mock.patch.object(
                run_job, "_load",
                side_effect=lambda path, _: {
                    "motion.json": {}, "home.json": {},
                }[path],
            ),
        ):
            with self.assertRaisesRegex(
                run_job.ContractError, "TASK_DESTINATION_REQUIRED",
            ):
                run_job.resolve_inputs(without_destination)

        with (
            mock.patch.object(
                run_job, "validate_job_spec", return_value=pick_place_validated,
            ),
            mock.patch.object(
                run_job, "_load",
                side_effect=lambda path, _: {
                    "selected.json": {}, "motion.json": {}, "home.json": {},
                }[path],
            ),
            mock.patch.object(
                run_job, "bounded_place_coordinate", return_value=(60, -20),
            ),
            mock.patch.object(
                run_job, "resolve_motion_program", return_value={},
            ) as pick_place_resolve,
        ):
            run_job.resolve_inputs(
                value, scene_binding_call=lambda _, pose, _run_id: pose,
            )
        self.assertEqual(
            pick_place_resolve.call_args.kwargs["release_pose"],
            {"place_id": "PLACE_A", "yaw_deg": 0, "x_mm": 60, "y_mm": -20},
        )

        destination_job = {
            **pick_place_job,
            "job_id": "destination-job",
            "place_id": "PLACE_B",
            "cell_calibration_id": "cal-b",
            "sheet_manifest_digest": "sha256:" + "b" * 64,
            "x_mm": 10,
            "y_mm": -10,
        }
        destination_validated = {
            **pick_place_validated,
            "normalized_job": destination_job,
            "input_digests": {"cell_calibration": "sha256:" + "d" * 64},
            "resolved_job_digest": "sha256:" + "c" * 64,
        }
        cross = {
            **without_destination,
            "job": pick_place_job,
            "destination": {
                "job": destination_job,
                "selected_sheet": "destination-selected.json",
                "yaw0_sheet": "destination-yaw0.json",
                "motion_qualification": "destination-motion.json",
            },
        }
        self.assertEqual(
            run_job._run_payload(cross)["destination"]["job"]["place_id"],
            "PLACE_B",
        )
        with self.assertRaisesRegex(run_job.ContractError, "RUN_PAYLOAD"):
            run_job._run_payload({**cross, "recycle_x_mm": 1, "recycle_y_mm": 2})
        with (
            mock.patch.object(
                run_job, "validate_job_spec",
                side_effect=[pick_place_validated, destination_validated],
            ),
            mock.patch.object(
                run_job, "_load",
                side_effect=lambda path, _: {
                    "motion.json": {"source": True},
                    "destination-motion.json": {"destination": True},
                    "home.json": {},
                }[path],
            ),
            mock.patch.object(
                run_job, "resolve_motion_program", return_value={},
            ) as cross_resolve,
        ):
            cross_validated, _, binding = run_job.resolve_inputs(
                cross, scene_binding_call=lambda _, pose, _run_id: pose,
            )
        self.assertEqual(binding["place_id"], "PLACE_B")
        self.assertEqual(cross_validated["destination_resolved_inputs"], {
            key: destination_validated[key] for key in (
                "normalized_job", "input_digests", "resolved_job_digest",
            )
        })
        self.assertIsNot(cross_validated["destination_resolved_inputs"]["normalized_job"], destination_job)
        self.assertIsNone(cross_resolve.call_args.kwargs["release_pose"])
        self.assertEqual(
            cross_resolve.call_args.kwargs["release_validated"],
            destination_validated,
        )
        self.assertEqual(
            cross_resolve.call_args.kwargs["release_motion_qualification"],
            {"destination": True},
        )

        rotated = {**value, "recycle_yaw_deg": 450}
        with (
            mock.patch.object(run_job, "validate_job_spec", return_value=validated),
            mock.patch.object(run_job, "_load", side_effect=lambda path, _: {"selected.json": {}, "motion.json": {}, "home.json": {}}[path]),
            mock.patch.object(run_job, "bounded_place_coordinate", return_value=(60, -20)) as bounded,
            mock.patch.object(run_job, "resolve_motion_program", return_value={}) as resolve,
        ):
            _, _, binding = run_job.resolve_inputs(
                rotated, scene_binding_call=lambda _, pose, _run_id: pose,
            )
        bounded.assert_called_once_with(
            {}, 60, -20, yaw_deg=90, **coordinate_safety,
        )
        self.assertEqual(
            binding,
            {"place_id": "PLACE_A", "yaw_deg": 90, "x_mm": 60, "y_mm": -20},
        )
        self.assertEqual(resolve.call_args.kwargs["release_pose"], binding)

    def test_chain_landed_source_is_bound_by_the_root_resolver_before_live_side_effects(self):
        next_source_pose = {
            "place_id": "place-a", "yaw_deg": 32.01437316452481,
            "x_mm": 59.99527764922155, "y_mm": 0.17407256236996815,
        }
        def landed():
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            root = Path(directory.name) / "cells"
            store = run_job.SceneStateStore(root, "fr5-lab-a")
            start = store.update_object(
                instance_id="cube-1", object_profile_id="wood-cube", state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": -60, "y_mm": 0},
                source="HUMAN", updated_by="operator", expected_revision=0,
            )
            source_slot = run_job.release_slot(
                robot_system_id="fr5-lab-a",
                pose=yaw_preserving_destination(
                    {**next_source_pose, "place_id": "place-b"}, next_source_pose,
                ),
                object_profile_id="wood-cube", exclusion_geometry_digest=run_job.canonical_digest({"shape": "BOX", "dimensions_mm": [25, 25, 25]}),
                role="DESTINATION_THEN_NEXT_SOURCE",
            )
            evidence = {
                "schema_version": "data_factory.recycle_release_evidence.v1", "run_id": "run-1",
                "plan_digest": "sha256:" + "1" * 64, "release_slot_id": source_slot["slot_id"],
                "expected_scene_state_digest": start["scene_state_digest"], "expected_scene_revision": start["scene_state"]["revision"],
                "gripper_reference_m": 0.021, "gripper_feedback_m": 0.021,
                "terminal_phases": ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"],
                "post_retreat_snapshot_digest": "sha256:" + "2" * 64, "next_start_tolerance_rad": 0.01,
                "human_verdict": "LANDED",
            }
            landed_state = store.transition_release(
                instance_id="cube-1", release_slot=source_slot, evidence=evidence, updated_by="pickup-executor",
                expected_digest=start["scene_state_digest"], expected_revision=start["scene_state"]["revision"],
                allowed_next_run_id="run-2",
            )
            return root, store, source_slot, landed_state

        validated = {
            "normalized_job": {
                **JOB, **next_source_pose, "job_id": "run-2",
                "object_profile_id": "wood-cube",
            },
            "object_profile": {"dimensions_mm": [25, 25, 25]},
        }
        release_pose = {"place_id": "place-a", "yaw_deg": 0, "x_mm": 60, "y_mm": 0}
        root, store, source_slot, landed_state = landed()
        expected_source = {
            "slot_id": source_slot["slot_id"],
            "slot_digest": run_job.canonical_digest(landed_state["scene_state"]["slot_allocations"][source_slot["slot_id"]]),
            "allowed_run_id": "run-2",
        }
        before_binding = store.snapshot()
        binding = run_job._scene_binding(validated, release_pose, "run-2", root=root)
        self.assertEqual(binding["source_slot"], expected_source)
        self.assertEqual(binding["scene_state_digest"], landed_state["scene_state_digest"])
        for changed_pose in (
            {"y_mm": 0.1740725623699646}, {"x_mm": 60.0}, {"yaw_deg": 32.0},
        ):
            with self.subTest(changed_pose=changed_pose):
                mismatched = {
                    **validated,
                    "normalized_job": {**validated["normalized_job"], **changed_pose},
                }
                with self.assertRaisesRegex(run_job.ContractError, "SCENE_OBJECT_NOT_READY"):
                    run_job._scene_binding(mismatched, release_pose, "run-2", root=root)
        self.assertEqual(store.snapshot(), before_binding)

        same_slot_binding = run_job._scene_binding(
            validated, source_slot["pose"], "run-2", root=root,
        )
        self.assertEqual(
            (
                same_slot_binding["source_slot"],
                same_slot_binding["release_slot"]["slot_id"],
            ),
            (expected_source, source_slot["slot_id"]),
        )

        human_directory = tempfile.TemporaryDirectory()
        self.addCleanup(human_directory.cleanup)
        human_root = Path(human_directory.name) / "cells"
        run_job.SceneStateStore(human_root, "fr5-lab-a").update_object(
            instance_id="cube-1", object_profile_id="wood-cube", state="ON_SURFACE",
            pose=next_source_pose,
            source="HUMAN", updated_by="operator", expected_revision=0,
        )
        self.assertNotIn("source_slot", run_job._scene_binding(validated, release_pose, "run-2", root=human_root))

        stale_root, stale_store, _, stale_state = landed()
        stale_scene = copy.deepcopy(stale_state["scene_state"])
        stale_scene["revision"] += 1
        stale_scene["objects"]["cube-1"].update({
            "pose": {**next_source_pose, "x_mm": -60, "y_mm": 0},
            "source": "HUMAN",
        })
        stale_store._path().write_text(json.dumps(stale_scene), encoding="utf-8")
        stale_validated = copy.deepcopy(validated)
        stale_validated["normalized_job"].update(x_mm=-60, y_mm=0)
        with self.assertRaisesRegex(run_job.ContractError, "SCENE_SLOT_NOT_READY"):
            run_job._scene_binding(
                stale_validated,
                next_source_pose,
                "new-run",
                root=stale_root,
            )

        side_effects = []
        live_payload = payload("live")
        live_payload["run_id"] = "run-3"
        result = run_job.run_live(
            live_payload, threading.Event(), lambda _: None,
            resolver=lambda _: (validated, {}, run_job._scene_binding(validated, release_pose, "run-3", root=root)),
            executor_factory=lambda *_: side_effects.append("executor"),
            recorder_factory=lambda *_: side_effects.append("recorder"),
            camera_warmup_call=lambda *_: side_effects.append("camera"),
        )
        self.assertEqual((result["code"], side_effects), ("SCENE_SLOT_NEXT_RUN", []))

        missing_root, missing_store, missing_slot, missing_state = landed()
        missing_scene = {**missing_state["scene_state"], "slot_allocations": {}}
        missing_store._path().write_text(json.dumps(missing_scene), encoding="utf-8")
        with self.assertRaisesRegex(run_job.ContractError, "SCENE_SLOT_NEXT_RUN"):
            run_job._scene_binding(validated, release_pose, "run-2", root=missing_root)

        consumed_root, consumed_store, consumed_slot, consumed_state = landed()
        consumed_value = consumed_state["scene_state"]["slot_allocations"][consumed_slot["slot_id"]]
        consumed_store.consume_next_source(
            slot_id=consumed_slot["slot_id"], run_id="run-2",
            expected_scene_digest=consumed_state["scene_state_digest"],
            expected_slot_digest=run_job.canonical_digest(consumed_value),
        )
        with self.assertRaisesRegex(run_job.ContractError, "SCENE_SLOT_NEXT_RUN"):
            run_job._scene_binding(validated, release_pose, "run-2", root=consumed_root)

        observed = {}
        live_payload["run_id"] = "run-2"
        def inspect_live(value, cancel, publish, *, resolver, before_approval):
            observed["binding"] = resolver(value)[2]
            return run_job._response(ok=True, code="VALIDATED", state="COMPLETE", run_id=value["run_id"])
        with mock.patch.object(run_job, "resolve_inputs", return_value=(validated, {}, binding)), mock.patch.object(run_job, "run_live", side_effect=inspect_live):
            run_job._campaign_episode(live_payload, threading.Event(), lambda _: None, "RELEASE_DESTINATION", None, expected_source)
        self.assertEqual(observed["binding"]["source_slot"], expected_source)
        with mock.patch.object(run_job, "resolve_inputs", return_value=(validated, {}, binding)), mock.patch.object(run_job, "run_live", side_effect=inspect_live), self.assertRaisesRegex(run_job.ContractError, "SCENE_SLOT_NEXT_RUN"):
            run_job._campaign_episode(
                live_payload, threading.Event(), lambda _: None, "RELEASE_DESTINATION", None,
                {**expected_source, "slot_digest": "sha256:" + "9" * 64},
            )

    def test_live_collection_profile_binds_camera_and_recorder_settings(self):
        profile = dict(PROFILE)
        validated = {"collection_profile": profile}
        self.assertEqual(run_job._collection_profile(validated, payload("live"))["fps"], 30)
        bad = payload("live"); bad["camera_profile"] = "up-side"
        with self.assertRaisesRegex(run_job.ContractError, "COLLECTION_PROFILE_MISMATCH"):
            run_job._collection_profile(validated, bad)
        with self.assertRaisesRegex(run_job.ContractError, "COLLECTION_PROFILE_V2_REQUIRED"):
            run_job._collection_profile({"collection_profile": {"schema_version": "data_factory.collection_profile.v1"}}, payload("live"))
        with self.assertRaisesRegex(run_job.ContractError, "COLLECTION_FPS_REQUIRED"):
            run_job._collection_profile({"collection_profile": {**profile, "fps": 15}}, payload("live"))
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(run_job, "JsonlProcess", side_effect=lambda command, timeout_s: (command, timeout_s)):
            live_payload = payload("live"); live_payload["dataset_root"] = str(Path(directory) / "dataset")
            command_line, _ = run_job._recorder(live_payload, "pick up", profile, 12)
            self.assertFalse(Path(live_payload["dataset_root"]).exists())
            self.assertEqual(command_line[0], run_job.DATA_PYTHON)
            self.assertEqual(
                Path(command_line[command_line.index("--encoder-temp-dir") + 1]),
                Path(directory) / ".dataset.encoder_tmp",
            )
        for flag, value in (("--fps", "30"), ("--min-camera-source-fps-ratio", "0.95"), ("--width", "640"), ("--height", "480"), ("--writer-queue-size", "128"), ("--encoder-threads", "2"), ("--camera-profile", "up"), ("--up-image", "/camera/up/color/image_raw"), ("--repo-id", "local/test"), ("--disk-reserve-bytes", "300"), ("--batch-video-encoding", None), ("--resume", None)):
            self.assertIn(flag, command_line)
            if value is not None:
                self.assertEqual(command_line[command_line.index(flag) + 1], value)
        completed = SimpleNamespace(returncode=0, stdout="PASS")
        with mock.patch.object(run_job.subprocess, "run", return_value=completed) as invoked:
            self.assertEqual(run_job._technical_validator("dataset", {}, profile)["code"], "PASS")
        validator_command = invoked.call_args.args[0]
        self.assertEqual(validator_command[0], run_job.DATA_PYTHON)
        self.assertEqual(validator_command[validator_command.index("--repo-id") + 1], profile["repo_id"])
        self.assertEqual(validator_command[validator_command.index("--expected-fps") + 1], "30")
        self.assertIn("--require-hil-motion", validator_command)
        self.assertIn("--require-alignment-tail", validator_command)
        self.assertIn("--skip-decoded-image-diagnostics", validator_command)
        self.assertNotIn("--episode-locator-index", validator_command)
        self.assertNotIn("--incremental-episode-index", validator_command)
        self.assertNotIn("--append-manifest", validator_command)

    def test_camera_warmup_retries_only_the_camera_gate_and_preserves_compact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            run_job._prepare_run_dir(live_payload)
            failed = SimpleNamespace(returncode=1, stdout="source rate 14Hz below gate\n")
            passed = SimpleNamespace(returncode=0, stdout="image: source=30Hz\n")
            with mock.patch.object(run_job.subprocess, "run", side_effect=(failed, passed)) as invoked:
                evidence = run_job._camera_warmup(live_payload, PROFILE, threading.Event())
            saved = json.loads((Path(directory) / live_payload["run_id"] / "camera_warmup.json").read_text(encoding="utf-8"))
        self.assertEqual([attempt["status"] for attempt in evidence["attempts"]], ["FAIL", "PASS"])
        self.assertEqual(saved, evidence)
        self.assertEqual(invoked.call_count, 2)
        command = invoked.call_args.args[0]
        self.assertEqual(command[command.index("--image") + 1], PROFILE["camera_topics"]["up"])
        self.assertEqual(command[command.index("--expected-image-hz") + 1], "30")
        self.assertEqual(command[command.index("--min-image-fps-ratio") + 1], "0.95")
        self.assertEqual(command[command.index("--max-image-age-ms") + 1], "300.0")
        self.assertEqual(
            command[command.index("--min-image-observation-s") + 1], "2.0",
        )
        self.assertEqual(command[command.index("--image-qos-depth") + 1], "10")
        self.assertIn("--reliable-image", command)
        self.assertEqual(invoked.call_args.kwargs["timeout"], run_job.CAMERA_WARMUP_TIMEOUT_S)

    def test_technical_validator_reuses_a_canonical_locator_and_fails_closed_without_it(self):
        locator = run_job.build_lerobot_v3_episode_locator(
            repo_id=PROFILE["repo_id"], episode_index=0,
            data={
                "chunk_index": 0, "file_index": 0,
                "relative_path": "data/chunk-000/file-000.parquet",
                "file_row_start": 0, "file_row_end_exclusive": 30,
            },
            videos=[{
                "camera_key": "observation.images.up",
                "chunk_index": 0, "file_index": 0,
                "relative_path": "videos/observation.images.up/chunk-000/file-000.mp4",
                "file_frame_start": 0, "file_frame_end_exclusive": 30,
                "timestamp_start_s": 0.0, "timestamp_end_s": 1.0,
            }],
        )
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            run_dir = run_job._prepare_run_dir(live_payload)
            (run_dir / "result.json").write_text(json.dumps({
                "schema_version": "data_factory.recorder_result.v1",
                "run_id": live_payload["run_id"],
                "transaction_id": "tx-1", "episode_index": 0,
                "state": "COMMITTED", "reason_code": "COMMITTED",
                "rows": 30, "detail": "",
            }), encoding="utf-8")
            passed = SimpleNamespace(
                returncode=0,
                stdout=(
                    run_job.EPISODE_LOCATOR_PREFIX
                    + json.dumps(locator, sort_keys=True, separators=(",", ":"))
                    + "\nPASS\n"
                ),
            )
            with mock.patch.object(run_job.subprocess, "run", return_value=passed) as invoked:
                result = run_job._technical_validator(
                    "dataset", live_payload, PROFILE,
                    validation_scope="INCREMENTAL",
                    expected_append_manifest_digest="sha256:" + "4" * 64,
                )
            self.assertEqual(result["episode_locator"], locator)
            self.assertEqual((result["ok"], result["code"]), (True, "PASS"))
            command_line = invoked.call_args.args[0]
            self.assertEqual(
                command_line[command_line.index("--episode-locator-index") + 1], "0",
            )
            self.assertEqual(
                command_line[command_line.index("--incremental-episode-index") + 1],
                "0",
            )
            self.assertEqual(
                Path(command_line[command_line.index("--append-manifest") + 1]),
                run_dir / "staging_manifest.json",
            )
            self.assertEqual(
                command_line[
                    command_line.index("--append-manifest-digest") + 1
                ],
                "sha256:" + "4" * 64,
            )

            missing = SimpleNamespace(returncode=0, stdout="PASS\n")
            with mock.patch.object(run_job.subprocess, "run", return_value=missing):
                result = run_job._technical_validator(
                    "dataset", live_payload, PROFILE,
                    validation_scope="INCREMENTAL",
                    expected_append_manifest_digest="sha256:" + "4" * 64,
                )
            self.assertEqual((result["ok"], result["code"]), (False, "FAIL"))
            self.assertIsNone(result["episode_locator"])

        with mock.patch.object(run_job.subprocess, "run") as invoked:
            result = run_job._technical_validator(
                "dataset", {}, PROFILE, validation_scope="INCREMENTAL",
            )
        self.assertEqual((result["ok"], result["code"]), (False, "FAIL"))
        invoked.assert_not_called()

    def test_camera_warmup_measures_configured_roles_concurrently(self):
        dual = copy.deepcopy(PROFILE)
        dual.update(
            camera_profile="up-wrist",
            camera_roles=["up", "wrist"],
            camera_topics={
                "up": "/camera/up/color/image_raw",
                "wrist": "/camera/wrist/color/image_raw",
            },
        )
        both_started = threading.Event()
        started = set()
        lock = threading.Lock()

        def measure(command, **_kwargs):
            topic = command[command.index("--image") + 1]
            with lock:
                started.add(topic)
                if len(started) == 2:
                    both_started.set()
            if not both_started.wait(1.0):
                raise AssertionError("camera probes ran serially")
            return SimpleNamespace(returncode=0, stdout=f"{topic}: 30Hz")

        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            run_job._prepare_run_dir(live_payload)
            with mock.patch.object(run_job.subprocess, "run", side_effect=measure):
                evidence = run_job._camera_warmup(
                    live_payload, dual, threading.Event(),
                )

        self.assertEqual(started, set(dual["camera_topics"].values()))
        self.assertEqual(
            [role["role"] for role in evidence["attempts"][0]["roles"]],
            ["up", "wrist"],
        )
        self.assertEqual(evidence["attempts"][0]["status"], "PASS")

    def test_camera_warmup_all_fail_or_timeout_allows_only_concurrent_plan_only(self):
        validated = runtime_validated()
        ready_cell = SimpleNamespace(read=lambda: {"robot_system_id": "fr5-lab-a", "cell_ready": True})
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            dataset_root = Path(directory) / "dataset-must-not-exist"
            live_payload["dataset_root"] = str(dataset_root)
            timeout = subprocess.TimeoutExpired(["probe"], run_job.CAMERA_WARMUP_TIMEOUT_S, output="probe timed out")
            executor = Executor()
            executor.request = mock.Mock(wraps=executor.request)
            executor_factory = mock.Mock(return_value=executor)
            recorder_factory = mock.Mock()
            with mock.patch.object(run_job.subprocess, "run", side_effect=(timeout, timeout)), mock.patch.object(run_job, "CellStateStore", return_value=ready_cell):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, runtime_motion(validated), SCENE), executor_factory=executor_factory,
                    recorder_factory=recorder_factory,
                )
            evidence = json.loads((Path(directory) / live_payload["run_id"] / "camera_warmup.json").read_text(encoding="utf-8"))
            self.assertFalse(dataset_root.exists())
        self.assertEqual((result["ok"], result["code"], result["state"]), (False, "CAMERA_WARMUP_FAILED", "BLOCKED"))
        self.assertEqual([attempt["status"] for attempt in evidence["attempts"]], ["FAIL", "FAIL"])
        self.assertEqual([role["status"] for attempt in evidence["attempts"] for role in attempt["roles"]], ["TIMEOUT", "TIMEOUT"])
        executor_factory.assert_called_once()
        self.assertEqual(
            [call.args[0]["op"] for call in executor.request.call_args_list],
            ["plan"],
        )
        recorder_factory.assert_not_called()

    def test_camera_warmup_cancel_is_bounded_and_allows_only_concurrent_plan_only(self):
        validated = runtime_validated()
        ready_cell = SimpleNamespace(read=lambda: {"robot_system_id": "fr5-lab-a", "cell_ready": True})
        cancel = threading.Event()
        executor = Executor()
        executor.request = mock.Mock(wraps=executor.request)
        executor_factory = mock.Mock(return_value=executor)
        recorder_factory = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            with mock.patch.object(run_job, "CellStateStore", return_value=ready_cell):
                result = run_job.run_live(
                    live_payload, cancel, lambda _: None,
                    resolver=lambda _: (validated, runtime_motion(validated), SCENE), executor_factory=executor_factory,
                    recorder_factory=recorder_factory,
                    camera_warmup_call=lambda *_: (cancel.set(), {"schema_version": "data_factory.camera_warmup.v1", "attempts": []})[1],
                )
        self.assertEqual((result["ok"], result["code"], result["state"]), (False, "CANCELLED", "CANCELLED"))
        executor_factory.assert_called_once()
        self.assertEqual(
            [call.args[0]["op"] for call in executor.request.call_args_list],
            ["plan"],
        )
        recorder_factory.assert_not_called()

    def test_camera_warmup_and_plan_overlap_then_join_before_operator_decision(self):
        validated = runtime_validated(job={
            **JOB, "operator_or_agent_id": "operator", "instruction": "pick up",
        })
        ready_cell = SimpleNamespace(
            read=lambda: {"robot_system_id": "fr5-lab-a", "cell_ready": True},
        )
        camera_started = threading.Event()
        plan_started = threading.Event()

        class OverlapExecutor(Executor):
            def request(self, request, cancel=None):
                if request["op"] == "plan":
                    plan_started.set()
                    self.assert_overlap(camera_started.wait(1.0))
                return super().request(request, cancel)

            @staticmethod
            def assert_overlap(overlapped):
                if not overlapped:
                    raise AssertionError("camera warm-up did not overlap plan-only")

        def warmup(*_):
            camera_started.set()
            if not plan_started.wait(1.0):
                raise AssertionError("plan-only did not overlap camera warm-up")
            return {"schema_version": "data_factory.camera_warmup.v1", "attempts": []}

        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            with mock.patch.object(run_job, "CellStateStore", return_value=ready_cell):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, runtime_motion(validated), SCENE),
                    executor_factory=lambda *_: OverlapExecutor(),
                    recorder_factory=mock.Mock(),
                    camera_warmup_call=warmup,
                    decision_provider=lambda _: None,
                    decision_timeout_s=0,
                )
        self.assertEqual(
            (result["code"], result["state"]),
            ("PAUSED_AWAITING_OPERATOR", "PLANNED"),
        )
        self.assertTrue(camera_started.is_set())
        self.assertTrue(plan_started.is_set())
        self.assertFalse(any(
            thread.name.startswith("camera-warmup")
            for thread in threading.enumerate()
        ))

    def test_live_preserves_plan_owned_preflight_failure_code(self):
        ready_cell = SimpleNamespace(read=lambda: {"robot_system_id": "fr5-lab-a", "cell_ready": True})
        validated = runtime_validated()
        class Process:
            def request(self, request, *_):
                return {
                    "schema_version": "fr5.pickup_executor.response.v3",
                    "mode": "LIVE", "op_id": request["op_id"], "op": request["op"],
                    "ok": False, "code": "CONTROLLER_ACTION_GRAPH",
                    "run_id": None, "plan_digest": None, "state": "BLOCKED", "data": None,
                }

            def close(self, **_):
                return None

        recorder_factory = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            with mock.patch.object(run_job, "CellStateStore", return_value=ready_cell):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, runtime_motion(validated), SCENE),
                    executor_factory=lambda *_: Process(), recorder_factory=recorder_factory,
                    camera_warmup_call=lambda *_: (_ for _ in ()).throw(
                        run_job.ContractError("CAMERA_WARMUP_FAILED")
                    ),
                )
        self.assertEqual((result["ok"], result["code"], result["state"]), (False, "CONTROLLER_ACTION_GRAPH", "BLOCKED"))
        recorder_factory.assert_not_called()

    def test_live_tty_path_plans_before_recorder_then_validates_without_training_authority(self):
        calls, prompts = [], []
        job = {
            "schema_version": "data_factory.job.v1", "job_id": "runner-test", "task": "pickup_e2e",
            "robot_system_id": "fr5-lab-a", "collection_profile_id": "test", "place_id": "PLACE_A",
            "cell_calibration_id": "cell-r1", "sheet_manifest_digest": "sha256:" + "1" * 64,
            "yaw_deg": 0, "x_mm": -70, "y_mm": 35, "object_profile_id": "wood-cube",
            "grasp_profile_id": "grasp-r1", "instruction": "pick up", "episode_intent": "nominal pickup",
            "operator_or_agent_id": "operator", "approval_expiry": "2026-08-21T00:00:00Z", "dry_run_required": True,
        }
        profile = dict(PROFILE)
        inputs = {
            name: motion()["binding_digests"][name] for name in (
                "selected_sheet", "yaw0_sheet", "cell_calibration", "robot_system",
                "collection_profile", "object_profile", "grasp_profile",
            )
        }
        validated = runtime_validated(job=job, profile=profile, input_digests=inputs)
        resolved_job_digest = validated["resolved_job_digest"]
        bindings = dict(motion()["binding_digests"])
        bindings["collection_profile"] = validated["input_digests"]["collection_profile"]
        executable_program = runtime_motion(validated, continuous=True)
        plan = {
            "schema_version": "fr5.pickup_plan.v3", "run_id": "runner-test",
            "resolved_job_digest": resolved_job_digest, "binding_digests": bindings,
            "robot_system_id": "fr5-lab-a",
            "motion_program_digest": run_job.canonical_digest(executable_program),
        }
        digest = run_job.canonical_digest(plan)
        summary = {
            "path": list(PHASES),
            "flow": {"continuous_through": "LIFT_LIN", "next_human_hold": "POST_LIFT_SEMANTIC"},
            "speed": {"max_velocity_scaling": 0.1, "max_acceleration_scaling": 0.1},
            "clearance": {"status": "COLLISION_CHECKED_NO_DISTANCE", "collision_report_digest": "sha256:" + "8" * 64},
        }

        class Process:
            def request(self, request, *_):
                calls.append(("executor", request["op"]))
                return {"ok": True, "code": "PREFLIGHT_OK"}
            def close(self, **_):
                calls.append(("executor", "close"))

        class Recorder:
            def __init__(self):
                calls.append(("recorder", "spawn"))
            def close(self, **_):
                calls.append(("recorder", "close"))

        class Resource:
            def __init__(self, *_): self.round_trips = []
            def start(self): return self
            def set_pid(self, *_): return self
            def record_control_round_trip(self, value): self.round_trips.append(value)
            def finish(self, metrics, collection_settings=None):
                return {"schema_version": "data_factory.resource_usage.v1", "sampling": {"status": "AVAILABLE"}, "recorder": metrics, "collection_settings": collection_settings}

        class Cell:
            def __init__(self): self.value = {"robot_system_id": "fr5-lab-a", "cell_ready": True, "run_id": "runner-test", "plan_digest": digest}
            def read(self): return dict(self.value)
            def acknowledge_ready(self, operator, *, expected_run_id=None, expected_plan_digest=None):
                if (expected_run_id, expected_plan_digest) != ("runner-test", digest):
                    raise run_job.ContractError("STATE_CHANGED")
                self.value.update(cell_ready=True, reason_code="HUMAN_ACKNOWLEDGED", acknowledged_by=operator)
                return dict(self.value)

        class Scene:
            def update_object(self, **value):
                calls.append(("scene", value["state"], value["expected_revision"]))
                return {"scene_state_digest": "sha256:" + "5" * 64}

        class FakeJob:
            def __init__(self, recorder_call, _executor_call, cell_state_call=None):
                self.recorder_call = recorder_call
                self.cell_state_call = cell_state_call
                self.state = "IDLE"
                self.phase = "SEMANTIC_VERDICT"
            def set_lifecycle_event_call(self, call):
                self.lifecycle_event_call = call
            def poll(self):
                if self.phase == "SEMANTIC_VERDICT":
                    value = {"ok": True, "state": self.phase, "execution_evidence": {"post_lift_gripper_feedback_m": .011}}
                else:
                    value = {"ok": True, "state": "COMMITTED", "recorder_evidence": {"metrics": {"writer_queue_high_water": 3, "writer_queue_drops": 0, "alignment_failures": 0, "storage_usage": {
                        "episode_index": 7, "transaction_id": "runner-test:episode-000007", "staging_manifest_digest": "sha256:" + "4" * 64,
                        "disk_reserve_bytes": 300, "dataset_incremental_peak_bytes": 100, "encoder_temp_peak_bytes": 200,
                        "required_free_bytes_by_device": {"1": 600}, "dataset_bytes_before": 1000, "dataset_bytes_after": 1200,
                        "free_bytes_before_by_device": {"1": 5000}, "free_bytes_by_device": {"1": 4800}, "temp_peak_bytes_by_device": {"1": 150},
                        "filesystems": {"dataset": {"path": "/dataset", "device": 1, "free_bytes": 4800, "total_bytes": 10000}, "encoder_temp": {"path": "/dataset/.encoder_tmp", "device": 1, "free_bytes": 4800, "total_bytes": 10000}},
                    }}}}
                self.state = value["state"]
                return value
            def plan_only(self, run_id, *_):
                self.state = "PLANNED"
                evidence = {
                    "schema_version": "data_factory.precommit_evidence.v1", "run_id": run_id,
                    "approved_plan_digest": digest, "scene_binding_digest": run_job.canonical_digest(SCENE),
                    "expected_planning_scene_digest": "sha256:" + "1" * 64,
                    "planning_scene_readback": {"schema_version": "data_factory.planning_scene_readback.v1", "run_id": run_id, "plan_digest": digest, "expected_planning_scene_digest": "sha256:" + "1" * 64, "objects": []},
                    "collision_report": {"schema_version": "data_factory.collision_report.v1", "plan_digest": digest, "sample_count": 0, "samples": [], "failure_count": 0, "all_valid": True},
                    "plan_only_no_motion": {"schema_version": "data_factory.plan_only_no_motion.v1", "run_id": run_id, "plan_digest": digest, "before_snapshot": {}, "after_snapshot": {}, "max_joint_delta_rad": 0.0, "gripper_delta_m": 0.0, "execute_goal_count": 0, "gripper_goal_count": 0},
                }
                safety = {
                    "schema_version": "data_factory.precommit_safety.v1", "run_id": run_id,
                    "approved_plan_digest": digest, "scene_binding_digest": run_job.canonical_digest(SCENE),
                    "expected_planning_scene_digest": "sha256:" + "1" * 64,
                    "planning_scene_readback_digest": run_job.canonical_digest(evidence["planning_scene_readback"]),
                    "collision_report_digest": run_job.canonical_digest(evidence["collision_report"]),
                    "plan_only_no_motion_digest": run_job.canonical_digest(evidence["plan_only_no_motion"]),
                    "post_reset_safe_snapshot_digest": None, "status": "PENDING",
                }
                return {"ok": True, "code": "PLANNED", "state": "PLANNED", "run_id": run_id, "plan_digest": digest, "plan_envelope": {"plan": plan, "precommit_safety": safety, "precommit_evidence": evidence, "operator_summary": summary}}
            def approve(self, approval):
                calls.append(("job", "approve", approval["approval_scope"])); self.state = "APPROVED"
                return {"ok": True, "code": "APPROVED", "state": self.state}
            def start(self):
                calls.append(("job", "start")); cell.value.update(cell_ready=False, reason_code="EXECUTION_IN_PROGRESS"); self.state = "EXECUTING"
                return {"ok": True, "code": "EXECUTING", "state": self.state}
            def confirm(self, _):
                calls.append(("job", "confirm")); self.phase = "GRASP_VERDICT"; self.state = "EXECUTING"
                return {"ok": True, "code": "CONFIRMED", "state": self.state}
            def grasp_verdict(self, value, _, source=None):
                calls.append(("job", "grasp", value, source))
                if value == "FAIL":
                    self.state = "ABORTED"
                    return {"ok": False, "code": "GRASP_REJECTED", "state": self.state}
                calls.append(("job", "lift")); self.phase = "SEMANTIC_VERDICT"; self.state = "EXECUTING"
                return {"ok": True, "code": "GRASP_VERDICT_ACCEPTED", "state": self.state}
            def semantic_verdict(self, value, _, source=None):
                calls.append(("job", "semantic", value, source))
                if value == "FAIL":
                    self.state = "ABORTED"
                    return {"ok": False, "code": "SEMANTIC_REJECTED", "state": self.state}
                self.phase = "COMMITTED"; self.state = "EXECUTING"
                return {"ok": True, "code": "VERDICT_ACCEPTED", "state": self.state}
            def cancel(self):
                return {"ok": False, "code": "CANCELLED", "state": "ABORTED"}
            def finish(self):
                if (Path(directory) / "runner-test" / "candidate_admission.json").exists():
                    raise AssertionError("candidate admission must follow the terminal job gate")
                calls.append(("job", "finish")); self.state = "COMPLETE"
                return {"ok": True, "code": "COMPLETE", "state": self.state}

        cell, scene = Cell(), Scene()
        semantic_reply = "PASS"
        def decide(prompt, expected):
            prompts.append((prompt, expected))
            if expected == ("PASS", "FAIL"):
                self.assertNotIn(("job", "confirm"), calls)
                self.assertFalse(any(call[:2] == ("job", "grasp") for call in calls))
                return semantic_reply
            return None

        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            with mock.patch.object(run_job, "OneJob", FakeJob), mock.patch.object(run_job, "ResourceMonitor", Resource), mock.patch.object(run_job, "CellStateStore", return_value=cell), mock.patch.object(run_job, "SceneStateStore", return_value=scene):
                result = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, executable_program, SCENE), executor_factory=lambda *_: Process(), recorder_factory=lambda *_: Recorder(),
                    validator_call=lambda *_: {"ok": True, "code": "PASS", "result_digest": "sha256:" + "6" * 64}, tty_decision=decide,
                    camera_warmup_call=lambda *_: (calls.append(("camera_warmup", "PASS")), {"schema_version": "data_factory.camera_warmup.v1", "attempts": []})[1],
                )
            reference = json.loads((Path(directory) / live_payload["run_id"] / "technical_validator.json").read_text(encoding="utf-8"))
            admission_path = Path(directory) / live_payload["run_id"] / "candidate_admission.json"
            admission = json.loads(admission_path.read_text(encoding="utf-8"))
            preapproval = json.loads((Path(directory) / live_payload["run_id"] / "preapproval_evidence.json").read_text(encoding="utf-8"))
            storage = json.loads((Path(directory) / live_payload["run_id"] / "storage_usage.json").read_text(encoding="utf-8"))
            resource = json.loads((Path(directory) / live_payload["run_id"] / "resource_usage.json").read_text(encoding="utf-8"))
            condition = {
                "task_schema_version": "data_factory.job.v1", "task": "pickup_e2e", "robot_system_id": "fr5-lab-a",
                "place_id": "PLACE_A", "cell_calibration_id": "cell-r1", "cell_calibration_digest": bindings["cell_calibration"],
                "yaw_deg": 0, "x_mm": -70, "y_mm": 35, "object_profile_id": "wood-cube", "grasp_profile_id": "grasp-r1",
                "motion_recipe_digest": bindings["motion_qualification"], "collection_profile_digest": validated["input_digests"]["collection_profile"],
            }
            root = Path(directory)
            job_path = root / "job.json"
            domain_path = root / "coverage-domain.json"
            stored_path = root / "stored-episodes.json"
            job_path.write_text(json.dumps(job), encoding="utf-8")
            domain_path.write_text(json.dumps({
                "schema_version": "data_factory.coverage_domain.v1", "collection_profile_id": "test",
                "conditions": [condition, {**condition, "x_mm": -60}], "slots": [],
            }), encoding="utf-8")
            stored = {
                "schema_version": "data_factory.coverage_stored_episodes.v2",
                "episodes": [{
                    "episode_id": "runner-test", "job_spec_path": str(job_path), "job_spec_digest": run_job.canonical_digest(job),
                    "preapproval_evidence_path": str(root / "runner-test" / "preapproval_evidence.json"),
                    "preapproval_evidence_digest": run_job.canonical_digest(preapproval),
                    "technical_validator_path": str(root / "runner-test" / "technical_validator.json"),
                    "technical_validator_digest": run_job.canonical_digest(reference),
                    "candidate_admission_path": str(admission_path), "candidate_admission_digest": run_job.canonical_digest(admission),
                }],
            }
            stored_path.write_text(json.dumps(stored), encoding="utf-8")
            coverage_root = root / "coverage"
            coverage_cli = subprocess.run([
                sys.executable, "-m", "tools.data_factory.quality.coverage_report",
                "--domain-manifest", str(domain_path), "--stored-episodes", str(stored_path),
                "--output-root", str(coverage_root),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            coverage = json.loads(coverage_cli.stdout)

            bad_root = root / "bad-coverage"
            stored_path.write_text(json.dumps({**stored, "episodes": [{**stored["episodes"][0], "candidate_admission_digest": "sha256:" + "9" * 64}]}), encoding="utf-8")
            mismatched_cli = subprocess.run([
                sys.executable, "-m", "tools.data_factory.quality.coverage_report",
                "--domain-manifest", str(domain_path), "--stored-episodes", str(stored_path), "--output-root", str(bad_root),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            missing_cli = subprocess.run([
                sys.executable, "-m", "tools.data_factory.quality.coverage_report",
                "--domain-manifest", str(domain_path), "--stored-episodes", str(root / "missing.json"), "--output-root", str(bad_root),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            domain_path.write_text("{}", encoding="utf-8")
            malformed_cli = subprocess.run([
                sys.executable, "-m", "tools.data_factory.quality.coverage_report",
                "--domain-manifest", str(domain_path), "--stored-episodes", str(stored_path), "--output-root", str(bad_root),
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual((result["ok"], result["code"], result["state"]), (True, "VALIDATED", "COMPLETE"))
        self.assertEqual(
            [item[:2] for item in calls[:4]],
            [("camera_warmup", "PASS"), ("job", "approve"), ("recorder", "spawn"), ("job", "start")],
        )
        self.assertIn(("job", "approve", "HUMAN_GATED"), calls)
        self.assertLess(calls.index(("camera_warmup", "PASS")), calls.index(("recorder", "spawn")))
        self.assertEqual([expected for _, expected in prompts], [f"APPROVE {digest}", ("PASS", "FAIL"), f"SCENE_READY {digest}"])
        self.assertEqual(summary["flow"], {"continuous_through": "LIFT_LIN", "next_human_hold": "POST_LIFT_SEMANTIC"})
        self.assertNotIn(("job", "confirm"), calls)
        self.assertFalse(any(call[:2] == ("job", "grasp") for call in calls))
        self.assertIn(("job", "semantic", "PASS", "HUMAN"), calls)
        self.assertFalse(result["data"]["camera_semantic_authority"])
        self.assertFalse(result["data"]["training_authorized"])
        self.assertEqual((reference["status"], reference["expected_fps"], reference["plan_digest"]), ("PASS", 30, digest))
        self.assertEqual(admission, {
            "schema_version": "data_factory.candidate_admission.v1", "run_id": "runner-test",
            "operational_gate": "PASS", "operational_source": "HUMAN_GATED", "checklist_id": "pickup-v2",
            "review_context_digest": run_job.canonical_digest({
                "run_id": "runner-test", "resolved_job_digest": validated["resolved_job_digest"],
                "plan_digest": digest, "technical_validator_digest": run_job.canonical_digest(reference),
            }),
            "semantic_status": "PENDING", "reviewed_by": None, "reviewed_at": None, "reason": None,
        })
        self.assertEqual(coverage_cli.returncode, 0, coverage_cli.stderr)
        self.assertEqual(coverage["cells"][0]["counts"]["pending_review"], 1)
        self.assertEqual(coverage["cells"][0]["counts"]["human_semantic_pass"], 0)
        self.assertEqual(coverage["cells"][0]["counts"]["human_training_approved"], 0)
        self.assertEqual(coverage["suggest_next"]["x_mm"], -60)
        self.assertEqual((mismatched_cli.returncode, mismatched_cli.stderr.strip(), bad_root.exists()), (2, "COVERAGE_STORED_DIGEST", False))
        self.assertEqual((missing_cli.returncode, missing_cli.stderr.strip()), (2, "COVERAGE_IO"))
        self.assertEqual((malformed_cli.returncode, malformed_cli.stderr.strip()), (2, "COVERAGE_DOMAIN_MANIFEST"))
        self.assertEqual((preapproval["plan_digest"], preapproval["plan_envelope_digest"]), (digest, run_job.canonical_digest(preapproval["plan_envelope"])))
        self.assertEqual(run_job.canonical_digest(preapproval["plan_envelope"]["precommit_evidence"]["collision_report"]), preapproval["plan_envelope"]["precommit_safety"]["collision_report_digest"])
        self.assertEqual((storage["episode_ref"]["repo_id"], storage["dataset_delta_bytes"], storage["reference_scan_status"], storage["dataset_prunable"]), ("local/test", 200, "NOT_AVAILABLE", []))
        self.assertEqual(resource["sampling"]["status"], "AVAILABLE")
        self.assertIn(("scene", "ON_SURFACE", SCENE["revision"]), calls)
        self.assertIn(("job", "finish"), calls)

        calls.clear()
        prompts.clear()
        semantic_reply = "FAIL"
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload["run_root"] = directory
            with mock.patch.object(run_job, "OneJob", FakeJob), mock.patch.object(run_job, "ResourceMonitor", Resource), mock.patch.object(run_job, "CellStateStore", return_value=Cell()), mock.patch.object(run_job, "SceneStateStore", return_value=Scene()):
                rejected = run_job.run_live(
                    live_payload, threading.Event(), lambda _: None,
                    resolver=lambda _: (validated, executable_program, SCENE), executor_factory=lambda *_: Process(), recorder_factory=lambda *_: Recorder(),
                    validator_call=lambda *_: {"ok": True, "code": "PASS", "result_digest": "sha256:" + "6" * 64}, tty_decision=decide,
                    camera_warmup_call=lambda *_: ({"schema_version": "data_factory.camera_warmup.v1", "attempts": []}),
                )
        self.assertEqual((rejected["ok"], rejected["code"], rejected["state"]), (False, "SEMANTIC_REJECTED", "ABORTED"))
        self.assertIn(("job", "semantic", "FAIL", "HUMAN"), calls)
        self.assertNotIn(("job", "finish"), calls)

    def test_postcommit_validation_or_evidence_failure_keeps_cell_blocked_without_scene_ack(self):
        validated = runtime_validated(job={
            **JOB, "operator_or_agent_id": "operator", "instruction": "pick up",
            "place_id": "PLACE_A", "yaw_deg": 0, "x_mm": -70, "y_mm": 35,
            "object_profile_id": "wood-cube",
        })
        executable_program = runtime_motion(validated, continuous=True)
        plan = {
            "schema_version": "fr5.pickup_plan.v3", "run_id": "runner-test",
            "evidence": "fake",
            "motion_program_digest": run_job.canonical_digest(executable_program),
        }
        digest = run_job.canonical_digest(plan)
        evidence = {
            "schema_version": "data_factory.precommit_evidence.v1", "run_id": "runner-test",
            "approved_plan_digest": digest, "scene_binding_digest": run_job.canonical_digest(SCENE),
            "expected_planning_scene_digest": "sha256:" + "1" * 64,
            "planning_scene_readback": {"schema_version": "data_factory.planning_scene_readback.v1", "run_id": "runner-test", "plan_digest": digest, "expected_planning_scene_digest": "sha256:" + "1" * 64, "objects": []},
            "collision_report": {"schema_version": "data_factory.collision_report.v1", "plan_digest": digest, "sample_count": 0, "samples": [], "failure_count": 0, "all_valid": True},
            "plan_only_no_motion": {"schema_version": "data_factory.plan_only_no_motion.v1", "run_id": "runner-test", "plan_digest": digest, "before_snapshot": {}, "after_snapshot": {}, "max_joint_delta_rad": 0.0, "gripper_delta_m": 0.0, "execute_goal_count": 0, "gripper_goal_count": 0},
        }
        safety = {
            "schema_version": "data_factory.precommit_safety.v1", "run_id": "runner-test", "approved_plan_digest": digest,
            "scene_binding_digest": run_job.canonical_digest(SCENE), "expected_planning_scene_digest": "sha256:" + "1" * 64,
            "planning_scene_readback_digest": run_job.canonical_digest(evidence["planning_scene_readback"]),
            "collision_report_digest": run_job.canonical_digest(evidence["collision_report"]),
            "plan_only_no_motion_digest": run_job.canonical_digest(evidence["plan_only_no_motion"]),
            "post_reset_safe_snapshot_digest": None, "status": "PENDING",
        }
        class Process:
            def request(self, request, *_):
                return {"ok": True, "code": "PREFLIGHT_OK"} if request["op"] == "preflight" else {"ok": True}
            def close(self, **_):
                return None

        class Recorder:
            def close(self, **_):
                return None

        class Resource:
            def __init__(self, *_):
                pass
            def start(self):
                return self
            def set_pid(self, *_):
                return self
            def record_control_round_trip(self, _):
                return None
            def finish(self, metrics, collection_settings=None):
                return {"schema_version": "data_factory.resource_usage.v1", "sampling": {"status": "AVAILABLE"}, "recorder": metrics, "collection_settings": collection_settings}

        class Cell:
            def __init__(self):
                self.value = {"robot_system_id": "fr5-lab-a", "cell_ready": True, "run_id": "runner-test", "plan_digest": digest}
                self.blocked = []
                self.acks = 0
            def read(self):
                return dict(self.value)
            def mark_blocked(self, reason_code, run_id, plan_digest):
                self.blocked.append((reason_code, run_id, plan_digest))
                self.value.update(cell_ready=False, reason_code=reason_code, run_id=run_id, plan_digest=plan_digest)
                return self.read()
            def acknowledge_ready(self, *_args, **_kwargs):
                self.acks += 1
                raise AssertionError("must not acknowledge failed evidence")

        class Scene:
            def update_object(self, **_):
                raise AssertionError("must not update physical scene on failed evidence")

        class FailedCommitJob:
            instances = []
            def __init__(self, *_args, **_kwargs):
                self.finish_calls = 0
                self.__class__.instances.append(self)
            def set_lifecycle_event_call(self, call):
                self.lifecycle_event_call = call
            def plan_only(self, run_id, *_):
                return {"ok": True, "code": "PLANNED", "state": "PLANNED", "run_id": run_id, "plan_digest": digest, "plan_envelope": {"plan": plan, "precommit_safety": safety, "precommit_evidence": evidence, "operator_summary": {"path": list(PHASES), "flow": {"continuous_through": "LIFT_LIN", "next_human_hold": "POST_LIFT_SEMANTIC"}, "speed": {"max_velocity_scaling": 0.1}, "clearance": {"status": "COLLISION_CHECKED_NO_DISTANCE"}}}}
            def approve(self, _):
                return {"ok": True, "code": "APPROVED", "state": "APPROVED"}
            def start(self):
                cell.value.update(cell_ready=False, reason_code="EXECUTION_IN_PROGRESS")
                return {"ok": True, "code": "EXECUTING", "state": "EXECUTING"}
            def poll(self):
                return {"ok": True, "state": "COMMITTED", "recorder_evidence": {"metrics": {}}}
            def finish(self):
                self.finish_calls += 1
                return {"ok": True, "state": "COMPLETE"}

        for label, validator, storage_error, resource_available, cell_matches, expected in (
            ("validator", {"ok": False, "code": "FAIL", "result_digest": "sha256:" + "6" * 64}, False, True, True, "TECHNICAL_VALIDATOR_FAILED"),
            ("storage", {"ok": True, "code": "PASS", "result_digest": "sha256:" + "6" * 64}, True, True, True, "STORAGE_REFERENCE_ERROR"),
            ("resource", {"ok": True, "code": "PASS", "result_digest": "sha256:" + "6" * 64}, False, False, True, "RESOURCE_EVIDENCE_ERROR"),
            ("cell", {"ok": True, "code": "PASS", "result_digest": "sha256:" + "6" * 64}, False, True, False, "POSTCOMMIT_CELL_STATE"),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                live_payload = payload("live")
                live_payload["run_root"] = directory
                cell, prompts = Cell(), []
                if not cell_matches:
                    cell.value["run_id"] = "other-run"
                with mock.patch.object(run_job, "OneJob", FailedCommitJob), mock.patch.object(run_job, "ResourceMonitor", Resource), mock.patch.object(run_job, "CellStateStore", return_value=cell), mock.patch.object(run_job, "SceneStateStore", return_value=Scene()):
                    if storage_error:
                        storage_patch = mock.patch.object(run_job, "_write_storage_reference", side_effect=run_job.ContractError("STORAGE_REFERENCE_ERROR"))
                    else:
                        storage_patch = mock.patch.object(run_job, "_write_storage_reference", return_value={"preserved": True})
                    with storage_patch, mock.patch.object(run_job, "_write_resource_reference", return_value={"sampling": {"status": "AVAILABLE" if resource_available else "NOT_AVAILABLE"}}):
                        result = run_job.run_live(
                            live_payload, threading.Event(), lambda _: None,
                            resolver=lambda _: (validated, runtime_motion(validated, continuous=True), SCENE), executor_factory=lambda *_: Process(), recorder_factory=lambda *_: Recorder(),
                            validator_call=lambda *_: validator, tty_decision=lambda prompt, expected_text: prompts.append((prompt, expected_text)),
                            camera_warmup_call=lambda *_: {"schema_version": "data_factory.camera_warmup.v1", "attempts": []},
                        )
                self.assertEqual((result["ok"], result["code"], result["state"]), (False, expected, "BLOCKED"))
                if result["data"] is not None:
                    self.assertFalse(result["data"]["training_authorized"])
                self.assertEqual(cell.blocked, [] if label == "cell" else [(expected, "runner-test", digest)])
                self.assertEqual(cell.acks, 0)
                self.assertEqual(FailedCommitJob.instances[-1].finish_calls, 0)
                self.assertEqual([expected_text for _, expected_text in prompts], [f"APPROVE {digest}"])
                self.assertFalse((Path(directory) / live_payload["run_id"] / "candidate_admission.json").exists())

    def test_bound_production_live_commits_and_binds_one_candidate_to_one_ledger(self):
        self._bound_production_live()

    def test_bound_production_failed_plan_only_continuation_offers_review_and_releases_parent(self):
        self._bound_production_live("pending")

    def test_bound_production_continuation_preserves_error_when_candidate_write_fails(self):
        self._bound_production_live("candidate_write")

    def test_bound_production_continuation_preserves_error_when_ledger_bind_fails(self):
        self._bound_production_live("ledger_bind")

    def test_bound_production_uncertain_continuation_retains_exact_parent(self):
        self._bound_production_live("uncertain")

    def _bound_production_live(self, continuation=None):
        """Drive the real parent lifecycle through the public runner with fixture ports."""
        validated = runtime_validated(job={
            **JOB, "operator_or_agent_id": "operator", "instruction": "pick up",
            "place_id": "PLACE_A", "yaw_deg": 0, "x_mm": -70, "y_mm": 35,
            "object_profile_id": "wood-cube",
        })
        # Keep the recorder/executor builder's planned job joined to this resolver.
        resolved_patch = mock.patch.object(one_job_test, "RESOLVED", validated["resolved_job_digest"])
        resolved_patch.start()
        self.addCleanup(resolved_patch.stop)
        helper = one_job_test.OneJobTest()
        self.addCleanup(helper.doCleanups)
        recorder_call, executor_call = None, None
        job, calls = helper.make(
            ["RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"],
            ["PLANNED", "APPROVED", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"],
            first_row_rows=60,
            continuous=True,
            readiness_contract=run_job.RECORDER_READINESS_CONTRACT,
        )
        recorder_call, executor_call = job.recorder_call, job.executor_call
        timeline, validator_calls, published, test_case = [], [], [], self

        storage = {
            "episode_index": 0, "transaction_id": "tx", "staging_manifest_digest": "sha256:" + "4" * 64,
            "disk_reserve_bytes": 300, "dataset_incremental_peak_bytes": 100, "encoder_temp_peak_bytes": 200,
            "required_free_bytes_by_device": {"1": 600}, "dataset_bytes_before": 1000, "dataset_bytes_after": 1200,
            "free_bytes_before_by_device": {"1": 5000}, "free_bytes_by_device": {"1": 4800}, "temp_peak_bytes_by_device": {"1": 150},
            "filesystems": {"dataset": {"path": "/dataset", "device": 1, "free_bytes": 4800, "total_bytes": 10000}, "encoder_temp": {"path": "/dataset/.encoder_tmp", "device": 1, "free_bytes": 4800, "total_bytes": 10000}},
        }

        class Recorder:
            def __call__(self, request):
                return self.request(request)

            def request(self, request, **_):
                response = recorder_call(request)
                response = {**response, "metrics": dict(response["metrics"])}
                if request["op"] == "status":
                    response["metrics"].update(writer_queue=0, writer_queue_drops=0, alignment_failures=0, observed_monotonic_ns=time.monotonic_ns())
                    timeline.append(("recorder", "status", response["metrics"]["rows"]))
                else:
                    timeline.append(("recorder", request["op"]))
                if request["op"] == "commit":
                    response["metrics"]["storage_usage"] = storage
                return response

            def close(self, **_):
                timeline.append(("recorder", "close"))

            def preserve(self):
                recorder_call.preserve()

        class Executor:
            def request(self, request, *_):
                if request["op"] == "preflight":
                    timeline.append(("executor", "preflight"))
                    return {"ok": True, "code": "PREFLIGHT_OK"}
                response = executor_call(request)
                timeline.append(("executor", request["op"]))
                if request["op"] == "plan":
                    response["data"]["operator_summary"] = {
                        "path": list(PHASES),
                        "flow": {"continuous_through": "LIFT_LIN", "next_human_hold": "POST_LIFT_SEMANTIC"},
                        "speed": {"max_velocity_scaling": 0.1},
                        "clearance": {"status": "COLLISION_CHECKED_NO_DISTANCE"},
                    }
                    if continuation:
                        response["data"]["operator_summary"]["recycle"] = {
                            "release_slot_id": run_job.canonical_digest("slot-a"), "recording_boundary_after": "LIFT_LIN",
                            "path": list(PHASES)[5:],
                            "release_target": {"place_id": "PLACE_A", "yaw_deg": 0, "x_mm": 60, "y_mm": 0},
                            "safe_staging_joint_positions_rad": [0] * 6,
                            "plan_digest": response["plan_digest"],
                        }
                    cell.plan_digest = response["plan_digest"]
                elif request["op"] == "execute":
                    cell.value.update(cell_ready=False, reason_code="EXECUTION_IN_PROGRESS")
                if continuation and response["state"] == "COMPLETED":
                    release = {
                        "schema_version": "data_factory.recycle_release_evidence.v2",
                        "release_outcome": "EXPECTED_LANDED",
                        "outcome_source": "CAMPAIGN_CONTROL_PROXY", "release_slot_id": run_job.canonical_digest("slot-a"),
                    }
                    response["data"].update(
                        release_evidence=release,
                        scene_transition={
                            "scene_state_digest": run_job.canonical_digest("recorded-release"),
                            "release_evidence_digest": run_job.canonical_digest(release),
                        },
                    )
                return response

            def close(self, **_):
                timeline.append(("executor", "close"))

        class Resource:
            def __init__(self, *_):
                pass

            def start(self):
                return self

            def set_pid(self, *_):
                return self

            def record_control_round_trip(self, _):
                pass

            def finish(self, metrics, collection_settings=None):
                return {"schema_version": "data_factory.resource_usage.v1", "sampling": {"status": "AVAILABLE"}, "recorder": metrics, "collection_settings": collection_settings}

        class Cell:
            def __init__(self):
                self.plan_digest = None
                self.value = {"robot_system_id": "fr5-lab-a", "cell_ready": True, "reason_code": "HUMAN_ACKNOWLEDGED", "run_id": "run", "plan_digest": None, "acknowledged_by": "operator"}

            def read(self):
                return {**self.value, "plan_digest": self.plan_digest}

            def mark_blocked(self, reason_code, run_id, plan_digest):
                self.value.update(cell_ready=False, reason_code=reason_code, run_id=run_id)
                self.plan_digest = plan_digest
                return self.read()

            def acknowledge_ready(self, operator, *, expected_run_id=None, expected_plan_digest=None):
                test_case.assertEqual((expected_run_id, expected_plan_digest), ("run", self.plan_digest))
                self.value.update(cell_ready=True, reason_code="HUMAN_ACKNOWLEDGED", acknowledged_by=operator)
                timeline.append(("cell", "ack"))
                return self.read()

        class Scene:
            def update_object(self, **value):
                test_case.assertEqual(value["expected_revision"], SCENE["revision"])
                timeline.append(("scene", value["state"]))
                return {"scene_state_digest": "sha256:" + "5" * 64}

        cell, scene = Cell(), Scene()
        with tempfile.TemporaryDirectory() as directory:
            live_payload = payload("live")
            live_payload.update(run_id="run", run_root=directory, dataset_root=str(Path(directory) / "dataset"))
            roots = {
                "session_id": "session-r001", "run_id": "run",
                "data_disposition": "PRODUCTION",
                "run_root": str(Path(directory).resolve()),
                "cell_root": str((Path(directory) / "cells").resolve()),
                "dataset_root": str((Path(directory) / "dataset").resolve()),
                "production_writers_enabled": True,
            }
            roots["binding_digest"] = run_job.canonical_digest(roots)
            episode = {
                "binding_digest": run_job.canonical_digest("production-episode"),
                "start_binding_digest": run_job.canonical_digest("production-start"),
                "manifest_digest": run_job.canonical_digest("manifest-commit"),
                "intent_digest": run_job.canonical_digest("intent-commit"),
                "slot_digest": run_job.canonical_digest("slot-commit"),
                "data_disposition": "PRODUCTION",
                "expires_at": "2099-01-01T00:00:00Z",
            }
            ledger_context = {
                "manifest": {"manifest_digest": episode["manifest_digest"]},
                "intent": {
                    "intent_digest": episode["intent_digest"],
                    "slot": {"slot_id": "slot-commit"},
                },
            }
            planned_start = {
                "start_binding_digest": episode["start_binding_digest"],
                "evidence_digest": run_job.canonical_digest("production-planned-start"),
                "status": "PASS",
            }
            ledger_reference = {
                "path": str(Path(directory) / "run" / "episode_ledger.json"),
                "state_path": str(Path(directory) / "run" / "episode_ledger_state.json"),
            }
            bound_ledger_reference = {**ledger_reference, "review_status": "PENDING"}

            extra = {}
            stack = ExitStack()
            self.addCleanup(stack.close)
            if continuation:
                fixture = episode_ledger_test.EpisodeLedgerTest()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                fixture.run_id = "run"
                fixture.episode_ref.update(
                    transaction_id="run:episode-000000",
                    resolved_job_digest=validated["resolved_job_digest"],
                )
                live_payload["dataset_root"] = roots["dataset_root"] = str(fixture.dataset.resolve())
                roots["binding_digest"] = run_job.canonical_digest({
                    key: value for key, value in roots.items() if key != "binding_digest"
                })
                originals = {path: path.read_bytes() for path in fixture.dataset.rglob("*") if path.is_file()}
                ledger_snapshot = {}
                terminal_error = "ROS_GRIPPER_SETTINGS_UNVERIFIED"
                binding = {
                    "parent_run_id": "run", "continuation_run_id": "run-reposition",
                    "next_run_id": "next-run", "start_state": "ON_SURFACE",
                    "binding_digest": run_job.canonical_digest("reposition-fixture"),
                }
                scope = {
                    "scope_digest": run_job.canonical_digest("preapproval-fixture"),
                    "resolved_job_digest": run_job.canonical_digest("continuation-job"),
                }
                # Setup/authorization and motion are external ports; the public
                # postcommit failure branch and candidate/state writers stay real.
                for name, value in (
                    ("validate_object_reposition_binding", binding),
                    ("_validate_parent_reposition_edge", binding),
                    ("_load_reposition_yaw_profile", None),
                    ("validate_campaign_authorization", {}),
                    ("validate_runtime_campaign_scope", {}),
                    ("_write_object_reposition_preapproval", scope),
                    ("_validate_object_reposition_preapproval", scope),
                ):
                    stack.enter_context(mock.patch.object(run_job, name, return_value=value))

                def reposition(parent_payload, checked, *_args, **_kwargs):
                    self.assertEqual((job.state, job.recorder_state, job.executor_state),
                                     ("AWAITING_CELL_READY", "COMMITTED", "COMPLETED"))
                    timeline.append(("continuation", "plan_only"))
                    return run_job._object_reposition_result(
                        parent_payload, checked, status="FAIL", code=terminal_error,
                        plan_digest=None, resolved_job_digest=None, plan_artifact_digest=None,
                        preapproval_scope_digest=scope["scope_digest"],
                        execution_response={
                            "ok": False, "code": terminal_error, "state": "BLOCKED",
                            "executor_state": "IDLE", "plan_envelope": None,
                            "execution_evidence": None,
                            "cancel_error": "CANCEL_UNCONFIRMED" if continuation == "uncertain" else None,
                            "recording_scope": "OUT_OF_DATASET",
                        },
                    )

                def checkpoint(request):
                    bound = {key: request[key] for key in (
                        "kind", "run_id", "plan_digest", "prompt", "choices", "evidence",
                    )}
                    return {
                        "kind": request["kind"], "choice": request["choices"][0],
                        "run_id": request["run_id"], "plan_digest": request["plan_digest"],
                        "checkpoint_binding_digest": run_job.canonical_digest(bound),
                        "decision_source": "CAMPAIGN_AUTHORIZATION", "operator_label": "operator",
                    }

                extra = {
                    "object_reposition_binding": binding,
                    "preapproval_checklist": {"object_reposition_binding": binding},
                    "object_reposition_source_payload": live_payload,
                    "object_reposition_call": reposition,
                    "campaign_authorization": {"fixture": "authorization"},
                    "approval_scope": "HIL_NUMERIC_PROXY", "checkpoint_provider": checkpoint,
                }

                def materialize_ledger(*_args, **_kwargs):
                    artifacts = fixture._artifacts()
                    preapproval = json.loads((Path(directory) / "run" / "preapproval_evidence.json").read_text())
                    # Reuse the ledger builder's legacy envelope fixture while
                    # joining it to the exact plan and technical result from run_live.
                    plan = {key: preapproval[key] for key in (
                        "run_id", "resolved_job_digest", "plan_digest",
                        "plan_envelope", "plan_envelope_digest",
                    )}
                    plan["schema_version"] = "data_factory.preapproval_evidence.v1"
                    artifacts["plan"] = fixture._json("parent-plan.json", plan)
                    technical_path = Path(directory) / "run" / "technical_validator.json"
                    artifacts["technical"] = {
                        "artifact_path": str(technical_path),
                        "artifact_digest": run_job.canonical_digest(json.loads(technical_path.read_text())),
                    }
                    execution = json.loads(Path(artifacts["execution"]["artifact_path"]).read_text())
                    execution["plan_digest"] = plan["plan_digest"]
                    execution["data"]["precommit_safety"] = {
                        **plan["plan_envelope"]["precommit_safety"],
                        "status": "PASS", "post_reset_safe_snapshot_digest": run_job.canonical_digest("safe"),
                    }
                    artifacts["execution"] = fixture._json("parent-execution.json", execution)
                    ledger = fixture._compile(artifacts)
                    state = episode_ledger_test.project_episode_state(ledger=ledger, reclaim_state="REPACK_REQUIRED")
                    for key, value in (("path", ledger), ("state_path", state)):
                        path = Path(ledger_reference[key])
                        path.write_text(json.dumps(value), encoding="utf-8")
                        ledger_snapshot[key] = path.read_bytes()
                    originals.update({
                        Path(ref["artifact_path"]): Path(ref["artifact_path"]).read_bytes()
                        for ref in artifacts.values()
                    })
                    originals.update({
                        path: path.read_bytes() for path in (Path(directory) / "run").iterdir()
                        if path.is_file() and path.name != "episode_ledger_state.json"
                    })
                    return ledger_reference

            def approve(request):
                return {
                    "choice": "APPROVE", "run_id": request["run_id"],
                    "plan_digest": request["plan_digest"],
                    "approval_scope": request["approval_scope"],
                    "decision_binding_digest": run_job.canonical_digest({
                        "run_id": request["run_id"],
                        "plan_digest": request["plan_digest"],
                        "approval_scope": request["approval_scope"],
                        "decision_binding": request["decision_binding"],
                    }),
                    "decision_source": "CAMPAIGN_AUTHORIZATION" if continuation else "LOCAL_UI_BUTTON", "operator_label": "operator",
                }

            with (
                mock.patch.object(run_job, "ResourceMonitor", Resource),
                mock.patch.object(run_job, "CellStateStore", return_value=cell),
                mock.patch.object(run_job, "SceneStateStore", return_value=scene),
                mock.patch.object(run_job, "validate_runtime_root_binding", return_value=roots),
                mock.patch.object(run_job, "validate_runtime_episode_binding", return_value=episode),
                mock.patch.object(run_job, "validate_runtime_planned_start", return_value=planned_start),
                mock.patch.object(run_job, "_validate_episode_ledger_context", return_value=ledger_context),
                mock.patch.object(run_job, "_write_episode_ledger", side_effect=materialize_ledger if continuation else None, return_value=ledger_reference) as ledger_writer,
                mock.patch.object(run_job, "write_candidate_admission", wraps=run_job.write_candidate_admission if continuation else None, return_value=mock.DEFAULT if continuation else {"semantic_status": "PENDING"}) as candidate_writer,
                mock.patch.object(run_job, "bind_candidate_episode_state", wraps=run_job.bind_candidate_episode_state if continuation else None, return_value=mock.DEFAULT if continuation else bound_ledger_reference) as candidate_binder,
            ):
                def invoke():
                    return run_job.run_live(
                        live_payload, threading.Event(), published.append,
                        resolver=lambda _: (validated, runtime_motion(validated, continuous=True), SCENE), executor_factory=lambda *_: Executor(), recorder_factory=lambda *_: Recorder(),
                        validator_call=lambda *_: (validator_calls.append("validator"), timeline.append(("validator", "PASS")), {"ok": True, "code": "PASS", "result_digest": "sha256:" + "6" * 64})[2],
                        tty_decision=lambda _prompt, expected: "PASS" if expected == ("PASS", "FAIL") else None,
                        camera_warmup_call=lambda *_: {"schema_version": "data_factory.camera_warmup.v1", "run_id": "run", "camera_profile": "up", "attempts": []},
                        one_job=job,
                        decision_provider=approve,
                        runtime_root_binding={"fixture": "production-root"},
                        runtime_episode_binding={"fixture": "production-episode"},
                        runtime_start_binding={"fixture": "production-start"},
                        episode_ledger_context={"fixture": "production-ledger"},
                        **extra,
                    )
                if continuation:
                    atomic_write = run_job.write_json_atomic
                    def write(path, value):
                        if Path(path).name == "candidate_admission.json" and continuation == "candidate_write":
                            raise OSError("candidate disk failure")
                        if Path(path).name == "episode_ledger_state.json" and continuation == "ledger_bind":
                            raise run_job.ContractError("EPISODE_LEDGER_STATE_WRITE_ERROR")
                        return atomic_write(path, value)
                    factories = campaign_session_test.Factories()
                    factories.fake = mock.Mock(return_value=job)
                    _, _, _, session = campaign_session_test.make_session(2, factories=factories)
                    def episode_call(_intent, lifecycle, _cancel, _context):
                        nonlocal result
                        self.assertIs(lifecycle, job)
                        result = invoke()
                        self.assertEqual((result["ok"], result["code"], result["state"]),
                                         (False, terminal_error, "BLOCKED"), result)
                        raise run_job.ContractError(result["code"])
                    result = None
                    with mock.patch.object(run_job, "write_json_atomic", side_effect=write), mock.patch.object(job, "cancel", wraps=job.cancel) as cancel_job:
                        with self.assertRaisesRegex(run_job.ContractError, terminal_error):
                            session.run_next(
                                run_id="run",
                                scene_evidence=campaign_session_test.scene(run_job.canonical_digest("scene-0")),
                                episode_call=episode_call,
                            )
                    status = session.status()
                    uncertain = continuation == "uncertain"
                    self.assertEqual(status["active_child"], uncertain)
                    self.assertEqual(job.state, "AWAITING_CELL_READY" if uncertain else "COMMITTED_BLOCKED")
                    self.assertEqual(cancel_job.call_count, int(uncertain))
                    self.assertEqual(status["termination_error"], "CAMPAIGN_SESSION_CHILD_TERMINATION_UNCERTAIN" if uncertain else None)
                    if uncertain:
                        self.assertIs(session._active, job)
                    self.assertEqual(status["campaign"]["state"], "BLOCKED")
                    with self.assertRaises(run_job.ContractError):
                        session.open_next(run_id="next-run", scene_evidence=campaign_session_test.scene(run_job.canonical_digest("scene-0")))
                    factories.fake.assert_called_once_with()
                    self.assertEqual((job.recorder_state, job.executor_state), ("COMMITTED", "COMPLETED"))
                    self.assertEqual(calls.count(("recorder", "commit")), 1)
                    self.assertEqual(calls.count(("executor", "execute")), 1)
                    self.assertEqual(timeline.count(("continuation", "plan_only")), 1)
                    self.assertLess(timeline.index(("recorder", "commit")), timeline.index(("continuation", "plan_only")))
                    self.assertEqual(validator_calls, ["validator"])
                    continuation_result = result["data"]["object_reposition"]
                    self.assertEqual(continuation_result["execution_response"]["recording_scope"], "OUT_OF_DATASET")
                    self.assertIsNone(continuation_result["plan_digest"])
                    self.assertIsNone(continuation_result["plan_artifact_digest"])
                    self.assertNotIn(("recorder", "abort"), calls)
                    self.assertNotIn(("cell", "ack"), timeline)
                    self.assertFalse(cell.read()["cell_ready"])
                    self.assertEqual((cell.read()["reason_code"], cell.read()["run_id"], cell.read()["plan_digest"]),
                                     (terminal_error, "run", result["plan_digest"]))
                    self.assertEqual(result["data"]["postcommit_cell_state"], cell.read())
                    self.assertFalse(result["data"]["training_authorized"])
                    self.assertFalse(result["data"]["camera_semantic_authority"])
                    self.assertEqual(result["data"]["technical_validator"]["status"], "PASS")
                    ledger_writer.assert_called_once()
                    candidate_writer.assert_called_once()
                    self.assertEqual(Path(ledger_reference["path"]).read_bytes(), ledger_snapshot["path"])
                    for path, before in originals.items():
                        self.assertEqual(path.read_bytes(), before, str(path))
                    candidate_path = Path(directory) / "run" / "candidate_admission.json"
                    state_path = Path(ledger_reference["state_path"])
                    review_error = {
                        "candidate_write": "CANDIDATE_ADMISSION_WRITE_ERROR",
                        "ledger_bind": "EPISODE_LEDGER_STATE_WRITE_ERROR",
                    }.get(continuation)
                    self.assertEqual(result["data"]["candidate_review_error"], review_error)
                    if continuation == "candidate_write":
                        self.assertFalse(candidate_path.exists())
                        candidate_binder.assert_not_called()
                    else:
                        candidate_binder.assert_called_once_with(ledger_reference, candidate_path)
                        candidate = json.loads(candidate_path.read_text())
                        self.assertEqual((candidate["run_id"], candidate["semantic_status"], candidate["operational_source"]), ("run", "PENDING", "HIL_PROXY"))
                    if review_error:
                        self.assertIsNone(result["data"]["candidate_review_offer"])
                        self.assertEqual(state_path.read_bytes(), ledger_snapshot["state_path"])
                        self.assertEqual(result["data"]["episode_ledger"], ledger_reference)
                    else:
                        state = json.loads(state_path.read_text())
                        self.assertEqual((state["review"]["semantic_status"], state["review"]["training_status"]), ("PENDING", "NOT_AUTHORIZED"))
                        offer = result["data"]["candidate_review_offer"]
                        self.assertEqual((offer["run_id"], offer["candidate_path"]), ("run", str(candidate_path)))
                        self.assertEqual(offer["expected_file_digest"], run_job.canonical_digest(candidate))
                        self.assertEqual(offer["expected_review_context_digest"], candidate["review_context_digest"])
                        self.assertEqual(offer["ledger_reference"], result["data"]["episode_ledger"])
                        self.assertEqual(offer["ledger_reference"]["review_status"], "PENDING")
                        self.assertEqual(offer["ledger_reference"]["training_status"], "NOT_AUTHORIZED")
                        self.assertEqual(state["candidate"], {
                            "artifact_path": str(candidate_path),
                            "artifact_digest": run_job.canonical_digest(candidate),
                        })
                        self.assertEqual(state["retention"]["reclaim_state"], "REPACK_REQUIRED")
                    return
                result = invoke()
                self.assertTrue(result["ok"], result)
                preapproval = json.loads((Path(directory) / "run" / "preapproval_evidence.json").read_text(encoding="utf-8"))

        self.assertEqual((result["ok"], result["code"], result["state"]), (True, "VALIDATED", "COMPLETE"))
        self.assertEqual(
            [item["code"] for item in published],
            [
                "PLANNING", "CAMERA_WARMUP", "AWAITING_HUMAN_APPROVAL",
                "RECORDER_STARTING", "MOTION_STARTING", "EXECUTING",
                "SEMANTIC_REVIEW", "RECYCLING", "FINALIZING", "VALIDATING",
            ],
        )
        self.assertEqual(preapproval["plan_digest"], result["plan_digest"])
        first_status = next(
            index for index, item in enumerate(timeline) if item[:2] == ("recorder", "status")
        )
        self.assertLess(timeline.index(("executor", "plan")), timeline.index(("executor", "approve")))
        self.assertLess(timeline.index(("recorder", "begin")), first_status)
        self.assertLess(first_status, timeline.index(("executor", "execute")))
        self.assertLess(timeline.index(("recorder", "commit")), timeline.index(("validator", "PASS")))
        self.assertLess(timeline.index(("recorder", "commit")), timeline.index(("scene", "ON_SURFACE")))
        self.assertEqual(validator_calls, ["validator"])
        self.assertLess(timeline.index(("recorder", "commit")), len(timeline) - 1 - timeline[::-1].index(("recorder", "close")))
        self.assertFalse(result["data"]["camera_semantic_authority"])
        self.assertFalse(result["data"]["training_authorized"])
        self.assertNotIn("rgb", result["data"])
        self.assertNotIn("frames", result["data"])
        self.assertIn(("cell", "ack"), timeline)
        ledger_writer.assert_called_once()
        candidate_writer.assert_called_once()
        candidate_binder.assert_called_once_with(
            ledger_reference, Path(directory) / "run" / "candidate_admission.json",
        )
        self.assertEqual(result["data"]["episode_ledger"], bound_ledger_reference)

    def test_human_and_ai_share_plan_only_contract_and_jsonl_live_uses_same_session(self):
        with tempfile.TemporaryDirectory() as directory:
            job_path = Path(directory) / "job.json"
            job_path.write_text(json.dumps(JOB), encoding="utf-8")
            args = SimpleNamespace(
                mode="plan_only", run_id="runner-test", job=str(job_path),
                selected_sheet="selected.json", yaw0_sheet="yaw0.json",
                config_root="config/data_factory", motion_qualification="motion.json",
                home_candidate="home.json", urdf="robot.urdf",
                expected_robot_system_id="fr5-lab-a", camera_profile=None,
                dataset_root=None, run_root=None,
            )
            self.assertEqual(run_job._human_payload(args), run_job._run_payload(payload()))
            args.job = None
            with mock.patch.object(run_job.sys, "stdin", type("TTY", (), {"isatty": lambda self: True})()), mock.patch.object(run_job, "_build_job", return_value=JOB) as builder:
                self.assertEqual(run_job._human_payload(args), run_job._run_payload(payload()))
            builder.assert_called_once_with("selected.json", "yaw0.json", "config/data_factory")

        created = []
        def resolver(_):
            return {"normalized_job": JOB, "resolved_job_digest": motion()["resolved_job_digest"]}, motion(), SCENE
        def factory(_timeout):
            value = Executor()
            created.append(value)
            return value
        session = run_job.RunSession(lambda value, cancel, publish: run_job.run_plan_only(
            value, cancel, publish, resolver=resolver, executor_factory=factory,
        ))
        with mock.patch.object(run_job, "_write_episode_ledger", side_effect=AssertionError("plan-only retained collection evidence")) as writer:
            started = session.process(command())
            self.assertEqual((started["ok"], started["state"]), (True, "RUNNING"))
            result = session.events.get(timeout=1)
            session.worker.join(1)
            writer.assert_not_called()
        self.assertEqual(set(result), run_job.EVENT_KEYS)
        self.assertEqual((result["ok"], result["state"], result["origin_op_id"]), (True, "PLANNED", "run-1"))
        self.assertEqual(result["data"]["motion_program_digest"], run_job.canonical_digest(motion()))
        self.assertEqual(
            {key: result["data"]["plan_only_checks"][key] for key in ("all_valid", "execute_goal_count", "gripper_goal_count")},
            {"all_valid": True, "execute_goal_count": 0, "gripper_goal_count": 0},
        )
        self.assertFalse(result["data"]["camera_semantic_authority"])
        self.assertEqual(created[0].transport.calls, list(PHASES))

        failed = Executor()
        failed.request = lambda request, _cancel=None: {
            "schema_version":"fr5.pickup_executor.response.v3", "mode":"PRE_LIVE",
            "op_id":request["op_id"], "op":request["op"], "ok":False,
            "code":"PLAN_NOT_COMPLETE", "run_id":None, "plan_digest":None,
            "state":"IDLE", "data":None,
        }
        rejected = run_job.run_plan_only(payload(), threading.Event(), lambda _: None, resolver=resolver, executor_factory=lambda _: failed)
        self.assertEqual((rejected["code"], rejected["state"], rejected["data"]), ("PLAN_NOT_COMPLETE", "BLOCKED", None))

        called = []
        def live_fake(value, _cancel, _publish):
            called.append(value["mode"])
            return run_job._response(ok=True, code="PLANNED", state="PLANNED", run_id=value["run_id"])
        live = run_job.RunSession(live_fake)
        started = live.process(command(value=payload("live")))
        self.assertEqual((started["ok"], started["state"], started["data"]), (True, "RUNNING", {"mode": "live"}))
        event = live.events.get(timeout=1)
        live.worker.join(1)
        self.assertEqual((called, event["event"], event["origin_op_id"], event["code"]), (["live"], "RESULT", "run-1", "PLANNED"))

    def test_jsonl_status_cancel_eof_and_exactly_one_result(self):
        started = threading.Event()
        def slow(value, cancel, publish):
            publish(run_job._response(ok=True, code="PLANNING", state="PLANNING", run_id=value["run_id"]))
            started.set()
            cancel.wait(1)
            return run_job._response(ok=True, code="PLANNED", state="PLANNED", run_id=value["run_id"])

        session = run_job.RunSession(slow)
        lines = [
            command(),
            command("status-1", "status", {"run_id": "runner-test"}),
            command("run-2"),
            command("cancel-1", "cancel", {"run_id": "runner-test", "reason_code": "OPERATOR_CANCEL"}),
        ]
        class Input:
            def __iter__(self):
                yield json.dumps(lines[0]) + "\n"
                started.wait(1)
                for value in lines[1:]:
                    yield json.dumps(value) + "\n"
                while session.worker is not None and session.worker.is_alive():
                    time.sleep(.001)
        output = __import__("io").StringIO()
        self.assertFalse(run_job.run_jsonl(Input(), output, session))
        values = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([value.get("code") for value in values[:-1]], ["RUNNING", "STATUS", "RUN_ACTIVE", "CANCEL_REQUESTED"])
        self.assertEqual((values[-1]["event"], values[-1]["origin_op_id"], values[-1]["code"]), ("RESULT", "run-1", "OPERATOR_CANCEL"))
        self.assertEqual(sum(value.get("event") == "RESULT" for value in values), 1)
        self.assertEqual(set(values[0]), run_job.RESPONSE_KEYS)
        self.assertEqual(set(values[-1]), run_job.EVENT_KEYS)
        self.assertEqual(session.process(command("late", "status", {"run_id": "runner-test"}))["state"], "CANCELLED")
        self.assertEqual(session.process(command("again"))["code"], "ONE_JOB_ONLY")

        eof_started = threading.Event()
        eof_session = run_job.RunSession(lambda value, cancel, publish: (eof_started.set(), cancel.wait(1), run_job._response(ok=True, code="PLANNED", state="PLANNED", run_id=value["run_id"]))[-1])
        eof_session.process(command())
        eof_started.wait(1)
        self.assertTrue(eof_session.input_closed())
        eof = eof_session.events.get(timeout=1)
        eof_session.worker.join(1)
        self.assertEqual((eof["code"], eof["state"]), ("INPUT_EOF", "CANCELLED"))

        success = run_job.RunSession(lambda value, _cancel, _publish: run_job._response(ok=True, code="PLANNED", state="PLANNED", run_id=value["run_id"]))
        class CompleteInput:
            def __iter__(self):
                yield json.dumps(command()) + "\n"
                while success.worker is None:
                    time.sleep(.001)
                success.worker.join(1)
        completed = __import__("io").StringIO()
        self.assertTrue(run_job.run_jsonl(CompleteInput(), completed, success))
        self.assertEqual(sum(json.loads(line).get("event") == "RESULT" for line in completed.getvalue().splitlines()), 1)

        stubborn = JsonlProcess([sys.executable, "-u", "-c", "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"], timeout_s=.03)
        with self.assertRaises(run_job.ContractError) as raised:
            stubborn.close()
        self.assertEqual(raised.exception.code, "JSONL_EXIT_TIMEOUT")
        self.assertIsNotNone(stubborn.process.poll())

        exited = JsonlProcess([sys.executable, "-u", "-c", "pass"])
        exited.process.wait(1)
        self.assertEqual(exited.close(), 0)

        processes = []
        def hanging_factory(_timeout):
            value = JsonlProcess([sys.executable, "-u", "-c", "import sys,time;sys.stdin.readline();time.sleep(30)"], timeout_s=.05)
            processes.append(value)
            return value
        hanging = run_job.RunSession(lambda value, cancel, publish: run_job.run_plan_only(
            value, cancel, publish,
            resolver=lambda _: ({"normalized_job": JOB, "resolved_job_digest": motion()["resolved_job_digest"]}, motion(), SCENE),
            executor_factory=hanging_factory,
        ))
        hanging.process(command())
        while not processes:
            time.sleep(.001)
        hanging.process(command("stop", "cancel", {"run_id": "runner-test", "reason_code": "OPERATOR_CANCEL"}))
        stopped = hanging.events.get(timeout=1)
        hanging.worker.join(1)
        self.assertEqual((stopped["code"], stopped["state"]), ("OPERATOR_CANCEL", "CANCELLED"))
        self.assertIsNotNone(processes[0].process.poll())

        failure_ready, failure_release = threading.Event(), threading.Event()
        def failing(value, _cancel, _publish):
            failure_ready.set()
            failure_release.wait(1)
            return run_job._response(code="PLAN_NOT_COMPLETE", state="BLOCKED", run_id=value["run_id"])
        raced = run_job.RunSession(failing)
        raced.process(command())
        failure_ready.wait(1)
        raced.process(command("race-cancel", "cancel", {"run_id": "runner-test", "reason_code": "OPERATOR_CANCEL"}))
        failure_release.set()
        preserved = raced.events.get(timeout=1)
        raced.worker.join(1)
        self.assertEqual((preserved["code"], preserved["state"]), ("PLAN_NOT_COMPLETE", "BLOCKED"))

        interrupted = Executor()
        interrupted.closed_with = None
        interrupted.request = lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
        interrupted.close = lambda timeout_s=None: setattr(interrupted, "closed_with", timeout_s)
        cancelled = threading.Event()
        with self.assertRaises(KeyboardInterrupt):
            run_job.run_plan_only(
                payload(), cancelled, lambda _: None,
                resolver=lambda _: ({"normalized_job": JOB, "resolved_job_digest": motion()["resolved_job_digest"]}, motion(), SCENE),
                executor_factory=lambda _: interrupted,
            )
        self.assertTrue(cancelled.is_set())
        self.assertEqual(interrupted.closed_with, 1.0)

        class BrokenInput:
            def __iter__(self):
                raise OSError("broken")
        broken = __import__("io").StringIO()
        self.assertFalse(run_job.run_jsonl(BrokenInput(), broken))
        self.assertEqual(json.loads(broken.getvalue())["code"], "CONTROL_INPUT_FAILED")

    def test_runtime_cleanup_releases_preserved_child_without_leaving_a_process(self):
        calls = []

        class PreservedChild:
            preserved = True

            def release(self, timeout_s=None):
                calls.append(("release", timeout_s))

            def close(self, **_):
                raise AssertionError("preserved child must use explicit release")

        cancelled = threading.Event()
        cancelled.set()
        run_job._close_runtime_child(PreservedChild(), cancelled)
        run_job._close_runtime_child(PreservedChild(), threading.Event())
        self.assertEqual(calls, [("release", 1.0), ("release", 1.0)])

    def test_fake_commit_validator_and_async_boundaries_reuse_existing_core(self):
        helper = one_job_test.OneJobTest()
        self.addCleanup(helper.doCleanups)
        recorder = ["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"]
        executor = ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"]
        job, calls = helper.make(recorder, executor)
        def fake(value, _cancel, publish):
            result = run_one_job(
                job, one_job_test.PLAN, one_job_test.MOTION_APPROVAL,
                lambda state, _: {"PRECONTACT_HUMAN": "CONFIRM", "GRASP_VERDICT": "PASS", "SEMANTIC_VERDICT": "PASS", "AWAITING_CELL_READY": "READY"}[state],
                operator_id="operator", poll_interval_s=.001, sleep=lambda _: None,
            )
            if result["state"] == "COMPLETE":
                calls.append(("validator", "PASS"))
            return run_job._response(ok=result["ok"], code="VALIDATED" if result["ok"] else result["code"], state=result["state"], run_id=value["run_id"], plan_digest=result["plan_digest"], data={"validator": "PASS"})
        session = run_job.RunSession(fake)
        session.process(command())
        result = session.events.get(timeout=1)
        session.worker.join(1)
        self.assertEqual((result["state"], result["data"]), ("COMPLETE", {"validator": "PASS"}))
        self.assertLess(calls.index(("recorder", "commit")), calls.index(("validator", "PASS")))

        delayed_job, delayed_calls = helper.make(["RECORDING", "RECORDING"], ["PLANNED", "APPROVED", "EXECUTING", "EXECUTING"])
        helper.prepare_and_start(delayed_job)
        original = delayed_job.recorder_call
        status_entered, data_progress, release = threading.Event(), threading.Event(), threading.Event()
        def delayed(request):
            if request["op"] == "status":
                status_entered.set()
                release.wait(1)
            return original(request)
        delayed_job.recorder_call = delayed
        data_thread = threading.Thread(target=lambda: (status_entered.wait(1), data_progress.set(), release.set()))
        data_thread.start()
        self.assertTrue(delayed_job.poll()["ok"])
        data_thread.join(1)
        self.assertTrue(data_progress.is_set())
        self.assertIn(("executor", "heartbeat"), delayed_calls)

        status_process = JsonlProcess(
            [sys.executable, "-u", "-c", "import sys,time;sys.stdin.readline();time.sleep(30)"],
            timeout_s=.03,
        )
        timed, timed_calls = helper.make(["RECORDING"], ["PLANNED", "APPROVED", "EXECUTING", "BLOCKED"])
        helper.prepare_and_start(timed)
        timed.recorder_call = status_process
        started = time.monotonic()
        observed = timed.poll()
        self.assertLess(time.monotonic() - started, .2)
        self.assertEqual((observed["code"], observed["state"], observed["recorder_state"]), ("RECORDER_STATUS_TIMEOUT", "BLOCKED", "STATUS_UNCERTAIN"))
        self.assertIn(("executor", "cancel"), timed_calls)
        with self.assertRaises(run_job.ContractError):
            status_process.release()
        self.assertIsNotNone(status_process.process.poll())

        for elapsed, expected in ((.4999, (True, "EXECUTING")), (.5, (False, "RECORDER_HEALTH_STALE")), (.5001, (False, "RECORDER_HEALTH_STALE"))):
            with self.subTest(status_round_trip_s=elapsed):
                recorder_states = ["RECORDING", "RECORDING"] if elapsed < .5 else ["RECORDING", "RECORDING", "ABORTED"]
                executor_states = ["PLANNED", "APPROVED", "EXECUTING", "EXECUTING"] if elapsed < .5 else ["PLANNED", "APPROVED", "EXECUTING", "BLOCKED"]
                bounded, _ = helper.make(recorder_states, executor_states)
                helper.prepare_and_start(bounded)
                ticks = iter((0, elapsed))
                bounded.monotonic_clock = lambda: next(ticks)
                observed = bounded.poll()
                self.assertEqual((observed["ok"], observed["state"] if observed["ok"] else observed["code"]), expected)

        source = Path(run_job.__file__).read_text(encoding="utf-8")
        for forbidden in ("sensor_msgs", "cv2", "pyarrow", "save_episode", "add_frame", "PREGRASP_PTP", "GRIPPER_CLOSE"):
            self.assertNotIn(forbidden, source)

    def test_two_episode_campaign_reuses_scene_cas_and_stops_before_any_later_run(self):
        def manifest():
            first, second = payload("live"), payload("live")
            job = {
                "schema_version": "data_factory.job.v1", "task": "pickup_e2e", "robot_system_id": "fr5-lab-a",
                "collection_profile_id": "test", "place_id": "place-a", "cell_calibration_id": "cell-r1",
                "sheet_manifest_digest": "sha256:" + "1" * 64, "yaw_deg": 0, "y_mm": 0,
                "object_profile_id": "wood-cube", "grasp_profile_id": "grasp-r1", "instruction": "pick up",
                "episode_intent": "nominal pickup", "operator_or_agent_id": "operator",
                "approval_expiry": "2099-01-01T00:00:00Z", "dry_run_required": True,
            }
            first.update(run_id="run-a", recycle_x_mm=0, recycle_y_mm=0, job={**job, "job_id": "run-a", "x_mm": -60})
            second.update(run_id="run-b", recycle_x_mm=60, recycle_y_mm=0, job={**job, "job_id": "run-b", "x_mm": 0})
            return {
                "schema_version": "data_factory.campaign.v1", "campaign_id": "campaign-1", "max_episodes": 2,
                "episodes": [
                    {"run": first, "release_role": "DESTINATION_THEN_NEXT_SOURCE"},
                    {"run": second, "release_role": "RELEASE_DESTINATION"},
                ],
            }

        def execute(fail_at=None, stale=False, cancelled=False, quality_reject_first=False, reject_before_landing=False):
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            root = Path(directory.name) / "cells"
            store = run_job.SceneStateStore(root, "fr5-lab-a")
            store.update_object(
                instance_id="cube-1", object_profile_id="wood-cube", state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": -60, "y_mm": 0},
                source="HUMAN", updated_by="operator", expected_revision=0,
            )
            calls, timeline, prompts = [], [], []

            def episode_call(value, _cancel, _publish, role, next_run_id, source_slot, before_approval):
                run_id = value["run_id"]
                calls.append(run_id)
                timeline.append(("plan", run_id))
                if source_slot is not None:
                    plan_digest = "sha256:" + "c" * 64
                    planned_scene = store.snapshot()
                    before_approval(
                        f"Plan {plan_digest}",
                        {"plan_digest": plan_digest, "plan_envelope": {"plan": {"scene_binding": {
                            "scene_state_digest": planned_scene["scene_state_digest"],
                            "revision": planned_scene["scene_state"]["revision"],
                            "object_instance_id": "cube-1", "source_slot": source_slot,
                        }}}},
                    )
                    if stale:
                        store.update_object(
                            instance_id="cube-1", object_profile_id="wood-cube", state="ON_SURFACE",
                            pose=planned_scene["scene_state"]["objects"]["cube-1"]["pose"], source="HUMAN",
                            updated_by="operator", expected_revision=planned_scene["scene_state"]["revision"],
                        )
                    store.consume_next_source(
                        slot_id=source_slot["slot_id"], run_id=run_id,
                        expected_scene_digest=planned_scene["scene_state_digest"],
                        expected_slot_digest=source_slot["slot_digest"],
                    )
                timeline.append(("start", run_id))
                snapshot = store.snapshot()
                slot = run_job.release_slot(
                    robot_system_id="fr5-lab-a",
                    pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": value["recycle_x_mm"], "y_mm": value["recycle_y_mm"]},
                    object_profile_id="wood-cube", exclusion_geometry_digest="sha256:" + "e" * 64, role=role,
                )
                evidence = {
                    "schema_version": "data_factory.recycle_release_evidence.v1", "run_id": run_id,
                    "plan_digest": "sha256:" + ("a" if run_id == "run-a" else "b") * 64,
                    "release_slot_id": slot["slot_id"], "expected_scene_state_digest": snapshot["scene_state_digest"],
                    "expected_scene_revision": snapshot["scene_state"]["revision"], "gripper_reference_m": 0.021,
                    "gripper_feedback_m": 0.021,
                    "terminal_phases": ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"],
                    "post_retreat_snapshot_digest": "sha256:" + "4" * 64, "next_start_tolerance_rad": 0.01,
                    "human_verdict": "LANDED",
                }
                if reject_before_landing and run_id == "run-a":
                    return run_job._response(
                        code="QUALITY_REJECTED", state="ABORTED", run_id=run_id,
                        plan_digest=evidence["plan_digest"],
                    )
                transition = store.transition_release(
                    instance_id="cube-1", release_slot=slot, evidence=evidence, updated_by="pickup-executor",
                    expected_digest=snapshot["scene_state_digest"], expected_revision=snapshot["scene_state"]["revision"],
                    allowed_next_run_id=next_run_id,
                )
                if quality_reject_first and run_id == "run-a":
                    return run_job._response(
                        code="QUALITY_REJECTED", state="ABORTED", run_id=run_id,
                        plan_digest=evidence["plan_digest"],
                    )
                if fail_at == len(calls):
                    return run_job._response(code="STORAGE_REFERENCE_ERROR", state="BLOCKED", run_id=run_id)
                timeline.append(("technical-pass", run_id))
                return run_job._response(
                    ok=True, code="VALIDATED", state="COMPLETE", run_id=run_id,
                    data={
                        "technical_validator": {"run_id": run_id, "status": "PASS"},
                        "operator_summary": {"recycle": {"release_slot_id": slot["slot_id"]}},
                        "postcommit_scene_state_digest": transition["scene_state_digest"],
                        "training_authorized": False,
                    },
                )

            cancel = threading.Event()
            if cancelled:
                cancel.set()
            result = run_job.run_campaign(
                manifest(), cancel, lambda _: None, episode_call=episode_call,
                scene_store_factory=lambda *_: store,
                tty_decision=lambda prompt, expected: (prompts.append((prompt, expected)), timeline.append(("approval", expected))),
            )
            return result, store, calls, timeline, prompts

        result, store, calls, timeline, prompts = execute()
        self.assertEqual((result["ok"], result["code"], result["state"], calls), (True, "CAMPAIGN_COMPLETE", "COMPLETE", ["run-a", "run-b"]))
        self.assertLess(timeline.index(("technical-pass", "run-a")), timeline.index(("plan", "run-b")))
        self.assertLess(timeline.index(("approval", f"LANDED_AND_APPROVE_NEXT {result['data']['next_plan_digest']}")), timeline.index(("start", "run-b")))
        self.assertEqual([expected for _, expected in prompts], [f"LANDED_AND_APPROVE_NEXT {result['data']['next_plan_digest']}"])
        self.assertTrue(all("review" not in prompt.lower() and "semantic" not in prompt.lower() for prompt, _ in prompts))
        slots = store.snapshot()["scene_state"]["slot_allocations"].values()
        self.assertEqual(sorted(item["state"] for item in slots), ["CONSUMED_PENDING_REVIEW", "CONSUMED_PENDING_REVIEW"])
        self.assertFalse(result["data"]["training_authorized"])

        partial, _, calls, timeline, prompts = execute(quality_reject_first=True)
        self.assertEqual((partial["ok"], partial["code"], partial["state"], calls), (False, "CAMPAIGN_PARTIAL", "COMPLETE", ["run-a", "run-b"]))
        self.assertEqual(partial["data"]["episodes"][0]["code"], "QUALITY_REJECTED")
        self.assertEqual([expected for _, expected in prompts], [f"LANDED_AND_APPROVE_NEXT {partial['data']['next_plan_digest']}"])
        self.assertLess(timeline.index(("approval", f"LANDED_AND_APPROVE_NEXT {partial['data']['next_plan_digest']}")), timeline.index(("start", "run-b")))
        self.assertFalse(partial["data"]["training_authorized"])
        unsafe, _, calls, _, prompts = execute(reject_before_landing=True)
        self.assertEqual((unsafe["code"], unsafe["state"], calls, prompts), ("QUALITY_REJECTED", "ABORTED", ["run-a"], []))

        for fail_at, expected_calls in ((1, ["run-a"]), (2, ["run-a", "run-b"])):
            with self.subTest(fail_at=fail_at):
                failed, _, calls, _, _ = execute(fail_at=fail_at)
                self.assertEqual((failed["ok"], failed["code"], calls), (False, "STORAGE_REFERENCE_ERROR", expected_calls))
        stale, _, calls, timeline, _ = execute(stale=True)
        self.assertEqual((stale["ok"], stale["code"], calls), (False, "SCENE_STATE_CHANGED", ["run-a", "run-b"]))
        self.assertNotIn(("start", "run-b"), timeline)
        cancelled, _, calls, _, _ = execute(cancelled=True)
        self.assertEqual((cancelled["ok"], cancelled["code"], calls), (False, "CANCELLED", []))

        invalid = manifest()
        invalid["max_episodes"] = 3
        with self.assertRaisesRegex(run_job.ContractError, "CAMPAIGN_SCHEMA"):
            run_job._campaign_manifest(invalid)
        normalized = run_job._campaign_manifest(manifest())
        base_slot = run_job.release_slot(
            robot_system_id="fr5-lab-a", pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": 0, "y_mm": 0},
            object_profile_id="wood-cube", exclusion_geometry_digest="sha256:" + "e" * 64,
        )
        observed = {}
        def inspect_live(value, cancel, publish, *, resolver, before_approval):
            _, _, observed["binding"] = resolver(value)
            return run_job._response(ok=True, code="VALIDATED", state="COMPLETE", run_id=value["run_id"])
        with mock.patch.object(run_job, "resolve_inputs", return_value=({}, {}, {**SCENE, "release_slot": base_slot})), mock.patch.object(run_job, "run_live", side_effect=inspect_live):
            run_job._campaign_episode(
                normalized["episodes"][0]["run"], threading.Event(), lambda _: None,
                "DESTINATION_THEN_NEXT_SOURCE", "run-b",
            )
        self.assertEqual((observed["binding"]["release_slot"]["role"], observed["binding"]["allowed_next_run_id"]), ("DESTINATION_THEN_NEXT_SOURCE", "run-b"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(json.dumps(manifest()), encoding="utf-8")
            output = __import__("io").StringIO()
            expected = run_job._response(ok=True, code="CAMPAIGN_COMPLETE", state="COMPLETE", run_id="campaign-1", data={})
            order = []
            def campaign_finished(*_):
                order.append("children-closed")
                return expected
            def review_after(*_):
                self.assertEqual(order, ["children-closed"])
                order.append("review")
                return []
            with mock.patch.object(run_job, "run_campaign", side_effect=campaign_finished) as called, mock.patch.object(run_job, "_campaign_candidate_reviews", side_effect=review_after) as reviewed, mock.patch.object(run_job.sys, "stdout", output):
                self.assertEqual(run_job.main(("campaign", "--manifest", str(path))), 0)
            self.assertEqual(json.loads(output.getvalue()), expected)
            self.assertEqual(called.call_args.args[0]["schema_version"], run_job.CAMPAIGN_SCHEMA)
            self.assertEqual(order, ["children-closed", "review"])
            reviewed.assert_called_once()
            output = __import__("io").StringIO()
            pending = [{"run_id": "run-a", "path": "/run-a/candidate_admission.json", "file_digest": "sha256:" + "4" * 64, "semantic_status": "PENDING"}]
            with mock.patch.object(run_job, "_campaign_candidate_reviews", return_value=pending), mock.patch.object(run_job.sys, "stdout", output):
                self.assertEqual(run_job.main(("review", "--campaign", str(path))), 0)
            reviewed_result = json.loads(output.getvalue())
            self.assertEqual((reviewed_result["code"], reviewed_result["data"]), ("CANDIDATE_SEMANTIC_PENDING", {"candidate_admissions": pending, "training_authorized": False}))
            output = __import__("io").StringIO()
            with mock.patch.object(run_job, "run_campaign", return_value=run_job._response(ok=True, code="CAMPAIGN_COMPLETE", state="COMPLETE", run_id="campaign-1", data={})), mock.patch.object(run_job, "_campaign_candidate_reviews", return_value=pending), mock.patch.object(run_job.sys, "stdout", output):
                self.assertEqual(run_job.main(("campaign", "--manifest", str(path))), 0)
            self.assertEqual(json.loads(output.getvalue())["code"], "CANDIDATE_SEMANTIC_PENDING")

    def test_candidate_review_is_one_shot_digest_bound_and_tty_only(self):
        now = run_job.datetime(2026, 8, 21, 12, 0, tzinfo=run_job.timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def files(run_id):
                run_dir = root / run_id
                run_dir.mkdir()
                technical = {
                    "schema_version": "data_factory.technical_validator_result.v1", "run_id": run_id,
                    "resolved_job_digest": "sha256:" + "1" * 64, "plan_digest": "sha256:" + "2" * 64,
                    "dataset_root": str(root / "dataset"), "expected_fps": 30, "status": "PASS",
                    "result_digest": "sha256:" + "3" * 64,
                }
                context = run_job.canonical_digest({
                    "run_id": run_id, "resolved_job_digest": technical["resolved_job_digest"],
                    "plan_digest": technical["plan_digest"], "technical_validator_digest": run_job.canonical_digest(technical),
                })
                admission = {
                    "schema_version": "data_factory.candidate_admission.v1", "run_id": run_id,
                    "operational_gate": "PASS", "operational_source": "HUMAN_GATED", "checklist_id": "pickup-v2",
                    "review_context_digest": context, "semantic_status": "PENDING",
                    "reviewed_by": None, "reviewed_at": None, "reason": None,
                }
                (run_dir / "technical_validator.json").write_text(json.dumps(technical), encoding="utf-8")
                path = run_dir / "candidate_admission.json"
                path.write_text(json.dumps(admission), encoding="utf-8")
                return path, admission, context

            first_path, first, first_context = files("run-a")
            second_path, second, _ = files("run-b")
            reviewed = run_job.review_candidate_admission(
                first_path, expected_file_digest=run_job.canonical_digest(first),
                expected_review_context_digest=first_context, checklist_id="pickup-v2",
                semantic_status="PASS", reviewed_by="operator", clock=lambda: now,
            )
            self.assertEqual((reviewed["semantic_status"], reviewed["reviewed_by"], reviewed["reviewed_at"], reviewed["reason"]), ("PASS", "operator", "2026-08-21T12:00:00Z", None))
            retried = run_job.review_candidate_admission(
                first_path,
                expected_file_digest=run_job.canonical_digest(reviewed),
                expected_review_context_digest=first_context,
                checklist_id="pickup-v2", semantic_status="PASS",
                reviewed_by="operator",
                clock=lambda: (_ for _ in ()).throw(AssertionError("clock reused")),
            )
            self.assertEqual(retried, reviewed)
            with self.assertRaisesRegex(run_job.ContractError, "CANDIDATE_REVIEW_STATE"):
                run_job.review_candidate_admission(
                    first_path, expected_file_digest=run_job.canonical_digest(reviewed),
                    expected_review_context_digest=first_context, checklist_id="pickup-v2",
                    semantic_status="FAIL", reviewed_by="operator", reason="TASK_GOAL",
                )
            for changes, code in (
                ({"expected_file_digest": "sha256:" + "9" * 64}, "CANDIDATE_REVIEW_FILE_CHANGED"),
                ({"expected_review_context_digest": "sha256:" + "8" * 64}, "CANDIDATE_REVIEW_STATE"),
                ({"checklist_id": "pickup-v1"}, "CANDIDATE_REVIEW_SCHEMA"),
                ({"reviewed_by": "HUMAN"}, "CANDIDATE_REVIEW_SCHEMA"),
                ({"reason": "CALLER_SPECIFIC_REASON"}, "CANDIDATE_REVIEW_SCHEMA"),
            ):
                arguments = {
                    "expected_file_digest": run_job.canonical_digest(second),
                    "expected_review_context_digest": second["review_context_digest"], "checklist_id": "pickup-v2",
                    "semantic_status": "UNCERTAIN", "reviewed_by": "operator", "reason": "UNKNOWN",
                }
                with self.subTest(code=code), self.assertRaisesRegex(run_job.ContractError, code):
                    run_job.review_candidate_admission(second_path, **{**arguments, **changes})
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), second)

            campaign = {"campaign_id": "campaign-1", "episodes": [
                {"run": {"run_id": "run-a", "run_root": str(root), "job": {
                    "task": "pickup_e2e", "operator_or_agent_id": "operator",
                }}},
                {"run": {"run_id": "run-b", "run_root": str(root), "job": {
                    "task": "pickup_e2e", "operator_or_agent_id": "operator",
                }}},
            ]}
            choices = iter(("SKIP",))
            skipped = run_job._campaign_candidate_reviews(campaign, tty_decision=lambda *_: next(choices))
            self.assertEqual(skipped[-1]["semantic_status"], "PENDING")
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), second)
            interrupted = run_job._campaign_candidate_reviews(
                campaign, tty_decision=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
            )
            self.assertEqual(interrupted[-1]["semantic_status"], "PENDING")
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), second)
            missing_tty = run_job._campaign_candidate_reviews(
                campaign, tty_decision=lambda *_: (_ for _ in ()).throw(run_job.ContractError("HUMAN_TTY_REQUIRED")),
            )
            self.assertEqual((len(missing_tty), missing_tty[-1]["semantic_status"]), (2, "PENDING"))
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), second)
            with self.assertRaisesRegex(run_job.ContractError, "CANDIDATE_REVIEW_IO"):
                run_job._campaign_candidate_reviews(
                    campaign, tty_decision=lambda *_: (_ for _ in ()).throw(run_job.ContractError("CANDIDATE_REVIEW_IO")),
                )

            forged_file = {**second, "semantic_status": "PASS", "reviewed_by": "HUMAN", "reviewed_at": "2026-08-21T12:00:00Z"}
            second_path.write_text(json.dumps(forged_file), encoding="utf-8")
            with self.assertRaisesRegex(run_job.ContractError, "CANDIDATE_REVIEW_STATE"):
                run_job._campaign_candidate_reviews(campaign, tty_decision=lambda *_: "SKIP")
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), forged_file)

            forged = run_job.RunSession().process(command("review-1", "review", {
                "run_id": "run-b", "reviewed_by": "HUMAN", "semantic_status": "PASS",
            }))
            self.assertEqual(forged["code"], "COMMAND_SCHEMA")
            self.assertEqual(json.loads(second_path.read_text(encoding="utf-8")), forged_file)

    def test_campaign_failure_matrix_never_starts_the_later_condition_or_leaks_fake_resources(self):
        first, second = payload("live"), payload("live")
        job = {
            "schema_version": "data_factory.job.v1", "task": "pickup_e2e", "robot_system_id": "fr5-lab-a",
            "collection_profile_id": "test", "place_id": "place-a", "cell_calibration_id": "cell-r1",
            "sheet_manifest_digest": "sha256:" + "1" * 64, "yaw_deg": 0, "y_mm": 0,
            "object_profile_id": "wood-cube", "grasp_profile_id": "grasp-r1", "instruction": "pick up",
            "episode_intent": "nominal pickup", "operator_or_agent_id": "operator",
            "approval_expiry": "2099-01-01T00:00:00Z", "dry_run_required": True,
        }
        first.update(run_id="run-a", recycle_x_mm=0, recycle_y_mm=0, job={**job, "job_id": "run-a", "x_mm": -60})
        second.update(run_id="run-b", recycle_x_mm=60, recycle_y_mm=0, job={**job, "job_id": "run-b", "x_mm": 0})
        campaign = {
            "schema_version": "data_factory.campaign.v1", "campaign_id": "campaign-fault", "max_episodes": 2,
            "episodes": [
                {"run": first, "release_role": "DESTINATION_THEN_NEXT_SOURCE"},
                {"run": second, "release_role": "RELEASE_DESTINATION"},
            ],
        }
        for code, recorder, goal in (
            ("CANCELLED", 0, 0), ("JOB_EXPIRED", 0, 0), ("DISK_RESERVE", 0, 0),
            ("GRIPPER_AMBIGUOUS", 1, 1), ("RELEASE_UNCONFIRMED", 1, 1),
            ("SCENE_STATE_CHANGED", 0, 0), ("QUARANTINED_COMMIT", 1, 1),
        ):
            counts = {run_id: {name: 0 for name in ("plan", "recorder", "goal")} for run_id in ("run-a", "run-b")}
            open_resources = {name: 0 for name in ("child", "thread", "fd")}

            def fail(value, _cancel, _publish, _role, _next_run, _source_slot, _before_approval):
                run_id = value["run_id"]
                counts[run_id]["plan"] += 1
                for name in open_resources:
                    open_resources[name] += 1
                try:
                    counts[run_id]["recorder"] += recorder
                    counts[run_id]["goal"] += goal
                    return run_job._response(code=code, state="CANCELLED" if code == "CANCELLED" else "BLOCKED", run_id=run_id)
                finally:
                    for name in open_resources:
                        open_resources[name] -= 1

            with self.subTest(code=code):
                result = run_job.run_campaign(campaign, threading.Event(), lambda _: None, episode_call=fail)
                self.assertEqual((result["ok"], result["code"]), (False, code))
                self.assertEqual(counts["run-b"], {"plan": 0, "recorder": 0, "goal": 0})
                self.assertEqual(open_resources, {"child": 0, "thread": 0, "fd": 0})


if __name__ == "__main__":
    unittest.main()
