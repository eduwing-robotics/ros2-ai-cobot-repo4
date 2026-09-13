"""Synthetic native integration; only temporary data, no ROS node, model or GPU."""
import base64
import copy
import hashlib
import math
import threading
import time
import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from tools.fr5_data_factory import ContractError, canonical_digest, validate_motion_program
from tools.data_factory.learned_action_adapter import fake_rgb
from tools.data_factory.rollout.finite_plan import (
    FinitePolicyInference, JOINTS, compile_program, validate_proposal,
    _limits, GRIPPER_ENDPOINT_PROJECTION_QUANTA,
)
from tools.data_factory.motion.pickup_executor import PickupExecutor
from tools.data_factory.one_job import OneJob
from tools.data_factory.run_job import learned_run_diagnostic, _operator_summary
from tests.data_factory.test_motion import T, snapshot, SCENE
from tests.data_factory.operator.fixtures import motion

XML = '<robot name="synthetic">' + ''.join(
    f'<joint name="{name}" type="{"prismatic" if i == 6 else "revolute"}"><limit lower="{0 if i == 6 else -3}" upper="{.02 if i == 6 else 3}" velocity="{.1 if i == 6 else 10}"/></joint>'
    for i, name in enumerate(JOINTS)) + '</robot>'
CHECKPOINT = {"tree_digest": canonical_digest("synthetic-weights"),
              "training_receipt_digest": canonical_digest("synthetic-receipt"), "runtime": "SYNTHETIC_TEST_ONLY"}
INITIAL = [0.] * 6 + [.01]
ACTION = [.001] * 6 + [.0101]
OPTIONS = dict(instruction="synthetic probe", robot_description=XML, period_s=.1)
APPROVAL = {"source": "HUMAN", "approval_id": "synthetic-approval", "approved_by": "operator",
            "approval_expiry": "2099-01-01T00:00:00Z", "approval_scope": "HUMAN_GATED"}


def task_grant(src, scene, proposal, **changes):
    from tools.data_factory.rollout.task_authority import task_scope
    grant = {"schema_version": "data_factory.learned_task_grant.v1", "grant_id": "explicit-task",
             "issued_by": "operator", "run_id": "run", "scope": task_scope(src, scene, proposal),
             "deadline_s": 100., "terminal_reserve_s": 10., "max_outputs": 2, "revoked": False}
    grant.update(changes)
    grant["grant_digest"] = canonical_digest(grant)
    return grant


def observation():
    return {"source_clock": "SYSTEM_TIME", "source_timestamps_s": {key: 10. for key in ("state", "camera1", "camera2")},
            "observation.state": INITIAL[:], "observation.images.camera1": fake_rgb(), "observation.images.camera2": fake_rgb()}


def source():
    value = motion()
    value["binding_digests"]["robot_description_digest"] = 'sha256:' + hashlib.sha256(XML.encode()).hexdigest()
    for step in value["steps"]:
        step["limits"]["execution_timeout_s"] = 4.
    return value


def proposal():
    return FinitePolicyInference(lambda _: [ACTION[:]], CHECKPOINT, source_clock=lambda: 10.).propose(observation(), **OPTIONS)


def cross_workspace_source():
    value = source()
    destination = copy.deepcopy(value["binding_digests"])
    destination.update(cell_calibration=canonical_digest("cell-b"),
                       motion_qualification=canonical_digest("qualification-b"))
    endpoints = [
        {"workspace_id": place, "cell_calibration_id": cell,
         "cell_calibration_digest": bindings["cell_calibration"],
         "motion_recipe_digest": bindings["motion_qualification"]}
        for place, cell, bindings in (
            ("PLACE_A", "cal-a", value["binding_digests"]),
            ("PLACE_B", "cal-b", destination),
        )
    ]
    value.update(schema_version="fr5.motion_program.v4",
                 destination_resolved_job_digest=canonical_digest("destination-job"),
                 destination_binding_digests=destination,
                 endpoint_bindings=endpoints,
                 endpoint_bindings_digest=canonical_digest(endpoints))
    for step in value["steps"]:
        if step.get("pause_after") == "SEMANTIC_VERDICT":
            step.pop("pause_after")
        if step["phase"] == "RETREAT_LIN":
            step["pause_after"] = "SEMANTIC_VERDICT"
    return validate_motion_program(value)


def redigest(p):
    p["proposal_digest"] = canonical_digest({k: v for k, v in p.items() if k != "proposal_digest"})
    return p


class Transport(T):
    def __init__(self):
        super().__init__()
        self.sent = []
        self.active = False
        self.cancel_count = 0
        self.current = INITIAL[:]
        self.failure = None
        self.on_start = None
        self.hardware = False
        self.source_clock = lambda: 10.

    @property
    def owns_active_goal(self):
        return self.active

    def snapshot(self, *_):
        value = snapshot(self.current[:6], gripper_position=self.current[-1])
        value["gripper_controller"]["reference_position_m"] = self.current[-1]
        if self.hardware:
            from control_msgs.msg import DynamicJointState, InterfaceValue, JointTrajectoryControllerState
            from rclpy.serialization import serialize_message, deserialize_message
            from tools.data_factory.rollout.gripper_evidence import FIELDS, RESOURCE, CALENDAR, decode_dynamic_state
            from tools.data_factory.motion.moveit_transport import RosMoveItTransport
            now = self.source_clock()
            value["joint_state_stamp_ns"] = round(now * 1e9)
            wire = dict.fromkeys(FIELDS, 0.)
            wire.update(version=1., incarnation_0=1., incarnation_1=2., incarnation_2=3., incarnation_3=4.,
                        sample_system_s=now, sample_steady_s=now, raw_reference_m=self.current[-1],
                        feedback_m=self.current[-1], arm_resumed=1., valid=1.)
            date = datetime.fromtimestamp(now - .002, timezone.utc)
            wire.update(zip(CALENDAR, [date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond // 1000]))
            names = FIELDS
            if getattr(self, "hardware_current", False):
                from tools.data_factory.rollout.gripper_evidence import LIVE_FIELDS, CURRENT_FIELDS, calendar_s
                names = LIVE_FIELDS
                wire.update(dict.fromkeys(CURRENT_FIELDS, 0.))
                wire.update(version=2., current_valid=1., query_before_controller_s=now-.006,
                            query_after_controller_s=now, query_before_system_s=now-.006,
                            query_before_steady_s=now-.006, query_after_system_s=now,
                            query_after_steady_s=now, current_max_age_s=.3, host_clock_tolerance_s=.001,
                            certificate_frame=wire["frame"], certificate_source_s=calendar_s(wire),
                            certificate_sample_system_s=now, certificate_sample_steady_s=now,
                            certificate_generation=0., certificate_incarnation_0=1., certificate_incarnation_1=2.,
                            certificate_incarnation_2=3., certificate_incarnation_3=4.)
            binding = {"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                       "calendar_to_system_offset_s": 0., "uncertainty_s": .001,
                       "system_anchor_s": 9., "steady_anchor_s": 9., "valid_until_system_s": 100.}
            if getattr(self, "hardware_causal", False):
                from tools.data_factory.rollout.gripper_evidence import CAUSAL_FIELDS
                names = CAUSAL_FIELDS
                wire = {**dict.fromkeys(names, 0.), **wire, "version": 3.}
                binding = {"schema_version": "fr5.gripper_temporal_policy.v1", "incarnation": [1, 2, 3, 4],
                           "max_age_s": .3, "host_clock_tolerance_s": .001}
                if getattr(self, "hardware_selected", False):
                    from tools.data_factory.rollout.gripper_evidence import SELECTED_FIELDS
                    names = SELECTED_FIELDS
                    wire = {**dict.fromkeys(names, 0.), **wire, "version": 4.}
            packet = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
                interface_names=list(names), values=[float(wire[k]) for k in names])])
            value["gripper_controller"]["hardware_execution"] = decode_dynamic_state(
                deserialize_message(serialize_message(packet), DynamicJointState), binding, now)
            for key, names, positions in (("arm_controller", JOINTS[:6], self.current[:6]),
                                           ("gripper_controller", JOINTS[-1:], self.current[-1:])):
                message = JointTrajectoryControllerState(joint_names=names)
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(round(now * 1e9), 10**9)
                message.reference.positions = positions
                message.feedback.positions = positions
                native = deserialize_message(serialize_message(message), JointTrajectoryControllerState)
                value[key]["sample"] = RosMoveItTransport._controller_values(native, "ROS_CONTROLLER_STATE")["sample"]
        return value

    def build_learned_trajectory(self, p):
        self.calls.append("compile-learned")
        return repr(p["actions"]).encode()

    def build_learned_segment(self, p, segment):
        self.calls.append("compile-learned-segment")
        return repr((p["actions"], segment)).encode()

    def start_phase(self, step, **kwargs):
        if self.active:
            raise ContractError("ROS_EXEC_ACTIVE")
        self.active = True
        self.sent.append(copy.deepcopy(step))
        if self.on_start:
            self.on_start()
        if kwargs["cancel_event"].is_set():
            raise ContractError("ROS_EXEC_CANCELLED")

    def poll_active(self):
        if not self.active:
            return None
        if self.failure:
            raise ContractError(self.failure)
        self.active = False
        self.current = ACTION[:]
        return object()

    def cancel_active(self, *_):
        self.cancel_count += 1
        self.active = False


class Cell:
    def __init__(self):
        self.ready = True
        self.binding = {}
    def read(self):
        return {"robot_system_id": "fr5-lab-a", "cell_ready": self.ready, **self.binding}
    def mark_blocked(self, reason, run_id, plan_digest, *, expected_state_digest=None, blocking=True):
        if expected_state_digest is not None and canonical_digest(self.read()) != expected_state_digest:
            raise ContractError("STATE_CHANGED")
        self.ready = False
        self.binding = {"reason_code": reason, "run_id": run_id, "plan_digest": plan_digest}


class Scene:
    def __init__(self):
        self.updates = []
    @contextmanager
    def locked_snapshot(self, digest, *, blocking=True):
        yield {"scene_state_digest": digest, "scene_state": {"revision": 1, "objects": {"cube-1": {"state": "ON_SURFACE", "object_profile_id": "cube"}}}}
    def update_object(self, **kwargs):
        self.updates.append(kwargs)


class Recorder:
    def __init__(self, calls):
        self.calls = calls
        self.state = "IDLE"
    def __call__(self, request):
        op = request["op"]
        self.calls.append(("recorder", op))
        self.state = {"begin": "RECORDING", "freeze": "FROZEN", "abort": "ABORTED", "commit": "COMMITTED",
                      "retain": "QUARANTINED_COMMIT"}.get(op, self.state)
        return {"schema_version": "data_factory.recorder_response.v1", "op_id": request["op_id"], "op": op,
                "ok": True, "state": self.state, "reason_code": self.state, "run_id": "run", "transaction_id": "tx",
                "episode_index": 0, "metrics": {"rows": 1, "writer_queue": 0, "writer_queue_drops": 0,
                "alignment_failures": 0, "observed_monotonic_ns": time.monotonic_ns()}, "artifacts": {}, "detail": "",
                "writer_alive": True, "writer_error": None, "sampler_alive": True,
                **({"retention": {"schema_version": "data_factory.diagnostic_retention.v1",
                                  "transaction_id": "tx", "disposition": request["disposition"],
                                  "durable": True, "save_uncertain": False,
                                  "partial": False, "quarantined": True, "staging_state": "QUARANTINED",
                                  "training_eligible": False, "quality_accepted": False}}
                   if op == "retain" else {})}


class AdmissionFreshnessEvidenceTest(unittest.TestCase):
    def test_expired_admission_survives_one_job_without_effects(self):
        program = compile_program(source(), proposal())
        original = copy.deepcopy(program)
        transport, recorder = mock.Mock(), mock.Mock()
        clock = mock.Mock(return_value=10.31)
        executor = PickupExecutor(transport, source_clock=clock)
        responses = []
        commands = []

        def request(command):
            commands.append(copy.deepcopy(command))
            result = executor.process(command)
            responses.append(copy.deepcopy(result))
            return result

        result = OneJob(recorder, request).plan_only("age-check", program, SCENE)
        self.assertEqual(result["code"], "LEARNED_STALE_OBSERVATION")
        self.assertEqual(result["planning_response"], responses[0])
        self.assertEqual(responses[0]["data"]["planning_failure"], {
            "stage": "INITIAL_PLAN_ADMISSION",
            "motion_program_digest": canonical_digest(program),
            "checked_at_s": 10.31,
            "source_timestamps_s": program["learned_proposal"]["source_timestamps_s"],
            "inference_completed_at_s": 10.,
            "max_observation_age_s": .3,
        })
        self.assertEqual(executor.runs, {})
        self.assertEqual(program, original)
        transport.assert_not_called()
        self.assertEqual(transport.mock_calls, [])
        recorder.assert_not_called()
        clock.return_value = 20.
        self.assertEqual(executor.process(commands[0]), responses[0])
        self.assertEqual(clock.call_count, 1)

        for field, value in (("checked_at_s", 10.1), ("checked_at_s", float("nan")),
                             ("stage", "CONTINUATION_PLAN_ADMISSION"),
                             ("max_observation_age_s", 1.),
                             ("inference_completed_at_s", 9.9)):
            with self.subTest(field=field, value=value):
                altered = copy.deepcopy(responses[0])
                altered["data"]["planning_failure"][field] = value
                failed = OneJob(recorder, lambda _: altered).plan_only("age-check", program, SCENE)
                self.assertEqual(failed["code"], "EXECUTOR_RESPONSE")
                self.assertNotIn("planning_response", failed)
        recorder.assert_not_called()

    def test_public_plan_only_preserves_failure_and_closes_child(self):
        from tools.data_factory import run_job
        from tests.data_factory.operator.fixtures import payload, runtime_validated

        program = compile_program(source(), proposal())
        transport = mock.Mock()
        executor = PickupExecutor(transport, source_clock=lambda: 10.31)
        child = SimpleNamespace(request=lambda request, *_: executor.process(request), close=mock.Mock())
        with tempfile.TemporaryDirectory() as directory:
            result = run_job.run_plan_only(
                {**payload(), "run_root": directory}, threading.Event(), lambda _: None,
                resolver=lambda _: (runtime_validated(), program, SCENE),
                executor_factory=lambda *_: child,
            )
            self.assertEqual(result["code"], "LEARNED_STALE_OBSERVATION")
            failure = result["data"]["planning_response"]["data"]["planning_failure"]
            self.assertEqual(failure["checked_at_s"], 10.31)
            self.assertEqual(failure["motion_program_digest"], canonical_digest(program))
            self.assertEqual(list(Path(directory).iterdir()), [])
        child.close.assert_called_once()
        self.assertEqual(executor.runs, {})
        self.assertEqual(transport.mock_calls, [])

    def test_diagnostic_failure_preserves_original_rejection(self):
        from tools.data_factory.motion import pickup_executor
        program = compile_program(source(), proposal())
        transport = mock.Mock()
        executor = PickupExecutor(transport, source_clock=lambda: 10.31)
        real_digest = canonical_digest

        def digest(value):
            if value == program:
                raise ValueError("diagnostic unavailable")
            return real_digest(value)

        command = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "age-check",
                   "op": "plan", "payload": {"run_id": "age-check", "motion_program": program,
                                               "scene_binding": SCENE}}
        with mock.patch.object(pickup_executor, "canonical_digest", side_effect=digest):
            result = executor.process(command)
        self.assertEqual(result["code"], "LEARNED_STALE_OBSERVATION")
        self.assertIsNone(result["data"])
        self.assertEqual(executor.runs, {})
        self.assertEqual(transport.mock_calls, [])


class NativeTaskPlanConsumerTest(unittest.TestCase):
    """Reuse source fixtures, not finite compilation or its execution protocol."""

    @staticmethod
    def policy():
        return {"schema_version": "data_factory.native_learned_policy.v1",
                "checkpoint": copy.deepcopy(CHECKPOINT), **OPTIONS,
                "velocity_scaling": .1, "max_observation_age_s": .3}

    def owners(self):
        transport, recorder, cell = mock.Mock(), mock.Mock(), mock.Mock()
        executor = PickupExecutor(transport, source_clock=lambda: 10., monotonic_clock=lambda: 10.)
        job = OneJob(recorder, executor.process, cell)
        return job, executor, transport, recorder, cell

    def test_configuration_and_task_grant_do_not_compile_or_consume_a_prediction(self):
        job, executor, transport, recorder, cell = self.owners()
        src, policy, scene = cross_workspace_source(), self.policy(), copy.deepcopy(SCENE)
        result = job.plan_stream("run", src, scene, policy)
        self.assertTrue(result["ok"], result)
        plan = result["plan_envelope"]["plan"]
        self.assertEqual(result["plan_digest"], canonical_digest(plan))
        self.assertIsNone(result["dry_run_digest"])
        self.assertIsNone(result["execution_evidence"])
        self.assertEqual(plan["source_program"]["endpoint_bindings"], src["endpoint_bindings"])
        for key in ("steps", "learned_proposal", "learned_history", "precommit_safety"):
            self.assertNotIn(key, plan)
        grant = task_grant(src, scene, policy)
        self.assertTrue(job.admit_task(grant)["ok"])
        self.assertEqual(job.state, "APPROVED")
        self.assertNotIn("execution", executor.runs["run"])
        self.assertEqual((transport.mock_calls, recorder.mock_calls, cell.mock_calls), ([], [], []))
        src["endpoint_bindings"][0]["workspace_id"] = "MUTATED"
        policy["instruction"] = "MUTATED"
        scene["revision"] += 1
        self.assertEqual(executor.runs["run"]["plan"], plan)
        self.assertFalse(job.plan_stream("another", src, scene, policy)["ok"])

    def test_mismatched_grant_and_response_cannot_authorize_normal_task(self):
        from tools.data_factory.rollout.task_authority import check_grant

        job, executor, transport, recorder, cell = self.owners()
        src, policy = cross_workspace_source(), self.policy()
        self.assertTrue(job.plan_stream("run", src, SCENE, policy)["ok"])
        grant = task_grant(src, SCENE, {**policy, "instruction": "different task"})
        rejected = job.admit_task(grant)
        self.assertEqual(rejected["code"], "TASK_GRANT_SCOPE")
        self.assertEqual((job.state, executor.runs["run"]["state"]), ("PLANNED", "PLANNED"))
        expired = task_grant(src, SCENE, policy, deadline_s=9.)
        with self.assertRaisesRegex(ContractError, "TASK_DEADLINE_EXHAUSTED"):
            check_grant(expired, executor.runs["run"]["plan"], 10.)
        self.assertEqual((transport.mock_calls, recorder.mock_calls, cell.mock_calls), ([], [], []))

        def tampered(request):
            response = executor.process(request)
            response["data"]["plan"]["policy"]["instruction"] = "different task"
            response["plan_digest"] = canonical_digest(response["data"]["plan"])
            return response

        executor = PickupExecutor(transport, source_clock=lambda: 10., monotonic_clock=lambda: 10.)
        job = OneJob(recorder, tampered, cell)
        self.assertEqual(job.plan_stream("run", src, SCENE, policy)["code"], "EXECUTOR_RESPONSE")
        self.assertIsNone(job.approval_scope)


class FinitePlanTest(unittest.TestCase):
    def setUp(self):
        # Legacy lifecycle tests isolate contact just as they isolate planning.
        # Prospective fixtures below always call the real producer and services.
        from tools.data_factory.motion import contact_transition
        real_before, real_completed = contact_transition.before, contact_transition.completed
        def before(transport, plan, step, observation, context):
            if not context.get("isolated_lifecycle_fixture"):
                return real_before(transport, plan, step, observation, context)
            return {"segment_digest": canonical_digest(step), "start_observation": copy.deepcopy(observation)}
        def completed(transport, plan, step, observation, context, pending):
            if not context.get("isolated_lifecycle_fixture"):
                return real_completed(transport, plan, step, observation, context, pending)
            context["checked_segments"].append({**pending, "terminal_observation": copy.deepcopy(observation)})
        for name, value in (("before", before), ("completed", completed)):
            patcher = mock.patch.object(contact_transition, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(Transport, "prepare_contact_transition", staticmethod(self.lifecycle_contact), create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def lifecycle_contact(plan, scene_object, *, first, deadline_s):
        return {"status": "PROSPECTIVE", "isolated_lifecycle_fixture": True,
                "plan_digest": canonical_digest(plan), "scene_binding": copy.deepcopy(plan["scene_binding"]),
                "deadline_s": deadline_s, "checked_segments": [], "close": None}

    def test_robot_model_trial_scope_is_plan_only(self):
        from tools.data_factory import run_job

        plan_only = {
            "mode": "plan_only",
            "learned_checkpoint": "/tmp/checkpoint",
            "gripper_temporal_policy": "/tmp/policy.json",
            "learned_robot_model_trial_urdf": "/tmp/candidate.urdf",
        }
        options = run_job._learned_options(plan_only)
        self.assertEqual(
            options["robot_model_trial_urdf"],
            "/tmp/candidate.urdf",
        )

        with self.assertRaisesRegex(
            ContractError,
            "LEARNED_ROBOT_MODEL_TRIAL_SCOPE",
        ):
            run_job._learned_options({
                **plan_only,
                "mode": "live",
            })

    def test_robot_model_trial_execution_is_blocked_before_transport(self):
        transport = Transport()
        executor = PickupExecutor(
            transport=transport,
            execution_enabled=True,
            clock=lambda: datetime(
                2026, 9, 11, tzinfo=timezone.utc
            ),
        )
        digest = canonical_digest(
            "robot-model-trial-plan"
        )
        executor.runs["run"] = {
            "state": "APPROVED",
            "digest": digest,
            "precommit_safety": {
                "status": "PENDING",
            },
            "approval": {
                "approval_expiry":
                    "2099-01-01T00:00:00Z",
                "approval_scope": "HUMAN_GATED",
            },
            "plan": {
                "robot_model_trial": {
                    "schema_version":
                        "data_factory.robot_model_trial.v1",
                },
            },
        }

        response = executor._execute({
            "run_id": "run",
            "plan_digest": digest,
            "lease_id": "trial-lease",
        })

        self.assertFalse(response["ok"])
        self.assertEqual(
            response["code"],
            "ROBOT_MODEL_TRIAL_EXECUTION_BLOCKED",
        )
        self.assertEqual(response["state"], "APPROVED")
        self.assertEqual(transport.sent, [])

    def test_recorded4032_raw_output_retains_all_rows_and_rejects_limits(self):
        recorded = json.loads(Path(__file__).with_name("recorded4032.json").read_text())
        self.assertEqual(recorded["source_report_sha256"],
                         "0416a02d20ef41ef27137f5b7812b45ca83aa71fcfa3f53024e826a7201379c7")
        for sample in recorded["samples"]:
            with self.subTest(sample=sample["index"]):
                raw = copy.deepcopy(sample["actions"])
                self.assertEqual(len(raw), 50)
                self.assertTrue(all(len(row) == 7 for row in raw))
                obs = observation()
                obs["observation.state"] = sample["initial_state"]
                inference = FinitePolicyInference(lambda _: raw, recorded["checkpoint"],
                                                  source_clock=lambda: 10.)
                # Only acquisition time is synthetic; every recorded action is
                # unchanged. Artifact/runtime qualification is not rerun here.
                code = "LEARNED_VELOCITY_LIMIT" if sample["index"] == 3 else "LEARNED_JOINT_LIMIT"
                with self.assertRaisesRegex(ContractError, code):
                    inference.propose(obs, instruction="recorded CPU replay",
                                      robot_description=recorded["robot_description"], period_s=1 / 30)
                self.assertEqual(raw, sample["actions"])

    def test_public_live_native_path_keeps_one_child_and_honest_probe_evidence(self):
        self._public_native_consumer("live")

    def test_public_native_plan_only_never_starts_recorder_or_goal(self):
        self._public_native_consumer("plan_only")

    def test_public_causal_live_path_consumes_explicit_policy_without_clock_mapping(self):
        self._public_native_consumer("live", causal=True)

    def test_public_causal_plan_only_preserves_zero_execution_effects(self):
        self._public_native_consumer("plan_only", causal=True)

    def test_public_native_task_grant_source_change_sends_no_next_goal(self):
        self._public_native_consumer("live", causal=True, granted=True, source_changed=True)

    def test_public_native_task_grant_runs_two_outputs_without_clicks_and_hands_off(self):
        self._public_native_consumer("live", causal=True, granted=True)

    def test_public_scoped_native_consumer_binds_before_capture_and_continues(self):
        self._public_native_consumer("live", causal=True, granted=True, scoped=True)

    def test_public_scoped_plan_only_binds_before_capture_without_motion(self):
        self._public_native_consumer("plan_only", causal=True, scoped=True)

    def test_public_scoped_capture_and_inference_failures_retire_context(self):
        for failure in ("capture", "inference"):
            with self.subTest(failure=failure):
                self._public_native_consumer("plan_only", causal=True, scoped=True, generation_failure=failure)

    def test_public_scoped_cleanup_failure_keeps_original_generation_error(self):
        for mode in ("plan_only", "live"):
            for failure in ("capture", "inference"):
                for cleanup in ("exception", "rejected"):
                    with self.subTest(mode=mode, failure=failure, cleanup=cleanup):
                        self._public_native_consumer(mode, causal=True, scoped=True, granted=mode == "live",
                            generation_failure=failure, cleanup_failure=cleanup)

    def test_public_scoped_continuation_preserves_failure_and_separate_cleanup_evidence(self):
        for cleanup in ("exception", "rejected"):
            with self.subTest(cleanup=cleanup):
                self._public_native_consumer("live", causal=True, scoped=True, granted=True,
                    generation_failure="inference", cleanup_failure=cleanup, failure_on_next=True)

    def test_public_native_continuation_reuses_owner_and_recorder_with_exact_approval(self):
        self._public_native_consumer("live", causal=True, continuation=True)

    def test_public_native_continuation_cancel_does_not_send_next_output(self):
        self._public_native_consumer("live", causal=True, continuation=True, next_choice="CANCEL")

    def test_public_native_next_precontact_cancel_retains_prior_chunk_evidence(self):
        self._public_native_consumer("live", causal=True, continuation=True, precontact_cancel=True)

    def test_public_causal_plan_only_freezes_explicit_serialized_retiming(self):
        self._public_native_consumer("plan_only", causal=True, reference_mode="serialized_retime")

    def test_public_causal_plan_only_freezes_explicit_percent_representation(self):
        self._public_native_consumer("plan_only", causal=True, reference_mode="serialized_percent_retime")

    def _public_native_consumer(self, mode, *, causal=False, reference_mode=None, continuation=False, next_choice="APPROVE", precontact_cancel=False, granted=False, source_changed=False, scoped=False, generation_failure=None, cleanup_failure=None, failure_on_next=False):
        from tools.data_factory import run_job
        from tools.data_factory.learned_action_adapter import NativeSmolVLA
        from tests.data_factory.operator.fixtures import PROFILE, JOB, runtime_validated, payload
        profile = copy.deepcopy(PROFILE)
        profile.update(camera_profile="up-wrist", camera_roles=["up", "wrist"],
                       camera_serials={"up": "up", "wrist": "wrist"},
                       camera_topics={"up": "/up", "wrist": "/wrist"})
        validated = runtime_validated(job={**JOB, "instruction": "synthetic probe", "operator_or_agent_id": "operator"}, profile=profile)
        program = source()
        if reference_mode is not None:
            next(s for s in program["steps"] if s["phase"] == "GRIPPER_OPEN")["gripper_position_m"] = .02
        program["resolved_job_digest"] = validated["resolved_job_digest"]
        program["binding_digests"]["collection_profile"] = validated["input_digests"]["collection_profile"]
        calls, closed, observed_requests = [], [], []
        transport, cell, scene = Transport(), Cell(), Scene()
        transport.hardware = True
        transport.hardware_current = True
        transport.hardware_causal = causal
        transport.hardware_selected = causal
        def capture(topics, age):
            if generation_failure == "capture" and (not failure_on_next or calls.count(("executor", "capture_observation")) == 2):
                raise ContractError("LEARNED_STALE_OBSERVATION")
            self.assertEqual(topics, {"camera1": "/up", "camera2": "/wrist"})
            self.assertEqual(age, .3)
            value = observation()
            value["observation.state"] = transport.current[:]
            for name in ("camera1", "camera2"):
                image = value[f"observation.images.{name}"]
                image["data_hex"] = image.pop("data").hex()
            return value
        transport.capture_policy_observation = capture
        executor = PickupExecutor(transport, execution_enabled=True, cell_state_store=cell,
                                  scene_state_store=scene, source_clock=lambda: 10., monotonic_clock=lambda: 10.)
        def request(value, _cancel):
            calls.append(("executor", value["op"]))
            observed_requests.append(copy.deepcopy(value))
            if value["op"] == "cancel" and "generation_id" in value["payload"] and cleanup_failure is not None:
                if cleanup_failure == "exception":
                    raise ContractError("JSONL_BROKEN_PIPE")
                value = copy.deepcopy(value)
                value["payload"]["generation_id"] = "wrong-generation"
            return executor.process(value)
        child = SimpleNamespace(request=request, close=lambda **_: closed.append(True))
        recorder = Recorder(calls)
        recorder.close = lambda **_: None
        real_inference = FinitePolicyInference
        class Native:
            def warmup(self, *, instruction, height, width, cancel_event):
                calls.append(("native", "warmup"))
                self_test.assertFalse(cancel_event.is_set())
                return {"input_kind": "SYNTHETIC_ZERO_RGB_STATE", "image_shape": [height, width, 3],
                        "instruction_digest": canonical_digest(instruction), "device": "cpu", "model_calls": 1,
                        "output_disposition": "DISCARDED", "rng_state_restored": True, "duration_s": 2., "inference_duration_s": 1.6}
            @contextmanager
            def prepare_inference(self):
                calls.append(("native", "prepare"))
                if source_changed and calls.count(("native", "prepare")) == 2:
                    raise ContractError("LEARNED_CHECKPOINT_CHANGED")
                yield self
            checkpoint = CHECKPOINT
            def __call__(self, value):
                if generation_failure == "inference" and (not failure_on_next or calls.count(("native", "prepare")) == 2):
                    raise RuntimeError("synthetic inference failure")
                self_test.assertEqual(value["task"], "synthetic probe")
                return [ACTION[:]]
        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "robot.urdf").write_text(XML)
            mapping = {"observation.images.up": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
            (root / "train_config.json").write_text(json.dumps({"rename_map": mapping}))
            clock_binding = transport.snapshot()["gripper_controller"]["hardware_execution"]["clock_binding"]
            hardware_key = "gripper_temporal_policy" if causal else "gripper_source_clock"
            binding_path = root / ("temporal.json" if causal else "clock.json")
            binding_path.write_text(json.dumps(clock_binding))
            native = Native()
            native.policy_dir = root
            value = {**payload(mode), "run_id": "run",
                     "job": validated["normalized_job"], "urdf": str(root / "robot.urdf"),
                     "learned_checkpoint": str(root), hardware_key: str(binding_path)}
            if reference_mode is not None:
                value["learned_reference_mode"] = reference_mode
            if mode == "live":
                value.update(run_root=str(root / "runs"), dataset_root=str(root / "unused-dataset"), camera_profile="up-wrist")
            if granted or scoped:
                bound_proposal = proposal()
                if scoped:
                    bound_proposal["schema_version"] = "data_factory.finite_learned_proposal.v2"
                bound_proposal["period_s"] = 1 / profile["fps"]
                bound_proposal["runtime_inputs"] = {**run_job._learned_options(value), "clock_binding": clock_binding,
                    "camera_topics": {"camera1": "/up", "camera2": "/wrist"}, "camera_mapping": mapping,
                    "fps": profile["fps"], "hardware_wire_version": 4 if causal else 2}
                value["task_grant"] = task_grant(program, SCENE, bound_proposal)
            value = run_job._run_payload(value)
            factory = mock.Mock(return_value=child)
            def decide(_prompt, choices):
                calls.append(("operator", "decision"))
                return choices if isinstance(choices, str) else "PASS"
            caller_name = "run_live" if mode == "live" else "run_plan_only"
            caller = getattr(run_job, caller_name)
            def checkpoint(request):
                from tools.data_factory.operator.workflow.intents import OperatorCheckpointPort
                port = OperatorCheckpointPort(operator_label="operator")
                pending = port.offer(request)
                kind = request["kind"]
                if kind == "PRECONTACT_HUMAN":
                    choice = "CANCEL" if precontact_cancel and len(transport.sent) == 1 else "CONFIRM"
                elif kind == "LEARNED_NEXT_PLAN":
                    self.assertEqual(len(transport.sent), 1)
                    self.assertNotEqual(request["plan_digest"], request["evidence"]["previous_plan_digest"])
                    self.assertEqual(request["plan_digest"], canonical_digest(request["evidence"]["plan_envelope"]["plan"]))
                    choice = next_choice
                else:
                    self.assertEqual(kind, "LEARNED_CHUNK_COMPLETE")
                    self.assertEqual(request["evidence"]["execution_state"], "LEARNED_CHUNK_COMPLETE")
                    self.assertEqual(request["evidence"]["recorder_state"], "RECORDING")
                    self.assertIn("does not qualify task success or dataset commit", request["prompt"])
                    choice = "CONTINUE" if continuation and len(transport.sent) == 1 else "PASS"
                self.assertNotIn(("recorder", "freeze"), calls)
                port.resolve({"checkpoint_binding_digest": pending["binding_digest"], "choice": choice})
                return port.wait(.1)
            def live_ports(*args, **kwargs):
                if mode == "plan_only":
                    return caller(*args, **kwargs, resolver=lambda _: (validated, program, SCENE), executor_factory=factory)
                return caller(*args, **kwargs, resolver=lambda _: (validated, program, SCENE), executor_factory=factory,
                    recorder_factory=lambda *_: recorder, tty_decision=decide, checkpoint_provider=checkpoint,
                    camera_warmup_call=lambda *_: {"schema_version": "data_factory.camera_warmup.v1", "attempts": []})
            with mock.patch.object(NativeSmolVLA, "load", return_value=native), \
                 mock.patch("tools.data_factory.rollout.finite_plan.FinitePolicyInference", side_effect=lambda *a, **kw: real_inference(*a, **kw, source_clock=lambda: 10., monotonic_clock=lambda: 10.)), \
                 mock.patch.object(run_job, "CellStateStore", return_value=cell), \
                 mock.patch.object(run_job, "SceneStateStore", return_value=scene), \
                 mock.patch.object(run_job, "ResourceMonitor"), \
                 mock.patch.object(run_job, caller_name, side_effect=live_ports):
                session = run_job.RunSession()
                accepted = session.process(json.loads(json.dumps({"schema_version": run_job.COMMAND_SCHEMA,
                    "op_id": "native-live", "op": "run", "payload": value})))
                self.assertTrue(accepted["ok"], accepted)
                session.worker.join(5.)
                if session.worker.is_alive():
                    session.input_closed()
                    session.worker.join(2.)
                self.assertFalse(session.worker.is_alive())
                result = session.snapshot
            if generation_failure is not None:
                self.assertEqual(result["code"], "LEARNED_STALE_OBSERVATION" if generation_failure == "capture" else "LEARNED_POLICY_FAILED")
                self.assertEqual(calls.count(("executor", "cancel")), 2 if failure_on_next else 1)
                self.assertEqual(executor._generation["cancelled"], cleanup_failure is None)
                if cleanup_failure is not None:
                    self.assertEqual(result["data"]["generation_cleanup"], {"status": "UNCONFIRMED",
                        "code": "JSONL_BROKEN_PIPE" if cleanup_failure == "exception" else "LEARNED_GENERATION_CONTEXT"})
                self.assertEqual(len(transport.sent), 1 if failure_on_next else 0)
                self.assertEqual(calls.count(("recorder", "begin")), 1 if failure_on_next else 0)
                if failure_on_next:
                    self.assertTrue(executor.runs["run"]["cancel_event"].is_set())
                    self.assertFalse(transport.owns_active_goal)
                    self.assertEqual(result["data"]["task_effectiveness"], "UNKNOWN")
                    from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
                    retained = json.loads((Path(value["run_root"]) / "run" /
                                           "learned_lifecycle_result.json").read_text())
                    self.assertEqual(retained.get("generation_cleanup"), result["data"].get("generation_cleanup"))
                    self.assertEqual(build_run_diagnostic(retained), result["data"])
                    self.assertEqual(result["data"]["diagnostic_digest"], canonical_digest({
                        k: v for k, v in result["data"].items() if k != "diagnostic_digest"}))
                else:
                    self.assertFalse(any(target == "recorder" for target, _ in calls))
                self.assertEqual(closed, [True])
                return
            if granted and source_changed:
                self.assertEqual(result["code"], "LEARNED_CHECKPOINT_CHANGED", result)
                self.assertEqual(len(transport.sent), 1)
                self.assertEqual(calls.count(("recorder", "begin")), 1)
                self.assertFalse(any(target == "operator" for target, _ in calls))
                self.assertNotIn(("recorder", "commit"), calls)
                return
            if granted:
                self.assertEqual(result["code"], "MECHANICAL_TERMINAL_UNAVAILABLE", result)
                self.assertEqual(result["data"]["task_handoff"]["termination_reason"], "TASK_OUTPUT_LIMIT_REACHED")
                self.assertEqual(result["data"]["task_handoff"]["status"], "BLOCKED_UNAVAILABLE")
                self.assertEqual(len(transport.sent), 2)
                self.assertEqual(calls.count(("native", "prepare")), 2)
                self.assertEqual(calls.count(("recorder", "begin")), 1)
                self.assertEqual(calls.count(("executor", "admit_task")), 2)
                self.assertFalse(any(target == "operator" or op in {"approve", "approve_next", "confirm", "commit"} for target, op in calls))
                current = executor.runs["run"]
                if scoped:
                    self.assertEqual(calls.count(("executor", "begin_generation")), 2)
                    self.assertLess(calls.index(("executor", "begin_generation")), calls.index(("executor", "capture_observation")))
                    context = current["plan"]["learned_proposal"]["generation_context"]
                    self.assertEqual(context["predecessor_plan_digest"], current["learned_history"][0]["approval"]["plan_digest"])
                    self.assertEqual(context["task_deadline_monotonic_s"], current["task_deadline"])
                self.assertEqual(len(current["learned_history"]), 1)
                previous = current["learned_history"][0]
                self.assertNotEqual(previous["approval"]["plan_digest"], current["digest"])
                self.assertEqual(previous["approval"]["task_grant"], current["approval"]["task_grant"])
                self.assertEqual(current["task_grant"], value["task_grant"])
                self.assertEqual(closed, [True])
                return
            expected = "CANCELLED_BY_OPERATOR" if continuation and (next_choice == "CANCEL" or precontact_cancel) else "PRECOMMIT_SAFETY"
            self.assertEqual(result["code"], expected if mode == "live" else "PLANNED",
                             {k: v for k, v in result.items() if k != "data"})
            factory.assert_called_once()
            self.assertLess(calls.index(("native", "warmup")), calls.index(("executor", "capture_observation")))
            self.assertEqual(closed, [True])
            expected_prefix = ["begin_generation", "capture_observation", "plan"] if scoped else ["capture_observation", "plan"]
            self.assertEqual([op for target, op in calls if target == "executor"][:len(expected_prefix)], expected_prefix)
            if mode == "plan_only":
                self.assertEqual(transport.sent, [])
                self.assertFalse(any(target in {"operator", "recorder"} for target, _ in calls))
                self.assertFalse((root / "runs").exists())
                self.assertIsNone(result["data"]["trajectory_variant_binding"])
                prepared = result["data"]["finite_learned_plan"]["plan"]["learned_proposal"]
                self.assertEqual(prepared.get("raw_actions", prepared["actions"]), [ACTION])
                if reference_mode is not None:
                    frozen = result["data"]["finite_learned_plan"]["plan"]["learned_proposal"]
                    self.assertEqual(frozen["runtime_inputs"]["reference_mode"], reference_mode)
                    self.assertEqual(frozen["reference_timing"]["raw_actions_digest"], canonical_digest([ACTION]))
                    self.assertEqual(frozen["reference_timing"]["endpoint_changed"], reference_mode == "serialized_percent_retime")
                return
            self.assertLess(calls.index(("executor", "plan")), calls.index(("operator", "decision")))
            self.assertLess(calls.index(("operator", "decision")), calls.index(("recorder", "begin")))
            self.assertLess(calls.index(("recorder", "begin")), calls.index(("executor", "execute")))
            self.assertEqual(len(transport.sent), 2 if continuation and next_choice == "APPROVE" and not precontact_cancel else 1)
            self.assertEqual(calls.count(("recorder", "begin")), 1)
            self.assertEqual(calls.count(("native", "warmup")), 1)
            if continuation:
                self.assertEqual(calls.count(("native", "prepare")), 2)
                observations = [v["payload"] for v in observed_requests if v["op"] == "capture_observation"]
                self.assertEqual(observations[1]["run_id"], "run")
                self.assertIn("lease_id", observations[1])
                self.assertIn("plan_digest", observations[1])
                saved = json.loads((root / "runs" / "run" / "learned_lifecycle_result.json").read_text())
                history = saved["execution_evidence"].get("learned_history", [])
                self.assertEqual(len(history), 1 if next_choice == "APPROVE" else 0)
                if history:
                    self.assertNotEqual(saved["plan_digest"], history[0]["approval"]["plan_digest"])
                    self.assertEqual(saved["plan_digest"], result["plan_digest"])
            self.assertEqual(transport.sent[0]["learned_proposal"]["actions"], [ACTION])
            self.assertNotIn(("recorder", "commit"), calls)
            evidence = json.loads((root / "runs" / "run" / "preapproval_evidence.json").read_text())
            self.assertIsNone(evidence["trajectory_variant_binding"])
            self.assertIsNone(evidence["trajectory_variant_binding_digest"])
            plan = evidence["plan_envelope"]["plan"]
            self.assertEqual(plan["learned_source_program"], program)
            self.assertEqual(plan["learned_proposal"]["runtime_inputs"]["device"], "cpu")
            self.assertEqual(plan["learned_proposal"]["runtime_inputs"]["warmup"]["output_disposition"], "DISCARDED")
            self.assertEqual(plan["learned_proposal"]["runtime_inputs"]["clock_binding"], clock_binding)
            self.assertEqual(plan["learned_proposal"]["runtime_inputs"]["hardware_wire_version"], 4 if causal else 2)
            self.assertEqual(plan["learned_proposal"]["runtime_inputs"][hardware_key], str(binding_path))
            self.assertEqual(result["data"]["task_effectiveness"], "UNKNOWN")

    def make_held_job(self, initial_feedback=.021, *, controller_samples=True, arm_target=.001, recorded=None, quantize_gripper=False, max_observation_age_s=5., mechanical=False, contact_fixture=None, scoped=False):
        # Reuse this file's lifecycle fixtures with the actual ROS serializers,
        # action dispatch, polling and cancellation; no ROS node is constructed.
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTolerance
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import RobotTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rclpy.serialization import serialize_message, deserialize_message
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport

        from control_msgs.msg import DynamicJointState, InterfaceValue
        from tools.data_factory.rollout.gripper_evidence import FIELDS, RESOURCE, CALENDAR
        import datetime
        now = [10.]
        state = {"joints": [0.] * 6, "feedback": initial_feedback, "reference": .021, "age": 0., "complete": False,
                 "generation": 0, "completed_generation": 0, "started": 0., "finished": 0., "hardware_override": {}}
        if recorded is not None:
            sample, description, checkpoint = recorded
            state.update(joints=sample["initial_state"][:6], feedback=sample["initial_state"][-1], reference=.01176)
        mapping = {"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                   "calendar_to_system_offset_s": 0., "uncertainty_s": .001,
                   "system_anchor_s": 9., "steady_anchor_s": 9., "valid_until_system_s": 100.}
        def calendar(stamp):
            t = datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc)
            return [t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond // 1000]
        def hardware():
            if state["complete"] and state["generation"] > state["completed_generation"]:
                state["completed_generation"] = state["generation"]
                state["finished"] = now[0] - .002
            wire = dict.fromkeys(FIELDS, 0.)
            wire.update(version=1., incarnation_0=1., incarnation_1=2., incarnation_2=3., incarnation_3=4.,
                        generation=float(state["generation"]), completed_generation=float(state["completed_generation"]),
                        raw_reference_m=state["reference"], sample_system_s=now[0], sample_steady_s=now[0],
                        command_started_system_s=state["started"], feedback_m=state["feedback"],
                        completion_reason=2. if state["completed_generation"] else 0., arm_resumed=1., valid=1.)
            wire.update(zip(CALENDAR, calendar(now[0] - .002)))
            if state["finished"]:
                wire.update(zip(["completion_" + k for k in CALENDAR], calendar(state["finished"])))
            names = FIELDS
            if contact_fixture is not None:
                from tools.data_factory.rollout.gripper_evidence import SELECTED_FIELDS, calendar_s
                names = SELECTED_FIELDS
                wire = {**dict.fromkeys(names, 0.), **wire, "version": 4.}
                wire.update(current_valid=1., query_before_controller_s=now[0]-.006,
                    query_after_controller_s=now[0], query_before_system_s=now[0]-.006,
                    query_before_steady_s=now[0]-.006, query_after_system_s=now[0], query_after_steady_s=now[0],
                    current_max_age_s=.1, host_clock_tolerance_s=.001, certificate_source_s=calendar_s(wire),
                    certificate_sample_system_s=now[0], certificate_sample_steady_s=now[0],
                    certificate_generation=state["generation"], sample_position_percent=round(state["feedback"]/.021*100))
                for i in range(4):
                    wire[f"certificate_incarnation_{i}"] = i+1
                    wire[f"terminal_incarnation_{i}"] = i+1
                if state["finished"]:
                    end = state["finished"]
                    wire.update(proof_valid=1., proof_system_s=end+.002, proof_steady_s=end+.002,
                        ack_system_s=state["started"], ack_steady_s=state["started"],
                        terminal_sample_system_s=end, terminal_sample_steady_s=end,
                        terminal_position_percent=round(state["feedback"]/.021*100), terminal_motion_done=1.,
                        terminal_query_before_controller_s=end-.004, terminal_query_after_controller_s=end+.002,
                        terminal_query_before_system_s=end-.004, terminal_query_before_steady_s=end-.004,
                        terminal_query_after_system_s=end+.002, terminal_query_after_steady_s=end+.002,
                        terminal_max_age_s=.1, terminal_host_clock_tolerance_s=.001,
                        terminal_reference_m=state["reference"], terminal_generation=state["generation"],
                        selected_valid=1., selected_generation=state["generation"], selected_upper_m=.021,
                        selected_initial_percent=state.get("initial_percent", 100), selected_index=1.,
                        selected_position=math.floor(state["reference"]/.021*100+.5), selected_velocity=20.,
                        selected_force=20., selected_max_time=30000., selected_arg5=1.)
                    if wire["selected_position"] > wire["selected_initial_percent"]:
                        wire.update(selected_velocity=10., selected_force=50.)
                t._gripper_source_clock = {"schema_version":"fr5.gripper_temporal_policy.v1", "incarnation":[1,2,3,4],
                    "max_age_s":.1, "host_clock_tolerance_s":.001}
            wire.update(state["hardware_override"])
            message = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
                interface_names=list(names), values=[float(wire[k]) for k in names])])
            # Real wire round trip and the actual native transport callback/decoder.
            t._on_gripper_hardware_state(deserialize_message(serialize_message(message), DynamicJointState))
            return t._gripper_hardware_evidence()
        class SyntheticTransport(RosMoveItTransport):
            """Synthetic clients only; retain production rejection of synthetic ROS runs."""
        t = object.__new__(SyntheticTransport)
        if contact_fixture is None:
            t.prepare_contact_transition = self.lifecycle_contact
            t.mechanical_contact_context = lambda plan, scene, snapshot, **kw: RosMoveItTransport.mechanical_contact_context(t, plan, scene, snapshot)
        t._RobotTrajectory, t._JointTrajectoryPoint, t._Duration = RobotTrajectory, JointTrajectoryPoint, Duration
        t._ExecuteTrajectory, t._FollowJointTrajectory, t._JointTolerance = ExecuteTrajectory, FollowJointTrajectory, JointTolerance
        t._serialize_message, t._deserialize_message = serialize_message, deserialize_message
        t._goal_succeeded, t._goal_canceled, t._goal_aborted = 4, 5, 6
        t._moveit_success, t._gripper_success = 1, 0
        t._active, t._execution_locked = None, False
        t._execute_goal_count = t._gripper_goal_count = 0
        t._clock, t.graph_timeout_s, t.node = lambda: now[0], .1, object()
        t._gripper_source_clock = mapping
        t._rclpy = SimpleNamespace(spin_until_future_complete=lambda *a, **kw: None, spin_once=lambda *a, **kw: None)
        t.preflight, t.precommit_safety = T().preflight, T().precommit_safety
        def observe(*_):
            value = snapshot(state["joints"][:], gripper_position=state["feedback"], force=20 if contact_fixture else 50,
                             open_velocity=10 if contact_fixture else None, open_force=50 if contact_fixture else None)
            value["gripper_controller"]["reference_position_m"] = state["reference"]
            value["joint_state_age_s"] = state["age"]
            value["joint_state_stamp_ns"] = round(now[0] * 1e9)
            value["gripper_controller"]["hardware_execution"] = hardware()
            if controller_samples:
                from control_msgs.msg import JointTrajectoryControllerState
                for key, names, reference, feedback, output in (
                        ("arm_controller", JOINTS[:6], state["joints"], state["joints"], []),
                        ("gripper_controller", JOINTS[-1:], [state["reference"]], [state["feedback"]], [.021])):
                    message = JointTrajectoryControllerState(joint_names=names)
                    message.header.stamp.sec, message.header.stamp.nanosec = divmod(round(now[0] * 1e9), 10**9)
                    message.reference.positions = reference
                    message.feedback.positions = feedback
                    message.output.positions = output  # Explicitly old/unavailable reported output.
                    message.reference.time_from_start = Duration(sec=-1, nanosec=800000000)
                    message.feedback.time_from_start = Duration(sec=-1, nanosec=900000000)
                    native = deserialize_message(serialize_message(message), JointTrajectoryControllerState)
                    value[key]["sample"] = t._controller_values(native, "ROS_CONTROLLER_STATE")["sample"]
                    value[key]["sample"].update(state.get("sample_override", {}).get(key, {}))
            return value
        t.snapshot = observe
        sent, handles = [], []
        def send(goal):
            sent.append(goal)
            if isinstance(goal, FollowJointTrajectory.Goal):
                state["initial_percent"] = round(state["feedback"]/.021*100)
                state["generation"] += 1
                state["started"] = now[0]
            state["complete"] = False
            result = (ExecuteTrajectory.Result() if isinstance(goal, ExecuteTrajectory.Goal) else FollowJointTrajectory.Result())
            if isinstance(goal, ExecuteTrajectory.Goal):
                result.error_code.val = 1
            packet = SimpleNamespace(status=4, result=result)
            future = mock.Mock()
            future.done.side_effect = lambda: state["complete"]
            future.result.return_value = packet
            handle = mock.Mock(accepted=True)
            handle.get_result_async.return_value = future
            def cancel():
                state["complete"], packet.status = True, 5
                return mock.Mock(done=lambda: True, result=lambda: SimpleNamespace(goals_canceling=[object()]))
            handle.cancel_goal_async.side_effect = cancel
            handles.append(handle)
            return mock.Mock(done=lambda: True, result=lambda: handle)
        t.execute_trajectory, t.gripper = mock.Mock(), mock.Mock()
        t.execute_trajectory.send_goal_async.side_effect = send
        t.gripper.send_goal_async.side_effect = send
        src = source()
        xml = XML.replace('upper="0.02"', 'upper="0.021"')
        if recorded is not None:
            xml = description
            for step in src["steps"]:
                step["limits"]["execution_timeout_s"] = 10.
        src["binding_digests"]["robot_description_digest"] = 'sha256:' + hashlib.sha256(xml.encode()).hexdigest()
        src["gripper_requirements"].update(command_position_m=.01176, acceptable_feedback_m={"min": .01176, "max": .01218})
        for step in src["steps"]:
            if step["phase"] == "GRIPPER_OPEN":
                step["gripper_position_m"] = .021
                if mechanical:
                    step.update(release_position_m=.0126, release_hold_s=.5)
            if step["phase"] == "GRIPPER_CLOSE":
                step["gripper_position_m"] = .01176
                step["limits"]["completion_tolerance_m"] = .01218 - .01176
        obs = observation()
        obs["observation.state"] = [0.] * 6 + [initial_feedback]
        actions = [[0.] * 6 + [.021] for _ in range(4)] + [[arm_target] * 6 + [.01176] for _ in range(8)]
        if recorded is not None:
            actions = copy.deepcopy(sample["actions"])
            obs["observation.state"] = sample["initial_state"][:]
        calls, cell, scene = [], Cell(), Scene()
        if contact_fixture is not None:
            xml, scene = contact_fixture(src, t)
            if recorded is None:
                actions = getattr(t, "_fixture_actions", [[arm_target] * 6 + [.01176]])
            cell_read = cell.read
            cell.read = lambda: {**cell_read(), "robot_system_id": src["robot_system_id"]}
        executor = PickupExecutor(t, execution_enabled=True, cell_state_store=cell, scene_state_store=scene,
                                  source_clock=lambda: now[0], monotonic_clock=lambda: now[0])
        job = OneJob(Recorder(calls), executor.process)
        inference = FinitePolicyInference(lambda _: actions, checkpoint if recorded is not None else CHECKPOINT, source_clock=lambda: now[0])
        runtime_inputs = None
        if contact_fixture is not None:
            runtime_inputs = {"checkpoint": "/synthetic/checkpoint", "device": "cpu",
                "gripper_temporal_policy": "/synthetic/temporal.json", "hardware_wire_version": 4,
                "clock_binding": t.snapshot()["gripper_controller"]["hardware_execution"]["clock_binding"],
                "camera_topics": {"camera1": "/up", "camera2": "/wrist"},
                "camera_mapping": {"observation.images.up": "observation.images.camera1",
                                   "observation.images.wrist": "observation.images.camera2"}, "fps": 30.}
        options = {**OPTIONS, "robot_description": xml, "period_s": 1 / 30,
                   "held_gripper_targets": recorded is None and not mechanical, "serialized_references": recorded is not None or mechanical,
                   "quantize_gripper": quantize_gripper, "max_observation_age_s": .3 if scoped else max_observation_age_s,
                   "runtime_inputs": runtime_inputs}
        from tools.data_factory.rollout.finite_plan import generation_spec
        grant = task_grant(src, SCENE, generation_spec(inference.checkpoint, **options)) if scoped else None
        planned = job.plan_learned("run", src, SCENE, inference, obs, task_grant=grant, **options)
        self.assertTrue(planned["ok"], planned)
        return job, executor, t, state, now, sent, handles, calls

    def test_scoped_variants_reuse_actual_native_serializers_and_goal_consumer(self):
        for reference in (False, True):
            with self.subTest(reference=reference):
                job, executor, transport, state, now, sent, _, calls = self.make_held_job(scoped=True, mechanical=reference)
                p = job._program["learned_proposal"]
                expected = ("data_factory.finite_learned_serialized_reference_proposal.v2" if reference else
                            "data_factory.finite_learned_held_target_proposal.v2")
                self.assertEqual(p["schema_version"], expected)
                grant = task_grant(job._program["source_program"], SCENE, p)
                now[0] += .4
                self.assertTrue(job.admit_task(grant)["ok"])
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    self.assertTrue(job.start()["ok"])
                self.assertEqual(len(sent), 1)  # Actual serializer and native start method, synthetic action server only.
                self.assertNotIn(("recorder", "commit"), calls)
                self.assertEqual(executor.runs["run"]["task_deadline"], 100.)
                plain_job, *_ = self.make_job(scoped=True)
                plain = plain_job._program["learned_proposal"]
                raw = transport.build_learned_trajectory(plain)
                trajectory = transport._deserialize_message(raw, transport._RobotTrajectory).joint_trajectory
                self.assertEqual(trajectory.joint_names, JOINTS)
                self.assertEqual([list(point.positions) for point in trajectory.points], [INITIAL, ACTION])

    def test_native_percent_adaptation_preserves_transitions_and_exposes_changed_endpoint(self):
        data = json.loads(Path(__file__).with_name("recorded4032.json").read_text())
        sample = data["samples"][-1]
        job, executor, _, _, _, sent, _, _ = self.make_held_job(
            recorded=(sample, data["robot_description"], data["checkpoint"]), quantize_gripper=True)
        plan = executor.runs["run"]["plan"]
        p = plan["learned_proposal"]
        self.assertEqual(p["raw_actions"], sample["actions"])
        self.assertEqual([r[:6] for r in p["actions"]], [r[:6] for r in sample["actions"]])
        self.assertEqual(len(p["actions"]), 50)
        timing = p["reference_timing"]
        self.assertAlmostEqual(timing["max_gripper_change_m"], .00010245173782110102)
        self.assertAlmostEqual(timing["endpoint_delta_m"], -.00006693109482526667)
        self.assertTrue(timing["endpoint_changed"])
        segments = plan["steps"][0]["held_target_segments"]
        self.assertEqual(sum(s["type"] == "GRIPPER" for s in segments), 34)
        self.assertAlmostEqual(plan["steps"][0]["planned_duration_s"], 4.61)
        self.assertEqual(sent, [])
        staged = copy.deepcopy(job._program["source_program"])
        opened = next(s for s in staged["steps"] if s["phase"] == "GRIPPER_OPEN")
        opened.update(release_position_m=.015, release_hold_s=.5)
        staged_program = compile_program(staged, p)
        self.assertEqual(staged_program["source_program"], staged)
        self.assertEqual(staged_program["steps"], job._program["steps"])
        # Scripted staged-release context does not inject an extra policy row.
        # Only explicit percent-reference mode may project bounded gripper
        # endpoint noise. Raw policy output and all arm coordinates stay exact.
        limits = _limits(data["robot_description"])
        low, upper, _ = limits[-1]
        projection_slack = (
            GRIPPER_ENDPOINT_PROJECTION_QUANTA * upper / 100
        )

        for raw_sample in data["samples"][:-1]:
            obs = observation()
            obs["observation.state"] = raw_sample["initial_state"]

            # Retiming alone cannot legalize the out-of-range policy row.
            with self.assertRaisesRegex(
                ContractError, "LEARNED_JOINT_LIMIT"
            ):
                FinitePolicyInference(
                    lambda _: raw_sample["actions"],
                    data["checkpoint"],
                    source_clock=lambda: 10.,
                ).propose(
                    obs,
                    instruction="recorded CPU replay",
                    robot_description=data["robot_description"],
                    period_s=1 / 30,
                    serialized_references=True,
                    quantize_gripper=False,
                )

            adapted = FinitePolicyInference(
                lambda _: raw_sample["actions"],
                data["checkpoint"],
                source_clock=lambda: 10.,
            ).propose(
                obs,
                instruction="recorded CPU replay",
                robot_description=data["robot_description"],
                period_s=1 / 30,
                serialized_references=True,
                quantize_gripper=True,
            )

            self.assertEqual(
                adapted["raw_actions"], raw_sample["actions"]
            )
            self.assertEqual(
                [row[:6] for row in adapted["actions"]],
                [row[:6] for row in raw_sample["actions"]],
            )
            self.assertTrue(
                all(low <= row[-1] <= upper for row in adapted["actions"])
            )

            raw_excursion = max(
                max(low - row[-1], row[-1] - upper, 0.)
                for row in raw_sample["actions"]
            )
            self.assertGreater(raw_excursion, 0.)
            self.assertLessEqual(
                raw_excursion, projection_slack + 1e-12
            )

        # Anything beyond the explicit gripper projection budget stays illegal.
        bad_gripper = copy.deepcopy(data["samples"][0])
        bad_gripper["actions"][0][-1] = (
            upper + projection_slack + 1e-9
        )
        obs = observation()
        obs["observation.state"] = bad_gripper["initial_state"]
        with self.assertRaisesRegex(
            ContractError, "LEARNED_JOINT_LIMIT"
        ):
            FinitePolicyInference(
                lambda _: bad_gripper["actions"],
                data["checkpoint"],
                source_clock=lambda: 10.,
            ).propose(
                obs,
                instruction="recorded CPU replay",
                robot_description=data["robot_description"],
                period_s=1 / 30,
                serialized_references=True,
                quantize_gripper=True,
            )

        # Arm joint violations are never projected, even in percent mode.
        bad_arm = copy.deepcopy(data["samples"][0])
        bad_arm["actions"][0][0] = limits[0][1] + 1e-6
        obs = observation()
        obs["observation.state"] = bad_arm["initial_state"]
        with self.assertRaisesRegex(
            ContractError, "LEARNED_JOINT_LIMIT"
        ):
            FinitePolicyInference(
                lambda _: bad_arm["actions"],
                data["checkpoint"],
                source_clock=lambda: 10.,
            ).propose(
                obs,
                instruction="recorded CPU replay",
                robot_description=data["robot_description"],
                period_s=1 / 30,
                serialized_references=True,
                quantize_gripper=True,
            )
        for change in ("actions", "raw_actions", "reference_timing"):
            altered = copy.deepcopy(p)
            if change == "reference_timing":
                altered[change]["durations_s"][0] /= 2
            else:
                altered[change][0][0] += .0001
            with self.assertRaises(ContractError):
                validate_proposal(redigest(altered))

    def test_recorded4032_exact_references_reject_native_dispatch_budget(self):
        data = json.loads(Path(__file__).with_name("recorded4032.json").read_text())
        sample = data["samples"][-1]
        obs = observation()
        obs["observation.state"] = sample["initial_state"]
        p = FinitePolicyInference(lambda _: sample["actions"], data["checkpoint"], source_clock=lambda: 10.).propose(
            obs, instruction="recorded CPU replay", robot_description=data["robot_description"],
            period_s=1 / 30, serialized_references=True)
        self.assertEqual(p["actions"], sample["actions"])
        self.assertAlmostEqual(sum(p["reference_timing"]["durations_s"]) + 50 * .04, 5.18)
        src = source()
        src["binding_digests"]["robot_description_digest"] = 'sha256:' + hashlib.sha256(data["robot_description"].encode()).hexdigest()
        next(s for s in src["steps"] if s["phase"] == "GRIPPER_OPEN")["gripper_position_m"] = .021
        with self.assertRaisesRegex(ContractError, "LEARNED_HELD_HORIZON"):
            compile_program(src, p)

    def test_recorded4032_percent_references_reach_native_consumers_with_declared_changes(self):
        from tools.data_factory.quality.phase_events import validate_phase_event_sequence
        plan, events = self._recorded_reference_replay(True)
        self.assertEqual(validate_phase_event_sequence(events, plan=plan), events)
        event = next(item for item in events if item["event"] == "GOAL_ACCEPTED")
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_SEGMENT_BINDING"):
            validate_phase_event_sequence([{**event, "evidence_digest": "sha256:" + "0" * 64}], plan=plan)

    def _recorded_reference_replay(self, quantize):
        data = json.loads(Path(__file__).with_name("recorded4032.json").read_text())
        sample = data["samples"][-1]
        values = self.make_held_job(recorded=(sample, data["robot_description"], data["checkpoint"]), quantize_gripper=quantize)
        job, executor, transport, state, now, sent, _, calls = values
        plan = copy.deepcopy(executor.runs["run"]["plan"])
        events = []
        executor._phase_event_writer = SimpleNamespace(
            emit=lambda event: (events.append(event) or True), ready=True, error_code=None)
        executor.event_clock = lambda: (round(now[0] * 1e9), "SYSTEM_TIME")
        p = plan["learned_proposal"]
        self.assertEqual(p.get("raw_actions", p["actions"]), sample["actions"])
        if not quantize:
            self.assertEqual(p["actions"][-1], sample["actions"][-1])
        self.assertAlmostEqual(sum(p["reference_timing"]["durations_s"]), 3.25)
        segments = plan["steps"][0]["held_target_segments"]
        self.assertEqual(len(segments), 84 if quantize else 100)
        self.assertAlmostEqual(plan["steps"][0]["planned_duration_s"], 4.61)
        self.assertEqual(sent, [])  # all adaptation/compilation precedes approval
        job.approve(APPROVAL)
        self.assertTrue(job.start()["ok"])
        job.poll()
        consumed = []
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            self.assertTrue(job.confirm("operator")["ok"])
            for index, segment in enumerate(segments):
                self.assertEqual(len(sent), index + 1)
                self.assertEqual(executor.runs["run"]["plan"], plan)
                # No next command while native action/physical completion is pending.
                self.assertEqual(job.poll()["state"], "EXECUTING")
                self.assertEqual(len(sent), index + 1)
                if segment["type"] == "GRIPPER":
                    self.assertEqual([list(point.positions) for point in sent[-1].trajectory.points],
                                     [[segment["gripper_position_m"]]] * 2)
                    now[0] += segment["limits"]["command_duration_s"] + .002
                    state.update(reference=segment["gripper_position_m"], feedback=segment["gripper_position_m"])
                else:
                    begin, end = segment["action_range"]
                    points = sent[-1].trajectory.joint_trajectory.points
                    self.assertEqual([list(point.positions) for point in points[1:]],
                                     [row[:6] for row in p["actions"][begin:end]])
                    now[0] += sum(p["reference_timing"]["durations_s"][begin:end])
                    state["joints"] = segment["final_joint_state"][:]
                    consumed.extend(range(begin, end))
                # ROS header stamps have integer-nanosecond resolution. Keep
                # the synthetic source clock on that same grid, not 0.2 ns
                # before a header rounded from a nonuniform duration.
                now[0] = round(now[0], 9)
                state["complete"] = True
                result = job.poll()
                self.assertTrue(result["ok"], {"segment": index, "code": result["code"],
                                              "executor_failure": executor.runs["run"].get("failure_code")})
            self.assertEqual(result["state"], "LEARNED_CHUNK_COMPLETE", result)
        self.assertEqual(consumed, list(range(50)))
        self.assertEqual((transport._execute_goal_count, transport._gripper_goal_count), (50, 34 if quantize else 50))
        self.assertLess(now[0] - 10., 5.)
        self.assertNotIn(("recorder", "commit"), calls)
        self.assertEqual(executor.runs["run"]["plan"]["learned_proposal"].get("raw_actions", p["actions"]), sample["actions"])
        from tools.data_factory.rollout.finite_plan import validate_execution_trace
        trace = executor._execution_data(executor.runs["run"])["learned_execution"]
        checked = validate_execution_trace(plan, trace)
        self.assertEqual(checked["reference_consumption"]["completed_row_indices"], list(range(50)))
        self.assertEqual(checked["reference_consumption"]["completed_gripper_generations"], list(range(1, 35 if quantize else 51)))
        self.assertFalse(checked["reference_consumption"]["physical_sample_at_every_knot_proven"])
        broken = copy.deepcopy(trace)
        broken["reference_consumption"]["completed_row_indices"].pop()
        broken["trace_digest"] = canonical_digest({k: v for k, v in broken.items() if k != "trace_digest"})
        with self.assertRaisesRegex(ContractError, "LEARNED_TRACE_BINDING"):
            validate_execution_trace(plan, broken)
        # Return the actual executor events for the existing Quality consumer.
        return plan, events

    def test_serialized_references_deadline_and_changed_initial_reference_send_no_next_goal(self):
        data = json.loads(Path(__file__).with_name("recorded4032.json").read_text())
        for failure in ("start_reference", "wall_timeout"):
            with self.subTest(failure=failure):
                job, executor, t, state, now, sent, handles, _ = self.make_held_job(
                    recorded=(data["samples"][-1], data["robot_description"], data["checkpoint"]), quantize_gripper=True)
                job.approve(APPROVAL)
                job.start()
                job.poll()
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    if failure == "start_reference":
                        state["reference"] += .00001
                        self.assertEqual(job.confirm("operator")["code"], "LEARNED_START_STATE")
                        self.assertEqual(sent, [])
                    else:
                        self.assertTrue(job.confirm("operator")["ok"])
                        for _ in range(9):
                            now[0] += .5
                            self.assertEqual(job.poll()["state"], "EXECUTING")
                        now[0] += .501
                        result = job.poll()
                        self.assertEqual(result["code"], "LEARNED_REFERENCE_TIMEOUT")
                        self.assertEqual(len(sent), 1)
                        handles[0].cancel_goal_async.assert_called_once()

    def test_held_reference_replay_uses_one_gripper_goal_then_fresh_arm_start(self):
        job, executor, transport, state, now, sent, _, calls = self.make_held_job(initial_feedback=.02079, controller_samples=True)
        frozen = copy.deepcopy(executor.runs["run"]["plan"])
        from tools.data_factory.quality.phase_events import validate_phase_event, validate_phase_event_sequence
        events = []
        def emit(record):
            events.append(validate_phase_event(record, plan=frozen))
            return True
        executor._phase_event_writer = SimpleNamespace(emit=emit, ready=True, error_code=None)
        executor.event_clock = lambda: (int(now[0] * 1e9), "SYSTEM_TIME")
        self.assertEqual(sent, [])

        self.assertEqual(calls, [])
        self.assertTrue(job.approve(APPROVAL)["ok"])
        self.assertTrue(job.start()["ok"])
        self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            confirmed = job.confirm("operator")
            self.assertTrue(confirmed["ok"], confirmed["code"])
            self.assertEqual(len(sent), 1)
            state["complete"] = True  # first arm segment completed at open reference
            job.poll()
            self.assertEqual(len(sent), 2)
            self.assertEqual([list(p.positions) for p in sent[1].trajectory.points], [[.01176], [.01176]])
            for _ in range(11):
                job.poll()
            self.assertEqual(len(sent), 2)  # no repeated close / no early arm send
            now[0] += 11 / 30
            state.update(reference=.01176, feedback=.01218)
            job.poll()
            self.assertEqual(len(sent), 2)  # matching feedback is not terminal evidence
            now[0] += 1.01 - 11 / 30
            state["complete"] = True
            job.poll()
            self.assertEqual(len(sent), 3)
            self.assertEqual(transport._gripper_goal_count, 1)
            self.assertEqual(list(sent[2].trajectory.joint_trajectory.joint_names), JOINTS[:6])
            self.assertEqual(list(sent[2].trajectory.joint_trajectory.points[0].positions), [0.] * 6)
            start = executor.runs["run"]["execution"]["learned_start_observation"]
            self.assertEqual(start["snapshot"]["gripper_controller"]["feedback_position_m"], .01218)
            self.assertEqual(start["captured_at_s"], now[0])
            state.update(complete=True, joints=[.001] * 6)
            self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(executor.runs["run"]["plan"], frozen)
        self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
        diagnostic = learned_run_diagnostic(job.poll())
        self.assertEqual(diagnostic["execution_trace"]["status"], "COMPLETED")
        self.assertEqual(len(diagnostic["execution_trace"]["segments"]), 3)
        self.assertEqual(diagnostic["task_effectiveness"], "UNKNOWN")
        retained = diagnostic["execution_trace"]["segments"][1]["terminal_observation"]["snapshot"]
        sample = retained["gripper_controller"]["sample"]
        self.assertEqual((sample["reference_elapsed_ns"], sample["feedback_elapsed_ns"]), (-200000000, -100000000))
        self.assertEqual(sample["reference_positions"], [.01176])
        self.assertEqual(sample["feedback_positions"], [.01218])
        self.assertEqual(sample["reported_output_positions"], [.021])
        self.assertEqual(retained["arm_controller"]["sample"]["reported_output_positions"], [])
        self.assertEqual(sample["ros_stamp_ns"], round(diagnostic["execution_trace"]["segments"][1]["terminal_observation"]["captured_at_s"] * 1e9))
        self.assertNotIn(("recorder", "commit"), calls)
        self.assertEqual(validate_phase_event_sequence(events, plan=frozen), events)
        for event_type in ("GOAL_ACCEPTED", "ACTION_TERMINAL"):
            self.assertEqual([(e["segment_index"], e["segment_count"]) for e in events if e["event"] == event_type],
                             [(0, 3), (1, 3), (2, 3)])
        from tools.data_factory.rollout.finite_plan import validate_execution_trace
        for override, code in (({"reference_elapsed_ns": True}, "ROS_CONTROLLER_SAMPLE"),
                               ({"reference_positions": [.012]}, "ROS_CONTROLLER_SAMPLE_BINDING")):
            trace = copy.deepcopy(diagnostic["execution_trace"])
            trace["segments"][1]["terminal_observation"]["snapshot"]["gripper_controller"]["sample"].update(override)
            trace["trace_digest"] = canonical_digest({k: v for k, v in trace.items() if k != "trace_digest"})
            with self.assertRaisesRegex(ContractError, code):
                validate_execution_trace(frozen, trace)
        for controller in ("arm_controller", "gripper_controller"):
            trace = copy.deepcopy(diagnostic["execution_trace"])
            trace["segments"][1]["terminal_observation"]["snapshot"][controller]["speed_scaling"] = 0.
            trace["trace_digest"] = canonical_digest({k: v for k, v in trace.items() if k != "trace_digest"})
            with self.assertRaisesRegex(ContractError, "LEARNED_CONTROLLER_PAUSED"):
                validate_execution_trace(frozen, trace)
        for field, value, code in (("generation", 2., "LEARNED_HARDWARE_SUPERSEDED"),
                                    ("completion_reason", 0., "LEARNED_HARDWARE_COMPLETION"),
                                    ("sample_steady_s", 0., "LEARNED_HARDWARE_STALE")):
            trace = copy.deepcopy(diagnostic["execution_trace"])
            trace["segments"][1]["terminal_observation"]["snapshot"]["gripper_controller"]["hardware_execution"]["wire"][field] = value
            trace["trace_digest"] = canonical_digest({k: v for k, v in trace.items() if k != "trace_digest"})
            with self.assertRaisesRegex(ContractError, code):
                validate_execution_trace(frozen, trace)
        trace = copy.deepcopy(diagnostic["execution_trace"])
        trace["terminal_state"][0] += .001
        trace["trace_digest"] = canonical_digest({k: v for k, v in trace.items() if k != "trace_digest"})
        with self.assertRaisesRegex(ContractError, "LEARNED_TRACE_TERMINAL"):
            validate_execution_trace(frozen, trace)

    def test_invalid_controller_sample_fences_before_first_goal(self):
        for override in ({"ros_stamp_ns": -1}, {"reference_elapsed_ns": True},
                         {"reported_output_positions": [float("nan")]}, {"joint_names": []}):
            with self.subTest(override=override):
                job, _, _, state, now, sent, _, calls = self.make_held_job(controller_samples=True)
                job.approve(APPROVAL)
                job.start()
                job.poll()
                state["sample_override"] = {"gripper_controller": override}
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    result = job.confirm("operator")
                self.assertEqual(result["code"], "ROS_CONTROLLER_SAMPLE")
                self.assertEqual(sent, [])
                self.assertNotIn(("recorder", "commit"), calls)

    def test_native_hardware_fences_before_first_goal(self):
        cases = [({"incarnation_0": 9.}, "LEARNED_HARDWARE_INCARNATION"),
                 ({"sample_steady_s": 0.}, "LEARNED_HARDWARE_STALE"),
                 ({"second": 0.}, "LEARNED_HARDWARE_STALE"),
                 ({"pending": 1.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"stopped": 1.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"error": -1.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"valid": 0.}, "LEARNED_HARDWARE_INVALID"),
                 ({"generation": float(2**53)}, "LEARNED_HARDWARE_SCHEMA")]
        for override, code in cases:
            with self.subTest(override=override):
                job, executor, t, state, now, sent, _, calls = self.make_held_job()
                job.approve(APPROVAL)
                job.start()
                job.poll()
                state["hardware_override"] = override
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    result = job.confirm("operator")
                self.assertEqual(result["code"], code)
                self.assertEqual(sent, [])
                self.assertNotIn(("recorder", "commit"), calls)
        # Installed old driver or absent measured mapping cannot acquire completion authority.
        job, _, t, _, now, sent, _, _ = self.make_held_job()
        t._gripper_source_clock = None
        job.approve(APPROVAL)
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', return_value=now[0]):
            self.assertEqual(job.start()["code"], "LEARNED_HARDWARE_SCHEMA")
        self.assertEqual(sent, [])

    def test_same_command_hardware_completion_required_after_jtc_success(self):
        cases = [({"completed_generation": 0.}, "LEARNED_HARDWARE_COMPLETION"),
                 ({"generation": 2., "completed_generation": 2.}, "LEARNED_HARDWARE_SUPERSEDED"),
                 ({"completion_second": 9.}, "LEARNED_HARDWARE_COMPLETION"),
                 ({"arm_resumed": 0.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"rpc_active": 1.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"incarnation_1": 17.}, "LEARNED_HARDWARE_INCARNATION"),
                 ({"raw_reference_m": .012}, "GRIPPER_FEEDBACK_OUT_OF_RANGE")]
        for override, code in cases:
            with self.subTest(override=override):
                job, executor, t, state, now, sent, _, calls = self.make_held_job()
                job.approve(APPROVAL)
                job.start()
                job.poll()
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    job.confirm("operator")
                    state["complete"] = True
                    job.poll()
                    self.assertEqual(len(sent), 2)
                    now[0] += .1
                    state.update(complete=True, reference=.01176, feedback=.01218, hardware_override=override)
                    result = job.poll()
                self.assertEqual(result["code"], code)
                self.assertEqual(len(sent), 2)  # no next arm, despite JTC SUCCEEDED
                self.assertTrue(t.owns_active_goal)
                terminal = t.poll_terminal_evidence()
                self.assertEqual(terminal["result_status"], 4)  # actual JTC SUCCEEDED, not fabricated cancel
                self.assertFalse(t.owns_active_goal)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_paused_system_clock_after_native_deserialization_prevents_send(self):
        job, _, t, _, now, sent, _, _ = self.make_held_job()
        job.approve(APPROVAL)
        job.start()
        job.poll()
        deserialize = t._deserialize_message
        def delayed(*args):
            goal = deserialize(*args)
            t._clock = lambda: now[0] + .2
            return goal
        t._deserialize_message = delayed
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', return_value=now[0]):
            self.assertEqual(job.confirm("operator")["code"], "LEARNED_HARDWARE_SOURCE_CLOCK")
        self.assertEqual(sent, [])

    def make_pending_held_job(self):
        values = self.make_held_job()
        job, executor, t, state, now, sent, handles, calls = values
        job.approve(APPROVAL)
        job.start()
        job.poll()
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            self.assertTrue(job.confirm("operator")["ok"])
            state["complete"] = True
            job.poll()
            now[0] += .5
            job.poll()
            now[0] += .51
            state.update(complete=True, reference=.01176, feedback=.01218,
                         hardware_override={"active_generation": 1., "completed_generation": 0.,
                                            "completion_reason": 0., "arm_resumed": 0.})
            self.assertEqual(job.poll()["state"], "EXECUTING")
        self.assertEqual(len(sent), 2)
        self.assertTrue(t.owns_active_goal)
        self.assertTrue(t._active.action_succeeded)
        return values

    def test_later_native_completion_advances_once_with_original_deadline_and_targets(self):
        job, executor, t, state, now, sent, handles, calls = self.make_pending_held_job()
        frozen = copy.deepcopy(executor.runs["run"]["plan"])
        deadline = t._active.deadline
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            for _ in range(3):
                self.assertEqual(job.poll()["state"], "EXECUTING")
                self.assertEqual(t._active.deadline, deadline)
                self.assertIsNone(t.poll_terminal_evidence())
                self.assertTrue(t.owns_active_goal)
            with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
                t.start_phase(t._active.held_segment)
            now[0] += .1
            delivery = mock.Mock(side_effect=lambda *args, **kwargs: state.update(hardware_override={}))
            t._rclpy.spin_once = delivery
            self.assertEqual(job.poll()["state"], "EXECUTING")
            delivery.assert_called_once_with(t.node, timeout_sec=0.0)
            self.assertEqual(len(sent), 3)
            self.assertEqual(t._gripper_goal_count, 1)
            state.update(complete=True, joints=[.001] * 6)
            self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(executor.runs["run"]["plan"], frozen)
        self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
        trace = learned_run_diagnostic(job.poll())["execution_trace"]
        self.assertEqual(trace["status"], "COMPLETED")
        self.assertEqual(len(trace["segments"]), 3)
        terminal = trace["segments"][1]["terminal_observation"]
        self.assertEqual(terminal["captured_at_s"], 11.11)
        self.assertEqual(terminal["action_terminal"], {"result_status": 4, "error_code": 0,
                         "observed_at_s": 11.01, "observed_monotonic_s": 11.01})
        from tools.data_factory.rollout.finite_plan import validate_execution_trace
        # MoveIt and FollowJointTrajectory use different success codes. Keep
        # the actual ARM result readable without accepting a gripper code for it.
        arm_trace = copy.deepcopy(trace)
        arm_terminal = arm_trace["segments"][0]["terminal_observation"]
        arm_terminal["action_terminal"] = {
            "result_status": 4, "error_code": 1,
            "observed_at_s": arm_terminal["captured_at_s"],
            "observed_monotonic_s": arm_terminal["captured_monotonic_s"],
        }
        arm_trace["trace_digest"] = canonical_digest({k: v for k, v in arm_trace.items() if k != "trace_digest"})
        self.assertEqual(validate_execution_trace(frozen, arm_trace), arm_trace)
        arm_terminal["action_terminal"]["error_code"] = 0
        arm_trace["trace_digest"] = canonical_digest({k: v for k, v in arm_trace.items() if k != "trace_digest"})
        with self.assertRaisesRegex(ContractError, "LEARNED_TRACE_TERMINAL"):
            validate_execution_trace(frozen, arm_trace)
        for field, value in (("result_status", 5), ("observed_at_s", 11.12), ("observed_monotonic_s", 9.)):
            altered = copy.deepcopy(trace)
            altered["segments"][1]["terminal_observation"]["action_terminal"][field] = value
            altered["trace_digest"] = canonical_digest({k: v for k, v in altered.items() if k != "trace_digest"})
            with self.assertRaisesRegex(ContractError, "LEARNED_TRACE_TERMINAL"):
                validate_execution_trace(frozen, altered)
        self.assertNotIn(("recorder", "commit"), calls)
        handles[1].cancel_goal_async.assert_not_called()

    def test_arm_streaming_during_active_gripper_is_observation_not_completion(self):
        # Actual existing action consumer, synthetic controller telemetry only.
        job, executor, transport, state, now, sent, handles, _ = self.make_pending_held_job()
        active, deadline = transport._active, transport._active.deadline
        state["hardware_override"]["arm_resumed"] = 1.
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            for _ in range(3):
                self.assertEqual(job.poll()["state"], "EXECUTING")
                self.assertIs(transport._active, active)
                self.assertEqual(transport._active.deadline, deadline)
                self.assertTrue(transport.owns_active_goal)
                self.assertEqual(len(sent), 2)
            state["hardware_override"] = {}
            self.assertEqual(job.poll()["state"], "EXECUTING")
            self.assertEqual(len(sent), 3)  # Only completed-generation evidence advances.
        self.assertEqual(transport._gripper_goal_count, 1)
        handles[1].cancel_goal_async.assert_not_called()

    def test_pending_hardware_faults_are_not_retryable_waits(self):
        cases = [({"stopped": 1.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"error": -1.}, "LEARNED_HARDWARE_UNRESOLVED"),
                 ({"sample_steady_s": 0.}, "LEARNED_HARDWARE_STALE"),
                 ({"second": 0.}, "LEARNED_HARDWARE_STALE"),
                 ({"incarnation_0": 7.}, "LEARNED_HARDWARE_INCARNATION"),
                 ({"generation": 2., "active_generation": 2.}, "LEARNED_HARDWARE_SUPERSEDED"),
                 ({"raw_reference_m": .013}, "GRIPPER_FEEDBACK_OUT_OF_RANGE"),
                 ({"completion_reason": 3.}, "LEARNED_HARDWARE_SCHEMA"),
                 ({"completion_reason": 1.}, "LEARNED_HARDWARE_COMPLETION")]
        for override, code in cases:
            with self.subTest(override=override):
                job, executor, t, state, now, sent, handles, calls = self.make_pending_held_job()
                state["hardware_override"].update(override)
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    self.assertEqual(job.poll()["code"], code)
                self.assertEqual(len(sent), 2)
                self.assertEqual(executor.runs["run"]["state"], "BLOCKED")
                self.assertEqual(t.poll_terminal_evidence()["result_status"], 4)
                handles[1].cancel_goal_async.assert_not_called()
                self.assertNotIn(("recorder", "commit"), calls)

    def test_expected_queued_command_can_wait_without_claiming_it_started(self):
        for flag in ("pending", "rpc_active"):
            with self.subTest(flag=flag):
                job, _, t, state, now, sent, handles, _ = self.make_pending_held_job()
                state["hardware_override"].update(active_generation=0., command_started_system_s=0.)
                state["hardware_override"][flag] = 1.
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    self.assertEqual(job.poll()["state"], "EXECUTING")
                    self.assertTrue(t.owns_active_goal)
                    job.cancel()
                self.assertEqual(t.poll_terminal_evidence()["result_status"], 4)
                self.assertEqual(len(sent), 2)
                handles[1].cancel_goal_async.assert_not_called()

    def test_pending_wait_rejects_a_changed_controller_reference(self):
        job, executor, t, state, now, sent, handles, _ = self.make_pending_held_job()
        observe = t.snapshot
        def changed(*args):
            value = observe(*args)
            value["gripper_controller"]["reference_position_m"] = .021
            value["gripper_controller"]["sample"]["reference_positions"] = [.021]
            return value
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            with mock.patch.object(t, "snapshot", side_effect=changed):
                self.assertEqual(job.poll()["code"], "GRIPPER_FEEDBACK_OUT_OF_RANGE")
        self.assertEqual(len(sent), 2)
        self.assertEqual(t.poll_terminal_evidence()["result_status"], 4)
        handles[1].cancel_goal_async.assert_not_called()

    def test_pending_native_wait_preserves_deadline_cancel_and_real_action_terminal(self):
        for failure in ("deadline", "late_snapshot", "lease", "cancel", "paused_system"):
            with self.subTest(failure=failure):
                job, executor, t, state, now, sent, handles, _ = self.make_pending_held_job()
                step = copy.deepcopy(t._active.held_segment)
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    if failure == "deadline":
                        wall = t._active.deadline + .001
                        t._clock = lambda: wall
                        result = job.poll()
                        self.assertEqual(result["code"], "LEARNED_HARDWARE_COMPLETION_TIMEOUT")
                    elif failure == "late_snapshot":
                        observe = t.snapshot
                        def delayed(*args):
                            value = observe(*args)
                            t._clock = lambda: t._active.deadline + .001
                            return value
                        with mock.patch.object(t, "snapshot", side_effect=delayed):
                            result = job.poll()
                        self.assertEqual(result["code"], "LEARNED_HARDWARE_COMPLETION_TIMEOUT")
                    elif failure == "lease":
                        now[0] += 2.
                        result = job.poll()
                        self.assertEqual(result["code"], "HEARTBEAT_TIMEOUT")
                    elif failure == "paused_system":
                        t._clock = lambda: now[0] + .2
                        result = job.poll()
                        self.assertEqual(result["code"], "LEARNED_HARDWARE_SOURCE_CLOCK")
                    else:
                        job.cancel()
                        self.assertEqual(executor.runs["run"]["failure_code"], "CANCELLED_BY_OPERATOR")
                    self.assertTrue(t.owns_active_goal)
                    self.assertEqual(executor.runs["run"]["execution"]["cancel_error"], "ROS_EXEC_CANCEL_NOT_CANCELED")
                    terminal = t.poll_terminal_evidence()
                    self.assertEqual(terminal["result_status"], 4)
                    self.assertFalse(t.owns_active_goal)
                    state["hardware_override"] = {}
                    job.poll()
                    with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
                        t.start_phase(step)
                self.assertEqual(len(sent), 2)
                handles[1].cancel_goal_async.assert_not_called()

    def test_native_single_clock_interpolation_cannot_replace_mixed_reference_contract(self):
        import pathlib
        import subprocess
        import tempfile
        job, executor, transport, _, _, _, _, _ = self.make_held_job(initial_feedback=.02079)
        p = copy.deepcopy(executor.runs["run"]["plan"]["learned_proposal"])
        p.update(schema_version="data_factory.finite_learned_proposal.v1",
                 actions=[[.006 * i] * 6 + [.021 - .00011 * i] for i in range(1, 4)])
        p = redigest(p)
        # Actual native proposal validation and serializer, including unmodified
        # original 7D rows/timestamps, feed the installed C++ sampler.
        raw = transport.build_learned_trajectory(p)
        trajectory = transport._deserialize_message(raw, transport._RobotTrajectory).joint_trajectory
        self.assertEqual([list(point.positions) for point in trajectory.points], [p["initial_state"], *p["actions"]])
        lines = [str(len(trajectory.points))]
        for point in trajectory.points:
            ns = point.time_from_start.sec * 10**9 + point.time_from_start.nanosec
            lines.append(" ".join(map(str, [ns, *point.positions])))
        data = "\n".join(lines) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "sampling"
            ros = pathlib.Path("/opt/ros/jazzy")
            fixture = pathlib.Path(__file__).with_name("native_sampling_fixture.cpp")
            command = ["g++", "-std=c++17", *["-I" + str(x) for x in (ros / "include").iterdir() if x.is_dir()],
                       str(fixture), "-L" + str(ros / "lib"), "-Wl,-rpath," + str(ros / "lib"),
                       "-ljoint_trajectory_controller", "-lrclcpp", "-lrcutils", "-o", str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            def sample(mode, first_grip=0, pause=0, prior_reference=None, points=data, immediate=False):
                command = [str(binary), mode, "0", str(first_grip), str(pause)]
                if prior_reference is not None:
                    command.append(str(prior_reference))
                if immediate:
                    command.append("immediate")
                result = subprocess.run(command,
                                        input=points, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return [[float(x) for x in line.split()] for line in result.stdout.splitlines()]
            mixed, none, spline = [sample(mode) for mode in ("mixed", "none", "spline")]
            self.assertTrue(all(a[3:9] == b[3:9] for a, b in zip(mixed, spline)))
            self.assertTrue(all(a[-1] == b[-1] for a, b in zip(mixed, none)))
            # Finite-difference command rates, NOT measured physical angular speed.
            arm_rate = lambda rows: max(abs(b[3] - a[3]) / .01 for a, b in zip(rows, rows[1:]))
            self.assertLess(arm_rate(mixed), .181)
            self.assertAlmostEqual(arm_rate(none), .6)
            refs = [p["initial_state"][-1], *[row[-1] for row in p["actions"]]]
            self.assertTrue(all(row[-1] in refs for row in mixed))
            self.assertTrue(any(all(abs(row[-1] - value) > 1e-12 for value in refs) for row in spline))
            # Common future header + equal factor is insufficient if the two
            # controllers first sample in different cycles while time is paused.
            staggered = sample("mixed", first_grip=3, pause=6)
            self.assertAlmostEqual(staggered[10][2] - staggered[10][1], .03)
            running = sample("mixed", first_grip=3, pause=0)
            self.assertAlmostEqual(running[10][2] - running[10][1], 0.)

            desired_start = sample("mixed", prior_reference=.021)
            self.assertEqual(mixed[0][-1], p["initial_state"][-1])
            self.assertEqual(desired_start[0][-1], .021)
            self.assertTrue(all(a[-1] == b[-1] for a, b in zip(mixed[20:], desired_start[20:])))

            # Existing staged release uses two immediate, constant endpoint
            # goals. Its actual serialized target/hold is unchanged by the
            # prior-reference initialization branch. This is sampling evidence,
            # not proof that either hardware stage completed or held physically.
            for target, duration, previous in ((.0126, .5, .01), (.021, 1., .0126)):
                with self.subTest(release_target=target):
                    limits = dict(command_duration_s=duration, execution_timeout_s=2.,
                                  completion_tolerance_m=.000105)
                    encoded = transport.build_gripper_goal("GRIPPER_OPEN", target, limits)
                    goal = transport._deserialize_message(encoded, transport._FollowJointTrajectory.Goal)
                    self.assertEqual(goal.trajectory.header.stamp.sec, 0)
                    self.assertEqual(goal.trajectory.header.stamp.nanosec, 0)
                    release = [str(len(goal.trajectory.points))]
                    for point in goal.trajectory.points:
                        ns = point.time_from_start.sec * 10**9 + point.time_from_start.nanosec
                        self.assertEqual(list(point.positions), [target])
                        # Padding supplies unrelated zero arm coordinates only
                        # to the fixture; the native gripper sampler remains 1D.
                        release.append(" ".join(map(str, [ns, *([0.] * 6), *point.positions])))
                    self.assertEqual(ns / 1e9, duration)
                    stream = sample("mixed", prior_reference=previous,
                                    points="\n".join(release) + "\n", immediate=True)
                    self.assertTrue(all(row[-1] == target for row in stream))

    def test_held_float32_profile_reference_is_preserved_without_snapping(self):
        import struct
        job, executor, _, _, _, _, _, _ = self.make_held_job()
        p = copy.deepcopy(executor.runs["run"]["plan"]["learned_proposal"])
        for row in p["actions"]:
            row[-1] = struct.unpack('f', struct.pack('f', row[-1]))[0]
        program = compile_program(job._program["source_program"], redigest(p))
        segments = program["steps"][0]["held_target_segments"]
        target = p["actions"][4][-1]
        self.assertNotEqual(target, .01176)
        self.assertEqual(next(s for s in segments if s["action_range"] == [4, 4])["gripper_position_m"], target)

    def test_paused_controller_blocks_held_start_and_terminal_handoff(self):
        for controller in ("arm_controller", "gripper_controller"):
            for at_terminal in (False, True):
                with self.subTest(controller=controller, at_terminal=at_terminal):
                    job, executor, t, state, now, sent, _, _ = self.make_held_job()
                    job.approve(APPROVAL)
                    job.start()
                    job.poll()
                    observe = t.snapshot
                    def paused(*args):
                        value = observe(*args)
                        value[controller]["speed_scaling"] = 0.
                        return value
                    with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                        if at_terminal:
                            job.confirm("operator")
                            state["complete"] = True
                            job.poll()
                            state.update(complete=True, reference=.01176, feedback=.01218)
                        with mock.patch.object(t, 'snapshot', side_effect=paused):
                            result = job.poll() if at_terminal else job.confirm("operator")
                    self.assertEqual(result["code"], "LEARNED_CONTROLLER_PAUSED")
                    self.assertEqual(len(sent), 2 if at_terminal else 0)
                    self.assertEqual(executor.runs["run"]["state"], "BLOCKED")

    def test_frozen_controller_time_does_not_suspend_transport_deadline_or_cancel_owner(self):
        job, executor, t, state, now, sent, handles, _ = self.make_held_job()
        wall = [10.]
        t._clock = lambda: wall[0]
        job.approve(APPROVAL)
        job.start()
        job.poll()
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            job.confirm("operator")
            state["complete"] = True
            job.poll()
            # Source/controller observations stay frozen; only the transport's
            # monotonic clock advances while the gripper result is unresolved.
            wall[0] = t._active.deadline + .001
            result = job.poll()
            job.poll()
        self.assertEqual(result["code"], "ROS_EXEC_RESULT_TIMEOUT")
        self.assertEqual(len(sent), 2)
        handles[-1].cancel_goal_async.assert_called_once()
        self.assertFalse(t.owns_active_goal)
        self.assertEqual(executor.runs["run"]["state"], "BLOCKED")

    def test_held_reference_replay_failure_staleness_and_cancel_never_send_next_arm(self):
        failures = {"stale": "LEARNED_STALE_STATE", "future": "LEARNED_STALE_STATE",
                    "feedback": "GRIPPER_FEEDBACK_OUT_OF_RANGE", "reference": "GRIPPER_FEEDBACK_OUT_OF_RANGE",
                    "arm_drift": "LEARNED_TERMINAL_STATE", "aborted": "ROS_EXEC_FAILED",
                    "cancel": "CANCELLED_BY_OPERATOR", "unresolved": "CANCELLED_BY_OPERATOR"}
        for failure, expected in failures.items():
            with self.subTest(failure=failure):
                job, executor, t, state, now, sent, handles, _ = self.make_held_job()
                job.approve(APPROVAL)
                job.start()
                job.poll()
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    confirmed = job.confirm("operator")
                    self.assertTrue(confirmed["ok"], confirmed["code"])
                    state["complete"] = True
                    job.poll()
                    if failure == "cancel":
                        job.cancel()
                    elif failure == "unresolved":
                        handles[-1].cancel_goal_async.return_value = None
                        handles[-1].cancel_goal_async.side_effect = lambda: mock.Mock(done=lambda: False)
                        job.cancel()
                    else:
                        now[0] += .1
                        state.update(complete=True, reference=.01176, feedback=.01218)
                        if failure == "stale": state["age"] = 2.
                        if failure == "future": state["age"] = -1.
                        if failure == "feedback": state["feedback"] = .014
                        if failure == "reference": state["reference"] = .021
                        if failure == "arm_drift": state["joints"] = [.1] * 6
                        if failure == "aborted": handles[-1].get_result_async.return_value.result.return_value.status = 6
                        job.poll()
                    job.poll()
                    self.assertEqual(len(sent), 2)
                    self.assertEqual(executor.runs["run"]["state"], "BLOCKED")
                    self.assertEqual(executor.runs["run"]["failure_code"], expected)
                    if failure in {"cancel", "unresolved"}:
                        handles[-1].cancel_goal_async.assert_called_once()
                    if failure == "unresolved":
                        self.assertTrue(t.owns_active_goal)

    def test_held_target_contract_rejects_unbound_targets_limits_and_unbounded_holds(self):
        job, executor, transport, _, _, sent, _, _ = self.make_held_job()
        p = executor.runs["run"]["plan"]["learned_proposal"]
        src = job._program["source_program"]
        cases = [
            ("LEARNED_UNBOUND_GRIPPER_TARGET", lambda value: value["actions"][4].__setitem__(6, .012)),
            ("LEARNED_JOINT_LIMIT", lambda value: value["actions"][4].__setitem__(6, .022)),
            ("LEARNED_VELOCITY_LIMIT", lambda value: value["actions"][4].__setitem__(0, 2.)),
            ("LEARNED_ACTION_7D", lambda value: value["actions"].__setitem__(4, [0.] * 6)),
            ("LEARNED_HELD_HORIZON", lambda value: value.update(actions=[[0.] * 6 + [(.01176 if i % 2 == 0 else .021)] for i in range(6)])),
            ("LEARNED_VELOCITY_LIMIT", lambda value: value.update(schema_version="data_factory.finite_learned_proposal.v1")),
        ]
        for code, mutate in cases:
            with self.subTest(code=code):
                value = copy.deepcopy(p)
                mutate(value)
                with self.assertRaisesRegex(ContractError, code):
                    compile_program(src, redigest(value))
        with self.assertRaisesRegex(ContractError, "LEARNED_HELD_SEGMENTS_REQUIRED"):
            transport.build_learned_trajectory(p)
        self.assertEqual(sent, [])

    def test_continuous_references_and_staged_source_reject_without_rewriting_inputs(self):
        # These are in-limit continuous outputs, not an inferred close/open class.
        # A production staged source must not be stripped to make them executable.
        job, executor, _, _, now, sent, _, calls = self.make_held_job()
        xml = executor.runs["run"]["plan"]["learned_proposal"]["robot_description"]
        for staged, expected in ((False, "LEARNED_UNBOUND_GRIPPER_TARGET"),
                                 (True, "LEARNED_HELD_PROFILE_UNSUPPORTED")):
            with self.subTest(staged=staged):
                src = copy.deepcopy(job._program["source_program"])
                if staged:
                    opened = next(s for s in src["steps"] if s["phase"] == "GRIPPER_OPEN")
                    opened.update(release_position_m=.0126, release_hold_s=.5)
                validate_motion_program(src)
                actions = [[0.] * 6 + [.016342543065547943 + i * .00001] for i in range(50)]
                original_source, original_actions = copy.deepcopy(src), copy.deepcopy(actions)
                policy = mock.Mock(return_value=actions)
                inference = FinitePolicyInference(policy, CHECKPOINT, source_clock=lambda: now[0])
                consumer = OneJob(Recorder(calls), executor.process)
                obs = observation()
                obs["observation.state"] = [0.] * 6 + [.021]
                result = consumer.plan_learned(
                    "continuous", src, SCENE, inference, obs,
                    **{**OPTIONS, "robot_description": xml, "period_s": 1 / 30,
                       "held_gripper_targets": True, "max_observation_age_s": 5.},
                )
                self.assertFalse(result["ok"])
                self.assertEqual(result["code"], expected)
                policy.assert_called_once()
                self.assertEqual(src, original_source)
                self.assertEqual(actions, original_actions)
                self.assertNotIn("continuous", executor.runs)
                self.assertEqual(sent, [])
                self.assertEqual(calls, [])

    def test_held_start_snapshot_is_rechecked_after_deserialization_before_send(self):
        job, executor, t, _, now, sent, _, _ = self.make_held_job()
        job.approve(APPROVAL)
        job.start()
        job.poll()
        original = t._compiled_execution_goal
        def slow_decode(step):
            result = original(step)
            now[0] += 1.1  # proposal is still valid; the arm-start observation is stale
            return result
        with mock.patch.object(t, '_compiled_execution_goal', side_effect=slow_decode), mock.patch(
                'tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            result = job.confirm("operator")
        self.assertEqual(result["code"], "LEARNED_STALE_STATE")
        self.assertEqual(sent, [])
        self.assertEqual(executor.runs["run"]["state"], "BLOCKED")

    def test_native_transport_rejects_changed_held_arm_message(self):
        _, executor, t, _, _, sent, _, _ = self.make_held_job()
        step = copy.deepcopy(executor.runs["run"]["plan"]["steps"][0]["held_target_segments"][0])
        message = t._deserialize_message(base64.b64decode(step["trajectory_b64"]), t._RobotTrajectory)
        message.joint_trajectory.points[-1].positions[0] += .001
        step["trajectory_b64"] = base64.b64encode(t._serialize_message(message)).decode()
        with self.assertRaisesRegex(ContractError, "LEARNED_SERIALIZED_ACTION_MISMATCH"):
            t.start_phase(step)
        self.assertEqual(sent, [])

    def test_cancel_during_held_completion_snapshot_fences_late_next_segment(self):
        job, executor, t, state, now, sent, _, _ = self.make_held_job()
        job.approve(APPROVAL)
        job.start()
        job.poll()
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            job.confirm("operator")
            state["complete"] = True
            job.poll()
            now[0] += .1
            state.update(complete=True, reference=.01176, feedback=.01218)
            observe = t.snapshot
            def late_observation(*args):
                value = observe(*args)
                executor._fault(executor.runs["run"], "SYNTHETIC_CANCEL_DURING_SNAPSHOT")
                return value
            with mock.patch.object(t, 'snapshot', side_effect=late_observation):
                result = job.poll()
        self.assertEqual(result["code"], "SYNTHETIC_CANCEL_DURING_SNAPSHOT")
        self.assertEqual(len(sent), 2)
        self.assertTrue(t.owns_active_goal)
        self.assertEqual(t.poll_terminal_evidence()["result_status"], 4)
        self.assertFalse(t.owns_active_goal)

    def test_held_collision_sampling_covers_gripper_travel_and_feedback_bounds(self):
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity
        from sensor_msgs.msg import JointState
        _, executor, t, _, _, sent, _, _ = self.make_held_job()
        t._GetStateValidity, t._RobotState, t._JointState = GetStateValidity, RobotState, JointState
        requests = []
        def service(_kind, _endpoint, request, _code):
            requests.append(request)
            return SimpleNamespace(valid=True)
        t._service = service
        plan = executor.runs["run"]["plan"]
        report = t._check_plan_collision(plan, .021)
        self.assertTrue(report["all_valid"])
        self.assertTrue(all(request.group_name == '' for request in requests))
        values = [request.robot_state.joint_state.position[-1] for request in requests]
        self.assertIn(.01218, values)
        self.assertTrue(any(.01218 < value < .020 for value in values))
        self.assertEqual(sent, [])
        t._service = lambda *args: SimpleNamespace(valid=args[2].robot_state.joint_state.position[-1] != .01218)
        with self.assertRaisesRegex(ContractError, "COLLISION_DETECTED"):
            t._check_plan_collision(plan, .021)

    def held_phase_events(self, *, arm_target=.001, start_time=10., sequence_start=0):
        from tools.data_factory.quality.phase_events import validate_phase_event
        _, executor, _, _, _, _, _, _ = self.make_held_job(arm_target=arm_target)
        run = executor.runs["run"]
        run["execution"] = {"phase_event_sequence": sequence_start, "step_index": 0}
        events, rows, clock = [], [], [10.]
        def emit(record):
            events.append(validate_phase_event(record, plan=run["plan"]))
            return True
        executor._phase_event_writer = SimpleNamespace(emit=emit)
        executor.event_clock = lambda: (int(clock[0] * 1e9), "SYSTEM_TIME")
        for index, step in enumerate(run["plan"]["steps"][0]["held_target_segments"]):
            run["execution"]["learned_segment_index"] = index
            clock[0] = start_time + index * 2
            executor._emit_phase_event(run, "GOAL_ACCEPTED", step, "ACCEPTED", {"step": step, "accepted": True})
            rows.extend({"target_ros_s": clock[0] + (j + 1) / (index + 2)} for j in range(index + 1))
            clock[0] += 1.
            executor._emit_phase_event(run, "ACTION_TERMINAL", step, "SUCCEEDED", {"step": step, "terminal_status": "SUCCEEDED"})
        self.assertEqual(len(events), 6)
        return run["plan"], events, rows

    def test_multi_plan_phase_rows_preserve_both_chunks_in_existing_consumers(self):
        from tools.data_factory.quality.phase_events import read_phase_events, validate_phase_event_sequence
        from tools.data_factory.quality.phase_metrics import phase_row_windows, phase_timing_attribute
        from tools.data_factory.quality.execution_metrics import joint_execution_attribute
        from tools.data_factory.quality.interaction_metrics import interaction_quality_attribute
        from tools.data_factory.quality.episode_report import aggregate_episode_report
        first, a, ar = self.held_phase_events()
        second, b, br = self.held_phase_events(arm_target=.002, start_time=20., sequence_start=len(a))
        plans = {canonical_digest(p): p for p in (first, second)}
        events, rows = a + b, ar + br
        original = copy.deepcopy((plans, events, rows))
        for row in rows:
            p, start = (first, 10.) if row["target_ros_s"] < 20. else (second, 20.)
            segment = int((row["target_ros_s"] - start) // 2)
            target = p["steps"][0]["held_target_segments"][segment]["final_joint_state"]
            row.update({"observation.state": [*target, .012], "action": [*target, .012]})
        self.assertEqual(validate_phase_event_sequence(events, plans=plans), events)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_events.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            self.assertEqual(read_phase_events(path, plans=plans), events)
        windows, flags, _ = phase_row_windows(events=events, recorder_rows=rows, recorder_ros_clock_type="SYSTEM_TIME", plans=plans)
        self.assertEqual(flags, [])
        self.assertEqual([w["plan_digest"] for w in windows], [canonical_digest(first)] * 3 + [canonical_digest(second)] * 3)
        self.assertEqual([len(w["row_indices"]) for w in windows], [1, 2, 3, 1, 2, 3])
        self.assertEqual(sorted(i for w in windows for i in w["row_indices"]), list(range(12)))
        for p in (first, second):
            digest = canonical_digest(p)
            common = dict(run_id="run", resolved_job_digest=p["resolved_job_digest"], plan_digest=digest, plan=p,
                          plans=plans, events=events, recorder_rows=rows, recorder_rows_digest=canonical_digest(rows),
                          recorder_ros_clock_type="SYSTEM_TIME")
            timing = phase_timing_attribute(**common)
            self.assertEqual(timing["plan_digest"], digest)
            self.assertEqual(timing["status"], "AVAILABLE")
            self.assertEqual(timing["metrics"]["joined_row_count"], 6)
            self.assertEqual(timing["metrics"]["event_count"], 6)
            self.assertEqual(timing["metrics"]["source_event_count"], 12)
            self.assertEqual(timing["source_digests"]["pickup_plans"], canonical_digest(plans))
            joints = joint_execution_attribute(**common, stall_epsilon_rad=1e-4)
            self.assertEqual([item["segment_index"] for item in joints["metrics"]["phase_metrics"]], [0, 2])
            self.assertEqual([item["endpoint_joint_error_max_rad"] for item in joints["metrics"]["phase_metrics"]], [0., 0.])
            interaction = interaction_quality_attribute(**common, execution_evidence={"learned_execution": {"plan_digest": digest}})
            self.assertEqual(interaction["status"], "NOT_AVAILABLE")
            self.assertEqual(interaction["flags"], ["LEARNED_INTERACTION_UNQUALIFIED"])
            report = aggregate_episode_report([timing, joints, interaction], technical_validator={
                "schema_version": "data_factory.technical_validator_ref.v1", "status": "PASS",
                "result_digest": canonical_digest("synthetic-technical-result")})
            self.assertEqual(report["plan_digest"], digest)
            self.assertEqual(len(report["attributes"]), 3)
            self.assertIn("LEARNED_INTERACTION_UNQUALIFIED", report["flags"])
            with self.assertRaisesRegex(ContractError, "INTERACTION_QUALITY_BINDING"):
                interaction_quality_attribute(**common, execution_evidence={"learned_execution": {"plan_digest": canonical_digest("wrong")}})
        self.assertEqual((plans, events), original[:2])

    def test_multi_plan_event_lookup_rejects_missing_tampered_cross_run_and_replay(self):
        from tools.data_factory.quality.phase_events import validate_phase_event_sequence
        first, a, _ = self.held_phase_events()
        second, b, _ = self.held_phase_events(arm_target=.002, start_time=20., sequence_start=len(a))
        plans = {canonical_digest(p): p for p in (first, second)}
        events = a + b
        cases = [({canonical_digest(first): first}, "PHASE_EVENT_PLAN_BINDING"),
                 ({canonical_digest(first): second}, "PHASE_EVENT_PLANS_BINDING"),
                 ({}, "PHASE_EVENT_PLANS_BINDING")]
        other = {**second, "run_id": "other-run"}
        cases.append(({canonical_digest(first): first, canonical_digest(other): other}, "PHASE_EVENT_PLANS_BINDING"))
        for lookup, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(ContractError, code):
                validate_phase_event_sequence(events, plans=lookup)
        for malformed in ([], {}, None):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ContractError, "PHASE_EVENT_PLAN_BINDING"):
                validate_phase_event_sequence([{**a[0], "plan_digest": malformed}, *events[1:]], plans=plans)
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_SEGMENT_DUPLICATE"):
            validate_phase_event_sequence([*events, {**a[0], "sequence": 12}], plans=plans)
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_PLAN_BINDING"):
            validate_phase_event_sequence([{**a[0], "run_id": "other-run"}, *events[1:]], plans=plans)
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_SEGMENT_BINDING"):
            validate_phase_event_sequence([{**a[0], "evidence_digest": b[0]["evidence_digest"]}, *events[1:]], plans=plans)

    def test_multi_plan_missing_terminal_never_joins_across_plan_boundaries(self):
        from tools.data_factory.quality.phase_metrics import phase_row_windows, phase_intervals
        first, a, ar = self.held_phase_events()
        second, b, br = self.held_phase_events(arm_target=.002, start_time=20., sequence_start=len(a))
        plans = {canonical_digest(p): p for p in (first, second)}
        # Keep literal global sequence numbers. Another plan's terminal cannot
        # finish the first plan's accepted segment, even when phase/index match.
        events = [*a[:-1], *b]
        intervals, flags, _ = phase_intervals(events, plans=plans)
        self.assertIn("PHASE_TERMINAL_MISSING", flags)
        self.assertTrue(all(item["duration_s"] == 1. for item in intervals))
        windows, flags, _ = phase_row_windows(events=events, recorder_rows=ar+br,
                                             recorder_ros_clock_type="SYSTEM_TIME", plans=plans)
        self.assertEqual(windows, [])
        self.assertIn("RECORDER_ROWS_NOT_JOINED", flags)

    def test_held_phase_event_identity_requires_exact_plan_and_unique_segments(self):
        from tools.data_factory.quality.phase_events import validate_phase_event_sequence, validate_phase_event
        plan, events, _ = self.held_phase_events()
        self.assertEqual(validate_phase_event_sequence(events, plan=plan), events)
        self.assertEqual([(event["segment_index"], event["segment_count"]) for event in events[::2]], [(0, 3), (1, 3), (2, 3)])
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_PLAN_REQUIRED"):
            validate_phase_event_sequence(events)
        for changes, code in (({"segment_index": True}, "PHASE_EVENT_SEGMENT"),
                              ({"segment_count": 2.5}, "PHASE_EVENT_SEGMENT"),
                              ({"segment_count": 2}, "PHASE_EVENT_SEGMENT"),
                              ({"segment_index": 3}, "PHASE_EVENT_SEGMENT"),
                              ({"segment_index": 1}, "PHASE_EVENT_SEGMENT_BINDING"),
                              ({"plan_digest": canonical_digest("other")}, "PHASE_EVENT_PLAN_BINDING")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ContractError, code):
                validate_phase_event({**events[0], **changes}, plan=plan)
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_SEGMENT_DUPLICATE"):
            validate_phase_event_sequence([*events, {**events[-2], "sequence": 6}], plan=plan)
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_SEGMENT_ORDER"):
            validate_phase_event_sequence([*events[2:4], *events[:2]], plan=plan)

    def test_held_phase_rows_reach_existing_report_without_count_or_arm_aliasing(self):
        import tempfile
        from pathlib import Path
        from tools.data_factory.quality.episode_report import build_episode_report
        from tools.data_factory.quality.phase_events import PhaseEventWriter, read_phase_events
        from tools.data_factory.quality.phase_metrics import phase_timing_attribute
        plan, events, rows = self.held_phase_events()
        for row in rows:
            index = int((row["target_ros_s"] - 10.) // 2)
            step = plan["steps"][0]["held_target_segments"][index]
            row["action"] = [*step["final_joint_state"], step["gripper_position_m"]]
            row["observation.state"] = [*step["final_joint_state"], .021 if index == 0 else .01218]
        common = {"run_id": "run", "resolved_job_digest": plan["resolved_job_digest"],
                  "plan_digest": canonical_digest(plan), "plan": plan,
                  "recorder_rows": rows, "recorder_rows_digest": canonical_digest(rows),
                  "recorder_ros_clock_type": "SYSTEM_TIME"}
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "phase_events.jsonl"
            writer = PhaseEventWriter(sidecar, plan=plan)
            for event in events:
                self.assertTrue(writer.emit(event))
            self.assertTrue(writer.close())
            self.assertEqual(read_phase_events(sidecar, plan=plan), events)
            report = build_episode_report(Path(directory) / "episode_quality.json", **common,
                phase_events_path=sidecar, execution_evidence={}, stall_epsilon_rad=1e-4,
                technical_validator={"schema_version": "data_factory.technical_validator_ref.v1",
                                     "status": "PASS", "result_digest": canonical_digest("synthetic-only")})
        attributes = {item["attribute"]: item for item in report["attributes"]}
        timing = attributes["phase_timing_integrity"]
        self.assertEqual(timing["status"], "AVAILABLE")
        self.assertEqual([item["row_count"] for item in timing["metrics"]["phase_intervals"]], [1, 2, 3])
        self.assertEqual(timing["metrics"]["joined_row_count"], 6)
        joints = attributes["joint_execution_quality"]["metrics"]["phase_metrics"]
        self.assertEqual([item["segment_index"] for item in joints], [0, 2])
        self.assertEqual([item["row_count"] for item in joints], [1, 3])
        self.assertEqual([item["endpoint_joint_error_max_rad"] for item in joints], [0., 0.])
        interaction = attributes["interaction_quality"]
        self.assertEqual(interaction["status"], "NOT_AVAILABLE")
        self.assertEqual(interaction["flags"], ["LEARNED_INTERACTION_UNQUALIFIED"])
        self.assertIsNone(interaction["metrics"]["gripper_close"])
        self.assertIsNone(interaction["metrics"]["lift_continuity"])
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_SEGMENT_DUPLICATE"):
            phase_timing_attribute(**common, events=[*events, {**events[0], "sequence": 6}])
        with self.assertRaisesRegex(ContractError, "PHASE_EVENT_PLAN_REQUIRED"):
            phase_timing_attribute(**{k: v for k, v in common.items() if k != "plan"}, events=events)

    def test_post_inference_source_clock_freshness_and_reentrant_inference(self):
        now = [10.]
        def slow(_):
            now[0] += .4
            return [ACTION]
        inference = FinitePolicyInference(slow, CHECKPOINT, source_clock=lambda: now[0])
        with self.assertRaisesRegex(ContractError, "LEARNED_STALE_OBSERVATION"):
            inference.propose(observation(), **OPTIONS)
        inference = None
        def recursive(_):
            with self.assertRaisesRegex(ContractError, "LEARNED_REENTRANT_INFERENCE"):
                inference.propose(observation(), **OPTIONS)
            return [ACTION]
        inference = FinitePolicyInference(recursive, CHECKPOINT, source_clock=lambda: 10.)
        with self.assertRaisesRegex(ContractError, "LEARNED_CANCELLED"):
            inference.propose(observation(), **OPTIONS)

    def test_cancel_fences_late_result_from_another_thread(self):
        entered, release = threading.Event(), threading.Event()
        def policy(_):
            entered.set()
            self.assertTrue(release.wait(2))
            return [ACTION]
        inference = FinitePolicyInference(policy, CHECKPOINT, source_clock=lambda: 10.)
        failures = []
        def work():
            try:
                inference.propose(observation(), **OPTIONS)
            except ContractError as error:
                failures.append(error.code)
        thread = threading.Thread(target=work)
        thread.start()
        self.assertTrue(entered.wait(2))
        inference.cancel()
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, ["LEARNED_CANCELLED"])

    def test_exact_units_limits_source_clocks_and_horizon(self):
        mutations = [
            ("LEARNED_ACTION_CONTRACT", lambda p: p.update(units=["deg"] * 6 + ["m"])),
            ("LEARNED_ACTION_CONTRACT", lambda p: p.update(action_semantics="DELTA")),
            ("LEARNED_ACTION_7D", lambda p: p.update(actions=[[0.] * 6])),
            ("LEARNED_JOINT_LIMIT", lambda p: p.update(actions=[[0.] * 6 + [21.]])),
            ("LEARNED_VELOCITY_LIMIT", lambda p: p.update(actions=[[2.] * 6 + [.01]])),
            ("LEARNED_HORIZON", lambda p: p.update(actions=[ACTION] * 51)),
            ("LEARNED_HORIZON", lambda p: p.update(period_s=.001)),
            ("LEARNED_HORIZON", lambda p: p.update(period_s=6.)),
            ("LEARNED_STALE_OBSERVATION", lambda p: p["source_timestamps_s"].update(camera2=9.)),
            ("LEARNED_STALE_OBSERVATION", lambda p: p["source_timestamps_s"].update(state=11.)),
        ]
        for code, mutate in mutations:
            with self.subTest(code=code):
                p = proposal()
                mutate(p)
                with self.assertRaisesRegex(ContractError, code):
                    validate_proposal(redigest(p))

    def test_public_runtime_provenance_rejects_malformed_inputs_and_changed_clock_before_send(self):
        transport = Transport()
        transport.hardware = True
        inputs = {"checkpoint": "/synthetic/checkpoint", "gripper_source_clock": "/synthetic/clock.json", "device": "cpu",
                  "camera_topics": {"camera1": "/up", "camera2": "/wrist"},
                  "camera_mapping": {"observation.images.up": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"},
                  "fps": 10., "clock_binding": transport.snapshot()["gripper_controller"]["hardware_execution"]["clock_binding"]}
        p = proposal()
        p["runtime_inputs"] = inputs
        redigest(p)
        for field, invalid in (("device", []), ("fps", 30.), ("camera_topics", {}), ("camera_mapping", {})):
            with self.subTest(field=field):
                bad = copy.deepcopy(p)
                bad["runtime_inputs"][field] = invalid
                with self.assertRaisesRegex(ContractError, "LEARNED_RUNTIME_INPUTS"):
                    validate_proposal(redigest(bad))
        warmup = {"input_kind": "SYNTHETIC_ZERO_RGB_STATE", "image_shape": [1, 1, 3],
                  "instruction_digest": canonical_digest(p["instruction"]), "device": "cpu", "model_calls": 1,
                  "output_disposition": "DISCARDED", "rng_state_restored": True, "duration_s": 2., "inference_duration_s": 1.6}
        for field, invalid in (("output_disposition", "ADMITTED"), ("rng_state_restored", False),
                               ("model_calls", True), ("duration_s", -1.), ("instruction_digest", canonical_digest("another task"))):
            with self.subTest(warmup_field=field):
                bad = copy.deepcopy(p)
                bad["runtime_inputs"]["warmup"] = {**warmup, field: invalid}
                with self.assertRaisesRegex(ContractError, "LEARNED_WARMUP_INPUT"):
                    validate_proposal(redigest(bad))
        p["runtime_inputs"]["warmup"] = warmup
        redigest(p)
        calls = []
        executor = PickupExecutor(transport, execution_enabled=True, cell_state_store=Cell(), scene_state_store=Scene(),
                                  source_clock=lambda: 10., monotonic_clock=lambda: 10.)
        job = OneJob(Recorder(calls), executor.process)
        self.assertTrue(job.plan_only("run", compile_program(source(), p), SCENE)["ok"])
        self.assertTrue(job.approve(APPROVAL)["ok"])
        observe = transport.snapshot
        def rebound(*args):
            value = observe(*args)
            value["gripper_controller"]["hardware_execution"]["clock_binding"]["valid_until_system_s"] = 90.
            return value
        transport.snapshot = rebound
        self.assertEqual(job.start()["code"], "LEARNED_HARDWARE_CLOCK_BINDING")
        self.assertEqual(transport.sent, [])
        self.assertEqual(transport.cancel_count, 0)

    def make_job(self, *, hardware=True, scene_store=None, scene_binding=None, scoped=False, planning_delay=0.):
        calls = []
        transport, cell, scene = Transport(), Cell(), scene_store if scene_store is not None else Scene()
        now = [10.]
        transport.hardware = hardware
        transport.source_clock = lambda: now[0]
        executor = PickupExecutor(transport, execution_enabled=True, cell_state_store=cell,
                                  scene_state_store=scene, source_clock=lambda: now[0], monotonic_clock=lambda: now[0])
        def execute(request):
            calls.append(("executor", request["op"]))
            return executor.process(request)
        job = OneJob(Recorder(calls), execute)
        inference = FinitePolicyInference(lambda _: [ACTION[:]], CHECKPOINT, source_clock=lambda: now[0])
        snapshot = scene_store.snapshot() if scene_store is not None else None
        binding = ({"scene_state_digest": snapshot["scene_state_digest"],
                    "revision": snapshot["scene_state"]["revision"], "object_instance_id": "cube-1"}
                   if snapshot is not None else SCENE)
        if scene_binding is not None:
            binding = copy.deepcopy(scene_binding)
        grant = None
        if scoped:
            from tools.data_factory.rollout.finite_plan import generation_spec
            grant = task_grant(source(), binding, generation_spec(CHECKPOINT, **OPTIONS))
        original_plan = job.plan_only
        def delayed_plan(*args):
            now[0] += planning_delay
            return original_plan(*args)
        job.plan_only = delayed_plan
        planned = job.plan_learned("run", source(), binding, inference, observation(), task_grant=grant, **OPTIONS)
        self.assertTrue(planned["ok"], planned)
        return job, executor, transport, cell, scene, now, calls

    def test_scoped_generation_delay_is_consistent_before_and_after_plan_entry(self):
        for before, after in ((.8, 0.), (0., .8)):
            with self.subTest(before=before, after=after):
                job, executor, transport, _, _, now, calls = self.make_job(scoped=True, planning_delay=before)
                proposal = job._program["learned_proposal"]
                self.assertEqual(proposal["schema_version"], "data_factory.finite_learned_proposal.v2")
                self.assertEqual(proposal["source_timestamps_s"], observation()["source_timestamps_s"])
                self.assertEqual(proposal["actions"], [ACTION])
                self.assertEqual(transport.sent, [])
                self.assertFalse(any(target == "recorder" for target, _ in calls))
                original_deadline = executor.runs["run"]["task_deadline"]
                self.assertEqual(original_deadline, 100.)
                now[0] += after
                grant = task_grant(source(), SCENE, proposal)
                self.assertTrue(job.admit_task(grant)["ok"])
                self.assertEqual(executor.runs["run"]["task_deadline"], original_deadline)
                self.assertTrue(job.start()["ok"])
                self.assertEqual(len(transport.sent), 1)

    def test_scoped_generation_requires_grant_and_original_budget_at_admission(self):
        for failure in ("human", "grant", "expiry", "revoke", "cancel", "current_state", "hardware", "future"):
            with self.subTest(failure=failure):
                job, executor, transport, _, _, now, _ = self.make_job(scoped=True)
                grant = task_grant(source(), SCENE, job._program["learned_proposal"])
                if failure == "human":
                    result = job.approve(APPROVAL)
                else:
                    if failure == "grant":
                        grant["grant_id"] = "different"
                        grant.pop("grant_digest")
                        grant["grant_digest"] = canonical_digest(grant)
                    elif failure == "expiry":
                        executor.monotonic_clock = lambda: 101.
                    elif failure == "revoke":
                        executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "revoke-before-admit",
                            "op": "revoke_task", "payload": {"run_id": "run", "grant_digest": grant["grant_digest"]}})
                    elif failure == "cancel":
                        context = job._program["learned_proposal"]["generation_context"]
                        executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "cancel-before-admit",
                            "op": "cancel", "payload": {"run_id": "run", "generation_id": context["generation_id"]}})
                    elif failure == "future":
                        now[0] = 9.9
                    result = job.admit_task(grant)
                    if failure in {"current_state", "hardware"}:
                        self.assertTrue(result["ok"], result)
                        if failure == "current_state":
                            transport.current[0] += .1
                        else:
                            transport.hardware = False
                        result = job.start()
                self.assertFalse(result["ok"], result)
                self.assertEqual(transport.sent, [])

    def test_scoped_generation_rejects_archived_wrapper_and_changed_context(self):
        for mutation in ("wrapper", "run_id", "task_grant_digest", "scene_binding_digest", "predecessor_plan_digest", "future_completion"):
            with self.subTest(mutation=mutation):
                job, owner, _, _, _, _, _ = self.make_job(scoped=True)
                archived = copy.deepcopy(job._program)
                transport, cell, scene = Transport(), Cell(), Scene()
                transport.hardware = True
                fresh = PickupExecutor(transport, execution_enabled=True, cell_state_store=cell,
                    scene_state_store=scene, source_clock=lambda: 10., monotonic_clock=lambda: 10.)
                p = archived["learned_proposal"]
                grant = task_grant(source(), SCENE, p)
                from tools.data_factory.rollout.finite_plan import generation_spec
                opened = fresh.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "open",
                    "op": "begin_generation", "payload": {"run_id": "run", "grant": grant, "source_program": source(),
                    "scene_binding": SCENE, "proposal_spec": generation_spec(CHECKPOINT, **OPTIONS),
                    "predecessor_plan_digest": None, "lease_id": None}})
                self.assertTrue(opened["ok"], opened)
                if mutation != "wrapper":
                    # Even a rehashed proposal cannot substitute a different bound context.
                    p["generation_context"] = copy.deepcopy(opened["data"]["generation_context"])
                    if mutation == "future_completion":
                        p["inference_completed_at_s"] = 10.1
                    else:
                        p["generation_context"][mutation] = "wrong-run" if mutation == "run_id" else canonical_digest("wrong")
                    archived = compile_program(source(), redigest(p))
                response = fresh.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "plan",
                    "op": "plan", "payload": {"run_id": "run", "motion_program": archived, "scene_binding": SCENE}})
                self.assertFalse(response["ok"], response)
                self.assertEqual(transport.sent, [])

    def scoped_next_program(self, job, executor, now, *, policy=None):
        from tools.data_factory.rollout.finite_plan import generation_spec
        grant = executor.runs["run"]["task_grant"]
        response = executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "open-next",
            "op": "begin_generation", "payload": {"run_id": job.run_id, "grant": grant, "source_program": source(),
                "scene_binding": job.scene_binding, "proposal_spec": generation_spec(CHECKPOINT, **OPTIONS),
                "predecessor_plan_digest": job.plan_digest, "lease_id": job.lease_id}})
        self.assertTrue(response["ok"], response)
        obs = observation()
        obs["source_timestamps_s"] = {k: now[0] for k in obs["source_timestamps_s"]}
        obs["observation.state"] = ACTION[:]
        inference = FinitePolicyInference(policy or (lambda _: [ACTION[:]]), CHECKPOINT,
            source_clock=lambda: now[0], monotonic_clock=lambda: now[0])
        candidate = inference.propose(obs, generation_context=response["data"]["generation_context"], **OPTIONS)
        return compile_program(source(), candidate)

    def test_scoped_continuation_delay_on_either_side_of_compile_preserves_budget(self):
        for placement in ("before", "inside", "after"):
            with self.subTest(placement=placement):
                job, executor, transport, _, _, now, calls = self.make_job(scoped=True)
                self.assertTrue(job.admit_task(task_grant(source(), SCENE, job._program["learned_proposal"]))["ok"])
                self.assertTrue(job.start()["ok"])
                now[0] = 10.1
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                deadline = executor.runs["run"]["task_deadline"]
                candidate = self.scoped_next_program(job, executor, now)
                if placement == "before":
                    now[0] += .4
                elif placement == "inside":
                    original = executor._compile_plan
                    def delayed(*args, **kwargs):
                        result = original(*args, **kwargs)
                        now[0] += .4
                        return result
                    executor._compile_plan = delayed
                result = job.prepare_next_learned(candidate)
                self.assertTrue(result["ok"], result)
                if placement == "after":
                    now[0] += .4
                self.assertTrue(job.admit_task()["ok"])
                result = job.start_next_learned()
                self.assertTrue(result["ok"], result)
                self.assertEqual(len(transport.sent), 2)
                self.assertEqual(executor.runs["run"]["task_deadline"], deadline)
                self.assertEqual(calls.count(("recorder", "begin")), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_scoped_continuation_late_compile_and_invalid_inputs_send_no_new_goal(self):
        for failure in ("expired_compile", "cancelled_compile", "revoked_compile", "predecessor",
                        "current_state", "hardware", "stale_generation", "active"):
            with self.subTest(failure=failure):
                job, executor, transport, _, _, now, calls = self.make_job(scoped=True)
                grant = task_grant(source(), SCENE, job._program["learned_proposal"])
                self.assertTrue(job.admit_task(grant)["ok"])
                self.assertTrue(job.start()["ok"])
                now[0] = 10.1
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                if failure == "stale_generation":
                    def slow(_):
                        now[0] += .301
                        return [ACTION[:]]
                    with self.assertRaisesRegex(ContractError, "LEARNED_STALE_OBSERVATION|LEARNED_INFERENCE_TIMEOUT|LEARNED_HORIZON"):
                        self.scoped_next_program(job, executor, now, policy=slow)
                    self.assertEqual(len(transport.sent), 1)
                    continue
                candidate = self.scoped_next_program(job, executor, now)
                if failure == "predecessor":
                    candidate["learned_proposal"]["generation_context"]["predecessor_plan_digest"] = canonical_digest("foreign")
                    candidate = compile_program(source(), redigest(candidate["learned_proposal"]))
                elif failure in {"expired_compile", "cancelled_compile", "revoked_compile"}:
                    original = executor._compile_plan
                    def late(*args, **kwargs):
                        result = original(*args, **kwargs)
                        if failure == "expired_compile":
                            executor.monotonic_clock = lambda: 101.
                        elif failure == "cancelled_compile":
                            executor.runs["run"]["cancel_event"].set()
                        else:
                            executor.runs["run"]["task_revoked"] = True
                        return result
                    executor._compile_plan = late
                elif failure == "current_state":
                    transport.current[0] += .1
                elif failure == "hardware":
                    transport.hardware = False
                elif failure == "active":
                    transport.active = True
                result = job.prepare_next_learned(candidate)
                if failure == "hardware" and result["ok"]:
                    self.assertTrue(job.admit_task()["ok"])
                    result = job.start_next_learned()
                self.assertFalse(result["ok"], result["code"])
                self.assertEqual(len(transport.sent), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_scoped_monotonic_budget_is_not_reset_by_initial_compilation(self):
        # Wall time stalls while compilation consumes the original task budget.
        transport = Transport()
        transport.hardware = True
        steady = [10.]
        executor = PickupExecutor(transport, execution_enabled=True, cell_state_store=Cell(),
            scene_state_store=Scene(), source_clock=lambda: 10., monotonic_clock=lambda: steady[0])
        job = OneJob(Recorder([]), executor.process)
        from tools.data_factory.rollout.finite_plan import generation_spec
        grant = task_grant(source(), SCENE, generation_spec(CHECKPOINT, **OPTIONS))
        original = executor._compile_plan
        def late(*args, **kwargs):
            result = original(*args, **kwargs)
            steady[0] = 101.
            return result
        executor._compile_plan = late
        result = job.plan_learned("run", source(), SCENE,
            FinitePolicyInference(lambda _: [ACTION[:]], CHECKPOINT, source_clock=lambda: 10.),
            observation(), task_grant=grant, **OPTIONS)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["code"], "TASK_DEADLINE_EXHAUSTED")
        self.assertEqual(transport.sent, [])
        self.assertEqual(executor.runs, {})

    def test_scoped_generation_keeps_real_scene_guard_at_initial_and_next_dispatch(self):
        from tools.data_factory.scene_state import SceneStateStore
        for continued in (False, True):
            with self.subTest(continued=continued), tempfile.TemporaryDirectory() as directory:
                scene = SceneStateStore(directory, "fr5-lab-a")
                def human_pose(x):
                    scene.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                        source="HUMAN", updated_by="operator",
                        pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": x, "y_mm": 20.})
                human_pose(10.)
                job, executor, transport, _, _, now, _ = self.make_job(scoped=True, scene_store=scene)
                self.assertTrue(job.admit_task(task_grant(source(), job.scene_binding, job._program["learned_proposal"]))["ok"])
                if continued:
                    result = job.start()
                    self.assertTrue(result["ok"], result["code"])
                    now[0] = 10.1
                    self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                    candidate = self.scoped_next_program(job, executor, now)
                    self.assertTrue(job.prepare_next_learned(candidate)["ok"])
                    self.assertTrue(job.admit_task()["ok"])
                sent = len(transport.sent)
                human_pose(99.)
                changed = scene.snapshot()
                result = job.start_next_learned() if continued else job.start()
                self.assertFalse(result["ok"], result["code"])
                self.assertEqual(len(transport.sent), sent)
                self.assertEqual(scene.snapshot(), changed)

    def test_task_grant_scene_arming_release_revalidates_before_submission(self):
        from tools.data_factory.scene_state import SceneStateStore
        for scoped in (False, True):
            for mutate in (False, True):
                with self.subTest(scoped=scoped, mutate=mutate), tempfile.TemporaryDirectory() as directory:
                    scene = SceneStateStore(directory, "fr5-lab-a")
                    def human_pose(x):
                        scene.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                            source="HUMAN", updated_by="operator",
                            pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": x, "y_mm": 20.})
                    human_pose(10.)
                    job, executor, transport, _, _, _, _ = self.make_job(scoped=scoped, scene_store=scene)
                    self.assertTrue(job.admit_task(task_grant(source(), job.scene_binding, job._program["learned_proposal"]))["ok"])
                    locked = scene.locked_snapshot
                    armed_release = []
                    @contextmanager
                    def interleaved(*args, **kwargs):
                        with locked(*args, **kwargs) as value:
                            yield value
                        if not armed_release and "execution" in executor.runs["run"]:
                            armed_release.append(True)
                            if mutate:
                                human_pose(99.)
                    scene.locked_snapshot = interleaved
                    result = job.start()
                    self.assertEqual(result["ok"], not mutate, result["code"])
                    self.assertEqual(len(transport.sent), 0 if mutate else 1)
                    self.assertEqual(scene.snapshot()["scene_state"]["objects"]["cube-1"]["pose"]["x_mm"], 99. if mutate else 10.)

    def test_generation_retry_and_cancel_never_allocate_a_new_context_or_budget(self):
        from tools.data_factory.rollout.finite_plan import generation_spec
        transport, now = Transport(), [10.]
        executor = PickupExecutor(transport, source_clock=lambda: now[0], monotonic_clock=lambda: now[0])
        spec = generation_spec(CHECKPOINT, **OPTIONS)
        request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "original-generation",
            "op": "begin_generation", "payload": {"run_id": "run", "grant": task_grant(source(), SCENE, spec),
                "source_program": source(), "scene_binding": SCENE, "proposal_spec": spec,
                "predecessor_plan_digest": None, "lease_id": None}}
        first = executor.process(copy.deepcopy(request))
        self.assertTrue(first["ok"], first["code"])
        context = first["data"]["generation_context"]
        now[0] += 1.
        self.assertEqual(executor.process(copy.deepcopy(request)), first)
        stopped = executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "stop-generation",
            "op": "cancel", "payload": {"run_id": "run", "generation_id": context["generation_id"]}})
        self.assertTrue(stopped["ok"])
        # The idempotent receipt is historical; it cannot undo the owner's fence.
        self.assertEqual(executor.process(copy.deepcopy(request)), first)
        self.assertEqual(executor._generation["context"], context)
        self.assertTrue(executor._generation["cancelled"])
        request["op_id"] = "new-generation"
        self.assertEqual(executor.process(request)["code"], "LEARNED_GENERATION_REUSED")
        p = FinitePolicyInference(lambda _: [ACTION[:]], CHECKPOINT, source_clock=lambda: 10.).propose(
            observation(), generation_context=context, **OPTIONS)
        result = executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "late-plan",
            "op": "plan", "payload": {"run_id": "run", "motion_program": compile_program(source(), p), "scene_binding": SCENE}})
        self.assertEqual(result["code"], "LEARNED_CANCELLED")
        self.assertEqual(transport.sent, [])

    def test_scoped_one_job_preserves_policy_failure_when_cancel_is_unconfirmed(self):
        from tools.data_factory.rollout.finite_plan import generation_spec
        for broken_pipe in (False, True):
            with self.subTest(broken_pipe=broken_pipe):
                transport, calls = Transport(), []
                executor = PickupExecutor(transport, source_clock=lambda: 10., monotonic_clock=lambda: 10.)
                def request(value):
                    calls.append(value["op"])
                    if value["op"] == "cancel":
                        if broken_pipe:
                            raise BrokenPipeError("synthetic closed transport")
                        value = copy.deepcopy(value)
                        value["payload"]["generation_id"] = "wrong-generation"
                    return executor.process(value)
                recorder = mock.Mock(side_effect=AssertionError("recorder must not start"))
                job = OneJob(recorder, request)
                def failed_policy(_):
                    raise RuntimeError("original inference failure")
                inference = FinitePolicyInference(failed_policy, CHECKPOINT, source_clock=lambda: 10.)
                grant = task_grant(source(), SCENE, generation_spec(CHECKPOINT, **OPTIONS))
                result = job.plan_learned("run", source(), SCENE, inference, observation(), task_grant=grant, **OPTIONS)
                self.assertEqual(result["code"], "LEARNED_POLICY_FAILED")
                self.assertEqual(result["cancel_error"], "EXECUTOR_CALL_FAILED" if broken_pipe else "LEARNED_GENERATION_CONTEXT")
                self.assertEqual(result["state"], "BLOCKED")
                self.assertEqual(calls, ["begin_generation", "cancel"])
                self.assertEqual(transport.sent, [])
                recorder.assert_not_called()

    def start_job(self):
        values = self.make_job()
        job = values[0]
        self.assertTrue(job.approve(APPROVAL)["ok"])
        self.assertTrue(job.start()["ok"])
        self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
        return values

    def test_scene_transition_retains_real_store_snapshot_for_terminal_and_failure(self):
        from tools.data_factory.scene_state import SceneStateStore
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                store = SceneStateStore(directory, "fr5-lab-a")
                store.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                    source="HUMAN", updated_by="synthetic-operator",
                    pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": 10., "y_mm": 20.})
                job, executor, transport, cell, _, now, calls = self.make_job(scene_store=store)
                initial_binding = copy.deepcopy(job.scene_binding)
                self.assertTrue(job.approve(APPROVAL)["ok"])
                self.assertTrue(job.start()["ok"])
                self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
                self.assertTrue(job.confirm("operator")["ok"])
                if failure:
                    transport.failure = "SYNTHETIC_TRANSPORT_FAILURE"
                    result = job.poll()
                    self.assertEqual(result["code"], "SYNTHETIC_TRANSPORT_FAILURE")
                else:
                    self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                    self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
                    result = job.poll()
                    self.assertEqual(result["code"], "PRECOMMIT_SAFETY")
                historical = result["execution_evidence"]["scene_transition"]
                self.assertEqual(historical, store.snapshot())
                self.assertEqual(historical["scene_state_digest"], canonical_digest(historical["scene_state"]))
                self.assertEqual(historical["scene_state"]["revision"], initial_binding["revision"] + 1)
                item = historical["scene_state"]["objects"]["cube-1"]
                self.assertEqual((item["state"], item["pose"], item["source"]), ("UNKNOWN", None, "ROBOT_ACTION"))
                self.assertEqual(job.scene_binding, initial_binding)
                self.assertFalse(cell.ready)
                self.assertNotIn(("recorder", "commit"), calls)
                # The existing JSON response owns a historical snapshot, not a
                # live view onto whichever scene becomes current later.
                serialized = json.dumps(result, sort_keys=True)
                later = store.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                    source="HUMAN", updated_by="synthetic-operator", expected_revision=historical["scene_state"]["revision"],
                    pose={"place_id": "PLACE_A", "yaw_deg": 30., "x_mm": 40., "y_mm": 50.})
                self.assertNotEqual(later["scene_state_digest"], historical["scene_state_digest"])
                self.assertEqual(json.dumps(result, sort_keys=True), serialized)
                self.assertEqual(executor._execution_data(executor.runs["run"])["scene_transition"], historical)
                self.assertEqual(learned_run_diagnostic(result)["task_effectiveness"], "UNKNOWN")

    def test_zero_dispatch_cancel_preserves_real_scene_bytes_and_foreign_revision(self):
        from tools.data_factory.scene_state import SceneStateStore
        for foreign in (False, True):
            with self.subTest(foreign=foreign), tempfile.TemporaryDirectory() as directory:
                store = SceneStateStore(directory, "fr5-lab-a")
                store.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                    source="HUMAN", updated_by="synthetic-operator",
                    pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": 10., "y_mm": 20.})
                job, executor, transport, cell, _, _, calls = self.make_job(scene_store=store)
                self.assertTrue(job.approve(APPROVAL)["ok"])
                self.assertTrue(job.start()["ok"])
                self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
                self.assertIs(executor.runs["run"]["execution"]["motion_dispatch_attempted"], False)
                if foreign:
                    store.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                        source="HUMAN", updated_by="another-operator", expected_revision=store.read()["revision"],
                        pose={"place_id": "PLACE_A", "yaw_deg": 30., "x_mm": 40., "y_mm": 50.})
                before = store._path().read_bytes()
                job.cancel()
                self.assertEqual(store._path().read_bytes(), before)
                self.assertEqual(transport.sent, [])
                self.assertEqual(transport.cancel_count, 0)
                self.assertFalse(cell.ready)
                # The zero-dispatch fault must not be followed by a late send.
                with self.assertRaisesRegex(ContractError, "ROS_EXEC_CANCELLED"):
                    executor._start_phase(executor.runs["run"], {})
                self.assertEqual(transport.sent, [])
                self.assertIs(executor.runs["run"]["execution"]["motion_dispatch_attempted"], False)
                data = executor._execution_data(executor.runs["run"])
                self.assertEqual(data["scene_preservation"]["reason_code"], "NO_DISPATCH")
                self.assertEqual(data["scene_preservation"]["status"], "NOT_UPDATED")
                self.assertNotIn("scene_transition", data)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_transport_exception_without_ack_is_not_zero_dispatch_evidence(self):
        for record_send in (False, True):
            with self.subTest(record_send=record_send):
                job, executor, transport, _, scene, _, _ = self.start_job()
                def failed_start(step, **options):
                    if record_send:
                        transport.sent.append(copy.deepcopy(step))
                    raise RuntimeError("send/response ambiguity")
                transport.start_phase = failed_start
                self.assertFalse(job.confirm("operator")["ok"])
                self.assertIs(executor.runs["run"]["execution"]["motion_dispatch_attempted"], True)
                self.assertEqual(len(transport.sent), int(record_send))
                self.assertEqual([item["state"] for item in scene.updates], ["UNKNOWN"])
                self.assertNotIn("scene_preservation", executor._execution_data(executor.runs["run"]))

    def test_missing_dispatch_history_does_not_preserve_scene(self):
        job, executor, transport, _, scene, _, _ = self.start_job()
        executor.runs["run"]["execution"].pop("motion_dispatch_attempted")
        job.cancel()
        self.assertEqual(transport.sent, [])
        self.assertEqual([item["state"] for item in scene.updates], ["UNKNOWN"])

    def test_next_output_preserves_prior_dispatch_on_presend_failure(self):
        job, executor, transport, _, scene, _, _ = self.start_job()
        self.assertTrue(job.confirm("operator")["ok"])
        self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertTrue(job.prepare_next_learned(self.next_raw_program(job))["ok"])
        self.assertTrue(job.approve_next_learned({**APPROVAL, "approval_id": "next"})["ok"])
        def fail_before_next_send(run):
            self.assertIs(run["execution"]["motion_dispatch_attempted"], True)
            executor._fault(run, "SYNTHETIC_PRESEND_FAILURE")
            return run["state"]
        with mock.patch.object(executor, "_start_current_step", side_effect=fail_before_next_send):
            self.assertFalse(job.start_next_learned()["ok"])
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual([item["state"] for item in scene.updates], ["UNKNOWN"])
        self.assertNotIn("scene_preservation", executor._execution_data(executor.runs["run"]))

    def test_scene_transition_conflict_or_write_failure_cannot_publish_a_snapshot(self):
        from tools.data_factory.scene_state import SceneStateStore
        for conflict in (True, False):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as directory:
                store = SceneStateStore(directory, "fr5-lab-a")
                store.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                    source="HUMAN", updated_by="synthetic-operator",
                    pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": 10., "y_mm": 20.})
                job, executor, transport, _, _, _, calls = self.make_job(scene_store=store)
                self.assertTrue(job.approve(APPROVAL)["ok"])
                self.assertTrue(job.start()["ok"])
                self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
                self.assertTrue(job.confirm("operator")["ok"])
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                if conflict:
                    store.update_object(instance_id="cube-1", object_profile_id="cube", state="UNKNOWN",
                        source="HUMAN", updated_by="other-synthetic-operator", expected_revision=1)
                else:
                    store.update_object = mock.Mock(side_effect=OSError("synthetic write failure"))
                before = store._path().read_bytes()
                rejected = job.semantic_verdict("PASS", "operator")
                self.assertFalse(rejected["ok"])
                self.assertEqual(rejected["code"], "LEARNED_SCENE_UNCERTAIN")
                self.assertNotIn("scene_transition", rejected["execution_evidence"])
                self.assertEqual(rejected["execution_evidence"]["scene_state_error"],
                                 "SCENE_REVISION_CONFLICT" if conflict else "SCENE_STATE_WRITE_FAILED")
                self.assertEqual(store._path().read_bytes(), before)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_native_program_canonical_validation_and_plan_only_zero_effects(self):
        src = source()
        original = copy.deepcopy(src)
        p = proposal()
        program = compile_program(src, p)
        self.assertEqual(validate_motion_program(program), program)
        legacy = copy.deepcopy(program)
        legacy["steps"][0]["pause_after"] = "SEMANTIC_VERDICT"
        self.assertEqual(validate_motion_program(legacy), legacy)
        self.assertEqual(src, original)
        self.assertEqual([s["phase"] for s in program["steps"]], ["LEARNED_CHUNK"])
        program["steps"].append(src["steps"][-1])
        with self.assertRaisesRegex(ContractError, "LEARNED_PROGRAM_BINDING"):
            validate_motion_program(program)
        job, _, transport, cell, scene, _, calls = self.make_job()
        self.assertEqual(transport.sent, [])
        self.assertFalse(any(target == "recorder" for target, _ in calls))
        self.assertTrue(cell.ready)
        self.assertEqual(scene.updates, [])
        self.assertEqual(_operator_summary(job._result())["path"], ["LEARNED_CHUNK"])

    def test_cross_workspace_native_source_retains_both_endpoints(self):
        src = cross_workspace_source()
        original = copy.deepcopy(src)
        program = compile_program(src, proposal())
        self.assertEqual(validate_motion_program(program), program)
        self.assertEqual(program["source_program"], original)
        for key in ("destination_resolved_job_digest", "destination_binding_digests",
                    "endpoint_bindings", "endpoint_bindings_digest"):
            self.assertEqual(program[key], original[key])
        self.assertEqual(src, original)
        # Exercise the normal OneJob -> sole executor plan-only consumer too.
        with mock.patch(__name__ + ".source", return_value=src):
            job, _, transport, cell, scene, _, calls = self.make_job()
        self.assertEqual(job.plan_envelope["plan"]["learned_source_program"], original)
        self.assertEqual(transport.sent, [])
        self.assertEqual(scene.updates, [])
        self.assertTrue(cell.ready)
        self.assertFalse(any(target == "recorder" for target, _ in calls))

    def test_cross_workspace_native_source_rejects_invalid_endpoint_bindings(self):
        for corruption in ("missing", "reordered", "recipe", "incompatible"):
            with self.subTest(corruption=corruption):
                src = cross_workspace_source()
                if corruption == "missing":
                    src.pop("destination_binding_digests")
                elif corruption == "reordered":
                    src["endpoint_bindings"].reverse()
                elif corruption == "recipe":
                    src["endpoint_bindings"][1]["motion_recipe_digest"] = canonical_digest("other")
                else:
                    src["destination_binding_digests"]["robot_system"] = canonical_digest("other-robot")
                src["endpoint_bindings_digest"] = canonical_digest(src["endpoint_bindings"])
                with self.assertRaises(ContractError):
                    compile_program(src, proposal())

    @staticmethod
    def slotted_scene(directory):
        from tools.data_factory.scene_state import SceneStateStore, release_slot
        store = SceneStateStore(directory, "fr5-lab-a")
        pose = {"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": 10., "y_mm": 20.}
        start = store.update_object(instance_id="cube-1", object_profile_id="cube",
            state="ON_SURFACE", pose=pose, source="HUMAN", updated_by="fixture")
        slot = release_slot(robot_system_id="fr5-lab-a", pose=pose,
            object_profile_id="cube", exclusion_geometry_digest=canonical_digest("box"),
            role="DESTINATION_THEN_NEXT_SOURCE")
        evidence = {"schema_version": "data_factory.recycle_release_evidence.v1",
            "run_id": "previous", "plan_digest": canonical_digest("previous-plan"),
            "release_slot_id": slot["slot_id"], "expected_scene_state_digest": start["scene_state_digest"],
            "expected_scene_revision": start["scene_state"]["revision"],
            "gripper_reference_m": .01, "gripper_feedback_m": .01,
            "terminal_phases": ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"],
            "post_retreat_snapshot_digest": canonical_digest("snapshot"),
            "next_start_tolerance_rad": .01, "human_verdict": "LANDED"}
        landed = store.transition_release(instance_id="cube-1", release_slot=slot, evidence=evidence,
            updated_by="fixture", expected_digest=start["scene_state_digest"],
            expected_revision=start["scene_state"]["revision"], allowed_next_run_id="run")
        destination = release_slot(robot_system_id="fr5-lab-a", pose={**pose, "place_id": "PLACE_B"},
            object_profile_id="cube", exclusion_geometry_digest=slot["exclusion_geometry_digest"])
        binding = {"scene_state_digest": landed["scene_state_digest"],
            "revision": landed["scene_state"]["revision"], "object_instance_id": "cube-1",
            "release_slot": destination, "source_slot": {"slot_id": slot["slot_id"],
                "slot_digest": canonical_digest(landed["scene_state"]["slot_allocations"][slot["slot_id"]]),
                "allowed_run_id": "run"}}
        return store, binding

    def test_slotted_learned_plan_preserves_scope_without_recycle_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            store, binding = self.slotted_scene(directory)
            before = store._path().read_bytes()
            with mock.patch(__name__ + ".source", return_value=cross_workspace_source()):
                job, executor, transport, cell, _, _, calls = self.make_job(scene_store=store, scene_binding=binding)
            self.assertEqual(job.plan_envelope["plan"]["scene_binding"], binding)
            self.assertNotIn("recycle", job.plan_envelope["operator_summary"])
            self.assertIsNone(executor.runs["run"]["recycle_plan_digest"])
            self.assertEqual(store._path().read_bytes(), before)
            self.assertEqual(transport.sent, [])
            self.assertTrue(cell.ready)
            self.assertFalse(any(target == "recorder" for target, _ in calls))
            original = copy.deepcopy(job.plan_envelope["plan"])
            binding["release_slot"]["pose"]["x_mm"] += 1
            self.assertEqual(job.plan_envelope["plan"], original)

    def test_slotted_learned_completion_and_fault_use_consumed_revision(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                store, binding = self.slotted_scene(directory)
                job, executor, transport, _, _, _, _ = self.make_job(scene_store=store, scene_binding=binding)
                self.assertTrue(job.approve(APPROVAL)["ok"])
                self.assertTrue(job.start()["ok"])
                consumed = store.read()
                self.assertEqual(consumed["revision"], binding["revision"] + 1)
                self.assertEqual(consumed["slot_allocations"][binding["source_slot"]["slot_id"]]["state"], "CONSUMED_PENDING_REVIEW")
                self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
                self.assertTrue(job.confirm("operator")["ok"])
                if fail:
                    transport.failure = "SYNTHETIC_FAILURE"
                    result = job.poll()
                else:
                    self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                    self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
                    result = job.poll()
                actual = store.read()
                self.assertEqual(actual["revision"], consumed["revision"] + 1)
                self.assertEqual(actual["objects"]["cube-1"]["state"], "UNKNOWN")
                self.assertEqual(actual["slot_allocations"], consumed["slot_allocations"])
                self.assertNotIn("release_evidence", executor.runs["run"]["execution"])
                self.assertNotIn("scene_state_error", result["execution_evidence"])
                self.assertEqual(learned_run_diagnostic(result)["task_effectiveness"], "UNKNOWN")

    def test_slotted_learned_source_conflicts_send_no_goal(self):
        for corruption in ("run", "slot", "scene", "consumed"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                store, binding = self.slotted_scene(directory)
                if corruption == "run":
                    binding["source_slot"]["allowed_run_id"] = "other"
                elif corruption == "slot":
                    binding["source_slot"]["slot_digest"] = canonical_digest("other")
                elif corruption == "scene":
                    binding["scene_state_digest"] = canonical_digest("other")
                job, _, transport, _, _, _, _ = self.make_job(scene_store=store, scene_binding=binding)
                if corruption == "consumed":
                    store.consume_next_source(slot_id=binding["source_slot"]["slot_id"], run_id="run",
                        expected_scene_digest=binding["scene_state_digest"], expected_slot_digest=binding["source_slot"]["slot_digest"])
                before = store._path().read_bytes()
                self.assertTrue(job.approve(APPROVAL)["ok"])
                result = job.start()
                self.assertFalse(result["ok"], result)
                self.assertIn(result["code"], {"SCENE_SLOT_NEXT_RUN", "SCENE_SLOT_CHANGED", "SCENE_STATE_CHANGED"})
                self.assertEqual(transport.sent, [])
                self.assertEqual(store._path().read_bytes(), before)

    def test_slotted_learned_plan_rejects_malformed_or_foreign_slots(self):
        from tools.data_factory.scene_state import release_slot
        for corruption in ("unknown", "digest", "robot"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                store, binding = self.slotted_scene(directory)
                if corruption == "unknown":
                    binding["source_slot"]["unrecognized"] = True
                elif corruption == "digest":
                    binding["release_slot"]["slot_id"] = canonical_digest("wrong")
                else:
                    slot = binding["release_slot"]
                    binding["release_slot"] = release_slot(robot_system_id="other-robot",
                        pose=slot["pose"], object_profile_id=slot["object_profile_id"],
                        exclusion_geometry_digest=slot["exclusion_geometry_digest"])
                transport = Transport()
                executor = PickupExecutor(transport, source_clock=lambda: 10.)
                before = store._path().read_bytes()
                with self.assertRaises(ContractError):
                    executor._compile_plan({"run_id": "run", "motion_program": compile_program(source(), proposal()),
                                            "scene_binding": binding})
                self.assertEqual(transport.sent, [])
                self.assertEqual(transport.calls, [])
                self.assertEqual(store._path().read_bytes(), before)

    def test_native_source_still_rejects_unknown_or_nested_program_versions(self):
        for schema in ("fr5.motion_program.v3", "fr5.motion_program.v999", "fr5.learned_motion_program.v1"):
            with self.subTest(schema=schema):
                src = source()
                src["schema_version"] = schema
                with self.assertRaisesRegex(ContractError, "LEARNED_SOURCE_PROGRAM"):
                    compile_program(src, proposal())

    def test_chunk_boundary_keeps_one_recorder_and_bound_fresh_observation(self):
        job, executor, transport, cell, scene, now, calls = self.start_job()
        self.assertTrue(job.confirm("operator")["ok"])
        boundary = job.poll()
        self.assertEqual(boundary["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(job.recorder_state, "RECORDING")
        self.assertIsNone(job.semantic)
        frozen_plan = copy.deepcopy(job.plan_envelope)
        tx, lease = job.transaction_id, job.lease_id
        def capture(topics, age):
            self.assertEqual(topics, {"camera1": "/up", "camera2": "/wrist"})
            # Exercise the actual native capture method and ROS serializers;
            # only subscription delivery is synthetic, without a ROS node.
            from sensor_msgs.msg import JointState, Image
            from rclpy.serialization import serialize_message, deserialize_message
            from tools.data_factory.motion.moveit_transport import RosMoveItTransport
            joint = JointState(name=JOINTS, position=ACTION)
            joint.header.stamp.sec = 10
            image = Image(height=1, width=1, encoding="rgb8", step=3, data=bytes([1, 2, 3]))
            image.header.stamp.sec = 10
            joint = deserialize_message(serialize_message(joint), JointState)
            image = deserialize_message(serialize_message(image), Image)
            native = object.__new__(RosMoveItTransport)
            native._active, native._execution_locked = None, False
            native._clock, native.graph_timeout_s = lambda: 10., .01
            native._joint_state = native._joint_state_received_at = None
            callbacks = []
            native.node = SimpleNamespace(
                get_parameter=lambda _: SimpleNamespace(value=False),
                create_subscription=lambda _type, _topic, callback, _qos: callbacks.append(callback),
                destroy_subscription=lambda _: None)
            def spin(*_, **__):
                native._joint_state, native._joint_state_received_at = joint, 10.
                for callback in callbacks:
                    callback(image)
            native._rclpy = SimpleNamespace(spin_once=spin)
            with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=10.):
                return native.capture_policy_observation(topics, age)
        transport.capture_policy_observation = capture
        original_poll = job.poll
        def poll():
            if ("executor", "capture_observation") in calls:
                self.assertNotIn("observation", job.execution_evidence)
                self.assertNotIn("observation", job.execution_response["data"])
            return original_poll()
        job.poll = poll
        result = job.observe_learned_boundary({"camera1": "/up", "camera2": "/wrist"})
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["observation"]["observation.state"], ACTION)
        self.assertEqual(result["observation"]["observation.images.camera1"]["data_hex"], "010203")
        self.assertEqual(result["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual((job.transaction_id, job.lease_id), (tx, lease))
        self.assertEqual(job.plan_envelope, frozen_plan)
        self.assertEqual(calls.count(("recorder", "begin")), 1)
        self.assertNotIn(("recorder", "freeze"), calls)
        self.assertNotIn(("recorder", "commit"), calls)
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(scene.updates, [])
        self.assertFalse(cell.ready)
        # Observation is no approval to replace a plan or replay its goal.
        self.assertEqual(job.start()["code"], "START_STATE")
        self.assertEqual(job.plan_only("run", source(), SCENE)["code"], "ONE_JOB_ONLY")
        self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
        self.assertEqual(calls.count(("recorder", "freeze")), 1)
        self.assertEqual(job.poll()["code"], "PRECOMMIT_SAFETY")
        self.assertNotIn(("recorder", "commit"), calls)

    def next_raw_program(self, job, now=10.):
        obs = observation()
        obs["observation.state"] = ACTION[:]
        obs["source_timestamps_s"] = dict.fromkeys(("state", "camera1", "camera2"), now)
        inference = FinitePolicyInference(lambda _: [ACTION[:]], CHECKPOINT, source_clock=lambda: now)
        return compile_program(source(), inference.propose(obs, **OPTIONS))

    def test_task_grant_rechecks_native_source_bytes(self):
        from tools.data_factory.rollout.task_authority import check_runtime_source
        from tools.data_factory.learned_action_adapter import NativeSmolVLA
        from tools.data_factory.training_receipts import tree_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "checkpoint" / "pretrained_model"
            policy.mkdir(parents=True)
            processor = policy / "processor.json"
            processor.write_text('{"scale": 1}')
            temporal = root / "temporal.json"
            temporal.write_text('{"bound": true}')
            p = proposal()
            p["checkpoint"] = {**CHECKPOINT, "runtime": "lerobot-0.6.1-native", "tree_digest": tree_digest(policy)}
            p["runtime_inputs"] = {"checkpoint": str(policy.parent), "gripper_temporal_policy": str(temporal),
                                   "clock_binding": {"bound": True}}
            native = NativeSmolVLA.__new__(NativeSmolVLA)
            native._inference_lock = threading.Lock()
            native.policy_dir, native.checkpoint = policy, p["checkpoint"]
            with native.prepare_inference():
                check_runtime_source(p["runtime_inputs"])
            processor.write_text('{"scale": 2}')
            with self.assertRaisesRegex(ContractError, "LEARNED_CHECKPOINT_CHANGED"):
                with native.prepare_inference():
                    self.fail("changed checkpoint acquired inference authority")
            processor.write_text('{"scale": 1}')
            temporal.write_text('{"bound": false}')
            with self.assertRaisesRegex(ContractError, "TASK_SOURCE_CHANGED"):
                with native.prepare_inference():
                    check_runtime_source(p["runtime_inputs"])

    def test_task_dispatch_rechecks_current_state_after_snapshot_and_native_decode(self):
        for failure in ("old_header", "slow_decode"):
            with self.subTest(failure=failure):
                job, executor, transport, _, now, sent, _, _ = self.make_held_job(max_observation_age_s=.3)
                grant = task_grant(job._program["source_program"], job.scene_binding, job._program["learned_proposal"])
                self.assertTrue(job.admit_task(grant)["ok"])
                now[0] += 2.  # Frozen input timestamps stay at 10; current state is separate.
                snapshot, decode = transport.snapshot, transport._compiled_execution_goal
                def old_header(*args):
                    value = snapshot(*args)
                    value["joint_state_stamp_ns"] = 10_000_000_000
                    return value
                def slow_decode(step):
                    value = decode(step)
                    now[0] += 1.1
                    return value
                if failure == "old_header":
                    transport.snapshot = old_header
                else:
                    transport._compiled_execution_goal = slow_decode
                with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
                    result = job.start()
                self.assertEqual(result["code"], "LEARNED_STALE_STATE")
                self.assertEqual(sent, [])
                self.assertFalse(transport.owns_active_goal)

    def test_task_plan_uses_current_state_after_recorder_preparation_and_across_segments(self):
        from tools.data_factory.one_job import RECORDER_READINESS_CONTRACT
        job, executor, _, state, now, sent, _, calls = self.make_held_job(max_observation_age_s=.3)
        plan = copy.deepcopy(executor.runs["run"]["plan"])
        grant = task_grant(job._program["source_program"], job.scene_binding, plan["learned_proposal"])
        self.assertTrue(job.admit_task(grant)["ok"])
        recorder_call = job.recorder_call
        def prepare_recorder(request):
            response = recorder_call(request)
            if request["op"] == "begin":
                # The native runner collects this prefix after plan admission.
                now[0] += (RECORDER_READINESS_CONTRACT["min_durable_rows"]
                           / RECORDER_READINESS_CONTRACT["target_fps"])
            return response
        job.recorder_call = prepare_recorder
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time', side_effect=lambda: now[0]):
            self.assertTrue(job.start()["ok"])
            deadline = executor.runs["run"]["task_deadline"]
            for index, segment in enumerate(plan["steps"][0]["held_target_segments"]):
                self.assertEqual(len(sent), index + 1)
                if segment["type"] == "GRIPPER":
                    duration = segment["limits"]["command_duration_s"] + .002
                else:
                    begin, end = segment["action_range"]
                    duration = (end - begin) * plan["learned_proposal"]["period_s"]
                finish = round(now[0] + duration, 9)
                # Keep the ordinary heartbeat and fresh native state alive while
                # a command takes longer than the original camera input-age bound.
                while now[0] + .1 < finish:
                    now[0] = round(now[0] + .1, 9)
                    self.assertTrue(job.poll()["ok"])
                    self.assertEqual(len(sent), index + 1)
                now[0] = finish
                if segment["type"] == "GRIPPER":
                    state.update(reference=segment["gripper_position_m"], feedback=segment["gripper_position_m"])
                else:
                    state["joints"] = segment["final_joint_state"][:]
                state["complete"] = True
                result = job.poll()
                self.assertTrue(result["ok"], result["code"])
                self.assertEqual(executor.runs["run"]["task_deadline"], deadline)
        self.assertEqual(result["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertGreater(now[0] - 10., .3)
        self.assertLess(now[0], deadline)
        self.assertEqual(executor.runs["run"]["plan"], plan)
        self.assertEqual(len(sent), 3)
        self.assertNotIn(("recorder", "commit"), calls)

    def test_native_source_verification_leaves_motion_deadline_and_revocation_responsive(self):
        from tools.data_factory.learned_action_adapter import NativeSmolVLA
        from tools.data_factory.run_job import _infer_native_program
        for stop in ("deadline", "revoke", "cancel", "slow"):
            with self.subTest(stop=stop):
                job, executor, transport, _, _, now, _ = self.make_job()
                grant = task_grant(source(), SCENE, job._program["learned_proposal"])
                self.assertTrue(job.admit_task(grant)["ok"])
                self.assertTrue(job.start()["ok"])
                now[0] = 10.1
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                original_deadline = executor.runs["run"]["task_deadline"]
                native = NativeSmolVLA.__new__(NativeSmolVLA)
                native._inference_lock = threading.Lock()
                native.policy_dir, native.checkpoint = Path("unused-cpu-fixture"), CHECKPOINT
                native._predict = mock.Mock(return_value=[ACTION[:]])
                entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
                results, errors, captures = [], [], []
                def verify(_):
                    entered.set()
                    if not release.wait(2.):
                        raise AssertionError("test failed to release source verification")
                    return CHECKPOINT["tree_digest"]
                def capture(*_):
                    captures.append(now[0])
                    value = observation()
                    value["observation.state"] = ACTION[:]
                    value["source_timestamps_s"] = dict.fromkeys(("state", "camera1", "camera2"), now[0])
                    return value
                transport.capture_policy_observation = capture
                def observe():
                    result = job.observe_learned_boundary({"camera1": "/up", "camera2": "/wrist"})
                    if not result["ok"]:
                        raise ContractError(result["code"])
                    return result["observation"]
                def infer():
                    try:
                        results.append(_infer_native_program(native, source(), executor, cancel,
                            urdf="unused-cpu-fixture", instruction=OPTIONS["instruction"], period_s=.1, observation=observe))
                    except Exception as exc:
                        errors.append(exc)
                with mock.patch("tools.data_factory.training_receipts.tree_digest", side_effect=verify) as digest, \
                     mock.patch.object(Path, "read_text", return_value=XML), \
                     mock.patch("tools.data_factory.rollout.finite_plan.FinitePolicyInference",
                                side_effect=lambda *a, **kw: FinitePolicyInference(*a, **kw, source_clock=lambda: now[0])):
                    worker = threading.Thread(target=infer)
                    worker.start()
                    try:
                        self.assertTrue(entered.wait(1.))
                        if stop == "deadline":
                            now[0] = 101.
                            executor.tick()
                        elif stop == "revoke":
                            result = executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "revoke-during-verify",
                                "op": "revoke_task", "payload": {"run_id": "run", "grant_digest": grant["grant_digest"]}})
                            self.assertTrue(result["ok"], result)
                        elif stop == "cancel":
                            self.assertFalse(job.cancel()["ok"])
                            cancel.set()
                        else:
                            now[0] += .31
                        self.assertTrue(worker.is_alive())
                        self.assertEqual(captures, [])
                        if stop != "slow":
                            self.assertEqual(executor.runs["run"]["state"], "BLOCKED")
                            self.assertTrue(executor.runs["run"]["cancel_event"].is_set())
                    finally:
                        release.set()
                        worker.join(2.)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(digest.call_count, 1)
                self.assertEqual(executor.runs["run"]["task_deadline"], original_deadline)
                if stop == "slow":
                    self.assertEqual(errors, [])
                    self.assertEqual(captures, [10.41])
                    # The one byte check preceded capture; admission/send do no I/O.
                    with mock.patch("tools.data_factory.training_receipts.tree_digest", side_effect=AssertionError("motion owner hashed")):
                        self.assertTrue(job.prepare_next_learned(results[0])["ok"])
                        self.assertTrue(job.admit_task()["ok"])
                        self.assertTrue(job.start_next_learned()["ok"])
                    self.assertEqual(len(transport.sent), 2)
                else:
                    self.assertEqual(results, [])
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], ContractError)
                    self.assertEqual(len(transport.sent), 1)
                    native._predict.assert_not_called()

    def test_task_grant_policy_reserve_handoff_and_monotonic_deadline(self):
        for rollback in (False, True):
            with self.subTest(rollback=rollback):
                job, executor, transport, _, _, now, calls = self.make_job()
                grant = task_grant(source(), SCENE, job._program["learned_proposal"], max_outputs=10)
                self.assertTrue(job.admit_task(grant)["ok"])
                self.assertTrue(job.start()["ok"])
                now[0] = 10.1
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                executor.runs["run"]["execution"].update(lease_deadline=200., wait_deadline=200.)
                if rollback:
                    executor.source_clock = lambda: 10.1
                    now[0] = 100.
                    result = job.poll()
                    self.assertEqual(result["code"], "TASK_DEADLINE_EXHAUSTED")
                else:
                    now[0] = 85.
                    result = job.task_boundary()
                    self.assertEqual(result["task_handoff"]["termination_reason"], "TASK_POLICY_BUDGET_EXHAUSTED")
                    self.assertEqual(result["task_handoff"]["deadline_s"], 100.)
                self.assertFalse(result["ok"])
                self.assertEqual(len(transport.sent), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_native_terminal_illumination_uses_current_mapped_scene_image(self):
        from sensor_msgs.msg import Image, JointState
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        native = object.__new__(RosMoveItTransport)
        native._active, native._execution_locked = None, False
        native._clock, native.graph_timeout_s = lambda: 10., .01
        native._joint_state = native._joint_state_received_at = None
        callbacks, destroyed = {}, []
        native.node = SimpleNamespace(
            get_parameter=lambda _: SimpleNamespace(value=False),
            create_subscription=lambda _type, topic, callback, _qos: callbacks.update({topic: callback}) or topic,
            destroy_subscription=destroyed.append)
        inputs = {"camera_topics": {"camera1": "/wrist", "camera2": "/up"},
                  "camera_mapping": {"observation.images.up": "observation.images.camera2",
                                     "observation.images.wrist": "observation.images.camera1"}}
        plan = {"learned_proposal": {"runtime_inputs": inputs, "max_observation_age_s": .3},
                "learned_source_program": {"synthetic_test": True}, "scene_binding": SCENE}
        frame_value, stamp = [100], [10]
        def spin(*_, **__):
            joint = JointState(name=JOINTS, position=ACTION)
            joint.header.stamp.sec = 10
            native._joint_state, native._joint_state_received_at = joint, 10.
            for topic, callback in callbacks.items():
                # An occluded wrist must not stand in for the UP scene camera.
                value = frame_value[0] if topic == "/up" else 0
                image = Image(height=1, width=1, encoding="rgb8", step=3, data=bytes([value] * 3))
                image.header.stamp.sec = stamp[0]
                callback(image)
        native._rclpy = SimpleNamespace(spin_once=spin)
        frozen = copy.deepcopy(plan)
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", return_value=10.):
            for brightness in (100, 0):
                frame_value[0] = brightness
                result = native.capture_scene_illumination(plan)
                self.assertEqual(result["brightness_mean"], brightness)
                self.assertEqual(result["camera_topic"], "/up")
                self.assertEqual(result["source_timestamp_s"], 10.)
                self.assertEqual(result["valid_until_s"], 10.3)
                self.assertNotIn("physical_success", result)
            context = native.mechanical_contact_context(plan, {}, {})
            self.assertEqual(context["status"], "BLOCKED_UNAVAILABLE")
            self.assertIs(context["physical_success"], False)
            self.assertEqual(context["illumination"]["brightness_mean"], 0.)
            self.assertNotIn("CURRENT_REQUIRED_ILLUMINATION", context["missing_measurements"])
            stamp[0] = 9
            with self.assertRaisesRegex(ContractError, "LEARNED_STALE_OBSERVATION"):
                native.capture_scene_illumination(plan)
            self.assertEqual(len(destroyed), 8)
            native._active = object()
            with self.assertRaisesRegex(ContractError, "ROS_EXEC_ACTIVE"):
                native.capture_scene_illumination(plan)
            native._active = None

            with self.assertRaisesRegex(ContractError, "MECHANICAL_ILLUMINATION_UNAVAILABLE"):
                native.capture_scene_illumination({"learned_proposal": {}})
        self.assertEqual(plan, frozen)

    def test_calibrated_plateau_does_not_determine_axial_release_clearance(self):
        """Source geometry counterexample; neither pose is a physical observation."""
        import xml.etree.ElementTree as ET
        root = Path(__file__).resolve().parents[3]
        config = root / "config/data_factory"
        grasp = json.loads((config / "grasps/wood-cube-24mm-top-3p5mm-r001.json").read_text())
        obj = json.loads((config / "objects/wood-cube-24mm-r001.json").read_text())
        qualification = json.loads((config / "motion_qualifications/fr5-place-a-wood-cube-24mm-r001.json").read_text())
        model_bytes = (root / "src/fairino_description/urdf/fairino5_v6.urdf").read_bytes()
        self.assertEqual("sha256:" + hashlib.sha256(model_bytes).hexdigest(), qualification["robot_description_digest"])
        model = ET.fromstring(model_bytes)
        def z(element):
            return float(element.attrib["xyz"].split()[2])
        mount = sum(z(model.find(f"joint[@name='{name}']/origin"))
                    for name in ("gripper_adapter_joint", "gripper_joint"))
        geometry = grasp["grasp_geometry"]
        nominal = qualification["tool_to_tcp"]["translation_m"][2] - mount + geometry["datum_to_tcp_grasp"]["translation_m"][2]
        shift = geometry["depth_from_top_mm"] / 1000
        half_cube = obj["dimensions_mm"][2] / 2000
        for side in ("left", "right"):
            joint = model.find(f"joint[@name='finger_{side}_joint']")
            collision = model.find(f"link[@name='finger_tip_{side}_link']/collision")
            center = z(joint.find("origin")) + z(collision.find("origin"))
            half_tip = float(collision.find("geometry/box").attrib["size"].split()[2]) / 2
            # Translating the cube along the finger leaves opposed-face width
            # and thus the calibrated closure reading unchanged in both cases.
            for datum in (nominal, nominal + shift):
                self.assertGreater(min(center + half_tip, datum + half_cube)
                                   - max(center - half_tip, datum - half_cube), 0)
        self.assertAlmostEqual(nominal, .149352939145247)
        clearance = geometry["release_clearance_mm"] / 1000
        self.assertGreater(clearance, 0)
        self.assertAlmostEqual(clearance - shift, -.0015)

    def test_native_contact_producer_consumes_settled_closure_without_inventing_pose(self):
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        for rejection in (None, "motion_done_only", "pending", "outside_range", "wrong_reference"):
            with self.subTest(rejection=rejection):
                job, executor, native, state, now, sent, _, calls = self.make_held_job(mechanical=True)
                # Legacy diagnostic consumer, with prospective coverage isolated
                # by the lifecycle fixture; no AVAILABLE relation is supplied.
                source_program = job._program["source_program"]
                grant = task_grant(source_program, SCENE, job._program["learned_proposal"],
                                   max_outputs=1, terminal_reserve_s=30.)
                self.assertTrue(job.admit_task(grant)["ok"])
                with mock.patch("tools.data_factory.motion.moveit_transport.time.time", side_effect=lambda: now[0]):
                    self.assertTrue(job.start()["ok"])
                    while job.state != "LEARNED_CHUNK_COMPLETE":
                        goal = sent[-1]
                        if hasattr(goal.trajectory, "joint_trajectory"):
                            point = goal.trajectory.joint_trajectory.points[-1]
                            state["joints"] = list(point.positions)
                        else:
                            point = goal.trajectory.points[-1]
                            reference = point.positions[0]
                            state.update(reference=reference, feedback=.01218 if reference == .01176 else reference)
                        now[0] = round(now[0] + point.time_from_start.sec + point.time_from_start.nanosec/1e9 + .002, 9)
                        state["complete"] = True
                        result = job.poll()
                        self.assertTrue(result["ok"], result["code"])
                    count = len(sent)
                    if rejection == "motion_done_only": state["hardware_override"]["completion_reason"] = 1.
                    if rejection == "pending": state["hardware_override"]["pending"] = 1.
                    if rejection == "outside_range": state["feedback"] = .02
                    if rejection == "wrong_reference": state["reference"] = .021
                    result = job.task_boundary()
                self.assertFalse(result["ok"])
                context = result["execution_evidence"]["mechanical_contact_diagnostic"]
                contact = context["closure_contact"]
                if rejection is None:
                    self.assertEqual(contact["status"], "CALIBRATED_CLOSURE_PLATEAU")
                    self.assertEqual(contact["feedback_position_m"], .01218)
                    self.assertEqual(contact["command_position_m"], .01176)
                    self.assertEqual(contact["gripper_evidence_digest"], source_program["gripper_requirements"]["evidence_digest"])
                    self.assertEqual(context["relation_status"], "UNOBSERVED_AFTER_LEARNED_MOTION")
                else:
                    self.assertEqual(contact["status"], "UNAVAILABLE")
                self.assertEqual(context["status"], "BLOCKED_UNAVAILABLE")
                self.assertFalse(context["physical_success"])
                self.assertNotIn("carried_object", context)
                self.assertEqual(len(sent), count)
                self.assertIn(("recorder", "retain"), calls)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_native_mechanical_terminal_same_owner_stages_and_retains(self):
        self._native_mechanical_terminal_case()

    def test_prospective_native_contact_reaches_release(self):
        for reason in (1, 2):
            with self.subTest(reason=reason):
                self._native_mechanical_terminal_case(prospective=reason)
        self._native_mechanical_terminal_case("delayed_services", prospective=1)

    def test_continuous_equivalent_references_reach_real_contact_and_release(self):
        self._native_mechanical_terminal_case(prospective=1,
            actions=[[.001]*6+[.01177], [.001]*6+[.01178], [.001]*6+[.01179]])

    def test_continuous_calibrated_plateau_survives_arm_and_next_chunk(self):
        plan, result = self._native_mechanical_terminal_case(prospective=2,
            actions=[[.001]*6+[.01177], [.002]*6+[.01177]])
        from tools.data_factory.rollout.finite_plan import validate_execution_trace, held_target_segments
        trace = result["execution_evidence"]["learned_execution"]
        self.assertEqual(validate_execution_trace(plan, trace), trace)
        segments = plan["steps"][0]["held_target_segments"]
        self.assertLess(segments[1]["acceptable_feedback_m"]["max"], .01218)
        self.assertEqual(trace["segments"][1]["terminal_observation"]["snapshot"]["gripper_controller"]["feedback_position_m"], .01218)
        for key, value in (("selected_force", 50.), ("selected_generation", 99.),
                           ("selected_initial_percent", 55.), ("raw_reference_m", .01178),
                           ("feedback_m", .01219), ("selected_valid", 0.)):
            with self.subTest(tampered=key):
                changed = copy.deepcopy(trace)
                wire = changed["segments"][1]["terminal_observation"]["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
                wire[key] = value
                changed["trace_digest"] = canonical_digest({k: v for k, v in changed.items() if k != "trace_digest"})
                with self.assertRaises(ContractError):
                    validate_execution_trace(plan, changed)
        p = plan["learned_proposal"]
        source_program = plan["learned_source_program"]
        ordinary = [{k: v for k, v in s.items() if k != "native_close_feedback"}
                    for s in held_target_segments(source_program, p)]
        for version in (None, 3):
            legacy = copy.deepcopy(p)
            if version is None:
                legacy.pop("runtime_inputs")
            else:
                legacy["runtime_inputs"]["hardware_wire_version"] = version
            redigest(legacy)
            before = copy.deepcopy(legacy)
            self.assertEqual(held_target_segments(source_program, legacy), ordinary)
            self.assertEqual(legacy, before)
        self._native_mechanical_terminal_case(prospective=2, actions=[[.001]*6+[.01177]],
            next_actions=[[.001]*6+[.01178], [.001]*6+[.01179]])

    def test_continuous_plateau_rejects_bad_selection_and_outside_calibration(self):
        for failure in ("tuple_generation", "tuple_force", "tuple_initial", "tuple_missing", "plateau_outside"):
            with self.subTest(failure=failure):
                self._native_mechanical_terminal_case(failure, prospective=2,
                    actions=[[.001]*6+[.01177], [.002]*6+[.01177]])

    def test_real_contact_relation_survives_predecessor_bound_chunks(self):
        self._native_mechanical_terminal_case(prospective=1, actions=[[.001]*6+[.01177]],
            next_actions=[[.001]*6+[.01178],[.001]*6+[.01179]])

    def test_continuous_contact_rejects_unbound_native_tuple_and_missing_coverage(self):
        for failure in ("tuple_generation", "tuple_endpoint", "tuple_force", "tuple_velocity", "tuple_index",
                        "tuple_max_time", "tuple_arg5", "tuple_missing", "coverage_removed", "changed_contact_scene"):
            with self.subTest(failure=failure):
                self._native_mechanical_terminal_case(failure, prospective=1)
        self._native_mechanical_terminal_case("held_opening", prospective=1,
            actions=[[.001]*6+[.01177],[.001]*6+[.0126]])

    def test_carried_relation_survives_arm_only_suffix(self):
        self._native_mechanical_terminal_case(prospective=1,
            actions=[[.001]*6+[.01177],[.002]*6+[.01177]])

    def test_actual8000_raw_output_topology_and_native_contact_boundaries(self):
        from collections import Counter
        data = json.loads(Path(__file__).with_name("recorded8000.json").read_text())
        self.assertEqual(data["source_sha256"], "cf5d6fc6ba7ed27c3dbe1de2a15f7e34d2dd084c19509b11a726dab6f67837aa")
        self.assertEqual(len(data["samples"]), 18)
        xml = (Path(__file__).resolve().parents[3]/"src/fairino_description/urdf/fairino5_v6.urdf").read_text()
        counts, built = Counter(), []
        for sample in data["samples"]:
            rows = sample["actions"]
            self.assertEqual(len(rows), 50)
            self.assertTrue(all(len(row)==7 and abs(row[6]-.01176)>1e-9 for row in rows))
            self.assertEqual(sum(a[6]!=b[6] for a,b in zip([sample["initial_state"],*rows],rows)), 50)
            obs = observation()
            obs["observation.state"] = sample["initial_state"]
            try:
                p = FinitePolicyInference(lambda _: rows, CHECKPOINT, source_clock=lambda:10.).propose(
                    obs, instruction="stored CPU replay", robot_description=xml, period_s=1/30, serialized_references=True)
                src = source()
                src["binding_digests"]["robot_description_digest"] = "sha256:"+hashlib.sha256(xml.encode()).hexdigest()
                next(s for s in src["steps"] if s["phase"]=="GRIPPER_OPEN")["gripper_position_m"] = .021
                plan = compile_program(src,p)
                self.assertEqual(p["actions"],rows)
                self.assertEqual([s["type"] for s in plan["steps"][0]["held_target_segments"]], ["ARM","GRIPPER"]*50)
                built.append(sample)
            except ContractError as exc:
                counts[exc.code]+=1
        self.assertEqual(counts, {"LEARNED_JOINT_LIMIT":12,"LEARNED_REFERENCE_HORIZON":3})
        self.assertEqual([(s["episode_index"],s["frame_index"]) for s in built], [(27,568),(28,567),(32,467)])
        # Only service geometry and hardware feedback are simulated. Every one
        # of the 50 stored policy rows remains present, including rejected suffixes.
        for sample in built:
            with self.subTest(episode=sample["episode_index"]):
                self._native_mechanical_terminal_case(prospective=1, recorded=(sample,xml,CHECKPOINT))

    def test_native_task_without_contact_coverage_sends_zero_goals(self):
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        job, executor, native, _, now, sent, *_ = self.make_held_job(mechanical=True)
        native.prepare_contact_transition = RosMoveItTransport.prepare_contact_transition.__get__(native)
        self.assertTrue(job.admit_task(task_grant(job._program["source_program"],SCENE,job._program["learned_proposal"]))["ok"])
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time',side_effect=lambda:now[0]):
            result = job.start()
            self.assertEqual(result["code"], "CONTACT_PROFILE_UNAVAILABLE")
        self.assertEqual(sent, [])
        self.assertIs(executor.runs["run"]["execution"]["motion_dispatch_attempted"], False)
        self.assertEqual(executor.scene_state_store.updates, [])
        self.assertEqual(result["execution_evidence"]["scene_preservation"]["reason_code"], "NO_DISPATCH")
        self.assertNotIn("scene_transition", result["execution_evidence"])

    def test_closure_centering_envelope_contains_source_geometry_counterexample(self):
        from tools.data_factory.motion.contact_transition import prepare, lateral_envelope
        job, executor, native, *_ = self.make_held_job(mechanical=True, contact_fixture=self.prospective_contact_fixture)
        with executor.scene_state_store.locked_snapshot(SCENE["scene_state_digest"]) as scene:
            context = prepare(native, executor.runs["run"]["plan"], scene["scene_state"]["objects"]["cube-1"])
        shift = 2 * job._program["source_program"]["planning"]["goal_tolerances"]["position_m"]
        relation = {"translation_m": [context["jaw_midplane_m"] + shift, 0., .15],
                    "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]}
        width = context["dimensions_m"][0]
        self.assertAlmostEqual(shift - (context["closed_gap_bound_m"] - width)/2, .00177)
        envelope, dimensions = lateral_envelope(relation, context["dimensions_m"], context["jaw_midplane_m"], context["closed_gap_bound_m"])
        lo, hi = envelope["translation_m"][0] - dimensions[0]/2, envelope["translation_m"][0] + dimensions[0]/2
        for center in (relation["translation_m"][0], context["jaw_midplane_m"]):
            self.assertLessEqual(lo, center - width/2)
            self.assertGreaterEqual(hi, center + width/2)

    def test_prospective_native_contact_rejections_send_no_next_goal(self):
        for failure in ("earlier_contact", "geometry", "wrong_generation", "stale_state", "absent_relation",
                        "nonfinger_contact", "failed_constraint", "empty_contact", "wrong_body_type", "close_pose"):
            with self.subTest(failure=failure):
                self._native_mechanical_terminal_case(failure, prospective=1)

    def prospective_contact_fixture(self, src, transport):
        """Temporary synthetic qualification, never selected for a live caller."""
        import subprocess
        from tools.fr5_data_factory import compose_rigid_transform, inverse_rigid_transform, validate_cell_calibration_document
        repo = Path(__file__).resolve().parents[3]
        root = repo / "config/data_factory"
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        transport.contact_config_root = Path(temporary.name)
        docs = {}
        for key, folder, name in (
            ("object_profile", "objects", "wood-cube-24mm-r001"),
            ("grasp_profile", "grasps", "wood-cube-24mm-top-3p5mm-r001"),
            ("robot_system", "robot_systems", "fr5-lab-a-tcp-r002"),
            ("cell_calibration", "cells", "place-b-yaw0-r001"),
            ("yaw0_sheet", "workspace_sheets", "place-b-yaw0-r001_yaw0_sheet")):
            value = json.loads((root / folder / (name + ".json")).read_text())
            directory = transport.contact_config_root / folder
            directory.mkdir()
            (directory / (name + ".json")).write_text(json.dumps(value))
            docs[key] = value
            src["binding_digests"][key] = canonical_digest(value)
        xml = subprocess.check_output(["git", "show", "60dfb2b49fe7c5154068e8bd4b411c1342508c78:src/fairino_description/urdf/fairino5_v6_gripper_opening_candidate.urdf"], cwd=repo, text=True)
        transport._robot_description = xml
        src["binding_digests"]["robot_description_digest"] = "sha256:" + hashlib.sha256(xml.encode()).hexdigest()
        q = json.loads((root / "motion_qualifications/fr5-place-b-wood-cube-24mm-r001.json").read_text())
        q.update(motion_qualification_id="synthetic-contact-only", frames=src["frames"],
                 robot_description_digest=src["binding_digests"]["robot_description_digest"])
        directory = transport.contact_config_root / "motion_qualifications"
        directory.mkdir()
        (directory / "synthetic-contact-only.json").write_text(json.dumps(q))
        src["binding_digests"]["motion_qualification"] = canonical_digest(q)
        src["gripper_requirements"] = copy.deepcopy(docs["grasp_profile"]["gripper_close"])
        src["gripper_requirements"].update(open_velocity_percent=docs["grasp_profile"]["gripper_open"]["velocity_percent"],
                                           open_force_percent=docs["grasp_profile"]["gripper_open"]["force_percent"])
        src["robot_system_id"] = docs["robot_system"]["robot_system_id"]
        src["planning"]["max_joint_state_age_s"] = .1
        cal = validate_cell_calibration_document(docs["cell_calibration"], yaw0=docs["yaw0_sheet"],
            robot=docs["robot_system"], required_status="QUALIFIED")
        datum = {"translation_m": cal["center"], "rotation_columns": [cal["x"],cal["y"],cal["z"]]}
        tcp = compose_rigid_transform(datum, q["datum_to_tcp_grasp"])
        tool = compose_rigid_transform(tcp, inverse_rigid_transform(q["tool_to_tcp"]))
        for step in src["steps"]:
            if step["phase"] == "GRIPPER_OPEN":
                step["limits"]["completion_tolerance_m"] = docs["grasp_profile"]["gripper_open"]["completion_tolerance_m"]
            if step["phase"] == "FINAL_APPROACH_LIN":
                step["target"] = {"base_tcp": tcp, "base_tool": tool}
        transport._fixture_contact_tool = tool
        transport._fixture_contact_gripper = compose_rigid_transform(tool,
            {"translation_m": [0.,0.,.109], "rotation_columns": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]})
        item = {"state": "ON_SURFACE", "object_profile_id": docs["object_profile"]["object_profile_id"],
                "pose": {"place_id": "PLACE_B", "yaw_deg": 0., "x_mm": 0., "y_mm": 0.}}
        class InitialScene(Scene):
            @contextmanager
            def locked_snapshot(self, digest, *, blocking=True):
                yield {"scene_state_digest": digest, "scene_state": {"revision": 1, "objects": {"cube-1": item}}}
        return xml, InitialScene()

    def test_native_mechanical_terminal_rejection_boundaries(self):
        for failure in ("stale_contact","dark","scope","scene","revoke","partial_feedback","incarnation",
                        "dark_dispatch", "stale_dispatch", "expired_during_decode", "dark_staged_open"):
            with self.subTest(failure=failure):
                self._native_mechanical_terminal_case(failure)

    def _native_mechanical_terminal_case(self, failure=None, prospective=False, recorded=None, actions=None, next_actions=None):
        """Explicit simulated contact/FK services; never a live qualification."""
        from moveit_msgs import msg, srv
        from geometry_msgs.msg import Pose, PoseStamped
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rclpy.serialization import serialize_message
        def contact_fixture(src, transport):
            value = self.prospective_contact_fixture(src, transport)
            if actions is not None:
                transport._fixture_actions = actions
                for step in src["steps"]:
                    if step["phase"] == "SAFE_POSE_PTP":
                        step["limits"]["execution_timeout_s"] = 10.
            return value
        job, executor, transport, state, now, sent, _, calls = self.make_held_job(mechanical=True, max_observation_age_s=.3,
            recorded=recorded, contact_fixture=contact_fixture if prospective else None)
        for name in ("CollisionObject", "PlanningScene", "PlanningSceneComponents", "RobotState"):
            setattr(transport, "_"+name, getattr(msg,name))
        for name in ("ApplyPlanningScene", "GetPlanningScene", "GetStateValidity"):
            setattr(transport, "_"+name, getattr(srv,name))
        transport._Pose, transport._SolidPrimitive, transport._JointState = Pose, SolidPrimitive, JointState
        world, attached, services, starts = {}, [], [], []
        def service(kind, endpoint, request, code):
            if prospective:
                from rclpy.serialization import deserialize_message
                request = deserialize_message(serialize_message(request), type(request))
                if failure == "delayed_services" and not attached and executor.runs["run"]["state"] == "EXECUTING":
                    now[0] = round(now[0] + .01, 9)
            services.append(endpoint)
            if kind is srv.ApplyPlanningScene:
                for obj in request.scene.world.collision_objects:
                    if obj.operation == obj.REMOVE:
                        world.pop(obj.id,None)
                    else:
                        world[obj.id] = copy.deepcopy(obj)
                for obj in request.scene.robot_state.attached_collision_objects:
                    attached[:] = [] if obj.object.operation == obj.object.REMOVE else [copy.deepcopy(obj)]
                return SimpleNamespace(success=True)
            if kind is srv.GetPlanningScene:
                scene=msg.PlanningScene()
                scene.world.collision_objects=[] if failure=="scene" else list(world.values())
                scene.robot_state.attached_collision_objects=attached[:]
                return SimpleNamespace(scene=scene)
            if kind is srv.GetStateValidity:
                if not prospective:
                    self.assertTrue(request.robot_state.is_diff)
                self.assertEqual(request.group_name,"")
                self.assertEqual(len(attached),0 if "cube-1" in world else 1)
                if prospective and "cube-1" in world and not attached:
                    if failure == "earlier_contact":
                        return srv.GetStateValidity.Response(valid=False)
                    if recorded is None and request.robot_state.joint_state.position[-1] < .02:
                        contact = msg.ContactInformation()
                        contact.header.frame_id = "base_link"
                        contact.contact_body_1, contact.body_type_1 = "cube-1", contact.WORLD_OBJECT
                        contact.contact_body_2, contact.body_type_2 = "finger_tip_left_link", contact.ROBOT_LINK
                        contact.normal.x = 1.
                        if failure == "nonfinger_contact": contact.contact_body_2 = "wrist3_link"
                        if failure == "wrong_body_type": contact.body_type_2 = contact.WORLD_OBJECT
                        return srv.GetStateValidity.Response(valid=False,
                            contacts=[] if failure == "empty_contact" else [contact],
                            constraint_result=[msg.ConstraintEvalResult(result=False, distance=1.)] if failure == "failed_constraint" else [])
                return SimpleNamespace(valid=True)
            if kind is srv.GetPositionFK:
                pose=PoseStamped()
                pose.header.frame_id=request.header.frame_id
                pose.pose.orientation.w=1.
                pose.pose.position.z=.3
                if prospective:
                    from tools.data_factory.motion.moveit_transport import _rotation_quaternion
                    transform = transport._fixture_contact_gripper if request.fk_link_names == ["gripper_link"] else transport._fixture_contact_tool
                    pose.pose.position.x,pose.pose.position.y,pose.pose.position.z = transform["translation_m"]
                    pose.pose.orientation.x,pose.pose.orientation.y,pose.pose.orientation.z,pose.pose.orientation.w = _rotation_quaternion(transform["rotation_columns"])
                    if failure == "close_pose":
                        pose.pose.position.x += 2 * executor.runs["run"]["plan"]["planning"]["goal_tolerances"]["position_m"]
                return SimpleNamespace(error_code=SimpleNamespace(val=1),fk_link_names=request.fk_link_names,pose_stamped=[pose])
            self.fail(endpoint)
        transport._service=service
        def plan_arm(phase,target,joints,limits,frames,planning,start, **options):
            if phase in {"RETREAT_LIN", "SAFE_POSE_PTP"}:
                diff = options["planning_scene_diff"]
                self.assertEqual(len(attached), 1)
                self.assertNotIn("cube-1", world)
                self.assertEqual(diff.robot_state.attached_collision_objects[0].object.operation, msg.CollisionObject.REMOVE)
                self.assertEqual(diff.world.collision_objects[0].id, "cube-1")
                self.assertTrue(serialize_message(diff))
            starts.append(start[:])
            trajectory=msg.RobotTrajectory()
            trajectory.joint_trajectory.joint_names=JOINTS[:6]
            point=JointTrajectoryPoint(positions=list(joints or start))
            point.time_from_start.nanosec=100000000
            trajectory.joint_trajectory.points=[point]
            return {"terminal_status":"SUCCEEDED","moveit_success":True,
                    "serialized_trajectory":serialize_message(trajectory),"final_joint_state":list(joints or start)}
        transport.plan_arm=plan_arm
        if prospective:
            from moveit_msgs.action import MoveGroup
            from tools.data_factory.motion.moveit_transport import RosMoveItTransport
            transport._MoveGroup = MoveGroup
            for name in ("Constraints", "JointConstraint", "PositionConstraint", "OrientationConstraint"):
                setattr(transport, "_" + name, getattr(msg, name))
            def planner_send(goal):
                self.assertTrue(goal.planning_options.plan_only)
                self.assertTrue(serialize_message(goal))
                constraint = goal.request.goal_constraints[0]
                start = list(goal.request.start_state.joint_state.position)
                joints = [c.position for c in constraint.joint_constraints] or None
                options = ({"planning_scene_diff": goal.planning_options.planning_scene_diff}
                           if constraint.name in {"RETREAT_LIN", "SAFE_POSE_PTP"} else {})
                if options:
                    self.assertEqual(list(goal.planning_options.planning_scene_diff.robot_state.joint_state.position), [.021])
                built = plan_arm(constraint.name, None, joints, None, None, None, start, **options)
                result = MoveGroup.Result()
                result.error_code.val = 1
                result.planned_trajectory = transport._deserialize_message(built["serialized_trajectory"], msg.RobotTrajectory)
                future = mock.Mock(done=lambda: True, result=lambda: SimpleNamespace(status=4, result=result))
                handle = mock.Mock(accepted=True, get_result_async=lambda: future)
                return mock.Mock(done=lambda: True, result=lambda: handle)
            transport.move_group = mock.Mock(send_goal_async=planner_send)
            transport.plan_arm = RosMoveItTransport.plan_arm.__get__(transport)
            self.assertIs(transport.mechanical_contact_context.__func__, RosMoveItTransport.mechanical_contact_context)
        illumination_captures = []
        def illumination_sample():
            return {"kind": "CURRENT_REQUIRED_ILLUMINATION", "camera_topic": "/up",
                    "brightness_mean": 100., "source_timestamp_s": now[0], "valid_until_s": now[0] + .3}
        def capture_illumination(plan):
            self.assertIsNone(transport._active)
            self.assertFalse(transport._execution_locked)
            illumination_captures.append(now[0])
            sample = illumination_sample()
            if failure == "dark_dispatch" or failure == "dark_staged_open" and len(illumination_captures) == 4:
                sample["brightness_mean"] = 0.
            if failure == "stale_dispatch":
                sample["source_timestamp_s"] -= 1.
                sample["valid_until_s"] -= 1.
            return sample
        transport.capture_scene_illumination = capture_illumination
        if failure == "coverage_removed":
            decode = transport._compiled_execution_goal
            def remove_coverage(step):
                result = decode(step)
                executor.runs["run"]["execution"].pop("prospective_contact",None)
                return result
            transport._compiled_execution_goal = remove_coverage
        if failure == "expired_during_decode":
            decode = transport._compiled_execution_goal
            def delayed_decode(step):
                result = decode(step)
                if step["phase"] != "LEARNED_CHUNK":
                    now[0] += .4
                return result
            transport._compiled_execution_goal = delayed_decode
        def simulated_contact(plan,scene,snapshot, **kwargs):
            context={"status":"AVAILABLE","semantics":"MODEL_BASED_EXPECTATION","physical_success":False,
                "source_program_digest":canonical_digest(plan["learned_source_program"]),
                "snapshot_digest":canonical_digest(snapshot),"scene_binding":copy.deepcopy(plan["scene_binding"]),
                "qualification_source_digest":plan["binding_digests"]["motion_qualification"],"valid_until_s":90.,
                "illumination":illumination_sample(),
                "carried_object":{"object_id":"cube-1","link_name":"gripper_link",
                    "touch_links":["finger_tip_right_link","finger_tip_left_link"],"dimensions_m":[.024]*3,
                    "translation_m":[0.,0.,.1],"rotation_xyzw":[0.,0.,0.,1.]}}
            if failure=="stale_contact": context["valid_until_s"]=now[0]-.1
            if failure=="dark": context["illumination"]["brightness_mean"]=0.
            if failure=="scope": context["source_program_digest"]=canonical_digest("foreign")
            return context
        if not prospective:
            transport.mechanical_contact_context=simulated_contact
        original=copy.deepcopy(executor.runs["run"]["plan"])
        if failure == "geometry":
            transport._robot_description += " "
        grant=task_grant(job._program["source_program"],SCENE,original["learned_proposal"],max_outputs=2 if next_actions else 1,terminal_reserve_s=30.)
        self.assertTrue(job.admit_task(grant)["ok"])
        with mock.patch('tools.data_factory.motion.moveit_transport.time.time',side_effect=lambda:now[0]):
            started = job.start()
            if failure in {"earlier_contact", "geometry", "coverage_removed"}:
                self.assertFalse(started["ok"])
                self.assertEqual(sent, [])
                return
            self.assertTrue(started["ok"], started["code"])
            deadline=executor.runs["run"]["task_deadline"]
            def complete_goal():
                goal=sent[-1]
                if hasattr(goal,"trajectory") and hasattr(goal.trajectory,"joint_trajectory"):
                    point=goal.trajectory.joint_trajectory.points[-1]
                    state["joints"]=list(point.positions)
                else:
                    point=goal.trajectory.points[-1]
                    value=point.positions[0]
                    state.update(reference=value,feedback=value)
                    if recorded is not None or actions is not None:
                        state["feedback"] = math.floor(value/.021*100+.5)*.021/100
                        state["hardware_override"]["completion_reason"] = 1.
                    if prospective and math.floor(value/.021*100+.5) == 56:
                        state["hardware_override"]["completion_reason"] = float(prospective)
                        if prospective == 2:
                            state["feedback"] = .01219 if failure == "plateau_outside" else .01218
                        changes = {"tuple_generation":("selected_generation",99.), "tuple_endpoint":("selected_position",55.),
                            "tuple_force":("selected_force",50.), "tuple_velocity":("selected_velocity",10.),
                            "tuple_index":("selected_index",2.), "tuple_max_time":("selected_max_time",10000.),
                            "tuple_arg5":("selected_arg5",0.), "tuple_missing":("selected_valid",0.),
                            "tuple_initial":("selected_initial_percent",55.)}
                        if failure in changes:
                            key,value = changes[failure]
                            state["hardware_override"][key] = value
                duration=point.time_from_start.sec+point.time_from_start.nanosec/1e9+.002
                while duration>.2:
                    now[0]=round(now[0]+.2,9)
                    duration-=.2
                    self.assertTrue(job.poll()["ok"])
                    if point.positions == [.0126]:
                        self.assertIs(sent[-1], goal)
                now[0]=round(now[0]+duration,9)
                state["complete"]=True
                if failure == "changed_contact_scene":
                    world.pop("cube-1",None)
                return job.poll()
            while job.state != "LEARNED_CHUNK_COMPLETE":
                completed=complete_goal()
                if failure and failure.startswith("tuple_") or failure in {"changed_contact_scene","held_opening","plateau_outside"}:
                    if not completed["ok"]:
                        self.assertEqual(len(sent), 1 if failure=="changed_contact_scene" else 3 if failure=="held_opening" else 2)
                        self.assertNotIn("mechanical_terminal",executor.runs["run"])
                        if prospective == 2 and (failure.startswith("tuple_") or failure == "plateau_outside"):
                            # Rejection occurs inside the native handoff, before
                            # a verified stop. Preserve the recording; do not
                            # manufacture successful cancellation or export.
                            self.assertEqual(job.cancel_error, "ROS_EXEC_CANCEL_NOT_CANCELED")
                            self.assertEqual(job.recorder_state, "RECORDING")
                            self.assertTrue(transport.owns_active_goal)
                            self.assertNotIn(("recorder", "retain"), calls)
                            self.assertNotIn(("recorder", "abort"), calls)
                        else:
                            self.assertIn(("recorder","retain"),calls)
                        self.assertNotIn(("recorder", "commit"), calls)
                        return
                if recorded is not None and not completed["ok"]:
                    self.assertEqual(completed["code"], "CONTACT_HELD_EVOLUTION_UNSUPPORTED")
                    self.assertEqual(executor.runs["run"]["plan"], original)
                    trace = executor._execution_data(executor.runs["run"])["learned_execution"]
                    self.assertLess(len(trace["reference_consumption"]["completed_row_indices"]), 50)
                    self.assertEqual(original["learned_proposal"]["actions"], recorded[0]["actions"])
                    self.assertLess(len(sent), 100)
                    self.assertNotIn("mechanical_terminal", executor.runs["run"])
                    return
                if failure in {"nonfinger_contact", "failed_constraint", "empty_contact", "wrong_body_type", "close_pose"}:
                    self.assertFalse(completed["ok"])
                    self.assertEqual(len(sent), 1)
                    return
                self.assertTrue(completed["ok"],completed["code"])
            if next_actions is not None:
                transaction, lease = job.transaction_id, job.lease_id
                old_context = copy.deepcopy(executor.runs["run"]["execution"]["prospective_contact"])
                obs = observation()
                obs["observation.state"] = [*state["joints"],state["feedback"]]
                obs["source_timestamps_s"] = dict.fromkeys(("state","camera1","camera2"),now[0])
                p = FinitePolicyInference(lambda _:next_actions,CHECKPOINT,source_clock=lambda:now[0]).propose(obs,
                    instruction="synthetic probe",robot_description=transport._robot_description,period_s=1/30,
                    max_observation_age_s=.3,serialized_references=True,
                    runtime_inputs=original["learned_proposal"].get("runtime_inputs"))
                built = compile_program(original["learned_source_program"],p)
                prepared = job.prepare_next_learned(built)
                self.assertTrue(prepared["ok"],prepared["code"])
                self.assertTrue(job.admit_task()["ok"])
                started = job.start_next_learned()
                self.assertTrue(started["ok"],started["code"])
                self.assertEqual((transaction,lease),(job.transaction_id,job.lease_id))
                history = executor.runs["run"]["learned_history"]
                self.assertEqual(history[-1]["plan_envelope"]["plan"],original)
                self.assertEqual(executor.runs["run"]["execution"]["prospective_contact"]["close"],old_context["close"])
                original = copy.deepcopy(executor.runs["run"]["plan"])
                while job.state != "LEARNED_CHUNK_COMPLETE":
                    result = complete_goal()
                    self.assertTrue(result["ok"],result["code"])
                self.assertEqual(original["learned_proposal"]["actions"],next_actions)
                self.assertEqual(state["generation"],3)
            learned_count=len(sent)
            if failure == "wrong_generation":
                state["generation"] += 1
            if failure == "stale_state":
                state["age"] = 2.
            if failure == "absent_relation":
                executor.runs["run"]["execution"]["prospective_contact"]["close"] = None
            if failure=="revoke":
                recorder=job.recorder_call
                def revoke_on_freeze(request):
                    result=recorder(request)
                    if request["op"]=="freeze":
                        executor.runs["run"]["task_revoked"]=True
                    return result
                job.recorder_call=revoke_on_freeze
            result=job.task_boundary()
            if recorded is not None:
                self.assertFalse(result["ok"])
                self.assertEqual(len(sent), 100)
                self.assertEqual(original["learned_proposal"]["actions"], recorded[0]["actions"])
                self.assertNotIn("mechanical_terminal", executor.runs["run"])
                return
            if failure in {"wrong_generation", "stale_state", "absent_relation"}:
                self.assertFalse(result["ok"])
                self.assertEqual(len(sent), learned_count)
                self.assertIn(("recorder", "retain"), calls)
                return
            if failure in {"stale_contact","dark","scope","scene","revoke",
                           "dark_dispatch", "stale_dispatch", "expired_during_decode"}:
                self.assertFalse(result["ok"])
                self.assertEqual(len(sent),learned_count)
                self.assertIn(("recorder","retain"),calls)
                self.assertNotIn(("recorder","commit"),calls)
                if failure in {"stale_contact","dark","scope"}: self.assertEqual(services,[])
                return
            self.assertEqual(result["code"],"MECHANICAL_TERMINAL_STARTED")
            self.assertEqual(job.recorder_state,"FROZEN")
            self.assertEqual(starts[0],original["learned_proposal"]["actions"][-1][:6])
            for index in range(6):
                if failure=="incarnation" and index==0:
                    state["hardware_override"]["incarnation_0"]=99.
                if failure=="partial_feedback" and index==2:
                    # Complete the partial command with incorrect measured feedback.
                    original_poll=job.poll
                    def bad_feedback():
                        state["feedback"]=.016
                        return original_poll()
                    job.poll=bad_feedback
                result=complete_goal()
                if (failure=="incarnation" and index==0 or failure=="partial_feedback" and index==2
                        or failure=="dark_staged_open" and index==2):
                    self.assertFalse(result["ok"])
                    self.assertEqual(len(sent)-learned_count,index+1)
                    self.assertIn(("recorder","retain"),calls)
                    self.assertNotIn(("recorder","commit"),calls)
                    return
                if index < 5:
                    self.assertTrue(result["ok"],(index,result["code"]))
            terminal=result["execution_evidence"]["mechanical_terminal"]
            self.assertEqual(terminal["status"],"COMPLETED",(result["code"],terminal.get("failure_code")))
            self.assertEqual(terminal["terminal_phases"],["RECYCLE_APPROACH_PTP","LOWER_LIN","GRIPPER_OPEN","RETREAT_LIN","SAFE_POSE_PTP"])
            self.assertEqual(len(sent)-learned_count,6)
            self.assertEqual(len(illumination_captures), 7 if prospective else 6)
            self.assertGreater(now[0], terminal["plan"]["contact_evidence"]["illumination"]["valid_until_s"])
            self.assertEqual(terminal["last_illumination"]["source_timestamp_s"], illumination_captures[-1])
            self.assertEqual([sent[learned_count+i].trajectory.points[-1].positions[0] for i in (2,3)],[.0126,.021])
            self.assertEqual(executor.runs["run"]["plan"],original)
            self.assertEqual(executor.runs["run"]["task_deadline"],deadline)
            self.assertEqual(terminal["plan"]["grant_digest"],grant["grant_digest"])
            if prospective:
                contact = terminal["plan"]["contact_evidence"]
                self.assertEqual(contact["closure_contact"]["status"], "CALIBRATED_ENDPOINT_COMPLETION" if prospective == 1 else "CALIBRATED_CLOSURE_PLATEAU")
                self.assertEqual(contact["relation_semantics"], "OPPOSED_JAW_LATERAL_ENVELOPE")
                self.assertGreaterEqual(contact["carried_object"]["dimensions_m"][0], .024)
                self.assertIs(contact["physical_success"], False)
                if failure == "delayed_services":
                    first = contact["prospective_transition"]["checked_segments"][0]
                    self.assertGreater(first["start_observation"]["captured_at_s"] - first["checked_start_observation"]["captured_at_s"], .1)
            self.assertEqual(result["execution_evidence"]["learned_execution"]["terminal_phases"],["LEARNED_CHUNK"])
            self.assertEqual(terminal["release_readback"]["physical_landing"],"NOT_OBSERVED")
            self.assertIn(("recorder","retain"),calls)
            self.assertNotIn(("recorder","commit"),calls)
            self.assertEqual(attached,[])
            self.assertIn("cube-1",world)
            self.assertIn("/compute_fk",services)
            diagnostic=learned_run_diagnostic(result)
            self.assertEqual(diagnostic["mechanical_terminal"],terminal)
            self.assertEqual(diagnostic["task_effectiveness"],"UNKNOWN")
            self.assertFalse(diagnostic["training_authorized"])
            return original, result

    def test_task_grant_rejects_illegal_expired_and_revoked_without_goals(self):
        for change, expected in (({"revoked": True}, "TASK_GRANT_REVOKED"),
                                 ({"deadline_s": 10.}, "TASK_DEADLINE_EXHAUSTED"),
                                 ({"scope": {}}, "TASK_GRANT_SCOPE"),
                                 ({"deadline_s": 24.}, "TASK_POLICY_BUDGET_EXHAUSTED")):
            with self.subTest(change=change):
                job, executor, transport, _, _, _, calls = self.make_job()
                grant = task_grant(source(), SCENE, job._program["learned_proposal"], **change)
                result = job.admit_task(grant)
                self.assertEqual(result["code"], expected, result)
                self.assertEqual(transport.sent, [])
                self.assertNotIn(("recorder", "begin"), calls)

    def test_task_grant_deadline_and_revocation_fence_active_goal(self):
        for revoked in (True, False):
            with self.subTest(revoked=revoked):
                job, executor, transport, _, _, now, calls = self.make_job()
                grant = task_grant(source(), SCENE, job._program["learned_proposal"])
                self.assertTrue(job.admit_task(grant)["ok"])
                self.assertTrue(job.start()["ok"])
                deadline = executor.runs["run"]["task_deadline"]
                self.assertEqual(len(transport.sent), 1)
                if revoked:
                    transport.poll_active = lambda: None
                    response = executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "revoke",
                        "op": "revoke_task", "payload": {"run_id": "run", "grant_digest": grant["grant_digest"]}})
                    self.assertTrue(response["ok"], response)
                else:
                    now[0] = 100.
                result = job.poll()
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["code"], "TASK_GRANT_REVOKED" if revoked else "TASK_DEADLINE_EXHAUSTED")
                self.assertEqual(executor.runs["run"]["task_deadline"], deadline)
                self.assertFalse(transport.owns_active_goal)
                self.assertGreaterEqual(transport.cancel_count, 1)
                self.assertEqual(len(transport.sent), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_task_grant_same_owner_rejects_changed_stale_and_unresolved_next(self):
        for failure in ("scope", "stale", "active", "late", "reserve"):
            with self.subTest(failure=failure):
                job, executor, transport, _, _, now, calls = self.make_job()
                grant = task_grant(source(), SCENE, job._program["learned_proposal"])
                self.assertTrue(job.admit_task(grant)["ok"])
                self.assertTrue(job.start()["ok"])
                now[0] = 10.1
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                candidate = self.next_raw_program(job, now[0])
                if failure == "scope":
                    candidate["learned_proposal"]["instruction"] = "different task"
                    candidate = compile_program(source(), redigest(candidate["learned_proposal"]))
                elif failure == "stale":
                    now[0] = 10.5
                elif failure == "active":
                    transport.active = True
                elif failure == "reserve":
                    now[0] = 86.
                    # Keep the unrelated heartbeat lease current to isolate task budget.
                    executor.runs["run"]["execution"].update(lease_deadline=99., wait_deadline=99.)
                else:
                    original = executor._compile_plan
                    def late(*args, **kwargs):
                        result = original(*args, **kwargs)
                        now[0] = 100.
                        return result
                    executor._compile_plan = late
                rejected = job.prepare_next_learned(candidate)
                self.assertFalse(rejected["ok"], rejected)
                self.assertEqual(len(transport.sent), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_next_chunk_keeps_one_owner_and_exact_approved_history(self):
        from tools.data_factory.rollout.finite_plan import validate_execution_history
        job, executor, transport, cell, scene, now, calls = self.start_job()
        from tools.data_factory.quality.phase_events import PhaseEventWriter, read_phase_events
        from tools.data_factory.quality.episode_report import build_episode_report
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        sidecar = Path(directory.name) / "phase_events.jsonl"
        writer = PhaseEventWriter(sidecar, plan=job.plan_envelope["plan"])
        self.addCleanup(writer.close)
        executor._phase_event_writer = writer
        executor.event_clock = lambda: (round(now[0] * 1e9), "SYSTEM_TIME")
        self.assertTrue(job.confirm("operator")["ok"])
        now[0] = round(now[0] + .1, 9)
        self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
        original_plan = copy.deepcopy(job.plan_envelope)
        transaction, lease, digest = job.transaction_id, job.lease_id, job.plan_digest
        program = self.next_raw_program(job, now[0])
        result = job.prepare_next_learned(program)
        self.assertTrue(result["ok"], result)
        candidate = result["pending_chunk"]["plan_digest"]
        self.assertEqual(job.plan_digest, digest)
        self.assertEqual(job.plan_envelope, original_plan)
        self.assertEqual(job.state, "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(job.start_next_learned()["code"], "LEARNED_NEXT_NOT_APPROVED")
        self.assertEqual(job.approve_next_learned(APPROVAL)["code"], "LEARNED_APPROVAL_REUSED")
        approval = {**APPROVAL, "approval_id": "next-exact-approval"}
        self.assertTrue(job.approve_next_learned(approval)["ok"])
        result = job.start_next_learned()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["state"], "PRECONTACT_HUMAN")
        self.assertEqual(job.plan_digest, candidate)
        self.assertEqual(len(transport.sent), 1)
        now[0] = round(now[0] + .1, 9)
        self.assertTrue(job.confirm("operator")["ok"])
        now[0] = round(now[0] + .1, 9)
        result = job.poll()
        self.assertEqual(result["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(len(transport.sent), 2)
        self.assertEqual((job.transaction_id, job.lease_id), (transaction, lease))
        self.assertEqual(calls.count(("recorder", "begin")), 1)
        self.assertNotIn(("recorder", "freeze"), calls)
        self.assertNotIn(("recorder", "commit"), calls)
        history = result["execution_evidence"]["learned_history"]
        self.assertEqual(history[0]["plan_envelope"], original_plan)
        self.assertEqual(history[0]["approval"]["plan_digest"], digest)
        self.assertEqual(history[0]["execution_evidence"]["learned_execution"]["terminal_phases"], ["LEARNED_CHUNK"])
        self.assertEqual(validate_execution_history(job.plan_envelope["plan"], history), history)
        self.assertEqual(job.plan_envelope["plan"]["learned_proposal"]["actions"], program["learned_proposal"]["actions"])
        self.assertEqual(scene.updates, [])
        self.assertFalse(cell.ready)
        self.assertTrue(writer.close())
        plans = {digest: original_plan["plan"], candidate: job.plan_envelope["plan"]}
        events = read_phase_events(sidecar, plans=plans)
        self.assertEqual([event["sequence"] for event in events], list(range(len(events))))
        self.assertEqual({event["plan_digest"] for event in events}, set(plans))
        rows = [{"target_ros_s": stamp, "action": ACTION[:], "observation.state": ACTION[:]}
                for stamp in (10.05, 10.25)]
        for plan_digest, plan in plans.items():
            report_dir = Path(directory.name) / plan_digest.replace(":", "-")
            report_dir.mkdir()
            evidence = (history[0]["execution_evidence"] if plan_digest == digest else result["execution_evidence"])
            report = build_episode_report(report_dir / "episode_quality.json", run_id="run",
                resolved_job_digest=plan["resolved_job_digest"], plan_digest=plan_digest, plan=plan, plans=plans,
                phase_events_path=sidecar, recorder_rows=rows, recorder_rows_digest=canonical_digest(rows),
                recorder_ros_clock_type="SYSTEM_TIME", execution_evidence=evidence, stall_epsilon_rad=1e-4,
                technical_validator={"schema_version": "data_factory.technical_validator_ref.v1", "status": "PASS",
                                     "result_digest": canonical_digest("synthetic-only")})
            timing = next(item for item in report["attributes"] if item["attribute"] == "phase_timing_integrity")
            self.assertEqual(timing["metrics"]["joined_row_count"], 1)
            self.assertEqual(report["plan_digest"], plan_digest)
        for mutate in (lambda h: h.clear(), lambda h: h.append(copy.deepcopy(h[0])),
                       lambda h: h[0]["approval"].update(plan_digest=candidate),
                       lambda h: h[0]["approval"].update(approved_by="different-operator")):
            changed = copy.deepcopy(history)
            mutate(changed)
            with self.assertRaisesRegex(ContractError, "LEARNED_HISTORY_BINDING"):
                validate_execution_history(job.plan_envelope["plan"], changed)
        for number in range(3):
            request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": f"history-cache-{number}",
                       "op": "status", "payload": {"run_id": "run", "plan_digest": candidate}}
            response = executor.process(request)
            self.assertTrue(response["ok"])
            self.assertIs(executor.cache[request["op_id"]][1]["data"]["learned_history"], executor.runs["run"]["learned_history"])
            response["data"]["learned_history"].clear()
            self.assertEqual(executor.process(request)["data"]["learned_history"], history)
        # A third chunk exercises the predecessor chain, not only one transition.
        self.assertTrue(job.prepare_next_learned(self.next_raw_program(job, now[0]))["ok"])
        self.assertTrue(job.approve_next_learned({**APPROVAL, "approval_id": "third-exact"})["ok"])
        next_start = job.start_next_learned()
        self.assertTrue(next_start["ok"], next_start["code"])
        self.assertTrue(job.confirm("operator")["ok"])
        now[0] = round(now[0] + .1, 9)
        third = job.poll()
        self.assertEqual(third["state"], "LEARNED_CHUNK_COMPLETE")
        history = third["execution_evidence"]["learned_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(validate_execution_history(job.plan_envelope["plan"], history), history)
        with self.assertRaisesRegex(ContractError, "LEARNED_HISTORY_BINDING"):
            validate_execution_history(job.plan_envelope["plan"], list(reversed(history)))
        self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
        terminal = job.poll()
        self.assertEqual(terminal["code"], "PRECOMMIT_SAFETY")
        diagnostic = learned_run_diagnostic(terminal)
        self.assertEqual(diagnostic["execution_history"], history)
        self.assertEqual(diagnostic["task_effectiveness"], "UNKNOWN")
        self.assertIsNone(diagnostic["episode_ledger"])
        self.assertNotIn(("recorder", "commit"), calls)

    def test_next_held_chunk_native_transport_keeps_completed_reference_without_new_gripper_goal(self):
        job, executor, transport, state, now, sent, _, calls = self.make_held_job()
        self.assertTrue(job.approve(APPROVAL)["ok"])
        self.assertTrue(job.start()["ok"])
        self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", side_effect=lambda: now[0]):
            self.assertTrue(job.confirm("operator")["ok"])
            state["complete"] = True
            job.poll()
            now[0] += .5
            state.update(reference=.01176, feedback=.01218)
            self.assertTrue(job.poll()["ok"])
            now[0] += .51
            state["complete"] = True
            job.poll()
            state.update(joints=[.001] * 6, complete=True)
            self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
            self.assertEqual(len(sent), 3)
            self.assertEqual(transport._gripper_goal_count, 1)
            before = copy.deepcopy(job.plan_envelope)
            old = job._program["learned_proposal"]
            obs = observation()
            obs["observation.state"] = [.001] * 6 + [.01218]
            obs["source_timestamps_s"] = dict.fromkeys(("state", "camera1", "camera2"), now[0])
            actions = [[.002] * 6 + [.01176] for _ in range(4)]
            inference = FinitePolicyInference(lambda _: actions, CHECKPOINT, source_clock=lambda: now[0])
            p = inference.propose(obs, **{**OPTIONS, "robot_description": old["robot_description"],
                "period_s": old["period_s"], "held_gripper_targets": True, "max_observation_age_s": old["max_observation_age_s"]})
            program = compile_program(job._program["source_program"], p)
            result = job.prepare_next_learned(program)
            self.assertTrue(result["ok"], result)
            candidate = result["pending_chunk"]["plan_envelope"]["plan"]
            self.assertEqual([s["type"] for s in candidate["steps"][0]["held_target_segments"]], ["ARM"])
            self.assertEqual(candidate["learned_proposal"]["actions"], actions)
            self.assertTrue(job.approve_next_learned({**APPROVAL, "approval_id": "next-native"})["ok"])
            result = job.start_next_learned()
            self.assertTrue(result["ok"], result)
            self.assertEqual(len(sent), 3)
            self.assertTrue(job.confirm("operator")["ok"])
            self.assertEqual(len(sent), 4)
            self.assertEqual(transport._gripper_goal_count, 1)
            points = sent[-1].trajectory.joint_trajectory.points
            self.assertEqual([list(point.positions) for point in points[1:]], [row[:6] for row in actions])
            now[0] += .2
            state.update(joints=[.002] * 6, complete=True)
            result = job.poll()
            self.assertEqual(result["state"], "LEARNED_CHUNK_COMPLETE")
            history = result["execution_evidence"]["learned_history"]
            self.assertEqual(history[0]["plan_envelope"], before)
            self.assertEqual(result["execution_evidence"]["learned_execution"]["segments"][0]["start_observation"]["snapshot"]["gripper_controller"]["feedback_position_m"], .01218)
            self.assertEqual(calls.count(("recorder", "begin")), 1)
            self.assertNotIn(("recorder", "freeze"), calls)
            self.assertNotIn(("recorder", "commit"), calls)

    def test_foreign_cell_lock_does_not_hold_learned_arming_or_fault_cleanup(self):
        import fcntl
        from tools.data_factory.cell_state import CellStateStore

        for point in ("initial", "fault", "next"):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as directory:
                job, executor, transport, _, scene, _, calls = self.make_job()
                self.assertTrue(job.approve(APPROVAL)["ok"])
                if point != "initial":
                    self.assertTrue(job.start()["ok"])
                    self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
                    self.assertTrue(job.confirm("operator")["ok"])
                if point == "next":
                    self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                    self.assertTrue(job.prepare_next_learned(self.next_raw_program(job))["ok"])
                    self.assertTrue(job.approve_next_learned({**APPROVAL, "approval_id": "next"})["ok"])
                store = CellStateStore(directory, "fr5-lab-a")
                store.mark_blocked("EXECUTION_IN_PROGRESS", "run", job.plan_digest)
                if point == "initial":
                    store.acknowledge_ready("synthetic-operator")
                executor.cell_state_store = store
                path = store.runtime_path("state.json")
                before, sent = path.read_bytes(), len(transport.sent)
                results, errors = [], []
                def work():
                    try:
                        if point == "fault":
                            results.append(executor._fault(executor.runs["run"], "SCENE_STATE_BUSY"))
                        else:
                            results.append(job.start() if point == "initial" else job.start_next_learned())
                    except Exception as exc:
                        errors.append(exc)
                # Independent open descriptions contend on the actual kernel
                # lock; no mocked store/persistence can hide the blocking call.
                with store.runtime_path("state.lock").open("rb") as holder:
                    fcntl.flock(holder, fcntl.LOCK_EX)
                    worker = threading.Thread(target=work, daemon=True)
                    worker.start()
                    try:
                        worker.join(1)
                        returned_while_locked = not worker.is_alive()
                        unchanged_while_locked = path.read_bytes() == before
                    finally:
                        fcntl.flock(holder, fcntl.LOCK_UN)
                        worker.join(2)
                self.assertTrue(returned_while_locked, point)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertTrue(unchanged_while_locked)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(len(transport.sent), sent)
                self.assertNotIn(("recorder", "commit"), calls)
                if point == "fault":
                    run = executor.runs["run"]
                    self.assertEqual(run["state"], "BLOCKED")
                    self.assertTrue(run["cancel_event"].is_set())
                    self.assertFalse(run["execution"]["durable_blocked"])
                    self.assertEqual(run["execution"]["cell_state_error"], "STATE_BUSY")
                    self.assertEqual(transport.cancel_count, 1)
                    self.assertEqual(results, ["SCENE_STATE_BUSY"])
                else:
                    self.assertFalse(results[0]["ok"], results)
                    self.assertEqual(results[0]["code"], "STATE_BUSY")

    def test_initial_scene_lock_does_not_wait_behind_cell_bound_scene_writer(self):
        import fcntl
        from tools.data_factory.cell_state import CellStateStore
        from tools.data_factory.scene_state import SceneStateStore

        with tempfile.TemporaryDirectory() as directory:
            scene = SceneStateStore(directory, "fr5-lab-a")
            scene.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                source="HUMAN", updated_by="synthetic-operator",
                pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": 10., "y_mm": 20.})
            job, executor, transport, _, _, _, calls = self.make_job(scene_store=scene)
            self.assertTrue(job.approve(APPROVAL)["ok"])
            cell = CellStateStore(directory, "fr5-lab-a")
            cell.acknowledge_ready("synthetic-operator")
            executor.cell_state_store = cell
            before = scene.snapshot()
            cell_before = cell.runtime_path("state.json").read_bytes()
            cell_wait = threading.Event()
            lock_calls, writer_errors, results = [], [], []
            acquire = scene._flock
            def observe_lock(descriptor, blocking):
                lock_calls.append(descriptor)
                if len(lock_calls) == 2:
                    cell_wait.set()  # Scene acquired; writer now awaits Cell.
                return acquire(descriptor, blocking)
            scene._flock = observe_lock
            def write_scene():
                try:
                    scene.update_object(instance_id="cube-1", object_profile_id="cube", state="UNKNOWN",
                        source="HUMAN", updated_by="synthetic-operator", expected_cell_digest=canonical_digest("stale"))
                except ContractError as exc:
                    writer_errors.append(exc.code)
            with cell.runtime_path("state.lock").open("rb") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                writer = threading.Thread(target=write_scene, daemon=True)
                starter = threading.Thread(target=lambda: results.append(job.start()), daemon=True)
                writer.start()
                try:
                    self.assertTrue(cell_wait.wait(1))
                    starter.start()
                    starter.join(1)
                    returned_while_locked = not starter.is_alive()
                finally:
                    fcntl.flock(holder, fcntl.LOCK_UN)
                    writer.join(2)
                    if starter.ident is not None:
                        starter.join(2)
            self.assertTrue(returned_while_locked)
            self.assertFalse(writer.is_alive())
            self.assertFalse(starter.is_alive())
            self.assertEqual(writer_errors, ["STATE_CHANGED"])
            self.assertFalse(results[0]["ok"])
            self.assertEqual(results[0]["code"], "SCENE_STATE_BUSY")
            self.assertEqual(scene.snapshot(), before)
            self.assertEqual(cell.runtime_path("state.json").read_bytes(), cell_before)
            self.assertEqual(transport.sent, [])
            self.assertNotIn(("recorder", "commit"), calls)

    def test_next_chunk_deadline_expiry_releases_actual_scene_lock(self):
        import faulthandler
        import os
        import subprocess
        import sys
        from tools.data_factory.cell_state import CellStateStore
        from tools.data_factory.scene_state import SceneStateStore

        case = os.environ.get("FR5_SCENE_LOCK_REPLAY_CASE")
        if case is None:
            for point in ("scene", "cell"):
                for deadline in ("lease", "wait"):
                    with self.subTest(point=point, deadline=deadline):
                        env = dict(os.environ, FR5_SCENE_LOCK_REPLAY_CASE=f"{point}:{deadline}")
                        try:
                            test_id = f"tests.data_factory.rollout.test_finite_plan.{type(self).__name__}.{self._testMethodName}"
                            result = subprocess.run([sys.executable, "-m", "unittest", test_id],
                                env=env, capture_output=True, text=True, timeout=5)
                        except subprocess.TimeoutExpired as exc:
                            self.fail(f"Scene lock deadlock: {point}/{deadline}\n{exc.stderr}")
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return

        point, deadline = case.split(":")
        with tempfile.TemporaryDirectory() as directory:
            scene = SceneStateStore(directory, "fr5-lab-a")
            scene.update_object(instance_id="cube-1", object_profile_id="cube", state="ON_SURFACE",
                source="HUMAN", updated_by="synthetic-operator",
                pose={"place_id": "PLACE_A", "yaw_deg": 0., "x_mm": 10., "y_mm": 20.})
            job, executor, transport, _, _, now, calls = self.make_job(scene_store=scene)
            self.assertTrue(job.approve(APPROVAL)["ok"])
            self.assertTrue(job.start()["ok"])
            self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
            self.assertTrue(job.confirm("operator")["ok"])
            self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
            self.assertTrue(job.prepare_next_learned(self.next_raw_program(job))["ok"])
            self.assertTrue(job.approve_next_learned({**APPROVAL, "approval_id": "next"})["ok"])
            original_digest = job.plan_digest
            cell = CellStateStore(directory, "fr5-lab-a")
            cell.mark_blocked("EXECUTION_IN_PROGRESS", "run", original_digest)
            executor.cell_state_store = cell
            if deadline == "wait":
                executor.runs["run"]["execution"]["wait_deadline"] = now[0] + .1
            delay = 100. if deadline == "lease" else .5
            if point == "scene":
                locked_snapshot = scene.locked_snapshot
                @contextmanager
                def delayed_snapshot(digest, **options):
                    with locked_snapshot(digest, **options) as snapshot:
                        now[0] += delay
                        yield snapshot
                scene.locked_snapshot = delayed_snapshot
            else:
                mark_blocked = cell.mark_blocked
                def delayed_mark(*args, **kwargs):
                    result = mark_blocked(*args, **kwargs)
                    now[0] += delay
                    return result
                cell.mark_blocked = delayed_mark
            process = executor.process
            def checked_process(request):
                result = process(request)
                if request["op"] == "execute_next":
                    # Native expiry handling must finish before returning,
                    # without relying on OneJob's subsequent abort/poll.
                    self.assertEqual(executor.runs["run"]["state"], "BLOCKED")
                    self.assertEqual(scene.snapshot()["scene_state"]["objects"]["cube-1"]["state"], "UNKNOWN")
                return result
            executor.process = checked_process
            faulthandler.dump_traceback_later(1)
            try:
                result = job.start_next_learned()
            finally:
                faulthandler.cancel_dump_traceback_later()
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["code"], "HEARTBEAT_TIMEOUT" if deadline == "lease" else "LEARNED_CHUNK_TIMEOUT")
            self.assertEqual(executor.runs["run"]["digest"], original_digest)
            self.assertEqual(executor.runs["run"]["state"], "BLOCKED")
            self.assertEqual(len(transport.sent), 1)
            self.assertFalse(transport.owns_active_goal)
            self.assertNotIn(("recorder", "commit"), calls)
            self.assertFalse(cell.read()["cell_ready"])
            self.assertEqual(scene.snapshot()["scene_state"]["objects"]["cube-1"]["state"], "UNKNOWN")

    def test_next_chunk_rejects_late_compile_changed_cell_and_superseded_hardware(self):
        for failure in ("late", "paused_source", "reentrant", "cell", "superseded", "wrong_incarnation", "paused_controller", "cell_race", "cancel"):
            with self.subTest(failure=failure):
                job, executor, transport, cell, scene, now, calls = self.start_job()
                self.assertTrue(job.confirm("operator")["ok"])
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                original = copy.deepcopy(job.plan_envelope)
                if failure in {"late", "paused_source", "reentrant"}:
                    if failure == "paused_source":
                        executor.source_clock = lambda: 10.
                    compile_goal = transport.build_learned_trajectory
                    def slow(p):
                        result = compile_goal(p)
                        if failure == "reentrant":
                            executor.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "recursive",
                                "op": "status", "payload": {"run_id": "run", "plan_digest": job.plan_digest}})
                        else:
                            now[0] += .4 if failure == "paused_source" else 100.
                        return result
                    transport.build_learned_trajectory = slow
                result = job.prepare_next_learned(self.next_raw_program(job))
                if failure in {"late", "paused_source", "reentrant"}:
                    self.assertFalse(result["ok"])
                    if failure == "paused_source":
                        self.assertEqual(result["code"], "LEARNED_STALE_OBSERVATION")
                    self.assertEqual(job.plan_envelope, original)
                    self.assertNotIn("pending_chunk", executor.runs["run"])
                else:
                    self.assertTrue(result["ok"], result)
                    self.assertTrue(job.approve_next_learned({**APPROVAL, "approval_id": "next"})["ok"])
                    preserved_cell = None
                    if failure == "cell_race":
                        from tools.data_factory.cell_state import CellStateStore
                        directory = tempfile.TemporaryDirectory()
                        self.addCleanup(directory.cleanup)
                        store = CellStateStore(directory.name, "fr5-lab-a")
                        store.mark_blocked("EXECUTION_IN_PROGRESS", "run", job.plan_digest)
                        executor.cell_state_store = store
                        mark = store.mark_blocked
                        preserved_cell = []
                        def intervene(reason, run_id, plan_digest, **options):
                            if not preserved_cell:
                                mark("FOREIGN_FAILURE", "foreign-run", canonical_digest("foreign-plan"))
                                preserved_cell.append(store.runtime_path("state.json").read_bytes())
                            return mark(reason, run_id, plan_digest, **options)
                        store.mark_blocked = intervene
                    elif failure == "cell":
                        cell.binding["plan_digest"] = canonical_digest("another-plan")
                    elif failure in {"superseded", "wrong_incarnation", "paused_controller"}:
                        observe = transport.snapshot
                        def changed(*args):
                            value = observe(*args)
                            hw = value["gripper_controller"]["hardware_execution"]
                            if failure == "superseded":
                                hw["wire"]["generation"] = 1.
                            elif failure == "wrong_incarnation":
                                hw["wire"]["incarnation_0"] = 2.
                                hw["clock_binding"]["incarnation"][0] = 2
                            else:
                                value["arm_controller"]["speed_scaling"] = 0.
                            return value
                        transport.snapshot = changed
                    else:
                        executor.runs["run"]["cancel_event"].set()
                    rejected = job.start_next_learned()
                    self.assertFalse(rejected["ok"])
                    if failure == "cell_race":
                        self.assertEqual(rejected["code"], "STATE_CHANGED")
                        self.assertEqual(store.runtime_path("state.json").read_bytes(), preserved_cell[0])
                    if failure == "cell":
                        self.assertEqual(cell.binding["plan_digest"], canonical_digest("another-plan"))
                    self.assertEqual(job.plan_envelope, original)
                self.assertEqual(len(transport.sent), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_retried_command_cannot_bypass_existing_lease_tick(self):
        job, executor, transport, _, _, now, calls = self.start_job()
        request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "same-confirm",
                   "op": "confirm", "payload": {"run_id": "run", "plan_digest": job.plan_digest,
                   "confirmed_by": "operator", "source": "HUMAN"}}
        # Repeated operation keeps its receipt but must not prevent the sole
        # owner from enforcing the elapsed lease.
        response = executor.process(request)
        self.assertTrue(response["ok"], response)
        now[0] += 100.
        self.assertEqual(executor.process(request), response)
        self.assertEqual(executor.runs["run"]["failure_code"], "HEARTBEAT_TIMEOUT")
        self.assertEqual(transport.cancel_count, 1)
        self.assertEqual(len(transport.sent), 1)

    def test_chunk_observation_rejects_binding_active_stale_cancel_and_expired_lease(self):
        for failure in ("wrong_plan", "wrong_lease", "active", "stale", "cancel", "expired", "boundary_timeout", "loosened_age"):
            with self.subTest(failure=failure):
                job, executor, transport, _, _, now, calls = self.start_job()
                self.assertTrue(job.confirm("operator")["ok"])
                self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
                captured = []
                def capture(*_):
                    captured.append(True)
                    value = observation()
                    if failure == "stale":
                        value["source_timestamps_s"]["camera1"] -= 1.
                    elif failure == "cancel":
                        executor._fault(executor.runs["run"], "CANCELLED_BY_OPERATOR")
                    elif failure == "expired":
                        now[0] += 100.
                    elif failure == "boundary_timeout":
                        run = executor.runs["run"]
                        now[0] = run["execution"]["wait_deadline"] + .01
                        run["execution"]["lease_deadline"] = now[0] + 1.
                    return value
                transport.capture_policy_observation = capture
                if failure in {"wrong_plan", "wrong_lease"}:
                    request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "bad-observation", "op": "capture_observation", "payload": {
                        "run_id": "run", "plan_digest": canonical_digest("wrong") if failure == "wrong_plan" else job.plan_digest,
                        "lease_id": "wrong" if failure == "wrong_lease" else job.lease_id,
                        "camera_topics": {"camera1": "/up", "camera2": "/wrist"}, "max_observation_age_s": .3}}
                    result = executor.process(json.loads(json.dumps(request)))
                    self.assertFalse(captured)
                else:
                    transport.active = failure == "active"
                    result = job.observe_learned_boundary({"camera1": "/up", "camera2": "/wrist"},
                        max_observation_age_s=1. if failure == "loosened_age" else .3)
                    if failure in {"active", "loosened_age"}:
                        self.assertFalse(captured)
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["code"], {
                    "wrong_plan": "PLAN_DIGEST_MISMATCH", "wrong_lease": "LEASE_BINDING", "active": "ROS_EXEC_ACTIVE",
                    "stale": "LEARNED_STALE_OBSERVATION", "cancel": "CANCELLED_BY_OPERATOR", "expired": "HEARTBEAT_TIMEOUT",
                    "boundary_timeout": "LEARNED_CHUNK_TIMEOUT", "loosened_age": "LEARNED_STALE_OBSERVATION"}[failure])
                self.assertEqual(len(transport.sent), 1)
                self.assertNotIn(("recorder", "commit"), calls)

    def test_completed_probe_flows_to_diagnostic_and_cannot_commit_pending_safety(self):
        job, executor, transport, cell, scene, _, calls = self.start_job()
        self.assertTrue(job.confirm("operator")["ok"])
        self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
        self.assertEqual(job.recorder_state, "RECORDING")
        self.assertTrue(job.semantic_verdict("PASS", "operator")["ok"])
        result = job.poll()
        self.assertEqual((result["code"], result["recorder_state"]), ("PRECOMMIT_SAFETY", "QUARANTINED_COMMIT"))
        self.assertEqual(executor.runs["run"]["state"], "COMPLETED")
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(transport.sent[0]["learned_proposal"]["actions"], [ACTION])
        self.assertFalse(cell.ready)
        self.assertEqual([entry["state"] for entry in scene.updates], ["UNKNOWN"])
        self.assertNotIn(("recorder", "commit"), calls)
        self.assertLess(calls.index(("recorder", "begin")), calls.index(("executor", "execute")))
        diagnostic = learned_run_diagnostic(result)
        self.assertEqual(diagnostic["execution_trace"]["status"], "COMPLETED")
        self.assertEqual(diagnostic["task_effectiveness"], "UNKNOWN")
        self.assertEqual(diagnostic["human_semantic_decision"]["review_scope"], "FINITE_LEARNED_CHUNK")
        self.assertIsNone(diagnostic["episode_ledger"])
        self.assertFalse(diagnostic["training_authorized"])
        tampered = copy.deepcopy(result)
        tampered["plan_envelope"]["plan"]["learned_proposal"]["actions"][0][0] += .001
        with self.assertRaises(ContractError):
            learned_run_diagnostic(tampered)

    def test_approval_delay_uses_current_state_without_renewing_inference_sources(self):
        job, _, transport, _, _, now, _ = self.start_job()
        now[0] += .4
        result = job.confirm("operator")
        self.assertTrue(result["ok"], result)
        self.assertEqual(len(transport.sent), 1)
        self.assertEqual(transport.sent[0]["learned_proposal"]["source_timestamps_s"], dict.fromkeys(("state", "camera1", "camera2"), 10.))

    def test_stale_state_after_confirmation_never_activates_or_cancels_a_goal(self):
        job, executor, transport, _, _, now, _ = self.start_job()
        now[0] += .4
        observe = transport.snapshot
        def stale(*args):
            value = observe(*args)
            value["joint_state_stamp_ns"] = 8_000_000_000
            return value
        transport.snapshot = stale
        self.assertEqual(job.confirm("operator")["code"], "LEARNED_STALE_STATE")
        self.assertEqual(transport.sent, [])
        self.assertEqual(transport.cancel_count, 0)
        self.assertFalse(executor.runs["run"]["execution"]["active"])

    def test_delayed_approved_full_chunk_retains_current_start_in_canonical_diagnostic(self):
        job, executor, transport, _, _, now, _ = self.make_job()
        self.assertTrue(job.approve(APPROVAL)["ok"])
        now[0] = 20.  # Existing approval remains valid; input evidence stays at 10.
        job.start()
        self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN")
        now[0] += .4
        self.assertTrue(job.confirm("operator")["ok"])
        self.assertEqual(job.poll()["state"], "LEARNED_CHUNK_COMPLETE")
        job.semantic_verdict("PASS", "operator")
        result = job.poll()
        diagnostic = learned_run_diagnostic(result)
        trace = diagnostic["execution_trace"]
        self.assertEqual(trace["start_observation"]["captured_at_s"], 20.4)
        self.assertEqual(trace["start_observation"]["snapshot"]["joint_state_stamp_ns"], 20400000000)
        self.assertEqual(trace["task_effectiveness"], "UNKNOWN")
        self.assertEqual(transport.sent[0]["learned_proposal"]["actions"], [ACTION])
        self.assertEqual(transport.sent[0]["learned_proposal"]["source_timestamps_s"]["camera1"], 10.)
        tampered = copy.deepcopy(trace)
        tampered["start_observation"]["snapshot"]["joint_state_stamp_ns"] = 10_000_000_000
        tampered["trace_digest"] = canonical_digest({k: v for k, v in tampered.items() if k != "trace_digest"})
        from tools.data_factory.rollout.finite_plan import validate_execution_trace
        with self.assertRaisesRegex(ContractError, "LEARNED_STALE_STATE"):
            validate_execution_trace(executor.runs["run"]["plan"], tampered)

    def test_new_execution_state_rejects_stale_rebound_superseded_or_changed_inputs(self):
        cases = ["old_header", "old_controller", "future_header", "old_receipt", "paused", "rebound", "superseded", "joint_limit", "start", "unbound"]
        for case in cases:
            with self.subTest(case=case):
                job, _, transport, _, _, now, _ = self.make_job(hardware=case != "unbound")
                job.approve(APPROVAL)
                now[0] += 10.
                observe = transport.snapshot
                def invalid(*args):
                    value = observe(*args)
                    if case == "old_header": value["joint_state_stamp_ns"] = 10_000_000_000
                    if case == "old_controller": value["arm_controller"]["sample"]["ros_stamp_ns"] = 10_000_000_000
                    if case == "future_header": value["joint_state_stamp_ns"] = 21_000_000_000
                    if case == "old_receipt": value["joint_state_age_s"] = 10.
                    if case == "paused": value["arm_controller"]["speed_scaling"] = 0.
                    if case in {"rebound", "superseded"}:
                        hw = value["gripper_controller"]["hardware_execution"]
                        if case == "rebound":
                            hw["wire"]["incarnation_0"] = hw["clock_binding"]["incarnation"][0] = 5
                        else:
                            hw["wire"].update(generation=1., completed_generation=1., command_started_system_s=1., completion_reason=2.,
                                              completion_year=1970., completion_month=1., completion_day=1., completion_second=2.)
                    if case in {"joint_limit", "start"}:
                        value["joint_positions"][0] = 4. if case == "joint_limit" else .1
                    return value
                transport.snapshot = invalid
                result = job.start()
                expected = {"old_header": "LEARNED_STALE_STATE", "old_controller": "LEARNED_STALE_STATE", "future_header": "LEARNED_STALE_STATE",
                            "old_receipt": "LEARNED_STALE_STATE", "paused": "LEARNED_CONTROLLER_PAUSED",
                            "rebound": "LEARNED_HARDWARE_INCARNATION", "superseded": "LEARNED_HARDWARE_SUPERSEDED",
                            "joint_limit": "LEARNED_JOINT_LIMIT", "start": "START_STATE_MISMATCH", "unbound": "LEARNED_HARDWARE_UNBOUND"}[case]
                self.assertEqual(result["code"], expected, result)
                self.assertEqual(transport.sent, [])

    def test_controller_fault_uses_existing_cancel_owner_and_failure_diagnostic(self):
        job, _, transport, cell, scene, _, calls = self.start_job()
        self.assertTrue(job.confirm("operator")["ok"])
        transport.failure = "ROS_EXEC_RESULT_TIMEOUT"
        result = job.poll()
        self.assertEqual(result["code"], "ROS_EXEC_RESULT_TIMEOUT")
        self.assertEqual((len(transport.sent), transport.cancel_count), (1, 1))
        self.assertEqual(learned_run_diagnostic(result)["execution_trace"]["status"], "FAILED")
        job.cancel()
        self.assertEqual(transport.cancel_count, 1)
        self.assertFalse(cell.ready)
        self.assertNotIn(("recorder", "commit"), calls)

    def test_failed_learned_job_retains_native_payload_after_motion_stops(self):
        from tests.test_recorder_transaction import RecorderTransactionTest, dataset_snapshot
        from tools.fr5_lerobot_recorder import process_recorder_control_line
        for fault in (None, "cancel_uncertain", "retention_response", "release_claim"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                job, _, transport, cell, _, _, calls = self.start_job()
                self.assertTrue(job.confirm("operator")["ok"])
                transport.poll_active = lambda: None  # Keep the goal active until cancel.
                recorder = RecorderTransactionTest().retention_fixture(directory, run_id="run")
                before = dataset_snapshot(recorder.args.root)
                preserved = []

                class RecorderPort:
                    def __call__(self, request):
                        raise AssertionError("terminal retention must not use motion request timeout")

                    def request_terminal(self, request):
                        calls.append(("recorder", request["op"]))
                        if fault == "retention_response":
                            raise ContractError("JSONL_PROCESS_EXIT")
                        response = process_recorder_control_line(recorder, json.dumps(request), {})
                        if fault == "release_claim":
                            response = copy.deepcopy(response)
                            response["retention"]["partial"] = True
                        return response

                    def preserve(self):
                        preserved.append(True)

                job.recorder_call = RecorderPort()
                job.transaction_id = recorder._transaction["transaction_id"]
                job.episode_index = recorder._transaction["episode_index"]
                job.recorder_state = recorder.episode_state
                cancel = transport.cancel_active

                def observed_cancel(*args):
                    calls.append(("motion", "cancel"))
                    if fault == "cancel_uncertain":
                        raise ContractError("CANCEL_UNCONFIRMED")
                    return cancel(*args)

                transport.cancel_active = observed_cancel
                result = job.cancel()
                self.assertFalse(result["ok"])
                self.assertNotIn(("recorder", "abort"), calls)
                self.assertNotIn(("recorder", "commit"), calls)
                self.assertEqual(dataset_snapshot(recorder.args.root), before)
                if fault == "cancel_uncertain":
                    self.assertTrue(preserved)
                    self.assertNotIn(("recorder", "retain"), calls)
                    self.assertEqual(result["state"], "BLOCKED")
                else:
                    self.assertLess(calls.index(("motion", "cancel")), calls.index(("recorder", "retain")))
                    if fault in {"retention_response", "release_claim"}:
                        self.assertTrue(preserved)
                        self.assertIn("retention_error", result)
                        self.assertEqual(result["state"], "BLOCKED")
                        if fault == "release_claim":
                            self.assertEqual(result["retention_error"], "RECORDER_RETENTION_RESPONSE")
                    else:
                        receipt = result["recorder_evidence"]["retention"]
                        self.assertEqual(receipt["disposition"], "cancel")
                        self.assertEqual(receipt["rows"], 2)
                        self.assertTrue(receipt["durable"])
                        self.assertFalse(receipt["training_eligible"])
                        # Releasing recorder staging is not permission to resume
                        # motion or claim that this failed attempt succeeded.
                        self.assertEqual(result["state"], "ABORTED")
                        self.assertEqual(result["recorder_state"], "ABORTED")
                        self.assertEqual(receipt["staging_state"], "RELEASED")
                        self.assertFalse(receipt["quarantined"])
                        self.assertFalse(cell.ready)
                        self.assertTrue(list(Path(receipt["destination"]).rglob("*.parquet")))
                        self.assertFalse(learned_run_diagnostic(result)["training_authorized"])
                        count = len(calls)
                        job.cancel()
                        self.assertEqual(len(calls), count)
                        next_recorder = RecorderTransactionTest().make_recorder(directory)
                        next_recorder.dataset = recorder.dataset
                        begin = {"schema_version": "data_factory.recorder_command.v1", "op_id": "begin-next",
                                 "op": "begin", "transaction": {
                                     **RecorderTransactionTest().transaction(directory), "run_id": "next-run"}}
                        started = process_recorder_control_line(next_recorder, json.dumps(begin), {})
                        self.assertTrue(started["ok"], started)
                        self.assertTrue(next_recorder.abort_episode()["ok"])
                        self.assertEqual(dataset_snapshot(recorder.args.root), before)
                recorder._release_transaction_lock()

    def test_recursive_executor_command_during_send_cannot_dispatch_again(self):
        job, executor, transport, _, _, _, _ = self.start_job()
        responses = []
        transport.on_start = lambda: responses.append(executor.process({"schema_version": "fr5.pickup_executor.command.v4",
            "op_id": "recursive", "op": "status", "payload": {"run_id": "run", "plan_digest": job.plan_digest}}))
        result = job.confirm("operator")
        self.assertFalse(result["ok"])
        self.assertEqual(responses[0]["code"], "REENTRANT_COMMAND")
        self.assertEqual((len(transport.sent), transport.cancel_count), (1, 1))


if __name__ == '__main__':
    unittest.main()
