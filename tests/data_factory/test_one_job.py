import copy
import json
import sys
import threading
import time
import unittest
from unittest.mock import patch

from tools.data_factory.one_job import (
    JsonlProcess, OneJob, TEST_ONLY_READINESS_CONTRACT, hil_numeric_gripper_verdict, run_one_job,
)
from tools.data_factory.scene_state import release_slot
from tools.fr5_data_factory import ContractError, canonical_digest


PROGRAM = {"schema_version": "fr5.motion_program.v2"}
PHASES = ("PREGRASP_PTP","APPROACH_STOP_LIN","FINAL_APPROACH_LIN","GRIPPER_CLOSE","LIFT_LIN","RECYCLE_APPROACH_PTP","LOWER_LIN","GRIPPER_OPEN","RETREAT_LIN","SAFE_POSE_PTP")
BINDINGS = {key: "sha256:" + "c" * 64 for key in ("selected_sheet", "yaw0_sheet", "cell_calibration", "robot_system", "collection_profile", "object_profile", "grasp_profile", "planning_scene_digest")}
RESOLVED = "sha256:" + "b" * 64
SETUP_APPROVAL = {"source":"HUMAN", "approval_id":"setup-1", "approved_by":"operator", "approval_expiry":"2099-01-01T00:00:00Z", "resolved_job_digest":RESOLVED}
SCENE = {"scene_state_digest":"sha256:" + "8" * 64, "revision":1, "object_instance_id":"cube-1"}
RELEASE_SCENE = {**SCENE, "release_slot": release_slot(
    robot_system_id="fr5-lab-a",
    pose={"place_id":"place-a", "yaw_deg":0, "x_mm":60, "y_mm":0},
    object_profile_id="wood-cube-25mm-r001",
    exclusion_geometry_digest="sha256:" + "e" * 64,
)}
PLAN = {"run_id":"run", "motion_program":PROGRAM, "scene_binding":SCENE, "setup_approval":SETUP_APPROVAL}
RELEASE_PLAN = {**PLAN, "scene_binding":RELEASE_SCENE}
MOTION_APPROVAL = {"source":"HUMAN", "approval_id":"a", "approved_by":"operator", "approval_expiry":"2099-01-01T00:00:00Z", "approval_scope":"HUMAN_GATED"}


class StreamRevisionRequestTest(unittest.TestCase):
    def owner(self, *, ok=True, code="STREAM_REVISION_CHECKING"):
        from tools.data_factory.rollout.stream_plan import PLAN_SCHEMA
        calls = []

        def executor(request):
            calls.append(copy.deepcopy(request))
            # Model a callee mutating its own detached request, never the caller's
            # retained selected native rows.
            if "selection" in request["payload"]:
                request["payload"]["selection"]["rows"][0] = 999
            return {"schema_version": "fr5.pickup_executor.response.v3",
                    "mode": "PRE_LIVE", "op_id": request["op_id"], "op": request["op"],
                    "ok": ok, "code": code, "run_id": "run", "plan_digest": RESOLVED,
                    "state": "EXECUTING", "data": {"native_stream": {"status": code},
                    "native_revisions": [], "task_outcome": "UNKNOWN"}}

        def no_effect(_):
            raise AssertionError("Revision request must not touch recorder or Cell")

        job = OneJob(no_effect, executor, no_effect)
        job.state, job.approval_scope = "EXECUTING", "SCOPED_TASK_GRANT"
        job.run_id, job.plan_digest, job.lease_id = "run", RESOLVED, "lease"
        job.plan_envelope = {"plan": {"schema_version": PLAN_SCHEMA}}
        return job, calls

    def test_prepare_and_commit_use_same_bound_owner_without_ack_or_recorder_effect(self):
        job, calls = self.owner()
        selection = {"rows": [1]}
        self.assertTrue(job.prepare_stream_revision(selection)["ok"])
        self.assertEqual(selection, {"rows": [1]})
        epoch = 123_456_789_012
        self.assertTrue(job.commit_stream_revision(RESOLVED, epoch)["ok"])
        self.assertEqual([call["op"] for call in calls],
                         ["prepare_stream_revision", "commit_stream_revision"])
        for call in calls:
            self.assertEqual({key: call["payload"][key] for key in
                              ("run_id", "plan_digest", "lease_id")},
                             {"run_id": "run", "plan_digest": RESOLVED, "lease_id": "lease"})
        self.assertEqual(calls[-1]["payload"]["start_time_ns"], epoch)
        self.assertEqual(job.execution_evidence["native_revisions"], [])
        self.assertEqual(job.execution_evidence["task_outcome"], "UNKNOWN")

    def test_candidate_rejection_does_not_abort_current_task(self):
        job, calls = self.owner(ok=False, code="START_STATE_MISMATCH")
        result = job.commit_stream_revision(RESOLVED, 123_456_789_012)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "START_STATE_MISMATCH")
        self.assertEqual(job.state, "EXECUTING")
        self.assertEqual(len(calls), 1)

    def test_not_executing_normal_scoped_task_never_reaches_owner(self):
        for field, value in (("state", "PLANNED"), ("approval_scope", "HUMAN_GATED"),
                             ("plan_envelope", {"plan": {"schema_version": "fr5.pickup_plan.v3"}})):
            with self.subTest(field=field):
                job, calls = self.owner()
                setattr(job, field, value)
                self.assertEqual(job.prepare_stream_revision({})["code"], "TASK_GRANT_STATE")
                self.assertEqual(job.commit_stream_revision(RESOLVED, 10)["code"], "TASK_GRANT_STATE")
                self.assertEqual(calls, [])


class OneJobTest(unittest.TestCase):
    def make(self, recorder_states, executor_states, *, first_row_rows=1, continuous=False, release=False,
             readiness_contract=None, status_mutation=None, quality_mutation=None,
             allow_synthetic_test_operator=False):
        calls = []
        scene = RELEASE_SCENE if release else SCENE
        steps = [{"phase":phase} for phase in PHASES]
        if not continuous:
            steps[2]["requires_confirmation"] = "PRECONTACT_HUMAN"
            steps[3]["pause_after"] = "GRASP_VERDICT"
        steps[4]["pause_after"] = "SEMANTIC_VERDICT"
        normalized = {**PROGRAM, "robot_system_id":"fr5-lab-a", "resolved_job_digest":RESOLVED, "binding_digests":BINDINGS, "execution_timeouts_s":{"heartbeat_lease":1,"semantic_verdict":30}, "gripper_requirements":{"command_position_m":.01,"acceptable_feedback_m":{"min":.01,"max":.012}}, "steps":steps}
        planned = {"schema_version":"fr5.pickup_plan.v3", "run_id":"run", "scene_binding":scene, "motion_program_digest":canonical_digest(normalized), "resolved_job_digest":RESOLVED, "binding_digests":BINDINGS, "steps":[{"phase":phase, "start_joint_state":[0]*6, "final_joint_state":[0]*6} for phase in PHASES]}
        plan_digest = canonical_digest(planned)
        readback = {"schema_version":"data_factory.planning_scene_readback.v1", "run_id":"run", "plan_digest":plan_digest, "expected_planning_scene_digest":BINDINGS["planning_scene_digest"], "objects":[]}
        collision = {"schema_version":"data_factory.collision_report.v1", "plan_digest":plan_digest, "sample_count":0, "samples":[], "failure_count":0, "all_valid":True}
        no_motion = {"schema_version":"data_factory.plan_only_no_motion.v1", "run_id":"run", "plan_digest":plan_digest, "before_snapshot":{}, "after_snapshot":{}, "max_joint_delta_rad":0., "gripper_delta_m":0., "execute_goal_count":0, "gripper_goal_count":0}
        safety = {"schema_version":"data_factory.precommit_safety.v1", "run_id":"run", "approved_plan_digest":plan_digest, "scene_binding_digest":canonical_digest(scene), "expected_planning_scene_digest":BINDINGS["planning_scene_digest"], "planning_scene_readback_digest":canonical_digest(readback), "collision_report_digest":canonical_digest(collision), "plan_only_no_motion_digest":canonical_digest(no_motion), "post_reset_safe_snapshot_digest":None, "status":"PENDING"}
        evidence = {"schema_version":"data_factory.precommit_evidence.v1", "run_id":"run", "approved_plan_digest":plan_digest, "scene_binding_digest":canonical_digest(scene), "expected_planning_scene_digest":BINDINGS["planning_scene_digest"], "planning_scene_readback":readback, "collision_report":collision, "plan_only_no_motion":no_motion}
        envelope = {"plan":planned, "precommit_safety":safety, "precommit_evidence":evidence, "operator_summary":{}}
        first_status_after_begin = False
        last_readiness_rows = first_row_rows
        def recorder(request):
            nonlocal first_status_after_begin, last_readiness_rows
            calls.append(("recorder", request["op"]))
            if request["op"] == "begin":
                first_status_after_begin = True
            if request["op"] == "trim_readiness_prefix":
                if readiness_contract != TEST_ONLY_READINESS_CONTRACT:
                    raise AssertionError("production recorder must not trim readiness prefix")
                item = "RECORDING"
                rows = 0
            elif request["op"] == "status" and first_status_after_begin:
                first_status_after_begin = False
                item = "RECORDING"
                rows = first_row_rows
            else:
                item = recorder_states.pop(0)
                rows = 1
            if isinstance(item, dict): return item
            rejected = item == "QUALITY_REJECTED"
            metrics = {
                "rows":rows, "writer_queue":0, "writer_queue_drops":0, "alignment_failures":0,
                "observed_monotonic_ns":time.monotonic_ns(),
            }
            response = {"schema_version":"data_factory.recorder_response.v1", "op_id":request["op_id"], "op":request["op"], "ok":not rejected, "state":"FROZEN" if rejected else item, "reason_code":item, "run_id":"run", "transaction_id":"tx", "episode_index":0, "metrics":metrics, "artifacts":{}, "detail":"", "writer_alive":True, "writer_error":None, "sampler_alive":True}
            if request["op"] == "status" and status_mutation is not None:
                status_mutation(response)
            if request["op"] == "status":
                last_readiness_rows = response["metrics"]["rows"]
            if request["op"] == "trim_readiness_prefix":
                quality_reasons = [] if last_readiness_rows >= 60 else [
                    f"frames {last_readiness_rows} < minimum 60"
                ]
                quality = {
                    "accepted":not quality_reasons, "reasons":quality_reasons,
                    "frames":last_readiness_rows, "target_fps":30, "effective_fps":30.0,
                    "cameras":{"up":{"source_fps":30.0, "brightness_mean":120.0}}, "writer_queue_drops":0,
                    "alignment_failures":0, "image_quality_warnings":[],
                }
                if quality_mutation is not None:
                    quality_mutation(quality)
                response["quality"] = quality
                if quality["accepted"]:
                    response["reason_code"] = "READINESS_PREFIX_TRIMMED"
                else:
                    response.update(ok=False, reason_code="READINESS_PREFIX_UNSAFE")
                    response["metrics"]["rows"] = last_readiness_rows
            return response
        recorder.preserve = lambda: calls.append(("recorder", "preserve"))
        def executor(request):
            calls.append(("executor", request["op"])); state = executor_states.pop(0)
            if isinstance(state, dict): return state
            if state == "COMPLETED": calls.append(("executor_completed",))
            data = envelope if state == "PLANNED" else {"durable_blocked":True} if state == "BLOCKED" else {"gripper_feedback_m":.011,"gripper_reference_m":.01} if state == "GRASP_VERDICT" else {"gripper_feedback_m":.011,"gripper_reference_m":.01,"post_lift_gripper_feedback_m":.011} if state == "SEMANTIC_VERDICT" else {"precommit_safety":{**safety, "post_reset_safe_snapshot_digest":"sha256:" + "4" * 64, "status":"PASS"}} if state == "COMPLETED" else None
            return {"schema_version":"fr5.pickup_executor.response.v3", "mode":"PRE_LIVE", "op_id":request["op_id"], "op":request["op"], "ok":state != "BLOCKED", "code":state, "run_id":"run", "plan_digest":plan_digest, "state":state, "data":data}
        patcher = patch("tools.data_factory.one_job.validate_motion_program", lambda _: normalized)
        patcher.start(); self.addCleanup(patcher.stop)
        job = None
        def cell():
            return {"robot_system_id":"fr5-lab-a", "cell_ready":True, "reason_code":"HUMAN_ACKNOWLEDGED", "run_id":job.run_id, "plan_digest":job.plan_digest, "acknowledged_by":"operator"}
        job = OneJob(
            recorder, executor, cell, readiness_contract=readiness_contract,
            allow_synthetic_test_operator=allow_synthetic_test_operator,
        )
        return job, calls

    @staticmethod
    def prepare_and_start(job):
        job.prepare(PLAN)
        job.approve(MOTION_APPROVAL)
        job.start()

    def test_happy_pass_orders_begin_before_execute_and_commits(self):
        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
        job.set_lifecycle_event_call(
            lambda code: calls.append(("lifecycle", code)),
        )
        self.prepare_and_start(job); job.poll(); self.assertEqual(job.poll()["state"], "PRECONTACT_HUMAN"); job.confirm("operator"); job.poll(); job.grasp_verdict("PASS", "operator"); job.poll(); job.semantic_verdict("PASS", "operator")
        self.assertEqual(job.poll()["state"], "AWAITING_CELL_READY")
        self.assertEqual(job.finish()["state"], "COMPLETE")
        self.assertLess(calls.index(("recorder", "begin")), calls.index(("executor", "execute")))
        self.assertLess(calls.index(("recorder", "freeze")), calls.index(("executor", "semantic_verdict")))
        self.assertLess(calls.index(("lifecycle", "MOTION_STARTING")), calls.index(("executor", "execute")))
        self.assertLess(calls.index(("lifecycle", "RECYCLING")), calls.index(("executor", "semantic_verdict")))
        self.assertLess(calls.index(("lifecycle", "FINALIZING")), calls.index(("recorder", "commit")))

    def test_continuous_plan_needs_only_post_lift_semantic_verdict(self):
        job, calls = self.make(
            ["RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"],
            ["PLANNED", "APPROVED", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"],
            continuous=True,
        )
        self.prepare_and_start(job)
        result = job.poll()
        self.assertEqual((result["state"], result["grasp_verdict"]), ("SEMANTIC_VERDICT", None))
        job.semantic_verdict("PASS", "operator")
        self.assertEqual(job.poll()["state"], "AWAITING_CELL_READY")
        self.assertNotIn(("executor", "confirm"), calls)
        self.assertNotIn(("executor", "grasp_verdict"), calls)
        self.assertLess(calls.index(("recorder", "freeze")), calls.index(("executor", "semantic_verdict")))

    def test_committed_block_is_terminal_without_cell_ack_or_data_abort(self):
        job, calls = self.make(
            ["RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"],
            ["PLANNED", "APPROVED", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"],
            continuous=True,
        )
        self.prepare_and_start(job)
        job.poll()
        job.semantic_verdict("PASS", "operator")
        self.assertEqual(job.poll()["state"], "AWAITING_CELL_READY")
        before = list(calls)
        result = job.block_committed("ROS_GRIPPER_SETTINGS_UNVERIFIED")
        self.assertEqual((result["ok"], result["state"]), (False, "COMMITTED_BLOCKED"))
        self.assertEqual((result["recorder_state"], result["executor_state"]), ("COMMITTED", "COMPLETED"))
        self.assertEqual(job.finish()["code"], "CELL_READY_STATE")
        self.assertEqual(job.cancel()["code"], "CANCEL_STATE")
        self.assertEqual(calls, before)

    def test_committed_block_cannot_settle_an_unfinished_or_uncertain_owner(self):
        for recorder, executor, cancel_error in (
            ("RECORDING", "COMPLETED", None), ("COMMITTED", "EXECUTING", None),
            ("QUARANTINED_COMMIT", "COMPLETED", None),
            ("COMMITTED", "COMPLETED", "CANCEL_UNCONFIRMED"),
        ):
            with self.subTest(recorder=recorder, executor=executor, cancel_error=cancel_error):
                job, calls = self.make([], [])
                job.state = "AWAITING_CELL_READY"
                job.recorder_state, job.executor_state, job.cancel_error = recorder, executor, cancel_error
                self.assertEqual(job.block_committed("FAULT")["code"], "POSTCOMMIT_BLOCK_STATE")
                self.assertEqual(job.state, "AWAITING_CELL_READY")
                self.assertEqual(calls, [])

    def test_recycle_stays_frozen_until_release_transition_then_commits(self):
        job, calls = self.make(
            ["RECORDING", "RECORDING", "FROZEN", "FROZEN", "FROZEN", "COMMITTED"],
            ["PLANNED", "APPROVED", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "RELEASE_VERDICT", "COMPLETED", "COMPLETED"],
            continuous=True,
            release=True,
        )
        job.prepare(RELEASE_PLAN); job.approve(MOTION_APPROVAL); job.start()
        self.assertEqual(job.poll()["state"], "SEMANTIC_VERDICT")
        job.semantic_verdict("PASS", "operator")
        self.assertEqual(job.poll()["state"], "RELEASE_VERDICT")
        self.assertEqual((job.frozen_rows, job.rows_after_recycle), (1, 1))
        self.assertTrue(job.release_verdict("LANDED", "operator")["ok"])
        self.assertEqual(job.poll()["state"], "AWAITING_CELL_READY")
        self.assertEqual(job.finish()["state"], "COMPLETE")
        self.assertLess(calls.index(("executor", "release_verdict")), calls.index(("recorder", "commit")))

    def test_test_operator_release_and_cell_ack_are_test_only(self):
        local_approval = {
            **MOTION_APPROVAL,
            "source": "LOCAL_UI_BUTTON",
            "approval_scope": "HIL_NUMERIC_PROXY",
        }
        job, calls = self.make(
            ["RECORDING", "RECORDING", "FROZEN", "FROZEN", "FROZEN", "COMMITTED"],
            ["PLANNED", "APPROVED", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "RELEASE_VERDICT", "COMPLETED", "COMPLETED"],
            first_row_rows=60,
            continuous=True,
            release=True,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT,
            allow_synthetic_test_operator=True,
        )
        job.prepare(RELEASE_PLAN); job.approve(local_approval); job.start()
        self.assertEqual(job.poll()["state"], "SEMANTIC_VERDICT")
        self.assertTrue(job.semantic_verdict("PASS", "test-operator", source="HIL_PROXY")["ok"])
        self.assertEqual(job.poll()["state"], "RELEASE_VERDICT")
        self.assertTrue(job.release_verdict("LANDED", "test-operator", source="TEST_OPERATOR")["ok"])
        self.assertEqual(job.poll()["state"], "AWAITING_CELL_READY")
        job.cell_state_call = lambda: {
            "robot_system_id": "fr5-lab-a", "cell_ready": True,
            "reason_code": "TEST_OPERATOR_ACKNOWLEDGED", "run_id": job.run_id,
            "plan_digest": job.plan_digest, "acknowledged_by": "test-operator",
        }
        self.assertEqual(job.finish()["state"], "COMPLETE")
        self.assertLess(calls.index(("executor", "release_verdict")), calls.index(("recorder", "commit")))

        production, production_calls = self.make([], [])
        production.state = "RELEASE_VERDICT"
        self.assertEqual(
            production.release_verdict("LANDED", "test-operator", source="TEST_OPERATOR")["code"],
            "RELEASE_VERDICT_SOURCE",
        )
        production.state = "AWAITING_CELL_READY"
        production._program = {"robot_system_id": "fr5-lab-a"}
        production.run_id, production.plan_digest = "run", "sha256:" + "1" * 64
        production.cell_state_call = lambda: {
            "robot_system_id": "fr5-lab-a", "cell_ready": True,
            "reason_code": "TEST_OPERATOR_ACKNOWLEDGED", "run_id": production.run_id,
            "plan_digest": production.plan_digest, "acknowledged_by": "test-operator",
        }
        self.assertEqual(production.finish()["code"], "CELL_READY_REQUIRED")
        self.assertEqual(production_calls, [])

        readiness_only, readiness_calls = self.make(
            [], [], readiness_contract=TEST_ONLY_READINESS_CONTRACT,
        )
        readiness_only.state = "RELEASE_VERDICT"
        self.assertEqual(
            readiness_only.release_verdict("LANDED", "test-operator", source="TEST_OPERATOR")["code"],
            "RELEASE_VERDICT_SOURCE",
        )
        self.assertEqual(readiness_calls, [])
        with self.assertRaisesRegex(ContractError, "TEST_OPERATOR_SCOPE"):
            OneJob(lambda _: None, lambda _: None, allow_synthetic_test_operator=True)

    def test_first_row_timeout_never_executes(self):
        job, calls = self.make(["RECORDING", "ABORTED"], ["PLANNED", "APPROVED"], first_row_rows=0)
        ticks = iter((0.0, 6.0))
        job.monotonic_clock = lambda: next(ticks)
        job.prepare(PLAN); job.approve(MOTION_APPROVAL)
        result = job.start()
        self.assertEqual((result["state"], result["code"]), ("ABORTED", "RECORDER_FIRST_ROW_TIMEOUT"))
        self.assertNotIn(("executor", "execute"), calls)
        self.assertNotIn(("executor", "cancel"), calls)
        self.assertEqual(calls.count(("recorder", "abort")), 1)
        self.assertNotIn(("recorder", "preserve"), calls)

    def test_cancel_while_waiting_for_first_row_never_executes(self):
        job, calls = self.make(["RECORDING", "ABORTED"], ["PLANNED", "APPROVED"])
        cancelled = [False]
        recorder = job.recorder_call

        def cancel_after_status(request):
            response = recorder(request)
            if request["op"] == "status":
                cancelled[0] = True
            return response

        job.recorder_call = cancel_after_status
        job.prepare(PLAN); job.approve(MOTION_APPROVAL)
        result = job.start(cancel_event=lambda: cancelled[0])
        self.assertEqual((result["state"], result["code"]), ("ABORTED", "START_CANCELLED"))
        self.assertNotIn(("executor", "execute"), calls)
        self.assertNotIn(("executor", "cancel"), calls)
        self.assertEqual(calls.count(("recorder", "abort")), 1)
        self.assertNotIn(("recorder", "preserve"), calls)

    def test_test_only_readiness_records_evidence_then_executes_once(self):
        def warnings_are_diagnostic(quality):
            quality["image_quality_warnings"] = ["up brightness warning"]

        job, calls = self.make(
            ["RECORDING"], ["PLANNED", "APPROVED", "EXECUTING"], first_row_rows=60,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT, quality_mutation=warnings_are_diagnostic,
        )
        self.prepare_and_start(job)
        evidence = job._result()["readiness_evidence"]
        self.assertEqual(evidence["run_id"], "run")
        self.assertEqual(evidence["transaction_id"], "tx")
        self.assertEqual(evidence["collection_profile_digest"], BINDINGS["collection_profile"])
        self.assertEqual(evidence["quality_contract_digest"], canonical_digest(TEST_ONLY_READINESS_CONTRACT))
        self.assertEqual(evidence["metrics"]["durable_rows"], 60)
        self.assertEqual(evidence["metrics"]["camera_source_fps"], {"up":30.0})
        self.assertEqual(evidence["metrics"]["image_quality_warnings"], ["up brightness warning"])
        self.assertEqual(calls.count(("recorder", "trim_readiness_prefix")), 1)
        self.assertEqual(calls.count(("executor", "execute")), 1)
        self.assertLess(calls.index(("recorder", "begin")), calls.index(("recorder", "status")))
        self.assertLess(calls.index(("recorder", "status")), calls.index(("recorder", "trim_readiness_prefix")))
        self.assertLess(calls.index(("recorder", "trim_readiness_prefix")), calls.index(("executor", "execute")))
        with self.assertRaisesRegex(ContractError, "RECORDER_READINESS_CONTRACT"):
            OneJob(lambda _: None, lambda _: None, readiness_contract={})

    def test_scene_darkness_blocks_new_motion_without_changing_technical_quality(self):
        for brightness in (0.0, 19.9, None, float("nan"), True, 256.0):
            with self.subTest(brightness=brightness):
                def dark_scene(quality):
                    quality["cameras"]["up"]["brightness_mean"] = brightness
                    self.assertTrue(quality["accepted"])
                job, calls = self.make(
                    ["RECORDING", "ABORTED"], ["PLANNED", "APPROVED"], first_row_rows=60,
                    readiness_contract=TEST_ONLY_READINESS_CONTRACT,
                    quality_mutation=dark_scene,
                )
                self.assertTrue(job.prepare(PLAN)["ok"])
                self.assertTrue(job.approve(MOTION_APPROVAL)["ok"])
                result = job.start()
                self.assertFalse(result["ok"])
                self.assertIn(result["code"], {
                    "RECORDER_READINESS_SCENE_DARK", "RECORDER_READINESS_SCENE_VISIBILITY",
                })
                self.assertNotIn(("executor", "execute"), calls)
                self.assertNotIn(("executor", "cancel"), calls)
                self.assertEqual(calls.count(("recorder", "abort")), 1)

    def test_scene_visibility_uses_up_camera_not_occluded_wrist(self):
        def visible_scene(quality):
            quality["cameras"]["up"]["brightness_mean"] = 20.0
            quality["cameras"]["wrist"] = {"source_fps":30.0, "brightness_mean":0.0}
            quality["image_quality_warnings"] = ["wrist brightness warning"]
        job, calls = self.make(
            ["RECORDING"], ["PLANNED", "APPROVED", "EXECUTING"], first_row_rows=60,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT, quality_mutation=visible_scene,
        )
        self.prepare_and_start(job)
        self.assertEqual(calls.count(("executor", "execute")), 1)
        self.assertEqual(job.readiness_evidence["metrics"]["scene_brightness_mean"], 20.0)

    def test_test_only_readiness_enters_barrier_with_live_writer_backlog(self):
        def active_writer(status):
            status["metrics"].update(rows=60, writer_queue=14)

        job, calls = self.make(
            ["RECORDING"], ["PLANNED", "APPROVED", "EXECUTING"],
            first_row_rows=60, readiness_contract=TEST_ONLY_READINESS_CONTRACT,
            status_mutation=active_writer,
        )
        self.prepare_and_start(job)
        self.assertEqual(job.state, "EXECUTING")
        self.assertEqual(calls.count(("recorder", "status")), 1)
        self.assertEqual(calls.count(("recorder", "trim_readiness_prefix")), 1)
        self.assertEqual(calls.count(("executor", "execute")), 1)

    def test_production_start_does_not_trim_readiness_prefix(self):
        job, calls = self.make(["RECORDING"], ["PLANNED", "APPROVED", "EXECUTING"])
        self.prepare_and_start(job)
        self.assertEqual(job.state, "EXECUTING")
        self.assertIsNone(job._result()["readiness_evidence"])
        self.assertNotIn(("recorder", "trim_readiness_prefix"), calls)
        self.assertLess(calls.index(("recorder", "begin")), calls.index(("executor", "execute")))

    def test_local_button_approval_is_test_only_and_does_not_claim_human_source(self):
        local = {
            **MOTION_APPROVAL,
            "source": "LOCAL_UI_BUTTON",
            "approval_scope": "HIL_NUMERIC_PROXY",
        }
        test_only, calls = self.make(
            ["RECORDING"], ["PLANNED", "APPROVED"], first_row_rows=60,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT,
        )
        self.assertTrue(test_only.prepare(PLAN)["ok"])
        self.assertTrue(test_only.approve(local)["ok"])
        self.assertIn(("executor", "approve"), calls)

        production, calls = self.make(["RECORDING"], ["PLANNED"])
        self.assertTrue(production.prepare(PLAN)["ok"])
        rejected = production.approve(local)
        self.assertEqual((rejected["ok"], rejected["code"]), (False, "APPROVAL_SCHEMA"))
        self.assertNotIn(("executor", "approve"), calls)

    def test_test_only_readiness_failures_abort_without_execute(self):
        cases = {
            "stale": ("status", lambda status: status["metrics"].update(
                observed_monotonic_ns=time.monotonic_ns() - 600_000_000
            ), "RECORDER_READINESS_STALE"),
            "row_fps": ("quality", lambda quality: quality.update(
                effective_fps=26.9, accepted=False, reasons=["row fps is too low"],
            ), "RECORDER_READINESS_ROW_FPS"),
            "camera_fps": ("quality", lambda quality: (
                quality["cameras"]["up"].update(source_fps=28.4),
                quality.update(accepted=False, reasons=["camera source fps is too low"]),
            ), "RECORDER_READINESS_CAMERA_FPS"),
            "drops": ("status", lambda status: status["metrics"].update(writer_queue_drops=1), "RECORDER_READINESS_DROPS"),
            "alignment": ("status", lambda status: status["metrics"].update(alignment_failures=1), "RECORDER_READINESS_ALIGNMENT"),
            "writer_fault": ("status", lambda status: status.update(writer_error="disk fault"), "RECORDER_WRITER_FAULT"),
            "quality_reason": ("quality", lambda quality: quality.update(
                accepted=False, reasons=["up image repeat ratio is too high"]
            ), "RECORDER_READINESS_QUALITY"),
            "prefix_mismatch": ("quality", lambda quality: quality.update(
                frames=59, accepted=False, reasons=["source provenance rows do not match"],
            ), "RECORDER_READINESS_MISMATCH"),
        }
        for name, (target, mutation, code) in cases.items():
            with self.subTest(name=name):
                job, calls = self.make(
                    ["RECORDING", "ABORTED"], ["PLANNED", "APPROVED"], first_row_rows=60,
                    readiness_contract=TEST_ONLY_READINESS_CONTRACT,
                    status_mutation=mutation if target == "status" else None,
                    quality_mutation=mutation if target == "quality" else None,
                )
                job.prepare(PLAN); job.approve(MOTION_APPROVAL)
                result = job.start()
                self.assertEqual((result["state"], result["code"]), ("ABORTED", code))
                self.assertEqual(result["readiness_failure_evidence"]["code"], code)
                self.assertEqual(job.transaction_id, "tx")
                self.assertEqual(calls.count(("recorder", "abort")), 1)
                self.assertEqual(calls.count(("executor", "execute")), 0)

        job, calls = self.make(
            ["RECORDING", "ABORTED"], ["PLANNED", "APPROVED"], first_row_rows=59,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT,
        )
        ticks = iter((0.0, 6.0))
        job.monotonic_clock = lambda: next(ticks)
        job.prepare(PLAN); job.approve(MOTION_APPROVAL)
        result = job.start()
        self.assertEqual((result["state"], result["code"]), ("ABORTED", "RECORDER_READINESS_TIMEOUT"))
        self.assertEqual(calls.count(("executor", "execute")), 0)

    def test_test_only_readiness_cancel_aborts_without_execute(self):
        job, calls = self.make(
            ["RECORDING", "ABORTED"], ["PLANNED", "APPROVED"], first_row_rows=60,
            readiness_contract=TEST_ONLY_READINESS_CONTRACT,
        )
        cancelled = [False]
        recorder = job.recorder_call

        def cancel_after_status(request):
            response = recorder(request)
            if request["op"] == "status":
                cancelled[0] = True
            return response

        job.recorder_call = cancel_after_status
        job.prepare(PLAN); job.approve(MOTION_APPROVAL)
        result = job.start(cancel_event=lambda: cancelled[0])
        self.assertEqual((result["state"], result["code"]), ("ABORTED", "START_CANCELLED"))
        self.assertEqual(calls.count(("recorder", "abort")), 1)
        self.assertEqual(calls.count(("executor", "execute")), 0)

    def test_semantic_fail_waits_for_executor_and_rejections_abort_once(self):
        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "ABORTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
        self.prepare_and_start(job); job.poll(); job.confirm("operator"); job.poll(); job.grasp_verdict("PASS", "operator"); job.poll(); job.semantic_verdict("FAIL", "operator")
        self.assertEqual(job.poll()["state"], "ABORTED")
        self.assertLess(calls.index(("executor_completed",)), calls.index(("recorder", "abort")))
        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "QUALITY_REJECTED", "ABORTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
        self.prepare_and_start(job); job.poll(); job.confirm("operator"); job.poll(); job.grasp_verdict("PASS", "operator"); job.poll(); job.semantic_verdict("PASS", "operator"); self.assertEqual(job.poll()["state"], "ABORTED")
        self.assertEqual(calls.count(("recorder", "commit")), 1); self.assertEqual(calls.count(("recorder", "abort")), 1)

    def test_quarantine_and_malformed_response_fail_closed_without_commit(self):
        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "QUARANTINED_COMMIT"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
        self.prepare_and_start(job); job.poll(); job.confirm("operator"); job.poll(); job.grasp_verdict("PASS", "operator"); job.poll(); job.semantic_verdict("PASS", "operator"); self.assertEqual(job.poll()["state"], "QUARANTINED_COMMIT")
        self.assertNotIn(("recorder", "abort"), calls)
        job, calls = self.make([], [{"schema_version":"wrong"}])
        self.assertFalse(job.prepare(PLAN)["ok"])
        self.assertNotIn(("recorder", "commit"), calls)
        rejected = {"schema_version":"fr5.pickup_executor.response.v3", "mode":"PRE_LIVE", "op_id":"01-plan", "op":"plan", "ok":False, "code":"PLAN_NOT_COMPLETE", "run_id":None, "plan_digest":None, "state":"IDLE", "data":None}
        job, calls = self.make([], [rejected])
        self.assertEqual(job.prepare(PLAN)["code"], "PLAN_NOT_COMPLETE")
        drift = {"schema_version":"data_factory.recorder_response.v1", "op_id":"06-status", "op":"status", "ok":True, "state":"RECORDING", "reason_code":"OK", "run_id":"other", "transaction_id":"other", "episode_index":0, "metrics":{}, "artifacts":{}, "detail":"", "writer_alive":True, "writer_error":None, "sampler_alive":True}
        job, calls = self.make(["RECORDING", drift, "ABORTED"], ["PLANNED", "APPROVED", "EXECUTING", "BLOCKED"])
        self.prepare_and_start(job); result = job.poll(); self.assertEqual((result["state"], result["code"]), ("BLOCKED", "RECORDER_BINDING"))
        self.assertNotIn(("recorder", "commit"), calls)

        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "ABORTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "BLOCKED", "BLOCKED"])
        self.prepare_and_start(job); job.poll(); job.confirm("operator"); job.poll()
        result = job.grasp_verdict("FAIL", "operator")
        self.assertEqual((result["state"], result["grasp_verdict"]), ("BLOCKED", "FAIL"))
        self.assertNotIn(("recorder", "commit"), calls)

        failed_cancel = {"schema_version":"fr5.pickup_executor.response.v3", "mode":"LIVE", "op_id":"05-cancel", "op":"cancel", "ok":False, "code":"CANCEL_UNCONFIRMED", "run_id":"run", "plan_digest":None, "state":"EXECUTING", "data":{"cancel_error":"ROS_EXEC_CANCEL_ACK_TIMEOUT"}}
        job, calls = self.make(["RECORDING", "ABORTED"], ["PLANNED", "APPROVED", "EXECUTING", failed_cancel])
        self.prepare_and_start(job); result = job.cancel()
        self.assertEqual((result["state"], result["recorder_state"]), ("BLOCKED", "ABORTED"))
        self.assertEqual(calls.count(("recorder", "abort")), 1)
        self.assertNotIn(("recorder", "preserve"), calls)

        job, calls = self.make(["RECORDING", "FROZEN"], ["PLANNED", "APPROVED", "EXECUTING", failed_cancel])
        self.prepare_and_start(job); result = job.cancel()
        self.assertEqual((result["state"], result["recorder_state"]), ("BLOCKED", "FROZEN"))
        self.assertEqual(calls.count(("recorder", "abort")), 1)
        self.assertIn(("recorder", "preserve"), calls)

    def test_precommit_safety_must_stay_exact_and_pass_before_commit(self):
        job, calls = self.make([], ["PLANNED"])
        executor = job.executor_call
        def invalid_preflight(request):
            response = executor(request)
            if request["op"] == "plan":
                response["data"]["precommit_safety"]["collision_report_digest"] = "not-a-digest"
            return response
        job.executor_call = invalid_preflight
        self.assertEqual(job.prepare(PLAN)["code"], "PRECOMMIT_SAFETY_SCHEMA")
        self.assertNotIn(("recorder", "commit"), calls)
        job, calls = self.make([], ["PLANNED"])
        executor = job.executor_call
        def tampered_evidence(request):
            response = executor(request)
            if request["op"] == "plan":
                response["data"]["precommit_evidence"]["collision_report"]["all_valid"] = False
            return response
        job.executor_call = tampered_evidence
        self.assertEqual(job.prepare(PLAN)["code"], "PRECOMMIT_EVIDENCE_BINDING")
        self.assertNotIn(("recorder", "commit"), calls)
        for mutate, digest_key in (
            (lambda evidence: evidence["collision_report"].update(all_valid=False), "collision_report_digest"),
            (lambda evidence: evidence["collision_report"].update(failure_count=1), "collision_report_digest"),
            (lambda evidence: evidence["plan_only_no_motion"].update(execute_goal_count=1), "plan_only_no_motion_digest"),
        ):
            job, _ = self.make([], ["PLANNED"])
            executor = job.executor_call
            def unsafe_evidence(request):
                response = executor(request)
                if request["op"] == "plan":
                    evidence = response["data"]["precommit_evidence"]
                    mutate(evidence)
                    response["data"]["precommit_safety"][digest_key] = canonical_digest(
                        evidence["collision_report"] if digest_key == "collision_report_digest" else evidence["plan_only_no_motion"]
                    )
                return response
            job.executor_call = unsafe_evidence
            self.assertEqual(job.prepare(PLAN)["code"], "PRECOMMIT_EVIDENCE_UNSAFE")
        job, _ = self.make([], ["PLANNED"])
        prepared = job.prepare(PLAN)
        self.assertEqual(prepared["plan_envelope"]["precommit_evidence"]["collision_report"]["all_valid"], True)
        for mutation in (lambda safety: safety.update(status="FAIL"), lambda safety: safety.pop("collision_report_digest"), lambda safety: safety.update(collision_report_digest="sha256:" + "f" * 64)):
            job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "ABORTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
            self.assertEqual(job.prepare(PLAN)["plan_envelope"]["operator_summary"], {})
            executor = job.executor_call
            def failed_precommit(request):
                response = executor(request)
                if request["op"] == "heartbeat" and response["state"] == "COMPLETED":
                    mutation(response["data"]["precommit_safety"])
                return response
            job.executor_call = failed_precommit
            job.approve(MOTION_APPROVAL); job.start(); job.poll(); job.confirm("operator"); job.poll(); job.grasp_verdict("PASS", "operator"); job.poll(); job.semantic_verdict("PASS", "operator")
            self.assertEqual(job.poll()["state"], "ABORTED")
            self.assertNotIn(("recorder", "commit"), calls)

    def test_jsonl_process_and_driver_keep_holds_alive(self):
        child = "import json,sys; [print(json.dumps(json.loads(line)),flush=True) for line in sys.stdin]"
        with JsonlProcess([sys.executable, "-u", "-c", child]) as process:
            self.assertEqual(process({"hello":"world"}), {"hello":"world"})
        cancelled = threading.Event()
        cancelled.set()
        with JsonlProcess([sys.executable, "-u", "-c", child]) as process:
            with self.assertRaisesRegex(
                ContractError, "JSONL_REQUEST_CANCELLED",
            ):
                process.request({"must_not_be_sent": True}, cancelled)
            self.assertEqual(process({"sent": "after-test"}), {"sent": "after-test"})
        process = JsonlProcess([sys.executable, "-u", "-c", child])
        with process:
            process.preserve()
            self.assertTrue(process.preserved)
        self.assertIsNone(process.process.poll())
        self.assertEqual(process.release(), 0)
        self.assertFalse(process.preserved)
        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
        decision_gate = threading.Event()
        def decide(state, _):
            if state == "PRECONTACT_HUMAN":
                decision_gate.wait(1)
                return "CONFIRM"
            return {"GRASP_VERDICT":"PASS", "SEMANTIC_VERDICT":"PASS", "AWAITING_CELL_READY":"READY"}.get(state)
        def keep_alive(_):
            decision_gate.set()
            time.sleep(.001)
        result = run_one_job(job, PLAN, MOTION_APPROVAL, decide, operator_id="operator", poll_interval_s=.001, sleep=keep_alive)
        self.assertEqual(result["state"], "COMPLETE")
        self.assertGreaterEqual(calls.count(("executor", "heartbeat")), 4)
        job, calls = self.make(["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"], ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"])
        result = run_one_job(job, PLAN, {**MOTION_APPROVAL, "approval_scope":"HIL_NUMERIC_PROXY"}, lambda state, _: {"PRECONTACT_HUMAN":"CONFIRM", "AWAITING_CELL_READY":"READY"}.get(state), operator_id="operator", poll_interval_s=.001, sleep=lambda _: None)
        self.assertEqual(result["state"], "COMPLETE")
        job, _ = self.make([], ["PLANNED"])
        with self.assertRaisesRegex(ContractError, "ONE_JOB_POLL_INTERVAL"):
            run_one_job(job, PLAN, {}, lambda *_: None, operator_id="operator", poll_interval_s=.5)

        heartbeats = []
        def slow_recorder(request):
            time.sleep(.12)
            return {"schema_version":"data_factory.recorder_response.v1", "op_id":request["op_id"], "op":"freeze", "ok":True, "state":"FROZEN", "reason_code":"FROZEN", "run_id":"run", "transaction_id":"tx", "episode_index":0, "metrics":{}, "artifacts":{}, "detail":""}
        def live_executor(request):
            heartbeats.append(request["op"])
            return {"schema_version":"fr5.pickup_executor.response.v3", "mode":"LIVE", "op_id":request["op_id"], "op":"heartbeat", "ok":True, "code":"SEMANTIC_VERDICT", "run_id":"run", "plan_digest":"sha256:" + "d"*64, "state":"SEMANTIC_VERDICT", "data":{}}
        job = OneJob(slow_recorder, live_executor)
        job.run_id, job.plan_digest, job.lease_id = "run", "sha256:" + "d"*64, "lease"
        job.transaction_id, job.episode_index, job.recorder_state = "tx", 0, "RECORDING"
        job._program = {"execution_timeouts_s":{"heartbeat_lease":.1,"semantic_verdict":1}}
        job._freeze_recorder_with_heartbeats({"writer_alive":True, "writer_error":None})
        self.assertEqual(job.recorder_state, "FROZEN")
        self.assertGreaterEqual(heartbeats.count("heartbeat"), 2)

        calls, preserved = [], []
        def stuck_recorder(request):
            calls.append(("recorder", request["op"]))
            if request["op"] == "freeze":
                time.sleep(.2)
                state = "FROZEN"
            else:
                state = "RECORDING"
            return {"schema_version":"data_factory.recorder_response.v1", "op_id":request["op_id"], "op":request["op"], "ok":True, "state":state, "reason_code":state, "run_id":"run", "transaction_id":"tx", "episode_index":0, "metrics":{}, "artifacts":{}, "detail":"", "writer_alive":True, "writer_error":None, "sampler_alive":True}
        stuck_recorder.preserve = lambda: preserved.append(True)
        def blocking_executor(request):
            calls.append(("executor", request["op"]))
            state = "BLOCKED" if request["op"] == "cancel" else "SEMANTIC_VERDICT"
            return {"schema_version":"fr5.pickup_executor.response.v3", "mode":"LIVE", "op_id":request["op_id"], "op":request["op"], "ok":state != "BLOCKED", "code":state, "run_id":"run", "plan_digest":"sha256:" + "d"*64, "state":state, "data":{"durable_blocked":True} if state == "BLOCKED" else {}}
        job = OneJob(stuck_recorder, blocking_executor)
        job.run_id, job.plan_digest, job.lease_id = "run", "sha256:" + "d"*64, "lease"
        job.transaction_id, job.episode_index = "tx", 0
        job.state, job.grasp, job.recorder_state, job.executor_state = "EXECUTING", "PASS", "RECORDING", "EXECUTING"
        job._program = {"execution_timeouts_s":{"heartbeat_lease":.03,"semantic_verdict":.05}, "steps":[{"pause_after":"GRASP_VERDICT"}]}
        started = time.monotonic()
        result = job.poll()
        self.assertLess(time.monotonic() - started, .15)
        self.assertEqual((result["state"], result["code"], result["cancel_error"]), ("BLOCKED", "RECORDER_FREEZE_TIMEOUT", "RECORDER_FREEZE_TIMEOUT"))
        self.assertTrue(preserved)
        self.assertNotIn(("recorder", "abort"), calls)
        time.sleep(.2)
        self.assertEqual(job.recorder_state, "FREEZE_UNCERTAIN")

    def test_jsonl_terminal_response_has_no_motion_deadline_and_timeout_taints_stream(self):
        delayed_echo = (
            "import json,sys,time; "
            "line=sys.stdin.readline(); time.sleep(.08); "
            "print(json.dumps(json.loads(line)),flush=True)"
        )
        with JsonlProcess(
            [sys.executable, "-u", "-c", delayed_echo], timeout_s=.02,
        ) as process:
            started = time.monotonic()
            self.assertEqual(
                process.request_terminal({"op":"commit"}), {"op":"commit"},
            )
            self.assertGreaterEqual(time.monotonic() - started, .06)

        pending = JsonlProcess(
            [sys.executable, "-u", "-c", delayed_echo], timeout_s=.02,
        )
        with self.assertRaisesRegex(ContractError, "JSONL_RESPONSE_TIMEOUT"):
            pending.request({"op":"status"})
        with self.assertRaisesRegex(ContractError, "JSONL_RESPONSE_PENDING"):
            pending.request({"op":"abort"})
        with self.assertRaises(ContractError):
            pending.close(timeout_s=.02)

        exited = JsonlProcess([
            sys.executable, "-u", "-c",
            "import sys; sys.stdin.readline(); raise SystemExit(3)",
        ], timeout_s=1)
        with self.assertRaises(ContractError) as raised:
            exited.request_terminal({"op":"commit"})
        self.assertEqual(raised.exception.code, "JSONL_PROCESS_EXIT")
        self.assertEqual(exited.close(), 3)

    def test_one_job_routes_only_commit_through_terminal_transport(self):
        job, _ = self.make(
            ["RECORDING", "RECORDING", "RECORDING", "RECORDING", "FROZEN", "FROZEN", "COMMITTED"],
            ["PLANNED", "APPROVED", "EXECUTING", "PRECONTACT_HUMAN", "EXECUTING", "GRASP_VERDICT", "EXECUTING", "SEMANTIC_VERDICT", "EXECUTING", "COMPLETED"],
        )
        recorder = job.recorder_call
        terminal_ops = []

        class RecorderPort:
            def __call__(self, request):
                if request["op"] == "commit":
                    raise AssertionError("commit must use terminal transport")
                return recorder(request)

            def request_terminal(self, request):
                terminal_ops.append(request["op"])
                return recorder(request)

            def preserve(self):
                return recorder.preserve()

        job.recorder_call = RecorderPort()
        result = run_one_job(
            job, PLAN, MOTION_APPROVAL,
            lambda state, _: {
                "PRECONTACT_HUMAN":"CONFIRM", "GRASP_VERDICT":"PASS",
                "SEMANTIC_VERDICT":"PASS", "AWAITING_CELL_READY":"READY",
            }[state],
            operator_id="operator", poll_interval_s=.001, sleep=lambda _: None,
        )
        self.assertEqual(result["state"], "COMPLETE")
        self.assertEqual(terminal_ops, ["commit"])

    def test_hil_numeric_gripper_verdict_is_a_pure_mechanical_proxy(self):
        required = {"command_position_m":.01, "acceptable_feedback_m":{"min":.01, "max":.012}}
        evidence = {"gripper_reference_m":.01, "gripper_feedback_m":.011, "post_lift_gripper_feedback_m":.012}
        self.assertEqual(hil_numeric_gripper_verdict("GRASP_VERDICT", evidence, required), "PASS")
        self.assertEqual(hil_numeric_gripper_verdict("SEMANTIC_VERDICT", evidence, required), "PASS")
        self.assertEqual(hil_numeric_gripper_verdict("GRASP_VERDICT", {**evidence, "gripper_reference_m":.02}, required), "FAIL")
        self.assertEqual(hil_numeric_gripper_verdict("SEMANTIC_VERDICT", {}, required), "FAIL")
        self.assertEqual(hil_numeric_gripper_verdict("HUMAN", evidence, required), "FAIL")
