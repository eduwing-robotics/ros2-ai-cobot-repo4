import io
import copy
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from tools.data_factory import scene_state
from tools.fr5_data_factory import ContractError, canonical_digest


class SceneStateTest(unittest.TestCase):
    def test_initial_contact_recovery_preserves_observation_and_rejects_changed_evidence(self):
        # Synthetic store transaction tests; canonical lifecycle validation is
        # a separate existing boundary, explicitly substituted in this fixture.
        for fault in (None, "cell", "scene", "prior_pose", "other_failure", "dispatched",
                      "history", "predecessor", "selected", "incarnation", "phase",
                      "preapproval", "canonical", "receipt_io", "scene_io", "cell_io",
                      "model", "model_null", "model_context", "model_history", "model_phase",
                      "other_contact", "chain", "chain_tamper", "chain_missing"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                success = fault in (None, "model", "chain")
                model_failure = str(fault).startswith(("model", "chain")) or fault == "other_contact"
                failure = ("CONTACT_MODEL_GEOMETRY" if fault == "other_contact" else
                           "CONTACT_MODEL_BINDING" if model_failure else "CONTACT_PROFILE_UNAVAILABLE")
                store = scene_state.SceneStateStore(directory, "fr5-lab-a")
                initial = store.update_object(instance_id="cube", object_profile_id="cube-24mm",
                    state="ON_SURFACE", source="HUMAN", updated_by="original-operator",
                    pose={"place_id":"PLACE_A", "yaw_deg":0, "x_mm":10, "y_mm":20})
                prior_cell = store._cell.acknowledge_ready("original-operator")
                confirmation = {"after_scene":initial, "after_cell":prior_cell}
                if str(fault).startswith("chain"):
                    prior_cell = {**prior_cell, "run_id":"synthetic-prior-recovery", "reason_code":"NO_DISPATCH_CONTINUITY"}
                    confirmation = {"schema_version":"data_factory.no_dispatch_recovery.v1",
                        "basis":"INITIAL_CONTACT_PROFILE_REJECTION_BEFORE_START_PHASE",
                        "prior_confirmation_digest":canonical_digest(confirmation),
                        "after_scene":initial["scene_state"], "after_cell":prior_cell,
                        "new_human_confirmation":False, "execution_authorized":False}
                    confirmation["receipt_digest"] = canonical_digest(confirmation)
                    retained = store._cell.runtime_path("no_dispatch_recovery-synthetic-prior-recovery.json")
                    scene_state.write_json_atomic(retained, confirmation)
                binding = {"scene_state_digest":initial["scene_state_digest"],
                           "revision":initial["scene_state"]["revision"], "object_instance_id":"cube"}
                clock = {"schema_version":"fr5.gripper_temporal_policy.v2", "incarnation":[1,2,3,4]}
                plan = {"run_id":"synthetic-contact-rejection", "robot_system_id":"fr5-lab-a",
                        "scene_binding":binding, "steps":[{"initial_hardware_binding":{"generation":0,"incarnation":[1,2,3,4]}}],
                        "learned_proposal":{"generation_context":{"predecessor_plan_digest":None},
                                            "runtime_inputs":{"clock_binding":clock}}}
                envelope = {"plan":plan}
                preapproval = {"run_id":plan["run_id"], "plan_digest":canonical_digest(plan),
                    "plan_envelope":envelope, "plan_envelope_digest":canonical_digest(envelope)}
                phase = Path(directory) / plan["run_id"] / "phase_events.jsonl"
                phase.parent.mkdir()
                phase.write_bytes(b"")
                failed = store.update_object(instance_id="cube", object_profile_id="cube-24mm",
                    state="UNKNOWN", source="ROBOT_ACTION", updated_by="pickup-executor")
                cell = store._cell.mark_blocked(failure, plan["run_id"], canonical_digest(plan))
                wire = dict.fromkeys(("generation", "active_generation", "completed_generation", "selected_generation",
                                      "selected_valid", "error", "device_main_error", "device_sub_error"), 0)
                wire.update(version=5, **{f"incarnation_{i}":i+1 for i in range(4)})
                trace = {"status":"FAILED", "failure_code":failure, "segments":[],
                         "terminal_phases":[], "terminal_state":None}
                result = {"run_id":plan["run_id"], "plan_digest":canonical_digest(plan), "plan_envelope":envelope,
                    "scene_binding":binding, "code":failure, "state":"ABORTED", "executor_state":"BLOCKED",
                    "execution_evidence":{"failure_code":failure, "step_index":0,
                        "prospective_contact":{"status":"UNAVAILABLE", "code":"CONTACT_PROFILE_UNAVAILABLE"},
                        "scene_transition":failed, "phase_events_path":str(phase),
                        "snapshot":{"gripper_controller":{"hardware_execution":{"wire":wire,"clock_binding":clock}}}}}
                if model_failure: result["execution_evidence"].pop("prospective_contact")
                if fault == "model_null": result["execution_evidence"]["prospective_contact"] = None
                if fault == "model_context": result["execution_evidence"]["prospective_contact"] = {"status":"PROSPECTIVE"}
                if fault == "chain_tamper": confirmation["after_cell"]["acknowledged_by"] = "forged"
                if fault == "chain_missing": retained.rename(retained.with_suffix(".unavailable"))
                if fault == "cell": store._cell.mark_blocked("OTHER_OWNER", "other", canonical_digest("other"))
                if fault == "scene": store.update_object(instance_id="cube", object_profile_id="cube-24mm", state="UNKNOWN", source="HUMAN", updated_by="other")
                if fault == "prior_pose": confirmation["after_scene"]["scene_state"]["objects"]["cube"]["pose"]["x_mm"] = 99
                if fault == "other_failure": result["code"] = "CANCEL_UNCONFIRMED"
                if fault == "dispatched": result["execution_evidence"]["step_index"] = 1
                if fault in ("history", "model_history"): result["execution_evidence"]["learned_history"] = [{}]
                if fault == "predecessor": plan["learned_proposal"]["generation_context"]["predecessor_plan_digest"] = canonical_digest("prior")
                if fault == "selected": wire["selected_valid"] = 1
                if fault == "incarnation": wire["incarnation_0"] = 99
                if fault in ("phase", "model_phase"): phase.write_text('{"event":"DISPATCH_REQUESTED"}\n')
                if fault == "preapproval": preapproval["plan_digest"] = canonical_digest("other")
                before_scene, before_cell = copy.deepcopy(store.read()), store._cell.read()
                original_observation = copy.deepcopy(initial["scene_state"]["objects"]["cube"])
                writes = []
                real_write = scene_state.write_json_atomic
                def write(path, value):
                    writes.append(Path(path).name)
                    if fault == {1:"receipt_io",2:"scene_io",3:"cell_io"}.get(len(writes)):
                        raise OSError("synthetic write failure")
                    return real_write(path, value)
                diagnostic = {"execution_trace":trace, "lifecycle_result_digest":canonical_digest(result)}
                with mock.patch("tools.data_factory.rollout.evidence_boundary.build_run_diagnostic",
                                side_effect=ContractError("SYNTHETIC_CANONICAL_REJECTION") if fault == "canonical" else None,
                                return_value=diagnostic) as canonical, mock.patch.object(scene_state, "write_json_atomic", side_effect=write):
                    if not success:
                        with self.assertRaises((ContractError, OSError)):
                            store.recover_initial_contact_rejection(prior_confirmation=confirmation,
                                lifecycle_result=result, preapproval_evidence=preapproval, expected_cell=cell)
                        self.assertEqual(store._cell.read(), before_cell)
                        if fault != "cell_io": self.assertEqual(store.read(), before_scene)
                    else:
                        receipt = store.recover_initial_contact_rejection(prior_confirmation=confirmation,
                            lifecycle_result=result, preapproval_evidence=preapproval, expected_cell=cell)
                        self.assertEqual(store.read()["objects"]["cube"], original_observation)
                        self.assertEqual(store.read()["revision"], before_scene["revision"] + 1)
                        self.assertEqual(store._cell.read()["reason_code"], "NO_DISPATCH_CONTINUITY")
                        self.assertEqual(store._cell.read()["acknowledged_by"], "original-operator")
                        self.assertTrue(store._cell.read()["cell_ready"])
                        self.assertFalse(receipt["new_human_confirmation"])
                        self.assertFalse(receipt["execution_authorized"])
                        self.assertEqual(receipt["prior_confirmation_digest"], canonical_digest(confirmation))
                        self.assertEqual(receipt["basis"], "INITIAL_CONTACT_MODEL_REJECTION_BEFORE_START_PHASE" if model_failure
                                         else "INITIAL_CONTACT_PROFILE_REJECTION_BEFORE_START_PHASE")
                        self.assertEqual(json.loads(store._cell.runtime_path(writes[0]).read_text()), receipt)
                        self.assertEqual(writes[1:], ["scene_state.json", "state.json"])
                        with self.assertRaises(ContractError):
                            store.recover_initial_contact_rejection(prior_confirmation=confirmation,
                                lifecycle_result=result, preapproval_evidence=preapproval, expected_cell=cell)
                    canonical.assert_called()

    def test_native_motion_only_release_vacates_source_under_parent_cell_owner(self):
        from datetime import datetime, timezone
        from tools.data_factory.cell_state import CellStateStore
        from tests.data_factory import test_motion as fixture
        for outcome in ("landed", "unknown", "wrong_parent", "wrong_parent_plan", "partial", "stale_source"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                store = scene_state.SceneStateStore(directory, "fr5-lab-a")
                cell = CellStateStore(directory, "fr5-lab-a")
                pose = {"place_id": "place-a", "yaw_deg": 0, "x_mm": 60, "y_mm": 0}
                initial = store.update_object(instance_id="cube-1", object_profile_id="wood-cube-25mm-r001",
                                              state="ON_SURFACE", pose=pose, source="HUMAN", updated_by="fixture")
                source = scene_state.release_slot(robot_system_id="fr5-lab-a", pose=pose,
                    object_profile_id="wood-cube-25mm-r001", exclusion_geometry_digest=canonical_digest({"shape":"BOX","dimensions_mm":[25,25,25]}), role="DESTINATION_THEN_NEXT_SOURCE")
                parent_plan = canonical_digest("parent-plan")
                evidence = {"schema_version":"data_factory.recycle_release_evidence.v1", "run_id":"parent-run", "plan_digest":parent_plan,
                    "release_slot_id":source["slot_id"], "expected_scene_state_digest":initial["scene_state_digest"],
                    "expected_scene_revision":initial["scene_state"]["revision"], "gripper_reference_m":.01, "gripper_feedback_m":.01,
                    "terminal_phases":["RECYCLE_APPROACH_PTP","LOWER_LIN","GRIPPER_OPEN","RETREAT_LIN","SAFE_POSE_PTP"],
                    "post_retreat_snapshot_digest":canonical_digest("snapshot"), "next_start_tolerance_rad":.01, "human_verdict":"LANDED"}
                cell.mark_blocked("EXECUTION_IN_PROGRESS", "parent-run", parent_plan)
                original_cell = cell.read()
                landed = store.transition_release(instance_id="cube-1", release_slot=source, evidence=evidence,
                    updated_by="pickup-executor", expected_digest=initial["scene_state_digest"], expected_revision=initial["scene_state"]["revision"],
                    allowed_next_run_id="reposition-run")
                original = landed["scene_state"]["slot_allocations"][source["slot_id"]]
                target = scene_state.release_slot(robot_system_id="fr5-lab-a", pose={**pose,"yaw_deg":20},
                    object_profile_id="wood-cube-25mm-r001", exclusion_geometry_digest=source["exclusion_geometry_digest"], role="DESTINATION_THEN_NEXT_SOURCE")
                binding = {"scene_state_digest":landed["scene_state_digest"], "revision":landed["scene_state"]["revision"],
                    "object_instance_id":"cube-1", "release_slot":target, "allowed_next_run_id":"next-run",
                    "source_slot":{"slot_id":source["slot_id"],"slot_digest":canonical_digest(original),"allowed_run_id":"reposition-run"}}

                class Live(fixture.T):
                    def __init__(self): super().__init__(); self.position=[0.]*6
                    def snapshot(self,*_): return fixture.snapshot(self.position)
                    def start_phase(self,step): self.position=step["final_joint_state"]
                    def poll_active(self): return object()
                    def cancel_active(self,*_): pass

                program = fixture.motion(True)
                node = fixture.e.PickupExecutor(Live(), clock=lambda:datetime(2026,1,1,tzinfo=timezone.utc), monotonic_clock=lambda:0,
                    cell_state_store=cell, scene_state_store=store, execution_enabled=True,
                    **fixture.motion_only(canonical_digest("reposition"),parent_plan,"reposition-run",program,binding))
                req = fixture.Test().req
                plan = node.process(req("plan", {"run_id":"reposition-run","motion_program":program,"scene_binding":binding}))
                self.assertTrue(plan["ok"], plan)
                node.process(req("approve", {"approval_id":"approval-1","approved_by":"operator","run_id":"reposition-run",
                    "resolved_job_digest":program["resolved_job_digest"],"plan_digest":plan["plan_digest"],"approval_expiry":"2026-01-02T00:00:00Z","approval_scope":"HIL_NUMERIC_PROXY"}))
                executed = node.process(req("execute", {"run_id":"reposition-run","plan_digest":plan["plan_digest"],"lease_id":"lease-1"}))
                self.assertTrue(executed["ok"], executed)
                self.assertEqual(cell.read(), original_cell)
                for _ in range(5): node.tick()
                node.process(req("semantic_verdict", {"run_id":"reposition-run","plan_digest":plan["plan_digest"],"verdict":"PASS","decided_by":"operator","source":"HUMAN"}))
                for _ in range(5): node.tick()
                self.assertEqual(node.runs["reposition-run"]["state"], "RELEASE_VERDICT")
                execution = node.runs["reposition-run"]["execution"]
                if outcome == "wrong_parent":
                    cell.mark_blocked("EXECUTION_IN_PROGRESS", "other-parent", canonical_digest("other-plan"))
                if outcome == "wrong_parent_plan":
                    cell.mark_blocked("EXECUTION_IN_PROGRESS", "parent-run", canonical_digest("other-plan"))
                if outcome == "partial": execution["release_evidence"]["terminal_phases"].pop()
                if outcome == "stale_source": execution["parent_cell_binding"]["source_slot_digest"] = canonical_digest("stale")
                result = node.process(req("release_verdict", {"run_id":"reposition-run","plan_digest":plan["plan_digest"],
                    "verdict":"UNCERTAIN" if outcome=="unknown" else "LANDED","decided_by":"operator","source":"LOCAL_UI_BUTTON"}))
                allocation = store.read()["slot_allocations"][source["slot_id"]]
                self.assertEqual(allocation["state"], "AVAILABLE" if outcome=="landed" else "CONSUMED_PENDING_REVIEW", result)
                for key in ("evidence_digest","evidence_run_id","evidence_plan_digest"):
                    self.assertEqual(allocation[key], original[key])
                if outcome == "landed":
                    self.assertEqual(result["state"], "COMPLETED")
                    self.assertEqual(cell.read(), original_cell)
                    self.assertEqual(store.read()["objects"]["cube-1"]["pose"], target["pose"])
                    from tools.data_factory.run_job import _scene_binding
                    next_binding = _scene_binding({"normalized_job": {"robot_system_id":"fr5-lab-a",
                        "object_profile_id":"wood-cube-25mm-r001", **target["pose"]},
                        "object_profile":{"dimensions_mm":[25,25,25]}}, pose, "next-run", root=directory)
                    self.assertEqual(next_binding["release_slot"]["pose"], pose)
                    cell.acknowledge_ready("parent-completed", expected_run_id="parent-run",
                                           expected_plan_digest=parent_plan)
                    restarted = scene_state.SceneStateStore(directory, "fr5-lab-a")
                    position = restarted.object_position(object_profile_id="wood-cube-25mm-r001", dimensions_mm=[25,25,25])
                    self.assertEqual(position["status"], "AVAILABLE", position)
                    self.assertEqual(position["pose"], target["pose"])
                    self.assertEqual((position["run_id"], position["plan_digest"]), ("reposition-run", plan["plan_digest"]))
                    before = restarted.snapshot()
                    ready_cell = cell.read()
                    rebound = restarted.rebind_landed_source(
                        object_profile_id="wood-cube-25mm-r001", dimensions_mm=[25,25,25], run_id="restarted-run",
                        expected_scene_digest=position["scene_state_digest"], expected_slot_digest=position["slot_digest"],
                        expected_cell_digest=position["cell_state_digest"])
                    self.assertEqual(rebound["scene_state"]["objects"], before["scene_state"]["objects"])
                    self.assertEqual(rebound["scene_state"]["slot_allocations"][source["slot_id"]], allocation)
                    old_target = before["scene_state"]["slot_allocations"][target["slot_id"]]
                    self.assertEqual(rebound["scene_state"]["slot_allocations"][target["slot_id"]],
                                     {**old_target, "allowed_run_id":"restarted-run"})
                    self.assertEqual(cell.read(), ready_cell)
                    with self.assertRaisesRegex(ContractError, "SCENE_STATE_CHANGED"):
                        restarted.rebind_landed_source(
                            object_profile_id="wood-cube-25mm-r001", dimensions_mm=[25,25,25], run_id="conflicting-restart",
                            expected_scene_digest=position["scene_state_digest"], expected_slot_digest=position["slot_digest"],
                            expected_cell_digest=position["cell_state_digest"])
                    import copy
                    from tools.data_factory_recovery import write_json_atomic
                    for conflict in ("newer_owner", "parent_plan", "active", "failed", "source_pending",
                                     "wrong_child", "missing_source", "ambiguous_parent", "later_source", "unknown"):
                        with self.subTest(conflict=conflict):
                            state = copy.deepcopy(rebound["scene_state"])
                            source_state = state["slot_allocations"][source["slot_id"]]
                            write_json_atomic(cell.runtime_path("state.json"), ready_cell)
                            if conflict in {"newer_owner", "parent_plan", "active", "failed"}:
                                cell.mark_blocked("ROS_EXEC_FAILED" if conflict == "failed" else "EXECUTION_IN_PROGRESS",
                                    "other-parent" if conflict == "newer_owner" else "parent-run",
                                    canonical_digest("other-plan") if conflict == "parent_plan" else parent_plan)
                                if conflict in {"newer_owner", "parent_plan"}:
                                    cell.acknowledge_ready("fixture")
                            elif conflict == "source_pending": source_state["state"] = "CONSUMED_PENDING_REVIEW"
                            elif conflict == "wrong_child": source_state["allowed_run_id"] = "other-child"
                            elif conflict == "missing_source": del state["slot_allocations"][source["slot_id"]]
                            elif conflict == "ambiguous_parent": state["slot_allocations"][canonical_digest("duplicate")] = source_state
                            elif conflict == "later_source": source_state["updated_at"] = ready_cell["updated_at"]
                            elif conflict == "unknown": state["objects"]["cube-1"].update(state="UNKNOWN", pose=None, source="ROBOT_ACTION")
                            write_json_atomic(restarted._path(), state)
                            blocked = restarted.object_position(object_profile_id="wood-cube-25mm-r001", dimensions_mm=[25,25,25])
                            self.assertEqual(blocked["status"], "BLOCKED", blocked)
                            with self.assertRaises(ContractError):
                                restarted.rebind_landed_source(
                                    object_profile_id="wood-cube-25mm-r001", dimensions_mm=[25,25,25], run_id="unsafe-restart",
                                    expected_scene_digest=blocked["scene_state_digest"], expected_slot_digest=canonical_digest(old_target),
                                    expected_cell_digest=blocked["cell_state_digest"])
                            self.assertEqual(restarted.read(), state)

    def test_only_confirmed_matching_execution_vacates_its_consumed_source(self):
        import copy
        from tests.data_factory.operator.test_object_position import ObjectPositionContinuityTests, POSE
        for condition in ("landed", "unknown", "partial", "wrong_run", "wrong_plan"):
            with self.subTest(condition=condition):
                fixture = ObjectPositionContinuityTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                before = fixture.scene.snapshot()
                allocation = before["scene_state"]["slot_allocations"][fixture.slot["slot_id"]]
                unrelated = {
                    canonical_digest(state): {**allocation, "state": state, "allowed_run_id": "old-e3"}
                    for state in ("RESERVED", "QUARANTINED")
                }
                unrelated[canonical_digest("other-run")] = {**allocation, "state": "CONSUMED_PENDING_REVIEW", "allowed_run_id": "other-run"}
                from tools.data_factory_recovery import write_json_atomic
                initial = before["scene_state"]
                initial["slot_allocations"].update(unrelated)
                write_json_atomic(fixture.cells / fixture.robot / "scene_state.json", initial)
                before = fixture.scene.snapshot()
                fixture.scene.consume_next_source(
                    slot_id=fixture.slot["slot_id"], run_id="old-e3",
                    expected_scene_digest=before["scene_state_digest"], expected_slot_digest=canonical_digest(allocation),
                )
                plan = canonical_digest("new-plan")
                fixture.cell.mark_blocked("EXECUTION_IN_PROGRESS", "other-run" if condition == "wrong_run" else "old-e3",
                                          canonical_digest("other-plan") if condition == "wrong_plan" else plan)
                before = fixture.scene.snapshot()
                consumed = before["scene_state"]["slot_allocations"][fixture.slot["slot_id"]]
                destination = scene_state.release_slot(
                    robot_system_id=fixture.robot, pose={**POSE, "place_id": "PLACE_B"},
                    object_profile_id=fixture.job["object_profile_id"], exclusion_geometry_digest=fixture.slot["exclusion_geometry_digest"],
                )
                evidence = copy.deepcopy(json.loads(fixture.original)["release_evidence"])
                evidence.update(run_id="old-e3", plan_digest=plan, release_slot_id=destination["slot_id"],
                                expected_scene_state_digest=before["scene_state_digest"], expected_scene_revision=before["scene_state"]["revision"])
                if condition == "unknown":
                    evidence.update(release_outcome="UNCERTAIN", outcome_source="EXECUTOR_FAILURE", terminal_phases=[])
                if condition == "partial":
                    evidence["terminal_phases"] = evidence["terminal_phases"][:-1]
                kwargs = dict(instance_id="cube", release_slot=destination, evidence=evidence, updated_by="pickup-executor",
                              expected_digest=before["scene_state_digest"], expected_revision=before["scene_state"]["revision"])
                if condition == "partial":
                    with self.assertRaisesRegex(ContractError, "RELEASE_EVIDENCE"):
                        fixture.scene.transition_release(**kwargs)
                    self.assertEqual(fixture.scene.snapshot(), before)
                else:
                    fixture.scene.transition_release(**kwargs)
                    with self.assertRaisesRegex(ContractError, "SCENE_STATE_CHANGED"):
                        fixture.scene.transition_release(**kwargs)
                expected = {**consumed, "state": "AVAILABLE"} if condition == "landed" else consumed
                self.assertEqual(fixture.scene.read()["slot_allocations"][fixture.slot["slot_id"]], expected)
                for slot_id, allocation in unrelated.items():
                    self.assertEqual(fixture.scene.read()["slot_allocations"][slot_id], allocation)
                self.assertEqual(fixture.episode.read_bytes(), fixture.original)

    def test_restart_source_handoff_is_atomic_preserves_release_and_rejects_new_motion(self):
        from concurrent.futures import ThreadPoolExecutor
        from tools.data_factory.cell_state import CellStateStore
        with tempfile.TemporaryDirectory() as directory:
            store = scene_state.SceneStateStore(directory, "robot")
            cell = CellStateStore(directory, "robot")
            pose = {"place_id": "place-a", "yaw_deg": -24.25, "x_mm": 12.125, "y_mm": -7.5}
            start = store.update_object(instance_id="cube", object_profile_id="wood", state="ON_SURFACE",
                                        source="HUMAN", updated_by="fixture", pose=pose)
            slot = scene_state.release_slot(robot_system_id="robot", pose=pose, object_profile_id="wood",
                                            exclusion_geometry_digest=canonical_digest({"shape": "BOX", "dimensions_mm": [24, 24, 24]}))
            evidence = {"schema_version": "data_factory.recycle_release_evidence.v1", "run_id": "old-run",
                        "plan_digest": "sha256:" + "a" * 64, "release_slot_id": slot["slot_id"],
                        "expected_scene_state_digest": start["scene_state_digest"], "expected_scene_revision": start["scene_state"]["revision"],
                        "gripper_reference_m": .021, "gripper_feedback_m": .021,
                        "terminal_phases": ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"],
                        "post_retreat_snapshot_digest": "sha256:" + "b" * 64, "next_start_tolerance_rad": .01, "human_verdict": "LANDED"}
            cell.mark_blocked("EXECUTION_IN_PROGRESS", "old-run", evidence["plan_digest"])
            landed = store.transition_release(instance_id="cube", release_slot=slot, evidence=evidence, updated_by="pickup-executor",
                                              expected_digest=start["scene_state_digest"], expected_revision=start["scene_state"]["revision"])
            cell.acknowledge_ready("fixture", expected_run_id="old-run", expected_plan_digest=evidence["plan_digest"])
            def binding():
                position = store.object_position(object_profile_id="wood", dimensions_mm=[24, 24, 24])
                return dict(object_profile_id="wood", dimensions_mm=[24, 24, 24],
                            expected_scene_digest=position["scene_state_digest"], expected_slot_digest=position["slot_digest"],
                            expected_cell_digest=position["cell_state_digest"])
            expected = binding()
            before_cell = cell.read()
            def compete(run_id):
                try:
                    return store.rebind_landed_source(run_id=run_id, **expected)
                except ContractError as exc:
                    return exc.code
            with ThreadPoolExecutor(max_workers=2) as workers:
                results = list(workers.map(compete, ["fresh-a", "fresh-b"]))
            self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
            self.assertIn("SCENE_STATE_CHANGED", results)
            after = store.snapshot()
            self.assertEqual(after["scene_state"]["objects"], landed["scene_state"]["objects"])
            allocation = after["scene_state"]["slot_allocations"][slot["slot_id"]]
            self.assertEqual(allocation["evidence_digest"], canonical_digest(evidence))
            self.assertEqual(allocation["evidence_plan_digest"], evidence["plan_digest"])
            self.assertEqual(cell.read(), before_cell)
            self.assertEqual(store.rebind_landed_source(run_id=allocation["allowed_run_id"], **binding()), after)
            stale_cell = binding()
            cell.mark_blocked("EXECUTION_IN_PROGRESS", "newer-motion", "sha256:" + "c" * 64)
            with self.assertRaisesRegex(ContractError, "STATE_CHANGED"):
                store.rebind_landed_source(run_id="fresh-c", **stale_cell)
            with self.assertRaisesRegex(ContractError, "STATE_CHANGED"):
                store.update_object(instance_id="cube", object_profile_id="wood", state="ON_SURFACE",
                                    source="HUMAN", updated_by="fixture", pose=pose,
                                    expected_revision=after["scene_state"]["revision"],
                                    expected_cell_digest=stale_cell["expected_cell_digest"])
            with self.assertRaisesRegex(ContractError, "OBJECT_POSITION_EXECUTION_IN_PROGRESS"):
                store.rebind_landed_source(run_id="fresh-c", **binding())
            self.assertEqual(store.snapshot(), after)
            cell.mark_blocked("ROS_EXEC_FAILED", "newer-motion", "sha256:" + "c" * 64)
            with self.assertRaisesRegex(ContractError, "OBJECT_POSITION_NEWER_EXECUTION"):
                store.rebind_landed_source(run_id="fresh-c", **binding())
            self.assertEqual(store.snapshot(), after)

    def test_intermediate_campaign_binding_has_previous_source_and_next_destination(self):
        destination = scene_state.release_slot(
            robot_system_id="fr5-lab-a",
            pose={"place_id": "place-a", "yaw_deg": -90, "x_mm": 35, "y_mm": 0},
            object_profile_id="wood-cube-25mm-r001",
            exclusion_geometry_digest="sha256:" + "e" * 64,
            role="DESTINATION_THEN_NEXT_SOURCE",
        )
        binding = {
            "scene_state_digest": "sha256:" + "a" * 64,
            "revision": 2,
            "object_instance_id": "cube-1",
            "release_slot": destination,
            "allowed_next_run_id": "run-3",
            "source_slot": {
                "slot_id": "sha256:" + "b" * 64,
                "slot_digest": "sha256:" + "c" * 64,
                "allowed_run_id": "run-2",
            },
        }

        self.assertEqual(scene_state.validate_scene_binding(binding), binding)

    def test_human_robot_and_external_updates_share_one_revisioned_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs/data_factory/cells"
            root.mkdir(parents=True)
            store = scene_state.SceneStateStore(root, "fr5-lab-a")
            empty = store.snapshot()
            self.assertEqual((empty["scene_state"]["revision"], empty["scene_state"]["objects"]), (0, {}))
            surface = store.update_object(
                instance_id="cube-1",
                object_profile_id="wood-cube-25mm-r001",
                state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": 0, "y_mm": 0},
                source="HUMAN",
                updated_by="project-owner",
                expected_revision=0,
            )
            self.assertEqual(surface["scene_state"]["objects"]["cube-1"]["pose"], {"place_id": "place-a", "yaw_deg": 0, "x_mm": 0, "y_mm": 0})
            self.assertEqual(surface["scene_state_digest"], canonical_digest(surface["scene_state"]))
            with self.assertRaisesRegex(ContractError, "SCENE_REVISION_CONFLICT"):
                store.update_object(instance_id="cube-1", object_profile_id="wood-cube-25mm-r001", state="HELD", source="ROBOT_ACTION", updated_by="one-job", expected_revision=0)
            held = store.update_object(instance_id="cube-1", object_profile_id="wood-cube-25mm-r001", state="HELD", source="ROBOT_ACTION", updated_by="one-job", expected_revision=1)
            self.assertEqual((held["scene_state"]["revision"], held["scene_state"]["objects"]["cube-1"]["pose"]), (2, None))
            unknown = store.update_object(instance_id="cube-1", object_profile_id="wood-cube-25mm-r001", state="UNKNOWN", source="HUMAN", updated_by="project-owner", expected_revision=2)
            self.assertEqual(unknown["scene_state"]["objects"]["cube-1"]["state"], "UNKNOWN")
            ai = store.update_object(instance_id="cube-1", object_profile_id="wood-cube-25mm-r001", state="UNKNOWN", source="AI", updated_by="factory-agent", expected_revision=3)
            self.assertEqual((ai["scene_state"]["revision"], ai["scene_state"]["objects"]["cube-1"]["source"]), (4, "AI"))

            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = scene_state.main(("show", "--root", str(root), "--robot-system-id", "fr5-lab-a"))
            self.assertEqual((code, err.getvalue(), json.loads(out.getvalue())["scene_state"]["revision"]), (0, "", 4))

            malformed = store.read()
            malformed["updated_at"] = "2026-01-01 00:00:00+00:00"
            path = root / "fr5-lab-a/scene_state.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "SCENE_TIMESTAMP"):
                store.read()

            path.unlink()
            os.symlink(root / "outside.json", path)
            with self.assertRaisesRegex(ContractError, "STATE_PATH"):
                store.read()

    def test_release_transition_updates_object_and_slot_atomically(self):
        import fcntl, subprocess, sys
        if os.environ.get("FR5_SCENE_STORE_LOCK_REPLAY") != "1":
            result = subprocess.run([sys.executable, "-m", "unittest",
                "tests.data_factory.test_scene_state.SceneStateTest.test_release_transition_updates_object_and_slot_atomically"],
                env=dict(os.environ, FR5_SCENE_STORE_LOCK_REPLAY="1"), capture_output=True, text=True, timeout=6)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs/data_factory/cells"
            store = scene_state.SceneStateStore(root, "fr5-lab-a")
            start = store.update_object(
                instance_id="cube-1",
                object_profile_id="wood-cube-25mm-r001",
                state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": -60, "y_mm": 0},
                source="HUMAN",
                updated_by="project-owner",
                expected_revision=0,
            )
            slot = scene_state.release_slot(
                robot_system_id="fr5-lab-a",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": 60, "y_mm": 0},
                object_profile_id="wood-cube-25mm-r001",
                exclusion_geometry_digest="sha256:" + "e" * 64,
            )

            def evidence(verdict, terminals, snapshot_digest="sha256:" + "4" * 64):
                return {
                    "schema_version": "data_factory.recycle_release_evidence.v1",
                    "run_id": "run-1",
                    "plan_digest": "sha256:" + "a" * 64,
                    "release_slot_id": slot["slot_id"],
                    "expected_scene_state_digest": start["scene_state_digest"],
                    "expected_scene_revision": start["scene_state"]["revision"],
                    "gripper_reference_m": 0.021 if verdict == "LANDED" else None,
                    "gripper_feedback_m": 0.021 if verdict == "LANDED" else None,
                    "terminal_phases": terminals,
                    "post_retreat_snapshot_digest": snapshot_digest,
                    "next_start_tolerance_rad": 0.01,
                    "human_verdict": verdict,
                }

            descriptor = os.open(store._cell.runtime_path("scene_state.lock"), os.O_RDWR)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            before = store._path().read_bytes()
            try:
                with self.assertRaisesRegex(ContractError, "SCENE_STATE_BUSY"):
                    with store.locked_snapshot(start["scene_state_digest"], blocking=False):
                        self.fail("foreign lock must exclude a reader")
                with self.assertRaisesRegex(ContractError, "SCENE_STATE_BUSY"):
                    store.update_object(instance_id="cube-1", object_profile_id="wood-cube-25mm-r001",
                        state="UNKNOWN", source="ROBOT_ACTION", updated_by="pickup-executor", blocking=False)
                with self.assertRaisesRegex(ContractError, "SCENE_STATE_BUSY"):
                    store.transition_release(instance_id="cube-1", release_slot=slot,
                        evidence=evidence("UNCERTAIN", []), updated_by="pickup-executor",
                        expected_digest=start["scene_state_digest"], expected_revision=start["scene_state"]["revision"], blocking=False)
                self.assertEqual(store._path().read_bytes(), before)
            finally:
                os.close(descriptor)
            with store.locked_snapshot(start["scene_state_digest"], blocking=False) as current:
                self.assertEqual(current, start)
            terminals = ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"]
            landed = store.transition_release(
                instance_id="cube-1", release_slot=slot, evidence=evidence("LANDED", terminals),
                updated_by="pickup-executor", expected_digest=start["scene_state_digest"],
                expected_revision=start["scene_state"]["revision"],
            )
            scene = landed["scene_state"]
            self.assertEqual((scene["schema_version"], scene["revision"]), (scene_state.SCHEMA_VERSION, 2))
            self.assertEqual((scene["objects"]["cube-1"]["pose"], scene["slot_allocations"][slot["slot_id"]]["state"]), (slot["pose"], "CONSUMED_PENDING_REVIEW"))
            persisted = (root / "fr5-lab-a/scene_state.json").read_bytes()
            with self.assertRaisesRegex(ContractError, "SCENE_STATE_CHANGED"):
                store.transition_release(
                    instance_id="cube-1", release_slot=slot, evidence=evidence("UNCERTAIN", []),
                    updated_by="pickup-executor", expected_digest=start["scene_state_digest"],
                    expected_revision=start["scene_state"]["revision"],
                )
            self.assertEqual((root / "fr5-lab-a/scene_state.json").read_bytes(), persisted)

            second = scene_state.SceneStateStore(root, "fr5-lab-b")
            second_start = second.update_object(
                instance_id="cube-1", object_profile_id="wood-cube-25mm-r001", state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": -60, "y_mm": 0},
                source="HUMAN", updated_by="project-owner", expected_revision=0,
            )
            second_slot = scene_state.release_slot(
                robot_system_id="fr5-lab-b", pose=slot["pose"],
                object_profile_id="wood-cube-25mm-r001", exclusion_geometry_digest="sha256:" + "e" * 64,
            )
            uncertain_evidence = evidence("UNCERTAIN", [])
            uncertain_evidence.update(
                release_slot_id=second_slot["slot_id"],
                expected_scene_state_digest=second_start["scene_state_digest"],
                expected_scene_revision=second_start["scene_state"]["revision"],
            )
            uncertain = second.transition_release(
                instance_id="cube-1", release_slot=second_slot, evidence=uncertain_evidence,
                updated_by="pickup-executor", expected_digest=second_start["scene_state_digest"],
                expected_revision=second_start["scene_state"]["revision"],
            )["scene_state"]
            self.assertEqual((uncertain["objects"]["cube-1"]["state"], uncertain["slot_allocations"][second_slot["slot_id"]]["state"]), ("UNKNOWN", "QUARANTINED"))

    def test_chain_release_is_consumed_once_by_the_exact_next_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs/data_factory/cells"
            store = scene_state.SceneStateStore(root, "fr5-lab-a")
            start = store.update_object(
                instance_id="cube-1", object_profile_id="wood-cube-25mm-r001", state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": -60, "y_mm": 0},
                source="HUMAN", updated_by="project-owner", expected_revision=0,
            )
            slot = scene_state.release_slot(
                robot_system_id="fr5-lab-a",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": 0, "y_mm": 0},
                object_profile_id="wood-cube-25mm-r001", exclusion_geometry_digest="sha256:" + "e" * 64,
                role="DESTINATION_THEN_NEXT_SOURCE",
            )
            terminals = ["RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN", "RETREAT_LIN", "SAFE_POSE_PTP"]
            evidence = {
                "schema_version": "data_factory.recycle_release_evidence.v1", "run_id": "run-1",
                "plan_digest": "sha256:" + "a" * 64, "release_slot_id": slot["slot_id"],
                "expected_scene_state_digest": start["scene_state_digest"],
                "expected_scene_revision": start["scene_state"]["revision"],
                "gripper_reference_m": 0.021, "gripper_feedback_m": 0.021,
                "terminal_phases": terminals, "post_retreat_snapshot_digest": "sha256:" + "4" * 64,
                "next_start_tolerance_rad": 0.01, "human_verdict": "LANDED",
            }
            with self.assertRaisesRegex(ContractError, "SCENE_SLOT_NEXT_RUN"):
                store.transition_release(
                    instance_id="cube-1", release_slot=slot, evidence=evidence,
                    updated_by="pickup-executor", expected_digest=start["scene_state_digest"],
                    expected_revision=start["scene_state"]["revision"],
                )

            landed = store.transition_release(
                instance_id="cube-1", release_slot=slot, evidence=evidence,
                updated_by="pickup-executor", expected_digest=start["scene_state_digest"],
                expected_revision=start["scene_state"]["revision"], allowed_next_run_id="run-2",
            )
            slot_value = landed["scene_state"]["slot_allocations"][slot["slot_id"]]
            self.assertEqual((slot_value["state"], slot_value["role"], slot_value["allowed_run_id"]), ("LANDED_FOR_NEXT_SOURCE", "DESTINATION_THEN_NEXT_SOURCE", "run-2"))
            with self.assertRaisesRegex(ContractError, "SCENE_SLOT_CHANGED"):
                store.consume_next_source(
                    slot_id=slot["slot_id"], run_id="run-2",
                    expected_scene_digest=landed["scene_state_digest"], expected_slot_digest="sha256:" + "9" * 64,
                )
            with self.assertRaisesRegex(ContractError, "SCENE_SLOT_NEXT_RUN"):
                store.consume_next_source(
                    slot_id=slot["slot_id"], run_id="run-3",
                    expected_scene_digest=landed["scene_state_digest"], expected_slot_digest=canonical_digest(slot_value),
                )
            consumed = store.consume_next_source(
                slot_id=slot["slot_id"], run_id="run-2",
                expected_scene_digest=landed["scene_state_digest"], expected_slot_digest=canonical_digest(slot_value),
            )
            self.assertEqual(consumed["scene_state"]["slot_allocations"][slot["slot_id"]]["state"], "CONSUMED_PENDING_REVIEW")
            with self.assertRaisesRegex(ContractError, "SCENE_SLOT_NEXT_RUN"):
                store.consume_next_source(
                    slot_id=slot["slot_id"], run_id="run-2",
                    expected_scene_digest=consumed["scene_state_digest"],
                    expected_slot_digest=canonical_digest(consumed["scene_state"]["slot_allocations"][slot["slot_id"]]),
                )

            terminal_slot = {**slot, "role": "RELEASE_DESTINATION"}
            terminal_evidence = {
                **evidence,
                "run_id": "wrong-run",
                "plan_digest": "sha256:" + "b" * 64,
                "expected_scene_state_digest": consumed["scene_state_digest"],
                "expected_scene_revision": consumed["scene_state"]["revision"],
            }
            with self.assertRaisesRegex(ContractError, "SCENE_SLOT_UNAVAILABLE"):
                store.transition_release(
                    instance_id="cube-1", release_slot=terminal_slot,
                    evidence=terminal_evidence, updated_by="pickup-executor",
                    expected_digest=consumed["scene_state_digest"],
                    expected_revision=consumed["scene_state"]["revision"],
                )
            terminal_evidence["run_id"] = "run-2"
            terminal = store.transition_release(
                instance_id="cube-1", release_slot=terminal_slot,
                evidence=terminal_evidence, updated_by="pickup-executor",
                expected_digest=consumed["scene_state_digest"],
                expected_revision=consumed["scene_state"]["revision"],
            )["scene_state"]
            terminal_allocation = terminal["slot_allocations"][slot["slot_id"]]
            self.assertEqual(
                (
                    terminal["objects"]["cube-1"]["pose"],
                    terminal_allocation["state"], terminal_allocation["role"],
                    terminal_allocation["evidence_run_id"],
                ),
                (slot["pose"], "CONSUMED_PENDING_REVIEW", "RELEASE_DESTINATION", "run-2"),
            )

    def test_reposition_slot_is_owned_by_a_distinct_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = scene_state.SceneStateStore(
                Path(directory) / "cells", "fr5-lab-a",
            )
            start = store.update_object(
                instance_id="cube-1",
                object_profile_id="wood-cube-25mm-r001",
                state="ON_SURFACE",
                pose={
                    "place_id": "place-a", "yaw_deg": 0,
                    "x_mm": 0, "y_mm": 0,
                },
                source="HUMAN", updated_by="operator", expected_revision=0,
            )
            slot = scene_state.release_slot(
                robot_system_id="fr5-lab-a",
                pose={
                    "place_id": "place-a", "yaw_deg": 0,
                    "x_mm": 30, "y_mm": 0,
                },
                object_profile_id="wood-cube-25mm-r001",
                exclusion_geometry_digest="sha256:" + "e" * 64,
                role="DESTINATION_THEN_NEXT_SOURCE",
            )
            evidence = {
                "schema_version": "data_factory.recycle_release_evidence.v1",
                "run_id": "episode-run",
                "plan_digest": "sha256:" + "a" * 64,
                "release_slot_id": slot["slot_id"],
                "expected_scene_state_digest": start["scene_state_digest"],
                "expected_scene_revision": start["scene_state"]["revision"],
                "gripper_reference_m": 0.021,
                "gripper_feedback_m": 0.021,
                "terminal_phases": [
                    "RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN",
                    "RETREAT_LIN", "SAFE_POSE_PTP",
                ],
                "post_retreat_snapshot_digest": "sha256:" + "4" * 64,
                "next_start_tolerance_rad": 0.01,
                "human_verdict": "LANDED",
            }
            with self.assertRaisesRegex(ContractError, "SCENE_SLOT_NEXT_RUN"):
                store.transition_release(
                    instance_id="cube-1", release_slot=slot,
                    evidence=evidence, updated_by="pickup-executor",
                    expected_digest=start["scene_state_digest"],
                    expected_revision=start["scene_state"]["revision"],
                    allowed_next_run_id="episode-run",
                )
            landed = store.transition_release(
                instance_id="cube-1", release_slot=slot, evidence=evidence,
                updated_by="pickup-executor",
                expected_digest=start["scene_state_digest"],
                expected_revision=start["scene_state"]["revision"],
                allowed_next_run_id="episode-run-reposition",
            )
            allocation = landed["scene_state"]["slot_allocations"][
                slot["slot_id"]
            ]
            self.assertEqual(
                (allocation["state"], allocation["allowed_run_id"]),
                ("LANDED_FOR_NEXT_SOURCE", "episode-run-reposition"),
            )
            consumed = store.consume_next_source(
                slot_id=slot["slot_id"], run_id="episode-run-reposition",
                expected_scene_digest=landed["scene_state_digest"],
                expected_slot_digest=canonical_digest(allocation),
            )
            self.assertEqual(
                consumed["scene_state"]["slot_allocations"][slot["slot_id"]][
                    "state"
                ],
                "CONSUMED_PENDING_REVIEW",
            )

    def test_human_start_confirmation_discards_prior_robot_slot_leases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs/data_factory/cells"
            store = scene_state.SceneStateStore(root, "fr5-lab-a")
            start = store.update_object(
                instance_id="cube-1", object_profile_id="wood-cube-25mm-r001",
                state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": -60, "y_mm": 0},
                source="HUMAN", updated_by="project-owner", expected_revision=0,
            )
            slot = scene_state.release_slot(
                robot_system_id="fr5-lab-a",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": 60, "y_mm": 0},
                object_profile_id="wood-cube-25mm-r001",
                exclusion_geometry_digest="sha256:" + "e" * 64,
                role="DESTINATION_THEN_NEXT_SOURCE",
            )
            evidence = {
                "schema_version": "data_factory.recycle_release_evidence.v1",
                "run_id": "old-run", "plan_digest": "sha256:" + "a" * 64,
                "release_slot_id": slot["slot_id"],
                "expected_scene_state_digest": start["scene_state_digest"],
                "expected_scene_revision": start["scene_state"]["revision"],
                "gripper_reference_m": 0.021, "gripper_feedback_m": 0.021,
                "terminal_phases": [
                    "RECYCLE_APPROACH_PTP", "LOWER_LIN", "GRIPPER_OPEN",
                    "RETREAT_LIN", "SAFE_POSE_PTP",
                ],
                "post_retreat_snapshot_digest": "sha256:" + "4" * 64,
                "next_start_tolerance_rad": 0.01, "human_verdict": "LANDED",
            }
            landed = store.transition_release(
                instance_id="cube-1", release_slot=slot, evidence=evidence,
                updated_by="pickup-executor", expected_digest=start["scene_state_digest"],
                expected_revision=start["scene_state"]["revision"],
                allowed_next_run_id="old-next-run",
            )

            confirmed = store.update_object(
                instance_id="cube-1", object_profile_id="wood-cube-25mm-r001",
                state="ON_SURFACE",
                pose={"place_id": "place-a", "yaw_deg": 0, "x_mm": 0, "y_mm": 0},
                source="HUMAN", updated_by="project-owner",
                expected_revision=landed["scene_state"]["revision"],
            )

            self.assertEqual(confirmed["scene_state"]["slot_allocations"], {})


if __name__ == "__main__":
    unittest.main()
