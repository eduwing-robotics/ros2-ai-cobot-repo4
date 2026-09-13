"""Stored native evidence through the actual Collection composition, without devices."""
import copy
import functools
import json
from pathlib import Path
import subprocess
import threading
import unittest
from unittest import mock

from tests.data_factory import test_collection_recommendation as evidence_fixture
from tests.data_factory.operator import test_object_position as position_fixture
from tools.data_factory.campaign_authoring import compile_collection_campaign
from tools.data_factory.operator.catalog import selected_motion_preset
from tools.data_factory.operator.composition import build_physical_operator_application
from tools.data_factory.operator.web.bridge import LoopbackBridge
from tools.fr5_data_factory import ContractError, canonical_digest, load_json_strict


class AcquisitionAdviceTests(unittest.TestCase):
    def setUp(self):
        self.native = position_fixture.ObjectPositionContinuityTests()
        self.native.setUp()
        self.addCleanup(self.native.doCleanups)
        self.root = self.native.root
        self.sequence = 0
        seed_app = self.application("evidence-seed")
        self.send(seed_app, "compile_draft", {"draft_id": seed_app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.fixture, self.runs = self.store_evidence(seed_app)
        self.source = {"run_directories": self.runs, "scene_state_path": str(self.native.cells / self.native.robot / "scene_state.json")}
        self.app = self.application("acquisition-consumer")

    def store_evidence(self, seed_app, name="evidence"):
        owner = seed_app._campaign.campaign_operator
        fixture = evidence_fixture.RecommendationFixture(dataset_root=str(self.root / (name + "-data")),
            evidence_root=str(self.root / "outputs/data_factory/runs" if name == "evidence" else self.root / name))
        fixture.hypothesis = copy.deepcopy(owner.hypothesis)
        fixture.draft = copy.deepcopy(owner.draft)
        fixture.manifest, fixture.receipt = compile_collection_campaign(fixture.draft, hypothesis=fixture.hypothesis)
        endpoints = {item["workspace_id"]: item for item in seed_app.catalog["combinations"]
                     if item["combination_digest"] in {entry["combination_digest"] for entry in seed_app._workspace_cycle()}}
        fixture.evidence = []
        for index in range(2):
            slot = fixture.manifest["slots"][index]
            condition = next(item["coverage_condition"] for item in fixture.hypothesis["base_conditions"]
                             if item["base_condition_digest"] == slot["base_condition_digest"])
            digests = endpoints[condition["place_id"]]["source_digests"]
            # Bind the synthetic artifact builder to this real compiled domain.
            def digest(value):
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] in {"cell", "object", "grasp"}:
                    return digests[value[0]]
                return canonical_digest(value)
            with mock.patch.object(evidence_fixture, "digest", side_effect=digest):
                fixture.evidence.append(fixture.episode(index, index))
            episode = fixture.evidence[-1]
            runtime = episode["artifacts"]["runtime_binding"]
            runtime.update(schema_version="data_factory.production_episode_binding.v1", data_disposition="PRODUCTION",
                           state_initialization_digest=None, scene_observation_digest=canonical_digest("synthetic-observation"))
            evidence_fixture.redigest(runtime, "binding_digest")
            fixture.rebind_episode(episode)
        return fixture, fixture.store()

    def send(self, app, op, payload=None):
        self.sequence += 1
        return app.core.consume(self.native.request(app, op, payload or {}, f"acquisition-{self.sequence}"))

    def application(self, session, source=None, rollout=None):
        app = self.native.application(session, builder=functools.partial(
            build_physical_operator_application, collection_evidence_call=source,
            **({"rollout_lifecycle_path": rollout} if rollout is not None else {})))
        self.send(app, "update_camera_bindings", {"bindings": {
            "usb-Generic_USB2.0_PC_CAMERA-video-index0": "UP", "usb-Generic_USB2.0_PC_CAMERA_2-video-index0": "WRIST"}})
        preset = load_json_strict(self.root / "config/data_factory/motion_presets/demonstration-rhythm-r001.json")
        for value in ({"selection": {"task": "pick_place"}}, {"selection": {"variant": "TWO_STAGE_ALIGN_V2"}},
                      {"motion_preset": {"id": preset["motion_preset_id"], "digest": canonical_digest(preset)}},
                      {"requested_count": 2}):
            self.send(app, "update_draft", {"draft_id": app.draft["draft_id"], **value})
        return app

    def refresh(self):
        self.send(self.app, "refresh_collection_advice")
        advice = self.app.projection()["collection_advice"]
        self.assertEqual(advice["status"], "READY", advice)
        return advice

    def transition_rollout(self, original_index=0, *, scoped_terminal=False):
        """Actual native resolver/program/language producers, synthetic learned review."""
        from tools.data_factory import run_job
        from tools.data_factory.rollout.finite_plan import compile_program
        from tools.data_factory.scene_state import release_slot
        captured = []
        resolve = run_job.resolve_inputs
        def retain(*args, **kwargs):
            result = resolve(*args, **kwargs)
            captured.append(copy.deepcopy(result))
            return result
        seed = self.application("transition-source")
        self.send(seed, "update_draft", {"draft_id": seed.draft["draft_id"], "selection": {"variant": "DIRECT"}})
        if original_index:
            self.send(seed, "update_draft", {"draft_id": seed.draft["draft_id"], "requested_count": original_index + 2})
        with mock.patch.object(run_job, "resolve_inputs", side_effect=retain):
            self.send(seed, "compile_draft", {"draft_id": seed.draft["draft_id"], "data_disposition": "PRODUCTION"})
        fixture, runs = self.store_evidence(seed, "transition-evidence")
        instruction = copy.deepcopy(seed._campaign._episode_instruction_bindings[original_index])
        source_pose, destination = [e["pose"] for e in instruction["task_binding"]["spatial_bindings"]]
        resolved, source, _ = next(item for item in captured if
            {k: item[0]["normalized_job"][k] for k in source_pose} == source_pose
            and item[1]["schema_version"] == "fr5.motion_program.v4")
        order = next(i for i, b in enumerate(fixture.hypothesis["base_conditions"])
                     if b["resolved_job_digest"] == resolved["resolved_job_digest"])
        lifecycle = evidence_fixture.learned_lifecycle(fixture, order=order, reviewed=True)
        plan = lifecycle["plan_envelope"]["plan"]
        proposal = plan["learned_proposal"]
        # Synthetic CPU proposal uses its own explicit test URDF; both native
        # endpoint pins retain the same description identity required by v4.
        xml_digest = plan["binding_digests"]["robot_description_digest"]
        for bindings in (source["binding_digests"], source["destination_binding_digests"]):
            bindings["robot_description_digest"] = xml_digest
        proposal["instruction"] = instruction["instruction"]
        evidence_fixture.redigest(proposal, "proposal_digest")
        program = compile_program(source, proposal)
        plan.update(learned_source_program=source, binding_digests=copy.deepcopy(source["binding_digests"]),
                    steps=program["steps"], motion_program_digest=canonical_digest(program))
        plan["scene_binding"] = {"scene_state_digest": canonical_digest("original-transition-scene"),
            "revision": 1, "object_instance_id": "cube", "release_slot": release_slot(
                robot_system_id=plan["robot_system_id"], pose=destination,
                object_profile_id=instruction["object_profile_id"],
                exclusion_geometry_digest=canonical_digest({"shape": "BOX", "dimensions_mm": [24., 24., 24.]}))}
        lifecycle["scene_binding"] = copy.deepcopy(plan["scene_binding"])
        lifecycle["execution_evidence"]["learned_execution"]["proposal_digest"] = proposal["proposal_digest"]
        evidence_fixture.AcquisitionRolloutTests.rebind(lifecycle)
        if scoped_terminal:
            lifecycle = evidence_fixture.scoped_terminal_lifecycle(lifecycle)
        receipt = {k: copy.deepcopy(resolved[k]) for k in ("normalized_job", "input_digests", "resolved_job_digest")}
        preapproval = {"schema_version": "data_factory.preapproval_evidence.v4", "run_id": lifecycle["run_id"],
            "plan_digest": lifecycle["plan_digest"], "resolved_job_digest": resolved["resolved_job_digest"],
            "plan_envelope": copy.deepcopy(lifecycle["plan_envelope"]),
            "plan_envelope_digest": canonical_digest(lifecycle["plan_envelope"]),
            "resolved_inputs": receipt, "destination_resolved_inputs": resolved["destination_resolved_inputs"],
            "episode_instruction_binding": instruction, "episode_instruction_binding_digest": instruction["binding_digest"]}
        root = self.root / "transition-rollout"
        root.mkdir()
        path = root / "learned_lifecycle_result.json"
        path.write_text(json.dumps(lifecycle))
        (root / "preapproval_evidence.json").write_text(json.dumps(preapproval))
        context = {"run_directories": runs, "scene_state_path": self.source["scene_state_path"],
                   "rollout_lifecycle_path": str(path)}
        self.app = self.application("transition-consumer", source=lambda: context, rollout=path)
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "selection": {"variant": "DIRECT"}})
        return lifecycle, preapproval, context

    def test_scoped_terminal_proposes_original_transition_validation(self):
        _lifecycle, preapproval, _context = self.transition_rollout(original_index=2, scoped_terminal=True)
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "requested_count": 4})
        advice = self.refresh()
        recommendation = advice["recommendation"]
        target = recommendation["input_snapshot"]["rollout_condition"]
        self.assertEqual(target["purpose"], "ORIGINAL_CONDITION_VALIDATION")
        self.assertEqual(target["reason_code"], "COMPLETED_SCOPED_ATTEMPT_LINEAGE")
        self.assertEqual(target["task_effectiveness"], "UNKNOWN")
        self.assertEqual(target["data_deficit"], "UNKNOWN")
        self.assertNotEqual(recommendation["object_poses"][0], target["source"])
        self.assertEqual(recommendation["object_poses"][2:4], [target["source"], target["destination"]])
        self.assertEqual(target["episode_instruction_binding"], preapproval["episode_instruction_binding"])
        self.assertIn("VALIDATE_ORIGINAL_TRANSITION", recommendation["reason_codes"])
        self.assertNotIn("HUMAN_REVIEWED_CHUNK_FAILURE", recommendation["reason_codes"])
        self.send(self.app, "choose_collection_advice", {
            "choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.send(self.app, "compile_draft", {
            "draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertEqual(self.app._campaign._episode_instruction_bindings[2], preapproval["episode_instruction_binding"])
        self.assertIsNone(self.app.projection()["campaign_authorization"])
        self.native.forbidden.assert_not_called()

    def test_original_transition_roundtrips_native_paired_draft_and_compile(self):
        lifecycle, preapproval, context = self.transition_rollout()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        advice = self.refresh()
        recommendation = copy.deepcopy(self.app._collection_advice["recommendation"])
        target = recommendation["input_snapshot"]["rollout_condition"]
        self.assertEqual(target["condition_indices"], [0])
        self.assertEqual(target["episode_instruction_binding"], preapproval["episode_instruction_binding"])
        self.assertEqual(recommendation["conditions"][0]["destination"], target["destination"])
        from tools.data_factory.collection_recommendation_io import _acquisition_context, _load_run
        from tools.data_factory.collection_recommendation import validate_collection_recommendation
        acquisition = {"catalog": self.app.catalog, "selection": self.app.selection,
            "scene_state_path": context["scene_state_path"],
            "expected_scene_digest": self.app.draft["object_position"]["scene_state_digest"], "object_instance_id": "cube",
            **{k: self.app.draft[k] for k in ("requested_count", "repeat", "normalized_seed", "motion_preset")}}
        args = dict(acquisition=_acquisition_context(acquisition), episode_evidence=[_load_run(r)[0] for r in context["run_directories"]],
                    rollout_evidence_analysis=lifecycle, rollout_preapproval_evidence=preapproval)
        self.assertEqual(validate_collection_recommendation(recommendation, **args), recommendation)
        forged = copy.deepcopy(recommendation)
        forged["conditions"][0]["destination"]["x_mm"] += 1
        evidence_fixture.redigest(forged, "recommendation_digest")
        with self.assertRaisesRegex(ContractError, "INPUT_CHANGED"):
            validate_collection_recommendation(forged, **args)
        self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.assertEqual(self.app.draft["authoring_mode"], "DIRECT_EDIT")
        self.assertIsNone(self.app.draft["direct_pairs"][-1]["start_pose_id"])
        self.assertEqual(self.app.projection()["collection_advice"]["status"], "APPLIED")
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertEqual(self.app._campaign._episode_instruction_bindings[0], preapproval["episode_instruction_binding"])
        self.assertIsNone(self.app.projection()["campaign_authorization"])
        self.native.forbidden.assert_not_called()

    def test_recovered_transition_keeps_current_pose_then_original_pair(self):
        lifecycle, preapproval, context = self.transition_rollout(original_index=2)
        scene_path = Path(context["scene_state_path"])
        current = copy.deepcopy(load_json_strict(scene_path)["objects"]["cube"]["pose"])
        self.send(self.app, "refresh_collection_advice")
        self.assertEqual(self.app._collection_advice["reason_codes"], ["COLLECTION_ACQUISITION_ROLLOUT_BUDGET_INSUFFICIENT"])
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "requested_count": 4})
        advice = self.refresh()
        target = advice["recommendation"]["input_snapshot"]["rollout_condition"]
        self.assertEqual(target["condition_indices"], [2])
        poses = advice["recommendation"]["object_poses"]
        self.assertEqual(poses[0], current)
        self.assertEqual(poses[2:4], [target["source"], target["destination"]])
        self.assertNotEqual(poses[0], poses[2])
        self.assertEqual(target["original_scene_binding"], preapproval["plan_envelope"]["plan"]["scene_binding"])
        # The existing update_draft pair intents can represent these exact N+1
        # poses, including the terminal destination's absent start pose.
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "authoring_mode": "DIRECT_EDIT"})
        for pair in list(reversed(self.app.draft["direct_pairs"][1:])):
            self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "remove_pair": pair})
        start = self.app.draft["selected_start_pose_ids"][0]
        for index, pose in enumerate(poses[1:], 1):
            self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"],
                "add_pair": {**pose, "start_pose_id": None if index == 4 else start}})
        self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertEqual(self.app._campaign._episode_instruction_bindings[2], preapproval["episode_instruction_binding"])
        self.assertEqual(load_json_strict(scene_path)["objects"]["cube"]["pose"], current)
        self.assertIsNone(self.app.projection()["campaign_authorization"])
        self.native.forbidden.assert_not_called()

    def test_transition_missing_and_retargeted_original_evidence_rejects(self):
        lifecycle, preapproval, context = self.transition_rollout()
        path = Path(context["rollout_lifecycle_path"])
        approval_path = path.with_name("preapproval_evidence.json")
        cases = {}
        missing = copy.deepcopy(preapproval); missing.pop("destination_resolved_inputs")
        cases["missing-destination"] = missing
        for field in ("x_mm", "y_mm", "yaw_deg", "cell_calibration_id", "sheet_manifest_digest"):
            changed = copy.deepcopy(preapproval)
            job = changed["destination_resolved_inputs"]["normalized_job"]
            job[field] = job[field] + 1 if isinstance(job[field], (float, int)) else (
                canonical_digest("changed") if field == "sheet_manifest_digest" else "changed")
            receipt = changed["destination_resolved_inputs"]
            receipt["resolved_job_digest"] = canonical_digest({"job": job, "input_digests": receipt["input_digests"]})
            cases[field] = changed
        changed = copy.deepcopy(preapproval)
        changed["plan_envelope"]["plan"]["learned_proposal"]["checkpoint"]["tree_digest"] = canonical_digest("other")
        changed["plan_envelope_digest"] = canonical_digest(changed["plan_envelope"])
        cases["other-plan-checkpoint"] = changed
        for endpoint_index in (0, 1):
            for field in ("family_digest", "region_binding"):
                changed = copy.deepcopy(preapproval)
                language = changed["episode_instruction_binding"]
                endpoint = language["task_binding"]["spatial_bindings"][endpoint_index]
                if field == "family_digest":
                    endpoint[field] = canonical_digest("different-family")
                else:
                    endpoint[field]["physical_binding_status"] = "NOT_CONFIGURED"
                    endpoint[field].update(layout_id=None, layout_digest=None, region_id=None)
                evidence_fixture.redigest(language["task_binding"], "binding_digest")
                from tools.data_factory.task_recipe import task_binding_instruction
                language["instruction"] = task_binding_instruction(language["task_binding"], language["object_description"])
                evidence_fixture.redigest(language, "binding_digest")
                changed["episode_instruction_binding_digest"] = language["binding_digest"]
                cases[f"endpoint-{endpoint_index}-{field}"] = changed
        changed = copy.deepcopy(preapproval)
        language = changed["episode_instruction_binding"]
        language["task_binding"]["spatial_bindings"].reverse()
        for endpoint, role in zip(language["task_binding"]["spatial_bindings"], ("SOURCE", "DESTINATION")):
            endpoint["role"] = role
        evidence_fixture.redigest(language["task_binding"], "binding_digest")
        language["instruction"] = task_binding_instruction(language["task_binding"], language["object_description"])
        evidence_fixture.redigest(language, "binding_digest")
        changed["episode_instruction_binding_digest"] = language["binding_digest"]
        cases["reversed-direction"] = changed
        changed = copy.deepcopy(preapproval)
        language = changed["episode_instruction_binding"]
        from tools.data_factory.task_recipe import compile_task_binding
        language["task_binding"] = compile_task_binding("pickup_e2e", source=language["task_binding"]["spatial_bindings"][0])
        language["instruction"] = task_binding_instruction(language["task_binding"], language["object_description"])
        evidence_fixture.redigest(language, "binding_digest")
        changed["episode_instruction_binding_digest"] = language["binding_digest"]
        cases["different-task"] = changed
        missing = copy.deepcopy(preapproval); missing.pop("episode_instruction_binding")
        cases["missing-instruction"] = missing
        draft = copy.deepcopy(self.app.draft)
        for name, changed in cases.items():
            with self.subTest(case=name):
                approval_path.write_text(json.dumps(changed))
                self.send(self.app, "refresh_collection_advice")
                self.assertEqual(self.app._collection_advice["status"], "UNAVAILABLE")
                self.assertEqual(self.app.draft, draft)
        approval_path.write_text(json.dumps(preapproval))
        self.refresh()
        self.native.forbidden.assert_not_called()

    def test_original_rollout_condition_applies_and_compiles_after_recovery(self):
        # Existing physical composition and compiler, synthetic stored outcomes;
        # no device, model, motion, recorder or approval callback is available.
        seed = self.application("pickup-source")
        self.send(seed, "update_draft", {"draft_id": seed.draft["draft_id"], "selection": {"task": "pickup_e2e"}})
        self.send(seed, "compile_draft", {"draft_id": seed.draft["draft_id"], "data_disposition": "PRODUCTION"})
        fixture, pickup_runs = self.store_evidence(seed, "pickup-evidence")
        lifecycle = evidence_fixture.learned_lifecycle(fixture, order=1, reviewed=True)
        plan = lifecycle["plan_envelope"]["plan"]
        preset = selected_motion_preset(seed.catalog, seed.draft["motion_preset"])
        qualification = preset["qualifications"][seed.selection["motion_id"]]["digest"]
        plan["binding_digests"]["motion_qualification"] = qualification
        plan["learned_source_program"]["binding_digests"]["motion_qualification"] = qualification
        plan["binding_digests"]["motion_preset"] = preset["digest"]
        plan["learned_source_program"]["binding_digests"]["motion_preset"] = preset["digest"]
        receipt = copy.deepcopy(next(r for r in fixture.hypothesis["resolver_receipts"]
                                    if r["resolved_job_digest"] == plan["resolved_job_digest"]))
        inputs = {key: plan["binding_digests"][key] for key in receipt["input_digests"]}
        resolved = canonical_digest({"job": receipt["normalized_job"], "input_digests": inputs})
        plan["resolved_job_digest"] = plan["learned_source_program"]["resolved_job_digest"] = resolved
        plan["scene_binding"] = {"scene_state_digest": canonical_digest("synthetic-original-scene"),
                                 "revision": 1, "object_instance_id": "cube"}
        lifecycle["scene_binding"] = copy.deepcopy(plan["scene_binding"])
        evidence_fixture.AcquisitionRolloutTests.rebind(lifecycle)
        root = self.root / "outputs/data_factory/runs" / lifecycle["run_id"]
        root.mkdir()
        path = root / "learned_lifecycle_result.json"
        path.write_text(json.dumps(lifecycle))
        preapproval = {"schema_version": "data_factory.preapproval_evidence.v4",
            "run_id": lifecycle["run_id"], "plan_digest": lifecycle["plan_digest"],
            "resolved_job_digest": resolved, "plan_envelope": lifecycle["plan_envelope"],
            "plan_envelope_digest": canonical_digest(lifecycle["plan_envelope"]),
            "resolved_inputs": {"normalized_job": receipt["normalized_job"],
                                "input_digests": inputs, "resolved_job_digest": resolved}}
        (root / "preapproval_evidence.json").write_text(json.dumps(preapproval))
        source = {"run_directories": pickup_runs, "scene_state_path": self.source["scene_state_path"],
                  "rollout_lifecycle_path": str(path)}
        self.app = self.application("pickup-recollect", source=lambda: source, rollout=path)
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "selection": {"task": "pickup_e2e"}})
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "normalized_seed": 987})
        original_pose = {k: receipt["normalized_job"][k] for k in ("place_id", "x_mm", "y_mm", "yaw_deg")}
        self.assertNotEqual(original_pose, self.app.draft["current_object_pose"])
        before = self.native.scene.read()["objects"], self.native.cell.read(), path.read_bytes()
        self.assertEqual(qualification, selected_motion_preset(self.app.catalog, self.app.draft["motion_preset"])
                         ["qualifications"][self.app.selection["motion_id"]]["digest"])
        advice = self.refresh()
        self.assertEqual(advice["recommendation"]["sampling"]["authoring_mode"], "DIRECT_EDIT")
        self.assertEqual(advice["recommendation"]["object_poses"], [position_fixture.POSE, original_pose])
        retained = path.read_bytes()
        unchanged_draft = copy.deepcopy(self.app.draft)
        path.write_text("{}")
        with self.assertRaisesRegex(ContractError, "COLLECTION_ADVICE_STALE"):
            self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.assertEqual(self.app.draft, unchanged_draft)
        path.write_bytes(retained)
        self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.assertEqual(self.app.draft["authoring_mode"], "DIRECT_EDIT")
        self.assertEqual(self.app.projection()["collection_advice"]["status"], "APPLIED")
        path.write_text("{}")
        with self.assertRaisesRegex(ContractError, "COLLECTION_ADVICE_STALE"):
            self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertIsNone(self.app._campaign)
        path.write_bytes(retained)
        self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        owner = self.app._campaign.campaign_operator
        conditions = {b["base_condition_digest"]: b["coverage_condition"] for b in owner.hypothesis["base_conditions"]}
        self.assertEqual([{k: conditions[s["base_condition_digest"]][k] for k in original_pose}
                          for s in owner.manifest["slots"]], [position_fixture.POSE, original_pose])
        self.assertIsNone(self.app.projection()["campaign_authorization"])
        self.assertEqual((self.native.scene.read()["objects"], self.native.cell.read(), path.read_bytes()), before)
        self.native.forbidden.assert_not_called()

    def test_native_evidence_apply_and_compile_preserve_pose_and_authority(self):
        from tools.data_factory import collection_recommendation_io as io
        probe = mock.patch.object(io, "discover_stored_collection", wraps=io.discover_stored_collection)
        discovery = probe.start()
        self.addCleanup(probe.stop)
        before = self.native.scene.snapshot(), self.native.cell.read(), self.native.episode.read_bytes()
        self.app.core.snapshot()
        self.assertEqual(discovery.call_count, 0)
        advice = self.refresh()
        self.assertEqual(discovery.call_count, 1)
        expected = copy.deepcopy(self.app._collection_advice["recommendation"])
        self.assertEqual(expected["observed_semantic_pass_by_source"], {"PLACE_A": 1, "PLACE_B": 1})
        self.assertEqual(expected["object_poses"][0], position_fixture.POSE)
        self.assertEqual(len(expected["object_poses"]), 3)
        self.assertNotIn(str(self.root), json.dumps(advice))
        retained = copy.deepcopy(self.app._collection_advice)
        self.app._collection_advice.pop("mode")
        self.app._collection_advice["status"] = "UNAVAILABLE"
        self.assertNotIn(str(self.root), json.dumps(self.app._advice_projection()))
        self.app._collection_advice = retained
        self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.assertIsNone(self.app._campaign)
        self.assertEqual(self.app.draft["authoring_mode"], "ASSISTED")
        self.assertEqual(self.app.projection()["collection_advice"]["status"], "APPLIED")
        self.assertEqual(discovery.call_count, 2)
        self.assertEqual((self.native.scene.snapshot(), self.native.cell.read(), self.native.episode.read_bytes()), before)
        self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertEqual(self.app.projection()["workflow_state"], "REVIEW_CAMPAIGN")
        self.assertEqual(len(self.app._campaign.campaign_operator.manifest["slots"]), 2)
        self.assertIsNone(self.app.projection()["campaign_authorization"])
        self.assertEqual(discovery.call_count, 3)
        # The existing compiler may rebind the released slot to the new run;
        # it must retain the original release and exact object position.
        self.assertEqual(self.native.scene.read()["objects"], before[0]["scene_state"]["objects"])
        self.assertEqual(self.native.cell.read(), before[1])
        self.assertEqual(self.native.episode.read_bytes(), before[2])
        self.native.forbidden.assert_not_called()

    def test_accepted_advice_revalidates_catalog_and_stored_evidence_before_compile(self):
        advice = self.refresh()
        before = copy.deepcopy(self.app.draft)
        with mock.patch.object(self.app, "catalog_reload_call", return_value={"catalog_digest": canonical_digest("changed-catalog")}):
            with self.assertRaisesRegex(ContractError, "COLLECTION_ADVICE_STALE"):
                self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.assertEqual(self.app.draft, before)
        self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        # Corrupt only a synthetic stored candidate; discovery must exclude it.
        (self.runs[1] / "candidate_admission.json").write_text("{}")
        before = self.native.scene.snapshot(), self.native.cell.read()
        with self.assertRaisesRegex(ContractError, "COLLECTION_ADVICE_STALE"):
            self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertEqual((self.native.scene.snapshot(), self.native.cell.read()), before)
        self.assertIsNone(self.app._campaign)
        self.native.forbidden.assert_not_called()

    def test_edits_constraints_and_changed_source_cannot_overwrite_draft(self):
        advice = self.refresh()
        payload = {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]}
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "normalized_seed": 88})
        edited = copy.deepcopy(self.app.draft)
        with self.assertRaisesRegex(ContractError, "OPERATOR_INTENT_OP"):
            self.send(self.app, "choose_collection_advice", payload)
        self.assertEqual(self.app.draft, edited)
        fresh = self.refresh()
        self.assertNotEqual(fresh["recommendation_digest"], advice["recommendation_digest"])
        from tools.data_factory.operator.workflow.collection_advice import derive_next_draft
        for constraint in ({"pinned": ["existing-pin"]}, {"excluded": ["existing-exclusion"]},
                           {"state_space_design_profile": {"custom": "preserved"}}):
            constrained = {**copy.deepcopy(self.app.draft), **constraint}
            checked, candidate = derive_next_draft(self.source, catalog=self.app.catalog, selection=self.app.selection,
                                                  draft=constrained, paired=True)
            self.assertEqual(checked["reason_codes"], ["COLLECTION_ADVICE_CONSTRAINED_DRAFT"])
            self.assertIsNone(candidate)
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "authoring_mode": "DIRECT_EDIT"})
        edited = copy.deepcopy(self.app.draft)
        self.send(self.app, "refresh_collection_advice")
        self.assertEqual(self.app.projection()["collection_advice"]["reason_codes"], ["COLLECTION_ADVICE_CONSTRAINED_DRAFT"])
        self.assertEqual(self.app.draft, edited)
        self.send(self.app, "update_draft", {"draft_id": self.app.draft["draft_id"], "authoring_mode": "ASSISTED"})
        advice = self.refresh()
        self.native.cell.mark_blocked("EXECUTION_IN_PROGRESS", "newer-motion", canonical_digest("newer-plan"))
        with self.assertRaisesRegex(ContractError, "COLLECTION_ADVICE_STALE"):
            self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        self.send(self.app, "refresh_collection_advice")
        self.assertEqual(self.app.projection()["collection_advice"]["reason_codes"], ["COLLECTION_ADVICE_SOURCE_CHANGED"])
        self.assertIsNone(self.app._campaign)
        self.native.forbidden.assert_not_called()

    def test_native_compiler_mismatch_rejects_before_campaign_or_source_effects(self):
        from tools.data_factory.operator import composition
        advice = self.refresh()
        self.send(self.app, "choose_collection_advice", {"choice": "APPLY", "expected_recommendation_digest": advice["recommendation_digest"]})
        before = self.native.scene.snapshot(), self.native.cell.read()
        sampler = composition.project_workspace_cycle_poses
        def changed(*args, **kwargs):
            poses = sampler(*args, **kwargs)
            poses[1]["x_mm"] += .1
            return poses
        with mock.patch.object(composition, "project_workspace_cycle_poses", side_effect=changed):
            with self.assertRaisesRegex(ContractError, "COLLECTION_ADVICE_COMPILER_MISMATCH"):
                self.send(self.app, "compile_draft", {"draft_id": self.app.draft["draft_id"], "data_disposition": "PRODUCTION"})
        self.assertIsNone(self.app._campaign)
        self.assertEqual((self.native.scene.snapshot(), self.native.cell.read()), before)
        self.native.forbidden.assert_not_called()

    def test_shipped_ui_apply_keep_response_loss_and_no_post_replay(self):
        for choice in ("APPLY", "KEEP"):
            app = self.app if choice == "APPLY" else self.application("keep-consumer")
            bridge = LoopbackBridge(core=app.core, ui_root=evidence_fixture.ROOT / "operator-ui", port=0)
            thread = threading.Thread(target=bridge.serve_forever)
            thread.start()
            try:
                process = subprocess.run(["node", "operator-ui/tests/collection-advice-recovery.cjs", bridge.origin,
                                          "operator-ui/app.js", choice], cwd=evidence_fixture.ROOT,
                                         capture_output=True, text=True, timeout=60)
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads(process.stdout)
                self.assertEqual([item["method"] for item in result["requests"]], ["POST", "GET"])
                advice = result["canonical"]["projection"]["collection_advice"]
                self.assertEqual(advice["status"], "APPLIED" if choice == "APPLY" else "KEPT")
                self.assertIn("수집 제안", result["conditions"])
                self.assertIn("→", result["conditions"])
                self.assertNotIn(str(self.root), json.dumps(advice))
                self.assertTrue(result["applyHidden"])
                with self.assertRaisesRegex(ContractError, "OPERATOR_INTENT_OP"):
                    self.send(app, "choose_collection_advice", {"choice": "KEEP" if choice == "APPLY" else "APPLY",
                                                               "expected_recommendation_digest": advice["recommendation_digest"]})
                self.assertIsNone(app._campaign)
            finally:
                bridge.close()
                thread.join(5)
        self.native.forbidden.assert_not_called()


class AcquisitionLaunchTests(unittest.TestCase):
    def test_cli_forwards_explicit_original_source_without_starting_runtime(self):
        from tools.data_factory.operator import cli
        with mock.patch.object(cli, "_serve", return_value=0) as serve:
            self.assertEqual(cli.main(["--effect-scope", "PHYSICAL", "--no-auto-prepare",
                                      "--rollout-lifecycle", "/synthetic/learned_lifecycle_result.json"]), 0)
        self.assertEqual(serve.call_args.kwargs["rollout_lifecycle"], Path("/synthetic/learned_lifecycle_result.json"))
        self.assertFalse(serve.call_args.kwargs["auto_prepare"])

    def test_rollout_source_cannot_silently_enter_an_unrelated_runtime(self):
        from tools.data_factory.operator.composition import build_operator_runtime
        for scope in ("FAKE", "LEARNED_RUN", "TRAINING_REVIEW", "CURATOR_REVIEW"):
            with self.subTest(scope=scope), self.assertRaisesRegex(ContractError, "COLLECTION_ROLLOUT_CONFIGURATION"):
                build_operator_runtime(effect_scope=scope, rollout_lifecycle="/synthetic/not-read.json")
