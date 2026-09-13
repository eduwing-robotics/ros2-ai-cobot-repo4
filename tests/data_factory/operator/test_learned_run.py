"""Scoped learned-run task wording is derived without runtime effects."""
from pathlib import Path
import unittest
from unittest import mock

from tools.data_factory.operator.workflow.learned_run import (
    LearnedRunApplication,
    compile_learned_episode_instruction,
)
from tools.fr5_data_factory import ContractError, canonical_digest, load_json_strict


ROOT = Path(__file__).resolve().parents[3]


def request():
    config = ROOT / "config/data_factory"
    source = {
        "place_id": "PLACE_A",
        "x_mm": 28.81496713705198,
        "y_mm": 12.825932993746031,
        "yaw_deg": 35.89085497339096,
    }
    destination = {
        "place_id": "PLACE_B",
        "x_mm": 19.51301470772826,
        "y_mm": 53.74217821394695,
        "yaw_deg": 35.89085497339096,
    }
    sheet_a = config / "test_only_physical/goal2-place1/yaw0_sheet.json"
    sheet_b = config / "workspace_sheets/place-b-yaw0-r001_yaw0_sheet.json"
    base = load_json_strict(
        config / "jobs/center-live-24mm-20260903-r002.job.json"
    )
    job = {
        **base,
        **source,
        "job_id": "scoped-lineage-test",
        "task": "pick_place",
        "instruction": "pick up the 24 mm wooden cube and place it at the destination",
        "episode_intent": "nominal pick and place",
    }
    destination_job = {
        **job,
        **destination,
        "cell_calibration_id": "place-b-yaw0-r001",
        "sheet_manifest_digest": canonical_digest(load_json_strict(sheet_b)),
    }
    preset = load_json_strict(
        config / "motion_presets/demonstration-rhythm-r001.json"
    )
    return {
        "mode": "live",
        "run_id": "scoped-lineage-test",
        "job": job,
        "selected_sheet": str(sheet_a),
        "yaw0_sheet": str(sheet_a),
        "config_root": str(config),
        "motion_qualification": str(
            config / "motion_qualifications/fr5-place-a-wood-cube-24mm-r001-demonstration-rhythm-r001.json"
        ),
        "home_candidate": str(
            config / "home_candidates/fr5-lab-a-tcp-r002-home-r001.json"
        ),
        "urdf": str(ROOT / "src/fairino_description/urdf/fairino5_v6.urdf"),
        "expected_robot_system_id": "fr5-lab-a-tcp-r002",
        "destination": {
            "job": destination_job,
            "selected_sheet": str(sheet_b),
            "yaw0_sheet": str(sheet_b),
            "motion_qualification": str(
                config / "motion_qualifications/fr5-place-b-wood-cube-24mm-r001-demonstration-rhythm-r001.json"
            ),
        },
        "motion_preset": {
            "id": preset["motion_preset_id"],
            "digest": canonical_digest(preset),
        },
        "trajectory_variant_id": "TWO_STAGE_ALIGN_V2",
        "trajectory_sampling_seed": 0,
        "camera_profile": "up-wrist",
        "dataset_root": "/unused-dataset",
        "run_root": "/unused-runs",
        "learned_checkpoint": "/unused-checkpoint",
        "gripper_temporal_policy": "/unused-policy",
        "learned_device": "cuda",
        "learned_reference_mode": "serialized_percent_retime",
    }


def grant(instruction):
    value = {
        "schema_version": "data_factory.learned_task_grant.v1",
        "grant_id": "scoped-lineage-grant",
        "issued_by": "test-owner",
        "run_id": "scoped-lineage-test",
        "scope": {
            "task": instruction,
            "adaptation": {
                "schema_version": "data_factory.finite_learned_serialized_reference_proposal.v2"
            },
        },
        "deadline_s": 9999999999.0,
        "terminal_reserve_s": 30.0,
        "max_outputs": 1,
        "revoked": False,
    }
    value["grant_digest"] = canonical_digest(value)
    return value


class LearnedRunLineageTests(unittest.TestCase):
    def test_compiler_derives_current_red_to_blue_binding(self):
        binding = compile_learned_episode_instruction(
            request(), repository_root=ROOT,
        )
        source, destination = binding["task_binding"]["spatial_bindings"]
        self.assertEqual(
            binding["instruction"],
            "pick up the 24 mm wooden cube from the red zone and place it in the blue zone",
        )
        self.assertEqual((source["workspace_id"], source["region_binding"]["region_id"]), ("PLACE_A", "RED"))
        self.assertEqual((destination["workspace_id"], destination["region_binding"]["region_id"]), ("PLACE_B", "BLUE"))
        self.assertEqual(source["region_binding"]["physical_binding_status"], "PREPARED_NOT_VERIFIED")

    def test_compiler_rejects_wrong_frame_sheet_and_region(self):
        for mutation in ("frame", "sheet"):
            value = request()
            if mutation == "frame":
                value["destination"]["job"]["cell_calibration_id"] = "place-a-yaw0-r003"
            else:
                value["destination"]["selected_sheet"] = value["selected_sheet"]
                value["destination"]["yaw0_sheet"] = value["yaw0_sheet"]
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                compile_learned_episode_instruction(value, repository_root=ROOT)

        value = request()
        from tools.data_factory.operator.registries.region import load_workspace_region_binding
        from tools.a4_place_yaw.region_layout import make_red_blue_region_layout
        persisted = load_workspace_region_binding(ROOT, make_red_blue_region_layout())
        persisted["bindings"][1]["region_id"] = "RED"
        with mock.patch(
            "tools.data_factory.operator.workflow.learned_run.load_workspace_region_binding",
            return_value=persisted,
        ), self.assertRaisesRegex(ContractError, "EPISODE_INSTRUCTION_SCOPE"):
            compile_learned_episode_instruction(value, repository_root=ROOT)

    def test_scoped_application_passes_binding_and_rejects_mismatched_task(self):
        value = request()
        binding = compile_learned_episode_instruction(value, repository_root=ROOT)
        value["task_grant"] = grant(binding["instruction"])
        calls = []

        def run_live(*_args, **kwargs):
            calls.append(kwargs)
            return {"data": None}

        application = LearnedRunApplication(
            payload=value,
            operator_label="local-operator",
            repository_root=ROOT,
            run_live_call=run_live,
        )
        application._run()
        self.assertEqual(calls[0]["episode_instruction_binding"], binding)
        self.assertEqual(calls[0]["repository_root"], ROOT)

        value = request()
        value["task_grant"] = grant("generic destination wording")
        run = mock.Mock()
        with self.assertRaisesRegex(ContractError, "TASK_GRANT_SCOPE"):
            LearnedRunApplication(
                payload=value,
                operator_label="local-operator",
                repository_root=ROOT,
                run_live_call=run,
            )
        run.assert_not_called()

    def test_legacy_application_passes_no_episode_binding(self):
        calls = []
        value = request()

        def run_live(*_args, **kwargs):
            calls.append(kwargs)
            return {"data": None}

        application = LearnedRunApplication(
            payload=value,
            operator_label="local-operator",
            repository_root=ROOT,
            run_live_call=run_live,
        )
        application._run()
        self.assertNotIn("episode_instruction_binding", calls[0])
        self.assertEqual(application.state, "TERMINAL")


if __name__ == "__main__":
    unittest.main()
