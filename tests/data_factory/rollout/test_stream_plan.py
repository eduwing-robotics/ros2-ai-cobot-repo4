import copy
import hashlib
import unittest

from tools.data_factory.rollout.finite_plan import JOINTS
from tools.data_factory.rollout.stream_plan import (
    PLAN_SCHEMA,
    POLICY_SCHEMA,
    build_stream_plan,
    validate_stream_plan,
)
from tools.fr5_data_factory import ContractError, canonical_digest, validate_motion_program
from tests.data_factory.operator.fixtures import motion


XML = '<robot name="synthetic">' + ''.join(
    f'<joint name="{name}" type="{"prismatic" if index == 6 else "revolute"}">'
    f'<limit lower="{0 if index == 6 else -3}" upper="{.02 if index == 6 else 3}" '
    f'velocity="{.1 if index == 6 else 10}"/></joint>'
    for index, name in enumerate(JOINTS)
) + '</robot>'


def source_program():
    source = motion()
    source["binding_digests"]["robot_description_digest"] = (
        "sha256:" + hashlib.sha256(XML.encode()).hexdigest()
    )
    destination = copy.deepcopy(source["binding_digests"])
    destination.update(
        cell_calibration=canonical_digest("cell-b"),
        motion_qualification=canonical_digest("qualification-b"),
    )
    endpoints = [
        {
            "workspace_id": workspace,
            "cell_calibration_id": calibration,
            "cell_calibration_digest": bindings["cell_calibration"],
            "motion_recipe_digest": bindings["motion_qualification"],
        }
        for workspace, calibration, bindings in (
            ("PLACE_A", "cal-a", source["binding_digests"]),
            ("PLACE_B", "cal-b", destination),
        )
    ]
    source.update(
        schema_version="fr5.motion_program.v4",
        destination_resolved_job_digest=canonical_digest("destination-job"),
        destination_binding_digests=destination,
        endpoint_bindings=endpoints,
        endpoint_bindings_digest=canonical_digest(endpoints),
    )
    for step in source["steps"]:
        if step.get("pause_after") == "SEMANTIC_VERDICT":
            step.pop("pause_after")
        if step["phase"] == "RETREAT_LIN":
            step["pause_after"] = "SEMANTIC_VERDICT"
    return validate_motion_program(source)


def policy():
    instruction = "Pick the red block and place it in the blue region."
    temporal = {
        "schema_version": "fr5.gripper_temporal_policy.v2",
        "incarnation": [1, 2, 3, 4],
        "connection_epoch": 1,
        "configuration_epoch": 0,
        "max_age_s": .1,
        "host_clock_tolerance_s": .001,
    }
    runtime = {
        "checkpoint": "/models/original-12k",
        "device": "cpu",
        "gripper_temporal_policy": "/runtime/gripper-policy.json",
        "clock_binding": temporal,
        "camera_topics": {"camera1": "/camera/up", "camera2": "/camera/wrist"},
        "camera_mapping": {
            "observation.images.up": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
        "fps": 30,
        "hardware_wire_version": 5,
        "warmup": {
            "input_kind": "SYNTHETIC_ZERO_RGB_STATE",
            "image_shape": [480, 640, 3],
            "instruction_digest": canonical_digest(instruction),
            "device": "cpu",
            "model_calls": 1,
            "output_disposition": "DISCARDED",
            "rng_state_restored": True,
            "duration_s": .2,
            "inference_duration_s": .1,
        },
    }
    return {
        "schema_version": POLICY_SCHEMA,
        "checkpoint": {
            "tree_digest": canonical_digest("weights"),
            "training_receipt_digest": canonical_digest("training"),
            "runtime": "SYNTHETIC_TEST_ONLY",
        },
        "instruction": instruction,
        "robot_description": XML,
        "velocity_scaling": .1,
        "period_s": 1 / 30,
        "max_observation_age_s": .3,
        "runtime_inputs": runtime,
    }


def scene_binding():
    return {
        "scene_state_digest": canonical_digest("scene"),
        "revision": 7,
        "object_instance_id": "red-block-1",
    }


class StreamPlanTest(unittest.TestCase):
    def test_builds_detached_static_native_task_identity(self):
        source, settings, scene = source_program(), policy(), scene_binding()
        plan = build_stream_plan("normal-run-1", source, scene, settings)

        self.assertEqual(
            set(plan),
            {"schema_version", "run_id", "source_program", "resolved_job_digest", "scene_binding", "policy"},
        )
        self.assertEqual(plan["schema_version"], PLAN_SCHEMA)
        self.assertEqual(plan["resolved_job_digest"], source["resolved_job_digest"])
        self.assertEqual(plan["source_program"]["endpoint_bindings"], source["endpoint_bindings"])
        self.assertNotIn("actions", plan["policy"])
        self.assertNotIn("learned_proposal", plan)

        source["endpoint_bindings"][0]["workspace_id"] = "CHANGED"
        settings["checkpoint"]["runtime"] = "CHANGED"
        scene["revision"] = 8
        self.assertEqual(plan, validate_stream_plan(plan))
        self.assertEqual(plan["scene_binding"]["revision"], 7)

        candidate = copy.deepcopy(plan)
        checked = validate_stream_plan(candidate)
        candidate["source_program"]["endpoint_bindings"][0]["workspace_id"] = "CHANGED"
        candidate["policy"]["checkpoint"]["runtime"] = "CHANGED"
        self.assertEqual(checked, plan)

    def test_rejects_noncanonical_or_rebound_source(self):
        source, settings, scene = source_program(), policy(), scene_binding()
        cases = []
        legacy = copy.deepcopy(source)
        for key in (
            "destination_resolved_job_digest", "destination_binding_digests",
            "endpoint_bindings", "endpoint_bindings_digest",
        ):
            legacy.pop(key)
        legacy["schema_version"] = "fr5.motion_program.v2"
        cases.append((legacy, settings, "LEARNED_STREAM_SOURCE_PROGRAM"))
        wrong_robot = copy.deepcopy(settings)
        wrong_robot["robot_description"] = XML.replace('name="synthetic"', 'name="other"')
        cases.append((source, wrong_robot, "LEARNED_ROBOT_BINDING"))

        for candidate_source, candidate_policy, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(ContractError, code):
                build_stream_plan("normal-run-1", candidate_source, scene, candidate_policy)

        plan = build_stream_plan("normal-run-1", source, scene, settings)
        plan["resolved_job_digest"] = canonical_digest("other-job")
        with self.assertRaisesRegex(ContractError, "LEARNED_STREAM_SOURCE_PROGRAM"):
            validate_stream_plan(plan)

    def test_rejects_finite_rows_and_changed_static_limits(self):
        source, settings, scene = source_program(), policy(), scene_binding()
        changes = (
            ("actions", [[0.] * 7], "LEARNED_STREAM_POLICY_SCHEMA"),
            ("schema_version", "data_factory.finite_learned_proposal.v2", "LEARNED_STREAM_POLICY_SCHEMA"),
            ("period_s", .01, "LEARNED_STREAM_POLICY_LIMITS"),
            ("velocity_scaling", .2, "LEARNED_STREAM_POLICY_LIMITS"),
            ("max_observation_age_s", .4, "LEARNED_STREAM_POLICY_LIMITS"),
        )
        for key, value, code in changes:
            with self.subTest(key=key), self.assertRaisesRegex(ContractError, code):
                changed = copy.deepcopy(settings)
                changed[key] = value
                build_stream_plan("normal-run-1", source, scene, changed)

        unhashable_runtime = copy.deepcopy(settings)
        unhashable_runtime["checkpoint"]["runtime"] = []
        with self.assertRaisesRegex(ContractError, "LEARNED_CHECKPOINT_BINDING"):
            build_stream_plan("normal-run-1", source, scene, unhashable_runtime)

    def test_rejects_runtime_rebinding_and_nonfinite_values(self):
        source, settings, scene = source_program(), policy(), scene_binding()
        for mutate, code in (
            (lambda value: value.update(runtime_inputs=None), "LEARNED_RUNTIME_INPUTS"),
            (lambda value: value["runtime_inputs"].update(fps=29), "LEARNED_RUNTIME_INPUTS"),
            (lambda value: value["runtime_inputs"].update(reference_mode="serialized_retime"), "LEARNED_RUNTIME_INPUTS"),
            (lambda value: value["runtime_inputs"]["warmup"].update(instruction_digest=canonical_digest("other")), "LEARNED_WARMUP_INPUT"),
            (lambda value: value.update(period_s=10**400), "LEARNED_HORIZON"),
        ):
            with self.subTest(code=code), self.assertRaisesRegex(ContractError, code):
                changed = copy.deepcopy(settings)
                mutate(changed)
                build_stream_plan("normal-run-1", source, scene, changed)

        plan = build_stream_plan("normal-run-1", source, scene, settings)
        plan["revision"] = {"rows": [[0.] * 7]}
        with self.assertRaisesRegex(ContractError, "LEARNED_STREAM_PLAN_SCHEMA"):
            validate_stream_plan(plan)

    def test_runtime_inputs_are_optional_static_deployment_binding(self):
        settings = policy()
        settings.pop("runtime_inputs")
        plan = build_stream_plan(
            "normal-run-1", source_program(), scene_binding(), settings,
        )
        self.assertNotIn("runtime_inputs", plan["policy"])


if __name__ == "__main__":
    unittest.main()
