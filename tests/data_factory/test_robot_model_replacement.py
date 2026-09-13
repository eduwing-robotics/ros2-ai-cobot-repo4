"""Opening-coordinate candidate: native CPU compatibility, not contact qualification."""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import threading
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from fr5_lerobot_recorder import FR5LeRobotRecorder
from tools.fr5_data_factory import ContractError, canonical_digest, validate_home_candidate
from tools.data_factory.learned_action_adapter import fake_rgb
from tools.data_factory.rollout.finite_plan import (
    FinitePolicyInference,
    JOINTS,
    _limits,
    build_robot_model_trial,
    compile_program,
)
from tests.data_factory.operator.fixtures import motion

ORIGINAL = ROOT / "src/fairino_description/urdf/fairino5_v6.urdf"
CANDIDATE = ORIGINAL.with_name("fairino5_v6_gripper_opening_candidate.urdf")
ORIGINAL_SHA = "a9108b594739b64eac42a39d9aa961ece688ffd8d06ff5af03df99cbf63ab345"


def finger_geometry(root, q):
    """Evaluate these unrotated parallel box fingers; not a physical estimator."""
    centers, widths = [], []
    for side in ("right", "left"):
        joint = root.find(f"./joint[@name='finger_{side}_joint']")
        origin = joint.find("origin")
        assert origin.get("rpy") == "0 0 0"
        xyz = list(map(float, origin.get("xyz").split()))
        axis = list(map(float, joint.find("axis").get("xyz").split()))
        assert axis[1:] == [0., 0.]
        mimic = joint.find("mimic")
        position = q if mimic is None else q * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
        collision = root.find(f"./link[@name='finger_tip_{side}_link']/collision")
        assert collision.find("origin").get("rpy") == "0 0 0"
        centers.append(xyz[0] + axis[0] * position + float(collision.find("origin").get("xyz").split()[0]))
        widths.append(float(collision.find("geometry/box").get("size").split()[0]))
    return centers[0] - centers[1] - sum(widths) / 2, sum(centers) / 2


class RobotModelReplacementTest(unittest.TestCase):
    def test_native_expansion_binds_full_model_not_xml_format_or_control_settings(self):
        from tools.data_factory.motion.contact_transition import bound_robot_description
        source = CANDIDATE.read_text()
        plan = {"learned_source_program": {"binding_digests": {
            "robot_description_digest": "sha256:" + hashlib.sha256(source.encode()).hexdigest()}},
            "learned_proposal": {"robot_description": source}}
        root = ET.fromstring(source)
        control = ET.SubElement(root, "ros2_control", name="FR5System", type="system")
        ET.SubElement(ET.SubElement(control, "hardware"), "plugin").text = "fairino_hardware/FairinoHardwareInterface"
        ET.indent(root)
        xml = ET.tostring(root, encoding="unicode")
        self.assertEqual(bound_robot_description(SimpleNamespace(_robot_description=xml), plan), xml)
        self.assertEqual(bound_robot_description(SimpleNamespace(_robot_description=source), plan), source)
        for fault in ("arm", "finger", "collision", "tcp", "missing_joint", "extra_control", "root", "source"):
            with self.subTest(fault=fault):
                changed, proposal = copy.deepcopy(root), copy.deepcopy(plan)
                if fault == "arm": changed.find("joint[@name='j1']/limit").set("upper", "1.2")
                if fault == "finger": changed.find("joint[@name='finger_right_joint']/axis").set("xyz", "-1 0 0")
                if fault == "collision": changed.find("link[@name='finger_tip_left_link']/collision/geometry/box").set("size", "0.001 0.001 0.001")
                if fault == "tcp": changed.find("joint[@name='gripper_adapter_joint']/origin").set("xyz", "0 0 0.199")
                if fault == "missing_joint": changed.remove(changed.find("joint[@name='finger_left_joint']"))
                if fault == "extra_control": changed.append(copy.deepcopy(control))
                if fault == "root": changed.set("name", "other_robot")
                if fault == "source": proposal["learned_proposal"]["robot_description"] = ORIGINAL.read_text()
                with self.assertRaisesRegex(ContractError, "CONTACT_MODEL_BINDING"):
                    bound_robot_description(SimpleNamespace(_robot_description=ET.tostring(changed, encoding="unicode")), proposal)

    def test_successor_qualifications_resolve_normal_contact_without_trial_bypass(self):
        from tools.data_factory.run_job import resolve_inputs
        from tools.data_factory.motion.contact_transition import prepare
        config = ROOT / "config/data_factory"
        model = ORIGINAL.with_name("fairino5_v6_gripper_opening_r001.urdf")
        self.assertEqual(ET.tostring(ET.parse(model).getroot()),
                         ET.tostring(ET.parse(CANDIDATE).getroot()))
        home = config / "home_candidates/fr5-lab-a-tcp-r002-home-r002-opening-coordinate.json"
        home_value = json.loads(home.read_text())
        result = validate_home_candidate(home_value, urdf=model,
            expected_robot_system_id=home_value["robot_system_id"])
        self.assertFalse(result["motion_allowed"])
        preset = json.loads((config / "motion_presets/demonstration-rhythm-r001.json").read_text())
        for side, calibration in (("a", "place-a-yaw0-r003"), ("b", "place-b-yaw0-r001")):
            with self.subTest(side=side):
                prefix = f"fr5-place-{side}-wood-cube-24mm-r001-demonstration-rhythm-r001"
                old_path = config / "motion_qualifications" / (prefix + ".json")
                new_path = old_path.with_name(prefix + "-opening-coordinate-r001.json")
                old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
                self.assertEqual({k for k in old if old[k] != new[k]}, {
                    "motion_qualification_id", "home_candidate_digest", "robot_description_digest", "qualified_at"})
                self.assertEqual(new["qualification_status"], "QUALIFIED")
                self.assertEqual(new["home_candidate_digest"], canonical_digest(home_value))
                cell = json.loads((config / "cells" / (calibration + ".json")).read_text())
                job = json.loads((config / "jobs/center-live-24mm-20260903-r002.job.json").read_text())
                job.update(place_id=cell["place_id"], cell_calibration_id=calibration,
                           sheet_manifest_digest=cell["yaw0_manifest_digest"])
                sheet = config / "workspace_sheets" / (calibration + "_yaw0_sheet.json")
                payload = dict(job=job, config_root=str(config), selected_sheet=str(sheet), yaw0_sheet=str(sheet),
                    motion_qualification=str(new_path), home_candidate=str(home), urdf=str(model),
                    expected_robot_system_id=job["robot_system_id"], run_id="cpu-coordinate-binding",
                    motion_preset={"id": preset["motion_preset_id"], "digest": canonical_digest(preset)})
                # Pure normal resolver; synthetic scene projection, no store or process.
                _, source, scene = resolve_inputs(payload, scene_binding_call=lambda *_: {})
                self.assertNotIn("robot_model_trial", source)
                pose = {k: job[k] for k in ("place_id", "x_mm", "y_mm", "yaw_deg")}
                contact = prepare(SimpleNamespace(_robot_description=model.read_text(), contact_config_root=config),
                    {"learned_source_program": source, "scene_binding": scene},
                    {"state": "ON_SURFACE", "object_profile_id": job["object_profile_id"], "pose": pose})
                self.assertEqual(contact["status"], "PROSPECTIVE")
                self.assertEqual(contact["qualification"], new)
                with self.assertRaisesRegex(ContractError, "HOME_ROBOT_BINDING"):
                    resolve_inputs({**payload, "urdf": str(ORIGINAL)}, scene_binding_call=lambda *_: {})

    def test_place_a_contact_sheet_resolves_existing_calibration_identity(self):
        from tools.data_factory.motion.contact_transition import bound_document
        config = ROOT / "config/data_factory"
        cell = json.loads((config / "cells/place-a-yaw0-r003.json").read_text())
        _, sheet = bound_document(config, "workspace_sheets", cell["yaw0_manifest_digest"])
        original = json.loads((config / "test_only_physical/goal2-place1/yaw0_sheet.json").read_text())
        self.assertEqual(sheet, original)
        self.assertEqual(sheet["place_id"], cell["place_id"])

    def test_only_finger_coordinate_expression_changes_and_old_model_is_preserved(self):
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), ORIGINAL_SHA)
        old, new = ET.parse(ORIGINAL).getroot(), ET.parse(CANDIDATE).getroot()
        self.assertIn("UNQUALIFIED MODEL CANDIDATE", CANDIDATE.read_text())
        self.assertEqual(_limits(ORIGINAL.read_text()), _limits(CANDIDATE.read_text()))
        restored = copy.deepcopy(new)
        for side in ("right", "left"):
            for tag in ("origin", "axis"):
                selector = f"./joint[@name='finger_{side}_joint']/{tag}"
                restored.find(selector).attrib = dict(old.find(selector).attrib)
        # Includes arm kinematics, every mesh/link/limit, fixed TCP chain and mimic.
        self.assertEqual(ET.tostring(restored), ET.tostring(old))

    def test_opening_geometry_matches_coordinate_direction_not_a_physical_claim(self):
        old, new = ET.parse(ORIGINAL).getroot(), ET.parse(CANDIDATE).getroot()
        profile = json.loads((ROOT / "config/data_factory/motion_qualifications/fr5-place-a-wood-cube-24mm-r001.json").read_text())
        closed, opened = (profile["gripper_positions_m"][key] for key in ("closed", "open"))
        # Falsifier: the original model reverses the empirically used command.
        self.assertLess(finger_geometry(old, opened)[0], finger_geometry(old, closed)[0])
        self.assertGreater(finger_geometry(new, opened)[0], finger_geometry(new, closed)[0])
        for q in np.linspace(0., opened, 21):
            self.assertAlmostEqual(finger_geometry(new, q)[0], finger_geometry(old, opened - q)[0])
            self.assertAlmostEqual(finger_geometry(new, q)[1], finger_geometry(old, q)[1])
        self.assertAlmostEqual(finger_geometry(new, opened)[0], .0421)
        self.assertAlmostEqual(finger_geometry(new, closed)[0], .02362)
        # These are model numbers, NOT measured custom-tip aperture or attachment.

    def test_native_recording_and_proposals_keep_values_but_old_approval_cannot_rebind(self):
        recorder = FR5LeRobotRecorder.__new__(FR5LeRobotRecorder)
        recorder.lock = threading.Lock()
        recorder.arm_actions, recorder.gripper_actions, recorder.joint_states = deque(), deque(), deque()
        stamp = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)))
        arm = [0.] * 6
        recorder._on_arm_state(SimpleNamespace(header=stamp.header, joint_names=JOINTS[:6], reference=SimpleNamespace(positions=arm)))
        recorder._on_gripper_state(SimpleNamespace(header=stamp.header, joint_names=JOINTS[-1:], reference=SimpleNamespace(positions=[.01176])))
        recorder._on_joint_state(SimpleNamespace(header=stamp.header, name=JOINTS, position=arm + [.01218]))
        recorded = np.array([*recorder.arm_actions[-1][1], recorder.gripper_actions[-1][1]], dtype=np.float32)
        feedback = recorder.joint_states[-1][1]
        np.testing.assert_array_equal(recorded, np.array(arm + [.01176], dtype=np.float32))
        np.testing.assert_array_equal(feedback, np.array(arm + [.01218], dtype=np.float32))
        before = (recorded.tobytes(), feedback.tobytes())
        observation = {"source_clock": "SYSTEM_TIME", "source_timestamps_s": {k: 10. for k in ("state", "camera1", "camera2")},
                       "observation.state": feedback.tolist(), "observation.images.camera1": fake_rgb(), "observation.images.camera2": fake_rgb()}
        checkpoint = {"tree_digest": canonical_digest("synthetic-weights"), "training_receipt_digest": canonical_digest("synthetic-receipt"), "runtime": "SYNTHETIC_TEST_ONLY"}
        proposals = [FinitePolicyInference(lambda _: [recorded.tolist()], checkpoint, source_clock=lambda: 10.).propose(
            observation, instruction="synthetic coordinate compatibility", robot_description=p.read_text(), period_s=1., velocity_scaling=.03)
            for p in (ORIGINAL, CANDIDATE)]
        for proposal in proposals:
            self.assertEqual(proposal["actions"], [recorded.tolist()])
            self.assertEqual(proposal["initial_state"], feedback.tolist())
            self.assertEqual(proposal["checkpoint"], checkpoint)
        self.assertEqual(before, (recorded.tobytes(), feedback.tobytes()))
        source = motion()  # Existing declared synthetic fixture; no live approval.
        source["binding_digests"]["robot_description_digest"] = "sha256:" + ORIGINAL_SHA
        compile_program(source, proposals[0])
        with self.assertRaisesRegex(ContractError, "LEARNED_ROBOT_BINDING"):
            compile_program(source, proposals[1])

        trial = build_robot_model_trial(
            source,
            ORIGINAL.read_text(),
            CANDIDATE.read_text(),
        )
        candidate_program = compile_program(
            source,
            proposals[1],
            robot_model_trial=trial,
        )
        self.assertFalse(
            candidate_program["robot_model_trial"][
                "execution_authorized"
            ]
        )
        self.assertEqual(
            candidate_program["robot_model_trial"][
                "source_robot_description_digest"
            ],
            "sha256:" + ORIGINAL_SHA,
        )
        self.assertEqual(
            candidate_program["robot_model_trial"][
                "candidate_robot_description_digest"
            ],
            "sha256:" + hashlib.sha256(
                CANDIDATE.read_bytes()
            ).hexdigest(),
        )

        forged = ET.parse(CANDIDATE).getroot()
        forged_limit = forged.find("./joint[@name='j1']/limit")
        self.assertIsNotNone(forged_limit)
        forged_limit.set("upper", "1.23456789")
        forged_xml = ET.tostring(
            forged,
            encoding="unicode",
        )
        with self.assertRaisesRegex(
            ContractError,
            "LEARNED_ROBOT_MODEL_TRIAL_DELTA",
        ):
            build_robot_model_trial(
                source,
                ORIGINAL.read_text(),
                forged_xml,
            )

    def test_old_home_and_default_selection_remain_bound_to_original_model(self):
        home = json.loads((ROOT / "config/data_factory/home_candidates/fr5-lab-a-tcp-r002-home-r001.json").read_text())
        validate_home_candidate(home, urdf=ORIGINAL, expected_robot_system_id=home["robot_system_id"])
        with self.assertRaisesRegex(ContractError, "HOME_ROBOT_BINDING"):
            validate_home_candidate(home, urdf=CANDIDATE, expected_robot_system_id=home["robot_system_id"])
        xacro = (ROOT / "src/fairino5_v6_moveit2_config/config/fairino5_v6_robot.urdf.xacro").read_text()
        self.assertIn('name="robot_model_file" default="$(find fairino_description)/urdf/fairino5_v6.urdf"', xacro)
        self.assertIn('filename="$(arg robot_model_file)"', xacro)
        self.assertNotIn(CANDIDATE.name, xacro)

    def test_native_launch_selection_expands_one_model_without_starting_processes(self):
        # ROS launch uses system Python (lark is deliberately not added to the
        # training venv). Only resolve native launch/xacro parameters; no launch.
        script = r'''
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET
from xacro import XacroException
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from moveit_configs_utils import MoveItConfigsBuilder

root = Path(sys.argv[1])
package = root / "src/fairino5_v6_moveit2_config"
spec = importlib.util.spec_from_file_location("model_launch", package / "launch/real_robot.launch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class SourceBuilder(MoveItConfigsBuilder):
    def robot_description(self, **kwargs):
        return super().robot_description(file_path=package / "config/fairino5_v6_robot.urdf.xacro", **kwargs)

def expand(model=None, fake="false"):
    # ParameterValue caches evaluation within one launch. A new model selection
    # is a new launch, never a hot swap of the running description.
    context = LaunchContext()
    context.launch_configurations["use_fake_hardware"] = fake
    if model is not None:
        context.launch_configurations["robot_model_file"] = str(model)
    with patch.object(module, "MoveItConfigsBuilder", SourceBuilder), patch.object(module, "generate_demo_launch", return_value=LaunchDescription()) as consume:
        description = module.generate_launch_description()
        consume.assert_not_called()
        for item in description.entities:
            assert isinstance(item, (DeclareLaunchArgument, OpaqueFunction)), "unexpected process action"
            item.execute(context)
        consume.assert_called_once()
        config = consume.call_args.args[0]
    value = config.robot_description["robot_description"]
    return ET.fromstring(value.evaluate(context) if hasattr(value, "evaluate") else value)

original = root / "src/fairino_description/urdf/fairino5_v6.urdf"
candidate = original.with_name("fairino5_v6_gripper_opening_candidate.urdf")
for fake in ("false", "true"):
    default, old, new = expand(fake=fake), expand(original, fake), expand(candidate, fake)
    assert ET.tostring(default) == ET.tostring(old), "default model changed"
    assert list(map(float, new.find("joint[@name='finger_right_joint']/axis").get("xyz").split())) == [1., 0., 0.]
    for side in ("right", "left"):
        for tag in ("origin", "axis"):
            selector = f"joint[@name='finger_{side}_joint']/{tag}"
            new.find(selector).attrib = dict(old.find(selector).attrib)
    assert ET.tostring(new) == ET.tostring(old), "non-finger model/control change"
try:
    expand(original.with_name("missing-model.urdf"))
except XacroException:
    pass
else:
    raise AssertionError("missing model silently fell back")
print("native launch/xacro: unchanged default, explicit candidate, both hardware modes, no fallback PASS")
'''
        result = subprocess.run(["/usr/bin/python3", "-c", script, str(ROOT)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


    def test_nested_runtime_launches_propagate_selected_robot_model(self):
        script = r"""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument

root = Path(sys.argv[1])
launch_dir = root / "src/fairino5_v6_moveit2_config/launch"

original = (
    root / "src/fairino_description/urdf/fairino5_v6.urdf"
)
candidate = original.with_name(
    "fairino5_v6_gripper_opening_candidate.urdf"
)

cases = (
    ("rsp.launch.py", "generate_rsp_launch"),
    ("move_group.launch.py", "generate_move_group_launch"),
    ("moveit_rviz.launch.py", "generate_moveit_rviz_launch"),
)

def load_case(filename, generator):
    path = launch_dir / filename
    spec = importlib.util.spec_from_file_location(
        filename.replace(".", "_"),
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch.object(
        module,
        generator,
        return_value=LaunchDescription(),
    ) as consume:
        description = module.generate_launch_description()

    config = consume.call_args.args[0]
    return description, config

def expand(description, config, model=None, fake="true"):
    context = LaunchContext()
    context.launch_configurations["use_fake_hardware"] = fake

    if model is not None:
        context.launch_configurations["robot_model_file"] = str(model)

    for item in description.entities:
        if isinstance(item, DeclareLaunchArgument):
            item.execute(context)

    value = config.robot_description["robot_description"]
    text = value.evaluate(context) if hasattr(value, "evaluate") else value
    return ET.fromstring(text)

def axis(robot, name):
    return robot.find(
        f"./joint[@name='{name}']/axis"
    ).get("xyz")

def expand_case(filename, generator, model=None, fake="true"):
    # ParameterValue/Xacro evaluation is launch-instance scoped.
    # A different robot model selection must use a fresh config object,
    # exactly as a new ros2 launch invocation would.
    description, config = load_case(filename, generator)
    return expand(description, config, model, fake)

for filename, generator in cases:
    default = expand_case(filename, generator)
    assert axis(default, "finger_right_joint") == "-1 0 0"
    assert axis(default, "finger_left_joint") == "1 0 0"

    fake = expand_case(
        filename, generator, candidate, "true"
    )
    assert axis(fake, "finger_right_joint") == "1 0 0"
    assert axis(fake, "finger_left_joint") == "-1 0 0"
    assert (
        fake.findtext(".//ros2_control/hardware/plugin")
        == "mock_components/GenericSystem"
    )

    real = expand_case(
        filename, generator, candidate, "false"
    )
    assert axis(real, "finger_right_joint") == "1 0 0"
    assert axis(real, "finger_left_joint") == "-1 0 0"
    assert (
        real.findtext(".//ros2_control/hardware/plugin")
        == "fairino_hardware/FairinoHardwareInterface"
    )

print(
    "nested launch propagation: "
    "default preserved, candidate propagated, "
    "fake/real hardware selection PASS"
)
"""
        result = subprocess.run(
            ["/usr/bin/python3", "-c", script, str(ROOT)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
