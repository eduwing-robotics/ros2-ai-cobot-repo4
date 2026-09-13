"""CPU native-method extraction and real ROS serializer acceptance; no ROS init."""
import json
import re
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from tools.fr5_data_factory import ContractError
from tools.data_factory.rollout.gripper_evidence import (
    FIELDS, RESOURCE, check_hardware, decode_dynamic_state, validate_clock_binding, native_clock_parameter,
)

ROOT = Path(__file__).resolve().parents[3]


def patched_source(path):
    patch = (ROOT / "patches/frcobot_ros2.patch").read_text()
    section = patch.split(f"diff --git a/{path} b/{path}\n", 1)[1].split("diff --git ", 1)[0]
    # Diff alignment may retain a new method's closing brace as context. Never
    # concatenate across an omitted part of a native method and test altered code.
    lines, next_line = [], None
    for line in section.splitlines(True):
        if line.startswith("@@"):
            start = int(re.search(r"\+(\d+)", line).group(1))
            if next_line is not None and start != next_line:
                lines.append("#error omitted patch context\n")
            next_line = start
        elif line.startswith(("+", " ")) and not line.startswith("+++"):
            lines.append(line[1:])
            next_line += 1
    return "".join(lines)


def method(source, name):
    start = re.search(r"(?:void|bool|hardware_interface::return_type) FairinoHardwareInterface::" + name + r"\(", source).start()
    opening = source.index("{", start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if not depth:
            if "#error omitted patch context" in source[start:end + 1]:
                raise AssertionError("native method requires complete patch context: " + name)
            return source[start:end + 1] + "\n"
    raise AssertionError("native method incomplete")


class NativeGripperEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        path = Path(cls.temp.name)
        header = patched_source("fairino_hardware_v3_9_7/include/fairino_hardware/gripper_execution_evidence.hpp")
        (path / "gripper_execution_evidence.hpp").write_text(header)
        source = patched_source("fairino_hardware_v3_9_7/src/fairino_hardware_interface.cpp")
        native = (method(source, "certified_gripper_observation") + method(source, "refresh_gripper_freshness") + method(source, "gripper_worker") + method(source, "sample_gripper_evidence") + method(source, "sample_coherent_evidence")
                  + method(source, "gripper_release_ready") + method(source, "write") + method(source, "stop_gripper_worker"))
        fixture = Path(__file__).with_name("gripper_native_fixture.cpp").read_text()
        (path / "native.cpp").write_text(fixture.replace("// NATIVE_METHODS", native))
        cls.binary = path / "native"
        result = subprocess.run(["g++", "-std=c++17", "-pthread", str(path / "native.cpp"), "-o", str(cls.binary)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)

    def run_native(self, scenario, parameter=None):
        args = [str(self.binary), scenario]
        if parameter is not None:
            args.append(",".join(map(repr, parameter)))
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_native_release_uses_serialized_command_pinned_clock(self):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from rcl_interfaces.msg import ParameterValue, ParameterType
        from rclpy.serialization import deserialize_message, serialize_message
        scenarios = ("fresh", "delayed_fresh", "max_incarnation", "cached", "wrong_incarnation", "fractional_incarnation",
                     "oversized_incarnation", "reordered_incarnation", "paused_binding", "invalid_calendar",
                     "missing_binding", "invalid_number", "expired_move", "rebind_expired",
                     "stop_before_resume", "stop_during_resume", "error_before_resume", "move_error", "resume_error", "late_resume",
                     "superseded", "stale_read", "wrong_incarnation_read", "limit")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                now, steady = time.time(), time.monotonic()
                binding = {"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                           "calendar_to_system_offset_s": 0., "uncertainty_s": .002,
                           "system_anchor_s": now, "steady_anchor_s": steady,
                           "valid_until_system_s": now + 10.}
                if scenario == "max_incarnation":
                    binding["incarnation"][0] = 2**32 - 1
                message = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                         double_array_value=native_clock_parameter(binding, .3))
                parameter = list(deserialize_message(serialize_message(message), ParameterValue).double_array_value)
                packet = self.run_native("clock_" + scenario, parameter)
                self.assertEqual(packet["released"], scenario in ("fresh", "delayed_fresh", "max_incarnation", "superseded"))
                if scenario not in ("fresh", "delayed_fresh", "max_incarnation"):
                    continue
                # Actual sampler -> actual ROS wire -> existing held clock validator.
                state = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
                    interface_names=packet["names"], values=packet["wire"])])
                received = time.monotonic()
                decoded = decode_dynamic_state(deserialize_message(serialize_message(state), DynamicJointState),
                                               binding, received)
                captured, captured_steady = time.time(), time.monotonic()
                evidence = {"captured_at_s": captured, "captured_monotonic_s": captured_steady,
                            "snapshot": {"gripper_controller": {"hardware_execution": decoded}}}
                self.assertEqual(check_hardware(evidence, captured, captured_steady, .3,
                                                completion=True)["completed_generation"], 1)

    def test_native_clock_encoding_rejects_unbounded_age_without_effects(self):
        binding = {"schema_version": "fr5.gripper_source_clock.v1", "incarnation": [1, 2, 3, 4],
                   "calendar_to_system_offset_s": 0., "uncertainty_s": .001,
                   "system_anchor_s": 10., "steady_anchor_s": 20., "valid_until_system_s": 11.}
        for age in (0., -1., float("nan"), float("inf"), True):
            with self.subTest(age=age), self.assertRaises(ContractError):
                native_clock_parameter(binding, age)
        for value in (.5, 2**32, 2**64 - 1):
            with self.subTest(incarnation=value), self.assertRaises(ContractError):
                native_clock_parameter({**binding, "incarnation": [value, 2, 3, 4]}, .3)

    def test_actual_worker_stop_error_supersession_and_completion(self):
        for scenario in ("completed", "settled", "cached", "superseded", "stop_before_move", "error_before_move",
                         "stop_during_move", "stop_before_resume", "stop_during_resume", "move_error", "resume_error"):
            with self.subTest(scenario=scenario):
                self.run_native(scenario)

    def test_distinct_small_reference_is_not_discarded_or_allowed_to_supersede_pending(self):
        for scenario in ("distinct_reference", "legacy_deadband"):
            with self.subTest(scenario=scenario):
                self.assertEqual(self.run_native(scenario), {"native_write_only": True})

    def test_current_driver_pauses_arm_during_gripper_despite_fresh_state(self):
        self.assertEqual(self.run_native("overlap_write"), {
            "initial_arm_sends": 0, "active_arm_sends": 0,
            "fault_stops_sends": True, "physical_overlap": "UNKNOWN"})

    def test_off_target_old_done_needs_command_activity_or_dwell(self):
        # Reuse the actual worker with only device telemetry varied. No ROS/network.
        fixture = Path(__file__).with_name("gripper_native_fixture.cpp").read_text().split("int main(")[0]
        source = patched_source("fairino_hardware_v3_9_7/src/fairino_hardware_interface.cpp")
        native = "\n".join(method(source, name) for name in (
            "certified_gripper_observation", "refresh_gripper_freshness", "gripper_worker", "sample_gripper_evidence", "sample_coherent_evidence",
            "gripper_release_ready", "write", "stop_gripper_worker"))
        main = r'''int main(int argc,char **argv) {
          using namespace fairino_hardware;
          FairinoHardwareInterface h; auto &r=*h._ptr_robot;
          r.clock_mode=true; r.terminal_scenario=std::stoi(argv[1]);
          if(r.terminal_scenario==4) h._pending_gripper_position=100;
          h._gripper_evidence.activate(); h._gripper_max_time=160; h._gripper_settle_time_ms=500;
          h._gripper_thread=std::thread([&]{h.gripper_worker();});
          auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(300);
          while(h._arm_stream_paused && !h._gripper_error && std::chrono::steady_clock::now()<until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          const bool completed=h._gripper_evidence.completed==1;
          h.stop_gripper_worker();
          assert(completed==(r.terminal_scenario==2 || r.terminal_scenario==4));
          assert(r.resumes==int(completed));
        }'''
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/"gripper_execution_evidence.hpp").write_text(patched_source(
                "fairino_hardware_v3_9_7/include/fairino_hardware/gripper_execution_evidence.hpp"))
            (path/"worker.cpp").write_text(fixture.replace("// NATIVE_METHODS",native)+main)
            subprocess.run(["g++","-std=c++17","-pthread",str(path/"worker.cpp"),"-o",str(path/"worker")],check=True,capture_output=True)
            for scenario in (1,2,3,4):
                with self.subTest(scenario=scenario):
                    subprocess.run([str(path/"worker"),str(scenario)],check=True,timeout=2,capture_output=True)

    def test_invalid_clock_file_rejects_before_ros_initialization(self):
        from tools.data_factory.motion.pickup_executor import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clock.json"
            path.write_text('{"schema_version":"unknown"}')
            with mock.patch("rclpy.init") as init, mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit) as caught:
                    main(["--factory-jsonl", "--ros-plan-only", "--gripper-source-clock", str(path)])
            self.assertEqual(caught.exception.code, 2)
            init.assert_not_called()

    def test_native_export_round_trip_and_cached_completion_rejection(self):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from rclpy.serialization import deserialize_message, serialize_message
        for scenario in ("completed", "cached"):
            packet = self.run_native("completed")
            if scenario == "cached":
                packet["wire"][28:35] = [1970.,1.,1.,0.,0.,9.,0.]
            message = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
                interface_names=packet["names"], values=packet["wire"])])
            self.assertEqual(tuple(packet["names"]), FIELDS)
            wire = dict(zip(FIELDS, packet["wire"]))
            mapping = {"schema_version": "fr5.gripper_source_clock.v1",
                       "incarnation": [wire[f"incarnation_{i}"] for i in range(4)],
                       "calendar_to_system_offset_s": 0., "uncertainty_s": .001,
                       "system_anchor_s": 10., "steady_anchor_s": 20., "valid_until_system_s": 11.}
            decoded = decode_dynamic_state(deserialize_message(serialize_message(message), DynamicJointState), mapping, 20.2)
            self.assertEqual(list(decoded["wire"].values()), packet["wire"])
            # Worker uses actual host time for command start; this synthetic replay
            # maps its command instant to 10.0 while retaining exact native source calendar.
            decoded["wire"]["command_started_system_s"] = 10.
            evidence = {"captured_at_s": 10.2, "captured_monotonic_s": 20.2,
                        "snapshot": {"gripper_controller": {"hardware_execution": decoded}}}
            if scenario == "completed":
                self.assertEqual(check_hardware(evidence, 10.2, 20.2, .3, completion=True)["completed_generation"], 1)
            else:
                # Fresh local sample/receipt never repairs the pre-command completion frame.
                with self.assertRaisesRegex(ContractError, "LEARNED_HARDWARE_COMPLETION"):
                    check_hardware(evidence, 10.2, 20.2, .3, completion=True)
            for changes, expected in (({"incarnation": [0, 0, 0, 0]}, "LEARNED_HARDWARE_CLOCK_BINDING"),
                                      ({"uncertainty_s": float("nan")}, "LEARNED_HARDWARE_CLOCK_BINDING")):
                with self.assertRaisesRegex(ContractError, expected):
                    validate_clock_binding({**mapping, **changes})
            message.interface_values[0].interface_names[0] = FIELDS[1]
            with self.assertRaisesRegex(ContractError, "LEARNED_HARDWARE_SCHEMA"):
                decode_dynamic_state(message, mapping, 20.2)


if __name__ == "__main__":
    unittest.main()


class CoherentSnapshotTest(unittest.TestCase):
    """Actual selected SDK publication -> native methods -> ROS/Python consumer.

    The SDK transport constructors and robot command calls are injected. This is
    software integration evidence; it cannot qualify a controller or a Pick.
    """
    @classmethod
    def setUpClass(cls):
        import os
        cls.sdk = Path(os.environ.get("FAIRINO_SNAPSHOT_SDK_ROOT", ""))
        if not os.environ.get("FAIRINO_SNAPSHOT_SDK_ROOT"):
            raise unittest.SkipTest("select immutable FAIRINO_SNAPSHOT_SDK_ROOT for candidate integration")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        path = Path(cls.temp.name)
        (path / "gripper_execution_evidence.hpp").write_text(patched_source(
            "fairino_hardware_v3_9_7/include/fairino_hardware/gripper_execution_evidence.hpp"))
        source = patched_source("fairino_hardware_v3_9_7/src/fairino_hardware_interface.cpp")
        native = "\n".join(method(source, name) for name in (
            "certified_gripper_observation", "refresh_gripper_freshness", "gripper_worker", "sample_gripper_evidence",
            "sample_coherent_evidence", "gripper_release_ready", "write", "stop_gripper_worker"))
        fixture = Path(__file__).with_name("gripper_native_fixture.cpp").read_text().split("int main(")[0]
        code = Path(__file__).with_name("snapshot_sdk_fixture.cpp").read_text().replace(
            "// NATIVE_FIXTURE", fixture.replace("// NATIVE_METHODS", native))
        (path / "snapshot.cpp").write_text(code)
        includes = cls.sdk / "libfairino/src/include"
        library = cls.sdk / "libfairino/LinuxBuild/bin"
        cls.binary = path / "snapshot"
        result = subprocess.run(["g++", "-std=c++17", "-pthread", *[f"-I{includes / d}" for d in
            ("Robot-EN", "CNDE", "ComClient", "Base", "Log", "XmlRpc")], str(path / "snapshot.cpp"),
            f"-L{library}", "-lfairino", "-o", str(cls.binary)], capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.env = {**os.environ, "LD_LIBRARY_PATH": str(library) + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        binding = subprocess.check_output(["ldd", str(cls.binary)], env=cls.env, text=True)
        if str(library / "libfairino.so.2") not in binding:
            raise AssertionError("wrong SDK binding: " + binding)
        print("candidate SDK binding:", next(line.strip() for line in binding.splitlines() if "libfairino.so" in line), flush=True)

    def packet(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True, timeout=3, env=self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout[result.stdout.rfind('{'):])

    def evidence(self, packet):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from rclpy.serialization import serialize_message, deserialize_message
        from tools.data_factory.rollout.gripper_evidence import SNAPSHOT_FIELDS
        self.assertEqual(tuple(packet["names"]), SNAPSHOT_FIELDS)
        wire = dict(zip(packet["names"], packet["wire"]))
        policy = {"schema_version": "fr5.gripper_temporal_policy.v2", "incarnation": [1, 2, 3, 4],
                  "connection_epoch": 1, "configuration_epoch": 0, "max_age_s": .08, "host_clock_tolerance_s": .002}
        msg = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
            interface_names=packet["names"], values=packet["wire"])])
        msg = deserialize_message(serialize_message(msg), DynamicJointState)
        # Deterministic consumer clock just after the original SDK receipt.
        steady = wire["sample_steady_s"] + .0001
        now = wire["sample_system_s"] + .0001
        hw = decode_dynamic_state(msg, policy, steady)
        evidence = {"captured_at_s": now, "captured_monotonic_s": steady, "snapshot": {
            "joint_positions": [wire[f"arm_j{i}_rad"] for i in range(1, 7)],
            "gripper_controller": {"feedback_position_m": wire["feedback_m"], "hardware_execution": hw}}}
        return evidence, now, steady

    def test_actual_sdk_native_normal_and_historical_completion_reach_python(self):
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        for mode in ("normal", "complete"):
            with self.subTest(mode=mode):
                packet = self.packet(mode)
                self.assertEqual(packet["arm_sends"], 1)
                if mode == "normal":
                    self.assertEqual(packet["queries"], 0)
                evidence, now, steady = self.evidence(packet)
                hw = evidence["snapshot"]["gripper_controller"]["hardware_execution"]
                wire = check_hardware(evidence, now, steady, .08, completion=mode == "complete")
                self.assertEqual(wire["current_valid"], 0)
                transport = object.__new__(RosMoveItTransport)
                transport._gripper_source_clock = hw["clock_binding"]
                transport._clock = lambda: steady
                transport._gripper_hardware_evidence = lambda: hw
                self.assertTrue(transport._native_current_ready(.08))
                with mock.patch.dict(wire, {"sample_steady_s": steady-.2, "host_receive_steady_s": steady-.2}):
                    self.assertFalse(transport._native_current_ready(.08))
                    with self.assertRaises(ContractError):
                        check_hardware(evidence, now, steady, .08)
                if mode == "complete":
                    with mock.patch.dict(wire, {"terminal_query_before_steady_s": wire["ack_steady_s"]-.01}):
                        with self.assertRaises(ContractError):
                            check_hardware(evidence, now, steady, .08)

    def test_native_delivery_and_command_falsifiers_have_zero_arm_sends(self):
        for mode in ("busy", "stale", "repeat_source", "reconnect", "configuration", "regression", "device_error", "cancel", "old_completion", "deadline", "cancel_query"):
            with self.subTest(mode=mode):
                self.assertEqual(self.packet(mode)["arm_sends"], 0)

    def test_coherent_pending_observation_is_not_completion_or_readiness(self):
        evidence, now, steady = self.evidence(self.packet("normal"))
        wire = evidence["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        # Replay the overlap candidate's between-RPC progress state over the
        # actual SDK/native/ROS v5 sample. No physical overlap is claimed.
        wire.update(generation=1, active_generation=1, completed_generation=0,
                    completion_reason=0, pending=0, rpc_active=0, arm_resumed=1)
        self.assertEqual(check_hardware(evidence, now, steady, .08, allow_pending=True), wire)
        for options in ({}, {"completion": True, "allow_pending": True}):
            with self.subTest(options=options), self.assertRaises(ContractError):
                check_hardware(evidence, now, steady, .08, **options)

    def test_python_rejects_mixed_frame_epoch_and_fake_source_progress(self):
        import copy
        from tools.data_factory.rollout.gripper_evidence import check_transition
        evidence, now, steady = self.evidence(self.packet("normal"))
        for key, value in (("connection_epoch", 2), ("configuration_epoch", 1), ("device_main_error", 1), ("arm_j1_rad", 99)):
            changed = copy.deepcopy(evidence)
            changed["snapshot"]["gripper_controller"]["hardware_execution"]["wire"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                check_hardware(changed, now, steady, .08)
        changed = copy.deepcopy(evidence)
        w = changed["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        w["producer_sequence"] += 1
        w["source_progress_steady_s"] += .00001
        with self.assertRaises(ContractError):
            check_transition(evidence, changed, command=False)
