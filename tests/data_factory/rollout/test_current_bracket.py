"""Exact native producer/sampler/release/write, then the real Python wire consumer."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from tools.fr5_data_factory import ContractError
from tools.data_factory.rollout.gripper_evidence import (
    LIVE_FIELDS, CAUSAL_FIELDS, SELECTED_FIELDS, RESOURCE, check_hardware, decode_dynamic_state, native_clock_parameter, native_temporal_parameter,
)
from tests.data_factory.rollout.test_gripper_evidence import patched_source, method


def temporal_policy(*, incarnation=(1, 2, 3, 4), max_age_s=.08, host_clock_tolerance_s=.002):
    """CPU fixture; deployed values must come from the existing approved limits."""
    return {"schema_version": "fr5.gripper_temporal_policy.v1", "incarnation": list(incarnation),
            "max_age_s": max_age_s, "host_clock_tolerance_s": host_clock_tolerance_s}


class CurrentBracketTest(unittest.TestCase):
    def test_mapping_expiry_falsifiers(self):
        for mode, moves, completed in (("expiry_first", 1, 1), ("expiry_second", 2, 2), ("expiry_during", 1, 1)):
            packet, _ = self.packet(mode)
            self.assertEqual(packet, {"moves": moves, "completed": completed, "error": 0})
            print(mode, packet, flush=True)

    @classmethod
    def setUpClass(cls):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from rclpy.serialization import serialize_message, deserialize_message
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        (root / "gripper_execution_evidence.hpp").write_text(patched_source(
            "fairino_hardware_v3_9_7/include/fairino_hardware/gripper_execution_evidence.hpp"))
        source = patched_source("fairino_hardware_v3_9_7/src/fairino_hardware_interface.cpp")
        native = "\n".join(method(source, name) for name in (
            "certified_gripper_observation", "refresh_gripper_freshness", "gripper_worker", "sample_gripper_evidence", "sample_coherent_evidence",
            "gripper_release_ready", "write", "stop_gripper_worker"))
        fixture = Path(__file__).with_name("gripper_native_fixture.cpp").read_text().split("int main(")[0]
        code = fixture.replace("// NATIVE_METHODS", native) + Path(__file__).with_name("bracket_native_main.cpp").read_text()
        (root / "native.cpp").write_text(code)
        cls.binary = root / "native"
        subprocess.run(["g++", "-std=c++17", "-pthread", str(root / "native.cpp"), "-o", str(cls.binary)], check=True, capture_output=True)

    def packet(self, mode="fresh", *, command_budget_ms=None, query_delay_ms=0, max_age_s=.08):
        policy = temporal_policy(max_age_s=max_age_s)
        argv = [str(self.binary), mode, ",".join(map(repr, native_temporal_parameter(policy)))]
        if command_budget_ms is not None or query_delay_ms:
            argv.extend((str(command_budget_ms or 0), str(query_delay_ms)))
        result = subprocess.run(argv,
                                capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout), policy

    def test_native_acquisition_latency_characterization_without_commands(self):
        # Spend one original age budget, not independent per-query fractions.
        # 37 ms approximates an observed RPC tail, not a qualified latency bound.
        for delay, accepted in ((2, True), (37, True), (60, False), (120, False)):
            with self.subTest(query_delay_ms=delay):
                packet, policy = self.packet("query_budget", query_delay_ms=delay, max_age_s=.1)
                self.assertEqual(packet, {"accepted": accepted, "error": 0 if accepted else -5,
                                          "moves": 0, "arm_sends": 0})
                self.assertEqual(policy["max_age_s"], .1)

    def test_single_acquisition_pass_does_not_prove_continuous_renewal(self):
        packet, _ = self.packet("renewal_schedule")
        self.assertEqual(packet, {"first_ready": True, "old_valid_during_renewal": False,
                                  "old_valid_at_next_ready": False})

    def test_native_continuous_renewal_accepts_interspersed_rpc_tail(self):
        packet, policy = self.packet("renewal_tail", max_age_s=.1)
        self.assertEqual(packet["error"], 0, packet)
        self.assertEqual(packet["moves"], 0, packet)
        self.assertGreater(packet["queries"], 10, packet)
        self.assertGreater(packet["arm_sends"], 10, packet)
        self.assertEqual(policy["max_age_s"], .1)

    def test_native_continuous_renewal_stops_sends_on_actual_expiry(self):
        for mode in ("renewal_expired", "renewal_sustained"):
            with self.subTest(mode=mode):
                packet, _ = self.packet(mode, max_age_s=.1)
                self.assertNotEqual(packet["error"], 0, packet)
                self.assertGreater(packet["arm_sends"], 0, packet)
                self.assertEqual(packet["moves"], 0, packet)
                self.assertEqual(packet["sends_after_fault"], 0, packet)

    def evidence(self, packet, binding):
        from control_msgs.msg import DynamicJointState, InterfaceValue
        from rclpy.serialization import serialize_message, deserialize_message
        message = DynamicJointState(joint_names=[RESOURCE], interface_values=[InterfaceValue(
            interface_names=packet["names"], values=packet["wire"])])
        # Replay at the actual native certificate's observation time. The source
        # has stopped: Python import/serialization delay is not a live sample age.
        at, mono = packet["wire"][40:42] if len(packet["wire"]) >= 42 else packet["wire"][9:11]
        decoded = decode_dynamic_state(deserialize_message(serialize_message(message), DynamicJointState), binding, mono)
        return {"captured_at_s": at, "captured_monotonic_s": mono,
                "snapshot": {"gripper_controller": {"hardware_execution": decoded}}}

    def test_delayed_wire_replay_does_not_relabel_stale_data_as_live(self):
        packet, policy = self.packet()
        time.sleep(.1)  # Reproduce Python being scheduled after the 80 ms live bound.
        evidence = self.evidence(packet, policy)
        at, mono = evidence["captured_at_s"], evidence["captured_monotonic_s"]
        self.assertEqual((at, mono), tuple(packet["wire"][40:42]))
        check_hardware(evidence, at, mono, .08)
        with self.assertRaisesRegex(ContractError, "LEARNED_HARDWARE_STALE"):
            check_hardware(evidence, at+.1, mono+.1, .08)

    def test_completed_proof_survives_new_current_brackets_and_old_expiry(self):
        packet, binding = self.packet()
        self.assertEqual(tuple(packet["names"]), SELECTED_FIELDS)
        self.assertTrue(packet["proof_unchanged"])
        self.assertGreater(packet["renewed_arm_sends"], 0)
        evidence = self.evidence(packet, binding)
        self.assertGreater(evidence["captured_at_s"] - packet["wire"][44], .15)
        wire = check_hardware(evidence, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)
        self.assertEqual(wire["completed_generation"], 1)
        self.assertEqual(wire["version"], 4)
        self.assertEqual(wire["selected_generation"], wire["completed_generation"])
        self.assertEqual(wire["selected_initial_percent"], 100)
        self.assertEqual(wire["selected_upper_m"], .021)
        self.assertEqual([wire[k] for k in SELECTED_FIELDS[-10:]], packet["sdk_tuple"])
        archived = copy.deepcopy(evidence)
        old = archived["snapshot"]["gripper_controller"]["hardware_execution"]
        old["wire"] = {k: wire[k] for k in CAUSAL_FIELDS}
        old["wire"]["version"] = 3
        check_hardware(archived, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)

    def test_pending_gripper_observation_does_not_require_arm_suppression(self):
        packet, policy = self.packet()
        evidence = self.evidence(packet, policy)
        wire = evidence["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        wire.update(pending=0, rpc_active=0, active_generation=1,
                    completed_generation=0, completion_reason=0, arm_resumed=1)
        at, mono = evidence["captured_at_s"], evidence["captured_monotonic_s"]
        original = copy.deepcopy(evidence)
        self.assertEqual(check_hardware(evidence, at, mono, .08, allow_pending=True), wire)
        self.assertEqual(evidence, original)
        # Progress observation is neither admission readiness nor completion.
        for options in ({}, {"allow_pending": True, "completion": True}):
            with self.subTest(options=options), self.assertRaises(ContractError):
                check_hardware(evidence, at, mono, .08, **options)
        for changed in ({"active_generation": 2}, {"completed_generation": 1},
                        {"completion_reason": 1}, {"error": -5}, {"stopped": 1},
                        {"pending": 1}, {"rpc_active": 1}):
            candidate = copy.deepcopy(evidence)
            candidate["snapshot"]["gripper_controller"]["hardware_execution"]["wire"].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                check_hardware(candidate, at, mono, .08, allow_pending=True)
        with self.assertRaisesRegex(ContractError, "LEARNED_HARDWARE_STALE"):
            check_hardware(evidence, at+.1, mono+.1, .08, allow_pending=True)

    def test_actual_worker_same_integer_endpoint_keeps_distinct_direction_tuple(self):
        from tools.data_factory.rollout.gripper_evidence import native_close_equivalence
        required = {"command_position_m":.01176,"velocity_percent":20,"force_percent":20}
        for mode, initial, selected in (("tuple_open",55,[10,50]),("tuple_close",56,[20,20])):
            with self.subTest(mode=mode):
                packet, binding = self.packet(mode)
                evidence = self.evidence(packet,binding)
                wire = check_hardware(evidence,evidence["captured_at_s"],evidence["captured_monotonic_s"],.08)
                self.assertEqual(wire["raw_reference_m"],.01177)
                self.assertEqual(wire["selected_position"],56)
                self.assertEqual(wire["selected_initial_percent"],initial)
                self.assertEqual(packet["sdk_tuple"][2:4],selected)
                self.assertEqual([wire[k] for k in SELECTED_FIELDS[-10:]],packet["sdk_tuple"])
                if mode=="tuple_open":
                    with self.assertRaisesRegex(ContractError,"CONTACT_NATIVE_COMMAND_NOT_EQUIVALENT"):
                        native_close_equivalence(wire,required,.021)
                else:
                    self.assertEqual(native_close_equivalence(wire,required,.021)["generation"],1)

    def test_native_missing_expired_wrong_sample_reset_and_stop_block(self):
        for mode in ("expired", "wrong_sample", "absent", "reset", "old_generation", "incomplete", "regressed", "stopped", "pre_ack", "terminal_wrong_generation", "terminal_wrong_reference",
                     "terminal_wrong_incarnation", "terminal_stale", "terminal_tie", "terminal_fault", "terminal_wrong_done"):
            with self.subTest(mode=mode):
                self.packet(mode)

    def test_actual_causal_command_activity_and_faults(self):
        for mode in ("command_old_done", "command_changed_old_done", "command_stale", "command_stop", "command_error",
                     "command_policy_change", "command_late_resume", "command_cancel_query", "command_99", "command_busy_done", "command_settled"):
            with self.subTest(mode=mode):
                packet, _ = self.packet(mode)
                success = mode in ("command_99", "command_busy_done", "command_settled")
                self.assertEqual(packet["completed"], int(success), packet)
                self.assertEqual(packet["moves"], 0 if mode=="command_policy_change" else 1)
                if success:
                    self.assertEqual(packet["error"], 0)

    def test_settled_completion_budget_is_separate_from_sample_freshness(self):
        # Three causal observations plus two 50 ms polls need more than the old
        # 120 ms fixture budget when each clock query takes only 5 ms.
        for budget, completed, error in ((120, 0, -1), (None, 1, 0)):
            with self.subTest(command_budget_ms=budget):
                packet, policy = self.packet("command_settled", command_budget_ms=budget, query_delay_ms=5)
                self.assertEqual(packet, {"moves": 1, "completed": completed, "error": error})
                self.assertEqual(policy["max_age_s"], .08)

    def test_python_causal_terminal_proof_and_archive_separation(self):
        from tools.data_factory.rollout.gripper_evidence import calendar_s
        packet, policy = self.packet()
        evidence = self.evidence(packet, policy)
        w = evidence["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        changes = ({"ack_steady_s": w["terminal_query_before_steady_s"]+.001},
                   {"terminal_generation": 2}, {"terminal_reference_m": .1}, {"terminal_incarnation_0": 9},
                   {"terminal_query_before_steady_s": w["terminal_query_before_steady_s"]-1},
                   {"terminal_query_before_controller_s": calendar_s(w,"completion_")-.001},
                   {"terminal_motion_done": 0}, {"terminal_fault": 1}, {"proof_valid":0},
                   {"proof_clock_version": 1}, {"terminal_max_age_s": 99})
        for change in changes:
            candidate = copy.deepcopy(evidence)
            candidate["snapshot"]["gripper_controller"]["hardware_execution"]["wire"].update(change)
            with self.subTest(change=change), self.assertRaises(ContractError):
                check_hardware(candidate, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)
        # Historical terminal remains readable; it cannot be handed off as newly complete.
        with self.assertRaises(ContractError):
            check_hardware(evidence, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08, completion=True)
        # A v3 policy cannot promote old v2 evidence to the new live contract.
        old = copy.deepcopy(evidence)
        old["snapshot"]["gripper_controller"]["hardware_execution"]["wire"] = {k:w[k] for k in LIVE_FIELDS}
        old["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]["version"] = 2
        with self.assertRaises(ContractError):
            check_hardware(old, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)

    def test_archived_v2_mapping_proof_still_reads_after_expiry(self):
        packet, policy = self.packet()
        evidence = self.evidence(packet, policy)
        hw = evidence["snapshot"]["gripper_controller"]["hardware_execution"]
        w = {k: hw["wire"][k] for k in LIVE_FIELDS}
        w["version"] = 2
        at, mono = w["proof_system_s"], w["proof_steady_s"]
        binding = {"schema_version":"fr5.gripper_source_clock.v1", "incarnation":[1,2,3,4],
                   "calendar_to_system_offset_s":0., "uncertainty_s":.002,
                   "system_anchor_s":at-.1, "steady_anchor_s":mono-.1, "valid_until_system_s":at+.05}
        values = native_clock_parameter(binding, .08)
        for key, value in zip(LIVE_FIELDS[46:57], values):
            w[key] = value
        hw.update(wire=w, clock_binding=binding)
        self.assertGreater(evidence["captured_at_s"], binding["valid_until_system_s"])
        self.assertEqual(check_hardware(evidence, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)["version"], 2)

    def test_bootstrap_identity_decode_does_not_grant_readiness(self):
        from tools.data_factory.rollout.gripper_evidence import FIELDS, identity
        packet, policy = self.packet()
        for names, version in ((FIELDS, 1), (CAUSAL_FIELDS, 3)):
            startup = {"names": list(names), "wire": packet["wire"][:len(names)]}
            startup["wire"][0] = version
            startup["wire"][27] = 0
            if version == 3:
                startup["wire"][35] = 0
            evidence = self.evidence(startup, policy)
            self.assertEqual(identity(evidence["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]), [1,2,3,4])
            with self.assertRaises(ContractError):
                check_hardware(evidence, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)

    def test_python_rejects_relabel_and_incomplete_proof(self):
        packet, binding = self.packet()
        evidence = self.evidence(packet, binding)
        wire = evidence["snapshot"]["gripper_controller"]["hardware_execution"]["wire"]
        for changes in ({"current_valid":0}, {"query_before_steady_s":wire["query_before_steady_s"]-1},
                        {"frame":99}, {"millisecond":999}, {"proof_valid":0}, {"incarnation_0":99},
                        {"query_after_controller_s":wire["query_before_controller_s"]-1}):
            candidate = copy.deepcopy(evidence)
            candidate["snapshot"]["gripper_controller"]["hardware_execution"]["wire"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                check_hardware(candidate, evidence["captured_at_s"], evidence["captured_monotonic_s"], .08)

class PreciseReaderTest(unittest.TestCase):
    def test_query_failure_retains_transport_cause_without_network_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "precise_controller_clock.hpp").write_text(patched_source(
                "fairino_hardware_v3_9_7/include/fairino_hardware/precise_controller_clock.hpp"))
            (root / "failure.cpp").write_text(r'''
#include "precise_controller_clock.hpp"
#include <cassert>
#include <cstdarg>
int calls=0;
extern "C" CURLcode __wrap_curl_easy_perform(CURL*) {
  ++calls; return CURLE_OPERATION_TIMEDOUT;
}
extern "C" CURLcode __wrap_curl_easy_getinfo(CURL*, CURLINFO info, ...) {
  va_list args; va_start(args, info);
  if(info==CURLINFO_RESPONSE_CODE) *va_arg(args,long*)=0;
  else if(info==CURLINFO_TOTAL_TIME) *va_arg(args,double*)=.025175;
  else {va_end(args);return CURLE_UNKNOWN_OPTION;}
  va_end(args); return CURLE_OK;
}
int main() {
  fairino_hardware::PreciseControllerClock client("127.0.0.1");
  try {client.read(25,[]{return false;});assert(false);}
  catch(const std::runtime_error &e) {
    assert(std::string(e.what())=="clock query: curl=28 http=0 cancelled=0 elapsed_s=0.025175 timeout_ms=25");
  }
  assert(calls==1);
  try {client.read(25,[]{return true;});assert(false);}
  catch(const std::runtime_error &e) {assert(std::string(e.what())=="clock cancelled");}
  assert(calls==1);
}
''')
            binary = root / "failure"
            subprocess.run(["g++", "-std=c++17", str(root / "failure.cpp"),
                            "-Wl,--wrap=curl_easy_perform", "-Wl,--wrap=curl_easy_getinfo",
                            "-lcurl", "-ltinyxml2", "-o", str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True, timeout=2)

    def test_actual_parser_keeps_epoch_precision_and_cancellation_has_no_query(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "precise_controller_clock.hpp").write_text(patched_source(
                "fairino_hardware_v3_9_7/include/fairino_hardware/precise_controller_clock.hpp"))
            (root / "reader.cpp").write_text(r'''
#include "precise_controller_clock.hpp"
#include <cassert>
int main() {
  using fairino_hardware::PreciseControllerClock;
  const std::string prefix="<methodResponse><params><param><value><array><data><value><int>0</int></value><value><double>";
  const std::string suffix="</double></value></data></array></value></param></params></methodResponse>";
  auto a=PreciseControllerClock::parse(prefix+"1788875552807.385"+suffix);
  auto b=PreciseControllerClock::parse(prefix+"1788875552808.385"+suffix);
  assert(b>a && b-a<.002);
  for(const char *bad:{"nan","inf","-1","0"}) {
    bool rejected=false;try{PreciseControllerClock::parse(prefix+bad+suffix);}catch(...){rejected=true;}assert(rejected);
  }
  PreciseControllerClock client("127.0.0.1"); // cancelled before any network call
  bool cancelled=false;try{client.read(10,[]{return true;});}catch(...){cancelled=true;}assert(cancelled);
}
''')
            binary = root / "reader"
            subprocess.run(["g++", "-std=c++17", str(root / "reader.cpp"), "-lcurl", "-ltinyxml2", "-o", str(binary)],
                           check=True, capture_output=True)
            subprocess.run([str(binary)], check=True, timeout=2)

class VendorPatchBuildTest(unittest.TestCase):
    def test_patch_applies_to_pinned_vendor_and_real_translation_unit_compiles(self):
        import os
        selected_sdk = os.environ.get("FAIRINO_SNAPSHOT_SDK_INCLUDE")
        if not selected_sdk:
            self.skipTest("opt-in: set FAIRINO_SNAPSHOT_SDK_INCLUDE to candidate snapshot SDK headers")
        sdk_include = Path(selected_sdk)
        self.assertTrue((sdk_include / "robot_state_snapshot.h").exists(), "selected snapshot SDK header missing")
        from tests.data_factory.rollout.test_gripper_evidence import ROOT
        common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip())
        if not common.is_absolute():
            common = ROOT / common
        vendor = common.resolve().parent / "src/frcobot_ros2"
        if not (vendor / ".git").exists():
            self.skipTest("pinned vendor object database unavailable")
        pinned = subprocess.check_output(["git", "rev-parse", "HEAD:src/frcobot_ros2"], cwd=ROOT, text=True).strip()
        patch = ROOT / "patches/frcobot_ros2.patch"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for section in patch.read_text().split("diff --git ")[1:]:
                old = section.split(" b/", 1)[0][2:]
                if "--- /dev/null\n" in section:
                    continue
                target = root / old
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(subprocess.check_output(["git", "show", f"{pinned}:{old}"], cwd=vendor))
            checked = subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
            package = root / "fairino_hardware_v3_9_7"
            vendor_package = vendor / "fairino_hardware_v3_9_7"
            includes = [package / "include", vendor_package / "include", vendor_package / "include/fairino_hardware", vendor_package, Path("/usr/include/eigen3")]
            includes.insert(0, sdk_include)
            includes.extend(p for p in Path("/opt/ros/jazzy/include").iterdir() if p.is_dir())
            result = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", *[f"-I{p}" for p in includes],
                                     str(package / "src/fairino_hardware_interface.cpp")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
