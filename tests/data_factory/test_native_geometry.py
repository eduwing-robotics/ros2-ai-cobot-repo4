"""CPU IPC/liveness checks only. No ROS initialization, model or robot calls."""
import sys
import threading
import time
import unittest
from unittest import mock

from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import GetStateValidity
from rclpy.serialization import serialize_message

from tools.data_factory.motion import native_geometry as module
from tools.fr5_data_factory import ContractError, canonical_digest


POSE = {"translation_m": [0., 0., 0.], "rotation_columns": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
BINDING = canonical_digest("revision-scene-contact-timing")
SAMPLES = (("source", ((0., 0., 0., 0., 0., 0., .012),)),)
CHILD = r'''
import json, sys, time
mode, cdr = sys.argv[1:]
for line in sys.stdin:
    req = json.loads(line)
    out = dict(op=req['op'], id=req['id'], ok=True)
    if req['op'] == 'query':
        if mode == 'stall':
            time.sleep(30)
        if mode == 'wrong_id':
            out['id'] += 1
        if mode == 'malformed':
            print('{"ok":true,"ok":false}', flush=True)
            continue
        out['variants'] = [dict(hypothesis=v['hypothesis'], samples=[
            dict(response_cdr_hex=cdr + ('00' if mode == 'trailing' else ''), saturated=mode == 'saturated',
                 gripper_pose=dict(translation_m=[0.,0.,0.], rotation_columns=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]))
            for row in v['states_cdr_hex']]) for v in req['variants']]
    print(json.dumps(out), flush=True)
'''


class NativeGeometryTest(unittest.TestCase):
    def client(self, mode="normal", valid=True):
        state = RobotState(is_diff=True)
        state.joint_state.name = ["j1", "j2", "j3", "j4", "j5", "j6", "finger_right_joint"]
        state.joint_state.position = [0., 0., 0., 0., 0., 0., .012]
        patches = [mock.patch.object(module, "bind_native_request_geometry", return_value=lambda *_: (state, [])),
                   mock.patch.object(module, "bind_request_contacts", return_value=lambda *_a, **_kw: False)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        cdr = serialize_message(GetStateValidity.Response(valid=valid)).hex()
        client = module.NativeGeometry(urdf="fixture", srdf="fixture", scene=PlanningScene(),
            context={}, plan={}, deadline=time.monotonic() + 5, command=[sys.executable, "-c", CHILD, mode, cdr])
        self.addCleanup(self.finish, client)
        self.assertEqual(self.result(client)["status"], "READY")
        return client

    def result(self, client, binding=BINDING):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            result = client.poll(binding=binding)
            if result is not None:
                return result
            time.sleep(.001)
        self.fail("CPU helper did not finish within test harness deadline")

    def finish(self, client):
        end = time.monotonic() + 5
        while not client.close():
            if time.monotonic() >= end:
                self.fail("owned CPU process/worker did not close")
            time.sleep(.001)

    def test_native_cdr_result_keeps_binding_and_is_not_execution_authority(self):
        client = self.client()
        client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
        with self.assertRaisesRegex(ContractError, "NATIVE_GEOMETRY_BUSY"):
            client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
        result = self.result(client)
        self.assertEqual(result["binding"], BINDING)
        self.assertEqual(result["status"], "CHECKED")
        self.assertEqual(set(result), {"status", "binding", "query_digest", "initialization_digest",
                                      "scene_cdr_digest", "variants"})
        sample = result["variants"][0]["samples"][0]
        self.assertTrue(sample["allowed"])
        self.assertEqual(sample["gripper_pose"], POSE)
        self.assertEqual(sample["response"], GetStateValidity.Response(valid=True))

    def test_superseded_query_is_discarded_and_next_query_uses_same_cpu_process(self):
        client = self.client()
        process = client._process
        client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
        with self.assertRaisesRegex(ContractError, "NATIVE_GEOMETRY_SUPERSEDED"):
            self.result(client, canonical_digest("new-scene"))
        self.assertIsNone(client.poll(binding=BINDING))
        next_binding = canonical_digest("new-revision-scene-contact-timing")
        client.submit(SAMPLES, binding=next_binding, deadline=time.monotonic() + 5)
        self.assertEqual(self.result(client, next_binding)["binding"], next_binding)
        self.assertIs(client._process, process)

    def test_blocked_native_query_does_not_block_owner_poll_or_owned_teardown(self):
        client = self.client("stall")
        client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
        # Simulated sole-owner iterations remain possible while native work is
        # blocked. A regression calling result()/readline() in poll would hang.
        for _ in range(100):
            self.assertIsNone(client.poll(binding=BINDING))
        self.finish(client)
        self.assertIsNotNone(client._process.poll())
        with self.assertRaisesRegex(ContractError, "NATIVE_GEOMETRY_CLOSED"):
            client.poll(binding=BINDING)

    def test_expired_query_cannot_be_consumed_even_if_worker_has_finished(self):
        client = self.client()
        client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
        # Even a completed valid result must satisfy the consumer's deadline.
        client._future.result(timeout=5)  # Test harness only, not owner code.
        client._deadline = time.monotonic() - 1
        with self.assertRaisesRegex(ContractError, "NATIVE_GEOMETRY_TIMEOUT"):
            client.poll(binding=BINDING)
        self.finish(client)

    def test_exact_ipc_framing_and_correspondence(self):
        for mode in ("wrong_id", "malformed", "trailing"):
            with self.subTest(mode=mode):
                client = self.client(mode)
                client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
                with self.assertRaisesRegex(ContractError, "NATIVE_GEOMETRY_PROTOCOL"):
                    self.result(client)
                self.finish(client)

    def test_saturated_or_unexplained_collision_never_becomes_allowed(self):
        for mode, valid in (("saturated", True), ("normal", False)):
            with self.subTest(mode=mode):
                client = self.client(mode, valid)
                client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5)
                self.assertFalse(self.result(client)["variants"][0]["samples"][0]["allowed"])
                self.finish(client)

    def test_mutable_or_nonfinite_query_cannot_cross_worker_boundary(self):
        client = self.client()
        for samples in ([*SAMPLES], (("source", ([0.] * 7,)),),
                        (("source", ((float("nan"),) * 7,)),), (*SAMPLES, *SAMPLES)):
            with self.subTest(samples=samples), self.assertRaisesRegex(ContractError, "NATIVE_GEOMETRY_SAMPLES"):
                client.submit(samples, binding=BINDING, deadline=time.monotonic() + 5)
        self.assertIsNone(client._future)

    def test_close_during_setup_prevents_a_late_subprocess_spawn(self):
        entered, release = threading.Event(), threading.Event()

        def delayed_bind(*_args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test harness did not release setup")
            return lambda *_: None

        with mock.patch.object(module, "bind_native_request_geometry", side_effect=delayed_bind), \
                mock.patch.object(module, "bind_request_contacts", return_value=lambda *_: False), \
                mock.patch.object(module.subprocess, "Popen") as spawn:
            client = module.NativeGeometry(urdf="fixture", srdf="fixture", scene=PlanningScene(),
                context={}, plan={}, deadline=time.monotonic() + 5, command=[sys.executable, "-c", "pass"])
            self.addCleanup(self.finish, client)
            try:
                self.assertTrue(entered.wait(5))
                self.assertFalse(client.close())
            finally:
                release.set()
            self.finish(client)
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
