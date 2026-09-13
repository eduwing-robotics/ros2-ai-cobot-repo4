"""CPU IPC/liveness checks only. No ROS initialization, model or robot calls."""
import copy
import sys
import threading
import time
import unittest
from unittest import mock

from moveit_msgs.msg import PlanningScene, RobotState
from moveit_msgs.srv import GetStateValidity
from rclpy.serialization import deserialize_message, serialize_message

from tools.data_factory.motion import native_geometry as module
from tools.fr5_data_factory import ContractError, canonical_digest


POSE = {"translation_m": [0., 0., 0.], "rotation_columns": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
BINDING = canonical_digest("revision-scene-contact-timing")
SAMPLES = (("source", ((0., 0., 0., 0., 0., 0., .012),)),)
CLOSED_SAMPLES = (("source", ((0., 0., 0., 0., 0., 0., .01176),)),)
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
    def derived_client(self):
        from tools.data_factory.motion.contact_transition import TIPS
        plan = {"fixture": "derived-intent-cdr"}
        source = {**copy.deepcopy(POSE), "translation_m": [.00255, .014, 0.]}
        context = {"schema_version": "data_factory.request_contact_geometry.v1", "status": "PROSPECTIVE",
            "physical_success": False, "plan_digest": canonical_digest(plan),
            "source_object_id": "source", "planning_frame": "base_link",
            "source_datum": source, "released_datum": copy.deepcopy(source),
            "source_dimensions_m": [.024] * 3,
            "fingertip_boxes": [{"center": [.0101, 0., 0.], "dimensions": [.015, .015, .018]},
                                 {"center": [-.005, 0., 0.], "dimensions": [.015, .015, .018]}],
            "open_m": .021, "release_m": .0126, "jaw_midplane_m": .00255,
            "closed_gap_bound_m": .02446, "orientation_tolerance_rad": .01,
            "proxies": {}}
        for hypothesis, link, touch in (("carried", "gripper_link", TIPS), ("released", "base_link", [])):
            context["proxies"][hypothesis] = {"object_id": "source::prospective:" + hypothesis,
                "link_name": link, "touch_links": touch[:], "dimensions_m": [.024] * 3,
                "translation_m": [0.] * 3, "rotation_xyzw": [0., 0., 0., 1.]}
        context["geometry_digest"] = canonical_digest(context)
        cdr = serialize_message(GetStateValidity.Response(valid=True)).hex()
        client = module.NativeGeometry(urdf="fixture", srdf="fixture", scene=PlanningScene(),
            context=context, plan=plan, deadline=time.monotonic() + 5,
            command=[sys.executable, "-c", CHILD, "normal", cdr])
        self.addCleanup(self.finish, client)
        self.assertEqual(self.result(client)["status"], "READY")
        calls, exchange = [], client._exchange

        def record(request):
            calls.append(copy.deepcopy(request))
            return exchange(request)

        client._exchange = record  # Spy actual IPC/CDR bytes, do not replace them.
        return client, context, calls

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

    def test_derived_intent_uses_native_fk_and_expanded_box_in_actual_query_cdr(self):
        client, context, calls = self.derived_client()
        rows = ((0.,) * 6 + (.021,), (0.,) * 6 + (.01176,), (0.,) * 6 + (.0126,))
        client.submit((("source", rows),), binding=BINDING, deadline=time.monotonic() + 5, derive_intent=True)
        result = self.result(client)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])
        self.assertEqual([v["hypothesis"] for v in calls[0]["variants"]], ["source"])
        self.assertEqual([v["hypothesis"] for v in calls[1]["variants"]], ["carried", "released"])
        self.assertNotEqual(result["source_query_digest"], result["query_digest"])
        intent = result["reference_intent"]
        self.assertEqual(intent["intent"]["geometry_digest"], context["geometry_digest"])
        envelope = intent["carried_envelope"]
        self.assertAlmostEqual(envelope["translation_m"][1], .014)
        for wire in calls[0]["variants"][0]["states_cdr_hex"]:
            self.assertEqual(deserialize_message(bytes.fromhex(wire), RobotState).attached_collision_objects, [])
        for wire in calls[1]["variants"][0]["states_cdr_hex"]:
            state = deserialize_message(bytes.fromhex(wire), RobotState)
            body = state.attached_collision_objects[0]
            self.assertTrue(state.is_diff)
            self.assertEqual(body.link_name, "gripper_link")
            self.assertEqual(list(body.object.primitives[0].dimensions), envelope["dimensions_m"])
            point = body.object.primitive_poses[0].position
            self.assertEqual([point.x, point.y, point.z], envelope["translation_m"])
        released = calls[1]["variants"][1]
        self.assertEqual(len(released["world_objects_cdr_hex"]), 1)
        self.assertEqual(deserialize_message(bytes.fromhex(released["states_cdr_hex"][0]), RobotState).attached_collision_objects, [])
        self.assertEqual(client._context, context)  # Expanded private geometry did not overwrite the task context.

    def test_derived_source_only_reuses_first_query_and_does_not_adopt_previous_result(self):
        client, context, calls = self.derived_client()
        client.submit(CLOSED_SAMPLES, binding=BINDING, deadline=time.monotonic() + 5, derive_intent=True)
        first = self.result(client)
        self.assertIsNotNone(first["reference_intent"]["carried_envelope"])
        calls.clear()
        opened = (("source", ((0.,) * 6 + (.021,),)),)
        client.submit(opened, binding=BINDING, deadline=time.monotonic() + 5, derive_intent=True)
        result = self.result(client)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["source_query_digest"], result["query_digest"])
        self.assertEqual(result["assignments"], (("source", (0,)),))
        self.assertIsNone(result["reference_intent"]["carried_envelope"])
        self.assertEqual(client._context, context)

    def test_derived_prior_is_original_context_bound_and_detached_on_submit_and_return(self):
        client, context, calls = self.derived_client()
        client.submit(CLOSED_SAMPLES, binding=BINDING, deadline=time.monotonic() + 5, derive_intent=True)
        first = self.result(client)
        prior = copy.deepcopy(first["reference_intent"]["intent"])
        expected = copy.deepcopy(prior)
        opened = (("source", ((0.,) * 6 + (.021,),)),)
        client.submit(opened, binding=BINDING, deadline=time.monotonic() + 5,
                      derive_intent=True, prior_intent=prior)
        prior["carried_envelope"]["dimensions_m"][0] = 99.
        first["reference_intent"]["intent"]["carried_envelope"]["translation_m"][0] = 99.
        result = self.result(client)
        self.assertEqual(result["reference_intent"]["evidence"]["prior_intent_digest"], expected["intent_digest"])
        self.assertEqual(result["reference_intent"]["carried_envelope"], expected["carried_envelope"])
        self.assertEqual(result["reference_intent"]["intent"]["geometry_digest"], context["geometry_digest"])
        result["reference_intent"]["carried_envelope"]["translation_m"][0] = 88.
        self.assertNotEqual(result["reference_intent"]["intent"]["carried_envelope"]["translation_m"][0], 88.)
        self.assertEqual(client._context, context)

    def test_derived_foreign_prior_cannot_publish_a_final_geometry_result(self):
        client, _context, calls = self.derived_client()
        client.submit(CLOSED_SAMPLES, binding=BINDING, deadline=time.monotonic() + 5, derive_intent=True)
        prior = self.result(client)["reference_intent"]["intent"]
        prior["geometry_digest"] = canonical_digest("other-context")
        prior["intent_digest"] = canonical_digest({k: v for k, v in prior.items() if k != "intent_digest"})
        calls.clear()
        client.submit(SAMPLES, binding=BINDING, deadline=time.monotonic() + 5,
                      derive_intent=True, prior_intent=prior)
        with self.assertRaisesRegex(ContractError, "CONTACT_REFERENCE_INTENT"):
            self.result(client)
        self.assertEqual(len(calls), 1)  # FK query is not a final admission result.


if __name__ == "__main__":
    unittest.main()
