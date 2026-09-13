"""Synthetic clients and native ROS messages/futures; no node or services."""
import copy
import unittest
from unittest import mock

from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.task import Future

from tools.data_factory.motion.geometry_capture import GeometryCapture
from tools.fr5_data_factory import ContractError


class Client:
    def __init__(self):
        self.ready = False
        self.future = Future()
        self.requests = []
        self.on_call = None

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.on_call:
            self.on_call()
        return self.future

    def wait_for_service(self, *args, **kwargs):
        raise AssertionError("capture must not wait for a service")


class Node:
    def __init__(self):
        self.clients = [Client(), Client()]
        self.created, self.destroyed = [], []
        self.on_create = None
        self.destroy_ok = True

    def create_client(self, kind, endpoint):
        if self.on_create:
            self.on_create(kind, endpoint)
        result = self.clients[len(self.created)]
        self.created.append((kind, endpoint, result))
        return result

    def destroy_client(self, client):
        if self.destroy_ok:
            self.destroyed.append(client)
        return self.destroy_ok


def responses():
    scene = GetPlanningScene.Response(scene=PlanningScene(is_diff=False, robot_model_name="fr5"))
    params = GetParameters.Response(values=[ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=text)
                                           for text in ("<robot name='fr5'/>", "<robot name='fr5'/>")])
    return scene, params


class GeometryCaptureTest(unittest.TestCase):
    def setUp(self):
        self.node = Node()
        self.now = [10.]
        self.capture = GeometryCapture(self.node, deadline=11., clock=lambda: self.now[0])
        self.addCleanup(self.capture.close)

    def complete(self):
        values = responses()
        for client, value in zip(self.node.clients, values):
            client.ready = True
            client.future.set_result(value)
        return values

    def test_readiness_poll_uses_exact_native_requests_without_wait_or_spin(self):
        self.assertEqual(self.node.created, [])
        with mock.patch("rclpy.spin_once", side_effect=AssertionError("no spin")), mock.patch(
                "rclpy.spin_until_future_complete", side_effect=AssertionError("no spin")):
            self.assertIsNone(self.capture.poll())
            self.assertEqual([r[:2] for r in self.node.created], [
                (GetPlanningScene, "/get_planning_scene"), (GetParameters, "/move_group/get_parameters")])
            for _ in range(3):
                self.assertIsNone(self.capture.poll())
            self.assertTrue(all(not client.requests for client in self.node.clients))
            self.node.clients[0].ready = True
            self.assertIsNone(self.capture.poll())
            flags = self.node.clients[0].requests[0].components.components
            self.assertEqual(flags, sum(getattr(PlanningSceneComponents, name) for name in (
                "SCENE_SETTINGS", "ROBOT_STATE", "ROBOT_STATE_ATTACHED_OBJECTS", "WORLD_OBJECT_NAMES", "WORLD_OBJECT_GEOMETRY",
                "OCTOMAP", "TRANSFORMS", "ALLOWED_COLLISION_MATRIX", "LINK_PADDING_AND_SCALING", "OBJECT_COLORS")))
            self.node.clients[1].ready = True
            self.assertIsNone(self.capture.poll())
            self.assertEqual(self.node.clients[1].requests[0].names, ["robot_description", "robot_description_semantic"])
            self.assertIsNone(self.capture.poll())
            self.assertEqual([len(client.requests) for client in self.node.clients], [1, 1])

    def test_returns_detached_full_scene_once_only_after_both_done(self):
        scene, params = responses()
        for client in self.node.clients:
            client.ready = True
        self.node.clients[0].future.set_result(scene)
        with mock.patch.object(self.node.clients[1].future, "result", side_effect=AssertionError("premature result")):
            self.assertIsNone(self.capture.poll())
        self.node.clients[1].future.set_result(params)
        result = self.capture.poll()
        self.assertEqual(set(result), {"scene", "urdf", "srdf"})
        self.assertEqual(result["scene"], scene.scene)
        self.assertIsNot(result["scene"], scene.scene)
        result["scene"].robot_model_name = "changed"
        self.assertEqual(scene.scene.robot_model_name, "fr5")
        scene.scene.name = "late change"
        self.assertNotEqual(result["scene"].name, scene.scene.name)
        self.now[0] = 20.
        self.assertIsNone(self.capture.poll())
        self.assertTrue(self.capture.close())
        self.assertEqual(self.node.destroyed, self.node.clients)

    def test_expired_and_late_responses_never_reissue_or_renew_deadline(self):
        self.complete()
        self.now[0] = 11.
        with self.assertRaisesRegex(ContractError, "GEOMETRY_CAPTURE_TIMEOUT"):
            self.capture.poll()
        self.assertFalse(self.node.created)
        self.assertIsNone(self.capture.poll())

    def test_inflight_timeout_discards_even_completed_late_responses(self):
        for completed in (False, True):
            node, now = Node(), [10.]
            capture = GeometryCapture(node, deadline=11., clock=lambda: now[0])
            self.addCleanup(capture.close)
            for client in node.clients:
                client.ready = True
            self.assertIsNone(capture.poll())
            now[0] = 11.
            if completed:
                for client, response in zip(node.clients, responses()):
                    client.future.set_result(response)
            with self.subTest(completed=completed), mock.patch.object(
                    node.clients[0].future, "result", side_effect=AssertionError("late result must not be consumed")):
                with self.assertRaisesRegex(ContractError, "GEOMETRY_CAPTURE_TIMEOUT"):
                    capture.poll()
            self.assertEqual([len(client.requests) for client in node.clients], [1, 1])
            self.assertEqual(node.destroyed, node.clients)
            self.assertIsNone(capture.poll())

    def test_deadline_during_client_setup_and_conversion_discards_result(self):
        self.node.on_create = lambda *_: self.now.__setitem__(0, 11.)
        with self.assertRaisesRegex(ContractError, "GEOMETRY_CAPTURE_TIMEOUT"):
            self.capture.poll()
        self.assertFalse(self.node.clients[0].requests)
        self.assertEqual(self.node.destroyed, [self.node.clients[0]])
        other = GeometryCapture(Node(), deadline=12., clock=lambda: self.now[0])
        self.addCleanup(other.close)
        for client, value in zip(other._node.clients, responses()):
            client.ready = True
            client.future.set_result(value)
        native_copy = copy.deepcopy
        def late(value):
            result = native_copy(value)
            if isinstance(value, dict) and "scene" in value:
                self.now[0] = 12.
            return result
        with mock.patch("tools.data_factory.motion.geometry_capture.copy.deepcopy", side_effect=late):
            with self.assertRaisesRegex(ContractError, "GEOMETRY_CAPTURE_TIMEOUT"):
                other.poll()

    def test_pending_close_cancels_native_futures_without_waiting_for_done(self):
        for client in self.node.clients:
            client.ready = True
        self.assertIsNone(self.capture.poll())
        self.assertTrue(self.capture.close())
        self.assertTrue(all(client.future.cancelled() and not client.future.done() for client in self.node.clients))
        self.assertEqual(self.node.destroyed, self.node.clients)
        self.assertTrue(self.capture.close())
        self.assertEqual(len(self.node.destroyed), 2)
        for client, value in zip(self.node.clients, responses()):
            client.future.set_result(value)  # A late callback cannot republish.
        self.assertIsNone(self.capture.poll())

    def test_reentrant_close_during_first_call_discards_new_future(self):
        self.node.clients[0].ready = True
        self.node.clients[0].on_call = self.capture.close
        self.assertIsNone(self.capture.poll())
        self.assertTrue(self.node.clients[0].future.cancelled())
        self.assertFalse(self.node.clients[1].requests)
        self.assertTrue(self.capture.close())

    def test_cancelled_read_and_malformed_responses_fail_without_bundle(self):
        for change in ("cancelled", "diff", "model", "count", "type", "empty", "wrong_response"):
            node = Node()
            capture = GeometryCapture(node, deadline=11., clock=lambda: 10.)
            self.addCleanup(capture.close)
            scene, params = responses()
            if change == "diff": scene.scene.is_diff = True
            elif change == "model": scene.scene.robot_model_name = " "
            elif change == "count": params.values.pop()
            elif change == "type": params.values[1].type = ParameterType.PARAMETER_INTEGER
            elif change == "empty": params.values[0].string_value = ""
            elif change == "wrong_response": scene = object()
            for client, value in zip(node.clients, (scene, params)):
                client.ready = True
                client.future.set_result(value)
            if change == "cancelled":
                node.clients[0].future = Future()
                node.clients[0].future.cancel()
            with self.subTest(change=change), self.assertRaisesRegex(ContractError, "GEOMETRY_CAPTURE_(CANCELLED|RESPONSE)"):
                capture.poll()
            self.assertIsNone(capture.poll())

    def test_primary_setup_or_result_error_survives_retryable_cleanup_failure(self):
        original = RuntimeError("native result failure")
        self.complete()
        self.node.clients[0].future.set_exception(original)
        self.node.destroy_ok = False
        with self.assertRaises(RuntimeError) as raised:
            self.capture.poll()
        self.assertIs(raised.exception, original)
        self.assertFalse(self.capture.close())
        self.node.destroy_ok = True
        self.assertTrue(self.capture.close())
        node = Node()
        count = [0]
        def create(*_):
            count[0] += 1
            if count[0] == 2:
                raise original
        node.on_create = create
        capture = GeometryCapture(node, deadline=11., clock=lambda: 10.)
        with self.assertRaises(RuntimeError) as raised:
            capture.poll()
        self.assertIs(raised.exception, original)
        self.assertEqual(node.destroyed, [node.clients[0]])
        self.assertTrue(capture.close())


if __name__ == "__main__":
    unittest.main()
