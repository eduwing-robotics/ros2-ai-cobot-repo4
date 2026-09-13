"""Actual ROS message serialization and native capture methods, without ROS init."""
import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Image, JointState

from tools.fr5_data_factory import ContractError, canonical_digest
from tools.data_factory.motion.moveit_transport import RosMoveItTransport
from tools.data_factory.motion.pickup_executor import PickupExecutor
from tools.data_factory.rollout.finite_plan import JOINTS


class PolicyObservationTest(unittest.TestCase):
    def task_observer(self):
        """An already-admitted synthetic task; not physical plan qualification."""
        from tools.data_factory.one_job import OneJob
        from tools.data_factory.rollout.task_authority import task_scope
        observation = {"source_timestamps_s": dict.fromkeys(("state", "camera1", "camera2"), 100.),
                       "observation.state": [0.] * 7,
                       "observation.images.camera1": {"data_hex": "010203"}}
        stream = SimpleNamespace(poll=mock.Mock(return_value=observation), close=mock.Mock())
        transport = SimpleNamespace(policy_observation_stream=mock.Mock(return_value=stream),
            poll_policy_observation=lambda observer: observer.poll(),
            poll_active=mock.Mock(return_value=None))
        executor = PickupExecutor(transport, source_clock=lambda: 100., monotonic_clock=lambda: 20.)
        source = {"robot_system_id": "fr5", "steps": []}
        topics = {"camera1": "/up", "camera2": "/wrist"}
        proposal = {"checkpoint": {}, "instruction": "pick", "schema_version": "test-observation-fixture",
                    "robot_description": "test", "velocity_scaling": .1, "period_s": 1/30,
                    "max_observation_age_s": .3, "runtime_inputs": {"camera_topics": topics}}
        plan = {"run_id": "active-task", "scene_binding": {}, "learned_source_program": source,
                "learned_proposal": proposal}
        grant = {"schema_version": "data_factory.learned_task_grant.v1", "grant_id": "grant",
                 "issued_by": "test", "run_id": plan["run_id"], "scope": task_scope(source, {}, proposal),
                 "deadline_s": 200., "terminal_reserve_s": 10., "max_outputs": 2, "revoked": False}
        grant["grant_digest"] = canonical_digest(grant)
        digest = canonical_digest(plan)
        run = {"plan": plan, "digest": digest, "state": "EXECUTING", "task_grant": grant,
               "task_deadline": 40., "cancel_event": threading.Event(),
               "execution": {"lease_id": "lease", "lease_deadline": 40., "active": True}}
        executor.runs[plan["run_id"]] = run
        recorder = mock.Mock(side_effect=AssertionError("observation called recorder"))
        job = OneJob(recorder, executor.process)
        job.state, job.approval_scope = "EXECUTING", "SCOPED_TASK_GRANT"
        job.executor_state = "EXECUTING"
        job.run_id, job.plan_digest, job.lease_id = plan["run_id"], digest, "lease"
        job.plan_envelope = {"plan": plan}
        job.execution_evidence = {"actual_owner_event": "unchanged"}
        return job, executor, transport, stream, run, topics

    def test_normal_owner_observation_does_not_require_or_replace_execution_trace(self):
        job, executor, transport, stream, run, topics = self.task_observer()
        with mock.patch("tools.data_factory.rollout.finite_plan.validate_execution_trace",
                        side_effect=AssertionError("observation required finite execution trace")):
            first = job.observe_policy(topics)
            self.assertTrue(first["ok"], first)
            self.assertEqual(first["code"], "LEARNED_OBSERVATION")
            self.assertTrue(run["execution"]["active"])
            self.assertEqual(job.execution_evidence, {"actual_owner_event": "unchanged"})
            self.assertEqual(run["execution"]["lease_deadline"], 40.)
            transport.poll_active.assert_not_called()
            for _ in range(4):
                self.assertTrue(job.observe_policy(topics)["ok"])
            self.assertEqual(transport.policy_observation_stream.call_count, 1)
            bodies = [r for _, r in executor.cache.values() if "observation" in (r.get("data") or {})]
            self.assertEqual(len(bodies), 1)
        executor.close()
        stream.close.assert_called_once()

    def test_owner_observation_pending_is_not_failure_or_task_outcome(self):
        job, executor, transport, stream, run, topics = self.task_observer()
        stream.poll.return_value = None
        result = job.observe_policy(topics)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["code"], "LEARNED_OBSERVATION_PENDING")
        self.assertNotIn("observation", result)
        self.assertEqual(job.state, "EXECUTING")
        self.assertIsNone(job.semantic)
        self.assertTrue(job.observe_policy(topics)["ok"])
        self.assertEqual(transport.policy_observation_stream.call_count, 1)
        executor.close()

    def test_owner_observation_binding_and_late_cancel_remain_enforced(self):
        for field, value, code in (("run_id", "foreign", "RUN_NOT_FOUND"),
                                   ("plan_digest", "f" * 64, "PLAN_DIGEST_MISMATCH"),
                                   ("lease_id", "foreign", "LEASE_BINDING")):
            job, executor, transport, stream, run, topics = self.task_observer()
            setattr(job, field, value)
            self.assertEqual(job.observe_policy(topics)["code"], code)
            self.assertEqual(job.executor_state, "EXECUTING")
            transport.policy_observation_stream.assert_not_called()
        job, executor, transport, stream, run, topics = self.task_observer()
        self.assertEqual(job.observe_policy({**topics, "camera1": "/foreign"})["code"], "LEARNED_CAMERA_MAPPING")
        transport.policy_observation_stream.assert_not_called()
        def cancel_during_read():
            run["cancel_event"].set()
            return {"source_timestamps_s": dict.fromkeys(("state", "camera1", "camera2"), 100.)}
        stream.poll.side_effect = cancel_during_read
        result = job.observe_policy(topics)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "LEARNED_CANCELLED")
        self.assertNotIn("observation", result)
        self.assertEqual(job.execution_evidence, {"actual_owner_event": "unchanged"})
        executor.close()

    def test_cancel_retires_cached_observation_before_idempotent_retry(self):
        job, executor, transport, stream, run, topics = self.task_observer()
        with mock.patch.object(executor, "_process", wraps=executor._process) as process:
            self.assertTrue(job.observe_policy(topics)["ok"])
            request = process.call_args.args[0]
        run["cancel_event"].set()
        retry = executor.process(request)
        self.assertFalse(retry["ok"])
        self.assertIsNone(retry["data"])
        self.assertEqual(stream.poll.call_count, 1)
        executor.close()
        stream.close.assert_called_once()
        self.assertEqual(job.execution_evidence, {"actual_owner_event": "unchanged"})

    def test_malformed_observation_retry_preserves_schema_rejection(self):
        job, executor, transport, stream, run, topics = self.task_observer()
        request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "malformed",
                   "op": "observe_policy", "payload": {
                       "run_id": job.run_id, "plan_digest": job.plan_digest,
                       "camera_topics": topics, "max_observation_age_s": .3}}
        for _ in range(2):
            response = executor.process(request)
            self.assertFalse(response["ok"])
            self.assertEqual(response["code"], "LEARNED_OBSERVATION_SCHEMA")
            self.assertIsNone(response["data"])
        transport.poll_active.assert_not_called()
        transport.policy_observation_stream.assert_not_called()

    def test_transport_observation_pump_does_not_wait_for_motion_completion(self):
        transport = object.__new__(RosMoveItTransport)
        transport.node = object()
        transport._rclpy = SimpleNamespace(spin_once=mock.Mock())
        stream = SimpleNamespace(poll=mock.Mock(return_value={"sample": "fresh"}))
        # Neither an active action nor an idle chunk boundary prevents callbacks.
        for active in (object(), None):
            transport._active = active
            self.assertEqual(transport.poll_policy_observation(stream), {"sample": "fresh"})
            transport._rclpy.spin_once.assert_called_with(transport.node, timeout_sec=0.0)
            self.assertIs(transport._active, active)

    def observer(self):
        callbacks, destroyed = {}, []
        transport = object.__new__(RosMoveItTransport)
        transport._active = object()  # Observation is not actuation admission.
        transport._execution_locked = False
        transport._joint_state = None
        transport._joint_state_received_at = None
        transport._clock = lambda: 20.
        def subscribe(_type, topic, callback, _qos):
            callbacks[topic] = callback
            return topic
        transport.node = SimpleNamespace(create_subscription=subscribe,
            destroy_subscription=destroyed.append,
            get_parameter=lambda _: SimpleNamespace(value=False))
        transport._rclpy = SimpleNamespace(spin_once=mock.Mock(side_effect=AssertionError("poll spun ROS")))
        stream = transport.policy_observation_stream({"camera1": "/up", "camera2": "/wrist"}, .3)
        return transport, stream, callbacks, destroyed

    @mock.patch("tools.data_factory.motion.policy_observation.time.time", return_value=100.)
    def test_stream_observes_without_waiting_for_active_motion_or_spinning(self, _wall):
        transport, stream, callbacks, destroyed = self.observer()
        active = transport._active
        try:
            self.assertIsNone(stream.poll())
            state = JointState(name=list(JOINTS), position=[0.] * 7)
            state.header.stamp.sec = 100
            transport._joint_state = state
            transport._joint_state_received_at = 20.
            image = Image(height=1, width=1, encoding="rgb8", step=3, data=b"\x01\x02\x03")
            image.header.stamp.sec = 100
            for callback in callbacks.values():
                callback(image)
            first = stream.poll()
            self.assertEqual(stream.poll(), first)
            self.assertIs(transport._active, active)
            transport._rclpy.spin_once.assert_not_called()
            for _ in range(100):
                callbacks["/up"](image)
            self.assertEqual(len(stream.frames), 2)
            self.assertEqual(len(stream.subscriptions), 2)
            first["observation.state"][0] = 12.
            self.assertEqual(stream.poll()["observation.state"][0], 0.)
            with mock.patch("tools.data_factory.motion.policy_observation.time.time", return_value=100.31):
                transport._clock = lambda: 20.31
                stream.clock = transport._clock
                with self.assertRaisesRegex(ContractError, "LEARNED_STALE_OBSERVATION"):
                    stream.poll()
        finally:
            stream.close()
        self.assertCountEqual(destroyed, ["/up", "/wrist"])
        self.assertFalse(stream.frames)
        callbacks["/up"](image)  # Late callbacks cannot revive a closed source.
        self.assertFalse(stream.frames)
        with self.assertRaisesRegex(ContractError, "LEARNED_OBSERVATION_CLOSED"):
            stream.poll()
        stream.close()
        self.assertEqual(len(destroyed), 2)

    @mock.patch("tools.data_factory.motion.policy_observation.time.time", return_value=100.)
    def test_partial_subscription_failure_releases_created_subscription(self, _wall):
        from tools.data_factory.motion.policy_observation import PolicyObservationStream
        node = SimpleNamespace(
            get_parameter=lambda _: SimpleNamespace(value=False),
            create_subscription=mock.Mock(side_effect=["first", RuntimeError("DDS setup")]),
            destroy_subscription=mock.Mock())
        with self.assertRaisesRegex(RuntimeError, "DDS setup"):
            PolicyObservationStream(node, lambda: (None, None),
                {"camera1": "/up", "camera2": "/wrist"}, .3, clock=lambda: 20.)
        node.destroy_subscription.assert_called_once_with("first")

    @mock.patch("tools.data_factory.motion.policy_observation.time.time", return_value=100.)
    def test_legacy_capture_subscription_setup_consumes_original_deadline(self, _wall):
        now, destroyed = [20.], []
        transport = object.__new__(RosMoveItTransport)
        transport._active, transport._execution_locked = None, False
        transport._joint_state = transport._joint_state_received_at = None
        transport._clock = lambda: now[0]
        transport.graph_timeout_s = .1
        def subscribe(_type, topic, _callback, _qos):
            now[0] += .06
            return topic
        transport.node = SimpleNamespace(create_subscription=subscribe,
            destroy_subscription=destroyed.append,
            get_parameter=lambda _: SimpleNamespace(value=False))
        transport._rclpy = SimpleNamespace(spin_once=mock.Mock())
        with mock.patch("tools.data_factory.motion.moveit_transport.time.monotonic", side_effect=lambda: now[0]):
            with self.assertRaisesRegex(ContractError, "LEARNED_OBSERVATION_UNAVAILABLE"):
                transport.capture_policy_observation({"camera1": "/up", "camera2": "/wrist"}, .3)
        transport._rclpy.spin_once.assert_not_called()
        self.assertCountEqual(destroyed, ["/up", "/wrist"])

    def capture(self, *, stamp=100., received=20., state=True, malformed=False,
                active=False, simulated=False, after_conversion=100., executor=False,
                paused_system=False, cached=False):
        def native(message):
            message.header.stamp.sec = int(stamp)
            message.header.stamp.nanosec = round((stamp - int(stamp)) * 1e9)
            return deserialize_message(serialize_message(message), type(message))
        joint = native(JointState(name=list(reversed(JOINTS)), position=[.012, 6., 5., 4., 3., 2., 1.]))
        # BGR with row padding: use the recorder's existing conversion behavior.
        image = native(Image(height=1, width=1, encoding="bgr8", step=4,
                             data=bytes([1, 2, 3, 99]) if not malformed else b""))
        callbacks, destroyed = {}, []
        t = object.__new__(RosMoveItTransport)
        t._active, t._execution_locked = (object() if active else None), False
        steady = [20.]
        t._clock = lambda: steady[0]
        t.graph_timeout_s = .001
        t._joint_state = joint if cached else None
        t._joint_state_received_at = None
        def subscribe(_type, topic, callback, _qos):
            callbacks[topic] = callback
            return topic
        t.node = SimpleNamespace(create_subscription=subscribe,
                                 destroy_subscription=destroyed.append,
                                 get_parameter=lambda _: SimpleNamespace(value=simulated))
        def spin(*_, **__):
            if paused_system:
                steady[0] = 21.
            if state:
                t._joint_state, t._joint_state_received_at = joint, received
            for callback in callbacks.values():
                callback(image)
        t._rclpy = SimpleNamespace(spin_once=spin)
        self.callbacks, self.destroyed, self.transport = callbacks, destroyed, t
        options = {"camera_topics": {"camera1": "/up", "camera2": "/wrist"}, "max_observation_age_s": .3}
        with mock.patch("tools.data_factory.motion.moveit_transport.time.time", side_effect=[100., 100., after_conversion]):
            if executor:
                e = PickupExecutor(t)
                return e.process({"schema_version": "fr5.pickup_executor.command.v4", "op_id": "capture-1",
                                  "op": "capture_observation", "payload": options})
            return t.capture_policy_observation(options["camera_topics"], options["max_observation_age_s"])

    def test_serialized_capture_preserves_source_stamp_full_state_rgb_and_json(self):
        result = json.loads(json.dumps(self.capture(executor=True)))
        self.assertTrue(result["ok"], result)
        observation = result["data"]["observation"]
        self.assertEqual(observation["observation.state"], [1., 2., 3., 4., 5., 6., .012])
        self.assertEqual(observation["source_timestamps_s"], dict.fromkeys(("state", "camera1", "camera2"), 100.))
        self.assertEqual(observation["observation.images.camera1"], {
            "dtype": "uint8", "color_space": "RGB", "shape": [1, 1, 3], "data_hex": "030201"})
        self.assertCountEqual(self.destroyed, ["/up", "/wrist"])
        self.assertIsNone(self.transport._active)

    def test_stale_paused_future_receipt_and_conversion_fail_without_goals(self):
        for options, code in [({"stamp": 99.}, "LEARNED_STALE_OBSERVATION"),
                              ({"stamp": 101.}, "LEARNED_STALE_OBSERVATION"),
                              ({"received": 21.}, "LEARNED_STALE_OBSERVATION"),
                              ({"received": 19.}, "LEARNED_OBSERVATION_UNAVAILABLE"),
                              ({"state": False}, "LEARNED_OBSERVATION_UNAVAILABLE"),
                              ({"cached": True}, "LEARNED_OBSERVATION_UNAVAILABLE"),
                              ({"paused_system": True}, "LEARNED_SOURCE_CLOCK"),
                              ({"after_conversion": 100.4}, "LEARNED_STALE_OBSERVATION"),
                              ({"malformed": True}, "LEARNED_IMAGE"),
                              ({"simulated": True}, "LEARNED_SOURCE_CLOCK"),
                              ({"active": True}, "ROS_EXEC_ACTIVE")]:
            with self.subTest(options=options):
                with self.assertRaises(ContractError) as error:
                    self.capture(**options)
                self.assertEqual(error.exception.code, code)
            self.assertCountEqual(self.destroyed, list(self.callbacks))

    def test_capture_retry_cache_retains_one_body_and_never_recaptures_retired_ids(self):
        clock, calls = [100., 20.], []
        def capture(*_):
            calls.append(True)
            return {"source_timestamps_s": dict.fromkeys(("state", "camera1", "camera2"), clock[0]),
                    "observation.images.camera1": {"data_hex": "010203"},
                    "observation.images.camera2": {"data_hex": "040506"}}
        executor = PickupExecutor(SimpleNamespace(capture_policy_observation=capture),
                                  source_clock=lambda: clock[0], monotonic_clock=lambda: clock[1])
        request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "capture-0",
                   "op": "capture_observation", "payload": {"camera_topics": {"camera1": "/up", "camera2": "/wrist"},
                                                            "max_observation_age_s": .3}}
        original = executor.process(request)
        self.assertEqual(executor.process(request), original)
        self.assertEqual(len(calls), 1)
        for index in range(1, 8):
            self.assertTrue(executor.process({**request, "op_id": f"capture-{index}"})["ok"])
            bodies = [response for _, response in executor.cache.values()
                      if isinstance(response.get("data"), dict) and "observation" in response["data"]]
            self.assertEqual(len(bodies), 1)
        self.assertEqual(executor.process(request)["code"], "LEARNED_OBSERVATION_RETIRED")
        self.assertEqual(len(calls), 8)
        self.assertIn("observation", original["data"])  # caller's returned value is immutable
        changed = {**request, "payload": {**request["payload"], "max_observation_age_s": .2}}
        self.assertEqual(executor.process(changed)["code"], "OP_ID_CONFLICT")
        clock[1] += .31  # paused source clock cannot preserve a cached image
        expired = executor.process({**request, "op_id": "capture-7"})
        self.assertEqual(expired["code"], "LEARNED_STALE_OBSERVATION")
        self.assertFalse(expired["ok"])
        self.assertIsNone(expired["data"])
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(response["data"] is None for _, response in executor.cache.values()))
        self.assertTrue(executor.process({**request, "op_id": "source-age"})["ok"])
        clock[0] += 1.
        self.assertEqual(executor.process({**request, "op_id": "source-age"})["code"], "LEARNED_STALE_OBSERVATION")
        self.assertTrue(executor.process({**request, "op_id": "shutdown"})["ok"])
        executor.close()
        self.assertIsNone(executor.cache["shutdown"][1]["data"])
        self.assertEqual(executor.process({**request, "op_id": "shutdown"})["code"], "LEARNED_OBSERVATION_RETIRED")

    def test_child_never_captures_after_plan_or_on_non_native_transport(self):
        request = {"schema_version": "fr5.pickup_executor.command.v4", "op_id": "capture-1",
                   "op": "capture_observation", "payload": {"camera_topics": {"camera1": "/up", "camera2": "/wrist"},
                                                            "max_observation_age_s": .3}}
        executor = PickupExecutor(SimpleNamespace())
        self.assertEqual(executor.process(request)["code"], "LEARNED_OBSERVATION_UNAVAILABLE")
        executor.runs["previous"] = {"state": "PLANNED"}
        request["op_id"] = "capture-2"
        self.assertEqual(executor.process(request)["code"], "ONE_JOB_ONLY")
