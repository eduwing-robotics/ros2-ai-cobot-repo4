"""Actual native producer/queue with a synthetic sampler, no model or hardware."""
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from lerobot.policies.rtc import RTCConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.rollout.inference.factory import RTCInferenceConfig, create_inference_engine
from lerobot.rollout.inference.rtc import supports_rtc_inference
from lerobot_strategy_fr5.acknowledged_queue import (
    ObservationProvenance,
    RawActionIndex,
    start_with_acknowledged_queue,
)


class SyntheticSampler(SmolVLAPolicy):
    """Keep native predict/capability methods; replace only expensive sampling."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(adapt_to_pi_aloha=False, rtc_config=RTCConfig(enabled=False))
        self._queues = {}
        self.calls = 0
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        self.calls += 1
        if self.calls == 2:
            self.second_started.set()
            if not self.release_second.wait(3.):
                raise TimeoutError("synthetic sampler was not released")
        return batch["observation.state"].unsqueeze(1).repeat(1, 4, 1) + self.calls


class IdentityProcessor:
    steps = ()

    def __call__(self, value):
        return value


class ProvenanceSampler(SyntheticSampler):
    def __init__(self, *, fail_first=False):
        super().__init__()
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.fail_first = fail_first

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            if not self.release_first.wait(3.):
                raise TimeoutError("synthetic sampler was not released")
            if self.fail_first:
                raise RuntimeError("synthetic first inference failure")
        return batch["observation.state"].unsqueeze(1).repeat(1, 4, 1)


class NativeAsyncTest(unittest.TestCase):
    @staticmethod
    def _provenance(stamp):
        return ObservationProvenance(
            "sha256:" + str(int(stamp)) * 64,
            "SYSTEM_TIME",
            (("camera1", stamp), ("camera2", stamp), ("state", stamp)),
        )

    def test_native_output_binds_sampled_observation_not_latest_notification(self):
        policy = ProvenanceSampler()
        engine = self._engine(policy)
        queue = start_with_acknowledged_queue(
            engine, require_observation_provenance=True,
        )
        thread = engine._rtc_thread
        keys = [f"j{i}" for i in range(7)]
        first, second = self._provenance(1.), self._provenance(2.)
        try:
            engine.notify_observation(queue.observed_input(dict.fromkeys(keys, 1.), first))
            engine.resume()
            self.assertTrue(policy.first_started.wait(2.))
            engine.notify_observation(queue.observed_input(dict.fromkeys(keys, 2.), second))
            policy.release_first.set()
            deadline = time.monotonic() + 2.
            while queue.generation < 1 and time.monotonic() < deadline:
                time.sleep(.005)
            snapshot = queue.snapshot()
            self.assertEqual(snapshot.observation_provenance, (first,) * 4)
            torch.testing.assert_close(snapshot.processed_actions, torch.ones(4, 7))

            queue.advance_if_current(
                generation=snapshot.generation,
                expected_index=snapshot.queue_index,
                count=4,
            )
            deadline = time.monotonic() + 2.
            while queue.generation < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            following = queue.snapshot()
            self.assertEqual(following.observation_provenance, (second,) * 4)
            torch.testing.assert_close(following.processed_actions, torch.full((4, 7), 2.))
        finally:
            policy.release_first.set()
            engine.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(engine.failed)

    def test_failed_inference_does_not_reuse_its_observation_identity(self):
        policy = ProvenanceSampler(fail_first=True)
        engine = self._engine(policy)
        queue = start_with_acknowledged_queue(
            engine, require_observation_provenance=True,
        )
        thread = engine._rtc_thread
        keys = [f"j{i}" for i in range(7)]
        first, second = self._provenance(1.), self._provenance(2.)
        try:
            engine.notify_observation(queue.observed_input(dict.fromkeys(keys, 1.), first))
            engine.resume()
            self.assertTrue(policy.first_started.wait(2.))
            engine.notify_observation(queue.observed_input(dict.fromkeys(keys, 2.), second))
            policy.release_first.set()
            deadline = time.monotonic() + 3.
            while queue.generation < 1 and time.monotonic() < deadline:
                time.sleep(.005)
            snapshot = queue.snapshot()
            self.assertEqual(snapshot.observation_provenance, (second,) * 4)
            torch.testing.assert_close(snapshot.processed_actions, torch.full((4, 7), 2.))
        finally:
            policy.release_first.set()
            engine.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(engine.failed)

    def test_generation_eligibility_native_failed_retry_uses_new_sampling_start_and_identity(self):
        policy = ProvenanceSampler(fail_first=True)
        engine = self._engine(policy)
        engine._rtc_queue_threshold = 0
        queue = start_with_acknowledged_queue(engine, require_observation_provenance=True,
                                               max_observation_age_s=.3)
        thread = engine._rtc_thread
        clocks = [1.1, 10.1]
        queue._system_clock, queue._steady_clock = lambda: clocks[0], lambda: clocks[1]
        keys = [f"j{i}" for i in range(7)]
        try:
            engine.notify_observation(queue.observed_input(dict.fromkeys(keys, 1.), self._provenance(1.)))
            engine.resume()
            self.assertTrue(policy.first_started.wait(2.))
            clocks[:] = [2.1, 11.1]
            second = self._provenance(2.)
            engine.notify_observation(queue.observed_input(dict.fromkeys(keys, 2.), second))
            policy.release_first.set()
            deadline = time.monotonic() + 2.
            while queue.generation < 1 and time.monotonic() < deadline:
                time.sleep(.005)
            snapshot = queue.snapshot()
            self.assertEqual(len(snapshot.generation_eligibility), 4)
            receipt = snapshot.generation_eligibility[0]
            self.assertEqual(receipt.observation, second)
            self.assertEqual(receipt.sampling_started_at_s, 2.1)
            self.assertEqual(receipt.sampling_started_monotonic_s, 11.1)
            self.assertEqual(receipt.merge_completed_at_s, 2.1)
            torch.testing.assert_close(snapshot.processed_actions, torch.full((4, 7), 2.))
        finally:
            policy.release_first.set()
            engine.stop()
        self.assertFalse(thread.is_alive())

    def test_generation_native_merge_staleness_rejects_without_publishing_or_new_retry_loop(self):
        policy = ProvenanceSampler()
        engine = self._engine(policy)
        queue = start_with_acknowledged_queue(engine, require_observation_provenance=True,
                                               max_observation_age_s=.3)
        thread = engine._rtc_thread
        clocks, rejected = [1.1, 10.1], threading.Event()
        queue._system_clock, queue._steady_clock = lambda: clocks[0], lambda: clocks[1]
        native_merge = queue.merge

        def observe_rejection(*args, **kwargs):
            try:
                return native_merge(*args, **kwargs)
            except RuntimeError as exc:
                if "GENERATION_STALE" in str(exc):
                    engine.pause()  # Bounded harness, not new production retry policy.
                    rejected.set()
                raise

        with mock.patch.object(queue, "merge", side_effect=observe_rejection):
            try:
                engine.notify_observation(queue.observed_input(dict.fromkeys([f"j{i}" for i in range(7)], 1.), self._provenance(1.)))
                engine.resume()
                self.assertTrue(policy.first_started.wait(2.))
                clocks[:] = [1.4, 10.4]
                policy.release_first.set()
                self.assertTrue(rejected.wait(2.))
                self.assertEqual(queue.generation, 0)
                self.assertEqual(queue.snapshot().generation_eligibility, ())
                self.assertEqual(queue.qsize(), 0)
                # Native resets consecutive_errors before merge. A rejection
                # is not a promise that engine.failed will become true.
                self.assertFalse(engine.failed)
            finally:
                policy.release_first.set()
                engine.stop()
        self.assertFalse(thread.is_alive())

    def test_unwrapped_observation_after_failure_cannot_inherit_stale_identity(self):
        policy = ProvenanceSampler(fail_first=True)
        engine = self._engine(policy)
        keys = [f"j{i}" for i in range(7)]
        first = self._provenance(1.)
        queue = start_with_acknowledged_queue(
            engine, require_observation_provenance=True,
        )
        thread = engine._rtc_thread
        missing = threading.Event()
        native_merge = queue.merge
        def observed_merge(*args, **kwargs):
            try:
                return native_merge(*args, **kwargs)
            except RuntimeError as exc:
                if "OBSERVATION_PROVENANCE_MISSING" in str(exc):
                    missing.set()
                raise
        with mock.patch.object(queue, "merge", side_effect=observed_merge):
            try:
                engine.notify_observation(
                    queue.observed_input(dict.fromkeys(keys, 1.), first)
                )
                engine.resume()
                self.assertTrue(policy.first_started.wait(2.))
                # Deliberately bypass the required owner wrapper. This must fail
                # closed, not make B's tensors appear to originate from A.
                engine.notify_observation(dict.fromkeys(keys, 2.))
                policy.release_first.set()
                self.assertTrue(missing.wait(2.))
                engine.pause()
                self.assertEqual(policy.calls, 2)
                self.assertEqual(queue.generation, 0)
                self.assertEqual(queue.qsize(), 0)
                self.assertEqual(queue.snapshot().observation_provenance, ())
                self.assertFalse(engine.failed)
            finally:
                policy.release_first.set()
                engine.stop()
        self.assertFalse(thread.is_alive())

    def test_native_smolvla_capability_is_not_its_rtc_enabled_flag(self):
        policy = SyntheticSampler()
        self.assertFalse(policy._rtc_enabled())
        self.assertTrue(policy.supports_rtc())
        self.assertTrue(supports_rtc_inference(policy))

    def _engine(self, policy):
        keys = [f"j{i}" for i in range(7)]
        features = {"observation.state": {"dtype": "float32", "shape": (7,), "names": keys}}
        return create_inference_engine(
            RTCInferenceConfig(rtc=policy.config.rtc_config, queue_threshold=2),
            policy=policy, preprocessor=IdentityProcessor(), postprocessor=IdentityProcessor(),
            robot_wrapper=SimpleNamespace(robot_type="fr5"), hw_features=features,
            dataset_features=features, ordered_action_keys=keys, task="synthetic pick",
            fps=30., device="cpu")

    def test_existing_factory_keeps_consumption_available_during_next_inference(self):
        policy = SyntheticSampler()
        keys = [f"j{i}" for i in range(7)]
        engine = self._engine(policy)
        engine.start()
        thread = engine._rtc_thread  # Retain actual thread across native stop's cleanup.
        try:
            engine.notify_observation(dict.fromkeys(keys, 0.))
            engine.resume()
            deadline = time.monotonic() + 3.
            while engine.action_queue.qsize() < 4 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(engine.action_queue.qsize(), 4)
            self.assertFalse(engine.failed)
            first = engine.get_action(None)
            torch.testing.assert_close(first, torch.ones(7))
            engine.get_action(None)
            self.assertTrue(policy.second_started.wait(2.))
            # Second sampling is still blocked, but native queue consumption is not.
            self.assertFalse(policy.release_second.is_set())
            torch.testing.assert_close(engine.get_action(None), torch.ones(7))
            torch.testing.assert_close(engine.get_action(None), torch.ones(7))
            self.assertIsNone(engine.get_action(None))
            self.assertTrue(thread.is_alive())
            self.assertFalse(policy._rtc_enabled())
        finally:
            policy.release_second.set()
            engine.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(engine.failed)

    def test_native_producer_uses_acknowledged_queue_without_eager_pop(self):
        policy = SyntheticSampler()
        engine = self._engine(policy)
        queue = start_with_acknowledged_queue(engine)
        thread = engine._rtc_thread
        try:
            self.assertIs(queue, engine.action_queue)
            self.assertFalse(engine._policy_active.is_set())
            with self.assertRaisesRegex(RuntimeError, "REQUIRES_FRESH_ENGINE"):
                start_with_acknowledged_queue(engine)
            self.assertIs(queue, engine.action_queue)
            engine.notify_observation(dict.fromkeys([f"j{i}" for i in range(7)], 0.))
            engine.resume()
            deadline = time.monotonic() + 3.
            while queue.qsize() < 4 and time.monotonic() < deadline:
                time.sleep(.005)
            snapshot = queue.snapshot()
            self.assertEqual(snapshot.rows, tuple(RawActionIndex(1, i) for i in range(4)))
            self.assertEqual(queue.get_action_index(), 0)
            self.assertEqual(queue.advance_if_current(
                generation=snapshot.generation, expected_index=0, count=2),
                (RawActionIndex(1, 0), RawActionIndex(1, 1)))
            self.assertTrue(policy.second_started.wait(2.))
            self.assertEqual(queue.advance_if_current(
                generation=snapshot.generation, expected_index=2, count=2),
                (RawActionIndex(1, 2), RawActionIndex(1, 3)))
            self.assertTrue(queue.empty())
            engine.pause()
            policy.release_second.set()
            deadline = time.monotonic() + 3.
            while queue.generation < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            # Native pause is not cancellation: the in-flight call can publish.
            following = queue.snapshot()
            self.assertEqual(following.rows, tuple(RawActionIndex(2, i) for i in range(4)))
            torch.testing.assert_close(following.processed_actions, torch.full((4, 7), 2.))
            with self.assertRaisesRegex(RuntimeError, "STALE_GENERATION"):
                queue.advance_if_current(generation=snapshot.generation, expected_index=4, count=0)
        finally:
            policy.release_second.set()
            engine.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(engine.failed)

    def test_checked_prefix_consumption_survives_actual_inflight_native_append(self):
        from tools.data_factory.rollout.stream_progress import ReferenceProgress

        progress = ReferenceProgress("native-one", [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000])
        def feedback(actuator, elapsed_ns):
            names = [f"j{i}" for i in range(1, 7)] if actuator == "arm" else ["finger_right_joint"]
            sec, nanosec = divmod(elapsed_ns, 1_000_000_000)
            return {"revision": "native-one", "event": "FEEDBACK", "actuator": actuator,
                    "goal_id": [0] * 15 + [1 if actuator == "arm" else 2],
                    "received_monotonic_s": 10., "samples_since_poll": 1,
                    "feedback": {"joint_names": names, "desired": {
                        "positions": [0.] * len(names), "velocities": [], "accelerations": [], "effort": [],
                        "time_from_start": {"sec": sec, "nanosec": nanosec}}}}

        policy = SyntheticSampler()
        engine = self._engine(policy)
        queue = start_with_acknowledged_queue(engine)
        thread = engine._rtc_thread
        try:
            engine.notify_observation(dict.fromkeys([f"j{i}" for i in range(7)], 0.))
            engine.resume()
            deadline = time.monotonic() + 3.
            while queue.qsize() < 4 and time.monotonic() < deadline:
                time.sleep(.005)
            snapshot = queue.snapshot()
            self.assertEqual(len(snapshot.rows), 4)
            self.assertEqual(progress.observe({"revision": "native-one", "event": "PAIR_ACCEPTED", "predecessor": None}), 0)
            self.assertEqual(queue.qsize(), 4)  # Acceptance is not row consumption.
            progress.observe(feedback("arm", 200_000_000))
            progress.observe(feedback("gripper", 100_000_000))
            self.assertEqual(progress.selected_count, 1)
            self.assertEqual(queue.advance_unchanged_prefix(snapshot, count=progress.selected_count), snapshot.rows[:1])
            self.assertFalse(policy.second_started.is_set())
            progress.observe(feedback("gripper", 200_000_000))
            self.assertEqual(queue.advance_unchanged_prefix(snapshot, offset=1, count=progress.selected_count - 1), snapshot.rows[1:2])
            self.assertTrue(policy.second_started.wait(2.))
            engine.pause()
            policy.release_second.set()
            deadline = time.monotonic() + 3.
            while queue.generation < 2 and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(queue.generation, 2)
            self.assertEqual(queue.qsize(), 6)
            # A late cumulative update catches up both rows without per-row
            # terminal waits, even after real native append/compaction occurred.
            progress.observe(feedback("arm", 400_000_000))
            progress.observe(feedback("gripper", 400_000_000))
            self.assertEqual(queue.advance_unchanged_prefix(snapshot, offset=2, count=progress.selected_count - 2), snapshot.rows[2:])
            self.assertEqual(queue.snapshot().rows, tuple(RawActionIndex(2, i) for i in range(4)))
            torch.testing.assert_close(queue.get_processed_left_over(), torch.full((4, 7), 2.))
            self.assertEqual(policy.calls, 2)
        finally:
            policy.release_second.set()
            engine.stop()
        self.assertFalse(thread.is_alive())
        self.assertFalse(engine.failed)

    def test_already_resumed_engine_is_not_started_or_replaced(self):
        engine = self._engine(SyntheticSampler())
        engine.resume()
        with self.assertRaisesRegex(RuntimeError, "REQUIRES_FRESH_ENGINE"):
            start_with_acknowledged_queue(engine)
        self.assertIsNone(engine._rtc_thread)
        self.assertIsNone(engine.action_queue)

    def test_prediction_tail_is_not_committed_by_short_controller_submission(self):
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from tools.data_factory.motion.moveit_transport import RosMoveItTransport
        from tools.data_factory.rollout.stream_progress import ReferenceProgress

        policy = SyntheticSampler()
        engine = self._engine(policy)
        queue = start_with_acknowledged_queue(engine)
        thread = engine._rtc_thread
        try:
            engine.notify_observation(dict.fromkeys([f"j{i}" for i in range(7)], 0.))
            engine.resume()
            deadline = time.monotonic() + 3.
            while queue.qsize() < 4 and time.monotonic() < deadline:
                time.sleep(.005)
            engine.pause()
            snapshot = queue.snapshot()
            self.assertEqual(len(snapshot.rows), 4)
            # A consumer may select less than the prediction without changing
            # native inference or creating a second cursor/scheduler. These
            # synthetic values test framing only, not physical admission.
            committed_rows = snapshot.processed_actions[:2].tolist()
            transport = object.__new__(RosMoveItTransport)
            transport._FollowJointTrajectory = FollowJointTrajectory
            transport._JointTrajectoryPoint = JointTrajectoryPoint
            arm, gripper = transport.build_learned_actuator_goals(
                [0.] * 7, committed_rows, period_s=.1, start_time_ns=42_000_000_000,
            )
            times = [p.time_from_start.sec * 10**9 + p.time_from_start.nanosec
                     for p in arm.trajectory.points]
            self.assertEqual(times, [0, 100_000_000, 200_000_000])
            self.assertEqual(len(gripper.trajectory.points), 3)
            progress = ReferenceProgress("short-prefix", times)
            self.assertEqual(progress.observe({
                "revision": "short-prefix", "event": "PAIR_ACCEPTED", "predecessor": None,
            }), 0)
            self.assertEqual(queue.qsize(), 4)
            for actuator in ("arm", "gripper"):
                progress.observe({"revision": "short-prefix", "actuator": actuator,
                                  "event": "TERMINAL", "result_status": 4, "error_code": 0})
            self.assertEqual(progress.selected_count, 2)
            self.assertEqual(queue.advance_unchanged_prefix(snapshot, count=progress.selected_count),
                             snapshot.rows[:2])
            remaining = queue.snapshot()
            self.assertEqual(remaining.rows, snapshot.rows[2:])
            torch.testing.assert_close(remaining.processed_actions, snapshot.processed_actions[2:])
            torch.testing.assert_close(remaining.original_actions, snapshot.original_actions[2:])
            self.assertEqual(policy.calls, 1)
        finally:
            policy.release_second.set()
            engine.stop()
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
