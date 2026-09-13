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

    def test_already_resumed_engine_is_not_started_or_replaced(self):
        engine = self._engine(SyntheticSampler())
        engine.resume()
        with self.assertRaisesRegex(RuntimeError, "REQUIRES_FRESH_ENGINE"):
            start_with_acknowledged_queue(engine)
        self.assertIsNone(engine._rtc_thread)
        self.assertIsNone(engine.action_queue)


if __name__ == "__main__":
    unittest.main()
