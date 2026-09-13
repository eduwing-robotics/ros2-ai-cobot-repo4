"""CPU processors and canonical normalization; model/dataset admission are synthetic seams."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from safetensors.numpy import save_file

from tools.data_factory.learned_action_adapter import NativeSmolVLA, fake_rgb
from tools.fr5_data_factory import ContractError


def saved_processor_fixture(policy_dir, fault=None):
    """Reusable CPU fixture for Learning's validator and actual Rollout processing.

    Both state files contain identical admitted tensors; only configuration is
    changed. No checkpoint admission, policy weights or dataset are fabricated.
    """
    stats = {"observation.state": {"mean": [1.] * 7, "std": [2.] * 7},
             "action": {"mean": [.01] * 7, "std": [2.] * 7}}
    tensors = {f"{key}.{name}": np.asarray(value, dtype=np.float32)
               for key, values in stats.items() for name, value in values.items()}
    for pipeline, registry in (("policy_preprocessor", "normalizer_processor"),
                               ("policy_postprocessor", "unnormalizer_processor")):
        config = {"features": {"observation.state": {"type": "STATE", "shape": [7]},
                               "action": {"type": "ACTION", "shape": [7]}},
                  "norm_map": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}, "eps": 1e-8}
        if pipeline == "policy_preprocessor":
            if fault == "excluded":
                config["normalize_observation_keys"] = []
            elif fault == "missing":
                config["features"].pop("observation.state")
            elif fault == "wrong-type":
                config["features"]["observation.state"]["type"] = "VISUAL"
            elif fault == "identity":
                config["norm_map"]["STATE"] = "IDENTITY"
        if fault == ("inline-state" if pipeline == "policy_preprocessor" else "inline-action"):
            feature = "observation.state" if pipeline == "policy_preprocessor" else "action"
            config["stats"] = {feature: {"mean": [0.] * 7, "std": [1.] * 7}}
        state_file = pipeline + ".safetensors"
        steps = [{"registry_name": registry, "config": config, "state_file": state_file}]
        if pipeline == "policy_preprocessor":
            steps.insert(0, {"registry_name": "to_batch_processor", "config": {}})
        (policy_dir / (pipeline + ".json")).write_text(json.dumps({"name": pipeline, "steps": steps}))
        save_file(tensors, policy_dir / state_file)
    return {"stats": stats}


class _AsyncPipeline:
    def __init__(self, transform=lambda value: value):
        self.steps = []
        self.transform = transform
        self.reset_calls = 0

    def __call__(self, value):
        return self.transform(value)

    def reset(self):
        self.reset_calls += 1


class _AsyncPolicy:
    def __init__(self, *, entered=None, release=None):
        self.config = SimpleNamespace(
            input_features={
                "observation.state": SimpleNamespace(shape=(7,)),
                "observation.images.camera1": SimpleNamespace(shape=(3, 1, 1)),
                "observation.images.camera2": SimpleNamespace(shape=(3, 1, 1)),
            },
            rtc_config=None,
        )
        self.entered = entered
        self.release = release
        self.reset_calls = 0
        self.calls = []

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, batch, **kwargs):
        self.calls.append(({key: value.clone() if isinstance(value, torch.Tensor) else value
                            for key, value in batch.items()}, kwargs))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait()
        return torch.full((1, 2, 7), float(len(self.calls)))


class SavedProcessorBypassTest(unittest.TestCase):
    def test_config_bypasses_identical_saved_tensors_in_installed_cpu_processors(self):
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.numpy import load_file
        with tempfile.TemporaryDirectory() as directory:
            policy_dir = Path(directory)
            baseline = saved_processor_fixture(policy_dir)
            pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=directory)
            torch.testing.assert_close(pre({"observation.state": torch.zeros(7)})["observation.state"],
                                       torch.full((1, 7), -.5))
            torch.testing.assert_close(post(torch.zeros((1, 1, 7))), torch.full((1, 1, 7), .01))
            for fault in ("excluded", "missing", "wrong-type", "identity", "inline-state", "inline-action"):
                with self.subTest(fault=fault):
                    self.assertEqual(saved_processor_fixture(policy_dir, fault), baseline)
                    for pipeline in ("policy_preprocessor", "policy_postprocessor"):
                        saved = load_file(policy_dir / (pipeline + ".safetensors"))
                        for key, values in baseline["stats"].items():
                            for stat, value in values.items():
                                np.testing.assert_array_equal(saved[f"{key}.{stat}"], np.asarray(value, dtype=np.float32))
                    pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=directory)
                    actual = (post(torch.zeros((1, 1, 7))) if fault == "inline-action" else
                              pre({"observation.state": torch.zeros(7)})["observation.state"])
                    torch.testing.assert_close(actual, torch.zeros_like(actual))


class NativePolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy_dir = self.root / 'pretrained_model'
        self.policy_dir.mkdir()
        (self.root / 'fr5_training_receipt.json').write_text('{}')
        (self.root / 'fr5_training_split.json').write_text('{}')
        save_file({'synthetic.weight': np.ones(1, dtype=np.float32)}, self.policy_dir / 'model.safetensors')
        self.mean_action = np.array([.01] * 6 + [.012], dtype=np.float32)
        self.normalization = {"stats": {
            "observation.state": {"mean": [1.] * 7, "std": [2.] * 7},
            "action": {"mean": self.mean_action.tolist(), "std": [2.] * 7},
        }}
        for name, registry in (
            ('policy_preprocessor', 'normalizer_processor'),
            ('policy_postprocessor', 'unnormalizer_processor'),
        ):
            config = {'features': {'observation.state': {'type': 'STATE', 'shape': [7]},
                                   'action': {'type': 'ACTION', 'shape': [7]}},
                      'norm_map': {'STATE': 'MEAN_STD', 'ACTION': 'MEAN_STD'}, 'eps': 1e-8}
            steps = [{'registry_name': 'device_processor', 'config': {'device': 'cpu'}}]
            if name == 'policy_preprocessor':
                steps.insert(0, {'registry_name': 'to_batch_processor', 'config': {}})
            steps.append({'registry_name': registry, 'config': config, 'state_file': name + '.safetensors'})
            (self.policy_dir / (name + '.json')).write_text(json.dumps({'name': name, 'steps': steps}))
            save_file({f"{key}.{stat}": np.asarray(value, dtype=np.float32)
                       for key, stats in self.normalization["stats"].items() for stat, value in stats.items()},
                      self.policy_dir / (name + '.safetensors'))
        def admitted_checkpoint(path, *, verify_dataset):
            from tools.validate_training_checkpoint import validate_normalization_state
            self.assertEqual(path, self.policy_dir)
            self.assertIs(verify_dataset, True)
            # Only dataset/receipt admission is synthetic; the canonical saved
            # artifact checks execute unchanged, as they do in validate_checkpoint.
            validate_normalization_state(path, self.normalization)
            return self.policy_dir, self.root
        self.admission = mock.patch('tools.validate_training_checkpoint.validate_checkpoint', side_effect=admitted_checkpoint)
        self.admitted_checkpoint = self.admission.start()
        self.addCleanup(self.admission.stop)
        self.saved_view = self.enterContext(mock.patch(
            'tools.validate_training_checkpoint.validate_saved_observation_view',
            return_value={"representation": "raw", "transform_application": "none",
                          "training_transform": "raw_once"}))
        self.offline = mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        self.offline.start()
        self.addCleanup(self.offline.stop)

    def async_native(self, policy, pre=None, post=None):
        from tools.data_factory.training_receipts import tree_digest

        native = NativeSmolVLA()
        native.policy = policy
        native.preprocessor = pre or _AsyncPipeline()
        native.postprocessor = post or _AsyncPipeline()
        native.policy_dir = self.policy_dir
        native.checkpoint = {"tree_digest": tree_digest(self.policy_dir)}
        native.observation_view = {
            "representation": "raw", "transform_application": "none",
        }
        native.device = "cpu"
        return native

    def test_task_scope_uses_native_async_queue_and_resets_once_across_chunks(self):
        from lerobot.rollout.inference.rtc import RTCInferenceEngine
        from lerobot_strategy_fr5.acknowledged_queue import AcknowledgedActionQueue
        from tools.data_factory.training_receipts import tree_digest

        policy = _AsyncPolicy()
        pre, post = _AsyncPipeline(), _AsyncPipeline()
        native = self.async_native(policy, pre, post)
        native.observation_view = {
            "representation": "baked", "transform_application": "rollout_once",
        }
        native._transform_raw_up = mock.Mock(return_value=fake_rgb(b"\xff\x00\x00"))
        value = {
            "observation.state": [float(index) for index in range(7)],
            "observation.images.camera1": fake_rgb(),
            "observation.images.camera2": fake_rgb(b"\x00\x00\xff"),
        }

        with mock.patch(
            "tools.data_factory.training_receipts.tree_digest", wraps=tree_digest,
        ) as digest:
            with native.prepare_task_inference(
                instruction="move the block", fps=30, queue_threshold=0,
            ) as (engine, queue, notify):
                self.assertIsInstance(engine, RTCInferenceEngine)
                self.assertIsInstance(queue, AcknowledgedActionQueue)
                self.assertIs(engine.action_queue, queue)
                self.assertFalse(engine._rtc_config.enabled)
                self.assertIsNone(policy.config.rtc_config)
                notify(value)
                value["observation.state"][0] = 999.
                value["observation.images.camera1"] = fake_rgb(b"\x00\xff\x00")
                engine.resume()
                deadline = time.monotonic() + 2.
                while queue.qsize() < 2 and time.monotonic() < deadline:
                    time.sleep(.005)
                first = queue.snapshot()
                self.assertEqual(len(first.rows), 2)
                queue.advance_if_current(
                    generation=first.generation,
                    expected_index=first.queue_index,
                    count=2,
                )
                while len(policy.calls) < 2 and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertGreaterEqual(len(policy.calls), 2)
                with self.assertRaisesRegex(ContractError, "LEARNED_REENTRANT_INFERENCE"):
                    with native.prepare_inference():
                        self.fail("concurrent one-shot scope admitted")

            self.assertEqual(digest.call_count, 1)
        with self.assertRaisesRegex(ContractError, "LEARNED_INFERENCE_SCOPE"):
            notify(value)
        self.assertEqual(policy.reset_calls, 1)
        self.assertEqual(pre.reset_calls, 1)
        self.assertEqual(post.reset_calls, 1)
        self.assertEqual(policy.calls[0][0]["observation.state"][0, 0].item(), 0.)
        torch.testing.assert_close(
            policy.calls[0][0]["observation.images.camera1"],
            torch.tensor([[[[1.]], [[0.]], [[0.]]]]),
        )
        native._transform_raw_up.assert_called_once()
        self.assertTrue(native._inference_lock.acquire(blocking=False))
        native._inference_lock.release()

    def test_task_scope_retains_loaded_model_owner_after_native_stop_timeout(self):
        from lerobot.rollout.inference import rtc as rtc_module

        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        policy = _AsyncPolicy(entered=entered, release=release)
        native = self.async_native(policy)
        value = {
            "observation.state": [0.] * 7,
            "observation.images.camera1": fake_rgb(),
            "observation.images.camera2": fake_rgb(),
        }
        producer = None
        with mock.patch.object(rtc_module, "_RTC_JOIN_TIMEOUT_S", .01):
            with self.assertRaisesRegex(
                ContractError, "LEARNED_INFERENCE_STOP_UNCONFIRMED",
            ):
                with native.prepare_task_inference(
                    instruction="move the block", fps=30, queue_threshold=0,
                ) as (engine, _queue, notify):
                    producer = engine._rtc_thread
                    notify(value)
                    engine.resume()
                    self.assertTrue(entered.wait(1.))
                    # The caller may request early native stop. Its timeout
                    # clears the engine's handle, but not model ownership.
                    engine.stop()
                    self.assertIsNone(engine._rtc_thread)

        self.assertIsNotNone(producer)
        self.assertTrue(producer.is_alive())
        with self.assertRaisesRegex(ContractError, "LEARNED_INFERENCE_SCOPE"):
            notify(value)
        with self.assertRaisesRegex(ContractError, "LEARNED_REENTRANT_INFERENCE"):
            with native.prepare_inference():
                self.fail("model owner released before producer exited")
        release.set()
        producer.join(1.)
        self.assertFalse(producer.is_alive())
        with native.prepare_inference():
            pass

    def test_task_scope_preserves_body_failure_when_stop_is_unconfirmed(self):
        from lerobot.rollout.inference import rtc as rtc_module

        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        native = self.async_native(_AsyncPolicy(entered=entered, release=release))
        value = {
            "observation.state": [0.] * 7,
            "observation.images.camera1": fake_rgb(),
            "observation.images.camera2": fake_rgb(),
        }
        producer = None
        with mock.patch.object(rtc_module, "_RTC_JOIN_TIMEOUT_S", .01):
            with self.assertRaisesRegex(RuntimeError, "primary task failure") as raised:
                with native.prepare_task_inference(
                    instruction="move the block", fps=30, queue_threshold=0,
                ) as (engine, _queue, notify):
                    producer = engine._rtc_thread
                    notify(value)
                    engine.resume()
                    self.assertTrue(entered.wait(1.))
                    raise RuntimeError("primary task failure")
        self.assertIn(
            "LEARNED_INFERENCE_STOP_UNCONFIRMED",
            getattr(raised.exception, "__notes__", []),
        )
        self.assertTrue(producer.is_alive())
        release.set()
        producer.join(1.)

    def test_component_loader_accepts_canonically_admitted_native_image_slots(self):
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from tools.validate_training_checkpoint import validate_policy_feature_contract

        inputs = {
            "observation.state": {"type": "STATE", "shape": [7]},
            "observation.images.camera1": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.camera2": {"type": "VISUAL", "shape": [3, 480, 640]},
        }
        feature = {"profile": "smolvla", "policy_argv": [
            "--policy.empty_cameras=1",
            "--policy.input_features=" + json.dumps(inputs),
            "--policy.output_features=" + json.dumps(
                {"action": {"type": "ACTION", "shape": [7]}}
            ),
        ]}
        config = SmolVLAConfig(
            device="cpu", push_to_hub=False, empty_cameras=1,
            input_features={
                key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
                for key, value in inputs.items()
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )
        config.input_features["observation.images.camera3"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 256, 256)
        )
        config.validate_features()
        config.save_pretrained(self.policy_dir)
        validate_policy_feature_contract(json.loads((self.policy_dir / "config.json").read_text()), feature)

        policy = mock.Mock()
        for saved_vlm_load in (True, False):
            with self.subTest(load_vlm_weights=saved_vlm_load):
                config.load_vlm_weights = saved_vlm_load
                config.save_pretrained(self.policy_dir)
                saved_bytes = (self.policy_dir / "config.json").read_bytes()
                with mock.patch.object(SmolVLAPolicy, "from_pretrained", return_value=policy) as weight_load:
                    loaded, preprocessor, postprocessor = NativeSmolVLA._load_components(
                        self.policy_dir, "cpu"
                    )
                self.assertIs(weight_load.call_args.kwargs["config"].load_vlm_weights, saved_vlm_load)
                self.assertEqual((self.policy_dir / "config.json").read_bytes(), saved_bytes)
        self.assertIs(loaded, policy)
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(postprocessor)
        self.assertTrue(weight_load.call_args.kwargs["strict"])
        self.assertTrue(weight_load.call_args.kwargs["local_files_only"])

        one_view = SmolVLAConfig(
            device="cpu", push_to_hub=False, empty_cameras=2,
            input_features={
                key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
                for key, value in inputs.items() if key != "observation.images.camera2"
            },
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )
        one_view.validate_features()
        one_view.save_pretrained(self.policy_dir)
        with mock.patch.object(SmolVLAPolicy, "from_pretrained") as weight_load:
            with self.assertRaisesRegex(ContractError, "LEARNED_MODEL_CAMERAS"):
                NativeSmolVLA._load_components(self.policy_dir, "cpu")
            weight_load.assert_not_called()

    def test_discarded_warmup_restores_rng_and_releases_owner_on_success_failure_cancel(self):
        import random
        from lerobot.policies.factory import make_pre_post_processors
        python_state, numpy_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, python_state)
        self.addCleanup(np.random.set_state, numpy_state)
        pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        policy = mock.Mock()
        with mock.patch.object(NativeSmolVLA, "_load_components", return_value=(policy, pre, post)):
            native = NativeSmolVLA.load(self.policy_dir)
        observation = {"observation.state": [0.] * 7, "task": "probe",
                       "observation.images.camera1": fake_rgb(), "observation.images.camera2": fake_rgb()}
        for outcome in ("success", "failure", "cancel"):
            with self.subTest(outcome=outcome), torch.random.fork_rng(devices=[]):
                cancel = threading.Event()
                def infer(batch):
                    self.assertEqual(tuple(batch["observation.images.camera1"].shape), (1, 3, 1, 1))
                    random.random()
                    np.random.rand()
                    output = torch.randn(1, 2, 7)
                    if outcome == "failure":
                        raise RuntimeError("synthetic warmup failure")
                    if outcome == "cancel":
                        cancel.set()
                    return output
                policy.predict_action_chunk.side_effect = infer
                torch_state = torch.get_rng_state().clone()
                py_state, np_state = random.getstate(), np.random.get_state()
                if outcome == "success":
                    report = native.warmup(instruction="probe", height=1, width=1, cancel_event=cancel)
                    self.assertEqual(report["output_disposition"], "DISCARDED")
                    self.assertTrue(report["rng_state_restored"])
                    self.assertGreaterEqual(report["duration_s"], report["inference_duration_s"])
                    self.assertNotIn("actions", report)
                else:
                    with self.assertRaisesRegex(ContractError, "LEARNED_CANCELLED" if outcome == "cancel" else "LEARNED_WARMUP_FAILED"):
                        native.warmup(instruction="probe", height=1, width=1, cancel_event=cancel)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                self.assertEqual(random.getstate(), py_state)
                self.assertEqual(repr(np.random.get_state()), repr(np_state))
                self.assertTrue(native._inference_lock.acquire(blocking=False))
                native._inference_lock.release()
                if outcome == "success":
                    after_warmup = native(observation)
                    torch.set_rng_state(torch_state)
                    np.testing.assert_array_equal(native(observation), after_warmup)

    def test_prepared_capture_owns_loaded_cpu_tensors_and_rejects_next_changed_scope(self):
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_model
        from tools.data_factory.training_receipts import tree_digest
        class TinyPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.synthetic = torch.nn.Linear(1, 1, bias=False)
                self.synthetic.weight = torch.nn.Parameter(torch.zeros(1))
            def reset(self):
                pass
            def predict_action_chunk(self, batch):
                # Actual copied model storage and both saved CPU processors.
                return (self.synthetic.weight - 1 + batch["observation.state"].mean() + .5).expand(1, 1, 7)
        policy = TinyPolicy()
        load_model(policy, self.policy_dir / "model.safetensors")
        pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        with mock.patch.object(NativeSmolVLA, "_load_components", return_value=(policy, pre, post)):
            native = NativeSmolVLA.load(self.policy_dir)
        value = {"observation.state": [0.] * 7, "task": "synthetic probe",
                 "observation.images.camera1": fake_rgb(), "observation.images.camera2": fake_rgb()}
        original = native(value)
        with mock.patch("tools.data_factory.training_receipts.tree_digest", wraps=tree_digest) as digest:
            with native.prepare_inference() as predict:
                self.assertEqual(digest.call_count, 1)  # Before acquiring any input.
                with self.assertRaisesRegex(ContractError, "LEARNED_REENTRANT_INFERENCE"):
                    native(value)
                # In-place same-inode writes challenge mmap-backed CPU state,
                # unlike rename/replace, which leaves the old mapping intact.
                for path in self.policy_dir.glob("*.safetensors"):
                    with path.open("r+b") as stream:
                        offset = 8 + int.from_bytes(stream.read(8), "little")
                        size = path.stat().st_size - offset
                        stream.seek(offset)
                        stream.write(np.full(size // 4, 9., dtype=np.float32).tobytes())
                np.testing.assert_allclose(predict(value), original)
                self.assertEqual(digest.call_count, 1)  # No hash in capture-to-action scope.
                with self.assertRaisesRegex(ContractError, "LEARNED_INFERENCE_SCOPE"):
                    predict(value)
            with self.assertRaisesRegex(ContractError, "LEARNED_INFERENCE_SCOPE"):
                predict(value)
            with self.assertRaisesRegex(ContractError, "LEARNED_CHECKPOINT_CHANGED"):
                with native.prepare_inference():
                    self.fail("changed artifacts admitted another capture")
        self.assertTrue(native._inference_lock.acquire(blocking=False))
        native._inference_lock.release()

    def test_installed_sample_actions_and_saved_processors_reach_offline_comparison(self):
        from lerobot.policies.smolvla import modeling_smolvla as module
        from lerobot.policies.factory import make_pre_post_processors
        from tools.data_factory.learned_action_adapter import NativeSmolVLA, fake_rgb
        import copy
        from types import MethodType
        from tools.data_factory.rollout.solver_efficiency import compare_native, offline_solver
        config = SimpleNamespace(device="cpu", chunk_size=2, max_action_dim=32,
                                 num_steps=10, use_cache=True, rtc_config=None, adapt_to_pi_aloha=False)
        model = SimpleNamespace(config=config, rtc_processor=None, _rtc_enabled=lambda: False)
        model.embed_prefix = mock.Mock(return_value=(torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool),
                                                     torch.zeros(1, 1, dtype=torch.bool)))
        model.vlm_with_expert = SimpleNamespace(forward=mock.Mock(return_value=(None, None)))
        model.denoise_step = mock.Mock(side_effect=lambda **kw: torch.ones_like(kw["x_t"]))
        model.sample_actions = MethodType(module.VLAFlowMatching.sample_actions, model)
        original = model.sample_actions
        original_global = module.euler_integrate
        policy = SimpleNamespace(config=config, model=model, reset=mock.Mock())
        def predict(batch, noise):
            self.assertTrue(torch.allclose(batch["observation.state"], torch.full((1, 7), -.5)))
            return model.sample_actions(None, None, None, None, batch["observation.state"], noise=noise)[..., :7]
        policy.predict_action_chunk = predict
        pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        with mock.patch.object(NativeSmolVLA, "_load_components", return_value=(policy, pre, post)):
            native = NativeSmolVLA.load(self.policy_dir)
        observation = {"observation.state": [0.] * 7, "task": "synthetic offline comparison",
                       "observation.images.camera1": fake_rgb(), "observation.images.camera2": fake_rgb()}
        original_observation = copy.deepcopy(observation)
        report = compare_native(native, observation, seeds=(0,), repeats=1, warmups=0)
        self.assertEqual(report["action_units"], ["rad"] * 6 + ["m"])
        self.assertEqual([row["nfe"] for row in report["rows"]], [10, 5, 2, 6])
        self.assertEqual(model.embed_prefix.call_count, 4)
        self.assertEqual(model.denoise_step.call_count, 23)
        self.assertEqual(observation, original_observation)
        self.assertIs(model.sample_actions, original)
        self.assertIs(module.euler_integrate, original_global)
        self.assertEqual(config.num_steps, 10)
        with self.assertRaisesRegex(RuntimeError, "test unwind"):
            with offline_solver(model, lambda *args: None):
                raise RuntimeError("test unwind")
        self.assertIs(model.sample_actions, original)
        self.assertTrue(native._inference_lock.acquire(blocking=False))
        native._inference_lock.release()
        model.denoise_step.side_effect = RuntimeError("synthetic expert failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic expert failure"):
            compare_native(native, observation, seeds=(0,), repeats=1, warmups=0)
        self.assertIs(model.sample_actions, original)
        self.assertIs(module.euler_integrate, original_global)
        self.assertTrue(native._inference_lock.acquire(blocking=False))
        native._inference_lock.release()

    def test_saved_pre_post_normalization_and_chunk_api_are_consumed(self):
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = mock.Mock()
        def predict(batch):
            self.assertTrue(torch.allclose(batch['observation.state'], torch.full((1, 7), -.5)))
            self.assertEqual(tuple(batch['observation.images.camera1'].shape), (1, 3, 1, 1))
            return torch.zeros((1, 2, 7))
        policy.predict_action_chunk.side_effect = predict
        config = SimpleNamespace(input_features={'observation.state': SimpleNamespace(shape=[7]),
            'observation.images.camera1': object(), 'observation.images.camera2': object()},
            output_features={'action': SimpleNamespace(shape=[7])}, empty_cameras=1,
            adapt_to_pi_aloha=False, rtc_config=None)
        with mock.patch.object(SmolVLAConfig, 'from_pretrained', return_value=config), \
             mock.patch.object(SmolVLAPolicy, 'from_pretrained', return_value=policy) as load:
            native = NativeSmolVLA.load(self.policy_dir)
        self.admitted_checkpoint.assert_called_once_with(self.policy_dir, verify_dataset=True)
        self.assertTrue(load.call_args.kwargs['strict'])
        self.assertTrue(load.call_args.kwargs['local_files_only'])
        value = {'observation.state': [0.] * 7, 'observation.images.camera1': fake_rgb(),
                 'observation.images.camera2': fake_rgb(), 'task': 'synthetic probe'}
        actions = native(value)
        np.testing.assert_allclose(actions, [self.mean_action, self.mean_action])
        policy.forward.assert_not_called()
        policy.reset.assert_called_once()
        # The same saved-processor/model seam is consumed by the real OneJob and
        # canonical validator; only the model and transport are synthetic.
        from tools.data_factory.one_job import OneJob
        from tools.data_factory.motion.pickup_executor import PickupExecutor
        from tools.data_factory.rollout.finite_plan import FinitePolicyInference
        from tests.data_factory.rollout.test_finite_plan import source, observation, XML, Transport, SCENE
        transport = Transport()
        transport.current = [0.] * 7
        executor = PickupExecutor(transport, source_clock=lambda: 10.)
        job = OneJob(lambda _: self.fail("plan-only recorder effect"), executor.process)
        original = source()
        original["steps"][-1]["limits"]["execution_timeout_s"] = 6.
        obs = observation()
        obs["observation.state"] = [0.] * 7
        result = job.plan_learned("native-probe", original, SCENE,
            FinitePolicyInference(native, native.checkpoint, source_clock=lambda: 10.), obs,
            instruction="synthetic probe", robot_description=XML, period_s=1.5)
        self.assertTrue(result["ok"], result)
        frozen = result["plan_envelope"]["plan"]["learned_proposal"]
        self.assertEqual(frozen["checkpoint"], native.checkpoint)
        np.testing.assert_allclose(frozen["actions"], [self.mean_action, self.mean_action])
        self.assertEqual(transport.sent, [])
        from tools.data_factory.run_job import run_learned_plan_only
        import threading
        import time
        (self.root / "robot.urdf").write_text(XML)
        transport2 = Transport()
        transport2.current = [0.] * 7
        executor2 = PickupExecutor(transport2)
        child = SimpleNamespace(request=lambda request, _cancel: executor2.process(request), close=lambda **_: None)
        def capture():
            value = observation()
            value["observation.state"] = [0.] * 7
            value["source_timestamps_s"] = {key: time.time() for key in ("state", "camera1", "camera2")}
            return value
        payload = {"run_id": "native-runner", "urdf": str(self.root / "robot.urdf")}
        with mock.patch.object(NativeSmolVLA, "load", return_value=native):
            result = run_learned_plan_only(payload, threading.Event(), lambda _: None,
                checkpoint=self.policy_dir, observation=capture, instruction="synthetic probe", period_s=1.5,
                resolver=lambda _: ({"normalized_job": {}, "resolved_job_digest": original["resolved_job_digest"]}, original, SCENE),
                executor_factory=lambda _: child)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["data"]["finite_learned_plan"]["plan"]["execution_kind"], "FINITE_LEARNED_PROBE")
        self.assertEqual(transport2.sent, [])
        # The same normal entrypoint can acquire its observation from the sole
        # child after model load, then reuse that child for exact-plan creation.
        captured_transport = Transport()
        captured_transport.current = [0.] * 7
        loaded, requests, closed = [], [], []
        def capture_wire(topics, age):
            self.assertEqual(loaded, [True])
            self.assertEqual(topics, {"camera1": "/up", "camera2": "/wrist"})
            self.assertEqual(age, .3)
            value = capture()
            for key in ("observation.images.camera1", "observation.images.camera2"):
                value[key]["data_hex"] = value[key].pop("data").hex()
            return value
        captured_transport.capture_policy_observation = capture_wire
        captured_executor = PickupExecutor(captured_transport)
        def request_capture(request, _cancel):
            requests.append(request["op"])
            return captured_executor.process(request)
        captured_child = SimpleNamespace(request=request_capture, close=lambda **_: closed.append(True))
        factory = mock.Mock(return_value=captured_child)
        def load(*_, **__):
            loaded.append(True)
            return native
        with mock.patch.object(NativeSmolVLA, "load", side_effect=load):
            captured_result = run_learned_plan_only({**payload, "run_id": "native-capture"}, threading.Event(), lambda _: None,
                checkpoint=self.policy_dir, camera_topics={"camera1": "/up", "camera2": "/wrist"},
                gripper_source_clock="/synthetic/measured-clock.json",
                instruction="synthetic probe", period_s=1.5,
                resolver=lambda _: ({"normalized_job": {}, "resolved_job_digest": original["resolved_job_digest"]}, original, SCENE),
                executor_factory=factory)
        self.assertTrue(captured_result["ok"], captured_result)
        frozen_capture = captured_result["data"]["finite_learned_plan"]["plan"]["learned_proposal"]
        np.testing.assert_allclose(frozen_capture["actions"], [self.mean_action, self.mean_action])
        self.assertEqual(requests, ["capture_observation", "plan"])
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs, {"gripper_source_clock": "/synthetic/measured-clock.json"})
        self.assertEqual(closed, [True])
        self.assertEqual(captured_transport.sent, [])
        # The top native API must expose the same explicit held-target opt-in;
        # exercise its real inference/program/planner consumers, not a kwargs spy.
        held_source = source()
        held_source["steps"][-1]["limits"]["execution_timeout_s"] = 6.
        held_source["gripper_requirements"].update(command_position_m=.012, acceptable_feedback_m={"min": .012, "max": .014})
        close = next(step for step in held_source["steps"] if step["phase"] == "GRIPPER_CLOSE")
        close["gripper_position_m"] = .012
        close["limits"]["completion_tolerance_m"] = .014 - .012
        held_transport = Transport()
        held_transport.current = [0.] * 7
        held_transport.build_learned_segment = mock.Mock(return_value=b"synthetic-held-segment")
        held_executor = PickupExecutor(held_transport)
        held_child = SimpleNamespace(request=lambda request, _cancel: held_executor.process(request), close=lambda **_: None)
        with mock.patch.object(NativeSmolVLA, "load", return_value=native):
            held_result = run_learned_plan_only({**payload, "run_id": "native-held-runner"}, threading.Event(), lambda _: None,
                checkpoint=self.policy_dir, observation=capture, instruction="synthetic probe", period_s=1.5,
                held_gripper_targets=True,
                resolver=lambda _: ({"normalized_job": {}, "resolved_job_digest": held_source["resolved_job_digest"]}, held_source, SCENE),
                executor_factory=lambda _: held_child)
        self.assertTrue(held_result["ok"], held_result)
        held_plan = held_result["data"]["finite_learned_plan"]["plan"]
        self.assertEqual(held_plan["learned_proposal"]["schema_version"], "data_factory.finite_learned_held_target_proposal.v1")
        self.assertEqual([step["type"] for step in held_plan["steps"][0]["held_target_segments"]], ["GRIPPER", "ARM"])
        self.assertEqual(held_transport.sent, [])
        (self.policy_dir / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ContractError, 'LEARNED_CHECKPOINT_CHANGED'):
            native(value)

    def test_baked_observation_view_transforms_camera1_once(self):
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        seen = {}
        policy = mock.Mock()
        def predict(batch):
            seen["camera1"] = batch["observation.images.camera1"].clone()
            seen["camera2"] = batch["observation.images.camera2"].clone()
            return torch.zeros((1, 1, 7))
        policy.predict_action_chunk.side_effect = predict
        config = SimpleNamespace(input_features={
            'observation.state': SimpleNamespace(shape=[7]),
            'observation.images.camera1': object(), 'observation.images.camera2': object()},
            output_features={'action': SimpleNamespace(shape=[7])}, empty_cameras=1,
            adapt_to_pi_aloha=False, rtc_config=None)
        self.saved_view.return_value = {"representation": "baked", "transform_application": "rollout_once"}
        with mock.patch.object(SmolVLAConfig, 'from_pretrained', return_value=config), \
             mock.patch.object(SmolVLAPolicy, 'from_pretrained', return_value=policy):
            native = NativeSmolVLA.load(self.policy_dir)
        native._transform_raw_up = mock.Mock(return_value=fake_rgb(b"\xff\x00\x00"))
        value = {'observation.state': [0.] * 7, 'observation.images.camera1': fake_rgb(),
                 'observation.images.camera2': fake_rgb(b"\x00\x00\xff"), 'task': 'synthetic probe'}
        native(value)
        native._transform_raw_up.assert_called_once_with(value["observation.images.camera1"])
        torch.testing.assert_close(seen["camera1"], torch.tensor([[[[1.]], [[0.]], [[0.]]]]))
        torch.testing.assert_close(seen["camera2"], torch.tensor([[[[0.]], [[0.]], [[1.]]]]))

    def test_launch_manifests_must_survive_model_construction_unchanged(self):
        from lerobot.policies.factory import make_pre_post_processors

        for filename in ('fr5_training_split.json', 'fr5_training_receipt.json'):
            path = self.root / filename
            for change in ('remove', 'replace'):
                with self.subTest(filename=filename, change=change):
                    path.write_text('{}')
                    def load(*args):
                        if change == 'remove':
                            path.unlink()
                        else:
                            path.write_text('{"changed":true}')
                        return object(), *make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
                    with mock.patch.object(NativeSmolVLA, '_load_components', side_effect=load):
                        with self.assertRaisesRegex(ContractError, 'LEARNED_CHECKPOINT_CHANGED'):
                            NativeSmolVLA.load(self.policy_dir)
                    path.write_text('{}')

    def test_pending_launch_manifests_remain_supported(self):
        from lerobot.policies.factory import make_pre_post_processors

        for filename in ('fr5_training_split.json', 'fr5_training_receipt.json'):
            pending = Path(str(self.root) + '.' + filename + '.pending')
            self.addCleanup(pending.unlink, missing_ok=True)
            (self.root / filename).rename(pending)
        processors = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        with mock.patch.object(NativeSmolVLA, '_load_components', return_value=(object(), *processors)):
            native = NativeSmolVLA.load(self.policy_dir)
        self.assertEqual(native.observation_view['representation'], 'raw')

    def test_launch_manifest_change_during_saved_view_validation_is_rejected(self):
        from lerobot.policies.factory import make_pre_post_processors

        def validate(*args):
            (self.root / 'fr5_training_split.json').write_text('{"changed":true}')
            return {"representation": "raw"}
        self.saved_view.side_effect = validate
        processors = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        with mock.patch.object(NativeSmolVLA, '_load_components', return_value=(object(), *processors)):
            with self.assertRaisesRegex(ContractError, 'LEARNED_CHECKPOINT_CHANGED'):
                NativeSmolVLA.load(self.policy_dir)

    def test_empty_weights_and_missing_normalization_never_reach_model_load(self):
        with mock.patch.object(NativeSmolVLA, '_load_components') as components:
            (self.policy_dir / 'model.safetensors').write_bytes(b'')
            with self.assertRaisesRegex(ContractError, 'LEARNED_CHECKPOINT_LOAD_FAILED'):
                NativeSmolVLA.load(self.policy_dir)
            components.assert_not_called()
        save_file({'synthetic.weight': np.ones(1, dtype=np.float32)}, self.policy_dir / 'model.safetensors')
        (self.policy_dir / 'policy_postprocessor.safetensors').unlink()
        with mock.patch.object(NativeSmolVLA, '_load_components') as components:
            with self.assertRaisesRegex(ContractError, 'LEARNED_CHECKPOINT_LOAD_FAILED'):
                NativeSmolVLA.load(self.policy_dir)
            components.assert_not_called()

    def test_separate_inference_consumers_cannot_reset_one_active_native_model(self):
        from lerobot.policies.factory import make_pre_post_processors
        from tools.data_factory.rollout.finite_plan import FinitePolicyInference
        from tests.data_factory.rollout.test_finite_plan import observation, XML
        pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        entered, release = threading.Event(), threading.Event()
        policy = mock.Mock()
        def predict(_):
            if policy.predict_action_chunk.call_count == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test consumer did not release inference")
            return torch.zeros((1, 1, 7))
        policy.predict_action_chunk.side_effect = predict
        with mock.patch.object(NativeSmolVLA, "_load_components", return_value=(policy, pre, post)):
            native = NativeSmolVLA.load(self.policy_dir)
        def propose():
            return FinitePolicyInference(native, native.checkpoint, source_clock=lambda: 10.,
                monotonic_clock=lambda: 10.).propose(observation(), instruction="synthetic probe",
                                                  robot_description=XML, period_s=1.5)
        completed, errors = [], []
        def first_consumer():
            try:
                completed.append(propose())
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=first_consumer)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(ContractError, "LEARNED_POLICY_FAILED") as rejected:
                propose()
            self.assertEqual(str(rejected.exception.__cause__), "LEARNED_REENTRANT_INFERENCE")
            self.assertEqual(policy.reset.call_count, 1)
            self.assertEqual(policy.predict_action_chunk.call_count, 1)
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(completed), 1)
        np.testing.assert_allclose(propose()["actions"], completed[0]["actions"])
        policy.predict_action_chunk.side_effect = RuntimeError("synthetic inference failure")
        with self.assertRaisesRegex(ContractError, "LEARNED_POLICY_FAILED"):
            propose()
        policy.predict_action_chunk.side_effect = predict
        np.testing.assert_allclose(propose()["actions"], completed[0]["actions"])

    def test_saved_feature_contract_cannot_disable_state_normalization(self):
        cases = (
            ("policy_preprocessor", "observation.state", "missing"),
            ("policy_preprocessor", "observation.state", "wrong-type"),
            ("policy_preprocessor", "observation.state", "wrong-shape"),
            ("policy_preprocessor", "observation.state", "excluded"),
            ("policy_preprocessor", "observation.state", "string-filter"),
            ("policy_postprocessor", "action", "missing"),
            ("policy_postprocessor", "action", "wrong-type"),
            ("policy_postprocessor", "action", "wrong-shape"),
        )
        for filename, feature, fault in cases:
            with self.subTest(filename=filename, fault=fault):
                path = self.policy_dir / (filename + ".json")
                original = path.read_bytes()
                document = json.loads(original)
                config = document["steps"][-1]["config"]
                if fault == "missing":
                    config["features"].pop(feature)
                elif fault == "wrong-type":
                    config["features"][feature]["type"] = "VISUAL"
                elif fault == "wrong-shape":
                    config["features"][feature]["shape"] = [6]
                elif fault == "excluded":
                    config["normalize_observation_keys"] = []
                else:
                    config["normalize_observation_keys"] = "observation.state"
                path.write_text(json.dumps(document))
                try:
                    with mock.patch.object(NativeSmolVLA, "_load_components", return_value=(object(), object(), object())) as components:
                        with self.assertRaisesRegex(ContractError, "LEARNED_CHECKPOINT_LOAD_FAILED"):
                            NativeSmolVLA.load(self.policy_dir)
                        components.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_explicit_state_filter_consumes_saved_normalization(self):
        from lerobot.policies.factory import make_pre_post_processors
        path = self.policy_dir / "policy_preprocessor.json"
        document = json.loads(path.read_text())
        config = document["steps"][-1]["config"]
        batch = {"observation.state": torch.zeros(7)}
        # Reproduce LeRobot's silent bypass independently of our loader guard.
        config["normalize_observation_keys"] = []
        path.write_text(json.dumps(document))
        pre, _ = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        torch.testing.assert_close(pre(batch)["observation.state"], torch.zeros((1, 7)))
        config["normalize_observation_keys"] = ["observation.state"]
        path.write_text(json.dumps(document))
        pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
        with mock.patch.object(NativeSmolVLA, "_load_components", return_value=(mock.Mock(), pre, post)):
            native = NativeSmolVLA.load(self.policy_dir)
        torch.testing.assert_close(native.preprocessor(batch)["observation.state"], torch.full((1, 7), -.5))

    def test_native_model_load_exception_is_typed_and_not_a_receipt(self):
        with mock.patch.object(NativeSmolVLA, '_load_components', side_effect=RuntimeError('tensor mismatch')):
            with self.assertRaisesRegex(ContractError, 'LEARNED_CHECKPOINT_LOAD_FAILED'):
                NativeSmolVLA.load(self.policy_dir)

    def test_canonical_admission_failure_prevents_native_component_loading(self):
        failure = ValueError("synthetic canonical admission rejection")
        self.admitted_checkpoint.side_effect = failure
        with mock.patch.object(NativeSmolVLA, "_load_components") as components:
            with self.assertRaisesRegex(ContractError, "LEARNED_CHECKPOINT_LOAD_FAILED") as rejected:
                NativeSmolVLA.load(self.policy_dir)
            self.assertIs(rejected.exception.__cause__, failure)
            components.assert_not_called()

    def test_runtime_restricts_processors_after_canonical_normalization_passes(self):
        for filename, step in (
            ("policy_preprocessor.json", {"registry_name": "unsupported_processor", "config": {}}),
            ("policy_postprocessor.json", {"registry_name": "device_processor", "config": {}, "class": "custom.Device"}),
        ):
            with self.subTest(filename=filename):
                path = self.policy_dir / filename
                original = path.read_bytes()
                document = json.loads(original)
                document["steps"].append(step)
                path.write_text(json.dumps(document))
                try:
                    with mock.patch.object(NativeSmolVLA, "_load_components") as components:
                        with self.assertRaisesRegex(ContractError, "LEARNED_PROCESSOR_CONTRACT"):
                            NativeSmolVLA.load(self.policy_dir)
                        components.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_inline_statistics_cannot_override_valid_saved_tensors(self):
        from lerobot.policies.factory import make_pre_post_processors
        for filename, feature in (("policy_preprocessor", "observation.state"),
                                  ("policy_postprocessor", "action")):
            with self.subTest(filename=filename):
                path = self.policy_dir / (filename + ".json")
                original = path.read_bytes()
                document = json.loads(original)
                document["steps"][-1]["config"]["stats"] = {
                    feature: {"mean": [0.] * 7, "std": [1.] * 7}}
                path.write_text(json.dumps(document))
                try:
                    # The installed library uses inline stats instead of its
                    # saved nonzero means; native admission must reject this.
                    pre, post = make_pre_post_processors(SimpleNamespace(), pretrained_path=str(self.policy_dir))
                    actual = (pre({feature: torch.zeros(7)})[feature]
                              if feature == "observation.state" else post(torch.zeros((1, 1, 7))))
                    torch.testing.assert_close(actual, torch.zeros_like(actual))
                    with mock.patch.object(NativeSmolVLA, "_load_components") as components:
                        with self.assertRaisesRegex(ContractError, "LEARNED_CHECKPOINT_LOAD_FAILED"):
                            NativeSmolVLA.load(self.policy_dir)
                        components.assert_not_called()
                finally:
                    path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
