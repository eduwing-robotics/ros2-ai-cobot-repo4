"""Legacy fake fault harness and strict local SmolVLA inference for finite plans."""

from __future__ import annotations

import copy
import math
import threading
import time
from typing import Callable
from contextlib import contextmanager


IDLE = "IDLE"
ACTIVE = "ACTIVE"
STOPPED = "STOPPED"
FAULT = "FAULT"
TERMINAL_STATES = {STOPPED, FAULT}
_OBSERVATION_KEYS = {
    "captured_at_s", "observation.state", "observation.images.camera1",
    "observation.images.camera2",
}
_RGB_KEYS = {"dtype", "color_space", "shape", "data"}


class FakeCommandSink:
    """In-memory sink that permits one active fake command owner."""

    def __init__(self, *, fail_send: bool = False):
        self.active_owner: str | None = None
        self.commands: list[tuple[str, tuple[float, ...]]] = []
        self.fail_send = fail_send

    def claim(self, owner: str) -> None:
        if self.active_owner is not None:
            raise RuntimeError("COMMAND_OWNER_BUSY")
        self.active_owner = owner

    def send(self, owner: str, action: tuple[float, ...]) -> None:
        if owner != self.active_owner:
            raise RuntimeError("COMMAND_OWNER_MISMATCH")
        if self.fail_send:
            raise RuntimeError("SINK_FAULT")
        self.commands.append((owner, action))

    def release(self, owner: str) -> None:
        if self.active_owner == owner:
            self.active_owner = None


def fake_rgb(data: bytes = b"\x00\x00\x00", *, height: int = 1, width: int = 1) -> dict:
    if (
        isinstance(height, bool) or isinstance(width, bool) or not isinstance(height, int)
        or not isinstance(width, int) or height < 1 or width < 1 or not isinstance(data, bytes)
        or len(data) != height * width * 3
    ):
        raise ValueError("RGB_FRAME")
    return {"dtype": "uint8", "color_space": "RGB", "shape": [height, width, 3], "data": data}


def fake_observation(
    captured_at_s: float,
    *,
    state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    camera1: dict | None = None,
    camera2: dict | None = None,
) -> dict:
    return {
        "captured_at_s": captured_at_s,
        "observation.state": list(state),
        "observation.images.camera1": copy.deepcopy(camera1) if camera1 is not None else fake_rgb(),
        "observation.images.camera2": copy.deepcopy(camera2) if camera2 is not None else fake_rgb(),
    }


def _finite_number(value: object) -> bool:
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


def _rgb(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != _RGB_KEYS:
        raise ValueError("RGB_FRAME")
    shape = value["shape"]
    if (
        value["dtype"] != "uint8" or value["color_space"] != "RGB"
        or not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
        or not isinstance(value["data"], bytes) or len(value["data"]) != math.prod(shape)
    ):
        raise ValueError("RGB_FRAME")
    return copy.deepcopy(value)


def _action(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 7 or not all(
        _finite_number(item) for item in value
    ):
        raise ValueError("INVALID_ACTION")
    return tuple(float(item) for item in value)


class LearnedActionAdapter:
    """Single-goal synchronous adapter for fake policy and fake sink qualification."""

    def __init__(
        self,
        policy: Callable[[dict], object],
        sink: FakeCommandSink,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_observation_age_s: float = 0.3,
        watchdog_timeout_s: float = 1.0,
        owner_id: str = "learned-action-fake",
    ):
        if (
            not callable(policy) or not isinstance(sink, FakeCommandSink) or not callable(clock)
            or not _finite_number(max_observation_age_s) or max_observation_age_s <= 0
            or not _finite_number(watchdog_timeout_s) or watchdog_timeout_s <= 0
            or not isinstance(owner_id, str) or not owner_id
        ):
            raise ValueError("ADAPTER_CONFIG")
        self.policy = policy
        self.sink = sink
        self.clock = clock
        self.max_observation_age_s = float(max_observation_age_s)
        self.watchdog_timeout_s = float(watchdog_timeout_s)
        self.owner_id = owner_id
        self.state = IDLE
        self.terminal_reason: str | None = None
        self.active_goal_id: str | None = None
        self.last_progress_s: float | None = None
        self.policy_calls = 0
        self._owns_sink = False
        self._in_step = False

    def _now(self) -> float:
        value = self.clock()
        if not _finite_number(value):
            raise ValueError("CLOCK")
        return float(value)

    def _terminal(self, state: str, reason: str) -> str:
        if self.state in TERMINAL_STATES:
            return self.state
        self.state = state
        self.terminal_reason = reason
        self.active_goal_id = None
        if self._owns_sink:
            self.sink.release(self.owner_id)
            self._owns_sink = False
        return self.state

    def start(self, goal_id: str) -> str:
        if self.state in TERMINAL_STATES:
            return self.state
        if self.state == ACTIVE:
            return self._terminal(FAULT, "COMPETING_GOAL")
        if not isinstance(goal_id, str) or not goal_id or "\x00" in goal_id:
            return self._terminal(FAULT, "GOAL_ID")
        try:
            now = self._now()
            self.sink.claim(self.owner_id)
            self._owns_sink = True
        except Exception:
            return self._terminal(FAULT, "COMMAND_OWNER")
        self.state = ACTIVE
        self.active_goal_id = goal_id
        self.last_progress_s = now
        return self.state

    def stop(self) -> str:
        return self._terminal(STOPPED, "STOP_REQUESTED")

    def cancel(self) -> str:
        return self._terminal(STOPPED, "CANCELLED")

    def check_watchdog(self) -> str:
        if self.state != ACTIVE:
            return self.state
        try:
            expired = self._now() - self.last_progress_s > self.watchdog_timeout_s
        except Exception:
            return self._terminal(FAULT, "CLOCK")
        return self._terminal(FAULT, "WATCHDOG") if expired else self.state

    def _observation(self, value: object, now: float) -> dict:
        if not isinstance(value, dict) or set(value) != _OBSERVATION_KEYS:
            raise ValueError("OBSERVATION_SCHEMA")
        captured = value["captured_at_s"]
        state = value["observation.state"]
        if (
            not _finite_number(captured) or now < captured or now - captured > self.max_observation_age_s
        ):
            raise ValueError("STALE_OBSERVATION")
        if not isinstance(state, (list, tuple)) or len(state) != 7 or not all(
            _finite_number(item) for item in state
        ):
            raise ValueError("OBSERVATION_STATE")
        return {
            "observation.state": [float(item) for item in state],
            "observation.images.camera1": _rgb(value["observation.images.camera1"]),
            "observation.images.camera2": _rgb(value["observation.images.camera2"]),
        }

    def step(self, observation: object) -> str:
        if self._in_step:
            return self._terminal(FAULT, "REENTRANT_STEP")
        self._in_step = True
        try:
            return self._step(observation)
        finally:
            self._in_step = False

    def _step(self, observation: object) -> str:
        if self.state != ACTIVE:
            return self.state
        if self.check_watchdog() != ACTIVE:
            return self.state
        try:
            policy_input = self._observation(observation, self._now())
        except Exception as error:
            reason = str(error) if str(error) else "INVALID_OBSERVATION"
            return self._terminal(FAULT, reason)
        try:
            self.policy_calls += 1
            output = self.policy(policy_input)
        except Exception:
            return self._terminal(FAULT, "POLICY_EXCEPTION")
        if self.state != ACTIVE:
            return self.state
        if self.check_watchdog() != ACTIVE:
            return self.state
        try:
            action = _action(output)
        except Exception:
            return self._terminal(FAULT, "INVALID_ACTION")
        try:
            # Inference can outlive source freshness without tripping the watchdog.
            self._observation(observation, self._now())
        except Exception as error:
            return self._terminal(FAULT, str(error) or "STALE_OBSERVATION")
        try:
            self.sink.send(self.owner_id, action)
            self.last_progress_s = self._now()
        except Exception:
            return self._terminal(FAULT, "SINK_FAULT")
        return self.state


class NativeSmolVLA:
    """Strict local checkpoint + saved processor inference, with no execution API.

    Resource qualification is external. Offline dependency caches must already
    exist; this loader cannot download or install model/tokenizer dependencies.
    """

    def __init__(self):
        self._inference_lock = threading.Lock()
        self._inference_guard = threading.Lock()
        self._retained_inference_thread = None

    @classmethod
    def load(cls, checkpoint, *, device="cpu"):
        import importlib.metadata
        import json
        import os
        from pathlib import Path
        from tools.fr5_data_factory import ContractError, canonical_digest
        from tools.data_factory.training_receipts import tree_digest
        from tools.validate_training_checkpoint import validate_checkpoint

        try:
            # Admission is necessary but does not prove that weights can load.
            policy_dir, output_dir = validate_checkpoint(Path(checkpoint), verify_dataset=True)
            if importlib.metadata.version("lerobot") != "0.6.1":
                raise ContractError("LEARNED_RUNTIME_VERSION")
            if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
                raise ContractError("LEARNED_OFFLINE_RUNTIME_REQUIRED")
            receipt = output_dir / "fr5_training_receipt.json"
            if not receipt.is_file():
                receipt = Path(str(output_dir) + ".fr5_training_receipt.json.pending")
            split_path = output_dir / "fr5_training_split.json"
            if not split_path.is_file():
                split_path = output_dir.with_name(output_dir.name + ".fr5_training_split.json.pending")
            # Launch evidence lives outside policy_dir: bind it separately across
            # model construction, without substituting raw pixels if it vanishes.
            manifests = {path: path.read_bytes() for path in (split_path, receipt)}
            before = tree_digest(policy_dir)
            from safetensors import safe_open
            with safe_open(policy_dir / "model.safetensors", framework="numpy") as tensors:
                if not tensors.keys():
                    raise ContractError("LEARNED_EMPTY_WEIGHTS")
            # Canonical admission owns saved normalization semantics/state.
            # This runtime only permits its supported processor implementations.
            for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
                config = json.loads((policy_dir / filename).read_text())
                allowed = ({"rename_observations_processor", "to_batch_processor", "smolvla_new_line_processor",
                            "tokenizer_processor", "device_processor", "normalizer_processor"}
                           if filename == "policy_preprocessor.json"
                           else {"unnormalizer_processor", "device_processor"})
                if any("class" in step or step.get("registry_name") not in allowed for step in config["steps"]):
                    raise ContractError("LEARNED_PROCESSOR_CONTRACT")
            policy, preprocessor, postprocessor = cls._load_components(policy_dir, device)
            # CPU .to() can retain safetensors mmap storage. Own the small saved
            # processor tensors before checking the final loaded-artifact digest.
            for processor in (preprocessor, postprocessor):
                for step in processor.steps:
                    state = step.state_dict()
                    if state:
                        step.load_state_dict({key: tensor.clone() for key, tensor in state.items()})
            if tree_digest(policy_dir) != before:
                raise ContractError("LEARNED_CHECKPOINT_CHANGED")
            from tools.fr5_data_factory import load_json_strict
            from tools.validate_training_checkpoint import validate_saved_observation_view
            if any(not path.is_file() or path.read_bytes() != value for path, value in manifests.items()):
                raise ContractError("LEARNED_CHECKPOINT_CHANGED")
            split = load_json_strict(split_path)
            receipt_value = load_json_strict(receipt)
            observation_view = validate_saved_observation_view(split, receipt_value)
            if any(not path.is_file() or path.read_bytes() != value for path, value in manifests.items()):
                raise ContractError("LEARNED_CHECKPOINT_CHANGED")
            instance = cls()
            instance.policy, instance.preprocessor, instance.postprocessor = policy, preprocessor, postprocessor
            instance.checkpoint = {"tree_digest": before,
                                   "training_receipt_digest": canonical_digest(receipt_value),
                                   "runtime": "lerobot-0.6.1-native"}
            instance.policy_dir = policy_dir
            instance.observation_view = observation_view
            instance.device = device
            return instance
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError("LEARNED_CHECKPOINT_LOAD_FAILED") from exc

    @staticmethod
    def _load_components(policy_dir, device):
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from tools.fr5_data_factory import ContractError

        config = SmolVLAConfig.from_pretrained(policy_dir, local_files_only=True)
        if (list(config.input_features["observation.state"].shape) != [7]
                or list(config.output_features["action"].shape) != [7]
                or config.adapt_to_pi_aloha or config.rtc_config is not None):
            raise ContractError("LEARNED_MODEL_FEATURES")
        # validate_checkpoint owns the exact admitted image features, including
        # source-proven inert slots retained by the native SmolVLA serializer.
        if (config.empty_cameras != 1 or not {
                "observation.images.camera1", "observation.images.camera2",
        }.issubset(config.input_features)):
            raise ContractError("LEARNED_MODEL_CAMERAS")
        config.device = device
        # Preserve saved construction settings: changing load_vlm_weights changes
        # parameter dtypes before native checkpoint tensors are copied into them.
        policy = SmolVLAPolicy.from_pretrained(policy_dir, config=config, strict=True, local_files_only=True)
        pre, post = make_pre_post_processors(
            config, pretrained_path=str(policy_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        return policy, pre, post

    def warmup(self, *, instruction, height, width, cancel_event=None):
        """Discard one shape-matched call before capture, restoring RNG streams."""
        import random
        import numpy as np
        import torch
        from tools.fr5_data_factory import ContractError, canonical_digest

        if (not isinstance(instruction, str) or not instruction.strip()
                or any(type(n) is not int or n < 1 for n in (height, width))):
            raise ContractError("LEARNED_WARMUP_INPUT")
        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise ContractError("LEARNED_CANCELLED")
        check_cancel()
        started = time.perf_counter()
        frame = fake_rgb(bytes(height * width * 3), height=height, width=width)
        observation = {"observation.state": [0.] * 7, "task": instruction,
                       "observation.images.camera1": frame, "observation.images.camera2": frame}
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with self.prepare_inference() as predict:
            check_cancel()
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=devices):
                    try:
                        inference_started = time.perf_counter()
                        predict(observation)  # No proposal: these synthetic outputs are discarded.
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_s = time.perf_counter() - inference_started
                        check_cancel()
                    finally:
                        self.policy.reset()
                        self.preprocessor.reset()
                        self.postprocessor.reset()
            except ContractError:
                raise
            except Exception as exc:
                raise ContractError("LEARNED_WARMUP_FAILED") from exc
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        return {"input_kind": "SYNTHETIC_ZERO_RGB_STATE", "image_shape": [height, width, 3],
                "instruction_digest": canonical_digest(instruction), "device": self.device,
                "model_calls": 1, "output_disposition": "DISCARDED", "rng_state_restored": True,
                "duration_s": time.perf_counter() - started, "inference_duration_s": inference_s}

    @contextmanager
    def prepare_inference(self):
        """Check artifacts before capture and reserve one loaded-model call.

        The scope uses owned loaded tensors, not a promise that files cannot
        change. A subsequent scope rechecks bytes; this scope never reloads them.
        """
        from tools.fr5_data_factory import ContractError
        from tools.data_factory.training_receipts import tree_digest

        self._acquire_inference()
        available = False
        owner = threading.get_ident()
        try:
            if tree_digest(self.policy_dir) != self.checkpoint["tree_digest"]:
                raise ContractError("LEARNED_CHECKPOINT_CHANGED")
            available = True
            def predict(observation):
                nonlocal available
                if not available or threading.get_ident() != owner:
                    raise ContractError("LEARNED_INFERENCE_SCOPE")
                available = False
                return self._predict(observation)
            yield predict
        finally:
            available = False
            self._release_inference()

    @contextmanager
    def prepare_task_inference(
        self,
        *,
        instruction,
        fps,
        shutdown_event=None,
        queue_threshold=30,
    ):
        """Own one native asynchronous producer for a complete task.

        Yields ``(engine, queue, notify_observation)``. The callback accepts the
        same canonical FR5 observation as the one-shot API, while the engine
        retains LeRobot's preprocessing, inference loop, and queue merge.
        """
        import numpy as np
        from types import SimpleNamespace
        from lerobot.policies.rtc import RTCConfig
        from lerobot.rollout.inference import RTCInferenceConfig, create_inference_engine
        from lerobot_strategy_fr5.acknowledged_queue import start_with_acknowledged_queue
        from tools.fr5_data_factory import ContractError
        from tools.data_factory.training_receipts import tree_digest

        if (
            not isinstance(instruction, str)
            or not instruction.strip()
            or not _finite_number(fps)
            or fps <= 0
            or isinstance(queue_threshold, bool)
            or not isinstance(queue_threshold, int)
            or queue_threshold < 0
        ):
            raise ContractError("LEARNED_TASK_INFERENCE_INPUT")

        self._acquire_inference()
        engine = None
        producer_thread = None
        retain_owner = False
        available = False
        body_error = None
        notification_lock = threading.Lock()
        try:
            # This is the only artifact revalidation in the task scope. It
            # precedes processor reset, producer start, and observation capture.
            if tree_digest(self.policy_dir) != self.checkpoint["tree_digest"]:
                raise ContractError("LEARNED_CHECKPOINT_CHANGED")

            state_names = (
                "j1.pos", "j2.pos", "j3.pos", "j4.pos", "j5.pos", "j6.pos",
                "finger_right_joint.pos",
            )
            hw_features = {
                "observation.state": {
                    "dtype": "float32", "shape": (7,), "names": list(state_names),
                },
            }
            for name in ("camera1", "camera2"):
                key = f"observation.images.{name}"
                shape = tuple(self.policy.config.input_features[key].shape)
                if len(shape) != 3 or shape[0] != 3 or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                    for value in shape
                ):
                    raise ContractError("LEARNED_MODEL_CAMERAS")
                hw_features[key] = {
                    "dtype": "image",
                    "shape": (shape[1], shape[2], shape[0]),
                    "names": ["height", "width", "channels"],
                }
            dataset_features = {
                **hw_features,
                "action": {
                    "dtype": "float32", "shape": (7,), "names": list(state_names),
                },
            }
            robot_view = SimpleNamespace(
                robot_type="fr5",
                action_features={name: float for name in state_names},
            )

            # The loaded policy config remains exactly as saved (rtc_config is
            # None). RTCConfig(enabled=False) selects only the native async
            # producer/append queue, with no RTC guidance algorithm.
            config = RTCInferenceConfig(
                rtc=RTCConfig(enabled=False), queue_threshold=queue_threshold,
            )
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            engine = create_inference_engine(
                config,
                policy=self.policy,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                robot_wrapper=robot_view,
                hw_features=hw_features,
                dataset_features=dataset_features,
                ordered_action_keys=list(state_names),
                task=instruction,
                fps=float(fps),
                device=self.device,
                use_torch_compile=False,
                shutdown_event=shutdown_event,
            )
            queue = start_with_acknowledged_queue(engine)
            # Native stop() clears its handle even when join times out. Retain
            # the actual thread now, including if the caller stops it early.
            producer_thread = engine._rtc_thread

            def notify_observation(observation):
                with notification_lock:
                    if not available:
                        raise ContractError("LEARNED_INFERENCE_SCOPE")
                    try:
                        state = observation["observation.state"]
                        if not isinstance(state, (list, tuple)) or len(state) != 7 or not all(
                            _finite_number(value) for value in state
                        ):
                            raise ValueError("OBSERVATION_STATE")
                        raw = {name: float(value) for name, value in zip(state_names, state)}
                        for name in ("camera1", "camera2"):
                            key = f"observation.images.{name}"
                            frame = _rgb(observation[key])
                            if (
                                key == "observation.images.camera1"
                                and self.observation_view.get("representation") == "baked"
                                and self.observation_view.get("transform_application") == "rollout_once"
                            ):
                                frame = self._transform_raw_up(frame)
                            if tuple(frame["shape"]) != tuple(hw_features[key]["shape"]):
                                raise ValueError("OBSERVATION_IMAGE_SHAPE")
                            raw[name] = np.frombuffer(frame["data"], dtype=np.uint8).reshape(
                                frame["shape"]
                            ).copy()
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ContractError(str(exc) or "LEARNED_OBSERVATION") from exc
                    engine.notify_observation(raw)

            available = True
            try:
                yield engine, queue, notify_observation
            except BaseException as exc:
                body_error = exc
                raise
        finally:
            with notification_lock:
                available = False
            stop_error = None
            try:
                if engine is not None:
                    try:
                        engine.stop()
                    except BaseException as exc:
                        stop_error = exc
                    finally:
                        retain_owner = (
                            producer_thread is not None and producer_thread.is_alive()
                        )
            finally:
                if retain_owner:
                    with self._inference_guard:
                        self._retained_inference_thread = producer_thread
                else:
                    self._release_inference()
            if retain_owner:
                note = "LEARNED_INFERENCE_STOP_UNCONFIRMED"
                if body_error is not None:
                    body_error.add_note(note)
                elif stop_error is not None:
                    stop_error.add_note(note)
                    raise stop_error
                else:
                    raise ContractError(note)
            elif stop_error is not None:
                if body_error is not None:
                    body_error.add_note(
                        f"LEARNED_INFERENCE_STOP_FAILED: {stop_error!r}"
                    )
                else:
                    raise stop_error

    def _acquire_inference(self):
        """Acquire the loaded model, reaping a producer that has really exited."""
        from tools.fr5_data_factory import ContractError

        with self._inference_guard:
            retained = self._retained_inference_thread
            if retained is not None and not retained.is_alive():
                self._retained_inference_thread = None
                self._inference_lock.release()
            if not self._inference_lock.acquire(blocking=False):
                raise ContractError("LEARNED_REENTRANT_INFERENCE")

    def _release_inference(self):
        with self._inference_guard:
            self._inference_lock.release()

    def __call__(self, observation):
        with self.prepare_inference() as predict:
            return predict(observation)

    def _predict(self, observation):
        import numpy as np
        import torch
        from tools.fr5_data_factory import ContractError

        value = {"observation.state": torch.tensor(observation["observation.state"], dtype=torch.float32),
                 "task": observation["task"]}
        for key in ("observation.images.camera1", "observation.images.camera2"):
            frame = _rgb(observation[key])
            if (key == "observation.images.camera1"
                    and self.observation_view.get("representation") == "baked"
                    and self.observation_view.get("transform_application") == "rollout_once"):
                frame = self._transform_raw_up(frame)
            value[key] = torch.frombuffer(bytearray(frame["data"]), dtype=torch.uint8).reshape(frame["shape"]).permute(2, 0, 1).float() / 255
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        with torch.inference_mode():
            output = self.postprocessor(self.policy.predict_action_chunk(self.preprocessor(value)))
        if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 7:
            raise ContractError("LEARNED_ACTION_7D")
        return output[0].detach().cpu().tolist()

    def _transform_raw_up(self, frame: dict) -> dict:
        """Apply Curator's published up-view transform exactly once for raw Rollout input."""
        import numpy as np
        from pathlib import Path
        from tools.fr5_data_factory import ContractError
        from tools.data_factory.curator.core.errors import CuratorError
        from tools.data_factory.curator.profile.registry import load_profile_assets, resolve_view_profile
        from tools.data_factory.curator.profile.schema import load_view_profile
        from tools.data_factory.curator.profile.transform import apply_up_view

        profile = self.observation_view["view_profile"]
        path = profile["path"]
        try:
            spec = load_view_profile(path)
            resolved = resolve_view_profile(
                str(Path(path).parent), spec.value["profile_id"],
                binding_root=spec.binding_path.parent,
                collection_profile_root=spec.collection_profile_path.parent,
            )
            if (resolved.config_file_sha256 != profile.get("file_sha256")
                    or resolved.profile["profile_digest"] != profile.get("profile_digest")):
                raise ContractError("LEARNED_VIEW_PROFILE_CHANGED")
            mask, plate = load_profile_assets(resolved)
        except ContractError:
            raise
        except (CuratorError, OSError, ValueError, KeyError) as error:
            raise ContractError("LEARNED_VIEW_PROFILE_CHANGED") from error
        rgb = np.frombuffer(frame["data"], dtype=np.uint8).reshape(frame["shape"])
        if [frame["shape"][1], frame["shape"][0]] != [spec.value["width"], spec.value["height"]]:
            raise ContractError("LEARNED_VIEW_FRAME_SHAPE")
        transformed = apply_up_view(rgb, mask, plate)
        return {**frame, "shape": list(transformed.shape), "data": transformed.tobytes()}
