from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState


JOINTS = (
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
    "j6",
    "finger_right_joint",
)


@dataclass
class _Sample:
    message: object
    received_monotonic_s: float
    generation: int


def _source_stamp_s(message) -> float:
    stamp = message.header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
        raise RuntimeError("FR5_SOURCE_CLOCK")
    return stamp.sec + stamp.nanosec / 1e9


def _image_to_rgb(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "yuyv": 2,
        "yuy2": 2,
    }.get(encoding)

    if channels is None:
        raise RuntimeError(
            f"FR5_IMAGE_ENCODING: {message.encoding}"
        )

    width = int(message.width)
    height = int(message.height)
    step = int(message.step)

    if width <= 0 or height <= 0 or step < width * channels:
        raise RuntimeError("FR5_IMAGE_SHAPE")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise RuntimeError("FR5_IMAGE_TRUNCATED")

    image = (
        raw[: height * step]
        .reshape(height, step)[:, : width * channels]
        .reshape(height, width, channels)
    )

    if encoding == "rgb8":
        rgb = np.ascontiguousarray(image)
    elif encoding == "bgr8":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding == "rgba8":
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif encoding == "bgra8":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif encoding == "mono8":
        rgb = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2)

    return np.ascontiguousarray(rgb)



# Bound to the training-time RGB-to-target synchronization contract.
# Do not relax independently from the qualified dataset contract.
OBSERVATION_ALIGNMENT_SLOP_S = 0.050


def _validate_source_alignment(
    source_timestamps: dict[str, float],
) -> None:
    required = {"state", "up", "wrist"}

    if set(source_timestamps) != required:
        raise RuntimeError("FR5_OBSERVATION_INCOMPLETE")

    state_stamp = source_timestamps["state"]

    for camera in ("up", "wrist"):
        delta_s = abs(
            source_timestamps[camera] - state_stamp
        )

        if (
            not math.isfinite(delta_s)
            or delta_s
            > OBSERVATION_ALIGNMENT_SLOP_S + 1e-9
        ):
            raise RuntimeError(
                f"FR5_OBSERVATION_MISALIGNED: {camera}"
            )


def build_observation_from_samples(
    samples: dict[str, _Sample],
    *,
    max_age_s: float,
    expected_shape: tuple[int, int, int] = (480, 640, 3),
):
    required = {"state", "up", "wrist"}
    if set(samples) != required:
        raise RuntimeError("FR5_OBSERVATION_INCOMPLETE")

    now_wall = time.time()
    now_mono = time.monotonic()

    source_timestamps = {}
    source_ages = {}
    receive_ages = {}

    for key, sample in samples.items():
        source = _source_stamp_s(sample.message)
        source_age = now_wall - source
        receive_age = now_mono - sample.received_monotonic_s

        if (
            source_age < 0
            or source_age > max_age_s
            or receive_age < 0
            or receive_age > max_age_s
        ):
            raise RuntimeError(
                f"FR5_OBSERVATION_STALE: {key}"
            )

        source_timestamps[key] = source
        source_ages[key] = source_age
        receive_ages[key] = receive_age

    _validate_source_alignment(source_timestamps)

    state: JointState = samples["state"].message

    names = list(state.name)
    positions = list(state.position)

    if (
        len(names) != len(set(names))
        or len(names) != len(positions)
        or not set(JOINTS).issubset(names)
    ):
        raise RuntimeError("FR5_JOINT_STATE")

    by_name = dict(zip(names, positions))
    values = []

    for joint in JOINTS:
        value = float(by_name[joint])
        if not math.isfinite(value):
            raise RuntimeError("FR5_JOINT_STATE")
        values.append(value)

    up = _image_to_rgb(samples["up"].message)
    wrist = _image_to_rgb(samples["wrist"].message)

    if up.shape != expected_shape or wrist.shape != expected_shape:
        raise RuntimeError(
            f"FR5_IMAGE_SHAPE: up={up.shape} wrist={wrist.shape}"
        )

    observation = {
        **{
            f"{joint}.pos": value
            for joint, value in zip(JOINTS, values)
        },
        "up": up,
        "wrist": wrist,
    }

    evidence = {
        "interface": "lerobot.robot.get_observation",
        "source_clock": "SYSTEM_TIME",
        "source_timestamps_s": source_timestamps,
        "source_ages_s": source_ages,
        "receive_ages_s": receive_ages,
        "source_span_s": (
            max(source_timestamps.values())
            - min(source_timestamps.values())
        ),
        "max_observation_age_s": max_age_s,
        "state": values,
        "images": {
            "up": {
                "shape": list(up.shape),
                "dtype": str(up.dtype),
                "sha256": hashlib.sha256(
                    up.tobytes()
                ).hexdigest(),
            },
            "wrist": {
                "shape": list(wrist.shape),
                "dtype": str(wrist.dtype),
                "sha256": hashlib.sha256(
                    wrist.tobytes()
                ).hexdigest(),
            },
        },
    }

    return observation, evidence


class FR5RosObservationBridge:
    def __init__(
        self,
        *,
        joint_topic: str,
        up_topic: str,
        wrist_topic: str,
        max_age_s: float,
        timeout_s: float,
        initial_timeout_s: float,
    ):
        self.joint_topic = joint_topic
        self.up_topic = up_topic
        self.wrist_topic = wrist_topic
        self.max_age_s = float(max_age_s)
        self.timeout_s = float(timeout_s)
        self.initial_timeout_s = float(initial_timeout_s)

        self.node = None
        self._owns_rclpy = False
        self._samples: dict[str, _Sample] = {}
        self._generation = 0
        self._initial_complete = False

    def _receive(self, key: str, message) -> None:
        self._generation += 1
        self._samples[key] = _Sample(
            message=message,
            received_monotonic_s=time.monotonic(),
            generation=self._generation,
        )

    def start(self) -> None:
        if self.node is not None:
            return

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self.node = rclpy.create_node(
            f"lerobot_fr5_observation_{os.getpid()}"
        )

        self.node.create_subscription(
            JointState,
            self.joint_topic,
            lambda msg: self._receive("state", msg),
            10,
        )
        self.node.create_subscription(
            Image,
            self.up_topic,
            lambda msg: self._receive("up", msg),
            qos_profile_sensor_data,
        )
        self.node.create_subscription(
            Image,
            self.wrist_topic,
            lambda msg: self._receive("wrist", msg),
            qos_profile_sensor_data,
        )

    def read(self):
        if self.node is None:
            raise RuntimeError("FR5_OBSERVATION_BRIDGE_NOT_STARTED")

        required = ("state", "up", "wrist")

        baseline = {
            key: sample.generation
            for key, sample in self._samples.items()
        }

        wait_budget_s = (
            self.timeout_s
            if self._initial_complete
            else self.initial_timeout_s
        )
        deadline = time.monotonic() + wait_budget_s
        last_alignment_error = None

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rclpy.spin_once(
                self.node,
                timeout_sec=max(0.0, min(0.02, remaining)),
            )

            if not all(
                key in self._samples
                and self._samples[key].generation
                > baseline.get(key, -1)
                for key in required
            ):
                continue

            samples = {
                key: self._samples[key]
                for key in required
            }

            try:
                result = build_observation_from_samples(
                    samples,
                    max_age_s=self.max_age_s,
                )
            except RuntimeError as exc:
                if not str(exc).startswith(
                    "FR5_OBSERVATION_MISALIGNED:"
                ):
                    raise

                last_alignment_error = str(exc)

                # Reject this asynchronous tuple and require every
                # source to advance before considering another one.
                baseline = {
                    key: self._samples[key].generation
                    for key in required
                }
                continue

            self._initial_complete = True
            return result

        waiting = [
            key
            for key in required
            if (
                key not in self._samples
                or self._samples[key].generation
                <= baseline.get(key, -1)
            )
        ]

        detail = (
            ""
            if last_alignment_error is None
            else f" last_alignment_error={last_alignment_error}"
        )

        raise RuntimeError(
            "FR5_OBSERVATION_TIMEOUT: "
            f"waiting={waiting} "
            f"seen={sorted(self._samples)} "
            f"initial={not self._initial_complete}"
            f"{detail}"
        )

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None

        self._samples.clear()
        self._initial_complete = False

        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

        self._owns_rclpy = False
