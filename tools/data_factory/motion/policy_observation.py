"""Bounded latest ROS observations, independent of actuation completion.

The existing node owner spins callbacks. This observer never spins, waits for a
motion result, sends commands or writes evidence. Its consumer owns publication
and applies the task's authority; a valid observation is not motion admission.
"""
import math
import time

from tools.fr5_data_factory import ContractError
from tools.ros_image import image_message_to_rgb


class PolicyObservationStream:
    def __init__(self, node, state_sample, camera_topics, max_age_s, *, clock):
        from sensor_msgs.msg import Image
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.validate_full_topic_name import validate_full_topic_name
        from rclpy.exceptions import InvalidTopicNameException

        if (not isinstance(camera_topics, dict) or set(camera_topics) != {"camera1", "camera2"}
                or any(not isinstance(t, str) or not t.startswith("/") or t == "/"
                       for t in camera_topics.values())
                or len(set(camera_topics.values())) != 2):
            raise ContractError("LEARNED_CAMERA_TOPICS")
        try:
            for topic in camera_topics.values():
                validate_full_topic_name(topic)
        except InvalidTopicNameException as exc:
            raise ContractError("LEARNED_CAMERA_TOPICS") from exc
        if (isinstance(max_age_s, bool) or not isinstance(max_age_s, (int, float))
                or not math.isfinite(max_age_s) or not 0 < max_age_s <= 5):
            raise ContractError("LEARNED_SOURCE_CLOCK")
        if node.get_parameter("use_sim_time").value is not False:
            raise ContractError("LEARNED_SOURCE_CLOCK")
        self.node, self.state_sample, self.clock = node, state_sample, clock
        self.max_age_s = max_age_s
        self.started, self.started_system = clock(), time.time()
        self.initial_state = state_sample()[0]
        self.frames, self.subscriptions = {}, []
        self.closed = False
        try:
            for name, topic in camera_topics.items():
                self.subscriptions.append(node.create_subscription(
                    Image, topic, lambda message, name=name: self._receive(name, message),
                    qos_profile_sensor_data))
        except BaseException:
            self.close()
            raise

    def _receive(self, name, message):
        if not self.closed:
            self.frames[name] = (message, self.clock())

    def poll(self):
        """Return the latest source-timed tuple, or None until first readiness.

        At most two image messages are retained. A repeated poll preserves the
        original timestamps; it neither requires motion completion nor renews
        stale samples. RGB conversion belongs on the low-rate observation path.
        """
        if self.closed:
            raise ContractError("LEARNED_OBSERVATION_CLOSED")
        state, received = self.state_sample()
        if (len(self.frames) != 2 or state is None or state is self.initial_state
                or received is None or received < self.started):
            return None
        samples = {"state": (state, received), **self.frames}
        now, steady = time.time(), self.clock()
        age = self.max_age_s
        if (steady < self.started
                or abs((now - self.started_system) - (steady - self.started)) > age
                or self.node.get_parameter("use_sim_time").value is not False):
            raise ContractError("LEARNED_SOURCE_CLOCK")
        stamps = {}
        for name, (message, received) in samples.items():
            stamp = message.header.stamp
            if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                raise ContractError("LEARNED_SOURCE_CLOCK")
            source = stamp.sec + stamp.nanosec / 1e9
            if not 0 <= now - source <= age or not 0 <= steady - received <= age:
                raise ContractError("LEARNED_STALE_OBSERVATION")
            stamps[name] = source
        from tools.data_factory.rollout.finite_plan import JOINTS, check_freshness
        names, positions = list(state.name), list(state.position)
        if (len(names) != len(set(names)) or len(names) != len(positions)
                or not set(JOINTS).issubset(names)
                or any(not math.isfinite(value) for value in positions)):
            raise ContractError("ROS_JOINT_STATE")
        by_name = dict(zip(names, positions))
        observation = {"source_clock": "SYSTEM_TIME", "source_timestamps_s": stamps,
                       "observation.state": [by_name[name] for name in JOINTS]}
        for name in ("camera1", "camera2"):
            try:
                rgb = image_message_to_rgb(samples[name][0])
            except (ValueError, TypeError) as exc:
                raise ContractError("LEARNED_IMAGE", str(exc)) from exc
            observation[f"observation.images.{name}"] = {
                "dtype": "uint8", "color_space": "RGB", "shape": list(rgb.shape),
                "data_hex": rgb.tobytes().hex()}
        finished, finished_steady = time.time(), self.clock()
        check_freshness({"source_timestamps_s": stamps, "max_observation_age_s": age}, finished)
        if abs((finished - self.started_system) - (finished_steady - self.started)) > age:
            raise ContractError("LEARNED_SOURCE_CLOCK")
        if any(not 0 <= self.clock() - received <= age for _, received in samples.values()):
            raise ContractError("LEARNED_STALE_OBSERVATION")
        if self.node.get_parameter("use_sim_time").value is not False:
            raise ContractError("LEARNED_SOURCE_CLOCK")
        return observation

    def close(self):
        self.closed = True
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
        self.subscriptions.clear()
        self.frames.clear()
