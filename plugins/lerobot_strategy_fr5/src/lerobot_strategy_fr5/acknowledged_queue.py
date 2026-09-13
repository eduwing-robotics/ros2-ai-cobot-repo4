"""Controller-paced acknowledgment for LeRobot's native action queue.

The adapter adds an atomic snapshot/conditional-advance seam.  It does not
interpret action-goal acceptance or terminal state as consumption: callers
must advance only from commanded-reference progress for the complete action
row.  In particular, ARM-only progress is not acknowledgment of a 7D row.
"""

import math
from dataclasses import dataclass
from importlib.metadata import version
from threading import local, RLock

from torch import Tensor

from lerobot.policies.rtc import ActionQueue, RTCConfig


LEROBOT_VERSION = "0.6.1"


def start_with_acknowledged_queue(engine, *, require_observation_provenance=False):
    """Start a fresh native producer with the controller-facing queue adapter.

    The task owner must call this before publishing observations or resuming
    inference. LeRobot 0.6.1 starts paused with no observation, so replacing its
    initial empty queue here cannot race a merge. No producer loop is copied or
    patched globally. The same owner retains native pause/resume/stop ownership.
    This installs no robot consumer and grants no execution authority.
    """
    from lerobot.rollout.inference.rtc import RTCInferenceEngine

    if not isinstance(engine, RTCInferenceEngine):
        raise TypeError("FR5_ACK_QUEUE_REQUIRES_NATIVE_ASYNC_ENGINE")
    if (engine.action_queue is not None or engine._rtc_thread is not None
            or engine._policy_active.is_set()):
        raise RuntimeError("FR5_ACK_QUEUE_REQUIRES_FRESH_ENGINE")
    queue = AcknowledgedActionQueue(
        engine._rtc_config,
        require_observation_provenance=require_observation_provenance,
    )
    engine.start()
    # start() has no queue-factory parameter in the pinned upstream version.
    # This one private assignment is the compatibility seam; scheduling,
    # processing, inference and merge remain native responsibilities.
    engine._action_queue = queue
    return queue


@dataclass(frozen=True, order=True)
class RawActionIndex:
    """Stable identity of one row in one native queue merge."""

    chunk: int
    index: int


@dataclass(frozen=True)
class ObservationProvenance:
    """Immutable source identity for the observation sampled by one inference."""

    observation_digest: str
    source_clock: str
    source_timestamps_s: tuple[tuple[str, float], ...]

    def __post_init__(self):
        if (
            not isinstance(self.observation_digest, str)
            or not self.observation_digest.startswith("sha256:")
            or len(self.observation_digest) != 71
            or any(character not in "0123456789abcdef" for character in self.observation_digest[7:])
            or not isinstance(self.source_clock, str)
            or not self.source_clock
            or type(self.source_timestamps_s) is not tuple
            or any(type(item) is not tuple or len(item) != 2 for item in self.source_timestamps_s)
        ):
            raise ValueError("FR5_ACK_QUEUE_OBSERVATION_PROVENANCE")
        names = tuple(name for name, _ in self.source_timestamps_s)
        if names != ("camera1", "camera2", "state") or any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for name, value in self.source_timestamps_s
        ):
            raise ValueError("FR5_ACK_QUEUE_OBSERVATION_PROVENANCE")


@dataclass(frozen=True)
class ActionQueueSnapshot:
    """Detached view of the currently unconsumed native queue."""

    generation: int
    queue_index: int
    rows: tuple[RawActionIndex, ...]
    observation_provenance: tuple[ObservationProvenance | None, ...]
    original_actions: Tensor | None
    processed_actions: Tensor | None


class _ObservedInput(dict):
    """Pinned 0.6.1 input whose native feature reads mark the sampled identity."""

    def __init__(self, values, queue, provenance):
        super().__init__(values)
        self._queue = queue
        self._provenance = provenance

    def __getitem__(self, key):
        self._queue._bind_sampled_observation(self._provenance)
        return super().__getitem__(key)


class AcknowledgedActionQueue(ActionQueue):
    """LeRobot 0.6.1 ``ActionQueue`` with compare-and-advance support.

    Native ``get``, ``clear`` and ``merge`` behavior is retained.  ``merge``
    and ``clear`` additionally invalidate old snapshots; row identities remain
    stable across the native non-RTC append/compaction operation.

    RTC-enabled replacement is also observable, but its native delay policy is
    deliberately unchanged.  This class does not make RTC controller-paced.
    """

    def __init__(self, cfg: RTCConfig, *, require_observation_provenance=False):
        installed = version("lerobot")
        if installed != LEROBOT_VERSION:
            raise RuntimeError(f"FR5_ACK_QUEUE_UNAUDITED: {installed}")
        if require_observation_provenance and cfg.enabled:
            raise ValueError("FR5_ACK_QUEUE_PROVENANCE_REQUIRES_APPEND_ONLY")

        super().__init__(cfg)
        # Native methods acquire ``self.lock`` internally.  Reentrancy lets the
        # adapter compose metadata updates with those methods atomically rather
        # than copying their merge/clear algorithms.
        self.lock = RLock()
        self._generation = 0
        self._next_chunk = 0
        self._rows: tuple[RawActionIndex, ...] = ()
        self._observation_provenance: tuple[ObservationProvenance | None, ...] = ()
        self._sampled_observation = local()
        self._require_observation_provenance = require_observation_provenance

    def observed_input(self, values: dict, provenance: ObservationProvenance) -> dict:
        """Bind an owned input to the exact native feature-read/merge thread.

        This task-internal association is not an authenticity guarantee for the
        source clock or payload; their admission remains the caller's contract.
        """

        if type(values) is not dict or not isinstance(provenance, ObservationProvenance):
            raise TypeError("FR5_ACK_QUEUE_OBSERVATION_PROVENANCE")
        return _ObservedInput(values, self, provenance)

    def _bind_sampled_observation(self, provenance):
        self._sampled_observation.value = provenance

    def get_action_index(self) -> int:
        """Retain the native cursor read and begin a new provenance attempt."""

        # Pinned RTCInferenceEngine calls this immediately before it reads the
        # selected observation. A failed inference never reaches merge, so clear
        # its thread-local identity here before the next feature-read can bind.
        if hasattr(self._sampled_observation, "value"):
            del self._sampled_observation.value
        return super().get_action_index()

    @property
    def generation(self) -> int:
        with self.lock:
            return self._generation

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
    ):
        with self.lock:
            provenance = getattr(self._sampled_observation, "value", None)
            if self._require_observation_provenance and provenance is None:
                raise RuntimeError("FR5_ACK_QUEUE_OBSERVATION_PROVENANCE_MISSING")
            saved_queue = self.queue
            saved_original_queue = self.original_queue
            saved_last_index = self.last_index
            retained_rows = self._rows[self.last_index :]
            retained_provenance = self._observation_provenance[self.last_index :]
            chunk = self._next_chunk + 1

            try:
                result = super().merge(
                    original_actions,
                    processed_actions,
                    real_delay,
                    action_index_before_inference,
                )

                if self.cfg.enabled:
                    queued = 0 if self.queue is None else len(self.queue)
                    raw_start = len(processed_actions) - queued
                    rows = tuple(
                        RawActionIndex(chunk, raw_start + i) for i in range(queued)
                    )
                    provenance_rows = (provenance,) * queued
                else:
                    rows = retained_rows + tuple(
                        RawActionIndex(chunk, i) for i in range(len(processed_actions))
                    )
                    provenance_rows = retained_provenance + (
                        (provenance,) * len(processed_actions)
                    )

                queued = 0 if self.queue is None else len(self.queue)
                if len(rows) != queued or len(provenance_rows) != queued:
                    raise RuntimeError("FR5_ACK_QUEUE_NATIVE_LAYOUT_CHANGED")
            except BaseException:
                self.queue = saved_queue
                self.original_queue = saved_original_queue
                self.last_index = saved_last_index
                raise
            finally:
                if hasattr(self._sampled_observation, "value"):
                    del self._sampled_observation.value

            self._rows = rows
            self._observation_provenance = provenance_rows
            self._next_chunk = chunk
            self._generation += 1
            return result

    def clear(self) -> None:
        with self.lock:
            super().clear()
            self._rows = ()
            self._observation_provenance = ()
            self._generation += 1

    def snapshot(self) -> ActionQueueSnapshot:
        """Clone the unconsumed raw/processed rows with their stable IDs."""

        with self.lock:
            original = (
                None
                if self.original_queue is None
                else self.original_queue[self.last_index :].clone()
            )
            processed = (
                None if self.queue is None else self.queue[self.last_index :].clone()
            )
            return ActionQueueSnapshot(
                generation=self._generation,
                queue_index=self.last_index,
                rows=self._rows[self.last_index :],
                observation_provenance=self._observation_provenance[self.last_index :],
                original_actions=original,
                processed_actions=processed,
            )

    def advance_if_current(
        self,
        *,
        generation: int,
        expected_index: int,
        count: int,
    ) -> tuple[RawActionIndex, ...]:
        """Advance only if neither queue contents nor its cursor have changed.

        The returned identities name the rows advanced.  This method carries no
        robot acceptance, completion, or observation semantics.
        """

        for name, value in (
            ("generation", generation),
            ("expected_index", expected_index),
            ("count", count),
        ):
            if type(value) is not int:
                raise TypeError(
                    f"FR5_ACK_QUEUE_EXPECTED_INTEGER: {name}={value!r}"
                )

        with self.lock:
            if generation != self._generation:
                raise RuntimeError(
                    f"FR5_ACK_QUEUE_STALE_GENERATION: {generation} != {self._generation}"
                )
            if expected_index != self.last_index:
                raise RuntimeError(
                    f"FR5_ACK_QUEUE_CURSOR_MOVED: {expected_index} != {self.last_index}"
                )
            if count < 0 or self.last_index + count > len(self._rows):
                raise ValueError(f"FR5_ACK_QUEUE_INVALID_COUNT: {count}")

            start = self.last_index
            self.last_index += count
            return self._rows[start : self.last_index]

    def advance_unchanged_prefix(
        self, snapshot: ActionQueueSnapshot, *, count: int, offset: int = 0,
    ) -> tuple[RawActionIndex, ...]:
        """Consume a checked prefix even if native inference appended meanwhile.

        Unlike generation-wide compare-and-advance, this permits a new tail or
        native compaction only when the requested front rows, input provenance,
        and both raw/processed tensors are still identical. Comparison and
        advancement share the native lock; there is no blind retry or second
        cursor. The caller supplies reference progress, not goal acceptance or
        physical target-arrival claims. An already consumed prefix cannot replay.
        ``offset`` identifies the next unacknowledged row in that same detached
        revision snapshot; it is checked against the native queue front.
        """
        if not isinstance(snapshot, ActionQueueSnapshot):
            raise TypeError("FR5_ACK_QUEUE_EXPECTED_SNAPSHOT")
        if type(count) is not int or type(offset) is not int:
            raise TypeError("FR5_ACK_QUEUE_EXPECTED_INTEGER: count/offset")
        if count < 0 or offset < 0 or offset + count > len(snapshot.rows):
            raise ValueError(f"FR5_ACK_QUEUE_INVALID_COUNT: {count}")
        with self.lock:
            if count == 0:
                return ()
            end = self.last_index + count
            if (self._rows[self.last_index:end] != snapshot.rows[offset:offset + count]
                    or self._observation_provenance[self.last_index:end]
                    != snapshot.observation_provenance[offset:offset + count]):
                raise RuntimeError("FR5_ACK_QUEUE_PREFIX_CHANGED")
            for saved, current in ((snapshot.original_actions, self.original_queue),
                                   (snapshot.processed_actions, self.queue)):
                if not isinstance(saved, Tensor) or not isinstance(current, Tensor):
                    raise RuntimeError("FR5_ACK_QUEUE_PREFIX_CHANGED")
                expected, actual = saved[offset:offset + count], current[self.last_index:end]
                if (expected.shape != actual.shape or expected.dtype != actual.dtype
                        or expected.device != actual.device or not expected.equal(actual)):
                    raise RuntimeError("FR5_ACK_QUEUE_PREFIX_CHANGED")
            return self.advance_if_current(
                generation=self._generation, expected_index=self.last_index, count=count,
            )
