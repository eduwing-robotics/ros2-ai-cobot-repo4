"""Controller-paced acknowledgment for LeRobot's native action queue.

The adapter adds an atomic snapshot/conditional-advance seam.  It does not
interpret action-goal acceptance or terminal state as consumption: callers
must advance only from commanded-reference progress for the complete action
row.  In particular, ARM-only progress is not acknowledgment of a 7D row.
"""

from dataclasses import dataclass
from importlib.metadata import version
from threading import RLock

from torch import Tensor

from lerobot.policies.rtc import ActionQueue, RTCConfig


LEROBOT_VERSION = "0.6.1"


def start_with_acknowledged_queue(engine):
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
    queue = AcknowledgedActionQueue(engine._rtc_config)
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
class ActionQueueSnapshot:
    """Detached view of the currently unconsumed native queue."""

    generation: int
    queue_index: int
    rows: tuple[RawActionIndex, ...]
    original_actions: Tensor | None
    processed_actions: Tensor | None


class AcknowledgedActionQueue(ActionQueue):
    """LeRobot 0.6.1 ``ActionQueue`` with compare-and-advance support.

    Native ``get``, ``clear`` and ``merge`` behavior is retained.  ``merge``
    and ``clear`` additionally invalidate old snapshots; row identities remain
    stable across the native non-RTC append/compaction operation.

    RTC-enabled replacement is also observable, but its native delay policy is
    deliberately unchanged.  This class does not make RTC controller-paced.
    """

    def __init__(self, cfg: RTCConfig):
        installed = version("lerobot")
        if installed != LEROBOT_VERSION:
            raise RuntimeError(f"FR5_ACK_QUEUE_UNAUDITED: {installed}")

        super().__init__(cfg)
        # Native methods acquire ``self.lock`` internally.  Reentrancy lets the
        # adapter compose metadata updates with those methods atomically rather
        # than copying their merge/clear algorithms.
        self.lock = RLock()
        self._generation = 0
        self._next_chunk = 0
        self._rows: tuple[RawActionIndex, ...] = ()

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
            saved_queue = self.queue
            saved_original_queue = self.original_queue
            saved_last_index = self.last_index
            retained_rows = self._rows[self.last_index :]
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
                else:
                    rows = retained_rows + tuple(
                        RawActionIndex(chunk, i) for i in range(len(processed_actions))
                    )

                queued = 0 if self.queue is None else len(self.queue)
                if len(rows) != queued:
                    raise RuntimeError("FR5_ACK_QUEUE_NATIVE_LAYOUT_CHANGED")
            except BaseException:
                self.queue = saved_queue
                self.original_queue = saved_original_queue
                self.last_index = saved_last_index
                raise

            self._rows = rows
            self._next_chunk = chunk
            self._generation += 1
            return result

    def clear(self) -> None:
        with self.lock:
            super().clear()
            self._rows = ()
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
