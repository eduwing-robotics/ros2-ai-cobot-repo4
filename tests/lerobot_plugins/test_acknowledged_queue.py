"""CPU-only checks for the LeRobot ActionQueue acknowledgment seam."""

import unittest
import json
from dataclasses import replace
from unittest.mock import patch

import torch
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot_strategy_fr5.acknowledged_queue import (
    AcknowledgedActionQueue,
    GenerationEligibility,
    ObservationProvenance,
    RawActionIndex,
)


def actions(start: int, rows: int = 4) -> torch.Tensor:
    return torch.arange(start, start + rows * 7, dtype=torch.float32).reshape(rows, 7)


class AcknowledgedActionQueueTest(unittest.TestCase):
    def eligible_queue(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False),
            require_observation_provenance=True, max_observation_age_s=.3)
        clocks = [100., 10.]
        queue._system_clock = lambda: clocks[0]
        queue._steady_clock = lambda: clocks[1]
        return queue, clocks

    @staticmethod
    def generation_input(queue, stamp, digest="a"):
        provenance = ObservationProvenance("sha256:" + digest * 64, "SYSTEM_TIME",
            tuple((name, stamp) for name in ("camera1", "camera2", "state")))
        queue.get_action_index()  # Exact native attempt-start seam.
        queue.observed_input({"state": 1}, provenance)["state"]
        return provenance

    def test_generation_receipt_qualifies_at_actual_merge_completion_not_later_dispatch(self):
        queue, clocks = self.eligible_queue()
        provenance = self.generation_input(queue, 99.95)
        native_merge = ActionQueue.merge

        def finish(*args, **kwargs):
            result = native_merge(*args, **kwargs)
            clocks[:] = [100.15, 10.15]
            return result

        with patch.object(ActionQueue, "merge", finish):
            queue.merge(actions(0, 2), actions(100, 2), real_delay=0)
        snapshot = queue.snapshot()
        receipt = snapshot.generation_eligibility[0]
        self.assertIsInstance(receipt, GenerationEligibility)
        self.assertIs(snapshot.generation_eligibility[1], receipt)
        self.assertEqual(receipt.observation, provenance)
        self.assertEqual((receipt.sampling_started_at_s, receipt.merge_completed_at_s), (100., 100.15))
        wire = receipt.to_dict()
        self.assertEqual(json.loads(json.dumps(wire)), wire)
        self.assertTrue(wire["eligible"])
        self.assertEqual(wire["max_observation_age_s"], .3)
        wire["source_timestamps_s"]["state"] = -1
        self.assertEqual(receipt.to_dict()["source_timestamps_s"]["state"], 99.95)
        clocks[:] = [1000., 1000.]
        self.assertEqual(queue.advance_unchanged_prefix(snapshot, count=2), snapshot.rows)
        self.assertEqual(queue.snapshot().generation_eligibility, ())

    def test_generation_stale_append_rolls_back_native_tensors_cursor_and_receipts(self):
        queue, clocks = self.eligible_queue()
        self.generation_input(queue, 99.95)
        queue.merge(actions(0, 2), actions(100, 2), real_delay=0)
        queue.get()
        before = queue.snapshot()
        clocks[:] = [100.1, 10.1]
        self.generation_input(queue, 100.05, "b")
        clocks[:] = [100.5, 10.5]
        with self.assertRaisesRegex(RuntimeError, "GENERATION_STALE"):
            queue.merge(actions(1000, 2), actions(2000, 2), real_delay=0)
        after = queue.snapshot()
        self.assertEqual((after.generation, after.queue_index, after.rows, after.generation_eligibility),
                         (before.generation, before.queue_index, before.rows, before.generation_eligibility))
        self.assertTrue(torch.equal(after.original_actions, before.original_actions))
        self.assertTrue(torch.equal(after.processed_actions, before.processed_actions))
        self.assertEqual(after.observation_provenance, before.observation_provenance)
        self.generation_input(queue, 100.45, "c")
        queue.merge(actions(3000, 1), actions(4000, 1), real_delay=0)
        appended = queue.snapshot()
        self.assertEqual([r.chunk_id for r in appended.generation_eligibility], [1, 2])
        self.assertEqual(queue.advance_unchanged_prefix(before, count=1), before.rows)
        queue.clear()
        self.assertEqual(queue.snapshot().generation_eligibility, ())

    def test_generation_retry_clears_failed_inference_and_failed_merge_attempt(self):
        queue, clocks = self.eligible_queue()
        self.generation_input(queue, 99.95, "a")
        # A failed inference never called merge. Native get_action_index resets
        # both its source token and sampling clocks before B's feature reads.
        clocks[:] = [100.2, 10.2]
        second = self.generation_input(queue, 100.15, "b")
        clocks[:] = [100.25, 10.25]
        queue.merge(actions(0, 1), actions(100, 1), real_delay=0)
        receipt = queue.snapshot().generation_eligibility[0]
        self.assertEqual(receipt.observation, second)
        self.assertEqual(receipt.sampling_started_at_s, 100.2)
        queue.get_action_index()
        with self.assertRaisesRegex(RuntimeError, "PROVENANCE_MISSING"):
            queue.merge(actions(0, 1), actions(100, 1), real_delay=0)
        self.assertFalse(hasattr(queue._sampled_observation, "started"))
        self.assertFalse(hasattr(queue._sampled_observation, "value"))
        self.assertEqual(queue.generation, 1)

    def test_generation_native_merge_exception_remains_primary_and_does_not_publish_receipt(self):
        queue, _ = self.eligible_queue()
        self.generation_input(queue, 100.)
        queue.merge(actions(0, 2), actions(100, 2), real_delay=0)
        queue.get()
        before = queue.snapshot()
        self.generation_input(queue, 100.)
        with patch.object(queue, "_system_clock", side_effect=RuntimeError("secondary clock")):
            with self.assertRaises(RuntimeError) as raised:
                queue.merge(actions(1000, 2), torch.ones(2, 6), real_delay=0)
        self.assertNotIn("secondary clock", str(raised.exception))
        self.assertNotIn("GENERATION", str(raised.exception))
        after = queue.snapshot()
        self.assertEqual(after.generation_eligibility, before.generation_eligibility)
        self.assertEqual(after.rows, before.rows)
        self.assertTrue(torch.equal(after.original_actions, before.original_actions))
        self.assertTrue(torch.equal(after.processed_actions, before.processed_actions))
        self.assertFalse(hasattr(queue._sampled_observation, "started"))

    def test_generation_configuration_requires_existing_provenance_and_positive_bound(self):
        with self.assertRaisesRegex(ValueError, "GENERATION_CONFIG"):
            AcknowledgedActionQueue(RTCConfig(enabled=False), max_observation_age_s=.3)
        for age in (False, 0., -1., float("nan"), float("inf")):
            with self.subTest(age=age), self.assertRaisesRegex(ValueError, "GENERATION_CONFIG"):
                AcknowledgedActionQueue(RTCConfig(enabled=False),
                    require_observation_provenance=True, max_observation_age_s=age)

    def test_generation_clock_future_source_and_missing_start_fail_closed(self):
        for finish in ((99.99, 10.1), (100.1, 9.99), (100.1, 10.31), (float("nan"), 10.1)):
            queue, clocks = self.eligible_queue()
            self.generation_input(queue, 100.)
            clocks[:] = finish
            with self.subTest(finish=finish), self.assertRaisesRegex(RuntimeError, "GENERATION_(CLOCK|STALE)"):
                queue.merge(actions(0, 1), actions(100, 1), real_delay=0)
            self.assertEqual(queue.generation, 0)
        queue, clocks = self.eligible_queue()
        future = self.generation_input(queue, 100.01)
        clocks[:] = [100.1, 10.1]
        with self.assertRaisesRegex(RuntimeError, "GENERATION_STALE"):
            queue.merge(actions(0, 1), actions(100, 1), real_delay=0)
        queue.observed_input({"state": 1}, future)["state"]
        with self.assertRaisesRegex(RuntimeError, "GENERATION_START_MISSING"):
            queue.merge(actions(0, 1), actions(100, 1), real_delay=0)

    def test_generation_receipt_is_checked_by_atomic_prefix_but_legacy_remains_optional(self):
        queue, _ = self.eligible_queue()
        self.generation_input(queue, 100.)
        queue.merge(actions(0, 1), actions(100, 1), real_delay=0)
        original = queue.snapshot()
        for receipts in ((), (None,)):
            changed = replace(original, generation_eligibility=receipts)
            with self.assertRaisesRegex(RuntimeError, "PREFIX_CHANGED"):
                queue.advance_unchanged_prefix(changed, count=1)
        legacy = AcknowledgedActionQueue(RTCConfig(enabled=False))
        legacy.merge(actions(0, 1), actions(100, 1), real_delay=0)
        snapshot = legacy.snapshot()
        self.assertEqual(snapshot.generation_eligibility, (None,))
        self.assertEqual(legacy.advance_unchanged_prefix(replace(snapshot, generation_eligibility=()), count=1), snapshot.rows)

    def test_observed_input_binds_without_changing_native_tensors(self):
        queue = AcknowledgedActionQueue(
            RTCConfig(enabled=False), require_observation_provenance=True,
        )
        provenance = ObservationProvenance(
            "sha256:" + "a" * 64,
            "SYSTEM_TIME",
            (("camera1", 2.), ("camera2", 3.), ("state", 1.)),
        )
        observed = queue.observed_input({"later": 2, "earlier": 1}, provenance)
        self.assertEqual(observed["later"], 2)
        self.assertEqual(observed["earlier"], 1)
        self.assertEqual(observed["later"], 2)

        original, processed = actions(0, 2), actions(100, 2)
        expected_original, expected_processed = original.clone(), processed.clone()
        queue.merge(original, processed, real_delay=0)
        snapshot = queue.snapshot()
        self.assertEqual(snapshot.observation_provenance, (provenance, provenance))
        self.assertTrue(torch.equal(snapshot.original_actions, expected_original))
        self.assertTrue(torch.equal(snapshot.processed_actions, expected_processed))
        self.assertTrue(torch.equal(original, expected_original))
        self.assertTrue(torch.equal(processed, expected_processed))

        with self.assertRaisesRegex(RuntimeError, "PROVENANCE_MISSING"):
            queue.merge(actions(1000, 1), actions(2000, 1), real_delay=0)
        unchanged = queue.snapshot()
        self.assertEqual(unchanged.generation, snapshot.generation)
        self.assertEqual(unchanged.rows, snapshot.rows)
        self.assertEqual(
            unchanged.observation_provenance, snapshot.observation_provenance,
        )
        self.assertTrue(torch.equal(unchanged.original_actions, snapshot.original_actions))
        self.assertTrue(torch.equal(unchanged.processed_actions, snapshot.processed_actions))

    def test_required_provenance_rejects_rtc_guidance(self):
        with self.assertRaisesRegex(ValueError, "PROVENANCE_REQUIRES_APPEND_ONLY"):
            AcknowledgedActionQueue(
                RTCConfig(enabled=True), require_observation_provenance=True,
            )

    def test_snapshot_is_detached_and_does_not_consume(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        original = actions(0)
        processed = actions(100)
        queue.merge(original, processed, real_delay=0)

        original.fill_(-1)
        processed.fill_(-2)
        snapshot = queue.snapshot()

        self.assertEqual(snapshot.generation, 1)
        self.assertEqual(snapshot.queue_index, 0)
        self.assertEqual(
            snapshot.rows,
            tuple(RawActionIndex(1, index) for index in range(4)),
        )
        self.assertTrue(torch.equal(snapshot.original_actions, actions(0)))
        self.assertTrue(torch.equal(snapshot.processed_actions, actions(100)))
        self.assertEqual(queue.get_action_index(), 0)

        snapshot.original_actions.fill_(-3)
        snapshot.processed_actions.fill_(-4)
        saved = queue.snapshot()
        self.assertTrue(torch.equal(saved.original_actions, actions(0)))
        self.assertTrue(torch.equal(saved.processed_actions, actions(100)))

    def test_conditional_advance_preserves_native_get_behavior(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        processed = actions(100)
        queue.merge(actions(0), processed, real_delay=0)
        snapshot = queue.snapshot()

        advanced = queue.advance_if_current(
            generation=snapshot.generation,
            expected_index=snapshot.queue_index,
            count=2,
        )
        self.assertEqual(advanced, (RawActionIndex(1, 0), RawActionIndex(1, 1)))
        self.assertEqual(queue.qsize(), 2)

        returned = queue.get()
        self.assertTrue(torch.equal(returned, processed[2]))
        returned.fill_(-1)
        self.assertTrue(torch.equal(queue.get_processed_left_over(), processed[3:]))

        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_CURSOR_MOVED"):
            queue.advance_if_current(
                generation=snapshot.generation,
                expected_index=snapshot.queue_index,
                count=1,
            )

    def test_append_invalidates_snapshot_and_retains_raw_row_identity(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        first_raw = actions(0)
        first_processed = actions(100)
        queue.merge(first_raw, first_processed, real_delay=0)
        stale = queue.snapshot()
        queue.advance_if_current(generation=1, expected_index=0, count=2)

        second_raw = actions(1000, rows=2)
        second_processed = actions(2000, rows=2)
        queue.merge(second_raw, second_processed, real_delay=0)

        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_STALE_GENERATION"):
            queue.advance_if_current(
                generation=stale.generation,
                expected_index=stale.queue_index,
                count=1,
            )

        current = queue.snapshot()
        self.assertEqual(current.generation, 2)
        self.assertEqual(current.queue_index, 0)
        self.assertEqual(
            current.rows,
            (
                RawActionIndex(1, 2),
                RawActionIndex(1, 3),
                RawActionIndex(2, 0),
                RawActionIndex(2, 1),
            ),
        )
        self.assertTrue(
            torch.equal(current.original_actions, torch.cat([first_raw[2:], second_raw]))
        )
        self.assertTrue(
            torch.equal(
                current.processed_actions,
                torch.cat([first_processed[2:], second_processed]),
            )
        )

    def test_unchanged_prefix_survives_native_append_and_compaction_not_replay(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        queue.get()
        queue.get()
        snapshot = queue.snapshot()
        queue.merge(actions(1000, 2), actions(2000, 2), real_delay=0)
        self.assertNotEqual(snapshot.generation, queue.generation)
        self.assertNotEqual(snapshot.queue_index, queue.get_action_index())
        advanced = queue.advance_unchanged_prefix(snapshot, count=1)
        self.assertEqual(advanced, (RawActionIndex(1, 2),))
        self.assertEqual(queue.qsize(), 3)
        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_PREFIX_CHANGED"):
            queue.advance_unchanged_prefix(snapshot, count=1)
        self.assertTrue(torch.equal(queue.get(), actions(100)[3]))
        self.assertTrue(torch.equal(queue.get_processed_left_over(), actions(2000, 2)))

    def test_unchanged_prefix_checks_identity_provenance_and_both_tensor_representations(self):
        provenance = ObservationProvenance("sha256:" + "a" * 64, "SYSTEM_TIME",
            (("camera1", 1.), ("camera2", 1.), ("state", 1.)))
        for change in ("raw", "processed", "dtype", "identity", "provenance", "clear", "cursor", "rtc"):
            with self.subTest(change=change):
                queue = AcknowledgedActionQueue(RTCConfig(enabled=change == "rtc"))
                queue.merge(actions(0), actions(100), real_delay=0)
                snapshot = queue.snapshot()
                if change == "raw":
                    snapshot.original_actions[0, 0] += 1
                elif change == "processed":
                    snapshot.processed_actions[0, 0] += 1
                elif change == "dtype":
                    snapshot = replace(snapshot, processed_actions=snapshot.processed_actions.double())
                elif change == "identity":
                    snapshot = replace(snapshot, rows=(RawActionIndex(99, 0), *snapshot.rows[1:]))
                elif change == "provenance":
                    snapshot = replace(snapshot, observation_provenance=(provenance, *snapshot.observation_provenance[1:]))
                elif change == "clear":
                    queue.clear()
                    queue.merge(actions(0), actions(100), real_delay=0)
                elif change == "cursor":
                    queue.get()
                else:
                    queue.merge(actions(0), actions(100), real_delay=0)
                before = queue.get_action_index()
                with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_PREFIX_CHANGED"):
                    queue.advance_unchanged_prefix(snapshot, count=1)
                self.assertEqual(queue.get_action_index(), before)

    def test_unchanged_prefix_does_not_claim_or_consume_a_changed_tail(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        snapshot = queue.snapshot()
        snapshot.original_actions[2:] = -1
        snapshot.processed_actions[2:] = -2
        self.assertEqual(queue.advance_unchanged_prefix(snapshot, count=2), snapshot.rows[:2])
        self.assertTrue(torch.equal(queue.get_processed_left_over(), actions(100)[2:]))

    def test_current_prefix_check_survives_append_without_advancing(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        queue.get()
        snapshot = queue.snapshot()

        # Native append compacts the consumed head while retaining the exact
        # detached prefix. Membership remains true without moving the cursor.
        compared, attempted, release = Event(), Event(), Event()
        compare = queue._check_unchanged_prefix_locked

        def hold_after_comparison(*args, **kwargs):
            result = compare(*args, **kwargs)
            compared.set()
            if not release.wait(3):
                raise TimeoutError("test did not release current-prefix check")
            return result

        def append():
            if not compared.wait(3):
                raise TimeoutError("prefix was never compared")
            acquired = queue.lock.acquire(blocking=False)
            if acquired:
                queue.lock.release()
            attempted.set()
            queue.merge(actions(1000, 2), actions(2000, 2), real_delay=0)
            return acquired

        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(
                queue, "_check_unchanged_prefix_locked", side_effect=hold_after_comparison):
            consumer = pool.submit(queue.check_unchanged_prefix, snapshot, count=2)
            producer = pool.submit(append)
            try:
                self.assertTrue(attempted.wait(3))
            finally:
                release.set()
            self.assertEqual(consumer.result(timeout=3), snapshot.rows[:2])
            self.assertFalse(producer.result(timeout=3))

        before_check = queue.get_action_index()
        self.assertEqual(
            queue.check_unchanged_prefix(snapshot, count=2),
            snapshot.rows[:2],
        )
        self.assertEqual(queue.get_action_index(), before_check)
        self.assertEqual(queue.qsize(), 5)

        queue.get()
        moved = queue.get_action_index()
        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_PREFIX_CHANGED"):
            queue.check_unchanged_prefix(snapshot, count=2)
        self.assertEqual(queue.get_action_index(), moved)

        for count in (False, 0, -1, len(snapshot.rows) + 1):
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                queue.check_unchanged_prefix(snapshot, count=count)
            self.assertEqual(queue.get_action_index(), moved)

    def test_cumulative_progress_reuses_bound_snapshot_with_explicit_offset(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        snapshot = queue.snapshot()
        self.assertEqual(queue.advance_unchanged_prefix(snapshot, count=2), snapshot.rows[:2])
        queue.merge(actions(1000, 2), actions(2000, 2), real_delay=0)
        self.assertEqual(queue.advance_unchanged_prefix(snapshot, offset=2, count=1), snapshot.rows[2:3])
        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_PREFIX_CHANGED"):
            queue.advance_unchanged_prefix(snapshot, offset=2, count=1)
        self.assertEqual(queue.advance_unchanged_prefix(snapshot, offset=3, count=1), snapshot.rows[3:])
        self.assertEqual(queue.snapshot().rows, (RawActionIndex(2, 0), RawActionIndex(2, 1)))

    def test_unchanged_prefix_comparison_and_advance_share_native_lock(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        snapshot = queue.snapshot()
        compared, attempted, release = Event(), Event(), Event()
        advance = queue.advance_if_current
        def after_comparison(**kwargs):
            compared.set()
            if not release.wait(3):
                raise TimeoutError("test did not release prefix advancement")
            return advance(**kwargs)
        def append():
            if not compared.wait(3):
                raise TimeoutError("prefix was never compared")
            acquired = queue.lock.acquire(blocking=False)
            if acquired:
                queue.lock.release()
            attempted.set()
            queue.merge(actions(1000, 2), actions(2000, 2), real_delay=0)
            return acquired
        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(
                queue, "advance_if_current", side_effect=after_comparison):
            consumer = pool.submit(queue.advance_unchanged_prefix, snapshot, count=2)
            producer = pool.submit(append)
            try:
                self.assertTrue(attempted.wait(3))
            finally:
                release.set()
            self.assertEqual(consumer.result(timeout=3), snapshot.rows[:2])
            self.assertFalse(producer.result(timeout=3))
        self.assertEqual(queue.snapshot().rows,
                         (RawActionIndex(1, 2), RawActionIndex(1, 3), RawActionIndex(2, 0), RawActionIndex(2, 1)))

    def test_unchanged_prefix_invalid_counts_do_not_move_native_cursor(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        snapshot = queue.snapshot()
        for count in (True, 1., "1", None, -1, 5):
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                queue.advance_unchanged_prefix(snapshot, count=count)
            self.assertEqual(queue.get_action_index(), 0)
        self.assertEqual(queue.advance_unchanged_prefix(snapshot, count=0), ())
        self.assertEqual(queue.get_action_index(), 0)
        for offset in (True, 1., "1", None, -1, 4):
            with self.subTest(offset=offset), self.assertRaises((TypeError, ValueError)):
                queue.advance_unchanged_prefix(snapshot, offset=offset, count=1)
            self.assertEqual(queue.get_action_index(), 0)

    def test_native_rtc_replace_delay_and_staleness_are_unchanged(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=True))
        first = actions(0)
        queue.merge(first, first + 100, real_delay=1)
        stale = queue.snapshot()
        self.assertEqual(
            stale.rows,
            tuple(RawActionIndex(1, index) for index in range(1, 4)),
        )

        second = actions(1000)
        queue.merge(second, second + 100, real_delay=2)
        current = queue.snapshot()
        self.assertEqual(current.generation, 2)
        self.assertEqual(
            current.rows,
            (RawActionIndex(2, 2), RawActionIndex(2, 3)),
        )
        self.assertTrue(torch.equal(current.original_actions, second[2:]))
        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_STALE_GENERATION"):
            queue.advance_if_current(
                generation=stale.generation,
                expected_index=stale.queue_index,
                count=1,
            )

    def test_failed_native_append_rolls_back_queue_and_metadata(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        original = actions(0, rows=2)
        processed = actions(100, rows=2)
        queue.merge(original, processed, real_delay=0)
        self.assertTrue(torch.equal(queue.get(), processed[0]))
        before = queue.snapshot()

        with self.assertRaises(RuntimeError):
            queue.merge(
                torch.full((2, 7), 999.0),
                torch.full((2, 6), 999.0),
                real_delay=0,
            )

        after = queue.snapshot()
        self.assertEqual(after.generation, before.generation)
        self.assertEqual(after.queue_index, before.queue_index)
        self.assertEqual(after.rows, before.rows)
        self.assertTrue(torch.equal(after.original_actions, before.original_actions))
        self.assertTrue(torch.equal(after.processed_actions, before.processed_actions))
        self.assertEqual(
            queue.advance_if_current(
                generation=before.generation,
                expected_index=before.queue_index,
                count=1,
            ),
            (RawActionIndex(1, 1),),
        )

    def test_clear_invalidates_snapshot_and_keeps_chunk_ids_monotonic(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0), actions(100), real_delay=0)
        stale = queue.snapshot()
        queue.clear()

        self.assertEqual(queue.generation, 2)
        self.assertTrue(queue.empty())
        self.assertIsNone(queue.snapshot().original_actions)
        with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_STALE_GENERATION"):
            queue.advance_if_current(
                generation=stale.generation,
                expected_index=stale.queue_index,
                count=0,
            )

        queue.merge(actions(1000, 1), actions(2000, 1), real_delay=0)
        self.assertEqual(queue.snapshot().rows, (RawActionIndex(2, 0),))

    def test_invalid_advance_does_not_move_cursor(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        queue.merge(actions(0, 2), actions(100, 2), real_delay=0)
        snapshot = queue.snapshot()

        for count in (-1, 3):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "FR5_ACK_QUEUE_INVALID_COUNT"):
                    queue.advance_if_current(
                        generation=snapshot.generation,
                        expected_index=snapshot.queue_index,
                        count=count,
                    )
                self.assertEqual(queue.get_action_index(), 0)

    def test_non_integer_advance_inputs_leave_queue_unchanged(self):
        queue = AcknowledgedActionQueue(RTCConfig(enabled=False))
        processed = actions(100, 2)
        queue.merge(actions(0, 2), processed, real_delay=0)
        before = queue.snapshot()

        valid = {
            "generation": before.generation,
            "expected_index": before.queue_index,
            "count": 1,
        }
        for name in valid:
            for invalid in (0.5, float("nan"), True):
                with self.subTest(name=name, invalid=invalid):
                    arguments = {**valid, name: invalid}
                    with self.assertRaisesRegex(
                        TypeError, "FR5_ACK_QUEUE_EXPECTED_INTEGER"
                    ):
                        queue.advance_if_current(**arguments)

        after = queue.snapshot()
        self.assertEqual(after.generation, before.generation)
        self.assertEqual(after.queue_index, before.queue_index)
        self.assertEqual(after.rows, before.rows)
        self.assertTrue(torch.equal(after.original_actions, before.original_actions))
        self.assertTrue(torch.equal(after.processed_actions, before.processed_actions))
        self.assertTrue(torch.equal(queue.get(), processed[0]))

    def test_base_merge_and_clear_results_match_native_queue(self):
        native = ActionQueue(RTCConfig(enabled=False))
        adapted = AcknowledgedActionQueue(RTCConfig(enabled=False))

        for raw, processed in (
            (actions(0), actions(100)),
            (actions(1000, 2), actions(2000, 2)),
        ):
            native.merge(raw, processed, 0)
            adapted.merge(raw, processed, 0)
            self.assertTrue(torch.equal(native.get(), adapted.get()))
            self.assertEqual(native.get_action_index(), adapted.get_action_index())
            self.assertTrue(torch.equal(native.get_left_over(), adapted.get_left_over()))
            self.assertTrue(
                torch.equal(native.get_processed_left_over(), adapted.get_processed_left_over())
            )

        native.clear()
        adapted.clear()
        self.assertEqual(native.qsize(), adapted.qsize())
        self.assertIsNone(native.get())
        self.assertIsNone(adapted.get())

    def test_rejects_unaudited_lerobot_version(self):
        with patch(
            "lerobot_strategy_fr5.acknowledged_queue.version",
            return_value="0.6.2",
        ):
            with self.assertRaisesRegex(RuntimeError, "FR5_ACK_QUEUE_UNAUDITED: 0.6.2"):
                AcknowledgedActionQueue(RTCConfig(enabled=False))


if __name__ == "__main__":
    unittest.main()
