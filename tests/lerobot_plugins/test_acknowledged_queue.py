"""CPU-only checks for the LeRobot ActionQueue acknowledgment seam."""

import unittest
from unittest.mock import patch

import torch
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot_strategy_fr5.acknowledged_queue import (
    AcknowledgedActionQueue,
    ObservationProvenance,
    RawActionIndex,
)


def actions(start: int, rows: int = 4) -> torch.Tensor:
    return torch.arange(start, start + rows * 7, dtype=torch.float32).reshape(rows, 7)


class AcknowledgedActionQueueTest(unittest.TestCase):
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
