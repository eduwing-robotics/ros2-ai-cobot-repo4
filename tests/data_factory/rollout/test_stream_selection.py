import copy
import json
import subprocess
import sys
import unittest
from dataclasses import replace

import torch
from lerobot.policies.rtc import RTCConfig
from lerobot_strategy_fr5.acknowledged_queue import (
    AcknowledgedActionQueue,
    ObservationProvenance,
    RawActionIndex,
)

from tools.data_factory.rollout.execution_state import JOINTS
from tools.data_factory.rollout.stream_selection import (
    acknowledge_stream_selection,
    select_stream_revision,
    validate_stream_selection,
)
from tools.fr5_data_factory import ContractError, canonical_digest


ROBOT = '<robot name="test">' + ''.join(
    f'<joint name="{name}" type="{"prismatic" if index == 6 else "revolute"}">'
    f'<limit lower="{0 if index == 6 else -3}" upper="{.02 if index == 6 else 3}" '
    f'velocity="{.1 if index == 6 else 10}"/></joint>'
    for index, name in enumerate(JOINTS)
) + '</robot>'


def provenance(stamp=1.):
    return ObservationProvenance(
        canonical_digest({"observation": stamp}), "SYSTEM_TIME",
        (("camera1", stamp), ("camera2", stamp + .01), ("state", stamp + .02)),
    )


def merge(queue, raw, processed, source=None):
    source = provenance() if source is None else source
    queue.observed_input({"state": 1}, source)["state"]
    queue.merge(raw, processed, real_delay=0)


class StreamSelectionTest(unittest.TestCase):
    def setUp(self):
        self.queue = AcknowledgedActionQueue(
            RTCConfig(enabled=False), require_observation_provenance=True,
        )

    def test_json_validator_import_does_not_load_torch_or_lerobot(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; import tools.data_factory.rollout.stream_selection; "
                "assert 'torch' not in sys.modules; "
                "assert not any(name == 'lerobot' or name.startswith('lerobot.') "
                "for name in sys.modules)"
            )],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selects_remaining_first_native_chunk_and_projects_controller_rows(self):
        raw = torch.tensor([[99.] * 7, [100.] * 7, [101.] * 7])
        processed = torch.tensor([
            [.0] * 6 + [.01], [.1] * 6 + [-.0003], [.2] * 6 + [.01991],
        ])
        merge(self.queue, raw, processed)
        initial = self.queue.snapshot()
        self.queue.advance_unchanged_prefix(initial, count=1)
        merge(self.queue, torch.tensor([[200.] * 7, [201.] * 7]),
              torch.tensor([[.3] * 6 + [.01], [.4] * 6 + [.01]]), provenance(2.))
        snapshot = self.queue.snapshot()
        before = self.queue.qsize()

        selected = select_stream_revision(snapshot, robot_description=ROBOT)

        self.assertEqual((selected["source_chunk"], selected["source_row_indices"]), (1, [1, 2]))
        self.assertEqual(selected["raw_actions"], raw[1:].tolist())
        self.assertEqual(selected["processed_actions"], processed[1:].tolist())
        self.assertEqual(selected["controller_actions"][0][:-1], selected["processed_actions"][0][:-1])
        self.assertEqual(selected["controller_actions"][1][:-1], selected["processed_actions"][1][:-1])
        self.assertEqual([row[-1] for row in selected["controller_actions"]], [0., .02])
        self.assertEqual(selected["gripper_projection"], {"upper_m": .02, "projection_quanta": 2})
        self.assertEqual(select_stream_revision(
            snapshot, robot_description=ROBOT, row_count=1,
        )["source_row_indices"], [1])
        self.assertEqual(self.queue.qsize(), before)
        json.dumps(selected, allow_nan=False)
        checked = validate_stream_selection(selected, robot_description=ROBOT)
        checked["controller_actions"][0][0] = 9.
        self.assertNotEqual(checked, selected)

    def test_snapshot_and_json_counterexamples_fail_before_queue_mutation(self):
        raw = torch.tensor([[99.] * 7, [100.] * 7])
        processed = torch.tensor([[.1] * 6 + [.01], [.2] * 6 + [.019]])
        merge(self.queue, raw, processed)
        snapshot = self.queue.snapshot()
        selected = select_stream_revision(snapshot, robot_description=ROBOT)
        before = self.queue.qsize()

        bad_snapshots = (
            replace(snapshot, rows=()),
            replace(snapshot, rows=(RawActionIndex(1, 0), RawActionIndex(1, 2))),
            replace(snapshot, observation_provenance=(None, None)),
            replace(snapshot, processed_actions=torch.zeros(2, 6)),
        )
        for bad in bad_snapshots:
            with self.subTest(snapshot=bad.rows), self.assertRaises(ContractError):
                select_stream_revision(bad, robot_description=ROBOT)
        for count in (False, 0, 3):
            with self.subTest(count=count), self.assertRaises(ContractError):
                select_stream_revision(snapshot, robot_description=ROBOT, row_count=count)

        def redigest(value):
            value["selection_digest"] = canonical_digest({
                key: item for key, item in value.items() if key != "selection_digest"
            })
            return value

        mutations = []
        changed = copy.deepcopy(selected)
        changed["controller_actions"][0][0] += .1
        mutations.append(redigest(changed))
        changed = copy.deepcopy(selected)
        changed["processed_actions"][0][0] = 4.
        changed["controller_actions"][0][0] = 4.
        mutations.append(redigest(changed))
        changed = copy.deepcopy(selected)
        changed["processed_actions"][0][-1] = .03
        changed["controller_actions"][0][-1] = .02
        mutations.append(redigest(changed))
        changed = copy.deepcopy(selected)
        changed["source_row_indices"][1] = 3
        mutations.append(redigest(changed))
        changed = copy.deepcopy(selected)
        changed["gripper_projection"]["projection_quanta"] = 3
        mutations.append(redigest(changed))
        changed = copy.deepcopy(selected)
        changed["source_observation"]["source_clock"] = "OTHER_TIME"
        mutations.append(redigest(changed))
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                validate_stream_selection(changed, robot_description=ROBOT)
        changed = copy.deepcopy(selected)
        changed["processed_actions"][0][0] += .1  # Stale digest is independently rejected.
        with self.assertRaisesRegex(ContractError, "SELECTION_DIGEST"):
            validate_stream_selection(changed, robot_description=ROBOT)
        with self.assertRaisesRegex(ContractError, "SELECTION_MODEL"):
            validate_stream_selection(
                selected, robot_description=ROBOT.replace('upper="0.02"', 'upper="0.021"'),
            )
        self.assertEqual(self.queue.qsize(), before)

    def test_acknowledges_only_new_cumulative_progress_on_exact_snapshot(self):
        merge(self.queue, torch.arange(21, dtype=torch.float32).reshape(3, 7),
              torch.tensor([[.1] * 6 + [.001], [.2] * 6 + [.01], [.3] * 6 + [.019]]))
        snapshot = self.queue.snapshot()
        selected = select_stream_revision(snapshot, robot_description=ROBOT)
        merge(self.queue, torch.ones(1, 7), torch.tensor([[.4] * 6 + [.01]]), provenance(2.))

        self.assertEqual(acknowledge_stream_selection(
            self.queue, snapshot, selected, selected_count=1,
            robot_description=ROBOT), 1)
        self.assertEqual(self.queue.qsize(), 3)
        self.assertEqual(acknowledge_stream_selection(
            self.queue, snapshot, selected, selected_count=3, acknowledged_count=1,
            robot_description=ROBOT), 3)
        self.assertEqual(self.queue.snapshot().rows, (RawActionIndex(2, 0),))
        self.assertEqual(acknowledge_stream_selection(
            self.queue, snapshot, selected, selected_count=3, acknowledged_count=3,
            robot_description=ROBOT), 3)

        before = self.queue.qsize()
        with self.assertRaises(RuntimeError):
            acknowledge_stream_selection(
                self.queue, snapshot, selected, selected_count=3, acknowledged_count=1,
                robot_description=ROBOT)
        changed = copy.deepcopy(selected)
        changed["source_chunk"] = 2
        changed["selection_digest"] = canonical_digest({
            key: item for key, item in changed.items() if key != "selection_digest"
        })
        with self.assertRaisesRegex(ContractError, "SELECTION_SNAPSHOT"):
            acknowledge_stream_selection(
                self.queue, snapshot, changed, selected_count=0,
                robot_description=ROBOT)
        with self.assertRaisesRegex(ContractError, "SELECTION_PROGRESS"):
            acknowledge_stream_selection(
                self.queue, snapshot, selected, selected_count=4,
                robot_description=ROBOT)
        self.assertEqual(self.queue.qsize(), before)


if __name__ == "__main__":
    unittest.main()
