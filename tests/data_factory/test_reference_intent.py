"""Pure prospective reference masks; no native nodes or actuator dependencies."""
import copy
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from tools.data_factory.motion.reference_intent import compute_reference_intent
from tools.fr5_data_factory import ContractError, canonical_digest


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def pose(x=0., y=0., z=0., rotation=None):
    return {"translation_m": [x, y, z], "rotation_columns": copy.deepcopy(rotation or IDENTITY)}


def context():
    # Actual qualified jaw/object/profile dimensions, with synthetic datum poses.
    config = ROOT / "config/data_factory"
    grasp = json.loads((config / "grasps/wood-cube-24mm-top-3p5mm-r001.json").read_text())
    obj = json.loads((config / "objects/wood-cube-24mm-r001.json").read_text())
    xml = ET.parse(ROOT / "src/fairino_description/urdf/fairino5_v6_gripper_opening_r001.urdf").getroot()
    boxes = []
    for side in ("right", "left"):
        joint = xml.find(f"joint[@name='finger_{side}_joint']")
        collision = xml.find(f"link[@name='finger_tip_{side}_link']/collision")
        origin = [float(v) for v in joint.find("origin").get("xyz").split()]
        offset = [float(v) for v in collision.find("origin").get("xyz").split()]
        boxes.append({"center": [a + b for a, b in zip(origin, offset)],
                      "dimensions": [float(v) for v in collision.find("geometry/box").get("size").split()]})
    right, left = boxes
    mid = (right["center"][0] + left["center"][0]) / 2
    gap = (right["center"][0] - left["center"][0]
           - (right["dimensions"][0] + left["dimensions"][0]) / 2
           + 2 * max(grasp["gripper_close"]["acceptable_feedback_m"].values()))
    result = {"schema_version": "data_factory.request_contact_geometry.v1", "status": "PROSPECTIVE",
        "physical_success": False, "plan_digest": canonical_digest("synthetic-plan"),
        "source_datum": pose(mid, 0., right["center"][2]),
        "released_datum": pose(mid + .3, 0., right["center"][2]),
        "source_dimensions_m": [v / 1000 for v in obj["dimensions_mm"]],
        "fingertip_boxes": boxes, "open_m": grasp["gripper_open"]["command_position_m"],
        "release_m": grasp["gripper_open"]["release_position_m"],
        "jaw_midplane_m": mid, "closed_gap_bound_m": gap}
    result["geometry_digest"] = canonical_digest(result)
    return result


class ReferenceIntentTest(unittest.TestCase):
    def setUp(self):
        self.context = context()

    def compute(self, samples, **kwargs):
        return compute_reference_intent(self.context, samples, **kwargs)

    def test_open_approach_does_not_add_phantom_carried_or_released_objects(self):
        result = self.compute(((pose(-.1), .021), (pose(), .021)))
        self.assertEqual(result["assignments"], (("source", (0, 1)),))
        self.assertIsNone(result["carried_envelope"])
        self.assertFalse(result["intent"]["released_possible"])
        self.assertTrue(result["evidence"]["source_world_retained"])

    def test_closure_can_begin_while_arm_moves_without_stationary_completion(self):
        result = self.compute(((pose(-.05), .021), (pose(-.05), .01176),
                               (pose(), .01176), (pose(.15), .01176)))
        self.assertEqual(dict(result["assignments"])["carried"], (2, 3))
        self.assertEqual(dict(result["assignments"])["source"], (0, 1))
        self.assertEqual(result["evidence"]["candidate_relations"][0]["sample_index"], 2)
        self.assertFalse(result["evidence"]["physical_success"])

    def test_fixed_pose_closing_sweep_includes_interval_entry(self):
        result = self.compute(((pose(), .021), (pose(), .01176), (pose(.1), .01176)))
        self.assertEqual(result["assignments"], (("carried", (0, 1, 2)),))
        self.assertTrue(result["evidence"]["candidate_relations"][0]["jaw_sweep"])

    def test_off_nominal_capture_retains_actual_relation_not_centered_nominal_box(self):
        result = self.compute(((pose(y=.014), .021), (pose(y=.014), .01176)))
        box = result["carried_envelope"]
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box["translation_m"][1], -.014)
        self.assertAlmostEqual(box["dimensions_m"][1], .024)
        self.assertAlmostEqual(result["evidence"]["candidate_relations"][0]["source_to_gripper"]["translation_m"][1], -.014)

    def test_uncertain_opening_does_not_remove_carried_support(self):
        result = self.compute(((pose(), .021), (pose(), .01176),
                               (pose(.3), .01176), (pose(.3), .0126), (pose(.3, z=.1), .021)))
        self.assertEqual(dict(result["assignments"])["carried"], (0, 1, 2, 3, 4))
        self.assertEqual(dict(result["assignments"])["released"], (3, 4))
        self.assertFalse(result["evidence"]["release_confirmed"])
        later = self.compute(((pose(.4, z=.2), .021),), prior=result["intent"])
        self.assertEqual(later["assignments"], (("carried", (0,)), ("released", (0,))))
        self.assertEqual(later["carried_envelope"], result["carried_envelope"])

    def test_opening_elsewhere_and_unrelated_open_destination_do_not_claim_release(self):
        result = self.compute(((pose(), .01176), (pose(.15), .021)))
        self.assertEqual(result["assignments"], (("carried", (0, 1)),))
        self.assertFalse(result["intent"]["released_possible"])
        empty = self.compute(((pose(.3), .021),))
        self.assertEqual(empty["assignments"], (("source", (0,)),))

    def test_existing_percent_projection_identifies_release_intent_not_raw_equality(self):
        result = self.compute(((pose(), .01176), (pose(.3), .01259)))
        self.assertEqual(dict(result["assignments"])["released"], (1,))

    def test_prior_envelope_union_never_shrinks_and_inputs_are_detached(self):
        first = self.compute(((pose(y=.014), .01176),))
        previous = json.loads(json.dumps(first["intent"]))
        samples = ((pose(y=-.014), .01176),)
        before = copy.deepcopy((self.context, previous, samples))
        result = self.compute(samples, prior=previous)
        box = result["carried_envelope"]
        self.assertAlmostEqual(box["translation_m"][1], 0.)
        self.assertAlmostEqual(box["dimensions_m"][1], .052)
        self.assertEqual((self.context, previous, samples), before)
        result["carried_envelope"]["translation_m"][0] = 999
        self.assertNotEqual(result["intent"]["carried_envelope"]["translation_m"][0], 999)
        self.assertIsInstance(json.dumps(result), str)

    def test_rotated_off_nominal_source_is_enclosed_in_gripper_frame(self):
        rotation = [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]
        result = self.compute(((pose(rotation=rotation), .01176),))
        candidate = result["evidence"]["candidate_relations"][0]["source_to_gripper"]
        box = result["carried_envelope"]
        self.assertEqual(box["rotation_columns"], IDENTITY)
        self.assertNotEqual(candidate["rotation_columns"], IDENTITY)
        for index in range(3):
            self.assertLessEqual(box["translation_m"][index] - box["dimensions_m"][index] / 2,
                                 candidate["translation_m"][index] - .012 + 1e-9)
            self.assertGreaterEqual(box["translation_m"][index] + box["dimensions_m"][index] / 2,
                                    candidate["translation_m"][index] + .012 - 1e-9)

    def test_exact_prior_support_is_not_lost_to_rounding_or_speculative_tail_bridge(self):
        first = self.compute(((pose(x=.00213457891, y=.01112347685), .01176),))
        previous = first["intent"]
        for y in (-.00998651, .005854953, .013213742):
            result = self.compute(((pose(y=y), .01176),), prior=previous)
            old, new = previous["carried_envelope"], result["carried_envelope"]
            for axis in range(3):
                self.assertLessEqual(new["translation_m"][axis] - new["dimensions_m"][axis] / 2,
                                     old["translation_m"][axis] - old["dimensions_m"][axis] / 2)
                self.assertGreaterEqual(new["translation_m"][axis] + new["dimensions_m"][axis] / 2,
                                        old["translation_m"][axis] + old["dimensions_m"][axis] / 2)
            previous = result["intent"]
        empty = self.compute(((pose(), .021), (pose(.2), .021)))
        # Last candidate endpoint is not used as an imaginary closing sweep
        # onto the next actual anchor. Only retained occupancy crosses revisions.
        successor = self.compute(((pose(.4), .01176),), prior=empty["intent"])
        self.assertIsNone(successor["carried_envelope"])

    def test_broad_phase_candidate_does_not_authorize_top_or_body_contact(self):
        result = self.compute(((pose(z=-.015), .01176),))
        self.assertIsNotNone(result["carried_envelope"])
        self.assertFalse(result["evidence"]["contact_authorized"])
        self.assertFalse(result["evidence"]["continuous_motion_proof"])

    def test_malformed_and_foreign_prior_cannot_be_rebound(self):
        valid = ((pose(), .01176),)
        prior = self.compute(valid)["intent"]
        variants = [copy.deepcopy(prior) for _ in range(3)]
        variants[0]["geometry_digest"] = canonical_digest("other")
        variants[0]["intent_digest"] = canonical_digest({k: v for k, v in variants[0].items() if k != "intent_digest"})
        variants[1]["carried_envelope"]["dimensions_m"][0] = -1.
        variants[2]["unexpected"] = 1
        for value in variants:
            with self.subTest(prior=value), self.assertRaisesRegex(ContractError, "CONTACT_REFERENCE_INTENT"):
                self.compute(valid, prior=value)
        for value in ([], (), ((pose(), float("nan")),), ((pose(), True),), ((pose(), -.001),),
                      (({**pose(), "extra": 1}, .01176),)):
            with self.subTest(samples=value), self.assertRaisesRegex(ContractError, "CONTACT_REFERENCE_INTENT"):
                self.compute(value)


if __name__ == "__main__":
    unittest.main()
