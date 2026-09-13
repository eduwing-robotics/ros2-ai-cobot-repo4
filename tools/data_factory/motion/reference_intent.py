"""Prospective occupancy masks over sampled native references, not task phases.

No contact is authorized here. Broad-phase overlap deliberately retains possible
occupancy; the native whole-scene query and existing contact predicate still
decide geometry admission. No result establishes grasp, release, or Scene truth.
"""
import copy
import math

from tools.fr5_data_factory import (
    ContractError, canonical_digest, compose_rigid_transform,
    inverse_rigid_transform, validate_rigid_transform,
)
from tools.data_factory.rollout.action_projection import project_gripper_position
from .contact_transition import lateral_envelope


_CODE = "CONTACT_REFERENCE_INTENT"
_SCHEMA = "data_factory.prospective_reference_intent.v1"
_IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError(_CODE)
    return value


def _vector(value, *, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError(_CODE)
    result = [_number(v) for v in value]
    if positive and any(v <= 0 for v in result):
        raise ContractError(_CODE)
    return result


def _bounds(transform, dimensions):
    half = [sum(abs(transform["rotation_columns"][j][i]) * dimensions[j] / 2
                for j in range(3)) for i in range(3)]
    return ([v - h for v, h in zip(transform["translation_m"], half)],
            [v + h for v, h in zip(transform["translation_m"], half)])


def _union(a, b):
    return ([min(x, y) for x, y in zip(a[0], b[0])],
            [max(x, y) for x, y in zip(a[1], b[1])])


def _overlap(a, b):
    # Same numerical convention as the native model/contact consumers, not an
    # extra physical clearance or successful-capture tolerance.
    return all(lo <= hi + 1e-9 for lo, hi in zip(
        [max(x, y) for x, y in zip(a[0], b[0])],
        [min(x, y) for x, y in zip(a[1], b[1])]))


def _box(bounds):
    low, high = bounds
    centers, sizes = [], []
    for a, b in zip(low, high):
        center = _number(a / 2 + b / 2)
        size = _number(max(2 * (center - a), 2 * (b - center)))
        # Round outward when converting a union back to native box geometry.
        # This is representational rounding, not added physical clearance.
        while center - size / 2 > a or center + size / 2 < b:
            size = _number(math.nextafter(size, math.inf))
        centers.append(center)
        sizes.append(size)
    return {"translation_m": centers,
            "rotation_columns": copy.deepcopy(_IDENTITY),
            "dimensions_m": sizes}


def _box_bounds(value):
    if type(value) is not dict or set(value) != {"translation_m", "rotation_columns", "dimensions_m"}:
        raise ContractError(_CODE)
    transform = validate_rigid_transform({k: value[k] for k in ("translation_m", "rotation_columns")}, _CODE)
    if transform["rotation_columns"] != _IDENTITY:
        raise ContractError(_CODE)
    return _bounds(transform, _vector(value["dimensions_m"], positive=True))


def compute_reference_intent(context, samples, *, prior=None):
    """Return detached assignments, carried envelope, evidence, and next intent.

    ``samples`` is an immutable tuple of ``(gripper_pose, reference_m)`` pairs,
    beginning with the actual native anchor and including interval-entry points.
    Fixed-pose jaw-reference jumps include the swept finger boxes. Moving ARM
    geometry remains sampled, not a continuous-motion collision proof.

    ``prior`` is the same task owner's previously committed returned ``intent``.
    It retains possibilities, not an assumed executed suffix. No prior final
    reference is spliced onto the next native anchor. Pending candidates must
    not replace the owner's committed intent. The caller must use the returned
    envelope in its carried query; selecting only the mask is insufficient.
    """
    try:
        context, samples, prior = copy.deepcopy((context, samples, prior))
        if (type(context) is not dict
                or context.get("schema_version") != "data_factory.request_contact_geometry.v1"
                or context.get("status") != "PROSPECTIVE" or context.get("physical_success") is not False
                or context.get("geometry_digest") != canonical_digest({k: v for k, v in context.items() if k != "geometry_digest"})
                or type(samples) is not tuple or not samples):
            raise ContractError(_CODE)
        source = validate_rigid_transform(context["source_datum"], _CODE)
        destination = validate_rigid_transform(context["released_datum"], _CODE)
        dimensions = _vector(context["source_dimensions_m"], positive=True)
        opened, release = _number(context["open_m"]), _number(context["release_m"])
        mid, closed_gap = _number(context["jaw_midplane_m"]), _number(context["closed_gap_bound_m"])
        if not 0 < release < opened or closed_gap <= 0:
            raise ContractError(_CODE)
        boxes = context["fingertip_boxes"]
        if type(boxes) is not list or len(boxes) != 2:
            raise ContractError(_CODE)
        for box in boxes:
            if type(box) is not dict or set(box) != {"center", "dimensions"}:
                raise ContractError(_CODE)
            _vector(box["center"])
            _vector(box["dimensions"], positive=True)
        checked = []
        for sample in samples:
            if type(sample) is not tuple or len(sample) != 2:
                raise ContractError(_CODE)
            pose = validate_rigid_transform(sample[0], _CODE)
            g = _number(sample[1])
            if not 0 <= g <= opened:
                raise ContractError(_CODE)
            checked.append((pose, g))

        envelope, released = None, False
        if prior is not None:
            if (type(prior) is not dict or set(prior) != {
                    "schema_version", "geometry_digest", "carried_envelope", "released_possible", "intent_digest"}
                    or prior["schema_version"] != _SCHEMA
                    or prior["geometry_digest"] != context["geometry_digest"]
                    or type(prior["released_possible"]) is not bool
                    or prior["intent_digest"] != canonical_digest({k: v for k, v in prior.items() if k != "intent_digest"})):
                raise ContractError(_CODE)
            if prior["carried_envelope"] is not None:
                envelope = _box_bounds(prior["carried_envelope"])
            released = prior["released_possible"]
            if released and envelope is None:
                raise ContractError(_CODE)

        def fingers(g):
            return [_bounds({"translation_m": [box["center"][0] + sign * g, *box["center"][1:]],
                             "rotation_columns": _IDENTITY}, box["dimensions"])
                    for box, sign in zip(boxes, (1, -1))]

        carry_indices, release_indices, candidates = set(), set(), []
        release_percent = project_gripper_position(release, upper_m=opened, projection_quanta=0)["fairino_percent"]
        for index, (pose, g) in enumerate(checked):
            inv = inverse_rigid_transform(pose)
            relation = compose_rigid_transform(inv, source)
            source_bounds = _bounds(relation, dimensions)
            current_fingers = fingers(g)
            previous_g = checked[index - 1][1] if index else g
            fixed_pose = index > 0 and checked[index - 1][0] == pose
            sweeps = ([_union(a, b) for a, b in zip(fingers(previous_g), current_fingers)]
                      if fixed_pose and g < previous_g else current_fingers)
            gap = current_fingers[0][0][0] - current_fingers[1][1][0]
            possible = (g <= previous_g and gap <= source_bounds[1][0] - source_bounds[0][0] + 1e-9
                        and any(_overlap(source_bounds, finger) for finger in sweeps))
            if possible:
                transform, size = lateral_envelope(relation, dimensions, mid, closed_gap)
                candidate = _bounds(transform, size)
                envelope = candidate if envelope is None else _union(envelope, candidate)
                candidates.append({"sample_index": index, "source_to_gripper": relation,
                                   "jaw_sweep": bool(fixed_pose and g < previous_g)})
                if fixed_pose and g < previous_g:
                    carry_indices.add(index - 1)
            if envelope is not None:
                carry_indices.add(index)
                # This is possible placement intent, not an observed landing.
                # The bounding volume between the open fingertips is derived
                # from the existing boxes, not a new pose-proximity threshold.
                aperture = _union(*current_fingers)
                target = _bounds(compose_rigid_transform(inv, destination), dimensions)
                percent = project_gripper_position(g, upper_m=opened, projection_quanta=0)["fairino_percent"]
                if percent >= release_percent and _overlap(aperture, target):
                    released = True
            if released:
                release_indices.add(index)

        carried = None if envelope is None else _box(envelope)
        intent = {"schema_version": _SCHEMA, "geometry_digest": context["geometry_digest"],
                  "carried_envelope": carried, "released_possible": released}
        intent["intent_digest"] = canonical_digest(intent)
        source_indices = set(range(len(checked))) - carry_indices - release_indices
        assignments = tuple((name, tuple(sorted(indices))) for name, indices in (
            ("source", source_indices), ("carried", carry_indices), ("released", release_indices)) if indices)
        evidence = {"status": "PROSPECTIVE", "physical_success": False,
            "semantics": "SAMPLED_REFERENCE_POSSIBLE_OCCUPANCY", "source_world_retained": True,
            "samples_digest": canonical_digest(samples), "geometry_digest": context["geometry_digest"],
            "prior_intent_digest": None if prior is None else prior["intent_digest"],
            "intent_digest": intent["intent_digest"], "candidate_relations": candidates,
            "carried_frame": "gripper_link", "contact_authorized": False,
            "continuous_motion_proof": False, "release_confirmed": False}
        return {"assignments": assignments, "carried_envelope": copy.deepcopy(carried),
                "evidence": evidence, "intent": intent}
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise ContractError(_CODE) from exc
