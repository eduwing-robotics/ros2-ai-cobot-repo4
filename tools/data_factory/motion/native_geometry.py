"""Task-owned CPU geometry query; never a command owner or motion scheduler.

The caller pumps poll() alongside native progress/watchdog/cancel. Construction
belongs before fresh policy acquisition. A query concerns the supplied sampled
states only; its binding is not a current-state/Scene/authority admission. The
sole executor must compare those identities again immediately before dispatch.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import time

from tools.fr5_data_factory import ContractError, DIGEST, canonical_digest, validate_rigid_transform
from .contact_transition import bind_native_request_geometry, bind_request_contacts


MAX_LINE_BYTES = 16 * 1024 * 1024  # IPC resource bound, not a motion tolerance.


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class NativeGeometry:
    """One initialized model/scene, one outstanding query, one I/O worker.

    All pipe waits, JSON/CDR conversion, native computation and contact
    classification are off the owner loop. Inputs are detached at setup and
    query samples must be immutable tuples. No concurrent submitters: the
    existing task executor owns start/submit/poll/close.
    """

    def __init__(self, *, urdf, srdf, scene, context, plan, deadline, command=None):
        if type(urdf) is not str or type(srdf) is not str:
            raise ContractError("NATIVE_GEOMETRY_MODEL")
        self._deadline = self._check_deadline(deadline)
        self._lock = threading.Lock()
        self._closed = False
        self._ready = False
        self._process = None
        self._serial = 1
        self._binding = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fr5-geometry-io")
        # Setup runs before live observation acquisition; retain no mutable
        # ROS message/context owned by the caller across the worker boundary.
        retained = copy.deepcopy((scene, context, plan))
        self._future = self._worker.submit(self._initialize, urdf, srdf, retained,
                                           None if command is None else tuple(command))

    @staticmethod
    def _check_deadline(deadline):
        if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ContractError("NATIVE_GEOMETRY_DEADLINE")
        return deadline

    def _initialize(self, urdf, srdf, retained, command):
        from rclpy.serialization import serialize_message
        from moveit_msgs.msg import PlanningScene
        scene, context, plan = retained
        if not isinstance(scene, PlanningScene) or scene.is_diff:
            raise ContractError("NATIVE_GEOMETRY_FULL_SCENE_REQUIRED")
        self._build = bind_native_request_geometry(context, plan)
        self._classify = bind_request_contacts(context, plan, released_world=True)
        self._context, self._plan = context, plan
        if command is None:
            from ament_index_python.packages import get_package_prefix
            command = [str(Path(get_package_prefix("fr5_motion_geometry")) /
                           "lib/fr5_motion_geometry/fr5_native_geometry")]
        with self._lock:
            if self._closed:
                raise ContractError("NATIVE_GEOMETRY_CLOSED")
            # Task-owned CPU subprocess; no ROS node, detached shell, hardware
            # authority or unread stderr pipe. Its lifetime is closed below.
            self._process = subprocess.Popen(tuple(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        scene_bytes = serialize_message(scene)
        scene_hex = scene_bytes.hex()
        request = dict(op="init", id=1, urdf=urdf, srdf=srdf, scene_cdr_hex=scene_hex)
        response, digest = self._exchange(request)
        if set(response) != {"op", "id", "ok"}:
            raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
        self._scene_digest = hashlib.sha256(scene_bytes).hexdigest()
        self._initialization_digest = digest
        return {"status": "READY", "initialization_digest": digest}

    def _exchange(self, request):
        raw = json.dumps(request, allow_nan=False, separators=(",", ":")).encode()
        if len(raw) > MAX_LINE_BYTES:
            raise ContractError("NATIVE_GEOMETRY_PAYLOAD_LIMIT")
        self._process.stdin.write(raw + b"\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline(MAX_LINE_BYTES + 2)
        if not line.endswith(b"\n") or len(line) > MAX_LINE_BYTES + 1:
            raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
        try:
            def invalid_constant(_value):
                raise ValueError("nonfinite JSON value")
            response = json.loads(line, object_pairs_hook=_object, parse_constant=invalid_constant)
        except (ValueError, UnicodeError) as exc:
            raise ContractError("NATIVE_GEOMETRY_PROTOCOL") from exc
        if (type(response) is not dict or response.get("op") != request["op"]
                or type(response.get("id")) is not int or response["id"] != request["id"]
                or type(response.get("ok")) is not bool):
            raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
        if not response["ok"]:
            if set(response) != {"op", "id", "ok", "error"} or not isinstance(response["error"], str):
                raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
            raise ContractError("NATIVE_GEOMETRY_REJECTED", response["error"])
        return response, "sha256:" + hashlib.sha256(raw).hexdigest()

    def submit(self, samples, *, binding, deadline, derive_intent=False, prior_intent=None):
        """Submit ((hypothesis, ((j1,..,j6,g), ...)), ...), not a motion goal.

        The binding belongs to the caller's exact revision/Scene/contact/timing
        selection. A future candidate gets a different binding; no late result
        can be silently relabeled as that candidate's check.
        """
        if self._closed or not self._ready:
            raise ContractError("NATIVE_GEOMETRY_NOT_READY")
        if self._future is not None:
            raise ContractError("NATIVE_GEOMETRY_BUSY")
        if not isinstance(binding, str) or not DIGEST.fullmatch(binding):
            raise ContractError("NATIVE_GEOMETRY_BINDING")
        if (type(samples) is not tuple or not samples
                or any(type(v) is not tuple or len(v) != 2 or type(v[0]) is not str
                       or v[0] not in {"source", "carried", "released"} or type(v[1]) is not tuple or not v[1]
                       or any(type(row) is not tuple or len(row) != 7
                              or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)
                              for row in v[1]) for v in samples)
                or len({v[0] for v in samples}) != len(samples)):
            raise ContractError("NATIVE_GEOMETRY_SAMPLES")
        if (type(derive_intent) is not bool
                or derive_intent and (len(samples) != 1 or samples[0][0] != "source")
                or not derive_intent and prior_intent is not None):
            raise ContractError("NATIVE_GEOMETRY_SAMPLES")
        self._deadline = self._check_deadline(deadline)
        # A derived check may issue FK/source then expanded-occupancy queries.
        # Each exchange gets a distinct identity even within one revision.
        self._serial += 2 if derive_intent else 1
        self._binding = binding
        self._future = (self._worker.submit(self._reference_query, samples[0][1], binding,
                                            self._serial - 1, copy.deepcopy(prior_intent)) if derive_intent else
                        self._worker.submit(self._query, samples, binding, self._serial))

    def _reference_query(self, rows, binding, serial, prior):
        """Derive possible occupancy from native FK on this same CPU worker.

        The source query is not motion admission. Its poses select prospective
        masks and an enclosing carried box for the final full-scene check.
        Neither a pending query nor its inferred occupancy updates Scene truth.
        """
        from .reference_intent import compute_reference_intent
        source = self._query((("source", rows),), binding, serial)
        samples = tuple((sample["gripper_pose"], row[6])
                        for sample, row in zip(source["variants"][0]["samples"], rows))
        intent = compute_reference_intent(self._context, samples, prior=prior)
        assignments = intent["assignments"]
        if assignments == (("source", tuple(range(len(rows)))),):
            result = source
        else:
            context = copy.deepcopy(self._context)
            envelope = intent["carried_envelope"]
            if envelope is not None:
                context["proxies"]["carried"].update(
                    translation_m=envelope["translation_m"], dimensions_m=envelope["dimensions_m"],
                    rotation_xyzw=[0., 0., 0., 1.])
            context["geometry_digest"] = canonical_digest({k: v for k, v in context.items() if k != "geometry_digest"})
            result = self._query(tuple((h, tuple(rows[i] for i in indices)) for h, indices in assignments),
                                 binding, serial + 1, build=bind_native_request_geometry(context, self._plan),
                                 classify=bind_request_contacts(context, self._plan, released_world=True))
        result.update(assignments=assignments, reference_intent=intent,
                      source_query_digest=source["query_digest"])
        return result

    def _query(self, samples, binding, serial, *, build=None, classify=None):
        from rclpy.serialization import serialize_message, deserialize_message
        from moveit_msgs.srv import GetStateValidity
        build = self._build if build is None else build
        classify = self._classify if classify is None else classify
        variants = []
        for hypothesis, rows in samples:
            states, objects = [], []
            for row in rows:
                state, world = build(hypothesis, row[:6], row[6])
                if not states:
                    objects = [serialize_message(obj).hex() for obj in world]
                states.append(serialize_message(state).hex())
            variants.append(dict(hypothesis=hypothesis, world_objects_cdr_hex=objects, states_cdr_hex=states))
        response, digest = self._exchange(dict(op="query", id=serial, variants=variants))
        if (set(response) != {"op", "id", "ok", "variants"} or type(response["variants"]) is not list
                or len(response["variants"]) != len(samples)):
            raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
        results = []
        for (hypothesis, rows), variant in zip(samples, response["variants"]):
            if (type(variant) is not dict or set(variant) != {"hypothesis", "samples"}
                    or variant["hypothesis"] != hypothesis or type(variant["samples"]) is not list
                    or len(variant["samples"]) != len(rows)):
                raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
            checked = []
            for row, sample in zip(rows, variant["samples"]):
                if (type(sample) is not dict or set(sample) != {"response_cdr_hex", "saturated", "gripper_pose"}
                        or type(sample["saturated"]) is not bool):
                    raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
                try:
                    hex_value = sample["response_cdr_hex"]
                    raw = bytes.fromhex(hex_value)
                    validity = deserialize_message(raw, GetStateValidity.Response)
                    # Native CDR padding bytes need not equal Python's padding.
                    # Extent still rejects a valid prefix with trailing bytes.
                    if raw.hex() != hex_value or len(serialize_message(validity)) != len(raw):
                        raise ValueError("CDR framing")
                    pose = validate_rigid_transform(sample["gripper_pose"], "NATIVE_GEOMETRY_PROTOCOL")
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise ContractError("NATIVE_GEOMETRY_PROTOCOL") from exc
                if validity.valid and validity.contacts:
                    raise ContractError("NATIVE_GEOMETRY_PROTOCOL")
                allowed = not sample["saturated"] and not validity.constraint_result and (
                    validity.valid or bool(validity.contacts) and all(
                        classify(hypothesis, contact, gripper_pose=pose, gripper_m=row[6])
                        for contact in validity.contacts))
                checked.append(dict(allowed=allowed, saturated=sample["saturated"],
                                    response=validity, gripper_pose=pose))
            results.append(dict(hypothesis=hypothesis, samples=checked))
        return dict(status="CHECKED", binding=binding, query_digest=digest,
                    initialization_digest=self._initialization_digest,
                    scene_cdr_digest="sha256:" + self._scene_digest, variants=results)

    def poll(self, *, binding=None):
        """Return None while pending, never spin/wait on the CPU worker.

        Mismatch discards a completed query, not the current actuator revision.
        Timeout kills only this query process; the executor owns whether its
        independently monitored current trajectory may continue or must stop.
        """
        if self._closed:
            raise ContractError("NATIVE_GEOMETRY_CLOSED")
        if self._future is None:
            return None
        if time.monotonic() >= self._deadline:
            self.close()
            raise ContractError("NATIVE_GEOMETRY_TIMEOUT")
        if not self._future.done():
            return None
        future, self._future = self._future, None
        try:
            result = future.result()  # done() above: no computation wait.
        except Exception:
            self.close()
            raise
        if self._binding is None:
            self._ready = True
        elif binding != self._binding:
            raise ContractError("NATIVE_GEOMETRY_SUPERSEDED")
        return result

    def close(self):
        """Pollable teardown, limited to the owned CPU process (never ROS)."""
        with self._lock:
            self._closed = True
            process = self._process
            if process is not None and process.poll() is None:
                process.kill()
        self._worker.shutdown(wait=False, cancel_futures=True)
        if self._future is not None and not self._future.done():
            return False
        if process is not None:
            if process.poll() is None:
                return False
            process.stdin.close()
            process.stdout.close()
        return True
