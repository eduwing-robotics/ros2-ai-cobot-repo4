"""One normal HUMAN_GATED run, presented through existing operator decision ports."""
from __future__ import annotations

import copy
from pathlib import Path
import threading
import time

from tools.data_factory import run_job
from tools.data_factory.operator.registries.region import (
    load_workspace_region_binding,
)
from tools.data_factory.operator.workflow.intents import (
    ButtonDecisionPort, OperatorCheckpointPort, OperatorIntentCore, INTENT_SCHEMA,
)
from tools.data_factory.task_recipe import (
    compile_episode_instruction_binding,
    compile_task_binding,
)
from tools.a4_place_yaw.region_layout import (
    make_red_blue_region_layout,
    workspace_region,
)
from tools.fr5_data_factory import (
    ContractError,
    canonical_digest,
    load_json_strict,
    validate_job_spec,
)


def compile_learned_episode_instruction(payload, *, repository_root):
    """Compile one scoped pick-place label from the repository's region SSOT."""
    checked = run_job._run_payload(copy.deepcopy(payload))
    destination = checked.get("destination")
    if (
        checked["mode"] != "live"
        or run_job._learned_options(checked) is None
        or checked["job"].get("task") != "pick_place"
        or not isinstance(destination, dict)
    ):
        raise ContractError("EPISODE_INSTRUCTION_SCOPE")
    source = validate_job_spec(
        checked["job"],
        paths={
            "selected_sheet": checked["selected_sheet"],
            "yaw0_sheet": checked["yaw0_sheet"],
        },
        config_root=checked["config_root"],
    )
    target = validate_job_spec(
        destination["job"],
        paths={
            "selected_sheet": destination["selected_sheet"],
            "yaw0_sheet": destination["yaw0_sheet"],
        },
        config_root=checked["config_root"],
    )
    shared_job = (
        "task", "robot_system_id", "collection_profile_id",
        "object_profile_id", "grasp_profile_id", "instruction",
        "episode_intent",
    )
    shared_inputs = (
        "robot_system", "collection_profile", "object_profile",
        "grasp_profile",
    )
    if (
        target["normalized_job"]["place_id"]
        == source["normalized_job"]["place_id"]
        or any(
            target["normalized_job"][field]
            != source["normalized_job"][field]
            for field in shared_job
        )
        or any(
            target["input_digests"][field]
            != source["input_digests"][field]
            for field in shared_inputs
        )
    ):
        raise ContractError("EPISODE_INSTRUCTION_SCOPE")

    try:
        repository = Path(repository_root).resolve(strict=True)
        layout = make_red_blue_region_layout()
        persisted = load_workspace_region_binding(repository, layout)
        persisted_endpoints = {
            item["place_id"]: item for item in persisted["bindings"]
        }

        def endpoint(resolved, role):
            job = resolved["normalized_job"]
            region = workspace_region(layout, job["place_id"])
            stored = persisted_endpoints.get(job["place_id"])
            if stored != {
                "place_id": job["place_id"],
                "frame_id": job["cell_calibration_id"],
                "region_id": region["region_id"],
            }:
                raise ContractError("EPISODE_INSTRUCTION_SCOPE")
            return {
                "role": role,
                "workspace_id": job["place_id"],
                "frame_id": job["cell_calibration_id"],
                "pose": {
                    key: job[key]
                    for key in ("place_id", "yaw_deg", "x_mm", "y_mm")
                },
                "sheet_digest": job["sheet_manifest_digest"],
                "family_digest": resolved["calibration"]["document"][
                    "a4_family_digest"
                ],
                "region_binding": {
                    "layout_id": layout["layout_id"],
                    "layout_digest": layout["layout_digest"],
                    "region_id": region["region_id"],
                    "physical_binding_status": persisted[
                        "physical_binding_status"
                    ],
                },
            }

        task = compile_task_binding(
            "pick_place",
            source=endpoint(source, "SOURCE"),
            destination=endpoint(target, "DESTINATION"),
        )
        return compile_episode_instruction_binding(task, source["object_profile"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ContractError("EPISODE_INSTRUCTION_SCOPE") from exc


class LearnedRunApplication:
    def __init__(self, *, payload, operator_label, session_id=None,
                 run_live_call=None, repository_root=None):
        self.payload = run_job._run_payload(copy.deepcopy(payload))
        if (self.payload["mode"] != "live" or run_job._learned_options(self.payload) is None
                or self.payload["job"].get("operator_or_agent_id") != operator_label):
            raise ContractError("LEARNED_WEB_CONFIGURATION")
        from tools.data_factory.rollout.task_authority import uses_scoped_generation
        self.repository_root = repository_root or run_job.ROOT
        self.episode_instruction_binding = None
        if (
            uses_scoped_generation(self.payload.get("task_grant"))
            and self.payload["job"].get("task") == "pick_place"
        ):
            self.episode_instruction_binding = compile_learned_episode_instruction(
                self.payload,
                repository_root=self.repository_root,
            )
            if (
                self.payload["task_grant"]["scope"].get("task")
                != self.episode_instruction_binding["instruction"]
            ):
                raise ContractError("TASK_GRANT_SCOPE")
        self.session_id = session_id or "learned-web-" + self.payload["run_id"]
        self.plan = ButtonDecisionPort(session_id=self.session_id + "-plan", operator_label=operator_label)
        self.checkpoint = OperatorCheckpointPort(operator_label=operator_label)
        self.run_live_call = run_live_call or run_job.run_live
        self.cancel_event = threading.Event()
        self.worker = None
        self.state = "READY"
        self.latest = self.result = self.lifecycle_result = self.plan_envelope = None
        self.error = None
        self.core = OperatorIntentCore(session_id=self.session_id, projection_call=self.projection,
            handlers={"start_learned_run": self.start, "cancel_learned_run": self.cancel,
                      "approve_exact_plan": self.approve, "resolve_checkpoint": self.resolve_checkpoint})

    def projection(self):
        pending = self.plan.core.snapshot()["projection"]["pending_plan"]
        checkpoint = self.checkpoint.projection()
        active = self.state == "ACTIVE" and not self.cancel_event.is_set()
        ops = ["start_learned_run"] if self.state == "READY" else []
        if active:
            ops.append("cancel_learned_run")
            if pending is not None:
                ops.append("approve_exact_plan")
            if checkpoint is not None:
                ops.append("resolve_checkpoint")
        return {"kind": "LEARNED_RUN", "run_id": self.payload["run_id"], "state": self.state,
                "instruction": self.payload["job"].get("instruction"), "available_ops": ops,
                "pending_plan": copy.deepcopy(pending), "plan_envelope": copy.deepcopy(self.plan_envelope),
                "checkpoint": checkpoint, "latest": copy.deepcopy(self.latest),
                "result": copy.deepcopy(self.result), "lifecycle_result": copy.deepcopy(self.lifecycle_result),
                "error": self.error, "approval_scope": "HUMAN_GATED", "training_authority": False}

    def _decision(self, request):
        # The normal owner publishes this exact evidence before requesting approval.
        evidence = load_json_strict(Path(self.payload["run_root"]) / self.payload["run_id"] / "preapproval_evidence.json")
        envelope = evidence["plan_envelope"]
        if (canonical_digest(envelope) != request["decision_binding"]["plan_envelope_digest"]
                or canonical_digest(envelope["plan"]) != request["plan_digest"]):
            raise ContractError("LEARNED_WEB_PLAN_CHANGED")
        def offer():
            self.plan_envelope = copy.deepcopy(envelope)
            self.plan.offer(run_id=request["run_id"], plan_digest=request["plan_digest"],
                decision_binding=request["decision_binding"], approval_scope=request["approval_scope"])
        self.core.transition(offer)
        deadline = None if request["timeout_s"] is None else time.monotonic() + request["timeout_s"]
        while not self.cancel_event.is_set():
            remaining = .1 if deadline is None else min(.1, deadline - time.monotonic())
            if remaining <= 0:
                return None
            decision = self.plan.wait(remaining)
            if decision is not None:
                return decision
        return None

    def _checkpoint(self, request):
        def offer():
            if request["kind"] == "LEARNED_NEXT_PLAN":
                envelope = request["evidence"]["plan_envelope"]
                if canonical_digest(envelope["plan"]) != request["plan_digest"]:
                    raise ContractError("LEARNED_WEB_PLAN_CHANGED")
                self.plan_envelope = copy.deepcopy(envelope)
            self.checkpoint.offer(request)
        self.core.transition(offer)
        return self.checkpoint.wait(request["timeout_s"])

    def _publish(self, value):
        self.core.transition(lambda: setattr(self, "latest", copy.deepcopy(value)))

    def _run(self):
        try:
            def no_tty(*_args, **_kwargs):
                raise ContractError("LEARNED_WEB_TTY_FORBIDDEN")
            instruction = (
                {}
                if self.episode_instruction_binding is None
                else {
                    "episode_instruction_binding": copy.deepcopy(
                        self.episode_instruction_binding
                    ),
                    "repository_root": self.repository_root,
                }
            )
            result = self.run_live_call(copy.deepcopy(self.payload), self.cancel_event, self._publish,
                decision_provider=self._decision, checkpoint_provider=self._checkpoint,
                approval_scope="HUMAN_GATED", tty_decision=no_tty, **instruction)
            self.core.transition(lambda: setattr(self, "result", copy.deepcopy(result)))
            lifecycle = None
            path = Path(self.payload["run_root"]) / self.payload["run_id"] / "learned_lifecycle_result.json"
            if path.is_file() and result.get("data") is not None:
                from tools.data_factory.rollout.evidence_boundary import build_run_diagnostic
                lifecycle = load_json_strict(path)
                if lifecycle.get("run_id") != self.payload["run_id"] or build_run_diagnostic(lifecycle) != result["data"]:
                    raise ContractError("LEARNED_WEB_RESULT_CHANGED")
            def finish():
                self.result = copy.deepcopy(result)
                self.lifecycle_result = lifecycle
                self.state = "TERMINAL"
            self.core.transition(finish)
        except Exception as exc:
            def failed():
                self.error = getattr(exc, "code", "LEARNED_WEB_FAILED")
                self.state = "TERMINAL"
            self.core.transition(failed)
        finally:
            self.checkpoint.close()

    def start(self, payload, _view):
        if payload or self.state != "READY" or self.worker is not None:
            raise ContractError("LEARNED_WEB_START")
        self.state = "ACTIVE"
        self.worker = threading.Thread(target=self._run, name="learned-web-run")
        self.worker.start()
        return {"started": True, "run_id": self.payload["run_id"]}

    def approve(self, payload, _view):
        if self.state != "ACTIVE" or self.cancel_event.is_set():
            raise ContractError("LEARNED_WEB_INACTIVE")
        view = self.plan.core.snapshot()
        return self.plan.core.consume({"schema_version": INTENT_SCHEMA, "intent_id": self.session_id + "-approve",
            "session_id": view["session_id"], "view_revision": view["revision"], "view_digest": view["view_digest"],
            "op": "approve_exact_plan", "payload": payload})["result"]

    def resolve_checkpoint(self, payload, _view):
        if self.state != "ACTIVE" or self.cancel_event.is_set():
            raise ContractError("LEARNED_WEB_INACTIVE")
        return self.checkpoint.resolve(payload)

    def cancel(self, payload, _view):
        if payload or self.state != "ACTIVE":
            raise ContractError("LEARNED_WEB_CANCEL")
        self.cancel_event.set()
        self.checkpoint.close()
        self.state = "CANCELLING"
        return {"cancel_requested": True, "stop_confirmed": False}

    def close(self):
        self.cancel_event.set()
        self.checkpoint.close()
        if self.worker is not None:
            self.worker.join(5)
            if self.worker.is_alive():
                raise ContractError("LEARNED_WEB_OWNER_ACTIVE")
