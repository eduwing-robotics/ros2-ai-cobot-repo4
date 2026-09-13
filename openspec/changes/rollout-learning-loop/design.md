# Execution ownership and temporal qualification

## Outcome and decision status

The target remains a real learned Pick, qualified mechanical release, attributable
diagnosis and targeted recollection. A successful clock query, CPU replay or
framework adapter is not that outcome. Optimize total implementation, recurring
debugging and operating cost, not changed-line count. Runtime adoption remains
pending; the characterization tests below do not change a deployed threshold.

## Reuse boundary

### September 13 product takeover: scoped temporal responsibility decision

Decision: **REOPEN temporal responsibility**, limited to the existing bounded
learned-task path. Main `2bab180` and its retained runtime evidence supersede the
old coordinator's remembered deployment state. Thin natural-use work is closed;
this is product work, not a Router experiment or executor replacement.

The current source checks original input age at generation completion and again
on entry to initial planning. Continuation additionally checks it after planning.
After admission, however, approved recorder/approval delay already uses fresh
execution state instead of the original image deadline. Thus `300 ms` is not a
current observation-to-first-action deadline: placing the same delay on opposite
sides of initial admission can change the verdict. OpenSpec explicitly required
that behavior; changing it is a prospective contract change, not a claim that
the implementation violated its specification.

The last returned-user input reached inference completion at age295.097ms after
169.110ms inference, then initial planning rejected before any learned/gripper
send. The historical child admission time remains UNKNOWN. The saved-proposal
12.442ms CPU observation is neither that missing time nor a latency guarantee.
Optimizing for the remaining4.903ms alone leaves the semantic asymmetry open.

Selected implementation direction, **not yet deployed or qualified**:

- Preserve legacy proposal/admission semantics and their diagnostic readers.
  Introduce an explicitly version-bound frozen-at-generation mode for existing
  scoped task grants; do not silently reinterpret archived proposals.
- Keep the existing original-stamp capture/pre/post-inference qualification.
  A qualified immutable full output may become a candidate within its original
  run, grant, Scene binding and predecessor context, not a motion authorization.
- Retain generation context before capture and bind it into the proposal digest:
  run identity, task-grant digest, Scene-binding digest and predecessor plan
  identity (absent for the initial output). Do not attach that authority later
  while importing an old output. A mode flag alone does not prevent replay.
- The existing owner validates the matching grant before granting this temporal
  interpretation. Initial and continuation planning use the same rule. Planning
  must not restart the grant's wall/monotonic budget or terminal reserve; exact
  plan admission and sole-executor dispatch remain separate later operations.
- Preserve current full-state/start agreement, controller/hardware identity and
  freshness, collision/contact, Scene/Cell, lease, cancellation and task bounds.
  A HUMAN_GATED receipt alone cannot authorize the new scoped mode.

This direction relies on the already approved structured Scene domain, just as
the current post-admission delay does. It does not establish that unchanged
robot joints or Scene revision detect arbitrary physical object movement. No
new camera/person-absence approval or visual-validity detector is introduced.
Unexpected physical changes remain subject to existing Scene/Cell invalidation.

Implementation acceptance is a focused controlled-clock comparison: identical
generation/context/current evidence with delay before versus after plan entry
must receive the same new-mode temporal decision. Old output rewrapped under a
new grant/run/Scene or reused with another predecessor, stale-at-generation or
future completion, expired/revoked authority, cancelled continuation, changed
hardware/start/Scene and stale current-state cases must reject with zero sends.
Preserve original bytes/timestamps, strict JSONL and plan-only zero effects.
Legacy freshness/rejection fixtures must retain their original meaning.

Native capture/serialization optimization is an independent cost question, not
a replacement for this decision. LeRobot continues to own loading, processors
and inference. Only a concrete measured saving warrants an adapter change.
Fresh physical qualification and Pick/release/recollection remain open.

LeRobot owns policy implementation, saved processors and supported inference
mechanisms. FR5 already calls `SmolVLAPolicy.from_pretrained`, saved native
processors and `predict_action_chunk` in `learned_action_adapter.py`. Do not
reimplement these capabilities or add a competing generic rollout framework.

The installed LeRobot 0.6.1 also has `Robot.get_observation/send_action`,
`rollout/inference/`, and `lerobot-rollout`. Its `SyncInferenceEngine.get_action`
returns one action via `select_action`; `strategies/core.send_next_action`
interpolates/processes that action, calls the robot, and returns the pre-robot
action dictionary rather than the robot's returned actual action. The default
teardown can issue interpolated return-to-initial commands. These are observed
local interface semantics, not a claim that upstream cannot support FR5.

Prefer a supported LeRobot interface when its adapter removes more owned code
and recurring work than it introduces. Compare at the actual seam: preserve
full model output and consumed indices for trajectory admission, distinguish
proposed/processed/sent/measured actions, and route normal motion and teardown
through the same FR5 execution authority. A per-row adapter must not acknowledge
buffered rows as physically executed, bypass collision admission, or introduce
a second device/recorder owner. No dependency upgrade or whole-runner replacement
is justified solely by the presence of a CLI.

References: [hardware interface](https://huggingface.co/docs/lerobot/integrate_hardware),
[policy deployment](https://huggingface.co/docs/lerobot/inference).
Local installed source remains the integration authority; documentation can
describe a different revision.

## Responsibilities to retain or repair

Apply requirements at the action that consumes them. Reuse static qualification
within its source/model/configuration scope; reopen it only for relevant changes
or counterevidence. Recheck mutable scene, cell, ownership and freshness at their
existing consumption boundaries. Do not make later training utility or whole-task
success a prerequisite for a bounded hardware diagnostic. An implementation's
chosen timeout is revisable engineering policy, not automatically a physical
safety invariant.

The approved product domain and existing SceneStateStore/scene/cell contracts
govern the environment assumptions. Do not add coordinator visual approval,
per-attempt person-absence approval, or a new perception gate. Existing mapped
illumination checks belong in the system execution path; an agent's preview is
diagnostic assistance, not runtime authority. This does not claim that the scene
store detects arbitrary people or unregistered obstacles.

- Policy proposes actions; it neither approves collision safety nor sends SDK
  commands. Admission validates the proposed motion against current state,
  registered geometry and selected limits. The sole executor owns dispatch,
  cancellation and qualified mechanical release.
- Collision checking currently samples knots and four intermediate states per
  interval. It is not a continuous collision proof or a model of unregistered
  obstacles. Grasp contact and carried geometry require their existing bindings.
- Native feedback acquisition owns source/receipt timestamps and clock-query
  validity. Completion evidence remains command/incarnation-specific. Neither
  network response time nor a held reference establishes physical completion.
- Observation quality, execution-state freshness, RPC waiting and total command
  duration are different contracts. Do not infer a hardware safety limit from
  camera FPS or silently reuse one numeric bound as another authority.

## Minimum sufficient execution evidence

The next opt-in adapter shall separate coherent state delivery, command completion
and cross-clock alignment quality. This is a successor contract, not permission
to reinterpret or bypass the deployed version 3/4 certificate reader.

Current-state delivery must preserve one complete native frame, its original host
receipt time, publication sequence and connection/configuration identity. A getter
or ROS republish cannot renew that receipt. Missing/expired delivery, a changed
connection, regressing source state, device faults and unresolved motion ownership
remain execution concerns. A recent receipt alone does not establish physical
acquisition age or rule out a buffered TCP backlog; those limitations must remain
explicit in qualification, not hidden behind a `source_timestamp` name.

Precise clock alignment is a separate quality result. A slow optional clock query
must not by itself turn otherwise qualified arm-state delivery into hardware
ERROR. Where post-command timing is needed to establish gripper completion, that
evidence remains required at the completion/release boundary, with its original
command, generation, deadline and proof instant. Historical completion is not
recertified on every later arm write. A failed required command proof cannot be
downgraded to a quality warning. No new safety threshold is selected here.

This separation follows the distinction between receipt and reconstructed device
time in the [UR RTDE publisher](https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_rtde_ros2_publisher/doc/usage.html).
Its timing offsets and robot-specific assumptions are not FR5 limits. In
ros2_control, ERROR invokes the hardware lifecycle error path; it is not an
appropriate label for missing optional analysis precision.

The matching manufacturer source at `fairino-cpp-sdk` commit
`0553c35d760a4e76c9b8d2fc0208ca83e6d731cd` exposes a prerequisite: the CNDE receive
thread mutates shared state field by field, while `GetRobotRealTimeState` copies
it without the receive mutex. Reuse that decoder with a short coherent publication
boundary, not another connection or the mutex held across blocking receive.
This source finding is not evidence that recorded datasets are corrupted. Fields
3–75 include the last ServoJ target, but not field 76's command count; neither is
an established same-command completion acknowledgement.

## Why timeout-only qualification is insufficient

The baseline native `refresh_gripper_freshness` before `e7f22e7` allocates
one quarter of the selected age to each RPC and half to the first-query/frame
stage. Its exception path sets `_gripper_error=-5`; `read()` then returns hardware
ERROR. The query already runs outside the main read/write thread, so merely
adding another asynchronous wrapper would not remove this error coupling.

The baseline CPU acquisition characterization at `b9931d6` reproduces rejection
of a 37 ms query under the selected 100 ms policy, while a 2 ms query succeeds without command sends.
This approximates an observed 36.838 ms response tail; it is not a measured
worst-case controller bound. The test intentionally characterizes the current
cutoff; the candidate regression now requires the same bounded acquisition to succeed.

A deterministic native-predicate counterexample is stronger than first-query
success: a hypothetical 74 ms acquisition is valid on publication, but its
original 100 ms lease expires before another serial 74 ms acquisition completes.
This is a synthetic schedule, not a claim that every physical query takes 37 ms.
Thus a remaining-total-budget correction alone needs a renewal test, not just
an initial readiness test.

Compare that correction with a bounded correction of certification scheduling
and error classification. Distinguish a pending/failed refresh while previous
evidence remains valid from actual evidence expiry, source regression,
incarnation change or controller error. Never extend old timestamps, ignore
expiry, replay a motion command or claim safe automatic recovery. Determine
whether certification cadence can sustain the selected bound before changing
fault propagation; moving a query to another thread is not sufficient.

This matches the purpose, not an automatic adoption, of ros2_control's
[asynchronous hardware](https://control.ros.org/jazzy/doc/ros2_control/hardware_interface/doc/asynchronous_components.html).
Its [read/write error handling](https://control.ros.org/jazzy/doc/ros2_control/hardware_interface/doc/handling_errors_during_read_write.html)
also makes hardware ERROR a lifecycle event, not an ordinary retry status.

## Candidate selected by native continuous replay

The first budget-only candidate still failed the repeating 2/2/37 ms CPU query
schedule: the worker added its fixed post-query sleep even when acquisition had
already consumed the reuse horizon. The revised candidate derives the next wait
from the original certificate start and existing reuse horizon, rather than
starting a new sleep budget at publication. RPCs and the frame wait share the
original whole acquisition budget. The age/tolerance, original anchors and
read/write expiry checks are unchanged; no query exception is hidden or downgraded.

This repairs scheduling in the existing asynchronous owner rather than adding a
second sampler, fault/recovery state machine or generic execution layer. Tests
exercise native producer/sampler/write together, accept interspersed tails, and
require zero later sends for sustained delays that exhaust evidence validity.
Mock servo counters are not physical execution evidence. Independent review,
full regression and physical renewal qualification remain required before
runtime promotion; the candidate does not prove that 100 ms is a physical
stopping-distance guarantee or tolerate arbitrary communication delays.

## Required verification before runtime promotion

### Attribute the query path before changing its temporal contract

The installed manufacturer `libfairino.so.2.3.7` remains byte-identical to the
pinned vendor commit. FR5 changes the ROS adapter, including a separate libcurl
`GetSystemClock` client; it has not patched this SDK binary. Repeated query
certification and propagation of its expiry into hardware ERROR are FR5-owned
integration decisions, not evidence that the vendor SDK is defective.

Installed SDK inspection and a read-only native call establish that existing
`GetRobotRealTimeState` consumes a CNDE cache from TCP 20005. Its configuration
readback returns an 8 ms period and fields 3–75, including `RobotTime` and gripper
feedback. `GetRobotRealtimeStateSamplePeriod` instead names TCP 20004 in the
pinned header; a same-name XMLRPC fault cannot disprove a local SDK operation.
Reuse the installed interface before proposing another CNDE connection, changing
its field set/period or adopting a newer SDK. Configured period is not a measured
delivery bound or a new freshness/completion authority. The manufacturer's
[CNDE interface](https://fairino-doc-en.readthedocs.io/latest/RobotCommunication/cnde_introduction.html)
describes streaming feedback, but latest documentation is not installed-firmware
qualification.

Wire and client measurements during bounded current-position hold agree on a
48.6 ms clock-response delay while UDP servo requests continue at roughly 10 ms
intervals with replies. This rules against a comparable userspace scheduling
delay in that sample, not every host/network cause. A later read-only SDK-close
comparison also produced timeouts; controller servo load is not established as
the sole cause. See the handoff's exact artifacts. Do not substitute another RPC
method into q0/frame/q1 proof slots, treat tracing overhead as production timing,
or weaken the lease on these observations. Any correction must account for
current stream age, same-command completion and shutdown effects separately.

Exercise initial acquisition, repeated renewal, isolated and sustained delay,
true expiry, cancellation, source regression and generation changes with the
native producer and read/write consumer. Preserve original HOST anchors, exact
frame enclosure, selected age/tolerance and immutable terminal proof. Show
normal-path throughput as well as zero new sends after invalid evidence.

Then qualify the exact built driver at bounded HOME and the authorized learned
execution path with fresh environment checks. CPU tests alone cannot establish
the physical reaction bound or complete-task success. Original data/checkpoints
and the user-owned vendor worktree stay unchanged.
## LeRobot integration and pick-place continuation (2026-09-12)

The current user-selected task is bidirectional pick-and-place, not a promotion
of the older raw26 pickup probe. LeRobot 0.6.1 owns model inference, saved
processors and its action queue. FR5 adapts their exact outputs into the existing
sole executor and retains device-specific authority, cancellation and actual
transport evidence. Do not implement another policy queue or per-row publisher
to evade full-chunk admission. The current plugin deliberately cannot send;
its CLI's presence is not an execution qualification.

The existing proposal bridge now optionally records a candidate before numerical
validation through the existing EvidenceSink. It preserves exact raw and already
postprocessed chunk bytes without a second processor call, originating observation
identity/timestamps, projected candidate and the original validator result. A
distinct read-only proposal diagnostic replays that validator. It does not invent
a plan, terminal trace or manipulation failure from a velocity rejection.
`NOT_ATTEMPTED` is explicitly scoped to the proposal builder, not a claim about
all commands in the containing run. A storage failure cannot turn rejection into
approval; an unmatched candidate remains incomplete. Event hashes/order detect
accidental changes, not authenticated provenance or power-loss durability.

Read retained attempts with the existing plugin module entry point:

```sh
direnv exec . python3 -m lerobot_strategy_fr5.evidence /path/to/sidecar.jsonl
```

This is the bridge/diagnostic slice, not the full closed loop. The rollout CLI
still needs the existing executor connection. Failure-conditioned pick-place
recollection must preserve both original endpoints, direction and task binding;
the existing pickup-only guard cannot simply be removed. Failed policy actions
are diagnostic data, not automatically expert training targets. Model comparison
uses the approved TRAIN-only normalization and fixed heldout cohort, separately
from online execution qualification. Old data, processors and approval artifacts
remain unchanged.

Offline J6 deviation, gripper error and rejection fractions diagnose hypotheses;
they do not qualify or disqualify a whole checkpoint for physical evaluation.
Additional learning continues independently rather than becoming an execution
dependency. The next current proposal is checked by the existing execution owner.
The `.03` plugin default, `.1` proposal scaling ceiling and five-second finite
budget are selected software contracts, not demonstrated manufacturer limits.
Revising a consumption contract requires explicit semantics and focused evidence,
not an offline-score threshold or silently bypassing the current validator.

For evaluation, the [official SmolVLA results](https://huggingface.co/blog/smolvla)
compare task success, completion time and completed tasks within a fixed time
window. Those are useful complementary outcome measurements, not FR5 admission
thresholds. Until real FR5 outcomes exist, the relationship between offline
action error and task success remains unmeasured here.


## Original-transition recollection (software integration)

### Canonical Scene slots remain part of learned execution scope

The actual native r10 plan-only call exposed a stale three-key Scene allowlist:
the ordinary resolver already supplies release and previously landed source
slots. Preserve that canonical binding in the learned plan, precommit digest and
task grant, including the release slot's robot identity. The existing executor
consumes a source slot once after approval through Scene CAS; subsequent learned
completion/failure uses that consumed revision, not the earlier plan revision.
Merely retaining a release slot must not emit an ordinary recycle plan summary
or release evidence. Finite learned completion/failure keeps object state UNKNOWN;
qualified mechanical placement remains a separate measured effect. No new Scene
store, motion owner, tolerance or authority is introduced.

The existing acquisition recommendation joins a human-reviewed finite learned
chunk to its original native v4 source program, preapproval resolver receipts,
and episode instruction binding. Retain the destination resolver receipt in the
existing preapproval producer: job metadata participates in its resolved digest,
so reconstructing it from the source job would guess historical inputs. Validate
both endpoint poses, sheets, calibration, family/region identities, object/grasp,
cameras, direction and selected motion preset against the current native catalog.
Missing historical destination or language evidence remains explicitly unavailable.

Use the existing paired DIRECT_EDIT workspace cycle for the exact directed pair.
Current Scene owns the first pose only. If recovery changed that pose within the
source workspace, reserve the next two native cycle edges to return to the original
source, then propose its original destination; reject insufficient caller budget or
unsafe native yaw transitions. Original releases that change yaw are explicitly
unsupported by this yaw-preserving Collection authoring path; never substitute a
next-source yaw reset for the original destination. A different current workspace requires qualified
reposition/selection by the existing operator, not a rewritten failure condition.
The normal advice choice re-reads evidence and the normal compiler consumes the
same paired draft. No semantic classification, Scene writes, execution, data
mutation or approval authority is added. A failed finite chunk remains a reviewed
re-demonstration hypothesis with unknown complete-task effect and data deficit.

## Opening-coordinate model: bounded qualification transfer (2026-09-13)

The `opening-coordinate-r001` A/B motion qualifications reuse their respective
`demonstration-rhythm-r001` predecessors' unchanged physical scope. Their
`qualified_at` is this compatibility assessment time, **not a new physical trial**.
The selected successor is `fairino5_v6_gripper_opening_r001.urdf`
(`sha256:6a56a1b42bd52e4a341e90ded1511cd2d16c1bc1bdfb89454a6a38e06f946a9e`),
bound through `fr5-lab-a-tcp-r002-home-r002-opening-coordinate`. Original model,
home, qualifications, data, saved processors and checkpoints remain unchanged.

Basis: the native CPU replacement checks establish exactly four fingertip
origin/axis changes, unchanged arm/TCP, limits, jaw midpoint and collision-box
dimensions, and unchanged raw command/feedback coordinates. The existing
`collection-production-rhythm-ab-r5-campaign-0001-run-1-e12`, `e13`, `e14`
episode ledgers were reopened and validated: COMPLETE, LANDED, semantic PASS;
close command `0.01176 m`, contact and post-lift feedback `0.01197–0.01218 m`.
Their ledger digests respectively are
`sha256:cb4fa0cc2eb6fbbb2e77c9b04af77993420a234f7e2217e9f15d04584250fb02`,
`sha256:30a1cfad579bae70f7c4e3607569289b5896f19a035de82cec87f16fb0165037`,
`sha256:993cb1b9336eeb627f17a8752f64d3b8c48b85706f6139b5c1bd429b1a5fb29a`.
The original staged-release `gripper-staged-release-20260903-r002` evidence
also records separate 60% then 100% opening commands with a 0.579 s hold.
This supports reusing the unchanged 24 mm grasp, feedback band, force/velocity,
release timing and arm-home recipe; no repeated Collection or aperture survey
is required solely to correct the model's reversed coordinate expression.

Historical collision verdicts do **not** transfer. Every new plan still uses
the exact selected model and current Scene/hardware/collision admission through
the sole executor. Calculated jaw gaps are not new measured aperture evidence;
arbitrary learned contact, attachment/slip and complete Pick outcome remain
UNKNOWN until observed. The inactive candidate and its `TEST_ONLY_PLAN_ONLY`
trial API stay non-executable: normal model selection and newly bound plans
consume these successor qualifications, never a live trial bypass.

## Execution granularity and terminal observation (2026-09-13)

The retained `learned-scoped-return-20260913-r7` preapproval contains 50
policy rows compiled into 50 ARM and 22 GRIPPER segments. Its first four ARM
segments hold the same `0.021 m` gripper reference. The executor records two
completed segments before `LEARNED_TERMINAL_STATE`; this is not a successful
Pick. The rejected terminal sample and exact third action-result time were not
retained, so later recorder convergence cannot retrospectively prove that sample.

**Accepted responsibility correction:** evidence sampling granularity must not
force Python-level completion barriers at every ARM reference. LeRobot owns
policy inference/processors and, when selected, its existing async/RTC machinery.
The existing JTC owns timed waypoint interpolation. FR5 retains whole-proposal
admission, robot-specific command semantics, collision/contact scope, one active
execution owner and cancellation. Monitoring requests cancellation through that
owner; it does not become a second actuator or authorize later commands.

The [official LeRobot async example](https://huggingface.co/docs/lerobot/async)
separates prediction from action consumption. Installed LeRobot 0.6.1 additionally
exposes `predict_action_chunk`; its `select_action` queue pops are not physical
completion evidence. [RTC](https://huggingface.co/docs/lerobot/rtc) handles overlap
between successive chunks, not the current first-chunk terminal-observation bug.
The [Jazzy JTC](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
already consumes timed multi-waypoint trajectories; no replacement high-rate
Python controller is required.

The user clarified that the product requires continuous learned rollout, not a
separate finite-playback product. The proposed 72-to-45 grouped-ARM mode is
therefore **not adopted**: it preserves the same serial runtime abstraction and
would add another maintained selection. Remove that unpromoted implementation
candidate rather than expand it. Reuse native LeRobot prediction/chunk-consumption
facilities and existing FR5 transport/controller capabilities for the normal
runtime; retire row-terminal barriers from that path. Preserve historical evidence
readability without retaining a second executable playback product merely for
compatibility. Immutable command identity and admission remain useful, but must
not dictate row-by-row execution. Connect evidence produced by the actual owners;
do not build an unused parallel execution architecture to produce that evidence.
Do not silently discard gripper changes or claim concurrent actuator behavior
before establishing the actual native contract.

The remaining gripper barrier must be attributed rather than declared a hardware
law: the pinned SDK and [manufacturer peripheral API](https://fairino-doc-en.readthedocs.io/3.9.7/SDKManual/CPPRobotPeripherals.html)
define `MoveGripper(block=1)` as non-blocking, already selected by our worker.
The SDK wrapper forwards that flag to XMLRPC; our ROS patch explicitly pauses
the ARM stream until gripper completion (servo-lifecycle change `056f688f`).
Neither non-blocking RPC return nor that code comment proves concurrent physical
servo/gripper behavior on the installed controller. Before removing this barrier,
attribute the native servo-mode interaction using existing runtime evidence;
do not add a duplicate Python scheduler or silently reinterpret command order.

Historical attribution found `_restart_servo_after_gripper` in `55ffaf0` and
pause-until-completion in `056f688f`. The pinned SDK `0553c35`'s `MoveGripper`
wrapper sends XML-RPC without locally changing the ARM servo session. Archived
September 2 UDP errors also occur during startup before any gripper command;
they do not establish that gripper overlap caused the servo failures. Accordingly,
pause-until-completion is an unverified integration assumption, not a manufacturer
requirement. Its current protective behavior is not removed by this attribution.
The native ARM writer and non-realtime gripper worker are reusable interfaces;
their simultaneous physical operation on this controller remains UNKNOWN. A
future bounded overlap check must distinguish continued hold packets from actual
ARM motion, RPC acknowledgement from gripper completion, and native rejection
from application-inserted pause/restart.

The shared hardware observation consumer must not require ARM suppression to
recognize a still-active gripper generation. In its explicit pending-observation
mode, an unpaused producer between RPCs is progress, not completion. Generation,
source freshness, incarnation and faults remain checked; normal readiness and
completion modes still reject incomplete gripper work. This does not introduce
a concurrent-command authority or qualify the isolated overlap candidate.

Scene uncertainty is not an automatic consequence of task failure. The sole
executor preserves Scene bytes when its complete attempt history positively
establishes that no transport dispatch was entered. It records
`NOT_UPDATED/NO_DISPATCH`, leaves Cell blocked, and never restores an old bound
snapshot over a newer Scene revision. Entering transport remains ambiguous even
without an acceptance response; continuation preserves earlier dispatch history,
and missing history cannot claim zero dispatch. Once motion may have occurred,
this narrow correction does not assert that the object stayed put. Existing
object-effect/outcome evidence or a new human observation must resolve that case.

Terminal handling must retain the exact action result and the coherent native
observation actually used to confirm completion. A fresh buffered sample is not
necessarily ordered after that result. Existing v5 delivery/source-progress
evidence does not prove physical acquisition after the result; do not introduce
such a claim or require a later host receipt when a valid endpoint is already
available. After action success, an otherwise valid but off-target native sample
means endpoint confirmation is pending within the original deadline, not an
immediate physical-failure verdict. Fault, stale evidence, malformed data,
incarnation changes and supersession still fail through existing checks.
Persistent endpoint mismatch must expire with its actual witness retained.
No tolerance increase, timestamp renewal, new settling budget or retry follows.
The executor consumes the checked witness rather than replacing it with an
unrelated second snapshot. Cancellation cannot relabel an already-successful
action as canceled. CPU checks establish software behavior; fresh bounded
physical execution and Pick/release outcomes remain unproven.

### Native rolling-trajectory migration

#### Nonblocking geometry consumer: bounded source evidence

The existing native-pair checker makes `1 + 6N` synchronous state-validity
service calls. Those calls spin callbacks but retain the sole Python command
owner until the entire batch ends; placing them on each revision would again
couple geometry work to lease/cancel arbitration. Installed native MoveIt C++
provides local PlanningScene/RobotState collision and FK; its Python bindings
are not installed. The selected implementation direction is one task-owned,
node-less native batch helper over a detached full scene, with completion polled
by the existing owner. It has no actuator or Scene-write authority. Initialize
the bound model before fresh inference inputs; preserve world, attachments, ACM,
transforms, octomap and padding/scaling rather than reconstructing a weaker scene.
The owner must still bind checked geometry to the actual selected reference and
current dispatch conditions. Normal task start now polls full-Scene/model
acquisition and native helper initialization through the existing transport.
`OPENING` becomes `WAITING_FOR_POLICY` only on initialization completion; neither
status proves dispatch. The transport now consumes a geometry-checked relative
template through its native pair submission seam; normal executor/caller
selection, auto-commit and queue-ACK wiring remain unconnected.

The Scene and model parameter services are read independently, not claimed as an
atomic snapshot transaction. The captured MoveGroup URDF is checked against the
qualified task model; its actual SRDF and full Scene initialize the CPU helper.
A detached scene adds missing qualified floor/wall/source geometry from existing
task inputs while preserving all native obstacles, attachments and settings.
Conflicting existing objects/attachments or permissive contact overrides reject;
they are not silently replaced. Existing pose-roundoff equivalence is reused.
No shared PlanningScene apply operation is introduced. Read clients and the CPU
helper remain task-owned and close through the existing fence/drain path.

Current admitted execution, monitoring and cancellation continue while the next
revision's geometry request is pending. Only the new revision waits for its
result. The immutable request/result must match the selected revision, full
Scene/contact model, sampled states and relative timing; the owner rechecks
current authority/Scene/state at commit. Mismatched/expired results are discarded,
not relabeled, and the owner prepares a fresh check where still applicable.
Initialization belongs before fresh inference acquisition; JSON/CDR, collision
work and contact classification run off the owner loop. The CPU child has no
ROS node, command or shared-Scene mutation authority; owned teardown is pollable.

Geometry checks a zero-epoch template with unchanged ARM-linear/gripper-next-point
references. It is not directly dispatchable. At commit, the existing owner supplies
the current Scene identity and an explicit common start epoch; the transport binds
`selection_revision + geometry_binding + template_digest + epoch + executable_pair_digest`
into the actual controller revision. Only that epoch is filled, never point times,
positions, order or derivatives. The original check deadline still applies through
the final owner guard. This separates geometric relative-time validity from the
actual submission epoch rather than renewing an expired check or freezing a start
time before asynchronous work. No arbitrary controller lead time is selected here;
current-state/start validity and real native acceptance timing remain to be qualified
by the normal consumer. Native send failures preserve the executable revision
identity without acceptance, reference-consumption or physical-outcome claims.

The contact component must explicitly assign prospective hypotheses to selected
reference samples; the query binding covers that assignment and every sample.
The transport neither selects a convenient passing hypothesis nor demands all
three everywhere. A normal applicability rule is still engineering work: aperture,
queue progress and controller completion are not observed object disposition.
Do not import the historical stationary-close barriers to supply this rule.
OneJob/PickupExecutor retain task/start/commit/fault/close decisions; inference,
queue, geometry, actuator and reference-progress computation remain with their
existing components. No new lifecycle framework or caller-side motion authority.

Do not infer seamless future-tail blending from the word "replacement". The
[official Jazzy JTC documentation](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html#trajectory-replacement)
states that the implementation forgets the old trajectory. Installed 4.40.1 and
the separately staged result-delivery fix do not by themselves qualify smooth
retargeting. Keep the plain baseline and verify the actual handover semantics
before adopting a rolling replacement strategy.

Two CPU-only r7 probes support this choice, not physical qualification. Existing
bounded gripper projection exactly reproduced retained actions. Across 301
sampled states, source/carried hypotheses had no native collisions, but a
base-link-attached released proxy collided with the conservative floor at every
sample (about 1 mm). A separate private-world-object variant removed that
representation artifact without changing floor tolerance or ACM. A synthetic
cube placed against a fingertip at an actual retained joint state still produced
four native top contacts. Thus a stationary released model belongs in the private
scene world, not on the robot; do not add a floor-contact exemption. This does
not choose a phase detector or require all three hypotheses at runtime.

Each probe ran one stored batch of 903 native checks: roughly 41/44 ms excluding
model initialization (1.56/1.31 s). These are single-run feasibility observations,
not latency guarantees. The original Python classifier took about 1.97 s for
1,204 contacts because it rebound the full context/plan per contact; the new
owned batch classifier performs that binding once without changing predicates.
The native helper now maps native-to-ROS body types explicitly (WORLD/ATTACHED
enums differ), retains contact saturation, and the adapter classifies released
WORLD_OBJECT side contact without relaxing top-contact checks. Production still
needs revision/commit identity checks and normal policy submission wiring, plus
actual runtime verification of the newly connected full-scene acquisition path.
CPU tests are not a control-cadence or loaded-system latency guarantee.

Evidence is retained in `.agent-local/work/lerobot-fr5/` as
`request-geometry-batch-probe{.py,.cpp,-result.json}` and
`request-geometry-private-world-probe{.py,.cpp,-result.json}`. Both use r7
`preapproval_evidence.json` SHA-256
`8e8dfba201ff735e2085a8bdceb15b744485f526ff4e05fa61f97b080f26f02c`.
The original negative probe's enum adapter is not reusable for general contact
classification; its sole forbidden proxy/floor result is unaffected, and the
variant maps enums explicitly. Current local SRDF, default ACM and reconstructed
floor/wall/source geometry are not evidence of a retained live full scene.

#### Baseline boundaries

Finish one plain normal baseline and a bounded physical rollout before adding
execution optimizations. Prediction horizon belongs to the saved policy;
execution/commit horizon belongs to the selected policy-row prefix; controller
submission horizon belongs to the exact native trajectory pair and its timing.
These may coincide in the baseline but SHALL NOT be inferred from one another:
the initial anchor is not a policy row, and controller acceptance is not
consumption. Preserve row identity at that existing boundary so a later short
rolling horizon, tail replacement or overlap blending can be evaluated there
without rebuilding inference or adding another executor. Do not implement a
strategy registry, scheduler framework or additional prediction model now.

Bind the normal task's source/destination, Scene and saved policy configuration
once, independently of generated action rows. The same OneJob/PickupExecutor
owns that plan and its existing task grant; revisions bind their own selected
rows, native timing and consumption facts. Do not manufacture a finite plan,
precommit trajectory report or row-terminal history to enter the normal path.
Configuration admission alone proves neither current hardware readiness nor
permission to submit an unchecked revision.

Successive policy runs reuse that runtime, not independent actuator stacks.
Each has its own run identity, policy state, recording and outcome. Previous-run
optional semantic review, diagnosis and model comparison are not prerequisites
for the next run. Actual motion ownership must be settled and the next physical
starting condition established; a policy's completed queue is not object-pose
evidence. Preserve the existing recorder's durability and bounded resource
contracts until an actual sealed-record handoff supports asynchronous saving;
this decision does not claim that current finalization already overlaps runs.

The selected rhythm40 checkpoint is trained for bidirectional pick-and-place,
not Pick alone. Normal rollout must let the policy consume fresh observations
through pickup, transfer and placement under its SOURCE/DESTINATION instruction.
A qualified mechanical release/reset is a separately attributed recovery or
next-attempt preparation, not a substitute for the policy's placement or evidence
of policy task success.

Use the existing six-joint JTC's `FollowJointTrajectory` endpoint for rolling
ARM updates, under `RosMoveItTransport`'s same exclusive motion slot. JTC and
the existing hardware writer retain interpolation/ServoJ ownership; do not add
a Python row-level ServoJ controller. MoveIt remains the collision/admission
dependency, not an additional rolling-goal execution manager. LeRobot retains
inference, saved processors and selected native chunk/RTC behavior. Admission
must bind the actual executable timing and suffix, not just the unmerged output.

`motion/arm_stream.py` is the non-blocking native-handle part of this migration,
not another selectable playback product. It retains pending/current/predecessor
handles, exact guarded goals, the original deadline and native cancellation
responses/results. Accepted replacement is not RT adoption or predecessor
completion; neither canceled action status nor an empty handle set proves a
physical stop. Normal OneJob/LeRobot admission and evidence consumers are not
yet connected to this port, and physical ARM/gripper coexistence remains unresolved.

The transport now exposes one composite `ActuatorStream` rather than the unused
ARM-only public port. It reserves the same exclusive transport slot and owns two
subordinate native handles for the existing disjoint ARM and gripper controllers.
One guard validates the exact pair before either native send. Both trajectories
share an explicit start stamp; that is not proof of atomic server acceptance or
physical synchronization. Each native acceptance/result remains separately
observable, and a partial failure fences both channels while retaining ambiguous
handles and actual send-call counts. No row completion barrier, second command
producer, inference scheduler or cursor advancement is introduced. This software
seam does not itself qualify concurrent hardware commands; normal task admission,
driver coexistence/retargeting and full 7D consumption still require integration.

The native-pair geometry consumer checks the actual position-only ARM-linear /
gripper-next-point references through the existing whole-robot MoveIt validity
query. It includes interval entry, where the new gripper reference is already
selected but ARM remains at the preceding point. A unified seven-axis linear
interpolation can miss a collision at that combination. The detached report
binds the original plan and exact pair (including timestamps); it neither sends
goals nor changes Collection interpolation. This is sampled reference geometry,
not physical tracking or close/carry/open qualification. Normal stream admission
must still supply its current Scene/contact context and consume this report;
the old stationary, source-only contact checks cannot authorize a whole learned
pick-and-place horizon merely because this reference check passes.

The installed JTC `4.40.1-1noble.20260615.171409` predates the native preemption
result-delivery fix; its binary retains the old `Current goal cancelled due to
new incoming action.` branch. Waiting for its lost predecessor result could
stall subsequent rolling updates. Reuse the official Jazzy
[fix `4ab98223`](https://github.com/ros-controls/ros2_controllers/commit/4ab98223c2377857a8568eb111344e46404ae223)
(released in 4.41.0) and its `preempted_goal_receives_aborted_result` regression
instead of inventing a terminal result or weakening ownership. Mocked fixed
controller results do not qualify the installed binary. Stage/verify the native
correction before live selection; do not hot-replace the active robot stack.

The staged 4.42.1 release library (SHA-256
`38539ff288bf4328d3047cc9cb386ba83f59f57d82051ab553e5693a756cc5fd`)
passed the upstream preemption regression using vector-backed fake joint
interfaces, not the FR5 hardware plugin. This verifies the packaged native
result-delivery correction, not live deployment or physical behavior.

The ARM port retains goal-UUID-bound action feedback separately from acceptance
and terminal results. The native source stamp and `desired.time_from_start`
remain controller-reference evidence; host receipt time is recorded separately.
Feedback callbacks only copy the latest sample per owned handle. Polling exposes
the number of coalesced samples rather than claiming a complete feedback stream;
the existing recorder remains the state/action time-series owner. Neither this
feedback nor ARM-only progress proves full seven-dimensional command adoption,
gripper completion or task success. Diagnostic projection failure is reported as
unavailable feedback without becoming an actuator cancellation condition.

`rollout.stream_progress.ReferenceProgress` reduces the two native actuator
references to the prefix whose policy endpoints both timelines have crossed.
The time-zero initial anchor is not a policy row: acceptance and elapsed zero
acknowledge zero rows; crossing the first policy endpoint acknowledges one.
The slower ARM/gripper reference determines the cumulative count, not wall
time, feedback frequency, queue merge, or physical target arrival. Delayed or
coalesced feedback may advance several rows together without terminal waits.
Missing feedback cannot advance the prefix and does not itself cancel motion.
A positive native terminal may reconcile only its own actuator's final prefix.
The existing native queue lock atomically compares row identity, observation
provenance and raw/processed values before advancing an unchanged prefix, even
after native append/compaction. This adds no producer scheduler or second cursor.
These tested seams still require the normal execution-owner caller below.

For the first native async integration, RTC guidance is not required: installed
SmolVLA `supports_rtc()` returns true independently of `RTCConfig.enabled`, and
the native `RTCInferenceEngine` appends chunks when guidance is disabled. The
earlier capability-based incompatibility assumption is disproved by actual
native factory/thread tests with synthetic sampling; no admission patch follows.
`start_with_acknowledged_queue` only replaces a fresh paused engine's empty queue
before observations/resume, because pinned 0.6.1 exposes no queue factory. It
does not copy the producer loop, change merge policy or install a robot consumer.
Native pause does not cancel an in-flight inference, and stop's timeout is not
proof of thread exit; the task owner must retain resource ownership until actual
termination. Normal task-lifetime loading, observation binding and full 7D
reference adoption remain integration work, not completed by this seam.

Task-lifetime inference now retains the actual sampled observation's existing
canonical digest and source timestamps with each raw queue row. In pinned
LeRobot 0.6.1, native feature construction reads that owned observation mapping
on the producer thread, and the same thread's native merge consumes its immutable
identity. A later observation notification cannot relabel an in-flight output.
The native inference-start cursor read clears the prior thread-local identity:
a failed inference followed by an unwrapped input cannot inherit the old label.
This adapter does not replace the inference loop, processors or merge policy;
it must be revisited if native field-read semantics change. Provenance-required
mode currently supports non-guided append queues only, not RTC-guided blending.
These associations neither authenticate sensor timestamps nor prove command
consumption. The actual 7D motion owner and normal runner integration remain open.
Native 0.6.1 resets its error counter before merge, so a rejected publication
does not necessarily make `engine.failed` true. Missing provenance publishes no
rows; caller stop/deadline ownership is not replaced by that native error flag.
