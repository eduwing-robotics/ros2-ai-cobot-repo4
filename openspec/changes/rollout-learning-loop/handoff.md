# Coordinator handoff — 2026-09-09

## September 13 returned-user boundary on main0259002

The user's **복귀** and subsequent camera-check/continue instructions supersede
the physical pause below prospectively. Existing hardware, scene, human,
exact-plan and physical-binding gates remain. Manual UP framing restoration
ended with a raw frame visually close to TRAIN0/31 and EVAL38; this is neither
calibrated tolerance qualification nor evidence that framing caused stale input.

The first returned-user native call took478ms and failed the unchanged300ms
input-age gate before planning. One bounded offline diagnosis then observed
164/154/152ms on one saved observation, whose input differs from the failed live
input. The next fresh plan-only attempt used the unchanged rhythm40 12k
checkpoint, original URDF and exact selected v5 plugin/SDK. Readiness passed
10.0065s/4004samples. All50 native outputs took169.110ms; oldest input age was
125.846ms before inference and295.097ms afterward. Eight parent checks passed,
but normal `OneJob.plan_only` returned `LEARNED_STALE_OBSERVATION`. The final
child gate timestamp and intervening stage costs were not retained; passing
parent checks does not qualify complete observation-to-plan latency.

There were zero learned/gripper goals and no recorder, dataset or Scene writes;
task outcome remains `NOT_EVALUATED`, physical outcome UNKNOWN. Separate
bring-up activated hardware/current-position holds. Owned model, robot and
camera processes exited; their three visible Orca tabs were closed afterward.
Exact logs, source identities, final native output and limits are retained under
`.agent-local/work/lerobot-fr5/owner-20260913/` in
`post-restore-{native-plan.log,runtime-identity.json,boundary-result.json}` and
`latency-diagnosis-result.json`. The historical slot run ID was preview scope,
not authority for a fresh physical attempt.

No automatic retry, threshold relaxation or runtime architecture change follows
from this result. User-requested primary-source architecture assessment now
consumed this natural failure through Task `task_94de9a3c7b88`, result
`msg_8ae27ef04576`, accepted by the pinned Kernel. Its scoped conclusion is
**REOPEN temporal responsibility**, provisionally retaining LeRobot inference,
OneJob and the sole executor. The same completion timestamp underlies seven
parent checks; they are not seven fresh admission-time observations. Only4.903ms
remained after inference. Final rejection timing is missing, so transfer/copy/
validation cost is a candidate explanation rather than a measured bottleneck.
Fresh controller feedback also does not establish current visual validity of
a frozen plan. First-plan versus continuation validity requires an explicit
owner decision, not a silent change to the300ms predicate.

The comparison used official [LeRobot inference](https://huggingface.co/docs/lerobot/en/inference),
[async](https://huggingface.co/docs/lerobot/en/async),
[RTC](https://huggingface.co/docs/lerobot/en/rtc) and
[ROS Jazzy JTC](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)
sources against installed interfaces. Native async/RTC does not remove the
first-chunk delay; FR5 retiming and gripper waits make queue indices insufficient
evidence of actual completion. No runner replacement is authorized by this
review. The next consumer is the existing Rollout owner: retain exact final
rejection timing through the current failure response and decide temporal
validity scope before selecting a local repair. Full report and applicability
limits: `owner-20260913/runtime-architecture/decision.json` under the evidence
directory above. No tests, inference or hardware ran in this review. Independent
S/D software preparation carries no model-load/inference or physical
qualification claim.

## September 12 continuation: LeRobot pick-place integration

### Current boundary: Scene-slot correction CPU-reviewed; physical replay paused

The user explicitly paused all physical execution until they say **복귀** on
September 12, after the r10 probe. Do not restart robot bring-up, current-pose
holds or physical rollout in the meantime. Root's r10 robot, camera and model
processes have exited; the shared heavy lock was acquired read-only and released
to confirm availability. Independent CPU/software and Portfolio work may continue.

On main `7f6e0cc`, r10 loaded the unchanged rhythm40 12k checkpoint before
starting the bounded runtime, retained all 50 native output rows, and compiled
the canonical v4 source through `_infer_native_program`. Inference took
0.160154 s; all eight original 300 ms freshness checks passed, with maximum
observed input age 0.271183 s. The actual normal `OneJob.plan_only` consumer then
returned `PLAN_REJECTED / LEARNED_SCENE_SCOPE`, with zero learned/gripper goals,
no recorder calls, no dataset or Scene writes, and task outcome `NOT_EVALUATED`.
Bring-up issued current-position hold/lifecycle calls; this is not a claim of
zero hardware effects. Exact log:
`.agent-local/work/lerobot-fr5/native-rhythm40-probe-20260912-r10.log`, SHA-256
`20379f9859ebdd3604a3b6f42fc2c416cc07e89f5c087ea1c3d55c3fb36b6fd8`.

Source diagnosis: `run_job._scene_binding` preserves `release_slot` and, for
the current previously placed object, `source_slot`. At the r10 cutoff, the learned
branch of `PickupExecutor._compile_plan` admitted only the three basic Scene keys.
Do not discard slot fields to pass this guard. A correction must retain source
consumption/CAS, original destination and grant bindings while keeping finite
learned completion distinct from qualified recycle/known placement. In particular,
ordinary recycle-summary emission and the Scene revision used after source-slot
consumption require review together. This is a software integration gap, not an
offline model-quality gate. Existing Rollout owner received bounded CPU-only
follow-up `msg_a5a75a83eb73`, superseded route binding `msg_6201a8aead66`;
delivery alone does not prove acceptance or implementation. That unaccepted
assignment was withdrawn in `msg_a0c3ea643d38` after terminal activity could not
be verified; root implemented the correction without a parallel writer.

The candidate preserves canonical Scene slots in learned plans, validates the
release robot identity, omits the ordinary recycle summary, and uses the consumed
Scene revision for terminal UNKNOWN/fault updates. Learned failure does not emit
ordinary release evidence or allocate a destination. Focused command
`direnv exec . python3 -m unittest tests.data_factory.rollout.test_finite_plan tests.data_factory.test_scene_state`
passes **108 tests in 73.480 s**, exit0. Log
`.agent-local/work/lerobot-fr5/scene-slot-integration-focused-20260912.log`, SHA-256
`4d60d2f5982d9267d548cac694a895a23d9559fcd501f0f216948357e2d7d6f2`.
New fixtures use a temporary real SceneStateStore release/consume lifecycle and
synthetic transport, covering immutable slotted plan-only, completion/fault,
stale/foreign/repeated source claims, and malformed/foreign-robot release slots.
An initial fixture called the diagnostic before the final OneJob poll; correcting
that test ordering required no diagnostic product change.

Correction `240d47266d59a91648576b9e1799da7e19ddbc1a` is pushed to main.
Independent immutable CPU review against `b073ad4` reported **NO FINDINGS** in
Orca `msg_f7d8fa084e66`: four focused tests PASS in 0.350 s, plus four independent
falsifiers PASS. These cover grant rejection after slot mutation, unchanged
ordinary recycle summary, zero goals after a post-CAS arming failure, and refusal
to overwrite a foreign Scene revision at learned completion. Source consumption
after approval remains a durable one-shot CAS even if arming then fails; it is
not proof of physical consumption. The reviewer did not reuse root's 108-test
result or run full discovery. This closes the scoped source/CPU review, not fresh
model or physical qualification. The user's physical pause still applies.

The r9 apparent hardware-stale rejection was traced to the agent-local readiness
observer taking its clock sample before `transport.snapshot()` spun incoming DDS
callbacks. A newer callback could therefore appear later than the claimed capture
time. Moving the local clock sample after the snapshot, as the product executor
already does, passed a 10.004183 s hold on the same runtime: 3,907 samples and 976
producer progressions under unchanged 100 ms/1 ms settings. No product code or
threshold changed. Log SHA-256
`3d088306178c0259afbc9db6abd7e17b315a9b38a20e35a6d08ae25d702938c6`
at `native-rhythm40-readiness-after-snapshot-20260912-r9.log` in the same folder.
R10 separately passed 10.000398 s readiness before inference.

The earlier r8 camera error was USB device re-enumeration; the same configured
RealSense auto-recovered and three subsequently observed UP frames had mean
brightness about124.47. This is historical camera recovery evidence, not a
current illumination certificate or permission to override the physical pause.
The existing teardown fault remains separate and is not qualified by these probes.

Learning's completed paired 15k/18k results are integrated from owner `8457eae`
as `5349c90`; see `../learning-evaluation-loop/design.md`. The fixed-cohort late
curve is near-flat/slightly regressed under the inherited cooled schedule, with
a modest roughness gain. Original12k remains the reference, not a proven physical
winner; no new training fork or physical gate was created.

During the physical pause, the user requested independent Learning progress.
Root accepted the existing owner's proposal (`msg_3a67398d2b94`) in
`msg_d319051f4bf3`; the same owner's turn start was confirmed. The next bounded
experiment compares balanced TRAIN and heldout observations using the same 12k
checkpoint, processors, normalization, fixed10 and metric denominators, reporting
per-axis flow and physical-action residuals. The observation cohorts differ;
matched computation does not make them identical samples. Existing compatible
reports are reused. This diagnosis prioritizes optimization versus data/observation
hypotheses; it neither proves their cause nor adds a rollout gate. The longer
9k-hold training fork is not started. Learning owns its diagnostic implementation,
results and OpenSpec experiment record; no duplicate evaluator or owner is created.

That bounded diagnosis is now complete and integrated as `1be433a` / `e125f58`.
Fresh main `direnv exec . python3 -m unittest tests.test_offline_evaluation`
passes 24 tests in 12.824 s; strict OpenSpec validation and diff checks pass.
The same 12k checkpoint has sampled J6 RMSE 2.661 degrees on the selected TRAIN
observations versus 18.321 degrees on heldout, with large errors concentrated in
heldout episodes 35–39 and later sampled time fractions. These are descriptive
findings, not evidence of a particular cause or a physical admission condition.
Exact native execution, cohort, denominator, hashes and limits are recorded in
`../learning-evaluation-loop/design.md`. Learning continues the authorized
CPU/read-only source and nearest-TRAIN-condition investigation around 34→35
completed as owner `8afc0bf`, integrated `1115b74`. Declared yaw+35.890855° is
absent from TRAIN0–31 and occurs in heldout35–39; their middle/late configurations
are also distant from all same-task TRAIN rows. Inspected provenance did not show
a new large state/reference offset or target-outside-bracket discontinuity.
This supports condition-coverage investigation, not yaw causality or corruption.
Exact limits and next controlled coverage proposal are in the Learning design.
Root also located an older same-direction pick-place candidate (proofloop episode2,
yaw+32.014373°). Learning completed the bounded read-only eligibility audit in
`msg_f36df7dfe300`: feature/camera/gripper/frame contracts match and four late
A-to-B observations have closer state support, but early observations do not;
there is no B-to-A support. The old .03 motion recipe takes54.17s versus25.03s
for the compared rhythm40 episode, so mixing it also changes temporal progress
within the same50-row horizon. Historical semantic/technical PASS was located;
current training inventory/approval for this separate episode was not located
by the bounded search, not proved absent everywhere. Root retains it as a
future mixed-recipe candidate rather than spending the current critical path on
admission/conversion of one slow episode. First controlled coverage comparison
should use the current recipe, independently varied XY and both directions,
retaining the original heldout and a matched-update existing-data control.
This is an experiment priority, not a prohibition on older data or a new rollout
gate. No heldout reassignment, admission, new training or physical execution
occurred. Learning's assigned CPU diagnosis/reuse review is complete; no further
GPU run or broad research assignment is pending. Real execution stays paused
until the user's **복귀**. Resume the existing native physical path before adding
unrelated analysis systems; learned Pick/reset and actual recollection effect
remain unproved.

Independent review of `e125f58` found a normal flow-report TRAIN/heldout label bug
(`msg_f68b82b0df9e`), not a numerical loss change. Root correction `9396750` passes
25 focused CPU tests in12.653s; independent immutable follow-up
`msg_90c70f958031` reports NO FINDINGS with two tests in3.187s and the original
falsifier closed. The review does not independently qualify the broader native
experimental numbers or physical behavior. Public receipt is in Learning tasks.

### Original-transition recollection is now software-connected

Owner commit `a62771a85f97015524be31d7b34d6779bf359420`, integrated as
`6ceca2762b8622536a74b04dbd92dfe5ccb1d289`, retains the destination resolver
receipt alongside the existing source receipt for learned v4 preapproval.
Recommendation generation now binds the original directed SOURCE/DESTINATION
pair, instruction, object, calibration, region and qualified motion preset,
then supplies normal paired `DIRECT_EDIT` authoring. Current Scene placement is
kept separately: when it differs within the supported cycle, qualified ordinary
transitions precede replay of the original pair instead of rewriting Scene
history. Missing historical destination evidence, incompatible domains,
insufficient episode budget and unsupported rotated release remain explicit
unavailable cases. Existing pickup recommendations remain supported.

The owner ran 80 focused CPU tests in 425.353 s, exit 0, covering acquisition,
recommendation, native draft application/compilation, evidence boundaries and
three producer/preapproval checks. Its `focused.log` SHA-256 is
`166fbe6eacfa2a61a7a8c48e994c5faf4f07b68b99f57a896ffe71df838cc8ca`.
Root inspected the actual producer/consumer diff. On integrated `6ceca27`,
`direnv exec . python3 -m unittest` with
`tests.data_factory.operator.test_acquisition_advice.AcquisitionAdviceTests.test_original_transition_roundtrips_native_paired_draft_and_compile`,
`tests.data_factory.operator.test_acquisition_advice.AcquisitionAdviceTests.test_recovered_transition_keeps_current_pose_then_original_pair`, and
`tests.data_factory.test_run_job.RunJobTest.test_learned_preapproval_retains_exact_original_resolver_inputs`
passes 3 tests in 136.526 s, exit 0. The log is
`.agent-local/work/lerobot-fr5/transition-main-integration-20260912.log`.
The owner's broader unchanged 80-test scope was not duplicated; full discovery
was not run. This closes a software connection, not a physical
loop: semantic failure remains distinct from causal data shortage, and advice
does not grant motion, semantic success or training authority. No real learned
Pick or targeted-recollection effectiveness is claimed.

### Latest learning checkpoint: 18k saved; paired comparison remains separate

Learning's `fefdc323bddbca0a635c8cb7341ae66f93707288` is integrated as
`5b9e438`: recreated EVAL workers terminate after evaluation, while TRAIN stays
persistent and separate worker seeding preserves policy RNG. Root inspected
the change against the installed LeRobot TRAIN-then-EVAL construction and ran
`direnv exec . env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m unittest tests.test_training_continuation`
on integrated main: 18 tests PASS in 36.970 s, exit 0. Log
`.agent-local/work/lerobot-fr5/continuation-main-eval-lifetime-20260912.log`, SHA-256
`6935e2cee6481f463142f002473afc7e9fa00192ac7c07518bee5091d9aee83c`.
The scope covers native CPU continuation, worker lifetime, sample/RNG/state
preservation and recovery; full discovery was not repeated for this bounded seam.

The distinct `smolvla-pickplace-rhythm40-continue-18000-r2` completed with exit 0
in 1,310.215 s. Its owner result reports parent/inventory unchanged; root read
that receipt and the actual complete 18k model, optimizer, scheduler and cursor
files. The final cursor is 71,997 examples, epoch 3 offset 5,100 of 22,299 TRAIN
frames; scheduler horizon remains 12k and LR 2.5e-6. This is successful native
continuation/save, not improved policy quality or physical success. The owner
retains saved-state evidence and sequential paired 15k/18k evaluation in its
existing learning-evaluation-loop work directory. GPU comparison is independent
of the original 12k rollout candidate and does not create a rollout gate.

Root's r7 model-first probe loaded and warmed the original 12k, then waited at
`READY_FOR_RUNTIME`; no `PROBE` input or new device bring-up was issued. Root
terminated that exact idle process after storage cleanup diverted the turn,
verified PID3609616/flock3598493 gone and GPU released, then returned the resource
to Learning in `msg_8ae9f56667a6`. Sequential comparison supervisor3619012 and
15k evaluator3619077 were subsequently observed live. Do not reuse r7 as fresh
runtime evidence. Orca message transport is responding again; the earlier
disconnection below is historical, not a reason to duplicate existing owners.

Camera-only measurements at21:25 KST passed the existing brightness threshold,
but cannot authorize a later light state. Next root consumer remains a fresh
native-input normal OneJob plan-only followed by the existing qualified live
path; real inference-driven motion consumes the system's current illumination
check. No personal image judgment or offline model metric becomes a new gate.

### Latest resource result: continuation stopped; original 12k remains usable

The approved 12k-to-18k run below is no longer running. Its owner resource
receipt reports exit -15, `PERSISTENT_HOST_PRESSURE`, and unchanged parent and
approved inventory. Native updates reached 15,000 (2.69 reported data passes);
the log records heldout loss 0.6508 at 21:01:03 KST, followed by checkpoint save.
The `015000` directory contains only `pretrained_model/config.json` (2,794 bytes):
no model weights or continuation state. This is an incomplete save, not a usable
15k checkpoint, a paired evaluation result, or a model promotion. Preserve it
as failure evidence and retain the original 12k for rollout preparation.

Read-only attribution narrows, but does not yet identify, the memory cause.
At elapsed 825 s the trainer finished updates and entered evaluation; available
host memory fell from 2.71 GB at 821 s to 0.68 GB at 836 s, before the checkpoint
log at approximately 837 s. Therefore serialization alone does not explain the
initial drop. Installed LeRobot constructs separate TRAIN and EVAL loaders with
the same four persistent workers; its first evaluation starts the EVAL iterator.
Saving then calls safetensors, which retains CPU copies of non-CPU tensors until
serialization finishes. The original model's header describes 906,639,456 bytes
of tensors. These are distinct candidate contributions, not measured per-process
attribution. Optimizer saving follows model saving and was not reached in the
normal path. Do not blame data corruption, change the optimizer, or repeat the
same expensive run on the basis of this host-wide trace.

The Learning owner should first distinguish evaluation-worker/decoder residency
from model-save allocation, preserving native RNG, sample cursor and save/reload
semantics. Root did not patch that owner's code or start competing GPU work.
At inspection the two recorded training PIDs were absent and no CUDA compute
process was listed. Orca messaging independently returned `runtime_timeout`;
the follow-up was not confirmed delivered. Do not treat a message timeout as a
worker exit or launch duplicate work. Original-transition recollection work is
already assigned in its isolated worktree and was observed executing focused CPU
tests; it does not depend on another training run or a new physical outcome.

Exact local evidence is retained under the Policy Learning owner's
`.agent-local/work/learning-evaluation-loop/`: `rhythm40-live-start-r1.json`,
`rhythm40-live-training-r1.log`, `rhythm40-live-resources-r1.jsonl` and
`rhythm40-live-result-r1.json`. No learned motion or task success is added by this
resource diagnosis. The next rollout step remains fresh native input through
normal OneJob plan-only, then the existing live authority path when its actual
conditions are met; longer training is not an entry gate.

Latest r5 preparation supersedes the earlier freshness rejection below. Loading
the unchanged model before starting the bounded device runtime produced 50 native
actions with 0.158648 s inference. All original 300 ms source-age checks passed;
the oldest input was 239.782 ms at post-inference admission. This single result
does not establish sustained freshness or a thread-count performance gain.
Compilation then exposed `LEARNED_SOURCE_PROGRAM`: the canonical A-to-B source
is `fr5.motion_program.v4`, but the learned wrapper allowed only v2.

Correction `ab41b19` accepts canonical v2/v4 sources while preserving the complete
source and destination bindings and invoking the unchanged source validator.
Unknown versions and nested learned sources remain rejected. The focused
finite-plan and data-factory suite passes 113 tests in 79.839 s, exit 0.
Independent immutable review of `ab41b193cc7872db4696a298b836257fc123a3cc`
against `92bece74de157a40eaba69224d8e380ad0def091` reports no scoped findings:
95 finite-plan tests pass in 106.656 s, plus 11 independent binding/mutation
falsifiers. These are source/CPU checks, not a live replay or full discovery.
The archive verifier had to preserve ROS environment paths and immutable Git
fixture access; its earlier harness failures are not product regressions.
No execution bound, motion owner,
checkpoint, data, scene or approval changed. No learned/gripper goal was sent.
The next physical-input check must use a new fresh observation and current
runtime, not replay the saved r5 output as live authority. GPU is meanwhile
assigned to Learning's approved 12k-to-18k continuation. Owner message
`msg_5758ebcbf18b` reports the production wrapper running as PID 3416314 under
supervisor 3416141, with original data/checkpoints preserved and capacity for
the two planned saves plus its existing reserve. Subsequent owner message
`msg_95cfc7033972` confirms native optimizer updates through absolute step 12,248,
about 5.3 updates/s, restored epoch 2/offset 3,400 and the original scheduler.
No new saved checkpoint or improvement is claimed at that observation. Further training
is not a prerequisite for rollout. Evidence and exact logs remain in the local
probe report linked below.

Earlier physical-input preparation used the existing native full-chunk path, not
the separate CLI plugin. With the original 12k checkpoint and current A-to-B
instruction, fresh cameras and seven-joint state produced all 50 actions.
The explicitly selected existing integer-percent gripper conversion passed its
range/timing construction, then admission returned `LEARNED_STALE_OBSERVATION`;
measured inference was 0.158651 s. No learned or gripper goal was sent. Two bounded
v5 current-position holds passed about ten seconds each. This is not learned
task success or qualified teardown: MoveIt reported a shutdown-time fault.

That diagnostic needed to separate capture, transfer, inference and final
admission age while retaining one loaded model. Four capture-only observations
arrived with oldest camera ages about 56–104 ms; they do not pinpoint the age
overrun in the model attempt. A first local probe omitted the existing gripper
conversion; a later probe outlived its temporary ROS runtime during model load.
Those are coordinator harness errors, not model defects. Exact logs and next
consumer are in `.agent-local/work/lerobot-fr5/native-rhythm40-probe-20260912.md`.
Do not loosen a bound, select favorable noise or require further training from
these observations. Existing system authority remains the execution boundary.

The local handoff `.agent-local/wiki/FR5_LeRobot_Rollout_Handoff_20260912.md`
matched all 19 supplied source fingerprints at inspection. The chosen dataset is
the separately approved bidirectional pick-place rhythm40 set, TRAIN 0–31 and
heldout 32–39, raw RGB without augmentation. Actual training reached 12,000 steps
and printed `End of training` at 18:40:50 KST; 3k/6k/9k/12k saved checkpoints
exist. This is not evidence of task success or a best checkpoint. Learning's
paired comparison is complete: 24 heldout observations with three shared noise
seeds per checkpoint favor 12k for the next no-effect proposal check, not physical
execution. Ten of its 72 chunks still reject gripper projection, and J6 jumps
remain substantial. The existing owner now reuses saved outputs to test the
current serialized/retimed execution contract without another model load.
Canonical comparison and limits are in
`.agent-local/work/lerobot-fr5/learning-rhythm40-20260912/comparison-r2/REPORT.md`
(SHA-256 `28438da6dc9f09e18db000c5c40c5788f78d62558c2bfec5f0f1f77e21b87014`).

Root's proposal-evidence slice optionally connects the actual existing chunk
bridge to the existing sidecar before validation. A read-only diagnostic replays
the numerical validator and preserves incomplete attempts, original errors and
no-authority semantics. Independent review of `645fe03` replayed 67 passing tests
and found two evidence defects: candidate-publication failure could mask the
original validation error, and retained processed bytes were not numerically
bound to proposed actions. The correction preserves rejection priority even
when candidate storage fails, projects/retains one tensor snapshot, and replays
a shared pure gripper projection while comparing all six arm values exactly.
The corrected plugin/evidence/recommendation regression passes 74 tests, including
both falsifiers and alternate float encodings. Independent immutable review of
`6e82d5585096dfea54fae67ea2d7ac0079b44aa5` against `645fe03` reports no scoped
findings and independently passes 74 tests in 9.744 seconds. It rejects
re-digested mismatches in all seven processed columns, accepts six supported
float encodings, and verifies validator-error priority and snapshot isolation.
Whole discovery at the `6e82d55` source cutoff completed with 1,339 tests in
1,608.228 seconds: 1,335 passed, two failed and two skipped. The failures were
the missing explicit candidate SDK include and an eager-launch test fixture
after the production launch became late-bound. The fixture now evaluates native
launch declarations and the opaque setup under the mocked demo generator;
default/selected model, hardware-mode and missing-model checks remain.
Its affected suite passes seven tests with one external-SDK skip. The separately
selected SDK build test passes in 2.823 seconds using a snapshot header whose
SHA-256 `357d84c0aebb75947d24ca6bf0bff961f993b9eb576b57e2d13de859e1ac82bc`
matches the canonical SDK patch. Missing selected headers still fail; an
unselected external SDK is explicitly opt-in, like the existing SDK integration.
No full discovery was repeated or relabeled as an exact new-cutoff PASS.
These source/CPU results do not qualify hardware execution. Hashes do not authenticate
an externally fabricated sidecar. This does not connect the rollout CLI to physical execution: plugin
`send_action` remains blocked and source/destination-bound pick-place failure
recollection remains open. Do not treat proposal diagnostics as terminal
execution traces or training targets.

Follow-up `23891ce` corrects the tap's prediction/consumption-length conflation:
LeRobot's `chunk_size` defines the full prediction; native `select_action` owns
`n_action_steps` consumption and reinference. The full tensor remains untouched
and independently retained. Root and independent immutable review each pass
26 focused tests; unequal-horizon queue falsifiers pass without a model load.
No saved/runtime horizon, speed, approval or physical-execution setting changed.
The current full-chunk proposal sidecar still requires equal raw/processed/action
row counts; this tap correction alone does not implement prefix proposal binding.

Learning's saved-output serialized check is complete at
`.agent-local/work/lerobot-fr5/learning-rhythm40-20260912/serialized-feasibility-r1/REPORT.md`.
At plugin-default scaling `.03`, none of 72 sampled 12k chunks fits the existing
five-second reference horizon after valid quantization. At native ceiling `.1`,
38 pass reference timing; ten reject earlier on gripper bounds, and a separate
source-derived held-time necessary budget leaves only 24 not yet excluded.
These are numerical checks, not execution approvals or physical failure labels.
Do not select favorable seeds, discard late chunks, or relax limits to claim
success. Deterministic four-worker continuation is integrated as `f9c9c47`:
main's focused continuation/checkpoint suite passes 24 tests; independent immutable
review of owner `9da584c` passes 18 continuation tests with no scoped findings.
It preserves native optimizer, scheduler, committed sample cursor and relevant
CPU RNG history for the admitted deterministic loader, not arbitrary worker RNG
or stochastic transforms. The new continuation handoff is described above;
source verification alone does not prove that the GPU job started.
Continuation verification is independent of physical rollout admission.

Preexisting user plugin edits were retained; a recoverable source copy is under
`.agent-local/work/lerobot-fr5/pre-attempt-evidence-2UjzcV`. Original observations,
datasets, review/approval artifacts, checkpoints and vendor working tree were
not modified by this slice. The September 11 hold below supersedes older failed
clock-query hold observations only for its explicitly bounded v5 qualification.

This is a session transition snapshot, not a second runtime ledger or a fixed
master plan. Re-read source, actual process state and Orca before relying on it.
`proposal.md`, `design.md`, `specs/` and `tasks.md` retain outcome/acceptance meaning;
Orca owns execution messages. Replace stale observations rather than accumulating
another history. No real rollout success is claimed here.

## Physical coherent-delivery qualification — 2026-09-11

The v5 coherent-delivery successor passed a bounded real-FR5 current-position
hold qualification. The run used the isolated coherent SDK/plugin overlay with
`FR5_REQUIRE_GRIPPER_SOURCE_CLOCK=true` and
`FR5_REQUIRE_COHERENT_SNAPSHOT=true`.

After the post-bind bootstrap publication, the readiness probe observed 10.0007 s
of valid wire-v5 delivery with producer sequence advancing from 593 to 1841,
999 distinct progress events, connection epoch 1 and configuration epoch 0
remaining stable. The probe accepted 3,999 reads; 3,000 repeated reads were not
counted as source progress. `READINESS_RC=0` and
`NATIVE_V5_READINESS_PASS` were produced. No learned trajectory or gripper goal
was submitted by the probe.

The earlier immediate `LEARNED_HARDWARE_STALE` result was a diagnostic bootstrap
race: policy binding completed before the first new coherent publication was
observed. The diagnostic was corrected to wait up to one second only for the
first strictly newer post-bind producer sequence; the existing 100 ms live
freshness bound was not relaxed after readiness.

This qualifies bounded physical coherent-state continuity for the current-position
hold only. It does not qualify learned Pick success, physical gripper
same-command completion, mechanical task success, or long-duration operation.
Shutdown-time RViz/controller-manager context errors occurred after successful
hardware deactivation and are not counted as execution evidence.

## Current continuation — coherent delivery and minimum sufficient guarantees

The user approved separating necessary execution evidence from optional temporal
precision. `design.md` and the successor requirement define this boundary;
existing version 3/4 readers and deployed thresholds are unchanged. No learned
Pick has executed. The earlier physical certificate-renewal failure remains open.

Matching manufacturer source is pinned to `fairino-cpp-sdk`
`0553c35d760a4e76c9b8d2fc0208ca83e6d731cd`. Its receive thread updates cached fields
in place while `GetRobotRealTimeState` copies them without a shared publication
lock. This is a coherence gap, not proof of historical dataset corruption. Public
header tokens match the installed headers after removing comments/whitespace;
that is source compatibility evidence, not rebuilt binary qualification.

SDK publication/getter candidate completed: Orca `msg_c38d80b9c884`, canonical
`patches/fairino-cpp-sdk-2.3.7.patch`, SHA-256
`57cdfbfe3304dfaebb956bcc4ee3ea8d4a72b76376a020701afac8e4dea478cf`.
Worker and independent root replay each passed seven actual producer/getter
cases. The committed opt-in `tests.data_factory.rollout.test_sdk_snapshot` then
exported the untouched pinned Git source, applied the canonical patch, built it
and passed all seven cases in 11.741 seconds with zero fixture network syscalls.
Ordinary test discovery skips this external-SDK integration test explicitly.

Bounded ABI checks retain every original exported FRRobot method and matching
FRRobot/state sizes and joint/calendar/gripper offsets; they are not complete
runtime ABI or physical qualification. Root's first fixture launch resolved the
old installed SDK through inherited LD_LIBRARY_PATH and failed with the new
symbol missing. The reproducible test now pins and verifies its candidate loader
path. No SDK was installed and no robot connection was attempted.

The coherent-delivery successor has now been recovered and integrated into
main source form for CPU qualification. Canonical `patches/frcobot_ros2.patch`
SHA-256 is
`647fd7704a46445235ffed49d4406019a78eb3abb0903558336aa679d3fa094b`.
Hardware wire v5 is bound specifically to
`fr5.gripper_temporal_policy.v2`; versions 3/4 retain temporal-policy v1 and
version 2 retains the legacy source-clock contract.

Independent main-tree replay passed the external SDK seven-case producer/getter
test, 31 focused native/transport tests (one environment-dependent skip), and
71 learned plan/protocol tests. A pristine vendor `60755d44` export accepted the
canonical patch, `fairino_hardware` configured and built successfully, and
`ldd -r` resolved `libfairino.so.2` to the selected coherent SDK candidate with
no unresolved dynamic symbols. This is CPU/software integration qualification
only: no SDK install, ROS deployment, robot connection, learned Pick or physical
continuity qualification is claimed. Preserve the existing dirty
`src/frcobot_ros2` working tree; it was not used as the integration target.

The selected 4032 policy-only tree was rehashed unchanged as
`sha256:aa5ca010131f5bd00c605ab0b9017f27f2c3906c8017502a1ba1915916b8f4e7`.
It remains an available execution candidate, not a proven best model or task
success. No retraining or model load is needed merely to repeat that identity.

Correction: configured fields 3–75 include `LastServoTarget` (75), but exclude
`ServoJCmdNum` (76). Neither currently establishes a device command-ID echo.
Do not use a presumed counter as freshness or completion evidence.

## Resumed qualification result — supersedes pending observations below

The user resumed the existing Goal and authorized removing redundant procedural
constraints, not bypassing existing execution authority. Full regression completed:
**1,313 tests, 1,526.949 seconds, exit 0**. Log SHA-256:
`dcf55f0695fcca5fa998afdb74d74504985a93393d8d0f2d3b8da19bf31e5e8e`.
PID 4129485 has exited and session 79732 returned its terminal result. Do not rerun
this suite solely because the old handoff below describes it as running.

Candidate physical continuity **failed**, despite the CPU PASS. The latest
13:26:57 KST launch added passive packet capture to the same bounded hold;
it did not change the candidate or selected 100 ms/1 ms temporal policy.
Request/reply wire timestamps `1788928022.124189` / `.172747` give 48.558 ms;
libcurl measured 48.625 ms to first byte. UDP 20007 requests continued at about
10 ms intervals with replies during this wait. The original certificate then
expired at `.200083`, before the next reply at `.203951`. Hardware ERROR
deactivated controllers; Python `ROS_JOINT_STATE_STALE` was downstream. This
supports response-path latency, not a proven firmware lock, network defect or
general ROS stall.

Exact evidence:

- `/tmp/fr5-clock-attribution-XBphaW/runtime.pcap`, SHA-256
  `8d6294ef7124f9a11db2033baf87cbb0a77a6df2cb93ea60b97bdf683a8e7b16`;
  64 packets, no capture drops, bounded prefixes (128-byte snap length).
- `/home/codelab/.ros/log/2026-09-09-13-26-57-494466-codelab-System-Product-Name-20582/launch.log`,
  SHA-256 `a626a2102bb8de4742a5d8d64c3cb1095caf6745a5701b44bce14051b68a0764`.
- Orca `msg_a5d5e663737b`: independent source/binary audit of the same cutoff.
- `/tmp/fr5-clock-attribution-XBphaW/sdk-readonly-config.log`: actual SDK CNDE
  readback, period 8 ms, fields 3–75. Before/during cached reads: 64/64 each,
  maximum 2.06/2.57 ms. After CloseRPC: 30 PASS / 34 timeout at a diagnostic
  25 ms query limit, exit 1; do not hide this counterexample or attribute it to
  ServoJ, which was not called. SDK state reads had no errors. Root cause of
  the post-close result remains unestablished.
- `/tmp/fr5-clock-attribution-XBphaW/clock-only-100ms.log`: subsequent separate
  process without SDK/ROS, 96/96 replies, mean 1.61 ms, maximum 2.28 ms,
  including three server-directed connection renewals. The diagnostic 100 ms
  wait did not alter the production policy. The native probe prints failures
  without a failing exit code; the 96/96 claim is from its records, not exit 0.
- `sdk-strace.*` in the same temporary directory confirms installed TCP 20005
  receive and server `Connection: close` after batches; tracing adds host
  overhead and its timeouts are not production latency evidence.

The bounded stack ended; latest PIDs 20810/20811 and earlier trial PIDs were
confirmed gone, with no remaining controller TCP connection. No learned or
gripper goal was submitted. Native activation/current-position hold did occur.
Do not promote this candidate as physically qualified or repeat it unchanged
without a new discriminating observation.

Launch used the temporary candidate package overlay and must explicitly set
`FR5_REQUIRE_GRIPPER_SOURCE_CLOCK=true` inside that shell. A preceding trial
omitted the opt-in, loaded the correct library but rejected the undeclared policy
parameter; that was not a renewal experiment. `/proc` mappings verified the
candidate library. Main and rollout-evolution installed libraries remain untouched.
The private readiness helper now adds the repository import root and allows 20s
for initial ROS discovery only; all selected freshness bounds remain unchanged.

The next engineering question is why acquisition cannot renew the original lease
during real hold, and whether the selected certification mechanism's cost and
fault propagation fit execution needs. Preserve raw evidence and distinguish
valid old evidence with renewal pending, true expiry, and actual controller fault.
Do not treat another timeout increase or another framework as an established fix.

## Outcome and nearest critical path (original transition snapshot)

Produce an actual trained Pick, qualified mechanical release/reset, retained
diagnosis and a consumed targeted-recollection path, then learning/portfolio
evidence. A software-only green result does not complete that loop. Optimize
total engineering and operating cost, not minimum diff size or extra edge-case
count. Reuse LeRobot where its actual interface fits; retain FR5 execution and
authority responsibilities. Do not introduce more research/features merely to
avoid the next valid physical result.

The immediate bottleneck was native controller-clock acquisition/renewal during
current-position hold, before any learned/gripper goal. A candidate correction
has passed focused tests and independent review. Full regression is still running;
candidate physical timing, cancellation and continuous hold are not qualified.

## Immutable software cutoff and preservation

- Candidate implementation: `e7f22e7646db8a4741307deb45fc4a7875f51471`, parent
  `b9931d661a634ad71be6f80c60c183dc40daca87`.
- Canonical `patches/frcobot_ros2.patch` SHA-256:
  `704f8ad4b62ebda4a56a061627364681dec761131c68d9b46adc3465c37c591a`.
- Main: `/home/codelab/Desktop/Project/fr5_ws`. Only pre-existing vendor
  submodule dirt was observed. Do not reset, stage or overwrite it.
- Vendor HEAD: `60755d44d521a5ad6bee8494cc19522f8801aa20`;
  `git -C src/frcobot_ros2 diff | sha256sum`:
  `94238e59939aa7fc4f84558bd80280c9c66312900e447bfaf8ef82e0af2b20d1`.
- Immutable datasets, saved model, calibration and installed runtime were not
  changed in this correction. Verify remote HEAD before claiming publication.
- Use `direnv exec .` for repository Python/ROS commands. `.envrc` matched
  origin/main; never auto-trust changed content.

## What changed, and what did not

`refresh_gripper_freshness` spends the remaining original acquisition budget
across both clock queries and frame waiting instead of prematurely imposing a
quarter-age timeout per query. The worker schedules renewal relative to the
original acquisition start, rather than adding a fixed sleep after a long query.

Budget-only correction was falsified: intermittent 2/2/37ms CPU query schedules
still lost continuity before the scheduling correction. Final CPU tests preserve
sends for that schedule and stop subsequent sends under expired or sustained
slow schedules. The measured historical wire tail was 36.838ms while the client
closed at 25.151ms; that one sample does not prove sustained latency bounds.

Original host/steady anchors, exact source enclosure, selected 100ms age/1ms
tolerance, generation/incarnation, immutable completion proof, cancellation and
single owner remain. These age/tolerance values are existing selected policy,
not asserted manufacturer physical-safety limits. No old evidence is relabelled
fresh and no error is suppressed. Continuous slow acquisition may still exhaust
the original lease and must stop.

Independent review notes: longer acquisition can delay recognition of a command
deadline because refresh precedes the deadline check. Late completion probes
produced no completed proof; guards remain before/after resume. Real libcurl
cancellation latency and hardware stop/hold behavior remain unmeasured by these
CPU mocks. Review grants no motion authorization.

## Verification and the live test process

Focused command:

```bash
direnv exec . python3 -m unittest \
  tests.data_factory.rollout.test_current_bracket \
  tests.data_factory.rollout.test_gripper_evidence \
  tests.data_factory.rollout.test_native_policy
```

- Root: 43 passed / 24.801s / exit 0.
- Independent reviewer: 43 passed / 24.605s / exit 0, plus adversarial
  producer/sampler/write probes. CPU-only, no repository edits or hardware access.
- Root repeated continuity/expiry stress: 20 passed / 10.306s / exit 0.
- Canonical patch checks against pristine vendor index and native CMake build
  passed. `git diff --check` and OpenSpec strict validation passed.
- MEX graph refreshed: 8,156 nodes, 28,373 edges, 336 files. `mex check` score 91;
  three existing parser warnings treat Status/Decision/Reasoning prose as
  dependencies in `.mex/context/decisions.md`; not new code failures.

**Do not start another full suite.** At handoff preparation the following was
observed live (about 13 minutes elapsed), with no terminal summary yet:

- PID `4129485`, parent `4129481`; unified-exec session `79732` in old session.
- Log: `.agent-local/work/full-regression-e7f22e7.log`.
- Command: `direnv exec . python3 -m unittest discover -s tests`, piped through
  `tee` with shell `pipefail` enabled.

New session may not inherit the exec handle. Check the exact PID/command and log:

```bash
ps -p 4129485 -o pid,ppid,etime,stat,rss,args
rg -n '^(Ran |OK$|FAILED|FAIL:|ERROR:)' .agent-local/work/full-regression-e7f22e7.log
```

If handle observation fails, verify process/log rather than restarting. A missing
process without a completed summary is not PASS. Attribute failures against this
candidate and baseline. Prior full-suite PASS was d56f0ef (1,309 tests, 1,557.102s),
not evidence for e7f22e7. GPU training was not started by this coordinator.

## Isolated candidate installation and rollback boundary

Source was archived from pristine vendor HEAD, canonical patch applied only in:
`/tmp/fr5-native-e7f22e7-aaZFgf`.

All native package targets built with one compiler job. CMake installation was
performed **only into that temporary prefix**, not either workspace installation.

- Build library SHA: `2c6cf5b7be06d35cdab2709095c06271861155501b62225b191ce09b76b3cf0c`.
- Installed candidate:
  `/tmp/fr5-native-e7f22e7-aaZFgf/install/lib/fairino_hardware_v3_9_7/libfairino_hardware.so`.
- Installed SHA: `bc13fdb160727f97f5d0d1a452e6bb768f5472461f75381528fbdd83346d5ca3`.
  CMake strips build RPATH on installation; these are deliberately distinct hashes.
- Existing main installed library SHA:
  `64e2870229d59fd768ef69801666741ca007385a9f2ee8b1660fe6f208db775b`.
- Existing rollout-evolution installed library SHA:
  `3307789b98ba08c57fd077eb86df48610ec655ad204a30d8281eddba36967b77`.
- SDK `libfairino.so.2.3.7` matches all three:
  `8c2066c36845cd4bda52041262d482b4c4e9aaf5d96e4108fbec31067bf5f7b5`.

Read-only shell resolution was verified with the standard package overlay:

```bash
direnv exec . bash -c '
  source /tmp/fr5-native-e7f22e7-aaZFgf/install/share/fairino_hardware_v3_9_7/local_setup.bash
  ros2 pkg prefix fairino_hardware_v3_9_7
'
```

It selects the temporary prefix; `ldd` also resolves its unchanged SDK there.
This is not proof of a running controller loading it. At an authorized launch,
verify the actual process library mapping/digest and exact qualified robot model.
Do not silently launch the different main-installed library. Rollback is ending
the exact trial process and not sourcing this overlay in the next shell; no
shared installation needs overwriting. `/tmp` is disposable; rebuild if absent.

## Physical next consumer and limits

No robot/recorder/ROS stack was started this turn. No current illumination or
controller state was observed. Historical scene revision 309 and cube pose are
navigation hints only: re-read SceneStateStore/Cell and actual qualification.
Never reset the assumed object position to A(0,0).

After software verification, use the existing approved bounded current-position
hold path, with current light/cell/scene/controller and sole-owner checks. Bring-up
itself can issue native current-position hold; it is not plan-only/zero effects.
The existing local diagnostic
`.agent-local/work/controller-clock-probe/readiness.py` submits no goal but configures
the selected clock policy and observes generation-zero readiness for ten seconds.
It tolerates initial unpublished evidence only, not expiry after first readiness.
Inspect before reuse. Do not confuse readiness with a successful learned Pick.

On qualification, progress to the actual admitted learned Pick and qualified
mechanical release/reset in the same lifecycle. Preserve original full chunk,
consumed indices, observed/commanded state, timestamps, outcome and diagnostic
artifacts for the recollection consumer. If qualified physical conditions fail,
stop affected effects; continue genuinely independent software work. Darkness
does not authorize bypass. Do not fabricate personal approval prompts beyond
existing system authority or silently waive existing gates.

The alternate opening-coordinate URDF remains unqualified and inactive. Original
model/calibration/data are unchanged. Two-stage gripper release timing is
intentional; preserve it. Mechanical completion is not semantic task success,
object placement truth or training authorization.

## Architecture and reuse decisions to retain

Installed LeRobot 0.6.1 already supplies policy loading, processors, Robot and
rollout/inference interfaces. FR5 uses native SmolVLA loading/predict_action_chunk.
The current FR5 executor and MoveIt transport own active goal/admission/collision
checks, not SmolVLA. Collision checks are sampled knots/interpolation, not a claim
of continuous collision proof or awareness of unregistered obstacles.

LeRobot reuse is not rejected: inspect its actual seam before replacing code.
Installed SyncInferenceEngine consumes single actions; `core.send_next_action`
returns the pre-Robot dictionary rather than `Robot.send_action`'s actual return;
teardown may interpolate a return to initial position. Full-chunk/consumed-index
evidence and sole mechanical lifecycle must survive any adapter. Do not rebuild
generic LeRobot infrastructure, nor migrate simply to hide native timing faults.

## Orca transition

- Run: `run_45e15721f588`.
- Previous coordinator: `term_29c8f930-59cf-428e-a606-9439c722cc98`.
- Review Task `task_e96b148c3312`, Dispatch `ctx_ef737274b6ae`, completion
  `msg_79755db31a07`: succeeded; exact external-created reviewer terminal closed
  through `.agent-local/bin/fr5-orca-ops` after identity/settlement checks.
- Review delivery `delivery_2a59b1be3e5a` acknowledged. Root status
  `msg_8983d2bb6334` records candidate/build/full-suite boundaries.
- Last liveness query: no active/ready structured Tasks; native lane owners were
  not observed by that helper. Do not infer their state or duplicate their work.

Read `orca orchestration check --run run_45e15721f588`, inspect relevant existing
lane state, and retain only independent work with a clear output/merge boundary.
Use Astra for lane ownership and Sol for suitable bounded work; user dislikes
Luna. No need to spawn a new worker for this already reviewed patch. Current
Router implementation route is bound/ready and Ponytail off by user preference;
do not install/change global routing just to continue. Use fresh routes only when
scope/authority actually changes.
