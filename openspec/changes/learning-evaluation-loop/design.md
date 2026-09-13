## Decision and scope

Use admitted selected FR5 data with the existing native SmolVLA trainer and saved processor/checkpoint contracts. First establish genuine optimization, reload and resource fit; then spend the smallest measured budget that can distinguish a learning hypothesis. Root coordinates admission and GPU ownership. Learning owns recipe selection, comparison protocols and interpretation, including physical trial comparisons; Curator owns selection/acquisition publication and Rollout owns physical trial execution and per-trial outcome evidence.

The start set is revisable. A model default, previous wiki or passing injected test does not establish suitability on FR5. Runtime measurements, exact launch arguments, checkpoint digests and attempts remain in native artifacts and Orca, not this design.

Batch one is a qualified starting point, not the permanent efficiency choice. After qualification, measure a larger batch against retained sample throughput, resource headroom and whole-run overhead; report exposure as well as update count and do not scale LR automatically. The smallest feasibility fork may warm-start a verified learned checkpoint through the existing native launch, explicitly resetting optimizer/scheduler/RNG/sample stream. Bind the immutable parent checkpoint and receipt, require the same dataset/partition/features/TRAIN normalization, and revalidate before publication and native consumption. This avoids unnecessary base-policy restarts without claiming exact continuation. Legacy same-run resume remains separate. Extending a completed schedule or changing batch during true continuation requires explicit scheduler-prefix meaning and a cumulative sample cursor verified through a second resume; native step-times-latest-batch is insufficient for mixed-batch history. Do not build that broader path merely to qualify batch memory/throughput.

## Evidence and transfer limits

Repeated checkpoint comparisons make sampled physical-unit action assessment a native evaluator responsibility. The existing evaluation CLI selects either its unchanged default flow-loss report or saved-policy/postprocessor action chunks. Before inference, select early/middle/late episode-local frames for every admitted held-out episode; retain raw observation identities, common noise, native solver settings, recorded target offsets/padding and separate arm/gripper errors. Sparse coverage is explicit and cannot masquerade as full-frame assessment. Reuse admission, dataset decoding, saved processors and native prediction; do not maintain a second agent-only evaluation policy. CPU tests and retained-output aggregation verify their own scope; actual native inference equivalence remains a separate completion condition.

- [SmolVLA authors](https://arxiv.org/html/2506.01844v1) study a pretrained vision-language backbone with an action expert; their [current LeRobot guide](https://huggingface.co/docs/lerobot/main/smolvla) recommends trying small batches and collecting task variations. This supports an inexpensive frozen-backbone starting point and testing variation in successful demonstrations. Neither source establishes FR5 memory fit, a minimum sufficient episode count or its best recipe.
- [Wen et al., Understanding Warmup-Stable-Decay](https://arxiv.org/html/2410.05192v1) examine horizon-dependent cosine training and alternatives for extending compute. Their theory assumes a particular loss landscape and their experiments concern language-model pretraining. This motivates an alternative schedule hypothesis and explicit continuation semantics; it does not show that WSD, constant LR or a particular warmup improves SmolVLA fine-tuning. No new scheduler is needed for the initial comparison.
- [Barannikov's original robotics experiments](https://huggingface.co/blog/m1b/validation-loss-robotics) compare validation loss, action MSE and simulated rollout success for Diffusion/PushT and ACT/Transfer Cube. Those proxies fail to consistently track success in those experiments. This supports retaining several qualified diagnostics and physical confirmation; it does not prove an absence of useful offline signal in FR5 or justify abandoning held-out data.
- Installed consumers decide what actually executes: `TrainPipelineConfig.validate`, SmolVLA's preset methods, `CosineDecayWithWarmupSchedulerConfig.build`, `set_requires_grad` in the VLM/expert and policy modules, and the official update/save loop. The scheduler scales warmup and decay to a shorter horizon; preset resolution can replace top-level optimizer settings. CPU configuration tests cover these consequential behaviors without model loading.
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/2.11/amp.html#gradient-scaling) distinguishes autocast from gradient scaling and describes compatibility limits for BF16-pretrained models under FP16. [Accelerate's native API](https://huggingface.co/docs/accelerate/package_reference/accelerator) makes mixed precision an explicit execution choice. The installed SmolVLA path has BF16 and FP32 trainable parameters; its AMP fallback selects FP16 scaling, whereas native mixed-dtype execution without autocast avoids that scaler. Actual first-update evidence, rather than the flag name, must establish the selected path's finite gradients and parameter changes.

## Minimal start set and competing hypotheses

| Choice | Initial decision and reason | Alternative and trigger |
| --- | --- | --- |
| Trainable components | Retain cached SmolVLA `freeze_vision_encoder`, `train_expert_only` and `train_state_proj`; train the expert and enabled projections/time modules. Verify names/counts and optimizer membership on the loaded model. | Visual adaptation, including LoRA, becomes a candidate after condition-specific perception errors persist with usable optimization and measured memory headroom. Freezing is neither a permanent rule nor evidence that visual features suffice. |
| Optimizer/LR | Native AdamW preset, peak LR `1e-4`, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `1e-10`, clipping `10`. Minimize simultaneous changes until actual update behavior is known. | Compare peak LR `5e-5` after persistent non-finite/skipped updates, unstable gradients, or sustained matched-development regression despite finite improving training updates, after excluding data/configuration and evaluation-scope differences. An isolated noisy minibatch or validation loss is insufficient. |
| Schedule | Native cosine preset, including its automatic warmup/decay scaling to the declared horizon. A manually “budget-matched cosine” would duplicate existing behavior. | If early updates destabilize before the shortened warmup ends, test longer **resolved** warmup while holding peak LR fixed. If repeated horizon extensions become the main cost, test native constant-with-warmup as a distinct continuation recipe. Do not change LR and warmup together in the first diagnosis or implement a WSD framework speculatively. |
| Batch/precision | One local process, batch 1, workers 0, native `use_amp=false` with Accelerator mixed precision `no`, no assumed accumulation. Retain native BF16/FP32 parameters and explicit operation casts; this is not all-FP32 training. Record actual dtypes, scaler and parameter changes. | Qualify finite updates, save and independent reload before increasing batch; choose an intermediate batch only when it answers a remaining resource question. Proper BF16 autocast or FP32 trainable weights with FP16 autocast are alternatives only if measured native-path stability or throughput warrants their added changes; an unsupported dtype flag or ineffective environment override is not a recipe. |
| Transforms | No image augmentation for the first fit and throughput measurement. Preserve the qualified camera/state/action contract. | Modest photometric augmentation is a competing generalization hypothesis if error follows lighting/view appearance in held-out evidence. Geometric changes require camera/action consistency; do not enable them merely because a native transform exists. This ordering is FR5 engineering judgment, not a claim that no augmentation is optimal. |
| Budget/checkpoints | One short engineering smoke, then choose a sustained horizon from measured update, evaluation and save/reload costs. Fix that horizon before launching. Preserve a midpoint and final checkpoint only when their evaluation fits the budget. | A cooled smoke may be continued as an explicitly different schedule history, not silently treated as the prefix of a fresh longer run. An epoch count alone neither demonstrates learning nor sets a useful minimum. |
| Offline assessment | Fixed admitted held-out episodes, train-only processor binding, seed, coverage, batch and precision. Use full per-episode and frame-weighted evidence for comparison; bounded first-task frames are reload evidence. | Repeat only an ambiguous checkpoint comparison with another fixed evaluation seed to estimate stochastic sensitivity. A loss difference within that sensitivity remains unresolved. Postprocessed physical-unit output errors can aid diagnosis but cannot establish task success. |

## Immediate experiment, cost and falsifiers

Use the qualified larger batch for the next continued-learning hypothesis while retaining the established freeze scope, no-scaler mixed-weight precision, transforms and TRAIN processors. Native preset precedence and horizon scaling remain verified configuration constraints. Historical batch-one throughput is a resource comparator, not a matched learning arm; report samples, optimizer updates and whole-chain cost separately. Do not require an intermediate batch or another from-base restart after a larger batch has demonstrated finite updates, save/reload and usable headroom.

The immediate learning experiment uses the already-qualified native warm-start path from an immutable learned checkpoint with an explicitly declared new schedule and reset optimizer/RNG/sample stream. This preserves learned weights and amortizes preparation over a meaningful adaptation budget; it does not isolate batch-size effects or claim exact continuation. Compare its implementation and run cost with genuine continuation before making new state-restoration plumbing a prerequisite for learning evidence. Genuine continuation remains a separate product question. Reuse native optimizer, scheduler and RNG restoration while explicitly retaining the consumed sample position across a batch change and a second resume. Preserve the already resolved scheduler prefix; extending the budget must not silently rebuild a different LR curve. A floor-preserving continuation is a bounded alternative once its state contract is executable; it separates inherited schedule behavior from a declared restart. Warm-start remains an available, honestly labelled reset path, not a substitute for this contract. Verify those semantics on CPU before consuming another GPU allocation. The prepared data loader must retain the committed epoch and offset, rather than its eager iterator or prefetched position. Recreating a partial-epoch iterator must not consume an extra policy RNG draw; an epoch-boundary restart must retain the ordinary iterator RNG consumption. Compare interrupted and uninterrupted native optimizer/scheduler/RNG traces through repeated resumes and an uneven last batch. Qualify the actual supported single-process, zero-worker, deterministic-transform path before extending those guarantees to other loader modes.

The competing hypothesis is that a cooled schedule limits useful adaptation despite affordable larger-batch throughput. Test a declared schedule intervention only when continued updates show insufficient progress after excluding no-update/precision, stochastic evaluation and data-contract defects. Do not scale LR with batch automatically. Stop an actual attempt on OOM, non-finite updates, sustained memory pressure, exhausted storage reserve, deadline or changed authority/resource conditions; do not repeat failed attempts automatically.

For the first sustained run, compute the affordable update count from measured setup, steady update, planned full held-out evaluation, checkpoint save/reload and storage costs, leaving measured host/device headroom. Compare checkpoint progress inside that fixed schedule before adding another recipe arm. Failure to improve offline diagnostics is evidence to investigate, not proof of physical failure; improvement is not proof of physical success. Keep optimizer instability, insufficient exposure, stochastic metric variation and perception/coverage limitations as competing explanations.

## Verification required from the first native smoke

1. Bind the exact source/cache revisions, admitted selected IDs, canonical train/held-out split, train-only normalization and native saved `train_config.json`/receipt. Confirm requested settings survived native resolution; feature profiles remain feature contracts.
2. Record actual loaded trainable/frozen names and counts by module, optimizer membership, parameter/gradient/autocast precision, initial and observed LR, scheduler state and scaler/skipped-update behavior. Establish a nonzero trainable-parameter change after a successful update; record what frozen components were checked. Flags alone do not prove this.
3. Separate startup/data preparation, steady loading/update, save and independent reload timing. Record device and allocator peaks with their measurement scope, host RAM/swap pressure and checkpoint bytes. The trainer's per-update peak reset and window-averaged logging are not a whole-run memory high-water mark.
4. Validate checkpoint/processor lineage and independently reload through the native offline consumer. Preserve actual per-episode sample coverage, finite stochastic loss and evaluation settings. The smoke establishes only the operations and samples actually observed.

If a measurement is unavailable, report it as unavailable and narrow the claim. Avoid running extra large scans merely to duplicate a verified immutable receipt. Original data remain read-only and no research helper becomes a second dataset, admission path or experiment registry.

The shared checkpoint validator compares both saved policy configuration and saved train configuration with the admitted feature contract. It recognizes only source-verified, inert native SmolVLA missing-image slots for the two-camera/one-blank profile; real image order, shapes, state/action features and blank count remain exact. CPU execution of the installed image-preparation consumer establishes slot equivalence; it does not prove model reload. Do not rewrite a genuine checkpoint or normalize away an unknown feature mismatch to make validation succeed.

## Successful-data utility and cross-owner contracts

The first acquisition hypothesis is that missing successful condition variation may be more valuable than additional repetitions in a well-covered condition. Its rival is that existing coverage is sufficient but optimization or sample exposure is inadequate. Join existing per-episode evaluation with Curator's immutable condition metadata by dataset/split/receipt and episode ID, preserving frame and equal-episode summaries. Correlated adjacent frames and one repeated session do not establish many independent generalization trials.

Ask Curator for a proposed comparison with matched cohort/count and a fixed held-out set; its selector must not silently change the evaluation suffix. If different train subsets change normalization, normalized flow losses cannot rank their data utility. Require outputs after each saved postprocessor on a shared physical scale or matched Rollout evidence, while keeping held-out statistics out of training. A variation-benefit hypothesis is weakened if its apparent gain vanishes after matching exposure/split/metric scale or repeats within evaluation variability. This is a proposed comparison contract, not permission to relabel, select new data or drive the robot.

## Historical quality comparison and solver handoff

The continuation protocol below records the earlier optimization hypothesis. It is not a standing next-run order after a retained warm-start has regressed. The bounded current comparison preparation is specified at the end of this design.

Prefer **continued optimization of the admitted learned checkpoint using the resource-qualified batch** when full-scope development loss continues improving and no optimization plateau has been established. Sparse physical-output regressions qualify the result but do not require every joint and episode to improve before further training. Compare the information gained from a larger fixed horizon with its measured marginal update cost and repeated setup/evaluation overhead before choosing an obligatory throughput pilot, residual intervention or selection change. Keep frozen components, native no-scaler precision, transformations and train-only normalization fixed. Bind inherited weights, optimizer/RNG state, sample position and schedule explicitly; distinguish any reset or schedule intervention from exact continuation. A fresh longer cosine recipe is an alternative requiring justification, not the default workaround for a completed horizon. Current author guidance supports adapting duration to observed performance, not importing another robot's step count, batch size or minimum episode requirement.

Fix each longer horizon before launch and save only midpoint and final checkpoints under the existing limit. Evaluate the final checkpoint on the same complete held-out scope and evaluation seed, then obtain its frozen observation outputs at both retained action-noise seeds. Reuse the reference checkpoint's completed outputs. Bound the whole training/evaluation comparison by measured setup/update/save/full-evaluation costs and reserve disk for both genuine checkpoints. A longer horizon tests affordable further adaptation, not a sufficient-learning threshold or minimum epoch rule. Execution requires an allocated GPU resource window and native authority revalidation.

The hypothesis is that further adaptation improves the qualified offline diagnostic across episodes. The initial loss screen uses at least ten percent lower frame-weighted loss and improvement in at least four of the six held-out episodes; these are engineering decision thresholds, not significance estimates or task-success criteria. Once sampled physical outputs are available, retain the componentwise comparison and concentrated regressions at both retained seeds. Report the near-constant joint separately, preserving its physical scale. Mixed outcomes remain a tradeoff rather than a favorable-seed selection; they do not automatically stop optimization or establish physical inferiority. A smaller gain at one longer horizon likewise does not prove convergence. Distinguish optimizer instability, stochastic sensitivity and condition-specific errors before collecting more repetitions or changing LR/visual adaptation. The saved midpoint remains available for a targeted check if the final result suggests degradation; do not automatically evaluate every checkpoint or repeat reference evaluations.

Success-condition evidence can contradict a coarse coverage explanation: different errors within the same yaw group require checking position/trajectory/optimization explanations before declaring yaw variation missing. One later-session demonstration cannot establish session generalization. Use immutable condition bindings and per-episode metrics; do not treat correlated frames as independent generalization trials.

For Rollout's numerical solver comparison, freeze observations from every held-out episode at fixed early/middle/late time fractions before solver outcomes. Preserve raw uint8 RGB, absolute seven-dimensional state, recorded task, source episode/frame/index, decoder output hashes and checkpoint/config/processor/split identity. These fractions provide temporal coverage, not semantic phase labels. Learning owns cohort and comparison interpretation; Rollout owns the existing native solver runner and per-trial measurements. Use paired common noise and report arm-radian and gripper-meter deviations separately. A faster candidate with no worse paired deviation than fixed5 is a conservative numerical-efficiency candidate; a speed/deviation tradeoff remains unresolved. Fixed10 is not action ground truth, and neither this screen nor the frozen observation cohort authorizes physical execution or establishes task usefulness.

## Decision after the matched checkpoint comparison

A complete matched-scope improvement can support affordable further adaptation of existing successful data; it does not permanently favor more updates over acquisition. If a later sustained warm-start materially regresses on the same full held-out loss cohort, retain the previous reference and reject automatic promotion on that diagnostic; preserve the child for separately qualified action-output and physical evidence. Do not repeat the same adaptation recipe merely to increase exposure. Preserve optimizer-reset and schedule differences as competing explanations rather than attributing the result to batch size or data scarcity alone. Prioritize the already-bounded new-data consumer and require a discriminating hypothesis before another same-data training allocation. Runtime comparisons and their exact costs remain in Orca and native evaluation artifacts.

The following was the **pre-batch4 planned** discriminating outcome, retained as historical context rather than a new execution request: a paired physical-output diagnostic before another training recipe or subset comparison. Reuse the frozen observations and obtain each checkpoint's fixed10 action chunks with the same explicit noise seeds through the existing native inference consumer and its saved postprocessor. Retain the actual finite action arrays with checkpoint, observation and seed identities; timing-repeat copies are not independent policy samples. Rollout owns that consumer and Learning owns the comparison, so neither a second inference implementation nor a new experiment registry is required.

Align recorded demonstration action chunks through the native dataset's configured action delta timestamps and padding mask. Bind them to the frozen observation hashes and original episode/frame indices, without training on them or changing normalization. Report per-joint MAE/RMSE over valid chunk positions, retaining per-observation and equal-episode summaries; keep six arm-radian axes and the gripper-meter axis separate. These are open-loop errors against one recorded continuation, not unique correct actions, safety tolerances or task-success measurements.

Near-constant TRAIN action dimensions make normalized and physical errors different diagnostics. The saved native postprocessor multiplies predicted normalized endpoints by the saved standard deviation; this does not provide a conversion from stochastic flow-matching velocity loss to action endpoint error. Inspect per-axis physical errors before blaming a near-constant joint for aggregate loss or altering its normalizer. A loss improvement with unchanged or worse sampled action errors falsifies the proposed proxy agreement and calls for investigation, not automatic deployment or threshold tuning.

For a later selection comparison, prefer Curator's immutable TRAIN-only XY contrast with matched yaw histogram, episode count, total frames and held-out IDs over the earlier yaw-confounded pair. This narrows one confound; it does not match visual content, temporal phases or sampled optimizer exposure. Preserve each arm's train-only processor and compare on the shared physical output scale. Request readiness is not training authority, and utility remains unmeasured until native candidate checkpoints and comparable outputs exist.

The paired native output diagnostic supports improvement on the shared physical scale while exposing per-joint episode regressions hidden by aggregate flow loss. Preserve that disagreement. A persistence reference that repeats the raw observed state can reveal remaining prediction error on short horizons, but cannot demonstrate task completion; report it explicitly as a diagnostic reference rather than a deployable controller. Separate first-action and full-chunk errors when interpreting a suspected temporal limitation.

When a per-joint episode regression conflicts with pooled improvement, one additional common action-noise seed on the same frozen observations and checkpoints can distinguish stable regression from sampling sensitivity. Keep the solver and processors fixed, retain both outcomes, and do not choose a favorable seed. The paired native evidence retains pooled physical-scale improvement while reversing some per-episode joint-regression directions. Treat those directions as sampling-sensitive, not stable condition rankings; do not launch another seed merely to resolve every uncertainty. Two seeds and a sparse reused validation cohort do not establish statistical confidence or physical readiness.

Keep recorded support separate from sampled prediction errors. Held-out targets beyond TRAIN extrema establish a scalar coverage gap which a subset of that same TRAIN pool cannot fill. Large sampled errors also occur within TRAIN extrema, so range coverage alone does not explain the remaining limitation. Curator should use existing successful command/condition records to propose matched acquisition variants; Learning should examine the optimization and chunk-output evidence before attributing those errors to missing data. No new phase labels, altered normalization or selection based on a favorable noise draw are warranted by this diagnostic.

Pooled loss and action improvement can coexist with an episode's per-joint regression at both retained seeds, preventing an unqualified dominance claim. A targeted saved-midpoint comparison within the same longer schedule exposes regressions already present at the midpoint that partially recover by the final checkpoint, alongside other errors that worsen later. Earlier stopping therefore does not uniformly resolve the tradeoff. Bounded localization in retained action arrays can distinguish observation-specific and forecast-offset errors without new model calls or semantic phase assumptions. Once that question is answered, prioritize a meaningful optimization horizon over further diagnostic detail while the full-scope curve still improves. Preserve all compared checkpoints; neither mixed componentwise outcomes nor the lowest aggregate loss selects a physically ready policy. Native per-run allowances remain mandatory at execution, but a coordinator-recorded step limit is not evidence of hardware capacity or sufficient learning; request an evidence-backed revision from its owner when the chosen budget needs one.


## Completed diagnostic design: loss regression versus saved action error

Before requesting another training run, use the retained reference checkpoint and its same-data warm-start child to test whether the observed full-development loss regression is accompanied by larger postprocessed action error. Reuse the reference's already retained fixed-solver, fixed-noise observation outputs; prepare only the missing child's native sampled-action input. This is a within-policy optimization comparison, not a solver experiment, batch-size causal claim, or evidence for targeted acquisition. The child already fails the retained full-loss promotion screen; this small comparison qualifies what regressed and does not authorize promotion based on a favorable sparse metric. When paired action MAE improves while loss worsens, retain both facts and both checkpoints: the loss-only claim of poorer action predictions is falsified, while tail errors and episode regressions still prevent a dominance claim. Close this diagnostic branch and hand the qualified candidate evidence to existing new-data/Rollout owners rather than automatically adding seeds or another training horizon.

Require identical original heldout identities, observation tensors, language, padding/target chunks, normalization contents, solver, explicit noise and precision before comparing per-axis errors. Preserve joint radians and gripper meters separately, per-episode regressions and both first-action/full-chunk scopes. Reject pairing if any binding differs; do not regenerate the frozen cohort to make it pass. Larger action error supports a broader open-loop regression; mixed or improved action error falsifies the claim that loss regression alone proves worse action predictions. Neither outcome establishes physical success, independent generalization, or why the recipe regressed. Existing successful data and acquisition remain separate hypotheses; do not infer data scarcity from this reset/schedule comparison.

Use existing offline evaluator, shared saved checkpoint/processor loader and deterministic view validator. Native flow-loss forward remains independent from action-chunk inference. The offline action path supplies explicit noise and recorded observations while online proposals consume current observations and retain their own state/effect lifecycle; shared computation must preserve those distinctions. No new registry, report store or execution framework is needed. A later shared action-call seam should accept the same already-loaded processors and policy without moving training loss or transport into it.

Record actual marginal admission/load/inference/report cost separately from the historical training/setup/evaluation investment and collection/reset/save/retry/human costs. Missing historical wall-time receipts remain unknown rather than inferred from step count. An ACT baseline remains optional contextual policy utility, with its ImageNet backbone initialization explicit; it is not a prerequisite or a causal VLM representation experiment. A future targeted-versus-balanced acquisition comparison must predeclare development/final conditions and control additional optimization when necessary; it is not automatically scheduled by this preparation.

## Next bounded comparison: same-parent lower-peak adaptation

After the matched loss/action diagnostic establishes disagreement, compare one lower-peak warm-start sibling with the retained same-parent, same-exposure reference child. This tests whether reducing the restart's LR excursion improves development behavior while retaining sampled-action gains. It does not isolate optimizer reset from inherited optimizer state, establish a local minimum, or estimate final physical utility. Decreasing training loss with increasing validation loss is compatible with poorer generalization and metric mismatch; a global empirical minimum need not optimize either validation behavior or task success.

Use the existing native warm-start wrapper and policy-level LR override. Keep parent weights, selected TRAIN/EVAL identities, raw view, normalization, batch, seed, trainable components, optimizer hyperparameters other than peak LR, precision, transforms and resolved schedule timing fixed. Keep the absolute decay floor fixed too; this changes the LR excursion, not every LR by a constant factor. Check the native parsed configuration and built LR trajectory before requesting allocation. Preserve the retained child as a historical single-seed comparator, with runtime/source or nondeterministic differences disclosed; do not rerun it automatically or treat a shorter independently cooled child as an intermediate checkpoint.

Save midpoint and final states within one predeclared horizon. Keep the trainer's evaluation timing identical to the historical comparator and evaluate saved candidates through the separate native evaluator after training, so additional validation calls cannot change training RNG. Full fixed-development loss, equal-episode summaries, tails and common-noise physical-unit action errors are separate diagnostics. A midpoint provides within-run direction; it is not a matched midpoint comparison if the historical arm did not retain one. Repeated use makes these episodes development validation, not an untouched final test.

Retain reference/best candidate weights separately from the last native training state. Warm-start explicitly resets optimizer, scheduler, RNG, sample stream and step. Legacy same-run resume remains available but does not establish uninterrupted sample/RNG equivalence; an interrupted comparison must be labelled and requalified, not silently pooled. Genuine new-output continuation was unnecessary for this selected warm-start comparison. The extension below addresses the subsequent longer-curve outcome; extending a fully cooled schedule still answers a different question from maintaining a useful LR. Do not rewrite a completed run's horizon or default to an arbitrary very long run.

## Native continuation and the subsequent longer curve

The initial workers0 qualification below is extended to the deterministic rhythm40 recipe: workers4, prefetch4, persistent workers, spawn. The prepared native loaders receive independent worker-seeding generators for recreated TRAIN workers and previously started EVAL workers. Unlike the workers0 iterator, persistent workers do not ordinarily consume a new base-seed draw at each epoch boundary. Their recreation must therefore avoid an extra policy CPU RNG draw both mid-epoch and at a boundary. A first-ever EVAL iterator still uses its ordinary native draw. The native sampler's seed/epoch order and committed cursor, not the prefetched position, determine consumed samples.

The first actual rhythm40 attempt reached step15000 but exhausted host headroom when its EVAL pool joined the persistent TRAIN pool before checkpoint serialization. The supervisor stopped it for sustained memory pressure; only the new config file was written, so there is no recoverable 15k state or paired result from that attempt. Original12k and approval hashes remained unchanged. The continuation consumer now releases already-started EVAL workers when each evaluation finishes, using the existing native worker-options factory (TRAIN is constructed first, EVAL second). Their isolated generator already prevents additional policy RNG draws. TRAIN remains persistent; the first-ever EVAL lifetime is unchanged. The saved parent recipe is not rewritten; this logged EVAL process-lifetime adaptation changes no examples, batch, forward computation, LR, optimizer or policy RNG. CPU fixtures assert only the four TRAIN workers remain at native save, alongside exact uninterrupted/resumed traces. Actual retry must retain the failed output and start from the last complete12k state in a distinct output; unsaved3k updates are lost compute, not extra committed sample exposure.

The workers4 v2 sidecar adds only `persistent_eval_started`; the immutable initialization binding carries its inherited value. Validation combines that value with evaluation events between the inherited and saved steps, because native evaluation precedes checkpoint saving. This retains the distinction when later evaluation cadence changes. Workers0 v1 state remains unchanged. No worker RNG or queue is claimed restored: the supported data/transform path is deterministic, while policy forward RNG, optimizer moments, schedule and processors remain native. CPU fixtures include clean uninterrupted reference traces, successive resumed TRAIN/EVAL loaders, uneven epochs, first-ever EVAL and an intentionally wrong EVAL-history negative control. Matching exact CPU fixture values establishes this scoped mechanism; it is not a new generic hardware reproducibility gate. Root owns actual-policy loading, resource allocation, training and independent reload.

The supported extension uses the existing wrapper, admission, launch receipt and checkpoint validator. Native `resume=true` with the parent's `train_config.json` restores optimizer/scheduler/RNG and policy/processors. Scoped process-local hooks preserve the original scheduler construction horizon, supply the committed sampler position, correct recreated partial-epoch iterator RNG consumption, and append committed state after native checkpoint saving. The official loop still owns every forward/backward, optimizer step, evaluation and save. No vendor edit or second trainer is required. Merely supplying a larger `steps` value to existing native resume would rescale the earlier cosine schedule; native absolute-step/latest-batch cursor reconstruction also cannot represent mixed-batch history through another resume.

New receipt initialization mode `continuation` binds the immutable parent checkpoint tree and receipt, has `reset: []`, and carries the parent's committed step/cursor plus schedule descriptor. Only new output, absolute ending step, batch and evaluation/save cadence can change alongside the explicit schedule-tail choice. Existing full recipe, dataset, partition, feature, normalization and authority checks remain authoritative. `preserve` keeps the original native horizon and inherited future. `hold` uses the original LR function through the parent step and clamps its future argument there, so the next optimizer update uses the actual saved parent LR. Scheduler step and AdamW moments continue. An inherited hold cannot be undone on another resume. Saved config alone is insufficient to interpret this extension: the validated receipt and required `fr5_continuation_state.json` travel with the ordinary native checkpoint.

The committed cursor counts actual anchor samples, including uneven final batches, after successful native optimizer updates. The adapter restores the prepared loader's epoch/offset and excludes eager iterator position. Python and NumPy Gaussian caches supplement native serialization; legacy omitted caches cannot be recovered, so legacy entry equivalence begins from the actually serialized native state. Reject legacy resumed parents without cumulative cursor evidence. Qualify one process, workers0, deterministic map-style data, no AMP/compilation/weighted sampling/drop-tail first. Native evaluation consumes RNG; matching evaluation cadence is part of an uninterrupted/repeated-resume comparison. CPU-small actual-native tests exercise two-parameter AdamW, native sampler/Accelerator, checkpoint serialization, all three RNG streams, changed batch size, uneven batches, epoch boundaries and repeated resumes. Real policy loading and GPU reproducibility remain root's qualification, not a claim from fixtures.

Explicit same-output recovery reuses the receipt already validated by the resume consumer, including pending manifests after interrupted publication. Reopening only a finalized receipt would contradict that admission and fail before the native loop. Public CPU-small coverage now interrupts native checkpoint saving twice, resumes through the public consumer, and validates the resulting native configurations that point to their own earlier checkpoints; a foreign-output source is rejected. This preserves the separate automatic delegated-request rule that refuses pending publication.

Native processor reload reconstructs statistics through [`from_tensor_to_numpy`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/converters.py), which calls `.item()` for a one-element tensor. A subsequent [`NormalizerProcessorStep.to()`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/processor/normalize_processor.py) materializes those counts as scalars when saving again. The shared normalization validator therefore compares scalar counts against admitted singleton counts by exact value; every operational statistic retains exact shape/value comparison. This recognizes the native representation without changing processors, checkpoint bytes, model behavior or admission authority. CPU-native coverage performs two load/move/save cycles and preserves normalized and unnormalized outputs while rejecting changed counts and operational tensor shapes.

Primary evidence checked for this design: [LeRobot v0.6.1 scheduler source](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/optim/schedulers.py) explains the native horizon rescaling; [PyTorch DataLoader documentation](https://docs.pytorch.org/docs/2.11/data.html) documents iterator base-seed RNG consumption; [Accelerate loader API](https://huggingface.co/docs/accelerate/main/en/package_reference/torch_wrappers) supplies the prepared loader epoch seam. [Wen et al.](https://arxiv.org/html/2410.05192v1) motivate avoiding horizon-dependent repeated decay, but neither that language-pretraining study nor SmolVLA's guide establishes an optimal FR5 duration or constant LR. The chosen hold is a testable local engineering hypothesis, not a claimed WSD reproduction.

The completed lower-peak comparison retains conflicting metric directions. Continue from its immutable midpoint with a declared constant tail as the next learning hypothesis: useful optimization at the midpoint LR over several more dataset passes may reveal a longer trend than repeatedly cooling short runs. Preserve the original decayed endpoint as a sibling comparator at matched exposure; do not splice it into the constant-tail branch or call the midpoint a winner. Full DEV loss above the reference and improved sampled arm errors can coexist with worse gripper tails. These do not establish a local minimum, physical utility, OOD behavior, or a solver ranking across learned checkpoints.

The concrete handoff uses batch4 and the lowpeak run's 4k checkpoint at LR2.625e-5, continuing to absolute12k and then24k with a persistent hold boundary4k, snapshots every4k and native small evaluation every8k. The first block supplies two new curve points before review; the second adds three more. One 4k interval is 16000 committed anchors, about one pass over TRAIN15492. At24k the lowpeak stream has96000 anchors (6.20 passes); parent10k contributes another10000 ancestral weight-training anchors but its optimizer/RNG history was explicitly reset by the earlier warm-start. Do not label that older ancestry as one continuous optimizer run.

Budget from retained native execution: about0.21s/update and the successful lowpeak chain's3053s for8k updates plus two complete evaluation pairs imply roughly50–55min for the first block and75–80min for the second, including repeated admission/load overhead. Reserve65/100min respectively, five new full checkpoints at about1.23GiB each, scratch space and the existing8GiB free-disk reserve. Root rechecks live memory/GPU availability and retains the shared heavy lock and supervisor pressure/nonfinite/deadline behavior. These estimates are not new measurements. No parallel heavy jobs or new authority are issued by preparation.

Evaluate each saved point independently using unchanged full DEV4644 anchors, batch1/workers0/seed1000 flow loss and the frozen18-observation fixed10/seed0 action cohort, preserving observation/noise/targets/masks and physical-unit per-joint/episode/tail summaries. Run external evaluation after each block so it cannot perturb the training RNG. The native32-sample check remains a separate diagnostic. Keep solver comparison checkpoint-local; use fixed10 for curve comparability even if a faster solver wins a numerical cost tradeoff elsewhere.

Review the12k point for interpretation and runtime correctness, without stopping on a single DEV fluctuation or treating8k as sufficient. Unless state/resource failure or severe persistent deterioration invalidates the experiment, complete the24k curve. After at least the16/20/24k points, extend one further8k block with the same hold only if a repeatable DEV or physical-error trend is still improving without a persistent opposing tail deterioration and the marginal cost remains acceptable. If all late points plateau/worsen across both metric families, preserve candidates and investigate scaling/coverage/objectives rather than prescribing300k. Thresholds and paired-seed checks for ambiguous trends are specified in the local execution handoff; they are operational decision rules, not physical acceptance thresholds.

## Executed rhythm40 continuation, 2026-09-12

The later explicit handoff allocated actual execution to the retained Learning owner. This is the clean rhythm40 stream, separate from the historical raw26 hypotheses above. Owner source `fefdc323bddbca0a635c8cb7341ae66f93707288` consumed immutable `smolvla-pickplace-rhythm40-clean-12000-r1/checkpoints/012000/pretrained_model` through the existing production wrapper into distinct `smolvla-pickplace-rhythm40-continue-18000-r2`. The failed r1 output remains intact. The successful attempt exited0 after1310.215s (21:25:26–21:47 KST); native updates began21:28:41 and training ended21:47:04. No scratch/warm-start, source data, approval, augmentation, split or original checkpoint change was made. The supervisor's complete parent/approval hash inventory matched after exit.

The exact executable command, from the owner worktree, was:

```sh
direnv exec . bash scripts/train_policy.sh --continue-from /home/codelab/Desktop/Project/fr5_ws/outputs/local-learning/smolvla-pickplace-rhythm40-clean-12000-r1/checkpoints/012000/pretrained_model --output /home/codelab/Desktop/Project/fr5_ws/outputs/local-learning/smolvla-pickplace-rhythm40-continue-18000-r2 --approved-inventory /home/codelab/Desktop/Project/fr5_ws/outputs/training-admission/fr5-pickplace-rhythm-ab-curated40-20260912/approval/training_approved.json --steps 18000 --batch-size 4 --eval-steps 3000 --save-freq 3000 --continuation-schedule preserve
```

Complete15k and18k snapshots each occupy1,319,414,244/1,319,414,243bytes. All155 active AdamW step scalars equal15000/18000. Scheduler last_epoch matches each cut, original native horizon12000 remains, and LR stays2.5e-6. Committed samples advance47998→59998→71997, or2.1525→2.6906→3.2287 nominal passes over22299 TRAIN frames. The new interval therefore adds23999 committed anchors; the earlier unsaved attempt is not additional model exposure. The18k cursor is epoch3/offset5100, preserving the uneven final batch. Saved normalization means/std/min/max and count values/dtypes are identical; four preprocessor counts undergo the already qualified native singleton-to-scalar serialization, so processor byte identity is not claimed. Both processor JSON files and postprocessor tensors are byte-identical.

Native32-sample eval loss is0.6508 at15k and0.4038 at18k. Its selected observations and stochastic draws differ from the frozen paired cohort; it cannot alone establish the requested learning trend or be compared directly with the earlier sparse paired loss. Independent15k/18k reload and paired evaluation use the retained evaluator, fixed10, all heldout8 episodes, q=.1/.5/.9 and seeds0/1/2, with saved processors and TRAIN-only normalization. Further learning and offline metrics remain advisory, never extra rollout admission gates. Learning stops this allocation at18k and interprets improvement, plateau or regression under the inherited LR floor; neither18k nor a flat curve proves a global minimum.

Local execution evidence is in the owner `.agent-local/work/learning-evaluation-loop/rhythm40-live-r2/`, with resource samples in `rhythm40-live-resources-r2.jsonl` and a header-only saved-state audit in `saved-state-evidence.json`. Across262 five-second samples, minimum host availability was2.156GiB and minimum disk availability2.292GiB; memory full-PSI peaked11.04 but did not persist to the30-second stop condition. These are sampled bounds, not instantaneous peaks. Root separately freed unrelated storage during this allocation; this owner deleted no checkpoint/data/cache. The EVAL-lifetime correction has fresh18 continuation tests passing and24 combined continuation/checkpoint tests passing; no full regression was repeated. GPU allocation returns to root after the bounded paired comparison, coordinated through existing Run `run_45e15721f588`.

The paired reloads subsequently completed, exit0 each, in175.966/173.258s including137–147s of native admission. The retained resource supervisor acquired the shared heavy lock and ran one model at a time. One earlier evaluator attempt stopped on an occupied lock before loading any model; its manifest is retained separately. Successful raw reports, source/input manifest, resource traces and `analysis.json` are under owner `.agent-local/work/learning-evaluation-loop/rhythm40-live-comparison-r2/`. Reproduce the new evaluations with `direnv exec . python3 -B .agent-local/work/learning-evaluation-loop/rhythm40-live-evaluate.py --supervise` in a distinct evidence output (the helper deliberately refuses overwrite), then run `rhythm40-live-analyze.py`. The helper reuses main's existing `learning-rhythm40-20260912/compare.py` supervisor and public evaluator read-only. Initial evaluator source cutoff was main79ad8b31; root later integrated the exact memory fix as5b9e438 and pushed7f6e0cc, with another18 focused tests passing. Imported evaluator/native source hashes and library versions match the original comparison; only the checkpoint validator differs. The complete supervisor input/source hash inventory is unchanged after both reloads. Learning released its GPU allocation and reported `msg_b07c742f499f`; subsequent GPU use belongs to root.

The dataset remains `fr5_pickplace_rhythm_ab_20260907_r01`, TRAIN0–31/EVAL32–39, digest `ff78d9cf00eef980905d83d670889f52b254a6933606ae0ef560ebc71381f8e6`, split `03d55846a3c85bc5d4907b2d00853df78af35b6b84be99924741b0382c62f2bd`, approved inventory `052280e7c2bd73e64ecf6b45db2024334b8d60d800191f791e2501ae24a9285f`. Saved15k/18k model SHA256 values are `95c0016203b2e6c10b65fdaf5c143fe114865a48acb9c173690cffa9727ab070` and `196173564dcfe6bb5a9b6c01779df840ef4b48c2b0a46830c6faaa117a37c0f6`. LeRobot0.6.1, Torch2.11.0, Transformers5.5.4, Datasets4.8.5 and Safetensors0.8.0 are unchanged. All588 pairing assertions passed across9k/12k/15k/18k: dataset/split/authority, saved normalization lineage, exact observations/images/states/targets/masks, fixed10 and noise seeds, and explicit native flow draws. Original9k/12k metrics recompute identically. The earlier3k/6k rows below reuse their completed same-cohort summaries; no original checkpoint was inferred again.

| Absolute updates | Paired flow loss | Arm component RMSE (degrees) | J6 RMSE (degrees) | Gripper RMSE (mm) | J6 internal velocity RMS (rad/s) | Projection-rejected rows / chunks |
| --- | --- | --- | --- | --- | --- | --- |
| 3000 | 0.270207 | 8.799 | 20.482 | 2.8355 | 0.759152 | 89 / 19 |
| 6000 | 0.270152 | 9.042 | 21.606 | 1.1094 | 0.636150 | 69 / 16 |
| 9000 | 0.232749 | 7.977 | 19.062 | 0.6953 | 0.557929 | 93 / 19 |
| 12000 | 0.226981 | 7.662 | 18.321 | 0.5401 | 0.556999 | 20 / 10 |
| 15000 | 0.236296 | 7.828 | 18.749 | 0.5218 | 0.545784 | 37 / 13 |
| 18000 | 0.235777 | 7.719 | 18.462 | 0.5780 | 0.546838 | 36 / 13 |

Each checkpoint has24 unique observations from8 heldout episodes, with3 paired seeds producing72 chunks/3600 valid action rows:3600 errors per axis (21600 arm scalar errors),72 first jumps per axis,3528 internal first differences and3456 second differences per axis. The72 native flow draws contain25200 valid timestep-axis residuals. Instruction/time/seed groups have36/24/24 chunks. These repeated observations and overlapping rows are development evidence, not independent trials or OOD testing. Projection uses the unchanged existing processor and limits; its numerical rejects and raw30Hz derivatives do not establish execution feasibility or success, and serialized retiming was not rerun.

Relative to12k,18k paired loss is3.88% higher, J6 RMSE0.77% higher and gripper RMSE7.02% higher, while J6 internal velocity RMS is1.82% lower. J6 first-jump RMS is17.316→17.733→17.583degrees at12/15/18k; corresponding maxima46.387→47.110→46.062degrees. J6 second-difference RMS is28.318→27.690→27.743rad/s². Thus the roughness improvement is real in this cohort but small, and the high absolute residual persists. Both instructions and all3 seeds have worse J6 error and paired flow loss at both new cuts than12k. Seed-pooled J6 improves on5/24 and9/24 observations at15/18k versus12k;18k improves on18/24 relative to15k but does not recover the12k aggregate. J6 roughness improves on23/24 observations at18k versus12k.

Gripper outcomes are direction-dependent: blue→red RMSE0.5657→0.5717→0.6876mm, red→blue0.5133→0.4665→0.4422mm. At18k,16/24 seed-pooled observations improve versus12k despite worse pooled RMSE, because the worse late errors carry greater squared-error weight. At q=.1/.5/.9,18k J6 RMSE is3.547/25.574/18.865degrees and gripper RMSE0.241/0.349/0.907mm;12k was3.603/25.570/18.442degrees and0.267/0.446/0.778mm. All13 newly rejected chunks occur at q=.9 (seed0 five, seed1 zero, seed2 eight); no favorable seed was selected. Native32 eval's0.6508→0.4038 decline therefore does not imply improvement on the different frozen paired cohort.

The predeclared evidence for consistent benefit from more optimization is not met for either J6 or gripper. This supports a near-flat/slightly regressed late development curve under this cooled recipe, with a modest opposing roughness gain; it does not prove overfitting, a local/global minimum, a model-capacity limit, or a specifically J6-only defect. Retain12k as the reference candidate for root's separately authorized execution comparison;15k's best aggregate gripper error and18k's modest roughness change do not justify automatic promotion.

The most diagnostic next single training comparison is a declared schedule-tail fork from the latest saved non-floor point009000, holding its actual saved LR9.456213946819498e-6 through absolute18000 with the existing `--continuation-schedule hold`, same batch/data/augmentation/processors and3000-step eval/save cadence. Its9000-step prefix and native optimizer/RNG history stay intact; only the future LR schedule differs from the retained cooled branch at matched12/15/18k exposure. This is not a claim of continuing the18k state at a retroactively raised LR. The9k scheduler/RNG/optimizer files exist and saved last_epoch/LR were inspected; this new fork has not been admitted or launched in this allocation. Compare the same cohort against the retained cooled branch, with per-direction/seed/observation consistency and tails. If the held-LR branch improves those errors consistently, early cooling gains support; if it cannot, repeating more floor-LR updates is a weaker next use of compute than a matched TRAIN-versus-heldout residual/coverage diagnosis. Neither outcome establishes global optimality. At measured~0.175s/update,9000 updates plus startup,3 eval/save events and3 separate paired reloads suggests about40min total; reserve45min and three complete snapshots (~3.69GiB) plus existing1GiB reserve and small evidence/cache writes. Root owns the next resource decision; no automatic extension or new rollout gate is created.

### User-authorized root execution of the LR-tail comparison (2026-09-12)

After the condition-support diagnosis and old-data reuse review, the user explicitly
requested root to execute the proposed overnight comparison directly. Missing
condition support does not rule out an optimization benefit. The single admitted
experiment branches from the original009000 full state and holds its saved
LR9.456213946819498e-6 to18000 via the existing native continuation path; it does
not restart from18k or reset the optimizer. Batch4, frozen VLM/action-expert
training selection, disabled transforms, original TRAIN/EVAL partitions and
saved TRAIN normalization stay unchanged. Compare new12/15/18k against retained
cooled references on the same observation/noise cohort, preserving per-direction,
per-observation and gripper/J6 tradeoffs. No automatic promotion or physical gate
is introduced. Root owns the training/evaluation resource allocation; no duplicate
Learning worker run is requested. Reserve three complete checkpoints plus the
existing1GiB disk reserve, supervise existing memory/PSI/nonfinite/OOM conditions,
bound training to45minutes, and use the retained15-minute-per-checkpoint evaluation
supervisor. The roughly45-minute whole comparison is an estimate, not a measured
completion time. Partial failures are retained, never automatically retried.
Original data/parents stay immutable. Physical work remains prohibited until
the user's **복귀**. Local root supervisor and receipts:
`.agent-local/work/lerobot-fr5/hold9k-20260912.py` and `hold9k-20260912/`.

During this run the user reaffirmed rollout→diagnosis→targeted recollection as
the next priority and made this the last current learning comparison. Finish
this fixed comparison and preserve its evidence; do not open another training
fork or metric search. If no clear improvement appears, leave the coordinator
blocked for physical return rather than filling the wait with more learning.
This is a resource/priority decision, not proof that learning cannot improve,
and offline metric improvement is not a new precondition for the existing
bounded physical rollout path.

### Completed LR-tail evidence and stopping decision (2026-09-13)

Root completed this exact branch from source8f23ba2 in1935.471s, exit0, with no
resource/numerical stop and unchanged parent/inventory hashes. All three native
snapshots retained LR9.456213946819498e-6 and155 AdamW step scalars at their exact
12/15/18k counters. Committed sample counts are47998/59998/71997, corresponding
to2.152/2.691/3.229 passes over22299 TRAIN anchors; counts reflect short epoch-tail
batches, not step×batch estimates. Saved processors retain the original values;
only singleton normalization-count serialization becomes scalar. Frozen-VLM
selection, batch4 and disabled transforms remain unchanged. This is continuation
from9k, not a new random initialization or an18k LR reset.

Independent native reloads exited0 in185.012/180.013/180.016s. The existing public
evaluator and retained local comparison callback produced72 chunks and72 paired
native flow draws per checkpoint on the unchanged24 development observations.
All1029 comparison checks passed across four retained and three new reports:
data/split/approval, exact images/state/targets/masks, solver/noise and flow draws.
Existing baseline metrics recompute identically. Native library source hashes and
versions match; the only recorded project source differences are the previously
reviewed checkpoint validator and TRAIN diagnostic helpers/report labels, which
do not change this heldout numerical path. Evaluation input/source hashes remain
unchanged after all reloads. No full discovery or new product-code change was
needed for this experiment.

| Held9k branch checkpoint | Paired flow mean | J6 RMSE (degrees) | Gripper RMSE (mm) | J6 internal velocity RMS (rad/s) |
| --- | --- | --- | --- | --- |
| 12000 | 0.226781 | 17.932 | 0.5071 | 0.538284 |
| 15000 | 0.230258 | 18.142 | 0.4748 | 0.551020 |
| 18000 | 0.225343 | 17.358 | 0.6322 | 0.528041 |

Compared with the original12k reference, held18k reduces J6 RMSE5.25% and flow
mean0.72%, but increases gripper RMSE17.05%. J6 improves in both instructions and
all three inference seeds; flow does not, with red→blue and two seeds worse.
Gripper p95 absolute error improves0.743→0.666mm while its maximum rises
8.360→8.944mm, so a better typical tail cutoff does not erase rare large residuals.
Held12k has small pooled improvements in both J6 and gripper; held15k has the best
pooled gripper metric among these cuts. Neither is uniformly superior on every
instruction/seed/metric. At matched18k exposure, held versus cooled J6 improves
on19/24 seed-pooled observations, gripper on11/24. These correlated observations
and three inference seeds are not independent training replicates or statistical
significance evidence. Internal trainer eval_loss0.6063/0.6042/0.3817 uses a
different bounded sampling/RNG path and must not be substituted for this paired
comparison or physical task success.

**Decision:** this finds a modest optimization effect, not a decisive learning
solution or proof that learning is impossible. Preserve all candidates and the
original reference; no automatic deployment/promotion or new rollout quality
gate follows. Per the user's latest priority, stop additional training/metric
search, retain the physical pause until **복귀**, and resume at the existing
bounded rollout→diagnosis→targeted-recollection path. The missing-condition
hypothesis remains relevant; this experiment did not change data coverage.

Five-second training samples had minimum host availability2.545GiB, disk3.780GiB
and maximum full-memory PSI avg10 of4.19; no supervisor stop was triggered. All
training/evaluation children exited, GPU compute was empty and the shared heavy
lock was released. The separate user-owned Portfolio visual task is not paused
by this Learning closure.

Evidence remains under `.agent-local/work/lerobot-fr5/hold9k-20260912/`:
`start.json`, `result.json`, `training.log`, resource traces,
`saved-state-12000.json`, `saved-state-15000-18000.json`, and
`comparison/{plan,012000,015000,018000,immutability}.json`.
`analysis.json` SHA256 is `af33e77b98bb0fa0562b067740c9b196e9ef735decd61418fbe260067c0de1eb`;
it includes all report identities and source deltas. New training receipt digest
is `a23934b3eb46dc587193df46178c20d03026628ce1b1905c0ef9b5a59168d85e`.
Saved model SHA256 values at12/15/18k are respectively
`a2969640e3573aa017121ddefe82a7bca5811fd05f5c2ea070ebcf0dc670006a`,
`18237ea818e0320f1b4f8f851cc503fd4170c1cb906fb846ed16275a362de65d`,
`b1046cb4b7ec69e78f372d57992bdf6da740cf9185bcdd67bc7294a5abc23651`.
Root-local `check_state.py` and `analyze.py` reuse existing native evidence and
metric functions; they are not product CLI additions. All model/data/runtime
artifacts remain ignored and originals are preserved.

## Subsequent 12k TRAIN-versus-heldout diagnosis

User/root message `msg_d319051f4bf3` explicitly allocated this independent offline diagnosis while physical activity was stopped. No further training or robot/ROS/camera operation was performed. The unchanged12k checkpoint above was loaded once with its saved processors and TRAIN-only normalization. Before inference, TRAIN episodes2,3,10,11,18,19,26,27 were fixed as the middle pair in each consecutive eight-episode TRAIN block: four episodes per instruction spread across the32-episode partition. Each contributes q=.1/.5/.9, giving24 TRAIN observations. The original24 heldout observations from32–39 and all3 noise seeds remain unchanged. Time fractions are not annotated semantic phases. Observations differ between partitions; the native flow draw rule is identical, but `100000*seed+dataset_index` produces different draws on their different dataset indices.

Owner source591daf3 adds only two existing-evaluator diagnostic helpers and truthful TRAIN report scope. `training_diagnostic_admission` derives a unique sorted TRAIN subset from a previously admitted heldout checkpoint without modifying its split or the public heldout admission contract. `native_flow_residual_per_axis` observes one actual native model forward, excludes padded timesteps and non-FR5 dimensions, and requires the mean of seven axis losses to agree with native `reduction="none"`. Because native SmolVLA calls `.forward` directly, a scoped method observer is used instead of a Module hook; the original method is restored even on failure. No trainer, alternate policy loader or new normalization authority is introduced. Focused `direnv exec . python3 -m unittest tests.test_offline_evaluation` passed24 tests in13.170s, including six new tests for disjoint TRAIN scope, time/dimension masking, changed denominator rejection, observer restoration, nonfinite residuals and empty targets. Existing heldout admission and outputs retain their default behavior.

Only missing evidence was recomputed:72 TRAIN action chunks and144 per-axis flow draws (72 in each partition). All72 heldout action chunks were reused byte-for-value from the original report. Their scalar flow losses, input tensors, states, targets/masks, native noise identities and times were checked against the earlier report; scalar losses match within1e-7. Every new axis decomposition agrees with its own native scalar loss. Per partition there are8 episodes,24 distinct observations,72 chunks/draws,3600 valid target timesteps per axis and25200 flow residual entries; each instruction/time-fraction/seed group has36/24/24 chunks. All heldout physical metrics recompute identically. TRAIN covers8/32 episodes, and neither partition is exhaustively evaluated over time.

| Metric | TRAIN diagnostic subset | Heldout diagnostic cohort |
| --- | --- | --- |
| Native flow loss, sample mean | 0.058278 | 0.226981 |
| J1 flow MSE | 0.039188 | 0.075535 |
| J2 flow MSE | 0.049130 | 0.062029 |
| J3 flow MSE | 0.058134 | 0.069871 |
| J4 flow MSE | 0.065931 | 0.093847 |
| J5 flow MSE | 0.120621 | 0.260582 |
| J6 flow MSE | 0.043247 | 0.996604 |
| Gripper flow MSE | 0.031698 | 0.030398 |
| J6 sampled physical RMSE | 2.661degrees | 18.321degrees |
| Gripper sampled physical RMSE | 0.3503mm | 0.5401mm |
| Constant observed-J6 baseline RMSE | 4.899degrees | 4.680degrees |

J6 accounts for62.72% of heldout flow loss and80.73% of the TRAIN/heldout flow gap. Flow MSE is in the normalized native objective, not degrees or millimetres; the gripper's slightly lower heldout flow MSE does not contradict worse sampled physical error. The constant-observed-state calculation uses existing observations and recorded future targets only; it is a numerical no-change reference, never a proposed command or physical hold. It indicates sampled J6 predictions improve on that reference on TRAIN but add substantial deviation on heldout.

The J6 gap occurs in both instructions (blue→red2.613 versus17.745degrees; red→blue2.708 versus18.879degrees) and every seed (TRAIN2.333/3.282/2.242 versus heldout19.195/16.700/18.964degrees). At q=.1/.5/.9, TRAIN J6 RMSE is2.668/3.113/2.106degrees, heldout3.603/25.570/18.442degrees. Corresponding gripper errors are0.278/0.236/0.485mm versus0.267/0.446/0.778mm. Therefore the aggregate difference is concentrated in later time fractions, not uniformly present at the first sampled observation.

A posthoc descriptive breakdown retains every heldout episode:32/33/34 J6 RMSE is2.878/3.436/3.503degrees, while35/36/37/38/39 gives25.163/28.458/19.126/24.398/15.771degrees. Full-TRAIN saved state statistics bound J6 at[-57.075,50.608]degrees. Three heldout observations fall below that range and have24.841degrees pooled error, but the other21 observations still have17.188degrees error. Inclusion in one scalar range does not establish coverage of joint-state/image/task combinations; this result rejects only the explanation that all large errors are simple one-axis range extrapolation. Episode-index and range groupings are exploratory descriptions, not newly selected evaluation cohorts.

The selected TRAIN observed-J6 range/mean is[-49.221,40.144]/-1.571degrees; heldout is[-67.461,38.068]/-15.873degrees. Recorded future-target means are-2.065 versus-16.125degrees. Sampled prediction signed J6 bias is-0.209degrees on TRAIN versus+10.863degrees on heldout. These conditional distribution differences are preserved in `observed-target-distributions.json`, including both instructions; they are not a proof of a coordinate error. J5 illustrates why normalized flow and physical errors must remain separate: its substantial flow contribution accompanies physical RMSE only0.000024/0.000039degrees on TRAIN/heldout.

This diagnosis weakens the blanket hypothesis that SmolVLA cannot fit J6 more accurately: the same checkpoint fits the selected TRAIN observations much better. It prioritizes conditional coverage/generalization and observation/trajectory consistency over simply adding more updates, but does not identify a cause, establish overfitting, prove a capacity limit or validate a real rollout. In particular, the two populations differ in scenes and trajectories, the TRAIN subset is small, native flow draws differ, and only one training seed/checkpoint is diagnosed. Three action-noise seeds are not independent learned models. The next least-cost investigation should inspect existing immutable source records around episode34→35 and match the problematic mid/late observations to TRAIN state/image/task neighborhoods, checking coordinate conventions and state/action temporal consistency before modifying a recipe. If a genuine missing condition is established, target that condition in a later separately admitted data intervention; if comparable condition coverage and alignment are demonstrated, state-conditioning/optimization becomes the next controlled hypothesis. This evidence takes priority over automatically launching the previously proposed40-minute9k LR-hold fork. No additional diagnostic or training job is launched by this recommendation.

Execution used `direnv exec . python3 -B .agent-local/work/learning-evaluation-loop/train-heldout-diagnostic.py --supervise` from the owner worktree, reusing the retained supervisor and exclusive heavy lock; analysis used `direnv exec . python3 -B .agent-local/work/learning-evaluation-loop/analyze-train-heldout.py`. Native child3697792 exited0 in87.177s including56.096s admission; supervised duration90.006s. Peak process RSS was3,917,430,784bytes and scoped CUDA peak allocation977,684,480bytes. Five-second samples had minimum host availability6.685GiB, disk8.021GiB and maximum full-PSI0.68. Source/reference/model/approval input hashes remained unchanged. GPU compute was empty after exit, the lock was released and root received `msg_5611a2948e78`; physical-stop authority remains unchanged.

Canonical evidence is owner `.agent-local/work/learning-evaluation-loop/train-heldout-diagnostic-r1/`: raw `012000.json` SHA256 `be1ad52aca638cc264fb88c0c31c13f40ef67617d743c56a16a8067700f98735`, `analysis.json` SHA256 `e337f65243dea71be2ee27020c4d044eab1b2e5ae8cfce68793590c16d2a1189`, and `coverage-range-check.json` SHA256 `9476f7ca65ec089c47e5a4449d9adab5af44feda8d1726f0a17f4f7ad177c7af`, plus exact source/reference hashes, supervisor manifest, resource trace and immutable-input audit. Parent/data/split/inventory identities are the same as the completed rhythm40 comparison above. Helpers and runtime evidence remain ignored; no duplicate Task/Goal/session or separate planning authority was created.

## Lighting feedback: a separate photometric recipe hypothesis

The user's reported slight between-dataset lighting changes justify testing the existing `light-photometric` preset as a separate admitted recipe after root resolves data/task priorities. R7 used disabled transforms; continuation retains that complete recipe and rejects stochastic transforms. Do not enable augmentation in an existing continuation or reinterpret its RNG history. A matched warm-start pair may compare `none` versus `light-photometric` from the same parent, explicitly resetting both runs and matching data, TRAIN normalization, batch, resolved schedule, exposure and checkpoint/evaluation settings. This is preparation only; no additional raw26 or GPU run is ordered.

The existing preset offers brightness/contrast factors0.9–1.1, saturation0.8–1.2, hue±0.02 and sharpness0.8–1.2. Native weighted sampling selects exactly two distinct operations per call, in configured order; brightness and contrast are not guaranteed each time. The reader calls transforms separately per camera, so it does not model one shared, episode-constant lighting shift. The preset is a broad photometric robustness hypothesis, not a brightness-only causal test. Native factory coverage verifies TRAIN gets the transform, DEV gets none, and a small uint8 tensor preserves shape/dtype without input mutation. It does not establish learned robustness. The affine variant changes geometry and adds an unsupported explanation for lighting-only feedback; retain it outside this comparison.

Use natural, unaugmented development observations and the same frozen flow/action settings for the primary comparison. Evaluate any actually observed lighting-condition groups separately once their source/session identities and admission are available; avoid claiming generalization from synthetic jitter or repeatedly tuned raw26 DEV. Benefits should repeat across lighting groups without persistent per-joint/gripper degradation. If brightness-only attribution matters, declare a narrower existing-native brightness/contrast configuration as another recipe rather than quietly editing the named preset. Jitter cannot recover lost information from clipped highlights or prove that current errors are caused by lighting. These mechanisms follow the [official native transforms](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/transforms/transforms.py), [TRAIN/DEV factory](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/factory.py) and [Torchvision ColorJitter](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html); suitability for FR5 remains an empirical hypothesis.

Bound allocation using observed whole-chain cost plus the additional checkpoint/evaluation cost and reserve. Complete the fixed horizon unless resource, authority or numerical failure requires stopping; a single validation fluctuation is not an early-stop rule. Require sustained development evidence plus componentwise sampled-action qualification before proposing an extension, and preserve mixed outcomes without automatic promotion. Learning owns interpretation and cross-model comparisons, root owns the next resource/authority decision, and Rollout owns separately admitted physical trials. Exact commands, immutable artifact identities, costs and numerical decision thresholds stay in the native owner handoff.
# CPU source-condition followup (2026-09-12)

The completed 12k diagnosis was followed by a bounded read-only source audit, recorded in owner `.agent-local/work/learning-evaluation-loop/j6-source-neighborhood-r1/{report.md,evidence.json}`. Episodes 35–39 declare +35.890855° yaw absent from TRAIN 0–31; episodes 32–34 share −11.034425° with TRAIN 30–31. Episode 34 explicitly binds an out-of-dataset reposition to episode 35's changed XY/yaw. Workspace frame/sheet/family contracts remain unchanged. This supports a condition-coverage hypothesis, not yaw-only causality or a data corruption verdict.

All 24 heldout observations were matched to the nearest same-task row among all 22,299 TRAIN rows by six-joint absolute-position Euclidean distance. The ten q=.5/.9 observations in episodes 35–39 remain 14.726–20.244° away, versus 0.754–1.948° for their five q=.1 observations. These are configuration distances, not per-joint errors or semantic phases. Across 7,301 provenance rows (episodes 30–39), no state/reference bracket excludes its target; episode 34/35 J6 reference-minus-state maxima are under 1°. Physical calibration, historical executable identity and visual/XY/yaw confounding remain unresolved.

Next discriminating proposal: separately admitted missing-condition coverage with varied XY and both instructions, compared with matched-update existing-data continuation, while retaining the original heldout and processors. Do not promote heldout episodes into TRAIN or launch collection/training under this analysis authority. No GPU, evaluator edit, dataset mutation or physical effect was used. Local helper assertions checked row/match counts and source hashes before/after; no full regression was repeated.

## Candidate research: convergence is not coverage (September 13)

This is independent, bounded research preparation, not a new Rollout gate or
replacement for the open learned Pick → release/reset → diagnosis → targeted
Collection acceptance. The Thin handoff preserved D software preparation under
`.agent-local/work/lerobot-fr5/owner-20260913/research-software-prep/`; it performed
zero model loads/NFE. Its raw26/10k inputs are historical, not automatically the
right cohort for the current rhythm40/12k question. S and D utility are UNKNOWN.

Keep two hypotheses separate, never a weighted uncertainty score:

- **S:** a cheap TRAIN-only representation-distance support proxy can order
  existing physical coverage strata. Reuse frozen native representations and
  begin with a prototype or kNN distance, not a learned uncertainty estimator.
  Physical metadata is evaluation-only, never a new policy input. Use only strata
  that existing artifacts actually support; leave unavailable groups UNKNOWN.
- **D:** a same-endpoint coarse/refined solver defect can predict the effect of
  additional numerical refinement on identical observation/noise. Count all
  actual neural evaluations, including measurement overhead. Keep native fixed10
  as a behavioral reference and check that a higher-NFE numerical reference is
  itself sufficiently resolved; neither reference establishes task correctness.

Use existing TRAIN/EVAL splits, saved processors and analysis paths. Exclude
self-neighbors when checking TRAIN support; preserve episode/session dependence,
task/time-fraction and visual confounding rather than claiming every frame is
an independent trial. Do not equate S with true data sufficiency or D with policy
confidence. Freeze selection and comparison before inspecting results.

First work may inspect sources, prepare small scripts and run pure CPU checks in
parallel with product implementation. Model experiments use one explicitly
allocated process with bounded observations/output, no concurrent model owners,
new weights, dataset copies or manual measurement campaign. Actual runtime
readiness and available RAM/disk must be checked at allocation. Do not defer
independent software work merely because GPU use is unavailable.

Stop S if useful separation is absent and rescuing it requires learned estimators
or new labeling. Stop D if defect does not predict reference discrepancy or its
measurement cost defeats the intended saving. Keep either useful branch alone;
negative results do not authorize infrastructure expansion. Only if both survive
independently, examine their joint distribution without forcing four quadrants.
Scheduler, runtime support veto and recollection integration remain deferred
until those results justify them; poor support alone does not prove that extra
compute is useless. No new Router, telemetry or benchmark framework is required.

Portfolio interpretation follows the same evidence path: clearly separate the
working data engine, the candidate S/D idea and measured positive/partial/negative
results. A later joint support/defect plot is useful only if it represents real
matched observations. Preserve source/data/model/split/noise provenance so the
same evidence can be reinterpreted as the product evolves without rebuilding a
parallel presentation dataset or claiming unmeasured runtime benefit.

### First bounded S/D evidence — September 13

One original rhythm40/12k native model run completed in **87.252s**, including
68.539s loading, with 1,120 actual solver NFE and 32 additional image-encoder
calls. Peak RSS was 3,801,014,272 bytes and CUDA allocator peak 995,901,440 bytes.
The process exited zero and released the exclusive GPU allocation. No training,
live observation or robot command occurred. The first cohort was one q=.5
observation from TRAIN 2,3,10,11,18,19,26,27 and EVAL 32–39; q is time fraction,
not semantic phase. Saved native inputs/processors/noise remained unchanged;
fixed10 reproduced all retained 8×50×7 physical outputs exactly.

- **S: descriptive only, utility inconclusive.** Sampled-TRAIN and EVAL median
  visual distances were .21543 and .22100 with broad overlap. Only 8/32 TRAIN
  episodes were represented; leave-one-episode-out TRAIN queries had three
  same-instruction neighbors versus four for EVAL. This is not a calibrated
  support threshold or a clean physical-strata separation. Retain counterexamples
  and visual/XY/yaw/task confounding instead of promoting a favorable correlation.
- **D: reference unresolved, utility inconclusive.** E40–E80 residual was
  11.68–14.10% of E5–E80 discrepancy; none passed the predeclared 10% engineering
  screen. That screen is not a universal convergence bound or physical gate,
  and the near miss does not prove D invalid. The defect probe costs ten NFE,
  including five extra probes; bare fixed10 also costs ten and was closer to
  E80 in all eight observations. No immediate compute-efficiency win follows.
- **Interpretation counterexample:** near-constant J5 contributed 32.56% of the
  normalized E5–E80 squared discrepancy but only 3.88e-6 degrees physical RMS.
  Normalized numerical discrepancy is not physical importance. Do not silently
  retune axis weights or turn this observation into a new admission condition.

End this resource round: no scheduler, support veto, E160, larger estimator or
automatic extra run. These scoped results neither qualify nor block product
rollout. Subsequent integration needs new utility evidence, not more research
infrastructure. Existing local `sd-preparation-20260913/{results.md,summary.json}`
retains per-episode values, counterexamples and stdlib postprocessing checks;
the frozen raw JSONL SHA-256 is
`9621ff1e7d94e1495a66f68a5336870c33b8923188e8c1e711d5629454e24b18`.
Those files are under `.agent-local/work/lerobot-fr5/`, not public Git assets.
Portfolio may present this as inexpensive hypothesis discrimination alongside
the working data-engine path, not an adaptive-inference performance result.
