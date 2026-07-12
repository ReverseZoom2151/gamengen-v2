# GameNGen v2: Remediation, Paper-Fidelity, and Feature Roadmap

**Status:** Planning baseline<br>
**Created:** 2026-07-13<br>
**Target paper:** Valevski et al., *Diffusion Models Are Real-Time Game Engines*, ICLR 2025 / arXiv v2<br>
**Paper:** [`paper/GameNGen_ICLR2025.pdf`](../paper/GameNGen_ICLR2025.pdf)<br>
**Scope:** The main repository, informed by the four local reference repositories under `repos/`

## 1. Executive summary

The current repository should be treated as a research scaffold inspired by GameNGen, not as a complete, production-ready, or paper-faithful reproduction. It contains useful implementations and a broad set of planned capabilities, but several foundational paths are incorrect, incomplete, internally inconsistent, or represented by placeholders.

The implementation order in this roadmap is deliberate:

1. Make public claims truthful and establish dependable engineering gates.
2. Define correct environment, action, transition, dataset, and configuration contracts.
3. Repair the core action-conditioned diffusion implementation.
4. Make training deterministic, resumable, scalable, and configuration-driven.
5. Build a trustworthy evaluation system.
6. Reproduce the paper's method and ablations where practical.
7. Integrate the strongest ideas from the external repositories.
8. Add experimental backends and deployment features only after correctness is demonstrated.

Feature expansion before the first five stages would risk building on corrupted trajectories, ineffective conditioning, invalid metrics, and irreproducible training.

## 2. Goals and success criteria

### 2.1 Primary goals

- Produce a reliable, maintainable implementation of an action-conditioned neural game engine.
- Make the DOOM pipeline as faithful as practical to the ICLR 2025 paper.
- Clearly separate exact paper settings from consumer-hardware adaptations.
- Support deterministic CPU tests and realistic GPU integration tests.
- Make every documented configuration option either functional or rejected as unsupported.
- Prevent silent dataset corruption, unsafe artifact loading, and misleading evaluation.
- Create a foundation on which experimental features can be evaluated honestly.

### 2.2 Definition of “paper-faithful”

A subsystem may be labelled paper-faithful only when:

- its algorithm and conditioning contract match the paper;
- its important hyperparameters match the paper or the deviation is documented;
- it is exercised by automated tests;
- it has been used in at least one reproducible run;
- its output is measured using the paper's evaluation protocol where possible; and
- its limitations and hardware differences are disclosed.

Configuration values alone do not establish fidelity. The executing code must consume and implement them.

### 2.3 Two supported operating profiles

The finished project should distinguish two profiles explicitly:

#### Paper profile

- DOOM-focused.
- Stable Diffusion 1.4 architecture.
- Context length 64.
- Paper action, observation, reward, noise, CFG, decoder, and evaluation contracts.
- Large-scale distributed training.
- Intended for reproduction research, not ordinary workstation use.

#### Practical profile

- Reduced datasets and batch sizes.
- Gradient accumulation and one or several GPUs.
- Optional lightweight model backend.
- Shorter experiments with the same data and conditioning semantics.
- No claim of matching paper metrics unless actually measured.

## 3. Current-state assessment

### 3.1 Repository condition

- The local `main` branch tracks `origin/main`.
- The only existing worktree difference at planning time is the local `/repos/` entry in `.gitignore`.
- The external repositories are local research references and are not intended to be committed as part of the main project.
- Python syntax parses successfully across the workspace.
- Pytest collects 19 main-project tests, but the current suite is not a reliable quality signal.
- The development linters declared in `requirements-dev.txt` are not installed in the current environment.

### 3.2 Critical correctness findings

#### Environment and agent

- Tier 1 trains against `SimpleDinoEnv`, which emits synthetic/random observations rather than Chrome Dino gameplay.
- The ViZDoom action vectors are hard-coded instead of being derived from enabled scenario buttons.
- ViZDoom game variables are interpreted by array position rather than named `GameVariable` values.
- Frame-skip rewards are accumulated and then discarded in favor of only the last reward.
- The paper reward is incomplete.
- The PPO agent does not receive the paper's map input or previous 32 actions.
- Tier 2 and Tier 3 configuration fields do not line up with the keys consumed by PPO training.

#### Recording and datasets

- Eight PPO environments write into one shared episode buffer, interleaving unrelated trajectories.
- DQN and PPO use different action/frame alignment semantics.
- Batch naming can overwrite the previously written full batch when a partial final batch is flushed.
- Restarting collection can reuse earlier batch IDs.
- `compress` is recorded in metadata but no compression is performed.
- `finalize()` does not guarantee that an unfinished current episode is handled explicitly.
- PPO observations can be channel-first while the diffusion dataset assumes channel-last input.
- Dataset startup unpickles every shard and builds one Python dictionary per overlapping window.
- A cache miss can reload a full pickle batch for a single sample.
- Unrestricted pickle loading is unsafe for untrusted data.

#### Core diffusion model

- Velocity prediction, scheduler device/dtype handling, and observation CFG dropout have been corrected; fixture-level mathematical verification remains to be added.
- The interactive player applies a user's action after generating the frame, creating a one-frame control lag.
- The same action-timing error exists in modding and video-export paths.
- Mixed-precision evaluation does not consistently use autocast or cast inputs to component dtype.
- The configured inference context-noise level is not fully integrated.
- Decoder artifacts are now provenance-tagged and inference-loadable; no end-to-end quality validation exists yet.

#### Training

- Optimizer selection, gradient accumulation, scheduler, warmup, held-out validation, atomic checkpointing, and scaler/RNG resume state are implemented; distributed and sampler resume support remains outstanding.
- Several configured logging, debugging, profiling, and W&B behaviors are ignored.
- Seed handling is applied to core diffusion training; full worker/environment reproducibility and data-loader sampler state remain outstanding.

#### Evaluation and advanced modules

- The FVD trajectory helper assigns a generated trajectory as its own real reference.
- The fallback I3D implementation does not provide a paper-comparable FVD score.
- The human-evaluation code does not implement a real blinded comparison workflow.
- Distillation does not execute the teacher correctly and does not train the fake-score model correctly.
- Hierarchical memory does not affect generation and assumes incompatible latent spatial dimensions.
- Text conditioning contains an untrained projection path.
- Interactive inference is hard-coded to Dino behavior even when supplied DOOM configs.

#### Packaging, CI, and documentation

- The installed package is named `src` rather than `gamengen`.
- `setup.py` reports version `0.1.0` while documentation claims `1.0.0`.
- Optional and required dependencies are mixed together.
- ViZDoom is used by Tier 2/3 but absent from the primary dependency file.
- CI runs a script that expects missing paths, treats CPU-only execution as failure, and may fail to import the project when executed directly.
- Several pytest functions catch errors and return `False`, which does not fail pytest correctly.
- Some tests instantiate or download Stable Diffusion and are not suitable for offline CI.
- README claims such as “production-ready,” “all tests passing,” “12 guides,” and paper-equivalent evaluation are not currently supported.

## 4. Paper-fidelity matrix

The following matrix uses the April 2025 ICLR/arXiv v2 paper as the authority.

| Paper requirement | Paper setting | Current status | Required action |
|---|---|---|---|
| Data source | All agent training and evaluation trajectories | Partial and corruptible | Repair per-env recording and record policy phase/evaluation provenance |
| Agent algorithm | PPO through Stable-Baselines3 | Present | Retain after repairing inputs, reward, seeding, and collection |
| Agent observation | Frame and in-game map at 160×120 | Missing | Add a controlled Dict observation and custom feature extractor |
| Agent history | Previous 32 actions | Missing | Add action-history observation and encoder |
| Parallelism | 8 games | Configured in code | Preserve with isolated per-env buffers and ranked seeds |
| Agent training scale | 10M environment steps | Config says 50M | Correct the paper profile to 10M |
| Smooth play | Repeat action for 4 frames and bias action repetition | Only frame repeat is partial | Add configurable repetition bias and test it |
| Reward | Ten-term DOOM reward from Appendix A.3 | Partial/incorrect | Implement all terms using named game variables and controlled scenario data |
| Training corpus | 900M frames in v2 paper | Config says 70M | Correct documentation/config; permit practical reduced profiles |
| Resolution | 320×240 padded to 320×256 | Resized to 320×256 | Add deterministic padding and record original/padded geometry |
| Base model | Stable Diffusion 1.4 | Present | Pin revision and artifact provenance |
| U-Net | All parameters unfrozen | Present conceptually | Verify trainable parameter set in tests |
| Observation conditioning | VAE latents concatenated along channels | Present | Verify ordering, padding, dtype, and context contract |
| Action conditioning | One learned token per action through cross-attention | Present conceptually | Add current-action timing tests and temporal positions |
| Text conditioning | Removed | Mostly removed | Ensure no unused/random text path in paper profile |
| Diffusion objective | Velocity prediction with linear schedule | Inconsistent | Explicitly configure/checkpoint/test `v_prediction` |
| Context length | 64 frames and 64 actions | Tier 3 config only | Make schema-enforced in paper profile |
| Noise augmentation | Uniform to 0.7, 10 buckets | Present conceptually | Verify exact formula, bucket edges, inference control, and ablation |
| CFG training | Drop observation condition with probability 0.1 | Missing | Implement observation-only dropout |
| CFG inference | Observation CFG only, weight 1.5 | Partial | Use a trained unconditional observation branch while preserving action conditioning |
| Optimizer | Adafactor, no weight decay | Configured but ignored | Use a maintained Adafactor implementation |
| Denoiser batch | 128 | Configured ambiguously | Define global versus per-device batch and distributed accumulation |
| Learning rate | Constant 2e-5 | Partially configured | Implement constant scheduler and validate effective LR |
| Gradient clipping | 1.0 | Hard-coded | Drive it from validated config |
| Denoiser training | 700k steps | Configured | Add exact resume and distributed execution |
| Training hardware | 128 TPU-v5e devices | Not reproducible locally | Document deviation; implement GPU distributed equivalent without claiming identical hardware |
| Decoder training | Decoder only, MSE, batch 2,048 | Partial | Integrate artifact loading, optimizer settings, validation, and provenance |
| Sampling | DDIM | Present | Validate scheduler contract and sampler determinism |
| Standard inference | 4 denoising steps | Present in config | Benchmark on target hardware before FPS claims |
| Distillation | Three U-Nets, 1,000 steps, batch 128 | Nonfunctional and config differs | Reimplement from the described objective and validate independently |
| Teacher-forced metrics | PSNR/LPIPS on 2,048 held-out trajectories from five levels | Missing protocol | Build immutable held-out manifests and metric runner |
| Autoregressive FVD | 512 trajectories, lengths 16 and 32 | Invalid implementation | Use verified pretrained features and distinct real/generated distributions |
| Human evaluation | Blinded side-by-side short clips | Stub | Build a functional blinded tool with randomized sides and recorded decisions |
| Context ablation | 1/2/4/8/16/32/64 | Missing | Add experiment sweep after the baseline is validated |
| Noise ablation | With and without context noise over 64-step rollouts | Missing | Add experiment sweep and plots |
| Agent/random ablation | Agent and uniform-random corpora | Missing | Add reproducible data manifests and comparison workflow |

## 5. Lessons from the external repositories

### 5.1 Features to adapt

| Feature | Reference | Decision | Reason |
|---|---|---|---|
| Per-environment recording and episode IDs | `GameNGen-main` | Adapt immediately | Corrects the main project's most severe data corruption issue |
| VAE posterior/latent precomputation | `GameNGen-main` | Adapt in Phase 5 | Removes repeated VAE encoding from each training step |
| Parquet and Hugging Face dataset adapters | `GameNGen-main`, `gameNgen-repro` | Optional adapter | Useful for distribution and inspection, but not mandatory for the internal storage format |
| Accelerate-based distributed training | `GameNGen-main` | Adapt | Provides practical GPU data parallelism and accumulation |
| Playable DOOM interface and action logging | `GameNGen-main` | Adapt | Makes Tier 2/3 genuinely interactive and testable |
| Fixed rollout fixtures | Mario diffusion repo | Reimplement cleanly | Enables reproducible comparison and regression testing |
| EMA and validation-driven checkpoints | Mario diffusion repo | Reimplement cleanly | Improves long-run stability and recovery |
| Counterfactual all-actions grid | Mario diffusion repo | Reimplement cleanly | Detects ignored action conditioning and control collapse |
| Temporal action positions | Mario diffusion repo | Reimplement and ablate | Makes repeated actions distinguishable by historical position |
| Compact diffusion backend | Mario and Tetris repos | Clean-room design | Enables CPU/small-GPU development and fast integration tests |
| EDM/Karras sampling | Tetris repo | Experimental backend | Useful for controlled sampler comparisons after DDIM correctness |
| ONNX and browser inference | Tetris repo | Defer | Most practical for a compact or distilled backend |

### 5.2 Patterns not to copy

- Do not copy source from the Mario repository without a clear compatible license.
- Do not keep pickle as the primary long-term dataset format.
- Do not load arbitrary Python model objects from untrusted checkpoints.
- Do not use `strict=False` without reporting and validating missing/unexpected keys.
- Do not delete recording directories automatically.
- Do not accumulate trajectories with repeated `np.append` calls.
- Do not randomly split overlapping windows; split complete episodes first.
- Do not retain hard-coded global paths, `sys.path` mutation, duplicated sampler logic, or deprecated Gym APIs.

## 6. Target architecture

The exact package names may be refined during implementation, but the target separation should resemble:

```text
gamengen/
  cli/
    collect.py
    train_agent.py
    train_model.py
    evaluate.py
    play.py
    export.py
  config/
    models.py
    loader.py
    validation.py
  environments/
    base.py
    factory.py
    chrome_dino.py
    vizdoom.py
    rewards.py
    actions.py
  agents/
    dqn.py
    ppo.py
    observations.py
  data/
    schema.py
    recorder.py
    manifest.py
    shards.py
    dataset.py
    latent_cache.py
    migration.py
  models/
    conditioning.py
    diffusion.py
    samplers.py
    decoder.py
    artifacts.py
    compact/
  training/
    engine.py
    optimizer.py
    checkpoint.py
    distributed.py
    logging.py
  evaluation/
    image_metrics.py
    rollout.py
    fvd.py
    action_grid.py
    human_eval.py
  experimental/
    distillation/
    memory/
    text_conditioning/
    modding/
tests/
  unit/
  data_contract/
  integration/
  gpu/
configs/
  paper/
  practical/
```

Important boundaries:

- Environments emit canonical transitions; recorders do not infer their semantics.
- Storage schemas are independent from training datasets.
- Model components do not parse YAML or mutate paths.
- Training engines receive validated configuration objects.
- Evaluation never uses training loaders implicitly.
- Experimental modules cannot be advertised as supported until they pass their own gates.

## 7. Phased implementation plan

## Phase 0 — Truth and safety baseline

### Objectives

- Stop exposing unsupported claims as completed functionality.
- Establish an explicit capability and fidelity vocabulary.
- Prevent users from spending substantial compute on known-broken paths.

### Work

- Rewrite project status and tier descriptions.
- Replace “Ready” with supported/experimental/planned labels.
- Correct source-line, guide-count, storage, hardware, and timing claims.
- Explain paper profile versus practical profile.
- Disable or clearly quarantine nonfunctional distillation and evaluation commands.
- Add a known-limitations document and `SECURITY.md`.
- Add a model/data provenance policy.

### Completion gate

- Every README command is backed by an automated smoke test or explicitly labelled unavailable.
- No benchmark is presented as achieved without a stored run manifest and artifact.

## Phase 1 — Packaging, configuration, and CI

### Objectives

- Make the repository installable and testable without network or GPU access.
- Create a single source of truth for configuration.

### Work

- Add `pyproject.toml` and migrate to a `gamengen` package.
- Add console entry points.
- Define Pydantic or dataclass-based config models.
- Reject unknown keys and unsupported combinations.
- Add semantic validation for resolution, context length, actions, sampler, dtype, batch size, and device.
- Split dependency extras.
- Add constraints/lock strategy for CPU and supported CUDA environments.
- Configure Ruff, formatter, type checker, pytest, coverage, and pre-commit.
- Rebuild CI around offline CPU unit tests.
- Pin GitHub Actions to immutable revisions.
- Add dependency, secret, and static security scanning.

### Tests

- Config parse and invalid-config tests.
- Wheel/sdist build and clean-install tests.
- CLI help and packaged-config discovery.
- Import tests with every optional extra absent.

### Completion gate

- Clean installs pass on Python 3.10 and 3.12.
- Offline CPU CI passes without downloading models.
- Lint, formatting, typing, tests, and package build are required checks.

## Phase 2 — Transition, environment, and dataset contracts

### Objectives

- Guarantee that recorded data is correctly aligned, isolated, recoverable, and scalable.

### Canonical transition

```text
observation_t
action_t
reward_t
observation_t_plus_1
terminated_t
truncated_t
episode_id
environment_id
step_id
scenario_id
policy_phase
metadata
```

The diffusion sample contract must state exactly which action produces the target frame.

### Work

- Implement one recorder stream per environment.
- Use monotonic shard IDs and atomic temporary-file replacement.
- Add append/resume detection.
- Define schema version and migration hooks.
- Add per-shard checksums and a dataset manifest.
- Store episode-level split assignments.
- Normalize HWC and CHW at the environment boundary.
- Choose a scalable internal format after a benchmark spike:
  - compressed frame/video shards plus Parquet metadata;
  - Zarr arrays plus Parquet metadata; or
  - WebDataset-style shards.
- Retain a read-only legacy pickle migration utility.
- Implement scenario-derived ViZDoom actions.
- Implement named game variables and the complete paper reward.
- Add 320×240-to-320×256 padding.
- Make real Chrome Dino the Tier 1 environment; keep a deterministic mock only in tests.

### Tests

- Eight simulated environment streams remain isolated.
- Partial final shards never overwrite full shards.
- Restarting continues from the next shard and episode IDs.
- HWC and CHW fixtures yield identical canonical frames.
- Every action maps to the expected enabled buttons.
- Reward terms are checked independently.
- Dataset corruption is detected by checksum/schema validation.
- Episode-level splits contain no cross-split windows.

### Completion gate

- A short DQN and PPO collection produces validated manifests and replayable episodes.
- Collection can be interrupted and resumed without duplication or loss.

## Phase 3 — Core diffusion correctness

### Objectives

- Make training, sampling, and interactive conditioning mathematically and temporally consistent.

### Work

- Explicitly set and serialize scheduler prediction type.
- Verify linear noise schedule compatibility with the paper profile.
- Move/cast scheduler coefficients safely.
- Formalize ordering of context frames and action tokens.
- Include the active/current action before next-frame generation.
- Add temporal positions for action tokens and ablate their impact.
- Implement observation-condition dropout with probability 0.1.
- Preserve action conditioning during the CFG unconditional-observation pass.
- Implement context noise augmentation with exact bucket semantics.
- Make dtype handling explicit for VAE, U-Net, embeddings, and inputs.
- Include fine-tuned decoder identity/weights in the artifact contract.
- Pin the Stable Diffusion base revision.
- Separate model construction from artifact loading.

### Tests

- Known timestep fixtures verify velocity target construction.
- Scheduler `step()` receives the expected prediction type.
- Current action affects the frame being generated, not the following frame.
- An all-actions counterfactual fixture detects ignored actions.
- CFG conditional and observation-unconditional branches receive expected inputs.
- CPU FP32 and mocked CUDA/AMP paths preserve dtype/device contracts.
- Save/load round trips reproduce output with fixed noise.

### Completion gate

- A tiny deterministic model can overfit a small transition fixture.
- Autoregressive rollout follows action changes without a one-frame lag.

## Phase 4 — Training reliability and reproducibility

### Objectives

- Make long-running training configuration-driven, resumable, measurable, and distributed.

### Work

- Route optimizer construction through the validated config.
- Use maintained AdamW and Adafactor implementations.
- Implement gradient accumulation correctly.
- Add scheduler and warmup support.
- Drive gradient clipping from config.
- Add Accelerate or equivalent distributed orchestration.
- Define global/per-device/effective batch size explicitly.
- Seed Python, NumPy, PyTorch, CUDA, environments, workers, samplers, and PPO.
- Add separate train and validation loaders.
- Add EMA weights.
- Add latest, best, and periodic checkpoints.
- Save complete resume state.
- Use atomic checkpoint writes and numeric retention.
- Record code revision, model revision, data manifest, configuration hash, dependencies, and hardware.
- Integrate structured logging and optional W&B.

### Tests

- Gradient accumulation matches a larger non-accumulated batch in a deterministic fixture.
- Resume produces equivalent early losses and optimizer state to uninterrupted training.
- Distributed smoke test produces one coherent artifact set.
- Best/latest retention works across digit boundaries.

### Completion gate

- A tiny GPU run completes train, validation, checkpoint, resume, and inference under FP32 and supported mixed precision.

## Phase 5 — Scalable latent pipeline

### Objectives

- Avoid repeatedly encoding 64 context frames during every training step.
- Support large datasets without materializing all window descriptors.

### Work

- Precompute VAE posterior parameters or sampled latents with pinned VAE revision.
- Store latents in safe, sharded tensor/array format.
- Preserve links to source frames, actions, episode, and preprocessing metadata.
- Use arithmetic window indexing instead of one Python dictionary per window.
- Add bounded open-shard and decoded-sample caches.
- Add optional Hugging Face import/export adapters.
- Add a migration/verification command that compares cached and on-the-fly encodings.

### Benchmarks

- Storage bytes per frame/transition.
- Shard-open and sample latency.
- Dataloader examples per second.
- Host RAM and page-cache behavior.
- GPU utilization and step time.
- Checkpoint time and size.

### Completion gate

- Latent-native and on-the-fly pipelines produce statistically equivalent training inputs.
- The latent pipeline materially improves measured training throughput.

## Phase 6 — Trustworthy evaluation

### Objectives

- Make quality claims reproducible and resistant to data leakage or metric misuse.

### Work

- Build immutable held-out manifests by episode and level.
- Add teacher-forced PSNR and LPIPS evaluation.
- Add 16/32/64-frame autoregressive rollouts.
- Verify real and generated inputs are distinct before FVD computation.
- Adopt a verified pretrained FVD feature implementation.
- Pin preprocessing, resize, normalization, and temporal sampling.
- Add identical-versus-perturbed metric sanity fixtures.
- Add fixed-seed rollout videos.
- Add action counterfactual grids.
- Add sampling-step quality/latency sweeps.
- Build a blinded human-evaluation UI with side randomization and auditable results.
- Store evaluation reports with model/data/config hashes.

### Paper ablations

- Context: 1, 2, 4, 8, 16, 32, 64.
- Context noise: enabled versus disabled.
- Data policy: trained agent versus uniform random.
- Sampling steps: distilled, 1, 2, 4, 8, 16, 32, 64.

### Completion gate

- Identical videos yield near-zero distribution distance.
- Perturbations monotonically worsen at least the appropriate sanity metrics.
- Repeated evaluation with fixed artifacts is stable within documented tolerance.

## Phase 7 — Paper reproduction campaign

### Objectives

- Move from method-level fidelity to an evidence-backed reproduction attempt.

### Work

- Implement the paper PPO observation architecture.
- Implement the complete reward and smooth-action policy behavior.
- Train for the paper's 10M environment steps in the paper profile.
- Record all training/evaluation trajectories with policy-phase metadata.
- Decide and document the feasible corpus scale; paper v2 reports 900M frames.
- Train the denoiser for 700k steps with the paper optimizer and effective batch.
- Fine-tune the decoder separately.
- Evaluate on the paper-style level and trajectory splits.
- Produce a reproducibility report distinguishing matched, approximated, and unavailable conditions.

### Completion gate

- The final report links every reported value to immutable artifacts and run manifests.
- “Paper-faithful” is used only for components whose algorithmic and validation gates pass.

## Phase 8 — Advanced and external-repo-derived features

### Recommended after the baseline

- Full DOOM playable interface with configurable key mapping and action logs.
- Curriculum-aware and skill-stratified collection.
- Safe Hugging Face model/dataset publishing.
- Lightweight CPU/small-GPU diffusion backend.
- Sampler abstraction with experimental EDM/Karras support.
- Compact-model ONNX export and local browser demo.

### Advanced modules requiring redesign

- Three-U-Net one-step distillation.
- Hierarchical memory with a trained conditioning path.
- Text conditioning with a defined training objective.
- Image-based modding with bounded compositing and evaluation.
- Multi-scenario conditioning and scenario identity handling.

Each advanced module should live under an experimental namespace until it has:

- a written objective;
- a training path;
- unit and integration tests;
- a reproducible example; and
- a metric demonstrating benefit.

## 8. Verification matrix

| Gate | Required verification |
|---|---|
| Static | Ruff/format/import checks, typing, YAML/schema validation, compile checks |
| Unit CPU | Config, embeddings, action mapping, rewards, DQN, checkpoint retention, metrics |
| Data contract | Multi-env isolation, alignment, layouts, partial shards, append/resume, checksums |
| Packaging | Build wheel/sdist, install in clean Python 3.10/3.12, verify CLI and packaged configs |
| Offline CI | No CUDA, Chrome, ViZDoom, network, or model download required |
| GPU smoke | Tiny FP32 and mixed-precision train/evaluate/resume cycle |
| Environment integration | Short real Dino and optional ViZDoom episodes |
| Diffusion integration | Tiny overfit, deterministic sampling, action responsiveness, autoregressive rollout |
| Evaluation | Distinct real/generated data, known sanity fixtures, fixed preprocessing |
| Reproducibility | Same seed/config/manifest produces matching early behavior |
| Security | Safe deserialization, dependency audit, CodeQL, secret scan, pinned actions/revisions |
| Performance | Storage, dataloader, RAM, VRAM, step time, checkpoint time, sustained inference FPS |
| Documentation | Execute every quick-start command in a clean environment |

## 9. Milestones and dependency order

### Milestone A — Honest, installable project

Includes Phases 0 and 1.

Result: trustworthy documentation, real packaging, typed configs, and reliable offline CI.

### Milestone B — Correct data foundation

Includes Phase 2.

Result: real environments and valid, replayable, resumable trajectories.

### Milestone C — Correct neural engine

Includes Phases 3 and 4.

Result: mathematically consistent diffusion training and deterministic long-running jobs.

### Milestone D — Scalable training and credible metrics

Includes Phases 5 and 6.

Result: efficient latent training and evaluation capable of supporting research claims.

### Milestone E — Reproduction evidence

Includes Phase 7.

Result: a documented reproduction attempt with explicit matched and unmatched conditions.

### Milestone F — Research extensions and deployment

Includes Phase 8.

Result: optional compact models, new samplers, distillation, memory, modding, and browser deployment.

## 10. First implementation batch

The first implementation batch should remain narrow enough to review safely while removing the most dangerous failure modes.

### Proposed Batch 1

1. Correct project status documentation and quarantine invalid commands.
2. Add typed configuration validation without moving the package yet.
3. Replace false-return tests with real assertions.
4. Repair CI to run offline CPU tests.
5. Define the canonical transition schema.
6. Fix recorder shard overwrites and current-episode handling.
7. Add per-environment recorder buffers.
8. Add regression tests for all recorder defects.

### Batch 1 exit conditions

- CI passes from a clean environment.
- No test reports success by returning a false value.
- Multi-env recordings cannot interleave episodes.
- Final partial shards cannot overwrite earlier data.
- Dataset manifests identify schema version, shard IDs, and episode IDs.

Package migration and diffusion changes should follow in separate reviewable batches once this foundation is green.

## 11. Risks and decisions requiring explicit confirmation

### Dataset format

A short benchmark should choose between compressed video plus metadata, Zarr, and WebDataset-style shards. Parquet is suitable for metadata and small encoded values but is not automatically the best container for hundreds of millions of raw image frames.

### Paper scale

The v2 paper reports 900M frames and 128 TPU-v5e devices. Matching the algorithm does not imply matching this infrastructure. The practical profile must remain honest about scale and expected quality.

### Backward compatibility

Legacy pickle datasets and checkpoints may be supported only through explicit migration commands. New code should not silently trust or reinterpret them.

### External code licensing

External repositories are references for architecture and experiments. Concepts may be adapted, but source should only be copied when its license clearly permits it and attribution obligations are satisfied.

### Experimental scope

Distillation, memory, text conditioning, and modding must not delay the correctness of the core teacher-forced and autoregressive model.

## 12. Documentation deliverables

As implementation progresses, this roadmap should be accompanied by:

- `docs/CAPABILITIES.md` — supported versus experimental features.
- `docs/PAPER_FIDELITY.md` — claim-by-claim implementation and evidence matrix.
- `docs/DATA_FORMAT.md` — transition, manifest, shard, and migration schemas.
- `docs/REPRODUCIBILITY.md` — seeds, artifacts, manifests, and exact resume behavior.
- `docs/EVALUATION.md` — teacher-forced, autoregressive, FVD, and human protocols.
- `docs/SECURITY.md` or root `SECURITY.md` — safe artifact and dataset handling.
- `docs/EXPERIMENTS.md` — canonical run commands and measured results.

## 13. Roadmap maintenance

This document is a planning baseline, not evidence that the work is complete.

For each merged implementation batch:

1. Update the relevant phase status.
2. Link tests and run artifacts.
3. Record deviations from the paper.
4. Move features from experimental to supported only after their exit gate passes.
5. Remove obsolete compatibility paths rather than retaining two indefinite implementations.

Suggested status labels:

- **Planned** — designed but not implemented.
- **In progress** — implementation exists on an active branch.
- **Implemented** — code and unit tests exist.
- **Validated** — integration/reproduction evidence exists.
- **Paper-faithful** — method, settings, and evidence meet the definition in Section 2.2.
