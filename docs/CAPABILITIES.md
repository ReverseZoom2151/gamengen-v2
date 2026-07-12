# Capability status

This document distinguishes implemented code from validated functionality.

| Area | Status | Notes |
|---|---|---|
| Versioned recording shards | Implemented and unit-tested | NPZ shards, atomic writes, per-environment buffers, and legacy pickle read compatibility |
| Config validation | Implemented and unit-tested | Top-level typo detection and core semantic checks |
| Offline CPU tests | Implemented | No model downloads or game runtimes required |
| Chrome Dino training | In repair | Real environment must be validated before training claims |
| ViZDoom PPO collection | In repair | Transition isolation is implemented; action/reward/scenario fidelity remains work in progress |
| Stable Diffusion training | In repair | Scheduler and CFG corrections are underway; no reproduced checkpoint exists |
| Decoder fine-tuning | Experimental | Artifact integration and evaluation are incomplete |
| Distillation | Experimental / unavailable | Do not use for research claims |
| FVD | Experimental / unavailable | Current implementation is not paper-comparable |
| Human evaluation | Experimental / unavailable | Current implementation is not a blinded study tool |
| Hierarchical memory | Experimental / unavailable | It requires a trained conditioning path |
| Text conditioning and image modding | Experimental / unavailable | No validated end-to-end training/evaluation path |

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for promotion criteria.
