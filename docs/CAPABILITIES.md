# Capability status

This document distinguishes implemented code from validated functionality.

| Area | Status | Notes |
|---|---|---|
| Versioned recording shards | Implemented and unit-tested | NPZ shards, atomic writes, per-environment buffers, checksum/schema validation, and legacy pickle read compatibility |
| Config validation | Implemented and unit-tested | Top-level typo detection, semantic validation, and executable tier profiles |
| Offline CPU tests | Implemented | No model downloads or game runtimes required |
| Chrome Dino training | In repair | Real environment must be validated before training claims |
| ViZDoom PPO collection | In repair | Transition isolation is implemented; action/reward/scenario fidelity remains work in progress |
| Stable Diffusion training | In repair | Configured optimizer/scheduler/accumulation, episode-held-out validation, and resumable checkpoints are implemented; no reproduced checkpoint exists |
| Decoder fine-tuning | Experimental | Artifacts are provenance-tagged and loadable; end-to-end evaluation remains incomplete |
| Distillation | Quarantined / unavailable | Command fails explicitly; do not use for research claims |
| FVD | Experimental / unavailable | Current implementation is not paper-comparable |
| Human evaluation | Experimental / unavailable | Current implementation is not a blinded study tool |
| Hierarchical memory | Experimental / unavailable | It requires a trained conditioning path |
| Text conditioning and image modding | Experimental / unavailable | No validated end-to-end training/evaluation path |

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for promotion criteria.
