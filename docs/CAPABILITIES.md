# Capability status

This document distinguishes implemented code from validated functionality.

| Area | Status | Notes |
|---|---|---|
| Versioned recording shards | Implemented and unit-tested | NPZ shards, atomic writes, per-environment buffers, checksum/schema validation, and legacy pickle read compatibility |
| Config validation | Implemented and unit-tested | Top-level typo detection, semantic validation, and executable tier profiles |
| Offline CPU tests | Implemented | No model downloads or game runtimes required |
| Chrome Dino training | In repair | Real environment must be validated before training claims |
| ViZDoom PPO collection | In repair | Transition isolation plus a map/action-history PPO input contract are unit-tested; real ViZDoom runtime, reward, and scenario validation remain work in progress |
| Interactive game initialization | Implemented and unit-tested | Inference now selects real Chrome Dino or ViZDoom from the validated config; it no longer silently substitutes a placeholder Dino environment for DOOM |
| Stable Diffusion training | In repair | Configured optimizer/scheduler/accumulation, EMA validation/sampling weights, episode-held-out validation, and resumable checkpoints are implemented; no reproduced checkpoint exists |
| Action counterfactual grid | Implemented and unit-tested | Identical contexts are generated under selected current actions; zero pairwise change is reported as control collapse rather than a successful control result |
| Decoder fine-tuning | Experimental | Artifacts are provenance-tagged and loadable; end-to-end evaluation remains incomplete |
| Distillation | Quarantined / unavailable | Command fails explicitly; do not use for research claims |
| FVD | Experimental / unavailable | Invalid self-comparison is rejected; pretrained I3D and paired trajectory workflow remain required |
| Human evaluation | Experimental / unavailable | Blinded protocol support exists; no study UI or results exist |
| Hierarchical memory | Experimental / unavailable | Variable latent maps work; it still requires a trained conditioning path |
| Text conditioning and image modding | Experimental / unavailable | No validated end-to-end training/evaluation path |

See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for promotion criteria.
