# Paper-fidelity status

This report compares the current code to *Diffusion Models Are Real-Time Game
Engines*. It is an implementation-status document, not a reproduction result.

| Paper element | Current state | Evidence / limitation |
|---|---|---|
| Stable Diffusion 1.4 base | Pinned in profiles | All tier profiles pin `CompVis/stable-diffusion-v1-4` at revision `133a221b8aa7292a167afc5127cb63fb5005638b`; no local model artifact is included |
| 320×240 game frames padded to 320×256 | Implemented | ViZDoom contract center-pads instead of vertically stretching |
| 64-frame context | Configured in Tier 2/3 | No trained checkpoint validates long rollout behavior |
| Current action / action history | Implemented | Fixed-length current-action contract and learned temporal action positions are unit-tested |
| Counterfactual action responsiveness | Implemented probe | A deterministic grid changes only the current action and reports pairwise frame differences; it needs trained-artifact results before supporting a control claim |
| Context noise (max 0.7, 10 buckets) | Configured and implemented | Distribution/rollout ablation not yet run |
| Observation CFG dropout (0.1) | Implemented | Actions remain conditioned; no trained validation artifact |
| Velocity prediction | Implemented | Mathematical target has a unit fixture; no end-to-end model fixture |
| 4-step DDIM | Configured | Runtime speed/quality has not been measured locally |
| Adafactor, no weight decay | Configured | Uses maintained Transformers implementation when optional dependency is installed |
| EMA validation/sampling weights | Implemented enhancement | Checkpointed EMA is used for validation and inference when present; this is a stability enhancement, not a paper-matched setting |
| PPO, 8 parallel games, 10M steps | Partially implemented | Tier 3 uses 160×120 screen/automap observations and the prior 32 applied actions through PPO's multi-input policy. The paper does not quantify its action-repeat increase; Tier 3's 0.2 value is an explicit approximation. A real runtime run is still required |
| 900M recorded frames | Configured target | No such corpus is included |
| Decoder-only fine-tuning | Experimental | Artifacts are provenance-tagged/loadable; no quality result exists |
| One-step distillation | Unavailable | Explicitly quarantined because historical logic was invalid |
| FVD and human study | Unavailable as evidence | FVD requires paired trajectories/pretrained I3D; human protocol is blinded but no study exists |

## Reproduction gate

Do not describe this repository as paper-faithful until an immutable run bundle
links the exact code revision, pinned model revision, recording metadata and
episode split, training configuration, checkpoint, generated rollouts, and
evaluation outputs. The report must separately identify matched, approximated,
and unavailable hardware/data conditions.
