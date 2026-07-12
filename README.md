# GameNGen: Neural Game Engine Research Implementation

This repository is an in-progress, test-driven implementation of the ICLR 2025
paper [“Diffusion Models Are Real-Time Game Engines”](https://arxiv.org/abs/2408.14837).
It does not include trained weights, reproduced benchmarks, or a paper-faithful
claim. See [the implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md),
[capability status](docs/CAPABILITIES.md), and
[paper-fidelity status](docs/PAPER_FIDELITY.md) before running expensive experiments.

## Status

- Versioned NPZ recordings, checksum validation, isolated environment buffers,
  episode-disjoint validation splitting, and resumable diffusion checkpoints are implemented.
- Tier configuration profiles contain paper-reference settings, but no local
  reproduction has been validated.
- Chrome Dino, ViZDoom, Stable Diffusion, decoder training, FVD, distillation,
  text conditioning, hierarchical memory, and modding remain research paths,
  not supported product features.
- Paper-reported 20/50 FPS and quality metrics are not local results.

## Install

```bash
git clone https://github.com/ReverseZoom2151/gamengen-v2.git
cd gamengen-v2

# Base package and development checks
pip install -e '.[dev]'

# Runtime extras as required by the path being exercised
pip install -e '.[doom,dino,metrics]'

# Offline foundation checks; no model download or game runtime is required
python -m pytest -q
```

Python 3.10+ is required. GPU model training requires an independently
configured supported PyTorch/CUDA installation. Optional game and model
dependencies are intentionally not installed by the offline test path.
See [dependency policy](docs/DEPENDENCY_POLICY.md) for version and platform
compatibility rules, and [ViZDoom integration notes](docs/VIZDOOM_INTEGRATION.md)
for the DOOM runtime, assets, map input, and transition boundary.
See [human-evaluation guidance](docs/HUMAN_EVALUATION.md) before running a
blinded participant study.
See the [security policy](docs/SECURITY.md) for CI gates and responsible
disclosure guidance.

## Commands

The public Python package is `gamengen`; legacy `src` imports remain compatible
while migration is completed. Development entry points are:

```bash
gamengen-dino --config configs/tier1_chrome_dino.yaml
gamengen-doom --config configs/tier2_doom_lite.yaml
gamengen-train --config configs/tier1_chrome_dino.yaml
```

These are development entry points, not guarantees of a ready-to-train or
paper-reproducing system. They need real game runtimes, data recordings,
external model access, and environment-specific validation.

Run a mode-specific dependency preflight before starting one:

```bash
python -m src.preflight --config configs/tier2_doom_lite.yaml --mode train
```

## Architecture

The intended pipeline is:

```text
game environment → canonical transitions → validated recording shards
                 → action-conditioned diffusion training → autoregressive sampling
```

The current diffusion path adapts Stable Diffusion 1.4 with frame-history
latents, discrete-action conditioning, observation-condition dropout, noise
augmentation, velocity prediction, and DDIM sampling. The paper profile sets
64-frame context, 4 DDIM sampling steps, and the reported optimizer/data scale;
those values alone do not establish paper fidelity.

## Data and evaluation integrity

- Recorders store `observation_t, action_t, reward_t, observation_t+1`, terminal
  flags, metadata, manifests, and checksums in atomic NPZ shards.
- Dataset loading verifies new-format structure and checksums before training.
- Validation splitting is episode-disjoint.
- FVD refuses the former invalid generated-vs-generated comparison and requires
  separately supplied real/fake trajectories plus pretrained I3D weights.
- Human-study protocols keep the answer key separate from the public protocol;
  no human-study results are included.

## Tiers

| Tier | Scope | Current status |
|---|---|---|
| 1 | Chrome Dino data/environment validation | In repair |
| 2 | Small DOOM integration | In repair |
| 3 | Full DOOM paper-reproduction target | Planned research |

## Contributing

Run `python -m pytest -q` before submitting changes. Keep capability claims
backed by reproducible artifacts and update the roadmap or capability status
when changing a research path.

## License and citation

Released under the [MIT License](LICENSE). If this work informs research,
cite the original GameNGen paper:

```bibtex
@inproceedings{valevski2025diffusion,
  title={Diffusion Models Are Real-Time Game Engines},
  author={Valevski, Dani and Leviathan, Yaniv and Arar, Moab and Fruchter, Shlomi},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://arxiv.org/abs/2408.14837}
}
```
