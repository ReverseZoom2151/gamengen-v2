# GameNGen

<p align="center"><img src="doom-guy.gif" width="64" alt="DOOM Guy"></p>

An in-progress, test-driven research implementation inspired by the ICLR 2025
paper [*Diffusion Models Are Real-Time Game Engines*](https://arxiv.org/abs/2408.14837).

GameNGen is building an action-conditioned neural game-engine pipeline around
validated gameplay transitions, Stable Diffusion–based world modelling, and
reproducible evaluation. It is not a paper reproduction yet: this repository
contains no trained weights, published benchmark results, or claimed 20/50 FPS
measurements.

## What is implemented

- Atomic, checksummed NPZ gameplay shards with per-environment isolation,
  resume support, safe legacy migration, and corpus verification.
- Episode-disjoint training/validation splits, immutable held-out evaluation
  manifests, and provenance-linked latent-cache artifacts.
- Action-conditioned diffusion contracts: velocity prediction, observation CFG
  dropout, temporal action positions, context noise, EMA checkpoints, and
  revision-pinned Stable Diffusion 1.4 loading.
- DOOM contracts for scenario-derived actions, reward variables, 320×240 →
  320×256 padding, map/action-history PPO observations, and applied-action
  recording.
- Evaluation foundations: image metrics, valid paired FVD guards, action
  counterfactual grids, fixed-data sampling sweeps, and blinded human-study
  tooling.
- Offline tests, immutable-action CI, secret scanning, dependency-review
  support, dependency policy, and reproducibility documentation.

## What still requires execution

Real game/model dependencies, trained artifacts, GPU runs, pretrained I3D
weights, and participant studies are intentionally separate from the offline
test path. See [capabilities](docs/CAPABILITIES.md),
[paper fidelity](docs/PAPER_FIDELITY.md), and the
[implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md) before spending
compute.

## Install

```bash
git clone https://github.com/ReverseZoom2151/gamengen-v2.git
cd gamengen-v2

# Offline development and contract checks
pip install -e '.[dev]'
python -m pytest -q

# Add runtime dependencies only for the path being exercised
pip install -e '.[doom,dino,metrics]'
```

Python 3.10–3.12 is supported. Configure a matching CPU/CUDA PyTorch build
independently before GPU training. Read the [dependency policy](docs/DEPENDENCY_POLICY.md)
for supported-version and platform guidance.

## Core workflow

```text
game environment
  → canonical transitions
  → validated recording shards
  → episode-held-out datasets / latent caches
  → action-conditioned diffusion training
  → autoregressive evaluation and study artifacts
```

Useful commands:

```bash
# Validate runtime requirements before a real run
python -m src.preflight --config configs/tier2_doom_lite.yaml --mode train

# Development entry points
gamengen-dino --config configs/tier1_chrome_dino.yaml
gamengen-doom --config configs/tier2_doom_lite.yaml
gamengen-train --config configs/tier1_chrome_dino.yaml

# Reproducible data/evaluation preparation
gamengen-cache-latents --config configs/tier2_doom_lite.yaml \
  --shard data/recordings/shard_000000.npz --output data/latents/shard_000000.npz
gamengen-create-eval-manifest --data-dir data/recordings \
  --output artifacts/evaluation_manifest.json
gamengen-plan-ablations --config configs/tier3_full_doom.yaml \
  --kind context --output artifacts/context_plan.json
```

## Documentation

- [Implementation roadmap](docs/IMPLEMENTATION_ROADMAP.md)
- [Capability status](docs/CAPABILITIES.md)
- [Paper-fidelity status](docs/PAPER_FIDELITY.md)
- [Data format](docs/DATA_FORMAT.md) and [latent cache](docs/LATENT_CACHE.md)
- [ViZDoom integration](docs/VIZDOOM_INTEGRATION.md)
- [Human evaluation](docs/HUMAN_EVALUATION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md) and [security policy](docs/SECURITY.md)

## Contributing

Run `python -m pytest -q` before submitting changes. Keep claims tied to
versioned artifacts and update the capability/paper-fidelity documents when a
research path changes status.

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
