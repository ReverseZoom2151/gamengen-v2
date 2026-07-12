# Optional dependencies

`pyproject.toml` is the authoritative dependency map. The offline test suite
uses only its lightweight dependencies and does not download models or launch
games.

| Capability | Install | Status |
|---|---|---|
| ViZDoom environments | `pip install -e '.[doom]'` | Required for Tier 2/3 runtime only |
| Chrome Dino browser environment | `pip install -e '.[dino]'` | Required for real Dino runtime only |
| LPIPS, SSIM, SciPy FVD math | `pip install -e '.[metrics]'` | Required only when computing those metrics |
| W&B | `pip install -e '.[tracking]'` | Configuration path is not yet fully integrated |
| Tests/build/Ruff | `pip install -e '.[dev]'` | Development workflow |

## FVD

The project deliberately has no simplified/random I3D fallback. A valid FVD
calculation requires:

1. separately recorded real and autoregressively generated trajectories;
2. an installed `pytorch_i3d` implementation; and
3. an explicit trusted pretrained I3D weights file.

Without all three, FVD is unavailable and must not be used as a paper-quality
or relative-quality claim.

## Preflight

Run offline contract tests first:

```bash
python -m pytest -q
```

For a recorded corpus, run the dependency-light verifier before model training:

```bash
python -m src.utils.verify_recordings recordings
```

Use only trusted legacy pickle input with the explicit migration command
documented in [DATA_FORMAT.md](DATA_FORMAT.md).
