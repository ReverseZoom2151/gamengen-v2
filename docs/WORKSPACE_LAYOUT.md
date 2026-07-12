# Workspace layout

GameNGen keeps source, reproducible inputs, generated artifacts, and local
reference material deliberately separate. This makes a fresh clone usable for
offline development without accidentally committing model weights, recordings,
or copied upstream repositories.

```text
assets/branding/       Repository-owned visual assets used by project docs
configs/               Versioned tier and experiment configuration files
data/recordings/       Local gameplay recordings (empty placeholder tracked)
docs/                  Technical, operational, and research-status documents
paper/                 The reference paper used for fidelity checks
scripts/               Standalone maintenance and analysis utilities
src/                   Installable Python implementation
tests/                 Offline unit and contract tests
repos/                 Ignored local reference repositories; never a dependency
artifacts/             Ignored generated manifests, plans, and evaluation output
checkpoints/, logs/    Ignored runtime training output
```

## Conventions

- Keep repository-level files limited to project metadata, contribution and
  security policy, and the main README.
- Put new static project assets under `assets/`, grouped by purpose.
- Put durable documentation in `docs/`; link user-facing material from the
  README.
- Treat `data/recordings/` as an input/output boundary. Recordings and latent
  caches remain local because they can be large or derive from licensed runtime
  assets.
- Treat `repos/` as read-only local study material. It is intentionally ignored
  and must not be imported by the package or required in CI.
- Put one-off generated run output under ignored `artifacts/`, `logs/`, or
  `checkpoints/`, not at the repository root.

The parent Downloads directory is outside this repository’s ownership. This
project does not rename, move, or delete its unrelated files or neighboring
repositories.
