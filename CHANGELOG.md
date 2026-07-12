# Changelog

This project is a research implementation. Version labels describe repository
maintenance, not paper reproduction or benchmark results.

## [0.2.0] - Unreleased

### Foundation and data

- Replaced unsafe default recording flow with versioned atomic NPZ shards,
  checksums, explicit legacy migration, and a corpus verifier.
- Added isolated environment episode buffers, interruption recovery, persistent
  episode-disjoint validation splits, provenance metadata, and bounded shard caching.
- Added semantic configuration checks, public `gamengen` package namespace, and
  clean-install CI validation.

### Diffusion and training

- Added velocity-target fixtures, observation CFG dropout, temporal action
  positions, centralized current-action conditioning, and checkpoint validation.
- Implemented configured optimizer/scheduler/accumulation handling, atomic
  checkpoints, RNG/scaler/scheduler resume state, provenance manifests, and
  optional experiment tracking.
- Decoder artifacts now record/reuse split and run provenance.

### Environment and evaluation

- Added ViZDoom frame-padding/reward/action contracts and seeded multi-scenario
  selection.
- Corrected batch metric aggregation, provenance-linked evaluation reports,
  FVD guardrails, and blinded human-study protocol handling.

### Safety and status

- Quarantined the invalid one-step distillation prototype.
- Added data, reproducibility, security, capability, and paper-fidelity docs.

## Historical note

Earlier repository material described a complete v1.0.0 implementation. Those
claims are superseded by the current README, capability matrix, implementation
roadmap, and paper-fidelity report.
