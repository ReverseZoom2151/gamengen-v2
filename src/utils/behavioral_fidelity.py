"""Offline, reproducible measurements of gameplay behavioral fidelity."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _series(trajectory: Mapping[str, object], name: str) -> np.ndarray:
    if name not in trajectory:
        raise ValueError(f"trajectory is missing {name!r}")
    values = np.asarray(trajectory[name], dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name!r} must be a finite one-dimensional series with at least two values")
    return values


def wasserstein_1d(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Deterministic empirical 1-Wasserstein distance without a SciPy dependency."""
    reference, candidate = np.sort(np.asarray(reference, dtype=np.float64)), np.sort(np.asarray(candidate, dtype=np.float64))
    if not len(reference) or not len(candidate):
        raise ValueError("Wasserstein samples must be non-empty")
    quantiles = np.linspace(0.0, 1.0, max(len(reference), len(candidate)))
    return float(np.mean(np.abs(np.quantile(reference, quantiles) - np.quantile(candidate, quantiles))))


def camera_motion(angles: np.ndarray) -> dict[str, np.ndarray]:
    """Return angular velocity, acceleration, and jerk for a camera-angle trace."""
    velocity = np.diff(angles)
    acceleration = np.diff(velocity)
    jerk = np.diff(acceleration)
    return {"velocity": velocity, "acceleration": acceleration, "jerk": jerk}


def occupancy_histogram(positions: np.ndarray, bins: int = 16) -> np.ndarray:
    """Create a normalized two-dimensional occupancy distribution."""
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError("positions must have shape (N, 2) and be non-empty")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    histogram, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=bins)
    return histogram / histogram.sum()


def behavioral_fidelity(reference: Mapping[str, object], candidate: Mapping[str, object], bins: int = 16) -> dict[str, float]:
    """Compare motion and map occupancy while keeping human references private."""
    reference_angles, candidate_angles = _series(reference, "camera_angle"), _series(candidate, "camera_angle")
    reference_motion, candidate_motion = camera_motion(reference_angles), camera_motion(candidate_angles)
    result = {
        f"camera_{kind}_wasserstein": wasserstein_1d(reference_motion[kind], candidate_motion[kind])
        for kind in reference_motion
    }
    reference_map = occupancy_histogram(np.asarray(reference["position"]), bins=bins)
    candidate_map = occupancy_histogram(np.asarray(candidate["position"]), bins=bins)
    result["occupancy_l1"] = float(np.abs(reference_map - candidate_map).sum())
    return result
