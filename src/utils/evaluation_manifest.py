"""Immutable episode manifests for trustworthy held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

from src.utils.data_recorder import load_npz_shard, shard_checksum


def create_evaluation_manifest(data_dir: str | Path, destination: str | Path, limit: int | None = None) -> Path:
    """Select complete recorded episodes and bind them to exact source checksums."""
    data_dir, destination = Path(data_dir), Path(destination)
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    selected, sources = [], {}
    for shard in sorted(data_dir.glob("shard_*.npz")):
        checksum = shard_checksum(shard)
        episodes = load_npz_shard(shard, metadata.get("shard_checksums", {}).get(shard.name))
        sources[shard.name] = checksum
        for episode in episodes:
            selected.append({
                "shard": shard.name,
                "episode_id": episode["episode_id"],
                "env_id": episode["env_id"],
                "length": episode["length"],
                "scenario": episode.get("metadata", {}).get("scenario", "unknown"),
            })
            if limit is not None and len(selected) >= limit:
                break
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError("no validated recording episodes are available for evaluation")
    payload = {"format_version": 1, "episodes": selected, "source_shards": sources}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def validate_evaluation_manifest(path: str | Path, data_dir: str | Path) -> dict:
    """Reject a manifest when any referenced recording shard has changed."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not payload.get("episodes"):
        raise ValueError("invalid evaluation manifest")
    for name, expected in payload.get("source_shards", {}).items():
        if shard_checksum(Path(data_dir) / name) != expected:
            raise ValueError(f"evaluation source checksum mismatch: {name}")
    return payload


def create_human_benchmark_manifest(data_dir: str | Path, destination: str | Path, limit: int | None = None) -> Path:
    """Create an immutable held-out manifest from recordings explicitly tagged as human."""
    data_dir, destination = Path(data_dir), Path(destination)
    selected, sources = [], {}
    for shard in sorted(data_dir.glob("shard_*.npz")):
        checksum = shard_checksum(shard)
        for episode in load_npz_shard(shard):
            metadata = episode.get("metadata", {})
            if metadata.get("demonstration_source") != "human":
                continue
            selected.append({"shard": shard.name, "episode_id": episode["episode_id"], "length": episode["length"], "scenario": metadata.get("scenario", "unknown"), "participant_id": metadata.get("participant_id", "unknown")})
            sources[shard.name] = checksum
            if limit is not None and len(selected) >= limit:
                break
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        raise ValueError("no human-tagged recording episodes are available for evaluation")
    payload = {"format_version": 1, "kind": "human_demonstration_benchmark", "episodes": selected, "source_shards": sources}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a checksum-bound GameNGen evaluation manifest")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    path = create_evaluation_manifest(args.data_dir, args.output, limit=args.limit)
    print(f"Wrote immutable evaluation manifest: {path}")
