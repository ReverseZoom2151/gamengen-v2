# Latent cache

Use `gamengen-cache-latents` only on validated NPZ recording shards. The command
loads the revision pinned in the selected configuration and stores one latent per
source frame, along with the original transition arrays and source-shard SHA-256.

```bash
gamengen-cache-latents --config configs/tier2_doom_lite.yaml \
  --shard data/recordings/shard_000000.npz \
  --output data/latents/shard_000000.npz
```

A cache is invalid when its source-shard checksum differs. Regenerate it after
changing recordings, preprocessing, model revision, or VAE encoding behavior.
The cache format is safe NPZ only; it never loads pickle artifacts.
