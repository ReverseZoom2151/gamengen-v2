# Recording data format

New recordings use atomic `shard_XXXXXX.npz` files and `metadata.json`.
Every shard contains a JSON manifest plus HWC uint8 RGB frames, int32 actions,
float32 rewards, and terminal/truncation flags. A canonical episode has exactly
one more frame than actions: action `t` takes frame `t` to frame `t + 1`.

`metadata.json` records schema version, monotonic next shard ID, total counts,
and SHA-256 checksums. New readers validate all of these before training.
Episode-level train/validation splits are deterministic and disjoint.

Legacy `batch_*.pkl` files are unsafe for untrusted input. Migrate only trusted
local datasets explicitly:

```bash
python -m src.utils.migration old_recordings new_recordings --trusted-legacy-input
```

Before training, verify a corpus without loading model or game dependencies:

```bash
python -m src.utils.verify_recordings recordings
```
