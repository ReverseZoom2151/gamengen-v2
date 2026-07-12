# Security policy

Recording shards use NPZ with `allow_pickle=False` and are checksum-validated.
Legacy pickle recordings are supported only for migration and must be treated as
trusted local input. Never load untrusted pickle files.

PyTorch checkpoints execute deserialization logic when loaded with unrestricted
settings. Use artifacts from trusted sources only. Model downloads and game
environment runtimes are optional dependencies and should be pinned/audited by
the experiment owner.
