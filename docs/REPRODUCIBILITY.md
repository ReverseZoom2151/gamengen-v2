# Reproducibility and artifacts

Every diffusion checkpoint records its optimizer, scheduler, AMP scaler, Python/
NumPy/Torch RNG state, resolved configuration, and a run manifest. The manifest
contains a configuration hash, code revision when Git is available, platform and
Torch versions, CUDA availability, and the SHA-256 of recording metadata.

To resume, retain `latest_checkpoint.pt` and the exact recording directory. A
resumed job restores model, optimizer, scheduler, scaler, and RNG state. It does
not yet restore worker scheduling or an external game emulator state; therefore
bit-identical continuation is not claimed for multi-worker or environment runs.

Only report a benchmark with its immutable checkpoint, config, recording
metadata, evaluator version, and generated outputs. Paper values are reference
values unless a local artifact set proves otherwise.

Use `save_evaluation_report` to write metric results atomically with their run
manifest; do not rely on terminal output as an evaluation record.
