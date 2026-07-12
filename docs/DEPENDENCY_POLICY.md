# Dependency policy

`pyproject.toml` is the only authoritative dependency declaration. Lower bounds
are reviewed against current PyPI releases and raised together; lockfiles or
constraints should be generated per supported OS/CUDA target before a runtime
reproduction campaign.

The current project supports Python 3.10–3.12. Where an upstream newest release
has dropped Python 3.10, the declared minimum is the newest release still
supporting the full project matrix: NumPy 2.2.6, scikit-image 0.25.2, and SciPy
1.15.3. All other direct dependencies use their current verified PyPI release
as of this update.

PyTorch/torchvision must be installed from the matching CPU or CUDA index for
the target platform; do not assume the generic PyPI wheel has the desired CUDA
runtime.
