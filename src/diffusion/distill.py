"""Quarantined one-step-distillation entry point.

The previous prototype did not implement the paper's three-network objective:
it used the generator as its fake-score model and returned clean target latents
as a teacher result. It was removed rather than left callable as research code.
"""

import argparse


def distill_model(config: dict) -> None:
    """Refuse unvalidated distillation until a verified objective is implemented."""
    del config
    raise NotImplementedError(
        "one-step distillation is not validated; do not use this command for "
        "training or performance claims"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill GameNGen to one-step sampling")
    parser.add_argument("--config", default="configs/tier3_full_doom.yaml")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--steps", type=int)
    parser.parse_args()
    distill_model({})


if __name__ == "__main__":
    main()
