"""Optional experiment-tracking integration with a safe no-op default."""

from typing import Any, Mapping


class NullTracker:
    def log(self, values: Mapping[str, Any], step: int) -> None:
        del values, step

    def finish(self) -> None:
        return None


class WandbTracker:
    def __init__(self, project: str, name: str, config: Mapping[str, Any]):
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("logging.use_wandb requires the optional tracking dependency") from error
        self.run = wandb.init(project=project, name=name, config=dict(config))

    def log(self, values: Mapping[str, Any], step: int) -> None:
        self.run.log(dict(values), step=step)

    def finish(self) -> None:
        self.run.finish()


def create_tracker(config: Mapping[str, Any]) -> NullTracker | WandbTracker:
    logging = config.get("logging", {})
    if logging.get("use_wandb", False):
        return WandbTracker(logging["wandb_project"], config["experiment_name"], config)
    return NullTracker()
