from src.utils.tracking import NullTracker, create_tracker


def test_default_tracker_is_noop():
    tracker = create_tracker({"logging": {"use_wandb": False}})
    assert isinstance(tracker, NullTracker)
    tracker.log({"loss": 1.0}, 1)
    assert tracker.finish() is None
