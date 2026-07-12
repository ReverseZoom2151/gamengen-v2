import numpy as np

from src.agent.dqn_agent import ReplayBuffer


def test_replay_buffer_state_round_trip_preserves_transitions():
    buffer = ReplayBuffer(3)
    buffer.push(np.array([1]), 2, 3.0, np.array([4]), False)
    restored = ReplayBuffer(1)
    restored.load_state_dict(buffer.state_dict())
    assert len(restored) == 1
    states, actions, rewards, next_states, dones = restored.sample(1)
    assert states.tolist() == [[1]]
    assert actions.tolist() == [2]
    assert rewards.tolist() == [3.0]
    assert next_states.tolist() == [[4]]
    assert dones.tolist() == [0.0]
