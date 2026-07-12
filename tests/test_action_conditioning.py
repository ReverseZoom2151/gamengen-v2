import torch

from src.diffusion.conditioning import ActionEmbedding, condition_current_action


def test_action_positions_make_permuted_histories_distinct():
    embedding = ActionEmbedding(num_actions=3, embedding_dim=4, max_length=4)
    with torch.no_grad():
        embedding.embedding.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        embedding.position.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    first = embedding(torch.tensor([[0, 1]]))
    second = embedding(torch.tensor([[1, 0]]))
    assert not torch.equal(first[:, 0], second[:, 0])
    # The set of action/position tokens differs, not only their token order.
    assert not torch.equal(first[:, 0] + 2 * first[:, 1], second[:, 0] + 2 * second[:, 1])


def test_legacy_action_embedding_state_can_initialize_new_positions():
    embedding = ActionEmbedding(num_actions=3, embedding_dim=4, max_length=4)
    result = embedding.load_state_dict({"embedding.weight": torch.zeros(3, 4)}, strict=False)
    assert result.missing_keys == ["position.weight"]


def test_current_action_replaces_only_final_context_token():
    original = torch.tensor([[1, 2, 0], [0, 1, 2]])
    conditioned = condition_current_action(original, 7)
    assert conditioned.tolist() == [[1, 2, 7], [0, 1, 7]]
    assert original.tolist() == [[1, 2, 0], [0, 1, 2]]
