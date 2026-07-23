from __future__ import annotations

import pytest
import tynx
from tynx.nn import functional as F


def _weight() -> tynx.Tensor:
    return tynx.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_functional_embedding_supports_arbitrary_index_shape() -> None:
    indices = tynx.Tensor([[2, 0], [1, 2]], dtype="int64")

    output = F.embedding(indices, _weight())

    assert output.shape == (2, 2, 2)
    assert output.tolist() == [
        [[5.0, 6.0], [1.0, 2.0]],
        [[3.0, 4.0], [5.0, 6.0]],
    ]


def test_repeated_indices_accumulate_dense_weight_gradients() -> None:
    layer = tynx.nn.Embedding(3, 2)
    layer.weight.copy_(_weight())
    indices = tynx.Tensor([[0, 1, 0, 2]], dtype="int64")

    layer(indices).sum().backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.tolist() == [[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]]


def test_padding_idx_preserves_forward_value_but_blocks_gradient() -> None:
    layer = tynx.nn.Embedding(3, 2, padding_idx=0)
    layer.weight.copy_(_weight())
    indices = tynx.Tensor([0, 1, 0], dtype="int64")

    output = layer(indices)
    output.sum().backward()

    assert output.tolist() == [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]
    assert layer.weight.grad is not None
    assert layer.weight.grad.tolist() == [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]


def test_compile_replays_complete_embedding_training_step() -> None:
    layer = tynx.nn.Embedding(3, 2, padding_idx=0)
    layer.weight.copy_(_weight())
    optimizer = tynx.optim.SGD(layer.parameters(), lr=0.1)
    calls = 0

    @tynx.compile(fullgraph=True)
    def train_step(indices: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = layer(indices).sum()
        loss.backward()
        optimizer.step()
        return loss

    indices = tynx.Tensor([0, 2, 2, 1], dtype="int64")
    first = train_step(indices)
    second = train_step(indices)

    assert second.item() < first.item()
    assert layer.weight.tolist()[0] == [1.0, 2.0]
    assert calls == 1
    assert train_step.compile_count == 1
    assert train_step.fallback_count == 0
    assert train_step.replay_count == 1

    with pytest.raises(IndexError, match=r"embedding index 3.*size 3"):
        train_step(tynx.Tensor([0, 3, 2, 1], dtype="int64"))


def test_embedding_initialization_uses_tynx_seeded_rng() -> None:
    tynx.manual_seed(31)
    first = tynx.nn.Embedding(4, 3).weight.tolist()
    tynx.manual_seed(31)
    second = tynx.nn.Embedding(4, 3).weight.tolist()

    assert first == second


def test_embedding_rejects_sparse_and_other_unsupported_modes() -> None:
    with pytest.raises(NotImplementedError, match="sparse"):
        tynx.nn.Embedding(3, 2, sparse=True)
    with pytest.raises(NotImplementedError, match="max_norm"):
        F.embedding(tynx.arange(2), _weight(), max_norm=1.0)
    with pytest.raises(ValueError, match="padding_idx"):
        tynx.nn.Embedding(3, 2, padding_idx=3)


@pytest.mark.parametrize("index", [-3, 99])
def test_embedding_rejects_out_of_range_indices(index: int) -> None:
    layer = tynx.nn.Embedding(5, 3)
    indices = tynx.Tensor([0, index], dtype="int64")

    with pytest.raises(IndexError, match=rf"embedding index {index}.*size 5"):
        layer(indices)
