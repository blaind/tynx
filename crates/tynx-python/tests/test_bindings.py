"""Tests for the native Python bindings."""

import math
from collections.abc import Callable
from pathlib import Path

import pytest
import tynx


def test_module_metadata() -> None:
    assert tynx.__version__
    assert callable(tynx.Session)
    assert callable(tynx.Tensor)
    assert callable(tynx.Parameter)


def test_tensor_metadata_and_host_conversion() -> None:
    tensor = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert tensor.shape == (2, 2)
    assert tensor.ndim == 2
    assert tensor.numel == 4
    assert tensor.dtype == "float32"
    assert tensor.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert tynx.Tensor(3.5).shape == (1,)
    assert tynx.Tensor([3.5]).item() == pytest.approx(3.5)


def test_tensor_integer_and_boolean_storage() -> None:
    integers = tynx.Tensor([[1, -2], [3, 4]], dtype="int64")
    booleans = tynx.Tensor([[True, False], [False, True]], dtype="bool")

    assert integers.dtype == "int64"
    assert integers.shape == (2, 2)
    assert integers.tolist() == [[1, -2], [3, 4]]
    assert integers.grad is None
    assert not integers.requires_grad
    assert not integers.is_leaf
    assert isinstance(tynx.Tensor([7], dtype="int64").item(), int)

    assert booleans.dtype == "bool"
    assert booleans.shape == (2, 2)
    assert booleans.tolist() == [[True, False], [False, True]]
    assert isinstance(tynx.Tensor([True], dtype="bool").item(), bool)
    assert "dtype=int64" in repr(integers)


def test_tensor_typed_storage_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="unsupported Tensor dtype"):
        tynx.Tensor([1.0], dtype="float64")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="compatible scalar"):
        tynx.Tensor([1.5], dtype="int64")
    with pytest.raises(TypeError, match="compatible scalar"):
        tynx.Tensor([1], dtype="bool")
    with pytest.raises(TypeError, match="requires a float32 Tensor"):
        tynx.Tensor([1], dtype="int64", requires_grad=True)
    with pytest.raises(TypeError, match="requires a float32 Tensor"):
        tynx.Tensor([1], dtype="int64").relu()


def test_tensor_float_comparisons_return_detached_boolean_masks() -> None:
    values = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bounds = tynx.Tensor([[2.0], [4.0]])

    assert (values == bounds).tolist() == [[False, True], [False, True]]
    assert (values != bounds).tolist() == [[True, False], [True, False]]
    assert (values < bounds).tolist() == [[True, False], [True, False]]
    assert (values <= 2).tolist() == [[True, True], [False, False]]
    assert (values > 2).tolist() == [[False, False], [True, True]]
    assert (2 < values).tolist() == [[False, False], [True, True]]  # noqa: SIM300
    result = values >= bounds
    assert result.tolist() == [[False, True], [False, True]]
    assert result.dtype == "bool"
    assert not result.requires_grad


def test_tensor_integer_and_boolean_comparisons() -> None:
    values = tynx.Tensor([[1, 2], [3, 4]], dtype="int64")

    assert (values == 3).tolist() == [[False, False], [True, False]]
    assert (values != 3).tolist() == [[True, True], [False, True]]
    assert (values < 3).tolist() == [[True, True], [False, False]]
    assert (values <= 3).tolist() == [[True, True], [True, False]]
    assert (values > 3).tolist() == [[False, False], [False, True]]
    assert (values >= 3).tolist() == [[False, False], [True, True]]

    mask = tynx.Tensor([True, False], dtype="bool")
    assert (mask == True).tolist() == [True, False]  # noqa: E712
    assert (mask != False).tolist() == [True, False]  # noqa: E712


def test_tensor_boolean_mask_algebra_broadcasts() -> None:
    left = tynx.Tensor([[True], [False]], dtype="bool")
    right = tynx.Tensor([[True, False]], dtype="bool")

    assert (left & right).tolist() == [[True, False], [False, False]]
    assert (left | right).tolist() == [[True, True], [True, False]]
    assert (left ^ right).tolist() == [[False, True], [True, False]]
    assert (~left).tolist() == [[False], [True]]


def test_tensor_comparisons_and_masks_reject_invalid_types() -> None:
    floats = tynx.Tensor([1.0])
    integers = tynx.Tensor([1], dtype="int64")
    booleans = tynx.Tensor([True], dtype="bool")

    with pytest.raises(TypeError, match="matching dtypes"):
        _ = floats == integers
    with pytest.raises(TypeError, match="integer scalar"):
        _ = integers > 1.5
    with pytest.raises(TypeError, match="not defined for bool"):
        _ = booleans < True
    with pytest.raises(TypeError, match="requires bool Tensors"):
        _ = booleans & floats
    with pytest.raises(TypeError, match="requires a bool Tensor"):
        _ = ~floats


def test_tensor_where_selects_broadcast_tensor_and_scalar_branches() -> None:
    condition = tynx.Tensor([[True], [False]], dtype="bool")
    values = tynx.Tensor([[1.0, 2.0]])
    other = tynx.Tensor([[10.0], [20.0]])

    expected = [[1.0, 2.0], [20.0, 20.0]]
    assert tynx.where(condition, values, other).tolist() == expected
    assert values.where(condition, other).tolist() == expected
    assert tynx.where(condition, values, -1).tolist() == [[1.0, 2.0], [-1.0, -1.0]]
    assert tynx.where(condition, -1, other).tolist() == [[-1.0], [20.0]]

    integers = tynx.Tensor([[1, 2]], dtype="int64")
    assert integers.where(condition, 0).tolist() == [[1, 2], [0, 0]]
    booleans = tynx.Tensor([[True, False]], dtype="bool")
    assert booleans.where(condition, False).tolist() == [
        [True, False],
        [False, False],
    ]


def test_tensor_where_routes_gradients_to_selected_float_branches() -> None:
    condition = tynx.Tensor([[True], [False]], dtype="bool")
    selected = tynx.Tensor([[1.0, 2.0]], requires_grad=True)
    otherwise = tynx.Tensor([[10.0], [20.0]], requires_grad=True)

    tynx.where(condition, selected, otherwise).sum().backward()

    assert selected.grad is not None
    assert selected.grad.tolist() == [[1.0, 1.0]]
    assert otherwise.grad is not None
    assert otherwise.grad.tolist() == [[0.0], [2.0]]


def test_tensor_where_rejects_invalid_conditions_and_branches() -> None:
    condition = tynx.Tensor([True], dtype="bool")
    values = tynx.Tensor([1.0])

    with pytest.raises(TypeError, match="condition must be a bool Tensor"):
        tynx.where(values, values, 0)
    with pytest.raises(TypeError, match="matching dtypes"):
        tynx.where(condition, values, tynx.Tensor([1], dtype="int64"))
    with pytest.raises(TypeError, match="real scalar"):
        tynx.where(condition, values, "invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="at least one Tensor branch"):
        tynx.where(condition, 1, 0)


def test_tensor_gather_supports_typed_values_and_negative_dimensions() -> None:
    indices = tynx.Tensor([[2, 1], [0, 0]], dtype="int64")
    floats = tynx.Tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    integers = tynx.Tensor([[10, 11, 12], [20, 21, 22]], dtype="int64")
    booleans = tynx.Tensor([[True, False, False], [False, True, True]], dtype="bool")

    assert floats.gather(-1, indices).tolist() == [[12.0, 11.0], [20.0, 20.0]]
    assert integers.gather(1, indices).tolist() == [[12, 11], [20, 20]]
    assert booleans.gather(1, indices).tolist() == [[False, False], [False, False]]

    prefix_indices = tynx.Tensor([[2, 0]], dtype="int64")
    assert floats.gather(1, prefix_indices).tolist() == [[12.0, 10.0]]


def test_tensor_gather_backward_accumulates_repeated_indices() -> None:
    values = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    indices = tynx.Tensor([[1, 1, 2], [0, 2, 2]], dtype="int64")

    values.gather(1, indices).sum().backward()

    assert values.grad is not None
    assert values.grad.tolist() == [[0.0, 2.0, 1.0], [1.0, 0.0, 2.0]]


def test_tensor_gather_rejects_invalid_indices_and_shapes() -> None:
    values = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(TypeError, match="must be an int64 Tensor"):
        values.gather(1, tynx.Tensor([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="rank 1 must match input rank 2"):
        values.gather(1, tynx.Tensor([0, 1], dtype="int64"))
    with pytest.raises(ValueError, match="exceeds input size"):
        values.gather(1, tynx.Tensor([[0], [1], [2]], dtype="int64"))
    with pytest.raises(ValueError, match="dimension 2 is out of range"):
        values.gather(2, tynx.Tensor([[0], [0]], dtype="int64"))


def test_tensor_value_min_max_reductions_follow_shape_policy() -> None:
    values = tynx.Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

    assert values.max().shape == (1,)
    assert values.max().tolist() == [6.0]
    assert values.min().tolist() == [1.0]
    assert values.max(dim=0).tolist() == [4.0, 5.0, 6.0]
    assert values.min(dim=-1, keepdim=True).tolist() == [[1.0], [2.0]]
    assert values.max(dim=(0, 1), keepdim=True).shape == (1, 1)

    integers = tynx.Tensor([[1, 4], [3, 2]], dtype="int64")
    booleans = tynx.Tensor([[True, False], [False, False]], dtype="bool")
    assert integers.min(dim=0).tolist() == [1, 2]
    assert booleans.max(dim=0).tolist() == [True, False]
    assert booleans.min().tolist() == [False]


def test_tensor_value_min_max_reductions_are_differentiable() -> None:
    values = tynx.Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True)

    (values.max(dim=1) + values.min(dim=1)).sum().backward()

    assert values.grad is not None
    assert values.grad.tolist() == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]


def test_tensor_elementwise_minimum_maximum_broadcast_and_gradients() -> None:
    left = tynx.Tensor([[-2.0], [3.0]], requires_grad=True)
    right = tynx.Tensor([[1.0, 2.0]], requires_grad=True)

    maximum = tynx.maximum(left, right)
    assert maximum.tolist() == [[1.0, 2.0], [3.0, 3.0]]
    assert left.minimum(0).tolist() == [[-2.0], [0.0]]
    assert tynx.minimum(left, right).tolist() == [[-2.0, -2.0], [1.0, 2.0]]

    maximum.sum().backward()
    assert left.grad is not None
    assert left.grad.tolist() == [[0.0], [2.0]]
    assert right.grad is not None
    assert right.grad.tolist() == [[1.0, 1.0]]


def test_tensor_elementwise_minimum_maximum_support_typed_values_and_errors() -> None:
    integers = tynx.Tensor([[1], [4]], dtype="int64")
    masks = tynx.Tensor([[True], [False]], dtype="bool")

    assert integers.maximum(3).tolist() == [[3], [4]]
    assert masks.maximum(False).tolist() == [[True], [False]]
    assert masks.minimum(True).tolist() == [[True], [False]]

    with pytest.raises(TypeError, match="matching Tensor dtypes"):
        integers.maximum(tynx.Tensor([1.0]))
    with pytest.raises(TypeError, match="integer scalar"):
        integers.minimum(1.5)


def test_sgd_updates_parameters_without_clearing_gradients() -> None:
    parameter = tynx.Parameter([2.0])
    optimizer = tynx.optim.SGD([parameter, parameter], lr=0.1)

    (parameter * 3).backward()
    optimizer.step()

    assert parameter.tolist() == pytest.approx([1.7])
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([3.0])
    assert parameter.name is None
    assert optimizer.parameter_count == 1
    optimizer.zero_grad()
    assert parameter.grad is None


def test_sgd_configuration_momentum_and_mutable_learning_rate() -> None:
    parameter = tynx.Parameter([2.0])
    optimizer = tynx.optim.SGD([parameter], lr=0.1, momentum=0.9)

    (parameter * 2).backward()
    optimizer.step()
    optimizer.zero_grad()
    (parameter * 2).backward()
    optimizer.lr = 0.2
    optimizer.step()

    assert optimizer.lr == pytest.approx(0.2)
    assert optimizer.learning_rate == pytest.approx(0.2)
    assert optimizer.state_size == 1
    assert parameter.tolist() == pytest.approx([1.04])
    assert "momentum=0.9" in repr(optimizer)


def test_sgd_rejects_invalid_parameters_and_configuration() -> None:
    with pytest.raises(ValueError, match="at least one Parameter"):
        tynx.optim.SGD([], lr=0.1)
    with pytest.raises(TypeError, match="only Parameter objects"):
        tynx.optim.SGD([tynx.Tensor([1.0])], lr=0.1)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="learning rate"):
        tynx.optim.SGD([tynx.Parameter([1.0])], lr=-0.1)
    with pytest.raises(ValueError, match="Nesterov"):
        tynx.optim.SGD([tynx.Parameter([1.0])], lr=0.1, nesterov=True)


def test_sgd_trains_an_authored_python_linear_model() -> None:
    weight = tynx.Parameter([0.0], name="weight")
    bias = tynx.Parameter([0.0], name="bias")
    optimizer = tynx.optim.SGD([weight, bias], lr=0.1)
    inputs = tynx.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    targets = tynx.Tensor([-3.0, -1.0, 1.0, 3.0, 5.0])

    for _ in range(100):
        optimizer.zero_grad()
        prediction = inputs * weight + bias
        error = prediction - targets
        (error * error).mean().backward()
        optimizer.step()

    assert weight.item() == pytest.approx(2.0, abs=1e-4)
    assert bias.item() == pytest.approx(1.0, abs=1e-4)


def test_tensor_eager_operators() -> None:
    left = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])
    right = tynx.Tensor([[2.0, 0.5], [1.0, 2.0]])

    assert (left + right).tolist() == [[3.0, 2.5], [4.0, 6.0]]
    assert (left - right).tolist() == [[-1.0, 1.5], [2.0, 2.0]]
    assert (left * right).tolist() == [[2.0, 1.0], [3.0, 8.0]]
    assert (left / right).tolist() == [[0.5, 4.0], [3.0, 2.0]]
    assert (left @ right).tolist() == [[4.0, 4.5], [10.0, 9.5]]
    assert (-left).tolist() == [[-1.0, -2.0], [-3.0, -4.0]]


def test_tensor_scalar_and_reverse_operators() -> None:
    value = tynx.Tensor([2.0, 4.0])

    assert (value + 2).tolist() == pytest.approx([4.0, 6.0])
    assert (2 + value).tolist() == pytest.approx([4.0, 6.0])
    assert (value - 1.5).tolist() == pytest.approx([0.5, 2.5])
    assert (10 - value).tolist() == pytest.approx([8.0, 6.0])
    assert (value * 3).tolist() == pytest.approx([6.0, 12.0])
    assert (3 * value).tolist() == pytest.approx([6.0, 12.0])
    assert (value / 2).tolist() == pytest.approx([1.0, 2.0])
    assert (8 / value).tolist() == pytest.approx([4.0, 2.0])

    with pytest.raises(TypeError, match="Tensor or a real number"):
        value + "invalid"  # type: ignore[operator]


def test_tensor_reverse_scalar_operators_are_differentiable() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)

    (3 - value * 2 + 8 / value).backward()

    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([-4.0])


def test_tensor_unary_activations_and_math() -> None:
    signed = tynx.Tensor([-1.0, 0.0, 1.0])
    positive = tynx.Tensor([1.0, 4.0])

    assert signed.relu().tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert signed.sigmoid().tolist() == pytest.approx(
        [1.0 / (1.0 + math.e), 0.5, math.e / (1.0 + math.e)]
    )
    assert signed.tanh().tolist() == pytest.approx([-math.tanh(1.0), 0.0, math.tanh(1.0)])
    assert positive.exp().tolist() == pytest.approx([math.e, math.exp(4.0)])
    assert positive.log().tolist() == pytest.approx([0.0, math.log(4.0)])
    assert positive.sqrt().tolist() == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize(
    ("operation", "point"),
    [
        (lambda value: value.relu(), 2.0),
        (lambda value: value.sigmoid(), 0.5),
        (lambda value: value.tanh(), 0.5),
        (lambda value: value.exp(), 0.5),
        (lambda value: value.log(), 2.0),
        (lambda value: value.sqrt(), 4.0),
        (lambda value: value.gelu(), 0.5),
    ],
)
def test_tensor_unary_gradients_match_finite_differences(
    operation: Callable[[tynx.Tensor], tynx.Tensor], point: float
) -> None:
    value = tynx.Tensor([point], requires_grad=True)
    operation(value).backward()

    epsilon = 1e-3
    above = operation(tynx.Tensor([point + epsilon])).item()
    below = operation(tynx.Tensor([point - epsilon])).item()
    numerical = (above - below) / (2 * epsilon)

    assert value.grad is not None
    assert value.grad.item() == pytest.approx(numerical, rel=2e-3, abs=2e-3)


def test_tensor_softmax_log_softmax_gelu_and_clamp() -> None:
    logits = tynx.Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    probabilities = logits.softmax(-1).tolist()
    log_probabilities = logits.log_softmax(-1).exp().tolist()

    denominator = math.exp(1.0) + math.exp(2.0) + math.exp(3.0)
    expected = [math.exp(value) / denominator for value in (1.0, 2.0, 3.0)]
    assert probabilities[0] == pytest.approx(expected)
    assert probabilities[1] == pytest.approx([1.0 / 3.0] * 3)
    assert log_probabilities[0] == pytest.approx(expected)
    assert log_probabilities[1] == pytest.approx([1.0 / 3.0] * 3)

    assert tynx.Tensor([-1.0, 0.0, 1.0]).gelu().tolist() == pytest.approx(
        [-0.15865526, 0.0, 0.8413447], abs=1e-6
    )
    values = tynx.Tensor([-2.0, -0.5, 0.5, 2.0])
    assert values.clamp(-1.0, 1.0).tolist() == pytest.approx([-1.0, -0.5, 0.5, 1.0])
    assert values.clamp(min=0.0).tolist() == pytest.approx([0.0, 0.0, 0.5, 2.0])
    assert values.clip(max=0.0).tolist() == pytest.approx([-2.0, -0.5, 0.0, 0.0])


def test_tensor_softmax_and_clamp_gradients() -> None:
    logits = tynx.Tensor([0.0, 0.0], requires_grad=True)
    logits.softmax(0).backward(tynx.Tensor([1.0, 0.0]))
    assert logits.grad is not None
    assert logits.grad.tolist() == pytest.approx([0.25, -0.25])

    logits.zero_grad()
    logits.log_softmax(0).backward(tynx.Tensor([1.0, 0.0]))
    assert logits.grad is not None
    assert logits.grad.tolist() == pytest.approx([0.5, -0.5])

    values = tynx.Tensor([-2.0, 0.0, 2.0], requires_grad=True)
    values.clamp(-1.0, 1.0).sum().backward()
    assert values.grad is not None
    assert values.grad.tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_tensor_softmax_and_clamp_validate_arguments() -> None:
    value = tynx.Tensor([1.0, 2.0])

    with pytest.raises(ValueError, match="out of range"):
        value.softmax(1)
    with pytest.raises(TypeError, match="integers"):
        value.log_softmax(True)
    with pytest.raises(ValueError, match="at least one"):
        value.clamp()


def test_tensor_shape_operations() -> None:
    value = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert value.reshape(3, 2).shape == (3, 2)
    assert value.reshape((3, 2)).tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert value.reshape(-1, 2).shape == (3, 2)
    assert value.flatten().tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    volume = value.reshape(1, 2, 3)
    assert volume.flatten(1).shape == (1, 6)
    assert value.transpose(0, 1).tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert value.permute(-1, 0).shape == (3, 2)
    assert value.unsqueeze(0).shape == (1, 2, 3)
    assert value.unsqueeze(-1).shape == (2, 3, 1)
    assert value.unsqueeze(1).squeeze(1).shape == value.shape
    assert value.squeeze().shape == value.shape
    assert tynx.Tensor([[[1.0]]]).squeeze().shape == (1,)


def test_tensor_shape_operations_preserve_gradients() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    value.transpose(0, 1).reshape(4).sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_tensor_shape_operations_validate_arguments() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="at most one"):
        value.reshape(-1, -1)
    with pytest.raises(ValueError, match="cannot reshape"):
        value.reshape(3, 2)
    with pytest.raises(ValueError, match="appears more than once"):
        value.permute(0, 0)
    with pytest.raises(ValueError, match="cannot come after"):
        value.flatten(1, 0)
    with pytest.raises(ValueError, match="out of range"):
        value.unsqueeze(3)
    with pytest.raises(TypeError, match="integers"):
        value.transpose(True, 1)


def test_tensor_rejects_ragged_or_empty_data() -> None:
    with pytest.raises(ValueError, match="ragged"):
        tynx.Tensor([[1.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="empty"):
        tynx.Tensor([])


def test_tensor_item_requires_one_element() -> None:
    with pytest.raises(ValueError, match="one-element"):
        tynx.Tensor([1.0, 2.0]).item()


def test_tensor_backward_accumulates_leaf_gradients() -> None:
    value = tynx.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    loss = (value * value).mean()
    assert loss.shape == (1,)
    assert loss.requires_grad
    assert not loss.is_leaf
    assert loss.grad is None
    loss.backward()

    assert value.is_leaf
    assert value.requires_grad
    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([2.0 / 3.0, 4.0 / 3.0, 2.0])

    (value * value).mean().backward()
    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([4.0 / 3.0, 8.0 / 3.0, 4.0])

    value.zero_grad()
    assert value.grad is None


def test_tensor_reductions_follow_dim_and_keepdim_shapes() -> None:
    value = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert value.sum().shape == (1,)
    assert value.sum().item() == pytest.approx(21.0)
    assert value.sum(0).shape == (3,)
    assert value.sum(0).tolist() == pytest.approx([5.0, 7.0, 9.0])
    assert value.mean(-1).shape == (2,)
    assert value.mean(-1).tolist() == pytest.approx([2.0, 5.0])
    assert value.mean((0, -1)).shape == (1,)
    assert value.mean((0, -1)).item() == pytest.approx(3.5)
    assert value.sum((0, 1), keepdim=True).shape == (1, 1)
    assert value.sum(1, keepdim=True).shape == (2, 1)
    assert value.sum(1, keepdim=True).tolist() == [[6.0], [15.0]]
    assert value.mean(None, keepdim=True).shape == (1, 1)
    assert value.sum(()).item() == pytest.approx(21.0)


def test_tensor_reduction_gradients_survive_axis_squeezing() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    value.mean(0).sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_tensor_reductions_reject_invalid_dims() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="out of range"):
        value.sum(2)
    with pytest.raises(ValueError, match="more than once"):
        value.mean((0, -2))
    with pytest.raises(TypeError, match="tuple must contain only integers"):
        value.sum((0, "bad"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int, a tuple"):
        value.sum(True)


def test_tensor_detach_stops_gradient_tracking() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    detached = (value * value).detach()

    assert detached.shape == (1,)
    assert not detached.requires_grad
    assert not detached.is_leaf
    with pytest.raises(ValueError, match="autodiff graph"):
        detached.backward()
    assert value.grad is None


def test_tensor_backward_rejects_non_scalar_output() -> None:
    value = tynx.Tensor([1.0, 2.0], requires_grad=True)

    with pytest.raises(ValueError, match="one-element"):
        (value * value).backward()


def test_tensor_backward_accepts_explicit_gradient() -> None:
    value = tynx.Tensor([1.0, 2.0], requires_grad=True)
    output = value * value

    output.backward(tynx.Tensor([3.0, 4.0]))

    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([6.0, 16.0])

    with pytest.raises(ValueError, match="gradient shape"):
        (value * value).backward(tynx.Tensor([1.0]))


def test_no_grad_is_nested_and_restores_tracking() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    assert tynx.is_grad_enabled()

    with tynx.no_grad():
        assert not tynx.is_grad_enabled()
        first = value * value
        assert not first.requires_grad
        with tynx.no_grad():
            assert not tynx.is_grad_enabled()
            second = -value
            assert not second.requires_grad
        assert not tynx.is_grad_enabled()

    assert tynx.is_grad_enabled()
    assert (value * value).requires_grad


def test_parameter_is_a_named_trainable_tensor() -> None:
    parameter = tynx.Parameter([2.0], name="weight")

    assert isinstance(parameter, tynx.Tensor)
    assert parameter.name == "weight"
    assert parameter.requires_grad
    assert parameter.is_leaf
    assert parameter.tolist() == pytest.approx([2.0])


def test_parameter_accumulates_and_zeros_gradients() -> None:
    parameter = tynx.Parameter([2.0])

    (parameter * parameter).mean().backward()
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([4.0])

    # One target entry still carries the full derivative from both uses of the slot.
    (parameter + parameter).mean().backward()
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([6.0])

    parameter.zero_grad()
    assert parameter.grad is None

    (parameter * 3).backward()
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([3.0])


def test_missing_model_raises_os_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.onnx"

    with pytest.raises(OSError):
        tynx.Session(missing_model)


def test_invalid_model_raises_value_error(tmp_path: Path) -> None:
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"not an ONNX model")

    with pytest.raises(ValueError, match="failed to parse"):
        tynx.Session(invalid_model)
