"""Tests for the native Python bindings."""

import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tynx


def test_module_metadata() -> None:
    assert tynx.__version__


def test_public_modules_do_not_expose_imported_typing_helpers() -> None:
    assert not {
        "Literal",
        "Mapping",
        "Optional",
        "PathLike",
        "Union",
        "overload",
    }.intersection(dir(tynx))
    assert not {"Callable", "Generic", "Mapping", "TypeVar", "Union", "cast"}.intersection(
        dir(tynx.optim)
    )
    assert not {"Callable", "Generic", "Optional", "TypeVar", "Union", "overload"}.intersection(
        dir(tynx.compiler)
    )
    assert not {"Optional", "Union"}.intersection(dir(tynx.distributions))
    assert callable(tynx.Session)
    assert callable(tynx.Tensor)
    assert callable(tynx.Parameter)
    device = tynx.get_default_device()
    assert isinstance(device, tynx.Device)
    assert str(device)
    tynx.synchronize()
    tynx.synchronize(device)


def test_manual_seed_replays_authored_module_initialization() -> None:
    random.seed(17)
    expected_application_draw = random.random()
    random.seed(17)

    tynx.manual_seed(5)
    first_linear = tynx.nn.Linear(4, 4)
    first_conv = tynx.nn.Conv2d(2, 3, 3)

    tynx.manual_seed(5)
    second_linear = tynx.nn.Linear(4, 4)
    second_conv = tynx.nn.Conv2d(2, 3, 3)

    assert first_linear.weight.tolist() == second_linear.weight.tolist()
    assert first_linear.bias is not None and second_linear.bias is not None
    assert first_linear.bias.tolist() == second_linear.bias.tolist()
    assert first_conv.weight.tolist() == second_conv.weight.tolist()
    assert random.random() == expected_application_draw


def test_tensor_metadata_and_host_conversion() -> None:
    tensor = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert tensor.shape == (2, 2)
    assert tensor.ndim == 2
    assert tensor.numel == 4
    assert tensor.dtype == "float32"
    assert tensor.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert tynx.Tensor(3.5).shape == (1,)
    assert tynx.Tensor([3.5]).item() == pytest.approx(3.5)
    assert isinstance(tensor.device, tynx.Device)
    assert str(tensor.device)
    assert tensor.device == tynx.get_default_device()
    assert len(tensor) == 2
    assert tynx.Tensor(tensor).tolist() == tensor.tolist()
    assert tynx.Tensor(range(3)).tolist() == [0.0, 1.0, 2.0]

    integers = tynx.Tensor([1, 2], dtype="int64")
    assert tynx.Tensor(integers).dtype == "int64"


def test_tensor_python_numeric_protocols() -> None:
    values = tynx.Tensor([-2.0, 3.0])

    assert abs(values).tolist() == [2.0, 3.0]
    assert (values**2).tolist() == pytest.approx([4.0, 9.0])
    assert (2 ** tynx.Tensor([1.0, 3.0])).tolist() == pytest.approx([2.0, 8.0])


def test_tensor_matmul_supports_vector_cases() -> None:
    vector = tynx.Tensor([1.0, 2.0])
    matrix = tynx.Tensor([[3.0, 4.0], [5.0, 6.0]])

    assert (matrix @ vector).tolist() == pytest.approx([11.0, 17.0])
    assert (vector @ matrix).tolist() == pytest.approx([13.0, 16.0])
    assert (vector @ vector).tolist() == pytest.approx([5.0])


def test_tensor_shape_mismatches_raise_value_error_before_dispatch() -> None:
    with pytest.raises(
        ValueError,
        match=r"cannot broadcast shapes \[2, 3\] and \[4, 5\]",
    ):
        tynx.Tensor([[1.0] * 3] * 2) + tynx.Tensor([[1.0] * 5] * 4)

    with pytest.raises(ValueError, match="matmul inner dimensions must match"):
        tynx.Tensor([[1.0, 2.0]]) @ tynx.Tensor([[1.0, 2.0]])


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("float32", [[1.0, 2.0], [3.0, 4.0]]),
        ("int64", [[1, 2], [3, 4]]),
        ("bool", [[True, False], [False, True]]),
    ],
)
def test_tensor_numpy_round_trip(
    dtype: Literal["float32", "int64", "bool"], values: list[list[object]]
) -> None:
    array = np.asarray(values, dtype=dtype).T

    tensor = tynx.Tensor(array)
    output = tensor.numpy()

    assert tensor.shape == array.shape
    assert tensor.dtype == dtype
    assert output.dtype == array.dtype
    np.testing.assert_array_equal(output, array)


def test_tensor_numpy_rejects_unsupported_or_mismatched_dtype() -> None:
    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        tynx.Tensor(np.asarray([1.0], dtype=np.float64))
    with pytest.raises(TypeError, match="must match requested Tensor dtype float32"):
        tynx.Tensor(np.asarray([1], dtype=np.int64), dtype="float32")


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


def test_tensor_value_min_max_reductions_propagate_nan() -> None:
    values = tynx.Tensor([1.0, float("nan"), 3.0])

    assert math.isnan(values.min().item())
    assert math.isnan(values.max().item())


def test_tensor_value_min_max_reductions_are_differentiable() -> None:
    values = tynx.Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True)

    (values.max(dim=1) + values.min(dim=1)).sum().backward()

    assert values.grad is not None
    assert values.grad.tolist() == [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]


def test_tensor_arg_extrema_return_first_indices() -> None:
    values = tynx.Tensor([[1.0, 5.0, 5.0], [4.0, 2.0, -1.0]])

    assert values.argmax().dtype == "int64"
    assert values.argmax().tolist() == [1]
    assert values.argmin().tolist() == [5]
    assert values.argmax(dim=1).tolist() == [1, 0]
    assert values.argmin(dim=-1, keepdim=True).tolist() == [[0], [2]]
    assert values.argmax(keepdim=True).shape == (1, 1)

    integers = tynx.Tensor([[1, 3], [5, 2]], dtype="int64")
    assert integers.argmax(dim=0).tolist() == [1, 0]

    with pytest.raises(TypeError, match="do not support bool"):
        tynx.Tensor([True, False], dtype="bool").argmax()


def test_tensor_arg_extrema_select_first_nan() -> None:
    values = tynx.Tensor([[1.0, float("nan"), 3.0], [float("nan"), 5.0, float("nan")]])

    assert values.argmax().tolist() == [1]
    assert values.argmin().tolist() == [1]
    assert values.argmax(dim=1).tolist() == [1, 0]
    assert values.argmin(dim=1).tolist() == [1, 0]


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


def test_adam_updates_tied_parameters_and_preserves_gradients() -> None:
    parameter = tynx.Parameter([2.0])
    optimizer = tynx.optim.Adam([parameter, parameter], lr=0.1)

    (parameter * 3).backward()
    optimizer.step()

    assert parameter.tolist() == pytest.approx([1.9])
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([3.0])
    assert optimizer.parameter_count == 1
    assert optimizer.state_size == 1
    optimizer.zero_grad()
    assert parameter.grad is None


def test_adam_configuration_state_and_mutable_learning_rate() -> None:
    parameter = tynx.Parameter([2.0])
    optimizer = tynx.optim.Adam(
        [parameter],
        lr=0.1,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=0.02,
        amsgrad=True,
    )

    (parameter * 2).backward()
    optimizer.step()
    optimizer.zero_grad()
    (parameter * 2).backward()
    optimizer.learning_rate = 0.2
    optimizer.step()

    assert optimizer.lr == pytest.approx(0.2)
    assert optimizer.betas == pytest.approx((0.8, 0.95))
    assert optimizer.eps == pytest.approx(1e-7)
    assert optimizer.weight_decay == pytest.approx(0.02)
    assert optimizer.amsgrad is True
    assert optimizer.state_size == 1
    assert "amsgrad=true" in repr(optimizer).lower()


def test_adam_and_adamw_use_coupled_and_decoupled_weight_decay() -> None:
    adam_parameter = tynx.Parameter([2.0])
    adamw_parameter = tynx.Parameter([2.0])
    adam = tynx.optim.Adam([adam_parameter], lr=0.1, betas=(0.0, 0.0), eps=0.0, weight_decay=0.5)
    adamw = tynx.optim.AdamW([adamw_parameter], lr=0.1, betas=(0.0, 0.0), eps=0.0, weight_decay=0.5)

    adam_parameter.backward()
    adamw_parameter.backward()
    adam.step()
    adamw.step()

    assert adam_parameter.tolist() == pytest.approx([1.9])
    assert adamw_parameter.tolist() == pytest.approx([1.8])
    assert tynx.optim.AdamW([tynx.Parameter([1.0])]).weight_decay == pytest.approx(0.01)


def test_adam_rejects_invalid_parameters_and_configuration() -> None:
    with pytest.raises(ValueError, match="at least one Parameter"):
        tynx.optim.Adam([])
    with pytest.raises(TypeError, match="only Parameter objects"):
        tynx.optim.Adam([tynx.Tensor([1.0])])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="beta1"):
        tynx.optim.Adam([tynx.Parameter([1.0])], betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="epsilon"):
        tynx.optim.AdamW([tynx.Parameter([1.0])], eps=-1.0)


def test_adam_trains_an_authored_python_linear_model() -> None:
    weight = tynx.Parameter([0.0], name="weight")
    bias = tynx.Parameter([0.0], name="bias")
    optimizer = tynx.optim.Adam([weight, bias], lr=0.05)
    inputs = tynx.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    targets = tynx.Tensor([-3.0, -1.0, 1.0, 3.0, 5.0])

    for _ in range(300):
        optimizer.zero_grad()
        prediction = inputs * weight + bias
        error = prediction - targets
        (error * error).mean().backward()
        optimizer.step()

    assert weight.item() == pytest.approx(2.0, abs=1e-3)
    assert bias.item() == pytest.approx(1.0, abs=1e-3)


def test_named_adam_state_dict_resumes_exactly_on_a_fresh_model() -> None:
    def train_step(model: tynx.nn.Linear, optimizer: tynx.optim.Adam) -> None:
        optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(model(tynx.Tensor([[2.0]])), tynx.Tensor([[5.0]]))
        loss.backward()
        optimizer.step()

    model = tynx.nn.Linear(1, 1)
    model.weight.copy_(tynx.Tensor([[1.5]]))
    assert model.bias is not None
    model.bias.copy_(tynx.Tensor([0.5]))
    optimizer = tynx.optim.Adam(
        model.named_parameters(),
        lr=0.03,
        betas=(0.8, 0.95),
        eps=1e-6,
        amsgrad=True,
    )
    train_step(model, optimizer)
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    assert optimizer_state["version"] == 1
    assert optimizer_state["optimizer"] == "Adam"
    assert optimizer_state["parameter_names"] == ["bias", "weight"]
    native_state = optimizer_state["state"]
    assert isinstance(native_state, dict)
    assert sorted(native_state) == ["bias", "weight"]
    weight_state = native_state["weight"]
    assert isinstance(weight_state, dict)
    assert isinstance(weight_state["exp_avg"], tynx.Tensor)
    assert isinstance(weight_state["max_exp_avg_sq"], tynx.Tensor)

    train_step(model, optimizer)
    expected = model.state_dict()

    resumed_model = tynx.nn.Linear(1, 1)
    resumed_model.load_state_dict(model_state)
    resumed = tynx.optim.Adam(resumed_model.named_parameters(), lr=0.9, amsgrad=True)
    resumed.load_state_dict(optimizer_state)
    assert resumed.lr == pytest.approx(0.03)
    assert resumed.betas == pytest.approx((0.8, 0.95))
    train_step(resumed_model, resumed)

    for name, value in resumed_model.state_dict().items():
        assert value.tolist() == expected[name].tolist()


def test_optimizer_state_dict_requires_stable_names_and_rejects_incompatible_payloads() -> None:
    parameter = tynx.Parameter([2.0])
    unnamed = tynx.optim.SGD([parameter], lr=0.1, momentum=0.9)
    with pytest.raises(ValueError, match="named_parameters"):
        unnamed.state_dict()

    with pytest.raises(TypeError, match="cannot mix"):
        tynx.optim.SGD([parameter, ("weight", parameter)], lr=0.1)

    optimizer = tynx.optim.SGD([("weight", parameter)], lr=0.1, momentum=0.9)
    parameter.backward()
    optimizer.step()
    state = optimizer.state_dict()
    assert state["optimizer"] == "SGD"

    destination = tynx.Parameter([parameter.item()])
    wrong_name = tynx.optim.SGD([("other", destination)], lr=0.7, momentum=0.9)
    with pytest.raises(ValueError, match="parameter names do not match"):
        wrong_name.load_state_dict(state)
    assert wrong_name.lr == pytest.approx(0.7)

    adamw = tynx.optim.AdamW([("weight", destination)])
    with pytest.raises(ValueError, match="cannot load SGD state into AdamW"):
        adamw.load_state_dict(state)

    invalid = dict(state)
    invalid["version"] = 999
    with pytest.raises(ValueError, match="unsupported optimizer state_dict version"):
        optimizer.load_state_dict(invalid)


def test_clip_grad_norm_returns_pre_clip_norm_and_deduplicates_parameters() -> None:
    parameter = tynx.Parameter([3.0, 4.0])
    (parameter * parameter).sum().backward()

    total_norm = tynx.nn.utils.clip_grad_norm_([parameter, parameter], 5.0)

    assert total_norm.shape == (1,)
    assert total_norm.item() == pytest.approx(10.0)
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([3.0, 4.0], abs=1e-5)


def test_clip_grad_value_clamps_gradients_and_skips_missing_ones() -> None:
    parameter = tynx.Parameter([-3.0, 4.0])
    missing = tynx.Parameter([1.0])
    (parameter * parameter).sum().backward()

    tynx.nn.utils.clip_grad_value_([parameter, parameter, missing], 2.0)

    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([-2.0, 2.0])
    assert missing.grad is None


def test_gradient_clipping_validates_arguments_and_handles_missing_gradients() -> None:
    parameter = tynx.Parameter([3.0, 4.0])

    assert tynx.nn.utils.clip_grad_norm_([parameter], 1.0).item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="at least one Parameter"):
        tynx.nn.utils.clip_grad_norm_([], 1.0)
    with pytest.raises(ValueError, match=r"norm_type=2\.0"):
        tynx.nn.utils.clip_grad_norm_([parameter], 1.0, norm_type=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        tynx.nn.utils.clip_grad_value_([parameter], -1.0)


def test_state_discovery_walks_supported_objects_and_stabilizes_aliases() -> None:
    class Layer:
        def __init__(self) -> None:
            self.weight = tynx.Parameter([1.0], name="ignored_nested_name")
            self.bias = tynx.Parameter([0.0])

    class Model:
        def __init__(self) -> None:
            shared = Layer()
            self.right = shared
            self.left = shared
            self.heads = {"value": [tynx.Parameter([2.0])]}
            self.self_cycle = self

    model = Model()

    named = tynx.nn.state.named_parameters(model)
    aliases = tynx.nn.state.get_parameter_aliases(model)

    assert [name for name, _ in named] == ["heads.value.0", "left.bias", "left.weight"]
    assert tynx.nn.state.get_parameters(model) == [parameter for _, parameter in named]
    assert aliases == {
        "heads.value.0": (),
        "left.bias": ("right.bias",),
        "left.weight": ("right.weight",),
    }


def test_state_discovery_never_invokes_dynamic_python_behavior() -> None:
    class Trap:
        def __init__(self) -> None:
            self.parameter = tynx.Parameter([1.0])

        @property
        def exploding_property(self) -> object:
            raise AssertionError("property was invoked")

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"__getattr__ was invoked for {name}")

        def __iter__(self) -> object:
            raise AssertionError("iterator was invoked")

    assert [name for name, _ in tynx.nn.state.named_parameters(Trap())] == ["parameter"]


def test_state_discovery_rejects_slotted_objects_and_ambiguous_dictionary_keys() -> None:
    class Slotted:
        __slots__ = ("parameter",)

        def __init__(self) -> None:
            self.parameter = tynx.Parameter([1.0])

    with pytest.raises(TypeError, match="__slots__ objects are unsupported"):
        tynx.nn.state.get_parameters(Slotted())
    with pytest.raises(TypeError, match="require string keys"):
        tynx.nn.state.get_parameters({1: tynx.Parameter([1.0])})
    with pytest.raises(ValueError, match="cannot contain"):
        tynx.nn.state.get_parameters({"ambiguous.path": tynx.Parameter([1.0])})


def test_functional_mse_loss_reductions_and_gradients() -> None:
    prediction = tynx.Tensor([1.0, 3.0], requires_grad=True)
    target = tynx.Tensor([0.0, 1.0])

    assert tynx.nn.functional.mse_loss(prediction, target, "none").tolist() == [1.0, 4.0]
    assert tynx.nn.functional.mse_loss(prediction, target, "sum").item() == pytest.approx(5.0)
    loss = tynx.nn.functional.mse_loss(prediction, target)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad.tolist() == pytest.approx([1.0, 2.0])


def test_functional_cross_entropy_matches_reference_and_backpropagates() -> None:
    logits = tynx.Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], requires_grad=True)
    targets = tynx.Tensor([0, 2], dtype="int64")

    losses = tynx.nn.functional.cross_entropy(logits, targets, reduction="none")
    loss = tynx.nn.functional.cross_entropy(logits, targets)

    assert losses.tolist() == pytest.approx([0.40760595, 0.40760595])
    assert loss.item() == pytest.approx(0.40760595)
    loss.backward()
    assert logits.grad is not None
    gradient = logits.grad.tolist()
    assert gradient[0] == pytest.approx([-0.1673795, 0.12236424, 0.04501529], abs=1e-6)
    assert gradient[1] == pytest.approx([0.04501529, 0.12236424, -0.1673795], abs=1e-6)

    assert targets.unsqueeze(1).shape == (2, 1)
    assert tynx.Tensor([True, False], dtype="bool").reshape(1, 2).tolist() == [[True, False]]


def test_functional_binary_cross_entropy_with_logits_is_stable() -> None:
    logits = tynx.Tensor([-100.0, 0.0, 100.0], requires_grad=True)
    targets = tynx.Tensor([0.0, 1.0, 1.0])

    losses = tynx.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    assert losses.tolist() == pytest.approx([0.0, 0.6931472, 0.0], abs=1e-6)
    losses.sum().backward()
    assert logits.grad is not None
    assert logits.grad.tolist() == pytest.approx([0.0, -0.5, 0.0], abs=1e-6)


def test_functional_losses_validate_shapes_dtypes_and_reductions() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        tynx.nn.functional.mse_loss(tynx.Tensor([1.0]), tynx.Tensor([[1.0]]))
    with pytest.raises(ValueError, match="rank-2 float32"):
        tynx.nn.functional.cross_entropy(tynx.Tensor([1.0, 2.0]), tynx.Tensor([0], dtype="int64"))
    with pytest.raises(ValueError, match="rank-1 int64"):
        tynx.nn.functional.cross_entropy(tynx.Tensor([[1.0, 2.0]]), tynx.Tensor([0.0]))
    with pytest.raises(ValueError, match="reduction must"):
        tynx.nn.functional.mse_loss(
            tynx.Tensor([1.0]),
            tynx.Tensor([1.0]),
            "invalid",  # type: ignore[arg-type]
        )


def test_linear_supports_vector_batched_forward_and_gradients() -> None:
    layer = tynx.nn.Linear(2, 2)
    layer.weight = tynx.Parameter([[2.0, 0.0], [0.0, 3.0]], name="weight")
    layer.bias = tynx.Parameter([1.0, -1.0], name="bias")
    vector = tynx.Tensor([2.0, 4.0], requires_grad=True)
    batch = tynx.Tensor([[[1.0, 2.0]], [[3.0, 4.0]]])

    output = layer(vector)

    assert output.shape == (2,)
    assert output.tolist() == pytest.approx([5.0, 11.0])
    assert layer(batch).tolist() == [[[3.0, 5.0]], [[7.0, 11.0]]]
    output.sum().backward()
    assert vector.grad is not None
    assert vector.grad.tolist() == pytest.approx([2.0, 3.0])
    assert layer.weight.grad is not None
    assert layer.weight.grad.tolist()[0] == pytest.approx([2.0, 4.0])
    assert layer.weight.grad.tolist()[1] == pytest.approx([2.0, 4.0])


def test_module_sequential_discovery_and_recursive_modes() -> None:
    model = tynx.nn.Sequential(tynx.nn.Linear(2, 3), tynx.nn.ReLU(), tynx.nn.Linear(3, 1))

    class Wrapper:
        def __init__(self) -> None:
            self.model = model
            self.alias = model[0]
            self.cycle = self

    wrapper = Wrapper()

    assert [name for name, _ in model.named_parameters()] == [
        "layers.0.bias",
        "layers.0.weight",
        "layers.2.bias",
        "layers.2.weight",
    ]
    assert len(model.parameters()) == 4
    assert "in_features=2" in repr(model[0])
    assert all(module.training for module in (model, model[0], model[1], model[2]))

    assert tynx.nn.state.eval(wrapper) is wrapper
    assert not any(module.training for module in (model, model[0], model[1], model[2]))
    assert model.train() is model
    assert all(module.training for module in (model, model[0], model[1], model[2]))


def test_linear_and_sequential_validate_construction_and_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tynx.nn.Linear(0, 2)
    with pytest.raises(TypeError, match="bias must be a bool"):
        tynx.nn.Linear(2, 2, bias=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a Module"):
        tynx.nn.Sequential(tynx.nn.ReLU(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="final dimension"):
        tynx.nn.Linear(3, 2)(tynx.Tensor([[1.0, 2.0]]))
    with pytest.raises(TypeError, match="float32"):
        tynx.nn.Linear(2, 2)(tynx.Tensor([[1, 2]], dtype="int64"))


def test_authored_sequential_mlp_converges_without_a_trainer() -> None:
    tynx.manual_seed(0)
    model = tynx.nn.Sequential(tynx.nn.Linear(1, 4), tynx.nn.ReLU(), tynx.nn.Linear(4, 1))
    optimizer = tynx.optim.Adam(model.parameters(), lr=0.05)
    inputs = tynx.Tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    targets = tynx.Tensor([[-1.0], [0.0], [1.0], [2.0], [3.0]])

    for _ in range(400):
        optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()

    assert tynx.nn.functional.mse_loss(model(inputs), targets).item() < 1e-4


def test_layer_norm_values_affine_gradients_and_parameter_discovery() -> None:
    layer = tynx.nn.LayerNorm(2)
    input = tynx.Tensor([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)

    output = layer(input)

    assert output.tolist()[0] == pytest.approx([-0.999995, 0.999995], abs=1e-6)
    assert output.tolist()[1] == pytest.approx([-0.999995, 0.999995], abs=1e-6)
    assert [name for name, _ in layer.named_parameters()] == ["bias", "weight"]
    output.sum().backward()
    assert input.grad is not None
    assert input.grad.tolist()[0] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.tolist() == pytest.approx([-1.99999, 1.99999], abs=1e-5)
    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert layer.bias.grad.tolist() == pytest.approx([2.0, 2.0])


def test_layer_norm_supports_multi_axis_and_non_affine_modes() -> None:
    layer = tynx.nn.LayerNorm((2, 2), elementwise_affine=False)
    input = tynx.Tensor([[[1.0, 2.0], [3.0, 4.0]]])

    output = layer(input)

    assert output.shape == (1, 2, 2)
    assert output.mean((1, 2)).item() == pytest.approx(0.0, abs=1e-6)
    assert layer.weight is None
    assert layer.bias is None
    assert layer.parameters() == []


def test_layer_norm_validates_configuration_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tynx.nn.LayerNorm(())
    with pytest.raises(ValueError, match="positive integers"):
        tynx.nn.LayerNorm((2, 0))
    with pytest.raises(ValueError, match="non-negative"):
        tynx.nn.LayerNorm(2, eps=-1.0)
    with pytest.raises(TypeError, match="must be bool"):
        tynx.nn.LayerNorm(2, elementwise_affine=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trailing shape"):
        tynx.nn.LayerNorm(3)(tynx.Tensor([[1.0, 2.0]]))
    with pytest.raises(TypeError, match="float32"):
        tynx.nn.LayerNorm(2)(tynx.Tensor([[1, 2]], dtype="int64"))


def test_batch_norm1d_updates_running_stats_and_preserves_gradients() -> None:
    layer = tynx.nn.BatchNorm1d(2, eps=0.0, momentum=0.1)
    input = tynx.Tensor([[1.0, 10.0], [3.0, 14.0]], requires_grad=True)

    output = layer(input)

    assert output.tolist()[0] == pytest.approx([-1.0, -1.0])
    assert output.tolist()[1] == pytest.approx([1.0, 1.0])
    assert layer.running_mean is not None
    assert layer.running_mean.tolist() == pytest.approx([0.2, 1.2])
    assert layer.running_var is not None
    assert layer.running_var.tolist() == pytest.approx([1.1, 1.7])
    assert [name for name, _ in layer.named_parameters()] == ["bias", "weight"]
    assert [name for name, _ in layer.named_buffers()] == ["running_mean", "running_var"]

    output.sum().backward()

    assert input.grad is not None
    for row in input.grad.tolist():
        assert row == pytest.approx([0.0, 0.0], abs=1e-6)
    assert layer.weight is not None
    assert layer.weight.grad is not None
    assert layer.weight.grad.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert layer.bias.grad.tolist() == pytest.approx([2.0, 2.0])


def test_batch_norm_eval_uses_running_stats_without_updating_them() -> None:
    layer = tynx.nn.BatchNorm1d(2, eps=0.0, momentum=0.1)
    input = tynx.Tensor([[1.0, 10.0], [3.0, 14.0]])
    layer(input)
    assert layer.running_mean is not None
    assert layer.running_var is not None
    mean_before = layer.running_mean.tolist()
    variance_before = layer.running_var.tolist()

    layer.eval()
    output = layer(input)

    assert output.tolist()[0] == pytest.approx(
        [(1.0 - 0.2) / math.sqrt(1.1), (10.0 - 1.2) / math.sqrt(1.7)]
    )
    assert output.tolist()[1] == pytest.approx(
        [(3.0 - 0.2) / math.sqrt(1.1), (14.0 - 1.2) / math.sqrt(1.7)]
    )
    assert layer.running_mean.tolist() == mean_before
    assert layer.running_var.tolist() == variance_before


def test_batch_norm2d_supports_non_affine_batch_stats_without_buffers() -> None:
    layer = tynx.nn.BatchNorm2d(2, affine=False, track_running_stats=False)
    input = tynx.Tensor(
        [
            [[[1.0], [3.0]], [[10.0], [14.0]]],
            [[[5.0], [7.0]], [[18.0], [22.0]]],
        ]
    )

    output = layer.eval()(input)

    assert output.shape == input.shape
    assert output.mean((0, 2, 3)).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert layer.parameters() == []
    assert layer.buffers() == []


def test_batch_norm_validates_configuration_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tynx.nn.BatchNorm1d(0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tynx.nn.BatchNorm1d(2, momentum=1.1)
    with pytest.raises(TypeError, match="must be bool"):
        tynx.nn.BatchNorm1d(2, affine=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expects input rank"):
        tynx.nn.BatchNorm2d(2)(tynx.Tensor([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="expected 3 channels"):
        tynx.nn.BatchNorm1d(3)(tynx.Tensor([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="more than one value"):
        tynx.nn.BatchNorm1d(2)(tynx.Tensor([[1.0, 2.0]]))
    with pytest.raises(TypeError, match="float32"):
        tynx.nn.BatchNorm1d(2)(tynx.Tensor([[1, 2], [3, 4]], dtype="int64"))


def test_buffer_is_stable_non_trainable_state_and_not_an_optimizer_parameter() -> None:
    buffer = tynx.Buffer([1.0, 2.0], name="running_mean")

    assert isinstance(buffer, tynx.Tensor)
    assert buffer.name == "running_mean"
    assert buffer.requires_grad is False
    assert buffer.grad is None
    assert (buffer * 2).tolist() == pytest.approx([2.0, 4.0])
    assert (buffer * 2).requires_grad is False
    with pytest.raises(TypeError, match="only Parameter objects"):
        tynx.optim.SGD([buffer], lr=0.1)  # type: ignore[list-item]


def test_buffer_discovery_is_separate_cycle_safe_and_alias_aware() -> None:
    class Stateful(tynx.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = tynx.Buffer([0.0], name="ignored_nested_name")
            self.right = shared
            self.left = shared
            self.weight = tynx.Parameter([1.0])
            self.cycle = self

        def forward(self, input: tynx.Tensor) -> tynx.Tensor:
            return input * self.weight + self.left

    module = Stateful()

    assert [name for name, _ in module.named_parameters()] == ["weight"]
    assert [name for name, _ in module.named_buffers()] == ["left"]
    assert module.buffers() == [module.left]
    assert tynx.nn.state.get_buffer_aliases(module) == {"left": ("right",)}


def test_state_dict_snapshots_and_loads_existing_parameter_and_buffer_slots() -> None:
    class Stateful(tynx.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = tynx.Parameter([1.0])
            self.running = tynx.Buffer([2.0])

        def forward(self, input: tynx.Tensor) -> tynx.Tensor:
            return input * self.weight + self.running

    module = Stateful()
    weight_identity = id(module.weight)
    buffer_identity = id(module.running)
    snapshot = module.state_dict()

    module.weight.copy_(tynx.Tensor([5.0]))
    module.running.copy_(tynx.Tensor([7.0]))

    assert snapshot["weight"].tolist() == pytest.approx([1.0])
    assert snapshot["running"].tolist() == pytest.approx([2.0])
    result = module.load_state_dict(snapshot)
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert id(module.weight) == weight_identity
    assert id(module.running) == buffer_identity
    assert module.weight.tolist() == pytest.approx([1.0])
    assert module.running.tolist() == pytest.approx([2.0])


def test_state_dict_non_strict_results_and_validation_precede_mutation() -> None:
    class Pair(tynx.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = tynx.Parameter([1.0])
            self.right = tynx.Parameter([2.0])

        def forward(self, input: tynx.Tensor) -> tynx.Tensor:
            return input * self.left + self.right

    module = Pair()

    with pytest.raises(ValueError, match="key mismatch"):
        module.load_state_dict({"left": tynx.Tensor([9.0])})
    assert module.left.tolist() == pytest.approx([1.0])

    with pytest.raises(ValueError, match="expected shape"):
        module.load_state_dict({"left": tynx.Tensor([9.0]), "right": tynx.Tensor([8.0, 7.0])})
    assert module.left.tolist() == pytest.approx([1.0])
    assert module.right.tolist() == pytest.approx([2.0])

    result = module.load_state_dict(
        {"left": tynx.Tensor([3.0]), "extra": tynx.Tensor([4.0])}, strict=False
    )
    assert result.missing_keys == ("right",)
    assert result.unexpected_keys == ("extra",)
    assert module.left.tolist() == pytest.approx([3.0])
    assert module.right.tolist() == pytest.approx([2.0])


def test_state_dict_detaches_all_sources_before_swap_and_preserves_optimizer_identity() -> None:
    class Pair(tynx.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = tynx.Parameter([1.0])
            self.right = tynx.Parameter([2.0])

        def forward(self, input: tynx.Tensor) -> tynx.Tensor:
            return input * self.left + self.right

    module = Pair()
    optimizer = tynx.optim.SGD([module.left], lr=0.1)

    module.load_state_dict({"left": module.right, "right": module.left})

    assert module.left.tolist() == pytest.approx([2.0])
    assert module.right.tolist() == pytest.approx([1.0])
    module.left.backward()
    optimizer.step()
    assert module.left.tolist() == pytest.approx([1.9])


def test_stable_state_copy_and_state_dict_types_are_validated() -> None:
    with pytest.raises(TypeError, match="stable Parameter or Buffer"):
        tynx.Tensor([1.0]).copy_(tynx.Tensor([2.0]))
    with pytest.raises(TypeError, match="float32"):
        tynx.Buffer([1.0]).copy_(tynx.Tensor([2], dtype="int64"))
    with pytest.raises(TypeError, match="must be a Tensor"):
        tynx.nn.state.load_state_dict(
            tynx.nn.Linear(1, 1),
            {"weight": [1.0]},  # type: ignore[dict-item]
            strict=False,
        )


def test_target_network_soft_and_hard_updates_are_off_tape_and_identity_preserving() -> None:
    class Stateful(tynx.nn.Module):
        def __init__(self, weight: float, running: float) -> None:
            super().__init__()
            self.weight = tynx.Parameter([weight])
            self.running = tynx.Buffer([running])

        def forward(self, input: tynx.Tensor) -> tynx.Tensor:
            return input * self.weight + self.running

    target = Stateful(1.0, 10.0)
    source = Stateful(5.0, 30.0)
    weight_identity = id(target.weight)
    buffer_identity = id(target.running)
    target.weight.backward()
    optimizer = tynx.optim.SGD(target.parameters(), lr=0.1)

    tynx.nn.utils.soft_update_(target, source, 0.25)

    assert id(target.weight) == weight_identity
    assert id(target.running) == buffer_identity
    assert target.weight.tolist() == pytest.approx([2.0])
    assert target.running.tolist() == pytest.approx([30.0])
    assert target.weight.requires_grad
    assert target.weight.grad is not None
    assert target.weight.grad.tolist() == pytest.approx([1.0])
    assert source.weight.tolist() == pytest.approx([5.0])
    optimizer.step()
    assert target.weight.tolist() == pytest.approx([1.9])

    target.running.copy_(tynx.Tensor([10.0]))
    tynx.nn.utils.soft_update_(target, source, 0.25, buffer_policy="average")
    assert target.weight.tolist() == pytest.approx([2.675])
    assert target.running.tolist() == pytest.approx([15.0])

    tynx.nn.utils.hard_update_(target, source)
    assert target.weight.tolist() == pytest.approx([5.0])
    assert target.running.tolist() == pytest.approx([30.0])


def test_target_network_updates_validate_all_state_before_mutation() -> None:
    class Stateful(tynx.nn.Module):
        def __init__(self, data: tuple[float, ...], *, with_buffer: bool = True) -> None:
            super().__init__()
            self.weight = tynx.Parameter(data)
            if with_buffer:
                self.running = tynx.Buffer([2.0])

        def forward(self, input: tynx.Tensor) -> tynx.Tensor:
            return input * self.weight

    target = Stateful((1.0,))

    with pytest.raises(ValueError, match="source shape"):
        tynx.nn.utils.soft_update_(target, Stateful((3.0, 4.0)), 0.5)
    assert target.weight.tolist() == pytest.approx([1.0])
    assert target.running.tolist() == pytest.approx([2.0])

    with pytest.raises(ValueError, match="buffer names differ"):
        tynx.nn.utils.hard_update_(target, Stateful((3.0,), with_buffer=False))
    assert target.weight.tolist() == pytest.approx([1.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tynx.nn.utils.soft_update_(target, target, -0.1)
    with pytest.raises(ValueError, match="buffer_policy"):
        tynx.nn.utils.soft_update_(
            target,
            target,
            0.5,
            buffer_policy="invalid",  # type: ignore[arg-type]
        )


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


@pytest.mark.parametrize("operator", ["add", "sub", "mul", "truediv", "matmul", "pow"])
def test_tensor_inplace_arithmetic_fails_instead_of_losing_leaf_gradients(
    operator: str,
) -> None:
    value = tynx.Tensor([1.0], requires_grad=True)
    other = tynx.Tensor([2.0])

    with pytest.raises(RuntimeError, match="in-place arithmetic"):
        if operator == "add":
            value += other
        elif operator == "sub":
            value -= other
        elif operator == "mul":
            value *= other
        elif operator == "truediv":
            value /= other
        elif operator == "matmul":
            value @= other
        else:
            value **= other

    assert value.is_leaf
    assert value.requires_grad
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


def test_tensor_repeated_backward_reports_consumed_graph_without_dispatch() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    output = (value * value).sum()

    output.backward()
    with pytest.raises(ValueError, match="already freed by a previous backward"):
        output.backward()


def test_tensor_sibling_results_share_backward_consumption() -> None:
    value = tynx.Tensor([2.0, 3.0], requires_grad=True)
    shared = value * value
    first = shared.sum()
    second = shared.mean()

    first.backward()
    with pytest.raises(ValueError, match="already freed by a previous backward"):
        second.backward()


def test_tensor_leaf_allows_repeated_backward() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)

    value.backward()
    value.backward()

    assert value.grad is not None
    assert value.grad.tolist() == [2.0]


def test_parameter_allows_backward_after_value_generation_changes() -> None:
    parameter = tynx.Parameter([2.0])

    parameter.backward()
    parameter.copy_(tynx.Tensor([3.0]))
    parameter.zero_grad()
    parameter.backward()

    assert parameter.grad is not None
    assert parameter.grad.tolist() == [1.0]


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
