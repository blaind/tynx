from __future__ import annotations

import pytest
import tynx


def test_public_dtype_constants_work_across_dtype_entry_points() -> None:
    assert tynx.float32 == "float32"
    assert tynx.int64 == "int64"
    assert tynx.bool == "bool"
    assert tynx.ones((2,), dtype=tynx.float32).dtype == "float32"
    assert tynx.Tensor([1, 2], dtype=tynx.int64).dtype == "int64"
    assert tynx.Tensor([0.0, 1.0]).cast(tynx.bool).tolist() == [False, True]
    assert tynx.Tensor([1, 2], dtype="int64").to(dtype=tynx.float32).dtype == "float32"


def test_tensor_to_accepts_a_positional_dtype() -> None:
    value = tynx.Tensor([1, 2], dtype=tynx.int64)

    converted = value.to(tynx.float32)

    assert converted.dtype == "float32"
    assert converted.tolist() == [1.0, 2.0]


def test_tensor_to_rejects_ambiguous_or_invalid_positional_arguments() -> None:
    value = tynx.Tensor([1.0])

    with pytest.raises(TypeError, match="dtype both positionally and by keyword"):
        value.to(tynx.float32, dtype=tynx.int64)  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="must be a Device or dtype string"):
        value.to(3)  # type: ignore[call-overload]


def test_device_can_be_selected_during_construction_and_factories() -> None:
    cpu = tynx.Device("cpu")

    value = tynx.Tensor([1.0, 2.0], device=cpu)
    created = tynx.ones((2, 3), device=cpu)

    assert value.device == cpu
    assert created.device == cpu
    assert tynx.Device("flex") == cpu


def test_cast_converts_supported_dtype_pairs() -> None:
    floats = tynx.Tensor([-1.5, 0.0, 2.9])
    integers = floats.cast("int64")
    booleans = floats.cast("bool")

    assert integers.dtype == "int64"
    assert integers.tolist() == [-1, 0, 2]
    assert booleans.tolist() == [True, False, True]
    assert integers.cast("float32").tolist() == [-1.0, 0.0, 2.0]
    assert booleans.cast("int64").tolist() == [1, 0, 1]


def test_float_operations_explain_how_to_convert_discrete_tensors() -> None:
    floats = tynx.Tensor([1.0, 2.0])
    integers = tynx.Tensor([1, 2], dtype="int64")

    with pytest.raises(
        TypeError,
        match=r'got int64; convert it with \.cast\("float32"\)',
    ):
        _ = floats + integers

    assert (floats + integers.cast("float32")).tolist() == [2.0, 4.0]


def test_tensor_copy_constructor_can_cast_and_select_device() -> None:
    source = tynx.Tensor([1, 2, 3], dtype="int64")

    copied = tynx.Tensor(source, dtype="float32", device=tynx.Device("cpu"))

    assert copied.dtype == "float32"
    assert copied.tolist() == [1.0, 2.0, 3.0]
    assert copied.device == tynx.Device("cpu")


def test_float_to_preserves_gradient_path_while_discrete_cast_detaches() -> None:
    value = tynx.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    moved = value.to(tynx.Device("cpu"), dtype="float32")
    moved.sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [1.0, 1.0, 1.0]
    assert not value.cast("int64").requires_grad


def test_dtype_and_device_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        tynx.Device("cuda")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported Tensor dtype"):
        tynx.Tensor([1.0]).cast("float64")  # type: ignore[arg-type]
