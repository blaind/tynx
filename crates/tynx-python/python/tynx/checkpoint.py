"""Versioned, resumable model-and-optimizer checkpoints."""

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Protocol, Union, cast, overload

from ._tynx import Tensor
from .nn.state import (
    LoadStateResult,
    _apply_prepared_state,
    _prepare_state_dict_load,
    get_state_dict,
    load_state_dict,
)

_CHECKPOINT_FORMAT = "tynx.training"
_CHECKPOINT_VERSION = 1
_Path = Union[str, PathLike[str]]
_JsonScalar = Union[None, bool, int, float, str]
_Encoded = Union[_JsonScalar, list["_Encoded"], dict[str, "_Encoded"]]

if TYPE_CHECKING:
    from ._tynx import TensorData, TensorDType


class _Optimizer(Protocol):
    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...


class _Scheduler(Protocol):
    @property
    def optimizer(self) -> object: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...


@overload
def save_checkpoint(
    path: _Path,
    model: object,
    optimizer: Optional[_Optimizer] = None,
    *,
    scheduler: Optional[_Scheduler] = None,
) -> None: ...


@overload
def save_checkpoint(
    path: object,
    model: _Path,
    optimizer: None = None,
    *,
    scheduler: None = None,
) -> None: ...


@overload
def save_checkpoint(
    path: object,
    model: _Optimizer,
    optimizer: _Path,
    *,
    scheduler: Optional[_Scheduler] = None,
) -> None: ...


def save_checkpoint(
    path: object,
    model: object,
    optimizer: object = None,
    *,
    scheduler: Optional[_Scheduler] = None,
) -> None:
    """Atomically save model state and optional optimizer/scheduler state.

    Both ``save_checkpoint(path, model, optimizer)`` and the PyTorch-shaped
    ``save_checkpoint(model, optimizer, path)`` order are accepted. For weights-only
    checkpoints, use either ``save_checkpoint(path, model)`` or
    ``save_checkpoint(model, path)``.
    """
    target, source_model, source_optimizer = _normalize_save_arguments(path, model, optimizer)
    _validate_scheduler(source_optimizer, scheduler)
    model_state = get_state_dict(source_model)
    if not model_state:
        raise ValueError("cannot save checkpoint: model has no parameters or buffers")
    payload: dict[str, _Encoded] = {
        "format": _CHECKPOINT_FORMAT,
        "version": _CHECKPOINT_VERSION,
        "model": _encode(dict(model_state)),
        "optimizer": _encode(None if source_optimizer is None else source_optimizer.state_dict()),
        "scheduler": _encode(None if scheduler is None else scheduler.state_dict()),
    }
    serialized = json.dumps(payload, allow_nan=True, separators=(",", ":"), sort_keys=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


@overload
def load_checkpoint(
    path: _Path,
    model: object,
    optimizer: Optional[_Optimizer] = None,
    strict: bool = True,
    *,
    scheduler: Optional[_Scheduler] = None,
) -> LoadStateResult: ...


@overload
def load_checkpoint(
    path: object,
    model: _Path,
    optimizer: None = None,
    strict: bool = True,
    *,
    scheduler: None = None,
) -> LoadStateResult: ...


@overload
def load_checkpoint(
    path: object,
    model: _Optimizer,
    optimizer: _Path,
    strict: bool = True,
    *,
    scheduler: Optional[_Scheduler] = None,
) -> LoadStateResult: ...


def load_checkpoint(
    path: object,
    model: object,
    optimizer: object = None,
    strict: bool = True,
    *,
    scheduler: Optional[_Scheduler] = None,
) -> LoadStateResult:
    """Restore combined state without publishing a partially validated component.

    Both ``load_checkpoint(path, model, optimizer)`` and the PyTorch-shaped
    ``load_checkpoint(model, optimizer, path)`` order are accepted. For weights-only
    checkpoints, use either ``load_checkpoint(path, model)`` or
    ``load_checkpoint(model, path)``.
    """
    target, destination_model, destination_optimizer = _normalize_load_arguments(
        path, model, optimizer
    )
    payload = _read_payload(target)
    model_state = _decode_state_dictionary(_required(payload, "model"), "model")
    if not model_state:
        raise ValueError("cannot load checkpoint: model state is empty")
    encoded_optimizer = _required(payload, "optimizer")
    optimizer_state = (
        None if destination_optimizer is None else _decode_optimizer_dictionary(encoded_optimizer)
    )
    _validate_scheduler(destination_optimizer, scheduler)
    encoded_scheduler = payload.get("scheduler")
    scheduler_state = (
        None if scheduler is None else _decode_component_dictionary(encoded_scheduler, "scheduler")
    )

    current, prepared, result = _prepare_state_dict_load(destination_model, model_state, strict)
    previous_model = get_state_dict(destination_model)
    previous_optimizer = (
        None if destination_optimizer is None else destination_optimizer.state_dict()
    )
    previous_scheduler = None if scheduler is None else scheduler.state_dict()
    try:
        if destination_optimizer is not None:
            assert optimizer_state is not None
            destination_optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            assert scheduler_state is not None
            scheduler.load_state_dict(scheduler_state)
        _apply_prepared_state(current, prepared)
    except BaseException:
        load_state_dict(destination_model, previous_model)
        if destination_optimizer is not None:
            assert previous_optimizer is not None
            destination_optimizer.load_state_dict(previous_optimizer)
        if scheduler is not None:
            assert previous_scheduler is not None
            scheduler.load_state_dict(previous_scheduler)
        raise
    return result


def _normalize_save_arguments(
    first: object,
    second: object,
    third: object,
) -> tuple[Path, object, Optional[_Optimizer]]:
    source_optimizer: object
    if isinstance(first, (str, os.PathLike)):
        path = Path(first)
        source_model = second
        source_optimizer = third
    elif isinstance(third, (str, os.PathLike)):
        path = Path(third)
        source_model = first
        source_optimizer = second
    elif isinstance(second, (str, os.PathLike)) and third is None:
        path = Path(second)
        source_model = first
        source_optimizer = None
    else:
        raise TypeError(
            "save_checkpoint expects (path, model[, optimizer]) or (model[, optimizer], path)"
        )

    if isinstance(source_model, (str, os.PathLike)):
        raise TypeError("save_checkpoint model must be a model object, not a path")
    if source_optimizer is not None:
        state_dict = getattr(source_optimizer, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("save_checkpoint optimizer must provide state_dict()")
    return path, source_model, cast(Optional[_Optimizer], source_optimizer)


def _normalize_load_arguments(
    first: object,
    second: object,
    third: object,
) -> tuple[Path, object, Optional[_Optimizer]]:
    destination_optimizer: object
    if isinstance(first, (str, os.PathLike)):
        path = Path(first)
        destination_model = second
        destination_optimizer = third
    elif isinstance(third, (str, os.PathLike)):
        path = Path(third)
        destination_model = first
        destination_optimizer = second
    elif isinstance(second, (str, os.PathLike)) and third is None:
        path = Path(second)
        destination_model = first
        destination_optimizer = None
    else:
        raise TypeError(
            "load_checkpoint expects (path, model[, optimizer]) or (model[, optimizer], path)"
        )

    if isinstance(destination_model, (str, os.PathLike)):
        raise TypeError("load_checkpoint model must be a model object, not a path")
    if destination_optimizer is not None:
        state_dict = getattr(destination_optimizer, "state_dict", None)
        load_state = getattr(destination_optimizer, "load_state_dict", None)
        if not callable(state_dict) or not callable(load_state):
            raise TypeError(
                "load_checkpoint optimizer must provide state_dict() and load_state_dict()"
            )
    return path, destination_model, cast(Optional[_Optimizer], destination_optimizer)


def _validate_scheduler(optimizer: Optional[_Optimizer], scheduler: Optional[_Scheduler]) -> None:
    if scheduler is None:
        return
    if optimizer is None:
        raise ValueError("checkpointing a scheduler requires its optimizer")
    if not callable(getattr(scheduler, "state_dict", None)) or not callable(
        getattr(scheduler, "load_state_dict", None)
    ):
        raise TypeError("checkpoint scheduler must provide state_dict() and load_state_dict()")
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise ValueError("checkpoint scheduler must belong to the supplied optimizer")


def _read_payload(path: _Path) -> dict[str, _Encoded]:
    try:
        with Path(path).open(encoding="utf-8") as checkpoint:
            value = json.load(checkpoint)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Tynx checkpoint JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("Tynx checkpoint root must be a dictionary")
    payload = cast(dict[str, _Encoded], value)
    if payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError("file is not a Tynx training checkpoint")
    version = payload.get("version")
    if version != _CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported Tynx checkpoint version {version!r}; expected {_CHECKPOINT_VERSION}"
        )
    return payload


def _encode(value: object) -> _Encoded:
    if value is None or type(value) in (bool, int, float, str):
        return cast(_JsonScalar, value)
    if isinstance(value, Tensor):
        return {
            "__type__": "tensor",
            "data": _encode(value.tolist()),
            "dtype": value.dtype,
            "shape": [int(dimension) for dimension in value.shape],
        }
    if type(value) is tuple:
        return {
            "__type__": "tuple",
            "items": [_encode(item) for item in cast(tuple[object, ...], value)],
        }
    if type(value) is list:
        return [_encode(item) for item in cast(list[object], value)]
    if type(value) is dict:
        encoded: dict[str, _Encoded] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise TypeError(
                    f"checkpoint dictionaries require string keys, got {type(key).__qualname__}"
                )
            encoded[key] = _encode(item)
        return encoded
    raise TypeError(f"checkpoint cannot serialize {type(value).__qualname__}")


def _decode(value: _Encoded, autodiff_float: bool) -> object:
    if not isinstance(value, (dict, list)):
        return value
    if isinstance(value, list):
        return [_decode(item, autodiff_float) for item in value]
    marker = value.get("__type__")
    if marker == "tensor":
        dtype = value.get("dtype")
        shape = value.get("shape")
        if not isinstance(dtype, str) or not _is_shape(shape):
            raise ValueError("invalid tensor metadata in Tynx checkpoint")
        data = cast("TensorData", _decode(_required(value, "data"), autodiff_float))
        tensor = Tensor(
            data,
            dtype=cast("TensorDType", dtype),
            requires_grad=autodiff_float and dtype == "float32",
        ).detach()
        expected_shape = tuple(cast(list[int], shape))
        if tensor.shape != expected_shape:
            raise ValueError(
                f"checkpoint tensor data has shape {tensor.shape}, expected {expected_shape}"
            )
        return tensor
    if marker == "tuple":
        items = _required(value, "items")
        if not isinstance(items, list):
            raise ValueError("checkpoint tuple items must be a list")
        return tuple(_decode(item, autodiff_float) for item in items)
    if marker is not None:
        raise ValueError(f"unknown checkpoint value type {marker!r}")
    return {key: _decode(item, autodiff_float) for key, item in value.items()}


def _decode_state_dictionary(value: _Encoded, field: str) -> dict[str, Tensor]:
    decoded = _decode(value, True)
    if not isinstance(decoded, dict):
        raise ValueError(f"checkpoint field {field!r} must be a dictionary")
    result: dict[str, Tensor] = {}
    for name, tensor in decoded.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError(f"checkpoint field {field!r} must map names to Tynx Tensors")
        result[name] = tensor
    return result


def _decode_optimizer_dictionary(value: _Encoded) -> dict[str, object]:
    return _decode_component_dictionary(value, "optimizer")


def _decode_component_dictionary(value: Optional[_Encoded], field: str) -> dict[str, object]:
    if value is None:
        raise ValueError(f"checkpoint does not contain {field} state")
    decoded = _decode(value, True)
    if not isinstance(decoded, dict):
        raise ValueError(f"checkpoint field {field!r} must be a dictionary")
    return cast(dict[str, object], decoded)


def _required(dictionary: Mapping[str, _Encoded], key: str) -> _Encoded:
    try:
        return dictionary[key]
    except KeyError as error:
        raise ValueError(f"checkpoint is missing field {key!r}") from error


def _is_shape(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and item > 0 for item in value)
    )


__all__ = ["load_checkpoint", "save_checkpoint"]
