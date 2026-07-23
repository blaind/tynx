"""Python bindings for the Tynx neural network runtime."""

import builtins as _builtins
from collections.abc import Mapping as _Mapping
from numbers import Integral as _Integral
from numbers import Real as _Real
from os import PathLike as _PathLike
from typing import Any as _Any
from typing import Literal as _Literal
from typing import Optional as _Optional
from typing import Union as _Union
from typing import overload as _overload

from . import distributions, nn, optim
from ._tynx import (
    Buffer,
    Device,
    ImportedModel,
    Parameter,
    Session,
    Tensor,
    TrainabilityReport,
    __version__,
    arange,
    argsort,
    cat,
    chunk,
    empty,
    empty_like,
    full,
    full_like,
    get_default_device,
    index_select,
    is_grad_enabled,
    maximum,
    minimum,
    no_grad,
    nonzero,
    ones,
    ones_like,
    rand,
    rand_like,
    randint,
    randn,
    randn_like,
    sort,
    split,
    stack,
    synchronize,
    topk,
    where,
    zeros,
    zeros_like,
)
from ._tynx import (
    manual_seed as _manual_seed,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .compiler import CompiledFunction, compile


def manual_seed(seed: int) -> None:
    """Seed device sampling and authored-module parameter initialization."""
    _manual_seed(seed)


def _inferred_tensor_dtype(data: _Any) -> _Optional[_Literal["float32", "int64", "bool"]]:
    if isinstance(data, Tensor) or hasattr(data, "__array_interface__"):
        return None
    if isinstance(data, _builtins.bool):
        return "bool"
    if isinstance(data, _Integral):
        return "int64"
    if isinstance(data, _Real):
        return "float32"
    if isinstance(data, range):
        return "int64"
    if isinstance(data, (list, tuple)):
        inferred = {_inferred_tensor_dtype(value) for value in data}
        inferred.discard(None)
        if "float32" in inferred:
            return "float32"
        if "int64" in inferred:
            return "int64"
        if "bool" in inferred:
            return "bool"
    return None


def _coerce_inferred_tensor_data(
    data: _Any, dtype: _Optional[_Literal["float32", "int64", "bool"]]
) -> _Any:
    if dtype == "int64" and isinstance(data, _builtins.bool):
        return int(data)
    if isinstance(data, list):
        return [_coerce_inferred_tensor_data(value, dtype) for value in data]
    if isinstance(data, tuple):
        return tuple(_coerce_inferred_tensor_data(value, dtype) for value in data)
    return data


def tensor(
    data: _Any,
    *,
    dtype: _Optional[_Literal["float32", "int64", "bool"]] = None,
    device: _Optional[Device] = None,
    requires_grad: _builtins.bool = False,
) -> Tensor:
    """Create a Tensor with PyTorch-style bool, integer, and float inference."""
    inferred_dtype = _inferred_tensor_dtype(data) if dtype is None else dtype
    return Tensor(
        _coerce_inferred_tensor_data(data, inferred_dtype),
        dtype=inferred_dtype,
        device=device,
        requires_grad=requires_grad,
    )


@_overload
def load(
    path: _Union[str, _PathLike[str]],
    *,
    trainable: _Literal[True, "auto"],
    simplify: _builtins.bool = True,
    initializer_names: _Optional[_Mapping[str, str]] = None,
    outputs: _Optional[list[str]] = None,
    device: _Optional[Device] = None,
) -> ImportedModel: ...


@_overload
def load(
    path: _Union[str, _PathLike[str]],
    *,
    trainable: _Literal[False] = False,
    simplify: _builtins.bool = True,
    initializer_names: None = None,
    outputs: None = None,
    device: None = None,
) -> Session: ...


def load(
    path: _Union[str, _PathLike[str]],
    *,
    trainable: _Union[_builtins.bool, _Literal["auto"]] = False,
    simplify: _builtins.bool = True,
    initializer_names: _Optional[_Mapping[str, str]] = None,
    outputs: _Optional[list[str]] = None,
    device: _Optional[Device] = None,
) -> _Union[Session, ImportedModel]:
    """Load an inference Session or a callable slot-backed training model."""
    if trainable is False:
        if initializer_names is not None or outputs is not None or device is not None:
            raise ValueError(
                "initializer_names, outputs, and device are only valid for a trainable model"
            )
        return Session(path, simplify=simplify)
    if trainable is not True and trainable != "auto":
        raise ValueError("trainable must be False, True, or 'auto'")
    return ImportedModel(
        path,
        simplify=simplify,
        initializer_names=None if initializer_names is None else dict(initializer_names),
        outputs=outputs,
        device=device,
    )


# These public sentinels remain the canonical strings accepted by the native
# projection; they add PyTorch-style spelling without a second dtype system.
float32: _Literal["float32"] = "float32"
int64: _Literal["int64"] = "int64"
bool: _Literal["bool"] = "bool"


__all__ = [
    "Buffer",
    "CompiledFunction",
    "Device",
    "ImportedModel",
    "Parameter",
    "Session",
    "Tensor",
    "TrainabilityReport",
    "__version__",
    "arange",
    "argsort",
    "bool",
    "cat",
    "chunk",
    "compile",
    "distributions",
    "empty",
    "empty_like",
    "float32",
    "full",
    "full_like",
    "get_default_device",
    "index_select",
    "int64",
    "is_grad_enabled",
    "load",
    "load_checkpoint",
    "manual_seed",
    "maximum",
    "minimum",
    "nn",
    "no_grad",
    "nonzero",
    "ones",
    "ones_like",
    "optim",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "save_checkpoint",
    "sort",
    "split",
    "stack",
    "synchronize",
    "tensor",
    "topk",
    "where",
    "zeros",
    "zeros_like",
]
