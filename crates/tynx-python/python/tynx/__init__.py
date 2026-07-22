"""Python bindings for the Tynx neural network runtime."""

from collections.abc import Mapping as _Mapping
from os import PathLike as _PathLike
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
    cat,
    chunk,
    empty,
    empty_like,
    full,
    full_like,
    get_default_device,
    is_grad_enabled,
    maximum,
    minimum,
    no_grad,
    ones,
    ones_like,
    rand,
    rand_like,
    randint,
    randn,
    randn_like,
    split,
    stack,
    synchronize,
    where,
    zeros,
    zeros_like,
)
from ._tynx import (
    manual_seed as _manual_seed,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .compiler import CompiledFunction, compile
from .nn._random import seed as _seed_module_initialization


def manual_seed(seed: int) -> None:
    """Seed device sampling and authored-module parameter initialization."""
    _manual_seed(seed)
    _seed_module_initialization(seed)


@_overload
def load(
    path: _Union[str, _PathLike[str]],
    *,
    trainable: _Literal[True, "auto"],
    simplify: bool = True,
    initializer_names: _Optional[_Mapping[str, str]] = None,
    outputs: _Optional[list[str]] = None,
) -> ImportedModel: ...


@_overload
def load(
    path: _Union[str, _PathLike[str]],
    *,
    trainable: _Literal[False] = False,
    simplify: bool = True,
    initializer_names: None = None,
    outputs: None = None,
) -> Session: ...


def load(
    path: _Union[str, _PathLike[str]],
    *,
    trainable: _Union[bool, _Literal["auto"]] = False,
    simplify: bool = True,
    initializer_names: _Optional[_Mapping[str, str]] = None,
    outputs: _Optional[list[str]] = None,
) -> _Union[Session, ImportedModel]:
    """Load an inference Session or a callable slot-backed training model."""
    if trainable is False:
        if initializer_names is not None or outputs is not None:
            raise ValueError("initializer_names and outputs are only valid for a trainable model")
        return Session(path, simplify=simplify)
    if trainable is not True and trainable != "auto":
        raise ValueError("trainable must be False, True, or 'auto'")
    return ImportedModel(
        path,
        simplify=simplify,
        initializer_names=None if initializer_names is None else dict(initializer_names),
        outputs=outputs,
    )


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
    "cat",
    "chunk",
    "compile",
    "distributions",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "get_default_device",
    "is_grad_enabled",
    "load",
    "load_checkpoint",
    "manual_seed",
    "maximum",
    "minimum",
    "nn",
    "no_grad",
    "ones",
    "ones_like",
    "optim",
    "rand",
    "rand_like",
    "randint",
    "randn",
    "randn_like",
    "save_checkpoint",
    "split",
    "stack",
    "synchronize",
    "where",
    "zeros",
    "zeros_like",
]
