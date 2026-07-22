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
    get_default_device,
    is_grad_enabled,
    manual_seed,
    maximum,
    minimum,
    no_grad,
    synchronize,
    where,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .compiler import CompiledFunction, compile


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
    "compile",
    "distributions",
    "get_default_device",
    "is_grad_enabled",
    "load",
    "load_checkpoint",
    "manual_seed",
    "maximum",
    "minimum",
    "nn",
    "no_grad",
    "optim",
    "save_checkpoint",
    "synchronize",
    "where",
]
