"""Python bindings for the Tynx neural network runtime."""

from collections.abc import Mapping
from os import PathLike
from typing import Literal, Optional, Union, overload

from . import nn, optim
from ._tynx import (
    Buffer,
    ImportedModel,
    Parameter,
    Session,
    Tensor,
    __version__,
    is_grad_enabled,
    maximum,
    minimum,
    no_grad,
    where,
)
from .checkpoint import load_checkpoint, save_checkpoint


@overload
def load(
    path: Union[str, PathLike[str]],
    *,
    trainable: Literal[True, "auto"],
    simplify: bool = True,
    initializer_names: Optional[Mapping[str, str]] = None,
) -> ImportedModel: ...


@overload
def load(
    path: Union[str, PathLike[str]],
    *,
    trainable: Literal[False] = False,
    simplify: bool = True,
    initializer_names: None = None,
) -> Session: ...


def load(
    path: Union[str, PathLike[str]],
    *,
    trainable: Union[bool, Literal["auto"]] = False,
    simplify: bool = True,
    initializer_names: Optional[Mapping[str, str]] = None,
) -> Union[Session, ImportedModel]:
    """Load an inference Session or a callable slot-backed training model."""
    if trainable is False:
        if initializer_names is not None:
            raise ValueError("initializer_names is only valid for a trainable model")
        return Session(path, simplify=simplify)
    if trainable is not True and trainable != "auto":
        raise ValueError("trainable must be False, True, or 'auto'")
    return ImportedModel(
        path,
        simplify=simplify,
        initializer_names=None if initializer_names is None else dict(initializer_names),
    )


__all__ = [
    "Buffer",
    "ImportedModel",
    "Parameter",
    "Session",
    "Tensor",
    "__version__",
    "is_grad_enabled",
    "load",
    "load_checkpoint",
    "maximum",
    "minimum",
    "nn",
    "no_grad",
    "optim",
    "save_checkpoint",
    "where",
]
