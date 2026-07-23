"""Base class for Tynx-authored layers."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union

from ..._tynx import Buffer, Parameter, Tensor

if TYPE_CHECKING:
    from ..state import LoadStateResult


class Module:
    """Small callable layer base with optional state-discovery conveniences."""

    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_registered_buffer_names", {})
        self.training = True

    def __setattr__(self, name: str, value: object) -> None:
        try:
            registered = object.__getattribute__(self, "_registered_buffer_names")
        except AttributeError:
            registered = {}
        if name in registered:
            value = _coerce_buffer(name, value)
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        from ...compiler import _record_module_call

        _record_module_call(self)
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        from ..state import get_parameters

        return get_parameters(self)

    def named_parameters(self) -> list[tuple[str, Parameter]]:
        from ..state import named_parameters

        return named_parameters(self)

    def buffers(self) -> list[Buffer]:
        from ..state import get_buffers

        return get_buffers(self)

    def named_buffers(self) -> list[tuple[str, Buffer]]:
        from ..state import named_buffers

        return named_buffers(self)

    def register_buffer(
        self,
        name: str,
        tensor: Optional[Union[Tensor, Buffer]],
        persistent: bool = True,
    ) -> None:
        """Register persistent non-parameter state under an attribute name."""
        if not isinstance(name, str):
            raise TypeError(f"buffer name must be a string, got {type(name).__qualname__}")
        if not name or "." in name:
            raise ValueError("buffer name must be non-empty and cannot contain '.'")
        if type(persistent) is not bool:
            raise TypeError(f"persistent must be a bool, got {type(persistent).__qualname__}")
        if not persistent:
            raise NotImplementedError("non-persistent buffers are not supported")
        try:
            registered = object.__getattribute__(self, "_registered_buffer_names")
        except AttributeError:
            registered = {}
            object.__setattr__(self, "_registered_buffer_names", registered)
        if name not in registered and (name in self.__dict__ or hasattr(type(self), name)):
            raise KeyError(f"attribute {name!r} already exists")
        buffer = _coerce_buffer(name, tensor)
        registered[name] = None
        object.__setattr__(self, name, buffer)

    def zero_grad(self) -> None:
        """Clear gradients for every recursively discovered parameter."""
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> dict[str, Tensor]:
        from ..state import get_state_dict

        return get_state_dict(self)

    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> "LoadStateResult":
        from ..state import load_state_dict

        return load_state_dict(self, state_dict, strict)

    def train(self, mode: bool = True) -> "Module":
        from ..state import train

        train(self, mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        details = self.extra_repr()
        return f"{type(self).__name__}({details})"


Layer = Module


def _coerce_buffer(name: str, value: object) -> Optional[Buffer]:
    if value is None:
        return None
    if isinstance(value, Buffer):
        return value
    if isinstance(value, Tensor):
        return Buffer(value, name=name)
    raise TypeError(
        f"buffer value must be a Tensor, Buffer, or None, got {type(value).__qualname__}"
    )


__all__ = ["Layer", "Module"]
