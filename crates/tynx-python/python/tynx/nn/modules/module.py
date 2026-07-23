"""Base class for Tynx-authored layers."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ..._tynx import Buffer, Parameter, Tensor

if TYPE_CHECKING:
    from ..state import LoadStateResult


class Module:
    """Small callable layer base with optional state-discovery conveniences."""

    training: bool

    def __init__(self) -> None:
        self.training = True

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

__all__ = ["Layer", "Module"]
