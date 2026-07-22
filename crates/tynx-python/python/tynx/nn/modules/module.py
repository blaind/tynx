"""Base class for Tynx-authored layers."""

from ..._tynx import Buffer, Parameter, Tensor


class Module:
    """Small callable layer base with optional state-discovery conveniences."""

    training: bool

    def __init__(self) -> None:
        self.training = True

    def forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)

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
