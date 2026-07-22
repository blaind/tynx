"""Opt-in whole-function model capture."""

import warnings
from collections.abc import Callable
from typing import Generic, Optional, TypeVar, Union, cast, overload

from ._tynx import Tensor, _CapturedGraph, _CaptureSession

R = TypeVar("R")


class CompiledFunction(Generic[R]):
    """Callable exact-signature graph cache returned by :func:`tynx.compile`."""

    def __init__(self, function: Callable[..., R], *, fullgraph: bool = False) -> None:
        if not callable(function):
            raise TypeError(f"compile expected a callable, got {type(function).__qualname__}")
        self._function = function
        self._fullgraph = fullgraph
        self._graphs: list[_CapturedGraph] = []
        self._fallback = False
        self._warned = False
        self.compile_count = 0
        self.fallback_count = 0

    @property
    def graph_count(self) -> int:
        """Number of exact-signature native graphs in the cache."""
        return len(self._graphs)

    def clear_cache(self) -> None:
        """Discard captured graphs and retry capture on the next compatible call."""
        self._graphs.clear()
        self._fallback = False
        self._warned = False

    def __call__(self, *args: object, **kwargs: object) -> R:
        if self._fallback:
            self.fallback_count += 1
            return self._function(*args, **kwargs)

        if kwargs or not args or any(not isinstance(argument, Tensor) for argument in args):
            reason = (
                "the initial capture surface accepts one or more positional Tensor arguments "
                "and no keyword arguments"
            )
            if self._fullgraph:
                raise RuntimeError(f"tynx.compile(fullgraph=True) cannot capture {reason}")
            self._disable(reason)
            self.fallback_count += 1
            return self._function(*args, **kwargs)

        tensor_args = tuple(argument for argument in args if isinstance(argument, Tensor))
        for graph in self._graphs:
            if graph.matches(*tensor_args):
                return cast(R, graph(*tensor_args))

        session = _CaptureSession(fullgraph=self._fullgraph)
        traced_args = tuple(session.input(argument) for argument in tensor_args)
        output = self._function(*traced_args)
        if not isinstance(output, Tensor):
            reason = "a return value other than one Tensor"
            if self._fullgraph:
                raise RuntimeError(f"tynx.compile(fullgraph=True) cannot capture {reason}")
            self._disable(reason)
            self.fallback_count += 1
            return output

        captured_graph = session.finish(output)
        released_output = session.release(output)
        if captured_graph is None:
            self._disable("an unsupported tensor operation or trace-disconnected output")
            self.fallback_count += 1
            return cast(R, released_output)
        self._graphs.append(captured_graph)
        self.compile_count += 1
        return cast(R, released_output)

    def _disable(self, reason: str) -> None:
        self._fallback = True
        if not self._warned:
            warnings.warn(
                f"tynx.compile fell back to eager execution for the whole function: {reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True


@overload
def compile(function: Callable[..., R], *, fullgraph: bool = False) -> CompiledFunction[R]: ...


@overload
def compile(
    function: None = None, *, fullgraph: bool = False
) -> Callable[[Callable[..., R]], CompiledFunction[R]]: ...


def compile(
    function: Optional[Callable[..., R]] = None, *, fullgraph: bool = False
) -> Union[CompiledFunction[R], Callable[[Callable[..., R]], CompiledFunction[R]]]:
    """Capture a Tensor-only forward function on first use and replay it in Rust.

    The initial cache specializes on exact tensor signatures. Unsupported behavior falls back for
    the whole function unless ``fullgraph=True`` is requested.
    """

    def decorate(target: Callable[..., R]) -> CompiledFunction[R]:
        return CompiledFunction(target, fullgraph=fullgraph)

    return decorate if function is None else decorate(function)


__all__ = ["CompiledFunction", "compile"]
