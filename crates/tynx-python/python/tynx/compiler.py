"""Opt-in whole-function model capture."""

import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from inspect import BoundArguments, Parameter, Signature, signature
from typing import Any, Generic, Optional, TypeVar, Union, cast, overload

from ._tynx import Tensor, _CapturedGraph, _CaptureSession

R = TypeVar("R")


class CompiledFunction(Generic[R]):
    """Callable exact-signature graph cache returned by :func:`tynx.compile`."""

    def __init__(
        self,
        function: Callable[..., R],
        *,
        fullgraph: bool = False,
        static_argnames: Iterable[str] = (),
    ) -> None:
        if not callable(function):
            raise TypeError(f"compile expected a callable, got {type(function).__qualname__}")
        self._function = function
        self._signature = signature(function)
        self._fullgraph = fullgraph
        self._static_argnames = _validate_static_argnames(self._signature, static_argnames)
        self._graphs: list[tuple[tuple[tuple[str, type[object], object], ...], _CapturedGraph]] = []
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

        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        prepared = self._prepare_call(bound)
        if isinstance(prepared, str):
            reason = prepared
            if self._fullgraph:
                raise RuntimeError(f"tynx.compile(fullgraph=True) cannot capture {reason}")
            self._disable(reason)
            self.fallback_count += 1
            return self._function(*args, **kwargs)

        tensor_args, static_key = prepared
        for cached_static_key, graph in self._graphs:
            if cached_static_key == static_key and graph.matches(*tensor_args):
                return cast(R, graph(*tensor_args))

        session = _CaptureSession(fullgraph=self._fullgraph)
        traced = iter(session.input(argument) for argument in tensor_args)
        traced_bound = _replace_tensor_arguments(bound, traced)
        output = self._function(*traced_bound.args, **traced_bound.kwargs)
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
        self._graphs.append((static_key, captured_graph))
        self.compile_count += 1
        return cast(R, released_output)

    def _prepare_call(
        self, bound: BoundArguments
    ) -> Union[tuple[tuple[Tensor, ...], tuple[tuple[str, type[object], object], ...]], str]:
        tensors: list[Tensor] = []
        static_key: list[tuple[str, type[object], object]] = []
        for name, value in bound.arguments.items():
            parameter = self._signature.parameters[name]
            if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                return "variadic Python arguments"
            if name in self._static_argnames:
                if isinstance(value, Tensor):
                    return f"Tensor argument {name!r} declared static"
                try:
                    hash((type(value), value))
                except TypeError:
                    return f"unhashable static argument {name!r} of type {type(value).__qualname__}"
                static_key.append((name, type(value), value))
            elif isinstance(value, Tensor):
                tensors.append(value)
            else:
                return f"non-Tensor argument {name!r}; declare it with static_argnames"
        if not tensors:
            return "a function call without Tensor inputs"
        return tuple(tensors), tuple(static_key)

    def _disable(self, reason: str) -> None:
        self._fallback = True
        if not self._warned:
            warnings.warn(
                f"tynx.compile fell back to eager execution for the whole function: {reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True


def _replace_tensor_arguments(bound: BoundArguments, traced: Iterator[Tensor]) -> BoundArguments:
    arguments: OrderedDict[str, Any] = OrderedDict(
        (name, next(traced) if isinstance(value, Tensor) else value)
        for name, value in bound.arguments.items()
    )
    return BoundArguments(bound.signature, arguments)


def _validate_static_argnames(signature: Signature, names: Iterable[str]) -> frozenset[str]:
    if isinstance(names, str):
        names = (names,)
    normalized = tuple(names)
    if any(not isinstance(name, str) for name in normalized):
        raise TypeError("static_argnames must contain only strings")
    unknown = sorted(set(normalized).difference(signature.parameters))
    if unknown:
        raise ValueError(f"unknown static_argnames: {unknown}")
    return frozenset(normalized)


@overload
def compile(
    function: Callable[..., R],
    *,
    fullgraph: bool = False,
    static_argnames: Iterable[str] = (),
) -> CompiledFunction[R]: ...


@overload
def compile(
    function: None = None,
    *,
    fullgraph: bool = False,
    static_argnames: Iterable[str] = (),
) -> Callable[[Callable[..., R]], CompiledFunction[R]]: ...


def compile(
    function: Optional[Callable[..., R]] = None,
    *,
    fullgraph: bool = False,
    static_argnames: Iterable[str] = (),
) -> Union[CompiledFunction[R], Callable[[Callable[..., R]], CompiledFunction[R]]]:
    """Capture a Tensor-only forward function on first use and replay it in Rust.

    The initial cache specializes on exact tensor signatures. Unsupported behavior falls back for
    the whole function unless ``fullgraph=True`` is requested.
    """

    def decorate(target: Callable[..., R]) -> CompiledFunction[R]:
        return CompiledFunction(
            target,
            fullgraph=fullgraph,
            static_argnames=static_argnames,
        )

    return decorate if function is None else decorate(function)


__all__ = ["CompiledFunction", "compile"]
