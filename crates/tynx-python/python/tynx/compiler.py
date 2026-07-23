"""Opt-in whole-function model capture."""

import warnings as _warnings
from collections import OrderedDict as _OrderedDict
from collections.abc import Callable as _Callable
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
from contextvars import ContextVar as _ContextVar
from inspect import BoundArguments as _BoundArguments
from inspect import Parameter as _Parameter
from inspect import Signature as _Signature
from inspect import signature as _signature
from types import MethodType as _MethodType
from typing import Any as _Any
from typing import Generic as _Generic
from typing import Optional as _Optional
from typing import TypeVar as _TypeVar
from typing import Union as _Union
from typing import cast as _cast
from typing import overload as _overload
from weakref import ReferenceType as _ReferenceType
from weakref import WeakKeyDictionary as _WeakKeyDictionary
from weakref import ref as _ref

from ._tynx import Tensor, _CapturedGraph, _CaptureSession

R = _TypeVar("R")
_ModuleGuard = tuple[tuple[_ReferenceType[object], bool], ...]
_ACTIVE_MODULE_CALLS: _ContextVar[_Optional[dict[int, object]]] = _ContextVar(
    "tynx_compile_module_calls", default=None
)


def _record_module_call(module: object) -> None:
    """Record a module whose mode can affect the graph currently being captured."""
    active = _ACTIVE_MODULE_CALLS.get()
    if active is not None:
        active.setdefault(id(module), module)


class CompiledFunction(_Generic[R]):
    """Callable exact-signature graph cache returned by :func:`tynx.compile`."""

    def __init__(
        self,
        function: _Callable[..., R],
        *,
        fullgraph: bool = False,
        static_argnames: _Iterable[str] = (),
    ) -> None:
        if not callable(function):
            raise TypeError(f"compile expected a callable, got {type(function).__qualname__}")
        self._function = function
        self._signature = _signature(function)
        self._fullgraph = fullgraph
        self._static_argnames = _validate_static_argnames(self._signature, static_argnames)
        self._graphs: list[
            tuple[
                tuple[tuple[str, type[object], object], ...],
                _CapturedGraph,
                object,
                _ModuleGuard,
            ]
        ] = []
        self._fallback = False
        self._warned = False
        self.compile_count = 0
        self.invalidation_count = 0
        self.fallback_count = 0
        self.replay_count = 0
        self.last_fallback_reason: _Optional[str] = None
        self._bound_instances: _WeakKeyDictionary[object, CompiledFunction[R]] = (
            _WeakKeyDictionary()
        )

    @_overload
    def __get__(self, instance: None, owner: type[object]) -> "CompiledFunction[R]": ...

    @_overload
    def __get__(self, instance: object, owner: type[object]) -> "CompiledFunction[R]": ...

    def __get__(self, instance: _Optional[object], owner: type[object]) -> "CompiledFunction[R]":
        """Bind compiled functions used as class attributes like normal methods."""
        if instance is None:
            return self
        cached = self._bound_instances.get(instance)
        if cached is not None:
            return cached
        receiver = next(iter(self._signature.parameters), None)
        static_argnames = self._static_argnames.difference(
            {receiver} if receiver in ("self", "cls") else set()
        )
        bound = CompiledFunction(
            _cast(_Callable[..., R], _MethodType(self._function, instance)),
            fullgraph=self._fullgraph,
            static_argnames=static_argnames,
        )
        self._bound_instances[instance] = bound
        return bound

    @property
    def graph_count(self) -> int:
        """Number of exact-signature native graphs in the cache."""
        return len(self._graphs)

    @property
    def node_counts(self) -> tuple[int, ...]:
        """Recorded IR node count for each cached graph."""
        return tuple(graph.node_count for _, graph, _, _ in self._graphs)

    def clear_cache(self) -> None:
        """Discard captured graphs and retry capture on the next compatible call."""
        self._graphs.clear()
        for bound in self._bound_instances.values():
            bound.clear_cache()
        self._fallback = False
        self._warned = False
        self.last_fallback_reason = None

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
        valid_graphs = [
            entry
            for entry in self._graphs
            if entry[1].structure_valid and _module_guard_alive(entry[3])
        ]
        self.invalidation_count += len(self._graphs) - len(valid_graphs)
        self._graphs = valid_graphs
        for cached_static_key, graph, output_spec, module_guard in self._graphs:
            if (
                cached_static_key == static_key
                and _module_guard_matches(module_guard)
                and graph.matches(*tensor_args)
            ):
                self.replay_count += 1
                return _cast(R, _restore_output(iter(graph(*tensor_args)), output_spec))

        session = _CaptureSession(fullgraph=self._fullgraph)
        module_calls: dict[int, object] = {}
        module_token = _ACTIVE_MODULE_CALLS.set(module_calls)
        try:
            traced = iter(session.input(argument) for argument in tensor_args)
            traced_bound = _replace_tensor_arguments(bound, traced)
            output = self._function(*traced_bound.args, **traced_bound.kwargs)
        except BaseException:
            session.abort()
            raise
        finally:
            _ACTIVE_MODULE_CALLS.reset(module_token)
        flattened = _flatten_output(output)
        if isinstance(flattened, str):
            reason = flattened
            session.abort()
            if self._fullgraph:
                raise RuntimeError(f"tynx.compile(fullgraph=True) cannot capture {reason}")
            self._disable(reason)
            self.fallback_count += 1
            return output
        output_tensors, output_spec = flattened

        captured_graph = session.finish(tuple(output_tensors))
        released_output = _restore_output(
            iter(session.release(tensor) for tensor in output_tensors), output_spec
        )
        if captured_graph is None:
            self._disable("an unsupported tensor operation or trace-disconnected output")
            self.fallback_count += 1
            return _cast(R, released_output)
        module_guard = tuple(
            (_ref(module), bool(_cast(_Any, module).training)) for module in module_calls.values()
        )
        self._graphs.append((static_key, captured_graph, output_spec, module_guard))
        self.compile_count += 1
        return _cast(R, released_output)

    def _prepare_call(
        self, bound: _BoundArguments
    ) -> _Union[tuple[tuple[Tensor, ...], tuple[tuple[str, type[object], object], ...]], str]:
        tensors: list[Tensor] = []
        static_key: list[tuple[str, type[object], object]] = []
        for name, value in bound.arguments.items():
            parameter = self._signature.parameters[name]
            if parameter.kind in (_Parameter.VAR_POSITIONAL, _Parameter.VAR_KEYWORD):
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
        self.last_fallback_reason = reason
        if not self._warned:
            _warnings.warn(
                f"tynx.compile fell back to eager execution for the whole function: {reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned = True


def _replace_tensor_arguments(bound: _BoundArguments, traced: _Iterator[Tensor]) -> _BoundArguments:
    arguments: _OrderedDict[str, _Any] = _OrderedDict(
        (name, next(traced) if isinstance(value, Tensor) else value)
        for name, value in bound.arguments.items()
    )
    return _BoundArguments(bound.signature, arguments)


def _flatten_output(output: object) -> _Union[tuple[list[Tensor], object], str]:
    tensors: list[Tensor] = []

    def visit(value: object) -> object:
        if isinstance(value, Tensor):
            index = len(tensors)
            tensors.append(value)
            return ("tensor", index)
        if isinstance(value, tuple):
            return ("tuple", tuple(visit(item) for item in value))
        if isinstance(value, list):
            return ("list", tuple(visit(item) for item in value))
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            return ("dict", tuple((key, visit(item)) for key, item in value.items()))
        raise TypeError(type(value).__qualname__)

    try:
        spec = visit(output)
    except TypeError as error:
        return f"a return value containing unsupported type {error.args[0]}"
    if not tensors:
        return "a return value without any Tensor outputs"
    return tensors, spec


def _restore_output(tensors: _Iterator[Tensor], spec: object) -> object:
    kind, contents = _cast(tuple[str, object], spec)
    if kind == "tensor":
        return next(tensors)
    if kind == "tuple":
        return tuple(_restore_output(tensors, item) for item in _cast(tuple[object, ...], contents))
    if kind == "list":
        return [_restore_output(tensors, item) for item in _cast(tuple[object, ...], contents)]
    if kind == "dict":
        return {
            key: _restore_output(tensors, item)
            for key, item in _cast(tuple[tuple[str, object], ...], contents)
        }
    raise RuntimeError(f"unknown captured output kind {kind!r}")


def _module_guard_alive(guard: _ModuleGuard) -> bool:
    return all(reference() is not None for reference, _ in guard)


def _module_guard_matches(guard: _ModuleGuard) -> bool:
    for reference, training in guard:
        module = reference()
        if module is None or getattr(module, "training", None) is not training:
            return False
    return True


def _validate_static_argnames(signature: _Signature, names: _Iterable[str]) -> frozenset[str]:
    if isinstance(names, str):
        names = (names,)
    normalized = tuple(names)
    if any(not isinstance(name, str) for name in normalized):
        raise TypeError("static_argnames must contain only strings")
    unknown = sorted(set(normalized).difference(signature.parameters))
    if unknown:
        raise ValueError(f"unknown static_argnames: {unknown}")
    return frozenset(normalized)


@_overload
def compile(
    function: _Callable[..., R],
    *,
    fullgraph: bool = False,
    static_argnames: _Iterable[str] = (),
) -> CompiledFunction[R]: ...


@_overload
def compile(
    function: None = None,
    *,
    fullgraph: bool = False,
    static_argnames: _Iterable[str] = (),
) -> _Callable[[_Callable[..., R]], CompiledFunction[R]]: ...


def compile(
    function: _Optional[_Callable[..., R]] = None,
    *,
    fullgraph: bool = False,
    static_argnames: _Iterable[str] = (),
) -> _Union[CompiledFunction[R], _Callable[[_Callable[..., R]], CompiledFunction[R]]]:
    """Capture a Tensor-only forward function on first use and replay it in Rust.

    The initial cache specializes on exact tensor signatures. Unsupported behavior falls back for
    the whole function unless ``fullgraph=True`` is requested.
    """

    def decorate(target: _Callable[..., R]) -> CompiledFunction[R]:
        return CompiledFunction(
            target,
            fullgraph=fullgraph,
            static_argnames=static_argnames,
        )

    return decorate if function is None else decorate(function)


__all__ = ["CompiledFunction", "compile"]
