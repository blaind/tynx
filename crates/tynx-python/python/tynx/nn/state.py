"""Narrow, deterministic state discovery for ordinary Python model objects."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import FunctionType, MethodType, ModuleType
from typing import Generic, TypeVar, Union, cast

from .._tynx import Buffer, ImportedModel, Parameter, Tensor

_Path = tuple[str, ...]
_State = TypeVar("_State", Parameter, Buffer)
_StateValue = Union[Parameter, Buffer]
_RelativeState = tuple[_Path, _State]


@dataclass(frozen=True)
class LoadStateResult:
    """Missing and unexpected names returned by a non-strict state load."""

    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def get_parameters(obj: object) -> list[Parameter]:
    """Return each reachable parameter once in stable canonical-name order."""
    return [parameter for _, parameter in named_parameters(obj)]


def named_parameters(obj: object) -> list[tuple[str, Parameter]]:
    """Return canonical names and unique parameters discovered without dynamic attribute access."""
    canonical, _ = _parameter_names(obj)
    return [(name, parameter) for name, parameter, _ in canonical]


def get_parameter_aliases(obj: object) -> dict[str, tuple[str, ...]]:
    """Map each canonical parameter name to its other reachable state paths."""
    canonical, aliases = _parameter_names(obj)
    return {name: aliases.get(id(parameter), ()) for name, parameter, _ in canonical}


def get_buffers(obj: object) -> list[Buffer]:
    """Return each reachable buffer once in stable canonical-name order."""
    return [buffer for _, buffer in named_buffers(obj)]


def named_buffers(obj: object) -> list[tuple[str, Buffer]]:
    """Return canonical names and unique buffers discovered without dynamic attribute access."""
    canonical, _ = _state_names(obj, Buffer)
    return [(name, buffer) for name, buffer, _ in canonical]


def get_buffer_aliases(obj: object) -> dict[str, tuple[str, ...]]:
    """Map each canonical buffer name to its other reachable state paths."""
    canonical, aliases = _state_names(obj, Buffer)
    return {name: aliases.get(id(buffer), ()) for name, buffer, _ in canonical}


def get_state_dict(obj: object) -> dict[str, Tensor]:
    """Return detached snapshots of all canonical parameters and buffers."""
    return {name: state.detach() for name, state in _named_state(obj).items()}


def load_state_dict(
    obj: object,
    state_dict: Mapping[str, Tensor],
    strict: bool = True,
) -> LoadStateResult:
    """Load compatible snapshots into existing state identities by canonical name."""
    current, prepared, result = _prepare_state_dict_load(obj, state_dict, strict)
    _apply_prepared_state(current, prepared)
    return result


def _prepare_state_dict_load(
    obj: object,
    state_dict: Mapping[str, Tensor],
    strict: bool,
) -> tuple[dict[str, _StateValue], dict[str, Tensor], LoadStateResult]:
    """Validate a state load and detach every source before publishing any value."""
    if type(strict) is not bool:
        raise TypeError(f"strict must be a bool, got {type(strict).__qualname__}")
    current = _named_state(obj)
    supplied: dict[str, Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str):
            raise TypeError(f"state_dict keys must be strings, got {type(name).__qualname__}")
        if not isinstance(value, Tensor):
            raise TypeError(
                f"state_dict value for {name!r} must be a Tensor, got {type(value).__qualname__}"
            )
        supplied[name] = value

    missing = tuple(sorted(set(current) - set(supplied)))
    unexpected = tuple(sorted(set(supplied) - set(current)))
    result = LoadStateResult(missing_keys=missing, unexpected_keys=unexpected)
    if strict and (missing or unexpected):
        raise ValueError(
            f"state_dict key mismatch: missing={list(missing)!r}, unexpected={list(unexpected)!r}"
        )

    prepared: dict[str, Tensor] = {}
    for name in sorted(set(current) & set(supplied)):
        target = current[name]
        source = supplied[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                f"state_dict value for {name!r} has shape {source.shape} and dtype {source.dtype}; "
                f"expected shape {target.shape} and dtype {target.dtype}"
            )
        prepared[name] = source.detach()
    return current, prepared, result


def _apply_prepared_state(
    current: Mapping[str, _StateValue], prepared: Mapping[str, Tensor]
) -> None:
    """Publish a state load that has already passed `_prepare_state_dict_load`."""
    for name, source in prepared.items():
        current[name].copy_(source)


def train(obj: object, mode: bool = True) -> object:
    """Set every reachable Tynx module to training or evaluation mode."""
    if type(mode) is not bool:
        raise TypeError(f"training mode must be a bool, got {type(mode).__qualname__}")
    from .modules.module import Module

    for value in _walk_objects(obj):
        if isinstance(value, Module):
            object.__setattr__(value, "training", mode)
    return obj


def eval(obj: object) -> object:
    """Set every reachable Tynx module to evaluation mode."""
    return train(obj, False)


def _parameter_names(
    obj: object,
) -> tuple[list[tuple[str, Parameter, int]], dict[int, tuple[str, ...]]]:
    return _state_names(obj, Parameter)


def _named_state(obj: object) -> dict[str, _StateValue]:
    if isinstance(obj, ImportedModel):
        imported_state: dict[str, _StateValue] = dict(obj.named_parameters())
        for name, buffer in obj.named_buffers():
            if name in imported_state:
                raise ValueError(f"parameter and buffer share the canonical state name {name!r}")
            imported_state[name] = buffer
        return dict(sorted(imported_state.items()))

    combined: dict[str, _StateValue] = {}
    for name, state in named_parameters(obj):
        if name in combined:
            raise ValueError(f"parameter and buffer share the canonical state name {name!r}")
        combined[name] = state
    for name, buffer in named_buffers(obj):
        if name in combined:
            raise ValueError(f"parameter and buffer share the canonical state name {name!r}")
        combined[name] = buffer
    return dict(sorted(combined.items()))


def _state_names(
    obj: object, state_type: type[_State]
) -> tuple[list[tuple[str, _State, int]], dict[int, tuple[str, ...]]]:
    leaves = _Walker(state_type).walk(obj)
    paths_by_identity: dict[int, set[str]] = {}
    states: dict[int, _State] = {}
    for path, state in leaves:
        identity = id(state)
        states[identity] = state
        paths_by_identity.setdefault(identity, set()).add(_format_path(path, state))

    canonical: list[tuple[str, _State, int]] = []
    aliases: dict[int, tuple[str, ...]] = {}
    for identity, paths in paths_by_identity.items():
        ordered = sorted(paths)
        canonical.append((ordered[0], states[identity], identity))
        aliases[identity] = tuple(ordered[1:])
    canonical.sort(key=lambda entry: entry[0])
    return canonical, aliases


class _Walker(Generic[_State]):
    def __init__(self, state_type: type[_State]) -> None:
        self._state_type: type[_State] = state_type
        self._active: set[int] = set()
        self._memo: dict[int, list[_RelativeState[_State]]] = {}

    def walk(self, value: object) -> list[_RelativeState[_State]]:
        if isinstance(value, self._state_type):
            return [((), value)]
        if isinstance(value, (Parameter, Buffer)):
            return []
        if _is_terminal(value):
            return []

        identity = id(value)
        cached = self._memo.get(identity)
        if cached is not None:
            return cached
        if identity in self._active:
            return []

        self._active.add(identity)
        try:
            discovered = self._walk_container_or_object(value)
        finally:
            self._active.remove(identity)
        self._memo[identity] = discovered
        return discovered

    def _walk_container_or_object(self, value: object) -> list[_RelativeState[_State]]:
        if type(value) is list:
            items = [(str(index), item) for index, item in enumerate(cast(list[object], value))]
            return self._walk_items(items)
        if type(value) is tuple:
            items = [
                (str(index), item) for index, item in enumerate(cast(tuple[object, ...], value))
            ]
            return self._walk_items(items)
        if type(value) is dict:
            mapping = cast(dict[object, object], value)
            keys = [_validate_dictionary_key(key) for key in mapping]
            return self._walk_items([(key, mapping[key]) for key in sorted(keys)])

        if _declares_slots(value):
            raise TypeError(
                f"cannot discover state in {type(value).__qualname__}: __slots__ objects are "
                "unsupported; store Tynx state in __dict__ or a supported container"
            )
        try:
            namespace = object.__getattribute__(value, "__dict__")
        except AttributeError:
            return []
        if type(namespace) is not dict:
            raise TypeError(
                f"cannot discover state in {type(value).__qualname__}: __dict__ is not a plain dict"
            )
        attributes = cast(dict[object, object], namespace)
        names = [_validate_attribute_name(name) for name in attributes]
        return self._walk_items([(name, attributes[name]) for name in sorted(names)])

    def _walk_items(self, items: list[tuple[str, object]]) -> list[_RelativeState[_State]]:
        discovered: list[_RelativeState[_State]] = []
        for segment, child in items:
            for child_state in self.walk(child):
                relative: _Path = child_state[0]
                state: _State = child_state[1]
                discovered.append(((segment, *relative), state))
        return discovered


def _is_terminal(value: object) -> bool:
    return value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            bytearray,
            FunctionType,
            MethodType,
            ModuleType,
            type,
        ),
    )


def _declares_slots(value: object) -> bool:
    return any(
        "__slots__" in vars(class_) for class_ in type(value).__mro__ if class_ is not object
    )


def _validate_dictionary_key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError(f"state dictionaries require string keys, got {type(key).__qualname__}")
    return _validate_path_segment(key, "dictionary key")


def _validate_attribute_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError(f"object __dict__ keys must be strings, got {type(name).__qualname__}")
    return _validate_path_segment(name, "attribute name")


def _validate_path_segment(segment: str, kind: str) -> str:
    if not segment or "." in segment:
        raise ValueError(f"state {kind} must be non-empty and cannot contain '.', got {segment!r}")
    return segment


def _format_path(path: _Path, state: _StateValue) -> str:
    if path:
        return ".".join(path)
    if state.name:
        return state.name
    return "<root>"


def _walk_objects(root: object) -> list[object]:
    discovered: list[object] = []
    pending = [root]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        discovered.append(value)
        if isinstance(value, (Parameter, Buffer)) or _is_terminal(value):
            continue
        if type(value) is list:
            pending.extend(reversed(cast(list[object], value)))
            continue
        if type(value) is tuple:
            pending.extend(reversed(cast(tuple[object, ...], value)))
            continue
        if type(value) is dict:
            mapping = cast(dict[object, object], value)
            keys = [_validate_dictionary_key(key) for key in mapping]
            pending.extend(mapping[key] for key in reversed(sorted(keys)))
            continue
        if _declares_slots(value):
            raise TypeError(
                f"cannot propagate mode through {type(value).__qualname__}: __slots__ objects are "
                "unsupported; store Tynx layers in __dict__ or a supported container"
            )
        try:
            namespace = object.__getattribute__(value, "__dict__")
        except AttributeError:
            continue
        if type(namespace) is not dict:
            raise TypeError(
                f"cannot propagate mode through {type(value).__qualname__}: "
                "__dict__ is not a plain dict"
            )
        attributes = cast(dict[object, object], namespace)
        names = [_validate_attribute_name(name) for name in attributes]
        pending.extend(attributes[name] for name in reversed(sorted(names)))
    return discovered


__all__ = [
    "LoadStateResult",
    "eval",
    "get_buffer_aliases",
    "get_buffers",
    "get_parameter_aliases",
    "get_parameters",
    "get_state_dict",
    "load_state_dict",
    "named_buffers",
    "named_parameters",
    "train",
]
