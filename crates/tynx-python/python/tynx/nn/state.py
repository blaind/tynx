"""Narrow, deterministic state discovery for ordinary Python model objects."""

from types import FunctionType, MethodType, ModuleType
from typing import cast

from .._tynx import Parameter

_Path = tuple[str, ...]
_RelativeParameter = tuple[_Path, Parameter]


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


def _parameter_names(
    obj: object,
) -> tuple[list[tuple[str, Parameter, int]], dict[int, tuple[str, ...]]]:
    leaves = _Walker().walk(obj)
    paths_by_identity: dict[int, set[str]] = {}
    parameters: dict[int, Parameter] = {}
    for path, parameter in leaves:
        identity = id(parameter)
        parameters[identity] = parameter
        paths_by_identity.setdefault(identity, set()).add(_format_path(path, parameter))

    canonical: list[tuple[str, Parameter, int]] = []
    aliases: dict[int, tuple[str, ...]] = {}
    for identity, paths in paths_by_identity.items():
        ordered = sorted(paths)
        canonical.append((ordered[0], parameters[identity], identity))
        aliases[identity] = tuple(ordered[1:])
    canonical.sort(key=lambda entry: entry[0])
    return canonical, aliases


class _Walker:
    def __init__(self) -> None:
        self._active: set[int] = set()
        self._memo: dict[int, list[_RelativeParameter]] = {}

    def walk(self, value: object) -> list[_RelativeParameter]:
        if isinstance(value, Parameter):
            return [((), value)]
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

    def _walk_container_or_object(self, value: object) -> list[_RelativeParameter]:
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

    def _walk_items(self, items: list[tuple[str, object]]) -> list[_RelativeParameter]:
        discovered: list[_RelativeParameter] = []
        for segment, child in items:
            for relative, parameter in self.walk(child):
                discovered.append(((segment, *relative), parameter))
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


def _format_path(path: _Path, parameter: Parameter) -> str:
    if path:
        return ".".join(path)
    if parameter.name:
        return parameter.name
    return "<root>"


__all__ = ["get_parameter_aliases", "get_parameters", "named_parameters"]
