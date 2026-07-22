"""PyTorch-style parameter groups over Tynx's native optimizers."""

from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import Callable as _Callable
from typing import Generic as _Generic
from typing import TypeVar as _TypeVar
from typing import Union as _Union
from typing import cast as _cast

from ._tynx import SGD as _NativeSGD
from ._tynx import Adam as _NativeAdam
from ._tynx import AdamW as _NativeAdamW
from ._tynx import Parameter as _Parameter

OptimizerParameter = _Union[_Parameter, tuple[str, _Parameter]]
OptimizerParameterGroup = _Mapping[str, object]
_Parameters = _Iterable[_Union[OptimizerParameter, OptimizerParameterGroup]]
_Native = _TypeVar("_Native", _NativeSGD, _NativeAdam, _NativeAdamW)


def _float(value: object) -> float:
    return float(_cast(_Union[str, int, float], value))


class _GroupedOptimizer(_Generic[_Native]):
    _name: str
    _option_names: tuple[str, ...]

    def __init__(
        self,
        parameters: _Parameters,
        defaults: dict[str, object],
        factory: _Callable[[list[OptimizerParameter], dict[str, object]], _Native],
    ) -> None:
        items = list(parameters)
        if not items:
            raise ValueError(f"{self._name} requires at least one Parameter")
        grouped = isinstance(items[0], _Mapping)
        if any(isinstance(item, _Mapping) != grouped for item in items):
            raise TypeError(
                f"{self._name} parameters cannot mix parameter groups and individual parameters"
            )

        specifications: list[tuple[list[OptimizerParameter], dict[str, object]]] = []
        if grouped:
            for item in items:
                group = _cast(OptimizerParameterGroup, item)
                unknown = set(group).difference((*self._option_names, "params"))
                if unknown:
                    supported = ("params", *self._option_names)
                    raise ValueError(
                        f"{self._name} parameter group has unsupported options {sorted(unknown)!r}; "
                        f"supported options are {list(supported)!r}"
                    )
                if "params" not in group:
                    raise ValueError(f"{self._name} parameter group is missing 'params'")
                try:
                    group_parameters = list(_cast(_Iterable[OptimizerParameter], group["params"]))
                except TypeError as error:
                    raise TypeError(
                        f"{self._name} parameter group 'params' must be an iterable"
                    ) from error
                if not group_parameters:
                    raise ValueError(f"{self._name} parameter groups cannot be empty")
                options = dict(defaults)
                options.update({name: group[name] for name in self._option_names if name in group})
                specifications.append((group_parameters, options))
        else:
            specifications.append((_cast(list[OptimizerParameter], items), dict(defaults)))

        self._reject_duplicates(specifications)
        self._groups: list[dict[str, object]] = []
        self._native: list[_Native] = []
        self._synced_options: list[dict[str, object]] = []
        for group_parameters, options in specifications:
            native = factory(group_parameters, options)
            public = dict(options)
            public["params"] = group_parameters
            self._groups.append(public)
            self._native.append(native)
            self._synced_options.append(dict(options))

    @property
    def param_groups(self) -> list[dict[str, object]]:
        """Mutable group dictionaries; learning-rate changes take effect on the next step."""
        return self._groups

    @property
    def lr(self) -> float:
        return _float(self._groups[0]["lr"])

    @lr.setter
    def lr(self, value: float) -> None:
        for group, native in zip(self._groups, self._native):
            native.lr = value
            group["lr"] = native.lr

    @property
    def learning_rate(self) -> float:
        return self.lr

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.lr = value

    @property
    def parameter_count(self) -> int:
        return sum(native.parameter_count for native in self._native)

    @property
    def state_size(self) -> int:
        return sum(native.state_size for native in self._native)

    def zero_grad(self) -> None:
        for native in self._native:
            native.zero_grad()

    def step(self) -> None:
        self._sync_options()
        for native in self._native:
            native.step()

    def state_dict(self) -> dict[str, object]:
        self._sync_options()
        if len(self._native) == 1:
            return self._native[0].state_dict()
        return {
            "version": 2,
            "optimizer": self._name,
            "param_groups": [native.state_dict() for native in self._native],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if len(self._native) == 1:
            self._native[0].load_state_dict(state_dict)
            self._refresh_options()
            return
        if state_dict.get("version") != 2:
            raise ValueError("grouped optimizer state_dict must have version 2")
        if state_dict.get("optimizer") != self._name:
            raise ValueError(
                f"cannot load {state_dict.get('optimizer')} state into {self._name} optimizer"
            )
        states = state_dict.get("param_groups")
        if not isinstance(states, list) or len(states) != len(self._native):
            raise ValueError(
                f"{self._name} state has {len(states) if isinstance(states, list) else 'invalid'} "
                f"parameter groups; expected {len(self._native)}"
            )
        if not all(isinstance(state, dict) for state in states):
            raise TypeError("grouped optimizer param_groups must contain dictionaries")

        previous = [native.state_dict() for native in self._native]
        try:
            for native, state in zip(self._native, states):
                native.load_state_dict(_cast(dict[str, object], state))
        except BaseException:
            for native, state in zip(self._native, previous):
                native.load_state_dict(state)
            raise
        self._refresh_options()

    def _sync_options(self) -> None:
        for group, native, synced in zip(self._groups, self._native, self._synced_options):
            if group["lr"] != synced["lr"]:
                native.lr = _float(group["lr"])
                group["lr"] = native.lr
                synced["lr"] = native.lr

    def _refresh_options(self) -> None:
        for group, native, synced in zip(self._groups, self._native, self._synced_options):
            for name in self._option_names:
                group[name] = getattr(native, name)
                synced[name] = group[name]

    def _reject_duplicates(
        self, specifications: list[tuple[list[OptimizerParameter], dict[str, object]]]
    ) -> None:
        owners: dict[int, int] = {}
        for group_index, (parameters, _) in enumerate(specifications):
            for item in parameters:
                parameter = item[1] if isinstance(item, tuple) else item
                identity = id(parameter)
                owner = owners.setdefault(identity, group_index)
                if owner != group_index:
                    raise ValueError(
                        f"a Parameter cannot appear in more than one {self._name} parameter group"
                    )


class SGD(_GroupedOptimizer[_NativeSGD]):
    """Stochastic gradient descent with optional PyTorch-style parameter groups."""

    _name = "SGD"
    _option_names = ("lr", "momentum", "dampening", "weight_decay", "nesterov")

    def __init__(
        self,
        parameters: _Parameters,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        defaults: dict[str, object] = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }

        def factory(params: list[OptimizerParameter], options: dict[str, object]) -> _NativeSGD:
            return _NativeSGD(
                params,
                lr=_float(options["lr"]),
                momentum=_float(options["momentum"]),
                dampening=_float(options["dampening"]),
                weight_decay=_float(options["weight_decay"]),
                nesterov=bool(options["nesterov"]),
            )

        super().__init__(parameters, defaults, factory)

    @property
    def momentum(self) -> float:
        return _float(self._groups[0]["momentum"])

    @property
    def dampening(self) -> float:
        return _float(self._groups[0]["dampening"])

    @property
    def weight_decay(self) -> float:
        return _float(self._groups[0]["weight_decay"])

    @property
    def nesterov(self) -> bool:
        return bool(self._groups[0]["nesterov"])

    def __repr__(self) -> str:
        return f"SGD(param_groups={len(self._groups)}, lr={self.lr}, momentum={self.momentum})"

    def _sync_options(self) -> None:
        pending: list[tuple[_NativeSGD, dict[str, object], dict[str, object]]] = []
        for group, native, synced in zip(self._groups, self._native, self._synced_options):
            options = {name: group[name] for name in self._option_names}
            if options == synced:
                continue
            _NativeSGD(
                _cast(list[OptimizerParameter], group["params"]),
                lr=_float(options["lr"]),
                momentum=_float(options["momentum"]),
                dampening=_float(options["dampening"]),
                weight_decay=_float(options["weight_decay"]),
                nesterov=bool(options["nesterov"]),
            )
            pending.append((native, options, synced))
        for native, options, synced in pending:
            native._set_config(
                _float(options["lr"]),
                _float(options["momentum"]),
                _float(options["dampening"]),
                _float(options["weight_decay"]),
                bool(options["nesterov"]),
            )
            synced.clear()
            synced.update(options)


class _AdamBase(_GroupedOptimizer[_Native]):
    @property
    def betas(self) -> tuple[float, float]:
        return _cast(tuple[float, float], self._groups[0]["betas"])

    @property
    def eps(self) -> float:
        return _float(self._groups[0]["eps"])

    @property
    def weight_decay(self) -> float:
        return _float(self._groups[0]["weight_decay"])

    @property
    def amsgrad(self) -> bool:
        return bool(self._groups[0]["amsgrad"])

    def __repr__(self) -> str:
        return (
            f"{self._name}(param_groups={len(self._groups)}, lr={self.lr}, "
            f"betas={self.betas}, eps={self.eps}, weight_decay={self.weight_decay}, "
            f"amsgrad={self.amsgrad})"
        )


class Adam(_AdamBase[_NativeAdam]):
    """Adam with coupled weight decay and optional parameter groups."""

    _name = "Adam"
    _option_names = ("lr", "betas", "eps", "weight_decay", "amsgrad")

    def __init__(
        self,
        parameters: _Parameters,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        defaults: dict[str, object] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }

        def factory(params: list[OptimizerParameter], options: dict[str, object]) -> _NativeAdam:
            return _NativeAdam(
                params,
                lr=_float(options["lr"]),
                betas=_cast(tuple[float, float], options["betas"]),
                eps=_float(options["eps"]),
                weight_decay=_float(options["weight_decay"]),
                amsgrad=bool(options["amsgrad"]),
            )

        super().__init__(parameters, defaults, factory)

    def _sync_options(self) -> None:
        pending: list[tuple[_NativeAdam, dict[str, object], dict[str, object]]] = []
        for group, native, synced in zip(self._groups, self._native, self._synced_options):
            options = {name: group[name] for name in self._option_names}
            if options == synced:
                continue
            _NativeAdam(
                _cast(list[OptimizerParameter], group["params"]),
                lr=_float(options["lr"]),
                betas=_cast(tuple[float, float], options["betas"]),
                eps=_float(options["eps"]),
                weight_decay=_float(options["weight_decay"]),
                amsgrad=bool(options["amsgrad"]),
            )
            pending.append((native, options, synced))
        for native, options, synced in pending:
            native._set_config(
                _float(options["lr"]),
                _cast(tuple[float, float], options["betas"]),
                _float(options["eps"]),
                _float(options["weight_decay"]),
                bool(options["amsgrad"]),
            )
            synced.clear()
            synced.update(options)


class AdamW(_AdamBase[_NativeAdamW]):
    """AdamW with decoupled weight decay and optional parameter groups."""

    _name = "AdamW"
    _option_names = ("lr", "betas", "eps", "weight_decay", "amsgrad")

    def __init__(
        self,
        parameters: _Parameters,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        defaults: dict[str, object] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }

        def factory(params: list[OptimizerParameter], options: dict[str, object]) -> _NativeAdamW:
            return _NativeAdamW(
                params,
                lr=_float(options["lr"]),
                betas=_cast(tuple[float, float], options["betas"]),
                eps=_float(options["eps"]),
                weight_decay=_float(options["weight_decay"]),
                amsgrad=bool(options["amsgrad"]),
            )

        super().__init__(parameters, defaults, factory)

    def _sync_options(self) -> None:
        pending: list[tuple[_NativeAdamW, dict[str, object], dict[str, object]]] = []
        for group, native, synced in zip(self._groups, self._native, self._synced_options):
            options = {name: group[name] for name in self._option_names}
            if options == synced:
                continue
            _NativeAdamW(
                _cast(list[OptimizerParameter], group["params"]),
                lr=_float(options["lr"]),
                betas=_cast(tuple[float, float], options["betas"]),
                eps=_float(options["eps"]),
                weight_decay=_float(options["weight_decay"]),
                amsgrad=bool(options["amsgrad"]),
            )
            pending.append((native, options, synced))
        for native, options, synced in pending:
            native._set_config(
                _float(options["lr"]),
                _cast(tuple[float, float], options["betas"]),
                _float(options["eps"]),
                _float(options["weight_decay"]),
                bool(options["amsgrad"]),
            )
            synced.clear()
            synced.update(options)


__all__ = ["SGD", "Adam", "AdamW", "OptimizerParameter", "OptimizerParameterGroup"]
