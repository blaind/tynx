"""Learning-rate schedulers for Tynx optimizers."""

import math as _math
from typing import Optional as _Optional
from typing import Protocol as _Protocol
from typing import cast as _cast


class _Optimizer(_Protocol):
    @property
    def param_groups(self) -> list[dict[str, object]]: ...


class LRScheduler:
    """Base class for epoch-oriented learning-rate schedules."""

    def __init__(self, optimizer: _Optimizer, last_epoch: int = -1) -> None:
        if not hasattr(optimizer, "param_groups") or not optimizer.param_groups:
            raise TypeError("optimizer must be a Tynx optimizer with parameter groups")
        if type(last_epoch) is not int or last_epoch < -1:
            raise ValueError(f"last_epoch must be an integer >= -1, got {last_epoch!r}")
        self.optimizer = optimizer
        self.base_lrs = [_learning_rate(group) for group in optimizer.param_groups]
        self.last_epoch = 0 if last_epoch == -1 else last_epoch
        self._last_lr = [_learning_rate(group) for group in optimizer.param_groups]

    def step(self, epoch: _Optional[int] = None) -> None:
        """Advance the schedule once, normally after an optimizer step."""
        if epoch is None:
            next_epoch = self.last_epoch + 1
        elif type(epoch) is not int or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
        else:
            next_epoch = epoch
        rates = self._rates(next_epoch)
        if len(rates) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler returned the wrong number of learning rates")
        for group, rate in zip(self.optimizer.param_groups, rates):
            group["lr"] = rate
        self.last_epoch = next_epoch
        self._last_lr = rates

    def get_last_lr(self) -> list[float]:
        """Return the most recently computed learning rate for each group."""
        return list(self._last_lr)

    def state_dict(self) -> dict[str, object]:
        """Return resumable scheduler state without the optimizer."""
        return {
            "version": 1,
            "scheduler": type(self).__name__,
            "last_epoch": self.last_epoch,
            "base_lrs": list(self.base_lrs),
            "last_lrs": list(self._last_lr),
            "options": self._state_options(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore scheduler progress and its current group rates."""
        if state_dict.get("version") != 1:
            raise ValueError("scheduler state_dict must have version 1")
        if state_dict.get("scheduler") != type(self).__name__:
            raise ValueError(
                f"cannot load {state_dict.get('scheduler')} state into {type(self).__name__}"
            )
        last_epoch = state_dict.get("last_epoch")
        base_lrs = _rate_list(state_dict.get("base_lrs"), "base_lrs")
        last_lrs = _rate_list(state_dict.get("last_lrs"), "last_lrs")
        options = state_dict.get("options")
        group_count = len(self.optimizer.param_groups)
        if type(last_epoch) is not int or last_epoch < 0:
            raise ValueError("scheduler last_epoch must be a non-negative integer")
        if len(base_lrs) != group_count or len(last_lrs) != group_count:
            raise ValueError("scheduler state parameter-group count does not match the optimizer")
        if not isinstance(options, dict):
            raise TypeError("scheduler options must be a dictionary")
        self._load_state_options(options)
        self.last_epoch = last_epoch
        self.base_lrs = base_lrs
        self._last_lr = last_lrs
        for group, rate in zip(self.optimizer.param_groups, last_lrs):
            group["lr"] = rate

    def _rates(self, epoch: int) -> list[float]:
        raise NotImplementedError

    def _state_options(self) -> dict[str, object]:
        return {}

    def _load_state_options(self, options: dict[str, object]) -> None:
        if options:
            raise ValueError(f"unexpected scheduler options {sorted(options)!r}")


class StepLR(LRScheduler):
    """Decay each group learning rate by ``gamma`` every ``step_size`` epochs."""

    def __init__(
        self,
        optimizer: _Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if type(step_size) is not int or step_size <= 0:
            raise ValueError(f"step_size must be a positive integer, got {step_size!r}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.step_size = step_size
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def _rates(self, epoch: int) -> list[float]:
        factor = self.gamma ** (epoch // self.step_size)
        return [rate * factor for rate in self.base_lrs]

    def _state_options(self) -> dict[str, object]:
        return {"step_size": self.step_size, "gamma": self.gamma}

    def _load_state_options(self, options: dict[str, object]) -> None:
        step_size = options.get("step_size")
        gamma = options.get("gamma")
        if type(step_size) is not int or step_size <= 0:
            raise ValueError("StepLR state has an invalid step_size")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma < 0:
            raise ValueError("StepLR state has an invalid gamma")
        self.step_size = step_size
        self.gamma = float(gamma)


class ExponentialLR(LRScheduler):
    """Multiply each group learning rate by ``gamma`` after every epoch."""

    def __init__(
        self,
        optimizer: _Optimizer,
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def _rates(self, epoch: int) -> list[float]:
        factor = self.gamma**epoch
        return [rate * factor for rate in self.base_lrs]

    def _state_options(self) -> dict[str, object]:
        return {"gamma": self.gamma}

    def _load_state_options(self, options: dict[str, object]) -> None:
        gamma = options.get("gamma")
        if not isinstance(gamma, (int, float)) or isinstance(gamma, bool) or gamma < 0:
            raise ValueError("ExponentialLR state has an invalid gamma")
        self.gamma = float(gamma)


class CosineAnnealingLR(LRScheduler):
    """Anneal each group learning rate to ``eta_min`` over ``T_max`` epochs."""

    def __init__(
        self,
        optimizer: _Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if type(T_max) is not int or T_max <= 0:
            raise ValueError(f"T_max must be a positive integer, got {T_max!r}")
        self.T_max = T_max
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def _rates(self, epoch: int) -> list[float]:
        factor = (1.0 + _math.cos(_math.pi * epoch / self.T_max)) / 2.0
        return [self.eta_min + (rate - self.eta_min) * factor for rate in self.base_lrs]

    def _state_options(self) -> dict[str, object]:
        return {"T_max": self.T_max, "eta_min": self.eta_min}

    def _load_state_options(self, options: dict[str, object]) -> None:
        maximum = options.get("T_max")
        minimum = options.get("eta_min")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("CosineAnnealingLR state has an invalid T_max")
        if not isinstance(minimum, (int, float)) or isinstance(minimum, bool):
            raise ValueError("CosineAnnealingLR state has an invalid eta_min")
        self.T_max = maximum
        self.eta_min = float(minimum)


def _learning_rate(group: dict[str, object]) -> float:
    try:
        rate = float(_cast(float, group["lr"]))
    except (KeyError, TypeError, ValueError) as error:
        raise TypeError("optimizer parameter groups must contain numeric learning rates") from error
    if not _math.isfinite(rate) or rate < 0:
        raise ValueError(f"optimizer learning rates must be finite and non-negative, got {rate}")
    return rate


def _rate_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"scheduler {name} must be a list")
    try:
        rates = [float(_cast(float, item)) for item in value]
    except (TypeError, ValueError) as error:
        raise TypeError(f"scheduler {name} must contain numbers") from error
    if any(not _math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError(f"scheduler {name} must contain finite non-negative values")
    return rates


__all__ = ["CosineAnnealingLR", "ExponentialLR", "LRScheduler", "StepLR"]
