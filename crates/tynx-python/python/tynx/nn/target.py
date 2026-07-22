"""Off-tape state updates for frozen target networks."""

from collections.abc import Mapping
from typing import Literal, Union

from .._tynx import Buffer, Parameter, Tensor
from .state import named_buffers, named_parameters

BufferPolicy = Literal["copy", "average"]
_State = Union[Parameter, Buffer]


def hard_update_(target: object, source: object) -> None:
    """Copy all source parameters and buffers into matching target state."""
    _update_(target, source, tau=1.0, buffer_policy="copy")


def soft_update_(
    target: object,
    source: object,
    tau: float,
    *,
    buffer_policy: BufferPolicy = "copy",
) -> None:
    """Polyak-average parameters and copy or average matching buffers off-tape."""
    if not isinstance(tau, (int, float)) or isinstance(tau, bool) or not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be a real number in [0, 1], got {tau!r}")
    if buffer_policy not in ("copy", "average"):
        raise ValueError(f"buffer_policy must be 'copy' or 'average', got {buffer_policy!r}")
    _update_(target, source, tau=float(tau), buffer_policy=buffer_policy)


def _update_(
    target: object,
    source: object,
    *,
    tau: float,
    buffer_policy: BufferPolicy,
) -> None:
    target_parameters = dict(named_parameters(target))
    source_parameters = dict(named_parameters(source))
    target_buffers = dict(named_buffers(target))
    source_buffers = dict(named_buffers(source))
    _validate_model_state(target_parameters, target_buffers, "target")
    _validate_model_state(source_parameters, source_buffers, "source")
    _validate_matching_names(target_parameters, source_parameters, "parameter")
    _validate_matching_names(target_buffers, source_buffers, "buffer")

    target_state: dict[str, _State] = {**target_parameters, **target_buffers}
    source_state: dict[str, _State] = {**source_parameters, **source_buffers}
    for name in sorted(target_state):
        target_value = target_state[name]
        source_value = source_state[name]
        if target_value.shape != source_value.shape or target_value.dtype != source_value.dtype:
            raise ValueError(
                f"state {name!r} has target shape {target_value.shape} and dtype "
                f"{target_value.dtype}, but source shape {source_value.shape} and dtype "
                f"{source_value.dtype}"
            )

    target_snapshots = {name: value.detach() for name, value in target_state.items()}
    source_snapshots = {name: value.detach() for name, value in source_state.items()}
    prepared: dict[str, Tensor] = {}
    for name in sorted(target_parameters):
        prepared[name] = _average(target_snapshots[name], source_snapshots[name], tau)
    for name in sorted(target_buffers):
        source_snapshot = source_snapshots[name]
        if buffer_policy == "average" and source_snapshot.dtype == "float32":
            prepared[name] = _average(target_snapshots[name], source_snapshot, tau)
        else:
            prepared[name] = source_snapshot

    for name in sorted(prepared):
        target_state[name].copy_(prepared[name])


def _average(target: Tensor, source: Tensor, tau: float) -> Tensor:
    return target * (1.0 - tau) + source * tau


def _validate_model_state(
    parameters: dict[str, Parameter], buffers: dict[str, Buffer], side: str
) -> None:
    collisions = sorted(set(parameters) & set(buffers))
    if collisions:
        raise ValueError(f"{side} parameters and buffers share state names: {collisions!r}")


def _validate_matching_names(
    target: Mapping[str, _State], source: Mapping[str, _State], kind: str
) -> None:
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    if missing or unexpected:
        raise ValueError(
            f"target/source {kind} names differ: missing={missing!r}, unexpected={unexpected!r}"
        )


__all__ = ["BufferPolicy", "hard_update_", "soft_update_"]
