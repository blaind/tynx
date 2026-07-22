"""Native-RNG parameter initialization helpers."""

import math
from typing import Literal, Optional

from .._tynx import Tensor, rand, randn

FanMode = Literal["fan_in", "fan_out"]


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill a stable Parameter or Buffer from a uniform distribution."""
    _require_float_tensor(tensor)
    if a > b:
        raise ValueError(f"uniform_ requires a <= b, got {a} > {b}")
    tensor.copy_(rand(tensor.shape, device=tensor.device) * (b - a) + a)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill a stable Parameter or Buffer from a normal distribution."""
    _require_float_tensor(tensor)
    if std < 0:
        raise ValueError(f"normal_ requires std >= 0, got {std}")
    tensor.copy_(randn(tensor.shape, device=tensor.device) * std + mean)
    return tensor


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill using Glorot uniform initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill using Glorot normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0.0,
    mode: FanMode = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """Fill using He uniform initialization."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(float(fan))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0.0,
    mode: FanMode = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """Fill using He normal initialization."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    return normal_(tensor, 0.0, gain / math.sqrt(float(fan)))


def calculate_gain(nonlinearity: str, param: Optional[float] = None) -> float:
    """Return the recommended scaling factor for an activation."""
    if nonlinearity in {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
    }:
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        if isinstance(slope, bool) or not isinstance(slope, (int, float)):
            raise ValueError(f"negative_slope {slope!r} is not a valid number")
        return math.sqrt(2.0 / (1.0 + float(slope) ** 2))
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError(
            "fan-in and fan-out require a tensor with at least two dimensions, "
            f"got shape {tensor.shape}"
        )
    receptive_field_size = math.prod(tensor.shape[2:])
    return tensor.shape[1] * receptive_field_size, tensor.shape[0] * receptive_field_size


def _calculate_correct_fan(tensor: Tensor, mode: FanMode) -> int:
    if mode not in {"fan_in", "fan_out"}:
        raise ValueError(f"mode must be 'fan_in' or 'fan_out', got {mode!r}")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def _require_float_tensor(tensor: Tensor) -> None:
    if tensor.dtype != "float32":
        raise TypeError(f"initialization requires a float32 tensor, got {tensor.dtype}")


__all__ = [
    "calculate_gain",
    "kaiming_normal_",
    "kaiming_uniform_",
    "normal_",
    "uniform_",
    "xavier_normal_",
    "xavier_uniform_",
]
