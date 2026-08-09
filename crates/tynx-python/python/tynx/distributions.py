"""Small differentiable probability distributions over eager Tynx tensors."""

import math as _math
from typing import Optional as _Optional
from typing import Union as _Union

from ._tynx import Tensor, _categorical_sample, _normal_sample
from ._tynx import zeros as _zeros


def _broadcast_shape(
    left: tuple[int, ...],
    right: tuple[int, ...],
    operation: str,
) -> tuple[int, ...]:
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + left
    right = (1,) * (rank - len(right)) + right
    output = []
    for left_dim, right_dim in zip(left, right):
        if left_dim == right_dim:
            output.append(left_dim)
        elif left_dim == 1:
            output.append(right_dim)
        elif right_dim == 1:
            output.append(left_dim)
        else:
            raise ValueError(f"{operation} shape {left} is not broadcastable with shape {right}")
    return tuple(output)


def _sample_shape(value: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("sample_shape must be a tuple of non-negative integers")
    if any(type(dimension) is not int or dimension < 0 for dimension in value):
        raise ValueError("sample_shape must contain only non-negative integers")
    return value


def _expand_to(value: Tensor, shape: tuple[int, ...]) -> Tensor:
    aligned_shape = (1,) * (len(shape) - value.ndim) + value.shape
    if value.shape != aligned_shape:
        value = value.reshape(aligned_shape)
    return value if value.shape == shape else value.expand(shape)


class Categorical:
    """Categorical distribution parameterized by logits or probabilities."""

    def __init__(
        self,
        probs: _Optional[Tensor] = None,
        logits: _Optional[Tensor] = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError("Categorical requires exactly one of probs or logits")
        if logits is not None:
            self.logits = logits.log_softmax(dim=-1)
            self.probs = self.logits.exp()
        else:
            assert probs is not None
            normalized = probs / probs.sum(dim=-1, keepdim=True)
            self.probs = normalized
            self.logits = normalized.log()

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        seed: _Optional[int] = None,
    ) -> Tensor:
        """Draw detached int64 class indices, advancing native device RNG state."""
        sample_shape = _sample_shape(sample_shape)
        logits = _expand_to(
            self.logits,
            (*sample_shape, *self.logits.shape),
        )
        return _categorical_sample(logits, seed)

    def log_prob(self, value: Tensor) -> Tensor:
        """Return selected normalized log probabilities for int64 indices."""
        batch_shape = self.logits.shape[:-1]
        output_shape = _broadcast_shape(value.shape, batch_shape, "Categorical.log_prob value")

        logits_shape = (1,) * (len(output_shape) - len(batch_shape)) + self.logits.shape
        logits = (
            self.logits if logits_shape == self.logits.shape else self.logits.reshape(logits_shape)
        )
        expanded_logits_shape = (*output_shape, self.logits.shape[-1])
        if logits.shape != expanded_logits_shape:
            logits = logits.expand(expanded_logits_shape)

        index_shape = (1,) * (len(output_shape) - value.ndim) + value.shape + (1,)
        index = value.unsqueeze(-1)
        if index.shape != index_shape:
            index = index.reshape(index_shape)
        expanded_index_shape = (*output_shape, 1)
        if index.shape != expanded_index_shape:
            index = index.expand(expanded_index_shape)
        return logits.gather(-1, index).squeeze(-1)

    def entropy(self) -> Tensor:
        """Return Shannon entropy over the final category dimension."""
        return -(self.probs * self.logits).sum(dim=-1)


class Normal:
    """Elementwise normal distribution with differentiable parameters."""

    def __init__(
        self,
        loc: _Union[Tensor, float],
        scale: _Union[Tensor, float],
        *,
        validate_args: _Optional[bool] = None,
    ) -> None:
        """Construct a Normal distribution.

        The default validates scalar scales without synchronizing tensor values. Set
        ``validate_args=True`` to validate tensor-valued scales too; that explicit check
        reads the tensor on the host. Set it to ``False`` to disable both checks.
        """
        if validate_args is not None and not isinstance(validate_args, bool):
            raise TypeError("Normal validate_args must be a bool or None")
        if not isinstance(scale, Tensor) and validate_args is not False and not scale > 0.0:
            raise ValueError(f"Normal scale must be greater than zero, got {scale!r}")
        if not isinstance(loc, Tensor):
            loc = scale.detach() * 0.0 + loc if isinstance(scale, Tensor) else Tensor([loc])
        if not isinstance(scale, Tensor):
            scale = loc.detach() * 0.0 + scale
        if loc.dtype != "float32" or scale.dtype != "float32":
            raise TypeError("Normal loc and scale must be float32 Tensors")
        if validate_args is True and scale.numel > 0 and not scale.min().item() > 0.0:
            raise ValueError("Normal scale must contain only values greater than zero")
        self.loc = loc
        self.scale = scale

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        seed: _Optional[int] = None,
    ) -> Tensor:
        """Draw a detached sample, advancing native device RNG state."""
        sample_shape = _sample_shape(sample_shape)
        batch_shape = _broadcast_shape(self.loc.shape, self.scale.shape, "Normal parameter")
        output_shape = (*sample_shape, *batch_shape)
        if 0 in output_shape:
            if self.loc.device != self.scale.device:
                raise ValueError("Normal loc and scale must be on the same device")
            return _zeros(output_shape, device=self.loc.device)
        loc = _expand_to(self.loc, output_shape)
        scale = _expand_to(self.scale, output_shape)
        return _normal_sample(loc, scale, seed)

    def log_prob(self, value: Tensor) -> Tensor:
        """Return elementwise log density."""
        standardized = (value - self.loc) / self.scale
        return (
            -(standardized * standardized) / 2.0
            - self.scale.log()
            - 0.5 * _math.log(2.0 * _math.pi)
        )

    def entropy(self) -> Tensor:
        """Return elementwise differential entropy."""
        return self.scale.log() + 0.5 * _math.log(2.0 * _math.pi * _math.e)


__all__ = ["Categorical", "Normal"]
