"""Small differentiable probability distributions over eager Tynx tensors."""

import math as _math
from typing import Optional as _Optional
from typing import Union as _Union

from ._tynx import Tensor, _categorical_sample, _normal_sample


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

    def sample(self, *, seed: _Optional[int] = None) -> Tensor:
        """Draw detached int64 class indices, advancing native device RNG state."""
        return _categorical_sample(self.logits, seed)

    def log_prob(self, value: Tensor) -> Tensor:
        """Return selected normalized log probabilities for int64 indices."""
        index = value if self.logits.ndim == 1 else value.unsqueeze(-1)
        selected = self.logits.gather(-1, index)
        return selected if self.logits.ndim == 1 else selected.squeeze(-1)

    def entropy(self) -> Tensor:
        """Return Shannon entropy over the final category dimension."""
        return -(self.probs * self.logits).sum(dim=-1)


class Normal:
    """Elementwise normal distribution with differentiable parameters."""

    def __init__(self, loc: _Union[Tensor, float], scale: _Union[Tensor, float]) -> None:
        if not isinstance(loc, Tensor):
            loc = scale.detach() * 0.0 + loc if isinstance(scale, Tensor) else Tensor([loc])
        if not isinstance(scale, Tensor):
            scale = loc.detach() * 0.0 + scale
        if loc.dtype != "float32" or scale.dtype != "float32":
            raise TypeError("Normal loc and scale must be float32 Tensors")
        self.loc = loc
        self.scale = scale

    def sample(self, *, seed: _Optional[int] = None) -> Tensor:
        """Draw a detached sample, advancing native device RNG state."""
        return _normal_sample(self.loc, self.scale, seed)

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
