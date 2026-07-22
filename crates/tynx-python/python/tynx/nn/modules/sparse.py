"""Authored dense-gradient sparse-style layers."""

from typing import Literal, Optional

from ..._tynx import Device, Parameter, Tensor, arange, randn
from ..functional import embedding
from .module import Module


class Embedding(Module):
    """A dense lookup table with repeated-index gradient accumulation."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        *,
        device: Optional[Device] = None,
        dtype: Optional[Literal["float32"]] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = _positive_int(num_embeddings, "num_embeddings")
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        if max_norm is not None:
            raise NotImplementedError("Embedding max_norm is not supported")
        if norm_type != 2.0:
            raise NotImplementedError("Embedding norm_type is only meaningful with max_norm")
        if scale_grad_by_freq:
            raise NotImplementedError("Embedding scale_grad_by_freq=True is not supported")
        if sparse:
            raise NotImplementedError("Embedding sparse gradients are not supported")
        if padding_idx is not None:
            if (
                type(padding_idx) is not int
                or not -self.num_embeddings <= padding_idx < self.num_embeddings
            ):
                raise ValueError(
                    f"padding_idx must be within [-{self.num_embeddings}, "
                    f"{self.num_embeddings - 1}], got {padding_idx!r}"
                )
            padding_idx = padding_idx + self.num_embeddings if padding_idx < 0 else padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        weight = randn(
            (self.num_embeddings, self.embedding_dim),
            dtype="float32" if dtype is None else dtype,
            device=device,
        )
        if padding_idx is not None:
            mask = (arange(self.num_embeddings, device=device) != padding_idx).cast("float32")
            weight = weight * mask.unsqueeze(1)
        self.weight = Parameter(weight, name="weight")

    def forward(self, input: Tensor) -> Tensor:
        return embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        details = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            details += f", padding_idx={self.padding_idx}"
        return details


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = ["Embedding"]
