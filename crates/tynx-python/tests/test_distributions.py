"""Categorical and Normal distribution behavior."""

import math

import pytest
import tynx


def test_categorical_log_prob_entropy_and_gradients() -> None:
    logits = tynx.Tensor([[0.0, math.log(3.0)]], requires_grad=True)
    distribution = tynx.distributions.Categorical(logits=logits)
    action = tynx.Tensor([0], dtype="int64")

    log_prob = distribution.log_prob(action)
    entropy = distribution.entropy()
    (-log_prob).mean().backward()

    assert log_prob.tolist() == pytest.approx([math.log(0.25)])
    assert entropy.tolist() == pytest.approx([-(0.25 * math.log(0.25) + 0.75 * math.log(0.75))])
    assert logits.grad is not None
    assert logits.grad.flatten().tolist() == pytest.approx([-0.75, 0.75])


def test_categorical_sampling_is_seeded_detached_and_advances_rng() -> None:
    distribution = tynx.distributions.Categorical(logits=tynx.Tensor([[0.0, 0.0]] * 128))

    first = distribution.sample(seed=123)
    repeated = distribution.sample(seed=123)
    advanced = distribution.sample()

    assert first.dtype == "int64"
    assert first.shape == (128,)
    assert first.tolist() == repeated.tolist()
    assert advanced.tolist() != repeated.tolist()

    tynx.manual_seed(77)
    manual_first = distribution.sample()
    tynx.manual_seed(77)
    manual_repeated = distribution.sample()
    assert manual_first.tolist() == manual_repeated.tolist()


def test_categorical_log_prob_broadcasts_leading_sample_dimensions() -> None:
    distribution = tynx.distributions.Categorical(
        logits=tynx.Tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
    )

    selected = distribution.log_prob(tynx.Tensor([[1, 2], [0, 1]], dtype="int64"))

    first_normalizer = math.log(math.exp(1.0) + math.exp(2.0) + math.exp(0.5))
    second_normalizer = math.log(math.exp(0.0) + math.exp(1.0) + math.exp(3.0))
    assert selected.shape == (2, 2)
    assert selected.flatten().tolist() == pytest.approx(
        [
            2.0 - first_normalizer,
            3.0 - second_normalizer,
            1.0 - first_normalizer,
            1.0 - second_normalizer,
        ]
    )


def test_categorical_log_prob_broadcasts_singleton_batch_dimension() -> None:
    distribution = tynx.distributions.Categorical(
        logits=tynx.Tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
    )

    selected = distribution.log_prob(tynx.Tensor([[1], [2]], dtype="int64"))

    assert selected.shape == (2, 2)


def test_categorical_log_prob_rejects_incompatible_batch_shape() -> None:
    distribution = tynx.distributions.Categorical(
        logits=tynx.Tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
    )

    with pytest.raises(ValueError, match="not broadcastable"):
        distribution.log_prob(tynx.Tensor([[1, 2, 0]], dtype="int64"))


def test_normal_log_prob_entropy_gradients_and_seeded_sample() -> None:
    loc = tynx.Tensor([0.0, 1.0], requires_grad=True)
    scale = tynx.Tensor([1.0, 2.0], requires_grad=True)
    distribution = tynx.distributions.Normal(loc, scale)
    value = tynx.Tensor([1.0, 1.0])

    log_prob = distribution.log_prob(value)
    entropy = distribution.entropy()
    (-log_prob).sum().backward()

    assert log_prob.tolist() == pytest.approx(
        [-0.5 - 0.5 * math.log(2.0 * math.pi), -math.log(2.0) - 0.5 * math.log(2.0 * math.pi)]
    )
    assert entropy.tolist() == pytest.approx(
        [
            0.5 * math.log(2.0 * math.pi * math.e),
            math.log(2.0) + 0.5 * math.log(2.0 * math.pi * math.e),
        ]
    )
    assert loc.grad is not None
    assert loc.grad.tolist() == pytest.approx([-1.0, 0.0])
    assert scale.grad is not None
    assert scale.grad.tolist() == pytest.approx([0.0, 0.5])

    first = distribution.sample(seed=42)
    repeated = distribution.sample(seed=42)
    assert first.tolist() == repeated.tolist()
    assert first.requires_grad is False
    assert tynx.distributions.Normal(loc, 1.0).entropy().shape == (2,)


def test_distribution_argument_errors_are_visible() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        tynx.distributions.Categorical()
    with pytest.raises(ValueError, match="exactly one"):
        tynx.distributions.Categorical(
            probs=tynx.Tensor([0.5, 0.5]), logits=tynx.Tensor([0.0, 0.0])
        )
    with pytest.raises(TypeError, match="float32"):
        tynx.distributions.Normal(tynx.Tensor([0], dtype="int64"), tynx.Tensor([1.0]))
    for scale in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="greater than zero"):
            tynx.distributions.Normal(0.0, scale)
    for tensor_scale in (
        tynx.Tensor([1.0, 0.0]),
        tynx.Tensor([-1.0, 1.0]),
        tynx.Tensor([float("nan")]),
    ):
        with pytest.raises(ValueError, match="greater than zero"):
            tynx.distributions.Normal(tynx.Tensor([0.0]), tensor_scale, validate_args=True)
    with pytest.raises(TypeError, match="validate_args must be a bool or None"):
        tynx.distributions.Normal(0.0, 1.0, validate_args=1)  # type: ignore[arg-type]


def test_normal_tensor_scale_validation_is_explicit() -> None:
    scale = tynx.Tensor([1.0, 0.0])

    distribution = tynx.distributions.Normal(tynx.Tensor([0.0]), scale)

    assert distribution.scale is scale


def test_normal_validate_args_false_consistently_skips_scale_checks() -> None:
    scalar = tynx.distributions.Normal(0.0, -1.0, validate_args=False)
    tensor_scale = tynx.Tensor([-1.0])
    tensor = tynx.distributions.Normal(
        tynx.Tensor([0.0]),
        tensor_scale,
        validate_args=False,
    )

    assert scalar.scale.item() == pytest.approx(-1.0)
    assert tensor.scale is tensor_scale
