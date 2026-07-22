"""Authored and functional two-dimensional convolution behavior."""

import pytest
import tynx


def test_functional_conv2d_forward_and_backward() -> None:
    input = tynx.Tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        requires_grad=True,
    )
    weight = tynx.Parameter([[[[1.0, 1.0], [1.0, 1.0]]]])
    bias = tynx.Parameter([0.0])

    output = tynx.nn.functional.conv2d(input, weight, bias)

    assert output.shape == (1, 1, 2, 2)
    assert output.tolist() == [[[[12.0, 16.0], [24.0, 28.0]]]]
    output.sum().backward()
    assert input.grad is not None
    assert input.grad.tolist() == [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]
    assert weight.grad is not None
    assert weight.grad.tolist() == [[[[12.0, 16.0], [24.0, 28.0]]]]
    assert bias.grad is not None
    assert bias.grad.tolist() == pytest.approx([4.0])


def test_conv2d_layer_initializes_grouped_state_and_runs() -> None:
    layer = tynx.nn.Conv2d(4, 6, (3, 2), stride=2, padding=(1, 0), groups=2, bias=False)
    input = tynx.Tensor([[[[1.0] * 6 for _ in range(5)]] * 4])

    output = layer(input)

    assert layer.weight.shape == (6, 2, 3, 2)
    assert layer.bias is None
    assert output.shape == (1, 6, 3, 3)
    assert [name for name, _ in layer.named_parameters()] == ["weight"]
    assert "groups=2" in repr(layer)


def test_conv2d_validates_public_configuration_and_shapes() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tynx.nn.Conv2d(0, 2, 3)
    with pytest.raises(ValueError, match="divisible by groups"):
        tynx.nn.Conv2d(3, 4, 3, groups=2)
    with pytest.raises(ValueError, match="positive integers"):
        tynx.nn.Conv2d(1, 1, 0)
    with pytest.raises(ValueError, match="non-negative integers"):
        tynx.nn.Conv2d(1, 1, 3, padding=-1)
    with pytest.raises(TypeError, match="bias must be a bool"):
        tynx.nn.Conv2d(1, 1, 3, bias=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="padding_mode"):
        tynx.nn.Conv2d(1, 1, 3, padding_mode="reflect")
    with pytest.raises(ValueError, match="rank-4"):
        tynx.nn.Conv2d(1, 1, 1)(tynx.Tensor([[1.0]]))
    with pytest.raises(ValueError, match="input channels"):
        tynx.nn.Conv2d(2, 1, 1)(tynx.Tensor([[[[1.0]]]]))


def test_functional_conv2d_rejects_incompatible_kernel_and_bias() -> None:
    input = tynx.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    with pytest.raises(ValueError, match="exceeds padded input"):
        tynx.nn.functional.conv2d(input, tynx.Tensor([[[[1.0] * 3] * 3]]))
    with pytest.raises(ValueError, match="bias must have shape"):
        tynx.nn.functional.conv2d(input, tynx.Tensor([[[[1.0]]]]), tynx.Tensor([0.0, 0.0]))
