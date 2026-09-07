import pytest
import tynx


def test_silu_eager_autograd_and_capture_agree() -> None:
    layer = tynx.nn.SiLU()
    eager_input = tynx.Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
    eager = layer(eager_input)
    eager.sum().backward()

    compiled = tynx.compile(layer.forward, fullgraph=True)
    captured_input = tynx.Tensor([[-1.0, 0.0, 2.0]], requires_grad=True)
    captured = compiled(captured_input)
    captured.sum().backward()

    assert captured.flatten().tolist() == pytest.approx(eager.flatten().tolist())
    assert captured_input.grad is not None
    assert eager_input.grad is not None
    assert captured_input.grad.flatten().tolist() == pytest.approx(
        eager_input.grad.flatten().tolist()
    )
    assert compiled.compile_count == 1
