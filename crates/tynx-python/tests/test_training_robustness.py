"""Cross-cutting gradient, RNG, lifetime, and failure-boundary checks."""

import gc
from pathlib import Path

import pytest
import tynx

_GEMM_MODEL = bytes.fromhex(
    "08084202100d3a8c010a200a01780a067765696768740a04626961731201791a0468656164"
    "220447656d6d1215707974686f6e5f696d706f727465645f6d6f64656c2a140a0201011001"
    "22040000004042067765696768742a110a0101100122040000803f4204626961735a130a0178"
    "120e0a0c080112080a0208020a02080162130a0179120e0a0c080112080a0208020a020801"
)


def test_conv2d_input_gradient_matches_central_finite_difference() -> None:
    values = [0.2, -0.4, 0.7, 1.1]
    input = tynx.Tensor([[[values[:2], values[2:]]]], requires_grad=True)
    weight = tynx.Tensor([[[[0.5, -0.25], [0.75, 0.1]]]])

    tynx.nn.functional.conv2d(input, weight).sum().backward()
    tynx.synchronize(input.device)

    assert input.grad is not None
    epsilon = 1e-3

    def evaluate(first: float) -> float:
        candidate = [first, *values[1:]]
        tensor = tynx.Tensor([[[candidate[:2], candidate[2:]]]])
        output = tynx.nn.functional.conv2d(tensor, weight).sum()
        tynx.synchronize(output.device)
        return output.item()

    numerical = (evaluate(values[0] + epsilon) - evaluate(values[0] - epsilon)) / (2 * epsilon)
    assert input.grad.flatten().tolist()[0] == pytest.approx(numerical, rel=2e-3, abs=2e-3)


def test_layer_norm_input_gradient_matches_central_finite_difference() -> None:
    values = [0.5, -1.0, 2.0]
    coefficients = tynx.Tensor([0.25, -0.5, 1.5])
    layer = tynx.nn.LayerNorm(3)
    input = tynx.Tensor([values], requires_grad=True)

    (layer(input) * coefficients).sum().backward()
    tynx.synchronize(input.device)

    assert input.grad is not None
    epsilon = 1e-3

    def evaluate(middle: float) -> float:
        candidate = tynx.Tensor([[values[0], middle, values[2]]])
        output = (layer(candidate) * coefficients).sum()
        tynx.synchronize(output.device)
        return output.item()

    numerical = (evaluate(values[1] + epsilon) - evaluate(values[1] - epsilon)) / (2 * epsilon)
    assert input.grad.flatten().tolist()[1] == pytest.approx(numerical, rel=3e-3, abs=3e-3)


def test_random_consumers_share_one_replayable_advancing_stream() -> None:
    dropout = tynx.nn.Dropout(0.5)
    normal = tynx.distributions.Normal(tynx.Tensor([0.0] * 32), 1.0)
    categorical = tynx.distributions.Categorical(logits=tynx.Tensor([[0.0, 0.0]] * 32))

    def draw_sequence() -> tuple[list[object], list[object], list[object]]:
        return (
            dropout(tynx.Tensor([1.0] * 32)).tolist(),
            normal.sample().tolist(),
            categorical.sample().tolist(),
        )

    tynx.manual_seed(812)
    first = draw_sequence()
    advanced = draw_sequence()
    tynx.manual_seed(812)
    replay = draw_sequence()

    assert first == replay
    assert advanced != first


def test_imported_output_retains_parameter_tape_after_model_drop(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(_GEMM_MODEL)
    model = tynx.load(
        path,
        trainable="auto",
        simplify=False,
        initializer_names={"weight": "weight", "bias": "bias"},
    )
    parameters = dict(model.named_parameters())
    output = model(tynx.Tensor([[2.0], [-1.0]]))

    del model
    gc.collect()
    output.sum().backward()

    assert parameters["weight"].grad is not None
    assert parameters["weight"].grad.flatten().tolist() == pytest.approx([1.0])
    assert parameters["bias"].grad is not None
    assert parameters["bias"].grad.tolist() == pytest.approx([2.0])


def test_captured_output_retains_parameter_tape_after_wrapper_drop() -> None:
    layer = tynx.nn.Embedding(3, 2, padding_idx=0)
    layer.weight.copy_(tynx.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    weight = layer.weight

    @tynx.compile(fullgraph=True)
    def lookup(indices: tynx.Tensor) -> tynx.Tensor:
        return layer(indices)

    indices = tynx.Tensor([0, 2, 2, 1], dtype="int64")
    lookup(indices)
    output = lookup(indices)
    loss = output.sum()
    assert lookup.replay_count == 1

    del output
    del lookup
    gc.collect()
    tynx.synchronize(weight.device)
    loss.backward()
    tynx.synchronize(weight.device)

    assert weight.grad is not None
    assert weight.grad.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_multiple_forwards_share_one_parameter_generation_until_step() -> None:
    layer = tynx.nn.Linear(1, 1)
    layer.weight.copy_(tynx.Tensor([[2.0]]))
    assert layer.bias is not None
    layer.bias.copy_(tynx.Tensor([0.0]))

    first = layer(tynx.Tensor([[3.0]]))
    second = layer(tynx.Tensor([[-1.0]]))
    (first + second).sum().backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.flatten().tolist() == pytest.approx([2.0])
    assert layer.bias.grad is not None
    assert layer.bias.grad.tolist() == pytest.approx([2.0])


def test_backward_rejects_parameter_mutation_after_forward() -> None:
    weight = tynx.Parameter([1.0, 2.0, 3.0], name="weight")
    loss = (weight * tynx.Tensor([4.0, 5.0, 6.0])).sum()

    weight.copy_(tynx.Tensor([10.0, 20.0, 30.0]))

    with pytest.raises(RuntimeError, match=r"weight.*modified after the forward pass"):
        loss.backward()
    assert weight.grad is None


@pytest.mark.parametrize("new_branch_first", [False, True])
def test_backward_rejects_mixed_parameter_generations_regardless_of_operand_order(
    new_branch_first: bool,
) -> None:
    weight = tynx.Parameter([1.0], name="weight")
    old_branch = weight * tynx.Tensor([2.0])
    weight.copy_(tynx.Tensor([3.0]))
    new_branch = weight * tynx.Tensor([4.0])
    loss = new_branch + old_branch if new_branch_first else old_branch + new_branch

    with pytest.raises(RuntimeError, match=r"weight.*multiple value generations"):
        loss.backward()
    assert weight.grad is None


def test_failed_backward_and_native_shape_errors_do_not_corrupt_gradients() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    (value * value).backward()
    assert value.grad is not None
    before = value.grad.tolist()

    with pytest.raises(ValueError, match="gradient shape"):
        (value * value).backward(tynx.Tensor([1.0, 1.0]))
    assert value.grad is not None
    assert value.grad.tolist() == before

    image = tynx.Tensor([[[[1.0]]]])
    with pytest.raises(ValueError, match="stride"):
        tynx.nn.functional.conv2d(image, tynx.Tensor([[[[1.0]]]]), stride=0)
    with pytest.raises(ValueError, match="expected weight input channels"):
        tynx.nn.functional.conv2d(
            tynx.Tensor([[[[1.0]], [[2.0]]]]),
            tynx.Tensor([[[[1.0]]]]),
        )

    recovered = (value * value).sum()
    recovered.backward()
    tynx.synchronize(value.device)
    assert value.grad is not None
    assert value.grad.tolist() == [8.0]
