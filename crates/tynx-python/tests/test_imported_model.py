"""Callable imported ONNX training models."""

from pathlib import Path

import pytest
import tynx

_GEMM_MODEL = bytes.fromhex(
    "08084202100d3a8c010a200a01780a067765696768740a04626961731201791a0468656164"
    "220447656d6d1215707974686f6e5f696d706f727465645f6d6f64656c2a140a0201011001"
    "22040000004042067765696768742a110a0101100122040000803f4204626961735a130a0178"
    "120e0a0c080112080a0208020a02080162130a0179120e0a0c080112080a0208020a020801"
)


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "gemm.onnx"
    path.write_bytes(_GEMM_MODEL)
    return path


def _load_model(tmp_path: Path) -> tynx.ImportedModel:
    return tynx.load(
        _model_path(tmp_path),
        trainable="auto",
        simplify=False,
        initializer_names={"constant1_out1": "head.weight", "constant2_out1": "head.bias"},
    )


def test_imported_model_is_callable_and_preserves_the_eager_tape(tmp_path: Path) -> None:
    model = _load_model(tmp_path)
    input = tynx.Tensor([[2.0], [-1.0]], requires_grad=True)

    output = model(input)
    output.sum().backward()

    assert output.flatten().tolist() == pytest.approx([5.0, -1.0])
    assert input.grad is not None
    assert input.grad.flatten().tolist() == pytest.approx([2.0, 2.0])
    parameters = dict(model.named_parameters())
    assert sorted(parameters) == ["head.bias", "head.weight"]
    assert parameters["head.weight"].grad is not None
    assert parameters["head.weight"].grad.flatten().tolist() == pytest.approx([1.0])
    assert parameters["head.bias"].grad is not None
    assert parameters["head.bias"].grad.tolist() == pytest.approx([2.0])


def test_imported_optimizer_update_is_visible_to_the_next_call(tmp_path: Path) -> None:
    model = _load_model(tmp_path)
    optimizer = tynx.optim.SGD(model.named_parameters(), lr=0.1)
    input = tynx.Tensor([[2.0], [-1.0]])

    model(input).sum().backward()
    optimizer.step()

    assert model(input).flatten().tolist() == pytest.approx([4.6, -1.1])


def test_no_grad_imported_call_is_detached_and_plain_load_remains_inference(tmp_path: Path) -> None:
    path = _model_path(tmp_path)
    model = tynx.load(
        path,
        trainable=True,
        simplify=False,
        initializer_names={"constant1_out1": "head.weight", "constant2_out1": "head.bias"},
    )
    with tynx.no_grad():
        output = model(tynx.Tensor([[2.0], [-1.0]], requires_grad=True))

    assert output.requires_grad is False
    assert isinstance(tynx.load(path, simplify=False), tynx.Session)
