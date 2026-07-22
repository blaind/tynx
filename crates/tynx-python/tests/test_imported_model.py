"""Callable imported ONNX training models."""

from pathlib import Path
from typing import Protocol, cast

import pytest
import tynx


class _TrainableModel(Protocol):
    def __call__(self, input: tynx.Tensor) -> tynx.Tensor: ...

    def named_parameters(self) -> list[tuple[str, tynx.Parameter]]: ...


_GEMM_MODEL = bytes.fromhex(
    "08084202100d3a8c010a200a01780a067765696768740a04626961731201791a0468656164"
    "220447656d6d1215707974686f6e5f696d706f727465645f6d6f64656c2a140a0201011001"
    "22040000004042067765696768742a110a0101100122040000803f4204626961735a130a0178"
    "120e0a0c080112080a0208020a02080162130a0179120e0a0c080112080a0208020a020801"
)
_MULTI_MODEL = bytes.fromhex(
    "08084202100d3ab5010a1d0a01780a046d61736b120373756d1a0873756d5f6e6f64652203"
    "4164640a210a01780a046d61736b120770726f647563741a086d756c5f6e6f646522034d756c"
    "1212707974686f6e5f6d756c74695f6d6f64656c5a130a0178120e0a0c080112080a020802"
    "0a0208015a160a046d61736b120e0a0c080112080a0208020a02080162150a0373756d120e0a"
    "0c080112080a0208020a02080162190a0770726f64756374120e0a0c080112080a0208020a02"
    "0801"
)
_MATMUL_MODEL = bytes.fromhex(
    "080d3a7e0a260a01780a0e656e636f6465722e7765696768741201791a066d61746d756c"
    "22064d61744d756c120c6d61746d756c5f6d6f64656c2a1c080108011001220400000040"
    "420e656e636f6465722e7765696768745a130a0178120e0a0c080112080a0208020a020801"
    "62130a0179120e0a0c080112080a0208020a02080142040a00100d"
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
        initializer_names={"weight": "head.weight", "bias": "head.bias"},
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


def test_imported_model_checkpoint_restores_weights_and_optimizer_state(tmp_path: Path) -> None:
    model = _load_model(tmp_path)
    optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.03)
    input = tynx.Tensor([[2.0], [-1.0]])

    model(input).sum().backward()
    optimizer.step()
    saved = {name: value.tolist() for name, value in model.state_dict().items()}
    checkpoint = tmp_path / "imported.tynx"
    tynx.save_checkpoint(checkpoint, model, optimizer)

    resumed = _load_model(tmp_path)
    resumed_optimizer = tynx.optim.Adam(resumed.named_parameters(), lr=0.9)
    result = tynx.load_checkpoint(checkpoint, resumed, resumed_optimizer)

    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert resumed_optimizer.lr == pytest.approx(0.03)
    assert {name: value.tolist() for name, value in resumed.state_dict().items()} == saved


def test_imported_model_state_dict_strictly_validates_keys(tmp_path: Path) -> None:
    model = _load_model(tmp_path)
    state = model.state_dict()
    assert sorted(state) == ["head.bias", "head.weight"]

    with pytest.raises(ValueError, match="state_dict key mismatch"):
        model.load_state_dict({"head.weight": state["head.weight"]})


def test_no_grad_imported_call_is_detached_and_plain_load_remains_inference(tmp_path: Path) -> None:
    path = _model_path(tmp_path)
    model = tynx.load(
        path,
        trainable=True,
        simplify=False,
        initializer_names={"weight": "head.weight", "bias": "head.bias"},
    )
    with tynx.no_grad():
        output = model(tynx.Tensor([[2.0], [-1.0]], requires_grad=True))

    assert output.requires_grad is False
    session = tynx.load(path, simplify=False)
    assert isinstance(session, tynx.Session)
    input = tynx.Tensor([[2.0], [-1.0]], requires_grad=True)
    positional = session(input)
    named = session.run(x=input)
    assert isinstance(positional, tynx.Tensor)
    assert isinstance(named, tynx.Tensor)
    assert positional.flatten().tolist() == pytest.approx([5.0, -1.0])
    assert named.flatten().tolist() == pytest.approx([5.0, -1.0])


def test_imported_trainability_report_is_structured_and_output_specific(tmp_path: Path) -> None:
    model = _load_model(tmp_path)

    report = model.trainability_report()
    output_name = model.outputs[0]
    assert isinstance(report, tynx.TrainabilityReport)
    assert report.is_trainable is True
    assert report.selected_outputs == [output_name]
    assert sorted(report.trainable_parameters) == ["bias", "weight"]
    assert sorted(report.output_parameters[output_name]) == ["bias", "weight"]
    assert report.backward_issues == []
    assert {entry["role"] for entry in report.initializers}.issuperset({"parameter"})
    assert "Trainable parameters" in str(report)
    report.require_trainable()

    selected = model.require_trainable(outputs=[output_name])
    assert selected.selected_outputs == [output_name]
    with pytest.raises(ValueError, match=r"requested output.*not a declared"):
        model.require_trainable(outputs=["missing"])


def test_simplified_imported_trainability_uses_declared_output_names(tmp_path: Path) -> None:
    model = tynx.load(
        _model_path(tmp_path),
        trainable="auto",
        simplify=True,
        initializer_names={"weight": "head.weight", "bias": "head.bias"},
    )

    assert model.outputs == ["y"]
    report = model.trainability_report()
    assert report.selected_outputs == ["y"]
    assert "y" in report.output_parameters

    selected = model.require_trainable(outputs=["y"])
    assert selected.selected_outputs == ["y"]
    assert "y" in selected.output_parameters


@pytest.mark.parametrize("simplify", [False, True])
def test_matmul_initializer_preserves_onnx_name_without_override(
    tmp_path: Path, simplify: bool
) -> None:
    path = tmp_path / "matmul.onnx"
    path.write_bytes(_MATMUL_MODEL)

    model = tynx.load(path, trainable="auto", simplify=simplify)

    assert [name for name, _ in model.named_parameters()] == ["encoder.weight"]
    report = model.trainability_report()
    assert report.trainable_parameters == ["encoder.weight"]
    assert report.initializers[0]["synthetic_name"] is False
    assert model(tynx.Tensor([[3.0], [4.0]])).tolist() == [[6.0], [8.0]]


def test_imported_model_binds_multiple_named_inputs_and_returns_named_outputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "multi.onnx"
    path.write_bytes(_MULTI_MODEL)
    model = tynx.load(path, trainable="auto", simplify=False)
    first = tynx.Tensor([[1.0], [2.0]], requires_grad=True)
    second = tynx.Tensor([[3.0], [4.0]], requires_grad=True)

    positional = model(first, second)
    assert isinstance(positional, dict)
    assert list(positional) == model.outputs
    assert positional[model.outputs[0]].flatten().tolist() == pytest.approx([4.0, 6.0])
    assert positional[model.outputs[1]].flatten().tolist() == pytest.approx([3.0, 8.0])

    named = model(**{model.inputs[1]: second, model.inputs[0]: first})
    assert isinstance(named, dict)
    loss = named[model.outputs[0]].sum() + named[model.outputs[1]].sum()
    loss.backward()
    assert first.grad is not None
    assert second.grad is not None
    assert first.grad.flatten().tolist() == pytest.approx([4.0, 5.0])
    assert second.grad.flatten().tolist() == pytest.approx([2.0, 3.0])

    with pytest.raises(TypeError, match="missing required inputs"):
        model(first)
    with pytest.raises(TypeError, match="unexpected input"):
        model(first, unknown=second)
    with pytest.raises(TypeError, match="multiple values"):
        model(first, **{model.inputs[0]: second, model.inputs[1]: second})


def test_one_python_loop_trains_authored_and_imported_models(tmp_path: Path) -> None:
    authored = tynx.nn.Linear(1, 1)
    authored.weight.copy_(tynx.Tensor([[0.0]]))
    assert authored.bias is not None
    authored.bias.copy_(tynx.Tensor([0.0]))
    imported = _load_model(tmp_path)
    for _, parameter in imported.named_parameters():
        parameter.copy_(tynx.Tensor([[0.0]]) if parameter.ndim == 2 else tynx.Tensor([0.0]))

    inputs = tynx.Tensor([[-1.0], [2.0]])
    targets = tynx.Tensor([[-5.0], [4.0]])

    def train(model: _TrainableModel) -> tynx.Tensor:
        optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.05)
        for _ in range(300):
            optimizer.zero_grad()
            loss = tynx.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
        return model(inputs)

    assert train(authored).flatten().tolist() == pytest.approx([-5.0, 4.0], abs=1e-3)
    assert train(cast(_TrainableModel, imported)).flatten().tolist() == pytest.approx(
        [-5.0, 4.0], abs=1e-3
    )
