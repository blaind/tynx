"""Combined model-and-optimizer checkpoint coverage."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
import tynx


def _train_step(model: tynx.nn.Linear, optimizer: tynx.optim.Adam) -> None:
    optimizer.zero_grad()
    prediction = model(tynx.Tensor([[2.0], [-1.0]]))
    target = tynx.Tensor([[5.0], [-1.0]])
    tynx.nn.functional.mse_loss(prediction, target).backward()
    optimizer.step()


def _initialized_model() -> tynx.nn.Linear:
    model = tynx.nn.Linear(1, 1)
    model.weight.copy_(tynx.Tensor([[1.5]]))
    assert model.bias is not None
    model.bias.copy_(tynx.Tensor([0.5]))
    return model


def test_combined_checkpoint_resumes_model_and_adam_exactly(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.Adam(
        model.named_parameters(),
        lr=0.03,
        betas=(0.8, 0.95),
        eps=1e-6,
        amsgrad=True,
    )
    _train_step(model, optimizer)
    checkpoint = tmp_path / "training.tynx"

    tynx.save_checkpoint(checkpoint, model, optimizer)
    manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert manifest["format"] == "tynx.training"
    assert manifest["version"] == 1
    assert manifest["model"]["weight"]["dtype"] == "float32"
    assert manifest["model"]["weight"]["shape"] == [1, 1]

    _train_step(model, optimizer)
    expected = model.state_dict()

    resumed_model = tynx.nn.Linear(1, 1)
    resumed_optimizer = tynx.optim.Adam(resumed_model.named_parameters(), lr=0.9, amsgrad=True)
    result = tynx.load_checkpoint(checkpoint, resumed_model, resumed_optimizer)

    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert resumed_optimizer.lr == pytest.approx(0.03)
    assert resumed_optimizer.betas == pytest.approx((0.8, 0.95))
    _train_step(resumed_model, resumed_optimizer)
    for name, value in resumed_model.state_dict().items():
        assert value.tolist() == expected[name].tolist()


def test_checkpoint_supports_optimizer_constructed_from_parameters(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.Adam(model.parameters(), lr=0.03)
    _train_step(model, optimizer)
    checkpoint = tmp_path / "positional-optimizer.tynx"

    tynx.save_checkpoint(checkpoint, model, optimizer)
    _train_step(model, optimizer)
    expected = model.state_dict()

    resumed_model = _initialized_model()
    resumed_optimizer = tynx.optim.Adam(resumed_model.parameters(), lr=0.9)
    tynx.load_checkpoint(checkpoint, resumed_model, resumed_optimizer)
    _train_step(resumed_model, resumed_optimizer)

    for name, value in resumed_model.state_dict().items():
        assert value.tolist() == expected[name].tolist()


def test_checkpoint_optimizer_rejection_does_not_publish_model_state(tmp_path: Path) -> None:
    source = _initialized_model()
    source_optimizer = tynx.optim.Adam(source.named_parameters(), lr=0.03)
    _train_step(source, source_optimizer)
    checkpoint = tmp_path / "training.tynx"
    tynx.save_checkpoint(checkpoint, source, source_optimizer)

    destination = _initialized_model()
    destination.weight.copy_(tynx.Tensor([[9.0]]))
    before = destination.state_dict()
    wrong_optimizer = tynx.optim.AdamW(destination.named_parameters())

    with pytest.raises(ValueError, match="cannot load Adam state into AdamW"):
        tynx.load_checkpoint(checkpoint, destination, wrong_optimizer)

    for name, value in destination.state_dict().items():
        assert value.tolist() == before[name].tolist()


def test_checkpoint_rejects_unknown_versions_before_mutation(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.SGD(model.named_parameters(), lr=0.1)
    checkpoint = tmp_path / "training.tynx"
    tynx.save_checkpoint(checkpoint, model, optimizer)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["version"] = 999
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    before = model.state_dict()

    with pytest.raises(ValueError, match="unsupported Tynx checkpoint version"):
        tynx.load_checkpoint(checkpoint, model, optimizer)

    for name, value in model.state_dict().items():
        assert value.tolist() == before[name].tolist()


def test_checkpoint_rejects_models_and_payloads_with_no_state(tmp_path: Path) -> None:
    optimizer = tynx.optim.SGD([("weight", tynx.Parameter([1.0]))], lr=0.1)
    checkpoint = tmp_path / "empty.tynx"

    with pytest.raises(ValueError, match="model has no parameters or buffers"):
        tynx.save_checkpoint(checkpoint, object(), optimizer)

    checkpoint.write_text(
        json.dumps(
            {
                "format": "tynx.training",
                "version": 1,
                "model": {},
                "optimizer": optimizer.state_dict(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model state is empty"):
        tynx.load_checkpoint(checkpoint, object(), optimizer)


def test_weights_only_checkpoint_supports_both_argument_orders(tmp_path: Path) -> None:
    source = _initialized_model()
    first = tmp_path / "path-first.tynx"
    second = tmp_path / "model-first.tynx"

    tynx.save_checkpoint(first, source)
    tynx.save_checkpoint(source, second)

    for checkpoint in (first, second):
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert payload["optimizer"] is None
        destination = tynx.nn.Linear(1, 1)
        result = tynx.load_checkpoint(checkpoint, destination)
        assert result.missing_keys == ()
        assert result.unexpected_keys == ()
        for name, value in destination.state_dict().items():
            assert value.tolist() == source.state_dict()[name].tolist()


def test_checkpoint_supports_model_optimizer_path_order(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.03)
    _train_step(model, optimizer)
    checkpoint = tmp_path / "torch-order.tynx"

    tynx.save_checkpoint(model, optimizer, checkpoint)

    resumed = tynx.nn.Linear(1, 1)
    resumed_optimizer = tynx.optim.Adam(resumed.named_parameters())
    tynx.load_checkpoint(checkpoint, resumed, resumed_optimizer)
    for name, value in resumed.state_dict().items():
        assert value.tolist() == model.state_dict()[name].tolist()


def test_checkpoint_rejects_invalid_argument_order_up_front(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.Adam(model.named_parameters())

    with pytest.raises(TypeError, match="expects"):
        cast(Any, tynx.save_checkpoint)(model, optimizer)
    with pytest.raises(TypeError, match="model must be a model object"):
        tynx.save_checkpoint(tmp_path / "bad.tynx", "not-a-model", optimizer)
    with pytest.raises(TypeError, match=r"optimizer must provide state_dict\(\)"):
        cast(Any, tynx.save_checkpoint)(tmp_path / "bad.tynx", model, object())
