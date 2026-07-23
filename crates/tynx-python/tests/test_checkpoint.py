"""Combined model-and-optimizer checkpoint coverage."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
import tynx


def _train_step(model: tynx.nn.Linear, optimizer: tynx.optim.Adam) -> None:
    optimizer.zero_grad()
    prediction = model(tynx.Tensor([[2.0], [-1.0]], device=model.weight.device))
    target = tynx.Tensor([[5.0], [-1.0]], device=model.weight.device)
    tynx.nn.functional.mse_loss(prediction, target).backward()
    optimizer.step()


def _initialized_model(device: Any = None) -> tynx.nn.Linear:
    model = tynx.nn.Linear(1, 1)
    if device is not None:
        model.weight = tynx.Parameter(
            tynx.Tensor([[1.5]], device=device),
            name="weight",
        )
        model.bias = tynx.Parameter(
            tynx.Tensor([0.5], device=device),
            name="bias",
        )
    else:
        model.weight.copy_(tynx.Tensor([[1.5]]))
        assert model.bias is not None
        model.bias.copy_(tynx.Tensor([0.5]))
    return model


def _state_tensors(value: object) -> list[tynx.Tensor]:
    if isinstance(value, tynx.Tensor):
        return [value]
    if isinstance(value, dict):
        tensors: list[tynx.Tensor] = []
        for item in value.values():
            tensors.extend(_state_tensors(item))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_state_tensors(item))
        return tensors
    return []


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


def test_checkpoint_maps_model_and_optimizer_tensors_to_device(tmp_path: Path) -> None:
    source = _initialized_model()
    source_optimizer = tynx.optim.Adam(source.named_parameters(), lr=0.03)
    _train_step(source, source_optimizer)
    checkpoint = tmp_path / "device-remap.tynx"
    tynx.save_checkpoint(checkpoint, source, source_optimizer)

    cpu = tynx.Device("cpu")
    destination = _initialized_model(cpu)
    destination_optimizer = tynx.optim.Adam(
        destination.named_parameters(),
        lr=0.9,
    )
    tynx.load_checkpoint(
        checkpoint,
        destination,
        destination_optimizer,
        device=cpu,
    )

    assert all(value.device == cpu for value in destination.state_dict().values())
    optimizer_tensors = _state_tensors(destination_optimizer.state_dict())
    assert optimizer_tensors
    assert all(value.device == cpu for value in optimizer_tensors)
    _train_step(destination, destination_optimizer)


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


def test_checkpoint_resumes_scheduler_with_model_and_optimizer(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.08)
    scheduler = tynx.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    _train_step(model, optimizer)
    scheduler.step()
    checkpoint = tmp_path / "scheduled.tynx"

    tynx.save_checkpoint(checkpoint, model, optimizer, scheduler=scheduler)
    manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert manifest["scheduler"]["scheduler"] == "StepLR"

    _train_step(model, optimizer)
    scheduler.step()
    expected = model.state_dict()

    resumed_model = _initialized_model()
    resumed_optimizer = tynx.optim.Adam(resumed_model.named_parameters(), lr=0.9)
    resumed_scheduler = tynx.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=9, gamma=0.9)
    tynx.load_checkpoint(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        scheduler=resumed_scheduler,
    )

    assert resumed_scheduler.last_epoch == 1
    assert resumed_scheduler.step_size == 1
    assert resumed_scheduler.gamma == pytest.approx(0.5)
    assert resumed_optimizer.lr == pytest.approx(0.04)
    _train_step(resumed_model, resumed_optimizer)
    resumed_scheduler.step()
    for name, value in resumed_model.state_dict().items():
        assert value.tolist() == expected[name].tolist()


def test_checkpoint_scheduler_is_optional_and_backward_compatible(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.SGD(model.named_parameters(), lr=0.1)
    checkpoint = tmp_path / "legacy.tynx"
    tynx.save_checkpoint(checkpoint, model, optimizer)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload.pop("scheduler")
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    tynx.load_checkpoint(checkpoint, model, optimizer)
    scheduler = tynx.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    with pytest.raises(ValueError, match="does not contain scheduler state"):
        tynx.load_checkpoint(checkpoint, model, optimizer, scheduler=scheduler)


def test_checkpoint_rejects_scheduler_for_another_optimizer(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.SGD(model.named_parameters(), lr=0.1)
    other = tynx.optim.SGD([tynx.Parameter([1.0])], lr=0.2)
    scheduler = tynx.optim.lr_scheduler.StepLR(other, step_size=1)

    with pytest.raises(ValueError, match="must belong"):
        tynx.save_checkpoint(
            tmp_path / "wrong-scheduler.tynx",
            model,
            optimizer,
            scheduler=scheduler,
        )


def test_checkpoint_scheduler_rejection_rolls_back_optimizer(tmp_path: Path) -> None:
    source = _initialized_model()
    source_optimizer = tynx.optim.SGD(source.named_parameters(), lr=0.1)
    source_scheduler = tynx.optim.lr_scheduler.StepLR(source_optimizer, step_size=1)
    checkpoint = tmp_path / "scheduler-mismatch.tynx"
    tynx.save_checkpoint(
        checkpoint,
        source,
        source_optimizer,
        scheduler=source_scheduler,
    )

    destination = _initialized_model()
    destination.weight.copy_(tynx.Tensor([[9.0]]))
    destination_optimizer = tynx.optim.SGD(destination.named_parameters(), lr=0.7)
    destination_scheduler = tynx.optim.lr_scheduler.ExponentialLR(destination_optimizer, gamma=0.8)

    with pytest.raises(ValueError, match="cannot load StepLR state into ExponentialLR"):
        tynx.load_checkpoint(
            checkpoint,
            destination,
            destination_optimizer,
            scheduler=destination_scheduler,
        )

    assert destination.weight.item() == pytest.approx(9.0)
    assert destination_optimizer.lr == pytest.approx(0.7)
    assert destination_scheduler.last_epoch == 0
    assert destination_scheduler.gamma == pytest.approx(0.8)


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
    tynx.load_checkpoint(resumed, resumed_optimizer, checkpoint)
    for name, value in resumed.state_dict().items():
        assert value.tolist() == model.state_dict()[name].tolist()


def test_load_weights_only_checkpoint_supports_model_path_order(tmp_path: Path) -> None:
    source = _initialized_model()
    checkpoint = tmp_path / "weights-only.tynx"
    tynx.save_checkpoint(source, checkpoint)

    destination = tynx.nn.Linear(1, 1)
    result = tynx.load_checkpoint(destination, checkpoint)

    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    for name, value in destination.state_dict().items():
        assert value.tolist() == source.state_dict()[name].tolist()


def test_checkpoint_rejects_invalid_argument_order_up_front(tmp_path: Path) -> None:
    model = _initialized_model()
    optimizer = tynx.optim.Adam(model.named_parameters())

    with pytest.raises(TypeError, match="expects"):
        cast(Any, tynx.save_checkpoint)(model, optimizer)
    with pytest.raises(TypeError, match="model must be a model object"):
        tynx.save_checkpoint(tmp_path / "bad.tynx", "not-a-model", optimizer)
    with pytest.raises(TypeError, match=r"optimizer must provide state_dict\(\)"):
        cast(Any, tynx.save_checkpoint)(tmp_path / "bad.tynx", model, object())
    with pytest.raises(TypeError, match="expects"):
        cast(Any, tynx.load_checkpoint)(model, optimizer)
    with pytest.raises(TypeError, match="model must be a model object"):
        cast(Any, tynx.load_checkpoint)(tmp_path / "bad.tynx", "not-a-model", optimizer)
    with pytest.raises(TypeError, match=r"optimizer must provide state_dict\(\)"):
        cast(Any, tynx.load_checkpoint)(tmp_path / "bad.tynx", model, object())
