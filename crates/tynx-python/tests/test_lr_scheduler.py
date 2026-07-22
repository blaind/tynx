from __future__ import annotations

import pytest
import tynx
from tynx.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR


def _grouped_optimizer() -> tynx.optim.SGD:
    return tynx.optim.SGD(
        [
            {"params": [tynx.Parameter([1.0])], "lr": 1.0},
            {"params": [tynx.Parameter([1.0])], "lr": 0.5},
        ],
        lr=0.25,
    )


def test_step_lr_updates_each_parameter_group_at_boundaries() -> None:
    optimizer = _grouped_optimizer()
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    assert scheduler.last_epoch == 0
    assert scheduler.get_last_lr() == pytest.approx([1.0, 0.5])
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1.0, 0.5])
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.1, 0.05])


def test_exponential_lr_changes_the_native_optimizer_step_rate() -> None:
    parameter = tynx.Parameter([2.0])
    optimizer = tynx.optim.SGD([parameter], lr=0.2)
    scheduler = ExponentialLR(optimizer, gamma=0.5)

    scheduler.step()
    parameter.backward()
    optimizer.step()

    assert optimizer.lr == pytest.approx(0.1)
    assert parameter.item() == pytest.approx(1.9)


def test_cosine_annealing_reaches_eta_min() -> None:
    optimizer = tynx.optim.SGD([tynx.Parameter([1.0])], lr=1.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=0.2)

    scheduler.step()
    assert optimizer.lr == pytest.approx(0.6)
    scheduler.step()
    assert optimizer.lr == pytest.approx(0.2)


def test_scheduler_state_restores_epoch_and_group_rates() -> None:
    source_optimizer = _grouped_optimizer()
    source = StepLR(source_optimizer, step_size=2, gamma=0.1)
    source.step()
    source.step()

    destination_optimizer = _grouped_optimizer()
    destination = StepLR(destination_optimizer, step_size=7, gamma=0.9)
    destination.load_state_dict(source.state_dict())

    assert destination.last_epoch == 2
    assert destination.get_last_lr() == pytest.approx([0.1, 0.05])
    assert destination.step_size == 2
    assert destination.gamma == pytest.approx(0.1)
    assert [group["lr"] for group in destination_optimizer.param_groups] == pytest.approx(
        [0.1, 0.05]
    )


def test_schedulers_validate_configuration() -> None:
    optimizer = tynx.optim.SGD([tynx.Parameter([1.0])], lr=0.1)
    with pytest.raises(ValueError, match="step_size"):
        StepLR(optimizer, step_size=0)
    with pytest.raises(ValueError, match="gamma"):
        ExponentialLR(optimizer, gamma=-1.0)
    with pytest.raises(ValueError, match="T_max"):
        CosineAnnealingLR(optimizer, T_max=0)
