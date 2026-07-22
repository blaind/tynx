"""PyTorch-style optimizer parameter-group behavior."""

import pytest
import tynx


def _backward_pair(first: tynx.Parameter, second: tynx.Parameter) -> None:
    (first + second).sum().backward()


def test_sgd_parameter_groups_apply_independent_and_mutable_learning_rates() -> None:
    first = tynx.Parameter([2.0])
    second = tynx.Parameter([2.0])
    optimizer = tynx.optim.SGD(
        [
            {"params": [first], "lr": 0.1},
            {"params": [second], "lr": 0.01, "weight_decay": 0.0},
        ],
        lr=0.5,
        momentum=0.0,
    )

    _backward_pair(first, second)
    optimizer.step()
    assert first.item() == pytest.approx(1.9)
    assert second.item() == pytest.approx(1.99)
    assert optimizer.parameter_count == 2
    assert len(optimizer.param_groups) == 2

    optimizer.zero_grad()
    _backward_pair(first, second)
    optimizer.param_groups[1]["lr"] = 0.2
    optimizer.step()
    assert first.item() == pytest.approx(1.8)
    assert second.item() == pytest.approx(1.79)

    optimizer.lr = 0.3
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.3, 0.3])


def test_parameter_group_option_mutation_updates_the_native_configuration() -> None:
    parameter = tynx.Parameter([2.0])
    optimizer = tynx.optim.SGD([{"params": [parameter]}], lr=0.1)
    optimizer.param_groups[0]["weight_decay"] = 0.5

    parameter.backward()
    optimizer.step()

    assert parameter.item() == pytest.approx(1.8)
    assert optimizer.weight_decay == pytest.approx(0.5)


def test_parameter_groups_reject_cross_group_duplicates_and_unknown_options() -> None:
    parameter = tynx.Parameter([1.0])
    with pytest.raises(ValueError, match="more than one"):
        tynx.optim.Adam(
            [{"params": [parameter]}, {"params": [parameter]}],
        )
    with pytest.raises(
        ValueError,
        match=r"unsupported options \['learning_rate'\].*supported options.*'lr'.*'momentum'",
    ):
        tynx.optim.SGD([{"params": [parameter], "learning_rate": 0.1}], lr=0.1)


def test_grouped_adam_state_resumes_each_native_group_exactly() -> None:
    first = tynx.Parameter([2.0])
    second = tynx.Parameter([-1.0])
    optimizer = tynx.optim.Adam(
        [
            {"params": [("first", first)], "lr": 0.1, "betas": (0.8, 0.9)},
            {"params": [("second", second)], "lr": 0.02, "betas": (0.7, 0.95)},
        ],
        lr=0.5,
    )
    _backward_pair(first, second)
    optimizer.step()
    first_snapshot = first.item()
    second_snapshot = second.item()
    state = optimizer.state_dict()
    assert state["version"] == 2

    optimizer.zero_grad()
    _backward_pair(first, second)
    optimizer.step()
    expected = (first.item(), second.item())

    resumed_first = tynx.Parameter([first_snapshot])
    resumed_second = tynx.Parameter([second_snapshot])
    resumed = tynx.optim.Adam(
        [
            {"params": [("first", resumed_first)], "lr": 0.9, "betas": (0.8, 0.9)},
            {"params": [("second", resumed_second)], "lr": 0.8, "betas": (0.7, 0.95)},
        ]
    )
    resumed.load_state_dict(state)
    assert [group["lr"] for group in resumed.param_groups] == pytest.approx([0.1, 0.02])
    _backward_pair(resumed_first, resumed_second)
    resumed.step()
    assert (resumed_first.item(), resumed_second.item()) == pytest.approx(expected)
