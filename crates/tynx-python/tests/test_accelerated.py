"""Accelerated-device synchronization and eager tape-lifetime acceptance tests."""

import gc
import os

import pytest
import tynx


def _accelerated_device() -> tynx.Device:
    device = tynx.get_default_device()
    if "Vulkan" in str(device) or "Wgpu" in str(device):
        return device
    if os.environ.get("TYNX_REQUIRE_ACCELERATED") == "1":
        pytest.fail(f"accelerated test lane selected a non-WGPU device: {device}")
    pytest.skip(f"accelerated backend not enabled: {device}")


def test_accelerated_boolean_construction_avoids_backend_bool_upload() -> None:
    device = _accelerated_device()

    value = tynx.Tensor([[True, False], [False, True]], dtype="bool")

    tynx.synchronize(device)
    assert value.tolist() == [[True, False], [False, True]]


def test_accelerated_tape_survives_intermediate_drop_and_optimizer_step() -> None:
    device = _accelerated_device()
    input = tynx.Tensor([[1.0, 2.0]], requires_grad=True)
    weight = tynx.Parameter([[3.0], [4.0]])
    optimizer = tynx.optim.SGD([weight], lr=0.1)

    intermediate = input @ weight
    loss = (intermediate * intermediate).sum()
    del intermediate
    gc.collect()

    tynx.synchronize(device)
    loss.backward()
    tynx.synchronize(device)

    assert input.grad is not None
    assert input.grad.flatten().tolist() == pytest.approx([66.0, 88.0])
    assert weight.grad is not None
    assert weight.grad.flatten().tolist() == pytest.approx([22.0, 44.0])

    optimizer.step()
    tynx.synchronize(device)
    with tynx.no_grad():
        updated = input.detach() @ weight
    tynx.synchronize(updated.device)
    assert updated.item() == pytest.approx(0.0, abs=1e-5)
