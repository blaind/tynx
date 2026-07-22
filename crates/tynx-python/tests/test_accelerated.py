"""Accelerated-device synchronization and eager tape-lifetime acceptance tests."""

import gc
import math
import os
from pathlib import Path

import pytest
import tynx

_ACTOR_CRITIC_MODEL = bytes.fromhex(
    "08084202100d3ae3020a400a0c6f62736572766174696f6e730a0d706f6c6963795f776569"
    "6768740a0b706f6c6963795f6269617312066c6f676974731a06706f6c696379220447656d"
    "6d0a3d0a0c6f62736572766174696f6e730a0c76616c75655f7765696768740a0a76616c75"
    "655f62696173120676616c7565731a0576616c7565220447656d6d120c6163746f725f637269"
    "7469632a270a0202021001221000000000000000000000000000000000420d706f6c696379"
    "5f7765696768742a1c0a0102100122080000000000000000420b706f6c6963795f626961732a"
    "1e0a020201100122080000000000000000420c76616c75655f7765696768742a170a01011001"
    "220400000000420a76616c75655f626961735a1e0a0c6f62736572766174696f6e73120e0a0c"
    "080112080a0208040a02080262180a066c6f67697473120e0a0c080112080a0208040a020802"
    "62180a0676616c756573120e0a0c080112080a0208040a020801"
)


def _accelerated_device() -> tynx.Device:
    device = tynx.get_default_device()
    if "Vulkan" in str(device) or "Wgpu" in str(device):
        return device
    if os.environ.get("TYNX_REQUIRE_ACCELERATED") == "1":
        pytest.fail(f"accelerated test lane selected a non-WGPU device: {device}")
    pytest.skip(f"accelerated backend not enabled: {device}")


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


def test_accelerated_captured_imported_ppo_step_reuses_updated_weights(tmp_path: Path) -> None:
    device = _accelerated_device()
    path = tmp_path / "actor_critic.onnx"
    path.write_bytes(_ACTOR_CRITIC_MODEL)
    model = tynx.load(
        path,
        trainable="auto",
        simplify=False,
        initializer_names={
            "constant1_out1": "policy.weight",
            "constant2_out1": "policy.bias",
            "constant3_out1": "value.weight",
            "constant4_out1": "value.bias",
        },
    )
    assert isinstance(model, tynx.ImportedModel)
    logits_name, values_name = model.outputs
    optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.03)
    calls = 0

    @tynx.compile(fullgraph=True)
    def update(
        observations: tynx.Tensor,
        actions: tynx.Tensor,
        returns: tynx.Tensor,
        advantages: tynx.Tensor,
        old_log_prob: tynx.Tensor,
    ) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        outputs = model(observations)
        assert isinstance(outputs, dict)
        policy = tynx.distributions.Categorical(logits=outputs[logits_name])
        ratio = (policy.log_prob(actions) - old_log_prob).exp()
        policy_loss = -tynx.minimum(
            ratio * advantages,
            ratio.clamp(0.8, 1.2) * advantages,
        ).mean()
        value_loss = tynx.nn.functional.mse_loss(outputs[values_name].squeeze(1), returns)
        loss = policy_loss + value_loss * 0.5 - policy.entropy().mean() * 0.01
        loss.backward()
        optimizer.step()
        return loss

    observations = tynx.Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    actions = tynx.Tensor([0, 1, 0, 1], dtype="int64")
    returns = tynx.Tensor([1.0, -1.0, 0.0, -2.0])
    advantages = tynx.Tensor([1.0, 1.0, 1.0, 1.0])
    old_log_prob = tynx.Tensor([-math.log(2.0)] * 4)
    initial = model(observations)
    assert isinstance(initial, dict)
    initial_values = initial[values_name].flatten().tolist()

    for _ in range(3):
        update(observations, actions, returns, advantages, old_log_prob)
        tynx.synchronize(device)

    final = model(observations)
    assert isinstance(final, dict)
    tynx.synchronize(device)
    assert final[values_name].flatten().tolist() != pytest.approx(initial_values)
    assert calls == 1
    assert update.compile_count == 1
    assert update.replay_count == 2
