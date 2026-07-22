"""Fixed-rollout PPO acceptance over a multi-output imported actor-critic."""

import math
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


def test_imported_actor_critic_trains_with_user_composed_ppo_loss(tmp_path: Path) -> None:
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
    assert len(model.outputs) == 2
    logits_name, values_name = model.outputs
    report = model.trainability_report()
    assert sorted(report.output_parameters[logits_name]) == ["constant1_out1", "constant2_out1"]
    assert sorted(report.output_parameters[values_name]) == ["constant3_out1", "constant4_out1"]

    observations = tynx.Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    actions = tynx.Tensor([0, 1, 0, 1], dtype="int64")
    returns = tynx.Tensor([1.0, -1.0, 0.0, -2.0])
    advantages = tynx.Tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True).detach()

    initial_outputs = model(observations)
    assert isinstance(initial_outputs, dict)
    initial_policy = tynx.distributions.Categorical(logits=initial_outputs[logits_name])
    old_log_prob = initial_policy.log_prob(actions).detach()
    assert old_log_prob.tolist() == pytest.approx([-math.log(2.0)] * 4)
    assert initial_policy.sample(seed=17).shape == (4,)

    def ppo_loss() -> tuple[tynx.Tensor, tynx.Tensor]:
        outputs = model(observations)
        assert isinstance(outputs, dict)
        policy = tynx.distributions.Categorical(logits=outputs[logits_name])
        log_prob = policy.log_prob(actions)
        ratio = (log_prob - old_log_prob).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(0.8, 1.2) * advantages
        policy_loss = -tynx.minimum(unclipped, clipped).mean()
        values = outputs[values_name].squeeze(1)
        value_loss = tynx.nn.functional.mse_loss(values, returns)
        entropy_bonus = policy.entropy().mean()
        return policy_loss + value_loss * 0.5 - entropy_bonus * 0.01, value_loss

    initial_loss, initial_value_loss = ppo_loss()
    initial_loss_value = initial_loss.item()
    initial_value_loss_value = initial_value_loss.item()
    optimizer = tynx.optim.Adam(model.named_parameters(), lr=0.03)
    for _ in range(120):
        optimizer.zero_grad()
        loss, _ = ppo_loss()
        loss.backward()
        optimizer.step()

    final_loss, final_value_loss = ppo_loss()
    final_outputs = model(observations)
    assert isinstance(final_outputs, dict)
    chosen_probabilities = (
        tynx.distributions.Categorical(logits=final_outputs[logits_name])
        .probs.gather(1, actions.unsqueeze(1))
        .mean()
        .item()
    )

    assert final_loss.item() < initial_loss_value
    assert final_value_loss.item() < initial_value_loss_value * 0.01
    assert chosen_probabilities > 0.55
