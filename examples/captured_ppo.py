"""Capture a complete imported actor-critic PPO update."""

import math
import tempfile
from pathlib import Path

import tynx


def main() -> None:
    fixture = Path(__file__).parent / "models" / "actor_critic.onnx.hex"
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "actor_critic.onnx"
        model_path.write_bytes(bytes.fromhex(fixture.read_text().strip()))
        model = tynx.load(
            model_path,
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
        losses = [
            update(observations, actions, returns, advantages, old_log_prob).item()
            for _ in range(40)
        ]

    print(f"captured PPO: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"Python bodies: {calls}; native replays: {update.replay_count}")


if __name__ == "__main__":
    main()
