from pathlib import Path

import pytest
import tynx


def test_compile_replays_linear_relu_without_python_dispatch() -> None:
    weight = tynx.Parameter([[2.0, 0.0], [0.0, 1.0]], name="weight")
    bias = tynx.Parameter([[1.0, -1.0]], name="bias")
    calls = 0

    def forward(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return (input @ weight + bias).relu()

    compiled = tynx.compile(forward, fullgraph=True)
    first = compiled(tynx.Tensor([[1.0, -2.0], [3.0, 4.0]]))
    second = compiled(tynx.Tensor([[2.0, 1.0], [-1.0, 3.0]]))

    assert first.tolist() == [[3.0, 0.0], [7.0, 3.0]]
    assert second.tolist() == [[5.0, 0.0], [0.0, 2.0]]
    assert calls == 1
    assert compiled.compile_count == 1
    assert compiled.graph_count == 1
    assert compiled.replay_count == 1
    assert compiled.node_counts == (6,)


def test_compile_replays_softmax_without_fallback() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def policy(logits: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return logits.softmax(-1)

    first = policy(tynx.Tensor([[1.0, 2.0, 3.0]]))
    second = policy(tynx.Tensor([[3.0, 2.0, 1.0]]))

    assert first.tolist()[0] == pytest.approx([0.09003057, 0.24472848, 0.66524094])
    assert second.tolist()[0] == pytest.approx([0.66524094, 0.24472848, 0.09003057])
    assert calls == 1
    assert policy.compile_count == 1
    assert policy.fallback_count == 0
    assert policy.replay_count == 1


def test_compile_preserves_nested_multi_output_structure() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> dict[str, object]:
        nonlocal calls
        calls += 1
        activated = input.relu()
        return {"raw": input, "derived": (activated, [activated.sigmoid()])}

    first = forward(tynx.Tensor([-1.0, 2.0]))
    second = forward(tynx.Tensor([3.0, -4.0]))

    first_raw = first["raw"]
    second_raw = second["raw"]
    assert isinstance(first_raw, tynx.Tensor)
    assert first_raw.tolist() == [-1.0, 2.0]
    assert isinstance(second_raw, tynx.Tensor)
    assert second_raw.tolist() == [3.0, -4.0]
    derived = second["derived"]
    assert isinstance(derived, tuple)
    assert isinstance(derived[0], tynx.Tensor)
    assert derived[0].tolist() == [3.0, 0.0]
    assert isinstance(derived[1], list)
    assert isinstance(derived[1][0], tynx.Tensor)
    assert derived[1][0].tolist() == pytest.approx([0.95257413, 0.5])
    assert calls == 1
    assert forward.compile_count == 1
    assert forward.replay_count == 1


def test_compile_replays_integer_index_gather_and_gradient() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def select(logits: tynx.Tensor, actions: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return logits.gather(1, actions.unsqueeze(1)).squeeze(1)

    first_logits = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    first = select(first_logits, tynx.Tensor([0, 1], dtype="int64"))
    assert first.tolist() == [1.0, 4.0]

    second_logits = tynx.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    second = select(second_logits, tynx.Tensor([1, 0], dtype="int64"))
    assert second.tolist() == [6.0, 7.0]
    second.sum().backward()
    assert second_logits.grad is not None
    assert second_logits.grad.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert calls == 1
    assert select.compile_count == 1
    assert select.replay_count == 1


def test_compile_replays_ppo_math_primitives() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def ppo_loss(
        logits: tynx.Tensor,
        actions: tynx.Tensor,
        old_log_prob: tynx.Tensor,
        advantages: tynx.Tensor,
        values: tynx.Tensor,
        returns: tynx.Tensor,
    ) -> tuple[tynx.Tensor, tynx.Tensor]:
        nonlocal calls
        calls += 1
        policy = tynx.distributions.Categorical(logits=logits)
        log_prob = policy.log_prob(actions)
        ratio = (log_prob - old_log_prob).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(0.8, 1.2) * advantages
        policy_loss = -tynx.minimum(unclipped, clipped).mean()
        value_loss = tynx.nn.functional.mse_loss(values, returns)
        loss = policy_loss + value_loss * 0.5 - policy.entropy().mean() * 0.01
        return loss, value_loss

    actions = tynx.Tensor([0, 1], dtype="int64")
    old_log_prob = tynx.Tensor([-0.6931472, -0.6931472])
    advantages = tynx.Tensor([1.0, -0.5])
    returns = tynx.Tensor([1.0, -1.0])
    ppo_loss(
        tynx.Tensor([[0.0, 0.0], [0.0, 0.0]]),
        actions,
        old_log_prob,
        advantages,
        tynx.Tensor([0.0, 0.0]),
        returns,
    )

    logits = tynx.Tensor([[0.2, -0.1], [-0.3, 0.4]], requires_grad=True)
    loss, value_loss = ppo_loss(
        logits,
        actions,
        old_log_prob,
        advantages,
        tynx.Tensor([0.5, -0.25]),
        returns,
    )
    assert loss.item() == pytest.approx(-0.043811, abs=1e-5)
    assert value_loss.item() == pytest.approx(0.40625)
    loss.backward()
    assert logits.grad is not None
    assert all(value == value for value in logits.grad.flatten().tolist())
    assert calls == 1
    assert ppo_loss.replay_count == 1


def test_compile_replay_preserves_input_and_parameter_gradients() -> None:
    weight = tynx.Parameter([[2.0], [-1.0]], name="weight")

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        return input @ weight

    first_input = tynx.Tensor([[1.0, 3.0]], requires_grad=True)
    forward(first_input).sum().backward()
    assert first_input.grad is not None
    assert first_input.grad.tolist() == [[2.0, -1.0]]
    assert weight.grad is not None
    assert weight.grad.tolist() == [[1.0], [3.0]]

    first_input.zero_grad()
    weight.zero_grad()
    second_input = tynx.Tensor([[4.0, -2.0]], requires_grad=True)
    forward(second_input).sum().backward()
    assert second_input.grad is not None
    assert second_input.grad.tolist() == [[2.0, -1.0]]
    assert weight.grad is not None
    assert weight.grad.tolist() == [[4.0], [-2.0]]


def test_compatible_optimizer_updates_do_not_recompile() -> None:
    weight = tynx.Parameter([[1.0]], name="weight")
    optimizer = tynx.optim.SGD([weight], lr=0.1)

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        return input @ weight

    input = tynx.Tensor([[2.0]])
    loss = forward(input).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    assert forward(input).item() == pytest.approx(1.6)
    assert forward.compile_count == 1


def test_exact_shape_changes_create_a_second_graph() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return input.relu()

    forward(tynx.Tensor([-1.0, 2.0]))
    forward(tynx.Tensor([[-1.0, 2.0]]))
    forward(tynx.Tensor([3.0, -4.0]))

    assert calls == 2
    assert forward.compile_count == 2
    assert forward.graph_count == 2


def test_unsupported_operation_falls_back_for_whole_function() -> None:
    calls = 0

    def forward(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return abs(input)

    compiled = tynx.compile(forward)
    with pytest.warns(RuntimeWarning, match="fell back to eager"):
        assert compiled(tynx.Tensor([-1.0])).tolist() == [1.0]
    assert compiled(tynx.Tensor([-2.0])).tolist() == [2.0]
    assert calls == 2
    assert compiled.compile_count == 0
    assert compiled.fallback_count == 2
    assert compiled.last_fallback_reason is not None
    compiled.clear_cache()
    assert compiled.last_fallback_reason is None


def test_fullgraph_rejects_unsupported_operation_visibly() -> None:
    def function(input: tynx.Tensor) -> tynx.Tensor:
        return abs(input)

    compiled = tynx.compile(function, fullgraph=True)

    with pytest.raises(RuntimeError, match=r"fullgraph=True.*cannot capture"):
        compiled(tynx.Tensor([1.0]))


def test_tensor_dependent_python_control_flow_falls_back() -> None:
    calls = 0

    def forward(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return input.relu() if input else -input

    compiled = tynx.compile(forward)
    with pytest.warns(RuntimeWarning, match="fell back to eager"):
        assert compiled(tynx.Tensor([2.0])).tolist() == [2.0]
    assert compiled(tynx.Tensor([-3.0])).tolist() == [0.0]
    assert calls == 2
    assert compiled.compile_count == 0


def test_fullgraph_rejects_host_reads() -> None:
    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        return input.relu() if input.item() > 0 else -input

    with pytest.raises(RuntimeError, match=r"Tensor.item.*host reads"):
        forward(tynx.Tensor([1.0]))


def test_fullgraph_rejects_state_mutation_before_publication() -> None:
    state = tynx.Buffer([7.0], name="state")

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        state.copy_(input)
        return input.relu()

    with pytest.raises(RuntimeError, match=r"state mutation.*copy_"):
        forward(tynx.Tensor([3.0]))
    assert state.tolist() == [7.0]


def test_captured_dropout_advances_and_replays_the_eager_rng_stream() -> None:
    eager_layer = tynx.nn.Dropout(0.5)
    input = tynx.Tensor([1.0] * 32)
    tynx.manual_seed(314159)
    eager = [eager_layer(input).tolist() for _ in range(6)]

    calls = 0
    captured_layer = tynx.nn.Dropout(0.5)

    @tynx.compile(fullgraph=True)
    def forward(value: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return captured_layer(value)

    tynx.manual_seed(314159)
    captured = [forward(input).tolist() for _ in range(6)]

    assert captured == eager
    assert len({tuple(sample) for sample in captured}) > 1
    assert calls == 1
    assert forward.replay_count == 5


def test_captured_categorical_sampling_advances_and_matches_eager_rng() -> None:
    logits = tynx.Tensor([[0.1, 0.2, 0.7]] * 24)
    tynx.manual_seed(271828)
    eager = [tynx.distributions.Categorical(logits=logits).sample().tolist() for _ in range(6)]
    calls = 0

    @tynx.compile(fullgraph=True)
    def sample(value: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return tynx.distributions.Categorical(logits=value).sample()

    tynx.manual_seed(271828)
    captured = [sample(logits).tolist() for _ in range(6)]

    assert captured == eager
    assert len({tuple(result) for result in captured}) > 1
    assert calls == 1
    assert sample.compile_count == 1
    assert sample.replay_count == 5


def test_captured_normal_sampling_advances_matches_eager_and_stays_detached() -> None:
    loc = tynx.Tensor([0.0] * 32, requires_grad=True)
    scale = tynx.Tensor([1.0] * 32, requires_grad=True)
    tynx.manual_seed(161803)
    eager = [tynx.distributions.Normal(loc, scale).sample() for _ in range(6)]
    calls = 0

    @tynx.compile(fullgraph=True)
    def sample(mean: tynx.Tensor, stddev: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return tynx.distributions.Normal(mean, stddev).sample()

    tynx.manual_seed(161803)
    captured = [sample(loc, scale) for _ in range(6)]

    assert [value.tolist() for value in captured] == [value.tolist() for value in eager]
    assert len({tuple(value.tolist()) for value in captured}) > 1
    assert all(not value.requires_grad for value in captured)
    assert calls == 1
    assert sample.compile_count == 1
    assert sample.replay_count == 5


def test_declared_static_arguments_guard_separate_graphs() -> None:
    calls = 0

    @tynx.compile(fullgraph=True, static_argnames=("activation",))
    def forward(input: tynx.Tensor, activation: str = "relu") -> tynx.Tensor:
        nonlocal calls
        calls += 1
        if activation == "relu":
            return input.relu()
        return input.sigmoid()

    assert forward(tynx.Tensor([-1.0, 2.0]), activation="relu").tolist() == [0.0, 2.0]
    assert forward(tynx.Tensor([3.0, -4.0]), activation="relu").tolist() == [3.0, 0.0]
    sigmoid = forward(tynx.Tensor([0.0]), activation="sigmoid")
    assert sigmoid.item() == pytest.approx(0.5)
    assert calls == 2
    assert forward.compile_count == 2
    assert forward.graph_count == 2


def test_non_tensor_argument_must_be_declared_static() -> None:
    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor, enabled: bool) -> tynx.Tensor:
        return input.relu() if enabled else -input

    with pytest.raises(RuntimeError, match="declare it with static_argnames"):
        forward(tynx.Tensor([1.0]), True)


def test_unknown_or_unhashable_static_arguments_are_visible() -> None:
    def forward(input: tynx.Tensor, options: object) -> tynx.Tensor:
        return input.relu()

    with pytest.raises(ValueError, match="unknown static_argnames"):
        tynx.compile(forward, static_argnames=("missing",))

    compiled = tynx.compile(forward, static_argnames=("options",), fullgraph=True)
    with pytest.raises(RuntimeError, match="unhashable static argument 'options'"):
        compiled(tynx.Tensor([1.0]), [])


def test_undeclared_closure_values_are_frozen_at_capture_time() -> None:
    use_relu = True
    calls = 0

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return input.relu() if use_relu else input.sigmoid()

    assert forward(tynx.Tensor([-1.0])).item() == 0.0
    use_relu = False
    assert forward(tynx.Tensor([-2.0])).item() == 0.0
    assert calls == 1


def test_authored_sequential_model_captures_parameter_first_operations() -> None:
    model = tynx.nn.Sequential(
        tynx.nn.Linear(2, 3),
        tynx.nn.ReLU(),
        tynx.nn.Linear(3, 1),
    )
    calls = 0

    @tynx.compile(fullgraph=True)
    def forward(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return model(input)

    first = forward(tynx.Tensor([[1.0, -2.0], [3.0, 4.0]]))
    second = forward(tynx.Tensor([[1.0, -2.0], [3.0, 4.0]]))
    assert second.flatten().tolist() == pytest.approx(first.flatten().tolist())
    assert calls == 1
    assert forward.compile_count == 1

    second.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_whole_sgd_step_replay_matches_eager_updates() -> None:
    eager_weight = tynx.Parameter([[0.5]], name="weight")
    captured_weight = tynx.Parameter([[0.5]], name="weight")
    eager_optimizer = tynx.optim.SGD([eager_weight], lr=0.05, momentum=0.8)
    captured_optimizer = tynx.optim.SGD([captured_weight], lr=0.05, momentum=0.8)
    calls = 0

    def eager_step(input: tynx.Tensor, target: tynx.Tensor) -> tynx.Tensor:
        eager_optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(input @ eager_weight, target)
        loss.backward()
        eager_optimizer.step()
        return loss

    @tynx.compile(fullgraph=True)
    def captured_step(input: tynx.Tensor, target: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        captured_optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(input @ captured_weight, target)
        loss.backward()
        captured_optimizer.step()
        return loss

    input = tynx.Tensor([[1.0], [2.0], [3.0]])
    target = tynx.Tensor([[2.0], [4.0], [6.0]])
    for _ in range(40):
        eager_loss = eager_step(input, target)
        captured_loss = captured_step(input, target)
        assert captured_loss.item() == pytest.approx(eager_loss.item(), abs=1.0e-6)
        assert captured_weight.item() == pytest.approx(eager_weight.item(), abs=1.0e-6)

    assert calls == 1
    assert captured_step.compile_count == 1
    assert captured_step.replay_count == 39
    assert captured_weight.item() == pytest.approx(2.0, abs=0.05)


def test_whole_adam_step_replay_matches_eager_updates() -> None:
    eager_weight = tynx.Parameter([[0.5]], name="weight")
    captured_weight = tynx.Parameter([[0.5]], name="weight")
    eager_optimizer = tynx.optim.Adam(
        [eager_weight], lr=0.03, betas=(0.8, 0.95), eps=1.0e-6, amsgrad=True
    )
    captured_optimizer = tynx.optim.Adam(
        [captured_weight], lr=0.03, betas=(0.8, 0.95), eps=1.0e-6, amsgrad=True
    )
    calls = 0

    def eager_step(input: tynx.Tensor, target: tynx.Tensor) -> tynx.Tensor:
        eager_optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(input @ eager_weight, target)
        loss.backward()
        eager_optimizer.step()
        return loss

    @tynx.compile(fullgraph=True)
    def captured_step(input: tynx.Tensor, target: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        captured_optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(input @ captured_weight, target)
        loss.backward()
        captured_optimizer.step()
        return loss

    input = tynx.Tensor([[1.0], [2.0], [3.0]])
    target = tynx.Tensor([[2.0], [4.0], [6.0]])
    for _ in range(50):
        eager_loss = eager_step(input, target)
        captured_loss = captured_step(input, target)
        assert captured_loss.item() == pytest.approx(eager_loss.item(), abs=1.0e-6)
        assert captured_weight.item() == pytest.approx(eager_weight.item(), abs=1.0e-6)

    assert calls == 1
    assert captured_step.replay_count == 49
    assert captured_optimizer.state_size == eager_optimizer.state_size == 1


def test_whole_step_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    model = tynx.nn.Linear(1, 1)
    model.weight.copy_(tynx.Tensor([[0.5]]))
    assert model.bias is not None
    model.bias.copy_(tynx.Tensor([0.25]))
    optimizer = tynx.optim.Adam(
        model.named_parameters(), lr=0.02, betas=(0.8, 0.95), eps=1.0e-6, amsgrad=True
    )

    @tynx.compile(fullgraph=True)
    def train_step(input: tynx.Tensor, target: tynx.Tensor) -> tynx.Tensor:
        optimizer.zero_grad()
        loss = tynx.nn.functional.mse_loss(model(input), target)
        loss.backward()
        optimizer.step()
        return loss

    input = tynx.Tensor([[1.0], [2.0], [3.0]])
    target = tynx.Tensor([[2.0], [4.0], [6.0]])
    for _ in range(12):
        train_step(input, target)

    checkpoint = tmp_path / "captured-step.tynx"
    tynx.save_checkpoint(checkpoint, model, optimizer)

    expected_loss = train_step(input, target).item()
    expected_state = model.state_dict()

    resumed = tynx.nn.Linear(1, 1)
    resumed_optimizer = tynx.optim.Adam(
        resumed.named_parameters(), lr=0.9, betas=(0.8, 0.95), eps=1.0e-6, amsgrad=True
    )
    tynx.load_checkpoint(checkpoint, resumed, resumed_optimizer)
    resumed_optimizer.zero_grad()
    resumed_loss = tynx.nn.functional.mse_loss(resumed(input), target)
    resumed_loss.backward()
    resumed_optimizer.step()

    assert resumed_loss.item() == pytest.approx(expected_loss, abs=1.0e-6)
    for name, value in resumed.state_dict().items():
        assert value.flatten().tolist() == pytest.approx(expected_state[name].flatten().tolist())
