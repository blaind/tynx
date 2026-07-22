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
        return input + 1.0

    compiled = tynx.compile(forward)
    with pytest.warns(RuntimeWarning, match="fell back to eager"):
        assert compiled(tynx.Tensor([1.0])).tolist() == [2.0]
    assert compiled(tynx.Tensor([2.0])).tolist() == [3.0]
    assert calls == 2
    assert compiled.compile_count == 0
    assert compiled.fallback_count == 2


def test_fullgraph_rejects_unsupported_operation_visibly() -> None:
    def function(input: tynx.Tensor) -> tynx.Tensor:
        return input + 1.0

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


def test_fullgraph_rejects_random_dropout() -> None:
    layer = tynx.nn.Dropout(0.5)
    forward = tynx.compile(layer, fullgraph=True)

    with pytest.raises(RuntimeError, match="random Dropout"):
        forward(tynx.Tensor([1.0, 1.0]))


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
