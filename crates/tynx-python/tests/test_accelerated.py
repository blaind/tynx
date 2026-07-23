"""Accelerated-device synchronization and eager tape-lifetime acceptance tests."""

import gc
import math
import os
from pathlib import Path
from typing import cast

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
_GATHER_MODEL = bytes.fromhex(
    "08083ac4010a450a10656d62656464696e672e7765696768740a07696e646963657312087365"
    "6c65637465641a09656d62656464696e6722064761746865722a0b0a04617869731800a00102"
    "12146761746865725f747261696e696e675f746573742a320803080210014210656d62656464"
    "696e672e7765696768744a180000803f0000004000004040000080400000a0400000c0405a15"
    "0a07696e6469636573120a0a08080712040a020803621a0a0873656c6563746564120e0a0c08"
    "0112080a0208030a02080242040a00100d"
)
_LAYER_NORM_MODEL = bytes.fromhex(
    "08083adf010a610a01780a0b6e6f726d2e7765696768740a096e6f726d2e62696173120179"
    "1a046e6f726d22124c617965724e6f726d616c697a6174696f6e2a140a046178697318ffffff"
    "ffffffffffff01a001022a110a07657073696c6f6e15acc52737a0010112186c617965725f6e"
    "6f726d5f747261696e696e675f746573742a1b0802100122080000803f0000803f420b6e6f"
    "726d2e7765696768742a19080210012208000000000000000042096e6f726d2e626961735a"
    "130a0178120e0a0c080112080a0208020a02080262130a0179120e0a0c080112080a0208020a"
    "02080242040a001011"
)
_BATCHED_MATMUL_MODEL = bytes.fromhex(
    "08083aaa010a2d0a01780a1170726f6a656374696f6e2e7765696768741201791a0a70726f"
    "6a656374696f6e22064d61744d756c1212626174636865645f70726f6a656374696f6e2a33"
    "08030802100122180000803f00000000000000000000803f0000803f0000803f421170726f"
    "6a656374696f6e2e7765696768745a170a017812120a100801120c0a0208020a0208020a02"
    "080362170a017912120a100801120c0a0208020a0208020a02080242040a00100d"
)


def _accelerated_device() -> tynx.Device:
    device = tynx.get_default_device()
    if "Vulkan" in str(device) or "Wgpu" in str(device):
        return device
    if os.environ.get("TYNX_REQUIRE_ACCELERATED") == "1":
        pytest.fail(f"accelerated test lane selected a non-WGPU device: {device}")
    pytest.skip(f"accelerated backend not enabled: {device}")


def test_accelerated_multidimensional_extrema_avoid_i64_mask_reductions() -> None:
    device = _accelerated_device()
    value = tynx.Tensor([[3.0, -2.0, 7.0], [1.0, 7.0, 0.0]])

    assert value.max().item() == pytest.approx(7.0)
    assert value.min().item() == pytest.approx(-2.0)
    assert value.argmax().item() == 2
    assert value.argmin().item() == 1
    assert value.max(keepdim=True).shape == (1, 1)

    assert value.max(dim=0).tolist() == pytest.approx([3.0, 7.0, 7.0])
    assert value.min(dim=1).tolist() == pytest.approx([-2.0, 0.0])
    assert value.argmax(dim=0).tolist() == [0, 1, 0]
    assert value.argmin(dim=1).tolist() == [1, 2]

    with_nan = tynx.Tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
    maxima = cast(list[float], with_nan.max(dim=1).tolist())
    minima = cast(list[float], with_nan.min(dim=1).tolist())
    assert math.isnan(maxima[0])
    assert maxima[1] == pytest.approx(6.0)
    assert math.isnan(minima[0])
    assert minima[1] == pytest.approx(4.0)
    tynx.synchronize(device)


def test_accelerated_int64_extrema_avoid_backend_reduction_kernel() -> None:
    device = _accelerated_device()
    minimum = -(2**63)
    maximum = 2**63 - 1
    value = tynx.Tensor([[minimum, 0, 7], [maximum, -5, 3]], dtype="int64")

    assert value.max(dim=0).tolist() == [maximum, 0, 7]
    assert value.max().item() == maximum
    assert value.min(dim=1).tolist() == [minimum, -5]
    assert value.min().item() == minimum

    vector = tynx.Tensor([minimum, 0, maximum], dtype="int64")
    assert vector.max(dim=0).tolist() == [maximum]
    assert vector.max().item() == maximum
    assert vector.min(dim=0).tolist() == [minimum]
    assert vector.min().item() == minimum
    tynx.synchronize(device)


def test_accelerated_boolean_construction_and_factories_avoid_bool_scalars() -> None:
    device = _accelerated_device()

    value = tynx.Tensor([[True, False], [False, True]], dtype="bool")
    zeros = tynx.zeros((2,), dtype="bool")
    ones = tynx.ones((2,), dtype="bool")
    full = tynx.full((2,), True, dtype="bool")
    ones_like = tynx.ones_like(value)

    tynx.synchronize(device)
    assert value.tolist() == [[True, False], [False, True]]
    assert zeros.tolist() == [False, False]
    assert ones.tolist() == [True, True]
    assert full.tolist() == [True, True]
    assert ones_like.tolist() == [[True, True], [True, True]]


def test_accelerated_cross_backend_operations_raise_python_exceptions() -> None:
    device = _accelerated_device()
    gpu = tynx.ones((2,))
    cpu = tynx.ones((2,), device=tynx.Device("cpu"))

    with pytest.raises(ValueError, match="same device"):
        cpu + gpu
    with pytest.raises(ValueError, match="same device"):
        cpu.reshape(1, 2) @ gpu.reshape(2, 1)
    with pytest.raises(ValueError, match="same device"):
        tynx.maximum(cpu, gpu)
    with pytest.raises(NotImplementedError, match="cannot move tensors between backends"):
        gpu.to(tynx.Device("cpu"))
    with pytest.raises(NotImplementedError, match="cannot move tensors between backends"):
        cpu.to(device)
    cpu_index = tynx.Tensor([0], dtype="int64", device=tynx.Device("cpu"))
    with pytest.raises(ValueError, match="same device"):
        gpu[cpu_index]
    with pytest.raises(ValueError, match="same device"):
        gpu.index_select(0, cpu_index)
    for value in (2, -3):
        gpu_index = tynx.Tensor([value], dtype="int64")
        with pytest.raises(IndexError, match="out of bounds"):
            gpu[gpu_index]
        with pytest.raises(IndexError, match="out of bounds"):
            gpu.index_select(0, gpu_index)
    with (
        tynx.no_grad(),
        pytest.raises(NotImplementedError, match="cannot move tensors between backends"),
    ):
        gpu.to(tynx.Device("cpu"))

    tynx.synchronize(device)


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


def test_accelerated_conv2d_gradient_matches_central_finite_difference() -> None:
    device = _accelerated_device()
    values = [0.2, -0.4, 0.7, 1.1]
    input = tynx.Tensor([[[values[:2], values[2:]]]], requires_grad=True)
    weight = tynx.Tensor([[[[0.5, -0.25], [0.75, 0.1]]]])

    tynx.nn.functional.conv2d(input, weight).sum().backward()
    tynx.synchronize(device)
    assert input.grad is not None
    analytical = input.grad.flatten().tolist()[0]
    epsilon = 1e-3

    def evaluate(first: float) -> float:
        candidate = [first, *values[1:]]
        tensor = tynx.Tensor([[[candidate[:2], candidate[2:]]]])
        output = tynx.nn.functional.conv2d(tensor, weight).sum()
        tynx.synchronize(device)
        return output.item()

    numerical = (evaluate(values[0] + epsilon) - evaluate(values[0] - epsilon)) / (2 * epsilon)
    assert analytical == pytest.approx(numerical, rel=2e-3, abs=2e-3)


def test_accelerated_embedding_accumulates_repeated_index_gradients() -> None:
    device = _accelerated_device()
    layer = tynx.nn.Embedding(4, 2, padding_idx=0)
    layer.weight.copy_(tynx.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
    indices = tynx.Tensor([0, 2, 2, 1], dtype="int64")

    output = layer(indices)
    output.sum().backward()

    tynx.synchronize(device)
    assert output.flatten().tolist() == pytest.approx([1.0, 2.0, 5.0, 6.0, 5.0, 6.0, 3.0, 4.0])
    assert layer.weight.grad is not None
    assert layer.weight.grad.flatten().tolist() == pytest.approx(
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0]
    )


def test_accelerated_captured_output_retains_tape_after_wrapper_drop() -> None:
    device = _accelerated_device()
    layer = tynx.nn.Embedding(3, 2, padding_idx=0)
    layer.weight.copy_(tynx.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    weight = layer.weight

    @tynx.compile(fullgraph=True)
    def lookup(indices: tynx.Tensor) -> tynx.Tensor:
        return layer(indices)

    indices = tynx.Tensor([0, 2, 2, 1], dtype="int64")
    lookup(indices)
    output = lookup(indices)
    loss = output.sum()
    assert lookup.replay_count == 1

    del output
    del lookup
    gc.collect()
    tynx.synchronize(device)
    loss.backward()
    tynx.synchronize(device)

    assert weight.grad is not None
    assert weight.grad.flatten().tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])


def test_accelerated_recovers_after_rejected_index_and_synchronizes() -> None:
    device = _accelerated_device()
    value = tynx.Parameter([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(IndexError, match="out of bounds"):
        value[tynx.Tensor([2], dtype="int64")]

    selected = value[tynx.Tensor([1], dtype="int64")]
    selected.sum().backward()
    tynx.synchronize(device)

    assert selected.tolist() == [[3.0, 4.0]]
    assert value.grad is not None
    assert value.grad.tolist() == [[0.0, 0.0], [1.0, 1.0]]


def test_accelerated_imported_gather_layer_norm_and_batched_matmul(
    tmp_path: Path,
) -> None:
    device = _accelerated_device()

    gather_path = tmp_path / "gather.onnx"
    gather_path.write_bytes(_GATHER_MODEL)
    gather = tynx.load(gather_path, trainable="auto", simplify=False)
    gather(tynx.Tensor([0, 2, 0], dtype="int64")).sum().backward()
    gather_weight = dict(gather.named_parameters())["embedding.weight"]

    norm_path = tmp_path / "layer_norm.onnx"
    norm_path.write_bytes(_LAYER_NORM_MODEL)
    norm = tynx.load(norm_path, trainable="auto", simplify=False)
    norm(tynx.Tensor([[1.0, 3.0], [2.0, 6.0]])).sum().backward()
    norm_parameters = dict(norm.named_parameters())

    matmul_path = tmp_path / "batched_matmul.onnx"
    matmul_path.write_bytes(_BATCHED_MATMUL_MODEL)
    matmul = tynx.load(matmul_path, trainable="auto", simplify=False)
    projected = matmul(
        tynx.Tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
        )
    )
    projected.sum().backward()
    projection_weight = dict(matmul.named_parameters())["projection.weight"]

    tynx.synchronize(device)
    assert gather_weight.grad is not None
    assert gather_weight.grad.flatten().tolist() == pytest.approx([2.0, 2.0, 0.0, 0.0, 1.0, 1.0])
    assert norm_parameters["norm.weight"].grad is not None
    assert norm_parameters["norm.weight"].grad.flatten().tolist() == pytest.approx(
        [-1.999994, 1.999994], abs=1e-5
    )
    assert norm_parameters["norm.bias"].grad is not None
    assert norm_parameters["norm.bias"].grad.tolist() == pytest.approx([2.0, 2.0])
    assert projected.flatten().tolist() == pytest.approx(
        [4.0, 5.0, 10.0, 11.0, 16.0, 17.0, 22.0, 23.0]
    )
    assert projection_weight.grad is not None
    assert projection_weight.grad.flatten().tolist() == pytest.approx(
        [22.0, 22.0, 26.0, 26.0, 30.0, 30.0]
    )


def test_accelerated_pooling_layers_share_one_backward_tape() -> None:
    device = _accelerated_device()
    input = tynx.Parameter([[[[1.0, 2.0], [3.0, 4.0]]]])

    maximum = tynx.nn.MaxPool2d(2)(input)
    average = tynx.nn.AvgPool2d(2)(input)
    adaptive = tynx.nn.AdaptiveAvgPool2d((1, 1))(input)
    (maximum + average + adaptive).sum().backward()

    tynx.synchronize(device)
    assert maximum.item() == pytest.approx(4.0)
    assert average.item() == pytest.approx(2.5)
    assert adaptive.item() == pytest.approx(2.5)
    assert input.grad is not None
    assert input.grad.flatten().tolist() == pytest.approx([0.5, 0.5, 0.5, 1.5])


def test_accelerated_tensor_composition_preserves_gradients() -> None:
    device = _accelerated_device()
    value = tynx.Parameter([[1.0, 2.0], [3.0, 4.0]])

    joined = tynx.cat([value, value * 2.0], dim=0)
    left, right = joined.split(2, dim=0)
    first, second = tynx.stack([left, right], dim=0).chunk(2, dim=0)
    expanded = value.unsqueeze(0).expand(3, 2, 2)
    repeated = value.repeat(2, 3)
    (first.sum() + second.sum() + expanded.sum() + repeated.sum()).backward()

    tynx.synchronize(device)
    assert joined.shape == (4, 2)
    assert expanded.shape == (3, 2, 2)
    assert repeated.shape == (4, 6)
    assert value.grad is not None
    assert value.grad.flatten().tolist() == pytest.approx([12.0, 12.0, 12.0, 12.0])


def test_accelerated_capture_replays_conv2d_and_pooling_gradients() -> None:
    device = _accelerated_device()
    layer = tynx.nn.Conv2d(1, 1, 2)
    layer.weight.copy_(tynx.Tensor([[[[1.0, 1.0], [1.0, 1.0]]]]))
    assert layer.bias is not None
    layer.bias.copy_(tynx.Tensor([0.0]))
    calls = 0

    @tynx.compile(fullgraph=True)
    def pooled(input: tynx.Tensor) -> tuple[tynx.Tensor, tynx.Tensor, tynx.Tensor]:
        nonlocal calls
        calls += 1
        convolved = layer(input)
        return (
            tynx.nn.functional.max_pool2d(convolved, 2),
            tynx.nn.functional.avg_pool2d(convolved, 2),
            tynx.nn.functional.adaptive_avg_pool2d(convolved, (1, 1)),
        )

    first_input = tynx.Tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
    )
    second_input = tynx.Tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        requires_grad=True,
    )
    first = pooled(first_input)
    second = pooled(second_input)
    (second[0].sum() + second[1].sum() + second[2].sum()).backward()

    tynx.synchronize(device)
    assert [output.flatten().tolist() for output in second] == [
        output.flatten().tolist() for output in first
    ]
    assert second_input.grad is not None
    assert second_input.grad.shape == second_input.shape
    assert layer.weight.grad is not None
    assert calls == 1
    assert pooled.compile_count == 1
    assert pooled.fallback_count == 0
    assert pooled.replay_count == 1


def test_accelerated_capture_replays_tensor_composition_and_gradients() -> None:
    device = _accelerated_device()
    calls = 0

    @tynx.compile(fullgraph=True)
    def compose(input: tynx.Tensor) -> tuple[tynx.Tensor, tynx.Tensor, tynx.Tensor]:
        nonlocal calls
        calls += 1
        joined = tynx.cat([input, input * 2.0], dim=0)
        left, right = joined.split(2, dim=0)
        first, second = tynx.stack([left, right], dim=0).chunk(2, dim=0)
        return first, second, input.unsqueeze(0).expand(3, 2, 2)

    compose(tynx.Tensor([[1.0, 2.0], [3.0, 4.0]]))
    value = tynx.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    first, second, expanded = compose(value)
    (first.sum() + second.sum() + expanded.sum()).backward()

    tynx.synchronize(device)
    assert first.flatten().tolist() == pytest.approx([5.0, 6.0, 7.0, 8.0])
    assert second.flatten().tolist() == pytest.approx([10.0, 12.0, 14.0, 16.0])
    assert expanded.shape == (3, 2, 2)
    assert value.grad is not None
    assert value.grad.flatten().tolist() == pytest.approx([6.0, 6.0, 6.0, 6.0])
    assert calls == 1
    assert compose.compile_count == 1
    assert compose.fallback_count == 0
    assert compose.replay_count == 1


def test_accelerated_ordering_and_advanced_indexing_preserve_gradients() -> None:
    device = _accelerated_device()
    value = tynx.Parameter([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])

    ordered, indices = value.sort(dim=1)
    largest, largest_indices = value.topk(2, dim=1)
    selected = value[tynx.Tensor([1, 0, 1], dtype="int64")]
    masked = value[tynx.Tensor([True, False], dtype="bool")]
    (ordered.sum() + largest.sum() + selected.sum() + masked.sum()).backward()

    tynx.synchronize(device)
    assert ordered.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert indices.tolist() == [[1, 2, 0], [1, 2, 0]]
    assert largest.tolist() == [[3.0, 2.0], [6.0, 5.0]]
    assert largest_indices.tolist() == [[0, 2], [0, 2]]
    assert masked.tolist() == [[3.0, 1.0, 2.0]]
    assert value.grad is not None
    assert value.grad.tolist() == [[4.0, 3.0, 4.0], [4.0, 3.0, 4.0]]


def test_accelerated_captured_imported_ppo_step_reuses_updated_weights(tmp_path: Path) -> None:
    device = _accelerated_device()
    path = tmp_path / "actor_critic.onnx"
    path.write_bytes(_ACTOR_CRITIC_MODEL)
    model = tynx.load(
        path,
        trainable="auto",
        simplify=False,
        initializer_names={
            "policy_weight": "policy.weight",
            "policy_bias": "policy.bias",
            "value_weight": "value.weight",
            "value_bias": "value.bias",
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
