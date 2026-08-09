"""Explicit module buffer registration."""

from pathlib import Path
from typing import Optional, Union

import pytest
import tynx


class BufferedModule(tynx.nn.Module):
    running: tynx.Buffer
    optional: Optional[Union[tynx.Tensor, tynx.Buffer]]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running", tynx.Tensor([1.0, 2.0]))
        self.register_buffer("optional", None)

    def forward(self, input: tynx.Tensor) -> tynx.Tensor:
        return input + self.running


class TargetBufferModule(tynx.nn.Module):
    target: tynx.Buffer


class DirectBufferModule(tynx.nn.Module):
    running: tynx.Buffer


def test_register_buffer_wraps_tensor_and_participates_in_state() -> None:
    module = BufferedModule()

    assert isinstance(module.running, tynx.Buffer)
    assert module.optional is None
    assert module.named_buffers() == [("running", module.running)]
    assert list(module.state_dict()) == ["running"]
    assert module(tynx.Tensor([3.0, 4.0])).tolist() == [4.0, 6.0]


def test_register_buffer_preserves_existing_buffer_identity() -> None:
    module = TargetBufferModule()
    buffer = tynx.Buffer([1.0], name="source")

    module.register_buffer("target", buffer)

    assert module.target is buffer
    assert module.named_buffers() == [("target", buffer)]


def test_registered_none_reserves_a_buffer_slot_for_later_assignment() -> None:
    module = BufferedModule()

    module.optional = tynx.Tensor([3.0])

    assert isinstance(module.optional, tynx.Buffer)
    assert module.optional.tolist() == [3.0]
    assert dict(module.named_buffers())["optional"] is module.optional
    assert list(module.state_dict()) == ["optional", "running"]


def test_registered_buffer_slot_can_cycle_through_none() -> None:
    module = BufferedModule()
    module.optional = tynx.Tensor([3.0])

    module.optional = None

    assert module.optional is None
    assert list(module.state_dict()) == ["running"]

    module.register_buffer("optional", tynx.Tensor([4.0]))

    assert isinstance(module.optional, tynx.Buffer)
    assert module.optional.tolist() == [4.0]


def test_registered_buffer_rejects_invalid_later_assignment() -> None:
    module = BufferedModule()

    with pytest.raises(TypeError, match="Tensor, Buffer, or None"):
        module.optional = [1.0]  # type: ignore[assignment]

    assert module.optional is None


def test_registered_buffer_round_trips_through_checkpoint(tmp_path: Path) -> None:
    source = BufferedModule()
    checkpoint = tmp_path / "buffer.tynx"
    tynx.save_checkpoint(checkpoint, source)

    destination = BufferedModule()
    destination.running.copy_(tynx.Tensor([9.0, 9.0]))
    result = tynx.load_checkpoint(checkpoint, destination)

    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert destination.running.tolist() == [1.0, 2.0]


def test_direct_and_registered_buffer_attributes_are_equivalent() -> None:
    direct = DirectBufferModule()
    direct.running = tynx.Buffer([1.0])
    registered = tynx.nn.Module()
    registered.register_buffer("running", tynx.Tensor([1.0]))

    assert list(direct.state_dict()) == list(registered.state_dict()) == ["running"]


@pytest.mark.parametrize("name", ["", "nested.value"])
def test_register_buffer_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="non-empty and cannot contain"):
        tynx.nn.Module().register_buffer(name, tynx.Tensor([1.0]))


def test_register_buffer_rejects_collisions_and_invalid_values() -> None:
    module = tynx.nn.Module()
    module.register_buffer("value", tynx.Tensor([1.0]))

    module.register_buffer("value", tynx.Tensor([2.0]))
    assert module.value.tolist() == [2.0]  # type: ignore[attr-defined]
    with pytest.raises(KeyError, match="already exists"):
        module.register_buffer("training", tynx.Tensor([2.0]))
    with pytest.raises(TypeError, match="Tensor, Buffer, or None"):
        tynx.nn.Module().register_buffer("value", [1.0])  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="non-persistent"):
        tynx.nn.Module().register_buffer("value", tynx.Tensor([1.0]), persistent=False)
