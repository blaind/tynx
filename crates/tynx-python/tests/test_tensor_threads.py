from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from queue import Queue

import pytest
import tynx
from tynx import nn

_THREAD_ERROR = (
    r"thread-confined.*pass NumPy arrays between threads.*create the Tensor in the worker thread"
)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda tensor: tensor.tolist(), id="host-read"),
        pytest.param(lambda tensor: tensor + 1.0, id="computation"),
    ],
)
def test_tensor_cross_thread_access_raises_runtime_error(
    operation: Callable[[tynx.Tensor], object],
) -> None:
    tensor = tynx.Tensor([1.0, 2.0])
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            operation(tensor)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert re.search(_THREAD_ERROR, str(errors[0]))
    assert tensor.tolist() == [1.0, 2.0]


def test_tensor_destroyed_on_non_owner_thread_does_not_panic() -> None:
    script = """
import gc
import queue
import threading

import tynx

handoff = queue.Queue()
handoff.put(tynx.Tensor([1.0]))

def worker():
    tensor = handoff.get()
    del tensor
    gc.collect()

thread = threading.Thread(target=worker)
thread.start()
thread.join()
"""
    environment = os.environ.copy()
    environment["BURN_DEVICE"] = "flex"
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "panicked" not in result.stderr
    assert "PanicException" not in result.stderr


def test_module_and_parameter_cross_thread_access_raises_runtime_error() -> None:
    model = nn.Linear(2, 1)
    tensor = tynx.Tensor([[1.0, 2.0]])
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            model(tensor)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert re.search(_THREAD_ERROR, str(errors[0]))


def _tensor_from_finished_worker(data: object) -> tynx.Tensor:
    tensors: Queue[tynx.Tensor] = Queue()
    thread = threading.Thread(target=lambda: tensors.put(tynx.Tensor(data)))
    thread.start()
    thread.join()
    return tensors.get()


def _int_tensor_from_finished_worker(data: object) -> tynx.Tensor:
    tensors: Queue[tynx.Tensor] = Queue()
    thread = threading.Thread(target=lambda: tensors.put(tynx.Tensor(data, dtype="int64")))
    thread.start()
    thread.join()
    return tensors.get()


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda tensor: tynx.sort(tensor), id="sort"),
        pytest.param(lambda tensor: tynx.argsort(tensor), id="argsort"),
        pytest.param(lambda tensor: tynx.topk(tensor, 1), id="topk"),
    ],
)
def test_module_level_ordering_rejects_non_owner_tensor(
    operation: Callable[[tynx.Tensor], object],
) -> None:
    tensor = _tensor_from_finished_worker([2.0, 1.0])

    with pytest.raises(RuntimeError, match=_THREAD_ERROR):
        operation(tensor)


def test_index_select_checks_both_tensor_owners() -> None:
    local = tynx.Tensor([1.0, 2.0])
    foreign_values = _tensor_from_finished_worker([3.0, 4.0])
    foreign_indices = _int_tensor_from_finished_worker([0])

    with pytest.raises(RuntimeError, match=_THREAD_ERROR):
        tynx.index_select(foreign_values, 0, tynx.Tensor([0], dtype="int64"))
    with pytest.raises(RuntimeError, match=_THREAD_ERROR):
        tynx.index_select(local, 0, foreign_indices)


def test_backward_checks_explicit_gradient_owner() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    output = value * value
    gradient = _tensor_from_finished_worker([1.0])

    with pytest.raises(RuntimeError, match=_THREAD_ERROR):
        output.backward(gradient)


def test_capture_checks_input_owner_before_reading_trace_state() -> None:
    tensor = _tensor_from_finished_worker([1.0])

    @tynx.compile
    def compiled(value: tynx.Tensor) -> tynx.Tensor:
        return value + 1.0

    with pytest.raises(RuntimeError, match=_THREAD_ERROR):
        compiled(tensor)
