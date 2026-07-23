"""Runtime validation for authored module state dictionaries."""

from typing import Any

import pytest
import tynx


@pytest.mark.parametrize("state_dict", ["weight", 3, [], object()])
def test_module_load_state_dict_rejects_non_mapping_inputs(state_dict: Any) -> None:
    module = tynx.nn.Linear(2, 1)
    before = module.state_dict()

    with pytest.raises(TypeError, match=r"state_dict must be a mapping, got"):
        module.load_state_dict(state_dict)

    for name, value in module.state_dict().items():
        assert value.tolist() == before[name].tolist()


def test_state_loader_accepts_mapping_subclasses() -> None:
    class StateMapping(dict[str, tynx.Tensor]):
        pass

    source = tynx.nn.Linear(2, 1)
    destination = tynx.nn.Linear(2, 1)

    result = destination.load_state_dict(StateMapping(source.state_dict()))

    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
