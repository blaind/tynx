"""User-facing Python examples remain executable."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
    ("script", "summary"),
    [
        ("authored_training.py", "authored eager training:"),
        ("imported_finetuning.py", "imported ONNX fine-tuning:"),
        ("captured_ppo.py", "Python bodies: 1; native replays: 39"),
    ],
)
def test_python_training_example(script: str, summary: str) -> None:
    environment = os.environ.copy()
    environment["BURN_DEVICE"] = "flex"
    result = subprocess.run(
        [sys.executable, str(_ROOT / "examples" / script)],
        cwd=_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert summary in result.stdout
