# Contributing

## Patched dependencies

Before using Cargo, Maturin, or the separate benchmark workspace in a fresh checkout, prepare the
ignored Burn, CubeCL, and CubeK clones:

```sh
cargo vendor-patches
```

This standalone bootstrap command checks out the revisions pinned by the project and applies the
reviewed patches under `vendor/`. It is safe to run again; it rejects revision drift and unrelated
changes instead of resetting a dependency checkout.

## ONNX conformance

The ignored conformance test uses the official ONNX node cases pinned to the same Burn-ONNX
revision as `onnx-ir`. The corpus is fetched once into the gitignored `.cache` directory.

```sh
cargo xtask conformance fetch
cargo xtask conformance
cargo xtask conformance --case test_relu
```

When an intentional runtime change alters the results, inspect `target/conformance-report.json`
and update the committed baseline with `cargo xtask conformance bless`.

## Python bindings

Build the extension into the active virtual environment, then run its smoke test:

```sh
cd crates/tynx-python
python -m venv .venv
source .venv/bin/activate
python -m pip install "maturin>=1.9,<2"
maturin develop --locked --group test
pytest -n auto --maxprocesses 4
```

Run the Python linters without building the extension:

```sh
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group lint
ruff check python tests
ruff format --check python tests
mypy
```
