# Agent guidance

## Changes

- Branch focused work from `main`; keep unrelated and user-owned changes untouched.
- Follow existing Rust and Python structure, naming, error handling, and test conventions.

## Validation

- Rust: run `cargo fmt --all -- --check`, Clippy with warnings denied, and relevant tests.
- Python: rebuild bindings after Rust changes, then run Ruff, mypy, and relevant pytest tests from `crates/tynx-python`.

## Performance

- Benchmark PyTorch first and use representative sample shapes.
- Build Tynx with `maturin develop --release --locked`; never compare a debug build with release PyTorch.
- Match device, dtype, thread count, grad mode, and synchronization; warm up and report median absolute timings and ratios.
- Treat CPU and GPU results separately. Do not add a slower fast path without a demonstrated non-performance benefit.
