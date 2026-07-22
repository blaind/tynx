# Tynx benchmarks

This standalone Rust workspace compares the same ONNX model and input across:

- Tynx runtime interpretation
- ONNX Runtime
- burn-onnx ahead-of-time generated Rust

The registry starts with four workloads:

| Case | Purpose |
| --- | --- |
| `sign-11` | Protocol and dispatch baseline |
| `matmul-64x64` | Small two-input matrix operation |
| `matmul-256x256` | Compute-heavy dynamic-shape matrix operation |
| `matmul-add-relu-256x256` | Small multi-op graph with broadcast input |

These cases cover harness correctness and basic scaling. Larger representative models are still
needed before drawing system-level performance conclusions.

## CPU

```sh
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p tynx-bench
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p ort-bench
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p burn-aot-bench
```

ORT downloads its official CPU binary for the default configuration.

Select a case by setting `TYNX_BENCH_CASE` for any runner. The manual GitHub workflow runs every
registered case and uploads its JSON reports.

## GPU

```sh
cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p tynx-bench --no-default-features --features wgpu
cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p burn-aot-bench --no-default-features --features wgpu
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
  cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p ort-bench --no-default-features --features cuda
```

The CUDA runner fails if the CUDA execution provider cannot be registered. This prevents silent CPU
fallback. Use a GPU-enabled ONNX Runtime dynamic library for that command.

WGPU reports the high-performance adapter selected by the same `wgpu` heuristic used by Burn. This
makes accidental software-adapter results visible in the JSON report.

Do not use SwiftShader results as performance numbers. Run GPU comparisons on fixed physical
hardware, with the same driver and power configuration.

## Protocol

Every runner:

1. Loads or constructs its session outside steady-state timing.
2. Records first-run latency separately.
3. Warms up before collecting samples.
4. Includes host input creation and host output materialization in each timed inference.
5. Validates the first and final output against the registry.
6. Writes the same JSON result schema to standard output.

Override the selected case and sample counts with:

```sh
TYNX_BENCH_CASE=matmul-64x64 \
TYNX_BENCH_WARMUP=50 \
TYNX_BENCH_ITERATIONS=1000 \
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p tynx-bench
```

Performance runs must use `--release`. The runners reject debug builds.
