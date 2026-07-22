# Tynx benchmarks

This standalone Rust workspace compares the same ONNX model and input across:

- Tynx runtime interpretation
- ONNX Runtime
- burn-onnx ahead-of-time generated Rust

The registry starts with seven workloads:

| Case | Purpose |
| --- | --- |
| `sign-11` | Protocol and dispatch baseline |
| `matmul-64x64` | Small two-input matrix operation |
| `matmul-256x256` | Compute-heavy dynamic-shape matrix operation |
| `matmul-512x512` | Medium dynamic-shape scaling point |
| `matmul-1024x1024` | Large dynamic-shape scaling point |
| `matmul-add-relu-256x256` | Small multi-op graph with broadcast input |
| `tiny-cnn-32` | Compact image classifier with embedded parameters |

The tiny CNN runs Conv, ReLU, MaxPool, global average pooling, Gemm, and Softmax over a 32x32 RGB
input. Its deterministic parameters and reference probabilities make it a repeatable structural
workload, not an accuracy benchmark. Larger representative models are still needed before drawing
system-level performance conclusions.

## CPU

```sh
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p tynx-bench
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p ort-bench
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p burn-aot-bench
```

ORT downloads its official CPU binary for the default configuration.

Each runner executes every registered case by default and emits one JSON report array. Set
`TYNX_BENCH_CASE` to limit a run to one case. The manual GitHub workflow invokes each engine once,
adds a comparison table to the job summary, and uploads its JSON reports. Leave the workflow
iteration inputs blank to use the per-case defaults.

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

MatMul cases report estimated GFLOP/s using the conventional `2 * M * N * K` operation count. The
tiny CNN uses two operations per Conv and Gemm multiply-accumulate. Other operators are excluded
from its estimate. Timing includes host input construction and host output materialization, so the
value measures end-to-end inference rather than kernel-only throughput.

Override the selected case and sample counts with:

```sh
TYNX_BENCH_CASE=matmul-64x64 \
TYNX_BENCH_WARMUP=50 \
TYNX_BENCH_ITERATIONS=1000 \
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p tynx-bench
```

Performance runs must use `--release`. The runners reject debug builds.
