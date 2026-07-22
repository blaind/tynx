# Tynx benchmarks

This standalone Rust workspace compares the same ONNX model and input across:

- Tynx runtime interpretation
- ONNX Runtime
- burn-onnx ahead-of-time generated Rust

The registry contains ten self-contained workloads and one optional cached model:

| Case | Purpose |
| --- | --- |
| `sign-11` | Protocol and dispatch baseline |
| `matmul-64x64` | Small two-input matrix operation |
| `matmul-256x256` | Compute-heavy dynamic-shape matrix operation |
| `matmul-512x512` | Medium dynamic-shape scaling point |
| `matmul-1024x1024` | Large dynamic-shape scaling point |
| `matmul-add-relu-256x256` | Small multi-op graph with broadcast input |
| `tiny-cnn-32` | Compact image classifier, batch 1 |
| `tiny-cnn-32-b8` | Compact image classifier, batch 8 |
| `tiny-cnn-32-b32` | Compact image classifier, batch 32 |
| `tiny-cnn-32-b128` | Compact image classifier, batch 128 |
| `mobilenetv2-100-b1` | Pretrained MobileNetV2 1.0 ImageNet classifier |

The tiny CNN runs Conv, ReLU, MaxPool, global average pooling, Gemm, and Softmax over repeated 32x32
RGB inputs. Its dynamic batch dimension and deterministic reference probabilities make it a
repeatable scaling workload, not an accuracy benchmark.

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

## MobileNetV2

MobileNetV2 is opt-in so normal builds do not download or embed its 14 MB of weights. Fetch the
pinned model, then enable it for any runner:

```sh
export TYNX_BENCH_MOBILENET_PATH="$(benchmarks/models/fetch-mobilenetv2.sh)"
TYNX_BENCH_CASE=mobilenetv2-100-b1 \
  cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p tynx-bench --features mobilenet
```

The fetcher caches the model under `.cache/benchmark-models` and verifies SHA-256
`c1793982c0504e1808e7d0d99d4cc5972de35137d6b5e8492573ecb72b2e241f`. It is the Apache-2.0
licensed [ONNX Model Zoo MobileNetV2 1.0 opset 16 model](https://github.com/onnx/models/tree/d55d2baeb0d6641643d5295a4f42b545fcf9d9e2/Computer_Vision/mobilenetv2_100_Opset16_timm).

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
from its estimate. MobileNetV2 uses the same convention for Conv and Gemm. Batched model cases also
report images per second. Timing includes host input construction and host output materialization,
so the value measures end-to-end inference rather than kernel-only throughput.

Override the selected case and sample counts with:

```sh
TYNX_BENCH_CASE=matmul-64x64 \
TYNX_BENCH_WARMUP=50 \
TYNX_BENCH_ITERATIONS=1000 \
cargo run --manifest-path benchmarks/Cargo.toml --locked --release -p tynx-bench
```

Performance runs must use `--release`. The runners reject debug builds.
