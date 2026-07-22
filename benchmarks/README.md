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

Set `TYNX_BENCH_THREADS` to `1` for a matched single-thread run or `auto` for each runtime's
default pool. Tynx and burn-onnx AOT need the `multithread` feature in both modes so the
single-thread and runtime-default measurements use the same Rayon-enabled build:

```sh
TYNX_BENCH_THREADS=1 cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p tynx-bench --features multithread
TYNX_BENCH_THREADS=auto cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p tynx-bench --features multithread
TYNX_BENCH_THREADS=1 cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p ort-bench
TYNX_BENCH_THREADS=auto cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p ort-bench
```

Numeric values other than `1` request an exact thread count. JSON reports include the threading
runtime, requested mode, and actual pool size where it can be queried. Without `multithread`, the
Burn runners remain serial and reject explicit counts greater than one.

Each runner executes every registered case by default and emits one JSON report array. Set
`TYNX_BENCH_CASE` to limit a run to one case. The manual GitHub workflow invokes each engine in
matched single-thread and automatic runtime-default modes, adds a comparison table to the job summary,
and uploads its JSON reports. Leave the workflow iteration inputs blank to use the per-case
defaults.

## Training

The training suite compares an imported Tynx model with burn-onnx generated AOT Rust using the
same deterministic ONNX weights, rotating device-resident batches, MSE loss, plain SGD update, Burn
revision, and backend. The initial model is a two-layer `784 -> 512 -> 10` Gemm/ReLU MLP because
live trainable state currently supports Gemm, Linear, and MatMul. Trainable convolution benchmarks
will be added after slot-backed Conv execution lands.

```sh
TYNX_BENCH_CASE=mlp-784-512-10-b64 \
TYNX_BENCH_THREADS=1 \
cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p tynx-training-bench --features multithread

TYNX_BENCH_CASE=mlp-784-512-10-b64 \
TYNX_BENCH_THREADS=1 \
cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p burn-aot-training-bench --features multithread
```

The registry provides batch sizes 64, 1024, and 4096. Omit `TYNX_BENCH_CASE` to run the full
scaling sweep. The default boundary is one complete `zero_grad -> forward -> loss -> backward ->
optimizer step -> device sync` operation. Select another boundary or synchronization policy with:

```sh
TYNX_BENCH_TRAINING_MODE=forward_loss       # or forward_backward, train_step
TYNX_BENCH_SYNC_POLICY=each_step            # or final_only
```

`final_only` means no explicit sync between steps and one sync after each reset block. It does not
claim that backward is asynchronous. The report records empty-sync overhead so fixed device
round-trip cost remains visible.

Steady-state timing starts only after two ten-step median windows move by at most 2 percent for
three consecutive comparisons, with a minimum of 20 and maximum of 200 warmup steps. The actual
count and convergence status are reported. Set `TYNX_BENCH_WARMUP_MIN` and
`TYNX_BENCH_WARMUP_MAX` to override the bounds. State is restored outside timing at regular block
boundaries so loss and gradient magnitudes remain stationary.

Every result includes parse, preparation, cold-step, warmup, steady-step, throughput, estimated
training GFLOP/s, cache policy, synchronization policy, model SHA-256, parameter SHA-256, exact
trainable/frozen/gradient sets, and a five-step loss trajectory. Compare the two JSON outputs with
`jq -s -e -f benchmarks/check-training-trajectory.jq tynx.json burn-aot.json`; this requires matching
CPU parameter hashes and loss trajectories. Training benchmarks are intentionally not wired into
CI yet, and performance results have no latency gate.

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
