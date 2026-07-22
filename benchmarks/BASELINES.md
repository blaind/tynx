# Reference benchmark snapshots

These snapshots are reproducible reference points, not performance gates. Compare results only when
the workload, backend, threading mode, and hardware are equivalent.

## 2026-07-22 CPU inference

- Revision: `c2ab96a`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 logical CPUs)
- Rust: `rustc 1.98.0-nightly (423e3d252 2026-05-24)`
- Backend: Burn Flex, Rayon fixed to one thread
- Case: `matmul-64x64`, f32
- Warmup / measured iterations: 20 / 100
- Cold: 0.510155 ms
- Median: 0.005370 ms
- p95: 0.005580 ms
- Mean: 0.005454 ms
- Estimated throughput: 97.633 GFLOP/s
- Output checksum: 97.28019905090332

Command:

```sh
TYNX_BENCH_CASE=matmul-64x64 \
TYNX_BENCH_WARMUP=20 \
TYNX_BENCH_ITERATIONS=100 \
TYNX_BENCH_THREADS=1 \
cargo run --manifest-path benchmarks/Cargo.toml --locked --release \
  -p tynx-bench --features multithread
```

This is the first durable local reference recorded after the training foundation landed. It cannot
retroactively serve as a pre-autodiff measurement; it is the comparison point for subsequent work.

## 2026-07-22 Python model capture dispatch

- Revision: `336243d`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 logical CPUs)
- Python: CPython 3.13.7 with a release-mode abi3 extension
- Backend: Burn Flex CPU autodiff device
- Case: 32 sequential ReLUs over one 16-element f32 tensor
- Warmup / measured iterations / repeats: 100 / 2,000 / 7
- Captured IR nodes: 33
- Eager median: 28.497 µs per call
- Captured median: 23.472 µs per call
- Dispatch speedup: 1.214×

Command:

```sh
maturin develop --release --locked -m crates/tynx-python/Cargo.toml
python benchmarks/python-capture.py \
  --depth 32 --size 16 --warmup 100 --iterations 2000 --repeats 7
```

This microbenchmark isolates removal of repeated Python/PyO3 operation dispatch. Both paths execute
the same Burn tensor operations, so it is not a claim of faster kernels or backend fusion.

## 2026-07-22 Python whole-step training capture

- Revision: `6a8ff89`
- CPU: AMD Ryzen 9 9950X3D 16-Core Processor (16 cores, 32 logical CPUs)
- Python: CPython 3.13.7 with a release-mode abi3 extension
- Backend: Burn Flex CPU autodiff device
- Case: authored `32 -> 64 -> 8` MLP, batch 32, MSE, plain SGD
- Boundary: `zero_grad -> forward -> loss -> backward -> optimizer step`, then final block sync
- Warmup / measured iterations / repeats: 50 / 200 / 7
- Captured IR nodes: 20
- Eager median: 49.009 µs per step
- Captured median: 50.500 µs per step
- Captured/eager speed ratio: 0.970×

Command:

```sh
maturin develop --release --locked -m crates/tynx-python/Cargo.toml
BURN_DEVICE=flex python benchmarks/python-training-capture.py
```

This is deliberately non-gating. On this small workload, whole-step capture is effectively
break-even but 3.0% slower than eager execution. It proves removal of Python re-entry and provides a
stable optimization target; it is not evidence of a training throughput benefit yet.
