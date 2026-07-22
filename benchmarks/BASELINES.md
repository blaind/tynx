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
