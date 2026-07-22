# Tynx

**The ONNX runtime that trains.**

Tynx runs and fine-tunes ONNX models, and trains models you author in Python, on one small,
self-contained runtime built on [Burn](https://github.com/tracel-ai/burn). The same engine runs in
Rust, Python, and the browser.

- **One ~15 MB wheel.** `pip install tynx`. No CUDA toolkit, no cuDNN, no ROCm, no system
  dependencies. Kernels are JIT-generated at runtime, not shipped as binaries.
- **Every GPU vendor.** wgpu (Vulkan / Metal / DX12) means the same wheel trains on NVIDIA, AMD,
  Intel, and Apple hardware. CPU works everywhere via the Flex backend.
- **PyTorch-shaped, eager by default.** Ordinary Python classes, a dynamic autograd tape, explicit
  `zero_grad()` / `backward()` / `step()`. `tynx.compile` is an optional accelerator, never a
  requirement.
- **ONNX models are trainable objects.** Load a model exported from PyTorch or anywhere else and
  fine-tune it in place; the train-here / deploy-there split disappears.

```python
import tynx as tx

model = tx.load("policy.onnx", trainable="auto")
optimizer = tx.optim.Adam(model.parameters(), lr=3e-4)

for x, target in loader:
    optimizer.zero_grad()
    loss = model(x).cross_entropy(target)
    loss.backward()
    optimizer.step()
```

## Install

```sh
pip install tynx            # Python: CPU + wgpu GPU, ~15 MB
cargo add tynx              # Rust: inference core
cargo add tynx -F training  # Rust: + autodiff, parameters, optimizers
```

## Backends

The interpreter is backend-agnostic and executes through a Burn device. Flex, an optimized CPU
backend, is enabled by default and also works on `wasm32-unknown-unknown`.

WGPU/WebGPU, browser WebGPU, Vulkan, and CUDA feature wiring is being carried over and is not part
of the current default build.

## Benchmarks

Tynx includes a reproducible benchmark suite comparing its runtime with burn-onnx AOT and ONNX
Runtime. See the [benchmark suite](benchmarks/README.md) for the workloads, methodology, and local
commands. [CI benchmark runs](https://github.com/blaind/tynx/actions/workflows/benchmarks.yml)
publish job summaries and downloadable JSON results.

## Relationship to Burn

Tynx is built on Burn and complements [burn-onnx](https://github.com/tracel-ai/burn-onnx):
burn-onnx generates Rust code from ONNX at build time for maximum AOT performance; Tynx loads ONNX
at runtime with no codegen step, and trains. Burn supplies devices, kernels, autodiff, and fusion;
Tynx supplies the dynamic runtime, the ONNX graph frontend, and the Python experience.

## License

Tynx is licensed under either the MIT License or the Apache License, Version 2.0, at your option.
