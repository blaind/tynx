# Tynx

**Run ONNX models anywhere with Burn.**

Tynx is an early-stage ONNX runtime for Rust, Python, and WebAssembly, built on the
[Burn](https://github.com/tracel-ai/burn) deep-learning framework.

Unlike [Burn-ONNX](https://github.com/tracel-ai/burn-onnx), which generates Rust code and weight
files, Tynx loads and executes ONNX graphs directly at runtime without model-specific code
generation or recompilation.

## Backends

The interpreter is backend-agnostic and executes through a Burn device. Flex, an optimized CPU
backend, is enabled by default and also works on `wasm32-unknown-unknown`.

WGPU/WebGPU, browser WebGPU, Vulkan, and CUDA feature wiring is being carried over and is not part
of the current default build.

## License

Tynx is licensed under either the MIT License or the Apache License, Version 2.0, at your option.
