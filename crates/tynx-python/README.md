# tynx

**The ONNX runtime that trains.** Python bindings for [Tynx](https://github.com/blaind/tynx), a
lightweight neural network runtime built on [Burn](https://github.com/tracel-ai/burn).

Tynx loads ONNX models at runtime with no code generation step, runs on CPU and GPU through Burn
backends, and is growing into a full eager training library with a PyTorch-shaped API.

This is an early alpha. What works in this release:

- eager tensors with autograd: arithmetic, matmul, activations, reductions, `backward()`,
  `.grad`, `no_grad()`;
- `Parameter` values for building trainable modules;
- loading ONNX models and inspecting their inputs and outputs.

Model execution from Python, NumPy interop, layers, and optimizers land in upcoming releases.
See the [repository](https://github.com/blaind/tynx) for the roadmap, benchmarks, and the Rust
API.

```python
import tynx

x = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
loss = (x @ x).sum()
loss.backward()
print(x.grad.tolist())

session = tynx.Session("model.onnx")
print(session.inputs, session.outputs)
```

## License

MIT or Apache-2.0, at your option.
