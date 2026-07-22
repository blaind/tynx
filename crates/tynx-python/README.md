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
```

Opt-in model capture records a supported Tensor-only forward on its first call and replays the
graph wholly in Rust on matching calls:

```python
@tynx.compile(fullgraph=True, static_argnames=("activation",))
def forward(x, activation="relu"):
    return x.relu() if activation == "relu" else x.sigmoid()
```

The initial cache uses exact tensor signatures. Parameter value updates are read dynamically and do
not recompile; structural parameter changes and different declared static values select another
graph. Unsupported operations fall back for the whole function by default or raise with
`fullgraph=True`. Closure variables, globals, and arbitrary object attributes are frozen at trace
time, so changing values must be passed as Tensor inputs or declared static arguments.

Loading a model needs an ONNX file on disk; any exported `.onnx` works:

```python
session = tynx.Session("model.onnx")
print(session.inputs, session.outputs)
```

## License

MIT or Apache-2.0, at your option.
