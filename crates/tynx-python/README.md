# tynx

**The ONNX runtime that trains.** Python bindings for [Tynx](https://github.com/blaind/tynx), a
lightweight neural network runtime built on [Burn](https://github.com/tracel-ai/burn).

Tynx loads ONNX models at runtime with no code generation step, runs on CPU and GPU through Burn
backends, and is growing into a full eager training library with a PyTorch-shaped API.

This is an early alpha. The current training surface includes:

- eager float32 autograd plus int64 targets and boolean masks;
- authored `nn` layers, composable losses, SGD/Adam/AdamW, parameter groups, and checkpoints;
- callable trainable ONNX models with stable parameters, multiple inputs/outputs, and structured
  trainability reports;
- Categorical and Normal distributions for deployed RL workloads;
- opt-in model and whole-training-step capture with native backward, optimizer, imported-model, and
  random-state replay;
- NumPy interchange and Flex CPU or WGPU execution from the same package.

The complete support table and differences from PyTorch/tinygrad are documented in the
[Python training API](https://github.com/blaind/tynx/blob/main/docs/python-training.md). Runnable
examples cover [authored training, imported fine-tuning, and captured PPO](https://github.com/blaind/tynx/tree/main/examples).

```python
import tynx

x = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
loss = (x @ x).sum()
loss.backward()
print(x.grad.tolist())
```

Opt-in capture records supported Tensor code on its first call and replays the graph wholly in Rust
on matching calls. It may cover a forward or the complete backward/optimizer step:

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

Load an ONNX file for inference with `Session`, or request a slot-backed trainable model with
`load(..., trainable="auto")`:

```python
session = tynx.Session("model.onnx")
print(session.inputs, session.outputs)

model = tynx.load("model.onnx", trainable="auto")
model.require_trainable()
```

## License

MIT or Apache-2.0, at your option.
