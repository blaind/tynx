# Python training API

Tynx is eager by default. Each ordinary tensor operation executes through Burn immediately and
builds a dynamic autodiff tape when gradient tracking is enabled. A backend may still queue or fuse
device work internally; that does not change Python-visible eager semantics.

## Supported training surface

| Area | Current support |
| --- | --- |
| Tensor storage | `float32`, `int64`, and `bool`; ranks 1–6; Python scalars and rectangular sequences; NumPy interchange |
| Differentiation | Reverse-mode autograd for `float32`, leaf and parameter gradients, explicit or scalar `backward`, accumulation until `zero_grad`, `detach`, and `no_grad` |
| Core math | Broadcast arithmetic, matmul, reductions, extrema, activations, softmax/log-softmax, clamp, reshape/flatten/transpose/permute/squeeze/unsqueeze, masks/where, and differentiable gather |
| Layers | `Linear`, `Conv2d`, `LayerNorm`, `BatchNorm1d`, `BatchNorm2d`, `Dropout`, `ReLU`, and `Sequential` |
| Losses | MSE, cross entropy, and binary cross entropy with logits, each with `none`, `mean`, or `sum` reduction |
| Optimizers | SGD, Adam, and AdamW; parameter groups; gradient clipping; native resumable state |
| State | Stable `Parameter`/`Buffer` identity, deterministic named traversal, state dictionaries, hard/Polyak target updates, and atomic model-plus-optimizer checkpoints |
| Distributions | Categorical and Normal `sample`, `log_prob`, and `entropy`; shared or explicit seeding; detached samples |
| Imported ONNX | Multi-input/output calls, stable trainable initializer slots, user-composed losses, structured trainability reports, and shared authored/imported optimizer loops |
| Capture | Exact-signature forward or whole-step replay, structured outputs, stable parameter slots, backward/optimizer effects, imported model calls, and advancing Dropout/Categorical/Normal RNG |

NumPy inputs normalize common host-default dtypes at the boundary: `float64` is deterministically
narrowed to `float32`, while `int32` is losslessly widened to `int64`. Passing the corresponding
explicit `dtype="float32"` or `dtype="int64"` is accepted; contradictory or unsupported explicit
dtypes raise instead of silently selecting another representation.

The checked-in [examples](../examples/README.md) cover authored eager training, imported ONNX
fine-tuning, and a captured imported PPO update.

## PyTorch differences

- There are no rank-zero tensors yet. A scalar reduction has shape `(1,)`, `numel == 1`, supports
  `item()`, and can be the root of `backward()`.
- Per-dimension reductions follow PyTorch's `keepdim=False` default at the Python boundary, subject
  to the rank-one scalar floor above.
- Training is currently `float32` only. Integer and boolean tensors participate in indexing,
  selection, targets, and masks but do not carry gradients.
- Tensor ranks above six, general `tensor[...]` slicing, sparse tensors, distributed training, and
  mixed precision are not implemented.
- In-place arithmetic such as `+=` is rejected. Optimizer publication and explicit `copy_` into a
  stable `Parameter` or `Buffer` are off-tape state changes.
- `backward()` frees its eager graph. Gradients remain available and accumulate until explicitly
  cleared, matching the usual PyTorch optimizer loop.
- `nn.Module` is available as a convenience but is not mandatory. State discovery deliberately
  traverses `__dict__`, lists, tuples, dictionaries, and Tynx state objects without invoking
  properties, descriptors, arbitrary iterators, or `__getattr__`.
- Imported BatchNorm uses fixed running statistics and imported Dropout must be inactive in the
  initial training release. The trainability report rejects unsupported gradient paths before the
  first update.
- Model state access and optimizer publication are serialized in v1; concurrent readers cannot
  observe a partially published multi-parameter update.

## Capture differences from PyTorch and tinygrad

`tynx.compile` is an explicit tracing cache, not PyTorch Dynamo bytecode analysis. The first call
executes eagerly while recording supported tensor operations; matching later calls replay a native
Rust graph. It currently guards exact tensor shape, dtype, device, declared static arguments, and
parameter structure. Ordinary parameter value updates do not cause recompilation.

Python values reachable only through closures, globals, or arbitrary object attributes are frozen
at trace time. Pass changing values as Tensor arguments, or expose call arguments through
`static_argnames`. Unsupported behavior falls back for the whole function by default;
`fullgraph=True` raises instead. Failed traces are torn down and do not poison compatible cached
graphs.

This resembles tinygrad's explicit `TinyJit` usage more than PyTorch's transparent compiler, but
ordinary Tynx execution is eager rather than a user-visible lazy schedule. Capture is optional and
does not define eager semantics. The current whole-step CPU benchmark is approximately break-even
and is tracked as an optimization target, not advertised as a universal speedup.

## Imported-model boundary

Use `tynx.load(path, trainable="auto")` to request a trainable model. Loading for inference does not
imply that every selected output has a supported backward path. Inspect
`model.trainability_report()` or call `model.require_trainable()` before constructing the optimizer.
Initializers are classified as trainable parameters, mutable buffers, or constants; ambiguous and
unsupported uses are reported explicitly.

Imported and authored models share the same eager loop:

```python
optimizer.zero_grad()
prediction = model(input)
loss = tynx.nn.functional.mse_loss(prediction, target)
loss.backward()
optimizer.step()
```

Use `tynx.synchronize()` when a host boundary requires completion of queued accelerated work.
`BURN_DEVICE=flex` explicitly selects the CPU backend when a WGPU-enabled wheel is installed.
