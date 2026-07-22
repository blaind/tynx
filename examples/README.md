# Python training examples

Run these from the repository root after installing the local Python package:

```sh
python examples/authored_training.py
python examples/imported_finetuning.py
python examples/captured_ppo.py
```

- `authored_training.py` uses ordinary eager `nn` layers and Adam.
- `imported_finetuning.py` loads an ONNX model into stable parameter slots and fine-tunes it with
  the same eager optimizer API.
- `captured_ppo.py` captures an imported multi-output actor-critic forward, a user-composed PPO
  loss, backward, and Adam update. Matching calls replay wholly in Rust.

The tiny `.onnx.hex` fixtures are checked in to keep the examples self-contained. Real applications
pass their exported `.onnx` path directly to `tynx.load`.
