# Contributing

## ONNX conformance

The ignored conformance test uses the official ONNX node cases pinned to the same Burn-ONNX
revision as `onnx-ir`. The corpus is fetched once into the gitignored `.cache` directory.

```sh
cargo xtask conformance fetch
cargo xtask conformance
cargo xtask conformance --case test_relu
```

When an intentional runtime change alters the results, inspect `target/conformance-report.json`
and update the committed baseline with `cargo xtask conformance bless`.
