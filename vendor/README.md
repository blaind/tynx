# Patched upstream dependencies

Tynx carries small, reviewable patches for defects in its pinned Burn, CubeCL, and CubeK
revisions. The upstream repositories themselves are cloned into the ignored `vendor/repos/`
directory.

Prepare a fresh checkout before running Cargo, Maturin, or the benchmark workspace:

```sh
cargo vendor-patches
```

The standalone bootstrap tool checks out the exact revisions used by `Cargo.toml`, applies each
patch once, and rejects revision drift or unrelated local modifications. Generated clones can be
deleted at any time and recreated with the same command.
