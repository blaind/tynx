# CubeCL external WGPU patch series

This temporary patch series adds external WGPU buffer adoption and lease
retention to the exact CubeCL revision pinned by Tynx. It is intended only for
developing Tynx's engine interoperability until the corresponding CubeCL API is
available upstream.

Prepare a private patched checkout:

```console
cargo xtask external-wgpu prepare
```

Build or test Tynx against it without changing Cargo's global cache or the
workspace configuration:

```console
cargo xtask external-wgpu check
cargo xtask external-wgpu test
```

The adapter is additionally gated by Tynx's `external-wgpu` feature. The xtask
enables it automatically; ordinary `wgpu` and `vulkan` builds do not compile
against the temporary CubeCL-only API.

The checkout lives under `../.cache/tynx/cubecl-external-wgpu/`, outside the
Cargo workspace. The xtask verifies the base revision, patch commit headers,
patch hashes, and the applied checkout before use. Checks run from a temporary
source-only shadow workspace so Cargo never rewrites the real `Cargo.lock`.
Normal Cargo commands continue to use the upstream dependency.
