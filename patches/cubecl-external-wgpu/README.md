# CubeCL external WGPU patch series

These six patches are the reviewable source record for Tynx's temporary CubeCL
fork. Applied in lexical order to upstream CubeCL
`ba103c7f118524652647338dec09abf69a24a53e`, they produce the exact tree stored
at `blaind/cubecl@74718177ac8357a70241dba74159700f955b1c7c`:

```text
3e706371a24c3fde1618bb7e19246bef5be8e422
```

Tynx builds do not apply these files. The workspace Cargo patch redirects the
complete CubeCL dependency graph to that pinned fork so ordinary downstream
Cargo builds are reproducible without a preparation step or a local checkout.

Validate the pinned dependency topology and external-WGPU behavior with:

```console
cargo xtask external-wgpu check
cargo xtask external-wgpu test
```

The adapter remains gated by Tynx's `external-wgpu` feature. The xtask verifies
that every CubeCL package resolves to the fork revision and that the WGPU family
has one released version identity. It does not apply or hash this patch series.

Once the external-device and buffer-adoption API is available upstream, replace
the workspace Cargo patch with the qualified upstream revision and retire this
directory.
