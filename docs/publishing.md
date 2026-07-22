# Publishing

`.github/workflows/release.yml` publishes wheels to PyPI and a GitHub release. There is no sdist
or runtime crate release while dependencies remain git-pinned.

## Release

The sole version is `[workspace.package] version` in the root `Cargo.toml`. Crates inherit it,
maturin supplies it to the wheel, and the tag must be `v<version>`.

1. Land the release on `main` with green CI.
2. Bump the workspace version and run `cargo update --workspace`.
3. If needed, update `crates/tynx-python/README.md` for the PyPI page.
4. Commit and push, then tag that version:

   ```sh
   git tag v0.1.2
   git push origin v0.1.2
   ```

CI builds `--profile dist --features wgpu` wheels for Linux x86_64/aarch64, macOS arm64, and
Windows x86_64. Each must pass the 20 MiB limit, build-path scan, installation, and CPU smoke
test. Tag builds then publish through PyPI trusted publishing and create a GitHub release.

## Dry run

Run **Actions -> release -> Run workflow**. Manual runs execute all builds and checks but cannot
publish because publishing requires a `v*` tag.

## One-time setup

Configure the PyPI `tynx` trusted publisher for repository `blaind/tynx`, workflow `release.yml`,
and environment `pypi`. Optionally require a reviewer on that GitHub environment.

## Recovery

Replace an unpublished tag that points to the wrong version or commit:

```sh
git tag -d v0.1.2
git push origin :refs/tags/v0.1.2
# fix and commit, then recreate the tag
```

Published PyPI versions are immutable. Publish a new version to fix one; yank the old version if
installers should stop selecting it.
