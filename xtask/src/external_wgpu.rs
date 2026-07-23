use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

const CUBECL_REVISION: &str = "74718177ac8357a70241dba74159700f955b1c7c";
const CUBECL_SOURCE: &str = "git+https://github.com/blaind/cubecl?";
const UPSTREAM_CUBECL_SOURCE: &str = "git+https://github.com/tracel-ai/cubecl";
const CRATES_IO_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";

const CUBECL_CRATES: &[&str] = &[
    "cubecl",
    "cubecl-common",
    "cubecl-core",
    "cubecl-cpp",
    "cubecl-cpu",
    "cubecl-cuda",
    "cubecl-hip",
    "cubecl-ir",
    "cubecl-macros",
    "cubecl-macros-internal",
    "cubecl-opt",
    "cubecl-runtime",
    "cubecl-spirv",
    "cubecl-std",
    "cubecl-wgpu",
    "cubecl-zspace",
];

const WGPU_CRATES: &[(&str, &str)] = &[
    ("naga", "29.0.4"),
    ("wgpu", "29.0.3"),
    ("wgpu-core", "29.0.4"),
    ("wgpu-core-deps-apple", "29.0.4"),
    ("wgpu-core-deps-emscripten", "29.0.4"),
    ("wgpu-core-deps-windows-linux-android", "29.0.4"),
    ("wgpu-hal", "29.0.4"),
    ("wgpu-naga-bridge", "29.0.4"),
    ("wgpu-types", "29.0.4"),
];

#[derive(Deserialize)]
struct Lockfile {
    package: Vec<Package>,
}

#[derive(Debug, Deserialize)]
struct Package {
    name: String,
    version: String,
    source: Option<String>,
}

pub(crate) fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let command = args.next().ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }

    let root = workspace_root();
    match command.as_str() {
        "topology" => verify_topology(&root),
        "check" => {
            verify_topology(&root)?;
            cargo(
                &root,
                &[
                    "check",
                    "-p",
                    "tynx",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "external-wgpu,flex,training,wgpu",
                ],
            )?;
            cargo(
                &root,
                &[
                    "check",
                    "-p",
                    "tynx",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "external-wgpu,flex,training,vulkan",
                ],
            )
        }
        "test" => {
            verify_topology(&root)?;
            cargo(
                &root,
                &[
                    "test",
                    "-p",
                    "tynx",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "external-wgpu,flex,training,wgpu",
                    "external_wgpu",
                    "--",
                    "--test-threads=1",
                ],
            )?;
            cargo(
                &root,
                &[
                    "test",
                    "-p",
                    "tynx-python",
                    "--locked",
                    "--no-default-features",
                    "--features",
                    "embedding-tests",
                    "--test",
                    "external_embedding",
                    "--",
                    "--test-threads=1",
                ],
            )
        }
        _ => Err(usage()),
    }
}

fn verify_topology(root: &Path) -> Result<(), String> {
    let lock_path = root.join("Cargo.lock");
    let lock_data = fs::read_to_string(&lock_path)
        .map_err(|error| format!("failed to read {}: {error}", lock_path.display()))?;
    let lockfile: Lockfile = toml::from_str(&lock_data)
        .map_err(|error| format!("failed to parse {}: {error}", lock_path.display()))?;
    verify_lockfile(&lockfile)?;

    println!(
        "external-WGPU topology verified: {} CubeCL packages at blaind/cubecl@{CUBECL_REVISION}; {} crates.io WGPU-family packages",
        CUBECL_CRATES.len(),
        WGPU_CRATES.len()
    );
    Ok(())
}

fn verify_lockfile(lockfile: &Lockfile) -> Result<(), String> {
    for name in CUBECL_CRATES {
        let packages = lockfile
            .package
            .iter()
            .filter(|package| package.name == *name)
            .collect::<Vec<_>>();
        if packages.len() != 1 {
            return Err(format!(
                "expected one {name} package identity, found {}: {}",
                packages.len(),
                package_identities(&packages)
            ));
        }
        let package = packages[0];
        let source = package.source.as_deref().unwrap_or("<workspace/path>");
        if !source.starts_with(CUBECL_SOURCE) || !source.ends_with(&format!("#{CUBECL_REVISION}")) {
            return Err(format!(
                "{name} {} resolves from {source}, expected blaind/cubecl@{CUBECL_REVISION}",
                package.version
            ));
        }
    }

    for package in &lockfile.package {
        let source = package.source.as_deref().unwrap_or("<workspace/path>");
        if source.starts_with(UPSTREAM_CUBECL_SOURCE) {
            return Err(format!(
                "{} {} still resolves from upstream CubeCL: {source}",
                package.name, package.version
            ));
        }
        if is_cubecl_git_package(package) && !CUBECL_CRATES.contains(&package.name.as_str()) {
            return Err(format!(
                "unexpected CubeCL git package {} {}; update the complete topology contract",
                package.name, package.version
            ));
        }
    }

    for (name, version) in WGPU_CRATES {
        let packages = lockfile
            .package
            .iter()
            .filter(|package| package.name == *name)
            .collect::<Vec<_>>();
        if packages.len() != 1 {
            return Err(format!(
                "expected one {name} package identity, found {}: {}",
                packages.len(),
                package_identities(&packages)
            ));
        }
        let package = packages[0];
        if package.version != *version || package.source.as_deref() != Some(CRATES_IO_SOURCE) {
            return Err(format!(
                "{name} resolves as {} from {}, expected {version} from crates.io",
                package.version,
                package.source.as_deref().unwrap_or("<workspace/path>")
            ));
        }
    }

    for package in &lockfile.package {
        if package.name.starts_with("wgpu")
            && !WGPU_CRATES
                .iter()
                .any(|(expected, _)| package.name == *expected)
        {
            return Err(format!(
                "unexpected WGPU-family package {} {}; update the complete topology contract",
                package.name, package.version
            ));
        }
    }

    Ok(())
}

fn is_cubecl_git_package(package: &Package) -> bool {
    (package.name == "cubecl" || package.name.starts_with("cubecl-"))
        && package
            .source
            .as_deref()
            .is_some_and(|source| source.starts_with("git+"))
}

fn package_identities(packages: &[&Package]) -> String {
    if packages.is_empty() {
        return "<missing>".to_string();
    }
    packages
        .iter()
        .map(|package| {
            format!(
                "{} ({})",
                package.version,
                package.source.as_deref().unwrap_or("<workspace/path>")
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn cargo(root: &Path, args: &[&str]) -> Result<(), String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(root)
        .args(args)
        .status()
        .map_err(|error| format!("failed to run cargo: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("cargo exited with {status}"))
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the workspace")
        .to_path_buf()
}

fn usage() -> String {
    "usage: cargo xtask external-wgpu <topology | check | test>".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_lockfile() -> Lockfile {
        let mut package = CUBECL_CRATES
            .iter()
            .map(|name| Package {
                name: (*name).to_string(),
                version: "0.11.0-pre.1".to_string(),
                source: Some(format!(
                    "{CUBECL_SOURCE}rev={CUBECL_REVISION}#{CUBECL_REVISION}"
                )),
            })
            .collect::<Vec<_>>();
        package.extend(WGPU_CRATES.iter().map(|(name, version)| Package {
            name: (*name).to_string(),
            version: (*version).to_string(),
            source: Some(CRATES_IO_SOURCE.to_string()),
        }));
        Lockfile { package }
    }

    #[test]
    fn accepts_the_complete_pinned_dependency_family() {
        verify_lockfile(&valid_lockfile()).unwrap();
    }

    #[test]
    fn rejects_a_residual_upstream_cubecl_package() {
        let mut lockfile = valid_lockfile();
        lockfile.package.push(Package {
            name: "cubecl-new".to_string(),
            version: "0.11.0-pre.1".to_string(),
            source: Some(format!(
                "{UPSTREAM_CUBECL_SOURCE}?rev={CUBECL_REVISION}#{CUBECL_REVISION}"
            )),
        });

        let error = verify_lockfile(&lockfile).unwrap_err();

        assert!(error.contains("still resolves from upstream CubeCL"));
    }

    #[test]
    fn rejects_duplicate_wgpu_type_identities() {
        let mut lockfile = valid_lockfile();
        lockfile.package.push(Package {
            name: "wgpu".to_string(),
            version: "30.0.0".to_string(),
            source: Some(CRATES_IO_SOURCE.to_string()),
        });

        let error = verify_lockfile(&lockfile).unwrap_err();

        assert!(error.contains("expected one wgpu package identity, found 2"));
    }

    #[test]
    fn rejects_an_unexpected_wgpu_family_member() {
        let mut lockfile = valid_lockfile();
        lockfile.package.push(Package {
            name: "wgpu-new-platform".to_string(),
            version: "29.0.4".to_string(),
            source: Some(CRATES_IO_SOURCE.to_string()),
        });

        let error = verify_lockfile(&lockfile).unwrap_err();

        assert!(error.contains("unexpected WGPU-family package"));
    }
}
