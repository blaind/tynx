use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

const CUBECL_REVISION: &str = "74718177ac8357a70241dba74159700f955b1c7c";
const CUBECL_SOURCE: &str = "git+https://github.com/blaind/cubecl";
const WGPU_SOURCE: &str = "registry+https://github.com/rust-lang/crates.io-index";
const WGPU_VERSION: &str = "29.0.3";

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

#[derive(Deserialize)]
struct Lockfile {
    package: Vec<Package>,
}

#[derive(Deserialize)]
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

    for name in CUBECL_CRATES {
        let packages = lockfile
            .package
            .iter()
            .filter(|package| package.name == *name)
            .collect::<Vec<_>>();
        if packages.is_empty() {
            return Err(format!("dependency topology is missing {name}"));
        }
        for package in packages {
            let source = package.source.as_deref().unwrap_or("<workspace/path>");
            if !source.starts_with(CUBECL_SOURCE) || !source.contains(CUBECL_REVISION) {
                return Err(format!(
                    "{name} {} resolves from {source}, expected blaind/cubecl@{CUBECL_REVISION}",
                    package.version
                ));
            }
        }
    }

    let wgpu = lockfile
        .package
        .iter()
        .filter(|package| package.name == "wgpu")
        .collect::<Vec<_>>();
    if wgpu.len() != 1 {
        let identities = wgpu
            .iter()
            .map(|package| {
                format!(
                    "{} ({})",
                    package.version,
                    package.source.as_deref().unwrap_or("<workspace/path>")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "expected one WGPU package identity, found {}: {identities}",
            wgpu.len()
        ));
    }
    let wgpu = wgpu[0];
    if wgpu.version != WGPU_VERSION || wgpu.source.as_deref() != Some(WGPU_SOURCE) {
        return Err(format!(
            "wgpu resolves as {} from {}, expected {WGPU_VERSION} from crates.io",
            wgpu.version,
            wgpu.source.as_deref().unwrap_or("<workspace/path>")
        ));
    }

    println!(
        "external-WGPU topology verified: blaind/cubecl@{CUBECL_REVISION}, crates.io wgpu {WGPU_VERSION}"
    );
    Ok(())
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
