use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const CUBECL_REPOSITORY: &str = "https://github.com/tracel-ai/cubecl";
const CUBECL_REVISION: &str = "ba103c7f118524652647338dec09abf69a24a53e";
const PATCH_DIRECTORY: &str = "patches/cubecl-external-wgpu";
const CACHE_DIRECTORY: &str = ".cache/tynx/cubecl-external-wgpu";
const SHADOW_DIRECTORY: &str = ".cache/tynx/external-wgpu-workspace";
const MARKER: &str = "tynx-external-wgpu-series";
const PATCHED_TREE: &str = "1473e674f0199ac43b401410fa36e48a897a8370";

struct Patch {
    file: &'static str,
    commit: &'static str,
    blob: &'static str,
}

const PATCHES: &[Patch] = &[
    Patch {
        file: "0001-Register-external-WGPU-buffers.patch",
        commit: "b486a95f34b711bb65caf733aae8f489065893b0",
        blob: "452f90bfc9146132c29bc28b8a9dbd690ce436dd",
    },
    Patch {
        file: "0002-Retain-external-WGPU-leases.patch",
        commit: "790f2d04a06af514854b6f024326190cb61464ac",
        blob: "6bc8354f5f9f6e4f45522bed9bdc21afbed67f72",
    },
    Patch {
        file: "0003-Initialize-external-compiler-runtimes.patch",
        commit: "eba552aa8a5eadef0863b78c04d38e3d8896867e",
        blob: "3ecc7810c45a29418924b81302c144278bb64da9",
    },
];

const CUBECL_CRATES: &[(&str, &str)] = &[
    ("cubecl", "crates/cubecl"),
    ("cubecl-common", "crates/cubecl-common"),
    ("cubecl-core", "crates/cubecl-core"),
    ("cubecl-cpp", "crates/cubecl-cpp"),
    ("cubecl-cpu", "crates/cubecl-cpu"),
    ("cubecl-cuda", "crates/cubecl-cuda"),
    ("cubecl-hip", "crates/cubecl-hip"),
    ("cubecl-ir", "crates/cubecl-ir"),
    ("cubecl-macros", "crates/cubecl-macros"),
    ("cubecl-macros-internal", "crates/cubecl-macros-internal"),
    ("cubecl-opt", "crates/cubecl-opt"),
    ("cubecl-runtime", "crates/cubecl-runtime"),
    ("cubecl-spirv", "crates/cubecl-spirv"),
    ("cubecl-std", "crates/cubecl-std"),
    ("cubecl-wgpu", "crates/cubecl-wgpu"),
    ("cubecl-zspace", "crates/cubecl-zspace"),
];

pub(crate) fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let command = args.next().ok_or_else(usage)?;
    if args.next().is_some() {
        return Err(usage());
    }

    let root = workspace_root();
    match command.as_str() {
        "prepare" => {
            prepare(&root)?;
            Ok(())
        }
        "check" => {
            let checkout = prepare(&root)?;
            cargo(
                &root,
                &checkout,
                &[
                    "check",
                    "-p",
                    "tynx",
                    "--no-default-features",
                    "--features",
                    "external-wgpu,flex,training,wgpu",
                ],
            )?;
            cargo(
                &root,
                &checkout,
                &[
                    "check",
                    "-p",
                    "tynx",
                    "--no-default-features",
                    "--features",
                    "external-wgpu,flex,training,vulkan",
                ],
            )
        }
        "test" => {
            let checkout = prepare(&root)?;
            cargo(
                &root,
                &checkout,
                &[
                    "test",
                    "-p",
                    "tynx",
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
                &checkout,
                &[
                    "test",
                    "-p",
                    "tynx",
                    "--no-default-features",
                    "--features",
                    "external-wgpu,flex,training,wgpu",
                    "--test",
                    "external_lifecycle",
                ],
            )
        }
        _ => Err(usage()),
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the workspace")
        .to_path_buf()
}

fn prepare(root: &Path) -> Result<PathBuf, String> {
    verify_patches(root)?;

    let cache_root = root
        .parent()
        .ok_or_else(|| "workspace root has no parent for the CubeCL cache".to_string())?;
    let checkout = cache_root.join(CACHE_DIRECTORY).join(CUBECL_REVISION);
    if checkout.exists() {
        verify_checkout(&checkout)?;
        println!("using patched CubeCL checkout at {}", checkout.display());
        return Ok(checkout);
    }

    let parent = checkout
        .parent()
        .ok_or_else(|| "CubeCL cache has no parent".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;

    println!(
        "fetching CubeCL at {CUBECL_REVISION} into {}",
        checkout.display()
    );
    command(
        root,
        "git",
        &[
            OsString::from("init"),
            OsString::from("--quiet"),
            checkout.as_os_str().to_owned(),
        ],
    )?;
    command(
        &checkout,
        "git",
        &[
            OsString::from("remote"),
            OsString::from("add"),
            OsString::from("origin"),
            OsString::from(CUBECL_REPOSITORY),
        ],
    )?;
    command(
        &checkout,
        "git",
        &[
            OsString::from("fetch"),
            OsString::from("--quiet"),
            OsString::from("--depth=1"),
            OsString::from("origin"),
            OsString::from(CUBECL_REVISION),
        ],
    )?;
    command(
        &checkout,
        "git",
        &[
            OsString::from("checkout"),
            OsString::from("--quiet"),
            OsString::from("--detach"),
            OsString::from("FETCH_HEAD"),
        ],
    )?;

    for patch_path in patch_paths(root) {
        command(
            &checkout,
            "git",
            &[
                OsString::from("apply"),
                OsString::from("--index"),
                OsString::from("--check"),
                patch_path.as_os_str().to_owned(),
            ],
        )?;
        command(
            &checkout,
            "git",
            &[
                OsString::from("apply"),
                OsString::from("--index"),
                patch_path.as_os_str().to_owned(),
            ],
        )?;
    }
    command(
        &checkout,
        "git",
        &[OsString::from("diff"), OsString::from("--check")],
    )?;

    let marker = marker_contents();
    let marker_path = checkout.join(".git").join(MARKER);
    fs::write(&marker_path, marker)
        .map_err(|error| format!("failed to write {}: {error}", marker_path.display()))?;
    verify_checkout(&checkout)?;
    println!("prepared patched CubeCL checkout at {}", checkout.display());
    Ok(checkout)
}

fn verify_patches(root: &Path) -> Result<(), String> {
    for patch in PATCHES {
        let path = root.join(PATCH_DIRECTORY).join(patch.file);
        let contents = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let expected_header = format!("From {} Mon Sep 17 00:00:00 2001", patch.commit);
        if contents.lines().next() != Some(expected_header.as_str()) {
            return Err(format!(
                "{} does not contain expected commit {}",
                path.display(),
                patch.commit
            ));
        }

        let actual_blob = output(
            root,
            "git",
            &[OsString::from("hash-object"), path.as_os_str().to_owned()],
        )?;
        if actual_blob.trim() != patch.blob {
            return Err(format!(
                "{} has blob hash {}, expected {}",
                path.display(),
                actual_blob.trim(),
                patch.blob
            ));
        }
    }
    Ok(())
}

fn verify_checkout(checkout: &Path) -> Result<(), String> {
    let head = output(
        checkout,
        "git",
        &[OsString::from("rev-parse"), OsString::from("HEAD")],
    )?;
    if head.trim() != CUBECL_REVISION {
        return Err(invalid_cache(checkout, "base revision differs"));
    }

    let marker_path = checkout.join(".git").join(MARKER);
    let marker = fs::read_to_string(&marker_path)
        .map_err(|_| invalid_cache(checkout, "patch marker is missing"))?;
    if marker != marker_contents() {
        return Err(invalid_cache(checkout, "patch marker differs"));
    }

    let tree = output(checkout, "git", &[OsString::from("write-tree")])?;
    if tree.trim() != PATCHED_TREE {
        return Err(invalid_cache(checkout, "patched tree differs"));
    }
    command(
        checkout,
        "git",
        &[OsString::from("diff"), OsString::from("--quiet")],
    )
    .map_err(|_| invalid_cache(checkout, "working tree differs from patched index"))
}

fn marker_contents() -> String {
    let mut marker = format!("base {CUBECL_REVISION}\ntree {PATCHED_TREE}\n");
    for patch in PATCHES {
        marker.push_str(&format!("{} {}\n", patch.commit, patch.blob));
    }
    marker
}

fn patch_paths(root: &Path) -> Vec<PathBuf> {
    PATCHES
        .iter()
        .map(|patch| root.join(PATCH_DIRECTORY).join(patch.file))
        .collect()
}

fn cargo(root: &Path, checkout: &Path, args: &[&str]) -> Result<(), String> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let shadow = prepare_shadow_workspace(root)?;
    let target = root.join("target/external-wgpu");
    let mut command = Command::new(cargo);
    command.current_dir(&shadow).env("CARGO_TARGET_DIR", target);
    for (name, relative_path) in CUBECL_CRATES {
        let crate_path = checkout.join(relative_path);
        let crate_path = crate_path
            .to_str()
            .ok_or_else(|| format!("path is not valid UTF-8: {}", crate_path.display()))?;
        let value = format!("patch.\"{CUBECL_REPOSITORY}\".{name}.path={crate_path:?}");
        command.arg("--config").arg(value);
    }
    command.args(args);
    let result = run_command(command, "cargo");
    let cleanup = fs::remove_dir_all(&shadow)
        .map_err(|error| format!("failed to remove {}: {error}", shadow.display()));
    result.and(cleanup)
}

fn prepare_shadow_workspace(root: &Path) -> Result<PathBuf, String> {
    let cache_root = root
        .parent()
        .ok_or_else(|| "workspace root has no parent for the shadow workspace".to_string())?;
    let shadow = cache_root.join(SHADOW_DIRECTORY);
    if shadow.exists() {
        fs::remove_dir_all(&shadow)
            .map_err(|error| format!("failed to remove {}: {error}", shadow.display()))?;
    }
    fs::create_dir_all(&shadow)
        .map_err(|error| format!("failed to create {}: {error}", shadow.display()))?;

    let files = output(
        root,
        "git",
        &[
            OsString::from("ls-files"),
            OsString::from("--cached"),
            OsString::from("--others"),
            OsString::from("--exclude-standard"),
        ],
    )?;
    for relative in files.lines().filter(|line| !line.is_empty()) {
        let relative = Path::new(relative);
        if relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!(
                "refusing unexpected workspace path {}",
                relative.display()
            ));
        }
        let source = root.join(relative);
        if !source.is_file() {
            continue;
        }
        let destination = shadow.join(relative);
        let parent = destination
            .parent()
            .ok_or_else(|| format!("workspace file has no parent: {}", destination.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        fs::copy(&source, &destination).map_err(|error| {
            format!(
                "failed to copy {} to {}: {error}",
                source.display(),
                destination.display()
            )
        })?;
    }
    Ok(shadow)
}

fn command(cwd: &Path, program: &str, args: &[OsString]) -> Result<(), String> {
    let mut command = Command::new(program);
    command.current_dir(cwd).args(args);
    run_command(command, program)
}

fn output(cwd: &Path, program: &str, args: &[OsString]) -> Result<String, String> {
    let output = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!("{program} exited with {}", output.status));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("{program} produced non-UTF-8 output: {error}"))
}

fn run_command(mut command: Command, description: &str) -> Result<(), String> {
    std::io::stdout()
        .flush()
        .map_err(|error| format!("failed to flush output: {error}"))?;
    let status = command
        .status()
        .map_err(|error| format!("failed to run {description}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{description} exited with {status}"))
    }
}

fn invalid_cache(checkout: &Path, reason: &str) -> String {
    format!(
        "invalid CubeCL cache at {} ({reason}); remove that tool-owned directory and retry",
        checkout.display()
    )
}

fn usage() -> String {
    "usage: cargo xtask external-wgpu <prepare | check | test>".to_string()
}
