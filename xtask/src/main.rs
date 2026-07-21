use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use serde::Deserialize;

const REGISTRY: &str = "crates/tynx/tests/conformance.json";

#[derive(Deserialize)]
struct Registry {
    source: Source,
}

#[derive(Deserialize)]
struct Source {
    repository: String,
    revision: String,
    path: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    if args.next().as_deref() != Some("conformance") {
        return Err(usage());
    }

    let command = args.next();
    let mut case = None;
    let mut bless = false;
    match command.as_deref() {
        None => {}
        Some("fetch") => {
            if args.next().is_some() {
                return Err(usage());
            }
            let root = workspace_root();
            let registry = load_registry(&root)?;
            fetch(&root, &registry.source)?;
            return Ok(());
        }
        Some("bless") => bless = true,
        Some("--case") => {
            case = Some(args.next().ok_or_else(usage)?);
        }
        Some(_) => return Err(usage()),
    }

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--case" if !bless && case.is_none() => {
                case = Some(args.next().ok_or_else(usage)?);
            }
            _ => return Err(usage()),
        }
    }

    let root = workspace_root();
    let registry = load_registry(&root)?;
    let checkout = fetch(&root, &registry.source)?;
    let corpus = checkout.join(&registry.source.path);
    let report = root.join("target/conformance-report.json");
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut process = Command::new(cargo);
    process
        .current_dir(&root)
        .args([
            "test",
            "-p",
            "tynx",
            "--test",
            "conformance",
            "--",
            "--ignored",
            "--nocapture",
        ])
        .env("TYNX_ONNX_CORPUS", corpus)
        .env("TYNX_CONFORMANCE_REPORT", report)
        .env("TYNX_CONFORMANCE_REGISTRY", root.join(REGISTRY));
    if let Some(case) = case {
        process.env("TYNX_CONFORMANCE_CASE", case);
    }
    if bless {
        process.env("TYNX_CONFORMANCE_BLESS", "1");
    }

    let status = process
        .status()
        .map_err(|error| format!("failed to run conformance test: {error}"))?;
    if !status.success() {
        return Err(format!("conformance test exited with {status}"));
    }
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask must be inside the workspace")
        .to_path_buf()
}

fn load_registry(root: &Path) -> Result<Registry, String> {
    let path = root.join(REGISTRY);
    let data = fs::read_to_string(&path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_str(&data)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn fetch(root: &Path, source: &Source) -> Result<PathBuf, String> {
    let checkout = root.join(".cache/onnx-corpus").join(&source.revision);
    if checkout.join(&source.path).is_dir() {
        println!("using ONNX corpus at {}", checkout.display());
        return Ok(checkout);
    }
    if checkout.exists() {
        return Err(format!(
            "incomplete corpus cache at {}; remove it and retry",
            checkout.display()
        ));
    }

    let parent = checkout
        .parent()
        .ok_or_else(|| "corpus cache has no parent".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;

    println!(
        "fetching {} at {} into {}",
        source.repository,
        source.revision,
        checkout.display()
    );
    command(
        root,
        "git",
        &[
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "--depth=1",
            &source.repository,
            path_str(&checkout)?,
        ],
    )?;
    command(
        &checkout,
        "git",
        &["fetch", "--depth=1", "origin", &source.revision],
    )?;
    command(&checkout, "git", &["sparse-checkout", "set", &source.path])?;
    command(&checkout, "git", &["checkout", "--detach", "FETCH_HEAD"])?;
    Ok(checkout)
}

fn command(cwd: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .status()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} exited with {status}"))
    }
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}

fn usage() -> String {
    "usage: cargo xtask conformance [fetch | bless | --case CASE]".to_string()
}
