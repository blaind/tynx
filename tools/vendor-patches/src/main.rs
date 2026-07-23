use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Output, Stdio};

struct Repository {
    name: &'static str,
    url: &'static str,
    revision: &'static str,
    patch: &'static str,
    changed_paths: &'static [&'static str],
}

const REPOSITORIES: &[Repository] = &[
    Repository {
        name: "burn",
        url: "https://github.com/tracel-ai/burn",
        revision: "78f10aec1ca6c6ffb1edd17a0fa131ae59ad5403",
        patch: "vendor/burn.patch",
        changed_paths: &[
            "crates/burn-cubecl-fusion/Cargo.toml",
            "crates/burn-cubecl-fusion/src/optim/reduce/fuser.rs",
            "crates/burn-fusion/Cargo.toml",
            "crates/burn-fusion/src/stream/multi.rs",
        ],
    },
    Repository {
        name: "cubecl",
        url: "https://github.com/tracel-ai/cubecl",
        revision: "ba103c7f118524652647338dec09abf69a24a53e",
        patch: "vendor/cubecl.patch",
        changed_paths: &[
            "crates/cubecl-wgpu/Cargo.toml",
            "crates/cubecl-wgpu/src/compiler/wgsl/base.rs",
        ],
    },
    Repository {
        name: "cubek",
        url: "https://github.com/tracel-ai/cubek",
        revision: "e6a634f1bd0c3a72325228d308462ec0c5c96456",
        patch: "vendor/cubek.patch",
        changed_paths: &[
            "crates/cubek-reduce/Cargo.toml",
            "crates/cubek-reduce/src/components/instructions/argmax.rs",
            "crates/cubek-reduce/src/components/instructions/argmin.rs",
        ],
    },
];

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
    if env::args_os().len() != 1 {
        return Err("usage: cargo vendor-patches".to_string());
    }

    let root = workspace_root()?;
    let repos = root.join("vendor/repos");
    fs::create_dir_all(&repos)
        .map_err(|error| format!("failed to create {}: {error}", repos.display()))?;

    for repository in REPOSITORIES {
        prepare_repository(&root, &repos, repository)?;
    }

    println!("patched dependency clones are ready in {}", repos.display());
    Ok(())
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "bootstrap tool must be located under tools/vendor-patches".to_string())
}

fn prepare_repository(root: &Path, repos: &Path, repository: &Repository) -> Result<(), String> {
    let checkout = repos.join(repository.name);
    let patch = root.join(repository.patch);
    if !patch.is_file() {
        return Err(format!("missing patch {}", patch.display()));
    }

    if !checkout.exists() {
        clone_revision(repos, &checkout, repository)?;
    } else if !checkout.join(".git").is_dir() {
        return Err(format!(
            "{} exists but is not a Git checkout; move it aside and retry",
            checkout.display()
        ));
    }

    let revision = output(&checkout, "git", &["rev-parse", "HEAD"])?;
    if revision.trim() != repository.revision {
        return Err(format!(
            "{} is at {}, expected {}; move it aside and rerun",
            checkout.display(),
            revision.trim(),
            repository.revision
        ));
    }

    let status = output(
        &checkout,
        "git",
        &["status", "--porcelain", "--untracked-files=all"],
    )?;
    if status.trim().is_empty() {
        run_command(&checkout, "git", &["apply", "--check", path_str(&patch)?])?;
        run_command(&checkout, "git", &["apply", path_str(&patch)?])?;
        println!(
            "applied {} to {} at {}",
            repository.patch, repository.name, repository.revision
        );
    } else {
        verify_expected_changes(&checkout, repository, &status)?;
        if !command_succeeds(
            &checkout,
            "git",
            &["apply", "--reverse", "--check", path_str(&patch)?],
        )? {
            return Err(format!(
                "{} has changes that do not match {}; move it aside and rerun",
                checkout.display(),
                repository.patch
            ));
        }
        println!(
            "{} is already patched at {}",
            repository.name, repository.revision
        );
    }

    let final_status = output(
        &checkout,
        "git",
        &["status", "--porcelain", "--untracked-files=all"],
    )?;
    verify_expected_changes(&checkout, repository, &final_status)
}

fn clone_revision(repos: &Path, checkout: &Path, repository: &Repository) -> Result<(), String> {
    let temporary = repos.join(format!(".{}.tmp-{}", repository.name, std::process::id()));
    if temporary.exists() {
        return Err(format!(
            "temporary checkout {} already exists; remove it and retry",
            temporary.display()
        ));
    }

    fs::create_dir(&temporary)
        .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
    let result = (|| {
        run_command(&temporary, "git", &["init", "--quiet"])?;
        run_command(&temporary, "git", &["config", "core.autocrlf", "false"])?;
        run_command(
            &temporary,
            "git",
            &["remote", "add", "origin", repository.url],
        )?;
        run_command(
            &temporary,
            "git",
            &[
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                repository.revision,
            ],
        )?;
        run_command(
            &temporary,
            "git",
            &["checkout", "--quiet", "--detach", "FETCH_HEAD"],
        )
    })();

    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temporary);
        return Err(error);
    }

    fs::rename(&temporary, checkout).map_err(|error| {
        format!(
            "failed to move {} to {}: {error}",
            temporary.display(),
            checkout.display()
        )
    })?;
    Ok(())
}

fn verify_expected_changes(
    checkout: &Path,
    repository: &Repository,
    status: &str,
) -> Result<(), String> {
    let expected: BTreeSet<&str> = repository.changed_paths.iter().copied().collect();
    let mut actual = BTreeSet::new();

    for line in status.lines() {
        if line.len() < 4 {
            return Err(format!(
                "unexpected Git status in {}: {line}",
                checkout.display()
            ));
        }
        let path = &line[3..];
        if !line[..2].contains('M') || path.contains(" -> ") {
            return Err(format!(
                "unexpected Git status in {}: {line}",
                checkout.display()
            ));
        }
        actual.insert(path);
    }

    if actual != expected {
        return Err(format!(
            "{} has unexpected changes: expected {expected:?}, observed {actual:?}",
            checkout.display()
        ));
    }
    Ok(())
}

fn run_command(cwd: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .status()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} {} exited with {status}", args.join(" ")))
    }
}

fn command_succeeds(cwd: &Path, program: &str, args: &[&str]) -> Result<bool, String> {
    let status = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    Ok(status.success())
}

fn output(cwd: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let Output {
        status,
        stdout,
        stderr,
    } = Command::new(program)
        .current_dir(cwd)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !status.success() {
        return Err(format!(
            "{program} {} exited with {status}: {}",
            args.join(" "),
            String::from_utf8_lossy(&stderr).trim()
        ));
    }
    String::from_utf8(stdout).map_err(|error| format!("{program} output was not UTF-8: {error}"))
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {}", path.display()))
}
