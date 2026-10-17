//! Downloads the Imajev adapter and its Qwen3.5-4B base model from Hugging Face into a cache directory.
//!
//! Cache location: `$XDG_CACHE_HOME/imajev-rs/{imajev-4b,Qwen3.5-4B}`, falling back to `~/.cache/imajev-rs/...`
//! when `$XDG_CACHE_HOME` is unset. Files are fetched to a `.part` sibling and renamed into place once
//! complete, so a killed download never leaves a checkpoint that looks done. Set `IMAJEV_OFFLINE=1` to
//! forbid network access.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// The adapter repository (LoRA, trained readout, calibration).
pub const ADAPTER_REPO: &str = "mohit67890/imajev-4b";
/// The base model the adapter was trained on, pinned to the revision the adapter records.
pub const BASE_REPO: &str = "Qwen/Qwen3.5-4B";
pub const BASE_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

/// Files read from the adapter directory.
pub const ADAPTER_FILES: &[&str] = &["adapter_model.safetensors", "decision_readout.safetensors", "decision_readout.json", "adapter_config.json"];
/// Files read from the base model directory, besides the shards its index names.
pub const BASE_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors.index.json"];

/// `$XDG_CACHE_HOME/imajev-rs`, defaulting to `~/.cache/imajev-rs`.
pub fn cache_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("imajev-rs"))
}

/// Where [`download_adapter`] puts (or would put) the adapter.
pub fn adapter_cache_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(ADAPTER_REPO.rsplit('/').next().unwrap_or(ADAPTER_REPO)))
}

/// Where [`download_base`] puts (or would put) the base model.
pub fn base_cache_dir() -> Option<PathBuf> {
    Some(cache_dir()?.join(BASE_REPO.rsplit('/').next().unwrap_or(BASE_REPO)))
}

fn fetch_to_file(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = dest.with_extension(match dest.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });

    let resp = ureq::get(url).call().with_context(|| format!("downloading {url}"))?;
    let len: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
    eprint!("  {name} ...");
    std::io::stderr().flush().ok();

    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut written: u64 = 0;
    let mut last_report = 0u64;
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("reading body for {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).with_context(|| format!("writing {}", part.display()))?;
        written += n as u64;
        if written - last_report >= 512 << 20 {
            last_report = written;
            match len {
                Some(len) => eprint!(" {}%", written * 100 / len.max(1)),
                None => eprint!(" {} MiB", written >> 20),
            }
            std::io::stderr().flush().ok();
        }
    }
    drop(file);

    if let Some(len) = len {
        if written != len {
            let _ = std::fs::remove_file(&part);
            bail!("short read for {url}: got {written} bytes, expected {len}");
        }
    }
    std::fs::rename(&part, dest).with_context(|| format!("renaming {} to {}", part.display(), dest.display()))?;
    eprintln!(" done ({written} bytes)");
    Ok(())
}

fn ensure_file(dir: &Path, repo: &str, revision: &str, file: &str) -> Result<()> {
    let dest = dir.join(file);
    if dest.is_file() {
        return Ok(());
    }
    if std::env::var_os("IMAJEV_OFFLINE").is_some() {
        bail!("{file} is missing from {} and IMAJEV_OFFLINE is set; unset it to allow downloading, or pass --adapter/--base", dir.display());
    }
    fetch_to_file(&format!("https://huggingface.co/{repo}/resolve/{revision}/{file}"), &dest)
}

/// Downloads the adapter into the cache (skipping files already present) and returns its directory.
pub fn download_adapter() -> Result<PathBuf> {
    let dir = adapter_cache_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    for file in ADAPTER_FILES {
        if !dir.join(file).is_file() {
            eprintln!("Downloading the Imajev adapter from https://huggingface.co/{ADAPTER_REPO} into {}", dir.display());
        }
        ensure_file(&dir, ADAPTER_REPO, "main", file)?;
    }
    Ok(dir)
}

/// Downloads the Qwen3.5-4B base model (about 9 GB) into the cache and returns its directory.
pub fn download_base() -> Result<PathBuf> {
    let dir = base_cache_dir().context("no cache directory available (set $HOME or $XDG_CACHE_HOME)")?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let mut announced = false;
    let mut announce = |dir: &Path| {
        if !announced {
            eprintln!("Downloading the base model from https://huggingface.co/{BASE_REPO} into {}", dir.display());
            announced = true;
        }
    };
    for file in BASE_FILES {
        if !dir.join(file).is_file() {
            announce(&dir);
        }
        ensure_file(&dir, BASE_REPO, BASE_REVISION, file)?;
    }
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let mut shards: Vec<String> = index["weight_map"]
        .as_object()
        .context("model.safetensors.index.json has no weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    shards.sort();
    shards.dedup();
    for shard in &shards {
        if !dir.join(shard).is_file() {
            announce(&dir);
        }
        ensure_file(&dir, BASE_REPO, BASE_REVISION, shard)?;
    }
    Ok(dir)
}

/// Downloads both and returns `(adapter_dir, base_dir)`.
pub fn download_default() -> Result<(PathBuf, PathBuf)> {
    Ok((download_adapter()?, download_base()?))
}

/// The adapter and base directories to use: explicit paths win; otherwise `IMAJEV_BASE_DIR` /
/// `IMAJEV_MODEL_BUNDLE` (a JSON file whose `path` is the base directory) for the base; otherwise the cache,
/// downloading what is missing.
pub fn resolve(adapter: Option<PathBuf>, base: Option<PathBuf>) -> Result<(PathBuf, PathBuf)> {
    let adapter = match adapter {
        Some(dir) => dir,
        None => download_adapter()?,
    };
    let base = match base {
        Some(dir) => dir,
        None => match std::env::var_os("IMAJEV_MODEL_BUNDLE") {
            Some(bundle) => {
                let bundle = PathBuf::from(bundle);
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&bundle).with_context(|| format!("cannot read base model bundle {}", bundle.display()))?)?;
                PathBuf::from(value.get("path").and_then(|v| v.as_str()).context("model bundle has no path")?)
            }
            None => download_base()?,
        },
    };
    Ok((adapter, base))
}
