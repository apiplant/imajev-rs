//! imajev: classify / predict / serve for the Imajev image decision model.
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use clap::{Args, Parser, Subcommand, ValueEnum};
use imajev_rs::{answers_to_json, Answers, Engine, QuestionKind, State};
use serde_json::{json, Map, Value};

#[derive(Parser)]
#[command(name = "imajev", version, about = "Imajev image decision model (Qwen3.5-4B + LoRA, Candle)", after_help = "Examples:
  imajev classify page.jpg -l exercise,theory
  imajev classify listing.jpg -l \"red,blue,beige\" -q \"What colour is the product?\" --state '{\"listing\":{\"color\":\"red\"}}'
  imajev classify scans/*.png -l \"invoice:A bill to pay,receipt:Proof of payment\" --format json
  imajev predict --input requests.jsonl
  imajev serve --device cuda --port 8765")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// Adapter directory (adapter_model.safetensors, decision_readout.*). Defaults to the imajev-4b adapter in
    /// the imajev-rs cache directory (`$XDG_CACHE_HOME/imajev-rs`, else `~/.cache/imajev-rs`), downloaded there
    /// from Hugging Face first if it is not present.
    #[arg(long, global = true, env = "IMAJEV_ADAPTER_DIR")]
    adapter: Option<PathBuf>,
    /// Qwen3.5-4B base checkpoint directory. Defaults to the cache, downloaded first if needed
    /// (`IMAJEV_MODEL_BUNDLE` may name a JSON file whose `path` is this directory).
    #[arg(long, global = true, env = "IMAJEV_BASE_DIR")]
    base: Option<PathBuf>,
    /// cpu, cuda or cuda:N. The model is built for CUDA; the CPU path is a slow reference.
    #[arg(long, global = true, env = "IMAJEV_DEVICE", default_value = default_device())]
    device: DeviceArg,
}

const fn default_device() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    }
}

#[derive(Clone, Copy)]
struct DeviceArg(Option<usize>);

impl std::str::FromStr for DeviceArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Self(None)),
            "cuda" => Ok(Self(Some(0))),
            other => other
                .strip_prefix("cuda:")
                .and_then(|n| n.parse().ok())
                .map(|n| Self(Some(n)))
                .ok_or_else(|| format!("expected cpu, cuda or cuda:N, got {other:?}")),
        }
    }
}

impl EngineArgs {
    fn load(&self) -> Result<Engine> {
        let device = match self.device.0 {
            None => Device::Cpu,
            Some(index) => {
                if !cfg!(feature = "cuda") {
                    bail!("this imajev binary was built without CUDA; use --device cpu, or rebuild with `--features cuda`");
                }
                Device::new_cuda(index)?
            }
        };
        let (adapter, base) = imajev_rs::download::resolve(self.adapter.clone(), self.base.clone())?;
        let started = Instant::now();
        let engine = Engine::load(&adapter, &base, &device)?;
        eprintln!("loaded in {:.2?}", started.elapsed());
        Ok(engine)
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    /// Human-readable probabilities.
    Text,
    /// One JSON object per image, with the answers.
    Json,
}

#[derive(Subcommand)]
enum Command {
    /// Answer one choice question about each image: `imajev classify photo.jpg -l a,b,c`.
    Classify {
        #[command(flatten)]
        engine: EngineArgs,
        /// Images (JPEG or PNG). With none, the question is answered from `--state` alone.
        images: Vec<PathBuf>,
        /// Options, comma-separated; `option:description` adds a description. Separate with `;` instead when a
        /// description contains commas.
        #[arg(short, long)]
        labels: String,
        /// The question. Defaults to "Choose the best option."
        #[arg(short, long, default_value = "Choose the best option.")]
        question: String,
        /// Name of the question in the JSON output.
        #[arg(short = 'n', long, default_value = "choice")]
        name: String,
        /// Evidence next to the image: text, or a JSON object.
        #[arg(short, long)]
        state: Option<String>,
        #[arg(long, value_enum, default_value_t = Format::Text)]
        format: Format,
    },
    /// Read JSONL requests `{"state", "questions", "image": "path.jpg"}` (`image` is optional) and write JSONL answers.
    Predict {
        #[command(flatten)]
        engine: EngineArgs,
        /// Input JSONL (default: stdin).
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Serve the named-question API over HTTP (`POST /v1/systemone`, `GET /health`).
    /// Loads the model, binds, then runs until Ctrl-C/SIGTERM (an in-flight forward finishes first).
    #[cfg(feature = "server")]
    Serve {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, env = "IMAJEV_HOST", default_value = "127.0.0.1")]
        host: String,
        #[arg(long, env = "IMAJEV_PORT", default_value_t = 8765)]
        port: u16,
        /// Model identity reported by /health; a request's optional `model` field must match it.
        #[arg(long, default_value = "imajev-4b")]
        model_name: String,
        /// Waiting slots on top of the one in-flight request; beyond that requests get 429.
        #[arg(long, env = "IMAJEV_MAX_QUEUED", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
        max_queued: u32,
        /// Allow browser pages from this origin (`*` for any) to call the server.
        #[arg(long)]
        cors: Option<String>,
    },
    /// Download the adapter and base model into the cache (about 10 GB) and print where they are.
    Download,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Classify { engine, images, labels, question, name, state, format } => classify(&engine, &images, &labels, &question, &name, state, format),
        Command::Predict { engine, input } => predict(&engine, input),
        #[cfg(feature = "server")]
        Command::Serve { engine, host, port, model_name, max_queued, cors } => {
            let config = imajev_rs::server::ServerConfig { model_name, max_queued: max_queued as usize, cors_origin: cors };
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(serve(engine, host, port, config))
        }
        Command::Download => {
            let (adapter, base) = imajev_rs::download::download_default()?;
            println!("adapter: {}\nbase:    {}", adapter.display(), base.display());
            Ok(())
        }
    }
}

#[cfg(feature = "server")]
async fn serve(args: EngineArgs, host: String, port: u16, config: imajev_rs::server::ServerConfig) -> Result<()> {
    let engine = args.load()?;
    let handle = imajev_rs::server::start_server(&host, port, engine, config).await?;
    eprintln!("imajev serving '{}' on http://{} (/v1/systemone, /health)", handle.model_name, handle.addr);
    // Ctrl-C/SIGTERM: stop accepting, finish the in-flight request, exit.
    handle.wait().await;
    eprintln!("imajev stopped");
    Ok(())
}

/// `a,b:desc` (or `a;b:desc`) into a criteria object.
fn parse_labels(labels: &str) -> Result<Map<String, Value>> {
    let separator = if labels.contains(';') { ';' } else { ',' };
    let mut criteria = Map::new();
    for item in labels.split(separator).map(str::trim).filter(|s| !s.is_empty()) {
        let (name, description) = match item.split_once(':') {
            Some((name, description)) => (name.trim(), Some(description.trim())),
            None => (item, None),
        };
        criteria.insert(name.to_string(), description.filter(|d| !d.is_empty()).map_or(Value::Null, Value::from));
    }
    if criteria.len() < 2 {
        bail!("--labels needs at least two options");
    }
    Ok(criteria)
}

fn parse_state(state: Option<String>) -> Result<State> {
    match state {
        None => Ok(State(json!({}))),
        Some(text) => {
            let trimmed = text.trim_start();
            if trimmed.starts_with('{') || trimmed.starts_with('[') {
                let value: Value = serde_json::from_str(&text).context("--state looks like JSON but does not parse")?;
                State::from_value(&value).context("state must be text or a JSON object/array")
            } else {
                Ok(State::from(text))
            }
        }
    }
}

fn classify(args: &EngineArgs, images: &[PathBuf], labels: &str, question: &str, name: &str, state: Option<String>, format: Format) -> Result<()> {
    let questions = json!({ name: {"type": "choice", "instructions": question, "criteria": parse_labels(labels)?} });
    let state = parse_state(state)?;
    if images.is_empty() && state.0 == json!({}) {
        bail!("give at least one image, or text evidence with --state");
    }
    let mut engine = args.load()?;
    let mut out = std::io::stdout().lock();
    if images.is_empty() {
        let started = Instant::now();
        let answers = engine.decide(&state, &questions, None)?;
        eprintln!("text only: {:.2?}", started.elapsed());
        match format {
            Format::Json => writeln!(out, "{}", json!({"answers": answers_to_json(&answers)}))?,
            Format::Text => write_text(&mut out, "(text only)", &answers)?,
        }
        return Ok(());
    }
    for path in images {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let started = Instant::now();
        let answers = engine.decide(&state, &questions, Some(&bytes)).with_context(|| path.display().to_string())?;
        eprintln!("{}: {:.2?}", path.display(), started.elapsed());
        match format {
            Format::Json => writeln!(out, "{}", json!({"image": path.display().to_string(), "answers": answers_to_json(&answers)}))?,
            Format::Text => write_text(&mut out, &path.display().to_string(), &answers)?,
        }
    }
    Ok(())
}

fn write_text(out: &mut impl Write, title: &str, answers: &Answers) -> Result<()> {
    writeln!(out, "{title}")?;
    for (id, answer) in answers {
        let verdict = match answer.kind {
            QuestionKind::Choice => answer.choice.as_deref().map_or("can't tell".to_string(), str::to_string),
            QuestionKind::Noul => format!("{:.3} yes", answer.value.unwrap_or(0.0)),
            QuestionKind::Score => format!("score {:.2}", answer.value.unwrap_or(0.0)),
        };
        writeln!(out, "  {id}: {verdict}  (confidence {:.2}, can't-tell {:.2})", answer.confidence, answer.unknown_probability)?;
        let width = answer.probabilities.iter().map(|(name, _)| name.chars().count()).max().unwrap_or(0);
        for (name, p) in &answer.probabilities {
            writeln!(out, "    {name:<width$}  {p:>5.3}  {}", "█".repeat((p * 24.0).round() as usize))?;
        }
    }
    Ok(())
}

fn predict(args: &EngineArgs, input: Option<PathBuf>) -> Result<()> {
    let text = match input {
        Some(path) => std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?,
        None => std::io::read_to_string(std::io::stdin())?,
    };
    let mut requests = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let value: Value = serde_json::from_str(line).with_context(|| format!("line {}: invalid JSON", i + 1))?;
        let image = match value.get("image") {
            None | Some(Value::Null) => None,
            Some(v) => Some(PathBuf::from(v.as_str().with_context(|| format!("line {}: \"image\" must be a file path", i + 1))?)),
        };
        let questions = value.get("questions").with_context(|| format!("line {}: \"questions\" is required", i + 1))?.clone();
        let state = match value.get("state") {
            Some(v) => State::from_value(v).with_context(|| format!("line {}: state must be text or a JSON object/array", i + 1))?,
            None => State(json!({})),
        };
        requests.push((i + 1, state, questions, image));
    }
    let mut engine = args.load()?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (line, state, questions, image) in requests {
        let bytes = image.as_ref().map(|path| std::fs::read(path).with_context(|| format!("line {line}: reading {}", path.display()))).transpose()?;
        let answers = engine.decide(&state, &questions, bytes.as_deref()).with_context(|| format!("line {line}"))?;
        writeln!(out, "{}", json!({"answers": answers_to_json(&answers)}))?;
        out.flush()?;
    }
    Ok(())
}
