# imajev-rs

A Rust port of the [Imajev](https://huggingface.co/mohit67890/imajev-4b) image decision model, built on
[candle](https://github.com/huggingface/candle). Imajev reads a photo and the record that goes with it and answers in
the options you set, with a probability on each answer and an explicit *can't tell*, so your system acts when it is
sure and hands the rest to a person.

It runs [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (revision `851bf6e`) with the Imajev LoRA adapter
and its trained 256-code decision readout. Each question is shown in four option orders, the log probabilities are
averaged and calibrated (temperature 1.305), as in the reference runtime.

- **Text-only works too:** leave the image out and the question is answered from the `state` alone, through the same prompt
  without the vision tokens.
- **Question types:** `choice` (pick one option), `noul` (yes/no, answered as a probability) and `score` (pick a level
  from an ordered list), with `state` as text or JSON next to the image.
- **Use it as** a CLI (`imajev classify`, `imajev predict`), an HTTP server (`imajev serve`) or a Rust library.
- **The model is built for CUDA** (BF16, fused Gated DeltaNet kernels): about 0.3–0.6 s per image on an RTX 4090. The CPU
  path gives the same answers in F32 but is a slow reference and needs about 17 GB of RAM.

## Install

```bash
brew tap apiplant/tap && brew install apiplant/tap/imajev-rs      # macOS, Linux
sudo apt install imajev-rs      # Debian/Ubuntu, after adding apt.apiplant.com
sudo pacman -S imajev-rs        # Arch, after adding apiplant.github.io/pacman
cargo install imajev-rs         # from source (add --features cuda for GPU)
```

CUDA builds (Linux x86_64, NVIDIA GPU with compute capability 8.0 or newer) are separate packages, `imajev-rs-cuda`, that conflict with `imajev-rs`. Setup
commands for the apt and pacman repositories, plain archives and the release process are in
[`packaging/README.md`](packaging/README.md).

The first run downloads the model files (about 10 GB: the 490 MB adapter and the Qwen3.5-4B base) into
`~/.cache/imajev-rs/` (`$XDG_CACHE_HOME/imajev-rs`). `imajev download` does that ahead of time and prints the paths;
`IMAJEV_OFFLINE=1` forbids downloading. Point at your own copies with `--adapter DIR` / `--base DIR`
(`IMAJEV_ADAPTER_DIR`, `IMAJEV_BASE_DIR`).

## Models

Models live in `~/.cache/imajev-rs/` (`$XDG_CACHE_HOME/imajev-rs`), one directory per Hugging Face repo: the adapter `imajev-4b` and its base `Qwen3.5-4B` (about 10 GB together). `imajev download` fetches both ahead of time and prints the paths.

If you already have the files (from `git lfs`, `hf download` or another machine), skip the download by putting them
where it looks, or symlink them there:

```sh
mkdir -p ~/.cache/imajev-rs
ln -s /path/to/imajev-4b ~/.cache/imajev-rs/imajev-4b
```

The cache holds:

```
~/.cache/imajev-rs/
├── imajev-4b/      adapter_model.safetensors, decision_readout.safetensors,
│                   decision_readout.json, adapter_config.json
└── Qwen3.5-4B/     config.json, tokenizer.json, model.safetensors.index.json + the weight shards
```

Link each directory under its repo name, e.g. `ln -s /path/to/Qwen3.5-4B ~/.cache/imajev-rs/Qwen3.5-4B`. `--adapter DIR` / `--base DIR` (`IMAJEV_ADAPTER_DIR`, `IMAJEV_BASE_DIR`) point at copies anywhere else, and `IMAJEV_OFFLINE=1` makes a missing file an error instead of a download.

## Classify

```bash
imajev classify page.jpg -l exercise,theory
imajev classify listing.jpg -q 'Which field of `listing` does this photo contradict?' \
  -l 'listing.color,listing.product_type,none of these:the photo agrees with every field' \
  --state '{"listing": {"color": "red", "product_type": "shoe"}}'
imajev classify scans/*.png -l 'invoice:A bill to pay,receipt:Proof of payment' --format json
```

```
listing.jpg
  choice: listing.color  (confidence 0.86, can't-tell 0.01)
    listing.color         0.914  ██████████████████████
    listing.product_type  0.006
    none of these         0.080  ██
```

`-l` takes `option` or `option:description`, comma-separated (use `;` when descriptions contain commas). `--state` is
text or a JSON object. With no image, `imajev classify -l billing,shipping -q 'Which team should handle this request?' -s 'My card was charged twice'` answers from the text alone. `--format json` prints one object per image.

## Predict (JSONL)

`imajev predict` reads one request per line and writes one answer object per line, loading the model once:

```bash
echo '{"image": "listing.jpg", "state": {"listing": {"color": "red", "product_type": "shoe"}}, "questions": {
  "color_matches": {"type": "noul", "instructions": "The product in the photo matches `listing.color`."},
  "formality": {"type": "score", "instructions": "How formal are these shoes?", "criteria": ["casual", "smart casual", "formal"]}}}' \
  | imajev predict
```

Questions are `{"type": "choice" | "noul" | "score", "instructions", "criteria"}`. Instructions may name state keys in
backticks. `choice` criteria are `{option: description | null}`, `noul` criteria are an optional
`{"true": .., "false": ..}`, `score` criteria are an ordered list of 2–10 level descriptions. A request has up to 8
questions and an optional image (JPEG or PNG, up to 20 MiB).

## Serve over HTTP

```bash
imajev serve                                # 127.0.0.1:8765, CUDA when built with it
imajev serve --device cuda:1 --host 0.0.0.0 --port 9000 --cors '*'
```

```bash
curl -s localhost:8765/v1/systemone \
  -F 'request={"state": {"page_text": "Practice questions"}, "questions": {"page_type": {"type": "choice",
      "instructions": "Choose the best page role.",
      "criteria": {"exercise": "A question or task for a student.", "theory": "Explanations or reference material."}}}}' \
  -F 'image=@page.jpg;type=image/jpeg'
# {"model":"imajev-4b","answers":{"page_type":{"type":"choice","choice":"exercise","probabilities":{...},
#   "confidence":0.93,"unknown_probability":0.01,"abstained":false}},"usage":{"total_ms":412.5}}
```

| Endpoint | Description |
| --- | --- |
| `POST /v1/systemone` | multipart: `request` (JSON `{"state", "questions", "model"?}`) and an optional `image` |
| `POST /v1/classifier` | Exact alias of `/v1/systemone` |
| `GET /health` | `{"status":"ready","model":"imajev-4b"}`, no inference |

Answers by type: `choice` has `choice`, `probabilities` (without the can't-tell mass) and `confidence`; `noul` has the
probability that the statement is true; `score` has the expected level, its `legend` and `probabilities`. Every answer
reports `unknown_probability` and `abstained` (the model's top pick was *can't tell*; `choice` is then `__unknown__`).

Requests run one at a time behind a bounded queue (`--max-queued`, default 16): when it is full the server answers `429`
with `Retry-After: 1`. Bad input is `422` with `{"error": {"message", "type": "invalid_request_error", "code": 422,
"param"}}`; a failed forward is `500 {"detail": "internal error"}`. Ctrl-C or SIGTERM stops gracefully: the listener
closes and an in-flight request finishes first. `--cors ORIGIN` lets a browser page call the server (including Chrome's
private-network preflight), which is how the [website](https://imajev-rs.apiplant.com)'s try-it page talks to a local
server.

## Use as a library

```toml
[dependencies]
imajev-rs = "0.1"                                       # CPU
# imajev-rs = { version = "0.1", features = ["cuda"] }  # + CUDA (needs nvcc to build)
```

```rust
use candle_core::Device;
use imajev_rs::{answers_to_json, Engine, State};

// Downloads the adapter and Qwen3.5-4B into ~/.cache/imajev-rs on first use (about 10 GB);
// or `Engine::load("path/to/imajev-4b", "path/to/Qwen3.5-4B", &device)` for local copies.
let mut engine = Engine::from_pretrained(&Device::new_cuda(0)?)?;
let questions = serde_json::json!({"page_type": {"type": "choice",
    "instructions": "Choose the best page role.",
    "criteria": {"exercise": "A question or task for a student.", "theory": "Explanations or reference material."}}});
let answers = engine.decide(&State::from("Practice questions"), &questions, Some(&std::fs::read("page.jpg")?))?;
println!("{:?} {:.2}", answers[0].1.choice, answers[0].1.confidence);
println!("{}", answers_to_json(&answers));
```

`cli` (the `imajev` binary: clap) and `server` (`imajev_rs::server` and `imajev serve`: axum, tokio) are on by default so
that `cargo install imajev-rs` gives the full binary. As a library you rarely need them:

```toml
imajev-rs = { version = "0.1", default-features = false }                      # engine only
# imajev-rs = { version = "0.1", default-features = false, features = ["server"] }   # + the HTTP server module
```

`imajev_rs::server::start_server` serves any `Decider`, so an application can embed the API next to its own routes
(`routes(state)` is a plain axum `Router`).

## Build

```bash
cargo build --release                    # CPU
cargo build --release --features cuda    # + CUDA
cargo test                               # no weights needed
```

`.cargo/config.toml` pins `CUDARC_CUDA_VERSION=13030` (CUDA 13.3). If your toolkit differs, set that variable yourself
(major×1000 + minor×10, e.g. `12060`); an environment value wins over the file.

## Layout

| Path | Contents |
| --- | --- |
| `src/engine.rs` | question parsing, prompts, rotations, calibration, `Engine` |
| `src/model/` | Qwen3.5 text stack (full attention + Gated DeltaNet, LoRA) and the CUDA DeltaNet kernels |
| `src/vision/` | Qwen3.5 vision tower and image preprocessing |
| `src/server.rs` | the HTTP API |
| `src/download.rs` | model download into the cache |
| `website/` | the [imajev-rs.apiplant.com](https://imajev-rs.apiplant.com) site (Solid + Vite, deployed with Wrangler) |
| `packaging/` | Homebrew, apt, pacman and CUDA package definitions |

## Credits

Imajev is by Mohit Garg ([mohit67890/imajev-4b](https://huggingface.co/mohit67890/imajev-4b), Apache-2.0); the base
model is [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0). This repository is an independent Rust
runtime for those weights.
