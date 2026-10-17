//! Rust (candle) inference for the Imajev image decision model.
//!
//! Imajev reads a photo and the record that goes with it and answers in the options you set, with a
//! probability on each and an explicit "can't tell". This crate runs the Qwen3.5-4B vision model, the Imajev
//! LoRA adapter and its trained choice readout.
//!
//! ```no_run
//! use imajev_rs::{answers_to_json, Engine, State};
//! use candle_core::Device;
//!
//! // Downloads the adapter and base model into ~/.cache/imajev-rs on first use (about 10 GB);
//! // or `Engine::load("path/to/imajev-4b", "path/to/Qwen3.5-4B", &device)`.
//! let mut engine = Engine::from_pretrained(&Device::new_cuda(0)?)?;
//! let questions = serde_json::json!({"page_type": {"type": "choice",
//!     "instructions": "Choose the best page role.",
//!     "criteria": {"exercise": "A question or task for a student.", "theory": "Explanations or reference material."}}});
//! let answers = engine.decide(&State::from("Practice questions"), &questions, Some(&std::fs::read("page.jpg")?))?;
//! println!("{}", answers_to_json(&answers));
//! # anyhow::Ok(())
//! ```

#[cfg(not(target_arch = "wasm32"))]
pub mod download;
pub mod engine;
pub mod model;
#[cfg(feature = "server")]
pub mod server;
pub mod vision;

pub use engine::{
    answers_to_json, parse_questions, python_json, validate_request, Answer, Answers, Engine, InvalidRequest, Question, QuestionKind, State, MAX_IMAGE_BYTES,
    MAX_INPUT_TOKENS, MAX_QUESTIONS, ROTATIONS, UNKNOWN,
};
