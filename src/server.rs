//! HTTP server for the named-question API (native only), built on axum.
//!
//! * `POST /v1/systemone` (alias `POST /v1/classifier`): multipart form data with
//!   `request` (JSON `{"state": text | object, "questions": {id: {"type": "choice", "instructions", "criteria"}}}`)
//!   and optionally one image (`image`, JPEG or PNG, up to 20 MiB); without an image the answer comes from the state alone. Answers come back as
//!   `{"model", "answers": {id: {type, choice, probabilities, confidence, unknown_probability, abstained}}, "usage"}`.
//! * `GET /health`: `{"status":"ready","model":..}`, no inference.
//!
//! Requests run strictly one at a time (one forward in flight) behind a bounded admission queue: a request that
//! finds `1 + max_queued` requests already admitted gets `429` with `Retry-After: 1`. Bad input is `422` with
//! `{"error": {message, type: "invalid_request_error", code: 422, param}}`; a failed forward is
//! `500 {"detail":"internal error"}` (the cause goes to stderr).
//!
//! [`start_server`] must run inside a tokio runtime. Ctrl-C and SIGTERM stop the listener, let an in-flight
//! forward finish, and resolve [`ServerHandle::wait`].

use crate::engine::{Answers, Engine, InvalidRequest, State, MAX_IMAGE_BYTES};
use anyhow::Result;
use axum::extract::{DefaultBodyLimit, Multipart, State as AxumState};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{oneshot, Mutex, Semaphore};

/// Headroom over the image limit for the JSON part and multipart framing.
const BODY_SLACK: usize = 1024 * 1024;

/// What the server needs from a model; [`Engine`] implements it, tests use a stand-in.
pub trait Decider: Send + 'static {
    /// Most options a choice question may have; requests are validated against it before queueing.
    fn max_options(&self) -> usize;
    /// One request (synchronous; runs on the blocking pool).
    fn decide(&mut self, state: &State, questions: &Value, image: Option<&[u8]>) -> Result<Answers>;
}

impl Decider for Engine {
    fn max_options(&self) -> usize {
        Engine::max_options(self)
    }

    fn decide(&mut self, state: &State, questions: &Value, image: Option<&[u8]>) -> Result<Answers> {
        Engine::decide(self, state, questions, image)
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Model identity reported by `/health` and in answers; a request's optional `model` field must match it.
    pub model_name: String,
    /// Waiting slots on top of the one in-flight request.
    pub max_queued: usize,
    /// Allowed browser origin (`*` for any). When set, responses carry CORS headers and `OPTIONS` preflights
    /// (including Chrome's private-network preflight) are answered, so a web page can call a local server.
    pub cors_origin: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { model_name: "imajev-4b".into(), max_queued: 16, cors_origin: None }
    }
}

#[derive(Debug)]
pub enum ExecError {
    /// 422: the request is not answerable (`param` names the offending field).
    Invalid { param: &'static str, message: String },
    /// 429: the admission queue is full.
    QueueFull,
    /// 500: the forward pass failed.
    Internal(String),
}

struct Inner {
    decider: Arc<Mutex<Box<dyn Decider>>>,
    max_options: usize,
    config: ServerConfig,
    admission: Arc<Semaphore>,
}

/// Shared state behind the routes (cheap to clone).
#[derive(Clone)]
pub struct ServerState(Arc<Inner>);

impl ServerState {
    pub fn new(decider: impl Decider, config: ServerConfig) -> Self {
        let admission = Arc::new(Semaphore::new(config.max_queued.saturating_add(1)));
        let max_options = decider.max_options();
        Self(Arc::new(Inner { decider: Arc::new(Mutex::new(Box::new(decider))), max_options, config, admission }))
    }

    pub fn model_name(&self) -> &str {
        &self.0.config.model_name
    }

    /// Admission slots currently free (0 means the next request is rejected with 429).
    pub fn admission_available_slots(&self) -> usize {
        self.0.admission.available_permits()
    }

    /// Validate, queue, and answer one request.
    pub async fn execute(&self, state: State, questions: Value, image: Option<Vec<u8>>) -> Result<Answers, ExecError> {
        // Validation never waits for the model lock: a bad request is answered even while a forward runs.
        crate::engine::validate_request(&questions, image.as_deref(), self.0.max_options).map_err(|e| ExecError::Invalid { param: "questions", message: e.to_string() })?;
        let permit = self.0.admission.clone().try_acquire_owned().map_err(|_| ExecError::QueueFull)?;
        let decider = self.0.decider.clone();
        // The permit and lock move into the blocking closure, so a client disconnect (dropping this future)
        // can never let a second forward overlap one that is still running.
        let lock = decider.lock_owned().await;
        tokio::task::spawn_blocking(move || {
            let mut lock = lock;
            let result = lock.decide(&state, &questions, image.as_deref());
            drop(lock);
            drop(permit);
            result
        })
        .await
        .map_err(|e| ExecError::Internal(format!("decision task failed: {e}")))?
        .map_err(|e| match e.downcast::<InvalidRequest>() {
            Ok(invalid) => ExecError::Invalid { param: "request", message: invalid.0 },
            Err(other) => ExecError::Internal(format!("{other:#}")),
        })
    }
}

fn json_response(status: StatusCode, retry_after: bool, body: Value) -> Response {
    let mut response = (status, Json(body)).into_response();
    if retry_after {
        response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
    }
    response
}

fn invalid(param: &str, message: impl Into<String>) -> Response {
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        false,
        json!({"error": {"message": message.into(), "type": "invalid_request_error", "code": 422, "param": param}}),
    )
}

async fn systemone(AxumState(state): AxumState<ServerState>, mut multipart: Multipart) -> Response {
    let mut request: Option<Value> = None;
    let mut image: Option<Vec<u8>> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => return invalid("body", format!("invalid multipart body: {e}")),
        };
        match field.name().unwrap_or("") {
            "request" => {
                let text = match field.text().await {
                    Ok(t) => t,
                    Err(e) => return invalid("request", e.to_string()),
                };
                match serde_json::from_str::<Value>(&text) {
                    Ok(v) => request = Some(v),
                    Err(e) => return invalid("request", format!("invalid JSON: {e}")),
                }
            }
            "image" | "images" | "image[]" => {
                if image.is_some() {
                    return invalid("image", "send exactly one page image");
                }
                match field.bytes().await {
                    Ok(b) => image = Some(b.to_vec()),
                    Err(e) => return invalid("image", e.to_string()),
                }
            }
            _ => {}
        }
    }
    let Some(request) = request else { return invalid("request", "multipart form must include a 'request' JSON field") };
    let Some(body) = request.as_object() else { return invalid("request", "request must be a JSON object") };
    if let Some(model) = body.get("model") {
        if model.as_str() != Some(state.model_name()) {
            return invalid("model", format!("Loaded model is '{}'", state.model_name()));
        }
    }
    let Some(request_state) = body.get("state").and_then(State::from_value).or_else(|| body.get("state").is_none().then(|| State(json!({})))) else {
        return invalid("state", "state must be text or a JSON object/array");
    };
    let Some(questions) = body.get("questions") else { return invalid("questions", "questions is required") };

    let started = Instant::now();
    match state.execute(request_state, questions.clone(), image).await {
        Ok(answers) => json_response(
            StatusCode::OK,
            false,
            json!({
                "model": state.model_name(),
                "answers": crate::engine::answers_to_json(&answers),
                "usage": {"total_ms": (started.elapsed().as_secs_f64() * 1e4).round() / 10.0},
            }),
        ),
        Err(ExecError::Invalid { param, message }) => invalid(param, message),
        Err(ExecError::QueueFull) => json_response(StatusCode::TOO_MANY_REQUESTS, true, json!({"detail": "Scoring queue is full"})),
        Err(ExecError::Internal(cause)) => {
            eprintln!("decision failed: {cause}");
            json_response(StatusCode::INTERNAL_SERVER_ERROR, false, json!({"detail": "internal error"}))
        }
    }
}

async fn health(AxumState(state): AxumState<ServerState>) -> Response {
    json_response(StatusCode::OK, false, json!({"status": "ready", "model": state.model_name()}))
}

async fn cors(AxumState(state): AxumState<ServerState>, request: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let preflight = request.method() == axum::http::Method::OPTIONS;
    let mut response = if preflight { StatusCode::NO_CONTENT.into_response() } else { next.run(request).await };
    if let Some(origin) = state.0.config.cors_origin.as_deref().and_then(|o| HeaderValue::from_str(o).ok()) {
        let headers: &mut HeaderMap = response.headers_mut();
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
        headers.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
        headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("content-type"));
        headers.insert("access-control-allow-private-network", HeaderValue::from_static("true"));
    }
    response
}

/// The router: `axum::serve(listener, routes(state))`.
pub fn routes(state: ServerState) -> Router {
    let mut router = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/classifier", post(systemone))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(MAX_IMAGE_BYTES + BODY_SLACK));
    if state.0.config.cors_origin.is_some() {
        router = router.layer(axum::middleware::from_fn_with_state(state.clone(), cors));
    }
    router.with_state(state)
}

/// A running server: the bound address, the model identity, and the serving task.
pub struct ServerHandle {
    pub addr: SocketAddr,
    pub model_name: String,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl ServerHandle {
    /// Resolves once the server has stopped (Ctrl-C, SIGTERM or [`Self::stop`]).
    pub async fn wait(self) {
        let _ = self.task.await;
    }

    /// Graceful stop: no new connections, in-flight requests finish.
    pub async fn stop(mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        let _ = self.task.await;
    }
}

async fn shutdown_signal(stop: oneshot::Receiver<()>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
        _ = stop => {}
    }
}

/// Bind `host:port` (port 0 = ephemeral) and serve. The model must already be loaded.
pub async fn start_server(host: &str, port: u16, decider: impl Decider, config: ServerConfig) -> Result<ServerHandle> {
    let model_name = config.model_name.clone();
    let state = ServerState::new(decider, config);
    let listener = tokio::net::TcpListener::bind((host, port)).await.map_err(|e| anyhow::anyhow!("failed to bind {host}:{port}: {e}"))?;
    let addr = listener.local_addr()?;
    let (tx, rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, routes(state)).with_graceful_shutdown(shutdown_signal(rx)).await {
            eprintln!("server error: {e}");
        }
    });
    Ok(ServerHandle { addr, model_name, stop: Some(tx), task })
}
