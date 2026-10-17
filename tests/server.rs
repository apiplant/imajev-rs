//! HTTP contract tests against a stand-in model (no weights needed).
use anyhow::Result;
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use http_body_util_shim::to_bytes;
use imajev_rs::server::{routes, Decider, ServerConfig, ServerState};
use imajev_rs::{parse_questions, Answer, Answers, State};
use serde_json::{json, Value};
use tower::ServiceExt;

mod http_body_util_shim {
    pub async fn to_bytes(body: axum::body::Body) -> Vec<u8> {
        axum::body::to_bytes(body, usize::MAX).await.unwrap().to_vec()
    }
}

struct Fake {
    fail: bool,
}

impl Decider for Fake {
    fn max_options(&self) -> usize {
        255
    }

    fn decide(&mut self, _state: &State, questions: &Value, _image: Option<&[u8]>) -> Result<Answers> {
        if self.fail {
            anyhow::bail!("boom");
        }
        Ok(parse_questions(questions, 255)?
            .into_iter()
            .map(|q| {
                let p = 1.0 / q.keys.len() as f32;
                let answer = Answer {
                    kind: q.kind,
                    choice: Some(q.keys[0].clone()),
                    probabilities: q.keys.iter().map(|n| (n.clone(), p)).collect(),
                    value: None,
                    confidence: 0.5,
                    unknown_probability: 0.1,
                    abstained: false,
                    legend: Vec::new(),
                };
                (q.id, answer)
            })
            .collect())
    }
}

const BOUNDARY: &str = "XBOUNDARYX";

fn multipart(request: Option<&str>, image: Option<&[u8]>) -> Vec<u8> {
    let mut body = Vec::new();
    if let Some(request) = request {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n{request}\r\n").into_bytes());
    }
    if let Some(image) = image {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").into_bytes());
        body.extend(image);
        body.extend(b"\r\n");
    }
    body.extend(format!("--{BOUNDARY}--\r\n").into_bytes());
    body
}

async fn post(state: ServerState, body: Vec<u8>) -> (StatusCode, Value) {
    let request = Request::post("/v1/systemone")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(Body::from(body))
        .unwrap();
    let response = routes(state).oneshot(request).await.unwrap();
    let status = response.status();
    (status, serde_json::from_slice(&to_bytes(response.into_body()).await).unwrap_or(Value::Null))
}

const GOOD: &str = r#"{"state":"Practice","questions":{"page_type":{"type":"choice","instructions":"Role?","criteria":{"exercise":"task","theory":null}}}}"#;

fn state(fail: bool) -> ServerState {
    ServerState::new(Fake { fail }, ServerConfig::default())
}

#[tokio::test]
async fn answers_a_request() {
    let (status, body) = post(state(false), multipart(Some(GOOD), Some(b"img"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "imajev-4b");
    assert_eq!(body["answers"]["page_type"]["choice"], "exercise");
    assert!(body["usage"]["total_ms"].is_number());
}

#[tokio::test]
async fn answers_text_only_requests() {
    let (status, body) = post(state(false), multipart(Some(GOOD), None)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["answers"]["page_type"]["choice"], "exercise");
}

#[tokio::test]
async fn rejects_bad_requests_with_422() {
    for (request, image, param) in [
        (None, Some(&b"img"[..]), "request"),
        (Some("{"), Some(&b"img"[..]), "request"),
        (Some(r#"{"state":"x","questions":{"q":{"type":"multi","instructions":"x"}}}"#), Some(&b"img"[..]), "questions"),
        (Some(r#"{"model":"other","state":"x","questions":{}}"#), Some(&b"img"[..]), "model"),
    ] {
        let (status, body) = post(state(false), multipart(request, image)).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{request:?}");
        assert_eq!(body["error"]["param"], param, "{request:?}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
}

#[tokio::test]
async fn failed_forward_is_500_without_the_cause() {
    let (status, body) = post(state(true), multipart(Some(GOOD), Some(b"img"))).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body, json!({"detail": "internal error"}));
}

#[tokio::test]
async fn health_reports_the_model() {
    let response = routes(state(false)).oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(&to_bytes(response.into_body()).await).unwrap();
    assert_eq!(body, json!({"status": "ready", "model": "imajev-4b"}));
}

#[tokio::test]
async fn cors_preflight_is_answered_when_enabled() {
    let state = ServerState::new(Fake { fail: false }, ServerConfig { cors_origin: Some("*".into()), ..Default::default() });
    let response = routes(state).oneshot(Request::options("/v1/systemone").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(response.headers()["access-control-allow-origin"], "*");
    assert_eq!(response.headers()["access-control-allow-private-network"], "true");
}
