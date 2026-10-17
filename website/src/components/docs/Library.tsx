import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>imajev-rs</IC> crate is the engine behind the CLI and the server: load the model once, then call{" "}
        <IC>decide</IC> for each image.
      </Lead>

      <Section>
        <H2>Add it</H2>
        <CopyBlock command={`cargo add imajev-rs                      # CPU
cargo add imajev-rs --features cuda      # + CUDA (needs nvcc to build)`} />
        <P>
          <IC>cli</IC> (clap) and <IC>server</IC> (axum, tokio) are on by default so that{" "}
          <IC>cargo install imajev-rs</IC> gives the full binary. As a library you rarely need them:
        </P>
        <Pre caption="Cargo.toml" lang="bash">{`imajev-rs = { version = "0.1", default-features = false }                           # engine only
imajev-rs = { version = "0.1", default-features = false, features = ["server"] }    # + the HTTP server module`}</Pre>
      </Section>

      <Section>
        <H2>Decide</H2>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::Device;
use imajev_rs::{answers_to_json, Engine, State};

// Downloads the adapter and Qwen3.5-4B into ~/.cache/imajev-rs on first use (about 10 GB);
// or Engine::load("path/to/imajev-4b", "path/to/Qwen3.5-4B", &device) for local copies.
let mut engine = Engine::from_pretrained(&Device::new_cuda(0)?)?;

let questions = serde_json::json!({
    "color_matches": {"type": "noul", "instructions": "The product in the photo matches \`listing.color\`."},
    "contradicted_field": {"type": "choice",
        "instructions": "Which field of \`listing\` does this photo contradict?",
        "criteria": {"listing.color": null, "listing.product_type": null, "none of these": null}},
});
let state = State::from_value(&serde_json::json!({"listing": {"color": "red", "product_type": "shoe"}})).unwrap();
let answers = engine.decide(&state, &questions, &std::fs::read("listing.jpg")?)?;

for (id, answer) in &answers {
    println!("{id}: {:?} confidence {:.2}", answer.choice, answer.confidence);
}
println!("{}", answers_to_json(&answers)); // the same JSON the HTTP API returns`}</Pre>
        <P>
          <IC>decide</IC> returns <IC>Answers</IC>, a list of <IC>(id, Answer)</IC> in question order. An{" "}
          <IC>Answer</IC> has <IC>kind</IC>, <IC>choice</IC> (<IC>None</IC> when it abstained),{" "}
          <IC>probabilities</IC>, <IC>value</IC> (the <IC>noul</IC> probability or the <IC>score</IC>),{" "}
          <IC>confidence</IC>, <IC>unknown_probability</IC>, <IC>abstained</IC> and the score <IC>legend</IC>.
        </P>
        <P>
          Bad requests (an unsupported question, an unreadable image, more than 4096 processed tokens) come back
          as an <IC>InvalidRequest</IC> error you can <IC>downcast_ref</IC>; anything else is a failed forward.
          <IC>Engine::validate</IC> and <IC>validate_request</IC> check a request without running the model.
        </P>
      </Section>

      <Section>
        <H2>Embed the server</H2>
        <P>
          <IC>imajev_rs::server</IC> serves anything that implements <IC>Decider</IC> (the engine does).{" "}
          <IC>start_server</IC> binds and runs with graceful shutdown; <IC>routes(state)</IC> is a plain axum{" "}
          <IC>Router</IC> to merge into your own application.
        </P>
        <Pre caption="serve" lang="rust">{`use imajev_rs::server::{start_server, ServerConfig};

let handle = start_server("127.0.0.1", 8765, engine, ServerConfig::default()).await?;
println!("listening on http://{}", handle.addr);
handle.wait().await; // until Ctrl-C or SIGTERM`}</Pre>
      </Section>

      <Section>
        <H2>Notes</H2>
        <P>
          <IC>decide</IC> takes <IC>&mut self</IC> because the recurrent state of the DeltaNet layers lives in the
          model, so one engine answers one request at a time. Share it behind a mutex, as the server does.
        </P>
      </Section>
    </DocsLayout>
  );
}
