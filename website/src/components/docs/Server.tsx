import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsServer() {
  return (
    <DocsLayout>
      <H1>HTTP server</H1>
      <Lead>
        <IC>imajev serve</IC> loads the model once and answers named-question requests over HTTP, with the same
        request and response shapes as the reference Imajev server.
      </Lead>

      <Section>
        <CopyBlock command={`imajev serve                                  # 127.0.0.1:8765
imajev serve --device cuda:1 --host 0.0.0.0 --port 9000`} />
        <CopyBlock
          command={`curl -s localhost:8765/v1/systemone \\
  -F 'request={"state": {"page_text": "Practice questions"}, "questions": {"page_type": {"type": "choice",
      "instructions": "Choose the best page role.",
      "criteria": {"exercise": "A question or task for a student.", "theory": "Explanations or reference material."}}}}' \\
  -F 'image=@page.jpg;type=image/jpeg'`}
        />
      </Section>

      <Section>
        <H2>Endpoints</H2>
        <FlagTable
          rows={[
            { flag: "POST /v1/systemone", meaning: <>Multipart form: <IC>request</IC> (JSON <IC>{'{"state", "questions", "model"?}'}</IC>) and one <IC>image</IC>. Alias: <IC>/v1/classifier</IC>.</> },
            { flag: "GET /health", meaning: <><IC>{'{"status":"ready","model":"imajev-4b"}'}</IC>, no inference.</> },
          ]}
        />
        <P>
          A <IC>model</IC> field in the request, if sent, must equal <IC>--model-name</IC>. The response is{" "}
          <IC>{'{"model", "answers", "usage": {"total_ms"}}'}</IC>; see the{" "}
          <a href="/docs" class="text-accent hover:text-accent-dim">overview</a> for the answer shapes.
        </P>
      </Section>

      <Section>
        <H2>Errors and back-pressure</H2>
        <FlagTable
          rows={[
            { flag: "422", meaning: <>Bad input (invalid JSON, unsupported question, missing or oversized image): <IC>{'{"error": {"message", "type": "invalid_request_error", "code": 422, "param"}}'}</IC>.</> },
            { flag: "429", meaning: <>The admission queue is full. <IC>Retry-After: 1</IC>.</> },
            { flag: "500", meaning: <><IC>{'{"detail": "internal error"}'}</IC>; the cause goes to stderr.</> },
          ]}
        />
        <P>
          Requests run strictly one at a time behind a bounded queue (<IC>--max-queued</IC>, default 16). Ctrl-C or
          SIGTERM stops the listener and lets an in-flight request finish first.
        </P>
      </Section>

      <Section>
        <H2>From a browser</H2>
        <P>
          Start the server with <IC>--cors '*'</IC> (or your site's origin) and a web page can call it. The{" "}
          <a href="/try" class="text-accent hover:text-accent-dim">try-it page</a> does exactly that. Preflights,
          including Chrome's private-network preflight for pages calling <IC>localhost</IC>, are answered.
        </P>
        <Pre caption="response headers with --cors" lang="text">{`access-control-allow-origin: *
access-control-allow-methods: GET, POST, OPTIONS
access-control-allow-headers: content-type
access-control-allow-private-network: true`}</Pre>
      </Section>
    </DocsLayout>
  );
}
