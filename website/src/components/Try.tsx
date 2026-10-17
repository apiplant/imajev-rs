import { createSignal, For, Show } from "solid-js";
import { Button } from "./ui";
import { JsonView } from "./demo/JsonView";
import { CopyBlock } from "./Code";

/* A request builder for `imajev serve`. The model is far too large for WebAssembly, so the page does not run
   inference itself: it sends a real multipart request from the browser to a server the reader runs. */

interface Preset {
  label: string;
  state: string;
  questions: string;
}

const PRESETS: Preset[] = [
  {
    label: "Photo against a listing",
    state: JSON.stringify({ listing: { title: "Men's suede boat shoes", color: "red", product_type: "shoe" } }, null, 2),
    questions: JSON.stringify(
      {
        contradicted_field: {
          type: "choice",
          instructions: "Which field of `listing` does this photo contradict?",
          criteria: {
            "listing.color": null,
            "listing.product_type": null,
            "none of these": "the photo agrees with every field",
          },
        },
        color_matches: { type: "noul", instructions: "The product in the photo matches `listing.color`." },
      },
      null,
      2,
    ),
  },
  {
    label: "Page type",
    state: "Practice questions\nPage 1 of 1",
    questions: JSON.stringify(
      {
        page_type: {
          type: "choice",
          instructions: "Choose the best page role.",
          criteria: {
            exercise: "Contains a question or task for a student.",
            theory: "Contains explanations or reference material without a student task.",
          },
        },
      },
      null,
      2,
    ),
  },
  {
    label: "What is it?",
    state: "",
    questions: JSON.stringify(
      {
        subject: {
          type: "choice",
          instructions: "What is the main subject of the photo?",
          criteria: { person: null, animal: null, product: null, document: null, landscape: null },
        },
        formality: {
          type: "score",
          instructions: "How formal is what the photo shows?",
          criteria: ["casual", "smart casual", "formal"],
        },
      },
      null,
      2,
    ),
  },
];

interface ChoiceLike {
  type?: string;
  choice?: string;
  probabilities?: Record<string, number>;
  noul?: number;
  score?: number;
  confidence?: number;
  unknown_probability?: number;
  abstained?: boolean;
}

const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

function AnswerCard(props: { id: string; answer: ChoiceLike }) {
  const a = () => props.answer;
  const verdict = () => {
    if (a().type === "noul") return `${((a().noul ?? 0) * 100).toFixed(1)}% yes`;
    if (a().type === "score") return `score ${(a().score ?? 0).toFixed(2)}`;
    return a().abstained ? "can't tell" : (a().choice ?? "");
  };
  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex flex-wrap items-baseline justify-between gap-2">
        <p class="font-mono text-xs text-faint">{props.id}</p>
        <p class="text-xs text-faint">
          can't tell {((a().unknown_probability ?? 0) * 100).toFixed(1)}%
          <Show when={a().confidence !== undefined}> · confidence {(a().confidence ?? 0).toFixed(2)}</Show>
        </p>
      </div>
      <p class={`mt-1 text-lg font-semibold ${a().abstained ? "text-warn" : "text-ink"}`}>{verdict()}</p>
      <Show when={a().probabilities}>
        {(probs) => (
          <ul class="mt-3 space-y-1.5">
            <For each={Object.entries(probs())}>
              {([name, p]) => (
                <li class="grid grid-cols-[minmax(0,10rem)_1fr_3.5rem] items-center gap-3 text-xs">
                  <span class="truncate font-mono text-muted" title={name}>
                    {name}
                  </span>
                  <span class="h-2 overflow-hidden rounded-full bg-surface-3">
                    <span class="block h-full rounded-full bg-accent" style={{ width: `${Math.max(p * 100, 0.5)}%` }} />
                  </span>
                  <span class="text-right text-muted">{(p * 100).toFixed(1)}%</span>
                </li>
              )}
            </For>
          </ul>
        )}
      </Show>
    </div>
  );
}

export function Try() {
  const [server, setServer] = createSignal("http://127.0.0.1:8765");
  const [preset, setPreset] = createSignal(0);
  const [state, setState] = createSignal(PRESETS[0].state);
  const [questions, setQuestions] = createSignal(PRESETS[0].questions);
  const [file, setFile] = createSignal<File | null>(null);
  const [preview, setPreview] = createSignal<string | null>(null);
  const [status, setStatus] = createSignal<string | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [running, setRunning] = createSignal(false);
  const [result, setResult] = createSignal<{ answers: Record<string, ChoiceLike>; model: string; usage?: { total_ms?: number } } | null>(null);

  const base = () => server().trim().replace(/\/+$/, "");

  function choosePreset(index: number) {
    setPreset(index);
    setState(PRESETS[index].state);
    setQuestions(PRESETS[index].questions);
  }

  function onFile(next: File | null) {
    const old = preview();
    if (old) URL.revokeObjectURL(old);
    setFile(next);
    setPreview(next ? URL.createObjectURL(next) : null);
  }

  /** `state` is sent as JSON when it parses as an object/array, else as text. */
  function requestBody(): unknown {
    const text = state().trim();
    let stateValue: unknown = text;
    if (text.startsWith("{") || text.startsWith("[")) {
      try {
        stateValue = JSON.parse(text);
      } catch {
        throw new Error("State looks like JSON but does not parse.");
      }
    }
    let q: unknown;
    try {
      q = JSON.parse(questions());
    } catch {
      throw new Error("Questions are not valid JSON.");
    }
    return text === "" ? { questions: q } : { state: stateValue, questions: q };
  }

  async function check() {
    setError(null);
    setStatus("Checking…");
    try {
      const r = await fetch(`${base()}/health`);
      const body = await r.json();
      setStatus(`Connected: ${body.model ?? "ready"}`);
    } catch {
      setStatus(null);
      setError(
        `Cannot reach ${base()}. Start a server with: imajev serve --cors '*'  (or --cors ${location.origin}).`,
      );
    }
  }

  async function run() {
    setError(null);
    setResult(null);
    setStatus(null);
    const image = file();
    if (!image) return setError("Choose an image first.");
    if (image.size > MAX_IMAGE_BYTES) return setError("The server accepts images up to 20 MiB.");
    let body: unknown;
    try {
      body = requestBody();
    } catch (e) {
      return setError(e instanceof Error ? e.message : String(e));
    }
    const form = new FormData();
    form.set("request", JSON.stringify(body));
    form.set("image", image, image.name);
    setRunning(true);
    try {
      const r = await fetch(`${base()}/v1/systemone`, { method: "POST", body: form });
      const json = await r.json().catch(() => null);
      if (!r.ok) {
        setError(json?.error?.message ?? json?.detail ?? `Server answered ${r.status}.`);
        return;
      }
      setResult(json);
    } catch {
      setError(`Cannot reach ${base()}. Start a server with: imajev serve --cors '*'`);
    } finally {
      setRunning(false);
    }
  }

  const curl = () => {
    let body = "{}";
    try {
      body = JSON.stringify(requestBody());
    } catch {
      /* the editors show the problem */
    }
    return `curl -s ${base()}/v1/systemone \\\n  -F 'request=${body.replace(/'/g, "'\\''")}' \\\n  -F 'image=@${file()?.name ?? "photo.jpg"};type=image/jpeg'`;
  };

  const field = "w-full rounded-xl border border-line bg-surface p-3 font-mono text-[0.8125rem] text-ink outline-none focus:border-accent";

  return (
    <div class="mx-auto w-full max-w-5xl px-5 py-10">
      <p class="font-mono text-xs text-accent">talks to an imajev server you run · nothing is uploaded elsewhere</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight text-ink">Try it</h1>
      <p class="mt-2 max-w-2xl leading-relaxed text-muted">
        Imajev is a 4B-parameter vision-language model, too large to run in a page. Start{" "}
        <code class="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.85em] text-ink">imajev serve --cors '*'</code>{" "}
        on a machine with a GPU, point this page at it, and send a photo with your questions. The request goes
        straight from your browser to your server.
      </p>

      <div class="mt-6 grid gap-6 lg:grid-cols-2">
        <div class="min-w-0 space-y-4">
          <div>
            <label class="text-xs font-semibold uppercase tracking-[0.12em] text-faint" for="server">
              Server
            </label>
            <div class="mt-1.5 flex gap-2">
              <input
                id="server"
                value={server()}
                onInput={(e) => setServer(e.currentTarget.value)}
                class={`${field} !py-2`}
                spellcheck={false}
              />
              <Button onClick={check}>Check</Button>
            </div>
            <Show when={status()}>
              <p class="mt-1.5 text-xs text-success">{status()}</p>
            </Show>
          </div>

          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.12em] text-faint">Image</p>
            <label class="mt-1.5 flex min-h-32 cursor-pointer items-center justify-center overflow-hidden rounded-xl border border-dashed border-line-strong bg-surface text-sm text-muted hover:border-accent">
              <input
                type="file"
                accept="image/jpeg,image/png"
                class="sr-only"
                onChange={(e) => onFile(e.currentTarget.files?.[0] ?? null)}
              />
              <Show when={preview()} fallback={<span>Choose a JPEG or PNG</span>}>
                <img src={preview()!} alt="Selected" class="max-h-64 w-full object-contain" />
              </Show>
            </label>
          </div>

          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.12em] text-faint">Example</p>
            <div class="mt-1.5 flex flex-wrap gap-2">
              <For each={PRESETS}>
                {(p, i) => (
                  <Button size="sm" variant={preset() === i() ? "primary" : "secondary"} onClick={() => choosePreset(i())}>
                    {p.label}
                  </Button>
                )}
              </For>
            </div>
          </div>

          <div>
            <label class="text-xs font-semibold uppercase tracking-[0.12em] text-faint" for="state">
              State (text or JSON, optional)
            </label>
            <textarea
              id="state"
              rows={4}
              value={state()}
              onInput={(e) => setState(e.currentTarget.value)}
              class={`${field} mt-1.5`}
              spellcheck={false}
            />
          </div>
        </div>

        <div class="min-w-0 space-y-4">
          <div>
            <label class="text-xs font-semibold uppercase tracking-[0.12em] text-faint" for="questions">
              Questions (JSON)
            </label>
            <textarea
              id="questions"
              rows={16}
              value={questions()}
              onInput={(e) => setQuestions(e.currentTarget.value)}
              class={`${field} mt-1.5`}
              spellcheck={false}
            />
          </div>
          <Button variant="primary" disabled={running()} onClick={run}>
            {running() ? "Waiting for the model…" : "Send"}
          </Button>
          <Show when={error()}>
            <p class="text-sm text-danger">{error()}</p>
          </Show>
        </div>
      </div>

      <Show when={result()}>
        {(r) => (
          <section class="mt-8 space-y-4">
            <div class="flex flex-wrap items-baseline justify-between gap-2">
              <h2 class="text-lg font-semibold tracking-tight text-ink">Answers</h2>
              <p class="text-xs text-faint">
                {r().model}
                <Show when={r().usage?.total_ms}> · {r().usage?.total_ms} ms</Show>
              </p>
            </div>
            <div class="grid gap-3 md:grid-cols-2">
              <For each={Object.entries(r().answers)}>{([id, answer]) => <AnswerCard id={id} answer={answer} />}</For>
            </div>
            <JsonView value={r()} />
          </section>
        )}
      </Show>

      <section class="mt-10">
        <h2 class="text-lg font-semibold tracking-tight text-ink">The same request with curl</h2>
        <div class="mt-3">
          <CopyBlock command={curl()} />
        </div>
      </section>
    </div>
  );
}
