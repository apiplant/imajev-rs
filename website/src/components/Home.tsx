import { For } from "solid-js";
import { Badge, LinkButton, Mono } from "./ui";
import { CopyBlock } from "./Code";
import { Pre } from "./docs/Prose";
import { highlight } from "../lib/highlight";
import { GITHUB_URL } from "../lib/links";
import { PLATFORMS, CUDA_PLATFORM, LATEST_RELEASE_URL, assetName, downloadUrl } from "../lib/release";

/* A terminal panel showing a real invocation and its (real) output. */
function TerminalDemo() {
  const command =
    "imajev classify listing.jpg \\\n" +
    "  -q 'Which field of `listing` does this photo contradict?' \\\n" +
    "  -l 'listing.color,listing.product_type,none of these:the photo agrees with every field' \\\n" +
    `  --state '{"listing": {"color": "red", "product_type": "shoe"}}'`;
  const output = `listing.jpg
  choice: listing.color  (confidence 0.86, can't-tell 0.01)
    listing.color         0.914  ██████████████████████
    listing.product_type  0.006
    none of these         0.080  ██`;

  return (
    <div class="min-w-0 overflow-hidden rounded-xl border border-line shadow-2xl">
      <div class="flex items-center gap-2 border-b border-line bg-surface px-4 py-2.5">
        <span class="h-2.5 w-2.5 rounded-full bg-danger" />
        <span class="h-2.5 w-2.5 rounded-full bg-warn" />
        <span class="h-2.5 w-2.5 rounded-full bg-success" />
        <span class="ml-2 font-mono text-xs text-faint">imajev</span>
      </div>
      <pre class="overflow-x-auto bg-code-bg px-4 py-4 font-mono text-[0.78rem] leading-relaxed">
        <code class="language-bash">
          <span class="select-none text-faint">$ </span>
          <span innerHTML={highlight(command, "bash")} />
        </code>
        {"\n\n"}
        <code class="text-muted">{output}</code>
      </pre>
    </div>
  );
}

const MODES = [
  {
    name: "imajev classify",
    href: "/docs/cli",
    tagline: "One question about a pile of images",
    body: "Name the options, point at the images, get probabilities. Text or JSON evidence goes next to the photo with --state.",
  },
  {
    name: "imajev predict",
    href: "/docs/cli#predict",
    tagline: "JSONL in, JSONL out",
    body: "Several typed questions per image (choice, yes/no, score), the model loaded once, answers streamed as they finish.",
  },
  {
    name: "imajev serve",
    href: "/docs/server",
    tagline: "The named-question API over HTTP",
    body: "POST /v1/systemone with a photo and your questions. One forward at a time behind a bounded queue, 429 when full, graceful shutdown.",
  },
  {
    name: "imajev-rs (crate)",
    href: "/docs/library",
    tagline: "A Rust library",
    body: "Engine::from_pretrained, then engine.decide(state, questions, image). The server module embeds in your own axum app.",
  },
];

function Modes() {
  return (
    <div class="grid gap-4 sm:grid-cols-2">
      <For each={MODES}>
        {(m) => (
          <a
            href={m.href}
            class="group block rounded-xl border border-line bg-surface p-5 transition-colors hover:border-line-strong"
          >
            <div class="flex items-center justify-between gap-2">
              <h3 class="font-mono text-[0.9375rem] font-semibold tracking-tight text-ink">{m.name}</h3>
              <span class="text-xs text-faint transition-colors group-hover:text-accent">docs →</span>
            </div>
            <p class="mt-1 text-xs font-medium uppercase tracking-[0.1em] text-accent">{m.tagline}</p>
            <p class="mt-2.5 text-sm leading-relaxed text-muted">{m.body}</p>
          </a>
        )}
      </For>
    </div>
  );
}

const FEATURES = [
  {
    title: "Typed answers",
    body: "choice (pick one option), noul (a yes/no statement, answered as a probability) and score (a level from an ordered list).",
  },
  {
    title: "An explicit can't-tell",
    body: "Every answer reports unknown_probability and abstained. Act when the model is sure, send the rest to a person.",
  },
  {
    title: "Photo against a record",
    body: "State goes in as text or JSON next to the image. Backticked names in the question point at state keys, like `listing.color`.",
  },
  {
    title: "Calibrated probabilities",
    body: "Four option orders are averaged per question, then a fitted temperature is applied, as in the reference runtime.",
  },
  {
    title: "Qwen3.5-4B + LoRA",
    body: "The vision tower, the hybrid attention/Gated DeltaNet text stack, the rank-64 adapter and its 256-code readout, all in candle.",
  },
  {
    title: "CUDA first",
    body: "BF16 with fused DeltaNet kernels: about 0.3 to 0.6 s per image on an RTX 4090. The CPU path gives the same answers in F32, slowly.",
  },
  {
    title: "Downloads itself",
    body: "The first run fetches the adapter and the base model (about 10 GB) into ~/.cache/imajev-rs. Point at your own copies with --adapter and --base.",
  },
  {
    title: "Small surface",
    body: "One binary, one crate. Turn off the cli and server features and the library is just the engine.",
  },
];

function Features() {
  return (
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <For each={FEATURES}>
        {(f) => (
          <div class="rounded-xl border border-line bg-surface p-5">
            <h3 class="text-[0.9375rem] font-semibold tracking-tight text-ink">{f.title}</h3>
            <p class="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        )}
      </For>
    </div>
  );
}

const stepCard =
  "grid min-w-0 gap-5 rounded-2xl border bg-surface p-5 sm:p-6 lg:grid-cols-[minmax(14rem,0.7fr)_minmax(0,1.3fr)] lg:items-start";

function InstallSteps() {
  const homebrewCommands = `brew tap apiplant/tap
brew install apiplant/tap/imajev-rs`;
  const pacmanCommands = `curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy imajev-rs        # or imajev-rs-cuda`;
  const aptCommands = `curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install imajev-rs        # or imajev-rs-cuda`;
  const cargoCommands = `cargo add imajev-rs         # as a library dependency
cargo install imajev-rs     # the imajev binary
cargo install imajev-rs --features cuda  # with CUDA support`;

  return (
    <div class="mt-8 space-y-4 sm:mt-10">
      <div class={`${stepCard} border-accent-line`}>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs text-accent">01</span>
            <Badge tone="accent">Recommended</Badge>
          </div>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Use a package manager</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            macOS (Apple Silicon), Arch Linux and Debian/Ubuntu are published to the apiplant shared
            repositories. With an NVIDIA GPU on Linux x86_64, install the <Mono>imajev-rs-cuda</Mono>{" "}
            package instead.
          </p>
        </div>

        <div class="min-w-0 space-y-5">
          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">Homebrew</p>
            <CopyBlock command={homebrewCommands} />
          </div>
          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">Arch Linux / pacman</p>
            <CopyBlock command={pacmanCommands} />
          </div>
          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">Debian / Ubuntu</p>
            <CopyBlock command={aptCommands} />
          </div>
        </div>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-accent">02</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Download the archive</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            One archive per platform holding <Mono>imajev</Mono> and the README. Unpack and run. The model
            is built for CUDA, so on Linux x86_64 with an NVIDIA GPU take the CUDA archive and pass{" "}
            <Mono>--device cuda</Mono>.
          </p>
        </div>

        <ul class="min-w-0 space-y-1 border-t border-line pt-4 lg:border-t-0 lg:pt-0">
          <For each={[CUDA_PLATFORM, ...PLATFORMS]}>
            {(platform) => (
              <li class="min-w-0">
                <a
                  href={downloadUrl(platform)}
                  title={assetName(platform)}
                  class="flex min-w-0 items-baseline justify-between gap-3 rounded-md py-1 text-muted transition-colors hover:text-ink"
                >
                  <span class="shrink-0 text-sm">{platform.label}</span>
                  <span class="min-w-0 truncate font-mono text-xs text-accent">{assetName(platform)}</span>
                </a>
              </li>
            )}
          </For>
        </ul>

        <a
          href={LATEST_RELEASE_URL}
          target="_blank"
          rel="noreferrer noopener"
          class="lg:col-start-2 text-sm font-medium text-accent hover:text-accent-dim"
        >
          All releases and checksums
        </a>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-faint">03</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Cargo</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            As a library dependency, or to build the binary from source via crates.io.
          </p>
        </div>
        <div class="min-w-0">
          <CopyBlock command={cargoCommands} />
        </div>
      </div>
    </div>
  );
}

export function Home() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5">
      <section class="grid items-center gap-10 py-16 sm:py-20 lg:grid-cols-2 lg:gap-12">
        <div class="min-w-0">
          <div class="flex flex-wrap items-center gap-2">
            <Badge tone="accent">v{__VERSION__}</Badge>
            <Badge>Rust</Badge>
            <Badge>candle</Badge>
            <Badge>Apache-2.0</Badge>
          </div>
          <h1 class="mt-5 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
            Imajev, <span class="text-accent">in Rust</span>.
          </h1>
          <p class="mt-4 max-w-lg text-lg leading-relaxed text-muted">
            Ask questions about a photo and the record that goes with it, or about the text alone. Get an answer in the options you
            set, a probability on each, and an explicit can't-tell. Run it from a CLI, an HTTP server or
            your own Rust code.
          </p>
          <div class="mt-7 flex flex-wrap gap-3">
            <LinkButton href="/try" variant="primary" size="lg">
              Try it against your server
            </LinkButton>
            <LinkButton href="/docs" size="lg">
              Read the docs
            </LinkButton>
            <LinkButton href={GITHUB_URL} size="lg">
              View on GitHub
            </LinkButton>
          </div>
          <p class="mt-5 text-sm text-faint">
            Runs <Mono>Qwen/Qwen3.5-4B</Mono> with the <Mono>mohit67890/imajev-4b</Mono> adapter. The model
            is too large for a browser, so there is no in-page demo; the try-it page talks to a server you
            run.
          </p>
        </div>
        <TerminalDemo />
      </section>

      <section class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">One binary, one crate</h2>
        <p class="mt-2 max-w-2xl text-muted">
          The same engine behind a classification command, a JSONL batch mode, an HTTP server and a library.
        </p>
        <div class="mt-8">
          <Modes />
        </div>
      </section>

      <section id="features" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">What it does</h2>
        <div class="mt-8">
          <Features />
        </div>
      </section>

      <section id="install" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Install</h2>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          Use Homebrew, pacman or apt when your platform has it. Otherwise take the prebuilt archive, or pull
          the crate from crates.io. The first run downloads the model files (about 10 GB) into{" "}
          <Mono>~/.cache/imajev-rs</Mono>.
        </p>
        <InstallSteps />
      </section>

      <section class="border-t border-line pb-20 pt-16">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-2xl font-semibold tracking-tight text-ink">As a library</h2>
          <LinkButton href="/docs/library" size="sm">
            Full library docs →
          </LinkButton>
        </div>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          One engine, one call. Every question in the request is answered about the same image.
        </p>
        <div class="mt-6">
          <CopyBlock command="cargo add imajev-rs --features cuda" />
        </div>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::Device;
use imajev_rs::{Engine, State};

let mut engine = Engine::from_pretrained(&Device::new_cuda(0)?)?;
let questions = serde_json::json!({"page_type": {"type": "choice",
    "instructions": "Choose the best page role.",
    "criteria": {"exercise": "A question or task for a student.",
                 "theory": "Explanations or reference material."}}});
let answers = engine.decide(&State::from("Practice questions"), &questions, Some(&std::fs::read("page.jpg")?))?;
println!("{:?}", answers[0].1.choice);`}</Pre>
      </section>
    </div>
  );
}
