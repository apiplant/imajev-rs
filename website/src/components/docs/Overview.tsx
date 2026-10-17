import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, Section } from "./Prose";
import { Mono } from "../ui";

const link = "text-accent hover:text-accent-dim";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        imajev-rs is a Rust (<Mono>candle</Mono>) port of{" "}
        <a href="https://huggingface.co/mohit67890/imajev-4b" target="_blank" rel="noreferrer noopener" class={link}>
          Imajev
        </a>
        , an image decision model. It reads a photo and the record that goes with it and answers in the options
        you set, with a probability on each and an explicit <em>can't tell</em>.
      </Lead>

      <Section>
        <H2>The model</H2>
        <P>
          imajev-rs runs <IC>Qwen/Qwen3.5-4B</IC> (revision <IC>851bf6e</IC>) with the Imajev LoRA adapter (rank
          64) and its trained 256-code decision readout. The model is not generative: each question is rendered
          as a prompt with lettered options, and the answer is read from the readout at the position where the
          model would start its reply. Every question is shown in four option orders, the log probabilities are
          averaged, and a fitted temperature (1.305) is applied.
        </P>
      </Section>

      <Section>
        <H2>Questions</H2>
        <P>A request is an image, optional evidence (the <IC>state</IC>), and up to eight named questions.</P>
        <UL>
          <LI>
            <IC>choice</IC>: pick one option. <IC>criteria</IC> is <IC>{"{option: description | null}"}</IC>.
          </LI>
          <LI>
            <IC>noul</IC>: a statement or a yes/no question. The answer is the probability that it is true.
            Optional <IC>criteria</IC> <IC>{'{"true": .., "false": ..}'}</IC>.
          </LI>
          <LI>
            <IC>score</IC>: pick a level from an ordered list of 2 to 10 descriptions; the answer is the expected
            level.
          </LI>
        </UL>
        <Pre caption="a request" lang="json">{`{
  "state": {"listing": {"title": "Men's suede boat shoes", "color": "red", "product_type": "shoe"}},
  "questions": {
    "contradicted_field": {
      "type": "choice",
      "instructions": "Which field of \`listing\` does this photo contradict?",
      "criteria": {"listing.color": null, "listing.product_type": null,
                   "none of these": "the photo agrees with every field"}
    },
    "color_matches": {"type": "noul", "instructions": "The product in the photo matches \`listing.color\`."}
  }
}`}</Pre>
        <P>
          <IC>state</IC> is text or a JSON object. Backticks in <IC>instructions</IC> refer to state keys or
          dotted paths. <IC>instructions</IC> can also be an object or an array, flattened to one line per entry.
          Images are JPEG or PNG, up to 20 MiB, one per request.
        </P>
      </Section>

      <Section>
        <H2>Answers</H2>
        <Pre caption="response" lang="json">{`{
  "model": "imajev-4b",
  "answers": {
    "contradicted_field": {
      "type": "choice",
      "choice": "listing.color",
      "probabilities": {"listing.color": 0.914, "listing.product_type": 0.006, "none of these": 0.080},
      "confidence": 0.864,
      "unknown_probability": 0.008,
      "abstained": false
    },
    "color_matches": {"type": "noul", "noul": 0.094, "unknown_probability": 0.007, "abstained": false}
  },
  "usage": {"total_ms": 600.2}
}`}</Pre>
        <UL>
          <LI>
            <IC>probabilities</IC> cover the real options, renormalised without the can't-tell mass;{" "}
            <IC>confidence</IC> is how concentrated they are, scaled by <IC>1 - unknown_probability</IC>.
          </LI>
          <LI>
            <IC>abstained</IC> is true when the model's top pick was can't-tell. A <IC>choice</IC> answer then
            reports <IC>__unknown__</IC>.
          </LI>
          <LI>
            <IC>noul</IC> is P(true) plus half the can't-tell mass, so an undeterminable statement sits near 0.5.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2 id="model-files">Model files</H2>
        <P>
          The adapter (<IC>mohit67890/imajev-4b</IC>, about 490 MB) and the base model (<IC>Qwen/Qwen3.5-4B</IC>,
          about 9.3 GB) are downloaded from Hugging Face on first use into{" "}
          <IC>$XDG_CACHE_HOME/imajev-rs</IC> (<IC>~/.cache/imajev-rs</IC>), as <IC>imajev-4b/</IC> and{" "}
          <IC>Qwen3.5-4B/</IC>. Files are fetched to a <IC>.part</IC> sibling and renamed when complete.
        </P>
        <UL>
          <LI>
            <IC>imajev download</IC> fetches both ahead of time and prints their paths.
          </LI>
          <LI>
            <IC>--adapter DIR</IC> / <IC>--base DIR</IC> (or <IC>IMAJEV_ADAPTER_DIR</IC> /{" "}
            <IC>IMAJEV_BASE_DIR</IC>) use local copies; nothing is downloaded for a directory you pass.
          </LI>
          <LI>
            <IC>IMAJEV_OFFLINE=1</IC> turns a missing file into an error instead of a download.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>Hardware</H2>
        <P>
          The text stack uses fused CUDA kernels for the Gated DeltaNet layers and runs in BF16: about 0.3 to
          0.6 s per image on an RTX 4090 with a few questions. Build with <IC>--features cuda</IC> or install the{" "}
          <IC>imajev-rs-cuda</IC> package. Without CUDA the same model runs on the CPU in F32 and gives the same
          answers, but needs about 17 GB of RAM and is slow; treat it as a reference.
        </P>
      </Section>

      <Section>
        <H2>Next</H2>
        <UL>
          <LI>
            <a href="/docs/cli" class={link}>Command line</a>: <IC>classify</IC>, <IC>predict</IC>, <IC>download</IC>
          </LI>
          <LI>
            <a href="/docs/server" class={link}>HTTP server</a>: <IC>imajev serve</IC> and the API
          </LI>
          <LI>
            <a href="/docs/library" class={link}>As a library</a>: <IC>Engine</IC>, <IC>decide</IC>, the server module
          </LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}
