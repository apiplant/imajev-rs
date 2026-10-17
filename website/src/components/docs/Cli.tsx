import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>Command line</H1>
      <Lead>
        <IC>imajev</IC> has four commands: <IC>classify</IC> for a quick question about some images,{" "}
        <IC>predict</IC> for JSONL batches, <IC>serve</IC> for HTTP and <IC>download</IC> to fetch the model.
      </Lead>

      <Section>
        <H2>Global options</H2>
        <FlagTable
          rows={[
            { flag: "--adapter DIR", meaning: <>Adapter directory (<IC>adapter_model.safetensors</IC>, <IC>decision_readout.*</IC>). Env <IC>IMAJEV_ADAPTER_DIR</IC>. Default: the cache, downloaded if missing.</> },
            { flag: "--base DIR", meaning: <>Qwen3.5-4B checkpoint directory. Env <IC>IMAJEV_BASE_DIR</IC>; <IC>IMAJEV_MODEL_BUNDLE</IC> may name a JSON file whose <IC>path</IC> is the directory.</> },
            { flag: "--device D", meaning: <><IC>cpu</IC>, <IC>cuda</IC> or <IC>cuda:N</IC>. Env <IC>IMAJEV_DEVICE</IC>. Defaults to <IC>cuda</IC> in a CUDA build, else <IC>cpu</IC>.</> },
          ]}
        />
      </Section>

      <Section>
        <H2 id="classify">imajev classify</H2>
        <P>One choice question, asked about each image.</P>
        <CopyBlock
          command={`imajev classify page.jpg -l exercise,theory
imajev classify listing.jpg -q 'Which field of \`listing\` does this photo contradict?' \\
  -l 'listing.color,listing.product_type,none of these:the photo agrees with every field' \\
  --state '{"listing": {"color": "red", "product_type": "shoe"}}'
imajev classify scans/*.png -l 'invoice:A bill to pay,receipt:Proof of payment' --format json`}
        />
        <FlagTable
          rows={[
            { flag: "IMAGE...", meaning: "One or more JPEG/PNG files. The model loads once." },
            { flag: "-l, --labels", meaning: <>Options, comma-separated. <IC>option:description</IC> adds a description; use <IC>;</IC> as the separator when a description has commas.</> },
            { flag: "-q, --question", meaning: <>The question (default <IC>Choose the best option.</IC>).</> },
            { flag: "-n, --name", meaning: <>Name of the question in the JSON output (default <IC>choice</IC>).</> },
            { flag: "-s, --state", meaning: "Evidence next to the image: text, or a JSON object." },
            { flag: "--format", meaning: <><IC>text</IC> (default, probability bars) or <IC>json</IC> (one object per image).</> },
          ]}
        />
        <Pre caption="text output" lang="console">{`listing.jpg
  choice: listing.color  (confidence 0.86, can't-tell 0.01)
    listing.color         0.914  ██████████████████████
    listing.product_type  0.006
    none of these         0.080  ██`}</Pre>
      </Section>

      <Section>
        <H2 id="predict">imajev predict</H2>
        <P>
          Reads one request per line from <IC>--input FILE</IC> or stdin and writes one answer object per line.
          Use it for several questions per image, or for <IC>noul</IC> and <IC>score</IC> questions.
        </P>
        <CopyBlock
          command={`echo '{"image": "listing.jpg", "state": {"listing": {"color": "red"}}, "questions": {
  "color_matches": {"type": "noul", "instructions": "The product in the photo matches \`listing.color\`."},
  "formality": {"type": "score", "instructions": "How formal are these shoes?",
                "criteria": ["casual", "smart casual", "formal"]}}}' | imajev predict`}
        />
        <P>
          Each line has <IC>image</IC> (a file path), <IC>questions</IC> and an optional <IC>state</IC>. Output
          lines are <IC>{'{"answers": {...}}'}</IC> in input order.
        </P>
      </Section>

      <Section>
        <H2>imajev serve</H2>
        <P>
          Runs the HTTP API. See <a href="/docs/server" class="text-accent hover:text-accent-dim">HTTP server</a>.
        </P>
        <FlagTable
          rows={[
            { flag: "--host, --port", meaning: <>Default <IC>127.0.0.1:8765</IC> (env <IC>IMAJEV_HOST</IC>, <IC>IMAJEV_PORT</IC>).</> },
            { flag: "--model-name", meaning: <>Identity reported by <IC>/health</IC> (default <IC>imajev-4b</IC>).</> },
            { flag: "--max-queued", meaning: "Waiting slots on top of the one in flight (default 16); beyond that, 429." },
            { flag: "--cors ORIGIN", meaning: <>Allow browser pages from this origin (<IC>*</IC> for any).</> },
          ]}
        />
      </Section>

      <Section>
        <H2>imajev download</H2>
        <P>
          Fetches the adapter and the base model into the cache (about 10 GB) and prints where they are. Re-running
          it skips files that are already complete.
        </P>
      </Section>
    </DocsLayout>
  );
}
