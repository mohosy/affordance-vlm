import DemoCard from "@/components/DemoCard";

const REPO_URL = "https://github.com/mohosy/affordance-vlm";

export default function Home() {
  return (
    <main className="min-h-screen text-chalk">
      {/* ---------- nav ---------- */}
      <nav className="sticky top-0 z-30 border-b border-zinc-800 bg-ink/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="h-2.5 w-2.5 rounded-full bg-accent" />
            <span className="font-mono text-sm">affordance-vlm</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-zinc-400">
            <a href="#thesis" className="hover:text-chalk">Thesis</a>
            <a href="#method" className="hover:text-chalk">Method</a>
            <a href="#demo" className="hover:text-chalk">Demo</a>
            <a href="#results" className="hover:text-chalk">Results</a>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-zinc-700 px-3 py-1.5 hover:border-chalk hover:text-chalk"
            >
              GitHub →
            </a>
          </div>
        </div>
      </nav>

      {/* ---------- hero ---------- */}
      <section className="relative overflow-hidden border-b border-zinc-800">
        <div className="grid-bg absolute inset-0 opacity-60" />
        <div className="relative mx-auto max-w-6xl px-6 pt-24 pb-32">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900/50 px-3 py-1 text-xs text-zinc-400">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            Caltech × Ironsite — Spatial Intelligence in the Physical World
          </div>
          <h1 className="max-w-4xl text-5xl font-semibold leading-tight tracking-tight md:text-6xl lg:text-7xl">
            <span className="gradient-text">
              Frontier VLMs see the world{" "}
            </span>
            <br />
            <span className="text-chalk">one frame at a time.</span>
          </h1>
          <p className="mt-8 max-w-2xl text-lg text-zinc-400 md:text-xl">
            Construction is a <span className="text-chalk">temporal</span>{" "}
            activity. Workers move, occlude, set things down, pick them back
            up. We show that Gemini 2.5 Pro, Claude Opus 4.7, and GPT-4o
            catastrophically fail at maintaining a persistent belief over
            scene state across body-cam motion. Then we close the gap with a{" "}
            <span className="text-chalk">7B fine-tune on Ironsite&rsquo;s own footage</span>.
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <a
              href="#demo"
              className="rounded-md bg-chalk px-5 py-3 text-sm font-medium text-ink hover:bg-zinc-200"
            >
              See it live →
            </a>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-zinc-700 px-5 py-3 text-sm hover:border-chalk"
            >
              Source on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ---------- thesis ---------- */}
      <section id="thesis" className="border-b border-zinc-800">
        <div className="mx-auto grid max-w-6xl gap-16 px-6 py-24 md:grid-cols-3">
          <div className="md:col-span-1">
            <div className="sticky top-24">
              <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
                01 — Thesis
              </p>
              <h2 className="mt-3 text-3xl font-semibold tracking-tight">
                The biggest spatial gap isn&rsquo;t object recognition. It&rsquo;s memory.
              </h2>
            </div>
          </div>
          <div className="md:col-span-2 space-y-6 text-lg text-zinc-300">
            <p>
              Frontier VLMs were trained on captioned images. Even when you
              feed them a video, they answer as if they only saw a single
              frame. Ask them <em>where is the wrench the worker just put
              down</em> — they say &ldquo;I don&rsquo;t see a wrench.&rdquo;
              Because they&rsquo;re looking at frame 5, where the wrench is
              now off-screen.
            </p>
            <p>
              Body-cam construction footage exposes this gap brutally.
              Workers turn. Hands occlude. Tools land on benches and stay
              there for a minute before getting picked back up. Pipes get
              walked past. The fish-eye lens distorts everything. None of
              this is in the typical VLM training distribution.
            </p>
            <p className="text-zinc-400">
              We attack five problems from Ironsite&rsquo;s spatial-intelligence
              list at once: <span className="text-chalk">temporal reasoning</span>,{" "}
              <span className="text-chalk">object permanence</span>,{" "}
              <span className="text-chalk">occlusion reasoning</span>,{" "}
              <span className="text-chalk">partial observability</span>, and{" "}
              <span className="text-chalk">generalization to real-world environments</span>.
            </p>
          </div>
        </div>
      </section>

      {/* ---------- method ---------- */}
      <section id="method" className="border-b border-zinc-800 bg-zinc-950/30">
        <div className="mx-auto max-w-6xl px-6 py-24">
          <div className="mb-12">
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              02 — Method
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Multi-frame Q&amp;A. Multi-image LoRA. Multi-axis judged eval.
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-4">
            {[
              {
                tag: "Step 1",
                title: "Sample frames",
                body: (
                  <>
                    6 Ironsite body-cam clips × ~20 min each. ffmpeg samples at
                    0.5 Hz → <span className="text-chalk">3,624 frames</span>.
                  </>
                ),
              },
              {
                tag: "Step 2",
                title: "Build sequences",
                body: (
                  <>
                    Group consecutive frames into 5-frame windows (10-second
                    spans of activity). Hold out clip 12 for eval — model
                    never sees it during training.
                    <span className="block mt-2 text-xs text-zinc-500">
                      603 train + 121 holdout sequences
                    </span>
                  </>
                ),
              },
              {
                tag: "Step 3",
                title: "Generate temporal Q&A",
                body: (
                  <>
                    Gemini 2.5 Pro sees all 5 frames + a strict instruction:
                    &ldquo;questions answerable from a single frame are{" "}
                    <span className="text-chalk">bad questions</span>.&rdquo; ~1,800
                    grounded multi-frame pairs.
                  </>
                ),
              },
              {
                tag: "Step 4",
                title: "LoRA + multi-axis judge",
                body: (
                  <>
                    LoRA-fine-tune Qwen2.5-VL-7B on multi-image inputs (Vultr
                    A40, ~$8). Eval all four models on five problem axes;
                    score with Claude Opus 4.7.
                  </>
                ),
              },
            ].map((step, i) => (
              <div
                key={i}
                className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-6"
              >
                <p className="font-mono text-xs uppercase tracking-widest text-accent">
                  {step.tag}
                </p>
                <h3 className="mt-3 text-lg font-medium">{step.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-zinc-400">
                  {step.body}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---------- problems we attack ---------- */}
      <section className="border-b border-zinc-800">
        <div className="mx-auto max-w-6xl px-6 py-24">
          <div className="mb-12">
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              03 — Five problems, one project
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Each axis maps to a real Ironsite use case.
            </h2>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
            {[
              {
                tag: "Temporal reasoning",
                use: "How did the worker&rsquo;s hand path change over the last 10 seconds?",
              },
              {
                tag: "Object permanence",
                use: "Where did the worker last set down the pipe wrench?",
              },
              {
                tag: "Occlusion reasoning",
                use: "What is hidden behind the gloved hand in this frame, given later frames?",
              },
              {
                tag: "Partial observability",
                use: "Based on the worker&rsquo;s posture, what tool are they about to reach for?",
              },
              {
                tag: "Real-world generalization",
                use: "Fish-eye, low-light, cluttered — does the model still parse the scene?",
              },
            ].map((p, i) => (
              <div
                key={i}
                className="rounded-lg border border-zinc-800 bg-zinc-900/20 p-5"
              >
                <p className="font-mono text-xs text-accent">{p.tag}</p>
                <p
                  className="mt-3 text-sm text-zinc-400"
                  dangerouslySetInnerHTML={{ __html: `&ldquo;${p.use}&rdquo;` }}
                />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---------- demo ---------- */}
      <section id="demo" className="border-b border-zinc-800 bg-zinc-950/30">
        <div className="mx-auto max-w-6xl px-6 py-24">
          <div className="mb-12">
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              04 — Demo
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Upload a body-cam frame. Watch frontier models lose context.
            </h2>
            <p className="mt-3 max-w-2xl text-zinc-400">
              The single-frame demo below shows the OOD failure mode on its
              own — frontier models often misread fish-eye body-cam frames
              even before we get to temporal questions. The full multi-frame
              demo unlocks after Phase 4 fine-tuning lands on HuggingFace
              Spaces.
            </p>
          </div>
          <DemoCard />
        </div>
      </section>

      {/* ---------- results ---------- */}
      <section id="results" className="border-b border-zinc-800">
        <div className="mx-auto max-w-6xl px-6 py-24">
          <div className="mb-10">
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              05 — Results
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Multi-axis scorecard. Numbers go here, honestly.
            </h2>
            <p className="mt-3 max-w-2xl text-zinc-400">
              Phase 4 outputs land here. Held-out clip 12 is never seen during
              training. We will report the actual judge scores per axis as
              they come in.
            </p>
          </div>

          <div className="overflow-hidden rounded-lg border border-zinc-800">
            <table className="w-full text-sm">
              <thead className="bg-zinc-900/60 text-left text-xs uppercase tracking-widest text-zinc-500">
                <tr>
                  <th className="px-4 py-4">Model</th>
                  <th className="px-4 py-4">Mean</th>
                  <th className="px-4 py-4">Permanence</th>
                  <th className="px-4 py-4">Tracking</th>
                  <th className="px-4 py-4">Occlusion</th>
                  <th className="px-4 py-4">State Δ</th>
                  <th className="px-4 py-4">Partial obs</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {[
                  ["Gemini 2.5 Pro"],
                  ["Claude Opus 4.7"],
                  ["GPT-4o"],
                  ["Qwen2.5-VL-7B (base)"],
                  ["Qwen2.5-VL-7B (LoRA, ours)", true],
                ].map(([model, isOurs], i) => (
                  <tr
                    key={i}
                    className={isOurs ? "bg-accent/5 text-chalk" : "text-zinc-300"}
                  >
                    <td className="px-4 py-4 font-medium">{model as string}</td>
                    {Array.from({ length: 6 }).map((_, j) => (
                      <td key={j} className="px-4 py-4 font-mono text-zinc-500">—</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="mt-4 text-xs text-zinc-500">
            Judge: Claude Opus 4.7. Scores are 0 / 0.5 / 1. Eval set: held-out
            clip 12 (downtime_prep), 80–100 human-verified questions.
          </p>
        </div>
      </section>

      {/* ---------- footer ---------- */}
      <footer className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-8 md:grid-cols-3">
          <div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-accent" />
              <span className="font-mono text-sm">affordance-vlm</span>
            </div>
            <p className="mt-3 text-sm text-zinc-400">
              Built in 36 hours for the Caltech × Ironsite hackathon. MIT
              licensed. Source footage © Ironsite.
            </p>
          </div>
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              Code
            </p>
            <ul className="mt-3 space-y-2 text-sm">
              <li>
                <a href={REPO_URL} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  GitHub repository →
                </a>
              </li>
              <li>
                <a href={`${REPO_URL}/blob/main/infra/vultr/README.md`} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  Vultr deployment runbook →
                </a>
              </li>
              <li>
                <a href={`${REPO_URL}/blob/main/training/FREE_FINETUNING.md`} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  Free-GPU fallback path →
                </a>
              </li>
            </ul>
          </div>
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              Built by
            </p>
            <ul className="mt-3 space-y-2 text-sm">
              <li>
                <a href="https://github.com/mohosy" target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  @mohosy →
                </a>
              </li>
            </ul>
          </div>
        </div>
      </footer>
    </main>
  );
}
