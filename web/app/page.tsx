import DemoCard from "@/components/DemoCard";

const REPO_URL = "https://github.com/mohosy/affordance-vlm";
const DATASET_URL = "https://huggingface.co/datasets/JiaaZ/HOVA-500K";
const PAPER_URL = "https://arxiv.org/abs/2505.11865";

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
            <span className="gradient-text">Frontier VLMs can&rsquo;t tell a robot</span>
            <br />
            <span className="text-chalk">where to grab a hammer.</span>
          </h1>
          <p className="mt-8 max-w-2xl text-lg text-zinc-400 md:text-xl">
            They recognize the hammer. They can describe it in detail. But ask
            them <span className="text-chalk">which end the gripper should close on</span> — and
            they fail. We close the gap with a 7B fine-tune on{" "}
            <a
              href={DATASET_URL}
              target="_blank"
              rel="noreferrer"
              className="underline decoration-zinc-700 underline-offset-4 hover:decoration-chalk"
            >
              HOVA-500K
            </a>
            .
          </p>
          <div className="mt-10 flex flex-wrap gap-3">
            <a
              href="#demo"
              className="rounded-md bg-chalk px-5 py-3 text-sm font-medium text-ink hover:bg-zinc-200"
            >
              Try the demo →
            </a>
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="rounded-md border border-zinc-700 px-5 py-3 text-sm hover:border-chalk"
            >
              View source on GitHub
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
                Object recognition is solved. Part-level affordance grounding isn&rsquo;t.
              </h2>
            </div>
          </div>
          <div className="md:col-span-2 space-y-6 text-lg text-zinc-300">
            <p>
              Modern VLMs are excellent at telling you{" "}
              <em>what an object is</em>. They are surprisingly bad at telling
              you <em>which part of it does what</em> — the kind of question a
              robot has to answer before it can act.
            </p>
            <p>
              In an embodied AI loop, a planner needs to know: which part to
              grasp, which part affords the action, where the geometry of the
              tool comes from. Frontier models trained on web text and image-caption
              pairs don&rsquo;t see enough of this signal to reliably ground it.
            </p>
            <p className="text-zinc-400">
              The bet: a small open model fine-tuned on a few thousand
              ground-truth-anchored Q&amp;A pairs can beat a 200B+ frontier
              model on this narrow task.
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
              Ground-truth-anchored Q&amp;A &nbsp;→&nbsp; LoRA &nbsp;→&nbsp; honest eval.
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-4">
            {[
              {
                tag: "Step 1",
                title: "Source the truth",
                body: (
                  <>
                    <a
                      href={DATASET_URL}
                      target="_blank"
                      rel="noreferrer"
                      className="underline decoration-zinc-700 underline-offset-4 hover:decoration-chalk"
                    >
                      HOVA-500K
                    </a>
                    {" "}provides 500K images with{" "}
                    <span className="text-chalk">
                      object + action + Gaussian affordance mask
                    </span>{" "}
                    annotations across 1,726 objects and 675 actions.
                  </>
                ),
              },
              {
                tag: "Step 2",
                title: "Generate grounded Q&A",
                body: (
                  <>
                    For each annotation, prompt Gemini 2.5 Pro with the image
                    plus the ground-truth label. Mask grounds the answer — the
                    model can&rsquo;t hallucinate the affordance location.
                  </>
                ),
              },
              {
                tag: "Step 3",
                title: "Self-consistency filter",
                body: (
                  <>
                    Re-prompt without the grounding hint. A judge call
                    discards pairs where the answers disagree. ~40% of raw
                    pairs survive.
                  </>
                ),
              },
              {
                tag: "Step 4",
                title: "LoRA + judged eval",
                body: (
                  <>
                    Fine-tune Qwen2.5-VL-7B with LoRA (rank 16, α 32). Compare
                    to frontier baselines on a hand-verified held-out set,
                    scored by Claude Opus 4.7 with a strict rubric.
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

      {/* ---------- demo ---------- */}
      <section id="demo" className="border-b border-zinc-800">
        <div className="mx-auto max-w-6xl px-6 py-24">
          <div className="mb-12">
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              03 — Demo
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Ask a part-level affordance question. See the gap.
            </h2>
            <p className="mt-3 max-w-2xl text-zinc-400">
              Upload an image of an object and ask which part to interact with
              and why. The frontier-model answers come from live API calls.
              The fine-tuned Qwen column lights up after Phase 4 of the
              hackathon.
            </p>
          </div>
          <DemoCard />
        </div>
      </section>

      {/* ---------- results ---------- */}
      <section id="results" className="border-b border-zinc-800 bg-zinc-950/30">
        <div className="mx-auto max-w-6xl px-6 py-24">
          <div className="mb-10">
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              04 — Results
            </p>
            <h2 className="mt-3 text-3xl font-semibold tracking-tight">
              Numbers go here. Honestly.
            </h2>
            <p className="mt-3 max-w-2xl text-zinc-400">
              Phase 4 outputs land here. We will report the actual judge scores
              from the held-out set as they come in — no cherry-picking, no
              re-running until the bar moves.
            </p>
          </div>

          <div className="overflow-hidden rounded-lg border border-zinc-800">
            <table className="w-full text-sm">
              <thead className="bg-zinc-900/60 text-left text-xs uppercase tracking-widest text-zinc-500">
                <tr>
                  <th className="px-6 py-4">Model</th>
                  <th className="px-6 py-4">Mean score</th>
                  <th className="px-6 py-4">Full credit</th>
                  <th className="px-6 py-4">Partial</th>
                  <th className="px-6 py-4">Wrong</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {[
                  ["Gemini 2.5 Pro", "—", "—", "—", "—"],
                  ["Claude Opus 4.7", "—", "—", "—", "—"],
                  ["GPT-4o", "—", "—", "—", "—"],
                  ["Qwen2.5-VL-7B (base)", "—", "—", "—", "—"],
                  ["Qwen2.5-VL-7B (LoRA, ours)", "—", "—", "—", "—", true],
                ].map(([model, ...rest], i) => {
                  const isOurs = rest[rest.length - 1] === true;
                  const cells = isOurs ? rest.slice(0, -1) : rest;
                  return (
                    <tr
                      key={i}
                      className={isOurs ? "bg-accent/5 text-chalk" : "text-zinc-300"}
                    >
                      <td className="px-6 py-4 font-medium">{model as string}</td>
                      {cells.map((c, j) => (
                        <td key={j} className="px-6 py-4 font-mono">
                          {c as string}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="mt-4 text-xs text-zinc-500">
            Judge: Claude Opus 4.7. Scores are 0 / 0.5 / 1 with a strict
            rubric for part-level correctness.
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
              licensed.
            </p>
          </div>
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
              Code + data
            </p>
            <ul className="mt-3 space-y-2 text-sm">
              <li>
                <a href={REPO_URL} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  GitHub repository →
                </a>
              </li>
              <li>
                <a href={DATASET_URL} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  HOVA-500K on HuggingFace →
                </a>
              </li>
              <li>
                <a href={PAPER_URL} target="_blank" rel="noreferrer" className="text-zinc-300 hover:text-chalk">
                  GLOVER++ paper →
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
