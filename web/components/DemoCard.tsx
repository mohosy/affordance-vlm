"use client";

import { useState, useRef } from "react";

type ModelId = "gemini" | "claude" | "gpt4o" | "qwen-finetuned";

type Result =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ok"; answer: string }
  | { status: "missing_key"; key: string }
  | { status: "not_ready"; reason: string }
  | { status: "error"; message: string };

const MODELS: { id: ModelId; label: string; sub: string }[] = [
  { id: "gemini", label: "Gemini 2.5 Pro", sub: "Google · API" },
  { id: "claude", label: "Claude Opus 4.7", sub: "Anthropic · API" },
  { id: "gpt4o", label: "GPT-4o", sub: "OpenAI · API" },
  {
    id: "qwen-finetuned",
    label: "Qwen2.5-VL-7B + LoRA",
    sub: "ours · HF Spaces (Phase 4)",
  },
];

const DEFAULT_QUESTION =
  "Which part of this object should the gripper close on, and what does that part afford?";

export default function DemoCard() {
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [results, setResults] = useState<Record<ModelId, Result>>({
    gemini: { status: "idle" },
    claude: { status: "idle" },
    gpt4o: { status: "idle" },
    "qwen-finetuned": {
      status: "not_ready",
      reason: "Awaiting Phase 4 fine-tune. Demo lights up after deploy.",
    },
  });
  const fileInput = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => setImageDataUrl(reader.result as string);
    reader.readAsDataURL(file);
  };

  const runAll = async () => {
    if (!imageDataUrl) return;
    const targets: ModelId[] = ["gemini", "claude", "gpt4o"];
    setResults((r) => ({
      ...r,
      ...Object.fromEntries(targets.map((m) => [m, { status: "loading" }])),
    } as Record<ModelId, Result>));

    await Promise.all(
      targets.map(async (model) => {
        try {
          const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ model, image: imageDataUrl, question }),
          });
          const data = await res.json();
          if (!res.ok) {
            if (data?.code === "missing_key") {
              setResults((r) => ({
                ...r,
                [model]: { status: "missing_key", key: data.key },
              }));
            } else {
              setResults((r) => ({
                ...r,
                [model]: {
                  status: "error",
                  message: data?.error || `HTTP ${res.status}`,
                },
              }));
            }
            return;
          }
          setResults((r) => ({
            ...r,
            [model]: { status: "ok", answer: data.answer || "" },
          }));
        } catch (e) {
          setResults((r) => ({
            ...r,
            [model]: { status: "error", message: (e as Error).message },
          }));
        }
      })
    );
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      {/* ---------- input column ---------- */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-6">
        <p className="font-mono text-xs uppercase tracking-widest text-zinc-500">
          Input
        </p>

        <div className="mt-4">
          {imageDataUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={imageDataUrl}
              alt="upload"
              className="h-64 w-full rounded-md border border-zinc-800 object-contain bg-zinc-950"
            />
          ) : (
            <div
              onClick={() => fileInput.current?.click()}
              className="flex h-64 cursor-pointer items-center justify-center rounded-md border border-dashed border-zinc-700 bg-zinc-950 text-sm text-zinc-500 hover:border-zinc-500 hover:text-zinc-300"
            >
              Click to upload an image (jpg / png)
            </div>
          )}
          <input
            ref={fileInput}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </div>

        <label className="mt-6 block">
          <span className="font-mono text-xs uppercase tracking-widest text-zinc-500">
            Question
          </span>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            className="mt-2 w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm focus:border-zinc-500 focus:outline-none"
          />
        </label>

        <button
          onClick={runAll}
          disabled={!imageDataUrl || !question.trim()}
          className="mt-4 w-full rounded-md bg-chalk px-4 py-3 text-sm font-medium text-ink hover:bg-zinc-200 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
        >
          Run on all frontier models
        </button>

        {imageDataUrl && (
          <button
            onClick={() => {
              setImageDataUrl(null);
              setResults({
                gemini: { status: "idle" },
                claude: { status: "idle" },
                gpt4o: { status: "idle" },
                "qwen-finetuned": {
                  status: "not_ready",
                  reason:
                    "Awaiting Phase 4 fine-tune. Demo lights up after deploy.",
                },
              });
            }}
            className="mt-2 w-full text-xs text-zinc-500 hover:text-zinc-300"
          >
            clear
          </button>
        )}
      </div>

      {/* ---------- results column ---------- */}
      <div className="space-y-4">
        {MODELS.map((m) => (
          <ModelResult
            key={m.id}
            label={m.label}
            sub={m.sub}
            highlighted={m.id === "qwen-finetuned"}
            result={results[m.id]}
          />
        ))}
      </div>
    </div>
  );
}

function ModelResult({
  label,
  sub,
  highlighted,
  result,
}: {
  label: string;
  sub: string;
  highlighted?: boolean;
  result: Result;
}) {
  return (
    <div
      className={
        "rounded-lg border p-5 " +
        (highlighted
          ? "border-accent/50 bg-accent/5"
          : "border-zinc-800 bg-zinc-900/30")
      }
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">{label}</p>
          <p className="font-mono text-xs text-zinc-500">{sub}</p>
        </div>
        <StatusPill result={result} />
      </div>
      <div className="mt-3 min-h-[40px] text-sm text-zinc-300">
        <Body result={result} />
      </div>
    </div>
  );
}

function StatusPill({ result }: { result: Result }) {
  const map: Record<Result["status"], { label: string; cls: string }> = {
    idle: { label: "idle", cls: "border-zinc-800 text-zinc-500" },
    loading: { label: "running…", cls: "border-accent/40 text-accent" },
    ok: { label: "ok", cls: "border-emerald-700 text-emerald-400" },
    missing_key: {
      label: "needs api key",
      cls: "border-amber-700 text-amber-400",
    },
    not_ready: { label: "phase 4", cls: "border-zinc-700 text-zinc-400" },
    error: { label: "error", cls: "border-red-700 text-red-400" },
  };
  const s = map[result.status];
  return (
    <span
      className={
        "rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest " +
        s.cls
      }
    >
      {s.label}
    </span>
  );
}

function Body({ result }: { result: Result }) {
  switch (result.status) {
    case "idle":
      return <span className="text-zinc-500">Upload an image and click run.</span>;
    case "loading":
      return <span className="text-zinc-400">Calling API…</span>;
    case "ok":
      return <p className="whitespace-pre-wrap text-zinc-200">{result.answer}</p>;
    case "missing_key":
      return (
        <span className="text-amber-300">
          Set <code className="font-mono">{result.key}</code> in Vercel
          environment variables to enable this model.
        </span>
      );
    case "not_ready":
      return <span className="text-zinc-500">{result.reason}</span>;
    case "error":
      return <span className="text-red-300">Error: {result.message}</span>;
  }
}
