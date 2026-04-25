"""
Verify the data pipeline end-to-end without making any paid API calls.

Exercises:
  1. All imports succeed.
  2. Annotation JSON files exist and load.
  3. Annotation loader produces non-empty, well-formed records on all 4 sources.
  4. Image paths resolve on disk for at least one downloaded subset (3doi).
  5. Prompt assembly produces the expected shape for both Q&A generation and
     the quality filter.
  6. annotation_to_record produces JSON-serializable output.
  7. Eval helpers (judge prompts, baseline adapters) construct without errors.
  8. Gemini / Anthropic clients can be instantiated when keys ARE present
     (skipped otherwise — does not fail the run).

Exit code:
  0  all checks passed
  >0 number of failures (each printed inline)

Run:
  python scripts/verify_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m~\033[0m"


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def run(self, name: str, fn) -> None:
        try:
            fn()
        except _Skip as e:
            self.skipped += 1
            print(f"  {SKIP} {name} — {e}")
        except AssertionError as e:
            self.failed += 1
            print(f"  {FAIL} {name}")
            print(f"    AssertionError: {e}")
        except Exception:
            self.failed += 1
            print(f"  {FAIL} {name}")
            traceback.print_exc()
        else:
            self.passed += 1
            print(f"  {PASS} {name}")

    def report(self) -> int:
        total = self.passed + self.failed + self.skipped
        print()
        print(f"  {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped")
        return self.failed


class _Skip(Exception):
    """Raise from a check to mark it skipped (not failed)."""


# ------------------------------------------------------------------ checks ---


def check_imports() -> None:
    # legacy HOVA pipeline
    import data_pipeline.annotations  # noqa: F401
    import data_pipeline.gemini_client  # noqa: F401
    import data_pipeline.generate_qa  # noqa: F401
    import data_pipeline.quality_filter  # noqa: F401
    import data_pipeline.run_pipeline  # noqa: F401
    import eval.judge  # noqa: F401
    import eval.run_baselines  # noqa: F401
    import eval.build_heldout  # noqa: F401
    # Big Swing temporal pipeline
    import data_pipeline.extract_frames  # noqa: F401
    import data_pipeline.build_sequences  # noqa: F401
    import data_pipeline.generate_qa_temporal  # noqa: F401
    import eval.run_baselines_temporal  # noqa: F401
    import eval.build_heldout_temporal  # noqa: F401


def check_annotations_present() -> None:
    annos = ROOT / "data" / "hova" / "annotations"
    if not annos.exists():
        raise _Skip(f"{annos} not downloaded; run "
                    "`python data_pipeline/download_hova.py --subsets annotations`")
    expected = [
        "train/handal.json", "train/3doi.json", "train/ego4d.json", "train/epic100.json",
        "test/handal.json", "test/ego4d.json", "test/epic100.json",
    ]
    for relp in expected:
        p = annos / relp
        assert p.exists(), f"missing {p}"
        # file is non-trivial JSON list
        with p.open() as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data) > 0, f"empty list in {p}"


def check_annotation_loader() -> None:
    from data_pipeline.annotations import iter_annotations, affordance_centroid
    annos = ROOT / "data" / "hova" / "annotations"
    if not annos.exists():
        raise _Skip("annotations not downloaded")
    counts = {}
    for src in ("handal", "3doi", "ego4d", "epic100"):
        n = 0
        for ann in iter_annotations(annos, src, "train"):
            assert ann.source == src
            assert ann.object_name, f"empty object_name on {src}"
            assert ann.gt_mask_relpath, f"empty gt_mask on {src}"
            n += 1
            if n >= 5:  # spot check
                break
        counts[src] = n
    assert all(v >= 1 for v in counts.values()), f"loader produced empty for: {counts}"

    # centroid & description on the FIRST handal entry
    ann = next(iter_annotations(annos, "handal", "train"))
    centroid = affordance_centroid(ann)
    assert centroid is not None, "centroid missing"
    cx, cy = centroid
    assert 0 < cx < ann.image_size[0], "centroid x out of bounds"
    assert 0 < cy < ann.image_size[1], "centroid y out of bounds"


def check_image_resolution_3doi() -> None:
    from data_pipeline.annotations import iter_annotations
    annos = ROOT / "data" / "hova" / "annotations"
    img_root = ROOT / "data" / "hova" / "3doi" / "images"
    if not img_root.exists():
        raise _Skip(f"{img_root} not downloaded; run "
                    "`python data_pipeline/download_hova.py --subsets 3doi`")
    n_resolved = 0
    n_total = 0
    for ann in iter_annotations(annos, "3doi", "train"):
        n_total += 1
        if (img_root / ann.image_relpath).exists():
            n_resolved += 1
        if n_total >= 100:
            break
    assert n_resolved == n_total, f"only {n_resolved}/{n_total} images on disk"


def check_prompt_assembly() -> None:
    from data_pipeline.annotations import iter_annotations
    from data_pipeline.generate_qa import _build_user_prompt, SYSTEM_INSTRUCTION
    annos = ROOT / "data" / "hova" / "annotations"
    if not annos.exists():
        raise _Skip("annotations not downloaded")

    handal_ann = next(iter_annotations(annos, "handal", "train"))
    prompt = _build_user_prompt(handal_ann, n_pairs=3)
    assert "GROUND TRUTH:" in prompt
    assert handal_ann.object_name in prompt
    assert "Generate exactly 3 Q&A pairs." in prompt
    if handal_ann.part_name:
        assert handal_ann.part_name in prompt

    epic_ann = next(iter_annotations(annos, "epic100", "train"))
    epic_prompt = _build_user_prompt(epic_ann, n_pairs=2)
    if epic_ann.action:
        assert epic_ann.action in epic_prompt or "canonical action:" in epic_prompt

    assert "robot controller" in SYSTEM_INSTRUCTION
    assert "json" in SYSTEM_INSTRUCTION.lower() or "JSON" in SYSTEM_INSTRUCTION


def check_serialization() -> None:
    from data_pipeline.annotations import iter_annotations
    from data_pipeline.generate_qa import annotation_to_record
    annos = ROOT / "data" / "hova" / "annotations"
    if not annos.exists():
        raise _Skip("annotations not downloaded")
    for src in ("handal", "3doi", "ego4d", "epic100"):
        ann = next(iter_annotations(annos, src, "train"))
        rec = annotation_to_record(ann)
        json.dumps(rec)  # raises if not JSON-safe
        assert "raw" not in rec, f"raw field leaked on {src}"
        assert rec["source"] == src


def check_eval_helpers() -> None:
    from eval.judge import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE, JUDGE_MODEL
    assert "ground" in JUDGE_SYSTEM.lower()
    assert "{question}" in JUDGE_USER_TEMPLATE
    assert "{ground_truth}" in JUDGE_USER_TEMPLATE
    assert "{candidate}" in JUDGE_USER_TEMPLATE
    assert JUDGE_MODEL.startswith("claude-")

    from eval.run_baselines import ADAPTERS
    expected = {"gemini-2.5-pro", "claude-opus-4-7", "gpt-4o", "qwen-base", "qwen-finetuned"}
    assert expected.issubset(ADAPTERS.keys()), f"missing adapters: {expected - ADAPTERS.keys()}"

    from eval.run_baselines_temporal import ADAPTERS as TADAPTERS
    assert expected.issubset(TADAPTERS.keys()), \
        f"temporal adapters missing: {expected - TADAPTERS.keys()}"


def check_videos_present() -> None:
    raw = ROOT / "data" / "raw"
    if not raw.exists():
        raise _Skip(f"{raw} not present (Ironsite videos)")
    videos = list(raw.glob("*.mp4"))
    if len(videos) == 0:
        raise _Skip("no .mp4 files under data/raw")
    assert len(videos) >= 1, f"expected >= 1 mp4, got {len(videos)}"


def check_frames_extracted() -> None:
    manifest = ROOT / "data" / "frames" / "manifest.jsonl"
    if not manifest.exists():
        raise _Skip(f"frame manifest not generated yet: {manifest}")
    n = 0
    with manifest.open() as f:
        for _ in f:
            n += 1
    assert n >= 100, f"expected many frames, got {n}"


def check_sequences_built() -> None:
    seq_path = ROOT / "data" / "sequences" / "sequences.jsonl"
    if not seq_path.exists():
        raise _Skip(f"sequences not built yet: {seq_path}")
    import json as _json
    with seq_path.open() as f:
        first = _json.loads(f.readline())
    assert "frames" in first and len(first["frames"]) >= 2, "sequence has <2 frames"
    assert "sequence_id" in first


def check_temporal_prompt_assembly() -> None:
    """Build a fake sequence and exercise the prompt building path."""
    seq_path = ROOT / "data" / "sequences" / "sequences.jsonl"
    if not seq_path.exists():
        raise _Skip("sequences not built yet")
    import json as _json
    from data_pipeline.generate_qa_temporal import _build_user_text, SYSTEM_INSTRUCTION
    with seq_path.open() as f:
        seq = _json.loads(f.readline())
    text = _build_user_text(seq, n_pairs=3)
    assert "chronological frames" in text
    assert "construction body cam" in text
    assert "EXACTLY 3" in text or "exactly 3" in text.lower(), \
        "user text must specify exact pair count"
    assert "pairs" in text, "user text must request the JSON wrapper field"
    assert "object_permanence" in SYSTEM_INSTRUCTION
    assert "tracking" in SYSTEM_INSTRUCTION
    assert "occlusion" in SYSTEM_INSTRUCTION


def check_training_config() -> None:
    from pathlib import Path as _P
    cfg_path = _P("training/configs/lora.yaml")
    if not cfg_path.exists():
        raise _Skip(f"{cfg_path} missing")
    from training.finetune_qwen import load_config
    cfg = load_config(cfg_path)
    assert "Qwen2.5-VL" in cfg.base_model, f"unexpected base model: {cfg.base_model}"
    assert cfg.lora_r > 0
    assert cfg.train_file.endswith(".jsonl")


def check_gemini_client_constructable() -> None:
    from data_pipeline.gemini_client import GeminiClient
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise _Skip("no GOOGLE_API_KEY in env (expected — pre-key verification)")
    GeminiClient()  # will raise if key invalid format


def check_anthropic_client_constructable() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise _Skip("no ANTHROPIC_API_KEY in env (expected — pre-key verification)")
    from eval.judge import Judge
    Judge()


def check_dotenv_loads() -> None:
    """Sanity check that .env is wired through dotenv where expected."""
    from dotenv import load_dotenv
    load_dotenv()  # smoke; never fails


def check_repo_hygiene() -> None:
    # No accidental .env tracked
    import subprocess
    out = subprocess.check_output(["git", "ls-files", ".env"], cwd=ROOT, text=True).strip()
    assert out == "", f".env is tracked! ({out!r})"

    # data/hova is NOT tracked
    out = subprocess.check_output(["git", "ls-files", "data/hova/"], cwd=ROOT, text=True).strip()
    assert out == "", f"data/hova content is tracked!\n{out}"


# ---------------------------------------------------------- entry point ----

def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()

    print("\n=== affordance-vlm pipeline verification ===\n")

    c = Checks()
    print("[1] imports + module-level construction")
    c.run("all modules import", check_imports)
    c.run("dotenv loads", check_dotenv_loads)

    print("\n[2] data on disk")
    c.run("HOVA annotation JSONs present + non-empty", check_annotations_present)
    c.run("3doi images resolve (100/100 spot check)", check_image_resolution_3doi)

    print("\n[3] annotation loader")
    c.run("loader produces records on all 4 sources", check_annotation_loader)
    c.run("annotation_to_record JSON-safe", check_serialization)

    print("\n[4] prompt assembly")
    c.run("Q&A generation prompts contain ground truth", check_prompt_assembly)
    c.run("eval helpers (judge + baselines) wired", check_eval_helpers)

    print("\n[5] API client construction (skipped without keys)")
    c.run("GeminiClient", check_gemini_client_constructable)
    c.run("Judge (Anthropic)", check_anthropic_client_constructable)

    print("\n[6] Big Swing pipeline (temporal)")
    c.run("Ironsite videos present", check_videos_present)
    c.run("frames extracted (manifest.jsonl)", check_frames_extracted)
    c.run("sequences built", check_sequences_built)
    c.run("temporal prompt assembly", check_temporal_prompt_assembly)
    c.run("training config loads", check_training_config)

    print("\n[7] repo hygiene")
    c.run("no .env tracked + data/hova not tracked", check_repo_hygiene)

    rc = c.report()
    return rc


if __name__ == "__main__":
    sys.exit(main())
