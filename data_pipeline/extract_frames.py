"""
Sample frames from Ironsite construction MP4s at a configurable rate.

Default: 1 frame every 2 seconds. With ~120 minutes of footage across
6 videos at 5 fps, that yields ~3,588 frames.

Each frame is written as JPEG plus an entry to a manifest JSONL with the
source video, timestamp, frame index, sampling rate, activity tag (parsed
from the filename), and the pixel dimensions. This manifest is the input
to the Q&A generation step.

Usage:
    python data_pipeline/extract_frames.py \\
        --videos data/raw \\
        --out data/frames \\
        --manifest data/frames/manifest.jsonl \\
        --hz 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm


# Filename like "09_prep_standby_production_mp.mp4" -> activity tags.
# We just extract anything between the leading clip number and the trailing
# "_mp.mp4" suffix, splitting on underscores.
_CLIP_NAME_RE = re.compile(r"^(?P<clip>\d+)_(?P<tags>.+?)(?:_mp)?\.mp4$", re.IGNORECASE)


def parse_clip_metadata(path: Path) -> tuple[str, list[str]]:
    """Return (clip_id, [activity tags]) parsed from a video filename."""
    m = _CLIP_NAME_RE.match(path.name)
    if not m:
        return path.stem, []
    return m.group("clip"), m.group("tags").split("_")


def ffprobe_duration(video: Path) -> float:
    """Return the video duration in seconds (float)."""
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        text=True,
    ).strip()
    return float(out)


def ffprobe_resolution(video: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video),
        ],
        text=True,
    ).strip()
    w, h = out.split("x")
    return int(w), int(h)


def extract(
    video: Path,
    out_dir: Path,
    hz: float,
    jpeg_quality: int = 4,  # ffmpeg -q:v scale 2-31, lower is better
) -> list[dict]:
    """
    Extract frames from one video using ffmpeg's `fps` filter.
    Returns a list of manifest entries, one per saved frame.
    """
    clip_id, tags = parse_clip_metadata(video)
    duration = ffprobe_duration(video)
    width, height = ffprobe_resolution(video)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = out_dir / f"{video.stem}_%05d.jpg"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={hz}",
        "-q:v", str(jpeg_quality),
        str(pattern),
        "-y",
    ]
    subprocess.check_call(cmd)

    saved = sorted(out_dir.glob(f"{video.stem}_*.jpg"))
    entries: list[dict] = []
    interval = 1.0 / hz
    for i, frame_path in enumerate(saved):
        # Frame i corresponds to t = i * interval in the source clip
        # (fps filter starts at the first frame >= t=0).
        timestamp = i * interval
        entries.append({
            "frame_id": frame_path.stem,
            "frame_path": str(frame_path),
            "video": str(video),
            "video_filename": video.name,
            "clip_id": clip_id,
            "activity_tags": tags,
            "timestamp_sec": round(timestamp, 3),
            "video_duration_sec": round(duration, 3),
            "width": width,
            "height": height,
            "sample_hz": hz,
        })
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=Path("data/raw"),
                        help="Directory containing the source MP4s.")
    parser.add_argument("--out", type=Path, default=Path("data/frames"),
                        help="Where to write extracted frames.")
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/frames/manifest.jsonl"))
    parser.add_argument("--hz", type=float, default=0.5,
                        help="Sample rate in Hz (default 0.5 = 1 frame per 2 sec).")
    parser.add_argument("--quality", type=int, default=4,
                        help="ffmpeg JPEG quality (2-31, lower is better; default 4).")
    args = parser.parse_args()

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        print(f"[error] no MP4s under {args.videos}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] {len(videos)} videos, sampling at {args.hz} Hz "
          f"(1 frame per {1/args.hz:.1f} sec)")

    all_entries: list[dict] = []
    with jsonlines.open(args.manifest, mode="w") as writer:
        for v in tqdm(videos, desc="videos"):
            entries = extract(
                video=v,
                out_dir=args.out,
                hz=args.hz,
                jpeg_quality=args.quality,
            )
            for e in entries:
                writer.write(e)
            all_entries.extend(entries)
            print(f"  {v.name}: {len(entries)} frames")

    # Coverage stats
    by_clip = {}
    for e in all_entries:
        by_clip.setdefault(e["clip_id"], 0)
        by_clip[e["clip_id"]] += 1
    print(f"[done] {len(all_entries)} total frames across {len(by_clip)} clips")
    print("[done] manifest:", args.manifest)
    print(json.dumps(by_clip, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
