"""
Download HOVA-500K subsets from HuggingFace.

Dataset: https://huggingface.co/datasets/JiaaZ/HOVA-500K
License: MIT (per TeleeMa/GLOVER repo)

The dataset is split across:
  - annotations.tar.gz (~24 MB)        labels + masks index
  - 3doi.tar.gz        (~2.9 GB)       3D object interaction images + GT_gaussian masks
  - HANDAL/part_*      (multi-part)    household/industrial tool images + masks
  - Ego4D/part_*       (multi-part)    Ego4D frames + masks
  - epic-100/part_*    (multi-part)    EPIC-Kitchens-100 frames + masks

For the hackathon we usually want annotations + 3doi (and optionally HANDAL).

Usage:
    python data_pipeline/download_hova.py --subsets annotations
    python data_pipeline/download_hova.py --subsets annotations,3doi --out data/hova/
    python data_pipeline/download_hova.py --subsets HANDAL --out data/hova/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

REPO_ID = "JiaaZ/HOVA-500K"
REPO_TYPE = "dataset"

# Top-level files vs. multi-part directory subsets
SINGLE_FILES = {
    "annotations": "annotations.tar.gz",
    "3doi": "3doi.tar.gz",
}
MULTIPART_DIRS = {
    "HANDAL": "HANDAL",
    "Ego4D": "Ego4D",
    "epic-100": "epic-100",
}


def download_single_file(name: str, dest: Path) -> Path:
    """Download a single .tar.gz from the HF repo."""
    filename = SINGLE_FILES[name]
    print(f"[download] {filename} -> {dest}")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(dest),
    )
    return Path(local_path)


def download_multipart(name: str, dest: Path) -> Path:
    """Download all part_* files for a multi-part subset."""
    folder = MULTIPART_DIRS[name]
    target = dest / folder
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] {folder}/part_* -> {target}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=[f"{folder}/*"],
        local_dir=str(dest),
    )
    return target


def merge_and_extract_multipart(name: str, dest: Path) -> Path:
    """Concatenate part_* files and extract the resulting tar.gz."""
    folder = dest / name
    parts = sorted(folder.glob("part_*"))
    if not parts:
        raise FileNotFoundError(f"No part_* files found in {folder}")
    merged = dest / f"{name}.tar.gz"
    print(f"[merge] {len(parts)} parts -> {merged}")
    with merged.open("wb") as out:
        for part in parts:
            with part.open("rb") as f:
                shutil.copyfileobj(f, out)
    return merged


def extract_tar(tar_path: Path, dest: Path) -> None:
    print(f"[extract] {tar_path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--subsets",
        type=str,
        default="annotations",
        help="Comma-separated subsets: annotations,3doi,HANDAL,Ego4D,epic-100",
    )
    parser.add_argument("--out", type=Path, default=Path("data/hova"))
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction; just download archives",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    valid = set(SINGLE_FILES) | set(MULTIPART_DIRS)
    invalid = [s for s in subsets if s not in valid]
    if invalid:
        print(f"[error] unknown subsets: {invalid}; valid options: {sorted(valid)}", file=sys.stderr)
        return 1

    # Sanity-check we can hit the repo before any heavy download.
    api = HfApi()
    try:
        info = api.dataset_info(REPO_ID)
        print(f"[ok] connected to {REPO_ID}, {len(info.siblings)} files visible")
    except Exception as e:
        print(f"[error] could not reach {REPO_ID}: {e}", file=sys.stderr)
        print("        make sure you have network access; if the repo is gated,", file=sys.stderr)
        print("        run: huggingface-cli login", file=sys.stderr)
        return 2

    archives: list[tuple[str, Path]] = []
    for name in subsets:
        if name in SINGLE_FILES:
            tar = download_single_file(name, args.out)
            archives.append((name, tar))
        else:
            download_multipart(name, args.out)
            tar = merge_and_extract_multipart(name, args.out)
            archives.append((name, tar))

    if args.no_extract:
        print("[done] archives downloaded; extraction skipped")
        return 0

    for name, tar in archives:
        extract_tar(tar, args.out)

    print(f"[done] subsets ready under {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
