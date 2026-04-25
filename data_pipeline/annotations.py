"""
Unified annotation loader for HOVA-500K subsets.

The four sub-datasets have different schemas:

  HANDAL (230k train / 2k test) — household + industrial tools
    keys: img_path, mask_path, noun, points, width, height, gt_path
    Part name is encoded in mask_path: e.g. ".../000192_000000_handle.png" -> "handle"
    No action verb; we infer canonical actions from the part later.

  3doi (50k train) — 3D object interactions
    keys: img_name, bbox, affordance, object, width, height, gt_path
    Single affordance point (normalized coords); no part name; no action verb.

  ego4d (187k train / 2k test) — egocentric daily activity
    keys: img_path, frame_number, points, width, height, action, noun, gt_path
    Has action verb (sometimes parenthesized synonym list).

  epic100 (34k train / 2k test) — EPIC-Kitchens-100
    keys: img_path, narration_id, points, verb, noun, gt_path
    Has verb. width/height not provided.

This module normalizes all of them into a single Annotation dataclass.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


# Map HANDAL part names to canonical actions used in Q&A generation.
# The mask filename contains a part token (handle, head, button, ...).
# This is heuristic but anchored in standard tool affordance conventions.
HANDAL_PART_ACTIONS = {
    "handle": ["grasp", "hold", "lift"],
    "head": ["strike", "use", "press"],
    "blade": ["cut", "slice"],
    "button": ["press", "push"],
    "trigger": ["squeeze", "pull"],
    "knob": ["turn", "twist"],
    "lever": ["pull", "push"],
    "spout": ["pour", "dispense"],
    "lid": ["open", "close"],
    "neck": ["grasp", "hold"],
    "body": ["hold", "support"],
    "tip": ["touch", "apply"],
    "cap": ["unscrew", "remove"],
    "grip": ["grasp", "hold"],
}

# Strip parenthesized synonym lists used in ego4d/epic100 (e.g. "bag_(bag,_grocery,...)" -> "bag")
_SYNONYM_RE = re.compile(r"_\([^)]*\)")


def _clean_synonym(label: str) -> str:
    """Drop the parenthesized synonym list and underscore-clean the label."""
    if not label:
        return ""
    base = _SYNONYM_RE.sub("", label).strip("_ ")
    return base.replace("_", " ").strip()


def _extract_part_from_mask_path(mask_path: str) -> str | None:
    """
    HANDAL mask filenames look like 000192_000000_handle.png.
    Extract the trailing token before .png.
    """
    if not mask_path:
        return None
    name = Path(mask_path).stem  # 000192_000000_handle
    parts = name.split("_")
    if len(parts) >= 3 and parts[-1].isalpha():
        return parts[-1].lower()
    return None


@dataclass
class Annotation:
    """Normalized HOVA-500K annotation for downstream Q&A generation."""

    source: str                          # "handal" | "3doi" | "ego4d" | "epic100"
    image_relpath: str                   # path under the dataset root
    object_name: str                     # cleaned noun / object label
    part_name: str | None                # only for HANDAL
    action: str | None                   # verb / action (None for handal/3doi)
    affordance_points: list[tuple[float, float]]  # absolute pixel coords
    bbox: tuple[float, float, float, float] | None    # only for 3doi (normalized)
    image_size: tuple[int, int] | None   # (width, height) — None when missing
    gt_mask_relpath: str                 # GT_gaussian filename
    raw: dict = field(repr=False, default_factory=dict)


def _load_handal(entry: dict) -> Annotation:
    return Annotation(
        source="handal",
        image_relpath=entry["img_path"],
        object_name=_clean_synonym(entry["noun"]),
        part_name=_extract_part_from_mask_path(entry.get("mask_path", "")),
        action=None,
        affordance_points=[(float(x), float(y)) for x, y in entry.get("points", [])],
        bbox=None,
        image_size=(int(entry["width"]), int(entry["height"])),
        gt_mask_relpath=entry["gt_path"],
        raw=entry,
    )


def _load_3doi(entry: dict) -> Annotation:
    aff = entry.get("affordance")
    width = int(entry["width"])
    height = int(entry["height"])
    # 3doi affordance is normalized (frac of width/height)
    points: list[tuple[float, float]] = []
    if aff:
        points.append((float(aff[0]) * width, float(aff[1]) * height))
    bbox = entry.get("bbox")
    bbox_t = tuple(float(b) for b in bbox) if bbox else None
    return Annotation(
        source="3doi",
        image_relpath=entry["img_name"],
        object_name=_clean_synonym(entry["object"]),
        part_name=None,
        action=None,
        affordance_points=points,
        bbox=bbox_t,  # type: ignore[arg-type]
        image_size=(width, height),
        gt_mask_relpath=entry["gt_path"],
        raw=entry,
    )


def _load_ego4d(entry: dict) -> Annotation:
    width = int(entry.get("width", 0)) or None
    height = int(entry.get("height", 0)) or None
    return Annotation(
        source="ego4d",
        image_relpath=entry["img_path"],
        object_name=_clean_synonym(entry["noun"]),
        part_name=None,
        action=_clean_synonym(entry.get("action", "")) or None,
        affordance_points=[(float(x), float(y)) for x, y in entry.get("points", [])],
        bbox=None,
        image_size=(width, height) if (width and height) else None,
        gt_mask_relpath=entry["gt_path"],
        raw=entry,
    )


def _load_epic100(entry: dict) -> Annotation:
    return Annotation(
        source="epic100",
        image_relpath=entry["img_path"],
        object_name=_clean_synonym(entry["noun"]),
        part_name=None,
        action=_clean_synonym(entry.get("verb", "")) or None,
        affordance_points=[(float(x), float(y)) for x, y in entry.get("points", [])],
        bbox=None,
        image_size=None,  # not in schema
        gt_mask_relpath=entry["gt_path"],
        raw=entry,
    )


_LOADERS = {
    "handal": _load_handal,
    "3doi": _load_3doi,
    "ego4d": _load_ego4d,
    "epic100": _load_epic100,
}


def iter_annotations(annotations_root: Path, source: str, split: str = "train") -> Iterator[Annotation]:
    """Stream Annotation objects from one source/split JSON file."""
    if source not in _LOADERS:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(_LOADERS)}")
    path = Path(annotations_root) / split / f"{source}.json"
    if not path.exists():
        raise FileNotFoundError(f"annotation file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"unexpected annotation file shape (not a list): {path}")
    loader = _LOADERS[source]
    for entry in data:
        yield loader(entry)


def load_annotations(annotations_root: Path, source: str, split: str = "train") -> list[Annotation]:
    """Eager load (mostly for small smoke tests)."""
    return list(iter_annotations(annotations_root, source, split))


def affordance_centroid(ann: Annotation) -> tuple[float, float] | None:
    """Average affordance points to get a single 'where' coordinate."""
    if not ann.affordance_points:
        return None
    xs = [p[0] for p in ann.affordance_points]
    ys = [p[1] for p in ann.affordance_points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def describe_location(point: tuple[float, float], size: tuple[int, int] | None) -> str:
    """Convert a pixel coordinate into a coarse spatial description.

    Used as a hint to the Gemini grounding prompt so it does not have to guess
    where the affordance region sits in the frame.
    """
    if size is None or size[0] <= 0 or size[1] <= 0:
        return f"pixel ({int(point[0])}, {int(point[1])})"
    fx = point[0] / size[0]
    fy = point[1] / size[1]
    horiz = "left" if fx < 0.34 else "center" if fx < 0.67 else "right"
    vert = "top" if fy < 0.34 else "middle" if fy < 0.67 else "bottom"
    return f"{vert}-{horiz} (~{fx:.0%} from left, {fy:.0%} from top)"


def canonical_actions_for_handal_part(part: str | None) -> list[str]:
    if not part:
        return []
    return HANDAL_PART_ACTIONS.get(part.lower(), [])


def resolve_image_path(ann: Annotation, image_roots: dict[str, Path]) -> Path | None:
    """
    Map an Annotation to its on-disk image. ``image_roots`` is keyed by source
    name (e.g. ``{"3doi": Path("data/hova/3doi/images")}``).

    Returns None if the file does not exist on disk yet.
    """
    root = image_roots.get(ann.source)
    if not root:
        return None
    candidate = Path(root) / ann.image_relpath
    return candidate if candidate.exists() else None
