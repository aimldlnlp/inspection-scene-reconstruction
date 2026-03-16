from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path | str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rgb(path: Path | str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_depth_mm(path: Path | str) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.uint16)


def save_rgb(path: Path | str, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def write_gif(
    frame_paths: list[Path],
    output_path: Path,
    fps: int = 12,
    max_frames: int = 120,
) -> int:
    if not frame_paths:
        return 0
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    total = len(frame_paths)
    step = max(1, int(np.ceil(total / max_frames)))
    selected = frame_paths[::step]

    frames = [Image.open(p).convert("RGB") for p in selected]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=False,
    )
    return len(frames)


def load_transforms(dataset_dir: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all")
    payload = load_json(dataset_dir / f"transforms_{split}.json")
    frames = payload["frames"]
    return payload, frames


def intrinsics_from_transforms(transforms: dict[str, Any]) -> dict[str, float]:
    return {
        "w": float(transforms["w"]),
        "h": float(transforms["h"]),
        "fx": float(transforms["fl_x"]),
        "fy": float(transforms["fl_y"]),
        "cx": float(transforms["cx"]),
        "cy": float(transforms["cy"]),
    }
