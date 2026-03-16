#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from recon3d.geometry import backproject_depth_to_world, voxel_downsample
from recon3d.io_utils import intrinsics_from_transforms, load_depth_mm, load_rgb, load_transforms, save_json


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Gaussian-like scene model from RGB-D frames")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/indoor_run/reconstruction"))
    parser.add_argument("--split", type=str, default="all", choices=["train", "all"])
    parser.add_argument("--depth-stride", type=int, default=3)
    parser.add_argument("--source-stride", type=int, default=1, help="Depth stride for source-view cache")
    parser.add_argument("--min-depth-m", type=float, default=0.3)
    parser.add_argument("--max-depth-m", type=float, default=12.0)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=220000)
    args = parser.parse_args()

    transforms, frames = load_transforms(args.dataset, split=args.split)
    intr = intrinsics_from_transforms(transforms)

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    source_points: list[np.ndarray] = []
    source_colors: list[np.ndarray] = []
    source_cam_positions: list[np.ndarray] = []
    source_frame_ids: list[str] = []

    for i, frame in enumerate(frames):
        rgb = load_rgb(args.dataset / frame["file_path"])
        depth_mm = load_depth_mm(args.dataset / frame["depth_file_path"])
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        pts, cols, scl = backproject_depth_to_world(
            depth_mm=depth_mm,
            c2w=c2w,
            fx=intr["fx"],
            fy=intr["fy"],
            cx=intr["cx"],
            cy=intr["cy"],
            rgb=rgb,
            stride=args.depth_stride,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )

        all_points.append(pts)
        all_colors.append(cols if cols is not None else np.zeros((0, 3), dtype=np.float32))
        all_scales.append(scl)

        if args.source_stride == args.depth_stride:
            src_pts = pts
            src_cols = cols if cols is not None else np.zeros((0, 3), dtype=np.float32)
        else:
            src_pts, src_cols, _ = backproject_depth_to_world(
                depth_mm=depth_mm,
                c2w=c2w,
                fx=intr["fx"],
                fy=intr["fy"],
                cx=intr["cx"],
                cy=intr["cy"],
                rgb=rgb,
                stride=args.source_stride,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
            )
            if src_cols is None:
                src_cols = np.zeros((0, 3), dtype=np.float32)

        source_points.append(src_pts.astype(np.float32))
        source_colors.append(src_cols.astype(np.float32))
        source_cam_positions.append(c2w[:3, 3].astype(np.float32))
        source_frame_ids.append(Path(frame["file_path"]).stem)

        if (i + 1) % 20 == 0 or i + 1 == len(frames):
            total = sum(len(x) for x in all_points)
            print(f"Processed frames: {i + 1}/{len(frames)} | collected points: {total}")

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    scales = np.concatenate(all_scales, axis=0).astype(np.float32)

    means, rgb, scl, conf = voxel_downsample(
        points=points,
        colors=colors,
        scales=scales,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "gaussians.npz",
        means=means,
        colors=rgb,
        scales=scl,
        confidence=conf,
    )

    offsets = [0]
    for pts in source_points:
        offsets.append(offsets[-1] + len(pts))
    source_points_concat = np.concatenate(source_points, axis=0).astype(np.float32)
    source_colors_concat = np.concatenate(source_colors, axis=0).astype(np.float32)
    np.savez_compressed(
        out_dir / "source_views.npz",
        points=source_points_concat,
        colors=source_colors_concat,
        offsets=np.array(offsets, dtype=np.int64),
        cam_positions=np.stack(source_cam_positions, axis=0).astype(np.float32),
        frame_ids=np.array(source_frame_ids),
    )

    write_ply(out_dir / "gaussians_preview.ply", means, rgb)

    summary = {
        "dataset": str(args.dataset),
        "split": args.split,
        "num_frames": len(frames),
        "raw_points": int(len(points)),
        "gaussian_points": int(len(means)),
        "depth_stride": args.depth_stride,
        "source_stride": args.source_stride,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "voxel_size": args.voxel_size,
    }
    save_json(out_dir / "reconstruction_summary.json", summary)

    print(f"Saved model: {out_dir / 'gaussians.npz'}")
    print(f"Saved source-view cache: {out_dir / 'source_views.npz'}")
    print(f"Saved preview point cloud: {out_dir / 'gaussians_preview.ply'}")
    print(f"Gaussian count: {len(means)}")


if __name__ == "__main__":
    main()
