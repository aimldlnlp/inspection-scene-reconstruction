#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from recon3d.io_utils import load_rgb, load_transforms, save_json
from recon3d.metrics import mae, psnr, ssim
from recon3d.render import render_scene_splats


def chamfer_distance_l2(pred_points: np.ndarray, ref_points: np.ndarray, max_samples: int = 50000) -> float:
    if len(pred_points) == 0 or len(ref_points) == 0:
        return float("nan")

    rng = np.random.default_rng(123)
    if len(pred_points) > max_samples:
        pred_points = pred_points[rng.choice(len(pred_points), size=max_samples, replace=False)]
    if len(ref_points) > max_samples:
        ref_points = ref_points[rng.choice(len(ref_points), size=max_samples, replace=False)]

    tree_ref = cKDTree(ref_points)
    tree_pred = cKDTree(pred_points)
    d_pr, _ = tree_ref.query(pred_points, k=1)
    d_rp, _ = tree_pred.query(ref_points, k=1)
    return float(np.mean(d_pr**2) + np.mean(d_rp**2))


def select_nearest_source_points(
    target_c2w: np.ndarray,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    source_offsets: np.ndarray,
    source_cam_positions: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_pos = target_c2w[:3, 3]
    d = np.linalg.norm(source_cam_positions - target_pos[None, :], axis=1)
    ids = np.argsort(d)[: max(1, min(k, len(source_cam_positions)))]

    chunks_pts = []
    chunks_col = []
    for i in ids:
        a = int(source_offsets[i])
        b = int(source_offsets[i + 1])
        chunks_pts.append(source_points[a:b])
        chunks_col.append(source_colors[a:b])

    if not chunks_pts:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    return np.concatenate(chunks_pts, axis=0), np.concatenate(chunks_col, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed scene on held-out views")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to gaussians.npz")
    parser.add_argument("--source-cache", type=Path, default=None, help="Path to source_views.npz")
    parser.add_argument("--render-mode", type=str, default="nearest_views", choices=["nearest_views", "global"])
    parser.add_argument("--source-k", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["test", "all"])
    parser.add_argument("--out", type=Path, default=Path("outputs/indoor_run/eval"))
    parser.add_argument("--hole-blend-distance", type=float, default=0.0)
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    args = parser.parse_args()

    scene = np.load(args.model)
    global_means = scene["means"].astype(np.float32)
    global_colors = scene["colors"].astype(np.float32)

    source_points = None
    source_colors = None
    source_offsets = None
    source_cam_positions = None

    source_cache_path = args.source_cache
    if source_cache_path is None:
        source_cache_path = args.model.parent / "source_views.npz"

    if args.render_mode == "nearest_views":
        if not source_cache_path.exists():
            raise FileNotFoundError(
                f"source cache not found at {source_cache_path}. "
                "Run reconstruct_gaussian_scene.py first or set --render-mode global."
            )
        source = np.load(source_cache_path)
        source_points = source["points"].astype(np.float32)
        source_colors = source["colors"].astype(np.float32)
        source_offsets = source["offsets"].astype(np.int64)
        source_cam_positions = source["cam_positions"].astype(np.float32)

    transforms, frames = load_transforms(args.dataset, split=args.split)
    width = int(transforms["w"])
    height = int(transforms["h"])
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "metrics_per_frame.csv"

    rows = []
    for i, frame in enumerate(frames):
        gt = load_rgb(args.dataset / frame["file_path"])
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        if args.render_mode == "nearest_views":
            means, colors = select_nearest_source_points(
                target_c2w=c2w,
                source_points=source_points,
                source_colors=source_colors,
                source_offsets=source_offsets,
                source_cam_positions=source_cam_positions,
                k=args.source_k,
            )
        else:
            means, colors = global_means, global_colors

        pred, _, coverage = render_scene_splats(
            means=means,
            colors=colors,
            c2w=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            hole_blend_distance=args.hole_blend_distance,
            blur_sigma=args.blur_sigma,
        )

        row = {
            "frame": Path(frame["file_path"]).name,
            "psnr": psnr(gt, pred),
            "ssim": ssim(gt, pred),
            "mae": mae(gt, pred),
            "coverage": float(coverage),
        }
        rows.append(row)

        if (i + 1) % 10 == 0 or i + 1 == len(frames):
            print(f"Evaluated {i + 1}/{len(frames)} views")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "psnr", "ssim", "mae", "coverage"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "render_mode": args.render_mode,
        "source_k": args.source_k,
        "num_frames": len(rows),
        "psnr_mean": float(np.mean([r["psnr"] for r in rows])),
        "psnr_std": float(np.std([r["psnr"] for r in rows])),
        "ssim_mean": float(np.mean([r["ssim"] for r in rows])),
        "ssim_std": float(np.std([r["ssim"] for r in rows])),
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "coverage_mean": float(np.mean([r["coverage"] for r in rows])),
    }

    ref_cloud_path = args.dataset / "scene_reference.npz"
    if ref_cloud_path.exists():
        ref = np.load(ref_cloud_path)
        chamfer_l2 = chamfer_distance_l2(global_means, ref["points"].astype(np.float32))
        summary["chamfer_l2"] = chamfer_l2

    save_json(args.out / "summary.json", summary)

    print(f"Saved per-frame metrics: {csv_path}")
    print(f"Saved summary: {args.out / 'summary.json'}")
    print(
        "Summary -> "
        f"PSNR: {summary['psnr_mean']:.3f}, "
        f"SSIM: {summary['ssim_mean']:.4f}, "
        f"MAE: {summary['mae_mean']:.3f}, "
        f"Coverage: {summary['coverage_mean']:.4f}"
    )
    if "chamfer_l2" in summary:
        print(f"Chamfer-L2: {summary['chamfer_l2']:.6f}")


if __name__ == "__main__":
    main()
