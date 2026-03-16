#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from recon3d.io_utils import load_rgb, load_transforms, save_rgb, write_gif
from recon3d.render import render_scene_splats


def make_side_by_side(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w, _ = gt.shape
    canvas = Image.new("RGB", (w * 2, h), (0, 0, 0))
    canvas.paste(Image.fromarray(gt), (0, 0))
    canvas.paste(Image.fromarray(pred), (w, 0))
    return np.array(canvas, dtype=np.uint8)


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
    parser = argparse.ArgumentParser(description="Render novel views from reconstructed scene")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to gaussians.npz")
    parser.add_argument("--source-cache", type=Path, default=None, help="Path to source_views.npz")
    parser.add_argument("--render-mode", type=str, default="nearest_views", choices=["nearest_views", "global"])
    parser.add_argument("--source-k", type=int, default=4, help="Number of nearest source views for rendering")
    parser.add_argument("--split", type=str, default="test", choices=["test", "all"])
    parser.add_argument("--out", type=Path, default=Path("outputs/indoor_run/renders"))
    parser.add_argument("--save-gt", action="store_true", help="Save ground-truth PNG copies")
    parser.add_argument("--save-comparison", action="store_true", help="Save side-by-side GT vs prediction")
    parser.add_argument("--export-gif", action="store_true")
    parser.add_argument("--gif-name", type=str, default="novel_views.gif")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=120)
    parser.add_argument("--hole-blend-distance", type=float, default=0.0)
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    args = parser.parse_args()

    global_scene = np.load(args.model)
    global_means = global_scene["means"].astype(np.float32)
    global_colors = global_scene["colors"].astype(np.float32)

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

    pred_dir = args.out / "pred"
    gt_dir = args.out / "gt"
    cmp_dir = args.out / "comparison"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)
    if args.save_comparison:
        cmp_dir.mkdir(parents=True, exist_ok=True)

    pred_paths: list[Path] = []
    coverages = []

    for i, frame in enumerate(frames):
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

        stem = Path(frame["file_path"]).stem
        pred_path = pred_dir / f"{stem}.png"
        save_rgb(pred_path, pred)
        pred_paths.append(pred_path)
        coverages.append(coverage)

        if args.save_gt or args.save_comparison:
            gt = load_rgb(args.dataset / frame["file_path"])
            if args.save_gt:
                save_rgb(gt_dir / f"{stem}.png", gt)
            if args.save_comparison:
                side = make_side_by_side(gt, pred)
                save_rgb(cmp_dir / f"{stem}.png", side)

        if (i + 1) % 10 == 0 or i + 1 == len(frames):
            print(f"Rendered {i + 1}/{len(frames)} views")

    if args.export_gif:
        count = write_gif(
            frame_paths=pred_paths,
            output_path=args.out / args.gif_name,
            fps=args.gif_fps,
            max_frames=args.gif_max_frames,
        )
        print(f"Saved GIF: {args.out / args.gif_name} ({count} frames)")

        if args.save_comparison:
            cmp_paths = sorted(cmp_dir.glob("*.png"))
            count_cmp = write_gif(
                frame_paths=cmp_paths,
                output_path=args.out / f"comparison_{args.gif_name}",
                fps=args.gif_fps,
                max_frames=args.gif_max_frames,
            )
            print(f"Saved comparison GIF: {args.out / f'comparison_{args.gif_name}'} ({count_cmp} frames)")

    print(f"Render mode: {args.render_mode}")
    print(f"Mean coverage: {float(np.mean(coverages)):.4f}")
    print(f"Output directory: {args.out}")


if __name__ == "__main__":
    main()
