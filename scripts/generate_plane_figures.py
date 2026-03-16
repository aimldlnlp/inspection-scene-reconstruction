#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon3d.io_utils import load_rgb, load_transforms
from recon3d.metrics import psnr, ssim
from recon3d.render import render_scene_splats


def load_source_cache(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {
        "points": payload["points"].astype(np.float32),
        "colors": payload["colors"].astype(np.float32),
        "offsets": payload["offsets"].astype(np.int64),
        "cam_positions": payload["cam_positions"].astype(np.float32),
    }


def select_nearest_source_points(
    target_c2w: np.ndarray,
    source: dict[str, np.ndarray],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_pos = target_c2w[:3, 3]
    d = np.linalg.norm(source["cam_positions"] - target_pos[None, :], axis=1)
    ids = np.argsort(d)[: max(1, min(k, len(source["cam_positions"]))) ]

    pts_chunks = []
    col_chunks = []
    for i in ids:
        a = int(source["offsets"][i])
        b = int(source["offsets"][i + 1])
        pts_chunks.append(source["points"][a:b])
        col_chunks.append(source["colors"][a:b])

    if not pts_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts_chunks, axis=0), np.concatenate(col_chunks, axis=0)


def evaluate_grid_blur_fixed(
    dataset_dir: Path,
    source: dict[str, np.ndarray],
    k_values: list[int],
    blend_values: list[float],
    blur_sigma: float,
    point_cap: int = 180000,
) -> tuple[np.ndarray, np.ndarray]:
    transforms, frames = load_transforms(dataset_dir, split="test")
    w = int(transforms["w"])
    h = int(transforms["h"])
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])

    psnr_grid = np.zeros((len(k_values), len(blend_values)), dtype=np.float32)
    ssim_grid = np.zeros((len(k_values), len(blend_values)), dtype=np.float32)

    for i, k in enumerate(k_values):
        for j, blend in enumerate(blend_values):
            ps_vals = []
            ss_vals = []
            for fr in frames:
                gt = load_rgb(dataset_dir / fr["file_path"])
                c2w = np.array(fr["transform_matrix"], dtype=np.float32)
                pts, cols = select_nearest_source_points(c2w, source, k)
                if len(pts) > point_cap:
                    rng = np.random.default_rng(1000 + i * 100 + j)
                    sel = rng.choice(len(pts), size=point_cap, replace=False)
                    pts = pts[sel]
                    cols = cols[sel]

                pred, _, _ = render_scene_splats(
                    means=pts,
                    colors=cols,
                    c2w=c2w,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    width=w,
                    height=h,
                    hole_blend_distance=float(blend),
                    blur_sigma=float(blur_sigma),
                )
                ps_vals.append(psnr(gt, pred))
                ss_vals.append(ssim(gt, pred))

            psnr_grid[i, j] = float(np.mean(ps_vals))
            ssim_grid[i, j] = float(np.mean(ss_vals))
            print(f"grid done: k={k}, blend={blend}, blur={blur_sigma} -> PSNR={psnr_grid[i,j]:.3f}")

    return psnr_grid, ssim_grid


def save_grid_csv(path: Path, k_values: list[int], blend_values: list[float], psnr_grid: np.ndarray, ssim_grid: np.ndarray, blur_sigma: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "blend", "blur", "psnr", "ssim"])
        for i, k in enumerate(k_values):
            for j, blend in enumerate(blend_values):
                writer.writerow([k, blend, blur_sigma, float(psnr_grid[i, j]), float(ssim_grid[i, j])])


def fig5_ablation_heatmap_2d(out_path: Path, k_values: list[int], blend_values: list[float], psnr_grid: np.ndarray, blur_sigma: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=210)
    im = ax.imshow(psnr_grid, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(blend_values)))
    ax.set_yticks(np.arange(len(k_values)))
    ax.set_xticklabels([f"{b:g}" for b in blend_values])
    ax.set_yticklabels([str(k) for k in k_values])
    ax.set_xlabel("hole_blend_distance")
    ax.set_ylabel("source_k")
    ax.set_title(f"Ablation Heatmap on 2D Parameter Plane (blur={blur_sigma})")

    for i in range(len(k_values)):
        for j in range(len(blend_values)):
            val = psnr_grid[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < np.max(psnr_grid) - 0.7 else "black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.04)
    cbar.set_label("PSNR (dB)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig5_ablation_surface_3d(out_path: Path, k_values: list[int], blend_values: list[float], psnr_grid: np.ndarray, blur_sigma: float) -> None:
    x, y = np.meshgrid(np.array(blend_values, dtype=np.float32), np.array(k_values, dtype=np.float32))
    z = psnr_grid

    fig = plt.figure(figsize=(8.0, 6.0), dpi=210)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, cmap="viridis", edgecolor="#1f2937", linewidth=0.6, alpha=0.95)
    ax.set_xlabel("hole_blend_distance")
    ax.set_ylabel("source_k")
    ax.set_zlabel("PSNR (dB)")
    ax.set_title(f"Ablation Surface on 3D Parameter Plane (blur={blur_sigma})")
    ax.view_init(elev=26, azim=36)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.1, label="PSNR (dB)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig9_topdown_trajectory_2d(dataset_dir: Path, out_path: Path) -> None:
    tr_all, frames_all = load_transforms(dataset_dir, split="all")
    _, frames_test = load_transforms(dataset_dir, split="test")

    test_set = {fr["file_path"] for fr in frames_test}
    poses = np.array([np.array(fr["transform_matrix"], dtype=np.float32)[:3, 3] for fr in frames_all], dtype=np.float32)
    x = poses[:, 0]
    z = poses[:, 2]

    mask_test = np.array([fr["file_path"] in test_set for fr in frames_all], dtype=bool)
    mask_train = ~mask_test

    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=210)
    ax.plot(x, z, color="#64748b", linewidth=1.2, alpha=0.8)
    ax.scatter(x[mask_train], z[mask_train], s=20, c="#1f77b4", label="Train Views", alpha=0.9)
    ax.scatter(x[mask_test], z[mask_test], s=32, c="#d62728", marker="^", label="Test Views", alpha=0.95)
    ax.scatter(x[0], z[0], s=70, c="#16a34a", marker="o", label="Start")

    ax.set_xlabel("X (world)")
    ax.set_ylabel("Z (world)")
    ax.set_title("Camera Trajectory on 2D Ground Plane (X-Z)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _frustum_corners_world(c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int, depth: float) -> np.ndarray:
    corners_px = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    cam = np.zeros((4, 3), dtype=np.float32)
    cam[:, 2] = depth
    cam[:, 0] = (corners_px[:, 0] - cx) * depth / fx
    cam[:, 1] = (cy - corners_px[:, 1]) * depth / fy

    r = c2w[:3, :3].astype(np.float32)
    c = c2w[:3, 3].astype(np.float32)
    world = cam @ r.T + c[None, :]
    return world


def fig9_scene_frustums_3d(dataset_dir: Path, model_npz: Path, out_path: Path) -> None:
    tr_all, frames_all = load_transforms(dataset_dir, split="all")
    width = int(tr_all["w"])
    height = int(tr_all["h"])
    fx = float(tr_all["fl_x"])
    fy = float(tr_all["fl_y"])
    cx = float(tr_all["cx"])
    cy = float(tr_all["cy"])

    scene = np.load(model_npz)
    pts = scene["means"].astype(np.float32)
    cols = scene["colors"].astype(np.float32)

    rng = np.random.default_rng(42)
    n = min(38000, len(pts))
    idx = rng.choice(len(pts), size=n, replace=False)
    pts = pts[idx]
    cols = cols[idx]

    cam_idx = np.linspace(0, len(frames_all) - 1, 12).astype(int)
    cams = [np.array(frames_all[i]["transform_matrix"], dtype=np.float32) for i in cam_idx]

    fig = plt.figure(figsize=(8.2, 6.9), dpi=210)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.7, alpha=0.28)

    for c2w in cams:
        c = c2w[:3, 3]
        corners = _frustum_corners_world(c2w, fx, fy, cx, cy, width, height, depth=0.8)
        ax.scatter([c[0]], [c[1]], [c[2]], c="#ef4444", s=12)
        for p in corners:
            ax.plot([c[0], p[0]], [c[1], p[1]], [c[2], p[2]], color="#ef4444", linewidth=0.8, alpha=0.8)
        ring = np.vstack([corners, corners[0]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#ef4444", linewidth=0.9, alpha=0.8)

    ax.set_title("3D Scene Plane with Camera Frustums")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=48)
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    dataset_dir = root / "data" / "indoor_inspection_v1"
    source_cache = root / "outputs" / "indoor_run_showcase" / "reconstruction" / "source_views.npz"
    model_npz = root / "outputs" / "indoor_run_showcase" / "reconstruction" / "gaussians.npz"
    out_dir = root / "outputs_ready" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_cache(source_cache)
    k_values = [4, 8, 10]
    blend_values = [0.0, 8.0, 12.0]
    blur_sigma = 0.2

    psnr_grid, ssim_grid = evaluate_grid_blur_fixed(
        dataset_dir=dataset_dir,
        source=source,
        k_values=k_values,
        blend_values=blend_values,
        blur_sigma=blur_sigma,
        point_cap=180000,
    )

    save_grid_csv(
        out_dir / "Fig5_plane_grid_blur0p2.csv",
        k_values=k_values,
        blend_values=blend_values,
        psnr_grid=psnr_grid,
        ssim_grid=ssim_grid,
        blur_sigma=blur_sigma,
    )

    fig5_ablation_heatmap_2d(
        out_path=out_dir / "Fig5_ablation_heatmap_2d.png",
        k_values=k_values,
        blend_values=blend_values,
        psnr_grid=psnr_grid,
        blur_sigma=blur_sigma,
    )

    fig5_ablation_surface_3d(
        out_path=out_dir / "Fig5_ablation_surface_3d.png",
        k_values=k_values,
        blend_values=blend_values,
        psnr_grid=psnr_grid,
        blur_sigma=blur_sigma,
    )

    fig9_topdown_trajectory_2d(
        dataset_dir=dataset_dir,
        out_path=out_dir / "Fig9_topdown_trajectory_2d.png",
    )

    fig9_scene_frustums_3d(
        dataset_dir=dataset_dir,
        model_npz=model_npz,
        out_path=out_dir / "Fig9_scene_frustums_3d.png",
    )

    print(f"Saved plane-style figures to: {out_dir}")


if __name__ == "__main__":
    main()
