#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image, ImageDraw

from recon3d.io_utils import load_rgb, load_transforms, save_json
from recon3d.metrics import mae, psnr, ssim
from recon3d.render import render_scene_splats

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.titleweight": "normal",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)


def read_metrics_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "frame": r["frame"],
                    "psnr": float(r["psnr"]),
                    "ssim": float(r["ssim"]),
                    "mae": float(r["mae"]),
                    "coverage": float(r["coverage"]),
                }
            )
    return rows


def select_frames_for_fig2(rows: list[dict[str, float | str]], num_frames: int = 6) -> list[str]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r["psnr"]))
    if len(ordered) <= num_frames:
        return [str(r["frame"]) for r in ordered]
    idx = np.linspace(0, len(ordered) - 1, num_frames).astype(int)
    return [str(ordered[i]["frame"]) for i in idx]


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


def resize_np_rgb(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(img).resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((8, 8, 260, 42), fill=(0, 0, 0))
    draw.text((14, 14), text, fill=(255, 255, 255))
    return np.array(pil, dtype=np.uint8)


def fig1_teaser_gif(pred_all_dir: Path, out_path: Path, fps: int = 12) -> None:
    frames = sorted(pred_all_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {pred_all_dir}")

    target_count = min(120, len(frames))
    idx = np.linspace(0, len(frames) - 1, target_count).astype(int)
    images = []
    for i in idx:
        arr = np.array(Image.open(frames[i]).convert("RGB"))
        arr = add_label(arr, "Indoor Reconstruction Flythrough")
        images.append(arr)
    imageio.mimsave(out_path, images, duration=1.0 / fps, loop=0)


def fig2_qualitative_grid(
    metric_rows: list[dict[str, float | str]],
    gt_dir: Path,
    pred_metric_dir: Path,
    pred_visual_dir: Path,
    out_path: Path,
) -> None:
    selected = select_frames_for_fig2(metric_rows, num_frames=6)
    row_map = {str(r["frame"]): r for r in metric_rows}

    cell_w, cell_h = 360, 202
    cols = ["GT", "Pred (Metric)", "Pred (Visual)", "Error Map"]

    fig, axes = plt.subplots(
        len(selected),
        4,
        figsize=(15.4, 2.45 * len(selected)),
        dpi=190,
        constrained_layout=True,
    )
    if len(selected) == 1:
        axes = np.array([axes])

    err_ax_for_cbar = None
    for r, frame in enumerate(selected):
        gt = np.array(Image.open(gt_dir / frame).convert("RGB"), dtype=np.uint8)
        pm = np.array(Image.open(pred_metric_dir / frame).convert("RGB"), dtype=np.uint8)
        pv = np.array(Image.open(pred_visual_dir / frame).convert("RGB"), dtype=np.uint8)

        err = np.mean(np.abs(gt.astype(np.float32) - pv.astype(np.float32)), axis=2)
        err_norm = np.clip(err / 40.0, 0.0, 1.0)

        panels = [gt, pm, pv]
        for c, p in enumerate(panels):
            axes[r, c].imshow(resize_np_rgb(p, (cell_w, cell_h)))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=10, fontweight="normal")

        axes[r, 3].imshow(
            np.array(Image.fromarray((err_norm * 255).astype(np.uint8)).resize((cell_w, cell_h), Image.Resampling.BILINEAR)),
            cmap="inferno",
            vmin=0.0,
            vmax=255.0,
        )
        err_ax_for_cbar = axes[r, 3].images[0]
        axes[r, 3].axis("off")
        if r == 0:
            axes[r, 3].set_title(cols[3], fontsize=10, fontweight="normal")

        row_info = row_map.get(frame)
        if row_info is not None:
            label = f"{frame.replace('.png', '')}\nPSNR={float(row_info['psnr']):.2f}"
        else:
            label = frame.replace(".png", "")
        axes[r, 0].set_ylabel(label, fontsize=8)

    if err_ax_for_cbar is not None:
        cbar = fig.colorbar(err_ax_for_cbar, ax=axes[:, 3], fraction=0.018, pad=0.015)
        cbar.set_label("Relative Error", rotation=90)

    fig.suptitle("Qualitative Comparison on Representative Test Views", fontsize=13, fontweight="normal")
    fig.savefig(out_path)
    plt.close(fig)


def fig3_error_heatmap(frame_names: list[str], gt_dir: Path, pred_visual_dir: Path, out_path: Path) -> None:
    acc = None
    for frame in frame_names:
        gt = np.array(Image.open(gt_dir / frame).convert("RGB"), dtype=np.float32)
        pv = np.array(Image.open(pred_visual_dir / frame).convert("RGB"), dtype=np.float32)
        err = np.mean(np.abs(gt - pv), axis=2)
        acc = err if acc is None else (acc + err)

    mean_err = acc / max(1, len(frame_names))
    vmax = float(np.percentile(mean_err, 99))
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=190)
    im = ax.imshow(mean_err, cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-6))
    ax.set_title("Aggregate Pixel Error Heatmap (Visual Profile)", fontweight="normal")
    ax.set_xlabel("Image X")
    ax.set_ylabel("Image Y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error (0-255)")
    ax.text(
        0.015,
        0.97,
        f"Mean={float(np.mean(mean_err)):.2f} | P95={float(np.percentile(mean_err,95)):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.25"),
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fig4_metrics_curve(metric_rows: list[dict[str, float | str]], showcase_rows: list[dict[str, float | str]], out_path: Path) -> None:
    def frame_to_idx(name: str) -> int:
        return int(name.replace("frame_", "").replace(".png", ""))

    x1 = [frame_to_idx(str(r["frame"])) for r in metric_rows]
    x2 = [frame_to_idx(str(r["frame"])) for r in showcase_rows]

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.2), dpi=190, sharex=True)

    axes[0].plot(x1, [float(r["psnr"]) for r in metric_rows], marker="o", lw=2.0, ms=4.5, label="Metric Profile", color="#1f77b4")
    axes[0].plot(x2, [float(r["psnr"]) for r in showcase_rows], marker="s", lw=2.0, ms=4.3, label="Showcase Profile", color="#d62728", alpha=0.88)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(alpha=0.28, linestyle="--")
    axes[0].legend()

    axes[1].plot(x1, [float(r["ssim"]) for r in metric_rows], marker="o", lw=2.0, ms=4.5, color="#1f77b4")
    axes[1].plot(x2, [float(r["ssim"]) for r in showcase_rows], marker="s", lw=2.0, ms=4.3, color="#d62728", alpha=0.88)
    axes[1].set_ylabel("SSIM")
    axes[1].grid(alpha=0.28, linestyle="--")

    axes[2].plot(x1, [float(r["mae"]) for r in metric_rows], marker="o", lw=2.0, ms=4.5, color="#1f77b4")
    axes[2].plot(x2, [float(r["mae"]) for r in showcase_rows], marker="s", lw=2.0, ms=4.3, color="#d62728", alpha=0.88)
    axes[2].set_ylabel("MAE")
    axes[2].set_xlabel("Frame Index")
    axes[2].grid(alpha=0.28, linestyle="--")

    fig.suptitle("Per-Frame Metrics Curve", fontsize=13, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_combo(
    frames: list[dict],
    dataset_dir: Path,
    transforms: dict,
    source: dict[str, np.ndarray],
    source_k: int,
    hole_blend_distance: float,
    blur_sigma: float,
    point_cap: int = 280000,
) -> dict[str, float]:
    w = int(transforms["w"])
    h = int(transforms["h"])
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])

    scores_psnr = []
    scores_ssim = []
    scores_mae = []
    scores_cov = []

    for fr in frames:
        gt = load_rgb(dataset_dir / fr["file_path"])
        c2w = np.array(fr["transform_matrix"], dtype=np.float32)
        pts, cols = select_nearest_source_points(c2w, source, source_k)
        if len(pts) > point_cap:
            rng = np.random.default_rng(123 + source_k)
            sel = rng.choice(len(pts), size=point_cap, replace=False)
            pts = pts[sel]
            cols = cols[sel]
        pred, _, cov = render_scene_splats(
            means=pts,
            colors=cols,
            c2w=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            hole_blend_distance=hole_blend_distance,
            blur_sigma=blur_sigma,
        )
        scores_psnr.append(psnr(gt, pred))
        scores_ssim.append(ssim(gt, pred))
        scores_mae.append(mae(gt, pred))
        scores_cov.append(cov)

    return {
        "psnr": float(np.mean(scores_psnr)),
        "ssim": float(np.mean(scores_ssim)),
        "mae": float(np.mean(scores_mae)),
        "coverage": float(np.mean(scores_cov)),
    }


def fig5_ablation_table(
    dataset_dir: Path,
    transforms: dict,
    frames: list[dict],
    source: dict[str, np.ndarray],
    png_out: Path,
    csv_out: Path,
) -> list[dict[str, float]]:
    combos = [
        (4, 0.0, 0.0),
        (8, 0.0, 0.0),
        (10, 0.0, 0.0),
        (4, 8.0, 0.2),
        (8, 8.0, 0.2),
        (10, 8.0, 0.2),
        (4, 12.0, 0.35),
        (8, 12.0, 0.35),
        (10, 12.0, 0.35),
    ]
    rows: list[dict[str, float]] = []
    for i, (k, blend, blur) in enumerate(combos, start=1):
        stats = evaluate_combo(
            frames=frames,
            dataset_dir=dataset_dir,
            transforms=transforms,
            source=source,
            source_k=k,
            hole_blend_distance=blend,
            blur_sigma=blur,
        )
        rows.append(
            {
                "source_k": k,
                "blend": blend,
                "blur": blur,
                "psnr": stats["psnr"],
                "ssim": stats["ssim"],
                "mae": stats["mae"],
                "coverage": stats["coverage"],
            }
        )
        print(f"Ablation {i}/{len(combos)} done: k={k}, blend={blend}, blur={blur}")

    rows_sorted = sorted(rows, key=lambda r: r["psnr"], reverse=True)

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_k", "blend", "blur", "psnr", "ssim", "mae", "coverage"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    _draw_ablation_figure(rows_sorted, png_out)
    return rows_sorted


def _draw_ablation_figure(rows_sorted: list[dict[str, float]], png_out: Path) -> None:
    baseline = [r for r in rows_sorted if r["source_k"] == 4 and r["blend"] == 0.0 and r["blur"] == 0.0][0]
    showcase = [r for r in rows_sorted if r["source_k"] == 10 and r["blend"] == 12.0 and r["blur"] == 0.35][0]

    fig, (ax_rank, ax_tbl) = plt.subplots(
        2,
        1,
        figsize=(11.4, 7.3),
        dpi=190,
        gridspec_kw={"height_ratios": [2.3, 1.15]},
        constrained_layout=True,
    )

    labels = [f"k={int(r['source_k'])}, b={r['blend']:.1f}, s={r['blur']:.2f}" for r in rows_sorted]
    psnr_vals = [float(r["psnr"]) for r in rows_sorted]
    y = np.arange(len(rows_sorted))

    colors = ["#c8d5e6"] * len(rows_sorted)
    colors[0] = "#8ecfa8"
    for i, r in enumerate(rows_sorted):
        if r["source_k"] == showcase["source_k"] and r["blend"] == showcase["blend"] and r["blur"] == showcase["blur"]:
            colors[i] = "#f3d9a6"
        if r["source_k"] == baseline["source_k"] and r["blend"] == baseline["blend"] and r["blur"] == baseline["blur"]:
            colors[i] = "#8ecfa8"

    ax_rank.barh(y, psnr_vals, color=colors, edgecolor="#243447", linewidth=0.8)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(labels, fontsize=8.5)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("PSNR (dB)")
    ax_rank.set_title("Ablation Ranking by PSNR", pad=8)
    ax_rank.grid(axis="x", alpha=0.25, linestyle="--")
    for i, v in enumerate(psnr_vals):
        ax_rank.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)

    ax_tbl.axis("off")
    top_k = rows_sorted[:4]
    selected = top_k + [showcase]
    cell_data = []
    for r in selected:
        tag = "Best" if r == rows_sorted[0] else ("Showcase" if r == showcase else "Top")
        cell_data.append(
            [
                tag,
                int(r["source_k"]),
                f"{r['blend']:.2f}",
                f"{r['blur']:.2f}",
                f"{r['psnr']:.3f}",
                f"{r['ssim']:.4f}",
                f"{r['mae']:.3f}",
                f"{r['coverage']:.4f}",
            ]
        )
    col_labels = ["Tag", "k", "blend", "blur", "PSNR", "SSIM", "MAE", "Coverage"]
    table = ax_tbl.table(cellText=cell_data, colLabels=col_labels, loc="center", bbox=[0.03, 0.02, 0.94, 0.9])
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.18)
    for j in range(len(col_labels)):
        hdr = table[(0, j)]
        hdr.set_facecolor("#1f2937")
        hdr.get_text().set_color("white")
    for ridx in range(1, len(cell_data) + 1):
        tag = cell_data[ridx - 1][0]
        if tag == "Best":
            row_color = "#d8f5e0"
        elif tag == "Showcase":
            row_color = "#fff1cc"
        else:
            row_color = "white"
        for j in range(len(col_labels)):
            table[(ridx, j)].set_facecolor(row_color)

    fig.suptitle("Ablation Summary", fontsize=13, y=1.01)
    fig.savefig(png_out)
    plt.close(fig)


def fig6_coverage_vs_quality(metric_rows: list[dict[str, float | str]], showcase_rows: list[dict[str, float | str]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=170)
    x_m = [float(r["coverage"]) for r in metric_rows]
    y_m = [float(r["psnr"]) for r in metric_rows]
    x_s = [float(r["coverage"]) for r in showcase_rows]
    y_s = [float(r["psnr"]) for r in showcase_rows]

    ax.scatter(x_m, y_m, label="Metric Profile", s=45, marker="o", alpha=0.85, color="#1f77b4")
    ax.scatter(x_s, y_s, label="Showcase Profile", s=45, marker="^", alpha=0.85, color="#ff7f0e")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Coverage vs Quality Trade-off", fontweight="normal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def centered_crop(arr: np.ndarray, cx: int, cy: int, half_w: int, half_h: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x0 = max(0, cx - half_w)
    x1 = min(w, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(h, cy + half_h)
    return arr[y0:y1, x0:x1]


def fig7_failure_cases(metric_rows: list[dict[str, float | str]], gt_dir: Path, pred_visual_dir: Path, out_path: Path) -> None:
    worst = sorted(metric_rows, key=lambda r: float(r["psnr"]))[:4]

    fig, axes = plt.subplots(len(worst), 5, figsize=(16, 3.0 * len(worst)), dpi=150)
    if len(worst) == 1:
        axes = np.array([axes])

    headers = ["GT", "Prediction", "Error Map", "GT Zoom", "Pred Zoom"]
    for c, h in enumerate(headers):
        axes[0, c].set_title(h, fontsize=10, fontweight="normal")

    for r, row in enumerate(worst):
        frame = str(row["frame"])
        gt = np.array(Image.open(gt_dir / frame).convert("RGB"), dtype=np.uint8)
        pred = np.array(Image.open(pred_visual_dir / frame).convert("RGB"), dtype=np.uint8)
        err = np.mean(np.abs(gt.astype(np.float32) - pred.astype(np.float32)), axis=2)

        yx = np.unravel_index(np.argmax(err), err.shape)
        cy, cx = int(yx[0]), int(yx[1])

        h, w = gt.shape[:2]
        x0 = max(0, cx - 110)
        x1 = min(w, cx + 110)
        y0 = max(0, cy - 70)
        y1 = min(h, cy + 70)

        gt_zoom = centered_crop(gt, cx=cx, cy=cy, half_w=110, half_h=70)
        pred_zoom = centered_crop(pred, cx=cx, cy=cy, half_w=110, half_h=70)

        axes[r, 0].imshow(gt)
        axes[r, 1].imshow(pred)
        axes[r, 2].imshow(err, cmap="inferno")
        axes[r, 3].imshow(gt_zoom)
        axes[r, 4].imshow(pred_zoom)

        for c in range(5):
            axes[r, c].axis("off")

        for col in [0, 1]:
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.4, edgecolor="#22c55e", facecolor="none")
            axes[r, col].add_patch(rect)

        axes[r, 0].set_ylabel(f"{frame}\nPSNR={float(row['psnr']):.2f}", fontsize=8)

    fig.suptitle("Failure Cases with Local Zoom Inspection", fontsize=13, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path)
    plt.close(fig)


def fig8_blink_comparison_gif(frame_names: list[str], gt_dir: Path, pred_visual_dir: Path, out_path: Path) -> None:
    images: list[np.ndarray] = []
    for frame in frame_names:
        gt = np.array(Image.open(gt_dir / frame).convert("RGB"), dtype=np.uint8)
        pred = np.array(Image.open(pred_visual_dir / frame).convert("RGB"), dtype=np.uint8)

        images.append(add_label(gt, f"GT - {frame}"))
        images.append(add_label(pred, f"Prediction - {frame}"))

    imageio.mimsave(out_path, images, duration=0.22, loop=0)


def fig9_pipeline_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 4.6), dpi=220)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 5.3)
    ax.axis("off")

    x_nodes = [1.3, 4.0, 6.7, 9.4, 12.1]
    y_node = 2.4
    radius = 0.46
    labels = [
        ("Data\nCapture", "RGB + Depth + Pose"),
        ("Reconstruction", "Gaussian Scene"),
        ("Source-View\nFusion", "Nearest-k Projection"),
        ("Rendering", "PNG + GIF Outputs"),
        ("Evaluation", "PSNR / SSIM / MAE"),
    ]
    fills = ["#d9ecff", "#dff4e4", "#fff0d5", "#ffe4e4", "#e8e2ff"]

    ax.plot([x_nodes[0], x_nodes[-1]], [y_node, y_node], color="#95a0ae", lw=1.1, alpha=0.7, zorder=0)
    for i in range(len(x_nodes) - 1):
        ax.annotate(
            "",
            xy=(x_nodes[i + 1] - radius - 0.08, y_node),
            xytext=(x_nodes[i] + radius + 0.08, y_node),
            arrowprops=dict(arrowstyle="-|>", lw=1.7, color="#2f3a4a"),
        )

    for idx, (x, ((title, subtitle), fill)) in enumerate(zip(x_nodes, zip(labels, fills)), start=1):
        circ = patches.Circle((x, y_node), radius=radius, facecolor=fill, edgecolor="#334155", linewidth=1.6)
        ax.add_patch(circ)
        ax.text(x, y_node, str(idx), ha="center", va="center", fontsize=12, color="#1f2937")
        ax.text(x, y_node - 0.88, title, ha="center", va="top", fontsize=10)
        ax.text(x, y_node - 1.46, subtitle, ha="center", va="top", fontsize=8.6, color="#475569")

    input_box = patches.FancyBboxPatch(
        (0.35, 4.2),
        4.1,
        0.7,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor="#334155",
        facecolor="#f8fafc",
    )
    output_box = patches.FancyBboxPatch(
        (9.95, 4.2),
        4.1,
        0.7,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor="#334155",
        facecolor="#f8fafc",
    )
    ax.add_patch(input_box)
    ax.add_patch(output_box)
    ax.text(2.4, 4.55, "Input: Multi-view capture sequence", ha="center", va="center", fontsize=9.2)
    ax.text(12.0, 4.55, "Output: Visual artifacts + quantitative metrics", ha="center", va="center", fontsize=9.2)

    ax.set_title("End-to-End Pipeline Overview", fontsize=13, pad=10)
    fig.savefig(out_path)
    plt.close(fig)


def fig10_model_snapshot(model_npz: Path, out_path: Path) -> None:
    scene = np.load(model_npz)
    pts = scene["means"].astype(np.float32)
    cols = scene["colors"].astype(np.float32)

    rng = np.random.default_rng(42)
    n = min(45000, len(pts))
    idx = rng.choice(len(pts), size=n, replace=False)
    pts = pts[idx]
    cols = cols[idx]

    views = [(18, 25), (18, 120), (35, 215), (70, 310)]

    fig = plt.figure(figsize=(12, 9), dpi=180)
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=0.75, alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"View {i}: elev={elev}, azim={azim}", fontsize=9)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.8))
        ax.grid(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.pane.set_facecolor((1, 1, 1, 0))

    fig.suptitle("Reconstructed Scene Snapshot (Gaussian Point Model)", fontsize=13, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def write_paper_manifest(out_dir: Path, ablation_rows: list[dict[str, float]]) -> None:
    best = ablation_rows[0]
    payload = {
        "paper_assets": [
            "Fig1_teaser.gif",
            "Fig2_qualitative_grid.png",
            "Fig3_error_heatmap.png",
            "Fig4_metrics_curve.png",
            "Fig5_ablation_table.png",
            "Fig5_ablation_results.csv",
            "Fig6_coverage_vs_quality.png",
            "Fig7_failure_cases.png",
            "Fig8_blink_comparison.gif",
            "Fig9_pipeline_overview.png",
            "Fig10_model_snapshot.png",
        ],
        "ablation_best": best,
    }
    save_json(out_dir / "paper_manifest.json", payload)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    dataset_dir = root / "data" / "indoor_inspection_v1"
    metric_eval_csv = root / "outputs" / "indoor_run_metric" / "eval_test" / "metrics_per_frame.csv"
    showcase_eval_csv = root / "outputs" / "indoor_run_showcase" / "eval_test" / "metrics_per_frame.csv"
    gt_dir = root / "outputs" / "indoor_run_visual" / "renders_test" / "gt"
    pred_metric_dir = root / "outputs" / "indoor_run_metric" / "renders_test" / "pred"
    pred_visual_dir = root / "outputs" / "indoor_run_visual" / "renders_test" / "pred"
    pred_all_dir = root / "outputs" / "indoor_run_showcase" / "renders_all" / "pred"
    source_cache_path = root / "outputs" / "indoor_run_showcase" / "reconstruction" / "source_views.npz"
    model_npz = root / "outputs" / "indoor_run_showcase" / "reconstruction" / "gaussians.npz"

    out_dir = root / "outputs_ready" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = read_metrics_csv(metric_eval_csv)
    showcase_rows = read_metrics_csv(showcase_eval_csv)
    frame_names = [str(r["frame"]) for r in metric_rows]

    print("Generating Fig1...")
    fig1_teaser_gif(pred_all_dir, out_dir / "Fig1_teaser.gif")

    print("Generating Fig2...")
    fig2_qualitative_grid(metric_rows, gt_dir, pred_metric_dir, pred_visual_dir, out_dir / "Fig2_qualitative_grid.png")

    print("Generating Fig3...")
    fig3_error_heatmap(frame_names, gt_dir, pred_visual_dir, out_dir / "Fig3_error_heatmap.png")

    print("Generating Fig4...")
    fig4_metrics_curve(metric_rows, showcase_rows, out_dir / "Fig4_metrics_curve.png")

    print("Generating Fig5 (ablation)...")
    transforms, frames_test = load_transforms(dataset_dir, split="test")
    source = load_source_cache(source_cache_path)
    ablation_rows = fig5_ablation_table(
        dataset_dir=dataset_dir,
        transforms=transforms,
        frames=frames_test,
        source=source,
        png_out=out_dir / "Fig5_ablation_table.png",
        csv_out=out_dir / "Fig5_ablation_results.csv",
    )

    print("Generating Fig6...")
    fig6_coverage_vs_quality(metric_rows, showcase_rows, out_dir / "Fig6_coverage_vs_quality.png")

    print("Generating Fig7...")
    fig7_failure_cases(metric_rows, gt_dir, pred_visual_dir, out_dir / "Fig7_failure_cases.png")

    print("Generating Fig8...")
    fig8_blink_comparison_gif(frame_names, gt_dir, pred_visual_dir, out_dir / "Fig8_blink_comparison.gif")

    print("Generating Fig9...")
    fig9_pipeline_overview(out_dir / "Fig9_pipeline_overview.png")

    print("Generating Fig10...")
    fig10_model_snapshot(model_npz, out_dir / "Fig10_model_snapshot.png")

    write_paper_manifest(out_dir, ablation_rows)

    print(f"Paper assets saved to: {out_dir}")


if __name__ == "__main__":
    main()
