#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full indoor 3D reconstruction pipeline")
    parser.add_argument("--dataset", type=Path, default=Path("data/indoor_inspection_v1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/indoor_run"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--test-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recon-split", type=str, default="all", choices=["train", "all"])
    parser.add_argument("--depth-stride", type=int, default=3)
    parser.add_argument("--source-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=220000)
    parser.add_argument("--render-mode", type=str, default="nearest_views", choices=["nearest_views", "global"])
    parser.add_argument("--source-k", type=int, default=10)
    parser.add_argument("--hole-blend-distance", type=float, default=12.0)
    parser.add_argument("--blur-sigma", type=float, default=0.35)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    output = args.output
    reconstruction_dir = output / "reconstruction"
    render_test_dir = output / "renders_test"
    render_all_dir = output / "renders_all"
    eval_dir = output / "eval"
    eval_test_dir = output / "eval_test"

    if not args.skip_generate:
        run_step(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_indoor_dataset.py"),
                "--out",
                str(dataset),
                "--frames",
                str(args.frames),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--test-stride",
                str(args.test_stride),
                "--seed",
                str(args.seed),
                "--export-gif",
                "--gif-name",
                "dataset_preview.gif",
                "--gif-fps",
                "12",
                "--save-reference-cloud",
            ]
        )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "reconstruct_gaussian_scene.py"),
            "--dataset",
            str(dataset),
            "--out",
            str(reconstruction_dir),
            "--split",
            str(args.recon_split),
            "--depth-stride",
            str(args.depth_stride),
            "--source-stride",
            str(args.source_stride),
            "--voxel-size",
            str(args.voxel_size),
            "--max-points",
            str(args.max_points),
        ]
    )

    model_path = reconstruction_dir / "gaussians.npz"

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_gaussian_views.py"),
            "--dataset",
            str(dataset),
            "--model",
            str(model_path),
            "--split",
            "test",
            "--out",
            str(render_test_dir),
            "--render-mode",
            str(args.render_mode),
            "--source-k",
            str(args.source_k),
            "--save-gt",
            "--save-comparison",
            "--export-gif",
            "--gif-name",
            "novel_views_test.gif",
            "--gif-fps",
            "12",
            "--hole-blend-distance",
            str(args.hole_blend_distance),
            "--blur-sigma",
            str(args.blur_sigma),
        ]
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "render_gaussian_views.py"),
            "--dataset",
            str(dataset),
            "--model",
            str(model_path),
            "--split",
            "all",
            "--out",
            str(render_all_dir),
            "--render-mode",
            str(args.render_mode),
            "--source-k",
            str(args.source_k),
            "--export-gif",
            "--gif-name",
            "flythrough_all.gif",
            "--gif-fps",
            "14",
            "--hole-blend-distance",
            str(args.hole_blend_distance),
            "--blur-sigma",
            str(args.blur_sigma),
        ]
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_reconstruction.py"),
            "--dataset",
            str(dataset),
            "--model",
            str(model_path),
            "--split",
            "all",
            "--out",
            str(eval_dir),
            "--render-mode",
            str(args.render_mode),
            "--source-k",
            str(args.source_k),
            "--hole-blend-distance",
            str(args.hole_blend_distance),
            "--blur-sigma",
            str(args.blur_sigma),
        ]
    )

    run_step(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_reconstruction.py"),
            "--dataset",
            str(dataset),
            "--model",
            str(model_path),
            "--split",
            "test",
            "--out",
            str(eval_test_dir),
            "--render-mode",
            str(args.render_mode),
            "--source-k",
            str(args.source_k),
            "--hole-blend-distance",
            str(args.hole_blend_distance),
            "--blur-sigma",
            str(args.blur_sigma),
        ]
    )

    print("\nPipeline completed successfully.")
    print(f"Dataset: {dataset}")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
