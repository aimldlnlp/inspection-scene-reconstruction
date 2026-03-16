#!/usr/bin/env python3
"""Generate an indoor multi-view dataset for 3D reconstruction.

Output structure:
  <out>/
    images/frame_0000.png
    depth/frame_0000.png
    preview.gif
    scene_reference.npz
    transforms_train.json
    transforms_test.json
    transforms_all.json
    intrinsics.json
    metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def look_at(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return camera-to-world matrix with camera forward +Z."""
    forward = normalize(target - camera_pos)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(forward, world_up))
    up = normalize(np.cross(right, forward))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


def sample_box_surfaces(
    center: np.ndarray,
    size: np.ndarray,
    n_points: int,
    color: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    sx, sy, sz = size / 2.0

    face_ids = rng.integers(0, 6, size=n_points)
    uv = rng.uniform(-1.0, 1.0, size=(n_points, 2))

    pts = np.zeros((n_points, 3), dtype=np.float32)
    pts[:, 0] = cx
    pts[:, 1] = cy
    pts[:, 2] = cz

    # +X, -X
    sel = face_ids == 0
    pts[sel, 0] += sx
    pts[sel, 1] += uv[sel, 0] * sy
    pts[sel, 2] += uv[sel, 1] * sz

    sel = face_ids == 1
    pts[sel, 0] -= sx
    pts[sel, 1] += uv[sel, 0] * sy
    pts[sel, 2] += uv[sel, 1] * sz

    # +Y, -Y
    sel = face_ids == 2
    pts[sel, 1] += sy
    pts[sel, 0] += uv[sel, 0] * sx
    pts[sel, 2] += uv[sel, 1] * sz

    sel = face_ids == 3
    pts[sel, 1] -= sy
    pts[sel, 0] += uv[sel, 0] * sx
    pts[sel, 2] += uv[sel, 1] * sz

    # +Z, -Z
    sel = face_ids == 4
    pts[sel, 2] += sz
    pts[sel, 0] += uv[sel, 0] * sx
    pts[sel, 1] += uv[sel, 1] * sy

    sel = face_ids == 5
    pts[sel, 2] -= sz
    pts[sel, 0] += uv[sel, 0] * sx
    pts[sel, 1] += uv[sel, 1] * sy

    colors = np.repeat(color[None, :], n_points, axis=0)
    colors += rng.normal(scale=0.03, size=colors.shape).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)
    return pts, colors


def sample_sphere(
    center: np.ndarray,
    radius: float,
    n_points: int,
    color: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = rng.normal(size=(n_points, 3)).astype(np.float32)
    xyz = normalize(xyz)
    r = radius * np.cbrt(rng.uniform(0.8, 1.0, size=(n_points, 1)).astype(np.float32))
    pts = center[None, :] + xyz * r
    colors = np.repeat(color[None, :], n_points, axis=0)
    colors += rng.normal(scale=0.02, size=colors.shape).astype(np.float32)
    return pts, np.clip(colors, 0.0, 1.0)


def build_indoor_pointcloud(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []

    room_half = 4.0
    room_height = 3.0

    n_floor = 70_000
    fx = rng.uniform(-room_half, room_half, size=n_floor).astype(np.float32)
    fz = rng.uniform(-room_half, room_half, size=n_floor).astype(np.float32)
    fy = np.zeros_like(fx)
    floor_pts = np.stack([fx, fy, fz], axis=1)
    checker = (
        (np.floor((fx + room_half) / 0.6) + np.floor((fz + room_half) / 0.6)) % 2
    ).astype(np.float32)
    floor_col = np.stack(
        [0.45 + 0.2 * checker, 0.38 + 0.18 * checker, 0.3 + 0.12 * checker], axis=1
    )
    points.append(floor_pts)
    colors.append(floor_col)

    n_ceiling = 25_000
    cx = rng.uniform(-room_half, room_half, size=n_ceiling).astype(np.float32)
    cz = rng.uniform(-room_half, room_half, size=n_ceiling).astype(np.float32)
    cy = np.full_like(cx, room_height)
    ceiling_pts = np.stack([cx, cy, cz], axis=1)
    ceiling_col = np.repeat(np.array([[0.82, 0.84, 0.86]], dtype=np.float32), n_ceiling, axis=0)
    points.append(ceiling_pts)
    colors.append(ceiling_col)

    n_wall = 36_000
    wy = rng.uniform(0.0, room_height, size=n_wall).astype(np.float32)
    wz = rng.uniform(-room_half, room_half, size=n_wall).astype(np.float32)
    wx = np.full_like(wy, -room_half)
    wall_left = np.stack([wx, wy, wz], axis=1)

    wy2 = rng.uniform(0.0, room_height, size=n_wall).astype(np.float32)
    wz2 = rng.uniform(-room_half, room_half, size=n_wall).astype(np.float32)
    wx2 = np.full_like(wy2, room_half)
    wall_right = np.stack([wx2, wy2, wz2], axis=1)

    wy3 = rng.uniform(0.0, room_height, size=n_wall).astype(np.float32)
    wx3 = rng.uniform(-room_half, room_half, size=n_wall).astype(np.float32)
    wz3 = np.full_like(wy3, -room_half)
    wall_back = np.stack([wx3, wy3, wz3], axis=1)

    wy4 = rng.uniform(0.0, room_height, size=n_wall).astype(np.float32)
    wx4 = rng.uniform(-room_half, room_half, size=n_wall).astype(np.float32)
    wz4 = np.full_like(wy4, room_half)
    wall_front = np.stack([wx4, wy4, wz4], axis=1)

    wall_pts = np.concatenate([wall_left, wall_right, wall_back, wall_front], axis=0)
    wall_col = np.repeat(np.array([[0.74, 0.76, 0.78]], dtype=np.float32), len(wall_pts), axis=0)
    wall_col += rng.normal(scale=0.03, size=wall_col.shape).astype(np.float32)
    points.append(wall_pts)
    colors.append(np.clip(wall_col, 0.0, 1.0))

    # Furniture: shelves, boxes, desk, plant
    furniture = [
        (np.array([-2.2, 0.9, -1.2], dtype=np.float32), np.array([1.2, 1.8, 0.45], dtype=np.float32), np.array([0.52, 0.37, 0.26], dtype=np.float32), 25_000),
        (np.array([2.3, 0.6, 1.5], dtype=np.float32), np.array([1.6, 1.2, 0.8], dtype=np.float32), np.array([0.58, 0.62, 0.66], dtype=np.float32), 22_000),
        (np.array([0.7, 0.35, -2.1], dtype=np.float32), np.array([1.4, 0.7, 0.7], dtype=np.float32), np.array([0.30, 0.32, 0.36], dtype=np.float32), 18_000),
        (np.array([-0.9, 0.25, 1.9], dtype=np.float32), np.array([0.9, 0.5, 0.9], dtype=np.float32), np.array([0.68, 0.22, 0.20], dtype=np.float32), 16_000),
    ]
    for center, size, color, n in furniture:
        p, c = sample_box_surfaces(center, size, n, color, rng)
        points.append(p)
        colors.append(c)

    sphere_pts, sphere_col = sample_sphere(
        center=np.array([1.9, 0.55, -0.2], dtype=np.float32),
        radius=0.35,
        n_points=12_000,
        color=np.array([0.16, 0.50, 0.23], dtype=np.float32),
        rng=rng,
    )
    points.append(sphere_pts)
    colors.append(sphere_col)

    all_points = np.concatenate(points, axis=0).astype(np.float32)
    all_colors = np.clip(np.concatenate(colors, axis=0).astype(np.float32), 0.0, 1.0)

    # Light jitter to reduce aliasing artifacts.
    all_points += rng.normal(scale=0.002, size=all_points.shape).astype(np.float32)
    return all_points, all_colors


def generate_camera_trajectory(n_frames: int) -> list[np.ndarray]:
    cams = []
    target = np.array([0.0, 1.2, 0.0], dtype=np.float32)
    for i in range(n_frames):
        a = 2.0 * math.pi * i / n_frames
        radius = 3.2 + 0.15 * math.sin(2.0 * a)
        x = radius * math.cos(a)
        z = radius * math.sin(a)
        y = 1.35 + 0.15 * math.sin(3.0 * a)
        c = np.array([x, y, z], dtype=np.float32)
        cams.append(look_at(c, target))
    return cams


def render_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    c = c2w[:3, 3]
    r = c2w[:3, :3]

    pc = (points - c[None, :]) @ r
    z = pc[:, 2]
    valid = z > 0.08

    x = pc[valid, 0]
    y = pc[valid, 1]
    z = z[valid]
    col = colors[valid]

    u = np.round(fx * (x / z) + (width * 0.5)).astype(np.int32)
    v = np.round((height * 0.5) - fy * (y / z)).astype(np.int32)

    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_img):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        return rgb, depth

    u = u[in_img]
    v = v[in_img]
    z = z[in_img]
    col = col[in_img]

    # Background gradient keeps frames visually coherent for quick inspection.
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    bg = np.concatenate(
        [
            0.84 - 0.24 * yy,
            0.89 - 0.28 * yy,
            0.94 - 0.32 * yy,
        ],
        axis=1,
    )
    rgb = np.repeat(bg[:, None, :], width, axis=1)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    lin = v * width + u
    order = np.argsort(z)
    lin_sorted = lin[order]
    _, first = np.unique(lin_sorted, return_index=True)
    pick = order[first]

    uu = u[pick]
    vv = v[pick]
    rgb[vv, uu] = col[pick]
    depth[vv, uu] = z[pick]

    # Gentle sensor-like noise.
    rgb += rng.normal(scale=0.008, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)

    return (rgb * 255.0).astype(np.uint8), depth


def save_depth_mm_png(path: Path, depth_m: np.ndarray) -> None:
    depth_mm = np.where(np.isfinite(depth_m), depth_m * 1000.0, 0.0)
    depth_mm = np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def write_transforms(
    out_path: Path,
    camera_angle_x: float,
    frames: list[dict],
    width: int,
    height: int,
    fx: float,
    fy: float,
) -> None:
    payload = {
        "camera_angle_x": camera_angle_x,
        "w": width,
        "h": height,
        "fl_x": fx,
        "fl_y": fy,
        "cx": width * 0.5,
        "cy": height * 0.5,
        "frames": frames,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_preview_gif(
    image_paths: list[Path],
    out_path: Path,
    fps: int,
    max_frames: int,
) -> int:
    if not image_paths or fps <= 0 or max_frames <= 1:
        return 0

    total = len(image_paths)
    step = max(1, math.ceil(total / max_frames))
    sampled_paths = image_paths[::step]

    frames = [Image.open(p).convert("RGB") for p in sampled_paths]
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indoor dataset generator")
    parser.add_argument("--out", type=Path, default=Path("data/indoor_inspection_v1"))
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--export-gif", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=96)
    parser.add_argument("--gif-name", type=str, default="preview.gif")
    parser.add_argument("--save-reference-cloud", action="store_true")
    args = parser.parse_args()

    if args.frames < 8:
        raise ValueError("--frames should be >= 8 for useful reconstruction")

    out = args.out
    images_dir = out / "images"
    depth_dir = out / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    points, colors = build_indoor_pointcloud(rng)

    fov = math.radians(args.fov_deg)
    fx = (args.width * 0.5) / math.tan(fov * 0.5)
    fy = fx
    camera_angle_x = 2.0 * math.atan(args.width / (2.0 * fx))

    cameras = generate_camera_trajectory(args.frames)

    train_frames = []
    test_frames = []
    all_frames = []
    rgb_paths: list[Path] = []

    for i, c2w in enumerate(cameras):
        rgb, depth = render_pointcloud(
            points=points,
            colors=colors,
            c2w=c2w,
            width=args.width,
            height=args.height,
            fx=fx,
            fy=fy,
            rng=rng,
        )

        stem = f"frame_{i:04d}"
        rgb_path = images_dir / f"{stem}.png"
        depth_path = depth_dir / f"{stem}.png"
        Image.fromarray(rgb).save(rgb_path)
        save_depth_mm_png(depth_path, depth)
        rgb_paths.append(rgb_path)

        frame = {
            "file_path": f"images/{stem}.png",
            "depth_file_path": f"depth/{stem}.png",
            "transform_matrix": c2w.tolist(),
        }
        all_frames.append(frame)

        if i % args.test_stride == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    write_transforms(
        out_path=out / "transforms_train.json",
        camera_angle_x=camera_angle_x,
        frames=train_frames,
        width=args.width,
        height=args.height,
        fx=fx,
        fy=fy,
    )
    write_transforms(
        out_path=out / "transforms_test.json",
        camera_angle_x=camera_angle_x,
        frames=test_frames,
        width=args.width,
        height=args.height,
        fx=fx,
        fy=fy,
    )
    write_transforms(
        out_path=out / "transforms_all.json",
        camera_angle_x=camera_angle_x,
        frames=all_frames,
        width=args.width,
        height=args.height,
        fx=fx,
        fy=fy,
    )

    intrinsics = {
        "width": args.width,
        "height": args.height,
        "fx": fx,
        "fy": fy,
        "cx": args.width * 0.5,
        "cy": args.height * 0.5,
        "fov_deg": args.fov_deg,
    }
    (out / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")

    meta = {
        "scene": "indoor_inspection_office_floor",
        "generator": "generate_indoor_dataset.py",
        "seed": args.seed,
        "num_frames": args.frames,
        "num_train": len(train_frames),
        "num_test": len(test_frames),
        "point_count": int(points.shape[0]),
        "notes": "Indoor multi-view capture package for reconstruction experiments.",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    gif_count = 0
    if args.export_gif:
        gif_count = write_preview_gif(
            image_paths=rgb_paths,
            out_path=out / args.gif_name,
            fps=args.gif_fps,
            max_frames=args.gif_max_frames,
        )

    if args.save_reference_cloud:
        np.savez_compressed(out / "scene_reference.npz", points=points, colors=colors)

    print(f"Saved dataset to: {out.resolve()}")
    print(f"Frames: total={args.frames}, train={len(train_frames)}, test={len(test_frames)}")
    print(f"Point cloud size used for rendering: {points.shape[0]}")
    if args.export_gif:
        print(f"GIF preview: {args.gif_name} ({gif_count} frames)")
    if args.save_reference_cloud:
        print("Saved scene reference cloud: scene_reference.npz")


if __name__ == "__main__":
    main()
