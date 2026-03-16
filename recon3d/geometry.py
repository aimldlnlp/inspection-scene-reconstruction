from __future__ import annotations

import numpy as np


def backproject_depth_to_world(
    depth_mm: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb: np.ndarray | None = None,
    stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 15.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Back-project a depth map into world coordinates.

    Returns:
      world_points: (N, 3) float32
      colors: (N, 3) float32 in [0, 1] or None
      scales: (N,) float32 in meters
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")

    h, w = depth_mm.shape
    v = np.arange(0, h, stride)
    u = np.arange(0, w, stride)
    uu, vv = np.meshgrid(u, v)

    z = depth_mm[vv, uu].astype(np.float32) / 1000.0
    valid = (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32) if rgb is not None else None,
            np.zeros((0,), dtype=np.float32),
        )

    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid]

    x = (uu - cx) * z / fx
    y = (cy - vv) * z / fy
    cam = np.stack([x, y, z], axis=1).astype(np.float32)

    r = c2w[:3, :3].astype(np.float32)
    c = c2w[:3, 3].astype(np.float32)
    world = cam @ r.T + c[None, :]

    scales = np.clip((z / fx) * max(1.0, float(stride)) * 1.5, 0.004, 0.08).astype(np.float32)

    out_colors: np.ndarray | None = None
    if rgb is not None:
        cols = rgb[vv.astype(np.int32), uu.astype(np.int32)].astype(np.float32) / 255.0
        out_colors = cols

    return world, out_colors, scales


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    voxel_size: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Voxel-downsample points and keep mean position/color/scale.

    Returns:
      means, rgb, scales, confidence
    """
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    vox = np.floor(points / voxel_size).astype(np.int32)
    uniq, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    m = len(uniq)

    pos_acc = np.zeros((m, 3), dtype=np.float64)
    col_acc = np.zeros((m, 3), dtype=np.float64)
    scl_acc = np.zeros((m,), dtype=np.float64)

    for j in range(3):
        pos_acc[:, j] = np.bincount(inv, weights=points[:, j], minlength=m)
        col_acc[:, j] = np.bincount(inv, weights=colors[:, j], minlength=m)
    scl_acc[:] = np.bincount(inv, weights=scales, minlength=m)

    denom = counts.astype(np.float64)
    means = (pos_acc / denom[:, None]).astype(np.float32)
    rgb = np.clip(col_acc / denom[:, None], 0.0, 1.0).astype(np.float32)
    scl = np.clip((scl_acc / denom).astype(np.float32), 0.003, 0.1)
    conf = np.clip(np.log1p(counts.astype(np.float32)) / np.log(12.0), 0.2, 1.0)

    if len(means) > max_points:
        sel = np.argsort(conf)[-max_points:]
        means = means[sel]
        rgb = rgb[sel]
        scl = scl[sel]
        conf = conf[sel]

    return means, rgb, scl, conf
