from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _project_world_to_image(
    points_world: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float,
    far: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = c2w[:3, 3].astype(np.float32)
    r = c2w[:3, :3].astype(np.float32)

    cam = (points_world - c[None, :]) @ r
    z = cam[:, 2]
    valid = (z > near) & (z < far)
    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    idx_valid = np.where(valid)[0]
    x = cam[valid, 0]
    y = cam[valid, 1]
    z = z[valid]

    u = np.round(fx * (x / z) + cx).astype(np.int32)
    v = np.round(cy - fy * (y / z)).astype(np.int32)

    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[in_img], v[in_img], z[in_img], idx_valid[in_img]


def _background_gradient(height: int, width: int) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    bg_line = np.concatenate(
        [
            0.84 - 0.24 * yy,
            0.89 - 0.28 * yy,
            0.94 - 0.32 * yy,
        ],
        axis=1,
    )
    return np.repeat(bg_line[:, None, :], width, axis=1)


def render_scene_splats(
    means: np.ndarray,
    colors: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near: float = 0.2,
    far: float = 20.0,
    hole_blend_distance: float = 0.0,
    blur_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Render a Gaussian-like splat image from a reconstructed point scene.

    This renderer uses nearest-depth point selection, then hole filling and a
    mild Gaussian smoothing pass to emulate splat blending while staying fast.
    """
    u, v, z, idx = _project_world_to_image(
        points_world=means,
        c2w=c2w,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        near=near,
        far=far,
    )

    bg = _background_gradient(height, width)
    img = bg.copy()
    depth = np.full((height, width), np.inf, dtype=np.float32)
    sparse = np.zeros((height, width, 3), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)

    if len(u) == 0:
        return (img * 255.0).astype(np.uint8), np.zeros((height, width), dtype=np.float32), 0.0

    lin = v * width + u
    order = np.argsort(z)
    lin_sorted = lin[order]
    _, first = np.unique(lin_sorted, return_index=True)
    pick = order[first]

    uu = u[pick]
    vv = v[pick]
    zz = z[pick]

    sparse[vv, uu] = colors[idx[pick]]
    valid[vv, uu] = True
    depth[vv, uu] = zz
    img[valid] = sparse[valid]

    coverage = float(valid.mean())

    # Fill missing pixels by nearest valid color with distance-based blending.
    if np.any(valid) and hole_blend_distance > 0.0:
        dist, idx = distance_transform_edt(~valid, return_indices=True)
        nearest = sparse[idx[0], idx[1]]
        blend = np.exp(-dist / max(hole_blend_distance, 1e-6)).astype(np.float32)
        blend = np.clip(blend, 0.0, 1.0)
        img = np.where(valid[..., None], sparse, nearest * blend[..., None] + bg * (1.0 - blend[..., None]))

        depth_nearest = depth[idx[0], idx[1]]
        depth = np.where(valid, depth, depth_nearest)
        depth = np.where(np.isfinite(depth), depth, 0.0)

    if blur_sigma > 0.0:
        img = gaussian_filter(img, sigma=(blur_sigma, blur_sigma, 0.0), mode="nearest")

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8), depth.astype(np.float32), coverage
