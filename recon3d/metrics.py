from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_f = gt.astype(np.float32) / 255.0
    pr_f = pred.astype(np.float32) / 255.0
    return float(peak_signal_noise_ratio(gt_f, pr_f, data_range=1.0))


def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_f = gt.astype(np.float32) / 255.0
    pr_f = pred.astype(np.float32) / 255.0
    return float(structural_similarity(gt_f, pr_f, data_range=1.0, channel_axis=2))


def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(gt.astype(np.float32) - pred.astype(np.float32))))
