"""SSIM：用 torchmetrics.functional，逐帧算跨帧平均。

约定输入：pred/gt 都是 (T, C, H, W) float in [0, 1]。
"""
from __future__ import annotations

import torch

from ._registry import register


@register("ssim")
def ssim(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0, device: str = "cpu") -> float:
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    if pred.shape != gt.shape:
        raise ValueError(f"ssim shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    pred = pred.float().to(device)
    gt = gt.float().to(device)
    # torchmetrics ssim 期望 (N, C, H, W)；我们把 T 当 N，一次算完即可
    val = tm_ssim(pred, gt, data_range=data_range, reduction="elementwise_mean")
    return float(val.item())
