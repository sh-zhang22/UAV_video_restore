"""LPIPS：AlexNet backbone，逐帧算跨帧平均。

约定输入：pred/gt 都是 (T, C, H, W) float in [0, 1]，通道数必须 3（RGB）。
LPIPS 内部期望 (-1, 1)，本模块负责归一化。

模型实例按 (backbone, device) 缓存，避免重复加载。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from ._registry import register

_MODEL_CACHE: Dict[Tuple[str, str], "object"] = {}


def _get_lpips_model(net: str, device: str):
    key = (net, device)
    if key not in _MODEL_CACHE:
        import lpips
        model = lpips.LPIPS(net=net, verbose=False)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


@register("lpips")
def lpips_metric(
    pred: torch.Tensor,
    gt: torch.Tensor,
    net: str = "alex",
    device: str = "cuda:0",
    batch: int = 8,
) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"lpips shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if pred.shape[1] != 3:
        raise ValueError(f"lpips 需要 RGB (C=3), got C={pred.shape[1]}")
    model = _get_lpips_model(net, device)
    # (T, 3, H, W) [0,1] → [-1, 1]
    pred_n = (pred.float().to(device) * 2.0 - 1.0)
    gt_n = (gt.float().to(device) * 2.0 - 1.0)

    T = pred_n.shape[0]
    vals = []
    with torch.no_grad():
        for i in range(0, T, batch):
            p = pred_n[i:i + batch]
            g = gt_n[i:i + batch]
            d = model(p, g)                        # (B, 1, 1, 1)
            vals.append(d.reshape(-1))
    return float(torch.cat(vals).mean().item())
