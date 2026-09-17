"""PSNR：逐帧 MSE → dB，跨帧平均。

约定输入：pred/gt 都是 (T, C, H, W) float in [0, 1]，通道数任意（RGB 通常 3）。
用 data_range=1.0；如果输入范围不同，调用方需自行归一化。
"""
from __future__ import annotations

import math

import torch

from ._registry import register


@register("psnr")
def psnr(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    if pred.shape != gt.shape:
        raise ValueError(f"psnr shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    pred = pred.float()
    gt = gt.float()
    # 逐帧 MSE
    T = pred.shape[0]
    mse_per_frame = ((pred - gt) ** 2).reshape(T, -1).mean(dim=1)   # (T,)
    # 完美相等的帧 → mse=0 → psnr=inf，用一个大值代替 avg 里的 inf 会污染均值
    # 这里按论文常见做法：跳过完美相等的帧，或全部相等时返回 float('inf')
    finite = mse_per_frame > 0
    if not finite.any():
        return float("inf")
    psnr_per_frame = 10.0 * torch.log10((data_range ** 2) / mse_per_frame[finite])
    return psnr_per_frame.mean().item()
