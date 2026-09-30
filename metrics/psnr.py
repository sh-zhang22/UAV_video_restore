"""PSNR：逐帧 MSE → dB，跨帧平均。

约定输入：pred/gt 都是 (T, C, H, W) float in [0, 1]，通道数任意（RGB 通常 3）。
用 data_range=1.0；如果输入范围不同，调用方需自行归一化。
"""
from __future__ import annotations

import math

import torch

from ._registry import register


def _rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """(T, 3, H, W) [0,1] RGB → (T, 1, H, W) Y，BT.601 limited range（Y ∈ [16,235]/255）。

    本项目的 mp4 均无色彩标签，解码器按 BT.601 limited 转 RGB，此公式即其逆变换，
    与码流里的真实 Y 平面一致；同时也是 BasicSR / MATLAB rgb2ycbcr 的 VSR 通用约定。
    """
    if x.shape[1] != 3:
        raise ValueError(f"rgb_to_y 需要 C=3，got shape={tuple(x.shape)}")
    w = torch.tensor([65.481, 128.553, 24.966], dtype=x.dtype, device=x.device) / 255.0
    return (x * w.view(1, 3, 1, 1)).sum(dim=1, keepdim=True) + 16.0 / 255.0


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


@register("psnr_y")
def psnr_y(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    """Y-PSNR：只算亮度通道（BT.601 limited，见 _rgb_to_y），data_range 仍取 1.0（VSR 惯例）。

    输入约定与 psnr() 一致：(T, 3, H, W) [0,1] RGB。
    """
    if pred.shape != gt.shape:
        raise ValueError(f"psnr_y shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    return psnr(_rgb_to_y(pred), _rgb_to_y(gt), data_range=data_range)
