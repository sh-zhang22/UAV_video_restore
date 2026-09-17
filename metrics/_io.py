"""指标模块公用 IO：视频加载 + shape 对齐检查。"""
from __future__ import annotations

import os

import numpy as np
import torch


def read_video_thwc_uint8(path: str) -> torch.Tensor:
    """读 mp4 → (T, H, W, 3) uint8，与 torchvision.io 行为一致但保 HWC 便于 skimage/PIL。"""
    from torchvision.io.video import read_video
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    vid, _, _ = read_video(path, output_format="THWC")   # (T,H,W,3) uint8
    return vid


def read_video_tchw_float(path: str) -> torch.Tensor:
    """读 mp4 → (T, 3, H, W) float32 in [0, 1]，PSNR/SSIM/LPIPS 通用输入。"""
    from torchvision.io.video import read_video
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    vid, _, _ = read_video(path, output_format="TCHW")   # (T,3,H,W) uint8
    return vid.float() / 255.0


def read_mask_thw_bool(path: str) -> torch.Tensor:
    """读 mask mp4/png/npy → (T, H, W) bool；灰度视为 mask 通道。
    - mp4:  用 torchvision.io 读取，若 3 通道则取第 0 通道
    - png:  单帧 → T=1；PIL 打开转灰度
    - npy:  (T,H,W) 或 (H,W)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp4":
        vid = read_video_thwc_uint8(path)                # (T,H,W,3)
        m = vid[..., 0]                                  # (T,H,W)
    elif ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image
        img = np.array(Image.open(path).convert("L"))    # (H,W)
        m = torch.from_numpy(img).unsqueeze(0)           # (1,H,W)
    elif ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None]
        m = torch.from_numpy(arr)
    else:
        raise ValueError(f"unsupported mask ext: {ext}")
    return m.to(torch.uint8) >= 128                       # (T,H,W) bool


def assert_same_shape(a: torch.Tensor, b: torch.Tensor, name_a: str = "pred", name_b: str = "gt"):
    """严格检查两个视频/mask tensor shape 一致，不做任何自动 resize。"""
    if a.shape != b.shape:
        raise ValueError(
            f"[metrics] shape mismatch: {name_a}.shape={tuple(a.shape)} vs "
            f"{name_b}.shape={tuple(b.shape)}. 不做自动 resize，请先对齐。"
        )
