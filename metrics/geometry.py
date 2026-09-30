"""SeedVR 输入端空间变换的精确复刻：AreaResize(bicubic, antialias) → clamp → DivisibleCrop(16)。

restored 视频处在该变换之后的坐标系里，所以 orig / compressed 与之比较前必须走同一变换；
bbox 在 orig ↔ restored 坐标之间映射也必须用同一组 (缩放, 裁剪偏移)。
与 third_party/SeedVR 的 NaResize + DivisibleCrop 逐位一致（见 scripts/eval/test_evaluator.py）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

DEFAULT_MAX_AREA = 720 * 1280
DIVISOR = 16


def seedvr_target_res(H: int, W: int, max_area: Optional[int]) -> Tuple[int, int]:
    """eval_compress_restore 传给 SeedVR 的 (res_h, res_w)：面积超过 max_area 才等比缩小。"""
    if max_area is not None and H * W > max_area:
        s = math.sqrt(max_area / (H * W))
        return round(H * s), round(W * s)
    return H, W


@dataclass(frozen=True)
class SeedVRGeometry:
    H: int          # orig 尺寸
    W: int
    H1: int         # AreaResize 之后
    W1: int
    Hc: int         # DivisibleCrop 之后 = restored 尺寸
    Wc: int
    top: int        # 中心裁剪偏移（在 H1×W1 坐标系里）
    left: int

    @classmethod
    def from_res(cls, H: int, W: int, res_h: int, res_w: int) -> "SeedVRGeometry":
        # NaResize(resolution=sqrt(res_h*res_w)) → AreaResize(max_area=resolution**2)，保留同一浮点路径
        max_area = ((res_h * res_w) ** 0.5) ** 2
        scale = math.sqrt(max_area / (H * W))
        H1, W1 = round(H * scale), round(W * scale)
        Hc, Wc = H1 - H1 % DIVISOR, W1 - W1 % DIVISOR
        # 与 TVF.center_crop 相同的取整（Python round，银行家舍入）
        top, left = int(round((H1 - Hc) / 2.0)), int(round((W1 - Wc) / 2.0))
        return cls(H, W, H1, W1, Hc, Wc, top, left)

    @classmethod
    def from_max_area(cls, H: int, W: int, max_area: Optional[int] = DEFAULT_MAX_AREA) -> "SeedVRGeometry":
        return cls.from_res(H, W, *seedvr_target_res(H, W, max_area))

    @property
    def in_size(self) -> Tuple[int, int]:
        return self.H, self.W

    @property
    def out_size(self) -> Tuple[int, int]:
        return self.Hc, self.Wc

    def apply_video(self, vid: torch.Tensor, device: str = "cpu", batch: int = 32) -> torch.Tensor:
        """(T,3,H,W) [0,1] → (T,3,Hc,Wc) [0,1]，结果放回 CPU。"""
        if tuple(vid.shape[-2:]) != self.in_size:
            raise ValueError(f"geometry 输入应为 {self.in_size}，收到 {tuple(vid.shape[-2:])}")
        out = []
        for i in range(0, vid.shape[0], batch):
            x = vid[i:i + batch].to(device)
            if (self.H1, self.W1) != self.in_size:
                x = TVF.resize(x, [self.H1, self.W1], interpolation=InterpolationMode.BICUBIC,
                               antialias=True)
            x = x.clamp_(0.0, 1.0)[..., self.top:self.top + self.Hc, self.left:self.left + self.Wc]
            out.append(x.cpu())
        return torch.cat(out)

    def boxes_to_orig(self, boxes) -> np.ndarray:
        """restored 坐标 xyxy (N,4) → orig 坐标。"""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] + self.left) * (self.W / self.W1)
        b[:, [1, 3]] = (b[:, [1, 3]] + self.top) * (self.H / self.H1)
        return b

    def boxes_from_orig(self, boxes) -> np.ndarray:
        """orig 坐标 xyxy (N,4) → restored 坐标（可能落到裁剪区外，调用方自行 clip）。"""
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]] * (self.W1 / self.W) - self.left
        b[:, [1, 3]] = b[:, [1, 3]] * (self.H1 / self.H) - self.top
        return b
