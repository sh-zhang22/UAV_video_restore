"""SeedVR2-3B + ControlNet-Lite 侧枝（mask 条件）适配器。

与 seedvr2_3b_ctrl 的核心差异：
- ctrl variant 复用 SR condition 的第 17 通道存 mask，用 LoRA 微调主干
- ctrlnet variant 保留第 17 通道原语义（全 1），额外挂一个 K=4 层的侧枝网络
  （~346M 参数），zero-init 保证训练启动等价 baseline

用法：
    Recover(
        video_path="in.mp4",
        recovered_path="out.mp4",
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b_ctrlnet",
        device="cuda:1",
        method_kwargs={
            "mask_path": "mask.png",          # 或 .npy / .mp4；None → 全 1 mask（sanity check）
            "ctrlnet_ckpt": None,             # 训练后传入，加载侧枝权重
            "ctrlnet_K": 4,                   # 侧枝层数，默认 4
            # ...其它 method_kwargs 与 seedvr2_3b 一致
        },
    )
"""
from __future__ import annotations

from typing import Optional

from methods._registry import register
from methods._seedvr_common import run_seedvr


@register("seedvr2_3b_ctrlnet")
def _run(*, video_path: str, recovered_path: str, ckpt_path: str,
         device: Optional[str] = None, **kwargs) -> str:
    return run_seedvr(
        variant="seedvr2_3b_ctrlnet",
        video_path=video_path,
        recovered_path=recovered_path,
        ckpt_path=ckpt_path,
        device=device,
        method_kwargs=kwargs,
    )
