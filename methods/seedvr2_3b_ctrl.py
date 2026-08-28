"""SeedVR2-3B + 第 17 通道 mask 控制 + LoRA + NaPatchIn 首层解冻 适配器。

用法：
    Recover(
        video_path="in.mp4",
        recovered_path="out.mp4",
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b_ctrl",
        device="cuda:1",
        method_kwargs={
            "mask_path": "mask.png",          # 或 .npy / .mp4；None → 全 1 mask（sanity check）
            "lora_ckpt": None,                # 训练后传入，加载 LoRA + vid_in.proj 权重
            # ...其它 method_kwargs 与 seedvr2_3b 一致（res_h/res_w/seed/sp_size/...）
        },
    )
"""
from __future__ import annotations

from typing import Optional

from methods._registry import register
from methods._seedvr_common import run_seedvr


@register("seedvr2_3b_ctrl")
def _run(*, video_path: str, recovered_path: str, ckpt_path: str,
         device: Optional[str] = None, **kwargs) -> str:
    return run_seedvr(
        variant="seedvr2_3b_ctrl",
        video_path=video_path,
        recovered_path=recovered_path,
        ckpt_path=ckpt_path,
        device=device,
        method_kwargs=kwargs,
    )
