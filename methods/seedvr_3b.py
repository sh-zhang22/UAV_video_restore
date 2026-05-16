"""SeedVR-3B 适配器（多步采样，cfg_scale=6.5、sample_steps=50）。"""
from __future__ import annotations

from typing import Optional

from methods._registry import register
from methods._seedvr_common import run_seedvr


@register("seedvr_3b")
def _run(*, video_path: str, recovered_path: str, ckpt_path: str,
         device: Optional[str] = None, **kwargs) -> str:
    return run_seedvr(
        variant="seedvr_3b",
        video_path=video_path,
        recovered_path=recovered_path,
        ckpt_path=ckpt_path,
        device=device,
        method_kwargs=kwargs,
    )
