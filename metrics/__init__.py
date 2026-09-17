"""UAV_video_repair 指标模块。

暴露：
- evaluate(pred_video, gt_video, ...) → dict[str, float]  # 统一入口，一次算全部
- 单指标函数：psnr / ssim / lpips_metric / miou / miou_from_videos
- read_video_tchw_float / read_video_thwc_uint8 / read_mask_thw_bool  # IO 工具
"""
from ._io import (
    read_video_tchw_float,
    read_video_thwc_uint8,
    read_mask_thw_bool,
    assert_same_shape,
)
from ._registry import REGISTRY, list_metrics
from .psnr import psnr
from .ssim import ssim
from .lpips import lpips_metric
from .miou import miou, miou_from_videos

__all__ = [
    "psnr", "ssim", "lpips_metric", "miou", "miou_from_videos",
    "read_video_tchw_float", "read_video_thwc_uint8", "read_mask_thw_bool",
    "assert_same_shape",
    "REGISTRY", "list_metrics",
]
