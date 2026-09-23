"""UAV_video_repair 指标模块。

暴露：
- Evaluator: 统一评测器（一次加载、多 cand 复用、批量目录扫描）
- 单指标函数：psnr / ssim / lpips_metric / miou / miou_from_videos / miou_bbox_match
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
from .miou import miou, miou_from_videos, miou_bbox_match
from .evaluator import Evaluator, DEFAULT_PIXEL_METRICS

__all__ = [
    "Evaluator", "DEFAULT_PIXEL_METRICS",
    "psnr", "ssim", "lpips_metric", "miou", "miou_from_videos", "miou_bbox_match",
    "read_video_tchw_float", "read_video_thwc_uint8", "read_mask_thw_bool",
    "assert_same_shape",
    "REGISTRY", "list_metrics",
]
