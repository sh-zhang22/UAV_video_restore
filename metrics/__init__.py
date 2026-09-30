"""UAV_video_repair 指标模块。

暴露：
- Evaluator: 统一评测器（一次加载、多 cand 复用、批量目录扫描）
- SeedVRGeometry / seedvr_target_res: SeedVR 输入端几何（视频对齐 + bbox 映射）
- 单指标函数：psnr / psnr_y / ssim / lpips_metric / miou / miou_from_videos / miou_bbox_match
- read_video_tchw_float / read_video_thwc_uint8 / read_mask_thw_bool  # IO 工具
"""
from ._io import (
    read_video_tchw_float,
    read_video_thwc_uint8,
    read_mask_thw_bool,
    assert_same_shape,
)
from ._registry import REGISTRY, list_metrics
from .psnr import psnr, psnr_y
from .ssim import ssim
from .lpips import lpips_metric
from .miou import miou, miou_from_videos, miou_bbox_match
from .geometry import SeedVRGeometry, seedvr_target_res, DEFAULT_MAX_AREA
from .evaluator import Evaluator, DEFAULT_PIXEL_METRICS

__all__ = [
    "Evaluator", "DEFAULT_PIXEL_METRICS",
    "SeedVRGeometry", "seedvr_target_res", "DEFAULT_MAX_AREA",
    "psnr", "psnr_y", "ssim", "lpips_metric", "miou", "miou_from_videos", "miou_bbox_match",
    "read_video_tchw_float", "read_video_thwc_uint8", "read_mask_thw_bool",
    "assert_same_shape",
    "REGISTRY", "list_metrics",
]
