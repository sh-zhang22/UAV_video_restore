"""mIoU：逐帧二值 mask IoU 跨帧平均。

两种输入方式（evaluate() 里根据传入参数分派）：
- 显式：pred_mask + gt_mask，都是 (T, H, W) bool
- 隐式：pred_video + gt_video，内部各自过 YOLO 生成 mask 再算

边界情况：某帧并集为空（两 mask 都全 0）→ 该帧 IoU 定义为 1.0（无物体、无预测 = 一致）
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ._registry import register


def _iou_per_frame(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """pred/gt: (T, H, W) bool → (T,) float IoU per frame."""
    T = pred.shape[0]
    pred_flat = pred.reshape(T, -1)
    gt_flat = gt.reshape(T, -1)
    inter = (pred_flat & gt_flat).sum(dim=1).float()
    union = (pred_flat | gt_flat).sum(dim=1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    return iou


@register("miou")
def miou(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
) -> float:
    """pred_mask / gt_mask: (T, H, W) bool。跨帧 mean IoU。"""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"miou shape mismatch: {tuple(pred_mask.shape)} vs {tuple(gt_mask.shape)}")
    if pred_mask.dtype != torch.bool:
        pred_mask = pred_mask.to(torch.bool)
    if gt_mask.dtype != torch.bool:
        gt_mask = gt_mask.to(torch.bool)
    iou = _iou_per_frame(pred_mask, gt_mask)
    return float(iou.mean().item())


def miou_from_videos(
    pred_video: str,
    gt_video: str,
    *,
    yolo_ckpt: str = "yolov8n.pt",
    classes: Optional[Sequence[str]] = None,
    conf: float = 0.25,
    iou_thr: float = 0.7,
    dilate: int = 0,
    device: str = "cuda:0",
) -> float:
    """两条视频各自过 YOLO 生成 mask 再算 mIoU。用于没有 gt_mask 但有 gt 视频的场景。"""
    from methods._yolo_mask import video_to_mask
    pred_m = video_to_mask(
        in_video=pred_video, yolo_ckpt=yolo_ckpt, classes=classes,
        conf=conf, iou=iou_thr, dilate=dilate, device=device,
    )                                                              # (T,H,W) uint8 {0,255}
    gt_m = video_to_mask(
        in_video=gt_video, yolo_ckpt=yolo_ckpt, classes=classes,
        conf=conf, iou=iou_thr, dilate=dilate, device=device,
    )
    pred_b = pred_m >= 128
    gt_b = gt_m >= 128
    return miou(pred_b, gt_b)
