"""mIoU：两种输入模式。

1. mask 模式（原实现）：pred/gt (T,H,W) bool → 逐帧二值 IoU 均值
2. bbox 匹配模式（新增）：pred/ref 每帧一个 xyxy list → 一对一 IoU 最大化匹配，
   漏检 IoU=0，误检不惩罚，小目标（面积<阈值）双向剔除。

边界情况：某帧并集为空（两 mask 都全 0）→ 该帧 IoU 定义为 1.0（无物体、无预测 = 一致）
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, List

import numpy as np
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


# ================================================================
# bbox 匹配式 mIoU（松弛版）
# ================================================================

BoxList = Sequence[Tuple[float, float, float, float]]     # 每帧一组 xyxy


def _pairwise_iou_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """(Na, 4) × (Nb, 4) xyxy → (Na, Nb) IoU 矩阵。numpy 向量化。"""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    xa1, ya1, xa2, ya2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    xb1, yb1, xb2, yb2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    inter_w = np.clip(np.minimum(xa2, xb2) - np.maximum(xa1, xb1), 0, None)
    inter_h = np.clip(np.minimum(ya2, yb2) - np.maximum(ya1, yb1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((xa2 - xa1) * (ya2 - ya1), 0, None)
    area_b = np.clip((xb2 - xb1) * (yb2 - yb1), 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)


def _filter_small_boxes(boxes: List[Tuple[float, float, float, float]],
                        min_side: float) -> Tuple[np.ndarray, int]:
    """按最短边过滤：min(w,h) >= min_side 保留。返回 (kept_xyxy, n_dropped)。"""
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), 0
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    w = arr[:, 2] - arr[:, 0]
    h = arr[:, 3] - arr[:, 1]
    keep = (w >= min_side) & (h >= min_side)
    return arr[keep], int((~keep).sum())


def _match_greedy(iou_mat: np.ndarray) -> List[Tuple[int, int, float]]:
    """贪心一对一匹配：反复找最大 IoU，标记该行/列已用。返回 [(i, j, iou), ...]。"""
    if iou_mat.size == 0:
        return []
    m = iou_mat.copy()
    matches = []
    while True:
        idx = np.argmax(m)
        i, j = np.unravel_index(idx, m.shape)
        v = m[i, j]
        if v <= 0.0:
            break
        matches.append((int(i), int(j), float(v)))
        m[i, :] = -1.0
        m[:, j] = -1.0
    return matches


def _match_hungarian(iou_mat: np.ndarray) -> List[Tuple[int, int, float]]:
    """匈牙利最优匹配（scipy）：最大化 IoU 之和。返回 [(i, j, iou>0), ...]。"""
    if iou_mat.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-iou_mat)      # 最大化 → 传 -M
    out = []
    for i, j in zip(rows, cols):
        v = float(iou_mat[i, j])
        if v > 0:                                     # 分配上但 IoU=0 的对不算匹配
            out.append((int(i), int(j), v))
    return out


def miou_bbox_match(
    pred_frames: Sequence[BoxList],
    ref_frames: Sequence[BoxList],
    *,
    min_side: float = 32.0,
    apply_min_side_to_pred: bool = True,
    match: str = "greedy",
) -> Dict:
    """bbox 匹配式 mIoU。

    规则（师浩大人版本）：
    - 每帧对 pred × ref 做 pairwise IoU 后一对一匹配（默认 greedy）
    - ref 中未匹配上的 → 记 IoU=0（漏检惩罚）
    - pred 中未匹配上的 → 丢弃（误检不惩罚）
    - 面积过滤：min(w,h) < min_side 的框剔除，默认 pred/ref 双向剔除

    Params
    ------
    pred_frames, ref_frames
        每帧一个 xyxy list；帧数取 min(len)，超出的忽略
    min_side
        最短边阈值，默认 32（对应"小于 32×32"不评估）
    apply_min_side_to_pred
        是否也剔小 pred；True 更公平（默认），False 只剔小 ref
    match
        "greedy" 或 "hungarian"；hungarian 需 scipy

    Return
    ------
    dict:
        miou            : 全部 ref_kept 的 IoU 均值（micro-average，micro=1 未匹配算 0）
        n_scored        : len(ref_kept) 累加，就是均值分母
        n_matched       : 匹配上的对数
        n_missed        : 漏检的 ref 数（贡献 0）
        n_ref_kept      : 过滤后 ref 总数（=n_scored）
        n_ref_dropped   : 因小尺寸剔除的 ref 数
        n_pred_dropped  : 因小尺寸剔除的 pred 数
        avg_iou_matched : 只对匹配对求均值（诊断用）
        T               : 参与评估的帧数
    """
    if match not in ("greedy", "hungarian"):
        raise ValueError(f"unknown match: {match}")

    T = min(len(pred_frames), len(ref_frames))

    all_ious: List[float] = []                # 每个 ref_kept 贡献一个 IoU（含 0）
    matched_ious: List[float] = []
    n_matched = n_missed = 0
    n_ref_kept = n_ref_dropped = 0
    n_pred_dropped = 0

    for t in range(T):
        pred_arr, pd = _filter_small_boxes(
            list(pred_frames[t]),
            min_side if apply_min_side_to_pred else 0.0,
        )
        ref_arr, rd = _filter_small_boxes(list(ref_frames[t]), min_side)
        n_ref_kept += ref_arr.shape[0]
        n_ref_dropped += rd
        n_pred_dropped += pd

        n_ref = ref_arr.shape[0]
        if n_ref == 0:
            continue                                  # 该帧 ref 全被剔或本来就空 → 不产生贡献

        iou_mat = _pairwise_iou_xyxy(pred_arr, ref_arr)
        if match == "greedy":
            pairs = _match_greedy(iou_mat)
        else:
            pairs = _match_hungarian(iou_mat)

        matched_ref = set(j for _, j, _ in pairs)
        for _, _, v in pairs:
            all_ious.append(v)
            matched_ious.append(v)
            n_matched += 1
        for j in range(n_ref):
            if j not in matched_ref:
                all_ious.append(0.0)                   # 漏检
                n_missed += 1

    miou_val = float(np.mean(all_ious)) if all_ious else None
    return {
        "miou": miou_val,
        "n_scored": len(all_ious),
        "n_matched": n_matched,
        "n_missed": n_missed,
        "n_ref_kept": n_ref_kept,
        "n_ref_dropped": n_ref_dropped,
        "n_pred_dropped": n_pred_dropped,
        "avg_iou_matched": float(np.mean(matched_ious)) if matched_ious else None,
        "T": T,
    }
