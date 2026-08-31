"""YOLO 检测生成 pixel-space mask。

设计原则：
- 纯工具，仅在被 make_mask_from_video.py 显式 import 时才 import ultralytics（延迟依赖）
- 输出 (T, H, W) uint8 in {0, 255}，方便 write_video 存 mp4
- bbox → 填 1（矩形掩码），可选 dilate 少量像素避免边缘紧贴目标
- 不追求 seg / temporal smoothing —— 追求稳定可用

消费端（methods/_seedvr_ctrl_utils.py::load_mask_as_TCHW）会：
  read_video → (T, 3, H, W) uint8 → mean(dim=1)/255.0 → (T, 1, H, W) float32 in [0,1]
所以我们只要写出「三通道复制的 uint8 mp4，白=1、黑=0」就 100% 兼容。
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch


# ------------------------- 主入口 ------------------------- #

def video_to_mask(
    in_video: str,
    yolo_ckpt: str = "yolov8n.pt",
    classes: Optional[Sequence[str]] = None,
    conf: float = 0.25,
    iou: float = 0.7,
    dilate: int = 0,
    device: str = "cuda:0",
    imgsz: int = 640,
    verbose: bool = False,
) -> torch.Tensor:
    """逐帧过 YOLO，返回 pixel-space mask。

    Args:
        in_video: 输入视频路径
        yolo_ckpt: Ultralytics YOLO ckpt 路径（yolov8n.pt / yolov8s.pt / 自定义）；
                  首次用会自动从 GitHub release 下到 ~/.config/Ultralytics/
        classes: 保留的类别名（COCO 80 类，如 ["person", "car", "truck"]）；
                None → 保留全部检测结果
        conf: 置信度阈值（低于此值 → 丢弃）
        iou: NMS IoU 阈值
        dilate: bbox mask 膨胀像素数（≥0）。>0 时用 max-pool 实现，avoid soft edge
        device: 推理设备，如 "cuda:0" / "cpu"
        imgsz: YOLO 推理分辨率（会等比缩放 + letterbox）；不影响 mask 输出分辨率
        verbose: 打开 ultralytics 自身日志

    Returns:
        mask: (T, H, W) uint8 in {0, 255}，H/W 与输入视频一致
    """
    from ultralytics import YOLO
    from torchvision.io.video import read_video

    if not os.path.isfile(in_video):
        raise FileNotFoundError(f"input video not found: {in_video}")

    # ---- 加载视频 ----
    vid, _, info = read_video(in_video, output_format="TCHW")    # (T, 3, H, W) uint8
    T, C, H, W = vid.shape
    assert C == 3, f"expected RGB, got C={C}"
    print(f"[yolo-mask] input: {in_video}, shape=({T},{H},{W}), fps={info.get('video_fps')}")

    # ---- 加载 YOLO ----
    model = YOLO(yolo_ckpt)
    # 类别名 → 类别 id
    class_ids: Optional[List[int]] = None
    if classes is not None:
        name_to_id = {v: k for k, v in model.names.items()}
        missing = [c for c in classes if c not in name_to_id]
        if missing:
            raise ValueError(
                f"class(es) not in model.names: {missing}\n"
                f"available: {sorted(name_to_id.keys())}"
            )
        class_ids = [name_to_id[c] for c in classes]
        print(f"[yolo-mask] filter classes: {classes} → ids {class_ids}")
    else:
        print(f"[yolo-mask] keep all classes")

    # ---- 逐帧推理 + 填 mask ----
    # ultralytics 支持 numpy / list of numpy 直接输入；一次给一帧简单稳定
    mask = np.zeros((T, H, W), dtype=np.uint8)
    n_boxes_total = 0
    for i in range(T):
        # (3, H, W) → (H, W, 3) numpy BGR-agnostic（ultralytics 内部会处理）
        frame = vid[i].permute(1, 2, 0).contiguous().numpy()
        res_list = model.predict(
            source=frame,
            conf=conf, iou=iou,
            classes=class_ids,
            device=device, imgsz=imgsz,
            verbose=verbose,
        )
        res = res_list[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.int32)  # (N, 4)
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W, x2))
            y2 = max(0, min(H, y2))
            if x2 > x1 and y2 > y1:
                mask[i, y1:y2, x1:x2] = 255
                n_boxes_total += 1

    print(f"[yolo-mask] total boxes across all frames: {n_boxes_total}")

    # ---- 可选 dilate ----
    if dilate > 0:
        mask = _dilate_binary(mask, radius=dilate)
        print(f"[yolo-mask] dilated by {dilate} px")

    coverage = float((mask > 0).mean())
    print(f"[yolo-mask] mask coverage: {coverage*100:.2f}%")

    return torch.from_numpy(mask)


# ------------------------- 辅助 ------------------------- #

def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """(T, H, W) uint8 二值 mask → max-pool 膨胀 radius 像素。"""
    import torch.nn.functional as F
    t = torch.from_numpy(mask).float()[:, None]                     # (T, 1, H, W)
    k = 2 * radius + 1
    t = F.max_pool2d(t, kernel_size=k, stride=1, padding=radius)
    return t.squeeze(1).clamp(0, 255).to(torch.uint8).numpy()


def save_mask_mp4(mask: torch.Tensor, out_path: str, fps: float = 29.0) -> None:
    """(T, H, W) uint8 → 三通道复制写成 mp4。消费端 mean(dim=1)/255.0 即等价二值 mask。"""
    from torchvision.io.video import write_video

    assert mask.ndim == 3 and mask.dtype == torch.uint8, mask.shape
    thw3 = mask[..., None].expand(-1, -1, -1, 3).contiguous()       # (T, H, W, 3)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    write_video(out_path, thw3, fps=fps)
    print(f"[yolo-mask] saved: {out_path} ({thw3.shape})")
