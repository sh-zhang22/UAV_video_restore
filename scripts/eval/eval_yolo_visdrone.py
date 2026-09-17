#!/usr/bin/env python3
"""VisDrone2019-VID-slices test-dev 上跑 YOLOv8 baseline，评估 mask IoU。

mIoU 定义（对齐下游 ControlNet mask 用途）：
    对每一帧
      pred_mask = ∪(YOLO 预测框 ∈ 目标类)
      gt_mask   = ∪(GT 框 ∈ 目标类)
      frame_iou = |pred ∩ gt| / |pred ∪ gt|  ; 若并集为 0，跳过
    mIoU = mean(frame_iou across all frames)

COCO→VisDrone 类别映射：
    COCO 0 person    -> VisDrone {1 pedestrian, 2 people}
    COCO 1 bicycle   -> VisDrone {3 bicycle}
    COCO 2 car       -> VisDrone {4 car, 5 van}
    COCO 3 motorcycle-> VisDrone {10 motor}
    COCO 5 bus       -> VisDrone {9 bus}
    COCO 7 truck     -> VisDrone {6 truck}
    未映射 VisDrone 类 {7 tricycle, 8 awning-tricycle, 11 others}
      → 从 GT mask 中排除（COCO 模型天然预测不出）

Usage:
    python eval_yolo_visdrone.py
    python eval_yolo_visdrone.py --imgsz 1280 --conf 0.10 --model yolov8s.pt
    python eval_yolo_visdrone.py --split test-dev,val
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/nas/datasets/yixin/UAV_Dataset/VisDrone2019-VID-slices")

# COCO id -> {VisDrone ids}
COCO_TO_VD = {
    0: {1, 2},
    1: {3},
    2: {4, 5},
    3: {10},
    5: {9},
    7: {6},
}
COCO_KEEP = sorted(COCO_TO_VD.keys())
VD_KEEP_COCO = sorted({v for s in COCO_TO_VD.values() for v in s})

# VisDrone 预训练模型直接对齐 GT 的 8 个有效类（排除 tricycle/awning-tricycle/others）
# GT 类：1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
#        7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others
# 训练模型 names（11 类，0-indexed）：与 GT 完全对应，只是 index 差 1
# 我们只关心 8 个"能对到 COCO/UAV 场景语义"的类：pedestrian, people, bicycle, car, van, truck, bus, motor
# 排除 tricycle/awning-tricycle/others 是为了与 baseline v1 公平对比
VD_MODEL_CLASSES = [0, 1, 2, 3, 4, 5, 8, 9]  # visdrone-pretrained 0-idx
VD_KEEP_NATIVE = [c + 1 for c in VD_MODEL_CLASSES]  # 对齐 GT 1-idx


def probe(mp4: Path) -> tuple[int, int, int]:
    """return (W, H, nb_frames)"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)["streams"][0]
    return int(d["width"]), int(d["height"]), int(d["nb_frames"])


def load_gt(txt: Path) -> dict[int, list[tuple[float, float, float, float, int]]]:
    """local_frame(1-idx) -> list of (l, t, w, h, category)"""
    frames: dict[int, list] = defaultdict(list)
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            fr = int(parts[0])
            l, t, w, h = (float(x) for x in parts[2:6])
            cat = int(parts[7])
            frames[fr].append((l, t, w, h, cat))
    return frames


def boxes_to_mask(boxes: list[tuple[float, float, float, float]],
                  W: int, H: int) -> np.ndarray:
    """List of (l,t,w,h) → uint8 mask (H, W)."""
    m = np.zeros((H, W), dtype=np.uint8)
    for l, t, w, h in boxes:
        x1 = max(0, int(round(l)))
        y1 = max(0, int(round(t)))
        x2 = min(W, int(round(l + w)))
        y2 = min(H, int(round(t + h)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m


def frame_iou(pred: np.ndarray, gt: np.ndarray) -> float | None:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return None
    return inter / union


def eval_one_video(mp4: Path, txt: Path, model, imgsz: int, conf: float,
                   device: str, model_type: str) -> dict:
    """model_type: 'coco' or 'visdrone'"""
    W, H, nb = probe(mp4)
    gt_frames = load_gt(txt)
    max_gt_frame = max(gt_frames.keys()) if gt_frames else 0

    if model_type == "coco":
        pred_classes = COCO_KEEP
        gt_keep_set = set(VD_KEEP_COCO)
    else:
        pred_classes = VD_MODEL_CLASSES
        gt_keep_set = set(VD_KEEP_NATIVE)

    ious = []
    frame_records = []  # (frame_idx, iou, pred_area, gt_area)
    n_pred_only = 0  # gt=0 but pred>0
    n_gt_only = 0    # pred=0 but gt>0
    n_both_empty = 0
    n_both = 0

    kwargs = dict(stream=True, verbose=False, classes=pred_classes,
                  conf=conf, imgsz=imgsz, device=device)
    for i, result in enumerate(model.predict(str(mp4), **kwargs)):
        frame_idx = i + 1  # 与 GT 1-indexed 对齐
        # 预测框
        pred_boxes = []
        for xyxy in result.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = xyxy
            pred_boxes.append((x1, y1, x2 - x1, y2 - y1))
        pred_mask = boxes_to_mask(pred_boxes, W, H)

        # GT 框（只保留可映射的 VisDrone 类）
        gt_boxes = [(l, t, w, h) for (l, t, w, h, cat) in gt_frames.get(frame_idx, [])
                    if cat in gt_keep_set]
        gt_mask = boxes_to_mask(gt_boxes, W, H)

        iou = frame_iou(pred_mask, gt_mask)
        if iou is None:
            n_both_empty += 1
        else:
            ious.append(iou)
            if pred_mask.any() and not gt_mask.any():
                n_pred_only += 1
            elif gt_mask.any() and not pred_mask.any():
                n_gt_only += 1
            else:
                n_both += 1
        frame_records.append({
            "frame": frame_idx, "iou": iou,
            "pred_pixels": int(pred_mask.sum()),
            "gt_pixels": int(gt_mask.sum()),
        })

    return {
        "video": mp4.name,
        "resolution": [W, H],
        "n_frames_processed": len(frame_records),
        "n_frames_annotated": max_gt_frame,
        "miou": float(np.mean(ious)) if ious else None,
        "n_frames_scored": len(ious),
        "n_both_empty": n_both_empty,
        "n_both_nonempty": n_both,
        "n_pred_only": n_pred_only,
        "n_gt_only": n_gt_only,
        "frame_records": frame_records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # 默认值 = 2026-09-15 v6 sweep 独立扫参最优（test-dev 10 videos）
    ap.add_argument("--model", default="ckpts_yolo/v9e/best.pt")
    ap.add_argument("--model_type", choices=["coco", "visdrone"], default="visdrone",
                    help="coco: 8 类 COCO 模型（yolov8*.pt），需类别映射；"
                         "visdrone: 11 类 VisDrone 微调模型，直接对齐 GT")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="test-dev",
                    help="逗号分隔：test-dev,val,train")
    ap.add_argument("--out", type=Path, default=Path("eval_baseline"))
    ap.add_argument("--limit", type=int, default=0, help="每个 split 最多 N 个视频（调试用）")
    args = ap.parse_args()

    splits = set(args.split.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    videos = []
    for d in ("M01", "M02", "M07", "M08"):
        # 视频用干净版（ROOT/<d>/*.mp4），GT txt 从 boxed/ 子目录读
        for mp4 in sorted((ROOT / d).glob("*.mp4")):
            split = mp4.stem.split("_", 1)[0]
            if split in splits:
                txt = ROOT / d / "boxed" / f"{mp4.stem}.txt"
                if txt.is_file():
                    videos.append((d, split, mp4, txt))
    if args.limit:
        videos = videos[:args.limit]
    print(f"[plan] {len(videos)} videos: splits={sorted(splits)}")

    all_results = []
    t0 = time.monotonic()
    for i, (subset, split, mp4, txt) in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] {subset}/{split}/{mp4.stem}")
        r = eval_one_video(mp4, txt, model, args.imgsz, args.conf, args.device,
                           args.model_type)
        r["subset"] = subset
        r["split"] = split
        all_results.append(r)
        miou = r["miou"]
        miou_s = f"{miou:.4f}" if miou is not None else "n/a"
        print(f"    miou={miou_s} scored={r['n_frames_scored']}/{r['n_frames_processed']}  "
              f"both={r['n_both_nonempty']} pred_only={r['n_pred_only']} gt_only={r['n_gt_only']}")
    dt = time.monotonic() - t0

    # 聚合
    all_ious = []
    for r in all_results:
        for fr in r["frame_records"]:
            if fr["iou"] is not None:
                all_ious.append(fr["iou"])
    overall_miou = float(np.mean(all_ious)) if all_ious else None
    per_video = {r["video"]: r["miou"] for r in all_results}

    summary = {
        "config": {
            "model": args.model, "model_type": args.model_type,
            "imgsz": args.imgsz, "conf": args.conf,
            "device": args.device, "splits": sorted(splits),
        },
        "overall_miou_frame_avg": overall_miou,
        "n_videos": len(all_results),
        "n_frames_scored": len(all_ious),
        "wall_seconds": round(dt, 1),
        "per_video_miou": per_video,
    }
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # 详细每帧 → 单独文件
    with open(args.out / "per_video_details.json", "w") as f:
        # 去掉 frame_records 以外的元数据都保留
        json.dump([{k: v for k, v in r.items()} for r in all_results],
                  f, ensure_ascii=False)
    # 简 csv：每视频 miou
    with open(args.out / "per_video.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "split", "video", "n_frames_scored", "miou"])
        for r in all_results:
            w.writerow([r["subset"], r["split"], r["video"],
                        r["n_frames_scored"], r["miou"]])
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
