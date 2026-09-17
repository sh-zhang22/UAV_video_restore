#!/usr/bin/env python3
"""在 VisDrone-VID-slices 视频上叠加 pred (绿) + GT (红) 框和 mask，
用于肉眼诊断 baseline mIoU 差距。

输出到 vis_analysis/<config_tag>/<video_stem>.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.eval_yolo_visdrone import (
    ROOT, COCO_TO_VD, COCO_KEEP, VD_KEEP_COCO,
    VD_MODEL_CLASSES, VD_KEEP_NATIVE, load_gt, boxes_to_mask, frame_iou,
)

# 类别名
VD_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "aw-tri", 9: "bus", 10: "motor", 11: "other",
}


def open_video(mp4: Path):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {mp4}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, W, H, fps, n


def render_video(mp4: Path, txt: Path, model, imgsz: int, conf: float,
                 device: str, model_type: str, out_mp4: Path,
                 mask_alpha: float = 0.35):
    cap, W, H, fps, nb = open_video(mp4)
    gt_frames = load_gt(txt)

    if model_type == "coco":
        pred_classes = COCO_KEEP
        gt_keep_set = set(VD_KEEP_COCO)
    else:
        pred_classes = VD_MODEL_CLASSES
        gt_keep_set = set(VD_KEEP_NATIVE)

    # 用 ffmpeg pipe 收 rawvideo，避免 opencv 输出 mp4 兼容性问题
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", f"{fps:.5f}",
         "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(out_mp4)],
        stdin=subprocess.PIPE,
    )

    # 预生成所有帧的预测（stream 用 mp4）
    kwargs = dict(stream=True, verbose=False, classes=pred_classes,
                  conf=conf, imgsz=imgsz, device=device)
    preds = []  # per-frame [(x1,y1,x2,y2,cls,conf), ...]
    for res in model.predict(str(mp4), **kwargs):
        cur = []
        if res.boxes is not None and len(res.boxes) > 0:
            for xyxy, cls, cf in zip(
                res.boxes.xyxy.cpu().tolist(),
                res.boxes.cls.cpu().tolist(),
                res.boxes.conf.cpu().tolist(),
            ):
                x1, y1, x2, y2 = xyxy
                cur.append((x1, y1, x2, y2, int(cls), float(cf)))
        preds.append(cur)

    # 逐帧渲染
    frame_idx = 0
    scored_ious = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # pred → mask + boxes
        pred_boxes_ltwh = []
        pred_labeled = []
        if frame_idx - 1 < len(preds):
            for x1, y1, x2, y2, cls_id, cf in preds[frame_idx - 1]:
                pred_boxes_ltwh.append((x1, y1, x2 - x1, y2 - y1))
                pred_labeled.append((x1, y1, x2, y2, cls_id, cf))
        pred_mask = boxes_to_mask(pred_boxes_ltwh, W, H)

        # GT → mask + boxes
        gt_here = [(l, t, w, h, cat) for (l, t, w, h, cat) in gt_frames.get(frame_idx, [])
                   if cat in gt_keep_set]
        gt_boxes_ltwh = [(l, t, w, h) for (l, t, w, h, _) in gt_here]
        gt_mask = boxes_to_mask(gt_boxes_ltwh, W, H)

        iou = frame_iou(pred_mask, gt_mask)
        if iou is not None:
            scored_ious.append(iou)

        # 叠 mask（半透明）
        overlay = frame.copy()
        # GT mask → 红色通道
        overlay[gt_mask == 1] = (0, 0, 200)
        cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)
        overlay = frame.copy()
        # pred mask → 绿色通道
        overlay[pred_mask == 1] = (0, 200, 0)
        cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)

        # GT 框（红）
        for l, t, w, h, cat in gt_here:
            x1, y1 = int(round(l)), int(round(t))
            x2, y2 = int(round(l + w)), int(round(t + h))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, VD_NAMES.get(cat, str(cat)), (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # 预测框（绿）
        for x1, y1, x2, y2, cls_id, cf in pred_labeled:
            xi1, yi1 = int(round(x1)), int(round(y1))
            xi2, yi2 = int(round(x2)), int(round(y2))
            cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 1)
            if model_type == "visdrone":
                name = VD_NAMES.get(cls_id + 1, str(cls_id))
            else:
                # COCO 名字简化标记
                coco_names = {0: "person", 1: "bicycle", 2: "car",
                              3: "motor", 5: "bus", 7: "truck"}
                name = coco_names.get(cls_id, str(cls_id))
            label = f"{name} {cf:.2f}"
            cv2.putText(frame, label, (xi1, min(yi2 + 10, H - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # HUD
        iou_str = f"{iou:.3f}" if iou is not None else "n/a"
        avg_iou = float(np.mean(scored_ious)) if scored_ious else 0
        hud = f"frame {frame_idx}/{nb}  IoU={iou_str}  running={avg_iou:.3f}  " \
              f"pred={len(pred_labeled)} gt={len(gt_here)}"
        # 黑底白字确保可读
        cv2.rectangle(frame, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(frame, hud, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ff.stdin.write(frame.tobytes())

    cap.release()
    ff.stdin.close()
    ff.wait()
    return {"video": mp4.name, "n_frames": frame_idx,
            "miou": float(np.mean(scored_ious)) if scored_ious else None}


def main():
    ap = argparse.ArgumentParser()
    # 默认值 = 2026-09-15 v6 sweep 独立扫参最优（test-dev 10 videos）
    ap.add_argument("--model", default="ckpts_yolo/v9e/best.pt")
    ap.add_argument("--model_type", choices=["coco", "visdrone"], default="visdrone")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", type=Path, default=Path("vis_analysis"))
    ap.add_argument("--tag", default="visdrone_x_imgsz960",
                    help="子目录名，用于区分不同配置")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频文件名（不带 .mp4）；不给则跑 test-dev 全部")
    args = ap.parse_args()

    # 收集视频
    videos = []
    for d in ("M01", "M02", "M07", "M08"):
        for mp4 in sorted((ROOT / d).glob("*.mp4")):
            split = mp4.stem.split("_", 1)[0]
            if split != "test-dev":
                continue
            if args.videos and mp4.stem not in args.videos:
                continue
            txt = ROOT / d / "boxed" / f"{mp4.stem}.txt"
            if txt.is_file():
                videos.append((d, mp4, txt))
    if not videos:
        raise SystemExit("no videos matched")
    print(f"[plan] {len(videos)} videos, tag={args.tag}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (subset, mp4, txt) in enumerate(videos):
        out_mp4 = out_dir / f"{subset}_{mp4.stem}.mp4"
        print(f"[{i+1}/{len(videos)}] {out_mp4.name}")
        r = render_video(mp4, txt, model, args.imgsz, args.conf,
                         args.device, args.model_type, out_mp4)
        results.append(r)
        print(f"    miou={r['miou']:.4f}  →  {out_mp4}")
    print(f"\nall done → {out_dir}/")


if __name__ == "__main__":
    main()
