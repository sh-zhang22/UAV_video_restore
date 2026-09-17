#!/usr/bin/env python3
"""VisDrone-VID-slices 上跑 YOLO + ByteTrack，评估 mask mIoU。

思路：ByteTrack 有二次关联（拿 conf ∈ [track_low_thresh, track_high_thresh] 的
低分框救回被遮挡/漏检的目标）。相对纯 detect：
  - detect 阈值放到很低（默认 0.01），让 ByteTrack 拿到所有候选
  - tracker 通过 IoU + score fusion 决定哪些框保留
  - 只用 boxes.xyxy（等价于最终"被 tracker 保留下来的检测"）合成 mask

与 eval_yolo_visdrone.py 的差异仅在推断循环里 predict → track，其它评测逻辑复用。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from eval_yolo_visdrone import (
    ROOT, COCO_KEEP, VD_KEEP_COCO,
    VD_MODEL_CLASSES, VD_KEEP_NATIVE,
    probe, load_gt, boxes_to_mask, frame_iou,
)


def eval_one_video_track(mp4: Path, txt: Path, model, imgsz: int, conf: float,
                          device: str, model_type: str, tracker: str) -> dict:
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
    frame_records = []
    n_pred_only = 0
    n_gt_only = 0
    n_both_empty = 0
    n_both = 0
    n_tracks_seen = set()

    kwargs = dict(
        stream=True, verbose=False, classes=pred_classes,
        conf=conf, imgsz=imgsz, device=device,
        tracker=tracker, persist=False,
    )
    for i, result in enumerate(model.track(str(mp4), **kwargs)):
        frame_idx = i + 1
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for xyxy in result.boxes.xyxy.cpu().tolist():
                x1, y1, x2, y2 = xyxy
                pred_boxes.append((x1, y1, x2 - x1, y2 - y1))
            if result.boxes.id is not None:
                for tid in result.boxes.id.cpu().tolist():
                    n_tracks_seen.add(int(tid))
        pred_mask = boxes_to_mask(pred_boxes, W, H)

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
        "n_tracks": len(n_tracks_seen),
        "frame_records": frame_records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ckpts_yolo/v9e/best.pt")
    ap.add_argument("--model_type", choices=["coco", "visdrone"], default="visdrone")
    ap.add_argument("--imgsz", type=int, default=960)
    # 默认值 = 2026-09-15 v6 sweep 独立扫参最优（test-dev 10 videos）
    ap.add_argument("--conf", type=float, default=0.15,
                    help="放到很低，让 ByteTrack 二次关联拿到低分框")
    ap.add_argument("--tracker", default="trackers/bytetrack_loose.yaml",
                    help="ultralytics tracker yaml；也可传自定义路径")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="test-dev")
    ap.add_argument("--out", type=Path, default=Path("eval_baseline_v4_track"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    splits = set(args.split.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    videos = []
    for d in ("M01", "M02", "M07", "M08"):
        for mp4 in sorted((ROOT / d).glob("*.mp4")):
            split = mp4.stem.split("_", 1)[0]
            if split in splits:
                txt = ROOT / d / "boxed" / f"{mp4.stem}.txt"
                if txt.is_file():
                    videos.append((d, split, mp4, txt))
    if args.limit:
        videos = videos[:args.limit]
    print(f"[plan] {len(videos)} videos, tracker={args.tracker}, conf={args.conf}, imgsz={args.imgsz}")

    all_results = []
    t0 = time.monotonic()
    for i, (subset, split, mp4, txt) in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] {subset}/{split}/{mp4.stem}")
        r = eval_one_video_track(mp4, txt, model, args.imgsz, args.conf,
                                  args.device, args.model_type, args.tracker)
        r["subset"] = subset
        r["split"] = split
        all_results.append(r)
        miou = r["miou"]
        miou_s = f"{miou:.4f}" if miou is not None else "n/a"
        print(f"    miou={miou_s} scored={r['n_frames_scored']}/{r['n_frames_processed']}  "
              f"tracks={r['n_tracks']}  both={r['n_both_nonempty']} "
              f"pred_only={r['n_pred_only']} gt_only={r['n_gt_only']}")
    dt = time.monotonic() - t0

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
            "tracker": args.tracker,
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
    with open(args.out / "per_video_details.json", "w") as f:
        json.dump([{k: v for k, v in r.items()} for r in all_results],
                  f, ensure_ascii=False)
    with open(args.out / "per_video.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "split", "video", "n_frames_scored", "n_tracks", "miou"])
        for r in all_results:
            w.writerow([r["subset"], r["split"], r["video"],
                        r["n_frames_scored"], r["n_tracks"], r["miou"]])
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
