#!/usr/bin/env python3
"""VisDrone-VID-slices：YOLO + ByteTrack + 时序 gap 线性插值补框（Hybrid 版）。

Hybrid 流程（只加不减）：
  Pass 1: predict() → 每帧原始所有 boxes（不经 tracker 过滤，避免删除短暂目标）
  Pass 2: track()   → 只收集 track_history = {tid: [(fidx, xyxy), ...]}
  Pass 3: 对每个 track，在相邻两次出现之间的 gap（1 < gap ≤ max_gap）
          用前后 box 做线性插值补齐中间帧，追加到该帧的 predict boxes
  Pass 4: 用补齐后 per-frame boxes 生成 mask，计算 IoU
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_yolo_visdrone import (
    ROOT, COCO_KEEP, VD_KEEP_COCO,
    VD_MODEL_CLASSES, VD_KEEP_NATIVE,
    probe, load_gt, boxes_to_mask, frame_iou,
)


def eval_one_video_track_interp(mp4: Path, txt: Path, model, imgsz: int, conf: float,
                                 device: str, model_type: str, tracker: str,
                                 max_gap: int, min_track_len: int = 1,
                                 max_disp_per_frame: float = 1e9) -> dict:
    W, H, nb = probe(mp4)
    gt_frames = load_gt(txt)
    max_gt_frame = max(gt_frames.keys()) if gt_frames else 0

    if model_type == "coco":
        pred_classes = COCO_KEEP
        gt_keep_set = set(VD_KEEP_COCO)
    else:
        pred_classes = VD_MODEL_CLASSES
        gt_keep_set = set(VD_KEEP_NATIVE)

    # Pass 1: track → 只收 track_history 用于插值
    # ★ track 必须放在 predict 前面：ultralytics 的 predictor 首次调用会固化内部状态，
    #   若首次是 predict，后续 predict 与 track-warmed 状态下的 predict 差异 ~0.045 mIoU。
    #   Batch 里从第 2 个视频开始 predict 自然是 warm（前一视频 track 已 warmup），
    #   为让单视频/首视频与后续视频行为一致，改为 track-first。
    track_history: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for i, r in enumerate(model.track(str(mp4), stream=True, verbose=False,
                                       classes=pred_classes, conf=conf,
                                       imgsz=imgsz, device=device,
                                       tracker=tracker, persist=False)):
        fidx = i + 1
        if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
            for xy, tid in zip(r.boxes.xyxy.cpu().tolist(),
                                r.boxes.id.cpu().tolist()):
                track_history[int(tid)].append((fidx, list(xy)))

    # Pass 2: predict → 每帧原始所有 boxes（不被 tracker 过滤，现在是 warm state）
    frame_boxes: list[list[list[float]]] = []
    for r in model.predict(str(mp4), stream=True, verbose=False,
                            classes=pred_classes, conf=conf,
                            imgsz=imgsz, device=device):
        cur = []
        if r.boxes is not None and len(r.boxes) > 0:
            for xy in r.boxes.xyxy.cpu().tolist():
                cur.append(list(xy))
        frame_boxes.append(cur)

    n_frames = len(frame_boxes)

    # 线性插值补 gap（含三重安全阀）
    n_filled_boxes = 0
    n_rejected_disp = 0
    n_rejected_short_track = 0
    filled_boxes = [list(b) for b in frame_boxes]
    for tid, seq in track_history.items():
        if len(seq) < min_track_len:
            n_rejected_short_track += 1
            continue
        for k in range(len(seq) - 1):
            f_a, b_a = seq[k]
            f_b, b_b = seq[k + 1]
            gap = f_b - f_a
            if gap <= 1 or gap > max_gap:
                continue
            # 位移过大 → 拒绝插值（可能是不同目标被误关联）
            ca_x = (b_a[0] + b_a[2]) / 2
            ca_y = (b_a[1] + b_a[3]) / 2
            cb_x = (b_b[0] + b_b[2]) / 2
            cb_y = (b_b[1] + b_b[3]) / 2
            disp = ((ca_x - cb_x) ** 2 + (ca_y - cb_y) ** 2) ** 0.5
            if disp / gap > max_disp_per_frame:
                n_rejected_disp += 1
                continue
            for step in range(1, gap):
                alpha = step / gap
                interp = [b_a[c] + alpha * (b_b[c] - b_a[c]) for c in range(4)]
                filled_boxes[f_a - 1 + step].append(interp)
                n_filled_boxes += 1

    # 算 mask IoU
    ious = []
    frame_records = []
    n_pred_only = 0
    n_gt_only = 0
    n_both_empty = 0
    n_both = 0
    for i, boxes_xyxy in enumerate(filled_boxes):
        fidx = i + 1
        pred_boxes_ltwh = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes_xyxy]
        pred_mask = boxes_to_mask(pred_boxes_ltwh, W, H)

        gt_boxes = [(l, t, w, h) for (l, t, w, h, cat) in gt_frames.get(fidx, [])
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
            "frame": fidx, "iou": iou,
            "pred_pixels": int(pred_mask.sum()),
            "gt_pixels": int(gt_mask.sum()),
        })

    return {
        "video": mp4.name,
        "resolution": [W, H],
        "n_frames_processed": n_frames,
        "n_frames_annotated": max_gt_frame,
        "miou": float(np.mean(ious)) if ious else None,
        "n_frames_scored": len(ious),
        "n_both_empty": n_both_empty,
        "n_both_nonempty": n_both,
        "n_pred_only": n_pred_only,
        "n_gt_only": n_gt_only,
        "n_tracks": len(track_history),
        "n_filled_boxes": n_filled_boxes,
        "n_rejected_short_track": n_rejected_short_track,
        "n_rejected_disp": n_rejected_disp,
        "frame_records": frame_records,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ckpts_yolo/v9e/best.pt")
    ap.add_argument("--model_type", choices=["coco", "visdrone"], default="visdrone")
    ap.add_argument("--imgsz", type=int, default=960)
    # 默认值 = 2026-09-15 v6 sweep 独立扫参最优拼接 config
    # (test-dev 10 videos, v9e VisDrone-finetuned, imgsz=960): mIoU=0.6892
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--tracker", default="trackers/bytetrack_loose.yaml")
    ap.add_argument("--max_gap", type=int, default=4,
                    help="≤ 该 gap 帧数才做插值补齐；过大会引入更多假框")
    ap.add_argument("--min_track_len", type=int, default=5,
                    help="track 至少出现 N 次才参与插值，过滤虚警短 track")
    ap.add_argument("--max_disp_per_frame", type=float, default=1e9,
                    help="track 相邻两帧中心位移 / gap > 此值则拒绝插值（像素/帧）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="test-dev")
    ap.add_argument("--out", type=Path, default=Path("eval_baseline_v4_track_interp"))
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
    print(f"[plan] {len(videos)} videos, tracker={args.tracker}, conf={args.conf}, "
          f"imgsz={args.imgsz}, max_gap={args.max_gap}")

    all_results = []
    t0 = time.monotonic()
    for i, (subset, split, mp4, txt) in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] {subset}/{split}/{mp4.stem}")
        r = eval_one_video_track_interp(mp4, txt, model, args.imgsz, args.conf,
                                         args.device, args.model_type, args.tracker,
                                         args.max_gap, args.min_track_len,
                                         args.max_disp_per_frame)
        r["subset"] = subset
        r["split"] = split
        all_results.append(r)
        miou = r["miou"]
        miou_s = f"{miou:.4f}" if miou is not None else "n/a"
        print(f"    miou={miou_s} tracks={r['n_tracks']} filled={r['n_filled_boxes']}")
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
            "tracker": args.tracker, "max_gap": args.max_gap,
            "min_track_len": args.min_track_len,
            "max_disp_per_frame": args.max_disp_per_frame,
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
        w.writerow(["subset", "split", "video", "n_frames_scored",
                    "n_tracks", "n_filled_boxes", "miou"])
        for r in all_results:
            w.writerow([r["subset"], r["split"], r["video"],
                        r["n_frames_scored"], r["n_tracks"],
                        r["n_filled_boxes"], r["miou"]])
    print(f"\nsaved to {args.out}/")


if __name__ == "__main__":
    main()
