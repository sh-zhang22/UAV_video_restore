#!/usr/bin/env python3
"""在 VisDrone-VID-slices 视频上叠加 YOLO+ByteTrack+插值 后的 pred (绿) + GT (红)。

流程：
  Pass 1: model.track() 收 track_history（tid -> [(fidx, xyxy), ...]）
  Pass 2: 用 predict() 拿每帧原始 boxes（tracker 会剪框，所以要单独拿）
  Pass 3: 在原始 boxes 之上按 track_history 做 gap 插值补齐
  Pass 4: 逐帧渲染 mask + 框 + HUD（每帧标注是否含插值补齐框）
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
    ROOT, COCO_KEEP, VD_KEEP_COCO,
    VD_MODEL_CLASSES, VD_KEEP_NATIVE, load_gt, boxes_to_mask, frame_iou,
)

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


def collect_predict_and_track(mp4: Path, model, imgsz: int, conf: float,
                               device: str, model_type: str, tracker: str):
    """跑两遍：predict 拿全部原始 boxes（不被 tracker 过滤），track 拿 track_history。"""
    if model_type == "coco":
        pred_classes = COCO_KEEP
    else:
        pred_classes = VD_MODEL_CLASSES

    # Pass 1: track → track_history（放在 predict 前 warmup predictor，与 eval 对齐）
    track_history = defaultdict(list)
    for i, r in enumerate(model.track(str(mp4), stream=True, verbose=False,
                                        classes=pred_classes, conf=conf,
                                        imgsz=imgsz, device=device,
                                        tracker=tracker, persist=False)):
        fidx = i + 1
        if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
            for xy, tid in zip(r.boxes.xyxy.cpu().tolist(),
                                r.boxes.id.cpu().tolist()):
                track_history[int(tid)].append((fidx, list(xy)))

    # Pass 2: predict → 每帧原始 boxes + cls + conf（现在是 warm state）
    pred_frames = []  # list of list of (xyxy, cls, conf)
    for r in model.predict(str(mp4), stream=True, verbose=False,
                            classes=pred_classes, conf=conf,
                            imgsz=imgsz, device=device):
        cur = []
        if r.boxes is not None and len(r.boxes) > 0:
            for xy, c, cf in zip(
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.cls.cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
            ):
                cur.append((list(xy), int(c), float(cf)))
        pred_frames.append(cur)
    return pred_frames, track_history


def interp_missing(pred_frames: list, track_history: dict, max_gap: int,
                   min_track_len: int = 1):
    """返回 filled_frames：每帧含 (xyxy, cls_or_None, conf_or_None, is_interp)。
    is_interp=True 表示这一框是插值补齐的（cls/conf=None）。"""
    filled = []
    for cur in pred_frames:
        # 原始 detection 标记 is_interp=False
        filled.append([(xy, c, cf, False) for (xy, c, cf) in cur])

    n_added = 0
    for tid, seq in track_history.items():
        if len(seq) < min_track_len:
            continue
        for k in range(len(seq) - 1):
            f_a, b_a = seq[k]
            f_b, b_b = seq[k + 1]
            gap = f_b - f_a
            if gap <= 1 or gap > max_gap:
                continue
            for step in range(1, gap):
                alpha = step / gap
                interp = [b_a[c] + alpha * (b_b[c] - b_a[c]) for c in range(4)]
                filled[f_a - 1 + step].append((interp, None, None, True))
                n_added += 1
    return filled, n_added


def render_video(mp4: Path, txt: Path, filled_frames: list, out_mp4: Path,
                 model_type: str, mask_alpha: float = 0.35):
    cap, W, H, fps, nb = open_video(mp4)
    gt_frames = load_gt(txt)

    if model_type == "coco":
        gt_keep_set = set(VD_KEEP_COCO)
    else:
        gt_keep_set = set(VD_KEEP_NATIVE)

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

    frame_idx = 0
    scored_ious = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # pred boxes → mask + 记录以画框
        pred_boxes_ltwh = []
        pred_labeled = []
        n_interp = 0
        if frame_idx - 1 < len(filled_frames):
            for xy, c, cf, is_interp in filled_frames[frame_idx - 1]:
                x1, y1, x2, y2 = xy
                pred_boxes_ltwh.append((x1, y1, x2 - x1, y2 - y1))
                pred_labeled.append((x1, y1, x2, y2, c, cf, is_interp))
                if is_interp:
                    n_interp += 1
        pred_mask = boxes_to_mask(pred_boxes_ltwh, W, H)

        gt_here = [(l, t, w, h, cat) for (l, t, w, h, cat) in gt_frames.get(frame_idx, [])
                   if cat in gt_keep_set]
        gt_boxes_ltwh = [(l, t, w, h) for (l, t, w, h, _) in gt_here]
        gt_mask = boxes_to_mask(gt_boxes_ltwh, W, H)

        iou = frame_iou(pred_mask, gt_mask)
        if iou is not None:
            scored_ious.append(iou)

        overlay = frame.copy()
        overlay[gt_mask == 1] = (0, 0, 200)
        cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)
        overlay = frame.copy()
        overlay[pred_mask == 1] = (0, 200, 0)
        cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)

        # GT 框（红）
        for l, t, w, h, cat in gt_here:
            x1, y1 = int(round(l)), int(round(t))
            x2, y2 = int(round(l + w)), int(round(t + h))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, VD_NAMES.get(cat, str(cat)), (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # 预测框：非插值绿，插值青（区分开）
        for x1, y1, x2, y2, cls_id, cf, is_interp in pred_labeled:
            xi1, yi1 = int(round(x1)), int(round(y1))
            xi2, yi2 = int(round(x2)), int(round(y2))
            color = (255, 200, 0) if is_interp else (0, 255, 0)  # 青色 vs 绿色
            cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), color, 1)
            if is_interp:
                label = "interp"
            elif cls_id is not None:
                if model_type == "visdrone":
                    name = VD_NAMES.get(cls_id + 1, str(cls_id))
                else:
                    coco_names = {0: "person", 1: "bicycle", 2: "car",
                                  3: "motor", 5: "bus", 7: "truck"}
                    name = coco_names.get(cls_id, str(cls_id))
                label = f"{name} {cf:.2f}"
            else:
                label = "?"
            cv2.putText(frame, label, (xi1, min(yi2 + 10, H - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        iou_str = f"{iou:.3f}" if iou is not None else "n/a"
        avg_iou = float(np.mean(scored_ious)) if scored_ious else 0
        hud = f"frame {frame_idx}/{nb}  IoU={iou_str}  running={avg_iou:.3f}  " \
              f"pred={len(pred_labeled)}(interp={n_interp}) gt={len(gt_here)}"
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
    ap.add_argument("--model", default="ckpts_yolo/v9e/best.pt")
    ap.add_argument("--model_type", choices=["coco", "visdrone"], default="visdrone")
    ap.add_argument("--imgsz", type=int, default=960)
    # 默认值 = 2026-09-15 v6 sweep 独立扫参最优 (mIoU=0.6892 on test-dev 10 videos)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--tracker", default="trackers/bytetrack_loose.yaml")
    ap.add_argument("--max_gap", type=int, default=4)
    ap.add_argument("--min_track_len", type=int, default=5,
                    help="track 至少出现 N 次才参与插值（同 eval_yolo_visdrone_track_interp）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", type=Path, default=Path("vis_analysis"))
    ap.add_argument("--tag", default="v9e_bt_loose_gap8_conf02")
    ap.add_argument("--videos", nargs="+", default=None)
    args = ap.parse_args()

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
        pred_frames, track_hist = collect_predict_and_track(
            mp4, model, args.imgsz, args.conf, args.device,
            args.model_type, args.tracker)
        filled, n_added = interp_missing(pred_frames, track_hist, args.max_gap,
                                          args.min_track_len)
        r = render_video(mp4, txt, filled, out_mp4, args.model_type)
        r["n_interp_boxes"] = n_added
        r["n_tracks"] = len(track_hist)
        results.append(r)
        print(f"    miou={r['miou']:.4f}  interp={n_added} tracks={r['n_tracks']}  →  {out_mp4}")
    print(f"\nall done → {out_dir}/")


if __name__ == "__main__":
    main()
