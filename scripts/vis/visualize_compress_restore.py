#!/usr/bin/env python3
"""compress_restore 4-panel 可视化。

四宫格布局：
    ┌───────────────┬───────────────┐
    │  orig  + YOLO │ compressed +YOLO│
    │  + GT (red)   │  + GT (red)     │
    ├───────────────┼───────────────┤
    │ restored+YOLO │  mask (gray)   │
    │  + GT (red)   │  soft mask      │
    └───────────────┴───────────────┘

三张 YOLO 面板都叠 GT 框（红）+ YOLO 框（绿）+ HUD (frame idx / IoU / boxes)。
mask 面板只显示 soft mask（灰度）+ ROI 覆盖率。

用法：
    python visualize_compress_restore.py --tag q22_46
    python visualize_compress_restore.py --tag q17_37 --videos test-dev_uav0000306_00230_v_full
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.eval_yolo_visdrone import (
    ROOT, VD_KEEP_NATIVE,
    load_gt, boxes_to_mask, frame_iou,
)

VD_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "aw-tri", 9: "bus", 10: "motor", 11: "other",
}


def open_video(mp4):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {mp4}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, W, H, fps, nb


def draw_boxes(frame, pred_xyxy, gt_ltwhc, gt_keep_set,
               W, H, mask_alpha=0.30):
    """在 frame 上先画 GT/pred mask overlay，再画框。"""
    # GT + pred mask overlay
    pred_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_xyxy]
    pred_mask = boxes_to_mask(pred_ltwh, W, H)
    gt_boxes_ltwh = [(l, t, w, h) for (l, t, w, h, cat) in gt_ltwhc
                     if cat in gt_keep_set]
    gt_mask = boxes_to_mask(gt_boxes_ltwh, W, H)
    iou = frame_iou(pred_mask, gt_mask)

    overlay = frame.copy()
    overlay[gt_mask == 1] = (0, 0, 200)
    cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)
    overlay = frame.copy()
    overlay[pred_mask == 1] = (0, 200, 0)
    cv2.addWeighted(overlay, mask_alpha, frame, 1 - mask_alpha, 0, dst=frame)

    # GT boxes (red)
    for l, t, w, h, cat in gt_ltwhc:
        if cat not in gt_keep_set:
            continue
        x1, y1 = int(round(l)), int(round(t))
        x2, y2 = int(round(l + w)), int(round(t + h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, VD_NAMES.get(cat, str(cat)),
                    (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    # pred boxes (green)
    for (x1, y1, x2, y2) in pred_xyxy:
        xi1, yi1 = int(round(x1)), int(round(y1))
        xi2, yi2 = int(round(x2)), int(round(y2))
        cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 1)
    return iou, len(pred_xyxy), len(gt_boxes_ltwh)


def label_panel(panel, title, iou, n_pred, n_gt, extra=""):
    H = panel.shape[0]
    hud_h = 22
    cv2.rectangle(panel, (0, 0), (panel.shape[1], hud_h), (0, 0, 0), -1)
    iou_str = f"{iou:.3f}" if iou is not None else "n/a"
    txt = f"{title}  IoU={iou_str}  pred={n_pred} gt={n_gt}"
    if extra:
        txt += "  " + extra
    cv2.putText(panel, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def gray_to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def scale_boxes(boxes, sx, sy):
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for (x1, y1, x2, y2) in boxes]


def render_4panel(mp4_orig, mp4_compressed, mp4_restored, mp4_mask,
                  yolo_orig, yolo_compressed, yolo_restored,
                  gt_frames, gt_keep_set, out_mp4, meta, tag):
    """4-panel 拼图输出。所有面板 resize 到 orig 尺寸。"""
    cap_o, Wo, Ho, fps, nb_o = open_video(mp4_orig)
    cap_c, Wc, Hc, _, nb_c = open_video(mp4_compressed)
    cap_r, Wr, Hr, _, nb_r = open_video(mp4_restored)
    cap_m, Wm, Hm, _, nb_m = open_video(mp4_mask)

    T = min(nb_o, nb_c, nb_r, nb_m,
            len(yolo_orig), len(yolo_compressed), len(yolo_restored))
    print(f"    render T={T} frames  orig={Wo}x{Ho} restored={Wr}x{Hr}")

    # restored 需缩到 orig 尺寸；boxes 也按比例
    sx_r, sy_r = Wo / Wr, Ho / Hr
    yolo_restored_scaled = [scale_boxes(f, sx_r, sy_r) for f in yolo_restored]

    # 4-panel 总尺寸：2Wo × 2Ho + 中缝
    gap = 4
    W_full = Wo * 2 + gap
    H_full = Ho * 2 + gap
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W_full}x{H_full}", "-r", f"{fps:.5f}",
         "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(out_mp4)],
        stdin=subprocess.PIPE,
    )

    running_iou = {"orig": [], "compressed": [], "restored": []}
    for i in range(T):
        ok_o, fr_o = cap_o.read()
        ok_c, fr_c = cap_c.read()
        ok_r, fr_r = cap_r.read()
        ok_m, fr_m = cap_m.read()
        if not (ok_o and ok_c and ok_r and ok_m):
            break
        fidx = i + 1

        # restored resize 回 orig 尺寸
        fr_r_up = cv2.resize(fr_r, (Wo, Ho), interpolation=cv2.INTER_LINEAR)
        # mask resize 到 orig 尺寸
        fr_m_up = cv2.resize(fr_m, (Wo, Ho), interpolation=cv2.INTER_LINEAR)

        gt_here = gt_frames.get(fidx, [])
        iou_o, n_po, n_g = draw_boxes(fr_o, yolo_orig[i], gt_here, gt_keep_set, Wo, Ho)
        iou_c, n_pc, _ = draw_boxes(fr_c, yolo_compressed[i], gt_here, gt_keep_set, Wo, Ho)
        iou_r, n_pr, _ = draw_boxes(fr_r_up, yolo_restored_scaled[i], gt_here,
                                     gt_keep_set, Wo, Ho)

        if iou_o is not None: running_iou["orig"].append(iou_o)
        if iou_c is not None: running_iou["compressed"].append(iou_c)
        if iou_r is not None: running_iou["restored"].append(iou_r)

        avg_o = np.mean(running_iou["orig"]) if running_iou["orig"] else 0
        avg_c = np.mean(running_iou["compressed"]) if running_iou["compressed"] else 0
        avg_r = np.mean(running_iou["restored"]) if running_iou["restored"] else 0

        label_panel(fr_o, f"orig[{fidx}/{T}]", iou_o, n_po, n_g,
                    f"avg={avg_o:.3f}")
        label_panel(fr_c, f"compressed({tag})", iou_c, n_pc, n_g,
                    f"avg={avg_c:.3f}")
        label_panel(fr_r_up, "restored (SeedVR2-3B)", iou_r, n_pr, n_g,
                    f"avg={avg_r:.3f}")

        # mask 面板：灰度→BGR，做点着色（用绿色通道显示）
        mask_gray = cv2.cvtColor(fr_m_up, cv2.COLOR_BGR2GRAY)
        mask_bgr = np.zeros_like(fr_o)
        mask_bgr[..., 1] = mask_gray  # 绿色通道显示 mask
        coverage = float(mask_gray.mean() / 255.0)
        cv2.rectangle(mask_bgr, (0, 0), (Wo, 22), (0, 0, 0), -1)
        cv2.putText(mask_bgr, f"soft ROI mask  coverage={coverage:.3f}",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 拼图
        canvas = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        canvas[:Ho, :Wo] = fr_o
        canvas[:Ho, Wo + gap:] = fr_c
        canvas[Ho + gap:, :Wo] = fr_r_up
        canvas[Ho + gap:, Wo + gap:] = mask_bgr

        ff.stdin.write(canvas.tobytes())

    cap_o.release()
    cap_c.release()
    cap_r.release()
    cap_m.release()
    ff.stdin.close()
    ff.wait()

    stats = {
        "T_rendered": T,
        "avg_iou_orig": float(np.mean(running_iou["orig"])) if running_iou["orig"] else None,
        "avg_iou_compressed": float(np.mean(running_iou["compressed"])) if running_iou["compressed"] else None,
        "avg_iou_restored": float(np.mean(running_iou["restored"])) if running_iou["restored"] else None,
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="eval_compress_restore")
    ap.add_argument("--tag", required=True, help="子目录名，例 q22_46 / q17_37")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频 stem 列表；不给则跑该 tag 下所有")
    ap.add_argument("--out_dir", default="vis_compress_restore")
    args = ap.parse_args()

    eval_base = Path(args.eval_dir) / args.tag
    if not eval_base.is_dir():
        raise SystemExit(f"eval dir not found: {eval_base}")

    out_base = Path(args.out_dir) / args.tag
    out_base.mkdir(parents=True, exist_ok=True)

    # 收集视频（从 eval 目录中的子目录）
    video_stems = []
    for d in sorted(eval_base.iterdir()):
        if not d.is_dir():
            continue
        stem = d.name
        if args.videos and stem not in args.videos:
            continue
        # 检查所需产物齐全
        needed = ["compressed.mp4", "restored.mp4", "mask.mp4",
                  "yolo_orig.json", "yolo_compressed.json", "yolo_restored.json",
                  "meta.json"]
        missing = [n for n in needed if not (d / n).is_file()]
        if missing:
            print(f"  skip {stem}: missing {missing}")
            continue
        video_stems.append(stem)
    if not video_stems:
        raise SystemExit("no complete videos found")
    print(f"[plan] {len(video_stems)} videos, tag={args.tag}")

    # 定位 orig mp4
    def find_orig(stem):
        for d in ("M01", "M02", "M07", "M08"):
            p = ROOT / d / f"{stem}.mp4"
            if p.is_file():
                return p, ROOT / d / "boxed" / f"{stem}.txt"
        raise FileNotFoundError(stem)

    gt_keep_set = set(VD_KEEP_NATIVE)
    all_stats = []
    for i, stem in enumerate(video_stems):
        print(f"[{i+1}/{len(video_stems)}] {stem}")
        d = eval_base / stem
        mp4_orig, txt_gt = find_orig(stem)
        yolo_orig = json.loads((d / "yolo_orig.json").read_text())
        yolo_compressed = json.loads((d / "yolo_compressed.json").read_text())
        yolo_restored = json.loads((d / "yolo_restored.json").read_text())
        meta = json.loads((d / "meta.json").read_text())
        gt_frames = load_gt(txt_gt)

        out_mp4 = out_base / f"{stem}_{args.tag}.mp4"
        try:
            stats = render_4panel(
                mp4_orig, d / "compressed.mp4", d / "restored.mp4", d / "mask.mp4",
                yolo_orig, yolo_compressed, yolo_restored,
                gt_frames, gt_keep_set, out_mp4, meta, args.tag)
            stats["video"] = stem
            all_stats.append(stats)
            print(f"    ✓ {out_mp4}  "
                  f"orig={stats['avg_iou_orig']:.4f} "
                  f"cmp={stats['avg_iou_compressed']:.4f} "
                  f"res={stats['avg_iou_restored']:.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    !! failed: {e}")

    (out_base / "vis_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_base}/")


if __name__ == "__main__":
    main()
