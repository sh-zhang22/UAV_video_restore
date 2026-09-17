#!/usr/bin/env python3
"""compress_restore diff 可视化：以 orig YOLO 为参照，标出压缩/修复后的框级差异。

布局（1×3 水平拼图）：
    ┌────────────┬────────────┬────────────┐
    │  orig      │ compressed │  restored  │
    │  绿=YOLO   │  diff vs orig│  diff vs orig│
    └────────────┴────────────┴────────────┘

差值颜色规则（compressed / restored 面板）：
    绿   保住的框：与 orig 某个框 IoU≥THR，画当前面板的框
    红   幻觉框：与 orig 所有框 IoU<THR（新增）
    黄虚 丢失的框：orig 有但当前面板没匹配上，画 orig 的框（虚线）

HUD:
    orig       : boxes={n}  IoU_vs_gt={x}
    compressed : keep={n}/orig hallucinate={n} miss={n}  IoU_vs_orig={x}
    restored   : keep={n}/orig hallucinate={n} miss={n}  IoU_vs_orig={x}
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
from scripts.eval.eval_yolo_visdrone import ROOT, VD_KEEP_NATIVE, load_gt, boxes_to_mask, frame_iou

MATCH_IOU = 0.5


def open_video(mp4):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {mp4}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, W, H, fps, nb


def box_iou(a, b):
    """两个 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds, refs, thr=MATCH_IOU):
    """
    对每个 pred 找最佳 ref 匹配（不允许一个 ref 被多个 pred 抢）。
    返回:
        pred_match: list[bool]，True=保住/matched
        ref_matched: set[int]，已经被匹配的 ref 索引
    """
    if not preds:
        return [], set()
    if not refs:
        return [False] * len(preds), set()
    # 贪心：按 IoU 从大到小配对
    pairs = []
    for i, p in enumerate(preds):
        for j, r in enumerate(refs):
            iou = box_iou(p, r)
            if iou >= thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    pred_match = [False] * len(preds)
    ref_matched = set()
    used_pred = set()
    for iou, i, j in pairs:
        if i in used_pred or j in ref_matched:
            continue
        used_pred.add(i)
        ref_matched.add(j)
        pred_match[i] = True
    return pred_match, ref_matched


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash=6):
    """虚线矩形（丢失框专用）。"""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_orig_panel(frame, pred_orig, gt_boxes, W, H):
    """orig 面板：只画绿色 YOLO 框，返回 IoU vs GT。"""
    for x1, y1, x2, y2 in pred_orig:
        xi1, yi1 = int(round(x1)), int(round(y1))
        xi2, yi2 = int(round(x2)), int(round(y2))
        cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (0, 255, 0), 1)
    # 顺便算 orig vs GT mIoU（mask 意义），用于 HUD
    pred_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_orig]
    pred_mask = boxes_to_mask(pred_ltwh, W, H)
    gt_ltwh = [(l, t, w, h) for (l, t, w, h) in gt_boxes]
    gt_mask = boxes_to_mask(gt_ltwh, W, H)
    return frame_iou(pred_mask, gt_mask), len(pred_orig)


def draw_diff_panel(frame, pred_cur, pred_orig, W, H):
    """
    diff 面板：以 pred_orig 为参照，pred_cur 相对它的差异。
        绿：cur 中与 orig 匹配的框（保住）
        红：cur 中未匹配 orig 的框（幻觉）
        黄虚：orig 中没被 cur 匹配的框（丢失）
    返回 (keep, hallu, miss, iou_vs_orig_mask)
    """
    cur_match, orig_matched = match_boxes(pred_cur, pred_orig, MATCH_IOU)
    keep = sum(cur_match)
    hallu = len(pred_cur) - keep
    miss = len(pred_orig) - len(orig_matched)

    # 先画丢失框（黄虚线，最底层）
    for j, (x1, y1, x2, y2) in enumerate(pred_orig):
        if j in orig_matched:
            continue
        xi1, yi1 = int(round(x1)), int(round(y1))
        xi2, yi2 = int(round(x2)), int(round(y2))
        draw_dashed_rect(frame, (xi1, yi1), (xi2, yi2), (0, 200, 255), 1, dash=6)
    # 再画 cur 的框（绿/红）
    for i, (x1, y1, x2, y2) in enumerate(pred_cur):
        xi1, yi1 = int(round(x1)), int(round(y1))
        xi2, yi2 = int(round(x2)), int(round(y2))
        color = (0, 255, 0) if cur_match[i] else (0, 0, 255)
        cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), color, 1)

    # mask IoU vs orig
    cur_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_cur]
    orig_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_orig]
    cur_mask = boxes_to_mask(cur_ltwh, W, H)
    orig_mask = boxes_to_mask(orig_ltwh, W, H)
    return keep, hallu, miss, frame_iou(cur_mask, orig_mask)


def label_panel(panel, text):
    hud_h = 44
    cv2.rectangle(panel, (0, 0), (panel.shape[1], hud_h), (0, 0, 0), -1)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(panel, line, (5, 16 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def scale_boxes(boxes, sx, sy):
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for (x1, y1, x2, y2) in boxes]


def render_3panel_diff(mp4_orig, mp4_compressed, mp4_restored,
                       yolo_orig, yolo_compressed, yolo_restored,
                       gt_frames, gt_keep_set, out_mp4, tag):
    cap_o, Wo, Ho, fps, nb_o = open_video(mp4_orig)
    cap_c, Wc, Hc, _, nb_c = open_video(mp4_compressed)
    cap_r, Wr, Hr, _, nb_r = open_video(mp4_restored)

    T = min(nb_o, nb_c, nb_r,
            len(yolo_orig), len(yolo_compressed), len(yolo_restored))
    print(f"    render T={T}  orig={Wo}x{Ho} restored={Wr}x{Hr}")

    # restored resize + boxes 缩放
    sx_r, sy_r = Wo / Wr, Ho / Hr
    yolo_restored_scaled = [scale_boxes(f, sx_r, sy_r) for f in yolo_restored]

    gap = 4
    W_full = Wo * 3 + gap * 2
    H_full = Ho
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

    running = {"orig_gt": [], "c_orig": [], "r_orig": []}
    tot = {"keep_c": 0, "hallu_c": 0, "miss_c": 0,
           "keep_r": 0, "hallu_r": 0, "miss_r": 0,
           "n_orig": 0, "n_c": 0, "n_r": 0}

    for i in range(T):
        ok_o, fr_o = cap_o.read()
        ok_c, fr_c = cap_c.read()
        ok_r, fr_r = cap_r.read()
        if not (ok_o and ok_c and ok_r):
            break
        fidx = i + 1

        # restored resize 回 orig 尺寸
        fr_r_up = cv2.resize(fr_r, (Wo, Ho), interpolation=cv2.INTER_LINEAR)

        gt_here_all = gt_frames.get(fidx, [])
        gt_here = [(l, t, w, h) for (l, t, w, h, cat) in gt_here_all if cat in gt_keep_set]

        # orig 面板
        iou_o_gt, n_o = draw_orig_panel(fr_o, yolo_orig[i], gt_here, Wo, Ho)
        # compressed diff 面板（vs orig）
        keep_c, hallu_c, miss_c, iou_c = draw_diff_panel(
            fr_c, yolo_compressed[i], yolo_orig[i], Wo, Ho)
        # restored diff 面板（vs orig）
        keep_r, hallu_r, miss_r, iou_r = draw_diff_panel(
            fr_r_up, yolo_restored_scaled[i], yolo_orig[i], Wo, Ho)

        if iou_o_gt is not None: running["orig_gt"].append(iou_o_gt)
        if iou_c is not None: running["c_orig"].append(iou_c)
        if iou_r is not None: running["r_orig"].append(iou_r)

        tot["keep_c"] += keep_c; tot["hallu_c"] += hallu_c; tot["miss_c"] += miss_c
        tot["keep_r"] += keep_r; tot["hallu_r"] += hallu_r; tot["miss_r"] += miss_r
        tot["n_orig"] += n_o
        tot["n_c"] += len(yolo_compressed[i])
        tot["n_r"] += len(yolo_restored_scaled[i])

        avg_ogt = np.mean(running["orig_gt"]) if running["orig_gt"] else 0
        avg_co = np.mean(running["c_orig"]) if running["c_orig"] else 0
        avg_ro = np.mean(running["r_orig"]) if running["r_orig"] else 0

        iou_o_str = f"{iou_o_gt:.3f}" if iou_o_gt is not None else "n/a"
        iou_c_str = f"{iou_c:.3f}" if iou_c is not None else "n/a"
        iou_r_str = f"{iou_r:.3f}" if iou_r is not None else "n/a"

        label_panel(fr_o,
            f"orig [{fidx}/{T}]  YOLO boxes={n_o}\n"
            f"IoU vs GT={iou_o_str}  avg={avg_ogt:.3f}")
        label_panel(fr_c,
            f"compressed ({tag})  keep={keep_c} hallu={hallu_c} miss={miss_c}\n"
            f"IoU vs orig={iou_c_str}  avg={avg_co:.3f}")
        label_panel(fr_r_up,
            f"restored (SeedVR2-3B)  keep={keep_r} hallu={hallu_r} miss={miss_r}\n"
            f"IoU vs orig={iou_r_str}  avg={avg_ro:.3f}")

        # 色标（画在 orig 面板底部第 1 行）
        cv2.rectangle(fr_o, (0, Ho - 22), (Wo, Ho), (0, 0, 0), -1)
        cv2.putText(fr_o,
            "green=YOLO(orig)   green=keep  red=hallucinate  yellow-dashed=miss",
            (5, Ho - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        canvas = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        canvas[:, :Wo] = fr_o
        canvas[:, Wo + gap:Wo * 2 + gap] = fr_c
        canvas[:, Wo * 2 + gap * 2:] = fr_r_up
        ff.stdin.write(canvas.tobytes())

    cap_o.release(); cap_c.release(); cap_r.release()
    ff.stdin.close(); ff.wait()

    return {
        "T": T,
        "avg_iou_orig_vs_gt": float(np.mean(running["orig_gt"])) if running["orig_gt"] else None,
        "avg_iou_compressed_vs_orig": float(np.mean(running["c_orig"])) if running["c_orig"] else None,
        "avg_iou_restored_vs_orig": float(np.mean(running["r_orig"])) if running["r_orig"] else None,
        "totals": tot,
        "keep_rate_compressed": tot["keep_c"] / max(tot["n_orig"], 1),
        "keep_rate_restored": tot["keep_r"] / max(tot["n_orig"], 1),
        "hallu_rate_compressed": tot["hallu_c"] / max(tot["n_c"], 1),
        "hallu_rate_restored": tot["hallu_r"] / max(tot["n_r"], 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="eval_compress_restore")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--videos", nargs="+", default=None)
    ap.add_argument("--out_dir", default="vis_compress_restore_diff")
    args = ap.parse_args()

    eval_base = Path(args.eval_dir) / args.tag
    if not eval_base.is_dir():
        raise SystemExit(f"eval dir not found: {eval_base}")
    out_base = Path(args.out_dir) / args.tag
    out_base.mkdir(parents=True, exist_ok=True)

    stems = []
    for d in sorted(eval_base.iterdir()):
        if not d.is_dir():
            continue
        stem = d.name
        if args.videos and stem not in args.videos:
            continue
        needed = ["compressed.mp4", "restored.mp4",
                  "yolo_orig.json", "yolo_compressed.json", "yolo_restored.json"]
        if any(not (d / n).is_file() for n in needed):
            continue
        stems.append(stem)
    if not stems:
        raise SystemExit("no complete videos")

    def find_orig(stem):
        for d in ("M01", "M02", "M07", "M08"):
            p = ROOT / d / f"{stem}.mp4"
            if p.is_file():
                return p, ROOT / d / "boxed" / f"{stem}.txt"
        raise FileNotFoundError(stem)

    gt_keep_set = set(VD_KEEP_NATIVE)
    all_stats = []
    for i, stem in enumerate(stems):
        print(f"[{i+1}/{len(stems)}] {stem}")
        d = eval_base / stem
        mp4_orig, txt_gt = find_orig(stem)
        yolo_orig = json.loads((d / "yolo_orig.json").read_text())
        yolo_c = json.loads((d / "yolo_compressed.json").read_text())
        yolo_r = json.loads((d / "yolo_restored.json").read_text())
        gt_frames = load_gt(txt_gt)
        out_mp4 = out_base / f"{stem}_{args.tag}_diff.mp4"
        stats = render_3panel_diff(
            mp4_orig, d / "compressed.mp4", d / "restored.mp4",
            yolo_orig, yolo_c, yolo_r,
            gt_frames, gt_keep_set, out_mp4, args.tag)
        stats["video"] = stem
        all_stats.append(stats)
        print(f"    ✓ {out_mp4}\n"
              f"    orig/GT={stats['avg_iou_orig_vs_gt']:.4f}  "
              f"cmp/orig={stats['avg_iou_compressed_vs_orig']:.4f}  "
              f"res/orig={stats['avg_iou_restored_vs_orig']:.4f}\n"
              f"    keep_rate cmp={stats['keep_rate_compressed']:.3f} "
              f"res={stats['keep_rate_restored']:.3f}  "
              f"hallu_rate cmp={stats['hallu_rate_compressed']:.3f} "
              f"res={stats['hallu_rate_restored']:.3f}")

    (out_base / "vis_diff_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\nsaved to {out_base}/")


if __name__ == "__main__":
    main()
