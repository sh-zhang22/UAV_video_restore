#!/usr/bin/env python3
"""compress_restore 三栏对比可视化。

┌────────────────┬──────────────────┬──────────────────┐
│  orig          │  compressed      │  restored        │
│  绿框: orig pred│  绿框: orig pred │  绿框: orig pred │
│                │  红框: cmp  pred │  红框: rst  pred │
└────────────────┴──────────────────┴──────────────────┘

restored 会 resize 回 orig 尺寸；对应 boxes 也按比例缩放。
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
from scripts.eval.eval_yolo_visdrone import ROOT


def open_video(mp4):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {mp4}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, W, H, fps, nb


def draw_boxes(frame, boxes_xyxy, color, thickness=1):
    for x1, y1, x2, y2 in boxes_xyxy:
        cv2.rectangle(frame,
                      (int(round(x1)), int(round(y1))),
                      (int(round(x2)), int(round(y2))),
                      color, thickness)


def label_panel(panel, title, extra=""):
    hud_h = 22
    cv2.rectangle(panel, (0, 0), (panel.shape[1], hud_h), (0, 0, 0), -1)
    txt = title if not extra else f"{title}  {extra}"
    cv2.putText(panel, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def scale_boxes(boxes, sx, sy):
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for (x1, y1, x2, y2) in boxes]


def find_orig(stem):
    for d in ("M01", "M02", "M07", "M08"):
        p = ROOT / d / f"{stem}.mp4"
        if p.is_file():
            return p
    raise FileNotFoundError(stem)


def render_triple(stem, eval_dir, out_mp4, tag):
    mp4_orig = find_orig(stem)
    mp4_cmp = eval_dir / "compressed.mp4"
    mp4_rst = eval_dir / "restored.mp4"
    yolo_orig = json.loads((eval_dir / "yolo_orig_ti.json").read_text())
    yolo_cmp = json.loads((eval_dir / "yolo_compressed_ti.json").read_text())
    yolo_rst = json.loads((eval_dir / "yolo_restored_ti.json").read_text())

    cap_o, Wo, Ho, fps, nb_o = open_video(mp4_orig)
    cap_c, Wc, Hc, _, nb_c = open_video(mp4_cmp)
    cap_r, Wr, Hr, _, nb_r = open_video(mp4_rst)
    T = min(nb_o, nb_c, nb_r, len(yolo_orig), len(yolo_cmp), len(yolo_rst))
    print(f"    T={T}  orig={Wo}x{Ho}  cmp={Wc}x{Hc}  rst={Wr}x{Hr}")

    # restored → orig 尺寸
    sx_r, sy_r = Wo / Wr, Ho / Hr
    yolo_rst_scaled = [scale_boxes(f, sx_r, sy_r) for f in yolo_rst]

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

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)

    for i in range(T):
        ok_o, fr_o = cap_o.read()
        ok_c, fr_c = cap_c.read()
        ok_r, fr_r = cap_r.read()
        if not (ok_o and ok_c and ok_r):
            break
        fidx = i + 1

        fr_r_up = cv2.resize(fr_r, (Wo, Ho), interpolation=cv2.INTER_LINEAR)

        # 左：orig + 绿 orig
        draw_boxes(fr_o, yolo_orig[i], GREEN)
        label_panel(fr_o, f"orig[{fidx}/{T}]",
                    f"pred={len(yolo_orig[i])}")

        # 中：compressed + 绿 orig + 红 compressed
        draw_boxes(fr_c, yolo_orig[i], GREEN)
        draw_boxes(fr_c, yolo_cmp[i], RED)
        label_panel(fr_c, f"compressed ({tag})",
                    f"orig={len(yolo_orig[i])} cmp={len(yolo_cmp[i])}")

        # 右：restored + 绿 orig + 红 restored（都在 orig 尺寸）
        draw_boxes(fr_r_up, yolo_orig[i], GREEN)
        draw_boxes(fr_r_up, yolo_rst_scaled[i], RED)
        label_panel(fr_r_up, "restored (SeedVR2-3B)",
                    f"orig={len(yolo_orig[i])} rst={len(yolo_rst[i])}")

        canvas = np.zeros((H_full, W_full, 3), dtype=np.uint8)
        canvas[:, :Wo] = fr_o
        canvas[:, Wo + gap:Wo * 2 + gap] = fr_c
        canvas[:, Wo * 2 + gap * 2:] = fr_r_up
        ff.stdin.write(canvas.tobytes())

    cap_o.release(); cap_c.release(); cap_r.release()
    ff.stdin.close()
    ff.wait()
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="eval_compress_restore")
    ap.add_argument("--tag", required=True, help="子目录名，例 q22_37")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频 stem 列表；不给则跑该 tag 下所有")
    ap.add_argument("--out_dir", default="vis_compress_restore")
    args = ap.parse_args()

    eval_base = Path(args.eval_dir) / args.tag
    if not eval_base.is_dir():
        raise SystemExit(f"eval dir not found: {eval_base}")

    out_base = Path(args.out_dir) / f"{args.tag}_triple"
    out_base.mkdir(parents=True, exist_ok=True)

    stems = []
    for d in sorted(eval_base.iterdir()):
        if not d.is_dir():
            continue
        if args.videos and d.name not in args.videos:
            continue
        needed = ["compressed.mp4", "restored.mp4",
                  "yolo_orig_ti.json", "yolo_compressed_ti.json", "yolo_restored_ti.json"]
        missing = [n for n in needed if not (d / n).is_file()]
        if missing:
            print(f"  skip {d.name}: missing {missing}")
            continue
        stems.append(d.name)
    if not stems:
        raise SystemExit("no complete videos")
    print(f"[plan] {len(stems)} videos, tag={args.tag}")

    for i, stem in enumerate(stems):
        print(f"[{i+1}/{len(stems)}] {stem}")
        out_mp4 = out_base / f"{stem}_{args.tag}_triple.mp4"
        T = render_triple(stem, eval_base / stem, out_mp4, args.tag)
        print(f"    → {out_mp4}  ({T} frames)")


if __name__ == "__main__":
    main()
