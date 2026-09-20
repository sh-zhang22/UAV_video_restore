#!/usr/bin/env python3
"""compressed / restored 相对 orig YOLO 检测的**像素级**诊断可视化（两栏）。

两栏水平拼接（左 compressed / 右 restored）。每帧把 orig 框列表和 panel 框列表
分别栅格化为二值 mask，逐像素三分类：

    - 绿（半透明）：TP —— orig_mask ∧ panel_mask（两边都覆盖）
    - 黄（半透明）：FN —— orig_mask ∧ ¬panel_mask（漏检）
    - 红（半透明）：FP —— ¬orig_mask ∧ panel_mask（虚警）

HUD 每栏显示：TP / FN / FP 像素数 + pixel IoU = TP / (TP + FN + FP)

用法：
    python scripts/vis/visualize_compress_restore_match.py --tag q22_37
    python scripts/vis/visualize_compress_restore_match.py --tag q22_37 --videos test-dev_uav0000306_00230_v_full
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
from scripts.eval.eval_yolo_visdrone import ROOT  # noqa: F401  (保留，可能其它脚本依赖)

ALPHA = 0.4   # 半透明填充强度


def orig_to_panel(bx, Wo, Ho, Wp, Hp):
    """orig 空间下的 xyxy 框列表 → panel 空间（compressed / restored）。

    - (Wp, Hp) == (Wo, Ho)：identity（compressed）
    - 否则走 SeedVR pipeline: NaResize(area, downsample_only=False) → DivisibleCrop((16,16), center)
      本项目 eval_compress_restore 传 res_h=Ho, res_w=Wo，故 max_area=Ho*Wo → scale=1，
      仅 center_crop 起作用。若尺寸不匹配（比如换了 res_h/res_w），退化到 scale-based。
    """
    if (Wp, Hp) == (Wo, Ho):
        return [list(b) for b in bx]
    import math
    max_area = Ho * Wo
    scale = math.sqrt(max_area / (Ho * Wo))  # = 1（保留通用形式）
    H1, W1 = round(Ho * scale), round(Wo * scale)
    Hp_expected = H1 - (H1 % 16)
    Wp_expected = W1 - (W1 % 16)
    if (Hp_expected, Wp_expected) != (Hp, Wp):
        sx, sy = Wp / Wo, Hp / Ho
        return [[x1*sx, y1*sy, x2*sx, y2*sy] for x1, y1, x2, y2 in bx]
    crop_left = (W1 - Wp) // 2
    crop_top = (H1 - Hp) // 2
    return [[x1*scale - crop_left, y1*scale - crop_top,
             x2*scale - crop_left, y2*scale - crop_top] for x1, y1, x2, y2 in bx]


def open_video(mp4):
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {mp4}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, W, H, fps, nb


def boxes_to_mask_xyxy(boxes, W, H):
    """xyxy 框列表 → (H, W) bool mask。"""
    m = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xi1 = max(0, int(round(x1)))
        yi1 = max(0, int(round(y1)))
        xi2 = min(W, int(round(x2)))
        yi2 = min(H, int(round(y2)))
        if xi2 > xi1 and yi2 > yi1:
            m[yi1:yi2, xi1:xi2] = True
    return m


def fill_mask(canvas, mask, color_bgr, alpha):
    """在 canvas 上按 mask=True 的像素半透明叠色。"""
    if not mask.any():
        return
    overlay = canvas.copy()
    overlay[mask] = color_bgr
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


def render_panel(frame, orig_in_panel, panel_boxes):
    """像素级三分类叠色。
    绿=TP (orig ∧ panel), 黄=FN (orig ∧ ¬panel, 漏检), 红=FP (¬orig ∧ panel, 虚警)。
    返回 (tp_px, fn_px, fp_px, iou)，其中 iou = tp/(tp+fn+fp) 或 None（分母 0）。
    """
    H, W = frame.shape[:2]
    m_o = boxes_to_mask_xyxy(orig_in_panel, W, H)
    m_p = boxes_to_mask_xyxy(panel_boxes, W, H)

    tp = m_o & m_p          # 绿
    fn = m_o & ~m_p         # 黄：orig 有 panel 无
    fp = ~m_o & m_p         # 红：panel 有 orig 无

    fill_mask(frame, tp, (0, 200, 0), ALPHA)
    fill_mask(frame, fn, (0, 220, 220), ALPHA)
    fill_mask(frame, fp, (0, 0, 220), ALPHA)

    tp_n = int(tp.sum()); fn_n = int(fn.sum()); fp_n = int(fp.sum())
    denom = tp_n + fn_n + fp_n
    iou = tp_n / denom if denom > 0 else None
    return tp_n, fn_n, fp_n, iou


def draw_hud(frame, W, H, title, tp_px, fn_px, fp_px, iou_val):
    cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
    cv2.putText(frame, title, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    iou_s = f"{iou_val:.3f}" if iou_val is not None else "n/a"
    line2 = f"TP={tp_px}  FN={fn_px}  FP={fp_px}  IoU={iou_s}"
    cv2.putText(frame, line2, (5, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)


def process_video(tag, video_stem, out_dir, fps_scale=1.0):
    q_dir = Path("eval_compress_restore") / tag / video_stem
    if not q_dir.is_dir():
        raise SystemExit(f"missing eval dir: {q_dir}")

    cmp_mp4 = q_dir / "compressed.mp4"
    rst_mp4 = q_dir / "restored.mp4"
    orig_json = q_dir / "yolo_orig_ti.json"
    cmp_json = q_dir / "yolo_compressed_ti.json"
    rst_json = q_dir / "yolo_restored_ti.json"

    for p in (cmp_mp4, rst_mp4, orig_json, cmp_json, rst_json):
        if not p.is_file():
            raise SystemExit(f"missing: {p}")

    pred_orig = json.loads(orig_json.read_text())
    pred_cmp = json.loads(cmp_json.read_text())
    pred_rst = json.loads(rst_json.read_text())

    cap_c, Wc, Hc, fps_c, nb_c = open_video(cmp_mp4)
    cap_r, Wr, Hr, fps_r, nb_r = open_video(rst_mp4)
    # orig 空间来源：pred_orig 是在 orig mp4 分辨率下的框
    meta = json.loads((q_dir / "meta.json").read_text())
    Wo, Ho = meta["resolution"]

    # 每栏 pad 到统一高度；两栏尺寸保持各自视频原生分辨率
    Hout = max(Hc, Hr)
    Wsep = 4
    out_W = Wc + Wsep + Wr
    out_H = Hout

    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{tag}__{video_stem}__match.mp4"

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_W}x{out_H}", "-r", f"{fps_c:.5f}",
         "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-pix_fmt", "yuv420p", str(out_mp4)],
        stdin=subprocess.PIPE,
    )

    T = min(len(pred_orig), len(pred_cmp), len(pred_rst), nb_c, nb_r)
    print(f"  [{video_stem}] T={T}  orig={Wo}x{Ho}  cmp={Wc}x{Hc}  rst={Wr}x{Hr}")

    running_cmp_ious = []
    running_rst_ious = []
    stats = {"cmp": {"tp":0, "fn":0, "fp":0},
             "rst": {"tp":0, "fn":0, "fp":0}}

    for fidx in range(T):
        ok_c, fc = cap_c.read()
        ok_r, fr = cap_r.read()
        if not ok_c or not ok_r:
            break

        orig_boxes = pred_orig[fidx]  # orig 空间 (Wo, Ho)
        cmp_boxes = pred_cmp[fidx]    # compressed 空间 (Wc, Hc)
        rst_boxes = pred_rst[fidx]    # restored 空间 (Wr, Hr)

        # 把 orig 框分别变换到 compressed / restored 空间（SeedVR 的 crop 逆变换）
        orig_in_cmp = orig_to_panel(orig_boxes, Wo, Ho, Wc, Hc)
        orig_in_rst = orig_to_panel(orig_boxes, Wo, Ho, Wr, Hr)

        # 像素级三分类叠色 + IoU（render_panel 内部一次搞定）
        tp_c, fn_c, fp_c, iou_c = render_panel(fc, orig_in_cmp, cmp_boxes)
        tp_r, fn_r, fp_r, iou_r = render_panel(fr, orig_in_rst, rst_boxes)
        if iou_c is not None: running_cmp_ious.append(iou_c)
        if iou_r is not None: running_rst_ious.append(iou_r)

        stats["cmp"]["tp"] += tp_c; stats["cmp"]["fn"] += fn_c; stats["cmp"]["fp"] += fp_c
        stats["rst"]["tp"] += tp_r; stats["rst"]["fn"] += fn_r; stats["rst"]["fp"] += fp_r

        # HUD
        draw_hud(fc, Wc, Hc, f"compressed  f={fidx+1}/{T}",
                 tp_c, fn_c, fp_c, iou_c)
        draw_hud(fr, Wr, Hr, f"restored  f={fidx+1}/{T}",
                 tp_r, fn_r, fp_r, iou_r)

        # 拼图（pad 到统一高度）
        def _pad(img, H):
            h, w = img.shape[:2]
            if h == H: return img
            out = np.zeros((H, w, 3), dtype=np.uint8)
            out[:h] = img
            return out

        canvas = np.zeros((out_H, out_W, 3), dtype=np.uint8)
        canvas[:, :Wc] = _pad(fc, out_H)
        canvas[:, Wc+Wsep:] = _pad(fr, out_H)
        # 分隔线
        canvas[:, Wc:Wc+Wsep] = 60

        ff.stdin.write(canvas.tobytes())

    cap_c.release(); cap_r.release()
    ff.stdin.close(); ff.wait()

    def _agg_iou(s):
        denom = s['tp'] + s['fn'] + s['fp']
        return s['tp'] / denom if denom > 0 else float('nan')

    print(f"    ✓ {out_mp4}")
    print(f"    cmp: TP={stats['cmp']['tp']} FN={stats['cmp']['fn']} FP={stats['cmp']['fp']}  "
          f"agg IoU={_agg_iou(stats['cmp']):.4f}  frame mean={np.mean(running_cmp_ious):.4f}")
    print(f"    rst: TP={stats['rst']['tp']} FN={stats['rst']['fn']} FP={stats['rst']['fp']}  "
          f"agg IoU={_agg_iou(stats['rst']):.4f}  frame mean={np.mean(running_rst_ious):.4f}")
    return out_mp4, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="eval_compress_restore/ 下的 tag 目录")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频 stem 列表；不给则跑该 tag 下所有子目录")
    ap.add_argument("--out_dir", type=Path, default=Path("vis_compress_restore_match"))
    args = ap.parse_args()

    tag_dir = Path("eval_compress_restore") / args.tag
    if not tag_dir.is_dir():
        raise SystemExit(f"missing {tag_dir}")

    videos = []
    for d in sorted(tag_dir.iterdir()):
        if not d.is_dir(): continue
        if args.videos and d.name not in args.videos:
            continue
        videos.append(d.name)
    if not videos:
        raise SystemExit("no videos matched")

    out_dir = args.out_dir / args.tag
    print(f"[plan] tag={args.tag}  videos={len(videos)}  out={out_dir}")

    for v in videos:
        process_video(args.tag, v, out_dir)


if __name__ == "__main__":
    main()
