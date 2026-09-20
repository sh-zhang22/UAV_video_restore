#!/usr/bin/env python3
"""VisDrone-VID-slices test-dev：压缩 + SeedVR2 修复 + YOLO 复测。

Per video 流程：
  1. 抽 clean YUV（保原分辨率、原 fps）
  2. YOLO v9e 检测 → soft ROI mask（σ=3, YOLO 参数同评测）
  3. VVenC 编两次：Q_ROI=22 / Q_BG=46
  4. YUV 域直接融合（GPU）→ compressed.mp4
  5. Recover(method='seedvr2_3b') → restored.mp4
  6. YOLO 分别检测 orig / compressed / restored
  7. mIoU 三组：
        - restored vs GT           （最终检测精度）
        - restored vs orig 预测    （修复保 YOLO 一致性）
        - compressed vs GT         （只压不修的对照）

产物：eval_compress_restore/<video_stem>/{compressed,mask,restored}.mp4
                          + yolo_{orig,compressed,restored}.json
                          + meta.json
      eval_compress_restore/{summary.json,summary.csv}
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# 挂根：让 `from recover import Recover` 和 `from scripts.eval.xxx import ...` 都能工作
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from recover import Recover
from scripts.eval.eval_yolo_visdrone import (
    ROOT, VD_MODEL_CLASSES, VD_KEEP_NATIVE,
    load_gt, boxes_to_mask, frame_iou,
)

_PROJ_ROOT = Path(__file__).resolve().parents[2]
VVENC = _PROJ_ROOT / "task3_video_codec_baselines_20260907/third_party/vvenc-1.14.0/bin/release-static/vvencFFapp"
CKPT_SEEDVR = _PROJ_ROOT / "third_party/SeedVR/ckpts/seedvr2_ema_3b.pth"

# 默认压缩参数（与 build_uavid_trainpairs.py 完全一致；可被 CLI 覆盖）
Q_ROI = 22
Q_BG = 46
MASK_SIGMA = 3.0

# YOLO 参数（与 eval_yolo_visdrone baseline 保持一致）
YOLO_MODEL = "ckpts_yolo/v9e/best.pt"
YOLO_CONF = 0.15   # 2026-09-15 v6 sweep 最优 conf
YOLO_IMGSZ = 960
# 时序管线（与 eval_yolo_visdrone_track_interp 默认一致，2026-09-15 v6 sweep）
TRACKER = "trackers/bytetrack_loose.yaml"
MAX_GAP = 4
MIN_TRACK_LEN = 5


def run(cmd, log_path=None):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if log_path:
        log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode:
        tail = proc.stdout[-3000:] if proc.stdout else "(no output)"
        raise RuntimeError(f"cmd failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return proc.stdout


def probe_video(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)["streams"][0]
    n, den = map(int, d["avg_frame_rate"].split("/"))
    fps = n / den
    W, H = int(d["width"]), int(d["height"])
    nb = int(d.get("nb_frames") or 0)
    if not nb:
        r2 = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True)
        nb = int(r2.stdout.strip())
    return W, H, fps, nb


def restored_to_orig_boxes(pred, W_o, H_o, W_r, H_r, max_area):
    """restored 空间下的 xyxy 框列表 → orig 空间。

    SeedVR pipeline: NaResize(area, downsample_only=False) → DivisibleCrop((16,16), center)。
    逆变换：先反向 center-crop（+offset），再反向 area resize（/scale）。
    max_area = res_h * res_w（SeedVR 目标像素面积）。

    当算出的 (H_r_expected, W_r_expected) 与 probe 得到的 (H_r, W_r) 不一致时，
    退化到按比例缩放（旧行为，不准但不崩），并打印警告。
    """
    import math
    scale = math.sqrt(max_area / (H_o * W_o))
    H1, W1 = round(H_o * scale), round(W_o * scale)
    H_r_expected = H1 - (H1 % 16)
    W_r_expected = W1 - (W1 % 16)
    if (H_r_expected, W_r_expected) != (H_r, W_r):
        print(f"    [warn] restored 空间与 SeedVR pipeline 推算不匹配："
              f"expected {W_r_expected}x{H_r_expected}, got {W_r}x{H_r}；退化到 scale-based 反变换")
        sx, sy = W_o / W_r, H_o / H_r
        return [[[x1*sx, y1*sy, x2*sx, y2*sy] for (x1, y1, x2, y2) in fr] for fr in pred]
    crop_left = (W1 - W_r) // 2
    crop_top = (H1 - H_r) // 2
    def _m(x1, y1, x2, y2):
        return [(x1 + crop_left) / scale, (y1 + crop_top) / scale,
                (x2 + crop_left) / scale, (y2 + crop_top) / scale]
    return [[_m(*b) for b in fr] for fr in pred]


def extract_clean_yuv(mp4, W, H, n_frames, out_yuv, work):
    """mp4 → yuv420p 8bit raw；保持原分辨率、原 fps。"""
    cmd = ["ffmpeg", "-y",
           "-i", str(mp4),
           "-frames:v", str(n_frames),
           "-pix_fmt", "yuv420p",
           "-f", "rawvideo",
           str(out_yuv)]
    run(cmd, work / "ffmpeg_extract.log")


def yolo_detect_mp4(mp4, model, device, n_frames_expected=None):
    """跑 YOLO v9e，返回 per-frame [[x1,y1,x2,y2], ...]。"""
    kwargs = dict(stream=True, verbose=False, classes=VD_MODEL_CLASSES,
                  conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device)
    frames = []
    for result in model.predict(str(mp4), **kwargs):
        cur = []
        if result.boxes is not None and len(result.boxes) > 0:
            for xy in result.boxes.xyxy.cpu().tolist():
                cur.append([float(v) for v in xy])
        frames.append(cur)
    if n_frames_expected is not None:
        # 有时 YOLO stream 会漏尾帧
        while len(frames) < n_frames_expected:
            frames.append([])
    return frames


def yolo_track_interp_mp4(mp4, model, device, n_frames_expected=None):
    """新推荐管线：YOLO v9e + ByteTrack + gap 插值，返回 per-frame [[x1,y1,x2,y2], ...]。
    与 eval_yolo_visdrone_track_interp.py::eval_one_video_track_interp 的检测逻辑对齐。"""
    # Pass 1: track → track_history（用于 gap 插值，只需要坐标+tid）
    # ★ track 必须放在 predict 前面：ultralytics 的 predictor 内部状态在首次调用后固化，
    #   若首次是 predict，后续 predict 的 NMS/fp16 配置与 track-warmed 状态不同，
    #   会导致 mIoU 差异 ~0.045（同视频 cold=0.5758 vs warm=0.6209）。
    track_history = defaultdict(list)
    track_kwargs = dict(stream=True, verbose=False, classes=VD_MODEL_CLASSES,
                        conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device,
                        tracker=TRACKER, persist=False)
    for i, r in enumerate(model.track(str(mp4), **track_kwargs)):
        fidx = i + 1
        if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
            for xy, tid in zip(r.boxes.xyxy.cpu().tolist(),
                               r.boxes.id.cpu().tolist()):
                track_history[int(tid)].append((fidx, [float(v) for v in xy]))

    # Pass 2: predict → 每帧原始 boxes（现在是 warm state）
    predict_kwargs = dict(stream=True, verbose=False, classes=VD_MODEL_CLASSES,
                          conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device)
    frame_boxes = []
    for r in model.predict(str(mp4), **predict_kwargs):
        cur = []
        if r.boxes is not None and len(r.boxes) > 0:
            for xy in r.boxes.xyxy.cpu().tolist():
                cur.append([float(v) for v in xy])
        frame_boxes.append(cur)

    # 线性插值补 gap（含 min_track_len 过滤）
    filled = [list(b) for b in frame_boxes]
    n_filled = 0
    for tid, seq in track_history.items():
        if len(seq) < MIN_TRACK_LEN:
            continue
        for k in range(len(seq) - 1):
            f_a, b_a = seq[k]
            f_b, b_b = seq[k + 1]
            gap = f_b - f_a
            if gap <= 1 or gap > MAX_GAP:
                continue
            for step in range(1, gap):
                alpha = step / gap
                interp = [b_a[c] + alpha * (b_b[c] - b_a[c]) for c in range(4)]
                filled[f_a - 1 + step].append(interp)
                n_filled += 1

    if n_frames_expected is not None:
        while len(filled) < n_frames_expected:
            filled.append([])
    print(f"    [track+interp] {len(track_history)} tracks, {n_filled} interp boxes filled")
    return filled


def boxes_to_softmask_u8(boxes_per_frame, W, H, sigma):
    """(T, H, W) uint8 in [0,255]，每帧 Gaussian σ 羽化。"""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        gaussian_filter = None
    T = len(boxes_per_frame)
    masks = np.zeros((T, H, W), dtype=np.float32)
    for i, boxes in enumerate(boxes_per_frame):
        hard = np.zeros((H, W), dtype=np.float32)
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            xi1 = max(0, int(np.floor(x1)))
            yi1 = max(0, int(np.floor(y1)))
            xi2 = min(W, int(np.ceil(x2)))
            yi2 = min(H, int(np.ceil(y2)))
            if xi2 > xi1 and yi2 > yi1:
                hard[yi1:yi2, xi1:xi2] = 1.0
        if sigma > 0 and gaussian_filter is not None:
            hard = gaussian_filter(hard, sigma=sigma)
        masks[i] = hard
    m_max = masks.max()
    if m_max > 0:
        masks = masks / m_max
    return (masks * 255.0).clip(0, 255).astype(np.uint8)


def vvenc_encode(yuv_in, W, H, fps, qp, rec_out, bs_out, n_frames, work):
    cmd = [str(VVENC),
           "-i", str(yuv_in),
           "-s", f"{W}x{H}",
           "--fps", f"{fps}/1",
           "-f", str(n_frames),
           "-b", str(bs_out),
           "-o", str(rec_out),
           "--OutputBitDepth=8",
           "--preset", "faster",
           "--QP", str(qp),
           "--PerceptQPA=0",
           "--GOPSize=32",
           "--PicReordering=1",
           "--DecodingRefreshType=cra",
           "--Threads=-1",
           "--MTProfile=-1"]
    t0 = time.monotonic()
    run(cmd, work / f"vvenc_qp{qp}.log")
    return time.monotonic() - t0


def compose_yuv420_direct(rec_hi_path, rec_lo_path, mask_u8, out_yuv,
                          W, H, n_frames, device):
    """GPU 在 YUV 域直接融合 = mask * rec_hi + (1-mask) * rec_lo。"""
    import torch
    y_size = H * W
    uv_size = (H // 2) * (W // 2)
    frame_bytes = y_size + 2 * uv_size

    hi = np.fromfile(rec_hi_path, dtype=np.uint8, count=frame_bytes * n_frames)
    lo = np.fromfile(rec_lo_path, dtype=np.uint8, count=frame_bytes * n_frames)
    if hi.size != frame_bytes * n_frames or lo.size != frame_bytes * n_frames:
        raise RuntimeError(f"YUV size mismatch: hi={hi.size} lo={lo.size} "
                           f"expected={frame_bytes * n_frames}")

    dev = torch.device(device)
    hi_t = torch.from_numpy(hi.reshape(n_frames, frame_bytes)).to(dev)
    lo_t = torch.from_numpy(lo.reshape(n_frames, frame_bytes)).to(dev)
    m = torch.from_numpy(mask_u8).to(dev).float() / 255.0  # (T, H, W)

    # Y
    y_hi = hi_t[:, :y_size].view(n_frames, H, W).float()
    y_lo = lo_t[:, :y_size].view(n_frames, H, W).float()
    y_out = (m * y_hi + (1 - m) * y_lo).round().clamp(0, 255).to(torch.uint8)
    # UV (下采 mask)
    m_ds = torch.nn.functional.avg_pool2d(m.unsqueeze(1), 2, 2).squeeze(1)
    u_hi = hi_t[:, y_size:y_size + uv_size].view(n_frames, H // 2, W // 2).float()
    u_lo = lo_t[:, y_size:y_size + uv_size].view(n_frames, H // 2, W // 2).float()
    v_hi = hi_t[:, y_size + uv_size:].view(n_frames, H // 2, W // 2).float()
    v_lo = lo_t[:, y_size + uv_size:].view(n_frames, H // 2, W // 2).float()
    u_out = (m_ds * u_hi + (1 - m_ds) * u_lo).round().clamp(0, 255).to(torch.uint8)
    v_out = (m_ds * v_hi + (1 - m_ds) * v_lo).round().clamp(0, 255).to(torch.uint8)

    out = torch.cat([y_out.view(n_frames, y_size),
                     u_out.view(n_frames, uv_size),
                     v_out.view(n_frames, uv_size)], dim=1)
    out.cpu().numpy().tofile(out_yuv)


def yuv_to_mp4(yuv, W, H, fps, out_mp4, work, crf=0):
    """YUV420p raw → mp4（默认 CRF=0 无损，保住压缩过后的视觉信息）。"""
    cmd = ["ffmpeg", "-y",
           "-f", "rawvideo", "-pix_fmt", "yuv420p",
           "-s", f"{W}x{H}", "-r", f"{fps:.5f}",
           "-i", str(yuv),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p",
           str(out_mp4)]
    run(cmd, work / f"yuv2mp4_{out_mp4.stem}.log")


def gray_to_mp4(mask_u8, fps, out_mp4, work):
    """(T,H,W) uint8 → 灰度 mp4（用于查看 mask 生成正误）。"""
    T, H, W = mask_u8.shape
    tmp = out_mp4.with_suffix(".gray_raw")
    mask_u8.tofile(tmp)
    cmd = ["ffmpeg", "-y",
           "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{W}x{H}", "-r", f"{fps:.5f}",
           "-i", str(tmp),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
           "-pix_fmt", "yuv420p",
           str(out_mp4)]
    run(cmd, work / f"mask2mp4_{out_mp4.stem}.log")
    tmp.unlink(missing_ok=True)


def compute_miou_vs_gt(pred_frames, gt_frames, gt_keep_set, W, H):
    """预测（xyxy list per frame） vs GT（1-idx local_frame → boxes）。"""
    ious = []
    n_both_empty = n_pred_only = n_gt_only = n_both = 0
    T = len(pred_frames)
    for i in range(T):
        fidx = i + 1
        pred_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_frames[i]]
        pred_mask = boxes_to_mask(pred_ltwh, W, H)
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
    return {"miou": float(np.mean(ious)) if ious else None,
            "n_scored": len(ious),
            "n_both_empty": n_both_empty, "n_both_nonempty": n_both,
            "n_pred_only": n_pred_only, "n_gt_only": n_gt_only}


def compute_miou_pred_pred(pred_a, pred_b, W, H):
    """两组预测的 mask mIoU（a 对 b，对称）。"""
    ious = []
    T = min(len(pred_a), len(pred_b))
    n_both_empty = n_a_only = n_b_only = n_both = 0
    for i in range(T):
        a_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_a[i]]
        b_ltwh = [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in pred_b[i]]
        ma = boxes_to_mask(a_ltwh, W, H)
        mb = boxes_to_mask(b_ltwh, W, H)
        iou = frame_iou(ma, mb)
        if iou is None:
            n_both_empty += 1
        else:
            ious.append(iou)
            if ma.any() and not mb.any():
                n_a_only += 1
            elif mb.any() and not ma.any():
                n_b_only += 1
            else:
                n_both += 1
    return {"miou": float(np.mean(ious)) if ious else None,
            "n_scored": len(ious),
            "n_both_empty": n_both_empty, "n_both_nonempty": n_both,
            "n_a_only": n_a_only, "n_b_only": n_b_only}


def process_video(mp4_orig, gt_txt, out_dir, work_dir, device,
                  yolo_model, sp_size=1, seedvr_kwargs_extra=None,
                  q_roi=Q_ROI, q_bg=Q_BG, skip_seedvr=False,
                  max_area=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    W, H, fps, nb = probe_video(mp4_orig)
    gt_frames = load_gt(gt_txt)
    gt_keep_set = set(VD_KEEP_NATIVE)
    max_gt = max(gt_frames.keys()) if gt_frames else 0
    print(f"  probe: {W}x{H} @ {fps:.2f}fps  nb={nb}  max_gt={max_gt}")

    meta_path = out_dir / "meta.json"
    compressed_mp4 = out_dir / "compressed.mp4"
    restored_mp4 = out_dir / "restored.mp4"
    mask_mp4 = out_dir / "mask.mp4"
    yolo_orig_json = out_dir / "yolo_orig_ti.json"
    yolo_compressed_json = out_dir / "yolo_compressed_ti.json"
    yolo_restored_json = out_dir / "yolo_restored_ti.json"

    timings = {}

    # Step A: YOLO on orig（顺便供 mask 生成，避免跑两遍）
    if yolo_orig_json.exists():
        pred_orig = json.loads(yolo_orig_json.read_text())
        print(f"  [A] yolo_orig cached: {sum(len(f) for f in pred_orig)} boxes")
    else:
        t0 = time.monotonic()
        pred_orig = yolo_track_interp_mp4(mp4_orig, yolo_model, device, n_frames_expected=nb)
        timings["yolo_orig"] = round(time.monotonic() - t0, 2)
        yolo_orig_json.write_text(json.dumps(pred_orig))
        print(f"  [A] yolo_orig: {sum(len(f) for f in pred_orig)} boxes "
              f"in {timings['yolo_orig']}s")

    # Step B/C/D: 压缩流程（若 compressed.mp4 不存在则跑完整个链路）
    if not compressed_mp4.exists():
        # B.1 抽 clean YUV
        clean_yuv = work_dir / "clean.yuv"
        t0 = time.monotonic()
        extract_clean_yuv(mp4_orig, W, H, nb, clean_yuv, work_dir)
        timings["extract_yuv"] = round(time.monotonic() - t0, 2)
        expected_bytes = (H * W + 2 * (H // 2) * (W // 2)) * nb
        actual_bytes = clean_yuv.stat().st_size
        if actual_bytes != expected_bytes:
            n_actual = actual_bytes // (H * W + 2 * (H // 2) * (W // 2))
            print(f"  [B.1] yuv frames mismatch: got {n_actual}, expected {nb} "
                  f"(clip to {n_actual})")
            nb = n_actual
            pred_orig = pred_orig[:nb]

        # B.2 mask 生成
        t0 = time.monotonic()
        mask_u8 = boxes_to_softmask_u8(pred_orig, W, H, MASK_SIGMA)
        timings["mask_gen"] = round(time.monotonic() - t0, 2)
        coverage = float(mask_u8.mean() / 255.0)
        print(f"  [B.2] mask: coverage={coverage:.3f}, in {timings['mask_gen']}s")
        gray_to_mp4(mask_u8, fps, mask_mp4, work_dir)

        # B.3 VVenC 双流
        rec_hi = work_dir / f"rec_qp{q_roi}.yuv"
        bs_hi = work_dir / f"stream_qp{q_roi}.vvc"
        t_hi = vvenc_encode(clean_yuv, W, H, fps, q_roi, rec_hi, bs_hi, nb, work_dir)
        timings["vvenc_hi"] = round(t_hi, 2)
        print(f"  [B.3a] vvenc Q_ROI={q_roi}: {t_hi:.1f}s, "
              f"bs={bs_hi.stat().st_size/1024:.1f}KB")

        rec_lo = work_dir / f"rec_qp{q_bg}.yuv"
        bs_lo = work_dir / f"stream_qp{q_bg}.vvc"
        t_lo = vvenc_encode(clean_yuv, W, H, fps, q_bg, rec_lo, bs_lo, nb, work_dir)
        timings["vvenc_lo"] = round(t_lo, 2)
        print(f"  [B.3b] vvenc Q_BG={q_bg}: {t_lo:.1f}s, "
              f"bs={bs_lo.stat().st_size/1024:.1f}KB")

        # B.4 YUV 域融合
        t0 = time.monotonic()
        compressed_yuv = work_dir / "compressed.yuv"
        compose_yuv420_direct(rec_hi, rec_lo, mask_u8, compressed_yuv,
                              W, H, nb, device)
        yuv_to_mp4(compressed_yuv, W, H, fps, compressed_mp4, work_dir, crf=0)
        timings["compose"] = round(time.monotonic() - t0, 2)
        print(f"  [B.4] compose+encode: {timings['compose']}s → "
              f"{compressed_mp4.stat().st_size/1024/1024:.1f}MB")

        # 清中间产物
        for f in [clean_yuv, compressed_yuv, rec_hi, rec_lo]:
            f.unlink(missing_ok=True)
    else:
        print(f"  [B/C/D] compressed.mp4 cached")

    # Step E: SeedVR2-3B 修复
    # 自适应 res_h/res_w：小视频用原分辨率；大视频等比缩到 max_area 面积上限（保持长宽比）
    import math as _m
    if max_area is not None and H * W > max_area:
        _s = _m.sqrt(max_area / (H * W))
        seedvr_res_h = round(H * _s)
        seedvr_res_w = round(W * _s)
    else:
        seedvr_res_h, seedvr_res_w = H, W
    effective_max_area = seedvr_res_h * seedvr_res_w

    if skip_seedvr:
        print(f"  [E] skip_seedvr=True，跳过修复；restored 相关字段留空")
        pred_restored_scaled = None
        W_r, H_r, nb_r = W, H, nb
    elif not restored_mp4.exists():
        t0 = time.monotonic()
        seedvr_kwargs = {"res_h": seedvr_res_h, "res_w": seedvr_res_w,
                         "sp_size": sp_size, "seed": 666}
        # 长视频分块推理：目标每段 T×A ≤ 1.2e8（约 79 GB 显存的 65% 余量）
        # 129 帧 × 921600 = 1.19e8，与已验证成功点 (T=308, A=518400, T×A=1.6e8) 同量级但更保守
        _seg_ta_budget = 1.2e8
        _seg_frames = max(9, int(_seg_ta_budget / effective_max_area))
        _seg_frames = ((_seg_frames - 1) // 4) * 4 + 1  # 对齐到 4k+1
        if nb > _seg_frames:
            seedvr_kwargs["chunk_frames"] = _seg_frames
            print(f"  [E] SeedVR res_h={seedvr_res_h} res_w={seedvr_res_w} "
                  f"(orig {W}x{H}, max_area={max_area}) chunk={_seg_frames}f (T={nb})")
        else:
            print(f"  [E] SeedVR res_h={seedvr_res_h} res_w={seedvr_res_w} "
                  f"(orig {W}x{H}, max_area={max_area}) 单段 (T={nb} ≤ {_seg_frames})")
        if seedvr_kwargs_extra:
            seedvr_kwargs.update(seedvr_kwargs_extra)
        Recover(
            video_path=str(compressed_mp4),
            recovered_path=str(restored_mp4),
            ckpt_path=str(CKPT_SEEDVR),
            method="seedvr2_3b",
            device=device,
            method_kwargs=seedvr_kwargs,
        )
        timings["seedvr"] = round(time.monotonic() - t0, 2)
        print(f"  [E] seedvr2_3b: {timings['seedvr']}s")
    else:
        print(f"  [E] restored.mp4 cached")

    # Step F: YOLO on compressed / restored
    if yolo_compressed_json.exists():
        pred_compressed = json.loads(yolo_compressed_json.read_text())
        print(f"  [F1] yolo_compressed cached")
    else:
        t0 = time.monotonic()
        pred_compressed = yolo_track_interp_mp4(compressed_mp4, yolo_model, device,
                                                n_frames_expected=nb)
        timings["yolo_compressed"] = round(time.monotonic() - t0, 2)
        yolo_compressed_json.write_text(json.dumps(pred_compressed))
        print(f"  [F1] yolo_compressed: {sum(len(f) for f in pred_compressed)} boxes "
              f"in {timings['yolo_compressed']}s")

    if yolo_restored_json.exists():
        pred_restored = json.loads(yolo_restored_json.read_text())
        print(f"  [F2] yolo_restored cached")
    elif skip_seedvr:
        pred_restored = None
        print(f"  [F2] skip_seedvr=True，跳过 restored YOLO")
    else:
        t0 = time.monotonic()
        # SeedVR 修复输出可能因 16 对齐 crop 尺寸略变，YOLO 用实际尺寸即可
        pred_restored = yolo_track_interp_mp4(restored_mp4, yolo_model, device,
                                              n_frames_expected=nb)
        timings["yolo_restored"] = round(time.monotonic() - t0, 2)
        yolo_restored_json.write_text(json.dumps(pred_restored))
        print(f"  [F2] yolo_restored: {sum(len(f) for f in pred_restored)} boxes "
              f"in {timings['yolo_restored']}s")

    # Step G: mIoU
    # 检查 restored 视频尺寸（SeedVR 内部 NaResize+DivisibleCrop(16, center) 可能改尺寸）
    if pred_restored is not None:
        W_r, H_r, _, nb_r = probe_video(restored_mp4)
        if (W_r, H_r) != (W, H):
            # SeedVR max_area = res_h * res_w，取自 process_video 里实际生效的 effective_max_area
            print(f"  [G] restored 尺寸变化 {W}x{H} → {W_r}x{H_r}，"
                  f"逆向 NaResize+DivisibleCrop(center) 映回 GT 空间 (max_area={effective_max_area})")
            pred_restored_scaled = restored_to_orig_boxes(
                pred_restored, W, H, W_r, H_r, effective_max_area)
        else:
            pred_restored_scaled = pred_restored
    else:
        pred_restored_scaled = None
        W_r, H_r, nb_r = W, H, nb

    # 帧数对齐（SeedVR 修复偶尔会 pad 1-2 帧，比 orig 多）
    if pred_restored_scaled is not None:
        T_min = min(len(pred_orig), len(pred_compressed), len(pred_restored_scaled),
                    nb, nb_r or nb)
        pred_restored_c = pred_restored_scaled[:T_min]
    else:
        T_min = min(len(pred_orig), len(pred_compressed), nb)
        pred_restored_c = None
    pred_orig_c = pred_orig[:T_min]
    pred_compressed_c = pred_compressed[:T_min]

    miou_orig_gt = compute_miou_vs_gt(pred_orig_c, gt_frames, gt_keep_set, W, H)
    miou_compressed_gt = compute_miou_vs_gt(pred_compressed_c, gt_frames, gt_keep_set, W, H)
    miou_compressed_vs_orig = compute_miou_pred_pred(pred_compressed_c, pred_orig_c, W, H)
    if pred_restored_c is not None:
        miou_restored_gt = compute_miou_vs_gt(pred_restored_c, gt_frames, gt_keep_set, W, H)
        miou_restored_vs_orig = compute_miou_pred_pred(pred_restored_c, pred_orig_c, W, H)
    else:
        miou_restored_gt = {"miou": None}
        miou_restored_vs_orig = {"miou": None}

    meta = {
        "video": mp4_orig.name,
        "resolution": [W, H],
        "restored_resolution": [W_r, H_r],
        "fps": fps,
        "n_frames": nb,
        "n_frames_scored": T_min,
        "compress": {"Q_ROI": q_roi, "Q_BG": q_bg, "mask_sigma": MASK_SIGMA},
        "yolo": {"model": YOLO_MODEL, "imgsz": YOLO_IMGSZ, "conf": YOLO_CONF,
                 "classes": VD_MODEL_CLASSES,
                 "pipeline": "track_interp",
                 "tracker": TRACKER, "max_gap": MAX_GAP,
                 "min_track_len": MIN_TRACK_LEN},
        "miou": {
            "orig_vs_gt": miou_orig_gt,
            "compressed_vs_gt": miou_compressed_gt,
            "restored_vs_gt": miou_restored_gt,
            "restored_vs_orig": miou_restored_vs_orig,
            "compressed_vs_orig": miou_compressed_vs_orig,
        },
        "timings_seconds": timings,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    restored_str = f"{miou_restored_gt['miou']:.4f}" if miou_restored_gt['miou'] is not None else "n/a"
    restored_vs_orig_str = f"{miou_restored_vs_orig['miou']:.4f}" if miou_restored_vs_orig['miou'] is not None else "n/a"
    print(f"  ✓ mIoU: orig/GT={miou_orig_gt['miou']:.4f}  "
          f"compressed/GT={miou_compressed_gt['miou']:.4f}  "
          f"restored/GT={restored_str}  "
          f"restored/orig={restored_vs_orig_str}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--sp_size", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("eval_compress_restore"))
    ap.add_argument("--work_root", type=Path, default=Path("/tmp/eval_compress_restore_work"))
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频 stem 列表；不给则跑 test-dev 全部 10 个")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--q_roi", type=int, default=Q_ROI)
    ap.add_argument("--q_bg", type=int, default=Q_BG)
    ap.add_argument("--tag", default=None,
                    help="子目录 tag；默认 q{q_roi}_{q_bg}。out/tag/<video>/ ...")
    ap.add_argument("--skip_seedvr", action="store_true",
                    help="只跑压缩，不跑 SeedVR 修复")
    ap.add_argument("--max_area", type=int, default=720 * 1280,
                    help="SeedVR 目标像素面积上限；H*W ≤ max_area 保留原分辨率，"
                         "H*W > max_area 时等比缩到 max_area（保持长宽比）")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = f"q{args.q_roi}_{args.q_bg}"

    # 收集视频（视频用干净版，GT txt 从 boxed/ 读）
    videos = []
    for d in ("M01", "M02", "M07", "M08"):
        for mp4 in sorted((ROOT / d).glob("test-dev_*.mp4")):
            if args.videos and mp4.stem not in args.videos:
                continue
            txt = ROOT / d / "boxed" / f"{mp4.stem}.txt"
            if txt.is_file():
                videos.append((d, mp4, txt))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit("no videos matched")
    print(f"[plan] {len(videos)} videos, device={args.device}, "
          f"Q_ROI={args.q_roi}, Q_BG={args.q_bg}, tag={args.tag}")

    out_base = args.out / args.tag
    work_base = args.work_root / args.tag
    out_base.mkdir(parents=True, exist_ok=True)
    work_base.mkdir(parents=True, exist_ok=True)

    # 一次性加载 YOLO
    from ultralytics import YOLO
    print(f"[init] loading YOLO {YOLO_MODEL}...")
    yolo_model = YOLO(YOLO_MODEL)

    all_meta = []
    t_all = time.monotonic()
    for i, (subset, mp4, txt) in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {subset}/{mp4.stem}")
        out_dir = out_base / mp4.stem
        work_dir = work_base / mp4.stem
        try:
            meta = process_video(mp4, txt, out_dir, work_dir, args.device,
                                 yolo_model, sp_size=args.sp_size,
                                 q_roi=args.q_roi, q_bg=args.q_bg,
                                 skip_seedvr=args.skip_seedvr,
                                 max_area=args.max_area)
            meta["subset"] = subset
            all_meta.append(meta)
        except Exception as e:
            print(f"  !! failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

    dt_all = time.monotonic() - t_all

    # 汇总
    def agg(key):
        vals = [m["miou"][key]["miou"] for m in all_meta
                if m["miou"][key]["miou"] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_videos": len(all_meta),
        "wall_seconds": round(dt_all, 1),
        "avg_miou_video": {
            "orig_vs_gt": agg("orig_vs_gt"),
            "compressed_vs_gt": agg("compressed_vs_gt"),
            "restored_vs_gt": agg("restored_vs_gt"),
            "restored_vs_orig": agg("restored_vs_orig"),
            "compressed_vs_orig": agg("compressed_vs_orig"),
        },
        "per_video": {m["video"]: {k: m["miou"][k]["miou"] for k in m["miou"]}
                      for m in all_meta},
        "config": {
            "Q_ROI": args.q_roi, "Q_BG": args.q_bg, "mask_sigma": MASK_SIGMA,
            "yolo_model": YOLO_MODEL, "yolo_imgsz": YOLO_IMGSZ, "yolo_conf": YOLO_CONF,
            "yolo_pipeline": "track_interp",
            "tracker": TRACKER, "max_gap": MAX_GAP, "min_track_len": MIN_TRACK_LEN,
            "skip_seedvr": args.skip_seedvr,
        },
    }
    (out_base / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_base / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "orig/GT", "compressed/GT", "restored/GT",
                    "restored/orig", "compressed/orig"])
        for m in all_meta:
            w.writerow([m["video"]] + [f"{m['miou'][k]['miou']:.4f}"
                                        if m['miou'][k]['miou'] is not None else "n/a"
                                        for k in ["orig_vs_gt", "compressed_vs_gt",
                                                  "restored_vs_gt", "restored_vs_orig",
                                                  "compressed_vs_orig"]])

    print(f"\n=== overall ===")
    print(json.dumps(summary["avg_miou_video"], indent=2))
    print(f"\nsaved to {out_base}/  ({dt_all:.1f}s total)")
    print(f"\nsaved to {args.out}/  ({dt_all:.1f}s total)")


if __name__ == "__main__":
    main()
