#!/usr/bin/env python3
"""VisDrone-VID-slices test-dev：压缩 + SeedVR2 修复 + YOLO 复测。

6 阶段批处理流水线（重构自单视频串行版）：
  Phase 1: YOLO on orig（载 1 次 → detect 全部 → 释放）
  Phase 2: mask 生成（CPU）
  Phase 3: VVenC 双 QP 压缩 + YUV 域融合（CPU + GPU 少量）
  Phase 4: SeedVR 修复（一次子进程跑完所有视频，模型只加载一次）
  Phase 5: YOLO on compressed + restored（载 1 次 → detect 全部 → 释放）
  Phase 6: mIoU 计算 + summary（CPU）

设计动机：
- Phase 隔离让 YOLO 和 SeedVR 永不共占 GPU，7B 不再 OOM
- SeedVR 模型只加载一次（原来每视频重加载 30-90s × 10 = 5-15min 浪费）
- 每阶段可断点续跑：已存在的产物直接跳过（--force 覆盖）
- 单视频失败进 error log，同阶段其它视频继续跑，最后统一报告

产物：eval_compress_restore/<tag>/<video_stem>/{compressed,mask,restored}.mp4
                                  + yolo_{orig,compressed,restored}_ti.json
                                  + meta.json
      eval_compress_restore/<tag>/{summary.json,summary.csv,phase_errors.json}
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 挂根：让 `from recover import Recover` 和 `from scripts.eval.xxx import ...` 都能工作
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.eval_yolo_visdrone import (
    ROOT, VD_MODEL_CLASSES, VD_KEEP_NATIVE,
    load_gt, boxes_to_mask, frame_iou,
)

_PROJ_ROOT = Path(__file__).resolve().parents[2]
VVENC = _PROJ_ROOT / "task3_video_codec_baselines_20260907/third_party/vvenc-1.14.0/bin/release-static/vvencFFapp"
CKPT_SEEDVR = _PROJ_ROOT / "third_party/SeedVR/ckpts/seedvr2_ema_3b.pth"

# 默认压缩参数
Q_ROI = 22
Q_BG = 37
MASK_SIGMA = 3.0

# YOLO 参数
YOLO_MODEL = "ckpts_yolo/v9e/best.pt"
YOLO_CONF = 0.15
YOLO_IMGSZ = 960
TRACKER = "trackers/bytetrack_loose.yaml"
MAX_GAP = 4
MIN_TRACK_LEN = 5


# ==================== 基础工具（沿用原实现） ====================

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
    cmd = ["ffmpeg", "-y",
           "-i", str(mp4),
           "-frames:v", str(n_frames),
           "-pix_fmt", "yuv420p",
           "-f", "rawvideo",
           str(out_yuv)]
    run(cmd, work / "ffmpeg_extract.log")


def yolo_detect_mp4(mp4, model, device, n_frames_expected=None):
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
        while len(frames) < n_frames_expected:
            frames.append([])
    return frames


def yolo_track_interp_mp4(mp4, model, device, n_frames_expected=None):
    """YOLO v9e + ByteTrack + gap 插值（细节见原实现注释）。"""
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

    predict_kwargs = dict(stream=True, verbose=False, classes=VD_MODEL_CLASSES,
                          conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device)
    frame_boxes = []
    for r in model.predict(str(mp4), **predict_kwargs):
        cur = []
        if r.boxes is not None and len(r.boxes) > 0:
            for xy in r.boxes.xyxy.cpu().tolist():
                cur.append([float(v) for v in xy])
        frame_boxes.append(cur)

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
    return filled, n_filled, len(track_history)


def boxes_to_softmask_u8(boxes_per_frame, W, H, sigma):
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
    m = torch.from_numpy(mask_u8).to(dev).float() / 255.0

    y_hi = hi_t[:, :y_size].view(n_frames, H, W).float()
    y_lo = lo_t[:, :y_size].view(n_frames, H, W).float()
    y_out = (m * y_hi + (1 - m) * y_lo).round().clamp(0, 255).to(torch.uint8)
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
    cmd = ["ffmpeg", "-y",
           "-f", "rawvideo", "-pix_fmt", "yuv420p",
           "-s", f"{W}x{H}", "-r", f"{fps:.5f}",
           "-i", str(yuv),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p",
           str(out_mp4)]
    run(cmd, work / f"yuv2mp4_{out_mp4.stem}.log")


def gray_to_mp4(mask_u8, fps, out_mp4, work):
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


# ==================== Pipeline 阶段 ====================

def _compose_dev_from(device: str) -> str:
    """把 '1,3' / 'cuda:1' / '1' 规范化为 compose_yuv420_direct 用的单卡 'cuda:1'。"""
    d = device.split(",")[0] if isinstance(device, str) else str(device)
    if d.startswith("cuda:"):
        return d
    if d.isdigit():
        return f"cuda:{d}"
    return d


def _load_probe_cached(video_ctx: Dict, mp4: Path) -> Tuple[int, int, float, int]:
    """probe 并缓存进 video_ctx，避免重复。"""
    if "probe" not in video_ctx:
        video_ctx["probe"] = probe_video(mp4)
    return video_ctx["probe"]


def phase1_yolo_orig(videos: List[Dict], device: str, force: bool = False) -> Dict[str, Dict]:
    """全部 orig 视频过一遍 YOLO，写 yolo_orig_ti.json。返回 {stem → {boxes, n_boxes}}。

    YOLO 只加载一次，遍历完释放。
    """
    print(f"\n{'='*60}\n[Phase 1] YOLO on {len(videos)} orig videos\n{'='*60}")
    todo = []
    results = {}
    for v in videos:
        j = v["out_dir"] / "yolo_orig_ti.json"
        if j.exists() and not force:
            pred = json.loads(j.read_text())
            results[v["stem"]] = {"boxes": pred, "cached": True}
            print(f"  [cached] {v['stem']}: {sum(len(f) for f in pred)} boxes")
        else:
            todo.append(v)
    if not todo:
        print("  all cached, skip YOLO load")
        return results

    from ultralytics import YOLO
    print(f"  loading YOLO ({YOLO_MODEL}) → device={device}")
    t_load = time.monotonic()
    model = YOLO(YOLO_MODEL)
    _yolo_dev = _compose_dev_from(device)
    print(f"  YOLO loaded in {time.monotonic()-t_load:.1f}s; device={_yolo_dev}")

    errors = []
    t_all = time.monotonic()
    for i, v in enumerate(todo, 1):
        stem = v["stem"]
        mp4 = v["mp4"]
        out_dir = v["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            W, H, fps, nb = _load_probe_cached(v.setdefault("ctx", {}), mp4)
            print(f"  [{i}/{len(todo)}] {stem} ({W}x{H}, {nb}f)")
            t0 = time.monotonic()
            pred, n_filled, n_tracks = yolo_track_interp_mp4(mp4, model, _yolo_dev, n_frames_expected=nb)
            dt = time.monotonic() - t0
            (out_dir / "yolo_orig_ti.json").write_text(json.dumps(pred))
            n_boxes = sum(len(f) for f in pred)
            print(f"    done: {n_boxes} boxes ({n_tracks} tracks, {n_filled} interp), "
                  f"{dt:.1f}s")
            results[stem] = {"boxes": pred, "cached": False}
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"stem": stem, "phase": "yolo_orig",
                          "error": f"{type(e).__name__}: {e}"})
    print(f"  wall: {time.monotonic()-t_all:.1f}s")

    # 释放 YOLO
    del model
    import torch as _t
    gc.collect(); _t.cuda.empty_cache()
    print(f"  YOLO released")
    if errors:
        results["_errors"] = errors
    return results


def phase2_mask(videos: List[Dict], yolo_orig: Dict, force: bool = False) -> Dict[str, Dict]:
    """从 yolo_orig 生成 mask.mp4 + numpy mask 数组（放内存等 Phase 3 用）。

    产物：mask.mp4（供人眼查看）。数组不落盘（Phase 3 立刻消费）。
    """
    print(f"\n{'='*60}\n[Phase 2] mask 生成\n{'='*60}")
    results = {}
    errors = []
    for i, v in enumerate(videos, 1):
        stem = v["stem"]
        if stem not in yolo_orig or "boxes" not in yolo_orig[stem]:
            print(f"  [{i}/{len(videos)}] {stem}: SKIP (no yolo_orig)")
            errors.append({"stem": stem, "phase": "mask", "error": "no yolo_orig"})
            continue
        try:
            W, H, fps, nb = _load_probe_cached(v["ctx"], v["mp4"])
            pred_orig = yolo_orig[stem]["boxes"]
            # 帧数对齐（YOLO 可能少几帧，见后面 clean_yuv 校准；这里保守裁到 min）
            pred_orig = pred_orig[:nb]
            t0 = time.monotonic()
            mask_u8 = boxes_to_softmask_u8(pred_orig, W, H, MASK_SIGMA)
            dt = time.monotonic() - t0
            coverage = float(mask_u8.mean() / 255.0)
            mask_mp4 = v["out_dir"] / "mask.mp4"
            if not mask_mp4.exists() or force:
                v["work_dir"].mkdir(parents=True, exist_ok=True)
                gray_to_mp4(mask_u8, fps, mask_mp4, v["work_dir"])
            print(f"  [{i}/{len(videos)}] {stem}: coverage={coverage:.3f}, {dt:.1f}s")
            results[stem] = {"mask_u8": mask_u8, "coverage": coverage}
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"stem": stem, "phase": "mask", "error": f"{type(e).__name__}: {e}"})
    if errors:
        results["_errors"] = errors
    return results


def phase3_compress(videos: List[Dict], masks: Dict, device: str,
                    q_roi: int, q_bg: int, force: bool = False) -> Dict[str, Dict]:
    """VVenC 双流 + YUV 融合 → compressed.mp4。返回 {stem → {timings, ...}}。

    CPU 主：VVenC 编码；GPU 少量：compose_yuv420_direct 一次几秒。
    """
    print(f"\n{'='*60}\n[Phase 3] VVenC 压缩 (Q_ROI={q_roi}, Q_BG={q_bg})\n{'='*60}")
    results = {}
    errors = []
    _compose_dev = _compose_dev_from(device)
    for i, v in enumerate(videos, 1):
        stem = v["stem"]
        out_dir = v["out_dir"]
        work_dir = v["work_dir"]
        compressed_mp4 = out_dir / "compressed.mp4"
        if compressed_mp4.exists() and not force:
            print(f"  [{i}/{len(videos)}] {stem}: [cached] compressed.mp4")
            results[stem] = {"cached": True}
            continue
        if stem not in masks or "mask_u8" not in masks[stem]:
            print(f"  [{i}/{len(videos)}] {stem}: SKIP (no mask)")
            errors.append({"stem": stem, "phase": "compress", "error": "no mask"})
            continue
        try:
            W, H, fps, nb = _load_probe_cached(v["ctx"], v["mp4"])
            mask_u8 = masks[stem]["mask_u8"]
            work_dir.mkdir(parents=True, exist_ok=True)
            timings = {}

            # B.1 抽 clean YUV
            clean_yuv = work_dir / "clean.yuv"
            t0 = time.monotonic()
            extract_clean_yuv(v["mp4"], W, H, nb, clean_yuv, work_dir)
            timings["extract_yuv"] = round(time.monotonic() - t0, 2)
            expected_bytes = (H * W + 2 * (H // 2) * (W // 2)) * nb
            actual_bytes = clean_yuv.stat().st_size
            if actual_bytes != expected_bytes:
                n_actual = actual_bytes // (H * W + 2 * (H // 2) * (W // 2))
                print(f"    [B.1] yuv frames mismatch: got {n_actual}, expected {nb} (clip)")
                nb = n_actual
                mask_u8 = mask_u8[:nb]
                # 更新 ctx 里的 nb（Phase 4/5 会用到）
                W_p, H_p, fps_p, _ = v["ctx"]["probe"]
                v["ctx"]["probe"] = (W_p, H_p, fps_p, nb)

            # B.3 VVenC 双流
            rec_hi = work_dir / f"rec_qp{q_roi}.yuv"
            bs_hi = work_dir / f"stream_qp{q_roi}.vvc"
            t_hi = vvenc_encode(clean_yuv, W, H, fps, q_roi, rec_hi, bs_hi, nb, work_dir)
            timings["vvenc_hi"] = round(t_hi, 2)

            rec_lo = work_dir / f"rec_qp{q_bg}.yuv"
            bs_lo = work_dir / f"stream_qp{q_bg}.vvc"
            t_lo = vvenc_encode(clean_yuv, W, H, fps, q_bg, rec_lo, bs_lo, nb, work_dir)
            timings["vvenc_lo"] = round(t_lo, 2)

            # B.4 融合
            t0 = time.monotonic()
            compressed_yuv = work_dir / "compressed.yuv"
            compose_yuv420_direct(rec_hi, rec_lo, mask_u8, compressed_yuv,
                                  W, H, nb, _compose_dev)
            yuv_to_mp4(compressed_yuv, W, H, fps, compressed_mp4, work_dir, crf=0)
            timings["compose"] = round(time.monotonic() - t0, 2)

            for f in [clean_yuv, compressed_yuv, rec_hi, rec_lo]:
                f.unlink(missing_ok=True)

            print(f"  [{i}/{len(videos)}] {stem}: hi={timings['vvenc_hi']:.1f}s "
                  f"lo={timings['vvenc_lo']:.1f}s compose={timings['compose']:.1f}s → "
                  f"{compressed_mp4.stat().st_size/1024/1024:.1f}MB")
            results[stem] = {"timings": timings, "coverage": masks[stem]["coverage"]}
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"stem": stem, "phase": "compress", "error": f"{type(e).__name__}: {e}"})
    if errors:
        results["_errors"] = errors
    # Phase 3 GPU 用量少，一次 empty_cache 即可
    import torch as _t
    _t.cuda.empty_cache()
    return results


def phase4_seedvr(videos: List[Dict], device: str, method: str, ckpt_seedvr: str,
                  sp_size: int, max_area: int, seg_ta_budget: float,
                  seedvr_kwargs_extra: Optional[Dict] = None,
                  skip_seedvr: bool = False, force: bool = False) -> Dict[str, Dict]:
    """一次子进程跑完所有视频的 SeedVR 修复。返回 {stem → {ok, ...}}。

    模型只加载一次，for 循环 N 个视频，中间不重启子进程。
    """
    print(f"\n{'='*60}\n[Phase 4] SeedVR {method}\n{'='*60}")
    if skip_seedvr:
        print("  skip_seedvr=True，跳过")
        return {v["stem"]: {"skipped": True} for v in videos}

    import math as _m
    manifest = []
    per_video_meta = {}
    for v in videos:
        stem = v["stem"]
        compressed_mp4 = v["out_dir"] / "compressed.mp4"
        restored_mp4 = v["out_dir"] / "restored.mp4"
        if not compressed_mp4.exists():
            print(f"  {stem}: SKIP (no compressed.mp4)")
            continue
        if restored_mp4.exists() and not force:
            print(f"  {stem}: [cached] restored.mp4")
            per_video_meta[stem] = {"cached": True}
            continue

        W, H, fps, nb = _load_probe_cached(v["ctx"], v["mp4"])
        # 自适应 res
        if max_area is not None and H * W > max_area:
            _s = _m.sqrt(max_area / (H * W))
            res_h = round(H * _s); res_w = round(W * _s)
        else:
            res_h, res_w = H, W
        effective_area = res_h * res_w

        _seg = max(9, int(seg_ta_budget / effective_area))
        _seg = ((_seg - 1) // 4) * 4 + 1
        chunk_frames = _seg if nb > _seg else 0

        item = {
            "in_video": str(compressed_mp4),
            "out_video": str(restored_mp4),
            "res_h": res_h,
            "res_w": res_w,
            "chunk_frames": chunk_frames,
        }
        manifest.append(item)
        per_video_meta[stem] = {"res_h": res_h, "res_w": res_w,
                                 "effective_area": effective_area,
                                 "chunk_frames": chunk_frames}
        print(f"  queued: {stem} res={res_h}x{res_w} chunk={chunk_frames or 'none'} T={nb}")

    if not manifest:
        print("  nothing to do")
        return per_video_meta

    # 传批处理调用
    from methods._seedvr_common import run_seedvr_batch
    common_kwargs = dict(sp_size=sp_size, seed=666)
    if seedvr_kwargs_extra:
        common_kwargs.update(seedvr_kwargs_extra)
    t0 = time.monotonic()
    print(f"  launching subprocess for {len(manifest)} videos "
          f"(sp_size={sp_size}, device={device})")
    result = run_seedvr_batch(
        variant=method,
        ckpt_path=ckpt_seedvr,
        device=device,
        manifest=manifest,
        common_kwargs=common_kwargs,
    )
    dt = time.monotonic() - t0
    print(f"  subprocess done in {dt:.1f}s; ok={result['n_ok']} err={result['n_err']}")
    per_video_meta["_batch_seconds"] = round(dt, 1)
    per_video_meta["_manifest_path"] = result["manifest_path"]
    if result["errors"]:
        per_video_meta["_errors"] = [
            {"stem": Path(e["in_video"]).parent.name, "phase": "seedvr",
             "error": e["error"]}
            for e in result["errors"]
        ]
    return per_video_meta


def phase5_yolo_cmp_rst(videos: List[Dict], device: str, skip_restored: bool = False,
                        force: bool = False) -> Dict[str, Dict]:
    """载一次 YOLO，遍历所有 compressed.mp4 + restored.mp4 检测；写入各自 json。"""
    print(f"\n{'='*60}\n[Phase 5] YOLO on compressed + restored\n{'='*60}")

    # 先检查是否需要加载 YOLO
    need_load = False
    tasks = []
    results = {}
    for v in videos:
        stem = v["stem"]
        cmp_mp4 = v["out_dir"] / "compressed.mp4"
        rst_mp4 = v["out_dir"] / "restored.mp4"
        cmp_json = v["out_dir"] / "yolo_compressed_ti.json"
        rst_json = v["out_dir"] / "yolo_restored_ti.json"
        cache = {"cmp_cached": False, "rst_cached": False}
        if cmp_json.exists() and not force:
            cache["cmp_cached"] = True
        elif cmp_mp4.exists():
            tasks.append((v, "cmp", cmp_mp4, cmp_json)); need_load = True
        if not skip_restored:
            if rst_json.exists() and not force:
                cache["rst_cached"] = True
            elif rst_mp4.exists():
                tasks.append((v, "rst", rst_mp4, rst_json)); need_load = True
        results[stem] = cache

    if not need_load:
        print("  all cached, skip YOLO load")
        return results

    from ultralytics import YOLO
    print(f"  loading YOLO ({YOLO_MODEL}) → device={device}")
    t_load = time.monotonic()
    model = YOLO(YOLO_MODEL)
    _yolo_dev = _compose_dev_from(device)
    print(f"  YOLO loaded in {time.monotonic()-t_load:.1f}s; device={_yolo_dev}")

    errors = []
    t_all = time.monotonic()
    for i, (v, kind, mp4, out_json) in enumerate(tasks, 1):
        stem = v["stem"]
        try:
            W, H, fps, nb = _load_probe_cached(v["ctx"], v["mp4"])
            t0 = time.monotonic()
            pred, n_filled, n_tracks = yolo_track_interp_mp4(mp4, model, _yolo_dev, n_frames_expected=nb)
            dt = time.monotonic() - t0
            out_json.write_text(json.dumps(pred))
            n_boxes = sum(len(f) for f in pred)
            print(f"  [{i}/{len(tasks)}] {stem}/{kind}: {n_boxes} boxes "
                  f"({n_tracks} tracks, {n_filled} interp), {dt:.1f}s")
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"stem": stem, "phase": f"yolo_{kind}",
                          "error": f"{type(e).__name__}: {e}"})
    print(f"  wall: {time.monotonic()-t_all:.1f}s")

    del model
    import torch as _t
    gc.collect(); _t.cuda.empty_cache()
    print(f"  YOLO released")
    if errors:
        results["_errors"] = errors
    return results


def phase6_eval(videos: List[Dict], yolo_orig: Dict, phase4_out: Dict,
                q_roi: int, q_bg: int, method: str, max_area: int,
                skip_seedvr: bool = False) -> List[Dict]:
    """计算每个视频的 mIoU + 写 meta.json，返回 metas list。"""
    print(f"\n{'='*60}\n[Phase 6] mIoU 评测\n{'='*60}")
    gt_keep_set = set(VD_KEEP_NATIVE)
    all_meta = []
    errors = []
    for i, v in enumerate(videos, 1):
        stem = v["stem"]
        out_dir = v["out_dir"]
        mp4 = v["mp4"]
        try:
            W, H, fps, nb = _load_probe_cached(v["ctx"], mp4)
            gt_frames = load_gt(v["txt"])

            cmp_json = out_dir / "yolo_compressed_ti.json"
            rst_json = out_dir / "yolo_restored_ti.json"
            if stem not in yolo_orig or "boxes" not in yolo_orig[stem]:
                print(f"  [{i}/{len(videos)}] {stem}: SKIP (no yolo_orig)")
                errors.append({"stem": stem, "phase": "eval", "error": "no yolo_orig"})
                continue
            pred_orig = yolo_orig[stem]["boxes"]
            if not cmp_json.exists():
                print(f"  [{i}/{len(videos)}] {stem}: SKIP (no yolo_compressed)")
                errors.append({"stem": stem, "phase": "eval", "error": "no yolo_compressed"})
                continue
            pred_compressed = json.loads(cmp_json.read_text())

            # restored
            pred_restored_scaled = None
            W_r, H_r, nb_r = W, H, nb
            if not skip_seedvr and rst_json.exists():
                pred_restored = json.loads(rst_json.read_text())
                rst_mp4 = out_dir / "restored.mp4"
                if rst_mp4.exists():
                    W_r, H_r, _, nb_r = probe_video(rst_mp4)
                    if (W_r, H_r) != (W, H):
                        eff_area = phase4_out.get(stem, {}).get("effective_area", max_area)
                        print(f"    restored {W}x{H} → {W_r}x{H_r}; 逆变换 (max_area={eff_area})")
                        pred_restored_scaled = restored_to_orig_boxes(
                            pred_restored, W, H, W_r, H_r, eff_area)
                    else:
                        pred_restored_scaled = pred_restored

            # 帧数对齐
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
                "video": mp4.name,
                "subset": v["subset"],
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
                "method": method,
                "miou": {
                    "orig_vs_gt": miou_orig_gt,
                    "compressed_vs_gt": miou_compressed_gt,
                    "restored_vs_gt": miou_restored_gt,
                    "restored_vs_orig": miou_restored_vs_orig,
                    "compressed_vs_orig": miou_compressed_vs_orig,
                },
            }
            (out_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            r_str = f"{miou_restored_gt['miou']:.4f}" if miou_restored_gt['miou'] is not None else "n/a"
            rvo_str = f"{miou_restored_vs_orig['miou']:.4f}" if miou_restored_vs_orig['miou'] is not None else "n/a"
            print(f"  [{i}/{len(videos)}] {stem}: orig/GT={miou_orig_gt['miou']:.4f} "
                  f"cmp/GT={miou_compressed_gt['miou']:.4f} rst/GT={r_str} rst/orig={rvo_str}")
            all_meta.append(meta)
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            errors.append({"stem": stem, "phase": "eval", "error": f"{type(e).__name__}: {e}"})
    if errors:
        all_meta.append({"_errors": errors})
    return all_meta


# ==================== main ====================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--sp_size", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("eval_compress_restore"))
    ap.add_argument("--work_root", type=Path, default=Path("/tmp/eval_compress_restore_work"))
    ap.add_argument("--videos", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--q_roi", type=int, default=Q_ROI)
    ap.add_argument("--q_bg", type=int, default=Q_BG)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip_seedvr", action="store_true")
    ap.add_argument("--max_area", type=int, default=720 * 1280)
    ap.add_argument("--method", default="seedvr2_3b",
                    choices=["seedvr2_3b", "seedvr2_7b"])
    ap.add_argument("--ckpt_seedvr", default=None)
    ap.add_argument("--seg_ta_budget", type=float, default=None)
    ap.add_argument("--force", action="store_true",
                    help="全部强制重跑（默认按产物断点续跑）")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["yolo_orig", "mask", "compress", "seedvr", "yolo_cmp_rst", "eval"],
                    help="只跑指定阶段（用于调试）；不给则全跑")
    args = ap.parse_args()

    if args.ckpt_seedvr is None:
        _ckpt_map = {
            "seedvr2_3b": _PROJ_ROOT / "third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
            "seedvr2_7b": _PROJ_ROOT / "third_party/SeedVR/ckpts/seedvr2_ema_7b.pth",
        }
        args.ckpt_seedvr = str(_ckpt_map[args.method])
    if args.seg_ta_budget is None:
        args.seg_ta_budget = {"seedvr2_3b": 1.2e8, "seedvr2_7b": 5e7}[args.method]
    if args.tag is None:
        if args.method == "seedvr2_3b":
            args.tag = f"q{args.q_roi}_{args.q_bg}"
        else:
            _mt = args.method.split("_")[-1]
            args.tag = f"q{args.q_roi}_{args.q_bg}_{_mt}"

    # 收集视频
    videos_raw = []
    for d in ("M01", "M02", "M07", "M08"):
        for mp4 in sorted((ROOT / d).glob("test-dev_*.mp4")):
            if args.videos and mp4.stem not in args.videos:
                continue
            txt = ROOT / d / "boxed" / f"{mp4.stem}.txt"
            if txt.is_file():
                videos_raw.append((d, mp4, txt))
    if args.limit:
        videos_raw = videos_raw[:args.limit]
    if not videos_raw:
        raise SystemExit("no videos matched")

    out_base = args.out / args.tag
    work_base = args.work_root / args.tag
    out_base.mkdir(parents=True, exist_ok=True)
    work_base.mkdir(parents=True, exist_ok=True)

    # 每个视频的完整 ctx（out_dir / work_dir / ctx 缓存）
    videos: List[Dict] = []
    for subset, mp4, txt in videos_raw:
        stem = mp4.stem
        videos.append({
            "stem": stem, "subset": subset, "mp4": mp4, "txt": txt,
            "out_dir": out_base / stem,
            "work_dir": work_base / stem,
            "ctx": {},
        })

    print(f"[plan] {len(videos)} videos, device={args.device}, sp_size={args.sp_size}, "
          f"Q_ROI={args.q_roi}, Q_BG={args.q_bg}, tag={args.tag}, "
          f"method={args.method}, seg_ta_budget={args.seg_ta_budget:.1e}, "
          f"force={args.force}, only={args.only or 'all'}")

    only = set(args.only) if args.only else None
    def _run(phase: str) -> bool:
        return only is None or phase in only

    all_errors = []
    t_all = time.monotonic()

    # Phase 1
    yolo_orig = {}
    if _run("yolo_orig"):
        yolo_orig = phase1_yolo_orig(videos, args.device, force=args.force)
        if "_errors" in yolo_orig:
            all_errors.extend(yolo_orig.pop("_errors"))
    else:
        # 读盘补齐 yolo_orig，供后续 phase 用
        for v in videos:
            j = v["out_dir"] / "yolo_orig_ti.json"
            if j.exists():
                yolo_orig[v["stem"]] = {"boxes": json.loads(j.read_text()), "cached": True}

    # Phase 2
    masks = {}
    if _run("mask") or _run("compress"):
        masks = phase2_mask(videos, yolo_orig, force=args.force)
        if "_errors" in masks:
            all_errors.extend(masks.pop("_errors"))

    # Phase 3
    if _run("compress"):
        r3 = phase3_compress(videos, masks, args.device, args.q_roi, args.q_bg,
                             force=args.force)
        if "_errors" in r3:
            all_errors.extend(r3.pop("_errors"))
    # 释放 masks（可能占几百 MB）
    del masks; gc.collect()

    # Phase 4
    phase4_out = {}
    if _run("seedvr"):
        phase4_out = phase4_seedvr(
            videos, args.device, args.method, args.ckpt_seedvr,
            args.sp_size, args.max_area, args.seg_ta_budget,
            skip_seedvr=args.skip_seedvr, force=args.force)
        if "_errors" in phase4_out:
            all_errors.extend(phase4_out.pop("_errors"))
    else:
        # 补齐 effective_area（Phase 6 需要）
        import math as _m
        for v in videos:
            W, H, _, _ = _load_probe_cached(v["ctx"], v["mp4"])
            if args.max_area is not None and H * W > args.max_area:
                _s = _m.sqrt(args.max_area / (H * W))
                phase4_out[v["stem"]] = {"effective_area": round(H*_s) * round(W*_s)}
            else:
                phase4_out[v["stem"]] = {"effective_area": H * W}

    # Phase 5
    if _run("yolo_cmp_rst"):
        r5 = phase5_yolo_cmp_rst(videos, args.device,
                                 skip_restored=args.skip_seedvr, force=args.force)
        if "_errors" in r5:
            all_errors.extend(r5.pop("_errors"))

    # Phase 6
    metas: List[Dict] = []
    if _run("eval"):
        metas = phase6_eval(videos, yolo_orig, phase4_out,
                           args.q_roi, args.q_bg, args.method, args.max_area,
                           skip_seedvr=args.skip_seedvr)
        # 拆出 errors 条目
        clean_metas = []
        for m in metas:
            if "_errors" in m:
                all_errors.extend(m["_errors"])
            else:
                clean_metas.append(m)
        metas = clean_metas

    dt_all = time.monotonic() - t_all

    # 汇总
    def agg(key):
        vals = [m["miou"][key]["miou"] for m in metas
                if m["miou"][key]["miou"] is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_videos": len(metas),
        "wall_seconds": round(dt_all, 1),
        "avg_miou_video": {
            "orig_vs_gt": agg("orig_vs_gt"),
            "compressed_vs_gt": agg("compressed_vs_gt"),
            "restored_vs_gt": agg("restored_vs_gt"),
            "restored_vs_orig": agg("restored_vs_orig"),
            "compressed_vs_orig": agg("compressed_vs_orig"),
        },
        "per_video": {m["video"]: {k: m["miou"][k]["miou"] for k in m["miou"]}
                      for m in metas},
        "config": {
            "Q_ROI": args.q_roi, "Q_BG": args.q_bg, "mask_sigma": MASK_SIGMA,
            "yolo_model": YOLO_MODEL, "yolo_imgsz": YOLO_IMGSZ, "yolo_conf": YOLO_CONF,
            "yolo_pipeline": "track_interp",
            "tracker": TRACKER, "max_gap": MAX_GAP, "min_track_len": MIN_TRACK_LEN,
            "skip_seedvr": args.skip_seedvr,
            "method": args.method,
        },
        "n_errors": len(all_errors),
    }
    (out_base / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_base / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "orig/GT", "compressed/GT", "restored/GT",
                    "restored/orig", "compressed/orig"])
        for m in metas:
            w.writerow([m["video"]] + [f"{m['miou'][k]['miou']:.4f}"
                                        if m['miou'][k]['miou'] is not None else "n/a"
                                        for k in ["orig_vs_gt", "compressed_vs_gt",
                                                  "restored_vs_gt", "restored_vs_orig",
                                                  "compressed_vs_orig"]])

    if all_errors:
        (out_base / "phase_errors.json").write_text(
            json.dumps(all_errors, indent=2, ensure_ascii=False))
        print(f"\n[!] {len(all_errors)} errors, see {out_base/'phase_errors.json'}")

    print(f"\n=== overall ({args.method}) ===")
    print(json.dumps(summary["avg_miou_video"], indent=2))
    print(f"\nsaved to {out_base}/  ({dt_all:.1f}s total)")


if __name__ == "__main__":
    main()
