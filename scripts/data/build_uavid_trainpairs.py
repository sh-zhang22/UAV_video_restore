#!/usr/bin/env python3
"""UAVid 训练三元组生成器：clean / compressed(双流合成) / roi_mask。

Pipeline（单 clip）：
    1) ffmpeg 缩 4K → 720×1280 中心 crop, 抽出 [start, start+n) 帧到 raw YUV
    2) YOLOv8s 检测每帧 ROI 框
    3) 逐帧: box → 硬 mask → gaussian σ=3 羽化 → soft mask ∈ [0,1]
    4) VVenC 编两次：QP=q_roi (高质量), QP=q_bg (低质量)，两次都产 recon YUV
    5) 合成: compressed = mask * rec_hi + (1-mask) * rec_lo
    6) 落盘: clean.mp4 / compressed.mp4 / mask.mp4 / meta.json

用法（冒烟，seq1 第 0 个 clip）：
    python build_uavid_trainpairs.py --seqs seq1 --clip_indices 0

全量：
    python build_uavid_trainpairs.py     # 默认 27 train+val seq × 5 clips
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# 复用 task3 代码（task3 目录在项目根下）
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "task3_video_codec_baselines_20260907"))
from task3_vtm.task3_vtm import raster_mask, run  # noqa: E402

UAVID_ROOT = Path("/nas/datasets/zsh/UAV_datasets/UAVid/uavid_v1.5_official_release")
OUT_ROOT = Path("/nas/datasets/zsh/UAV_datasets/UAVid_trainpairs")
WORK_ROOT = Path("/tmp/uavid_pairs_work")
VVENC = _ROOT / "task3_video_codec_baselines_20260907/third_party/vvenc-1.14.0/bin/release-static/vvencFFapp"

TARGET_W, TARGET_H = 1280, 720
FPS = 20
CLIP_FRAMES = 121
CLIPS_PER_SEQ = 5
Q_ROI = 22
Q_BG = 46
MASK_SIGMA = 3.0
YOLO_MODEL = "yolov8s.pt"
YOLO_CONF = 0.10
YOLO_IMGSZ = 1280  # 航拍视角小物体多，用大 imgsz 提高召回（默认 640 会漏很多）
YOLO_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck


def find_seq_video(split: str, seq_name: str) -> Path:
    p = UAVID_ROOT / split / seq_name / "images.mp4"
    return p if p.is_file() else None


def resolve_seq(seq_name: str) -> tuple[str, Path]:
    for split in ("uavid_train", "uavid_val", "uavid_test"):
        v = find_seq_video(split, seq_name)
        if v:
            return split, v
    raise FileNotFoundError(f"seq {seq_name} not found under {UAVID_ROOT}")


def clip_starts_for_seq(total_frames: int, n_clips: int, clip_len: int) -> list[int]:
    """在 seq 中均匀切 n_clips 个不重叠的 clip 起点。"""
    if total_frames < clip_len:
        raise ValueError(f"seq too short: {total_frames} < {clip_len}")
    if n_clips == 1:
        return [(total_frames - clip_len) // 2]
    # 均匀分布：起点 0, dt, 2*dt, ..., 保证末尾不越界
    span = total_frames - clip_len
    dt = span // (n_clips - 1) if n_clips > 1 else 0
    return [i * dt for i in range(n_clips)]


def extract_clip_yuv(src_mp4: Path, start_frame: int, n_frames: int, out_yuv: Path, work: Path):
    """ffmpeg 抽 clip → 缩到 1280×720 中心 crop → YUV420 8bit raw。"""
    # scale 保持 aspect ratio increase 再 crop 到目标尺寸 → 4K 16:9 直接 fit；4096×2160 会稍 crop 两侧
    vf = f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,crop={TARGET_W}:{TARGET_H}"
    # -ss 用 select filter 更准（-ss 前置会 keyframe 对齐失准，但 UAVid 是 all-I mpeg4，OK）
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_mp4),
        "-vf", f"select='between(n\\,{start_frame}\\,{start_frame + n_frames - 1})',{vf}",
        "-vsync", "vfr",
        "-frames:v", str(n_frames),
        "-pix_fmt", "yuv420p",
        "-f", "rawvideo",
        str(out_yuv),
    ]
    run(cmd, work / f"ffmpeg_extract_{start_frame}.log")


def decode_yuv_to_mp4(yuv_path: Path, out_mp4: Path, lossless: bool, work: Path):
    """YUV420 raw → mp4；lossless=True 用 H.264 CRF=0；灰度用 -pix_fmt gray。"""
    codec_args = ["-c:v", "libx264", "-preset", "veryfast"]
    if lossless:
        codec_args += ["-crf", "0"]
    else:
        codec_args += ["-crf", "18"]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", f"{TARGET_W}x{TARGET_H}", "-r", str(FPS),
        "-i", str(yuv_path),
    ] + codec_args + [
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    run(cmd, work / f"ffmpeg_yuv2mp4_{out_mp4.stem}.log")


def gray_frames_to_mp4(frames_u8: np.ndarray, out_mp4: Path, work: Path):
    """(T,H,W) uint8 → 灰度 mp4（H.264 CRF=0）。"""
    T, H, W = frames_u8.shape
    tmp_yuv = out_mp4.with_suffix(".gray_raw")
    # 灰度直接写 raw gray8，再让 ffmpeg 读
    frames_u8.tofile(tmp_yuv)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", str(tmp_yuv),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "0",
        "-pix_fmt", "yuv420p",   # 存成 yuv420p，Y=灰度值；U/V 常量 128
        str(out_mp4),
    ]
    run(cmd, work / f"ffmpeg_mask2mp4_{out_mp4.stem}.log")
    tmp_yuv.unlink(missing_ok=True)


def yolo_detect_yuv(yuv_path: Path, n_frames: int, device: str, work: Path) -> list[list[list[float]]]:
    """把 YUV 转成临时 mp4 后跑 YOLOv8 检测（Ultralytics 只吃 mp4）。"""
    # 转个临时 mp4（无损）供 YOLO 读
    tmp_mp4 = work / "for_yolo.mp4"
    decode_yuv_to_mp4(yuv_path, tmp_mp4, lossless=True, work=work)

    from ultralytics import YOLO
    kwargs = dict(stream=True, verbose=False, classes=YOLO_CLASSES,
                  conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=device)
    frames = []
    model = YOLO(YOLO_MODEL)
    for i, result in enumerate(model.predict(str(tmp_mp4), **kwargs)):
        frames.append(result.boxes.xyxy.cpu().tolist())
        if i + 1 >= n_frames:
            break
    while len(frames) < n_frames:
        frames.append([])
    tmp_mp4.unlink(missing_ok=True)
    return frames


def boxes_to_softmask(boxes_per_frame: list[list[list[float]]], n_frames: int,
                      sigma: float) -> np.ndarray:
    """(T, H, W) uint8 in [0,255]，soft mask。"""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        gaussian_filter = None
    masks = np.zeros((n_frames, TARGET_H, TARGET_W), dtype=np.float32)
    for i, boxes in enumerate(boxes_per_frame):
        hard = raster_mask(boxes, TARGET_W, TARGET_H).astype(np.float32)
        if sigma > 0 and gaussian_filter is not None:
            hard = gaussian_filter(hard, sigma=sigma)
        masks[i] = hard
    # 归一化到 [0,1]，再缩到 [0,255]
    m_max = masks.max()
    if m_max > 0:
        masks = masks / m_max
    return (masks * 255.0).clip(0, 255).astype(np.uint8)


def vvenc_encode(yuv_in: Path, qp: int, rec_out: Path, bitstream_out: Path,
                 n_frames: int, work: Path) -> float:
    """跑 VVenC faster，返回耗时秒。"""
    cmd = [
        str(VVENC),
        "-i", str(yuv_in),
        "-s", f"{TARGET_W}x{TARGET_H}",
        "--fps", f"{FPS}/1",
        "-f", str(n_frames),
        "-b", str(bitstream_out),
        "-o", str(rec_out),
        "--OutputBitDepth=8",
        "--preset", "faster",
        "--QP", str(qp),
        "--PerceptQPA=0",
        "--GOPSize=32",
        "--PicReordering=1",
        "--DecodingRefreshType=cra",
        "--Threads=-1",
        "--MTProfile=-1",
    ]
    t0 = time.monotonic()
    run(cmd, work / f"vvenc_qp{qp}.log")
    return time.monotonic() - t0


def compose_yuv420_direct(rec_hi_path: Path, rec_lo_path: Path,
                           mask_u8: np.ndarray, out_yuv: Path,
                           n_frames: int, device: str):
    """YUV 域直接融合，跳过 RGB↔YUV 往返。全部在 GPU。

    Layout: 每帧 YUV420 = Y (H*W) + U (H/2*W/2) + V (H/2*W/2)
    """
    import torch
    W, H = TARGET_W, TARGET_H
    y_size = H * W
    uv_size = (H // 2) * (W // 2)
    frame_bytes = y_size + 2 * uv_size

    # 一次性读两路 YUV 到 CPU
    hi_bytes = np.fromfile(rec_hi_path, dtype=np.uint8, count=frame_bytes * n_frames)
    lo_bytes = np.fromfile(rec_lo_path, dtype=np.uint8, count=frame_bytes * n_frames)
    if hi_bytes.size != frame_bytes * n_frames or lo_bytes.size != frame_bytes * n_frames:
        raise RuntimeError("YUV size mismatch")

    dev = torch.device(device)
    hi = torch.from_numpy(hi_bytes.reshape(n_frames, frame_bytes)).to(dev, non_blocking=True)
    lo = torch.from_numpy(lo_bytes.reshape(n_frames, frame_bytes)).to(dev, non_blocking=True)
    mask = torch.from_numpy(mask_u8).to(dev, non_blocking=True)  # (T, H, W) uint8

    # Y 分量融合（全分辨率）
    m_full = mask.float() / 255.0                                # (T, H, W)
    y_hi = hi[:, :y_size].view(n_frames, H, W).float()
    y_lo = lo[:, :y_size].view(n_frames, H, W).float()
    y_out = (m_full * y_hi + (1 - m_full) * y_lo).round().clamp(0, 255).to(torch.uint8)

    # UV 分量融合（下采到 H/2 × W/2）
    # 用 avg_pool2d 下采 mask
    m_ds = torch.nn.functional.avg_pool2d(
        m_full.unsqueeze(1), kernel_size=2, stride=2
    ).squeeze(1)                                                 # (T, H/2, W/2)
    u_hi = hi[:, y_size:y_size + uv_size].view(n_frames, H // 2, W // 2).float()
    u_lo = lo[:, y_size:y_size + uv_size].view(n_frames, H // 2, W // 2).float()
    v_hi = hi[:, y_size + uv_size:].view(n_frames, H // 2, W // 2).float()
    v_lo = lo[:, y_size + uv_size:].view(n_frames, H // 2, W // 2).float()
    u_out = (m_ds * u_hi + (1 - m_ds) * u_lo).round().clamp(0, 255).to(torch.uint8)
    v_out = (m_ds * v_hi + (1 - m_ds) * v_lo).round().clamp(0, 255).to(torch.uint8)

    # 拼回每帧 layout: [Y, U, V]
    y_flat = y_out.view(n_frames, y_size)
    u_flat = u_out.view(n_frames, uv_size)
    v_flat = v_out.view(n_frames, uv_size)
    out = torch.cat([y_flat, u_flat, v_flat], dim=1)             # (T, frame_bytes)
    out.cpu().numpy().tofile(out_yuv)


def process_clip(seq_name: str, src_mp4: Path, clip_idx: int, start_frame: int,
                 out_dir: Path, work: Path, device: str) -> dict:
    """跑一个 clip 的完整流水线，返回 meta dict。"""
    print(f"  [{seq_name}/clip_{clip_idx:03d}] start_frame={start_frame}")
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    # 1) 抽 clip 到 YUV420
    clean_yuv = work / "clean.yuv"
    t0 = time.monotonic()
    extract_clip_yuv(src_mp4, start_frame, CLIP_FRAMES, clean_yuv, work)
    t_extract = time.monotonic() - t0

    # 2) YOLO 检测
    t0 = time.monotonic()
    boxes = yolo_detect_yuv(clean_yuv, CLIP_FRAMES, device, work)
    t_yolo = time.monotonic() - t0
    n_with_roi = sum(bool(x) for x in boxes)
    n_total_boxes = sum(len(x) for x in boxes)

    # 3) mask 生成
    mask_u8 = boxes_to_softmask(boxes, CLIP_FRAMES, MASK_SIGMA)
    mask_coverage = float(mask_u8.mean() / 255.0)

    # 4) VVenC 两次
    rec_hi = work / f"rec_qp{Q_ROI}.yuv"
    bs_hi = work / f"stream_qp{Q_ROI}.vvc"
    t_hi = vvenc_encode(clean_yuv, Q_ROI, rec_hi, bs_hi, CLIP_FRAMES, work)

    rec_lo = work / f"rec_qp{Q_BG}.yuv"
    bs_lo = work / f"stream_qp{Q_BG}.vvc"
    t_lo = vvenc_encode(clean_yuv, Q_BG, rec_lo, bs_lo, CLIP_FRAMES, work)

    # 5) 合成：YUV 域直接融合（GPU）
    t0 = time.monotonic()
    compressed_yuv = work / "compressed.yuv"
    compose_yuv420_direct(rec_hi, rec_lo, mask_u8, compressed_yuv, CLIP_FRAMES, device)
    t_compose = time.monotonic() - t0

    # 6) 落盘三元组：都封成 H.264 CRF=0 mp4
    t0 = time.monotonic()
    decode_yuv_to_mp4(clean_yuv, out_dir / "clean.mp4", lossless=True, work=work)
    decode_yuv_to_mp4(compressed_yuv, out_dir / "compressed.mp4", lossless=True, work=work)
    gray_frames_to_mp4(mask_u8, out_dir / "mask.mp4", work=work)
    t_encode_out = time.monotonic() - t0

    meta = {
        "seq": seq_name,
        "clip_idx": clip_idx,
        "start_frame": start_frame,
        "n_frames": CLIP_FRAMES,
        "resolution": [TARGET_W, TARGET_H],
        "fps": FPS,
        "q_roi": Q_ROI,
        "q_bg": Q_BG,
        "mask_sigma": MASK_SIGMA,
        "yolo": {"model": YOLO_MODEL, "conf": YOLO_CONF, "classes": YOLO_CLASSES},
        "roi_stats": {
            "frames_with_roi": n_with_roi,
            "total_boxes": n_total_boxes,
            "mean_coverage": mask_coverage,
        },
        "bitstream_bytes": {"q_roi": bs_hi.stat().st_size, "q_bg": bs_lo.stat().st_size},
        "timing_seconds": {
            "extract": round(t_extract, 2),
            "yolo": round(t_yolo, 2),
            "vvenc_hi": round(t_hi, 2),
            "vvenc_lo": round(t_lo, 2),
            "compose": round(t_compose, 2),
            "encode_out": round(t_encode_out, 2),
        },
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 清临时
    for f in [clean_yuv, compressed_yuv, rec_hi, rec_lo, bs_hi, bs_lo]:
        f.unlink(missing_ok=True)

    print(f"    ✓ done in {sum(meta['timing_seconds'].values()):.1f}s, "
          f"cov={mask_coverage:.3f}, boxes={n_total_boxes}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="要处理的 seq 名列表，不给则默认 27 train+val seq")
    ap.add_argument("--clip_indices", nargs="+", type=int, default=None,
                    help="要处理的 clip 索引列表，不给则默认 0..CLIPS_PER_SEQ-1")
    ap.add_argument("--device", default="cuda:2", help="YOLO 设备")
    ap.add_argument("--out_root", default=str(OUT_ROOT), type=Path)
    ap.add_argument("--work_root", default=str(WORK_ROOT), type=Path)
    args = ap.parse_args()

    if not VVENC.is_file():
        raise SystemExit(f"VVenC 二进制未找到: {VVENC}")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"缺少 {tool}")

    # 默认 seq 列表：train + val
    if args.seqs is None:
        seqs = []
        for split in ("uavid_train", "uavid_val"):
            seqs.extend(sorted(p.name for p in (UAVID_ROOT / split).iterdir() if p.is_dir()))
        # dedup 保持顺序
        seen = set()
        seqs = [s for s in seqs if not (s in seen or seen.add(s))]
    else:
        seqs = args.seqs

    clip_indices = args.clip_indices if args.clip_indices is not None else list(range(CLIPS_PER_SEQ))

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for seq_name in seqs:
        split, src_mp4 = resolve_seq(seq_name)
        # probe frame count
        info = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "json", str(src_mp4)],
            capture_output=True, text=True, check=True).stdout)
        total_frames = int(info["streams"][0]["nb_frames"])
        starts = clip_starts_for_seq(total_frames, CLIPS_PER_SEQ, CLIP_FRAMES)

        for clip_idx in clip_indices:
            if clip_idx >= CLIPS_PER_SEQ:
                continue
            start_frame = starts[clip_idx]
            out_dir = args.out_root / seq_name / f"clip_{clip_idx:03d}"
            work_dir = args.work_root / seq_name / f"clip_{clip_idx:03d}"
            if (out_dir / "meta.json").is_file():
                print(f"[skip] {seq_name}/clip_{clip_idx:03d} already exists")
                continue
            try:
                meta = process_clip(seq_name, src_mp4, clip_idx, start_frame,
                                    out_dir, work_dir, args.device)
                all_meta.append(meta)
            except Exception as e:
                print(f"[error] {seq_name}/clip_{clip_idx:03d}: {e}")
                raise

    # 写全局 manifest
    if all_meta:
        manifest_path = args.out_root / "manifest.json"
        existing = []
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
        existing.extend(all_meta)
        manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"\nDone. manifest at {manifest_path} ({len(existing)} clips total)")


if __name__ == "__main__":
    main()
