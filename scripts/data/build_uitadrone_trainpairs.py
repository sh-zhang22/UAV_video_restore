#!/usr/bin/env python3
"""UIT-Adrone 训练三元组生成器（复用 build_uavid_trainpairs 主逻辑）。

与 UAVid pipeline 差异：
- 视频平铺在 videos/*.MP4（无 split 层级）
- 帧率 30fps（不重采样，保持原始）
- 分辨率 1920×1080 → 1280×720 完全等比缩放
- 每个视频按 step 帧步长切 clip（默认 121，无重叠）

用法（冒烟，只跑 DJI_0066）：
    python build_uitadrone_trainpairs.py --seqs DJI_0066

全量：
    python build_uitadrone_trainpairs.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import build_uavid_trainpairs as bu

VIDEO_ROOT = Path("/nas/datasets/zsh/UAV_datasets/UIT-Adrone/videos")
OUT_ROOT = Path("/nas/datasets/zsh/UAV_datasets/UIT-Adrone_trainpairs")
WORK_ROOT = Path("/tmp/uitadrone_pairs_work")

# 覆盖 bu 模块常量：UIT-Adrone 是 30fps
bu.FPS = 30

DEFAULT_STEP = 121  # 无重叠


def discover_seqs() -> list[Path]:
    return sorted(VIDEO_ROOT.glob("*.MP4"))


def probe_total_frames(mp4: Path) -> int:
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "json", str(mp4)],
        capture_output=True, text=True, check=True).stdout)
    return int(info["streams"][0]["nb_frames"])


def clip_starts_by_step(total_frames: int, clip_len: int, step: int) -> list[int]:
    starts = []
    s = 0
    while s + clip_len <= total_frames:
        starts.append(s)
        s += step
    return starts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="视频文件名（不带扩展名），如 DJI_0066；不给则全量")
    ap.add_argument("--clip_indices", nargs="+", type=int, default=None,
                    help="要处理的 clip 索引；不给则全跑")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP)
    ap.add_argument("--out_root", default=str(OUT_ROOT), type=Path)
    ap.add_argument("--work_root", default=str(WORK_ROOT), type=Path)
    args = ap.parse_args()

    if not bu.VVENC.is_file():
        raise SystemExit(f"VVenC 二进制未找到: {bu.VVENC}")
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"缺少 {tool}")

    seqs = discover_seqs()
    if args.seqs:
        wanted = set(args.seqs)
        seqs = [p for p in seqs if p.stem in wanted]
    if not seqs:
        raise SystemExit("没找到匹配的视频")

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    all_meta = []
    total_clips = 0
    for src_mp4 in seqs:
        seq_name = src_mp4.stem
        total_frames = probe_total_frames(src_mp4)
        starts = clip_starts_by_step(total_frames, bu.CLIP_FRAMES, args.step)
        total_clips += len(starts)
        print(f"[seq {seq_name}] total_frames={total_frames}, clips={len(starts)}")

        indices = args.clip_indices if args.clip_indices is not None else list(range(len(starts)))
        for clip_idx in indices:
            if clip_idx >= len(starts):
                continue
            start_frame = starts[clip_idx]
            out_dir = args.out_root / seq_name / f"clip_{clip_idx:04d}"
            work_dir = args.work_root / seq_name / f"clip_{clip_idx:04d}"
            if (out_dir / "meta.json").is_file():
                print(f"[skip] {seq_name}/clip_{clip_idx:04d}")
                continue
            try:
                meta = bu.process_clip(seq_name, src_mp4, clip_idx, start_frame,
                                       out_dir, work_dir, args.device)
                all_meta.append(meta)
            except Exception as e:
                print(f"[error] {seq_name}/clip_{clip_idx:04d}: {e}")
                raise

    print(f"\n[plan] total_seqs={len(seqs)}, total_clips_available={total_clips}")

    if all_meta:
        manifest_path = args.out_root / "manifest.json"
        existing = []
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
        existing.extend(all_meta)
        manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        print(f"Done. manifest at {manifest_path} ({len(existing)} clips total)")


if __name__ == "__main__":
    main()
