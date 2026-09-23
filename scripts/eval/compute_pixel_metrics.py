"""对 eval_compress_restore/<tag>/<video>/ 下的 compressed.mp4 与 restored.mp4，
以 orig（VisDrone-VID-slices 源视频）为参考，跑 PSNR / Y-PSNR / SSIM / LPIPS。

薄壳脚本：核心逻辑在 metrics.Evaluator.evaluate_folder。
- 分辨率不一致：默认 resize_mode="min"，全部下采到最小尺寸（避免 upscale 引入插值损失，
  在我们的 3 路视频场景下等价于对齐到 SeedVR 原生输出分辨率）
- 帧数不一致：min(T) 对齐（前对齐）
- LPIPS backbone 默认 vgg（更贴近感官；--lpips_net alex 可切回 alex，快 2-3x）
- 结果：metrics_pixel.csv / metrics_pixel.json 写到 <out>/<tag>/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from metrics import Evaluator, DEFAULT_PIXEL_METRICS                                # noqa: E402
from scripts.eval.eval_yolo_visdrone import ROOT as VD_ROOT                         # noqa: E402


def find_orig_mp4(stem: str) -> Path:
    """按 stem 到 VisDrone slices 各 M0x 目录找 orig .mp4。"""
    for d in ("M01", "M02", "M07", "M08"):
        p = VD_ROOT / d / f"{stem}.mp4"
        if p.is_file():
            return p
    raise FileNotFoundError(f"orig not found: {stem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="q22_37")
    ap.add_argument("--out", type=Path, default=Path("eval_compress_restore"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频 stem 列表；不给则跑该 tag 下全部")
    ap.add_argument("--lpips_net", default="vgg", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--resize_mode", default="min", choices=["min", "ref"],
                    help="min: 下采到 min(H,W)，对生成式 SR 公平（默认）；"
                         "ref: cand 上采到 ref 尺寸，传统 VSR 做法（会 penalize 生成式方法）")
    ap.add_argument("--which", nargs="+", default=list(DEFAULT_PIXEL_METRICS),
                    choices=["psnr", "psnr_y", "ssim", "lpips"],
                    help="指标子集")
    ap.add_argument("--out_csv", default="metrics_pixel.csv")
    ap.add_argument("--out_json", default="metrics_pixel.json")
    args = ap.parse_args()

    tag_dir = args.out / args.tag
    if not tag_dir.is_dir():
        raise SystemExit(f"tag dir not found: {tag_dir}")

    print(f"[plan] tag={args.tag}  device={args.device}  lpips_net={args.lpips_net}  "
          f"resize_mode={args.resize_mode}  which={args.which}")

    ev = Evaluator(device=args.device, lpips_net=args.lpips_net)
    ev.evaluate_folder(
        tag_dir=tag_dir,
        ref_lookup=find_orig_mp4,
        cand_names={"cmp": "compressed.mp4", "rst": "restored.mp4"},
        which=tuple(args.which),
        resize_mode=args.resize_mode,
        video_stems=args.videos,
        out_csv=args.out_csv,
        out_json=args.out_json,
        verbose=True,
    )


if __name__ == "__main__":
    main()
