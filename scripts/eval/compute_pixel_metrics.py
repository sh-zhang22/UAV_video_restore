"""对 eval_compress_restore/<tag>/<video>/ 下的 compressed.mp4 与 restored.mp4，
以 orig（VisDrone-VID-slices 源视频）为参考，跑 PSNR / Y-PSNR / SSIM / LPIPS。

薄壳脚本：核心逻辑在 metrics.Evaluator.evaluate_folder。
- 空间对齐：orig / compressed 走 SeedVR 输入端同款几何（面积保比 bicubic 缩放 + 16 倍数中心裁剪，
  metrics.SeedVRGeometry），与 restored 逐像素同坐标；--max_area 须与生成 restored 时一致
- 帧数不一致：min(T) 对齐（前对齐）
- LPIPS backbone 默认 vgg（官方推荐评测用 alex，--lpips_net alex 可切换；两者数值不可混比）
- 结果：metrics_pixel.csv / metrics_pixel.json 写到 <out>/<tag>/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from metrics import Evaluator, DEFAULT_PIXEL_METRICS, DEFAULT_MAX_AREA                              # noqa: E402
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
    ap.add_argument("--max_area", type=int, default=DEFAULT_MAX_AREA,
                    help="须与生成 restored 时 eval_compress_restore.py 的 --max_area 一致")
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
          f"max_area={args.max_area}  which={args.which}")

    ev = Evaluator(device=args.device, lpips_net=args.lpips_net)
    ev.evaluate_folder(
        tag_dir=tag_dir,
        ref_lookup=find_orig_mp4,
        cand_names={"cmp": "compressed.mp4", "rst": "restored.mp4"},
        which=tuple(args.which),
        max_area=args.max_area,
        video_stems=args.videos,
        out_csv=args.out_csv,
        out_json=args.out_json,
        verbose=True,
    )


if __name__ == "__main__":
    main()
