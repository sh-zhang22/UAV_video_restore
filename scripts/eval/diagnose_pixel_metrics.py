"""诊断 SeedVR2-3B 修复后 PSNR/SSIM/LPIPS 反而比 compressed 差的原因。

三个假设：
(A) resize_mode="ref" 的 bicubic upscale 破坏对齐 —— 主怀疑
(B) 色彩空间 / pix_fmt 导致的整体色偏
(C) 时序对齐错位（restored[0] 对不上 orig[0]）

对每个视频跑三种对比：
1. 默认 resize_mode="ref"（cand upscale 到 orig）
2. resize_mode="min"（都 downscale 到 min）
3. 手动全部 resize 到 restored 空间（把 orig/cmp 下采到 720p）

以及：
- RGB 三通道整片均值（诊断色偏）
- orig[i] 与 restored[0] 逐帧 PSNR（诊断帧错位）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metrics._io import read_video_tchw_float   # noqa: E402
from metrics.psnr import psnr as psnr_fn        # noqa: E402
from metrics.lpips import lpips_metric          # noqa: E402
from metrics.evaluator import _ssim_batched   # noqa: E402


def _bicubic_resize(vid, H, W):
    """仅供本诊断脚本对比旧 resize 方式；正式评估请用 metrics.SeedVRGeometry。"""
    if vid.shape[-2:] == (H, W):
        return vid
    return F.interpolate(vid, size=(H, W), mode="bicubic", align_corners=False,
                         antialias=True).clamp_(0.0, 1.0)


VD_ROOT = Path("/nas/datasets/yixin/UAV_Dataset/VisDrone2019-VID-slices")


def find_orig(stem: str) -> Path:
    for sub in ("M01", "M02", "M07", "M08"):
        p = VD_ROOT / sub / f"{stem}.mp4"
        if p.is_file():
            return p
    raise FileNotFoundError(stem)


def three_metrics(cand: torch.Tensor, ref: torch.Tensor, device: str) -> dict:
    """cand/ref: (T,3,H,W) [0,1] CPU. shape 必须一致。"""
    return {
        "psnr":  psnr_fn(cand, ref),
        "ssim":  _ssim_batched(cand, ref, device=device),
        "lpips": lpips_metric(cand, ref, device=device),
    }


def diagnose_one(stem: str, tag_dir: Path, device: str):
    print(f"\n{'='*70}\n视频：{stem}\n{'='*70}")
    orig_p = find_orig(stem)
    cmp_p  = tag_dir / stem / "compressed.mp4"
    rst_p  = tag_dir / stem / "restored.mp4"
    assert cmp_p.is_file() and rst_p.is_file(), f"{cmp_p} / {rst_p}"

    orig = read_video_tchw_float(str(orig_p))
    cmp  = read_video_tchw_float(str(cmp_p))
    rst  = read_video_tchw_float(str(rst_p))
    T = min(orig.shape[0], cmp.shape[0], rst.shape[0])
    orig, cmp, rst = orig[:T], cmp[:T], rst[:T]
    Ho, Wo = orig.shape[-2:]
    Hr, Wr = rst.shape[-2:]
    print(f"shape: orig={tuple(orig.shape)}  cmp={tuple(cmp.shape)}  rst={tuple(rst.shape)}")
    print(f"        orig空间 {Ho}×{Wo}    restored空间 {Hr}×{Wr}")
    print(f"        面积比 restored/orig = {(Hr*Wr) / (Ho*Wo):.3f}")

    # ---------- 模式 1: 默认 ref = orig 空间（cand upscale 到 1080p） ----------
    print("\n[模式 1] resize=ref（cand → orig 空间，当前默认，怀疑 upscale 破坏对齐）")
    cmp_ref = _bicubic_resize(cmp, Ho, Wo)
    rst_ref = _bicubic_resize(rst, Ho, Wo)
    m_cmp = three_metrics(cmp_ref, orig, device)
    m_rst = three_metrics(rst_ref, orig, device)
    print(f"  cmp: psnr={m_cmp['psnr']:.4f}  ssim={m_cmp['ssim']:.4f}  lpips={m_cmp['lpips']:.4f}")
    print(f"  rst: psnr={m_rst['psnr']:.4f}  ssim={m_rst['ssim']:.4f}  lpips={m_rst['lpips']:.4f}")

    # ---------- 模式 2: resize=min（都下采到 restored 空间） ----------
    print("\n[模式 2] resize=min（都 → restored 空间 = downscale orig, native rst）")
    orig_min = _bicubic_resize(orig, Hr, Wr)
    cmp_min  = _bicubic_resize(cmp, Hr, Wr)
    m_cmp2 = three_metrics(cmp_min, orig_min, device)
    m_rst2 = three_metrics(rst, orig_min, device)
    print(f"  cmp: psnr={m_cmp2['psnr']:.4f}  ssim={m_cmp2['ssim']:.4f}  lpips={m_cmp2['lpips']:.4f}")
    print(f"  rst: psnr={m_rst2['psnr']:.4f}  ssim={m_rst2['ssim']:.4f}  lpips={m_rst2['lpips']:.4f}")

    # ---------- 色偏诊断 ----------
    print("\n[色偏诊断] RGB 三通道整片均值")
    for name, v in [("orig", orig), ("cmp", cmp), ("rst", rst)]:
        m = v.mean(dim=(0, 2, 3)).tolist()
        print(f"  {name}: R={m[0]:.4f} G={m[1]:.4f} B={m[2]:.4f}")
    # DC 差（把 rst 均值 shift 到 orig，重新算 rst 在 ref 空间的 PSNR，看 shift 能"救回"多少）
    dc_shift = orig.mean(dim=(0, 2, 3)) - rst.mean(dim=(0, 2, 3))
    print(f"  DC shift (orig - rst): R={dc_shift[0]:+.4f} G={dc_shift[1]:+.4f} B={dc_shift[2]:+.4f}")
    rst_dc = (rst + dc_shift.view(1, 3, 1, 1)).clamp_(0, 1)
    rst_dc_ref = _bicubic_resize(rst_dc, Ho, Wo)
    p_dc = psnr_fn(rst_dc_ref, orig)
    print(f"  rst PSNR after DC shift (模式1 空间): {p_dc:.4f}  (原 {m_rst['psnr']:.4f})")

    # ---------- 时序错位诊断 ----------
    print("\n[时序错位] orig[i] vs restored[0]（先把 restored[0] upscale 到 orig 空间比）")
    rst0_up = _bicubic_resize(rst[:1], Ho, Wo)[0]  # (3, Ho, Wo)
    for i in range(min(6, T)):
        mse = ((orig[i] - rst0_up) ** 2).mean()
        psnr_i = 10.0 * torch.log10(1.0 / mse)
        marker = "  ← min"  # 后面标注
        print(f"  orig[{i}] vs rst[0]: psnr={psnr_i.item():.4f}")

    # ---------- 亚像素漂移诊断（rst 与 orig 都在 restored 空间下） ----------
    print("\n[亚像素漂移] restored 空间下 rst vs orig_min 的位移敏感度")
    # 假设：如果对 orig_min 施加 ±1 像素平移后 PSNR 明显变化，说明当前对齐已经在敏感区
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0)]:
        shifted = torch.roll(orig_min, shifts=(dy, dx), dims=(-2, -1))
        mse = ((rst - shifted) ** 2).mean()
        p = 10.0 * torch.log10(1.0 / mse)
        print(f"  shift (dx={dx:+d}, dy={dy:+d}): psnr={p.item():.4f}")

    return {
        "stem": stem,
        "shape_orig": (Ho, Wo), "shape_rst": (Hr, Wr),
        "mode_ref": {"cmp": m_cmp, "rst": m_rst},
        "mode_min": {"cmp": m_cmp2, "rst": m_rst2},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q22_37")
    ap.add_argument("--out", type=Path, default=Path("eval_compress_restore"))
    ap.add_argument("--videos", nargs="+",
                    default=["test-dev_uav0000306_00230_v_full",
                             "test-dev_uav0000073_00600_v_full"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    tag_dir = args.out / args.tag
    results = []
    for stem in args.videos:
        r = diagnose_one(stem, tag_dir, args.device)
        results.append(r)

    print(f"\n\n{'#'*70}\n汇总（模式1 = 默认 ref 空间，模式2 = restored 空间下比较）\n{'#'*70}")
    for r in results:
        s = r["stem"][:40]
        print(f"\n{s}")
        for mode in ["mode_ref", "mode_min"]:
            print(f"  {mode}:")
            for name in ["cmp", "rst"]:
                m = r[mode][name]
                print(f"    {name}: psnr={m['psnr']:.3f}  ssim={m['ssim']:.4f}  lpips={m['lpips']:.4f}")


if __name__ == "__main__":
    main()
