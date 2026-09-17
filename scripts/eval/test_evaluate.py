"""指标模块冒烟：三层测试，无需真实评测数据。

用法：
    python test_evaluate.py --device cuda:3

1. 自我一致性（pred == gt）：PSNR≈inf, SSIM≈1, LPIPS≈0, mIoU=1
2. 加小高斯噪声：所有指标应向"变差"偏移
3. 已有产物对比：test_recovered_seedvr2_3b.mp4 vs test.mp4（真实数值）
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluate import evaluate                              # noqa: E402
from metrics import psnr, ssim, lpips_metric, miou         # noqa: E402


def _save_noisy_copy(src: str, dst: str, sigma: float = 0.01, seed: int = 0):
    """把 src 视频加高斯噪声后写到 dst。用来构造"稍差"的 pred。"""
    from torchvision.io.video import read_video, write_video
    vid, _, info = read_video(src, output_format="THWC")   # (T,H,W,3) uint8
    fps = float(info.get("video_fps", 24.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma * 255.0, size=vid.shape).astype(np.float32)
    noisy = vid.float().numpy() + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    write_video(dst, torch.from_numpy(noisy), fps=fps)


def test_self_consistency(gt_video: str, device: str):
    print(f"\n=== 1) self-consistency: pred = gt = {gt_video} ===")
    scores = evaluate(
        pred_video=gt_video, gt_video=gt_video,
        which=("psnr", "ssim", "lpips"),        # miou 依赖 YOLO，第 3 步再测
        device=device, verbose=True,
    )
    # 判据
    assert scores["psnr"] > 60 or scores["psnr"] == float("inf"), scores
    assert scores["ssim"] > 0.999, scores
    assert scores["lpips"] < 1e-4, scores
    print(f"[pass] self-consistency OK")


def test_noise_direction(gt_video: str, device: str, tmpdir: str):
    print(f"\n=== 2) small noise (sigma=0.01) should worsen every metric ===")
    noisy = os.path.join(tmpdir, "noisy_pred.mp4")
    _save_noisy_copy(gt_video, noisy, sigma=0.01)
    scores = evaluate(
        pred_video=noisy, gt_video=gt_video,
        which=("psnr", "ssim", "lpips"),
        device=device, verbose=True,
    )
    # 判据：PSNR 应远低于 inf 但仍很高；SSIM 应 < 1 但 > 0.9；LPIPS > 0
    assert 20 < scores["psnr"] < 60, scores
    assert 0.8 < scores["ssim"] < 1.0, scores
    assert scores["lpips"] > 0.0, scores
    print(f"[pass] noise direction correct")


def test_miou_self(gt_video: str, device: str, yolo_ckpt: str):
    print(f"\n=== 3) miou self-consistency (pred_video = gt_video, YOLO both) ===")
    scores = evaluate(
        pred_video=gt_video, gt_video=gt_video,
        which=("miou",),
        device=device,
        yolo_ckpt=yolo_ckpt,
        verbose=True,
    )
    # 同一个视频跑两次 YOLO：mask 应该 bit-for-bit 一致 → mIoU=1.0
    assert scores["miou"] > 0.999, scores
    print(f"[pass] miou self-consistency OK")


def test_real_pair(pred: str, gt: str, device: str, yolo_ckpt: str):
    print(f"\n=== 4) real pair: {pred} vs {gt} ===")
    if not (os.path.isfile(pred) and os.path.isfile(gt)):
        print(f"[skip] missing {pred} or {gt}")
        return
    scores = evaluate(
        pred_video=pred, gt_video=gt,
        which=("psnr", "ssim", "lpips", "miou"),
        device=device,
        yolo_ckpt=yolo_ckpt,
        verbose=True,
    )
    # 只 sanity：数值合理即可（PSNR>10, SSIM in (0,1), LPIPS in (0,1), miou in [0,1]）
    assert 0 < scores["psnr"] < 100, scores
    assert 0 < scores["ssim"] <= 1, scores
    assert 0 <= scores["lpips"] < 2, scores
    assert 0 <= scores["miou"] <= 1, scores
    print(f"[pass] real pair values in expected range")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="test.mp4")
    ap.add_argument("--pred", default="test_recovered_seedvr2_3b.mp4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--yolo_ckpt", default="yolov8n.pt")
    ap.add_argument("--tmpdir", default="/tmp/metrics_smoke")
    ap.add_argument(
        "--skip", nargs="+", default=[],
        choices=["self", "noise", "miou_self", "real"],
        help="跳过某些子测试",
    )
    args = ap.parse_args()

    os.makedirs(args.tmpdir, exist_ok=True)
    print(f"device={args.device}, tmpdir={args.tmpdir}")

    if "self" not in args.skip:
        test_self_consistency(args.gt, args.device)
    if "noise" not in args.skip:
        test_noise_direction(args.gt, args.device, args.tmpdir)
    if "miou_self" not in args.skip:
        test_miou_self(args.gt, args.device, args.yolo_ckpt)
    if "real" not in args.skip:
        test_real_pair(args.pred, args.gt, args.device, args.yolo_ckpt)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
