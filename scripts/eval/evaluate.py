"""统一评估入口。一次读视频、算全部指标、返回 dict。

用法：
    from evaluate import evaluate
    scores = evaluate(
        pred_video="test_recovered_seedvr2_3b.mp4",
        gt_video="test.mp4",
        device="cuda:3",                    # LPIPS / YOLO 用
        yolo_ckpt="yolov8n.pt",
        yolo_classes=["person", "car"],
    )
    # scores = {'psnr': ..., 'ssim': ..., 'lpips': ..., 'miou': ...}

- pred/gt 视频必须等 shape（帧数 + 分辨率）；不等则 raise，不做自动 resize
- mIoU 优先用外部传入的 pred_mask / gt_mask（路径）；两者都缺时现场 YOLO 生成
- metrics 可通过 which 参数选子集：which=("psnr", "ssim") 只算这俩
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from metrics import (                                    # noqa: E402
    psnr, ssim, lpips_metric, miou, miou_from_videos,
    read_video_tchw_float, read_mask_thw_bool, assert_same_shape,
)

DEFAULT_METRICS = ("psnr", "ssim", "lpips", "miou")


def evaluate(
    pred_video: str,
    gt_video: str,
    *,
    which: Sequence[str] = DEFAULT_METRICS,
    device: str = "cuda:0",
    # mIoU 相关
    pred_mask: Optional[str] = None,
    gt_mask: Optional[str] = None,
    yolo_ckpt: str = "yolov8n.pt",
    yolo_classes: Optional[Sequence[str]] = None,
    yolo_conf: float = 0.25,
    yolo_dilate: int = 0,
    # LPIPS
    lpips_net: str = "alex",
    verbose: bool = True,
) -> dict:
    """跑一组指标，返回 {name: float}。"""
    which = tuple(which)
    scores: dict = {}

    # 视频侧一次性加载，PSNR/SSIM/LPIPS 共享
    need_pixel = any(m in which for m in ("psnr", "ssim", "lpips"))
    if need_pixel:
        pred_t = read_video_tchw_float(pred_video)       # (T,3,H,W) float [0,1]
        gt_t = read_video_tchw_float(gt_video)
        assert_same_shape(pred_t, gt_t, "pred_video", "gt_video")
        if verbose:
            print(f"[evaluate] video shape: {tuple(pred_t.shape)}")

    if "psnr" in which:
        scores["psnr"] = psnr(pred_t, gt_t)
    if "ssim" in which:
        # SSIM 在 GPU 上算更快；device 参数与 LPIPS 一致
        scores["ssim"] = ssim(pred_t, gt_t, device=device)
    if "lpips" in which:
        scores["lpips"] = lpips_metric(pred_t, gt_t, net=lpips_net, device=device)

    if "miou" in which:
        if pred_mask is not None and gt_mask is not None:
            # 外部传 mask 分支
            pm = read_mask_thw_bool(pred_mask)
            gm = read_mask_thw_bool(gt_mask)
            assert_same_shape(pm, gm, "pred_mask", "gt_mask")
            scores["miou"] = miou(pm, gm)
        elif pred_mask is None and gt_mask is None:
            # 现场 YOLO 生成
            scores["miou"] = miou_from_videos(
                pred_video=pred_video, gt_video=gt_video,
                yolo_ckpt=yolo_ckpt, classes=yolo_classes,
                conf=yolo_conf, dilate=yolo_dilate, device=device,
            )
        else:
            raise ValueError(
                "miou: pred_mask 与 gt_mask 必须同时给或同时不给；只给一个无意义"
            )

    if verbose:
        for k, v in scores.items():
            print(f"[evaluate] {k}: {v:.4f}")
    return scores


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pred_mask", default=None)
    p.add_argument("--gt_mask", default=None)
    p.add_argument("--yolo_ckpt", default="yolov8n.pt")
    p.add_argument("--yolo_classes", nargs="+", default=None)
    p.add_argument("--which", nargs="+", default=list(DEFAULT_METRICS))
    args = p.parse_args()

    scores = evaluate(
        pred_video=args.pred, gt_video=args.gt,
        which=args.which, device=args.device,
        pred_mask=args.pred_mask, gt_mask=args.gt_mask,
        yolo_ckpt=args.yolo_ckpt, yolo_classes=args.yolo_classes,
    )
    print("scores:", scores)
