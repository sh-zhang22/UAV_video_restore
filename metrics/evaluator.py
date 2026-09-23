"""统一评测封装：一次加载、多 candidate 复用、批量目录扫描。

设计目标：
- 减少视频重复 IO 与 GPU↔CPU 搬运
- shape 不一致时统一 resize（默认 cand → ref 尺寸，bicubic + antialias）
- PSNR / SSIM / LPIPS 走同一份加载好的 tensor
- mIoU 分三种输入模式：mask / video (YOLO) / boxes (VisDrone 场景)

用法示例：
    from metrics import Evaluator
    ev = Evaluator(device="cuda:1")
    res = ev.evaluate(
        ref="orig.mp4",
        cands={"cmp": "compressed.mp4", "rst": "restored.mp4"},
        which=("psnr", "ssim", "lpips"),
    )
    # {"cmp": {"psnr": 30.9, ...}, "rst": {...}}
"""
from __future__ import annotations

import csv
import gc
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ._io import read_video_tchw_float
from .psnr import psnr as psnr_fn
from .lpips import lpips_metric


DEFAULT_PIXEL_METRICS: Tuple[str, ...] = ("psnr", "ssim", "lpips")


def _ssim_batched(pred: torch.Tensor, gt: torch.Tensor,
                  device: str = "cpu", batch: int = 32) -> float:
    """torchmetrics SSIM 分块调，大分辨率 * 多帧一次算会 int32 溢出。"""
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    if pred.shape != gt.shape:
        raise ValueError(f"ssim shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    T = pred.shape[0]
    vals, counts = [], []
    with torch.no_grad():
        for i in range(0, T, batch):
            p = pred[i:i + batch].float().to(device)
            g = gt[i:i + batch].float().to(device)
            v = tm_ssim(p, g, data_range=1.0, reduction="elementwise_mean")
            vals.append(float(v.item()))
            counts.append(p.shape[0])
    total = sum(counts)
    return sum(v * c for v, c in zip(vals, counts)) / total


def _bicubic_resize(vid: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(T, C, H0, W0) → (T, C, H, W) bicubic + antialias, clamp [0,1]."""
    if vid.shape[-2] == H and vid.shape[-1] == W:
        return vid
    return F.interpolate(vid, size=(H, W), mode="bicubic",
                         align_corners=False, antialias=True).clamp_(0.0, 1.0)


class Evaluator:
    """一份 ref、多个 cand、可选 mIoU 的统一评测器。

    Params
    ------
    device : str
        SSIM / LPIPS 的 GPU 设备。PSNR 全 CPU。
    lpips_net : str
        LPIPS backbone，官方默认 "alex"；也支持 "vgg" / "squeeze"。
    lpips_batch : int
        LPIPS 单次前向的帧 batch。默认 8，1080p 单卡够用。
    ssim_batch : int
        SSIM 单次前向的帧 batch。1080p 32 帧稳定不溢出。
    """

    def __init__(
        self,
        device: str = "cuda:0",
        *,
        lpips_net: str = "alex",
        lpips_batch: int = 8,
        ssim_batch: int = 32,
    ):
        self.device = device
        self.lpips_net = lpips_net
        self.lpips_batch = lpips_batch
        self.ssim_batch = ssim_batch

    # -------- 公开 API --------

    def evaluate(
        self,
        ref: Union[str, Path, torch.Tensor],
        cands: Mapping[str, Union[str, Path, torch.Tensor]],
        *,
        which: Sequence[str] = DEFAULT_PIXEL_METRICS,
        resize_mode: str = "ref",
        max_frames: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """算 ref 与每个 cand 的三项像素指标。

        Params
        ------
        ref, cands
            可以是 mp4 路径（会用 `read_video_tchw_float` 加载）或 (T,3,H,W) [0,1] tensor。
        which
            指标子集，只能是 pixel 指标（psnr/ssim/lpips）；mIoU 单独接口
        resize_mode
            - "ref": cand resize 到 ref 尺寸（VSR 常用做法）
            - "min": ref 和 cand 都 resize 到 min(H,W)（少见）
            - None:  严格 shape，不匹配则 raise

        帧数不匹配总是取 min(T) 前对齐。
        """
        which = tuple(which)
        if not set(which).issubset({"psnr", "ssim", "lpips"}):
            raise ValueError(f"which 只支持 {{psnr, ssim, lpips}}，收到 {which}")

        ref_t = self._as_tensor(ref)
        cand_ts = {name: self._as_tensor(v) for name, v in cands.items()}

        # 帧数对齐：min(T)
        T = min([ref_t.shape[0]] + [t.shape[0] for t in cand_ts.values()])
        if max_frames is not None:
            T = min(T, max_frames)
        ref_t = ref_t[:T]
        cand_ts = {n: t[:T] for n, t in cand_ts.items()}

        # 尺寸对齐
        Href, Wref = ref_t.shape[-2:]
        if resize_mode == "ref":
            for n in list(cand_ts):
                cand_ts[n] = _bicubic_resize(cand_ts[n], Href, Wref)
        elif resize_mode == "min":
            Hmin = min([Href] + [t.shape[-2] for t in cand_ts.values()])
            Wmin = min([Wref] + [t.shape[-1] for t in cand_ts.values()])
            ref_t = _bicubic_resize(ref_t, Hmin, Wmin)
            for n in list(cand_ts):
                cand_ts[n] = _bicubic_resize(cand_ts[n], Hmin, Wmin)
        elif resize_mode is None:
            for n, t in cand_ts.items():
                if t.shape[-2:] != (Href, Wref):
                    raise ValueError(
                        f"[Evaluator] resize_mode=None 但 cand {n!r} shape "
                        f"{tuple(t.shape[-2:])} != ref {(Href, Wref)}"
                    )
        else:
            raise ValueError(f"unknown resize_mode: {resize_mode}")

        # 逐 cand 算指标
        out: Dict[str, Dict[str, float]] = {}
        for name, ct in cand_ts.items():
            out[name] = self._compute_pair(ref_t, ct, which)
        return out

    def evaluate_folder(
        self,
        tag_dir: Union[str, Path],
        ref_lookup: Callable[[str], Path],
        cand_names: Mapping[str, str],
        *,
        which: Sequence[str] = DEFAULT_PIXEL_METRICS,
        resize_mode: str = "ref",
        video_stems: Optional[Iterable[str]] = None,
        out_csv: Optional[str] = "metrics_pixel.csv",
        out_json: Optional[str] = "metrics_pixel.json",
        verbose: bool = True,
    ) -> Dict:
        """扫 tag_dir 下每个子目录（子目录名 = 视频 stem），跑指标并写 CSV/JSON。

        Params
        ------
        tag_dir
            例如 eval_compress_restore/q22_37/
        ref_lookup
            stem -> Path，返回该 stem 的 ref 视频路径（比如 orig .mp4）
        cand_names
            {"cmp": "compressed.mp4", "rst": "restored.mp4"} —— 逻辑名 → 子目录内文件名
        video_stems
            指定子目录白名单；None 表示扫全部
        out_csv / out_json
            相对 tag_dir 的文件名；None 表示不写

        返回 summary dict：{"n_videos", "wall_seconds", "avg": {...}, "per_video": {...}}
        """
        tag_dir = Path(tag_dir)
        if not tag_dir.is_dir():
            raise FileNotFoundError(f"tag_dir not found: {tag_dir}")

        # 枚举子目录
        stems_set = set(video_stems) if video_stems else None
        dirs = []
        for d in sorted(tag_dir.iterdir()):
            if not d.is_dir():
                continue
            if stems_set is not None and d.name not in stems_set:
                continue
            if all((d / fn).is_file() for fn in cand_names.values()):
                dirs.append(d)
        if not dirs:
            raise SystemExit(f"no video dirs under {tag_dir} matching cand files "
                             f"{list(cand_names.values())}")

        rows = []
        t_all = time.monotonic()
        for i, d in enumerate(dirs, 1):
            stem = d.name
            try:
                ref_p = ref_lookup(stem)
            except FileNotFoundError as e:
                if verbose:
                    print(f"[{i}/{len(dirs)}] {stem}: SKIP ({e})")
                continue
            cands_p = {n: str(d / fn) for n, fn in cand_names.items()}
            if verbose:
                print(f"[{i}/{len(dirs)}] {stem}")

            res = self.evaluate(str(ref_p), cands_p, which=which, resize_mode=resize_mode)

            # 展平成一行：video, T, H, W, {name}_{metric}...
            ref_t = self._as_tensor(str(ref_p))
            T, _, H, W = ref_t.shape
            row = {"video": stem, "T": int(T), "H": int(H), "W": int(W)}
            for name, scores in res.items():
                for k, v in scores.items():
                    row[f"{name}_{k}"] = v
            rows.append(row)
            del ref_t
            gc.collect()
            torch.cuda.empty_cache()

            if verbose:
                for name, scores in res.items():
                    line = f"  [{name}] " + "  ".join(
                        f"{k}={v:.4f}" for k, v in scores.items()
                    )
                    print(line)

        if not rows:
            raise SystemExit("no video produced valid metrics")

        # 均值 summary
        metric_keys = [k for k in rows[0] if k not in ("video", "T", "H", "W")]
        avg = {k: float(sum(r[k] for r in rows) / len(rows)) for k in metric_keys}
        summary = {
            "n_videos": len(rows),
            "wall_seconds": round(time.monotonic() - t_all, 1),
            "avg": {k: round(v, 4) for k, v in avg.items()},
            "per_video": {r["video"]: {k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in r.items() if k != "video"}
                          for r in rows},
        }

        # 写文件
        if out_csv:
            cols = ["video", "T", "H", "W"] + metric_keys
            with (tag_dir / out_csv).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items()})
        if out_json:
            (tag_dir / out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))

        if verbose:
            print("\n=== overall ===")
            print(json.dumps(summary["avg"], indent=2))
            print(f"\nwall={summary['wall_seconds']}s, n_videos={summary['n_videos']}")
            if out_csv:
                print(f"csv:  {tag_dir/out_csv}")
            if out_json:
                print(f"json: {tag_dir/out_json}")

        return summary

    # -------- mIoU 三种模式（薄壳，主要为 API 统一） --------

    @staticmethod
    def miou_from_masks(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
        from .miou import miou
        return miou(pred_mask, gt_mask)

    def miou_from_videos(
        self,
        pred_video: str,
        gt_video: str,
        *,
        yolo_ckpt: str = "yolov8n.pt",
        classes: Optional[Sequence[str]] = None,
        conf: float = 0.25,
        dilate: int = 0,
    ) -> float:
        from .miou import miou_from_videos
        return miou_from_videos(
            pred_video=pred_video, gt_video=gt_video,
            yolo_ckpt=yolo_ckpt, classes=classes, conf=conf,
            dilate=dilate, device=self.device,
        )

    @staticmethod
    def miou_from_boxes(
        pred_frames: Sequence[Sequence[Tuple[float, float, float, float]]],
        gt_frames: Sequence[Sequence[Tuple[float, float, float, float]]],
        W: int, H: int,
    ) -> Dict:
        """pred_frames/gt_frames 每帧一个 xyxy list；返回 {miou, n_scored, ...}。

        VisDrone 风格：两组预测框各自光栅化为 mask 再算 mIoU。
        """
        import numpy as np

        def _boxes_to_mask(xyxy_list, W, H):
            m = torch.zeros(H, W, dtype=torch.bool)
            for x1, y1, x2, y2 in xyxy_list:
                x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
                x2i, y2i = min(W, int(round(x2))), min(H, int(round(y2)))
                if x2i > x1i and y2i > y1i:
                    m[y1i:y2i, x1i:x2i] = True
            return m

        T = min(len(pred_frames), len(gt_frames))
        ious, n_both_empty = [], 0
        for i in range(T):
            pm = _boxes_to_mask(pred_frames[i], W, H)
            gm = _boxes_to_mask(gt_frames[i], W, H)
            inter = (pm & gm).sum().item()
            union = (pm | gm).sum().item()
            if union == 0:
                n_both_empty += 1
            else:
                ious.append(inter / union)
        return {
            "miou": float(np.mean(ious)) if ious else None,
            "n_scored": len(ious),
            "n_both_empty": n_both_empty,
            "T": T,
        }

    @staticmethod
    def miou_bbox_match(
        pred_frames,
        ref_frames,
        *,
        min_side: float = 32.0,
        apply_min_side_to_pred: bool = True,
        match: str = "greedy",
    ) -> Dict:
        """bbox 匹配式 mIoU（松弛版）。

        规则：一对一 IoU 最大化匹配；漏检 IoU=0；误检不惩罚；
        最短边 < min_side 双向剔除；micro-average。

        详见 metrics.miou.miou_bbox_match 文档。
        """
        from .miou import miou_bbox_match
        return miou_bbox_match(
            pred_frames, ref_frames,
            min_side=min_side,
            apply_min_side_to_pred=apply_min_side_to_pred,
            match=match,
        )

    # -------- 内部工具 --------

    def _as_tensor(self, x: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return read_video_tchw_float(str(x))       # (T,3,H,W) [0,1] CPU

    def _compute_pair(
        self, ref: torch.Tensor, cand: torch.Tensor, which: Sequence[str]
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if "psnr" in which:
            scores["psnr"] = psnr_fn(cand, ref)
        if "ssim" in which:
            scores["ssim"] = _ssim_batched(
                cand, ref, device=self.device, batch=self.ssim_batch,
            )
        if "lpips" in which:
            scores["lpips"] = lpips_metric(
                cand, ref, net=self.lpips_net, device=self.device,
                batch=self.lpips_batch,
            )
        return scores
