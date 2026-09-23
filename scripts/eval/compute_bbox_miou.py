"""对 eval_compress_restore/<tag>/<video>/ 下已有的 yolo_{orig,compressed,restored}_ti.json，
以 orig 为参考，跑松弛版 bbox 匹配式 mIoU（漏检 IoU=0；误检不惩罚；min_side=32 双向剔除）。

坐标空间：
- yolo_orig_ti.json / yolo_compressed_ti.json 都在 orig 空间
- yolo_restored_ti.json 在 restored 空间（SeedVR 内部 NaResize + DivisibleCrop）
  → 用 eval_compress_restore.restored_to_orig_boxes 反变换到 orig 空间再比

产物：
- <tag>/metrics_bbox_miou.csv
- <tag>/metrics_bbox_miou.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from metrics import Evaluator                                                  # noqa: E402
from scripts.eval.eval_compress_restore import restored_to_orig_boxes          # noqa: E402


def _bbox_row(pred_frames, ref_frames, *, min_side, match):
    """跑一遍 miou_bbox_match，返回浮点/整型友好的展平 dict。"""
    r = Evaluator.miou_bbox_match(
        pred_frames, ref_frames,
        min_side=min_side, apply_min_side_to_pred=True, match=match,
    )
    return {
        "miou": r["miou"] if r["miou"] is not None else float("nan"),
        "avg_iou_matched": r["avg_iou_matched"] if r["avg_iou_matched"] is not None else float("nan"),
        "n_scored": r["n_scored"],
        "n_matched": r["n_matched"],
        "n_missed": r["n_missed"],
        "n_ref_kept": r["n_ref_kept"],
        "n_ref_dropped": r["n_ref_dropped"],
        "n_pred_dropped": r["n_pred_dropped"],
        "T": r["T"],
    }


def process_one(video_dir: Path, *, min_side: float, match: str,
                max_area: int, verbose: bool = True) -> dict | None:
    """处理单个视频目录，返回一行 dict；缺文件返回 None。"""
    meta_p = video_dir / "meta.json"
    j_orig = video_dir / "yolo_orig_ti.json"
    j_cmp  = video_dir / "yolo_compressed_ti.json"
    j_rst  = video_dir / "yolo_restored_ti.json"
    for p in (meta_p, j_orig, j_cmp, j_rst):
        if not p.is_file():
            if verbose:
                print(f"  [skip] {video_dir.name}: 缺 {p.name}")
            return None

    meta = json.loads(meta_p.read_text())
    W_o, H_o = meta["resolution"]
    W_r, H_r = meta.get("restored_resolution", meta["resolution"])

    pred_orig = json.loads(j_orig.read_text())
    pred_cmp  = json.loads(j_cmp.read_text())
    pred_rst  = json.loads(j_rst.read_text())

    # restored → orig 空间。effective_max_area 复刻 process_video 的逻辑：
    # 若 orig 面积 ≤ max_area，SeedVR 用原分辨率 → effective_max_area = W_o*H_o；
    # 否则 rescale 到 max_area 面积上限。
    import math as _m
    if W_o * H_o > max_area:
        _s = _m.sqrt(max_area / (W_o * H_o))
        _seedvr_h = round(H_o * _s)
        _seedvr_w = round(W_o * _s)
    else:
        _seedvr_h, _seedvr_w = H_o, W_o
    effective_max_area = _seedvr_h * _seedvr_w
    if (W_r, H_r) != (W_o, H_o):
        pred_rst_o = restored_to_orig_boxes(pred_rst, W_o, H_o, W_r, H_r, effective_max_area)
    else:
        pred_rst_o = pred_rst

    cmp_row = _bbox_row(pred_cmp, pred_orig, min_side=min_side, match=match)
    rst_row = _bbox_row(pred_rst_o, pred_orig, min_side=min_side, match=match)

    row = {
        "video": video_dir.name,
        "T": min(len(pred_orig), len(pred_cmp), len(pred_rst_o)),
        "W_o": W_o, "H_o": H_o, "W_r": W_r, "H_r": H_r,
    }
    for k, v in cmp_row.items():
        row[f"cmp_{k}"] = v
    for k, v in rst_row.items():
        row[f"rst_{k}"] = v

    if verbose:
        print(f"  cmp: miou={cmp_row['miou']:.4f}  matched={cmp_row['n_matched']}/"
              f"{cmp_row['n_ref_kept']}  missed={cmp_row['n_missed']}")
        print(f"  rst: miou={rst_row['miou']:.4f}  matched={rst_row['n_matched']}/"
              f"{rst_row['n_ref_kept']}  missed={rst_row['n_missed']}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="q22_37")
    ap.add_argument("--out", type=Path, default=Path("eval_compress_restore"))
    ap.add_argument("--videos", nargs="+", default=None,
                    help="视频 stem 列表；不给则跑该 tag 下全部")
    ap.add_argument("--min_side", type=float, default=32.0,
                    help="最短边阈值，默认 32（小于该值的框双向剔除）")
    ap.add_argument("--match", default="greedy", choices=["greedy", "hungarian"])
    ap.add_argument("--max_area", type=int, default=720 * 1280,
                    help="SeedVR 目标像素面积，用于 restored→orig 反变换（与 eval_compress_restore 默认一致）")
    args = ap.parse_args()

    tag_dir = args.out / args.tag
    if not tag_dir.is_dir():
        raise SystemExit(f"tag dir not found: {tag_dir}")

    stems = set(args.videos) if args.videos else None
    dirs = []
    for d in sorted(tag_dir.iterdir()):
        if not d.is_dir():
            continue
        if stems is not None and d.name not in stems:
            continue
        dirs.append(d)
    if not dirs:
        raise SystemExit(f"no video dirs under {tag_dir}")

    print(f"[plan] tag={args.tag}  n_videos={len(dirs)}  "
          f"min_side={args.min_side}  match={args.match}")

    rows = []
    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] {d.name}")
        row = process_one(d, min_side=args.min_side, match=args.match,
                          max_area=args.max_area)
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit("no video produced metrics")

    # 均值
    metric_keys = [k for k in rows[0] if k not in
                   ("video", "T", "W_o", "H_o", "W_r", "H_r")]
    avg = {k: float(sum(r[k] for r in rows) / len(rows)) for k in metric_keys}
    summary = {
        "n_videos": len(rows),
        "min_side": args.min_side,
        "match": args.match,
        "avg": {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in avg.items()},
        "per_video": {r["video"]: {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in r.items() if k != "video"}
                      for r in rows},
    }

    cols = ["video", "T", "W_o", "H_o", "W_r", "H_r"] + metric_keys
    (tag_dir / "metrics_bbox_miou.csv").open("w", newline="").close()
    with (tag_dir / "metrics_bbox_miou.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    (tag_dir / "metrics_bbox_miou.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== overall ===")
    key_cols = ["cmp_miou", "cmp_n_matched", "cmp_n_missed",
                "rst_miou", "rst_n_matched", "rst_n_missed"]
    print(json.dumps({k: summary["avg"][k] for k in key_cols}, indent=2))
    print(f"\ncsv:  {tag_dir/'metrics_bbox_miou.csv'}")
    print(f"json: {tag_dir/'metrics_bbox_miou.json'}")


if __name__ == "__main__":
    main()
