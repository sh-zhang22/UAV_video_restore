#!/usr/bin/env python3
"""在干净版测试集上独立扫描 conf / tracker / max_gap / min_track_len，
每次只动一维，其它固定在起点 (conf=0.05, tracker=bytetrack.yaml, max_gap=8, min_track_len=1)。

假设参数解耦，最终把每维最优拼成推荐 config。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

# ─── 起点 ───
ANCHOR = dict(
    conf=0.05,
    tracker="bytetrack.yaml",
    max_gap=8,
    min_track_len=1,
)

# ─── 各维扫描点（起点会重复，脚本内去重） ───
SWEEP = {
    "conf":          [0.02, 0.05, 0.10, 0.15],
    "tracker":       ["bytetrack.yaml", "trackers/bytetrack_loose.yaml",
                      "trackers/bytetrack_permissive.yaml"],
    "max_gap":       [4, 8, 16, 32],
    "min_track_len": [1, 2, 3, 5],
}


def tag_of(cfg):
    tracker_short = {
        "bytetrack.yaml": "def",
        "trackers/bytetrack_loose.yaml": "loose",
        "trackers/bytetrack_permissive.yaml": "perm",
    }.get(cfg["tracker"], cfg["tracker"].split("/")[-1].replace(".yaml", ""))
    return (f"conf{cfg['conf']:.2f}_{tracker_short}_"
            f"gap{cfg['max_gap']}_mtl{cfg['min_track_len']}")


def build_configs(anchor, sweep):
    """一维扫描：每个维度 K 个点 → K 个 config。跨维度共享 anchor 那一点。"""
    seen = set()
    configs = []
    for dim, values in sweep.items():
        for v in values:
            cfg = dict(anchor)
            cfg[dim] = v
            key = tag_of(cfg)
            if key in seen:
                continue
            seen.add(key)
            cfg["_dim"] = dim
            configs.append(cfg)
    return configs


def run_one(cfg, out_root, device, model, imgsz):
    out_dir = out_root / tag_of(cfg)
    if (out_dir / "summary.json").is_file():
        print(f"  [skip] exists: {out_dir}")
        return json.load(open(out_dir / "summary.json"))

    cmd = [
        "python", "eval_yolo_visdrone_track_interp.py",
        "--model", model,
        "--model_type", "visdrone",
        "--imgsz", str(imgsz),
        "--conf", str(cfg["conf"]),
        "--tracker", cfg["tracker"],
        "--max_gap", str(cfg["max_gap"]),
        "--min_track_len", str(cfg["min_track_len"]),
        "--device", device,
        "--split", "test-dev",
        "--out", str(out_dir),
    ]
    print(f"  cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.load(open(out_dir / "summary.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, default=Path("eval_v6_sweep"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--model", default="ckpts_yolo/v9e/best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    configs = build_configs(ANCHOR, SWEEP)
    print(f"[plan] {len(configs)} configs to run, anchor={ANCHOR}")
    for i, c in enumerate(configs):
        print(f"  [{i+1}] dim={c['_dim']:14s} {tag_of(c)}")

    results = []
    t0 = time.monotonic()
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['_dim']}={cfg[cfg['_dim']]}  tag={tag_of(cfg)}")
        r = run_one(cfg, args.out_root, args.device, args.model, args.imgsz)
        results.append({
            "dim": cfg["_dim"],
            "conf": cfg["conf"],
            "tracker": cfg["tracker"],
            "max_gap": cfg["max_gap"],
            "min_track_len": cfg["min_track_len"],
            "tag": tag_of(cfg),
            "miou": r["overall_miou_frame_avg"],
            "n_videos": r["n_videos"],
            "n_frames_scored": r["n_frames_scored"],
            "wall_seconds": r["wall_seconds"],
        })
    dt = time.monotonic() - t0

    # leaderboard.csv
    lb = args.out_root / "leaderboard.csv"
    with open(lb, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in sorted(results, key=lambda x: -x["miou"]):
            w.writerow(r)

    # per-dim best
    per_dim = {}
    for r in results:
        d = r["dim"]
        if d not in per_dim or r["miou"] > per_dim[d]["miou"]:
            per_dim[d] = r

    # 推荐 config: 每维最优拼起来
    reco = dict(ANCHOR)
    for d, best in per_dim.items():
        reco[d] = best[d]

    summary = {
        "anchor": ANCHOR,
        "sweep": SWEEP,
        "n_configs": len(results),
        "wall_seconds_total": round(dt, 1),
        "per_dim_best": {d: {"value": r[d], "miou": r["miou"], "tag": r["tag"]}
                        for d, r in per_dim.items()},
        "recommended_config": reco,
        "recommended_tag": tag_of(reco),
    }
    (args.out_root / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("=== per-dim best ===")
    for d, r in per_dim.items():
        print(f"  {d:14s} → {d}={r[d]:<30}  miou={r['miou']:.4f}  tag={r['tag']}")
    print("\n=== recommended (每维最优拼接) ===")
    print(f"  {reco}")
    print(f"  tag={tag_of(reco)}")
    print(f"\n[timing] total {dt/60:.1f} min for {len(results)} configs")
    print(f"[out] {args.out_root}/leaderboard.csv, sweep_summary.json")


if __name__ == "__main__":
    main()
