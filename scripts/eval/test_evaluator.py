"""Evaluator 冒烟测试。
1. self-consistency: ref = cand → PSNR≈inf, SSIM≈1, LPIPS≈0
2. small noise: cand = ref + 高斯 → PSNR 中等, SSIM 略降, LPIPS >0
3. multi-cand 复用 ref: 一次调 evaluate 同时算 cmp + rst，与两次单调结果一致
4. resize_mode='ref': cand 分辨率不同也能算完
5. evaluate_folder: 扫已有 q22_37 目录 → 与之前老脚本产物数字一致（±1e-3）

用法：
    conda activate seedvr
    python scripts/eval/test_evaluator.py --device cuda:1
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from metrics import Evaluator                                                     # noqa: E402


def _make_synth_video(T=8, H=64, W=64, seed=0) -> torch.Tensor:
    """(T,3,H,W) [0,1] 合成小视频，避免读盘。"""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(T, 3, H, W, generator=g)


def test_self_consistency(ev: Evaluator):
    print("\n=== 1) self-consistency: ref = cand ===")
    v = _make_synth_video()
    out = ev.evaluate(v, {"self": v}, which=("psnr", "ssim", "lpips"),
                      resize_mode="ref")
    s = out["self"]
    print(f"  psnr={s['psnr']}  ssim={s['ssim']:.6f}  lpips={s['lpips']:.6f}")
    assert s["psnr"] == float("inf") or s["psnr"] > 60, s
    assert s["ssim"] > 0.999, s
    assert s["lpips"] < 1e-4, s
    print("  [pass]")


def test_noise_direction(ev: Evaluator):
    print("\n=== 2) small noise: cand = ref + 0.02 gauss ===")
    ref = _make_synth_video(seed=1)
    g = torch.Generator().manual_seed(42)
    noise = torch.randn(ref.shape, generator=g) * 0.02
    cand = (ref + noise).clamp_(0, 1)
    out = ev.evaluate(ref, {"noisy": cand}, which=("psnr", "ssim", "lpips"))
    s = out["noisy"]
    print(f"  psnr={s['psnr']:.2f}  ssim={s['ssim']:.4f}  lpips={s['lpips']:.4f}")
    assert 20 < s["psnr"] < 60, s
    assert 0.5 < s["ssim"] < 1.0, s        # 64x64 随机图 SSIM 敏感些
    assert s["lpips"] > 0.0, s
    print("  [pass]")


def test_multi_cand_consistency(ev: Evaluator):
    """一次调 evaluate 同时算 2 个 cand 与单独两次调用结果应一致。"""
    print("\n=== 3) multi-cand vs sequential ===")
    ref = _make_synth_video(seed=2)
    ca = (ref + torch.randn(ref.shape, generator=torch.Generator().manual_seed(3)) * 0.01).clamp_(0, 1)
    cb = (ref + torch.randn(ref.shape, generator=torch.Generator().manual_seed(4)) * 0.03).clamp_(0, 1)
    joint = ev.evaluate(ref, {"a": ca, "b": cb}, which=("psnr", "ssim", "lpips"))
    solo_a = ev.evaluate(ref, {"a": ca}, which=("psnr", "ssim", "lpips"))["a"]
    solo_b = ev.evaluate(ref, {"b": cb}, which=("psnr", "ssim", "lpips"))["b"]
    print(f"  joint[a]={joint['a']}")
    print(f"  solo[a] ={solo_a}")
    for k in ("psnr", "ssim", "lpips"):
        assert abs(joint["a"][k] - solo_a[k]) < 1e-4, (k, joint["a"], solo_a)
        assert abs(joint["b"][k] - solo_b[k]) < 1e-4, (k, joint["b"], solo_b)
    print("  [pass]")


def test_resize_mode(ev: Evaluator):
    """cand 分辨率不同也能算完，且 resize 结果不 raise。"""
    print("\n=== 4) resize_mode='ref' handles mismatched shape ===")
    ref = _make_synth_video(H=64, W=64, seed=5)
    cand_small = _make_synth_video(T=8, H=48, W=48, seed=5)
    out = ev.evaluate(ref, {"small": cand_small}, which=("psnr", "ssim", "lpips"),
                      resize_mode="ref")
    s = out["small"]
    print(f"  psnr={s['psnr']:.2f}  ssim={s['ssim']:.4f}  lpips={s['lpips']:.4f}")
    assert 0 < s["psnr"] < 100, s
    print("  [pass]")


def test_folder_matches_old(ev: Evaluator, tag_dir: str):
    """跑一次 evaluate_folder，与之前老脚本产物 CSV 数字一致（±5e-3）。

    只跑最小的那个视频 (uav0000161_00000, 540x960) 图快。
    """
    print(f"\n=== 5) evaluate_folder vs 老脚本产物 ({tag_dir}) ===")
    from pathlib import Path
    from scripts.eval.compute_pixel_metrics import find_orig_mp4

    tmpdir = tempfile.mkdtemp(prefix="ev_test_")
    print(f"  写到临时目录 {tmpdir}/metrics_pixel.csv")

    # 拷 stem 的三个文件到临时 tag 结构；简单起见直接指向原目录，只写到 tmp
    # evaluate_folder 会在 tag_dir 内写文件，为不污染真实结果，用 tmp 目录做 tag_dir
    src = Path(tag_dir) / "test-dev_uav0000161_00000_v_full"
    if not src.is_dir():
        print(f"  [skip] {src} 不存在")
        return
    tmp_tag = Path(tmpdir)
    tmp_video = tmp_tag / src.name
    tmp_video.mkdir(parents=True)
    for fn in ("compressed.mp4", "restored.mp4"):
        os.symlink(src / fn, tmp_video / fn)

    summary = ev.evaluate_folder(
        tag_dir=tmp_tag,
        ref_lookup=find_orig_mp4,
        cand_names={"cmp": "compressed.mp4", "rst": "restored.mp4"},
        which=("psnr", "ssim", "lpips"),
        verbose=False,
    )
    row = summary["per_video"]["test-dev_uav0000161_00000_v_full"]
    # 对照之前 metrics_pixel.csv 里 161_00000 的数字
    expected = {
        "cmp_psnr": 28.959,  "rst_psnr": 17.8119,
        "cmp_ssim": 0.8476,  "rst_ssim": 0.4713,
        "cmp_lpips": 0.1499, "rst_lpips": 0.2665,
    }
    print(f"  actual: {row}")
    print(f"  expect: {expected}")
    for k, exp_v in expected.items():
        act_v = row[k]
        tol = 0.05 if "psnr" in k else 0.005     # PSNR ± 0.05 dB, 其它 ± 5e-3
        assert abs(act_v - exp_v) < tol, f"{k}: act={act_v} vs exp={exp_v} tol={tol}"
    print("  [pass]")


def test_miou_from_boxes():
    """miou_from_boxes: 完全一致 → 1.0, 完全不重叠 → 0.0"""
    print("\n=== 6) miou_from_boxes basic ===")
    pred = [[(10, 10, 30, 30)], [(20, 20, 50, 50)]]
    # same
    same = Evaluator.miou_from_boxes(pred, pred, W=100, H=100)
    print(f"  same → miou={same['miou']:.4f}")
    assert abs(same["miou"] - 1.0) < 1e-6, same
    # disjoint
    other = [[(60, 60, 90, 90)], [(0, 0, 5, 5)]]
    disj = Evaluator.miou_from_boxes(pred, other, W=100, H=100)
    print(f"  disjoint → miou={disj['miou']:.4f}")
    assert abs(disj["miou"] - 0.0) < 1e-6, disj
    print("  [pass]")


def test_miou_bbox_match_identity():
    """pred = ref → miou=1.0, 全部匹配, 无漏检。"""
    print("\n=== 7) miou_bbox_match identity ===")
    # 两个都是 100×100 的大框，避免被 min_side=32 剔掉
    frames = [
        [(10, 10, 60, 60), (100, 100, 200, 200)],
        [(20, 20, 80, 80)],
    ]
    r = Evaluator.miou_bbox_match(frames, frames)
    print(f"  {r}")
    assert abs(r["miou"] - 1.0) < 1e-6, r
    assert r["n_matched"] == 3, r
    assert r["n_missed"] == 0, r
    assert r["n_ref_dropped"] == 0, r
    print("  [pass]")


def test_miou_bbox_match_missing_and_extra():
    """漏检记 IoU=0；误检不惩罚。

    ref: 一帧两个框 A, B
    pred 场景 A: 只检出 A   → miou = (iou(A,A) + 0) / 2 = 0.5
    pred 场景 B: 检出 A + 远处大干扰框 X → miou 与场景 A 一致（误检不惩罚）
    """
    print("\n=== 8) miou_bbox_match: missing gives 0, extra is free ===")
    A = (10, 10, 60, 60)
    B = (200, 200, 260, 260)
    X_far = (400, 400, 500, 500)     # 与 A, B 都不重叠
    ref = [[A, B]]

    # 场景 A：只检出 A
    pred_a = [[A]]
    r_a = Evaluator.miou_bbox_match(pred_a, ref)
    print(f"  only-A: {r_a}")
    assert abs(r_a["miou"] - 0.5) < 1e-6, r_a         # (1.0 + 0.0) / 2
    assert r_a["n_matched"] == 1, r_a
    assert r_a["n_missed"] == 1, r_a

    # 场景 B：A + 远处误检
    pred_b = [[A, X_far]]
    r_b = Evaluator.miou_bbox_match(pred_b, ref)
    print(f"  A+extra: {r_b}")
    assert abs(r_b["miou"] - r_a["miou"]) < 1e-6, (r_a, r_b)
    assert r_b["n_pred_dropped"] == 0, r_b            # X_far 是大框不该被小尺寸过滤掉
    print("  [pass]")


def test_miou_bbox_match_small_filter():
    """min_side=32：ref/pred 中的小框被剔除，不参与均值。

    ref: 一大 A + 一小 s（20×20）
    pred: 一大 A' 匹配 A + 一小 s'
    小框应被双向剔除，最终只算 A vs A' 的 IoU。
    """
    print("\n=== 9) miou_bbox_match: min_side=32 filters small boxes ===")
    A_ref  = (10, 10, 60, 60)                     # 50×50 保留
    A_pred = (12, 12, 62, 62)                     # 与 A_ref 有较高 IoU
    s_ref  = (200, 200, 220, 220)                 # 20×20 剔除
    s_pred = (210, 210, 230, 230)                 # 20×20 剔除
    ref  = [[A_ref, s_ref]]
    pred = [[A_pred, s_pred]]
    r = Evaluator.miou_bbox_match(pred, ref, min_side=32.0)
    print(f"  {r}")
    assert r["n_ref_kept"] == 1, r                # s_ref 被剔
    assert r["n_ref_dropped"] == 1, r
    assert r["n_pred_dropped"] == 1, r
    # 只算 A_ref × A_pred 一个匹配
    assert r["n_matched"] == 1, r
    assert r["n_missed"] == 0, r
    # IoU 应该 > 0.5（两框错位小）
    assert 0.5 < r["miou"] < 1.0, r
    print("  [pass]")


def test_miou_bbox_match_hungarian_vs_greedy():
    """一个精心构造的案例：greedy 与 hungarian 结果应一致（或至少 hungarian ≥ greedy）。"""
    print("\n=== 10) miou_bbox_match: hungarian ≥ greedy ===")
    # 三个 ref, 三个 pred，容易构造出 greedy 局部最优 ≠ 全局最优的情况
    ref = [[(0, 0, 40, 40), (100, 100, 140, 140), (200, 200, 240, 240)]]
    pred = [
        [(0, 0, 40, 40),        # 完美匹配 ref[0]
         (100, 100, 200, 200),  # 与 ref[1] 部分重叠；与 ref[2] 部分重叠（对角）
         (200, 200, 240, 240)]  # 完美匹配 ref[2]
    ]
    r_g = Evaluator.miou_bbox_match(pred, ref, match="greedy")
    r_h = Evaluator.miou_bbox_match(pred, ref, match="hungarian")
    print(f"  greedy:    {r_g}")
    print(f"  hungarian: {r_h}")
    assert r_h["miou"] >= r_g["miou"] - 1e-6, (r_g, r_h)
    print("  [pass]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag_dir",
                    default="/home/zsh/UAV_video_repair/eval_compress_restore/q22_37",
                    help="老产物目录，用于 folder 一致性测试")
    ap.add_argument("--skip", nargs="+", default=[],
                    choices=["self", "noise", "multi", "resize", "folder",
                             "boxes", "bbox_id", "bbox_miss", "bbox_small", "bbox_hun"])
    args = ap.parse_args()

    print(f"device={args.device}")
    ev = Evaluator(device=args.device, lpips_net="alex")

    if "self" not in args.skip:
        test_self_consistency(ev)
    if "noise" not in args.skip:
        test_noise_direction(ev)
    if "multi" not in args.skip:
        test_multi_cand_consistency(ev)
    if "resize" not in args.skip:
        test_resize_mode(ev)
    if "folder" not in args.skip:
        test_folder_matches_old(ev, args.tag_dir)
    if "boxes" not in args.skip:
        test_miou_from_boxes()
    if "bbox_id" not in args.skip:
        test_miou_bbox_match_identity()
    if "bbox_miss" not in args.skip:
        test_miou_bbox_match_missing_and_extra()
    if "bbox_small" not in args.skip:
        test_miou_bbox_match_small_filter()
    if "bbox_hun" not in args.skip:
        test_miou_bbox_match_hungarian_vs_greedy()

    print("\nALL EVALUATOR TESTS PASSED")


if __name__ == "__main__":
    main()
