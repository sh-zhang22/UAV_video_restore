"""端到端冒烟：SeedVR2-3B + 第 17 通道 mask 控制 + LoRA。

用法：
    # Step 1：无 LoRA + 全 1 mask（应与 baseline seedvr2_3b 视觉等价）
    python test_recover_seedvr2_3b_ctrl.py

    # Step 2：无 LoRA + 稀疏中心方框 mask（OOD，不崩即通过）
    python test_recover_seedvr2_3b_ctrl.py --mask center_box

    # Step 4：加载训练后的 LoRA + 全 1 mask
    python test_recover_seedvr2_3b_ctrl.py --lora_ckpt runs_ctrl/trainable.pt
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from recover import Recover, available_methods  # noqa: E402


def _make_center_box_mask(out_path: str, h: int = 720, w: int = 1280) -> str:
    """生成中心 1/4 边长（即 1/16 面积）的方框 mask，其余 0。"""
    from PIL import Image
    m = np.zeros((h, w), dtype=np.uint8)
    y0, y1 = h // 2 - h // 8, h // 2 + h // 8
    x0, x1 = w // 2 - w // 8, w // 2 + w // 8
    m[y0:y1, x0:x1] = 255
    Image.fromarray(m).save(out_path)
    return out_path


def _make_inv_center_box_mask(out_path: str, h: int = 720, w: int = 1280) -> str:
    """反转 centerbox：周围 = 1（sr 训练见过的语义），中心方框 = 0（OOD）。
    验证「周围正常修复、中心尝试保留原画」的假设。"""
    from PIL import Image
    m = np.full((h, w), 255, dtype=np.uint8)
    y0, y1 = h // 2 - h // 8, h // 2 + h // 8
    x0, x1 = w // 2 - w // 8, w // 2 + w // 8
    m[y0:y1, x0:x1] = 0
    Image.fromarray(m).save(out_path)
    return out_path


def _make_sat_center_box_mask(out_path: str, h: int = 720, w: int = 1280) -> str:
    """超饱和 centerbox：周围 = 1.0（训练语义），中心方框 = 2.0（数值 OOD 探针）。
    走 .npy 通路：load_mask_as_TCHW 里 max>1.5 时不除 255，直接保留 2.0。
    验证「>1 的值能否触发某种抑制修复的效应」。"""
    m = np.ones((h, w), dtype=np.float32)
    y0, y1 = h // 2 - h // 8, h // 2 + h // 8
    x0, x1 = w // 2 - w // 8, w // 2 + w // 8
    m[y0:y1, x0:x1] = 2.0
    np.save(out_path, m)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask",
        default="ones",
        choices=["ones", "center_box", "inv_center_box", "sat_center_box"],
        help=(
            "ones=全 1（sanity check）; "
            "center_box=周围 0 中心 1（当前状态）; "
            "inv_center_box=周围 1 中心 0（尝试保留中心原画）; "
            "sat_center_box=周围 1 中心 2（数值 OOD 探针）"
        ),
    )
    parser.add_argument("--mask_path", default=None,
                        help="显式给一个 mask 文件路径，覆盖 --mask 选项")
    parser.add_argument("--lora_ckpt", default=None,
                        help="加载 trainable.pt（LoRA + vid_in.proj 权重）")
    parser.add_argument("--out", default=None, help="输出 mp4 路径")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("methods:", available_methods())

    # 决定 mask_path
    if args.mask_path is not None:
        mask_path = args.mask_path
        mask_tag = os.path.splitext(os.path.basename(mask_path))[0]
    elif args.mask == "ones":
        mask_path = None                       # None → load_mask_as_TCHW 返回全 1
        mask_tag = "ones"
    elif args.mask == "center_box":
        mask_path = _make_center_box_mask(
            os.path.abspath("mask_center_box.png")
        )
        mask_tag = "centerbox"
    elif args.mask == "inv_center_box":
        mask_path = _make_inv_center_box_mask(
            os.path.abspath("mask_inv_center_box.png")
        )
        mask_tag = "invcenterbox"
    elif args.mask == "sat_center_box":
        mask_path = _make_sat_center_box_mask(
            os.path.abspath("mask_sat_center_box.npy")
        )
        mask_tag = "satcenterbox"
    else:
        raise ValueError(args.mask)

    lora_tag = "_lora" if args.lora_ckpt else ""
    out_path = args.out or f"test_recovered_seedvr2_3b_ctrl_{mask_tag}{lora_tag}.mp4"

    method_kwargs = {
        "res_h": 720,
        "res_w": 960,
        "sp_size": 1,
        "seed": 666,
    }
    if mask_path is not None:
        # 子进程会 chdir 到 SeedVR 根目录 → mask 相对路径会失效，必须转绝对
        method_kwargs["mask_path"] = os.path.abspath(mask_path)
    if args.lora_ckpt is not None:
        method_kwargs["lora_ckpt"] = os.path.abspath(args.lora_ckpt)

    print(f"mask={args.mask} (path={mask_path}), lora_ckpt={args.lora_ckpt}")
    print(f"out={out_path}")

    t0 = time.time()
    out = Recover(
        video_path="test.mp4",
        recovered_path=out_path,
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b_ctrl",
        device=args.device,
        method_kwargs=method_kwargs,
    )
    print(f"DONE: {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
