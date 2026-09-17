"""端到端冒烟：SeedVR2-3B + ControlNet-Lite 侧枝（mask 条件输入）。

用法：
    # N2：无侧枝权重 + 全 1 mask（应与 baseline seedvr2_3b 视觉一致，zero-init 保证）
    python test_recover_seedvr2_3b_ctrlnet.py

    # N2b：无侧枝权重 + 中心方框 mask（同样应等价 baseline，因为 zero_conv 输出恒 0）
    python test_recover_seedvr2_3b_ctrlnet.py --mask center_box

    # N4：加载训练后的侧枝权重
    python test_recover_seedvr2_3b_ctrlnet.py --ctrlnet_ckpt runs_ctrlnet/ctrlnet.pt --mask center_box
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


def _make_zeros_mask(out_path: str, h: int = 720, w: int = 1280) -> str:
    """生成全 0 mask，用来验证 zero-init 隔离性（无论 mask 是什么，未训权重时输出都应等价 baseline）。"""
    from PIL import Image
    m = np.zeros((h, w), dtype=np.uint8)
    Image.fromarray(m).save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mask",
        default="ones",
        choices=["ones", "zeros", "center_box"],
        help="ones=全 1（sanity check）; zeros=全 0（zero-init 隔离性验证）; center_box=中心方框（OOD 侧枝激活）",
    )
    parser.add_argument("--mask_path", default=None, help="显式给一个 mask 文件路径，覆盖 --mask 选项")
    parser.add_argument("--ctrlnet_ckpt", default=None, help="加载 ctrlnet.pt（侧枝权重）")
    parser.add_argument("--ctrlnet_K", type=int, default=4, help="侧枝层数，默认 4")
    parser.add_argument("--out", default=None, help="输出 mp4 路径")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("methods:", available_methods())

    if args.mask_path is not None:
        mask_path = args.mask_path
        mask_tag = os.path.splitext(os.path.basename(mask_path))[0]
    elif args.mask == "ones":
        mask_path = None
        mask_tag = "ones"
    elif args.mask == "zeros":
        mask_path = _make_zeros_mask(os.path.abspath("mask_zeros.png"))
        mask_tag = "zeros"
    elif args.mask == "center_box":
        mask_path = _make_center_box_mask(os.path.abspath("mask_center_box.png"))
        mask_tag = "centerbox"
    else:
        raise ValueError(args.mask)

    ckpt_tag = "_trained" if args.ctrlnet_ckpt else ""
    out_path = args.out or f"test_recovered_seedvr2_3b_ctrlnet_{mask_tag}{ckpt_tag}.mp4"

    method_kwargs = {
        "res_h": 720,
        "res_w": 960,
        "sp_size": 1,
        "seed": 666,
        "ctrlnet_K": args.ctrlnet_K,
    }
    if mask_path is not None:
        # 子进程会 chdir 到 SeedVR 根目录 → mask 相对路径会失效，必须转绝对
        method_kwargs["mask_path"] = os.path.abspath(mask_path)
    if args.ctrlnet_ckpt is not None:
        method_kwargs["ctrlnet_ckpt"] = os.path.abspath(args.ctrlnet_ckpt)

    print(f"mask={args.mask} (path={mask_path}), ctrlnet_ckpt={args.ctrlnet_ckpt}, K={args.ctrlnet_K}")
    print(f"out={out_path}")

    t0 = time.time()
    out = Recover(
        video_path="test.mp4",
        recovered_path=out_path,
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b_ctrlnet",
        device=args.device,
        method_kwargs=method_kwargs,
    )
    print(f"DONE: {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
