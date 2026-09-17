"""SeedVR2-3B + ControlNet-Lite 侧枝 训练冒烟脚本。

真实数据未到位前，用 test.mp4 + 随机稀疏 0/1 mask 跑 5 步，验证：
- ControlledDiT forward / backward / optimizer.step 通路
- 侧枝参数（mask_stem + side_blocks + zero_convs）确实收到梯度、可 save
- 主干严格 frozen（trainable = 侧枝参数量）
- 显存不炸（A800 80G）

产出：runs_ctrlnet/ctrlnet.pt

用法：
    python train_seedvr2_3b_ctrlnet.py                       # 随机 50% 稀疏 mask
    python train_seedvr2_3b_ctrlnet.py --mask ones          # 全 1 mask
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from recover import Recover, available_methods  # noqa: E402


def _make_sparse_random_mask(out_path: str, h: int = 720, w: int = 1280, p: float = 0.5, seed: int = 0) -> str:
    from PIL import Image
    rng = np.random.default_rng(seed)
    m = (rng.random((h, w)) < p).astype(np.uint8) * 255
    Image.fromarray(m).save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="sparse_random", choices=["ones", "sparse_random"])
    parser.add_argument("--save_dir", default="runs_ctrlnet")
    parser.add_argument("--train_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ctrlnet_K", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("methods:", available_methods())

    if args.mask == "ones":
        mask_path = None
    else:
        mask_path = _make_sparse_random_mask(
            os.path.abspath("mask_train_ctrlnet_sparse.png")
        )

    save_dir_abs = os.path.abspath(args.save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)

    # 冒烟阶段用较小 res：训练要多存激活，720x960 会挤爆 A800（另一个进程也占了 10G+）。
    # 512x768 面积约为 720x960 的 57%，激活约减半；真实训练到位后再放大。
    method_kwargs = {
        "res_h": 512,
        "res_w": 768,
        "sp_size": 1,
        "seed": 666,
        "train_mode": True,
        "save_dir": save_dir_abs,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "ctrlnet_K": args.ctrlnet_K,
    }
    if mask_path is not None:
        method_kwargs["mask_path"] = mask_path

    print(f"mask={args.mask} (path={mask_path}), save_dir={save_dir_abs}, K={args.ctrlnet_K}")

    t0 = time.time()
    out = Recover(
        video_path="test.mp4",
        recovered_path=os.path.join(args.save_dir, "_placeholder.mp4"),  # 训练模式不产出 mp4
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b_ctrlnet",
        device=args.device,
        method_kwargs=method_kwargs,
    )
    print(f"DONE: save_dir={out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
