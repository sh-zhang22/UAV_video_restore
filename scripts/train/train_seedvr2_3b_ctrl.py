"""SeedVR2-3B + mask control 训练冒烟脚本。

真实数据未到位前，用 test.mp4 + 随机稀疏 0/1 mask 跑 5 步，验证：
- forward / backward / optimizer.step 通路
- LoRA + vid_in.proj 参数确实收到梯度、可 save
- 显存不炸（A800 80G）

产出：runs_ctrl/trainable.pt
用法：
    python train_seedvr2_3b_ctrl.py                       # 全 1 mask
    python train_seedvr2_3b_ctrl.py --mask sparse_random  # 随机 50% 稀疏 mask
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import Recover, available_methods  # noqa: E402


def _make_sparse_random_mask(out_path: str, h: int = 720, w: int = 1280, p: float = 0.5, seed: int = 0) -> str:
    """生成随机稀疏 0/1 mask，模拟真实数据的分布。"""
    from PIL import Image
    rng = np.random.default_rng(seed)
    m = (rng.random((h, w)) < p).astype(np.uint8) * 255
    Image.fromarray(m).save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", default="sparse_random",
                        choices=["ones", "sparse_random"])
    parser.add_argument("--save_dir", default="runs_ctrl")
    parser.add_argument("--train_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("methods:", available_methods())

    if args.mask == "ones":
        mask_path = None
    else:
        mask_path = _make_sparse_random_mask(
            os.path.abspath("mask_train_sparse.png")
        )

    # 子进程会 chdir 到 SeedVR 根，save_dir 用绝对路径避免落到 third_party 里
    save_dir_abs = os.path.abspath(args.save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)

    method_kwargs = {
        "res_h": 720,
        "res_w": 960,
        "sp_size": 1,
        "seed": 666,
        "train_mode": True,
        "save_dir": save_dir_abs,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    if mask_path is not None:
        method_kwargs["mask_path"] = mask_path

    print(f"mask={args.mask} (path={mask_path}), save_dir={save_dir_abs}")

    t0 = time.time()
    out = Recover(
        video_path="test.mp4",
        recovered_path=os.path.join(args.save_dir, "_placeholder.mp4"),  # 训练模式不产出 mp4
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b_ctrl",
        device=args.device,
        method_kwargs=method_kwargs,
    )
    print(f"DONE: save_dir={out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
