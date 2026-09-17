"""SeedVR2-3B 层裁剪 + KD 蒸馏冒烟脚本。

策略：32 层教师 → 20 层学生（前 10 mm-layer 全保 + 后 22 均匀取 10）。
Loss = λ_out · MSE(z_S, z_T) + λ_feat · Σ (1 - cos(h_S[j], h_T[map[j]])) + λ_diff · MSE(z_S, v)

冒烟场景：test.mp4 + 随机 noise/timestep 重复 N 步，验证 forward/backward/save/load 通路。
真实数据到位后再展开 dataloader、真实 loss 权重、更长训练。

用法：
    # 默认 3 步，产出 runs_distill/student.pth
    python train_seedvr2_3b_distill.py

    # 指定步数 / 学习率 / 保存目录
    python train_seedvr2_3b_distill.py --train_steps 5 --lr 1e-5
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import Recover, available_methods  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs_distill")
    parser.add_argument("--train_steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lambda_out", type=float, default=1.0)
    parser.add_argument("--lambda_feat", type=float, default=0.5)
    parser.add_argument("--lambda_diff", type=float, default=1.0)
    parser.add_argument("--student_num_layers", type=int, default=20,
                        help="学生层数（当前只支持 20，layer_map 固定）")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    print("methods:", available_methods())

    save_dir_abs = os.path.abspath(args.save_dir)
    os.makedirs(save_dir_abs, exist_ok=True)

    method_kwargs = {
        "res_h": 720,
        "res_w": 960,
        "sp_size": 1,
        "seed": 666,
        "distill_mode": True,
        "student_num_layers": args.student_num_layers,
        "save_dir": save_dir_abs,
        "student_save_path": "student.pth",
        "train_steps": args.train_steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "distill_lambda_out": args.lambda_out,
        "distill_lambda_feat": args.lambda_feat,
        "distill_lambda_diff": args.lambda_diff,
    }

    print(f"save_dir={save_dir_abs}, train_steps={args.train_steps}, lr={args.lr}")
    print(f"loss weights: out={args.lambda_out}, feat={args.lambda_feat}, diff={args.lambda_diff}")

    t0 = time.time()
    out = Recover(
        video_path="test.mp4",
        recovered_path=os.path.join(args.save_dir, "_placeholder.mp4"),  # 蒸馏不产出 mp4
        ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
        method="seedvr2_3b",     # 走教师 variant + distill_mode 分派到蒸馏分支
        device=args.device,
        method_kwargs=method_kwargs,
    )
    print(f"DONE: save_dir={out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
