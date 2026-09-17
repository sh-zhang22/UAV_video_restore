"""SeedVR2-3B 蒸馏学生（20 层）推理冒烟。

前置：需要先跑 `python train_seedvr2_3b_distill.py` 产出 `runs_distill/student.pth`。

用法：
    python test_recover_seedvr2_3b_student.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import Recover, available_methods

print("methods:", available_methods())

STUDENT_CKPT = "runs_distill/student.pth"
if not os.path.isfile(STUDENT_CKPT):
    raise FileNotFoundError(
        f"未找到学生 ckpt: {STUDENT_CKPT}\n"
        f"请先跑 `python train_seedvr2_3b_distill.py` 产出。"
    )

t0 = time.time()
out = Recover(
    video_path="test.mp4",
    recovered_path="test_recovered_seedvr2_3b_student.mp4",
    ckpt_path=STUDENT_CKPT,
    method="seedvr2_3b",           # 学生复用教师 variant，只在 kwargs 里覆盖层数
    device="cuda:1",
    method_kwargs={
        "res_h": 720, "res_w": 960, "sp_size": 1, "seed": 666,
        "student_num_layers": 20,  # 关键：触发 _build_runner 派生 20 层 config
        # 学生 ckpt 目录里没有 VAE；VAE 复用官方 ckpt 同目录的 ema_vae.pth
        "vae_ckpt": "third_party/SeedVR/ckpts/ema_vae.pth",
    },
)
print(f"DONE: {out} in {time.time()-t0:.1f}s")
