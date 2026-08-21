"""端到端测试：通过 Recover 接口跑 SeedVR2-7B（默认单卡，OOM 时切多卡）。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import Recover, available_methods

print("methods:", available_methods())

# method_kwargs 默认 sp_size=1、nproc_per_node=sp_size；
# 单卡 OOM 时把 sp_size 和 nproc_per_node 一起改成 2/4 即可启用 sequence parallel。
SP_SIZE = int(os.environ.get("SP_SIZE", "1"))
DEVICE = os.environ.get("DEVICE", "cuda:1")  # SP_SIZE>1 时改成 "0,1" 这样

t0 = time.time()
out = Recover(
    video_path="test.mp4",
    recovered_path="test_recovered_seedvr2_7b.mp4",
    ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_7b.pth",
    method="seedvr2_7b",
    device=DEVICE,
    method_kwargs={
        "res_h": 720, "res_w": 960, "seed": 666,
        "sp_size": SP_SIZE, "nproc_per_node": SP_SIZE,
    },
)
print(f"DONE: {out} in {time.time()-t0:.1f}s")
