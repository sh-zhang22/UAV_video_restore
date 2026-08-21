"""端到端测试：通过 Recover 接口跑 SeedVR2-3B。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover import Recover, available_methods

print("methods:", available_methods())

t0 = time.time()
out = Recover(
    video_path="test.mp4",
    recovered_path="test_recovered_seedvr2_3b.mp4",
    ckpt_path="third_party/SeedVR/ckpts/seedvr2_ema_3b.pth",
    method="seedvr2_3b",
    device="cuda:1",
    method_kwargs={"res_h": 720, "res_w": 960, "sp_size": 1, "seed": 666},
)
print(f"DONE: {out} in {time.time()-t0:.1f}s")
