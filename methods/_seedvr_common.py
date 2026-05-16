"""SeedVR 适配器公用 dispatch：通过 torchrun 子进程调用 _seedvr_runner.py。

为什么走子进程：
- SeedVR 推断硬编码 ./configs_*/main.yaml、./ckpts/、pos_emb.pt 等相对路径，需要 chdir
  到 SeedVR 根目录。
- common.distributed.basic.init_torch() 强依赖 dist.init_process_group + RANK/WORLD_SIZE
  等环境变量，由 torchrun 注入最干净。
- 子进程退出后显存彻底释放，方便和后续其他 method 共存。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SEEDVR_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "third_party", "SeedVR"))
_RUNNER = os.path.join(_THIS_DIR, "_seedvr_runner.py")

# 默认在 seedvr conda env 里跑（环境里装了 apex/flash_attn 等）
_DEFAULT_TORCHRUN = os.path.expanduser("~/anaconda3/envs/seedvr/bin/torchrun")


def _resolve_device(device: Optional[str]) -> Optional[str]:
    """把 'cuda:2' / 'cuda' / '2' 之类规范化为 CUDA_VISIBLE_DEVICES 用的字符串。

    None 表示沿用调用方环境（不覆盖 CUDA_VISIBLE_DEVICES）。
    """
    if device is None:
        return None
    d = device.strip()
    if d.startswith("cuda:"):
        return d.split(":", 1)[1]
    if d == "cuda":
        return None
    return d  # 假定是 "0" / "0,1" 这类


def run_seedvr(
    *,
    variant: str,
    video_path: str,
    recovered_path: str,
    ckpt_path: str,
    device: Optional[str],
    method_kwargs: Dict[str, Any],
) -> str:
    """统一的 SeedVR 调用入口；四个 method 适配器都委托到这里。"""
    if not os.path.isdir(_SEEDVR_ROOT):
        raise FileNotFoundError(f"未找到 SeedVR 源码: {_SEEDVR_ROOT}")

    dit_ckpt = os.path.abspath(ckpt_path)
    vae_ckpt = method_kwargs.get("vae_ckpt") or os.path.join(
        os.path.dirname(dit_ckpt), "ema_vae.pth"
    )
    vae_ckpt = os.path.abspath(vae_ckpt)
    if not os.path.isfile(vae_ckpt):
        raise FileNotFoundError(
            f"未找到 VAE ckpt: {vae_ckpt}；请把 ema_vae.pth 放在 DiT ckpt 同目录，"
            f"或通过 method_kwargs['vae_ckpt'] 显式指定。"
        )

    # SeedVR 的 generation_loop 是按 basename 写出的；为了支持任意 recovered_path，
    # 我们把输入复制到一个临时目录，再把输出 rename 到 recovered_path。
    with tempfile.TemporaryDirectory(prefix="seedvr_io_", dir=os.path.dirname(os.path.abspath(recovered_path)) or None) as tmp:
        in_video_abs = os.path.abspath(video_path)
        out_video_abs = os.path.abspath(recovered_path)

        torchrun = method_kwargs.get("torchrun") or _DEFAULT_TORCHRUN
        if not os.path.isfile(torchrun):
            torchrun = shutil.which("torchrun") or torchrun
        master_port = str(method_kwargs.get("master_port", 29501))
        sp_size = int(method_kwargs.get("sp_size", 1))
        nproc = int(method_kwargs.get("nproc_per_node", sp_size))

        cmd = [
            torchrun,
            f"--nproc-per-node={nproc}",
            f"--master_port={master_port}",
            _RUNNER,
            "--variant", variant,
            "--seedvr_root", _SEEDVR_ROOT,
            "--in_video", in_video_abs,
            "--out_video", out_video_abs,
            "--dit_ckpt", dit_ckpt,
            "--vae_ckpt", vae_ckpt,
            "--sp_size", str(sp_size),
            "--res_h", str(method_kwargs.get("res_h", 720)),
            "--res_w", str(method_kwargs.get("res_w", 1280)),
            "--seed", str(method_kwargs.get("seed", 666)),
            "--cfg_rescale", str(method_kwargs.get("cfg_rescale", 0.0)),
        ]
        for opt_key in ("cfg_scale", "sample_steps", "cond_noise_scale", "out_fps"):
            if method_kwargs.get(opt_key) is not None:
                cmd += [f"--{opt_key}", str(method_kwargs[opt_key])]

        env = os.environ.copy()
        # PYTHONPATH 让子进程能 import SeedVR 仓库内的 common/ data/ projects/
        env["PYTHONPATH"] = _SEEDVR_ROOT + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        cvd = _resolve_device(device)
        if cvd is not None:
            env["CUDA_VISIBLE_DEVICES"] = cvd

        # tmp 仅用于隔离失败时的中间产物；主出参直接写到 recovered_path
        _ = tmp
        proc = subprocess.run(cmd, env=env, cwd=_SEEDVR_ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"SeedVR 子进程退出码非零: {proc.returncode}")

    if not os.path.isfile(recovered_path):
        raise RuntimeError(f"SeedVR 推断结束但未生成输出: {recovered_path}")
    return recovered_path
