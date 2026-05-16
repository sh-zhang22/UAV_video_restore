"""SeedVR / SeedVR2 子进程入口：被 torchrun 启动后执行真实推断。

由 methods/_seedvr_common.py 通过 subprocess + torchrun 调用，
不应被普通 import 调用——它会 os.chdir 到 SeedVR 根目录。
"""
from __future__ import annotations

import argparse
import datetime
import gc
import os
import sys

import torch


VARIANTS = {
    "seedvr_3b": {
        "config": "./configs_3b/main.yaml",
        "default_dit": "./ckpts/seedvr_ema_3b.pth",
        "cond_noise_scale": 0.1,
        "cfg_scale": 6.5,
        "sample_steps": 50,
    },
    "seedvr2_3b": {
        "config": "./configs_3b/main.yaml",
        "default_dit": "./ckpts/seedvr2_ema_3b.pth",
        "cond_noise_scale": 0.0,
        "cfg_scale": 1.0,
        "sample_steps": 1,
    },
    "seedvr_7b": {
        "config": "./configs_7b/main.yaml",
        "default_dit": "./ckpts/seedvr_ema_7b.pth",
        "cond_noise_scale": 0.1,
        "cfg_scale": 6.5,
        "sample_steps": 50,
    },
    "seedvr2_7b": {
        "config": "./configs_7b/main.yaml",
        "default_dit": "./ckpts/seedvr2_ema_7b.pth",
        "cond_noise_scale": 0.0,
        "cfg_scale": 1.0,
        "sample_steps": 1,
    },
}

def _patch_vae_ckpt(config, vae_ckpt: str) -> None:
    """SeedVR config 里 VAE ckpt 路径硬编码为 ./ckpts/ema_vae.pth，按需覆盖。"""
    from omegaconf import OmegaConf

    OmegaConf.set_readonly(config, False)
    config.vae.checkpoint = vae_ckpt


def _build_runner(variant: str, dit_ckpt: str, vae_ckpt: str, sp_size: int):
    from omegaconf import OmegaConf

    from common.config import load_config
    from common.distributed import init_torch
    from common.distributed.advanced import init_sequence_parallel
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    cfg = VARIANTS[variant]
    config = load_config(cfg["config"])
    runner = VideoDiffusionInfer(config)
    OmegaConf.set_readonly(runner.config, False)
    _patch_vae_ckpt(runner.config, vae_ckpt)

    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    if sp_size > 1:
        init_sequence_parallel(sp_size)
    runner.configure_dit_model(device="cuda", checkpoint=dit_ckpt)
    runner.configure_vae_model()
    if hasattr(runner.vae, "set_memory_limit"):
        runner.vae.set_memory_limit(**runner.config.vae.memory_limit)
    return runner


def _generation_step(runner, text_embeds_dict, cond_latents, cond_noise_scale: float):
    from einops import rearrange

    from common.distributed import get_device
    from common.distributed.ops import sync_data

    def _to_dev(x):
        return [i.to(get_device()) for i in x]

    noises = [torch.randn_like(l) for l in cond_latents]
    aug_noises = [torch.randn_like(l) for l in cond_latents]
    noises, aug_noises, cond_latents = sync_data((noises, aug_noises, cond_latents), 0)
    noises, aug_noises, cond_latents = map(_to_dev, (noises, aug_noises, cond_latents))
    noises, aug_noises, cond_latents = list(noises), list(aug_noises), list(cond_latents)

    def _add_noise(x, aug):
        t = torch.tensor([1000.0], device=get_device()) * cond_noise_scale
        shape = torch.tensor(x.shape[1:], device=get_device())[None]
        t = runner.timestep_transform(t, shape)
        return runner.schedule.forward(x, aug, t)

    conditions = [
        runner.get_condition(n, task="sr", latent_blur=_add_noise(lb, an))
        for n, an, lb in zip(noises, aug_noises, cond_latents)
    ]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        videos = runner.inference(
            noises=noises, conditions=conditions, dit_offload=True, **text_embeds_dict
        )
def _cut_videos(video, sp_size):
    t = video.size(1)
    if t == 1:
        return video
    if t <= 4 * sp_size:
        pad = torch.cat([video[:, -1:].clone()] * (4 * sp_size - t + 1), dim=1)
        return torch.cat([video, pad], dim=1)
    if (t - 1) % (4 * sp_size) == 0:
        return video
    pad = torch.cat([video[:, -1:].clone()] * (4 * sp_size - ((t - 1) % (4 * sp_size))), dim=1)
    return torch.cat([video, pad], dim=1)


def _run_one_video(
    runner,
    *,
    in_video: str,
    out_video: str,
    cfg_scale: float,
    cfg_rescale: float,
    sample_steps: int,
    seed: int,
    res_h: int,
    res_w: int,
    sp_size: int,
    out_fps,
    cond_noise_scale: float,
):
    import mediapy
    from einops import rearrange
    from torchvision.io.video import read_video
    from torchvision.transforms import Compose, Lambda, Normalize

    from common.distributed import get_device
    from common.distributed.advanced import get_sequence_parallel_rank
    from common.seed import set_seed
    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange

    runner.config.diffusion.cfg.scale = cfg_scale
    runner.config.diffusion.cfg.rescale = cfg_rescale
    runner.config.diffusion.timesteps.sampling.steps = sample_steps
    runner.configure_diffusion()
    set_seed(seed, same_across_ranks=True)

    use_colorfix = os.path.exists("./projects/video_diffusion_sr/color_fix.py")
    if use_colorfix:
        from projects.video_diffusion_sr.color_fix import wavelet_reconstruction

    text_pos = torch.load("pos_emb.pt")
    text_neg = torch.load("neg_emb.pt")
    text_embeds = {"texts_pos": [text_pos.to(get_device())], "texts_neg": [text_neg.to(get_device())]}

    video, _, info = read_video(in_video, output_format="TCHW")
    video = video.float() / 255.0
    save_fps = info.get("video_fps", 24.0) if out_fps is None else out_fps

    transform = Compose([
        NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    cond_latent = transform(video.to(get_device()))
    ori_length = cond_latent.size(1)
    input_video = cond_latent
    cond_latent = _cut_videos(cond_latent, sp_size)

    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_latents = runner.vae_encode([cond_latent])
    runner.vae.to("cpu")
    runner.dit.to(get_device())

    samples = _generation_step(runner, text_embeds, cond_latents, cond_noise_scale)
    runner.dit.to("cpu")

    if get_sequence_parallel_rank() != 0:
        return

    sample = samples[0]
    if ori_length < sample.shape[0]:
        sample = sample[:ori_length]
    inp = (
        rearrange(input_video[:, None] if input_video.ndim == 3 else input_video, "c t h w -> t c h w")
    )
    if use_colorfix:
        sample = wavelet_reconstruction(sample.to("cpu"), inp[: sample.size(0)].to("cpu"))
    else:
        sample = sample.to("cpu")
    sample = rearrange(sample[:, None] if sample.ndim == 3 else sample, "t c h w -> t h w c")
    sample = sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().to(torch.uint8).numpy()

    os.makedirs(os.path.dirname(os.path.abspath(out_video)) or ".", exist_ok=True)
    mediapy.write_video(out_video, sample, fps=save_fps)
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--seedvr_root", required=True)
    parser.add_argument("--in_video", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--dit_ckpt", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--res_h", type=int, default=720)
    parser.add_argument("--res_w", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=None)
    parser.add_argument("--cfg_rescale", type=float, default=0.0)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--cond_noise_scale", type=float, default=None)
    parser.add_argument("--out_fps", type=float, default=None)
    args = parser.parse_args()

    os.chdir(args.seedvr_root)
    if args.seedvr_root not in sys.path:
        sys.path.insert(0, args.seedvr_root)

    cfg = VARIANTS[args.variant]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else cfg["cfg_scale"]
    sample_steps = args.sample_steps if args.sample_steps is not None else cfg["sample_steps"]
    cond_noise_scale = (
        args.cond_noise_scale if args.cond_noise_scale is not None else cfg["cond_noise_scale"]
    )

    runner = _build_runner(args.variant, args.dit_ckpt, args.vae_ckpt, args.sp_size)
    _run_one_video(
        runner,
        in_video=args.in_video,
        out_video=args.out_video,
        cfg_scale=cfg_scale,
        cfg_rescale=args.cfg_rescale,
        sample_steps=sample_steps,
        seed=args.seed,
        res_h=args.res_h,
        res_w=args.res_w,
        sp_size=args.sp_size,
        out_fps=args.out_fps,
        cond_noise_scale=cond_noise_scale,
    )


if __name__ == "__main__":
    main()

