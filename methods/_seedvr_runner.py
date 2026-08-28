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
    # 新增 variant：复用 SeedVR2-3B 权重与 config，推理/训练加 mask 控制 + LoRA
    "seedvr2_3b_ctrl": {
        "config": "./configs_3b/main.yaml",
        "default_dit": "./ckpts/seedvr2_ema_3b.pth",
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


def _build_runner(
    variant: str,
    dit_ckpt: str,
    vae_ckpt: str,
    sp_size: int,
    override_num_layers: int = None,
):
    """构造 VideoDiffusionInfer。

    override_num_layers: 仅用于「学生模型推理」场景 —— 若传值，则在 load_config 后就地把
    config.dit.model 派生为学生结构（当前只支持 20 层，layer_map 固定），再用学生 ckpt 加载。
    None（默认）时行为与之前完全一致。
    """
    from omegaconf import OmegaConf

    from common.config import load_config
    from common.distributed import init_torch
    from common.distributed.advanced import init_sequence_parallel
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer

    cfg = VARIANTS[variant]
    config = load_config(cfg["config"])
    OmegaConf.set_readonly(config, False)

    if override_num_layers is not None:
        # 让 methods._seedvr_distill_utils 可 import
        _uav_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _uav_root not in sys.path:
            sys.path.insert(0, _uav_root)
        from methods._seedvr_distill_utils import (
            derive_student_config_inplace,
            TEACHER_LAYER_MAP,
            STUDENT_NUM_LAYERS,
        )
        assert override_num_layers == STUDENT_NUM_LAYERS, (
            f"当前学生层裁剪映射固定为 {STUDENT_NUM_LAYERS} 层，收到 --student_num_layers={override_num_layers}"
        )
        derive_student_config_inplace(config, TEACHER_LAYER_MAP)
        print(f"[distill] student inference: config.dit.model.num_layers → {STUDENT_NUM_LAYERS}")

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
    return [
        rearrange(v[:, None] if v.ndim == 3 else v, "c t h w -> t c h w") for v in videos
    ]


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


def _generation_step_ctrl(runner, text_embeds_dict, cond_latents, cond_noise_scale: float, mask_latent):
    """带 mask 控制的 generation step：在 get_condition 后覆盖第 17 通道。"""
    from einops import rearrange

    from common.distributed import get_device
    from common.distributed.ops import sync_data

    from methods._seedvr_ctrl_utils import override_cond_channel_17

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
    # 关键：把原本 sr 任务恒为 1.0 的第 17 通道替换为下采后的 mask_latent
    conditions = [override_cond_channel_17(c, mask_latent) for c in conditions]

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
        videos = runner.inference(
            noises=noises, conditions=conditions, dit_offload=True, **text_embeds_dict
        )
    return [
        rearrange(v[:, None] if v.ndim == 3 else v, "c t h w -> t c h w") for v in videos
    ]


def _run_one_video_ctrl(
    runner,
    *,
    in_video: str,
    out_video: str,
    mask_path,
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
    """seedvr2_3b_ctrl 推理分支：与 _run_one_video 完全对齐，额外注入 mask。"""
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

    from methods._seedvr_ctrl_utils import (
        load_mask_as_TCHW,
        mask_temporal_downsample_causal,
    )

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

    T_pixel, _, H_pixel, W_pixel = video.shape

    # video: (T,C,H,W) → NaResize + DivisibleCrop + Normalize + Rearrange → (C, T, H_align, W_align)
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

    # mask: 走完全相同的空间变换（去掉 Normalize，mask 是 [0,1]），确保与 video 严格对齐
    mask_tchw = load_mask_as_TCHW(
        mask_path, num_frames=T_pixel, height=H_pixel, width=W_pixel
    ).to(get_device())
    mask_transform = Compose([
        NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Rearrange("t c h w -> c t h w"),
    ])
    mask_ctchw = mask_transform(mask_tchw)                     # (1, T_pixel, H_align, W_align)
    mask_ctchw = _cut_videos(mask_ctchw, sp_size)              # (1, T_padded, H_align, W_align)
    mask_tchw_aligned = mask_ctchw.permute(1, 0, 2, 3).contiguous()  # (T_padded, 1, H, W)

    runner.dit.to("cpu")
    runner.vae.to(get_device())
    cond_latents = runner.vae_encode([cond_latent])
    runner.vae.to("cpu")
    runner.dit.to(get_device())

    # VAE encode 完成后才知道 latent 时间维；对 mask 做同构（causal 4x 时间 + 8x 空间）下采
    # cond_latents[0] 的 shape 是 (T, H, W, C)：SeedVR VAE 输出经过 "b c ... -> b ... c" 重排 + squeeze(0)
    T_latent = cond_latents[0].shape[0]
    mask_latent = mask_temporal_downsample_causal(
        mask_tchw_aligned, T_latent=T_latent, spatial_stride=8
    ).to(get_device())                                          # (T_latent, H/8, W/8, 1)

    samples = _generation_step_ctrl(
        runner, text_embeds, cond_latents, cond_noise_scale, mask_latent
    )
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
    # seedvr2_3b_ctrl 专用参数（其它 variant 不会传，全部 default=None/False）
    parser.add_argument("--mask_path", default=None)
    parser.add_argument("--lora_ckpt", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--train_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--train_mode", action="store_true")
    # 蒸馏专用参数（仅 --distill_mode 时启用；未启用时全部无副作用）
    parser.add_argument("--distill_mode", action="store_true",
                        help="启用 SeedVR2-3B 层裁剪 + KD 蒸馏训练分支")
    parser.add_argument("--student_num_layers", type=int, default=None,
                        help="推理时用学生模型（当前固定 20 层）；训练时也应该传相同的值")
    parser.add_argument("--student_save_path", default=None,
                        help="蒸馏产出的学生 ckpt 文件名（相对 save_dir 或绝对路径）")
    parser.add_argument("--distill_lambda_out", type=float, default=1.0)
    parser.add_argument("--distill_lambda_feat", type=float, default=0.5)
    parser.add_argument("--distill_lambda_diff", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
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

    # 推理路径若传了 --student_num_layers（且不是蒸馏训练），把 config 派生为学生结构；
    # 蒸馏训练路径要构造教师，override_num_layers 必须 None
    inference_student = (
        args.student_num_layers is not None and not args.distill_mode
    )
    runner = _build_runner(
        args.variant,
        args.dit_ckpt,
        args.vae_ckpt,
        args.sp_size,
        override_num_layers=args.student_num_layers if inference_student else None,
    )
    dev = torch.cuda.current_device()
    after_load_alloc = torch.cuda.memory_allocated(dev) / 1024**3
    after_load_reserved = torch.cuda.memory_reserved(dev) / 1024**3
    print(
        f"[GPU mem] after model load: allocated={after_load_alloc:.2f} GiB, "
        f"reserved={after_load_reserved:.2f} GiB"
    )
    torch.cuda.reset_peak_memory_stats(dev)

    # ---- 分支 A：蒸馏训练（优先级最高，只支持 seedvr2_3b variant） ----
    if args.distill_mode:
        assert args.variant == "seedvr2_3b", (
            f"蒸馏训练仅支持 --variant seedvr2_3b，收到 {args.variant}"
        )
        _uav_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _uav_root not in sys.path:
            sys.path.insert(0, _uav_root)
        from methods._seedvr_distill import run_distill
        run_distill(runner, args, cond_noise_scale=cond_noise_scale)
    # ---- 分支 B：ctrl variant ----
    elif args.variant == "seedvr2_3b_ctrl":
        # 让 methods/* 可 import：主进程 chdir 到 SeedVR 根后，UAV_video_repair 项目根不在 sys.path
        _uav_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        # __file__ 是 SeedVR 根目录内的相对路径？不，_seedvr_runner.py 的绝对路径是 UAV_video_repair/methods/_seedvr_runner.py
        # 从这里往上两级 = UAV_video_repair
        _uav_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _uav_root not in sys.path:
            sys.path.insert(0, _uav_root)

        from methods._seedvr_ctrl_utils import (
            attach_lora, unfreeze_vid_in_proj, load_trainable_state, count_trainable_params,
        )

        runner.dit = attach_lora(runner.dit, r=args.lora_r, alpha=args.lora_alpha)
        n_unfrozen = unfreeze_vid_in_proj(runner.dit)
        stats = count_trainable_params(runner.dit)
        print(
            f"[LoRA] trainable={stats['trainable']:,} / total={stats['total']:,} "
            f"({stats['ratio']*100:.3f}%); vid_in.proj unfrozen params (weight+bias): {n_unfrozen}"
        )

        if args.lora_ckpt:
            unexpected = load_trainable_state(runner.dit, args.lora_ckpt)
            if unexpected:
                print(f"[LoRA] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            else:
                print(f"[LoRA] loaded {args.lora_ckpt}")

        if args.train_mode:
            from methods._seedvr_train import run_train
            run_train(runner, args, cond_noise_scale=cond_noise_scale)
        else:
            _run_one_video_ctrl(
                runner,
                in_video=args.in_video,
                out_video=args.out_video,
                mask_path=args.mask_path,
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
    else:
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

    dev = torch.cuda.current_device()
    peak_alloc = torch.cuda.max_memory_allocated(dev) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(dev) / 1024**3
    free, total = torch.cuda.mem_get_info(dev)
    print(
        f"[GPU mem] inference peak (since reset): allocated={peak_alloc:.2f} GiB, "
        f"reserved={peak_reserved:.2f} GiB; current_free={free / 1024**3:.2f}/{total / 1024**3:.2f} GiB"
    )


if __name__ == "__main__":
    main()

