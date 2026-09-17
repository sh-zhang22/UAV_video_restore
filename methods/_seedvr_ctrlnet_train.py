"""seedvr2_3b_ctrlnet 训练分支：极简 loop，只为冒烟通路。

与 _seedvr_train.py（ctrl 分支）的差异：
- 不做 override_cond_channel_17（保留第 17 通道原 sr 语义）
- 在 forward 前 runner.dit.set_mask(mask_latent)，让 ControlledDiT 内部读取
- 只训练侧枝（mask_stem + side_blocks + zero_convs），主干 requires_grad=False（由 wrap_with_controlnet 保证）

真实数据到位后再展开：换掉假 latent_blur（=x0）为 low-res encode，加 dataloader、
logitnormal timestep 采样、v_lerp 精确 loss 等。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def run_train(runner, args, cond_noise_scale: float):
    """冒烟训练 loop。

    Args:
        runner: VideoDiffusionInfer 实例（runner.dit 已由 wrap_with_controlnet 包装成 ControlledDiT）
        args:   _seedvr_runner.py main() 里的 argparse Namespace
        cond_noise_scale: 冒烟阶段不用（保留兼容）
    """
    from einops import rearrange
    from torchvision.io.video import read_video
    from torchvision.transforms import Compose, Lambda, Normalize

    from common.distributed import get_device
    from common.seed import set_seed
    from data.image.transforms.divisible_crop import DivisibleCrop
    from data.image.transforms.na_resize import NaResize
    from data.video.transforms.rearrange import Rearrange
    from models.dit_v2 import na

    from methods._seedvr_runner import _cut_videos
    from methods._seedvr_ctrl_utils import (
        load_mask_as_TCHW,
        mask_temporal_downsample_causal,
    )
    from methods._seedvr_ctrlnet_utils import save_ctrlnet_state, count_ctrlnet_params

    device = get_device()
    save_dir = args.save_dir or "./runs_ctrlnet"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(args.seed, same_across_ranks=True)

    # 训练不启用 cfg / 多步采样（我们只跑单步 forward）
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    # ---- 加载 video + mask（与推理管线严格对齐） ----
    text_pos = torch.load("pos_emb.pt").to(device)

    video, _, _ = read_video(args.in_video, output_format="TCHW")
    video = video.float() / 255.0
    T_pixel, _, H_pixel, W_pixel = video.shape

    transform = Compose([
        NaResize(resolution=(args.res_h * args.res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    video_ct = transform(video.to(device))
    video_ct = _cut_videos(video_ct, args.sp_size)

    mask_tchw = load_mask_as_TCHW(
        args.mask_path, num_frames=T_pixel, height=H_pixel, width=W_pixel
    ).to(device)
    mask_transform = Compose([
        NaResize(resolution=(args.res_h * args.res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Rearrange("t c h w -> c t h w"),
    ])
    mask_ct = mask_transform(mask_tchw)
    mask_ct = _cut_videos(mask_ct, args.sp_size)
    mask_tchw_aligned = mask_ct.permute(1, 0, 2, 3).contiguous()

    # ---- VAE encode（no_grad） ----
    runner.dit.to("cpu")
    runner.vae.to(device)
    with torch.no_grad():
        x0_list = runner.vae_encode([video_ct])
    runner.vae.to("cpu")
    runner.dit.to(device)

    x0 = x0_list[0]                                        # (T', H', W', 16)
    T_latent = x0.shape[0]
    mask_latent = mask_temporal_downsample_causal(
        mask_tchw_aligned, T_latent=T_latent, spatial_stride=8
    ).to(device)                                            # (T', H', W', 1)

    # ---- optimizer（只训练侧枝，count_ctrlnet_params 打印 sanity） ----
    stats = count_ctrlnet_params(runner.dit)
    print(
        f"[ctrlnet-train] base={stats['base']:,} side={stats['side']:,} "
        f"trainable={stats['trainable']:,} ratio(side/base)={stats['ratio_side_over_base']*100:.2f}%"
    )
    trainable = [p for p in runner.dit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # 主干开 gradient checkpointing 省显存
    runner.dit.set_gradient_checkpointing(True)

    runner.dit.train()
    # 注入 mask（整个训练用同一个 mask，冒烟不追求多样性）
    runner.dit.set_mask(mask_latent)

    for step in range(args.train_steps):
        noise = torch.randn_like(x0)
        t_raw = torch.rand(1, device=device) * 600 + 200
        shape_thw = torch.tensor(x0.shape[:-1], device=device)[None]
        t = runner.timestep_transform(t_raw, shape_thw)

        x_t = runner.schedule.forward(x0, noise, t)

        # condition：get_condition 原样（第 17 通道保留原 sr 全 1 语义）
        cond = runner.get_condition(noise, task="sr", latent_blur=x0)   # (T', H', W', 17)

        vid_flat, vid_shape = na.flatten([x_t])
        cond_flat, _ = na.flatten([cond])
        dit_input = torch.cat([vid_flat, cond_flat], dim=-1)           # (l, 33)
        txt_flat, txt_shape = na.flatten([text_pos])

        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            pred = runner.dit(
                vid=dit_input,
                txt=txt_flat,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=t.repeat(1),
            ).vid_sample

        target_flat, _ = na.flatten([noise - x0])
        loss = F.mse_loss(pred.float(), target_flat.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"[ctrlnet-train] step {step+1}/{args.train_steps} "
            f"loss={loss.item():.4f} grad_norm={grad_norm.item():.4f} "
            f"peak_mem={peak:.2f} GiB"
        )
        torch.cuda.reset_peak_memory_stats(device)

    runner.dit.set_mask(None)

    ckpt_path = save_ctrlnet_state(runner.dit, save_dir)
    print(f"[ctrlnet-train] saved side weights: {ckpt_path}")
    return ckpt_path
