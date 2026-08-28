"""seedvr2_3b_ctrl 训练分支：极简 loop，只为冒烟通路。

冒烟目的：
- forward → backward → optimizer.step → save_trainable_state 全链路走通
- LoRA + vid_in.proj 的可训参数确实收到梯度且非 NaN
- 不追求收敛，也不追求 loss 目标严格（用简化的 v = noise - x0）

真实数据到位后再展开：换掉假 latent_blur（=x0）为 low-res encode，加 dataloader、
timestep 采样器（logitnormal）、真实的 flow-matching / v-lerp 目标。
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def run_train(runner, args, cond_noise_scale: float):
    """冒烟训练 loop。

    Args:
        runner: VideoDiffusionInfer 实例（已挂 LoRA + 解冻 vid_in.proj）
        args:   _seedvr_runner.py main() 里的 argparse Namespace
        cond_noise_scale: 训练目前不用（保留兼容）
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

    # 延迟 import 避免循环
    from methods._seedvr_runner import _cut_videos
    from methods._seedvr_ctrl_utils import (
        load_mask_as_TCHW,
        mask_temporal_downsample_causal,
        override_cond_channel_17,
        save_trainable_state,
        count_trainable_params,
    )

    device = get_device()
    save_dir = args.save_dir or "./runs_ctrl"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(args.seed, same_across_ranks=True)

    # 训练不启用 cfg，diffusion sampler 也不需要（我们只跑单步 forward）
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
        x0_list = runner.vae_encode([video_ct])  # [(T', H', W', 16)]
    runner.vae.to("cpu")
    runner.dit.to(device)

    x0 = x0_list[0]                                        # (T', H', W', 16)
    T_latent = x0.shape[0]
    mask_latent = mask_temporal_downsample_causal(
        mask_tchw_aligned, T_latent=T_latent, spatial_stride=8
    ).to(device)                                            # (T', H', W', 1)

    # ---- optimizer ----
    stats = count_trainable_params(runner.dit)
    print(
        f"[train] trainable={stats['trainable']:,} / total={stats['total']:,} "
        f"({stats['ratio']*100:.3f}%)"
    )
    trainable = [p for p in runner.dit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # ---- 训练前开启 gradient checkpointing（peft 包装下用 modules 递归找）----
    gc_enabled = 0
    for m in runner.dit.modules():
        if hasattr(m, "set_gradient_checkpointing"):
            m.set_gradient_checkpointing(True)
            gc_enabled += 1
    print(f"[train] gradient_checkpointing enabled on {gc_enabled} module(s)")

    # ---- 训练 loop ----
    runner.dit.train()
    for step in range(args.train_steps):
        # 1. 采 noise、timestep（冒烟阶段：随机在 [200, 800]）
        noise = torch.randn_like(x0)
        t_raw = torch.rand(1, device=device) * 600 + 200                          # (1,)
        shape_thw = torch.tensor(x0.shape[:-1], device=device)[None]               # (1, 3) = (T, H, W)
        t = runner.timestep_transform(t_raw, shape_thw)

        # 2. 加噪
        x_t = runner.schedule.forward(x0, noise, t)

        # 3. condition：用 x0 作 latent_blur（冒烟简化；真实训练要用 low-res encode）
        cond = runner.get_condition(noise, task="sr", latent_blur=x0)             # (T', H', W', 17)
        cond = override_cond_channel_17(cond, mask_latent)

        # 4. flatten + 拼 DiT 输入
        vid_flat, vid_shape = na.flatten([x_t])                                    # (l, 16), (1, 3)
        cond_flat, _ = na.flatten([cond])                                          # (l, 17)
        dit_input = torch.cat([vid_flat, cond_flat], dim=-1)                       # (l, 33)
        txt_flat, txt_shape = na.flatten([text_pos])

        # 5. forward（bfloat16 autocast，与推理一致）
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            pred = runner.dit(
                vid=dit_input,
                txt=txt_flat,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=t.repeat(1),
            ).vid_sample                                                           # (l, 16)

        # 6. loss：简化 v = noise - x0
        target_flat, _ = na.flatten([noise - x0])
        loss = F.mse_loss(pred.float(), target_flat.float())

        # 7. backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        # 显存 & 打印
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"[train] step {step+1}/{args.train_steps} "
            f"loss={loss.item():.4f} grad_norm={grad_norm.item():.4f} "
            f"peak_mem={peak:.2f} GiB"
        )
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 保存 ----
    ckpt_path = save_trainable_state(runner.dit, save_dir)
    print(f"[train] saved trainable weights: {ckpt_path}")
    return ckpt_path
