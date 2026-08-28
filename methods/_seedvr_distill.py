"""SeedVR2-3B 层裁剪 + KD 蒸馏训练分支（冒烟通路）。

策略：
- 教师 = seedvr2_3b（32 层 NaDiT，已加载官方权重），eval() + no_grad()
- 学生 = 派生自教师的 20 层 NaDiT，权重按 TEACHER_LAYER_MAP 从教师复制作为初始化
- 每步同一 x_t + condition，教师和学生分别 forward
- Loss = λ_out · MSE(z_S, z_T) + λ_feat · Σ (1 - cos(h_S[j], h_T[map[j]])) + λ_diff · MSE(z_S, v_target)

冒烟目的：走通 forward → backward → save → 反向加载推理 全链路，不追求收敛。
假数据：test.mp4 + 随机 noise + 随机 timestep（[200, 800]），同一 batch 重复 N 步。
"""
from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F


def _build_student_dit(teacher_runner, layer_map):
    """基于教师 runner 派生学生 dit（复用教师 config，改 num_layers），返回学生 dit（cuda）。

    - 复制教师 config → 就地改 num_layers/window_method/block_type/window
    - create_object 空 dit → 按 layer_map 从教师权重派生 state_dict → 装入
    - 学生 dit 会开 gradient_checkpointing
    """
    from omegaconf import OmegaConf
    from common.config import create_object

    from methods._seedvr_distill_utils import (
        derive_student_config_inplace,
        build_student_state_dict_from_teacher,
        count_params,
    )

    student_config = copy.deepcopy(teacher_runner.config)
    derive_student_config_inplace(student_config, layer_map)

    device = next(teacher_runner.dit.parameters()).device
    # NaDiT 参数在 configure_dit_model 里是先 CPU/meta 构造再搬到 cuda。学生直接在 cpu 建再 to cuda。
    with torch.device("cpu"):
        student_dit = create_object(student_config.dit.model)

    # gradient checkpointing 由训练循环显式开启（configure_dit_model 里默认从 config 读，这里保持 False）
    if hasattr(student_dit, "set_gradient_checkpointing"):
        student_dit.set_gradient_checkpointing(False)

    # 从教师取 state_dict（当前教师在 cuda，to cpu 拿轻量副本会占内存 → 直接引用即可，
    # 因为 build 里只做 key 映射不 clone tensor）
    teacher_sd = teacher_runner.dit.state_dict()
    student_sd = build_student_state_dict_from_teacher(teacher_sd, layer_map)

    info = student_dit.load_state_dict(student_sd, strict=True)
    print(f"[distill] student load_state_dict: missing={list(info.missing_keys)[:3]}, "
          f"unexpected={list(info.unexpected_keys)[:3]}")

    student_dit = student_dit.to(device)

    # 关键：让教师所有 32 层 + 学生 20 层的 attn.rope 全部共享同一个实例。
    # 原因：rope 内部用 lru_cache 存 (1024,128,128,D) 的 axial_freqs，每个 rope 实例独立
    # 缓存 → 32+20 份大 tensor，合计 100+ GiB。RotaryEmbedding.freqs 是 register_buffer
    # 且各层初始化时数值一致（rope_dim/theta 同一 config），无 learnable，共享数值等价。
    shared_rope = teacher_runner.dit.blocks[0].attn.rope
    if shared_rope is not None:
        # sanity：验证教师各层 rope freqs 数值一致（无 learnable，理论上必然一致）
        first_freqs = shared_rope.rope.freqs
        max_diff = 0.0
        for i, b in enumerate(teacher_runner.dit.blocks):
            other = b.attn.rope.rope.freqs
            diff = float((first_freqs - other).abs().max())
            max_diff = max(max_diff, diff)
        if max_diff > 1e-6:
            print(f"[distill] WARN: teacher rope.freqs 层间差异 max={max_diff:.2e}，共享可能引入误差")
        else:
            print(f"[distill] teacher rope.freqs 层间一致（max diff={max_diff:.2e}）")

        n_shared = 0
        for b in teacher_runner.dit.blocks:
            b.attn.rope = shared_rope
        for b in student_dit.blocks:
            b.attn.rope = shared_rope
            n_shared += 1
        print(f"[distill] shared single rope instance across teacher(32) + student({n_shared}) layers")

    stats_t = count_params(teacher_runner.dit)
    stats_s = count_params(student_dit)
    print(
        f"[distill] teacher params={stats_t['total']:,}; "
        f"student params={stats_s['total']:,} "
        f"(ratio={stats_s['total']/stats_t['total']*100:.2f}%)"
    )
    return student_dit


def _cosine_align_loss(h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """Feature 对齐：1 - cos_sim，按 hidden-dim 归一化。

    两侧同 shape (l, D)。学生侧带梯度，教师侧 no_grad（无 detach 也 safe）。
    """
    # cast to fp32 做 loss（教师是 bf16 输出）
    h_s = h_s.float()
    h_t = h_t.float().detach()
    # 归一化后点积 = cos_sim
    h_s_n = F.normalize(h_s, dim=-1, eps=1e-6)
    h_t_n = F.normalize(h_t, dim=-1, eps=1e-6)
    cos = (h_s_n * h_t_n).sum(dim=-1)   # (l,)
    return (1.0 - cos).mean()


def run_distill(teacher_runner, args, cond_noise_scale: float):
    """蒸馏训练 loop。

    Args:
        teacher_runner: VideoDiffusionInfer（seedvr2_3b 变体），dit 已加载 32 层教师权重
        args: _seedvr_runner.py 的 argparse Namespace
        cond_noise_scale: 未直接使用（sr 任务默认 0.0），保留兼容
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
    from methods._seedvr_distill_utils import (
        TEACHER_LAYER_MAP,
        BlockOutputCapture,
        count_params,
    )

    device = get_device()
    save_dir = args.save_dir or "./runs_distill"
    os.makedirs(save_dir, exist_ok=True)
    set_seed(args.seed, same_across_ranks=True)

    # ---- 学生 dit 构造 & 权重初始化 ----
    student_dit = _build_student_dit(teacher_runner, TEACHER_LAYER_MAP)

    # ---- 教师配置 & schedule ----
    teacher_runner.config.diffusion.cfg.scale = 1.0
    teacher_runner.config.diffusion.cfg.rescale = 0.0
    teacher_runner.config.diffusion.timesteps.sampling.steps = 1
    teacher_runner.configure_diffusion()

    text_pos = torch.load("pos_emb.pt").to(device)

    # ---- 读 test.mp4 + 前处理（与推理管线一致） ----
    video, _, _ = read_video(args.in_video, output_format="TCHW")
    video = video.float() / 255.0

    transform = Compose([
        NaResize(resolution=(args.res_h * args.res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])
    video_ct = transform(video.to(device))
    video_ct = _cut_videos(video_ct, args.sp_size)

    # ---- VAE encode → x0 ----
    teacher_runner.dit.to("cpu")
    teacher_runner.vae.to(device)
    with torch.no_grad():
        x0_list = teacher_runner.vae_encode([video_ct])
    teacher_runner.vae.to("cpu")
    teacher_runner.dit.to(device)
    x0 = x0_list[0]                                        # (T', H', W', 16)

    # ---- optimizer（只训学生）----
    teacher_runner.dit.eval()
    for p in teacher_runner.dit.parameters():
        p.requires_grad_(False)

    # 教师转 bf16 省 ~6G 显存（12.6G → 6.3G）。教师只做 forward 推理，autocast(bf16) 下参数
    # bf16 无副作用。**特殊处理共享 rope**：freqs 是位置编码累积计算的关键 buffer（外积
    # position*freqs），需要 fp32 精度；bfloat16() 后立即把 shared_rope 的所有 buffer 恢复到
    # fp32。apply_rotary_emb 内 `q.float()` 已强制 rope 计算 fp32，只要 freqs 是 fp32 就稳定。
    teacher_runner.dit.bfloat16()
    _shared_rope = teacher_runner.dit.blocks[0].attn.rope  # 与 _build_student_dit 里共享的同实例
    if _shared_rope is not None:
        _shared_rope.float()   # rope 及其 sub-module 全部回 fp32
    torch.cuda.empty_cache()

    student_dit.train()
    # 学生的 gradient checkpointing 只在训练时开
    n_gc = 0
    for m in student_dit.modules():
        if hasattr(m, "set_gradient_checkpointing"):
            m.set_gradient_checkpointing(True)
            n_gc += 1
    print(f"[distill] student gradient_checkpointing enabled on {n_gc} module(s)")

    stats = count_params(student_dit)
    print(f"[distill] student trainable={stats['trainable']:,} / total={stats['total']:,}")
    trainable = [p for p in student_dit.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    # warmup
    warmup_steps = max(0, int(args.warmup_steps))
    def _lr_scale(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    lambda_out = float(args.distill_lambda_out)
    lambda_feat = float(args.distill_lambda_feat)
    lambda_diff = float(args.distill_lambda_diff)
    print(f"[distill] loss weights: out={lambda_out}, feat={lambda_feat}, diff={lambda_diff}")

    # 中间层 hook 仅在 lambda_feat > 0 时启用。原因：教师 32 层 rope 各自 lru_cache 一份大
    # freqs（合计 60+GiB），加上学生 20 层反向图，直接 hook 全 32/20 层会 OOM。冒烟阶段
    # 先只做 output + diffusion loss；真实数据到位后要开 feature loss，可分块 backward 优化。
    teacher_capture = None
    student_capture = None
    if lambda_feat > 0:
        teacher_capture = BlockOutputCapture(teacher_runner.dit)
        student_capture = BlockOutputCapture(student_dit)
        print("[distill] feature-align hooks installed (lambda_feat > 0)")
    else:
        print("[distill] feature-align disabled (lambda_feat == 0); skip hooks")

    for step in range(args.train_steps):
        # 1. 采样 noise、timestep
        noise = torch.randn_like(x0)
        t_raw = torch.rand(1, device=device) * 600 + 200
        shape_thw = torch.tensor(x0.shape[:-1], device=device)[None]
        t = teacher_runner.timestep_transform(t_raw, shape_thw)

        # 2. 加噪
        x_t = teacher_runner.schedule.forward(x0, noise, t)

        # 3. condition（原 sr 任务：第 17 通道恒 1）
        cond = teacher_runner.get_condition(noise, task="sr", latent_blur=x0)  # (T', H', W', 17)

        # 4. flatten + 拼 DiT 输入
        vid_flat, vid_shape = na.flatten([x_t])                                 # (l, 16), (1, 3)
        cond_flat, _ = na.flatten([cond])                                       # (l, 17)
        dit_input = torch.cat([vid_flat, cond_flat], dim=-1)                    # (l, 33)
        txt_flat, txt_shape = na.flatten([text_pos])

        # 5. teacher forward（no_grad）
        if teacher_capture is not None:
            teacher_capture.clear()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            z_T = teacher_runner.dit(
                vid=dit_input,
                txt=txt_flat,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=t.repeat(1),
            ).vid_sample                                                        # (l, 16)

        # 6. student forward（train，带 grad）
        if student_capture is not None:
            student_capture.clear()
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            z_S = student_dit(
                vid=dit_input,
                txt=txt_flat,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                timestep=t.repeat(1),
            ).vid_sample                                                        # (l, 16)

        # 7. loss —— 三项
        # (a) output 蒸馏（学生 z 对教师 z）
        loss_out = F.mse_loss(z_S.float(), z_T.float().detach())

        # (b) feature 对齐（可选：仅当 lambda_feat > 0 且 hooks 装了）
        if teacher_capture is not None and student_capture is not None:
            h_S_all = student_capture.features
            h_T_all = teacher_capture.features
            assert len(h_S_all) == len(TEACHER_LAYER_MAP), (
                f"student captured {len(h_S_all)} vs expected {len(TEACHER_LAYER_MAP)}"
            )
            loss_feat_sum = 0.0
            n_feat = 0
            for j, t_idx in enumerate(TEACHER_LAYER_MAP):
                if h_S_all[j] is None or h_T_all[t_idx] is None:
                    continue
                loss_feat_sum = loss_feat_sum + _cosine_align_loss(h_S_all[j], h_T_all[t_idx])
                n_feat += 1
            loss_feat = loss_feat_sum / max(1, n_feat) if n_feat > 0 else torch.zeros((), device=device)
        else:
            loss_feat = torch.zeros((), device=device)

        # (c) 原扩散目标（保留 v = noise - x0 简化版）
        target_flat, _ = na.flatten([noise - x0])
        loss_diff = F.mse_loss(z_S.float(), target_flat.float())

        loss = lambda_out * loss_out + lambda_feat * loss_feat + lambda_diff * loss_diff

        # 8. backward
        cur_lr = args.lr * _lr_scale(step)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(
            f"[distill] step {step+1}/{args.train_steps} "
            f"lr={cur_lr:.2e} loss={loss.item():.4f} "
            f"(out={loss_out.item():.4f} feat={float(loss_feat):.4f} diff={loss_diff.item():.4f}) "
            f"grad_norm={grad_norm.item():.4f} peak_mem={peak:.2f} GiB"
        )
        torch.cuda.reset_peak_memory_stats(device)

    if teacher_capture is not None:
        teacher_capture.remove()
    if student_capture is not None:
        student_capture.remove()

    # ---- 保存学生 state_dict ----
    out_name = args.student_save_path or "student.pth"
    if not os.path.isabs(out_name):
        out_name = os.path.join(save_dir, out_name)
    os.makedirs(os.path.dirname(out_name) or ".", exist_ok=True)
    # 保存的是 CPU 副本，避免加载端设备耦合
    student_sd_cpu = {k: v.detach().cpu().clone() for k, v in student_dit.state_dict().items()}
    torch.save(student_sd_cpu, out_name)
    print(f"[distill] saved student weights: {out_name}")
    return out_name
