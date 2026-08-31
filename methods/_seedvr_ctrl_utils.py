"""seedvr2_3b_ctrl 专用工具：mask I/O、causal 时间下采、LoRA 挂载、可训权重存取。

设计原则：
- 纯函数，不 import SeedVR 源码顶层符号（延迟到函数内部），避免污染其它 method
- 只被 _seedvr_runner.py 在 variant == "seedvr2_3b_ctrl" 时调用
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F


# ------------------------- mask I/O ------------------------- #

def load_mask_as_TCHW(
    mask_path: str,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """把任意格式的 mask 读成 (T, 1, H, W) float32 in [0, 1]。

    支持：
    - .npy: shape 可为 (T,H,W) / (H,W) / (T,1,H,W)，自动广播/添加 channel
    - .png/.jpg/.jpeg: 单张，广播到 T 帧
    - .mp4/.avi: 逐帧读，取灰度
    - None / 空字符串: 返回全 1（等价 baseline sr 语义，用于 sanity check）
    """
    if not mask_path or mask_path.lower() in ("none", "null", ""):
        return torch.ones(num_frames, 1, height, width, dtype=torch.float32)

    # 子进程会 chdir 到 SeedVR 根 → 相对路径需要在调用方转成绝对；这里给一个明确的错误
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f"mask_path 找不到: {mask_path}\n"
            f"当前工作目录: {os.getcwd()}\n"
            f"提示：调用方需要传绝对路径（子进程会 chdir 到 SeedVR 根目录）"
        )

    ext = os.path.splitext(mask_path)[1].lower()

    if ext == ".npy":
        import numpy as np
        arr = np.load(mask_path)
        m = torch.from_numpy(arr).float()
        if m.max() > 1.5:                # 猜是 uint8 in [0,255]
            m = m / 255.0
        if m.ndim == 2:                  # (H, W)
            m = m[None, None].expand(num_frames, 1, -1, -1)
        elif m.ndim == 3:                # (T, H, W)
            m = m[:, None]
        elif m.ndim == 4:                # (T, 1, H, W) 或 (T, C, H, W)
            if m.shape[1] != 1:
                m = m.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"npy mask 维度不支持: {m.shape}")
    elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
        from PIL import Image
        img = Image.open(mask_path).convert("L")
        arr = torch.from_numpy(_pil_to_numpy(img)).float() / 255.0    # (H, W)
        m = arr[None, None].expand(num_frames, 1, -1, -1)
    elif ext in (".mp4", ".avi", ".mov"):
        from torchvision.io.video import read_video
        vid, _, _ = read_video(mask_path, output_format="TCHW")       # (T, C, H, W) uint8
        m = vid.float().mean(dim=1, keepdim=True) / 255.0             # 转灰度 → (T,1,H,W)
    else:
        raise ValueError(f"不支持的 mask 格式: {ext}")

    m = m.clamp(0.0, 1.0)

    # 时间广播/裁切到 num_frames
    if m.shape[0] == 1 and num_frames > 1:
        m = m.expand(num_frames, -1, -1, -1).contiguous()
    elif m.shape[0] < num_frames:
        # 循环补到 num_frames
        reps = (num_frames + m.shape[0] - 1) // m.shape[0]
        m = m.repeat(reps, 1, 1, 1)[:num_frames]
    elif m.shape[0] > num_frames:
        m = m[:num_frames]

    # 空间自动 resize 到目标 (H, W)（用户传的 mask 可能与 pixel-space video 完全一致，也可能不一致）
    if m.shape[-2] != height or m.shape[-1] != width:
        m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)

    return m.contiguous()


def _pil_to_numpy(img):
    import numpy as np
    return np.array(img)


# ------------------------- mask 空间/时间下采 ------------------------- #

def mask_temporal_downsample_causal(
    mask_tchw: torch.Tensor,
    T_latent: int,
    spatial_stride: int = 8,
) -> torch.Tensor:
    """把 (T, 1, H, W) 的 pixel-space mask 下采到 latent-space (T', H/8, W/8, 1)。

    - 空间：avg_pool2d(kernel=8)，得连续 [0,1]，物理意义 = 8x8 pixel patch 内的保护占比
    - 时间：匹配 VAE 的 causal 4x 结构：latent[0] 独占 pixel[0]，之后每 4 帧压 1

    要求：(T - 1) % 4 == 0（由现有 _cut_videos 保证）
    """
    assert mask_tchw.ndim == 4 and mask_tchw.shape[1] == 1, mask_tchw.shape
    T = mask_tchw.shape[0]

    m = F.avg_pool2d(mask_tchw, kernel_size=spatial_stride, stride=spatial_stride)  # (T,1,H',W')

    head = m[:1]                                                                    # (1, 1, H', W')
    tail = m[1:]                                                                    # (T-1, ...)
    assert tail.shape[0] % 4 == 0, (
        f"mask 时间下采要求 (T-1)%4==0，当前 T={T} → tail={tail.shape[0]}。"
        "调用方需保证 mask 与 video 走过同一 _cut_videos。"
    )
    tail = tail.reshape(-1, 4, *tail.shape[1:]).mean(dim=1)                         # ((T-1)/4, 1, H', W')
    m = torch.cat([head, tail], dim=0)                                              # (T', 1, H', W')
    assert m.shape[0] == T_latent, f"time downsample mismatch: {m.shape[0]} vs {T_latent}"

    return m.permute(0, 2, 3, 1).contiguous()                                       # (T', H', W', 1)


def override_cond_channel_17(cond: torch.Tensor, mask_latent: torch.Tensor) -> torch.Tensor:
    """把 condition 的最后 1 通道替换为 mask_latent。

    cond:        (T', H', W', 17)  —— get_condition() 的返回
    mask_latent: (T', H', W', 1)   —— mask_temporal_downsample_causal 的返回
    """
    assert cond.shape[:-1] == mask_latent.shape[:-1], f"{cond.shape} vs {mask_latent.shape}"
    cond = cond.clone()
    cond[..., -1:] = mask_latent.to(cond.dtype).to(cond.device)
    return cond


# ------------------------- LoRA ------------------------- #

# LoRA 挂载的目标 Linear。
#
# SeedVR 的 attn/ffn 用了自定义 MMModule 包装：内部是 .vid + .txt（前 10 层）或 .all（后 22 层）
# 两/一个 nn.Linear。peft 若用后缀列表匹配，会命中 MMModule 本身（非 Linear，报错），
# 因此改用 regex 精确匹配到 MMModule 里面的 Linear。
LORA_TARGET_REGEX = r".*\.(proj_qkv|proj_out|proj_in_gate|proj_in)\.(vid|txt|all)$"


def attach_lora(dit, r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """给 DiT 挂 LoRA，返回 PeftModel（替换原 dit）。

    必须在 configure_dit_model（加载完官方权重）之后调用。
    """
    import re
    import torch.nn as nn

    from peft import LoraConfig, get_peft_model

    # 挂载前 sanity check：列出 regex 命中的 Linear 数量，避免静默不匹配
    matched = [
        n for n, m in dit.named_modules()
        if isinstance(m, nn.Linear) and re.fullmatch(LORA_TARGET_REGEX, n)
    ]
    assert len(matched) > 0, (
        f"LoRA target regex `{LORA_TARGET_REGEX}` 没有命中任何 Linear，"
        f"peft 结构可能变了。"
    )
    print(f"[LoRA] target regex 命中 {len(matched)} 个 Linear，样例: {matched[:3]}")

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=LORA_TARGET_REGEX,   # 字符串 → peft 走 re.fullmatch
    )
    peft_dit = get_peft_model(dit, cfg)
    return peft_dit


def unfreeze_vid_in_proj(peft_dit) -> int:
    """解冻 NaPatchIn 的首层线性层（vid_in.proj）。

    peft 默认冻结所有非 LoRA 参数，我们把这个 33 → 2560 的入口层显式解冻，
    因为第 17 通道语义变了，这一层最直接受影响。

    返回：解冻的参数张量数（应为 2：weight + bias，或者 1：weight only）
    """
    count = 0
    for name, p in peft_dit.named_parameters():
        # peft 会把原模块包在 base_model.model.* 下，所以只判断后缀
        if name.endswith("vid_in.proj.weight") or name.endswith("vid_in.proj.bias"):
            p.requires_grad_(True)
            count += 1
    assert count > 0, "没找到 vid_in.proj 参数，peft 命名规则可能变了"
    return count


# ------------------------- 可训权重存取 ------------------------- #

def save_trainable_state(peft_dit, save_dir: str, filename: str = "trainable.pt"):
    """保存所有 requires_grad=True 的参数（LoRA + vid_in.proj）。"""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        n: p.detach().cpu().clone()
        for n, p in peft_dit.named_parameters()
        if p.requires_grad
    }
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_trainable_state(peft_dit, ckpt_path: str) -> List[str]:
    """加载 save_trainable_state 保存的权重，返回未匹配的 key 列表（用于调试）。"""
    state = torch.load(ckpt_path, map_location="cpu")
    missing_or_unexpected = peft_dit.load_state_dict(state, strict=False)
    # 只报告 "unexpected"（我们传入的 key 在模型里没找到）；missing 是预期的（模型里绝大多数 key 不在训练权重里）
    return list(missing_or_unexpected.unexpected_keys)


def count_trainable_params(peft_dit) -> dict:
    """统计可训 vs 总参数量，供 sanity check。"""
    trainable = sum(p.numel() for p in peft_dit.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_dit.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / total if total > 0 else 0.0,
    }
