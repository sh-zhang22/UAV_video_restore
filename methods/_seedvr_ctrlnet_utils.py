"""seedvr2_3b_ctrlnet 专用工具：ControlNet-Lite 侧枝（mask 条件）挂载与权重存取。

设计原则：
- 不改 third_party/SeedVR 一行代码。ControlledDiT 的 forward 从 nadit.py 复制骨架
  过来（30 行），在 for block loop 里插入侧枝调用。
- 侧枝用 K 个 shared_weights=True 的 NaMMSRTransformerBlock（约 80M/层）
  + K 个 zero-init Linear（约 6.6M/层），默认 K=4，共约 346M（占 3B 主干 ~11.5%）。
- zero-init 保证训练启动时侧枝输出全 0，主干等价 baseline seedvr2_3b。
- mask 通过 ControlledDiT.set_mask(mask_latent) 前置注入（推理和训练调用相同）。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ------------------------- MaskPatchIn ------------------------- #

class MaskPatchIn(nn.Module):
    """把 (T', H', W', 1) 的 mask_latent 映射到 (L, vid_dim) 的侧枝隐层输入。

    与主干 NaPatchIn 的输出 shape 严格对齐（同 patch_size, 同 slice_inputs 分片规则）。
    单视频语义：batch=1，直接 flatten。若需 batch>1 请调用方先 na.flatten。
    """
    def __init__(self, vid_dim: int, patch_size: Tuple[int, int, int]):
        super().__init__()
        pt, ph, pw = patch_size
        self.patch_size = (pt, ph, pw)
        # 每个 patch 内元素数 = pt * ph * pw * in_channels(=1)
        self.proj = nn.Linear(pt * ph * pw * 1, vid_dim)

    def forward(self, mask_latent: torch.Tensor) -> torch.Tensor:
        """mask_latent: (T', H', W', 1) → (L, vid_dim)，L = T' * (H'/ph) * (W'/pw)。"""
        from einops import rearrange
        from common.distributed.ops import slice_inputs

        pt, ph, pw = self.patch_size
        assert mask_latent.ndim == 4 and mask_latent.shape[-1] == 1, mask_latent.shape
        T_, H_, W_, _ = mask_latent.shape
        # patch_size=[1,2,2] 时不需要时间 pad，H/W 已经保证被 2 整除（因为主干 vid_in 走通同样规则）
        assert T_ % pt == 0 and H_ % ph == 0 and W_ % pw == 0, (
            f"mask_latent {mask_latent.shape} 与 patch_size {self.patch_size} 不齐"
        )
        # (T', H', W', 1) → (T'/pt, H'/ph, W'/pw, pt*ph*pw*1)
        x = rearrange(mask_latent, "(T t) (H h) (W w) c -> T H W (t h w c)",
                      t=pt, h=ph, w=pw)
        # flatten spatial → (L, C_patch)
        x = rearrange(x, "T H W c -> (T H W) c")
        # 与 NaPatchIn 一致：在 proj 之前 slice
        x = slice_inputs(x, dim=0)
        x = self.proj(x)                                    # (L, vid_dim)
        return x


# ------------------------- ControlledDiT ------------------------- #

class ControlledDiT(nn.Module):
    """外挂 ControlNet 侧枝到已加载的 NaDiT。

    forward 签名与 NaDiT 一致：(vid, txt, vid_shape, txt_shape, timestep, disable_cache=False)
    →  NaDiTOutput(vid_sample=...)

    侧枝用法：
        controlled = wrap_with_controlnet(runner.dit, K=4, ...)
        controlled.set_mask(mask_latent)  # (T', H', W', 1)
        out = controlled(vid=..., txt=..., ...)
        controlled.set_mask(None)         # 清空，避免下次 forward 拿旧 mask
    """
    def __init__(
        self,
        base_dit: nn.Module,
        K: int,
        vid_dim: int,
        patch_size: Tuple[int, int, int],
        side_block_kwargs: dict,
    ):
        super().__init__()
        self.base_dit = base_dit
        self.K = K
        self.vid_dim = vid_dim

        # 主干整体 freeze（侧枝训练时只反传到侧枝参数）
        for p in self.base_dit.parameters():
            p.requires_grad_(False)

        # 侧枝 mask stem
        self.mask_stem = MaskPatchIn(vid_dim=vid_dim, patch_size=patch_size)

        # 侧枝 K 层 shared_weights=True 的 mmsr block
        from models.dit_v2.nablocks import get_nablock
        NaBlock = get_nablock("mmdit_sr")
        self.side_blocks = nn.ModuleList([
            NaBlock(shared_weights=True, is_last_layer=False, **side_block_kwargs)
            for _ in range(K)
        ])

        # 侧枝 K 个 zero-init 融合层（把侧枝输出映射后加到主干对应层输出）
        self.zero_convs = nn.ModuleList([
            nn.Linear(vid_dim, vid_dim, bias=True) for _ in range(K)
        ])
        for zc in self.zero_convs:
            nn.init.zeros_(zc.weight)
            nn.init.zeros_(zc.bias)

        # 侧枝 rope 共享主干同层的 rope 实例，复用其 lru_cache 里的巨大 axial freq tensor
        # （rope 无可训练参数、无 batch state；共享安全且能省 K × ~8GiB 显存）
        n_share = min(K, len(base_dit.blocks))
        for i in range(n_share):
            try:
                self.side_blocks[i].attn.rope = base_dit.blocks[i].attn.rope
            except AttributeError:
                pass  # 主干该层没有 attn.rope（不应发生，K<=10 时主干必然是 mmdit_sr）

        # 待注入的 mask，由 set_mask 设置；forward 会读并在结束前保留（不清空，由调用方管理）
        self.pending_mask: Optional[torch.Tensor] = None

    def set_mask(self, mask_latent: Optional[torch.Tensor]):
        self.pending_mask = mask_latent

    # 转发给 base_dit，保持既有 API 兼容
    def set_gradient_checkpointing(self, enable: bool):
        self.base_dit.set_gradient_checkpointing(enable)

    @property
    def gradient_checkpointing(self):
        return self.base_dit.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, v):
        self.base_dit.gradient_checkpointing = v

    def forward(
        self,
        vid: torch.Tensor,
        txt,
        vid_shape: torch.Tensor,
        txt_shape,
        timestep,
        disable_cache: bool = False,
    ):
        """从 third_party/SeedVR/models/dit_v2/nadit.py NaDiT.forward 复制骨架，
        在 for block 循环中插入侧枝调用。原 nadit.py 一行不动。"""
        from common.cache import Cache
        from common.distributed.ops import slice_inputs
        from models.dit_v2 import na
        from models.dit_v2.nadit import NaDiTOutput, gradient_checkpointing

        # 用主干的 base_dit 引用短化后续代码
        base = self.base_dit
        cache = Cache(disable=disable_cache)
        # 侧枝走独立 cache，避免 window 索引等被主干 attn 复用出错
        side_cache = Cache(disable=True)

        # ---- txt（复制自 NaDiT.forward） ----
        if isinstance(txt, list):
            assert isinstance(base.txt_in, nn.ModuleList)
            txt = [
                na.unflatten(fc(i), s) for fc, i, s in zip(base.txt_in, txt, txt_shape)
            ]
            txt, txt_shape = na.flatten([torch.cat(t, dim=0) for t in zip(*txt)])
            txt = slice_inputs(txt, dim=0)
        else:
            txt = slice_inputs(txt, dim=0)
            txt = base.txt_in(txt)

        # ---- vid_in（内部会 slice_inputs） ----
        vid, vid_shape = base.vid_in(vid, vid_shape, cache)

        # ---- timestep embedding ----
        emb = base.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        # ---- 侧枝初始化：mask → mask_stem → 与 vid 同 shape ----
        vid_side: Optional[torch.Tensor] = None
        if self.pending_mask is not None and self.K > 0:
            # dtype 对齐 vid（可能是 bf16 autocast 下）
            mask_in = self.pending_mask.to(device=vid.device, dtype=vid.dtype)
            vid_side = self.mask_stem(mask_in)                          # (L, vid_dim)
            # sanity：侧枝 shape 必须与主干 vid_in 输出对齐
            if vid_side.shape != vid.shape:
                raise RuntimeError(
                    f"[ctrlnet] side stem shape {vid_side.shape} != main vid shape {vid.shape}; "
                    f"检查 mask_latent 形状与 patch_size 是否与主干一致"
                )

        # ---- body: 主干 32 层，前 K 层与侧枝耦合 ----
        for i, block in enumerate(base.blocks):
            # 侧枝先跑一层（i < K 才有效；vid_side 允许为 None 表 mask 未注入）
            if i < self.K and vid_side is not None:
                # 侧枝 block 的 txt 用主干 txt 的 detached clone，避免污染主干 txt path
                txt_side_in = txt.detach()
                vid_side, _txt_side_out, _vs, _ts = gradient_checkpointing(
                    self.side_blocks[i],
                    vid=vid_side,
                    txt=txt_side_in,
                    vid_shape=vid_shape,
                    txt_shape=txt_shape,
                    emb=emb,
                    cache=side_cache,
                    enabled=(self.gradient_checkpointing and self.training),
                )
                # zero_conv 融合到主干（zero-init 保证初始为 0）
                # zero_convs 是 fp32 参数；vid 在 autocast 下可能是 bf16，先转 vid 的 dtype
                zc = self.zero_convs[i]
                # 手动应用 Linear 以保证 dtype/精度可控
                w = zc.weight.to(dtype=vid.dtype)
                b = zc.bias.to(dtype=vid.dtype)
                vid = vid + torch.nn.functional.linear(vid_side, w, b)

            vid, txt, vid_shape, txt_shape = gradient_checkpointing(
                block,
                vid=vid,
                txt=txt,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
                emb=emb,
                cache=cache,
                enabled=(self.gradient_checkpointing and self.training),
            )

        # ---- 主干输出 norm/ada ----
        if base.vid_out_norm:
            vid = base.vid_out_norm(vid)
            vid = base.vid_out_ada(
                vid,
                emb=emb,
                layer="out",
                mode="in",
                hid_len=cache("vid_len", lambda: vid_shape.prod(-1)),
                cache=cache,
                branch_tag="vid",
            )

        # ---- 主干输出 ----
        vid, vid_shape = base.vid_out(vid, vid_shape, cache)
        return NaDiTOutput(vid_sample=vid)


# ------------------------- wrap / count / save / load ------------------------- #

def wrap_with_controlnet(
    base_dit: nn.Module,
    config_dit_model,
    K: int = 4,
) -> ControlledDiT:
    """给主干 DiT 挂 ControlNet 侧枝，返回 ControlledDiT（替换原 runner.dit）。

    Args:
        base_dit:  已加载权重的 NaDiT 实例
        config_dit_model: runner.config.dit.model（OmegaConf），用来构造侧枝 block
        K:  侧枝层数（注入到主干前 K 层输出上），默认 4
    """
    from models.dit_v2.modulation import get_ada_layer
    from models.dit_v2.normalization import get_norm_layer

    cfg = config_dit_model
    vid_dim = int(cfg.vid_dim)
    txt_dim = int(cfg.txt_dim)
    emb_dim = int(cfg.emb_dim)
    heads = int(cfg.heads)
    head_dim = int(cfg.head_dim)
    expand_ratio = int(cfg.expand_ratio)
    norm = get_norm_layer(cfg.norm)
    norm_eps = float(cfg.norm_eps)
    ada = get_ada_layer(cfg.ada)
    qk_bias = bool(cfg.qk_bias)
    qk_norm = get_norm_layer(cfg.qk_norm)
    mlp_type = str(cfg.mlp_type)
    rope_type = str(cfg.rope_type)
    rope_dim = int(cfg.rope_dim)
    # window / window_method 是 per-layer list；侧枝复用第 0 层的配置（前 K 层大多相同 or 交替）
    window = tuple(cfg.window[0]) if hasattr(cfg, "window") and cfg.window is not None else None
    # window_method 交替序列：偶数层用 [0]、奇数用 [1]，我们对侧枝每层取对应 index
    window_method_list = list(cfg.window_method) if (
        hasattr(cfg, "window_method") and cfg.window_method is not None
    ) else None
    msa_type = cfg.msa_type[0] if (
        hasattr(cfg, "msa_type") and cfg.msa_type is not None and not isinstance(cfg.msa_type, str)
    ) else getattr(cfg, "msa_type", None)

    patch_size = tuple(cfg.patch_size)
    if len(patch_size) == 1:
        patch_size = (patch_size[0],) * 3

    side_kwargs_base = dict(
        vid_dim=vid_dim,
        txt_dim=txt_dim,
        emb_dim=emb_dim,
        heads=heads,
        head_dim=head_dim,
        expand_ratio=expand_ratio,
        norm=norm,
        norm_eps=norm_eps,
        ada=ada,
        qk_bias=qk_bias,
        qk_norm=qk_norm,
        mlp_type=mlp_type,
        rope_type=rope_type,
        rope_dim=rope_dim,
        window=window,
        # window_method 需要 per-layer 传入；ControlledDiT 里 side_blocks 是一个 list，
        # 每层构造时用不同的 window_method。这里我们简化：先固定用第 0 层的方法给所有侧枝层，
        # 因为 shared_weights=True 的 mmsr_block 主要靠 attn 的 window 语义，K 层内用同一方法足够冒烟。
        window_method=(window_method_list[0] if window_method_list else None),
        msa_type=msa_type,
    )

    # ControlledDiT 会用这个 kwargs 构造 K 个 NaMMSRTransformerBlock(shared_weights=True)
    controlled = ControlledDiT(
        base_dit=base_dit,
        K=K,
        vid_dim=vid_dim,
        patch_size=patch_size,
        side_block_kwargs=side_kwargs_base,
    )
    return controlled


def count_ctrlnet_params(controlled: ControlledDiT) -> dict:
    """统计 base / side / total 参数量，供 sanity check。"""
    base_total = sum(p.numel() for p in controlled.base_dit.parameters())
    side_total = 0
    for name, p in controlled.named_parameters():
        if name.startswith("base_dit."):
            continue
        side_total += p.numel()
    trainable = sum(p.numel() for p in controlled.parameters() if p.requires_grad)
    return {
        "base": base_total,
        "side": side_total,
        "trainable": trainable,
        "total": base_total + side_total,
        "ratio_side_over_base": side_total / base_total if base_total > 0 else 0.0,
    }


def save_ctrlnet_state(controlled: ControlledDiT, save_dir: str, filename: str = "ctrlnet.pt") -> str:
    """只保存 requires_grad=True 的参数（侧枝：mask_stem + side_blocks + zero_convs）。"""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        n: p.detach().cpu().clone()
        for n, p in controlled.named_parameters()
        if p.requires_grad
    }
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_ctrlnet_state(controlled: ControlledDiT, ckpt_path: str) -> List[str]:
    """加载 save_ctrlnet_state 保存的权重；返回 unexpected keys（主干 keys 缺失是预期）。"""
    state = torch.load(ckpt_path, map_location="cpu")
    result = controlled.load_state_dict(state, strict=False)
    return list(result.unexpected_keys)
