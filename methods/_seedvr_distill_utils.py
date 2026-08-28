"""SeedVR2-3B DistilBERT-style 层裁剪 + KD 蒸馏工具。

思路：
- 教师: SeedVR2-3B NaDiT，num_layers=32（前 10 mm-layer + 后 22 shared-weights）
- 学生: NaDiT，num_layers=20（前 10 mm-layer 全保 + 后 22 均匀取 10）
- 结构完全同构，只 num_layers 不同 → 学生 state_dict 用教师权重直接初始化，只是 blocks 抽子集
- 推理时无需新 method：加 CLI 参数覆盖 config.dit.model.num_layers 即可

设计原则：
- 纯函数、延迟 import SeedVR 符号
- 不动 third_party/SeedVR/ 任何一行
- 未启用蒸馏路径时（默认）不会被 import 到，零副作用
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


# ------------------------- 层裁剪映射 ------------------------- #

TEACHER_TOTAL_LAYERS = 32
STUDENT_NUM_LAYERS = 20
MM_LAYERS = 10  # 前 10 层 shared_weights=False（有 .vid / .txt），保留全部

# 层保留策略（DistilBERT 均匀跨步）：
# - 前 10 mm-layer 全保 [0..9]
# - 后 22 shared 层里均匀取 10，端点包含 → round(linspace(10, 31, 10))
#   实际值：[10, 12, 15, 17, 19, 22, 24, 26, 29, 31]
TEACHER_LAYER_MAP: List[int] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 12, 15, 17, 19, 22, 24, 26, 29, 31,
]
assert len(TEACHER_LAYER_MAP) == STUDENT_NUM_LAYERS
assert TEACHER_LAYER_MAP[:MM_LAYERS] == list(range(MM_LAYERS)), (
    "前 10 mm-layer 必须完整保留：shared_weights=False 的结构与 shared=True 权重不兼容"
)
assert all(0 <= i < TEACHER_TOTAL_LAYERS for i in TEACHER_LAYER_MAP)
assert TEACHER_LAYER_MAP == sorted(TEACHER_LAYER_MAP), "layer_map 必须单调递增"


# window_method 每 2 层一循环："win", "swin"
WINDOW_METHOD_WIN = "720pwin_by_size_bysize"
WINDOW_METHOD_SWIN = "720pswin_by_size_bysize"


def _teacher_window_method(teacher_layer_idx: int) -> str:
    return WINDOW_METHOD_WIN if teacher_layer_idx % 2 == 0 else WINDOW_METHOD_SWIN


# ------------------------- config override ------------------------- #

def derive_student_config_inplace(config, layer_map: Sequence[int] = TEACHER_LAYER_MAP) -> None:
    """就地把 config.dit.model 从教师配置改为学生配置。

    - num_layers 32 → len(layer_map)
    - window_method: 显式列表，按 layer_map[j] 是奇是偶取教师原语义
    - block_type / window: 教师是 num_layers * [同一值]，学生同构不用改（OmegaConf eval 会重解析）
      但为了绕开 OmegaConf 的插值副作用，我们显式覆盖成 list（避免 num_layers 改后 eval 出错）
    - mm_layers: 保持 10 → 学生前 10 层 shared_weights=False，与 layer_map[:10]=[0..9] 对齐
    """
    from omegaconf import OmegaConf

    OmegaConf.set_readonly(config, False)
    model_cfg = config.dit.model

    n_new = len(layer_map)

    # OmegaConf ${eval:...} 表达式在 assign 时会立即 resolve。deepcopy 后可能已是普通 python list，
    # 不能再走 OmegaConf.to_container；直接迭代取值即可。
    def _as_py_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        # OmegaConf ListConfig 也可迭代
        return [v for v in x]

    old_block_type = _as_py_list(model_cfg.block_type)
    old_window = [_as_py_list(w) for w in model_cfg.window]

    # 断言老 config 与我们的 layer_map 前提一致
    assert len(old_block_type) == TEACHER_TOTAL_LAYERS
    assert len(old_window) == TEACHER_TOTAL_LAYERS
    assert all(bt == old_block_type[0] for bt in old_block_type), (
        f"教师 block_type 非全一致，学生映射策略需要重新审视: {set(old_block_type)}"
    )
    assert all(tuple(w) == tuple(old_window[0]) for w in old_window), (
        f"教师 window 非全一致，学生映射策略需要重新审视: {old_window[:3]}"
    )

    model_cfg.num_layers = n_new
    # 显式覆盖三个 list（避免 ${eval:...} 中的 num_layers 引用引发的重解析问题）
    model_cfg.block_type = [old_block_type[0]] * n_new
    model_cfg.window = [list(old_window[0]) for _ in range(n_new)]
    model_cfg.window_method = [_teacher_window_method(i) for i in layer_map]

    # mm_layers 保持 10 不变（layer_map[:10] 都是 <10 的教师层，同 shared_weights=False）


# ------------------------- state_dict 层拷贝 ------------------------- #

_BLOCK_PREFIX_RE = re.compile(r"^blocks\.(\d+)\.")


def build_student_state_dict_from_teacher(
    teacher_sd: Dict[str, torch.Tensor],
    layer_map: Sequence[int] = TEACHER_LAYER_MAP,
) -> Dict[str, torch.Tensor]:
    """从教师 state_dict 派生学生 state_dict。

    - 非 blocks.* 的 key（vid_in / vid_out / emb_in / vid_out_norm / vid_out_ada）直接原样保留
    - blocks.<i>.<sub> 只保留 i ∈ layer_map 的，重新映射为 blocks.<new_j>.<sub>
    """
    teacher_to_student_idx: Dict[int, int] = {
        t: s for s, t in enumerate(layer_map)
    }

    student_sd: Dict[str, torch.Tensor] = {}
    kept = 0
    for k, v in teacher_sd.items():
        m = _BLOCK_PREFIX_RE.match(k)
        if m is None:
            student_sd[k] = v
            continue
        t_idx = int(m.group(1))
        if t_idx not in teacher_to_student_idx:
            continue  # 教师层，学生不要
        s_idx = teacher_to_student_idx[t_idx]
        new_k = _BLOCK_PREFIX_RE.sub(f"blocks.{s_idx}.", k, count=1)
        student_sd[new_k] = v
        kept += 1

    print(
        f"[distill] state_dict: teacher_keys={len(teacher_sd)}, student_keys={len(student_sd)}, "
        f"block_keys_kept={kept}"
    )
    return student_sd


# ------------------------- 参数量统计 ------------------------- #

def count_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / total if total > 0 else 0.0,
    }


# ------------------------- 中间层特征 hook ------------------------- #

class BlockOutputCapture:
    """给 NaDiT.blocks 每层注册 forward_hook，捕获 vid 输出（tuple[0]）。

    NaDiT block forward 返回 (vid, txt, vid_shape, txt_shape)，我们只捕获 vid。
    使用：
        cap = BlockOutputCapture(dit)      # attach
        cap.clear()
        y = dit(...)                        # forward
        feats = cap.features                # list[Tensor]，长度 = num_layers
        cap.remove()                        # detach hooks
    """

    def __init__(self, dit: nn.Module):
        self.features: List[torch.Tensor] = []
        self._handles = []
        for i, block in enumerate(dit.blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            # output 是 tuple；取 vid（第 0 个）
            if isinstance(output, (tuple, list)):
                vid = output[0]
            else:
                vid = output
            # 保存引用（不 detach —— 学生侧需要梯度；教师侧上层用 no_grad 保证不建图）
            # 但为了避免 hook list 无限增长，用 [idx] 位置替换
            while len(self.features) <= idx:
                self.features.append(None)
            self.features[idx] = vid
        return hook

    def clear(self):
        self.features = []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
