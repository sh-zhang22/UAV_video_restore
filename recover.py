"""压缩 MP4 视频还原模块。

对外仅暴露 Recover 接口：输入一段经过压缩的 MP4，输出一段还原后（更清晰）的 MP4。

通过 method 字符串切换不同的开源方法。各方法的具体实现位于 methods/<name>.py，
依赖的原项目代码位于 third_party/<project_name>/。
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional

from methods._registry import REGISTRY, list_methods


def Recover(
    video_path: str,
    recovered_path: str,
    ckpt_path: str,
    *,
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
) -> str:
    """对压缩 MP4 进行还原，输出更清晰的 MP4。

    Args:
        video_path: 输入的压缩 MP4 文件路径。
        recovered_path: 还原后 MP4 的输出路径。
        ckpt_path: 还原模型的权重 checkpoint 路径。
        method: 还原方法名（已在 methods/ 下注册的字符串标识）。
        method_kwargs: 透传给具体方法的额外关键字参数，默认 None。
        device: 可选，运行设备（如 "cuda:0"）；为 None 时由具体方法自行决定。

    Returns:
        实际写出的 MP4 文件路径（一般等于 recovered_path）。

    Raises:
        FileNotFoundError: 输入视频或 checkpoint 不存在。
        KeyError: method 名未注册。
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"输入视频不存在: {video_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"权重 checkpoint 不存在: {ckpt_path}")

    out_dir = os.path.dirname(os.path.abspath(recovered_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _ensure_method_loaded(method)
    if method not in REGISTRY:
        available = ", ".join(sorted(list_methods())) or "(空)"
        raise KeyError(f"未注册的方法: {method}；当前可用: {available}")

    fn = REGISTRY[method]
    return fn(
        video_path=video_path,
        recovered_path=recovered_path,
        ckpt_path=ckpt_path,
        device=device,
        **(method_kwargs or {}),
    )


def available_methods() -> List[str]:
    """列出当前已成功注册的所有方法名。"""
    _ensure_all_methods_loaded()
    return sorted(list_methods())


def _ensure_method_loaded(method: str) -> None:
    """按需 import methods.<method>，触发其 @register 注册。

    单独 import 单个方法的好处：某个方法依赖未装也不会影响其他方法可用。
    """
    if method in REGISTRY:
        return
    try:
        importlib.import_module(f"methods.{method}")
    except ModuleNotFoundError:
        # 留给上层抛 KeyError，给出更友好的可用方法列表
        return


def _ensure_all_methods_loaded() -> None:
    methods_dir = os.path.join(os.path.dirname(__file__), "methods")
    if not os.path.isdir(methods_dir):
        return
    for name in os.listdir(methods_dir):
        if not name.endswith(".py"):
            continue
        if name.startswith("_"):
            continue
        mod = name[:-3]
        try:
            importlib.import_module(f"methods.{mod}")
        except Exception:
            # 某个方法导入失败不影响其他方法可用
            continue
