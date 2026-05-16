"""方法注册表与装饰器。

适配器统一签名：
    fn(*, video_path, recovered_path, ckpt_path, device, **kwargs) -> str
返回实际写出的 MP4 路径。
"""

from __future__ import annotations

from typing import Callable, Dict, List

RecoverFn = Callable[..., str]

REGISTRY: Dict[str, RecoverFn] = {}


def register(name: str) -> Callable[[RecoverFn], RecoverFn]:
    """把适配函数注册到 REGISTRY。

    Example:
        @register("basicvsrpp")
        def _run(*, video_path, recovered_path, ckpt_path, device, **kwargs):
            ...
            return recovered_path
    """

    def deco(fn: RecoverFn) -> RecoverFn:
        if name in REGISTRY:
            raise ValueError(f"方法名重复注册: {name}")
        REGISTRY[name] = fn
        return fn

    return deco


def list_methods() -> List[str]:
    return list(REGISTRY.keys())
