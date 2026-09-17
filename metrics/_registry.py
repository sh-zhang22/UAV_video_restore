"""指标注册表。风格与 methods/_registry.py 对齐。

单指标函数签名：
    fn(pred: torch.Tensor, gt: torch.Tensor, **kwargs) -> float
其中 pred/gt 的具体 shape/dtype 由各指标自行约定，evaluate() 会统一喂入。
"""
from __future__ import annotations

from typing import Callable, Dict, List

MetricFn = Callable[..., float]

REGISTRY: Dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def deco(fn: MetricFn) -> MetricFn:
        if name in REGISTRY:
            raise ValueError(f"指标名重复注册: {name}")
        REGISTRY[name] = fn
        return fn
    return deco


def list_metrics() -> List[str]:
    return list(REGISTRY.keys())
