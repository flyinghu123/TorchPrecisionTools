"""把 forward/backward 的输入输出结构展开成「槽位 -> 张量记录」字典。

槽位命名规则（跨运行稳定，作为对比对齐的 key）：
``input[0]`` / ``kwargs.attention_mask`` / ``output`` / ``output.1`` / ``output.loss``
``grad_input[0]`` / ``grad_output[1]`` / ``grad_of:output`` / ``param.weight``
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .sampling import TensorSampler, sample_tensor
from .summary import basic_info, full_tensor_hash, numeric_stats
from .util import RegexMatcher, short_repr

#: 对比时应忽略的“易变字段”（指针/版本每次运行都不同，不代表精度差异）
VOLATILE_BASIC_FIELDS = ("data_ptr", "version", "storage_offset")


def structure_sig(obj: Any, depth: int = 2) -> str:
    """给容器结构生成简短签名，用于发现“输出结构变了”这类问题。"""
    if isinstance(obj, torch.Tensor):
        return "Tensor"
    if depth <= 0:
        return type(obj).__name__
    if isinstance(obj, dict):
        return "{" + ",".join(f"{k}:{structure_sig(v, depth - 1)}" for k, v in list(obj.items())[:6]) + "}"
    if isinstance(obj, (list, tuple)):
        kind = "list" if isinstance(obj, list) else type(obj).__name__
        return f"{kind}[" + ",".join(structure_sig(v, depth - 1) for v in list(obj)[:6]) + "]"
    return type(obj).__name__


class SlotCollector:
    """一次调用内所有 IO 槽位的收集器，负责数量/深度限制与截断标记。"""

    def __init__(self, cfg, sampler: TensorSampler, full_saver: Callable | None = None):
        self.cfg = cfg
        self.sampler = sampler
        self.full_saver = full_saver
        self.slots: dict[str, Any] = {}
        self.n_tensors = 0
        self.truncated = False
        self.want_bits = "bits" in getattr(cfg, "sample_extra", ())

    def add(self, name: str, value: Any) -> None:
        self._walk(value, name, 0)

    def _walk(self, obj: Any, name: str, depth: int) -> None:
        cfg = self.cfg
        if isinstance(obj, torch.Tensor):
            if self.n_tensors >= cfg.max_tensors_per_call > 0:
                self.truncated = True
                return
            self.n_tensors += 1
            self.slots[name] = self.tensor_entry(obj)
            return
        if isinstance(obj, bool):
            self._scalar(name, obj)
        elif isinstance(obj, int):
            self._scalar(name, obj)
        elif isinstance(obj, float):
            from .util import encode_number

            self._scalar(name, encode_number(obj))
        elif obj is None or isinstance(obj, str):
            self._scalar(name, obj if obj is None else short_repr(obj, 80))
        elif isinstance(obj, dict):
            if depth >= cfg.traverse_depth:
                self.slots[name] = {"kind": "dict", "sig": structure_sig(obj, 1)}
                return
            for k, v in obj.items():
                key = k if isinstance(k, str) else short_repr(k, 24)
                self._walk(v, f"{name}.{key}", depth + 1)
        elif isinstance(obj, (list, tuple)):
            if depth >= cfg.traverse_depth:
                self.slots[name] = {"kind": type(obj).__name__, "sig": structure_sig(obj, 1)}
                return
            for i, v in enumerate(obj):
                self._walk(v, f"{name}[{i}]", depth + 1)
        elif hasattr(obj, "detach") and hasattr(obj, "shape"):  # 类 tensor 对象（如 TFLOPS 包装、Parameter 之外的扩展）
            self.n_tensors += 1
            try:  # 先取真实 tensor，否则 basic_info/dtype 等属性访问会报错而丢掉整条事件
                obj = obj.detach()
            except Exception:
                pass
            self.slots[name] = self.tensor_entry(obj)
        else:
            self.slots[name] = {"kind": "object", "cls": type(obj).__name__, "repr": short_repr(obj)}

    def _scalar(self, name: str, value: Any) -> None:
        if not self.cfg.record_scalars:
            return
        self.slots[name] = {"kind": "scalar", "value": value}

    def tensor_entry(self, t: "torch.Tensor") -> dict[str, Any]:
        cfg = self.cfg
        entry: dict[str, Any] = {"kind": "tensor"}
        entry["basic"] = basic_info(t)
        if cfg.stats:
            entry["stats"] = numeric_stats(t, cfg.stats_dtype)
        if cfg.sample_mode != "off":
            sample = sample_tensor(t, self.sampler, want_bits=self.want_bits)
            if sample:
                entry["sample"] = sample
        if cfg.full_hash:
            h = full_tensor_hash(t)
            if h:
                entry["hash"] = h
        if self.full_saver is not None:
            saved = self.full_saver(t)
            if saved:
                entry["full_tensor"] = saved
        return entry


def collect_args(args: tuple, kwargs: dict, collector: SlotCollector, prefix: str = "input") -> None:
    for i, v in enumerate(args):
        collector.add(f"{prefix}[{i}]", v)
    for k, v in (kwargs or {}).items():
        collector.add(f"kwargs.{k}", v)


def collect_params(module: "torch.nn.Module", collector: SlotCollector, limit: int = 8) -> None:
    """记录模块参数（默认关闭）：权重加载/转换导致的精度差异靠它定位。"""
    try:
        for i, (name, p) in enumerate(module.named_parameters(recurse=False)):
            if limit and i >= limit:
                return
            collector.add(f"param.{name}", p)
    except Exception:
        pass


def has_nan(slots: dict[str, Any]) -> bool:
    for entry in slots.values():
        if isinstance(entry, dict) and entry.get("kind") == "tensor":
            if (entry.get("stats") or {}).get("nan_count", 0):
                return True
    return False


def apply_include_exclude(name: str, cls_name: str, include: RegexMatcher, exclude: RegexMatcher,
                          cls_include: RegexMatcher, cls_exclude: RegexMatcher) -> bool:
    """include 优先（命中即保留），随后 exclude 剔除；都为空时全量保留。"""
    if include.active and not include.match(name):
        return False
    if exclude.active and exclude.match(name):
        return False
    if cls_include.active and not cls_include.match(cls_name):
        return False
    if cls_exclude.active and cls_exclude.match(cls_name):
        return False
    return True
