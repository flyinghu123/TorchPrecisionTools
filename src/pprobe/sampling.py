"""张量元素采样：均匀 / 随机 / 头部连续，以及数值与逻辑下标的提取。

设计要点：
* 采样个数与模式全部由 ``PPROBE_SAMPLE_MODE`` / ``PPROBE_SAMPLE_N`` 控制，默认均匀 50 个；
* 随机模式由 ``PPROBE_SEED`` 控制：未设置则每个进程用熵源随机种子（并记录到 manifest，
  便于复现），设置了则用该种子构造独立的 CPU 生成器；
* 除数值外还记录扁平下标与逻辑多维下标，跨运行对比时按“相同下标”逐元素对齐。
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .util import encode_number, log


def _unravel(flat_idx: int, shape: list[int]) -> list[int]:
    """扁平下标 -> 逻辑多维下标（按 C 序，与实际 flatten 语义一致）。"""
    out = [0] * len(shape)
    rem = flat_idx
    for i in range(len(shape) - 1, -1, -1):
        dim = shape[i] or 1
        out[i] = rem % dim
        rem //= dim
    return out


class TensorSampler:
    def __init__(self, mode: str = "uniform", n: int = 50, seed: int | None = None, layout: str = "flat"):
        self.mode = mode
        self.n = max(0, int(n))
        self.seed = seed
        self.layout = layout
        self.calls = 0
        if seed is None:
            self.seed_used = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
            self.seed_is_user = False
        else:
            self.seed_used = int(seed) & ((1 << 62) - 1)
            self.seed_is_user = True
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(self.seed_used)
        self.failures: dict[str, int] = {}

    # ------------------------------------------------------------------
    def count_for(self, numel: int) -> int:
        if self.mode == "off" or self.n == 0:
            return 0
        return min(self.n, int(numel))

    def uniform_indices(self, numel: int, k: int) -> list[int]:
        """两端对齐的等间隔下标；k==numel 时即全量。"""
        if k <= 0 or numel <= 0:
            return []
        if k >= numel:
            return list(range(numel))
        if k == 1:
            return [0]
        step = (numel - 1) / (k - 1)
        idx = [min(numel - 1, int(round(i * step))) for i in range(k)]
        return list(dict.fromkeys(idx))  # 去重保序

    def grid_indices(self, shape: list[int], k: int) -> list[int]:
        """逐维等间隔网格采样，适合 attention/conv 这类需要“看结构”的张量。"""
        return grid_sample_indices(shape, k)

    def random_indices(self, numel: int, k: int) -> list[int]:
        if k <= 0 or numel <= 0:
            return []
        if k >= numel:
            return list(range(numel))
        t = torch.randperm(numel, generator=self._gen)[:k]
        return sorted(int(x) for x in t.tolist())

    def indices(self, shape: list[int]) -> list[int]:
        numel = 1
        for s in shape:
            numel *= max(0, int(s))
        k = self.count_for(numel)
        if k == 0:
            return []
        if self.mode == "head":
            return list(range(min(k, numel)))
        if self.mode == "random":
            return self.random_indices(numel, k)
        if self.layout == "grid" and len(shape) > 1:
            return self._grid(shape, k)
        return self.uniform_indices(numel, k)

    def _grid(self, shape: list[int], k: int) -> list[int]:
        numel = 1
        for s in shape:
            numel *= max(0, int(s))
        try:
            return grid_sample_indices(shape, k)
        except Exception as e:  # 网格采样失败时退回扁平均匀采样
            self._note("grid:" + type(e).__name__)
            return self.uniform_indices(numel, k)

    def _note(self, key: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1


def grid_sample_indices(shape: list[int], k: int) -> list[int]:
    """按维度等间隔取网格点，返回排序后的扁平下标列表。"""
    numel = 1
    for s in shape:
        numel *= max(0, int(s))
    if k <= 0 or numel <= 0:
        return []
    ndim = max(1, len(shape))
    per_dim = max(1, int(round(k ** (1.0 / ndim))))
    while per_dim > 1 and per_dim**ndim > 4 * k:
        per_dim -= 1
    axes = []
    for size in shape:
        size = max(1, int(size))
        axes.append(_axis(size, min(per_dim, size)))
    flat: list[int] = []
    row = [0] * ndim
    _walk(flat, row, axes, 0, shape)
    flat.sort()
    return flat


def _axis(size: int, k: int) -> list[int]:
    if k >= size:
        return list(range(size))
    if k <= 1:
        return [0]
    step = (size - 1) / (k - 1)
    return list(dict.fromkeys(min(size - 1, int(round(i * step))) for i in range(k)))


def _walk(out: list[int], cur: list[int], axes: list[list[int]], depth: int, shape: list[int]) -> None:
    if depth == len(shape):
        lin = 0
        for i, size in enumerate(shape):
            lin = lin * (size or 1) + cur[i]
        out.append(lin)
        return
    for idx in axes[depth]:
        cur[depth] = idx
        _walk(out, cur, axes, depth + 1, shape)


# ----------------------------------------------------------------------
_QUANT_DTYPES = (torch.qint8, torch.quint8, torch.qint32, torch.quint4x2)


def describe_unsupported(t) -> str | None:
    """判断张量是否无法取值，返回原因字符串（None 表示可以采样）。"""
    try:
        if t.is_meta:
            return "meta"
    except Exception:
        pass
    try:
        if t.is_sparse or t.is_sparse_csr:
            return "sparse"
    except Exception:
        pass
    if getattr(t, "dtype", None) in _QUANT_DTYPES:
        return "quantized"
    try:
        if t.numel() == 0:
            return "empty"
    except Exception:
        return "no-numel"
    return None


def _to_display_dtype(t) -> torch.dtype:
    """把低精度浮点提升到 fp32/fp64，避免统计与 dump 溢出/截断。

    注意：``Tensor.is_floating_point`` / ``is_complex`` 是**方法**，直接当属性判真值会恒为 True。
    """
    if t.dtype == torch.bool:
        return torch.int8
    if torch.is_floating_point(t):
        return torch.float64 if t.dtype == torch.float64 else torch.float32
    if torch.is_complex(t):
        return torch.complex128
    return torch.int64


def sample_tensor(t: "torch.Tensor", sampler: TensorSampler, want_bits: bool = False) -> dict[str, Any] | None:
    """采样单个张量，返回 ``{"n","idx","idx_nd","vals",...}``；不可采样时返回 None。"""
    reason = describe_unsupported(t)
    if reason:
        return {"unavailable": reason}
    shape = list(t.shape)
    idx = sampler.indices(shape)
    if not idx:
        return {"n": 0}
    try:
        flat = t.detach().flatten()
        dev = flat.device
        idx_t = torch.tensor(idx, dtype=torch.long, device=dev if dev.type != "meta" else "cpu")
        gathered = flat[idx_t]
        cast = _to_display_dtype(t)
        if cast in (torch.float32, torch.float64):
            gathered = gathered.to(cast).to(torch.float64) if cast != torch.float64 else gathered
            cpu = gathered.detach().to("cpu")
            vals = [encode_number(float(v)) for v in cpu.tolist()]
        elif cast == torch.complex128:
            gathered = gathered.to(cast)
            cpu = gathered.detach().to("cpu")
            vals = [[encode_number(float(v.real)), encode_number(float(v.imag))] for v in cpu.tolist()]
        else:
            gathered = gathered.long()
            vals = [int(v) for v in gathered.detach().to("cpu").tolist()]
        out: dict[str, Any] = {
            "n": len(idx),
            "idx": idx,
            "vals": vals,
        }
        if want_bits:
            bits = _raw_bits(flat[idx_t], t.dtype)
            if bits is not None:
                out["bits"] = bits
        if 1 < len(shape) <= 8 and all(s > 0 for s in shape):
            out["idx_nd"] = [_unravel(i, shape) for i in idx]
        return out
    except Exception as e:  # 单个张量采样失败不能影响训练
        sampler._note(f"sample:{type(e).__name__}")
        log(f"采样张量失败（已跳过）: {e!r}")
        return {"unavailable": f"error:{type(e).__name__}"}


def _raw_bits(gathered: "torch.Tensor", dtype: torch.dtype) -> list[int] | None:
    """按原始位模式导出采样值，跨平台逐 bit 对比时最有信息量。"""
    try:
        view_dtype = {
            torch.float32: torch.int32,
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float64: torch.int64,
        }.get(dtype)
        if view_dtype is None:
            return None
        g = gathered.contiguous().view(view_dtype)
        return [int(v) for v in g.to("cpu").tolist()]
    except Exception:
        return None
