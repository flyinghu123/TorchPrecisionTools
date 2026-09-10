"""张量 summary 信息：基本信息 + 数值统计，全部对 nan/inf/低精度做防护。

统计一律在提升后的精度上累加（float64 或 float32），避免 fp16/bf16 直接求和溢出。
"""

from __future__ import annotations

from typing import Any

import torch

from .util import encode_number

_MAX_REDUCE_ELEMS = 1 << 31  # 超过该规模只做 max/min，跳过 mean/var 以控制耗时
NAN = float("nan")
_INT_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
_FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}


def dtype_limits(dtype: torch.dtype) -> dict[str, Any] | None:
    """记录 dtype 的表示能力，定位“精度损失 vs bug”时非常关键。"""
    try:
        if dtype == torch.bool:
            return None
        if dtype in _FLOAT_DTYPES:
            fi = torch.finfo(dtype)
            return {"eps": encode_number(float(fi.eps)), "tiny": encode_number(float(fi.tiny)),
                    "max": encode_number(float(fi.max)), "min": encode_number(float(fi.min))}
        ii = torch.iinfo(dtype)
        return {"max": int(ii.max), "min": int(ii.min)}
    except Exception:
        return None


def basic_info(t: "torch.Tensor") -> dict[str, Any]:
    """与数值无关的结构信息（shape/stride/device/dtype 等）。"""
    info: dict[str, Any] = {}
    try:
        info["cls"] = type(t).__name__
        info["shape"] = list(t.shape)
        info["stride"] = list(t.stride())
        info["dtype"] = str(t.dtype)
        info["device"] = str(t.device)
        info["numel"] = int(t.numel())
        info["requires_grad"] = bool(t.requires_grad)
        info["is_leaf"] = bool(getattr(t, "is_leaf", True))
        info["contiguous"] = bool(t.is_contiguous())
        info["storage_offset"] = int(t.storage_offset())
        try:
            info["nbytes"] = int(t.numel() * t.element_size())
        except Exception:
            pass
        grad_fn = getattr(t, "grad_fn", None)
        if grad_fn is not None:
            info["grad_fn"] = type(grad_fn).__name__
        try:
            info["version"] = int(t._version)
        except Exception:
            pass
        try:
            info["data_ptr"] = int(t.data_ptr())
        except Exception:
            pass
        info["is_meta"] = bool(t.is_meta)
        try:
            info["is_sparse"] = bool(t.is_sparse or t.is_sparse_csr)
        except Exception:
            info["is_sparse"] = bool(getattr(t, "is_sparse", False))
        lim = dtype_limits(t.dtype)
        if lim:
            info["dtype_limits"] = lim
    except Exception as e:
        info["error"] = repr(e)
    return info


def numeric_stats(t: "torch.Tensor", stats_dtype: str = "float64") -> dict[str, Any]:
    """数值概览：最值/均值/方差/nan 个数/inf 个数/checksum。"""
    out: dict[str, Any] = {}
    td = t.detach()
    try:
        if td.is_meta or td.numel() == 0:
            return {"skipped": "meta-or-empty"}
        if getattr(td, "is_sparse", False):
            td = td.coalesce().values() if hasattr(td, "coalesce") else td
        acc = torch.float64 if stats_dtype == "float64" else torch.float32

        if torch.is_complex(td):
            _complex_stats(td, out, acc)
        elif td.dtype == torch.bool:
            _bool_stats(td, out)
        elif td.dtype in _INT_DTYPES:
            _int_stats(td, out, acc)
        else:
            _float_stats(td, out, acc)
        return out
    except Exception as e:
        out["stats_error"] = f"{type(e).__name__}: {e}"
        return out


def _float_stats(td: "torch.Tensor", out: dict[str, Any], acc: torch.dtype) -> None:
    # 先数特殊值；全有限时不做 boolean 压缩，避开一次同尺寸拷贝
    # 注：isinf/isfinite/isposinf 对 int/bool dtype 会报 "imag is not implemented"，故只用于浮点
    inf_mask = torch.isinf(td)
    nan_n = _count(torch.isnan(td))
    posinf_n = _count(inf_mask & (td > 0))
    neginf_n = _count(inf_mask & (td < 0))
    out["nan_count"] = nan_n
    out["posinf_count"] = posinf_n
    out["neginf_count"] = neginf_n
    out["inf_count"] = posinf_n + neginf_n
    total = int(td.numel())
    finite_n = total - nan_n - posinf_n - neginf_n
    out["count"] = finite_n
    if finite_n <= 0:
        # 必须走 encode_number：直接写 float('nan') 会让 json.dumps 输出裸 NaN（非法 JSON）
        out["max"] = out["min"] = out["mean"] = out["std"] = out["var"] = encode_number(NAN)
        out["sum"] = out["checksum"] = encode_number(NAN)
        out["all_nonfinite"] = True
        return
    src = td if finite_n == total else td.masked_select(torch.isfinite(td))
    if finite_n != total:
        out["finite_only"] = True

    mn, mx = _reduce(lambda: torch.aminmax(src), lambda: torch.aminmax(src.to(acc)))
    out["max"] = encode_number(_f(mx, acc))
    out["min"] = encode_number(_f(mn, acc))
    out["zero_count"] = _count(torch.eq(src, 0))
    if finite_n > _MAX_REDUCE_ELEMS:
        out["stats_skipped"] = f"numel>{_MAX_REDUCE_ELEMS}"
        return
    s1 = _reduce(lambda: torch.sum(src, dtype=acc), lambda: torch.sum(src.to(acc), dtype=acc))
    mean = s1 / finite_n
    # 用融合实现的 L2 范数反推 sum(x^2)，避免为 var 额外分配同尺寸中间张量
    l2 = _reduce(
        lambda: torch.linalg.vector_norm(src, ord=2, dtype=acc),
        lambda: torch.linalg.vector_norm(src.to(acc), ord=2, dtype=acc),
    )
    var = ((l2 * l2) - finite_n * mean * mean) / max(1, finite_n - 1)
    var = var.clamp(min=0)
    out["sum"] = encode_number(_f(s1, acc))
    out["mean"] = encode_number(_f(mean, acc))
    out["var"] = encode_number(_f(var, acc))
    out["std"] = encode_number(_f(var.sqrt(), acc))
    out["absmax"] = encode_number(max(abs(_f(mn, acc)), abs(_f(mx, acc))))
    out["l2norm"] = encode_number(_f(l2, acc))
    out["checksum"] = encode_number(_f(s1, acc))


def _bool_stats(td: "torch.Tensor", out: dict[str, Any]) -> None:
    total = int(td.numel())
    true_n = _count(td) if td.dtype == torch.bool else int(td.sum().item())
    out["true_count"] = true_n
    out["zero_count"] = total - true_n
    out["mean"] = encode_number(true_n / total if total else 0.0)
    out["count"] = total
    out["nan_count"] = 0
    out["inf_count"] = 0


def _int_stats(td: "torch.Tensor", out: dict[str, Any], acc: torch.dtype) -> None:
    """整数张量（position_ids / attention_mask 常见）：不做浮点拷贝，只给计数与最值。"""
    total = int(td.numel())
    mn, mx = torch.aminmax(td)
    out["min"] = int(mn.item())
    out["max"] = int(mx.item())
    out["zero_count"] = _count(torch.eq(td, 0))
    out["count"] = total
    out["nan_count"] = 0
    out["inf_count"] = 0
    if total <= _INT_FULL_STATS_MAX:
        s1 = torch.sum(td, dtype=acc)
        mean = s1 / total
        l2 = torch.linalg.vector_norm(td.to(acc), ord=2, dtype=acc)
        var = ((l2 * l2) - total * mean * mean) / max(1, total - 1)
        out["sum"] = encode_number(float(s1))
        out["mean"] = encode_number(float(mean))
        out["var"] = encode_number(float(var.clamp(min=0)))
        out["std"] = encode_number(float(var.clamp(min=0).sqrt()))
        out["l2norm"] = encode_number(float(l2))
        out["checksum"] = encode_number(float(s1))
    else:
        out["stats_skipped"] = f"int-numel>{_INT_FULL_STATS_MAX}"


_INT_FULL_STATS_MAX = 4 << 20


def _reduce(prim, fallback):
    """先试原生 dtype + dtype= 累加（零拷贝），低精度不支持时再提升一次。"""
    try:
        return prim()
    except Exception:
        return fallback()


def _f(v: "torch.Tensor", acc: torch.dtype) -> float:
    try:
        return float(v.to(acc))
    except Exception:
        return float(v)


def _count(mask: "torch.Tensor") -> int:
    try:
        return int(mask.sum().item())
    except Exception:
        return -1


def _complex_stats(td: "torch.Tensor", out: dict[str, Any], acc: torch.dtype) -> None:
    absv = td.abs().to(acc)
    out["nan_count"] = _count(torch.isnan(td.real) | torch.isnan(td.imag))
    out["inf_count"] = _count(torch.isinf(td.real) | torch.isinf(td.imag))
    out["absmax"] = encode_number(float(absv.max()))
    out["absmean"] = encode_number(float(absv.mean()))
    # linalg.vector_norm 对复数输入不接受 dtype=（会报 dtype 不匹配），改用已提升精度的模长
    out["l2norm"] = encode_number(float((absv * absv).sum().sqrt()))
    out["checksum"] = encode_number(float(td.sum(dtype=torch.complex128).real))
    out["count"] = int(td.numel())


def full_tensor_hash(t: "torch.Tensor") -> dict[str, Any] | None:
    """原始位模式 blake2b，跨平台“逐 bit 是否一致”的终极判据（开销较大，默认关闭）。"""
    import hashlib

    try:
        td = t.detach()
        if td.is_meta or td.numel() == 0 or getattr(td, "is_sparse", False):
            return None
        cpu = td.contiguous().to("cpu").flatten()
        # 统一按字节流取哈希：避开 numpy 不支持的 dtype（bf16 等）
        buf = cpu.view(torch.uint8).numpy().tobytes()
        return {
            "hash": hashlib.blake2b(buf, digest_size=16).hexdigest(),
            "bytes": len(buf),
        }
    except Exception as e:
        return {"hash_error": f"{type(e).__name__}: {e}"}
