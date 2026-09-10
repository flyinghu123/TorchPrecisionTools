"""summary 层（需求 4）：基本信息 + 数值统计，含 nan/inf 计数与低精度防护。"""

from __future__ import annotations

import json

import pytest
import torch

from pprobe.summary import basic_info, dtype_limits, full_tensor_hash, numeric_stats


# ----------------------------------------------------------------------
def test_basic_info_has_required_fields():
    t = torch.randn(4, 6, requires_grad=True)
    info = basic_info(t)
    for key in ("shape", "stride", "device", "dtype", "numel", "requires_grad", "contiguous",
                "storage_offset", "nbytes", "is_meta"):
        assert key in info, key
    assert info["shape"] == [4, 6]
    assert info["stride"] == [6, 1]
    assert info["dtype"] == "torch.float32"
    assert info["device"] == "cpu"
    assert info["numel"] == 24 and info["nbytes"] == 96
    assert info["requires_grad"] is True


def test_basic_info_noncontiguous_and_grad_fn():
    t = torch.randn(4, 6).t()
    info = basic_info(t)
    assert info["shape"] == [6, 4] and info["stride"] == [1, 6]
    assert info["contiguous"] is False
    out = torch.randn(4, 4, requires_grad=True) * 2
    assert basic_info(out)["grad_fn"] == "MulBackward0"
    assert basic_info(out)["is_leaf"] is False


# ----------------------------------------------------------------------
def test_float_stats_basics():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    st = numeric_stats(t)
    assert st["max"] == 4.0 and st["min"] == 1.0
    assert st["mean"] == pytest.approx(2.5)
    assert st["var"] == pytest.approx(1.6666666, abs=1e-6)
    assert st["std"] == pytest.approx(1.2909944, abs=1e-6)
    assert st["sum"] == 10.0 and st["checksum"] == 10.0
    assert st["count"] == 4 and st["nan_count"] == 0 and st["inf_count"] == 0
    assert st["absmax"] == 4.0 and st["l2norm"] == pytest.approx(float(t.norm()))


def test_nan_and_inf_counts():
    t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), float("nan"), 2.0])
    st = numeric_stats(t)
    assert st["nan_count"] == 2
    assert st["posinf_count"] == 1 and st["neginf_count"] == 1
    assert st["inf_count"] == 2
    assert st["count"] == 2                                   # 只剩 2 个有限值
    assert st["finite_only"] is True
    assert st["max"] == 2.0 and st["min"] == 1.0              # 最值只统计有限值


def test_all_nonfinite_uses_json_tokens():
    """全 NaN/Inf 时不能写出裸 NaN（非法 JSON），必须是 token 字符串。"""
    st = numeric_stats(torch.tensor([float("nan"), float("inf")]))
    assert st["all_nonfinite"] is True and st["count"] == 0
    assert st["max"] == "NaN" and st["mean"] == "NaN"
    assert json.dumps(st) and json.loads(json.dumps(st))["max"] != 0


def test_zero_count_and_bool_int_complex():
    assert numeric_stats(torch.zeros(5))["zero_count"] == 5
    b = numeric_stats(torch.tensor([True, True, False]))
    assert b["true_count"] == 2 and b["zero_count"] == 1 and b["mean"] == pytest.approx(2 / 3)
    i = numeric_stats(torch.tensor([1, 2, 3, 4]))
    assert i["max"] == 4 and i["min"] == 1 and i["sum"] == 10.0 and i["nan_count"] == 0
    c = numeric_stats(torch.tensor([1 + 2j, 3 - 1j]))
    assert "stats_error" not in c, c
    assert c["nan_count"] == 0 and c["count"] == 2 and "absmax" in c and "absmean" in c


def test_half_precision_does_not_overflow():
    """fp16 求和极易溢出，必须提升到 float64 累加（需求 4 的统计可信度）。"""
    t = torch.full((4096,), 30000.0, dtype=torch.float16)
    st = numeric_stats(t)
    assert st["inf_count"] == 0
    assert st["sum"] == pytest.approx(30000.0 * 4096, rel=1e-6)
    bf = torch.full((1024,), 3.0e38, dtype=torch.bfloat16)
    assert numeric_stats(bf)["inf_count"] == 0
    assert numeric_stats(bf)["sum"] > 3e41                    # fp32 累加会直接爆掉


def test_stats_dtype_switch():
    t = torch.randn(100)
    assert numeric_stats(t, "float32")["mean"] == pytest.approx(numeric_stats(t, "float64")["mean"], abs=1e-6)


def test_empty_and_meta_are_skipped_not_crashed():
    assert numeric_stats(torch.zeros(0)) == {"skipped": "meta-or-empty"}
    assert numeric_stats(torch.empty(3, device="meta")) == {"skipped": "meta-or-empty"}


def test_sparse_stats_uses_values():
    t = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0]).to_sparse()
    st = numeric_stats(t)
    assert "stats_error" not in st, st
    assert st["count"] == 3 and st["max"] == 5.0


def test_stats_error_is_empty():
    st = numeric_stats(torch.randn(8))
    assert "stats_error" not in st, st.get("stats_error")


# ----------------------------------------------------------------------
def test_dtype_limits():
    f = dtype_limits(torch.float16)
    assert f["max"] == 65504.0 and set(f) == {"eps", "tiny", "max", "min"}
    assert dtype_limits(torch.int64) == {"max": 2**63 - 1, "min": -(2**63)}
    assert dtype_limits(torch.bfloat16)["max"] > 3e38
    assert dtype_limits(torch.bool) is None
    assert basic_info(torch.arange(3))["dtype_limits"]["max"] == 2**63 - 1


# ----------------------------------------------------------------------
def test_full_tensor_hash_detects_bit_difference():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 2.0])
    c = torch.tensor([1.0, 2.5])          # 必须能被 fp32 精确区分（2.0000001 在 fp32 里就是 2.0）
    assert full_tensor_hash(a)["hash"] == full_tensor_hash(b)["hash"]
    assert full_tensor_hash(a)["hash"] != full_tensor_hash(c)["hash"]
    # 同一数值不同 dtype 也算不同（跨平台保存精度变化的典型信号）
    assert full_tensor_hash(a)["hash"] != full_tensor_hash(a.to(torch.float64))["hash"]
    assert full_tensor_hash(torch.randn(4, dtype=torch.bfloat16))["hash"]
    assert full_tensor_hash(torch.zeros(0)) is None
    assert full_tensor_hash(torch.empty(2, device="meta")) is None
