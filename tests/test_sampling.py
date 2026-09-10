"""采样层（需求 3）：均匀/随机/头部/关闭、seed 可控复现、位模式、异常张量兜底。"""

from __future__ import annotations

import pytest
import torch

from pprobe.sampling import (TensorSampler, _unravel, describe_unsupported, grid_sample_indices,
                             sample_tensor)


# ----------------------------------------------------------------------
def test_uniform_indices_are_end_aligned_and_deduped():
    s = TensorSampler("uniform", 5)
    idx = s.uniform_indices(100, 5)
    assert idx[0] == 0 and idx[-1] == 99 and len(idx) == 5
    assert idx == [0, 25, 50, 74, 99]                    # 等间隔 + 两端对齐
    assert s.uniform_indices(100, 1) == [0]
    assert s.uniform_indices(4, 10) == [0, 1, 2, 3]      # k > numel 时退化为全量
    assert s.uniform_indices(1, 5) == [0]
    assert s.uniform_indices(0, 5) == []
    # 同一下标序列在两次运行里必须完全一致（对比命令依赖这点）
    assert TensorSampler("uniform", 5).uniform_indices(999, 5) == s.uniform_indices(999, 5)


def test_default_sampler_is_uniform_50():
    """需求 3 的默认值：每个 tensor 均匀取 50 个元素。"""
    s = TensorSampler()
    assert s.mode == "uniform" and s.n == 50 and s.seed_is_user is False
    out = sample_tensor(torch.randn(1000), s)
    assert out["n"] == 50 and out["idx"][0] == 0 and out["idx"][-1] == 999
    assert len(set(out["idx"])) == 50


def test_count_smaller_than_n():
    s = TensorSampler("uniform", 50)
    out = sample_tensor(torch.randn(3), s)
    assert out["n"] == 3 and out["idx"] == [0, 1, 2]


def test_head_and_off_and_zero_n():
    assert TensorSampler("head", 4).indices([10]) == [0, 1, 2, 3]
    assert TensorSampler("off", 4).indices([10]) == []
    assert TensorSampler("uniform", 0).indices([10]) == []
    assert sample_tensor(torch.randn(10), TensorSampler("off", 4)) == {"n": 0}


def test_random_seed_reproducible_and_unseeded_not():
    """需求 3：seed 由环境变量控制；不设则随机，设了必须可复现。"""
    a = TensorSampler("random", 10, seed=42).indices([1000])
    b = TensorSampler("random", 10, seed=42).indices([1000])
    assert a == b and len(a) == 10 and a == sorted(a)
    c = TensorSampler("random", 10, seed=7).indices([1000])
    assert c != a
    unseeded = {tuple(TensorSampler("random", 10).indices([1000])) for _ in range(8)}
    assert len(unseeded) > 1                            # 不给 seed 就是随机
    assert TensorSampler("random", 10).seed_is_user is False
    assert TensorSampler("random", 10, seed=42).seed_is_user is True


def test_random_full_when_k_ge_numel():
    assert TensorSampler("random", 50, seed=1).indices([6]) == [0, 1, 2, 3, 4, 5]


def test_grid_layout_covers_each_dim():
    s = TensorSampler("uniform", 16, layout="grid")
    idx = s.indices([4, 8])
    rows = {i // 8 for i in idx}
    cols = {i % 8 for i in idx}
    assert len(rows) > 1 and len(cols) > 1 and idx == sorted(set(idx))
    flat_only = TensorSampler("uniform", 16, layout="flat").indices([4, 8])
    assert len(flat_only) <= 16


def test_grid_indices_helper():
    assert grid_sample_indices([3, 3], 0) == []
    assert grid_sample_indices([0, 5], 4) == []
    assert grid_sample_indices([2, 2], 4) == [0, 1, 2, 3]


def test_unravel():
    assert _unravel(5, [2, 3]) == [1, 2]
    assert _unravel(0, [4, 4]) == [0, 0]
    assert _unravel(11, [3, 4]) == [2, 3]


# ----------------------------------------------------------------------
def test_values_and_index_nd_are_consistent():
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    out = sample_tensor(t, TensorSampler("uniform", 6))
    assert out["idx"] == [0, 5, 9, 14, 18, 23]
    assert out["vals"] == [0.0, 5.0, 9.0, 14.0, 18.0, 23.0]
    assert len(out["vals"]) == out["n"] == len(out["idx"])
    assert out["idx_nd"] == [_unravel(i, [4, 6]) for i in out["idx"]]


def test_nan_inf_survive_as_tokens():
    t = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])
    out = sample_tensor(t, TensorSampler("head", 4))
    assert out["vals"] == ["NaN", "Inf", "-Inf", 1.0]


def test_integer_and_bool_and_complex():
    assert sample_tensor(torch.arange(4), TensorSampler("head", 4))["vals"] == [0, 1, 2, 3]
    assert sample_tensor(torch.tensor([True, False]), TensorSampler("head", 2))["vals"] == [1, 0]
    cx = sample_tensor(torch.tensor([1 + 2j]), TensorSampler("head", 1))
    assert cx["vals"] == [[1.0, 2.0]]


def test_half_precision_is_upcast_for_display():
    t = torch.arange(8, dtype=torch.float16) / 8
    out = sample_tensor(t, TensorSampler("uniform", 8))
    assert all(isinstance(v, float) for v in out["vals"])
    assert out["vals"][-1] == pytest.approx(0.875, abs=1e-3)


def test_raw_bits_for_float32():
    import struct

    t = torch.tensor([1.0, -2.5])
    expect = [struct.unpack("<i", struct.pack("<f", float(v)))[0] for v in t]
    out = sample_tensor(t, TensorSampler("head", 2), want_bits=True)
    assert out["bits"] == expect
    # 非浮点没有位模式视图，安全返回 None
    assert sample_tensor(torch.arange(2), TensorSampler("head", 2), want_bits=True).get("bits") is None


def test_bf16_bits_and_float64_bits():
    assert len(sample_tensor(torch.randn(4, dtype=torch.bfloat16), TensorSampler("head", 4),
                            want_bits=True)["bits"]) == 4
    assert len(sample_tensor(torch.randn(4, dtype=torch.float64), TensorSampler("head", 4),
                            want_bits=True)["bits"]) == 4


# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "maker,reason",
    [
        (lambda: torch.empty(4, device="meta"), "meta"),
        (lambda: torch.zeros(3).to_sparse(), "sparse"),
        (lambda: torch.zeros(0), "empty"),
    ],
)
def test_unsupported_tensors(maker, reason):
    t = maker()
    assert describe_unsupported(t) == reason
    assert sample_tensor(t, TensorSampler("uniform", 5)) == {"unavailable": reason}


def test_quantized_described():
    try:
        q = torch.quantize_per_tensor(torch.rand(4, 4), 0.1, 0, torch.qint8)
    except Exception:  # pragma: no cover - 精简版 torch 可能没编进量化内核
        pytest.skip("当前 torch 不支持量化张量")
    assert describe_unsupported(q) == "quantized"


def test_cuda_tensor_sampled_if_available():
    if not torch.cuda.is_available():
        pytest.skip("无可用 GPU")
    try:
        t = torch.arange(16, dtype=torch.float32, device="cuda")
    except Exception as e:  # 驱动能枚举但 kernel 不可用（架构不匹配）等情况
        pytest.skip(f"GPU 不可用于计算: {type(e).__name__}")
    out = sample_tensor(t, TensorSampler("uniform", 4))
    assert out["n"] == 4 and out["vals"][-1] == 15.0


def test_failure_is_recorded_not_raised(monkeypatch):
    s = TensorSampler("uniform", 4)

    def boom(*a, **kw):
        raise RuntimeError("nope")

    monkeypatch.setattr("pprobe.sampling._to_display_dtype", boom)
    assert sample_tensor(torch.randn(8), s) == {"unavailable": "error:RuntimeError"}
    assert s.failures["sample:RuntimeError"] == 1
