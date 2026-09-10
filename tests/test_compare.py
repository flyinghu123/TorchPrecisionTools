"""对比算法（需求 6）：数值/统计/基本信息/标量/堆栈差异的判定与聚合。"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch

from pprobe.compare import (Comparator, Options, compare_basic, compare_samples, compare_stats,
                            compare_values, is_different)
from pprobe.result import Result


def _t(*vals):
    return torch.tensor(vals, dtype=torch.float32)


A0 = torch.arange(24, dtype=torch.float32).reshape(4, 6)
A1 = A0 + 0.5


def _run(build_run, name, tensors, **kw):
    spec = {"tensors": tensors}
    spec.update(kw)
    return build_run(name, [spec])


# ----------------------------------------------------------------------
# 单点判定
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,b,atol,rtol,want",
    [
        (None, None, 0.0, 1e-5, False),
        (1.0, 1.0, 0.0, 1e-5, False),
        (1.0, 1.0 + 1e-6, 0.0, 1e-5, False),
        (1.0, 1.0 + 1e-3, 0.0, 1e-5, True),
        (0.0, 1e-8, 0.0, 1e-5, True),                 # 0 附近只能靠 atol
        (0.0, 1e-8, 1e-6, 1e-5, False),
        ("NaN", "NaN", 0.0, 1e-5, False),             # 都是 NaN → 视为一致
        ("NaN", "Inf", 0.0, 1e-5, True),
        ("NaN", 1.0, 0.0, 1e-5, True),
        (1.0, "Inf", 0.0, 1e-5, True),
        ("Inf", "Inf", 0.0, 1e-5, False),
        ("causal", "causal", 0.0, 1e-5, False),
        ("causal", "packed", 0.0, 1e-5, True),
        (True, False, 0.0, 1e-5, True),
    ],
)
def test_is_different_table(a, b, atol, rtol, want):
    assert is_different(a, b, atol, rtol)[0] is want


def test_is_different_returns_deltas():
    diff, ad, rel = is_different(2.0, 1.0, 0.0, 1e-5)
    assert diff and ad == 1.0 and rel == 0.5
    assert is_different(1.0, 1.0, 0.0, 1e-5)[1:] == (0.0, 0.0)
    # 一方是 inf 时相对差无意义，直接给 inf
    assert is_different(_t(float("nan")).item(), 1.0, 0.0, 1e-5)[1:] == (math.inf, math.inf)


def test_compare_values_basics():
    out = compare_values([1.0, 2.0, 3.0], [1.0, 2.5, 3.0], 0.0, 1e-5, idx=[10, 20, 30])
    assert out["n"] == 3 and out["n_diff"] == 1
    assert out["max_abs"] == 0.5 and out["mean_abs"] == 0.5
    assert out["worst"][0] == {"i": 1, "a": 2.0, "b": 2.5, "abs": 0.5, "rel": 0.2, "idx": 20}
    same = compare_values([1.0], [1.0], 0.0, 1e-5)
    assert same["n_diff"] == 0 and same["worst"] == [] and same["mean_abs"] == 0.0


def test_compare_values_length_mismatch_and_special():
    out = compare_values([1.0, 2.0], [1.0], 0.0, 1e-5)
    assert out["len_a"] == 2 and out["len_b"] == 1 and out["len_mismatch"] is True
    sp = compare_values([1.0, float("nan")], [1.0, "Inf"], 0.0, 1e-5)
    assert sp["n_diff"] == 1 and sp["special_changes"] == 1
    assert sp["max_abs"] == "Inf"                      # 非有限差值以 token 落盘


def test_compare_values_complex_pairs():
    out = compare_values([[1.0, 2.0]], [[1.0, 2.5]], 0.0, 1e-5)
    assert out["n_diff"] == 1 and out["max_abs"] == 0.5
    assert compare_values([[1.0, 2.0]], [[1.0, 2.0]], 0.0, 1e-5)["n_diff"] == 0


def test_compare_samples_same_index():
    sa = {"idx": [0, 1, 2], "vals": [1.0, 2.0, 3.0], "n": 3}
    sb = {"idx": [0, 1, 2], "vals": [1.0, 2.0, 9.0], "n": 3}
    out = compare_samples(sa, sb, 0.0, 1e-5)
    assert out["aligned_by"] == "same-index" and out["n_diff"] == 1
    assert out["worst"][0]["idx"] == 2


def test_compare_samples_index_intersect():
    sa = {"idx": [0, 5, 9], "vals": [0.0, 5.0, 9.0]}
    sb = {"idx": [5, 9, 13], "vals": [5.0, 9.0, 13.0]}
    out = compare_samples(sa, sb, 0.0, 1e-5)
    assert out["aligned_by"] == "index-intersect"
    assert out["index_mismatch"] is True
    assert out["n_only_a"] == 1 and out["n_only_b"] == 1
    assert out["n"] == 2 and out["n_diff"] == 0
    assert out["idx_a"] == [5, 9]
    # 下标不同、值不同时要能报出 idx_b，方便直接对比元素位置
    out2 = compare_samples({"idx": [0, 1], "vals": [1.0, 2.0]}, {"idx": [1, 0], "vals": [1.0, 2.0]}, 0.0, 1e-5)
    assert out2["n_diff"] == 2
    assert {w["idx"]: w.get("idx_b") for w in out2["worst"]} == {0: 1, 1: 0}


def test_compare_samples_degenerated():
    assert compare_samples({}, {"idx": [0], "vals": [1.0]}, 0.0, 1e-5) is None
    assert compare_samples({"idx": [], "vals": []}, {"idx": [], "vals": []}, 0.0, 1e-5) is None
    out = compare_samples({"unavailable": "meta"}, {"idx": [0], "vals": [1.0]}, 0.0, 1e-5)
    assert out == {"skipped": "unavailable:meta/None"}


def test_compare_stats_kinds():
    diffs = compare_stats(
        {"count": 10, "mean": 1.0, "max": 3.0, "nan_count": 0, "finite_only": True, "std": 1.0},
        {"count": 12, "mean": 1.0000001, "max": "Inf", "nan_count": 2, "finite_only": False, "std": None},
        0.0, 1e-5,
    )
    by = {d["field"]: d for d in diffs}
    assert by["count"]["kind"] == "count" and by["count"]["delta"] == 2
    assert by["nan_count"]["kind"] == "count" and by["nan_count"]["delta"] == 2
    assert by["max"]["kind"] == "numeric" and by["max"]["rel"] == "Inf"
    assert by["std"]["kind"] == "missing"
    assert by["finite_only"]["kind"] == "flag"
    assert "mean" not in by                            # 在 rtol 之内


def test_compare_basic_skips_volatile():
    diffs = compare_basic(
        {"shape": [4, 6], "dtype": "torch.float32", "data_ptr": 111, "version": 1, "storage_offset": 0},
        {"shape": [4, 5], "dtype": "torch.bfloat16", "data_ptr": 222, "version": 9, "storage_offset": 6},
    )
    fields = {d["field"] for d in diffs}
    assert fields == {"shape", "dtype"}
    assert not compare_basic({"device": "cpu"}, {"device": "cpu"})


# ----------------------------------------------------------------------
@pytest.fixture
def compare(build_run):
    """跑一次对比，返回 (report, markdown 之外的纯结构)。"""
    from pprobe.render import render

    def _cmp(path_a, path_b, **opt_kw):
        stacks = opt_kw.pop("stacks", True)
        detail = opt_kw.pop("detail", 10)
        opts = Options(**opt_kw)
        rep = Comparator(Result(path_a), Result(path_b), opts).run()
        rep["__md__"] = render(rep, stacks=stacks, detail=detail)
        return rep

    return _cmp


def _spec(tensors, **kw):
    d = {"tensors": tensors}
    d.update(kw)
    return d


def test_identical_runs_have_no_difference(build_run, compare):
    a = build_run("same-a", [_spec({"input[0]": A0, "output": A1})])
    b = build_run("same-b", [_spec({"input[0]": A0, "output": A1})])
    rep = compare(a, b)
    assert rep["totals"]["divergent_events"] == 0
    assert rep["divergences"] == []
    assert rep["totals"]["first_divergence"] is None
    assert rep["ranks"]["0"]["events_compared"] == 1
    assert "未发现任何差异" in rep["__md__"]
    assert rep["config_diff"] == {}


def test_numeric_difference_detected(build_run, compare):
    a = build_run("num-a", [_spec({"output": A0})])
    b = build_run("num-b", [_spec({"output": A0 + 1.0})])
    rep = compare(a, b)
    t = rep["totals"]
    assert t["sample_diff"] == 1 and t["stats_diff"] == 1
    d = rep["divergences"][0]
    assert set(d["kinds"]) == {"sample", "stats"}
    assert d["max_abs"] == 1.0 and d["slots"]["output"]["changes"]["sample"]["n_diff"] == 8
    assert t["max_abs_overall"] == 1.0
    fd = t["first_divergence"]
    assert fd["key"] == "forward:net.linear#0" and fd["seq_a"] == 1
    assert "采样:" in rep["__md__"] and "- 槽位 `output`  [sample,stats]" in rep["__md__"]


def test_tolerance_hides_small_noise(build_run, compare):
    a = build_run("tol-a", [_spec({"output": A0})])
    b = build_run("tol-b", [_spec({"output": A0 * (1 + 1e-6)})])
    assert compare(a, b, rtol=1e-5)["totals"]["divergent_events"] == 0
    assert compare(a, b, rtol=0.0)["totals"]["divergent_events"] == 1


def test_info_and_special_differences(build_run, compare):
    a = build_run("kind-a", [_spec({"output": A0})])
    b = build_run("kind-b", [_spec({"output": A0.reshape(6, 4)})])
    rep = compare(a, b)
    d = rep["divergences"][0]
    assert "info" in d["kinds"]
    assert {"field": "shape", "a": [4, 6], "b": [6, 4]} in d["slots"]["output"]["changes"]["basic"]
    assert rep["totals"]["info_diff"] == 1

    c = build_run("kind-c", [_spec({"output": A0})])
    e = build_run("kind-e", [_spec({"output": torch.where(A0 > 10, _t(float("nan")), A0)})])
    rep2 = compare(c, e)
    assert "special" in rep2["divergences"][0]["kinds"]
    assert rep2["totals"]["special"] == 1
    assert "NaN/Inf" in rep2["__md__"]


def test_scalar_and_presence_differences(build_run, compare):
    a = build_run("sc-a", [_spec({"input[0]": A0, "kwargs.eps": 1e-5})])
    b = build_run("sc-b", [_spec({"input[0]": A0, "kwargs.eps": 1e-2})])
    rep = compare(a, b)
    d = rep["divergences"][0]
    assert d["kinds"] == ["scalar"] and "kwargs.eps" in d["slots"]
    assert rep["totals"]["scalar_diff"] == 1
    assert "输入输出常量" in rep["__md__"]

    c = build_run("sc-c", [_spec({"input[0]": A0, "output": A1})])
    e = build_run("sc-e", [_spec({"input[0]": A0})])
    rep2 = compare(c, e)
    assert "presence" in rep2["divergences"][0]["kinds"]
    assert rep2["totals"]["structural"] == 1


def test_module_cls_difference_is_info(build_run, compare):
    a = build_run("cls-a", [_spec({"input[0]": A0}, cls="Linear")])
    b = build_run("cls-b", [_spec({"input[0]": A0}, cls="LoRALinear")])
    rep = compare(a, b)
    assert "info" in rep["divergences"][0]["kinds"]
    assert rep["divergences"][0]["slots"]["__module_cls__"]["changes"]["basic"][0]["b"] == "LoRALinear"


def test_stack_id_difference(build_run, compare):
    a = build_run("st-a", [_spec({"input[0]": A0}, stack_id="s1")])
    b = build_run("st-b", [_spec({"input[0]": A0}, stack_id="s2")])
    rep = compare(a, b)
    assert rep["divergences"][0]["kinds"] == ["stack_diff"]
    assert rep["totals"]["stack_diff"] == 1
    md = rep["__md__"]
    assert "调用堆栈" in md and "pprobe stack s1 --rank 0" in md
    assert "堆栈查询:" in md
    assert "堆栈查询:" not in compare(a, b, stacks=False)["__md__"]


def test_index_mismatch_between_sample_modes(build_run, compare):
    big = torch.arange(100, dtype=torch.float32)
    a = build_run("im-a", [_spec({"output": big})], sample_mode="uniform", sample_n=8)
    b = build_run("im-b", [_spec({"output": big})], sample_mode="head", sample_n=8)
    rep = compare(a, b)
    d = rep["divergences"][0]
    assert "index_mismatch" in d["kinds"]
    assert d["slots"]["output"]["changes"]["sample"]["aligned_by"] == "index-intersect"
    assert rep["totals"]["index_mismatch"] == 1
    assert "通常因为 SAMPLE_MODE=random" in rep["__md__"]
    assert "sample_n" in rep["config_diff"] or "sample_mode" in rep["config_diff"]


def test_unmatched_events_counted(build_run, compare):
    a = build_run("mu-a", [_spec({"input[0]": A0}, module="net.x"), _spec({"input[0]": A0}, module="net.y")])
    b = build_run("mu-b", [_spec({"input[0]": A0}, module="net.x"), _spec({"input[0]": A0}, module="net.z")])
    rep = compare(a, b)
    t = rep["totals"]
    assert t["events_only_a"] == 1 and t["events_only_b"] == 1
    assert t["events_compared"] == 1
    assert rep["ranks"]["0"]["only_a_sample"] == ["forward:net.y#1"]
    assert rep["ranks"]["0"]["only_b_sample"] == ["forward:net.z#1"]


def test_phase_include_exclude_and_ignore_slots(build_run, compare):
    a = build_run("f-a", [
        _spec({"output": A0}, module="encoder.layer1"),
        _spec({"output": A0}, module="decoder.layer1"),
        _spec({"output": A0}, module="encoder.layer1", phase="backward"),
    ])
    b = build_run("f-b", [
        _spec({"output": A0 + 1}, module="encoder.layer1"),
        _spec({"output": A0 + 1}, module="decoder.layer1"),
        _spec({"output": A0 + 1}, module="encoder.layer1", phase="backward"),
    ])
    assert {d["key"][0] for d in compare(a, b, phase="forward")["divergences"]} == {"forward"}
    assert {d["module"] for d in compare(a, b, include=r"^encoder\.")["divergences"]} == {"encoder.layer1"}
    assert {d["module"] for d in compare(a, b, exclude=r"^encoder\.")["divergences"]} == {"decoder.layer1"}
    rep = compare(a, b, ignore_slots=["output"])
    assert rep["divergences"] == [] and rep["totals"]["events_compared"] == 3


def test_top_and_sort(build_run, compare):
    specs_a = [_spec({"output": _t(float(i + 1))}, module=f"net.m{i}") for i in range(5)]
    specs_b = [_spec({"output": _t(float((i + 1) ** 2) + 0.5)}, module=f"net.m{i}") for i in range(5)]
    a = build_run("top-a", specs_a)
    b = build_run("top-b", specs_b)
    rep = compare(a, b, top=2)
    assert len(rep["divergences"]) == 2
    assert rep["totals"]["divergent_events"] == 5 and rep["totals"]["divergences_kept"] == 2
    assert [d["seq_a"] for d in rep["divergences"]] == [1, 2]        # 默认按 seq（最早优先）
    rep2 = compare(a, b, top=2, sort="max_abs")
    assert [d["module"] for d in rep2["divergences"]] == ["net.m4", "net.m3"]
    rep3 = compare(a, b, top=0)
    assert len(rep3["divergences"]) == 5


def test_module_summary_aggregation(build_run, compare):
    a = build_run("ms-a", [
        _spec({"output": A0}, module="net.a", call_index=0),
        _spec({"output": A0}, module="net.a", call_index=1),
        _spec({"output": A0}, module="net.b", call_index=0),
    ])
    b = build_run("ms-b", [
        _spec({"output": A0 + 1}, module="net.a", call_index=0),
        _spec({"output": A0 + 3}, module="net.a", call_index=1),
        _spec({"output": A0 + 2}, module="net.b", call_index=0),
    ])
    rows = compare(a, b)["module_summary"]
    assert [r["module"] for r in rows] == ["net.a", "net.b"]
    assert rows[0]["n_divergent_events"] == 2 and rows[0]["max_abs"] == 3.0
    assert rows[0]["first_seq"] == 1 and rows[0]["kinds"] == ["sample", "stats"]
    assert "按模块汇总" in compare(a, b)["__md__"]


# ----------------------------------------------------------------------
def test_multi_rank_pairs_and_missing(build_run, compare):
    a0 = build_run("mr-a", [_spec({"output": A0})], rank=0)
    build_run("mr-a", [_spec({"output": A0})], rank=1)
    build_run("mr-b", [_spec({"output": A0 + 1})], rank=0)
    build_run("mr-b", [_spec({"output": A0 + 1})], rank=2)
    rep = compare(a0, os.path.join(os.path.dirname(a0), "mr-b"))
    assert rep["a"]["ranks"] == [0, 1] and rep["b"]["ranks"] == [0, 2]
    assert rep["missing_ranks"] == {"a_only": [1], "b_only": [2]}
    assert list(rep["ranks"]) == ["0"]
    assert rep["totals"]["sample_diff"] == 1            # 只对比了 rank0

    # 需求 8：也可以手工指定 rank 配对（例如两边数据并行度不同）
    a1 = Result(os.path.join(os.path.dirname(a0), "mr-a"))
    b1 = Result(os.path.join(os.path.dirname(a0), "mr-b"))
    rep2 = Comparator(a1, b1, Options(ranks=[(1, 0)])).run()
    assert rep2["totals"]["sample_diff"] == 1
    assert rep2["ranks"]["1"]["rank_b"] == 0


def test_env_and_cmdline_diff(build_run, compare):
    a = build_run("env-a", [_spec({"output": A0})])
    b = build_run("env-b", [_spec({"output": A0})])
    path_b = os.path.join(b, "rank0", "env.json")
    env = json.load(open(path_b, encoding="utf-8"))
    env["torch"] = "2.99.0"
    env["device"] = "cuda(0 gpu): FAKE-GPU/sm90/1MiB"
    env["pid"] = 999999
    json.dump(env, open(path_b, "w", encoding="utf-8"))
    rep = compare(a, b)
    diffs = rep["env_diff"]["0"]
    assert diffs["torch"]["b"] == "2.99.0"
    assert "pid" not in diffs                           # 每次必然不同，忽略
    assert diffs["device"]["a"] != diffs["device"]["b"]
    md = rep["__md__"]
    assert "环境差异" in md and "2.99.0" in md
    assert compare(a, b, compare_env=False)["env_diff"] == {}


def test_missing_rank_dir_reported(build_run, compare, tmp_path):
    a = build_run("gap-a", [_spec({"output": A0})])
    b = build_run("gap-b", [_spec({"output": A0})])
    os.remove(os.path.join(b, "rank0", "events.jsonl"))
    rep = compare(a, b)
    assert rep["totals"]["events_compared"] == 0
    assert os.path.isdir(b)
