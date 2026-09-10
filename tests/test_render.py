"""报告渲染（需求 6 的输出）：Markdown 结构、明细展开条数、默认输出路径。"""

from __future__ import annotations

import os

import torch

from pprobe.compare import Comparator, Options
from pprobe.render import (KIND_LABEL, _fmt, default_out_path, render, render_totals)
from pprobe.result import Result

T0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)


def _spec(tensors, **kw):
    d = {"tensors": tensors}
    d.update(kw)
    return d


def _report(build_run, specs_a, specs_b, **opt_kw):
    a = build_run("rd-a", specs_a)
    b = build_run("rd-b", specs_b)
    rep = Comparator(Result(a), Result(b), Options(**opt_kw)).run()
    return rep, a, b


# ----------------------------------------------------------------------
def test_fmt_rendering_rules():
    assert _fmt(None) == "-"
    assert _fmt(True) == "true" and _fmt(False) == "false"
    assert _fmt(0.0) == "0"
    assert _fmt(float("nan")) == "NaN" and _fmt(float("-inf")) == "-Inf"
    assert _fmt("NaN") == "NaN"                         # token 原样显示
    assert _fmt(1e6) == "1e+06" and _fmt(1e-6) == "1e-06"
    assert _fmt(1.5) == "1.5" and _fmt(3) == "3"
    assert _fmt([1, 2]) == "[1, 2]"


def test_render_totals_text():
    assert render_totals({"totals": {}}) == "no-difference"
    assert render_totals({}) == "no-difference"
    txt = render_totals({"totals": {"sample_diff": 3, "first_divergence": {"a": 1}, "info_diff": 0}})
    assert txt == "sample_diff=3"                       # 0 与 dict 字段被过滤


def test_default_out_path():
    p = default_out_path("/data/base_run", "/data/cand_run")
    assert os.path.dirname(p) == "/data"
    assert os.path.basename(p) == "pprobe_compare_base_run_vs_cand_run.md".replace(".md", "")
    assert default_out_path("/x/a/", "/y/b/").endswith("pprobe_compare_a_vs_b")
    assert default_out_path("a", "b", out_dir="/tmp/out").startswith("/tmp/out/pprobe_compare_")
    assert default_out_path("/", "/").endswith("pprobe_compare_a_vs_b")


# ----------------------------------------------------------------------
def test_render_sections_and_order(build_run):
    rep, a, _b = _report(build_run, [_spec({"output": T0}, stack_id="s1")],
                         [_spec({"output": T0 + 1.0}, stack_id="s1")])
    md = render(rep)
    heads = [ln for ln in md.splitlines() if ln.startswith("#")]
    assert heads[0] == "# pprobe 精度对比报告"
    for want in ("## 总览", "## 最早发散点（根因候选）", "## 环境差异", "## 按模块汇总", "## 差异明细"):
        assert any(ln.startswith(want) for ln in heads), want
    assert f"- A: `{a}`" in md
    assert "容差: atol=0, rtol=1e-05" in md
    assert "存在差异的调用 | 1" in md
    assert "查看调用链: `pprobe stack s1 --rank 0" in md
    assert md.rstrip().endswith("机器可读版本见同名 `.json`（`pprobe compare --json`）。")


def test_render_no_difference(build_run):
    rep, _a, _b = _report(build_run, [_spec({"output": T0})], [_spec({"output": T0})])
    md = render(rep)
    assert "未发现任何差异 —— 两次运行在采样与统计精度内完全一致。" in md
    assert "无差异。" in md
    assert render_totals(rep) == "events_compared=1"   # 只有配对的调用数，没有任何差异计数


def test_render_detail_controls_expansion(build_run):
    specs_a = [_spec({"output": T0}, module=f"m{i}") for i in range(4)]
    specs_b = [_spec({"output": T0 + 1.0}, module=f"m{i}") for i in range(4)]
    rep, _a, _b = _report(build_run, specs_a, specs_b, top=4)
    md = render(rep, detail=2)
    assert "## 差异明细（展开 2 / 共 4 条）" in md
    assert md.count("- 槽位 `output`") == 2              # 只有前 2 条展开逐元素明细
    assert md.count("### [") == 4                       # 但 4 条都有标题与 meta
    assert "### [4]" in md
    assert render(rep, detail=0).count("- 槽位") == 0


def test_render_kind_labels_all_used():
    assert set(KIND_LABEL) >= {"presence", "info", "special", "stats", "sample", "scalar",
                               "index_mismatch", "stack_diff"}


def test_render_per_rank_table_and_missing_ranks(build_run):
    a = build_run("rr-a", [_spec({"output": T0})], rank=0)
    build_run("rr-a", [_spec({"output": T0}, module="other")], rank=1)
    b = build_run("rr-b", [_spec({"output": T0 + 1})], rank=0)
    build_run("rr-b", [_spec({"output": T0 + 2}, module="other")], rank=1)
    rep = Comparator(Result(a), Result(b), Options()).run()
    md = render(rep)
    assert "### 各 rank 最早发散点" in md
    assert "| rank | 最早发散调用 | A seq | 类型 | 堆栈 | 最大绝对差 |" in md
    assert "other#0" in md and "forward:net.linear#0" in md

    # 只有一边有的 rank → 顶部警告
    c = build_run("rr-c", [_spec({"output": T0})], rank=2)
    rep2 = Comparator(Result(a), Result(c), Options()).run()
    md2 = render(rep2)
    assert "未参与对比的 rank" in md2 and "仅 A = [0, 1]" in md2 and "仅 B = [2]" in md2


def test_render_legacy_report_with_list_changes():
    """旧版报告里 changes 是 list，渲染要能兼容（用户拿历史数据复跑）。"""
    rep = {
        "a": {"path": "/x/a", "ranks": [0]}, "b": {"path": "/x/b", "ranks": [0]},
        "options": {"atol": 0.0, "rtol": 1e-5, "sort": "seq", "phase": None},
        "ranks": {}, "totals": {"divergent_events": 1, "events_compared": 1},
        "divergences": [{
            "rank": 0, "phase": "forward", "module": "net", "call_index": 0, "seq_a": 1, "seq_b": 1,
            "step_a": 0, "step_b": 0, "kinds": ["presence"], "max_abs": None, "max_rel": None,
            "slots": {"output": {"kinds": ["presence"], "changes": [
                {"field": "presence", "a": "present", "b": "absent"}]}},
        }],
    }
    md = render(rep)
    assert "存在性 `presence`: A=present → B=absent" in md
    assert "kind: 只在一边存在" in md                     # presence → 中文标签映射


def test_render_sample_block(build_run):
    a = build_run("smp-a", [_spec({"output": torch.tensor([1.0, 2.0, 3.0, 4.0])})], sample_n=4)
    b = build_run("smp-b", [_spec({"output": torch.tensor([1.0, 2.0, 3.0, 9.0])})], sample_n=4)
    md = render(Comparator(Result(a), Result(b), Options()).run())
    assert "1/4 个点超容差" in md
    assert "idx=3" in md and "B=9" in md
    assert "max_abs=5" in md
