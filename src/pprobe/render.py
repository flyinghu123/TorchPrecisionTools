"""把 :mod:`pprobe.compare` 的 report 渲染成人读文本 / Markdown。

设计目标：拿到报告就能直接回答三个问题——
1. **从哪一步开始不一样**（first divergence）；
2. **是不是环境/配置导致的**（env_diff / config_diff）；
3. **差异最大的是哪些模块，用什么命令继续深挖**（stack id + 提示命令）。
"""

from __future__ import annotations

import os
from typing import Any

from .util import short_repr

KIND_ORDER = ("structural", "info", "special", "stats", "sample", "scalar", "index_mismatch", "stack_diff", "presence")

KIND_LABEL = {
    "structural": "结构",
    "presence": "只在一边存在",
    "info": "基本信息",
    "special": "NaN/Inf",
    "stats": "数值统计",
    "sample": "采样元素",
    "scalar": "标量参数",
    "index_mismatch": "采样下标不一致",
    "stack_diff": "调用堆栈",
}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v in (float("inf"), float("-inf")):
            return "Inf" if v > 0 else "-Inf"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e5 or a < 1e-4:
            return f"{v:.6g}"
        return f"{v:.8g}"
    if isinstance(v, bool):
        return str(v).lower()
    if v is None:
        return "-"
    return str(v)


def _num(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def render(report: dict[str, Any], res_a=None, res_b=None, stacks: bool = True,
           detail: int = 10) -> str:
    """渲染整份对比报告；``detail`` 控制展开多少个发散点的明细。"""
    L: list[str] = []
    totals = report.get("totals") or {}
    a, b = report.get("a") or {}, report.get("b") or {}

    L.append("# pprobe 精度对比报告")
    L.append("")
    L.append(f"- A: `{a.get('path')}`  ranks={a.get('ranks')}")
    L.append(f"- B: `{b.get('path')}`  ranks={b.get('ranks')}")
    opts = report.get("options") or {}
    L.append(f"- 容差: atol={_fmt(opts.get('atol'))}, rtol={_fmt(opts.get('rtol'))}, "
             f"排序={opts.get('sort')}, phase={opts.get('phase') or '全部'}")
    L.append("")
    L.append("## 总览")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("| --- | --- |")
    order = [("events_compared", "成功配对的调用"), ("events_only_a", "仅 A 存在的调用"),
             ("events_only_b", "仅 B 存在的调用"), ("divergent_events", "存在差异的调用"),
             ("structural", "结构性差异"), ("info_diff", "基本信息差异"), ("special", "NaN/Inf 差异"),
             ("scalar_diff", "标量参数差异"), ("stats_diff", "数值统计差异"), ("sample_diff", "采样元素差异"),
             ("index_mismatch", "采样下标不一致"), ("stack_diff", "堆栈差异"),
             ("max_abs_overall", "全局最大绝对差")]
    for k, label in order:
        v = totals.get(k)
        if v is None:
            continue
        L.append(f"| {label} | {_fmt(v)} |")
    L.append("")

    missing = report.get("missing_ranks") or {}
    if missing.get("a_only") or missing.get("b_only"):
        L.append(f"> ⚠️ 未参与对比的 rank：仅 A = {missing.get('a_only')}，仅 B = {missing.get('b_only')}")
        L.append("")

    fd = totals.get("first_divergence")
    L.append("## 最早发散点（根因候选）")
    L.append("")
    if not fd:
        L.append("未发现任何差异 —— 两次运行在采样与统计精度内完全一致。")
    else:
        L.append(f"- rank{fd.get('rank')}  `{fd.get('key')}`  (A seq={_fmt(fd.get('seq_a'))}, B seq={_fmt(fd.get('seq_b'))})")
        L.append(f"- 差异类型: {', '.join(KIND_LABEL.get(k, k) for k in fd.get('kinds') or [])}")
        if fd.get("max_abs") is not None:
            L.append(f"- 该点最大绝对差: {_fmt(fd.get('max_abs'))}")
        if fd.get("stack_a"):
            L.append(f"- A 堆栈 `{fd['stack_a']}` / B 堆栈 `{fd.get('stack_b')}`")
            L.append(f"- 查看调用链: `pprobe stack {fd['stack_a']} --rank {fd.get('rank')} {a.get('path')}`")
        L.append("")
        L.append("> 提示：按计算图顺序，**第一个**发散的模块之后的所有差异都可能是它的传播结果；"
                 "优先看它以及它的直接输入来源。")
    L.append("")

    per_rank = [(k, v.get("first_divergence")) for k, v in sorted((report.get("ranks") or {}).items(),
                key=lambda kv: _rank_sort_key(kv[0]))]
    per_rank = [(k, fd) for k, fd in per_rank if fd]
    if len(per_rank) > 1:
        L.append("### 各 rank 最早发散点")
        L.append("")
        L.append("| rank | 最早发散调用 | A seq | 类型 | 堆栈 | 最大绝对差 |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for _k, fd in per_rank:
            L.append(f"| rank{fd.get('rank')} | `{fd.get('key')}` | {_fmt(fd.get('seq_a'))} | "
                     f"{','.join(fd.get('kinds') or [])} | {fd.get('stack_a') or '-'} | {_fmt(fd.get('max_abs'))} |")
        L.append("")

    env_diff = report.get("env_diff") or {}
    cmdline_diff = report.get("cmdline_diff") or {}
    if cmdline_diff:
        L.append("## 命令行差异")
        L.append("")
        for rk, v in sorted(cmdline_diff.items(), key=lambda kv: _rank_sort_key(kv[0])):
            L.append(f"- rank{rk}  A: `{v.get('a')}`")
            L.append(f"- rank{rk}  B: `{v.get('b')}`")
        L.append("")
    if env_diff:
        L.append("## 环境差异（跨平台/跨配置精度问题的首要原因）")
        L.append("")
        for rk, diffs in sorted(env_diff.items(), key=lambda kv: _rank_sort_key(kv[0])):
            L.append(f"### rank{rk}")
            L.append("")
            L.append("| 项 | A | B |")
            L.append("| --- | --- | --- |")
            for k, v in list(diffs.items())[:80]:
                L.append(f"| `{k}` | {_fmt(v.get('a'))} | {_fmt(v.get('b'))} |")
            if len(diffs) > 80:
                L.append(f"| ... | 共 {len(diffs)} 项 | |")
            L.append("")
    else:
        L.append("## 环境差异")
        L.append("")
        L.append("env.json 完全一致（或其中一方缺失）。")
        L.append("")

    cfg_diff = report.get("config_diff") or {}
    if cfg_diff:
        L.append("## 探针配置差异")
        L.append("")
        L.append("> 采样个数/模式/种子不同会让「采样元素差异」失去可比性，先确认这里。")
        L.append("")
        L.append("| 配置 | A | B |")
        L.append("| --- | --- | --- |")
        for k, v in sorted(cfg_diff.items()):
            L.append(f"| `{k}` | {short_repr(v.get('a'), 60)} | {short_repr(v.get('b'), 60)} |")
        L.append("")

    mods = report.get("module_summary") or []
    if mods:
        L.append("## 按模块汇总")
        L.append("")
        L.append("| 模块 | 差异调用数 | 类型 | 最大绝对差 | 最大相对差 | 最早 seq(A) |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for m in mods[:40]:
            L.append(f"| `{m.get('module')}` | {m.get('n_divergent_events')} | "
                     f"{','.join(m.get('kinds') or [])} | {_fmt(m.get('max_abs'))} | "
                     f"{_fmt(m.get('max_rel'))} | {_fmt(m.get('first_seq'))} |")
        if len(mods) > 40:
            L.append(f"| ... | 共 {len(mods)} 个模块 | | | | |")
        L.append("")

    divs = report.get("divergences") or []
    L.append(f"## 差异明细（展开 {min(detail, len(divs))} / 共 {len(divs)} 条）")
    L.append("")
    if not divs:
        L.append("无差异。")
        L.append("")
    for i, d in enumerate(divs):
        L.extend(_render_divergence(i, d, report, a, stacks=stacks, verbose=i < detail))
        L.append("")

    L.append("---")
    L.append(f"报告由 pprobe 生成；机器可读版本见同名 `.json`（`pprobe compare --json`）。")
    return "\n".join(L)


def _rank_sort_key(k: Any) -> int:
    try:
        return int(k)
    except (TypeError, ValueError):
        return 1 << 30


def _render_divergence(i: int, d: dict[str, Any], report: dict, a: dict, stacks: bool,
                       verbose: bool) -> list[str]:
    L: list[str] = []
    kinds = d.get("kinds") or []
    head = (f"### [{i + 1}] rank{d.get('rank')} `{d.get('phase')}` "
            f"`{d.get('module')}` 第 {d.get('call_index')} 次调用")
    L.append(head)
    meta = [f"kind: {', '.join(KIND_LABEL.get(k, k) for k in kinds)}",
            f"A seq={_fmt(d.get('seq_a'))} step={_fmt(d.get('step_a'))}",
            f"B seq={_fmt(d.get('seq_b'))} step={_fmt(d.get('step_b'))}"]
    if d.get("max_abs") is not None:
        meta.append(f"max_abs={_fmt(d.get('max_abs'))}")
        meta.append(f"max_rel={_fmt(d.get('max_rel'))}")
    L.append("* " + " | ".join(meta))
    stack_line = []
    if d.get("stack_a"):
        stack_line.append(f"A stack=`{d['stack_a']}`")
    if d.get("stack_b"):
        stack_line.append(f"B stack=`{d['stack_b']}`")
    if stack_line:
        L.append("* " + " ".join(stack_line))
        if stacks and d.get("stack_a"):
            L.append(f"* 堆栈查询: `pprobe stack {d['stack_a']} --rank {d.get('rank')} {a.get('path')}`")
    if not verbose:
        return L
    for slot, sd in sorted((d.get("slots") or {}).items()):
        L.extend(_render_slot(slot, sd))
    return L


def _render_slot(slot: str, sd: dict[str, Any]) -> list[str]:
    L: list[str] = []
    kinds = sd.get("kinds") or []
    L.append(f"- 槽位 `{slot}`  [{','.join(kinds)}]")
    changes = sd.get("changes") or {}
    if isinstance(changes, list):  # 兼容旧版报告
        changes = {"presence": changes}
    for c in changes.get("basic") or []:
        L.append(f"    - 基本信息 `{c['field']}`: A={_fmt(c.get('a'))} → B={_fmt(c.get('b'))}")
    for c in changes.get("presence") or []:
        L.append(f"    - 存在性 `{c['field']}`: A={_fmt(c.get('a'))} → B={_fmt(c.get('b'))}")
    for c in changes.get("value") or []:
        L.append(f"    - 输入输出常量: A={_fmt(c.get('a'))} → B={_fmt(c.get('b'))}")
    for c in changes.get("stats") or []:
        extra = ""
        if c.get("abs") is not None:
            extra = f" (abs={_fmt(c['abs'])}, rel={_fmt(c.get('rel'))})"
        if c.get("delta") is not None:
            extra = f" (Δ={_fmt(c['delta'])})"
        L.append(f"    - 统计 `{c['field']}`: A={_fmt(c.get('a'))} → B={_fmt(c.get('b'))}{extra}")
    sm = changes.get("sample")
    if isinstance(sm, dict):
        if sm.get("skipped"):
            L.append(f"    - 采样: 跳过（{sm['skipped']}）")
        else:
            L.append(f"    - 采样: {sm.get('n_diff', 0)}/{sm.get('n', 0)} 个点超容差，"
                     f"max_abs={_fmt(sm.get('max_abs'))} max_rel={_fmt(sm.get('max_rel'))} "
                     f"对齐={sm.get('aligned_by')}")
            if sm.get("index_mismatch"):
                L.append(f"    - 采样下标集合不一致（A 独有 {sm.get('n_only_a')}，B 独有 {sm.get('n_only_b')}），"
                         f"通常因为 SAMPLE_MODE=random 且未固定 SEED")
            for w in (sm.get("worst") or [])[:5]:
                i = w.get("i")
                idx = w.get("idx")
                extra = ""
                if idx is not None:
                    extra = f" idx={idx}"
                    if w.get("idx_b") is not None:
                        extra += f"(B={w['idx_b']})"
                L.append(f"        - #{i}{extra}: "
                         f"A={_fmt(w.get('a'))} → B={_fmt(w.get('b'))} "
                         f"abs={_fmt(w.get('abs'))} rel={_fmt(w.get('rel'))}")
    return L


def render_totals(report: dict[str, Any]) -> str:
    t = report.get("totals") or {}
    parts = [f"{k}={v}" for k, v in sorted(t.items()) if not isinstance(v, dict) and v]
    return " ".join(parts) or "no-difference"


def default_out_path(a_path: str, b_path: str, out_dir: str | None = None) -> str:
    """默认把报告写到 B 结果目录旁边的 compare 文件里，方便和数据处理。"""
    base = f"pprobe_compare_{os.path.basename(a_path.rstrip('/') or 'a')}_vs_{os.path.basename(b_path.rstrip('/') or 'b')}"
    d = out_dir or os.path.dirname(os.path.abspath(b_path)) or "."
    return os.path.join(d, base)
