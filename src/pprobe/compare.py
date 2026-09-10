"""两次运行 / 两个平台的结果目录对比。

对齐策略（精度调试最关键的一点）：**不使用 seq 对齐**，而是用
``(rank, phase, module_path, call_index)`` 作为 key —— 因为一次运行里多插一层
日志/一个模块就会让 seq 全部错位，而「某个模块的第 N 次调用」在两次运行间是稳定的。

差异分级：
* ``presence``   结构性差异（只在一边出现 / 槽位缺失 / kind 不同）
* ``info``       tensor 基本信息差异（shape/stride/dtype/device/grad_fn...）
* ``special``    特殊值变化（有限值 ↔ NaN/Inf，或 NaN 个数变化）
* ``stats``      数值 summary 差异（超容差）
* ``sample``     采样元素逐点差异（超容差）
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterator

from .result import RankResult, Result, tensor_slots
from .util import decode_number, is_special, short_repr

VOLATILE_BASIC = ("data_ptr", "version", "storage_offset")

#: 差异类型 → 计数器字段（统一命名，否则 totals 永远为 0）
KIND_COUNTER = {
    "presence": "structural",
    "index_mismatch": "index_mismatch",
    "info": "info_diff",
    "special": "special",
    "stats": "stats_diff",
    "sample": "sample_diff",
    "scalar": "scalar_diff",
    "stack_diff": "stack_diff",
}

#: env.json 里每次运行必然不同、对归因无意义的字段
VOLATILE_ENV = ("pid", "ppid")


class Options:
    def __init__(self, atol: float = 0.0, rtol: float = 1e-5, top: int = 30, only_diff: bool = True,
                 ranks: list[tuple[int, int]] | None = None, phase: str | None = None,
                 include: str | None = None, exclude: str | None = None,
                 sort: str = "seq", ignore_slots: tuple[str, ...] = (), compare_env: bool = True):
        self.atol = atol
        self.rtol = rtol
        self.top = top
        self.only_diff = only_diff
        self.ranks = ranks
        self.phase = phase
        self.include = include
        self.exclude = exclude
        self.sort = sort
        self.ignore_slots = tuple(ignore_slots)
        self.compare_env = compare_env


# ----------------------------------------------------------------------
def is_different(a: Any, b: Any, atol: float, rtol: float) -> tuple[bool, float, float]:
    """返回 (是否不同, 绝对差, 相对差)。NaN/Inf 用 token 表达，需特判。"""
    if a is None and b is None:
        return False, 0.0, 0.0
    if isinstance(a, str) and not isinstance(b, str):
        return True, math.inf, math.inf
    if isinstance(b, str) and not isinstance(a, str):
        return True, math.inf, math.inf
    if not isinstance(a, (int, float)) and not isinstance(b, (int, float)):
        return (a != b), 0.0, 0.0
    try:
        fa, fb = decode_number(a), decode_number(b)
    except (TypeError, ValueError):
        return a != b, 0.0, 0.0
    na, nb = math.isnan(fa), math.isnan(fb)
    if na and nb:
        return False, 0.0, 0.0
    if na != nb:
        return True, math.inf, math.inf
    if math.isinf(fa) or math.isinf(fb):
        return (fa != fb), (0.0 if fa == fb else math.inf), math.inf
    absd = abs(fa - fb)
    denom = max(abs(fb), abs(fa))
    rel = absd / denom if denom > 0 else (0.0 if absd == 0 else math.inf)
    diff = absd > atol + rtol * abs(fb)
    return diff, absd, rel


def compare_values(va: list, vb: list, atol: float, rtol: float,
                   idx: list | None = None) -> dict[str, Any]:
    """按下标一一对应的等长列表对比。"""
    n = min(len(va), len(vb))
    stats = _diff_loop(va[:n], vb[:n], atol, rtol, idx=idx[:n] if idx else None)
    stats["len_a"] = len(va)
    stats["len_b"] = len(vb)
    if len(va) != len(vb):
        stats["len_mismatch"] = True
    return stats


def _diff_loop(a: list, b: list, atol: float, rtol: float,
               idx: list | None = None, ib_idx: dict[int, int] | None = None) -> dict[str, Any]:
    max_abs = 0.0
    max_rel = 0.0
    worst: list[dict[str, Any]] = []
    n_diff = 0
    special_changes = 0
    sum_abs = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, list) or isinstance(y, list):  # complex: [re, im]
            for j in range(2):
                xs = x[j] if isinstance(x, list) and j < len(x) else None
                ys = y[j] if isinstance(y, list) and j < len(y) else None
                d, ad, rl = is_different(xs, ys, atol, rtol)
                if d:
                    n_diff += 1
                    sum_abs += ad if math.isfinite(ad) else 0.0
                    if ad > max_abs:
                        max_abs = ad
                    if rl > max_rel:
                        max_rel = rl
            continue
        d, ad, rl = is_different(x, y, atol, rtol)
        if d:
            n_diff += 1
            if is_special(x) or is_special(y):
                special_changes += 1
            sum_abs += ad if math.isfinite(ad) else 0.0
            if ad > max_abs:
                max_abs = ad
            if rl > max_rel:
                max_rel = rl
            if len(worst) < 5:
                worst.append(_worst(i, x, y, ad, rl, idx, ib_idx))
    n = min(len(a), len(b))
    return {
        "n": n,
        "n_diff": n_diff,
        "max_abs": _num(max_abs),
        "max_rel": _num(max_rel),
        "mean_abs": _num(sum_abs / n_diff) if n_diff else 0.0,
        "special_changes": special_changes,
        "worst": worst,
    }


def _worst(i: int, x: Any, y: Any, ad: float, rl: float, idx: list | None,
           ib_idx: dict[int, int] | None) -> dict[str, Any]:
    """记录差异点时带上 tensor 内的原始扁平下标，方便直接定位元素。"""
    row: dict[str, Any] = {"i": i, "a": x, "b": y, "abs": _num(ad), "rel": _num(rl)}
    if idx is not None and i < len(idx):
        row["idx"] = idx[i]
        if ib_idx:
            row["idx_b"] = ib_idx.get(idx[i])
    return row


def _num(v: float) -> float | str:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if math.isinf(v):
        return "Inf"
    if math.isnan(v):
        return "NaN"
    return v


def compare_samples(sa: dict[str, Any], sb: dict[str, Any], atol: float, rtol: float) -> dict[str, Any] | None:
    """采样元素对比：优先按相同扁平下标对齐，下标不一致时取交集。"""
    if not sa or not sb:
        return None
    if "unavailable" in sa or "unavailable" in sb:
        return {"skipped": f"unavailable:{sa.get('unavailable')}/{sb.get('unavailable')}"}
    ia, ib = sa.get("idx") or [], sb.get("idx") or []
    va, vb = sa.get("vals") or [], sb.get("vals") or []
    if not ia and not ib:
        return None
    if ia == ib:
        out = compare_values(va, vb, atol, rtol, idx=ia)
        out["aligned_by"] = "same-index"
        return out
    pos_b = {idx: k for k, idx in enumerate(ib)}
    common = [(k, pos_b[idx]) for k, idx in enumerate(ia) if idx in pos_b]
    out = _diff_loop([va[k] for k, _ in common], [vb[j] for _, j in common], atol, rtol,
                     idx=[ia[k] for k, _ in common], ib_idx=pos_b)
    out["aligned_by"] = "index-intersect"
    out["index_mismatch"] = True
    out["n_only_a"] = len(ia) - len(common)
    out["n_only_b"] = len(ib) - len(common)
    out["idx_a"] = [ia[k] for k, _ in common[:5]]
    return out


def compare_stats(sA: dict[str, Any], sB: dict[str, Any], atol: float, rtol: float) -> list[dict[str, Any]]:
    diffs = []
    for k in sorted(set(sA) | set(sB)):
        if k in ("stats_skipped", "stats_error", "finite_only", "all_nonfinite", "skipped"):
            va, vb = sA.get(k), sB.get(k)
            if va != vb:
                diffs.append({"field": k, "a": va, "b": vb, "kind": "flag"})
            continue
        va, vb = sA.get(k), sB.get(k)
        if k.endswith("_count") or k in ("count", "true_count", "zero_count"):
            if va != vb:
                d = {"field": k, "a": va, "b": vb, "kind": "count"}
                if isinstance(va, int) and isinstance(vb, int):
                    d["delta"] = vb - va
                diffs.append(d)
            continue
        if va is None or vb is None:
            if va != vb:
                diffs.append({"field": k, "a": va, "b": vb, "kind": "missing"})
            continue
        diff, ad, rl = is_different(va, vb, atol, rtol)
        if diff:
            diffs.append({"field": k, "a": va, "b": vb, "abs": _num(ad), "rel": _num(rl), "kind": "numeric"})
    return diffs


def compare_basic(bA: dict[str, Any], bB: dict[str, Any]) -> list[dict[str, Any]]:
    diffs = []
    for k in sorted(set(bA) | set(bB)):
        if k in VOLATILE_BASIC:
            continue
        va, vb = bA.get(k), bB.get(k)
        if va != vb:
            diffs.append({"field": k, "a": va, "b": vb})
    return diffs


# ----------------------------------------------------------------------
class Comparator:
    def __init__(self, res_a: Result, res_b: Result, opts: Options):
        self.a = res_a
        self.b = res_b
        self.opts = opts
        self.report: dict[str, Any] = {
            "tool": "pprobe-compare",
            "a": {"path": res_a.path, "ranks": res_a.rank_ids()},
            "b": {"path": res_b.path, "ranks": res_b.rank_ids()},
            "options": {"atol": opts.atol, "rtol": opts.rtol, "sort": opts.sort, "phase": opts.phase},
            "ranks": {},
            "divergences": [],
            "totals": {},
        }

    # ------------------------------------------------------------------
    def rank_pairs(self) -> list[tuple[int, int]]:
        if self.opts.ranks:
            return self.opts.ranks
        ra, rb = self.a.rank_ids(), self.b.rank_ids()
        pairs = [(r, r) for r in ra if r in rb]
        return pairs

    def run(self) -> dict[str, Any]:
        rep = self.report
        rep["env_diff"] = self._compare_env() if self.opts.compare_env else {}
        rep["config_diff"] = self._compare_config()
        totals = {"events_compared": 0, "events_only_a": 0, "events_only_b": 0, "divergent_events": 0,
                  "structural": 0, "info_diff": 0, "special": 0, "stats_diff": 0, "sample_diff": 0,
                  "scalar_diff": 0, "stack_diff": 0, "index_mismatch": 0}
        numeric_keys = list(totals)  # 下面会往 totals 里放 dict，累加时只认数值字段
        for r_a, r_b in self.rank_pairs():
            one = self._compare_rank(r_a, r_b)
            rep["ranks"][str(r_a)] = one
            for k in numeric_keys:
                totals[k] += one.get(k, 0)
            if one.get("first_divergence"):
                cur = totals.get("first_divergence")
                fd = one["first_divergence"]
                fd_seq = fd["seq_a"] if isinstance(fd["seq_a"], int) else 1 << 60
                cur_seq = cur["seq_a"] if isinstance((cur or {}).get("seq_a"), int) else -1
                if cur is None or fd_seq < cur_seq:
                    totals["first_divergence"] = dict(fd, rank=fd.get("rank", r_a))
        rep["divergences"] = self._sorted_top(rep["divergences"])
        rep["module_summary"] = self._module_summary()
        raw = rep.pop("raw_divergences", [])
        totals["divergent_events"] = len(set((d["rank"], tuple(d["key"])) for d in raw))
        totals["divergences_kept"] = len(rep["divergences"])
        totals["max_abs_overall"] = rep.get("_max_abs")
        totals.setdefault("first_divergence", None)     # 无差异时也给稳定字段，方便下游读
        rep.pop("_max_abs", None)
        rep["totals"] = totals
        rep["missing_ranks"] = {
            "a_only": sorted(set(self.a.rank_ids()) - {p[0] for p in self.rank_pairs()}),
            "b_only": sorted(set(self.b.rank_ids()) - {p[1] for p in self.rank_pairs()}),
        }
        return rep

    # ------------------------------------------------------------------
    def _compare_rank(self, r_a: int, r_b: int) -> dict[str, Any]:
        A: RankResult | None = self.a.ranks.get(r_a)
        B: RankResult | None = self.b.ranks.get(r_b)
        out: dict[str, Any] = {"rank_a": r_a, "rank_b": r_b, "events_only_a": 0, "events_only_b": 0,
                              "events_compared": 0, "structural": 0, "info_diff": 0, "special": 0,
                              "stats_diff": 0, "sample_diff": 0, "scalar_diff": 0, "stack_diff": 0,
                              "index_mismatch": 0, "first_divergence": None}
        if A is None or B is None:
            out["error"] = "rank 目录缺失"
            return out
        idx_a = A.index()
        idx_b = B.index()
        keys_b = set(idx_b.keys())
        only_a = [k for k in idx_a.order if k not in keys_b]
        only_b = [k for k in idx_b.order if k not in idx_a.mapping]
        out["events_only_a"] = len(only_a)
        out["events_only_b"] = len(only_b)
        out["only_a_sample"] = [_key_repr(k) for k in only_a[:20]]
        out["only_b_sample"] = [_key_repr(k) for k in only_b[:20]]
        for k in idx_a.order:
            if k not in keys_b:
                continue
            if self.opts.phase and k[0] != self.opts.phase:
                continue
            if not self._module_allowed(k[1]):
                continue
            ev_a = idx_a.get(k)
            ev_b = idx_b.get(k)
            if ev_a is None or ev_b is None:
                continue
            out["events_compared"] += 1
            d = self._compare_event(r_a, k, ev_a, ev_b)
            if d:
                self.report.setdefault("raw_divergences", []).append(d)
                self.report["divergences"].append(d)
                for kind in d["kinds"]:
                    col = KIND_COUNTER.get(kind)
                    if col:
                        out[col] = out.get(col, 0) + 1
                if out["first_divergence"] is None:
                    out["first_divergence"] = {
                        "seq_a": ev_a.get("seq"), "seq_b": ev_b.get("seq"), "key": _key_repr(k),
                        "rank": r_a, "stack_a": d.get("stack_a"), "stack_b": d.get("stack_b"),
                        "kinds": d["kinds"], "max_abs": d.get("max_abs"),
                    }
                abs_v = d.get("max_abs")
                if isinstance(abs_v, (int, float)):
                    cur = self.report.get("_max_abs")
                    if cur is None or abs_v > cur:
                        self.report["_max_abs"] = abs_v
        return out

    def _module_allowed(self, module: str) -> bool:
        import re

        if self.opts.include and not re.search(self.opts.include, module or ""):
            return False
        if self.opts.exclude and re.search(self.opts.exclude, module or ""):
            return False
        return True

    def _compare_event(self, rank: int, key: tuple, ev_a: dict, ev_b: dict) -> dict[str, Any] | None:
        slot_diffs: dict[str, Any] = {}
        kinds: set[str] = set()
        max_abs = 0.0
        max_rel = 0.0
        names_a = dict(tensor_slots(ev_a))
        names_b = dict(tensor_slots(ev_b))
        for slot in dict.fromkeys(list(names_a) + list(names_b)):
            if slot in self.opts.ignore_slots:
                continue
            ea, eb = names_a.get(slot), names_b.get(slot)
            one = _compare_slot(slot, ea, eb, self.opts)
            if not one:
                continue
            slot_diffs[slot] = one
            kinds.update(one["kinds"])
            if isinstance(one.get("max_abs"), (int, float)) and one["max_abs"] > max_abs:
                max_abs = one["max_abs"]
            if isinstance(one.get("max_rel"), (int, float)) and one["max_rel"] > max_rel:
                max_rel = one["max_rel"]
        # 非张量槽位（标量参数 / 容器结构）差异：eps、reduction、size_average 等常常就是根因
        sd_a = {k: v for k, v in (ev_a.get("tensors") or {}).items() if not isinstance(v, dict) or v.get("kind") != "tensor"}
        sd_b = {k: v for k, v in (ev_b.get("tensors") or {}).items() if not isinstance(v, dict) or v.get("kind") != "tensor"}
        for slot in dict.fromkeys(list(sd_a) + list(sd_b)):
            va, vb = sd_a.get(slot), sd_b.get(slot)
            if va != vb:
                kind = "presence" if (va is None or vb is None) else "scalar"
                slot_diffs.setdefault(slot, {"kinds": [kind], "changes": {}})["changes"]["value"] = [
                    {"field": "value", "a": _scalar_repr(va), "b": _scalar_repr(vb)}
                ]
                kinds.add(kind)
        stack_a, stack_b = ev_a.get("stack_id"), ev_b.get("stack_id")
        if stack_a != stack_b and stack_a and stack_b:
            # 堆栈 id 不同意味着这个模块在两次运行里由不同调用路径触发（或同一路径但被重新编号）
            kinds.add("stack_diff")
        # 模块类名变了（例如被换成 LoRA/量化包装）即使数值一致也是重要信息，必须在早退之前判
        if ev_a.get("module_cls") != ev_b.get("module_cls"):
            kinds.add("info")
            slot_diffs["__module_cls__"] = {"kinds": ["info"], "changes": {"basic": [
                {"field": "module_cls", "a": ev_a.get("module_cls"), "b": ev_b.get("module_cls")}]}}
        if not slot_diffs and "stack_diff" not in kinds:
            return None
        return {
            "rank": rank,
            "key": list(key),
            "phase": key[0],
            "module": key[1],
            "call_index": key[2],
            "seq_a": ev_a.get("seq"),
            "seq_b": ev_b.get("seq"),
            "step_a": ev_a.get("step"),
            "step_b": ev_b.get("step"),
            "module_cls_a": ev_a.get("module_cls"),
            "module_cls_b": ev_b.get("module_cls"),
            "stack_a": stack_a,
            "stack_b": stack_b,
            "fwd_seq_a": ev_a.get("fwd_seq"),
            "fwd_seq_b": ev_b.get("fwd_seq"),
            "kinds": sorted(kinds),
            "max_abs": _num(max_abs) if kinds & {"sample", "stats"} else None,
            "max_rel": _num(max_rel) if kinds & {"sample", "stats"} else None,
            "slots": slot_diffs,
        }

    # ------------------------------------------------------------------
    def _sorted_top(self, divs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.opts.sort == "max_abs":
            divs.sort(key=lambda d: (d.get("max_abs") if isinstance(d.get("max_abs"), (int, float)) else -1), reverse=True)
        elif self.opts.sort == "max_rel":
            divs.sort(key=lambda d: (d.get("max_rel") if isinstance(d.get("max_rel"), (int, float)) else -1), reverse=True)
        else:  # seq：按事件发生顺序，最早发散的点最可能是根因
            # 先按 seq 再按 rank：多卡时两个 rank 的 seq 空间各自独立，
            # 若按 (rank, seq) 排，--top 会被前几个 rank 吃满，后面的 rank 一条明细都看不到。
            divs.sort(key=lambda d: (d["seq_a"] if isinstance(d["seq_a"], int) else 1 << 30, d["rank"]))
        return divs[: self.opts.top] if self.opts.top > 0 else divs

    def _module_summary(self) -> list[dict[str, Any]]:
        agg: dict[str, dict[str, Any]] = {}
        for d in self.report.get("raw_divergences", []):
            m = d["module"]
            row = agg.setdefault(m, {"module": m, "n_divergent_events": 0, "kinds": set(),
                                     "max_abs": 0.0, "max_rel": 0.0, "first_seq": None})
            row["n_divergent_events"] += 1
            row["kinds"].update(d["kinds"])
            for f in ("max_abs", "max_rel"):
                v = d.get(f)
                if isinstance(v, (int, float)) and v > row[f]:
                    row[f] = v
            if row["first_seq"] is None or (isinstance(d["seq_a"], int) and d["seq_a"] < row["first_seq"]):
                row["first_seq"] = d["seq_a"]
        rows = []
        for row in agg.values():
            row["kinds"] = sorted(row["kinds"])
            row["max_abs"] = _num(row["max_abs"])
            row["max_rel"] = _num(row["max_rel"])
            rows.append(row)
        rows.sort(key=lambda r: (-r["n_divergent_events"], r["module"]))
        return rows

    # ------------------------------------------------------------------
    def _compare_env(self) -> dict[str, Any]:
        """环境差异 —— 跨平台精度不一致时最先看这里。"""
        out: dict[str, Any] = {}
        cmdlines: dict[str, Any] = {}
        for r_a, r_b in self.rank_pairs():
            A = self.a.ranks.get(r_a)
            B = self.b.ranks.get(r_b)
            if not A or not B:
                continue
            ca, cb = (A.env or {}).get("cmdline"), (B.env or {}).get("cmdline")
            if ca != cb:
                cmdlines[str(r_a)] = {"a": ca, "b": cb}
            flat_a = _flatten(_without(A.env, "cmdline"))
            flat_b = _flatten(_without(B.env, "cmdline"))
            diffs = {}
            for k in sorted(set(flat_a) | set(flat_b)):
                if k in VOLATILE_ENV:
                    continue
                va, vb = flat_a.get(k), flat_b.get(k)
                if str(va) != str(vb):
                    diffs[k] = {"a": va, "b": vb}
            if diffs:
                out[str(r_a)] = diffs
        self.report["cmdline_diff"] = cmdlines
        return out

    def _compare_config(self) -> dict[str, Any]:
        A = next(iter(self.a.ranks.values()), None)
        B = next(iter(self.b.ranks.values()), None)
        if not A or not B:
            return {}
        ca, cb = A.config or {}, B.config or {}
        diffs = {}
        for k in sorted(set(ca) | set(cb)):
            if k in ("out_dir", "run_name"):
                continue
            if _norm(ca.get(k)) != _norm(cb.get(k)):
                diffs[k] = {"a": ca.get(k), "b": cb.get(k)}
        return diffs


def _compare_slot(slot: str, ea: dict | None, eb: dict | None, opts: Options) -> dict[str, Any] | None:
    if ea is None or eb is None:
        return {"kinds": ["presence"], "changes": {"presence": [
            {"field": "presence", "a": "present" if ea else "absent",
             "b": "present" if eb else "absent"}]}}
    if ea.get("kind") != eb.get("kind"):
        return {"kinds": ["presence"], "changes": {"presence": [
            {"field": "kind", "a": ea.get("kind"), "b": eb.get("kind")}]}}
    changes: dict[str, Any] = {}
    kinds: set[str] = set()
    ba, bb = ea.get("basic") or {}, eb.get("basic") or {}
    info = compare_basic(ba, bb)
    if info:
        changes["basic"] = info
        kinds.add("info")
    sa, sb = ea.get("stats") or {}, eb.get("stats") or {}
    st = compare_stats(sa, sb, opts.atol, opts.rtol)
    if st:
        changes["stats"] = st
        kinds.add("stats")
        for c in st:
            if c["field"].endswith("_count") and c["field"].startswith(("nan", "inf", "posinf", "neginf")):
                kinds.add("special")
    sm = compare_samples(ea.get("sample") or {}, eb.get("sample") or {}, opts.atol, opts.rtol)
    if sm:
        changes["sample"] = sm
        if sm.get("n_diff"):
            kinds.add("sample")
            if sm.get("special_changes"):
                kinds.add("special")
    if sm and sm.get("index_mismatch"):
        kinds.add("index_mismatch")
    if kinds:
        return {
            "kinds": sorted(kinds),
            "max_abs": sm.get("max_abs") if sm else None,
            "max_rel": sm.get("max_rel") if sm else None,
            "changes": changes,
        }
    return None


def _key_repr(key: tuple) -> str:
    phase, module, call = key
    return f"{phase}:{module}#{call}"


def _without(d: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if k not in keys}


def _scalar_repr(v: Any) -> Any:
    """把 {'kind':'scalar','value':1e-05} 压成 1e-05，报告里直接可读。"""
    if isinstance(v, dict):
        if "value" in v:
            return v.get("value")
        if v.get("kind"):
            return v.get("kind")
    return short_repr(v, 90)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, name + "."))
        else:
            out[name] = v
    return out


def _norm(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return v
