"""结果目录的查询能力：堆栈 id 反查完整堆栈、事件过滤、单目录速览报告。

对应 ``pprobe stack`` / ``pprobe query`` / ``pprobe report`` 三条命令。
注意：堆栈 id 是 **rank 内**独立编号的（每个进程各自维护 stacks.json）。
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterator

from .result import Result, RankResult, tensor_slots
from .util import decode_number, human_bytes


# ----------------------------------------------------------------------
# 堆栈查询
# ----------------------------------------------------------------------
def resolve_stacks(res: Result, ids: list[str], ranks: list[int] | None = None,
                   with_events: bool = False) -> list[dict[str, Any]]:
    """按堆栈 id 查询完整堆栈；未指定 rank 时在所有 rank 里查。"""
    out: list[dict[str, Any]] = []
    targets = ranks if ranks is not None else res.rank_ids()
    for sid in ids:
        for r in targets:
            rank: RankResult | None = res.ranks.get(r)
            if rank is None:
                continue
            entry = rank.stack(sid)
            if entry is None:
                continue
            item: dict[str, Any] = {"rank": r, "id": sid, "stack": entry}
            if with_events:
                item["events"] = events_by_stack(rank, sid, limit=20)
            out.append(item)
    return out


def format_stack(item: dict[str, Any], color: bool = False) -> str:
    entry = item["stack"]
    lines = [f"=== rank{item['rank']}  堆栈 {item['id']}  (被 {entry.get('count', '?')} 条事件引用) ==="]
    frames = entry.get("frames") or []
    for i, f in enumerate(frames):
        src = f.get("source") or ""
        lines.append(f"  #{i:<3d} {f.get('text', '')}")
        if src:
            lines.append(f"        | {src}")
    if not frames:
        lines.append("  (无帧信息)")
    sig = entry.get("signature") or ""
    files = sorted({f.get("file", "") for f in frames if f.get("file")})
    lines.append(f"  涉及文件 {len(files)} 个，帧数 {len(frames)}")
    if item.get("events"):
        lines.append("  引用该堆栈的事件:")
        for ev in item["events"]:
            lines.append(f"    seq={ev['seq']:>7}  {ev['phase']:<8} {ev['module']}  #{ev.get('call_index')}")
    return "\n".join(lines)


def events_by_stack(rank: RankResult, sid: str, limit: int = 50) -> list[dict[str, Any]]:
    out = []
    for ev in rank.iter_events():
        if ev.get("stack_id") == sid or ev.get("fwd_stack_id") == sid:
            out.append({k: ev.get(k) for k in ("seq", "phase", "module", "call_index", "step")})
            if len(out) >= limit:
                break
    return out


def stack_ids_for_module(rank: RankResult, module_re: str | None = None, phase: str | None = None,
                         limit: int = 2000) -> list[str]:
    rx = re.compile(module_re) if module_re else None
    ids: list[str] = []
    seen: set[str] = set()
    for ev in rank.iter_events():
        if rx and not rx.search(ev.get("module") or ""):
            continue
        if phase and ev.get("phase") != phase:
            continue
        sid = ev.get("stack_id")
        if sid and sid not in seen:
            seen.add(sid)
            ids.append(sid)
            if len(ids) >= limit:
                break
    return ids


# ----------------------------------------------------------------------
# 事件查询
# ----------------------------------------------------------------------
def query_events(res: Result, module_re: str | None = None, phase: str | None = None,
                 ranks: list[int] | None = None, seq_min: int | None = None, seq_max: int | None = None,
                 only_nan: bool = False, only_inf: bool = False, slot_re: str | None = None,
                 limit: int = 40) -> list[dict[str, Any]]:
    rx = re.compile(module_re) if module_re else None
    srx = re.compile(slot_re) if slot_re else None
    rows: list[dict[str, Any]] = []
    for r in (ranks if ranks is not None else res.rank_ids()):
        rank = res.ranks.get(r)
        if rank is None:
            continue
        for ev in rank.iter_events():
            if rx and not rx.search(ev.get("module") or ""):
                continue
            if phase and ev.get("phase") != phase:
                continue
            seq = ev.get("seq") or 0
            if seq_min is not None and seq < seq_min:
                continue
            if seq_max is not None and seq > seq_max:
                continue
            hits = _slot_hits(ev, srx, only_nan, only_inf)
            if only_nan or only_inf:
                if not hits:
                    continue
            rows.append({
                "rank": r, "seq": seq, "phase": ev.get("phase"), "module": ev.get("module"),
                "cls": ev.get("module_cls"), "call_index": ev.get("call_index"),
                "step": ev.get("step"), "stack_id": ev.get("stack_id"),
                "n_tensors": ev.get("n_tensors"), "slots": hits or _slot_names(ev),
            })
            if len(rows) >= limit:
                return rows
    return rows


def _slot_names(ev: dict[str, Any]) -> list[str]:
    return list((ev.get("tensors") or {}).keys())[:8]


def _slot_hits(ev: dict[str, Any], srx: re.Pattern | None, only_nan: bool, only_inf: bool) -> list[str]:
    out = []
    for name, entry in (ev.get("tensors") or {}).items():
        if not isinstance(entry, dict) or entry.get("kind") != "tensor":
            continue
        if srx and not srx.search(name):
            continue
        st = entry.get("stats") or {}
        nan = int(st.get("nan_count") or 0)
        # inf_count 本身已经是 posinf+neginf，不能再加一遍
        inf = int(st.get("inf_count") or 0) or int(st.get("posinf_count") or 0) + int(st.get("neginf_count") or 0)
        if (only_nan and nan) or (only_inf and inf):
            out.append(f"{name}(nan={nan},inf={inf})")
    return out


def format_event_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "没有匹配的事件"
    head = f"{'rank':>4} {'seq':>6} {'phase':<9} {'step':>4} {'stack':<6} {'module':<52} slots"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['rank']:>4} {r['seq']:>6} {str(r['phase']):<9} {str(r['step']):>4} "
            f"{str(r['stack_id'] or '-'):<6} {str(r['module'])[:52]:<52} {', '.join(r['slots'][:4])}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 单目录速览
# ----------------------------------------------------------------------
def survey_rank(rank: RankResult, top: int = 15) -> dict[str, Any]:
    """一个 rank 的整体画像：NaN/Inf 热点、最早出现 NaN 的位置、量级最大的模块。"""
    n_events = 0
    per_phase = {"forward": 0, "backward": 0}
    dtype_hist: dict[str, int] = {}
    device_hist: dict[str, int] = {}
    nan_hotspots: dict[str, dict[str, Any]] = {}
    first_nan: dict[str, Any] | None = None
    biggest: list[tuple[float, int, str, str]] = []
    zero_grad_like = 0
    max_abs_seen = 0.0
    nbytes = 0
    for ev in rank.iter_events():
        n_events += 1
        per_phase[ev.get("phase") or "?"] = per_phase.get(ev.get("phase") or "?", 0) + 1
        for name, entry in tensor_slots(ev):
            b = entry.get("basic") or {}
            st = entry.get("stats") or {}
            dtype_hist[str(b.get("dtype"))] = dtype_hist.get(str(b.get("dtype")), 0) + 1
            device_hist[str(b.get("device"))] = device_hist.get(str(b.get("device")), 0) + 1
            nbytes += int(b.get("nbytes") or 0)
            nan = int(st.get("nan_count") or 0)
            inf = int(st.get("inf_count") or 0)
            key = f"{ev.get('phase')}:{ev.get('module')}.{name}"
            if nan or inf:
                hot = nan_hotspots.setdefault(key, {"module": ev.get("module"), "phase": ev.get("phase"),
                                                    "slot": name, "n_events": 0, "nan": 0, "inf": 0,
                                                    "first_seq": ev.get("seq"), "stack_id": ev.get("stack_id")})
                hot["n_events"] += 1
                hot["nan"] += nan
                hot["inf"] += inf
                if nan and first_nan is None:
                    first_nan = {"seq": ev.get("seq"), "phase": ev.get("phase"), "module": ev.get("module"),
                                 "slot": name, "stack_id": ev.get("stack_id"), "nan": nan,
                                 "call_index": ev.get("call_index"), "step": ev.get("step")}
            for f in ("absmax", "max"):
                v = st.get(f)
                if isinstance(v, (int, float)) or isinstance(v, str):
                    try:
                        fv = abs(decode_number(v))
                    except (TypeError, ValueError):
                        continue
                    if fv == float("inf"):
                        continue
                    if fv > max_abs_seen:
                        max_abs_seen = fv
                    if len(biggest) < 400:
                        biggest.append((fv, ev.get("seq") or 0, f"{ev.get('module')}.{name}", str(ev.get("phase"))))
                    break
    biggest.sort(key=lambda x: -x[0])
    hot_list = sorted(nan_hotspots.values(), key=lambda h: (-h["nan"], -h["inf"]))[:top]
    return {
        "rank": rank.rank,
        "events": n_events,
        "per_phase": per_phase,
        "dtypes": dtype_hist,
        "devices": device_hist,
        "logical_bytes": nbytes,
        "max_abs_seen": max_abs_seen,
        "first_nan": first_nan,
        "nan_hotspots": hot_list,
        "largest": [{"absmax": v, "seq": s, "slot": n, "phase": p} for v, s, n, p in biggest[:top]],
        "unique_stacks": len(rank.stacks),
        "manifest_state": (rank.manifest or {}).get("state"),
        "limit_reason": (rank.manifest or {}).get("limit_reason"),
        "counters": (rank.manifest or {}).get("counters"),
        "sampling": rank.sampling,
    }


def format_survey(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        out.append(f"### rank{row['rank']}")
        out.append(f"事件: {row['events']} (forward={row['per_phase'].get('forward', 0)}, "
                   f"backward={row['per_phase'].get('backward', 0)}), 唯一堆栈: {row['unique_stacks']}")
        out.append(f"dtype 分布: {row['dtypes']}  device 分布: {row['devices']}")
        out.append(f"张量逻辑体积合计: {human_bytes(row['logical_bytes'])}，采样到的最大 |value|: {row['max_abs_seen']:.6g}")
        st = row.get("counters") or {}
        if st.get("errors"):
            out.append(f"探针内部异常: {st['errors']}")
        if st.get("sample_failures"):
            out.append(f"采样失败: {st['sample_failures']}")
        if row["first_nan"]:
            fn = row["first_nan"]
            out.append(f"最早出现 NaN 的位置: seq={fn['seq']} {fn['phase']} {fn['module']} "
                       f"[{fn['slot']}] 堆栈={fn['stack_id']}  (pprobe stack {fn['stack_id']} --rank {row['rank']})")
        else:
            out.append("未发现 NaN/Inf")
        if row["nan_hotspots"]:
            out.append("NaN/Inf 热点 (前 10):")
            for h in row["nan_hotspots"][:10]:
                out.append(f"  {h['phase']:<9} {h['module']}.{h['slot']:<24} nan={h['nan']:<8} inf={h['inf']:<8} "
                           f"事件数={h['n_events']} first_seq={h['first_seq']} stack={h['stack_id']}")
        if row["largest"]:
            out.append("量级最大的采样位置 (前 10):")
            for x in row["largest"][:10]:
                out.append(f"  |{x['slot']}|={x['absmax']:.6g}  seq={x['seq']}  {x['phase']}")
        out.append("")
    return "\n".join(out)
