"""读取层与查询命令的底层能力（需求 4/7）：rank 发现、事件索引、堆栈反查、速览报告。"""

from __future__ import annotations

import os
import torch

from pprobe.query import (events_by_stack, format_event_rows, format_stack, format_survey,
                          query_events, resolve_stacks, stack_ids_for_module, survey_rank)
from pprobe.result import EventIndex, Result, find_rank_dirs, tensor_slots


def _frames(*specs):
    """造 stacks.json 里的帧：(file, line, name)。"""
    return [
        {"file": f"/app/{f}", "line": ln, "name": nm, "text": f"/app/{f}:{ln} in {nm}"}
        for f, ln, nm in specs
    ]


def _real_frames():
    """指向本测试文件自己，才能验证 linecache 拓到的源码行非空。"""
    path = os.path.abspath(__file__)
    return [{"file": path, "line": 1, "name": "<module>", "text": f"{os.path.basename(path)}:1 in <module>"}]




# ----------------------------------------------------------------------
def test_find_rank_dirs_variants(tmp_path):
    # 正常：rankN 目录
    root = tmp_path / "a"
    for r in (0, 3):
        os.makedirs(root / f"rank{r}")
    (root / "rankx").mkdir()                       # 非法名字要忽略
    assert find_rank_dirs(str(root)) == {0: str(root / "rank0"), 3: str(root / "rank3")}

    # 单 rank 直接写在 root 下（PPROBE_PER_RANK_DIR=0）
    flat = tmp_path / "b"
    (flat / "tensors").mkdir(parents=True)
    (flat / "events.jsonl").write_text("{}\n")
    assert find_rank_dirs(str(flat)) == {0: str(flat)}

    # 退化：rank 目录没有 manifest，但有 events.jsonl
    loose = tmp_path / "c"
    (loose / "rank2").mkdir(parents=True)
    (loose / "rank2" / "events.jsonl").write_text("{}\n")
    assert list(find_rank_dirs(str(loose))) == [2]


def test_result_discovery_and_describe(build_run, tiny_data):
    x, _ = tiny_data()
    a = build_run("ra", [{"tensors": {"input[0]": x}}, {"tensors": {"output": x}}], rank=0)
    build_run("rb", [{"tensors": {"input[0]": x}}], rank=1)
    res = Result(a)
    assert res.rank_ids() == [0]
    assert res.total_events() == 2
    desc = res.describe()
    assert "ranks=[0]" in desc and "events=2" in desc and "state=finalized" in desc


def test_result_missing_dir_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        Result(str(tmp_path / "nope"))


def test_rank_metadata_accessors(build_run, tiny_data):
    x, _ = tiny_data()
    path = build_run("meta", [{"tensors": {"input[0]": x}}])
    rank = Result(path).ranks[0]
    assert rank.has_events
    assert rank.manifest["tool"] == "pprobe"
    assert rank.config["sample_n"] == 8
    assert rank.sampling["mode"] == "uniform"
    assert rank.env["rank"] == 0 and rank.env["world_size"] == 1
    assert rank.env["torch"] == torch.__version__ and "cmdline" in rank.env
    backends = rank.env["backends"]
    # 影响数值的开关必须被抓到（TF32 / SDP / 确定性），这是跨平台归因的第一依据
    assert "deterministic_algorithms" in backends and "float32_matmul_precision" in backends
    assert any(k.startswith(("cuda.matmul", "cudnn", "sdp.", "fp32_precision")) for k in backends)
    assert Result(path).run["finalized_by_rank0"] is True


def test_missing_optional_files_are_empty(tmp_path):
    d = tmp_path / "bare" / "rank0"
    d.mkdir(parents=True)
    rank = Result(str(tmp_path / "bare")).ranks[0]
    assert rank.manifest == {} and rank.env == {} and rank.stacks == {}
    assert not rank.has_events and list(rank.iter_events()) == []
    assert rank.index().mapping == {}


def test_iter_events_tolerates_truncated_tail(build_run, tiny_data):
    """需求 5：被 kill 时最后一行可能只写了一半，读取端必须能跳过。"""
    x, _ = tiny_data()
    path = build_run("tail", [{"tensors": {"input[0]": x}} for _ in range(3)])
    ev = os.path.join(path, "rank0", "events.jsonl")
    with open(ev, "a", encoding="utf-8") as f:
        f.write('{"seq": 4, "module": "net"   <<< 残缺')
    rows = list(Result(path).ranks[0].iter_events())
    assert [r["seq"] for r in rows] == [1, 2, 3]


def test_event_index_aligns_by_call_index(build_run, tiny_data):
    """对齐 key 用 (phase, module, call_index) 而不是 seq，seq 跨运行不稳定。"""
    x, _ = tiny_data()
    path = build_run("idx", [
        {"module": "net.a", "tensors": {"input[0]": x}, "call_index": 0},
        {"module": "net.a", "tensors": {"input[0]": x}, "call_index": 1},
        {"module": "net.b", "tensors": {"input[0]": x}, "call_index": 0},
    ])
    rank = Result(path).ranks[0]
    idx = rank.index()
    assert len(idx) == 3
    assert ("forward", "net.a", 1) in idx.keys()
    assert idx.get(("forward", "net.b", 0))["module"] == "net.b"
    assert idx.get(("forward", "net.b", 9)) is None
    assert idx.order[0] == ("forward", "net.a", 0)


def test_event_index_dedup_first_vs_last(tmp_path, build_run, tiny_data):
    x, _ = tiny_data()
    path = build_run("dup", [
        {"module": "net.a", "tensors": {"input[0]": x}, "call_index": 0, "extra": {"tag": "first"}},
        {"module": "net.a", "tensors": {"input[0]": x}, "call_index": 0, "extra": {"tag": "second"}},
    ])
    rank = Result(path).ranks[0]
    assert rank.index().get(("forward", "net.a", 0))["tag"] == "first"
    assert EventIndex.build(rank, dedup="last").get(("forward", "net.a", 0))["tag"] == "second"


def test_tensor_slots_only_tensors(build_run, tiny_data):
    x, _ = tiny_data()
    path = build_run("slots", [{"tensors": {"input[0]": x, "input[1]": 3}}])
    ev = next(iter(Result(path).ranks[0].iter_events()))
    assert [n for n, _ in tensor_slots(ev)] == ["input[0]"]


# ----------------------------------------------------------------------
def test_resolve_stacks_across_ranks(build_run, tiny_data):
    x, _ = tiny_data()
    fr = _real_frames()
    path = build_run("stk", [
        {"tensors": {"input[0]": x}, "stack_frames": fr},
        {"tensors": {"input[0]": x}, "stack_frames": fr},
    ])
    res = Result(path)
    sid = stack_ids_for_module(res.ranks[0])[0]
    items = resolve_stacks(res, [sid])
    assert len(items) == 1 and items[0]["rank"] == 0
    assert items[0]["stack"]["count"] == 2
    assert items[0]["stack"]["frames"][0]["source"]          # 真实源码行被抓进来了
    assert resolve_stacks(res, [sid], ranks=[7]) == []       # 该 rank 不存在
    assert resolve_stacks(res, ["s999"]) == []


def test_format_stack_text(build_run, tiny_data):
    x, _ = tiny_data()
    fr = _frames(("train.py", 10, "main"), ("train.py", 42, "step"))
    path = build_run("fmt", [{"tensors": {"input[0]": x}, "stack_frames": fr}])
    res = Result(path)
    rank = res.ranks[0]
    sid = next(iter(rank.stacks))
    item = resolve_stacks(res, [sid], with_events=True)[0]
    text = format_stack(item)
    assert f"堆栈 {sid}" in text and "train.py:42 in step" in text
    assert "涉及文件 1 个，帧数 2" in text
    assert "引用该堆栈的事件:" in text and "forward" in text
    assert format_stack({"rank": 0, "id": "sx", "stack": {"count": 0}}).count("(无帧信息)") == 1


def test_events_by_stack_and_module_lookup(build_run, tiny_data):
    x, _ = tiny_data()
    f1 = _frames(("train.py", 1, "a"))
    f2 = _frames(("train.py", 2, "b"))
    path = build_run("q", [
        {"module": "net.enc", "tensors": {"input[0]": x}, "stack_frames": f1},
        {"module": "net.dec", "tensors": {"input[0]": x}, "stack_frames": f2},
        {"module": "net.enc", "phase": "backward", "tensors": {"input[0]": x}, "stack_frames": f1},
    ])
    rank = Result(path).ranks[0]
    ids = stack_ids_for_module(rank)
    assert len(ids) == 2
    assert stack_ids_for_module(rank, module_re=r"^net\.dec$") == ids[1:2]
    assert stack_ids_for_module(rank, phase="backward") == [ids[0]]
    assert len(events_by_stack(rank, ids[0])) == 2
    assert events_by_stack(rank, ids[0], limit=1)


# ----------------------------------------------------------------------
def test_query_events_filters(build_run, tiny_data):
    x, _ = tiny_data()
    bad = torch.tensor([1.0, float("nan"), float("inf")])
    path = build_run("qe", [
        {"module": "net.a", "tensors": {"input[0]": x, "output": x}},
        {"module": "net.b", "tensors": {"input[0]": bad}},
        {"module": "net.b", "phase": "backward", "tensors": {"grad": x}},
    ])
    res = Result(path)
    assert [r["module"] for r in query_events(res)] == ["net.a", "net.b", "net.b"]
    assert [r["seq"] for r in query_events(res, module_re=r"^net\.b$")] == [2, 3]
    assert [r["seq"] for r in query_events(res, phase="backward")] == [3]
    assert [r["seq"] for r in query_events(res, seq_min=2, seq_max=2)] == [2]
    nan_rows = query_events(res, only_nan=True)
    assert [r["seq"] for r in nan_rows] == [2]
    assert nan_rows[0]["slots"] == ["input[0](nan=1,inf=1)"]
    assert query_events(res, only_nan=True, slot_re=r"^out$") == []
    assert [r["seq"] for r in query_events(res, only_inf=True)] == [2]
    assert [r["seq"] for r in query_events(res, ranks=[0], limit=1)] == [1]
    assert query_events(res, ranks=[5]) == []
    txt = format_event_rows(query_events(res, limit=2))
    assert "rank" in txt.splitlines()[0] and len(txt.splitlines()) == 4
    assert format_event_rows([]) == "没有匹配的事件"


def test_query_events_reports_slot_names(build_run, tiny_data):
    x, _ = tiny_data()
    path = build_run("qn", [{"tensors": {"input[0]": x, "kwargs.mask": x}}])
    row = query_events(Result(path))[0]
    assert row["slots"] == ["input[0]", "kwargs.mask"]
    assert row["n_tensors"] == 2 and row["cls"] == "Linear"


# ----------------------------------------------------------------------
def test_survey_rank_nan_hotspots(build_run, tiny_data):
    x, _ = tiny_data()
    nan_t = torch.tensor([float("nan"), 1.0])
    inf_t = torch.tensor([float("inf"), 2.0])
    path = build_run("sv", [
        {"module": "net.a", "tensors": {"output": x}},
        {"module": "net.b", "tensors": {"output": nan_t}},
        {"module": "net.c", "tensors": {"output": inf_t}},
    ])
    row = survey_rank(Result(path).ranks[0])
    assert row["events"] == 3
    assert row["per_phase"] == {"forward": 3, "backward": 0}
    assert row["dtypes"]["torch.float32"] == 3
    assert row["devices"]["cpu"] == 3
    assert row["first_nan"]["seq"] == 2 and row["first_nan"]["module"] == "net.b"
    assert row["first_nan"]["stack_id"] is None
    assert [h["module"] for h in row["nan_hotspots"]] == ["net.b", "net.c"]
    assert row["max_abs_seen"] >= 1.0
    assert row["unique_stacks"] == 0
    assert row["manifest_state"] == "finalized"
    assert row["counters"]["events"] == 3
    text = format_survey([row])
    assert "最早出现 NaN 的位置: seq=2" in text and "pprobe stack" in text
    assert "NaN/Inf 热点" in text and "rank0" in text


def test_survey_rank_clean(build_run, tiny_data):
    x, _ = tiny_data()
    path = build_run("svok", [{"module": "net.a", "tensors": {"output": x * 3.0}}])
    row = survey_rank(Result(path).ranks[0])
    assert row["first_nan"] is None and row["nan_hotspots"] == []
    assert row["max_abs_seen"] > 3.0                        # x*3 后的采样最大值
    assert row["largest"][0]["slot"] == "net.a.output"
    assert "未发现 NaN/Inf" in format_survey([row])
