"""记录器（需求 5、8）：延迟开目录、定期落盘、上限收尾、信号兜底、按 rank 分目录。"""

from __future__ import annotations

import json
import os
import signal
import threading
import time

import pytest
import torch

from pprobe.config import Config
from pprobe.recorder import Recorder, PProbeLimitReached
from pprobe.result import Result

EVENTS = "events.jsonl"


def _lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in f.read().splitlines() if ln.strip()]


# ----------------------------------------------------------------------
def test_lazy_recorder_touches_nothing(tmp_path, make_config):
    """torchrun 的 agent 进程也会 import torch，绝不能留下空 rank 目录/覆盖元信息。"""
    out = tmp_path / "lazy"
    rec = Recorder(make_config(out_dir=str(out)), rank=0, lazy=True)
    assert rec.flush() == 0
    assert rec.on_optimizer_step() is None
    rec.finalize("never-opened")
    assert not out.exists(), "没发生任何前向时不应创建结果目录"
    assert not hasattr(rec, "rank_dir")
    # 一旦真的记录，目录才出现
    assert rec.active is True
    assert (out / "rank0" / EVENTS).parent.is_dir()


def test_rank_dirs_and_metadata(tmp_path, recorder):
    rec = recorder("meta", rank=2)
    assert rec.rank == 2 and rec.rank_dir.endswith("rank2")
    assert os.path.isdir(os.path.join(rec.rank_dir, "tensors"))
    assert rec.cfg.record_env and os.path.isfile(os.path.join(rec.rank_dir, "env.json"))
    man = json.load(open(os.path.join(rec.rank_dir, "manifest.json")))
    assert man["tool"] == "pprobe" and man["state"] == "running"
    assert man["rank"] == 2 and man["world_size"] == 1
    assert man["config"]["sample_n"] == 8
    assert man["sampling"]["seed_user"] is None and man["sampling"]["seed_used"]
    assert man["counters"]["events"] == 0


def test_per_rank_dir_off(tmp_path, recorder):
    rec = recorder("flat", rank=1, per_rank_dir=False)
    assert rec.rank_dir == rec.root and rec.rank_dir.endswith("flat")


def test_only_ranks_skips_other_ranks(recorder):
    rec = recorder("filtered", rank=0, only_ranks=(1,))
    assert rec.active is False
    assert rec.record("forward", "net", "Linear", {}) is None


# ----------------------------------------------------------------------
def test_flush_interval_writes_incrementally(recorder, add_event, tiny_data):
    """需求 5：每隔多少次调用落盘一次，中断也能拿到已发生的数据。"""
    x, _ = tiny_data()
    rec = recorder("flush", flush_interval=2)
    path = os.path.join(rec.rank_dir, EVENTS)
    assert not os.path.exists(path)
    for i in range(5):
        add_event(rec, {"input[0]": x}, module="net", call_index=i)
        got = len(_lines(path)) if os.path.exists(path) else 0
        assert got == min(2 * ((i + 1) // 2), 4), (i, got)
    assert rec.counters()["buffered"] == 1
    rec.flush()
    assert len(_lines(path)) == 5


def test_flush_by_time(recorder, add_event, tiny_data):
    """需求 5：除了按次数，也可以按时间间隔落盘（快速迭代但调用很少的场景）。"""
    x, _ = tiny_data()
    rec = recorder("flushtime", flush_interval=1000, flush_secs=0.02)
    path = os.path.join(rec.rank_dir, EVENTS)
    add_event(rec, {"input[0]": x})
    assert not os.path.exists(path)                     # 次数没到、时间也没到
    time.sleep(0.05)
    add_event(rec, {"input[0]": x}, module="net2", call_index=0)
    assert len(_lines(path)) == 2


def test_manual_flush_only_when_dirty(recorder, add_event, tiny_data):
    x, _ = tiny_data()
    rec = recorder("flushtime2", flush_interval=1000, flush_secs=0.01)
    add_event(rec, {"input[0]": x})
    time.sleep(0.02)
    assert rec.flush(fsync=False) == 1                  # 手动 flush 把残留缓冲写掉
    assert rec.flush(fsync=False) == 0                  # 空缓冲不再动文件


def test_finalize_writes_tail_and_exit_file(recorder, add_event, tiny_data):
    x, _ = tiny_data()
    rec = recorder("fin", flush_interval=100)
    for i in range(3):
        add_event(rec, {"output": x}, module="net", call_index=i)
    path = os.path.join(rec.rank_dir, EVENTS)
    assert not os.path.exists(path)                     # 还在缓冲里
    rec.finalize("done")
    assert len(_lines(path)) == 3
    assert rec.active is False
    man = json.load(open(os.path.join(rec.rank_dir, "manifest.json")))
    assert man["state"] == "finalized" and man["counters"]["events"] == 3
    assert json.load(open(os.path.join(rec.rank_dir, "exit.json")))["reason"] == "done"
    rec.finalize("again")                               # 幂等
    assert len(_lines(path)) == 3


def test_max_events_keeps_last_event_and_stops(recorder, add_event, tiny_data):
    """需求 5：限制总次数，且达到上限那一条也要保存。"""
    x, _ = tiny_data()
    rec = recorder("cap", max_events=3, flush_interval=100)
    for i in range(6):
        add_event(rec, {"output": x}, module="net", call_index=i)
    rows = _lines(os.path.join(rec.rank_dir, EVENTS))
    assert len(rows) == 3
    assert [json.loads(r)["seq"] for r in rows] == [1, 2, 3]
    man = json.load(open(os.path.join(rec.rank_dir, "manifest.json")))
    assert "max_events=3" in man["limit_reason"]
    assert rec.record("forward", "net", "Linear", {}) is None


def test_max_steps_finalizes(recorder, add_event, tiny_data):
    x, _ = tiny_data()
    rec = recorder("steps", max_steps=2, flush_interval=100)
    add_event(rec, {"output": x})
    rec.on_optimizer_step()
    assert rec.step == 1
    assert rec.active is True
    add_event(rec, {"output": x}, module="net2", call_index=0)
    rec.on_optimizer_step()
    assert rec.active is False
    assert "max_steps=2" in json.load(open(os.path.join(rec.rank_dir, "manifest.json")))["limit_reason"]


def test_signal_handler_flushes_then_chains(recorder, add_event, tiny_data):
    """需求 5：人为 kill / Ctrl-C 也要先把数据写完整，再把信号交回原 handler。"""
    x, _ = tiny_data()
    rec = recorder("sig", signals=False)                # 不真的改全局信号表
    add_event(rec, {"output": x}, module="net", call_index=0)
    handler = rec._make_handler(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)
    assert len(_lines(os.path.join(rec.rank_dir, EVENTS))) == 1
    exit_info = json.load(open(os.path.join(rec.rank_dir, "exit.json")))
    assert exit_info["reason"] == "signal:SIGINT"


def test_signal_handler_ignores_when_finalized(recorder):
    rec = recorder("sig2", signals=False)
    rec.finalize("first")
    handler = rec._make_handler(signal.SIGTERM)
    rec._prev_handlers[signal.SIGTERM] = signal.SIG_IGN
    handler(signal.SIGTERM, None)                       # 已收尾 → 不重复写，也不抛
    assert json.load(open(os.path.join(rec.rank_dir, "exit.json")))["reason"] == "first"


def test_stop_on_nan_raises_after_flush(recorder, add_event):
    rec = recorder("nan", stop_on_nan=True, flush_interval=100)
    add_event(rec, {"output": torch.tensor([1.0, float("nan")])})
    with pytest.raises(PProbeLimitReached):
        rec.record_nan("net.linear", "forward")
    assert len(_lines(os.path.join(rec.rank_dir, EVENTS))) == 1


# ----------------------------------------------------------------------
def test_call_index_and_budget(recorder, add_event, tiny_data):
    x, _ = tiny_data()
    rec = recorder("budget", max_calls_per_module=2, flush_interval=1)
    assert rec.call_index("net") == 0                   # 跨运行稳定的对齐 key（需求 6）
    assert rec.call_index("net") == 1
    assert rec.call_index("other") == 0
    # 预算在 hook 层执行，recorder 只统计已记录次数
    assert rec.budget_left("net") is True
    add_event(rec, {"output": x}, module="net", call_index=0)
    assert rec.budget_left("net") is True
    add_event(rec, {"output": x}, module="net", call_index=1)
    assert rec.budget_left("net") is False
    assert rec.budget_left("other") is True


def test_include_exclude_filters(recorder):
    rec = recorder("filt", include=[r"^encoder\."], exclude=[r"drop"], include_cls=[r"^(Linear|Layer)$"],
                   exclude_cls=["Dropout"])
    assert rec.want_module("encoder.linear", "Linear")
    assert not rec.want_module("decoder.linear", "Linear")
    assert not rec.want_module("encoder.drop", "Linear")
    assert not rec.want_module("encoder.linear", "Conv2d")
    assert not rec.want_module("encoder.linear", "Dropout")


def test_stack_ids_resolve_after_flush(recorder, add_event, tiny_data):
    """需求 4：事件里只存 id，完整堆栈在 stacks.json，且计数与引用一致。"""
    x, _ = tiny_data()
    rec = recorder("stk", flush_interval=1)
    for i in range(4):
        sid = rec.stacks.capture()
        assert add_event(rec, {"output": x}, module="net", call_index=i, stack_id=sid) is not None
    assert rec.stacks.unique == 1 and rec.stacks.get(sid)["count"] == 4
    rec.finalize("done")                                # force=True 把节流期内的计数补齐
    data = json.load(open(os.path.join(rec.rank_dir, "stacks.json")))
    assert data["count"] == 1 and data["total_hits"] == 4
    assert list(data["stacks"]) == [sid]
    res = Result(rec.root)
    rank0 = res.ranks[rec.rank]
    ev = list(rank0.iter_events())
    assert len(ev) == 4
    for e in ev:
        assert rank0.stack(e["stack_id"])["id"] == e["stack_id"]


def test_threads_do_not_lose_events(recorder, add_event, tiny_data):
    """autograd 的 backward hook 在引擎线程里回调，并发记录不能互相丢弃。"""
    x, _ = tiny_data()
    rec = recorder("threads", flush_interval=1000)
    n_threads, per_thread = 4, 10

    def work(k):
        for i in range(per_thread):
            add_event(rec, {"output": x}, module=f"net.{k}", call_index=i)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(n_threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rec.finalize("threads-done")
    seqs = [json.loads(ln)["seq"] for ln in _lines(os.path.join(rec.rank_dir, EVENTS))]
    assert len(seqs) == n_threads * per_thread
    assert len(set(seqs)) == len(seqs)                   # 序号不重复


def test_record_never_raises(recorder, add_event, tiny_data, monkeypatch):
    x, _ = tiny_data()
    rec = recorder("boom")
    monkeypatch.setattr("pprobe.recorder.dumps", lambda obj: (_ for _ in ()).throw(RuntimeError("x")))
    assert add_event(rec, {"output": x}) is None
    assert rec.counters()["errors"].get("record:RuntimeError") == 1


def test_forked_child_does_not_write(recorder, add_event, tiny_data):
    """DataLoader worker 等 fork 出的子进程不能覆盖父进程元信息。"""
    x, _ = tiny_data()
    rec = recorder("forked")
    add_event(rec, {"output": x})
    rec._forked_child()
    assert rec.active is False
    rec.finalize("child-exit")
    assert json.load(open(os.path.join(rec.rank_dir, "manifest.json")))["state"] == "running"


def test_other_pid_cannot_finalize(recorder, add_event, tiny_data):
    x, _ = tiny_data()
    rec = recorder("pid")
    add_event(rec, {"output": x})
    rec._owner_pid = os.getpid() + 12345
    assert rec.active is False
    rec.finalize("stale")
    assert not os.path.exists(os.path.join(rec.rank_dir, "exit.json"))


# ----------------------------------------------------------------------
def test_save_full_tensor(recorder, tiny_data):
    x, _ = tiny_data()
    rec = recorder("full", save_full=[r"^net"], flush_interval=1)
    saver = rec.make_full_saver("net.linear", "forward", 0)
    assert saver is not None
    info = saver(x)
    assert info["path"].startswith("rank0/tensors/") and info["path"].endswith("net.linear.pt")
    saved = torch.load(os.path.join(rec.root, info["path"]), weights_only=True)
    assert torch.equal(saved, x)
    # 二次同名不覆盖，也不重复写
    assert rec.make_full_saver("other", "forward", 0) is None
    rec2 = recorder("full2", save_full=[r"^net"], save_full_max_bytes=1024, save_full_every=2)
    assert rec2.make_full_saver("net.big", "forward", 0)(torch.zeros(4096))["skipped"].startswith("too-large")
    assert rec2.make_full_saver("net.even", "forward", 1)(torch.zeros(4)) is None      # 不被采样那一次
    assert rec2.make_full_saver("net.even", "forward", 2)(torch.zeros(4))["path"]


def test_env_and_run_json_written_once(recorder, add_event, tiny_data):
    x, _ = tiny_data()
    a = recorder("runjson", rank=0)
    b = recorder("runjson", rank=1)
    add_event(a, {"output": x})
    add_event(b, {"output": x})
    a.finalize("a")
    b.finalize("b")
    assert os.path.isfile(os.path.join(a.root, "run.json"))
    assert os.path.isfile(os.path.join(a.root, "HINTS.txt"))
    res = Result(a.root)
    assert res.rank_ids() == [0, 1] and res.total_events() == 2
    assert len(_lines(os.path.join(res.ranks[1].path, EVENTS))) == 1


def test_seed_is_recorded_for_reproducibility(recorder):
    """需求 3：随机采样未指定 seed 时，实际用的种子必须记进 manifest。"""
    rec = recorder("seed", sample_mode="random", sample_seed=None)
    man = json.load(open(os.path.join(rec.rank_dir, "manifest.json")))
    assert man["sampling"]["seed_from_user"] is False
    assert isinstance(man["sampling"]["seed_used"], int)
    assert rec.sampler.seed_used == man["sampling"]["seed_used"]
    rec2 = recorder("seed2", sample_mode="random", sample_seed=99)
    assert rec2.sampler.seed_used == 99 and rec2.seed_from_user is True


def test_unwritable_directory_disables_probe(tmp_path, make_config):
    bad = tmp_path / "ro" / "x"
    os.makedirs(tmp_path / "ro", exist_ok=True)
    os.chmod(tmp_path / "ro", 0o500)
    try:
        rec = Recorder(make_config(out_dir=str(bad)), rank=0, lazy=True)
        assert rec.active is False
        assert rec._disabled is True
    finally:
        os.chmod(tmp_path / "ro", 0o700)


def test_config_from_env_end_to_end(tmp_path, tiny_data, add_event):
    x, _ = tiny_data()
    os.environ.update({"PPROBE_ENABLE": "1", "PPROBE_OUT": str(tmp_path / "byenv"),
                       "PPROBE_SAMPLE_N": "4", "PPROBE_FLUSH_INTERVAL": "1"})
    cfg = Config.from_env()
    rec = Recorder(cfg, lazy=False)
    add_event(rec, {"output": x})
    rec.finalize("ok")
    assert rec.sampler.n == 4
    assert json.loads(_lines(os.path.join(rec.rank_dir, EVENTS))[0])["tensors"]["output"]["sample"]["n"] == 4
