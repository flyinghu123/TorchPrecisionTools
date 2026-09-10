"""需求 8 的端到端验证：torchrun 起多进程 → 分 rank 存储 → 分 rank 对比定位坏 rank。

全部用真实子进程（``python -m torch.distributed.run``），因为只有这样才能确认
``.pth`` 注入在 launcher 拉起的每个 worker 里都生效、rank 检测取到的是 torchrun 注入的
``RANK``，以及 rank0 的收尾文件不会被其它 rank 覆盖。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from pprobe import inject
from pprobe.cli import main
from pprobe.compare import Comparator, Options
from pprobe.result import Result

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "examples", "ddp_simple.py")
NPROC = 2


def _skip_reason() -> str | None:
    if not inject.find_existing():
        return "当前解释器未安装 pprobe.pth，多进程注入无法验证"
    d = torch.distributed
    if not (d.is_gloo_available() or d.is_nccl_available()):
        return "torch 未编译 gloo/nccl 后端"
    return None


def _env(out_dir: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PPROBE_")}
    env.update({
        "PPROBE_ENABLE": "1",
        "PPROBE_OUT": out_dir,
        "PPROBE_VERBOSE": "0",
        "PPROBE_SAMPLE_N": "12",
        "PPROBE_FLUSH_INTERVAL": "25",
        "OMP_NUM_THREADS": "1",
        "PYTHONWARNINGS": "ignore",
    })
    return env


def _torchrun(out_dir: str, extra: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", str(NPROC), SCRIPT, *extra]
    return subprocess.run(cmd, env=_env(out_dir), capture_output=True, text=True,
                          timeout=900, cwd=os.path.dirname(out_dir))


@pytest.fixture(scope="module")
def ddp(tmp_path_factory):
    """跑两遍 2-rank 训练：base 正常，bad 让 rank1 的 LayerNorm eps 变大 10 倍。"""
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    root = tmp_path_factory.mktemp("ddp")
    base, bad = str(root / "base"), str(root / "bad")
    for out, extra in ((base, ["--steps", "5"]), (bad, ["--steps", "5", "--bad-rank", "1"])):
        proc = _torchrun(out, extra)
        assert proc.returncode == 0, proc.stdout[-3000:] + "\n" + proc.stderr[-3000:]
        for r in range(NPROC):
            assert f"dist-ok rank={r} world={NPROC}" in proc.stdout
    return {"root": str(root), "base": base, "bad": bad}


# ----------------------------------------------------------------------
# 分 rank 存储
# ----------------------------------------------------------------------
def test_each_rank_writes_its_own_dir(ddp):
    base = ddp["base"]
    res = Result.load(base)
    assert res.rank_ids() == [0, 1]
    assert {os.path.basename(r.path) for r in res.ranks.values()} == {"rank0", "rank1"}
    # launcher 进程自己不跑前向，绝不能多出 no-rank/rank0 目录
    assert sorted(os.listdir(base)) == ["HINTS.txt", "rank0", "rank1", "run.json"]

    run = json.load(open(os.path.join(base, "run.json"), encoding="utf-8"))
    assert run["world_size"] == NPROC and run["ranks"] == [0, 1]

    pids, modules = set(), {}
    for rank, rr in res.ranks.items():
        man = rr.manifest
        assert man["rank"] == rank and man["world_size"] == NPROC
        assert man["state"] == "finalized" and man["counters"]["events"] > 0
        assert os.path.isfile(os.path.join(rr.path, "env.json"))
        assert os.path.isfile(os.path.join(rr.path, "stacks.json"))
        env = rr.env
        assert env["rank"] == rank and env["world_size"] == NPROC
        assert env["cmdline"] and "ddp_simple.py" in env["cmdline"]
        pids.add(env["pid"])
        evs = list(rr.iter_events())
        assert {e["phase"] for e in evs} == {"forward", "backward"}
        assert all(e["rank"] == rank for e in evs)
        modules[rank] = {e["module"] for e in evs}
    assert len(pids) == NPROC                        # 每个 rank 一个进程
    # 两个 rank 的模型结构一致（DDP 包了一层，路径都从 DistributedDataParallel 起）
    assert modules[0] == modules[1]
    wrapped = {m for m in modules[0] if m.startswith("DistributedDataParallel")}
    assert {"DistributedDataParallel.module.norm",
            "DistributedDataParallel.module.fc2"} <= wrapped
    # 没被 DDP 包住的部分（loss 模块）仍按自己的类名记录
    assert modules[0] - wrapped == {"MSELoss"}


def test_ranks_hold_different_data(ddp):
    """分 rank 存的意义所在：同一步、同一模块，两个 rank 的输入本来就不该相同。"""
    res = Result.load(ddp["base"])
    key = ("forward", "DistributedDataParallel.module.norm", 0)

    def first_vals(rank):
        rank_res = res.ranks[rank]
        for ev in rank_res.iter_events():
            if (ev["phase"], ev["module"], ev["call_index"]) == key:
                return ev["tensors"]["input[0]"]
        raise AssertionError(f"rank{rank} 没有 {key}")

    a, b = first_vals(0), first_vals(1)
    assert a["basic"]["shape"] == b["basic"]["shape"]
    assert a["sample"]["vals"] != b["sample"]["vals"]
    assert a["stats"]["checksum"] != b["stats"]["checksum"]
    assert a["sample"]["n"] == 12 == len(a["sample"]["vals"])


# ----------------------------------------------------------------------
# 分 rank 对比
# ----------------------------------------------------------------------
def _compare(ddp, **kw):
    opts = Options(rtol=1e-5, top=400, **kw)
    return Comparator(Result.load(ddp["base"]), Result.load(ddp["bad"]), opts).run()


def test_compare_reports_every_rank_pair(ddp):
    rep = _compare(ddp)
    assert set(rep["ranks"]) == {"0", "1"}
    assert rep["missing_ranks"] == {"a_only": [], "b_only": []}
    assert rep["totals"]["events_compared"] > 40
    assert rep["totals"]["divergent_events"] > 0

    fd = rep["totals"]["first_divergence"]
    assert fd["rank"] == 1, "坏的是 rank1，全局最早发散点必须落在它身上"
    assert fd["key"].endswith("module.norm#0") and fd.get("stack_a")
    seq1 = rep["ranks"]["1"]["first_divergence"]["seq_a"]
    seq0 = rep["ranks"]["0"]["first_divergence"]["seq_a"]
    assert seq1 < seq0, "rank1 第一次前向就发散，rank0 要到参数更新之后才被带偏"
    assert "sample" in rep["ranks"]["1"]["first_divergence"]["kinds"]
    for k in ("0", "1"):
        one = rep["ranks"][k]
        assert one["events_compared"] > 0
        assert one["events_only_a"] == 0 and one["events_only_b"] == 0


def test_compare_identical_runs_has_no_divergence(ddp):
    """同一份数据再跑一遍必须完全一致，否则说明对比本身不稳定。"""
    out = os.path.join(ddp["root"], "base2")
    proc = _torchrun(out, ["--steps", "5"])
    assert proc.returncode == 0, proc.stderr[-2000:]
    rep = Comparator(Result.load(ddp["base"]), Result.load(out), Options(top=50)).run()
    assert rep["totals"]["divergent_events"] == 0
    assert rep["totals"]["first_divergence"] is None


def test_cli_compare_writes_file_with_both_ranks(ddp, capsys):
    md = os.path.join(ddp["root"], "diff.md")
    rc = main(["compare", ddp["base"], ddp["bad"], "--out", md, "--fail-if-diff"])
    out = capsys.readouterr().out
    assert rc == 1                                    # 有差异 → 退出码 1
    assert "最早发散点 rank1" in out
    text = open(md, encoding="utf-8").read()
    assert "### 各 rank 最早发散点" in text
    assert "`forward`" in text and "rank0" in text and "rank1" in text
    # seq 排序下两个 rank 的明细都要能看到（不能被 rank0 吃满 top）
    assert text.count("rank1") > 0 and "| rank1 |" in text


def test_stack_query_per_rank(ddp, capsys):
    res = Result.load(ddp["base"])
    sid = next(iter(res.ranks[1].stacks))
    rc = main(["stack", sid, "--result", ddp["base"], "--rank", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"=== rank1" in out and sid in out
    assert "ddp_simple.py" in out
    # 反查：坏模块的堆栈 id 只在 rank 上展开，不串号
    rc = main(["stack", "--result", ddp["base"], "--module", r"module\.norm$", "--rank", "1"])
    out = capsys.readouterr().out
    assert rc == 0 and "rank1/s" in out and "rank0/" not in out


# ----------------------------------------------------------------------
# 中断保护（需求 5 在多进程下的表现）
# ----------------------------------------------------------------------
def test_sigterm_still_flushes_every_rank(ddp):
    out = os.path.join(ddp["root"], "killed")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", str(NPROC), SCRIPT, "--steps", "300", "--dim", "512"]
    proc = subprocess.Popen(cmd, env=_env(out), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=ddp["root"], start_new_session=True)
    events = {r: os.path.join(out, f"rank{r}", "events.jsonl") for r in range(NPROC)}
    deadline = time.time() + 180
    try:
        while time.time() < deadline:
            if all(os.path.isfile(p) and sum(1 for _ in open(p, encoding="utf-8")) >= 30
                   for p in events.values()):
                break
            time.sleep(0.5)
        else:
            pytest.fail("等不到两个 rank 写出事件")
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:  # pragma: no cover
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=60)

    for rank, path in events.items():
        lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
        assert lines, f"rank{rank} 没有落盘任何事件"
        json.loads(lines[-1])                          # 最后一行也必须完整
        man = json.load(open(os.path.join(os.path.dirname(path), "manifest.json"), encoding="utf-8"))
        assert man["state"] == "finalized" and man["rank"] == rank
        assert man["counters"]["events"] == len(lines)
        assert "signal" in man["limit_reason"] or "SIG" in man["limit_reason"].upper()
