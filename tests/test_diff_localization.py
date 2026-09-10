"""两次运行/两种配置的精度差异定位（需求 6 的实战效果）。

用 ``examples/mlp_simple.py`` 跑两遍，唯一区别是 LayerNorm 的 eps（真实场景里
「两个框架/两个版本默认 eps 不同」「CPU 与 GPU kernel 实现不同」都是这类问题），
检查探针能否把根因指到第一个发散的模块，而不是发散的 loss。

GPU 相关的对比（``--device cuda``）需要 sm_70 以上的卡，本机跑不了就不假装验证：
没有可用 CUDA 时这些用例会 skip。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from pprobe import inject
from pprobe.cli import main
from pprobe.compare import Comparator, Options
from pprobe.result import Result

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "examples", "mlp_simple.py")


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool((torch.zeros(1, device="cuda") + 1).item() == 1)
    except Exception:
        return False


def _run(out_dir: str, extra: list[str], enable: bool = True) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PPROBE_")}
    if enable:
        env.update({"PPROBE_ENABLE": "1", "PPROBE_OUT": out_dir, "PPROBE_VERBOSE": "0",
                    "PPROBE_SAMPLE_N": "16", "PPROBE_SEED": "2026"})
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONWARNINGS"] = "ignore"
    return subprocess.run([sys.executable, SCRIPT, *extra], env=env, capture_output=True,
                          text=True, timeout=600, cwd=os.path.dirname(out_dir) or os.getcwd())


@pytest.fixture(scope="module")
def eps_pair(tmp_path_factory):
    if not inject.find_existing():
        pytest.skip("当前解释器未安装 pprobe.pth")
    root = str(tmp_path_factory.mktemp("eps"))
    args = ["--device", "cpu", "--steps", "2", "--batch", "8", "--seed", "42"]
    a, b = os.path.join(root, "A"), os.path.join(root, "B")
    pa = _run(a, args + ["--eps", "1e-5"])
    pb = _run(b, args + ["--eps", "1e-3"])
    assert pa.returncode == 0, pa.stderr[-2000:]
    assert pb.returncode == 0, pb.stderr[-2000:]
    la = [ln for ln in pa.stdout.splitlines() if ln.startswith("mean_loss")][-1]
    lb = [ln for ln in pb.stdout.splitlines() if ln.startswith("mean_loss")][-1]
    assert la != lb, "两次运行结果一样，这个用例就没有意义了"
    return {"root": root, "a": a, "b": b, "args": args}


def test_probe_off_creates_nothing(tmp_path):
    """零影响承诺：不设 PPROBE_ENABLE 时不能留下任何目录/文件。"""
    out = tmp_path / "should-not-exist"
    proc = _run(str(out), ["--device", "cpu", "--steps", "1", "--seed", "1"], enable=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "pprobe 已通过 .pth 自动注入" not in proc.stdout
    assert not out.exists()


def test_first_divergence_points_at_layernorm(eps_pair):
    ra, rb = Result.load(eps_pair["a"]), Result.load(eps_pair["b"])
    assert ra.rank_ids() == [0] and rb.rank_ids() == [0]
    rep = Comparator(ra, rb, Options(rtol=1e-5, top=500)).run()
    totals = rep["totals"]
    assert totals["divergent_events"] > 0
    fd = totals["first_divergence"]
    assert fd and fd["rank"] == 0
    # 根因必须是被改了 eps 的那个模块，而且是它的第一次调用（第 0 层 block）
    root_key = fd["key"]
    assert root_key.startswith("forward:") and ".blocks.0.norm" in root_key, root_key
    assert root_key.endswith("#0")
    assert isinstance(fd["max_abs"], (int, float)) and fd["max_abs"] > 1e-6

    # 差异向下游传播：head / 顶层模型 / 反向都被带偏，但按 seq 排序它们都在 norm 之后
    mods = [d["module"] for d in rep["divergences"]]
    assert any(m.endswith("head") for m in mods), mods
    assert mods[0] == root_key.split(":", 1)[1].split("#")[0]
    assert mods[-1] != mods[0]
    # 结构没变（只是数值差异），所以不该出现「只在一边存在」
    assert totals["events_only_a"] == 0 and totals["events_only_b"] == 0
    assert totals["stack_diff"] == 0


def test_tight_and_loose_tolerance_agree_on_root_cause(eps_pair):
    """atol/rtol 会改变差异数量，但最早发散点不该变。"""
    ra, rb = Result.load(eps_pair["a"]), Result.load(eps_pair["b"])
    keys = set()
    for atol, rtol in ((0.0, 1e-9), (1e-6, 1e-5), (1e-4, 1e-3)):
        rep = Comparator(ra, rb, Options(atol=atol, rtol=rtol, top=500)).run()
        fd = rep["totals"]["first_divergence"]
        if fd:
            keys.add(fd["key"])
    assert keys == {".blocks.0.norm#0"} or all(".blocks.0.norm" in k for k in keys), keys


def test_report_and_query_on_collected_run(eps_pair, capsys):
    assert main(["report", eps_pair["a"]]) == 0
    out = capsys.readouterr().out
    assert "rank0" in out
    assert main(["query", eps_pair["a"], "--module", r"blocks\.0\.norm$", "--json"]) == 0
    assert "blocks.0.norm" in capsys.readouterr().out
    # 采样种子固定 → 两次运行的下标序列一致，采样值才能逐元素对比
    man = Result.load(eps_pair["a"]).ranks[0].manifest
    assert man["sampling"]["seed_user"] == 2026 and man["sampling"]["n"] == 16


@pytest.mark.skipif(not _cuda_usable(), reason="本机没有可用的 CUDA kernel")
def test_cpu_vs_cuda_first_divergence_is_reported(eps_pair):
    """跨平台对比：同一份权重/数据，cpu 与 cuda 的差异要能定位到最早发散的模块。"""
    root = eps_pair["root"]
    cpu, gpu = os.path.join(root, "cpu"), os.path.join(root, "cuda")
    base = ["--steps", "2", "--batch", "8", "--seed", "42", "--eps", "1e-5"]
    for out, dev in ((cpu, "cpu"), (gpu, "cuda")):
        proc = _run(out, base + ["--device", dev])
        assert proc.returncode == 0, proc.stderr[-2000:]
    rep = Comparator(Result.load(cpu), Result.load(gpu), Options(atol=1e-7, rtol=1e-5, top=500)).run()
    fd = rep["totals"]["first_divergence"]
    assert fd is not None, "cpu/cuda 至少应有浮点实现差异"
    assert fd["key"].startswith("forward:")
