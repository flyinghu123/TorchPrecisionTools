"""真实 torch 链路上的 hook 端到端验证（需求 2/3/4/5 的最终落地效果）。

与 test_hooks.py（只测路径推导等纯函数）不同，这里全部通过 ``pprobe.init`` 装好探针，
跑真的 forward/backward/optimizer，再从结果目录读回事件校验字段。
"""

from __future__ import annotations

import itertools
import os

import pytest
import torch

import pprobe
from pprobe.result import Result

DIM = 16
_OUT_SEQ = itertools.count()


class TinyNet(torch.nn.Module):
    def __init__(self, dim: int = DIM, eps: float = 1e-5):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * 2)
        self.norm = torch.nn.LayerNorm(dim * 2, eps=eps)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.act(self.norm(self.fc1(x)))
        return self.fc2(h) + x


class NanNet(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x, float("nan"))


def _data(batch: int = 4, dim: int = DIM, seed: int = 0, requires_grad: bool = False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, dim, generator=g)
    if requires_grad:
        x = x.requires_grad_(True)
    return x


def _reset() -> None:
    """同一进程里 ``init`` 是幂等的（复用已有 recorder），所以要彻底卸干净才能采下一段。"""
    from pprobe import bootstrap, hooks
    from pprobe import recorder as recorder_mod

    pprobe.stop("segment-end")
    hooks.uninstall()
    recorder_mod._RECORDER = None
    bootstrap._booted = False


def _collect(tmp_path, body, **cfg_kw):
    """装探针 → 跑 body → 收尾，返回 ``(recorder, Result, body 返回值)``。"""
    cfg_kw.setdefault("sample_n", 8)
    cfg_kw.setdefault("verbose", False)
    _reset()
    out = str(tmp_path / f"res{next(_OUT_SEQ)}")
    rec = pprobe.init(out_dir=out, **cfg_kw)
    assert rec is not None
    try:
        value = body()
    finally:
        _reset()
    return rec, Result(out), value


def _events(res, phase: str | None = None):
    if not res.ranks:
        return []
    evs = list(res.ranks[0].iter_events())
    return [e for e in evs if phase is None or e["phase"] == phase]


def _by_module(evs, module: str, phase: str | None = None):
    return [e for e in evs if e["module"] == module and (phase is None or e["phase"] == phase)]


# ----------------------------------------------------------------------
# forward
# ----------------------------------------------------------------------
def test_forward_hook_paths_and_summary(tmp_path):
    """需求 2/3/4：层级模块路径 + 槽位命名 + basic/stats/sample 三件套。"""
    model = TinyNet()
    x = _data()
    ref = model(x).detach().clone()
    _rec, res, out = _collect(tmp_path, lambda: model(x), hooks=("forward",))
    assert torch.allclose(out.detach(), ref)          # 探针不许改变数值

    evs = _events(res)
    assert [(e["module"], e["phase"]) for e in evs] == [
        ("TinyNet.fc1", "forward"),
        ("TinyNet.norm", "forward"),
        ("TinyNet.act", "forward"),
        ("TinyNet.fc2", "forward"),
        ("TinyNet", "forward"),
    ]
    assert [e["module_cls"] for e in evs] == ["Linear", "LayerNorm", "GELU", "Linear", "TinyNet"]
    assert [e["call_index"] for e in evs] == [0, 0, 0, 0, 0]
    assert [e["seq"] for e in evs] == sorted(e["seq"] for e in evs)

    fc1 = _by_module(evs, "TinyNet.fc1")[0]
    assert set(fc1["tensors"]) == {"input[0]", "output"}
    entry = fc1["tensors"]["input[0]"]
    assert entry["kind"] == "tensor"
    assert entry["basic"]["shape"] == [4, DIM] and entry["basic"]["dtype"] == "torch.float32"
    assert entry["basic"]["device"] == "cpu" and entry["basic"]["numel"] == 4 * DIM
    assert entry["stats"]["count"] == 4 * DIM
    for key in ("max", "min", "mean", "var", "std", "checksum", "nan_count", "inf_count"):
        assert key in entry["stats"], key
    assert entry["sample"]["n"] == 8 == len(entry["sample"]["vals"]) == len(entry["sample"]["idx"])
    assert entry["sample"]["idx_nd"] and len(entry["sample"]["idx_nd"][0]) == 2
    # 顶层模块同时看到残差结构签名与 grad 状态
    assert fc1["output_sig"] == "Tensor"
    assert fc1["grad_enabled"] is True
    assert _by_module(evs, "TinyNet")[0]["tensors"]["output"]["basic"]["shape"] == [4, DIM]


def test_stack_ids_are_deduplicated(tmp_path):
    """需求 4：堆栈用 id 表示，同一个 json 里 id → 完整堆栈，不重复存。"""
    model = TinyNet()
    x = _data()
    _rec, res, _out = _collect(tmp_path, lambda: model(x), hooks=("forward",))
    evs = _events(res)
    assert all(e["stack_id"] for e in evs)
    rank = res.ranks[0]
    inner = rank.stack(_by_module(evs, "TinyNet.fc1")[0]["stack_id"])
    outer = rank.stack(_by_module(evs, "TinyNet")[0]["stack_id"])
    assert inner and outer and inner != outer
    assert "test_hooks_e2e.py" in str(inner["frames"])
    # 两次相同调用链复用同一个 id
    _rec2, res2, _o2 = _collect(tmp_path, lambda: (model(x), model(x)), hooks=("forward",),
                                sample_mode="off")
    evs2 = [e for e in _events(res2) if e["module"] == "TinyNet.fc1"]
    assert len(evs2) == 2 and evs2[0]["stack_id"] == evs2[1]["stack_id"]
    assert evs2[1]["call_index"] == 1


# ----------------------------------------------------------------------
# backward
# ----------------------------------------------------------------------
def test_backward_module_hook_slots(tmp_path):
    """需求 2：反向拿到模块边界梯度，并用 fwd_seq 回填对应的那次前向。"""
    model = TinyNet()
    x = _data(requires_grad=True)

    def body():
        out = model(x)
        out.square().mean().backward()
        return out

    rec, res, _out = _collect(tmp_path, body, hooks=("forward", "backward"))
    evs = _events(res)
    fwd = {e["module"]: e for e in evs if e["phase"] == "forward"}
    bwd = {e["module"]: e for e in evs if e["phase"] == "backward"}
    assert {"TinyNet.fc1", "TinyNet.fc2", "TinyNet.norm"} <= set(bwd)

    b2 = bwd["TinyNet.fc2"]
    assert b2["module_cls"] == "Linear"
    assert b2["bwd_src"] == "module-hook"
    assert b2["bwd_index"] == 1 == rec.bwd_index
    assert b2["fwd_seq"] == fwd["TinyNet.fc2"]["seq"]
    assert b2["fwd_stack_id"] == fwd["TinyNet.fc2"]["stack_id"]
    assert b2["call_index"] == fwd["TinyNet.fc2"]["call_index"]
    for slot, shape in (("grad_input[0]", [4, 2 * DIM]), ("grad_output[0]", [4, DIM])):
        assert b2["tensors"][slot]["kind"] == "tensor"
        assert b2["tensors"][slot]["basic"]["shape"] == shape
    assert rec.counters()["backward_calls"] == 1
    # 反向事件顺序由 autograd 引擎决定，但一定排在所有前向之后
    assert min(e["seq"] for e in evs if e["phase"] == "backward") > max(
        e["seq"] for e in evs if e["phase"] == "forward")


def test_backward_call_index_points_at_its_forward(tmp_path):
    """多轮迭代下反向的 call_index 必须指回触发它的那次前向（最容易 off-by-one 的地方）。"""
    model = TinyNet()
    x = _data(requires_grad=True)

    def body():
        for _ in range(3):
            model(x).square().mean().backward()

    _rec, res, _out = _collect(tmp_path, body, hooks=("forward", "backward"))
    evs = _events(res)
    fwd_seq = {(e["module"], e["call_index"]): e["seq"] for e in evs if e["phase"] == "forward"}
    bwd = [e for e in evs if e["phase"] == "backward"]
    assert len(bwd) >= 12
    for e in bwd:
        assert e["fwd_seq"] == fwd_seq[(e["module"], e["call_index"])], e
    assert sorted(e["call_index"] for e in _by_module(bwd, "TinyNet.fc2")) == [0, 1, 2]


def test_backward_tensor_mode(tmp_path):
    """``bwd_mode=tensor`` 走 Tensor.register_hook 兜底路径（模块无参数时唯一选择）。"""
    model = TinyNet()
    x = _data(requires_grad=True)

    def body():
        model(x).square().mean().backward()

    _rec, res, _out = _collect(tmp_path, body, hooks=("forward", "backward"), bwd_mode="tensor")
    evs = _events(res, "backward")
    assert evs
    assert {e["bwd_src"] for e in evs} == {"tensor-hook"}
    for e in evs:
        assert list(k for k in e["tensors"] if k.startswith("grad_of:"))
        # 同一个张量可能被多个模块当边界张量 hook（x 既是 TinyNet 的输入也是 fc1 的），
        # 所以 occurrence 可以 >0，但不能超过 4 次上限
        assert 0 <= e["occurrence"] <= 3
        assert e["fwd_seq"] is not None
    # tensor 模式下不注册模块级 full hook，但每个被调用的模块都不该缺席
    assert {e["module"] for e in evs} >= {"TinyNet.fc1", "TinyNet.norm", "TinyNet.act", "TinyNet.fc2"}


def test_backward_skipped_when_no_grad(tmp_path):
    model = TinyNet()
    x = _data()
    with torch.no_grad():
        _rec, res, out = _collect(tmp_path, lambda: model(x), hooks=("forward", "backward"))
    assert out is not None
    assert [e["phase"] for e in _events(res)] == ["forward"] * 5
    assert all(e["grad_enabled"] is False for e in _events(res))


# ----------------------------------------------------------------------
# functional / optimizer
# ----------------------------------------------------------------------
def test_functional_hook_events(tmp_path):
    """``func`` hook 覆盖没有 Module 的算子，并带上当前模块上下文。"""
    model = TinyNet()
    x = _data()
    _rec, res, _out = _collect(tmp_path, lambda: model(x), hooks=("forward", "func"),
                               func_targets=("linear", "layer_norm", "gelu"))
    evs = _events(res)
    funcs = [e for e in evs if e["module"].startswith("functional.")]
    assert {e["module"] for e in funcs} == {"functional.linear", "functional.layer_norm",
                                            "functional.gelu"}
    assert {e["module_cls"] for e in funcs} == {"Function"}
    owner = {(e["module"], e["in_module"]) for e in funcs}
    assert ("functional.linear", "TinyNet.fc1") in owner
    assert ("functional.linear", "TinyNet.fc2") in owner
    assert ("functional.layer_norm", "TinyNet.norm") in owner
    assert ("functional.gelu", "TinyNet.act") in owner
    # 回归：路径 join 错误会产出 "TinyNet.TinyNet.fc1" 这种重复前缀
    assert "TinyNet.TinyNet" not in str([e["module"] + "|" + e.get("in_module", "") for e in evs])
    lin = next(e for e in funcs if e["module"] == "functional.linear")
    assert {"input[0]", "input[1]", "output"} <= set(lin["tensors"])
    assert [e["call_index"] for e in funcs if e["module"] == "functional.linear"] == [0, 1]


def test_functional_hook_backward_and_kwargs(tmp_path):
    """函数级算子也用 register_hook 拿到输出梯度，并把 kwargs 记成槽位。"""
    model = TinyNet()
    x = _data(requires_grad=True)

    def body():
        model(x).square().mean().backward()

    _rec, res, _out = _collect(tmp_path, body, hooks=("forward", "backward", "func"),
                               func_targets=("layer_norm",))
    fb = [e for e in _events(res, "backward") if e["module"] == "functional.layer_norm"]
    assert fb and fb[0]["bwd_src"] == "tensor-hook"
    assert fb[0]["tensors"]["grad_of:output"]["kind"] == "tensor"

    # torch.nn.functional 直接调用（不经过任何 Module）→ in_module 为空但仍被记录
    _rec2, res2, _o2 = _collect(tmp_path, lambda: torch.nn.functional.gelu(x, approximate="tanh"),
                               hooks=("forward", "func"), func_targets=("gelu",))
    ev = _events(res2)[0]
    assert ev["module"] == "functional.gelu" and ev["in_module"] == ""
    # 字符串走 short_repr（带引号），跨运行对比时更稳
    assert ev["tensors"]["kwargs.approximate"] == {"kind": "scalar", "value": "'tanh'"}


def test_optimizer_step_labels_and_limit(tmp_path):
    """需求 5：optimizer step 给事件打 step 标签，``max_steps`` 到点落盘收尾。"""
    model = TinyNet()
    x = _data()

    def body(steps=2):
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            opt.step()
        return opt

    rec, res, _opt = _collect(tmp_path, body, hooks=("forward", "backward", "optim"))
    assert rec.step == 2
    evs = _events(res)
    assert min(e["step"] for e in evs) == 0 and max(e["step"] for e in evs) == 1
    assert rec.counters()["steps"] == 2
    man = res.ranks[0].manifest
    assert man["state"] == "finalized"

    # 再开一个 recorder 时 step 计数必须重新生效（optimizer 壳要被正确卸载）
    rec2, _res2, _o2 = _collect(tmp_path, lambda: body(1), hooks=("forward", "optim"),
                                sample_mode="off")
    assert rec2.step == 1

    rec3, res3, _o3 = _collect(tmp_path, lambda: body(3), hooks=("forward", "backward", "optim"),
                              sample_mode="off", max_steps=1)
    assert rec3.step == 1 and not rec3.active
    man3 = res3.ranks[0].manifest
    assert "max_steps=1" in man3["limit_reason"] and man3["counters"]["steps"] == 1


def test_mark_step_without_optimizer(tmp_path):
    """自定义训练循环（megatron/deepspeed）里手动打点。"""
    model = TinyNet()
    x = _data()

    def body():
        for _ in range(2):
            model(x)
            pprobe.mark_step()

    rec, res, _ = _collect(tmp_path, body, hooks=("forward",))
    assert rec.step == 2
    assert [e["step"] for e in _events(res)] == [0] * 5 + [1] * 5


# ----------------------------------------------------------------------
# 过滤与配额（需求 5）
# ----------------------------------------------------------------------
def test_include_exclude_filters(tmp_path):
    model = TinyNet()
    x = _data()
    _rec, res, _out = _collect(tmp_path, lambda: model(x), hooks=("forward",), include_cls=(r"Linear",))
    assert {e["module_cls"] for e in _events(res)} == {"Linear"}

    _rec, res, _out = _collect(tmp_path, lambda: model(x), hooks=("forward",),
                               exclude_cls=(r"GELU", r"TinyNet$"))
    mods = [e["module"] for e in _events(res)]
    assert "TinyNet.act" not in mods and "TinyNet" not in mods and "TinyNet.fc1" in mods

    _rec, res, _out = _collect(tmp_path, lambda: model(x), hooks=("forward",), include=(r"\.fc2$",))
    assert [e["module"] for e in _events(res)] == ["TinyNet.fc2"]


def test_max_calls_per_module_budget(tmp_path):
    """热点模块只记前 N 次调用（``PPROBE_MAX_CALLS_PER_MODULE``）。"""
    model = TinyNet()
    x = _data()

    def body():
        for _ in range(3):
            model(x)

    _rec, res, _out = _collect(tmp_path, body, hooks=("forward",), max_calls_per_module=2)
    evs = _events(res)
    counts: dict[str, int] = {}
    for e in evs:
        counts[e["module"]] = counts.get(e["module"], 0) + 1
    assert counts == {m: 2 for m in counts}
    assert sorted(e["call_index"] for e in _by_module(evs, "TinyNet.fc1")) == [0, 1]


def test_max_events_stops_recording_but_not_training(tmp_path):
    """需求 5：达到上限先落盘收尾，之后的前向走原始路径，训练结果不受影响。"""
    model = TinyNet()
    x = _data()
    ref = model(x).detach().clone()
    _rec, res, out = _collect(tmp_path, lambda: (model(x), model(x)), hooks=("forward",),
                              max_events=3)
    assert torch.allclose(out[1].detach(), ref)
    evs = _events(res)
    assert len(evs) == 3 and [e["seq"] for e in evs] == [1, 2, 3]
    man = res.ranks[0].manifest
    assert man["state"] == "finalized" and "max_events=3" in man["limit_reason"]


def test_disable_module_skips_only_itself(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(DIM, DIM), torch.nn.ReLU(), torch.nn.Linear(DIM, DIM))
    x = _data()
    pprobe_out = str(tmp_path / "res")
    pprobe.init(out_dir=pprobe_out, verbose=False, hooks=("forward",))
    pprobe.disable_module(model[1])
    model(x)
    pprobe.stop("done")
    mods = [e["module"] for e in _events(Result(pprobe_out))]
    assert "Sequential.1" not in mods
    assert {"Sequential.0", "Sequential.2", "Sequential"} == set(mods)


# ----------------------------------------------------------------------
# NaN 中断与自身容错
# ----------------------------------------------------------------------
def test_stop_on_nan_interrupts_forward(tmp_path):
    """``PPROBE_STOP_ON_NAN=1``：前向发现 NaN 直接把异常抛回用户代码（配合 pdb 定位）。"""
    from pprobe.recorder import PProbeLimitReached

    model = NanNet()
    x = _data()

    def body():
        with pytest.raises(PProbeLimitReached, match="NaN"):
            model(x)
        return "raised"

    _rec, res, out = _collect(tmp_path, body, hooks=("forward",), stop_on_nan=True)
    assert out == "raised"
    evs = _events(res)
    assert len(evs) == 1 and evs[0]["module"] == "NanNet"
    st = evs[0]["tensors"]["output"]["stats"]
    assert st["nan_count"] == 4 * DIM and st["count"] == 0 and st["all_nonfinite"] is True
    # 抛出前已经 flush，一条数据都不丢
    assert os.path.isfile(os.path.join(res.ranks[0].path, "events.jsonl"))
    assert res.ranks[0].manifest["state"] == "finalized"


def test_nan_recorded_without_stop(tmp_path):
    model = NanNet()
    x = _data()
    _rec, res, out = _collect(tmp_path, lambda: model(x), hooks=("forward",))
    assert torch.isnan(out).all()
    evs = _events(res)
    assert len(evs) == 1
    assert evs[0]["tensors"]["output"]["stats"]["nan_count"] == 4 * DIM


def test_probe_internal_error_does_not_break_training(tmp_path, monkeypatch):
    """探针自己出错时必须吞掉并记账，训练结果与不装探针完全一致。"""
    from pprobe import hooks

    model = TinyNet()
    x = _data()
    ref = model(x).detach().clone()

    def boom(*_a, **_kw):
        raise RuntimeError("故意炸")

    monkeypatch.setattr(hooks, "_make_collector", boom)
    rec, res, out = _collect(tmp_path, lambda: model(x), hooks=("forward",))
    assert torch.allclose(out.detach(), ref)
    assert not _events(res)
    assert any(k.startswith("fwd") for k in rec.counters()["errors"])
