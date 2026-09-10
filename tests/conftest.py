"""pytest 公共夹具。

探针是有进程级状态的（``torch.nn.Module.__call__`` 被替换、``Recorder`` 单例、
``sys.meta_path`` 上的 watcher），所以每个用例前后都必须彻底还原，否则用例之间
会互相污染（例如第二个用例拿不到干净的 recorder，或 functional 调用序号接着涨）。
"""

from __future__ import annotations

import os

import pytest
import torch


# ----------------------------------------------------------------------
# 隔离
# ----------------------------------------------------------------------
def _release() -> None:
    from pprobe import bootstrap, hooks
    from pprobe import recorder as recorder_mod

    try:
        hooks.uninstall()
    except Exception:  # pragma: no cover - 兜底
        pass
    hooks._FUNC_CALLS.clear()
    hooks._GRAD_OCCURRENCE.clear()
    rec = recorder_mod.get_recorder()
    if rec is not None:
        try:
            rec.finalize("test-teardown")
        except Exception:  # pragma: no cover
            pass
    recorder_mod._RECORDER = None
    bootstrap._booted = False
    if bootstrap._watcher is not None:
        bootstrap._watcher.detach()
        bootstrap._watcher = None


@pytest.fixture(autouse=True)
def isolate():
    """清掉 PPROBE_* 环境变量，用例结束后还原全局状态。"""
    saved = {k: v for k, v in os.environ.items() if k.startswith("PPROBE_")}
    for k in saved:
        del os.environ[k]
    _release()
    yield
    _release()
    for k in [k for k in os.environ if k.startswith("PPROBE_")]:
        del os.environ[k]
    os.environ.update(saved)


# ----------------------------------------------------------------------
# 构造结果目录
# ----------------------------------------------------------------------
@pytest.fixture
def make_config():
    from pprobe.config import Config

    def _make(**kw):
        base = {"enable": True, "sample_n": 8, "sample_mode": "uniform", "verbose": False}
        base.update(kw)
        return Config(**base)

    return _make


@pytest.fixture
def recorder(tmp_path, make_config):
    """建一个（已打开的）Recorder，用例结束后 finalize。"""
    from pprobe.recorder import Recorder

    made: list = []

    def _make(name="run", rank=0, lazy=False, **cfg_kw):
        cfg_kw.setdefault("out_dir", str(tmp_path / name))
        rec = Recorder(make_config(**cfg_kw), rank=rank, lazy=lazy)
        if not lazy:
            assert rec._ensure_open(), "结果目录创建失败"
        made.append(rec)
        return rec

    yield _make
    for r in made:
        r.finalize("fixture-teardown")


@pytest.fixture
def slot_collector():
    from pprobe.events import SlotCollector

    def _make(rec, full_saver=None):
        return SlotCollector(rec.cfg, rec.sampler, full_saver=full_saver)

    return _make


@pytest.fixture
def add_event(slot_collector):
    """把 ``{槽位名: 值}`` 通过真实采集链路打包成一条事件。"""

    def _add(rec, tensors: dict, phase="forward", module="net.linear", cls="Linear",
             call_index=0, stack_id=None, extra=None):
        c = slot_collector(rec)
        for name, value in tensors.items():
            c.add(name, value)
        return rec.record(phase, module, cls, c.slots, stack_id=stack_id,
                          call_index=call_index, extra=extra)

    return _add


@pytest.fixture
def build_run(tmp_path, recorder, add_event):
    """按 spec 列表生成一个标准结果目录，返回其路径。

    spec 元素形如 ``{"module": "net", "tensors": {"input[0]": t}, "call_index": 0}``。
    """

    def _build(name, specs, rank=0, **cfg_kw):
        rec = recorder(name=name, rank=rank, **cfg_kw)
        for i, spec in enumerate(specs):
            kw = {k: v for k, v in spec.items() if k not in ("tensors", "stack_frames")}
            kw.setdefault("call_index", i)
            if spec.get("stack_frames"):
                sid = rec.stacks.intern(spec["stack_frames"])
                kw["stack_id"] = sid
            add_event(rec, spec.get("tensors", {}), **kw)
        rec.finalize("built")
        return os.path.join(str(tmp_path), name)

    return _build


# ----------------------------------------------------------------------
# 小模型（端到端用例共用）
# ----------------------------------------------------------------------
class TinyNet(torch.nn.Module):
    """两层 MLP + LayerNorm，够用来触发 forward/backward/functional 三条链路。"""

    def __init__(self, dim: int = 16, eps: float = 1e-5):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim * 2)
        self.norm = torch.nn.LayerNorm(dim * 2, eps=eps)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.act(self.norm(self.fc1(x)))
        return self.fc2(h) + x


@pytest.fixture
def tiny_data():
    def _make(batch=4, dim=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(batch, dim, generator=g), torch.randn(batch, dim, generator=g))

    return _make
