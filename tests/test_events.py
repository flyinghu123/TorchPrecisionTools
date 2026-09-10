"""槽位收集器（需求 4 的落盘结构）：命名规则、深度/数量限制、标量与异常兜底。"""

from __future__ import annotations

import torch
import torch.nn as nn

from pprobe.events import (SlotCollector, apply_include_exclude, collect_args, collect_params,
                           has_nan, structure_sig)
from pprobe.util import RegexMatcher
from pprobe.sampling import TensorSampler


def _collector(cfg, **kw):
    return SlotCollector(cfg, TensorSampler(cfg.sample_mode, cfg.sample_n, 1), **kw)


# ----------------------------------------------------------------------
def test_tensor_slot_shape(make_config):
    cfg = make_config()
    c = _collector(cfg)
    c.add("input[0]", torch.randn(2, 3))
    entry = c.slots["input[0]"]
    assert entry["kind"] == "tensor"
    assert entry["basic"]["shape"] == [2, 3] and entry["basic"]["dtype"] == "torch.float32"
    assert entry["stats"]["count"] == 6
    assert entry["sample"]["n"] == 6                 # numel < sample_n 时全采
    assert c.n_tensors == 1 and not c.truncated


def test_config_switches_drop_sections(make_config):
    c = _collector(make_config(stats=False, sample_mode="off"))
    c.add("output", torch.randn(4))
    entry = c.slots["output"]
    assert "stats" not in entry and "sample" not in entry
    assert "basic" in entry

    c2 = _collector(make_config(full_hash=True))
    c2.add("output", torch.randn(4))
    assert c2.slots["output"]["hash"]["hash"]


def test_scalar_and_container_slots(make_config):
    c = _collector(make_config())
    c.add("input[1]", 7)
    c.add("input[2]", 1e-5)
    c.add("input[3]", "causal")
    c.add("input[4]", None)
    c.add("input[5]", True)
    assert c.slots["input[1]"] == {"kind": "scalar", "value": 7}
    assert c.slots["input[2]"]["value"] == 1e-5
    assert c.slots["input[3]"]["value"] == "'causal'"
    assert c.slots["input[4]"] == {"kind": "scalar", "value": None}
    assert c.slots["input[5]"] == {"kind": "scalar", "value": True}
    assert c.n_tensors == 0


def test_scalars_can_be_disabled(make_config):
    c = _collector(make_config(record_scalars=False))
    c.add("input[1]", 7)
    assert c.slots == {}


def test_nested_traversal_naming(make_config):
    c = _collector(make_config(traverse_depth=3))
    c.add("output", {"loss": torch.zeros(1), "aux": [torch.ones(1), (torch.zeros(1),)]})
    assert set(c.slots) == {"output.loss", "output.aux[0]", "output.aux[1][0]"}


def test_depth_cutoff_collapses_to_signature(make_config):
    c = _collector(make_config(traverse_depth=1))
    c.add("input[0]", [torch.ones(2)])
    assert c.slots["input[0][0]"]["kind"] == "tensor"          # 一层还能展开
    c.add("input[1]", [[torch.ones(2)]])
    assert c.slots["input[1][0]"]["kind"] == "list"            # 超深就只留结构签名
    assert c.slots["input[1][0]"]["sig"] == "list[Tensor]"


def test_max_tensors_per_call_truncates(make_config):
    c = _collector(make_config(max_tensors_per_call=2))
    for i in range(5):
        c.add(f"input[{i}]", torch.ones(2))
    assert len(c.slots) == 2 and c.truncated


def test_unknown_object_is_repr_only(make_config):
    class Weird:
        pass

    c = _collector(make_config())
    c.add("input[3]", Weird())
    assert c.slots["input[3]"]["kind"] == "object"
    assert "Weird" in c.slots["input[3]"]["repr"]


def test_tensor_like_object_is_collected(make_config):
    """FSDP/自定义包装常带 detach+shape，要按张量处理（需求 2 的兼容性）。"""

    class Fake:
        def __init__(self, t):
            self._t = t

        def detach(self):
            return self._t

        @property
        def shape(self):
            return self._t.shape

    c = _collector(make_config())
    c.add("output", Fake(torch.arange(4, dtype=torch.float32)))
    assert c.slots["output"]["kind"] == "tensor"
    assert c.slots["output"]["sample"]["vals"] == [0.0, 1.0, 2.0, 3.0]


def test_full_saver_hooked(make_config):
    seen = []

    def saver(t):
        seen.append(int(t.numel()))
        return {"path": "tensors/x.pt"}

    c = _collector(make_config(), full_saver=saver)
    c.add("output", torch.ones(3))
    assert c.slots["output"]["full_tensor"]["path"] == "tensors/x.pt"
    assert seen == [3]


# ----------------------------------------------------------------------
def test_collect_args_kwargs(make_config):
    c = _collector(make_config())
    x, m = torch.ones(2), torch.zeros(2)
    collect_args((x, 3), {"attention_mask": m, "scale": 0.5}, c)
    assert sorted(c.slots) == ["input[0]", "input[1]", "kwargs.attention_mask", "kwargs.scale"]
    assert c.slots["input[1]"]["value"] == 3
    assert c.slots["kwargs.attention_mask"]["kind"] == "tensor"


def test_collect_params(make_config):
    c = _collector(make_config())
    collect_params(nn.Linear(4, 4), c, limit=8)
    assert sorted(c.slots) == ["param.bias", "param.weight"]
    c2 = _collector(make_config())
    collect_params(nn.Linear(3, 3), c2, limit=1)      # 只取前 limit 个参数
    assert len(c2.slots) == 1


# ----------------------------------------------------------------------
def test_structure_sig():
    assert structure_sig(torch.ones(2)) == "Tensor"
    assert structure_sig({"a": torch.ones(1), "b": [1, 2]}) == "{a:Tensor,b:list[int,int]}"
    assert structure_sig((torch.ones(1), 2)) == "tuple[Tensor,int]"
    assert structure_sig(3.5) == "float"


def test_has_nan(make_config):
    cfg = make_config(sample_mode="off", sample_n=0, traverse_depth=1)
    col = SlotCollector(cfg, TensorSampler("off", 0))
    col.add("output", torch.tensor([float("nan")]))
    assert has_nan(col.slots)
    col2 = SlotCollector(cfg, TensorSampler("off", 0))
    col2.add("output", torch.ones(2))
    assert not has_nan(col2.slots)
    assert not has_nan({"x": {"kind": "scalar", "value": 1}})


# ----------------------------------------------------------------------
def _matcher(pats):
    return RegexMatcher(pats, "t")


def test_apply_include_exclude():
    no = _matcher(None)
    assert apply_include_exclude("a.b", "Linear", no, no, no, no)
    assert not apply_include_exclude("a.b", "Linear", _matcher([r"^c\."]), no, no, no)
    assert apply_include_exclude("c.x", "Linear", _matcher([r"^c\."]), no, no, no)
    assert not apply_include_exclude("a.b", "Linear", no, _matcher([r"\.b$"]), no, no)
    assert not apply_include_exclude("a.b", "Dropout", no, no, no, _matcher(["Dropout"]))
    assert not apply_include_exclude("a.b", "Linear", no, no, _matcher(["Embedding"]), no)
    assert apply_include_exclude("a.b", "Embedding", no, no, _matcher(["Embedding"]), no)
