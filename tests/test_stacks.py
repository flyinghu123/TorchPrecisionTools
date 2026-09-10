"""堆栈层（需求 4）：id 表示 + 集中存储去重 + 命中计数一致。"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from pprobe.stacks import StackStore


def _frames(line=10, name="train", file="/proj/train.py"):
    return [{"file": file, "line": line, "name": name, "text": f"{file}:{line} in {name}"}]


# ----------------------------------------------------------------------
def test_intern_dedups_and_counts():
    st = StackStore()
    a = st.intern(_frames())
    b = st.intern(_frames())
    c = st.intern(_frames(line=99))
    assert a == b == "s1" and c == "s2"
    assert st.unique == 2
    assert st.get(a)["count"] == 2 and st.get(c)["count"] == 1
    assert st.to_dict()["total_hits"] == 3
    assert st.to_dict()["count"] == 2


def test_entry_fields():
    st = StackStore()
    sid = st.intern(_frames())
    entry = st.get(sid)
    assert entry["id"] == sid
    assert entry["signature"] == "/proj/train.py:10:train"
    assert entry["top"] == "/proj/train.py:10 in train"
    assert entry["caller"] == entry["top"]
    assert entry["frames"][0]["source"] == ""       # 文件不存在也不能崩


def test_save_throttles_counts_but_writes_new_stacks(tmp_path):
    """新堆栈必须立刻落盘（否则事件里的 id 找不到），仅计数变化可以节流。"""
    path = str(tmp_path / "stacks.json")
    st = StackStore(path=path)
    st.intern(_frames())
    assert st.save(path) is True
    results = []
    for _ in range(9):
        st.intern(_frames())
        results.append(st.save(path))
    assert results.count(True) == 1 and results[-1] is True
    assert st.save(path, force=True) is True
    assert st.to_dict()["stacks"]["s1"]["count"] == 10


def test_save_without_path_is_noop(tmp_path):
    st = StackStore()
    assert st.save() is False                     # 没有路径直接放弃
    st.intern(_frames())
    path = str(tmp_path / "s.json")
    assert st.save(path) is True
    assert st.save(path) is False                 # 既无新堆栈也未到节流周期


def test_restart_recovers_ids_and_counts(tmp_path):
    path = str(tmp_path / "stacks.json")
    st = StackStore(path=path)
    st.intern(_frames())
    st.intern(_frames(line=20, name="eval"))
    st.save(path)

    again = StackStore(path=path)
    assert again.unique == 2
    assert again.get("s1")["count"] == 1
    # 恢复后相同内容必须复用旧 id，而不是新开 s3
    assert again.intern(_frames()) == "s1"
    assert again.unique == 2


def test_capture_trims_framework_frames():
    st = StackStore(trim=True)
    holder: dict = {}

    class Net(nn.Module):
        def forward(self, x):
            holder["sid"] = st.capture()
            return x * 2

    Net()(torch.ones(2))            # 中间经过 nn.Module.__call__ 的内部帧
    entry = st.get(holder["sid"])
    files = [f["file"] for f in entry["frames"]]
    assert not any(os.path.join("torch", "nn", "modules", "module.py") in f for f in files)
    assert not any("pprobe" in f for f in files)
    assert any("test_stacks.py" in f for f in files)


def test_capture_without_trim_keeps_framework_frames():
    st = StackStore(trim=False)
    holder: dict = {}

    class Net(nn.Module):
        def forward(self, x):
            holder["sid"] = st.capture()
            return x * 2

    Net()(torch.ones(2))
    files = [f["file"] for f in st.get(holder["sid"])["frames"]]
    assert any(os.path.join("torch", "nn", "modules", "module.py") in f for f in files)


def test_capture_returns_none_when_no_frames():
    st = StackStore(trim=True, limit=48)
    assert st.capture(skip=99) is None            # 跳出栈顶，采不到任何帧


def test_limit_truncates_deep_stacks():
    st = StackStore(trim=True, limit=2)
    holder: dict = {}

    class Net(nn.Module):
        def forward(self, x):
            def inner3():
                return st.capture()

            def inner2():
                return inner3()

            holder["sid"] = inner2()
            return x * 2

    Net()(torch.ones(2))
    frames = st.get(holder["sid"])["frames"]
    assert len(frames) == 2
