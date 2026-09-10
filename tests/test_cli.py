"""CLI 入口（需求 6/7 的落地命令）：compare / stack / query / report / env / status 的退出码与输出。"""

from __future__ import annotations

import json
import os
import re

import pytest
import torch

from pprobe.cli import main
from pprobe.config import ENV_DOCS

T0 = torch.arange(12, dtype=torch.float32).reshape(3, 4)


def _frames(*specs):
    return [{"file": f"/app/{f}", "line": ln, "name": nm, "text": f"/app/{f}:{ln} in {nm}"}
            for f, ln, nm in specs]


def _spec(tensors, **kw):
    d = {"tensors": tensors}
    d.update(kw)
    return d


@pytest.fixture
def pair(build_run):
    """一有一无差异的两个结果目录：(A, B)；每次调用用新名字，避开 events.jsonl 追写污染。"""
    fr = _frames(("train.py", 10, "main"), ("train.py", 42, "step"))
    calls = iter(range(100))

    def _make(diff=True):
        k = next(calls)
        a = build_run(f"cli-a{k}", [_spec({"output": T0}, stack_frames=fr),
                                    _spec({"output": T0 + 100.0}, module="net.big", stack_frames=fr)])
        b = build_run(f"cli-b{k}", [_spec({"output": T0 + (1.0 if diff else 0.0)}, stack_frames=fr),
                                    _spec({"output": T0 + 100.0}, module="net.big", stack_frames=fr)])
        return a, b
    return _make


# ----------------------------------------------------------------------
def test_main_no_args_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage: pprobe" in out and "pprobe stack s12 --result" in out


def test_main_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "pprobe" in capsys.readouterr().out


def test_unknown_env_var_is_reported_via_check(capsys, monkeypatch):
    assert main(["env", "--check"]) == 0
    monkeypatch.setenv("PPROBE_SAMPLE_NOPE", "1")
    assert main(["env", "--check"]) == 1
    assert "PPROBE_SAMPLE_NOPE" in capsys.readouterr().out


def test_env_listing_and_markdown(capsys):
    assert main(["env"]) == 0
    out = capsys.readouterr().out
    assert "PPROBE_SAMPLE_N" in out and "PPROBE_FLUSH_INTERVAL" in out
    assert out.count("PPROBE_") >= len(ENV_DOCS)

    assert main(["env", "--markdown"]) == 0
    md = capsys.readouterr().out.splitlines()
    assert md[0] == "| 环境变量 | 默认值 | 类型 | 说明 |"
    assert md[1] == "| --- | --- | --- | --- |"
    assert len(md) == 2 + len(ENV_DOCS)

    assert main(["env", "--filter", "sample", "--markdown"]) == 0
    filtered = capsys.readouterr().out.splitlines()
    rows = [ln for ln in filtered if ln.startswith("| ") and "环境变量" not in ln and "---" not in ln]
    assert rows and len(filtered) < len(md)
    assert all("SAMPLE" in ln.upper() or "SEED" in ln.upper() for ln in rows)


_README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def test_readme_env_table_is_in_sync(capsys):
    """README 的环境变量表是由 `pprobe env --markdown` 生成的，漂了就报错。"""
    if not os.path.isfile(_README):               # 打包安装后可能只带代码不带文档
        pytest.skip("没有 README.md")
    with open(_README, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"<!-- ENV-TABLE-START.*?\n(.*?)\n<!-- ENV-TABLE-END", text, re.S)
    assert m, "README 里缺少 ENV-TABLE 标记块"
    doc_rows = [ln for ln in m.group(1).splitlines() if ln.startswith("|")]
    assert main(["env", "--markdown"]) == 0
    live_rows = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("|")]
    assert doc_rows == live_rows, "README 环境变量表过期，请用 `pprobe env --markdown` 重新生成"


def test_status_reports_interpreter(capsys):
    code = main(["status"])
    assert code in (0, 1)                                  # 取决于是否已 install
    out = capsys.readouterr().out
    assert "解释器" in out and "pth 已安装" in out and "下次启动是否生效" in out


def test_install_dry_run_writes_nothing(capsys):
    assert main(["install", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "将写入 pprobe.pth" in out
    assert '__import__("pprobe.bootstrap"' in out and "PPROBE_ENABLE" in out
    assert "候选目录" in out


# ----------------------------------------------------------------------
def test_compare_writes_report_files(pair, capsys):
    a, b = pair()
    assert main(["compare", a, b]) == 0
    out = capsys.readouterr().out
    assert "最早发散点 rank0" in out and "sample_diff=1" in out
    # 缺省写到 B 目录旁边（与 B 同级），不污染结果目录本身
    stem = os.path.join(os.path.dirname(b), "pprobe_compare_cli-a0_vs_cli-b0")
    md, js = stem + ".md", stem + ".json"
    assert os.path.isfile(md) and os.path.isfile(js)
    assert not os.path.exists(os.path.join(b, "pprobe_compare_cli-a0_vs_cli-b0.md"))
    text = open(md, encoding="utf-8").read()
    assert "# pprobe 精度对比报告" in text and "forward:net.linear#0" in text
    report = json.load(open(js, encoding="utf-8"))
    assert report["totals"]["divergent_events"] == 1
    assert report["divergences"][0]["stack_a"] == "s1"      # 需求 6：差异里带堆栈 id


def test_compare_out_and_json_paths(pair, tmp_path, capsys):
    a, b = pair()
    dest = tmp_path / "sub" / "diff.md"
    assert main(["compare", a, b, "--out", str(dest), "--no-stack"]) == 0
    capsys.readouterr()
    assert dest.is_file()
    assert dest.with_suffix(".json").is_file()
    assert "堆栈查询" not in dest.read_text(encoding="utf-8")

    only_json = tmp_path / "rep.json"
    assert main(["compare", a, b, "--no-write", "--json", str(only_json)]) == 0
    assert only_json.is_file() and json.load(open(only_json, encoding="utf-8"))["tool"] == "pprobe-compare"


def test_compare_fail_if_diff_and_clean_run(pair, tmp_path, capsys):
    a, b = pair()
    assert main(["compare", a, b, "--fail-if-diff", "--out", str(tmp_path / "x.md")]) == 1
    capsys.readouterr()
    a2, b2 = pair(diff=False)
    assert main(["compare", a2, b2, "--fail-if-diff", "--out", str(tmp_path / "y.md")]) == 0


def test_compare_filters_and_stdout_md(pair, tmp_path, capsys):
    a, b = pair()
    assert main(["compare", a, b, "--out", str(tmp_path / "z.md"), "--include", r"net\.big",
                 "--stdout-md"]) == 0
    out = capsys.readouterr().out
    assert "# pprobe 精度对比报告" in out
    assert "net.big" not in out.split("## 差异明细")[-1]     # 被 include 过滤后没有该模块的差异

    assert main(["compare", a, b, "--out", str(tmp_path / "z2.md"), "--exclude", r"net\.big",
                 "--phase", "forward", "--rank", "0:0", "--ignore-slot", "output", "--no-env"]) == 0
    rep = json.load(open(str(tmp_path / "z2.json"), encoding="utf-8"))
    assert rep["totals"]["divergent_events"] == 0 and rep["env_diff"] == {}

    assert main(["compare", a, b, "--out", str(tmp_path / "z3.md"), "--atol", "2", "--rtol", "0",
                 "--top", "1", "--detail", "0", "--sort", "max_abs"]) == 0


def test_compare_missing_dir_exit_2(tmp_path, capsys):
    assert main(["compare", str(tmp_path / "nope"), str(tmp_path / "nope2")]) == 2
    assert "结果目录不存在" in capsys.readouterr().err


# ----------------------------------------------------------------------
def test_stack_command_by_id(pair, capsys):
    a, _b = pair()
    assert main(["stack", "s1", "--result", a, "--events"]) == 0
    out = capsys.readouterr().out
    assert "堆栈 s1" in out and "/app/train.py:42 in step" in out
    assert "引用该堆栈的事件:" in out and "forward" in out
    assert "net.big" in out


def test_stack_command_json_and_directory_token(pair, capsys):
    a, _b = pair()
    assert main(["stack", a, "s1", "--json"]) == 0          # 目录也可以直接当位置参数
    items = json.loads(capsys.readouterr().out)
    assert items[0]["rank"] == 0 and items[0]["id"] == "s1"
    assert items[0]["stack"]["count"] == 2


def test_stack_accepts_comma_separated_ids(build_run, capsys):
    """`s1,s2` 和 `s1 s2` 等价，重复 id 不重复输出（从报告里粘一串 id 很方便）。"""
    a = build_run("cli-comma", [
        _spec({"output": T0}, stack_frames=_frames(("train.py", 1, "one"))),
        _spec({"output": T0}, module="net.big", stack_frames=_frames(("train.py", 2, "two"))),
    ])
    assert main(["stack", "s1,s2", "--result", a]) == 0
    out = capsys.readouterr().out
    assert "堆栈 s1" in out and "堆栈 s2" in out

    assert main(["stack", "s1,s1", "s1", "--result", a, "--json"]) == 0
    items = json.loads(capsys.readouterr().out)
    assert [it["id"] for it in items] == ["s1"]


def test_stack_by_module_reverse_lookup(pair, capsys):
    a, _b = pair()
    assert main(["stack", "--result", a, "--module", r"net\.big"]) == 0
    out = capsys.readouterr().out
    assert "涉及 1 个 (rank/堆栈 id)" in out and "rank0/s1" in out
    assert "自动展开第一个" in out and "堆栈 s1" in out
    assert main(["stack", "--result", a, "--module", r"nothing\.matches", "--phase", "forward"]) == 1
    assert "涉及 0 个" in capsys.readouterr().out


def test_stack_unknown_id_returns_1(pair, capsys):
    a, _b = pair()
    assert main(["stack", "s404", "--result", a]) == 1
    out = capsys.readouterr().out
    assert "没有找到堆栈" in out and "可用 id 示例: s1" in out


def test_stack_without_result_fails(capsys):
    assert main(["stack", "s1"]) == 2                       # FileNotFoundError → 2
    assert "--result" in capsys.readouterr().err


# ----------------------------------------------------------------------
@pytest.fixture
def nan_run(build_run):
    bad = torch.tensor([1.0, float("nan"), float("inf")])
    return build_run("nanrun", [
        _spec({"output": T0}, module="net.a"),
        _spec({"input[0]": bad}, module="net.b", cls="Dropout"),
    ])


def test_query_command(nan_run, capsys):
    assert main(["query", nan_run, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["module"] for r in rows] == ["net.a", "net.b"]
    assert main(["query", nan_run, "--nan", "--slot", r"input"]) == 0
    out = capsys.readouterr().out
    assert "net.b" in out and "input[0](nan=1,inf=1)" in out
    assert main(["query", nan_run, "--module", r"net\.a", "--phase", "backward"]) == 0
    assert "没有匹配的事件" in capsys.readouterr().out
    assert main(["query", nan_run, "--seq-min", "2", "--seq-max", "2", "--limit", "1"]) == 0
    assert "net.b" in capsys.readouterr().out
    assert main(["query", nan_run, "--inf", "--rank", "0"]) == 0
    assert "net.b" in capsys.readouterr().out


def test_report_command(nan_run, capsys):
    assert main(["report", nan_run]) == 0
    out = capsys.readouterr().out
    assert "### rank0" in out and "最早出现 NaN 的位置" in out
    assert "pprobe stack" in out or "未发现 NaN/Inf" in out
    assert main(["report", nan_run, "--markdown", "--top", "1"]) == 0
    md = capsys.readouterr().out
    assert md.strip().startswith("```text") and md.rstrip().endswith("```")
    assert main(["report", nan_run, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["rank"] == 0
    assert main(["report", nan_run, "--rank", "9"]) == 1
    assert "没有可分析的 rank 数据" in capsys.readouterr().out


def test_cli_traceback_env(pair, capsys, monkeypatch):
    """命令行出错时默认只打一行，PPROBE_TRACEBACK=1 才给栈。"""
    monkeypatch.delenv("PPROBE_TRACEBACK", raising=False)
    assert main(["compare", pair()[0], "/nonexistent-dir-xyz"]) == 2
    assert "Traceback" not in capsys.readouterr().err
