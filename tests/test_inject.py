"""解释器注入（需求 1）：pth 内容、安装/卸载，以及子进程里真实生效与否。"""

from __future__ import annotations

import json
import os
import site
import subprocess
import sys
import textwrap

import pytest

from pprobe import inject

REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

_CHILD = textwrap.dedent(
    """
    import json, os, site, sys
    site.addsitedir({site_dir!r})
    state = {{"pprobe_loaded": "pprobe.bootstrap" in sys.modules,
             "torch_loaded": "torch" in sys.modules,
             "watcher": [getattr(f, "name", "") for f in sys.meta_path if getattr(f, "_pprobe_watcher", False)],
             "patched": None, "rank_dir": None, "events": 0}}
    if os.environ.get("PPROBE_ENABLE"):
        import torch
        state["patched"] = "patched_call" in repr(torch.nn.Module.__call__)
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        out = m(torch.randn(2, 4))
        out.sum().backward()
        state["rank_dir"] = __import__("pprobe").result_dir()
    print("PPROBE-STATE " + json.dumps(state))
    """
)


def _events_in(rank_dir):
    path = os.path.join(rank_dir, "events.jsonl")
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return len([ln for ln in f if ln.strip()])


def _spawn(tmp_path, env):
    """用 `-S` 起子进程：跳过环境里已安装的全局 pprobe.pth，只验证本用例写入的那一份。"""
    sitedir = tmp_path / "site"
    sitedir.mkdir(exist_ok=True)
    (sitedir / inject.PTH_NAME).write_text(inject._PTH_CONTENT, encoding="utf-8")
    script = tmp_path / "child.py"
    script.write_text(_CHILD.format(site_dir=str(sitedir)), encoding="utf-8")
    full_env = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO_SRC] + site.getsitepackages()))
    for k in list(full_env):
        if k.startswith("PPROBE_"):
            del full_env[k]
    full_env.update(env)
    proc = subprocess.run([sys.executable, "-S", str(script)], env=full_env, capture_output=True,
                          text=True, timeout=300)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("PPROBE-STATE ")), None)
    assert line, f"子进程没有输出状态: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"
    return json.loads(line[len("PPROBE-STATE "):]), proc


# ----------------------------------------------------------------------
def test_pth_line_shortcircuits_without_env(tmp_path):
    """需求 1 的零开销承诺：没开 PPROBE_ENABLE 时连 pprobe 都不该被 import。"""
    state, proc = _spawn(tmp_path, {})
    assert proc.returncode == 0
    assert state["pprobe_loaded"] is False
    assert state["torch_loaded"] is False
    assert state["watcher"] == []


def test_pth_line_boots_probe_and_hooks_forward_backward(tmp_path):
    """需求 1 + 2：设一个环境变量就自动 hook 到 forward/backward，无需改任何代码。"""
    out = tmp_path / "probe_out"
    state, proc = _spawn(tmp_path, {"PPROBE_ENABLE": "1", "PPROBE_OUT": str(out)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert state["pprobe_loaded"] is True
    assert state["watcher"] == ["torch"]                 # torch 尚未导入 → 挂监听器
    assert state["patched"] is True                      # torch 导入完成后 Module.__call__ 被替换
    assert state["rank_dir"] and state["rank_dir"].endswith("rank0")
    assert _events_in(state["rank_dir"]) >= 3            # Linear/ReLU/Linear 前向都被抓到


def test_probe_honors_env_knobs_in_child(tmp_path):
    """需求 3/5：采样个数与最大事件数在子进程里同样生效。"""
    out = tmp_path / "knob"
    state, _proc = _spawn(tmp_path, {"PPROBE_ENABLE": "1", "PPROBE_OUT": str(out),
                                     "PPROBE_SAMPLE_N": "6", "PPROBE_MAX_EVENTS": "2",
                                     "PPROBE_FLUSH_INTERVAL": "1"})
    n = _events_in(state["rank_dir"])
    assert n == 2
    ev = json.loads(open(os.path.join(state["rank_dir"], "events.jsonl"), encoding="utf-8").readline())
    assert ev["tensors"]["input[0]"]["sample"]["n"] == 6
    assert ev["tensors"]["input[0]"]["stats"]["count"] == 8
    man = json.load(open(os.path.join(state["rank_dir"], "manifest.json"), encoding="utf-8"))
    assert man["state"] == "finalized" and "max_events=2" in man["limit_reason"]


# ----------------------------------------------------------------------
def test_install_and_uninstall_target_dir(tmp_path):
    d = tmp_path / "site-packages"
    res = inject.install(target_dir=str(d))
    assert res["writable"] and not res["already"]
    path = d / inject.PTH_NAME
    assert path.is_file()
    assert inject.pth_source() in path.read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("# pprobe")

    again = inject.install(target_dir=str(d))
    assert again["already"] is True and again["path"] == str(path)

    assert inject.uninstall(target_dir=str(d)) == [str(path)]
    assert not path.exists()
    assert inject.uninstall(target_dir=str(d)) == []


def test_dry_run_writes_nothing(tmp_path):
    d = tmp_path / "nope"
    res = inject.install(target_dir=str(d), dry_run=True)
    assert res["path"] == str(d / inject.PTH_NAME)
    assert not (d / inject.PTH_NAME).exists()


def test_uninstall_refuses_foreign_file(tmp_path):
    d = tmp_path / "site-packages"
    d.mkdir(parents=True)
    p = d / inject.PTH_NAME
    p.write_text("# 别人放的同名文件\nimport os\n", encoding="utf-8")
    assert inject.uninstall(target_dir=str(d)) == []
    assert p.exists()
    assert inject.install(target_dir=str(d))["already"] is False   # 内容不同 → 会覆盖安装


def test_find_existing_and_status():
    dirs = inject.candidate_dirs()
    assert dirs and all(os.path.isabs(d) for d in dirs)
    st = inject.status()
    assert st["interpreter"] == sys.executable
    assert st["candidate_dirs"] == dirs
    assert st["installed"] == bool(st["pth_files"])
    assert st["will_activate"] is False                  # 测试环境里 PPROBE_ENABLE 未设置
    assert st["enable_env"] is None


def test_candidate_dirs_for_other_interpreter(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    os.chmod(fake, 0o755)
    with pytest.raises(RuntimeError):
        inject.candidate_dirs(str(fake))


# ----------------------------------------------------------------------
def test_cli_run_injects_and_collects(tmp_path, capsys):
    """`pprobe run` 不依赖事先 install 也能临时注入（对 CI/一次性排查很关键）。"""
    from pprobe.cli import main
    from pprobe.result import Result

    if not inject.find_existing():
        pytest.skip("当前解释器未安装 pprobe.pth，跳过 `pprobe run` 端到端")
    script = tmp_path / "train.py"
    script.write_text(
        "import torch\n"
        "torch.manual_seed(0)\n"
        "m = torch.nn.Linear(4, 2)\n"
        "opt = torch.optim.SGD(m.parameters(), lr=0.1)\n"
        "opt.zero_grad()\n"
        "loss = m(torch.randn(3, 4)).sum()\n"
        "loss.backward()\n"
        "opt.step()\n"
        "print('train-ok')\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"
    rc = main(["run", "--out", str(out), "--sample-n", "5", "--set", "HOOK=forward,backward,optim", str(script)])
    capsys.readouterr()
    assert rc == 0
    res = Result(str(out))
    assert res.rank_ids() == [0]
    events = list(res.ranks[0].iter_events())
    assert [e["phase"] for e in events] == ["forward", "backward"]
    assert events[0]["tensors"]["input[0]"]["sample"]["n"] == 5
    assert events[0]["step"] == 0 and events[1]["fwd_seq"] == events[0]["seq"]
    man = json.load(open(os.path.join(res.ranks[0].path, "manifest.json"), encoding="utf-8"))
    assert man["config"]["hooks"] == ["forward", "backward", "optim"]
