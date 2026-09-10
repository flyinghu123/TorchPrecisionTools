"""pprobe 命令行入口。

子命令一览：

* ``install`` / ``uninstall`` / ``status``：管理 ``pprobe.pth`` 解释器注入
* ``env``：列出全部 ``PPROBE_*`` 环境变量（``--markdown`` 直接生成文档表格）
* ``run``：临时注入并运行脚本（不需要事先 install）
* ``compare``：**按 rank 对比两次结果目录**，把差异与堆栈 id 写到文件（需求 6）
* ``stack``：**按堆栈 id 反查完整调用栈**（需求 7）
* ``query`` / ``report``：事件过滤查询、单目录速览
* ``selftest``：端到端自检注入 + 采集链路
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

from . import __version__
from .compare import Comparator, Options
from .config import ENV_DOCS, ENV_PREFIX, unknown_env
from .query import (format_event_rows, format_stack, format_survey, resolve_stacks,
                    stack_ids_for_module, survey_rank, query_events)
from .render import default_out_path, render
from .result import Result
from .util import dumps_pretty, log


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pprobe",
        description="PyTorch 训练精度问题定位探针（forward/backward hook + 采样 + summary + 堆栈 id）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              pprobe install                          # 注入当前解释器
              PPROBE_ENABLE=1 python train.py         # megatron/llamafactory/ms-swift 同样适用
              pprobe report ./pprobe_out              # 单目录速览：NaN 热点、最早出错位置
              pprobe compare ./base ./cand --out diff.md   # 按 rank 对比，报告含堆栈 id
              pprobe stack s12 --result ./base        # 用堆栈 id 查完整调用栈
        """),
    )
    p.add_argument("--version", action="version", version=f"pprobe {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")

    # -- install / uninstall / status -----------------------------------
    for name, help_ in (("install", "写入 pprobe.pth 完成解释器注入"),
                        ("uninstall", "删除 pprobe.pth")):
        s = sub.add_parser(name, help=help_, description=help_)
        s.add_argument("--python", help="目标解释器路径（默认当前）")
        s.add_argument("--dir", help="直接指定 site-packages 目录")
        if name == "install":
            s.add_argument("--dry-run", action="store_true", help="只打印将要写入的内容")
        s.set_defaults(func=cmd_inject)

    s = sub.add_parser("status", help="查看注入状态与生效情况")
    s.add_argument("--python", help="目标解释器路径（默认当前）")
    s.set_defaults(func=cmd_status)

    # -- env ------------------------------------------------------------
    s = sub.add_parser("env", help="列出所有 PPROBE_* 环境变量及当前取值")
    s.add_argument("--markdown", action="store_true", help="输出 Markdown 表格（README 用）")
    s.add_argument("--filter", help="只显示名字包含该子串的变量")
    s.add_argument("--check", action="store_true", help="只检查当前环境里有没有拼错的变量")
    s.set_defaults(func=cmd_env)

    # -- run ------------------------------------------------------------
    s = sub.add_parser("run", help="临时注入运行脚本（等价于设好环境变量后 exec python）")
    s.add_argument("--out", help="结果目录（PPROBE_OUT）")
    s.add_argument("--set", dest="kv", action="append", default=[], metavar="NAME=VALUE",
                   help="额外环境变量，名字可省略 PPROBE_ 前缀，可重复")
    s.add_argument("--sample-n", type=int, help="每个 tensor 采样元素个数")
    s.add_argument("--sample-mode", choices=("uniform", "random", "head", "off"))
    s.add_argument("--seed", type=int, help="随机采样种子")
    s.add_argument("--max-events", type=int)
    s.add_argument("--flush-interval", type=int)
    s.add_argument("--include", help="模块路径正则（逗号分隔）")
    s.add_argument("--hook", help="hook 类别，如 forward,backward,func,optim")
    s.add_argument("script", help="要运行的 python 脚本或 -m 模块后的参数")
    s.add_argument("script_args", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_run)

    # -- compare --------------------------------------------------------
    s = sub.add_parser("compare", help="按 rank 对比两次运行的结果目录（需求 6）")
    s.add_argument("a", help="基准结果目录（A）")
    s.add_argument("b", help="候选结果目录（B）")
    s.add_argument("--out", help="报告输出路径（.md/.txt）；缺省写到 B 目录旁边")
    s.add_argument("--json", dest="json_out", help="同时输出机器可读 JSON 报告")
    s.add_argument("--no-write", action="store_true", help="只打印到 stdout，不落文件")
    s.add_argument("--atol", type=float, default=0.0, help="绝对容差，默认 0")
    s.add_argument("--rtol", type=float, default=1e-5, help="相对容差，默认 1e-5")
    s.add_argument("--top", type=int, default=50, help="最多保留多少条差异（0=不限）")
    s.add_argument("--detail", type=int, default=10, help="报告里展开多少条差异的逐元素明细")
    s.add_argument("--rank", dest="rank_specs", action="append", default=[],
                   metavar="A:B", help="指定配对 rank，可重复；如 --rank 0:0 --rank 1:3")
    s.add_argument("--ranks", help="只对比这些 rank（逗号分隔，两边同号）")
    s.add_argument("--phase", choices=("forward", "backward"), help="只对比某一阶段")
    s.add_argument("--sort", choices=("seq", "max_abs", "max_rel"), default="seq",
                   help="差异排序：seq=最早发散（默认，最有用）")
    s.add_argument("--include", help="只对比模块路径匹配该正则的模块")
    s.add_argument("--exclude", help="跳过模块路径匹配该正则的模块")
    s.add_argument("--ignore-slot", dest="ignore_slots", action="append", default=[],
                   help="忽略指定槽位（如 param.name），可重复")
    s.add_argument("--no-env", action="store_true", help="不对比 env.json")
    s.add_argument("--no-stack", action="store_true", help="报告里不展开堆栈查询命令")
    s.add_argument("--stdout-md", action="store_true", help="把完整 Markdown 报告也打到 stdout")
    s.add_argument("--fail-if-diff", action="store_true", help="发现差异时退出码为 1（CI 用）")
    s.set_defaults(func=cmd_compare)

    # -- stack ----------------------------------------------------------
    s = sub.add_parser("stack", help="按堆栈 id 查询完整调用栈（需求 7）")
    s.add_argument("ids", nargs="*", help="堆栈 id（s1 s2 或 s1,s2）；也可以传结果目录")
    s.add_argument("-r", "--result", help="结果目录（含 rankN/stacks.json）")
    s.add_argument("--rank", dest="ranks", help="只看指定 rank（默认全部）")
    s.add_argument("--events", action="store_true", help="同时列出引用该堆栈的事件")
    s.add_argument("--module", help="反查：列出模块路径匹配该正则的所有堆栈 id")
    s.add_argument("--phase", choices=("forward", "backward"))
    s.add_argument("--json", action="store_true", help="以 JSON 输出")
    s.set_defaults(func=cmd_stack)

    # -- query ----------------------------------------------------------
    s = sub.add_parser("query", help="按条件查询事件")
    s.add_argument("result", help="结果目录")
    s.add_argument("--module", help="模块路径正则")
    s.add_argument("--slot", help="槽位名正则（配合 --nan/--inf）")
    s.add_argument("--phase", choices=("forward", "backward"))
    s.add_argument("--rank", dest="ranks", help="只看指定 rank")
    s.add_argument("--seq-min", type=int)
    s.add_argument("--seq-max", type=int)
    s.add_argument("--nan", action="store_true", help="只看含 NaN 的事件")
    s.add_argument("--inf", action="store_true", help="只看含 Inf 的事件")
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_query)

    # -- report ---------------------------------------------------------
    s = sub.add_parser("report", help="单个结果目录速览：NaN 热点、最早异常位置、量级排行")
    s.add_argument("result", help="结果目录")
    s.add_argument("--rank", dest="ranks", help="只看指定 rank")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--json", action="store_true")
    s.add_argument("--markdown", action="store_true")
    s.set_defaults(func=cmd_report)

    # -- selftest -------------------------------------------------------
    s = sub.add_parser("selftest", help="端到端自检：注入 → 采集 → 查询 → 对比")
    s.add_argument("--keep", action="store_true", help="保留临时结果目录")
    s.add_argument("--python", help="用哪个解释器跑（默认当前）")
    s.set_defaults(func=cmd_selftest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as e:
        log(f"错误: {e}")
        return 2
    except KeyboardInterrupt:
        log("已中断")
        return 130
    except Exception as e:  # noqa: BLE001
        log(f"命令执行失败: {type(e).__name__}: {e}")
        if os.environ.get("PPROBE_TRACEBACK"):
            import traceback

            traceback.print_exc()
        return 1


# ----------------------------------------------------------------------
# 注入相关
# ----------------------------------------------------------------------
def cmd_inject(args) -> int:
    from . import inject

    if args.cmd == "install":
        if args.dry_run:
            print(f"# 将写入 {inject.PTH_NAME}：\n{inject.pth_source()}")
            for d in inject.candidate_dirs(args.python):
                print(f"#   候选目录: {d}")
            return 0
        res = inject.install(python=args.python, target_dir=getattr(args, "dir", None))
        if not res.get("writable"):
            log("所有 site-packages 都不可写；请用 --dir 指定可写目录，或用 pprobe run 临时注入")
            return 1
        if res.get("already"):
            print(f"已注入: {res['path']}")
        else:
            print(f"写入成功: {res['path']}")
        print("下一步：PPROBE_ENABLE=1 PPROBE_OUT=./pprobe_out python your_train.py")
        return 0

    removed = inject.uninstall(python=args.python, target_dir=getattr(args, "dir", None))
    print(f"已删除: {', '.join(removed) if removed else '（没有找到 pprobe.pth）'}")
    return 0


def cmd_status(args) -> int:
    from . import inject

    st = inject.status(python=args.python)
    print(f"解释器         : {st.get('interpreter')}")
    print(f"sys.prefix     : {st.get('prefix')}")
    print(f"pth 已安装     : {'是' if st.get('installed') else '否'}  {st.get('pth_files') or ''}")
    print(f"候选目录       : {', '.join(st.get('candidate_dirs') or [])}")
    print(f"用户 site 可用 : {st.get('user_site_enabled')}"
          + ("（PYTHONNOUSERSITE 已设置，用户目录会被忽略）" if st.get("pythonnousersite") else ""))
    print(f"PPROBE_ENABLE  : {st.get('enable_env')!r}")
    print(f"下次启动是否生效: {'✅ 会' if st.get('will_activate') else '❌ 不会'}")
    print(f"探针 meta_path 已挂载: {st.get('self_pth_imported')}")
    bad = unknown_env()
    if bad:
        print(f"⚠️ 无法识别的环境变量: {', '.join(bad)}")
    return 0 if st.get("installed") else 1


def cmd_env(args) -> int:
    if args.check:
        bad = unknown_env()
        if bad:
            print("发现无法识别的变量: " + ", ".join(bad))
            return 1
        print("当前 PPROBE_* 变量全部合法")
        return 0
    cur = {k[len(ENV_PREFIX):]: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX)}
    if args.markdown:
        print("| 环境变量 | 默认值 | 类型 | 说明 |")
        print("| --- | --- | --- | --- |")
        # 类型里带 ``a|b`` 会把 markdown 表格列打散，必须转义
        esc = "\\|"
        for name, default, typ, desc in ENV_DOCS:
            if args.filter and args.filter.lower() not in name.lower():
                continue
            print(f"| ``{ENV_PREFIX}{name}`` | {default} | {typ.replace('|', esc)} | {desc} |")
        return 0
    width = max(len(n) for n, _, _, _ in ENV_DOCS) + len(ENV_PREFIX) + 2
    for name, default, _typ, desc in ENV_DOCS:
        if args.filter and args.filter.lower() not in name.lower():
            continue
        full = ENV_PREFIX + name
        live = cur.get(name)
        mark = f"当前={live}" if live is not None else ""
        print(f"{full:<{width}} {default:<18} {desc}" + (f"   [{mark}]" if mark else ""))
    return 0


def cmd_run(args) -> int:
    env = dict(os.environ)
    env["PPROBE_ENABLE"] = "1"
    for kv in args.kv:
        k, _, v = kv.partition("=")
        k = k.strip()
        if not k.startswith(ENV_PREFIX):
            k = ENV_PREFIX + k
        env[k] = v
    for flag, name in (("out", "OUT"), ("sample_n", "SAMPLE_N"), ("sample_mode", "SAMPLE_MODE"),
                       ("seed", "SEED"), ("max_events", "MAX_EVENTS"), ("flush_interval", "FLUSH_INTERVAL"),
                       ("include", "INCLUDE"), ("hook", "HOOK")):
        v = getattr(args, flag, None)
        if v is not None:
            env[f"{ENV_PREFIX}{name}"] = str(v)
    out = env.get(f"{ENV_PREFIX}OUT", "./pprobe_out")
    print(f"[pprobe] run: PPROBE_OUT={os.path.abspath(out)}")
    target = [args.script, *args.script_args]
    # 允许 `pprobe run -- python -m foo` 里省略解释器
    if target and target[0] in ("python", "python3"):
        target = target[1:]
    proc = subprocess.run([sys.executable, *target], env=env)
    print(f"[pprobe] 退出码 {proc.returncode}，结果目录 {os.path.abspath(out)}")
    print(f"[pprobe] 速览: pprobe report {os.path.abspath(out)}")
    return proc.returncode


# ----------------------------------------------------------------------
# 对比
# ----------------------------------------------------------------------
def _parse_rank_specs(specs: list[str], ranks_arg: str | None) -> list[tuple[int, int]] | None:
    pairs: list[tuple[int, int]] = []
    for s in specs or []:
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                a, b = part.split(":", 1)
                pairs.append((int(a), int(b)))
            else:
                pairs.append((int(part), int(part)))
    if ranks_arg:
        for part in ranks_arg.split(","):
            part = part.strip()
            if part:
                pairs.append((int(part), int(part)))
    return pairs or None


def cmd_compare(args) -> int:
    res_a = Result.load(args.a)
    res_b = Result.load(args.b)
    opts = Options(
        atol=args.atol, rtol=args.rtol, top=args.top, ranks=_parse_rank_specs(args.rank_specs, args.ranks),
        phase=args.phase, include=args.include, exclude=args.exclude, sort=args.sort,
        ignore_slots=tuple(args.ignore_slots), compare_env=not args.no_env,
    )
    print(f"[pprobe] 对比 {res_a.path} (A) ↔ {res_b.path} (B) "
          f"rank: {res_a.rank_ids()} ↔ {res_b.rank_ids()} atol={args.atol} rtol={args.rtol}")
    cmp = Comparator(res_a, res_b, opts)
    report = cmp.run()
    totals = report.get("totals") or {}
    text = render(report, res_a, res_b, stacks=not args.no_stack, detail=args.detail)

    fd = totals.get("first_divergence")
    if fd:
        print(f"[pprobe] ⚠️ 最早发散点 rank{fd.get('rank')}: {fd.get('key')}  "
              f"(kinds={','.join(fd.get('kinds') or [])}, stack={fd.get('stack_a')})")
    print("[pprobe] " + " | ".join(
        f"{k}={totals.get(k, 0)}" for k in
        ("events_compared", "divergent_events", "info_diff", "special", "stats_diff", "sample_diff",
         "stack_diff", "events_only_a", "events_only_b") if k in totals))
    if args.stdout_md:
        print()
        print(text)

    written: list[str] = []
    if not args.no_write:
        base = args.out or default_out_path(res_a.path, res_b.path)
        md_path = base if base.endswith((".md", ".txt")) else base + ".md"
        _write(md_path, text + "\n")
        written.append(md_path)
        json_path = args.json_out or (os.path.splitext(md_path)[0] + ".json")
        _write(json_path, dumps_pretty(report) + "\n")
        written.append(json_path)
    elif args.json_out:
        _write(args.json_out, dumps_pretty(report) + "\n")
        written.append(args.json_out)
    if written:
        print("[pprobe] 报告已写出: " + ", ".join(written))
    return (1 if args.fail_if_diff and totals.get("divergent_events") else 0)


def _write(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------------------------------------------------
# 堆栈查询
# ----------------------------------------------------------------------
def cmd_stack(args) -> int:
    tokens = list(args.ids or [])
    result_path = args.result
    ids: list[str] = []
    for t in tokens:
        if os.path.isdir(os.path.expanduser(t)):
            result_path = result_path or t
        else:
            # 报告里常一串 id 连着粘，`s1,s2` 和 `s1 s2` 都收
            ids.extend(p for p in t.replace(" ", ",").split(",") if p)
    ids = list(dict.fromkeys(ids))
    if not result_path:
        raise FileNotFoundError("请用 --result 指定结果目录（或在参数里给出目录路径）")
    res = Result.load(result_path)
    ranks = _int_list(args.ranks)

    if args.module:
        hit: list[str] = []
        for r in (ranks if ranks is not None else res.rank_ids()):
            rank = res.ranks.get(r)
            if rank is None:
                continue
            for sid in stack_ids_for_module(rank, args.module, args.phase):
                hit.append(f"rank{r}/{sid}")
        print(f"模块 {args.module!r} 涉及 {len(hit)} 个 (rank/堆栈 id): {', '.join(hit) or '无'}")
        if hit and not ids:
            # 没给 id 时默认展开第一个命中的堆栈，省去二次敲命令
            first, sid = hit[0].split("/")
            ids = [sid]
            ranks = [int(first[4:])]
            print(f"[pprobe] 自动展开第一个: rank{ranks[0]} {sid}")

    items = resolve_stacks(res, ids, ranks=ranks, with_events=args.events)
    if not items:
        avail = ", ".join(sorted({s for r in res.ranks.values() for s in list(r.stacks)[:20]}))
        print(f"没有找到堆栈 {ids or '(无)'}；可用 id 示例: {avail}")
        return 1
    if args.json:
        print(dumps_pretty(items))
    else:
        for it in items:
            print(format_stack(it))
            print()
    return 0


# ----------------------------------------------------------------------
# 查询 / 速览
# ----------------------------------------------------------------------
def cmd_query(args) -> int:
    res = Result.load(args.result)
    rows = query_events(res, module_re=args.module, phase=args.phase, ranks=_int_list(args.ranks),
                        seq_min=args.seq_min, seq_max=args.seq_max, only_nan=args.nan,
                        only_inf=args.inf, slot_re=args.slot, limit=args.limit)
    print(dumps_pretty(rows) if args.json else format_event_rows(rows))
    return 0


def cmd_report(args) -> int:
    res = Result.load(args.result)
    ranks = _int_list(args.ranks)
    rows = []
    for r in (ranks if ranks is not None else res.rank_ids()):
        rank = res.ranks.get(r)
        if rank is None:
            continue
        rows.append(survey_rank(rank, top=args.top))
    if not rows:
        print("目录里没有可分析的 rank 数据")
        return 1
    if args.json:
        print(dumps_pretty(rows))
        return 0
    if args.markdown:
        print("```text")
    print(res.describe())
    print()
    print(format_survey(rows))
    if args.markdown:
        print("```")
    return 0


def _int_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x) for x in text.replace(":", ",").split(",") if x.strip() != ""]


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
_SELFTEST_SCRIPT = '''
import torch, torch.nn as nn, torch.nn.functional as F

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
    def forward(self, x):
        return self.b(F.gelu(self.a(x)))

torch.manual_seed(0)
m = M()
opt = torch.optim.SGD(m.parameters(), lr=0.05)
x = torch.randn(4, 8)
for step in range(2):
    opt.zero_grad()
    loss = F.mse_loss(m(x), torch.ones(4, 8))
    loss.backward()
    opt.step()
print("selftest-train-ok", float(loss.detach()))
'''


def cmd_selftest(args) -> int:
    from . import inject

    py = args.python or sys.executable
    keep = args.keep
    # 不用系统 /tmp：沙箱/容器里常常是只读的，写到当前工作目录最稳
    tmp = os.path.abspath("pprobe_selftest" if keep else ".pprobe_selftest")
    os.makedirs(tmp, exist_ok=True)
    ok = True
    steps: list[tuple[str, bool, str]] = []

    st = inject.status(python=py)
    steps.append(("pth 已安装", bool(st.get("installed")), str(st.get("pth_files"))))

    env_a = dict(os.environ, PPROBE_ENABLE="1", PPROBE_OUT=os.path.join(tmp, "A"),
                 PPROBE_SEED="1234", PPROBE_SAMPLE_N="20", PPROBE_VERBOSE="0")
    env_b = dict(env_a, PPROBE_OUT=os.path.join(tmp, "B"), PPROBE_VERBOSE="0")

    def run(env: dict[str, str], script: str) -> tuple[int, str]:
        p = subprocess.run([py, script], env=env, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr)[-1500:]

    # B 换一个初始化种子，制造真实的精度差异
    script_a = os.path.join(tmp, "_st_a.py")
    script_b = os.path.join(tmp, "_st_b.py")
    with open(script_a, "w", encoding="utf-8") as f:
        f.write(_SELFTEST_SCRIPT)
    with open(script_b, "w", encoding="utf-8") as f:
        f.write(_SELFTEST_SCRIPT.replace("torch.manual_seed(0)", "torch.manual_seed(1)"))

    rc, out = run(env_a, script_a)
    steps.append(("A 正常运行", rc == 0, _pick(out, "selftest-train-ok")))
    p_rc, p_out = run(env_b, script_b)
    steps.append(("B 正常运行", p_rc == 0, _pick(p_out, "selftest-train-ok")))

    try:
        ra, rb = Result.load(os.path.join(tmp, "A")), Result.load(os.path.join(tmp, "B"))
        n_ev = sum(1 for ev in ra.ranks[0].iter_events())
        steps.append(("A 采集到事件", n_ev > 0, f"events={n_ev}"))
        steps.append(("A 有 forward 与 backward",
                      any(ev.get("phase") == "forward" for ev in ra.ranks[0].iter_events())
                      and any(ev.get("phase") == "backward" for ev in ra.ranks[0].iter_events()), ""))
        st_a = ra.ranks[0].stacks
        steps.append(("堆栈 id 可反查", bool(st_a) and "frames" in next(iter(st_a.values()), {}),
                      f"unique={len(st_a)}"))
        sid = next(iter(ra.ranks[0].stacks), None) if ra.ranks[0].stacks else None
        steps.append(("stack 查询有帧", bool(resolve_stacks(ra, [sid] if sid else [])), f"stack={sid}"))
        rep = Comparator(ra, rb, Options(atol=0.0, rtol=1e-6, top=20)).run()
        steps.append(("compare 能出报告", bool(rep.get("totals") is not None),
                      f"divergent={rep['totals'].get('divergent_events')}"))
        txt = render(rep, stacks=True, detail=1)
        steps.append(("报告渲染非空", len(txt) > 200, f"{len(txt)} chars"))
    except Exception as e:  # noqa: BLE001
        steps.append(("结果读取", False, f"{type(e).__name__}: {e}"))
        ok = False

    print(f"pprobe selftest  (临时目录 {tmp})")
    print("-" * 72)
    for name, passed, info in steps:
        ok = ok and passed
        print(f"  {'✅' if passed else '❌'} {name:<24} {info[:200]}")
    print("-" * 72)
    print("结论:", "全部通过" if ok else "存在失败项")
    if not keep:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"结果保留在 {tmp}，可执行: pprobe report {os.path.join(tmp, 'A')}")
    return 0 if ok else 1


def _pick(text: str, marker: str) -> str:
    """从子进程输出里挑出含关键信息的那一行（而不是末尾的警告回显）。"""
    for line in reversed(text.splitlines()):
        if marker in line:
            return line.strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

