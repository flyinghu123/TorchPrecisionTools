#!/usr/bin/env bash
# capture_compare.sh —— 用 pprobe 采两份数据并按 rank 对比，一次出精度差异报告。
#
# 1) A/B 两次运行各采一份数据（沿用当前 shell 里的 PPROBE_* 配置）
# 2) 校验两边的采样/统计配置一致，否则判为伪差异直接中止
# 3) 调 pprobe compare 输出 diff.md + diff.json
#
# 用法:
#   capture_compare.sh [选项] "<A 命令>" "<B 命令>" [输出前缀=./pprobe_cmp] [透传给 compare 的参数...]
#     ./capture_compare.sh "python train.py" "python train.py --tf32" ./cmp --rtol 1e-3 --detail 20
#     PPROBE_SAMPLE_N=200 PPROBE_SEED=1 ./capture_compare.sh "python run.py --device cpu" \
#                                                                  "python run.py --device cuda"
# 选项:
#   --force   两边探针配置不一致时也继续对比
#   -h|help   显示本用法
# 环境变量:
#   PPROBE_BIN  pprobe 可执行文件（默认找 PATH，再退到 "$PYTHON -m pprobe.cli"）
#   PYTHON      目标解释器（默认 python3）

set -euo pipefail

usage() { sed -n '3,18p' "$0" | cut -c3-; }

FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help|help) usage; exit 0 ;;
        --force) FORCE=1; shift ;;
        --) shift; break ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    usage >&2
    echo "错误: 需要两条命令（基准 A 与候选 B）" >&2
    exit 2
fi

CMD_A="$1"
CMD_B="$2"
PREFIX="${3:-./pprobe_cmp}"
shift $(( $# >= 3 ? 3 : $# ))
COMPARE_ARGS=("$@")

PY="${PYTHON:-python3}"
if [ -n "${PPROBE_BIN:-}" ]; then
    read -r -a PPROBE <<<"$PPROBE_BIN"
elif command -v pprobe >/dev/null 2>&1; then
    PPROBE=(pprobe)
elif "$PY" -c "import pprobe.cli" 2>/dev/null; then
    PPROBE=("$PY" -m pprobe.cli)
else
    echo "错误: 找不到 pprobe。先执行: pip install -e . && pprobe install" >&2
    exit 3
fi

# 探针靠 site-packages/pprobe.pth 注入，光设 PPROBE_ENABLE 不会生效；status 未安装时退出码非 0
if ! "${PPROBE[@]}" status >/dev/null 2>&1; then
    echo "⚠️ 探针没注入（pprobe status 失败），训练进程不会记录任何东西。" >&2
    echo "   修复: ${PPROBE[*]} install   然后 ${PPROBE[*]} status 确认「下次启动是否生效」" >&2
    exit 3
fi

mkdir -p "$PREFIX"
ROOT="$(cd "$PREFIX" && pwd)"
RUN_A="$ROOT/run_a"
RUN_B="$ROOT/run_b"

echo "[capture] A: $CMD_A  →  $RUN_A"
PPROBE_ENABLE=1 PPROBE_OUT="$RUN_A" bash -c "$CMD_A"
echo "[capture] B: $CMD_B  →  $RUN_B"
PPROBE_ENABLE=1 PPROBE_OUT="$RUN_B" bash -c "$CMD_B"

# 只有真记录到事件才会建 rankN/ 目录
if ! ls -d "$RUN_A"/rank* >/dev/null 2>&1; then
    echo "错误: A 里没有任何 rankN 目录 —— 探针未生效或训练没走过 nn.Module" >&2
    exit 4
fi

# 采样/统计相关配置必须两边一致，否则 sample 与 stats 的差异没有意义
if ! CFG_DIFF="$("$PY" - "$RUN_A" "$RUN_B" <<'PY'
import json, os, sys

CRITICAL = ("sample_mode", "sample_n", "sample_layout", "sample_seed", "sample_extra",
            "stats", "stats_dtype", "checksum", "full_hash", "hooks", "bwd_mode",
            "record_scalars", "record_params", "traverse_depth", "max_tensors_per_call")


def cfg(root):
    """取第一个有 manifest 的 rank 的配置快照（各 rank 配置必然相同）。"""
    if os.path.isdir(root):
        for r in sorted(os.listdir(root)):
            p = os.path.join(root, r, "manifest.json")
            if r.startswith("rank") and os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    return json.load(f).get("config") or {}
    return {}


ca, cb = cfg(sys.argv[1]), cfg(sys.argv[2])
bad = [(k, ca.get(k), cb.get(k)) for k in CRITICAL if ca.get(k) != cb.get(k)]
for k, a, b in bad:
    print(f"    {k}: A={a!r}  B={b!r}")
sys.exit(1 if bad else 0)
PY
)"; then
    echo "⚠️ 两次运行的探针配置不一致（采样值因此失去可比性）:" >&2
    echo "$CFG_DIFF" >&2
    echo "   常见原因：某一侧漏了 PPROBE_SEED / PPROBE_SAMPLE_N，或一侧改了 PPROBE_HOOK。" >&2
    if [ "$FORCE" != 1 ]; then
        echo "   已中止；确认无妨请加 --force。" >&2
        exit 5
    fi
    echo "   已给 --force，继续对比。" >&2
fi

echo "[capture] 对比: ${PPROBE[*]} compare run_a run_b --out $ROOT/diff.md ${COMPARE_ARGS[*]:-}"
"${PPROBE[@]}" compare "$RUN_A" "$RUN_B" --out "$ROOT/diff.md" ${COMPARE_ARGS[@]+"${COMPARE_ARGS[@]}"}

echo
echo "[capture] 报告      : $ROOT/diff.md（同名 .json 是机器可读版）"
echo "[capture] 单目录速览: ${PPROBE[*]} report $RUN_B"
echo "[capture] 展开堆栈  : ${PPROBE[*]} stack <报告里的 id> --result $RUN_A --events"
