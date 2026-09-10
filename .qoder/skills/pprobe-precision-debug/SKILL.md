---
name: pprobe-precision-debug
description: 用 pprobe 定位 PyTorch 训练的数值精度问题：通过 .pth 把探针注入解释器，hook 每个模块 forward/backward 的输入输出并按采样+summary+堆栈 id 落盘，再按 rank 对比两次运行或两个平台的差异，找出最早发散的模块。当用户提到 loss 对不上、精度漂移、数值不一致、CPU/GPU 结果不同、TF32、bf16/fp16 误差、NaN/Inf、第一个 NaN、某张卡数值不对、megatron/LLaMA-Factory/ms-swift/transformers/deepspeed 训练数值问题，或提到 pprobe、dump 张量、对比两次训练时使用。
---

# pprobe：PyTorch 训练精度问题定位

工具仓库根目录：`/home/flyinghu/project/i-t`（本机开发/验证环境是 conda `tmp`，
解释器 `/home/flyinghu/miniconda3/envs/tmp/bin/python`，命令 `/home/flyinghu/miniconda3/envs/tmp/bin/pprobe`）。

## 0. 先确认探针可用

```bash
pprobe status        # 看「下次启动是否生效」与 pth 路径
pprobe selftest      # 注入 → 采集 → 查询 → 对比，9 项端到端自检
```

`status` 显示未安装时：`pip install -e . && pprobe install`。
**不想污染环境**就用临时注入：`pprobe run --out ./dbg --sample-n 200 -- python train.py`
（等价于设好 `PPROBE_*` 再 exec，可加 `--seed/--hook/--max-events/--set K=V`）。

只有 `PPROBE_ENABLE=1` 时探针才存在；不设时连 `pprobe` 模块都不会被 import。

## 1. 按现象选流程

| 现象 | 走哪个流程 |
| --- | --- |
| 换 torch 版本 / 换硬件 / 开 TF32 / 改参数后 loss 对不上 | 流程 A |
| 多卡里某一张卡数值不对 | 流程 B |
| 出 NaN / Inf，要找第一个 | 流程 C |
| 怀疑权重加载、格式转换（fp32→bf16、LoRA 合并）出错 | 流程 D |

### 流程 A：两次运行 / 两个平台对比

```bash
PPROBE_ENABLE=1 PPROBE_SEED=1 PPROBE_OUT=./runs/a <基准训练命令>
PPROBE_ENABLE=1 PPROBE_SEED=1 PPROBE_OUT=./runs/b <候选训练命令>
pprobe compare ./runs/a ./runs/b --out diff.md --detail 20
```

一行式版本（采两份 + 校配置 + 出报告）：

```bash
bash .qoder/skills/pprobe-precision-debug/scripts/capture_compare.sh \
    "python examples/mlp_simple.py --device cpu" \
    "python examples/mlp_simple.py --device cpu --eps 1e-8" ./cmp
```

产出 `<前缀>/run_a`、`run_b`、`diff.md`、`diff.json`；两边采样/统计配置不一致时会中止
（退出码 2=参数不足、3=探针未注入、4=没采到数据、5=配置不一致，`--force` 可跳过）。
可用 `PPROBE_BIN` / `PYTHON` 指定解释器与命令。

读报告顺序：**① 环境差异 / 探针配置差异**（先排除“采样个数或 seed 不同”这种伪差异）
→ **② 最早发散点**（stdout 里 `⚠️ 最早发散点 rankN: ...` 与报告的同名小节）
→ **③ 逐元素明细**。最早发散点就是根因候选，它下游的差异通常是传播结果。

### 流程 B：分 rank 定位坏卡

```bash
PPROBE_ENABLE=1 PPROBE_OUT=./runs/base torchrun --nproc_per_node=8 train.py
PPROBE_ENABLE=1 PPROBE_OUT=./runs/bad  torchrun --nproc_per_node=8 train.py
pprobe compare ./runs/base ./runs/bad --out ddp.md
```

报告里**每个 rank 各有一个最早发散点**。判读：坏 rank 在 `#0`（第一次前向）就不对 → 本地
数据/权重/内核差异；好 rank 要到 `#1` 之后才飘 → 被 all-reduce 带偏，真凶是那个 seq 更小的 rank。
8 卡太占盘时只采关键 rank：`PPROBE_ONLY_RANKS=0,1`。两边 rank 编号不同（如 TP/PP 混排）用
`--rank 0:8 --rank 1:9` 手工配对（`--ranks 0,1` 是“只筛这些 rank”，两者不同）。

### 流程 C：NaN / Inf 溯源

```bash
pprobe report ./runs/b                 # 哪个模块哪个槽位有 NaN、第一次出现的 seq
pprobe query ./runs/b --nan --json     # 逐条列出含 NaN 的事件
pprobe stack <stack_id> --result ./runs/b --events   # 展开成调用链
```

要直接在**第一个** NaN 处断下来（配合 pdb 看 locals）：

```bash
PPROBE_STOP_ON_NAN=1 PPROBE_ENABLE=1 python -m pdb train.py
```

### 流程 D：权重加载 / 转换

```bash
PPROBE_ENABLE=1 PPROBE_RECORD_PARAMS=1 PPROBE_INCLUDE='.*embed.*|.*qkv.*' python train.py
pprobe query ./pprobe_out --module qkv --json    # 看 param.weight 的 stats/checksum
```

然后照流程 A 与“正确版本”的目录对比，`info`/`stats` 差异会直接指出哪个权重第一次进入时的数值就不对。

## 2. 必须知道的约束（违反会得出错误结论）

* **采样可比性**：两边 `PPROBE_SAMPLE_N` / `SAMPLE_MODE` / `SAMPLE_LAYOUT` / `SEED` 必须完全一致，
  否则 `sample` 差异没有意义。`uniform` + 同 shape 时下标天然一致；`random` 必须固定 `PPROBE_SEED`。
* 想判“是否逐 bit 全等”：`PPROBE_FULL_HASH=1`（整个 tensor 取 blake2b，慢）；跨平台逐 bit 对拍用
  `PPROBE_SAMPLE_EXTRA=bits`（采样值带原始位模式）。
* **容差要按 dtype 给**：`compare` 默认 `--atol 0 --rtol 1e-5`，对 bf16/fp16 太严，会给 `--rtol 1e-2` 级别；
  反过来怀疑严格不一致时把 `--rtol 0` 收紧。
* **控制体量**：`PPROBE_MAX_EVENTS` / `PPROBE_MAX_STEPS` / `PPROBE_MAX_CALLS_PER_MODULE=2` /
  `PPROBE_INCLUDE=<可疑模块>` / `PPROBE_STATS_DTYPE=float32` / `PPROBE_SAMPLE_N=16`。
  几百层 × 每 step × 每 rank 很容易写出几 GB。
* **防丢数据**：最多丢 `PPROBE_FLUSH_INTERVAL`（默认 50）条事件；长跑建议 `PPROBE_FLUSH_INTERVAL=10`
  加 `PPROBE_FLUSH_SECS=5`。`kill -9` 之后只能保住已 fsync 的部分。
* 反向缺席时先确认 `PPROBE_HOOK` 含 `backward`；模块边界梯度拿不到时改 `PPROBE_BWD_MODE=tensor`
  （或 `both`），槽位会变成 `grad_of:<前向槽位>`。
* 没有对应 `nn.Module` 的算子要额外开 `PPROBE_HOOK=forward,backward,optim,func`，
  事件记成 `module="functional.gelu"` 并用 `in_module` 标出宿主模块。
* 探针自身异常一律被吞掉并计入 `manifest.counters.errors`，不会搞崩训练；调试探针本身用
  `PPROBE_TRACEBACK=1`。
* CI 里想让“有差异即失败”：`pprobe compare ... --fail-if-diff`（退出码 1）。

## 3. 读结果目录 / 堆栈

结果是 `<OUT>/rankN/`：`events.jsonl`（一行一事件）、`stacks.json`（堆栈 id → 完整调用链，去重）、
`manifest.json`（配置快照、采样种子、计数器、结束原因）、`env.json`（torch/cuda/后端开关/命令行）、
`exit.json`（退出原因）。目录旁有 `HINTS.txt`，里面的命令可直接执行。

```bash
pprobe stack s4,s7 --result ./runs/b            # 逗号或空格分隔均可
pprobe stack --result ./runs/b --module '.*norm.*'   # 反查某模块涉及哪些堆栈 id
```

差异类型（`kinds`）判读：`info` 基本信息不同（dtype/shape/device/类名，多为配置或加载问题）>
`special` NaN/Inf 个数不同 > `stats` 统计量不同 > `sample` 逐元素不同 > `scalar` 标量参数不同
（eps、reduction 这类）> `stack_diff` 调用链不同；`presence` 表示只在一边存在（步数/日志条数不同）。
详见 [reference.md](reference.md)。

## 4. 更多资料

* 全量环境变量：`pprobe env`（带当前取值）/ `pprobe env --markdown`（表格），权威定义在
  `src/pprobe/config.py` 的 `ENV_DOCS`。
* 原理与限制、结果字段字典：仓库根 `README.md`（§3 结构、§6 变量表、§8 原理、§9 开销）。
* 拼错变量名检查：`pprobe env --check`。
