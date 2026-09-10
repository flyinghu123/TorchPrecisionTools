# pprobe —— PyTorch 训练精度问题定位探针

给基于 PyTorch 的训练框架（megatron-lm、LLaMA-Factory、ms-swift、transformers、deepspeed …）
定位**数值精度问题**的辅助工具：不改动一行训练代码，用一个环境变量把探针注入解释器，
hook 每个模块 forward / backward 的输入输出，按采样 + summary + 堆栈 id 落盘，
再用命令行按 rank 对比两次运行（或两个平台）的差异，直接指出**最早发散的模块**。

典型场景：

* 换了硬件 / 换了 torch 版本 / 开了 TF32，loss 对不上，要找出是哪一层开始飘
* 8 卡里有 1 张卡数值不对（数据分片、通信 dtype、驱动差异），要定位到 rank 与模块
* loss 出 NaN/Inf，要知道**第一个** NaN 是哪个模块的哪个输出产生的
* 权重加载/转换（fp32→bf16、checkpoint 合并、LoRA 合并）后数值变化

不需要 GPU、不需要多机就能全部验证：`pytest tests/`（含 torchrun 多进程用例），
或 `pprobe selftest` 一键端到端自检。

目录：[1 安装与注入](#1-安装与注入) · [2 快速开始](#2-快速开始) ·
[3 结果目录结构](#3-结果目录结构) · [4 命令一览](#4-命令一览) ·
[5 四个典型定位流程](#5-四个典型定位流程) · [6 环境变量参考表](#6-环境变量参考表) ·
[7 接到真实框架上](#7-接到真实框架上) · [8 工作原理](#8-工作原理) ·
[9 开销与限制](#9-开销与限制) · [10 开发与测试](#10-开发与测试) ·
[11 目录结构](#11-目录结构) · [12 需求对照](#12-需求对照)

---

## 1. 安装与注入

```bash
conda activate tmp                    # 要被注入的那个环境（本仓库开发用的是 conda tmp 环境）
pip install -e .                      # 运行时零强依赖，torch 由目标环境提供
pprobe install                        # 写入 site-packages/pprobe.pth，完成解释器注入
pprobe status                         # 确认「下次启动是否生效」
```

`pprobe.pth` 里只有一行（前面一行注释），且用 `and` 短路：

```python
import os as _pprobe_os, sys as _pprobe_sys; _pprobe_os.environ.get("PPROBE_ENABLE", "").strip().lower() in ("1", "true", "yes", "y", "on") and __import__("pprobe.bootstrap", fromlist=["boot_from_pth"]).boot_from_pth()
```

**不设 `PPROBE_ENABLE` 时连 `pprobe` 都不会被 import**，对同一个环境里其它 Python 进程零影响
（`tests/test_inject.py` 用子进程黑盒验证了这一点）。注入覆盖 `python train.py`、
`torchrun --nproc_per_node=N`、`python -m ...`、notebook 内核等所有解释器入口。

不想污染环境，可以只用临时注入：

```bash
pprobe run --out ./dbg --sample-n 200 -- python train.py --lr 1e-4   # 等价于设好环境变量再 exec
pprobe uninstall                                                      # 删除 pprobe.pth
```

## 2. 快速开始

```bash
# ① 两次运行各采一份数据（除了 OUT，其它环境变量都可以不给）
PPROBE_ENABLE=1 PPROBE_OUT=./runs/base python examples/mlp_simple.py --device cpu
PPROBE_ENABLE=1 PPROBE_OUT=./runs/eps   python examples/mlp_simple.py --device cpu --eps 1e-8

# ② 对比：谁先发散、差多少、在哪个模块/哪次调用
pprobe compare ./runs/base ./runs/eps --out diff.md      # 同时写出 diff.json

# ③ 报告里给的是堆栈 id，用它查完整调用链
pprobe stack s4 --result ./runs/base --events
```

`compare` 输出的关键几行（本机真实输出，torch 2.12 / CPU / 3 step）：

```
[pprobe] ⚠️ 最早发散点 rank0: forward:TinyModel.blocks.0.norm#0  (kinds=sample,stats, stack=s4)
[pprobe] events_compared=96 | divergent_events=94 | stats_diff=85 | sample_diff=90 | stack_diff=0
```

`#0` 是该模块的第 0 次调用 —— 也就是**第一层第一个 forward 就已经不一样了**，
它下游的所有差异都可能是传播结果。

## 3. 结果目录结构

```
runs/base/
├── run.json            # 全局：world_size、参与的 rank、配置快照
├── HINTS.txt           # 拿来即用的后续命令（report/compare/stack/query），含本次采样配置
└── rank0/              # 每个 rank 一个目录（需求 8）
    ├── events.jsonl    # 一行一条 forward/backward 事件（追加写，可被 kill）
    ├── stacks.json     # 堆栈 id → 完整调用栈，去重存放（需求 4）
    ├── manifest.json   # 配置、采样种子、计数器、结束原因
    ├── env.json        # torch/cuda/后端开关/命令行等环境快照
    ├── exit.json       # 退出原因与最终计数（atexit / signal / limit 都写）
    └── tensors/*.pt    # 只有 PPROBE_SAVE_FULL 命中时才有内容
```

一条 forward 事件（`events.jsonl`）的骨架，字段取自真实产物：

```json
{"seq": 1, "phase": "forward", "rank": 0, "step": 0, "pid": 209421,
 "module": "TinyModel.embed", "module_cls": "Linear", "call_index": 0,
 "stack_id": "s2", "output_sig": "Tensor", "grad_enabled": true,
 "n_tensors": 2, "t_abs": 1789057662.918, "wall": 0.51,
 "tensors": {"input[0]": {"kind": "tensor",
     "basic": {"shape": [16,32], "stride": [32,1], "dtype": "torch.float32", "device": "cpu",
               "numel": 512, "nbytes": 2048, "contiguous": true, "storage_offset": 0,
               "version": 0, "requires_grad": false, "is_leaf": true, "is_sparse": false,
               "is_meta": false, "cls": "Tensor",
               "dtype_limits": {"eps": 1.19e-07, "max": 3.4e+38, "min": -3.4e+38, "tiny": 1.18e-38}},
     "stats": {"count": 512, "max": 3.185, "min": -2.552, "mean": 0.0143, "var": 0.9748,
               "std": 0.9873, "sum": 7.3007, "checksum": 7.3007, "absmax": 3.185,
               "l2norm": 22.321, "zero_count": 0, "nan_count": 0,
               "inf_count": 0, "posinf_count": 0, "neginf_count": 0},
     "sample": {"n": 50, "idx": [0, 10, 21, "..."], "idx_nd": [[0,0], [0,10], "..."],
                "vals": [0.0248, "..."]}}}}
```

backward 事件把 `output_sig`/`grad_enabled` 换成四个回填字段：`fwd_seq`（能对回哪一次前向）、
`fwd_stack_id`、`bwd_index`（第几次 `.backward()` 调用）、`bwd_src`（`module-hook` 或 `tensor-hook`）。

`stacks.json` 是 `{"count": 10, "total_hits": 96, "stacks": {"s2": {...}}}`，每条堆栈：

```json
{"id": "s2", "caller": "/…/examples/mlp_simple.py:86 in main", "count": 3,
 "signature": "…:<module>\n…:main", "frames": [
   {"file": "…/mlp_simple.py", "line": 100, "name": "<module>",
    "text": "…/mlp_simple.py:100 in <module>", "source": "raise SystemExit(main())"},
   {"file": "…/mlp_simple.py", "line": 86, "name": "main", "source": "logits = model(x)"}]}
```

* **采样**：默认 `uniform` 均匀取 50 个元素（需求 3），`idx_nd` 给出对应的多维下标；
  `PPROBE_SAMPLE_MODE=random` + `PPROBE_SEED` 可复现；`PPROBE_SAMPLE_EXTRA=bits` 会再多一个
  `bits` 字段（每个采样元素的原始位模式，跨平台逐 bit 对比时用）。
* **summary**：`basic`（shape/stride/device/dtype/contiguous/version/dtype_limits…）+
  `stats`（最值/均值/方差/checksum/nan 个数/inf 个数/zero 个数）正是需求 4 要求的信息。
* **槽位命名**：`input[i]`、`kwargs.<名>`、`output`、`output[i]`/`output.<键>`、`param.<名>`（需 `PPROBE_RECORD_PARAMS=1`）；
  反向为 `grad_input[i]`、`grad_output[i]`（模块级 hook）或 `grad_of:<槽位>`（张量级 hook）。
* 没有对应 Module 的算子（`PPROBE_HOOK` 加 `func`）记成 `module="functional.gelu"`，并用 `in_module` 标出当时所在的模块。
* JSON 严格合法：NaN/Inf 编码成 `"NaN"` / `"Inf"` / `"-Inf"` 字符串，不会被 JSON 解析器吞掉。

## 4. 命令一览

| 命令 | 作用 |
| --- | --- |
| `pprobe install [--dir D] [--python PY] [--dry-run]` | 写入 `pprobe.pth`（需求 1） |
| `pprobe uninstall` | 删除自己写的 `pprobe.pth`（别人的同名文件不碰） |
| `pprobe status` | 注入状态、候选目录、`PPROBE_ENABLE` 当前值、下次启动是否生效 |
| `pprobe env [--check] [--markdown] [--filter KEY]` | 全部 `PPROBE_*` 变量与当前取值；`--check` 会揪出拼错的变量名 |
| `pprobe run [--out --sample-n --sample-mode --seed --max-events --flush-interval --include --hook --set K=V] -- 脚本` | 不安装也临时注入运行 |
| `pprobe report DIR [--rank 0,1] [--top] [--json] [--markdown]` | 单目录速览：NaN/Inf 热点、量级排行、可疑模块 |
| `pprobe query DIR [--module RE] [--phase] [--nan] [--inf] [--slot RE] [--seq-min/--seq-max] [--json]` | 按条件捞事件 |
| `pprobe compare A B [--out F] [--json --no-write] [--atol --rtol --top --detail] [--rank 0:3] [--ranks 0,1] [--phase] [--include/--exclude/--ignore-slot] [--sort seq\|max_abs\|max_rel] [--no-env] [--no-stack] [--stdout-md] [--fail-if-diff]` | **按 rank 对比**两次运行（需求 6），输出差异 + 堆栈 id |
| `pprobe stack ID[,ID] --result DIR [--rank R] [--events] [--module RE] [--json]` | 堆栈 id → 完整调用栈（需求 7）；`--module` 可反查某模块涉及的 id |
| `pprobe selftest` | 端到端自检：注入 → 采集 → 查询 → 对比，一次跑完 |

对比时的对齐 key 是 `(rank, phase, 模块路径, call_index)`（不是自增 `seq`），所以即使两次运行的
事件总数不同、多打了几个日志，能配对的部分依然对齐；配不上对的会单独列为「仅 A / 仅 B 存在」。

差异类型（`kinds`）：`presence` 只在一边存在、`info` 基本信息不同（dtype/shape/device/模块类名）、
`special` NaN/Inf 个数不同、`stats` 数值统计不同、`sample` 采样元素不同、`scalar` 标量参数不同
（eps、reduction 这类）、`index_mismatch` 采样下标不一致、`stack_diff` 调用栈不同。

`--rank 0:3` 是**跨目录配对 rank**（两边编号不同时用），`--ranks 0,1` 是**只筛这些 rank**（两边同号）。

## 5. 四个典型定位流程

### 5.1 换平台/换版本后 loss 对不上（CPU ↔ GPU、TF32）

```bash
PPROBE_ENABLE=1 PPROBE_OUT=./runs/cpu python examples/mlp_simple.py --device cpu
PPROBE_ENABLE=1 PPROBE_OUT=./runs/gpu python examples/mlp_simple.py --device cuda
pprobe compare ./runs/cpu ./runs/gpu --out gpu.md --detail 20
```

先看报告顶部的**环境差异**（TF32、matmul precision、torch/cuda 版本、后端开关都在这里）
与**探针配置差异**（采样个数/种子不同会让采样值失去可比性，先确认这两块），
再看「最早发散点」——如果只有少数模块超容差而下游全部被带偏，根因就是它。
逐 bit 对比可以加 `PPROBE_SAMPLE_EXTRA=bits`，判断是否要求全等则加 `PPROBE_FULL_HASH=1`。

### 5.2 多卡里某一张卡不对（分 rank 采集 + 分 rank 对比）

```bash
PPROBE_ENABLE=1 PPROBE_OUT=./runs/base torchrun --nproc_per_node=2 examples/ddp_simple.py
PPROBE_ENABLE=1 PPROBE_OUT=./runs/bad  torchrun --nproc_per_node=2 examples/ddp_simple.py --bad-rank 1
pprobe compare ./runs/base ./runs/bad --out ddp.md
```

报告里每个 rank 都有自己的最早发散点（本机真实输出）：

```
### 各 rank 最早发散点
| rank | 最早发散调用 | A seq | 类型 | 堆栈 | 最大绝对差 |
| --- | --- | --- | --- | --- | --- |
| rank0 | `forward:DistributedDataParallel.module.fc1#1` | 15 | sample | s3 | 1.2517e-06 |
| rank1 | `forward:DistributedDataParallel.module.norm#0` |  2 | sample,stats | s3 | 0.0002450943 |
```

一眼看出 rank1 在**第一次前向**就不对（eps 被改过），而 rank0 要到第一次参数更新之后
才被 all-reduce 的梯度带偏（`fc1` 的第 1 次调用）——这就是根因在 rank1 的证据链。
只想在个别 rank 上采集（8 卡太占盘）用 `PPROBE_ONLY_RANKS=0,1`；
两边 rank 编号不同时用 `pprobe compare A B --rank 0:0 --rank 1:3` 手工配对。

### 5.3 NaN / Inf 溯源

```bash
pprobe report ./pprobe_out                       # 哪个模块哪个槽位有 NaN、第一次出现的 seq
pprobe query ./pprobe_out --nan --module '.*'    # 逐条列出含 NaN 的事件
pprobe stack <报告里的 stack_id> --result ./pprobe_out --events
```

想直接在第一个 NaN 处断下来：

```bash
PPROBE_STOP_ON_NAN=1 PPROBE_ENABLE=1 python train.py   # 前向发现 NaN 立刻抛 PProbeLimitReached
PPROBE_STOP_ON_NAN=1 python -m pdb train.py            # 崩在案发现场，locals 齐全
```

### 5.4 怀疑权重加载/转换有问题

```bash
PPROBE_RECORD_PARAMS=1 PPROBE_INCLUDE='.*embed.*|.*qkv.*' PPROBE_ENABLE=1 python train.py
pprobe query ./pprobe_out --module qkv --json   # param.weight 的 stats/checksum 两次对比
```

## 6. 环境变量参考表

唯一的权威定义在 `src/pprobe/config.py` 的 `ENV_DOCS`，下表由 `pprobe env --markdown` 生成；
`pprobe env` 还会在后面附上当前进程里的实际取值。

<!-- ENV-TABLE-START：由 `pprobe env --markdown` 生成，勿手改 -->
| 环境变量 | 默认值 | 类型 | 说明 |
| --- | --- | --- | --- |
| ``PPROBE_ENABLE`` | 0 | bool | 总开关；不设则 .pth 注入完全不生效（连 pprobe 都不会 import） |
| ``PPROBE_OUT`` | ./pprobe_out | path | 结果根目录，下面按 rankN/ 分目录 |
| ``PPROBE_RUN_NAME`` |  | str | 附加到 manifest 的标签，方便区分两次运行 |
| ``PPROBE_VERBOSE`` | 1 | bool | 启动/结束时向 stderr 打印探针状态 |
| ``PPROBE_PER_RANK_DIR`` | 1 | bool | 是否按 rank 分目录（多卡必须为 1） |
| ``PPROBE_RANK`` | 自动检测 | int | 强制指定 rank（优先于 torch.distributed / 环境变量） |
| ``PPROBE_WORLD_SIZE`` | 自动检测 | int | 强制指定 world_size（优先于 torch.distributed / WORLD_SIZE） |
| ``PPROBE_LOCAL_RANK`` | 自动检测 | int | 指定 local rank（只在 ``LOCAL_RANK`` 没设时生效，写进 env.json） |
| ``PPROBE_ONLY_RANKS`` | 全部 | list[int] | 只在指定 rank 上采集，如 ``0`` 或 ``0,3`` |
| ``PPROBE_RECORD_ENV`` | 1 | bool | 写 rankN/env.json（torch/cuda/后端开关等环境快照） |
| ``PPROBE_SAMPLE_MODE`` | uniform | uniform\|random\|head\|off | 每个 tensor 的采样方式 |
| ``PPROBE_SAMPLE_N`` | 50 | int | 每个 tensor 采样元素个数 |
| ``PPROBE_SEED`` | 不设=随机 | int | 随机采样的种子；不设则用熵源种子并记入 manifest |
| ``PPROBE_SAMPLE_LAYOUT`` | flat | flat\|grid | 均匀采样的布局：扁平等间隔或逐维网格 |
| ``PPROBE_SAMPLE_EXTRA`` |  | list | 附加采样信息，如 ``bits``（原始位模式，跨平台逐 bit 对比） |
| ``PPROBE_SUMMARY`` | 1 | bool | 记录 tensor 基本信息（shape/stride/device/dtype 等） |
| ``PPROBE_STATS`` | 1 | bool | 记录数值统计（max/min/mean/var/nan/inf 个数等） |
| ``PPROBE_STATS_DTYPE`` | float64 | float64\|float32 | 统计累加精度；GPU 上想省时间可设 float32 |
| ``PPROBE_CHECKSUM`` | 1 | bool | 记录 float64 求和 checksum（同一下标序列可跨运行对齐） |
| ``PPROBE_FULL_HASH`` | 0 | bool | 对完整 tensor 取位模式 blake2b（很慢，定位“是否完全一致”时用） |
| ``PPROBE_RECORD_SCALARS`` | 1 | bool | 记录输入里的 int/float/str/None 标量（eps、dtype 等） |
| ``PPROBE_RECORD_PARAMS`` | 0 | bool | 首次调用时记录模块参数（权重加载差异） |
| ``PPROBE_HOOK`` | forward,backward,optim | list | 开启哪几类 hook；``func`` 额外 hook nn.functional |
| ``PPROBE_BWD_MODE`` | module | module\|tensor\|both | 反向采集方式；module=模块边界梯度，tensor=张量 register_hook |
| ``PPROBE_INCLUDE`` |  | list[regex] | 只记录模块路径匹配任一正则的模块 |
| ``PPROBE_EXCLUDE`` |  | list[regex] | 排除模块路径匹配任一正则的模块 |
| ``PPROBE_INCLUDE_CLS`` |  | list[regex] | 只记录类名匹配的模块 |
| ``PPROBE_EXCLUDE_CLS`` |  | list[regex] | 排除类名匹配的模块（如 Dropout） |
| ``PPROBE_FUNC_TARGETS`` | 内置 12 个 | list | PPROBE_HOOK 含 func 时要包住的 nn.functional 函数名 |
| ``PPROBE_TRAVERSE_DEPTH`` | 3 | int | 展开嵌套 list/tuple/dict 输入输出的最大层数 |
| ``PPROBE_MAX_TENSORS_PER_CALL`` | 64 | int | 单次 forward 最多记录多少个 tensor 槽位（0=不限） |
| ``PPROBE_MAX_EVENTS`` | 0 | int | 最多记录多少次 forward/backward，达到即落盘收尾 |
| ``PPROBE_MAX_STEPS`` | 0 | int | 最多记录多少个 optimizer step |
| ``PPROBE_MAX_CALLS_PER_MODULE`` | 0 | int | 单个模块最多记录多少次调用（控制热点模块体量） |
| ``PPROBE_FLUSH_INTERVAL`` | 50 | int | 每多少条事件 flush + fsync 一次（防丢数据的关键参数） |
| ``PPROBE_FLUSH_SECS`` | 0 | float | 额外按时间定期 flush（秒），0=关闭 |
| ``PPROBE_SIGNALS`` | 1 | bool | 捕获 SIGINT/SIGTERM/SIGHUP/SIGQUIT 先落盘再交回原 handler |
| ``PPROBE_STOP_ON_NAN`` | 0 | bool | 发现 NaN 立刻抛异常中断训练（配合 pdb 定位） |
| ``PPROBE_FAULTHANDLER`` | 0 | bool | 启用 faulthandler，段错误时输出 C 层堆栈 |
| ``PPROBE_SAVE_FULL`` |  | list[regex] | 对匹配模块把完整 tensor 存成 rankN/tensors/*.pt |
| ``PPROBE_SAVE_FULL_MAX_BYTES`` | 64M | bytes | 完整 tensor 落盘的体积上限（超过则只记 summary） |
| ``PPROBE_SAVE_FULL_EVERY`` | 1 | int | 每 N 次调用存一份完整 tensor |
| ``PPROBE_STACKS`` | 1 | bool | 记录堆栈（以 id 引用，完整堆栈集中在 stacks.json） |
| ``PPROBE_STACK_TRIM`` | 1 | bool | 裁掉 torch 内部帧，只保留用户代码调用链 |
| ``PPROBE_STACK_LIMIT`` | 48 | int | 单条堆栈最多保留多少帧 |
| ``PPROBE_KEEP_WARNINGS`` | 0 | bool | 保留探针自己触发的 UserWarning（默认静默） |

<!-- ENV-TABLE-END -->

上表之外还有三个变量：**`PPROBE_TRACEBACK=1`** 让探针自己的异常完整 traceback 到 stderr
（默认只打一行摘要，因为探针绝不能搞崩训练）；**`PPROBE_SEED_USED`** 是探针回写的——
`SAMPLE_MODE=random` 且没给 `PPROBE_SEED` 时，实际用到的种子会被放回环境，
子进程读到它就沿用同一个种子（因而采样下标能对齐），不需要手设；**`PPROBE_MODULE_NAME_MAXLEN`**（默认 200）
只是 `Config` 的内部字段，不从环境读，不是可设的环境变量。

表由 `pprobe env --markdown` 生成，`tests/test_cli.py::test_readme_env_table_is_in_sync` 会校验它没漂。

## 7. 接到真实框架上

* **torchrun / accelerate / deepspeed**：`PPROBE_ENABLE=1` 会随环境变量继承到每个 worker；
  rank 的取值优先级是 `PPROBE_RANK` → 已初始化的 `dist.get_rank()` → `RANK` → 0；
  `LOCAL_RANK`/`PPROBE_LOCAL_RANK` 只记进 env.json，不参与 rank 判定。
* **megatron-lm**：`PPROBE_INCLUDE='.*\.(linear_attn|qkv|core_attn|mlp)\..*'` 缩小范围；
  流水线/上下文并行下全局 rank 依然唯一，跨目录编号不同就用 `compare --rank 0:8` 手工配对。
* **LLaMA-Factory / ms-swift**：直接 `PPROBE_ENABLE=1 PPROBE_OUT=./dbg llamafactory-cli train examples/train_lora/xxx.yaml`；
  想只看基座不要 LoRA 分支：`PPROBE_EXCLUDE_CLS='.*lora.*'`。
* **自定义训练循环（不走 torch.optim 子类）**：每步调一次 `pprobe.mark_step()`，事件上的 `step` 标签就正确了。
* **DataLoader worker（fork）**：子进程被 `os.register_at_fork` 识别，不会重复写数据。
* **`model = torch.compile(...)` / jit trace**：dynamo 追溯阶段的调用会被跳过，不会把编译期符号当成真实数值。
* 显式启动（不想用 .pth）：`import pprobe; pprobe.init(out_dir="./dbg", sample_n=200, max_events=2000)`。

## 8. 工作原理

| 环节 | 做法 | 为什么 |
| --- | --- | --- |
| 注入 | `site-packages/pprobe.pth` 里一行 `and` 短路的 import | 不改代码、不进 sitecustomize、未开启时零开销 |
| 装载时机 | `sys.meta_path` 上挂 `_ImportWatcher`，torch 导入完成才装 hook | 解释器启动阶段不 import torch，拖慢不了训练 |
| forward hook | 替换 `torch.nn.Module.__call__`（类级） | 覆盖**已经创建好的**模型实例与子类自定义 `__call__`；一次拿到 args/kwargs/output/层级路径 |
| backward hook | `register_full_backward_hook` 为主，注册失败或 `bwd_mode=tensor/both` 时用 `Tensor.register_hook` 兜底 | 模块边界梯度最有可比性；无参数模块/多输入歧义靠张量级 hook 补上 |
| 反向事件回填 | 记录 `fwd_seq`/`fwd_stack_id`/`call_index`/`bwd_index` | 反向能对回是哪一次前向，跨运行还能对齐 |
| step 计数 | 包 `Optimizer.__init__`，在实例化时包具体子类的 `step` | Adam/SGD 自己定义了 `step`，包基类不生效 |
| 堆栈 | 按签名 blake2b 去重成 `s1/s2/...`，集中在 `stacks.json`，写入按命中次数节流 | 同一条调用链被上千次调用只存一份 |
| 落盘 | JSONL 缓冲 + 每 N 条 `flush + fsync` + atexit + 信号链式转发 + 原子写元信息 | 被 kill / Ctrl-C / OOM 也尽量不丢数据（需求 5） |
| 目录延迟创建 | 第一次真正记录事件才建 `rankN/` | torchrun 的 launcher 进程自己会 import torch，不能污染 rank0 |
| 线程安全 | `RLock` + `threading.local` 重入保护 | autograd 引擎在自己的线程里回调 hook |

## 9. 开销与限制

* 单条事件的代价主要是 `stats`（若干次全 tensor reduction）与采样取值。**只想快速看一眼**：
  `PPROBE_STATS_DTYPE=float32`、`PPROBE_SAMPLE_N=16`、`PPROBE_MAX_CALLS_PER_MODULE=2`、`PPROBE_INCLUDE=<可疑模块>`。
* 生产建议加 `PPROBE_MAX_EVENTS` / `PPROBE_MAX_STEPS`，否则几百层 × 每步 × 每个 rank 会写很多。
* `SAVE_FULL` 存 `.pt` 全量，体积按 `SAVE_FULL_MAX_BYTES` 硬拦。
* 只 hook `torch.nn.Module`；手写函数式网络（没有 Module）用 `PPROBE_HOOK=forward,func` 兜底，
  或在关键位置 `with torch.autograd.profiler.record_function("x")` 之外自己 `pprobe.recorder().record(...)`。
* 探针自身异常一律被吞掉并计入 `manifest.counters.errors`，绝不会把训练带崩；
  唯一的例外是你显式要求的 `PPROBE_STOP_ON_NAN=1`。
* 多机（world_size > 单机卡数）需要把各机的 `PPROBE_OUT` 指到同一共享目录，或各自取回后
  用 `pprobe compare --rank 0:8` 手工配对 rank。
* 本仓库的测试是在 CPU 上跑的，2 个依赖真 GPU 的用例在没有可用 kernel 的机器上自动 skip
  （`tests/test_diff_localization.py::test_cpu_vs_cuda_*` 与 `tests/test_sampling.py` 的 GPU 采样用例）；
  GPU 侧的实测需要 torch 构建支持的 sm（本机是 sm_61 的 1050 Ti，而 2.12+cu132 需要 sm_75+，故 skip）。

## 10. 开发与测试

```bash
conda activate tmp                    # 本仓库的开发/验证环境（python 3.12 + torch 2.12.1+cu132）
pip install -e .[dev]
python -m pytest tests/                # 239 passed, 2 skipped（含 4 次真实 torchrun 拉起 2 rank）
pprobe selftest                        # 不装 pytest 也能一键验证注入→采集→查询→对比
```

用例分层：`test_sampling/test_summary/test_stacks/test_events/test_recorder/test_config/test_util` 是纯单元；
`test_result_query/test_compare/test_render/test_cli` 用 `conftest.build_run` 造标准结果目录；
`test_inject`（子进程 + `site.addsitedir` 验证 .pth 语义）、`test_hooks_e2e`（真实 forward/backward/optimizer/functional）、
`test_dist_e2e`（torchrun 2 进程、分 rank、SIGTERM 落盘）、`test_diff_localization`（两次运行差异定位）是端到端。
验证用的端到端流程也可手工复现：`examples/mlp_simple.py` 跑两份 + `pprobe compare`，
`examples/ddp_simple.py --bad-rank 1` 造一个“只有 rank1 不对”的场景（见 5.1 / 5.2）。

## 11. 目录结构

```
src/pprobe/
  bootstrap.py   解释器注入入口 + torch 懒加载 hook 监听
  inject.py      pprobe.pth 的安装/卸载/状态
  config.py      Config 与 ENV_DOCS（环境变量的唯一权威表）
  hooks.py       Module.__call__ / backward / functional / optimizer 四类 hook
  events.py      槽位收集（限深、限量、截断标记）
  sampling.py    uniform/random/head/off 采样与位模式
  summary.py     basic info + 数值统计 + checksum/full hash
  stacks.py      堆栈去重 intern 与 stacks.json 节流写
  recorder.py    JSONL 缓冲、flush/fsync、信号与 atexit、manifest/env/run.json
  distributed.py rank/world_size 探测与环境快照
  result.py      读结果目录、偏移索引、坏行容忍
  query.py       事件查询、堆栈反查、单目录速览
  compare.py     按 rank 配对比较（需求 6 的算法本体）
  render.py      Markdown 差异报告渲染
  cli.py         pprobe 命令行
  util.py        严格 JSON 编码、原子写、字节/列表解析
examples/        mlp_simple.py（单进程）、ddp_simple.py（多进程/bad-rank）
tests/           单元 + 端到端（torchrun、子进程注入）
```

## 12. 需求对照

| # | 需求 | 实现 | 验证方式 |
| --- | --- | --- | --- |
| 1 | 环境变量直接把工具注入解释器（pth / site-packages） | `inject.py` 写 `site-packages/pprobe.pth`；`bootstrap.py` 的 `boot_from_pth()` + `sys.meta_path` 监听 torch 导入 | `pprobe status`、`tests/test_inject.py`（子进程黑盒）、`pprobe selftest` |
| 2 | hook forward/backward 的输入输出定位精度问题 | `hooks.py`：替换 `Module.__call__`、`register_full_backward_hook`（失败回退 `Tensor.register_hook`）、`nn.functional`、`Optimizer.step` | `tests/test_hooks_e2e.py`（真实 fwd/bwd/optim/functional 事件） |
| 3 | 采样个数可控制；随机采样由 seed 管理，不设则真随机 | `sampling.py`（uniform/random/head/off）+ `PPROBE_SAMPLE_N`、`PPROBE_SEED`；实际用到的种子写进 `manifest.sampling.seed_used` | `tests/test_sampling.py`；`pprobe env --filter SAMPLE` |
| 4 | 采样元素 + summary（shape/stride/device/dtype + 最大/最小/均值/方差/nan/inf 个数）+ 堆栈用 id 表示、id→完整堆栈单独一个 json | `summary.py`（`basic`/`stats`）、`events.py`（槽位收集）、`stacks.py`（blake2b 去重成 `s1/s2/…`，集中写 `stacks.json`） | `tests/test_summary.py`、`tests/test_stacks.py`、`test_result_query.py` |
| 5 | 限制总 forward/backward 次数 + 按间隔定期落盘 + 捕获中断/人为停止 | `PPROBE_MAX_EVENTS`/`MAX_STEPS`/`MAX_CALLS_PER_MODULE`；`recorder.py` 的缓冲 + 每 `FLUSH_INTERVAL` 条 `flush+fsync`（可加 `FLUSH_SECS`）+ SIGINT/SIGTERM/SIGHUP/SIGQUIT 链式转发 + `atexit` + 原子写 | `tests/test_recorder.py`、`tests/test_dist_e2e.py`（SIGTERM 后数据完整）、`test_hooks_e2e.py`（达到上限即停） |
| 6 | 对比两 result 目录（采样值/summary/基本信息/堆栈），定位两次运行或两个平台的精度差异，输出到文件并记录差异与堆栈 id | `compare.py`（按 rank、按 `(phase, 模块, call_index)` 对齐）+ `render.py`（Markdown）；`compare` 默认把报告写到 B 目录旁的 `pprobe_compare_<A>_vs_<B>.md` 与同名 `.json` | `tests/test_compare.py`、`test_render.py`、`test_diff_localization.py`（CPU↔GPU、双运行差异定位） |
| 7 | 通过堆栈 id + result 目录查完整堆栈 | `query.resolve_stacks` / `format_stack`；`pprobe stack s5 --result DIR --events`，`--module RE` 可反向查 id | `tests/test_cli.py::…stack…`；报告里每条差异都附了可直接执行的 stack 命令 |
| 8 | 多卡分布式分 rank 存储，对比也分 rank | `distributed.py` 探 rank/world_size（`torch.distributed` 优先，回退 `RANK`/`LOCAL_RANK`）；`PPROBE_PER_RANK_DIR`、`PPROBE_ONLY_RANKS`；`compare` 逐 rank 配对并各自给出最早发散点 | `tests/test_dist_e2e.py`（真实 torchrun 2 进程）、`examples/ddp_simple.py --bad-rank`（见 5.2） |
