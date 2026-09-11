# Torch Precision Debugger (TPD)

用于定位基于 PyTorch 训练框架（如 Megatron、LLaMA-Factory、ms-swift 等）精度问题的辅助工具。

## 功能特性

- **自动注入**: 通过环境变量和 `.pth` 文件机制，无需修改代码即可注入到 Python 解释器
- **Hook 技术**: 自动 hook 所有 `nn.Module` 的 forward/backward，捕获输入输出张量
- **智能采样**: 支持均匀采样和随机采样，可配置采样数量和随机种子
- **张量摘要**: 记录 shape、stride、device、dtype、max、min、mean、var、nan_count、inf_count
- **堆栈追踪**: 使用 ID 去重机制，避免重复存储相同堆栈
- **定期保存**: 可配置保存间隔，防止程序中断导致数据丢失
- **信号处理**: 捕获 SIGINT/SIGTERM 等信号，优雅退出并保存数据
- **多卡支持**: 支持分布式训练，按 rank 分别存储和对比
- **对比工具**: 提供 CLI 命令对比两次运行的结果，定位精度差异
- **总结报告**: 提供 `report` 命令一键生成全面的总结报告，快速了解问题全貌
- **结果查询**: 提供强大的 `cmp` 子命令系统，高效查询比较结果
- **问题定位**: 自动检测 NaN/Inf、大差异、shape 不匹配等问题
- **堆栈查询**: 提供 CLI 命令通过 ID 查询完整堆栈

## 安装

```bash
# 创建 conda 环境
conda create -n tpd python=3.12
conda activate tpd

# 安装 PyTorch
pip install torch

# 安装 TPD
cd torch-precision-debugger
pip install -e .
```

## 环境变量配置

| 环境变量                  | 说明                                               | 默认值            |
| ------------------------- | -------------------------------------------------- | ----------------- |
| `TPD_ENABLED`           | 启用 TPD（设为 1 启用）                            | `0`             |
| `TPD_OUTPUT_DIR`        | 输出目录                                           | `./tpd_results` |
| `TPD_SAMPLE_COUNT`      | 每个张量采样的元素个数                             | `50`            |
| `TPD_SAMPLE_MODE`       | 采样模式：`uniform`（均匀）或 `random`（随机） | `uniform`       |
| `TPD_SAMPLE_SEED`       | 随机采样种子（不设置则使用随机种子）               | 无                |
| `TPD_MAX_STEPS`         | 最大 hook 调用次数（0 表示无限制）                 | `0`             |
| `TPD_SAVE_INTERVAL`     | 定期保存间隔（每 N 次 hook 调用保存一次）          | `100`           |
| `TPD_MODULE_FILTER`     | 模块过滤器（逗号分隔的模块名前缀）                 | 空（所有模块）    |
| `RANK` / `LOCAL_RANK` | 分布式训练的 rank                                  | `0`             |
| `WORLD_SIZE`            | 分布式训练的总进程数                               | `1`             |

## 使用方法

### 1. 基本使用

```bash
# 启用 TPD 运行训练脚本
TPD_ENABLED=1 python train.py

# 自定义输出目录和采样配置
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./results_run1 \
TPD_SAMPLE_COUNT=100 \
TPD_SAMPLE_MODE=random \
TPD_SAMPLE_SEED=42 \
python train.py
```

### 2. 限制 hook 次数

```bash
# 只捕获前 1000 次 hook 调用
TPD_ENABLED=1 \
TPD_MAX_STEPS=1000 \
TPD_SAVE_INTERVAL=100 \
python train.py
```

### 3. 过滤特定模块

```bash
# 只 hook Linear 和 Conv2d 模块
TPD_ENABLED=1 \
TPD_MODULE_FILTER=Linear,Conv2d \
python train.py
```

### 4. 多卡分布式训练

```bash
# 使用 torchrun 启动分布式训练
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./results \
torchrun --nproc_per_node=4 train.py
```

每个 rank 会生成独立的结果文件：

- `rank0.jsonl`
- `rank1.jsonl`
- `stacks_rank0.json`
- `stacks_rank1.json`
- ...

### 5. 对比两次运行结果

```bash
# 对比两个结果目录
tpd compare ./results_run1 ./results_run2 -o comparison.json

# 指定 rank 和容差
tpd compare ./results_run1 ./results_run2 \
  --rank 0 \
  --tolerance 1e-5 \
  -o comparison.json
```

对比结果包含：

- 只在 dir1 中存在的条目
- 只在 dir2 中存在的条目
- 共同条目中的差异：
  - 基本信息差异（shape、stride、device、dtype）
  - 数值统计差异（max、min、mean、var、nan_count、inf_count）
  - 采样值差异

### 6. 生成总结报告

对比完成后，先生成一份全面的总结报告，快速了解问题全貌：

```bash
# 生成总结报告（默认输出到 comparison.report.txt）
tpd report comparison.json

# 指定输出文件
tpd report comparison.json -o my_report.txt

# 调整大差异阈值
tpd report comparison.json --threshold 0.5
```

**报告内容包括**：

1. **概览**：对比基本信息和统计
2. **问题统计**：NaN/Inf、大差异、Shape 不匹配的数量和严重程度
3. **首次出现位置**：各类问题首次出现的 step、module、tensor
4. **关键问题**：所有 NaN/Inf 问题详情
5. **Top 10 最大差异**：差异最大的条目排名
6. **Shape 不匹配**：所有 shape 问题详情
7. **Hook 类型分布**：问题在不同 hook 类型的分布
8. **排查建议**：根据问题类型给出针对性建议

**报告示例**：

```
================================================================================
  TPD Precision Debug Report
================================================================================
  Generated: 2026-09-11 14:30:00
  Source:    comparison.json

────────────────────────────────────────────────────────────────────────────────
  1. Overview
────────────────────────────────────────────────────────────────────────────────
  Dir1:                     ./results_run1
  Dir2:                     ./results_run2
  Rank:                     0
  Tolerance:                1e-06
  Total entries dir1:       92
  Total entries dir2:       92
  Common entries:           92
  Entries with differences: 92

────────────────────────────────────────────────────────────────────────────────
  2. Issue Statistics
────────────────────────────────────────────────────────────────────────────────
  Total entries with differences: 92
  ├── NaN/Inf issues:             1  ⚠️  CRITICAL
  ├── Large diff issues:          26  (threshold: 1.0)
  └── Shape mismatches:           0  ✓

  Overall Severity: HIGH

────────────────────────────────────────────────────────────────────────────────
  3. First Occurrences (by step)
────────────────────────────────────────────────────────────────────────────────
  First NaN/Inf Issue:
    Index:     87
    Step:      27
    Hook Type: forward_input
    Module:    torch.nn.modules.linear.Linear
    Tensor:    args[0]
    Stack ID:  S000012_88808c57116f

  First Large Diff Issue:
    Index:     0
    Step:      1
    Hook Type: backward_grad_output
    Module:    __main__.SimpleModel
    Tensor:    grad_output[0]
    Max Diff:  8.991413e-02
    Stack ID:  S000005_a36a855ab094

...
```

**使用流程**：

```bash
# 1. 对比结果
tpd compare ./run1 ./run2 -o comparison.json

# 2. 生成报告（先看全貌）
tpd report comparison.json

# 3. 查看报告
cat comparison.report.txt

# 4. 根据报告中的索引，使用 cmp 命令细查
tpd cmp show comparison.json 87 --window 3
tpd cmp first comparison.json --type naninf --window 3
```

### 7. 查询比较结果（cmp 子命令）

TPD 提供了一套强大的 `cmp` 子命令系统，用于高效查询和探索比较结果，避免加载整个 JSON 文件浪费上下文。

#### 7.1 查看概览

```bash
# 显示比较结果概览（含问题分类统计）
tpd cmp summary comparison.json
```

输出示例：

```
=================================================================
  TPD Comparison Summary
=================================================================
  Dir1:                     ./results_run1
  Dir2:                     ./results_run2
  Rank:                     0
  Tolerance:                1e-06
  Total entries dir1:       92
  Total entries dir2:       92
  Common entries:           92
  Entries with differences: 92
-----------------------------------------------------------------
  NaN/Inf issues:           1
  Large diff issues:        26
  Shape mismatch issues:    0
=================================================================
```

#### 7.2 定位首次出现的问题

```bash
# 查找首次出现 NaN/Inf 的位置（默认窗口大小 3）
tpd cmp first comparison.json --type naninf --window 3

# 查找首次出现大差异的位置（阈值可调）
tpd cmp first comparison.json --type large-diff --threshold 0.5 --window 2

# 查找首次出现 shape 不匹配的位置
tpd cmp first comparison.json --type shape
```

该命令会：
- 按 step 顺序找到首次出现该问题的条目
- 显示触发该问题的具体 diff 信息
- 显示前后窗口范围内的上下文条目

#### 7.3 列出所有差异条目

```bash
# 列出所有差异条目（按 step 排序）
tpd cmp list comparison.json --sort step

# 只列出 NaN/Inf 问题
tpd cmp list comparison.json --type naninf

# 只列出大差异问题（阈值 1.0）
tpd cmp list comparison.json --type large-diff --threshold 1.0

# 只列出 shape 不匹配问题
tpd cmp list comparison.json --type shape
```

输出示例：

```
  Idx    Step   Type                   Issues   Module / Tensor
  ----------------------------------------------------------------------------------------------------
  0      1      backward_grad_output   7        __main__.SimpleModel [grad_output[0]]
  1      1      forward_input          7        __main__.SimpleModel [args[0]]
  2      2      backward_grad_input    6        torch.nn.modules.normalization.LayerNorm [grad_input[0]]
  ...
```

#### 7.4 查看指定条目详情

```bash
# 查看索引 5 的条目详情
tpd cmp show comparison.json 5

# 查看索引 5 的条目，前后各显示 2 个上下文条目
tpd cmp show comparison.json 5 --window 2

# 查看索引 5~10 范围的条目
tpd cmp show comparison.json 5-10
```

#### 7.5 统计差异条目

```bash
# 统计所有差异条目（按问题类型和 hook_type 分类）
tpd cmp count comparison.json

# 只统计 NaN/Inf 问题
tpd cmp count comparison.json --type naninf

# 统计大差异问题（阈值可调）
tpd cmp count comparison.json --type large-diff --threshold 0.5
```

输出示例：

```
==================================================
  Diff Count Summary
==================================================
  Total entries with diffs: 92
  |-- NaN/Inf issues:       1
  |-- Large diff issues:    26  (>= 1.0)
  +-- Shape mismatches:     0

  By hook type:
    forward_input: 28
    forward_output: 28
    backward_grad_output: 24
    backward_grad_input: 12
```

#### 7.6 问题类型说明

`cmp` 命令支持三种问题类型过滤：

- **`naninf`**: NaN/Inf 问题（`nan_count` 或 `inf_count` 在两次运行间出现差异）
- **`large-diff`**: 大数值差异（`abs_diff` 超过阈值，默认 1.0）
- **`shape`**: Shape 不匹配（shape、stride、numel、dtype、device 等元信息不一致）

### 8. 查询堆栈追踪

```bash
# 通过堆栈 ID 查询完整堆栈
tpd stack ./results_run1 S000001_abc123def456

# 指定 rank
tpd stack ./results_run1 S000001_abc123def456 --rank 0
```

## 输出文件格式

### JSONL 文件（rank0.jsonl）

每行一个 JSON 记录，包含：

```json
{
  "step": 1,
  "hook_type": "forward_input",
  "module_name": "torch.nn.modules.linear.Linear",
  "module_id": 140234567890,
  "tensor_path": "args[0]",
  "stack_id": "S000001_abc123def456",
  "summary": {
    "shape": [32, 768],
    "stride": [768, 1],
    "device": "cuda:0",
    "dtype": "torch.float32",
    "numel": 24576,
    "max": 2.345,
    "min": -1.234,
    "mean": 0.012,
    "var": 0.987,
    "nan_count": 0,
    "inf_count": 0,
    "raw_max": 2.345,
    "raw_min": -1.234
  },
  "samples": [0.123, -0.456, 0.789, ...]
}
```

### 堆栈文件（stacks_rank0.json）

```json
{
  "S000001_abc123def456": [
    "  File \"train.py\", line 42, in <module>\n    model(inputs)\n",
    "  File \"model.py\", line 156, in forward\n    x = self.linear(x)\n",
    ...
  ],
  ...
}
```

## 工作原理

1. **注入机制**: 安装时创建 `tpd.pth` 文件到 site-packages，Python 启动时自动加载
2. **模块监控**: 使用 `sys.settrace` 监控所有 `nn.Module` 的创建
3. **Hook 注册**: 为每个模块注册 forward pre-hook、forward hook 和 backward hook
4. **数据采集**: 在 hook 中捕获张量，计算摘要信息并采样
5. **堆栈追踪**: 捕获调用堆栈，使用 MD5 哈希去重
6. **持久化**: 以 JSONL 格式追加写入，定期保存堆栈信息
7. **信号处理**: 捕获中断信号，确保数据完整保存

## 开发

```bash
# 开发模式安装
pip install -e .

# 运行测试
python test_tpd.py
```

## 许可证

MIT
