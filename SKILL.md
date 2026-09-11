# TPD 精度排查工具使用指南

## 概述

TPD (Torch Precision Debugger) 是一个用于定位 PyTorch 训练框架精度问题的辅助工具。通过自动 hook 所有 `nn.Module` 的 forward/backward，捕获张量数据，再通过 `tpd compare` 对比两次运行结果，并用 `tpd report` 一键生成总结报告（问题统计、首次出现位置、排查建议），帮助快速定位 NaN/Inf、数值偏差、shape 不匹配等问题。

## 适用场景

- 模型训练出现 NaN/Inf 导致 loss 爆炸
- 两次训练结果不一致，需要定位首次出现差异的位置
- 模型迁移（如 FP32→FP16、单机→多机）后精度下降
- 需要对比不同超参数/架构对中间层激活值的影响
- 分布式训练中某个 rank 出现异常

## 工作流程

### 第一步：数据采集

运行两次训练（正常 vs 异常，或两个不同配置），采集张量快照。

```bash
# 运行 1：基准运行（如 FP32、单卡、已知正确的版本）
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./results_baseline \
TPD_SAMPLE_COUNT=50 \
TPD_SAMPLE_MODE=uniform \
TPD_MAX_STEPS=500 \
python train.py

# 运行 2：问题运行（如 FP16、多卡、出现 NaN 的版本）
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./results_problem \
TPD_SAMPLE_COUNT=50 \
TPD_SAMPLE_MODE=uniform \
TPD_MAX_STEPS=500 \
python train.py
```

**关键参数说明**：

- `TPD_SAMPLE_COUNT`: 每个张量采样元素数（建议 20-100，越多越精确但文件越大）
- `TPD_SAMPLE_MODE`: `uniform`（均匀采样，推荐）或 `random`（随机采样）
- `TPD_MAX_STEPS`: 限制 hook 调用次数（避免文件过大，建议 500-2000）
- `TPD_SAVE_INTERVAL`: 保存间隔（默认 100，防止中断丢失数据）
- `TPD_MODULE_FILTER`: 只监控特定模块（如 `Linear,LayerNorm`）

**分布式训练**：

```bash
# 每个 rank 会自动生成独立文件
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./results \
torchrun --nproc_per_node=4 train.py
# 生成: rank0.jsonl, rank1.jsonl, rank2.jsonl, rank3.jsonl
```

### 第二步：对比结果

```bash
# 对比两个运行结果
tpd compare ./results_baseline ./results_problem \
  --rank 0 \
  --tolerance 1e-6 \
  -o comparison.json
```

**对比内容**：

- 基本信息：shape、stride、device、dtype、numel
- 数值统计：max、min、mean、var、nan_count、inf_count
- 采样值：逐个对比采样元素的差异

### 第三步：生成总结报告（重要！）

**对比完成后，先生成一份全面的总结报告，快速了解问题全貌，再决定如何细查。**

```bash
# 生成总结报告
tpd report comparison.json

# 默认输出到 comparison.report.txt
# 也可以指定输出文件
tpd report comparison.json -o my_report.txt

# 调整大差异阈值
tpd report comparison.json --threshold 0.5
```

**报告内容**：

1. **概览**：对比基本信息和统计
2. **问题统计**：NaN/Inf、大差异、Shape 不匹配的数量和严重程度
3. **首次出现位置**：各类问题首次出现的 step、module、tensor（**最重要的信息**）
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
```

**为什么先生成报告？**

- ✅ **快速了解全貌**：不需要逐个查询，一眼看出问题类型和严重程度
- ✅ **定位首次出现**：报告直接给出各类问题的首次出现位置（step 最早）
- ✅ **获取索引号**：报告中每个条目都有索引号，可直接用于 `cmp show` 命令
- ✅ **针对性建议**：根据问题类型自动给出排查建议
- ✅ **节省时间**：避免盲目查询，先看报告再精准定位

### 第四步：查询和定位问题

根据报告中的信息，使用 `tpd cmp` 子命令系统精准查询。

#### 4.1 查看概览

```bash
tpd cmp summary comparison.json
```

**输出解读**：

- `Entries with differences`: 有多少条目存在差异
- `NaN/Inf issues`: NaN/Inf 问题数量（最严重，优先处理）
- `Large diff issues`: 大数值差异数量（阈值默认 1.0）
- `Shape mismatch issues`: Shape 不匹配数量（通常是 bug）

#### 4.2 定位首次出现的问题

**场景 A：出现 NaN/Inf**

```bash
# 方法 1：查看报告中的首次出现位置（推荐）
cat comparison.report.txt  # 查看 "3. First Occurrences" 部分

# 方法 2：使用命令查找
# 找到首次出现 NaN/Inf 的位置，查看前后 3 个条目的上下文
tpd cmp first comparison.json --type naninf --window 3
```

输出会显示：
- 首次出现的 step、module、tensor_path
- 触发条件（如 `inf_count` 从 0 变为 3）
- 前后窗口内的上下文条目

**场景 B：数值偏差过大**

```bash
# 找到首次出现大差异的位置（阈值 0.5）
tpd cmp first comparison.json --type large-diff --threshold 0.5 --window 2
```

**场景 C：Shape 不匹配**

```bash
# 找到 shape 不一致的位置
tpd cmp first comparison.json --type shape
```

#### 4.3 列出特定类型的问题

```bash
# 只列出 NaN/Inf 问题
tpd cmp list comparison.json --type naninf --sort step

# 只列出大差异问题
tpd cmp list comparison.json --type large-diff --threshold 1.0

# 只列出 shape 问题
tpd cmp list comparison.json --type shape
```

输出格式：

```
Idx    Step   Type                   Issues   Module / Tensor
----------------------------------------------------------------------------------------------------
0      1      backward_grad_output   7        __main__.SimpleModel [grad_output[0]]
1      1      forward_input          7        __main__.SimpleModel [args[0]]
...
```

- `Idx`: 条目索引（用于 `show` 命令）
- `Step`: 训练步数
- `Type`: Hook 类型（forward_input/output, backward_grad_input/output）
- `Issues`: 该条目包含的差异数量
- `Module / Tensor`: 模块名和张量路径

#### 4.4 查看条目详情

根据报告中的索引号，直接查看指定条目：

```bash
# 查看索引 5 的条目
tpd cmp show comparison.json 5

# 查看索引 5，前后各 2 个上下文条目
tpd cmp show comparison.json 5 --window 2

# 查看索引 5~10 的范围
tpd cmp show comparison.json 5-10
```

**输出解读**：

```
=================================================================
  Diff Entry [5]
=================================================================
  Step:        2
  Hook Type:   forward_output
  Module:      torch.nn.modules.linear.Linear
  Tensor Path: output
  Stack ID 1:  S000002_ea622568dcc5
  Stack ID 2:  S000002_ea622568dcc5
  Differences: 7
    Diff #0: [numerical_stat] max  (abs_diff: 3.713015e-01)
    Dir1: 1.640414e+00
    Dir2: 1.269112e+00
    ...
```

- `Stack ID`: 堆栈追踪 ID（用于查询完整调用栈）
- `Differences`: 差异详情列表
  - `[basic_info]`: 基本信息差异（shape/dtype 等）
  - `[numerical_stat]`: 数值统计差异（max/min/mean/var/nan_count/inf_count）
  - `[sample_values]`: 采样值差异（显示具体哪些元素不同）

#### 4.5 统计问题分布

```bash
# 总体统计
tpd cmp count comparison.json

# 按类型统计
tpd cmp count comparison.json --type naninf
```

输出：

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

**分析技巧**：

- 如果 `backward_grad_output` 问题多 → 可能是 loss 计算或梯度缩放问题
- 如果 `forward_output` 问题多 → 可能是某层的前向计算有问题
- 如果集中在某个模块 → 检查该模块的实现

### 第五步：查询堆栈追踪

找到问题条目后，通过 Stack ID 查询完整调用栈：

```bash
# 查询堆栈
tpd stack ./results_problem S000002_ea622568dcc5 --rank 0
```

输出：

```
Stack trace for ID: S000002_ea622568dcc5
================================================================================
  File "train.py", line 42, in <module>
    loss = model(inputs)
  File "model.py", line 156, in forward
    x = self.linear(x)
  File "torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
================================================================================
```

**定位到具体代码行**，检查该处的：
- 数据类型转换（FP32↔FP16）
- 梯度裁剪/缩放逻辑
- 初始化参数
- 自定义算子实现

## 常见问题排查策略

### 策略 1：NaN/Inf 问题

```bash
# 1. 定位首次出现位置
tpd cmp first comparison.json --type naninf --window 3

# 2. 查看该条目详情
tpd cmp show comparison.json <idx>

# 3. 查询堆栈
tpd stack ./results_problem <stack_id>

# 4. 检查代码中的：
#    - 除零操作
#    - log(0)
#    - 梯度爆炸（检查梯度裁剪）
#    - FP16 溢出（检查 loss scaling）
```

### 策略 2：数值偏差累积

```bash
# 1. 统计大差异分布
tpd cmp count comparison.json --type large-diff --threshold 0.1

# 2. 按 step 列出，观察从哪一步开始偏差变大
tpd cmp list comparison.json --type large-diff --threshold 0.1 --sort step

# 3. 查看首次大差异的上下文
tpd cmp first comparison.json --type large-diff --threshold 0.1 --window 5

# 4. 检查：
#    - 随机种子是否一致
#    - 数据加载顺序
#    - 初始化权重
#    - 优化器状态
```

### 策略 3：Shape 不匹配

```bash
# 1. 列出所有 shape 问题
tpd cmp list comparison.json --type shape

# 2. 查看详情
tpd cmp show comparison.json <idx>

# 3. 检查：
#    - 模型架构是否一致
#    - batch size 配置
#    - 序列长度 padding
#    - 分布式切分逻辑
```

### 策略 4：多卡训练问题

```bash
# 对比不同 rank 的结果
tpd compare ./results/rank0 ./results/rank1 -o cmp_rank01.json
tpd cmp summary cmp_rank01.json

# 如果 rank 间差异大，检查：
# - 数据分片是否正确
# - 梯度同步（allreduce）
# - 随机种子（每个 rank 应不同）
# - BN/LayerNorm 的统计量同步
```

## 高级技巧

### 1. 缩小监控范围

如果文件太大，可以只监控关键模块：

```bash
# 只监控 Linear 和 LayerNorm
TPD_ENABLED=1 \
TPD_MODULE_FILTER=Linear,LayerNorm \
python train.py
```

### 2. 调整采样策略

- **均匀采样**（推荐）：`TPD_SAMPLE_MODE=uniform`，覆盖整个张量
- **随机采样**：`TPD_SAMPLE_MODE=random TPD_SAMPLE_SEED=42`，适合超大张量

### 3. 限制数据量

```bash
# 只捕获前 500 次 hook 调用
TPD_MAX_STEPS=500 TPD_SAVE_INTERVAL=50 python train.py
```

### 4. 对比不同 rank

```bash
# 检查 rank 0 和 rank 1 是否一致
tpd compare ./results_rank0 ./results_rank1 -o cmp_ranks.json
tpd cmp summary cmp_ranks.json
```

### 5. 多次对比定位

```bash
# 对比不同超参数的影响
tpd compare ./results_lr1e-3 ./results_lr1e-4 -o cmp_lr.json

# 对比不同精度
tpd compare ./results_fp32 ./results_fp16 -o cmp_precision.json

# 对比不同架构
tpd compare ./results_v1 ./results_v2 -o cmp_arch.json
```

## 输出文件说明

### 目录结构

```
results/
├── rank0.jsonl          # Rank 0 的张量快照（JSONL 格式）
├── rank1.jsonl          # Rank 1 的张量快照
├── stacks_rank0.json    # Rank 0 的堆栈追踪
├── stacks_rank1.json    # Rank 1 的堆栈追踪
└── config_rank0.json    # Rank 0 的配置信息
```

### JSONL 记录格式

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

## 注意事项

1. **性能影响**：TPD 会增加 10-30% 的训练开销，建议只在调试时启用
2. **磁盘空间**：JSONL 文件可能较大（几百 MB 到几 GB），注意磁盘空间
3. **采样代表性**：采样数量过少可能遗漏问题，建议至少 20-50 个元素
4. **容差选择**：`--tolerance` 应根据精度类型选择（FP32: 1e-6, FP16: 1e-3, BF16: 1e-2）
5. **随机性**：确保两次运行的随机种子、数据顺序一致，否则差异可能来自随机性而非 bug

## 完整示例

```bash
# 1. 安装 TPD
cd torch-precision-debugger
pip install -e .

# 2. 运行基准版本
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./baseline \
TPD_SAMPLE_COUNT=50 \
TPD_MAX_STEPS=500 \
python train.py --seed 42

# 3. 运行问题版本
TPD_ENABLED=1 \
TPD_OUTPUT_DIR=./problem \
TPD_SAMPLE_COUNT=50 \
TPD_MAX_STEPS=500 \
python train.py --seed 42 --use-fp16

# 4. 对比结果
tpd compare ./baseline ./problem -o comparison.json --tolerance 1e-3

# 5. 生成总结报告（重要！先看全貌）
tpd report comparison.json

# 6. 查看报告
cat comparison.report.txt
# 重点关注：
#   - "2. Issue Statistics" - 了解问题类型和数量
#   - "3. First Occurrences" - 找到首次出现问题的位置
#   - "4. Critical Issues" - 查看 NaN/Inf 详情
#   - "5. Top 10 Largest Differences" - 查看最大差异

# 7. 根据报告中的索引，细查问题条目
tpd cmp show comparison.json 87 --window 3  # 查看报告中提到的索引 87

# 8. 或使用 first 命令定位
tpd cmp first comparison.json --type naninf --window 3

# 9. 查询堆栈追踪
tpd stack ./problem S000012_88808c57116f

# 10. 根据堆栈定位到代码，修复问题！
```

## 故障排除

### Q: TPD 没有生效？

检查：
- `TPD_ENABLED=1` 是否设置
- `pip install -e .` 是否成功
- Python 环境是否正确

### Q: 对比结果全是差异？

可能原因：
- 随机种子不一致
- 数据加载顺序不同
- 模型初始化不同
- 容差设置过小

### Q: 文件太大？

解决方案：
- 减少 `TPD_SAMPLE_COUNT`
- 减少 `TPD_MAX_STEPS`
- 使用 `TPD_MODULE_FILTER` 过滤模块

### Q: 找不到 NaN/Inf？

可能 NaN/Inf 出现在未采样的元素中，尝试：
- 增加 `TPD_SAMPLE_COUNT`
- 检查 `nan_count` / `inf_count` 字段（即使未采样也会统计）

## 相关资源

- [README.md](./README.md) - 完整文档
- [examples/](./examples/) - 示例脚本
- [Issues](https://github.com/your-repo/torch-precision-debugger/issues) - 问题反馈
