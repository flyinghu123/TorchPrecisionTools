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

### 6. 查询堆栈追踪

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
