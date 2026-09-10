# pprobe 参考

`SKILL.md` 的延伸：字段字典、槽位命名、kinds 判定、常见误判、框架接入注意。
命令与环境变量的完整清单以 `pprobe env` / 仓库 `README.md` §4 §6 为准（本文件不重复，避免漂移）。

## 1. 事件字段字典（`rankN/events.jsonl`，一行一条）

| 字段 | 含义 |
| --- | --- |
| `seq` | 本 rank 内的自增序号，**只用于定位，不能跨运行对齐**（两次运行事件总数常不同） |
| `phase` | `forward` / `backward` |
| `rank` / `step` / `pid` |  rank、第几个优化步、进程号（多进程排查用） |
| `module` / `module_cls` | 层级路径（如 `TinyModel.blocks.0.norm`）与类名；函数式算子记作 `functional.<名>` |
| `call_index` | 该模块的第几次调用（从 0 开始）——**与 `phase`+`module` 一起构成对比对齐 key** |
| `stack_id` | 指向 `stacks.json` 的堆栈 id（需求 4 的“用 id 表示堆栈”） |
| `in_module` | 仅函数式算子有：当时所在的 `nn.Module` 路径 |
| `output_sig` / `n_tensors` | 输出结构签名（`Tensor`/`tuple[Tensor,dict]`…）与本次记录的 tensor 个数 |
| `grad_enabled` | 仅前向：当时是否处于 `torch.no_grad()` 之外 |
| `fwd_seq` / `fwd_stack_id` / `bwd_index` / `bwd_src` | 仅反向：能对回哪次前向、第几次 `.backward()`、来源是 `module-hook` 还是 `tensor-hook` |
| `t_abs` / `wall` | 绝对时间戳与该进程启动后的相对秒数（看发散发生的时间顺序用） |
| `tensors` | 槽位名 → 槽位条目（见下） |
| `truncated` | 仅前向、且被 `PPROBE_MAX_TENSORS_PER_CALL` 截断时为 `true` |

单个槽位里：`basic` 是 shape/stride/dtype/device/numel/nbytes/contiguous/storage_offset/version/
requires_grad/is_leaf/is_sparse/is_meta/dtype_limits；`stats` 是 count/max/min/mean/var/std/sum/
checksum/absmax/l2norm/zero_count/nan_count/inf_count/posinf_count/neginf_count；`sample` 是 n/idx/idx_nd/vals（`bits` 需 `PPROBE_SAMPLE_EXTRA=bits`）。

张量槽位是 `{kind: "tensor", basic, stats, sample, hash, full_tensor}`（`hash` 需
`PPROBE_FULL_HASH=1`，`full_tensor` 需 `PPROBE_SAVE_FULL` 命中）；标量槽位是
`{kind: "scalar", value}`；没法直接取值的容器记成 `{kind, sig}`（结构签名）或
`{kind, cls, repr}`。对比时槽位缺失或 `kind` 不同（一边 tensor、一边 scalar）都会判成 `presence`。

NaN/Inf 在 JSON 里编码成 `"NaN"` / `"Inf"` / `"-Inf"` 字符串，所以文件永远是合法 JSON。

## 2. 槽位命名

| 槽位 | 来源 |
| --- | --- |
| `input[i]`、`kwargs.<名>` | 模块 forward 的入参 |
| `output`、`output[i]`、`output.<键>` | 输出是 tensor / tuple / dict 时的对应位置 |
| `param.<名>` | 模块参数（需 `PPROBE_RECORD_PARAMS=1`，首次调用时记一次） |
| `<名>`（非 tensor） | `PPROBE_RECORD_SCALARS=1` 时记 int/float/str/None 标量，如 `kwargs.eps` |
| `grad_input[i]`、`grad_output[i]` | 反向 · 模块边界梯度（`PPROBE_BWD_MODE=module`） |
| `grad_of:<前向槽位>` | 反向 · 张量级 `register_hook`（`bwd_mode=tensor`/`both`，或模块级注册失败自动回退） |

`param.<名>` 只收模块**直接**参数、最多 8 个（`named_parameters(recurse=False)`），要看更深的参数
就挑子模块名当 `PPROBE_INCLUDE`。

被 `PPROBE_MAX_TENSORS_PER_CALL`（默认 64）截断时事件会带 `"truncated": true`——**两边的槽位数量
不同会被判成 `presence`/`info` 差异**，排查前先确认没被截断。

## 3. 差异 kinds 与优先级

`compare` 的对齐 key 是 `(rank, phase, 模块路径, call_index)`，配不上的进
`events_only_a` / `events_only_b`。单条事件内可能出现的 kinds：

* `info` —— 基本信息不同（dtype/shape/device/stride/模块类名）。**优先级最高**：通常是配置、
  权重加载或设备放置问题，不是“算错了”。
* `special` —— NaN/Inf 相关变化（从 `stats` 的 nan/inf 计数或 `sample` 的值里提升出来），
  永远和 `stats`/`sample` 一起出现。
* `stats` —— max/min/mean/var/sum/checksum 不同（按 `--atol/--rtol` 判）。
* `sample` —— 采样元素逐个不同（要求两边采样配置一致，见 SKILL.md §2）。
* `scalar` —— 标量入参不同（eps、reduction、training 标志这类）。
* `index_mismatch` —— 两边采样下标不一致，`sample` 比较无效。
* `stack_diff` —— 同一次模块调用来自不同调用链（代码分支/框架包装差异的信号）。
* `presence` —— 只在一边存在。

**排序默认 `--sort seq`（最早发散）**，这是最有用的视角；想看“谁差得最多”用 `--sort max_abs`，
但量级大的层往往只是被上游带偏，别把最大差当根因。缩小范围用 `--include/--exclude/--phase/--ignore-slot`。

## 4. 常见误判

| 误判 | 实际原因 | 处理 |
| --- | --- | --- |
| “到处都有差异，找不到根因” | 两边探针配置不同（`SAMPLE_N`/`SEED`/`STATS_DTYPE`） | 看报告开头的探针配置差异小节；固定同一套 `PPROBE_*` 重采 |
| bf16/fp16 下几乎每层都报 `stats` 差异 | 默认 `--rtol 1e-5` 比该 dtype 的表示误差还小 | `--rtol 1e-2`，或只对 `--sort seq` 的第一条深挖 |
| 第一个差异就是 loss | 只采了 loss 所在的小子图，上游没被记录 | 去掉 `PPROBE_INCLUDE` 或改成更宽的模块正则 |
| 没有 backward 事件 | 训练里没调 `.backward()`（如在 `no_grad` 下评估），或 `PPROBE_HOOK` 去掉了 `backward` | 确认 hook 配置；纯前向场景就只比前向 |
| 某些模块的梯度槽位变成 `grad_of:*` | 无参数模块 / 多输入模块的 `register_full_backward_hook` 注册失败，自动回退张量级 hook | 正常现象；要统一形态就显式设 `PPROBE_BWD_MODE=tensor` |
| 结果目录里多了个空 rank0 | `torchrun` 的 launcher 进程也 import 了 torch | 探针延迟建目录，只有真记录事件才建；确认 launcher 没参与 rendezvous 训练 |
| 采到的数据比预期少很多 | 达到 `PPROBE_MAX_EVENTS`/`MAX_CALLS_PER_MODULE` 上限 | 看 `manifest.limit_reason` 与 `counters` |
| GPU 上慢得明显 | 每个 tensor 若干次全 tensor reduction + `.item()` 同步 | 降 `SAMPLE_N`、`STATS_DTYPE=float32`、`PPROBE_INCLUDE` 缩范围、`MAX_CALLS_PER_MODULE` |

## 5. 框架接入注意

* **torchrun / accelerate / deepspeed**：环境变量随 fork/exec 继承到每个 worker；rank 判定顺序是
  `PPROBE_RANK` → 已初始化的 `dist.get_rank()` → `RANK` → 0。
* **megatron-lm**：`PPROBE_INCLUDE='.*\.(linear_attn|qkv|core_attn|mlp)\..*'` 缩小范围；TP/PP/DP 混排时
  全局 rank 唯一，但两次实验的 rank 编号可能不同，用 `compare --rank A:B` 配对。
* **LLaMA-Factory / ms-swift**：`PPROBE_ENABLE=1 PPROBE_OUT=./dbg llamafactory-cli train xxx.yaml`；
  只看基座不要 LoRA 分支：`PPROBE_EXCLUDE_CLS='.*lora.*'`。
* **HuggingFace Trainer**：它可能不走标准 `torch.optim` 子类（如 fp16 场景用 apex/deepspeed 的 optimizer），
  `step` 计数会不动——事件本身照样记录，需要正确的 `step` 标签时每步调一次 `pprobe.mark_step()`。
* **DataLoader worker（fork）**：`os.register_at_fork` 会让子进程不重复写数据。
* **`torch.compile` / jit trace**：dynamo 追溯阶段的调用被跳过（`torch.compiler.is_compiling()`），
  不会把编译期符号当成真实数值；但 compile 后的真实前向在图里，hook 只能看到模块边界。
* **多机**：把各机 `PPROBE_OUT` 指到同一共享目录，或各自取回后按 rank 手工配对。
* **不走 Module 的函数式网络**：`PPROBE_HOOK=forward,func` 兜底，或在关键位置显式
  `pprobe.recorder().record(...)`。
