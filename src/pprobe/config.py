"""环境变量配置。

所有可调项都集中在 :class:`Config`，通过 ``PPROBE_*`` 环境变量控制；
单元测试/嵌入使用时可以直接构造 ``Config(...)`` 绕过环境变量。
"""

from __future__ import annotations

from typing import Any

from .util import log, parse_flag, parse_float, parse_int, parse_list, parse_str

ENV_PREFIX = "PPROBE_"

#: 默认被 hook 的 torch.nn.functional 算子（精度问题高发点，且没有对应 Module）
DEFAULT_FUNC_TARGETS = [
    "linear",
    "softmax",
    "log_softmax",
    "layer_norm",
    "group_norm",
    "dropout",
    "scaled_dot_product_attention",
    "cross_entropy",
    "nll_loss",
    "silu",
    "gelu",
    "relu",
]

VALID_SAMPLE_MODES = ("uniform", "random", "head", "off")
VALID_HOOKS = ("forward", "backward", "func", "optim")
VALID_BWD_MODES = ("tensor", "module", "both")

#: 环境变量唯一权威表：(名字, 默认值, 类型, 说明)。``pprobe env --markdown`` 直接由此生成文档。
ENV_DOCS: list[tuple[str, str, str, str]] = [
    ("ENABLE", "0", "bool", "总开关；不设则 .pth 注入完全不生效（连 pprobe 都不会 import）"),
    ("OUT", "./pprobe_out", "path", "结果根目录，下面按 rankN/ 分目录"),
    ("RUN_NAME", "", "str", "附加到 manifest 的标签，方便区分两次运行"),
    ("VERBOSE", "1", "bool", "启动/结束时向 stderr 打印探针状态"),
    ("PER_RANK_DIR", "1", "bool", "是否按 rank 分目录（多卡必须为 1）"),
    ("RANK", "自动检测", "int", "强制指定 rank（优先于 torch.distributed / 环境变量）"),
    ("WORLD_SIZE", "自动检测", "int", "强制指定 world_size（优先于 torch.distributed / WORLD_SIZE）"),
    ("LOCAL_RANK", "自动检测", "int", "指定 local rank（只在 ``LOCAL_RANK`` 没设时生效，写进 env.json）"),
    ("ONLY_RANKS", "全部", "list[int]", "只在指定 rank 上采集，如 ``0`` 或 ``0,3``"),
    ("RECORD_ENV", "1", "bool", "写 rankN/env.json（torch/cuda/后端开关等环境快照）"),
    ("SAMPLE_MODE", "uniform", "uniform|random|head|off", "每个 tensor 的采样方式"),
    ("SAMPLE_N", "50", "int", "每个 tensor 采样元素个数"),
    ("SEED", "不设=随机", "int", "随机采样的种子；不设则用熵源种子并记入 manifest"),
    ("SAMPLE_LAYOUT", "flat", "flat|grid", "均匀采样的布局：扁平等间隔或逐维网格"),
    ("SAMPLE_EXTRA", "", "list", "附加采样信息，如 ``bits``（原始位模式，跨平台逐 bit 对比）"),
    ("SUMMARY", "1", "bool", "记录 tensor 基本信息（shape/stride/device/dtype 等）"),
    ("STATS", "1", "bool", "记录数值统计（max/min/mean/var/nan/inf 个数等）"),
    ("STATS_DTYPE", "float64", "float64|float32", "统计累加精度；GPU 上想省时间可设 float32"),
    ("CHECKSUM", "1", "bool", "记录 float64 求和 checksum（同一下标序列可跨运行对齐）"),
    ("FULL_HASH", "0", "bool", "对完整 tensor 取位模式 blake2b（很慢，定位“是否完全一致”时用）"),
    ("RECORD_SCALARS", "1", "bool", "记录输入里的 int/float/str/None 标量（eps、dtype 等）"),
    ("RECORD_PARAMS", "0", "bool", "首次调用时记录模块参数（权重加载差异）"),
    ("HOOK", "forward,backward,optim", "list", "开启哪几类 hook；``func`` 额外 hook nn.functional"),
    ("BWD_MODE", "module", "module|tensor|both", "反向采集方式；module=模块边界梯度，tensor=张量 register_hook"),
    ("INCLUDE", "", "list[regex]", "只记录模块路径匹配任一正则的模块"),
    ("EXCLUDE", "", "list[regex]", "排除模块路径匹配任一正则的模块"),
    ("INCLUDE_CLS", "", "list[regex]", "只记录类名匹配的模块"),
    ("EXCLUDE_CLS", "", "list[regex]", "排除类名匹配的模块（如 Dropout）"),
    ("FUNC_TARGETS", "内置 12 个", "list", "PPROBE_HOOK 含 func 时要包住的 nn.functional 函数名"),
    ("TRAVERSE_DEPTH", "3", "int", "展开嵌套 list/tuple/dict 输入输出的最大层数"),
    ("MAX_TENSORS_PER_CALL", "64", "int", "单次 forward 最多记录多少个 tensor 槽位（0=不限）"),
    ("MAX_EVENTS", "0", "int", "最多记录多少次 forward/backward，达到即落盘收尾"),
    ("MAX_STEPS", "0", "int", "最多记录多少个 optimizer step"),
    ("MAX_CALLS_PER_MODULE", "0", "int", "单个模块最多记录多少次调用（控制热点模块体量）"),
    ("FLUSH_INTERVAL", "50", "int", "每多少条事件 flush + fsync 一次（防丢数据的关键参数）"),
    ("FLUSH_SECS", "0", "float", "额外按时间定期 flush（秒），0=关闭"),
    ("SIGNALS", "1", "bool", "捕获 SIGINT/SIGTERM/SIGHUP/SIGQUIT 先落盘再交回原 handler"),
    ("STOP_ON_NAN", "0", "bool", "发现 NaN 立刻抛异常中断训练（配合 pdb 定位）"),
    ("FAULTHANDLER", "0", "bool", "启用 faulthandler，段错误时输出 C 层堆栈"),
    ("SAVE_FULL", "", "list[regex]", "对匹配模块把完整 tensor 存成 rankN/tensors/*.pt"),
    ("SAVE_FULL_MAX_BYTES", "64M", "bytes", "完整 tensor 落盘的体积上限（超过则只记 summary）"),
    ("SAVE_FULL_EVERY", "1", "int", "每 N 次调用存一份完整 tensor"),
    ("STACKS", "1", "bool", "记录堆栈（以 id 引用，完整堆栈集中在 stacks.json）"),
    ("STACK_TRIM", "1", "bool", "裁掉 torch 内部帧，只保留用户代码调用链"),
    ("STACK_LIMIT", "48", "int", "单条堆栈最多保留多少帧"),
    ("KEEP_WARNINGS", "0", "bool", "保留探针自己触发的 UserWarning（默认静默）"),
]

KNOWN_ENV = {f"PPROBE_{name}" for name, _, _, _ in ENV_DOCS} | {"PPROBE_SEED_USED", "PPROBE_TRACEBACK"}


class Config:
    """运行期配置。字段与 ``PPROBE_*`` 环境变量一一对应（见 README 表格）。"""

    # ---- 激活与输出 ----
    enable: bool = False
    out_dir: str = ""
    run_name: str = ""
    verbose: bool = True
    per_rank_dir: bool = True
    only_ranks: tuple[int, ...] | None = None
    rank_override: int | None = None
    record_env: bool = True

    # ---- 采样 ----
    sample_mode: str = "uniform"
    sample_n: int = 50
    sample_seed: int | None = None
    sample_layout: str = "flat"
    sample_extra: tuple[str, ...] = ()

    # ---- summary ----
    summary: bool = True
    stats: bool = True
    checksum: bool = True
    stats_dtype: str = "float64"
    full_hash: bool = False
    record_scalars: bool = True
    record_params: bool = False

    # ---- hook 范围 ----
    hooks: tuple[str, ...] = ("forward", "backward", "optim")
    bwd_mode: str = "module"
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    include_cls: tuple[str, ...] = ()
    exclude_cls: tuple[str, ...] = ()
    func_targets: tuple[str, ...] = ()
    traverse_depth: int = 3
    max_tensors_per_call: int = 64
    module_name_maxlen: int = 200

    # ---- 数量上限与落盘 ----
    max_events: int = 0
    max_steps: int = 0
    max_calls_per_module: int = 0
    flush_interval: int = 50
    flush_secs: float = 0.0
    signals: bool = True
    stop_on_nan: bool = False
    faulthandler: bool = False

    # ---- 全量张量落盘 ----
    save_full: tuple[str, ...] = ()
    save_full_max_bytes: int = 64 * 1024 * 1024
    save_full_every: int = 1

    # ---- 堆栈 ----
    stacks: bool = True
    stack_trim: bool = True
    stack_limit: int = 48

    def __init__(self, **overrides: Any):
        for k, v in overrides.items():
            if not hasattr(Config, k):
                raise TypeError(f"未知配置项: {k}")
            setattr(self, k, v)

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        """从环境变量（默认 ``os.environ``）构造配置，非法值降级为默认并告警。"""
        import os

        e = os.environ if env is None else env

        def g(name):
            return e.get(ENV_PREFIX + name)

        def flag(name, default):
            return parse_flag(g(name), default)

        def cint(name, default):
            return parse_int(g(name), default)

        def cfloat(name, default):
            return parse_float(g(name), default)

        def cstr(name, default):
            return parse_str(g(name), default)

        def clist(name, default=None):
            return parse_list(g(name), default)

        cfg = cls()
        cfg.enable = flag("ENABLE", False)
        cfg.out_dir = cstr("OUT", "./pprobe_out") or "./pprobe_out"
        cfg.run_name = cstr("RUN_NAME", "") or ""
        cfg.verbose = flag("VERBOSE", True)
        cfg.per_rank_dir = flag("PER_RANK_DIR", True)
        cfg.record_env = flag("RECORD_ENV", True)
        cfg.rank_override = cint("RANK", None)

        only_ranks = clist("ONLY_RANKS", [])
        cfg.only_ranks = tuple(int(x) for x in only_ranks) if only_ranks else None

        mode = (cstr("SAMPLE_MODE", "uniform") or "uniform").lower()
        if mode not in VALID_SAMPLE_MODES:
            log(f"PPROBE_SAMPLE_MODE={mode!r} 非法，回退 uniform（可选 {VALID_SAMPLE_MODES}）")
            mode = "uniform"
        cfg.sample_mode = mode
        cfg.sample_n = cint("SAMPLE_N", 50)
        if cfg.sample_n is None or cfg.sample_n < 0:
            cfg.sample_n = 50
        cfg.sample_seed = cint("SEED", None)
        if cfg.sample_seed is None:
            # 父进程里探针随机选中的种子会被回写到 PPROBE_SEED_USED，
            # 子进程（DataLoader worker / 复现脚本）沿用它才能对上同一批采样下标
            cfg.sample_seed = parse_int(g("SEED_USED"), None)
        layout = (cstr("SAMPLE_LAYOUT", "flat") or "flat").lower()
        if layout not in ("flat", "grid"):
            log(f"PPROBE_SAMPLE_LAYOUT={layout!r} 非法，回退 flat")
            layout = "flat"
        cfg.sample_layout = layout
        cfg.sample_extra = tuple(x.lower() for x in clist("SAMPLE_EXTRA", []))

        cfg.summary = flag("SUMMARY", True)
        cfg.stats = flag("STATS", True)
        cfg.checksum = flag("CHECKSUM", True)
        sd = (cstr("STATS_DTYPE", "float64") or "float64").lower()
        if sd not in ("float64", "float32"):
            log(f"PPROBE_STATS_DTYPE={sd!r} 非法，回退 float64")
            sd = "float64"
        cfg.stats_dtype = sd
        cfg.full_hash = flag("FULL_HASH", False)
        cfg.record_scalars = flag("RECORD_SCALARS", True)
        cfg.record_params = flag("RECORD_PARAMS", False)

        hooks = tuple(x.lower() for x in clist("HOOK", ["forward", "backward", "optim"]))
        bad = [h for h in hooks if h not in VALID_HOOKS]
        if bad:
            log(f"PPROBE_HOOK 含未知项 {bad}，已忽略（可选 {VALID_HOOKS}）")
        hooks = tuple(h for h in hooks if h in VALID_HOOKS) or ("forward",)
        cfg.hooks = hooks
        bwd = (cstr("BWD_MODE", "module") or "module").lower()
        if bwd not in VALID_BWD_MODES:
            log(f"PPROBE_BWD_MODE={bwd!r} 非法，回退 module")
            bwd = "module"
        cfg.bwd_mode = bwd
        cfg.include = tuple(clist("INCLUDE", []))
        cfg.exclude = tuple(clist("EXCLUDE", []))
        cfg.include_cls = tuple(clist("INCLUDE_CLS", []))
        cfg.exclude_cls = tuple(clist("EXCLUDE_CLS", []))
        ft = clist("FUNC_TARGETS", [])
        cfg.func_targets = tuple(ft) if ft else tuple(DEFAULT_FUNC_TARGETS)
        cfg.traverse_depth = cint("TRAVERSE_DEPTH", 3) or 3
        cfg.max_tensors_per_call = cint("MAX_TENSORS_PER_CALL", 64) or 0
        cfg.module_name_maxlen = cint("MODULE_NAME_MAXLEN", 200) or 200

        cfg.max_events = cint("MAX_EVENTS", 0) or 0
        cfg.max_steps = cint("MAX_STEPS", 0) or 0
        cfg.max_calls_per_module = cint("MAX_CALLS_PER_MODULE", 0) or 0
        cfg.flush_interval = cint("FLUSH_INTERVAL", 50)
        if not cfg.flush_interval or cfg.flush_interval < 1:
            cfg.flush_interval = 1
        cfg.flush_secs = cfloat("FLUSH_SECS", 0.0) or 0.0
        cfg.signals = flag("SIGNALS", True)
        cfg.stop_on_nan = flag("STOP_ON_NAN", False)
        cfg.faulthandler = flag("FAULTHANDLER", False)

        cfg.save_full = tuple(clist("SAVE_FULL", []))
        cfg.save_full_max_bytes = cint("SAVE_FULL_MAX_BYTES", 64 * 1024 * 1024) or 0
        cfg.save_full_every = cint("SAVE_FULL_EVERY", 1) or 1

        cfg.stacks = flag("STACKS", True)
        cfg.stack_trim = flag("STACK_TRIM", True)
        cfg.stack_limit = cint("STACK_LIMIT", 48) or 48

        return cfg

    # ------------------------------------------------------------------
    def replace(self, **overrides: Any) -> "Config":
        kw = self.to_dict()
        kw.update(overrides)
        return Config(**kw)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._field_names()}

    def to_json(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.to_dict().items()}

    @staticmethod
    def _field_names() -> list[str]:
        names: list[str] = []
        for klass in reversed(Config.__mro__):
            names.extend(getattr(klass, "__annotations__", {}).keys())
        return [n for n in dict.fromkeys(names) if not n.startswith("_")]

    def env_snapshot(self) -> dict[str, str]:
        """当前进程里所有 PPROBE_* 原始值，写进 manifest 方便复现。"""
        import os

        return {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIX)}

    def has(self, hook: str) -> bool:
        return hook in self.hooks

    def needs_backward(self) -> bool:
        return "backward" in self.hooks and self.bwd_mode != "off"

    def banner(self) -> str:
        parts = [
            f"mode={self.sample_mode}/n={self.sample_n}",
            f"seed={'随机' if self.sample_seed is None else self.sample_seed}",
            f"hooks={'+'.join(self.hooks)}",
            f"max_events={self.max_events or '∞'}",
            f"flush_every={self.flush_interval}",
        ]
        if self.include:
            parts.append(f"include={list(self.include)}")
        if self.exclude:
            parts.append(f"exclude={list(self.exclude)}")
        return ", ".join(parts)


def unknown_env(env: dict[str, str] | None = None) -> list[str]:
    """返回无法识别的 PPROBE_* 变量名（防止拼写错误导致“设了没生效”）。"""
    import os

    e = os.environ if env is None else env
    return sorted(k for k in e if k.startswith(ENV_PREFIX) and k not in KNOWN_ENV)
