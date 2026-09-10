"""pprobe —— PyTorch 训练精度问题定位探针。

两种接入方式：

1. 解释器注入（推荐，零改代码）::

       pprobe install                     # 写入 site-packages/pprobe.pth
       PPROBE_ENABLE=1 python train.py    # megatron / llamafactory / ms-swift 同理
       PPROBE_ENABLE=1 torchrun --nproc_per_node=8 pretrain_gpt.py ...

2. 代码内显式启动::

       import pprobe
       pprobe.init(out_dir="./dbg", sample_n=200, max_events=2000)

本模块刻意不导入 torch，保证解释器启动阶段足够轻。
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "init",
    "start",
    "boot_from_pth",
    "flush",
    "stop",
    "mark_step",
    "disable_module",
    "result_dir",
    "recorder",
    "Config",
]


def init(**overrides: Any):
    """启动探针（等价 ``PPROBE_ENABLE=1``），返回 Recorder 或 None。"""
    from .bootstrap import init as _init

    return _init(**overrides)


def start(**overrides: Any):
    """:func:`init` 的别名。"""
    return init(**overrides)


def boot_from_pth() -> None:
    from .bootstrap import boot_from_pth as _boot

    _boot()


def flush(fsync: bool = True) -> int:
    """手动把缓冲事件落盘（长任务里可以在每个 checkpoint 后调用）。"""
    from .recorder import get_recorder

    rec = get_recorder()
    return rec.flush(fsync=fsync, reason="manual") if rec else 0


def stop(reason: str = "manual-stop") -> None:
    """结束采集并写完整元信息。"""
    from .recorder import get_recorder

    rec = get_recorder()
    if rec:
        rec.finalize(reason)


def mark_step() -> None:
    """自定义训练循环里手动标一个 step（megatron/deepspeed 等非 ``torch.optim`` 子类时用）。"""
    from .recorder import get_recorder

    rec = get_recorder()
    if rec:
        rec.on_optimizer_step()


def disable_module(module) -> None:
    """把某个模块排除在采集之外（例如已经定位完的子模块，降低开销）。"""
    try:
        module.__dict__["_pprobe_off"] = True
    except Exception:
        pass


def result_dir() -> str | None:
    """当前进程的 rank 结果目录；尚未发生任何前向时返回 None（目录是延迟创建的）。"""
    from .recorder import get_recorder

    rec = get_recorder()
    return getattr(rec, "rank_dir", None) if rec else None


def recorder():
    from .recorder import get_recorder

    return get_recorder()


def __getattr__(name: str):  # 惰性导出 Config，避免 import pprobe 触发额外导入
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
