"""事件记录器：缓冲、定期落盘、上限控制、退出/信号兜底、按 rank 分目录。

落盘策略以「任何时刻被 kill 都能拿到已发生的数据」为目标：
* ``events.jsonl`` 追加写 + 每 ``PPROBE_FLUSH_INTERVAL`` 条 flush/fsync；
* ``stacks.json`` / ``manifest.json`` 每次 flush 原子重写；
* ``atexit`` + SIGINT/SIGTERM/SIGHUP 兜底落盘，处理完立即把信号交回原 handler。
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import sys
import threading
import time
from typing import Any, Callable

from . import distributed
from .config import Config
from .stacks import StackStore
from .util import (RegexMatcher, atomic_write_json, human_bytes, log, dumps)

TOOL_VERSION = "0.1.0"
EVENTS_FILE = "events.jsonl"
STACKS_FILE = "stacks.json"
MANIFEST_FILE = "manifest.json"
ENV_FILE = "env.json"
EXIT_FILE = "exit.json"


class PProbeLimitReached(Exception):
    """达到 PPROBE_MAX_EVENTS/STOP_ON_NAN 上限时抛出，便于直接中断训练定位问题。"""


class Recorder:
    def __init__(self, cfg: Config, rank: int | None = None, lazy: bool = True):
        """构造一个（可选延迟开的）记录器。

        真正的 rank 解析 / 建目录 / 写元信息 / 装信号全部放到 :meth:`_ensure_open`，
        因为 ``.pth`` 让每个 Python 进程都会 import 到 torch（包括 torchrun 的 agent
        进程、只是 ``import torch`` 的工具脚本），它们不应在结果目录里留下空
        rankN/，更不应覆盖真实 rank 进度的 manifest。
        """
        self.cfg = cfg
        self._rank_hint = rank
        self._owner_pid = os.getpid()
        self._opened = False
        self._finalized = False
        self._child = False
        self._disabled = False
        self._tls = threading.local()          # 每线程重入保护（见 record）
        self._hooks_installed = False
        self._errors: dict[str, int] = {}
        self._prev_handlers: dict[int, Any] = {}
        self._limit_reason = ""
        self._sampler: Any = None
        # 几个计数器必须在未打开时也存在（hook 包裹的入口会先于 _ensure_open 访问它们）
        self._seq = 0
        self._step = 0
        self.bwd_index = 0
        self._n_events = 0
        self.max_module_seq: dict[str, int] = {}
        # 种子在纯 Python 层先定下来：TensorSampler 需要 torch，延后到装 hook 时再创建
        if cfg.sample_seed is None:
            self.sample_seed_used = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)
            self.seed_from_user = False
        else:
            self.sample_seed_used = int(cfg.sample_seed) & ((1 << 62) - 1)
            self.seed_from_user = True
        if not lazy:
            self._ensure_open()

    # ------------------------------------------------------------------
    def _ensure_open(self) -> bool:
        """首次真正需要记录时才开目录（由 :attr:`active` / :meth:`flush` 触发）。"""
        if self._opened:
            return True
        self._opened = True
        cfg = self.cfg
        try:
            if cfg.rank_override is not None:
                self.rank = int(cfg.rank_override)
            elif self._rank_hint is not None:
                self.rank = int(self._rank_hint)
            else:
                self.rank = distributed.get_rank()
        except Exception as e:
            self.rank = 0
            self._errors[f"rank:{type(e).__name__}"] = 1
        self.world_size = distributed.get_world_size()
        self.root = os.path.abspath(os.path.expanduser(cfg.out_dir or "./pprobe_out"))
        self.rank_dir = os.path.join(self.root, distributed.rank_dir_name(self.rank)) if cfg.per_rank_dir else self.root
        try:
            os.makedirs(os.path.join(self.rank_dir, "tensors"), exist_ok=True)
        except OSError as e:
            log(f"无法创建输出目录 {self.rank_dir}: {e}")
            self._disabled = True
            self._errors["mkdir"] = self._errors.get("mkdir", 0) + 1
            return False

        self.stacks = StackStore(
            trim=cfg.stack_trim, limit=cfg.stack_limit, path=os.path.join(self.rank_dir, STACKS_FILE)
        )
        self.include = RegexMatcher(list(cfg.include), "PPROBE_INCLUDE")
        self.exclude = RegexMatcher(list(cfg.exclude), "PPROBE_EXCLUDE")
        self.include_cls = RegexMatcher(list(cfg.include_cls), "PPROBE_INCLUDE_CLS")
        self.exclude_cls = RegexMatcher(list(cfg.exclude_cls), "PPROBE_EXCLUDE_CLS")
        self.save_full_matcher = RegexMatcher(list(cfg.save_full), "PPROBE_SAVE_FULL")

        self._lock = threading.RLock()
        self._buffer: list[str] = []
        self._events_path = os.path.join(self.rank_dir, EVENTS_FILE)
        self._fh = None
        self._n_skipped = 0
        self._n_flushes = 0
        self._bytes_written = 0
        self._call_counts: dict[str, int] = {}
        self._slot_counts: dict[str, int] = {}
        self._last_flush = time.monotonic()
        self._t0 = time.time()
        self._monotonic0 = time.monotonic()

        self._write_manifest("running")
        self._write_env()
        if cfg.rank_override is None and cfg.only_ranks is not None and self.rank not in cfg.only_ranks:
            self._disabled = True
            log(f"rank={self.rank} 不在 PPROBE_ONLY_RANKS={list(cfg.only_ranks)} 中，本进程不记录")
        if cfg.signals:
            self._install_signals()
        atexit.register(self._atexit_hook)
        try:
            os.register_at_fork(after_in_child=self._forked_child)
        except (AttributeError, ValueError):
            pass
        if cfg.verbose:
            log(f"开始记录 rank={self.rank} → {self.rank_dir}（world_size={self.world_size}）")
        return True

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    @property
    def sampler(self):
        """TensorSampler 需要 torch，因此解释器启动阶段不创建（见 bootstrap）。"""
        if self._sampler is None:
            from .sampling import TensorSampler

            self._sampler = TensorSampler(
                self.cfg.sample_mode, self.cfg.sample_n, self.sample_seed_used, self.cfg.sample_layout
            )
        return self._sampler

    @property
    def active(self) -> bool:
        """只有真的有模块被调用时才会开目录（避免 launcher / 工具进程留下空 rank 目录）。"""
        if not self._opened:
            self._ensure_open()
        return not self._disabled and not self._finalized and os.getpid() == self._owner_pid

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def step(self) -> int:
        return self._step

    def new_seq(self) -> int:
        self._seq += 1
        return self._seq

    def call_index(self, module: str) -> int:
        """该模块（或算子）第几次被调用，跨运行稳定的对齐 key。"""
        with self._lock:
            self._slot_counts[module] = self._slot_counts.get(module, 0) + 1
            return self._slot_counts[module] - 1

    def note_error(self, key: str) -> None:
        self._errors[key] = self._errors.get(key, 0) + 1

    def _forked_child(self) -> None:
        """DataLoader worker 等 fork 出来的子进程不写数据，避免污染/重复落盘。"""
        self._disabled = True
        self._child = True
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._fh = None

    # ------------------------------------------------------------------
    # 过滤
    # ------------------------------------------------------------------
    def want_module(self, module: str, cls_name: str) -> bool:
        from .events import apply_include_exclude

        return apply_include_exclude(module, cls_name, self.include, self.exclude,
                                      self.include_cls, self.exclude_cls)

    def budget_left(self, module: str) -> bool:
        cap = self.cfg.max_calls_per_module
        if cap <= 0:
            return True
        return self._call_counts.get(module, 0) < cap

    # ------------------------------------------------------------------
    # 记录主入口
    # ------------------------------------------------------------------
    def record(self, phase: str, module: str, cls_name: str, slots: dict[str, Any],
               stack_id: str | None = None, fwd_seq: int | None = None, call_index: int | None = None,
               extra: dict[str, Any] | None = None) -> int | None:
        """记录一条 forward/backward 事件（任何内部异常都会被吞掉，不影响训练）。"""
        if not self.active or getattr(self._tls, "recording", False):
            return None
        # 重入保护必须是线程局部的：autograd 引擎会在自己的工作线程里回调 backward hook，
        # 用实例级标志会让并发线程互相“挡路”，整条事件被静默丢弃。
        self._tls.recording = True
        try:
            with self._lock:  # seq / 计数器 / 缓冲区必须整体加锁，否则序号会重复
                return self._record_locked(phase, module, cls_name, slots, stack_id, fwd_seq,
                                           call_index, extra)
        except Exception as e:  # 记录器本身出错绝不能带崩训练
            self.note_error(f"record:{type(e).__name__}")
            log(f"记录事件失败（已忽略）: {e!r}")
            return None
        finally:
            self._tls.recording = False

    def _record_locked(self, phase: str, module: str, cls_name: str, slots: dict[str, Any],
                       stack_id: str | None, fwd_seq: int | None, call_index: int | None,
                       extra: dict[str, Any] | None) -> int | None:
        seq = self.new_seq()
        event: dict[str, Any] = {
            "seq": seq,
            "phase": phase,
            "rank": self.rank,
            "pid": os.getpid(),
            "step": self._step,
            "module": module[-self.cfg.module_name_maxlen :] if module else module,
            "module_cls": cls_name,
            "call_index": call_index if call_index is not None else self.call_index(module),
            "wall": round(time.time() - self._t0, 6),
            "t_abs": round(self._t0, 3),
            "stack_id": stack_id,
            "tensors": slots,
            "n_tensors": sum(1 for v in slots.values() if isinstance(v, dict) and v.get("kind") == "tensor"),
        }
        if fwd_seq is not None:
            event["fwd_seq"] = fwd_seq
        if extra:
            event.update(extra)
        self._buffer.append(dumps(event))
        self._n_events += 1
        self._call_counts[module] = self._call_counts.get(module, 0) + 1
        self.max_module_seq[module] = seq
        due_by_count = len(self._buffer) >= self.cfg.flush_interval
        due_by_time = self.cfg.flush_secs > 0 and (time.monotonic() - self._last_flush) >= self.cfg.flush_secs
        if due_by_count or due_by_time:
            self.flush(fsync=due_by_count)
        if self.cfg.max_events and self._n_events >= self.cfg.max_events:
            # 最后一条也要保存，不能浪费配额
            self.finalize(f"max_events={self.cfg.max_events} 已达成")
        return seq

    def record_nan(self, module: str, phase: str) -> None:
        if self.cfg.stop_on_nan:
            self.flush(fsync=True)
            raise PProbeLimitReached(f"{phase} 阶段 {module} 出现 NaN（PPROBE_STOP_ON_NAN=1）")

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    def flush(self, fsync: bool = True, reason: str = "") -> int:
        if not self._opened:
            return 0
        with self._lock:
            if not self._buffer and not fsync:
                return 0
            n = len(self._buffer)
            try:
                if self._buffer:
                    fh = self._ensure_file()
                    text = "".join(line + "\n" for line in self._buffer)
                    fh.write(text)
                    self._bytes_written += len(text.encode("utf-8", "replace"))
                    self._buffer.clear()
                fh_exists = self._fh is not None
                if fh_exists:
                    self._fh.flush()
                    if fsync:
                        os.fsync(self._fh.fileno())
                self.stacks.save()
                self._write_manifest(reason or "running")
                self._n_flushes += 1
                self._last_flush = time.monotonic()
            except Exception as e:
                self.note_error(f"flush:{type(e).__name__}")
                log(f"落盘失败: {e!r}")
            return n

    def _ensure_file(self):
        if self._fh is None:
            self._fh = open(self._events_path, "a", encoding="utf-8", buffering=1)
        return self._fh

    def finalize(self, reason: str = "normal-exit") -> None:
        # 从没打开过（没发生过任何前向）→ 什么都不写，避免污染别人的结果目录
        if not self._opened:
            return
        # 只允许拥有本实例的主进程 finalize；fork 出的子进程不能重写父目录元信息
        if self._finalized or self._child or os.getpid() != self._owner_pid:
            return
        self._finalized = True
        with self._lock:
            self._limit_reason = reason
            self.flush(fsync=True, reason="finalize")
            try:
                if self._fh is not None:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                    self._fh.close()
            except Exception:
                pass
            self._fh = None
            self.stacks.save(force=True)  # 把节流期内的命中计数补写完整
            self._write_manifest("finalized")
            self._write_env()
            self._write_exit(reason)
            if self.rank == 0:
                self._write_run_json()
                self._write_hints()
            if self.cfg.verbose:
                log(f"已保存 {self._n_events} 条事件到 {self.rank_dir}（原因: {reason}）")

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    def counters(self) -> dict[str, Any]:
        return {
            "events": self._n_events,
            "skipped": self._n_skipped,
            "flushes": self._n_flushes,
            "unique_stacks": self.stacks.unique,
            "unique_modules": len(self._call_counts),
            "bytes": self._bytes_written,
            "steps": self._step,
            "backward_calls": self.bwd_index,
            "buffered": len(self._buffer),
            "sample_failures": dict(self._sampler.failures) if self._sampler is not None else {},
            "errors": dict(self._errors),
        }

    def _write_manifest(self, state: str) -> None:
        data = {
            "tool": "pprobe",
            "version": TOOL_VERSION,
            "state": state,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": distributed.get_local_rank(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "start_time": self._t0,
            "now": time.time(),
            "elapsed": round(time.monotonic() - self._monotonic0, 3),
            "limit_reason": self._limit_reason,
            "config": self.cfg.to_json(),
            "env": self.cfg.env_snapshot(),
            "counters": self.counters(),
            "sampling": {
                "mode": self.cfg.sample_mode,
                "n": self.cfg.sample_n,
                "seed_user": self.cfg.sample_seed,
                "seed_used": self.sample_seed_used,
                "seed_from_user": self.seed_from_user,
                "layout": self.cfg.sample_layout,
            },
            "files": {"events": EVENTS_FILE, "stacks": STACKS_FILE},
        }
        atomic_write_json(os.path.join(self.rank_dir, MANIFEST_FILE), data)

    def _write_env(self) -> None:
        if not self.cfg.record_env:
            return
        info = distributed.env_info(self.rank)
        info["cmdline"] = " ".join(sys.argv)[:2000]
        atomic_write_json(os.path.join(self.rank_dir, ENV_FILE), info)

    def _write_exit(self, reason: str) -> None:
        atomic_write_json(
            os.path.join(self.rank_dir, EXIT_FILE),
            {"reason": reason, "counters": self.counters(), "now": time.time(),
             "elapsed": round(time.monotonic() - self._monotonic0, 3)},
        )

    def _write_run_json(self) -> None:
        ranks = []
        try:
            ranks = [r for r in distributed.rank_dirs(self.root)]
        except Exception:
            pass
        atomic_write_json(
            os.path.join(self.root, "run.json"),
            {
                "tool": "pprobe",
                "version": TOOL_VERSION,
                "out_dir": self.root,
                "world_size": self.world_size,
                "ranks": ranks,
                "config": self.cfg.to_json(),
                "env": self.cfg.env_snapshot(),
                "finalized_by_rank0": True,
                "stop_reason": self._limit_reason,
            },
        )

    def _write_hints(self) -> None:
        text = f"""pprobe 结果目录: {self.root}
rank 目录: {', '.join('rank%d' % r for r in distributed.rank_dirs(self.root))}
事件条数(rank0): {self._n_events}，唯一堆栈: {self.stacks.unique}

常用命令：
  # 查看某个堆栈 id 的完整调用栈
  pprobe stack s1 --result {self.root}
  # 按 rank 对比两次运行，输出精度差异报告
  pprobe compare {self.root} <另一个结果目录> --out {self.root}/diff.md
  # 单目录速览：NaN/Inf 热点、异常模块
  pprobe report {self.root}
  # 查询事件（按模块名正则过滤）
  pprobe query {self.root} --module 'encoder.*linear' --phase forward

采样配置：mode={self.cfg.sample_mode} n={self.cfg.sample_n} seed={self.sample_seed_used}
（seed 由 PPROBE_SEED {'指定' if self.seed_from_user else '自动生成，下次运行不可复现，建议设置 PPROBE_SEED'}）
"""
        try:
            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, "HINTS.txt"), "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 全量张量落盘（可选）
    # ------------------------------------------------------------------
    def make_full_saver(self, module: str, phase: str, call_index: int) -> Callable | None:
        if not self.save_full_matcher.active or not self.save_full_matcher.match(module):
            return None

        def save(t) -> str | None:
            try:
                import torch

                if call_index % max(1, self.cfg.save_full_every):
                    return None
                if t.numel() * t.element_size() > self.cfg.save_full_max_bytes:
                    return {"skipped": f"too-large:{human_bytes(t.numel() * t.element_size())}"}
                fname = f"{phase}_{call_index}_{_safe_name(module)}.pt"
                path = os.path.join(self.rank_dir, "tensors", fname)
                if not os.path.exists(path):
                    tmp = path + ".tmp"
                    torch.save(t.detach().to("cpu").clone(), tmp)
                    os.replace(tmp, path)
                return {"path": os.path.relpath(path, self.root), "phase": phase}
            except Exception as e:
                self.note_error(f"save_full:{type(e).__name__}")
                return None

        return save

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def on_optimizer_step(self) -> None:
        if not self._opened:
            return
        self._step += 1
        if self.cfg.max_steps and self._step >= self.cfg.max_steps:
            self.finalize(f"max_steps={self.cfg.max_steps} 已达成")

    def _install_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None),
                    getattr(signal, "SIGQUIT", None)):
            if sig is None:
                continue
            try:
                prev = signal.getsignal(sig)
                self._prev_handlers[sig] = prev
                signal.signal(sig, self._make_handler(sig))
            except (ValueError, OSError, RuntimeError):
                pass  # 非主线程无法装信号处理器

    def _make_handler(self, sig: int):
        def handler(signum, frame):
            try:
                if os.getpid() == self._owner_pid and not self._finalized:
                    self.finalize(f"signal:{signal.Signals(signum).name}")
            except BaseException as e:  # 兜底阶段任何异常都不能吞掉原信号语义
                log(f"信号 {signum} 落盘异常: {e!r}")
            self._chain(signum, frame)

        return handler

    def _chain(self, sig: int, frame) -> None:
        """落盘完成后把信号交回原来的 handler，不改变框架原有的退出行为。"""
        prev = self._prev_handlers.get(sig, signal.SIG_DFL)
        if prev is signal.SIG_IGN:
            return
        if prev is signal.SIG_DFL:
            if sig == signal.SIGINT:
                raise KeyboardInterrupt
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)
            return
        if callable(prev):
            prev(sig, frame)
            return
        raise SystemExit(128 + sig)

    def _atexit_hook(self) -> None:
        if os.getpid() != self._owner_pid:
            return
        self.finalize("atexit")


def _safe_name(module: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in module)[-120:] or "root"


# ----------------------------------------------------------------------
# 进程级单例
# ----------------------------------------------------------------------
_RECORDER: Recorder | None = None


def get_recorder() -> Recorder | None:
    return _RECORDER


def init_recorder(cfg: Config) -> Recorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = Recorder(cfg)
    return _RECORDER


def reset_recorder() -> None:
    """测试用：释放单例。"""
    global _RECORDER
    if _RECORDER is not None:
        try:
            _RECORDER.finalize("reset")
        except Exception:
            pass
    _RECORDER = None
