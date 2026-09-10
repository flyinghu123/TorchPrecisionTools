"""分布式 rank 识别与运行环境快照。

优先级：``PPROBE_RANK`` 显式覆盖 > ``torch.distributed`` 已初始化进程组 >
torchrun 注入的 ``RANK``/``LOCAL_RANK`` 环境变量 > 0。
"""

from __future__ import annotations

import os
import platform
import socket
import sys
from typing import Any


def _int_env(*names: str) -> int | None:
    for n in names:
        raw = os.environ.get(n)
        if raw is None:
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return None


def get_rank() -> int:
    """当前进程的全局 rank。"""
    r = _int_env("PPROBE_RANK")
    if r is not None:
        return r
    try:  # 延迟导入：非 torch 进程里不付出任何代价
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    r = _int_env("RANK")
    if r is not None:
        return r
    return 0


def get_world_size() -> int:
    r = _int_env("PPROBE_WORLD_SIZE")
    if r is not None:
        return r
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    r = _int_env("WORLD_SIZE")
    return r if r and r > 0 else 1


def get_local_rank() -> int | None:
    r = _int_env("LOCAL_RANK")
    if r is not None:
        return r
    return _int_env("PPROBE_LOCAL_RANK")


def rank_dir_name(rank: int) -> str:
    return f"rank{rank}"


def rank_dirs(out_dir: str) -> list[int]:
    """扫描结果目录，返回所有 rank 编号（对比命令按 rank 对齐用）。"""
    ranks: list[int] = []
    if not os.path.isdir(out_dir):
        return ranks
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isdir(p) and name.startswith("rank") and name[4:].isdigit():
            ranks.append(int(name[4:]))
        elif os.path.isdir(p) and name == "no-rank":
            ranks.append(0)
    return sorted(set(ranks)) or ([0] if os.path.isdir(out_dir) else [])


def resolve_rank_dir(out_dir: str, rank: int) -> str:
    """兼容单进程直接写在 out_dir 根下的情况。"""
    d = os.path.join(out_dir, rank_dir_name(rank))
    if os.path.isdir(d):
        return d
    if os.path.isfile(os.path.join(out_dir, "events.jsonl")):
        return out_dir
    return d


def device_description() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = []
            for i in range(min(n, 8)):
                try:
                    props = torch.cuda.get_device_properties(i)
                    names.append(f"cuda:{i}={props.name}/sm{props.major}{props.minor}/{props.total_memory // (1024**2)}MiB")
                except Exception:
                    names.append(f"cuda:{i}=?")
            return f"cuda({n} gpu): " + "; ".join(names)
        if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "unknown"


def backend_flags() -> dict[str, Any]:
    """采集影响数值结果的开关——跨平台/跨运行精度差异最常见的成因。"""
    flags: dict[str, Any] = {}
    try:
        import torch

        m = torch.backends.cuda.matmul
        c = torch.backends.cudnn
        try:
            flags["cuda.matmul.allow_tf32"] = bool(m.allow_tf32)
        except Exception:
            # torch 2.x 新式 fp32_precision 字符串
            try:
                flags["cuda.matmul.fp32_precision"] = str(m.fp32_precision)
            except Exception:
                pass
        try:
            flags["cudnn.allow_tf32"] = bool(c.allow_tf32)
        except Exception:
            pass
        try:
            flags["cudnn.benchmark"] = bool(c.benchmark)
            flags["cudnn.deterministic"] = bool(c.deterministic)
        except Exception:
            pass
        try:
            flags["fp32_precision"] = str(torch.backends.fp32_precision)
        except Exception:
            pass
        for key in ("flash_sdp", "mem_efficient_sdp", "cudnn_attention", "math_sdp"):
            fn = getattr(torch.backends.cuda, key, None)
            if fn is not None:
                try:
                    flags[f"sdp.{key}"] = bool(fn())
                except Exception:
                    pass
        try:
            flags["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:
            pass
        try:
            flags["float32_matmul_precision"] = str(torch.get_float32_matmul_precision())
        except Exception:
            pass
    except Exception as e:
        flags["error"] = repr(e)
    return {k: v for k, v in flags.items() if v is not None}


def env_info(rank: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """运行环境快照，写入 ``rankN/env.json``，供 compare 命令做环境差异归因。"""
    info: dict[str, Any] = {
        "rank": rank,
        "world_size": get_world_size(),
        "local_rank": get_local_rank(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": (platform.processor() or platform.uname().processor or "")[:120],
        "libc": " ".join(platform.libc_ver()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mpi_backend": os.environ.get("MPI_BACKEND", ""),
        "torchrun": bool(os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("LOCAL_RANK")),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch.cuda_version"] = str(torch.version.cuda)
        info["torch.cudnn_version"] = str(torch.backends.cudnn.version())
        info["torch.git_version"] = str(getattr(torch.version, "git_version", ""))
        try:
            info["nccl_version"] = str(torch.cuda.nccl.version())
        except Exception:
            pass
        info["device"] = device_description()
        info["backends"] = backend_flags()
    except Exception as e:
        info["torch"] = f"unavailable: {e!r}"
    if extra:
        info.update(extra)
    return info
