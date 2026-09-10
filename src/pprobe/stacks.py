"""调用堆栈采集与去重存储。

堆栈在事件里只用一个短 id 表示（``s12``），完整帧信息集中放在 ``stacks.json``，
避免同一条堆栈在成千上万条事件里重复存储。
"""

from __future__ import annotations

import hashlib
import linecache
import os
import sys
from typing import Any

from .util import atomic_write_json, read_json

#: 这些文件的帧属于框架/自身实现，默认从堆栈里裁掉（PPROBE_STACK_TRIM=0 可保留）
_FRAMEWORK_HINTS = (
    os.path.join("torch", "nn", "modules", "module.py"),
    os.path.join("torch", "nn", "_functions.py"),
    os.path.join("torch", "_ops.py"),
    os.path.join("torch", "functional.py"),
    os.path.join("torch", "autograd", "graph_fn.py"),
    os.path.join("torch", "autograd", "function.py"),
    os.path.join("torch", "_compile.py"),
    os.path.join("torch", "_dynamo", ""),
    os.path.join("torch", "fx", "_symbolic_trace.py"),
    os.path.join("importlib", ""),
    os.path.join("pprobe", "hooks.py"),
    os.path.join("pprobe", "recorder.py"),
    os.path.join("pprobe", "stacks.py"),
)


def _is_pprobe_frame(filename: str) -> bool:
    return os.sep + "pprobe" + os.sep in filename or filename.endswith(
        ("pprobe\\hooks.py", "pprobe\\recorder.py")
    )


class StackStore:
    def __init__(self, trim: bool = True, limit: int = 48, path: str | None = None):
        self.trim = trim
        self.limit = max(1, limit)
        self.path = path
        self._ids: dict[str, str] = {}  # 堆栈内容哈希 -> 短 id
        self._entries: dict[str, dict[str, Any]] = {}  # 短 id -> 详情
        self._counts: dict[str, int] = {}  # 短 id -> 命中次数
        self._dirty = False
        self._counts_dirty = False
        self._saves = 0
        if path and os.path.isfile(path):  # 断点续跑（同一目录二次写入）时恢复
            try:
                data = read_json(path)
                for sid, entry in (data.get("stacks") or {}).items():
                    self._entries[sid] = entry
                    sig = entry.get("signature") or ""
                    self._ids[hashlib.blake2b(sig.encode(), digest_size=16).hexdigest()] = sid
                    self._counts[sid] = entry.get("count", 0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def capture(self, skip: int = 0) -> str | None:
        """采集当前堆栈并返回 id；空堆栈返回 ``None``。"""
        frames = self._collect(skip + 1)
        if not frames:
            return None
        return self.intern(frames)

    def _collect(self, skip: int) -> list[dict[str, Any]]:
        try:
            frame = sys._getframe(skip)
        except ValueError:
            return []
        raw: list[tuple[str, int, str]] = []
        guard = 0
        while frame is not None and guard < self.limit * 6 + 64:
            code = frame.f_code
            filename = code.co_filename
            if not _is_pprobe_frame(filename):
                if not (self.trim and self._is_framework(filename)):
                    raw.append((filename, frame.f_lineno, code.co_name))
            frame = frame.f_back
            guard += 1
        raw.reverse()  # 外层 -> 内层
        raw = raw[-self.limit :]
        frames = []
        for filename, lineno, name in raw:
            frames.append(
                {
                    "file": filename,
                    "line": lineno,
                    "name": name,
                    "text": f"{_shorten(filename)}:{lineno} in {name}",
                }
            )
        return frames

    @staticmethod
    def _is_framework(filename: str) -> bool:
        return any(h in filename for h in _FRAMEWORK_HINTS)

    # ------------------------------------------------------------------
    def intern(self, frames: list[dict[str, Any]]) -> str:
        signature = "\n".join(f"{f['file']}:{f['line']}:{f['name']}" for f in frames)
        digest = hashlib.blake2b(signature.encode("utf-8", "replace"), digest_size=16).hexdigest()
        sid = self._ids.get(digest)
        if sid is None:
            sid = f"s{len(self._entries) + 1}"
            self._ids[digest] = sid
            # 只有新堆栈才读源码行（linecache 有开销，命中缓存后极快）
            entry = {
                "id": sid,
                "frames": [
                    {
                        **f,
                        "source": _source_line(f["file"], f["line"]),
                    }
                    for f in frames
                ],
                "signature": signature,
                "count": 0,
                "top": frames[0]["text"] if frames else "",
                "caller": frames[-1]["text"] if frames else "",
            }
            self._entries[sid] = entry
            self._dirty = True
        self._counts[sid] = self._counts.get(sid, 0) + 1
        self._entries[sid]["count"] = self._counts[sid]
        self._counts_dirty = True
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        return self._entries.get(sid)

    @property
    def unique(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self._entries),
            "total_hits": sum(self._counts.values()),
            "stacks": self._entries,
        }

    def save(self, path: str | None = None, force: bool = False) -> bool:
        """堆栈表落盘。

        新堆栈必须立刻写（否则事件里的 id 会找不到），但仅命中计数变化很廉价，
        节流到每 10 次 save 重写一次，避免长训练里反复写同一个大文件。
        """
        path = path or self.path
        if not path:
            return False
        self._saves += 1
        counts_due = self._counts_dirty and self._saves % 10 == 0
        if not (self._dirty or counts_due or force):
            return False
        atomic_write_json(path, self.to_dict())
        self._dirty = False
        self._counts_dirty = False
        return True


def _source_line(filename: str, lineno: int) -> str:
    try:
        text = linecache.getline(filename, lineno, globals().get("_pprobe_loader_cache"))
    except Exception:
        return ""
    return text.strip()[:240]


def _shorten(filename: str) -> str:
    """把 site-packages / dist-packages 前缀折叠掉，堆栈更易读也更省空间。"""
    for marker in (os.sep + "site-packages" + os.sep, os.sep + "dist-packages" + os.sep):
        i = filename.rfind(marker)
        if i >= 0:
            return "..." + filename[i : i + len(marker)] + filename[i + len(marker) :]
    return filename
