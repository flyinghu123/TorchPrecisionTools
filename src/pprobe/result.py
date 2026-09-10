"""结果目录的读取层：rank 发现、事件索引（按字节偏移流式读取）、堆栈表。

被 ``pprobe compare`` / ``pprobe stack`` / ``pprobe report`` 共用。
"""

from __future__ import annotations

import glob
import os
from typing import Any, Iterator

from .util import dumps, loads, read_json

EVENTS_FILE = "events.jsonl"
STACKS_FILE = "stacks.json"
MANIFEST_FILE = "manifest.json"
ENV_FILE = "env.json"
EXIT_FILE = "exit.json"


class CorruptLineError(Exception):
    pass


def find_rank_dirs(root: str) -> dict[int, str]:
    """``{rank: 目录}``；兼容单 rank 直接写在 root 下的情况。"""
    out: dict[int, str] = {}
    for path in sorted(glob.glob(os.path.join(root, "rank*"))):
        base = os.path.basename(path)
        num = base[4:]
        if os.path.isdir(path) and num.isdigit():
            out[int(num)] = path
    if not out and os.path.isfile(os.path.join(root, EVENTS_FILE)):
        out[0] = root
    if not out:
        # 只有 rank0 目录被中断等情况：退化为扫描任意含 events.jsonl 的子目录
        for path in sorted(glob.glob(os.path.join(root, "*", EVENTS_FILE))):
            d = os.path.dirname(path)
            name = os.path.basename(d)
            if name.startswith("rank") and name[4:].isdigit():
                out[int(name[4:])] = d
    return out


class RankResult:
    def __init__(self, rank: int, path: str):
        self.rank = rank
        self.path = path
        self.events_path = os.path.join(path, EVENTS_FILE)
        self._stacks: dict[str, Any] | None = None
        self._manifest: dict[str, Any] | None = None
        self._env: dict[str, Any] | None = None

    @property
    def has_events(self) -> bool:
        return os.path.isfile(self.events_path)

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = _safe_json(os.path.join(self.path, MANIFEST_FILE))
        return self._manifest

    @property
    def env(self) -> dict[str, Any]:
        if self._env is None:
            self._env = _safe_json(os.path.join(self.path, ENV_FILE))
        return self._env

    @property
    def sampling(self) -> dict[str, Any]:
        return (self.manifest or {}).get("sampling", {}) or {}

    @property
    def config(self) -> dict[str, Any]:
        return (self.manifest or {}).get("config", {}) or {}

    # -- 堆栈 ------------------------------------------------------------
    @property
    def stacks(self) -> dict[str, Any]:
        if self._stacks is None:
            data = _safe_json(os.path.join(self.path, STACKS_FILE))
            self._stacks = data.get("stacks", data) if isinstance(data, dict) else {}
        return self._stacks

    def stack(self, sid: str | None) -> dict[str, Any] | None:
        if not sid:
            return None
        return self.stacks.get(sid)

    # -- 事件 ------------------------------------------------------------
    def iter_events(self) -> Iterator[dict[str, Any]]:
        """逐行读取；容忍被 kill 造成的最后一行残缺。"""
        if not self.has_events:
            return
        with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield loads(line)
                except Exception:
                    continue

    def index(self) -> "EventIndex":
        return EventIndex.build(self)


class EventIndex:
    """``(phase, module, call_index) -> (offset, size)`` 的轻量索引，避免整文件载入内存。"""

    def __init__(self, rank: "RankResult", mapping: dict[tuple, tuple[int, int]], order: list[tuple]):
        self.rank = rank
        self.mapping = mapping
        self.order = order

    @staticmethod
    def key(event: dict[str, Any]) -> tuple:
        return (event.get("phase"), event.get("module"), event.get("call_index"))

    @classmethod
    def build(cls, rank: RankResult, dedup: str = "first") -> "EventIndex":
        mapping: dict[tuple, tuple[int, int]] = {}
        order: list[tuple] = []
        if not rank.has_events:
            return cls(rank, mapping, order)
        with open(rank.events_path, "rb") as f:
            offset = f.tell()
            raw = f.readline()
            while raw:
                stripped = raw.strip()
                if stripped:
                    try:
                        ev = loads(stripped.decode("utf-8", "replace"))
                        k = cls.key(ev)
                        if k not in mapping:
                            mapping[k] = (offset, len(raw))
                            order.append(k)
                        elif dedup == "last":
                            mapping[k] = (offset, len(raw))
                    except Exception:
                        pass
                offset = f.tell()
                raw = f.readline()
        return cls(rank, mapping, order)

    def __len__(self) -> int:
        return len(self.mapping)

    def get(self, key: tuple) -> dict[str, Any] | None:
        loc = self.mapping.get(key)
        if loc is None:
            return None
        offset, size = loc
        with open(self.rank.events_path, "rb") as f:
            f.seek(offset)
            raw = f.read(size).strip()
        try:
            return loads(raw.decode("utf-8", "replace"))
        except Exception:
            return None

    def keys(self):
        return self.mapping.keys()


class Result:
    """一次运行的完整结果目录。"""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"结果目录不存在: {path}")
        self.ranks: dict[int, RankResult] = {
            r: RankResult(r, p) for r, p in find_rank_dirs(self.path).items()
        }
        self.run = _safe_json(os.path.join(self.path, "run.json"))

    @classmethod
    def load(cls, path: str) -> "Result":
        return cls(path)

    def rank_ids(self) -> list[int]:
        return sorted(self.ranks)

    def total_events(self) -> int:
        return sum(len(self.ranks[r].index()) for r in self.ranks)

    def describe(self) -> str:
        bits = [f"{self.path}: ranks={self.rank_ids()}"]
        for r in self.ranks.values():
            m = r.manifest or {}
            c = m.get("counters", {}) or {}
            bits.append(
                f"  rank{r.rank}: events={c.get('events', '?')} stacks={c.get('unique_stacks', '?')} "
                f"state={m.get('state', '?')} reason={m.get('limit_reason', '')}"
            )
        return "\n".join(bits)


def _safe_json(path: str) -> dict[str, Any]:
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def tensor_slots(event: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    tensors = event.get("tensors") or {}
    for name, entry in tensors.items():
        if isinstance(entry, dict) and entry.get("kind") == "tensor":
            yield name, entry


def dump_json(obj: Any, path: str) -> None:
    from .util import atomic_write_json

    atomic_write_json(path, obj)


def line(obj: Any) -> str:
    return dumps(obj)
