"""通用工具：原子写、JSON 特殊浮点值编码、体积格式化、正则匹配器等。

本模块刻意不依赖 torch，保证解释器启动阶段（.pth 注入）可以极低成本导入。
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
from typing import Any, Iterable

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "t"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "f", ""}

# JSON 中无法表达 nan/inf，这里统一编码为字符串，保证严格 JSON 合法，
# 同时便于跨运行逐元素对比。
NAN_TOKEN = "NaN"
POS_INF_TOKEN = "Inf"
NEG_INF_TOKEN = "-Inf"


def log(msg: str) -> None:
    sys.stderr.write(f"[pprobe] {msg}\n")
    sys.stderr.flush()


def parse_flag(raw: str | None, default: bool = False) -> bool:
    """纯字符串解析，供 Config.from_env 复用（不隐式读 os.environ）。"""
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    log(f"环境变量值 {raw!r} 无法识别真假，回退默认值 {default}")
    return default


def env_flag(name: str, default: bool = False) -> bool:
    return parse_flag(os.environ.get(name), default)


def parse_int(raw: str | None, default: int | None) -> int | None:
    if raw is None or raw.strip() == "":
        return default
    try:
        text = raw.strip().lower().replace("_", "")
        mult = 1
        for suffix, factor in (("k", 1024), ("m", 1024**2), ("g", 1024**3)):
            if text.endswith(suffix):
                text = text[:-1]
                mult = factor
                break
        return int(float(text)) * mult
    except ValueError:
        log(f"环境变量值 {raw!r} 不是合法整数，回退默认值 {default}")
        return default


def env_int(name: str, default: int | None) -> int | None:
    return parse_int(os.environ.get(name), default)


def parse_float(raw: str | None, default: float | None) -> float | None:
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log(f"环境变量值 {raw!r} 不是合法浮点数，回退默认值 {default}")
        return default


def env_float(name: str, default: float | None) -> float | None:
    return parse_float(os.environ.get(name), default)


def parse_str(raw: str | None, default: str | None) -> str | None:
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def env_str(name: str, default: str | None) -> str | None:
    return parse_str(os.environ.get(name), default)


def parse_list(raw: str | None, default: Iterable[str] | None = None) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(default or [])
    parts = re.split(r"[,;:\s]+", raw.strip())
    return [p for p in parts if p]


def env_list(name: str, default: Iterable[str] | None = None) -> list[str]:
    return parse_list(os.environ.get(name), default)


def encode_number(x: Any) -> Any:
    """把 float/int 转成严格 JSON 可表达的值；nan/inf 用固定 token 表示。"""
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x):
            return NAN_TOKEN
        if math.isinf(x):
            return POS_INF_TOKEN if x > 0 else NEG_INF_TOKEN
        return x
    return x


def decode_number(x: Any) -> float:
    """encode_number 的逆操作，返回 float（nan/inf token 还原为浮点特殊值）。"""
    if isinstance(x, str):
        if x == NAN_TOKEN:
            return math.nan
        if x == POS_INF_TOKEN:
            return math.inf
        if x == NEG_INF_TOKEN:
            return -math.inf
        try:
            return float(x)
        except ValueError:
            return math.nan
    if x is None:
        return math.nan
    return float(x)


def is_special(x: Any) -> bool:
    return isinstance(x, str) and x in (NAN_TOKEN, POS_INF_TOKEN, NEG_INF_TOKEN)


def dumps(obj: Any) -> str:
    """紧凑 JSON：key 顺序稳定（便于 diff），允许浮点为 nan/inf token。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def jdump(obj: Any, fp) -> None:
    json.dump(obj, fp, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def loads(line: str) -> Any:
    return json.loads(line)


def atomic_write_text(path: str, text: str) -> None:
    """先写临时文件再 rename，避免中断产生半截文件。"""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=".swap")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_text(path, dumps_pretty(obj) + "\n")


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}PiB"


class RegexMatcher:
    """按正则列表做 include/exclude 匹配（对 None 短路，hook 热路径零开销）。"""

    __slots__ = ("patterns", "_compiled", "name")

    def __init__(self, patterns: list[str] | None, name: str = "matcher"):
        self.patterns = list(patterns or [])
        self.name = name
        self._compiled: list[re.Pattern] | None = None

    @property
    def active(self) -> bool:
        return bool(self.patterns)

    def _get(self) -> list[re.Pattern]:
        if self._compiled is None:
            compiled = []
            for p in self.patterns:
                try:
                    compiled.append(re.compile(p))
                except re.error as e:  # 非法正则不影响主流程
                    log(f"{self.name} 中正则 {p!r} 非法: {e}")
            self._compiled = compiled
        return self._compiled

    def match(self, text: str) -> bool:
        if not self.patterns:
            return False
        return any(rx.search(text) for rx in self._get())

    def first_match_index(self, text: str) -> int:
        """返回命中的第一条正则下标，未命中返回 -1（对比时用于分组）。"""
        if not self.patterns:
            return -1
        for i, rx in enumerate(self._get()):
            if rx.search(text):
                return i
        return -1


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def short_repr(obj: Any, limit: int = 160) -> str:
    try:
        r = repr(obj)
    except Exception as e:  # 某些框架对象的 __repr__ 会抛异常
        r = f"<repr failed {type(obj).__name__}: {e}>"
    r = r.replace("\n", "\\n")
    if len(r) > limit:
        r = r[: limit - 3] + "..."
    return r
