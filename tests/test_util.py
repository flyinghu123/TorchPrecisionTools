"""util 层：JSON 特殊值编码、环境变量解析、原子写、正则匹配器。"""

from __future__ import annotations

import math

import pytest

from pprobe.util import (RegexMatcher, atomic_write_json, clamp, decode_number, dumps,
                         encode_number, human_bytes, is_special, parse_float, parse_flag,
                         parse_int, parse_list, read_json, short_repr)


def test_nan_inf_roundtrip():
    assert encode_number(float("nan")) == "NaN"
    assert encode_number(float("inf")) == "Inf"
    assert encode_number(float("-inf")) == "-Inf"
    assert math.isnan(decode_number("NaN"))
    assert decode_number("Inf") == math.inf
    assert decode_number("-Inf") == -math.inf
    assert decode_number(None) is math.nan or math.isnan(decode_number(None))
    assert is_special("NaN") and not is_special(1.0)


def test_dumps_is_strict_json():
    """nan/inf 编码后必须是合法 JSON（否则结果目录读不回来）。"""
    import json

    text = dumps({"a": encode_number(float("nan")), "b": [encode_number(1.5), encode_number(2)]})
    assert "NaN" in text
    assert json.loads(text)["b"] == [1.5, 2]


def test_encode_number_keeps_int_and_bool():
    assert encode_number(7) == 7
    assert encode_number(True) is True
    assert encode_number("abc") == "abc"


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False),
     ("off", False), ("", False), (None, False), ("garbage", True)],
)
def test_parse_flag(raw, expected):
    assert parse_flag(raw, True if raw == "garbage" else False) is expected


@pytest.mark.parametrize(
    "raw,expected",
    [("50", 50), ("1k", 1024), ("2M", 2 * 1024**2), ("1_000", 1000), ("", 7),
     (None, 7), ("abc", 7)],
)
def test_parse_int(raw, expected):
    assert parse_int(raw, 7) == expected


def test_parse_float_and_list():
    assert parse_float("1e-3", 0.0) == pytest.approx(1e-3)
    assert parse_float("bad", 2.5) == 2.5
    assert parse_list("a,b;c d") == ["a", "b", "c", "d"]
    assert parse_list(None, ["x"]) == ["x"]


def test_atomic_write_json(tmp_path):
    p = tmp_path / "sub" / "a.json"
    atomic_write_json(str(p), {"z": 1, "a": [1, 2]})
    assert read_json(str(p)) == {"a": [1, 2], "z": 1}
    # 覆盖写不留临时文件
    atomic_write_json(str(p), {"z": 2})
    assert not [f for f in p.parent.iterdir() if f.name.startswith(".tmp_")]


def test_human_bytes():
    assert human_bytes(512) == "512B"
    assert human_bytes(2048).startswith("2.0KiB")
    assert human_bytes(3 * 1024**3) == "3.0GiB"


def test_regex_matcher():
    m = RegexMatcher([r"blocks\.\d+\.norm", "bad(("], "t")
    assert m.active
    assert m.match("net.blocks.3.norm")
    assert m.first_match_index("net.blocks.3.norm") == 0
    assert m.first_match_index("nothing") == -1
    assert not RegexMatcher(None).active
    assert not RegexMatcher(None).match("anything")


def test_clamp_and_short_repr():
    assert clamp(5, 0, 3) == 3
    assert clamp(-5, 0, 3) == 0
    assert clamp(1.5, 0, 3) == 1.5
    long = short_repr({"x": "y" * 300}, 40)
    assert len(long) == 40 and long.endswith("...")
    assert "\n" not in short_repr("a\nb")


class _BadRepr:
    def __repr__(self):
        raise RuntimeError("boom")


def test_short_repr_survives_broken_repr():
    assert "repr failed" in short_repr(_BadRepr())
