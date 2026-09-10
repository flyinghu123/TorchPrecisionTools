"""Config / 环境变量表：非法值降级、未知变量提示（需求 1/3/5 的入口）。"""

from __future__ import annotations

from pprobe.config import ENV_DOCS, KNOWN_ENV, VALID_SAMPLE_MODES, Config, unknown_env


def test_defaults_match_documented_table():
    cfg = Config()
    assert cfg.sample_mode == "uniform"
    assert cfg.sample_n == 50          # 需求 3：默认均匀采样 50 个元素
    assert cfg.sample_seed is None      # 不设 seed 就是随机
    assert cfg.flush_interval == 50
    assert cfg.max_events == 0
    assert cfg.hooks == ("forward", "backward", "optim")
    assert cfg.stacks is True


def test_from_env_reads_all_knobs():
    env = {
        "PPROBE_ENABLE": "1",
        "PPROBE_OUT": "/tmp/x",
        "PPROBE_SAMPLE_MODE": "random",
        "PPROBE_SAMPLE_N": "200",
        "PPROBE_SEED": "1234",
        "PPROBE_MAX_EVENTS": "1k",
        "PPROBE_FLUSH_INTERVAL": "3",
        "PPROBE_HOOK": "forward,backward,func,optim",
        "PPROBE_INCLUDE": "blocks\\..*,embed",
        "PPROBE_ONLY_RANKS": "0,2",
        "PPROBE_RANK": "5",
        "PPROBE_STACKS": "0",
        "PPROBE_SAMPLE_EXTRA": "bits",
    }
    cfg = Config.from_env(env)
    assert cfg.enable and cfg.sample_mode == "random" and cfg.sample_n == 200
    assert cfg.sample_seed == 1234
    assert cfg.max_events == 1024          # k/m/g 后缀
    assert cfg.flush_interval == 3
    assert cfg.has("func") and cfg.needs_backward()
    assert cfg.include == ("blocks\\..*", "embed")
    assert cfg.only_ranks == (0, 2)
    assert cfg.rank_override == 5
    assert cfg.stacks is False
    assert cfg.sample_extra == ("bits",)


def test_invalid_values_fall_back_to_defaults():
    cfg = Config.from_env({
        "PPROBE_SAMPLE_MODE": "bogus",
        "PPROBE_SAMPLE_LAYOUT": "circular",
        "PPROBE_HOOK": "forward,magic",
        "PPROBE_BWD_MODE": "nope",
        "PPROBE_STATS_DTYPE": "float16",
        "PPROBE_FLUSH_INTERVAL": "0",
        "PPROBE_SAMPLE_N": "-3",
    })
    assert cfg.sample_mode in VALID_SAMPLE_MODES
    assert cfg.sample_layout == "flat"
    assert cfg.hooks == ("forward",)
    assert cfg.bwd_mode == "module"
    assert cfg.stats_dtype == "float64"
    assert cfg.flush_interval == 1          # 至少每事件一次，不能退化成 0
    assert cfg.sample_n == 50


def test_forward_only_hook_is_never_empty():
    assert Config.from_env({"PPROBE_HOOK": "nonsense"}).hooks == ("forward",)
    assert not Config.from_env({"PPROBE_HOOK": "forward"}).needs_backward()


def test_unknown_env_detects_typos():
    assert unknown_env({"PPROBE_SAMPLE_N": "10", "PPROBE_SAMPLE_NB": "1"}) == ["PPROBE_SAMPLE_NB"]
    assert unknown_env({"PATH": "/x"}) == []
    # 表里每个变量都必须能被识别，否则文档与实际不一致
    for name, _, _, _ in ENV_DOCS:
        assert f"PPROBE_{name}" in KNOWN_ENV
    assert "PPROBE_MODULE_NAME_MAXLEN" not in KNOWN_ENV  # 只作为内部配置项存在


def test_seed_used_fallback_aligns_child_process():
    """父进程回写的 ``PPROBE_SEED_USED`` 会被子进程沿用（需求 3 的可复现）。"""
    cfg = Config.from_env({"PPROBE_SAMPLE_MODE": "random", "PPROBE_SEED_USED": "777"})
    assert cfg.sample_seed == 777
    # 显式给了 PPROBE_SEED 就以它为准
    assert Config.from_env({"PPROBE_SEED": "1", "PPROBE_SEED_USED": "777"}).sample_seed == 1
    # 两个都没给才是真随机
    assert Config.from_env({}).sample_seed is None
    # 不合法的残留值不能弄坏配置
    assert Config.from_env({"PPROBE_SEED_USED": "abc"}).sample_seed is None


def test_replace_and_serialize_roundtrip():
    cfg = Config.from_env({"PPROBE_SAMPLE_N": "10", "PPROBE_INCLUDE": "a,b"})
    new = cfg.replace(sample_n=99, out_dir="/other")
    assert new.sample_n == 99 and new.include == cfg.include and new.out_dir == "/other"
    dumped = new.to_json()
    assert isinstance(dumped["include"], list)          # tuple -> list，JSON 可写
    assert dumped["sample_n"] == 99
    assert Config(**dumped).sample_n == 99


def test_unknown_override_rejected():
    import pytest

    with pytest.raises(TypeError):
        Config(nope=1)


def test_env_snapshot_only_pprobe(monkeypatch):
    monkeypatch.setenv("PPROBE_SAMPLE_N", "7")
    monkeypatch.setenv("SOMETHING_ELSE", "1")
    snap = Config().env_snapshot()
    assert snap["PPROBE_SAMPLE_N"] == "7"
    assert "SOMETHING_ELSE" not in snap


def test_banner_mentions_key_limits():
    text = Config.from_env({"PPROBE_SAMPLE_MODE": "random", "PPROBE_SAMPLE_N": "9",
                            "PPROBE_SEED": "3", "PPROBE_MAX_EVENTS": "100"}).banner()
    assert "random/n=9" in text and "seed=3" in text and "max_events=100" in text
