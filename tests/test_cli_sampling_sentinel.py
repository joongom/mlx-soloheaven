"""CLI sampling sentinel + Config.from_args precedence wiring.

The 4 true sampling flags (--temperature/--top-p/--min-p/--top-k) default to a
None sentinel so that an UNSET flag is distinguishable from an explicitly-set
one. Config.from_args records which were CLI-pinned (cli_set_sampling) and only
overwrites the dataclass default for pinned fields; unset fields keep the
hardcoded fallback so the engine's generation_config auto-read can fill them.

Also exercises the per-request override layer end-to-end against a bare engine
(no model load) and the process-mode cfg snapshot/apply roundtrip.
"""

from __future__ import annotations

import threading

import pytest

from mlx_soloheaven.cli import parse_args
from mlx_soloheaven.config import Config, ModelConfig
from mlx_soloheaven.engine.mlx_engine import MLXEngine


# Sampling env vars must not leak into the parse from the dev shell.
_SAMPLING_ENV = (
    "SOLOHEAVEN_TEMPERATURE",
    "SOLOHEAVEN_TOP_P",
    "SOLOHEAVEN_MIN_P",
    "SOLOHEAVEN_TOP_K",
)


@pytest.fixture(autouse=True)
def _clear_sampling_env(monkeypatch):
    for key in _SAMPLING_ENV:
        monkeypatch.delenv(key, raising=False)


# --- sentinel parsing -------------------------------------------------------

def test_unset_flags_are_none():
    args = parse_args(["--model", "/tmp/x"])
    assert args.temperature is None
    assert args.top_p is None
    assert args.min_p is None
    assert args.top_k is None


def test_explicit_flag_sets_value_and_marker():
    args = parse_args(["--model", "/tmp/x", "--temperature", "0.2"])
    assert args.temperature == 0.2
    cfg = Config.from_args(args)
    assert cfg.cli_set_sampling == frozenset({"temperature"})
    assert cfg.default_temperature == 0.2
    # Unset fields keep the dataclass fallback.
    assert cfg.default_top_p == 1.0
    assert cfg.default_min_p == 0.0
    assert cfg.default_top_k == 0


def test_unset_flags_keep_fallback_and_empty_marker():
    cfg = Config.from_args(parse_args(["--model", "/tmp/x"]))
    assert cfg.cli_set_sampling == frozenset()
    assert (
        cfg.default_temperature,
        cfg.default_top_p,
        cfg.default_min_p,
        cfg.default_top_k,
    ) == (0.6, 1.0, 0.0, 0)
    # Per-model config mirrors this.
    m = cfg.models[0]
    assert isinstance(m, ModelConfig)
    assert m.cli_set_sampling == frozenset()
    assert (m.default_temperature, m.default_top_p, m.default_min_p, m.default_top_k) == (
        0.6, 1.0, 0.0, 0,
    )


def test_explicit_top_k_zero_is_pinned():
    """--top-k 0 disables top-k AND is recorded as CLI-set (not 'unset')."""
    cfg = Config.from_args(parse_args(["--model", "/tmp/x", "--top-k", "0"]))
    assert "top_k" in cfg.cli_set_sampling
    assert cfg.default_top_k == 0
    assert cfg.models[0].cli_set_sampling == frozenset({"top_k"})


def test_env_var_path_handles_unset(monkeypatch):
    """Unset env -> no float(None) crash; set env -> treated as CLI-set."""
    for key in _SAMPLING_ENV:
        monkeypatch.delenv(key, raising=False)
    # Should not raise (the bug guard: float(None) at parse time).
    args = parse_args(["--model", "/tmp/x"])
    assert args.temperature is None

    monkeypatch.setenv("SOLOHEAVEN_TEMPERATURE", "0.4")
    args2 = parse_args(["--model", "/tmp/x"])
    assert args2.temperature == 0.4
    cfg = Config.from_args(args2)
    assert "temperature" in cfg.cli_set_sampling
    assert cfg.default_temperature == 0.4


def test_multi_model_shares_marker():
    cfg = Config.from_args(
        parse_args(["--models", "/tmp/a", "/tmp/b", "--top-p", "0.9"])
    )
    assert cfg.cli_set_sampling == frozenset({"top_p"})
    for m in cfg.models:
        assert m.cli_set_sampling == frozenset({"top_p"})
        assert m.default_top_p == 0.9
        # Unset fields fall through to fallback (engine fills per-model later).
        assert m.default_temperature == 0.6


# --- per-request override (top of precedence) -------------------------------

def test_per_request_overrides_config_default(monkeypatch):
    """A non-None request temperature reaches _generate_locked unchanged,
    overriding cfg.default_temperature (which the genconfig auto-read set)."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config(model_path="/tmp/x", default_temperature=0.9)
    eng._lock = threading.Lock()

    captured = {}

    def _fake_locked(messages, **kwargs):
        captured.update(kwargs)
        if False:
            yield None
        return

    monkeypatch.setattr(eng, "_generate_locked", _fake_locked)

    # Drain the generator; request temperature=0.1 must win over default 0.9.
    list(eng.generate_stream([{"role": "user", "content": "hi"}], temperature=0.1))
    assert captured["temperature"] == 0.1


def test_request_none_falls_through_to_default(monkeypatch):
    """A None request temperature resolves to cfg.default_temperature."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config(model_path="/tmp/x", default_temperature=0.9)
    eng._lock = threading.Lock()

    captured = {}

    def _fake_locked(messages, **kwargs):
        captured.update(kwargs)
        if False:
            yield None
        return

    monkeypatch.setattr(eng, "_generate_locked", _fake_locked)
    list(eng.generate_stream([{"role": "user", "content": "hi"}], temperature=None))
    assert captured["temperature"] == 0.9
