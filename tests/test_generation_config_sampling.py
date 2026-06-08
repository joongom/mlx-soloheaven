"""Engine auto-read of generation_config.json sampling defaults.

Covers the helper ``_load_generation_config_sampling`` (pure file read) and the
load-time apply step ``MLXEngine._apply_generation_config_sampling`` that
realizes the precedence:

    per-request > explicitly-set CLI flag > generation_config.json > fallback

These tests NEVER load a real model — they write a temp dir holding only a
``generation_config.json`` and drive the read/apply logic in isolation.
"""

from __future__ import annotations

import json
import os

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import (
    MLXEngine,
    _load_generation_config_sampling,
)


def _write_genconfig(model_dir: str, payload) -> str:
    path = os.path.join(model_dir, "generation_config.json")
    with open(path, "w") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f)
    return path


def _bare_engine(model_path: str, **cfg_kwargs) -> MLXEngine:
    """A bare MLXEngine wired only enough to run the sampling apply step."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config(model_path=model_path, **cfg_kwargs)
    eng.model_id = "stub"
    return eng


# --- helper: _load_generation_config_sampling -------------------------------

def test_reads_all_four_fields(tmp_path):
    _write_genconfig(
        str(tmp_path),
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.05},
    )
    out = _load_generation_config_sampling(str(tmp_path))
    assert out == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.05}


def test_missing_file_returns_empty(tmp_path):
    # No generation_config.json written at all -> configless path.
    assert _load_generation_config_sampling(str(tmp_path)) == {}


def test_partial_fields(tmp_path):
    _write_genconfig(str(tmp_path), {"temperature": 0.9})
    assert _load_generation_config_sampling(str(tmp_path)) == {"temperature": 0.9}


def test_does_not_read_repetition_penalty(tmp_path):
    # repetition_penalty is intentionally out of scope (gemma4 FIX 2).
    _write_genconfig(
        str(tmp_path),
        {"temperature": 1.0, "repetition_penalty": 1.0, "eos_token_id": 1},
    )
    out = _load_generation_config_sampling(str(tmp_path))
    assert out == {"temperature": 1.0}
    assert "repetition_penalty" not in out


def test_malformed_json_returns_empty(tmp_path):
    _write_genconfig(str(tmp_path), "{ this is : not valid json")
    # Must never raise.
    assert _load_generation_config_sampling(str(tmp_path)) == {}


def test_wrong_types_are_skipped(tmp_path):
    # temperature as string, top_k as float, top_p as bool -> all skipped.
    _write_genconfig(
        str(tmp_path),
        {"temperature": "hot", "top_k": 1.5, "top_p": True, "min_p": 0.1},
    )
    out = _load_generation_config_sampling(str(tmp_path))
    # Only the well-typed min_p survives.
    assert out == {"min_p": 0.1}


# --- apply step: MLXEngine._apply_generation_config_sampling ----------------

def test_precedence_cli_overrides_genconfig(tmp_path):
    """CLI-pinned field keeps its value; an UNPINNED field takes genconfig."""
    _write_genconfig(str(tmp_path), {"temperature": 0.9, "top_p": 0.8})
    eng = _bare_engine(
        str(tmp_path),
        default_temperature=0.3,
        cli_set_sampling=frozenset({"temperature"}),
    )
    applied = eng._apply_generation_config_sampling()
    # CLI wins for temperature; top_p (unpinned) takes the genconfig value.
    assert eng.cfg.default_temperature == 0.3
    assert eng.cfg.default_top_p == 0.8
    assert applied == {"top_p": 0.8}


def test_precedence_genconfig_overrides_fallback(tmp_path):
    """No CLI sampling flags -> genconfig fills, others keep fallback."""
    _write_genconfig(str(tmp_path), {"top_k": 40})
    eng = _bare_engine(str(tmp_path))  # cli_set_sampling defaults to empty
    eng._apply_generation_config_sampling()
    assert eng.cfg.default_top_k == 40          # from generation_config
    assert eng.cfg.default_temperature == 0.6   # fallback untouched
    assert eng.cfg.default_top_p == 1.0
    assert eng.cfg.default_min_p == 0.0


def test_configless_no_regression(tmp_path):
    """No CLI flags + no generation_config -> hardcoded defaults intact."""
    eng = _bare_engine(str(tmp_path))
    eng._apply_generation_config_sampling()
    assert (
        eng.cfg.default_temperature,
        eng.cfg.default_top_p,
        eng.cfg.default_min_p,
        eng.cfg.default_top_k,
    ) == (0.6, 1.0, 0.0, 0)


def test_explicit_top_k_zero_disables_and_is_not_overridden(tmp_path):
    """--top-k 0 (pinned) must stay 0 even if generation_config sets top_k=64."""
    _write_genconfig(str(tmp_path), {"top_k": 64, "temperature": 1.0})
    eng = _bare_engine(
        str(tmp_path),
        default_top_k=0,
        cli_set_sampling=frozenset({"top_k"}),
    )
    eng._apply_generation_config_sampling()
    # top_k pinned to 0 (disabled) — generation_config CANNOT re-enable it.
    assert eng.cfg.default_top_k == 0
    # temperature was not pinned -> takes the generation_config value.
    assert eng.cfg.default_temperature == 1.0


def test_explicit_temperature_zero_greedy_is_not_overridden(tmp_path):
    """--temperature 0.0 (pinned, greedy) must stay 0.0 even if
    generation_config sets a conflicting temperature."""
    _write_genconfig(str(tmp_path), {"temperature": 0.9, "top_k": 64})
    eng = _bare_engine(
        str(tmp_path),
        default_temperature=0.0,
        cli_set_sampling=frozenset({"temperature"}),
    )
    eng._apply_generation_config_sampling()
    # temperature pinned to 0.0 (greedy) — generation_config CANNOT override it.
    assert eng.cfg.default_temperature == 0.0
    # top_k was not pinned -> takes the generation_config value.
    assert eng.cfg.default_top_k == 64


def test_explicit_top_p_one_disabled_is_not_overridden(tmp_path):
    """--top-p 1.0 (pinned, nucleus disabled) must stay 1.0 even if
    generation_config sets a conflicting top_p."""
    _write_genconfig(str(tmp_path), {"top_p": 0.8, "top_k": 64})
    eng = _bare_engine(
        str(tmp_path),
        default_top_p=1.0,
        cli_set_sampling=frozenset({"top_p"}),
    )
    eng._apply_generation_config_sampling()
    # top_p pinned to 1.0 (disabled) — generation_config CANNOT re-enable it.
    assert eng.cfg.default_top_p == 1.0
    # top_k was not pinned -> takes the generation_config value.
    assert eng.cfg.default_top_k == 64


def test_explicit_min_p_zero_disabled_is_not_overridden(tmp_path):
    """--min-p 0.0 (pinned, disabled) must stay 0.0 even if
    generation_config sets a conflicting min_p."""
    _write_genconfig(str(tmp_path), {"min_p": 0.05, "top_k": 64})
    eng = _bare_engine(
        str(tmp_path),
        default_min_p=0.0,
        cli_set_sampling=frozenset({"min_p"}),
    )
    eng._apply_generation_config_sampling()
    # min_p pinned to 0.0 (disabled) — generation_config CANNOT re-enable it.
    assert eng.cfg.default_min_p == 0.0
    # top_k was not pinned -> takes the generation_config value.
    assert eng.cfg.default_top_k == 64


def test_apply_returns_only_applied_keys(tmp_path):
    """Return dict reports exactly the keys written into cfg (diagnostics)."""
    _write_genconfig(
        str(tmp_path),
        {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
    )
    eng = _bare_engine(
        str(tmp_path),
        cli_set_sampling=frozenset({"top_p"}),  # pin top_p
        default_top_p=0.5,
    )
    applied = eng._apply_generation_config_sampling()
    assert applied == {"temperature": 1.0, "top_k": 64}
    assert eng.cfg.default_top_p == 0.5  # pinned, untouched
