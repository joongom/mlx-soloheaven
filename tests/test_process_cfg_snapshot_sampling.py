"""Process-mode (default) cfg-snapshot roundtrip for sampling defaults.

Process mode is the DEFAULT engine_mode. The child engine's load_model mutates
self.cfg.default_temperature/top_p/min_p/top_k from generation_config.json, but
the API resolves per-request None against the PARENT proxy's cfg view, which is
populated from the child's ``ready`` frame snapshot (_cfg_snapshot) via
``_apply_ready``. These tests verify the 4 sampling defaults survive that
roundtrip so genconfig-derived defaults actually reach the API in the default
mode — WITHOUT spawning a child process.
"""

from __future__ import annotations

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.process_client import EngineProcessProxy
from mlx_soloheaven.engine.process_worker import _cfg_snapshot


def test_cfg_snapshot_includes_all_four_sampling_fields():
    cfg = Config(
        model_path="/tmp/x",
        default_temperature=1.0,
        default_top_p=0.95,
        default_min_p=0.05,
        default_top_k=64,
    )
    snap = _cfg_snapshot(cfg)
    assert snap["default_temperature"] == 1.0
    assert snap["default_top_p"] == 0.95
    assert snap["default_min_p"] == 0.05
    assert snap["default_top_k"] == 64


def test_apply_ready_copies_genconfig_derived_values_to_parent():
    """The parent proxy's cfg picks up the child's (genconfig-derived) defaults
    from the ready snapshot, so per-request None resolution sees them."""
    # Parent starts with plain fallback defaults... (real __init__ — cheap,
    # pipes/process live in _spawn — so _state_lock/_ready_event exist for
    # _apply_ready's single-critical-section publication).
    parent_cfg = Config(model_path="/tmp/x")
    proxy = EngineProcessProxy(parent_cfg)

    # ...child reports genconfig-derived defaults via the ready frame.
    child_cfg = Config(
        model_path="/tmp/x",
        default_temperature=1.0,
        default_top_p=0.95,
        default_min_p=0.0,
        default_top_k=64,
    )
    frame = {
        "model_id": "stub",
        "model_family": "gemma4",
        "enable_thinking": True,
        "think_end_token": 7,
        "cfg": _cfg_snapshot(child_cfg),
    }
    proxy._apply_ready(frame)

    assert proxy.cfg.default_temperature == 1.0
    assert proxy.cfg.default_top_p == 0.95
    assert proxy.cfg.default_top_k == 64
    assert proxy._ready_event.is_set()


def test_apply_ready_top_k_zero_not_dropped_as_none():
    """top_k=0 is a real value (disabled), distinct from None — the snapshot
    carries it and _apply_ready must NOT skip it as 'unset'."""
    parent_cfg = Config(model_path="/tmp/x", default_top_k=64)
    proxy = EngineProcessProxy(parent_cfg)

    child_cfg = Config(model_path="/tmp/x", default_top_k=0)
    frame = {"cfg": _cfg_snapshot(child_cfg)}
    proxy._apply_ready(frame)
    # 0 is not None -> copied through; parent now reflects the disabled top_k.
    assert proxy.cfg.default_top_k == 0
