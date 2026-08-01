"""Batch D — cross-model KV cache safety (spec item 9).

cache_dir is shared across models and disk files are keyed only by session_id,
so a KV built by model A and a same-architecture model B (different weights)
could otherwise silently reuse each other's cache. Model identity is now folded
into the prompt-contract fingerprint AND stamped in disk metadata; the disk
loader refuses a mismatched-identity cache (in addition to the existing
layer/type checks).

Covers:
  * _prompt_fingerprint carries model identity (different id -> different hash;
    same id -> same; empty id -> historical hash unchanged; existing
    tools/thinking/template_rev gates still work);
  * the disk loader REFUSES a cache stamped by a different model (honest MISS),
    HITs the same model, round-trips the identity in metadata, and lets a legacy
    (unstamped) cache through for one cold rebuild;
  * the preflight / in-memory fingerprint gate reports a MISS across models.
"""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, load_prompt_cache

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import session_cache_mixin as scm_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState
from mlx_vlm.generate import PromptCacheState


# ==========================================================================
# 1. Fingerprint carries model identity.
# ==========================================================================

def test_fingerprint_differs_by_model_identity():
    fp_a = MLXEngine._prompt_fingerprint(None, True, model_identity="modelA")
    fp_b = MLXEngine._prompt_fingerprint(None, True, model_identity="modelB")
    assert fp_a != fp_b
    # Same identity -> same hash (stable).
    assert MLXEngine._prompt_fingerprint(None, True, model_identity="modelA") == fp_a


def test_fingerprint_empty_identity_is_historical():
    # An absent identity must NOT change the historical fingerprint (so existing
    # caches / tests are untouched and a legacy cache takes one cold rebuild).
    base = MLXEngine._prompt_fingerprint(None, True)
    assert MLXEngine._prompt_fingerprint(None, True, model_identity="") == base
    # And a stamped identity DOES change it (a legacy vs current mismatch).
    assert MLXEngine._prompt_fingerprint(None, True, model_identity="m") != base


def test_existing_fingerprint_gates_still_work_with_identity():
    ident = "modelZ"
    # tools change -> different
    a = MLXEngine._prompt_fingerprint([{"x": 1}], True, model_identity=ident)
    b = MLXEngine._prompt_fingerprint([{"x": 2}], True, model_identity=ident)
    assert a != b
    # thinking change -> different
    c = MLXEngine._prompt_fingerprint(None, False, model_identity=ident)
    d = MLXEngine._prompt_fingerprint(None, True, model_identity=ident)
    assert c != d
    # template_rev change -> different
    e = MLXEngine._prompt_fingerprint(None, True, template_rev="v1", model_identity=ident)
    f = MLXEngine._prompt_fingerprint(None, True, template_rev="v2", model_identity=ident)
    assert e != f


# ==========================================================================
# 2. Disk loader refuses a cache from a different model.
# ==========================================================================

def _make_kvcache(seq_len: int = 4, num_heads: int = 2, head_dim: int = 8) -> KVCache:
    c = KVCache()
    k = mx.arange(num_heads * seq_len * head_dim, dtype=mx.float32).reshape(
        1, num_heads, seq_len, head_dim
    )
    v = k * 0.5
    c.update_and_fetch(k, v)
    return c


def _make_engine(tmp_path, identity: str) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.data_dir = str(tmp_path)  # cache_dir = data_dir/cache — SHARED across models
    cfg.disk_budget_gb = 1.0
    eng.cfg = cfg
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._disk_session_ids = set()
    eng._language_model = SimpleNamespace()
    eng._use_vlm = False
    eng._draft_kind = None
    eng.model_family = "chatml"
    eng._model_identity_str = identity
    return eng


def _save_session(eng: MLXEngine, sid: str) -> str:
    cache_state = PromptCacheState()
    cache_state.cache = [_make_kvcache(seq_len=4)]
    cache_state.token_ids = list(range(4))
    session = SessionState(
        cache_state=cache_state,
        messages=[{"role": "user", "content": "hi"}],
        total_cache_tokens=4,
        last_used=1.0,
        prompt_fingerprint=MLXEngine._prompt_fingerprint(
            None, True, model_identity=eng._model_identity(),
        ),
    )
    assert eng._save_session_to_disk(sid, session) is True
    return os.path.join(eng.cfg.cache_dir, f"session_{sid}.safetensors")


def _patch_make_prompt_cache(monkeypatch, n_layers: int = 1):
    monkeypatch.setattr(
        scm_module, "make_prompt_cache",
        lambda lm: [KVCache() for _ in range(n_layers)],
    )


def test_disk_metadata_round_trips_model_identity(tmp_path):
    eng = _make_engine(tmp_path, "modelA")
    path = _save_session(eng, "s")
    _, meta = load_prompt_cache(path, return_metadata=True)
    assert meta.get("model_identity") == "modelA"


def test_disk_load_refused_for_different_model(tmp_path, monkeypatch):
    # Model A builds + persists the cache; a DIFFERENT model B (same dir + sid)
    # must NOT reuse it.
    eng_a = _make_engine(tmp_path, "modelA")
    _save_session(eng_a, "s")

    eng_b = _make_engine(tmp_path, "modelB")  # same tmp_path -> same cache_dir
    _patch_make_prompt_cache(monkeypatch)
    assert eng_b._load_session_from_disk("s") is None  # REFUSED -> honest MISS


def test_disk_load_hits_same_model(tmp_path, monkeypatch):
    eng = _make_engine(tmp_path, "modelA")
    _save_session(eng, "s")
    _patch_make_prompt_cache(monkeypatch)
    loaded = eng._load_session_from_disk("s")
    assert loaded is not None
    assert loaded.messages == [{"role": "user", "content": "hi"}]


def test_disk_load_legacy_unstamped_cache_allowed(tmp_path, monkeypatch):
    # A legacy file (identity '') is NOT refused by the loader — it takes one
    # cold rebuild via the fingerprint gate at generation time instead.
    eng_legacy = _make_engine(tmp_path, "")  # no identity -> no stamp
    _save_session(eng_legacy, "s")

    eng_b = _make_engine(tmp_path, "modelB")
    _patch_make_prompt_cache(monkeypatch)
    loaded = eng_b._load_session_from_disk("s")
    assert loaded is not None  # allowed through (legacy migration path)


# ==========================================================================
# 3. Preflight / in-memory fingerprint gate reports MISS across models.
# ==========================================================================

_MSGS = [{"role": "user", "content": "hi"}]


def _preflight_shell(tmp_path, identity: str) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "chatml"
    eng._lock = threading.Lock()
    eng.cfg = SimpleNamespace(cache_dir=str(tmp_path), enable_thinking=True)
    eng._sessions = {}
    eng._has_disk_cache = lambda sid: False
    eng._messages_match = lambda stored, incoming, **kw: True
    eng._model_identity_str = identity
    return eng


def _install_memory_session(eng: MLXEngine, sid: str, *, built_by_identity: str):
    eng._sessions[sid] = SimpleNamespace(
        messages=list(_MSGS),
        total_cache_tokens=3,
        last_used=0.0,
        cache_state=SimpleNamespace(cache=[object()]),  # cache_live True
        thinking=True,
        prompt_fingerprint=MLXEngine._prompt_fingerprint(
            None, True, model_identity=built_by_identity,
        ),
    )


def test_preflight_memory_session_miss_across_models(tmp_path):
    # Session built by model A, engine is now model B -> advisory MISS.
    eng = _preflight_shell(tmp_path, "modelB")
    _install_memory_session(eng, "s", built_by_identity="modelA")
    out = eng.session_cache_preflight("s", _MSGS + [{"role": "user", "content": "u"}])
    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_rebuild"


def test_preflight_memory_session_hit_same_model(tmp_path):
    eng = _preflight_shell(tmp_path, "modelB")
    _install_memory_session(eng, "s", built_by_identity="modelB")
    out = eng.session_cache_preflight("s", _MSGS + [{"role": "user", "content": "u"}])
    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"


# ==========================================================================
# 4. _model_identity plumbing.
# ==========================================================================

def _write_weights(mdir, values, *, dtype=mx.float32):
    """Write a VALID safetensors shard (real 8-byte length prefix + JSON header
    + tensor data) so the STRENGTHENED marker (mtime_ns + header hash + sampled
    data digest, finding 2) can actually parse it. Fake raw bytes are NOT valid
    safetensors and now fail-close (see the marker tests below)."""
    mdir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(mdir / "model.safetensors"), {"w": mx.array(values, dtype=dtype)})


def test_model_identity_defaults_empty_for_shell_engine():
    eng = MLXEngine.__new__(MLXEngine)
    assert eng._model_identity() == ""


def test_compute_model_identity_is_stable_and_path_sensitive(tmp_path):
    _write_weights(tmp_path / "modelA", [1.0] * 64)
    _write_weights(tmp_path / "modelB", [1.0] * 64)  # same weights, different DIR
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = SimpleNamespace(model_path=str(tmp_path / "modelA"))
    eng.model_id = "modelA"
    cfg_json = {"model_type": "qwen", "vocab_size": 100}
    id1 = eng._compute_model_identity(cfg_json)
    id2 = eng._compute_model_identity(cfg_json)
    assert id1 and id1 == id2  # stable

    # Different path (a different model directory) -> different identity.
    eng2 = MLXEngine.__new__(MLXEngine)
    eng2.cfg = SimpleNamespace(model_path=str(tmp_path / "modelB"))
    eng2.model_id = "modelB"
    assert eng2._compute_model_identity(cfg_json) != id1

    # Same path but different config (weights/arch build) -> different identity.
    eng3 = MLXEngine.__new__(MLXEngine)
    eng3.cfg = SimpleNamespace(model_path=str(tmp_path / "modelA"))
    eng3.model_id = "modelA"
    assert eng3._compute_model_identity({"model_type": "qwen", "vocab_size": 200}) != id1


# ==========================================================================
# 5. Strengthened weight-build marker (finding 2): value-only / layout swaps at
#    a STABLE path (config unchanged) change the identity; a same-build restart
#    keeps it stable; an unverifiable dir FAILS CLOSED (unique-per-load).
# ==========================================================================

def _identity_for(model_dir, cfg_json, model_id="model"):
    """Compute identity from a fresh engine — models a server RESTART pointed at
    the SAME on-disk build."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = SimpleNamespace(model_path=str(model_dir))
    eng.model_id = model_id
    return eng._compute_model_identity(cfg_json)


def test_identity_stable_across_restart_same_weights(tmp_path):
    mdir = tmp_path / "model"
    _write_weights(mdir, [1.0] * 64)
    cfg_json = {"model_type": "qwen", "vocab_size": 100}
    # Two fresh engines over the SAME untouched files == a same-build reboot.
    a = _identity_for(mdir, cfg_json)
    b = _identity_for(mdir, cfg_json)
    assert a and a == b  # stable -> a same-model reboot still HITs (no cold rebuild)


def test_identity_changes_on_value_only_same_size_swap(tmp_path):
    # Finding 2: SAME layout + SAME size, only the VALUES change (different bytes,
    # mtime_ns bumped). The header hash + file size are unchanged; the SAMPLED
    # data digest is what catches it -> a DIFFERENT identity.
    mdir = tmp_path / "model"
    _write_weights(mdir, [1.0] * 64)
    cfg_json = {"model_type": "qwen", "vocab_size": 100}  # config UNCHANGED
    id_before = _identity_for(mdir, cfg_json)
    _write_weights(mdir, [2.0] * 64)  # same shape/dtype/size, different values
    id_after = _identity_for(mdir, cfg_json)
    assert id_before and id_after and id_before != id_after


def test_identity_changes_on_layout_change(tmp_path):
    # A dtype/layout change flips the safetensors HEADER (value-independent) ->
    # a different identity even if the sampled data window matched.
    mdir = tmp_path / "model"
    _write_weights(mdir, [1.0] * 64, dtype=mx.float32)
    cfg_json = {"model_type": "qwen"}
    id_f32 = _identity_for(mdir, cfg_json)
    _write_weights(mdir, [1.0] * 64, dtype=mx.float16)  # dtype/layout change
    id_f16 = _identity_for(mdir, cfg_json)
    assert id_f32 and id_f16 and id_f32 != id_f16


def test_weight_build_marker_none_for_missing_dir():
    # Finding 2 fail-closed: a nonexistent / unreadable path returns None (the
    # "cannot verify the build" signal) — NOT '' (which used to fall back to a
    # weak path+config identity a different model could share).
    assert MLXEngine._weight_build_marker("/no/such/model/dir") is None


def test_weight_build_marker_none_for_dir_without_safetensors(tmp_path):
    # A real, readable dir that holds NO safetensors cannot be vouched for.
    (tmp_path / "empty").mkdir()
    assert MLXEngine._weight_build_marker(str(tmp_path / "empty")) is None


def test_identity_fail_closed_unique_when_unverifiable(tmp_path):
    # Finding 2 fail-closed: a REAL model dir whose build cannot be verified
    # yields a UNIQUE-per-load identity (never a shareable path+config fallback),
    # so two loads NEVER match -> no cross-model reuse; a cold rebuild every boot.
    empty = tmp_path / "unreadable"
    empty.mkdir()
    cfg_json = {"model_type": "qwen"}
    id1 = _identity_for(empty, cfg_json)
    id2 = _identity_for(empty, cfg_json)
    assert id1 and id2 and id1 != id2  # unique-per-load, non-matching


def test_replaced_weights_refused_by_disk_loader(tmp_path, monkeypatch):
    """End-to-end: a cache built by the pre-swap build is REFUSED after weights
    are value-only replaced in place (config + layout unchanged) — no silent
    cross-build KV reuse."""
    mdir = tmp_path / "model"
    _write_weights(mdir, [1.0] * 64)
    cfg_json = {"model_type": "qwen"}

    id_before = _identity_for(mdir, cfg_json)
    eng = _make_engine(tmp_path, id_before)  # stamps the cache with the pre-swap id
    _save_session(eng, "s")

    _write_weights(mdir, [2.0] * 64)  # value-only in-place swap (same size/layout)
    id_after = _identity_for(mdir, cfg_json)
    assert id_after != id_before

    eng2 = _make_engine(tmp_path, id_after)
    _patch_make_prompt_cache(monkeypatch)
    assert eng2._load_session_from_disk("s") is None  # REFUSED -> honest MISS


def test_unverifiable_model_refuses_prior_cache_disk_loader(tmp_path, monkeypatch):
    """Finding 2 fail-closed end-to-end: a cache stamped by a verifiable build is
    REFUSED after the model dir becomes unverifiable (the new identity is a
    unique-per-load value that cannot match the stamp)."""
    mdir = tmp_path / "model"
    _write_weights(mdir, [1.0] * 64)
    cfg_json = {"model_type": "qwen"}
    id_ok = _identity_for(mdir, cfg_json)
    eng = _make_engine(tmp_path, id_ok)
    _save_session(eng, "s")

    # The weights vanish (dir unreadable / no safetensors) -> fail-closed unique.
    (mdir / "model.safetensors").unlink()
    id_broken = _identity_for(mdir, cfg_json)
    assert id_broken and id_broken != id_ok

    eng2 = _make_engine(tmp_path, id_broken)
    _patch_make_prompt_cache(monkeypatch)
    assert eng2._load_session_from_disk("s") is None  # REFUSED


# ==========================================================================
# 6. Finding 5 — a different-model disk cache is refused CHEAPLY, via the
#    header-only metadata read, BEFORE load_prompt_cache builds the KV.
# ==========================================================================

def test_disk_load_refused_without_loading_kv(tmp_path, monkeypatch):
    eng_a = _make_engine(tmp_path, "modelA")
    _save_session(eng_a, "s")

    eng_b = _make_engine(tmp_path, "modelB")
    _patch_make_prompt_cache(monkeypatch)

    # load_prompt_cache (the heavy KV load) MUST NOT run for a refused cache.
    calls = {"n": 0}

    def _must_not_load(*a, **k):
        calls["n"] += 1
        raise AssertionError(
            "load_prompt_cache called for a mismatched-model cache "
            "(finding 5: refusal must be header-only)"
        )

    monkeypatch.setattr(scm_module, "load_prompt_cache", _must_not_load)
    assert eng_b._load_session_from_disk("s") is None
    assert calls["n"] == 0  # refused BEFORE any tensor load


def test_disk_load_same_model_still_loads_kv(tmp_path, monkeypatch):
    # The header-first check must NOT block a matching model — the heavy load
    # still runs and the session is returned.
    eng = _make_engine(tmp_path, "modelA")
    _save_session(eng, "s")
    _patch_make_prompt_cache(monkeypatch)

    loaded = {"n": 0}
    real_load = scm_module.load_prompt_cache

    def _counting_load(*a, **k):
        loaded["n"] += 1
        return real_load(*a, **k)

    monkeypatch.setattr(scm_module, "load_prompt_cache", _counting_load)
    result = eng._load_session_from_disk("s")
    assert result is not None
    assert loaded["n"] == 1  # same model -> the KV load DID run
