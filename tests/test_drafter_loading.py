"""Drafter loading auto-detect tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine, _maybe_load_drafter


# ---------------------------------------------------------------------------
# _maybe_load_drafter
# ---------------------------------------------------------------------------

def test_maybe_load_drafter_returns_none_when_unset():
    model, kind = _maybe_load_drafter(None)
    assert model is None
    assert kind is None


def test_maybe_load_drafter_returns_none_for_empty_string():
    model, kind = _maybe_load_drafter("")
    assert model is None
    assert kind is None


def test_maybe_load_drafter_auto_detects_mtp_for_gemma4_assistant(monkeypatch):
    fake_model = SimpleNamespace(
        config=SimpleNamespace(block_size=4, model_type="gemma4_assistant"),
    )

    import mlx_vlm.speculative as spec_module

    def _fake_load_drafter(path, kind=None, **_kwargs):
        assert path == "/fake/drafter/path"
        if kind is None:
            return fake_model, "mtp"
        return fake_model, kind

    monkeypatch.setattr(spec_module, "load_drafter", _fake_load_drafter)

    model, kind = _maybe_load_drafter("/fake/drafter/path")
    assert model is fake_model
    assert kind == "mtp"


def test_maybe_load_drafter_falls_back_to_dflash_for_unknown(monkeypatch):
    fake_model = SimpleNamespace(
        config=SimpleNamespace(block_size=4, model_type="some_unknown_model"),
    )

    import mlx_vlm.speculative as spec_module

    def _fake_load_drafter(path, kind=None, **_kwargs):
        if kind is None:
            return fake_model, "dflash"
        return fake_model, kind

    monkeypatch.setattr(spec_module, "load_drafter", _fake_load_drafter)

    model, kind = _maybe_load_drafter("/fake/path/unknown_drafter")
    assert kind == "dflash"


def test_maybe_load_drafter_respects_explicit_kind(monkeypatch):
    fake_model = SimpleNamespace(config=SimpleNamespace(block_size=8))

    import mlx_vlm.speculative as spec_module

    captured = {}

    def _fake_load_drafter(path, kind=None, **_kwargs):
        captured["kind"] = kind
        return fake_model, kind or "dflash"

    monkeypatch.setattr(spec_module, "load_drafter", _fake_load_drafter)

    _, resolved = _maybe_load_drafter("/fake/path", kind="mtp")
    assert resolved == "mtp"
    assert captured["kind"] == "mtp"


# ---------------------------------------------------------------------------
# Engine init: drafter-on with mlx-lm legacy must refuse
# ---------------------------------------------------------------------------

def _bare_engine(*, use_vlm: bool, draft_model: str | None) -> MLXEngine:
    """Construct an MLXEngine just enough to exercise path selection."""
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.draft_model = draft_model
    eng.cfg.draft_kind = None
    eng.cfg.draft_block_size = None
    eng._use_vlm = use_vlm
    eng._vlm_model = SimpleNamespace()
    eng._processor = SimpleNamespace()
    eng._language_model = SimpleNamespace()
    eng.tokenizer = SimpleNamespace()
    eng.model_id = "stub"
    eng._drafter = None
    eng._draft_kind = None
    eng._safe_to_reuse_cache = lambda cs: True
    return eng


# ---------------------------------------------------------------------------
# Legacy path drafter rejection
# ---------------------------------------------------------------------------

def test_run_lm_legacy_rejects_drafter():
    eng = _bare_engine(use_vlm=False, draft_model="/some/path")
    eng._drafter = SimpleNamespace()  # truthy drafter
    eng._draft_kind = "mtp"

    from mlx_vlm.generate import PromptCacheState
    cache_state = PromptCacheState()

    with pytest.raises(RuntimeError) as exc:
        eng._run_lm_legacy(
            cache_state=cache_state,
            prompt_token_ids=[1, 2, 3],
            max_tokens=4,
            sampler=lambda lg: lg,
            logits_processors=None,
        )
    assert "draft-model" in str(exc.value) or "drafter" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# VLM path drafter pass-through
# ---------------------------------------------------------------------------

def test_run_vlm_passes_drafter_kwargs(monkeypatch):
    eng = _bare_engine(use_vlm=True, draft_model="/fake/drafter")
    fake_drafter = SimpleNamespace(accept_lens=[])
    eng._drafter = fake_drafter
    eng._draft_kind = "mtp"
    eng.cfg.draft_block_size = 4

    captured = {}

    def _fake_stream(*args, **kwargs):
        captured.update(kwargs)
        # Generator with no items.
        if False:
            yield None
        return

    monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)

    from mlx_vlm.generate import PromptCacheState
    cache_state = PromptCacheState()

    result = eng._run_vlm(
        cache_state=cache_state,
        prompt_token_ids=[1, 2, 3],
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        logits_processors=None,
        session_id="s",
        total_prompt_tokens=3,
    )
    # Drain the generator (it yields nothing) to confirm no errors.
    list(result)

    assert captured.get("draft_model") is fake_drafter
    assert captured.get("draft_kind") == "mtp"
    assert captured.get("draft_block_size") == 4


def test_run_vlm_no_drafter_kwargs_when_unset(monkeypatch):
    """No drafter loaded → call shape must stay byte-equal to baseline."""
    eng = _bare_engine(use_vlm=True, draft_model=None)
    eng._drafter = None
    eng._draft_kind = None

    captured = {}

    def _fake_stream(*args, **kwargs):
        captured.update(kwargs)
        if False:
            yield None
        return

    monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake_stream)

    from mlx_vlm.generate import PromptCacheState
    cache_state = PromptCacheState()

    list(eng._run_vlm(
        cache_state=cache_state,
        prompt_token_ids=[1, 2, 3],
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        logits_processors=None,
        session_id="s",
        total_prompt_tokens=3,
    ))

    assert "draft_model" not in captured
    assert "draft_kind" not in captured
    assert "draft_block_size" not in captured
