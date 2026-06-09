"""Backend-selection gate tests (mlx-lm-first migration, PR1).

After the migration, ``MLXEngine.load_model`` defaults TEXT models to the
mlx-lm backend and reserves mlx-vlm for genuinely multimodal configs (auto)
or an explicit ``--backend mlx-vlm`` opt-in (the MTP drafter stack).

These tests exercise the decision logic WITHOUT loading any real weights:
- ``MLXEngine._is_multimodal`` is a pure staticmethod over the config dict.
- The gate decision is ``MLXEngine._select_backend`` itself — the SAME
  production method ``load_model`` calls — driven against a bare
  ``MLXEngine.__new__`` instance with a stub cfg. ``_vlm_supports`` is
  stubbed so no mlx-vlm registry lookup or weight load happens. There is no
  longer a test-local copy of the gate, so the production logic cannot
  regress while the tests stay green.
- The drafter guard (``--draft-model`` on the mlx-lm path) is driven directly
  on the legacy ``_run_lm_legacy`` entry to assert the new error message.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine.mlx_engine import MLXEngine


def _engine(*, backend: str, model_type: str, vlm_supported: bool = True) -> MLXEngine:
    """Bare engine for path-selection only — no weights, no tokenizer.

    ``_vlm_supports`` is normally a registry import; stub it so the test
    isolates the want_vlm gate rather than mlx-vlm's installed model set.
    """
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.backend = backend
    eng._model_type = model_type
    eng._vlm_supports = lambda mt: vlm_supported  # type: ignore[method-assign]
    return eng


TEXT_CONFIG = {"model_type": "gemma3_text"}  # no vision/audio/image_token
VISION_CONFIG = {"model_type": "gemma3", "vision_config": {"hidden_size": 1152}}
AUDIO_CONFIG = {"model_type": "qwen2_audio", "audio_config": {"d_model": 1280}}
IMAGE_TOKEN_CONFIG = {"model_type": "llava", "image_token_index": 32000}


# --- _is_multimodal unit coverage ------------------------------------------

def test_is_multimodal_text_only_false():
    assert MLXEngine._is_multimodal(TEXT_CONFIG) is False
    assert MLXEngine._is_multimodal({}) is False
    assert MLXEngine._is_multimodal(None) is False


def test_is_multimodal_vision_audio_image_true():
    assert MLXEngine._is_multimodal(VISION_CONFIG) is True
    assert MLXEngine._is_multimodal(AUDIO_CONFIG) is True
    assert MLXEngine._is_multimodal(IMAGE_TOKEN_CONFIG) is True


def test_is_multimodal_falsy_values_ignored():
    # Present-but-empty/zero must NOT count as multimodal.
    assert MLXEngine._is_multimodal({"vision_config": {}}) is False
    assert MLXEngine._is_multimodal({"audio_config": None}) is False
    assert MLXEngine._is_multimodal({"image_token_index": 0}) is False


# --- gate cases (a)-(e), driven through REAL _select_backend ----------------

def test_a_text_auto_loads_mlx_lm():
    """(a) text config + backend=auto => mlx-lm (no vision/audio/image)."""
    eng = _engine(backend="auto", model_type="gemma3_text")
    assert eng._select_backend(TEXT_CONFIG) is False  # -> mlx-lm


def test_b_vision_auto_loads_vlm():
    """(b) vision_config + backend=auto => mlx-vlm."""
    eng = _engine(backend="auto", model_type="gemma3")
    assert eng._select_backend(VISION_CONFIG) is True  # -> mlx-vlm


def test_c_explicit_vlm_on_text_loads_vlm():
    """(c) backend=mlx-vlm + text => want vlm (explicit opt-in, supported)."""
    eng = _engine(backend="mlx-vlm", model_type="gemma3_text")
    assert eng._select_backend(TEXT_CONFIG) is True  # -> mlx-vlm


def test_d_forced_mlx_lm_on_multimodal_loads_mlx_lm():
    """(d) backend=mlx-lm + multimodal => mlx-lm (forced, want_vlm False)."""
    eng = _engine(backend="mlx-lm", model_type="gemma3")
    assert eng._select_backend(VISION_CONFIG) is False  # -> mlx-lm


def test_e_explicit_vlm_unsupported_model_falls_to_mlx_lm(caplog):
    """(e) backend=mlx-vlm but model_type not in registry => mlx-lm + warning.

    want_vlm is True but vlm_supported is False, so the production gate
    returns False (caller falls through to mlx-lm) AND emits a warning. This
    path now runs through real production code (`_select_backend`), asserted
    via caplog rather than a mirrored copy.
    """
    eng = _engine(
        backend="mlx-vlm", model_type="some_text_only_type", vlm_supported=False
    )
    with caplog.at_level(logging.WARNING):
        assert eng._select_backend(TEXT_CONFIG) is False  # -> mlx-lm

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a fallback warning from _select_backend"
    msg = warnings[-1].getMessage()
    assert "some_text_only_type" in msg
    assert "mlx-vlm" in msg
    assert "mlx-lm" in msg


# --- invalid backend validation (FINDING 1) --------------------------------

def test_config_rejects_invalid_backend():
    """Config.__post_init__ validates backend (env/programmatic bypass argparse
    `choices`): an invalid value must fail LOUDLY at construction."""
    with pytest.raises(ValueError) as exc:
        Config(backend="mlx-lmm")  # typo
    msg = str(exc.value)
    assert "mlx-lmm" in msg
    assert "auto/mlx-lm/mlx-vlm" in msg


def test_config_normalizes_backend_case():
    """A valid (but differently-cased) backend is lowercased, not rejected."""
    assert Config(backend="MLX-VLM").backend == "mlx-vlm"
    assert Config(backend="Auto").backend == "auto"


def test_model_config_rejects_invalid_backend():
    """Per-model ModelConfig also validates its backend field."""
    from mlx_soloheaven.config import ModelConfig

    with pytest.raises(ValueError) as exc:
        ModelConfig(model_path="/x", backend="vlm")  # invalid
    assert "auto/mlx-lm/mlx-vlm" in str(exc.value)


def test_valid_backends_unaffected():
    """All three valid values construct without error and pass through."""
    for value in ("auto", "mlx-lm", "mlx-vlm"):
        assert Config(backend=value).backend == value


def test_select_backend_raises_on_invalid_value():
    """Defense-in-depth: a programmatic bad cfg.backend that bypassed Config
    validation must raise in _select_backend, not silently route to mlx-lm."""
    eng = _engine(backend="auto", model_type="gemma3_text")
    eng.cfg.backend = "bogus"  # set directly, bypassing __post_init__
    with pytest.raises(ValueError) as exc:
        eng._select_backend(TEXT_CONFIG)
    assert "auto/mlx-lm/mlx-vlm" in str(exc.value)


# --- drafter guard on the mlx-lm path --------------------------------------

def test_drafter_on_mlx_lm_path_raises_new_message():
    """Drafter set on the mlx-lm legacy path => RuntimeError, new wording."""
    eng = MLXEngine.__new__(MLXEngine)
    eng._drafter = SimpleNamespace()  # truthy drafter present
    with pytest.raises(RuntimeError) as exc:
        eng._run_lm_legacy(
            cache_state=SimpleNamespace(cache=None, token_ids=[]),
            prompt_token_ids=[1, 2, 3],
            max_tokens=8,
            sampler=None,
            logits_processors=None,
        )
    msg = str(exc.value)
    assert "--backend mlx-vlm" in msg
    assert "--pld" in msg
