"""Backend-selection gate tests (mlx-lm-first migration, PR1).

After the migration, ``MLXEngine.load_model`` defaults to the mlx-lm backend
whenever mlx-lm SUPPORTS the model_type, and reserves mlx-vlm for an explicit
``--backend mlx-vlm`` opt-in (the MTP/vision drafter stack) or — under
``--backend auto`` — for a model_type mlx-lm cannot load.

The criterion is **mlx-lm-first BY SUPPORT, NOT by multimodal-ness**.
soloheaven is a TEXT-only server, so a config carrying ``vision_config`` does
NOT force mlx-vlm. gemma4 is a VLM family whose config ALWAYS has
``vision_config``, yet mlx-lm loads its text checkpoint (byte-identical to LM
Studio) — so gemma4 routes to mlx-lm under ``auto``.

These tests exercise the decision logic WITHOUT loading any real weights:
- The gate decision is ``MLXEngine._select_backend`` itself — the SAME
  production method ``load_model`` calls — driven against a bare
  ``MLXEngine.__new__`` instance with a stub cfg. ``_mlx_lm_supports`` and
  ``_vlm_supports`` are stubbed so no registry import or weight load happens.
  There is no test-local copy of the gate, so the production logic cannot
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


def _engine(
    *,
    backend: str,
    model_type: str,
    mlx_lm_supported: bool = True,
    vlm_supported: bool = True,
) -> MLXEngine:
    """Bare engine for path-selection only — no weights, no tokenizer.

    ``_mlx_lm_supports`` and ``_vlm_supports`` are normally registry imports;
    stub them so the test isolates the want_vlm gate rather than the installed
    mlx-lm / mlx-vlm model sets.
    """
    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng.cfg.backend = backend
    eng._model_type = model_type
    eng._mlx_lm_supports = lambda mt: mlx_lm_supported  # type: ignore[method-assign]
    eng._vlm_supports = lambda mt: vlm_supported  # type: ignore[method-assign]
    return eng


# gemma4 is a VLM family: its config ALWAYS carries vision_config — yet it
# must route to mlx-lm under auto because mlx-lm supports the gemma4 type.
GEMMA4_VISION_CONFIG = {
    "model_type": "gemma4",
    "vision_config": {"hidden_size": 1152},
}
# A made-up type that mlx-lm has no module for (text-only, vlm-only).
VLM_ONLY_CONFIG = {"model_type": "some_vlm_only_type"}


# --- _mlx_lm_supports unit coverage ----------------------------------------

def test_mlx_lm_supports_real_type_true():
    """mlx-lm has a module for gemma4 -> True (real registry probe)."""
    assert MLXEngine._mlx_lm_supports("gemma4") is True


def test_mlx_lm_supports_remapped_type_true():
    """A MODEL_REMAPPING alias resolves before the import probe.

    `mistral` remaps to `mlx_lm.models.llama`, so support is True even though
    there is no `mlx_lm.models.mistral` module.
    """
    assert MLXEngine._mlx_lm_supports("mistral") is True


def test_mlx_lm_supports_unknown_type_false():
    """A type mlx-lm has no module for (after remapping) -> False, defensively."""
    assert MLXEngine._mlx_lm_supports("definitely_not_a_real_model_type") is False
    assert MLXEngine._mlx_lm_supports("") is False
    assert MLXEngine._mlx_lm_supports(None) is False  # type: ignore[arg-type]


# --- gate cases (a)-(e), driven through REAL _select_backend ----------------

def test_a_gemma4_with_vision_auto_loads_mlx_lm():
    """(a) gemma4 (HAS vision_config) + auto => mlx-lm.

    The migration bug fix: vision_config must NOT force mlx-vlm. Because
    mlx-lm supports gemma4, auto routes to mlx-lm even though _vlm_supports
    would also accept it.
    """
    eng = _engine(
        backend="auto",
        model_type="gemma4",
        mlx_lm_supported=True,
        vlm_supported=True,
    )
    assert eng._select_backend(GEMMA4_VISION_CONFIG) is False  # -> mlx-lm


def test_b_mlx_lm_unsupported_type_auto_loads_vlm():
    """(b) a model_type mlx-lm lacks + auto => mlx-vlm.

    want_vlm becomes True only because mlx-lm cannot load the type; mlx-vlm
    supports it, so it routes to vlm.
    """
    eng = _engine(
        backend="auto",
        model_type="some_vlm_only_type",
        mlx_lm_supported=False,
        vlm_supported=True,
    )
    assert eng._select_backend(VLM_ONLY_CONFIG) is True  # -> mlx-vlm


def test_c_explicit_vlm_on_gemma4_loads_vlm():
    """(c) --backend mlx-vlm + gemma4 => mlx-vlm (explicit MTP/vision opt-in)."""
    eng = _engine(backend="mlx-vlm", model_type="gemma4")
    assert eng._select_backend(GEMMA4_VISION_CONFIG) is True  # -> mlx-vlm


def test_d_forced_mlx_lm_on_gemma4_loads_mlx_lm():
    """(d) --backend mlx-lm + gemma4 => mlx-lm (forced, want_vlm False)."""
    eng = _engine(backend="mlx-lm", model_type="gemma4")
    assert eng._select_backend(GEMMA4_VISION_CONFIG) is False  # -> mlx-lm


def test_e_auto_neither_supports_falls_to_mlx_lm(caplog):
    """(e) auto + a type neither backend supports => mlx-lm + warning.

    mlx-lm lacks the type, so want_vlm is True; but mlx-vlm also lacks it, so
    vlm_supported is False. The production gate returns False (caller falls
    through to mlx-lm) AND emits a warning. Runs through real production code
    (`_select_backend`), asserted via caplog rather than a mirrored copy.
    """
    eng = _engine(
        backend="auto",
        model_type="some_unsupported_type",
        mlx_lm_supported=False,
        vlm_supported=False,
    )
    with caplog.at_level(logging.WARNING):
        assert eng._select_backend(VLM_ONLY_CONFIG) is False  # -> mlx-lm

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a fallback warning from _select_backend"
    msg = warnings[-1].getMessage()
    assert "some_unsupported_type" in msg
    assert "mlx-vlm" in msg
    assert "mlx-lm" in msg


def test_explicit_vlm_unsupported_model_falls_to_mlx_lm(caplog):
    """--backend mlx-vlm but model_type not in vlm registry => mlx-lm + warning.

    want_vlm is True (explicit opt-in) but vlm_supported is False, so the gate
    returns False and warns. Complements case (e) for the explicit-opt-in path.
    """
    eng = _engine(
        backend="mlx-vlm",
        model_type="some_text_only_type",
        vlm_supported=False,
    )
    with caplog.at_level(logging.WARNING):
        assert eng._select_backend({"model_type": "some_text_only_type"}) is False

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
    eng = _engine(backend="auto", model_type="gemma4")
    eng.cfg.backend = "bogus"  # set directly, bypassing __post_init__
    with pytest.raises(ValueError) as exc:
        eng._select_backend(GEMMA4_VISION_CONFIG)
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
