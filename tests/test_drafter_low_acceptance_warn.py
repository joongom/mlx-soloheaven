"""Drafter low-acceptance WARNING surfaces without log grepping."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx_vlm.generate  # noqa: F401 — ensures sys.modules entry exists
from mlx_vlm.generate import PromptCacheState

from mlx_soloheaven.config import Config
from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
from mlx_soloheaven.engine.mlx_engine import MLXEngine

_LOG = "mlx_soloheaven.engine.mlx_engine"


def _engine():
    cfg = Config()
    cfg.pld_enabled = False
    cfg.prefill_step_size = 512
    eng = MLXEngine(cfg)
    eng._use_vlm = True
    eng._vlm_model = SimpleNamespace()
    eng._processor = SimpleNamespace()
    eng._drafter = SimpleNamespace(accept_lens=[])
    eng._draft_kind = "mtp"
    eng._safe_to_reuse_cache = lambda cs, pids=None: True
    return eng


def _drive(eng, monkeypatch, lens):
    # _run_vlm resets accept_lens to [] just before vlm_stream_generate,
    # so we populate it from the stub to mimic the real drafter's writes.
    def _fake(*_a, **_kw):
        eng._drafter.accept_lens.extend(lens)
        return iter([])
    monkeypatch.setattr(mlx_engine_module, "vlm_stream_generate", _fake)
    # PERF refactor: drafter-stats finalize is no longer wrapped per-token;
    # it is stashed on ``eng._pending_drafter_finalize`` and invoked by
    # ``_run`` post-loop. Mimic that here.
    list(eng._run_vlm(
        cache_state=PromptCacheState(), prompt_token_ids=[1, 2, 3],
        max_tokens=4, temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
        logits_processors=None, session_id="s1", total_prompt_tokens=3))
    finalize = getattr(eng, "_pending_drafter_finalize", None)
    if finalize is not None:
        finalize()
        eng._pending_drafter_finalize = None


def _low_accept_msgs(caplog):
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "low acceptance" in r.getMessage()]


def test_low_acceptance_fires_warning(monkeypatch, caplog):
    eng = _engine()  # mean=0.30, rounds=20 → WARNING
    try:
        with caplog.at_level(logging.WARNING, logger=_LOG):
            _drive(eng, monkeypatch, [0] * 14 + [1] * 6)
        msgs = _low_accept_msgs(caplog)
        assert len(msgs) == 1 and "mean_accepted=0.30" in msgs[0] and "over 20 rounds" in msgs[0]
    finally:
        eng.close()


def test_high_acceptance_no_warning(monkeypatch, caplog):
    eng = _engine()  # mean=1.50, rounds=20 → no warning
    try:
        with caplog.at_level(logging.WARNING, logger=_LOG):
            _drive(eng, monkeypatch, [1, 2] * 10)
        assert _low_accept_msgs(caplog) == []
    finally:
        eng.close()
