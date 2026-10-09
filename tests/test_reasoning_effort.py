"""Per-request ``reasoning_effort`` — API resolution, template gating, KV contract.

Qwen3.8's chat template declares a ``reasoning_effort`` render variable
(xhigh default / medium / low) that changes the SYSTEM preamble it emits.
Before this feature the server extracted only ``enable_thinking`` from
``chat_template_kwargs`` and every other key was silently dropped, so every
request rendered at the template's default.

Covers:
  * API resolution — top-level wins over chat_template_kwargs, nested is used
    when top-level is absent, neither -> None;
  * API validation — a label that could never be a template word is a 400,
    while a well-formed-but-unknown label is NOT rejected here (only the
    model's own template knows its vocabulary);
  * template gating — the label is forwarded to apply_chat_template ONLY when
    the template declares the variable, so fleets that mix families do not
    shard their KV cache on a model that ignores it;
  * KV contract — the label participates in the prompt fingerprint, and an
    absent label leaves the historical fingerprint byte-identical (no
    spurious rebuild for existing sessions);
  * SessionState carries the label the cache was BUILT under.
"""

from __future__ import annotations

from types import SimpleNamespace

from mlx_soloheaven.api import openai_compat as oc
from mlx_soloheaven.api.schemas import ChatCompletionRequest
from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState


def _req(**kw) -> ChatCompletionRequest:
    kw.setdefault("model", "m")
    kw.setdefault("messages", [{"role": "user", "content": "hi"}])
    return ChatCompletionRequest(**kw)


# ==========================================================================
# 1. API resolution — precedence mirrors _effective_thinking.
# ==========================================================================

def test_toplevel_reasoning_effort_wins_over_nested():
    r = _req(reasoning_effort="low",
             chat_template_kwargs={"reasoning_effort": "xhigh"})
    assert oc._effective_reasoning_effort(r) == "low"


def test_nested_reasoning_effort_used_when_no_toplevel():
    r = _req(chat_template_kwargs={"reasoning_effort": "medium"})
    assert oc._effective_reasoning_effort(r) == "medium"


def test_no_reasoning_effort_is_none():
    assert oc._effective_reasoning_effort(_req()) is None
    # An unrelated chat_template_kwargs payload must not invent one.
    assert oc._effective_reasoning_effort(
        _req(chat_template_kwargs={"enable_thinking": False})) is None


# ==========================================================================
# 2. API validation — shape only, vocabulary belongs to the template.
# ==========================================================================

def test_valid_labels_pass_validation():
    for label in (None, "low", "medium", "xhigh", "high", "minimal"):
        assert oc._validate_reasoning_effort(label) is None


def test_malformed_labels_are_400():
    for bad in (123, True, "", " ", "very high", "x" * 40, ["low"], {"a": 1},
                "low\n", "\nlow"):
        resp = oc._validate_reasoning_effort(bad)
        assert resp is not None, f"{bad!r} should be rejected"
        assert resp.status_code == 400


# ==========================================================================
# 3. Template gating — probe + effective label.
# ==========================================================================

def _engine(template: str | None) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "test-model"
    eng.tokenizer = SimpleNamespace(chat_template=template)
    return eng


def test_probe_detects_declaring_template():
    eng = _engine("{%- set e = reasoning_effort|default('xhigh') %}")
    assert eng._template_declares_reasoning_effort() is True


def test_probe_rejects_non_declaring_and_missing_template():
    assert _engine("{{ messages[0].content }}")._template_declares_reasoning_effort() is False
    assert _engine(None)._template_declares_reasoning_effort() is False


def test_effective_label_dropped_when_template_ignores_it():
    declaring = _engine("reasoning_effort")
    ignoring = _engine("plain template")
    assert declaring._effective_reasoning_effort("low") == "low"
    # Silently dropped, NOT an error: one client label may fan out to a fleet
    # whose other models never declare the variable.
    assert ignoring._effective_reasoning_effort("low") is None
    # Absent label stays absent on both.
    assert declaring._effective_reasoning_effort(None) is None
    assert declaring._effective_reasoning_effort("") is None


# ==========================================================================
# 4. Render — the kwarg reaches apply_chat_template, gated by the probe.
# ==========================================================================

class _RecordingTokenizer:
    def __init__(self, template: str):
        self.chat_template = template
        self.calls: list[dict] = []

    def apply_chat_template(self, formatted, **kwargs):
        self.calls.append(kwargs)
        return [1, 2, 3]


def _engine_with_recorder(template: str):
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "test-model"
    eng.tokenizer = _RecordingTokenizer(template)
    return eng


def test_label_is_forwarded_to_the_template_that_declares_it():
    eng = _engine_with_recorder("reasoning_effort")
    eng._tokenize_prompt([{"role": "user", "content": "hi"}],
                         thinking=True, reasoning_effort="low")
    assert eng.tokenizer.calls[-1]["reasoning_effort"] == "low"
    assert eng.tokenizer.calls[-1]["enable_thinking"] is True


def test_label_is_not_forwarded_to_a_template_without_it():
    eng = _engine_with_recorder("no such variable here")
    eng._tokenize_prompt([{"role": "user", "content": "hi"}],
                         thinking=True, reasoning_effort="low")
    assert "reasoning_effort" not in eng.tokenizer.calls[-1]


def test_absent_label_never_adds_the_kwarg():
    eng = _engine_with_recorder("reasoning_effort")
    eng._tokenize_prompt([{"role": "user", "content": "hi"}], thinking=True)
    assert "reasoning_effort" not in eng.tokenizer.calls[-1]


# ==========================================================================
# 5. KV contract — the label is part of the prompt fingerprint.
# ==========================================================================

def test_fingerprint_differs_by_reasoning_effort():
    low = MLXEngine._prompt_fingerprint(None, True, reasoning_effort="low")
    xhigh = MLXEngine._prompt_fingerprint(None, True, reasoning_effort="xhigh")
    assert low != xhigh
    # Stable for the same label.
    assert MLXEngine._prompt_fingerprint(None, True, reasoning_effort="low") == low


def test_absent_reasoning_effort_keeps_the_historical_fingerprint():
    # Existing sessions predate this key: an empty label must hash EXACTLY as
    # before, or every live cache takes a spurious rebuild on upgrade.
    base = MLXEngine._prompt_fingerprint(None, True)
    assert MLXEngine._prompt_fingerprint(None, True, reasoning_effort="") == base
    assert MLXEngine._prompt_fingerprint(None, True, reasoning_effort="low") != base


def test_existing_fingerprint_gates_still_work_with_effort():
    eff = "medium"
    a = MLXEngine._prompt_fingerprint([{"x": 1}], True, reasoning_effort=eff)
    b = MLXEngine._prompt_fingerprint([{"x": 2}], True, reasoning_effort=eff)
    assert a != b
    c = MLXEngine._prompt_fingerprint(None, False, reasoning_effort=eff)
    d = MLXEngine._prompt_fingerprint(None, True, reasoning_effort=eff)
    assert c != d
    e = MLXEngine._prompt_fingerprint(None, True, model_identity="mA", reasoning_effort=eff)
    f = MLXEngine._prompt_fingerprint(None, True, model_identity="mB", reasoning_effort=eff)
    assert e != f


# ==========================================================================
# 6. SessionState remembers the label the cache was built under.
# ==========================================================================

def test_session_state_carries_reasoning_effort():
    s = SessionState(cache_state=None, messages=[])
    assert s.reasoning_effort is None          # default = template default
    s2 = SessionState(cache_state=None, messages=[], reasoning_effort="low")
    assert s2.reasoning_effort == "low"


# ==========================================================================
# 7. Disk round-trip — a restored session keeps the label it was built under.
# ==========================================================================

def test_reasoning_effort_survives_a_disk_round_trip(tmp_path, monkeypatch):
    """The label is part of the prompt contract, so it must survive a restart.

    Without it a resumed session compacts / rebuilds under the TEMPLATE
    DEFAULT while its stored fingerprint says otherwise — a silent preamble
    change mid-conversation, or a spurious rebuild every turn.
    """
    import os

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from mlx_vlm.generate import PromptCacheState

    from mlx_soloheaven.config import Config

    eng = MLXEngine.__new__(MLXEngine)
    cfg = Config()
    cfg.data_dir = str(tmp_path)
    cfg.disk_budget_gb = 1.0
    eng.cfg = cfg
    eng._sessions = {}
    eng._dirty_sessions = set()
    eng._disk_session_ids = set()
    eng._language_model = SimpleNamespace()
    eng._use_vlm = False
    eng._draft_kind = None
    eng.model_family = "chatml"
    eng._model_identity_str = "qwen38"

    kv = KVCache()
    k = mx.arange(2 * 4 * 8, dtype=mx.float32).reshape(1, 2, 4, 8)
    kv.update_and_fetch(k, k * 0.5)
    cache_state = PromptCacheState()
    cache_state.cache = [kv]
    cache_state.token_ids = list(range(4))

    session = SessionState(
        cache_state=cache_state,
        messages=[{"role": "user", "content": "hi"}],
        total_cache_tokens=4,
        last_used=1.0,
        reasoning_effort="low",
        prompt_fingerprint=MLXEngine._prompt_fingerprint(
            None, True, model_identity="qwen38", reasoning_effort="low",
        ),
    )
    assert eng._save_session_to_disk("s-effort", session) is True
    assert os.path.exists(
        os.path.join(eng.cfg.cache_dir, "session_s-effort.safetensors")
    )

    # The loader rebuilds the cache container from the language model; this
    # shell engine has none (same scaffolding as test_cross_model_cache_safety).
    from mlx_soloheaven.engine import session_cache_mixin as scm_module
    monkeypatch.setattr(
        scm_module, "make_prompt_cache", lambda lm: [KVCache()],
    )

    restored = eng._load_session_from_disk("s-effort")
    assert restored is not None
    assert restored.reasoning_effort == "low"
    # And the restored contract still matches a request carrying the same
    # label (no spurious rebuild), while a different label does not.
    same = MLXEngine._prompt_fingerprint(
        None, True, model_identity="qwen38", reasoning_effort="low")
    other = MLXEngine._prompt_fingerprint(
        None, True, model_identity="qwen38", reasoning_effort="xhigh")
    assert restored.prompt_fingerprint == same
    assert restored.prompt_fingerprint != other


# ==========================================================================
# 8. Codex review round 1 — the label must not shard a cache it never changed.
# ==========================================================================

def test_label_is_inert_when_thinking_is_off():
    """Qwen3.8's template skips the whole effort block when thinking is off, so
    low/medium/xhigh render identically there. Keeping the label in the
    contract would rebuild the KV cache on every label change for a prompt that
    never changed."""
    eng = _engine_with_recorder("reasoning_effort")
    eng._tokenize_prompt([{"role": "user", "content": "hi"}],
                         thinking=False, reasoning_effort="low")
    assert "reasoning_effort" not in eng.tokenizer.calls[-1]
    # ...and the fingerprint form drops it in lockstep, so render and contract
    # can never disagree.
    assert eng._fp_reasoning_effort("low", False) == ""
    assert eng._fp_reasoning_effort("low", True) == "low"


def test_render_and_fingerprint_drop_the_label_together():
    """The coupling IS the safety property: whatever leaves the prompt leaves
    the contract, so no cache built under one preamble can serve another."""
    for template, thinking in (("reasoning_effort", False),
                               ("no variable", True),
                               ("no variable", False)):
        eng = _engine_with_recorder(template)
        eng._tokenize_prompt([{"role": "user", "content": "hi"}],
                             thinking=thinking, reasoning_effort="low")
        rendered = "reasoning_effort" in eng.tokenizer.calls[-1]
        fingerprinted = bool(eng._fp_reasoning_effort("low", thinking))
        assert rendered == fingerprinted is False


# ==========================================================================
# 9. Codex review round 1 — error attribution by experiment, not by presence.
# ==========================================================================

class _RaisingTokenizer:
    """Raises for a reason of OUR making (not the caller's label)."""

    chat_template = "reasoning_effort"

    def apply_chat_template(self, formatted, **kwargs):
        raise ValueError("No user query found in messages.")


class _LabelRejectingTokenizer:
    """Raises ONLY when the label is present — the real Qwen3.8 behavior."""

    chat_template = "reasoning_effort"

    def apply_chat_template(self, formatted, **kwargs):
        if "reasoning_effort" in kwargs:
            raise ValueError(
                "Unexpected reasoning effort high. Supported types are "
                "xhigh (default), medium, and low."
            )
        return [1, 2, 3]


def _engine_with(tokenizer):
    eng = MLXEngine.__new__(MLXEngine)
    eng.model_id = "test-model"
    eng.tokenizer = tokenizer
    return eng


def test_template_error_not_caused_by_the_label_keeps_its_type():
    """Blaming every render failure on reasoning_effort would downgrade a real
    500 to a 400 AND echo internal template text back to the caller."""
    from mlx_soloheaven.engine.types import InvalidPromptParameterError

    eng = _engine_with(_RaisingTokenizer())
    try:
        eng._tokenize_prompt([{"role": "system", "content": "s"}],
                             thinking=True, reasoning_effort="low")
    except InvalidPromptParameterError:  # pragma: no cover - the bug
        raise AssertionError("misattributed an unrelated template error")
    except ValueError as e:
        assert "No user query" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected the original error")


def test_template_error_caused_by_the_label_becomes_a_400():
    from mlx_soloheaven.engine.types import InvalidPromptParameterError

    eng = _engine_with(_LabelRejectingTokenizer())
    try:
        eng._tokenize_prompt([{"role": "user", "content": "hi"}],
                             thinking=True, reasoning_effort="high")
    except InvalidPromptParameterError as e:
        assert "reasoning_effort" in str(e)
        assert "Supported types are" in str(e)   # template's own message kept
    else:  # pragma: no cover
        raise AssertionError("expected InvalidPromptParameterError")


# ==========================================================================
# 10. Codex review round 1 — both input forms answer the SAME status.
# ==========================================================================

def test_both_forms_reject_a_non_string_the_same_way():
    """A wrong type must not answer 422 top-level and 400 nested — the two
    forms are documented as equivalent."""
    top = _req(reasoning_effort=123)
    nested = _req(chat_template_kwargs={"reasoning_effort": 123})
    for r in (top, nested):
        resp = oc._validate_reasoning_effort(oc._effective_reasoning_effort(r))
        assert resp is not None and resp.status_code == 400


# ==========================================================================
# 11. Codex review round 2 — every engine entry point must accept the label.
# ==========================================================================

def test_async_wrappers_accept_every_generate_stream_parameter():
    """The API passes the same keyword set to whichever engine serves it. In
    PROCESS mode the proxy takes ``**params``, so a parameter missing from the
    in-process async wrappers only breaks INPROCESS mode — every stream=true
    request there died with a TypeError while production looked healthy.

    The wrappers create and own ``cancel_event`` themselves, so it is the one
    parameter they are allowed not to expose."""
    import inspect

    def params(fn):
        return {n for n in inspect.signature(fn).parameters if n != "self"}

    base = params(MLXEngine.generate_stream) - {"cancel_event"}
    for name in ("generate_stream_async", "generate_stream_batches_async"):
        missing = base - params(getattr(MLXEngine, name))
        assert not missing, f"MLXEngine.{name} does not accept {sorted(missing)}"
    assert "reasoning_effort" in params(MLXEngine.complete)
