"""Qwen3.5/3.6 MTP (qwen3_5_mtp head) on the mlx-lm path — hermetic tests.

No real model weights: the head loader is exercised with a tiny synthetic
checkpoint, and the round loop with duck-typed mock ops + mock caches
(ArraysCache-like: untrimmable, snapshot/restorable ``.cache`` list;
KVCache-like: ``.offset``/``.trim``).

Covers:
  * head loader — strict key accounting, full-attention forced (the
    layer_types trap), NO +1 norm shift, quantized round-trip, model_type
    validation;
  * round loop — greedy output identical to the plain chain under partial
    draft rejections (rollback exactness, proved on the ArraysCache mock's
    recorded token history);
  * per-layer fail-closed — a lying KVCache (trim that does not trim) fires
    on_cache_corruption and permanently disables speculation, stream alive;
  * logits-processor history — exactly one call per emitted token, history
    == suffix + emitted (PLD FIX-4 contract);
  * finalize/pending — head_offset == target_offset after the stream and
    append-only multi-turn reuse through validate_mtp_cache_reuse;
  * engine wiring — _run_lm_legacy dispatches qwen_mtp, appends head cache
    entries, keeps the mlx-vlm raise for other drafter kinds, and disables
    MTP for structured-output requests.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import List

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_soloheaven.engine.qwen_mtp import (
    MTPCacheCorruption,
    QwenMTPHead,
    classify_cache,
    load_qwen_mtp_head,
    make_head_cache,
    qwen_mtp_generate_step,
    read_model_type,
    validate_mtp_cache_reuse,
)


# ---------------------------------------------------------------------------
# Tiny synthetic head checkpoint
# ---------------------------------------------------------------------------

TINY_TEXT_CFG = {
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "rms_norm_eps": 1e-6,
    "vocab_size": 99,
    "full_attention_interval": 4,
    "mtp_num_hidden_layers": 1,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "rope_parameters": {
        "type": "default",
        "mrope_section": [11, 11, 10],
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
}


def _build_tiny_head():
    from mlx_lm.models.qwen3_5 import TextModelArgs

    args = TextModelArgs.from_dict(TINY_TEXT_CFG)
    return QwenMTPHead(args, num_layers=1)


def _write_checkpoint(tmp_path, head, quantization=None):
    cfg = {
        "model_type": "qwen3_5_mtp",
        "block_size": 3,
        "text_config": TINY_TEXT_CFG,
    }
    if quantization:
        cfg["quantization"] = quantization
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(cfg, f)
    weights = dict(tree_flatten(head.parameters()))
    mx.save_safetensors(os.path.join(tmp_path, "model.safetensors"), weights)
    return weights


def test_loader_roundtrip_unquantized_no_norm_shift(tmp_path):
    head = _build_tiny_head()
    # Distinctive norm value: a +1 shift (the mlx-lm 0.31.3 sanitize bug for
    # checkpoints with mtp keys) would turn 0.5 into 1.5 — must NOT happen.
    head.norm.weight = mx.full(head.norm.weight.shape, 0.5)
    head.pre_fc_norm_hidden.weight = mx.full(
        head.pre_fc_norm_hidden.weight.shape, 0.25
    )
    saved = _write_checkpoint(str(tmp_path), head)

    loaded, info = load_qwen_mtp_head(str(tmp_path))
    assert info["model_type"] == "qwen3_5_mtp"
    assert info["block_size"] == 3
    assert info["num_layers"] == 1
    assert info["hidden_size"] == 64
    assert info["vocab_size"] == 99
    assert info["num_weights"] == len(saved)
    # Verbatim norms (no +1 shift, never through qwen3_5 sanitize).
    assert mx.allclose(loaded.norm.weight, mx.full((64,), 0.5)).item()
    assert mx.allclose(
        loaded.pre_fc_norm_hidden.weight, mx.full((64,), 0.25)
    ).item()
    # The layer_types trap: the head layer MUST be full attention even
    # though the config carries the target's full_attention_interval=4.
    assert loaded.layers[0].is_linear is False
    assert hasattr(loaded.layers[0], "self_attn")
    assert not hasattr(loaded.layers[0], "linear_attn")
    # fc: 2*hidden -> hidden.
    assert loaded.fc.weight.shape == (64, 128)


def test_loader_quantized_roundtrip(tmp_path):
    head = _build_tiny_head()
    nn.quantize(head, group_size=32, bits=5, mode="affine")
    _write_checkpoint(
        str(tmp_path),
        head,
        quantization={"group_size": 32, "bits": 5, "mode": "affine"},
    )
    loaded, info = load_qwen_mtp_head(str(tmp_path))
    # Quantized modules rebuilt (fc has scales) + strict load consumed all.
    assert hasattr(loaded.fc, "scales")
    assert loaded.fc.weight.dtype == mx.uint32
    # Norms stay unquantized float vectors.
    assert loaded.norm.weight.dtype != mx.uint32


def test_loader_rejects_wrong_model_type(tmp_path):
    head = _build_tiny_head()
    _write_checkpoint(str(tmp_path), head)
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5", "text_config": TINY_TEXT_CFG}, f)
    with pytest.raises(ValueError) as exc:
        load_qwen_mtp_head(str(tmp_path))
    assert "qwen3_5_mtp" in str(exc.value)


def test_loader_strict_rejects_unconsumed_keys(tmp_path):
    head = _build_tiny_head()
    weights = dict(tree_flatten(head.parameters()))
    weights["bogus.weight"] = mx.zeros((4,))
    cfg = {"model_type": "qwen3_5_mtp", "block_size": 3, "text_config": TINY_TEXT_CFG}
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump(cfg, f)
    mx.save_safetensors(os.path.join(tmp_path, "model.safetensors"), weights)
    with pytest.raises(Exception):
        load_qwen_mtp_head(str(tmp_path))


def test_read_model_type(tmp_path):
    assert read_model_type(str(tmp_path / "nope")) is None
    with open(os.path.join(tmp_path, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5_mtp"}, f)
    assert read_model_type(str(tmp_path)) == "qwen3_5_mtp"


# ---------------------------------------------------------------------------
# Mock caches + ops for the round loop
# ---------------------------------------------------------------------------


class MockArrays:
    """ArraysCache-like: recurrent state, untrimmable, replaced functionally.
    cache[0] records the exact consumed-token history (ground truth for
    rollback exactness)."""

    def __init__(self):
        self.cache = [()]


class MockKV:
    """KVCache-like: offset + honest trim."""

    def __init__(self):
        self.offset = 0
        self.keys = None
        self.values = None

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class LyingKV(MockKV):
    """trim() claims success but never decrements — per-layer fail-closed
    verification must catch it."""

    def trim(self, n):
        return n


class MockOps:
    """Duck-typed stand-in for QwenMTPOps.

    Hidden encoding: position hidden == the token consumed at that position
    (shape (1, T, 1)), so target logits depend on the actual forwarded token
    — exactly the real semantics, and ghost tokens in a mis-rolled-back
    cache would surface as wrong outputs.
    """

    def __init__(self, next_map, head_map, vocab=64):
        self.next_map = next_map
        self.head_map = head_map
        self.vocab = vocab

    def _hid(self, toks):
        return mx.array([[[float(t)] for t in toks]])

    def _logits(self, toks, fn):
        rows = []
        for t in toks:
            row = [0.0] * self.vocab
            row[fn(int(t))] = 1e9
            rows.append(row)
        return mx.array([rows])

    def target_hidden(self, toks2d, cache):
        toks = [int(t) for t in toks2d[0].tolist()]
        for c in cache:
            if isinstance(getattr(c, "cache", None), list):
                # Functional replacement (mirrors GatedDeltaNet): the OLD
                # tuple object survives in any snapshot references.
                c.cache = [c.cache[0] + tuple(toks)]
            else:
                c.offset += len(toks)
        return self._hid(toks)

    def target_logits(self, h):
        toks = [int(round(v[0])) for v in h[0].tolist()]
        return self._logits(toks, self.next_map)

    def head_hidden(self, hidden, next_ids, mtp_cache):
        toks = [int(t) for t in next_ids[0].tolist()]
        for c in mtp_cache:
            c.offset += len(toks)
        return self._hid(toks)

    def head_logits(self, x):
        toks = [int(round(v[0])) for v in x[0].tolist()]
        return self._logits(toks, self.head_map)


def _mk_caches(kv_cls=MockKV):
    """5 target entries (3 arrays-like + 2 kv-like, qwen hybrid shape) + 1
    head KV entry, as one 6-entry prompt_cache list."""
    target = [MockArrays(), MockArrays(), kv_cls(), MockArrays(), kv_cls()]
    return target + [MockKV()]


def _chain(start, n, fn):
    out, t = [], start
    for _ in range(n):
        t = fn(t)
        out.append(t)
    return out


def _run(
    suffix,
    next_map,
    head_map,
    *,
    max_tokens,
    block_size=3,
    cache=None,
    processors=None,
    callbacks=None,
):
    cache = cache if cache is not None else _mk_caches()
    fired = {"corruption": 0, "pending": "UNSET"}

    def on_corruption():
        fired["corruption"] += 1

    def on_finalize(tok):
        fired["pending"] = tok

    ops = MockOps(next_map, head_map)
    gen = qwen_mtp_generate_step(
        mx.array(suffix, mx.uint32),
        model=None,
        head=None,
        block_size=block_size,
        max_tokens=max_tokens,
        sampler=None,  # greedy
        logits_processors=processors,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=on_corruption,
        on_finalize=on_finalize,
        ops=ops,
    )
    out = [(t, fd) for t, _lp, fd in gen]
    if callbacks is not None:
        callbacks.update(fired)
    return [t for t, _ in out], [fd for _, fd in out], cache, fired


def test_greedy_identity_with_partial_rejections():
    # Target truth: t -> t+1 (mod 50). Head: correct EXCEPT after even
    # tokens (proposes 63) -> rounds alternate accepts/rejections.
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: 63 if t % 2 == 0 else (t + 1) % 50
    suffix = [1, 2, 3]
    N = 12
    truth = _chain(suffix[-1], N, next_map)

    out, flags, cache, fired = _run(
        suffix, next_map, head_map, max_tokens=N
    )
    assert out == truth, f"MTP output diverged: {out} vs {truth}"
    # Some drafts accepted, some rejected (both paths exercised).
    assert any(flags), "expected at least one accepted draft"
    assert not all(flags), "expected at least one non-draft token"

    # ROLLBACK EXACTNESS: the ArraysCache history must be exactly the
    # consumed tokens (suffix + emitted[:-1]); any rejected-draft ghost
    # token would appear here.
    consumed = tuple(suffix + out[:-1])
    for i in (0, 1, 3):
        assert cache[i].cache[0] == consumed, (
            f"layer {i} recurrent history has ghost tokens:\n"
            f"  got      {cache[i].cache[0]}\n  expected {consumed}"
        )
    # KV offsets settled to exactly the consumed length.
    for i in (2, 4):
        assert cache[i].offset == len(consumed)
    # Finalize: head committed the pending pair -> head == target offset.
    assert cache[5].offset == len(consumed)
    assert fired["pending"] == out[-1]
    assert fired["corruption"] == 0


def test_greedy_identity_full_accept():
    next_map = lambda t: (t + 3) % 40
    out, flags, cache, fired = _run(
        [5, 6], next_map, next_map, max_tokens=9
    )
    truth = _chain(6, 9, next_map)
    assert out == truth
    # Perfect head: every non-bonus round token came from a draft.
    assert sum(flags) >= 6
    assert fired["pending"] == out[-1]
    consumed = tuple([5, 6] + out[:-1])
    assert cache[0].cache[0] == consumed
    assert cache[5].offset == cache[2].offset == len(consumed)


def test_lying_cache_fires_fail_closed():
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: 63  # always wrong -> first round must roll back
    cache = _mk_caches(kv_cls=LyingKV)
    cache[5] = MockKV()  # head cache stays honest
    out, flags, cache, fired = _run(
        [1, 2, 3], next_map, head_map, max_tokens=8, cache=cache
    )
    # Fail-closed fired: corruption callback + speculation disabled.
    assert fired["corruption"] >= 1
    # Stream stayed alive to max_tokens.
    assert len(out) == 8
    # After the trip, no draft is ever accepted again (plain decode).
    first_true = next((i for i, f in enumerate(flags) if f), None)
    assert first_true is None, "no draft should be accepted (head always wrong)"
    # Broken stream must NOT report a pending finalize (head desynced).
    assert fired["pending"] is None


def test_restore_is_reference_exact():
    """A rejected round must restore the EXACT pre-round ArraysCache state
    object contents (reference snapshot), then replay only consumed tokens."""
    next_map = lambda t: (t + 1) % 30
    head_map = lambda t: 29 - (t % 7)  # mostly wrong
    suffix = [4, 5]
    out, _flags, cache, _f = _run(suffix, next_map, head_map, max_tokens=6)
    assert out == _chain(5, 6, next_map)
    assert cache[0].cache[0] == tuple(suffix + out[:-1])


class _Recorder:
    def __init__(self):
        self.seen: List[List[int]] = []

    def __call__(self, tokens, logits):
        if tokens is not None:
            self.seen.append(
                tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
            )
        return logits


def test_processor_history_exact_one_call_per_token():
    # Head accepts exactly 1 draft then rejects (head right only after odd
    # tokens with t->t+1 truth: tokens alternate parity) -> every round
    # emits 2 tokens; max_tokens = 1 + 2r lands exactly on a round boundary
    # so processor call : emitted token is exactly 1:1.
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: (t + 1) % 50 if t % 2 == 1 else 63
    suffix = [1, 2, 3]
    rec = _Recorder()
    N = 9
    out, _flags, _cache, _f = _run(
        suffix, next_map, head_map, max_tokens=N, processors=[rec]
    )
    assert out == _chain(3, N, next_map)
    assert len(rec.seen) == len(out), (
        f"{len(rec.seen)} processor calls != {len(out)} emitted tokens"
    )
    for j, seen in enumerate(rec.seen):
        assert seen == suffix + out[:j], (
            f"call {j}: history {seen} != suffix+emitted {suffix + out[:j]}"
        )


def test_processor_history_full_accept():
    next_map = lambda t: (t + 2) % 40
    rec = _Recorder()
    out, _flags, _cache, _f = _run(
        [7, 8], next_map, next_map, max_tokens=9, processors=[rec]
    )
    assert out == _chain(8, 9, next_map)
    assert len(rec.seen) == len(out)
    for j, seen in enumerate(rec.seen):
        assert seen == [7, 8] + out[:j]


def test_early_close_settles_exactly():
    """Consumer break (EOS path: the adapter closes the generator at a
    mid-round yield) — finally must settle to exactly prompt + yielded - 1
    and finalize the pending pair."""
    next_map = lambda t: (t + 1) % 50
    suffix = [1, 2, 3]
    cache = _mk_caches()
    fired = {}
    got: List[int] = []

    gen = qwen_mtp_generate_step(
        mx.array(suffix, mx.uint32),
        model=None,
        head=None,
        block_size=3,
        max_tokens=100,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: fired.__setitem__("corruption", True),
        on_finalize=lambda tok: fired.__setitem__("pending", tok),
        ops=MockOps(next_map, next_map),
    )
    for t, _lp, _fd in gen:
        got.append(t)
        if len(got) == 3:  # break INSIDE a speculative round (1 bootstrap + 2)
            break
    gen.close()

    assert got == _chain(3, 3, next_map)
    consumed = tuple(suffix + got[:-1])
    for i in (0, 1, 3):
        assert cache[i].cache[0] == consumed, (
            f"layer {i}: early-close left ghost tokens: {cache[i].cache[0]} "
            f"!= {consumed}"
        )
    for i in (2, 4):
        assert cache[i].offset == len(consumed)
    assert cache[5].offset == len(consumed)  # finalize ran
    assert fired.get("pending") == got[-1]
    assert "corruption" not in fired


def test_multiturn_append_reuse():
    """Turn 2 reuses the finalized cache: validate_mtp_cache_reuse accepts
    the append-only prompt + matching pending token, and the second run is
    byte-exact from the committed state."""
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: (t + 1) % 50 if t % 3 else 63
    suffix1 = [1, 2, 3]
    out1, _fl, cache, fired = _run(
        suffix1, next_map, head_map, max_tokens=8
    )
    pending = fired["pending"]
    assert pending == out1[-1]
    stored_ids = suffix1 + out1[:-1]  # what the engine reconcile records
    assert cache[2].offset == len(stored_ids)

    # Next turn: append-only prompt = stored + pending + new user tokens.
    new_prompt = stored_ids + [pending, 20, 21]
    ok, why = validate_mtp_cache_reuse(cache, stored_ids, new_prompt, 5, 1, pending)
    assert ok, why

    suffix2 = new_prompt[len(stored_ids):]
    out2, _fl2, cache, fired2 = _run(
        suffix2, next_map, head_map, max_tokens=6, cache=cache
    )
    assert out2 == _chain(21, 6, next_map)
    consumed = tuple(stored_ids + suffix2 + out2[:-1])
    assert cache[0].cache[0] == consumed
    assert cache[5].offset == cache[2].offset == len(consumed)
    assert fired2["pending"] == out2[-1]


def test_validate_mtp_cache_reuse_fail_closed():
    next_map = lambda t: (t + 1) % 50
    out, _fl, cache, fired = _run([1, 2, 3], next_map, next_map, max_tokens=6)
    stored = [1, 2, 3] + out[:-1]
    pend = fired["pending"]
    good = stored + [pend, 30]

    ok, _ = validate_mtp_cache_reuse(cache, stored, good, 5, 1, pend)
    assert ok
    # layout: missing head entry
    ok, why = validate_mtp_cache_reuse(cache[:5], stored, good, 5, 1, pend)
    assert not ok and "layout" in why
    # head offset desync
    cache[5].offset -= 1
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, pend)
    assert not ok and "head offset" in why
    cache[5].offset += 1
    # divergence (ArraysCache untrimmable -> must cold-fill)
    diverged = stored[:-1] + [49, pend, 30]
    ok, why = validate_mtp_cache_reuse(cache, stored, diverged, 5, 1, pend)
    assert not ok and "divergence" in why
    # pending-pair token mismatch (stale head slot)
    bad_pend = stored + [(pend + 1) % 50, 30]
    ok, why = validate_mtp_cache_reuse(cache, stored, bad_pend, 5, 1, pend)
    assert not ok and "pending" in why
    # unknown pending (e.g. session reloaded from disk) -> fail closed
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, None)
    assert not ok and "pending" in why
    # no new suffix
    ok, why = validate_mtp_cache_reuse(cache, stored, list(stored), 5, 1, pend)
    assert not ok
    # target offset != stored ids
    cache[2].offset += 1
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, pend)
    assert not ok
    cache[2].offset -= 1


def test_classify_cache():
    cache = _mk_caches()
    arrays_idx, kv_idx, other_idx = classify_cache(cache[:5])
    assert arrays_idx == [0, 1, 3]
    assert kv_idx == [2, 4]
    assert other_idx == []
    arrays_idx, kv_idx, other_idx = classify_cache([object()])
    assert other_idx == [0]


def test_max_tokens_one_no_round():
    next_map = lambda t: (t + 1) % 50
    out, flags, cache, fired = _run([1, 2], next_map, next_map, max_tokens=1)
    assert out == [next_map(2)] == [3]
    assert flags == [False]
    # Target consumed exactly the suffix; head finalized to match.
    assert cache[2].offset == 2
    assert cache[5].offset == 2
    assert fired["pending"] == 3


# ---------------------------------------------------------------------------
# Engine wiring (_run_lm_legacy dispatch)
# ---------------------------------------------------------------------------


def _mtp_engine():
    from mlx_soloheaven.config import Config
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.cfg = Config()
    eng._use_vlm = False
    eng.model_id = "stub"
    eng._drafter = SimpleNamespace(layers=[object()])
    eng._draft_kind = "qwen_mtp"
    eng._mtp_block_size = 3
    eng._language_model = SimpleNamespace(
        layers=[object()] * 5,
        make_cache=lambda: [
            MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(),
        ],
    )
    eng.tokenizer = SimpleNamespace(decode=lambda ids: "")
    return eng


def test_run_lm_legacy_dispatches_qwen_mtp(monkeypatch):
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_vlm.generate import PromptCacheState

    captured = {}

    def fake_step(prompt, model, head, **kwargs):
        captured.update(kwargs)
        captured["prompt_len"] = int(prompt.size)
        return iter(())

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    eng = _mtp_engine()
    cache_state = PromptCacheState()
    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=[1, 2, 3, 4],
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
    )
    list(gen_iter)  # drain the (empty) adapter

    assert captured["prompt_len"] == 4
    assert captured["n_target_layers"] == 5
    assert captured["block_size"] == 3
    # Head KV entries appended after the 5 target entries.
    assert len(prompt_cache) == 6
    assert captured["prompt_cache"] is prompt_cache


def test_run_lm_legacy_qwen_mtp_structured_falls_back(monkeypatch):
    """response_format json_object -> MTP off for the request (FSM keeps
    single-step advance); no head entries appended; plain path used."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_vlm.generate import PromptCacheState

    def boom(*a, **k):
        raise AssertionError("qwen_mtp_generate_step must not run for FSM")

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", boom)

    plain = {}

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        plain["prompt"] = list(prompt)
        plain.update(kwargs)
        return iter(())

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

    eng = _mtp_engine()
    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=PromptCacheState(),
        prompt_token_ids=[1, 2, 3],
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
        response_format=SimpleNamespace(type="json_object"),
    )
    list(gen_iter)
    assert plain["prompt"] == [1, 2, 3]
    assert len(prompt_cache) == 5  # NO head entries for the non-MTP path


def test_run_lm_legacy_qwen_mtp_cold_fills_unreusable_cache(monkeypatch):
    """A reused 5-entry (head-less) cache with offset>0 must COLD-FILL into
    a fresh 6-entry layout (fail-closed gate)."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_vlm.generate import PromptCacheState

    captured = {}

    def fake_step(prompt, model, head, **kwargs):
        captured.update(kwargs)
        captured["prompt_len"] = int(prompt.size)
        return iter(())

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    eng = _mtp_engine()
    old_cache = eng._language_model.make_cache()
    for c in old_cache:
        if hasattr(c, "offset"):
            c.offset = 3
    cache_state = PromptCacheState()
    cache_state.cache = old_cache
    cache_state.token_ids = [1, 2, 3]

    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=[1, 2, 3, 4, 5],
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
    )
    list(gen_iter)
    assert prompt_cache is not old_cache
    assert len(prompt_cache) == 6
    # Full prompt re-prefilled (no stale prefix slicing).
    assert captured["prompt_len"] == 5


def test_run_lm_legacy_still_rejects_vlm_drafter_kinds():
    from mlx_vlm.generate import PromptCacheState

    eng = _mtp_engine()
    eng._draft_kind = "mtp"  # gemma4_assistant kind — mlx-vlm only
    with pytest.raises(RuntimeError) as exc:
        eng._run_lm_legacy(
            cache_state=PromptCacheState(),
            prompt_token_ids=[1, 2, 3],
            max_tokens=8,
            sampler=None,
            logits_processors=None,
        )
    msg = str(exc.value)
    assert "--backend mlx-vlm" in msg
    assert "qwen3_5_mtp" in msg
