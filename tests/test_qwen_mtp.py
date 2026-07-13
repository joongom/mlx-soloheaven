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
  * finalize/resume — finalize forwards the final (stop) token through the
    TARGET (stored ids include it, matching the plain mlx-lm lookahead
    path) and leaves head_offset == target_offset - 1 with the last slot
    committed LAZILY at resume; append-only multi-turn reuse — including
    the natural-EOS + "\n"-template-glue production scenario — through
    validate_mtp_cache_reuse + the runner's lazy last-slot commit;
  * engine wiring — _run_lm_legacy dispatches qwen_mtp, appends head cache
    entries, keeps the mlx-vlm raise for other drafter kinds, and disables
    MTP for structured-output requests;
  * post-stream AHEAD reconcile (client-cancel path) — when the cache ends
    AHEAD of the recorded token_ids (cancel drops the in-flight frame but
    gen_iter.close() still runs the finalize), the reconcile must either
    trim back cleanly (verified per-layer) or INVALIDATE — a lying or
    throwing trim must never leave a partially-trimmed (desynced) cache
    live; the MTP finalize-hidden stash is cleared with the cache;
  * HIT-session early-return commit (C1) — a cancelled or empty
    (thinking-only) HIT turn must never leave session.messages desynced
    from the in-place-advanced session cache: partial output is committed
    as an assistant turn, cancel-before-any-token rolls the prefilled
    suffix back (or invalidates when untrimmable), and the next
    same-history request never splices user_N a second time. Amendments:
    (A1) the turn close is TRI-STATE (TurnCloseResult) — gemma4/glm or an
    already-terminated tail commit as-is (NOT_REQUIRED), a verified
    <|im_end|> forward commits (CLOSED), and a required-but-unavailable
    close (vlm path / head-bearing cache / no eot / non-callable target /
    forward failure / offset mismatch) INVALIDATES fail-closed (FAILED)
    instead of committing an unterminated ChatML turn; (A2) the
    empty-response commit verifies the recorded tail IS an end-of-turn
    token before assuming natural termination (max_tokens exhaustion must
    close or invalidate); (A3) closing the engine generator mid-stream
    (GeneratorExit — the client-disconnect teardown both production
    drivers produce) runs the same reconcile+commit via a rescue
    finalization instead of skipping the post-loop entirely.
"""

from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace
from typing import List

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_soloheaven.engine.qwen_mtp import (
    MTPCacheCorruption,
    QwenMTPHead,
    REUSE_COLD_FILL,
    REUSE_FALLBACK_PLAIN,
    REUSE_FALLBACK_STRIP_HEAD,
    REUSE_MTP,
    classify_cache,
    load_qwen_mtp_head,
    make_head_cache,
    plan_mtp_cache_reuse,
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
        # Every head forward as (hidden-position tokens, pair tokens) —
        # lets tests assert the lazy last-slot commit pairing exactly.
        self.head_calls: List[tuple] = []

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
        self.head_calls.append(
            ([int(round(v[0])) for v in hidden[0].tolist()], toks)
        )
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
    resume_hidden=None,
):
    cache = cache if cache is not None else _mk_caches()
    fired = {"corruption": 0, "pending": "UNSET", "hidden": "UNSET"}

    def on_corruption():
        fired["corruption"] += 1

    def on_finalize(tok, hid):
        fired["pending"] = tok
        fired["hidden"] = hid

    ops = MockOps(next_map, head_map)
    fired["ops"] = ops
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
        resume_hidden=resume_hidden,
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

    # ROLLBACK EXACTNESS: the ArraysCache history must be exactly suffix +
    # ALL emitted tokens (finalize forwards the final one through the
    # target, plain-path parity); any rejected-draft ghost token would
    # appear here.
    final_state = tuple(suffix + out)
    for i in (0, 1, 3):
        assert cache[i].cache[0] == final_state, (
            f"layer {i} recurrent history has ghost tokens:\n"
            f"  got      {cache[i].cache[0]}\n  expected {final_state}"
        )
    # KV offsets settled to exactly suffix + emitted (stop included).
    for i in (2, 4):
        assert cache[i].offset == len(final_state)
    # Finalize: pending pair committed, head ONE behind target (the lazy
    # last slot pairs with next turn's first token).
    assert cache[5].offset == len(final_state) - 1
    assert fired["pending"] == out[-1]
    assert fired["hidden"] is not None
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
    assert fired["hidden"] is not None
    final_state = tuple([5, 6] + out)
    assert cache[0].cache[0] == final_state
    assert cache[2].offset == len(final_state)
    assert cache[5].offset == len(final_state) - 1


def test_lying_cache_fires_fail_closed():
    """U6 contract: cache corruption TERMINATES the stream at the corruption
    point — it must NOT fall back to plain decoding from the unverifiable
    target cache (pre-U6 the stream stayed alive to max_tokens, generating
    every remaining token from possibly corrupted state)."""
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: 63  # always wrong -> first round must roll back
    cache = _mk_caches(kv_cls=LyingKV)
    cache[5] = MockKV()  # head cache stays honest
    out, flags, cache, fired = _run(
        [1, 2, 3], next_map, head_map, max_tokens=8, cache=cache
    )
    # Fail-closed fired: corruption callback + stream TERMINATED.
    assert fired["corruption"] >= 1
    # Only the tokens yielded BEFORE the corrupt round's settle survive
    # (bootstrap + the round's bonus token) — nothing generated after.
    assert len(out) == 2, f"stream must terminate at corruption, got {out}"
    # The corrupt round's drafts were all wrong — nothing accepted.
    assert not any(flags), "no draft should be accepted (head always wrong)"
    # Broken stream must NOT finalize (head desynced): no pending token,
    # no resume hidden -> next turn's gate cold-fills.
    assert fired["pending"] is None
    assert fired["hidden"] is None


def test_restore_is_reference_exact():
    """A rejected round must restore the EXACT pre-round ArraysCache state
    object contents (reference snapshot), then replay only consumed tokens."""
    next_map = lambda t: (t + 1) % 30
    head_map = lambda t: 29 - (t % 7)  # mostly wrong
    suffix = [4, 5]
    out, _flags, cache, _f = _run(suffix, next_map, head_map, max_tokens=6)
    assert out == _chain(5, 6, next_map)
    assert cache[0].cache[0] == tuple(suffix + out)


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
    mid-round yield) — finally must settle to exactly prompt + yielded - 1,
    then finalize: pending pair into the head + pending token through the
    TARGET (so every yielded token is in the cache), head == target - 1."""
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
        on_finalize=lambda tok, hid: fired.update(pending=tok, hidden=hid),
        ops=MockOps(next_map, next_map),
    )
    for t, _lp, _fd in gen:
        got.append(t)
        if len(got) == 3:  # break INSIDE a speculative round (1 bootstrap + 2)
            break
    gen.close()

    assert got == _chain(3, 3, next_map)
    final_state = tuple(suffix + got)
    for i in (0, 1, 3):
        assert cache[i].cache[0] == final_state, (
            f"layer {i}: early-close left ghost tokens: {cache[i].cache[0]} "
            f"!= {final_state}"
        )
    for i in (2, 4):
        assert cache[i].offset == len(final_state)
    assert cache[5].offset == len(final_state) - 1  # finalize ran, lazy slot open
    assert fired.get("pending") == got[-1]
    assert fired.get("hidden") is not None
    assert "corruption" not in fired


def test_multiturn_append_reuse_length_stop():
    """Turn 1 ends by max_tokens; turn 2 appends ANY continuation (no
    byte-match requirement). The gate accepts and the resume path commits
    the head's lazy last pair from the finalize hidden."""
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: (t + 1) % 50 if t % 3 else 63
    suffix1 = [1, 2, 3]
    out1, _fl, cache, fired = _run(
        suffix1, next_map, head_map, max_tokens=8
    )
    assert fired["pending"] == out1[-1]
    stored_ids = suffix1 + out1  # the engine records ALL yielded tokens
    assert cache[2].offset == len(stored_ids)
    assert cache[5].offset == len(stored_ids) - 1

    # Next turn: append-only prompt = stored + new user tokens.
    new_prompt = stored_ids + [20, 21]
    ok, why = validate_mtp_cache_reuse(
        cache, stored_ids, new_prompt, 5, 1, fired["hidden"]
    )
    assert ok, why

    suffix2 = new_prompt[len(stored_ids):]
    out2, _fl2, cache, fired2 = _run(
        suffix2, next_map, head_map, max_tokens=6, cache=cache,
        resume_hidden=fired["hidden"],
    )
    # The FIRST head call of turn 2 is the lazy last-slot commit:
    # pair (h_{N-1} == hidden of stored_ids[-1], first suffix token).
    assert fired2["ops"].head_calls[0] == ([stored_ids[-1]], [20])
    assert out2 == _chain(21, 6, next_map)
    final_state = tuple(stored_ids + suffix2 + out2)
    assert cache[0].cache[0] == final_state
    assert cache[2].offset == len(final_state)
    assert cache[5].offset == len(final_state) - 1
    assert fired2["pending"] == out2[-1]


def test_multiturn_natural_eos_append_reuse():
    """THE OBSERVED PRODUCTION SCENARIO (turn-2 permanent cold-fill bug):
    turn 1 ends at a natural stop token — the adapter breaks at the stop
    yield and closes the generator; the engine records ALL yielded tokens
    (stop included, the 248046-like id) into stored token_ids. Turn 2's
    prompt is stored_ids + chat-template glue ("\\n<|im_start|>user..." —
    starts with a 198-like "\\n" token, NEVER re-sends the stop). The
    finalized cache must be reusable: gate OK (no first-token byte-match),
    resume commits the head's lazy last pair (h_{N-1}, first_suffix_token),
    and turn 2's output is byte-exact."""
    STOP, NL = 7, 40
    next_map = lambda t: 11 if t == STOP else (t + 1) % 50
    suffix1 = [1, 2, 3]
    cache = _mk_caches()
    fired = {}
    got: List[int] = []
    ops1 = MockOps(next_map, next_map)

    gen = qwen_mtp_generate_step(
        mx.array(suffix1, mx.uint32),
        model=None,
        head=None,
        block_size=3,
        max_tokens=100,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: fired.__setitem__("corruption", True),
        on_finalize=lambda tok, hid: fired.update(pending=tok, hidden=hid),
        ops=ops1,
    )
    for t, _lp, _fd in gen:
        got.append(t)
        if t == STOP:
            break  # exactly what _pld_response_adapter does on EOS
    gen.close()

    assert got == [4, 5, 6, STOP]
    stored_ids = suffix1 + got  # engine stores ALL yielded incl. the stop
    # Finalize forwarded the stop through the TARGET (plain-path parity)...
    assert cache[2].offset == cache[4].offset == len(stored_ids)
    for i in (0, 1, 3):
        assert cache[i].cache[0] == tuple(stored_ids)
    # ...and left the head ONE behind: its last slot pairs with turn 2's
    # first token, unknowable at finalize time.
    assert cache[5].offset == len(stored_ids) - 1
    assert fired["pending"] == STOP
    assert fired["hidden"] is not None
    assert "corruption" not in fired

    # Turn 2: template glue then user tokens — the stop is NOT re-sent.
    new_prompt = stored_ids + [NL, 20, 21]
    ok, why = validate_mtp_cache_reuse(
        cache, stored_ids, new_prompt, 5, 1, fired["hidden"]
    )
    # Pre-fix this failed with "pending pair token 7 != first suffix token 40".
    assert ok, why

    suffix2 = new_prompt[len(stored_ids):]
    out2, _fl2, cache, fired2 = _run(
        suffix2, next_map, next_map, max_tokens=5, cache=cache,
        resume_hidden=fired["hidden"],
    )
    # The FIRST head call of turn 2 must be the lazy last-slot commit:
    # pair (h_{N-1} == hidden of STOP, first suffix token NL).
    assert fired2["ops"].head_calls[0] == ([STOP], [NL])
    assert out2 == _chain(21, 5, next_map)
    final_state = tuple(stored_ids + suffix2 + out2)
    assert cache[0].cache[0] == final_state
    assert cache[2].offset == len(final_state)
    assert cache[5].offset == len(final_state) - 1
    assert fired2["pending"] == out2[-1]


def test_mid_stream_abort_cold_fills():
    """A broken (fail-closed) stream never finalizes: on_finalize(None,
    None), head/target left desynced — next turn's gate must cold-fill."""
    next_map = lambda t: (t + 1) % 50
    head_map = lambda t: 63  # always wrong
    cache = _mk_caches(kv_cls=LyingKV)
    cache[5] = MockKV()
    out, _fl, cache, fired = _run(
        [1, 2, 3], next_map, head_map, max_tokens=8, cache=cache
    )
    assert fired["corruption"] >= 1
    assert fired["pending"] is None
    assert fired["hidden"] is None
    stored = [1, 2, 3] + out
    ok, why = validate_mtp_cache_reuse(
        cache, stored, stored + [30], 5, 1, fired["hidden"]
    )
    assert not ok


def _resumed_cache(stored_len=5, head_offset=None):
    """Finalized-resume cache shape: targets at stored_len, head one behind
    (or at an explicit bogus offset for the invariant-violation case)."""
    cache = _mk_caches()
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = stored_len
    cache[5].offset = stored_len - 1 if head_offset is None else head_offset
    for i in (0, 1, 3):
        cache[i].cache = [tuple(range(1, stored_len + 1))]
    return cache


def test_resume_validation_missing_hidden_terminates_not_raises():
    """F6: a resumed-shape cache (head == target - 1) WITHOUT resume_hidden
    used to raise MTPCacheCorruption OUT of the generator (the validation
    runs before the main try) — the engine saw an exception instead of the
    error terminal. Now it TERMINATES the (empty) stream and fires the
    invalidation exactly once."""
    next_map = lambda t: (t + 1) % 50
    cache = _resumed_cache()
    out, flags, cache, fired = _run(
        [9, 10], next_map, next_map, max_tokens=4, cache=cache,
        resume_hidden=None,
    )
    assert out == [] and flags == []          # no exception, zero frames
    assert fired["corruption"] == 1           # invalidation happens ONCE
    assert fired["pending"] == "UNSET"        # finalize never ran


def test_resume_entry_invariant_violation_terminates_not_raises():
    """F6: an entry-invariant violation (head neither fresh 0/0 nor
    target-1) terminates with one invalidation instead of raising."""
    next_map = lambda t: (t + 1) % 50
    cache = _resumed_cache(head_offset=3)  # head == target - 2: illegal
    out, flags, cache, fired = _run(
        [9, 10], next_map, next_map, max_tokens=4, cache=cache,
        resume_hidden=mx.array([[[5.0]]]),
    )
    assert out == [] and flags == []
    assert fired["corruption"] == 1
    assert fired["pending"] == "UNSET"


def test_resume_lazy_commit_shortfall_terminates_not_raises():
    """F6 (the codex-cited path): the lazy last-slot commit mutates the head
    but does not land on the target offset — pre-fix this fired the
    invalidation AND THEN raised out of the generator (double signalling,
    no error terminal). Now: terminate, exactly one invalidation."""
    next_map = lambda t: (t + 1) % 50

    class _StuckHeadOps(MockOps):
        """head_hidden never advances the head cache offsets — the lazy
        commit therefore cannot land on T0."""

        def head_hidden(self, hidden, next_ids, mtp_cache):
            toks = [int(t) for t in next_ids[0].tolist()]
            self.head_calls.append(
                ([int(round(v[0])) for v in hidden[0].tolist()], toks)
            )
            return self._hid(toks)

    cache = _resumed_cache()
    fired = {"corruption": 0, "pending": "UNSET"}
    gen = qwen_mtp_generate_step(
        mx.array([9, 10], mx.uint32),
        model=None,
        head=None,
        block_size=3,
        max_tokens=4,
        sampler=None,
        logits_processors=None,
        prompt_cache=cache,
        n_target_layers=5,
        on_cache_corruption=lambda: fired.__setitem__(
            "corruption", fired["corruption"] + 1
        ),
        on_finalize=lambda tok, hid: fired.__setitem__("pending", tok),
        resume_hidden=mx.array([[[5.0]]]),
        ops=_StuckHeadOps(next_map, next_map),
    )
    out = list(gen)  # must NOT raise
    assert out == []
    assert fired["corruption"] == 1
    assert fired["pending"] == "UNSET"


def test_validate_mtp_cache_reuse_fail_closed():
    next_map = lambda t: (t + 1) % 50
    out, _fl, cache, fired = _run([1, 2, 3], next_map, next_map, max_tokens=6)
    stored = [1, 2, 3] + out  # ALL yielded tokens are stored (stop included)
    hidden = fired["hidden"]
    good = stored + [30, 31]  # any continuation — no byte-match requirement

    ok, _ = validate_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert ok
    # layout: missing head entry
    ok, why = validate_mtp_cache_reuse(cache[:5], stored, good, 5, 1, hidden)
    assert not ok and "layout" in why
    # head == target: stale pre-fix shape (lazy slot filled with an unknown
    # pair) -> must cold-fill
    cache[5].offset += 1
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert not ok and "head offset" in why
    # head == target - 2: mid-stream shape (finalize never ran)
    cache[5].offset -= 2
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert not ok and "head offset" in why
    cache[5].offset += 1
    # divergence (ArraysCache untrimmable -> must cold-fill)
    diverged = stored[:-1] + [49, 30]
    ok, why = validate_mtp_cache_reuse(cache, stored, diverged, 5, 1, hidden)
    assert not ok and "divergence" in why
    # missing resume hidden (mid-stream abort / disk reload / stale stash)
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, None)
    assert not ok and "hidden" in why
    # no new suffix
    ok, why = validate_mtp_cache_reuse(cache, stored, list(stored), 5, 1, hidden)
    assert not ok
    # target offset != stored ids
    cache[2].offset += 1
    ok, why = validate_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert not ok
    cache[2].offset -= 1
    # fresh-shaped cache (just cold-filled) is trivially reusable
    fresh = _mk_caches()
    ok, _ = validate_mtp_cache_reuse(fresh, [], good, 5, 1, None)
    assert ok


def test_plan_mtp_cache_reuse_categories():
    """FEATURE A decision matrix: the plan wraps validate and maps each
    failure mode to MTP-reuse / plain-fallback / strip-head / cold-fill."""
    next_map = lambda t: (t + 1) % 50
    out, _fl, cache, fired = _run([1, 2, 3], next_map, next_map, max_tokens=6)
    stored = [1, 2, 3] + out
    hidden = fired["hidden"]
    good = stored + [30, 31]

    # Finalized 6-entry cache + hidden -> run MTP.
    action, why = plan_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert action == REUSE_MTP and why == ""

    # Head-less 5-entry layout, target pure-append-consistent -> plain
    # fallback keeping the cache.
    action, why = plan_mtp_cache_reuse(cache[:5], stored, good, 5, 1, hidden)
    assert action == REUSE_FALLBACK_PLAIN and "layout" in why

    # 6-entry layout, resume hidden missing -> strip head, plain fallback.
    action, why = plan_mtp_cache_reuse(cache, stored, good, 5, 1, None)
    assert action == REUSE_FALLBACK_STRIP_HEAD and "hidden" in why

    # 6-entry layout, head offset stale (== target) -> strip head.
    cache[5].offset += 1
    action, why = plan_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert action == REUSE_FALLBACK_STRIP_HEAD and "head offset" in why
    cache[5].offset -= 1

    # Divergence -> cold-fill (ArraysCache untrimmable), with or without head.
    diverged = stored[:-1] + [49, 30]
    action, why = plan_mtp_cache_reuse(cache, stored, diverged, 5, 1, hidden)
    assert action == REUSE_COLD_FILL and "divergence" in why
    action, _ = plan_mtp_cache_reuse(cache[:5], stored, diverged, 5, 1, hidden)
    assert action == REUSE_COLD_FILL

    # No new suffix tokens -> cold-fill (nothing for the runner to chew).
    action, _ = plan_mtp_cache_reuse(cache, stored, list(stored), 5, 1, hidden)
    assert action == REUSE_COLD_FILL

    # Target offsets desynced from stored ids -> cold-fill, never fallback.
    cache[2].offset += 1
    action, _ = plan_mtp_cache_reuse(cache, stored, good, 5, 1, hidden)
    assert action == REUSE_COLD_FILL
    action, _ = plan_mtp_cache_reuse(cache[:5], stored, good, 5, 1, hidden)
    assert action == REUSE_COLD_FILL
    cache[2].offset -= 1

    # Nothing cached (empty stored ids on a populated layout) -> cold-fill;
    # a fallback here would silently disable MTP on a free cold start.
    fresh5 = [MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV()]
    action, _ = plan_mtp_cache_reuse(fresh5, [], good, 5, 1, None)
    assert action == REUSE_COLD_FILL

    # Fresh-shaped 6-entry cache stays MTP-reusable (validate's fresh branch).
    fresh = _mk_caches()
    action, why = plan_mtp_cache_reuse(fresh, [], good, 5, 1, None)
    assert action == REUSE_MTP, why


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
    # Finalize forwarded the single emitted token through the target too
    # (suffix + 1); the head trails by the lazy last slot.
    assert cache[2].offset == 3
    assert cache[5].offset == 2
    assert fired["pending"] == 3
    assert fired["hidden"] is not None


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
    # Fresh cache: no lazy last-slot commit to perform.
    assert captured["resume_hidden"] is None


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


def test_run_lm_legacy_mtp_target_only_cache_plain_fallback(monkeypatch):
    """FEATURE A case 1: a reused 5-entry (head-less) pure-append cache —
    e.g. a plain base cache or a session written by a non-MTP server — must
    NOT cold-fill: MTP is disabled for the request, the cache is KEPT, and
    only the suffix is fed down the plain path."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
    from mlx_vlm.generate import PromptCacheState

    def boom(*a, **k):
        raise AssertionError("qwen_mtp_generate_step must not run on fallback")

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", boom)

    plain = {}

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        plain["prompt"] = list(prompt)
        plain.update(kwargs)
        return iter(())

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

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
    # The cache survived (no cold-fill, no head extend) and only the
    # 2-token suffix was fed to the plain runner.
    assert prompt_cache is old_cache
    assert cache_state.cache is old_cache
    assert len(prompt_cache) == 5
    assert plain["prompt"] == [4, 5]
    assert plain["prompt_cache"] is old_cache


def test_run_lm_legacy_mtp_divergence_still_cold_fills(monkeypatch):
    """FEATURE A case 3 (regression of today's fail-closed): a diverged
    prompt can never be rewound on the untrimmable hybrid — the gate must
    keep COLD-FILLING with MTP, never plain-fallback onto a diverged
    cache."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    captured = {}

    def fake_step(prompt, model, head, **kwargs):
        captured.update(kwargs)
        captured["prompt_len"] = int(prompt.size)
        return iter(())

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    eng = _mtp_engine()
    stored = [1, 2, 3, 4, 90]
    cache_state, cache = _finalized_cache_state(stored, mx.zeros((1, 1, 4)))

    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=stored[:-1] + [77, 50, 51],  # diverges at stored[-1]
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
    )
    list(gen_iter)
    assert prompt_cache is not cache  # cold-filled
    assert len(prompt_cache) == 6
    assert captured["prompt_len"] == 7  # full prompt re-prefilled
    assert captured["resume_hidden"] is None
    assert cache_state.cache is prompt_cache or cache_state.cache is None


def _finalized_cache_state(stored, hidden):
    """A PromptCacheState in the finalized MTP shape: 6-entry cache with
    target offsets == len(stored), head == len(stored) - 1, plus the
    finalize-hidden stash tagged with the cache offset."""
    from mlx_vlm.generate import PromptCacheState

    cache = [MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    cs = PromptCacheState()
    cs.cache = cache
    cs.token_ids = list(stored)
    cs.mtp_last_hidden = hidden
    cs.mtp_hidden_offset = len(stored)
    return cs, cache


def test_run_lm_legacy_mtp_reuses_finalized_eos_cache(monkeypatch):
    """ENGINE-LEVEL regression for the observed bug: stored token_ids end
    with the stop token (248046-like) and the next prompt's suffix starts
    with template glue (198-like "\\n") — the gate must PASS (no cold-fill),
    slice the prompt to the 3-token suffix, hand the runner the stashed
    finalize hidden, and consume the single-use stash."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    captured = {}

    def fake_step(prompt, model, head, **kwargs):
        captured.update(kwargs)
        captured["prompt_len"] = int(prompt.size)
        return iter(())

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    eng = _mtp_engine()
    STOP, NL = 90, 91  # stand-ins for 248046 (<|im_end|>) and 198 ("\n")
    stored = [1, 2, 3, 4, STOP]
    hidden = mx.zeros((1, 1, 4))
    cache_state, cache = _finalized_cache_state(stored, hidden)

    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=stored + [NL, 50, 51],
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
    )
    list(gen_iter)
    assert prompt_cache is cache  # NO cold-fill
    assert captured["prompt_len"] == 3  # only the new suffix prefilled
    assert captured["resume_hidden"] is hidden
    # Single-use stash consumed (a failed stream can't resurrect it).
    assert cache_state.mtp_last_hidden is None
    assert cache_state.mtp_hidden_offset is None


def test_run_lm_legacy_mtp_stale_hidden_strips_head_plain_fallback(monkeypatch):
    """FEATURE A case 2: a finalize-hidden stash whose offset tag no longer
    matches the cache (e.g. an intervening non-MTP request advanced the
    session) is dropped — but the target side is pure-append-consistent, so
    instead of cold-filling the head entry is STRIPPED and the request runs
    plain, reusing the target cache. The session continues as a plain
    5-entry session."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    def boom(*a, **k):
        raise AssertionError("qwen_mtp_generate_step must not run on fallback")

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", boom)

    plain = {}

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        plain["prompt"] = list(prompt)
        plain.update(kwargs)
        return iter(())

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

    eng = _mtp_engine()
    stored = [1, 2, 3, 4, 90]
    cache_state, cache = _finalized_cache_state(stored, mx.zeros((1, 1, 4)))
    cache_state.mtp_hidden_offset = len(stored) - 2  # stale tag

    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=stored + [91, 50, 51],
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
    )
    list(gen_iter)
    # Head stripped IN PLACE (cache_state.cache aliases the list), target
    # side reused: only the 3-token suffix goes to the plain runner.
    assert prompt_cache is cache
    assert cache_state.cache is cache
    assert len(prompt_cache) == 5
    assert plain["prompt"] == [91, 50, 51]
    assert plain["prompt_cache"] is cache
    # Single-use stash consumed; nothing MTP survives the request.
    assert cache_state.mtp_last_hidden is None
    assert cache_state.mtp_hidden_offset is None


def test_run_lm_legacy_pre_gate_disable_strips_head_and_stash(monkeypatch):
    """FINDING-1 invariant: a per-request MTP disable that fires BEFORE the
    reuse gate (response_format json_object -> use_mtp=False at the FSM
    check) skips the gate entirely — the 6-entry MTP-finalized session
    cache must STILL be stripped to the 5-entry target layout IN PLACE
    before plain generation, the single-use finalize-hidden stash cleared,
    and accounting stay consistent (prefix-trim sees a target-only layout,
    only the suffix reaches the plain runner)."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

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
    stored = [1, 2, 3, 4, 90]
    cache_state, cache = _finalized_cache_state(stored, mx.zeros((1, 1, 4)))

    gen_iter, prompt_cache = eng._run_lm_legacy(
        cache_state=cache_state,
        prompt_token_ids=stored + [91, 50, 51],
        max_tokens=8,
        sampler=lambda lp: mx.argmax(lp, axis=-1),
        logits_processors=None,
        response_format=SimpleNamespace(type="json_object"),
    )
    list(gen_iter)
    # Head stripped IN PLACE before generation; the cache itself is KEPT
    # (no cold-fill) and cache_state.cache still aliases the same list.
    assert prompt_cache is cache
    assert cache_state.cache is cache
    assert len(prompt_cache) == 5
    # Accounting consistent: only the 3-token suffix went to the plain
    # runner, on the kept (stripped) cache.
    assert plain["prompt"] == [91, 50, 51]
    assert plain["prompt_cache"] is cache
    # The stale single-use stash must not survive a plain turn it didn't
    # belong to.
    assert cache_state.mtp_last_hidden is None
    assert cache_state.mtp_hidden_offset is None


def test_response_adapter_always_emits_terminal_eos_frame():
    """The adapter must mirror mlx-lm's stream_generate: the terminal frame
    carries the EOS token id even with no buffered text left — the engine's
    post-loop records it into token_ids (the MTP finalize forwards the stop
    through the target, so stored ids must include it for offset ==
    len(token_ids) to hold)."""
    from mlx_soloheaven.engine.mlx_engine import _pld_response_adapter

    tok = SimpleNamespace(eos_token_ids=[9], decode=lambda ids: "x")
    frames = list(
        _pld_response_adapter(
            iter([(5, None, False), (9, None, False)]), tok, label="T"
        )
    )
    assert [f.token for f in frames] == [5, 9]
    assert frames[-1].text == ""


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


# ---------------------------------------------------------------------------
# Post-stream AHEAD reconcile (client-cancel path) — fail-closed trim-back
# ---------------------------------------------------------------------------


class ThrowingKV(MockKV):
    """trim() raises mid-reconcile — earlier layers in the list have already
    been rewound, so failing open here leaves a PARTIALLY-trimmed cache."""

    def trim(self, n):
        raise RuntimeError("trim exploded")


def _generate_engine(cache, stored, *, mtp, suffix):
    """An MLXEngine stub that can drive ``_generate_locked`` end-to-end on
    the mlx-lm path with a PRE-SEEDED cache-hit session. The session's
    cache_state object is mutated in place by the post-loop reconcile, so a
    cancelled stream's outcome (clean trim vs invalidation) is observable
    directly on it; since the C1 fix, cancellation additionally COMMITS the
    session (messages + total_cache_tokens) instead of skipping the save.
    No real model."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine, SessionState
    from mlx_vlm.generate import PromptCacheState

    eng = _mtp_engine()
    if not mtp:
        eng._drafter = None
        eng._draft_kind = None
    eng.cfg.enable_thinking = False
    eng.cfg.think_end_token = -1
    eng.cfg.pld_enabled = False
    eng.cfg.kv_bits = 0
    eng._sessions = {}
    eng._touch_gpu = lambda: None
    eng._has_disk_cache = lambda sid: False
    eng._find_base_cache = lambda messages, tools=None: None
    eng._maybe_register_base_cache = lambda *a, **k: None
    eng._has_rotating_cache = False
    eng._sliding_window_size = 0
    eng.model_family = "chatml"
    # decode must be non-empty so the response adapter emits one frame per
    # token (empty text would be buffered as a partial-UTF8 segment).
    eng.tokenizer = SimpleNamespace(decode=lambda ids: "x", eos_token_ids=[])
    eng._messages_match = lambda stored_msgs, incoming: True
    eng._suffix_tokens = lambda msgs, thinking=True: list(suffix)

    cs = PromptCacheState()
    cs.cache = cache
    cs.token_ids = list(stored)
    sess_messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    eng._sessions["s"] = SessionState(
        cache_state=cs,
        messages=sess_messages,
        total_cache_tokens=len(stored),
        # U21/F5: stamp the contract the drives use (tools=None,
        # thinking=False) — a legacy fp=None session now always takes a
        # cold rebuild (never a lenient HIT), so a HIT-path harness must
        # look like a session the new engine built.
        tools=None,
        thinking=False,
        prompt_fingerprint=MLXEngine._prompt_fingerprint(None, False),
    )
    messages = sess_messages + [{"role": "user", "content": "u2"}]
    return eng, cs, messages


def _restamp_contract(eng, *, thinking, tools=None, sid="s"):
    """Re-stamp the harness session's prompt contract for drives that use a
    non-default thinking flag / tools (U21: the HIT gate refuses a
    fingerprint mismatch, and F5 refuses fp=None outright)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    s = eng._sessions[sid]
    s.tools = tools
    s.thinking = thinking
    s.prompt_fingerprint = MLXEngine._prompt_fingerprint(tools, thinking)


def _drive_cancelled(eng, messages, *, cancel_after=2, max_tokens=8):
    """Consume ``_generate_locked`` and set cancel_event after
    ``cancel_after`` yielded tokens. The engine's stream loop checks the
    event AFTER pulling the next frame from gen_iter and BEFORE recording
    it — exactly the 'yielded but never recorded' client-cancel gap."""
    cancel = threading.Event()
    chunks = []
    for ch in eng._generate_locked(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        session_id="s",
        tools=None,
        cancel_event=cancel,
        thinking=False,
        thinking_budget=0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        repetition_penalty=1.0,
        response_format=None,
    ):
        chunks.append(ch)
        if len(chunks) == cancel_after:
            cancel.set()
    return chunks


def _assert_no_desynced_cache(cache_state, *, mtp_head_trailing=False):
    """The reconcile contract: the surviving cache (if any) has EVERY
    offset-bearing layer exactly at len(token_ids); otherwise everything —
    cache, token_ids, MTP finalize-hidden stash — was invalidated.

    ``mtp_head_trailing=True`` encodes the FINALIZED QwenMTP healthy-end
    layout (f372f2b lazy last-pair commit): the LAST cache entry is the MTP
    head whose offset trails the target by EXACTLY one — its lazy last-slot
    pair commits at next-turn resume. This is the state the engine reconcile
    deliberately tolerates (``_get_cache_offset`` reads the first
    offset-bearing TARGET layer, never the head) and the state
    ``validate_mtp_cache_reuse`` REQUIRES (``head == target - 1``)."""
    if cache_state.cache is None:
        assert cache_state.token_ids is None
        assert getattr(cache_state, "mtp_last_hidden", None) is None
        assert getattr(cache_state, "mtp_hidden_offset", None) is None
    else:
        n = len(cache_state.token_ids)
        layers = list(cache_state.cache)
        if mtp_head_trailing:
            head = layers.pop()
            assert getattr(head, "offset", None) == n - 1, (
                f"MTP head offset {getattr(head, 'offset', None)} != "
                f"token_ids len {n} - 1 (lazy last slot)"
            )
        offs = [c.offset for c in layers if hasattr(c, "offset")]
        assert offs and all(o == n for o in offs), (
            f"desynced cache survived: offsets {offs} != token_ids len {n}"
        )


def test_generate_locked_client_cancel_mtp_invalidates_desynced_cache(
    monkeypatch,
):
    """ENGINE-LEVEL client-cancel on the QwenMTP path: cancel_event trips
    between a yielded frame and its recording, then gen_iter.close() runs
    the runner finalize — which forwards the pending token through the
    TARGET — so the cache lands AHEAD of the recorded token_ids. The hybrid
    cache (untrimmable ArraysCache layers) can never be rewound: the
    post-loop reconcile must INVALIDATE the session cache AND clear the
    finalize-hidden stash (which on_finalize re-set during close). No
    desynced cache survives."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    STOP = 7
    next_map = lambda t: (t + 1) % 50
    stored = [1, 2, 3, 4, STOP]
    suffix = [40, 20, 21]  # template glue + user tokens

    # Finalized hybrid cache: targets at len(stored), head one behind.
    cache = [
        MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    cs.mtp_last_hidden = mx.array([[[float(STOP)]]])
    cs.mtp_hidden_offset = len(stored)

    runner_yielded: List[int] = []

    def fake_step(prompt, model, head, **kwargs):
        inner = qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

        def spy():
            try:
                for item in inner:
                    runner_yielded.append(int(item[0]))
                    yield item
            finally:
                inner.close()

        return spy()

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    chunks = _drive_cancelled(eng, messages, cancel_after=2)

    # The cancel dropped exactly the in-flight frame: the runner yielded one
    # more token than the engine recorded.
    assert len(chunks) == 2
    assert len(runner_yielded) == 3
    assert [c.token for c in chunks] == runner_yielded[:2]
    # Reconcile invalidated everything (pre-fix the untrimmable branch
    # nulled the cache but LEAKED the stash that on_finalize set at close).
    assert cs.cache is None
    assert cs.token_ids is None
    assert cs.mtp_last_hidden is None
    assert cs.mtp_hidden_offset is None
    _assert_no_desynced_cache(cs)


def test_generate_locked_cancel_lying_trim_invalidates(monkeypatch):
    """AHEAD reconcile, trimmable branch, LYING trim: every layer's trim()
    returns without raising but never rewinds. Pre-fix this failed OPEN —
    the desynced cache stayed live while token_ids was already reconciled
    to the shorter history. The per-layer post-trim verification must
    INVALIDATE."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    stored = [1, 2, 3, 4, 5]
    suffix = [40, 20, 21]
    cache = [LyingKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:  # prefill the suffix
                c.offset += len(prompt)
            for tid in (101, 102, 103, 104):
                for c in prompt_cache:  # one decode step per token
                    c.offset += 1
                yield SimpleNamespace(
                    text="x", token=tid, prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2  # third frame pulled but dropped -> cache AHEAD
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)


def test_generate_locked_cancel_throwing_trim_invalidates(monkeypatch):
    """AHEAD reconcile, trimmable branch, THROWING trim: one mid-list layer
    raises after its neighbours already rewound — a partially-trimmed cache.
    Pre-fix the exception was only logged and the desynced cache survived;
    now ANY trim exception invalidates."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    stored = [1, 2, 3, 4, 5]
    suffix = [40, 20, 21]
    cache = [MockKV(), ThrowingKV(), MockKV(), MockKV(), MockKV()]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                c.offset += len(prompt)
            for tid in (101, 102, 103, 104):
                for c in prompt_cache:
                    c.offset += 1
                yield SimpleNamespace(
                    text="x", token=tid, prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)


def test_generate_locked_cancel_honest_trim_survives_verified(monkeypatch):
    """Positive control: with honest trimmable layers the cancel-path
    reconcile trims the cache back to EXACTLY len(token_ids) (verified
    per-layer) and the cache survives — the fail-closed guard must not
    degrade the legitimate trim-back into a blanket cold-fill. Since the
    tri-state amendment the C1 commit additionally template-CLOSES the
    interrupted ChatML turn (this engine is close-capable), so the eot id
    trails the recorded history."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    EOT = 77
    stored = [1, 2, 3, 4, 5]
    suffix = [40, 20, 21]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": EOT},
    )

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                c.offset += len(prompt)
            for tid in (101, 102, 103, 104):
                for c in prompt_cache:
                    c.offset += 1
                yield SimpleNamespace(
                    text="x", token=tid, prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2
    # Cache survived: trimmed back to exactly the recorded history, then
    # template-closed (eot forwarded + recorded) by the C1 commit.
    assert cs.cache is cache
    assert cs.token_ids == stored + suffix + [101, 102, EOT]
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)


# ---------------------------------------------------------------------------
# Server-path accounting leak: buffered (empty-segment) tokens must still be
# yielded to the engine — committed-but-unrecorded tokens otherwise leave the
# cache AHEAD of token_ids at reconcile and the hybrid cache INVALIDATES on
# every healthy stream (production: "cache ahead of recorded ids by N").
# ---------------------------------------------------------------------------


def test_response_adapter_yields_frame_for_buffered_empty_segments():
    """THE PRODUCTION LEAK SHAPE: tokens whose detok segment is empty (BPE-
    held spaces like id 220, partial UTF-8 multi-byte sequences) used to be
    swallowed by the adapter (`continue`), so the runner committed them into
    the target cache but the engine never recorded them into token_ids. The
    adapter must yield EXACTLY one frame per pulled token — mlx-lm
    stream_generate parity — empty text included."""
    from mlx_soloheaven.engine.mlx_engine import _pld_response_adapter

    SPACE = 220
    tok = SimpleNamespace(
        eos_token_ids=[9],
        decode=lambda ids: "" if ids == [SPACE] else "x",
    )
    runner = iter(
        [
            (5, None, False),
            (SPACE, None, True),   # buffered: empty segment, MUST still frame
            (SPACE, None, False),
            (6, None, True),
            (9, None, False),      # EOS — terminal frame
        ]
    )
    frames = list(_pld_response_adapter(runner, tok, label="T"))
    # 1:1 frame-per-token — the engine records every committed token id.
    assert [f.token for f in frames] == [5, SPACE, SPACE, 6, 9]
    assert [f.text for f in frames] == ["x", "", "", "x", ""]
    assert [f.from_draft for f in frames] == [False, True, False, True, False]


def test_generate_locked_healthy_eos_midround_exact_accounting(monkeypatch):
    """ENGINE-LEVEL regression for the server-only accounting leak: a
    HEALTHY stream (natural EOS mid-round, no cancel) whose output contains
    buffered empty-segment tokens. Pre-fix the adapter dropped those frames:
    committed > recorded -> cache AHEAD at reconcile -> untrimmable
    (ArraysCache) branch INVALIDATED the session cache on EVERY turn (the
    user's 'reuse never happens' complaint). Post-fix the accounting is
    EXACT: offset == len(token_ids), no invalidation, the finalize-hidden
    stash survives, and the next append-only turn passes the MTP reuse
    gate."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    STOP, SPACE = 7, 23  # SPACE: mid-stream token whose decode is ""
    chain = {21: 22, 22: SPACE, SPACE: STOP}
    next_map = lambda t: chain.get(t, (t + 1) % 50)
    stored = [1, 2, 3, 4, 90]
    suffix = [40, 20, 21]

    cache = [
        MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    # EOS detection + one empty-segment (buffered) token mid-stream.
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "" if ids == [SPACE] else "x",
        eos_token_ids=[STOP],
    )
    cs.mtp_last_hidden = mx.array([[[90.0]]])
    cs.mtp_hidden_offset = len(stored)

    def fake_step(prompt, model, head, **kwargs):
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    chunks = list(
        eng._generate_locked(
            messages,
            max_tokens=16,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )

    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [22, SPACE, STOP]
    # The buffered token's text is empty, but its ID was recorded.
    assert [c.text for c in token_frames] == ["x", "", ""]
    assert chunks[-1].finish_reason == "stop"

    # EXACT accounting: the cache SURVIVED reconcile (no invalidation) and
    # every offset-bearing layer sits at len(token_ids); the recurrent
    # (ArraysCache) history equals the recorded ids exactly (no ghost
    # tokens from the rolled-back post-EOS round tail).
    expected_ids = stored + suffix + [22, SPACE, STOP]
    assert cs.cache is cache
    assert cs.token_ids == expected_ids
    for i in (2, 4):
        assert cache[i].offset == len(expected_ids)
    for i in (0, 1, 3):
        assert cache[i].cache[0] == tuple(expected_ids)
    assert cache[5].offset == len(expected_ids) - 1  # lazy last slot
    _assert_no_desynced_cache(cs, mtp_head_trailing=True)

    # The finalize-hidden stash survived (offset-tagged to the live cache)
    # and the next append-only turn passes the MTP reuse gate -> HIT.
    assert cs.mtp_last_hidden is not None
    assert cs.mtp_hidden_offset == len(expected_ids)
    ok, why = validate_mtp_cache_reuse(
        cache, cs.token_ids, expected_ids + [40, 50, 51], 5, 1,
        cs.mtp_last_hidden,
    )
    assert ok, why


def test_generate_locked_plain_fallback_keeps_session_reusable(monkeypatch):
    """FEATURE A end-to-end: a cache-HIT session carrying a head-less
    5-entry hybrid cache on an MTP server takes the plain-decode fallback —
    the runner never fires, the cache is reused (suffix-only prefill) — and
    the post-loop leaves offsets == len(token_ids) so the NEXT turn falls
    back again (still reusable, never a cold-fill cascade)."""
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    def boom(*a, **k):
        raise AssertionError("qwen_mtp_generate_step must not run on fallback")

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", boom)

    stored = [1, 2, 3, 4, 5]
    suffix = [40, 20, 21]
    cache = [MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]
    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompt_cache = kwargs["prompt_cache"]
        assert prompt_cache is cache  # reuse, no cold-fill
        assert list(prompt) == suffix  # suffix-only prefill

        def gen():
            # Plain mlx-lm contract: prefill + per-token lookahead advance.
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset += len(prompt)
            for tid in (101, 102, 103):
                for c in prompt_cache:
                    if hasattr(c, "offset"):
                        c.offset += 1
                yield SimpleNamespace(
                    text="x", token=tid, prompt_tps=1.0, generation_tps=1.0,
                )

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)

    chunks = list(
        eng._generate_locked(
            messages,
            max_tokens=3,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )

    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [101, 102, 103]

    # The session survived reconcile: cache kept, offsets == len(token_ids).
    expected_ids = stored + suffix + [101, 102, 103]
    assert cs.cache is cache
    assert cs.token_ids == expected_ids
    for i in (2, 4):
        assert cache[i].offset == len(expected_ids)
    _assert_no_desynced_cache(cs)

    # Next turn: the gate falls back to plain AGAIN (reuse, not cold-fill).
    action, _why = plan_mtp_cache_reuse(
        cache, cs.token_ids, expected_ids + [40, 50], 5, 1, None
    )
    assert action == REUSE_FALLBACK_PLAIN


def test_generate_locked_healthy_length_stop_exact_accounting(monkeypatch):
    """Length-stop variant of the leak shape: no EOS — the runner stops at
    max_tokens with buffered empty-segment tokens in the output. Accounting
    must be exact (offset == len(token_ids)), no invalidation."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    SPACE = 23  # decode("") -> buffered/empty segment
    next_map = lambda t: (t + 1) % 50
    stored = [1, 2, 3, 4, 90]
    suffix = [40, 20, 21]

    cache = [
        MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "" if ids == [SPACE] else "x",
        eos_token_ids=[],
    )
    cs.mtp_last_hidden = mx.array([[[90.0]]])
    cs.mtp_hidden_offset = len(stored)

    def fake_step(prompt, model, head, **kwargs):
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)

    chunks = list(
        eng._generate_locked(
            messages,
            max_tokens=4,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )

    token_frames = [c for c in chunks if c.finish_reason is None]
    assert [c.token for c in token_frames] == [22, SPACE, 24, 25]

    expected_ids = stored + suffix + [22, SPACE, 24, 25]
    assert cs.cache is cache
    assert cs.token_ids == expected_ids
    for i in (2, 4):
        assert cache[i].offset == len(expected_ids)
    for i in (0, 1, 3):
        assert cache[i].cache[0] == tuple(expected_ids)
    assert cache[5].offset == len(expected_ids) - 1
    _assert_no_desynced_cache(cs, mtp_head_trailing=True)
    ok, why = validate_mtp_cache_reuse(
        cache, cs.token_ids, expected_ids + [40, 50, 51], 5, 1,
        cs.mtp_last_hidden,
    )
    assert ok, why


# ---------------------------------------------------------------------------
# C1: HIT-session early-return cache pollution — cancel / empty-response
# turns must COMMIT (or roll back) so session.messages never desyncs from the
# in-place-advanced session cache, and the next same-history request never
# splices user_N a second time after a dangling partial output.
# ---------------------------------------------------------------------------


def _use_real_messages_match(eng):
    """Swap the helper's always-True matcher for the REAL prefix matcher so
    the committed interrupted turn has to survive the production leniency
    rules (last-stored-assistant content tolerance etc.)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng._messages_match = MLXEngine._messages_match.__get__(eng)


def _content_token_suffix(eng):
    """Suffix stub that encodes each non-assistant message's content ("uN")
    as one distinctive token (50+N) plus the assistant-header token 99 — a
    re-spliced user message is then visible as its token appearing twice in
    the prompt handed to the runner."""

    def fake_suffix(msgs, thinking=True):
        out = []
        for m in msgs:
            if m.get("role") != "assistant":
                out.append(50 + int(str(m.get("content"))[1:]))
        out.append(99)
        return out

    eng._suffix_tokens = fake_suffix


def _scripted_lm_stream(monkeypatch, prompts_seen, scripts, *, advance="pre"):
    """Monkeypatch lm_stream_generate with per-call token scripts.

    ``advance="pre"``: each layer's offset advances BEFORE the frame is
    yielded (a dropped in-flight frame leaves the cache AHEAD — the shape
    the reconcile trim-back handles). ``advance="post"``: mlx-lm lookahead
    semantics — at yield of y_i the cache holds exactly prompt+i tokens, so
    a cancel on the FIRST frame leaves the cache exactly at the prompt.
    """
    from mlx_soloheaven.engine import mlx_engine as mlx_engine_module

    calls = {"n": 0}

    def fake_lm_stream(model, tokenizer, prompt, **kwargs):
        prompts_seen.append(list(prompt))
        script = scripts[min(calls["n"], len(scripts) - 1)]
        calls["n"] += 1
        prompt_cache = kwargs["prompt_cache"]

        def gen():
            for c in prompt_cache:
                if hasattr(c, "offset"):
                    c.offset += len(prompt)
            for tid, text in script:
                if advance == "pre":
                    for c in prompt_cache:
                        if hasattr(c, "offset"):
                            c.offset += 1
                yield SimpleNamespace(
                    text=text, token=tid, prompt_tps=1.0, generation_tps=1.0,
                )
                if advance == "post":
                    for c in prompt_cache:
                        if hasattr(c, "offset"):
                            c.offset += 1

        return gen()

    monkeypatch.setattr(mlx_engine_module, "lm_stream_generate", fake_lm_stream)


def _drive_full(eng, messages, *, max_tokens=8, thinking=False):
    """Drive _generate_locked to natural completion (no cancel)."""
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=None,
            thinking=thinking,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )


def _drive_pre_cancelled(eng, messages, *, max_tokens=8):
    """Cancel BEFORE any token is recorded: the event is set up front, so the
    stream loop pulls exactly one frame, sees the cancel, and drops it."""
    cancel = threading.Event()
    cancel.set()
    return list(
        eng._generate_locked(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            session_id="s",
            tools=None,
            cancel_event=cancel,
            thinking=False,
            thinking_budget=0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            repetition_penalty=1.0,
            response_format=None,
        )
    )


def test_generate_locked_cancel_commits_turn_no_double_inject(monkeypatch):
    """C1 (a): cancel mid-generation on a HIT session. The partial turn must
    be COMMITTED (messages + total_cache_tokens consistent with token_ids),
    and the NEXT same-history request must extend with user_{N+1} ONLY —
    pre-fix, the stale session.messages matched the resent history and
    user_N's suffix tokens were spliced a SECOND time after the dangling
    partial output.

    Tri-state amendment (A1): a ChatML commit is only allowed when the turn
    is (or can be made) template-valid — this engine is close-CAPABLE
    (callable target + <|im_end|> in vocab), so the close reconcile returns
    CLOSED: the eot id lands in token_ids and the commit proceeds."""
    EOT = 77
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": EOT},
    )

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(101, "x"), (102, "x"), (103, "x"), (104, "x")],
            [(201, "y"), (202, "y")],
        ],
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2

    # Turn 1 prompt: stored history + suffix([u2]) = [52, 99].
    assert prompts_seen[0] == [52, 99]

    # Committed session: messages ↔ token_ids consistent, turn CLOSED (eot).
    sess = eng._sessions["s"]
    assert sess is not orig_sess  # committed via a fresh SessionState
    assert sess.cache_state is cs  # alias preserved
    assert cs.token_ids == stored + [52, 99] + [101, 102, EOT]
    assert [m.get("role") for m in sess.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert sess.messages[-1]["content"] == "xx"  # partial output, no thinking
    assert sess.total_cache_tokens == len(cs.token_ids)
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)

    # Turn 2: client resends its (truncated) view of the partial assistant
    # turn + a new user message. _messages_match's last-assistant leniency
    # accepts the content difference -> HIT -> suffix for u3 ONLY.
    messages2 = list(messages) + [
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "u3"},
    ]
    chunks2 = _drive_full(eng, messages2, max_tokens=2)
    assert chunks2[-1].finish_reason == "stop"
    # THE BUG SHAPE: pre-fix this prompt was [52, 53, 99] (u2 re-spliced).
    assert prompts_seen[1] == [53, 99]
    assert cs.token_ids == (
        stored + [52, 99] + [101, 102, EOT] + [53, 99] + [201, 202]
    )
    _assert_no_desynced_cache(cs)
    sess2 = eng._sessions["s"]
    assert [m.get("content") for m in sess2.messages] == [
        "u1", "a1", "u2", "xx", "u3", "yy",
    ]
    assert sess2.total_cache_tokens == len(cs.token_ids)


class _CallableTargetModel:
    """Callable stand-in for the mlx-lm target: forwarding T tokens advances
    every offset-bearing cache layer by T (what the C1 turn-close forward
    relies on)."""

    def __init__(self, n_layers=5):
        self.layers = [object()] * n_layers

    def __call__(self, toks, cache=None):
        n = int(toks.shape[1])
        for c in cache or []:
            if hasattr(c, "offset"):
                c.offset += n

    def make_cache(self):
        return [MockKV() for _ in range(5)]


def test_generate_locked_cancel_commit_template_closes_chatml(monkeypatch):
    """C1 principle 2: on the chatml/mlx-lm path with a target-only cache,
    the cancel commit template-closes the interrupted assistant turn by
    forwarding <|im_end|> through the target (1-token forward) — token_ids
    gains the eot id, every layer lands on len(token_ids), and the cached
    sequence is template-valid for the next turn."""
    EOT = 77
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": EOT},
    )

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(101, "x"), (102, "x"), (103, "x"), (104, "x")],
            [(201, "y")],
        ],
    )

    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2

    # Turn closed: eot forwarded through the target and recorded.
    assert cs.token_ids == stored + [52, 99] + [101, 102, EOT]
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)
    sess = eng._sessions["s"]
    assert sess.messages[-1] == {
        "role": "assistant",
        "content": "xx",
        "interrupted": True,  # U1: C1-committed turns carry the marker
    }
    assert sess.total_cache_tokens == len(cs.token_ids)

    # Next turn extends the template-valid sequence with u3's suffix only.
    messages2 = list(messages) + [
        {"role": "assistant", "content": "xx"},
        {"role": "user", "content": "u3"},
    ]
    _drive_full(eng, messages2, max_tokens=1)
    assert prompts_seen[1] == [53, 99]


def test_generate_locked_empty_response_hit_commits_consistent(monkeypatch):
    """C1 (b): a thinking-only (no content) HIT turn used to 'SKIP SAVE' —
    but the session's live cache had already been advanced in place. The
    guard must COMMIT the thinking-only assistant turn (same shape the
    normal save stores: <think>\\n prefix + accumulated text) so the next
    request extends honestly instead of double-splicing user_N.

    A2 amendment: the commit no longer ASSUMES natural termination — it
    verifies the recorded tail against the end-of-turn ids. Here the stream
    DID end naturally (the terminal frame's token 302 is a tokenizer EOS id,
    recorded + forwarded like mlx-lm's real terminal frame), so the close
    reconcile returns NOT_REQUIRED and the turn commits as-is (no eot
    injection)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    _restamp_contract(eng, thinking=True)  # this test drives thinking=True
    # The terminal frame's token IS an end-of-turn token: natural EOS shape.
    eng.tokenizer = SimpleNamespace(decode=lambda ids: "x", eos_token_ids=[302])

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(301, "pondering"), (302, "</think>")],
            [(401, "z")],
        ],
    )

    chunks = _drive_full(eng, messages, max_tokens=8, thinking=True)
    # The guard still yields the terminal stop frame to the client.
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].text == ""

    sess = eng._sessions["s"]
    assert sess.messages[-1] == {
        "role": "assistant",
        "content": "<think>\npondering</think>",
        "interrupted": True,  # U1: C1-committed turns carry the marker
    }
    assert cs.token_ids == stored + [52, 99] + [301, 302]
    assert sess.total_cache_tokens == len(cs.token_ids)
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)

    # Next turn: client's assistant view is empty content (it never received
    # any) — normalization strips the stored thinking, both sides match, and
    # only u3's suffix is spliced (pre-fix: [52, 53, 99] double-inject).
    messages2 = list(messages) + [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "u3"},
    ]
    _drive_full(eng, messages2, max_tokens=1, thinking=True)
    assert prompts_seen[1] == [53, 99]
    _assert_no_desynced_cache(cs)


def test_generate_locked_cancel_before_any_token_rolls_suffix_back(monkeypatch):
    """C1 principle 3 (trimmable): cancel before ANY generated token was
    recorded. Committing an empty assistant turn would be wrong — the
    prefilled suffix is ROLLED BACK (verified per-layer) to the pre-turn
    history, and session.messages (unchanged) exactly describes the result.
    The next request re-splices u2 fresh, exactly once."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)

    prompts_seen: list = []
    # advance="post": mlx-lm lookahead — at the first yield the cache holds
    # exactly the prompt, so the dropped frame leaves offset == prompt_len.
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(101, "x"), (102, "x"), (103, "x")],
            [(201, "y")],
        ],
        advance="post",
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_pre_cancelled(eng, messages)
    assert chunks == []  # nothing reached the client

    # Suffix rolled back: cache and token_ids at the pre-turn history,
    # session object untouched (messages already accurate).
    assert eng._sessions["s"] is orig_sess
    assert cs.token_ids == stored
    for c in cache:
        assert c.offset == len(stored)
    assert orig_sess.total_cache_tokens == len(stored)
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]
    _assert_no_desynced_cache(cs)

    # Next request (same history): HIT again, u2 spliced exactly once.
    _drive_full(eng, messages, max_tokens=1)
    assert prompts_seen[1] == [52, 99]
    assert cs.token_ids == stored + [52, 99] + [201]
    _assert_no_desynced_cache(cs)


def test_generate_locked_cancel_before_any_token_hybrid_invalidates(monkeypatch):
    """C1 principle 3 (untrimmable): same cancel-before-any-token shape on a
    hybrid cache (ArraysCache recurrent layers cannot trim) — the suffix
    cannot be rolled back, so the session cache is INVALIDATED (fail-closed;
    stale messages are then harmless) and the next request honestly MISSes
    into a full cold-fill instead of splicing after a dangling suffix."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV()]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(101, "x"), (102, "x")],
            [(201, "y")],
        ],
        advance="post",
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_pre_cancelled(eng, messages)
    assert chunks == []

    # Fail-closed: cache + token_ids + stash gone, messages untouched.
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)
    assert eng._sessions["s"] is orig_sess
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]

    # Next request: no live cache -> MISS -> full tokenized prompt cold-fill.
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13, 14]
    _drive_full(eng, messages, max_tokens=1)
    assert prompts_seen[1] == [11, 12, 13, 14]
    new_cs = eng._sessions["s"].cache_state
    assert new_cs is not cs
    _assert_no_desynced_cache(new_cs)


def test_generate_locked_mtp_cancel_session_consistent_next_turn(monkeypatch):
    """C1 test 4: cancel on an MTP-finalized (target+head) HIT session. The
    reconcile invalidates the hybrid cache (finalize ran at close, cache
    lands AHEAD, ArraysCache cannot rewind) and the C1 commit's Path 0
    leaves the stale messages harmless: no desynced survivor (stash/head
    included), and the next same-history request MISSes into an honest full
    cold-fill whose new session passes _assert_no_desynced_cache with the
    finalized head layout."""
    from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod

    STOP = 7
    next_map = lambda t: (t + 1) % 50
    stored = [1, 2, 3, 4, STOP]
    suffix = [40, 20, 21]

    cache = [
        MockArrays(), MockArrays(), MockKV(), MockArrays(), MockKV(), MockKV(),
    ]
    for c in cache:
        if hasattr(c, "offset"):
            c.offset = len(stored)
    cache[5].offset = len(stored) - 1
    for i in (0, 1, 3):
        cache[i].cache = [tuple(stored)]

    eng, cs, messages = _generate_engine(cache, stored, mtp=True, suffix=suffix)
    cs.mtp_last_hidden = mx.array([[[float(STOP)]]])
    cs.mtp_hidden_offset = len(stored)

    prompt_lens: List[int] = []

    def fake_step(prompt, model, head, **kwargs):
        prompt_lens.append(int(prompt.size))
        return qwen_mtp_generate_step(
            prompt, model=None, head=None,
            ops=MockOps(next_map, next_map), **kwargs,
        )

    monkeypatch.setattr(qwen_mtp_mod, "qwen_mtp_generate_step", fake_step)
    # Mock-vs-real artifact: the turn-2 cold-fill appends a REAL mlx-lm
    # KVCache head after the 5 MOCK target entries, and _get_cache_offset's
    # first pass (type name == "KVCache") would then read the HEAD's offset
    # instead of a target's. In production every full-attention TARGET layer
    # is a real KVCache and precedes the head, so the head is never selected
    # — keep the mock cache homogeneous instead.
    monkeypatch.setattr(
        qwen_mtp_mod, "make_head_cache",
        lambda n: [MockKV() for _ in range(max(1, n))],
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2

    # Invalidated fail-closed; stash/head cannot survive desynced.
    assert cs.cache is None and cs.token_ids is None
    assert cs.mtp_last_hidden is None and cs.mtp_hidden_offset is None
    _assert_no_desynced_cache(cs)
    # Path 0: session untouched — stale messages are harmless w/o a cache.
    assert eng._sessions["s"] is orig_sess
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]

    # Next same-history request: HIT requires a live cache -> honest MISS ->
    # the FULL tokenized prompt is cold-filled (no stale splice), and the
    # new MTP session finalizes into the consistent head-trailing layout.
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13, 14]
    chunks2 = _drive_full(eng, messages, max_tokens=3)
    assert chunks2[-1].finish_reason == "stop"
    assert prompt_lens[-1] == 4  # full prompt, not a suffix splice
    sess2 = eng._sessions["s"]
    new_cs = sess2.cache_state
    assert new_cs is not cs
    _assert_no_desynced_cache(new_cs, mtp_head_trailing=True)
    assert [m.get("content") for m in sess2.messages[:-1]] == ["u1", "a1", "u2"]
    assert sess2.messages[-1]["role"] == "assistant"
    assert sess2.total_cache_tokens == len(new_cs.token_ids)


def test_generate_locked_normal_eos_save_regression(monkeypatch):
    """C1 test 5 (regression): a normal, uncancelled turn with real content
    must save the session EXACTLY as before the fix — one assistant message
    appended, token_ids == prompt + generated, no interrupted-turn commit
    artifacts (no eot injection, no rollback)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    # A vocab-bearing tokenizer proves the eot close does NOT fire on the
    # healthy path even when it is detectable.
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": 77},
    )

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(101, "x"), (102, "x")]],
    )

    chunks = _drive_full(eng, messages, max_tokens=2)
    assert chunks[-1].finish_reason == "stop"

    sess = eng._sessions["s"]
    assert [m.get("content") for m in sess.messages] == ["u1", "a1", "u2", "xx"]
    assert cs.token_ids == stored + [52, 99] + [101, 102]  # no 77 injected
    assert sess.total_cache_tokens == len(cs.token_ids)
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)


# ---------------------------------------------------------------------------
# C1 amendments — A1 tri-state close, A2 empty-response tail verification,
# A3 GeneratorExit (stream teardown) finalization.
# ---------------------------------------------------------------------------


def test_try_close_interrupted_turn_tri_state():
    """A1 unit coverage: every close path maps to the right tri-state, and
    every FAILED path invalidates fail-closed — 'close unnecessary'
    (gemma4/glm, EOT tail) commits as-is, while 'close unavailable' must
    NEVER leave a live cache for an unterminated ChatML commit."""
    from mlx_soloheaven.engine.mlx_engine import TurnCloseResult

    def fresh(family="chatml", *, vlm=False, model=None, tokenizer=None,
              n_cache=5):
        stored = [1, 2, 3, 4, 5]
        cache = [MockKV() for _ in range(n_cache)]
        for c in cache:
            c.offset = len(stored)
        eng, cs, _ = _generate_engine(cache, stored, mtp=False, suffix=[])
        eng.model_family = family
        eng._use_vlm = vlm
        if model is not None:
            eng._language_model = model
        if tokenizer is not None:
            eng.tokenizer = tokenizer
        return eng, cs

    vocab_tok = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": 77},
    )

    # gemma4 / glm: template needs no closer — NOT_REQUIRED, cache untouched.
    for family in ("gemma4", "glm"):
        eng, cs = fresh(family)
        assert (
            eng._try_close_interrupted_turn("s", cs)
            is TurnCloseResult.NOT_REQUIRED
        )
        assert cs.cache is not None and cs.token_ids == [1, 2, 3, 4, 5]

    # A2: recorded tail already an end-of-turn token — NOT_REQUIRED even on
    # an otherwise close-incapable engine (natural termination verified).
    eng, cs = fresh(
        tokenizer=SimpleNamespace(decode=lambda ids: "x", eos_token_ids=[5]),
    )
    assert (
        eng._try_close_interrupted_turn("s", cs)
        is TurnCloseResult.NOT_REQUIRED
    )
    assert cs.cache is not None
    assert cs.token_ids[-1] == 5  # nothing injected

    # ChatML + mlx-vlm path: close unavailable — FAILED + invalidated.
    eng, cs = fresh(vlm=True, tokenizer=vocab_tok, model=_CallableTargetModel())
    assert eng._try_close_interrupted_turn("s", cs) is TurnCloseResult.FAILED
    assert cs.cache is None and cs.token_ids is None

    # ChatML + head-bearing (target+head) cache: FAILED + invalidated.
    eng, cs = fresh(n_cache=6, tokenizer=vocab_tok, model=_CallableTargetModel())
    assert eng._try_close_interrupted_turn("s", cs) is TurnCloseResult.FAILED
    assert cs.cache is None and cs.token_ids is None

    # ChatML + non-callable target: FAILED + invalidated.
    eng, cs = fresh(tokenizer=vocab_tok)
    assert eng._try_close_interrupted_turn("s", cs) is TurnCloseResult.FAILED
    assert cs.cache is None and cs.token_ids is None

    # ChatML + no detectable end-of-turn token: FAILED + invalidated.
    eng, cs = fresh(model=_CallableTargetModel())
    assert eng._try_close_interrupted_turn("s", cs) is TurnCloseResult.FAILED
    assert cs.cache is None and cs.token_ids is None

    # ChatML + callable target + vocab: CLOSED — eot recorded + verified.
    eng, cs = fresh(tokenizer=vocab_tok, model=_CallableTargetModel())
    assert eng._try_close_interrupted_turn("s", cs) is TurnCloseResult.CLOSED
    assert cs.token_ids == [1, 2, 3, 4, 5, 77]
    for c in cs.cache:
        assert c.offset == len(cs.token_ids)


def test_generate_locked_cancel_chatml_close_unavailable_invalidates(
    monkeypatch,
):
    """A1 end-to-end: cancel on a ChatML HIT where the close is UNAVAILABLE
    (no <|im_end|> in the vocab, non-callable target). Pre-amendment this
    COMMITTED the unterminated turn — the next HIT's suffix splice
    (_suffix_tokens_chatml) then assumed a prior <|im_end|> that was never
    in the KV (template corruption). Now: FAILED → session cache
    invalidated fail-closed, messages left stale-but-harmless, and the next
    same-history request honestly MISSes into a full cold-fill."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    # Default _generate_engine stub: no get_vocab (eot undetectable) and a
    # non-callable SimpleNamespace target — close-incapable by construction.

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(101, "x"), (102, "x"), (103, "x")],
            [(201, "y")],
        ],
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_cancelled(eng, messages, cancel_after=2)
    assert len(chunks) == 2

    # NOT committed unterminated: invalidated fail-closed instead.
    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)
    assert eng._sessions["s"] is orig_sess  # no commit happened
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]

    # Next same-history request: HIT requires a live cache → honest MISS →
    # the FULL tokenized prompt cold-fills (no stale splice).
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13, 14]
    _drive_full(eng, messages, max_tokens=1)
    assert prompts_seen[1] == [11, 12, 13, 14]


def test_generate_locked_empty_response_exhaustion_closes_chatml(monkeypatch):
    """A2 end-to-end: a thinking-only HIT turn whose stream ended WITHOUT an
    end-of-turn tail — the max_tokens-exhaustion shape (token 302 decodes to
    '</think>' but is NOT an EOS id, so the cached tail is not a turn
    terminator). The old close_turn=False commit assumed 'ended naturally';
    now the tail verification fails and the commit CLOSES the turn (eot
    forwarded through the target + recorded) before committing."""
    EOT = 77
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    _restamp_contract(eng, thinking=True)  # this test drives thinking=True
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],  # 302 is NOT an EOS: exhaustion, not natural end
        get_vocab=lambda: {"<|im_end|>": EOT},
    )

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(301, "pondering"), (302, "</think>")],
            [(401, "z")],
        ],
    )

    chunks = _drive_full(eng, messages, max_tokens=8, thinking=True)
    assert chunks[-1].finish_reason == "stop"

    # Turn CLOSED then committed: eot in token_ids, every layer on target.
    sess = eng._sessions["s"]
    assert sess.messages[-1] == {
        "role": "assistant",
        "content": "<think>\npondering</think>",
        "interrupted": True,  # U1: C1-committed turns carry the marker
    }
    assert cs.token_ids == stored + [52, 99] + [301, 302, EOT]
    assert sess.total_cache_tokens == len(cs.token_ids)
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)

    # Next turn extends the now template-valid sequence with u3 only.
    messages2 = list(messages) + [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "u3"},
    ]
    _drive_full(eng, messages2, max_tokens=1, thinking=True)
    assert prompts_seen[1] == [53, 99]
    _assert_no_desynced_cache(cs)


def test_generate_locked_empty_response_exhaustion_close_unavailable_invalidates(
    monkeypatch,
):
    """A2 fail-closed arm: the same thinking-only exhaustion shape (no EOS
    tail) on a close-INCAPABLE ChatML engine — committing the unterminated
    turn is forbidden, so the session cache invalidates; the client still
    receives the terminal stop frame."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    _restamp_contract(eng, thinking=True)  # this test drives thinking=True

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(301, "pondering"), (302, "</think>")]],
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_full(eng, messages, max_tokens=8, thinking=True)
    assert chunks[-1].finish_reason == "stop"

    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)
    assert eng._sessions["s"] is orig_sess  # no commit happened
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]


def _drive_generator_exit(eng, messages, *, frames=2, max_tokens=8):
    """Pull ``frames`` chunks then .close() the ENGINE generator at the
    suspended streaming yield — the client-disconnect teardown shape (A3):
    both production drivers (generate_stream_async/_batches_async `_run`
    threads and the process worker's _run_generate) break out of their
    consumer loop on cancel_event and DROP the generator, which closes it
    (GeneratorExit at the yield) BEFORE the engine loop can observe the
    event — skipping every post-loop commit."""
    gen = eng._generate_locked(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        session_id="s",
        tools=None,
        cancel_event=None,
        thinking=False,
        thinking_budget=0,
        top_p=1.0,
        min_p=0.0,
        top_k=0,
        repetition_penalty=1.0,
        response_format=None,
    )
    chunks = [next(gen) for _ in range(frames)]
    gen.close()
    return chunks


def test_generate_locked_generator_exit_mid_stream_commits_consistent(
    monkeypatch,
):
    """A3: .close() on the engine generator mid-stream on a HIT session —
    GeneratorExit at the streaming yield skips the whole post-loop, which
    pre-amendment resurrected the original C1 desync (stale messages +
    advanced live cache → double-spliced user_N). The rescue finalization
    must reconcile + commit exactly like a cancel: turn template-closed,
    messages ↔ token_ids consistent, and the next same-history request
    splices u3 ONLY."""
    EOT = 77
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)
    eng._language_model = _CallableTargetModel()
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {"<|im_end|>": EOT},
    )

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [
            [(101, "x"), (102, "x"), (103, "x"), (104, "x")],
            [(201, "y")],
        ],
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_generator_exit(eng, messages, frames=2)
    assert len(chunks) == 2

    # Rescue committed the interrupted turn (closed + consistent).
    sess = eng._sessions["s"]
    assert sess is not orig_sess
    assert sess.cache_state is cs
    assert cs.token_ids == stored + [52, 99] + [101, 102, EOT]
    assert sess.messages[-1] == {
        "role": "assistant",
        "content": "xx",
        "interrupted": True,  # U1: C1-committed turns carry the marker
    }
    assert sess.total_cache_tokens == len(cs.token_ids)
    for c in cache:
        assert c.offset == len(cs.token_ids)
    _assert_no_desynced_cache(cs)

    # Next request: HIT, u3 spliced exactly once after the closed turn.
    messages2 = list(messages) + [
        {"role": "assistant", "content": "xx"},
        {"role": "user", "content": "u3"},
    ]
    _drive_full(eng, messages2, max_tokens=1)
    assert prompts_seen[1] == [53, 99]
    _assert_no_desynced_cache(cs)


def test_generate_locked_generator_exit_close_unavailable_invalidates(
    monkeypatch,
):
    """A3 fail-closed arm: GeneratorExit mid-stream on a close-INCAPABLE
    ChatML HIT — no desync survives: the rescue's commit reaches the FAILED
    close and invalidates the session cache (messages stale-but-harmless,
    next request MISSes)."""
    stored = [1, 2, 3, 4, 5]
    cache = [MockKV() for _ in range(5)]
    for c in cache:
        c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=[])
    _use_real_messages_match(eng)
    _content_token_suffix(eng)

    prompts_seen: list = []
    _scripted_lm_stream(
        monkeypatch, prompts_seen,
        [[(101, "x"), (102, "x"), (103, "x")]],
    )

    orig_sess = eng._sessions["s"]
    chunks = _drive_generator_exit(eng, messages, frames=2)
    assert len(chunks) == 2

    assert cs.cache is None
    assert cs.token_ids is None
    _assert_no_desynced_cache(cs)
    assert eng._sessions["s"] is orig_sess
    assert [m.get("content") for m in orig_sess.messages] == ["u1", "a1"]
