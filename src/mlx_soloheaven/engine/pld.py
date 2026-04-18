"""Prompt Lookup Decoding (PLD) for MLX.

Replaces the draft *model* in speculative decoding with an n-gram lookup over
prompt + generated tokens. For RAG / long-context / agentic workloads the model
frequently copies spans from the prompt, so a cheap Rabin-Karp index yields
high-quality drafts at effectively zero cost.

Exports ``PLDMatcher`` and ``pld_generate_step`` (drop-in replacement for
``mlx_lm.generate.speculative_generate_step`` with the draft model removed).
"""

from __future__ import annotations

import functools
import logging
from collections import defaultdict
from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models import cache as cache_mod

logger = logging.getLogger(__name__)


class PLDMatcher:
    """N-gram matcher for Prompt Lookup Decoding.

    Builds ``dict[tuple[int,...], list[int]]`` per k in ``1..max_k`` mapping
    each k-gram to sorted end-positions. Append is O(max_k); match is O(1)
    amortized (hash lookup + bounded self-match skip).

    Selection heuristic: *most recent* match (iterate positions from tail);
    falls back from the requested k down to 1 (mirrors the reference PLD impl
    at https://github.com/apoorvumang/prompt-lookup-decoding).

    Usage::

        matcher = PLDMatcher(prompt_tokens, max_k=3)
        matcher.append(new_token)           # after each accepted token
        draft = matcher.match(suffix, k=3, n=10)
    """

    def __init__(self, prompt_tokens: Sequence[int], max_k: int = 3):
        self.max_k = max_k
        self.tokens: List[int] = list(prompt_tokens)
        self.index: List[defaultdict] = [defaultdict(list) for _ in range(max_k + 1)]
        n = len(self.tokens)
        for k in range(1, max_k + 1):
            idx = self.index[k]
            for end in range(k - 1, n):
                idx[tuple(self.tokens[end - k + 1 : end + 1])].append(end)

    def append(self, token: int) -> None:
        """Register ``token`` as the new tail; O(max_k)."""
        self.tokens.append(int(token))
        end = len(self.tokens) - 1
        for k in range(1, self.max_k + 1):
            if end - k + 1 < 0:
                break
            self.index[k][tuple(self.tokens[end - k + 1 : end + 1])].append(end)

    def truncate(self, new_len: int) -> None:
        """Roll back to ``new_len`` tokens (for rejected speculative drafts)."""
        if new_len >= len(self.tokens):
            return
        for k in range(1, self.max_k + 1):
            idx = self.index[k]
            for end in range(len(self.tokens) - 1, new_len - 1, -1):
                if end - k + 1 < 0:
                    continue
                gram = tuple(self.tokens[end - k + 1 : end + 1])
                lst = idx.get(gram)
                if lst and lst[-1] == end:
                    lst.pop()
                    if not lst:
                        del idx[gram]
        self.tokens = self.tokens[:new_len]

    def match(self, suffix: Sequence[int], k: int, n: int) -> List[int]:
        """Return up to ``n`` tokens following the most-recent match of
        ``suffix[-k:]``. Falls back k -> k-1 -> ... -> 1. Skips the tail
        self-match (end == len(tokens) - 1)."""
        if not suffix or n <= 0:
            return []
        tokens = self.tokens
        total = len(tokens)
        max_try = min(k, len(suffix), self.max_k)
        for kk in range(max_try, 0, -1):
            positions = self.index[kk].get(tuple(suffix[-kk:]))
            if not positions:
                continue
            for i in range(len(positions) - 1, -1, -1):
                start = positions[i] + 1
                if start >= total:
                    continue  # self-match at current tail
                return tokens[start : min(start + n, total)]
        return []


def pld_generate_step(
    prompt: mx.array,
    model: nn.Module,
    *,
    num_draft_tokens: int = 10,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    ngram_k: int = 3,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """PLD variant of ``speculative_generate_step``.

    Yields ``(token, logprobs, from_draft)`` triples. ``from_draft=True``
    when the accepted token was proposed by the PLD matcher.

    If the cache is not trimmable the function logs a warning and falls back
    to single-token generation (``from_draft`` is always False).
    """
    y = prompt.astype(mx.uint32)
    prev_tokens: Optional[mx.array] = None
    model_cache = prompt_cache if prompt_cache is not None else cache_mod.make_prompt_cache(model)
    speculative = cache_mod.can_trim_prompt_cache(model_cache)
    if not speculative:
        logger.warning("pld_generate_step: prompt_cache is not trimmable; "
                       "falling back to non-speculative generation.")
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(toks, logits):
        if logits_processors:
            for p in logits_processors:
                logits = p(toks, logits)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return sampler(lp), lp

    def _step(cache_, y_, n_predict: int = 1):
        nonlocal prev_tokens
        with mx.stream(generation_stream):
            logits = model(y_[None], cache=cache_)[:, -n_predict:, :]
            quantize_cache_fn(cache_)
            if logits_processors:
                out_y, out_lp = [], []
                y_trim = y_[: -(n_predict - 1)] if n_predict > 1 else y_
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, y_trim])
                        if prev_tokens is not None else y_trim
                    )
                    yi, lpi = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(yi)
                    out_lp.append(lpi)
                    y_trim = yi
                return mx.concatenate(out_y, axis=0), mx.concatenate(out_lp, axis=0)
            return _process_and_sample(None, logits.squeeze(0))

    def _prefill(cache_, y_):
        while y_.size > prefill_step_size:
            model(y_[:prefill_step_size][None], cache=cache_)
            quantize_cache_fn(cache_)
            mx.eval([c.state for c in cache_])
            y_ = y_[prefill_step_size:]
            mx.clear_cache()
        return y_

    def _rewind(nd, na):
        if nd > na:
            cache_mod.trim_prompt_cache(model_cache, nd - na)

    matcher = PLDMatcher(prompt.tolist(), max_k=max(1, ngram_k))
    with mx.stream(generation_stream):
        y = _prefill(model_cache, y)

    ntoks, num_draft, n = 0, 0, 0
    try:
        while True:
            if max_tokens >= 0 and ntoks >= max_tokens:
                break

            draft_list: List[int] = []
            if speculative and num_draft_tokens > 0:
                remaining = max_tokens - ntoks if max_tokens >= 0 else num_draft_tokens
                want = min(num_draft_tokens, max(1, remaining) - 1)
                if want > 0:
                    draft_list = matcher.match(matcher.tokens, k=ngram_k, n=want)

            if not draft_list:
                # No draft available -> single-token step.
                tok, lp = _step(model_cache, y, n_predict=1)
                mx.eval(tok)
                tok_i = tok.tolist()[0] if hasattr(tok, "tolist") else int(tok)
                matcher.append(tok_i)
                ntoks += 1
                num_draft, n = 0, 0
                yield tok_i, lp, False
                if ntoks == max_tokens:
                    break
                y = mx.array([tok_i], mx.uint32)
                continue

            # Verify draft.
            num_draft = len(draft_list)
            draft_arr = mx.array(draft_list, mx.uint32)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
            y = mx.concatenate([y, draft_arr])
            tokens, logprobs = _step(model_cache, y, n_predict=num_draft + 1)
            mx.eval(tokens)
            tokens_l = tokens.tolist()

            n = 0
            while n < num_draft:
                if tokens_l[n] != draft_list[n]:
                    break
                ntoks += 1
                matcher.append(tokens_l[n])
                yield tokens_l[n], logprobs[n], True
                n += 1
                if ntoks == max_tokens:
                    break

            if ntoks < max_tokens:
                bonus = tokens_l[n]
                matcher.append(bonus)
                ntoks += 1
                yield bonus, logprobs[n], False

            if ntoks == max_tokens:
                break

            y = mx.array([tokens_l[n]], mx.uint32)
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
            _rewind(num_draft, n)
            num_draft, n = 0, 0
    finally:
        _rewind(num_draft, n)


if __name__ == "__main__":
    # Test 1: basic match with fallback
    m = PLDMatcher([1, 2, 3, 4, 1, 2, 3, 5, 6, 7], max_k=3)
    assert m.match([1, 2, 3], k=3, n=3) == [5, 6, 7]
    assert m.match([9, 9, 9], k=3, n=3) == []
    assert m.match([9, 9, 4], k=3, n=1) == [1]  # fallback to k=1 on trailing 4
    print("test 1 (basic match) ok")

    # Test 2: incremental append + truncate
    m2 = PLDMatcher([1, 2, 3, 4, 1, 2, 3, 5, 6, 7], max_k=3)
    for t in (9, 1, 2, 3):
        m2.append(t)
    # Tail (1,2,3) at end 13 is self-match; most-recent prior at end 6 -> [5,6]
    assert m2.match([1, 2, 3], k=3, n=2) == [5, 6]
    m2.truncate(10)
    assert m2.tokens == [1, 2, 3, 4, 1, 2, 3, 5, 6, 7]
    assert m2.match([5, 6], k=2, n=1) == [7]
    print("test 2 (incremental append + truncate) ok")

    # Test 3: pld_generate_step with a mocked model that always picks token 0.
    class _MockCache:
        def __init__(self):
            self.offset = 0
            self.state = mx.array([0])
        def is_trimmable(self): return True
        def trim(self, n):
            self.offset = max(0, self.offset - n)
            return n

    class _MockModel(nn.Module):
        def __init__(self, vocab: int = 16):
            super().__init__()
            self.vocab = vocab
            self.layers = [object()]
        def make_cache(self): return [_MockCache()]
        def __call__(self, x, cache=None):
            T = x.shape[1]
            row = mx.concatenate([mx.array([[1e9]]), mx.zeros((1, self.vocab - 1))], axis=1)
            if cache is not None:
                cache[0].offset += T
            return mx.broadcast_to(row[:, None, :], (1, T, self.vocab))

    model = _MockModel()
    prompt = mx.array([1, 2, 3, 0, 0, 0], dtype=mx.uint32)
    out = [t for t, _lp, _fd in pld_generate_step(
        prompt, model, num_draft_tokens=4, max_tokens=5, ngram_k=3,
        prompt_cache=model.make_cache())]
    assert out == [0, 0, 0, 0, 0], f"expected all zeros, got {out}"
    print("test 3 (pld_generate_step with mock model) ok")

    # Test 4: rejection path — model diverges from matcher draft, _rewind trims cache.
    class _TrackingCache(_MockCache):
        def __init__(self):
            super().__init__()
            self.trim_calls: List[int] = []
        def trim(self, n):
            self.trim_calls.append(n)
            return super().trim(n)

    class _DivergentModel(nn.Module):
        """Deterministic next-token mapping; diverges from matcher's draft copy."""
        def __init__(self, vocab: int = 16):
            super().__init__()
            self.vocab = vocab
            self.layers = [object()]
        def make_cache(self): return [_TrackingCache()]
        def _next(self, tok: int) -> int:
            mapping = {5: 6, 6: 10, 10: 11, 11: 12, 7: 8, 8: 9, 1: 2, 2: 3, 3: 4, 4: 5, 0: 1}
            return mapping.get(tok, (tok + 1) % self.vocab)
        def __call__(self, x, cache=None):
            T = x.shape[1]
            toks = x[0].tolist()
            rows = []
            for t in toks:
                row = mx.zeros((1, self.vocab))
                row[0, self._next(int(t))] = 1e9
                rows.append(row)
            out = mx.stack(rows, axis=1)
            if cache is not None:
                cache[0].offset += T
            return out

    # Prompt [1..8, 1..5]: matcher sees suffix (3,4,5) also at pos 2..4, proposes
    # next tokens [6,7,8]. Model truth: 5→6→10→11 — draft[0]=6 ok, draft[1]=7 vs
    # model 10 rejects, triggering _rewind.
    dmodel = _DivergentModel(vocab=16)
    dcache = dmodel.make_cache()
    dprompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5], dtype=mx.uint32)
    dout = [t for t, _lp, _fd in pld_generate_step(
        dprompt, dmodel, num_draft_tokens=3, max_tokens=4, ngram_k=3,
        prompt_cache=dcache)]
    # Non-speculative ground truth: 5→6→10→11→12
    assert dout == [6, 10, 11, 12], f"expected [6,10,11,12], got {dout}"
    assert any(c > 0 for c in dcache[0].trim_calls), \
        f"expected trim() called with >0 on rejection, got {dcache[0].trim_calls}"
    print(f"test 4 (rejection path, trim calls={dcache[0].trim_calls}) ok")

    # Test 5: logits processor sees monotonically growing cumulative tokens.
    class _Recorder:
        def __init__(self): self.seen: List[List[int]] = []
        def __call__(self, tokens, logits):
            if tokens is None:
                return logits
            self.seen.append(tokens.tolist() if hasattr(tokens, "tolist") else list(tokens))
            return logits

    rec = _Recorder()
    rmodel = _DivergentModel(vocab=16)
    rcache = rmodel.make_cache()
    rprompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5], dtype=mx.uint32)
    rout = [t for t, _lp, _fd in pld_generate_step(
        rprompt, rmodel, num_draft_tokens=3, max_tokens=4, ngram_k=3,
        prompt_cache=rcache, logits_processors=[rec])]
    assert rout == [6, 10, 11, 12], f"expected [6,10,11,12], got {rout}"
    assert len(rec.seen) > 0, "recorder saw no calls"
    lens = [len(s) for s in rec.seen]
    expected_prefix = rprompt.tolist()
    # Every call must include the full prompt as prefix; lengths may dip on
    # rejection (prev_tokens trimmed) then resume, but max must exceed initial.
    for s in rec.seen:
        assert s[:len(expected_prefix)] == expected_prefix, "prompt prefix lost"
    final = rec.seen[-1]
    extension = final[len(expected_prefix):]
    assert all(tok in rout for tok in extension), \
        f"stray extension tokens {extension} vs output {rout}"
    assert max(lens) > lens[0], f"recorder never grew: {lens}"
    print(f"test 5 (logits processor, {len(rec.seen)} calls, lens={lens}) ok")

    # Test 6: long-prompt smoke — init & match perf on 50K tokens.
    import random
    import time as _time
    random.seed(0)
    long_prompt = [random.randrange(1000) for _ in range(50_000)]
    t0 = _time.perf_counter()
    lm = PLDMatcher(long_prompt, max_k=3)
    init_ms = (_time.perf_counter() - t0) * 1000
    N = 100
    suffixes = [[random.randrange(1000) for _ in range(3)] for _ in range(N)]
    t1 = _time.perf_counter()
    for s in suffixes:
        lm.match(s, k=3, n=10)
    avg_us = (_time.perf_counter() - t1) / N * 1e6
    assert init_ms < 500, f"init too slow: {init_ms:.1f}ms"
    assert avg_us < 50, f"match too slow: avg {avg_us:.2f}us"
    if init_ms > 200: print(f"  warning: init_ms={init_ms:.1f} >200ms soft target")
    if avg_us > 10: print(f"  warning: avg_match_us={avg_us:.2f} >10us soft target")
    print(f"test 6 (long prompt, init={init_ms:.1f}ms, match_avg={avg_us:.2f}us) ok")

    print("\nall tests passed")
