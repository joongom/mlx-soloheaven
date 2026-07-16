"""Logits processors for generation control."""

import codecs
import re

import mlx.core as mx


# ---------------------------------------------------------------------------
# U22 round 2 (codex batch-5 F4): shared UTF-8 tail inspection + bounded
# force-deferral gate. Used by BOTH the plain-path ThinkingBudgetProcessor and
# the mlx-vlm MTP clone's history-derived force site (mlx_engine).
#
# Design (byte-tail check): decode()-only inspection cannot distinguish a
# TRUNCATION-induced trailing U+FFFD (the detokenizer replaced an incomplete
# multi-byte tail) from a GENUINE U+FFFD the model literally emitted (bytes
# EF BF BD) — both decode to the same text. The sound check inspects the raw
# token BYTES: recover the tail tokens' bytes via the tokenizer's vocab
# convention (SentencePiece ``<0xHH>`` byte-fallback pieces, GPT-2-style
# byte-level BPE strings, or plain-text pieces) and ask an incremental UTF-8
# decoder whether the byte stream ends with a VALID-BUT-INCOMPLETE multi-byte
# sequence (only that case can be completed by waiting):
#   * ...ED / ...ED 95      -> incomplete '한' prefix -> defer;
#   * ...EF BF BD (real �)  -> complete sequence      -> force now;
#   * ...FF (invalid byte)  -> can never complete      -> force now.
# When token bytes are NOT recoverable (exotic tokenizer surface), the
# legacy decode-endswith-U+FFFD heuristic is kept as a fallback — it can
# false-positive on a genuine �, but the deferral stays bounded (max 4
# tokens), so the worst case is 4 extra thinking tokens.
# ---------------------------------------------------------------------------

_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_BYTE_MODE_ATTR = "_soloheaven_byte_mode"

_GPT2_BYTE_DECODER: "dict[str, int] | None" = None


def _gpt2_byte_decoder() -> "dict[str, int]":
    """char -> byte inverse of the standard GPT-2 ``bytes_to_unicode`` map
    used by byte-level BPE tokenizers (qwen/gemma-BPE style vocab strings).
    Built once (module-level memo)."""
    global _GPT2_BYTE_DECODER
    if _GPT2_BYTE_DECODER is None:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _GPT2_BYTE_DECODER = {chr(c): b for b, c in zip(bs, cs)}
    return _GPT2_BYTE_DECODER


def _tokenizer_byte_mode(tokenizer) -> str:
    """Classify how the tokenizer's vocab strings encode bytes (memoized on
    the instance): ``"spm"`` (SentencePiece with ``<0xHH>`` byte-fallback
    pieces), ``"bytelevel"`` (GPT-2 style byte-level BPE), ``"plain"``
    (pieces are whole characters — a mid-character tail is impossible), or
    ``"unknown"`` (no vocab surface — byte recovery unavailable)."""
    mode = getattr(tokenizer, _BYTE_MODE_ATTR, None)
    if isinstance(mode, str):
        return mode
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    mode = "unknown"
    try:
        get_vocab = getattr(hf, "get_vocab", None)
        if get_vocab is not None:
            vocab = get_vocab()
            if "<0x00>" in vocab or "<0x0A>" in vocab:
                mode = "spm"
            elif "Ġ" in vocab or "ĠĠ" in vocab:
                mode = "bytelevel"
            else:
                mode = "plain"
    except Exception:  # noqa: BLE001 — best-effort probe only
        mode = "unknown"
    try:
        setattr(tokenizer, _BYTE_MODE_ATTR, mode)
    except Exception:  # noqa: BLE001 — slotted/frozen wrapper: just re-probe
        pass
    return mode


def _token_str_to_bytes(tok_str: str, mode: str) -> bytes:
    """Bytes of one vocab token string under the tokenizer's byte
    convention (see ``_tokenizer_byte_mode``)."""
    if mode == "spm":
        m = _BYTE_FALLBACK_RE.match(tok_str)
        if m is not None:
            return bytes([int(m.group(1), 16)])
        # Plain SentencePiece piece: whole characters ('▁' = space marker).
        return tok_str.replace("▁", " ").encode("utf-8", "replace")
    if mode == "bytelevel":
        decoder = _gpt2_byte_decoder()
        out = bytearray()
        for ch in tok_str:
            b = decoder.get(ch)
            if b is None:
                # Added/special token (e.g. "<|im_end|>" beyond the map):
                # plain text, complete by construction.
                out += ch.encode("utf-8", "replace")
            else:
                out.append(b)
        return bytes(out)
    # "plain": pieces are whole characters — encode is exact.
    return tok_str.encode("utf-8", "replace")


def _token_ids_to_bytes(tokenizer, ids) -> "bytes | None":
    """Recover the raw byte stream of ``ids`` via the tokenizer's vocab
    convention. Returns ``None`` when the surface needed for recovery is
    unavailable (caller falls back to the decode heuristic)."""
    mode = _tokenizer_byte_mode(tokenizer)
    if mode == "unknown":
        return None
    hf = getattr(tokenizer, "_tokenizer", tokenizer)
    convert = getattr(hf, "convert_ids_to_tokens", None)
    if convert is None:
        return None
    try:
        toks = convert(list(ids))
    except Exception:  # noqa: BLE001
        return None
    out = bytearray()
    for t in toks:
        if not isinstance(t, str):
            return None
        out += _token_str_to_bytes(t, mode)
    return bytes(out)


def _utf8_incomplete_tail(data: bytes) -> bool:
    """True iff ``data`` ends with a VALID but INCOMPLETE multi-byte UTF-8
    sequence — the only case a bounded wait can complete. Complete text,
    a genuine U+FFFD (EF BF BD), and invalid bytes (e.g. FF) all return
    False. Uses the stdlib incremental decoder: with ``final=False`` it
    BUFFERS a truncated valid sequence at end-of-input (invalid bytes are
    replaced immediately, never buffered)."""
    if not data:
        return False
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    dec.decode(data, False)
    buffered = dec.getstate()[0]
    return len(buffered) > 0


def tail_ends_mid_utf8(tokenizer, token_ids, window: int = 8) -> bool:
    """Whether the emitted byte stream currently ends MID multi-byte UTF-8
    character (U22): forcing a close now would stamp a truncation U+FFFD at
    the end of ``reasoning_content``.

    Primary check: recover the last ``window`` tokens' raw BYTES and test
    for a valid-but-incomplete trailing sequence (sound — a genuine literal
    U+FFFD is a complete sequence and returns False). Fallback when bytes
    are unrecoverable: the legacy decode-endswith-U+FFFD heuristic (may
    false-positive on a genuine �; safe because the caller's deferral is
    bounded). Fails open (False → force immediately, the historical
    behaviour) on any error."""
    try:
        ids = [int(t) for t in list(token_ids)[-window:]]
        if not ids:
            return False
        data = _token_ids_to_bytes(tokenizer, ids)
        if data is not None:
            return _utf8_incomplete_tail(data)
        text = tokenizer.decode(ids)
        return bool(text) and text.endswith("�")
    except Exception:  # noqa: BLE001 — the deferral is best-effort only
        return False


class ForceDeferralGate:
    """Bounded 'skip the forced close this round' gate (U22 round 2).

    Same gate implementation/semantics for ``ThinkingBudgetProcessor``
    (plain/stateful path) and the mlx-vlm MTP clone's history-derived force
    site — but deliberately SEPARATE instances (the clone constructs its
    own), so nothing can mutate the verifier's gate during its synchronous
    per-position walk: when the budget says a
    ``think_end`` must be forced but the emitted byte stream ends mid
    multi-byte character, the force is deferred — up to ``max_deferrals``
    consults — letting the model finish the character. After the bound the
    force fires regardless (the gate can never loop). ``rearm()`` resets the
    budget when a thinking block closes (a later block gets a fresh bound).
    A ``None`` tokenizer disables deferral entirely (immediate force —
    byte-identical historical behaviour).

    Batch-5 round 4 (codex finding 1): the decision and the state change are
    SPLIT — ``should_defer_preview`` is the pure decision and
    ``commit_deferral`` the explicit state change — so the MTP speculative
    verifier can PREVIEW the gate for every provisional draft position and
    commit deferrals only for the positions actually emitted (accepted
    prefix + target bonus). Pre-split, ``should_defer`` mutated inline for
    every provisional position, so positions discarded after the first
    draft/target mismatch still exhausted the shared 4-deferral budget and
    a later REAL mid-character tail was hard-forced into ``'�</think>'``.
    ``should_defer`` (preview + commit in one call) remains for the
    call==emit paths — the plain ``ThinkingBudgetProcessor`` and the MTP
    plain-step/bonus sites emit every position they see, so their inline
    mutation stays correct and byte-identical."""

    def __init__(self, tokenizer, max_deferrals: int = 4, window: int = 8):
        self.tokenizer = tokenizer
        self.max_deferrals = max_deferrals
        self.window = window
        self.deferrals = 0

    def should_defer_preview(
        self, token_ids, pending: int = 0, base: "int | None" = None
    ) -> bool:
        """Pure decision (never mutates): would a deferral be granted for
        this byte tail? ``pending`` counts deferrals PROVISIONALLY granted
        to earlier not-yet-committed positions of the same speculative
        block, so a per-position simulation still respects the shared
        bound. ``base`` (batch-5 round 6) overrides the committed count
        used for the bound — the MTP verifier passes its preview-local
        epoch base, which resets to 0 when the provisional accepted prefix
        folds a think_end, so a REOPENED thinking cycle previews with the
        fresh allowance sequential decoding would grant (the emitted-fold
        replay re-arms the committed count at the same point). ``None``
        keeps the live ``self.deferrals``. Commit with ``commit_deferral()``
        only for positions that are actually emitted."""
        committed = self.deferrals if base is None else base
        if (
            self.tokenizer is None
            or committed + pending >= self.max_deferrals
        ):
            return False
        return tail_ends_mid_utf8(self.tokenizer, token_ids, self.window)

    def commit_deferral(self) -> None:
        """Consume one deferral for a position that was ACTUALLY emitted."""
        self.deferrals += 1

    def should_defer(self, token_ids) -> bool:
        """Consult (and consume) the gate: True -> skip the force this
        round. Increments the deferral count only when a deferral is
        actually granted. Correct only at call==emit sites; provisional
        (speculative-verify) positions must use ``should_defer_preview`` +
        a post-walk ``commit_deferral`` replay instead."""
        granted = self.should_defer_preview(token_ids)
        if granted:
            self.commit_deferral()
        return granted

    def rearm(self) -> None:
        self.deferrals = 0


# Thinking-state tuple layout: ``(in_thinking, count, bare_idx, content_seen)``.
#   in_thinking  — currently inside a thinking block.
#   count        — tokens emitted in the CURRENT block (think_start counts as #1).
#   bare_idx     — how many leading tokens of the BARE thought opener have matched
#                  contiguously from generation start (gen-start lookbehind for the
#                  multi-token ``thought\n`` opener; see ``advance_think_state``).
#   content_seen — whether any token that is NOT part of an in-progress bare-opener
#                  match has been emitted yet. Mirrors ``ThinkingRouter._content_seen``
#                  (tool_parser FIX 4): once True, a BARE opener can no longer fire,
#                  so a literal ``thought\n`` mid-content/tool-args is NOT mis-read
#                  as a thinking opener. Always-True for chatml (starts in_thinking).
# The last two slots are dormant (and the tuple semantically a 2-tuple) unless a
# non-empty ``bare_open_tokens`` is threaded into ``advance_think_state``; with
# bare detection off the behaviour is byte-identical to the original 2-tuple form.


def initial_think_state(model_family: str = "chatml") -> tuple[bool, int, int, bool]:
    """Initial ``(in_thinking, count, bare_idx, content_seen)`` for a fresh
    generation.

    ChatML starts INSIDE the thinking block (``in_thinking=True``); Gemma 4
    starts outside it. Matches ``ThinkingBudgetProcessor.__init__``. ``bare_idx``
    starts at 0; ``content_seen`` starts False for gemma4 (the only family that
    can see a bare opener at gen-start) and True for chatml (already in thinking
    — no gen-start bare detection applies).
    """
    in_thinking = model_family != "gemma4"
    return (in_thinking, 0, 0, in_thinking)


def advance_think_state(
    state: tuple[bool, int, int, bool],
    tok: int,
    *,
    think_start_token: int,
    think_end_token: int,
    bare_open_tokens: tuple[int, ...] | list[int] | None = None,
) -> tuple[bool, int, int, bool]:
    """Advance the thinking state by one EMITTED token ``tok``.

    Mirrors one ``ThinkingBudgetProcessor.__call__`` step (the bookkeeping part,
    not the logit mutation): think_start opens thinking and counts as token #1,
    think_end closes it, every other in-thinking token increments the count.
    Pure and incremental — O(1) — so it can be folded over the emitted history
    without re-scanning, then branched per draft position cheaply.

    BARE-OPENER DETECTION (Option A, gemma4 sliding-window fix): past the 1024
    sliding window the ``<|channel>`` think_start prime falls out of the window
    and the model emits a BARE ``thought\\n`` opener with NO ``<|channel>`` token,
    so the token-id based open never fires and the budget never caps thinking.
    When ``bare_open_tokens`` (the token-id sequence of ``"thought\\n"``) is
    supplied, this matches that sequence token-by-token at GENERATION START ONLY
    (``not in_thinking and not content_seen``) via the ``bare_idx`` lookbehind;
    on a full contiguous match thinking opens (``count`` = matched length, so the
    opener tokens count toward the budget exactly like the ``<|channel>`` path).
    Mirrors ``ThinkingRouter`` FIX 4: a literal ``thought\\n`` appearing AFTER any
    content (``content_seen``) is plain content and never opens thinking. With
    ``bare_open_tokens`` None/empty, the bare path is inert and this is identical
    to the original ``<|channel>``-only behaviour.
    """
    in_thinking, count, bare_idx, content_seen = state

    # Full ``<|channel>`` opener (short context) — unchanged, takes precedence.
    if not in_thinking and tok == think_start_token:
        return (True, 1, 0, True)

    if in_thinking:
        if tok == think_end_token:
            # Block closed; a later think_start re-opens (no gen-start bare
            # opener after content — content_seen latched True).
            return (False, 0, 0, True)
        return (True, count + 1, 0, True)

    # Not in thinking. Try the gen-start-only bare ``thought\n`` opener.
    if bare_open_tokens and not content_seen:
        if tok == bare_open_tokens[bare_idx]:
            new_idx = bare_idx + 1
            if new_idx >= len(bare_open_tokens):
                # Full bare opener matched: open thinking. The matched opener
                # tokens count as the first ``len`` thinking tokens (parity with
                # the <|channel> path counting the start token as #1).
                return (True, new_idx, 0, True)
            # Partial match still at gen-start: stay outside thinking, keep
            # content_seen False so the next opener token can extend the match.
            return (False, 0, new_idx, False)
        # Mismatch: the gen-start window is broken; this is real content, so the
        # bare opener can never fire again (latch content_seen True).
        return (False, 0, 0, True)

    # No bare detection (off, or content already seen): plain content, latch.
    return (in_thinking, count, 0, True)


def force_end_from_state(state: tuple[bool, int, int, bool], budget: int) -> bool:
    """Whether the next sampled token must be forced to ``think_end`` given the
    thinking state DERIVED from the already-emitted history. ``budget <= 0`` →
    never (no-op)."""
    if budget <= 0:
        return False
    in_thinking, count = state[0], state[1]
    return in_thinking and count >= budget


def should_force_think_end(
    history: list[int],
    *,
    budget: int,
    think_start_token: int,
    think_end_token: int,
    model_family: str = "chatml",
    bare_open_tokens: tuple[int, ...] | list[int] | None = None,
) -> bool:
    """History-derived thinking-budget enforcement decision (pure, stateless).

    Given the list of tokens ACTUALLY EMITTED so far (``history``), decide
    whether the NEXT sampled token must be forced to ``think_end_token`` because
    the thinking budget has been reached and the model is still inside a
    thinking block.

    This is the stateless counterpart of ``ThinkingBudgetProcessor``: instead
    of mutating a per-call counter (which over-counts in the MTP speculative
    verify, where the sampler runs per draft position but only ``accepted+1``
    tokens are emitted), it RE-DERIVES the thinking state from the emitted
    history every call. The caller is responsible for passing the exact
    per-position cumulative history so the decision fires at the right position.

    Semantics (match ``ThinkingBudgetProcessor``):

    * ``budget <= 0`` or no ``think_end_token`` (< 0) → never force (no-op).
    * ChatML (``model_family != "gemma4"``): generation begins INSIDE the
      thinking block, so ``in_thinking`` starts True; it ends at the first
      ``think_end_token``. A later ``think_start_token`` re-opens it.
    * Gemma 4: generation begins OUTSIDE thinking; ``in_thinking`` flips True
      after a ``think_start_token`` (the ``<|channel>`` opener) is emitted OR —
      when ``bare_open_tokens`` is supplied — after the BARE ``thought\\n`` opener
      token sequence is emitted at GENERATION START (the long-context sliding-
      window variant with no ``<|channel>`` token; see ``advance_think_state``),
      and ends at the first following ``think_end_token``.
    * thinking_token_count = number of tokens emitted since the most recent
      ``think_start`` (ChatML: since block start) and before any ``think_end``.
    * Returns True iff currently in_thinking AND thinking_token_count >= budget
      (i.e. the budget is reached and the block has not closed).

    Once a ``think_end_token`` already appears as the LAST emitted token (the
    forced close was just emitted), in_thinking is False → returns False (no
    double-force).

    Parity note: for Gemma 4 this matches ``ThinkingBudgetProcessor`` EXACTLY —
    the ``<|channel>`` think_start anchors the count inside the GENERATED stream,
    so prompt tokens before it never count. For ChatML there is no anchoring
    think_start and the stateful processor counts the token being sampled (it
    increments on every call, including the prompt-seeing one), whereas this
    helper counts only emitted tokens; the two then agree to within one token at
    the budget boundary (the helper fires at most one token LATER — conservative,
    never early). Gemma 4 is the runaway target, so exact parity there is what
    matters.
    """
    if budget <= 0 or think_end_token < 0:
        return False

    state = initial_think_state(model_family)
    for tok in history:
        state = advance_think_state(
            state,
            tok,
            think_start_token=think_start_token,
            think_end_token=think_end_token,
            bare_open_tokens=bare_open_tokens,
        )
    return force_end_from_state(state, budget)


class ThinkingBudgetProcessor:
    """Forces thinking-end token after a budget is reached.

    Works as a logits_processor for mlx-vlm's stream_generate.

    - ChatML: generation starts inside <think> block → in_thinking=True.
    - Gemma 4: model generates <|channel>thought first → in_thinking starts False.

    PER-CYCLE semantics: every thinking block in a turn is budgeted. A
    ``think_end`` closes the current block and RESETS the per-cycle counter; a
    later ``think_start`` re-opens thinking and re-budgets the new block. This
    matches the MTP helper (``advance_think_state`` / ``force_end_from_state``)
    exactly, so the plain/non-drafter path and the speculative path are
    lockstep-identical even for multi-block turns (START...END...START...).

    There is intentionally NO permanent "done after first close" latch: that
    latch is what diverged from the MTP helper, which never stopped re-budgeting
    later blocks.

    U22 — UTF-8 boundary deferral (opt-in via ``tokenizer``): forcing the
    close while the streaming detokenizer holds the leading bytes of an
    unfinished multi-byte character (Korean/CJK/emoji tokens are routinely
    split across token boundaries) stamps a U+FFFD replacement char at the
    end of ``reasoning_content``. When a ``tokenizer`` is supplied, the force
    is DEFERRED — the budget logit override is simply skipped — for up to
    ``_FORCE_DEFER_MAX`` extra tokens while the emitted byte stream ends MID
    multi-byte character, letting the model finish the character first.
    Round 2 (codex F4b): the mid-character test is the shared
    ``ForceDeferralGate`` / ``tail_ends_mid_utf8`` byte-tail check — a
    GENUINE literal U+FFFD the model emitted is a complete byte sequence and
    is NOT deferred (the old decode-endswith-U+FFFD heuristic deferred it);
    the decode heuristic remains only as the fallback for tokenizers whose
    raw token bytes cannot be recovered. The deferral is bounded so it can
    never loop (after ``_FORCE_DEFER_MAX`` skips the force fires
    regardless), and it only *delays the logit override* — token accounting
    (``thinking_tokens``) keeps advancing normally, so no counter is ever
    corrupted. Without a tokenizer the behaviour is byte-identical to the
    historical immediate force. The mlx-vlm MTP clone consults the SAME gate
    at its history-derived force site (mlx_engine), so both paths share the
    exact deferral semantics.
    """

    # U22: bounded deferral — a UTF-8 character is at most 4 bytes, so a
    # handful of extra tokens always completes (or provably abandons) it.
    _FORCE_DEFER_MAX = 4
    # Tokens inspected for the tail check; generous enough to cover a 4-byte
    # char split across single-byte fallback tokens plus context.
    _FORCE_TAIL_WINDOW = 8

    def __init__(
        self,
        budget: int,
        think_end_token: int,
        think_start_token: int,
        model_family: str = "chatml",
        bare_open_tokens: tuple[int, ...] | list[int] | None = None,
        tokenizer=None,
    ):
        self.budget = budget
        self.think_end_token = think_end_token
        self.think_start_token = think_start_token
        self.model_family = model_family
        # U22: optional tokenizer enabling the UTF-8 boundary deferral of the
        # forced close (None = historical immediate-force behaviour). Round 2:
        # the deferral state lives in the SHARED gate (also used by the MTP
        # clone's force site); _force_deferrals stays exposed as a property.
        self._tokenizer = tokenizer
        self._defer_gate = ForceDeferralGate(
            tokenizer,
            max_deferrals=self._FORCE_DEFER_MAX,
            window=self._FORCE_TAIL_WINDOW,
        )
        # Token-id sequence of the BARE ``thought\n`` opener (gemma4 sliding-
        # window variant with no ``<|channel>`` token). Empty/None disables the
        # bare path entirely (identical to the original <|channel>-only behaviour).
        self.bare_open_tokens = tuple(bare_open_tokens) if bare_open_tokens else ()
        self.thinking_tokens = 0
        self.in_thinking = model_family != "gemma4"
        # Gen-start lookbehind state for the multi-token bare opener (mirrors
        # ``advance_think_state``). ``_content_seen`` latches True once any token
        # that is not part of an in-progress bare match is emitted, after which
        # the bare opener can no longer fire (ThinkingRouter FIX 4 parity).
        self._bare_idx = 0
        self._content_seen = self.in_thinking
        # PROMPT/GENERATION DOMAIN SPLIT (codex HIGH fix): mlx-vlm/mlx-lm call
        # this processor with the RUNNING ``tokens`` array, and the prefill step
        # folds the prompt tail (the chunked-prefill last token, or the WHOLE
        # prompt when unchunked) into ``tokens`` BEFORE running the processor —
        # so the first ``__call__`` sees prompt token(s) as ``last_token``. The
        # bare ``thought\n`` opener is a GENERATION-START phenomenon: a prompt
        # token mismatching the bare opener would latch ``_content_seen`` and the
        # GENERATED bare opener could then never fire (the long-context runaway).
        # ``_prompt_len`` captures the prompt boundary on the FIRST call:
        # ``tokens.size`` then == the count of prompt tokens visible to the
        # processor; every later token (index >= _prompt_len) is GENERATED. Bare
        # matching is gated to generated tokens, and the bare sub-state is RESET
        # at the boundary (iff outside thinking) so a generated bare opener can
        # open even though prompt tokens preceded it. Full ``<|channel>``
        # open/close + in-thinking counting still run on prompt tokens (parity
        # with the MTP seed fold using ``bare_open_tokens=()``).
        self._prompt_len: int | None = None
        self._bare_boundary_reset_done = False

    @property
    def _force_deferrals(self) -> int:
        """Deferral count (delegates to the shared gate — kept as an
        attribute-compatible surface for existing callers/tests)."""
        return self._defer_gate.deferrals

    @_force_deferrals.setter
    def _force_deferrals(self, value: int) -> None:
        self._defer_gate.deferrals = int(value)

    def _utf8_tail_incomplete(self, tokens: mx.array) -> bool:
        """U22: whether the emitted byte stream currently ends mid multi-byte
        UTF-8 character. Round 2: delegates to the shared byte-tail check
        (``tail_ends_mid_utf8``) — a genuine literal U+FFFD is NOT treated
        as incomplete. Fails open (False → force immediately, the historical
        behaviour) on any problem."""
        if self._tokenizer is None:
            return False
        try:
            n = int(tokens.size)
            if n == 0:
                return False
            start = max(0, n - self._FORCE_TAIL_WINDOW)
            tail_ids = [int(t) for t in tokens[start:].tolist()]
            return tail_ends_mid_utf8(
                self._tokenizer, tail_ids, self._FORCE_TAIL_WINDOW
            )
        except Exception:  # noqa: BLE001 — the deferral is best-effort only
            return False

    def _apply_forced_close(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """U22: the budget is reached — force ``think_end`` UNLESS the byte
        stream ends mid-character, in which case skip the override for this
        step (bounded: after ``_FORCE_DEFER_MAX`` skips, force regardless).
        Only the logit override is deferred; all counters advance normally."""
        try:
            n = int(tokens.size)
            start = max(0, n - self._FORCE_TAIL_WINDOW)
            tail_ids = [int(t) for t in tokens[start:].tolist()] if n else []
        except Exception:  # noqa: BLE001 — fail open: force immediately
            tail_ids = []
        if tail_ids and self._defer_gate.should_defer(tail_ids):
            return logits
        logits[:, self.think_end_token] = 1e9
        return logits

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        last_token = tokens[-1].item() if tokens.size > 0 else -1

        # Capture the prompt boundary on the first call; reset the bare
        # sub-state at the generation boundary (first GENERATED token) so the
        # prompt tail never latches the bare matcher (see __init__ note).
        if self._prompt_len is None:
            self._prompt_len = int(tokens.size)
        # ``last_token`` is GENERATED iff its index (tokens.size - 1) is at or
        # past the prompt boundary, i.e. tokens.size > _prompt_len.
        last_is_generated = int(tokens.size) > self._prompt_len
        if last_is_generated and not self._bare_boundary_reset_done:
            self._bare_boundary_reset_done = True
            if self.model_family == "gemma4" and not self.in_thinking:
                # Generation boundary: re-arm the bare matcher. ``in_thinking``
                # already reflects any full ``<|channel>`` markers folded over
                # the prompt; only re-arm the bare gen-start sub-state.
                self._bare_idx = 0
                self._content_seen = False

        if self.in_thinking and last_token == self.think_end_token:
            # Close the current block and reset the per-cycle counter; do NOT
            # latch — a later think_start re-opens and re-budgets.
            self.in_thinking = False
            self.thinking_tokens = 0
            self._bare_idx = 0
            self._content_seen = True
            # U22: re-arm the deferral budget for a potential later block.
            self._force_deferrals = 0
            return logits

        if not self.in_thinking and last_token == self.think_start_token:
            # Re-open thinking; the start token counts as token #1 of the block.
            self.in_thinking = True
            self.thinking_tokens = 0
            self._bare_idx = 0
            self._content_seen = True

        # BARE ``thought\n`` opener (gemma4 long-context): match the token-id
        # sequence at GENERATION START ONLY (not in_thinking, no content seen,
        # and only over GENERATED tokens — never the prompt tail, which would
        # otherwise latch ``_content_seen`` and defeat the opener; see __init__).
        if (
            last_is_generated
            and not self.in_thinking
            and self.bare_open_tokens
            and not self._content_seen
            and last_token != -1
        ):
            if last_token == self.bare_open_tokens[self._bare_idx]:
                self._bare_idx += 1
                if self._bare_idx >= len(self.bare_open_tokens):
                    # Full bare opener matched: open thinking; the matched opener
                    # tokens count as the first thinking tokens of the block.
                    self.in_thinking = True
                    self.thinking_tokens = len(self.bare_open_tokens)
                    self._bare_idx = 0
                    self._content_seen = True
                    if self.thinking_tokens >= self.budget:
                        return self._apply_forced_close(tokens, logits)
                    return logits
                # Partial match still at gen-start: nothing emitted as content,
                # keep matching on the next call.
                return logits
            # Mismatch: real content at gen-start; the bare opener can never fire.
            self._bare_idx = 0
            self._content_seen = True

        if self.in_thinking:
            self.thinking_tokens += 1
            if self.thinking_tokens >= self.budget:
                return self._apply_forced_close(tokens, logits)

        return logits


class RepetitionPenaltyProcessor:
    """
    Logits processor that applies repetition penalty to previously generated tokens.

    penalty > 1.0 discourages repetition, < 1.0 encourages it, 1.0 = no effect.
    """

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size == 0 or self.penalty == 1.0:
            return logits
        # Get unique token IDs from generated sequence
        selected_logits = logits[:, tokens]
        # Apply penalty: divide positive logits, multiply negative logits
        selected_logits = mx.where(
            selected_logits > 0,
            selected_logits / self.penalty,
            selected_logits * self.penalty,
        )
        logits[:, tokens] = selected_logits
        return logits
