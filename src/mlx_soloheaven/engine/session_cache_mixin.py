"""SessionCacheMixin — session-cache matcher / fingerprint cluster.

Behavior-preserving extraction from mlx_engine.py. ``MLXEngine`` inherits
this mixin, so every ``self.*`` call and every ``MLXEngine.<staticmethod>``
call site keeps resolving via inheritance — no test edits required. This
module must NEVER import mlx_engine; the dependency arrow is

    mlx_engine -> session_cache_mixin -> session_cache_types

Intra-body references to this mixin's OWN moved staticmethods are written
``SessionCacheMixin._X`` (not ``MLXEngine._X``) so the module never names
MLXEngine.
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import (
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)
from mlx_vlm.generate import PromptCacheState

from mlx_soloheaven.engine.session_cache_types import (
    BaseCacheEntry,
    SessionState,
    _NORMALIZE_RE_IMAGE_REMOVED,
    _NORMALIZE_RE_SYSTEM_REMINDER,
    _NORMALIZE_RE_TODAYS_DATE,
    _NORMALIZE_RE_TOOL_CALL_CHANNEL,
    _NORMALIZE_RE_TOOL_CALL_XML,
    _content_channel_union,
)
from mlx_soloheaven.engine.tool_parser import (
    _parse_glm_tool_calls,
    get_tool_markers,
    parse_tool_calls,
    split_thinking_and_content,
)

# Bind to the ENGINE's logging channel (not __name__) so the moved DEBUG
# lines keep emitting under the exact same logger name as before the move —
# strictly behavior-preserving for log output.
logger = logging.getLogger("mlx_soloheaven.engine.mlx_engine")


class SessionCacheMixin:
    """Session-cache matcher, prompt-fingerprint, and (Pass A chunk 3)
    base-cache-pool helpers extracted from MLXEngine (behavior-preserving)."""

    # --- Prompt contract (U3/U21) ----------------------------------------

    @staticmethod
    def _canonical_tools(tools: list | None) -> list | None:
        """Canonical (JSON-serializable) form of a request's tool schema.

        Pydantic models are dumped to plain dicts so the result can be
        stored on SessionState, persisted in safetensors metadata, and
        hashed deterministically. ``None``/empty → ``None`` (toolless)."""
        if not tools:
            return None
        return [t.model_dump() if hasattr(t, "model_dump") else t for t in tools]

    @staticmethod
    def _prompt_fingerprint(
        tools_canonical: list | None, thinking: bool, template_rev: str = "",
    ) -> str:
        """Hash of everything that alters the tokenized prompt prefix
        OUTSIDE the messages: the canonical tool schema + the thinking flag.
        Compared on every HIT (U21) — a mismatch means the cached prefix was
        built under a different contract and reuse would silently keep the
        stale schema in context, so the turn takes an honest MISS instead.

        ``template_rev`` (GLM natural-EOS fix) is the suffix-builder revision
        for families whose builder semantics changed: sessions whose stored
        token_ids were spliced by an OLD builder must not mix old and new
        renderings in one KV, so the revision participates in the contract
        hash and the established U21/F5 gate demotes every pre-fix session
        to ONE honest MISS rebuild that restamps the new contract. The key
        is omitted when empty so families without a revision (chatml,
        gemma4) keep their historical fingerprints — no spurious rebuilds."""
        payload_obj: dict = {"tools": tools_canonical, "thinking": bool(thinking)}
        if template_rev:
            payload_obj["template_rev"] = template_rev
        payload = json.dumps(
            payload_obj,
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # GLM suffix-builder revision (bump on any change to what
    # _suffix_tokens_glm / the GLM HIT reconciliation splices): v2 = official
    # GLM-4.7-Flash template semantics — tool turns open with
    # ``<|observation|>`` (one marker per run, no newline padding),
    # thinking=False generation prompt renders ``<|assistant|></think>``,
    # and the recorded chat-EOS tail is reconciled at HIT time
    # (_reconcile_glm_recorded_eos). Pre-v2 glm sessions spliced drifted
    # token streams (doubled ``<|user|>`` after a natural stop, ``<|user|>``
    # where the template renders ``<|observation|>``, missing ``</think>``)
    # — their stored ids already diverged from full retokenization, so the
    # only honest continuation is a one-time MISS rebuild (see
    # _prompt_fingerprint's template_rev doc).
    _GLM_SUFFIX_REV = "glm-suffix-v2"

    def _suffix_template_rev(self) -> str:
        """Suffix-builder revision that participates in the prompt-contract
        fingerprint for the current model family ('' = no revision)."""
        if getattr(self, "model_family", None) == "glm":
            return self._GLM_SUFFIX_REV
        return ""

    # --- Base cache pool ---

    @staticmethod
    def _system_hash(messages: list[dict], tools: list | None = None) -> str | None:
        """Hash the first system message (+ tools) for base cache lookup.

        Returns a hash even when there's no explicit system message — uses empty
        string as content. This supports models where the template auto-generates
        a system prefix (e.g. Gemma 4).
        """
        if messages and messages[0].get("role") in ("system", "developer"):
            content = messages[0].get("content", "")
        else:
            # No explicit system message — use empty content as hash key
            # (template may still generate a system prefix)
            content = ""
        h = hashlib.sha256(content.encode())
        if tools:
            h.update(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _is_compacted_tool(s_content: str, i_content: str) -> bool:
        """Check if either side is a compacted/cleared tool result placeholder."""
        for c in (s_content, i_content):
            if c.startswith("[") and ("cleared]" in c or "compacted:" in c):
                return True
        return False

    @staticmethod
    def _flatten_multipart(content) -> str:
        """Flatten OpenAI multi-part content to a plain string.

        Drops image/video parts and client-inserted
        "[image data removed ...]" placeholders so that a turn with an
        image blob and a subsequent turn where the client replaced the
        blob with a placeholder normalize to the same text.
        """
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if not isinstance(content, list):
            return str(content)
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype and ptype != "text":
                continue  # image, image_url, video, etc.
            txt = p.get("text", "") or ""
            if _NORMALIZE_RE_IMAGE_REMOVED.match(txt):
                continue
            parts.append(txt)
        return "\n".join(parts)

    @staticmethod
    def _normalize_for_match(
        content, role: str, model_family: str = "chatml",
        thinking_active: bool = True,
    ) -> str:
        """Normalize message content for comparison.

        Codex round 9, finding 2: the assistant thinking-channel reduction
        follows the authoritative router machinery under the message's
        ``thinking_active`` contract (threaded from _messages_match — the
        round-3/8 threading covered the tool-call helpers but left this one
        flag-blind and suffix-only):

        - gemma4: the content channel is the MULTI-CYCLE union of all
          content segments (_content_channel_union). The old
          last-``<channel|>`` slice matched suffix-only (multi-cycle stored
          'thought → important → thought → done' wrong-HIT against a bare
          incoming 'done'), and the unconditional bare ``thought\\n`` strip
          let a thinking-DISABLED stored turn genuinely starting with those
          words ('thought\\nSECRET<channel|>answer' — all content under the
          router's enable_thinking gate) wrong-HIT against 'answer'.
        - chatml/glm, thinking active: FIRST-close reduction under the
          router contract (codex round 11, finding 1 — the router never
          re-enters thinking, tool_parser.content_segments). A raw side
          (leading ``<think>`` — the engine always stores the opener, see
          _make_full_assistant_content) reduces to the text after the
          FIRST ``</think>``; any later ``</think>`` is a literal quote
          INSIDE the content channel. An already-split side (no leading
          opener) is kept WHOLE — the old LAST-close (rindex) reduction
          collapsed distinct content channels onto their final suffix, so
          a forged plain resend of that suffix wrong-HIT even on the
          strict anon path. The degenerate UNCLOSED shape (leading opener,
          no close) is kept RAW — codex round 13, finding 1; see the
          inline comment.
        - chatml/glm, thinking DISABLED: pure pass-through — a literal
          ``</think>`` in content is a quote, never a boundary (round 3,
          finding 4 semantics, previously missing here).
        """
        content = SessionCacheMixin._flatten_multipart(content)
        if role == "system":
            # Normalize dynamic date (e.g. "Today's date: Tue Mar 10 2026" → placeholder)
            content = _NORMALIZE_RE_TODAYS_DATE.sub(
                "Today's date: __DATE__",
                content,
            )
        # Strip <system-reminder>...</system-reminder> tags injected dynamically by clients
        content = _NORMALIZE_RE_SYSTEM_REMINDER.sub(
            "",
            content,
        )
        # Strip thinking and tool calls from assistant messages for comparison.
        # Only the actual text content matters for cache matching.
        if role == "assistant":
            # Strip thinking blocks (channel reduction per family/contract —
            # see the docstring).
            if model_family == "gemma4":
                content = _content_channel_union(
                    content, model_family, thinking_active,
                )
            elif thinking_active:
                # Codex round 11, finding 1: FIRST-close channel reduction.
                # The router contract (content_segments /
                # split_thinking_and_content) makes the FIRST ``</think>``
                # authoritative — chatml/glm never re-enter thinking, so any
                # later ``</think>`` is a literal quote INSIDE the content
                # channel (both families share the <think> markers). Shape
                # alignment:
                # - leading ``<think>``: raw wire text (the engine always
                #   stores the opener — _make_full_assistant_content);
                #   content is everything after the FIRST close, quotes
                #   included. No close at all → kept RAW (codex round 13,
                #   finding 1; see below).
                # - no leading opener: the text IS a content channel — a
                #   literal ``</think>`` in it is a quote, never a boundary.
                #   Kept WHOLE: shapes that cannot be aligned one-to-one
                #   against a stored raw (e.g. a bare leading ``</think>``
                #   with no open) must never normalize onto a plain resend
                #   — an honest MISS, not a forgeable suffix HIT (the old
                #   rindex reduction collapsed distinct content channels
                #   onto their final suffix).
                body = content.lstrip()
                if body.startswith("<think>"):
                    close = body.find("</think>", len("<think>"))
                    if close == -1:
                        # Codex round 13, finding 1: an UNCLOSED raw (leading
                        # opener, no close) is kept RAW — returned verbatim
                        # (outer whitespace trim only), skipping every
                        # reduction including the tool-call strips below.
                        # Under an active chatml/glm contract the router
                        # routes an unclosed stream ENTIRELY to reasoning
                        # with EMPTY content, so the legacy bare-opener strip
                        # normalized stored '<think>SECRET' equal to a plain
                        # incoming 'SECRET' — a forgeable suffix even on the
                        # strict/anon path. Normalizing to '' (the router-
                        # faithful content channel) would instead collide
                        # with genuinely-empty assistant content — another
                        # many-to-one. The legitimate interrupted-resend
                        # equivalence is handled by _interrupted_resend_equiv
                        # (which splits the channel itself under exact
                        # rules), never by plain equality: an unclosed-
                        # thinking turn must normalize equal to nothing but
                        # its byte-identical self.
                        return content.strip()
                    content = body[close + len("</think>"):]
            # Strip tool call blocks (both ChatML and Gemma 4 formats)
            content = _NORMALIZE_RE_TOOL_CALL_CHANNEL.sub("", content)
            content = _NORMALIZE_RE_TOOL_CALL_XML.sub("", content)
        return content.strip()

    @staticmethod
    def _canonicalize_calls(calls: list) -> list:
        """Canonical ``[(name, canonical-args-json), ...]`` form of a list
        of OpenAI-shaped tool-call dicts. JSON-string arguments are parsed
        so a re-serialized resend (key order / whitespace) still compares
        equal. Call ids are deliberately EXCLUDED: XML-parsed calls get
        fresh random ids, and the tool ROLE's tool_call_id comparison
        already pins the id chain; name + canonical arguments are what make
        two calls "the same call" for cache validity."""
        out: list = []
        for tc in calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            out.append((
                fn.get("name") or "",
                json.dumps(args, sort_keys=True, ensure_ascii=False, default=str),
            ))
        return out

    @staticmethod
    def _content_xml_calls_for_match(
        raw_content: str, model_family: str, thinking_active: bool = True,
    ) -> list:
        """Canonical calls parsed from tool-call XML embedded in RAW content
        (legacy engine-stored turns kept the template XML in content) —
        empty list when no start marker is present or nothing parses.

        This is THE content-side parse chain, shared by
        _tool_calls_for_match (legacy fallback), _has_residual_tool_markers
        (N3) and _structured_content_call_conflict (round 4): the family
        parser first, then the N2 bare-name GLM canonicalization for a
        COMPLETE block the family parser could not read
        (``<tool_call>name[<arg_key>..</arg_value>]</tool_call>`` — a shape
        some clients replay verbatim), so a legitimate bare-name turn
        compares STRUCTURALLY instead of hitting the one-sided FAIL. A
        PARTIAL block (no end marker after the start) stays unparseable on
        purpose: against a side with >=1 canonical call it is mismatch
        evidence, not indeterminate.

        Round 5: matching requires CLOSED blocks. The GLM parser
        intentionally tolerates a MISSING ``</tool_call>`` closer (the
        ``\\Z`` alternate — generation-time robustness for a stream cut
        mid-block), so an UNCLOSED trailing GLM block canonicalized exactly
        like the closed one — equal to the structured call it mirrors — and
        sailed past the one-sided (N2), residual-marker (N3) AND
        structured-content conflict (round 4) checks straight into the
        marker-strip/wildcard acceptance, despite the tokenized prompts
        differing. The parser's leniency is deliberately untouched (do not
        change generation-time parsing); at MATCH time the lenient parse is
        DISCOUNTED: at most one call can come from an unclosed block (only
        the final ``\\Z``-terminated match) and it is always the LAST parsed
        call, so when more calls parsed than CLOSED
        ``<tool_call>...</tool_call>`` spans exist the trailing call is
        dropped. The dropped call then surfaces through the existing checks
        (one-sided / residual marker / self-conflict). A spurious drop can
        only reject (honest MISS), never produce a wrong HIT.

        U12: only the CONTENT channel is scanned — tool-call XML inside a
        thinking segment is a rehearsal the engine no longer records, so at
        match time it must not count as an extractable call (or a residual
        marker) either; otherwise every turn after a rehearsal would FAIL
        the one-sided check against the client's clean resend. Round 2: the
        channel reduction is the UNION of all content segments
        (_content_channel_union) — gemma4 multi-cycle output keeps a
        legitimate call in an EARLIER content cycle visible here, while
        thinking segments stay excluded. Round 3 (finding 4):
        ``thinking_active`` is the router-active state the message's turn
        ran under (the session's stored thinking flag) — with thinking
        disabled a literal ``</think>`` in content is a quote, not a
        boundary, so a call BEFORE it stays extractable."""
        raw_content = _content_channel_union(
            raw_content or "", model_family, thinking_active,
        )
        start_tag, end_tag = get_tool_markers(model_family)
        if not raw_content or start_tag not in raw_content:
            return []
        _, calls = parse_tool_calls(raw_content, model_family=model_family)
        if (
            not calls
            and start_tag == "<tool_call>"
            and end_tag in raw_content[raw_content.find(start_tag):]
        ):
            _, calls = _parse_glm_tool_calls(raw_content)
        if calls and start_tag == "<tool_call>":
            # Round 5 closed-block discount (see docstring). The closed-span
            # count uses the same non-greedy left-to-right scan as the GLM
            # block pattern, so an orphan closer BEFORE the first start
            # marker cannot masquerade as a closer for a trailing unclosed
            # block (a naive closers-vs-starts count would miss that).
            closed = sum(
                1 for _ in _NORMALIZE_RE_TOOL_CALL_XML.finditer(raw_content)
            )
            if len(calls) > closed:
                calls = calls[:-1]
        return SessionCacheMixin._canonicalize_calls(calls)

    @staticmethod
    def _tool_calls_for_match(
        msg: dict, raw_content: str, model_family: str,
        thinking_active: bool = True,
    ) -> list | None:
        """F3: canonical, comparable form of an assistant message's tool
        calls — ``[(name, canonical-args-json), ...]`` or ``None`` when no
        call is extractable.

        Sources, in order:
        - the structured ``tool_calls`` field (OpenAI clients / the engine's
          normal save);
        - tool-call XML embedded in the RAW content (legacy engine-stored
          turns kept the template XML in content) via
          _content_xml_calls_for_match.

        When the structured field is present it WINS and content XML is not
        consulted here — _structured_content_call_conflict (round 4) is the
        companion check that rejects a message whose content XML DISAGREES
        with its structured list."""
        out = SessionCacheMixin._canonicalize_calls(msg.get("tool_calls") or [])
        if out:
            return out
        # Fall back to tool-call XML in the raw content (legacy stored shape).
        return (
            SessionCacheMixin._content_xml_calls_for_match(
                raw_content, model_family, thinking_active,
            )
            or None
        )

    @staticmethod
    def _structured_content_call_conflict(
        msg: dict, raw_content: str, model_family: str,
        thinking_active: bool = True,
    ) -> bool:
        """Round 4: True when a message carries BOTH tool-call
        representations — a structured ``tool_calls`` field AND tool-call
        start markers in its RAW content — and they do NOT agree
        canonically.

        _tool_calls_for_match returns the STRUCTURED list as soon as it is
        non-empty, ignoring content entirely, so a side holding the same
        structured call but content that ALSO embeds a fully parseable
        ``<tool_call>`` block for a DIFFERENT call (e.g. delete_files)
        compared canonically EQUAL to a side with just the structured call;
        _has_residual_tool_markers stayed False (every marker parses) and
        the marker-strip content shortcut in _messages_match then accepted
        the message on both the strict and the lenient path — despite the
        tokenized prompts differing.

        When both representations exist they must agree: the canonical
        SEQUENCE of content-XML calls must equal the canonical structured
        list. A start marker whose block fails to parse counts as
        disagreement (same rule as the one-sided garbled block, N2). A
        self-consistent turn — the same call(s) rendered both ways, or
        either representation alone — never trips this; a false positive
        (a literal marker inside prose next to a structured call) degrades
        to an honest MISS, never a wrong HIT."""
        structured = SessionCacheMixin._canonicalize_calls(msg.get("tool_calls") or [])
        if not structured:
            return False
        # U12: a marker inside a thinking segment is a rehearsal, not a
        # conflicting representation — only content-channel XML must agree
        # with the structured list. Round 2: the channel is the UNION of all
        # content segments (multi-cycle aware); the RAW content is handed to
        # _content_xml_calls_for_match, which applies the same reduction
        # itself (the reduction is not idempotent for chatml text quoting a
        # literal ``</think>`` in its content, so it must run exactly once).
        content_union = _content_channel_union(
            raw_content or "", model_family, thinking_active,
        )
        start_tag, _ = get_tool_markers(model_family)
        if not content_union or start_tag not in content_union:
            return False
        return (
            SessionCacheMixin._content_xml_calls_for_match(
                raw_content or "", model_family, thinking_active,
            )
            != structured
        )

    @staticmethod
    def _has_residual_tool_markers(
        raw_content: str, model_family: str, thinking_active: bool = True,
    ) -> bool:
        """Round 3 (N3): True when ``raw_content`` holds tool-call START
        markers the family parse chain does NOT consume into canonical calls
        (a partial/garbled block trailing — or preceding — valid blocks).

        _tool_calls_for_match returns only the calls that PARSE, so a side
        with one valid block plus a residual ``<tool_call>`` fragment compares
        canonically EQUAL to a side with just the valid block — and the
        marker-strip content shortcut in _messages_match then accepts the
        message on both the strict and the lenient path. Residual markers are
        mismatch evidence (same rule as the one-sided garbled block, N2).

        Detection mirrors _tool_calls_for_match's exact parse chain (family
        parser, then the N2 bare-name GLM fallback) and compares the number
        of canonical calls against the number of start-marker occurrences.
        Counting the WHOLE content is consistent with existing semantics: the
        family parsers scan the full text (a marker inside a code fence /
        quote is already treated as tool-call territory by parse_tool_calls),
        and each parsed block consumes exactly one start marker. A false
        positive (e.g. a parameter VALUE containing the literal marker)
        degrades to an honest MISS, never a wrong HIT — and only when the two
        sides' raw contents already differ (byte-identical sides skip this
        check entirely).

        U12: markers inside thinking segments are rehearsals, not
        residuals — only the content channel is counted. Round 2: the
        channel is the UNION of all content segments (mirrors the reduction
        in _content_xml_calls_for_match, so parsed-call and marker counts
        stay aligned; the RAW content is handed down so the reduction runs
        exactly once)."""
        content_union = _content_channel_union(
            raw_content or "", model_family, thinking_active,
        )
        start_tag, _ = get_tool_markers(model_family)
        if not content_union or start_tag not in content_union:
            return False
        calls = SessionCacheMixin._content_xml_calls_for_match(
            raw_content or "", model_family, thinking_active,
        )
        return len(calls) < content_union.count(start_tag)

    def _interrupted_resend_equiv(
        self, s_content: str, i_norm: str, *, exact: bool = False,
        thinking_active: bool = True,
    ) -> bool:
        """U1 narrow equivalence for a C1-committed INTERRUPTED assistant turn.

        The stored message (marked ``interrupted=True`` by
        _commit_interrupted_hit_turn) holds the engine-side content INCLUDING
        thinking; the client resends only what it received on the wire —
        the content channel (thinking stripped), possibly TRUNCATED at the
        cancel point, or the thinking reconstructed as plain text.

        ``exact=False`` (EXPLICIT session ids only — F2): accept exactly
        these shapes and nothing else:

        - incoming (normalized) is a PREFIX of the stored CONTENT channel
          (includes the empty resend — a cancel before any content frame);
        - incoming (normalized) is a PREFIX of
          ``{stored thinking}\\n\\n{stored content}`` — the client replayed
          the (possibly truncated) thinking as plain content.

        ``exact=True`` (the ANON resolver / strict path — F2): prefix
        equivalence is FORGEABLE for session-less requests (an empty resend
        would match every interrupted turn, and any shared prefix could
        select another conversation's session), so anon resolution requires
        EXACT content-channel equality after thinking-strip normalization —
        an empty resend matches ONLY an empty stored content channel.

        Arbitrary divergence — anything that is not (a prefix of, or under
        ``exact`` equal to) what was actually streamed — does NOT match
        (honest MISS / new anon session), unlike the old
        last-stored-assistant wildcard this replaces.

        ``thinking_active`` (codex round 9, finding 2): the stored turn's
        thinking contract, threaded into the split AND the normalization —
        the default-active split treated a thinking-DISABLED gemma4 turn's
        bare ``thought\\n`` content as a reasoning span, so its content
        channel shrank to the post-``<channel|>`` suffix and a forged bare
        resend passed even the EXACT (anon/strict) rule."""
        s_content = s_content or ""
        family = getattr(self, "model_family", "chatml")
        started = s_content.lstrip().startswith("<think>")
        if family == "gemma4" or (thinking_active and started):
            # gemma4 keeps its positional-router split; a chatml/glm stored
            # raw always carries the leading opener
            # (_make_full_assistant_content), and the split's Case 1 is
            # already FIRST-close (non-greedy) — any later ``</think>`` in
            # the resulting content is a literal quote it preserves.
            thinking, content = split_thinking_and_content(
                s_content,
                model_family=family,
                started_in_thinking=started,
                thinking_active=thinking_active,
            )
        else:
            # Codex round 11, finding 1: no leading opener (or thinking
            # disabled) — the stored text IS the content channel and a
            # literal ``</think>`` in it is a quote, never a boundary. The
            # old unconditional split reduced it at the close, so a forged
            # resend of the post-quote suffix passed even the EXACT
            # (anon/strict) rule. Kept whole → honest MISS instead.
            thinking, content = None, s_content
        c_norm = self._normalize_for_match(
            content or "", "assistant", family, thinking_active,
        )
        if exact:
            return c_norm == i_norm
        if c_norm.startswith(i_norm):
            return True
        if thinking:
            t_norm = thinking.strip()
            combo = f"{t_norm}\n\n{c_norm}".strip() if c_norm else t_norm
            if combo.startswith(i_norm):
                return True
        return False

    def _messages_match(
        self,
        stored: list[dict],
        incoming: list[dict],
        *,
        last_assistant_wildcard: bool = True,
        thinking_active: bool = True,
    ) -> bool:
        """Check if incoming messages start with the stored conversation.

        ``last_assistant_wildcard`` (U1/F2): when True (explicit-session HIT
        path — the historical behavior), a content mismatch on the LAST
        stored assistant message is tolerated wholesale (client may hold a
        truncated/reformatted view of the reply), interrupted turns get the
        NARROW prefix equivalence, and the thinking-prepend suffix leniency
        applies. The anon prefix RESOLVER passes False — STRICT mode: for
        session-less requests every one of those leniencies is a forgery
        vector (a DIFFERENT anonymous conversation — prefix-equal except an
        assistant turn — could hijack another conversation's session and
        cache), so anon resolution requires EXACT normalized assistant
        content; an interrupted turn matches only on exact content-channel
        equality (empty resend == empty stored content only) and the
        generic suffix equivalence is bypassed entirely.

        Structured fields (F3) are compared on BOTH paths whenever present:
        assistant ``tool_calls`` (canonically serialized name+arguments,
        incl. legacy XML-in-content stored turns) and the tool role's
        ``tool_call_id`` — two turns with identical (even empty) content but
        different calls are different conversations.

        Round-2 hardening (N1/N2):
        - N1: the compacted/cleared tool-result equivalence ("[cleared]" /
          "[compacted:…]" placeholders) applies to EXPLICIT sessions only —
          in strict (anon) mode a placeholder would match ANY stored tool
          result, so exact tool-result content is required.
        - N2: when tool calls are extractable on exactly ONE side, the
          message FAILS on both paths — merely containing a start marker no
          longer defers to the content rules (the marker-strip shortcut
          could accept a valid stored call against a partial/different
          block). Complete bare-name blocks are canonicalized first (see
          _tool_calls_for_match), so only genuine parse failures reject.

        Round-3 hardening (N3): RESIDUAL unparsed tool-call start markers
        (a valid call followed by a partial/different block — canonically
        equal to the valid call alone) are mismatch evidence when the two
        sides' raw contents differ: FAIL on both paths (see
        _has_residual_tool_markers).

        Round-4 hardening: a message carrying BOTH representations —
        structured ``tool_calls`` AND tool-call XML in content — must agree
        with itself; a conflicting side (the structured comparison prefers
        the field, so content XML for a DIFFERENT call was invisible) FAILS
        on both paths (see _structured_content_call_conflict).

        Codex round 3, finding 4: ``thinking_active`` is the session's
        stored thinking contract — threaded into the content-channel
        reduction of every helper below so a literal ``</think>`` in a
        NON-thinking turn's content is a quote, never a channel boundary
        that hides tool-call XML from matching. Callers pass the STORED
        session's flag (the HIT gate only reaches here after the
        fingerprint check, so it equals the request's flag there; the anon
        resolver passes each candidate's own flag)."""
        if len(incoming) < len(stored):
            logger.debug(
                f"[Match] FAIL: incoming({len(incoming)}) < stored({len(stored)})"
            )
            return False
        for i, s_msg in enumerate(stored):
            i_msg = incoming[i]
            if s_msg.get("role") != i_msg.get("role"):
                # DEBUG (not INFO): _messages_match runs PER CANDIDATE session
                # in the anon prefix scan, so per-candidate FAILs are expected
                # noise; the aggregate "no prefix match among N sessions" line
                # stays at INFO.
                logger.debug(
                    f"[Match] FAIL at msg[{i}]: role {s_msg.get('role')!r} != {i_msg.get('role')!r}"
                )
                return False
            s_content = self._flatten_multipart(s_msg.get("content"))
            i_content = self._flatten_multipart(i_msg.get("content"))
            role = s_msg.get("role", "")

            # F3: structured fields participate in matching on ALL paths —
            # content comparison alone lets two assistant turns with equal
            # (often empty) content but DIFFERENT tool calls match, poisoning
            # context across conversations (A's cached call + B's tool
            # result). Compared via the class staticmethod directly so
            # lightweight test stubs without the helper bound still work.
            family = getattr(self, "model_family", "chatml")
            if role == "assistant":
                s_calls = SessionCacheMixin._tool_calls_for_match(
                    s_msg, s_content, family, thinking_active,
                )
                i_calls = SessionCacheMixin._tool_calls_for_match(
                    i_msg, i_content, family, thinking_active,
                )
                if s_calls is not None and i_calls is not None:
                    if s_calls != i_calls:
                        logger.debug(
                            f"[Match] FAIL at msg[{i}]: assistant tool_calls "
                            f"differ ({s_calls} != {i_calls})"
                        )
                        return False
                elif s_calls != i_calls:
                    # Exactly one side carries extractable calls (a non-None
                    # result always holds >=1 canonical call) — FAIL on BOTH
                    # paths (N2). This is mismatch evidence, never
                    # indeterminate:
                    # - the bare side shows NO tool-call evidence at all (no
                    #   structured field, no start marker) — structurally
                    #   different conversations (the historical rule); or
                    # - it carries a start marker whose block FAILS to parse
                    #   (even after the bare-name canonicalization fallback
                    #   in _tool_calls_for_match) against >=1 canonical call
                    #   on the other side. Pre-fix, merely containing the
                    #   marker fell through and the marker-strip shortcut
                    #   below accepted a VALID stored call against an
                    #   arbitrary partial/different <tool_call> block — a
                    #   session-forgery vector on the STRICT (anon) path and
                    #   cross-conversation poisoning on the lenient one. A
                    #   legitimate resend that still fails to canonicalize
                    #   degrades to an honest MISS, never a wrong HIT.
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: tool_calls extractable on "
                        f"one side only (stored="
                        f"{'yes' if s_calls is not None else 'no'}, incoming="
                        f"{'yes' if i_calls is not None else 'no'})"
                    )
                    return False
                # N3 (round 3): RESIDUAL unparsed tool-call start markers.
                # _tool_calls_for_match returns only the calls that PARSE, so
                # a side with one VALID call plus a trailing partial/different
                # <tool_call> fragment compared canonically EQUAL to a side
                # with just the valid call — and the marker-strip shortcut
                # below then accepted on BOTH the strict and the lenient
                # path. A residual marker on a side whose raw content differs
                # from the other side is mismatch evidence (same rule as the
                # one-sided garbled block above): FAIL on both paths.
                # Byte-identical contents carry no divergence to hide, so
                # verbatim replays of a degenerate turn still match.
                if s_content != i_content and (
                    SessionCacheMixin._has_residual_tool_markers(
                        s_content, family, thinking_active,
                    )
                    or SessionCacheMixin._has_residual_tool_markers(
                        i_content, family, thinking_active,
                    )
                ):
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: residual unparsed "
                        f"tool-call marker(s) on a differing side "
                        f"(stored_len={len(s_content)}, "
                        f"incoming_len={len(i_content)})"
                    )
                    return False
                # Round 4: a side carrying BOTH representations (structured
                # tool_calls AND tool-call XML in content) must agree with
                # ITSELF. The canonical comparison above prefers the
                # structured field, so a side whose content ALSO holds a
                # fully parseable block for a DIFFERENT call (no residual —
                # every marker parses) compared EQUAL, and the marker-strip
                # shortcut below then accepted it on the strict AND the
                # lenient path despite the tokenized prompts differing. A
                # self-conflicting side is mismatch evidence on BOTH paths;
                # content XML that canonically AGREES with the structured
                # list (the same call rendered both ways) never trips this.
                if (
                    SessionCacheMixin._structured_content_call_conflict(
                        s_msg, s_content, family, thinking_active,
                    )
                    or SessionCacheMixin._structured_content_call_conflict(
                        i_msg, i_content, family, thinking_active,
                    )
                ):
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: structured tool_calls "
                        f"conflict with tool-call XML in content on at "
                        f"least one side (stored_len={len(s_content)}, "
                        f"incoming_len={len(i_content)})"
                    )
                    return False
            elif role == "tool":
                s_tcid = s_msg.get("tool_call_id") or ""
                i_tcid = i_msg.get("tool_call_id") or ""
                if (s_tcid or i_tcid) and s_tcid != i_tcid:
                    logger.debug(
                        f"[Match] FAIL at msg[{i}]: tool_call_id "
                        f"{s_tcid!r} != {i_tcid!r}"
                    )
                    return False

            # Codex round 13, finding 2: the historical first-<tool_call>
            # PREFIX SHORTCUT that lived here (initial commit — OpenCode
            # strips tool-call XML from content and moves it to the
            # structured tool_calls field, so stored
            # "text\n\n<tool_call>..." had to match incoming "text")
            # compared ONLY the text before the FIRST start marker and then
            # accepted the whole message on BOTH paths. Everything after
            # that marker was invisible: divergent content AFTER the call
            # blocks ("...ALPHA" vs "...BETA"), and — because a rehearsal
            # call inside a thinking segment also carries the marker (U12:
            # not extractable, so the structural comparison above sees
            # None on both sides) — the entire post-thinking content
            # channel, a wrong-HIT vector even in strict/anon mode. It is
            # REMOVED: its two legitimate shapes are covered end-to-end by
            # the modern machinery — (a) call lists compare STRUCTURALLY
            # above (F3/N2, with N3/round-4 rejecting any unvalidated
            # marker on a differing side), and (b) _normalize_for_match
            # strips the validated closed blocks, so the SURROUNDING
            # content channel is byte-compared below; both must agree. The
            # streaming-truncated mid-call stored turn is not a lost
            # leniency either: the N3 residual check above already
            # rejected that shape before the shortcut could run (honest
            # MISS), on the strict and the lenient path alike.

            # Normalize and compare (round 9, finding 2: the channel
            # reduction inside runs under the same family/contract as the
            # tool-call helpers above).
            s_norm = self._normalize_for_match(
                s_content, role, family, thinking_active,
            )
            i_norm = self._normalize_for_match(
                i_content, role, family, thinking_active,
            )
            if s_norm != i_norm:
                # Tool content compacted/cleared by client (either direction)
                # KV cache still valid — the tokens were already processed.
                # N1: EXPLICIT sessions only (last_assistant_wildcard=True).
                # On the STRICT anon path a "[cleared]"/"[compacted:…]"
                # placeholder is equivalent to EVERY stored tool result, so
                # with this leniency active two anonymous conversations
                # sharing a call chain (same tool_call_ids) could resolve
                # onto each other's session — anon resolution requires EXACT
                # tool-result content.
                if (
                    role == "tool"
                    and last_assistant_wildcard
                    and self._is_compacted_tool(s_content, i_content)
                ):
                    logger.debug(
                        f"[Match] msg[{i}] tool content compacted — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # U1/F2: last-stored-assistant tolerance is CONDITIONAL now.
                # - C1-committed interrupted turns (any position): the NARROW
                #   prefix equivalence (thinking-channel stripping /
                #   wire-truncation shapes) is reserved for EXPLICIT session
                #   ids (last_assistant_wildcard=True). The anon resolver
                #   (False) requires EXACT content-channel equality — prefix
                #   shapes are forgeable for session-less requests (an empty
                #   resend would match every interrupted turn).
                # - Otherwise, the historical last-stored-assistant wildcard
                #   applies ONLY when the caller allows it (explicit-session
                #   HIT keeps current behavior; the anon resolver disables it
                #   — the hijack vector).
                if role == "assistant":
                    if s_msg.get("interrupted"):
                        if self._interrupted_resend_equiv(
                            s_content, i_norm,
                            exact=not last_assistant_wildcard,
                            thinking_active=thinking_active,
                        ):
                            logger.debug(
                                f"[Match] msg[{i}] interrupted-turn "
                                f"{'narrow' if last_assistant_wildcard else 'exact'} "
                                f"equivalence — accepting (stored="
                                f"{len(s_content)}, incoming={len(i_content)})"
                            )
                            continue
                        # Narrow/exact rule failed: fall through to the
                        # generic leniencies below, then the FAIL log — an
                        # interrupted turn never gets the wildcard.
                    elif i == len(stored) - 1 and last_assistant_wildcard:
                        logger.debug(
                            f"[Match] msg[{i}] assistant content mismatch at last stored msg — "
                            f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                        )
                        continue

                # Client may reconstruct assistant content as "{thinking}\n\n{final}"
                # without <think> tags. Stored normalized is just "{final}"; incoming
                # normalized is "{thinking}\n\n{final}". KV cache reflects what the
                # model actually processed (stored), so if the final answer matches
                # as a suffix of the incoming, the cache is still valid.
                # F2: explicit sessions only — for the anon resolver this
                # generic suffix equivalence accepts non-prefix values and is
                # therefore forgeable; strict mode bypasses it.
                if role == "assistant" and last_assistant_wildcard and (
                    s_norm and len(s_norm) >= 8
                ) and (
                    i_norm.endswith(s_norm) or s_norm.endswith(i_norm)
                ):
                    logger.debug(
                        f"[Match] msg[{i}] assistant content suffix match — "
                        f"accepting (stored={len(s_content)}, incoming={len(i_content)})"
                    )
                    continue

                # Assistant tool_call turn: stored was (optional <think>...</think>
                # + <tool_call>...</tool_call>), which normalizes to empty because
                # both blocks are stripped. OpenAI-format clients move the tool
                # call to tool_calls[] and may reconstruct the thinking as plain
                # text in content (no <think> tags). KV cache is still valid —
                # it reflects the tokens the model actually emitted.
                # (Safe on BOTH paths since F3: when calls are extractable on
                # both sides the structured comparison above has already
                # verified they are the SAME calls — this leniency only
                # bridges the content-channel reconstruction.)
                if (
                    role == "assistant"
                    and not s_norm
                    and i_msg.get("tool_calls")
                    and ("<tool_call>" in s_content or s_msg.get("tool_calls"))
                ):
                    logger.debug(
                        f"[Match] msg[{i}] assistant tool_call turn — "
                        f"accepting reconstructed content "
                        f"(stored_len={len(s_content)}, incoming_len={len(i_content)})"
                    )
                    continue

                # Find exact divergence point
                diff_pos = next(
                    (j for j in range(min(len(s_norm), len(i_norm))) if s_norm[j] != i_norm[j]),
                    min(len(s_norm), len(i_norm)),
                )
                s_ctx = s_norm[max(0, diff_pos-30):diff_pos+70].replace('\n', '\\n')
                i_ctx = i_norm[max(0, diff_pos-30):diff_pos+70].replace('\n', '\\n')
                # DEBUG (not INFO): per-candidate diagnostic — see the role
                # FAIL above; aggregate summaries stay at INFO.
                logger.debug(
                    f"[Match] FAIL at msg[{i}] role={role}: "
                    f"stored_len={len(s_content)} vs incoming_len={len(i_content)} | "
                    f"diff at char {diff_pos} | "
                    f"stored=...{s_ctx!r}... | incoming=...{i_ctx!r}..."
                )
                return False
        return True

    def _base_caches_memory_gb(self) -> float:
        """Total resident bytes (GB) held by the base-cache pool (U2).

        Sizes are measured once at registration (entries are immutable
        snapshots), so this is a cheap sum, not a re-walk."""
        base_caches = getattr(self, "_base_caches", None)
        if not base_caches:
            return 0.0
        return sum(e.size_bytes for e in base_caches.values()) / 1e9

    def _evict_base_caches_lru(
        self, over_gb_fn, *, mru_allowance_gb: "float | None" = None,
    ) -> tuple[int, int]:
        """Evict base caches LRU-first while ``over_gb_fn()`` is True (U2).

        MRU protection is CONDITIONAL (round 2, codex F3): the single
        most-recently-used entry is kept while it can actually FIT the
        budget — base caches are auto re-registered with a FULL secondary
        system-prompt prefill on the next MISS, so evicting an entry the
        active workload is using would thrash (register -> evict ->
        re-prefill every turn). But an UNCONDITIONAL keep-MRU defeats the
        budget entirely: a single base entry larger than the budget left
        ZERO eviction candidates, and the next clone could OOM.
        ``mru_allowance_gb`` is the budget headroom left for the MRU entry
        after everything this sweep can never reclaim (the protected/MRU
        session + the prefix pool — see the caller); if we are STILL over
        budget after the LRU pass and the MRU entry exceeds that allowance,
        it is evicted too (loud WARNING). ``mru_allowance_gb=None`` keeps
        the historical unconditional protection (direct callers without
        budget context). Base caches are memory-only derived state (never
        persisted), so eviction is a plain drop.

        Caller MUST hold ``self._lock`` — the same lock that guards
        registration (_register_base_cache runs under _generate_locked /
        _mutate_locked), so eviction never races a half-built entry.

        Returns (evicted_count, freed_bytes)."""
        base_caches = getattr(self, "_base_caches", None)
        if not base_caches:
            return 0, 0
        # LRU first; the last (MRU) entry is protected during this pass.
        by_recency = sorted(base_caches.items(), key=lambda kv: kv[1].last_used)
        lru_hashes = [h for h, _ in by_recency[:-1]]
        mru_hash = by_recency[-1][0]
        evicted = 0
        freed_bytes = 0
        for h in lru_hashes:
            if not over_gb_fn():
                break
            entry = base_caches.pop(h, None)
            if entry is None:
                continue
            evicted += 1
            freed_bytes += entry.size_bytes
            logger.info(
                f"[Base Cache] EVICTED (LRU, over budget) | hash={h} | "
                f"{entry.token_count} tokens | {entry.size_bytes / 1e6:.1f}MB "
                f"| hits={entry.hit_count}"
            )
        # Round 2 (codex F3): waive the MRU anti-thrash protection when the
        # entry provably cannot fit — still over budget after shedding every
        # other entry, and the MRU alone exceeds the remaining allowance.
        # When it FITS, MRU stays protected (anti-thrash intent preserved).
        if (
            mru_allowance_gb is not None
            and mru_hash in base_caches
            and over_gb_fn()
        ):
            entry = base_caches[mru_hash]
            entry_gb = entry.size_bytes / 1e9
            if entry_gb > mru_allowance_gb:
                base_caches.pop(mru_hash, None)
                evicted += 1
                freed_bytes += entry.size_bytes
                logger.warning(
                    f"[Base Cache] EVICTED MRU entry (budget cannot fit it) | "
                    f"hash={mru_hash} | {entry.token_count} tokens | "
                    f"{entry.size_bytes / 1e6:.1f}MB "
                    f"({entry_gb:.2f} GB > allowance "
                    f"{max(mru_allowance_gb, 0.0):.2f} GB) | anti-thrash "
                    f"protection waived — it will re-prefill on the next MISS; "
                    f"raise memory_budget_gb to keep it resident"
                )
        return evicted, freed_bytes

    def _find_base_cache(self, messages: list[dict], tools: list | None = None) -> BaseCacheEntry | None:
        """Find a matching base cache for the given messages' system prompt."""
        h = self._system_hash(messages, tools=tools)
        if h and h in self._base_caches:
            return self._base_caches[h]
        return None

    def _register_base_cache(
        self, messages: list[dict], cache: list, system_tokens: list[int],
        tools: list | None = None, mtp_resume_hidden=None,
    ):
        """Register a base cache from the system prompt portion of a processed cache."""
        h = self._system_hash(messages, tools=tools)
        # Existing entries are kept (skip, never overwrite) — a stale plain
        # 40-entry base under an MTP server still seeds sessions correctly:
        # the MTP gate takes the plain-decode fallback instead of
        # cold-filling, so reuse is preserved either way.
        if not h or h in self._base_caches:
            return
        # Deep copy the cache at current state (after system prompt processing)
        import copy
        base_snapshot = copy.deepcopy(cache)
        self._eval_cache(base_snapshot)
        # U2: measure the entry once at registration — the same helper that
        # byte-counts session caches for the memory budget. The MTP boundary
        # hidden is tiny (1,1,H) but counted for honesty. getattr guard:
        # partially-constructed shell engines (unit tests via __new__) have no
        # cache_manager — size 0 there, never a registration failure.
        _cm = getattr(self, "cache_manager", None)
        size_bytes = _cm._estimate_cache_size(base_snapshot) if _cm else 0
        hidden_nbytes = getattr(mtp_resume_hidden, "nbytes", 0)
        if isinstance(hidden_nbytes, int):
            size_bytes += hidden_nbytes
        entry = BaseCacheEntry(
            system_hash=h,
            cache=base_snapshot,
            tokens=system_tokens,
            token_count=len(system_tokens),
            mtp_layout=mtp_resume_hidden is not None,
            mtp_resume_hidden=mtp_resume_hidden,
            size_bytes=size_bytes,
        )
        self._base_caches[h] = entry
        logger.debug(
            f"[Base Cache] REGISTERED | hash={h} | {len(system_tokens)} tokens | "
            f"{size_bytes / 1e6:.1f}MB | mtp={entry.mtp_layout} | "
            f"pool_size={len(self._base_caches)}"
        )

    def _clone_base_cache(self, base: BaseCacheEntry) -> list:
        """Clone a base cache for a new session."""
        import copy
        cloned = copy.deepcopy(base.cache)
        # Force evaluation of cloned arrays to avoid lazy-eval aliasing issues
        self._eval_cache(cloned)
        base.hit_count += 1
        # U2: LRU touch — every actual use (clone) marks recency for the
        # budget-driven base-cache eviction sweep.
        base.last_used = time.time()
        logger.debug(
            f"[Base Cache] CLONE | hash={base.system_hash} | "
            f"{base.token_count} tokens | hits={base.hit_count}"
        )
        return cloned

    def base_cache_stats(self) -> list[dict]:
        """Return stats for all base caches.

        U15: reads ``_base_caches`` under the engine lock (bounded — raises
        EngineBusyError while a generation holds it) so a concurrent
        registration/clear never surfaces a half-built entry."""
        with self._read_locked("base cache stats"):
            return [
                {
                    "system_hash": e.system_hash,
                    "token_count": e.token_count,
                    "hit_count": e.hit_count,
                    "created": e.created,
                    "mtp": e.mtp_layout,
                    # U2: byte accounting + LRU recency.
                    "size_mb": round(e.size_bytes / 1e6, 1),
                    "last_used": e.last_used,
                }
                for e in self._base_caches.values()
            ]

    def _resolve_anon_session_id_locked(
        self, messages: list[dict], prompt_fingerprint: str | None = None,
    ) -> str:
        """Resolve a session-less request onto a concrete per-conversation id.

        Session-less requests (no OpenAI ``user`` field — e.g. OpenCode and
        most agents) used to collapse onto the single shared ``"anon"`` key, so
        any two interleaved conversations (agent A vs agent B, or a client
        changing its system prompt mid-flow) thrashed the one slot: every
        request from the *other* conversation was a MISS → full cold-fill.

        Instead, scan the resident sessions for one whose stored
        ``session.messages`` is a (proper or exact) prefix of the incoming
        messages, reusing the same ``_messages_match`` logic the HIT path
        uses — but with ``last_assistant_wildcard=False`` (U1): the HIT
        path's last-stored-assistant content wildcard would let a DIFFERENT
        anonymous conversation, prefix-equal except its last assistant turn,
        hijack this conversation's session + cache (cross-conversation
        pollution). Candidate selection is therefore STRICT content
        matching; the only divergence tolerated is the narrow
        interrupted-turn equivalence (see _interrupted_resend_equiv), which
        applies identically on the HIT path, so selection and the subsequent
        cache decision still cannot disagree. Pick the LONGEST matched
        message prefix; tie-break by most-recently-used. An EXACT match
        (stored == incoming) is selected too — _generate_locked then takes
        its existing "retry" path (same messages re-sent → discard cache,
        re-process), unchanged.

        ``prompt_fingerprint`` (U21 interplay): when both sides carry a
        fingerprint, a mismatching candidate is skipped — two conversations
        with identical message prefixes but different tool schemas are
        DIFFERENT conversations and must not share a session slot. A
        ``None`` on either side (legacy session / direct caller) skips the
        filter; the HIT gate still applies the contract check downstream.

        If nothing matches, mint a NEW unique ``anon-<8 hex>`` id so each
        session-less conversation owns its own cache entry. These sessions
        live in ``self._sessions`` like any other, so the active-session LRU
        eviction (memory_budget_gb) bounds them; busy-protection applies
        because the caller busy-marks the id returned HERE.

        Scope notes:
        - Linear scan is fine: the session count is small and LRU-bounded by
          _evict_active_sessions_if_needed; _messages_match early-outs on
          length/role before any content comparison.
        - In-memory candidates only. Enumerating disk-cached sessions would
          need a safetensors metadata read PER candidate PER request just to
          compare messages — an evicted anon conversation simply degrades to
          one cold-fill (exactly today's behavior) and is resident again
          afterwards.
        - The legacy ``"anon"`` slot participates only when the ENGINE's own
          fallback minted it (see _generate_locked): provenance, not name. A
          pre-existing disk "anon" cache from before this feature simply stops
          being reused by anonymous requests — one cold-fill, no migration.

        Caller MUST hold ``self._lock`` (``self._sessions`` and
        ``self._anon_minted_ids`` are mutated under it) and must busy-mark the
        RETURNED id, not "anon".
        """
        # Defensive: partially-constructed shell engines (unit tests build via
        # __new__) may lack _sessions / _anon_minted_ids — same pattern as the
        # eviction sweep.
        sessions = getattr(self, "_sessions", None) or {}
        minted = getattr(self, "_anon_minted_ids", None)
        if minted is None:
            minted = set()
        best_sid: str | None = None
        best_len = -1
        best_used = -1.0
        for sid, session in sessions.items():
            # PROVENANCE-based isolation: only sessions the engine itself
            # minted for session-less requests are candidates. A name check
            # (sid.startswith("anon-")) is NOT enough — ChatCompletionRequest
            # .user is unrestricted and passed verbatim as session_id, so a
            # client could create an EXPLICIT session keyed "anon-aaaaaaaa";
            # it must never be selected — and then mutated — by an anonymous
            # request, even on a perfect message-prefix match. Stale ids in
            # the set (session since evicted/deleted) are harmless: the scan
            # iterates self._sessions and merely checks membership; removal
            # points additionally discard ids best-effort (see
            # _evict_active_sessions_if_needed / delete_session /
            # clear_caches).
            if sid not in minted:
                continue
            stored = session.messages
            # Empty stored messages would prefix-match ANYTHING — skip.
            if not stored or len(stored) > len(messages):
                continue
            # U21: different tool contract → different conversation, even on
            # a perfect message-prefix match (None on either side = legacy /
            # unknown → filter skipped, HIT gate decides downstream).
            _sess_fp = getattr(session, "prompt_fingerprint", None)
            if (
                prompt_fingerprint is not None
                and _sess_fp is not None
                and _sess_fp != prompt_fingerprint
            ):
                continue
            if not self._messages_match(
                stored, messages, last_assistant_wildcard=False,
                # Round 3, finding 4: the candidate's own stored thinking
                # contract drives its content-channel reduction.
                thinking_active=getattr(session, "thinking", True),
            ):
                continue
            n = len(stored)
            if n > best_len or (n == best_len and session.last_used > best_used):
                best_sid = sid
                best_len = n
                best_used = session.last_used
        if best_sid is not None:
            logger.info(
                f"[KV Cache] anon request matched session={best_sid} "
                f"(prefix {best_len}/{len(messages)} msgs)"
            )
            return best_sid
        new_sid = f"anon-{uuid.uuid4().hex[:8]}"
        # Record provenance AT MINT TIME so the next anonymous request can
        # find this session (shell engines without the set just lose tracking,
        # which only means no reuse — never cross-session mutation).
        if getattr(self, "_anon_minted_ids", None) is not None:
            self._anon_minted_ids.add(new_sid)
        logger.info(
            f"[KV Cache] anon request: no prefix match among "
            f"{len(sessions)} sessions — new session={new_sid}"
        )
        return new_sid

    def _mtp_base_caches_active(self) -> bool:
        """True iff base caches must be built in the MTP-finalized layout —
        the qwen_mtp drafter actually runs on this server's decode path
        (mlx-lm backend, no --kv-bits, which disables MTP globally). All
        other configurations (plain, gemma4/mlx-vlm, PLD) keep the
        historical target-only layout untouched."""
        return (
            not self._use_vlm
            and getattr(self, "_drafter", None) is not None
            and getattr(self, "_draft_kind", None) == "qwen_mtp"
            and not self.cfg.kv_bits
        )

    # --- Disk cache persistence ---

    def _session_cache_path(self, session_id: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"session_{session_id}.safetensors")

    def _save_session_to_disk(self, session_id: str, session: SessionState):
        """Save session's KV cache + token history to disk. Caller MUST hold _lock.

        Returns True on success, False if save is not possible (e.g. empty arrays).
        """
        if session.cache_state is None or session.cache_state.cache is None:
            return True
        t0 = time.perf_counter()
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        path = self._session_cache_path(session_id)
        _sess_tools = getattr(session, "tools", None)
        metadata = {
            "session_id": session_id,
            "messages": json.dumps(session.messages, ensure_ascii=False),
            "total_cache_tokens": str(session.total_cache_tokens),
            "last_used": str(session.last_used),
            "token_ids": json.dumps(session.cache_state.token_ids or []),
            # Prompt contract (U3/U21) — must survive a restart so rebuilds
            # keep the tool schema and the HIT gate can verify it.
            "tools": json.dumps(_sess_tools, ensure_ascii=False) if _sess_tools else "",
            "thinking": "1" if getattr(session, "thinking", True) else "0",
            "prompt_fingerprint": getattr(session, "prompt_fingerprint", None) or "",
        }
        def _do_save():
            # WHY: VLM KV-cache tensors are lazy and bound to the
            # _vlm_executor worker thread's generation_stream (post-gen
            # _eval_cache is skipped on the VLM path for perf, see F3 below).
            # Materializing AND serializing must therefore happen on that
            # same worker thread, or mx.eval raises "There is no Stream(gpu, 1)
            # in current thread." _eval_cache covers keys/values AND ArraysCache
            # .state arrays (DeltaNet recurrent state). On the legacy mlx-lm
            # path this runs inline on the request/flush thread (default stream).
            self._eval_cache(session.cache_state.cache)
            save_prompt_cache(path, session.cache_state.cache, metadata=metadata)

        try:
            if getattr(self, "execution_mode", "worker") == "main_thread":
                # main_thread mode: model + cache tensors are bound to THIS
                # thread's stream slot. Materialize + serialize inline.
                _do_save()
            elif getattr(self, "_use_vlm", False) and getattr(self, "_vlm_executor", None) is not None:
                if threading.get_ident() == getattr(self, "_vlm_worker_ident", None):
                    # RE-ENTRANCY: we are ALREADY running on the single
                    # _vlm_executor worker thread (e.g. post-generation eviction
                    # driven inside generate_stream's finally, which is itself
                    # invoked on the worker). Submitting to the same one-worker
                    # pool and blocking on fut.result() would deadlock against
                    # ourselves for the full 60s timeout. The cache tensors are
                    # already bound to THIS thread's generation_stream, so just
                    # materialize + serialize inline.
                    _do_save()
                else:
                    try:
                        fut = self._vlm_executor.submit(_do_save)
                    except RuntimeError:
                        # Executor already shut down (can happen during
                        # _flush_all_on_shutdown). Best-effort inline save.
                        _do_save()
                    else:
                        # Exceptions raised inside _do_save propagate out of
                        # .result(), so the surrounding except still classifies
                        # them (empty-array permanent skip vs unexpected re-raise).
                        fut.result(timeout=60)
            else:
                _do_save()
        except Exception as e:
            if "empty array" in str(e).lower() or "cannot serialize" in str(e).lower():
                # Some models (GLM MoE) have empty arrays that safetensors can't handle
                logger.info(
                    f"[KV Cache] session={session_id} | DISK SAVE SKIP | "
                    f"cache not serializable: {e}"
                )
                return False  # permanent failure, don't retry
            raise  # re-raise unexpected errors

        elapsed = time.perf_counter() - t0
        if hasattr(self, "_disk_session_ids"):
            self._disk_session_ids.add(session_id)
        fsize = os.path.getsize(path) / 1e6
        logger.info(
            f"[KV Cache] session={session_id} | DISK SAVE | "
            f"{session.total_cache_tokens} tokens, {len(session.messages)} msgs, "
            f"{fsize:.1f}MB, {elapsed:.2f}s"
        )
        # LRU eviction: keep total disk usage under budget
        self._evict_disk_sessions_if_needed(protect_session_id=session_id)
        return True

    def _evict_disk_sessions_if_needed(self, protect_session_id: str | None = None):
        """Scan cache_dir and delete oldest session files if disk usage exceeds budget.

        Protects:
        - The session we just saved (protect_session_id)
        - Any session currently in self._sessions (in-memory, active)
        """
        budget_bytes = int(self.cfg.disk_budget_gb * 1e9)
        cache_dir = self.cfg.cache_dir
        if not os.path.isdir(cache_dir):
            return

        # Gather session file info: path, size, mtime, session_id
        entries = []
        total_size = 0
        for fname in os.listdir(cache_dir):
            if not fname.startswith("session_") or not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(cache_dir, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            total_size += st.st_size
            # Extract session_id: session_<id>.safetensors or session_<id>_ckpt.safetensors
            sid_part = fname[len("session_"):-len(".safetensors")]
            sid = sid_part[:-len("_ckpt")] if sid_part.endswith("_ckpt") else sid_part
            entries.append((st.st_mtime, st.st_size, fpath, sid))

        if total_size <= budget_bytes:
            return

        # Sort oldest first
        entries.sort(key=lambda e: e[0])
        protected = set(self._sessions.keys())
        if protect_session_id:
            protected.add(protect_session_id)

        deleted = 0
        freed = 0
        for mtime, size, fpath, sid in entries:
            if total_size <= budget_bytes:
                break
            if sid in protected:
                continue
            try:
                os.remove(fpath)
                total_size -= size
                freed += size
                deleted += 1
                if hasattr(self, "_disk_session_ids"):
                    self._disk_session_ids.discard(sid)
            except OSError as e:
                logger.debug(f"[Disk LRU] failed to delete {fpath}: {e}")

        if deleted:
            logger.info(
                f"[Disk LRU] evicted {deleted} files, freed {freed/1e9:.2f} GB "
                f"(total now {total_size/1e9:.2f}/{budget_bytes/1e9:.2f} GB)"
            )

    def _session_cache_bytes(self, session: "SessionState") -> int:
        """Estimate the resident KV bytes held by a single session."""
        cache = session.cache_state.cache if session.cache_state else None
        if cache is None:
            return 0
        return self.cache_manager._estimate_cache_size(cache)

    def _mark_dirty(self, session_id: str):
        """Mark a session for disk save on next idle cycle."""
        with self._dirty_lock:
            self._dirty_sessions.add(session_id)

    def _flush_dirty_sessions(self):
        """Flush all dirty sessions to disk. Caller MUST hold _lock.

        U16 — POP-ONE-SAVE-ONE (chosen over drain-then-iterate): the process
        worker's SIGTERM handler raises a ``BaseException`` sentinel
        (``_GracefulShutdown``) that can land while this runs from the idle
        flush. The old shape drained the WHOLE dirty set up front and only
        ``except Exception`` re-marked — a BaseException unwinding mid-loop
        lost every drained-but-unsaved id, so the shutdown flush saw an empty
        set and the sessions were silently dropped. Now each id is popped
        individually right before its save, and a BaseException re-marks the
        single in-flight id before re-raising — an interrupt therefore loses
        NOTHING (already-saved ids need no retry; unprocessed ids were never
        popped; the in-flight one is re-marked and the shutdown flush's own
        pass saves it).

        ``processed`` guards against re-popping an id this call already
        handled: a failing save re-marks its id for the NEXT flush cycle,
        which must not turn into an infinite retry loop here.

        F1 (codex batch-3 review, round 2): the ENTIRE pop→save→handle-error
        body sits inside ONE ``try``/``finally``. ``settled`` flips to True
        only once the in-flight id needs NO restore (saved, permanently
        unsaveable, evicted, or the ordinary-failure handler already made
        its own re-mark decision); the ``finally`` re-marks otherwise. The
        earlier shape used a sibling ``except BaseException`` re-mark — but
        a BaseException (SIGTERM sentinel) raised INSIDE the
        exception-handling path itself (during logging / just before the
        re-mark) escapes sibling excepts, so the popped id was lost. The
        ``finally`` re-mark covers every unwind path, and it is a bare
        ``set.add`` — cheap, idempotent, exception-safe. sid selection stays
        OUTSIDE the guard: an interrupt there finds the id still un-popped
        (nothing to restore). BaseException still propagates to abort the
        loop (never swallow the sentinel).
        """
        processed: set[str] = set()
        attempted = 0
        saved = 0
        while True:
            with self._dirty_lock:
                remaining = self._dirty_sessions - processed
                if not remaining:
                    break
                sid = next(iter(remaining))
            # True once this id needs NO finally re-mark (see F1 above).
            settled = False
            try:
                with self._dirty_lock:
                    self._dirty_sessions.discard(sid)
                processed.add(sid)
                session = self._sessions.get(sid)
                if session is None:
                    # Evicted since it was marked: drop, never re-mark.
                    settled = True
                    continue
                attempted += 1
                success = self._save_session_to_disk(sid, session)
                if success:
                    saved += 1
                # If success=False (permanent failure like empty arrays),
                # don't retry — either way the id is settled.
                settled = True
            except Exception as e:
                logger.error(f"[KV Cache] session={sid} | FLUSH SAVE FAILED | {e}")
                # Terminate this call's retry loop even if the failure landed
                # before processed.add ran (idempotent when it already did).
                processed.add(sid)
                with self._dirty_lock:
                    if sid in self._sessions:
                        self._dirty_sessions.add(sid)
                settled = True
            finally:
                # U16/F1: anything still un-settled here is a BaseException
                # unwind (shutdown sentinel / KeyboardInterrupt) — from the
                # pop→save gap, the save itself, or the except-block above.
                # Re-mark the in-flight id so the shutdown flush retries it;
                # the unwind then proceeds untouched. Re-adding an id the
                # interrupt caught BEFORE the pop is a harmless no-op.
                if not settled:
                    with self._dirty_lock:
                        self._dirty_sessions.add(sid)

        if saved:
            logger.info(f"[Idle Flush] saved {saved}/{attempted} dirty sessions")

    @classmethod
    def _flush_all_on_shutdown(cls, lock_timeout: float = 10.0):
        """Save all dirty sessions across all engines on shutdown.

        IDEMPOTENT: an engine whose dirty set is empty is skipped with a
        cheap PEEK (no lock acquire), so a second call (worker-loop flush
        followed by the atexit backstop, or atexit after a signal handler)
        is a cheap no-op.

        LOCK-FIRST ORDER (codex round 7, finding 2b): the engine lock is
        acquired BEFORE anything is drained from the dirty set. The old
        shape drained a snapshot first and RE-MARKED it when the bounded
        lock acquire timed out — that re-mark could land AFTER a concurrent
        straggler's self-flush (which holds the engine lock) had already
        drained the set, so the re-marked ids were stranded dirty forever
        (the self-flush was the last flush that could ever run). With the
        lock acquired first, a timeout drained NOTHING: the ids simply stay
        marked, and the lock-holding straggler's self-flush (which rescans
        — finding 2c) saves them before releasing the lock.

        FAIL-CLOSED: the lock acquire stays BOUNDED. In main_thread mode a
        signal handler runs on the same thread as an in-flight generation
        that holds the non-reentrant lock — a blocking acquire there would
        self-deadlock shutdown forever. On timeout the engine is skipped
        with a warning (nothing drained, nothing lost). Per-session and
        per-engine save failures are logged and never propagate.

        Draining goes through ``_flush_dirty_sessions``, which keeps the
        U16/R2-F1 pop-one-save-one + settled-flag semantics intact: each id
        is popped individually right before its save, a BaseException
        unwind re-marks exactly the in-flight id via the ``finally``
        re-mark, and the loop keeps draining ids marked while it runs
        (bounded by ``processed`` — each id at most once per call).
        """
        for engine in cls._all_engines:
            try:
                # Cheap idempotency peek — never touches the engine lock.
                with engine._dirty_lock:
                    n_dirty = len(engine._dirty_sessions)
                if not n_dirty:
                    continue
                logger.info(
                    f"[Shutdown] Flushing {n_dirty} dirty sessions for "
                    f"{engine.model_id}"
                )
                # Finding 2b: LOCK FIRST — only a holder of the engine lock
                # may drain the dirty set during shutdown.
                if not engine._lock.acquire(timeout=lock_timeout):
                    logger.warning(
                        f"[Shutdown] engine lock not acquired within "
                        f"{lock_timeout}s (in-flight generation?) — skipping "
                        f"flush for {engine.model_id} (nothing drained; the "
                        f"ids stay marked for the lock holder's self-flush)"
                    )
                    continue
                try:
                    engine._flush_dirty_sessions()
                finally:
                    engine._lock.release()
            except Exception as e:  # noqa: BLE001 — never block shutdown
                logger.error(
                    f"[Shutdown] flush failed for engine "
                    f"{getattr(engine, 'model_id', '?')}: {e}"
                )

    def _load_session_from_disk(self, session_id: str) -> Optional[SessionState]:
        """Load session's KV cache + token history from disk."""
        path = self._session_cache_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            t0 = time.perf_counter()
            cache, metadata = load_prompt_cache(path, return_metadata=True)
            messages = json.loads(metadata.get("messages", "[]"))
            total_tokens = int(metadata.get("total_cache_tokens", "0"))
            last_used = float(metadata.get("last_used", "0"))
            token_ids = json.loads(metadata.get("token_ids", "[]"))
            # Prompt contract (U3/U21). Legacy files (pre-fingerprint) load
            # as tools=None / fingerprint=None — the HIT gate then takes ONE
            # unconditional cold rebuild that stamps the fingerprint (F5:
            # never a lenient HIT; the legacy contract is unknowable).
            _tools_meta = metadata.get("tools", "")
            sess_tools = json.loads(_tools_meta) if _tools_meta else None
            sess_thinking = metadata.get("thinking", "1") != "0"
            sess_fp = metadata.get("prompt_fingerprint") or None

            # Verify loaded cache matches model structure (leading slice).
            # MTP-finalized sessions (qwen_mtp) persist n_target + n_head
            # entries, but the finalize hidden the MTP gate would need is an
            # in-memory stash that is never written to disk — MTP reuse is
            # impossible after a restart regardless. Plain reuse of the full
            # token history is what matters: accept the load and STRIP the
            # extra trailing (head) entries, so the next turn plans
            # REUSE_FALLBACK_PLAIN over the whole history instead of
            # cold-filling. Stripping is gated to EXACTLY the qwen_mtp
            # finalized layout (engine mtp-capable + n_extra == head layer
            # count + trailing entries are KVCache heads at
            # len(token_ids) - 1); any other oversized layout is rejected.
            model_cache = make_prompt_cache(self._language_model)
            n_model = len(model_cache)
            if len(cache) < n_model:
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                    f"layer count mismatch: {len(cache)} vs {n_model}"
                )
                return None

            type_ok = all(
                type(c).__name__ == type(m).__name__
                for c, m in zip(cache[:n_model], model_cache)
            )
            if not type_ok:
                logger.error(
                    f"[KV Cache] session={session_id} | DISK LOAD FAILED | cache type mismatch"
                )
                return None

            if len(cache) > n_model:
                # Fail-closed strip — tightened contract: extra trailing
                # entries are ONLY strippable when they are exactly THIS
                # server's qwen_mtp finalized layout, i.e. ALL of:
                #   (a) the engine is qwen-mtp-capable right now
                #       (_mtp_base_caches_active: mlx-lm backend, qwen_mtp
                #       drafter loaded, no --kv-bits),
                #   (b) n_extra == the drafter's head layer count (the same
                #       count make_head_cache was sized with at save time),
                #   (c) every trailing extra is a head entry of the exact
                #       type make_head_cache produces (KVCache), sitting at
                #       the finalized lazy-last-slot offset
                #       len(token_ids) - 1 (head trails the target by one),
                #   (d) after dropping them, EVERY offset-bearing target
                #       layer sits exactly at len(token_ids).
                # Anything else — a foreign cache with extra target layers,
                # a same-type larger layout, a plain/non-MTP server reading
                # a head-carrying file — is a layout we don't understand:
                # reject the whole load (as before the strip feature).
                from mlx_soloheaven.engine.pld import _layer_offsets
                from mlx_soloheaven.engine import qwen_mtp as qwen_mtp_mod
                n_extra = len(cache) - n_model
                if not self._mtp_base_caches_active():
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"{n_extra} extra trailing entries but this engine is "
                        f"not qwen-mtp-capable — refusing to strip an unknown "
                        f"layout"
                    )
                    return None
                _n_head = max(1, len(getattr(self._drafter, "layers", [])) or 1)
                if n_extra != _n_head:
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"{n_extra} extra trailing entries != drafter head "
                        f"layer count {_n_head} — not the MTP-finalized layout"
                    )
                    return None
                _head_type = type(qwen_mtp_mod.make_head_cache(1)[0])
                _head_off = len(token_ids) - 1
                _bad_head = [
                    (n_model + i, type(c).__name__, getattr(c, "offset", None))
                    for i, c in enumerate(cache[n_model:])
                    if type(c) is not _head_type
                    or getattr(c, "offset", None) != _head_off
                ]
                if _bad_head:
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"trailing entries are not finalized MTP head entries "
                        f"(need {_head_type.__name__} at offset {_head_off}): "
                        f"{_bad_head[:8]}"
                    )
                    return None
                cache = cache[:n_model]
                _bad = [
                    (i, off)
                    for i, off in enumerate(_layer_offsets(cache))
                    if off is not None and off != len(token_ids)
                ]
                if _bad:
                    logger.error(
                        f"[KV Cache] session={session_id} | DISK LOAD FAILED | "
                        f"stripped {n_extra} trailing entries but target "
                        f"offsets != {len(token_ids)} stored ids: {_bad[:8]}"
                    )
                    return None
                logger.info(
                    f"[KV Cache] session={session_id} | DISK LOAD | stripped "
                    f"{n_extra} trailing MTP head entries ({n_model}-layer "
                    f"target) — session continues via plain-fallback reuse"
                )

            loaded_offset = self._get_cache_offset(cache)
            elapsed = time.perf_counter() - t0

            # Reconstruct PromptCacheState
            cache_state = PromptCacheState()
            cache_state.cache = cache
            cache_state.token_ids = token_ids if token_ids else None

            session = SessionState(
                cache_state=cache_state,
                messages=messages,
                total_cache_tokens=loaded_offset,
                last_used=last_used,
                tools=sess_tools,
                thinking=sess_thinking,
                prompt_fingerprint=sess_fp,
                # U26: a disk-reloaded session reclaims its in-memory drafter
                # stats (the registry outlives active-LRU eviction).
                drafter_stats=self._drafter_stats_for(session_id),
            )

            fsize = os.path.getsize(path) / 1e6
            logger.info(
                f"[KV Cache] session={session_id} | DISK LOAD | "
                f"{loaded_offset} tokens, {len(messages)} msgs, "
                f"{fsize:.1f}MB, {elapsed:.2f}s"
            )
            return session
        except Exception as e:
            logger.error(f"[KV Cache] session={session_id} | DISK LOAD FAILED | {e}")
            return None

    def _build_disk_index(self):
        """Scan cache_dir for saved session caches."""
        cache_dir = self.cfg.cache_dir
        if not os.path.isdir(cache_dir):
            return

        self._disk_session_ids: set[str] = set()
        count = 0
        for fname in os.listdir(cache_dir):
            if fname.startswith("session_") and fname.endswith(".safetensors") and "_ckpt" not in fname:
                sid = fname[len("session_"):-len(".safetensors")]
                self._disk_session_ids.add(sid)
                count += 1

        if count:
            logger.debug(f"[KV Cache] Disk index: {count} saved session caches")
        else:
            logger.debug("[KV Cache] Disk index: no saved session caches")

    def _has_disk_cache(self, session_id: str) -> bool:
        return hasattr(self, "_disk_session_ids") and session_id in self._disk_session_ids

    # Defensive bound on the safetensors JSON header read by the preflight's
    # metadata parse — a corrupt length prefix must never trigger a giant
    # allocation (real session-cache headers are tens of KB).
    _SAFETENSORS_HEADER_MAX_BYTES = 100 * 1024 * 1024

    @staticmethod
    def _read_safetensors_metadata(path: str) -> dict | None:
        """Read ONLY the ``__metadata__`` block of a safetensors file with
        plain Python file IO — the format is a little-endian 8-byte header
        length followed by a JSON header; tensor data is never touched and
        NO MLX call is involved (F5 round 2: the preflight must not create
        arrays on a non-owning thread). Returns None on any read/parse
        failure (fail-closed advisory)."""
        try:
            with open(path, "rb") as f:
                raw_len = f.read(8)
                if len(raw_len) != 8:
                    return None
                header_len = int.from_bytes(raw_len, "little")
                if not 0 < header_len <= SessionCacheMixin._SAFETENSORS_HEADER_MAX_BYTES:
                    return None
                raw = f.read(header_len)
                if len(raw) != header_len:
                    # Short read == truncated file (codex round 3, finding
                    # 6): the available prefix may still parse as JSON, but
                    # the file is provably damaged — fail closed rather than
                    # report a disk hit from a header we only partially saw.
                    return None
                header = json.loads(raw)
        except (OSError, ValueError):
            return None
        if not isinstance(header, dict):
            return None
        meta = header.get("__metadata__")
        return meta if isinstance(meta, dict) else None

    # --- Active-session memory bounding (LRU eviction) ----------------------

    def _active_sessions_memory_gb(self) -> float:
        """Total resident KV memory (GB) across all in-memory sessions.

        This is the number that memory_budget_gb is supposed to bound but did
        NOT before: idle-flush saves dirty sessions to disk yet leaves their KV
        caches resident in self._sessions, so without active eviction they
        accumulate until the Mac OOMs.
        """
        return sum(
            self._session_cache_bytes(s) for s in self._sessions.values()
        ) / 1e9

    def _evict_active_sessions_if_needed(self, protect_session_id: str | None = None):
        """Best-effort bound on total resident KV memory toward memory_budget_gb.

        Active per-session KV caches (self._sessions) are the dominant memory
        consumer but were previously unbounded — only delete_session removed
        them. This evicts the LEAST-RECENTLY-USED idle session when the active
        KV total PLUS the separate LRU prefix-reuse pool PLUS the base-cache
        pool (U2) exceeds the budget: the session is persisted to disk (so its
        next request transparently reloads it) and then dropped from
        self._sessions so MLX frees the buffers. Base caches are shed FIRST
        (LRU, CONDITIONAL keep-MRU — see _evict_base_caches_lru; round 2,
        codex F3: the MRU base entry is evicted too when it cannot fit the
        budget allowance left after the unevictable sessions): they are
        memory-only derived state, auto re-registered on the next MISS.

        Never evicts:
          - a session with an in-flight generation (_busy_sessions),
          - the just-used session (protect_session_id),
          - the single most-recently-used session (always keep one resident).

        The bound is BEST-EFFORT, not hard: because the protected/MRU/last
        session is always preserved, the sweep can legitimately exit still over
        budget when that one un-evictable session alone exceeds the budget. In
        that case it logs a WARNING and admin status reports
        ``budget_unmet=True`` (see status_dict's ``memory`` block) — the caller
        should raise memory_budget_gb or use fewer concurrent long sessions.

        Caller MUST hold ``self._lock`` (it persists caches + mutates
        self._sessions, exactly like _flush_dirty_sessions).
        """
        # Defensive: a partially-constructed shell engine (e.g. unit tests that
        # build via __new__ and only set cfg + _lock) has no session/cache
        # machinery — there is nothing to evict.
        if getattr(self, "_sessions", None) is None or getattr(self, "cache_manager", None) is None:
            return

        budget_gb = float(getattr(self.cfg, "memory_budget_gb", 0) or 0)
        if budget_gb <= 0:
            return

        pool_gb = self.cache_manager._memory_usage_gb()

        # U2: base caches are charged against the SAME memory_budget_gb as
        # the active sessions + the prefix-reuse pool — they are the third
        # (previously unaccounted, unbounded) resident-KV consumer.
        def _total_gb() -> float:
            return (
                self._active_sessions_memory_gb()
                + pool_gb
                + self._base_caches_memory_gb()
            )

        if _total_gb() <= budget_gb:
            return

        # U2: shed base caches FIRST (LRU, conditional keep-MRU) — they are
        # memory-only derived state, auto re-registered from the next MISS's
        # prefill, while evicting a session costs a disk save + reload. Only
        # if the total is still over budget does the session sweep below run.
        #
        # Round 2 (codex F3): the MRU base entry's protection is conditional
        # on it FITTING. Its allowance = budget minus everything this sweep
        # can never reclaim: the prefix pool plus the session(s) the session
        # sweep below always keeps resident (the protected just-used session
        # and the MRU session — often the same one). A base entry larger
        # than that allowance would leave the sweep permanently over budget
        # with zero candidates (the original U2 hole: an 80GB lone base
        # cache under a 64GB budget was never evictable), so it is evicted
        # too — see _evict_base_caches_lru.
        unevictable_gb = pool_gb
        if self._sessions:
            protected_sids = {
                max(
                    self._sessions.items(), key=lambda kv: kv[1].last_used
                )[0]
            }
            if protect_session_id and protect_session_id in self._sessions:
                protected_sids.add(protect_session_id)
            unevictable_gb += sum(
                self._session_cache_bytes(self._sessions[sid])
                for sid in protected_sids
            ) / 1e9
        base_evicted, base_freed = self._evict_base_caches_lru(
            lambda: _total_gb() > budget_gb,
            mru_allowance_gb=budget_gb - unevictable_gb,
        )
        if base_evicted:
            logger.info(
                f"[Base Cache] evicted {base_evicted} entr"
                f"{'y' if base_evicted == 1 else 'ies'}, freed "
                f"~{base_freed / 1e9:.2f} GB (over memory budget)"
            )
        if _total_gb() <= budget_gb:
            if base_evicted:
                # Base-cache shedding alone brought us under budget — release
                # the now-unreferenced Metal buffers before returning.
                try:
                    if hasattr(mx, "clear_cache"):
                        mx.clear_cache()
                except Exception:  # noqa: BLE001
                    pass
            return

        with self._busy_lock:
            busy = set(self._busy_sessions)

        # Candidates = evictable session IDs, LRU first (oldest last_used
        # first). Hold IDs only (not the SessionState objects) so that once a
        # session is popped from _sessions nothing in this sweep keeps a strong
        # reference to its KV cache — otherwise mx.clear_cache() below could not
        # free the buffers while this list is still alive.
        candidate_ids = [
            sid
            for sid, _ in sorted(
                self._sessions.items(), key=lambda kv: kv[1].last_used
            )
        ]
        # The single most-recently-used session is always kept resident.
        keep_mru = candidate_ids[-1] if candidate_ids else None

        evicted = 0
        freed_bytes = 0
        for sid in candidate_ids:
            if _total_gb() <= budget_gb:
                break
            # len() check after each eviction so we never drop the last one.
            if len(self._sessions) <= 1:
                break
            if sid in busy:
                continue
            if protect_session_id and sid == protect_session_id:
                continue
            if sid == keep_mru:
                continue

            session = self._sessions.get(sid)
            if session is None:
                continue
            sess_bytes = self._session_cache_bytes(session)
            try:
                # Persist so the next request for this session reloads it from
                # disk (see _generate_locked / truncate / branch disk-cache
                # consult paths). save returning False = permanently unsaveable
                # cache (e.g. GLM empty arrays) — it would never round-trip, so
                # there is nothing on disk to reload either way.
                saved = self._save_session_to_disk(sid, session)
                save_error: Exception | None = None
            except Exception as e:  # noqa: BLE001
                # TRANSIENT failure (timeout, unexpected error). Do NOT drop the
                # session: dropping it here would lose the KV cache with no disk
                # copy to reload from, forcing a full from-scratch rebuild on the
                # client's next request. Keep it resident and skip this sweep.
                saved = False
                save_error = e

            if save_error is not None:
                logger.warning(
                    f"[Active LRU] session={sid} | SAVE FAILED, keeping resident "
                    f"(no durable copy — would be a lossy rebuild) | {save_error}"
                )
                continue

            # Persist-then-reload is only durable when the save actually
            # succeeded. A returning-False save is a PERMANENT unsaveable cache:
            # there is no disk copy to reload, but the session can rebuild from
            # its `messages`, so evicting it to reclaim RAM is acceptable. Any
            # other failure was already handled (continue) above.
            self._sessions.pop(sid, None)
            with self._dirty_lock:
                self._dirty_sessions.discard(sid)
            # Anon-provenance hygiene: best-effort discard so the set tracks
            # live sessions (a stale id would be harmless — the prefix scan
            # iterates self._sessions — this just bounds growth).
            anon_ids = getattr(self, "_anon_minted_ids", None)
            if anon_ids is not None:
                anon_ids.discard(sid)
            # Same hygiene for the gemma4 nothink log-once set (U18
            # follow-up round 2): it keys on session ids and is pure
            # observability — without this discard, sessionless gemma
            # traffic grows it without bound. Worst case the demotion logs
            # once more if this session is ever reloaded.
            nothink_logged = getattr(self, "_gemma4_nothink_miss_logged", None)
            if nothink_logged is not None:
                nothink_logged.discard(sid)
            evicted += 1
            freed_bytes += sess_bytes
            if not saved:
                logger.warning(
                    f"[Active LRU] session={sid} | evicted WITHOUT disk save "
                    f"(cache not serializable) — will rebuild from messages"
                )
            # Drop our only remaining strong ref to the evicted session (and its
            # KV cache) so the buffers are unreferenced before clear_cache().
            session = None

        # The loop variable can still pin the last-inspected (possibly evicted)
        # session — drop it before asking MLX to release buffers.
        session = None

        if evicted:
            # Encourage MLX to release the now-unreferenced Metal buffers.
            try:
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass
            logger.info(
                f"[Active LRU] evicted {evicted} idle session(s), "
                f"freed ~{freed_bytes / 1e9:.2f} GB | "
                f"active KV now {self._active_sessions_memory_gb():.2f} GB "
                f"+ pool {pool_gb:.2f} GB "
                f"+ base {self._base_caches_memory_gb():.2f} GB "
                f"/ budget {budget_gb:.1f} GB"
            )

        # Best-effort, not a hard cap: the protected / MRU / last-remaining
        # session (and the MRU base cache) is never evicted. If that
        # un-evictable residue alone still exceeds the budget, surface it
        # loudly rather than silently lying.
        if _total_gb() > budget_gb:
            logger.warning(
                f"[Active LRU] still OVER budget after sweep: resident "
                f"{_total_gb():.2f} GB > budget {budget_gb:.1f} GB "
                f"(protected/MRU/last session + MRU base cache are "
                f"un-evictable) — raise memory_budget_gb or reduce "
                f"concurrent long sessions"
            )

    # --- Web-chat cache preflight (advisory, read-only) --------------------

    def session_cache_preflight(self, session_id: str, messages: list[dict]) -> dict:
        """Web-chat cache preflight: ADVISORY report of whether this
        session's next turn will reuse its KV cache (and how). Returns
        ``{"cache_hit": bool, "cache_info": dict}`` — informational only
        (the generation itself re-resolves — and actually loads — the cache
        under its own lock, on its own thread).

        F5 (codex batch-3 review, round 2): METADATA-ONLY. The round-1
        shape ran ``_load_session_from_disk`` (mx.load + KV-cache object
        construction) on the ``engine-read`` executor thread and published
        the result into ``_sessions`` — generation later consumed those
        arrays on the engine's owning thread, violating the same-thread
        rule documented at ``_save_session_to_disk`` (VLM thread
        ownership) — and it held the engine lock across the whole disk
        load. Now:

        - the bounded ``_read_locked`` acquire (EngineBusyError on expiry,
          callers degrade like every other busy path) covers ONLY in-memory
          dict lookups + shallow snapshot copies — never disk IO, never MLX;
        - the disk fallback parses ONLY the safetensors JSON header
          (``_read_safetensors_metadata``, plain file IO) AFTER the lock is
          released — a stale answer is acceptable for an advisory preflight;
        - ``_sessions`` is never written; the authoritative disk load stays
          where it always ran: on the generation thread during the request.

        Callers run it off the event loop (reads executor); the process-mode
        proxy has no such method and callers report a neutral marker."""
        # Phase 1 — SHORT bounded lock: in-memory presence only (dict
        # lookups + shallow copies; the messages list is copied so the
        # match below never walks a list a generation is appending to).
        stored_messages: list | None = None
        cached_tokens = 0
        cache_live = False
        stored_thinking = True
        stored_fingerprint: str | None = None
        with self._read_locked("cache preflight"):
            session_state = self._sessions.get(session_id)
            if session_state is not None:
                stored_messages = list(session_state.messages)
                cached_tokens = session_state.total_cache_tokens
                cache_live = (
                    session_state.cache_state is not None
                    and session_state.cache_state.cache is not None
                )
                stored_thinking = bool(getattr(session_state, "thinking", True))
                # U18 follow-up round 3: capture the stored prompt-contract
                # fingerprint so the gate below mirrors generation
                # (_generate_locked ~5852). None == a legacy/hand-built
                # session with an unknown contract (fail-closed to a MISS).
                stored_fingerprint = getattr(
                    session_state, "prompt_fingerprint", None
                )
            check_disk = session_state is None and self._has_disk_cache(session_id)

        if session_state is not None:
            source = "memory"
        elif check_disk:
            # Phase 2 — disk fallback WITHOUT the engine lock: header-only
            # metadata parse (no mx.load, no cache objects, no _sessions
            # write).
            meta = self._read_safetensors_metadata(
                self._session_cache_path(session_id)
            )
            if meta is not None:
                # U18 follow-up round 2: the save path persists the
                # session's thinking contract ("thinking" metadata key, see
                # _save_session_to_disk) — parse it exactly like
                # _load_session_from_disk so the advisory match and the
                # gemma4 gate below run under the STORED contract instead
                # of the previous hard-coded True.
                stored_thinking = meta.get("thinking", "1") != "0"
                # U18 follow-up round 3: parse the persisted prompt-contract
                # fingerprint (the "prompt_fingerprint" key _save_session_to_disk
                # writes, read exactly like _load_session_from_disk — an empty
                # string is a legacy, pre-fingerprint file == None). The gate
                # below then reports the same MISS generation takes for a
                # pre-revision GLM file (glm-suffix-v2) or a tools/thinking-
                # changed contract, instead of advertising a stale HIT.
                stored_fingerprint = meta.get("prompt_fingerprint") or None
                try:
                    stored_messages = json.loads(meta.get("messages", "[]"))
                    cached_tokens = int(meta.get("total_cache_tokens", "0"))
                except (ValueError, TypeError):
                    stored_messages = None
            if not isinstance(stored_messages, list):
                return {
                    "cache_hit": False,
                    "cache_info": {
                        "type": "none",
                        "detail": (
                            "Disk cache present but metadata unreadable — "
                            "cache state unknown"
                        ),
                    },
                }
            # A persisted file implies a reloadable cache (the generation
            # thread performs — and verifies — the actual load).
            cache_live = True
            source = "disk"
        else:
            return {
                "cache_hit": False,
                "cache_info": {"type": "none", "detail": "New session"},
            }

        # Phase 3 — pure-Python match on the snapshots (no lock, no MLX).
        # Round 3, finding 4: advisory match runs under the stored session's
        # thinking contract (BOTH sources now — memory from SessionState,
        # disk from the persisted "thinking" metadata; the generation
        # re-resolves authoritatively).
        #
        # U18 follow-up round 2 — mirror of the generation-time gemma4
        # fail-closed gate (FINDING B in _generate_locked): gemma4 with
        # thinking disabled NEVER reuses a cached prefix (the template's
        # forced empty-thought block makes cached ids unreconcilable with
        # full retokenization), but this preflight used to compute
        # cache_hit = cache_live without that rule, so the web stream's
        # "start" event advertised a HIT the generation then demoted.
        # Effective thinking mirrors the generation's derivation for this
        # caller: the preflight receives neither a per-request ``thinking``
        # override nor the request's response_format (the web chat sends
        # only session_id + messages, and passes neither to generation
        # either), so the engine default ``cfg.enable_thinking`` IS the
        # request's effective flag here. BOUNDARY: the U9 structured-output
        # suppression (response_format forcing thinking off for one
        # request) cannot be mirrored without that context — should a
        # future caller add it, this advisory may still report a HIT that
        # the generation (authoritative) demotes to a MISS; the residual
        # error is report-only and never fail-open on the actual cache
        # decision. Applied at this convergence point so it covers the
        # memory AND disk branches identically.
        _gemma4_nothink = (
            getattr(self, "model_family", None) == "gemma4"
            and not bool(
                getattr(getattr(self, "cfg", None), "enable_thinking", True)
            )
        )
        # U18 follow-up round 3 (codex, FINDING) — mirror the generation-time
        # prompt-contract fingerprint gate (_generate_locked ~5804/~5852).
        # GENERATION rejects a HIT whenever the STORED fingerprint != the
        # INCOMING EFFECTIVE fingerprint, and a legacy session with NO stored
        # fingerprint takes ONE unconditional cold rebuild (U21/U3/F5). This
        # advisory used to decide the HIT purely from message-match + liveness
        # (+ the gemma-nothink gate), so a pre-v2 GLM disk session (revisionless
        # fingerprint), or any tools/thinking-changed session, was advertised
        # as a HIT the generation then correctly demoted — the web stream's
        # "start" event (chat.py) was observably wrong.
        #
        # The incoming effective fingerprint is computed from what THIS caller
        # sees. The web chat passes only session_id + messages and drives
        # generation with tools=None / response_format=None, so the effective
        # contract here is (tools=None, thinking=cfg.enable_thinking,
        # template_rev=family rev) — exactly what generation computes for the
        # same request, covering the glm-suffix-v2 and gemma-nothink cases.
        # Fail-CLOSED: any uncertainty (missing/unreadable stored fp, or an
        # incoming fp we cannot compute) reports MISS — an advisory MISS only
        # makes the UI show a cold-fill and generation re-resolves
        # authoritatively.
        # BOUNDARY: a future caller routing per-request tools or a
        # response_format thinking override THROUGH this preflight without
        # passing them here could still see an advisory HIT the generation
        # demotes; that residual error is report-only (never fail-open on the
        # actual cache decision), and structured-output OpenAI requests do not
        # use this preflight at all.
        try:
            _incoming_fp: str | None = self._prompt_fingerprint(
                self._canonical_tools(None),
                bool(getattr(getattr(self, "cfg", None), "enable_thinking", True)),
                template_rev=self._suffix_template_rev(),
            )
        except Exception:  # noqa: BLE001 — advisory: any failure => MISS
            _incoming_fp = None
        _contract_ok = (
            stored_fingerprint is not None
            and _incoming_fp is not None
            and stored_fingerprint == _incoming_fp
        )

        cache_hit = False
        gate_demoted = False
        fp_demoted = False
        if self._messages_match(
            stored_messages, messages, thinking_active=stored_thinking,
        ):
            cache_hit = cache_live
            if cache_hit and _gemma4_nothink:
                # Same rule, same direction as the generation gate: report
                # the honest MISS/rebuild it will actually take.
                cache_hit = False
                gate_demoted = True
            if cache_hit and not _contract_ok:
                # Prompt contract diverged (or unknown for a legacy session):
                # the same honest MISS/rebuild the generation fingerprint gate
                # takes. Checked after the gemma gate so a gemma-nothink miss
                # keeps its more specific reason.
                cache_hit = False
                fp_demoted = True
            new_msgs = messages[len(stored_messages):]
            suffix_desc = (
                f"{len(new_msgs)} new message(s)" if new_msgs else "retry"
            )
            cache_info = {
                "type": "kv_cache_hit" if cache_hit else "kv_cache_rebuild",
                "detail": (
                    f"KV Cache reuse ({source}): {cached_tokens} tokens "
                    f"cached, {suffix_desc}"
                    if cache_hit
                    else (
                        f"gemma4 with thinking disabled never reuses a "
                        f"cached prefix — rebuilding {len(messages)} messages"
                        if gate_demoted
                        else (
                            f"prompt contract "
                            f"{'unknown (legacy)' if stored_fingerprint is None else 'changed'}"
                            f" — rebuilding {len(messages)} messages"
                            if fp_demoted
                            else f"Rebuilding KV cache for {len(messages)} messages"
                        )
                    )
                ),
                "cached_tokens": cached_tokens,
                "stored_msgs": len(stored_messages),
                "source": source,
            }
        else:
            cache_info = {
                "type": "kv_cache_miss",
                "detail": (
                    f"Conversation changed, reprocessing "
                    f"{len(messages)} messages"
                ),
                "stored_msgs": len(stored_messages),
                "incoming_msgs": len(messages),
            }
        return {"cache_hit": cache_hit, "cache_info": cache_info}
