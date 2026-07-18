"""Finding A — GLM natural-EOS suffix drift + GLM suffix builder mismatches.

THE BUGS (codex triage, verified against the official GLM-4.7-Flash
chat_template.jinja from huggingface.co/zai-org/GLM-4.7-Flash and its
generation_config eos set {<|endoftext|>, <|user|>, <|observation|>}):

1. Natural-EOS drift (the GLM analog of gemma4 U18): on the mlx-lm backend
   a NATURAL stop records the fired chat-EOS id into cache_state.token_ids
   (the terminal frame carries token=eos; generate_step's lookahead already
   forwarded it through the KV). Full retokenization renders NO terminator
   after an assistant turn — the next turn simply OPENS with its own role
   marker. So:
     - a ``<|user|>`` stop followed by a user turn DOUBLED the marker
       (suffix leads with ``<|user|>``), and
     - a FOREIGN recorded eos (``<|endoftext|>`` always; a role marker
       mismatching the next turn's role) left a token in the KV that full
       retokenization would never render — a bare dedupe cannot fix that
       shape; the token must be TRIMMED out of the cache (C1 machinery) or
       the HIT must demote to an honest MISS.
2. Tool turns rendered ``<|user|><tool_response>\\n{content}\\n</tool_response>``
   where the official template renders ``<|observation|>`` opening a RUN of
   tool messages, each ``<tool_response>{content}</tool_response>`` with no
   newline padding.
3. thinking=False emitted the bare ``<|assistant|>`` generation prompt where
   the template renders ``<|assistant|></think>`` (a primed CLOSED think
   block).

FIX: _suffix_tokens_glm mirrors the official template; the HIT path runs
_reconcile_glm_recorded_eos (dedupe duplicate leading role marker / trim a
foreign recorded eos when every flattened leaf is trimmable and verifies /
demote to an honest MISS otherwise; trim failure invalidates fail-closed).
Old glm sessions (built by the pre-fix builder) were ALREADY drifted from
full retokenization — the suffix-builder revision now participates in the
prompt-contract fingerprint (template_rev), so they take ONE honest MISS
rebuild through the established U21/F5 gate.

No local GLM model exists under ~/.lmstudio/models, so token-exact tests run
on FakeGLMTokenizer, whose apply_chat_template mirrors the official jinja
verbatim for the tested shapes (user/assistant/tool turns, tool_calls,
thinking on/off, reasoning extraction — rendered with transformers'
trim_blocks + lstrip_blocks semantics) and whose encode is injective, so
token-list equality is byte equality.

PRE-FIX FAILURES (verified by reverting the engine changes):
- every *_token_exact / gen-prompt test here fails (missing ``</think>``,
  ``<|user|>`` tool marker, doubled/foreign eos, and the reconcile helper
  does not exist),
- the wiring tests fail: the HIT spliced the drifted sequence
  (test_hit_dedupes_leading_user_after_natural_stop,
  test_hit_trims_foreign_endoftext_then_splices,
  test_hit_foreign_eos_untrimmable_honest_miss,
  test_old_glm_session_fingerprint_migrates_to_miss).
- test_unreconciled_splice_diverges documents the drift itself and passes
  both pre- and post-fix (regression documentation).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_qwen_mtp import (  # noqa: E402
    LyingKV,
    MockKV,
    _drive_full,
    _generate_engine,
    _scripted_lm_stream,
)

# ---------------------------------------------------------------------------
# FakeGLMTokenizer — official GLM-4.7-Flash template mirror
# ---------------------------------------------------------------------------

_SPECIALS = {
    "[gMASK]": 1,
    "<sop>": 2,
    "<|system|>": 3,
    "<|user|>": 4,
    "<|assistant|>": 5,
    "<|observation|>": 6,
    "<|endoftext|>": 7,
    "<think>": 8,
    "</think>": 9,
    "<tool_response>": 10,
    "</tool_response>": 11,
    "<tool_call>": 12,
    "</tool_call>": 13,
    "<arg_key>": 14,
    "</arg_key>": 15,
    "<arg_value>": 16,
    "</arg_value>": 17,
}
USER = _SPECIALS["<|user|>"]
ASSISTANT = _SPECIALS["<|assistant|>"]
OBSERVATION = _SPECIALS["<|observation|>"]
ENDOFTEXT = _SPECIALS["<|endoftext|>"]
THINK_OPEN = _SPECIALS["<think>"]
THINK_CLOSE = _SPECIALS["</think>"]


class FakeGLMTokenizer:
    """Injective tokenizer + a faithful Python port of the official
    GLM-4.7-Flash chat_template.jinja for the shapes the tests exercise.

    Rendering semantics ported 1:1 from the template (as rendered by
    transformers' jinja env with trim_blocks=True, lstrip_blocks=True):
    - preamble '[gMASK]<sop>' (tests pass no tools);
    - user:  '<|user|>' + content;
    - assistant: '<|assistant|>' + ('<think>'+reasoning.strip()+'</think>'
      when reasoning is present AND the turn is after the last user turn,
      else '</think>') + content.strip() (skipped when empty) +
      '<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...
      </tool_call>' per tool call;
    - tool: '<|observation|>' only when the previous message is not also a
      tool message, then '<tool_response>' + content + '</tool_response>';
    - generation prompt: '<|assistant|>' + ('</think>' when thinking is
      disabled else '<think>').
    """

    eos_token_ids = [ENDOFTEXT, USER, OBSERVATION]

    def __init__(self):
        self._sorted_specials = sorted(_SPECIALS, key=len, reverse=True)
        self.vocab_calls = 0

    def get_vocab(self):
        self.vocab_calls += 1
        return dict(_SPECIALS)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for sp in self._sorted_specials:
                if text.startswith(sp, i):
                    ids.append(_SPECIALS[sp])
                    i += len(sp)
                    break
            else:
                ids.append(100 + ord(text[i]))
                i += 1
        return ids

    def decode(self, ids) -> str:
        inv = {v: k for k, v in _SPECIALS.items()}
        return "".join(inv.get(i, chr(i - 100)) for i in ids)

    # --- official template port -------------------------------------------
    def render(self, messages, add_generation_prompt=True, enable_thinking=True):
        out = "[gMASK]<sop>"
        last_user = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user = i
        for i, m in enumerate(messages):
            role = m.get("role")
            content = m.get("content", "") or ""
            if role == "user":
                out += "<|user|>" + content
            elif role == "assistant":
                out += "<|assistant|>"
                reasoning = ""
                if isinstance(m.get("reasoning_content"), str):
                    reasoning = m["reasoning_content"]
                elif "</think>" in content:
                    reasoning = (
                        content.split("</think>")[0]
                        .rstrip("\n")
                        .split("<think>")[-1]
                        .lstrip("\n")
                    )
                    content = content.split("</think>")[-1].lstrip("\n")
                if reasoning and i > last_user:
                    out += "<think>" + reasoning.strip() + "</think>"
                else:
                    out += "</think>"
                if content.strip():
                    out += content.strip()
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    out += "<tool_call>" + fn["name"]
                    for k, v in fn["arguments"].items():
                        val = v if isinstance(v, str) else json.dumps(
                            v, ensure_ascii=False
                        )
                        out += f"<arg_key>{k}</arg_key><arg_value>{val}</arg_value>"
                    out += "</tool_call>"
            elif role == "tool":
                if i == 0 or messages[i - 1].get("role") != "tool":
                    out += "<|observation|>"
                out += "<tool_response>" + content + "</tool_response>"
            elif role == "system":
                out += "<|system|>" + content
        if add_generation_prompt:
            out += "<|assistant|>" + ("<think>" if enable_thinking else "</think>")
        return out

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, **kwargs,
    ):
        text = self.render(
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return self.encode(text) if tokenize else text


def _glm_shell():
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "glm"
    eng.tokenizer = FakeGLMTokenizer()
    return eng


def _cs(token_ids, cache):
    """Minimal cache_state carrier for the reconcile helper."""
    return SimpleNamespace(
        token_ids=list(token_ids), cache=cache,
        mtp_last_hidden=None, mtp_hidden_offset=None,
    )


Q1 = "What is 2 + 2?"
A1 = "The answer is 4."
Q2 = "And 3 + 3?"


def _turn1(eng, thinking: bool, eos_id: int, generated: str = A1):
    """Stored ids per the mlx-lm recording contract: turn-1 prompt via the
    real template, the generated reply, and the recorded chat-EOS id (the
    terminal frame carries token=eos, lookahead already forwarded it)."""
    tok = eng.tokenizer
    prompt1 = tok.apply_chat_template(
        [{"role": "user", "content": Q1}],
        tokenize=True, add_generation_prompt=True, enable_thinking=thinking,
    )
    return prompt1 + tok.encode(generated) + [eos_id]


# ---------------------------------------------------------------------------
# Builder shapes — token-exact against the official template render
# ---------------------------------------------------------------------------


def test_thinking_false_gen_prompt_closed_think():
    """The thinking=False generation prompt must render
    '<|assistant|></think>' (official template), not the bare marker.
    PRE-FIX: fails — the old builder emitted '<|assistant|>' only."""
    eng = _glm_shell()
    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    assert suffix == eng.tokenizer.encode(f"<|user|>{Q2}<|assistant|></think>")
    assert suffix[-1] == THINK_CLOSE


def test_thinking_true_gen_prompt_open_think():
    """thinking=True keeps the '<|assistant|><think>' opener (unchanged
    contract guard)."""
    eng = _glm_shell()
    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    assert suffix == eng.tokenizer.encode(f"<|user|>{Q2}<|assistant|><think>")
    assert suffix[-1] == THINK_OPEN


def test_tool_suffix_uses_observation_marker_no_padding():
    """Tool messages render '<|observation|><tool_response>{content}
    </tool_response>' per the official template. PRE-FIX: fails — the old
    builder rendered '<|user|><tool_response>\\n{content}\\n</tool_response>'."""
    eng = _glm_shell()
    suffix = eng._suffix_tokens_glm(
        [{"role": "tool", "content": "sunny"}], thinking=False,
    )
    assert suffix == eng.tokenizer.encode(
        "<|observation|><tool_response>sunny</tool_response>"
        "<|assistant|></think>"
    )
    assert suffix[0] == OBSERVATION


def test_consecutive_tool_messages_share_one_observation_marker():
    """A RUN of tool messages gets ONE '<|observation|>' (the template emits
    the marker only when the previous message is not also a tool message).
    PRE-FIX: fails — one '<|user|>' marker per message."""
    eng = _glm_shell()
    tok = eng.tokenizer
    msgs = [
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
    ]
    suffix = eng._suffix_tokens_glm(msgs, thinking=False)
    assert suffix.count(OBSERVATION) == 1
    assert suffix.count(USER) == 0
    # Byte-exact vs the official render of the same tail.
    expected = tok.encode(
        "<|observation|><tool_response>r1</tool_response>"
        "<tool_response>r2</tool_response><|assistant|></think>"
    )
    assert suffix == expected


# ---------------------------------------------------------------------------
# Natural-EOS reconciliation — splice == full retokenization, byte-exact
# ---------------------------------------------------------------------------


def test_natural_user_stop_splice_token_exact_nothink():
    """<|user|> recorded at a natural stop, next turn is a user turn: the
    reconcile drops the suffix's duplicate leading marker and the splice
    equals apply_chat_template over the whole conversation. PRE-FIX: fails
    (doubled <|user|> AND missing </think> in the generation prompt)."""
    eng = _glm_shell()
    tok = eng.tokenizer
    stored = _turn1(eng, thinking=False, eos_id=USER)
    cs = _cs(stored, [MockKV()])

    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    assert suffix[0] == USER
    suffix = eng._reconcile_glm_recorded_eos(suffix, cs, "s")
    assert suffix is not None and suffix[0] != USER
    assert cs.token_ids == stored  # dedupe never mutates the cache

    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert cs.token_ids + suffix == full


def test_unreconciled_splice_diverges():
    """Regression documentation (the finding-A drift itself): WITHOUT the
    reconcile the splice diverges from full retokenization with a doubled
    back-to-back <|user|>. Passes pre- and post-fix."""
    eng = _glm_shell()
    tok = eng.tokenizer
    stored = _turn1(eng, thinking=False, eos_id=USER)
    raw_suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    spliced = stored + raw_suffix
    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert spliced != full
    assert any(
        spliced[i] == USER and spliced[i + 1] == USER
        for i in range(len(spliced) - 1)
    )
    assert not any(
        full[i] == USER and full[i + 1] == USER for i in range(len(full) - 1)
    )


def test_natural_observation_stop_before_tool_turn_token_exact():
    """The model emitted a tool call and stopped with <|observation|> (its
    tool-boundary eos); the next message IS the tool response, whose suffix
    leads with the same marker — dedupe, splice byte-exact. PRE-FIX: fails
    (old tool suffix shape + no reconcile)."""
    eng = _glm_shell()
    tok = eng.tokenizer
    tc_text = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Seoul</arg_value></tool_call>"
    )
    stored = _turn1(eng, thinking=False, eos_id=OBSERVATION, generated=tc_text)
    cs = _cs(stored, [MockKV()])

    suffix = eng._suffix_tokens_glm(
        [{"role": "tool", "content": "sunny"}], thinking=False,
    )
    assert suffix[0] == OBSERVATION
    suffix = eng._reconcile_glm_recorded_eos(suffix, cs, "s")
    assert suffix is not None and suffix[0] != OBSERVATION
    assert cs.token_ids == stored

    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Seoul"},
                    },
                }],
            },
            {"role": "tool", "content": "sunny"},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert cs.token_ids + suffix == full


def test_foreign_endoftext_stop_trims_and_splices_token_exact():
    """<|endoftext|> recorded at a natural stop: full retokenization renders
    NO token there — the reconcile trims the single trailing id out of the
    (fully trimmable) cache and the splice equals full retokenization.
    PRE-FIX: fails — the foreign token stayed in the spliced context."""
    eng = _glm_shell()
    tok = eng.tokenizer
    stored = _turn1(eng, thinking=False, eos_id=ENDOFTEXT)
    cache = [MockKV(), MockKV(), MockKV()]
    for c in cache:
        c.offset = len(stored)
    cs = _cs(stored, cache)

    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    out = eng._reconcile_glm_recorded_eos(suffix, cs, "s")
    assert out == suffix  # suffix untouched; the CACHE side was reconciled
    assert cs.token_ids == stored[:-1]
    assert all(c.offset == len(stored) - 1 for c in cache)

    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert cs.token_ids + out == full


def test_role_mismatched_recorded_eos_is_foreign_and_trimmed():
    """<|user|> recorded but the next message is a TOOL turn (and the
    mirror shape): the marker mismatches the suffix's leading marker, so it
    is FOREIGN — trimmed, then byte-exact. PRE-FIX: fails."""
    eng = _glm_shell()
    tok = eng.tokenizer
    tc_text = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        "<arg_value>Seoul</arg_value></tool_call>"
    )
    # Shape 1: <|user|> stop, tool turn next.
    stored = _turn1(eng, thinking=False, eos_id=USER, generated=tc_text)
    cache = [MockKV(), MockKV()]
    for c in cache:
        c.offset = len(stored)
    cs = _cs(stored, cache)
    suffix = eng._suffix_tokens_glm(
        [{"role": "tool", "content": "sunny"}], thinking=False,
    )
    out = eng._reconcile_glm_recorded_eos(suffix, cs, "s")
    assert out == suffix
    assert cs.token_ids == stored[:-1]
    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Seoul"},
                    },
                }],
            },
            {"role": "tool", "content": "sunny"},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert cs.token_ids + out == full

    # Shape 2 (mirror): <|observation|> stop, user turn next.
    stored2 = _turn1(eng, thinking=False, eos_id=OBSERVATION)
    cache2 = [MockKV()]
    cache2[0].offset = len(stored2)
    cs2 = _cs(stored2, cache2)
    suffix2 = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    out2 = eng._reconcile_glm_recorded_eos(suffix2, cs2, "s")
    assert out2 == suffix2
    assert cs2.token_ids == stored2[:-1]
    full2 = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert cs2.token_ids + out2 == full2


def test_foreign_eos_untrimmable_demotes_to_miss():
    """A foreign recorded eos on an untrimmable cache returns None (honest
    MISS) and leaves the cache INTACT and coherent (offset still equals
    len(token_ids); the rebuild's save replaces it)."""
    eng = _glm_shell()
    stored = _turn1(eng, thinking=False, eos_id=ENDOFTEXT)
    untrimmable = SimpleNamespace(offset=len(stored))  # no trim() at all
    cs = _cs(stored, [untrimmable])
    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    assert eng._reconcile_glm_recorded_eos(suffix, cs, "s") is None
    assert cs.token_ids == stored
    assert cs.cache is not None
    assert untrimmable.offset == len(stored)


def test_foreign_eos_trim_failure_invalidates_fail_closed():
    """A lying trim (claims success, never decrements) fails the per-leaf
    verification — the session cache is INVALIDATED (half-rewound layers
    must never survive into reuse)."""
    eng = _glm_shell()
    stored = _turn1(eng, thinking=False, eos_id=ENDOFTEXT)
    good, liar = MockKV(), LyingKV()
    good.offset = liar.offset = len(stored)
    cs = _cs(stored, [good, liar])
    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=False,
    )
    assert eng._reconcile_glm_recorded_eos(suffix, cs, "s") is None
    assert cs.cache is None
    assert cs.token_ids is None
    assert cs.mtp_last_hidden is None
    assert cs.mtp_hidden_offset is None


def test_reconcile_edge_cases():
    """Contract guards: content tail (interrupted turn) splices untouched;
    empty stored/suffix untouched; non-glm families untouched; degraded
    marker detection fails open (never trims what it cannot verify)."""
    eng = _glm_shell()
    content_tail = [1000, 1001, 1002]  # no eos recorded (C1 interrupted)
    cs = _cs(content_tail, [MockKV()])
    suffix = [USER, 555]
    assert eng._reconcile_glm_recorded_eos(suffix, cs, "s") == suffix
    assert cs.token_ids == content_tail

    assert eng._reconcile_glm_recorded_eos([], cs, "s") == []
    cs_empty = _cs([], [MockKV()])
    assert eng._reconcile_glm_recorded_eos(suffix, cs_empty, "s") == suffix

    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    other = MLXEngine.__new__(MLXEngine)
    other.model_family = "chatml"
    other.tokenizer = FakeGLMTokenizer()
    cs2 = _cs([1, USER], [MockKV()])
    assert other._reconcile_glm_recorded_eos([USER, 5], cs2, "s") == [USER, 5]

    degraded = MLXEngine.__new__(MLXEngine)
    degraded.model_family = "glm"
    degraded.tokenizer = SimpleNamespace(
        get_vocab=lambda: {},
        encode=lambda t, add_special_tokens=False: [],
        eos_token_ids=[],
    )
    cs3 = _cs([1, USER], [MockKV()])
    assert degraded._reconcile_glm_recorded_eos([USER, 5], cs3, "s") == [USER, 5]
    assert cs3.token_ids == [1, USER]


def test_marker_ids_memoized():
    """_glm_marker_ids builds the vocab dict at most once per engine."""
    eng = _glm_shell()
    ids1 = eng._glm_marker_ids()
    ids2 = eng._glm_marker_ids()
    assert ids1 is ids2
    assert eng.tokenizer.vocab_calls == 3  # one _detect_token_id per marker
    assert ids1["<|user|>"] == USER
    assert ids1["<|observation|>"] == OBSERVATION
    assert ids1["<|endoftext|>"] == ENDOFTEXT


def test_three_turn_natural_stops_no_drift_nothink():
    """Two natural stops in a row (mixed <|user|> then <|endoftext|>): the
    reconcile keeps the accumulated ids token-exact against the 3-turn
    retokenization — no per-turn drift. PRE-FIX: fails."""
    eng = _glm_shell()
    tok = eng.tokenizer
    A2 = "That makes 6."
    Q3 = "Now multiply them."
    stored = _turn1(eng, thinking=False, eos_id=USER)
    cache = [MockKV()]
    cache[0].offset = len(stored)
    cs = _cs(stored, cache)

    s2 = eng._reconcile_glm_recorded_eos(
        eng._suffix_tokens_glm([{"role": "user", "content": Q2}], thinking=False),
        cs, "s",
    )
    cs.token_ids = cs.token_ids + s2 + tok.encode(A2) + [ENDOFTEXT]
    cache[0].offset = len(cs.token_ids)

    s3 = eng._reconcile_glm_recorded_eos(
        eng._suffix_tokens_glm([{"role": "user", "content": Q3}], thinking=False),
        cs, "s",
    )
    spliced = cs.token_ids + s3

    full = tok.apply_chat_template(
        [
            {"role": "user", "content": Q1},
            {"role": "assistant", "content": A1},
            {"role": "user", "content": Q2},
            {"role": "assistant", "content": A2},
            {"role": "user", "content": Q3},
        ],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    assert spliced == full


def test_thinking_true_dedupe_boundary():
    """thinking=True conversations retain reasoning in the cache BY DESIGN
    (the campaign's retained-reasoning deviation — history in the KV is what
    the model generated, while retokenization strips it), so full-sequence
    equality is not the contract there. The recorded-EOS reconciliation is
    boundary-local and must behave identically: dedupe the duplicated
    <|user|> and produce the template's exact next-turn tail."""
    eng = _glm_shell()
    tok = eng.tokenizer
    generated = "some reasoning</think>The answer is 4."
    stored = _turn1(eng, thinking=True, eos_id=USER, generated=generated)
    cs = _cs(stored, [MockKV()])
    suffix = eng._suffix_tokens_glm(
        [{"role": "user", "content": Q2}], thinking=True,
    )
    out = eng._reconcile_glm_recorded_eos(suffix, cs, "s")
    assert out == tok.encode(f"{Q2}<|assistant|><think>")  # marker deduped
    assert cs.token_ids == stored


# ---------------------------------------------------------------------------
# Engine wiring — the reconcile runs on the real HIT path
# ---------------------------------------------------------------------------


def _glm_hit_engine(stored, suffix, *, cache=None):
    """HIT-capable mlx-lm-path harness re-labeled glm, with a vocab that
    resolves the three markers, and the session contract restamped with the
    glm suffix-builder revision (the fingerprint migration would otherwise
    demote every drive to a MISS)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    if cache is None:
        cache = [MockKV() for _ in range(5)]
        for c in cache:
            c.offset = len(stored)
    eng, cs, messages = _generate_engine(cache, stored, mtp=False, suffix=suffix)
    eng.model_family = "glm"
    eng.tokenizer = SimpleNamespace(
        decode=lambda ids: "x",
        eos_token_ids=[],
        get_vocab=lambda: {
            "<|user|>": USER,
            "<|observation|>": OBSERVATION,
            "<|endoftext|>": ENDOFTEXT,
        },
        encode=lambda text, add_special_tokens=False: [],
    )
    sess = eng._sessions["s"]
    sess.prompt_fingerprint = MLXEngine._prompt_fingerprint(
        None, False, template_rev=MLXEngine._GLM_SUFFIX_REV,
    )
    return eng, cs, messages


def test_hit_dedupes_leading_user_after_natural_stop(monkeypatch):
    """Stored tail IS <|user|> (natural stop): the HIT prefill must receive
    the suffix WITHOUT the duplicate leading marker. PRE-FIX: fails — the
    prefill got [USER, 55, 66]."""
    stored = [1, 2, 3, 4, USER]
    eng, cs, messages = _glm_hit_engine(stored, suffix=[USER, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [55, 66]
    assert cs.token_ids == stored + [55, 66] + [101, 102]


def test_hit_trims_foreign_endoftext_then_splices(monkeypatch):
    """Stored tail is <|endoftext|> (foreign to the next turn's rendering):
    the HIT trims it from cache + token_ids, then splices the full suffix.
    PRE-FIX: fails — the foreign id stayed in the model context."""
    stored = [1, 2, 3, 4, ENDOFTEXT]
    eng, cs, messages = _glm_hit_engine(stored, suffix=[USER, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [USER, 55, 66]
    assert cs.token_ids == stored[:-1] + [USER, 55, 66] + [101, 102]
    # Every offset-bearing layer landed exactly on the recorded ids.
    assert all(c.offset == len(cs.token_ids) for c in cs.cache)


def test_hit_foreign_eos_untrimmable_honest_miss(monkeypatch):
    """Foreign recorded eos + untrimmable cache: the HIT demotes to an
    honest MISS — full retokenization is prefilled and the rebuilt session
    replaces the old cache. PRE-FIX: fails — the HIT spliced the drifted
    sequence."""
    stored = [1, 2, 3, 4, ENDOFTEXT]
    untrimmable = [SimpleNamespace(offset=len(stored)) for _ in range(3)]
    eng, cs, messages = _glm_hit_engine(
        stored, suffix=[USER, 55, 66], cache=untrimmable,
    )
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13]
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [11, 12, 13]
    new_cs = eng._sessions["s"].cache_state
    assert new_cs is not cs
    assert new_cs.token_ids == [11, 12, 13, 101, 102]


def test_hit_interrupted_tail_splices_unchanged(monkeypatch):
    """Stored tail is content (C1 interrupted commit — no eos recorded):
    the suffix splices as built (contract guard, passes pre- and post-fix)."""
    stored = [1, 2, 3, 4, 5]
    eng, cs, messages = _glm_hit_engine(stored, suffix=[USER, 55, 66])
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [USER, 55, 66]
    assert cs.token_ids == stored + [USER, 55, 66] + [101, 102]


def test_old_glm_session_fingerprint_migrates_to_miss(monkeypatch):
    """A session stamped with the PRE-REVISION fingerprint (built by the old
    suffix builder — its stored ids were already drifted) must take ONE
    honest MISS rebuild that restamps the revised contract. PRE-FIX: fails —
    old and new fingerprints were identical, so the session kept HITting."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    stored = [1, 2, 3, 4, USER]
    eng, cs, messages = _glm_hit_engine(stored, suffix=[USER, 55, 66])
    # Overwrite with the OLD (rev-less) stamp — what a pre-fix session file
    # carries on disk.
    eng._sessions["s"].prompt_fingerprint = MLXEngine._prompt_fingerprint(
        None, False,
    )
    eng._tokenize_prompt = lambda msgs, thinking=True, tools=None: [11, 12, 13]
    prompts_seen: list = []
    _scripted_lm_stream(monkeypatch, prompts_seen, [[(101, "a"), (102, "b")]])

    _drive_full(eng, messages)

    assert prompts_seen[0] == [11, 12, 13]  # honest MISS, not a HIT splice
    new_sess = eng._sessions["s"]
    assert new_sess.prompt_fingerprint == MLXEngine._prompt_fingerprint(
        None, False, template_rev=MLXEngine._GLM_SUFFIX_REV,
    )

    # Second drive on the restamped session: contract now matches → HIT
    # (suffix prefill, not another full retokenization; the rebuilt tail is
    # a content token, so the suffix splices whole).
    messages2 = list(new_sess.messages) + [{"role": "user", "content": "u3"}]
    _drive_full(eng, messages2)
    assert prompts_seen[1] == [USER, 55, 66]


def test_fingerprint_rev_only_changes_glm():
    """The template_rev key is omitted when empty: chatml/gemma4 fingerprints
    are byte-identical to the historical hash (no spurious one-time MISS for
    non-glm sessions)."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    base = MLXEngine._prompt_fingerprint(None, True)
    assert MLXEngine._prompt_fingerprint(None, True, template_rev="") == base
    assert MLXEngine._prompt_fingerprint(
        None, True, template_rev=MLXEngine._GLM_SUFFIX_REV,
    ) != base

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "chatml"
    assert eng._suffix_template_rev() == ""
    eng.model_family = "gemma4"
    assert eng._suffix_template_rev() == ""
    eng.model_family = "glm"
    assert eng._suffix_template_rev() == MLXEngine._GLM_SUFFIX_REV


# ---------------------------------------------------------------------------
# U18 follow-up round 3 (codex, FINDING) — session_cache_preflight must
# compare the STORED prompt-contract fingerprint against the INCOMING
# EFFECTIVE fingerprint, exactly as generation does (_generate_locked
# ~5804/~5852). Pre-fix the advisory reported cache_hit purely from
# message-match + liveness (+ the gemma-nothink gate), so a pre-revision GLM
# session (a revisionless / glm-suffix-v1 fingerprint) was advertised as a
# HIT the generation then correctly demoted to a MISS — the web stream's
# "start" event (chat.py) was observably wrong. This is the glm-suffix-v2
# instance of the general tools/thinking-changed (U21/U3) fingerprint
# divergence the advisory now mirrors on BOTH the memory and disk branches.
#
# PRE-FIX FAILURES (verified against the fingerprint-blind preflight):
# - test_preflight_memory_glm_prerev_fingerprint_reports_miss (cache_hit True),
# - test_preflight_disk_glm_legacy_fingerprint_reports_miss[absent/prerev]
#   (cache_hit True).
# The matching-fingerprint HITs and the non-glm regression (in the nothink
# gate file) stay green: a genuinely reusable session carries the matching
# fingerprint, so the new comparison never spuriously MISSes.
# ---------------------------------------------------------------------------


PREFLIGHT_MSGS = [
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
]


def _glm_preflight_shell(tmp_path, *, enable_thinking=False):
    """Minimal session_cache_preflight shell re-labeled glm (conventions from
    tests/test_gemma4_nothink_hit_gate.py). The advisory matcher is stubbed
    True so these tests isolate the prompt-contract fingerprint gate; the
    real _prompt_fingerprint / _canonical_tools / _suffix_template_rev run."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = MLXEngine.__new__(MLXEngine)
    eng.model_family = "glm"
    eng._lock = threading.Lock()
    eng.cfg = SimpleNamespace(
        cache_dir=str(tmp_path), enable_thinking=enable_thinking,
    )
    eng._sessions = {}
    eng._has_disk_cache = lambda sid: False
    eng._messages_match = lambda stored, incoming, **kw: True
    return eng


def _incoming_preflight_fp(eng) -> str:
    """The fingerprint the preflight computes for a web-chat request here:
    tools=None, thinking=cfg.enable_thinking, template_rev=glm-suffix-v2."""
    return eng._prompt_fingerprint(
        eng._canonical_tools(None),
        bool(eng.cfg.enable_thinking),
        template_rev=eng._suffix_template_rev(),
    )


def _glm_live_session(*, fingerprint):
    return SimpleNamespace(
        messages=list(PREFLIGHT_MSGS),
        total_cache_tokens=3,
        cache_state=SimpleNamespace(cache=object(), token_ids=[1, 2, 3]),
        thinking=False,
        prompt_fingerprint=fingerprint,
    )


def _write_disk_header(tmp_path, session_id, metadata):
    """Hand-crafted safetensors file: 8-byte LE header length + JSON header
    carrying ``__metadata__`` (the layout mx.save_safetensors writes). Plain
    IO, no MLX — matches session_cache_preflight's header-only parse."""
    header = json.dumps({"__metadata__": metadata}).encode()
    path = tmp_path / f"session_{session_id}.safetensors"
    with open(path, "wb") as f:
        f.write(len(header).to_bytes(8, "little"))
        f.write(header)
    return path


def test_preflight_memory_glm_prerev_fingerprint_reports_miss(tmp_path):
    """A live, message-matching in-memory GLM session stamped with the
    PRE-REVISION (rev-less) fingerprint — what a pre-v2 session carried —
    must report the honest MISS/rebuild the generation fingerprint gate takes,
    never advertise a HIT. PRE-FIX: cache_hit was True."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = _glm_preflight_shell(tmp_path)
    eng._sessions["s1"] = _glm_live_session(
        fingerprint=MLXEngine._prompt_fingerprint(None, False),  # rev-less
    )

    out = eng.session_cache_preflight(
        "s1", PREFLIGHT_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_rebuild"
    assert "prompt contract" in out["cache_info"]["detail"]


def test_preflight_memory_glm_matching_fingerprint_hits(tmp_path):
    """Scope guard: a GLM session stamped with the CURRENT glm-suffix-v2
    fingerprint (== the incoming effective one) still preflight HITs — the new
    comparison never spuriously MISSes a genuinely reusable session."""
    eng = _glm_preflight_shell(tmp_path)
    eng._sessions["s1"] = _glm_live_session(
        fingerprint=_incoming_preflight_fp(eng),
    )

    out = eng.session_cache_preflight(
        "s1", PREFLIGHT_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"
    assert out["cache_info"]["source"] == "memory"


@pytest.mark.parametrize(
    "stored_fp",
    [None, "rev-less"],
    ids=["absent", "prerev"],
)
def test_preflight_disk_glm_legacy_fingerprint_reports_miss(tmp_path, stored_fp):
    """The DISK branch parses the persisted "prompt_fingerprint" metadata and
    reports a MISS for a pre-v2 GLM file — whether the key is ABSENT (a
    legacy, pre-fingerprint save, read as None) or present but REV-LESS (a
    session built by the old suffix builder). PRE-FIX: cache_hit was True on
    both."""
    from mlx_soloheaven.engine.mlx_engine import MLXEngine

    eng = _glm_preflight_shell(tmp_path)
    meta = {
        "messages": json.dumps(PREFLIGHT_MSGS),
        "total_cache_tokens": "3",
        "thinking": "0",
    }
    if stored_fp == "rev-less":
        meta["prompt_fingerprint"] = MLXEngine._prompt_fingerprint(None, False)
    _write_disk_header(tmp_path, "s2", meta)
    eng._has_disk_cache = lambda sid: sid == "s2"

    out = eng.session_cache_preflight(
        "s2", PREFLIGHT_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is False
    assert out["cache_info"]["type"] == "kv_cache_rebuild"
    assert "prompt contract" in out["cache_info"]["detail"]


def test_preflight_disk_glm_matching_fingerprint_hits(tmp_path):
    """A GLM disk file carrying the CURRENT glm-suffix-v2 fingerprint reports
    a HIT — the disk gate mirrors generation only for the honest divergence,
    not for an up-to-date contract."""
    eng = _glm_preflight_shell(tmp_path)
    _write_disk_header(tmp_path, "s2", {
        "messages": json.dumps(PREFLIGHT_MSGS),
        "total_cache_tokens": "3",
        "thinking": "0",
        "prompt_fingerprint": _incoming_preflight_fp(eng),
    })
    eng._has_disk_cache = lambda sid: sid == "s2"

    out = eng.session_cache_preflight(
        "s2", PREFLIGHT_MSGS + [{"role": "user", "content": "u2"}]
    )

    assert out["cache_hit"] is True
    assert out["cache_info"]["type"] == "kv_cache_hit"
    assert out["cache_info"]["source"] == "disk"
