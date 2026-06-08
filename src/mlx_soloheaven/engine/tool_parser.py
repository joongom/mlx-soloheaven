"""Parse model XML tool calls to/from OpenAI JSON format.

Supports Qwen/ChatML, GLM, and Gemma 4 tool call formats.
"""

import json
import re
import uuid
from typing import Optional


def generate_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


# Per-family tool_call block markers. The streaming emitter uses these to
# detect block boundaries and extract the function name as soon as it's
# determinable, so the first OpenAI-format chunk can be emitted early
# instead of waiting for the whole block to close.
_TOOL_MARKERS = {
    "chatml": ("<tool_call>", "</tool_call>"),
    "qwen":   ("<tool_call>", "</tool_call>"),
    "glm":    ("<tool_call>", "</tool_call>"),
    "gemma4": ("<|tool_call>", "<tool_call|>"),
}


def get_tool_markers(model_family: str) -> tuple[str, str]:
    return _TOOL_MARKERS.get(model_family, _TOOL_MARKERS["chatml"])


def try_extract_tool_name(buf_after_start: str, model_family: str) -> Optional[str]:
    """Extract function name from text buffered *after* the start marker.

    Returns the name if determinable, else None (need more text).
    """
    if model_family == "gemma4":
        m = re.match(r"\s*call:(\w+)\s*\{", buf_after_start)
        return m.group(1) if m else None
    if model_family == "glm":
        # GLM: name is bare text between <tool_call> and first <arg_key>
        # (or </tool_call> for no-args calls). Some GLM variants follow Qwen.
        first_ak = buf_after_start.find("<arg_key>")
        first_fn = buf_after_start.find("<function=")
        first_end = buf_after_start.find("</tool_call>")
        if first_fn >= 0 and (first_ak < 0 or first_fn < first_ak):
            m = re.match(r"\s*<function=(\w+)>", buf_after_start)
            return m.group(1) if m else None
        cutoffs = [p for p in (first_ak, first_end) if p >= 0]
        if not cutoffs:
            return None
        name = buf_after_start[:min(cutoffs)].strip().lstrip("\n").strip()
        return name or None
    # Qwen / chatml default
    m = re.match(r"\s*<function=(\w+)>", buf_after_start)
    return m.group(1) if m else None


def parse_tool_calls(text: str, model_family: str = "chatml") -> tuple[str, list[dict]]:
    """
    Parse tool_call blocks from model output.

    Supports:
    - ChatML/Qwen: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    - GLM: <tool_call>function_name<arg_key>key</arg_key><arg_value>value</arg_value>...</tool_call>
    - Gemma 4: <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>

    Returns:
        (content_text, tool_calls)
        - content_text: text before any tool_call block
        - tool_calls: list of OpenAI-format tool call dicts
    """
    if model_family == "gemma4":
        return _parse_gemma4_tool_calls(text)
    if model_family == "glm":
        # GLM format has <arg_key>/<arg_value> pairs; if not found, fall through
        # to Qwen-style parsing (some GLM variants follow Qwen format).
        if "<arg_key>" in text or "<arg_value>" in text:
            return _parse_glm_tool_calls(text)
    return _parse_chatml_tool_calls(text)


def _parse_chatml_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse Qwen/ChatML XML tool call format."""
    tool_call_pattern = re.compile(
        r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )

    first_tc = text.find("<tool_call>")
    if first_tc == -1:
        return text, []

    content_text = text[:first_tc].rstrip()
    tool_calls = []

    for match in tool_call_pattern.finditer(text):
        func_name = match.group(1)
        params_block = match.group(2)

        param_pattern = re.compile(
            r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
        )
        arguments = {}
        for pm in param_pattern.finditer(params_block):
            key = pm.group(1)
            value = pm.group(2).strip()
            try:
                arguments[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = value

        tool_calls.append({
            "id": generate_call_id(),
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    return content_text, tool_calls


def _parse_glm_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse GLM-family tool call format.

    Format (from GLM chat_template.jinja):
        <tool_call>{function_name}
        <arg_key>{key1}</arg_key><arg_value>{val1}</arg_value>
        <arg_key>{key2}</arg_key><arg_value>{val2}</arg_value>
        ...
        </tool_call>

    The function name is the bare text immediately following <tool_call>
    and preceding the first <arg_key>. Key/value pairs alternate.
    Values may be JSON strings, numbers, booleans, or nested JSON; we
    attempt json.loads per value and fall back to raw string.
    """
    first_tc = text.find("<tool_call>")
    if first_tc == -1:
        return text, []

    content_text = text[:first_tc].rstrip()
    tool_calls = []

    # Full tool_call block pattern (non-greedy, tolerate missing </tool_call>)
    tc_pattern = re.compile(
        r"<tool_call>(.*?)(?:</tool_call>|\Z)",
        re.DOTALL,
    )
    kv_pattern = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
        re.DOTALL,
    )

    for tc_match in tc_pattern.finditer(text):
        inner = tc_match.group(1)
        # Function name: text before the first <arg_key> (or the whole inner
        # text if there are no args)
        first_arg = inner.find("<arg_key>")
        if first_arg >= 0:
            func_name = inner[:first_arg].strip()
            args_text = inner[first_arg:]
        else:
            func_name = inner.strip()
            args_text = ""
        # Some models prepend a newline or whitespace inside <tool_call>
        func_name = func_name.strip().lstrip("\n").strip()
        if not func_name:
            continue

        arguments = {}
        for kv in kv_pattern.finditer(args_text):
            key = kv.group(1).strip()
            value = kv.group(2).strip()
            # Try to parse as JSON (handles numbers, booleans, objects, arrays)
            try:
                arguments[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                arguments[key] = value

        tool_calls.append({
            "id": generate_call_id(),
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    return content_text, tool_calls


def _parse_gemma4_value(raw: str):
    """Parse a single value from Gemma 4 tool call format.

    Values can be:
    - String: delimited by <|"|>...<|"|>
    - Number: bare digits (int or float)
    - Boolean: true/false
    - Nested object: {...}
    - Array: [...]
    """
    raw = raw.strip()
    if not raw:
        return raw
    # Try as number
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    # Boolean
    if raw == "true":
        return True
    if raw == "false":
        return False
    return raw


def _parse_gemma4_array(arr_str: str) -> list:
    """Parse the inner contents of a Gemma 4 array (without outer [] brackets).

    Splits on TOP-LEVEL commas only, respecting:
    - <|"|>...<|"|> string spans (commas inside a string are not separators;
      the delimiter tokens are stripped from the resulting string value)
    - {...} object nesting (recurse into _parse_gemma4_args -> dict)
    - [...] array nesting (recurse into _parse_gemma4_array -> list)
    Bare tokens between top-level commas go through _parse_gemma4_value.
    """
    delim = "<|\"" + "|>"
    elements = []
    i = 0
    n = len(arr_str)
    elem_start = 0  # start of the current top-level element

    def _flush(start: int, end: int) -> None:
        seg = arr_str[start:end].strip()
        if not seg:
            return
        if seg.startswith(delim) and seg.endswith(delim) and len(seg) >= 2 * len(delim):
            elements.append(seg[len(delim):-len(delim)])
        elif seg.startswith("{"):
            elements.append(_parse_gemma4_args(seg))
        elif seg.startswith("["):
            inner = seg[1:-1] if seg.endswith("]") else seg[1:]
            elements.append(_parse_gemma4_array(inner))
        else:
            elements.append(_parse_gemma4_value(seg))

    while i < n:
        if arr_str.startswith(delim, i):
            # Skip over a string span (its inner commas are not separators)
            end = arr_str.find(delim, i + len(delim))
            i = (end + len(delim)) if end != -1 else n
        elif arr_str[i] == "{":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
        elif arr_str[i] == "[":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
        elif arr_str[i] == ",":
            # Top-level separator
            _flush(elem_start, i)
            i += 1
            elem_start = i
        else:
            i += 1

    # Flush the final element
    _flush(elem_start, n)
    return elements


def _parse_gemma4_args(args_str: str) -> dict:
    """Parse Gemma 4 custom struct format: {key:<|"|>val<|"|>,key2:val2}

    The <|"|> token is used as string delimiter (like quotes).
    Keys are bare identifiers separated by colons.
    """
    if not args_str or args_str == "{}":
        return {}

    # Strip outer braces
    s = args_str.strip()
    if s.startswith("{"):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]

    result = {}
    i = 0
    while i < len(s):
        # Skip whitespace/commas
        while i < len(s) and s[i] in " ,\n\t":
            i += 1
        if i >= len(s):
            break

        # Read key (until ':')
        key_start = i
        while i < len(s) and s[i] != ":":
            i += 1
        key = s[key_start:i].strip()
        if not key:
            break
        i += 1  # skip ':'

        # Read value
        if i >= len(s):
            break

        if s[i:].startswith("<|\"" + "|>"):
            # String value: <|"|>...<|"|>
            delim = "<|\"" + "|>"
            i += len(delim)
            end = s.find(delim, i)
            if end == -1:
                result[key] = s[i:]
                break
            result[key] = s[i:end]
            i = end + len(delim)
        elif s[i] == "{":
            # Nested object — find matching brace
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            result[key] = _parse_gemma4_args(s[i:j])
            i = j
        elif s[i] == "[":
            # Array — find matching bracket
            depth = 1
            j = i + 1
            while j < len(s) and depth > 0:
                if s[j] == "[":
                    depth += 1
                elif s[j] == "]":
                    depth -= 1
                j += 1
            # Span-aware parsing: split only on top-level commas, respecting
            # <|"|> string spans and nested {}/[].
            arr_str = s[i + 1 : j - 1]
            result[key] = _parse_gemma4_array(arr_str)
            i = j
        else:
            # Bare value (number, boolean) — read until comma or end
            val_start = i
            while i < len(s) and s[i] not in ",}":
                i += 1
            result[key] = _parse_gemma4_value(s[val_start:i])

    return result


def _parse_gemma4_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse Gemma 4 tool call format: <|tool_call>call:name{args}<tool_call|>"""
    pattern = re.compile(
        r"<\|tool_call>call:(\w+)(\{.*?\})<tool_call\|>",
        re.DOTALL,
    )

    first_tc = text.find("<|tool_call>")
    if first_tc == -1:
        return text, []

    content_text = text[:first_tc].rstrip()
    tool_calls = []

    for match in pattern.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2)
        arguments = _parse_gemma4_args(args_str)

        tool_calls.append({
            "id": generate_call_id(),
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    return content_text, tool_calls


# Gemma 4 thinking-channel markers. The well-formed shape is
# ``<|channel>thought\n...reasoning...<channel|>answer``; degenerate
# multi-cycle / sliding-window outputs emit repeated or orphaned markers.
_GEMMA4_CHANNEL_OPEN = "<|channel>"
_GEMMA4_CHANNEL_CLOSE = "<channel|>"
# Thought-channel opener tokens. Gemma 4 emits the full ``<|channel>thought\n``
# form; the sliding-window / degenerate variant drops the ``<|channel>`` tag and
# starts straight at ``thought\n``. We detect BOTH so a thought span is entered
# no matter which form the model produces.
_GEMMA4_THOUGHT_OPEN_FULL = _GEMMA4_CHANNEL_OPEN + "thought\n"  # "<|channel>thought\n"
_GEMMA4_THOUGHT_OPEN_BARE = "thought\n"
# ChatML / GLM thinking close. The opening ``<think>`` lives in the prompt
# suffix (not the model output), so the stream starts already inside reasoning
# and we only ever see the ``</think>`` close in the generated text.
_CHATML_THINK_OPEN = "<think>"
_CHATML_THINK_CLOSE = "</think>"
# Matches a full thought span, with or without the opening ``<|channel>`` tag
# (the sliding-window variant drops it and starts straight at ``thought\n``).
_GEMMA4_THOUGHT_SPAN_RE = re.compile(
    r"(?:<\|channel>)?thought\n.*?<channel\|>\s*", re.DOTALL
)


def _is_proper_prefix(s: str, marker: str) -> bool:
    """True if ``s`` is a non-empty PROPER prefix of ``marker`` (i.e. the start
    of an incomplete marker, not the whole marker). Used by ``flush`` to drop a
    held partial-marker fragment instead of leaking it onto a channel."""
    return bool(s) and len(s) < len(marker) and marker.startswith(s)


def _partial_marker_tail(buf: str, markers: tuple[str, ...]) -> int:
    """Length of the suffix of ``buf`` that could begin ANY of ``markers``.

    Returns the longest proper-prefix-of-a-marker that is a suffix of ``buf``
    (so a marker split across chunk boundaries is held, never leaked/emitted on
    the wrong channel). 0 if no marker could start in the tail.
    """
    best = 0
    for marker in markers:
        max_len = min(len(buf), len(marker) - 1)
        for n in range(max_len, best, -1):
            if buf.endswith(marker[:n]):
                best = n
                break
    return best


# Channel labels emitted by ``ThinkingRouter``.
CHANNEL_REASONING = "reasoning"
CHANNEL_CONTENT = "content"


class ThinkingRouter:
    """Streaming-safe router that splits model output into reasoning vs content.

    This is the shared engine of the OpenAI-compat streaming path AND the web
    chat SSE path, so both surfaces route thinking identically. It supersedes
    the old ``_Gemma4ThinkingStripper`` (which DROPPED reasoning); instead of
    suppressing the thought channel, it now ROUTES each segment to a channel:

      * ``CHANNEL_REASONING`` — model reasoning (LM-Studio ``reasoning_content``)
      * ``CHANNEL_CONTENT``   — the visible answer

    Two thinking families are handled:

    * gemma4: ``<|channel>thought\\n...reasoning...<channel|>answer`` (and the
      sliding-window / degenerate variants where the opening ``<|channel>`` tag
      is dropped and reasoning starts straight at ``thought\\n``). The stream
      STARTS in content mode and only enters reasoning on a thought opener, so a
      model that answers with NO channel markers passes straight through as
      content. Multi-cycle output (a NEW opener after visible content) re-enters
      reasoning and routes each cycle correctly.
    * chatml / glm: the opening ``<think>`` lives in the prompt suffix, so the
      generated stream STARTS already inside reasoning and we route everything
      up to the ``</think>`` close as reasoning, the remainder as content. (A
      stray opening ``<think>`` in the output, if any, is consumed.)

    Partial markers straddling a chunk boundary are held back in ``_pending``
    and never emitted on the wrong channel.

    ``feed(text)`` -> list[(channel, text)] segments (in order, possibly empty).
    ``flush()`` -> list[(channel, text)] remainder safely emittable at stream
    end. For non-thinking / non-routed callers construct with ``active=False``
    and ``feed`` is a pass-through that yields ``(CHANNEL_CONTENT, text)``.
    """

    def __init__(self, active: bool, model_family: str = "chatml"):
        self.active = active
        self.model_family = model_family
        self._is_gemma4 = model_family == "gemma4"
        # gemma4 starts in CONTENT mode (enter reasoning on an opener); chatml /
        # glm start in REASONING mode (the <think> opener is in the prompt
        # suffix, so generated output begins inside the thought block).
        self._in_reasoning = active and not self._is_gemma4
        self._pending = ""  # held-back possible-partial marker tail
        # FIX 4 (gemma4): the BARE ``thought\n`` opener is only valid at the
        # very START of generation (the sliding-window variant where the
        # ``<|channel>`` tag fell out of the window, so the model emits
        # ``thought\n`` directly as its first token). Once ANY content has been
        # emitted by the router, a literal ``thought\n`` line in content / tool
        # args must NOT be mis-read as a re-opener — only the FULL
        # ``<|channel>thought\n`` opener may re-open mid-content (real
        # multi-cycle). Tracks whether the router has emitted any content yet.
        self._content_seen = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not self.active:
            return [(CHANNEL_CONTENT, text)] if text else []
        if self._is_gemma4:
            return self._feed_gemma4(text)
        return self._feed_chatml(text)

    # --- gemma4: <|channel>thought\n ... <channel|> answer (multi-cycle) ---

    def _feed_gemma4(self, text: str) -> list[tuple[str, str]]:
        buf = self._pending + text
        self._pending = ""
        out: list[tuple[str, str]] = []

        while buf:
            if self._in_reasoning:
                idx = buf.find(_GEMMA4_CHANNEL_CLOSE)
                if idx == -1:
                    # No close yet — emit reasoning, but hold a possible partial
                    # close-marker tail so a split marker never leaks.
                    keep = _partial_marker_tail(buf, (_GEMMA4_CHANNEL_CLOSE,))
                    body = buf[: len(buf) - keep] if keep else buf
                    if body:
                        out.append((CHANNEL_REASONING, body))
                    self._pending = buf[len(buf) - keep:] if keep else ""
                    buf = ""
                    break
                # Reasoning up to the close marker; then leave reasoning mode and
                # reprocess the remainder (it may open a NEW thought cycle).
                if idx:
                    out.append((CHANNEL_REASONING, buf[:idx]))
                self._in_reasoning = False
                buf = buf[idx + len(_GEMMA4_CHANNEL_CLOSE):]
                continue

            # CONTENT mode: scan for a NEW thought opener. The FULL
            # ``<|channel>thought\n`` opener always re-opens (real multi-cycle).
            # The BARE ``thought\n`` opener is recognized ONLY at the very START
            # of generation — FIX 4 — i.e. before ANY content has been emitted
            # (``not _content_seen``) AND at the very first position (``idx 0``,
            # no content preceding it in this buffer). So a literal ``thought\n``
            # line inside content / tool args is NOT mis-routed to reasoning,
            # whether it appears after earlier content or after content in the
            # same buffer.
            at_gen_start = not self._content_seen
            full = buf.find(_GEMMA4_THOUGHT_OPEN_FULL)
            bare = buf.find(_GEMMA4_THOUGHT_OPEN_BARE)
            # The bare opener only counts at absolute generation start (position
            # 0 of the first content); any text before it is content, which would
            # demote it to a post-content re-opener (not allowed for bare).
            if not (at_gen_start and bare == 0):
                bare = -1
            # Across a chunk boundary we hold a partial FULL-opener prefix
            # always; we hold a partial BARE-opener prefix only at generation
            # start (otherwise a trailing partial ``thought\n`` is plain content).
            openers = (
                (_GEMMA4_THOUGHT_OPEN_FULL, _GEMMA4_THOUGHT_OPEN_BARE)
                if at_gen_start
                else (_GEMMA4_THOUGHT_OPEN_FULL,)
            )
            candidates = [i for i in (full, bare) if i != -1]
            if not candidates:
                keep = _partial_marker_tail(buf, openers)
                body = buf[: len(buf) - keep] if keep else buf
                if body:
                    out.append((CHANNEL_CONTENT, body))
                    self._content_seen = True
                self._pending = buf[len(buf) - keep:] if keep else ""
                buf = ""
                break
            idx = min(candidates)
            if idx:
                out.append((CHANNEL_CONTENT, buf[:idx]))
                self._content_seen = True
            self._in_reasoning = True
            if buf.startswith(_GEMMA4_THOUGHT_OPEN_FULL, idx):
                buf = buf[idx + len(_GEMMA4_THOUGHT_OPEN_FULL):]
            else:
                buf = buf[idx + len(_GEMMA4_THOUGHT_OPEN_BARE):]
            continue

        return out

    # --- chatml / glm: stream starts in reasoning, </think> closes it ---

    def _feed_chatml(self, text: str) -> list[tuple[str, str]]:
        buf = self._pending + text
        self._pending = ""
        out: list[tuple[str, str]] = []

        while buf:
            if self._in_reasoning:
                idx = buf.find(_CHATML_THINK_CLOSE)
                if idx == -1:
                    # Hold a possible partial close (and a possible stray opening
                    # <think> prefix) so neither marker leaks into reasoning text.
                    keep = _partial_marker_tail(
                        buf, (_CHATML_THINK_CLOSE, _CHATML_THINK_OPEN)
                    )
                    body = buf[: len(buf) - keep] if keep else buf
                    # Drop a stray full <think> opener that may have been echoed
                    # at the very start of generation.
                    body = body.replace(_CHATML_THINK_OPEN, "")
                    if body:
                        out.append((CHANNEL_REASONING, body))
                    self._pending = buf[len(buf) - keep:] if keep else ""
                    buf = ""
                    break
                head = buf[:idx].replace(_CHATML_THINK_OPEN, "")
                if head:
                    out.append((CHANNEL_REASONING, head))
                self._in_reasoning = False
                buf = buf[idx + len(_CHATML_THINK_CLOSE):]
                continue

            # CONTENT mode (post-</think>): pass through. A second <think> in the
            # answer is unexpected for chatml; we leave it alone (rare/degenerate)
            # except we never re-enter reasoning here.
            out.append((CHANNEL_CONTENT, buf))
            buf = ""

        return out

    def flush(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        pending = self._pending
        self._pending = ""
        if not pending:
            return out
        if self._is_gemma4:
            # In reasoning mode a held tail is reasoning-channel text that never
            # closed (degenerate) — drop it. In content mode it is a partial
            # opener prefix that never completed → real content, emit it.
            if not self._in_reasoning:
                out.append((CHANNEL_CONTENT, pending))
        else:
            # chatml: a held tail is either a partial </think> close (drop — it
            # was reasoning that never closed) or, once in content mode, trailing
            # content. While still in reasoning, treat as reasoning remainder.
            if self._in_reasoning:
                # FIX 2: the held tail may be a partial marker (a proper prefix
                # of </think> or <think>) split across the final chunk. Do NOT
                # leak that fragment onto the reasoning channel — drop it (mirror
                # the gemma4 reasoning-mode flush, which drops a held close-tail).
                if _is_proper_prefix(pending, _CHATML_THINK_CLOSE) or _is_proper_prefix(
                    pending, _CHATML_THINK_OPEN
                ):
                    return out
                channel = CHANNEL_REASONING
            else:
                channel = CHANNEL_CONTENT
            body = pending.replace(_CHATML_THINK_OPEN, "")
            if body:
                out.append((channel, body))
        return out


def _gemma4_extract_content(text: str) -> tuple[Optional[str], str]:
    """Strip ALL Gemma 4 thought channels and return ``(thinking, content)``.

    Robust to degenerate multi-cycle output: removes every
    ``[<|channel>]thought\\n...<channel|>`` span (not just the first), then
    treats any remaining orphan ``<channel|>`` as a reasoning/answer boundary
    (content is everything after the LAST orphan close), and finally drops
    leftover orphan ``<|channel>`` / ``<channel|>`` markers. ``thinking`` is the
    concatenation of the removed reasoning (best-effort, for inspection).
    """
    # 1. Capture + remove all well-formed thought spans.
    thoughts: list[str] = []

    def _grab(m: re.Match) -> str:
        span = m.group(0)
        # Strip the open/close markers + the leading ``thought\n`` for the
        # captured reasoning text.
        inner = span
        if inner.startswith(_GEMMA4_CHANNEL_OPEN):
            inner = inner[len(_GEMMA4_CHANNEL_OPEN):]
        if inner.startswith("thought\n"):
            inner = inner[len("thought\n"):]
        close = inner.rfind(_GEMMA4_CHANNEL_CLOSE)
        if close != -1:
            inner = inner[:close]
        t = inner.strip()
        if t:
            thoughts.append(t)
        return ""

    cleaned = _GEMMA4_THOUGHT_SPAN_RE.sub(_grab, text)

    # 2. Any orphan ``<channel|>`` left (reasoning that never opened a proper
    #    channel, or a final degenerate cycle) — split at the LAST one; the tail
    #    is the real answer, the head is leftover reasoning.
    last_close = cleaned.rfind(_GEMMA4_CHANNEL_CLOSE)
    if last_close != -1:
        head = cleaned[:last_close]
        head_t = head.replace(_GEMMA4_CHANNEL_OPEN, "").replace(
            _GEMMA4_CHANNEL_CLOSE, ""
        ).strip()
        if head_t:
            thoughts.append(head_t)
        cleaned = cleaned[last_close + len(_GEMMA4_CHANNEL_CLOSE):]

    # 3. Drop any leftover orphan open markers (degenerate, no close at all).
    cleaned = cleaned.replace(_GEMMA4_CHANNEL_OPEN, "").replace(
        _GEMMA4_CHANNEL_CLOSE, ""
    )

    thinking = "\n".join(thoughts).strip() or None
    return thinking, cleaned.strip()


def split_thinking_and_content(
    text: str,
    model_family: str = "chatml",
    started_in_thinking: bool = False,
) -> tuple[Optional[str], str]:
    """
    Split thinking from content in model output.

    Supports ChatML (<think>...</think>) and Gemma 4 (<|channel>thought...<channel|>).

    ``started_in_thinking`` (chatml/glm only): set True when generation began
    INSIDE the thought block (thinking enabled — the ``<think>`` opener is in the
    prompt suffix, not the output). It only changes the DEGENERATE no-``</think>``
    case, routing the whole output to reasoning (content="") so the non-streaming
    split matches the streaming ``ThinkingRouter``. Default False preserves the
    legacy "no markers -> content" behavior for non-thinking output.
    """
    if model_family == "gemma4":
        # Strengthened (FIX 3): handle ALL thought spans + orphan markers +
        # degenerate multi-cycle/trailing reasoning, not just the first block.
        if _GEMMA4_CHANNEL_OPEN in text or _GEMMA4_CHANNEL_CLOSE in text or text.startswith("thought\n"):
            return _gemma4_extract_content(text)
        return None, text

    # ChatML: <think>...</think>
    # Case 1: Has <think>...</think> wrapper (full tags in output)
    think_match = re.match(r"<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), think_match.group(2).strip()

    # Case 2: Starts inside thinking block (no opening <think>, just </think>)
    end_idx = text.find("</think>")
    if end_idx != -1:
        thinking = text[:end_idx].strip()
        content = text[end_idx + len("</think>"):].strip()
        return thinking, content

    # Case 3: No </think> close.
    # FIX 1 — align with the streaming ``ThinkingRouter``: for chatml/glm the
    # opening ``<think>`` lives in the PROMPT SUFFIX, so when thinking is active
    # the generated stream begins INSIDE the thought block. If it never emits
    # ``</think>``, the whole output is reasoning that produced no final answer.
    # The streaming router (active=True) routes ALL of it to reasoning; mirror
    # that here so stream == non-stream. We can only know the stream "started in
    # thinking" from the caller, hence ``started_in_thinking``:
    #   * started_in_thinking=True  -> reasoning=full_text, content="" (the
    #     degenerate unclosed case; matches the router). A stray leading
    #     ``<think>`` echoed at generation start is dropped.
    #   * started_in_thinking=False -> legacy behavior: plain answer is content
    #     (thinking disabled, or a non-thinking model — must NOT be misrouted to
    #     reasoning, which would empty the content).
    if started_in_thinking:
        body = text.lstrip()
        if body.startswith(_CHATML_THINK_OPEN):
            body = body[len(_CHATML_THINK_OPEN):]
        return (body.strip() or None), ""
    return None, text


def normalize_content(content) -> str:
    """Normalize message content to a plain string.

    Some clients send content as a list of parts:
      [{"type": "text", "text": "..."}, ...]
    Convert these to a single string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content) if content else ""


def strip_thinking_tags(messages: list[dict], model_family: str = "chatml") -> list[dict]:
    """Strip thinking tags from assistant messages and normalize content.

    Supports ChatML (<think>...</think>) and Gemma 4 (<|channel>thought...<channel|>).
    """
    result = []
    for msg in messages:
        m = msg
        if msg.get("content") and not isinstance(msg["content"], str):
            m = {**msg, "content": normalize_content(msg["content"])}

        if m.get("role") == "assistant" and m.get("content"):
            content = m["content"]
            # Strip Gemma 4 thinking channels — strengthened (FIX 3) to remove
            # ALL <|channel>thought...<channel|> spans + orphan markers +
            # degenerate trailing reasoning (the old single-span re.sub +
            # first-<channel|> split left repeated cycles to replay raw).
            if _GEMMA4_CHANNEL_OPEN in content or _GEMMA4_CHANNEL_CLOSE in content or content.startswith("thought\n"):
                _, cleaned = _gemma4_extract_content(content)
            else:
                cleaned = content
            # Strip ChatML thinking tags
            cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL)
            end_idx = cleaned.find("</think>")
            if end_idx != -1:
                cleaned = cleaned[end_idx + len("</think>"):].lstrip()
            result.append({**m, "content": cleaned})
        else:
            result.append(m)
    return result
