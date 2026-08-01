"""Parse model XML tool calls to/from OpenAI JSON format.

Supports Qwen/ChatML, GLM, Gemma 4, and Harmony (OpenAI gpt-oss) tool call
formats.
"""

import json
import re
import uuid
from typing import Optional


def generate_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


# Per-family tool_call block markers. The streaming emitter uses these to
# detect block boundaries; tool-call deltas (id/name, then arguments) are
# deliberately deferred until the closed block parses successfully, so a
# malformed block can fall back to raw-content passthrough without ever
# having emitted a phantom tool_calls delta.
_TOOL_MARKERS = {
    "chatml": ("<tool_call>", "</tool_call>"),
    "qwen":   ("<tool_call>", "</tool_call>"),
    "glm":    ("<tool_call>", "</tool_call>"),
    "gemma4": ("<|tool_call>", "<tool_call|>"),
    # Harmony (gpt-oss): a tool call is a commentary-channel message with a
    # functions recipient — ``<|channel|>commentary to=functions.NAME
    # <|constrain|>json<|message|>{json}<|call|>``. The harmony
    # ThinkingRouter branch re-emits complete tool blocks on the CONTENT
    # channel in exactly this canonical shape (start marker at segment
    # start, ``<|call|>`` terminator), so the streaming tool FSM's
    # start/end scan works unchanged.
    "harmony": ("<|channel|>commentary", "<|call|>"),
}


def get_tool_markers(model_family: str) -> tuple[str, str]:
    return _TOOL_MARKERS.get(model_family, _TOOL_MARKERS["chatml"])


# U11: legal tool/function/parameter name charset. OpenAI function names
# allow [A-Za-z0-9_.-] (MCP-style clients routinely register dotted/dashed
# names like "server.tool-name"); the old ``\w+`` patterns failed to match
# them and the whole block was silently deleted from the output. ``\w``
# is kept (superset of A-Za-z0-9_) and dot/dash added; the XML formats
# delimit names with ``>`` / ``{`` so the widened class stays unambiguous.
_NAME = r"[\w.\-]+"


def try_extract_tool_name(buf_after_start: str, model_family: str) -> Optional[str]:
    """Extract function name from text buffered *after* the start marker.

    Returns the name if determinable, else None (need more text).
    """
    if model_family == "gemma4":
        m = re.match(r"\s*call:(" + _NAME + r")\s*\{", buf_after_start)
        return m.group(1) if m else None
    if model_family == "harmony":
        # After "<|channel|>commentary": " to=functions.NAME ..." (the
        # generated header form; the router canonicalizes blocks to it).
        m = re.match(r"\s*to=functions\.(" + _NAME + r")", buf_after_start)
        return m.group(1) if m else None
    if model_family == "glm":
        # GLM: name is bare text between <tool_call> and first <arg_key>
        # (or </tool_call> for no-args calls). Some GLM variants follow Qwen.
        first_ak = buf_after_start.find("<arg_key>")
        first_fn = buf_after_start.find("<function=")
        first_end = buf_after_start.find("</tool_call>")
        if first_fn >= 0 and (first_ak < 0 or first_fn < first_ak):
            m = re.match(r"\s*<function=(" + _NAME + r")>", buf_after_start)
            return m.group(1) if m else None
        cutoffs = [p for p in (first_ak, first_end) if p >= 0]
        if not cutoffs:
            return None
        name = buf_after_start[:min(cutoffs)].strip().lstrip("\n").strip()
        return name or None
    # Qwen / chatml default
    m = re.match(r"\s*<function=(" + _NAME + r")>", buf_after_start)
    return m.group(1) if m else None


def parse_tool_calls(
    text: str,
    model_family: str = "chatml",
    *,
    rstrip_content: bool = True,
) -> tuple[str, list[dict]]:
    """
    Parse tool_call blocks from model output.

    Supports:
    - ChatML/Qwen: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    - GLM: <tool_call>function_name<arg_key>key</arg_key><arg_value>value</arg_value>...</tool_call>
    - Gemma 4: <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>
    - Harmony: <|channel|>commentary to=functions.name <|constrain|>json<|message|>{json}<|call|>

    Returns:
        (content_text, tool_calls)
        - content_text: the input text with successfully parsed tool_call
          blocks REMOVED. U11 round 2 (per-block fail-closed preservation):
          the text is walked block by block — an unparseable block, an
          unclosed trailing block, and ALL surrounding ordinary text are
          RETAINED in the content in their original order (matching what the
          streaming FSM emits, which handles blocks independently and passes
          garbage through), never silently deleted. When at least one call
          parsed the content is right-stripped (historical batch behavior);
          when nothing parses the raw text is returned byte-identical.
        - tool_calls: list of OpenAI-format tool call dicts

    ``rstrip_content=False`` (codex round 3, finding 3 — SEGMENT MODE):
    callers that parse one CONTENT SEGMENT of a larger text (the stored-
    content tool-XML strip walks segments and reassembles them) must keep
    the segment's trailing whitespace — it is INTERNAL whitespace of the
    reassembled string, and the historical per-call right-strip silently
    glued the segment to the following thinking marker. The top-level
    caller applies ONE right-strip to the final reconstructed string; the
    default True preserves the historical whole-output batch behavior.
    """
    if model_family == "gemma4":
        return _parse_gemma4_tool_calls(text, rstrip_content=rstrip_content)
    if model_family == "harmony":
        return _parse_harmony_tool_calls(text, rstrip_content=rstrip_content)
    if model_family == "glm":
        # GLM format has <arg_key>/<arg_value> pairs; if not found, fall through
        # to Qwen-style parsing (some GLM variants follow Qwen format).
        if "<arg_key>" in text or "<arg_value>" in text:
            return _parse_glm_tool_calls(text, rstrip_content=rstrip_content)
    return _parse_chatml_tool_calls(text, rstrip_content=rstrip_content)


# ChatML/Qwen block span (the FSM's block boundary: start marker to the
# FIRST end marker) and the strict call shape a span must fully match to
# count as a real call.
_CHATML_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_CHATML_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(" + _NAME + r")>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_CHATML_PARAM_RE = re.compile(
    r"<parameter=(" + _NAME + r")>(.*?)</parameter>", re.DOTALL
)


def _parse_chatml_tool_calls(
    text: str, *, rstrip_content: bool = True,
) -> tuple[str, list[dict]]:
    """Parse Qwen/ChatML XML tool call format.

    U11 round 2 (per-block preservation): walk closed
    ``<tool_call>...</tool_call>`` spans left to right; a span that fully
    matches the call shape is consumed into ``tool_calls`` and removed from
    the content; every other span (garbage body, unclosed trailing block)
    and all surrounding ordinary text stay in the content in original order
    — converging with the streaming FSM, which passes unparseable closed
    blocks through as raw content. Pre-fix, a single parseable block caused
    the content to be cut at the FIRST start marker, silently deleting any
    malformed sibling block plus the ordinary text around it.
    """
    if "<tool_call>" not in text:
        return text, []

    tool_calls: list[dict] = []
    content_parts: list[str] = []
    pos = 0
    for block in _CHATML_BLOCK_RE.finditer(text):
        call = _CHATML_CALL_RE.fullmatch(block.group(0))
        if call is None:
            # Unparseable block: NOT consumed — it is retained in content
            # via the text[pos:...] span of the next consumed block (or the
            # final tail append below).
            continue
        func_name = call.group(1)
        params_block = call.group(2)

        arguments = {}
        for pm in _CHATML_PARAM_RE.finditer(params_block):
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
        content_parts.append(text[pos:block.start()])
        pos = block.end()
    content_parts.append(text[pos:])

    # U11 fail-closed: a start marker was present but NOTHING parsed —
    # pass the raw text through byte-identical (no vanished block).
    if not tool_calls:
        return text, []
    content = "".join(content_parts)
    return (content.rstrip() if rstrip_content else content), tool_calls


def _parse_glm_tool_calls(
    text: str, *, rstrip_content: bool = True,
) -> tuple[str, list[dict]]:
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

    U11 round 2 (per-block preservation): blocks are walked left to right;
    a block that yields a function name is consumed and removed from the
    content, an unparseable block (empty name) and all surrounding ordinary
    text are retained in content in original order (see
    _parse_chatml_tool_calls). The deliberate leniency for a MISSING
    trailing ``</tool_call>`` (generation cut mid-stream, the ``\\Z``
    alternate) is unchanged — match-time consumers discount it separately
    (see MLXEngine._content_xml_calls_for_match, round 5).
    """
    if "<tool_call>" not in text:
        return text, []

    tool_calls: list[dict] = []
    content_parts: list[str] = []
    pos = 0

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
            # Unparseable block: retained in content (not consumed).
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
        content_parts.append(text[pos:tc_match.start()])
        pos = tc_match.end()
    content_parts.append(text[pos:])

    # U11 fail-closed: never silently delete an unparseable block (see
    # _parse_chatml_tool_calls).
    if not tool_calls:
        return text, []
    content = "".join(content_parts)
    return (content.rstrip() if rstrip_content else content), tool_calls


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


# Gemma 4 block span (start marker to the FIRST end marker) and the strict
# call shape a span must fully match to count as a real call.
_GEMMA4_BLOCK_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
_GEMMA4_CALL_RE = re.compile(
    r"<\|tool_call>call:(" + _NAME + r")(\{.*?\})<tool_call\|>",
    re.DOTALL,
)


def _parse_gemma4_tool_calls(
    text: str, *, rstrip_content: bool = True,
) -> tuple[str, list[dict]]:
    """Parse Gemma 4 tool call format: <|tool_call>call:name{args}<tool_call|>

    U11 round 2 (per-block preservation): parsed blocks are consumed and
    removed from the content; unparseable/unclosed blocks and all
    surrounding ordinary text are retained in original order (see
    _parse_chatml_tool_calls).
    """
    if "<|tool_call>" not in text:
        return text, []

    tool_calls: list[dict] = []
    content_parts: list[str] = []
    pos = 0
    for block in _GEMMA4_BLOCK_RE.finditer(text):
        call = _GEMMA4_CALL_RE.fullmatch(block.group(0))
        if call is None:
            # Unparseable block: retained in content (not consumed).
            continue
        func_name = call.group(1)
        args_str = call.group(2)
        arguments = _parse_gemma4_args(args_str)

        tool_calls.append({
            "id": generate_call_id(),
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
        content_parts.append(text[pos:block.start()])
        pos = block.end()
    content_parts.append(text[pos:])

    # U11 fail-closed: never silently delete an unparseable block (see
    # _parse_chatml_tool_calls).
    if not tool_calls:
        return text, []
    content = "".join(content_parts)
    return (content.rstrip() if rstrip_content else content), tool_calls


# ---------------------------------------------------------------------------
# Harmony (OpenAI gpt-oss, model_type "gpt_oss") channel markers.
#
# Wire format (derived from the model's chat_template.jinja and verified
# against the real o200k_harmony tokenizer — see tests/test_harmony_family.py):
# every message is ``<|start|>{role}[ to={recipient}]<|channel|>{channel}
# [ to={recipient}][ <|constrain|>{type}]<|message|>{body}{terminator}``.
# The generation prompt ends with ``<|start|>assistant``, so generated
# output BEGINS at a ``<|channel|>`` header. Channels:
#   * analysis   — chain-of-thought; body routes to CHANNEL_REASONING.
#   * final      — the user-visible answer; body routes to CHANNEL_CONTENT.
#   * commentary — with a ``to=functions.*`` recipient: a TOOL CALL (JSON
#     args body, ``<|call|>`` terminator); without one: a user-visible
#     preamble (routes to content).
# Historical terminator is ``<|end|>``; a natural generated stop is
# ``<|return|>`` (final) or ``<|call|>`` (tool call) — both chat EOS ids.
# ``<|start|>assistant`` between generated blocks is STRUCTURAL (dropped).
# ---------------------------------------------------------------------------
_HARMONY_START = "<|start|>"
_HARMONY_CHANNEL = "<|channel|>"
_HARMONY_MESSAGE = "<|message|>"
_HARMONY_END = "<|end|>"
_HARMONY_RETURN = "<|return|>"
_HARMONY_CALL = "<|call|>"
# All markers a harmony stream can split across chunk boundaries.
_HARMONY_MARKERS = (
    _HARMONY_START, _HARMONY_CHANNEL, _HARMONY_MESSAGE,
    _HARMONY_END, _HARMONY_RETURN, _HARMONY_CALL,
)
# Body terminators that are CONSUMED (the block ended); a new block header
# marker (start/channel) also terminates the current body but is REPROCESSED
# as the next block's opener (harmony messages are header-delimited).
_HARMONY_BODY_TERMINATORS = (_HARMONY_END, _HARMONY_RETURN, _HARMONY_CALL)
_HARMONY_HEADER_OPENERS = (_HARMONY_START, _HARMONY_CHANNEL)
# Markers that END an already-open message body. The set DIFFERS by the
# body's channel (codex batch-5 P1-3 STRUCTURAL PRECEDENCE, refined by
# batch-6 FINDING B):
#
#   * A REASONING (analysis) body also ends at a ``<|start|>`` (reprocessed
#     as the next block's opener): a degenerate analysis that dropped its
#     ``<|end|>`` and ran straight into the next message is recovered — the
#     model LEGITIMATELY continues after analysis (to final, or to a
#     commentary tool call), so a ``<|start|>`` there is a genuine boundary.
#
#   * A CONTENT (final / preamble) body ends ONLY at a real terminator
#     (``<|end|>``/``<|return|>``/``<|call|>``) — NOT at a ``<|start|>``.
#     Within one generated response a final is TERMINAL: the model does not
#     open a NEW message after a final has begun. So a ``<|start|>`` appearing
#     INSIDE an open final body is the model QUOTING a frame (e.g. showing the
#     ``<|start|>assistant<|channel|>commentary...<|call|>`` format as an
#     example), and must stay VISIBLE CONTENT — never be sent back to
#     structural processing where the quoted frame would execute as a phantom
#     tool call (FINDING B). Ambiguity resolves toward content: we NEVER
#     execute a call that is not cleanly ``<|call|>``-terminated at a genuine
#     top-level boundary.
#
# A bare ``<|channel|>`` is DELIBERATELY EXCLUDED from BOTH — in valid harmony
# a real channel switch is ALWAYS preceded by ``<|start|>{role}`` (or is the
# gen-start opener), so a bare ``<|channel|>...`` inside an open body is
# QUOTED content, not a switch (P1-3).
_HARMONY_REASONING_BODY_ENDERS = _HARMONY_BODY_TERMINATORS + (_HARMONY_START,)
_HARMONY_CONTENT_BODY_ENDERS = _HARMONY_BODY_TERMINATORS
# The EXACT channel-header sequences the harmony chat template RAISES on when
# present in assistant content / thinking (chat_template.jinja ~264). ONLY
# these trigger strip_thinking_tags' destructive history reduction
# (ground-truth correction): a bare marker SUBSTRING that is not one of these
# real headers is legitimate quoted/incomplete text and must pass through
# unchanged.
_HARMONY_RAISE_SEQUENCES = (
    _HARMONY_CHANNEL + "analysis" + _HARMONY_MESSAGE,
    _HARMONY_CHANNEL + "final" + _HARMONY_MESSAGE,
)

_HARMONY_RECIPIENT_RE = re.compile(r"to=(" + _NAME + r")")


def _harmony_channel_of(header: str) -> Optional[str]:
    """The channel word of a harmony header body (text between
    ``<|channel|>`` and ``<|message|>``): the first whitespace-delimited
    token. None for an empty header."""
    stripped = header.strip()
    if not stripped:
        return None
    return stripped.split()[0]


def _harmony_recipient(*texts: str) -> Optional[str]:
    """First ``to={recipient}`` found across the given header fragments
    (the generated form carries it after the channel word; the template's
    history form carries it between ``<|start|>assistant`` and
    ``<|channel|>``)."""
    for t in texts:
        m = _HARMONY_RECIPIENT_RE.search(t)
        if m:
            return m.group(1)
    return None


# Harmony tool-call block: a commentary-channel header carrying a
# ``to=functions.*`` recipient, a ``<|message|>`` body, and a ``<|call|>``
# terminator — the tool-call eos. FINDINGS A/B (codex batch-6): ONLY a
# ``<|call|>``-terminated block is an EXECUTABLE tool call. A real generated
# tool call ALWAYS ends with ``<|call|>`` (the P1-1 synthesis re-attaches it
# on a natural CALL stop), so an ``<|end|>``/``<|return|>``/``<|start|>``/EOF
# termination means the block is NOT a call — it is raw CONTENT and must never
# be promoted (the old ``<|end|>`` alternate here canonicalized a non-call
# frame into a phantom call, diverging batch from stream). The header may not
# contain another marker (tempered dot).
_HARMONY_BLOCK_RE = re.compile(
    r"<\|channel\|>commentary"
    r"((?:(?!<\|message\|>|<\|channel\|>|<\|start\|>).)*)"
    r"<\|message\|>"
    r"(.*?)"
    r"<\|call\|>",
    re.DOTALL,
)


def _parse_harmony_tool_calls(
    text: str, *, rstrip_content: bool = True,
) -> tuple[str, list[dict]]:
    """Parse harmony (gpt-oss) tool-call blocks:
    ``<|channel|>commentary to=functions.NAME [<|constrain|>json]
    <|message|>{json args}<|call|>``.

    U11 per-block preservation semantics (mirrors _parse_chatml_tool_calls):
    a block is consumed only when it is ``<|call|>``-terminated AND has a
    ``functions.*`` recipient AND its body parses as a JSON object; every
    other block (missing/foreign recipient, non-object/garbage body,
    non-``<|call|>`` termination, unterminated trailing block) and all
    surrounding ordinary text are RETAINED in the content in original order.

    FINDINGS A/B: ONLY a ``<|call|>`` terminator makes a block a tool call.
    A real generated call always ends with ``<|call|>`` (the tool-call eos;
    the P1-1 synthesis re-attaches it on a natural stop), and BOTH the batch
    parse (positional spans over raw text) and the streaming router now emit
    a non-``<|call|>``-terminated functions-commentary block as RAW CONTENT
    with the terminator consumed — so the two sides extract the same calls by
    construction. A block with NO ``<|call|>`` at all (``<|end|>``-closed,
    ``<|start|>``-bounded, or stream truncated mid-call) stays unparsed — raw
    passthrough, like the chatml truncated-block behavior.
    """
    if "<|channel|>commentary" not in text:
        return text, []

    tool_calls: list[dict] = []
    content_parts: list[str] = []
    pos = 0
    for block in _HARMONY_BLOCK_RE.finditer(text):
        header = block.group(1)
        recipient = _harmony_recipient(header)
        if not recipient or not recipient.startswith("functions."):
            # Preamble / foreign-recipient commentary: not a functions tool
            # call — retained in content (not consumed).
            continue
        func_name = recipient[len("functions."):]
        if not func_name:
            continue
        body = block.group(2).strip()
        try:
            arguments = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            # Unparseable body: block retained in content (U11).
            continue
        if not isinstance(arguments, dict):
            # OpenAI tool-call arguments must be a JSON object.
            continue
        tool_calls.append({
            "id": generate_call_id(),
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })
        content_parts.append(text[pos:block.start()])
        pos = block.end()
    content_parts.append(text[pos:])

    # U11 fail-closed: markers present but nothing parsed — byte-identical
    # passthrough (no vanished block).
    if not tool_calls:
        return text, []
    content = "".join(content_parts)
    return (content.rstrip() if rstrip_content else content), tool_calls


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
# Codex round 7, finding 3: FULL-marker-only variant — used when the thinking
# contract is INACTIVE, where the bare ``thought\n`` opener is ambiguous
# (legitimate content may start with those words) and must not be treated as
# a thought span; the model-emitted ``<|channel>`` open stays authoritative.
_GEMMA4_THOUGHT_SPAN_FULL_RE = re.compile(
    r"<\|channel>thought\n.*?<channel\|>\s*", re.DOTALL
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


def thinking_router_active(model_family: str, enable_thinking: bool) -> bool:
    """Codex round 5, finding 3 — the SINGLE routing-activation policy for
    both streaming APIs (OpenAI-compat and the web chat SSE path).

    gemma4 markers are ALWAYS authoritative: the model emits its own
    ``<|channel>thought`` / ``<channel|>`` channel markers regardless of the
    prompt-side thinking contract, so the router stays MARKER-ACTIVE even
    with ``enable_thinking=False``. Pre-fix, a thinking=False request made
    the router a pure passthrough — a tool call the model REHEARSED inside a
    thought span reached the streaming tool FSM and was emitted as a REAL
    call (finish_reason=tool_calls), while the batch parse and the session
    store (content_segments, which already ignores the flag for gemma4 —
    round 4 / U12) excluded it as a rehearsal. chatml/glm keep flag-driven
    activation: their thought block exists only when the prompt suffix opens
    it, so an inactive router is correct there (a literal ``</think>`` in
    non-thinking output is a quote).

    harmony (gpt-oss) is marker-active for the same reason: the model emits
    its own ``<|channel|>`` headers unconditionally (the template has no
    enable_thinking rendering at all), and an inactive passthrough would
    leak the raw channel markers — including the whole analysis block —
    into ``content``. With thinking OFF the analysis body still routes to
    the reasoning channel (never content), mirroring the gemma4 policy.
    """
    return bool(enable_thinking) or model_family in ("gemma4", "harmony")


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

    ``enable_thinking`` (codex round 7, finding 3 — gemma4 only): the
    EFFECTIVE thinking contract of the request. The gemma4 router stays
    marker-ACTIVE regardless of the flag (round 6 / U12: FULL
    ``<|channel>thought`` markers are model-emitted and unambiguous), but the
    AMBIGUOUS BARE ``thought\n`` opener heuristic is only trusted when the
    thinking contract is active — under ``enable_thinking=False`` legitimate
    content that genuinely starts with the words ``thought\n...`` must stay
    content. ``None`` (legacy callers) falls back to ``active``.
    """

    def __init__(
        self,
        active: bool,
        model_family: str = "chatml",
        enable_thinking: Optional[bool] = None,
    ):
        self.active = active
        self.model_family = model_family
        self._is_gemma4 = model_family == "gemma4"
        self._is_harmony = model_family == "harmony"
        # --- harmony channel-FSM state (see _feed_harmony) ---
        # mode: "text"   — plain passthrough scanning for a header opener;
        #       "struct" — consuming ``<|start|>{role}[ to=...]`` up to the
        #                  ``<|channel|>``/``<|message|>`` marker (structural,
        #                  emits nothing; a recipient found here is kept);
        #       "header" — consuming ``<|channel|>...<|message|>``;
        #       "reasoning" / "content" — streaming a message body;
        #       "tool"   — buffering a functions-commentary body (emitted as
        #                  one canonical content-channel block on close).
        self._h_mode = "text"
        self._h_recipient: Optional[str] = None
        self._h_tool_header = ""
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
        # Codex round 7, finding 3: bare-opener recognition additionally
        # requires the thinking contract to be active (full markers stay
        # authoritative regardless — the router itself stays active).
        self._bare_opener_active = active and (
            enable_thinking if enable_thinking is not None else active
        )

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not self.active:
            return [(CHANNEL_CONTENT, text)] if text else []
        if self._is_gemma4:
            return self._feed_gemma4(text)
        if self._is_harmony:
            return self._feed_harmony(text)
        return self._feed_chatml(text)

    # --- harmony: <|channel|>{ch}<|message|>{body}{terminator} blocks -----

    def _feed_harmony(self, text: str) -> list[tuple[str, str]]:
        """Incremental harmony channel FSM.

        Routing: analysis body -> reasoning; final body (and preamble /
        unknown-channel bodies) -> content; a commentary body whose header
        carries a ``to=functions.*`` recipient is BUFFERED and re-emitted on
        close as ONE canonical content segment ``<|channel|>commentary
        {header}<|message|>{body}<|call|>`` for the streaming tool FSM
        (get_tool_markers("harmony")). Structural text — ``<|start|>{role}``
        prefixes and the channel headers themselves — is never emitted, so
        content deltas contain no markers. Markers split across chunk
        boundaries are held in ``_pending`` (never emitted on any channel).

        A REASONING body ends at ``<|end|>``/``<|return|>``/``<|call|>``
        (consumed) or at a ``<|start|>`` (degenerate missing terminator —
        reprocessed as the next block's opener). A CONTENT/final body ends
        ONLY at a real terminator — a ``<|start|>`` inside it is a quoted
        frame kept as content (FINDING B). A functions-commentary block is
        canonicalized to a ``<|call|>``-terminated tool block ONLY when it is
        itself terminated by ``<|call|>`` (FINDING A); a block terminated by
        anything else (header opener, ``<|end|>``, or STREAM END) is emitted
        RAW as content with no fabricated terminator, so an incomplete / quoted
        / non-call frame stays a raw-content passthrough and is never promoted
        (chatml parity, U11) — keeping the positional twin (_harmony_walk) and
        the batch parse in lockstep with the wire.
        """
        buf = self._pending + text
        self._pending = ""
        out: list[tuple[str, str]] = []

        def _find_earliest(s: str, markers: tuple[str, ...]):
            best_idx, best_m = -1, None
            for m in markers:
                i = s.find(m)
                if i != -1 and (best_idx == -1 or i < best_idx):
                    best_idx, best_m = i, m
            return best_idx, best_m

        while buf:
            if self._h_mode == "text":
                idx, marker = _find_earliest(buf, _HARMONY_HEADER_OPENERS)
                if idx == -1:
                    keep = _partial_marker_tail(buf, _HARMONY_HEADER_OPENERS)
                    body = buf[: len(buf) - keep] if keep else buf
                    if body:
                        out.append((CHANNEL_CONTENT, body))
                    self._pending = buf[len(buf) - keep:] if keep else ""
                    buf = ""
                    break
                if idx:
                    out.append((CHANNEL_CONTENT, buf[:idx]))
                if marker == _HARMONY_START:
                    self._h_mode = "struct"
                    self._h_recipient = None
                else:
                    self._h_mode = "header"
                buf = buf[idx + len(marker):]
                continue

            if self._h_mode == "struct":
                # ``<|start|>{role}[ to=...]`` — ends at <|channel|> (normal)
                # or directly at <|message|> (degenerate header-less block).
                idx, marker = _find_earliest(
                    buf, (_HARMONY_CHANNEL, _HARMONY_MESSAGE),
                )
                if idx == -1:
                    self._pending = buf
                    buf = ""
                    break
                self._h_recipient = _harmony_recipient(buf[:idx])
                if marker == _HARMONY_CHANNEL:
                    self._h_mode = "header"
                    buf = buf[idx + len(marker):]
                else:
                    # No channel at all -> default to visible content.
                    self._h_mode = "content"
                    buf = buf[idx + len(marker):]
                continue

            if self._h_mode == "header":
                # ``{channel}[ to=...][ <|constrain|>type]`` up to <|message|>.
                idx = buf.find(_HARMONY_MESSAGE)
                if idx == -1:
                    self._pending = buf
                    buf = ""
                    break
                header = buf[:idx]
                buf = buf[idx + len(_HARMONY_MESSAGE):]
                channel = _harmony_channel_of(header)
                recipient = _harmony_recipient(header) or self._h_recipient
                if channel == "analysis":
                    # Analysis ALWAYS routes to reasoning — even a
                    # ``to=browser.*`` builtin call or a rehearsed
                    # functions call inside analysis stays reasoning (U12).
                    self._h_mode = "reasoning"
                elif channel == "commentary" and recipient and recipient.startswith(
                    "functions."
                ):
                    self._h_mode = "tool"
                    # Canonical re-emission header: keep the raw header
                    # body; inject the recipient after the channel word when
                    # it only appeared in the structural prefix (template's
                    # history form) so the block stays self-describing.
                    if _HARMONY_RECIPIENT_RE.search(header) is None:
                        header = header.replace(
                            "commentary", f"commentary to={recipient}", 1,
                        )
                    self._h_tool_header = header
                elif channel in ("final", "commentary"):
                    # final answer, or a preamble commentary WITHOUT a
                    # functions recipient — both are user-visible content.
                    self._h_mode = "content"
                else:
                    # P2-6: an UNKNOWN / unrecognized channel (a misspelled
                    # 'analysis', a novel channel word, or an empty header)
                    # fails SAFE toward HIDING — route to reasoning so model
                    # reasoning can never leak into visible content through a
                    # channel this FSM does not recognize (the original
                    # confidentiality bug was the fail-OPEN-to-content default).
                    self._h_mode = "reasoning"
                continue

            if self._h_mode == "tool":
                # The open block's body-so-far always rides in ``_pending``
                # (folded into ``buf`` at feed start), so nothing is emitted
                # until the block closes — no partial-tail bookkeeping needed.
                #
                # FINDINGS A/B: a functions-commentary block is an EXECUTABLE
                # tool call ONLY when terminated by ``<|call|>`` (the tool-call
                # eos; P1-1 synthesizes it on a natural stop). ANY other
                # terminator — a ``<|start|>``/``<|channel|>`` header opener, a
                # stray ``<|end|>``/``<|return|>``/``<|message|>``, or flush/EOF
                # — means it is NOT a call: emit the buffered block RAW as
                # content (never fabricate ``<|call|>``), so a quoted /
                # truncated / degenerate frame is never promoted and the
                # positional twin (_harmony_walk) agrees by construction.
                idx, marker = _find_earliest(buf, _HARMONY_MARKERS)
                if idx == -1:
                    self._pending = buf
                    buf = ""
                    break
                body = buf[:idx]
                if marker == _HARMONY_CALL:
                    # Genuine tool call: canonical content-channel block.
                    out.append((
                        CHANNEL_CONTENT,
                        _HARMONY_CHANNEL + self._h_tool_header
                        + _HARMONY_MESSAGE + body + _HARMONY_CALL,
                    ))
                else:
                    # Not ``<|call|>``-terminated: RAW passthrough, NO ``<|call|>``.
                    out.append((
                        CHANNEL_CONTENT,
                        _HARMONY_CHANNEL + self._h_tool_header
                        + _HARMONY_MESSAGE + body,
                    ))
                self._h_tool_header = ""
                self._h_mode = "text"
                if marker in _HARMONY_HEADER_OPENERS:
                    # Header opener: reprocess as the next block's opener.
                    buf = buf[idx:]
                else:
                    # ``<|call|>`` / ``<|end|>`` / ``<|return|>`` / ``<|message|>``:
                    # a consumed terminator.
                    buf = buf[idx + len(marker):]
                continue

            # reasoning / content body streaming. P1-3 STRUCTURAL PRECEDENCE
            # + FINDING B: a bare ``<|channel|>`` mid-body is ALWAYS quoted
            # content (never an ender). A ``<|start|>`` ends a REASONING body
            # (degenerate analysis missing ``<|end|>``) but NOT a CONTENT/final
            # body (there it is a quoted frame — kept content, never promoted).
            if self._h_mode == "reasoning":
                channel = CHANNEL_REASONING
                enders = _HARMONY_REASONING_BODY_ENDERS
            else:
                channel = CHANNEL_CONTENT
                enders = _HARMONY_CONTENT_BODY_ENDERS
            idx, marker = _find_earliest(buf, enders)
            if idx == -1:
                keep = _partial_marker_tail(buf, _HARMONY_MARKERS)
                body = buf[: len(buf) - keep] if keep else buf
                if body:
                    out.append((channel, body))
                self._pending = buf[len(buf) - keep:] if keep else ""
                buf = ""
                break
            if idx:
                out.append((channel, buf[:idx]))
            self._h_mode = "text"
            if marker in _HARMONY_BODY_TERMINATORS:
                buf = buf[idx + len(marker):]
            else:
                # ``<|start|>`` ending a REASONING body (degenerate missing
                # terminator): the marker itself opens the next block —
                # reprocess it in text mode.
                buf = buf[idx:]
            continue

        return out

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
            # same buffer. Codex round 7, finding 3: it is additionally gated
            # on the thinking contract being active — with thinking disabled a
            # position-zero ``thought\n`` is legitimate content, not the
            # sliding-window degenerate opener.
            at_gen_start = self._bare_opener_active and not self._content_seen
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
        if self._is_harmony:
            if self._h_mode == "tool":
                # Stream ended inside a tool block: flush the RAW truncated
                # block (no fabricated terminator) as content — the
                # downstream FSM / batch parser treats an unterminated block
                # as an unparseable raw passthrough (chatml parity, U11).
                out.append((
                    CHANNEL_CONTENT,
                    _HARMONY_CHANNEL + self._h_tool_header
                    + _HARMONY_MESSAGE + pending,
                ))
                self._h_tool_header = ""
            elif self._h_mode in ("struct", "header"):
                # Incomplete structural/header text: never emit (leaking a
                # half-header would put raw markers on the content channel).
                pass
            elif self._h_mode == "reasoning":
                # A held tail here is a partial-marker fragment (the body
                # before it was already emitted) — drop it (gemma4 parity).
                pass
            elif pending:
                # text / content: a marker that never completed is real text.
                out.append((CHANNEL_CONTENT, pending))
            self._h_mode = "text"
            return out
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


def _gemma4_extract_content(
    text: str, *, bare_opener_active: bool = True,
) -> tuple[Optional[str], str]:
    """Strip ALL Gemma 4 thought channels and return ``(thinking, content)``.

    Robust to degenerate multi-cycle output: removes every
    ``[<|channel>]thought\\n...<channel|>`` span (not just the first), then
    treats any remaining orphan ``<channel|>`` as a reasoning/answer boundary
    (content is everything after the LAST orphan close), and finally drops
    leftover orphan ``<|channel>`` / ``<channel|>`` markers. ``thinking`` is the
    concatenation of the removed reasoning (best-effort, for inspection).

    ``bare_opener_active=False`` (codex round 7, finding 3 — thinking
    contract inactive): only FULL ``<|channel>thought\\n`` spans are treated
    as thought spans; a BARE ``thought\\n`` opener is ambiguous under an
    inactive contract (legitimate content may start with those words) and is
    left in the content. The orphan-``<channel|>`` boundary reduction (step
    2) is skipped too — with no thinking contract, a close that never had an
    open span is model noise inside CONTENT the client already received (the
    router passes it through), so hiding everything before it would lose
    real content. Stray orphan markers themselves are still dropped from the
    replayed prompt (step 3), and FULL spans strip regardless (U12).
    """
    span_re = (
        _GEMMA4_THOUGHT_SPAN_RE if bare_opener_active
        else _GEMMA4_THOUGHT_SPAN_FULL_RE
    )
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

    cleaned = span_re.sub(_grab, text)

    # 2. Any orphan ``<channel|>`` left (reasoning that never opened a proper
    #    channel, or a final degenerate cycle) — split at the LAST one; the tail
    #    is the real answer, the head is leftover reasoning. Skipped when the
    #    thinking contract is inactive (codex round 7, finding 3): without a
    #    contract the head is content, not leftover reasoning.
    last_close = (
        cleaned.rfind(_GEMMA4_CHANNEL_CLOSE) if bare_opener_active else -1
    )
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


def _gemma4_router_split(
    text: str, thinking_active: bool = True,
) -> tuple[Optional[str], str]:
    """Codex round 5, finding 3: the gemma4 thinking/content split with
    POSITIONAL-ROUTER semantics — the streaming ``ThinkingRouter`` is the
    authority on what reached each channel, and this is its whole-text twin
    (the same state machine ``content_segments`` simulates, additionally
    collecting the reasoning spans):

    * content  = the byte-exact concatenation of the router's
      content-channel output (== the ``content_segments`` union);
    * thinking = the concatenation of the reasoning-channel output
      (stripped for storage; ``None`` when empty).

    Divergences from the legacy ``_gemma4_extract_content`` regex
    extraction this replaces for the split/persistence path:
    - a BARE ``thought\\n`` opener AFTER content is CONTENT (FIX 4: bare
      openers only count at absolute generation start — the extraction
      matched them anywhere and hid streamed content as 'thinking');
    - an orphan ``<channel|>`` with no open thought span is CONTENT, never
      a reasoning/answer boundary (the extraction discarded everything
      before the LAST orphan close — text the router had already streamed
      to the client as content);
    - orphan ``<|channel>`` markers in content are preserved byte-for-byte
      (the router passes them through; store == wire).

    ``thinking_active`` (codex round 7, finding 3): the effective thinking
    contract the text streamed under. FULL markers stay authoritative
    regardless; the AMBIGUOUS BARE ``thought\\n`` opener is only recognized
    when the contract is active (the router's ``enable_thinking`` gate).
    """
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    pos = 0
    in_reasoning = False
    n = len(text)
    # The BARE opener is only recognized at the absolute start of generation
    # (FIX 4 — before any content has been emitted) AND under an active
    # thinking contract (codex round 7, finding 3).
    if thinking_active and text.startswith(_GEMMA4_THOUGHT_OPEN_BARE):
        in_reasoning = True
        pos = len(_GEMMA4_THOUGHT_OPEN_BARE)
    while pos < n:
        if in_reasoning:
            close = text.find(_GEMMA4_CHANNEL_CLOSE, pos)
            if close == -1:
                # Unclosed reasoning runs to the end (router: reasoning).
                reasoning_parts.append(text[pos:])
                break
            reasoning_parts.append(text[pos:close])
            pos = close + len(_GEMMA4_CHANNEL_CLOSE)
            in_reasoning = False
        else:
            opener = text.find(_GEMMA4_THOUGHT_OPEN_FULL, pos)
            if opener == -1:
                content_parts.append(text[pos:])
                break
            if opener > pos:
                content_parts.append(text[pos:opener])
            pos = opener + len(_GEMMA4_THOUGHT_OPEN_FULL)
            in_reasoning = True
    thinking = "".join(reasoning_parts).strip() or None
    return thinking, "".join(content_parts)


def content_segments(
    text: str, model_family: str = "chatml", thinking_active: bool = True,
) -> list[tuple[int, int]]:
    """U12 round 2 / codex round 3, finding 4: ``(start, end)`` spans of the
    CONTENT channel in raw assistant text, in order.

    This is the POSITIONAL twin of the streaming ``ThinkingRouter`` — the
    router is the AUTHORITY on what reached the client's content channel —
    for callers that must keep or strip RAW text per channel (the engine's
    stored-content tool-XML stripping and the cache-match helpers). gemma4
    explicitly supports MULTIPLE reasoning/content cycles (thought → content
    → thought → content), so reducing content to the suffix after the LAST
    close marker drops legitimate earlier content segments; this returns
    every one.

    ``thinking_active`` is the router-active state the text streamed under:

    - chatml/glm, ``thinking_active=False``: the inactive router is a pure
      pass-through — the WHOLE text is one content span, and a literal
      ``</think>`` in it (a quote) is content, never a boundary (round 3,
      finding 4 repro b).
    - chatml/glm, ``thinking_active=True``: generation starts inside the
      thought block (the opener lives in the prompt suffix); the FIRST
      ``</think>`` ends reasoning and the router never re-enters it —
      everything after is one content span. No close marker → the whole
      text is content (KEPT legacy divergence: the router routes the
      degenerate unclosed stream entirely to reasoning; callers that need
      that exact shape use ``split_thinking_and_content`` with
      ``started_in_thinking=True``, as the engine's batch parse does — the
      shapes that reach THIS helper always carry the close).
    - gemma4: the router's exact policy, simulated positionally — start in
      CONTENT mode scanning for THOUGHT OPENERS ONLY (full
      ``<|channel>thought\\n`` anywhere; the BARE ``thought\\n`` opener only
      at the very start of generation, FIX 4); a ``<channel|>`` close only
      ends an OPEN thought span. An orphan close with no prior thought-open
      is therefore CONTENT (round 3, finding 4 repro a — the old
      _gemma4_extract_content-style orphan-close reduction discarded
      everything before it, text the router had already emitted as
      content). FULL markers deliberately IGNORE the flag: the model emits
      its own channel markers regardless of the prompt-side thinking
      contract, and the U12 rehearsal protection (tool XML inside a thought
      span is never a call) must hold for stored turns of either contract.
      Codex round 7, finding 3: the AMBIGUOUS BARE opener is the exception —
      it is only recognized when ``thinking_active`` is True (with thinking
      disabled, content genuinely starting with the words ``thought\\n``
      stays content, matching the router's ``enable_thinking`` gate).
    """
    if not text:
        return []
    if model_family == "harmony":
        return _harmony_content_segments(text)
    if model_family != "gemma4":
        if not thinking_active:
            return [(0, len(text))]
        idx = text.find(_CHATML_THINK_CLOSE)
        start = idx + len(_CHATML_THINK_CLOSE) if idx != -1 else 0
        return [(start, len(text))] if start < len(text) else []
    # gemma4: positional simulation of the (active) router.
    segments: list[tuple[int, int]] = []
    pos = 0
    in_reasoning = False
    n = len(text)
    # The BARE opener is only recognized at the absolute start of generation
    # (FIX 4 — before any content has been emitted) AND under an active
    # thinking contract (codex round 7, finding 3).
    if thinking_active and text.startswith(_GEMMA4_THOUGHT_OPEN_BARE):
        in_reasoning = True
        pos = len(_GEMMA4_THOUGHT_OPEN_BARE)
    while pos < n:
        if in_reasoning:
            close = text.find(_GEMMA4_CHANNEL_CLOSE, pos)
            if close == -1:
                # Unclosed reasoning runs to the end (router: reasoning).
                pos = n
                break
            pos = close + len(_GEMMA4_CHANNEL_CLOSE)
            in_reasoning = False
        else:
            opener = text.find(_GEMMA4_THOUGHT_OPEN_FULL, pos)
            if opener == -1:
                segments.append((pos, n))
                pos = n
                break
            if opener > pos:
                segments.append((pos, opener))
            pos = opener + len(_GEMMA4_THOUGHT_OPEN_FULL)
            in_reasoning = True
    return [(s, e) for s, e in segments if s < e]


def _harmony_find_earliest(
    text: str, pos: int, markers: tuple[str, ...],
) -> tuple[int, Optional[str]]:
    """Earliest occurrence of any of ``markers`` in ``text[pos:]`` as an
    absolute index (-1 when none)."""
    best_idx, best_m = -1, None
    for m in markers:
        i = text.find(m, pos)
        if i != -1 and (best_idx == -1 or i < best_idx):
            best_idx, best_m = i, m
    return best_idx, best_m


def _harmony_walk(text: str) -> list[tuple[str, int, int]]:
    """Positional twin of the harmony ``ThinkingRouter`` branch: walk the
    raw text with the same channel FSM and return ``(kind, start, end)``
    spans, where kind is ``"reasoning"`` (analysis body), ``"content"``
    (final / preamble / unknown-channel body, marker-less plain text, OR a
    functions-commentary block that is NOT ``<|call|>``-terminated — raw,
    never a call, FINDINGS A/B) or ``"tool"`` (a ``<|call|>``-terminated
    functions-commentary BLOCK — span starts at its ``<|channel|>`` marker and
    includes the ``<|call|>`` terminator; the router emits the same block).
    Structural ``<|start|>...`` prefixes and channel headers yield no span.
    """
    spans: list[tuple[str, int, int]] = []
    n = len(text)
    pos = 0
    mode = "text"
    recipient: Optional[str] = None
    tool_block_start = 0
    while pos < n:
        if mode == "text":
            idx, marker = _harmony_find_earliest(
                text, pos, _HARMONY_HEADER_OPENERS,
            )
            if idx == -1:
                spans.append(("content", pos, n))
                break
            if idx > pos:
                spans.append(("content", pos, idx))
            if marker == _HARMONY_START:
                mode = "struct"
                recipient = None
            else:
                mode = "header"
                tool_block_start = idx  # a tool block's span starts here
            pos = idx + len(marker)
            continue
        if mode == "struct":
            idx, marker = _harmony_find_earliest(
                text, pos, (_HARMONY_CHANNEL, _HARMONY_MESSAGE),
            )
            if idx == -1:
                break  # incomplete structural tail: no span (router drops it)
            recipient = _harmony_recipient(text[pos:idx])
            if marker == _HARMONY_CHANNEL:
                mode = "header"
                tool_block_start = idx
            else:
                mode = "content"
            pos = idx + len(marker)
            continue
        if mode == "header":
            idx = text.find(_HARMONY_MESSAGE, pos)
            if idx == -1:
                break  # incomplete header tail: no span
            header = text[pos:idx]
            channel = _harmony_channel_of(header)
            rcp = _harmony_recipient(header) or recipient
            pos = idx + len(_HARMONY_MESSAGE)
            if channel == "analysis":
                mode = "reasoning"
            elif channel == "commentary" and rcp and rcp.startswith("functions."):
                mode = "tool"
            elif channel in ("final", "commentary"):
                # final / preamble commentary — visible content.
                mode = "content"
            else:
                # P2-6 (batch twin): unknown channel fails SAFE to reasoning.
                mode = "reasoning"
            continue
        if mode == "tool":
            # FINDINGS A/B (batch twin of _feed_harmony's tool mode): a
            # functions-commentary block is a ``"tool"`` span (executable) ONLY
            # when terminated by ``<|call|>``. ANY other termination (header
            # opener, ``<|end|>``/``<|return|>``/``<|message|>``, or EOF) makes
            # it a ``"content"`` span (raw, never promoted) with the terminator
            # dropped exactly as the streaming FSM drops it — so the positional
            # union equals the wire.
            idx, marker = _harmony_find_earliest(text, pos, _HARMONY_MARKERS)
            if idx == -1:
                # Truncated tool block runs to the end — NOT ``<|call|>``-
                # terminated, so it is content (raw), never a tool call.
                spans.append(("content", tool_block_start, n))
                break
            if marker == _HARMONY_CALL:
                end = idx + len(marker)
                spans.append(("tool", tool_block_start, end))
                mode = "text"
                pos = end
            elif marker in _HARMONY_HEADER_OPENERS:
                # Non-``<|call|>`` header opener: raw content, reprocess opener.
                spans.append(("content", tool_block_start, idx))
                mode = "text"
                pos = idx
            else:
                # ``<|end|>``/``<|return|>``/``<|message|>``: raw content,
                # consume the terminator.
                spans.append(("content", tool_block_start, idx))
                mode = "text"
                pos = idx + len(marker)
            continue
        # reasoning / content body. P1-3 STRUCTURAL PRECEDENCE + FINDING B
        # (batch twin): a bare ``<|channel|>`` mid-body is always quoted
        # content; a ``<|start|>`` ends a REASONING body but is quoted content
        # inside a CONTENT/final body (keep batch == stream).
        if mode == "reasoning":
            kind = "reasoning"
            enders = _HARMONY_REASONING_BODY_ENDERS
        else:
            kind = "content"
            enders = _HARMONY_CONTENT_BODY_ENDERS
        idx, marker = _harmony_find_earliest(text, pos, enders)
        if idx == -1:
            spans.append((kind, pos, n))
            break
        if idx > pos:
            spans.append((kind, pos, idx))
        mode = "text"
        pos = idx if marker in _HARMONY_HEADER_OPENERS else idx + len(marker)
    return [s for s in spans if s[1] < s[2]]


def _harmony_content_segments(text: str) -> list[tuple[int, int]]:
    """Content-channel spans of raw harmony assistant text — the positional
    union the router would have emitted as content: final/preamble bodies,
    marker-less plain text, and functions-commentary TOOL BLOCKS (raw span
    including markers; the router emits the same block with its terminator
    canonicalized to ``<|call|>`` — the harmony tool parser accepts both, so
    parse/strip consumers converge). Analysis bodies and all structural
    markers are excluded — a tool call REHEARSED inside analysis is never
    extractable (U12)."""
    return [
        (s, e) for kind, s, e in _harmony_walk(text) if kind != "reasoning"
    ]


def _harmony_router_split(text: str) -> tuple[Optional[str], str]:
    """Whole-text harmony thinking/content split with ROUTER-AUTHORITATIVE
    semantics: literally drive the streaming ``ThinkingRouter`` over the
    full text (feed + flush), so the batch split is byte-equal to the
    concatenation of what the router streams on the wire — the same
    single-authority principle as the gemma4 round-6 positional split, made
    exact by construction. content includes tool-call blocks re-emitted in
    canonical form (the tool FSM / parse_tool_calls consumes them)."""
    router = ThinkingRouter(active=True, model_family="harmony")
    segs = router.feed(text)
    segs.extend(router.flush())
    thinking = "".join(t for c, t in segs if c == CHANNEL_REASONING)
    content = "".join(t for c, t in segs if c == CHANNEL_CONTENT)
    return (thinking.strip() or None), content


def split_thinking_and_content(
    text: str,
    model_family: str = "chatml",
    started_in_thinking: bool = False,
    thinking_active: bool = True,
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

    ``thinking_active`` (gemma4 only — codex round 7, finding 3): the
    effective thinking contract the text was generated under. FULL channel
    markers stay authoritative regardless; only the AMBIGUOUS BARE
    ``thought\\n`` opener recognition is gated on it (see
    ``_gemma4_router_split``). Default True preserves the round-6 behavior
    for callers without a contract in scope.
    """
    if model_family == "gemma4":
        # Codex round 5, finding 3: POSITIONAL-ROUTER semantics — this split
        # feeds persistence (web chat / OpenAI-compat session stores) and the
        # engine batch parse, all of which must agree byte-for-byte with what
        # the streaming ThinkingRouter emitted on the wire. The old regex
        # extraction (_gemma4_extract_content, still used for INPUT-history
        # stripping in strip_thinking_tags) disagreed on degenerate shapes:
        # a bare ``thought\n`` opener after content and an orphan
        # ``<channel|>`` are CONTENT under the router. Marker-free text is
        # (None, text) either way.
        return _gemma4_router_split(text, thinking_active=thinking_active)

    if model_family == "harmony":
        # Router-authoritative split (driven by the streaming router itself,
        # so batch == stream by construction). ``started_in_thinking`` and
        # ``thinking_active`` are intentionally unused: harmony generation
        # always starts at its own ``<|channel|>`` header (never inside a
        # prompt-opened thought block), and the model-emitted channel
        # markers are authoritative regardless of the thinking contract —
        # analysis routes to reasoning, never content (gemma4 full-marker
        # policy; harmony has no ambiguous bare opener to gate).
        return _harmony_router_split(text)

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


def strip_thinking_tags(
    messages: list[dict],
    model_family: str = "chatml",
    thinking_active: bool = True,
    translation_schema: bool = False,
) -> list[dict]:
    """Strip thinking tags from assistant messages and normalize content.

    Supports ChatML (<think>...</think>) and Gemma 4 (<|channel>thought...<channel|>).

    ``thinking_active`` (codex round 7, finding 3): the session/request
    thinking contract of the prompt-side caller. FULL gemma4 channel markers
    are always stripped (model-emitted, unambiguous); the AMBIGUOUS BARE
    ``thought\\n`` opener is only treated as a thinking span when the
    contract is active — with thinking disabled an assistant turn genuinely
    starting with the words ``thought\\n...`` is content and must replay
    into the prompt untouched. Default True preserves round-6 behavior.

    When ``translation_schema`` is set (translategemma-style models), user
    messages carry structured list content
    {type, source_lang_code, target_lang_code, text} that the model's chat
    template consumes verbatim to synthesize the translator prompt — those are
    passed through untouched instead of being flattened by normalize_content.
    """
    result = []
    for msg in messages:
        m = msg
        if (
            translation_schema
            and msg.get("role") == "user"
            and isinstance(msg.get("content"), list)
        ):
            result.append(msg)
            continue
        if msg.get("content") and not isinstance(msg["content"], str):
            m = {**msg, "content": normalize_content(msg["content"])}

        if m.get("role") == "assistant" and m.get("content"):
            content = m["content"]
            # Harmony (gpt-oss): a client replaying raw channel-marked wire
            # text (e.g. the pre-fix chatml-fallback leak, or a stateless
            # client echoing reasoning) must NEVER reach the chat template —
            # it raises on real <|channel|>analysis/final headers in content
            # and would render analysis into history the template's own
            # semantics DROP. Reduce to the router-authoritative content
            # channel, with PARSED functions-commentary tool blocks removed
            # (the structured tool_calls field, when present, is authoritative
            # for the template's tool-call rendering; an unparseable block
            # stays raw per U11).
            #
            # GROUND-TRUTH CORRECTION: trigger the reduction ONLY when the
            # content actually contains the EXACT header sequences the
            # template RAISES on (chat_template.jinja ~264:
            # ``<|channel|>analysis<|message|>`` or
            # ``<|channel|>final<|message|>``) — NOT on any marker substring.
            # The old "any marker present" gate destructively rewrote
            # legitimate content that merely QUOTED or partially typed a
            # marker (e.g. a user pasting ``<|channel|>`` into a question, or
            # an incomplete ``<|call|>`` fragment) even though the template
            # would never raise on it.
            if model_family == "harmony" and any(
                seq in content for seq in _HARMONY_RAISE_SEQUENCES
            ):
                _, cleaned = _harmony_router_split(content)
                if "<|channel|>commentary" in cleaned:
                    cleaned, _ = _parse_harmony_tool_calls(cleaned)
                result.append({**m, "content": cleaned.strip()})
                continue
            # Strip Gemma 4 thinking channels — strengthened (FIX 3) to remove
            # ALL <|channel>thought...<channel|> spans + orphan markers +
            # degenerate trailing reasoning (the old single-span re.sub +
            # first-<channel|> split left repeated cycles to replay raw).
            # Codex round 7, finding 3: the bare-opener TRIGGER (and the
            # extraction's bare-span matching) is gated on the thinking
            # contract; full markers keep triggering regardless.
            if (
                _GEMMA4_CHANNEL_OPEN in content
                or _GEMMA4_CHANNEL_CLOSE in content
                or (thinking_active and content.startswith("thought\n"))
            ):
                _, cleaned = _gemma4_extract_content(
                    content, bare_opener_active=thinking_active,
                )
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
