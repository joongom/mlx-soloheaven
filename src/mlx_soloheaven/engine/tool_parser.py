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
            # Simple array parsing: split by comma, parse each element
            arr_str = s[i + 1 : j - 1]
            elements = []
            for elem in arr_str.split(","):
                elem = elem.strip()
                if elem.startswith("<|\"" + "|>") and elem.endswith("<|\"" + "|>"):
                    delim = "<|\"" + "|>"
                    elements.append(elem[len(delim):-len(delim)])
                else:
                    elements.append(_parse_gemma4_value(elem))
            result[key] = elements
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


def split_thinking_and_content(text: str, model_family: str = "chatml") -> tuple[Optional[str], str]:
    """
    Split thinking from content in model output.

    Supports ChatML (<think>...</think>) and Gemma 4 (<|channel>thought...<channel|>).
    """
    if model_family == "gemma4":
        # Gemma 4: <|channel>thought\n...<channel|>content
        m = re.match(r"<\|channel>thought\n(.*?)<channel\|>\s*(.*)", text, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # Sliding window fallback: model generates "thought\n..." without <|channel>
        # when <|think|> is out of the sliding window (prompt > 1024 tokens)
        m = re.match(r"thought\n(.*?)<channel\|>\s*(.*)", text, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # Bare <channel|> split (no opening tag at all)
        end_idx = text.find("<channel|>")
        if end_idx != -1:
            thinking = text[:end_idx].strip()
            content = text[end_idx + len("<channel|>"):].strip()
            return thinking, content
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

    # Case 3: No thinking markers
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
            # Strip Gemma 4 thinking channels
            cleaned = re.sub(r"<\|channel>thought\n.*?<channel\|>\s*", "", content, flags=re.DOTALL)
            end_idx = cleaned.find("<channel|>")
            if end_idx != -1:
                cleaned = cleaned[end_idx + len("<channel|>"):].lstrip()
            # Strip ChatML thinking tags
            cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL)
            end_idx = cleaned.find("</think>")
            if end_idx != -1:
                cleaned = cleaned[end_idx + len("</think>"):].lstrip()
            result.append({**m, "content": cleaned})
        else:
            result.append(m)
    return result
