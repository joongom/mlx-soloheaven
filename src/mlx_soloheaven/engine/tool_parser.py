"""Parse model XML tool calls to/from OpenAI JSON format.

Supports Qwen-style XML tool call format:
    <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
"""

import json
import re
import uuid
from typing import Optional


def generate_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def parse_tool_calls(text: str, model_family: str = "chatml") -> tuple[str, list[dict]]:
    """
    Parse tool_call blocks from model output.

    Supports:
    - ChatML/Qwen: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    - Gemma 4: <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>

    Returns:
        (content_text, tool_calls)
        - content_text: text before any tool_call block
        - tool_calls: list of OpenAI-format tool call dicts
    """
    if model_family == "gemma4":
        return _parse_gemma4_tool_calls(text)
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
