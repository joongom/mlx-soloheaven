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


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """
    Parse XML tool_call blocks from model output.

    Returns:
        (content_text, tool_calls)
        - content_text: text before any <tool_call> block
        - tool_calls: list of OpenAI-format tool call dicts
    """
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


def split_thinking_and_content(text: str) -> tuple[Optional[str], str]:
    """
    Split thinking from content in model output.

    The model generates text starting INSIDE a <think> block (because the
    prompt template ends with `<think>\\n`). So the generated text looks like:
        "reasoning...\\n</think>\\n\\nactual response"
    NOT:
        "<think>reasoning</think>actual response"

    Handles both cases for robustness.
    """
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


def strip_thinking_tags(messages: list[dict]) -> list[dict]:
    """Strip <think>...</think> from assistant messages and normalize content.

    Some clients (e.g. OpenCode) include thinking tags in conversation
    history. Remove them to prevent thinking tokens from accumulating
    across turns. Also normalizes list-format content to plain strings.
    """
    result = []
    for msg in messages:
        m = msg
        if msg.get("content") and not isinstance(msg["content"], str):
            m = {**msg, "content": normalize_content(msg["content"])}

        if m.get("role") == "assistant" and m.get("content"):
            content = m["content"]
            cleaned = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
            end_idx = cleaned.find("</think>")
            if end_idx != -1:
                cleaned = cleaned[end_idx + len("</think>"):].lstrip()
            result.append({**m, "content": cleaned})
        else:
            result.append(m)
    return result
