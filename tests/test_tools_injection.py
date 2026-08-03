"""Tool definitions must reach the model even when the chat template has no
slot for them.

DeepSeek-V4's template renders tool RESULTS but never mentions `tools`, so
`apply_chat_template(..., tools=[...])` dropped them silently: the model never
saw the tool names or the call syntax, invented its own (`<bash>...</bash>`),
the parser found no `<tool_call>` block, and the client re-sent the same
prompt — which reads as the prompt echoing back.
"""
from __future__ import annotations

import types

import pytest

from mlx_soloheaven.engine.mlx_engine import MLXEngine

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "List a directory",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "depth": {"type": "integer"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]

DEEPSEEK_TMPL = (
    "{%- for m in messages -%}{%- if m['role'] == 'system' -%}{{- m['content'] -}}"
    "{%- elif m['role'] == 'user' -%}{{- '<|User|>' + m['content'] -}}{%- endif -%}"
    "{%- endfor -%}"
)
QWEN_TMPL = "{% if tools %}{{ tools | tojson }}{% endif %}{{ messages[0]['content'] }}"


def _engine(template: str) -> MLXEngine:
    eng = MLXEngine.__new__(MLXEngine)          # no model load
    eng.tokenizer = types.SimpleNamespace(chat_template=template)
    return eng


def test_definitions_are_injected_when_the_template_has_no_tools_slot():
    eng = _engine(DEEPSEEK_TMPL)
    out = eng._inject_tools_if_template_lacks_slot(
        [{"role": "system", "content": "You are an agent."},
         {"role": "user", "content": "what is in src?"}], TOOLS)

    sys_text = out[0]["content"]
    assert out[0]["role"] == "system"
    assert sys_text.startswith("You are an agent.")          # original kept
    assert "list_dir(path: string, depth?: integer)" in sys_text
    assert "read_file(path: string)" in sys_text
    assert "— List a directory" in sys_text                 # description carried
    # the syntax the parser actually accepts, not prose
    assert "<tool_call><function=NAME><parameter=KEY>" in sys_text
    assert out[1] == {"role": "user", "content": "what is in src?"}


def test_a_template_that_renders_tools_is_left_alone():
    eng = _engine(QWEN_TMPL)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    assert eng._inject_tools_if_template_lacks_slot(msgs, TOOLS) == msgs


@pytest.mark.parametrize("tools", [None, []])
def test_no_tools_means_no_change(tools):
    eng = _engine(DEEPSEEK_TMPL)
    msgs = [{"role": "user", "content": "u"}]
    assert eng._inject_tools_if_template_lacks_slot(msgs, tools) == msgs


def test_a_system_message_is_created_when_the_conversation_has_none():
    eng = _engine(DEEPSEEK_TMPL)
    out = eng._inject_tools_if_template_lacks_slot(
        [{"role": "user", "content": "u"}], TOOLS)
    assert out[0]["role"] == "system" and "list_dir" in out[0]["content"]
    assert out[1]["role"] == "user"


def test_injection_is_idempotent():
    """Prompt building runs twice per turn (text for hashing, ids for the
    cache); a second pass must not describe the tools twice."""
    eng = _engine(DEEPSEEK_TMPL)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    once = eng._inject_tools_if_template_lacks_slot(msgs, TOOLS)
    twice = eng._inject_tools_if_template_lacks_slot(once, TOOLS)
    assert twice == once
    assert once[0]["content"].count("Available tools:") == 1


def test_the_caller_s_messages_are_not_mutated():
    eng = _engine(DEEPSEEK_TMPL)
    msgs = [{"role": "system", "content": "sys"}]
    eng._inject_tools_if_template_lacks_slot(msgs, TOOLS)
    assert msgs == [{"role": "system", "content": "sys"}]


def test_assistant_tool_calls_are_written_back_into_content():
    """The other half of the same hole: DeepSeek-V4's assistant branch is
    `'<|Assistant|></think>' + (content or '') + eos` — `tool_calls` are never
    rendered. A history containing a call therefore replayed as an EMPTY
    assistant turn followed by a tool result the model never asked for, and it
    re-issued the same call. Measured: the round-trip named 0 of 3 files from
    the tool result and looped straight back into a tool call.
    """
    eng = _engine(DEEPSEEK_TMPL)
    tcs = [{"id": "call_1", "type": "function", "function": {
        "name": "list_dir", "arguments": {"path": "src", "depth": 2}}}]
    out = eng._inject_tools_if_template_lacks_slot(
        [{"role": "user", "content": "what is in src?"},
         {"role": "assistant", "content": "", "tool_calls": tcs},
         {"role": "tool", "content": "engine.py"}], None)

    asst = out[1]
    assert "tool_calls" not in asst, "must not also leave the structured field"
    assert asst["content"] == (
        "<tool_call><function=list_dir><parameter=path>src</parameter>"
        "<parameter=depth>2</parameter></function></tool_call>")


def test_assistant_text_before_a_call_is_kept():
    eng = _engine(DEEPSEEK_TMPL)
    tcs = [{"id": "c", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]
    out = eng._inject_tools_if_template_lacks_slot(
        [{"role": "assistant", "content": "Let me look.", "tool_calls": tcs}], None)
    assert out[0]["content"].startswith("Let me look.\n<tool_call>")
    assert "<parameter=path>a.py</parameter>" in out[0]["content"]


def test_call_writeback_happens_even_when_this_turn_offers_no_tools():
    """A follow-up request may omit `tools` while the HISTORY still contains a
    call; the history must still render."""
    eng = _engine(DEEPSEEK_TMPL)
    tcs = [{"id": "c", "type": "function",
            "function": {"name": "list_dir", "arguments": {"path": "."}}}]
    out = eng._inject_tools_if_template_lacks_slot(
        [{"role": "assistant", "content": "", "tool_calls": tcs}], None)
    assert "<tool_call><function=list_dir>" in out[0]["content"]


def test_a_template_that_renders_tools_keeps_structured_calls():
    eng = _engine(QWEN_TMPL)
    tcs = [{"id": "c", "type": "function",
            "function": {"name": "list_dir", "arguments": {"path": "."}}}]
    msgs = [{"role": "assistant", "content": "", "tool_calls": tcs}]
    assert eng._inject_tools_if_template_lacks_slot(msgs, TOOLS) == msgs
