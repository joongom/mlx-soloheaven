"""Unit tests for parse_tool_calls across Qwen / GLM / Gemma 4 formats.

Fixtures use realistic outputs captured from the corresponding chat templates.
Run with: pytest tests/test_tool_parser.py -v
"""
import json

import pytest

from mlx_soloheaven.engine.tool_parser import (
    get_tool_markers,
    parse_tool_calls,
    try_extract_tool_name,
)


# ---------- Qwen / ChatML ----------

QWEN_SINGLE = (
    "I'll search for that.\n"
    "<tool_call><function=web_search>"
    "<parameter=query>apple silicon</parameter>"
    "<parameter=limit>5</parameter>"
    "</function></tool_call>"
)

QWEN_MULTI = (
    "<tool_call><function=read_file>"
    "<parameter=path>/etc/hosts</parameter>"
    "</function></tool_call>"
    "<tool_call><function=read_file>"
    "<parameter=path>/etc/resolv.conf</parameter>"
    "</function></tool_call>"
)


def test_qwen_single():
    text, calls = parse_tool_calls(QWEN_SINGLE, model_family="chatml")
    assert text == "I'll search for that."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"query": "apple silicon", "limit": 5}
    assert calls[0]["id"].startswith("call_")


def test_qwen_multi():
    _, calls = parse_tool_calls(QWEN_MULTI, model_family="chatml")
    assert len(calls) == 2
    assert [json.loads(c["function"]["arguments"])["path"] for c in calls] == [
        "/etc/hosts", "/etc/resolv.conf",
    ]
    # Distinct ids
    assert calls[0]["id"] != calls[1]["id"]


# ---------- GLM ----------

GLM_SINGLE = (
    "I'll look that up.\n"
    "<tool_call>web_search"
    "<arg_key>query</arg_key><arg_value>\"hello world\"</arg_value>"
    "<arg_key>limit</arg_key><arg_value>3</arg_value>"
    "</tool_call>"
)

GLM_NEWLINE_NAME = (
    "<tool_call>\nlist_dir\n"
    "<arg_key>path</arg_key><arg_value>\"/tmp\"</arg_value>"
    "</tool_call>"
)

GLM_MULTI = (
    "<tool_call>read_file"
    "<arg_key>path</arg_key><arg_value>\"a.txt\"</arg_value>"
    "</tool_call>"
    "<tool_call>read_file"
    "<arg_key>path</arg_key><arg_value>\"b.txt\"</arg_value>"
    "</tool_call>"
)


def test_glm_single():
    text, calls = parse_tool_calls(GLM_SINGLE, model_family="glm")
    assert text == "I'll look that up."
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    args = json.loads(calls[0]["function"]["arguments"])
    # GLM template emits strings as raw (no quotes), non-strings as JSON.
    # Our fixture uses JSON-quoted strings which round-trip cleanly.
    assert args == {"query": "hello world", "limit": 3}


def test_glm_newline_name():
    _, calls = parse_tool_calls(GLM_NEWLINE_NAME, model_family="glm")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "list_dir"


def test_glm_multi():
    _, calls = parse_tool_calls(GLM_MULTI, model_family="glm")
    assert len(calls) == 2
    assert all(c["function"]["name"] == "read_file" for c in calls)


def test_glm_falls_through_to_chatml():
    """Some GLM deployments follow Qwen format. The dispatcher falls through
    when <arg_key> is absent."""
    _, calls = parse_tool_calls(QWEN_SINGLE, model_family="glm")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"


# ---------- Gemma 4 ----------

GEMMA4_STRING_ARG = (
    '<|tool_call>call:get_weather{location:<|"|>San Francisco<|"|>}<tool_call|>'
)

GEMMA4_MIXED_ARGS = (
    '<|tool_call>call:book_flight{'
    'origin:<|"|>SFO<|"|>,'
    'dest:<|"|>ICN<|"|>,'
    'passengers:2,'
    'direct:true'
    '}<tool_call|>'
)


def test_gemma4_string_arg():
    _, calls = parse_tool_calls(GEMMA4_STRING_ARG, model_family="gemma4")
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"location": "San Francisco"}


def test_gemma4_mixed_args():
    _, calls = parse_tool_calls(GEMMA4_MIXED_ARGS, model_family="gemma4")
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {
        "origin": "SFO",
        "dest": "ICN",
        "passengers": 2,
        "direct": True,
    }


# ---------- Streaming helpers ----------

def test_markers():
    assert get_tool_markers("chatml") == ("<tool_call>", "</tool_call>")
    assert get_tool_markers("glm") == ("<tool_call>", "</tool_call>")
    assert get_tool_markers("gemma4") == ("<|tool_call>", "<tool_call|>")
    assert get_tool_markers("unknown") == ("<tool_call>", "</tool_call>")


@pytest.mark.parametrize("family,buf,expected", [
    # Qwen: name follows <function=NAME>
    ("chatml", "<function=search>", "search"),
    ("chatml", "<function=search_web>", "search_web"),
    ("chatml", "<function=", None),   # incomplete
    ("chatml", "", None),
    # GLM: name precedes <arg_key>
    ("glm", "web_search<arg_key>q</arg_key>", "web_search"),
    ("glm", "\nlist_dir\n<arg_key>", "list_dir"),
    ("glm", "web_search", None),       # no terminator yet
    # GLM fall-through to Qwen format
    ("glm", "<function=legacy>", "legacy"),
    # Gemma 4: name between call: and {
    ("gemma4", "call:get_weather{", "get_weather"),
    ("gemma4", "call:get_weather", None),   # missing {
])
def test_try_extract_tool_name(family, buf, expected):
    assert try_extract_tool_name(buf, family) == expected


# ---------- No tool_call present ----------

def test_no_tool_call():
    text, calls = parse_tool_calls("Just plain text.", model_family="chatml")
    assert text == "Just plain text."
    assert calls == []


def test_no_tool_call_glm():
    text, calls = parse_tool_calls("Hello world.", model_family="glm")
    assert text == "Hello world."
    assert calls == []


def test_no_tool_call_gemma4():
    text, calls = parse_tool_calls("Hello there.", model_family="gemma4")
    assert text == "Hello there."
    assert calls == []
