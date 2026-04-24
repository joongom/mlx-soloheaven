"""Cache-match tolerance for assistant tool_call turns.

When the stored assistant turn was (optional <think>...</think> + <tool_call>...
</tool_call>), normalize strips both blocks leaving an empty string. OpenAI-format
clients move the tool call to the tool_calls[] field and may replay the thinking
as plain text content (no <think> tags). The KV cache is still valid because it
reflects the tokens the model actually emitted.
"""

from types import SimpleNamespace

from mlx_soloheaven.engine.mlx_engine import MLXEngine


def _match(stored, incoming):
    stub = SimpleNamespace()
    stub._flatten_multipart = MLXEngine._flatten_multipart
    stub._normalize_for_match = MLXEngine._normalize_for_match
    stub._is_compacted_tool = MLXEngine._is_compacted_tool
    return MLXEngine._messages_match(stub, stored, incoming)


def test_toolcall_turn_accepts_reconstructed_thinking_content():
    """Stored: <think>T</think>\\n\\n<tool_call>...</tool_call>
    Incoming: plain thinking text + tool_calls[] field."""
    stored = [
        {"role": "user", "content": "사피엔반도체 주가 어때?"},
        {"role": "assistant", "content": (
            "<think>\nLet me look at the specific article about 사피엔반도체's surge.\n</think>\n\n"
            "<tool_call>\n<function=web_fetch>\n<parameter=url>\nhttps://x.com\n"
            "</parameter>\n</function>\n</tool_call>"
        )},
        {"role": "tool", "tool_call_id": "1", "content": "..."},
    ]
    incoming = [
        {"role": "user", "content": "사피엔반도체 주가 어때?"},
        {
            "role": "assistant",
            "content": "Let me look at the specific article about 사피엔반도체's surge to understand",
            "tool_calls": [{
                "id": "1", "type": "function",
                "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://x.com\"}"},
            }],
        },
        {"role": "tool", "tool_call_id": "1", "content": "..."},
    ]
    assert _match(stored, incoming) is True


def test_toolcall_turn_accepts_empty_content():
    """Some clients drop content entirely when tool_calls is present."""
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nthinking\n</think>\n\n<tool_call>\nfoo\n</tool_call>"},
        {"role": "tool", "tool_call_id": "1", "content": "r"},
    ]
    incoming = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "type": "function",
                            "function": {"name": "foo", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "r"},
    ]
    assert _match(stored, incoming) is True


def test_toolcall_turn_requires_tool_calls_field():
    """Without tool_calls[] in incoming, the tolerance must NOT apply —
    otherwise any empty-normalized stored turn would match anything."""
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nt\n</think>\n\n<tool_call>\nfoo\n</tool_call>"},
        {"role": "user", "content": "next"},
    ]
    incoming = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "completely unrelated reconstructed text"},
        {"role": "user", "content": "next"},
    ]
    assert _match(stored, incoming) is False


def test_toolcall_turn_requires_stored_has_toolcall():
    """Empty s_norm alone shouldn't unlock the tolerance — stored must
    actually have had a tool_call (or tool_calls field)."""
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nt\n</think>"},  # think-only, no tool_call
        {"role": "user", "content": "next"},
    ]
    incoming = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "some text",
            "tool_calls": [{"id": "1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "user", "content": "next"},
    ]
    # But: this is the LAST stored msg, so the prior "last stored assistant"
    # tolerance would accept. Add a trailing msg to disable that path.
    stored.append({"role": "assistant", "content": "ok"})
    incoming.append({"role": "assistant", "content": "ok"})
    assert _match(stored, incoming) is False
