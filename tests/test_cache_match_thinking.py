"""Cache-match tolerance when client reconstructs assistant content as
`{thinking}\\n\\n{final}` without <think> tags."""

from types import SimpleNamespace

from mlx_soloheaven.engine.mlx_engine import MLXEngine


def _engine():
    return SimpleNamespace(_messages_match=MLXEngine._messages_match.__get__(SimpleNamespace(),),)


def _match(stored, incoming):
    # Bind MLXEngine instance methods onto a lightweight stub.
    stub = SimpleNamespace()
    stub._flatten_multipart = MLXEngine._flatten_multipart
    stub._normalize_for_match = MLXEngine._normalize_for_match
    stub._is_compacted_tool = MLXEngine._is_compacted_tool
    return MLXEngine._messages_match(stub, stored, incoming)


def test_assistant_suffix_match_accepts_thinking_prepend():
    """Stored has <think>{T}</think>\\n\\n{F}; client replays as {T}\\n\\n{F}."""
    T = "The user wants me to save something to memory."
    F = "아, 이미지 파일 보내는 방법 기억해둘게요."
    stored = [
        {"role": "user", "content": "기억 저장해"},
        {"role": "assistant", "content": f"<think>\n{T}\n</think>\n\n{F}"},
        {"role": "user", "content": "다음 질문"},
    ]
    incoming = [
        {"role": "user", "content": "기억 저장해"},
        {"role": "assistant", "content": f"{T}\n\n{F}"},
        {"role": "user", "content": "다음 질문"},
    ]
    assert _match(stored, incoming) is True


def test_assistant_exact_match_still_works():
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nthink text\n</think>\n\nhello"},
    ]
    incoming = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nthink text\n</think>\n\nhello"},
    ]
    assert _match(stored, incoming) is True


def test_assistant_suffix_match_rejects_unrelated_content():
    """If the incoming doesn't end with the stored final answer, fail."""
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nt\n</think>\n\nANSWER_A"},
        {"role": "user", "content": "next"},
    ]
    incoming = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ANSWER_B completely different"},
        {"role": "user", "content": "next"},
    ]
    assert _match(stored, incoming) is False


def test_assistant_suffix_match_ignores_trivial_final():
    """Trivial (<8 char) stored answer shouldn't slip through as suffix."""
    stored = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nlong thinking\n</think>\n\nok"},
        {"role": "user", "content": "next"},
    ]
    incoming = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "completely different content that happens to end with ok"},
        {"role": "user", "content": "next"},
    ]
    # Short "ok" final should NOT cause a false positive suffix match
    assert _match(stored, incoming) is False
