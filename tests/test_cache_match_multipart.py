"""Cache-match handling for OpenAI multi-part content (images + placeholders)."""

from mlx_soloheaven.engine.mlx_engine import MLXEngine


flatten = MLXEngine._flatten_multipart
normalize = MLXEngine._normalize_for_match


def test_flatten_plain_string():
    assert flatten("hello") == "hello"


def test_flatten_none():
    assert flatten(None) == ""


def test_flatten_text_parts():
    content = [
        {"type": "text", "text": "Screenshot saved to /path/foo.jpg"},
        {"type": "text", "text": "extra note"},
    ]
    assert flatten(content) == "Screenshot saved to /path/foo.jpg\nextra note"


def test_flatten_drops_image_part():
    content = [
        {"type": "text", "text": "Screenshot saved to /path/foo.jpg"},
        {"type": "image", "source": {"type": "base64", "data": "BLOB"}},
    ]
    assert flatten(content) == "Screenshot saved to /path/foo.jpg"


def test_flatten_drops_image_placeholder_text():
    content = [
        {"type": "text", "text": "Screenshot saved to /path/foo.jpg"},
        {"type": "text", "text": "[image data removed - already processed by model]"},
    ]
    assert flatten(content) == "Screenshot saved to /path/foo.jpg"


def test_flatten_drops_image_placeholder_case_insensitive():
    content = [
        {"type": "text", "text": "[Image Data Removed - whatever]"},
        {"type": "text", "text": "ok"},
    ]
    assert flatten(content) == "ok"


def test_image_turn_and_placeholder_turn_normalize_same():
    """The canonical OpenClaw case: first turn has image blob, later turns
    replace the blob with a placeholder string. Both should compare equal."""
    stored = [
        {"type": "text", "text": "Screenshot saved to /path/foo.jpg"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
    ]
    incoming = [
        {"type": "text", "text": "Screenshot saved to /path/foo.jpg"},
        {"type": "text", "text": "[image data removed - already processed by model]"},
    ]
    assert normalize(stored, "user") == normalize(incoming, "user")


def test_normalize_strips_system_reminder_in_multipart():
    content = [
        {"type": "text", "text": "base\n<system-reminder>ignored</system-reminder>"},
    ]
    assert normalize(content, "user") == "base"


def test_normalize_assistant_list_with_toolcall_xml_stripped():
    content = [
        {"type": "text", "text": "Done.\n<tool_call>foo</tool_call>"},
    ]
    assert normalize(content, "assistant") == "Done."


def test_flatten_ignores_unknown_part_types():
    content = [
        {"type": "text", "text": "a"},
        {"type": "video", "url": "https://x"},
        {"type": "audio", "data": "..."},
        {"type": "text", "text": "b"},
    ]
    assert flatten(content) == "a\nb"
