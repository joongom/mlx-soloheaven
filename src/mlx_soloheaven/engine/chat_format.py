"""Model-specific chat format strategies for suffix token generation.

Different models use different chat template formats:
- ChatML: Qwen, GLM, MiniMax, GPT-OSS (uses <|im_start|>/<|im_end|>)
- Gemma4: Google Gemma 4 (uses <|turn>/<turn|>)
"""

from abc import ABC, abstractmethod


class ChatFormat(ABC):
    """Base class for model-specific chat format token generation."""

    @abstractmethod
    def suffix_user(self, tokenizer, query: str, thinking: bool) -> list[int]:
        """Create suffix tokens for a new user turn + generation prompt."""
        ...

    @abstractmethod
    def suffix_tool_result(self, tokenizer, messages: list[dict], thinking: bool) -> list[int]:
        """Create suffix tokens for tool result messages + generation prompt."""
        ...

    @abstractmethod
    def dummy_suffix(self, thinking: bool) -> str:
        """Return dummy suffix text for base cache extraction (system+dummy_user)."""
        ...


class ChatMLFormat(ChatFormat):
    """Qwen, GLM, MiniMax, GPT-OSS — ChatML format."""

    def suffix_user(self, tokenizer, query: str, thinking: bool) -> list[int]:
        assistant_suffix = "\n<|im_start|>assistant\n<think>\n" if thinking else "\n<|im_start|>assistant\n"
        suffix_text = f"\n<|im_start|>user\n{query}<|im_end|>{assistant_suffix}"
        return tokenizer.encode(suffix_text, add_special_tokens=False)

    def suffix_tool_result(self, tokenizer, messages: list[dict], thinking: bool) -> list[int]:
        parts = []
        for msg in messages:
            # assistant messages are already in the cache (generated output) — skip
            if msg["role"] == "assistant":
                continue
            elif msg["role"] == "tool":
                parts.append(
                    f"\n<|im_start|>user\n<tool_response>\n"
                    f"{msg.get('content', '')}\n</tool_response><|im_end|>"
                )
            elif msg["role"] == "user":
                parts.append(
                    f"\n<|im_start|>user\n{msg.get('content', '')}<|im_end|>"
                )
        assistant_suffix = "\n<|im_start|>assistant\n<think>\n" if thinking else "\n<|im_start|>assistant\n"
        parts.append(assistant_suffix)
        return tokenizer.encode("".join(parts), add_special_tokens=False)

    def dummy_suffix(self, thinking: bool) -> str:
        if thinking:
            return "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return "\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


class Gemma4Format(ChatFormat):
    """Google Gemma 4 — turn-based format.

    Key differences from ChatML:
    - Thinking is activated via <|think|> in system prompt (template handles this)
    - Generation prompt: <|turn>model\n (thinking enabled) or
      <|turn>model\n<|channel>thought\n<channel|> (thinking disabled, 31B)
    - Tool calls: <|tool_call>call:name{key:<|"|>val<|"|>}<tool_call|>
    - Tool responses: <|tool_response>response:name{key:val}<tool_response|>
    """

    def suffix_user(self, tokenizer, query: str, thinking: bool) -> list[int]:
        # <turn|> closes the previous assistant turn (EOS not in cache)
        gen_prompt = "\n<|turn>model\n"
        if not thinking:
            gen_prompt += "<|channel>thought\n<channel|>"
        suffix_text = f"<turn|>\n<|turn>user\n{query}<turn|>{gen_prompt}"
        return tokenizer.encode(suffix_text, add_special_tokens=False)

    def suffix_tool_result(self, tokenizer, messages: list[dict], thinking: bool) -> list[int]:
        # <turn|> closes the previous assistant turn (EOS not in cache)
        parts = ["<turn|>"]
        for msg in messages:
            # assistant messages are already in the cache (generated output) — skip
            if msg["role"] == "assistant":
                continue
            elif msg["role"] == "tool":
                parts.append(
                    f"\n<|turn>user\n<|tool_response>\n"
                    f"response:{msg.get('name', '')}{{{msg.get('content', '')}}}\n"
                    f"<tool_response|><turn|>"
                )
            elif msg["role"] == "user":
                parts.append(f"\n<|turn>user\n{msg.get('content', '')}<turn|>")
        gen_prompt = "\n<|turn>model\n"
        if not thinking:
            gen_prompt += "<|channel>thought\n<channel|>"
        parts.append(gen_prompt)
        return tokenizer.encode("".join(parts), add_special_tokens=False)

    def dummy_suffix(self, thinking: bool) -> str:
        gen_prompt = "\n<|turn>model\n"
        if not thinking:
            gen_prompt += "<|channel>thought\n<channel|>"
        return f"\n<|turn>user\nhi<turn|>{gen_prompt}"


def get_chat_format(model_family: str) -> ChatFormat:
    """Get the appropriate chat format for a model family."""
    if model_family == "gemma4":
        return Gemma4Format()
    return ChatMLFormat()
