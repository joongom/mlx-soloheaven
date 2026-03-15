"""
Context compaction engine.

Summarizes conversation history into a structured checkpoint summary,
following the same format as OpenClaw's compaction system.
"""

import logging
from enum import Enum

logger = logging.getLogger(__name__)

# OpenClaw-compatible summarization prompt
SUMMARIZATION_PROMPT = """Please summarize the conversation above. Create a structured context checkpoint summary that another LLM will use to continue the work.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

# OpenClaw-compatible wrapper for compaction summary messages
COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"


class CompactionStrategy(str, Enum):
    SUMMARIZE = "summarize"


class CompactionEngine:
    def __init__(self, engine):
        self.engine = engine

    async def summarize(
        self,
        messages: list[dict],
        keep_recent: int = 6,
        custom_prompt: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Summarize conversation history, keeping recent messages.

        Args:
            messages: Full message list (including system prompt)
            keep_recent: Number of recent messages to keep as-is
            custom_prompt: Custom summarization prompt (uses default if None)

        Returns:
            {"summary": str, "kept_from": int, "summarized_count": int}
        """
        if len(messages) <= keep_recent + 2:
            return {"error": "Too few messages to compact"}

        # Messages to summarize (exclude system prompt and recent ones)
        has_system = messages[0].get("role") in ("system", "developer") if messages else False
        start_idx = 1 if has_system else 0
        end_idx = len(messages) - keep_recent

        if end_idx <= start_idx:
            return {"error": "Not enough messages to summarize"}

        msgs_to_summarize = messages[start_idx:end_idx]

        # Build summarization request: use original messages + summarization instruction
        # This is cache-friendly — the model already has these messages in KV cache
        prompt = custom_prompt or SUMMARIZATION_PROMPT
        summary_messages = list(messages[:end_idx])
        summary_messages.append({"role": "user", "content": prompt})

        return {
            "messages": summary_messages,
            "kept_from": end_idx,
            "summarized_count": len(msgs_to_summarize),
        }

    async def generate_summary_stream(
        self,
        summary_messages: list[dict],
        session_id: str | None = None,
    ):
        """Async generator that yields text chunks during summary generation."""
        from mlx_soloheaven.engine.tool_parser import split_thinking_and_content

        full_text = ""
        async for chunk in self.engine.generate_stream_async(
            summary_messages,
            max_tokens=4096,
            temperature=0.3,
            thinking_budget=2048,
            session_id=session_id,
        ):
            if chunk.text:
                full_text += chunk.text
                yield {"type": "text", "content": chunk.text}

        # Final: strip thinking, return clean summary
        _, summary_content = split_thinking_and_content(full_text)
        summary_content = (summary_content or full_text).strip()
        yield {"type": "result", "summary": summary_content}

    @staticmethod
    def wrap_summary(summary: str, keep_recent: int = 0) -> str:
        """Wrap summary in OpenClaw-compatible format for LLM consumption."""
        content = COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX
        if keep_recent > 0:
            content += f"\n<!-- keep_recent:{keep_recent} -->"
        return content
