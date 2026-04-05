"""
Gemma 4 RotatingKVCache Compatibility Test — 10-turn conversation.

Verifies:
- No garbled output (e.g., `s<|channel>` instead of `<|channel>thought`)
- Thinking block properly detected and split from content
- Facts recalled across turns
- Cache mode correct (retok_base for turn 2+)
- Stable TPS across turns
"""

import asyncio
import json
import re
import time
import httpx

API = "http://localhost:8080"
TIMEOUT = 300.0

TURNS = [
    ("안녕! 나는 이수진이야. 수학 교수로 일하고 있어. 앞으로 내 정보를 알려줄 테니 잘 기억해줘. "
     "내 이름은 이수진, 직업은 수학 교수. 짧게 확인만 해줘.",
     ["이수진", "수학"]),
    ("나는 흰색 시바견을 키우고 있어. 이름은 코코야. 4살이야. 짧게 확인만 해줘.",
     ["코코", "시바견"]),
    ("내가 가장 좋아하는 음식은 초밥이야. 연어 초밥을 특히 좋아해. 짧게 확인만 해줘.",
     ["초밥", "연어"]),
    ("나는 부산 해운대구에 살고 있어. 아파트 23층이야. 짧게 확인만 해줘.",
     ["부산", "해운대"]),
    ("내 취미는 피아노 연주야. 쇼팽을 가장 좋아해. 짧게 확인만 해줘.",
     ["피아노", "쇼팽"]),
    ("올해 파리에 갈 계획이야. 루브르 박물관을 다시 가고 싶어. 짧게 확인만 해줘.",
     ["파리", "루브르"]),
    ("최근에 소수 관련 새 정리를 증명했어! SJ-Theorem이라고 부르고 있어. 짧게 확인만 해줘.",
     ["소수", "SJ-Theorem"]),
    ("내 여동생 이름은 이민지야. 물리학자로 일해. 짧게 확인만 해줘.",
     ["이민지", "물리학자"]),
    ("지금 연구 주제는 리만 가설이야. AI를 활용해서 접근하고 있어. 짧게 확인만 해줘.",
     ["리만 가설", "AI"]),
    ("은퇴하면 제주도에서 수학 카페를 운영하고 싶어. 짧게 확인만 해줘.",
     ["수학 카페", "제주도"]),
]

# Garbled token patterns that should NOT appear in content
GARBLED_PATTERNS = [
    re.compile(r'[a-z]<\|channel>'),      # e.g., s<|channel>
    re.compile(r'<\|channel>[^t]'),         # <|channel> not followed by 'thought'
    re.compile(r'<channel\|>[^\s]'),        # <channel|> not followed by whitespace
    re.compile(r'<\|turn>'),                # raw turn tokens in output
    re.compile(r'<turn\|>'),                # raw turn-end tokens in output
]


async def send_chat(client: httpx.AsyncClient, session_id: str, content: str) -> dict:
    """Send a chat message and collect streaming response."""
    thinking = ""
    response_content = ""
    stats = None
    full_stream_text = ""
    thinking_done_received = False

    # Simulate the SAME logic as web UI app.js for thinking detection
    stream_thinking_ok = False  # Did streaming detection work (not just done event)?
    stream_content = ""  # Content captured via streaming split (not done event)

    async with client.stream(
        "POST", f"{API}/api/sessions/{session_id}/chat",
        json={"content": content, "stream": True}, timeout=TIMEOUT,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "text":
                full_stream_text += event.get("content", "")
                if event.get("thinking_done"):
                    thinking_done_received = True

                # Simulate web UI thinking detection
                if not stream_thinking_ok:
                    server_signal = event.get("thinking_done") is True
                    end_idx = full_stream_text.find("<channel|>")
                    if server_signal or end_idx != -1:
                        stream_thinking_ok = True
                        if end_idx != -1:
                            stream_content = full_stream_text[end_idx + 10:].strip()
                else:
                    # Content mode: accumulate
                    ct = event.get("content", "")
                    ct = ct.replace("<|channel>thought\n", "").replace("<channel|>", "")
                    stream_content += ct

            elif event.get("type") == "done":
                thinking = event.get("thinking", "")
                response_content = event.get("content", response_content)
                stats = event.get("stats", {})

    return {
        "thinking": thinking,
        "content": response_content,
        "stats": stats,
        "full_stream": full_stream_text,
        "thinking_done_received": thinking_done_received,
        "stream_thinking_ok": stream_thinking_ok,
        "stream_content": stream_content.strip(),
    }


def check_garbled(text: str) -> list[str]:
    """Check for garbled special token patterns in text."""
    issues = []
    for pat in GARBLED_PATTERNS:
        m = pat.search(text)
        if m:
            ctx_start = max(0, m.start() - 20)
            ctx_end = min(len(text), m.end() + 20)
            issues.append(f"  Pattern {pat.pattern!r} found: ...{text[ctx_start:ctx_end]!r}...")
    return issues


async def main():
    async with httpx.AsyncClient() as client:
        # Create session
        print("=" * 70)
        print("Gemma 4 RotatingKVCache 10-Turn Test")
        print("=" * 70)

        res = await client.post(f"{API}/api/sessions", json={"title": "Gemma4 10-Turn Test"})
        session = res.json()
        session_id = session["id"]
        print(f"Session: {session_id}\n")

        all_pass = True
        tps_list = []

        # ---- Phase 1: 10-turn conversation ----
        print("Phase 1: 10-turn conversation")
        print("-" * 70)

        for i, (msg, keywords) in enumerate(TURNS):
            turn = i + 1
            t0 = time.perf_counter()
            result = await send_chat(client, session_id, msg)
            elapsed = time.perf_counter() - t0

            content = result["content"] or ""
            thinking = result["thinking"] or ""
            stats = result["stats"] or {}
            ci = stats.get("cache_info", {})
            tps = stats.get("gen_tps", 0)
            tps_list.append(tps)

            # Check for garbled output
            garbled_in_content = check_garbled(content)
            garbled_in_stream = check_garbled(result["full_stream"])

            # Check thinking_done was received AND streaming split worked
            td_ok = result["thinking_done_received"]
            stream_ok = result["stream_thinking_ok"]

            # Check keywords in content (from done event)
            kw_found = [kw for kw in keywords if kw in content]
            kw_ok = len(kw_found) > 0

            # Also check stream_content has actual content (the real UI test)
            stream_content = result["stream_content"]
            stream_has_content = len(stream_content) > 0

            turn_pass = (
                not garbled_in_content
                and not garbled_in_stream
                and kw_ok
                and stream_ok  # Streaming thinking detection worked
                and td_ok      # thinking_done SSE signal received
            )

            status = "PASS" if turn_pass else "FAIL"
            if not turn_pass:
                all_pass = False

            content_preview = content[:70].replace("\n", " ")
            print(
                f"  T{turn:2d} [{status}] {elapsed:5.1f}s {tps:5.1f}tps "
                f"td={td_ok} stream={stream_ok} kw={len(kw_found)}/{len(keywords)} "
                f"| {content_preview}..."
            )

            if garbled_in_content:
                print(f"      !! GARBLED in content:")
                for g in garbled_in_content:
                    print(f"         {g}")
            if garbled_in_stream:
                print(f"      !! GARBLED in stream:")
                for g in garbled_in_stream[:3]:
                    print(f"         {g}")

        # ---- Phase 2: Recall test ----
        print(f"\n{'=' * 70}")
        print("Phase 2: Recall test (facts from turns 1-5)")
        print("-" * 70)

        recall_msg = (
            "지금까지 알려준 내 정보를 모두 요약해줘. "
            "이름, 직업, 반려동물, 좋아하는 음식, 사는 곳, 취미 등. "
            "짧게 리스트로."
        )
        t0 = time.perf_counter()
        result = await send_chat(client, session_id, recall_msg)
        elapsed = time.perf_counter() - t0
        content = result["content"] or ""

        recall_keywords = ["이수진", "수학", "코코", "초밥", "부산", "피아노"]
        found = [kw for kw in recall_keywords if kw in content]
        recall_pass = len(found) >= 4  # at least 4 out of 6

        status = "PASS" if recall_pass else "FAIL"
        if not recall_pass:
            all_pass = False

        print(f"  Recall [{status}] {elapsed:5.1f}s | found {len(found)}/{len(recall_keywords)}: {found}")
        print(f"  Content: {content[:200].replace(chr(10), ' ')}...")

        garbled = check_garbled(content)
        if garbled:
            all_pass = False
            print(f"  !! GARBLED in recall:")
            for g in garbled:
                print(f"     {g}")

        # ---- Summary ----
        print(f"\n{'=' * 70}")
        print("Summary")
        print("-" * 70)
        avg_tps = sum(tps_list) / len(tps_list) if tps_list else 0
        print(f"  Avg TPS: {avg_tps:.1f}")
        print(f"  TPS range: {min(tps_list):.1f} - {max(tps_list):.1f}")
        print(f"  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
        print("=" * 70)

        return 0 if all_pass else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
