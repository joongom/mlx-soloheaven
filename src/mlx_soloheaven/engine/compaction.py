
import logging
from enum import Enum
from typing import List, Dict

logger = logging.getLogger(__name__)


class CompactionStrategy(str, Enum):
    SUMMARIZE = "summarize"
    SUMMARIZE_RECENT = "summarize_recent"
    MEMORY_EXTRACT = "memory_extract"
    KEY_POINTS = "key_points"

class CompactionEngine:
    def __init__(self, engine):
        self.engine = engine

    async def summarize_history(self, messages: List[Dict]) -> str:
        """대화 기록을 요약하여 반환합니다."""
        if len(messages) <= 5: return "" # 너무 짧으면 요약 생략
        
        # 요약 요청을 위한 프롬프트 구성
        summary_prompt = "다음 대화 내용을 아주 간결하게 한 문단으로 요약해줘:\n\n"
        for m in messages[:-3]: # 최근 3개 메시지는 제외하고 요약
            role = m.get('role', 'user')
            content = m.get('content', '')
            summary_prompt += f"{role}: {content}\n"
        
        # 엔진을 사용하여 요약 생성 (비동기 처리)
        # 실제 구현에서는 별도의 'summary' 모델이나 옵션을 사용할 수 있음
        try:
            result = await self.engine.generate_stream_async(
                [{"role": "user", "content": summary_prompt}],
                max_tokens=1024,
                temperature=0.3
            )
            summary_text = ""
            async for chunk in result:
                if chunk.text: summary_text += chunk.text
            return summary_text.strip()
        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            return "대화 요약 실패 (이전 기록 압축됨)"

    def check_limit(self, total_tokens: int, limit: int) -> bool:
        """토큰 제한 도달 여부를 확인합니다."""
        return total_tokens >= (limit * 0.9) # 90% 도달 시 트리거
