"""查询改写：让 LLM 生成 2-3 条同义问法/子问题，用来提升召回。

改写属于“锦上添花”，任何失败都不能影响主链路：异常时直接回退原查询。
"""

from __future__ import annotations

import re

from app.core.errors import AppError
from app.core.logging import get_logger
from app.providers.llm import ChatMessage, LLMClient

logger = get_logger(__name__)

_REWRITE_SYSTEM = (
    "你是企业知识库的检索助手。请把用户问题改写成 2 条不同表述的检索查询，"
    "用于向量检索和关键词检索。要求：\n"
    "1. 保留原问题的全部关键实体、时间、数字和限定条件；\n"
    "2. 一条偏口语化，一条偏书面/术语化；\n"
    "3. 每行一条，只输出查询本身，不要编号、前缀、解释或空行。"
)

# 去掉 LLM 偶尔带出的 “1.” “-” “*” 等列表前缀
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.、)）])\s*")
_MAX_QUERY_LEN = 200


class QueryRewriter:
    """查询改写器：用 LLM 生成同义问法/子问题扩充召回；任何失败都回退原查询。"""

    def __init__(self, llm: LLMClient | None = None, *, max_variants: int = 3) -> None:
        self._llm = llm
        self._max_variants = max(1, max_variants)

    def rewrite(self, query: str) -> list[str]:
        """返回去重后的查询列表，第一条始终是用户原始问题。

        原问题永远参与召回：改写产出再差也不会丢掉原始意图，改写只是加分项。
        """
        base = query.strip()
        if not base:
            return []

        # 无 LLM（测试/降级场景）或只要 1 条时，直接返回原查询
        if self._llm is None or self._max_variants <= 1:
            return [base]

        try:
            # 低温（0.2）+ 限长：改写要稳定、保守地复述问题，不需要创意发挥
            response = self._llm.chat(
                [
                    ChatMessage(role="system", content=_REWRITE_SYSTEM),
                    ChatMessage(role="user", content=base),
                ],
                temperature=0.2,
                max_tokens=200,
            )
        except AppError as exc:
            logger.warning("查询改写失败，回退原查询：%s", exc.message)
            return [base]

        # 逐行解析 LLM 输出：去列表前缀 → 限长 → 去重 → 最多 max_variants 条
        variants: list[str] = [base]
        for line in response.text.splitlines():
            candidate = _LIST_PREFIX.sub("", line).strip()
            if not candidate or len(candidate) > _MAX_QUERY_LEN:
                continue
            if candidate in variants:
                continue
            variants.append(candidate)
            if len(variants) >= self._max_variants:
                break
        return variants
