"""多轮对话的追问改写（v2.0）。

检索是按"单条 query"设计的：追问里的"它/这个/还有呢"脱离上文无法召回。
本模块用一个轻量改写把追问补全成自包含问题，再交给检索链路；生成阶段仍使用
用户原始问题（回答语气更自然），只有检索 query 被改写。

成本与稳定性：
- 只在"短句 + 指代/省略特征"命中时才调用 LLM，普通问句零额外开销；
- LLM 失败直接返回原问题，不阻塞问答主链路。
"""

from __future__ import annotations

import re

from app.core.errors import AppError
from app.core.logging import get_logger
from app.providers.llm import ChatMessage, LLMClient

logger = get_logger(__name__)

# 指代与省略特征：出现这些词的短句大概率依赖上文
_FOLLOWUP_PATTERN = re.compile(r"它|这个|那个|这些|那些|该|上述|前面|其|还有|呢|那么|然后|其次")
# 超过该长度的问题通常已自包含，不必改写（也避免把长问题改跑偏）
_MAX_FOLLOWUP_LEN = 30
# 只取最近 N 条消息作为上下文，控制 prompt 长度
_MAX_HISTORY_MESSAGES = 4
_MAX_MESSAGE_CHARS = 300

_REWRITE_SYSTEM = (
    "你是检索式问答系统的查询改写器。根据对话历史，把用户的追问改写成一条"
    "自包含、可独立检索的问题：补全省略的主语/对象/条件，保留原问题的核心意图，"
    "不要回答，不要推断历史里没有的事实。只输出改写后的问题，一行，不要解释。"
)


class ConversationQueryRewriter:
    """把追问改写为独立检索问题；无 LLM 或失败时原样返回。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    @staticmethod
    def is_followup(question: str) -> bool:
        text = question.strip()
        return len(text) <= _MAX_FOLLOWUP_LEN and bool(_FOLLOWUP_PATTERN.search(text))

    def rewrite(self, question: str, history: list[tuple[str, str]] | None) -> str:
        """history 为 [(role, content)]，按时间正序；返回检索用 query。"""
        base = question.strip()
        if not base or not history or self._llm is None:
            return base
        if not self.is_followup(base):
            return base
        context = _format_history(history)
        try:
            response = self._llm.chat(
                [
                    ChatMessage(role="system", content=_REWRITE_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=f"对话历史：\n{context}\n\n追问：{base}\n\n改写后的问题：",
                    ),
                ],
                temperature=0.0,
                max_tokens=120,
            )
        except AppError as exc:
            logger.warning("追问改写失败，回退原问题：%s", exc.message)
            return base
        rewritten = response.text.strip().splitlines()[0].strip() if response.text else ""
        # 改写结果异常（过长/空）时保守回退，避免把检索带偏
        if not rewritten or len(rewritten) > 200:
            return base
        if rewritten != base:
            logger.info("多轮追问改写：%s -> %s", base[:40], rewritten[:60])
        return rewritten


def _format_history(history: list[tuple[str, str]]) -> str:
    recent = history[-_MAX_HISTORY_MESSAGES:]
    lines = []
    for role, content in recent:
        label = "用户" if role == "user" else "助手"
        clipped = content[:_MAX_MESSAGE_CHARS]
        lines.append(f"{label}：{clipped}")
    return "\n".join(lines)
