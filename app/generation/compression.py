"""抽取式上下文压缩（对应 9.6）：只留下与问题相关、且不含关键约束的句子。

与离线切块的区别：
- 切块是入库时一次性完成的，块粒度固定；
- 上下文压缩是问答时按当前 query 动态抽取，同一个块对不同问题产出的片段不同。

安全约束（企业 RAG 的红线）：
- 条件、否定、时间、版本、数值、单位、权限等限定信息绝对不能丢，
  即使它与 query 词面不匹配也要保留，否则会把"仅 2025 年后入职可申请"
  压缩成"员工可申请"，造成事实性错误；
- 只做抽取（截取原句），不生成新文字、不改写语序，风险最低；
- 定位信息（char_start/char_end/section_path 等 metadata）原样保留，
  压缩只改变送入 LLM 的正文，引用仍可回溯到原文位置。
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import replace

from app.core.text import token_set as _token_set
from app.domain.dto import RetrievedChunk

# 句子切分：中文句末标点、分号、换行都视为边界（保留标点在句内）
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")

# 关键约束信号：命中即强制保留（不看与 query 的词面相关性）
_CONSTRAINT_PATTERNS = (
    re.compile(r"(?:19|20)\d{2}\s*年|工作日|自然日|截[止至]|之前|之后|以前|以后|期限|有效期"),
    re.compile(r"不得|不能|不可|禁止|严禁|除外|除非|仅限|只有|无需|不需要|不适用|须|必须|应当"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|元|万元|天|日|小时|分钟|个|次|年|月|人|份|折)"),
    re.compile(r"权限|授权|审批|申请|角色|管理员|可见|访问|密级|安全级别"),
    re.compile(r"版本|修订|第[一二三四五六七八九十\d]+版|v\d+(?:\.\d+)*"),
    re.compile(r"如果|假如|一旦|若(?!干)|前提|条件|满足|例外"),
)


# 数字 token（Evidence Coverage 校验用）：压缩后原文数字丢失过多说明证据被误伤
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 指代词：选中句以此类词开头时把前一句一并保留，避免丢失指代对象（清单第 16 项）
_ANAPHORA_PREFIXES = ("这", "该", "其", "上述", "此", "本", "以上", "该等")


def split_sentences(text: str) -> list[str]:
    """按中文标点/换行切句；空白句丢弃，句内空白保留。"""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]
    return parts


def _is_constraint(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _CONSTRAINT_PATTERNS)


def _relevance(sentence: str, query_tokens: set[str]) -> float:
    """句子对 query 的词面覆盖率，并用句长轻微惩罚超长铺垫句。"""
    if not query_tokens:
        return 0.0
    tokens = _token_set(sentence)
    overlap = len(tokens & query_tokens)
    coverage = overlap / len(query_tokens)
    length_penalty = 1.0 + math.log1p(len(sentence)) * 0.02
    return coverage / length_penalty


def compress_text(
    question: str,
    text: str,
    *,
    max_sentences: int = 8,
    min_chars: int = 200,
) -> str:
    """从长文本中抽取与问题最相关、且保留全部约束的句子。

    - 文本不超过 min_chars 时原样返回（压缩收益低，避免无谓改写）；
    - 约束句强制保留；非约束句按相关性取前若干句；
    - 结果不足原文 60% 才采用压缩，否则回退原文（说明这段本来就很紧凑）。
    """
    stripped = text.strip()
    if not stripped or len(stripped) <= min_chars or max_sentences <= 0:
        return text

    sentences = split_sentences(stripped)
    if len(sentences) <= 2:
        return text

    query_tokens = _token_set(question)
    selected: set[int] = set()
    for index, sentence in enumerate(sentences):
        # 首句通常是主题句，与 query 不匹配也保留，避免丢失上下文指代
        if index == 0 or _is_constraint(sentence):
            selected.add(index)

    scored = sorted(
        range(len(sentences)),
        key=lambda index: (_relevance(sentences[index], query_tokens), -index),
        reverse=True,
    )
    for index in scored:
        if len(selected) >= max_sentences:
            break
        selected.add(index)

    # 前后文保留（清单第 16 项）：指代词开头的选中句把前一句一并带上，
    # 避免"这/该/上述"丢失指代对象（依赖关系的最小规则化近似）
    for index in sorted(selected):
        if (
            index > 0
            and index - 1 not in selected
            and sentences[index].startswith(_ANAPHORA_PREFIXES)
        ):
            selected.add(index - 1)

    compressed = "".join(sentences[index] for index in sorted(selected))
    if len(compressed) >= len(stripped) * 0.6:
        return text

    # Evidence Coverage 校验（清单第 16 项）：原文数字 token 若过半在压缩后
    # 丢失，说明关键证据被误伤，放弃压缩回退原文
    source_numbers = set(_NUMBER_RE.findall(stripped))
    if source_numbers:
        kept = sum(1 for number in source_numbers if number in compressed)
        if kept / len(source_numbers) < 0.5:
            return text
    return compressed


def compress_chunks(
    chunks: Sequence[RetrievedChunk],
    question: str,
    *,
    max_sentences: int = 8,
    min_chars: int = 200,
) -> list[RetrievedChunk]:
    """对每个候选片段做抽取式压缩，返回新的 chunk 列表（不改动输入对象）。

    压缩后的正文写回 `content` 与 `metadata["parent_content"]`（若存在）：
    提示词构建优先读 parent_content，两者都要压缩才真正减少送入 LLM 的字符。
    原始定位 metadata 不变，引用仍可回到原文位置。
    """
    compressed_items: list[RetrievedChunk] = []
    for chunk in chunks:
        source = str(chunk.metadata.get("parent_content") or chunk.content)
        compressed = compress_text(
            question, source, max_sentences=max_sentences, min_chars=min_chars
        )
        if compressed == source:
            compressed_items.append(chunk)
            continue

        metadata = dict(chunk.metadata)
        if metadata.get("parent_content"):
            metadata["parent_content"] = compressed
        metadata["compressed"] = True
        metadata["original_chars"] = len(source)
        compressed_items.append(
            replace(chunk, content=compressed, metadata=metadata)
        )
    return compressed_items
