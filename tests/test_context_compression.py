"""抽取式上下文压缩（9.6）：相关句抽取 + 关键约束强制保留 + 定位信息不变。"""

from __future__ import annotations

from app.domain.dto import RetrievedChunk
from app.generation.compression import compress_chunks, compress_text, split_sentences


def test_split_sentences_keeps_punctuation() -> None:
    sentences = split_sentences("第一句。第二句！第三句？\n第四句")
    assert sentences == ["第一句。", "第二句！", "第三句？", "第四句"]


def test_compression_keeps_time_and_negation_constraints() -> None:
    """即使约束句与问题词面不匹配，也不能被压缩掉。"""
    text = (
        "公司差旅制度适用于全体员工。"
        "员工出差前需要提交申请并由直属主管审批。"
        "相关背景说明与流程沿革内容较长，但与本次问题关系不大。"
        "只有 2025 年之后入职的员工才可以申请新款笔记本电脑。"
        "禁止将审批通过的预算用于制度外支出。"
        "这一段介绍制度形成过程中的背景材料和编写说明，篇幅较长但与当前问题没有直接关系。"
        "另外还附有若干历史资料的情况介绍，仅用于内部存档，不影响本问答的判断。"
    )
    compressed = compress_text("差旅报销需要哪些材料", text, max_sentences=3, min_chars=0)
    assert "2025 年之后入职" in compressed
    assert "禁止" in compressed
    assert "相关背景说明" not in compressed
    # 压缩必须真的减少长度，否则应回退原文
    assert len(compressed) < len(text)


def test_short_text_or_compact_text_is_not_compressed() -> None:
    assert compress_text("问题", "很短。", max_sentences=3, min_chars=200) == "很短。"
    compact = "第一句。第二句。第三句。第四句。"
    assert compress_text("问题", compact, max_sentences=8, min_chars=0) == compact


def test_compress_chunks_preserves_locators_and_original_payload() -> None:
    """压缩只改送模型的正文，metadata 定位信息与原始 content 语义不变。"""
    chunk = RetrievedChunk(
        chunk_id="c1",
        content="原始内容" * 10,
        score=0.8,
        doc_id="doc-1",
        source_uri="kb://doc",
        metadata={
            "char_start": 10,
            "char_end": 200,
            "section_path": "第三章/报销标准",
            "parent_content": (
                "本章介绍报销标准。"
                "员工需要在 30 天内提交报销申请。"
                "大段背景描述与流程沿革内容与问题无关。"
                "差旅住宿标准按城市等级执行，上限为 500 元。"
                "另外，制度调整过程涉及多个部门的历史讨论与多轮意见征询，"
                "这些内容仅作为背景记录，不构成执行要求。"
                "本章还记录了若干与报销金额无关的会议纪要与历史沿革说明。"
            ),
        },
    )
    compressed = compress_chunks([chunk], "住宿标准是多少", max_sentences=2, min_chars=0)
    result = compressed[0]
    assert result.metadata["char_start"] == 10
    assert result.metadata["char_end"] == 200
    assert result.metadata["section_path"] == "第三章/报销标准"
    assert result.metadata["compressed"] is True
    assert "500 元" in str(result.metadata["parent_content"])
    assert "背景描述" not in str(result.metadata["parent_content"])
    # 原始 chunk 不被就地修改，避免污染检索缓存/审计记录
    assert chunk.metadata["parent_content"] != result.metadata["parent_content"]
