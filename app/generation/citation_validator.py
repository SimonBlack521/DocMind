"""引用合法性校验：过滤 LLM 幻觉引用（输出了 [N] 但 N 不在引用范围内）。

LLM 偶尔会编造引用号（比如写 [99] 但只有 6 条引用），或者引用位置附近的文本
根本不在被引用 chunk 里。这两类都属于幻觉：前者编号越界、后者内容不对应。

处理策略：
1. 编号越界 → 直接从 citations 列表移除（或返回 None 引用）；
2. 内容不对应 → 标记但不强制移除（LLM 可能在总结而不是原文引用），
   但如果某个 chunk 被引用的所有位置都不对应、说明它被"凭空加进答案"，删掉。

实现要点：
- 用正则从答案抽所有 [N]，去重、排序；
- 每个 [N] 取它周围若干字符的答案文本作为 query，在对应 chunk 的 snippet 里做子串匹配；
- chunk 的被引用位置全部不匹配 → 删掉该 chunk 的引用，不影响答案其他部分。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.domain.dto import RetrievedChunk
from app.generation.citations import Citation

# [N] 编号格式：中文/英文方括号，数字可选零前缀
_REF_PATTERN = re.compile(r"[\[【](\d+)[\]】]")
# 句子切分：中英文句末标点或换行；用于"逐句核对关键结论是否有证据支持"
_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]|\n|$)")
# 实质结论的最小长度：过短句子（"如下：""综上所述"）不参与覆盖率判定
_MIN_CLAIM_CHARS = 10

ANSWER_VERIFIED = "verified"
ANSWER_UNVERIFIED = "unverified"
# 全部实质结论都找不到证据支持：调用方应拒答或换用保守文案
ANSWER_REJECTED = "rejected"

# 提取 [N] 周围上下文时的前后各 N 字符
_CONTEXT_RADIUS = 40
# 事实一致性只检查引用号附近的小窗口，避免把相邻句子的数字误算到本条引用
_NUMBER_RADIUS = 16
# 数字（含千分位/小数）；金额、比例、天数等事实值都用它比对
_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
# 字符窗口截断后残留的半截引用标记（"2]" 或缺右括号的 "[2"），清洗后再取数字
_REF_FRAGMENT_LEFT = re.compile(r"^[\d,]+[\]】]")
_REF_FRAGMENT_RIGHT = re.compile(r"[\[【]\d*$")
# L1 概括式引用的字符 bigram 覆盖率下限（真实链路实测：无关片段 0.16，概括引用 0.47+）
_MIN_BIGRAM_COVERAGE = 0.4


@dataclass(frozen=True, slots=True)
class CitationValidation:
    """引用校验结果。"""

    total_answer_refs: int
    out_of_range_refs: int
    unverified_citations: int
    cleaned_citations: list[Citation]
    # L2 事实冲突：引用号附近的数字在被引用片段中找不到（如答案 8000、原文 5000）
    fact_conflicts: int = 0
    # 冲突明细（"ref=[2] 数字 8000 未出现在片段"），供日志与离线评测定位数字幻觉
    conflict_details: list[str] = field(default_factory=list)


def extract_refs(answer: str) -> list[int]:
    """从答案里提取所有 [N] 引用号（去重保留顺序）。"""
    seen: set[int] = set()
    ordered: list[int] = []
    for match in _REF_PATTERN.finditer(answer):
        try:
            num = int(match.group(1))
        except ValueError:
            continue
        if num not in seen:
            seen.add(num)
            ordered.append(num)
    return ordered


def _find_ref_context(answer: str, ref: int) -> str:
    """取 [N] 附近的答案文本（移除引用号本身）作为要匹配的 query。"""
    pattern = re.compile(rf"[\[【]{ref}[\]】]")
    match = pattern.search(answer)
    if not match:
        return ""
    start = max(0, match.start() - _CONTEXT_RADIUS)
    end = min(len(answer), match.end() + _CONTEXT_RADIUS)
    context = answer[start:end]
    context = pattern.sub("", context).strip()
    return context


def _normalize(text: str) -> str:
    """压缩空白 + 去掉标点，便于子串匹配。"""
    return "".join(c for c in text.lower() if c.isalnum() or c.isspace())


def _numbers(text: str) -> set[str]:
    """提取规范化数字集合：去千分位、统一小数表示，便于 5,000 与 5000.0 视为相同。"""
    values: set[str] = set()
    for match in _NUMBER_PATTERN.finditer(text):
        raw = match.group().replace(",", "")
        try:
            values.add(f"{float(raw):g}")
        except ValueError:
            continue
    return values


def _raw_ref_window(answer: str, ref: int) -> str:
    """取引用号前后小窗口的原始答案文本（保留引用号，用于判断是否多引用聚合）。"""
    pattern = re.compile(rf"[\[【]{ref}[\]】]")
    match = pattern.search(answer)
    if not match:
        return ""
    start = max(0, match.start() - _NUMBER_RADIUS)
    end = min(len(answer), match.end() + _NUMBER_RADIUS)
    return answer[start:end]


def _ref_number_window(answer: str, ref: int) -> str:
    """取引用号前后小窗口内的答案文本（不含引用号本身），只用于数字事实校验。"""
    window = _REF_PATTERN.sub("", _raw_ref_window(answer, ref))
    # 窗口按字符数截断，边界可能把相邻引用标记切一半（"[2" 或 "2]"），
    # 残留的编号数字会被误当成答案里的事实数字，导致正常引用被判数字冲突
    window = _REF_FRAGMENT_LEFT.sub("", window)
    window = _REF_FRAGMENT_RIGHT.sub("", window)
    return window.strip()


def _facts_consistent(answer_window: str, snippet: str) -> bool:
    """L2 轻量事实校验：答案在引用号附近写出的数字，必须在被引用片段中出现。

    只做确定性检查（数字/千分位/小数归一化），不引入 NLI 模型；片段里没有数字
    或答案窗口没有数字时不做判定，避免误伤摘要式引用。金额、比例、天数等
    最容易出错的事实值都能覆盖。
    """
    answer_numbers = _numbers(answer_window)
    if not answer_numbers:
        return True
    snippet_numbers = _numbers(snippet)
    if not snippet_numbers:
        return True
    return answer_numbers <= snippet_numbers


def _fuzzy_match(query: str, corpus: str) -> bool:
    """query 与 corpus 是否存在足够重叠的语义内容。

    先走严格子串（原文摘录场景），再退化为字符 bigram 覆盖率：LLM 在合法引用时
    常做概括/合并（例如把"第1条规定"省略成"规定"），整句严格子串匹配会把这些
    正常引用全部判成幻觉。覆盖率阈值 0.4 是在真实链路上量化的折中——
    与主题无关片段（实测 0.16）仍会被剔除，概括式引用（实测 0.47~0.54）保留。
    """
    norm_q = _normalize(query)
    norm_c = _normalize(corpus)
    if not norm_q or not norm_c:
        return False
    # query 太短（3 字符内）视为不可信匹配
    if len(norm_q) < 4:
        return True
    if norm_q in norm_c:
        return True
    compact_q = "".join(ch for ch in norm_q if not ch.isspace())
    compact_c = "".join(ch for ch in norm_c if not ch.isspace())
    if len(compact_q) < 2:
        return True
    grams = [compact_q[i : i + 2] for i in range(len(compact_q) - 1)]
    hits = sum(1 for gram in grams if gram in compact_c)
    return hits / len(grams) >= _MIN_BIGRAM_COVERAGE


def clean_answer_refs(answer: str, citations: Iterable[Citation]) -> str:
    """把答案文本中"已失效"的引用标记同步清掉（与清理后的引用列表对齐）。

    validate_citations 只删引用列表、不动答案文本，导致答案里残留 [n]：
    前端按编号回查引用会落空，用户看到"有标注、无来源"的答案。这里把
    不在 cleaned_citations 里的编号从文本中移除（越界编号 + 内容不对应
    被剔除的编号都算失效）；多引用聚合（"[1][2][4]"）中失效项单独摘除。
    注意：只删标注、不改写正文——正文层面的错误数字需要重新生成或 NLI
    级校验，超出本轻量规则的职责范围。
    """
    valid = {citation.index for citation in citations}

    def _drop(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in valid else ""

    return _REF_PATTERN.sub(_drop, answer).rstrip()


def validate_citations(answer: str, citations: Iterable[Citation]) -> CitationValidation:
    """校验答案里的引用号与引用列表的对应关系。

    返回 Validation 结构，包含清理后的 citations（删除编号越界、内容全不对应的）。
    """
    citation_list = list(citations)
    max_index = max((c.index for c in citation_list), default=0)
    refs = extract_refs(answer)

    out_of_range = sum(1 for ref in refs if ref > max_index or ref < 1)
    index_to_citation = {c.index: c for c in citation_list}

    # 每个 chunk 被多少个答案片段引用、其中多少真的能在 snippet 里找到
    verified_per_citation: dict[int, int] = {c.index: 0 for c in citation_list}
    total_per_citation: dict[int, int] = {c.index: 0 for c in citation_list}

    fact_conflicts = 0
    conflict_details: list[str] = []
    for ref in refs:
        if ref not in index_to_citation:
            continue
        ctx = _find_ref_context(answer, ref)
        citation = index_to_citation[ref]
        total_per_citation[ref] = total_per_citation.get(ref, 0) + 1
        text_ok = _fuzzy_match(ctx, citation.snippet)
        # 多引用聚合在句尾时（"…编号为 001、002、004 [1][2][4]"），按单个引用取的
        # 数字窗口里必然混入兄弟引用的数字，逐条比对会把正常引用全部误判为数字冲突。
        # 这类聚合句跳过 L2（数字归属不可判定），L1 仍会拦截主题无关片段。
        window = _ref_number_window(answer, ref)
        window_refs = _REF_PATTERN.findall(_raw_ref_window(answer, ref))
        facts_ok = _facts_consistent(window, citation.snippet) if len(window_refs) <= 1 else True
        if not facts_ok:
            # 数字事实冲突独立计数：即使文本模糊匹配通过，也说明引用不支持该事实
            fact_conflicts += 1
            window_numbers = sorted(_numbers(window))
            snippet_numbers = sorted(_numbers(citation.snippet))
            conflict_details.append(
                f"ref=[{ref}] 答案数字 {window_numbers} 未全部出现在片段数字 {snippet_numbers}"
            )
        if text_ok and facts_ok:
            verified_per_citation[ref] = verified_per_citation.get(ref, 0) + 1

    # 清理：被引用但所有位置都不匹配（或数字事实冲突）的 chunk 删掉
    cleaned: list[Citation] = []
    unverified = 0
    for citation in citation_list:
        total = total_per_citation.get(citation.index, 0)
        verified = verified_per_citation.get(citation.index, 0)
        if total > 0 and verified == 0:
            unverified += 1
            continue
        cleaned.append(citation)

    return CitationValidation(
        total_answer_refs=len(refs),
        out_of_range_refs=out_of_range,
        unverified_citations=unverified,
        cleaned_citations=cleaned,
        fact_conflicts=fact_conflicts,
        conflict_details=conflict_details,
    )


@dataclass(frozen=True, slots=True)
class AnswerVerification:
    """独立的答案验证状态：引用列表非空 ≠ 答案有证据支持。

    字段语义：
    - status: verified / unverified / rejected（全部关键结论无支持）
    - claims_total: 实质结论句数量
    - cited_claims: 带"可识别且通过校验"的 [n] 的结论句数量
    - coverage: cited_claims / claims_total
    - unsupported_examples: 未带可识别引用的结论句（截断），供日志与离线评测
    """

    status: str
    claims_total: int
    cited_claims: int
    coverage: float
    reason: str = ""
    unsupported_examples: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return self.status != ANSWER_VERIFIED

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "claims_total": self.claims_total,
            "cited_claims": self.cited_claims,
            "coverage": round(self.coverage, 3),
            "reason": self.reason,
        }


def _claims(answer: str) -> list[str]:
    """切出实质结论句：过滤过短的引导语，保留长度足够的陈述。"""
    claims: list[str] = []
    for sentence in _SENTENCE_PATTERN.findall(answer):
        stripped = sentence.strip()
        if len(_normalize(stripped)) >= _MIN_CLAIM_CHARS:
            claims.append(stripped)
    return claims


def verify_answer_support(
    answer: str,
    citations: Iterable[Citation],
    evidence_chunks: Sequence[RetrievedChunk] | None = None,
    *,
    min_coverage: float = 0.5,
    fact_conflicts: int = 0,
) -> AnswerVerification:
    """按"实际送入模型的证据"独立校验答案结论是否被支持。

    与 validate_citations 的职责边界：
    - validate_citations 解决"引用列表与正文标注一致"（编号越界、位置不对应）；
    - 本函数解决"正文结论是否真有证据支持"。真实样本暴露过两件事：
      ① 20 条回答中 16 条正文没有任何可识别 [n]，但引用列表平均 7.9 条——
      返回候选资料不等于答案得到同等数量的证据支持；
      ② 数字事实冲突（原文 5000、答案 8000）删除引用后正文仍保留错误结论。
    因此结论句必须带"通过校验的可识别引用"才算有支持；证据覆盖不足或存在
    事实冲突时标记 unverified，全部结论都无支持时标记 rejected，由调用方
    选择重生成、标注未验证或拒答。

    evidence_chunks 传入"实际送入 LLM 的完整片段"时，额外用于关键词支撑判断
    （无引用但只要整句能在证据里找到足够重叠，视为摘要式支持）。
    """
    claims = _claims(answer)
    if not claims:
        # 澄清/拒答类短文本（"资料中未提及"）没有结论需要校验
        return AnswerVerification(
            status=ANSWER_VERIFIED, claims_total=0, cited_claims=0, coverage=1.0
        )

    valid_indices = {citation.index for citation in citations}
    evidence_text = ""
    if evidence_chunks:
        evidence_text = "\n".join(chunk.content for chunk in evidence_chunks)

    cited = 0
    unsupported: list[str] = []
    for claim in claims:
        refs = set(extract_refs(claim))
        if refs & valid_indices:
            cited += 1
            continue
        # 无引用但整句能在证据里找到足够重叠：视为摘要式支持（不计入引用覆盖，
        # 但不算"无支持"，避免把合法的概括句全部判成幻觉）
        if evidence_text and _fuzzy_match(claim, evidence_text):
            cited += 1
            continue
        unsupported.append(claim[:80])

    coverage = cited / len(claims)
    if fact_conflicts > 0:
        reason = f"fact_conflicts={fact_conflicts}"
        status = ANSWER_UNVERIFIED
    elif unsupported and not cited:
        reason = "no_supported_claims"
        status = ANSWER_REJECTED
    elif coverage < min_coverage:
        reason = "citation_coverage_low"
        status = ANSWER_UNVERIFIED
    else:
        reason = ""
        status = ANSWER_VERIFIED
    return AnswerVerification(
        status=status,
        claims_total=len(claims),
        cited_claims=cited,
        coverage=coverage,
        reason=reason,
        unsupported_examples=unsupported[:3],
    )
