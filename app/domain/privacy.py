"""隐私与合规：入库前的 PII 识别与脱敏。

为什么在解析阶段脱敏（而不是检索/展示阶段）：
- 原文一旦进了向量库与 ES，就可能在日志、导出、备份里被反复复制，事后清理成本高；
- 入库前统一替换成占位符，content 与 content_hash 都基于脱敏后文本，
  幂等/去重/版本裁决的口径保持一致；
- 原文仍保存在对象存储的原始文件里，受 ACL 与保留期策略保护，需要时可按权限回查。

识别范围（企业文档最常见的四类结构化 PII）：
    邮箱、中国大陆手机号、18 位身份证号、16-19 位银行卡号。
不做姓名/地址等 NER：误杀率太高，容易把正常制度文本（“张经理审批”）替换掉，
这类信息交给下游权限控制与审计，而不是自动改写正文。
"""

from __future__ import annotations

import re

# 顺序很重要：先匹配更"长/更严格"的模式，避免被宽松的手机号/银行卡号规则截断
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID_CARD_RE = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", _EMAIL_RE, "[邮箱已脱敏]"),
    ("id_card", _ID_CARD_RE, "[身份证号已脱敏]"),
    ("phone", _PHONE_RE, "[手机号已脱敏]"),
    ("bank_card", _BANK_CARD_RE, "[银行卡号已脱敏]"),
)


def detect_pii(text: str) -> dict[str, int]:
    """统计各类 PII 出现次数（脱敏前调用，用于质量报告与合规审计）。

    必须复用 redact_pii 的"先替换再匹配"顺序：18 位身份证号同时满足
    "16-19 位连续数字"的银行卡规则，若各类模式独立扫描会被重复计数；
    按顺序替换后，已被占位符覆盖的片段不会再被后续模式命中，
    保证统计口径与实际脱敏结果完全一致。
    """
    return redact_pii(text)[1]


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """返回（脱敏正文, 命中计数）；没有命中时原样返回，不做任何归一化改写。

    指纹计算在 loader 里统一走 compute_content_hash（内部已归一化），
    因此这里只做替换、不改空白，保持正文可读性。
    """
    findings: dict[str, int] = {}
    redacted = text
    for kind, pattern, placeholder in _PATTERNS:
        redacted, matched = pattern.subn(placeholder, redacted)
        if matched:
            findings[kind] = matched
    return redacted, findings
