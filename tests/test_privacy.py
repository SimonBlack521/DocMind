"""隐私合规测试：PII 识别/脱敏、入库口径一致性、无 PII 文档不被改写。"""

from __future__ import annotations

from app.domain.privacy import detect_pii, redact_pii
from app.ingestion.loader import DocumentLoader

SAMPLE = (
    "联系人：张三，手机号 13800138000，邮箱 zhangsan@example.com；"
    "身份证号 110101199003078888，工资卡 6222020202020201234。"
)


def test_detect_pii_counts_each_category() -> None:
    """四类 PII 都能被识别，且计数准确。"""
    findings = detect_pii(SAMPLE)
    assert findings == {"email": 1, "id_card": 1, "phone": 1, "bank_card": 1}


def test_redact_pii_replaces_sensitive_values_but_keeps_context() -> None:
    """脱敏只替换敏感值，保留业务上下文（制度文本仍可读、可检索）。"""
    redacted, findings = redact_pii(SAMPLE)
    assert findings == {"email": 1, "id_card": 1, "phone": 1, "bank_card": 1}
    assert "zhangsan@example.com" not in redacted
    assert "13800138000" not in redacted
    assert "110101199003078888" not in redacted
    assert "6222020202020201234" not in redacted
    assert "联系人：张三" in redacted
    assert "[手机号已脱敏]" in redacted


def test_normal_text_is_untouched() -> None:
    """没有 PII 的正文不应发生任何字符变化（避免脱敏误伤制度条款）。"""
    text = "一线城市住宿每晚不超过500元，报销需在出差结束后10个工作日内提交。"
    redacted, findings = redact_pii(text)
    assert redacted == text
    assert findings == {}


def test_loader_redacts_before_hashing_for_idempotent_ingest() -> None:
    """脱敏发生在指纹计算之前：同一份含 PII 的文档重复入库得到相同 content_hash。"""
    loader = DocumentLoader(tenant_id="default", kb_id="kb-hr", redact_pii=True)
    first = loader.load_bytes(None, text=SAMPLE, source_uri="kb://hr/pii.txt", suffix=".txt")
    second = loader.load_bytes(None, text=SAMPLE, source_uri="kb://hr/pii.txt", suffix=".txt")

    assert first.content == second.content
    assert first.content_hash == second.content_hash
    assert first.pii_findings == {"email": 1, "id_card": 1, "phone": 1, "bank_card": 1}
    assert "13800138000" not in first.content


def test_loader_without_redaction_keeps_original_text() -> None:
    """未开启脱敏时保持原文（内网离线评测/排查场景需要原文）。"""
    loader = DocumentLoader(tenant_id="default", kb_id="kb-hr")
    document = loader.load_bytes(None, text=SAMPLE, source_uri="kb://hr/pii.txt", suffix=".txt")
    assert "13800138000" in document.content
    assert document.pii_findings == {}
