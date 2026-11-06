"""OCR 钩子测试（v2.0）：扫描页兜底、文本页不触发、识别失败优雅降级。

不依赖真实 tesseract：用假 PdfReader 与假 OcrEngine 验证钩子的触发条件与容错。
"""

from __future__ import annotations

import pytest

from app.core.errors import DocumentParseError
from app.ingestion.ocr import OcrEngine, build_ocr_engine
from app.ingestion.parsers import extract_text


class FakeImage:
    def __init__(self, payload: bytes) -> None:
        self.data = payload


class FakePage:
    def __init__(self, text: str, images: list[FakeImage]) -> None:
        self._text = text
        self.images = images

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(self, stream: object) -> None:
        self.is_encrypted = False
        self.pages = [FakePage("", [FakeImage(b"image-bytes")])]


class FakeOcr:
    """记录调用次数的假 OCR 引擎。"""

    def __init__(self, text: str = "扫描识别文本") -> None:
        self.text = text
        self.calls = 0

    def image_to_text(self, image_bytes: bytes, *, language: str) -> str:
        self.calls += 1
        return self.text


def test_build_ocr_engine_disabled_returns_none() -> None:
    assert build_ocr_engine(enabled=False) is None


def test_scanned_pdf_uses_ocr_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)
    engine: OcrEngine = FakeOcr()

    text = extract_text(
        b"%PDF-1.4", ".pdf", source_uri="scan.pdf", ocr_engine=engine, ocr_min_chars=20
    )

    assert text == "## 第 1 页\n扫描识别文本"


def test_text_pdf_does_not_call_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    import pypdf

    class RichReader:
        def __init__(self, stream: object) -> None:
            self.is_encrypted = False
            self.pages = [FakePage("这是足够长的文本内容，不需要 OCR 识别。" * 3, [])]

    monkeypatch.setattr(pypdf, "PdfReader", RichReader)
    engine = FakeOcr()

    text = extract_text(
        b"%PDF-1.4", ".pdf", source_uri="text.pdf", ocr_engine=engine, ocr_min_chars=20
    )

    assert "足够长的文本内容" in text
    assert engine.calls == 0


def test_ocr_failure_keeps_original_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    import pypdf

    class BrokenOcr:
        def image_to_text(self, image_bytes: bytes, *, language: str) -> str:
            raise RuntimeError("tesseract not installed")

    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)

    # OCR 抛错被吞掉，页面仍为空白 → 走原有"未提取到文本"错误，而不是 OCR 异常
    with pytest.raises(DocumentParseError):
        extract_text(
            b"%PDF-1.4",
            ".pdf",
            source_uri="broken.pdf",
            ocr_engine=BrokenOcr(),
            ocr_min_chars=20,
        )
