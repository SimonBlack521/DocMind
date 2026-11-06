"""二进制文档解析：PDF / DOCX / PPTX / XLSX → 纯文本。

依赖全部 lazy import：只有真正上传对应格式时才需要对应解析库，
缺失时抛出带安装提示的 ConfigurationError，而非笼统的 ImportError。

文本型格式（txt/md/csv/json/html）的解析仍在 loader 中完成，本模块只处理二进制格式。
"""

from __future__ import annotations

import io
from collections.abc import Callable

from app.core.errors import ConfigurationError, DocumentParseError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.ingestion.ocr import OcrEngine

logger = get_logger(__name__)

BINARY_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".xlsm"})

# 解析后文本上限：防止超大表格/文档展开成数百 MB 文本把进程撑爆
MAX_EXTRACTED_CHARS = 5_000_000

_INSTALL_HINTS = {
    ".pdf": "pypdf",
    ".docx": "python-docx",
    ".pptx": "python-pptx",
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
}


def extract_text(
    data: bytes,
    suffix: str,
    *,
    source_uri: str,
    ocr_engine: OcrEngine | None = None,
    ocr_language: str = "chi_sim+eng",
    ocr_min_chars: int = 20,
) -> str:
    """按后缀抽取文本；解析失败统一转成 DocumentParseError（带来源便于定位）。

    ocr_engine 非空时，PDF 页面若抽不出足够文字（扫描件），会尝试对页面内嵌
    图片做 OCR 后再进入统一质量门禁；OCR 失败只告警，不影响其他页的解析。
    """
    if suffix not in BINARY_SUFFIXES:
        raise UnsupportedFileTypeError(
            f"暂不支持的二进制格式：{suffix or '<无后缀>'}",
            details={"supported": sorted(BINARY_SUFFIXES)},
        )

    try:
        if suffix == ".pdf":
            text = _parse_pdf(
                data,
                ocr_engine=ocr_engine,
                ocr_language=ocr_language,
                ocr_min_chars=ocr_min_chars,
            )
        else:
            text = _PARSERS[suffix](data)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            f"文件解析失败：{source_uri}",
            details={"suffix": suffix, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise DocumentParseError(
            f"文件中未提取到文本：{source_uri}",
            details={"suffix": suffix, "hint": "扫描版 PDF / 纯图片文档需先做 OCR"},
        )
    if len(text) > MAX_EXTRACTED_CHARS:
        raise DocumentParseError(
            f"解析出的文本超过 {MAX_EXTRACTED_CHARS} 字符，请拆分后再上传：{source_uri}",
            details={"chars": len(text), "limit": MAX_EXTRACTED_CHARS},
        )
    return text


# ---------------------------------------------------------------- PDF
def _parse_pdf(
    data: bytes,
    *,
    ocr_engine: OcrEngine | None = None,
    ocr_language: str = "chi_sim+eng",
    ocr_min_chars: int = 20,
) -> str:
    """逐页抽文本，每页前缀 "## 第 N 页" 便于切块后回溯页码；加密 PDF 尝试空口令。

    扫描版 PDF 的页面文本通常为空或极少：当本页字符数低于 ocr_min_chars 时，
    把该页内嵌图片交给 OCR 引擎识别，作为该页的兜底文本。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ConfigurationError("解析 PDF 需要 pypdf：pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentParseError("PDF 已加密，无法解析，请提供未加密版本") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if ocr_engine is not None and len(text) < ocr_min_chars:
            ocr_text = _ocr_page_text(page, ocr_engine, ocr_language)
            if ocr_text:
                text = f"{text}\n{ocr_text}".strip() if text else ocr_text
        if text:
            pages.append(f"## 第 {index} 页\n{text}")
    return "\n\n".join(pages)


def _ocr_page_text(page: object, engine: OcrEngine, language: str) -> str:
    """识别单页内嵌图片；任何异常只告警并返回空串，保证整份文档仍可继续解析。"""
    try:
        images = page.images  # type: ignore[attr-defined] - pypdf PageObject 动态属性
    except Exception as exc:  # 旧版 pypdf 或异常 PDF 可能没有 images
        logger.warning("读取 PDF 页面图片失败，跳过 OCR：%s", exc)
        return ""
    parts: list[str] = []
    for image in images:
        payload = getattr(image, "data", b"")
        if not payload:
            continue
        try:
            recognized = engine.image_to_text(payload, language=language)
        except Exception as exc:
            logger.warning("PDF OCR 识别失败，保留原文本：%s", exc)
            return ""
        if recognized:
            parts.append(recognized)
    return "\n".join(parts)


# ---------------------------------------------------------------- DOCX
def _parse_docx(data: bytes) -> str:
    """段落逐行输出；表格拍平成 "单元格 | 单元格" 的行文本，牺牲结构换召回率。"""
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ConfigurationError("解析 DOCX 需要 python-docx：pip install python-docx") from exc

    document = Document(io.BytesIO(data))
    parts: list[str] = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


# ---------------------------------------------------------------- PPTX
def _parse_pptx(data: bytes) -> str:
    """每页幻灯片抽所有文本框 + 演讲者备注（[备注] 前缀），按 "## 第 N 页" 拼接。"""
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ConfigurationError("解析 PPTX 需要 python-pptx：pip install python-pptx") from exc

    presentation = Presentation(io.BytesIO(data))
    slides: list[str] = []
    for index, slide in enumerate(presentation.slides, 1):
        blocks = []
        for shape in slide.shapes:
            # python-pptx 类型桩把 text_frame 标在 GraphicFrame 等子类上，
            # 基类 BaseShape 只有 has_text_frame 运行时属性，故走 getattr
            if not getattr(shape, "has_text_frame", False):
                continue
            text = str(shape.text_frame.text).strip()
            if text:
                blocks.append(text)
        notes = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                blocks.append(f"[备注] {notes}")
        if blocks:
            slides.append(f"## 第 {index} 页\n" + "\n".join(blocks))
    return "\n\n".join(slides)


# ---------------------------------------------------------------- XLSX
def _parse_xlsx(data: bytes) -> str:
    """逐工作表抽行，非空单元格以 " | " 连接；read_only + data_only 只取计算结果。"""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ConfigurationError("解析 XLSX 需要 openpyxl：pip install openpyxl") from exc

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        sheets: list[str] = []
        for worksheet in workbook.worksheets:
            rows: list[str] = []
            for row in worksheet.iter_rows(values_only=True):
                cells = [
                    str(value).strip()
                    for value in row
                    if value is not None and str(value).strip()
                ]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                sheets.append(f"## 工作表：{worksheet.title}\n" + "\n".join(rows))
        return "\n\n".join(sheets)
    finally:
        workbook.close()


_PARSERS: dict[str, Callable[[bytes], str]] = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".pptx": _parse_pptx,
    ".xlsx": _parse_xlsx,
    ".xlsm": _parse_xlsx,
}
