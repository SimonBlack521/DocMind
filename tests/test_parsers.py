"""二进制文档解析测试：PDF / DOCX / PPTX / XLSX 抽取文本与 MIME 识别。

测试文件均在 tmp_path 内即时生成，不引入外部样本。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import DocumentParseError, UnsupportedFileTypeError
from app.ingestion.loader import DocumentLoader


def _write_pdf(path: Path, text: str) -> None:
    """构造最小合法 PDF（单页单行文本），避免为测试引入 reportlab。"""
    content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))


def _loader() -> DocumentLoader:
    return DocumentLoader(tenant_id="t1", kb_id="kb-finance")


def test_loader_extracts_pdf_text(tmp_path: Path) -> None:
    pdf = tmp_path / "policy.pdf"
    _write_pdf(pdf, "Travel reimbursement limit is 500 CNY")

    document = _loader().load_path(pdf)

    assert "Travel reimbursement limit is 500 CNY" in document.content
    assert document.mime_type == "application/pdf"
    assert len(document.doc_id) == 64


def test_loader_extracts_docx_paragraphs_and_tables(tmp_path: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("差旅报销制度", level=1)
    doc.add_paragraph("住宿费一线城市每晚不超过 500 元。")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "城市"
    table.cell(0, 1).text = "限额"
    table.cell(1, 0).text = "北京"
    table.cell(1, 1).text = "500"
    path = tmp_path / "报销制度.docx"
    doc.save(path)

    document = _loader().load_path(path)

    assert "差旅报销制度" in document.content
    assert "住宿费一线城市每晚不超过 500 元。" in document.content
    assert "北京 | 500" in document.content
    assert document.mime_type.endswith("wordprocessingml.document")


def test_loader_extracts_pptx_slide_text(tmp_path: Path) -> None:
    from pptx import Presentation
    from pptx.util import Inches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(2))
    box.text_frame.text = "第三季度预算执行说明"
    path = tmp_path / "预算汇报.pptx"
    presentation.save(path)

    document = _loader().load_path(path)

    assert "第三季度预算执行说明" in document.content
    assert "第 1 页" in document.content


def test_loader_extracts_xlsx_sheets(tmp_path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "报销明细"
    sheet.append(["申请人", "金额", "状态"])
    sheet.append(["alice", 1200, "已通过"])
    # 第二张表：验证多 sheet 分段
    second = workbook.create_sheet("预算")
    second.append(["科目", "年度预算"])
    second.append(["差旅", 50000])
    path = tmp_path / "finance.xlsx"
    workbook.save(path)

    document = _loader().load_path(path)

    assert "工作表：报销明细" in document.content
    assert "alice | 1200 | 已通过" in document.content
    assert "工作表：预算" in document.content
    assert "差旅 | 50000" in document.content


def test_loader_rejects_binary_without_text(tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "empty.xlsx"
    Workbook().save(path)  # 仅空白 sheet，无任何单元格

    with pytest.raises(DocumentParseError):
        _loader().load_path(path)


def test_loader_rejects_corrupted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf")

    with pytest.raises(DocumentParseError):
        _loader().load_path(path)


def test_loader_rejects_legacy_doc(tmp_path: Path) -> None:
    legacy = tmp_path / "manual.doc"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0")

    with pytest.raises(UnsupportedFileTypeError):
        _loader().load_path(legacy)
