"""文档加载：负责稳定 doc_id、内容校验和与文本抽取。

doc_id 规则（全局唯一且可重复计算的锚点）：
    doc_id = sha256(tenant_id + kb_id + 归一化后的 source_uri)

这样同一份文件重复上传，无论文件名大小写、路径分隔符、URL fragment 如何变化，
都会落到同一个 doc_id，便于覆盖写入与幂等重跑。
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from importlib import metadata
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit, urlunsplit

from app.core.errors import DocumentParseError, UnsupportedFileTypeError
from app.domain.governance import compute_content_hash
from app.domain.privacy import redact_pii
from app.ingestion.code_parser import render_python_structure
from app.ingestion.ocr import OcrEngine
from app.ingestion.parsers import BINARY_SUFFIXES, MAX_EXTRACTED_CHARS, extract_text

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm",
    # 代码与配置类知识源：按文本读取，.py 额外做 AST 结构化
    ".py", ".js", ".ts", ".tsx", ".java", ".go", ".sql", ".yaml", ".yml",
}
ARCHIVE_SUFFIXES = {".zip"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | BINARY_SUFFIXES | ARCHIVE_SUFFIXES

# 解析器名称与发行包名：把"哪个解析器产出了这份文本"随文档一起留存，
# 便于解析结果出现回归时定位到具体的库版本（可复现性要求）。
PARSER_NAMES = {
    ".pdf": "pypdf",
    ".docx": "python-docx",
    ".pptx": "python-pptx",
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
    ".py": "python-ast",
    ".zip": "zipfile",
}
_PARSER_PACKAGES = {
    "pypdf": "pypdf",
    "python-docx": "python-docx",
    "python-pptx": "python-pptx",
    "openpyxl": "openpyxl",
}

_MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
}


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    doc_id: str
    source_uri: str
    title: str
    content: str
    mime_type: str
    # 归一化正文指纹：去重与内容漂移检测的依据
    content_hash: str = ""
    # 解析链路版本留存：解析器名 + 实际发行包版本
    parser_name: str = "builtin-text"
    parser_version: str = ""
    # 入库前脱敏命中的 PII 计数（邮箱/手机号/身份证/银行卡），随质量报告落库
    pii_findings: dict[str, int] = field(default_factory=dict)


def _parser_identity(suffix: str) -> tuple[str, str]:
    """返回 (解析器名, 发行包版本)；文本类走内置读取，版本记为 interpreter。"""
    parser_name = PARSER_NAMES.get(suffix, "builtin-text")
    package = _PARSER_PACKAGES.get(parser_name)
    if package is None:
        return parser_name, "stdlib"
    try:
        return parser_name, metadata.version(package)
    except metadata.PackageNotFoundError:  # pragma: no cover - 依赖未安装时才会走到
        return parser_name, "unknown"


def normalize_source_uri(source_uri: str) -> str:
    """归一化来源地址，作为 doc_id 的稳定输入。"""
    raw = source_uri.strip()
    if not raw:
        raise DocumentParseError("source_uri 不能为空")

    parts = urlsplit(raw)
    if parts.scheme and parts.netloc:
        # URL：协议/域名统一小写，去掉 fragment，去掉结尾多余的 /
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))

    # 本地路径：统一使用正斜杠并去掉结尾分隔符
    path = raw.replace("\\", "/").rstrip("/")
    return path


def compute_doc_id(tenant_id: str, kb_id: str, source_uri: str) -> str:
    """doc_id = sha256(tenant|kb|归一化 source_uri)，重复上传同文件必得同 id。

    业务语义（清单第 5 项）：同一 source_uri 即同一文档——内容更新走版本升级
    （version 递增），重复提交幂等复用；(kb_id, source_uri) 唯一约束保证同库
    同来源只有一条记录，不存在"同 uri 多 doc_id"的歧义。
    """
    normalized = normalize_source_uri(source_uri)
    payload = f"{tenant_id}|{kb_id}|{normalized}".encode()
    return hashlib.sha256(payload).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return path.read_text(encoding="gb18030")
    except (UnicodeDecodeError, OSError) as exc:
        raise DocumentParseError(f"无法解析文件编码：{path.name}") from exc


class _HtmlTextExtractor(HTMLParser):
    """去掉 script/style 后拼接文本，并把标题/块级元素转成 Markdown 结构标记。

    保留 `# 一级标题` 与换行，是为了让下游结构感知切块能识别小节边界；
    否则 HTML 正文会被压成一整段，只能按字符硬切。
    """

    _BLOCK_TAGS: ClassVar[set[str]] = {
        "p", "div", "br", "li", "tr", "table", "section", "article",
        "header", "footer", "ul", "ol", "blockquote", "pre",
    }

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit() and 1 <= int(tag[1]) <= 6:
            self._parts.append("\n" + "#" * int(tag[1]) + " ")
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if (len(tag) == 2 and tag[0] == "h" and tag[1].isdigit()) or tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text and self._skip_depth == 0:
            self._parts.append(text)

    def text(self) -> str:
        return "\n".join(self._parts)


def _extract_html(text: str) -> str:
    parser = _HtmlTextExtractor()
    parser.feed(text)
    parser.close()
    return parser.text()


class DocumentLoader:
    """按租户 + 知识库加载文档，所有入口都会产出稳定的 doc_id。"""

    def __init__(
        self,
        *,
        tenant_id: str,
        kb_id: str,
        redact_pii: bool = False,
        ocr_engine: OcrEngine | None = None,
        ocr_language: str = "chi_sim+eng",
        ocr_min_chars: int = 20,
    ) -> None:
        if not tenant_id or not kb_id:
            raise DocumentParseError("tenant_id 与 kb_id 不能为空")
        self._tenant_id = tenant_id
        self._kb_id = kb_id
        # 是否在入库前做 PII 脱敏：由全局 Settings.privacy_redaction_enabled 控制，
        # 关闭时保留原文（仅建议在纯内网/离线评测环境使用）
        self._redact_pii = redact_pii
        # 可选 OCR 引擎：None 表示关闭；扫描版 PDF 抽不出文字时才启用
        self._ocr_engine = ocr_engine
        self._ocr_language = ocr_language
        self._ocr_min_chars = ocr_min_chars

    def load_path(self, path: Path | str, *, source_uri: str | None = None) -> LoadedDocument:
        """从磁盘加载：按后缀分流到二进制解析或文本读取，再统一走 load_bytes。"""
        file_path = Path(path)
        if not file_path.is_file():
            raise DocumentParseError(f"文件不存在：{file_path}")

        suffix = file_path.suffix.lower()
        if suffix in BINARY_SUFFIXES or suffix in ARCHIVE_SUFFIXES:
            data: bytes | None = file_path.read_bytes()
            text: str | None = None
        elif suffix in TEXT_SUFFIXES:
            data = None
            text = _read_text(file_path)
        else:
            raise UnsupportedFileTypeError(
                f"暂不支持的文件类型：{suffix or '<无后缀>'}",
                details={"supported": sorted(SUPPORTED_SUFFIXES)},
            )

        return self.load_bytes(
            data=data,
            text=text,
            source_uri=source_uri or str(file_path.resolve()),
            title=file_path.stem,
            suffix=suffix,
        )

    def load_bytes(
        self,
        data: bytes | None,
        *,
        text: str | None = None,
        source_uri: str,
        title: str = "",
        suffix: str | None = None,
        mime_type: str | None = None,
    ) -> LoadedDocument:
        """支持从内存加载：data 与 text 二选一（二进制格式必须提供 data）。"""
        normalized_suffix = (suffix or Path(urlsplit(source_uri).path).suffix).lower()
        if normalized_suffix not in SUPPORTED_SUFFIXES:
            raise UnsupportedFileTypeError(
                f"暂不支持的文件类型：{normalized_suffix or '<无后缀>'}",
                details={"supported": sorted(SUPPORTED_SUFFIXES)},
            )

        if normalized_suffix in ARCHIVE_SUFFIXES:
            if data is None:
                raise DocumentParseError(f"压缩包必须提供文件内容：{source_uri}")
            content = _extract_zip(data, source_uri)
        elif normalized_suffix in BINARY_SUFFIXES:
            if data is None:
                raise DocumentParseError(f"二进制格式必须提供文件内容：{source_uri}")
            content = extract_text(
                data,
                normalized_suffix,
                source_uri=source_uri,
                ocr_engine=self._ocr_engine,
                ocr_language=self._ocr_language,
                ocr_min_chars=self._ocr_min_chars,
            )
        else:
            if text is None:
                if data is None:
                    raise DocumentParseError("data 与 text 必须提供其中一个")
                raw_text = _decode_bytes(data, source_uri)
            else:
                raw_text = text

            if normalized_suffix in {".html", ".htm"}:
                content = _extract_html(raw_text)
            elif normalized_suffix == ".py":
                content = render_python_structure(raw_text)
            else:
                content = raw_text

        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not content:
            raise DocumentParseError(f"文档内容为空：{source_uri}")

        # PII 脱敏发生在指纹计算之前：入库文本、content_hash、切块内容三者口径一致
        pii_findings: dict[str, int] = {}
        if self._redact_pii:
            content, pii_findings = redact_pii(content)

        normalized_uri = normalize_source_uri(source_uri)
        parser_name, parser_version = _parser_identity(normalized_suffix)
        return LoadedDocument(
            doc_id=compute_doc_id(self._tenant_id, self._kb_id, source_uri),
            source_uri=normalized_uri,
            title=title or Path(urlsplit(normalized_uri).path).stem or normalized_uri,
            content=content,
            mime_type=mime_type or _MIME_TYPES.get(normalized_suffix, "text/plain"),
            content_hash=compute_content_hash(content),
            parser_name=parser_name,
            parser_version=parser_version,
            pii_findings=pii_findings,
        )


def _extract_zip(data: bytes, source_uri: str) -> str:
    """代码仓库/文档包：逐文件抽取并加 `## 文件：路径` 结构标记。

    策略是"严格而非容错"：任一文件解析失败都让整包失败，避免压缩包里某些文件
    被静默跳过、索引里只剩残缺内容。单个文件过大或总量超限同样直接拒绝。
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentParseError(f"压缩包无法打开：{source_uri}") from exc

    parts: list[str] = []
    total_chars = 0
    # 解压安全预算：单条目解压后不得超过 64MB，全包累计解压量不得超过 256MB，
    # 压缩比（解压后/压缩后）超过 100 视为解压炸弹直接拒绝
    total_decompressed = 0
    for info in sorted(archive.infolist(), key=lambda item: item.filename):
        if info.is_dir():
            continue
        if len(parts) >= 500:
            raise DocumentParseError(f"压缩包内文件数超过 500：{source_uri}")
        if info.file_size > 64 * 1024 * 1024:
            raise DocumentParseError(
                f"压缩包内单个文件解压后超过 64MB：{info.filename}"
            )
        if info.file_size > 100 * max(info.compress_size, 1):
            raise DocumentParseError(
                f"压缩包内文件压缩比异常（疑似解压炸弹）：{info.filename}"
            )
        total_decompressed += info.file_size
        if total_decompressed > 256 * 1024 * 1024:
            raise DocumentParseError(
                f"压缩包累计解压量超过 256MB 预算：{source_uri}"
            )
        suffix = Path(info.filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        blob = archive.read(info)
        try:
            if suffix in ARCHIVE_SUFFIXES:
                # 不支持嵌套压缩包，避免解压炸弹与递归深度问题
                raise DocumentParseError(f"不支持嵌套压缩包：{info.filename}")
            if suffix in BINARY_SUFFIXES:
                text = extract_text(blob, suffix, source_uri=f"{source_uri}::{info.filename}")
            else:
                text = _decode_bytes(blob, f"{source_uri}::{info.filename}")
                if suffix in {".html", ".htm"}:
                    text = _extract_html(text)
                elif suffix == ".py":
                    text = render_python_structure(text)
        except (DocumentParseError, UnsupportedFileTypeError) as exc:
            raise DocumentParseError(
                f"压缩包内文件解析失败：{info.filename}，原因：{exc}"
            ) from exc

        stripped = text.strip()
        if not stripped:
            continue
        total_chars += len(stripped)
        if total_chars > MAX_EXTRACTED_CHARS:
            raise DocumentParseError(
                f"压缩包抽取文本超过上限 {MAX_EXTRACTED_CHARS} 字符：{source_uri}"
            )
        parts.append(f"## 文件：{info.filename}\n{stripped}")

    if not parts:
        raise DocumentParseError(f"压缩包内没有可解析的文件：{source_uri}")
    return "\n\n".join(parts)


def _decode_bytes(data: bytes, source_uri: str) -> str:
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError(f"无法解析文件编码：{source_uri}")
