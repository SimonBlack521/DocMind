"""多模态解析的可选 OCR 钩子（v2.0）：只处理"扫描版 PDF 抽不出文字"的场景。

设计原则：
- OCR 是可选能力：依赖（pytesseract + Pillow + 本机 tesseract）缺失时返回 None，
  调用方保持原有解析行为，不让整个入库链路因缺少可选依赖而失败；
- 只有页面文本少于阈值时才触发 OCR，避免对正常文本型 PDF 做昂贵且低质的识别；
- 引擎以 Protocol 抽象，便于离线评测时注入假引擎，也便于后续替换为云 OCR。
"""

from __future__ import annotations

import importlib.util
import io
from typing import Protocol

from app.core.logging import get_logger

logger = get_logger(__name__)


class OcrEngine(Protocol):
    """OCR 引擎抽象：输入图片字节，输出识别文本。"""

    def image_to_text(self, image_bytes: bytes, *, language: str) -> str:
        """识别单张图片中的文字；失败时应抛出异常由调用方兜底。"""
        ...


class TesseractOcrEngine:
    """基于 pytesseract 的本地 OCR 实现；依赖全部懒加载。"""

    def image_to_text(self, image_bytes: bytes, *, language: str) -> str:
        import pytesseract
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            return pytesseract.image_to_string(image, lang=language).strip()


def ocr_dependencies_available() -> bool:
    """只检查 Python 依赖是否存在；tesseract 可执行文件是否安装由实际调用时暴露。"""
    return (
        importlib.util.find_spec("pytesseract") is not None
        and importlib.util.find_spec("PIL") is not None
    )


def build_ocr_engine(*, enabled: bool) -> OcrEngine | None:
    """按配置构建 OCR 引擎；关闭或依赖缺失时返回 None（优雅降级）。"""
    if not enabled:
        return None
    if not ocr_dependencies_available():
        logger.warning("OCR 已启用但缺少 pytesseract/Pillow，跳过 OCR 直接按文本解析")
        return None
    return TesseractOcrEngine()
