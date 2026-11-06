"""统一的领域异常定义。

约定：所有可预期的业务错误都继承 AppError，API 层用 code 映射到 HTTP 状态码，
避免在业务代码里直接抛出 ValueError / KeyError 这类语义不明的异常。

错误码体系如何映射到 HTTP：
- 每个子类声明一对 (code, status_code)：code 是给前端/调用方的稳定机器可读标识
  （写进响应 JSON 的 "code" 字段），status_code 是对应的 HTTP 状态码；
- API 层注册统一的 exception handler：捕获 AppError 后调用 to_payload() 作为
  响应体、status_code 作为 HTTP 状态返回，业务路由里不需要 try/except 转译；
- 未继承 AppError 的异常（真正的 bug）不会被该 handler 捕获，交给 FastAPI
  默认的 500 处理，避免把内部堆栈误当业务错误暴露给调用方。

新增错误类型时：继承 AppError（或合适的中间类），起一个不冲突的 code，
并按语义选择状态码（4xx=调用方问题，5xx=服务端/下游问题），不要复用 code。
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """所有业务异常的基类。

    类属性：
    - code：稳定的机器可读错误码，出现在响应 JSON 中，前端可据此做分支处理；
    - status_code：该错误映射到的 HTTP 状态码，由 API 层统一 handler 读取。
    """

    code: str = "app_error"
    status_code: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """message 面向最终用户（可中文）；details 放结构化上下文（如校验错误列表）。"""
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        """序列化为 API 响应体；details 为空时省略该字段，保持响应紧凑。"""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigurationError(AppError):
    """配置缺失或非法。"""

    code = "configuration_error"
    status_code = 500


class KBConfigError(AppError):
    """知识库配置非法。"""

    code = "kb_config_error"
    status_code = 500


class KBNotFoundError(AppError):
    """知识库不存在或未启用。"""

    code = "kb_not_found"
    status_code = 404


class KBValidationError(AppError):
    """知识库配置内容非法（API 写入场景）。"""

    code = "kb_validation_error"
    status_code = 400


class KBConflictError(AppError):
    """知识库标识冲突：kb_code / kb_id / 配置文件已存在。"""

    code = "kb_conflict"
    status_code = 409


class KBInUseError(AppError):
    """知识库下仍有文档，拒绝删除。"""

    code = "kb_in_use"
    status_code = 409


class PermissionDeniedError(AppError):
    """当前用户无权访问该知识库/文档。"""

    code = "permission_denied"
    status_code = 403


class AuthenticationError(AppError):
    """身份认证失败：凭证缺失、非法或已过期。"""

    code = "authentication_error"
    status_code = 401


class TaskNotFoundError(AppError):
    """入库任务不存在。"""

    code = "task_not_found"
    status_code = 404


class ConversationNotFoundError(AppError):
    """多轮会话不存在（已删除或 id 错误）。"""

    code = "conversation_not_found"
    status_code = 404


class DocumentNotFoundError(AppError):
    """文档不存在。"""

    code = "document_not_found"
    status_code = 404


class DocumentParseError(AppError):
    """文档解析失败。"""

    code = "document_parse_error"
    status_code = 400


class UnsupportedFileTypeError(DocumentParseError):
    """不支持的文件类型。"""

    code = "unsupported_file_type"


class DocumentQualityError(DocumentParseError):
    """解析质量门禁未通过：文档被隔离，不会写入任何索引。

    与 DocumentParseError 的区别在于语义——文件本身能被打开，但产出质量不可用
    （如扫描件无文本、编码错乱、页面产出率过低），需要人工修复后重传。
    """

    code = "document_quality_rejected"
    status_code = 422


class VectorStoreError(AppError):
    """向量库读写失败。"""

    code = "vector_store_error"
    status_code = 502


class KeywordStoreError(AppError):
    """稀疏索引读写失败。"""

    code = "keyword_store_error"
    status_code = 502


class ObjectStoreError(AppError):
    """对象存储读写失败。"""

    code = "object_store_error"
    status_code = 502


class StructuredQueryError(AppError):
    """结构化查询失败：SQL 未通过安全校验、超时、执行失败等。

    这类错误可由上层降级回文档检索，因此状态码取 400（调用方问题）而非 5xx。
    """

    code = "structured_query_error"
    status_code = 400


class ModelUnavailableError(AppError):
    """模型后端不可用（远程超时、本地无 GPU/CPU 也跑不起来等）。"""

    code = "model_unavailable"
    status_code = 503


class ModelDimensionMismatchError(ModelUnavailableError):
    """降级模型的向量维度与知识库不一致。

    Embedding 降级必须同模型同维度，否则写入向量库会造成脏数据，因此直接报错。
    """

    code = "model_dimension_mismatch"
