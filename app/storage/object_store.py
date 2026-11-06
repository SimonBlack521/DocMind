"""对象存储：保存原始文件（MinIO / S3 兼容），与向量库解耦。

对象 key 规则：tenants/{tenant_id}/kbs/{kb_id}/docs/{doc_id}/v{version}/{filename}
同一文档重新上传会覆盖新版本目录，doc_id 不变。
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.errors import ObjectStoreError

_UNSAFE_FILENAME = re.compile(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+")


def _safe_filename(filename: str) -> str:
    """清洗文件名：剥离路径、只保留字母数字/点/横线/中文，防 key 注入与非法字符。"""
    name = Path(filename).name
    cleaned = _UNSAFE_FILENAME.sub("_", name).strip("._")
    return cleaned or "file"


def build_document_key(
    *, tenant_id: str, kb_id: str, doc_id: str, version: int, filename: str
) -> str:
    return (
        f"tenants/{tenant_id}/kbs/{kb_id}/docs/{doc_id}/"
        f"v{version}/{_safe_filename(filename)}"
    )


class ObjectStore(ABC):
    @abstractmethod
    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """写入对象，返回 key。"""

    @abstractmethod
    def put_fileobj(
        self, key: str, path: Path, *, content_type: str = "application/octet-stream"
    ) -> str:
        """从磁盘文件流式写入对象（不一次性读入内存），返回 key。"""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """读取对象内容。"""

    @abstractmethod
    def delete(self, key: str) -> None:
        """删除对象；不存在时视为成功。"""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """判断对象是否存在。"""

    @abstractmethod
    def presigned_get_url(self, key: str, *, expires_seconds: int = 3600) -> str:
        """生成临时下载链接。"""


class S3ObjectStore(ObjectStore):
    """boto3 实现的 S3/MinIO 存储；path-style 寻址兼容 MinIO，bucket 首次写入时惰性创建。"""

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
        use_ssl: bool = False,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ObjectStoreError(
                "未安装 boto3，无法使用 MinIO/S3 对象存储（pip install boto3）"
            ) from exc

        self._bucket = bucket
        self._bucket_ready = False
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            use_ssl=use_ssl,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self._bucket)
            except Exception as exc:
                raise ObjectStoreError(
                    f"对象存储 bucket 不可用：{self._bucket}", details={"error": str(exc)}
                ) from exc
        self._bucket_ready = True

    def put_bytes(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        self._ensure_bucket()
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
            )
        except Exception as exc:
            raise ObjectStoreError(f"对象写入失败：{key}", details={"error": str(exc)}) from exc
        return key

    def put_fileobj(
        self, key: str, path: Path, *, content_type: str = "application/octet-stream"
    ) -> str:
        self._ensure_bucket()
        try:
            with open(path, "rb") as fh:
                self._client.put_object(
                    Bucket=self._bucket, Key=key, Body=fh, ContentType=content_type
                )
        except Exception as exc:
            raise ObjectStoreError(f"对象写入失败：{key}", details={"error": str(exc)}) from exc
        return key

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            body: bytes = response["Body"].read()
        except Exception as exc:
            raise ObjectStoreError(f"对象读取失败：{key}", details={"error": str(exc)}) from exc
        return body

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            raise ObjectStoreError(f"对象删除失败：{key}", details={"error": str(exc)}) from exc

    def exists(self, key: str) -> bool:
        try:
            response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=key, MaxKeys=1)
        except Exception as exc:
            raise ObjectStoreError(f"对象状态查询失败：{key}", details={"error": str(exc)}) from exc
        return bool(response.get("Contents"))

    def presigned_get_url(self, key: str, *, expires_seconds: int = 3600) -> str:
        try:
            url: str = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_seconds,
            )
        except Exception as exc:
            raise ObjectStoreError(
                f"预签名链接生成失败：{key}", details={"error": str(exc)}
            ) from exc
        return url

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
