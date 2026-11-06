"""实时 API 检索（v2.0）：库存/订单状态等"必须现查"的问题走外部只读接口。

与 SQL 检索的差异：
- SQL 面向"已落库的历史统计"，API 面向"此刻的实时状态"（库存、工单进度等）；
- 接口地址必须由运维在知识库配置中显式登记，系统不接受任意 URL；
- 返回内容按 JSON/文本原样摘要，不做 LLM 二次加工，避免实时数字被改写；
- 任何网络/状态码/解析失败都转 StructuredQueryError，由编排层降级文档检索。
"""

from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from app.core.errors import StructuredQueryError
from app.core.logging import get_logger
from app.domain.kb_config import ApiEndpointConfig, ApiRetrievalConfig

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ApiAnswer:
    """实时 API 查询结果：答案文本 + 命中的端点 + HTTP 状态码（供审计）。"""

    answer: str
    endpoint_id: str
    status_code: int


class ApiRetriever:
    """按问题匹配已登记端点并执行 GET 查询；端点间无 LLM 选择，纯规则打分。"""

    def __init__(
        self,
        config: ApiRetrievalConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client

    def answer(self, question: str) -> ApiAnswer:
        """选择端点 → 调用 → 摘要返回；所有失败路径统一抛 StructuredQueryError。

        响应体以流式读取并执行严格字节预算（v2.1）：先完整接收再检查长度时，
        超大响应会在大小检查生效前就占满内存；流式读取每收一段就累加计数，
        超过 max_response_bytes 立即中断连接。
        """
        endpoint = self._select_endpoint(question)
        _ensure_safe_url(endpoint.url, allow_private=self._config.allow_private_hosts)
        client = self._client or httpx.Client(follow_redirects=False)
        try:
            with client.stream(
                "GET",
                endpoint.url,
                params={endpoint.query_param: question},
                headers=endpoint.headers,
                timeout=endpoint.timeout_s,
            ) as response:
                if response.status_code >= 400:
                    raise StructuredQueryError(
                        "实时接口返回错误",
                        details={
                            "endpoint": endpoint.endpoint_id,
                            "status_code": response.status_code,
                        },
                    )
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not any(
                    content_type.startswith(prefix)
                    for prefix in self._config.allowed_content_types
                ):
                    raise StructuredQueryError(
                        "实时接口返回了不支持的内容类型",
                        details={"content_type": content_type},
                    )
                payload_bytes = _read_with_budget(response, self._config.max_response_bytes)
        except httpx.HTTPError as exc:
            raise StructuredQueryError(
                "实时接口请求失败",
                details={"endpoint": endpoint.endpoint_id, "error": str(exc)[:200]},
            ) from exc

        payload = _summarize_payload(payload_bytes, content_type)
        answer = f"来源：{endpoint.endpoint_id}（HTTP {response.status_code}）\n{payload}"
        return ApiAnswer(
            answer=answer[: self._config.max_chars],
            endpoint_id=endpoint.endpoint_id,
            status_code=response.status_code,
        )

    def _select_endpoint(self, question: str) -> ApiEndpointConfig:
        """端点选择：单个端点直接使用；多个端点按"问题与端点描述/ID 的词重叠"打分。

        规则打分而不是让 LLM 选：端点数量通常很少，确定性选择可预测、无额外延迟；
        多端点时必须达到 `min_match_score`，否则返回路由失败——盲调第一个端点可能
        把库存问题发给订单接口，比"查不到"更危险。
        """
        endpoints = self._config.endpoints
        if not endpoints:
            raise StructuredQueryError("实时 API 检索未配置任何端点")
        if len(endpoints) == 1:
            return endpoints[0]

        text = question.lower()
        best = endpoints[0]
        best_score = -1
        for endpoint in endpoints:
            haystack = f"{endpoint.endpoint_id} {endpoint.description}".lower()
            score = sum(1 for token in _tokens(haystack) if token in text)
            if score > best_score:
                best = endpoint
                best_score = score
        if best_score < self._config.min_match_score:
            raise StructuredQueryError(
                "没有匹配的实时接口端点",
                details={"best_score": best_score, "question": question[:100]},
            )
        return best


def _tokens(text: str) -> set[str]:
    """粗粒度词元：英文按非字母数字切分，中文按连续汉字切 2~4 字片段。"""
    result: set[str] = set()
    for word in _split_words(text):
        if len(word) <= 1:
            continue
        result.add(word)
        # 中文端点描述常是整句，切成 2 字片段提高与问句的重叠概率
        if _is_cjk(word):
            result.update(word[index : index + 2] for index in range(len(word) - 1))
    return result


def _split_words(text: str) -> list[str]:
    """简单分词：连续字母数字或连续汉字视为一个词，其余字符为分隔符。"""
    words: list[str] = []
    buffer = ""
    for char in text:
        if char.isalnum():
            buffer += char
        elif buffer:
            words.append(buffer)
            buffer = ""
    if buffer:
        words.append(buffer)
    return words


def _is_cjk(word: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in word)


# 云厂商元数据/内部别名：即使被解析成公网地址也必须拒绝
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata",
    }
)


def _ensure_safe_url(url: str, *, allow_private: bool) -> None:
    """SSRF 防护：拒绝非常规 scheme、内嵌凭据、私有/回环/链路本地/云元数据地址。

    DNS 无法解析时不再做 IP 检查：解析失败意味着请求大概率无法建立连接，
    真正可达的域名仍会经过解析结果校验。DNS Rebinding（解析后换 IP）无法仅靠
    这里彻底消除，生产环境建议配合出站代理与目标 Host 白名单。
    """
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise StructuredQueryError("实时接口只允许 http(s) 地址")
    if parsed.username or parsed.password:
        raise StructuredQueryError("实时接口地址不允许内嵌认证信息")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise StructuredQueryError("实时接口地址缺少主机名")
    if host.startswith(("127.", "0.0.0.0")) or host in _BLOCKED_HOSTNAMES:
        raise StructuredQueryError(
            "实时接口地址指向本机或元数据服务，已拒绝", details={"host": host}
        )

    literal = _parse_ip(host)
    if literal is not None:
        _reject_private_ip(literal, host, allow_private=allow_private)
        return

    try:
        default_port = 443 if parsed.scheme == "https" else 80
        infos = socket.getaddrinfo(
            host, parsed.port or default_port, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        # 解析失败时交给实际请求去报错；不要因为测试域名/临时 DNS 抖动误拒绝
        logger.debug("实时接口域名解析失败 host=%s：%s", host, exc)
        return
    for info in infos:
        address = _parse_ip(str(info[4][0]))
        if address is not None:
            _reject_private_ip(address, host, allow_private=allow_private)


def _parse_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _reject_private_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    host: str,
    *,
    allow_private: bool,
) -> None:
    if allow_private:
        return
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise StructuredQueryError(
            "实时接口地址指向内网/保留地址，已拒绝（如确为内网接口请显式配置 allow_private_hosts）",
            details={"host": host, "ip": str(address)},
        )


def _read_with_budget(response: httpx.Response, max_bytes: int) -> bytes:
    """流式读取响应体并执行严格字节预算（v2.1，清单 P1-06）。

    旧实现先 client.get() 完整接收再检查长度：超大响应在大小检查生效前就已经
    占满内存。改为流式逐块读取、累加计数，超限立即抛错中断连接——攻击者
    即使不发 Content-Length（分块传输）也无法让服务端缓冲超预算数据。
    Content-Length 声明超限时直接拒绝，不建立读取。
    """
    length_header = response.headers.get("content-length")
    if (
        length_header
        and length_header.isdigit()
        and int(length_header) > max_bytes
    ):
        raise StructuredQueryError(
            "实时接口响应体超过大小上限",
            details={"content_length": int(length_header), "limit": max_bytes},
        )
    buffer = bytearray()
    for chunk in response.iter_bytes():
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise StructuredQueryError(
                "实时接口响应体超过大小上限",
                details={"size": len(buffer), "limit": max_bytes},
            )
    return bytes(buffer)


def _summarize_payload(data: bytes, content_type: str) -> str:
    """响应摘要：JSON 格式化输出，非 JSON 按原文截断；不做任何字段改写。"""
    if "json" in content_type:
        try:
            payload = json.loads(data.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            return json.dumps(payload, ensure_ascii=False, indent=2)
    text = data.decode("utf-8", errors="replace").strip()
    return text[:2000] if text else "（接口返回空响应）"
