"""轻量进程内运行指标：计数器 + 耗时观测，供 GET /metrics 暴露 JSON 快照。

不引入外部依赖（不假设部署侧一定有 Prometheus）；事件循环与线程池都会写入，
因此内部用锁保证线程安全。指标随进程重启清零，仅用于运行态观测与冒烟断言。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(slots=True)
class _Summary:
    """单次耗时观测的聚合：次数 / 总和 / 峰值。"""

    count: int = 0
    total: float = 0.0
    maximum: float = 0.0


class Metrics:
    """进程内指标注册表：计数器（inc）与耗时观测（observe）两类原语。

    设计取舍：
    - key 采用 Prometheus 风格的 "name{label=...}" 字符串，使 JSON 快照与
      Prometheus 文本格式可以共用同一份底层数据，无需维护两套标签结构；
    - 只有一把锁、粒度到整个 dict。写入路径是 O(1) 字典操作，临界区极短，
      对请求延迟的影响可忽略，不需要无锁结构。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._summaries: dict[str, _Summary] = {}
        self._started = time.monotonic()

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        """标签拼进 key：http_requests_total{method="POST",route="...",status="200"}。"""
        if not labels:
            return name
        inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{inner}}}"

    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        """计数器累加。如 metrics.inc("http_requests_total", method="POST", status="200")。"""
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def observe(self, name: str, value: float, **labels: str) -> None:
        """记录一次耗时/大小观测，聚合出 count / sum / max（snapshot 时算 avg）。"""
        key = self._key(name, labels)
        with self._lock:
            summary = self._summaries.setdefault(key, _Summary())
            summary.count += 1
            summary.total += value
            if value > summary.maximum:
                summary.maximum = value

    def snapshot(self) -> dict[str, Any]:
        """导出当前指标的 JSON 快照（GET /metrics 的响应体来源）。

        顺带计算检索缓存命中率（hits / (hits + misses)）；无任何缓存访问时
        返回 None 而不是 0.0，避免把"没流量"误读成"缓存全失效"。
        """
        with self._lock:
            counters = dict(self._counters)
            summaries = {
                key: {
                    "count": s.count,
                    "sum": round(s.total, 6),
                    "max": round(s.maximum, 6),
                    "avg": round(s.total / s.count, 6) if s.count else 0.0,
                }
                for key, s in self._summaries.items()
            }
        hits = counters.get("retrieval_cache_hits_total", 0.0)
        misses = counters.get("retrieval_cache_misses_total", 0.0)
        total = hits + misses
        return {
            "uptime_seconds": round(time.monotonic() - self._started, 3),
            "counters": counters,
            "summaries": summaries,
            "retrieval_cache_hit_rate": round(hits / total, 4) if total else None,
        }

    # ---------- Prometheus exposition format ----------

    # 指标 → HELP 描述（Prometheus 规范要求）
    _HELP: ClassVar[dict[str, str]] = {
        "http_requests_total": "Total number of HTTP requests.",
        "retrieval_cache_hits_total": "Retrieval cache hits.",
        "retrieval_cache_misses_total": "Retrieval cache misses.",
        "audit_events_total": "Total audit events written.",
        "http_request_duration_seconds": "HTTP request latency in seconds.",
        "llm_chat_duration_seconds": "LLM non-streaming chat latency in seconds.",
        "llm_stream_duration_seconds": "LLM streaming chat latency in seconds.",
        "multi_query_variants_count": "Number of query variants per retrieval.",
        "rerank_duration_seconds": "Reranker latency in seconds.",
        "retrieval_duration_seconds": "Hybrid retrieval latency in seconds.",
        "uptime_seconds": "Application uptime in seconds.",
    }

    def _prometheus_lines(self) -> list[str]:
        """把内部计数器/耗时聚合转成 Prometheus text exposition。"""
        lines: list[str] = []
        snapshot = self.snapshot()
        help_emitted: set[str] = set()

        # uptime（gauge）
        lines.append("# HELP uptime_seconds Application uptime in seconds.")
        lines.append("# TYPE uptime_seconds gauge")
        lines.append(f"uptime_seconds {snapshot['uptime_seconds']}")

        def _emit_help(base: str, metric_type: str) -> None:
            if base in help_emitted:
                return
            help_emitted.add(base)
            help_text = self._HELP.get(base, f"{base} (auto-generated)")
            lines.append(f"# HELP {base} {help_text}")
            lines.append(f"# TYPE {base} {metric_type}")

        # counters：name{labels} value，TYPE counter
        for key, value in sorted(snapshot["counters"].items()):
            base = key.split("{", 1)[0]
            _emit_help(base, "counter")
            lines.append(f"{key} {value}")

        # summaries：_count/_sum 是 Summary 标准后缀，TYPE summary；
        # _max 不是 Summary 标准样本（Summary 只有 quantile/_count/_sum），
        # 单独按 gauge 声明——否则部分指标会被误标为 Counter 却输出 _count/_sum/_max
        max_type_emitted: set[str] = set()
        for key, s in sorted(snapshot["summaries"].items()):
            base = key.split("{", 1)[0]
            _emit_help(base, "summary")
            suffix = key[len(base) :]  # {labels}
            lines.append(f"{base}_count{suffix} {s['count']}")
            lines.append(f"{base}_sum{suffix} {s['sum']}")
            if base not in max_type_emitted:
                max_type_emitted.add(base)
                lines.append(f"# TYPE {base}_max gauge")
            lines.append(f"{base}_max{suffix} {s['max']}")

        return lines

    def prometheus_exposition(self) -> str:
        """返回 Prometheus exposition format 文本（Content-Type: text/plain）。"""
        return "\n".join(self._prometheus_lines()) + "\n"


# 进程级单例：中间件、管线、缓存等各处直接引用
metrics = Metrics()
