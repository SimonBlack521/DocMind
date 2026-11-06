"""日志初始化：控制台 + 可选文件轮转，避免重复添加 handler。

在整体链路中的角色：进程启动早期（API 与 Celery worker 入口）调用一次
setup_logging()，此后各模块统一用 get_logger(__name__) 获取 logger，
禁止业务代码直接 print 或自行 basicConfig，保证全进程输出格式一致。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_dir: Path | str | None = None) -> None:
    """初始化 root logger：幂等，可安全重复调用（如测试多次初始化 app）。

    :param level: 日志级别字符串（"INFO" 等），内部转大写后设置；
    :param log_dir: 日志目录；None 表示只输出到控制台（适合容器 stdout 采集）。

    幂等的两层防护：
    - 控制台 handler：root 已有任意 handler 时不再重复添加，避免重复打印；
    - 文件 handler：按 RotatingFileHandler + 目标路径判重，重复调用不会
      对同一文件叠加多个 handler（否则每行日志会被写多份）。
    文件不可写（权限/只读盘）只降级为控制台输出并告警，不阻断启动。
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(console)

    if log_dir is None:
        return

    log_path = Path(log_dir) / "app.log"
    if any(
        isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path
        for handler in root.handlers
    ):
        return

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
    except OSError:
        # 日志目录不可写时只保留控制台输出，不影响主流程
        logging.getLogger(__name__).warning("日志目录不可写，已跳过文件日志：%s", log_path)
        return

    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """按模块名取 logger（约定传 __name__），便于按包路径过滤与定位日志来源。"""
    return logging.getLogger(name)
