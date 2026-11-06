# 通用 RAG 生产镜像：API 与 Celery worker 共用同一镜像，仅启动命令不同。
# 构建：docker build -t enterprise-rag:latest .
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml ./
COPY app ./app
RUN pip wheel --wheel-dir /wheels .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv/rag

RUN useradd --create-home --uid 10001 rag

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels enterprise-rag && rm -rf /wheels

COPY alembic ./alembic
COPY alembic.ini ./
COPY configs ./configs

# 知识库管理 API 会在 KB_CONFIG_DIR 下写 YAML，必须对运行用户可写
RUN mkdir -p logs && chown -R rag:rag /srv/rag

USER rag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
