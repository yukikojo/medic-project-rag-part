# ============================================================
# RAG 智慧医疗 AI 引擎 — Docker 镜像
# ============================================================
# 构建: docker build -t medic-rag-engine .
# 运行: docker-compose up -d
#
# GPU: 宿主机安装 nvidia-container-toolkit 后自动启用
#      未安装则自动降级 CPU 推理
# 模型: BGE-M3 + BGE-Reranker-v2-m3 首次启动自动从 HuggingFace 下载
# ============================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch GPU 版本 (CUDA 12.4, 需先装以确保不被 CPU 版覆盖)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Python 依赖 (利用层缓存)
COPY rag-db/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源码
COPY rag-db/src/ ./src/

# 数据目录
RUN mkdir -p /app/data/chromadb /app/data/models

ENV HF_HOME=/app/data/models
ENV TRANSFORMERS_CACHE=/app/data/models
ENV CHROMADB_PATH=/app/data/chromadb

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/rag/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
