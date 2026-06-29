# ============================================================
# RAG 智慧医疗 AI 引擎 — Docker 镜像
# ============================================================
# 构建: docker build -t medic-rag-engine .
# 运行: docker-compose up -d
#
# GPU 要求: nvidia-container-toolkit
# 模型: BGE-M3 + BGE-Reranker-v2-m3 首次启动自动从 HuggingFace 下载
# ============================================================

FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装 Python 3.12 + 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 应用目录
WORKDIR /app

# 先安装 Python 依赖 (利用 Docker 层缓存)
COPY rag-db/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY rag-db/src/ ./src/

# 数据目录 (ChromaDB 持久化 + 模型缓存)
RUN mkdir -p /app/data/chromadb /app/data/models

# HuggingFace 缓存目录 (模型自动下载到这里)
ENV HF_HOME=/app/data/models
ENV TRANSFORMERS_CACHE=/app/data/models

# ChromaDB 存储路径
ENV CHROMADB_PATH=/app/data/chromadb

# 暴露 FastAPI 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/rag/health')" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
