# ============================================================
# RAG 智慧医疗 AI 引擎 — Docker 镜像
# ============================================================
# 构建: docker-compose build --no-cache && docker-compose up -d
# GPU: 宿主机需安装 nvidia-container-toolkit
# ============================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ---- apt 镜像加速 (阿里云) ----
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ---- pip 镜像加速 (阿里云) ----
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# ---- PyTorch GPU (CUDA 12.4, 国内无镜像, 走官方源) ----
RUN pip install --no-cache-dir --default-timeout=600 --retries 10 \
    torch --index-url https://download.pytorch.org/whl/cu124

# ---- Python 依赖 ----
COPY rag-db/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 源码 ----
COPY rag-db/src/ ./src/

# ---- 数据目录 ----
RUN mkdir -p /app/data/chromadb /app/data/models

ENV HF_HOME=/app/data/models
ENV TRANSFORMERS_CACHE=/app/data/models
ENV CHROMADB_PATH=/app/data/chromadb
ENV HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/rag/health || exit 1

CMD ["python", "-m", "uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
