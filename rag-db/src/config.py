"""
config.py
Centralized configuration — loads .env once and exposes all path / model / DB constants.

Usage:
    from config import (
        EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH,
        DB_PATH, DATA_PATH, MYSQL_CONFIG,
    )

.env discovery priority:
  1. ENV_FILE env var (Docker: /app/.env)
  2. ../.env relative to src/  (dev: rag-db/.env)
  3. ../../.env relative to src/ (dev: project-root/.env)
  4. /app/.env (Docker fallback)

All absolute paths are read from .env or environment variables.
Fallback defaults are HuggingFace model IDs for auto-download in Docker.
"""

import os
import sys
from dotenv import load_dotenv

# ============================================================
# .env  discovery (centralized — other modules import from here)
# ============================================================

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_and_load_dotenv():
    """Find .env in multiple possible locations, load first match."""
    candidates = [
        os.environ.get("ENV_FILE"),                        # explicit env var
        os.path.join(_SRC_DIR, "..", ".env"),              # rag-db/.env
        os.path.join(_SRC_DIR, "..", "..", ".env"),        # project-root/.env
        "/app/.env",                                        # Docker
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            load_dotenv(path)
            return path
    return None

_ENV_LOADED = _find_and_load_dotenv()

# ============================================================
#  Model paths (HuggingFace model IDs as defaults → auto-download)
# ============================================================
# Docker: set EMBEDDING_MODEL_PATH=BAAI/bge-m3 to auto-download from HF
# Local:  set EMBEDDING_MODEL_PATH=D:\...\bge-m3 to use cached copy

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "BAAI/bge-m3",  # HF model ID — SentenceTransformer will auto-download
)

RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    "BAAI/bge-reranker-v2-m3",  # HF model ID — CrossEncoder will auto-download
)

# ============================================================
#  ChromaDB  path
# ============================================================
# Docker: set CHROMADB_PATH=/app/data/chromadb
# Local:  defaults to project-root/medical_rag_db

_DEFAULT_CHROMADB = os.path.normpath(
    os.path.join(_SRC_DIR, "..", "..", "medical_rag_db")
)

CHROMADB_PATH = os.getenv("CHROMADB_PATH", _DEFAULT_CHROMADB)

# Legacy alias for code that references DB_PATH
DB_PATH = CHROMADB_PATH

# ============================================================
#  Data  path (medical.json)
# ============================================================
# Docker: mount data at /app/data/ and set DATA_DIR=/app/data

DATA_DIR = os.getenv(
    "DATA_DIR",
    os.path.join(_SRC_DIR, "..", "..", "rag data", "openkg data"),
)

DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(DATA_DIR, "medical.json"),
)

# ============================================================
#  MySQL  config
# ============================================================
# Docker: MYSQL_HOST=mysql  (docker-compose service name)
# Local:  MYSQL_HOST=localhost

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "medical_rag"),
    "charset": "utf8mb4",
}

# ============================================================
#  Convenience: get MySQL connection
# ============================================================

def get_mysql_connection(cursorclass=None):
    """Create a pymysql connection using MYSQL_CONFIG. Import pymysql on demand."""
    import pymysql
    kwargs = dict(MYSQL_CONFIG)
    if cursorclass is not None:
        kwargs["cursorclass"] = cursorclass
    return pymysql.connect(**kwargs)


def get_mysql_connection_dict_cursor():
    """Shortcut: connection with DictCursor."""
    import pymysql
    return get_mysql_connection(cursorclass=pymysql.cursors.DictCursor)


# ============================================================
#  Startup info
# ============================================================

if __name__ == "__main__":
    print(f"Config loaded:")
    print(f"  .env:        {_ENV_LOADED or 'not found (using env vars)'}")
    print(f"  EMBEDDING:   {EMBEDDING_MODEL_PATH}")
    print(f"  RERANKER:    {RERANKER_MODEL_PATH}")
    print(f"  CHROMADB:    {CHROMADB_PATH}")
    print(f"  DATA:        {DATA_PATH}")
    print(f"  MYSQL:       {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
