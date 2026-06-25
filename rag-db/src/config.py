"""
config.py
Shared configuration — loads .env once and exposes all path constants.

Usage:
    from config import EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH

All absolute paths are read from the project-root .env file.
Fallback defaults preserved for backward compatibility.
"""

import os
from dotenv import load_dotenv

# Load .env from project root (rag-db/../.env)
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

# ============================================================
#  Model paths
# ============================================================

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    r"D:\floder-for-claude\medic\bge-m3",  # fallback default
)

RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    r"D:\floder-for-claude\medic\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
)

# ============================================================
#  Database paths (relative to project root)
# ============================================================

DB_PATH = os.path.join(_PROJECT_ROOT, "medical_rag_db")

DATA_PATH = os.path.join(_PROJECT_ROOT, "rag data", "openkg data", "medical.json")
