"""
download_reranker.py
One-click download script for the BGE-Reranker model.

Downloads BAAI/bge-reranker-v2-m3 to the local D: drive path
so it can be loaded offline (same pattern as the embedding model).

Usage:
    python src/download_reranker.py
"""

import os
import sys

# Add src to path for consistent imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from project root
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# Download target — configured via .env → RERANKER_MODEL_PATH
SAVE_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    r"BAAI/bge-reranker-v2-m3"
)


def main():
    print("=" * 60)
    print("  BGE-Reranker Model Download")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Save to: {SAVE_PATH}")
    print()

    # Check if already downloaded
    if os.path.exists(SAVE_PATH) and os.path.isdir(SAVE_PATH):
        files = os.listdir(SAVE_PATH)
        if any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            print(f"[OK] Model already exists at: {SAVE_PATH}")
            print(f"  Files: {len(files)} items")
            return

    print("Downloading model (this will take a few minutes on first run)...")
    print("The model is ~1.1 GB. Please wait.\n")

    try:
        from sentence_transformers import CrossEncoder

        # Download and load the model
        model = CrossEncoder(MODEL_NAME)
        # Save to local path
        model.save(SAVE_PATH)
        print(f"\n[OK] Model saved to: {SAVE_PATH}")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nManual download options:")
        print(f"  1. pip install sentence-transformers")
        print(f"  2. python -c \"from sentence_transformers import CrossEncoder;")
        print(f"     m = CrossEncoder('{MODEL_NAME}'); m.save('{SAVE_PATH}')\"")
        print(f"  3. Or download from HuggingFace: https://huggingface.co/{MODEL_NAME}")
        sys.exit(1)


if __name__ == "__main__":
    main()
