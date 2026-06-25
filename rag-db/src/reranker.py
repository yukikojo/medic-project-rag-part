"""
reranker.py
Cross-encoder reranker — second-stage retrieval refinement

Replaces simple cosine similarity scores with cross-encoder relevance scores
for much more accurate disease ranking.

Architecture position:
  ChromaDB cosine search (top_k=20) → Reranker.rerank_results() → top-5 re-ranked → LLM

Model: BAAI/bge-reranker-v2-m3 (cross-encoder, ~1.1GB)
  - Same family as BGE-M3 embedding model
  - Multilingual (Chinese + English, 100+ languages)
  - Takes (query, passage) pairs, outputs relevance scores

Usage:
    from reranker import Reranker

    reranker = Reranker()
    results = reranker.rerank_results(
        query="头痛发热咳嗽",
        disease_results=[...]  # from VectorStore.search_disease()
    )
    # results are re-scored and re-sorted by cross-encoder relevance
"""

import os
import time
from typing import Optional

from sentence_transformers import CrossEncoder

# ============================================================
# 配置
# ============================================================

from dotenv import load_dotenv as _load_dotenv

# Load .env from project root (rag-db/src/../../.env)
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

# BAAI/bge-reranker-v2-m3: cross-encoder based on XLM-RoBERTa
#   - Input: (query, passage) pairs
#   - Output: relevance score (higher = more relevant)
#   - Max length: 8192 tokens
#   - Size: ~1.1 GB
# Path configured via .env → RERANKER_MODEL_PATH
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    r"D:\floder-for-claude\medic\bge-reranker-v2-m3"
)


class Reranker:
    """
    Cross-encoder reranker for medical knowledge base retrieval.

    Loads a BGE-reranker model (cross-encoder) that takes (query, document)
    pairs and outputs fine-grained relevance scores, replacing the coarse
    cosine similarity from bi-encoder embeddings.

    Uses the same lazy-load pattern as VectorStore's embedding model —
    model is loaded on first use, not at construction time.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_fp16: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: Path to reranker model directory.
                        Defaults to D:\\floder-for-claude\\medic\\bge-reranker-v2-m3
            use_fp16: Use half-precision for faster inference and lower memory.
            verbose: Print loading progress.
        """
        self.model_path = model_path or RERANKER_MODEL_PATH
        self.use_fp16 = use_fp16
        self.verbose = verbose
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model on first access."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Reranker model not found at: {self.model_path}\n"
                    f"Please download BAAI/bge-reranker-v2-m3 to this path first.\n"
                    f"  Example: from sentence_transformers import CrossEncoder\n"
                    f"           model = CrossEncoder('BAAI/bge-reranker-v2-m3')\n"
                    f"           model.save('{self.model_path}')"
                )

            if self.verbose:
                print(f"Loading reranker model: {self.model_path} ...")

            start = time.time()

            # Load cross-encoder with explicit GPU + fp16 settings
            import torch as _torch

            kwargs = {
                "device": "cuda" if _torch.cuda.is_available() else "cpu",
            }
            if self.use_fp16 and _torch.cuda.is_available():
                # Use half precision for 2x faster inference + lower memory on GPU
                kwargs["model_kwargs"] = {"torch_dtype": _torch.float16}

            self._model = CrossEncoder(
                self.model_path,
                **kwargs,
            )

            elapsed = time.time() - start
            if self.verbose:
                print(f"Reranker model loaded, took {elapsed:.1f}s")

        return self._model

    # ================================================================
    # Core reranking methods
    # ================================================================

    def rerank(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 32,
    ) -> list[dict]:
        """
        Rerank a list of documents against a query using the cross-encoder.

        Args:
            query: The search query (e.g., user symptom description).
            documents: List of document texts to score.
            batch_size: Batch size for model.predict().

        Returns:
            List of dicts with {index, score, document}, sorted by score descending.
            Score is the raw cross-encoder relevance score (higher = more relevant).
        """
        if not documents:
            return []

        # Build (query, document) pairs
        pairs = [(query, doc) for doc in documents]

        # Cross-encoder scores
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=self.verbose,
        )

        # Build results
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append({
                "index": i,
                "score": float(score),
                "document": doc,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def rerank_results(
        self,
        query: str,
        disease_results: list[dict],
        normalize_scores: bool = True,
    ) -> list[dict]:
        """
        Re-rank disease search results using the cross-encoder.

        Takes the output of VectorStore.search_disease() and:
        1. Builds a search_text for each result (disease + symptoms + departments...)
        2. Runs cross-encoder on all (query, text) pairs
        3. Replaces the cosine 'score' with cross-encoder score
        4. Adds a 'cosine_score' field to preserve the original score
        5. Re-sorts by new score descending
        6. Updates the 'chain' reasoning field to note reranking

        Args:
            query: Original user symptom query.
            disease_results: List of result dicts from VectorStore.search_disease().
            normalize_scores: If True, normalize raw cross-encoder scores to 0-1 range
                             using sigmoid (since raw logits can be any range).

        Returns:
            Re-scored and re-sorted list (same dict structure, updated scores).
        """
        if not disease_results:
            return disease_results

        # Build document texts from disease results
        # Use the same structured format that was embedded
        documents = []
        for r in disease_results:
            doc_text = (
                f"疾病：{r['disease']}。"
                f"症状：{r['symptoms']}。"
                f"所属科室：{r['departments']}。"
                f"分类：{r.get('category', '')}。"
                f"简介：{r.get('desc', '')[:300]}"
            )
            documents.append(doc_text)

        # Run cross-encoder
        reranked = self.rerank(query, documents)

        # Map scores back to original result dicts
        score_map = {item["index"]: item["score"] for item in reranked}

        for i, r in enumerate(disease_results):
            raw_score = score_map.get(i, 0.0)

            if normalize_scores:
                # Apply sigmoid to convert logits → 0-1 probability range
                import math
                normalized = 1.0 / (1.0 + math.exp(-raw_score))
            else:
                normalized = raw_score

            # Preserve original cosine score
            r["cosine_score"] = r["score"]
            # Replace with cross-encoder score
            r["score"] = round(normalized, 4)

            # Update reasoning chain
            r["chain"] = f"{query} → {r['disease']} → {r['departments']} (reranked)"

        # Re-sort by new score descending
        disease_results.sort(key=lambda x: x["score"], reverse=True)

        return disease_results

    # ================================================================
    # Model info
    # ================================================================

    def get_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_path": self.model_path,
            "model_loaded": self._model is not None,
            "use_fp16": self.use_fp16,
        }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    print("=" * 60)
    print("  Reranker Quick Test")
    print("=" * 60)

    # Test 1: Basic rerank
    if os.path.exists(RERANKER_MODEL_PATH):
        reranker = Reranker(verbose=True)

        query = "头痛发热咳嗽流鼻涕"
        candidates = [
            "疾病：感冒。症状：头痛、发热、咳嗽、流鼻涕。所属科室：呼吸内科。分类：呼吸道感染。",
            "疾病：偏头痛。症状：头痛、恶心、畏光。所属科室：神经内科。分类：神经系统疾病。",
            "疾病：过敏性鼻炎。症状：流鼻涕、打喷嚏、鼻塞。所属科室：耳鼻喉科。分类：过敏性疾病。",
            "疾病：肺炎。症状：发热、咳嗽、咳痰、胸痛。所属科室：呼吸内科。分类：呼吸道感染。",
            "疾病：高血压。症状：头痛、头晕、心悸。所属科室：心内科。分类：心血管疾病。",
        ]

        results = reranker.rerank(query, candidates)

        print(f"\nQuery: {query}")
        print(f"Results (re-ranked by cross-encoder):")
        for i, r in enumerate(results):
            print(f"  {i+1}. score={r['score']:.4f} | {r['document'][:60]}...")
    else:
        print(f"\n[SKIP] Reranker model not found at: {RERANKER_MODEL_PATH}")
        print(f"  Download it first, or set RERANKER_MODEL_PATH to your local path.")
        print(f"  Model: BAAI/bge-reranker-v2-m3")
