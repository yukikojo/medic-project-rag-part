"""
query_engine.py
RAG 向量知识库查询引擎 — VectorStore 抽象层

对外暴露统一的检索接口, 底层可切换 Chroma/Qdrant/Milvus。
Spring Boot 后端通过 REST API 或子进程调用本模块的 search 方法。

使用示例:
    from retrieval.query_engine import VectorStore

    store = VectorStore()  # 默认 Chroma backend

    # 科室推荐: 症状 → 疾病 → 科室
    results = store.search_disease("头痛发热咳嗽")
    for r in results:
        print(f"{r['disease']} → {r['departments']} (相似度: {r['score']:.2%})")

    # 症状→科室直接映射
    results = store.search_by_symptom("肚子疼拉肚子")

    # 科室信息检索
    results = store.search_department("神经内科")
"""

import os
import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv as _load_dotenv

# Load .env from project root
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

# ============================================================
# 配置
# ============================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "medical_rag_db")

# Embedding model — local path, configured via .env → EMBEDDING_MODEL_PATH
# BAAI/bge-m3: 1024-dim, multilingual (CN/EN 100+), MTR hybrid retrieval, 8192 tokens, ~2.2GB
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_PATH",
    r"D:\floder-for-claude\medic\bge-m3"
)

# Reranker config — configured via .env → RERANKER_MODEL_PATH
# BAAI/bge-reranker-v2-m3: cross-encoder, multilingual, ~1.1GB
RERANKER_MODEL_PATH = os.getenv(
    "RERANKER_MODEL_PATH",
    r"D:\floder-for-claude\medic\bge-reranker-v2-m3"
)
RERANKER_FETCH_K = 20       # Number of candidates to fetch before reranking


class VectorStore:
    """
    RAG 向量知识库抽象层
    封装 ChromaDB 操作, 对外暴露 search/add 接口。
    切换底层数据库只需修改 backend 参数。
    """

    def __init__(
        self,
        backend: str = "chroma",
        db_path: Optional[str] = None,
        use_reranker: bool = False,
        reranker_model_path: Optional[str] = None,
    ):
        """
        Args:
            backend: 向量数据库后端, 可选 "chroma" / "qdrant" (预留)
            db_path: 数据库持久化路径, 默认 medical_rag_db/
            use_reranker: 是否启用 cross-encoder reranker (默认 False)
            reranker_model_path: Reranker 模型路径, 默认使用 RERANKER_MODEL_PATH
        """
        self.backend = backend
        self.db_path = db_path or DB_PATH

        if backend == "chroma":
            self.client = chromadb.PersistentClient(path=self.db_path)
        elif backend == "qdrant":
            raise NotImplementedError("Qdrant backend 尚未实现, 请使用 chroma")
        else:
            raise ValueError(f"不支持的 backend: {backend}")

        # 延迟加载模型 (首次查询时才加载)
        self._model = None
        self._collections = {}

        # Reranker (lazy-loaded)
        self._use_reranker = use_reranker
        self._reranker_model_path = reranker_model_path or RERANKER_MODEL_PATH
        self._reranker = None

    @property
    def model(self):
        """延迟加载嵌入模型"""
        if self._model is None:
            import torch as _torch

            print(f"加载嵌入模型: {EMBEDDING_MODEL} ...")
            start = time.time()

            # Explicit GPU device + fp16 for faster inference
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            model_kwargs = {}
            if device == "cuda":
                model_kwargs = {"torch_dtype": _torch.float16}

            self._model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device,
                model_kwargs=model_kwargs,
            )
            print(f"模型加载完成, 设备={device}, 耗时 {time.time() - start:.1f}s")
        return self._model

    @property
    def reranker(self):
        """延迟加载 reranker 模型"""
        if self._use_reranker and self._reranker is None:
            import importlib.util as _iu
            _rr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reranker", "reranker.py")
            _rr_spec = _iu.spec_from_file_location("reranker.reranker", _rr_path)
            _rr = _iu.module_from_spec(_rr_spec)
            _rr_spec.loader.exec_module(_rr)
            self._reranker = _rr.Reranker(
                model_path=self._reranker_model_path,
                verbose=True,
            )
        return self._reranker

    def get_collection(self, name: str):
        """获取或缓存 Collection"""
        if name not in self._collections:
            self._collections[name] = self.client.get_collection(name)
        return self._collections[name]

    # ================================================================
    # 查询接口
    # ================================================================

    def search_disease(self, query: str, top_k: int = 5) -> list[dict]:
        """
        语义检索疾病知识库 — 核心查询方法

        用途: 用户描述症状 → 检索最匹配的疾病 → 获取推荐科室
        这在需求文档中对应:
          - UC-AI-01 症状结构化分析 (需要症状知识库)
          - UC-AI-02 科室智能推荐 (需要症状→疾病→科室两跳推理)

        Args:
            query: 用户输入的症状描述, 如 "头痛发热咳嗽流鼻涕"
            top_k: 返回最大匹配数

        Returns:
            [
                {
                    "disease": "感冒",
                    "symptoms": "鼻塞, 流涕, 喷嚏, 咳嗽, 咽痛, 发热, 头痛",
                    "departments": "内科, 呼吸内科",
                    "category": "疾病百科, 内科, 呼吸内科",
                    "drugs": "复方氨酚烷胺片, 感冒灵颗粒",
                    "score": 0.952,  # 余弦相似度
                    "chain": "头痛发热 → 感冒 → 内科, 呼吸内科"  # 推理链
                },
                ...
            ]
        """
        collection = self.get_collection("disease_knowledge")
        query_embedding = self.model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance  # cosine distance → similarity

            output.append({
                "disease": meta.get("disease", ""),
                "symptoms": meta.get("symptoms", ""),
                "departments": meta.get("departments", ""),
                "category": meta.get("category", ""),
                "drugs": meta.get("drugs", ""),
                "desc": meta.get("desc", ""),
                "score": score,
                "chain": f"{query} → {meta.get('disease', '')} → {meta.get('departments', '')}",
            })

        return output

    def search_by_symptom(self, query: str, top_k: int = 3) -> list[dict]:
        """
        症状→科室直接映射检索

        用途: 用户输入症状关键词, 直接获取推荐的科室
        与 search_disease 的区别: 跳过"症状→疾病"这一步, 直接返回科室

        Args:
            query: 症状描述, 如 "肚子疼拉肚子"
            top_k: 返回最大匹配数

        Returns:
            [
                {
                    "symptom": "腹痛",
                    "departments": "消化内科, 普外科",
                    "disease_count": 672,
                    "score": 0.93,
                },
                ...
            ]
        """
        collection = self.get_collection("symptom_dept_direct")
        query_embedding = self.model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance

            output.append({
                "symptom": meta.get("symptom", ""),
                "departments": meta.get("departments", ""),
                "disease_count": meta.get("disease_count", 0),
                "score": score,
            })

        return output

    def search_department(self, query: str, top_k: int = 3) -> list[dict]:
        """
        科室信息检索

        用途: 根据科室名或症状, 检索科室的诊疗范围和常见症状
        对应 UC-A-04 管理医疗知识库 (管理员需要了解科室覆盖范围)

        Args:
            query: 科室名或症状, 如 "神经内科" 或 "骨头疼"

        Returns:
            [
                {
                    "department": "骨科",
                    "disease_count": 795,
                    "common_symptoms": "骨折, 关节痛, 腰背痛, ...",
                    "sample_diseases": "骨折, 颈椎病, 腰椎间盘突出, ...",
                    "score": 0.95,
                },
                ...
            ]
        """
        collection = self.get_collection("department_info")
        query_embedding = self.model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance

            output.append({
                "department": meta.get("department", ""),
                "disease_count": meta.get("disease_count", 0),
                "common_symptoms": meta.get("common_symptoms", ""),
                "sample_diseases": meta.get("sample_diseases", ""),
                "score": score,
            })

        return output

    def comprehensive_search(self, query: str, top_k: int = 5, use_reranker: bool = True, enrich_kg: bool = False) -> dict:
        """
        综合检索: 同时查询三个 Collection, 返回完整推理链

        当 reranker 启用时, 使用两阶段检索:
          第一阶段: ChromaDB 粗检索 (top_k * 4 候选)
          第二阶段: Cross-encoder 精排 (重打分 + 重排序, 取 top_k)

        当 enrich_kg 启用时, 每个疾病结果会补充知识图谱富化数据:
          推荐药品、常用药品、推荐食谱、宜吃/忌吃食物、建议检查、并发症、治疗方法

        这是对外暴露的主要接口, 供 API 调用。
        Spring Boot → POST /api/rag/search {"query": "头痛发热"}

        Args:
            query: 用户症状描述
            top_k: 最终返回的疾病结果数 (默认 5)
            use_reranker: 是否使用 reranker 精排 (需 VectorStore 初始化时启用)
            enrich_kg: 是否启用知识图谱富化 (补充用药/食谱/检查/并发症等)

        Returns:
            {
                "query": "头痛发热",
                "disease_results": [...],          # 疾病推荐 (含科室, 可能含 kg_enrichment)
                "symptom_direct": [...],           # 症状直接映射
                "all_departments": [...],          # 科室汇总
                "primary_recommendation": {...},   # 主推荐 (综合排序)
                "reranked": true/false,            # 是否使用了 reranker
                "kg_summary": {...},               # KG 富化汇总 (仅 enrich_kg=True)
            }
        """
        # Determine fetch size: more candidates if reranker is available
        reranker_available = self._use_reranker and use_reranker
        fetch_k = max(top_k * 4, RERANKER_FETCH_K) if reranker_available else top_k

        disease_results = self.search_disease(query, top_k=fetch_k)
        symptom_results = self.search_by_symptom(query, top_k=3)

        # Second-stage: cross-encoder reranking
        reranked = False
        if reranker_available and self.reranker and disease_results:
            disease_results = self.reranker.rerank_results(query, disease_results)
            disease_results = disease_results[:top_k]  # Cap to requested top_k
            reranked = True

        # 收集所有推荐的科室
        all_depts = set()
        for r in disease_results:
            for d in r["departments"].split(", "):
                if d:
                    all_depts.add(d)

        # 构建综合推荐
        primary = None
        if disease_results and disease_results[0]["score"] > 0.7:
            r = disease_results[0]
            primary = {
                "department": r["departments"].split(", ")[0] if r["departments"] else r["disease"],
                "disease": r["disease"],
                "confidence": r["score"],
                "reasoning": r["chain"],
            }

        result = {
            "query": query,
            "disease_results": disease_results[:3],
            "symptom_direct": symptom_results,
            "all_departments": list(all_depts)[:10],
            "primary_recommendation": primary,
            "reranked": reranked,
        }

        # Optional: KG enrichment (drugs, foods, checks, complications, cures)
        if enrich_kg and disease_results:
            try:
                from enrichment.kg_enricher import get_enricher
                enricher = get_enricher()
                enriched = enricher.enrich_comprehensive(
                    result, max_drugs=5, max_foods=5,
                )
                return enriched
            except Exception:
                # KG enrichment failed silently — return unenriched results
                pass

        return result

    # ================================================================
    # 写入接口 (增量更新)
    # ================================================================

    def add_diseases(self, documents: list[str], metadatas: list[dict]) -> None:
        """增量添加疾病记录"""
        collection = self.get_collection("disease_knowledge")
        existing_count = collection.count()
        embeddings = self.model.encode(documents).tolist()
        ids = [f"disease_{existing_count + i:04d}" for i in range(len(documents))]
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print(f"已添加 {len(documents)} 条疾病记录")

    def add_symptoms(self, documents: list[str], metadatas: list[dict]) -> None:
        """增量添加症状-科室映射"""
        collection = self.get_collection("symptom_dept_direct")
        existing_count = collection.count()
        embeddings = self.model.encode(documents).tolist()
        ids = [f"sym_{existing_count + i:04d}" for i in range(len(documents))]
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
        print(f"已添加 {len(documents)} 条症状映射")

    # ================================================================
    # 辅助接口
    # ================================================================

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        return {
            "backend": self.backend,
            "db_path": self.db_path,
            "collections": {
                "disease_knowledge": self.get_collection("disease_knowledge").count(),
                "symptom_dept_direct": self.get_collection("symptom_dept_direct").count(),
                "department_info": self.get_collection("department_info").count(),
            },
        }


# ============================================================
# 命令行快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  RAG 医疗知识库 — 查询测试")
    print("=" * 60)

    store = VectorStore()

    # 显示统计
    stats = store.get_stats()
    print(f"\n知识库统计: {stats['collections']}")

    # 测试查询列表
    test_queries = [
        "头痛发热咳嗽流鼻涕",
        "腹痛腹泻拉肚子恶心",
        "胸闷心慌气短胸痛",
        "皮肤痒红肿过敏湿疹",
        "腰疼关节疼腿麻",
    ]

    for query in test_queries:
        print(f"\n{'─' * 50}")
        result = store.comprehensive_search(query, top_k=5)
        primary = result.get("primary_recommendation")
        if primary:
            print(f">> 查询: {query}")
            print(f"  主推荐: {primary['department']}")
            print(f"  推理链: {primary['reasoning']}")
            print(f"  置信度: {primary['confidence']:.1%}")
            print(f"  Top-5 疾病:")
            for r in result["disease_results"]:
                print(f"    {r['score']:.1%} | {r['disease']} → {r['departments']}")
        else:
            print(f">> 查询: {query} → 未找到高置信度匹配")
