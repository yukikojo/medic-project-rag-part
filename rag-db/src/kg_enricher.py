"""
kg_enricher.py
知识图谱富化引擎 — 为 RAG 检索结果补充用药/食谱/检查/并发症等关联数据

从 OpenKG 疾病知识图谱的 relations.json (312,159 条关系) 中按疾病名
检索 8 类关联信息，将富化后的数据以结构化 JSON 返回给 Java 后端。

关系类型 (12 种, 仅 Disease 出发的被索引):
  1. recommand_drug  — 推荐药品 (59,465 条)
  2. common_drug    — 常用药品 (14,647 条)
  3. recommand_eat  — 推荐食谱 (40,221 条)
  4. do_eat         — 宜吃食物 (22,230 条)
  5. no_eat         — 忌吃食物 (22,239 条)
  6. need_check     — 所需检查 (39,418 条)
  7. acompany_with  — 并发症   (12,024 条)
  8. cure_way       — 治疗方法 (21,047 条)

架构位置:
  VectorStore.search_disease() 返回疾病列表
    → KGEnricher.enrich_results() 逐病检索 KG → 补充用药/食物/检查/并发症
    → 返回富化后的 JSON 给 Java 后端

使用示例:
    from kg_enricher import KGEnricher

    enricher = KGEnricher()
    # 单个疾病
    info = enricher.enrich_disease("感冒")
    # {"drugs": ["复方氨酚烷胺片", ...], "foods": {"recommend": [...], ...}, ...}

    # 批量
    results = enricher.enrich_results(rag_disease_results, top_drugs=5, top_foods=5)
"""

import os
import json
from typing import Optional
from collections import defaultdict


# ============================================================
# 配置
# ============================================================

_RELATIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "rag data", "openkg data", "relations.json"
)

# Disease-starting relation types that we want to index
# rel_type → output category
INDEXED_RELATIONS = {
    "recommand_drug": "recommand_drugs",     # 推荐药品
    "common_drug":    "common_drugs",        # 常用药品
    "recommand_eat":  "recommand_foods",     # 推荐食谱
    "do_eat":         "do_eat_foods",        # 宜吃食物
    "no_eat":         "no_eat_foods",        # 忌吃食物
    "need_check":     "need_checks",         # 所需检查
    "acompany_with":  "complications",       # 并发症
    "cure_way":       "cure_ways",           # 治疗方法
    "has_symptom":    "symptoms",            # 症状 (补充)
    "belongs_to":     "departments_raw",      # 所属科室 (KG原始)
}

# Human-readable labels for output
CATEGORY_LABELS = {
    "recommand_drugs":  "推荐药品",
    "common_drugs":     "常用药品",
    "recommand_foods":  "推荐食谱",
    "do_eat_foods":     "宜吃食物",
    "no_eat_foods":     "忌吃食物",
    "need_checks":      "建议检查项目",
    "complications":    "可能并发症",
    "cure_ways":        "治疗方法",
    "symptoms":         "相关症状",
    "departments_raw":  "所属科室(KG)",
}


class KGEnricher:
    """
    知识图谱富化器 — 为疾病检索结果补充关联知识。

    延迟加载 relations.json (首次使用时约 2-3 秒, ~120MB JSON),
    之后全部 O(1) 哈希查找。
    """

    def __init__(self, relations_path: Optional[str] = None, use_mysql: bool = True, verbose: bool = False):
        """
        Args:
            relations_path: relations.json 路径，默认自动定位 (MySQL不可用时使用)
            use_mysql: 是否优先从 MySQL rag_disease_kg 表读取 (True=MySQL, False=JSON)
            verbose: 是否打印加载日志
        """
        self.relations_path = relations_path or _RELATIONS_PATH
        self.use_mysql = use_mysql
        self.verbose = verbose

        # Lazy-loaded sources
        self._index: Optional[dict] = None        # JSON fallback index
        self._stats: Optional[dict] = None
        self._disease_count: int = 0
        self._mysql_available: Optional[bool] = None  # None=untested, True/False
        self._mysql_conn = None

    # ================================================================
    # 索引构建 (lazy)
    # ================================================================

    @property
    def index(self) -> dict:
        """Lazy-load and build disease lookup index."""
        if self._index is None:
            self._build_index()
        return self._index

    def _build_index(self):
        """
        读取 relations.json 并构建 disease_name → {category: [values]} 索引。

        仅索引 start_entity_type == "Disease" 的关系。
        跳过 drug/department/producer 等非疾病出发的关系。
        """
        if not os.path.exists(self.relations_path):
            raise FileNotFoundError(
                f"relations.json 未找到: {self.relations_path}\n"
                f"请确保 OpenKG 数据已放置在正确路径。"
            )

        if self.verbose:
            import time
            start = time.time()
            print(f"[KGEnricher] 加载知识图谱: {self.relations_path} ...")

        with open(self.relations_path, "r", encoding="utf-8") as f:
            groups = json.load(f)

        # disease_name → {category_key: set(values)}
        _raw = defaultdict(lambda: defaultdict(set))
        skipped_groups = 0
        total_skipped = 0

        for group in groups:
            rel_type = group.get("rel_type", "")
            start_type = group.get("start_entity_type", "")

            # Only index Disease-starting relations
            if start_type != "Disease":
                skipped_groups += 1
                total_skipped += len(group.get("rels", []))
                continue

            category = INDEXED_RELATIONS.get(rel_type)
            if category is None:
                total_skipped += len(group.get("rels", []))
                continue

            for rel in group.get("rels", []):
                disease = rel.get("start_entity_name", "")
                target = rel.get("end_entity_name", "")
                if disease and target:
                    _raw[disease][category].add(target)

        # Convert sets → sorted lists (for deterministic output)
        self._index = {}
        disease_set = set()
        for disease, categories in _raw.items():
            disease_set.add(disease)
            self._index[disease] = {
                cat: sorted(list(items))
                for cat, items in categories.items()
            }

        self._disease_count = len(disease_set)

        # Stats
        total_indexed = sum(
            sum(len(items) for items in categories.values())
            for categories in self._index.values()
        )

        self._stats = {
            "diseases_indexed": self._disease_count,
            "total_relations_indexed": total_indexed,
            "total_relations_raw": sum(len(g["rels"]) for g in groups),
            "skipped_groups": skipped_groups,
            "skipped_relations": total_skipped,
            "index_categories": list(INDEXED_RELATIONS.values()),
        }

        if self.verbose:
            import time
            elapsed = time.time() - start
            print(f"[KGEnricher] 索引构建完成, 耗时 {elapsed:.1f}s")
            print(f"  疾病数: {self._disease_count}")
            print(f"  索引关系: {total_indexed}")
            print(f"  跳过关系: {total_skipped}")

    # ================================================================
    # 查询接口
    # ================================================================

    def _get_mysql_conn(self):
        """Get or create MySQL connection (lazy)."""
        if self._mysql_conn is None:
            import pymysql
            from dotenv import load_dotenv as _ld
            _ld(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
            self._mysql_conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "medical_rag"),
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
            )
        else:
            try:
                self._mysql_conn.ping(reconnect=True)
            except Exception:
                import pymysql
                self._mysql_conn = pymysql.connect(
                    host=os.getenv("MYSQL_HOST", "localhost"),
                    port=int(os.getenv("MYSQL_PORT", "3306")),
                    user=os.getenv("MYSQL_USER", "root"),
                    password=os.getenv("MYSQL_PASSWORD", ""),
                    database=os.getenv("MYSQL_DATABASE", "medical_rag"),
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=True,
                )
        return self._mysql_conn

    def _enrich_from_mysql(self, disease_name: str) -> Optional[dict]:
        """
        Query rag_disease_kg table for enrichment data.

        Returns None if MySQL unavailable or no data found.
        """
        if self._mysql_available is False:
            return None

        SQL = """
            SELECT rel_category, rel_value
            FROM rag_disease_kg
            WHERE disease_name = %s AND status = 1
            ORDER BY rel_category
        """
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as cursor:
                cursor.execute(SQL, (disease_name,))
                rows = cursor.fetchall()

            if self._mysql_available is None:
                self._mysql_available = True
                if self.verbose:
                    print(f"[KGEnricher] MySQL rag_disease_kg 已就绪")

            if not rows:
                return None  # No data found — will try fuzzy or fallback

            # Group by category
            from collections import defaultdict
            grouped = defaultdict(list)
            for r in rows:
                grouped[r["rel_category"]].append(r["rel_value"])

            return {
                "recommand_drugs":  sorted(grouped.get("recommand_drug", [])),
                "common_drugs":     sorted(grouped.get("common_drug", [])),
                "recommand_foods":  sorted(grouped.get("recommand_food", [])),
                "do_eat_foods":     sorted(grouped.get("do_eat_food", [])),
                "no_eat_foods":     sorted(grouped.get("no_eat_food", [])),
                "need_checks":      sorted(grouped.get("need_check", [])),
                "complications":    sorted(grouped.get("complication", [])),
                "cure_ways":        sorted(grouped.get("cure_way", [])),
            }

        except Exception as e:
            self._mysql_available = False
            if self.verbose:
                print(f"[KGEnricher] MySQL 不可用: {e}，回退到 JSON 索引")
            return None

    def enrich_disease(
        self,
        disease_name: str,
        max_drugs: int = 10,
        max_foods: int = 10,
        max_checks: int = 10,
        max_complications: int = 5,
        max_cures: int = 5,
    ) -> dict:
        """
        查询单个疾病的全量知识图谱关联数据。

        Args:
            disease_name: 疾病名称 (如 "感冒")
            max_drugs: 最多返回的推荐药品数
            max_foods: 最多返回的食谱数
            max_checks: 最多返回的检查项目数
            max_complications: 最多返回的并发症数
            max_cures: 最多返回的治疗方法数

        Returns:
            {
                "disease": "感冒",
                "found": true,
                "drugs": {
                    "recommand": ["复方氨酚烷胺片", ...],   # 推荐药品
                    "common":    ["感冒灵颗粒", ...],       # 常用药品
                },
                "foods": {
                    "recommand": ["白菜肉末粥", ...],       # 推荐食谱
                    "do_eat":    ["苹果", "梨", ...],       # 宜吃
                    "no_eat":    ["辣椒", "油炸食品", ...], # 忌吃
                },
                "checks": ["血常规", "胸部X光", ...],        # 建议检查
                "complications": ["肺炎", "支气管炎", ...],   # 并发症
                "cures": ["休息", "多饮水", "对症治疗", ...], # 治疗方法
                "symptoms_from_kg": ["咳嗽", "发热", ...],    # KG 中的症状
            }
        """
        # --- Try MySQL first ---
        disease_data = None
        source = "none"

        if self.use_mysql:
            disease_data = self._enrich_from_mysql(disease_name)
            if disease_data is not None:
                source = "mysql"

        # --- Fallback: JSON index ---
        if disease_data is None:
            disease_data = self.index.get(disease_name)
            if disease_data is not None:
                source = "json"

        if not disease_data:
            # Fuzzy search — try to find the closest match
            match = None
            if self.use_mysql and self._mysql_available:
                match = self._fuzzy_match_mysql(disease_name)
            if not match:
                match = self._fuzzy_match_json(disease_name)
            return {
                "disease": disease_name,
                "found": False,
                "suggestion": match,
                "source": source,
                "drugs": {},
                "foods": {},
                "checks": [],
                "complications": [],
                "cures": [],
                "symptoms_from_kg": [],
            }

        # Build structured response
        drugs = {}
        if "recommand_drugs" in disease_data:
            drugs["recommand"] = disease_data["recommand_drugs"][:max_drugs]
        if "common_drugs" in disease_data:
            drugs["common"] = disease_data["common_drugs"][:max_drugs]

        foods = {}
        if "recommand_foods" in disease_data:
            foods["recommand"] = disease_data["recommand_foods"][:max_foods]
        if "do_eat_foods" in disease_data:
            foods["do_eat"] = disease_data["do_eat_foods"][:max_foods]
        if "no_eat_foods" in disease_data:
            foods["no_eat"] = disease_data["no_eat_foods"][:max_foods]

        return {
            "disease": disease_name,
            "found": True,
            "source": source,
            "drugs": drugs,
            "foods": foods,
            "checks": disease_data.get("need_checks", [])[:max_checks],
            "complications": disease_data.get("complications", [])[:max_complications],
            "cures": disease_data.get("cure_ways", [])[:max_cures],
            "symptoms_from_kg": disease_data.get("symptoms", []),
        }

    def enrich_results(
        self,
        disease_results: list[dict],
        max_drugs: int = 5,
        max_foods: int = 5,
        max_checks: int = 5,
    ) -> list[dict]:
        """
        为 RAG 检索结果列表批量富化 KG 数据。

        在现有 search_disease() 返回的基础上，为每条结果增加
        `kg_enrichment` 字段，包含用药/食谱/检查/并发症。

        Args:
            disease_results: VectorStore.search_disease() 的返回结果
            max_drugs: 每病最多药品数
            max_foods: 每病最多食谱数
            max_checks: 每病最多检查数

        Returns:
            增强后的 list[dict], 每条增加 `kg_enrichment` 字段
        """
        enriched = []
        for r in disease_results:
            disease_name = r.get("disease", "")
            kg = self.enrich_disease(
                disease_name,
                max_drugs=max_drugs,
                max_foods=max_foods,
                max_checks=max_checks,
            )
            r_enriched = dict(r)
            r_enriched["kg_enrichment"] = kg
            enriched.append(r_enriched)
        return enriched

    def enrich_comprehensive(
        self,
        comprehensive_result: dict,
        max_drugs: int = 5,
        max_foods: int = 5,
    ) -> dict:
        """
        富化 comprehensive_search() 的完整结果。

        为 disease_results 中的每条疾病补充 KG 数据,
        同时在顶层增加 `kg_summary` 汇总推荐。

        Args:
            comprehensive_result: VectorStore.comprehensive_search() 的返回值
            max_drugs: 每病最多药品数
            max_foods: 每病最多食谱数

        Returns:
            增强后的 dict, 增加 `kg_enrichment` 和 `kg_summary`
        """
        disease_results = comprehensive_result.get("disease_results", [])

        # Enrich each disease
        enriched_diseases = self.enrich_results(
            disease_results,
            max_drugs=max_drugs,
            max_foods=max_foods,
        )

        # Build summary: aggregate top recommendations across all results
        all_drugs = []
        all_foods = []
        all_checks = []
        seen_drugs = set()
        seen_foods = set()
        seen_checks = set()

        for d in enriched_diseases:
            kg = d.get("kg_enrichment", {})
            if not kg.get("found"):
                continue
            for drug in kg.get("drugs", {}).get("recommand", []):
                if drug not in seen_drugs:
                    all_drugs.append(drug)
                    seen_drugs.add(drug)
            for food in kg.get("foods", {}).get("recommand", []):
                if food not in seen_foods:
                    all_foods.append(food)
                    seen_foods.add(food)
            for check in kg.get("checks", []):
                if check not in seen_checks:
                    all_checks.append(check)
                    seen_checks.add(check)

        kg_summary = {
            "aggregated_recommand_drugs": all_drugs[:10],
            "aggregated_recommand_foods": all_foods[:10],
            "aggregated_checks": all_checks[:10],
            "diseases_enriched": sum(
                1 for d in enriched_diseases
                if d.get("kg_enrichment", {}).get("found")
            ),
        }

        result = dict(comprehensive_result)
        result["disease_results"] = enriched_diseases
        result["kg_summary"] = kg_summary
        return result

    # ================================================================
    # 模糊匹配
    # ================================================================

    def _fuzzy_match_json(self, disease_name: str) -> Optional[str]:
        """JSON index 模糊匹配 (fallback)."""
        if not self._index:
            return None
        for key in self._index:
            if disease_name in key or key in disease_name:
                return key
        return None

    def _fuzzy_match_mysql(self, disease_name: str) -> Optional[str]:
        """MySQL 模糊匹配: LIKE 子串查询."""
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as cursor:
                # Exact substring match
                cursor.execute(
                    "SELECT disease_name FROM rag_disease_kg WHERE disease_name LIKE %s AND status = 1 LIMIT 1",
                    (f"%{disease_name}%",)
                )
                row = cursor.fetchone()
                if row:
                    return row["disease_name"]
                # Reverse: disease_name contains query
                cursor.execute(
                    "SELECT disease_name FROM rag_disease_kg WHERE %s LIKE CONCAT('%%', disease_name, '%%') AND status = 1 LIMIT 1",
                    (disease_name,)
                )
                row = cursor.fetchone()
                if row:
                    return row["disease_name"]
        except Exception:
            pass
        return None

    # ================================================================
    # 元信息
    # ================================================================

    def get_stats(self) -> dict:
        """获取索引统计信息 (不触发构建)。"""
        if self._stats is None and os.path.exists(self.relations_path):
            _ = self.index  # trigger build
        return self._stats or {"status": "not_loaded"}

    def get_disease_count(self) -> int:
        """获取已索引的疾病数。"""
        if self._index is None:
            _ = self.index
        return self._disease_count

    def is_loaded(self) -> bool:
        """检查索引是否已加载。"""
        return self._index is not None

    def preload(self):
        """预加载索引 (在服务启动时调用, 避免首次请求等待)。"""
        _ = self.index
        return self._stats


# ============================================================
# 全局单例 (供 API 层使用)
# ============================================================

_global_enricher: Optional[KGEnricher] = None


def get_enricher(relations_path: Optional[str] = None, verbose: bool = False) -> KGEnricher:
    """获取全局 KGEnricher 单例。"""
    global _global_enricher
    if _global_enricher is None:
        _global_enricher = KGEnricher(relations_path=relations_path, verbose=verbose)
    return _global_enricher


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  KG Enricher — 知识图谱富化测试")
    print("=" * 65)

    enricher = KGEnricher(verbose=True)

    # Stats
    stats = enricher.get_stats()
    print(f"\n  索引统计:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Test 1: Single disease enrichment
    print("\n─── Test 1: 感冒 ───")
    info = enricher.enrich_disease("感冒", max_drugs=5, max_foods=5, max_checks=5)

    if info["found"]:
        print(f"  ✓ 疾病: {info['disease']}")
        print(f"  推荐药品: {info['drugs'].get('recommand', [])[:5]}")
        print(f"  常用药品: {info['drugs'].get('common', [])[:5]}")
        print(f"  推荐食谱: {info['foods'].get('recommand', [])[:5]}")
        print(f"  宜吃:     {info['foods'].get('do_eat', [])[:3]}")
        print(f"  忌吃:     {info['foods'].get('no_eat', [])[:3]}")
        print(f"  建议检查: {info['checks'][:5]}")
        print(f"  并发症:   {info['complications'][:5]}")
        print(f"  治疗方法: {info['cures'][:5]}")
    else:
        print(f"  ✗ 未找到: {info.get('suggestion', '无建议')}")

    # Test 2: Another disease
    print("\n─── Test 2: 糖尿病 ───")
    info2 = enricher.enrich_disease("糖尿病", max_drugs=3, max_foods=3)
    if info2["found"]:
        print(f"  ✓ 推荐药品: {info2['drugs'].get('recommand', [])[:3]}")
        print(f"  ✓ 推荐食谱: {info2['foods'].get('recommand', [])[:3]}")
        print(f"  ✓ 忌吃:     {info2['foods'].get('no_eat', [])[:3]}")

    # Test 3: Fuzzy match
    print("\n─── Test 3: 模糊匹配 '急性肠胃炎' ───")
    info3 = enricher.enrich_disease("急性肠胃炎")
    if info3["found"]:
        print(f"  ✓ 精确匹配成功: {info3['disease']}")
    else:
        print(f"  ✗ 精确匹配失败, 建议: {info3.get('suggestion')}")

    # Test 4: Enrich RAG results (simulated)
    print("\n─── Test 4: 批量富化 (模拟 RAG 结果) ───")
    mock_results = [
        {"disease": "感冒", "score": 0.85, "departments": "呼吸内科"},
        {"disease": "流行性感冒", "score": 0.78, "departments": "呼吸内科"},
        {"disease": "过敏性鼻炎", "score": 0.72, "departments": "耳鼻喉科"},
    ]
    enriched = enricher.enrich_results(mock_results, max_drugs=3, max_foods=3)
    for i, r in enumerate(enriched):
        kg = r.get("kg_enrichment", {})
        drugs = kg.get("drugs", {}).get("recommand", [])[:3]
        print(f"  {i+1}. {r['disease']} (score={r['score']:.1%})")
        print(f"     药品: {drugs}")
        print(f"     食谱: {kg.get('foods', {}).get('recommand', [])[:3]}")
