"""
test_kg_enrich.py
知识图谱富化 — 端到端测试

输入一个疾病名，返回该疾病在向量数据库 + 知识图谱中的完整关联数据：
  - 疾病基本信息 (症状 / 科室 / 分类 / 简介)
  - 推荐药品 + 常用药品
  - 推荐食谱 + 宜吃食物 + 忌吃食物
  - 建议检查项目
  - 可能并发症
  - 治疗方法
  - KG 原始症状

运行方式:
  cd "d:/medic project"
  python rag-db/tests/test_kg_enrich.py                          # 交互模式
  python rag-db/tests/test_kg_enrich.py 感冒                     # 指定疾病
  python rag-db/tests/test_kg_enrich.py 感冒 糖尿病 高血压       # 批量
  python rag-db/tests/test_kg_enrich.py --all --limit 5          # 随机5个
  python rag-db/tests/test_kg_enrich.py --json 感冒              # JSON 输出
"""

import os
import sys
import json
import time
import argparse
import importlib.util

# Path setup
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _src)


def load_module(name: str):
    """Load a module by file path from src/."""
    path = os.path.join(_src, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Test runner
# ============================================================

class KGDiseaseLookup:
    """单疾病全量知识查询器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._vector_store = None
        self._enricher = None

    @property
    def vector_store(self):
        if self._vector_store is None:
            qe = load_module("query_engine")
            self._vector_store = qe.VectorStore()
        return self._vector_store

    @property
    def enricher(self):
        if self._enricher is None:
            kg = load_module("kg_enricher")
            self._enricher = kg.KGEnricher(verbose=self.verbose)
        return self._enricher

    def lookup(self, disease_name: str) -> dict:
        """
        查询一个疾病的全部关联数据。

        数据来源:
          1. ChromaDB (disease_knowledge Collection) — 疾病基本信息
          2. relations.json (知识图谱) — 药品/食物/检查/并发症/治疗

        Returns:
            完整的疾病知识卡片 dict
        """
        start = time.time()

        # ---- 1. ChromaDB 检索 ----
        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"  查询疾病: {disease_name}")
            print(f"{'=' * 70}")
            print(f"  [1/3] ChromaDB 向量检索...")

        chroma_results = self.vector_store.search_disease(disease_name, top_k=5)
        chroma_match = None

        for r in chroma_results:
            if r["disease"] == disease_name:
                chroma_match = r
                break
        # Fallback: use top-1 if exact name mismatch
        if chroma_match is None and chroma_results:
            chroma_match = chroma_results[0]

        # ---- 2. KG 富化 ----
        if self.verbose:
            print(f"  [2/3] 知识图谱富化 (药品/食谱/检查/并发症/治疗)...")

        kg_info = self.enricher.enrich_disease(disease_name)

        # Try fuzzy match if exact not found
        if not kg_info["found"] and kg_info.get("suggestion"):
            suggestion = kg_info["suggestion"]
            if self.verbose:
                print(f"        精确匹配失败, 尝试: {suggestion}")
            kg_info = self.enricher.enrich_disease(suggestion)

        # ---- 3. Assemble result ----
        if self.verbose:
            print(f"  [3/3] 组装结果...")

        latency_ms = round((time.time() - start) * 1000, 1)

        result = {
            "query": disease_name,
            "disease_info": None,
            "kg_enrichment": kg_info,
            "metadata": {
                "latency_ms": latency_ms,
                "chroma_matched": bool(chroma_match),
                "kg_exact_match": kg_info["found"],
                "kg_disease_name": kg_info["disease"],
            },
        }

        if chroma_match:
            result["disease_info"] = {
                "disease": chroma_match["disease"],
                "symptoms": chroma_match["symptoms"],
                "departments": chroma_match["departments"],
                "category": chroma_match.get("category", ""),
                "drugs_from_chroma": chroma_match.get("drugs", ""),
                "description": chroma_match.get("desc", "")[:300],
                "vector_score": round(chroma_match["score"], 4),
                "chain": chroma_match.get("chain", ""),
            }

        return result

    def lookup_batch(self, disease_names: list[str]) -> list[dict]:
        """批量查询多个疾病"""
        results = []
        for name in disease_names:
            result = self.lookup(name.strip())
            results.append(result)
        return results

    def print_result(self, result: dict):
        """格式化打印单个疾病的查询结果"""
        print(f"\n{'=' * 70}")
        print(f"  [Disease] {result['query']}")
        print(f"{'=' * 70}")

        # Basic info from ChromaDB
        info = result.get("disease_info")
        if info:
            print(f"\n  +-- Basic Info (ChromaDB) {'-' * 45}")
            print(f"  |  Disease:   {info['disease']}")
            print(f"  |  Score:     {info['vector_score']:.2%}")
            print(f"  |  Symptoms:  {info['symptoms']}")
            print(f"  |  Dept:      {info['departments']}")
            print(f"  |  Category:  {info['category']}")
            if info.get("drugs_from_chroma"):
                print(f"  |  Drugs(DB): {info['drugs_from_chroma']}")
            if info.get("description"):
                desc = info["description"][:200]
                print(f"  |  Desc:      {desc}...")
            print(f"  +{'─' * 60}")
        else:
            print(f"\n  [WARN] ChromaDB: no exact match for '{result['query']}'")

        # KG enrichment
        kg = result.get("kg_enrichment", {})
        if kg.get("found"):
            print(f"\n  +-- KG Enrichment (OpenKG) {'-' * 40}")
            self._print_section(kg, "drugs", "[Drugs]")
            self._print_section(kg, "foods", "[Foods]")
            self._print_section(kg, "checks", "[Checks]")
            self._print_section(kg, "complications", "[Complications]")
            self._print_section(kg, "cures", "[Cures]")
            self._print_section(kg, "symptoms_from_kg", "[KG Symptoms]")
            print(f"  +{'─' * 60}")
        else:
            print(f"\n  [WARN] KG: no entry for '{kg.get('disease', result['query'])}'")
            if kg.get("suggestion"):
                print(f"     Suggestion: {kg['suggestion']}")

        # Metadata
        meta = result.get("metadata", {})
        print(f"\n  Latency: {meta.get('latency_ms', 0):.0f}ms")
        print(f"  ChromaDB match: {meta.get('chroma_matched', False)}")
        print(f"  KG exact match: {meta.get('kg_exact_match', False)}")

    def _print_section(self, kg: dict, key: str, label: str):
        """Print a KG enrichment section."""
        data = kg.get(key)

        if key == "drugs":
            rec = data.get("recommand", [])
            com = data.get("common", [])
            if rec or com:
                print(f"  |")
                print(f"  | {label}")
                if rec:
                    print(f"  |   Recommand: {', '.join(rec[:8])}")
                if com:
                    print(f"  |   Common:    {', '.join(com[:8])}")

        elif key == "foods":
            rec = data.get("recommand", [])
            do = data.get("do_eat", [])
            no = data.get("no_eat", [])
            if rec or do or no:
                print(f"  |")
                print(f"  | {label}")
                if rec:
                    print(f"  |   Recommand:  {', '.join(rec[:8])}")
                if do:
                    print(f"  |   Do Eat:     {', '.join(do[:8])}")
                if no:
                    print(f"  |   No Eat:     {', '.join(no[:8])}")

        elif key in ("checks", "complications", "cures", "symptoms_from_kg"):
            if data:
                print(f"  |")
                print(f"  | {label}")
                print(f"  |   {', '.join(data[:10])}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG 知识图谱疾病查询 — 输入疾病名，返回全量关联数据"
    )
    parser.add_argument(
        "diseases", nargs="*",
        help="疾病名称 (可多个，空格分隔)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="以 JSON 格式输出 (适合 Java 端对接)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="从知识库随机抽样测试"
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="随机抽样数量 (配合 --all, 默认5)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="交互模式: 持续输入疾病名查询, 输入 q 退出"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  RAG 知识图谱疾病查询 — 端到端测试")
    print("=" * 70)

    lookup = KGDiseaseLookup(verbose=not args.json)

    # Determine diseases to query
    diseases_to_query = []

    if args.interactive:
        # Interactive mode
        print("\n  输入疾病名查询, 输入 'q' 退出\n")
        while True:
            try:
                name = input("  [Search] Disease: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if name.lower() in ("q", "quit", "exit"):
                break
            if not name:
                continue
            result = lookup.lookup(name)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                lookup.print_result(result)
        return

    elif args.all:
        # Random sample from KG index
        kg = lookup.enricher
        all_diseases = list(kg.index.keys())
        import random
        random.shuffle(all_diseases)
        diseases_to_query = all_diseases[:args.limit]
        print(f"\n  随机抽样 {len(diseases_to_query)} 个疾病:")
        for d in diseases_to_query:
            print(f"    - {d}")

    elif args.diseases:
        diseases_to_query = args.diseases
    else:
        # Default: demo with common diseases
        diseases_to_query = ["感冒", "糖尿病", "高血压", "冠心病", "支气管哮喘"]
        print(f"\n  使用默认测试疾病: {diseases_to_query}")
        print(f"  可用参数: python test_kg_enrich.py 疾病名1 疾病名2 ...")
        print(f"  交互模式: python test_kg_enrich.py --interactive")

    # Execute queries
    if args.json:
        results = lookup.lookup_batch(diseases_to_query)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for name in diseases_to_query:
            result = lookup.lookup(name)
            lookup.print_result(result)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  测试完成。查询 {len(diseases_to_query)} 个疾病")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
