"""
test_reranker_comparison.py
Reranker 对比测试 — A/C 两组: 启用/禁用 BGE-Reranker 准确率对比

测试方案:
  A组 — 本地向量检索: 100个测试用例
    对比: VectorStore(use_reranker=False) vs VectorStore(use_reranker=True)
    指标: 准确率 / 平均置信度 / 平均延迟

  C组 — 查询优化效果: 80个测试用例
    对比: 2×2 矩阵 (无优化/有优化 × 无reranker/有reranker)
    指标: 平均置信度 / 置信度提升 / 延迟

运行:
  cd "d:/medic project"
  python rag-db/tests/test_reranker_comparison.py

输出:
  test_results/reranker_comparison_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import json
import time
import importlib.util
from datetime import datetime
from typing import Optional
from collections import defaultdict

# 路径设置
_RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_DIR)


# ============================================================
# 导入 test_runner 中定义的测试用例
# ============================================================
from test_runner import LOCAL_TEST_CASES, OPTIMIZATION_TEST_CASES


def _load_query_engine_module():
    """Load query_engine module once (singleton)."""
    if not hasattr(_load_query_engine_module, "_module"):
        spec = importlib.util.spec_from_file_location(
            "query_engine", os.path.join(_RAG_DIR, "query_engine.py")
        )
        qe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qe)
        _load_query_engine_module._module = qe
    return _load_query_engine_module._module


def load_optimizer():
    """Load QueryOptimizer (rule mode)"""
    spec = importlib.util.spec_from_file_location(
        "query_optimizer", os.path.join(_RAG_DIR, "query_optimizer.py")
    )
    qo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qo)
    return qo.QueryOptimizer(mode="rule", cache_enabled=True, verbose=False)


# ============================================================
# A group: compare search_disease (cosine only) vs comprehensive_search (reranked)
# ============================================================

def _check_accuracy(result_dict, expected_depts):
    """Check if a result dict matches expected departments.
    result_dict can be from search_disease() or comprehensive_search()['primary_recommendation'].
    """
    if isinstance(result_dict, dict) and "departments" in result_dict:
        # From search_disease / disease_results
        dept = result_dict.get("departments", "").split(", ")[0]
        conf = result_dict.get("score", 0)
        return any(ed in dept or dept in ed for ed in expected_depts), dept, conf
    elif isinstance(result_dict, dict) and "department" in result_dict:
        # From comprehensive_search primary_recommendation
        dept = result_dict.get("department", "")
        conf = result_dict.get("confidence", 0)
        return any(ed in dept or dept in ed for ed in expected_depts), dept, conf
    return False, "N/A", 0.0


def run_category_a_comparison(verbose: bool = True) -> dict:
    """
    Run A group test cases, comparing reranker ON vs OFF accuracy.
    Uses a SINGLE VectorStore instance to avoid duplicate model loading (memory).

    OFF: store.search_disease() — cosine similarity only
    ON: store.comprehensive_search() — cosine + cross-encoder rerank
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  A Group: Local Vector Retrieval — Reranker ON vs OFF")
        print("=" * 70)

    qe = _load_query_engine_module()
    store = qe.VectorStore(use_reranker=True)  # single instance, shares embedding model

    results = []
    correct_off = 0
    correct_on = 0
    total = 0

    category_stats = defaultdict(lambda: {
        "off_correct": 0, "on_correct": 0, "count": 0,
        "off_latencies": [], "on_latencies": [],
        "off_scores": [], "on_scores": [],
    })

    for query, expected_depts, category in LOCAL_TEST_CASES:
        # --- OFF: cosine similarity only (search_disease, no reranker) ---
        start_off = time.time()
        diseases_off = store.search_disease(query, top_k=5)
        lat_off = (time.time() - start_off) * 1000

        # --- ON: reranker enabled (comprehensive_search) ---
        start_on = time.time()
        rag_on = store.comprehensive_search(query, top_k=5)
        lat_on = (time.time() - start_on) * 1000

        # OFF: use top-1 disease from search_disease
        top_off = diseases_off[0] if diseases_off else {}
        is_correct_off, dept_off, conf_off = _check_accuracy(top_off, expected_depts)

        # ON: use primary_recommendation from comprehensive_search
        primary_on = rag_on.get("primary_recommendation") or {}
        is_correct_on, dept_on, conf_on = _check_accuracy(primary_on, expected_depts)
        if not primary_on:
            # fallback to top-1 disease
            top_on = rag_on["disease_results"][0] if rag_on.get("disease_results") else {}
            is_correct_on, dept_on, conf_on = _check_accuracy(top_on, expected_depts)

        if is_correct_off:
            correct_off += 1
        if is_correct_on:
            correct_on += 1
        total += 1

        stats = category_stats[category]
        stats["count"] += 1
        stats["off_latencies"].append(lat_off)
        stats["on_latencies"].append(lat_on)
        stats["off_scores"].append(conf_off)
        stats["on_scores"].append(conf_on)
        if is_correct_off:
            stats["off_correct"] += 1
        if is_correct_on:
            stats["on_correct"] += 1

        record = {
            "query": query,
            "category": category,
            "expected_departments": expected_depts,
            "off_dept": dept_off,
            "off_confidence": round(conf_off, 4),
            "off_correct": is_correct_off,
            "off_latency_ms": round(lat_off, 2),
            "on_dept": dept_on,
            "on_confidence": round(conf_on, 4),
            "on_correct": is_correct_on,
            "on_latency_ms": round(lat_on, 2),
            "confidence_delta": round(conf_on - conf_off, 4),
            "reranker_helped": (not is_correct_off) and is_correct_on,
            "reranker_hurt": is_correct_off and (not is_correct_on),
        }
        results.append(record)

        if verbose:
            if is_correct_off and is_correct_on:
                status = "  ..  "
            elif (not is_correct_off) and is_correct_on:
                status = "FIXED"
            elif is_correct_off and (not is_correct_on):
                status = "REGR "
            else:
                status = "FAIL "

            print(f"  [{status}] [{category}] {query[:28]:<28s}")
            print(f"         OFF: {dept_off:<14s} conf={conf_off:.1%} {'v' if is_correct_off else 'x'}  "
                  f"ON: {dept_on:<14s} conf={conf_on:.1%} {'v' if is_correct_on else 'x'}  "
                  f"delta={conf_on - conf_off:+.1%}")

    # Summary
    all_lat_off = [r["off_latency_ms"] for r in results]
    all_lat_on = [r["on_latency_ms"] for r in results]
    all_conf_off = [r["off_confidence"] for r in results]
    all_conf_on = [r["on_confidence"] for r in results]

    fixed_count = sum(1 for r in results if r["reranker_helped"])
    regr_count = sum(1 for r in results if r["reranker_hurt"])

    summary = {
        "total": total,
        "off_accuracy": round(correct_off / total, 4) if total > 0 else 0,
        "on_accuracy": round(correct_on / total, 4) if total > 0 else 0,
        "accuracy_delta": round((correct_on - correct_off) / total, 4) if total > 0 else 0,
        "off_avg_latency_ms": round(sum(all_lat_off) / len(all_lat_off), 2) if all_lat_off else 0,
        "on_avg_latency_ms": round(sum(all_lat_on) / len(all_lat_on), 2) if all_lat_on else 0,
        "latency_increase_ms": round(
            (sum(all_lat_on) / len(all_lat_on) - sum(all_lat_off) / len(all_lat_off)), 2
        ) if all_lat_on else 0,
        "off_avg_confidence": round(sum(all_conf_off) / len(all_conf_off), 4) if all_conf_off else 0,
        "on_avg_confidence": round(sum(all_conf_on) / len(all_conf_on), 4) if all_conf_on else 0,
        "confidence_delta": round(
            (sum(all_conf_on) / len(all_conf_on) - sum(all_conf_off) / len(all_conf_off)), 4
        ) if all_conf_on else 0,
        "fixed_by_reranker": fixed_count,
        "regression_by_reranker": regr_count,
        "by_category": {
            cat: {
                "count": s["count"],
                "off_accuracy": round(s["off_correct"] / s["count"], 4) if s["count"] > 0 else 0,
                "on_accuracy": round(s["on_correct"] / s["count"], 4) if s["count"] > 0 else 0,
                "off_avg_latency_ms": round(sum(s["off_latencies"]) / len(s["off_latencies"]), 2) if s["off_latencies"] else 0,
                "on_avg_latency_ms": round(sum(s["on_latencies"]) / len(s["on_latencies"]), 2) if s["on_latencies"] else 0,
                "off_avg_confidence": round(sum(s["off_scores"]) / len(s["off_scores"]), 4) if s["off_scores"] else 0,
                "on_avg_confidence": round(sum(s["on_scores"]) / len(s["on_scores"]), 4) if s["on_scores"] else 0,
            }
            for cat, s in category_stats.items()
        },
    }

    if verbose:
        print(f"\n  --- A Group Summary ---")
        print(f"  Accuracy:    OFF={summary['off_accuracy']:.1%}  -->  ON={summary['on_accuracy']:.1%}  "
              f"(delta={summary['accuracy_delta']:+.1%})")
        print(f"  Avg Conf:    OFF={summary['off_avg_confidence']:.1%}  -->  ON={summary['on_avg_confidence']:.1%}  "
              f"(delta={summary['confidence_delta']:+.1%})")
        print(f"  Avg Latency: OFF={summary['off_avg_latency_ms']:.0f}ms  -->  ON={summary['on_avg_latency_ms']:.0f}ms  "
              f"(+{summary['latency_increase_ms']:.0f}ms)")
        print(f"  Reranker fixed: {fixed_count} | Regressed: {regr_count}")

    return {"results": results, "summary": summary}


# ============================================================
# C组测试: 查询优化 + Reranker 2×2 对比
# ============================================================

def run_category_c_comparison(verbose: bool = True) -> dict:
    """
    Run C group test cases, comparing 2x2 matrix:
      - raw query + no reranker (search_disease)
      - raw query + reranker (comprehensive_search)
      - optimized query + no reranker (search_disease)
      - optimized query + reranker (comprehensive_search)
    Uses a SINGLE VectorStore instance.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  C Group: Query Optimization x Reranker — 2x2 Comparison")
        print("=" * 70)

    qe = _load_query_engine_module()
    store = qe.VectorStore(use_reranker=True)  # shared instance
    optimizer = load_optimizer()

    results = []

    def _get_top1_conf_dept(search_results):
        """Get top-1 confidence and department from search_disease or comprehensive_search results."""
        if isinstance(search_results, dict):
            # comprehensive_search return
            primary = search_results.get("primary_recommendation")
            if primary:
                return primary.get("confidence", 0), primary.get("department", "N/A")
            top1 = search_results["disease_results"][0] if search_results.get("disease_results") else {}
        elif isinstance(search_results, list):
            # search_disease return
            top1 = search_results[0] if search_results else {}
        else:
            top1 = {}
        return top1.get("score", 0), top1.get("departments", "N/A").split(", ")[0]

    stats = {
        "raw_off": {"confidences": [], "latencies": [], "depts": []},
        "raw_on":  {"confidences": [], "latencies": [], "depts": []},
        "opt_off": {"confidences": [], "latencies": [], "depts": []},
        "opt_on":  {"confidences": [], "latencies": [], "depts": []},
    }

    for raw_query, expected_direction in OPTIMIZATION_TEST_CASES:
        # Step 1: optimize
        opt_result = optimizer.optimize(raw_query)
        optimized_query = opt_result.get("optimized_query", raw_query)
        symptoms = opt_result.get("symptoms", [])

        # Step 2: 4 combinations
        # a) raw + no reranker (search_disease)
        t0 = time.time()
        r_raw_off = store.search_disease(raw_query, top_k=5)
        t1 = time.time()
        # b) raw + reranker (comprehensive_search)
        r_raw_on = store.comprehensive_search(raw_query, top_k=5)
        t2 = time.time()
        # c) opt + no reranker (search_disease)
        r_opt_off = store.search_disease(optimized_query, top_k=5)
        t3 = time.time()
        # d) opt + reranker (comprehensive_search)
        r_opt_on = store.comprehensive_search(optimized_query, top_k=5)
        t4 = time.time()

        conf_raw_off, dept_raw_off = _get_top1_conf_dept(r_raw_off)
        conf_raw_on,  dept_raw_on  = _get_top1_conf_dept(r_raw_on)
        conf_opt_off, dept_opt_off = _get_top1_conf_dept(r_opt_off)
        conf_opt_on,  dept_opt_on  = _get_top1_conf_dept(r_opt_on)

        lat_raw_off = (t1 - t0) * 1000
        lat_raw_on  = (t2 - t1) * 1000
        lat_opt_off = (t3 - t2) * 1000
        lat_opt_on  = (t4 - t3) * 1000

        for key, confs, lats, depts in [
            ("raw_off", [conf_raw_off], [lat_raw_off], [dept_raw_off]),
            ("raw_on",  [conf_raw_on],  [lat_raw_on],  [dept_raw_on]),
            ("opt_off", [conf_opt_off], [lat_opt_off], [dept_opt_off]),
            ("opt_on",  [conf_opt_on],  [lat_opt_on],  [dept_opt_on]),
        ]:
            stats[key]["confidences"].append(confs[0])
            stats[key]["latencies"].append(lats[0])
            stats[key]["depts"].append(depts[0])

        record = {
            "raw_query": raw_query,
            "optimized_query": optimized_query,
            "symptoms": symptoms,
            "raw_off_confidence": round(conf_raw_off, 4),
            "raw_off_dept": dept_raw_off,
            "raw_off_latency_ms": round(lat_raw_off, 2),
            "raw_on_confidence": round(conf_raw_on, 4),
            "raw_on_dept": dept_raw_on,
            "raw_on_latency_ms": round(lat_raw_on, 2),
            "opt_off_confidence": round(conf_opt_off, 4),
            "opt_off_dept": dept_opt_off,
            "opt_off_latency_ms": round(lat_opt_off, 2),
            "opt_on_confidence": round(conf_opt_on, 4),
            "opt_on_dept": dept_opt_on,
            "opt_on_latency_ms": round(lat_opt_on, 2),
            "reranker_gain_raw": round(conf_raw_on - conf_raw_off, 4),
            "reranker_gain_opt": round(conf_opt_on - conf_opt_off, 4),
            "optimizer_gain_off": round(conf_opt_off - conf_raw_off, 4),
            "optimizer_gain_on": round(conf_opt_on - conf_raw_on, 4),
            "combined_gain": round(conf_opt_on - conf_raw_off, 4),
        }
        results.append(record)

        if verbose:
            print(f"  [{raw_query[:25]:<25s}] -> opt: {optimized_query[:25]:<25s}")
            print(f"    RAW  OFF={conf_raw_off:.1%}    ON={conf_raw_on:.1%}    "
                  f"OPT  OFF={conf_opt_off:.1%}    ON={conf_opt_on:.1%}")
            print(f"    Reranker gain(raw): {record['reranker_gain_raw']:+.1%}  "
                  f"Reranker gain(opt): {record['reranker_gain_opt']:+.1%}  "
                  f"Combined: {record['combined_gain']:+.1%}")

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0

    summary = {
        "total": len(results),
        "avg_raw_off_confidence": avg(stats["raw_off"]["confidences"]),
        "avg_raw_on_confidence":  avg(stats["raw_on"]["confidences"]),
        "avg_opt_off_confidence": avg(stats["opt_off"]["confidences"]),
        "avg_opt_on_confidence":  avg(stats["opt_on"]["confidences"]),
        "avg_raw_off_latency_ms": avg(stats["raw_off"]["latencies"]),
        "avg_raw_on_latency_ms":  avg(stats["raw_on"]["latencies"]),
        "avg_opt_off_latency_ms": avg(stats["opt_off"]["latencies"]),
        "avg_opt_on_latency_ms":  avg(stats["opt_on"]["latencies"]),
        "avg_reranker_gain_raw": round(
            avg(stats["raw_on"]["confidences"]) - avg(stats["raw_off"]["confidences"]), 4
        ),
        "avg_reranker_gain_opt": round(
            avg(stats["opt_on"]["confidences"]) - avg(stats["opt_off"]["confidences"]), 4
        ),
        "avg_optimizer_gain_off": round(
            avg(stats["opt_off"]["confidences"]) - avg(stats["raw_off"]["confidences"]), 4
        ),
        "avg_optimizer_gain_on": round(
            avg(stats["opt_on"]["confidences"]) - avg(stats["raw_on"]["confidences"]), 4
        ),
        "avg_combined_gain": round(
            avg(stats["opt_on"]["confidences"]) - avg(stats["raw_off"]["confidences"]), 4
        ),
    }

    if verbose:
        print(f"\n  --- C Group 2x2 Matrix Summary ---")
        print(f"                    No Reranker         Reranker")
        print(f"    Raw query:      {summary['avg_raw_off_confidence']:.1%}               {summary['avg_raw_on_confidence']:.1%}")
        print(f"    Optimized:      {summary['avg_opt_off_confidence']:.1%}               {summary['avg_opt_on_confidence']:.1%}")
        print(f"  Reranker gain (raw): {summary['avg_reranker_gain_raw']:+.1%}")
        print(f"  Reranker gain (opt): {summary['avg_reranker_gain_opt']:+.1%}")
        print(f"  Optimizer gain (no reranker): {summary['avg_optimizer_gain_off']:+.1%}")
        print(f"  Optimizer gain (with reranker): {summary['avg_optimizer_gain_on']:+.1%}")
        print(f"  Combined gain (raw+OFF -> opt+ON): {summary['avg_combined_gain']:+.1%}")

    return {"results": results, "summary": summary}


# ============================================================
# 综合报告
# ============================================================

def print_final_report(a_data: dict, c_data: dict, duration_s: float):
    """Print final comparison report (ASCII-safe)"""
    a_summary = a_data["summary"]
    c_summary = c_data["summary"]

    print("\n")
    print("=" * 70)
    print("  Reranker (BAAI/bge-reranker-v2-m3) Comparison Test - Final Report")
    print("=" * 70)

    # A group
    print(f"""
  [A Group] Local Vector Retrieval Accuracy ({a_summary['total']} test cases)
  ------------------------------------------------------------
    Accuracy:   {a_summary['off_accuracy']:.1%}  -->  {a_summary['on_accuracy']:.1%}  (delta={a_summary['accuracy_delta']:+.1%})
    Confidence: {a_summary['off_avg_confidence']:.1%}  -->  {a_summary['on_avg_confidence']:.1%}  (delta={a_summary['confidence_delta']:+.1%})
    Latency:    {a_summary['off_avg_latency_ms']:.0f}ms  -->  {a_summary['on_avg_latency_ms']:.0f}ms  (+{a_summary['latency_increase_ms']:.0f}ms)
    Fixed: {a_summary['fixed_by_reranker']} cases  |  Regression: {a_summary['regression_by_reranker']} cases""")

    # C group
    print(f"""
  [C Group] Query Optimization x Reranker 2x2 Comparison ({c_summary['total']} cases)
  ------------------------------------------------------------
    Best config  (opt+ON):  {c_summary['avg_opt_on_confidence']:.1%}
    Baseline     (raw+OFF): {c_summary['avg_raw_off_confidence']:.1%}
    Combined gain:          {c_summary['avg_combined_gain']:+.1%}
    Reranker gain (raw):    {c_summary['avg_reranker_gain_raw']:+.1%}
    Reranker gain (opt):    {c_summary['avg_reranker_gain_opt']:+.1%}
    Optimizer gain (no reranker): {c_summary['avg_optimizer_gain_off']:+.1%}
    Optimizer gain (with reranker): {c_summary['avg_optimizer_gain_on']:+.1%}
    Latency:
      raw+OFF={c_summary['avg_raw_off_latency_ms']:.0f}ms  raw+ON={c_summary['avg_raw_on_latency_ms']:.0f}ms
      opt+OFF={c_summary['avg_opt_off_latency_ms']:.0f}ms  opt+ON={c_summary['avg_opt_on_latency_ms']:.0f}ms""")

    print("-" * 70)

    # Conclusion
    a_improved = a_summary["accuracy_delta"] > 0
    c_improved = c_summary["avg_reranker_gain_raw"] > 0

    conclusion_parts = []
    if a_improved:
        conclusion_parts.append(f"A-group accuracy improved by {a_summary['accuracy_delta']:+.1%}")
    else:
        conclusion_parts.append(f"A-group accuracy changed by {a_summary['accuracy_delta']:+.1%}")

    if c_improved:
        conclusion_parts.append(f"C-group reranker gain: {c_summary['avg_reranker_gain_raw']:+.1%}")
    else:
        conclusion_parts.append(f"C-group reranker gain: {c_summary['avg_reranker_gain_raw']:+.1%}")

    conclusion = " | ".join(conclusion_parts)
    print(f"  Conclusion: {conclusion}")
    print(f"  Test duration: {duration_s:.0f}s")
    print("=" * 70)


# ============================================================
# 入口
# ============================================================

def main():
    print("=" * 70)
    print("  Reranker Comparison Test — A/C Groups")
    print("  Model: BAAI/bge-reranker-v2-m3")
    print("=" * 70)

    total_start = time.time()

    # Pre-load reranker model once (shared across all queries)
    print("\n  Preloading reranker model...")
    qe = _load_query_engine_module()
    _store = qe.VectorStore(use_reranker=True)
    # Trigger a query to initialize models
    _store.comprehensive_search("test", top_k=1)
    print("  Reranker model ready.\n")

    # 运行 A 组对比
    print("\n" + "=" * 35 + " A组测试 " + "=" * 35)
    a_data = run_category_a_comparison(verbose=True)

    # 运行 C 组对比
    print("\n" + "=" * 35 + " C组测试 " + "=" * 35)
    c_data = run_category_c_comparison(verbose=True)

    total_duration = time.time() - total_start

    # 打印最终报告
    print_final_report(a_data, c_data, total_duration)

    # 保存结果
    output_dir = os.path.join(_PROJECT_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"reranker_comparison_{timestamp}.json")

    full_results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "model": "BAAI/bge-reranker-v2-m3",
            "test_version": "reranker_comparison_1.0",
            "total_duration_s": round(total_duration, 1),
        },
        "A_local_retrieval": a_data,
        "C_query_optimization": c_data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)

    print(f"\n  详细结果已保存: {filepath}")
    return full_results


if __name__ == "__main__":
    main()
