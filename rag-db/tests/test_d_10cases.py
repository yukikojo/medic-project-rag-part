"""
test_d_10cases.py
D组模式 10用例测试 — 各组件准确率 + 运行时间分解

仿照 test_runner.run_category_d() 端到端模式:
  口语化输入 → QueryOptimizer → VectorStore → Reranker → DeepSeek LLM

每个用例分解各环节的:
  - 运行耗时 (ms)
  - 准确率贡献 (标准化是否改善 / 检索置信度 / LLM置信度)

测试用例来源: test_runner.COMPREHENSIVE_TEST_CASES (精选10条覆盖全系统)

运行方式:
  cd "d:/medic project"
  python rag-db/tests/test_d_10cases.py

输出:
  - 终端表格: 每个用例 × 各组件 耗时+准确率
  - JSON: test_results/d_10cases_YYYYMMDD_HHMMSS.json
"""

import os
import sys
import json
import time
import importlib.util
from datetime import datetime
from collections import defaultdict

# ============================================================
# 路径设置
# ============================================================
_RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_DIR)


# ============================================================
# 从 test_runner 导入测试用例 (精选10条)
# ============================================================
from test_runner import COMPREHENSIVE_TEST_CASES

# 精选10条测试用例, 覆盖不同身体系统
# 格式: (口语化query, category)
PICKED_INDICES = [
    0,   # "肚子疼拉稀想吐没胃口" → 消化道
    6,   # "发烧咳嗽流鼻涕嗓子疼" → 呼吸道
    11,  # "心慌胸闷气短胸口疼" → 心血管
    17,  # "睡不着没精神心里发慌" → 精神神经
    21,  # "皮肤起红疙瘩痒得厉害" → 皮肤
    24,  # "腰疼腿麻关节疼走不动路" → 骨骼肌肉
    33,  # "月经不规律大姨妈不准经期推迟" → 妇科
    36,  # "小孩发烧咳嗽流鼻涕不爱吃饭" → 儿科
    43,  # "眼睛疼看不清眼干眼胀痛" → 眼科
    52,  # "突然意识不清晕倒抽搐" → 危急/神经
]

TEST_CASES = [COMPREHENSIVE_TEST_CASES[i] for i in PICKED_INDICES]

# ============================================================
# 动态导入
# ============================================================

def _load_module(name, filename):
    path = os.path.join(_RAG_DIR, filename)
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod_qe = _load_module("query_engine", "retrieval/query_engine.py")
_mod_reranker = _load_module("reranker", "reranker/reranker.py")
_mod_dc = _load_module("deepseek_client", "generation/deepseek_client.py")
_mod_qo = _load_module("query_optimizer", "retrieval/query_optimizer.py")

VectorStore = _mod_qe.VectorStore if _mod_qe else None
Reranker = _mod_reranker.Reranker if _mod_reranker else None
RAGPipeline = _mod_dc.RAGPipeline if _mod_dc else None
DeepSeekClient = _mod_dc.DeepSeekClient if _mod_dc else None
QueryOptimizer = _mod_qo.QueryOptimizer if _mod_qo else None


# ============================================================
# D组风格测试: 组件级耗时+准确率分解
# ============================================================

class DGroup10Test:
    """D组模式 — 10用例, 逐组件计时+评估"""

    def __init__(self):
        self.store = None
        self.reranker = None
        self.optimizer = None
        self.llm_client = None
        self.records = []  # 每个用例的完整记录

    def init_components(self):
        """初始化所有组件 (延迟加载)"""
        print("  初始化各组件...")

        # 1. VectorStore
        t0 = time.time()
        if VectorStore:
            self.store = VectorStore()
            print(f"    VectorStore 就绪 ({(time.time()-t0)*1000:.0f}ms)")

        # 2. QueryOptimizer (rule模式, 无网络)
        t0 = time.time()
        if QueryOptimizer:
            self.optimizer = QueryOptimizer(mode="rule", cache_enabled=True, verbose=False)
            print(f"    QueryOptimizer(rule) 就绪 ({(time.time()-t0)*1000:.0f}ms)")

        # 3. Reranker (如果模型存在)
        if Reranker:
            try:
                t0 = time.time()
                self.reranker = Reranker(verbose=False)
                # 触发加载
                _ = self.reranker.model
                print(f"    Reranker 就绪 ({(time.time()-t0)*1000:.0f}ms)")
            except FileNotFoundError:
                print(f"    Reranker 跳过 (模型文件不存在)")

        # 4. DeepSeekClient (如果API Key配置)
        if DeepSeekClient:
            try:
                t0 = time.time()
                self.llm_client = DeepSeekClient()
                print(f"    DeepSeekClient 就绪 ({(time.time()-t0)*1000:.0f}ms)")
            except (ValueError, Exception) as e:
                print(f"    DeepSeekClient 跳过 ({e})")

    def run_all(self):
        """执行全部10个测试用例"""
        print("=" * 80)
        print("  D组模式 — 10用例各组件准确率+耗时测试")
        print("=" * 80)
        print(f"  测试用例数: {len(TEST_CASES)}")
        print(f"  来源: test_runner.COMPREHENSIVE_TEST_CASES (精选索引: {PICKED_INDICES})")
        print()

        self.init_components()
        print()

        for idx, (raw_query, category) in enumerate(TEST_CASES):
            self._run_one(idx + 1, raw_query, category)

        self._print_final_report()
        self._save_json()

    def _run_one(self, case_num: int, raw_query: str, category: str):
        """
        执行单个测试用例, 分解为5个环节计时:

        [环节1] QueryOptimizer.optimize()     — 口语标准化
        [环节2] VectorStore.search_disease()  — 向量检索 (cosine only)
        [环节3] Reranker.rerank_results()     — 精排 (如果可用)
        [环节4] VectorStore.comprehensive_*   — 综合检索 (含reranker)
        [环节5] DeepSeekClient.recommend_*    — LLM推理 (用Reranker精排后的Top-5)
        """
        print(f"\n{'─' * 80}")
        print(f"  [Case {case_num:02d}/10] [{category}] {raw_query}")
        print(f"{'─' * 80}")

        record = {
            "case_num": case_num,
            "raw_query": raw_query,
            "category": category,
        }

        # ================================================================
        # 环节1: QueryOptimizer — 口语标准化
        # ================================================================
        t_opt_start = time.time()
        optimized_query = raw_query  # fallback
        opt_symptoms = []
        opt_note = "跳过"
        opt_body_parts = []
        opt_has_emergency = False

        if self.optimizer:
            opt_result = self.optimizer.optimize(raw_query)
            optimized_query = opt_result.get("optimized_query", raw_query)
            opt_symptoms = opt_result.get("symptoms", [])
            opt_note = opt_result.get("normalization_note", "N/A")
            opt_body_parts = opt_result.get("body_parts", [])
            opt_has_emergency = opt_result.get("has_emergency_signals", False)
        latency_opt = (time.time() - t_opt_start) * 1000

        # 优化准确率: 是否成功产生标准化症状
        opt_effective = len(opt_symptoms) > 0 and optimized_query != raw_query

        record["optimizer"] = {
            "original": raw_query,
            "optimized": optimized_query,
            "symptoms": opt_symptoms,
            "body_parts": opt_body_parts,
            "note": opt_note,
            "effective": opt_effective,
            "has_emergency": opt_has_emergency,
            "latency_ms": round(latency_opt, 2),
        }

        print(f"  [环节1] QueryOptimizer  ({latency_opt:6.1f}ms)  "
              f"'{raw_query[:18]}' → '{optimized_query[:25]}'  "
              f"symptoms={opt_symptoms}  {'✓' if opt_effective else '—'}")

        # ================================================================
        # 环节2: VectorStore.search_disease — 纯向量检索 (无reranker)
        # ================================================================
        t_search_start = time.time()
        disease_results_raw = []
        top1_disease_raw = "N/A"
        top1_dept_raw = "N/A"
        top1_score_raw = 0.0

        if self.store:
            disease_results_raw = self.store.search_disease(optimized_query, top_k=10)
            if disease_results_raw:
                top1_disease_raw = disease_results_raw[0].get("disease", "N/A")
                top1_dept_raw = disease_results_raw[0].get("departments", "N/A")
                top1_score_raw = disease_results_raw[0].get("score", 0.0)
        latency_search = (time.time() - t_search_start) * 1000

        record["search_disease"] = {
            "query_used": optimized_query,
            "candidate_count": len(disease_results_raw),
            "top1_disease": top1_disease_raw,
            "top1_department": top1_dept_raw,
            "top1_score": round(top1_score_raw, 4),
            "top3": [
                {"disease": d["disease"], "dept": d["departments"], "score": round(d["score"], 4)}
                for d in disease_results_raw[:3]
            ],
            "latency_ms": round(latency_search, 2),
        }

        print(f"  [环节2] search_disease  ({latency_search:6.1f}ms)  "
              f"top1={top1_disease_raw} → {top1_dept_raw}  (cosine={top1_score_raw:.1%})")

        # ================================================================
        # 环节3: Reranker.rerank_results — Cross-Encoder精排
        # ================================================================
        latency_rerank = 0.0
        reranker_info = {"available": False}
        reranked_scores = []
        reranked_order = []
        reranked_results = []  # 保存精排后的完整结果，供环节5 LLM使用

        if self.reranker and disease_results_raw:
            t_rerank_start = time.time()
            try:
                # 需要深拷贝避免修改原始结果
                candidates_copy = [dict(d) for d in disease_results_raw]
                reranked_results = self.reranker.rerank_results(optimized_query, candidates_copy)
                reranked_scores = [r["score"] for r in reranked_results]
                reranked_order = [r["disease"] for r in reranked_results]
                reranker_info = {
                    "available": True,
                    "reranked_count": len(reranked_results),
                    "top1_after_rerank": reranked_results[0]["disease"] if reranked_results else "N/A",
                    "top1_score_after": round(reranked_results[0]["score"], 4) if reranked_results else 0,
                    "score_delta": round(reranked_results[0]["score"] - top1_score_raw, 4) if reranked_results and top1_score_raw else 0,
                    "order_changed": reranked_order != [d["disease"] for d in disease_results_raw[:len(reranked_results)]],
                }
            except Exception as e:
                reranker_info = {"available": True, "error": str(e)}
            latency_rerank = (time.time() - t_rerank_start) * 1000

        record["reranker"] = {**reranker_info, "latency_ms": round(latency_rerank, 2)}

        if reranker_info.get("available"):
            delta = reranker_info.get("score_delta", 0)
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
            print(f"  [环节3] Reranker         ({latency_rerank:6.1f}ms)  "
                  f"{reranker_info.get('top1_after_rerank', 'N/A')}  "
                  f"score={reranker_info.get('top1_score_after', 0):.1%}  "
                  f"(Δ{delta:+.1%} {arrow})  "
                  f"order_changed={reranker_info.get('order_changed', False)}")
        else:
            print(f"  [环节3] Reranker          — 跳过 (模型未加载)")

        # ================================================================
        # 环节4: VectorStore.comprehensive_search — 综合检索 (含reranker)
        # ================================================================
        t_compr_start = time.time()
        compr_result = {}
        primary = None
        has_compr = False

        if self.store:
            compr_result = self.store.comprehensive_search(optimized_query, top_k=5)
            primary = compr_result.get("primary_recommendation")
            has_compr = primary is not None
        latency_compr = (time.time() - t_compr_start) * 1000

        compr_dept = primary.get("department", "N/A") if primary else "N/A"
        compr_disease = primary.get("disease", "N/A") if primary else "N/A"
        compr_conf = primary.get("confidence", 0) if primary else 0

        record["comprehensive_search"] = {
            "query_used": optimized_query,
            "department": compr_dept,
            "disease": compr_disease,
            "confidence": round(compr_conf, 4),
            "all_departments": compr_result.get("all_departments", [])[:5],
            "latency_ms": round(latency_compr, 2),
        }

        print(f"  [环节4] comprehensive    ({latency_compr:6.1f}ms)  "
              f"→ {compr_disease}  →  {compr_dept}  (conf={compr_conf:.1%})")

        # ================================================================
        # 环节5: DeepSeekClient.recommend_department — LLM推理
        # 优先使用 Reranker精排后的结果，不可用时回退到纯cosine结果
        # ================================================================
        latency_llm = 0.0
        llm_result = {"available": False}

        # 选择 LLM 的输入源: Reranker精排结果 > 纯cosine结果
        if reranked_results:
            llm_input_results = reranked_results[:5]
            llm_input_source = "reranker"
        else:
            llm_input_results = disease_results_raw[:5] if disease_results_raw else []
            llm_input_source = "cosine"

        if self.llm_client and llm_input_results:
            t_llm_start = time.time()
            try:
                rec = self.llm_client.recommend_department(
                    user_query=raw_query,  # 用原始输入保持上下文
                    rag_results=llm_input_results,
                    temperature=0.3,
                    max_tokens=600,
                )
                usage = rec.get("usage", {})
                llm_result = {
                    "available": True,
                    "department": rec.get("department", "N/A"),
                    "disease": rec.get("disease", "N/A"),
                    "confidence": rec.get("confidence", 0),
                    "reasoning": (rec.get("reasoning", "") or "")[:120],
                    "suggestion": (rec.get("suggestion", "") or "")[:120],
                    "alternatives": rec.get("alternative_departments", []),
                    "emergency_warning": rec.get("emergency_warning", False),
                    "tokens_total": usage.get("total_tokens", 0),
                    "tokens_prompt": usage.get("prompt_tokens", 0),
                    "tokens_completion": usage.get("completion_tokens", 0),
                    "error": rec.get("error"),
                    "input_source": llm_input_source,
                }
            except Exception as e:
                llm_result = {"available": True, "error": str(e), "input_source": llm_input_source}
            latency_llm = (time.time() - t_llm_start) * 1000

        record["llm"] = {**llm_result, "latency_ms": round(latency_llm, 2)}

        if llm_result.get("available") and not llm_result.get("error"):
            rec_conf = llm_result.get("confidence", 0)
            rec_dept = llm_result.get("department", "N/A")
            rec_disease = llm_result.get("disease", "N/A")
            tokens = llm_result.get("tokens_total", 0)
            em = "⚠" if llm_result.get("emergency_warning") else ""
            print(f"  [环节5] LLM recommend    ({latency_llm:6.1f}ms)  "
                  f"→ {rec_disease}  →  {rec_dept}  (conf={rec_conf}%, {tokens}tok) {em}")
            print(f"         输入源: {llm_input_source} (Top-5)")
        elif llm_result.get("error"):
            print(f"  [环节5] LLM recommend    — ERROR: {llm_result['error'][:60]}")
        else:
            print(f"  [环节5] LLM recommend    — 跳过 (API Key未配置)")

        # ================================================================
        # 总耗时
        # ================================================================
        total_latency = latency_opt + latency_search + latency_rerank + latency_compr + latency_llm
        record["total_latency_ms"] = round(total_latency, 2)

        # 用 comprehensive_search 的结果做检索准确率判断 (category匹配)
        # category 如 "消化道" / "呼吸道" / "心血管" 等
        cat_dept_map = {
            "消化道": ["消化内科", "内科", "肛肠科"],
            "呼吸道": ["呼吸内科", "内科", "耳鼻喉科"],
            "心血管": ["心内科", "心血管内科", "内科"],
            "精神神经": ["神经内科", "精神科", "中医科"],
            "神经": ["神经内科", "内科"],
            "精神": ["精神科", "心理科"],
            "皮肤": ["皮肤科", "皮肤性病科"],
            "骨骼肌肉": ["骨科", "骨外科", "康复科", "外科"],
            "妇科": ["妇科", "妇产科"],
            "儿科": ["儿科", "内科"],
            "眼科": ["眼科"],
            "耳鼻喉": ["耳鼻喉科"],
            "口腔": ["口腔科", "牙科"],
            "泌尿": ["泌尿外科", "肾内科"],
            "内科": ["内科", "急诊科", "感染科"],
        }
        expected_depts = cat_dept_map.get(category, [category])

        ret_correct = any(
            ed in compr_dept or compr_dept in ed
            for ed in expected_depts
        ) if compr_dept != "N/A" else False

        record["accuracy"] = {
            "category": category,
            "expected_departments": expected_depts,
            "retrieval_dept": compr_dept,
            "retrieval_correct": ret_correct,
        }

        print(f"        检索准确: {'✓' if ret_correct else '✗'}  "
              f"(got={compr_dept}, expected~{expected_depts})")

        # 累计总耗时
        print(f"        环节耗时分解: opt={latency_opt:.0f}ms + "
              f"search={latency_search:.0f}ms + "
              f"rerank={latency_rerank:.0f}ms + "
              f"compr={latency_compr:.0f}ms + "
              f"llm={latency_llm:.0f}ms = "
              f"{total_latency:.0f}ms")

        self.records.append(record)

    # ================================================================
    # 最终报告
    # ================================================================

    def _print_final_report(self):
        print(f"\n\n{'=' * 80}")
        print(f"  最终报告: 各组件准确率 + 运行时间汇总")
        print(f"{'=' * 80}")

        if not self.records:
            print("  无测试记录")
            return

        n = len(self.records)

        # ---- 表格1: 逐用例 × 各环节耗时 (ms) ----
        print(f"\n  ┌─ 表1: 逐用例各组件运行时间 (ms) ─────────────────────────────┐")
        header = (f"  │ {'#':<3s} {'用例':<24s} {'类别':<8s} "
                  f"{'Optimizer':>9s} {'Search':>8s} {'Reranker':>9s} "
                  f"{'Compr':>8s} {'LLM':>8s} {'总计':>8s} │")
        sep =     f"  │ {'─'*3} {'─'*24} {'─'*8} " + "─"*9 + " " + "─"*8 + " " + "─"*9 + " " + "─"*8 + " " + "─"*8 + " " + "─"*8 + " │"
        print(header)
        print(sep)

        # 汇总
        sum_opt = sum_sea = sum_rer = sum_com = sum_llm = sum_tot = 0.0
        opt_eff_count = 0
        ret_correct_count = 0

        for r in self.records:
            lat = lambda key: r.get(key, {}).get("latency_ms", 0)
            o, s, re, c, l = lat("optimizer"), lat("search_disease"), lat("reranker"), lat("comprehensive_search"), lat("llm")
            t = r.get("total_latency_ms", 0)
            sum_opt += o; sum_sea += s; sum_rer += re; sum_com += c; sum_llm += l; sum_tot += t

            q = r["raw_query"][:22]
            cat = r["category"][:7]

            print(f"  │ {r['case_num']:<3d} {q:<24s} {cat:<8s} "
                  f"{o:>8.0f}  {s:>7.0f}  {re:>8.0f}  {c:>7.0f}  {l:>7.0f}  {t:>7.0f} │")

            if r["optimizer"].get("effective"):
                opt_eff_count += 1
            if r["accuracy"].get("retrieval_correct"):
                ret_correct_count += 1

        print(sep)
        print(f"  │ {'平均':<28s} {'':<8s} "
              f"{sum_opt/n:>8.0f}  {sum_sea/n:>7.0f}  {sum_rer/n:>8.0f}  {sum_com/n:>7.0f}  {sum_llm/n:>7.0f}  {sum_tot/n:>7.0f} │")
        print(f"  └{'─'*78}┘")

        # ---- 表格2: 各组件准确率汇总 ----
        print(f"\n  ┌─ 表2: 各组件准确率汇总 ───────────────────────────────────────┐")

        # 优化器准确率: 有效标准化的比例
        opt_acc = opt_eff_count / n * 100 if n else 0
        print(f"  │ QueryOptimizer 标准化有效率: {opt_eff_count}/{n} = {opt_acc:.1f}%          │")
        print(f"  │   (判定标准: optimize后 symptoms 非空 且 optimized_query ≠ 原始输入) │")

        # 检索准确率: comprehensive_search 科室匹配
        ret_acc = ret_correct_count / n * 100 if n else 0
        print(f"  │ VectorStore 检索科室准确率:   {ret_correct_count}/{n} = {ret_acc:.1f}%          │")

        # Reranker 可用性
        rerank_available_count = sum(1 for r in self.records if r["reranker"].get("available"))
        print(f"  │ Reranker 可用:                {rerank_available_count}/{n}                         │")
        if rerank_available_count > 0:
            rerank_changed = sum(1 for r in self.records
                                if r["reranker"].get("order_changed"))
            avg_score_delta = sum(r["reranker"].get("score_delta", 0)
                                 for r in self.records if r["reranker"].get("available")) / rerank_available_count
            print(f"  │ Reranker 排序变更率:          {rerank_changed}/{rerank_available_count}                         │")
            print(f"  │ Reranker 平均分数Δ:           {avg_score_delta:+.1%}                       │")

        # LLM 可用性
        llm_available_count = sum(1 for r in self.records
                                  if r["llm"].get("available") and not r["llm"].get("error"))
        print(f"  │ DeepSeek LLM 可用:            {llm_available_count}/{n}                         │")
        if llm_available_count > 0:
            avg_llm_conf = sum(r["llm"].get("confidence", 0)
                              for r in self.records
                              if r["llm"].get("available") and not r["llm"].get("error")) / llm_available_count
            total_tokens = sum(r["llm"].get("tokens_total", 0)
                              for r in self.records
                              if r["llm"].get("available") and not r["llm"].get("error"))
            emergency_count = sum(1 for r in self.records
                                 if r["llm"].get("emergency_warning"))
            print(f"  │ LLM 平均置信度:               {avg_llm_conf:.0f}%                        │")
            print(f"  │ LLM 总Token消耗:              {total_tokens}                          │")
            print(f"  │ LLM 紧急信号检出:             {emergency_count}                           │")

        print(f"  └{'─'*64}┘")

        # ---- 表格3: 各环节耗时占比 ----
        print(f"\n  ┌─ 表3: 各组件平均耗时占比 ────────────────────────────────────┐")
        if sum_tot > 0:
            labels = ["Optimizer", "Search", "Reranker", "Compr", "LLM"]
            avgs = [sum_opt/n, sum_sea/n, sum_rer/n, sum_com/n, sum_llm/n]
            total_avg = sum_tot / n
            bar_width = 50
            for label, avg in zip(labels, avgs):
                pct = avg / total_avg * 100 if total_avg > 0 else 0
                bar_len = int(bar_width * pct / 100)
                bar = "█" * bar_len + "░" * (bar_width - bar_len)
                print(f"  │ {label:<12s} {bar} {avg:>7.0f}ms ({pct:>5.1f}%) │")
            print(f"  │ {'总计':<12s} {'─'*bar_width} {total_avg:>7.0f}ms (100.0%) │")
        print(f"  └{'─'*64}┘")

        # ---- 综合结论 ----
        print(f"\n  ┌─ 综合结论 ─────────────────────────────────────────────────┐")
        print(f"  │ 总测试用例:     {n}                                            │")
        print(f"  │ 优化器有效率:   {opt_acc:.1f}%                                        │")
        print(f"  │ 检索准确率:     {ret_acc:.1f}%                                        │")
        if llm_available_count > 0:
            print(f"  │ LLM可用:        {llm_available_count}/{n}  (conf avg={avg_llm_conf:.0f}%)                       │")
        print(f"  │ 平均端到端延迟: {total_avg:.0f}ms                                      │")
        bottleneck = labels[avgs.index(max(avgs))]
        print(f"  │ 瓶颈环节:       {bottleneck} ({max(avgs):.0f}ms, {max(avgs)/total_avg*100:.0f}%)                    │")
        print(f"  └{'─'*58}┘")

    # ================================================================
    # JSON保存
    # ================================================================

    def _save_json(self):
        output_dir = os.path.join(_PROJECT_DIR, "test_results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"d_10cases_{timestamp}.json")

        output = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "test_mode": "D组模式 (端到端组件分解)",
                "test_cases_count": len(TEST_CASES),
                "source": "test_runner.COMPREHENSIVE_TEST_CASES",
                "picked_indices": PICKED_INDICES,
            },
            "records": self.records,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n  详细结果已保存: {filepath}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    test = DGroup10Test()
    test.run_all()
