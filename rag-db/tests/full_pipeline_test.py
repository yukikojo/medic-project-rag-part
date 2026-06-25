"""
full_pipeline_test.py
RAG 医疗智能导诊系统 — 全流程端到端测试

覆盖 7 个阶段:
  Stage 1: MySQL 数据源 — rag_disease + rag_disease_kg 读取
  Stage 2: QueryOptimizer  — 口语化标准化
  Stage 3: VectorStore   — BGE-M3 向量检索 + ChromaDB
  Stage 4: Reranker      — Cross-Encoder 精排
  Stage 5: KG Enricher   — MySQL 知识图谱富化 (药品/食谱/检查/并发症)
  Stage 6: LLM Generation — DeepSeek 科室推荐 + 推理
  Stage 7: EMR Extraction — 病历要素结构化提取

运行: cd "d:/medic project" && python rag-db/tests/full_pipeline_test.py
"""
import os, sys, json, time, traceback

# Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _src)

# ============================================================
# Helpers
# ============================================================
PASS = 0; FAIL = 0; RESULTS = []

def ok(msg):
    global PASS; PASS += 1
    print(f"  [PASS] {msg}")

def err(msg):
    global FAIL; FAIL += 1
    print(f"  [FAIL] {msg}")

def stage(num, title):
    print(f"\n{'='*65}")
    print(f"  Stage {num}: {title}")
    print(f"{'='*65}")

def check(name, condition, detail=""):
    if condition: ok(name + (f"  ({detail})" if detail else ""))
    else: err(name + (f"  ({detail})" if detail else ""))

# ============================================================
# Stage 1: MySQL 数据源
# ============================================================
def test_stage1_mysql():
    stage(1, "MySQL 数据源 (rag_disease + rag_disease_kg)")

    try:
        import pymysql
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_src, "..", "..", ".env"))

        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST","localhost"),
            port=int(os.getenv("MYSQL_PORT","3306")),
            user=os.getenv("MYSQL_USER","root"),
            password=os.getenv("MYSQL_PASSWORD",""),
            database=os.getenv("MYSQL_DATABASE","medical_rag"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
        check("1.1 MySQL 连接", True, f"{os.getenv('MYSQL_HOST')}:{os.getenv('MYSQL_PORT')}")

        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) as n FROM rag_disease WHERE status=1")
            n = c.fetchone()["n"]
            check("1.2 rag_disease 表", n == 8808, f"{n} rows (expected 8808)")

            c.execute("SELECT COUNT(*) as n FROM rag_disease_kg WHERE status=1")
            n = c.fetchone()["n"]
            check("1.3 rag_disease_kg 表", n == 231291, f"{n} rows (expected 231291)")

            c.execute("""
                SELECT rel_category, COUNT(*) as n FROM rag_disease_kg WHERE status=1
                GROUP BY rel_category ORDER BY n DESC
            """)
            cats = {r["rel_category"]: r["n"] for r in c.fetchall()}
            print(f"          KG categories: {dict(list(cats.items())[:5])}...")
            check("1.4 KG 类别数", len(cats) == 8, f"{len(cats)} categories")

        conn.close()
    except Exception as e:
        err(f"1.x MySQL error: {e}")

# ============================================================
# Stage 2: QueryOptimizer
# ============================================================
def test_stage2_optimizer():
    stage(2, "QueryOptimizer 口语化标准化")

    try:
        from query_optimizer import QueryOptimizer
        opt = QueryOptimizer(mode="rule", cache_enabled=True, verbose=False)

        tests = [
            ("肚子疼拉稀想吐没胃口", ["腹痛", "腹泻", "恶心", "食欲不振"]),
            ("发烧咳嗽流鼻涕嗓子疼", ["发热", "咳嗽", "流涕", "咽痛"]),
            ("心慌胸闷气短胸口疼",   ["心悸", "胸闷", "呼吸困难", "胸痛"]),
        ]
        all_ok = True
        for raw, expected in tests:
            result = opt.optimize(raw)
            matched = all(any(e in s for s in result["symptoms"]) for e in expected)
            if not matched:
                all_ok = False
                print(f"         '{raw[:20]}...' -> {result['symptoms']} (expected {expected})")

        check("2.1 口语标准化(3条)", all_ok, "全部正确转换" if all_ok else "部分失败")

        # Emergency detection
        r = opt.optimize("剧烈胸痛呼吸困难出冷汗")
        check("2.2 危急信号检测", r.get("has_emergency_signals") == True, str(r.get("has_emergency_signals")))

        r = opt.optimize("普通感冒发烧咳嗽")
        check("2.3 非危急信号", r.get("has_emergency_signals") == False)

        # Body part inference
        r = opt.optimize("腹痛腹泻恶心")
        parts = r.get("body_parts", [])
        check("2.4 身体部位推断", "腹部" in str(parts) or "消化" in str(parts), str(parts))

    except Exception as e:
        err(f"2.x Optimizer error: {e}")

# ============================================================
# Stage 3: VectorStore + ChromaDB
# ============================================================
def test_stage3_retrieval():
    stage(3, "VectorStore 向量检索 (BGE-M3 + ChromaDB)")

    try:
        from query_engine import VectorStore
        store = VectorStore()

        # Stats
        stats = store.get_stats()
        colls = stats["collections"]
        check("3.1 disease_knowledge", colls.get("disease_knowledge") == 8808, str(colls.get("disease_knowledge")))
        check("3.2 symptom_dept_direct", colls.get("symptom_dept_direct") == 4826, str(colls.get("symptom_dept_direct")))
        check("3.3 department_info", colls.get("department_info") == 54, str(colls.get("department_info")))

        # Search test
        results = store.search_disease("头痛发热咳嗽", top_k=5)
        check("3.4 疾病检索", len(results) == 5, f"{len(results)} results")
        if results:
            top = results[0]
            check("3.5 Top-1 置信度", top["score"] > 0.5, f"score={top['score']:.2%} disease={top['disease']}")
            check("3.6 推理链完整", len(top.get("departments","")) > 0, top.get("departments",""))

        # Cross-collection
        r = store.comprehensive_search("发热咳嗽咽痛", top_k=3)
        check("3.7 综合检索", len(r.get("disease_results",[])) > 0)
        check("3.8 症状直接映射", len(r.get("symptom_direct",[])) > 0)
        check("3.9 科室汇总", len(r.get("all_departments",[])) > 0, str(r.get("all_departments",[])[:3]))

        # Performance
        t = time.time()
        _ = store.search_disease("头痛", top_k=5)
        lat = (time.time() - t) * 1000
        check("3.10 检索延迟", lat < 50, f"{lat:.1f}ms")

    except Exception as e:
        err(f"3.x Retrieval error: {e}")

# ============================================================
# Stage 4: Reranker
# ============================================================
def test_stage4_reranker():
    stage(4, "Reranker Cross-Encoder 精排")

    try:
        from reranker import Reranker, RERANKER_MODEL_PATH
        import os as _os

        if _os.path.exists(RERANKER_MODEL_PATH):
            rr = Reranker(verbose=False)

            query = "头痛发热咳嗽流鼻涕"
            candidates = [
                "疾病：感冒。症状：头痛、发热、咳嗽、流鼻涕。所属科室：呼吸内科。",
                "疾病：偏头痛。症状：头痛、恶心、畏光。所属科室：神经内科。",
                "疾病：过敏性鼻炎。症状：流鼻涕、打喷嚏。所属科室：耳鼻喉科。",
                "疾病：肺炎。症状：发热、咳嗽、胸痛。所属科室：呼吸内科。",
                "疾病：高血压。症状：头痛、头晕。所属科室：心内科。",
            ]
            results = rr.rerank(query, candidates[:3])

            check("4.1 Reranker 加载", True, f"model at {RERANKER_MODEL_PATH[:50]}...")
            check("4.2 精排结果", len(results) >= 3, f"{len(results)} results")
            if results:
                check("4.3 分数排序", results[0]["score"] >= results[-1]["score"], "sorted descending")

            # Check rerank_results preserves cosine_score
            mock = [
                {"disease":"感冒","symptoms":"头痛发热","departments":"呼吸内科","category":"","desc":"","score":0.85,"chain":""},
                {"disease":"偏头痛","symptoms":"头痛","departments":"神经内科","category":"","desc":"","score":0.80,"chain":""},
            ]
            enriched = rr.rerank_results(query, mock)
            check("4.4 cosine_score 保留", "cosine_score" in enriched[0])
        else:
            print("         Reranker model not found — skipping Stage 4")

    except Exception as e:
        err(f"4.x Reranker error: {e}")

# ============================================================
# Stage 5: KG Enricher (MySQL)
# ============================================================
def test_stage5_kg():
    stage(5, "KG Enricher 知识图谱富化 (MySQL rag_disease_kg)")

    try:
        from kg_enricher import KGEnricher
        enricher = KGEnricher(use_mysql=True, verbose=False)

        # Exact match
        info = enricher.enrich_disease("感冒", max_drugs=3, max_foods=3, max_checks=3)
        check("5.1 精确匹配", info["found"] == True)
        check("5.2 数据源", info.get("source") == "mysql", str(info.get("source")))
        check("5.3 推荐药品", len(info["drugs"].get("recommand",[])) >= 3, str(len(info["drugs"].get("recommand",[]))))
        check("5.4 推荐食谱", len(info["foods"].get("recommand",[])) >= 3)
        check("5.5 忌吃食物", len(info["foods"].get("no_eat",[])) >= 2)
        check("5.6 建议检查", len(info["checks"]) >= 3)
        check("5.7 并发症",   len(info["complications"]) >= 1)
        check("5.8 治疗方法", len(info["cures"]) >= 1)

        # Another disease
        info2 = enricher.enrich_disease("糖尿病", max_drugs=3, max_foods=2)
        check("5.9 糖尿病", info2["found"] == True and len(info2["drugs"].get("recommand",[])) >= 2)

        # Batch enrich
        mock = [{"disease":"感冒","score":0.85},{"disease":"糖尿病","score":0.72}]
        enriched = enricher.enrich_results(mock, max_drugs=2, max_foods=2)
        check("5.10 批量富化", all("kg_enrichment" in r for r in enriched))

    except Exception as e:
        err(f"5.x KG error: {e}")
        traceback.print_exc()

# ============================================================
# Stage 6: LLM Generation (DeepSeek)
# ============================================================
def test_stage6_llm():
    stage(6, "LLM Generation (DeepSeek 科室推荐)")

    try:
        from deepseek_client import DeepSeekClient, RAGPipeline
        from query_engine import VectorStore

        pipeline = RAGPipeline(reranker_enabled=True, optimizer_mode="rule", verbose=False)

        # Health check
        hc = pipeline.llm.health_check()
        check("6.1 LLM 健康检查", hc.get("status") == "ok", str(hc.get("model","")))

        # Full pipeline query
        result = pipeline.query("头痛发热咳嗽流鼻涕", top_k=5)
        rec = result.get("recommendation", {})
        rag = result.get("rag_results", {})

        check("6.2 Pipeline 查询成功", not rec.get("error"), rec.get("department","N/A"))
        check("6.3 推荐科室", len(rec.get("department","")) > 0, rec.get("department",""))
        check("6.4 置信度", isinstance(rec.get("confidence"), (int,float)), str(rec.get("confidence")))
        check("6.5 推理依据", len(rec.get("reasoning","")) > 10, (rec.get("reasoning","")[:60] + "..."))
        check("6.6 就医建议", len(rec.get("suggestion","")) > 5, (rec.get("suggestion","")[:60] + "..."))
        check("6.7 备选科室", isinstance(rec.get("alternative_departments"), list), str(rec.get("alternative_departments",[])))
        check("6.8 RAG 检索结果", len(rag.get("disease_results",[])) > 0)
        check("6.9 Token 消耗", rec.get("usage",{}).get("total_tokens",0) > 0, f"{rec.get('usage',{}).get('total_tokens',0)} tokens")

        # Symptom extraction
        sym = pipeline.llm.extract_symptoms("近一周头晕乏力没精神")
        check("6.10 症状提取", len(sym.get("main_symptoms",[])) > 0, str(sym.get("main_symptoms",[])))

        # Optimizer integration
        result2 = pipeline.query("肚子疼拉稀想吐没胃口", top_k=3)
        opt = result2.get("query_optimization", {})
        check("6.11 查询优化集成", opt is not None and len(opt.get("symptoms",[])) > 0,
              str(opt.get("symptoms",[])[:4] if opt else "None"))

    except Exception as e:
        err(f"6.x LLM error: {e}")
        traceback.print_exc()

# ============================================================
# Stage 7: EMR Extraction
# ============================================================
def test_stage7_emr():
    stage(7, "EMR Extraction 病历要素提取")

    try:
        from emr_extractor import EMRProcessor
        processor = EMRProcessor(verbose=False)

        result = processor.extract_medical_record(
            symptom_text="近3天反复发热，最高39.2度，咳嗽咳黄痰，右侧胸痛，呼吸困难",
            health_record={
                "past_illness": "高血压5年，2型糖尿病3年",
                "allergy": "青霉素过敏",
                "surgery_history": "阑尾切除术 2019",
                "medication": "硝苯地平 30mg qd, 二甲双胍 500mg bid",
            },
            patient_info={"age": 65, "gender": "男"},
        )

        if result.error:
            err(f"7.x EMR error: {result.error}")
        else:
            data = result.to_dict()
            checks = [
                ("7.1 主诉", data.get("chief_complaint"), len(data.get("chief_complaint","")) > 3),
                ("7.2 现病史", data.get("present_illness"), len(data.get("present_illness","")) > 20),
                ("7.3 既往史含高血压", data.get("past_history"), "高血压" in (data.get("past_history","") or "")),
                ("7.4 过敏史含青霉素", data.get("allergy_history"), "青霉素" in (data.get("allergy_history","") or "")),
                ("7.5 家族史", data.get("family_history"), data.get("family_history") is not None),
                ("7.6 用药史", data.get("medication_hist"), len(data.get("medication_hist","")) > 0),
                ("7.7 诊断(AI)", data.get("diagnosis"), len(data.get("diagnosis","")) > 5),
                ("7.8 处理意见(AI)", data.get("treatment"), len(data.get("treatment","")) > 5),
            ]
            for name, val, cond in checks:
                check(name, cond, (str(val)[:50] + "...") if val else "None")

            check("7.9 Token 消耗", result.usage is not None and result.usage.get("total_tokens",0) > 0,
                  str(result.usage))
    except Exception as e:
        err(f"7.x EMR error: {e}")
        traceback.print_exc()


# ============================================================
# Summary
# ============================================================
def main():
    start = time.time()

    print("=" * 65)
    print("  RAG 医疗智能导诊系统 — 全流程端到端测试")
    print("  7 Stages: MySQL → Optimizer → Retrieval → Reranker")
    print("            → KG Enricher → LLM → EMR")
    print("=" * 65)

    tests = [
        ("MySQL",       test_stage1_mysql),
        ("Optimizer",   test_stage2_optimizer),
        ("Retrieval",   test_stage3_retrieval),
        ("Reranker",    test_stage4_reranker),
        ("KG Enricher", test_stage5_kg),
        ("LLM",         test_stage6_llm),
        ("EMR",         test_stage7_emr),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  *** Stage {name} CRASHED: {e}")
            traceback.print_exc()

    elapsed = time.time() - start
    total = PASS + FAIL
    rate = PASS / total * 100 if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  TEST SUMMARY")
    print(f"{'='*65}")
    print(f"  Total:  {total}")
    print(f"  Passed: {PASS}")
    print(f"  Failed: {FAIL}")
    print(f"  Rate:   {rate:.1f}%")
    print(f"  Time:   {elapsed:.1f}s")
    print()

    if FAIL == 0:
        print("  [OK] ALL STAGES PASSED!")
    else:
        print(f"  [WARN] {FAIL} checks failed")

if __name__ == "__main__":
    main()
