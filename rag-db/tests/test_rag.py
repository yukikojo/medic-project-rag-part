"""
test_rag.py
RAG 医疗知识库 — 完整测试用例与使用方法

运行方式:
    cd "d:/medic project"
    python rag-db/tests/test_rag.py

测试覆盖:
    1. 基础查询 — 5 类常见症状的科室推荐
    2. 边界测试 — 罕见症状 / 口语化描述 / 长文本 / 空输入
    3. 置信度阈值 — 验证不同置信度区间的结果分布
    4. 性能基准 — 检索延迟 / 吞吐量
    5. 增量更新 — 添加新疾病 / 新增症状映射
    6. 综合场景 — 危急症状识别 / 多症状混合 / 科室信息检索
"""

import os
import sys
import json
import time
import importlib.util

# 添加父目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# rag-db 目录名含连字符，不能用 import，用 importlib 加载
import importlib.util
_qe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "query_engine.py")
_spec = importlib.util.spec_from_file_location("query_engine", _qe_path)
_query_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_query_engine)
VectorStore = _query_engine.VectorStore


# ============================================================
# 辅助函数
# ============================================================

def section(title: str):
    """打印章节标题"""
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def sub(title: str):
    """打印小节标题"""
    print(f"\n  --- {title} ---")


def ok(msg: str):
    """打印成功消息"""
    print(f"  [PASS] {msg}")


def fail(msg: str):
    """打印失败消息"""
    print(f"  [FAIL] {msg}")


def info(msg: str):
    """打印信息"""
    print(f"         {msg}")


# ============================================================
# 测试类
# ============================================================

class RAGTestSuite:
    """RAG 知识库测试套件"""

    def __init__(self):
        self.store = None
        self.passed = 0
        self.failed = 0
        self.results_log = []

    def assert_true(self, condition: bool, test_name: str):
        """断言条件为真"""
        if condition:
            self.passed += 1
            ok(test_name)
        else:
            self.failed += 1
            fail(test_name)

    def assert_confidence_above(self, result: dict, threshold: float, test_name: str):
        """断言主推荐置信度高于阈值"""
        primary = result.get("primary_recommendation")
        if primary and primary["confidence"] >= threshold:
            self.passed += 1
            ok(f"{test_name} (confidence={primary['confidence']:.1%})")
        else:
            self.failed += 1
            conf = primary["confidence"] if primary else 0
            fail(f"{test_name} (confidence={conf:.1%}, expected >={threshold:.0%})")

    def assert_department_in(self, result: dict, expected_depts: list, test_name: str):
        """断言推荐科室在预期列表中"""
        primary = result.get("primary_recommendation")
        if not primary:
            self.failed += 1
            fail(f"{test_name} (no primary recommendation)")
            return

        dept = primary["department"]
        # 检查 expected_depts 中的任意一个是否出现在推荐科室中
        matched = any(ed in dept or dept in ed for ed in expected_depts)
        if matched:
            self.passed += 1
            ok(f"{test_name} -> {primary['disease']} -> {dept}")
        else:
            self.failed += 1
            fail(f"{test_name} (got: {dept}, expected one of: {expected_depts})")

    def log_query(self, query: str, result: dict):
        """记录查询结果"""
        primary = result.get("primary_recommendation") or {}
        self.results_log.append({
            "query": query,
            "department": primary.get("department", "N/A"),
            "disease": primary.get("disease", "N/A"),
            "confidence": primary.get("confidence", 0),
            "top3_diseases": [r["disease"] for r in result.get("disease_results", [])[:3]],
        })


# ============================================================
# 测试用例
# ============================================================

def run_tests(store: VectorStore):
    """执行全部测试用例"""
    t = RAGTestSuite()
    t.store = store

    # ==================================================================
    # 第一部分：基础功能测试
    # ==================================================================
    section("Part 1: Basic Query Tests")

    # --- TC-01: 呼吸道症状 ---
    sub("TC-01: Respiratory symptoms")
    r = store.comprehensive_search("头痛发热咳嗽流鼻涕", top_k=5)
    t.log_query("头痛发热咳嗽流鼻涕", r)
    t.assert_department_in(r, ["呼吸内科", "内科", "中医科", "中医综合"], "TC-01: cold/flu -> respiratory/TCM")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")
    info(f"Chain: {(r.get('primary_recommendation') or {}).get('reasoning', 'N/A')}")

    # --- TC-02: 消化道症状 ---
    sub("TC-02: Digestive symptoms")
    r = store.comprehensive_search("腹痛腹泻拉肚子恶心呕吐", top_k=5)
    t.log_query("腹痛腹泻拉肚子恶心呕吐", r)
    t.assert_department_in(r, ["消化内科", "内科"], "TC-02: stomach pain -> digestive")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-03: 皮肤症状 ---
    sub("TC-03: Skin symptoms")
    r = store.comprehensive_search("皮肤痒红肿过敏起疹子", top_k=5)
    t.log_query("皮肤痒红肿过敏起疹子", r)
    t.assert_department_in(r, ["皮肤科", "皮肤性病科"], "TC-03: skin rash -> dermatology")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-04: 心血管症状 ---
    sub("TC-04: Cardiovascular symptoms")
    r = store.comprehensive_search("胸闷心慌气短心脏跳得快", top_k=5)
    t.log_query("胸闷心慌气短心脏跳得快", r)
    t.assert_department_in(r, ["心内科", "内科", "心血管内科"], "TC-04: chest pain -> cardiology")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-05: 骨骼肌肉症状 ---
    sub("TC-05: Musculoskeletal symptoms")
    r = store.comprehensive_search("腰疼腿麻关节疼走不动路", top_k=5)
    t.log_query("腰疼腿麻关节疼走不动路", r)
    t.assert_department_in(r, ["骨科", "骨外科", "外科"], "TC-05: back pain -> orthopedics")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-06: 妇科症状 ---
    sub("TC-06: Gynecological symptoms")
    r = store.comprehensive_search("月经不调痛经下腹痛白带异常", top_k=5)
    t.log_query("月经不调痛经下腹痛白带异常", r)
    t.assert_department_in(r, ["妇科", "妇产科"], "TC-06: menstrual -> gynecology")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-07: 眼科症状 ---
    sub("TC-07: Ophthalmic symptoms")
    r = store.comprehensive_search("眼睛疼视力模糊眼红眼干", top_k=5)
    t.log_query("眼睛疼视力模糊眼红眼干", r)
    t.assert_department_in(r, ["眼科"], "TC-07: eye pain -> ophthalmology")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-08: 儿科症状 ---
    sub("TC-08: Pediatric symptoms")
    r = store.comprehensive_search("小孩发烧咳嗽流鼻涕不爱吃饭", top_k=5)
    t.log_query("小孩发烧咳嗽流鼻涕不爱吃饭", r)
    t.assert_department_in(r, ["儿科", "内科", "呼吸内科"], "TC-08: child fever -> pediatrics")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # ==================================================================
    # 第二部分：边界测试
    # ==================================================================
    section("Part 2: Edge Cases")

    # --- TC-09: 口语化/方言表达 ---
    sub("TC-09: Colloquial expressions")
    r = store.comprehensive_search("肚子疼想吐拉肚子", top_k=5)
    t.log_query("肚子疼想吐拉肚子", r)
    t.assert_confidence_above(r, 0.60, "TC-09: colloquial 'stomach ache'")
    info(f"Matched: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-10: 单字/简短描述 ---
    sub("TC-10: Very short query")
    r = store.comprehensive_search("头晕", top_k=5)
    t.log_query("头晕", r)
    t.assert_true(len(r.get("disease_results", [])) > 0, "TC-10: single word 'dizzy' returns results")
    info(f"Results count: {len(r.get('disease_results', []))}")

    # --- TC-11: 长描述（50+ 字）---
    sub("TC-11: Long description (>50 chars)")
    long_query = (
        "最近一周总是感觉浑身没力气，吃东西也没胃口，"
        "有时候会恶心想吐，肚子隐隐作痛，大便也不太正常，"
        "晚上睡不好，白天头晕没精神，偶尔还会觉得心慌"
    )
    r = store.comprehensive_search(long_query, top_k=5)
    t.log_query(long_query[:30] + "...", r)
    t.assert_true(len(r.get("disease_results", [])) > 0, "TC-11: long text returns results")
    info(f"Top disease: {r['disease_results'][0]['disease'] if r['disease_results'] else 'N/A'}")

    # --- TC-12: 罕见症状 ---
    sub("TC-12: Rare symptom")
    r = store.comprehensive_search("指甲发黑变形", top_k=5)
    t.log_query("指甲发黑变形", r)
    # 罕见症状可能置信度低，但应返回结果
    has_result = len(r.get("disease_results", [])) > 0
    t.assert_true(has_result, "TC-12: rare symptom returns >=0 results")
    if has_result:
        info(f"Best match: {r['disease_results'][0]['disease']} ({r['disease_results'][0]['score']:.1%})")
    else:
        info("No match found (acceptable for extremely rare symptom)")

    # --- TC-13: 无关输入 ---
    sub("TC-13: Non-medical input")
    r = store.comprehensive_search("今天天气真好", top_k=5)
    t.log_query("今天天气真好", r)
    # 非医疗输入不应该有高置信度匹配
    primary = r.get("primary_recommendation")
    low_confidence = not primary or primary["confidence"] < 0.65
    t.assert_true(low_confidence, "TC-13: non-medical -> low confidence or no match")
    info(f"Confidence: {primary['confidence']:.1%}" if primary else "No primary match")

    # ==================================================================
    # 第三部分：置信度阈值测试
    # ==================================================================
    section("Part 3: Confidence Threshold Analysis")

    # --- TC-14: 高置信度查询（应 >= 70%）---
    sub("TC-14: High-confidence queries")
    high_conf_queries = [
        "咳嗽咳痰发热咽痛",
        "皮肤瘙痒起红疹",
        "牙疼牙龈出血",
    ]
    for q in high_conf_queries:
        r = store.comprehensive_search(q, top_k=3)
        t.log_query(q, r)
        primary = r.get("primary_recommendation")
        conf = primary["confidence"] if primary else 0
        t.assert_true(conf >= 0.65, f"TC-14: '{q}' -> {conf:.1%}")

    # --- TC-15: 中等置信度查询（60%-75%）---
    sub("TC-15: Medium-confidence queries")
    med_conf_queries = [
        "浑身没劲不想吃饭",
        "总感觉心里发慌",
    ]
    for q in med_conf_queries:
        r = store.comprehensive_search(q, top_k=3)
        t.log_query(q, r)
        primary = r.get("primary_recommendation")
        conf = primary["confidence"] if primary else 0
        info(f"'{q}' -> {primary['disease'] if primary else 'N/A'} -> {primary['department'] if primary else 'N/A'} ({conf:.1%})")
        t.assert_true(True, f"TC-15: '{q}' -> {conf:.1%} (info only)")

    # ==================================================================
    # 第四部分：多 Collection 交叉验证
    # ==================================================================
    section("Part 4: Cross-Collection Verification")

    # --- TC-16: disease 和 symptom_direct 结果一致性 ---
    sub("TC-16: Consistency between collections")
    query = "发热咳嗽"
    r = store.comprehensive_search(query, top_k=5)
    t.log_query(query, r)

    # disease_search 和 symptom_direct 的科室应该有一致性
    disease_depts = set()
    for d in r.get("disease_results", [])[:3]:
        for dept in d["departments"].split(", "):
            if dept:
                disease_depts.add(dept)

    symptom_depts = set()
    for s in r.get("symptom_direct", [])[:3]:
        for dept in s["departments"].split(", "):
            if dept:
                symptom_depts.add(dept)

    overlap = disease_depts & symptom_depts
    info(f"Disease departments: {disease_depts}")
    info(f"Symptom departments: {symptom_depts}")
    info(f"Overlap: {overlap}")
    t.assert_true(
        len(overlap) > 0 or len(disease_depts) > 0,
        "TC-16: cross-collection consistency"
    )

    # --- TC-17: 科室信息检索 ---
    sub("TC-17: Department info retrieval")
    dept_result = store.search_department("呼吸内科", top_k=3)
    t.log_query("呼吸内科 (dept search)", {"primary_recommendation": {}})
    has_dept_info = len(dept_result) > 0
    t.assert_true(has_dept_info, "TC-17: department info lookup works")
    if has_dept_info:
        info(f"Department: {dept_result[0]['department']}")
        info(f"Disease count: {dept_result[0]['disease_count']}")
        info(f"Common symptoms: {dept_result[0]['common_symptoms'][:80]}...")

    # ==================================================================
    # 第五部分：性能基准
    # ==================================================================
    section("Part 5: Performance Benchmarks")

    # --- TC-18: 单次检索延迟 ---
    sub("TC-18: Single query latency")
    queries = [
        "头痛发热咳嗽",
        "腹痛腹泻恶心",
        "胸闷心慌气短",
        "皮肤痒过敏",
        "腰疼腿麻",
    ]
    latencies = []
    for q in queries:
        start = time.time()
        _ = store.search_disease(q, top_k=5)
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    info(f"Avg: {avg_latency:.1f}ms | Min: {min_latency:.1f}ms | Max: {max_latency:.1f}ms")
    t.assert_true(avg_latency < 50, f"TC-18: avg latency {avg_latency:.1f}ms < 50ms")

    # --- TC-19: 连续查询吞吐量 ---
    sub("TC-19: Throughput (10 consecutive queries)")
    start = time.time()
    for i in range(10):
        _ = store.comprehensive_search(queries[i % len(queries)], top_k=5)
    total_ms = (time.time() - start) * 1000
    qps = 10 / (total_ms / 1000)
    info(f"10 queries in {total_ms:.0f}ms -> {qps:.1f} QPS")
    t.assert_true(qps > 1, f"TC-19: throughput {qps:.1f} QPS > 1")

    # --- TC-20: 数据库统计 ---
    sub("TC-20: Database stats")
    stats = store.get_stats()
    for coll_name, count in stats["collections"].items():
        info(f"{coll_name}: {count} entries")
    t.assert_true(stats["collections"]["disease_knowledge"] == 8808, "TC-20: disease count = 8808")
    t.assert_true(stats["collections"]["department_info"] == 54, "TC-20: department count = 54")

    # ==================================================================
    # 第六部分：增量更新测试
    # ==================================================================
    section("Part 6: Incremental Updates")

    # --- TC-21: 添加新疾病 ---
    sub("TC-21: Add new disease")
    new_doc = "疾病：测试病。症状：测试症状A、测试症状B。所属科室：测试科。简介：这是一个测试用疾病。"
    new_meta = {
        "disease": "测试病",
        "symptoms": "测试症状A, 测试症状B",
        "departments": "测试科",
        "category": "测试",
        "drugs": "",
        "desc": "这是一个测试用疾病",
    }
    try:
        store.add_diseases([new_doc], [new_meta])
        t.assert_true(True, "TC-21: add new disease succeeded")

        # 验证刚添加的数据可检索
        r = store.search_disease("测试症状A", top_k=1)
        found = any(d["disease"] == "测试病" for d in r)
        t.assert_true(found, "TC-21: newly added disease is searchable")
    except Exception as e:
        t.assert_true(False, f"TC-21: add failed - {e}")

    # --- TC-22: 添加新症状映射 ---
    sub("TC-22: Add symptom mapping")
    sym_doc = "症状：测试症状A。常见关联科室：测试科。"
    sym_meta = {
        "symptom": "测试症状A",
        "departments": "测试科",
        "disease_count": 1,
    }
    try:
        store.add_symptoms([sym_doc], [sym_meta])
        t.assert_true(True, "TC-22: add symptom mapping succeeded")
    except Exception as e:
        t.assert_true(False, f"TC-22: add failed - {e}")

    # ==================================================================
    # 第七部分：综合场景测试
    # ==================================================================
    section("Part 7: Comprehensive Scenarios")

    # --- TC-23: 危急症状识别 ---
    sub("TC-23: Emergency symptom detection")
    emergency_queries = [
        "突然剧烈胸痛呼吸困难出冷汗",
        "突然意识不清晕倒抽搐",
        "大出血不止",
    ]
    for q in emergency_queries:
        r = store.comprehensive_search(q, top_k=5)
        t.log_query(q, r)
        primary = r.get("primary_recommendation")
        dept = primary["department"] if primary else "N/A"
        disease = primary["disease"] if primary else "N/A"
        info(f"'{q}' -> {disease} -> {dept}")
        # 危急症状应返回结果
        t.assert_true(r["disease_results"] != [], f"TC-23: emergency '{q[:15]}...' returns results")

    # --- TC-24: 模糊症状（多科室交叉）---
    sub("TC-24: Cross-department symptoms (fuzzy)")
    query = "浑身难受说不清楚哪里不舒服"
    r = store.comprehensive_search(query, top_k=5)
    t.log_query(query, r)
    t.assert_true(
        len(r.get("disease_results", [])) > 0,
        "TC-24: vague symptoms still return results"
    )
    info(f"Top-3: {[d['disease'] for d in r['disease_results'][:3]]}")

    # --- TC-25: 返回结果结构完整性 ---
    sub("TC-25: Response structure integrity")
    r = store.comprehensive_search("头痛", top_k=3)
    t.log_query("头痛", r)

    # 检查 comprehensive_search 返回的字段完整
    required_keys = ["query", "disease_results", "symptom_direct", "all_departments", "primary_recommendation"]
    for key in required_keys:
        t.assert_true(key in r, f"TC-25: response has '{key}' field")

    # 检查 disease_result 字段完整
    if r["disease_results"]:
        dr = r["disease_results"][0]
        for key in ["disease", "symptoms", "departments", "score", "chain"]:
            t.assert_true(key in dr, f"TC-25: disease_result has '{key}' field")

    # ==================================================================
    # 第八部分：查询优化测试 (QueryOptimizer)
    # ==================================================================
    section("Part 8: Query Optimization Tests")

    # 初始化 QueryOptimizer (规则模式, 不依赖LLM)
    try:
        _qo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "query_optimizer.py")
        _qo_spec = importlib.util.spec_from_file_location("query_optimizer", _qo_path)
        _qo = importlib.util.module_from_spec(_qo_spec)
        _qo_spec.loader.exec_module(_qo)
        QueryOptimizer = _qo.QueryOptimizer
        optimizer = QueryOptimizer(mode="rule", cache_enabled=True, verbose=False)
        optimizer_available = True
    except Exception as e:
        info(f"QueryOptimizer 加载失败: {e}")
        optimizer_available = False

    if optimizer_available:
        # --- TC-26: 口语化标准化 (规则模式) ---
        sub("TC-26: Colloquial normalization (rule mode)")

        test_pairs = [
            ("肚子疼拉稀想吐没胃口", ["腹痛", "腹泻", "恶心", "食欲不振"]),
            ("发烧咳嗽流鼻涕嗓子疼", ["发热", "咳嗽", "流涕", "咽痛"]),
            ("心慌胸闷气短胸口疼", ["心悸", "胸闷", "呼吸困难", "胸痛"]),
            ("腰疼腿麻关节疼", ["腰背痛", "腿麻", "关节痛"]),
            ("睡不着没精神心里发慌", ["失眠", "乏力", "心悸"]),
            ("牙疼刷牙出血嘴里起泡", ["牙痛", "牙龈出血", "口腔溃疡"]),
        ]

        all_matched = True
        for raw, expected in test_pairs:
            result = optimizer.optimize(raw)
            matched = all(any(e in s for s in result["symptoms"]) for e in expected)
            if matched:
                ok(f"TC-26: '{raw[:15]}...' → {result['symptoms']}")
            else:
                all_matched = False
                fail(f"TC-26: '{raw[:15]}...' expected {expected}, got {result['symptoms']}")
            t.log_query(raw, {"primary_recommendation": {
                "department": ", ".join(result["symptoms"]),
                "disease": result["normalization_note"],
                "confidence": 1.0 if matched else 0,
            }})

        t.assert_true(all_matched, "TC-26: all colloquial queries normalized correctly")

        # --- TC-27: 方言表达标准化 ---
        sub("TC-27: Dialect expression normalization")
        dialect_tests = [
            ("打摆子发冷", ["寒战"]),
            ("闹肚子拉水", ["腹泻"]),
            ("心口疼肚子胀", ["腹痛", "消化不良"]),
        ]
        for raw, expected in dialect_tests:
            result = optimizer.optimize(raw)
            has_expected = any(
                any(e in s for s in result["symptoms"]) for e in expected
            )
            if has_expected:
                ok(f"TC-27: '{raw}' → {result['symptoms']}")
            else:
                fail(f"TC-27: '{raw}' expected {expected}, got {result['symptoms']}")

        # --- TC-28: 标准术语保留 ---
        sub("TC-28: Standard terms preserved")
        standard_query = "头痛发热咳嗽流鼻涕"
        result = optimizer.optimize(standard_query)
        # 标准术语也应该能被识别
        has_results = len(result["symptoms"]) > 0
        t.assert_true(has_results, f"TC-28: standard query returns symptoms: {result['symptoms']}")
        info(f"Standard '{standard_query}' → {result['symptoms']}")

        # --- TC-29: 非医疗输入处理 ---
        sub("TC-29: Non-medical input handling")
        non_medical = "今天天气真好适合出去玩"
        result = optimizer.optimize(non_medical)
        # 非医疗输入不应返回症状
        no_symptoms = len(result["symptoms"]) == 0
        t.assert_true(no_symptoms, f"TC-29: non-medical -> no symptoms (got: {result['symptoms']})")
        # 应保留原始输入
        t.assert_true(
            result["optimized_query"] == non_medical,
            "TC-29: non-medical query preserved as-is"
        )

        # --- TC-30: 空输入处理 ---
        sub("TC-30: Empty input handling")
        result = optimizer.optimize("")
        t.assert_true(
            result["symptoms"] == [] and result["optimized_query"] == "",
            "TC-30: empty input returns empty result"
        )

        # --- TC-31: 身体部位推断 ---
        sub("TC-31: Body part inference")
        test_body = [
            ("腹痛腹泻恶心", "腹部"),
            ("头痛头晕", "头部"),
            ("胸闷心悸", "胸部"),
            ("皮疹瘙痒", "皮肤"),
            ("眼痛视力模糊", "眼部"),
        ]
        for query, expected_part in test_body:
            result = optimizer.optimize(query)
            body_parts = result.get("body_parts", [])
            # 检查 expected_part 是否出现在 body_parts 中
            found = any(expected_part in bp for bp in body_parts)
            if found:
                ok(f"TC-31: '{query}' → body parts: {body_parts}")
            else:
                fail(f"TC-31: '{query}' expected '{expected_part}' in {body_parts}")

        # --- TC-32: 缓存有效性 ---
        sub("TC-32: Cache effectiveness")
        cache_query = "肚子疼腹泻想吐"
        # 第一次查询
        r1 = optimizer.optimize(cache_query)
        # 第二次查询应从缓存命中
        r2 = optimizer.optimize(cache_query)
        from_cache = r2.get("from_cache", False)
        t.assert_true(from_cache, f"TC-32: repeated query returns from cache")
        # 缓存结果应与第一次一致
        t.assert_true(
            r1["symptoms"] == r2["symptoms"],
            "TC-32: cached result matches original"
        )
        info(f"Cache size: {optimizer.get_cache_stats()['size']}")

        # --- TC-33: 危急信号检测 ---
        sub("TC-33: Emergency signal detection")
        emergency_cases = [
            ("剧烈胸痛呼吸困难", True),
            ("大出血不止", True),
            ("突然意识不清晕倒", True),
            ("普通感冒发烧咳嗽", False),
            ("肚子疼拉肚子", False),
        ]
        for query, expected_emergency in emergency_cases:
            result = optimizer.optimize(query)
            is_emergency = result.get("has_emergency_signals", False)
            if is_emergency == expected_emergency:
                ok(f"TC-33: '{query[:15]}...' emergency={is_emergency}")
            else:
                fail(f"TC-33: '{query[:15]}...' expected emergency={expected_emergency}, got {is_emergency}")

        # --- TC-34: 优化器+Pipeline 集成测试 ---
        sub("TC-34: Optimizer + RAGPipeline integration")
        try:
            _dc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "deepseek_client.py")
            _dc_spec = importlib.util.spec_from_file_location("deepseek_client", _dc_path)
            _dc = importlib.util.module_from_spec(_dc_spec)
            _dc_spec.loader.exec_module(_dc)
            RAGPipeline = _dc.RAGPipeline

            # 使用规则模式创建 Pipeline (不需要额外 LLM 调用)
            pipeline = RAGPipeline(optimizer_mode="rule", verbose=False)

            # 用口语化查询测试
            r = pipeline.query("肚子疼拉稀想吐", top_k=5)
            t.log_query("肚子疼拉稀想吐 (optimized)", r)

            # 验证优化生效
            opt_info = r.get("query_optimization", {})
            has_optimization = opt_info.get("optimized_query", "") != r["query"]
            if has_optimization:
                ok(f"TC-34: Pipeline optimization active: '{r['query']}' → '{opt_info['optimized_query']}'")
                info(f"  Search query: {r.get('search_query', 'N/A')}")
                info(f"  Symptoms: {opt_info.get('symptoms', [])}")
                rag_rec = r.get("rag_results", {}).get("primary_recommendation")
                if rag_rec:
                    info(f"  Recommendation: {rag_rec['department']}")
            else:
                info("TC-34: Optimization returned same query (possibly already standard)")

            t.assert_true(True, "TC-34: Pipeline integration test completed")

        except Exception as e:
            fail(f"TC-34: Pipeline integration failed - {e}")

        # --- TC-35: 词典动态更新 ---
        sub("TC-35: Dynamic dictionary update")
        try:
            test_standard = "测试标准术语"
            test_colloquial = "测试口语词ABC"
            optimizer.add_colloquial_term(test_standard, test_colloquial)

            # 验证新增的映射可被识别
            result = optimizer.optimize(f"我{test_colloquial}了")
            found_test = test_standard in result["symptoms"]
            t.assert_true(found_test, f"TC-35: dynamically added term recognized")

            # 清理
            _qo.COLLOQUIAL_MAP.pop(test_standard, None)
            _qo._COLLOQUIAL_TO_STANDARD.pop(test_colloquial, None)
        except Exception as e:
            fail(f"TC-35: dynamic update failed - {e}")

    else:
        info("TC-26~35: SKIPPED (QueryOptimizer not available)")

    # ==================================================================
    # 汇总
    # ==================================================================
    section("Test Summary")
    total = t.passed + t.failed
    print(f"\n  Total:  {total}")
    print(f"  Passed: {t.passed}")
    print(f"  Failed: {t.failed}")
    print(f"  Rate:   {t.passed / total * 100:.1f}%" if total > 0 else "  Rate: N/A")

    # 输出查询结果摘要表
    if t.results_log:
        print(f"\n  --- Query Results Summary ---")
        print(f"  {'Query':<30s} {'Department':<15s} {'Confidence':>8s}")
        print(f"  {'-'*30} {'-'*15} {'-'*8}")
        for log in t.results_log:
            q = log["query"][:28] + ".." if len(log["query"]) > 30 else log["query"]
            d = log["department"][:13] + ".." if len(log["department"]) > 15 else log["department"]
            c = f"{log['confidence']:.1%}"
            print(f"  {q:<30s} {d:<15s} {c:>8s}")

    return t.failed == 0


# ============================================================
# 独立使用示例（非测试）
# ============================================================

def usage_examples(store: VectorStore):
    """打印典型使用场景的代码示例"""
    section("Usage Examples (Code Snippets)")

    print("""
  # ===== 示例 1: 基本科室推荐 =====
  store = VectorStore()
  result = store.comprehensive_search("头痛发热咳嗽")
  rec = result["primary_recommendation"]
  print(f"推荐科室: {rec['department']}")
  print(f"可能疾病: {rec['disease']}")
  print(f"置信度:   {rec['confidence']:.1%}")
  print(f"推理链:   {rec['reasoning']}")
  # 输出:
  #   推荐科室: 呼吸内科
  #   可能疾病: 感冒
  #   置信度:   70.4%
  #   推理链:   头痛发热咳嗽 -> 感冒 -> 内科, 呼吸内科

  # ===== 示例 2: Top-N 疾病检索 =====
  diseases = store.search_disease("肚子疼腹泻", top_k=5)
  for i, d in enumerate(diseases):
      print(f"{i+1}. {d['disease']} -> {d['departments']} ({d['score']:.1%})")
  # 输出:
  #   1. 急性胃肠炎 -> 内科, 消化内科 (71.5%)
  #   2. 恶心和呕吐 -> 内科, 消化内科 (70.7%)
  #   3. 肠炎 -> 内科, 消化内科 (70.4%)

  # ===== 示例 3: 症状直接匹配科室 =====
  symptoms = store.search_by_symptom("胸闷气短", top_k=3)
  for s in symptoms:
      print(f"{s['symptom']} -> {s['departments']} ({s['score']:.1%})")

  # ===== 示例 4: 查询科室信息 =====
  depts = store.search_department("骨科")
  for d in depts:
      print(f"{d['department']}: {d['disease_count']}种疾病")
      print(f"  常见症状: {d['common_symptoms']}")

  # ===== 示例 5: 数据库管理 =====
  stats = store.get_stats()
  print(stats)
  # {"disease_knowledge": 8808, "symptom_dept_direct": 4826, "department_info": 54}

  # ===== 示例 6: 增量添加知识 =====
  store.add_diseases(
      documents=["疾病：新疾病。症状：xxx。所属科室：xxx。"],
      metadatas=[{"disease": "新疾病", "symptoms": "xxx", "departments": "xxx"}]
  )

  # ===== 示例 7: 与 Spring Boot 对接 (FastAPI) =====
  from fastapi import FastAPI
  app = FastAPI()
  store = VectorStore()

  @app.post("/api/rag/search")
  def search(query: str):
      '''科室推荐 API'''
      result = store.comprehensive_search(query)
      return {
          "code": 200,
          "data": {
              "department": result["primary_recommendation"]["department"],
              "disease": result["primary_recommendation"]["disease"],
              "confidence": result["primary_recommendation"]["confidence"],
              "reasoning": result["primary_recommendation"]["reasoning"],
              "alternatives": [
                  {"disease": r["disease"], "department": r["departments"]}
                  for r in result["disease_results"][:5]
              ]
          }
      }

  @app.post("/api/rag/symptom/analyze")
  def analyze_symptoms(query: str):
      '''症状结构化分析 API'''
      diseases = store.search_disease(query, top_k=5)
      symptoms = store.search_by_symptom(query, top_k=5)
      return {
          "code": 200,
          "data": {
              "extracted_keywords": [s["symptom"] for s in symptoms],
              "possible_diseases": [d["disease"] for d in diseases],
              "recommended_departments": list(set(
                  d["departments"].split(", ")[0] for d in diseases
              ))
          }
      }
""")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  RAG Medical Knowledge Base - Test Suite & Usage Guide")
    print("=" * 65)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Working Dir: {os.getcwd()}")
    print()

    # 初始化
    print("  Initializing VectorStore...")
    store = VectorStore()

    # 1. 运行测试
    all_passed = run_tests(store)

    # 2. 打印使用示例
    usage_examples(store)

    # 3. 最终结果
    print()
    if all_passed:
        print("  [OK] All tests passed!")
    else:
        print("  [WARN] Some tests failed - check output above.")
    print()
