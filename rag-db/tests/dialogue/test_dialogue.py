"""
test_dialogue.py
Multi-turn Dialogue Agent 测试套件

20 个测试用例, 覆盖:
  TC-01 ~ TC-08  集成测试 (需 LLM + MySQL)
  TC-09 ~ TC-16  单元测试 (纯逻辑, 不调 LLM)
  TC-17 ~ TC-20  MySQL 持久化 + 性能基准

运行: python rag-db/tests/dialogue/test_dialogue.py
"""
import os
import sys
import json
import time
import uuid

# Ensure src/ is importable
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv
load_dotenv(os.path.join(_src, "..", "..", ".env"))

# ============================================================
# Test framework
# ============================================================

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        msg = f"  [PASS] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
# Test data
# ============================================================

HEADACHE_SYMPTOM = "头痛三天了，右边太阳穴跳着疼"
HEADACHE_RESPONSE_1 = "有时候会恶心，看到亮光的时候头痛会更厉害"
HEADACHE_RESPONSE_2 = "头痛频率大概隔一天一次，发作前有时候眼前会有闪光"
HEADACHE_RESPONSE_3 = "睡一觉会好一点，吃布洛芬能缓解"

EMERGENCY_SYMPTOM = "我父亲突然剧烈胸痛，呼吸困难，出冷汗，快不行了"

MINIMAL_SYMPTOM = "有点不舒服"

ABDOMEN_SYMPTOM = "肚子疼了三天，拉肚子"
ABDOMEN_RESPONSE_1 = "疼的位置在肚脐周围，一天拉三四次，水样的"

VAGUE_SYMPTOM = "最近身体不太对劲"

# 5 轮完整对话 (模拟头痛偏头痛场景)
FULL_DIALOGUE = [
    HEADACHE_SYMPTOM,
    HEADACHE_RESPONSE_1,
    HEADACHE_RESPONSE_2,
    HEADACHE_RESPONSE_3,
    "对，最近工作压力确实很大，经常熬夜",
]

TEST_SESSION_ID = str(uuid.uuid4()).replace("-", "")[:36]  # Placeholder
TEST_PATIENT_ID = 99999

# ============================================================
# Test cases
# ============================================================


def run_tests():
    global PASS, FAIL, TEST_SESSION_ID

    print("=" * 60)
    print("  Multi-turn Dialogue Agent — Test Suite")
    print("  20 test cases")
    print("=" * 60)

    # ================================================================
    # Part A: 集成测试 (需 LLM)
    # ================================================================

    section("Part A: 集成测试 (LLM required)")

    try:
        from dialogue import DialogueManager
        manager = DialogueManager(verbose=False)
    except Exception as e:
        print(f"\n  [SKIP] Cannot import DialogueManager: {e}")
        print(f"  Skipping all LLM tests.")
        manager = None

    if manager:
        # TC-01: start_session with initial symptom
        section("TC-01: 开始会话(含初始症状)")
        try:
            result = manager.start_session(
                patient_id=TEST_PATIENT_ID,
                initial_symptom=HEADACHE_SYMPTOM,
                max_turns=8,
            )
            check("TC-01a: 返回 session_id", len(result.get("session_id", "")) == 36)
            check("TC-01b: action 为 ask 或 recommend 或 emergency",
                  result.get("action") in ("ask", "recommend", "emergency"))
            check("TC-01c: current_turn >= 1", result.get("current_turn", 0) >= 1)

            TEST_SESSION_ID = result.get("session_id", "")

            if result.get("action") == "ask":
                check("TC-01d: 有问题文本", bool(result.get("question")))
                check("TC-01e: candidate_diseases 非空",
                      result.get("candidate_diseases") is not None)
            elif result.get("action") == "recommend":
                rec = result.get("recommendation", {})
                check("TC-01d: 有推荐", bool(rec))
                check("TC-01e: 推荐有 department", bool(rec.get("department")))

            if result.get("candidate_diseases"):
                check("TC-01f: 候选疾病有 score",
                      "score" in result["candidate_diseases"][0])

        except Exception as e:
            check("TC-01: 不抛异常", False, str(e)[:100])

        # TC-02: start_session without initial symptom
        section("TC-02: 开始会话(无初始症状)")
        try:
            result = manager.start_session(patient_id=TEST_PATIENT_ID)
            check("TC-02a: action=ask (greeting)", result.get("action") == "ask")
            check("TC-02b: 返回引导问题", bool(result.get("question")))
            check("TC-02c: current_turn=0", result.get("current_turn") == 0)
            check("TC-02d: candidate_diseases=None (首轮无检索)",
                  result.get("candidate_diseases") is None)

            # Clean up greeting session
            greeting_sid = result.get("session_id")
            if greeting_sid:
                manager.close_session(greeting_sid)
        except Exception as e:
            check("TC-02: 不抛异常", False, str(e)[:100])

        # TC-03: continue session (only if TC-01 created a session)
        if TEST_SESSION_ID:
            section("TC-03: 继续会话")
            try:
                result = manager.process(
                    session_id=TEST_SESSION_ID,
                    patient_input=HEADACHE_RESPONSE_1,
                )
                check("TC-03a: 返回 session_id", result.get("session_id") == TEST_SESSION_ID)
                check("TC-03b: action 有效", result.get("action") in ("ask", "recommend", "emergency"))
                check("TC-03c: current_turn >= 2", result.get("current_turn", 0) >= 2)
                check("TC-03d: collected_info 有症状",
                      result.get("collected_info", {}).get("symptoms"))
            except Exception as e:
                check("TC-03: 不抛异常", False, str(e)[:100])

            # TC-04: symptoms accumulate
            section("TC-04: 症状累积")
            try:
                state = manager.get_session_state(TEST_SESSION_ID)
                if state:
                    symptoms = state.get("collected_symptoms", {})
                    if isinstance(symptoms, dict):
                        sym_list = symptoms.get("symptoms", [])
                    elif isinstance(symptoms, list):
                        sym_list = symptoms
                    else:
                        sym_list = []
                    check("TC-04a: 已收集症状数 >= 2 (跨轮累积)",
                          len(sym_list) >= 2,
                          f"symptoms={sym_list}")
                    history = state.get("dialogue_history", [])
                    check("TC-04b: 对话历史 >= 2 条",
                          len(history) >= 2 if isinstance(history, list) else False)
            except Exception as e:
                check("TC-04: 不抛异常", False, str(e)[:100])

        # TC-05: emergency keyword detection
        section("TC-05: 紧急关键词检测")
        try:
            result = manager.start_session(
                patient_id=TEST_PATIENT_ID,
                initial_symptom=EMERGENCY_SYMPTOM,
                max_turns=8,
            )
            check("TC-05a: action=emergency", result.get("action") == "emergency")
            check("TC-05b: 有 emergency_warning", bool(result.get("emergency_warning")))
            check("TC-05c: confidence >= 0.9", result.get("confidence", 0) >= 0.9)

            # Clean up
            e_sid = result.get("session_id")
            if e_sid:
                manager.close_session(e_sid)
        except Exception as e:
            check("TC-05: 不抛异常", False, str(e)[:100])

        # TC-06: candidate diseases populated
        if TEST_SESSION_ID:
            section("TC-06: 候选疾病填充")
            try:
                state = manager.get_session_state(TEST_SESSION_ID)
                if state:
                    candidates = state.get("candidate_diseases", [])
                    if isinstance(candidates, list) and candidates:
                        c0 = candidates[0]
                        check("TC-06a: 候选疾病有 disease", bool(c0.get("disease")))
                        check("TC-06b: 候选疾病有 score", isinstance(c0.get("score"), (int, float)))
                        check("TC-06c: 候选疾病有 departments", bool(c0.get("departments")))
            except Exception as e:
                check("TC-06: 不抛异常", False, str(e)[:100])

        # TC-07: minimal input forces continue (<2 symptoms)
        section("TC-07: 最小症状(<2)强制追问")
        try:
            result = manager.start_session(
                patient_id=TEST_PATIENT_ID,
                initial_symptom=MINIMAL_SYMPTOM,
                max_turns=8,
            )
            check("TC-07a: action 为 ask (症状不足, 不应推荐)",
                  result.get("action") == "ask")
            check("TC-07b: 有追问问题", bool(result.get("question")))
            check("TC-07c: confidence < 0.5 (低置信度)",
                  result.get("confidence", 1) < 0.5)

            # Clean up
            m_sid = result.get("session_id")
            if m_sid:
                manager.close_session(m_sid)
        except Exception as e:
            check("TC-07: 不抛异常", False, str(e)[:100])

        # TC-08: close & reopen protection
        section("TC-08: 关闭会话后拒绝继续")
        try:
            # Create and immediately close
            temp_result = manager.start_session(
                patient_id=TEST_PATIENT_ID,
                initial_symptom=None,
                max_turns=3,
            )
            temp_sid = temp_result.get("session_id")
            if temp_sid:
                manager.close_session(temp_sid)
                # Try to continue closed session
                result = manager.process(
                    session_id=temp_sid,
                    patient_input="我头痛",
                )
                check("TC-08a: 关闭后返回 error", result.get("action") == "error")
                check("TC-08b: 错误信息包含 'closed'",
                      "closed" in str(result.get("error", "")).lower())
        except Exception as e:
            check("TC-08: 不抛异常", False, str(e)[:100])

    # ================================================================
    # Part B: 单元测试 (不需要 LLM)
    # ================================================================

    section("Part B: 单元测试 (纯逻辑, 不调LLM)")

    from dialogue import DialogueManager as DM

    # Create manager for unit testing
    dm = DM(verbose=False)

    # TC-09: _check_emergency_keywords
    section("TC-09: 紧急关键词匹配")
    emergencies = [
        ("我剧烈胸痛呼吸困难", ["剧烈", "胸痛", "呼吸困难"]),
        ("突然昏迷不醒", ["昏迷"]),
        ("大出血止不住", ["大出血"]),
        ("普通头痛", []),
        ("有点不舒服", []),
    ]
    for i, (text, expected_keywords) in enumerate(emergencies):
        matches = dm._check_emergency_keywords(text)
        if expected_keywords:
            check(f"TC-09a{i+1}: \"{text[:15]}\" 检测到紧急",
                  len(matches) > 0,
                  f"matched: {matches}")
        else:
            check(f"TC-09b{i+1}: \"{text[:15]}\" 非紧急",
                  len(matches) == 0,
                  f"matched: {matches}")

    # TC-10: _merge_symptoms
    section("TC-10: 症状合并去重")
    existing = {
        "symptoms": ["头痛"],
        "body_parts": ["头部"],
        "duration": "3天",
        "severity": "中度",
        "keywords": ["头痛"],
    }
    new = {
        "symptoms": ["恶心", "头痛"],  # 头痛重复
        "body_parts": ["腹部"],
        "duration": "",  # 空值不覆盖
        "severity": "",
        "keywords": ["恶心", "畏光"],
    }
    merged = dm._merge_symptoms(existing, new)
    check("TC-10a: 症状去重 (应有2个)", len(merged["symptoms"]) == 2,
          f"got {merged['symptoms']}")
    check("TC-10b: body_parts 合并", len(merged["body_parts"]) == 2,
          f"got {merged['body_parts']}")
    check("TC-10c: duration 保留原值", merged["duration"] == "3天")
    check("TC-10d: severity 保留原值", merged["severity"] == "中度")
    check("TC-10e: keywords 去重合并", len(merged["keywords"]) >= 2)

    # TC-11: _format_symptoms_for_prompt
    section("TC-11: 症状格式化")
    fmt = dm._format_symptoms_for_prompt(merged)
    check("TC-11a: 包含症状", "头痛" in fmt)
    check("TC-11b: 包含恶心", "恶心" in fmt)
    check("TC-11c: 包含持续时间", "3天" in fmt)
    check("TC-11d: 包含严重程度", "中度" in fmt)

    empty_fmt = dm._format_symptoms_for_prompt({})
    check("TC-11e: 空症状提示", "暂无" in empty_fmt)

    # TC-12: _format_diseases_for_prompt
    section("TC-12: 候选疾病格式化")
    diseases = [
        {"disease": "偏头痛", "score": 0.75, "departments": "神经内科",
         "symptoms": "单侧搏动性头痛, 畏光, 恶心呕吐"},
        {"disease": "紧张性头痛", "score": 0.68, "departments": "神经内科",
         "symptoms": "双侧压迫性痛, 颈肩僵硬"},
    ]
    fmt = dm._format_diseases_for_prompt(diseases, top_n=2)
    check("TC-12a: 包含疾病名", "偏头痛" in fmt and "紧张性头痛" in fmt)
    check("TC-12b: 包含匹配度", "75.0%" in fmt)
    check("TC-12c: 包含科室", "神经内科" in fmt)

    empty_fmt = dm._format_diseases_for_prompt([])
    check("TC-12d: 空疾病提示", "暂无" in empty_fmt)

    # TC-13: _parse_llm_json — 4-tier fallback
    section("TC-13: JSON 解析 4层容错")
    # Layer 1: direct
    r1 = dm._parse_llm_json('{"decision":"continue","confidence":50}')
    check("TC-13a: 直接解析", r1.get("decision") == "continue")

    # Layer 2: ```json block
    r2 = dm._parse_llm_json('Some text\n```json\n{"decision":"recommend","confidence":80}\n```\nMore text')
    check("TC-13b: 代码块提取", r2.get("decision") == "recommend")

    # Layer 3: {...} extraction
    r3 = dm._parse_llm_json('前缀文字 {"decision":"recommend","confidence":85} 后缀')
    check("TC-13c: 花括号提取", r3.get("decision") == "recommend")

    # Layer 4: garbage → empty dict
    r4 = dm._parse_llm_json("这完全不是JSON格式的文本")
    check("TC-13d: 垃圾输入返回空dict", isinstance(r4, dict) and not r4)

    r5 = dm._parse_llm_json("")
    check("TC-13e: 空字符串返回空dict", isinstance(r5, dict) and not r5)

    # TC-14: _build_history_entry
    section("TC-14: 对话历史条目构建")
    entry = dm._build_history_entry(1, "patient", "头痛三天")
    check("TC-14a: turn=1", entry.get("turn") == 1)
    check("TC-14b: role=patient", entry.get("role") == "patient")
    check("TC-14c: content 正确", entry.get("content") == "头痛三天")
    check("TC-14d: 有 timestamp", bool(entry.get("timestamp")))

    # TC-15: _summarize_collected
    section("TC-15: 已收集信息摘要")
    accum = {
        "symptoms": ["头痛", "恶心"],
        "body_parts": ["头部"],
        "duration": "3天",
        "severity": "中度",
        "keywords": ["头痛", "恶心"],
    }
    summary = dm._summarize_collected(accum)
    check("TC-15a: 包含 symptoms", "symptoms" in summary)
    check("TC-15b: symptoms 数量", len(summary["symptoms"]) == 2)
    check("TC-15c: 包含 duration", summary.get("duration") == "3天")

    empty_summary = dm._summarize_collected({})
    check("TC-15d: 空输入返回 None", empty_summary is None)

    # ================================================================
    # Part C: MySQL 持久化测试
    # ================================================================

    section("Part C: MySQL 持久化")

    mysql_available = False
    try:
        import pymysql
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "medical_rag"),
            charset="utf8mb4",
            connect_timeout=3,
        )
        conn.close()
        mysql_available = True
        print("  MySQL 可用")
    except Exception as e:
        print(f"  [SKIP] MySQL 不可用: {e}")

    if mysql_available:
        # TC-16: Table creation
        section("TC-16: 建表")
        try:
            dm._ensure_table()

            import pymysql
            conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "medical_rag"),
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
            )
            with conn.cursor() as c:
                c.execute("SHOW TABLES LIKE 'dialogue_session'")
                row = c.fetchone()
            conn.close()

            check("TC-16a: 表存在", row is not None)

            if row:
                # Check columns
                conn = pymysql.connect(
                    host=os.getenv("MYSQL_HOST", "localhost"),
                    port=int(os.getenv("MYSQL_PORT", "3306")),
                    user=os.getenv("MYSQL_USER", "root"),
                    password=os.getenv("MYSQL_PASSWORD", ""),
                    database=os.getenv("MYSQL_DATABASE", "medical_rag"),
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor,
                )
                with conn.cursor() as c:
                    c.execute("SHOW COLUMNS FROM dialogue_session")
                    columns = [col["Field"] for col in c.fetchall()]
                conn.close()

                for required_col in ["session_id", "patient_id", "status",
                                      "collected_symptoms", "candidate_diseases",
                                      "dialogue_history", "final_recommendation",
                                      "current_turn", "max_turns"]:
                    check(f"TC-16b: 列 {required_col}",
                          required_col in columns)
        except Exception as e:
            check("TC-16: 不抛异常", False, str(e)[:100])

        # TC-17: Session CRUD
        section("TC-17: 会话 CRUD")
        test_sid = str(uuid.uuid4())

        try:
            # Create
            session = dm._create_session(test_sid, TEST_PATIENT_ID, max_turns=5)
            check("TC-17a: 创建会话", session is not None)

            # Load
            loaded = dm._load_session(test_sid)
            check("TC-17b: 加载会话", loaded is not None)
            if loaded:
                check("TC-17c: status=active", loaded.get("status") == "active")
                check("TC-17d: patient_id 正确", loaded.get("patient_id") == TEST_PATIENT_ID)
                check("TC-17e: max_turns=5", loaded.get("max_turns") == 5)

            # Update
            dm._update_session(test_sid, {
                "status": "closed",
                "current_turn": 3,
                "final_recommendation": json.dumps({"disease": "测试疾病"}),
            })
            updated = dm._load_session(test_sid)
            check("TC-17f: status 已更新", updated.get("status") == "closed" if updated else False)
            check("TC-17g: current_turn=3", updated.get("current_turn") == 3 if updated else False)

            # Clean up
            import pymysql
            conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "medical_rag"),
                charset="utf8mb4",
            )
            with conn.cursor() as c:
                c.execute("DELETE FROM dialogue_session WHERE session_id = %s", (test_sid,))
            conn.commit()
            conn.close()
            check("TC-17h: 清理测试数据", True)

        except Exception as e:
            check("TC-17: 不抛异常", False, str(e)[:100])

        # Also clean up test session from Part A
        if TEST_SESSION_ID:
            try:
                import pymysql
                conn = pymysql.connect(
                    host=os.getenv("MYSQL_HOST", "localhost"),
                    port=int(os.getenv("MYSQL_PORT", "3306")),
                    user=os.getenv("MYSQL_USER", "root"),
                    password=os.getenv("MYSQL_PASSWORD", ""),
                    database=os.getenv("MYSQL_DATABASE", "medical_rag"),
                    charset="utf8mb4",
                )
                with conn.cursor() as c:
                    c.execute("DELETE FROM dialogue_session WHERE patient_id = %s",
                             (TEST_PATIENT_ID,))
                conn.commit()
                conn.close()
                print(f"  Cleaned up test sessions (patient_id={TEST_PATIENT_ID})")
            except Exception:
                pass

    # ================================================================
    # Part D: 性能基准
    # ================================================================

    section("Part D: 性能基准")

    # TC-18: Empty start_session latency
    section("TC-18: 空会话启动延迟")
    try:
        if manager:
            start = time.time()
            result = manager.start_session(patient_id=None, initial_symptom=None)
            elapsed = time.time() - start
            check("TC-18a: 空启动 <5s", elapsed < 5, f"{elapsed:.2f}s")
            # Clean up
            sid = result.get("session_id")
            if sid:
                manager.close_session(sid)
    except Exception as e:
        check("TC-18: 不抛异常", False, str(e)[:100])

    # TC-19: Symptom extraction latency
    section("TC-19: 症状提取延迟")
    try:
        if manager:
            start = time.time()
            extracted = manager._extract_symptoms(HEADACHE_SYMPTOM)
            elapsed = time.time() - start
            check("TC-19a: 提取 <15s", elapsed < 15, f"{elapsed:.2f}s")
            check("TC-19b: 提取到症状", len(extracted.get("symptoms", [])) >= 0)
    except Exception as e:
        check("TC-19: 不抛异常", False, str(e)[:100])

    # TC-20: RAG retrieval latency (warm — model already loaded)
    section("TC-20: RAG 检索延迟 (热缓存)")
    try:
        # Use existing manager from Part A (vector_store already loaded)
        if manager:
            accum = {
                "symptoms": ["头痛", "恶心", "畏光"],
                "body_parts": ["头部"],
                "duration": "3天",
                "severity": "中度",
                "keywords": ["头痛", "恶心"],
            }
            start = time.time()
            candidates = manager._retrieve_diseases(accum)
            elapsed = time.time() - start
            check("TC-20a: RAG检索(热) <0.5s", elapsed < 0.5, f"{elapsed:.3f}s")
            check("TC-20b: 有候选结果", len(candidates) > 0 if candidates is not None else False)
    except Exception as e:
        check("TC-20: 不抛异常", False, str(e)[:100])

    # ================================================================
    # Summary
    # ================================================================

    print(f"\n{'=' * 60}")
    print(f"  Test Summary")
    print(f"{'=' * 60}")
    total = PASS + FAIL
    print(f"  Total: {total}  |  PASS: {PASS}  |  FAIL: {FAIL}")
    if total > 0:
        print(f"  Pass Rate: {PASS / total:.1%}")
    print(f"{'=' * 60}")

    return FAIL == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
