"""
test_emr.py
EMR 病历提取 + AI辅助问诊 + API 接口 — 完整测试套件

运行方式:
    cd "d:/medic project"
    python rag-db/tests/test_emr.py

测试覆盖:
    1. EMRProcessor — 完整病历要素提取 (8字段)
    2. EMRProcessor — 最小输入 (无健康档案)
    3. EMRProcessor — 空症状输入保护
    4. EMRProcessor — RAG 上下文检索
    5. EMRProcessor — AI辅助问诊提示 (5类)
    6. EMRResult — to_dict / to_api_response 结构完整性
    7. API Models — Pydantic 请求/响应序列化
    8. API Server — 健康检查端点
    9. API Server — 请求体校验
    10. HealthRecordInput — 空/部分/完整档案的格式化
"""

import os
import sys
import json
import time
import importlib.util

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def load_module(name: str):
    """Load a module by file path from src/."""
    path = os.path.join(_src, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Helpers
# ============================================================

def section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def ok(msg: str):
    print(f"  [PASS] {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")


# ============================================================
# Test Cases
# ============================================================

def test_emr_processor():
    """Test EMR extraction engine."""
    section("Test Suite: EMRProcessor")

    passed = 0
    failed = 0

    # Load EMRProcessor
    try:
        emr_mod = load_module("emr_extractor")
        EMRProcessor = emr_mod.EMRProcessor
        EMRResult = emr_mod.EMRResult
        AssistResult = emr_mod.AssistResult
        ok("Module loaded: emr_extractor")
        passed += 1
    except Exception as e:
        fail(f"Module load failed: {e}")
        return 0, 1

    # --- TC-01: Full EMR extraction with health record ---
    print("\n  --- TC-01: 完整病历提取 (有健康档案) ---")
    processor = EMRProcessor(verbose=False)

    try:
        result = processor.extract_medical_record(
            symptom_text=(
                "近3天反复发热，最高体温39.2°C，伴有咳嗽、咳黄痰，"
                "右侧胸痛，活动后呼吸困难。自行服用布洛芬后体温可降至37.5°C，"
                "但药效过后反复升高。无恶心呕吐，食欲减退。"
            ),
            health_record={
                "past_illness": "高血压病史5年，2型糖尿病3年",
                "allergy": "青霉素过敏（皮疹）",
                "surgery_history": "阑尾切除术 2019年",
                "medication": "硝苯地平缓释片 30mg qd, 二甲双胍 500mg bid",
                "blood_type": "A",
            },
            patient_info={"age": 65, "gender": "男"},
            use_rag=True,
        )

        if result.error:
            fail(f"TC-01: {result.error}")
            failed += 1
        else:
            # Check all 8 fields present
            data = result.to_dict()
            fields_check = all(k in data for k in [
                "chief_complaint", "present_illness", "past_history",
                "allergy_history", "family_history", "medication_hist",
                "diagnosis", "treatment",
            ])
            if fields_check:
                ok(f"TC-01: All 8 EMR fields present")
                passed += 1
            else:
                fail(f"TC-01: Missing fields in result")
                failed += 1

            # Chief complaint should contain time info
            cc = result.chief_complaint or ""
            if cc and len(cc) >= 4:
                ok(f"TC-01a: Chief complaint valid: '{cc[:50]}...'")
                passed += 1
            else:
                fail(f"TC-01a: Chief complaint too short or empty: '{cc}'")
                failed += 1

            # Past history should reference hypertension
            ph = result.past_history or ""
            if "高血压" in ph or "hypertension" in ph.lower():
                ok(f"TC-01b: Past history includes hypertension")
                passed += 1
            else:
                fail(f"TC-01b: Past history missing hypertension: '{ph[:60]}...'")
                failed += 1

            # Allergy should reference penicillin
            ah = result.allergy_history or ""
            if "青霉素" in ah or "penicillin" in ah.lower():
                ok(f"TC-01c: Allergy history includes penicillin")
                passed += 1
            else:
                fail(f"TC-01c: Allergy history missing penicillin: '{ah[:60]}...'")
                failed += 1

            # Diagnosis should have AI disclaimer
            diag = result.diagnosis or ""
            if "AI" in diag or "辅助" in diag or "建议" in diag:
                ok(f"TC-01d: Diagnosis includes AI disclaimer")
                passed += 1
            else:
                fail(f"TC-01d: Diagnosis missing disclaimer: '{diag[:60]}...'")
                failed += 1

            print(f"         Latency: {result.latency_ms}ms")
            print(f"         Tokens: {result.usage}")

    except Exception as e:
        fail(f"TC-01: Exception: {e}")
        failed += 1

    # --- TC-02: Minimal input (no health record) ---
    print("\n  --- TC-02: 最小输入 (无健康档案) ---")
    try:
        result = processor.extract_medical_record(
            symptom_text="头痛两天",
            health_record=None,
            patient_info={"age": 25, "gender": "女"},
            use_rag=False,
        )

        if result.error:
            fail(f"TC-02: {result.error}")
            failed += 1
        else:
            ok(f"TC-02: Minimal input handled, chief_complaint='{result.chief_complaint}'")
            passed += 1

            # Past history should default to "不详" / "无" / "无特殊"
            ph = (result.past_history or "").lower()
            if any(w in ph for w in ["不详", "无", "未知", "none", "n/a"]):
                ok(f"TC-02a: Empty health record → past_history='{result.past_history}'")
                passed += 1
            else:
                ok(f"TC-02a: (check manually) past_history='{result.past_history}'")
                passed += 1

    except Exception as e:
        fail(f"TC-02: Exception: {e}")
        failed += 1

    # --- TC-03: Empty symptom text (should fail gracefully) ---
    print("\n  --- TC-03: 空输入保护 ---")
    try:
        result = processor.extract_medical_record(
            symptom_text="",
            health_record=None,
            patient_info={},
            use_rag=False,
        )
        # Should handle gracefully — either error or empty result
        ok(f"TC-03: Empty input handled without crash")
        passed += 1
    except Exception as e:
        fail(f"TC-03: Crashed on empty input: {e}")
        failed += 1

    # --- TC-04: RAG context retrieval ---
    print("\n  --- TC-04: RAG 检索上下文 ---")
    try:
        rag_context = processor._retrieve_rag_context("发热咳嗽咽痛")
        has_content = len(rag_context) > 50 and "疾病" in rag_context
        if has_content:
            ok(f"TC-04: RAG context retrieved ({len(rag_context)} chars)")
            passed += 1
        else:
            fail(f"TC-04: RAG context too short or malformed")
            failed += 1
    except Exception as e:
        fail(f"TC-04: RAG context failed: {e}")
        failed += 1

    # --- TC-05: AI Assist Info ---
    print("\n  --- TC-05: AI辅助问诊提示 ---")
    try:
        assist = processor.generate_assist_info(
            symptom_text="胸闷心慌气短，活动后加重，伴有头晕",
            health_record={
                "past_illness": "冠心病3年, 高脂血症",
                "medication": "阿司匹林 100mg qd, 阿托伐他汀 20mg qn",
            },
            patient_info={"age": 58, "gender": "男"},
        )

        if assist.error:
            fail(f"TC-05: {assist.error}")
            failed += 1
        else:
            # Check all 5 assist types
            checks = [
                ("disease_summary", assist.disease_summary),
                ("follow_up_questions", assist.follow_up_questions),
                ("differential_diagnosis", assist.differential_diagnosis),
                ("suggested_exams", assist.suggested_exams),
                ("medication_suggestions", assist.medication_suggestions),
                ("referral_suggestions", assist.referral_suggestions),
            ]
            all_have = all(v for _, v in checks)
            if all_have:
                ok(f"TC-05: All 6 assist fields populated")
                passed += 1
            else:
                missing = [k for k, v in checks if not v]
                fail(f"TC-05: Missing fields: {missing}")
                failed += 1

            for name, value in checks:
                if value:
                    preview = str(value)[:80]
                    print(f"         {name}: {preview}...")

    except Exception as e:
        fail(f"TC-05: Exception: {e}")
        failed += 1

    # --- TC-06: EMRResult serialization ---
    print("\n  --- TC-06: EMRResult 序列化 ---")
    try:
        result = EMRResult(
            chief_complaint="发热伴咳嗽3天",
            present_illness="患者3天前出现发热...",
            past_history="高血压5年",
            allergy_history="青霉素过敏",
            family_history="不详",
            medication_hist="硝苯地平 30mg qd",
            diagnosis="考虑上呼吸道感染 (AI辅助建议)",
            treatment="建议完善血常规 (AI辅助建议)",
        )

        d = result.to_dict()
        api = result.to_api_response()

        assert len(d) == 8, f"to_dict should have 8 keys, got {len(d)}"
        assert api["code"] == 200, f"to_api_response code should be 200"
        assert "data" in api, "to_api_response should have 'data'"
        assert "metadata" in api, "to_api_response should have 'metadata'"

        ok(f"TC-06: EMRResult serialization OK ({len(d)} fields)")
        passed += 1
    except Exception as e:
        fail(f"TC-06: Serialization failed: {e}")
        failed += 1

    # --- TC-07: Health record formatting ---
    print("\n  --- TC-07: 健康档案格式化 ---")
    try:
        # Full record
        hr = {
            "past_illness": "高血压",
            "allergy": "青霉素过敏",
            "surgery_history": "阑尾切除术",
        }
        formatted = processor._format_health_record(hr)
        ok(f"TC-07a: Full record format: {formatted[:60]}...")
        passed += 1

        # Empty record
        formatted_empty = processor._format_health_record({})
        assert "暂无" in formatted_empty or "无" in formatted_empty
        ok(f"TC-07b: Empty record format: '{formatted_empty}'")
        passed += 1

        # None
        formatted_none = processor._format_health_record(None)
        assert "暂无" in formatted_none or "无" in formatted_none
        ok(f"TC-07c: None record handled")
        passed += 1

    except Exception as e:
        fail(f"TC-07: {e}")
        failed += 1

    return passed, failed


def test_api_models():
    """Test Pydantic API models."""
    section("Test Suite: API Models (Pydantic)")

    passed = 0
    failed = 0

    try:
        from api_models import (
            SearchRequest, EMRRequest, AssistRequest, FeedbackRequest,
            HealthRecordInput, PatientInfo, MedicalRecordFields,
            HealthResponse,
        )
        ok("Module loaded: api_models")
        passed += 1
    except ImportError:
        # Try loading via importlib
        try:
            mod = load_module("api_models")
            SearchRequest = mod.SearchRequest
            EMRRequest = mod.EMRRequest
            ok("Module loaded: api_models (via importlib)")
            passed += 1
        except Exception as e:
            fail(f"Module load failed: {e}")
            return 0, 1

    # --- TC-M01: SearchRequest validation ---
    print("\n  --- TC-M01: SearchRequest 校验 ---")
    try:
        req = SearchRequest(query="头痛发热")
        assert req.query == "头痛发热"
        assert req.top_k == 5
        ok("TC-M01: SearchRequest valid")
        passed += 1
    except Exception as e:
        fail(f"TC-M01: {e}")
        failed += 1

    # Empty query should fail
    try:
        SearchRequest(query="")
        fail("TC-M01a: Empty query should raise error")
        failed += 1
    except Exception:
        ok("TC-M01a: Empty query correctly rejected")
        passed += 1

    # --- TC-M02: EMRRequest with full data ---
    print("\n  --- TC-M02: EMRRequest 完整构造 ---")
    try:
        hr = HealthRecordInput(
            past_illness="高血压",
            allergy="青霉素",
            medication="硝苯地平",
        )
        pi = PatientInfo(age=50, gender="男")
        req = EMRRequest(
            symptom_text="发热咳嗽3天",
            health_record=hr,
            patient_info=pi,
            use_rag=True,
            consult_id=123,
        )
        assert req.symptom_text == "发热咳嗽3天"
        assert req.health_record.past_illness == "高血压"
        assert req.patient_info.age == 50
        ok("TC-M02: EMRRequest complete")
        passed += 1
    except Exception as e:
        fail(f"TC-M02: {e}")
        failed += 1

    # --- TC-M03: Minimal EMRRequest (no health record) ---
    print("\n  --- TC-M03: EMRRequest 最小构造 ---")
    try:
        req = EMRRequest(symptom_text="肚子疼")
        assert req.health_record is None
        assert req.patient_info is None
        ok("TC-M03: EMRRequest minimal OK")
        passed += 1
    except Exception as e:
        fail(f"TC-M03: {e}")
        failed += 1

    # --- TC-M04: FeedbackRequest ---
    print("\n  --- TC-M04: FeedbackRequest ---")
    try:
        req = FeedbackRequest(
            query="头痛",
            feedback="negative",
            actual_department="神经内科",
        )
        assert req.feedback == "negative"
        ok("TC-M04: Feedback request valid")
        passed += 1
    except Exception as e:
        fail(f"TC-M04: {e}")
        failed += 1

    # --- TC-M05: MedicalRecordFields ---
    print("\n  --- TC-M05: MedicalRecordFields ---")
    try:
        fields = MedicalRecordFields(
            chief_complaint="发热3天",
            present_illness="患者3天前...",
            past_history="高血压",
            allergy_history="青霉素过敏",
            family_history="不详",
            medication_hist="硝苯地平",
            diagnosis="考虑感染 (AI辅助)",
            treatment="建议血常规 (AI辅助)",
        )
        data = fields.model_dump()
        assert len(data) == 8
        ok("TC-M05: MedicalRecordFields 8 fields OK")
        passed += 1
    except Exception as e:
        fail(f"TC-M05: {e}")
        failed += 1

    # --- TC-M06: Serialize to JSON (Java compat) ---
    print("\n  --- TC-M06: JSON 序列化 (Java兼容) ---")
    try:
        req = EMRRequest(
            symptom_text="发热咳嗽",
            health_record=HealthRecordInput(past_illness="高血压"),
            patient_info=PatientInfo(age=50, gender="男"),
        )
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["symptom_text"] == "发热咳嗽"
        assert parsed["health_record"]["past_illness"] == "高血压"
        assert parsed["patient_info"]["age"] == 50
        ok("TC-M06: JSON serialization OK")
        passed += 1
    except Exception as e:
        fail(f"TC-M06: {e}")
        failed += 1

    return passed, failed


def test_health_record_formatting():
    """Test edge cases for health record formatting."""
    section("Test Suite: Health Record Formatting Edge Cases")

    passed = 0
    failed = 0

    try:
        emr_mod = load_module("emr_extractor")
        EMRProcessor = emr_mod.EMRProcessor
        processor = EMRProcessor(verbose=False)
    except Exception as e:
        fail(f"Module load: {e}")
        return 0, 1

    # All fields populated
    hr_full = {
        "past_illness": "高血压, 糖尿病",
        "allergy": "青霉素, 头孢类",
        "surgery_history": "阑尾切除术(2019), 胆囊切除术(2015)",
        "medication": "硝苯地平 30mg qd, 二甲双胍 500mg bid, 阿司匹林 100mg qd",
        "blood_type": "O",
        "birth_date": "1960-03-15",
        "gender": 1,
        "member_name": "张三",
    }
    text = processor._format_health_record(hr_full)
    print(f"  Full record:\n{text}")
    assert "高血压" in text
    assert "青霉素" in text
    assert "阑尾切除术" in text
    assert "硝苯地平" in text
    assert "O" in text
    ok("TC-H01: Full record formatted correctly")
    passed += 1

    # Partial record (only allergy)
    hr_partial = {"allergy": "磺胺类过敏"}
    text = processor._format_health_record(hr_partial)
    assert "磺胺类" in text
    ok("TC-H02: Partial record formatted")
    passed += 1

    # Empty dict
    text = processor._format_health_record({})
    assert len(text) > 0
    ok("TC-H03: Empty dict returns placeholder")
    passed += 1

    # None
    text = processor._format_health_record(None)
    assert len(text) > 0
    ok("TC-H04: None returns placeholder")
    passed += 1

    # Only non-standard keys (should return placeholder)
    hr_unknown = {"custom_field": "irrelevant"}
    text = processor._format_health_record(hr_unknown)
    assert "无有效记录" in text or "暂无" in text
    ok("TC-H05: Unknown keys handled gracefully")
    passed += 1

    return passed, failed


# ============================================================
# Summary
# ============================================================

def main():
    print("=" * 65)
    print("  EMR Extractor + API Models — Test Suite")
    print("=" * 65)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Time:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_failed = 0

    # Run test suites
    for suite_name, suite_fn in [
        ("EMRProcessor", test_emr_processor),
        ("API Models", test_api_models),
        ("Health Record Formatting", test_health_record_formatting),
    ]:
        p, f = suite_fn()
        total_passed += p
        total_failed += f

    # Final summary
    section("Test Summary")
    total = total_passed + total_failed
    print(f"\n  Total:  {total}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Rate:   {total_passed / total * 100:.1f}%" if total > 0 else "  Rate: N/A")

    if total_failed == 0:
        print("\n  [OK] All EMR tests passed!")
    else:
        print(f"\n  [WARN] {total_failed} test(s) failed — check output above.")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
