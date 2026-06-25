"""
api_server.py
FastAPI 服务 — Java 后端与 Python AI 引擎之间的 HTTP 网关

这是 Java Spring Boot 调用所有 AI 功能的唯一入口。
所有接口返回统一 JSON 格式: {"code": 200, "message": "success", "data": {...}}

启动方式:
    cd "d:/medic project"
    uvicorn rag-db.src.api_server:app --host 0.0.0.0 --port 8000 --reload

    # 生产模式 (多worker):
    uvicorn rag-db.src.api_server:app --host 0.0.0.0 --port 8000 --workers 4

端点清单:
    GET  /api/rag/health              — 健康检查
    POST /api/rag/search              — 智能导诊科室推荐 (核心)
    POST /api/rag/symptom/analyze     — 症状结构化和优化 (不检索)
    POST /api/rag/emr/extract         — 病历要素提取 → medical_record 字段
    POST /api/rag/assist/info         — AI辅助问诊提示
    POST /api/rag/diseases/search     — 仅疾病检索 (不经过LLM)
    GET  /api/rag/departments         — 全部科室列表
    GET  /api/rag/department/{name}   — 科室详情
    POST /api/rag/feedback            — 用户反馈收集
"""

import os
import sys
import time
import traceback
from typing import Optional

# Ensure src/ is on the path for imports
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(_src_dir, "..", "..", ".env"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ============================================================
# App initialization
# ============================================================

app = FastAPI(
    title="RAG 医疗智能导诊 AI 引擎",
    description="提供症状科室推荐 / 病历要素提取 / AI辅助问诊 等 AI 能力",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow Java backend and dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global startup timestamp
_START_TIME = time.time()

# Lazy-loaded singletons (initialized on first request)
_pipeline = None
_emr_processor = None
_vector_store = None


def get_pipeline():
    """Get or create RAGPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        from deepseek_client import RAGPipeline
        _pipeline = RAGPipeline(
            reranker_enabled=True,
            optimizer_mode="hybrid",
            verbose=False,
        )
    return _pipeline


def get_emr_processor():
    """Get or create EMRProcessor singleton."""
    global _emr_processor
    if _emr_processor is None:
        from emr_extractor import EMRProcessor
        _emr_processor = EMRProcessor(verbose=False)
    return _emr_processor


def get_vector_store():
    """Get or create VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        from query_engine import VectorStore
        _vector_store = VectorStore()
    return _vector_store


# ============================================================
# Middleware: request timing + error handling
# ============================================================

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add X-Process-Time header and catch unhandled errors."""
    start = time.time()
    try:
        response = await call_next(request)
        response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.1f}ms"
        return response
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": "AI引擎内部错误",
                "detail": str(e),
                "latency_ms": round(elapsed, 1),
            },
        )


# ============================================================
# Health check
# ============================================================

@app.get("/api/rag/health", tags=["System"])
def health_check():
    """
    健康检查 — Java 启动时探测 Python 服务是否就绪。

    Spring Boot 可在 @PostConstruct 中调用此接口确认 AI 引擎可用。
    """
    services = {}

    # Check VectorStore (ChromaDB)
    try:
        store = get_vector_store()
        stats = store.get_stats()
        services["vector_store"] = {
            "status": "ok",
            "collections": stats["collections"],
        }
    except Exception as e:
        services["vector_store"] = {"status": "error", "error": str(e)}

    # Check LLM (DeepSeek API)
    try:
        pipeline = get_pipeline()
        hc = pipeline.llm.health_check()
        services["llm"] = hc
    except Exception as e:
        services["llm"] = {"status": "error", "error": str(e)}

    # Check EMR Processor
    try:
        emr = get_emr_processor()
        emr_hc = emr.health_check()
        services["emr_extractor"] = emr_hc
    except Exception as e:
        services["emr_extractor"] = {"status": "error", "error": str(e)}

    # Overall status
    all_ok = all(s.get("status") == "ok" for s in services.values())
    overall = "ok" if all_ok else "degraded"

    return {
        "code": 200,
        "status": overall,
        "version": "2.0.0",
        "services": services,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# Core: 智能导诊科室推荐
# ============================================================

@app.post("/api/rag/search", tags=["Core — 智能导诊"])
def search_department(request: dict):
    """
    完整科室推荐 (四阶段 Pipeline): 查询优化 → 向量粗排 → Cross-Encoder 精排 → LLM 推理。

    这是患者端"智能导诊"功能的核心 API。
    输入患者症状描述 (支持口语化/方言)，返回推荐科室和推理依据。

    请求体:
        {"query": "头痛发热咳嗽流鼻涕", "top_k": 5}

    响应:
        {"code": 200, "data": {..., "primary_recommendation": {...}}}
    """
    query = request.get("query", "").strip()
    top_k = request.get("top_k", 5)

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    pipeline = get_pipeline()
    result = pipeline.query(query, top_k=top_k)

    rag = result.get("rag_results", {})
    rec = result.get("recommendation", {})
    primary = rag.get("primary_recommendation") or {}

    return {
        "code": 200,
        "data": {
            "query": result["query"],
            "search_query": result.get("search_query"),
            "disease_results": rag.get("disease_results", [])[:5],
            "symptom_direct": rag.get("symptom_direct", []),
            "all_departments": rag.get("all_departments", []),
            "primary_recommendation": {
                "department": primary.get("department"),
                "disease": primary.get("disease"),
                "confidence": primary.get("confidence"),
                "reasoning": primary.get("reasoning"),
            } if primary else None,
            "reranked": rag.get("reranked", False),
            # LLM 生成层
            "llm_department": rec.get("department"),
            "llm_disease": rec.get("disease"),
            "llm_confidence": rec.get("confidence"),
            "llm_reasoning": rec.get("reasoning"),
            "llm_suggestion": rec.get("suggestion"),
            "alternative_departments": rec.get("alternative_departments", []),
            "emergency_warning": rec.get("emergency_warning", False),
            # 查询优化
            "query_optimization": result.get("query_optimization"),
        },
        "metadata": {
            "model": rec.get("model"),
            "usage": rec.get("usage"),
        },
    }


# ============================================================
# 症状结构化分析 (不检索, 仅优化)
# ============================================================

@app.post("/api/rag/symptom/analyze", tags=["Core — 智能导诊"])
def analyze_symptoms(request: dict):
    """
    仅执行查询优化和症状结构化提取，不进行向量检索和 LLM 生成。

    适用于前端实时展示标准化后的症状，让用户确认后再正式检索。
    响应快速 (~1ms rule模式, ~500ms LLM模式)。

    请求体:
        {"query": "肚子疼拉稀想吐没胃口"}

    响应:
        {"code": 200, "data": {"symptoms": ["腹痛","腹泻","恶心","食欲不振"], ...}}
    """
    query = request.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    pipeline = get_pipeline()
    result = pipeline.optimize_query(query)

    # Also extract symptoms via LLM for richer analysis
    try:
        llm_symptoms = pipeline.llm.extract_symptoms(query)
    except Exception:
        llm_symptoms = {"error": "LLM symptom extraction unavailable"}

    return {
        "code": 200,
        "data": {
            "original_query": result.get("original_query", query),
            "optimized_query": result.get("optimized_query", query),
            "symptoms": result.get("symptoms", []),
            "body_parts": result.get("body_parts", []),
            "severity": result.get("severity", "未知"),
            "has_emergency_signals": result.get("has_emergency_signals", False),
            "normalization_note": result.get("normalization_note", ""),
            "llm_analysis": llm_symptoms,
        },
    }


# ============================================================
# EMR: 病历要素提取
# ============================================================

@app.post("/api/rag/emr/extract", tags=["EMR — 病历提取"])
def extract_medical_record(request: dict):
    """
    从患者症状描述和健康档案中提取结构化病历要素。

    对应概要设计 medical_record 实体的 8 个核心字段:
      chief_complaint, present_illness, past_history, allergy_history,
      family_history, medication_hist, diagnosis, treatment

    Java 端调用时机:
      医生进入主诉详情页 → 自动调用本接口生成病历草稿。
      AI 结果写入 medical_record 表 (is_archived=0, 草稿状态),
      医生可在 Web 端编辑修改后归档。

    请求体:
        {
            "symptom_text": "近3天反复发热...",
            "health_record": {"past_illness": "高血压", "allergy": "青霉素", ...},
            "patient_info": {"patient_id": 1, "age": 65, "gender": "男"},
            "use_rag": true,
            "consult_id": 123
        }
    """
    symptom_text = request.get("symptom_text", "").strip()
    health_record = request.get("health_record")
    patient_info = request.get("patient_info")
    use_rag = request.get("use_rag", True)

    if not symptom_text:
        raise HTTPException(status_code=400, detail="symptom_text 不能为空")

    processor = get_emr_processor()
    result = processor.extract_medical_record(
        symptom_text=symptom_text,
        health_record=health_record,
        patient_info=patient_info,
        use_rag=use_rag,
    )

    if result.error:
        return {
            "code": 500,
            "message": "病历提取失败",
            "data": result.to_dict(),
            "metadata": {"error": result.error, "latency_ms": result.latency_ms},
        }

    return result.to_api_response()


# ============================================================
# Assist: AI辅助问诊提示
# ============================================================

@app.post("/api/rag/assist/info", tags=["EMR — 病历提取"])
def generate_assist_info(request: dict):
    """
    生成 AI 辅助问诊提示，为接诊医生提供临床决策支持。

    输出 5 类辅助信息:
      - 病情摘要 (disease_summary)
      - 追问问题清单 (follow_up_questions)
      - 鉴别诊断方向 (differential_diagnosis)
      - 建议检查项目 (suggested_exams)
      - 用药方向建议 (medication_suggestions)
      - 转诊建议 (referral_suggestions)

    请求体: 同 /api/rag/emr/extract
    """
    symptom_text = request.get("symptom_text", "").strip()
    health_record = request.get("health_record")
    patient_info = request.get("patient_info")
    use_rag = request.get("use_rag", True)

    if not symptom_text:
        raise HTTPException(status_code=400, detail="symptom_text 不能为空")

    processor = get_emr_processor()
    result = processor.generate_assist_info(
        symptom_text=symptom_text,
        health_record=health_record,
        patient_info=patient_info,
        use_rag=use_rag,
    )

    return result.to_api_response()


# ============================================================
# 疾病检索 (不经过LLM, 快速)
# ============================================================

@app.post("/api/rag/diseases/search", tags=["Core — 智能导诊"])
def search_diseases(request: dict):
    """
    仅执行向量疾病检索，不调用 LLM 生成层。

    适用于: 快速查看可能的疾病列表，无需完整导诊推理。
    响应时间 < 50ms (纯向量计算)。
    """
    query = request.get("query", "").strip()
    top_k = request.get("top_k", 5)

    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    store = get_vector_store()
    diseases = store.search_disease(query, top_k=min(top_k, 20))

    return {
        "code": 200,
        "data": {
            "query": query,
            "diseases": diseases,
            "count": len(diseases),
        },
    }


# ============================================================
# 科室信息查询
# ============================================================

@app.get("/api/rag/departments", tags=["Reference — 参考数据"])
def list_departments():
    """
    获取全部 54 个科室列表及其诊疗范围。

    适用于: 科室选择器下拉框、科室详情页。
    """
    store = get_vector_store()
    # Use empty query with top_k=54 to get all departments
    dept_results = store.search_department("", top_k=60)

    # Build simplified list
    departments = []
    for d in dept_results:
        departments.append({
            "department": d["department"],
            "disease_count": d["disease_count"],
            "common_symptoms": d["common_symptoms"],
            "sample_diseases": d["sample_diseases"],
        })

    # Remove duplicates by department name
    seen = set()
    unique = []
    for d in departments:
        if d["department"] not in seen:
            seen.add(d["department"])
            unique.append(d)

    return {
        "code": 200,
        "departments": sorted(unique, key=lambda x: x["department"]),
        "total": len(unique),
    }


@app.get("/api/rag/department/{name}", tags=["Reference — 参考数据"])
def get_department_detail(name: str):
    """
    查询指定科室的诊疗范围、常见症状和代表性疾病。

    GET /api/rag/department/呼吸内科
    """
    store = get_vector_store()
    results = store.search_department(name, top_k=3)

    if not results:
        raise HTTPException(status_code=404, detail=f"科室 '{name}' 未找到")

    best = results[0]
    return {
        "code": 200,
        "data": {
            "department": best["department"],
            "disease_count": best["disease_count"],
            "common_symptoms": best["common_symptoms"],
            "sample_diseases": best["sample_diseases"],
            "score": best["score"],
        },
    }


# ============================================================
# 用户反馈
# ============================================================

@app.post("/api/rag/feedback", tags=["System"])
def submit_feedback(request: dict):
    """
    收集用户对推荐结果的反馈。

    请求体:
        {
            "query": "头痛发热",
            "consult_id": 123,
            "recommended_department": "呼吸内科",
            "feedback": "negative",       # positive / negative / neutral
            "actual_department": "神经内科",  # negative时填写
            "comment": "实际是偏头痛"
        }

    Java 端可同时写入 feedback 表 + 触发离线分析。
    """
    query = request.get("query", "")
    feedback_type = request.get("feedback", "neutral")

    if feedback_type not in ("positive", "negative", "neutral"):
        raise HTTPException(status_code=400, detail="feedback 必须是 positive/negative/neutral")

    # For now: log to stdout + append to feedback log file
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "consult_id": request.get("consult_id"),
        "recommended_department": request.get("recommended_department"),
        "feedback": feedback_type,
        "actual_department": request.get("actual_department"),
        "comment": request.get("comment"),
    }

    # Append to feedback log (JSONL format) in project root
    import json
    _log_path = os.path.join(_src_dir, "..", "feedback_log.jsonl")
    try:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Silent fail — don't break the API for logging

    print(f"[Feedback] {feedback_type}: '{query[:40]}' → {log_entry.get('recommended_department')}")

    return {
        "code": 200,
        "message": "反馈已记录",
        "data": log_entry,
    }


# ============================================================
# 404 handler
# ============================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"code": 404, "message": "接口不存在", "detail": str(exc)},
    )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 65)
    print("  RAG 医疗智能导诊 AI 引擎 — FastAPI 服务")
    print("=" * 65)
    print(f"  API 文档:  http://localhost:8000/api/docs")
    print(f"  健康检查:  http://localhost:8000/api/rag/health")
    print(f"  端点数量:  9")
    print()

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
