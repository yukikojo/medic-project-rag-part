"""
emr_extractor.py
EMR 病历要素结构化提取引擎 — AI 辅助问诊核心模块

接收患者症状描述、健康档案和 RAG 检索结果, 由 LLM 输出符合概要设计
medical_record 实体规范的 8 个核心字段:
  1. chief_complaint  — 主诉 (主要症状及持续时间)
  2. present_illness  — 现病史 (本次发病情况)
  3. past_history     — 既往史 (既往病史摘要)
  4. allergy_history  — 过敏史 (过敏药物/食物)
  5. family_history   — 家族史 (家族遗传疾病)
  6. medication_hist  — 用药史 (近期用药情况)
  7. diagnosis        — AI辅助诊断建议
  8. treatment        — AI辅助处理建议

架构位置:
  Java → POST /api/rag/emr/extract → EMRProcessor → DeepSeekClient + VectorStore
  Java → POST /api/rag/assist/info → EMRProcessor → 辅助问诊提示

使用示例:
    from emr_extractor import EMRProcessor

    processor = EMRProcessor()
    result = processor.extract_medical_record(
        symptom_text="近3天反复发热，最高39.2°C，咳嗽咳黄痰，右侧胸痛",
        health_record={"past_illness": "高血压5年", "allergy": "青霉素过敏"},
        patient_info={"age": 65, "gender": "男"}
    )
    print(result["chief_complaint"])  # "发热伴咳嗽咳痰3天"
    print(result["present_illness"])  # "患者3天前无明显诱因出现发热..."
"""

import os
import json
import time
from typing import Optional

from dotenv import load_dotenv as _load_dotenv

# Load .env from project root
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL = "deepseek-v4-flash"   # EMR extraction uses fast model


class EMRResult:
    """
    Typed container for structured EMR extraction results.

    Mirrors the medical_record entity fields from the 概要设计 (Section 5.8).
    All fields default to None when extraction fails or data is unavailable.
    """

    __slots__ = (
        "chief_complaint", "present_illness", "past_history",
        "allergy_history", "family_history", "medication_hist",
        "diagnosis", "treatment",
        "raw_response", "model", "usage", "latency_ms", "error",
    )

    def __init__(
        self,
        chief_complaint: Optional[str] = None,
        present_illness: Optional[str] = None,
        past_history: Optional[str] = None,
        allergy_history: Optional[str] = None,
        family_history: Optional[str] = None,
        medication_hist: Optional[str] = None,
        diagnosis: Optional[str] = None,
        treatment: Optional[str] = None,
        raw_response: Optional[str] = None,
        model: Optional[str] = None,
        usage: Optional[dict] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ):
        self.chief_complaint = chief_complaint
        self.present_illness = present_illness
        self.past_history = past_history
        self.allergy_history = allergy_history
        self.family_history = family_history
        self.medication_hist = medication_hist
        self.diagnosis = diagnosis
        self.treatment = treatment
        self.raw_response = raw_response
        self.model = model
        self.usage = usage
        self.latency_ms = latency_ms
        self.error = error

    def to_dict(self) -> dict:
        """Convert to dict matching medical_record table columns."""
        return {
            "chief_complaint": self.chief_complaint,
            "present_illness": self.present_illness,
            "past_history": self.past_history,
            "allergy_history": self.allergy_history,
            "family_history": self.family_history,
            "medication_hist": self.medication_hist,
            "diagnosis": self.diagnosis,
            "treatment": self.treatment,
        }

    def to_api_response(self) -> dict:
        """Full API response including metadata."""
        return {
            "code": 200,
            "data": self.to_dict(),
            "metadata": {
                "model": self.model,
                "latency_ms": self.latency_ms,
                "usage": self.usage,
                "error": self.error,
            },
        }


class AssistResult:
    """
    AI-assisted consultation hints (辅助问诊提示).

    Provides 5 types of clinical decision support hints
    for doctors during consultation.
    """

    __slots__ = (
        "follow_up_questions", "differential_diagnosis",
        "necessary_tests", "medication_suggestions",
        "referral_depts", "disease_summary",
        "raw_response", "model", "usage", "latency_ms", "error",
    )

    def __init__(
        self,
        follow_up_questions: Optional[list[str]] = None,
        differential_diagnosis: Optional[list[str]] = None,
        necessary_tests: Optional[list[str]] = None,
        medication_suggestions: Optional[list[str]] = None,
        referral_depts: Optional[list[str]] = None,
        disease_summary: Optional[str] = None,
        raw_response: Optional[str] = None,
        model: Optional[str] = None,
        usage: Optional[dict] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ):
        self.follow_up_questions = follow_up_questions or []
        self.differential_diagnosis = differential_diagnosis or []
        self.necessary_tests = necessary_tests or []
        self.medication_suggestions = medication_suggestions or []
        self.referral_depts = referral_depts or []
        self.disease_summary = disease_summary
        self.raw_response = raw_response
        self.model = model
        self.usage = usage
        self.latency_ms = latency_ms
        self.error = error

    def to_api_response(self) -> dict:
        """Full API response including metadata."""
        return {
            "code": 200,
            "data": {
                "disease_summary": self.disease_summary,
                "follow_up_questions": self.follow_up_questions,
                "differential_diagnosis": self.differential_diagnosis,
                "necessary_tests": self.necessary_tests,
                "medication_suggestions": self.medication_suggestions,
                "referral_depts": self.referral_depts,
            },
            "metadata": {
                "model": self.model,
                "latency_ms": self.latency_ms,
                "usage": self.usage,
                "error": self.error,
            },
        }


# ============================================================
# Prompt Templates
# ============================================================

EMR_SYSTEM_PROMPT = """你是一位资深临床医师，擅长从患者信息和症状描述中提取结构化病历要素。
你的任务是阅读患者的基本信息、健康档案和当前症状，按照标准病历格式输出结构化字段。

## 重要原则
1. **主诉 (chief_complaint)**: 用1-2句话概括患者本次就诊的核心问题，格式为「主要症状+持续时间」。例如「发热伴咳嗽咳痰3天」「反复上腹痛1周」。必须包含时间信息。
2. **现病史 (present_illness)**: 详细描述本次发病情况，包括起病诱因、症状演变过程、严重程度、伴随症状、已采取的措施及效果。按时间顺序叙述，50-150字。
3. **既往史 (past_history)**: 从健康档案中提取既往疾病史，排除与本次无关的陈旧信息。如果健康档案为空，填"无特殊"或"不详"。
4. **过敏史 (allergy_history)**: 从健康档案中提取药物/食物过敏信息。如果未提供，填"无已知过敏史"。
5. **家族史 (family_history)**: 如果健康档案中未提供家族遗传病史信息，填"不详"。不要凭空捏造。
6. **用药史 (medication_hist)**: 从健康档案中提取当前用药，同时注意患者自述中是否有自行服药情况，合并呈现。
7. **诊断 (diagnosis)**: 基于症状和RAG检索的疾病知识，给出最可能的初步诊断（1-2个），标注为「AI辅助建议，请以医师最终诊断为准」。
8. **处理意见 (treatment)**: 列出建议的检查项目和初步治疗方向，同样标注AI辅助性质。

## 关键约束
- 不要捏造用户未提供的信息
- 如果某字段确无可用数据，填写"不详"或"无"，不要留空
- 诊断和处理意见必须附带 AI 免责声明
- 保持专业医学术语，但同时要清晰易懂
- 所有输出使用中文"""

EMR_USER_MESSAGE_TEMPLATE = """## 患者基本信息
年龄: {age}
性别: {gender}

## 患者症状描述
{symptom_text}

## 患者健康档案
{health_record_text}

## 医学知识库参考 (RAG检索结果)
{rag_context}

请根据以上信息，按 JSON 格式输出结构化病历要素。
严格遵守以下 JSON Schema（不要输出其他内容）:
{{
  "chief_complaint": "主要症状及持续时间, 1-2句话",
  "present_illness": "本次发病详细情况, 50-150字",
  "past_history": "既往病史摘要",
  "allergy_history": "过敏药物/食物",
  "family_history": "家族遗传疾病",
  "medication_hist": "近期用药情况",
  "diagnosis": "AI辅助诊断建议 (标注AI性质)",
  "treatment": "建议检查和治疗方向 (标注AI性质)"
}}"""

ASSIST_SYSTEM_PROMPT = """你是一位经验丰富的临床决策支持专家。根据患者信息，为接诊医生提供辅助问诊提示。

## 输出要求
请严格按照以下 JSON 格式输出:
{{
  "disease_summary": "病情摘要，50-100字，概述核心症状和可能的病理机制",
  "follow_up_questions": ["需追问的问题1", "问题2", "..."],
  "differential_diagnosis": ["可能的鉴别诊断1", "鉴别诊断2", "..."],
  "necessary_tests": ["建议检查项目1", "检查项目2", "..."],
  "medication_suggestions": ["用药方向1 (注明需结合临床)", "..."],
  "referral_depts": ["如需转诊的建议科室1", "..."]
}}

## 约束
- 所有建议均为辅助性质，需标注"请结合临床判断"
- 不要推荐特定品牌药品，只给药物类别方向
- 检查项目按优先级排序
- 鉴别诊断从高可能性到低可能性排列"""

ASSIST_USER_MESSAGE_TEMPLATE = """## 患者基本信息
年龄: {age}
性别: {gender}

## 患者症状描述
{symptom_text}

## 患者健康档案
{health_record_text}

## 科室推荐结果
{department_info}

## 医学知识库参考
{rag_context}

请根据以上信息，提供辅助问诊提示。"""


# ============================================================
# EMRProcessor
# ============================================================

class EMRProcessor:
    """
    电子病历结构化提取处理器。

    输入患者症状 + 健康档案, 调用 LLM 输出结构化 medical_record 字段。
    同时支持 AI 辅助问诊提示生成，为医生端提供临床决策支持。

    使用示例:
        processor = EMRProcessor(verbose=True)

        # 1. 提取病历要素
        emr = processor.extract_medical_record(
            symptom_text="发热咳嗽3天",
            health_record={"past_illness": "高血压", "allergy": "青霉素"},
            patient_info={"age": 50, "gender": "男"},
        )
        print(emr.to_dict())

        # 2. 辅助问诊提示
        assist = processor.generate_assist_info(
            symptom_text="发热咳嗽3天",
            health_record={"past_illness": "高血压"},
            patient_info={"age": 50, "gender": "男"},
        )
        print(assist.to_api_response())
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        verbose: bool = False,
    ):
        """
        Args:
            model: LLM model name, defaults to deepseek-v4-flash.
            temperature: Generation temperature (low = more deterministic).
            max_tokens: Max output tokens.
            verbose: Print progress logs.
        """
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Lazy-loaded components
        self._llm_client = None
        self._vector_store = None
        self._rag_pipeline = None

    # ================================================================
    # Lazy-loaded components
    # ================================================================

    @property
    def llm_client(self):
        """Lazy-load DeepSeekClient."""
        if self._llm_client is None:
            import importlib.util as _iu

            _dc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_client.py")
            _dc_spec = _iu.spec_from_file_location("deepseek_client", _dc_path)
            _dc = _iu.module_from_spec(_dc_spec)
            _dc_spec.loader.exec_module(_dc)
            self._llm_client = _dc.DeepSeekClient(model=self.model)
        return self._llm_client

    @property
    def vector_store(self):
        """Lazy-load VectorStore for RAG context retrieval."""
        if self._vector_store is None:
            import importlib.util as _iu

            _qe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_engine.py")
            _qe_spec = _iu.spec_from_file_location("query_engine", _qe_path)
            _qe = _iu.module_from_spec(_qe_spec)
            _qe_spec.loader.exec_module(_qe)
            self._vector_store = _qe.VectorStore()
        return self._vector_store

    # ================================================================
    # Core: EMR Extraction
    # ================================================================

    def extract_medical_record(
        self,
        symptom_text: str,
        health_record: Optional[dict] = None,
        patient_info: Optional[dict] = None,
        use_rag: bool = True,
    ) -> EMRResult:
        """
        从患者症状和健康档案中提取结构化病历要素。

        Args:
            symptom_text: 患者症状描述 (来自 consultation.symptom_text)。
            health_record: 健康档案 dict, 字段对应 health_record 表:
                {
                    "past_illness": "高血压, 2型糖尿病",
                    "allergy": "青霉素过敏",
                    "surgery_history": "阑尾切除术 2019",
                    "medication": "硝苯地平 30mg qd, 二甲双胍 500mg bid",
                    "blood_type": "A",
                    ...
                }
            patient_info: 患者基本信息:
                {"age": 65, "gender": "男"}
            use_rag: 是否启用 RAG 检索相关疾病知识作为 LLM 上下文。

        Returns:
            EMRResult with structured medical_record fields.
        """
        start_time = time.time()

        # Normalize inputs
        health_record = health_record or {}
        patient_info = patient_info or {}

        age = patient_info.get("age", "未知")
        gender = patient_info.get("gender", "未知")

        # Build health record text
        health_record_text = self._format_health_record(health_record)

        # RAG retrieval for disease context
        rag_context = "（未启用RAG检索）"
        if use_rag and symptom_text.strip():
            try:
                rag_context = self._retrieve_rag_context(symptom_text)
            except Exception as e:
                if self.verbose:
                    print(f"[EMRProcessor] RAG检索失败: {e}，使用空上下文")

        # Build user message
        user_message = EMR_USER_MESSAGE_TEMPLATE.format(
            age=age,
            gender=gender,
            symptom_text=symptom_text or "（未提供）",
            health_record_text=health_record_text,
            rag_context=rag_context,
        )

        if self.verbose:
            print(f"[EMRProcessor] 正在提取病历要素...")
            print(f"  症状: {symptom_text[:60]}...")
            print(f"  年龄: {age}, 性别: {gender}")

        # Load config (MySQL优先, 硬编码默认兜底)
        try:
            from ai_config_loader import get_prompt, get_params
            _sys_prompt = get_prompt("emr_extract")
            _cfg = get_params("emr_extract")
            _temp = _cfg["temperature"]
            _max_tok = _cfg["max_tokens"]
        except Exception:
            _sys_prompt = EMR_SYSTEM_PROMPT
            _temp = self.temperature
            _max_tok = self.max_tokens

        # Call LLM
        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _sys_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content
            parsed = json.loads(raw_text)

            latency_ms = round((time.time() - start_time) * 1000, 1)

            result = EMRResult(
                chief_complaint=parsed.get("chief_complaint"),
                present_illness=parsed.get("present_illness"),
                past_history=parsed.get("past_history"),
                allergy_history=parsed.get("allergy_history"),
                family_history=parsed.get("family_history"),
                medication_hist=parsed.get("medication_hist"),
                diagnosis=parsed.get("diagnosis"),
                treatment=parsed.get("treatment"),
                raw_response=raw_text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                latency_ms=latency_ms,
            )

            if self.verbose:
                print(f"[EMRProcessor] 提取完成, 耗时 {latency_ms}ms")
                print(f"  主诉: {result.chief_complaint}")
                print(f"  Token: {result.usage}")

            return result

        except json.JSONDecodeError as e:
            latency_ms = round((time.time() - start_time) * 1000, 1)
            if self.verbose:
                print(f"[EMRProcessor] JSON解析失败: {e}")
            return EMRResult(
                error=f"LLM返回格式异常: {str(e)}",
                raw_response=raw_text if 'raw_text' in dir() else None,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = round((time.time() - start_time) * 1000, 1)
            if self.verbose:
                print(f"[EMRProcessor] 提取失败: {e}")
            return EMRResult(
                error=f"AI服务调用失败: {str(e)}",
                latency_ms=latency_ms,
            )

    # ================================================================
    # Assist: Clinical Decision Support
    # ================================================================

    def generate_assist_info(
        self,
        symptom_text: str,
        health_record: Optional[dict] = None,
        patient_info: Optional[dict] = None,
        use_rag: bool = True,
    ) -> AssistResult:
        """
        生成 AI 辅助问诊提示，为医生提供临床决策支持。

        Args:
            symptom_text: 患者症状描述。
            health_record: 健康档案 dict。
            patient_info: 患者基本信息 {"age": int, "gender": str}。
            use_rag: 是否启用 RAG 检索。

        Returns:
            AssistResult with 5 types of clinical hints.
        """
        start_time = time.time()

        health_record = health_record or {}
        patient_info = patient_info or {}

        age = patient_info.get("age", "未知")
        gender = patient_info.get("gender", "未知")

        health_record_text = self._format_health_record(health_record)

        # RAG context
        rag_context = "（未启用RAG检索）"
        department_info = "（未获取科室推荐）"
        if use_rag and symptom_text.strip():
            try:
                rag_context = self._retrieve_rag_context(symptom_text)
                # Also get department recommendation for context
                rag_result = self.vector_store.comprehensive_search(symptom_text, top_k=3)
                dept_names = rag_result.get("all_departments", [])
                if dept_names:
                    department_info = f"AI推荐科室: {', '.join(dept_names[:5])}"
                    top_disease = rag_result.get("disease_results", [{}])[0] if rag_result.get("disease_results") else {}
                    if top_disease:
                        department_info += f" | 最可能疾病: {top_disease.get('disease', 'N/A')}"
            except Exception as e:
                if self.verbose:
                    print(f"[EMRProcessor] RAG检索失败: {e}")

        user_message = ASSIST_USER_MESSAGE_TEMPLATE.format(
            age=age,
            gender=gender,
            symptom_text=symptom_text or "（未提供）",
            health_record_text=health_record_text,
            department_info=department_info,
            rag_context=rag_context,
        )

        if self.verbose:
            print(f"[EMRProcessor] 正在生成辅助问诊提示...")

        # Load config (MySQL优先, 硬编码默认兜底)
        try:
            from ai_config_loader import get_prompt, get_params
            _sys_prompt = get_prompt("assist")
            _cfg = get_params("assist")
            _temp = _cfg["temperature"]
            _max_tok = _cfg["max_tokens"]
        except Exception:
            _sys_prompt = ASSIST_SYSTEM_PROMPT
            _temp = 0.3
            _max_tok = 1000

        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _sys_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content
            parsed = json.loads(raw_text)

            latency_ms = round((time.time() - start_time) * 1000, 1)

            result = AssistResult(
                disease_summary=parsed.get("disease_summary"),
                follow_up_questions=parsed.get("follow_up_questions", []),
                differential_diagnosis=parsed.get("differential_diagnosis", []),
                necessary_tests=parsed.get("necessary_tests", []),
                medication_suggestions=parsed.get("medication_suggestions", []),
                referral_depts=parsed.get("referral_depts", []),
                raw_response=raw_text,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
                latency_ms=latency_ms,
            )

            if self.verbose:
                print(f"[EMRProcessor] 辅助问诊提示生成完成, 耗时 {latency_ms}ms")

            return result

        except Exception as e:
            latency_ms = round((time.time() - start_time) * 1000, 1)
            if self.verbose:
                print(f"[EMRProcessor] 辅助问诊生成失败: {e}")
            return AssistResult(
                error=f"AI服务调用失败: {str(e)}",
                latency_ms=latency_ms,
            )

    # ================================================================
    # Helper methods
    # ================================================================

    def _format_health_record(self, health_record: dict) -> str:
        """Format health_record dict into a readable text block for LLM context."""
        if not health_record:
            return "（患者暂无健康档案记录）"

        lines = []
        field_labels = {
            "past_illness": "既往病史",
            "allergy": "过敏史",
            "surgery_history": "手术史",
            "medication": "当前用药",
            "blood_type": "血型",
            "birth_date": "出生日期",
            "gender": "性别",
            "member_name": "姓名",
        }

        for key, label in field_labels.items():
            value = health_record.get(key)
            if value:
                lines.append(f"- {label}: {value}")

        if not lines:
            return "（健康档案无有效记录）"

        return "\n".join(lines)

    def _retrieve_rag_context(self, symptom_text: str) -> str:
        """
        Retrieve relevant disease knowledge from ChromaDB as LLM context.

        Returns:
            Formatted string of Top-5 disease matches for LLM prompt injection.
        """
        diseases = self.vector_store.search_disease(symptom_text, top_k=5)

        if not diseases:
            return "（未检索到高度匹配的疾病知识）"

        lines = ["以下是从医学知识库中检索到的最相关疾病信息:"]
        for i, d in enumerate(diseases):
            lines.append(
                f"{i + 1}. {d['disease']} (相关度: {d['score']:.1%})\n"
                f"   症状: {d['symptoms']}\n"
                f"   科室: {d['departments']}\n"
                f"   简介: {d.get('desc', '')[:150]}"
            )

        return "\n".join(lines)

    def health_check(self) -> dict:
        """Check LLM connectivity for EMR service."""
        try:
            result = self.llm_client.health_check()
            return {
                "status": "ok",
                "model": result.get("model", self.model),
                "service": "emr_extractor",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "service": "emr_extractor",
            }


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  EMR Extractor — 病历要素提取测试")
    print("=" * 65)

    processor = EMRProcessor(verbose=True)

    # Test 1: EMR extraction with full health record
    print("\n─── Test 1: 完整病历提取 ───")
    result = processor.extract_medical_record(
        symptom_text="近3天反复发热，最高39.2°C，伴有咳嗽咳黄痰，右侧胸痛，"
                     "呼吸困难。自行服用布洛芬后体温可降至37.5°C但反复升高。",
        health_record={
            "past_illness": "高血压病史5年",
            "allergy": "青霉素过敏",
            "surgery_history": "阑尾切除术 2019年",
            "medication": "硝苯地平缓释片 30mg 每日1次",
            "blood_type": "A",
        },
        patient_info={"age": 65, "gender": "男"},
    )

    if result.error:
        print(f"  [ERROR] {result.error}")
    else:
        data = result.to_dict()
        for field, value in data.items():
            print(f"  [{field}]")
            print(f"    {value}")
        print(f"  Latency: {result.latency_ms}ms")
        print(f"  Tokens: {result.usage}")

    # Test 2: Minimal input (empty health record)
    print("\n─── Test 2: 最小输入 (无健康档案) ───")
    result2 = processor.extract_medical_record(
        symptom_text="肚子疼拉肚子两天了",
        health_record=None,
        patient_info={"age": 30, "gender": "女"},
    )

    if result2.error:
        print(f"  [ERROR] {result2.error}")
    else:
        print(f"  主诉: {result2.chief_complaint}")
        print(f"  既往史: {result2.past_history}")
        print(f"  过敏史: {result2.allergy_history}")

    # Test 3: Assist info
    print("\n─── Test 3: 辅助问诊提示 ───")
    assist = processor.generate_assist_info(
        symptom_text="胸闷心慌气短，活动后加重，伴有头晕",
        health_record={
            "past_illness": "冠心病 3年, 高脂血症",
            "medication": "阿司匹林 100mg qd, 阿托伐他汀 20mg qn",
        },
        patient_info={"age": 58, "gender": "男"},
    )

    if assist.error:
        print(f"  [ERROR] {assist.error}")
    else:
        print(f"  病情摘要: {assist.disease_summary}")
        print(f"  追问问题: {assist.follow_up_questions}")
        print(f"  鉴别诊断: {assist.differential_diagnosis}")
        print(f"  建议检查: {assist.necessary_tests}")
        print(f"  用药建议: {assist.medication_suggestions}")
        print(f"  转诊建议: {assist.referral_depts}")
