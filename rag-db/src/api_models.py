"""
api_models.py
Pydantic 请求/响应模型 — FastAPI 接口的数据契约层

定义了 Java 后端与 Python AI 引擎之间的全部 HTTP 接口数据结构。
所有模型对应概要设计中的实体描述，字段命名与 medical_record /
health_record / consultation 表一致。

使用方式:
    from api_models import SearchRequest, SearchResponse
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 通用结构
# ============================================================

class ApiResponse(BaseModel):
    """统一响应包装，匹配 Java 端统一返回格式 {code, message, data}"""
    code: int = 200
    message: str = "success"
    data: Optional[dict] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    code: int = 500
    message: str = "error"
    detail: Optional[str] = None


# ============================================================
# 智能导诊 — 科室推荐
# ============================================================

class SearchRequest(BaseModel):
    """
    科室推荐请求 (患者端 → /api/rag/search)

    对应概要设计 患者模块-智能导诊 的输入。
    """
    query: str = Field(
        ...,
        description="患者症状描述，支持口语化/方言表达",
        examples=["头痛发热咳嗽流鼻涕", "肚子疼拉稀想吐没胃口"],
        min_length=1,
        max_length=2000,
    )
    top_k: int = Field(
        default=5,
        description="返回的最大疾病匹配数",
        ge=1,
        le=20,
    )


class DiseaseResult(BaseModel):
    """单条疾病检索结果"""
    disease: str = Field(description="疾病名称")
    symptoms: str = Field(description="疾病关联症状")
    departments: str = Field(description="推荐科室，逗号分隔")
    category: str = Field(default="", description="疾病分类")
    drugs: str = Field(default="", description="常用药品")
    desc: str = Field(default="", description="疾病简介")
    score: float = Field(description="匹配置信度 (0-1)")
    cosine_score: Optional[float] = Field(default=None, description="原始余弦相似度 (reranker启用时保留)")
    chain: str = Field(description="推理链: 症状→疾病→科室")


class PrimaryRecommendation(BaseModel):
    """主推荐科室"""
    department: str = Field(description="首选科室名称")
    disease: str = Field(description="最可能疾病")
    confidence: float = Field(description="置信度 (0-1)")
    reasoning: str = Field(description="推理依据")


class SymptomDirectResult(BaseModel):
    """症状→科室直接映射结果"""
    symptom: str = Field(description="症状名")
    departments: str = Field(description="关联科室")
    disease_count: int = Field(default=0, description="关联疾病数")
    score: float = Field(description="匹配相似度")


class SearchResponse(BaseModel):
    """科室推荐响应"""
    query: str = Field(description="原始查询")
    search_query: Optional[str] = Field(default=None, description="优化后实际用于检索的查询")
    disease_results: list[DiseaseResult] = Field(default_factory=list, description="疾病检索结果 Top-N")
    symptom_direct: list[SymptomDirectResult] = Field(default_factory=list, description="症状直接映射")
    all_departments: list[str] = Field(default_factory=list, description="所有推荐科室汇总")
    primary_recommendation: Optional[PrimaryRecommendation] = Field(default=None, description="首选推荐")
    reranked: bool = Field(default=False, description="是否经过Cross-Encoder精排")

    # LLM 生成层
    department: Optional[str] = Field(default=None, description="LLM推荐的科室")
    disease: Optional[str] = Field(default=None, description="LLM判断的最可能疾病")
    confidence: Optional[int] = Field(default=None, description="LLM置信度 (0-100)")
    reasoning: Optional[str] = Field(default=None, description="LLM推理依据")
    suggestion: Optional[str] = Field(default=None, description="就医建议")
    alternative_departments: Optional[list[str]] = Field(default=None, description="备选科室")
    emergency_warning: Optional[bool] = Field(default=None, description="是否触发危急警告")

    # 查询优化
    query_optimization: Optional[dict] = Field(default=None, description="查询优化详情")

    # 元数据
    latency_ms: Optional[float] = Field(default=None, description="总延迟 (ms)")
    token_usage: Optional[dict] = Field(default=None, description="LLM Token 消耗")


# ============================================================
# 病历要素提取
# ============================================================

class HealthRecordInput(BaseModel):
    """
    健康档案输入 (对应 health_record 表)

    Java 端从 health_record 表查询后传入。
    字段可全部为空 — 表示患者尚未建立档案。
    """
    record_id: Optional[int] = Field(default=None, description="档案ID")
    member_name: Optional[str] = Field(default=None, description="档案人姓名")
    is_self: Optional[int] = Field(default=1, description="是否本人: 1本人 0家庭成员")
    gender: Optional[int] = Field(default=None, description="性别: 1男 2女")
    birth_date: Optional[str] = Field(default=None, description="出生日期")
    blood_type: Optional[str] = Field(default=None, description="血型: A/B/O/AB")
    allergy: Optional[str] = Field(default=None, description="过敏史 (药物/食物)")
    past_illness: Optional[str] = Field(default=None, description="既往病史")
    surgery_history: Optional[str] = Field(default=None, description="手术史")
    medication: Optional[str] = Field(default=None, description="当前用药情况")
    report_urls: Optional[list[str]] = Field(default=None, description="检查报告图片URL列表")
    ai_summary: Optional[str] = Field(default=None, description="AI生成的历史摘要")


class PatientInfo(BaseModel):
    """患者基本信息 (复用 patient 表字段)"""
    patient_id: Optional[int] = Field(default=None, description="患者ID")
    age: Optional[int] = Field(default=None, description="年龄")
    gender: Optional[str] = Field(default=None, description="性别: 男/女")


class EMRRequest(BaseModel):
    """
    病历要素提取请求 (医生端 → /api/rag/emr/extract)

    对应概要设计 医生模块-AI辅助问诊 的输入。
    """
    symptom_text: str = Field(
        ...,
        description="患者自述症状文本 (来自 consultation.symptom_text)",
        examples=["近3天反复发热，最高39.2°C，咳嗽咳黄痰，右侧胸痛，呼吸困难"],
        min_length=1,
        max_length=5000,
    )
    health_record: Optional[HealthRecordInput] = Field(
        default=None,
        description="患者健康档案 (来自 health_record 表，可为空)"
    )
    patient_info: Optional[PatientInfo] = Field(
        default=None,
        description="患者基本信息 (来自 patient 表)"
    )
    use_rag: bool = Field(
        default=True,
        description="是否启用 RAG 检索医学知识库作为 LLM 上下文"
    )
    consult_id: Optional[int] = Field(
        default=None,
        description="关联的问诊记录ID (来自 consultation 表，用于回写)"
    )


class MedicalRecordFields(BaseModel):
    """
    病历要素输出 (对应 medical_record 表核心字段)

    与概要设计 Section 5.8 的 medical_record 实体描述一一对应。
    """
    chief_complaint: Optional[str] = Field(default=None, description="主诉: 主要症状及持续时间")
    present_illness: Optional[str] = Field(default=None, description="现病史: 本次发病情况")
    past_history: Optional[str] = Field(default=None, description="既往史: 既往病史摘要")
    allergy_history: Optional[str] = Field(default=None, description="过敏史: 过敏药物/食物")
    family_history: Optional[str] = Field(default=None, description="家族史: 家族遗传疾病")
    medication_hist: Optional[str] = Field(default=None, description="用药史: 近期用药情况")
    diagnosis: Optional[str] = Field(default=None, description="AI辅助诊断建议")
    treatment: Optional[str] = Field(default=None, description="AI辅助处理建议")


class EMRResponse(BaseModel):
    """
    病历要素提取响应

    Java 端接收后写入 medical_record 表:
      INSERT INTO medical_record (consult_id, doctor_id, patient_id,
        chief_complaint, present_illness, past_history, allergy_history,
        family_history, medication_hist, diagnosis, treatment)
      VALUES (...)
    """
    code: int = 200
    data: Optional[MedicalRecordFields] = None
    metadata: Optional[dict] = Field(default=None, description="模型名/延迟/Token消耗/错误信息")


# ============================================================
# 辅助问诊提示
# ============================================================

class AssistRequest(BaseModel):
    """
    辅助问诊提示请求 (医生端 → /api/rag/assist/info)

    对应概要设计 医生模块-AI辅助问诊 的"辅助问诊提示"输出。
    """
    symptom_text: str = Field(
        ...,
        description="患者症状描述",
        min_length=1,
        max_length=5000,
    )
    health_record: Optional[HealthRecordInput] = Field(
        default=None,
        description="患者健康档案"
    )
    patient_info: Optional[PatientInfo] = Field(
        default=None,
        description="患者基本信息"
    )
    use_rag: bool = Field(default=True, description="是否启用RAG检索")


class AssistData(BaseModel):
    """辅助问诊提示内容"""
    disease_summary: Optional[str] = Field(default=None, description="AI病情摘要")
    follow_up_questions: list[str] = Field(default_factory=list, description="建议追问的问题清单")
    differential_diagnosis: list[str] = Field(default_factory=list, description="可能的鉴别诊断方向")
    necessary_tests: list[str] = Field(default_factory=list, description="建议的检查项目")
    medication_suggestions: list[str] = Field(default_factory=list, description="用药方向建议")
    referral_depts: list[str] = Field(default_factory=list, description="建议转诊科室")


class AssistResponse(BaseModel):
    """辅助问诊提示响应"""
    code: int = 200
    data: Optional[AssistData] = None
    metadata: Optional[dict] = Field(default=None, description="模型名/延迟/Token消耗/错误信息")


# ============================================================
# 科室信息查询
# ============================================================

class DepartmentInfo(BaseModel):
    """科室信息"""
    department: str = Field(description="科室名称")
    disease_count: int = Field(default=0, description="关联疾病数")
    common_symptoms: str = Field(default="", description="常见症状")
    sample_diseases: str = Field(default="", description="代表性疾病")
    score: float = Field(default=0.0, description="匹配相似度")


class DepartmentListResponse(BaseModel):
    """科室列表响应"""
    code: int = 200
    departments: list[dict] = Field(default_factory=list, description="科室列表")
    total: int = Field(default=0, description="科室总数")


class DepartmentDetailResponse(BaseModel):
    """科室详情响应"""
    code: int = 200
    data: Optional[DepartmentInfo] = None


# ============================================================
# 健康检查
# ============================================================

class HealthResponse(BaseModel):
    """服务健康检查"""
    status: str = Field(description="ok / degraded / error")
    version: str = Field(default="1.0.0")
    services: dict = Field(default_factory=dict, description="各子服务状态")
    uptime_seconds: Optional[float] = Field(default=None, description="运行时长")


# ============================================================
# 个性化生活建议
# ============================================================

class ConsultationInput(BaseModel):
    """
    问诊记录输入 (consultation 表中 AI 相关字段)

    Java 端从 consultation 表查询后传入。
    仅传入 AI 生成建议所需的字段，不传全部 21 个字段。
    """
    consult_id: Optional[int] = Field(default=None, description="问诊ID")
    symptom_text: Optional[str] = Field(default=None, description="患者自述症状")
    doctor_advice: Optional[str] = Field(default=None, description="医生诊疗建议")
    ai_analysis: Optional[dict] = Field(default=None, description="AI结构化症状分析结果JSON")
    consultation_dialog: Optional[str] = Field(default=None, description="多轮追问对话历史JSON")


class SuggestionItem(BaseModel):
    """单条生活建议"""
    title: str = Field(description="建议标题，10-20字")
    content: str = Field(description="建议详细内容，30-80字，通俗易懂")


class SuggestionCategory(BaseModel):
    """按分类分组的建议集合"""
    category: str = Field(description="建议类别: diet/exercise/sleep/medication/seasonal")
    items: list[SuggestionItem] = Field(description="该类别下的建议列表 (1-3条)")


class SuggestionRequest(BaseModel):
    """
    个性化生活建议请求 (Java → /api/rag/health-suggestion)

    对应 health_suggestion 表的输入:
      - health_record: 健康档案数据 (不含 ai_summary)
      - consultation: 问诊记录 AI 相关字段
    """
    health_record: Optional[HealthRecordInput] = Field(
        default=None,
        description="患者健康档案 (来自 health_record 表)"
    )
    consultation: Optional[ConsultationInput] = Field(
        default=None,
        description="患者问诊记录 (来自 consultation 表)"
    )


class SuggestionData(BaseModel):
    """建议输出数据"""
    suggestions: list[SuggestionCategory] = Field(
        description="5 类个性化生活建议"
    )


# ============================================================
# 用户反馈
# ============================================================

class FeedbackRequest(BaseModel):
    """用户反馈请求 (患者端 → /api/rag/feedback)"""
    query: str = Field(..., description="原始查询")
    consult_id: Optional[int] = Field(default=None, description="关联问诊ID")
    recommended_department: Optional[str] = Field(default=None, description="AI推荐的科室")
    feedback: str = Field(..., description="反馈类型: positive / negative / neutral")
    actual_department: Optional[str] = Field(default=None, description="用户实际就诊科室 (negative时填写)")
    comment: Optional[str] = Field(default=None, description="附加评论")


# ============================================================
# Multi-turn Dialogue Agent
# ============================================================

class DialogueStartRequest(BaseModel):
    """开始新对话会话请求 (Java → /api/rag/dialogue/start)"""
    patient_id: Optional[int] = Field(default=None, description="患者ID (未登录可空)")
    initial_symptom: Optional[str] = Field(default=None, description="初始症状描述 (可选，首轮直接输入)", max_length=2000)
    max_turns: Optional[int] = Field(default=8, ge=3, le=20, description="最大对话轮数")


class DialogueContinueRequest(BaseModel):
    """继续对话请求 (Java → /api/rag/dialogue/continue)"""
    session_id: str = Field(..., description="会话ID (UUID v4)", min_length=36, max_length=36)
    patient_input: str = Field(..., description="患者本轮回答", min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    """Agent 结构化响应 (DialogueManager → Java)"""
    action: str = Field(..., description="Agent 动作: 'ask' | 'recommend' | 'emergency'")
    session_id: str = Field(..., description="会话ID")
    current_turn: int = Field(..., description="当前轮次")
    question: Optional[str] = Field(default=None, description="追问问题 (action='ask' 时)")
    question_reasoning: Optional[str] = Field(default=None, description="追问的鉴别诊断逻辑 (action='ask' 时)")
    candidate_diseases: Optional[list[dict]] = Field(default=None, description="Top-3 候选疾病 [{disease, score, departments, distinguishing_symptoms}]")
    recommendation: Optional[dict] = Field(default=None, description="最终推荐 (action='recommend' 时): {department, disease, confidence, reasoning, suggestion, alternative_departments}")
    collected_info: Optional[dict] = Field(default=None, description="已收集信息摘要 {symptoms, body_parts, duration, severity}")
    emergency_warning: Optional[str] = Field(default=None, description="紧急警告信息 (action='emergency' 时)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="整体置信度 0.0-1.0")


class DialogueSessionState(BaseModel):
    """对话会话完整状态 (GET /api/rag/dialogue/{session_id})"""
    session_id: str = Field(..., description="会话ID")
    patient_id: Optional[int] = Field(default=None, description="患者ID")
    status: str = Field(..., description="会话状态: active / closed / emergency / timeout")
    current_turn: int = Field(..., description="当前轮次")
    max_turns: int = Field(..., description="最大允许轮数")
    collected_symptoms: Optional[list[dict]] = Field(default=None, description="已收集的症状列表")
    extracted_keywords: Optional[list[str]] = Field(default=None, description="提取的关键词")
    candidate_diseases: Optional[list[dict]] = Field(default=None, description="当前候选疾病")
    dialogue_history: Optional[list[dict]] = Field(default=None, description="完整对话历史")
    final_recommendation: Optional[dict] = Field(default=None, description="最终推荐结果")
    created_at: Optional[str] = Field(default=None, description="创建时间")
    updated_at: Optional[str] = Field(default=None, description="最后更新时间")
