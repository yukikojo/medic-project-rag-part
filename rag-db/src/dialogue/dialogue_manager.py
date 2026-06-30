"""
dialogue_manager.py
Multi-turn Medical Dialogue Agent — Agent-Skill 架构

DialogueManager 作为 Agent 编排对话循环:
  - Skill 1: Symptom Extraction (LLM) — 每轮提取结构化症状
  - Skill 2: RAG Retrieval (VectorStore) — 检索候选疾病
  - Skill 3: LLM Decision (DeepSeek) — 判断追问 or 推荐
  - Skill 4: LLM Followup — 生成鉴别诊断追问
  - Skill 5: LLM Recommendation — 生成最终科室推荐

决策逻辑 (B+C 组合):
  - Rule B: 症状数 < 2 → 强制追问
  - Rule B: 轮数 >= max_turns → 强制推荐
  - Rule C: 其他 → LLM 自主判断 (dialogue_decision scene)

追问策略 (策略 C):
  - 鉴别诊断驱动: 针对 Top-3 候选疾病的区分性特征提问

状态存储: MySQL medical_rag.dialogue_session

使用:
    manager = DialogueManager(verbose=True)

    # 开始新会话
    result = manager.start_session(patient_id=1, initial_symptom="头痛三天")

    # 继续会话
    result = manager.process(session_id="uuid-...", patient_input="有恶心畏光")
"""

import os
import sys
import json
import time
import re
import uuid
from typing import Optional
from datetime import datetime

# Ensure parent src/ is importable
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(_src, "..", "..", ".env"))

# ============================================================
# Constants
# ============================================================

MIN_SYMPTOM_THRESHOLD = 2       # Rule B: 最少需要的症状数
DEFAULT_MAX_TURNS = 8           # 默认最大轮数
SESSION_ID_LENGTH = 36          # UUID v4

EMERGENCY_KEYWORDS = [
    "剧烈疼痛", "昏迷", "大出血", "意识不清", "呼吸困难",
    "胸痛", "窒息", "抽搐", "休克", "无呼吸", "无心跳",
    "严重外伤", "大咯血", "呕血", "便血", "剧烈胸痛",
    "无法呼吸", "失去意识", "心脏骤停", "严重过敏",
]

SQL_CREATE_DIALOGUE_SESSION = """
    CREATE TABLE IF NOT EXISTS dialogue_session (
        session_id           VARCHAR(36)  PRIMARY KEY COMMENT 'UUID v4',
        patient_id           BIGINT       DEFAULT NULL COMMENT '患者ID',
        status               VARCHAR(20)  NOT NULL DEFAULT 'active'
                             COMMENT 'active/closed/emergency/timeout',
        collected_symptoms   TEXT         COMMENT 'JSON: accumulated symptoms',
        extracted_keywords   TEXT         COMMENT 'JSON: keyword list',
        candidate_diseases   TEXT         COMMENT 'JSON: top-5 candidate diseases',
        dialogue_history     TEXT         COMMENT 'JSON: full Q&A history',
        final_recommendation TEXT         COMMENT 'JSON: final recommendation',
        max_turns            INT          DEFAULT 8,
        current_turn         INT          DEFAULT 0,
        created_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_patient_id (patient_id),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    COMMENT='多轮对话会话状态表'
"""

# ============================================================
# Fallback prompts (used when MySQL ai_model_config unavailable)
# ============================================================

DEFAULT_EXTRACT_PROMPT = """你是一个医疗症状提取助手。从患者的描述中提取关键症状信息。

输出 JSON 格式:
{
  "symptoms": ["症状1", "症状2"],
  "body_parts": ["部位1"],
  "duration": "持续时间",
  "severity": "轻度/中度/重度",
  "keywords": ["关键词1", "关键词2"]
}"""

DEFAULT_DECISION_PROMPT = """你是一位经验丰富的临床医生。判断是否已收集足够信息来给出初步诊断推荐。

已收集症状: {accumulated_symptoms}
候选疾病: {candidate_diseases}
当前轮次: {current_turn}/{max_turns}

判断标准:
1. 已收集 ≥2 个明确症状
2. 某个候选疾病置信度明显高于其他
3. 再追问对区分疾病帮助不大

输出 JSON:
{"decision": "continue 或 recommend", "confidence": 0-100, "reasoning": "...", "key_symptoms_count": N}"""

DEFAULT_FOLLOWUP_PROMPT = """你是一位经验丰富的临床医生。根据患者已描述的症状和候选疾病，生成一个鉴别诊断问题。

已收集症状: {accumulated_symptoms}
候选疾病 (Top-3): {candidate_diseases}

要求:
1. 只生成一个最关键的问题，能最好地区分 Top-3 候选疾病
2. 针对疾病之间最具鉴别力的独特症状
3. 通俗易懂，尽量用"是否"或选择形式
4. 简短具体，一次只问一件事

输出 JSON:
{"question": "...", "reasoning": "...", "target_diseases": [...]}"""


# ============================================================
# DialogueManager
# ============================================================

class DialogueManager:
    """
    多轮医疗对话 Agent。

    编排对话循环: 提取症状 → RAG检索 → 决策(追问/推荐) → 生成输出

    使用:
        manager = DialogueManager(verbose=True)
        result = manager.start_session(initial_symptom="头痛三天")
        result = manager.process(session_id="...", patient_input="有恶心畏光")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        verbose: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Lazy-loaded Skills
        self._llm_client = None
        self._vector_store = None
        self._table_ensured = False

    # ============================================================
    # Lazy-loaded Skills (复用现有模块)
    # ============================================================

    @property
    def llm_client(self):
        """Lazy-load DeepSeekClient."""
        if self._llm_client is None:
            import importlib.util as _iu
            _dc = os.path.join(_src, "generation", "deepseek_client.py")
            _spec = _iu.spec_from_file_location("generation.deepseek_client", _dc)
            _mod = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self._llm_client = _mod.DeepSeekClient(model=self.model)
        return self._llm_client

    @property
    def vector_store(self):
        """Lazy-load VectorStore."""
        if self._vector_store is None:
            import importlib.util as _iu
            _qe = os.path.join(_src, "retrieval", "query_engine.py")
            _spec = _iu.spec_from_file_location("retrieval.query_engine", _qe)
            _mod = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self._vector_store = _mod.VectorStore()
        return self._vector_store

    # ============================================================
    # Public API
    # ============================================================

    def start_session(
        self,
        patient_id: Optional[int] = None,
        initial_symptom: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict:
        """
        开始新对话会话。

        Args:
            patient_id: 患者ID (可选)
            initial_symptom: 初始症状描述 (可选，传入则自动执行第一轮)
            max_turns: 最大对话轮数 (3-20)

        Returns:
            AgentResponse dict with action, session_id, question (if ask),
            or recommendation (if symptoms already sufficient)
        """
        self._ensure_table()

        session_id = str(uuid.uuid4())
        max_turns = max(3, min(20, max_turns))

        # Create session
        self._create_session(session_id, patient_id, max_turns)

        if self.verbose:
            print(f"[Dialogue] 新会话: {session_id}, patient={patient_id}, "
                  f"max_turns={max_turns}")

        # If no initial symptom, return greeting
        if not initial_symptom or not str(initial_symptom).strip():
            return {
                "action": "ask",
                "session_id": session_id,
                "current_turn": 0,
                "question": "您好，请问您有什么不舒服的症状？请尽量详细描述，"
                           "比如哪里不舒服、持续多久了、什么情况下会加重。",
                "question_reasoning": None,
                "candidate_diseases": None,
                "recommendation": None,
                "collected_info": None,
                "emergency_warning": None,
                "confidence": 0.0,
            }

        # Process first turn immediately
        return self.process(session_id, str(initial_symptom).strip(),
                           patient_id=patient_id)

    def process(
        self,
        session_id: str,
        patient_input: str,
        patient_id: Optional[int] = None,
    ) -> dict:
        """
        处理一轮患者输入 — Agent 核心循环。

        Args:
            session_id: 会话ID
            patient_input: 患者本轮输入
            patient_id: 患者ID (仅首轮需要，后续可空)

        Returns:
            AgentResponse dict:
              action="ask"       → question, candidate_diseases, collected_info
              action="recommend" → recommendation, candidate_diseases, collected_info
              action="emergency" → emergency_warning, candidate_diseases, collected_info
        """
        start = time.time()

        # 0. Ensure table exists
        self._ensure_table()

        # 1. Load session
        session = self._load_session(session_id)
        if session is None:
            return {
                "action": "error",
                "session_id": session_id,
                "error": "Session not found. Please start a new session.",
                "current_turn": 0,
                "confidence": 0.0,
            }

        if session["status"] != "active":
            return {
                "action": "error",
                "session_id": session_id,
                "error": f"Session is {session['status']}. "
                         f"Please start a new session.",
                "current_turn": session["current_turn"],
                "confidence": 0.0,
            }

        current_turn = session["current_turn"] + 1
        max_turns = session["max_turns"]

        if self.verbose:
            print(f"[Dialogue] Turn {current_turn}/{max_turns}: "
                  f"\"{patient_input[:80]}...\"")

        # 2. Emergency keyword check (fast path, no LLM)
        emergency_matches = self._check_emergency_keywords(patient_input)
        if emergency_matches:
            if self.verbose:
                print(f"[Dialogue] 紧急关键词: {emergency_matches}")

            # Do RAG search even for emergency (to show possible causes)
            accumulated = json.loads(session.get("collected_symptoms", "{}") or "{}")
            extracted = self._extract_symptoms(patient_input)
            accumulated = self._merge_symptoms(accumulated, extracted)
            candidates = self._retrieve_diseases(accumulated)

            # Update session
            history = json.loads(session.get("dialogue_history", "[]") or "[]")
            history.append(self._build_history_entry(current_turn, "patient",
                                                       patient_input))

            self._update_session(session_id, {
                "status": "emergency",
                "current_turn": current_turn,
                "collected_symptoms": json.dumps(accumulated, ensure_ascii=False),
                "candidate_diseases": json.dumps(candidates, ensure_ascii=False),
                "dialogue_history": json.dumps(history, ensure_ascii=False),
            })

            return {
                "action": "emergency",
                "session_id": session_id,
                "current_turn": current_turn,
                "emergency_warning": (
                    f"⚠️ 检测到紧急症状描述：{'、'.join(emergency_matches)}。"
                    f"建议立即拨打120或前往最近的急诊科就医！"
                ),
                "candidate_diseases": candidates[:3],
                "recommendation": None,
                "collected_info": self._summarize_collected(accumulated),
                "question": None,
                "question_reasoning": None,
                "confidence": 0.95,
            }

        # 3. Extract symptoms from this turn (LLM Skill)
        extracted = self._extract_symptoms(patient_input)

        # 4. Merge with accumulated symptoms
        prev_symptoms = json.loads(
            session.get("collected_symptoms", "{}") or "{}"
        )
        accumulated = self._merge_symptoms(prev_symptoms, extracted)

        # 5. RAG retrieval (VectorStore Skill)
        candidates = self._retrieve_diseases(accumulated)

        # 6. Decision: enough info? (Rule B + LLM Skill C)
        decision = self._decide_sufficient_info(
            accumulated_symptoms=accumulated,
            candidate_diseases=candidates,
            current_turn=current_turn,
            max_turns=max_turns,
        )

        # 7. Generate output based on decision
        question = None
        question_reasoning = None
        recommendation = None
        new_status = "active"

        if decision["decision"] == "recommend" or current_turn >= max_turns:
            # Generate final recommendation
            recommendation = self._generate_recommendation(accumulated, candidates)
            new_status = "closed"
            if self.verbose:
                print(f"[Dialogue] → recommend: "
                      f"{recommendation.get('disease', 'N/A')} "
                      f"(confidence: {recommendation.get('confidence', 0):.0%})")
        else:
            # Load existing history so LLM knows what was already asked
            session_history = json.loads(session.get("dialogue_history", "[]") or "[]")

            # Generate differential diagnosis question
            q_result = self._generate_followup_question(
                accumulated, candidates, session_history
            )
            question = q_result.get("question",
                       "请详细描述您的症状，包括持续时间、严重程度和伴随症状。")
            question_reasoning = q_result.get("reasoning", "")
            if self.verbose:
                print(f"[Dialogue] → ask: \"{question[:60]}...\"")

        # 8. Build dialogue history
        history = json.loads(session.get("dialogue_history", "[]") or "[]")
        history.append(self._build_history_entry(current_turn, "patient",
                                                   patient_input))
        if question:
            history.append(self._build_history_entry(current_turn, "agent",
                                                       question))
        elif recommendation:
            history.append(self._build_history_entry(
                current_turn, "agent",
                f"推荐: {recommendation.get('disease', '')} → "
                f"{recommendation.get('department', '')}"
            ))

        # 9. Update session in MySQL
        self._update_session(session_id, {
            "status": new_status,
            "current_turn": current_turn,
            "collected_symptoms": json.dumps(accumulated, ensure_ascii=False),
            "extracted_keywords": json.dumps(
                accumulated.get("keywords", []), ensure_ascii=False
            ),
            "candidate_diseases": json.dumps(candidates, ensure_ascii=False),
            "dialogue_history": json.dumps(history, ensure_ascii=False),
            "final_recommendation": json.dumps(recommendation, ensure_ascii=False)
            if recommendation else None,
        })

        # 10. Build response
        latency_ms = round((time.time() - start) * 1000, 1)

        return {
            "action": "recommend" if recommendation else "ask",
            "session_id": session_id,
            "current_turn": current_turn,
            "question": question,
            "question_reasoning": question_reasoning,
            "candidate_diseases": candidates[:3] if candidates else None,
            "recommendation": recommendation,
            "collected_info": self._summarize_collected(accumulated),
            "emergency_warning": None,
            "confidence": float(decision.get("confidence", 30)) / 100.0,
        }

    def get_session_state(self, session_id: str) -> Optional[dict]:
        """获取会话完整状态 (供 Java 查询)。"""
        self._ensure_table()
        session = self._load_session(session_id)
        if session is None:
            return None

        # Parse JSON fields
        for field in ["collected_symptoms", "candidate_diseases",
                       "dialogue_history", "final_recommendation"]:
            val = session.get(field)
            if isinstance(val, str) and val:
                try:
                    session[field] = json.loads(val)
                except json.JSONDecodeError:
                    pass

        # Parse keywords
        kw = session.get("extracted_keywords")
        if isinstance(kw, str) and kw:
            try:
                session["extracted_keywords"] = json.loads(kw)
            except json.JSONDecodeError:
                session["extracted_keywords"] = []

        return session

    def close_session(self, session_id: str) -> bool:
        """手动关闭会话。"""
        self._ensure_table()
        session = self._load_session(session_id)
        if session is None:
            return False

        self._update_session(session_id, {"status": "closed"})
        return True

    # ============================================================
    # Internal — Turn Logic
    # ============================================================

    def _check_emergency_keywords(self, text: str) -> list:
        """检查文本中是否包含紧急关键词 (快速规则匹配)。"""
        matched = []
        text_lower = text.lower()
        for kw in EMERGENCY_KEYWORDS:
            if kw.lower() in text_lower:
                matched.append(kw)
        return matched

    def _extract_symptoms(self, patient_input: str) -> dict:
        """使用 LLM 从患者输入中提取结构化症状 (Skill 1)。"""
        try:
            # Try using the existing extract_symptoms on DeepSeekClient
            result = self.llm_client.extract_symptoms(patient_input)
            if result and not result.get("error"):
                return {
                    "symptoms": result.get("main_symptoms", []),
                    "body_parts": result.get("body_parts", []),
                    "duration": result.get("duration", ""),
                    "severity": result.get("severity", ""),
                    "keywords": result.get("keywords", []),
                }
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] extract_symptoms 调用失败: {e}")

        # Fallback: lightweight extraction with own prompt
        try:
            response = self.llm_client.client.chat.completions.create(
                model=self.model or self.llm_client.model,
                messages=[
                    {"role": "system", "content": DEFAULT_EXTRACT_PROMPT},
                    {"role": "user", "content": patient_input},
                ],
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            data = self._parse_llm_json(raw)
            return {
                "symptoms": data.get("symptoms", []),
                "body_parts": data.get("body_parts", []),
                "duration": data.get("duration", ""),
                "severity": data.get("severity", ""),
                "keywords": data.get("keywords", []),
            }
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 症状提取 fallback 失败: {e}")
            # Ultimate fallback: treat entire input as one symptom
            return {
                "symptoms": [patient_input[:50]],
                "body_parts": [],
                "duration": "",
                "severity": "",
                "keywords": [patient_input[:20]],
            }

    def _merge_symptoms(self, existing: dict, new: dict) -> dict:
        """合并已有症状和新提取的症状 (去重)。"""
        if not existing:
            existing = {"symptoms": [], "body_parts": [], "duration": "",
                       "severity": "", "keywords": []}

        # Merge symptoms (deduplicate by lowercased name)
        seen = {s.lower() for s in existing.get("symptoms", [])}
        merged_symptoms = list(existing.get("symptoms", []))
        for s in new.get("symptoms", []):
            if s.lower() not in seen:
                merged_symptoms.append(s)
                seen.add(s.lower())

        # Merge body_parts
        seen_bp = {b.lower() for b in existing.get("body_parts", [])}
        merged_body_parts = list(existing.get("body_parts", []))
        for b in new.get("body_parts", []):
            if b.lower() not in seen_bp:
                merged_body_parts.append(b)
                seen_bp.add(b.lower())

        # Merge keywords
        seen_kw = {k.lower() for k in existing.get("keywords", [])}
        merged_keywords = list(existing.get("keywords", []))
        for k in new.get("keywords", []):
            if k.lower() not in seen_kw:
                merged_keywords.append(k)
                seen_kw.add(k.lower())

        return {
            "symptoms": merged_symptoms,
            "body_parts": merged_body_parts,
            # Use latest non-empty duration/severity
            "duration": new.get("duration") or existing.get("duration", ""),
            "severity": new.get("severity") or existing.get("severity", ""),
            "keywords": merged_keywords,
        }

    def _retrieve_diseases(self, accumulated_symptoms: dict) -> list:
        """RAG 检索候选疾病 (Skill 2: VectorStore)。"""
        symptoms = accumulated_symptoms.get("symptoms", [])
        keywords = accumulated_symptoms.get("keywords", [])

        # Build query from symptoms + keywords
        query_parts = symptoms[:3] + keywords[:2]
        if not query_parts:
            return []

        query = " ".join(query_parts)

        try:
            results = self.vector_store.search_disease(query, top_k=5)
            return [
                {
                    "disease": r.get("disease", ""),
                    "score": round(r.get("score", 0), 4),
                    "departments": r.get("departments", ""),
                    "symptoms": r.get("symptoms", ""),
                    "desc": r.get("desc", "")[:200],
                    "chain": r.get("chain", ""),
                    "category": r.get("category", ""),
                }
                for r in results
            ]
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] RAG 检索失败: {e}")
            return []

    def _decide_sufficient_info(
        self,
        accumulated_symptoms: dict,
        candidate_diseases: list,
        current_turn: int,
        max_turns: int,
    ) -> dict:
        """
        判断是否已收集足够信息 (Rule B + LLM Skill C)。

        Rule B guards:
          - symptom_count < MIN_SYMPTOM_THRESHOLD → force continue
          - current_turn >= max_turns → force recommend

        Otherwise → LLM decides (dialogue_decision scene)
        """
        symptom_count = len(accumulated_symptoms.get("symptoms", []))

        # Rule B: 症状数量门槛 — 强制追问
        if symptom_count < MIN_SYMPTOM_THRESHOLD and current_turn < max_turns:
            return {
                "decision": "continue",
                "confidence": 30,
                "reasoning": f"仅收集到 {symptom_count} 个症状，需要更多信息",
                "key_symptoms_count": symptom_count,
            }

        # Rule B: 最大轮数保护 — 强制推荐
        if current_turn >= max_turns:
            return {
                "decision": "recommend",
                "confidence": 60,
                "reasoning": "已达最大对话轮数，基于当前信息给出推荐",
                "key_symptoms_count": symptom_count,
            }

        # Rule C: LLM 自主判断
        try:
            symptoms_text = self._format_symptoms_for_prompt(accumulated_symptoms)
            diseases_text = self._format_diseases_for_prompt(candidate_diseases)

            # Load config from ai_config_loader
            try:
                from ai_config_loader import get_prompt, get_params
                system_prompt = get_prompt("dialogue_decision")
                cfg = get_params("dialogue_decision")
                _temp = cfg["temperature"]
                _max_tok = cfg["max_tokens"]
                _model = cfg["model"]
            except Exception:
                system_prompt = DEFAULT_DECISION_PROMPT
                _temp = 0.3
                _max_tok = 800
                _model = self.model or "qwen-flash"

            # Format placeholders in system_prompt
            system_prompt = system_prompt.replace(
                "{accumulated_symptoms}", symptoms_text
            ).replace("{candidate_diseases}", diseases_text).replace(
                "{current_turn}", str(current_turn)
            ).replace("{max_turns}", str(max_turns))

            user_message = (
                f"请根据已收集的症状和候选疾病，判断是否已有足够信息给出推荐。\n\n"
                f"已收集症状: {symptoms_text}\n\n"
                f"候选疾病: {diseases_text}\n\n"
                f"当前轮次: {current_turn}/{max_turns}"
            )

            response = self.llm_client.client.chat.completions.create(
                model=_model or self.model or self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            decision = self._parse_llm_json(raw)

            # Validate decision
            if decision.get("decision") not in ("continue", "recommend"):
                decision["decision"] = "continue"

            decision["key_symptoms_count"] = decision.get(
                "key_symptoms_count", symptom_count
            )
            return decision

        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 决策 LLM 调用失败: {e}")
            # Fallback: use simple heuristic
            if symptom_count >= 3:
                return {
                    "decision": "recommend",
                    "confidence": 60,
                    "reasoning": f"已收集 {symptom_count} 个症状 (fallback heuristic)",
                    "key_symptoms_count": symptom_count,
                }
            return {
                "decision": "continue",
                "confidence": 30,
                "reasoning": "决策 LLM 不可用，继续追问 (fallback)",
                "key_symptoms_count": symptom_count,
            }

    def _generate_followup_question(
        self,
        accumulated_symptoms: dict,
        candidate_diseases: list,
        session_history: list = None,
    ) -> dict:
        """
        生成鉴别诊断追问 (Skill 4: LLM Followup — dialogue_followup scene)。

        策略 C: 针对 Top-3 候选疾病的区分性特征提问。
        """
        if session_history is None:
            session_history = []

        try:
            symptoms_text = self._format_symptoms_for_prompt(accumulated_symptoms)
            diseases_text = self._format_diseases_for_prompt(candidate_diseases)

            # Build full dialogue context from history
            dialogue_lines = []
            for h in session_history[-12:]:  # Last 6 Q&A rounds
                role = "患者" if h.get("role") == "patient" else "医生"
                dialogue_lines.append(f"{role}: {h.get('content', '')}")
            dialogue_text = "\n".join(dialogue_lines) if dialogue_lines else "（首轮问诊）"

            # Load config
            try:
                from ai_config_loader import get_prompt, get_params
                system_prompt = get_prompt("dialogue_followup")
                cfg = get_params("dialogue_followup")
                _temp = cfg["temperature"]
                _max_tok = cfg["max_tokens"]
                _model = cfg["model"]
            except Exception:
                system_prompt = DEFAULT_FOLLOWUP_PROMPT
                _temp = 0.7
                _max_tok = 1024
                _model = self.model or "qwen-flash"

            # Inject FULL dialogue + constraint into system_prompt
            constraint = (
                f"\n\n## 完整问诊记录（已问过的不许再问）\n{dialogue_text}\n\n"
                f"## 核心规则\n"
                f"1. 阅读上述问诊记录，找出患者还没明确回答的鉴别方向\n"
                f"2. 如果某个方向已问过2次但患者无法回答，立刻换新方向\n"
                f"3. 绝对禁止用不同措辞问已出现过的问题"
            )
            system_prompt = system_prompt.replace(
                "{accumulated_symptoms}", symptoms_text
            ).replace("{candidate_diseases}", diseases_text) + constraint

            user_message = (
                f"已收集症状: {symptoms_text}\n"
                f"候选疾病: {diseases_text}\n\n"
                f"请基于问诊记录，选一个还没问过的新方向生成下一个问题。"
            )

            response = self.llm_client.client.chat.completions.create(
                model=_model or self.model or self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content.strip()
            return self._parse_llm_json(raw)

        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 追问生成失败: {e}")
            return {
                "question": "请更详细地描述您的症状，比如什么时候开始、"
                           "什么情况下加重或缓解？",
                "reasoning": "fallback — LLM 不可用，使用通用追问",
                "target_diseases": [],
            }

    def _generate_recommendation(
        self,
        accumulated_symptoms: dict,
        candidate_diseases: list,
    ) -> dict:
        """
        生成最终科室/疾病推荐 (Skill 5: LLM Recommendation)。

        复用 DeepSeekClient.recommend_department()。
        """
        try:
            # Build query from accumulated symptoms
            symptom_names = accumulated_symptoms.get("symptoms", [])
            keywords = accumulated_symptoms.get("keywords", [])
            query = " ".join(symptom_names + keywords) or "未明确症状"

            result = self.llm_client.recommend_department(
                user_query=query,
                rag_results=candidate_diseases,
            )

            if result and not result.get("parse_error"):
                return {
                    "department": result.get("department", ""),
                    "disease": result.get("disease", ""),
                    "confidence": round(result.get("confidence", 0), 2),
                    "reasoning": result.get("reasoning", ""),
                    "suggestion": result.get("suggestion", ""),
                    "alternative_departments": result.get(
                        "alternative_departments", []
                    ),
                    "emergency_warning": result.get("emergency_warning"),
                }
            else:
                # Fallback: use top RAG result
                if candidate_diseases:
                    top = candidate_diseases[0]
                    return {
                        "department": top.get("departments", "").split(",")[0].strip(),
                        "disease": top.get("disease", ""),
                        "confidence": round(top.get("score", 0.5), 2),
                        "reasoning": f"基于向量检索: {top.get('chain', '')}",
                        "suggestion": "建议尽快就诊，由医生进一步诊断确认。",
                        "alternative_departments": [],
                        "emergency_warning": None,
                    }
                return {
                    "department": "全科",
                    "disease": "待进一步诊断",
                    "confidence": 0.3,
                    "reasoning": "信息不足，建议线下就诊",
                    "suggestion": "建议前往社区医院或全科门诊进行初步检查。",
                    "alternative_departments": [],
                    "emergency_warning": None,
                }

        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 推荐生成失败: {e}")
            return {
                "department": "全科",
                "disease": "待进一步诊断",
                "confidence": 0.3,
                "reasoning": f"推荐服务暂时不可用: {str(e)[:50]}",
                "suggestion": "建议线下就诊，由医生面诊确认。",
                "alternative_departments": [],
                "emergency_warning": None,
            }

    # ============================================================
    # Internal — MySQL Persistence
    # ============================================================

    def _ensure_table(self) -> None:
        """Idempotent CREATE TABLE (runs once per manager instance)."""
        if self._table_ensured:
            return
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                c.execute(SQL_CREATE_DIALOGUE_SESSION)
            conn.commit()
            conn.close()
            self._table_ensured = True
            if self.verbose:
                print("[Dialogue] dialogue_session 表已就绪")
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 建表失败 (MySQL 可能未启动): {e}")

    def _get_mysql_conn(self):
        """创建 MySQL 连接 (pymysql, DictCursor, autocommit)。"""
        import pymysql
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "medical_rag"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _load_session(self, session_id: str) -> Optional[dict]:
        """从 MySQL 加载会话状态。"""
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                c.execute(
                    "SELECT * FROM dialogue_session WHERE session_id = %s",
                    (session_id,),
                )
                row = c.fetchone()
            conn.close()
            return row
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 加载会话失败: {e}")
            return None

    def _create_session(
        self,
        session_id: str,
        patient_id: Optional[int],
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict:
        """在 MySQL 中创建新会话。"""
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                c.execute(
                    "INSERT INTO dialogue_session "
                    "(session_id, patient_id, status, collected_symptoms, "
                    " extracted_keywords, candidate_diseases, dialogue_history, "
                    " final_recommendation, max_turns, current_turn) "
                    "VALUES (%s, %s, 'active', '{}', '[]', '[]', '[]', NULL, %s, 0)",
                    (session_id, patient_id, max_turns),
                )
            conn.commit()
            conn.close()

            return self._load_session(session_id)
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 创建会话失败: {e}")
            # Return in-memory session if MySQL is down
            return {
                "session_id": session_id,
                "patient_id": patient_id,
                "status": "active",
                "current_turn": 0,
                "max_turns": max_turns,
                "collected_symptoms": "{}",
                "extracted_keywords": "[]",
                "candidate_diseases": "[]",
                "dialogue_history": "[]",
                "final_recommendation": None,
            }

    def _update_session(self, session_id: str, updates: dict) -> bool:
        """更新 MySQL 中的会话状态。"""
        try:
            # Build SET clause
            set_parts = []
            values = []
            for key, value in updates.items():
                if value is not None or key in ("final_recommendation",):
                    set_parts.append(f"{key} = %s")
                    values.append(value)

            if not set_parts:
                return True

            values.append(session_id)
            sql = f"UPDATE dialogue_session SET {', '.join(set_parts)} WHERE session_id = %s"

            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                c.execute(sql, values)
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            if self.verbose:
                print(f"[Dialogue] 更新会话失败: {e}")
            return False

    # ============================================================
    # Internal — Formatting & Parsing
    # ============================================================

    def _format_symptoms_for_prompt(self, accumulated: dict) -> str:
        """格式化已收集症状为 LLM prompt 可读文本。"""
        if not accumulated:
            return "（暂无已收集的症状信息）"

        parts = []

        symptoms = accumulated.get("symptoms", [])
        if symptoms:
            parts.append(f"症状: {', '.join(symptoms)}")

        body_parts = accumulated.get("body_parts", [])
        if body_parts:
            parts.append(f"部位: {', '.join(body_parts)}")

        duration = accumulated.get("duration", "")
        if duration:
            parts.append(f"持续时间: {duration}")

        severity = accumulated.get("severity", "")
        if severity:
            parts.append(f"严重程度: {severity}")

        return "\n".join(parts) if parts else "（暂无已收集的症状信息）"

    def _format_diseases_for_prompt(
        self, diseases: list, top_n: int = 3
    ) -> str:
        """格式化候选疾病为 LLM prompt 可读文本。"""
        if not diseases:
            return "（暂无候选疾病）"

        lines = []
        for i, d in enumerate(diseases[:top_n]):
            lines.append(
                f"{i+1}. {d['disease']} "
                f"(匹配度: {d['score']:.1%}, 科室: {d.get('departments', 'N/A')})\n"
                f"   特征症状: {d.get('symptoms', 'N/A')[:150]}"
            )

        return "\n".join(lines)

    def _parse_llm_json(self, raw_text: str) -> dict:
        """
        4 层 JSON 解析容错 (与 suggestion_generator.py 一致)。

        1. 直接 json.loads
        2. 提取 ```json ... ``` 代码块
        3. 提取最外层 { ... }
        4. 返回安全默认值
        """
        # Layer 1
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # Layer 2
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Layer 3
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                pass

        # Layer 4: safe default
        return {}

    def _build_history_entry(
        self, turn: int, role: str, content: str
    ) -> dict:
        """构建对话历史条目。"""
        return {
            "turn": turn,
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _summarize_collected(self, accumulated: dict) -> dict:
        """生成已收集信息的可读摘要 (供 AgentResponse.collected_info)。"""
        if not accumulated:
            return None
        return {
            "symptoms": accumulated.get("symptoms", []),
            "body_parts": accumulated.get("body_parts", []),
            "duration": accumulated.get("duration", ""),
            "severity": accumulated.get("severity", ""),
        }
