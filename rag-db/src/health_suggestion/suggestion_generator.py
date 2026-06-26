"""
suggestion_generator.py
个性化生活建议生成器 — RAG 增强版

从 Java 后端传入的 health_record + consultation 两张表数据:
  - health_record: member_name, gender, birth_date, blood_type,
    allergy, past_illness, surgery_history, medication (不含 ai_summary)
  - consultation: symptom_text, doctor_advice, ai_analysis, consultation_dialog

→ RAG 知识库检索 → LLM 生成 5 类结构化生活建议

架构位置:
  Java → POST /api/rag/health-suggestion → 本模块
    → RAG 检索相关知识 → LLM 生成 5 类建议 (JSON)
    → Java 写入 health_suggestion 表

设计约束 (用户指定):
  - 输出: 结构化 JSON (方案 A — 按 category 分组)
  - 面向患者 (非医生), 语言通俗易懂
  - 增强: RAG 检索, 无匹配时 LLM 直接生成
  - 触发: Java 主动调用 API
  - 配置: MySQL ai_model_config 表 (scene=health_suggestion)"""

import os, sys, json, time, re
from typing import Optional

# Ensure parent src/ is importable
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(_src, "..", "..", ".env"))

# Categories defined in health_suggestion 表
CATEGORIES = ["diet", "exercise", "sleep", "medication", "seasonal"]
CATEGORY_LABELS = {
    "diet": "饮食建议",
    "exercise": "运动建议",
    "sleep": "睡眠建议",
    "medication": "用药建议",
    "seasonal": "季节性建议",
}


# ============================================================
# Prompt Template — 默认可被 MySQL 覆盖
# ============================================================

DEFAULT_SYSTEM_PROMPT = """你是一位资深健康管理师，兼具临床营养师、运动康复师和用药指导师的专业背景。
你的任务是根据患者的健康档案和问诊记录，结合医学知识库，为患者生成个性化的生活建议。

## 输出要求
请严格按照以下 JSON 格式输出，不要添加任何额外文字：

{
  "suggestions": [
    {
      "category": "diet",
      "items": [
        {"title": "建议标题", "content": "具体建议内容，约30-60字，通俗易懂"}
      ]
    },
    {
      "category": "exercise",
      "items": [...]
    },
    {
      "category": "sleep",
      "items": [...]
    },
    {
      "category": "medication",
      "items": [...]
    },
    {
      "category": "seasonal",
      "items": [...]
    }
  ]
}

## 五大类别说明

### 1. diet (饮食建议)
- 根据疾病和用药情况，给出具体的饮食调整建议
- 标注需要忌口或限制的食物
- 推荐有益的食物和营养素
- 考虑药物-食物相互作用

### 2. exercise (运动建议)
- 根据身体状况推荐合适的运动类型、频率和强度
- 标注运动禁忌和注意事项
- 慢性病患者给出针对性运动处方

### 3. sleep (睡眠建议)
- 根据疾病和用药情况给出睡眠卫生建议
- 标注可能影响睡眠的因素（如药物副作用、疼痛等）
- 提供改善睡眠质量的具体方法

### 4. medication (用药建议)
- 强调遵医嘱服药的重要性
- 提醒常见副作用及应对方法
- 标注药物相互作用风险和服药时间注意事项
- 如有过敏药物，突出警告

### 5. seasonal (季节性建议)
- 根据当前季节和患者疾病给出时令建议
- 流感季节的预防措施
- 慢性病在不同季节的管理要点
- 当前为夏季，重点关注防暑降温、饮食卫生、蚊虫防护等

## 约束
- 每个 category 必须有 1-3 条建议
- 语言通俗易懂，面向普通患者（非医生）
- 建议要具体、可操作，不要泛泛而谈
- 充分结合患者的具体疾病和用药情况
- 如有知识库检索到的信息，优先采纳
- 对过敏药物必须明确标注警告
- 不确定的内容请标注「建议咨询医生」
- 输出严格 JSON，不要有 markdown 代码块标记"""


# ============================================================
# HealthSuggestionGenerator
# ============================================================

class HealthSuggestionGenerator:
    """
    个性化生活建议生成器。

    使用:
        gen = HealthSuggestionGenerator(verbose=True)
        result = gen.generate(
            health_record={
                "member_name": "张三",
                "gender": 1,
                "past_illness": "高血压5年, 2型糖尿病3年",
                "allergy": "青霉素过敏",
                "medication": "硝苯地平 30mg qd, 二甲双胍 500mg bid",
            },
            consultation={
                "symptom_text": "最近经常头晕，血压偏高",
                "doctor_advice": "建议低盐低脂饮食，规律服药，每周监测血压",
            },
        )
        for cat in result["suggestions"]:
            print(f"【{cat['category']}】")
            for item in cat["items"]:
                print(f"  - {item['title']}: {item['content']}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_tokens: int = 1500,
        verbose: bool = False,
    ):
        """
        Args:
            model: LLM 模型名, 默认从 ai_config_loader 读取 health_suggestion 场景
            temperature: 生成温度 (建议 0.3-0.5)
            max_tokens: 最大输出长度
            verbose: 打印进度日志
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        # Lazy-loaded components
        self._llm_client = None
        self._vector_store = None

    # ============================================================
    # Lazy-loaded components (复用现有模块)
    # ============================================================

    @property
    def llm_client(self):
        """复用 DeepSeekClient。"""
        if self._llm_client is None:
            import importlib.util as _iu
            _dc = os.path.join(_src, "deepseek_client.py")
            _spec = _iu.spec_from_file_location("deepseek_client", _dc)
            _mod = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self._llm_client = _mod.DeepSeekClient(model=self.model)
        return self._llm_client

    @property
    def vector_store(self):
        """复用 VectorStore 做 RAG 检索。"""
        if self._vector_store is None:
            import importlib.util as _iu
            _qe = os.path.join(_src, "query_engine.py")
            _spec = _iu.spec_from_file_location("query_engine", _qe)
            _mod = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self._vector_store = _mod.VectorStore()
        return self._vector_store

    # ============================================================
    # Core: 生成建议
    # ============================================================

    def generate(self, health_record: Optional[dict] = None,
                 consultation: Optional[dict] = None) -> dict:
        """
        生成个性化生活建议。

        Args:
            health_record: health_record 表数据 (不含 ai_summary), 字段:
                {
                    "record_id":    1,                # optional
                    "patient_id":   1,                # optional
                    "member_name":  "张三",
                    "is_self":      1,                # 1=本人 0=家庭成员
                    "gender":       1,                # 1=男 2=女
                    "birth_date":   "1960-03-15",
                    "blood_type":   "A",
                    "allergy":      "青霉素过敏",
                    "past_illness": "高血压, 糖尿病",
                    "surgery_history": "阑尾切除术",
                    "medication":   "硝苯地平 30mg qd",
                }
            consultation: consultation 表 AI 相关字段:
                {
                    "consult_id":   1,                # optional
                    "symptom_text": "最近头晕乏力...",
                    "doctor_advice": "建议低盐饮食...",
                    "ai_analysis":  {...},            # optional
                    "consultation_dialog": "...",     # optional
                }

        Returns:
            {
                "suggestions": [
                    {
                        "category": "diet",
                        "items": [
                            {"title": "低盐饮食", "content": "建议每日食盐..."},
                        ],
                    },
                    ...
                ],
                "metadata": {
                    "model": "qwen-flash",
                    "latency_ms": 3500,
                    "tokens": {"total_tokens": 1200},
                },
                "rag_context_used": true,
            }
        """
        start = time.time()

        health_record = health_record or {}
        consultation = consultation or {}

        # 1. 格式化输入为可读文本
        hr_text = self._format_health_record(health_record)
        consult_text = self._format_consultation(consultation)

        # 2. RAG 检索
        rag_context = ""
        rag_used = False
        try:
            rag_context = self._retrieve_rag_context(health_record, consultation)
            rag_used = bool(rag_context)
        except Exception as e:
            if self.verbose:
                print(f"[HealthSuggestion] RAG 检索失败: {e}")

        # 3. 加载配置
        try:
            from ai_config_loader import get_prompt, get_params
            system_prompt = get_prompt("health_suggestion")
            cfg = get_params("health_suggestion")
            _temp = cfg["temperature"]
            _max_tok = cfg["max_tokens"]
            _model = cfg["model"]
        except Exception:
            system_prompt = DEFAULT_SYSTEM_PROMPT
            _temp = self.temperature
            _max_tok = self.max_tokens
            _model = self.model or "qwen-flash"

        # 4. 构建 user message
        user_message = f"""## 患者健康档案
{hr_text}

## 问诊记录
{consult_text}

## 医学知识库参考 (RAG 检索)
{rag_context or "（未检索到高度相关的疾病知识，请根据患者数据直接生成建议）"}

请根据以上信息，为该患者生成 5 类个性化生活建议。"""

        if self.verbose:
            print(f"[HealthSuggestion] 正在生成建议...")
            print(f"  档案: {hr_text[:80]}...")
            print(f"  问诊: {consult_text[:80]}...")

        # 5. 调用 LLM
        try:
            response = self.llm_client.client.chat.completions.create(
                model=_model or self.model or self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
            )

            raw_text = response.choices[0].message.content.strip()
            latency_ms = round((time.time() - start) * 1000, 1)

            # 6. 解析 JSON
            suggestions = self._parse_llm_response(raw_text)

            result = {
                "suggestions": suggestions,
                "metadata": {
                    "model": response.model,
                    "latency_ms": latency_ms,
                    "tokens": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    } if response.usage else None,
                },
                "rag_context_used": rag_used,
            }

            if self.verbose:
                total_items = sum(len(cat.get("items", [])) for cat in suggestions)
                print(f"[HealthSuggestion] 生成完成, {latency_ms}ms, "
                      f"{len(suggestions)} 类, {total_items} 条建议")

            return result

        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 1)
            return {
                "suggestions": self._build_fallback_suggestions(),
                "error": f"AI服务调用失败: {str(e)}",
                "metadata": {"latency_ms": latency_ms},
                "rag_context_used": rag_used,
            }

    # ============================================================
    # Helpers — 格式化输入
    # ============================================================

    def _format_health_record(self, record: dict) -> str:
        """将 health_record dict 格式化为 LLM 可读的文本 (不含 ai_summary)。"""
        if not record:
            return "（无健康档案数据）"

        lines = []

        # 基本信息
        name = record.get("member_name", "未知")
        gender_map = {1: "男", 2: "女"}
        gender = gender_map.get(record.get("gender"), "未知")
        birth = record.get("birth_date", "")

        # 计算年龄
        age_str = ""
        if birth:
            try:
                from datetime import date
                y = int(str(birth)[:4])
                age = date.today().year - y
                age_str = f"，{age}岁"
            except Exception:
                pass

        blood = record.get("blood_type", "")
        blood_str = f"，{blood}型血" if blood else ""

        is_self = record.get("is_self", 1)
        identity = "本人" if is_self == 1 else "家庭成员"

        lines.append(f"姓名: {name}，性别: {gender}{age_str}{blood_str}，档案类型: {identity}")

        # 关键字段
        field_map = {
            "past_illness": "既往病史",
            "allergy": "过敏史",
            "surgery_history": "手术史",
            "medication": "当前用药",
        }

        for key, label in field_map.items():
            value = record.get(key)
            if value and str(value).strip():
                lines.append(f"{label}: {value}")

        return "\n".join(lines)

    def _format_consultation(self, consult: dict) -> str:
        """将 consultation dict 格式化为 LLM 可读的文本。"""
        if not consult:
            return "（无问诊记录）"

        lines = []

        # 症状描述
        symptom = consult.get("symptom_text")
        if symptom and str(symptom).strip():
            lines.append(f"患者主诉: {symptom}")

        # 医生回复
        advice = consult.get("doctor_advice")
        if advice and str(advice).strip():
            lines.append(f"医生建议: {advice}")

        # AI 分析结果
        ai_analysis = consult.get("ai_analysis")
        if ai_analysis:
            if isinstance(ai_analysis, dict):
                # 提取关键字段
                urgency = ai_analysis.get("urgency", "")
                diseases = ai_analysis.get("possible_diseases", [])
                if urgency:
                    lines.append(f"紧急程度: {urgency}")
                if diseases:
                    disease_names = [d if isinstance(d, str) else d.get("name", "")
                                     for d in diseases]
                    lines.append(f"AI 分析可能疾病: {', '.join(disease_names)}")
            elif isinstance(ai_analysis, str) and ai_analysis.strip():
                lines.append(f"AI 分析: {ai_analysis[:200]}")

        # 对话历史 (取最后 500 字)
        dialog = consult.get("consultation_dialog")
        if dialog and str(dialog).strip():
            dialog_str = str(dialog)
            if len(dialog_str) > 500:
                dialog_str = dialog_str[-500:]
            lines.append(f"问诊对话摘要: {dialog_str}")

        if not lines:
            return "（无问诊记录）"

        return "\n".join(lines)

    # ============================================================
    # Helpers — RAG 检索
    # ============================================================

    def _retrieve_rag_context(self, health_record: dict,
                              consultation: dict) -> str:
        """
        RAG 检索: 根据既往病史 + 症状描述检索相关知识。

        检索策略:
          1. 从 past_illness 提取疾病关键词
          2. 从 symptom_text 提取症状关键词
          3. 从 medication 提取药物相关查询
          4. 聚合 Top-5 结果
        """
        keywords = set()

        # 从既往病史提取关键词
        past = health_record.get("past_illness", "")
        if past and str(past).strip():
            tokens = re.split(r'[,，\s、/；;]+', str(past))
            for k in tokens:
                k = k.strip()
                if len(k) >= 2:
                    keywords.add(k)

        # 从症状描述提取关键词
        symptom = consultation.get("symptom_text", "")
        if symptom and str(symptom).strip():
            tokens = re.split(r'[,，\s、/；;。！？]+', str(symptom))
            for k in tokens:
                k = k.strip()
                if 2 <= len(k) <= 12:
                    keywords.add(k)

        # 从用药提取药物类别
        medication = health_record.get("medication", "")
        if medication and str(medication).strip():
            keywords.add(f"药物注意事项 {str(medication)[:80]}")

        # 从 AI 分析提取可能疾病
        ai_analysis = consultation.get("ai_analysis")
        if isinstance(ai_analysis, dict):
            diseases = ai_analysis.get("possible_diseases", [])
            for d in diseases:
                name = d if isinstance(d, str) else d.get("name", "")
                if name and len(name) >= 2:
                    keywords.add(name)

        if not keywords:
            return ""

        # 用前 3 个关键词构建主查询
        keyword_list = list(keywords)[:3]
        main_query = " ".join(keyword_list)

        try:
            diseases = self.vector_store.search_disease(main_query, top_k=5)
        except Exception:
            return ""

        if not diseases:
            return ""

        lines = ["以下是与患者状况相关的医学知识 (供参考):"]
        for i, d in enumerate(diseases):
            lines.append(
                f"{i+1}. {d['disease']} (相关度: {d['score']:.1%})\n"
                f"   症状: {d['symptoms']}\n"
                f"   科室: {d['departments']}\n"
                f"   简介: {d.get('desc', '')[:150]}"
            )

        return "\n".join(lines)

    # ============================================================
    # Helpers — JSON 解析
    # ============================================================

    def _parse_llm_response(self, raw_text: str) -> list:
        """
        从 LLM 原始输出中解析结构化建议。

        容错策略:
          1. 尝试直接 json.loads
          2. 尝试提取 ```json ... ``` 代码块
          3. 尝试提取 { ... } 最外层 JSON
          4. 全部失败则返回 fallback
        """
        # 策略 1: 直接解析
        try:
            data = json.loads(raw_text)
            return self._validate_suggestions(data)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 ```json ... ``` 代码块
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_text)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                return self._validate_suggestions(data)
            except json.JSONDecodeError:
                pass

        # 策略 3: 提取最外层 {...}  (从第一个 { 到最后一个 })
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                data = json.loads(raw_text[start_idx:end_idx + 1])
                return self._validate_suggestions(data)
            except json.JSONDecodeError:
                pass

        # 全部失败: 返回空结构
        if self.verbose:
            print(f"[HealthSuggestion] JSON 解析失败，使用 fallback。原始输出前 200 字: "
                  f"{raw_text[:200]}")
        return self._build_fallback_suggestions()

    def _validate_suggestions(self, data: dict) -> list:
        """
        验证并规范化 LLM 输出的 suggestions 结构。
        确保 5 个 category 都存在，每个有 1-3 条 items。
        """
        suggestions = data.get("suggestions", [])

        if not isinstance(suggestions, list) or len(suggestions) == 0:
            return self._build_fallback_suggestions()

        # 构建 category → items 映射
        cat_map = {}
        for cat in suggestions:
            if isinstance(cat, dict):
                category = cat.get("category", "")
                items = cat.get("items", [])
                if isinstance(items, list):
                    valid_items = []
                    for item in items:
                        if isinstance(item, dict):
                            title = str(item.get("title", "")).strip()
                            content = str(item.get("content", "")).strip()
                            if title and content:
                                valid_items.append({"title": title, "content": content})
                    if category and valid_items:
                        cat_map[category] = valid_items[:3]  # 最多 3 条

        # 确保 5 个 category 都存在
        result = []
        for cat_name in CATEGORIES:
            items = cat_map.get(cat_name, [])
            if not items:
                # 该 category LLM 未生成，给一个通用提示
                items = [{
                    "title": f"建议咨询医生",
                    "content": f"关于{CATEGORY_LABELS[cat_name]}方面，建议您咨询主治医生获取个性化指导。"
                }]
            result.append({"category": cat_name, "items": items})

        return result

    def _build_fallback_suggestions(self) -> list:
        """
        构建降级输出 — 5 个空 category，提示咨询医生。
        保证 API 永远不会因为解析失败而返回 500。
        """
        return [
            {
                "category": cat,
                "items": [{
                    "title": "建议咨询医生",
                    "content": f"关于{CATEGORY_LABELS[cat]}方面，建议您咨询主治医生获取个性化指导。"
                }]
            }
            for cat in CATEGORIES
        ]

    # ============================================================
    # MySQL 持久化
    # ============================================================

    def save_to_mysql(self, suggestions: list, record_id: int,
                      patient_id: int, deactivate_old: bool = True) -> dict:
        """
        将生成的建议写入 medical_rag.health_suggestion 表。

        Args:
            suggestions: generate() 返回的 suggestions 列表
            record_id: 关联的健康档案ID
            patient_id: 患者ID
            deactivate_old: 是否先将该 record_id 的旧建议置为失效 (is_active=0)

        Returns:
            {"status": "ok", "inserted": N} 或 {"status": "error", "error": str}
        """
        import pymysql
        from datetime import datetime

        try:
            conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database="medical_rag",
                charset="utf8mb4",
            )

            with conn.cursor() as c:
                # 将旧建议标记为失效
                if deactivate_old:
                    c.execute(
                        "UPDATE health_suggestion SET is_active = 0 "
                        "WHERE record_id = %s AND is_active = 1",
                        (record_id,),
                    )

                # 插入新建议
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                inserted = 0
                insert_sql = (
                    "INSERT INTO health_suggestion "
                    "(record_id, patient_id, category, title, content, is_active, generated_at) "
                    "VALUES (%s, %s, %s, %s, %s, 1, %s)"
                )

                for cat in suggestions:
                    category = cat.get("category", "")
                    for item in cat.get("items", []):
                        title = item.get("title", "")
                        content = item.get("content", "")
                        if title and content:
                            c.execute(insert_sql, (
                                record_id, patient_id, category,
                                title, content, now,
                            ))
                            inserted += 1

            conn.commit()
            conn.close()

            if self.verbose:
                print(f"[HealthSuggestion] 已写入 MySQL: {inserted} 条建议 "
                      f"(record_id={record_id}, patient_id={patient_id})")

            return {"status": "ok", "inserted": inserted}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def fetch_from_mysql(self, record_id: int, active_only: bool = True) -> list:
        """
        从 medical_rag.health_suggestion 表读取建议。

        Args:
            record_id: 健康档案ID
            active_only: 是否只返回当前生效的建议 (is_active=1)

        Returns:
            [{"category": "diet", "items": [...]}, ...]
        """
        import pymysql

        try:
            conn = pymysql.connect(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database="medical_rag",
                charset="utf8mb4",
            )

            with conn.cursor(pymysql.cursors.DictCursor) as c:
                sql = (
                    "SELECT category, title, content FROM health_suggestion "
                    "WHERE record_id = %s"
                )
                if active_only:
                    sql += " AND is_active = 1"
                sql += " ORDER BY category, suggestion_id"
                c.execute(sql, (record_id,))
                rows = c.fetchall()

            conn.close()

            # 按 category 分组
            cat_map = {}
            for row in rows:
                cat = row["category"]
                if cat not in cat_map:
                    cat_map[cat] = []
                cat_map[cat].append({
                    "title": row["title"],
                    "content": row["content"],
                })

            return [
                {"category": cat, "items": cat_map.get(cat, [])}
                for cat in CATEGORIES
            ]

        except Exception as e:
            if self.verbose:
                print(f"[HealthSuggestion] MySQL 读取失败: {e}")
            return []


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Health Suggestion Generator — 测试")
    print("=" * 60)

    gen = HealthSuggestionGenerator(verbose=True)

    # 完整输入
    test_result = gen.generate(
        health_record={
            "member_name": "张三",
            "gender": 1,
            "birth_date": "1960-03-15",
            "blood_type": "A",
            "allergy": "青霉素过敏（皮疹）, 头孢类抗生素",
            "past_illness": "高血压病5年, 2型糖尿病3年, 高脂血症",
            "surgery_history": "阑尾切除术 2019年",
            "medication": "硝苯地平缓释片 30mg qd, 二甲双胍 500mg bid, 阿托伐他汀 20mg qn",
            "is_self": 1,
        },
        consultation={
            "symptom_text": "最近经常头晕，血压偏高，偶尔心慌",
            "doctor_advice": "建议低盐低脂饮食，规律服药，每周监测血压血糖，3个月后复查",
            "ai_analysis": {
                "urgency": "普通",
                "possible_diseases": ["原发性高血压", "2型糖尿病", "高脂血症"],
            },
        },
    )

    if test_result.get("error"):
        print(f"  [ERROR] {test_result['error']}")
    else:
        print(f"\n  Generated Suggestions:")
        for cat in test_result["suggestions"]:
            print(f"\n  [{cat['category']}] {CATEGORY_LABELS.get(cat['category'], '')}")
            for item in cat["items"]:
                print(f"    [Title] {item['title']}")
                print(f"    [Content] {item['content']}")
        print(f"\n  Metadata: {test_result['metadata']}")
        print(f"  RAG used: {test_result['rag_context_used']}")

    # 最小输入
    print(f"\n{'─' * 50}")
    print("  Minimal input test:")
    result2 = gen.generate(
        health_record={
            "member_name": "李四",
            "gender": 2,
            "past_illness": "缺铁性贫血",
            "is_self": 1,
        },
    )
    if not result2.get("error"):
        for cat in result2["suggestions"]:
            print(f"  [{cat['category']}] {len(cat['items'])} items")
