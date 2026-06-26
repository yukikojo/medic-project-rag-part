"""
summary_generator.py
健康档案 AI 摘要生成器 — RAG 增强版

从 Java 后端传入的 health_record 表数据:
  - member_name, gender, birth_date, blood_type
  - allergy, past_illness, surgery_history, medication
  - report_urls (可选)

→ 结合 RAG 知识库检索 → LLM 生成医生端的专业摘要

架构位置:
  Java UPDATE health_record → POST /api/rag/health-summary → 本模块
    → RAG 检索相关疾病 → LLM 生成 ai_summary 段落
    → Java 写入 health_record.ai_summary 字段

设计约束 (用户指定):
  - 输出: 自然语言段落 (非 JSON)
  - 增强: RAG 检索相关知识
  - 触发: Java 主动调用 API
  - 复用: ai_config_loader + DeepSeekClient"""

import os, sys, json, time
from typing import Optional

# Ensure parent src/ is importable
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(_src, "..", "..", ".env"))


# ============================================================
# Prompt Template — 默认可被 MySQL 覆盖
# ============================================================

DEFAULT_SYSTEM_PROMPT = """你是一位资深全科医师，擅长分析患者健康档案，输出专业的健康摘要。

## 任务
根据患者的健康档案数据，结合医学知识库检索结果，为接诊医生生成一份专业健康摘要。

## 输出格式
输出一段连续的自然语言文本（100-200字），包含以下要点：
1. 患者基本信息概括（年龄、性别、血型等）
2. 慢性病/既往病史总结，标注关键风险
3. 过敏史警告（如有）
4. 手术史概述（如有）
5. 当前用药汇总，标注可能的药物相互作用风险
6. 基于档案和知识库的综合风险评估与注意事项

## 约束
- 使用专业医学术语，面向医生读者
- 不要添加档案中不存在的信息
- 对已知过敏药物和高风险疾病组合做突出标注
- 如有知识库检索到的疾病风险信息，整合到摘要中
- 格式要求：纯文本段落，不分条，不换行"""


# ============================================================
# HealthSummaryGenerator
# ============================================================

class HealthSummaryGenerator:
    """
    健康档案 AI 摘要生成器。

    使用:
        gen = HealthSummaryGenerator(verbose=True)
        result = gen.generate({
            "member_name": "张三",
            "gender": 1,
            "birth_date": "1960-03-15",
            "blood_type": "A",
            "allergy": "青霉素过敏（皮疹）, 头孢类",
            "past_illness": "高血压病5年, 2型糖尿病3年",
            "surgery_history": "阑尾切除术 2019年",
            "medication": "硝苯地平缓释片 30mg qd, 二甲双胍 500mg bid",
            "is_self": 1,
        })
        print(result["ai_summary"])
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 600,
        verbose: bool = False,
    ):
        """
        Args:
            model: LLM 模型名, 默认从 ai_config_loader 读取 health_summary 场景
            temperature: 生成温度
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
    # Core: 生成摘要
    # ============================================================

    def generate(self, health_record: dict) -> dict:
        """
        生成健康档案 AI 摘要。

        Args:
            health_record: Java 端传入的 health_record 表数据, 字段对应概要设计:
                {
                    "record_id":    1,                # optional
                    "member_name":  "张三",            # 档案人姓名
                    "is_self":      1,                # 1=本人 0=家庭成员
                    "gender":       1,                # 1=男 2=女
                    "birth_date":   "1960-03-15",     # or None
                    "blood_type":   "A",              # or None
                    "allergy":      "青霉素过敏",       # or None
                    "past_illness": "高血压, 糖尿病",   # or None
                    "surgery_history": "阑尾切除术",    # or None
                    "medication":   "硝苯地平 30mg qd", # or None
                    "report_urls":  [...],            # optional
                }

        Returns:
            {
                "ai_summary": "患者张三，男，65岁，A型血。既往高血压病史5年...",
                "metadata": {
                    "model": "qwen-flash",
                    "latency_ms": 2500,
                    "tokens": {"total_tokens": 580},
                },
                "rag_context_used": true,
            }
        """
        start = time.time()

        # 1. 格式化档案为可读文本
        patient_text = self._format_record(health_record)

        # 2. RAG 检索相关疾病知识
        rag_context = ""
        rag_used = False
        try:
            rag_context = self._retrieve_rag_context(health_record)
            rag_used = bool(rag_context)
        except Exception as e:
            if self.verbose:
                print(f"[HealthSummary] RAG 检索失败: {e}")

        # 3. 加载配置
        try:
            from ai_config_loader import get_prompt, get_params
            system_prompt = get_prompt("health_summary")
            cfg = get_params("health_summary")
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
{patient_text}

## 医学知识库参考 (RAG检索)
{rag_context or "（未检索到高度相关的疾病知识）"}

请根据以上信息，生成该患者的专业健康摘要。"""

        if self.verbose:
            print(f"[HealthSummary] 正在生成摘要...")
            print(f"  档案: {patient_text[:80]}...")

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

            result = {
                "ai_summary": raw_text,
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
                print(f"[HealthSummary] 摘要生成完成, {latency_ms}ms")
                print(f"  摘要: {raw_text[:100]}...")

            return result

        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 1)
            return {
                "ai_summary": None,
                "error": f"AI服务调用失败: {str(e)}",
                "metadata": {"latency_ms": latency_ms},
                "rag_context_used": rag_used,
            }

    # ============================================================
    # Helpers
    # ============================================================

    def _format_record(self, record: dict) -> str:
        """将 health_record dict 格式化为 LLM 可读的文本。"""
        if not record:
            return "（无档案数据）"

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

        lines.append(f"姓名: {name}，性别: {gender}{age_str}{blood_str}")

        # 关键字段 (带标签)
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

        if lines:
            lines.append(f"档案类型: {'本人' if record.get('is_self', 1) == 1 else '家庭成员'}")

        return "\n".join(lines)

    def _retrieve_rag_context(self, record: dict) -> str:
        """
        RAG 检索: 根据档案中的既往病史和过敏信息检索相关知识。

        检索策略:
          1. 用 past_illness 作为 query 检索疾病库
          2. 用 allergy 作为 query 检索药物过敏相关知识
          3. 聚合 Top-5 结果作为 LLM 上下文
        """
        queries = []

        # 从既往病史中提取关键词
        past = record.get("past_illness", "")
        if past and str(past).strip():
            # 取前几个疾病关键词 (逗号/空格分隔)
            import re
            keywords = re.split(r'[,，\s、]+', str(past))
            queries.extend([k for k in keywords[:3] if len(k) >= 2])

        # 过敏信息
        allergy = record.get("allergy", "")
        if allergy and str(allergy).strip():
            queries.append(f"药物过敏 {str(allergy)[:50]}")

        if not queries:
            return ""

        # 对每个 query 检索, 取第一个 query 的结果为主
        main_query = " ".join(queries[:2])
        try:
            diseases = self.vector_store.search_disease(main_query, top_k=5)
        except Exception:
            return ""

        if not diseases:
            return ""

        lines = ["以下是与患者档案相关的医学知识:"]
        for i, d in enumerate(diseases):
            lines.append(
                f"{i+1}. {d['disease']} (相关度: {d['score']:.1%})\n"
                f"   症状: {d['symptoms']}\n"
                f"   科室: {d['departments']}\n"
                f"   简介: {d.get('desc', '')[:150]}"
            )

        return "\n".join(lines)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Health Summary Generator — 测试")
    print("=" * 60)

    gen = HealthSummaryGenerator(verbose=True)

    # 完整档案
    test_record = {
        "member_name": "张三",
        "gender": 1,
        "birth_date": "1960-03-15",
        "blood_type": "A",
        "allergy": "青霉素过敏（皮疹）, 头孢类抗生素",
        "past_illness": "高血压病5年, 2型糖尿病3年, 高脂血症",
        "surgery_history": "阑尾切除术 2019年",
        "medication": "硝苯地平缓释片 30mg qd, 二甲双胍 500mg bid, 阿托伐他汀 20mg qn",
        "is_self": 1,
    }

    result = gen.generate(test_record)
    if result.get("error"):
        print(f"  [ERROR] {result['error']}")
    else:
        print(f"\n  AI Summary:")
        print(f"  {result['ai_summary']}")
        print(f"\n  Metadata: {result['metadata']}")
        print(f"  RAG used: {result['rag_context_used']}")

    # 最小档案
    print(f"\n{'─' * 50}")
    print("  Minimal record test:")
    result2 = gen.generate({
        "member_name": "李四",
        "gender": 2,
        "past_illness": "缺铁性贫血",
        "is_self": 1,
    })
    if not result2.get("error"):
        print(f"  Summary: {result2['ai_summary'][:120]}...")
