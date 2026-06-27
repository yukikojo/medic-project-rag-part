"""
deepseek_client.py
通用 LLM 客户端 — 用于 RAG 系统的生成/推理层

支持所有兼容 OpenAI SDK 接口的 LLM 提供商（DeepSeek、通义千问、GLM、Moonshot 等）。
通过 .env 环境变量配置，保留 DeepSeek 默认值实现向后兼容。

配置方式 (.env 文件):
    # 生成层 (科室推荐) 配置
    LLM_API_KEY=sk-xxx        # 默认: DEEPSEEK_API_KEY
    LLM_BASE_URL=...          # 默认: https://api.deepseek.com
    LLM_MODEL=...             # 默认: deepseek-v4-flash

    # 查询优化器配置 (独立可设)
    OPTIMIZER_API_KEY=sk-xxx  # 默认: LLM_API_KEY → DEEPSEEK_API_KEY
    OPTIMIZER_BASE_URL=...    # 默认: LLM_BASE_URL → https://api.deepseek.com
    OPTIMIZER_MODEL=...       # 默认: deepseek-chat

使用示例:
    from generation.deepseek_client import DeepSeekClient

    # 方式1: 默认使用 DeepSeek (向后兼容)
    client = DeepSeekClient()

    # 方式2: 通过环境变量切换模型 (在 .env 中设置 LLM_API_KEY 等)

    # 方式3: 代码中显式指定
    client = DeepSeekClient(
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
    )

    # 基于 RAG 检索结果生成科室推荐
    result = client.recommend_department(
        user_query="头痛发热咳嗽",
        rag_results=[...]
    )
    print(result["department"])  # "呼吸内科"
"""

import os
import json
from typing import Optional

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from openai import OpenAI


# ============================================================
# 配置
# ============================================================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"          # DeepSeek-V4 Flash, 性价比最高
REASONER_MODEL = "deepseek-v4-pro"           # DeepSeek-V4 Pro, 复杂推理
OPTIMIZER_DEFAULT_MODEL = "deepseek-chat"    # 优化器默认模型 (轻量快速)


def _get_env_config(
    prefix: str,
    fallback_api_key_env: str = "DEEPSEEK_API_KEY",
    fallback_base_url: str = "https://api.deepseek.com",
    fallback_model: str = "deepseek-v4-flash",
) -> dict:
    """
    从环境变量读取指定前缀的 LLM 配置，支持多级 fallback。

    用于生成层和优化器各自独立读取配置，未设置时自动回退到 DeepSeek 默认值。

    Args:
        prefix: 环境变量前缀，如 "LLM" 或 "OPTIMIZER"
        fallback_api_key_env: api_key fallback 的环境变量名
        fallback_base_url: base_url 的最终 fallback 值
        fallback_model: model 的最终 fallback 值

    Returns:
        {"api_key": ..., "base_url": ..., "model": ...}
    """
    api_key = (
        os.getenv(f"{prefix}_API_KEY")
        or os.getenv(fallback_api_key_env)
    )
    base_url = (
        os.getenv(f"{prefix}_BASE_URL")
        or fallback_base_url
    )
    model = (
        os.getenv(f"{prefix}_MODEL")
        or fallback_model
    )
    return {"api_key": api_key, "base_url": base_url, "model": model}


class DeepSeekClient:
    """DeepSeek API 客户端 — 医疗导诊 RAG 的生成层"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # 多级 fallback: 参数 > LLM_* 环境变量 > DeepSeek 默认值
        _env = _get_env_config(
            prefix="LLM",
            fallback_api_key_env="DEEPSEEK_API_KEY",
            fallback_base_url=DEEPSEEK_BASE_URL,
            fallback_model=DEFAULT_MODEL,
        )
        self.api_key = api_key or _env["api_key"]
        if not self.api_key:
            raise ValueError(
                "LLM API Key 未设置。请在 .env 文件中设置 LLM_API_KEY 或 DEEPSEEK_API_KEY，"
                "或通过参数传入: DeepSeekClient(api_key='sk-...')"
            )

        self.base_url = base_url or _env["base_url"]
        self.model = model or _env["model"]

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    # ================================================================
    # 核心方法: 科室推荐 (RAG 检索 + LLM 推理)
    # ================================================================

    def recommend_department(
        self,
        user_query: str,
        rag_results: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        基于 RAG 检索结果, 用 DeepSeek 生成科室推荐 + 推理链

        这是 RAG 系统的核心方法:
          用户症状 → 向量检索 → DeepSeek 推理 → 结构化推荐

        Args:
            user_query: 用户症状描述, 如 "头痛发热咳嗽流鼻涕"
            rag_results: VectorStore.search_disease() 的返回结果
            model: 模型名, 默认 deepseek-chat
            temperature: 生成温度 (0-1), 越低越确定, 推荐 0.3
            max_tokens: 最大输出 token 数

        Returns:
            {
                "department": "呼吸内科",
                "disease": "感冒",
                "confidence": 85,
                "reasoning": "患者症状头痛、发热、咳嗽、流鼻涕与感冒高度匹配...",
                "suggestion": "建议前往呼吸内科就诊...",
                "alternative_departments": ["内科", "急诊科"],
                "emergency_warning": false,
                "raw_response": "..."  # 原始 LLM 输出
            }
        """
        # 加载配置 (MySQL 优先, 硬编码默认兜底)
        try:
            from ai_config_loader import get_prompt, get_params
            system_prompt = get_prompt("triage")
            cfg = get_params("triage")
            _temp = temperature if temperature is not None else cfg["temperature"]
            _max_tok = max_tokens if max_tokens is not None else cfg["max_tokens"]
        except Exception:
            system_prompt = """你是一个智能医疗导诊助手。根据用户的症状描述和医学知识库的检索结果，推荐最合适的就诊科室。

## 输出要求
请严格按照以下 JSON 格式输出 (不要输出其他内容):

{
  "department": "推荐的首选科室",
  "disease": "最可能的疾病",
  "confidence": 85,
  "reasoning": "推理依据 (50-100字，说明症状如何匹配到该疾病和科室)",
  "suggestion": "就医建议 (30-50字)",
  "alternative_departments": ["备选科室1", "备选科室2"],
  "emergency_warning": true/false
}

## 注意事项
1. department 必须是知识库中出现的科室名，不要捏造科室
2. 如果多个检索结果指向同一科室，提高该科室的推荐优先级
3. confidence 为 0-100 的整数
4. 危急描述触发 emergency_warning
5. 推理依据要用通俗语言解释，让患者能理解为什么推荐这个科室"""
            _temp = temperature if temperature is not None else 0.3
            _max_tok = max_tokens if max_tokens is not None else 600

        # 构造检索上下文
        context_parts = []
        for i, r in enumerate(rag_results[:5]):
            context_parts.append(
                f"{i+1}. {r['disease']} (相似度: {r['score']:.1%})\n"
                f"   症状: {r['symptoms']}\n"
                f"   科室: {r['departments']}\n"
                f"   分类: {r['category']}"
            )
        context = "\n".join(context_parts)

        user_message = f"""## 患者症状描述
{user_query}

## 知识库检索结果 (Top-{len(rag_results[:5])})
{context}

请根据以上信息，给出科室推荐。"""

        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content

            # 解析 JSON 输出
            try:
                result = json.loads(raw_text)
                # 确保 confidence 是整数
                if isinstance(result.get("confidence"), str):
                    conf_map = {"高": 85, "中": 60, "低": 35}
                    result["confidence"] = conf_map.get(result["confidence"], 50)
                elif not isinstance(result.get("confidence"), (int, float)):
                    result["confidence"] = 50
                else:
                    result["confidence"] = int(result["confidence"])
            except json.JSONDecodeError:
                # 如果 JSON 解析失败, 尝试从文本中提取
                result = {
                    "department": "无法解析",
                    "disease": "无法解析",
                    "confidence": 0,
                    "reasoning": "LLM 返回格式异常",
                    "suggestion": "请重新描述症状",
                    "alternative_departments": [],
                    "emergency_warning": False,
                    "parse_error": True,
                }

            result["raw_response"] = raw_text
            result["model"] = response.model
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return result

        except Exception as e:
            return {
                "department": "服务异常",
                "disease": "服务异常",
                "confidence": 0,
                "reasoning": f"DeepSeek API 调用失败: {str(e)}",
                "suggestion": "请稍后重试",
                "alternative_departments": [],
                "emergency_warning": False,
                "error": str(e),
            }

    # ================================================================
    # 辅助方法: 症状结构化提取
    # ================================================================

    def extract_symptoms(self, user_query: str) -> dict:
        """
        从用户自由文本中提取结构化症状关键词

        用途: UC-AI-01 症状结构化分析
        """
        # Try ai_config_loader first, fall back to hardcoded
        try:
            from ai_config_loader import get_prompt, get_params
            system_prompt = get_prompt("symptom_extract")
            cfg = get_params("symptom_extract")
            _temp = cfg["temperature"]
            _max_tok = cfg["max_tokens"]
            _model = cfg["model"]
        except Exception:
            system_prompt = """你是一个医疗文本分析助手。从患者的症状描述中提取结构化信息。

请严格按照以下 JSON 格式输出:
{
  "main_symptoms": ["症状1", "症状2", ...],
  "duration": "持续时间 (如 '3天'、'1周'，未知则填 '未知')",
  "severity": "轻/中/重/未知",
  "body_parts": ["头部", "胸部", ...],
  "keywords": ["关键词1", "关键词2", ...]
}"""
            _temp = 0.1
            _max_tok = 400
            _model = self.model

        try:
            response = self.client.chat.completions.create(
                model=_model or self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content
            return json.loads(raw_text)
        except Exception as e:
            return {"error": str(e), "main_symptoms": [], "keywords": []}

    # ================================================================
    # 通用对话
    # ================================================================

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """通用对话接口"""
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    # ================================================================
    # 健康检查
    # ================================================================

    def health_check(self) -> dict:
        """检查 API 连通性"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "回复 OK"}],
                max_tokens=10,
            )
            return {
                "status": "ok",
                "model": response.model,
                "latency_ms": int(response.usage.total_tokens),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ============================================================
# 完整 RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    完整 RAG Pipeline: 查询优化 → 向量检索 + LLM 生成

    流程:
       用户原始输入 → QueryOptimizer (口语标准化)
         → VectorStore 检索 (用标准化后的query)
         → DeepSeek LLM 推理生成 (基于检索结果)

    使用示例:
        # 基础使用 (自动启用查询优化)
        pipeline = RAGPipeline()
        result = pipeline.query("肚子疼拉稀想吐")
        print(result["recommendation"]["department"])  # "消化内科"

        # 对比: 不优化 vs 优化
        result = pipeline.query("肚子疼拉肚子", optimize=True)
        print(result["query_optimization"]["optimized_query"])  # "腹痛 腹泻"

        # 仅查询优化 (不执行完整 Pipeline)
        optimized = pipeline.optimize_query("心慌胸闷气短")
    """

    def __init__(
        self,
        deepseek_api_key: Optional[str] = None,
        deepseek_model: Optional[str] = None,
        deepseek_base_url: Optional[str] = None,
        optimizer_api_key: Optional[str] = None,
        optimizer_model: Optional[str] = None,
        optimizer_base_url: Optional[str] = None,
        optimizer_mode: str = "hybrid",   # "hybrid" | "llm" | "rule" | None (禁用)
        optimizer_cache: bool = True,
        reranker_enabled: bool = True,   # 是否启用 cross-encoder reranker
        verbose: bool = False,
    ):
        """
        Args:
            deepseek_api_key: 生成层 (科室推荐) LLM API Key，默认从 LLM_API_KEY 或 DEEPSEEK_API_KEY 读取
            deepseek_model: 生成层 LLM 模型名，默认从 LLM_MODEL 读取
            deepseek_base_url: 生成层 LLM API 地址，默认从 LLM_BASE_URL 读取
            optimizer_api_key: 优化器 LLM API Key，默认从 OPTIMIZER_API_KEY → LLM_API_KEY → DEEPSEEK_API_KEY 读取
            optimizer_model: 优化器 LLM 模型名，默认从 OPTIMIZER_MODEL 读取
            optimizer_base_url: 优化器 LLM API 地址，默认从 OPTIMIZER_BASE_URL → LLM_BASE_URL 读取
            optimizer_mode: 查询优化器模式
                - "hybrid": LLM + 规则兜底 (推荐)
                - "llm": 仅LLM
                - "rule": 仅规则词典
                - None: 禁用查询优化 (向后兼容)
            optimizer_cache: 是否启用优化器缓存
            reranker_enabled: 是否启用 cross-encoder reranker 精排
                - False (默认): 仅使用余弦相似度排序
                - True: 两阶段检索 — cosine粗排(20候选) → cross-encoder精排(top-5)
            verbose: 是否打印详细日志
        """
        # 延迟导入, 避免循环依赖
        import importlib.util
        _qe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "retrieval", "query_engine.py")
        _spec = importlib.util.spec_from_file_location("retrieval.query_engine", _qe_path)
        _qe = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_qe)

        self.vector_store = _qe.VectorStore(use_reranker=reranker_enabled)
        self.llm = DeepSeekClient(
            api_key=deepseek_api_key,
            model=deepseek_model,
            base_url=deepseek_base_url,
        )

        # 初始化查询优化器 (使用独立配置)
        self._optimizer = None
        self._optimizer_mode = optimizer_mode
        if optimizer_mode is not None:
            try:
                # 延迟导入 query_optimizer
                _qo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "retrieval", "query_optimizer.py")
                _qo_spec = importlib.util.spec_from_file_location("retrieval.query_optimizer", _qo_path)
                _qo = importlib.util.module_from_spec(_qo_spec)
                _qo_spec.loader.exec_module(_qo)
                self._optimizer = _qo.QueryOptimizer(
                    mode=optimizer_mode,
                    api_key=optimizer_api_key,
                    model=optimizer_model,
                    base_url=optimizer_base_url,
                    cache_enabled=optimizer_cache,
                    verbose=verbose,
                )
                if verbose:
                    print(f"[RAGPipeline] QueryOptimizer 已启用, 模式: {self._optimizer.mode}")
            except Exception as e:
                if verbose:
                    print(f"[RAGPipeline] QueryOptimizer 初始化失败: {e}, 将跳过查询优化")
                self._optimizer = None
                self._optimizer_mode = None

    def optimize_query(self, raw_query: str) -> dict:
        """
        仅执行查询优化, 不进行检索和推荐。
        适用于: 前端先展示标准化后的症状, 用户确认后再检索。

        Args:
            raw_query: 用户原始症状描述

        Returns:
            QueryOptimizer.optimize() 的完整返回字典
        """
        if self._optimizer:
            return self._optimizer.optimize(raw_query)
        else:
            return {
                "original_query": raw_query,
                "optimized_query": raw_query,
                "symptoms": [],
                "normalization_note": "查询优化器未启用",
            }

    def query(
        self,
        user_input: str,
        top_k: int = 5,
        optimize: bool = True,
    ) -> dict:
        """
        完整的 RAG 查询: 优化 → 检索 → 生成

        Args:
            user_input: 用户症状描述 (可包含口语化/方言表达)
            top_k: 检索返回的最大匹配数
            optimize: 是否启用查询优化 (默认 True)

        Returns:
            {
                "query": "肚子疼拉稀",              # 原始输入
                "query_optimization": {...},        # 查询优化结果 (新增)
                "rag_results": {...},               # 向量检索结果
                "recommendation": {...},            # LLM 生成的科室推荐
                "search_query": "腹痛 腹泻",        # 实际用于检索的query
            }
        """
        # Step 0: 查询优化 (标准化口语化/方言表达)
        query_optimization = None
        search_query = user_input  # 默认用原始输入检索

        if optimize and self._optimizer:
            opt_result = self._optimizer.optimize(user_input)
            query_optimization = opt_result

            # 使用优化后的查询进行向量检索
            optimized = opt_result.get("optimized_query", "").strip()
            if optimized and optimized != user_input:
                search_query = optimized
                if hasattr(self, '_verbose') and getattr(self, '_verbose', False):
                    print(f"[RAGPipeline] 查询优化: '{user_input}' → '{search_query}'")
        elif optimize and not self._optimizer:
            # 优化器不可用，使用原始输入 (静默降级)
            query_optimization = {
                "original_query": user_input,
                "optimized_query": user_input,
                "normalization_note": "查询优化器不可用, 使用原始输入",
            }

        # Step 1: 向量检索 (使用优化后的查询)
        rag_result = self.vector_store.comprehensive_search(search_query, top_k=top_k)

        # Step 2: LLM 推理生成 (传入原始输入保持上下文完整)
        recommendation = self.llm.recommend_department(
            user_query=user_input,
            rag_results=rag_result["disease_results"],
        )

        return {
            "query": user_input,
            "query_optimization": query_optimization,
            "rag_results": rag_result,
            "recommendation": recommendation,
            "search_query": search_query,
        }


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    pipeline = RAGPipeline(verbose=True)

    print("=" * 60)
    print("  RAG + DeepSeek 完整 Pipeline 测试 (含查询优化)")
    print("=" * 60)

    test_queries = [
        # (原始输入, 类型)
        ("头痛发热咳嗽流鼻涕", "标准术语"),
        ("肚子疼拉稀想吐没胃口", "口语化表达"),
        ("心慌胸闷气短胸口疼", "口语+标准混合"),
    ]

    for query, qtype in test_queries:
        print(f"\n{'─' * 50}")
        print(f">> 原始输入: {query} ({qtype})")

        # 先展示查询优化结果
        opt_result = pipeline.optimize_query(query)
        print(f"  优化查询: {opt_result.get('optimized_query', query)}")
        print(f"  提取症状: {opt_result.get('symptoms', [])}")
        print(f"  标准化方式: {opt_result.get('normalization_note', 'N/A')}")

        # 完整 Pipeline
        result = pipeline.query(query)

        rec = result.get("recommendation", {})
        if rec.get("error"):
            print(f"  [ERROR] {rec['error']}")
            continue

        print(f"  实际检索: {result.get('search_query', query)}")
        print(f"  推荐科室: {rec.get('department')}")
        print(f"  可能疾病: {rec.get('disease')}")
        print(f"  置信度:   {rec.get('confidence')}")
        print(f"  推理依据: {rec.get('reasoning', '')[:80]}...")
        print(f"  就医建议: {rec.get('suggestion', '')[:80]}...")
        print(f"  备选科室: {rec.get('alternative_departments', [])}")
        print(f"  危急警告: {rec.get('emergency_warning', False)}")
        if rec.get("usage"):
            u = rec["usage"]
            print(f"  Token 用量: {u['total_tokens']} (prompt={u['prompt_tokens']}, completion={u['completion_tokens']})")
