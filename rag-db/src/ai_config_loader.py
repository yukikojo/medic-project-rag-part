"""
ai_config_loader.py
AI 模型配置加载器 — 从 MySQL ai_model_config 表动态加载 Prompt 和 API 参数

Java 管理员通过管理后台修改 ai_model_config 表 → Python 实时读取生效
(带 60s 内存缓存, 也可通过 API 强制刷新)

支持的场景 (scene):
  triage          — 智能导诊科室推荐 (recommend_department)
  symptom_extract — 症状结构化提取 (extract_symptoms)
  query_optimize  — 查询优化标准化 (_optimize_with_llm)
  emr_extract     — 病历要素提取 (extract_medical_record)
  assist          — AI辅助问诊提示 (generate_assist_info)
  chat            — 通用对话 (chat)

使用:
  from ai_config_loader import get_config, get_prompt

  cfg = get_config("triage")
  # → {"model_name":"qwen-flash","temperature":0.3,"system_prompt":"你是一个...",...}

  prompt = get_prompt("emr_extract")
  # → "你是一位资深临床医师..."

API:
  POST /api/rag/config/refresh  — 强制刷新缓存
  GET  /api/rag/config/list     — 列出全部场景配置
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


# ============================================================
# MySQL 表创建 SQL
# ============================================================
SQL_CREATE_AI_CONFIG = """
CREATE TABLE IF NOT EXISTS ai_model_config (
    config_id    BIGINT PRIMARY KEY AUTO_INCREMENT,
    scene        VARCHAR(30)  NOT NULL COMMENT '业务场景: triage/symptom_extract/query_optimize/emr_extract/assist/chat',
    model_name   VARCHAR(50)  NOT NULL COMMENT '模型名称',
    api_base_url VARCHAR(200) NOT NULL COMMENT 'API 地址',
    api_key      VARCHAR(200) DEFAULT NULL COMMENT 'API Key (加密存储)',
    temperature  FLOAT        NOT NULL DEFAULT 0.3 COMMENT '生成多样性 0-1',
    max_tokens   INT          NOT NULL DEFAULT 1000 COMMENT '最大输出长度',
    top_p        FLOAT        NOT NULL DEFAULT 0.9 COMMENT '采样参数',
    system_prompt TEXT        NOT NULL COMMENT 'System Prompt 文本',
    status       TINYINT      NOT NULL DEFAULT 1 COMMENT '0=停用 1=启用',
    updated_by   BIGINT       DEFAULT NULL COMMENT '更新人ID',
    updated_at   DATETIME     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_at   DATETIME     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_scene (scene)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='AI模型配置 (Prompt+API参数)'
"""

# ============================================================
# 默认硬编码配置 (MySQL 不可用时的 fallback)
# ============================================================

_DEFAULT_CONFIGS = {
    "triage": {
        "scene": "triage",
        "model_name": os.getenv("LLM_MODEL", "qwen-flash"),
        "api_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.3,
        "max_tokens": 600,
        "top_p": 0.9,
        "system_prompt": """你是一个智能医疗导诊助手。根据用户的症状描述和医学知识库的检索结果，推荐最合适的就诊科室。

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
3. confidence 为 0-100 的整数, 表示推荐置信度百分比:
   - ≥80: 症状与知识库高度匹配, 推荐非常可靠
   - 60-79: 症状基本匹配, 有一定不确定性
   - 40-59: 症状部分匹配, 建议补充更多信息
   - <40: 匹配度较低, 建议用户详细描述症状
4. 如果症状包含"剧烈胸痛"、"大出血"、"意识不清"等危急描述，emergency_warning 应为 true，并在 suggestion 中建议立即就医
5. 推理依据要用通俗语言解释，让患者能理解为什么推荐这个科室""",
    },

    "symptom_extract": {
        "scene": "symptom_extract",
        "model_name": os.getenv("LLM_MODEL", "qwen-flash"),
        "api_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.1,
        "max_tokens": 400,
        "top_p": 0.9,
        "system_prompt": """你是一个医疗文本分析助手。从患者的症状描述中提取结构化信息。

请严格按照以下 JSON 格式输出:
{
  "main_symptoms": ["症状1", "症状2", ...],
  "duration": "持续时间 (如 '3天'、'1周'，未知则填 '未知')",
  "severity": "轻/中/重/未知",
  "body_parts": ["头部", "胸部", ...],
  "keywords": ["关键词1", "关键词2", ...]
}""",
    },

    "query_optimize": {
        "scene": "query_optimize",
        "model_name": os.getenv("OPTIMIZER_MODEL", "deepseek-v4-flash"),
        "api_base_url": os.getenv("OPTIMIZER_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("OPTIMIZER_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.1,
        "max_tokens": 500,
        "top_p": 0.9,
        "system_prompt": """你是一个中文医疗文本标准化助手。你的任务是将患者的口语化、方言化症状描述转化为标准化的医学症状术语。

## 规则
1. **口语→标准**: "肚子疼"→"腹痛"、"拉肚子"→"腹泻"、"想吐"→"恶心"、"发烧"→"发热"、"心慌"→"心悸"
2. **方言→标准**: "打摆子"→"寒战发热"、"心口疼"→"胃痛或心绞痛(根据上下文)"、"闹肚子"→"腹泻"
3. **保留关键修饰词**: 如果用户描述了程度/频率/位置 (如 "剧烈头痛"、"右下腹痛"、"持续咳嗽")，保留这些信息
4. **不要捏造**: 只标准化用户明确提到的症状，不要添加用户没说过的症状
5. **保留紧急信号**: 如 "剧烈胸痛"、"大出血"、"意识不清" 保留原样并标记 severity 为重
6. **多个症状用逗号分隔**

## 输出格式
请严格按照以下 JSON 格式输出 (不要输出其他内容):

{
  "standardized_text": "标准化后的症状描述, 用逗号分隔",
  "symptoms": ["标准化症状1", "标准化症状2", ...],
  "body_parts": ["头部", "胸部", ...],
  "duration": "持续时长 (未知则填'未知')",
  "severity": "轻/中/重/未知",
  "has_emergency_signals": false,
  "note": "标准化说明, 如'已将口语表达转换为标准术语'"
}""",
    },

    "emr_extract": {
        "scene": "emr_extract",
        "model_name": os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        "api_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.2,
        "max_tokens": 1200,
        "top_p": 0.9,
        "system_prompt": """你是一位资深临床医师，擅长从患者信息和症状描述中提取结构化病历要素。
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
- 所有输出使用中文""",
    },

    "assist": {
        "scene": "assist",
        "model_name": os.getenv("LLM_MODEL", "qwen-flash"),
        "api_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.3,
        "max_tokens": 1000,
        "top_p": 0.9,
        "system_prompt": """你是一位经验丰富的临床决策支持专家。根据患者信息，为接诊医生提供辅助问诊提示。

## 输出要求
请严格按照以下 JSON 格式输出:
{
  "disease_summary": "病情摘要，50-100字，概述核心症状和可能的病理机制",
  "follow_up_questions": ["需追问的问题1", "问题2", "..."],
  "differential_diagnosis": ["可能的鉴别诊断1", "鉴别诊断2", "..."],
  "suggested_exams": ["建议检查项目1", "检查项目2", "..."],
  "medication_suggestions": ["用药方向1 (注明需结合临床)", "..."],
  "referral_suggestions": ["如需转诊的建议科室1", "..."]
}

## 约束
- 所有建议均为辅助性质，需标注"请结合临床判断"
- 不要推荐特定品牌药品，只给药物类别方向
- 检查项目按优先级排序
- 鉴别诊断从高可能性到低可能性排列""",
    },

    "chat": {
        "scene": "chat",
        "model_name": os.getenv("LLM_MODEL", "qwen-flash"),
        "api_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "system_prompt": "你是一个智能医疗导诊助手，请基于医学知识回答用户的问题。",
    },

    "health_summary": {
        "scene": "health_summary",
        "model_name": os.getenv("LLM_MODEL", "qwen-flash"),
        "api_base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", ""),
        "temperature": 0.3,
        "max_tokens": 600,
        "top_p": 0.9,
        "system_prompt": """你是一位资深全科医师，擅长分析患者健康档案，输出专业的健康摘要。

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
- 格式要求：纯文本段落，不分条，不换行""",
    },
}

# ============================================================
# 配置加载器
# ============================================================

class AIConfigLoader:
    """
    MySQL-backed AI config loader with in-memory cache.

    Usage:
        loader = AIConfigLoader()
        cfg = loader.get_config("triage")
        prompt = loader.get_prompt("emr_extract")
    """

    def __init__(self, cache_ttl: int = 60, verbose: bool = False):
        """
        Args:
            cache_ttl: 缓存有效期 (秒)，默认 60s。0 = 永不过期。
            verbose: 是否打印日志
        """
        self.cache_ttl = cache_ttl
        self.verbose = verbose
        self._cache: dict = {}
        self._cache_time: float = 0.0
        self._mysql_available: Optional[bool] = None

    # ---- MySQL connection ----

    def _get_mysql_conn(self):
        import pymysql
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "medical_rag"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    # ---- Load from MySQL ----

    def _load_from_mysql(self) -> dict:
        """Load all active configs from MySQL ai_model_config table."""
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                c.execute(
                    "SELECT * FROM ai_model_config WHERE status = 1 ORDER BY scene"
                )
                rows = c.fetchall()
            conn.close()

            configs = {}
            for row in rows:
                configs[row["scene"]] = {
                    "scene": row["scene"],
                    "model_name": row["model_name"],
                    "api_base_url": row["api_base_url"],
                    "api_key": row.get("api_key") or _DEFAULT_CONFIGS.get(row["scene"], {}).get("api_key", ""),
                    "temperature": row["temperature"],
                    "max_tokens": row["max_tokens"],
                    "top_p": row["top_p"],
                    "system_prompt": row["system_prompt"],
                }

            self._mysql_available = True
            if self.verbose:
                print(f"[AIConfig] 从 MySQL 加载了 {len(configs)} 个场景配置")
            return configs

        except Exception as e:
            self._mysql_available = False
            if self.verbose:
                print(f"[AIConfig] MySQL 不可用: {e}，使用默认硬编码配置")
            return {}

    # ---- Public API ----

    def _refresh_if_needed(self):
        """Refresh cache if TTL expired."""
        now = time.time()
        if self.cache_ttl > 0 and (now - self._cache_time) > self.cache_ttl:
            self._cache = {}
        if not self._cache:
            mysql_configs = self._load_from_mysql()
            if mysql_configs:
                self._cache = mysql_configs
            else:
                self._cache = dict(_DEFAULT_CONFIGS)
            self._cache_time = now

    def get_config(self, scene: str) -> dict:
        """Get full config dict for a scene (temperature, max_tokens, prompt, etc.)."""
        self._refresh_if_needed()
        return self._cache.get(scene, _DEFAULT_CONFIGS.get(scene, {}))

    def get_prompt(self, scene: str) -> str:
        """Get system_prompt text for a scene."""
        cfg = self.get_config(scene)
        return cfg.get("system_prompt", "")

    def get_model(self, scene: str) -> str:
        """Get model_name for a scene."""
        return self.get_config(scene).get("model_name", "qwen-flash")

    def get_params(self, scene: str) -> dict:
        """Get LLM API params (model, temperature, max_tokens, top_p) for a scene."""
        cfg = self.get_config(scene)
        return {
            "model": cfg.get("model_name", "qwen-flash"),
            "temperature": cfg.get("temperature", 0.3),
            "max_tokens": cfg.get("max_tokens", 600),
            "top_p": cfg.get("top_p", 0.9),
        }

    def get_api_config(self, scene: str) -> dict:
        """Get API connection config (base_url, api_key, model) for a scene."""
        cfg = self.get_config(scene)
        return {
            "base_url": cfg.get("api_base_url", "https://api.deepseek.com"),
            "api_key": cfg.get("api_key", ""),
            "model": cfg.get("model_name", "qwen-flash"),
        }

    def list_scenes(self) -> list[str]:
        """List all available scenes."""
        self._refresh_if_needed()
        return sorted(self._cache.keys())

    def list_all(self) -> list[dict]:
        """List all configs with their details."""
        self._refresh_if_needed()
        result = []
        for scene, cfg in sorted(self._cache.items()):
            item = dict(cfg)
            # Mask api_key
            if item.get("api_key"):
                item["api_key"] = item["api_key"][:8] + "***"
            # Truncate prompt
            if item.get("system_prompt"):
                item["system_prompt_preview"] = item["system_prompt"][:80] + "..."
            result.append(item)
        return result

    def refresh(self) -> dict:
        """Force refresh cache from MySQL. Returns status."""
        self._cache = {}
        self._refresh_if_needed()
        return {
            "status": "ok",
            "scenes_loaded": len(self._cache),
            "source": "mysql" if self._mysql_available else "defaults",
        }

    def ensure_table(self) -> bool:
        """Ensure ai_model_config table exists (idempotent)."""
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                c.execute(SQL_CREATE_AI_CONFIG)
            conn.commit()
            conn.close()
            if self.verbose:
                print("[AIConfig] ai_model_config 表已就绪")
            return True
        except Exception as e:
            print(f"[AIConfig] 创建表失败: {e}")
            return False

    def seed_from_defaults(self) -> dict:
        """
        Seed the ai_model_config table from hardcoded defaults.
        Uses INSERT ... ON DUPLICATE KEY UPDATE so it's idempotent.
        """
        self.ensure_table()
        sql = """
            INSERT INTO ai_model_config
                (scene, model_name, api_base_url, api_key, temperature, max_tokens, top_p, system_prompt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                model_name = VALUES(model_name),
                api_base_url = VALUES(api_base_url),
                api_key = VALUES(api_key),
                temperature = VALUES(temperature),
                max_tokens = VALUES(max_tokens),
                top_p = VALUES(top_p),
                system_prompt = VALUES(system_prompt)
        """
        try:
            conn = self._get_mysql_conn()
            with conn.cursor() as c:
                for scene, cfg in _DEFAULT_CONFIGS.items():
                    c.execute(sql, (
                        scene,
                        cfg["model_name"],
                        cfg["api_base_url"],
                        cfg.get("api_key", ""),
                        cfg["temperature"],
                        cfg["max_tokens"],
                        cfg["top_p"],
                        cfg["system_prompt"],
                    ))
            conn.commit()
            conn.close()
            if self.verbose:
                print(f"[AIConfig] 默认配置已写入 {len(_DEFAULT_CONFIGS)} 个场景")
            return {"status": "ok", "seeded": len(_DEFAULT_CONFIGS)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ============================================================
# 全局单例
# ============================================================

_loader: Optional[AIConfigLoader] = None


def get_loader(cache_ttl: int = 60) -> AIConfigLoader:
    """获取全局 AIConfigLoader 单例"""
    global _loader
    if _loader is None:
        _loader = AIConfigLoader(cache_ttl=cache_ttl, verbose=True)
    return _loader


def get_config(scene: str) -> dict:
    """便捷函数: 获取场景配置"""
    return get_loader().get_config(scene)


def get_prompt(scene: str) -> str:
    """便捷函数: 获取场景 System Prompt"""
    return get_loader().get_prompt(scene)


def get_params(scene: str) -> dict:
    """便捷函数: 获取场景 LLM 参数"""
    return get_loader().get_params(scene)


# ============================================================
# 命令行: 建表 + 初始化数据
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Config Loader — 初始化 ai_model_config")
    print("=" * 60)

    loader = AIConfigLoader(verbose=True)

    # Create table + seed data
    print("\n[1] 建表...")
    loader.ensure_table()

    print("\n[2] 写入默认配置...")
    result = loader.seed_from_defaults()
    print(f"  Result: {result}")

    print("\n[3] 从 MySQL 读取验证...")
    loader.refresh()
    for scene in loader.list_scenes():
        cfg = loader.get_config(scene)
        print(f"  [{scene}] model={cfg['model_name']} temp={cfg['temperature']} max_tokens={cfg['max_tokens']}")
        print(f"           prompt preview: {cfg['system_prompt'][:60]}...")
