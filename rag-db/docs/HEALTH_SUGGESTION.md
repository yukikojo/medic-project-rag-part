# 个性化生活建议 — Health Suggestion Generator

> **版本**: v1.0  
> **日期**: 2026-06-26  
> **模块**: `src/health_suggestion/`, `ai_config_loader.py`, `api_server.py`, `api_models.py`

---

## 一、概述

为 Java 后端传入的 `health_record` + `consultation` 两张表的数据，生成面向**患者**的 5 类个性化生活建议，结合 RAG 知识库增强。输出存入 `health_suggestion` 表。

```
Java 传入:
  health_record (健康档案) + consultation (问诊记录)
      │
      ▼
RAG 检索 (根据既往病史 + 症状描述检索相关知识)
      │
      ▼ (无匹配时跳过 RAG，LLM 直接生成)
LLM 生成 (5 类结构化 JSON 建议)
      │
      ▼
返回 → Java 写入 health_suggestion 表 (5 类 × N 条记录)
```

---

## 二、与 health_summary 的区别

| 维度 | health_summary | health_suggestion |
|------|---------------|-------------------|
| 面向用户 | 医生 | 患者 |
| 输入 | health_record (单表) | health_record + consultation (双表) |
| 输出格式 | 自然语言段落 | 结构化 JSON (5 类分组) |
| 语言风格 | 专业医学术语 | 通俗易懂 |
| RAG 策略 | 必须 RAG | RAG 优先，无匹配时 LLM 直出 |
| 场景名称 | health_summary | health_suggestion |

---

## 三、输入输出

### 3.1 输入 (Java → Python)

#### health_record 字段（不含 ai_summary）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| member_name | str | 否 | 档案人姓名 |
| gender | int | 否 | 1=男 2=女 |
| birth_date | str | 否 | 出生日期 |
| blood_type | str | 否 | A/B/O/AB |
| allergy | str | 否 | 药物/食物过敏描述 |
| past_illness | str | 否 | 既往病史 |
| surgery_history | str | 否 | 手术史 |
| medication | str | 否 | 当前用药 |
| is_self | int | 否 | 1=本人 0=家庭成员 |
| record_id | int | 否 | 档案ID |
| patient_id | int | 否 | 患者ID |

#### consultation 字段（仅 AI 相关）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| symptom_text | str | 否 | 患者自述症状 |
| doctor_advice | str | 否 | 医生诊疗建议 |
| ai_analysis | dict | 否 | AI 结构化分析结果 JSON |
| consultation_dialog | str | 否 | 多轮追问对话历史 |
| consult_id | int | 否 | 问诊ID |

### 3.2 输出 (Python → Java)

```json
{
  "code": 200,
  "data": {
    "suggestions": [
      {
        "category": "diet",
        "items": [
          {
            "title": "低盐低脂饮食",
            "content": "建议每日食盐控制在5g以内，少吃腌制食品和加工肉制品，多吃新鲜蔬菜水果和全谷物。"
          },
          {
            "title": "控制碳水化合物摄入",
            "content": "主食定量，优先选择低GI食物如燕麦、糙米，每餐搭配优质蛋白延缓血糖上升。"
          }
        ]
      },
      {
        "category": "exercise",
        "items": [
          {
            "title": "中等强度有氧运动",
            "content": "建议每周进行3-5次快走、游泳或骑行，每次30-40分钟，运动时心率控制在(170-年龄)左右。"
          }
        ]
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
  },
  "metadata": {
    "model": "qwen-flash",
    "latency_ms": 3500,
    "tokens": {"prompt_tokens": 1200, "completion_tokens": 500, "total_tokens": 1700}
  },
  "rag_context_used": true
}
```

---

## 四、五大建议类别

| Category | 中文名 | 内容范围 | 示例 |
|----------|--------|---------|------|
| `diet` | 饮食建议 | 忌口食物、推荐食物、营养素补充、药物-食物相互作用 | 低盐<5g/日、避免西柚（他汀类药物） |
| `exercise` | 运动建议 | 运动类型、频率、强度、禁忌症 | 快走30min×5次/周、避免剧烈对抗运动 |
| `sleep` | 睡眠建议 | 作息规律、睡眠环境、药物对睡眠的影响 | 固定就寝时间、避免睡前使用电子设备 |
| `medication` | 用药建议 | 服药提醒、副作用应对、药物相互作用、过敏警告 | 二甲双胍餐后服用减轻胃肠反应 |
| `seasonal` | 季节性建议 | 时令防护、慢性病季节管理、疫苗接种提醒 | 夏季防暑补水、冬季流感预防 |

---

## 五、API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/rag/health-suggestion` | POST | 生成个性化生活建议 |

### Java 调用示例

```java
Map<String, Object> healthRecord = Map.of(
    "member_name", "张三",
    "gender", 1,
    "birth_date", "1960-03-15",
    "blood_type", "A",
    "allergy", "青霉素过敏，头孢类",
    "past_illness", "高血压5年, 2型糖尿病3年",
    "medication", "硝苯地平 30mg qd, 二甲双胍 500mg bid",
    "is_self", 1
);

Map<String, Object> consultation = Map.of(
    "symptom_text", "最近经常头晕，血压偏高",
    "doctor_advice", "建议低盐低脂饮食，规律服药，每周监测血压"
);

Map<String, Object> request = Map.of(
    "health_record", healthRecord,
    "consultation", consultation
);

Map result = restTemplate.postForObject(
    "http://localhost:8000/api/rag/health-suggestion", request, Map.class);

Map data = (Map) result.get("data");
List<Map> suggestions = (List<Map>) data.get("suggestions");

// 遍历写入 health_suggestion 表
for (Map category : suggestions) {
    String cat = (String) category.get("category");
    List<Map> items = (List<Map>) category.get("items");
    for (Map item : items) {
        healthSuggestionMapper.insert(
            recordId, patientId, cat,
            (String) item.get("title"),
            (String) item.get("content")
        );
    }
}
```

---

## 六、性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均延迟 | ~3-5s | LLM 生成 1500 tokens 的 JSON |
| Token 消耗 | ~1,700/次 | 5 类 × 2-3 条建议 |
| RAG 上下文 | ~1,200 字 | 检索 5 个相关疾病 |
| 输出结构 | 5 categories × 1-3 items | ~500 字总输出 |

---

## 七、对其他代码的修改

### 7.1 新建文件

| 文件 | 说明 |
|------|------|
| `src/health_suggestion/__init__.py` | 包入口，导出 `HealthSuggestionGenerator` |
| `src/health_suggestion/suggestion_generator.py` | 核心模块: `generate()` / `_format_*()` / `_retrieve_rag_context()` / `_parse_llm_response()` |
| `tests/health_suggestion/test_health_suggestion.py` | 14 个测试用例，覆盖完整/最小/空输入/儿科/心血管/RAG/容错/性能 |
| `docs/HEALTH_SUGGESTION.md` | 本文档 |

### 7.2 修改 `src/ai_config_loader.py`

**位置**: `_DEFAULT_CONFIGS` 字典末尾

**新增场景**: `health_suggestion`

- 场景数: 7 → 8
- model: qwen-flash
- temperature: 0.4
- max_tokens: 1500
- system_prompt: 健康管理师角色，输出 5 类 JSON

### 7.3 修改 `src/api_models.py`

**新增 5 个 Pydantic 模型**:

| 模型 | 用途 |
|------|------|
| `ConsultationInput` | consultation 表 AI 相关字段输入 |
| `SuggestionItem` | 单条建议 (title + content) |
| `SuggestionCategory` | 按 category 分组 (category + items[]) |
| `SuggestionRequest` | 请求体 (health_record + consultation) |
| `SuggestionData` | 响应 data (suggestions[]) |

### 7.4 修改 `src/api_server.py`

**新增端点**: `POST /api/rag/health-suggestion`

- 端点总数: 14 → 15
- Tags: `Health — 健康档案`
- 参数: `request: dict` (兼容 health_record + consultation)

### 7.5 未修改的文件

以下文件**不需要修改**（完全复用现有组件）:

| 组件 | 复用方式 |
|------|---------|
| `DeepSeekClient` | 通过 `llm_client` 属性延迟加载 |
| `VectorStore` | 通过 `vector_store` 属性延迟加载，提供 RAG 检索 |
| `ai_config_loader` | 读取 scene=`health_suggestion` 的 Prompt 和参数 |
| `query_engine.py` | 复用 `search_disease()` 方法 |
| `requirements.txt` | 无新依赖 |

---

## 八、设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 输出格式 | JSON 分组 (方案 A) | 用户指定；Java 按 category 遍历写入表 |
| RAG 无匹配 | LLM 直接生成 | 用户指定；不阻塞生成流程 |
| 输入模型 | dict（非强类型 Pydantic） | 与 health_summary 一致，保持简洁 |
| LLM 角色 | 健康管理师 (营养+运动+用药) | 面向患者，语言通俗 |
| JSON 容错 | 3 层 try-parse + fallback | 保证 API 永不因 LLM 格式异常返回 500 |
| temperature | 0.4 | 稍高于 health_summary(0.3)，保证建议多样性 |
| max_tokens | 1500 | 5 类 × 2-3 条 × ~100 字/条 |
| 文件位置 | `src/health_suggestion/` | 与 health_summary 平行，独立子包 |
| Prompt 管理 | MySQL `ai_model_config` | 统一管理，管理员可在线调优 |
| 数据表位置 | `medical_rag` 库 `health_suggestion` 表 | 该模块由 Python AI 引擎管理，Java 后端不参与表修改 |
| 测试位置 | `tests/health_suggestion/` | 对应模块结构 |
| 季节性策略 | Prompt 内指定夏季 | 简化实现，后续可改为参数传入当前月份 |

---

## 九、MySQL 数据库变更

### 9.1 `medical_rag` 库新增表

```sql
CREATE TABLE IF NOT EXISTS health_suggestion (
    suggestion_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    record_id     BIGINT      NOT NULL  COMMENT '关联健康档案ID',
    patient_id    BIGINT      NOT NULL  COMMENT '患者ID (冗余)',
    category      VARCHAR(20) NOT NULL  COMMENT 'diet/exercise/sleep/medication/seasonal',
    title         VARCHAR(100)NOT NULL  COMMENT '建议标题',
    content       TEXT        NOT NULL  COMMENT '建议正文',
    is_active     TINYINT     NOT NULL DEFAULT 1,
    generated_at  DATETIME    NOT NULL  COMMENT 'AI生成时间',
    expires_at    DATETIME    DEFAULT NULL,

    INDEX idx_record_id  (record_id),
    INDEX idx_patient_id (patient_id),
    INDEX idx_category   (category),
    INDEX idx_active     (is_active, generated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 9.2 `ai_model_config` 表新增场景

已通过 `seed_from_defaults()` 写入 `health_suggestion` 场景（8 个场景）:

| 字段 | 值 |
|------|-----|
| scene | `health_suggestion` |
| model_name | `qwen-flash` |
| temperature | `0.4` |
| max_tokens | `1500` |
| system_prompt | 健康管理师角色 Prompt（可从 Java 管理端在线修改） |
