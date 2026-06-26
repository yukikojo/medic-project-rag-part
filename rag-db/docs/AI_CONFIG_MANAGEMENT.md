# AI 模型配置管理 — MySQL 动态 Prompt 与参数

> **版本**: v2.0  
> **日期**: 2026-06-25  
> **模块**: `ai_config_loader.py`, `deepseek_client.py`, `emr_extractor.py`

---

## 一、解决的问题

项目的 System Prompt、temperature、max_tokens 等 AI 参数**硬编码在 Python 源码中**，每次调整都需要改代码重启服务。现在将这些配置存入 MySQL `ai_model_config` 表，Java 管理员通过管理后台直接修改，Python **实时生效**（60 秒缓存）。

```
之前: 修改 Prompt → 改 Python 代码 → 重启服务
现在: 修改 Prompt → UPDATE MySQL → POST /api/rag/config/refresh → 即时生效
```

---

## 二、表结构

```sql
CREATE TABLE ai_model_config (
    config_id    BIGINT PRIMARY KEY AUTO_INCREMENT,
    scene        VARCHAR(30)  NOT NULL UNIQUE,  -- 业务场景
    model_name   VARCHAR(50)  NOT NULL,          -- 模型名称
    api_base_url VARCHAR(200) NOT NULL,          -- API 地址
    api_key      VARCHAR(200),                   -- API Key
    temperature  FLOAT        DEFAULT 0.3,       -- 0-1
    max_tokens   INT          DEFAULT 1000,
    top_p        FLOAT        DEFAULT 0.9,
    system_prompt TEXT        NOT NULL,          -- ★ Java 可直接修改
    status       TINYINT      DEFAULT 1,
    updated_at   DATETIME     ON UPDATE CURRENT_TIMESTAMP
);
```

---

## 三、6 个业务场景

| scene | 用途 | Python 调用位置 | 默认模型 |
|-------|------|----------------|---------|
| `triage` | 智能导诊科室推荐 | `DeepSeekClient.recommend_department()` | qwen-flash |
| `symptom_extract` | 症状结构化提取 | `DeepSeekClient.extract_symptoms()` | qwen-flash |
| `query_optimize` | 查询优化标准化 | `QueryOptimizer._optimize_with_llm()` | deepseek-v4-flash |
| `emr_extract` | 病历要素提取 | `EMRProcessor.extract_medical_record()` | qwen-flash |
| `assist` | AI 辅助问诊提示 | `EMRProcessor.generate_assist_info()` | qwen-flash |
| `chat` | 通用对话 | `DeepSeekClient.chat()` | qwen-flash |

---

## 四、API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/rag/config/list` | GET | 列出全部场景配置（Prompt 截断，Key 脱敏） |
| `/api/rag/config/{scene}` | GET | 查看某场景完整 Prompt |
| `/api/rag/config/refresh` | POST | 强制刷新缓存（从 MySQL 重新加载） |
| `/api/rag/config/seed` | POST | 将默认配置写入 MySQL（首次初始化） |

---

## 五、Java 端使用

### 管理员修改 Prompt 后生效

```java
// Step 1: Java 直接 UPDATE MySQL（或通过管理后台）
jdbcTemplate.update(
    "UPDATE ai_model_config SET system_prompt = ?, temperature = ?, updated_at = NOW() WHERE scene = 'triage'",
    newPrompt, 0.5
);

// Step 2: 通知 Python 刷新缓存
restTemplate.postForObject("http://localhost:8000/api/rag/config/refresh", null, Map.class);

// Step 3: 下次 LLM 调用自动使用新 Prompt，无需重启
```

### 查看当前配置

```java
Map result = restTemplate.getForObject(
    "http://localhost:8000/api/rag/config/triage", Map.class);
String currentPrompt = (String) ((Map)result.get("data")).get("system_prompt");
```

---

## 六、架构设计

```
┌────────────────────┐     UPDATE        ┌──────────────────┐
│  Java Admin UI     │ ────────────────> │  MySQL            │
│  (管理后台)        │                   │  ai_model_config  │
└────────────────────┘                   └────────┬─────────┘
                                                  │ SELECT
                    ┌─────────────────────────────▼──────────┐
                    │  ai_config_loader.py                   │
                    │                                        │
                    │  ┌──────────────────────────────────┐  │
                    │  │  In-memory cache (TTL 60s)        │  │
                    │  │  scene → {prompt, temp, tokens}   │  │
                    │  └──────────────┬───────────────────┘  │
                    │                 │                       │
                    │  ┌──────────────▼───────────────────┐  │
                    │  │  Fallback: hardcoded defaults    │  │
                    │  │  (MySQL 不可用时自动降级)         │  │
                    │  └──────────────────────────────────┘  │
                    └──────────────┬─────────────────────────┘
                                   │ get_prompt("triage")
                    ┌──────────────▼─────────────────────────┐
                    │  deepseek_client / emr_extractor       │
                    │  (LLM 调用时自动读最新配置)             │
                    └────────────────────────────────────────┘
```

**容错**: MySQL 不可用时自动降级到 Python 硬编码默认值，不影响服务可用性。

---

## 七、初始化

MySQL 启动后运行一次：

```bash
python rag-db/src/ai_config_loader.py
# → 建表 + 将 6 个场景的默认 Prompt 写入 MySQL
```

或通过 API：

```bash
curl -X POST http://localhost:8000/api/rag/config/seed
```
