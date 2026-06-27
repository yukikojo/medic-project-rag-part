# 多轮对话 Agent — Multi-Turn Medical Dialogue Agent

> **版本**: v1.0
> **日期**: 2026-06-27
> **模块**: `src/dialogue/`, `ai_config_loader.py`, `api_server.py`, `api_models.py`

---

## 一、概述

将现有的单轮"症状→科室"导诊升级为**Agent-Skill 架构的多轮对话系统**。DialogueManager 作为 Agent 编排对话循环，通过鉴别诊断追问收集关键症状，在信息充足时给出疾病推荐。

**核心变化**：

| | 现有单轮模式 | 新多轮 Agent 模式 |
|---|---|---|
| 交互方式 | 一次请求-响应即结束 | 多轮追问直到信息充足 |
| 决策者 | 无决策，直接返回 Top-1 | Agent 自主判断：追问 or 推荐 |
| 症状收集 | 用户一次性输入 | 每轮提取+累积+去重 |
| 状态管理 | 无状态 | MySQL `dialogue_session` 表持久化 |
| 流程控制 | Java 传参→Python 计算→返回 | Python Agent 全权管理对话循环 |

**Agent-Skill 架构**：

```
┌──────────────────────────────────────────────────────────┐
│                    DialogueManager (Agent)                │
│                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │ Symptom      │  │ RAG          │  │ LLM          │  │
│   │ Extraction   │  │ Retrieval    │  │ Decision     │  │
│   │ (LLM Skill)  │  │ (VectorStore)│  │ (DeepSeek)   │  │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│          │                 │                 │           │
│   ┌──────┴─────────────────┴─────────────────┴───────┐   │
│   │              Decision Logic                       │   │
│   │  ┌─ 症状数 < 2? ──→ 强制追问                     │   │
│   │  ├─ 轮数 ≥ 最大? ──→ 强制推荐                     │   │
│   │  └─ 否则 ──→ LLM 自主判断                         │   │
│   └──────────────────────┬───────────────────────────┘   │
│                          │                               │
│   ┌──────────────────────┴───────────────────────────┐   │
│   │              Output Decision                       │   │
│   │  ┌─ ask ─────→ 鉴别诊断追问问题                    │   │
│   │  ├─ recommend → 科室/疾病/置信度推荐               │   │
│   │  └─ emergency → 紧急就医警告                       │   │
│   └──────────────────────────────────────────────────┘   │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │         MySQL dialogue_session (State)            │   │
│   └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**对话流程示意**（以头痛为例）：

```
Turn 1  患者: "我头痛三天了"
         Agent: "头痛时有没有伴随视力模糊或恶心？"  ← 鉴别偏头痛 vs 紧张性头痛
         
Turn 2  患者: "有，有时候恶心，光线强的时候更痛"
         Agent: "头痛的具体位置在哪里？是一侧还是整个头部？"  ← 进一步缩小范围

Turn 3  患者: "右边太阳穴附近，跳着疼"
         Agent → 置信度达标 → 返回推荐:
           ✓ 偏头痛 (confidence 82%)
           → 神经内科
           → 建议: 避免强光刺激、规律作息、记录头痛日记
```

---

## 二、三种 Agent 动作

| action | 触发条件 | Agent 行为 | 返回内容 |
|--------|----------|------------|----------|
| `ask` | 症状不足 (<2个) 或 LLM 判断需继续 | 生成一个鉴别诊断追问 | `question`, `question_reasoning`, `candidate_diseases`, `collected_info` |
| `recommend` | LLM 判断信息充足 OR 达到最大轮数 | 生成最终科室/疾病推荐 | `recommendation`, `candidate_diseases`, `collected_info` |
| `emergency` | 输入匹配到急诊关键词 | 立即返回紧急警告 | `emergency_warning`, `candidate_diseases` (如有) |

**紧急关键词列表**（快速规则匹配，不调 LLM）：

```
剧烈疼痛、昏迷、大出血、意识不清、呼吸困难、
胸痛、窒息、抽搐、休克、无呼吸、无心跳、
严重外伤、大咯血、呕血、便血
```

---

## 三、决策逻辑详解（B+C 组合）

### 决策流程

```
process(patient_input)
    │
    ├─ ① 提取症状 → extracted = {symptoms, body_parts, duration, severity}
    │
    ├─ ② 累积合并 → accumulated = merge(session.symptoms, extracted)
    │
    ├─ ③ RAG 检索 → candidates = search_disease(accumulated)
    │
    └─ ④ 判断信息是否充足 ── _decide_sufficient_info()
            │
            ├─ 紧急关键词? → action = "emergency" ✋
            │
            ├─ 症状数 < 2 AND 轮数 < max? → action = "ask" (门槛保护)
            │
            ├─ 轮数 ≥ max? → action = "recommend" (上限保护)
            │
            └─ 其他 → LLM 判断 (鉴别诊断)
                   │
                   ├─ LLM 返回 "continue" → action = "ask"
                   └─ LLM 返回 "recommend" → action = "recommend"
```

### 规则 B：症状数量门槛

```python
MIN_SYMPTOM_THRESHOLD = 2

if len(accumulated_symptoms["symptoms"]) < MIN_SYMPTOM_THRESHOLD:
    # 强制追问，不调用决策 LLM
    return {"decision": "continue", "confidence": 30}
```

### 规则 C：LLM 自主判断

当症状数 ≥2 且未达最大轮数时，调用 `dialogue_decision` 场景的 LLM 判断：

- **输入**：已收集症状 + Top-5 候选疾病 + 当前轮次
- **判断维度**：
  1. 症状数量是否足够（≥2）
  2. 某个候选疾病置信度是否明显高于其他（分数差异 >0.1）
  3. 再追问新问题的边际收益是否太低
- **输出**：`{"decision": "continue"|"recommend", "confidence": 0-100}`

---

## 四、鉴别诊断追问策略（策略 C）

### 设计原理

不是随机追问，而是**针对 Top-3 候选疾病的区别性特征**生成问题。

```
RAG 检索 → Top-3 候选:
  1. 偏头痛      (74.6%) — 特征: 单侧搏动性痛, 畏光, 恶心
  2. 紧张性头痛  (68.2%) — 特征: 双侧压迫性痛, 颈肩僵硬
  3. 颈椎病      (65.1%) — 特征: 颈后痛, 手臂麻木

鉴别分析 → LLM 识别区分点:
  偏头痛 vs 紧张性头痛: 单侧 vs 双侧, 是否畏光恶心
  
生成追问 → "头痛是一侧疼还是整个头部都疼？有没有畏光或恶心？"
```

### LLM Prompt 结构（`dialogue_followup` 场景）

```
角色: 经验丰富的临床医生，擅长鉴别诊断

输入:
  - 已收集症状: "头痛(3天), 搏动性痛, 右侧太阳穴"
  - 候选疾病 (Top-3, 含特征症状):
      1. 偏头痛 (score: 0.746) — 单侧搏动性头痛, 畏光, 恶心呕吐, 视觉先兆
      2. 紧张性头痛 (score: 0.682) — 双侧压迫性/紧箍感, 颈肩僵硬
      3. 颈椎病 (score: 0.651) — 颈后痛, 可放射至头部, 手臂麻木

要求:
  - 生成一个最关键的鉴别诊断问题
  - 该问题应能最好地区分 Top-3 候选疾病
  - 使用通俗易懂的语言（患者面向）
  - 简短、具体，一次只问一件事

输出: {"question": "...", "reasoning": "...", "target_diseases": [...]}
```

---

## 五、MySQL 数据模型

### dialogue_session 表

```sql
CREATE TABLE IF NOT EXISTS dialogue_session (
    session_id           VARCHAR(36)  PRIMARY KEY COMMENT 'UUID v4',
    patient_id           BIGINT       DEFAULT NULL COMMENT '患者ID (未登录可空)',
    status               VARCHAR(20)  NOT NULL DEFAULT 'active'
                         COMMENT '会话状态: active/closed/emergency/timeout',
    
    -- 症状累积 (每轮 LLM 提取后合并)
    collected_symptoms   TEXT         COMMENT 'JSON: [{"symptom":"头痛","duration":"3天",
                                          "severity":"中度","body_part":"右侧太阳穴"}]',
    extracted_keywords   TEXT         COMMENT 'JSON: ["头痛","恶心","畏光"]',
    
    -- RAG 检索结果 (每轮更新)
    candidate_diseases   TEXT         COMMENT 'JSON: [{"disease":"偏头痛","score":0.746,
                                          "departments":"神经内科","symptoms":"..."}]',
    
    -- 完整对话历史
    dialogue_history     TEXT         COMMENT 'JSON: [{"turn":1,"role":"patient",
                                          "content":"头痛三天","timestamp":"..."}]',
    
    -- 最终推荐 (状态变为 closed 时写入)
    final_recommendation TEXT         COMMENT 'JSON: {"department":"神经内科","disease":"偏头痛",
                                          "confidence":0.82,"reasoning":"...","suggestion":"..."}',
    
    max_turns            INT          DEFAULT 8 COMMENT '最大对话轮数',
    current_turn         INT          DEFAULT 0 COMMENT '当前轮次',
    created_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_patient_id (patient_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='多轮对话会话状态表';
```

### JSON 字段使用说明

所有 JSON 字段存储为 TEXT 类型，Python 端使用 `json.dumps()` 写入、`json.loads()` 读取，与项目现有惯例一致（避免 MySQL JSON 类型依赖）。

### 会话生命周期

```
active ──→ closed      (正常结束：LLM 给出推荐)
active ──→ emergency   (紧急结束：检测到急症关键词)
active ──→ timeout     (超时关闭：前端或 Java 主动关闭)
```

---

## 六、API 端点

### 端点总览

新增 4 个端点，系统总端点数: 20 → 24。

| # | Method | Path | 说明 | 延迟 |
|---|--------|------|------|------|
| 21 | POST | `/api/rag/dialogue/start` | 开始新对话会话 | ~2-4s |
| 22 | POST | `/api/rag/dialogue/continue` | 继续对话（患者回答） | ~2-4s |
| 23 | GET | `/api/rag/dialogue/{session_id}` | 获取会话完整状态 | <20ms |
| 24 | POST | `/api/rag/dialogue/{session_id}/close` | 手动关闭会话 | <20ms |

---

### 21. POST `/api/rag/dialogue/start` — 开始新对话

创建一个新的对话会话，可选地传入初始症状直接开始第一轮。

**请求**：

```json
{
    "patient_id": 1001,
    "initial_symptom": "头痛三天了，右边太阳穴跳着疼",
    "max_turns": 8
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `patient_id` | int | ❌ | null | 患者ID |
| `initial_symptom` | string | ❌ | null | 初始症状描述（传入则自动执行第一轮） |
| `max_turns` | int | ❌ | 8 | 最大对话轮数 (3-20) |

**响应（有初始症状 → 自动执行第一轮）**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "action": "ask",
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "current_turn": 1,
        "question": "头痛时有没有伴随视力模糊或恶心呕吐？",
        "question_reasoning": "偏头痛常伴畏光恶心，紧张性头痛通常无此症状，该问题可有效区分两者",
        "candidate_diseases": [
            {"disease": "偏头痛", "score": 0.746, "departments": "神经内科"},
            {"disease": "紧张性头痛", "score": 0.682, "departments": "神经内科"},
            {"disease": "颈椎病", "score": 0.651, "departments": "骨科, 康复科"}
        ],
        "collected_info": {
            "symptoms": ["头痛"],
            "body_parts": ["右侧太阳穴"],
            "duration": "3天",
            "severity": "未明确"
        },
        "confidence": 0.30
    },
    "metadata": {
        "agent": "dialogue_manager",
        "session_status": "active"
    }
}
```

**响应（无初始症状 → 返回通用引导）**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "action": "ask",
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "current_turn": 0,
        "question": "您好，请问您有什么不舒服的症状？请尽量详细描述，比如哪里不舒服、持续多久了、什么情况下会加重。",
        "question_reasoning": null,
        "candidate_diseases": null,
        "collected_info": null,
        "confidence": 0.0
    },
    "metadata": {
        "agent": "dialogue_manager",
        "session_status": "active"
    }
}
```

---

### 22. POST `/api/rag/dialogue/continue` — 继续对话

患者回答后，Agent 处理本轮输入并返回下一步动作。

**请求**：

```json
{
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "patient_input": "有时候会恶心，看到亮光的时候更疼"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | ✅ | 会话ID (36字符UUID) |
| `patient_input` | string | ✅ | 患者本轮回答 (1-2000字) |

**响应（继续追问）**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "action": "ask",
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "current_turn": 2,
        "question": "头痛的频率大概是怎样的？是每天都发作还是间歇性的？发作前有没有什么预兆？",
        "question_reasoning": "偏头痛常有前驱症状(视觉先兆/情绪变化)，发作频率和诱因有助于确诊",
        "candidate_diseases": [...],
        "collected_info": {
            "symptoms": ["头痛", "恶心", "畏光"],
            "body_parts": ["右侧太阳穴", "头部"],
            "duration": "3天",
            "severity": "中度"
        },
        "confidence": 0.45
    },
    "metadata": {
        "agent": "dialogue_manager",
        "session_status": "active"
    }
}
```

**响应（达到推荐条件）**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "action": "recommend",
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "current_turn": 4,
        "question": null,
        "recommendation": {
            "department": "神经内科",
            "disease": "偏头痛（无先兆型）",
            "confidence": 0.82,
            "reasoning": "患者表现为单侧搏动性头痛，伴恶心畏光，持续3天，符合偏头痛诊断标准。建议神经内科就诊，可考虑预防性治疗。",
            "suggestion": "建议记录头痛日记（发作时间、诱因、持续时间），避免强光和噪音刺激，规律作息。如每月发作>4次，可咨询医生预防性用药。",
            "alternative_departments": ["疼痛科", "中医科"]
        },
        "candidate_diseases": [...],
        "collected_info": {...},
        "confidence": 0.82
    },
    "metadata": {
        "agent": "dialogue_manager",
        "session_status": "closed"
    }
}
```

**响应（紧急情况）**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "action": "emergency",
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "current_turn": 1,
        "emergency_warning": "检测到紧急症状描述：剧烈胸痛、呼吸困难。请立即拨打120或前往最近的急诊科！",
        "candidate_diseases": [
            {"disease": "急性心肌梗死", "score": 0.891, "departments": "急诊科, 心内科"},
            {"disease": "主动脉夹层", "score": 0.723, "departments": "急诊科, 心外科"}
        ],
        "collected_info": {...},
        "confidence": 0.95
    },
    "metadata": {
        "agent": "dialogue_manager",
        "session_status": "emergency"
    }
}
```

---

### 23. GET `/api/rag/dialogue/{session_id}` — 获取会话状态

**响应**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "patient_id": 1001,
        "status": "active",
        "current_turn": 2,
        "max_turns": 8,
        "collected_symptoms": [
            {"symptom": "头痛", "duration": "3天", "severity": "中度", "body_part": "右侧太阳穴"},
            {"symptom": "恶心", "severity": "轻度"},
            {"symptom": "畏光", "severity": "中度"}
        ],
        "extracted_keywords": ["头痛", "恶心", "畏光", "搏动性"],
        "candidate_diseases": [
            {"disease": "偏头痛", "score": 0.746, "departments": "神经内科"},
            {"disease": "紧张性头痛", "score": 0.682, "departments": "神经内科"},
            {"disease": "颈椎病", "score": 0.651, "departments": "骨科, 康复科"}
        ],
        "dialogue_history": [
            {"turn": 1, "role": "patient", "content": "头痛三天了，右边太阳穴跳着疼", "timestamp": "2026-06-27T10:00:01"},
            {"turn": 1, "role": "agent", "content": "头痛时有没有伴随视力模糊或恶心呕吐？", "timestamp": "2026-06-27T10:00:04"},
            {"turn": 2, "role": "patient", "content": "有时候会恶心，看到亮光的时候更疼", "timestamp": "2026-06-27T10:00:30"},
            {"turn": 2, "role": "agent", "content": "头痛的频率大概是怎样的？...", "timestamp": "2026-06-27T10:00:33"}
        ],
        "final_recommendation": null,
        "created_at": "2026-06-27T10:00:01",
        "updated_at": "2026-06-27T10:00:33"
    }
}
```

---

### 24. POST `/api/rag/dialogue/{session_id}/close` — 手动关闭

**请求**：无 body

**响应**：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "status": "closed"
    }
}
```

---

## 七、Java 后端对接

### 完整 5 轮对话示例

```java
// 1. 开始会话
Map<String, Object> startReq = Map.of(
    "patient_id", 1001,
    "initial_symptom", "头痛三天了，右边太阳穴跳着疼"
);
Map startResp = restTemplate.postForObject(
    "http://localhost:8000/api/rag/dialogue/start", startReq, Map.class);

Map data = (Map) startResp.get("data");
String sessionId = (String) data.get("session_id");
String question = (String) data.get("question");
// → question: "头痛时有没有伴随视力模糊或恶心呕吐？"

// 2. 患者回答后继续
Map<String, Object> continueReq = Map.of(
    "session_id", sessionId,
    "patient_input", "有时候会恶心，看到亮光的时候更疼"
);
Map continueResp = restTemplate.postForObject(
    "http://localhost:8000/api/rag/dialogue/continue", continueReq, Map.class);

data = (Map) continueResp.get("data");
String action = (String) data.get("action");

// 3. 循环直到 action != "ask"
while ("ask".equals(action)) {
    String nextQuestion = (String) data.get("question");
    // ... 等待用户输入 ...
    continueReq = Map.of("session_id", sessionId,
                         "patient_input", userAnswer);
    continueResp = restTemplate.postForObject(
        "http://localhost:8000/api/rag/dialogue/continue", continueReq, Map.class);
    data = (Map) continueResp.get("data");
    action = (String) data.get("action");
}

// 4. 获取最终推荐
if ("recommend".equals(action)) {
    Map recommendation = (Map) data.get("recommendation");
    String dept = (String) recommendation.get("department");
    String disease = (String) recommendation.get("disease");
    Double confidence = (Double) recommendation.get("confidence");
    // → 神经内科, 偏头痛, 82%
}

// 5. 获取完整对话历史（可选）
Map sessionState = restTemplate.getForObject(
    "http://localhost:8000/api/rag/dialogue/" + sessionId, Map.class);

// 6. 手动关闭（可选，recommend 已自动关闭）
restTemplate.postForObject(
    "http://localhost:8000/api/rag/dialogue/" + sessionId + "/close",
    null, Map.class);
```

---

## 八、错误处理和降级策略

| 场景 | 处理方式 |
|------|----------|
| LLM API 调用失败 | 返回结构化的 fallback 响应，使用规则引擎代替 LLM 决策 |
| JSON 解析失败 | 4 层 fallback：直接解析 → ```json提取 → {…}提取 → 安全默认值 |
| MySQL 不可用 | 内存中维护会话状态（不持久化），记录 warning 日志 |
| 会话不存在 | 返回 `code: 404, "Session not found"` |
| 会话已关闭 | 返回 `code: 400, "Session is closed, please start a new session"` |
| 空输入 | 返回 `code: 400, "patient_input cannot be empty"` |
| 达到最大轮数 | 强制生成推荐（即使置信度不高），标注"基于有限信息" |
| RAG 检索无结果 | 仍然继续 LLM 追问（基于通用医学知识），不阻塞对话 |

### JSON 解析 4 层 Fallback

```python
def _parse_llm_json(raw_text):
    # Layer 1: direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    
    # Layer 2: extract from ```json ... ``` block
    match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Layer 3: extract outermost {...}
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start:end+1])
        except json.JSONDecodeError:
            pass
    
    # Layer 4: safe default
    return {"decision": "continue", "confidence": 30,
            "reasoning": "JSON parse failed, using default"}
```

---

## 九、AI 配置场景

新增 2 个 scene，总计 9 → 11（含已有的 9 个）：

| Scene | 用途 | temperature | max_tokens |
|-------|------|-------------|------------|
| `dialogue_followup` | 生成鉴别诊断追问 | 0.7 | 1024 |
| `dialogue_decision` | 判断信息是否充足 | 0.3 | 800 |

这两个 scene 与现有 9 个 scene 一样，存储在 MySQL `ai_model_config` 表中，支持管理员在线调优。`ai_config_loader` 提供 60s TTL 缓存 + 自动降级到 `_DEFAULT_CONFIGS`。

---

## 十、文件变更清单

### 新建文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/dialogue/__init__.py` | ~3 | 包初始化 |
| `src/dialogue/dialogue_manager.py` | ~400 | 核心 Agent 逻辑 |
| `tests/dialogue/test_dialogue.py` | ~300 | 20 个测试用例 |
| `docs/DIALOGUE_AGENT.md` | 本文档 | 功能设计文档 |

### 修改文件

| 文件 | 变更内容 | 行数变化 |
|------|----------|----------|
| `src/ai_config_loader.py` | 替换 `dialogue_followup` + 新增 `dialogue_decision` | +80 |
| `src/api_models.py` | 新增 4 个 Pydantic 模型 | +45 |
| `src/api_server.py` | 新增 4 个端点 + lazy-load singleton | +80 |

### 不变更文件（直接复用）

| 模块 | 复用方式 |
|------|----------|
| `retrieval/query_engine.py` | `VectorStore().search_disease()` — RAG 检索 |
| `generation/deepseek_client.py` | `DeepSeekClient` — LLM 调用 |
| `retrieval/query_optimizer.py` | `QueryOptimizer` — 症状关键词标准化 |
| `reranker/reranker.py` | Reranker — Cross-Encoder 精排 |
| `ai_config_loader.py` | `get_prompt()`, `get_params()` — 配置管理 |
| `enrichment/kg_enricher.py` | KG Enricher — 知识图谱富化 |

---

## 十一、设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| Agent 架构 | 单 `DialogueManager` 类 | 完全复用 `HealthSuggestionGenerator` 模式 (lazy-load, config loader, JSON fallback) |
| 对话循环 | Turn-based HTTP 轮询 | 无状态 FastAPI 架构，不需要 WebSocket 和长连接 |
| 会话 ID | UUID v4 (36 chars) | 全局唯一，无碰撞风险，不需中心化 ID 生成 |
| 最大轮数 | 8 (可配置) | 防止无限循环，平衡信息收集与用户体验 |
| 紧急检测 | 关键词优先 + LLM 补充 | 快速规则匹配短路（<1ms），省一次 LLM API 调用 |
| RAG 集成 | 复用 `VectorStore`，不重复 | `search_disease()` + `comprehensive_search()` 已满足需求 |
| MySQL JSON 存储 | TEXT 列 + `json.dumps` | 项目惯例，避免 MySQL 5.7 JSON 类型依赖 |
| 症状累积 | 每轮 LLM 提取 + 按症状名去重合并 | 确保信息不丢失、不重复 |
| 决策逻辑 | B+C 组合（症状数门槛 + LLM 判断） | 用户指定：规则兜底保证不早推，LLM 灵活判断"何时足够" |
| 追问策略 | 鉴别诊断驱动 (C) | 用户指定：针对 Top-3 候选疾病的区分性特征提问，提高信息收集效率 |
| Prompt 管理 | MySQL `ai_model_config` (10 scenes) | 统一管理，管理员可在线调优，60s TTL 缓存 |
| Decision LLM 合并 | 单次 LLM 调用判断 continue/recommend | 比"先 LLM 提问 + 再次 LLM 判断"减少一次 API 调用 |
| 测试模式 | 独立脚本 (PASS/FAIL 计数器) | 与项目 10 个测试文件一致，无 pytest 依赖 |
| 文件位置 | `src/dialogue/` | 与 `health_summary/`、`health_suggestion/` 平行，独立子包 |
| 流程控制 | Python 全权管理（方案 A） | 用户指定：Java 只做透传，减少 Java 端复杂度 |

---

## 十二、性能预估

| 操作 | 预估延迟 | 瓶颈 |
|------|----------|------|
| `start` (无初始症状) | <50ms | MySQL INSERT |
| `start` (有初始症状) | ~2-4s | LLM × 2 (symptom_extract + dialogue_decision/followup) |
| `continue` (ask) | ~2-4s | LLM × 3 (extract + decision + followup) |
| `continue` (recommend) | ~3-5s | LLM × 3 (extract + decision + triage recommendation) |
| `continue` (emergency) | <100ms | 关键词匹配 + MySQL UPDATE |
| `GET session` | <20ms | MySQL SELECT |
| `close` | <20ms | MySQL UPDATE |

**优化方向**（后续迭代）：
1. 合并 `symptom_extract` + `dialogue_decision` 为单次 LLM 调用（减少 1 次 API 调用）
2. 症状提取使用小模型（如 deepseek-chat，0.1 temperature）
3. Redis 缓存活跃会话状态（减少 MySQL 读写）
4. 首次 RAG 结果缓存（同一会话内不重复检索相同查询）
