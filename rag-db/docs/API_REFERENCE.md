# RAG AI 引擎 — API 接口文档

> **服务**: RAG 医疗智能导诊 AI 引擎 (FastAPI)  
> **版本**: v2.1  
> **Base URL**: `http://{host}:8000`  
> **日期**: 2026-06-30

---

## 一、概览

Python AI 引擎通过 FastAPI 对外暴露 **25 个 REST 端点**，是 Java Spring Boot 调用所有 AI 功能的唯一入口。端点按 7 个功能组划分：

| 分组 | 标签 | 端点数 | 说明 |
|------|------|--------|------|
| System | `System` | 2 | 健康检查、用户反馈 |
| Core | `Core — 智能导诊` | 4 | 智能导诊核心：科室推荐、症状分析、疾病检索、KG 增强 |
| EMR | `EMR — 病历提取` | 2 | 病历要素提取、AI 辅助问诊 |
| Dialogue | `Dialogue — 多轮对话` | 4 | 多轮对话 Agent：开始会话、继续追问、状态查询、关闭 |
| Health | `Health — 健康档案` | 2 | 健康档案摘要、个性化生活建议 |
| Advice | `Advice — 医嘱解读` | 1 | 医生建议 → 患者通俗语言 |
| Knowledge | `Knowledge — 知识库同步` | 4 | MySQL↔ChromaDB 同步、数据导入、状态查询 |
| Config | `Config — AI配置管理` | 5 | AI 模型配置 CRUD、缓存刷新、seed、Prompt 测试 |
| Reference | `Reference — 参考数据` | 2 | 科室列表、科室详情 |

统一响应格式：

```json
{
  "code": 200,
  "message": "success",
  "data": { ... }
}
```

---

## 二、System — 系统管理

### 2.1 GET /api/rag/health — 健康检查

Java 启动时探活，确认 AI 引擎就绪。

**请求**: 无

**响应**:
```json
{
  "code": 200,
  "status": "ok",
  "version": "2.0.0",
  "services": {
    "vector_store": { "status": "ok", "collections": { "disease_knowledge": "8808", ... }},
    "llm": { "status": "ok" },
    "emr_extractor": { "status": "ok" }
  },
  "uptime_seconds": 3600.5,
  "timestamp": "2026-06-27 12:00:00"
}
```

- `status`: `"ok"` (所有服务正常) 或 `"degraded"` (部分不可用)
- Java 启动时 `@PostConstruct` 调用此端点验证

---

### 2.2 POST /api/rag/feedback — 用户反馈收集

收集患者对导诊推荐结果的反馈，写入 `feedback_log.jsonl`。

**请求**:
```json
{
  "query": "头痛发热",
  "consult_id": 123,
  "recommended_department": "呼吸内科",
  "feedback": "negative",
  "actual_department": "神经内科",
  "comment": "实际是偏头痛"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | str | ✅ | 原始症状查询 |
| `feedback` | str | ✅ | `positive` / `negative` / `neutral` |
| `consult_id` | int | 否 | 关联问诊ID |
| `recommended_department` | str | 否 | AI 推荐的科室 |
| `actual_department` | str | 否 | 实际就诊科室 (negative 时填写) |
| `comment` | str | 否 | 附加说明 |

**响应**: `{"code": 200, "message": "反馈已记录", "data": {...}}`

---

## 三、Core — 智能导诊

### 3.1 POST /api/rag/search — 完整导诊推荐 ⭐核心

四阶段 Pipeline：查询优化 → 向量粗排 → Cross-Encoder 精排 → LLM 推理。支持口语化/方言症状描述。

**请求**:
```json
{
  "query": "头痛发热咳嗽流鼻涕",
  "top_k": 5
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `query` | str | ✅ | — | 症状描述，1-2000 字 |
| `top_k` | int | 否 | 5 | 返回疾病数，1-20 |

**响应**:
```json
{
  "code": 200,
  "data": {
    "query": "头痛发热咳嗽流鼻涕",
    "search_query": "头痛 发热 咳嗽 流鼻涕",
    "disease_results": [
      {
        "disease": "上呼吸道感染",
        "symptoms": "发热,头痛,咳嗽,鼻塞",
        "departments": "呼吸内科,耳鼻喉科",
        "category": "呼吸系统疾病",
        "drugs": "对乙酰氨基酚,布洛芬",
        "desc": "鼻腔、咽或喉部急性炎症的总称",
        "score": 0.85,
        "cosine_score": 0.72,
        "chain": "头痛发热咳嗽→上呼吸道感染→呼吸内科"
      }
    ],
    "symptom_direct": [
      { "symptom": "头痛", "departments": "神经内科,耳鼻喉科", "disease_count": 3, "score": 0.90 }
    ],
    "all_departments": ["呼吸内科", "耳鼻喉科"],
    "primary_recommendation": {
      "department": "呼吸内科",
      "disease": "上呼吸道感染",
      "confidence": 0.85,
      "reasoning": "患者症状符合典型上呼吸道感染..."
    },
    "reranked": true,
    "llm_department": "呼吸内科",
    "llm_disease": "上呼吸道感染",
    "llm_confidence": 85,
    "llm_reasoning": "头痛、发热、咳嗽、流鼻涕是典型的...",
    "llm_suggestion": "建议尽快到呼吸内科就诊...",
    "alternative_departments": ["耳鼻喉科"],
    "emergency_warning": false,
    "query_optimization": { "...": "..." }
  },
  "metadata": { "model": "qwen-flash", "usage": { "total_tokens": 580 } }
}
```

**Java 调用时机**: 患者输入症状、提交导诊请求时。

---

### 3.2 POST /api/rag/symptom/analyze — 症状结构化分析

仅做查询优化和症状提取，**不检索、不调 LLM 生成**（快速响应）。适合前端实时展示标准化症状供用户确认。

**请求**: `{"query": "肚子疼拉稀想吐没胃口"}`

**响应**:
```json
{
  "code": 200,
  "data": {
    "original_query": "肚子疼拉稀想吐没胃口",
    "optimized_query": "腹痛 腹泻 恶心 食欲减退",
    "symptoms": ["腹痛", "腹泻", "恶心", "食欲减退"],
    "body_parts": ["腹部", "消化系统"],
    "severity": "中",
    "has_emergency_signals": false,
    "normalization_note": "口语'肚子疼'→标准'腹痛', '拉稀'→'腹泻'",
    "llm_analysis": "..."
  }
}
```

- Rule 模式: ~1ms | LLM 模式: ~500ms

---

### 3.3 POST /api/rag/diseases/search — 纯疾病检索

仅向量检索，不调 LLM。适合快速查看可能疾病列表。

**请求**: `{"query": "头痛", "top_k": 5}`

**响应**: `{"code": 200, "data": {"query": "头痛", "diseases": [...], "count": 5}}`

- 延迟: <50ms（纯向量计算）

---

### > 3.4 POST /api/rag/search/enriched — 知识图谱增强导诊

> 在 `/api/rag/search` 基础上，额外为每个疾病补充：推荐药品、推荐食物/忌口食物、建议检查项目、可能的并发症、治疗方法，以及跨疾病聚合的 KG Summary。
>
> **当前状态**: 该端点因 docstring bug 未注册到路由，需修复后可用。

**请求**:
```json
{
  "query": "头痛发热咳嗽",
  "top_k": 5,
  "max_drugs": 5,
  "max_foods": 5
}
```

**响应**: 较 `/api/rag/search` 多出 `kg_enrichment` 和 `kg_summary` 字段。

---

## 四、EMR — 病历提取

### 4.1 POST /api/rag/emr/extract — 病历要素提取

从症状描述 + 健康档案中提取 8 个结构化病历字段，对应 `medical_record` 表。

**请求**:
```json
{
  "symptom_text": "近3天反复发热，最高39.2°C，咳嗽咳黄痰，右侧胸痛",
  "health_record": {
    "member_name": "张三",
    "gender": 1,
    "allergy": "青霉素过敏",
    "past_illness": "高血压5年"
  },
  "patient_info": { "patient_id": 1, "age": 45, "gender": "男" },
  "use_rag": true,
  "consult_id": 123
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `symptom_text` | str | ✅ | 患者症状描述，1-5000 字 |
| `health_record` | object | 否 | 健康档案数据 |
| `patient_info` | object | 否 | 患者基本信息 |
| `use_rag` | bool | 否 | 是否启用 RAG (默认 true) |
| `consult_id` | int | 否 | 关联问诊ID |

**响应**:
```json
{
  "code": 200,
  "data": {
    "chief_complaint": "发热伴咳嗽咳痰3天",
    "present_illness": "患者3天前无明显诱因出现发热...",
    "past_history": "高血压病史5年",
    "allergy_history": "青霉素过敏",
    "family_history": null,
    "medication_hist": null,
    "diagnosis": "社区获得性肺炎可能",
    "treatment": "建议行血常规+胸部CT..."
  },
  "metadata": { "model": "qwen-flash", "latency_ms": 2500, "token_usage": { "total_tokens": 1200 } }
}
```

**Java 调用时机**: 医生进入主诉详情页时自动生成病历草稿。

---

### 4.2 POST /api/rag/assist/info — AI 辅助问诊提示

为接诊医生提供 5 类临床决策支持：病情摘要、追问问题、鉴别诊断、检查建议、用药方向、转诊建议。

**请求**: 同 `/api/rag/emr/extract`（symptom_text + health_record + patient_info）

**响应**:
```json
{
  "code": 200,
  "data": {
    "disease_summary": "患者发热3天伴咳嗽咳黄痰...",
    "follow_up_questions": ["是否伴有呼吸困难？", "既往有无类似发作？"],
    "differential_diagnosis": ["社区获得性肺炎", "急性支气管炎", "肺结核"],
    "necessary_tests": ["血常规", "胸部CT", "C反应蛋白"],
    "medication_suggestions": ["抗生素方向(需结合病原学)", "退热对症"],
    "referral_depts": ["如确诊肺炎建议呼吸内科"]
  },
  "metadata": { "model": "qwen-flash", "latency_ms": 3000 }
}
```

---

## 五、Health — 健康档案

### 5.1 POST /api/rag/health-summary — 健康档案 AI 摘要

为 `health_record` 表数据生成面向**医生**的专业健康摘要（自然语言段落），存入 `health_record.ai_summary`。

**请求**: health_record 表字段（member_name / gender / birth_date / blood_type / allergy / past_illness / surgery_history / medication / is_self）

**响应**:
```json
{
  "code": 200,
  "data": {
    "ai_summary": "66岁男性，A型血。患有高血压病5年、2型糖尿病3年...",
    "rag_context_used": true
  },
  "metadata": { "model": "qwen-flash", "latency_ms": 2831 }
}
```

---

### 5.2 POST /api/rag/health-suggestion — 个性化生活建议

面向**患者**的 5 类结构化生活建议（饮食/运动/睡眠/用药/季节性），输入为 health_record + consultation 双表数据。

**请求**:
```json
{
  "health_record": {
    "member_name": "张三", "gender": 1, "birth_date": "1960-03-15",
    "blood_type": "A", "allergy": "青霉素过敏",
    "past_illness": "高血压5年, 2型糖尿病",
    "medication": "硝苯地平 30mg qd, 二甲双胍 500mg bid",
    "record_id": 1, "patient_id": 1
  },
  "consultation": {
    "symptom_text": "最近经常头晕，血压偏高",
    "doctor_advice": "建议低盐低脂饮食，规律服药",
    "ai_analysis": { "urgency": "普通", "possible_diseases": ["高血压"] },
    "consultation_dialog": "..."
  }
}
```

**响应**:
```json
{
  "code": 200,
  "data": {
    "suggestions": [
      {
        "category": "diet",
        "items": [
          { "title": "低盐饮食", "content": "建议每日食盐控制在5g以内..." },
          { "title": "控制碳水化合物", "content": "主食定量，优先选择低GI食物..." }
        ]
      },
      { "category": "exercise", "items": [{ "title": "...", "content": "..." }] },
      { "category": "sleep", "items": [...] },
      { "category": "medication", "items": [...] },
      { "category": "seasonal", "items": [...] }
    ],
    "mysql_saved": { "status": "ok", "inserted": 10 }
  },
  "metadata": { "model": "qwen-flash", "latency_ms": 3500 },
  "rag_context_used": true
}
```

- 若 `record_id` + `patient_id` 都存在，**自动持久化**到 `medical_rag.health_suggestion` 表。

---

## 六、Knowledge — 知识库同步

### 6.1 POST /api/rag/knowledge/rebuild — 全量重建 ChromaDB

从 MySQL 重新编码全部数据，重建 3 个 ChromaDB Collection。

**请求**: 空（可选 `{"force": true}`）

**响应**:
```json
{
  "code": 200,
  "message": "知识库全量重建完成",
  "data": { "disease_knowledge": 8808, "symptom_dept_direct": 4826, "department_info": 54 },
  "latency_ms": 45000
}
```

- 耗时: ~45s（BGE-M3 GPU 编码 8808 条）
- Java 调用时机: 管理员重建知识库 / 首次部署

---

### 6.2 POST /api/rag/knowledge/sync — 增量同步

Java 修改 MySQL 后调用，按 ID 增量更新 ChromaDB。

**请求**:
```json
{
  "updated_ids": [1, 2, 3],
  "deleted_ids": [4]
}
```

**响应**:
```json
{
  "code": 200,
  "message": "同步完成: 更新3条, 删除1条",
  "data": { "synced": 3, "deleted": 1, "errors": [] },
  "latency_ms": 150
}
```

---

### 6.3 GET /api/rag/knowledge/status — 知识库状态

MySQL vs ChromaDB 数据一致性检查。

**响应**:
```json
{
  "code": 200,
  "data": {
    "mysql": { "disease_knowledge": 8808, "symptom_dept_direct": 4826, "department_info": 54 },
    "chromadb": { "collections": { "disease_knowledge": {"count": 8808}, ... }, "db_path": "..." },
    "consistency": { "consistent": true, "mysql_count": 8808, "chromadb_count": 8808, "delta": 0 }
  }
}
```

---

### 6.4 POST /api/rag/knowledge/import-json — JSON 导入 MySQL

将 `medical.json` 导入 MySQL `rag_disease` 表（首次初始化）。

**请求**: `{"json_path": "D:/medic project/rag data/openkg data/medical.json"}` — 不传则用默认路径

**响应**: `{"code": 200, "message": "导入完成: 8808 条", "data": {"imported": 8808}}`

---

## 七、Config — AI 配置管理

### 7.1 POST /api/rag/config/seed — 写入默认配置

将硬编码默认配置写入 MySQL `ai_model_config` 表。

**响应**: `{"code": 200, "message": "已写入 10 个场景配置", "data": {"seeded": 10}}`

- INSERT ... ON DUPLICATE KEY UPDATE — **幂等操作**，可重复调用

---

### 7.2 GET /api/rag/config/list — 列出所有场景

**响应**:
```json
{
  "code": 200,
  "data": {
    "scenes": [
      { "scene": "triage", "model": "qwen-flash", "prompt_preview": "你是医疗分诊助手..." },
      { "scene": "summary", "model": "qwen-flash", "prompt_preview": "你是资深全科医师..." }
    ],
    "source": "mysql",
    "total": 10
  }
}
```

- Prompt 被截断（仅预览），API Key 脱敏（`sk-***masked***`）

---

### 7.3 GET /api/rag/config/{scene} — 查询单个场景

获取完整配置（含完整 Prompt）。

**路径参数**: `scene` = `triage` / `emr_extract` / `assist` / `summary` / `health_profile` / `dialogue_followup` / `dialogue_decision` / `query_optimize` / `symptom_extract` / `advice_interpret`

**响应**:
```json
{
  "code": 200,
  "data": {
    "scene": "triage",
    "model_name": "qwen-flash",
    "temperature": 0.3,
    "max_tokens": 800,
    "system_prompt": "你是医疗分诊助手...完整Prompt..."
  }
}
```

- 404: 场景不存在

---

### 7.4 POST /api/rag/config/refresh — 强制刷新缓存

从 MySQL 重新加载配置（缓存 TTL 默认 60s，此端点强制立即刷新）。

**响应**: `{"code": 200, "message": "配置缓存已刷新"}`

---

### 7.5 POST /api/rag/config/test — 测试 Prompt 效果

用指定场景的当前配置测试 LLM 调用，返回模型原始输出。对应 Java `AiModelConfigController.testConfig(scene, testInput)`。

**请求**:
```json
{
  "scene": "triage",
  "test_input": "头痛发热咳嗽"
}
```

**响应**:
```json
{
  "code": 200,
  "data": {
    "scene": "triage",
    "model": "qwen-flash",
    "raw_output": "{\"department\": \"呼吸内科\", ...}",
    "latency_ms": 1200,
    "tokens": { "prompt_tokens": 450, "completion_tokens": 120, "total_tokens": 570 }
  }
}
```

---

## 八、Dialogue — 多轮对话 Agent

> 多轮对话 Agent 采用 Agent-Skill 架构，DialogueManager 编排对话循环，通过鉴别诊断追问收集症状，信息充足后给出科室推荐。

### 8.1 POST /api/rag/dialogue/start — 开始新对话

创建会话并返回首轮引导或追问。对应 Java `ConsultationController.startDialogue()`。

**请求**:
```json
{
  "patient_id": 1001,
  "initial_symptom": "头痛三天了，右边太阳穴跳着疼",
  "max_turns": 8
}
```

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `patient_id` | int | 否 | — | 患者ID |
| `initial_symptom` | str | 否 | — | 初始症状（传入则自动执行首轮分析） |
| `max_turns` | int | 否 | 8 | 最大对话轮数 (3-20) |

**响应**（含初始症状 → 首轮追问）:
```json
{
  "code": 200,
  "data": {
    "action": "ask",
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "current_turn": 1,
    "question": "头痛时有没有伴随视力模糊或恶心？",
    "candidate_diseases": [
      { "disease": "偏头痛", "score": 0.746, "departments": "神经内科" }
    ],
    "collected_info": { "symptoms": ["头痛"], "duration": "3天" },
    "confidence": 0.30
  }
}
```

**响应**（无初始症状 → 引导语）:
```json
{
  "code": 200,
  "data": {
    "action": "ask",
    "session_id": "a1b2c3d4-...",
    "current_turn": 0,
    "question": "您好，请问您有什么不舒服的症状？...",
    "confidence": 0.0
  }
}
```

- `action`: `"ask"` (追问) / `"recommend"` (已出推荐) / `"emergency"` (紧急警告)
- `session_id`: UUID v4 (36 字符)，后续调用需要传入

---

### 8.2 POST /api/rag/dialogue/continue — 继续对话

提交患者回答，Agent 处理后返回下一步。对应 Java `ConsultationController.continueDialogue()`。

**请求**:
```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "patient_input": "有时候会恶心，看到亮光的时候更疼"
}
```

**响应**（继续追问）:
```json
{
  "code": 200,
  "data": {
    "action": "ask",
    "session_id": "a1b2c3d4-...",
    "current_turn": 2,
    "question": "头痛的频率大概是怎样的？",
    "candidate_diseases": [...],
    "collected_info": { "symptoms": ["头痛", "恶心", "畏光"] },
    "confidence": 0.45
  }
}
```

**响应**（信息充足 → 推荐）:
```json
{
  "code": 200,
  "data": {
    "action": "recommend",
    "session_id": "a1b2c3d4-...",
    "current_turn": 4,
    "recommendation": {
      "department": "神经内科",
      "disease": "偏头痛（无先兆型）",
      "confidence": 0.82,
      "reasoning": "单侧搏动性头痛，伴恶心畏光...",
      "suggestion": "建议记录头痛日记，避免强光刺激"
    },
    "confidence": 0.82
  }
}
```

**响应**（紧急）:
```json
{
  "code": 200,
  "data": {
    "action": "emergency",
    "emergency_warning": "⚠️ 检测到紧急症状：剧烈胸痛、呼吸困难。请立即就医！",
    "candidate_diseases": [{ "disease": "急性心肌梗死", "score": 0.89 }],
    "confidence": 0.95
  }
}
```

---

### 8.3 GET /api/rag/dialogue/{session_id} — 查询会话状态

**响应**: 完整会话数据，含 `dialogue_history`、`collected_symptoms`、`candidate_diseases`、`final_recommendation` 等。

- 404: 会话不存在

---

### 8.4 POST /api/rag/dialogue/{session_id}/close — 手动关闭会话

**响应**: `{"code": 200, "data": {"session_id": "...", "status": "closed"}}`

---

## 九、Advice — 医嘱解读

### 9.1 POST /api/rag/advice/interpret — 医嘱解读

将医生专业建议翻译为患者易懂语言。对应 Java `ConsultationServiceImpl.generateAdviceInterpretation()`。

**请求**:
```json
{
  "doctor_advice": "建议低盐低脂饮食，硝苯地平缓释片 30mg qd，每周监测血压",
  "patient_context": "患者张三，男，65岁，高血压5年"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `doctor_advice` | str | ✅ | 医生原始建议文本 |
| `patient_context` | str | 否 | 患者背景信息 |

**响应**:
```json
{
  "code": 200,
  "data": {
    "plain_explanation": "医生建议您...",
    "key_points": ["低盐低脂饮食", "按时服药", "每周测血压"],
    "medication_guide": "硝苯地平缓释片，每天一次...",
    "follow_up_advice": "请按医生要求定期复诊，如有不适及时就医"
  }
}
```

---

## 十、Reference — 参考数据

### 10.1 GET /api/rag/departments — 全部科室列表

**响应**:
```json
{
  "code": 200,
  "departments": [
    { "department": "呼吸内科", "disease_count": 15, "common_symptoms": "咳嗽,发热", "sample_diseases": "感冒,肺炎" }
  ],
  "total": 54
}
```

---

### 10.2 GET /api/rag/department/{name} — 科室详情

**路径参数**: `name` = `呼吸内科`（URL encode）

**响应**: `{"code": 200, "data": {"department": "呼吸内科", "disease_count": 15, ...}}`

---

## 十一、全部端点汇总

| # | Method | Path | Tags | 延迟 |
|---|--------|------|------|------|
| 1 | GET | `/api/rag/health` | System | <5ms |
| 2 | POST | `/api/rag/search` | Core | ~2-4s |
| 3 | POST | `/api/rag/symptom/analyze` | Core | ~1-500ms |
| 4 | POST | `/api/rag/search/enriched` | Core | ~2-4s |
| 5 | POST | `/api/rag/emr/extract` | EMR | ~2-3s |
| 6 | POST | `/api/rag/assist/info` | EMR | ~2-3s |
| 7 | POST | `/api/rag/diseases/search` | Core | <50ms |
| 8 | GET | `/api/rag/departments` | Reference | <10ms |
| 9 | GET | `/api/rag/department/{name}` | Reference | <10ms |
| 10 | POST | `/api/rag/knowledge/rebuild` | Knowledge | ~45s |
| 11 | POST | `/api/rag/knowledge/sync` | Knowledge | ~150ms |
| 12 | GET | `/api/rag/knowledge/status` | Knowledge | ~20ms |
| 13 | POST | `/api/rag/knowledge/import-json` | Knowledge | ~30s |
| 14 | POST | `/api/rag/health-summary` | Health | ~2-3s |
| 15 | POST | `/api/rag/health-suggestion` | Health | ~3-5s |
| 16 | POST | `/api/rag/config/refresh` | Config | <10ms |
| 17 | GET | `/api/rag/config/list` | Config | <5ms |
| 18 | GET | `/api/rag/config/{scene}` | Config | <5ms |
| 19 | POST | `/api/rag/config/seed` | Config | <20ms |
| 20 | POST | `/api/rag/config/test` | Config | ~1-3s |
| 21 | POST | `/api/rag/feedback` | System | <5ms |
| 22 | POST | `/api/rag/dialogue/start` | Dialogue | ~2-4s |
| 23 | POST | `/api/rag/dialogue/continue` | Dialogue | ~2-4s |
| 24 | GET | `/api/rag/dialogue/{session_id}` | Dialogue | <20ms |
| 25 | POST | `/api/rag/advice/interpret` | Advice | ~2-3s |

---

## 十二、Java 调用方式

```java
// 1. 健康检查
Boolean healthy = restTemplate.getForObject(
    "http://localhost:8000/api/rag/health", Map.class);

// 2. 智能导诊
Map<String, Object> request = Map.of("query", "头痛发热咳嗽", "top_k", 5);
Map result = restTemplate.postForObject(
    "http://localhost:8000/api/rag/search", request, Map.class);
Map data = (Map) result.get("data");
String dept = (String) data.get("department");

// 3. 病历提取
Map<String, Object> emrRequest = Map.of(
    "symptom_text", "...",
    "health_record", healthRecordMap,
    "patient_info", patientInfoMap
);
Map emrResult = restTemplate.postForObject(
    "http://localhost:8000/api/rag/emr/extract", emrRequest, Map.class);

// 4. 健康建议 (双表输入)
Map<String, Object> suggestionRequest = Map.of(
    "health_record", hrMap,
    "consultation", consultMap
);
Map suggestionResult = restTemplate.postForObject(
    "http://localhost:8000/api/rag/health-suggestion", suggestionRequest, Map.class);

// 5. 多轮对话 Agent
Map<String, Object> startReq = Map.of("patient_id", 1001, "initial_symptom", "头痛三天");
Map startResp = restTemplate.postForObject(
    "http://localhost:8000/api/rag/dialogue/start", startReq, Map.class);
Map dialogueData = (Map) startResp.get("data");
String sessionId = (String) dialogueData.get("session_id");

// 患者回答后继续
Map<String, Object> continueReq = Map.of("session_id", sessionId, "patient_input", "有恶心畏光");
Map continueResp = restTemplate.postForObject(
    "http://localhost:8000/api/rag/dialogue/continue", continueReq, Map.class);

// 6. 医嘱解读 (advice_interpret 场景)
Map<String, Object> adviceReq = Map.of(
    "doctor_advice", "低盐低脂饮食，硝苯地平缓释片 30mg qd",
    "patient_context", "患者张三，男，65岁，高血压5年"
);
Map adviceResp = restTemplate.postForObject(
    "http://localhost:8000/api/rag/advice/interpret", adviceReq, Map.class);

// 7. 测试 Prompt (管理员调优)
Map<String, Object> testReq = Map.of("scene", "triage", "test_input", "头痛发热");
Map testResp = restTemplate.postForObject(
    "http://localhost:8000/api/rag/config/test", testReq, Map.class);
```

---

## 十三、启动服务

```bash
# 开发模式 (热重载)
cd "d:/medic project"
uvicorn rag-db.src.api_server:app --host 0.0.0.0 --port 8000 --reload

# 生产模式 (4 worker)
uvicorn rag-db.src.api_server:app --host 0.0.0.0 --port 8000 --workers 4

# API 文档 (自动生成)
# Swagger UI:  http://localhost:8000/api/docs
# ReDoc:       http://localhost:8000/api/redoc
```
