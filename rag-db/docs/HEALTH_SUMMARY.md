# 健康档案 AI 摘要 — Health Summary Generator

> **版本**: v2.0  
> **日期**: 2026-06-26  
> **模块**: `src/health_summary/`, `ai_config_loader.py`, `api_server.py`

---

## 一、概述

为 Java 后端传入的 `health_record` 表数据生成面向医生的**专业自然语言健康摘要**，结合 RAG 知识库增强，输出存入 `health_record.ai_summary` 字段。

```
Java 传入 health_record 数据
    │
    ▼
RAG 检索 (根据既往病史检索相关知识)
    │
    ▼
LLM 生成 (专业自然语言段落, 100-200字)
    │
    ▼
返回 → Java 写入 health_record.ai_summary
```

---

## 二、输入输出

### 输入 (Java → Python)

对应概要设计 `health_record` 表字段：

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
| record_id | int | 否 | 档案ID (回写用) |

### 输出 (Python → Java)

```json
{
  "code": 200,
  "data": {
    "ai_summary": "66岁男性，A型血。患有高血压病5年、2型糖尿病3年及高脂血症..."
  },
  "metadata": {
    "model": "qwen-flash",
    "latency_ms": 2831,
    "tokens": {"total_tokens": 1546}
  }
}
```

---

## 三、生成效果示例

### 完整档案

```
输入: 张三，男，66岁，A型血。高血压5年 + 糖尿病3年 + 高脂血症。
      青霉素/头孢过敏。阑尾切除术2019。硝苯地平+二甲双胍+阿托伐他汀。

输出: 66岁男性，A型血。患有高血压病5年、2型糖尿病3年及高脂血症，
      三种基础疾病叠加，临床上需警惕多重心血管代谢综合征风险。
      **有青霉素及头孢类抗生素过敏史，须严格规避相关药物**。
      有阑尾切除术史，无其他重大手术史。目前用药方案：硝苯地平缓释片
      降压、二甲双胍降糖、阿托伐他汀降脂。需特别注意：二甲双胍在
      肾功能不全时乳酸酸中毒风险，以及他汀类药物的肝酶及肌酸激酶监测。
      建议定期监测肾功能、心电图、眼底检查，强化血糖血压血脂控制。
```

### 最小档案

```
输入: 李四，女，缺铁性贫血

输出: 患者李四，女性。诊断为缺铁性贫血（小细胞低色素性贫血），
      需警惕潜在病因如慢性失血或摄入不足。该病可表现为乏力、头晕、
      面色苍白、指甲异常，若未及时干预可能进展为心力衰竭。
      目前无明确过敏史及用药史记录。
```

---

## 四、API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/rag/health-summary` | POST | 生成健康档案 AI 摘要 |

### Java 调用示例

```java
Map<String, Object> request = Map.of(
    "member_name", "张三",
    "gender", 1,
    "birth_date", "1960-03-15",
    "blood_type", "A",
    "allergy", "青霉素过敏, 头孢类",
    "past_illness", "高血压5年, 2型糖尿病",
    "surgery_history", "阑尾切除术 2019",
    "medication", "硝苯地平 30mg qd",
    "is_self", 1
);

Map result = restTemplate.postForObject(
    "http://localhost:8000/api/rag/health-summary", request, Map.class);

Map data = (Map) result.get("data");
String summary = (String) data.get("ai_summary");

// 写入 health_record 表
healthRecordMapper.updateAiSummary(recordId, summary);
```

---

## 五、性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均延迟 | ~3s | BGE-M3 已加载于 GPU |
| Token 消耗 | ~1,500/次 | prompt + completion |
| RAG 上下文 | ~1,200 字 | 检索 5 个相关疾病 |
| 输出长度 | 100-400 字 | 自然语言段落 |

---

## 六、对其他代码的修改

### 6.1 新建文件

| 文件 | 说明 |
|------|------|
| `src/health_summary/__init__.py` | 包入口，导出 `HealthSummaryGenerator` |
| `src/health_summary/summary_generator.py` | 核心模块: `generate()` / `_format_record()` / `_retrieve_rag_context()` |
| `tests/health_summary/test_health_summary.py` | 14 个测试用例，52 个检查项 |

### 6.2 修改 `src/ai_config_loader.py`

**位置**: `_DEFAULT_CONFIGS` 字典末尾

**新增场景**: `health_summary`

```python
"health_summary": {
    "scene": "health_summary",
    "model_name": "qwen-flash",
    "temperature": 0.3,
    "max_tokens": 600,
    "system_prompt": """你是一位资深全科医师...""",
},
```

- 场景数: 6 → 7
- MySQL 表需重新 seed: `python rag-db/src/ai_config_loader.py`

### 6.3 修改 `src/api_server.py`

**位置**: `# AI 模型配置管理` 之前

**新增端点**: `POST /api/rag/health-summary`

```python
@app.post("/api/rag/health-summary", tags=["Health — 健康档案"])
def generate_health_summary(request: dict):
    ...
```

- 端点总数: 13 → 14

### 6.4 未修改的文件

以下文件**不需要修改**（完全复用现有组件）:

| 组件 | 复用方式 |
|------|---------|
| `DeepSeekClient` | 通过 `llm_client` 属性延迟加载 |
| `VectorStore` | 通过 `vector_store` 属性延迟加载，提供 RAG 检索 |
| `ai_config_loader` | 读取 scene=`health_summary` 的 Prompt 和参数 |
| `api_models.py` | 无新模型，直接使用 dict 输入输出 |

---

## 七、设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 输出格式 | 自然语言段落 | 医生快速浏览，存入 TEXT 字段 |
| RAG 增强 | 启用 | 检索相关疾病知识提升专业度 |
| 触发方式 | Java 主动调用 | 灵活控制，按需生成 |
| 文件位置 | `src/health_summary/` | 独立子包，职责清晰 |
| LLM 客户端 | 复用 `DeepSeekClient` | 统一管理 API Key 和连接 |
| Prompt 管理 | MySQL `ai_model_config` | 管理员可在线调优 |
| 测试位置 | `tests/health_summary/` | 独立目录，对应模块结构 |
