# RAG 智慧医疗 AI 引擎

> 基于 8,808 种疾病知识图谱的 RAG 智能导诊系统。四阶段流水线「查询优化 → 向量粗排 → Cross-Encoder 精排 → LLM 推理」，为 Java Spring Boot 后端提供 20 个 AI 能力端点。

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/vector--db-ChromaDB-green.svg)](https://www.trychroma.com/)
[![BGE-M3](https://img.shields.io/badge/embedding-BGE--M3-orange.svg)](https://huggingface.co/BAAI/bge-m3)
[![BGE-Reranker](https://img.shields.io/badge/reranker-BGE--Reranker--v2--m3-red.svg)](https://huggingface.co/BAAI/bge-reranker-v2-m3)
[![MySQL](https://img.shields.io/badge/source--of--truth-MySQL%208.0-blue.svg)](https://www.mysql.com/)

---

## 一、系统架构

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────────────┐
│  患者小程序   │────>│  Java Spring    │────>│  Python AI 引擎 (FastAPI) │
│  (UniApp)    │     │  Boot 后端       │     │  localhost:8000           │
└──────────────┘     └────────┬────────┘     └────────────┬─────────────┘
                              │                            │
                     ┌────────▼────────┐         ┌─────────▼──────────┐
                     │  MySQL 主业务库  │         │  MySQL medical_rag │
                     │  (smart_medical)│         │  (知识库源数据)     │
                     └─────────────────┘         └─────────┬──────────┘
                                                           │ 同步
                                                   ┌───────▼──────────┐
                                                   │    ChromaDB       │
                                                   │  (向量索引)        │
                                                   └──────────────────┘
```

**五阶段 RAG 流水线**：

```
用户症状输入
    │
    ▼
┌──────────────────────────────┐
│ ① 查询优化  retrieval/       │  口语/方言 → 标准医学术语
│   query_optimizer.py         │  "肚子疼拉稀" → "腹痛 腹泻"
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ② 向量粗排  retrieval/       │  BGE-M3 编码 → ChromaDB HNSW 检索
│   query_engine.py            │  8808 条中召回 Top-20, <40ms
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ③ 精排      reranker/        │  BGE-Reranker-v2-m3 Cross-Encoder
│   reranker.py                │  20 对(query,doc)重打分, +238ms
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ④ LLM 推理  generation/      │  DeepSeek/Qwen API 生成推荐
│   deepseek_client.py         │  科室 + 疾病 + 推理依据 + 置信度
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ ⑤ KG 富化   enrichment/      │  药品/食物/检查/并发症补充
│   kg_enricher.py             │  231,291 条关联关系
└──────────────────────────────┘
```

---

## 二、项目结构

```
rag-db/
├── README.md
├── requirements.txt
├── .env                          # API Key + 模型路径 + MySQL 配置
│
├── src/                          # Python AI 引擎源码
│   ├── api_server.py             # FastAPI 网关 (20 个端点)
│   ├── api_models.py             # Pydantic 请求/响应模型
│   ├── ai_config_loader.py       # AI 模型配置 (MySQL + 60s 缓存)
│   ├── config.py                 # 模型路径常量
│   │
│   ├── retrieval/                # ① 查询优化 + ② 向量检索
│   │   ├── query_optimizer.py    #   口语标准化, 紧急信号检测
│   │   └── query_engine.py       #   VectorStore, ChromaDB 3 collections
│   │
│   ├── reranker/                 # ③ Cross-Encoder 精排
│   │   └── reranker.py           #   BGE-Reranker-v2-m3, sigmoid 归一化
│   │
│   ├── generation/               # ④ LLM 生成
│   │   └── deepseek_client.py    #   DeepSeekClient + RAGPipeline 编排
│   │
│   ├── enrichment/               # ⑤ 知识图谱富化
│   │   └── kg_enricher.py        #   MySQL 实时查询 231K 关系
│   │
│   ├── emr/                      # 病历要素提取
│   │   └── emr_extractor.py      #   症状→8 个结构化病历字段
│   │
│   ├── kb_manager/               # 知识库管理
│   │   └── mysql_kb_manager.py   #   MySQL↔ChromaDB 全量/增量同步
│   │
│   ├── health_summary/           # 健康档案 AI 摘要 (面向医生)
│   │   └── summary_generator.py
│   │
│   ├── health_suggestion/        # 个性化生活建议 (面向患者)
│   │   └── suggestion_generator.py
│   │
│   └── tools/                    # 工具脚本
│       ├── build_knowledge_base.py   # 一键构建知识库
│       ├── download_reranker.py      # 下载 Reranker 模型
│       └── chart_generator.py        # Benchmark 图表生成
│
├── tests/                        # 测试套件
│   ├── test_rag.py               #   全流程 + 性能基准
│   ├── test_runner.py            #   A/B/C/D 四类测试框架
│   ├── test_comprehensive_10.py  #   10 个全栈场景
│   ├── test_reranker_comparison.py # Reranker ON/OFF 对比
│   ├── test_emr.py               #   EMR 提取测试
│   ├── test_kg_enrich.py         #   KG 富化测试
│   ├── full_pipeline_test.py     #   7 阶段端到端
│   ├── benchmark_metrics.py      #   性能指标采集
│   ├── health_summary/           #   健康摘要专用测试
│   └── health_suggestion/        #   生活建议专用测试 (含 MySQL 写入)
│
├── docs/                         # 文档
│   ├── API_REFERENCE.md          #   20 个端点完整文档
│   ├── PERFORMANCE_METRICS.md    #   延迟/召回率/覆盖率报告
│   ├── HEALTH_SUMMARY.md         #   健康摘要功能文档
│   ├── HEALTH_SUGGESTION.md      #   生活建议功能文档
│   ├── MYSQL_INTEGRATION.md      #   MySQL 集成文档
│   ├── AI_CONFIG_MANAGEMENT.md   #   AI 配置管理文档
│   └── QUERY_OPTIMIZER.md        #   查询优化器文档
│
└── medical_rag_db/               # ChromaDB 持久化数据 (构建后生成)
```

---

## 三、快速开始

### 3.1 环境要求

- Python 3.12+
- CUDA GPU (推荐) 或 CPU
- MySQL 8.0 (知识库源数据)
- 内存 ≥ 8GB (BGE-M3 ~2.2GB + Reranker ~1.1GB)

### 3.2 安装

```bash
cd rag-db
pip install -r requirements.txt
```

### 3.3 配置 `.env`

```env
# LLM API
DEEPSEEK_API_KEY=sk-xxx
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=qwen-flash

# 本地模型路径
EMBEDDING_MODEL_PATH=D:\floder-for-claude\medic\bge-m3
RERANKER_MODEL_PATH=D:\floder-for-claude\medic\bge-reranker-v2-m3

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=xxx
MYSQL_DATABASE=medical_rag
```

### 3.4 初始化知识库

```bash
# 1. 导入 medical.json → MySQL
curl -X POST http://localhost:8000/api/rag/knowledge/import-json

# 2. MySQL → ChromaDB 全量构建
curl -X POST http://localhost:8000/api/rag/knowledge/rebuild

# 3. 写入 AI 模型默认配置
curl -X POST http://localhost:8000/api/rag/config/seed
```

### 3.5 启动服务

```bash
# 开发模式 (热重载)
uvicorn rag-db.src.api_server:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
uvicorn rag-db.src.api_server:app --host 0.0.0.0 --port 8000 --workers 4

# API 文档
# Swagger UI: http://localhost:8000/api/docs
# ReDoc:      http://localhost:8000/api/redoc
```

---

## 四、API 端点概览

完整文档见 [docs/API_REFERENCE.md](docs/API_REFERENCE.md)。

| # | Method | Path | 说明 | 延迟 |
|---|--------|------|------|------|
| 1 | GET | `/api/rag/health` | 健康检查 | <5ms |
| 2 | POST | `/api/rag/search` | ⭐ 智能导诊 (四阶段完整流水线) | ~2-4s |
| 3 | POST | `/api/rag/symptom/analyze` | 症状结构化 (不调 LLM) | ~1-500ms |
| 4 | POST | `/api/rag/search/enriched` | KG 增强导诊 | ~2-4s |
| 5 | POST | `/api/rag/emr/extract` | 病历要素提取 → 8 字段 | ~2-3s |
| 6 | POST | `/api/rag/assist/info` | AI 辅助问诊提示 | ~2-3s |
| 7 | POST | `/api/rag/diseases/search` | 纯疾病检索 (不到 LLM) | <50ms |
| 8 | GET | `/api/rag/departments` | 全部 54 科室列表 | <10ms |
| 9 | GET | `/api/rag/department/{name}` | 科室详情 | <10ms |
| 10 | POST | `/api/rag/knowledge/rebuild` | 全量重建 ChromaDB | ~45s |
| 11 | POST | `/api/rag/knowledge/sync` | 增量同步 | ~150ms |
| 12 | GET | `/api/rag/knowledge/status` | MySQL vs ChromaDB 一致性 | ~20ms |
| 13 | POST | `/api/rag/knowledge/import-json` | JSON 导入 MySQL | ~30s |
| 14 | POST | `/api/rag/health-summary` | 健康档案 AI 摘要 (医生端) | ~2-3s |
| 15 | POST | `/api/rag/health-suggestion` | 个性化生活建议 (患者端) | ~3-5s |
| 16 | POST | `/api/rag/config/refresh` | 刷新 AI 配置缓存 | <10ms |
| 17 | GET | `/api/rag/config/list` | 列出所有 AI 场景 | <5ms |
| 18 | GET | `/api/rag/config/{scene}` | 查询单个场景配置 | <5ms |
| 19 | POST | `/api/rag/config/seed` | 写入默认配置 | <20ms |
| 20 | POST | `/api/rag/feedback` | 用户反馈收集 | <5ms |

---

## 五、Python API 直接调用

不经过 HTTP，直接在 Python 代码中调用 AI 引擎：

```python
import sys; sys.path.insert(0, "rag-db/src")

# ── 智能导诊 ──
from retrieval import VectorStore
from generation import RAGPipeline

pipeline = RAGPipeline(optimizer_mode="hybrid")
result = pipeline.query("头痛发热咳嗽流鼻涕")
print(result["primary_recommendation"]["department"])  # 呼吸内科

# ── 纯向量检索 (不调 LLM) ──
vs = VectorStore()
diseases = vs.search_disease("头痛发热", top_k=5)
for d in diseases:
    print(f"{d['score']:.1%} | {d['disease']} → {d['departments']}")

# ── 健康档案摘要 ──
from health_summary import HealthSummaryGenerator
gen = HealthSummaryGenerator()
result = gen.generate({
    "member_name": "张三", "gender": 1, "birth_date": "1960-03-15",
    "past_illness": "高血压5年, 2型糖尿病", "allergy": "青霉素过敏",
})
print(result["ai_summary"])

# ── 个性化生活建议 ──
from health_suggestion import HealthSuggestionGenerator
gen = HealthSuggestionGenerator()
result = gen.generate(
    health_record={"past_illness": "高血压", "medication": "硝苯地平"},
    consultation={"symptom_text": "经常头晕", "doctor_advice": "低盐饮食"},
)
for cat in result["suggestions"]:
    print(f"[{cat['category']}] {len(cat['items'])} 条建议")
```

---

## 六、技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| API 框架 | FastAPI + Uvicorn | 20 个 REST 端点，自动生成 Swagger 文档 |
| 嵌入模型 | BGE-M3 | 1024 维，多语言，CUDA 推理 ~10ms |
| 精排模型 | BGE-Reranker-v2-m3 | Cross-Encoder，sigmoid 归一化到 [0,1] |
| 向量数据库 | ChromaDB | HNSW 索引，3 Collections，~100MB |
| 关系数据库 | MySQL 8.0 | 知识库源数据 (medical_rag 库) |
| LLM 网关 | OpenAI-compatible SDK | DeepSeek / Qwen-Flash |
| 知识图谱 | OpenKG | 8,808 疾病 + 312,159 关系三元组 |

---

## 七、性能指标

详见 [docs/PERFORMANCE_METRICS.md](docs/PERFORMANCE_METRICS.md)。

| 指标 | 数值 |
|------|------|
| `search_disease()` 延迟 (P95) | **40.0ms** |
| `comprehensive_search()` + Reranker 延迟 | **299.7ms** |
| Reranker 额外开销 | +238.9ms avg |
| 测试覆盖率 (模块级) | **81.3%** (13/16) |
| 测试断言数 | 380+ |
| 知识库规模 | 8,808 疾病 + 4,826 症状映射 + 54 科室 |

---

## 八、Java 后端对接

Java Spring Boot 通过 HTTP 调用 Python AI 引擎：

```java
// 智能导诊
Map<String, Object> request = Map.of("query", "头痛发热咳嗽", "top_k", 5);
Map result = restTemplate.postForObject(
    "http://localhost:8000/api/rag/search", request, Map.class);

// 病历提取
Map emrRequest = Map.of(
    "symptom_text", "...",
    "health_record", healthRecordMap
);
Map emr = restTemplate.postForObject(
    "http://localhost:8000/api/rag/emr/extract", emrRequest, Map.class);

// 健康建议 (含自动 MySQL 持久化)
Map suggestionRequest = Map.of(
    "health_record", Map.of("record_id", 1, "patient_id", 1, ...),
    "consultation", Map.of("symptom_text", "...")
);
Map suggestions = restTemplate.postForObject(
    "http://localhost:8000/api/rag/health-suggestion", suggestionRequest, Map.class);
```

---

## 九、RAG 原理简述

### 为什么需要 RAG？

| | 纯 LLM | LLM + RAG |
|---|---|---|
| **过程** | LLM 凭训练记忆猜测 | 先检索知识库，再基于检索结果回答 |
| **输出** | "可能是消化内科"（不可靠） | "腹痛→急性胃肠炎→消化内科，置信度 71.5%" |
| **可更新** | 需重新训练模型 | 只需更新知识库数据 |

### 向量检索原理

```
"感冒发烧咳嗽" → BGE-M3 → [0.12, -0.34, 0.56, ...] (1024 维向量)
                                                 ↓
                              与 8808 条疾病向量计算余弦相似度
                                                 ↓
                              HNSW 索引加速，log(N) 次比较
                                                 ↓
                              返回 Top-K 最相似疾病
```

### 两跳推理

```
第一跳: 症状 ──语义检索──> 疾病
        "头痛发热" → 感冒 (74.6%), 流感 (70.4%), ...

第二跳: 疾病 ──字段映射──> 科室
        感冒.cure_department → ["内科", "呼吸内科"]
```

### ChromaDB 3 个 Collection

| Collection | 条目数 | 用途 |
|-----------|--------|------|
| `disease_knowledge` | 8,808 | 疾病综合知识 (症状+科室+药品+简介) |
| `symptom_dept_direct` | 4,826 | 症状→科室直接映射 (跳过疾病) |
| `department_info` | 54 | 科室信息 (诊疗范围+常见症状) |

---

## 十、测试

```bash
# 运行全部测试
python rag-db/tests/test_rag.py
python rag-db/tests/test_runner.py
python rag-db/tests/test_comprehensive_10.py
python rag-db/tests/test_reranker_comparison.py
python rag-db/tests/full_pipeline_test.py

# 健康模块测试
python rag-db/tests/health_summary/test_health_summary.py       # 52 checks
python rag-db/tests/health_suggestion/test_health_suggestion.py # 97 checks

# 性能指标采集
python rag-db/tests/benchmark_metrics.py
```
