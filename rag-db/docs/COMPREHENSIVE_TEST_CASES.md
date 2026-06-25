# RAG 医疗知识库 — 10个全覆盖测试用例

> **版本**: v1.0  
> **日期**: 2026-06-24  
> **测试文件**: [`rag-db/tests/test_comprehensive_10.py`](../tests/test_comprehensive_10.py)  
> **运行方式**: `cd "d:/medic project" && python rag-db/tests/test_comprehensive_10.py`

---

## 目录

1. [概述](#概述)
2. [系统组件清单](#系统组件清单)
3. [测试用例覆盖矩阵](#测试用例覆盖矩阵)
4. [TC-01: 全组件 Pipeline 端到端测试](#tc-01-全组件-pipeline-端到端测试)
5. [TC-02: 向量存储三 Collection 全覆盖检索](#tc-02-向量存储三-collection-全覆盖检索)
6. [TC-03: Reranker 精排全功能测试](#tc-03-reranker-精排全功能测试)
7. [TC-04: QueryOptimizer 全模式与功能测试](#tc-04-queryoptimizer-全模式与功能测试)
8. [TC-05: DeepSeekClient 全部 API 方法测试](#tc-05-deepseekclient-全部-api-方法测试)
9. [TC-06: 知识库构建与数据管理测试](#tc-06-知识库构建与数据管理测试)
10. [TC-07: 图表生成器全覆盖测试](#tc-07-图表生成器全覆盖测试)
11. [TC-08: 边界条件与异常处理测试](#tc-08-边界条件与异常处理测试)
12. [TC-09: 配置加载与模型验证测试](#tc-09-配置加载与模型验证测试)
13. [TC-10: 性能基准与压力测试](#tc-10-性能基准与压力测试)
14. [运行结果示例](#运行结果示例)

---

## 概述

本测试套件包含 **10个综合测试用例**，每个测试用例覆盖 RAG 系统的多个组成部分。测试采用**自下而上的覆盖策略**：从配置加载 → 知识库构建 → 向量检索 → Reranker精排 → 查询优化 → LLM生成 → 图表可视化 → 边界异常 → 性能基准，形成完整的测试覆盖链。

### 设计原则

| 原则 | 说明 |
|------|------|
| **全覆盖** | 每个测试用例调用 ≥3 个组件，10个用例合计覆盖全部 8 个模块的 50+ 方法 |
| **交叉验证** | 同一功能在不同测试用例中以不同角度验证（如 VectorStore 在 TC-01/02/06/08/10 中均有覆盖） |
| **优雅降级** | 当组件不可用时（如模型未下载、API Key 未配置），自动 SKIP 而非 FAIL |
| **可追溯** | 每个测试用例记录覆盖的组件清单、检查项详情、执行耗时 |

---

## 系统组件清单

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG 医疗知识库系统架构                          │
├───────────────┬─────────────────────────────────────────────────┤
│  #  │ 模块     │ 类/函数                     │ 方法数           │
├───────────────┼─────────────────────────────────────────────────┤
│  1 │ config.py          │ (常量)                      │ 4 常量          │
│  2 │ build_knowledge_   │ load_diseases               │ 4 函数          │
│    │ base.py            │ analyze_data                │                 │
│    │                    │ prepare_disease_knowledge    │                 │
│    │                    │ prepare_department_info      │                 │
│    │                    │ build_all                    │                 │
├────┼────────────────────┼─────────────────────────────┼─────────────────┤
│  3 │ query_engine.py    │ VectorStore                 │ 8 方法          │
│    │                    │ ├─ search_disease           │                 │
│    │                    │ ├─ search_by_symptom        │                 │
│    │                    │ ├─ search_department        │                 │
│    │                    │ ├─ comprehensive_search     │                 │
│    │                    │ ├─ add_diseases             │                 │
│    │                    │ ├─ add_symptoms             │                 │
│    │                    │ ├─ get_stats                │                 │
│    │                    │ └─ get_collection           │                 │
├────┼────────────────────┼─────────────────────────────┼─────────────────┤
│  4 │ reranker.py        │ Reranker                    │ 3 方法          │
│    │                    │ ├─ rerank                   │                 │
│    │                    │ ├─ rerank_results           │                 │
│    │                    │ └─ get_info                 │                 │
├────┼────────────────────┼─────────────────────────────┼─────────────────┤
│  5 │ deepseek_client.py │ DeepSeekClient              │ 4 方法          │
│    │                    │ ├─ recommend_department     │                 │
│    │                    │ ├─ extract_symptoms         │                 │
│    │                    │ ├─ chat                     │                 │
│    │                    │ └─ health_check             │                 │
│    │                    │ RAGPipeline                 │ 2 方法          │
│    │                    │ ├─ query                    │                 │
│    │                    │ └─ optimize_query           │                 │
│    │                    │ _get_env_config             │ 1 函数          │
├────┼────────────────────┼─────────────────────────────┼─────────────────┤
│  6 │ query_optimizer.py │ QueryOptimizer              │ 10 方法         │
│    │                    │ ├─ optimize                 │                 │
│    │                    │ ├─ _optimize_with_llm       │                 │
│    │                    │ ├─ _optimize_with_rules     │                 │
│    │                    │ ├─ _infer_body_parts        │                 │
│    │                    │ ├─ _check_emergency         │                 │
│    │                    │ ├─ clear_cache              │                 │
│    │                    │ ├─ get_cache_stats          │                 │
│    │                    │ ├─ get_dictionary_stats     │                 │
│    │                    │ ├─ add_colloquial_term      │                 │
│    │                    │ └─ batch_add_terms          │                 │
│    │                    │ get_optimizer (singleton)    │ 1 函数          │
│    │                    │ COLLOQUIAL_MAP (dict)        │ 85 标准术语     │
│    │                    │ _COLLOQUIAL_TO_STANDARD      │ 600+ 口语条目   │
├────┼────────────────────┼─────────────────────────────┼─────────────────┤
│  7 │ chart_generator.py │ ChartGenerator              │ 11 方法         │
│    │                    │ ├─ generate_all             │                 │
│    │                    │ ├─ chart_category_accuracy  │                 │
│    │                    │ ├─ chart_confidence_comp.   │                 │
│    │                    │ ├─ chart_latency_dist.      │                 │
│    │                    │ ├─ chart_latency_comp._ab   │                 │
│    │                    │ ├─ chart_optimization_b/a   │                 │
│    │                    │ ├─ chart_optimization_gain  │                 │
│    │                    │ ├─ chart_token_analysis     │                 │
│    │                    │ ├─ chart_comprehensive_t.   │                 │
│    │                    │ ├─ chart_radar_comparison   │                 │
│    │                    │ └─ chart_dashboard          │                 │
├────┼────────────────────┼─────────────────────────────┼─────────────────┤
│  8 │ download_reranker. │ main()                      │ 1 函数          │
│    │ py                 │ SAVE_PATH / MODEL_NAME      │ 2 常量          │
└────┴────────────────────┴─────────────────────────────┴─────────────────┘
```

---

## 测试用例覆盖矩阵

下表展示每个测试用例覆盖的组件（● 主要覆盖，○ 间接覆盖）：

| 组件 \ 测试用例 | TC-01 | TC-02 | TC-03 | TC-04 | TC-05 | TC-06 | TC-07 | TC-08 | TC-09 | TC-10 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **config.py** | ● | | | | | | | | ● | |
| **build_knowledge_base.py** | | | | | | ● | | | | |
| **query_engine.py (VectorStore)** | ● | ● | ○ | | ○ | ● | | ● | | ● |
| **reranker.py (Reranker)** | ● | | ● | | | | | ● | | ● |
| **deepseek_client.py (DeepSeekClient)** | ● | | | | ● | | | ● | | |
| **deepseek_client.py (RAGPipeline)** | ● | | | | | | | | | ○ |
| **query_optimizer.py** | ● | | | ● | | | | ● | | ● |
| **chart_generator.py** | | | | | | | ● | | | |
| **download_reranker.py** | | | | | | | | | ● | |

---

## TC-01: 全组件 Pipeline 端到端测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-01 |
| **用例名称** | 全组件 Pipeline 端到端测试 |
| **覆盖组件** | `config`, `VectorStore`, `Reranker`, `QueryOptimizer`, `DeepSeekClient`, `RAGPipeline` |
| **复杂度** | ⭐⭐⭐⭐⭐ (最高) |
| **预计耗时** | 30-120s (取决于 LLM API 延迟) |

### 测试目标

验证从用户原始输入到科室推荐的完整 RAG Pipeline：**查询优化 → 向量检索 → Reranker精排 → LLM生成**，确保各组件正确串联。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 1.1 | 加载配置路径 | `config.EMBEDDING_MODEL_PATH` 等 | 4个配置常量均已设置且路径存在 |
| 1.2 | 查询优化 (3条口语化输入) | `QueryOptimizer.optimize()` | 每条输入均提取出症状关键词 |
| 1.3 | 向量检索 (3条标准症状) | `VectorStore.comprehensive_search()` | 返回非空结果，科室匹配预期 |
| 1.4 | 单独检索方法 | `search_disease`, `search_by_symptom`, `search_department` | 三个方法均正常返回 |
| 1.5 | Reranker 精排 | `Reranker.rerank_results()` | 重排后保留 `cosine_score`，分数变化 |
| 1.6 | LLM 生成 | `DeepSeekClient.health_check()`, `recommend_department()`, `extract_symptoms()`, `chat()` | 健康检查通过，科室推荐有效，症状提取成功 |
| 1.7 | Pipeline 集成 | `RAGPipeline.query()`, `RAGPipeline.optimize_query()` | 返回结构完整 (query, query_optimization, rag_results, recommendation, search_query) |

### 预期结果

```json
{
  "query": "头痛发热咳嗽",
  "query_optimization": {
    "original_query": "头痛发热咳嗽",
    "optimized_query": "头痛 发热 咳嗽",
    "symptoms": ["头痛", "发热", "咳嗽"]
  },
  "rag_results": {
    "disease_results": [...],
    "symptom_direct": [...],
    "primary_recommendation": {
      "department": "呼吸内科",
      "disease": "感冒",
      "confidence": 0.75
    }
  },
  "recommendation": {
    "department": "呼吸内科",
    "disease": "感冒",
    "confidence": 85,
    "reasoning": "...",
    "suggestion": "..."
  },
  "search_query": "头痛 发热 咳嗽"
}
```

---

## TC-02: 向量存储三 Collection 全覆盖检索

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-02 |
| **用例名称** | 向量存储三 Collection 全覆盖检索 |
| **覆盖组件** | `VectorStore` 全部8个方法 |
| **复杂度** | ⭐⭐⭐ |
| **预计耗时** | 10-30s |

### 测试目标

对 ChromaDB 的三个 Collection 进行全面检索测试，验证数据一致性、统计准确性和跨 Collection 交叉验证。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 2.1 | 三Collection分别检索 | `search_disease("头痛发热咳嗽")`, `search_by_symptom("胸闷气短")`, `search_department("心内科")`, `search_department("骨科")` | 每个方法返回 ≥1 条结果，分数有效 |
| 2.2 | 综合检索 (5条查询) | `comprehensive_search()` | 返回结构完整: disease_results, symptom_direct, primary_recommendation, all_departments |
| 2.3 | 数据库统计 | `get_stats()` | 3个Collection均存在，disease=8808, department=54 |
| 2.4 | 跨Collection一致性 | `comprehensive_search("发热咳嗽咽痛")` | disease_results 和 symptom_direct 的科室存在交集 |

### 关键检查项

```
✓ Collection 'disease_knowledge' 存在: 8808 条记录
✓ Collection 'symptom_dept_direct' 存在: ~2000 条记录
✓ Collection 'department_info' 存在: 54 条记录
✓ search_disease → 呼吸内科 (score=72.3%)
✓ search_by_symptom → 呼吸内科 (score=68.1%)
✓ search_department → 呼吸内科 (56 diseases)
✓ 跨Collection一致性: overlap={'呼吸内科', '内科'}
```

---

## TC-03: Reranker 精排全功能测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-03 |
| **用例名称** | Reranker 精排全功能测试 |
| **覆盖组件** | `Reranker` (rerank, rerank_results, get_info), `VectorStore` (提供候选) |
| **复杂度** | ⭐⭐⭐⭐ |
| **预计耗时** | 15-60s (取决于GPU/CPU) |

### 测试目标

全面测试 BGE-reranker-v2-m3 Cross-Encoder 的模型加载、基础rerank、VectorStore集成、分数归一化、延迟影响等。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 3.1 | 模型信息 | `Reranker.get_info()` | 返回 model_path, model_loaded, use_fp16 |
| 3.2 | 基础rerank (5条候选) | `Reranker.rerank()` | 返回排序结果，score递减，包含 index/score/document |
| 3.3 | VectorStore集成重排 | `Reranker.rerank_results()` | cosine_score保留，分数更新且归一化到[0,1] |
| 3.4 | 延迟测试 (3次) | `Reranker.rerank_results()` × 3 | 平均延迟 < 5000ms |

### Reranker 工作流程

```
cosine搜索 (top-20) → Cross-Encoder评分 → sigmoid归一化 → 重排序 (top-5)
                                                              ↓
                                          原始cosine分数保存为 cosine_score
```

### 预期分数变化示例

```
原始排序 (cosine):     感冒(0.85) → 偏头痛(0.80) → 鼻炎(0.78)
重排后 (cross-encoder): 感冒(0.92) → 肺炎(0.76) → 鼻炎(0.71) → 偏头痛(0.58)
```

---

## TC-04: QueryOptimizer 全模式与功能测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-04 |
| **用例名称** | QueryOptimizer 全模式与功能测试 |
| **覆盖组件** | `QueryOptimizer` 全部10个方法 + `get_optimizer` + 词典数据 |
| **复杂度** | ⭐⭐⭐⭐⭐ |
| **预计耗时** | 5-30s (rule模式极快) |

### 测试目标

全面测试查询优化器的三种工作模式、缓存机制、紧急检测、部位推断、词典动态更新、单例工厂等全部功能。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 4.1 | Rule模式: 口语标准化 (6条) | `optimize()` | 6条口语化描述均正确标准化为医学术语 |
| 4.2 | Rule模式: 标准术语保留 | `optimize("头痛发热咳嗽流鼻涕")` | 已标准术语被正确识别 |
| 4.3 | 非医疗/空输入处理 | `optimize("今天天气真好")`, `optimize("")` | 非医疗无症状，空输入返回空 |
| 4.4 | 身体部位推断 (6组) | `_infer_body_parts()` | 症状→部位映射正确 (腹痛→腹部, 皮疹→皮肤等) |
| 4.5 | 紧急信号检测 (5条) | `_check_emergency()` | 胸痛/大出血/晕倒→True, 感冒/拉肚子→False |
| 4.6 | 缓存机制 | `optimize()` × 2 | 第二次命中缓存，结果一致 |
| 4.7 | 词典统计 | `get_dictionary_stats()` | standard_terms > 80, colloquial_entries > 500 |
| 4.8 | 词典动态更新 | `add_colloquial_term()`, `batch_add_terms()` | 新增映射可被识别 |
| 4.9 | 清空缓存 | `clear_cache()` | 返回清空的条目数 |
| 4.10 | 单例工厂 | `get_optimizer()` × 2 | 两次获取返回同一实例 |

### 三种工作模式对比

| 模式 | 原理 | 速度 | 准确度 | 网络依赖 |
|------|------|------|--------|----------|
| `rule` | 600+ 口语→标准词典匹配 | ~1ms | 中等 (覆盖已知表达) | 无 |
| `llm` | DeepSeek API 智能标准化 | ~500-2000ms | 高 (理解上下文) | 需要 |
| `hybrid` | LLM优先，失败回退rule | ~1-2000ms | 高 (LLM + 兜底) | 降级可用 |

---

## TC-05: DeepSeekClient 全部 API 方法测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-05 |
| **用例名称** | DeepSeekClient 全部 API 方法测试 |
| **覆盖组件** | `DeepSeekClient` (recommend_department, extract_symptoms, chat, health_check), `_get_env_config` |
| **复杂度** | ⭐⭐⭐⭐ |
| **预计耗时** | 15-60s (取决于 API 延迟) |

### 测试目标

全面测试LLM客户端的4个核心API方法、配置级联fallback机制、JSON解析错误处理、紧急症状识别。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 5.0 | 配置fallback (无API时) | `_get_env_config(prefix="LLM")`, `_get_env_config(prefix="OPTIMIZER")` | 多级fallback正确 |
| 5.1 | 连通性检查 | `health_check()` | status=ok, 返回模型名 |
| 5.2 | 科室推荐 | `recommend_department()` | 返回结构完整: department/disease/confidence/reasoning/suggestion/alternatives/emergency_warning/raw_response/usage |
| 5.3 | 症状提取 | `extract_symptoms()` | 返回 main_symptoms/duration/severity/body_parts |
| 5.4 | 通用对话 | `chat()` | 返回非空响应 |
| 5.5 | 紧急症状推荐 | `recommend_department("剧烈胸痛...")` | emergency_warning=true |

### API 配置多级 Fallback 链

```
参数传入 → LLM_API_KEY → DEEPSEEK_API_KEY → 报错
参数传入 → LLM_BASE_URL → https://api.deepseek.com
参数传入 → LLM_MODEL → deepseek-v4-flash
```

### JSON 解析失败处理

当 LLM 返回非标准 JSON 时，`recommend_department` 的 fallback 逻辑：

```python
{
  "department": "无法解析",
  "disease": "无法解析",
  "confidence": 0,
  "reasoning": "LLM 返回格式异常",
  "parse_error": True
}
```

---

## TC-06: 知识库构建与数据管理测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-06 |
| **用例名称** | 知识库构建与数据管理测试 |
| **覆盖组件** | `build_knowledge_base` (load_diseases, analyze_data, prepare_disease_knowledge, prepare_department_info), `VectorStore` (add_diseases, add_symptoms) |
| **复杂度** | ⭐⭐⭐ |
| **预计耗时** | 10-30s (仅数据加载，不重建知识库) |

### 测试目标

测试数据加载、质量分析、Collection准备、增量写入的全流程。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 6.1 | 加载JSONL数据 | `load_diseases(DATA_PATH)` | 加载 8808 条疾病记录 |
| 6.2 | 数据分析 | `analyze_data(diseases)` | 返回 total/empty_symptom/empty_dept/empty_desc/unique_depts/unique_symptoms |
| 6.3 | 主知识库准备 | `prepare_disease_knowledge(diseases)` | docs/metas/ids 数量一致，字段完整 |
| 6.4 | 科室信息准备 | `prepare_department_info(diseases, stats)` | 54个科室，dept/disease_count/common_symptoms/sample_diseases 字段完整 |
| 6.5 | 增量添加疾病 | `VectorStore.add_diseases()` | 执行成功，新疾病可检索 |
| 6.6 | 增量添加症状 | `VectorStore.add_symptoms()` | 执行成功 |

### 数据质量统计 (预期)

| 指标 | 预期值 |
|------|--------|
| 总疾病数 | 8,808 |
| 无症状字段 | ≤ 10% |
| 无科室字段 | ≤ 20% |
| 无描述字段 | ≤ 15% |
| 独立科室数 | 54 |
| 独立症状数 | > 20,000 |

---

## TC-07: 图表生成器全覆盖测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-07 |
| **用例名称** | 图表生成器全覆盖测试 |
| **覆盖组件** | `ChartGenerator` 全部 11 个方法 |
| **复杂度** | ⭐⭐⭐ |
| **预计耗时** | 5-15s |

### 测试目标

使用模拟数据测试 ChartGenerator 的全部 10 种图表（9种+全局generate_all），验证输出文件完整性。

### 测试步骤

| 步骤 | 操作 | 覆盖方法 | 验证点 |
|------|------|----------|--------|
| 7.0 | 构建模拟测试数据 | `_build_mock_test_data()` | 生成 A/B/C/D 四组模拟数据 + summary |
| 7.1 | 初始化图表生成器 | `ChartGenerator.__init__()` | 输出目录创建成功 |
| 7.2 | 逐个生成9种图表 | 9个 chart_* 方法 | 每个方法生成 PNG 文件 |
| 7.3 | 批量生成 | `generate_all()` | 生成 > 0 张图表 |

### 10种图表类型

| # | 图表名称 | 方法 | 数据源 |
|---|----------|------|--------|
| 1 | 分类准确率对比柱状图 | `chart_category_accuracy()` | A组 |
| 2 | 置信度分布箱线图 | `chart_confidence_comparison()` | A+B组 |
| 3 | 延迟分布直方图 | `chart_latency_distribution()` | A组 |
| 4 | 延迟对比柱状图 | `chart_latency_comparison_ab()` | A+B组 |
| 5 | 优化前后配对柱状图 | `chart_optimization_before_after()` | C组 |
| 6 | 优化增益瀑布图 | `chart_optimization_gain()` | C组 |
| 7 | Token消耗分析 (堆叠+饼图+散点) | `chart_token_analysis()` | B组 |
| 8 | 综合时序对比 | `chart_comprehensive_timing()` | D组 |
| 9 | 多维度雷达图 | `chart_radar_comparison()` | summary |
| 10 | 四合一仪表盘 | `chart_dashboard()` | summary |

---

## TC-08: 边界条件与异常处理测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-08 |
| **用例名称** | 边界条件与异常处理测试 |
| **覆盖组件** | 全部模块的边界路径 |
| **复杂度** | ⭐⭐⭐ |
| **预计耗时** | 5-20s |

### 测试目标

测试所有模块在边界条件和异常场景下的鲁棒性，确保系统不会因异常输入而崩溃。

### 测试步骤

| 步骤 | 操作 | 测试场景 | 验证点 |
|------|------|----------|--------|
| 8.1 | VectorStore 边界 | 超短查询 ("头")、超长查询 (150字)、仅空格、英文、特殊字符、非医疗、罕见症状 | 不崩溃，空结果可接受 |
| 8.2 | QueryOptimizer 边界 | 空字符串、仅空格、纯数字、纯符号、极短 ("疼")、极长 (250字)、中英混合、None-like | 不崩溃，返回合法结构 |
| 8.3 | Reranker 边界 | 空候选列表、单候选、空查询 | 空候选返回空，单候选正常处理 |
| 8.4 | DeepSeekClient 异常 | 代码级验证 JSONDecodeError / API异常 fallback | parse_error 和 服务异常 fallback 存在 |
| 8.5 | 重叠匹配优先级 | "肚子疼拉稀" | 长匹配优先，"肚子疼"+ "拉稀"均被识别 |

### 边界测试清单

| 类别 | 输入示例 | 预期行为 |
|------|----------|----------|
| 空/空白 | `""`, `"   "` | 静默返回空结果 |
| 极短 | `"头"` | 返回结果或空，不崩溃 |
| 极长 | `"头痛发热咳嗽..."` × 5 | 正常处理 |
| 非中文 | `"headache fever cough"` | 正常处理或返回低置信度 |
| 特殊字符 | `"!@#$%^&*()"` | 不崩溃 |
| 非医疗 | `"今天天气真好"` | 返回低置信度或空 |
| 紧急 | `"剧烈胸痛呼吸困难"` | emergency_warning=true |

---

## TC-09: 配置加载与模型验证测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-09 |
| **用例名称** | 配置加载与模型验证测试 |
| **覆盖组件** | `config.py`, `download_reranker.py`, `.env` |
| **复杂度** | ⭐⭐ |
| **预计耗时** | 1-3s |

### 测试目标

验证所有配置常量、环境变量、模型文件和数据文件的可用性。

### 测试步骤

| 步骤 | 操作 | 验证点 |
|------|------|--------|
| 9.1 | config.py 常量 | EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH, DB_PATH, DATA_PATH 已设置 |
| 9.2 | 模型文件存在性 | BGE-M3 目录含模型文件 (.safetensors/.bin)，BGE-Reranker 目录含模型文件 |
| 9.3 | ChromaDB 数据库 | DB_PATH 目录存在且含 chroma 文件 |
| 9.4 | 数据文件 | medical.json 存在，大小约 47MB |
| 9.5 | .env 环境变量 | DEEPSEEK_API_KEY, LLM_API_KEY, EMBEDDING_MODEL_PATH, RERANKER_MODEL_PATH 已设置 |
| 9.6 | 下载脚本 | download_reranker.py 存在，含 main() 和 SAVE_PATH |

### 预期路径配置

```
EMBEDDING_MODEL_PATH  →  D:\floder-for-claude\medic\bge-m3
RERANKER_MODEL_PATH   →  D:\floder-for-claude\medic\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots\...
DB_PATH               →  d:\medic project\medical_rag_db
DATA_PATH             →  d:\medic project\rag data\openkg data\medical.json
```

---

## TC-10: 性能基准与压力测试

### 基本信息

| 项目 | 内容 |
|------|------|
| **用例编号** | TC-10 |
| **用例名称** | 性能基准与压力测试 |
| **覆盖组件** | `VectorStore` (延迟/QPS), `Reranker` (开销), `QueryOptimizer` (缓存加速) |
| **复杂度** | ⭐⭐⭐ |
| **预计耗时** | 15-60s |

### 测试目标

建立各组件延迟基准线，测量 Reranker 开销、QueryOptimizer 缓存加速效果、重复查询稳定性。

### 测试步骤

| 步骤 | 操作 | 测量指标 | 基准线 |
|------|------|----------|--------|
| 10.1 | search_disease × 15 | avg/min/max/p95 延迟 | avg < 50ms, p95 < 100ms |
| 10.2 | QPS 吞吐量 (30 queries) | QPS | > 10 QPS |
| 10.3 | comprehensive_search × 10 | 综合延迟 | avg < 200ms |
| 10.4 | Reranker 延迟 × 5 (10候选) | 精排开销 | avg < 2000ms |
| 10.5 | QueryOptimizer 冷 vs 热查询 | 缓存加速比 | speedup > 10x |
| 10.6 | 重复查询稳定性 × 5 | 结果一致性 | 5次结果相同 |

### 性能基准线 (预期)

| 操作 | 预期延迟 | 说明 |
|------|----------|------|
| `search_disease` | 5-20ms | 余弦相似度，纯向量计算 |
| `comprehensive_search` (no reranker) | 10-30ms | 3 Collections 并发查询 |
| `comprehensive_search` (with reranker) | 100-2000ms | 取决于 GPU/CPU |
| `QueryOptimizer.optimize` (rule) | < 1ms | 词典匹配 |
| `QueryOptimizer.optimize` (llm) | 500-2000ms | DeepSeek API |
| `DeepSeekClient.recommend_department` | 1000-5000ms | LLM 推理 |

---

## 运行结果示例

### 运行命令

```bash
cd "d:/medic project"
python rag-db/tests/test_comprehensive_10.py
```

### 预期输出 (摘要)

```
======================================================================
  RAG Medical Knowledge Base — 10 Comprehensive Test Cases
  All Components Coverage Test Suite
======================================================================
  Python: 3.11.0
  Working Dir: d:\medic project
  Timestamp: 2026-06-24 12:00:00

======================================================================
  TC-01: 全组件 Pipeline 端到端测试
======================================================================
  --- 1.1 配置路径验证 ---
    [PASS] config.EMBEDDING_MODEL_PATH = D:\floder-for-claude\medic\bge-m3
    [PASS] config.RERANKER_MODEL_PATH = D:\floder-for-claude\...
    [PASS] config.DB_PATH = ...\medical_rag_db
    [PASS] config.DATA_PATH = ...\medical.json
  --- 1.2 QueryOptimizer 查询优化 ---
    [PASS] '肚子疼拉稀想吐没胃口' → ['腹痛', '腹泻', '恶心', '食欲不振']
  ... (省略中间输出)

======================================================================
  最终汇总报告
======================================================================

  TC     测试用例                                检查项    通过   结果
  ------ ----------------------------------- -------- ------ ------
  TC-01  全组件 Pipeline 端到端测试                 18     16   PARTIAL
  TC-02  向量存储三 Collection 全覆盖检索           16     15   PARTIAL
  TC-03  Reranker 精排全功能测试                   12     10   PARTIAL
  TC-04  QueryOptimizer 全模式与功能测试            18     18   PASS
  TC-05  DeepSeekClient 全部 API 方法测试           12      8   PARTIAL
  TC-06  知识库构建与数据管理测试                   12     12   PASS
  TC-07  图表生成器全覆盖测试                       5      5   PASS
  TC-08  边界条件与异常处理测试                     20     20   PASS
  TC-09  配置加载与模型验证测试                     15     13   PARTIAL
  TC-10  性能基准与压力测试                         8      8   PASS
  ------ ----------------------------------- -------- ------ ------
  TOTAL                                           136    125   91.9%

  --- 组件覆盖摘要 ---
  覆盖组件数: 48
    - ChartGenerator
    - DeepSeekClient
    - DeepSeekClient.chat
    - DeepSeekClient.extract_symptoms
    - DeepSeekClient.health_check
    - DeepSeekClient.recommend_department
    - QueryOptimizer
    - QueryOptimizer(hybrid)
    ... (共48个组件/方法)

  详细结果已保存: .../test_results/comprehensive_10_20260624_120000.json
```

### JSON 输出结构

```json
{
  "meta": {
    "timestamp": "2026-06-24T12:00:00",
    "test_version": "comprehensive_10_v1.0",
    "total_test_cases": 10
  },
  "results": [
    {
      "tc_id": "TC-01",
      "name": "全组件 Pipeline 端到端测试",
      "description": "验证从用户输入到科室推荐的完整 RAG Pipeline...",
      "components": ["config", "VectorStore", "Reranker", "QueryOptimizer", "DeepSeekClient", "RAGPipeline"],
      "total_checks": 18,
      "passed_checks": 16,
      "all_passed": false,
      "timing_ms": 45230.5,
      "checks": [
        {"name": "config.EMBEDDING_MODEL_PATH 存在", "passed": true, "detail": "..."},
        ...
      ]
    },
    ...
  ]
}
```

---

## 附录: 快速参考

### 单独运行某个测试用例

```python
# 在 Python 中
from test_comprehensive_10 import ComprehensiveTestSuite
suite = ComprehensiveTestSuite()
suite.tc04_query_optimizer_all_modes()  # 仅运行 TC-04
```

### 依赖要求

| 组件 | 依赖条件 | 不可用时行为 |
|------|----------|-------------|
| VectorStore | ChromaDB + BGE-M3 模型 | SKIP TC-02/06/10 部分 |
| Reranker | BGE-reranker-v2-m3 模型 | SKIP TC-03 全部 |
| DeepSeekClient | API Key (DEEPSEEK_API_KEY 或 LLM_API_KEY) | SKIP TC-05 全部 |
| ChartGenerator | matplotlib + numpy | SKIP TC-07 全部 |
| QueryOptimizer (rule) | 无 (纯Python) | 始终可用 |
| build_knowledge_base | medical.json 数据文件 | SKIP TC-06 部分 |

### 测试结果文件

所有测试结果以 JSON 格式保存到 `rag-db/test_results/comprehensive_10_YYYYMMDD_HHMMSS.json`，包含完整的检查项详情、通过/失败状态、执行耗时。
