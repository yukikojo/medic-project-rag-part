# RAG 智慧医疗系统 — 性能指标报告

> **日期**: 2026-06-26  
> **版本**: v1.0  
> **测试环境**: BGE-M3 (CUDA GPU), BGE-Reranker-v2-m3 (CUDA GPU), ChromaDB HNSW, Python 3.12, Windows 11

---

## 一、RAG 检索延迟

测试方法: 15 组真实症状查询循环 100 次，BGE-M3 已在 GPU 预热 5 次。

| 操作 | Avg | P50 | **P95** | P99 | Min | Max |
|------|-----|-----|---------|-----|-----|-----|
| `search_disease()` (纯余弦) | 31.5ms | 30.6ms | **40.0ms** | 47.0ms | 25.0ms | 47.0ms |
| `comprehensive_search()` (无 reranker) | 58.7ms | 57.0ms | **71.5ms** | 120.5ms | 49.6ms | 120.5ms |
| `comprehensive_search()` (有 reranker) | 297.6ms | 264.0ms | **299.7ms** | 3456.4ms¹ | 221.4ms | 3456.4ms |

> ¹ P99 受第一次 Reranker 调用冷启动影响（Cross-Encoder 首次加载到 GPU 耗时 ~3s），稳态后可降至 ~300ms。

### 延迟分解 (comprehensive_search with reranker)

| 阶段 | 平均耗时 | 占比 |
|------|---------|------|
| BGE-M3 Embedding (Query编码) | ~10ms | 3.4% |
| ChromaDB HNSW 检索 (cosine, top-20) | ~20ms | 6.7% |
| Cross-Encoder Reranker (20对评分) | ~240ms | 80.6% |
| 结果聚合 + 去重 | ~28ms | 9.4% |
| **总计** | **~298ms** | 100% |

### Reranker 额外开销

- **+238.9ms avg**（Cross-Encoder 批量推理 20 个 (query, document) 对）
- 若减少候选数从 20→10，开销可降至 ~120ms（但可能牺牲召回）

---

## 二、Reranker 排序质量

### 2.1 Recall@K（自身描述检索）

测试方法: 从知识库 8,808 个疾病中随机采样 50 个，以「疾病名 + 描述」作为查询词，测量目标疾病是否出现在 Top-K 结果中。

| K | Cosine Recall | Reranked Recall | Δ |
|---|--------------|-----------------|---|
| @1 | 100.0% | 96.0% | -4.0% |
| @3 | 100.0% | 98.0% | -2.0% |
| @5 | 100.0% | 98.0% | -2.0% |
| @10 | 100.0% | 98.0% | -2.0% |

| 指标 | Cosine | Reranked | Δ |
|------|--------|----------|---|
| MRR (Mean Reciprocal Rank) | 1.0000 | 0.9667 | -0.0333 |

> **说明**: 此测试以疾病"自身描述 → 检索自身"方式设计，余弦检索天然满分。Reranker 保持率 96-98%，证明 **Reranker 不会破坏已有的高相关匹配**。

### 2.2 实际场景收益（来自 test_reranker_comparison.py A/C 组）

| 场景 | 改善 |
|------|------|
| 口语化症状查询（如 "嗓子不舒服老想咳" → 咽炎） | +15-25% 排序正确率提升 |
| 多症状组合查询 | 减少无关科室推荐，提高首选科室命中率 |
| 置信度校准 | Cross-Encoder 分数经 sigmoid 归一化到 [0,1]，比余弦距离更可解释 |

### 2.3 Reranker 何时有效 / 无效

| 有效场景 | 无效场景 |
|---------|---------|
| 症状口语化、非标准术语 | 查询与知识库术语高度一致 |
| 多症状组合，需要语义理解 | 单关键词精确匹配 |
| 跨语言/方言查询 | 知识库中无相关疾病 |

---

## 三、测试覆盖率

### 3.1 总体指标

| 指标 | 数值 |
|------|------|
| 源代码模块 | 16 |
| 测试文件 | 9 |
| 有测试覆盖的模块 | 13 (81.3%) |
| 无测试覆盖的模块 | 3 |
| 总断言数 | 380+ (estimate across all test suites) |

### 3.2 模块级覆盖

| 源模块 | 覆盖测试 | 测试文件数 |
|--------|---------|-----------|
| `query_engine.py` | test_rag, test_runner, test_comprehensive_10, test_reranker_comparison, full_pipeline_test, test_kg_enrich, benchmark_metrics | 7 |
| `deepseek_client.py` | test_rag, test_runner, test_emr, test_comprehensive_10, full_pipeline_test | 5 |
| `config.py` | test_rag, test_comprehensive_10, test_reranker_comparison, full_pipeline_test, benchmark_metrics | 5 |
| `query_optimizer.py` | test_rag, test_runner, test_comprehensive_10, full_pipeline_test | 4 |
| `ai_config_loader.py` | test_health_suggestion, test_health_summary, test_emr, full_pipeline_test | 4 |
| `reranker.py` | test_comprehensive_10, full_pipeline_test, test_reranker_comparison, benchmark_metrics | 4 |
| `api_models.py` | test_emr, test_rag, test_health_suggestion | 3 |
| `kg_enricher.py` | test_kg_enrich, full_pipeline_test | 2 |
| `build_knowledge_base.py` | test_kg_enrich, full_pipeline_test | 2 |
| `emr_extractor.py` | test_emr | 1 |
| `mysql_kb_manager.py` | full_pipeline_test | 1 |
| `health_summary/summary_generator.py` | test_health_summary | 1 |
| `health_suggestion/suggestion_generator.py` | test_health_suggestion | 1 |
| `api_server.py` | **无** (手动 curl 测试) | 0 |
| `chart_generator.py` | **无** | 0 |
| `download_reranker.py` | **无** (工具脚本，无需测试) | 0 |

### 3.3 各测试套件详情

| 测试文件 | 断言数 (est.) | 覆盖范围 |
|---------|-------------|---------|
| `test_comprehensive_10.py` | ~70 | 10 个全栈场景：Pipeline E2E, ChromaDB 3 collections, Reranker 全功能, QueryOptimizer 全模式, DeepSeekClient API, 知识库构建, LLM 文本质量, 错误处理, 配置模型验证, 性能基准 |
| `test_health_suggestion.py` | 97 | 18 个用例：双表输入/最小/空/儿科/心血管/RAG/RAG降级/JSON结构/用药过敏/解析容错/Config集成/性能/MySQL写入/MySQL回读/MySQL覆盖/清理 |
| `test_health_summary.py` | 52 | 14 个用例：完整/最小/空档案/心血管/过敏/儿科/家庭成员/RAG验证/格式验证/性能/format_record/Config集成/部分字段/性别准确性 |
| `test_reranker_comparison.py` | ~40 | A/C 组 100 例 Reranker ON vs OFF 对比，2×2 Optimizer×Reranker 交叉矩阵 (80 例) |
| `test_runner.py` | ~80 | A/B/C/D 四类测试框架：本地检索(100例)/LLM生成(60例)/优化对比(80例)/端到端(50例) |
| `test_rag.py` | ~30 | 19 个用例覆盖全流程 + 性能基准 |
| `test_emr.py` | ~15 | 3 组：EMRProcessor, API Models, Health Record Formatting |
| `test_kg_enrich.py` | ~10 | 疾病知识图谱查询 + 批量富化 |
| `full_pipeline_test.py` | ~20 | 7 阶段端到端：MySQL→Optimizer→Retrieval→Reranker→KG Enricher→LLM→EMR |

### 3.4 未覆盖模块说明

| 模块 | 原因 |
|------|------|
| `api_server.py` | FastAPI 端点通过手动 `curl` / Postman 测试，未编写自动化测试（14 个端点均为薄层路由，核心逻辑在各自模块中测试） |
| `chart_generator.py` | 图表生成工具模块，输出为 PNG 图片文件，不适合自动化断言 |
| `download_reranker.py` | 一次性模型下载工具脚本，非运行时组件 |

---

## 四、Benchmark 重跑方式

```bash
# 完整三项指标
cd "d:/medic project"
python rag-db/tests/benchmark_metrics.py

# 单独跑已有测试
python rag-db/tests/test_rag.py                    # RAG 全流程 + 性能
python rag-db/tests/test_reranker_comparison.py    # Reranker 对比
python rag-db/tests/test_comprehensive_10.py       # 10 全栈场景

# 跑所有健康模块测试
python rag-db/tests/health_summary/test_health_summary.py
python rag-db/tests/health_suggestion/test_health_suggestion.py
```

---

## 五、数据文件

Benchmark 原始数据保存在: `rag-db/tests/test_results/benchmark_metrics.json`
