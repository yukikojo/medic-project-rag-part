# 测试套件说明

> 10 个活跃测试文件，覆盖 16 个源模块，380+ 断言。

---

## 目录结构

```
tests/
├── README.md                        ← 本文件
│
├── test_runner.py                   ← 最大数据集, 分类准确率报告
├── test_comprehensive_10.py         ← 最广组件覆盖, 10 全栈场景
├── test_reranker_comparison.py      ← Reranker ON/OFF A/B 对比
├── full_pipeline_test.py            ← MySQL→LLM 7 阶段端到端
├── test_emr.py                      ← EMR 提取 + Pydantic 模型
├── test_kg_enrich.py                ← KG 知识图谱富化
├── benchmark_metrics.py             ← 性能指标采集 (延迟/召回率/覆盖率)
├── diagnose_gpu.py                  ← GPU 环境诊断
│
├── health_summary/                  ← 健康档案 AI 摘要
│   └── test_health_summary.py
├── health_suggestion/               ← 个性化生活建议
│   └── test_health_suggestion.py
│
└── archived/                        ← 已归档 (被上述文件完全覆盖)
    ├── README.md
    ├── test_rag.py
    └── test_d_10cases.py
```

---

## 测试文件说明

### test_runner.py（数据集最全）

| 项目 | 说明 |
|------|------|
| **规模** | ~290 条精选测试用例 |
| **分类** | A 组: 向量检索 (100 例) / B 组: LLM 推理 (60 例) / C 组: 查询优化对比 (80 例) / D 组: 端到端 (50 例) |
| **LLM 调用** | B/D 组需要 (无法连接时跳过) |
| **输出** | 分类准确率、延迟、置信度统计、可选图表 |

```bash
python rag-db/tests/test_runner.py
```

---

### test_comprehensive_10.py（覆盖最广）

| 项目 | 说明 |
|------|------|
| **规模** | 10 个全栈测试场景 |
| **覆盖** | TC-01 全流水线 / TC-02 向量库 3 Collections / TC-03 Reranker 全功能 / TC-04 QueryOptimizer 全模式 / TC-05 DeepSeekClient API / TC-06 知识库构建 / TC-07 图表生成 / TC-08 边界与错误处理 / TC-09 配置与模型验证 / TC-10 性能基准 |
| **LLM 调用** | TC-01/TC-05 需要 |

```bash
python rag-db/tests/test_comprehensive_10.py
```

---

### test_reranker_comparison.py（Reranker A/B 对比）

| 项目 | 说明 |
|------|------|
| **规模** | A 组 100 例 + C 组 80 例 |
| **特点** | 唯一做 Reranker ON vs OFF 科学对比，输出"修正数"/"回退数" |
| **LLM 调用** | 无（纯本地对比） |

```bash
python rag-db/tests/test_reranker_comparison.py
```

---

### full_pipeline_test.py（MySQL→LLM 全链路）

| 项目 | 说明 |
|------|------|
| **规模** | 7 阶段：MySQL 数据源 → QueryOptimizer → VectorStore → Reranker → KG Enricher → LLM 生成 → EMR 提取 |
| **特点** | 唯一测试 MySQL 数据源和 KG 富化组合 |
| **LLM 调用** | 阶段 6/7 需要 |

```bash
python rag-db/tests/full_pipeline_test.py
```

---

### test_emr.py（EMR + API 模型）

| 项目 | 说明 |
|------|------|
| **规模** | 10 个 EMR 用例 + 6 个 Pydantic 模型用例 + 3 个格式化边界用例 |
| **特点** | 唯一测试 EMR 提取和 API 模型验证 |
| **LLM 调用** | TC-01/04/05 需要 |

```bash
python rag-db/tests/test_emr.py
```

---

### test_kg_enrich.py（知识图谱富化）

| 项目 | 说明 |
|------|------|
| **规模** | 疾病知识图谱查询（药品/食物/检查/并发症/治疗） |
| **特点** | 唯一测试 KG 富化集成 |
| **LLM 调用** | 无 |

```bash
# 交互模式
python rag-db/tests/test_kg_enrich.py --interactive

# 批量模式
python rag-db/tests/test_kg_enrich.py --all

# JSON 输出
python rag-db/tests/test_kg_enrich.py --json
```

---

### benchmark_metrics.py（性能指标采集）

| 项目 | 说明 |
|------|------|
| **规模** | 3 项基准：RAG 延迟 (P50/P95/P99)、Reranker Recall@K/MRR、测试覆盖率 |
| **特点** | 唯一做统计级延迟分析和召回率评估 |
| **LLM 调用** | 无 |
| **输出** | `test_results/benchmark_metrics.json` |

```bash
python rag-db/tests/benchmark_metrics.py
```

---

### diagnose_gpu.py（GPU 诊断）

| 项目 | 说明 |
|------|------|
| **规模** | 3 项检查：PyTorch/CUDA 环境、BGE-M3 编码速度、Reranker 推理速度 |
| **特点** | 一次性诊断工具，非持续运行的测试 |

```bash
python rag-db/tests/diagnose_gpu.py
```

---

### health_summary/test_health_summary.py（健康档案 AI 摘要）

| 项目 | 说明 |
|------|------|
| **用例数** | 14 个用例, 52 个检查项 |
| **覆盖** | 完整/最小/空档案、心血管、过敏、儿科、家庭成员、RAG 验证、格式验证、性能、Config 集成 |

```bash
python rag-db/tests/health_summary/test_health_summary.py
```

---

### health_suggestion/test_health_suggestion.py（个性化生活建议）

| 项目 | 说明 |
|------|------|
| **用例数** | 18 个用例, 97 个检查项 |
| **覆盖** | 双表输入/最小/空/儿科/心血管/RAG/RAG 降级/JSON 结构/用药/解析容错/Config/性能/MySQL 写入/MySQL 回读/MySQL 覆盖/清理 |

```bash
python rag-db/tests/health_suggestion/test_health_suggestion.py
```

---

## 运行全部测试

```bash
cd "d:/medic project"

# 核心流水线
python rag-db/tests/test_runner.py
python rag-db/tests/test_comprehensive_10.py
python rag-db/tests/test_reranker_comparison.py
python rag-db/tests/full_pipeline_test.py

# 功能模块
python rag-db/tests/test_emr.py
python rag-db/tests/test_kg_enrich.py

# 健康模块
python rag-db/tests/health_summary/test_health_summary.py
python rag-db/tests/health_suggestion/test_health_suggestion.py

# 性能采集
python rag-db/tests/benchmark_metrics.py
```

---

## 测试分类

| 类型 | 文件 | 需要 LLM |
|------|------|----------|
| **本地单元测试** | test_reranker_comparison, test_kg_enrich, benchmark_metrics, diagnose_gpu | ❌ |
| **部分需要 LLM** | test_runner (B/D 组), test_comprehensive_10 (TC-01/05), full_pipeline_test (阶段 6/7), test_emr (TC-01/04/05) | 部分 |
| **全部需要 LLM** | health_summary, health_suggestion | ✅ |
