# Reranker 精排模块 — 检索质量优化

## 概述

Reranker（重排序器）是 RAG 系统的**第二階段检索优化模块**。它在向量检索的粗排结果之上，使用 **cross-encoder（交叉编码器）** 对候选文档进行精细重打分，显著提升最终送入 LLM 的检索质量。

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline（含 Reranker）                  │
│                                                                   │
│  用户输入                                                        │
│    │                                                              │
│    ▼                                                              │
│  ┌──────────────────┐                                            │
│  │  QueryOptimizer  │  口语/方言 → 标准医学术语                    │
│  │  (查询优化器)     │  "肚子疼拉稀" → "腹痛 腹泻"                  │
│  └────────┬─────────┘                                            │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  VectorStore     │  第一階段：Bi-encoder 粗排                   │
│  │  (向量检索)       │  BGE-M3 embedding + Cosine Similarity       │
│  │                  │  从 8,808 条疾病中召回 Top-20 候选            │
│  └────────┬─────────┘                                            │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  Reranker        │  第二階段：Cross-encoder 精排  ← 【本模块】  │
│  │  (重排序器)       │  BGE-Reranker-v2-m3 逐对打分                │
│  │                  │  20 候选 → 重排序 → Top-5                    │
│  └────────┬─────────┘                                            │
│           ▼                                                       │
│  ┌──────────────────┐                                            │
│  │  DeepSeekClient  │  生成層：LLM 推理推荐                        │
│  │  (大模型生成)     │  基于 Top-5 精排结果生成科室推荐              │
│  └──────────────────┘                                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 为什么需要 Reranker

### 当前方案的问题

系统目前使用 **BGE-M3 双编码器（Bi-encoder）** 将查询和文档分别编码为向量，通过余弦相似度排序：

| 对比维度 | Bi-encoder（当前） | Cross-encoder（Reranker） |
|----------|-------------------|--------------------------|
| **编码方式** | Query 和 Document 独立编码 | Query 和 Document 联合编码 |
| **交互深度** | 无交互（向量点积） | 全交互（Attention 逐层交叉） |
| **速度** | 快（可预计算文档向量） | 慢（每对都要重新计算） |
| **精度** | 粗粒度语义匹配 | 细粒度相关性判断 |
| **适用场景** | 大规模初筛（8808 → 20） | 小规模精排（20 → 5） |

**Bi-encoder 的局限性示例：**

```
查询: "心口疼肚子胀"
  Bi-encoder 余弦相似度排序:
    1. 心绞痛          0.72  ← 关键词"心"匹配，但实际描述消化道症状
    2. 消化不良         0.68  ← 正确但排名第二
    3. 心肌梗死         0.65  ← 错误匹配
    ...

  Cross-encoder 精排后:
    1. 消化不良         0.94  ← 正确提升至第一
    2. 胃食管反流        0.87  ← 更精准的匹配
    3. 心绞痛           0.31  ← 被正确降权
    ...
```

### Reranker 的解決方案

**两階段检索（Two-Stage Retrieval）** 是业界标准方案：

1. **粗排（Recall）**：Bi-encoder 从全库快速召回 20 个候选，保证召回率
2. **精排（Precision）**：Cross-encoder 对 20 个候选逐对精细打分，保证准确率

这样既保留了向量检索的速度优势，又获得了接近全量 cross-encoder 的精度。

---

## 技术原理

### Cross-Encoder 架构

```
Bi-encoder（当前）:                    Cross-encoder（Reranker）:
                                       
  Query ──→ [Encoder] ──→ q_vec ─┐      [CLS] Query + Doc [SEP]     
                                   ├─ cos        │                    
  Doc  ──→ [Encoder] ──→ d_vec ─┘             [Transformer]          
                                                Layers × 24          
  • 独立编码，无交互                               │                    
  • 速度: O(1) 向量点积                         [CLS] → Score         
  • 精度: 中等                                                    
                                                 • 联合编码，全交互  
                                                 • 速度: O(n) 逐对计算
                                                 • 精度: 高
```

Cross-encoder 将 Query 和 Document 拼接后一起送入 Transformer，每一层 attention 都能在 query 和 document 之间交叉关注，捕捉到 bi-encoder 无法感知的细粒度语义关系。

### 模型选型：BGE-Reranker-v2-m3

| 属性 | 说明 |
|------|------|
| **模型名称** | `BAAI/bge-reranker-v2-m3` |
| **基础架构** | XLM-RoBERTa（Cross-Encoder） |
| **参数规模** | ~568M（约 1.1 GB） |
| **最大长度** | 8192 tokens |
| **支持语言** | 中文、英文等 100+ 语言 |
| **输出** | 相关性 logit（经过 sigmoid 转换为 0-1 概率） |
| **与 Embedding 的关系** | 与 BGE-M3（Bi-encoder）同属 BGE 系列，训练数据和方法论一致，配合使用效果最优 |

---

## 代码架构

### 文件结构

```
src/
├── reranker.py              ← Reranker 模块（本模块）
├── query_engine.py          ← VectorStore（集成 reranker 调用）
├── deepseek_client.py       ← RAGPipeline（传递 reranker_enabled 配置）
└── download_reranker.py     ← 模型下载脚本
```

### Reranker 类设计

```python
class Reranker:
    """
    Cross-encoder 重排序器

    延迟加载模式: 模型在首次调用时才加载（与 VectorStore 的 embedding 模型一致）
    """

    def rerank(query, documents) -> list[dict]
        """原始 rerank：对任意 (query, documents) 对打分排序"""

    def rerank_results(query, disease_results) -> list[dict]
        """专用于疾病检索结果的重排序：
        1. 从 disease_results 提取结构化文本
        2. Cross-encoder 逐对打分
        3. Sigmoid 归一化到 0-1
        4. 保留原始 cosine_score 字段
        5. 重排序后返回
        """
```

### 与 VectorStore 的集成

```python
# VectorStore 初始化时启用
store = VectorStore(use_reranker=True)

# comprehensive_search 自动使用两階段检索
result = store.comprehensive_search("头痛发热咳嗽", top_k=5)
# 内部流程:
#   1. search_disease(query, top_k=20)     ← 粗排 20 候选
#   2. reranker.rerank_results(query, ...)  ← 精排重打分
#   3. disease_results[:5]                 ← 取 Top-5
#   4. 构建 primary_recommendation          ← 与原逻辑一致

print(result["reranked"])  # True（标识使用了 reranker）
```

### 与 RAGPipeline 的集成

```python
# 默认不启用（向后兼容）
pipeline = RAGPipeline()  # reranker_enabled=False

# 启用 Reranker
pipeline = RAGPipeline(reranker_enabled=True)
result = pipeline.query("肚子疼拉稀想吐")
```

---

## 性能影响

| 指标 | 无 Reranker | 有 Reranker | 变化 |
|------|------------|------------|------|
| **粗排检索数** | 5 | 20 | ↑ 4x |
| **粗排延迟** | ~10ms | ~15ms | +5ms |
| **精排延迟** | — | ~50-100ms (CPU) | 新增 |
| **精排延迟** | — | ~5-10ms (GPU) | 新增 |
| **总延迟增量** | — | +50-100ms | 可接受 |
| **Top-1 准确率** | 基准 | 预期 +5%~15% | 提升 |
| **内存占用** | ~2.2GB (BGE-M3) | +~1.1GB (Reranker) | 共 ~3.3GB |

> **注**：精排延迟基于单次处理 20 个 (query, doc) 对的估计。实际延迟取决于硬件。

### 延迟优化策略

1. **批量处理**：20 个 pair 一次性送入模型，而非逐个处理
2. **FP16 推理**：默认启用半精度，速度提升 ~2x，精度损失可忽略
3. **GPU 推理**：如有 GPU，延迟可降至 5-10ms
4. **可选开关**：`reranker_enabled=False` 即可回退到纯余弦方案，零开销

---

## 使用方式

### 1. 下载模型（一次性）

```bash
cd "d:/medic project"
python rag-db/src/download_reranker.py
```

模型将下载到 `D:\floder-for-claude\medic\bge-reranker-v2-m3`（与 BGE-M3 同目录）。

### 2. 在代码中使用

```python
from src.deepseek_client import RAGPipeline

# 启用 Reranker
pipeline = RAGPipeline(reranker_enabled=True)

# 正常查询，自动使用两階段检索
result = pipeline.query("头痛发热咳嗽流鼻涕")

# 查看结果
print(result["rag_results"]["reranked"])       # True
print(result["recommendation"]["department"])  # 推荐的科室
print(result["recommendation"]["confidence"])  # 置信度
```

### 3. 仅使用 VectorStore（无 LLM）

```python
from src.query_engine import VectorStore

store = VectorStore(use_reranker=True)

# comprehensive_search 自动精排
result = store.comprehensive_search("肚子疼拉稀想吐", top_k=5)

# 查看精排后的疾病列表
for r in result["disease_results"]:
    print(f"{r['disease']}: {r['score']:.1%} (cosine: {r.get('cosine_score', 0):.1%})")
```

### 4. 对比测试（精排 vs 无精排）

```python
# 无精排
store_basic = VectorStore(use_reranker=False)
result_basic = store_basic.comprehensive_search("头痛发热咳嗽", top_k=5)

# 有精排
store_rerank = VectorStore(use_reranker=True)
result_rerank = store_rerank.comprehensive_search("头痛发热咳嗽", top_k=5)

# 对比 Top-1
print("无精排:", result_basic["disease_results"][0]["disease"])
print("有精排:", result_rerank["disease_results"][0]["disease"])
```

### 5. 命令行快速测试

```bash
# 测试 Reranker 独立功能（需要先下载模型）
python rag-db/src/reranker.py
```

---

## 配置参考

| 配置项 | 位置 | 默认值 | 说明 |
|--------|------|--------|------|
| `RERANKER_MODEL_PATH` | `reranker.py` / `query_engine.py` | `D:\floder-for-claude\medic\bge-reranker-v2-m3` | 模型本地路径 |
| `RERANKER_FETCH_K` | `query_engine.py` | `20` | 粗排召回候选数 |
| `use_reranker` | `VectorStore.__init__` | `False` | 是否启用精排 |
| `reranker_enabled` | `RAGPipeline.__init__` | `False` | Pipeline 级别开关 |
| `use_fp16` | `Reranker.__init__` | `True` | 半精度推理 |
| `normalize_scores` | `rerank_results()` | `True` | Sigmoid 归一化到 0-1 |

---

## 故障排查

### 模型未下载

```
FileNotFoundError: Reranker model not found at: D:\floder-for-claude\medic\bge-reranker-v2-m3
```

**解决**：运行 `python rag-db/src/download_reranker.py`

### 显存不足

```
CUDA out of memory
```

**解决**：
- 设置 `use_fp16=True`（默认已启用）
- 或强制使用 CPU：修改 `reranker.py` 中 `CrossEncoder` 加载参数 `device="cpu"`

### 延迟过高

**解决**：
- 减小 `RERANKER_FETCH_K`（如从 20 降到 10）
- 确认 FP16 已启用
- 考虑使用 GPU 推理

---

## 设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| Reranker 模型 | BAAI/bge-reranker-v2-m3 | 与现有 BGE-M3 embedding 同系列，中文效果好 |
| 加载方式 | 延迟加载（lazy-load） | 与 BGE-M3 保持一致，不使用时零开销 |
| 加载库 | sentence-transformers (CrossEncoder) | 已存在于依赖中，无需新增包 |
| 集成位置 | `comprehensive_search()` 内部 | 对上层调用者透明，API 不变 |
| 默认状态 | 关闭（opt-in） | 向后兼容，用户按需启用 |
| 分数归一化 | Sigmoid | 将原始 logit 转为 0-1 概率，与余弦分数分布一致 |
| 原始分数保留 | `cosine_score` 字段 | 方便对比分析和调试 |
| 结果标识 | `reranked: true/false` | 调用方可判断是否经过了精排 |

---

## 相关文档

- [RAG 系统运行原理](DeepSeek_RAG_运行原理.md) — 了解完整 RAG pipeline
- [Query Optimizer 文档](QUERY_OPTIMIZER.md) — 查询优化器（位于 Reranker 上游）
- [BGE-Reranker 官方](https://huggingface.co/BAAI/bge-reranker-v2-m3) — 模型详情
