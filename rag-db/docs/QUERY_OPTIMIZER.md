# QueryOptimizer — 查询优化器 使用与维护文档

## 一、概述

### 1.1 解决的问题

RAG 系统的核心是**语义向量匹配**。当用户输入口语化或方言表达时（如"肚子疼拉稀想吐"），直接向量化这些文本与知识库中用标准医学术语（"腹痛、腹泻、恶心"）构建的向量进行匹配，会出现**语义鸿沟**：

```
口语化输入: "肚子疼拉稀想吐没胃口"
    ↓ BGE-M3 向量化
  [0.08, -0.12, 0.34, ...]
    ↓ 与知识库向量做余弦相似度
  知识库中 "腹痛 腹泻 恶心 食欲不振" 的向量
    ↓
相似度: ~65% ← 偏低！因为口语词汇和标准术语的向量距离较远

标准化后: "腹痛 腹泻 恶心 食欲不振"
    ↓ BGE-M3 向量化
  [0.09, -0.11, 0.33, ...]
    ↓
相似度: ~78% ← 明显提升！
```

`QueryOptimizer` 在检索前将用户的口语化/方言表达转化为标准医学术语，缩小语义鸿沟，提升检索准确率。

### 1.2 在系统中的位置

```
┌──────────────────────────────────────────────────────────────┐
│                    RAGPipeline.query()                        │
│                                                               │
│  用户输入 ("肚子疼拉稀想吐")                                   │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────┐                                      │
│  │  QueryOptimizer     │  ← 本模块                           │
│  │  .optimize()        │                                      │
│  └─────────┬───────────┘                                      │
│            │ optimized_query: "腹痛 腹泻 恶心"                 │
│            ▼                                                  │
│  ┌─────────────────────┐                                      │
│  │  VectorStore        │                                      │
│  │  .search_disease()  │  用标准化后的query检索               │
│  └─────────┬───────────┘                                      │
│            │ rag_results: [...]                                │
│            ▼                                                  │
│  ┌─────────────────────┐                                      │
│  │  DeepSeekClient     │                                      │
│  │  .recommend()       │  基于检索结果+原始输入生成推荐       │
│  └─────────┬───────────┘                                      │
│            │                                                  │
│            ▼                                                  │
│  输出: {科室, 疾病, 置信度, 推理链, 查询优化信息}              │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、核心设计

### 2.1 双模式架构

| 模式 | `mode` 值 | 原理 | 优点 | 缺点 |
|------|-----------|------|------|------|
| **LLM模式** | `"llm"` | 调用 DeepSeek API 进行智能标准化 | 效果好、能处理复杂语境 | 需要API Key、有延迟、有费用 |
| **规则模式** | `"rule"` | 基于内置词典（600+ 条口语→标准映射） | 免费、快速、离线可用 | 词典覆盖有限 |
| **混合模式** | `"hybrid"` | LLM优先，失败自动降级到规则 | 兼顾效果和可用性 | — |

### 2.2 规则词典结构

词典位于 `COLLOQUIAL_MAP` 字典，按医学系统分类组织：

```python
COLLOQUIAL_MAP: dict[str, list[str]] = {
    # 标准术语        口语化/方言表达列表
    "腹痛":           ["肚子疼", "肚疼", "胃疼", "绞肚", ...],
    "腹泻":           ["拉肚子", "拉稀", "跑肚", "闹肚子", ...],
    "恶心":           ["想吐", "反胃", "干哕", "犯恶心", ...],
    # ... 600+ 条映射
}
```

启动时自动构建反向索引 `_COLLOQUIAL_TO_STANDARD` 实现 O(1) 查找：

```python
_COLLOQUIAL_TO_STANDARD = {
    "肚子疼": "腹痛",
    "拉肚子": "腹泻",
    "想吐":   "恶心",
    # ...
}
```

### 2.3 匹配算法（规则模式）

```
输入: "最近一周肚子老是隐隐作痛还拉稀想吐吃不下饭"

Step 1: 按口语表达长度降序遍历词典
        "吃不下饭"(4字) → 匹配! → ("食欲不振", pos=16)
        "隐隐作痛"(4字) → 无匹配
        "拉稀"(2字)     → 匹配! → ("腹泻", pos=11)
        "想吐"(2字)     → 匹配! → ("恶心", pos=13)
        "肚子疼"(3字)   → 匹配! → ("腹痛", pos=4)

Step 2: 按位置排序，去除重叠匹配
        [(4,7,"腹痛"), (11,13,"腹泻"), (13,15,"恶心"), (16,20,"食欲不振")]

Step 3: 提取标准术语 + 去重
        symptoms = ["腹痛", "腹泻", "恶心", "食欲不振"]

Step 4: 构建优化查询
        optimized_query = "腹痛 腹泻 恶心 食欲不振"

Step 5: 推断身体部位
        body_parts = ["腹部", "消化系统"]
```

### 2.4 缓存设计

为避免相同查询反复调用 LLM，内置内存 LRU 缓存：

- 缓存键：`MD5(raw_query)`
- 最大容量：512 条
- 淘汰策略：超过容量时淘汰最早条目
- 可通过 `optimizer.clear_cache()` 手动清空

---

## 三、使用指南

### 3.1 快速开始

#### 方式一：独立使用 QueryOptimizer

```python
from query_optimizer import QueryOptimizer

# 初始化（自动检测 LLM 可用性）
optimizer = QueryOptimizer(mode="hybrid")

# 优化查询
result = optimizer.optimize("肚子疼拉稀想吐没胃口")
print(result["optimized_query"])   # "腹痛 腹泻 恶心 食欲不振"
print(result["symptoms"])          # ["腹痛", "腹泻", "恶心", "食欲不振"]
print(result["body_parts"])        # ["腹部", "消化系统"]
print(result["normalization_note"]) # "LLM标准化"
```

#### 方式二：通过 RAGPipeline（推荐）

```python
from deepseek_client import RAGPipeline

# Pipeline 默认启用混合模式查询优化
pipeline = RAGPipeline(optimizer_mode="hybrid")

# 完整查询（优化→检索→推荐）
result = pipeline.query("肚子疼拉稀想吐")

# 查看优化详情
opt = result["query_optimization"]
print(f"原始: {opt['original_query']}")
print(f"优化: {opt['optimized_query']}")
print(f"提取症状: {opt['symptoms']}")

# 查看推荐结果
rec = result["recommendation"]
print(f"科室: {rec['department']}")
print(f"疾病: {rec['disease']}")
```

#### 方式三：仅优化不检索

```python
pipeline = RAGPipeline()
opt_result = pipeline.optimize_query("心慌胸闷气短")
# 前端可先展示标准化的症状让用户确认，再执行完整检索
```

### 3.2 配置选项

```python
# 场景1: 生产环境 — 混合模式 + 缓存
optimizer = QueryOptimizer(
    mode="hybrid",       # LLM优先，规则兜底
    cache_enabled=True,  # 启用缓存
    verbose=False,       # 不打印日志
)

# 场景2: 离线/脱网环境 — 纯规则模式
optimizer = QueryOptimizer(
    mode="rule",         # 仅用规则词典
    cache_enabled=True,
)

# 场景3: 高精度需求 — 纯LLM模式
optimizer = QueryOptimizer(
    mode="llm",          # 仅LLM，失败则抛异常
)

# 场景4: 向后兼容 — 禁用优化
pipeline = RAGPipeline(optimizer_mode=None)
# 此时 pipeline.query() 直接使用原始输入检索（与旧版本行为一致）
```

### 3.3 返回数据结构

```python
{
    # === 查询文本 ===
    "original_query": "肚子疼拉稀想吐没胃口",  # 用户原始输入 (保留)
    "optimized_query": "腹痛 腹泻 恶心 食欲不振",  # 标准化后 (用于检索)

    # === 结构化提取 ===
    "symptoms": ["腹痛", "腹泻", "恶心", "食欲不振"],  # 症状列表
    "body_parts": ["腹部", "消化系统"],  # 涉及的身体部位
    "duration": "未知",               # 持续时长
    "severity": "未知",               # 严重程度: 轻/中/重/未知

    # === 元信息 ===
    "keywords": ["腹痛", "腹泻", "恶心", "食欲不振"],
    "normalization_note": "LLM标准化",  # 标准化方式说明
    "has_emergency_signals": False,    # 是否包含危急信号
    "latency_ms": 123.4,              # 优化耗时（毫秒）
}
```

---

## 四、维护与扩展

### 4.1 添加新的口语化/方言映射

当发现新的口语表达未被覆盖时，有三种方式添加：

#### 方式一：代码中动态添加（热更新，无需重启）

```python
from query_optimizer import get_optimizer

optimizer = get_optimizer()

# 单条添加
optimizer.add_colloquial_term("标准术语", "口语表达")
# 例：optimizer.add_colloquial_term("头痛", "脑壳疼")

# 批量添加
optimizer.batch_add_terms([
    ("头痛", "脑壳疼"),
    ("头痛", "偏头疼"),
    ("腹泻", "拉肚子"),
])
```

#### 方式二：编辑词典文件（推荐，持久化）

编辑 `query_optimizer.py`，找到 `COLLOQUIAL_MAP` 字典，在对应分类下添加：

```python
COLLOQUIAL_MAP: dict[str, list[str]] = {
    # ...
    "头痛": [
        "头疼", "脑壳疼", "偏头疼", "头胀", "头重",
        "脑袋疼", "太阳穴疼", "后脑勺疼", "头不舒服",
        # ↓ 新增的表达
        "脑仁疼",      # 北京方言
        "脑瓜子疼",    # 东北方言
        "头壳疼",      # 闽南方言
    ],
    # ...
}
```

**维护建议**：
- 定期分析未命中日志，补充高频未覆盖的口语表达
- 按系统分类（消化道、呼吸道、心血管...）组织，方便查找
- 新增表达后运行 `python rag-db/test_rag.py` 验证

#### 方式三：LLM 自动发现（高级）

对于 LLM 模式下新发现的口语表达，LLM 会自动处理。你可以将 LLM 标准化的结果作为参考，反向补充到规则词典中：

```python
# 记录 LLM 的标准化结果
result = optimizer.optimize("新发现的口语表达")
# 如果 result["normalization_note"] == "LLM标准化"
# 且标准化结果准确，手动添加到 COLLOQUIAL_MAP
```

### 4.2 调整缓存策略

修改 `query_optimizer.py` 中的配置常量：

```python
MAX_CACHE_SIZE = 512  # 增大可提高命中率，但增加内存占用
                      # 建议值: 128(小规模) ~ 2048(大规模)
```

### 4.3 切换 LLM 模型

```python
# 使用 DeepSeek-R1 (更强的推理能力，但更贵更慢)
optimizer = QueryOptimizer(mode="llm", model="deepseek-reasoner")

# 使用其他兼容 OpenAI SDK 的模型
from openai import OpenAI
optimizer = QueryOptimizer(mode="hybrid")
# 自定义 client 需要在初始化后设置:
# optimizer._llm_client = OpenAI(base_url="...", api_key="...")
# optimizer._llm_model = "your-model"
```

### 4.4 性能调优

| 场景 | 建议配置 | 原因 |
|------|---------|------|
| 低延迟要求 (<5ms) | `mode="rule"` | 无网络调用开销 |
| 高准确率要求 | `mode="hybrid"` | LLM智能标准化 |
| 离线环境 | `mode="rule"` | 不依赖外部API |
| 方言多样化环境 | `mode="hybrid"` | LLM能处理词典未覆盖的方言 |
| 开发/测试 | `mode="rule"` | 无费用、可重复 |

### 4.5 查看运行状态

```python
optimizer = get_optimizer()

# 查看缓存情况
print(optimizer.get_cache_stats())
# {"size": 42, "max_size": 512, "enabled": true, "mode": "hybrid"}

# 查看词典覆盖
print(optimizer.get_dictionary_stats())
# {"standard_terms": 85, "colloquial_entries": 612, "body_part_mappings": 25}

# 清空缓存 (通常在更新词典后执行)
optimizer.clear_cache()
```

---

## 五、与现有系统的集成

### 5.1 在 Spring Boot 后端中使用

如果通过 FastAPI 微服务暴露 RAG 能力：

```python
# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from deepseek_client import RAGPipeline

app = FastAPI()
pipeline = RAGPipeline(optimizer_mode="hybrid")

class SearchRequest(BaseModel):
    query: str
    optimize: bool = True
    top_k: int = 5

@app.post("/api/rag/search")
def search(req: SearchRequest):
    result = pipeline.query(req.query, top_k=req.top_k, optimize=req.optimize)

    # 提取前端需要的信息
    return {
        "code": 200,
        "data": {
            "original_query": result["query"],
            "normalized_query": result["query_optimization"]["optimized_query"],
            "extracted_symptoms": result["query_optimization"]["symptoms"],
            "recommended_department": result["recommendation"]["department"],
            "possible_disease": result["recommendation"]["disease"],
            "confidence": result["recommendation"]["confidence"],
            "reasoning": result["recommendation"]["reasoning"],
            "suggestion": result["recommendation"]["suggestion"],
            "alternatives": result["recommendation"]["alternative_departments"],
            "emergency_warning": result["recommendation"]["emergency_warning"],
        }
    }

# 单独的查询优化接口（供前端"症状确认"步骤使用）
@app.post("/api/rag/optimize")
def optimize_query(query: str):
    result = pipeline.optimize_query(query)
    return {
        "code": 200,
        "data": result
    }
```

### 5.2 向后兼容性

`RAGPipeline` 完全向后兼容。旧代码无需修改：

```python
# 旧代码 (依然可用)
pipeline = RAGPipeline()
result = pipeline.query("头痛发热咳嗽")
# 结果结构与之前完全一致，只是多了 query_optimization 字段

# 显式禁用优化
pipeline = RAGPipeline(optimizer_mode=None)
result = pipeline.query("头痛发热咳嗽")
# 此时行为与旧版本完全相同
```

---

## 六、测试验证

### 6.1 运行测试

```bash
cd "d:\medic project"
python rag-db/test_rag.py
```

新增测试用例（Part 8）：

| 编号 | 测试内容 |
|------|---------|
| TC-26 | 口语化标准化（6类症状） |
| TC-27 | 方言表达标准化 |
| TC-28 | 标准术语保留 |
| TC-29 | 非医疗输入处理 |
| TC-30 | 空输入处理 |
| TC-31 | 身体部位推断 |
| TC-32 | 缓存有效性 |
| TC-33 | 危急信号检测 |
| TC-34 | 优化器+Pipeline 集成 |
| TC-35 | 词典动态更新 |

### 6.2 独立测试 QueryOptimizer

```bash
cd "d:\medic project"
python rag-db/query_optimizer.py
```

### 6.3 测试完整 RAG Pipeline（含优化）

```bash
cd "d:\medic project"
python rag-db/deepseek_client.py
```

---

## 七、核心指标监控

建议在日志中记录以下指标：

```python
# 查询优化命中率 (有多少查询被成功标准化)
optimizer = get_optimizer()
total = 0
optimized = 0

# ... 在每次查询后:
result = optimizer.optimize(query)
total += 1
if result["optimized_query"] != result["original_query"]:
    optimized += 1

hit_rate = optimized / total  # 目标: >60%
```

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 优化命中率 | >60% | 有多少查询的 optimized_query 与原始不同 |
| 规则词典覆盖 | >80% | 常见口语化表达的词典覆盖率 |
| LLM调用成功率 | >99% | 混合模式下 LLM 调用的成功率 |
| 优化延迟 (rule) | <5ms | 规则模式的优化耗时 |
| 优化延迟 (llm) | <500ms | LLM模式的优化耗时 |
| 缓存命中率 | >30% | 对于重复查询的缓存命中情况 |

---

## 八、故障排查

### 问题1：LLM 模式不可用，如何确认当前使用的模式？

```python
optimizer = get_optimizer()
print(f"当前模式: {optimizer.mode}")  
# 输出 "rule" 说明已自动降级
```

### 问题2：某些口语表达未被标准化

1. 确认当前模式：如果 `mode="rule"`，检查词典是否包含该表达
2. 查看日志：`optimizer = QueryOptimizer(verbose=True)`
3. 添加映射：使用 `optimizer.add_colloquial_term()` 或编辑 `COLLOQUIAL_MAP`

### 问题3：LLM 标准化结果不准确

1. 降低 temperature：修改 `_optimize_with_llm` 中的 `temperature=0.1` 为 `0.0`
2. 优化 system_prompt：修改 `_optimize_with_llm` 中的提示词
3. 切换到更强的模型：`model="deepseek-reasoner"`

### 问题4：缓存导致更新词典后旧结果仍在使用

```python
optimizer.clear_cache()  # 手动清空缓存
```

---

## 九、文件清单

```
rag-db/
├── query_optimizer.py          # 本模块 — 查询优化器
├── deepseek_client.py          # RAGPipeline 集成（已更新）
├── query_engine.py             # VectorStore 检索层（未修改）
├── test_rag.py                 # 测试套件（新增 Part 8）
└── QUERY_OPTIMIZER.md          # 本文档
```

---

## 十、设计决策记录

### 为什么保留原始查询？

优化后的查询用于向量检索，但 LLM 推荐环节传入的是**原始查询**。原因：

1. 保留用户原意，避免标准化过程中的信息丢失
2. LLM 可以综合分析原始表达和检索结果，做出更全面的判断
3. 推理链展示时使用原始输入更自然（"肚子疼 → 急性胃肠炎" 比 "腹痛 → 急性胃肠炎" 更能让用户理解匹配逻辑）

### 为什么混合模式是默认推荐？

| 场景 | 混合模式行为 |
|------|-------------|
| API Key 可用 + 网络正常 | → LLM 标准化（效果好） |
| API Key 不可用 | → 自动降级到规则模式 |
| LLM 调用超时/异常 | → 自动降级到规则模式 |
| 词典已有的常见表达 | → 规则模式秒级返回（走缓存） |

### 为什么不直接用 LLM 做症状提取后跳过向量检索？

保留向量检索环节的原因：

1. **知识库更新无需重新训练**：新增疾病只需 `add_diseases()`
2. **可追溯**：每次推荐的推理链可追溯到具体知识库条目
3. **成本控制**：向量检索几乎零成本，LLM 调用有费用
4. **延迟更低**：向量检索 <10ms，LLM 调用 >200ms
