# MySQL 知识库集成 — 架构与使用说明

> **版本**: v2.0  
> **日期**: 2026-06-25  
> **相关模块**: `mysql_kb_manager.py`, `kg_enricher.py`, `api_server.py`

---

## 一、概述

RAG 系统的源数据由 JSON 文件迁移至 MySQL，Java 管理后台作为数据写入方，Python AI 引擎作为数据读取方。

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流向                                  │
│                                                                  │
│  ┌──────────────┐     写入        ┌──────────────┐              │
│  │  Java Admin   │ ────────────> │    MySQL      │              │
│  │  (管理后台)   │               │  medical_rag  │              │
│  └──────────────┘               └──────┬───────┘              │
│                                        │ 读取                   │
│                               ┌────────▼──────────┐            │
│                               │   Python AI 引擎   │            │
│                               │                    │            │
│                               │ mysql_kb_manager   │ ← 全量/增量│
│                               │ kg_enricher        │ ← 实时查询 │
│                               │        │           │            │
│                               └────────┼───────────┘            │
│                                        │ BGE-M3 编码            │
│                                        ▼                        │
│                               ┌────────────────┐               │
│                               │    ChromaDB     │               │
│                               │  (向量索引)     │               │
│                               └────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**职责划分**:

| 角色 | 组件 | 职责 |
|------|------|------|
| **Writer** | Java Spring Boot | 管理员增删改知识库条目，写入 MySQL |
| **Reader** | Python FastAPI | 从 MySQL 读取数据，编码为向量，写入 ChromaDB |
| **Notification** | Java → Python API | 数据变更后调用 `/api/rag/knowledge/sync` |
| **Query** | Python kg_enricher | 实时从 MySQL `rag_disease_kg` 表查询 KG 富化数据 |

---

## 二、数据库 Schema

### 2.1 数据库: `medical_rag`

```
CREATE DATABASE medical_rag
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

### 2.2 表: `rag_disease` (疾病基础数据)

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGINT | PK AUTO_INCREMENT | 主键 |
| name | VARCHAR(200) | NOT NULL | 疾病名称 |
| symptoms | TEXT | | 症状列表 (JSON array) |
| cure_department | TEXT | | 科室列表 (JSON array) |
| category | TEXT | | 分类 (JSON array) |
| description | TEXT | | 疾病简介 |
| recommand_drug | TEXT | | 推荐药品 (JSON array) |
| common_drug | TEXT | | 常用药品 (JSON array) |
| status | TINYINT | DEFAULT 1 | 0=停用 1=启用 |
| version | INT | DEFAULT 1 | 版本号 (每次修改+1) |
| created_at | DATETIME | | 创建时间 |
| updated_at | DATETIME | | 更新时间 (自动更新) |

**数据来源**: `medical.json` (OpenKG 疾病知识图谱, 8,808 条)  
**存储格式**: symptoms/cure_department/category/drugs 以 JSON array 字符串存储  
**与 ChromaDB 关系**: 全量/增量编码后写入 `disease_knowledge` Collection

**Java 查询示例**:

```sql
-- 管理员搜索疾病
SELECT * FROM rag_disease WHERE name LIKE '%感冒%' AND status = 1;

-- 查看待审核变更 (version > 1)
SELECT name, version, updated_at FROM rag_disease WHERE version > 1 ORDER BY updated_at DESC;
```

---

### 2.3 表: `rag_disease_kg` (知识图谱关联数据)

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGINT | PK AUTO_INCREMENT | 主键 |
| disease_name | VARCHAR(200) | NOT NULL, INDEX | 疾病名称 |
| rel_category | VARCHAR(30) | NOT NULL, INDEX | 关系类别 |
| rel_value | VARCHAR(500) | NOT NULL | 关联值 |
| status | TINYINT | DEFAULT 1 | 0=停用 1=启用 |
| created_at | DATETIME | | 创建时间 |
| updated_at | DATETIME | | 更新时间 (自动更新) |

**索引**:
- `idx_disease` — `(disease_name)` — 按疾病名快速查询
- `idx_category` — `(rel_category)` — 按类别统计
- `idx_disease_cat` — `(disease_name, rel_category)` — 联合查询

**数据来源**: `relations.json` (OpenKG, 12 种关系类型, 231,291 条)

**rel_category 枚举值**:

| 类别 | 中文名 | 示例值 | 条数 |
|------|--------|--------|------|
| `recommand_drug` | 推荐药品 | 伤风停胶囊 | 59,465 |
| `recommand_food` | 推荐食谱 | 凉拌香椿 | 40,221 |
| `need_check` | 建议检查 | 内科检查 | 39,418 |
| `no_eat_food` | 忌吃食物 | 辣椒 | 22,239 |
| `do_eat_food` | 宜吃食物 | 南瓜子仁 | 22,230 |
| `cure_way` | 治疗方法 | 支持性治疗 | 21,047 |
| `common_drug` | 常用药品 | 感冒灵颗粒 | 14,647 |
| `complication` | 并发症 | 支气管炎 | 12,024 |

**与 ChromaDB 关系**: 不写入 ChromaDB。`kg_enricher.py` 实时从 MySQL 查询，返回给 API 调用方。

**Java 查询示例**:

```sql
-- 查询感冒的全部关联数据
SELECT rel_category, rel_value FROM rag_disease_kg
WHERE disease_name = '感冒' AND status = 1
ORDER BY rel_category;

-- 查询所有疾病的推荐药品 (带疾病名)
SELECT disease_name, rel_value FROM rag_disease_kg
WHERE rel_category = 'recommand_drug' AND status = 1
ORDER BY disease_name;

-- 管理员停用某条关联
UPDATE rag_disease_kg SET status = 0
WHERE disease_name = '感冒' AND rel_category = 'recommand_drug' AND rel_value = '过时药品名';

-- 管理员新增关联
INSERT INTO rag_disease_kg (disease_name, rel_category, rel_value)
VALUES ('感冒', 'recommand_drug', '新推荐药品');
```

---

## 三、Python 模块说明

### 3.1 `mysql_kb_manager.py` — MySQL ↔ ChromaDB 同步引擎

**文件**: [rag-db/src/mysql_kb_manager.py](../src/mysql_kb_manager.py)

| 方法 | 功能 | 触发时机 |
|------|------|---------|
| `ensure_table()` | 自动建表 (幂等) | 服务启动 |
| `import_from_json(path)` | JSON → MySQL 首次导入 | 部署初始化 |
| `rebuild_all()` | MySQL → ChromaDB 全量重建 | 管理后台"重建知识库" |
| `sync_by_ids(updated, deleted)` | 增量同步: 按 ID 更新/删除 | Java 修改后自动调用 |
| `check_consistency()` | MySQL vs ChromaDB 一致性校验 | 定时巡检或管理后台 |
| `fetch_all_diseases()` | 读取全部启用疾病 | rebuild 内部调用 |
| `fetch_diseases_by_ids(ids)` | 按 ID 读取 | sync 内部调用 |

**设计要点**:
- 同步方向永远是 **MySQL → ChromaDB** (单向)
- `sync_by_ids` 更新 `disease_knowledge` 后，自动重建 `symptom_dept_direct` 和 `department_info` (因为它们是聚合数据)
- 全量重建时间 ≈ 106s (BGE-M3 on CUDA, 8,808 条)

---

### 3.2 `kg_enricher.py` — 知识图谱富化引擎

**文件**: [rag-db/src/kg_enricher.py](../src/kg_enricher.py)

**数据源优先级**: MySQL `rag_disease_kg` 表 → JSON `relations.json` (fallback)

| 场景 | 数据源 | 备注 |
|------|--------|------|
| MySQL 可用 + 精确匹配 | `rag_disease_kg` 表 | `source: "mysql"` |
| MySQL 可用 + 模糊匹配 | MySQL LIKE 查询 → JSON fallback | |
| MySQL 不可用 | JSON `relations.json` 内存索引 | `source: "json"`, 启动耗时 0.5s |

**MySQL 查询 SQL** (精确匹配):

```sql
SELECT rel_category, rel_value
FROM rag_disease_kg
WHERE disease_name = '感冒' AND status = 1
ORDER BY rel_category
```

**关键特性**:
- MySQL 可用时 **不加载 JSON 到内存** (节省 0.5GB RAM + 0.5s 启动时间)
- 管理员修改 KG 数据后 **立即生效** (无需重建 ChromaDB)
- 8 类关联数据自动按类别分组 → 结构化 JSON 返回

---

## 四、API 端点

### 4.1 知识库同步

| 端点 | 方法 | 请求体 | 说明 |
|------|------|--------|------|
| `/api/rag/knowledge/rebuild` | POST | `{}` | 全量重建 ChromaDB |
| `/api/rag/knowledge/sync` | POST | `{"updated_ids":[1,2],"deleted_ids":[99]}` | 增量同步 |
| `/api/rag/knowledge/status` | GET | — | MySQL vs ChromaDB 一致性 |
| `/api/rag/knowledge/import-json` | POST | `{"json_path":"..."}` | JSON → MySQL 导入 |

### 4.2 KG 富化查询 (实时读 MySQL)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/rag/search/enriched` | POST | 检索 + MySQL KG 实时富化 |

### 4.3 Java 端调用流程

```java
// === 场景1: 管理员新增一条疾病 ===
// Step 1: Java 写入 MySQL
ragDiseaseMapper.insert(disease);      // INSERT INTO rag_disease
ragDiseaseKgMapper.batchInsert(rels);  // INSERT INTO rag_disease_kg (药品/食谱/...)

// Step 2: 通知 Python 增量同步 (仅 disease_knowledge 需要)
Map<String, Object> req = Map.of("updated_ids", List.of(disease.getId()));
restTemplate.postForObject(
    "http://localhost:8000/api/rag/knowledge/sync", req, Map.class);
// KG 数据 (rag_disease_kg) 不需要 sync — 实时从 MySQL 读取

// === 场景2: 管理员修改推荐药品 ===
// Step 1: Java 直接 UPDATE MySQL
kgMapper.update("UPDATE rag_disease_kg SET status=0 WHERE id=?");
kgMapper.insert("INSERT INTO rag_disease_kg (...) VALUES (...)");

// Step 2: 无需通知 Python — kg_enricher 实时查询 MySQL
```

---

## 五、环境配置

`.env` 文件中的 MySQL 配置:

```env
# ============================================================
# MySQL 数据库 (知识库源数据)
# ============================================================
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=qwe123456
MYSQL_DATABASE=medical_rag
```

---

## 六、首次部署步骤

```bash
# 1. 安装依赖
pip install pymysql

# 2. 配置 .env (MYSQL_* 变量)

# 3. 创建数据库 + 导入 JSON → MySQL
python -c "
import sys; sys.path.insert(0,'rag-db/src')
from mysql_kb_manager import MySQLKBManager
mgr = MySQLKBManager(verbose=True)
mgr.ensure_table()
mgr.import_from_json('rag data/openkg data/medical.json')
print(f'MySQL count: {mgr.fetch_count()}')
mgr.close()
"

# 4. 导入知识图谱关系到 MySQL
# (使用 _import_kg_to_mysql.py 脚本，或通过 API)

# 5. 全量构建 ChromaDB
curl -X POST http://localhost:8000/api/rag/knowledge/rebuild

# 6. 验证一致性
curl http://localhost:8000/api/rag/knowledge/status
# → {"mysql_count": 8808, "chromadb_count": 8808, "consistent": true}
```

---

## 七、日常运维

### 7.1 Java 管理员修改知识的正确流程

```
1. Java 开启事务
2. UPDATE/INSERT/DELETE rag_disease 或 rag_disease_kg
3. 提交事务
4. 如果修改了 rag_disease → POST /api/rag/knowledge/sync
   如果只修改了 rag_disease_kg → 无需通知 (实时生效)
```

### 7.2 数据一致性巡检

```java
@Scheduled(fixedRate = 3600000)  // 每小时
public void checkConsistency() {
    Map result = restTemplate.getForObject(
        "http://localhost:8000/api/rag/knowledge/status", Map.class);
    Map consistency = (Map) result.get("data").get("consistency");
    if (!(boolean) consistency.get("consistent")) {
        log.warn("Knowledge DB inconsistent! delta={}", consistency.get("delta"));
        // 可自动触发 rebuild
    }
}
```

### 7.3 从 JSON 恢复 MySQL

```bash
# 如果 MySQL 数据丢失，从 medical.json 恢复
curl -X POST http://localhost:8000/api/rag/knowledge/import-json
```

---

## 八、故障处理

| 故障 | 现象 | 处理 |
|------|------|------|
| MySQL 不可用 | `kg_enricher` 自动回退 JSON 索引 | 无需处理，系统正常降级 |
| MySQL 不可用 | `rebuild` / `sync` 报 500 | 检查 MySQL 服务状态 |
| 数据不一致 | `status` 返回 `consistent: false` | 执行 `rebuild` |
| 增量同步失败 | `sync` 返回 errors | 检查 ID 是否在 MySQL 中存在 |
| 首次导入慢 | import_from_json 耗时 10-30s | 正常 (INSERT 8,808 条) |

---

## 九、表关系总览

```
rag_disease (8,808 rows)              rag_disease_kg (231,291 rows)
┌──────────────────────┐              ┌──────────────────────────┐
│ id (PK)              │              │ id (PK)                  │
│ name                 │◄─────────────│ disease_name             │
│ symptoms (JSON)      │  通过疾病名   │ rel_category             │
│ cure_department      │  关联        │ rel_value                │
│ category             │  (非外键，    │ status                   │
│ description          │   字符串匹配) │ created_at               │
│ recommand_drug       │              │ updated_at               │
│ common_drug          │              └──────────────────────────┘
│ status               │
│ version              │              ChromaDB (向量索引)
│ created_at           │              ┌──────────────────────────┐
│ updated_at           │──────────────│ disease_knowledge (8808) │
└──────────────────────┘  全量/增量    │ symptom_dept_direct      │
                          编码同步     │ department_info (54)     │
                                       └──────────────────────────┘
```

**关键设计决策**: `rag_disease` 与 `rag_disease_kg` 通过 `disease_name` 字符串关联而非外键。因为:
1. 原始 OpenKG 数据中疾病名是自然键 (natural key)
2. 知识图谱关系数量大 (231K)，外键约束会拖慢批量导入
3. 疾病名在 OpenKG 中具有唯一性
