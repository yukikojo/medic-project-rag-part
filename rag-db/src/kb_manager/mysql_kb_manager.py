"""
mysql_kb_manager.py
MySQL 知识库连接器 + ChromaDB 同步引擎

连接 MySQL 中的 rag_disease 表，支持:
  1. 全量重建 — 从 MySQL 读取全部疾病 → BGE-M3 编码 → ChromaDB 三 Collection 重建
  2. 增量同步 — Java 端修改 MySQL 后，按 ID 增量更新 ChromaDB
  3. 状态查询 — 比较 MySQL vs ChromaDB 数据一致性

MySQL 表结构 (rag_disease):
  CREATE TABLE rag_disease (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    name          VARCHAR(200) NOT NULL COMMENT '疾病名称',
    symptoms      TEXT COMMENT '症状列表 (JSON array)',
    cure_department TEXT COMMENT '科室列表 (JSON array)',
    category      TEXT COMMENT '分类 (JSON array)',
    description   TEXT COMMENT '疾病简介',
    recommand_drug TEXT COMMENT '推荐药品 (JSON array)',
    common_drug   TEXT COMMENT '常用药品 (JSON array)',
    status        TINYINT DEFAULT 1 COMMENT '0=停用 1=启用',
    version       INT DEFAULT 1 COMMENT '版本号',
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
  );

配置方式 (.env):
  MYSQL_HOST=localhost
  MYSQL_PORT=3306
  MYSQL_USER=root
  MYSQL_PASSWORD=xxx
  MYSQL_DATABASE=medical_rag

使用示例:
    from kb_manager.mysql_kb_manager import MySQLKBManager

    mgr = MySQLKBManager()

    # 全量重建 ChromaDB 三 Collection
    result = mgr.rebuild_all()
    # {"disease_knowledge": 8808, "symptom_dept_direct": 4826, "department_info": 54}

    # 增量同步 (Java 修改后调用)
    result = mgr.sync_by_ids(updated_ids=[1,2,3], deleted_ids=[99])
    # {"synced": 3, "deleted": 1, "errors": []}

    # 数据一致性检查
    status = mgr.check_consistency()
    # {"mysql_count": 8808, "chromadb_count": 8808, "consistent": True}
"""

import os
import sys
import json
import time
from typing import Optional
from collections import Counter, defaultdict

from dotenv import load_dotenv as _load_dotenv

# Load .env from project root
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


# ============================================================
# MySQL 连接配置
# ============================================================

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "medical_rag"),
    "charset": "utf8mb4",
}


# ============================================================
# SQL 模板
# ============================================================

SQL_GET_ALL_ACTIVE = """
    SELECT id, name, symptoms, cure_department, category, description,
           recommand_drug, common_drug, version
    FROM rag_disease
    WHERE status = 1
    ORDER BY id
"""

SQL_GET_BY_IDS = """
    SELECT id, name, symptoms, cure_department, category, description,
           recommand_drug, common_drug, version
    FROM rag_disease
    WHERE id IN ({ids}) AND status = 1
"""

SQL_GET_COUNT = """
    SELECT COUNT(*) as cnt FROM rag_disease WHERE status = 1
"""

SQL_GET_MAX_UPDATED = """
    SELECT MAX(updated_at) as last_update, COUNT(*) as total
    FROM rag_disease WHERE status = 1
"""

SQL_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS rag_disease (
        id            BIGINT PRIMARY KEY AUTO_INCREMENT,
        name          VARCHAR(200) NOT NULL COMMENT '疾病名称',
        symptoms      TEXT COMMENT '症状列表 (JSON array)',
        cure_department TEXT COMMENT '科室列表 (JSON array)',
        category      TEXT COMMENT '分类 (JSON array)',
        description   TEXT COMMENT '疾病简介',
        recommand_drug TEXT COMMENT '推荐药品 (JSON array)',
        common_drug   TEXT COMMENT '常用药品 (JSON array)',
        status        TINYINT DEFAULT 1 COMMENT '0=停用 1=启用',
        version       INT DEFAULT 1 COMMENT '版本号，每次修改+1',
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='RAG 疾病知识库源数据'
"""


class MySQLKBManager:
    """
    MySQL ↔ ChromaDB 知识库同步管理器。

    设计原则:
      - MySQL 是源数据 (Source of Truth)，Java 端负责写入
      - ChromaDB 是向量索引 (Derived Data)，Python 端负责构建
      - 同步方向永远是: MySQL → ChromaDB (单向)
      - Java 修改 MySQL 后调用 sync 接口触发增量或全量更新
    """

    def __init__(
        self,
        mysql_config: Optional[dict] = None,
        db_path: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            mysql_config: MySQL 连接参数, 默认从 MYSQL_CONFIG 读取
            db_path: ChromaDB 持久化路径, 默认 ../medical_rag_db/
            verbose: 是否打印进度日志
        """
        self.mysql_config = mysql_config or MYSQL_CONFIG
        self.verbose = verbose

        # ChromaDB path
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "medical_rag_db")
        self.db_path = db_path

        # Lazy-loaded singletons
        self._mysql_conn = None
        self._chroma_client = None
        self._embedding_model = None

    # ================================================================
    # 数据库连接 (lazy)
    # ================================================================

    @property
    def mysql(self):
        """Lazy-load MySQL connection."""
        if self._mysql_conn is None:
            import pymysql
            self._mysql_conn = pymysql.connect(
                host=self.mysql_config["host"],
                port=self.mysql_config["port"],
                user=self.mysql_config["user"],
                password=self.mysql_config["password"],
                database=self.mysql_config["database"],
                charset=self.mysql_config["charset"],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
            )
            if self.verbose:
                print(f"[MySQLKB] MySQL 已连接: {self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
        else:
            # Reconnect if closed
            try:
                self._mysql_conn.ping(reconnect=True)
            except Exception:
                import pymysql
                self._mysql_conn = pymysql.connect(
                    host=self.mysql_config["host"],
                    port=self.mysql_config["port"],
                    user=self.mysql_config["user"],
                    password=self.mysql_config["password"],
                    database=self.mysql_config["database"],
                    charset=self.mysql_config["charset"],
                    cursorclass=pymysql.cursors.DictCursor,
                    autocommit=True,
                )
        return self._mysql_conn

    @property
    def chroma_client(self):
        """Lazy-load ChromaDB PersistentClient."""
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=self.db_path)
        return self._chroma_client

    @property
    def embedding_model(self):
        """Lazy-load BGE-M3 embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            import torch as _torch

            model_path = os.getenv(
                "EMBEDDING_MODEL_PATH",
                r"D:\floder-for-claude\medic\bge-m3"
            )

            if self.verbose:
                print(f"[MySQLKB] 加载嵌入模型: {model_path} ...")

            device = "cuda" if _torch.cuda.is_available() else "cpu"
            model_kwargs = {}
            if device == "cuda":
                model_kwargs = {"torch_dtype": _torch.float16}

            self._embedding_model = SentenceTransformer(
                model_path,
                device=device,
                model_kwargs=model_kwargs,
            )

            if self.verbose:
                print(f"[MySQLKB] 模型加载完成, device={device}")

        return self._embedding_model

    # ================================================================
    # MySQL 读取
    # ================================================================

    def fetch_all_diseases(self) -> list[dict]:
        """从 MySQL 读取全部启用的疾病记录 (status=1)。"""
        with self.mysql.cursor() as cursor:
            cursor.execute(SQL_GET_ALL_ACTIVE)
            rows = cursor.fetchall()
        return rows

    def fetch_diseases_by_ids(self, ids: list[int]) -> list[dict]:
        """按 ID 列表读取疾病记录。"""
        if not ids:
            return []
        placeholders = ", ".join(["%s"] * len(ids))
        sql = SQL_GET_BY_IDS.format(ids=placeholders)
        with self.mysql.cursor() as cursor:
            cursor.execute(sql, ids)
            rows = cursor.fetchall()
        return rows

    def fetch_count(self) -> int:
        """获取启用的疾病总数。"""
        with self.mysql.cursor() as cursor:
            cursor.execute(SQL_GET_COUNT)
            return cursor.fetchone()["cnt"]

    def fetch_stats(self) -> dict:
        """获取 MySQL 端统计信息。"""
        with self.mysql.cursor() as cursor:
            cursor.execute(SQL_GET_MAX_UPDATED)
            stats = cursor.fetchone()
        return {
            "total_active": stats["total"],
            "last_update": str(stats["last_update"]) if stats["last_update"] else None,
        }

    def ensure_table(self) -> bool:
        """确保 rag_disease 表存在 (幂等创建)。"""
        try:
            with self.mysql.cursor() as cursor:
                cursor.execute(SQL_CREATE_TABLE)
            if self.verbose:
                print("[MySQLKB] rag_disease 表已就绪")
            return True
        except Exception as e:
            print(f"[MySQLKB] 创建表失败: {e}")
            return False

    # ================================================================
    # 数据解析 (MySQL row → 标准化结构)
    # ================================================================

    def _parse_disease(self, row: dict) -> dict:
        """
        将 MySQL 行解析为标准化疾病 dict。

        MySQL TEXT 字段存储 JSON array 字符串 (如 '["症状A", "症状B"]')，
        解析为 Python list。兼容 NULL 和已解析的 list。
        """
        def _parse_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    return []
                try:
                    return json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    # Fallback: comma-separated
                    return [v.strip() for v in val.split(",") if v.strip()]
            return []

        return {
            "id": row["id"],
            "name": row["name"] or "",
            "symptom": _parse_list(row.get("symptoms")),
            "cure_department": _parse_list(row.get("cure_department")),
            "category": _parse_list(row.get("category")),
            "desc": row.get("description") or "",
            "recommand_drug": _parse_list(row.get("recommand_drug")),
            "common_drug": _parse_list(row.get("common_drug")),
            "version": row.get("version", 1),
        }

    # ================================================================
    # ChromaDB 构建逻辑 (与 build_knowledge_base.py 相同)
    # ================================================================

    def _build_search_text(self, d: dict) -> str:
        """构建用于向量检索的拼接文本 (与现有格式完全一致)。"""
        symptom_text = "、".join(d["symptom"]) if d["symptom"] else "暂无"
        dept_text = "、".join(d["cure_department"]) if d["cure_department"] else "暂无"
        cat_text = "、".join(d["category"]) if d["category"] else "暂无"
        desc_short = d["desc"][:300] if d["desc"] else "暂无"

        return (
            f"疾病：{d['name']}。"
            f"症状：{symptom_text}。"
            f"所属科室：{dept_text}。"
            f"分类：{cat_text}。"
            f"简介：{desc_short}"
        )

    def _build_metadata(self, d: dict) -> dict:
        """构建 ChromaDB 元数据 dict。"""
        return {
            "disease": d["name"],
            "symptoms": ", ".join(d["symptom"]) if d["symptom"] else "",
            "departments": ", ".join(d["cure_department"]) if d["cure_department"] else "",
            "category": ", ".join(d["category"]) if d["category"] else "",
            "drugs": ", ".join((d.get("recommand_drug") or []) + (d.get("common_drug") or [])),
            "desc": (d["desc"] or "")[:500],
        }

    # ================================================================
    # 全量重建
    # ================================================================

    def rebuild_all(self, batch_size: int = 200) -> dict:
        """
        从 MySQL 全量重建 ChromaDB 三 Collection。

        流程:
          1. 清空旧 Collection
          2. 从 MySQL 读取全部启用疾病
          3. BGE-M3 批量编码 → 写入 disease_knowledge
          4. 聚合症状→科室映射 → 写入 symptom_dept_direct
          5. 聚合科室信息 → 写入 department_info

        Returns:
            {"disease_knowledge": 8808, "symptom_dept_direct": 4826, "department_info": 54}
        """
        total_start = time.time()
        counts = {}

        # ---- 1. 读取 MySQL ----
        if self.verbose:
            print("\n[MySQLKB] ===== 全量重建 ChromaDB =====")
            print("[MySQLKB] [1/3] 从 MySQL 读取疾病数据...")

        rows = self.fetch_all_diseases()
        diseases = [self._parse_disease(r) for r in rows]

        if self.verbose:
            print(f"[MySQLKB]   已读取 {len(diseases)} 条疾病记录")

        if not diseases:
            print("[MySQLKB] [WARN] MySQL 中无数据，跳过重建")
            return {"disease_knowledge": 0, "symptom_dept_direct": 0, "department_info": 0}

        # ---- 2. 重建 disease_knowledge Collection ----
        if self.verbose:
            print(f"\n[MySQLKB] [2/3] 重建 disease_knowledge ({len(diseases)} 条)...")

        # Delete + recreate
        try:
            self.chroma_client.delete_collection("disease_knowledge")
        except Exception:
            pass

        coll1 = self.chroma_client.create_collection(
            name="disease_knowledge",
            metadata={"hnsw:space": "cosine", "description": "疾病知识库 - 症状+科室+描述 (MySQL源)"},
        )

        # Build documents, metadatas, ids
        docs = []
        metas = []
        ids = []
        for d in diseases:
            docs.append(self._build_search_text(d))
            metas.append(self._build_metadata(d))
            ids.append(f"disease_{d['id']:05d}")

        # Batch encode + insert
        embed_start = time.time()
        for i in range(0, len(docs), batch_size):
            batch_end = min(i + batch_size, len(docs))
            batch_docs = docs[i:batch_end]
            batch_metas = metas[i:batch_end]
            batch_ids = ids[i:batch_end]

            embeddings = self.embedding_model.encode(batch_docs, show_progress_bar=False)
            coll1.add(
                embeddings=embeddings.tolist(),
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )

            if self.verbose:
                pct = batch_end / len(docs) * 100
                print(f"  入库进度: {batch_end}/{len(docs)} ({pct:.0f}%)")

        counts["disease_knowledge"] = coll1.count()
        if self.verbose:
            print(f"  [OK] disease_knowledge: {counts['disease_knowledge']} 条, 耗时 {time.time() - embed_start:.1f}s")

        # ---- 3. 重建 symptom_dept_direct ----
        if self.verbose:
            print(f"\n[MySQLKB] [3/3] 重建 symptom_dept_direct + department_info ...")

        # Aggregate symptom → department mapping
        symptom_dept_map = defaultdict(lambda: defaultdict(int))
        symptom_counter = Counter()
        dept_diseases = defaultdict(list)
        dept_symptoms = defaultdict(list)

        for d in diseases:
            for sym in d["symptom"]:
                symptom_counter[sym] += 1
                for dept in d["cure_department"]:
                    symptom_dept_map[sym][dept] += 1
            for dept in d["cure_department"]:
                dept_diseases[dept].append(d["name"])
                dept_symptoms[dept].extend(d["symptom"])

        # symptom_dept_direct
        try:
            self.chroma_client.delete_collection("symptom_dept_direct")
        except Exception:
            pass

        coll2 = self.chroma_client.create_collection(
            name="symptom_dept_direct",
            metadata={"hnsw:space": "cosine", "description": "症状→科室直接映射 (MySQL源)"},
        )

        sym_docs, sym_metas, sym_ids = [], [], []
        idx = 0
        for sym, dept_counts in symptom_dept_map.items():
            if symptom_counter.get(sym, 0) < 2:
                continue  # 过滤仅出现1次的噪声症状
            top_depts = sorted(dept_counts.items(), key=lambda x: -x[1])[:5]
            dept_names = [d for d, _ in top_depts]

            sym_docs.append(f"症状：{sym}。常见关联科室：{'、'.join(dept_names)}。")
            sym_metas.append({
                "symptom": sym,
                "departments": ", ".join(dept_names),
                "disease_count": symptom_counter.get(sym, 0),
            })
            sym_ids.append(f"sym_{idx:04d}")
            idx += 1

        if sym_docs:
            embeddings = self.embedding_model.encode(sym_docs, show_progress_bar=False)
            coll2.add(embeddings=embeddings.tolist(), documents=sym_docs, metadatas=sym_metas, ids=sym_ids)

        counts["symptom_dept_direct"] = coll2.count()

        # department_info
        try:
            self.chroma_client.delete_collection("department_info")
        except Exception:
            pass

        coll3 = self.chroma_client.create_collection(
            name="department_info",
            metadata={"hnsw:space": "cosine", "description": "科室信息库 - 诊疗范围+常见症状 (MySQL源)"},
        )

        dept_docs, dept_metas, dept_ids = [], [], []
        for i, (dept, disease_list) in enumerate(dept_diseases.items()):
            top_symptoms = [s for s, _ in Counter(dept_symptoms[dept]).most_common(10)]
            sample = disease_list[:5] if len(disease_list) > 5 else disease_list

            dept_docs.append(
                f"科室：{dept}。"
                f"诊疗范围：涵盖{len(disease_list)}种疾病，"
                f"包括{'、'.join(sample)}等。"
                f"常见症状：{'、'.join(top_symptoms)}。"
            )
            dept_metas.append({
                "department": dept,
                "disease_count": len(disease_list),
                "common_symptoms": ", ".join(top_symptoms),
                "sample_diseases": ", ".join(sample),
            })
            dept_ids.append(f"dept_{i:03d}")

        if dept_docs:
            embeddings = self.embedding_model.encode(dept_docs, show_progress_bar=False)
            coll3.add(embeddings=embeddings.tolist(), documents=dept_docs, metadatas=dept_metas, ids=dept_ids)

        counts["department_info"] = coll3.count()

        total_elapsed = time.time() - total_start
        if self.verbose:
            print(f"\n[MySQLKB] [OK] 全量重建完成!")
            print(f"  总耗时: {total_elapsed:.1f}s")
            print(f"  disease_knowledge:   {counts['disease_knowledge']}")
            print(f"  symptom_dept_direct: {counts['symptom_dept_direct']}")
            print(f"  department_info:     {counts['department_info']}")

        return counts

    # ================================================================
    # 增量同步
    # ================================================================

    def sync_by_ids(
        self,
        updated_ids: Optional[list[int]] = None,
        deleted_ids: Optional[list[int]] = None,
    ) -> dict:
        """
        增量同步: Java 端修改 MySQL 后，按 ID 列表更新 ChromaDB。

        三 Collection 的处理策略:
          - disease_knowledge: 按 ID 精确 upsert/delete
          - symptom_dept_direct + department_info: 变化后全量重建
            (因为它们是聚合数据，单条疾病变化可能影响多个症状-科室映射)

        Args:
            updated_ids: 有变更的疾病 ID 列表 (INSERT 或 UPDATE)
            deleted_ids: 已删除的疾病 ID 列表 (DELETE 或 status=0)

        Returns:
            {"synced": N, "deleted": M, "errors": []}
        """
        updated_ids = updated_ids or []
        deleted_ids = deleted_ids or []
        errors = []

        if not updated_ids and not deleted_ids:
            return {"synced": 0, "deleted": 0, "errors": []}

        if self.verbose:
            print(f"\n[MySQLKB] ===== 增量同步 =====")
            print(f"  更新: {len(updated_ids)} 条, 删除: {len(deleted_ids)} 条")

        synced = 0
        deleted = 0

        # ---- disease_knowledge: 精确更新 ----
        try:
            coll = self.chroma_client.get_collection("disease_knowledge")
        except Exception:
            # Collection doesn't exist yet → full rebuild
            if self.verbose:
                print("[MySQLKB] Collection 不存在，触发全量重建")
            self.rebuild_all()
            return {"synced": len(updated_ids), "deleted": len(deleted_ids), "errors": errors}

        # Handle deletes
        if deleted_ids:
            chroma_ids = [f"disease_{did:05d}" for did in deleted_ids]
            try:
                coll.delete(ids=chroma_ids)
                deleted = len(deleted_ids)
                if self.verbose:
                    print(f"  已从 ChromaDB 删除 {deleted} 条")
            except Exception as e:
                errors.append(f"delete failed: {e}")

        # Handle updates
        if updated_ids:
            rows = self.fetch_diseases_by_ids(updated_ids)
            if rows:
                docs, metas, ids = [], [], []
                for row in rows:
                    d = self._parse_disease(row)
                    docs.append(self._build_search_text(d))
                    metas.append(self._build_metadata(d))
                    ids.append(f"disease_{d['id']:05d}")

                embeddings = self.embedding_model.encode(docs, show_progress_bar=False)

                try:
                    coll.upsert(
                        embeddings=embeddings.tolist(),
                        documents=docs,
                        metadatas=metas,
                        ids=ids,
                    )
                    synced = len(ids)
                    if self.verbose:
                        print(f"  已更新 ChromaDB {synced} 条")
                except Exception as e:
                    errors.append(f"upsert failed: {e}")
            else:
                if self.verbose:
                    print(f"  [WARN] 未在 MySQL 中找到 ID: {updated_ids} (可能已被删除或停用)")

        # ---- symptom_dept_direct + department_info: 重建 ----
        if updated_ids or deleted_ids:
            try:
                self._rebuild_symptom_dept_and_department_collections()
                if self.verbose:
                    print("  症状映射 + 科室信息 Collection 已重建")
            except Exception as e:
                errors.append(f"rebuild symptom/dept collections failed: {e}")

        return {"synced": synced, "deleted": deleted, "errors": errors}

    def _rebuild_symptom_dept_and_department_collections(self):
        """重新聚合构建 symptom_dept_direct 和 department_info (内部辅助)。"""
        rows = self.fetch_all_diseases()
        diseases = [self._parse_disease(r) for r in rows]

        symptom_dept_map = defaultdict(lambda: defaultdict(int))
        symptom_counter = Counter()
        dept_diseases = defaultdict(list)
        dept_symptoms = defaultdict(list)

        for d in diseases:
            for sym in d["symptom"]:
                symptom_counter[sym] += 1
                for dept in d["cure_department"]:
                    symptom_dept_map[sym][dept] += 1
            for dept in d["cure_department"]:
                dept_diseases[dept].append(d["name"])
                dept_symptoms[dept].extend(d["symptom"])

        # Rebuild symptom_dept_direct
        try:
            self.chroma_client.delete_collection("symptom_dept_direct")
        except Exception:
            pass

        coll2 = self.chroma_client.create_collection(
            name="symptom_dept_direct",
            metadata={"hnsw:space": "cosine", "description": "症状→科室直接映射"},
        )

        sym_docs, sym_metas, sym_ids = [], [], []
        idx = 0
        for sym, dept_counts in symptom_dept_map.items():
            if symptom_counter.get(sym, 0) < 2:
                continue
            top_depts = sorted(dept_counts.items(), key=lambda x: -x[1])[:5]
            dept_names = [d for d, _ in top_depts]
            sym_docs.append(f"症状：{sym}。常见关联科室：{'、'.join(dept_names)}。")
            sym_metas.append({
                "symptom": sym,
                "departments": ", ".join(dept_names),
                "disease_count": symptom_counter.get(sym, 0),
            })
            sym_ids.append(f"sym_{idx:04d}")
            idx += 1

        if sym_docs:
            embeddings = self.embedding_model.encode(sym_docs, show_progress_bar=False)
            coll2.add(embeddings=embeddings.tolist(), documents=sym_docs, metadatas=sym_metas, ids=sym_ids)

        # Rebuild department_info
        try:
            self.chroma_client.delete_collection("department_info")
        except Exception:
            pass

        coll3 = self.chroma_client.create_collection(
            name="department_info",
            metadata={"hnsw:space": "cosine", "description": "科室信息库"},
        )

        dept_docs, dept_metas, dept_ids = [], [], []
        for i, (dept, disease_list) in enumerate(dept_diseases.items()):
            top_symptoms = [s for s, _ in Counter(dept_symptoms[dept]).most_common(10)]
            sample = disease_list[:5] if len(disease_list) > 5 else disease_list
            dept_docs.append(
                f"科室：{dept}。"
                f"诊疗范围：涵盖{len(disease_list)}种疾病，"
                f"包括{'、'.join(sample)}等。"
                f"常见症状：{'、'.join(top_symptoms)}。"
            )
            dept_metas.append({
                "department": dept,
                "disease_count": len(disease_list),
                "common_symptoms": ", ".join(top_symptoms),
                "sample_diseases": ", ".join(sample),
            })
            dept_ids.append(f"dept_{i:03d}")

        if dept_docs:
            embeddings = self.embedding_model.encode(dept_docs, show_progress_bar=False)
            coll3.add(embeddings=embeddings.tolist(), documents=dept_docs, metadatas=dept_metas, ids=dept_ids)

    # ================================================================
    # 数据一致性检查
    # ================================================================

    def check_consistency(self) -> dict:
        """比较 MySQL 与 ChromaDB 数据条数是否一致。"""
        try:
            mysql_count = self.fetch_count()
        except Exception as e:
            return {"error": f"MySQL 查询失败: {e}"}

        try:
            coll = self.chroma_client.get_collection("disease_knowledge")
            chroma_count = coll.count()
        except Exception as e:
            chroma_count = 0

        return {
            "mysql_count": mysql_count,
            "chromadb_count": chroma_count,
            "consistent": mysql_count == chroma_count,
            "delta": mysql_count - chroma_count,
        }

    # ================================================================
    # 数据导入: JSON → MySQL
    # ================================================================

    def import_from_json(self, json_path: str, batch_size: int = 500) -> dict:
        """
        将 medical.json (JSONL) 导入 MySQL rag_disease 表。

        首次初始化时使用: 将现有 JSON 数据入库，之后 MySQL 即为 source of truth。

        Args:
            json_path: medical.json 文件路径
            batch_size: 批量 INSERT 大小

        Returns:
            {"imported": N, "skipped": M, "errors": [...]}
        """
        if not os.path.exists(json_path):
            return {"error": f"文件不存在: {json_path}"}

        # Ensure table exists
        self.ensure_table()

        # Read JSONL
        diseases = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    diseases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if self.verbose:
            print(f"[MySQLKB] 从 JSON 加载了 {len(diseases)} 条记录")

        imported = 0
        skipped = 0
        errors = []

        # Batch INSERT ... ON DUPLICATE KEY UPDATE
        sql = """
            INSERT INTO rag_disease (name, symptoms, cure_department, category,
                                     description, recommand_drug, common_drug, status, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 1, 1)
            ON DUPLICATE KEY UPDATE
                symptoms = VALUES(symptoms),
                cure_department = VALUES(cure_department),
                category = VALUES(category),
                description = VALUES(description),
                recommand_drug = VALUES(recommand_drug),
                common_drug = VALUES(common_drug),
                version = version + 1,
                updated_at = CURRENT_TIMESTAMP
        """

        for i in range(0, len(diseases), batch_size):
            batch = diseases[i:i + batch_size]
            values = []
            for d in batch:
                name = d.get("name", "")
                if not name:
                    skipped += 1
                    continue
                values.append((
                    name,
                    json.dumps(d.get("symptom", []), ensure_ascii=False),
                    json.dumps(d.get("cure_department", []), ensure_ascii=False),
                    json.dumps(d.get("category", []), ensure_ascii=False),
                    d.get("desc", ""),
                    json.dumps(d.get("recommand_drug", []), ensure_ascii=False),
                    json.dumps(d.get("common_drug", []), ensure_ascii=False),
                ))

            try:
                with self.mysql.cursor() as cursor:
                    cursor.executemany(sql, values)
                imported += len(values)
                if self.verbose:
                    pct = min(i + batch_size, len(diseases)) / len(diseases) * 100
                    print(f"  导入进度: {min(i + batch_size, len(diseases))}/{len(diseases)} ({pct:.0f}%)")
            except Exception as e:
                errors.append(f"batch {i // batch_size}: {e}")

        return {"imported": imported, "skipped": skipped, "errors": errors}

    # ================================================================
    # 清理
    # ================================================================

    def close(self):
        """关闭 MySQL 连接。"""
        if self._mysql_conn:
            try:
                self._mysql_conn.close()
            except Exception:
                pass
            self._mysql_conn = None


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  MySQLKB Manager — MySQL ↔ ChromaDB 同步工具")
    print("=" * 65)

    mgr = MySQLKBManager(verbose=True)

    # Test MySQL connection
    try:
        mgr.ensure_table()
        stats = mgr.fetch_stats()
        print(f"\n  MySQL 状态: {stats}")
    except Exception as e:
        print(f"\n  [ERROR] MySQL 连接失败: {e}")
        print(f"  配置: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
        print(f"  请确保 .env 中配置了 MYSQL_* 环境变量")
        sys.exit(1)

    # If MySQL is empty, offer to import from JSON
    if stats["total_active"] == 0:
        print("\n  MySQL rag_disease 表为空。")
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "rag data", "openkg data", "medical.json"
        )
        if os.path.exists(json_path):
            print(f"  是否从 {json_path} 导入数据?")
            print(f"  运行: python mysql_kb_manager.py --import-json")
        else:
            print(f"  请先准备数据文件，或通过 Java 管理后台添加知识库条目。")
    else:
        print(f"\n  MySQL 中已有 {stats['total_active']} 条疾病记录")

        # Check ChromaDB consistency
        cons = mgr.check_consistency()
        print(f"  ChromaDB 一致性: {cons}")

        if not cons.get("consistent"):
            print(f"\n  数据不一致! 运行全量重建:")
            print(f"    python mysql_kb_manager.py --rebuild")
            print(f"  或增量同步:")
            print(f"    python mysql_kb_manager.py --sync --ids 1,2,3")

    mgr.close()
