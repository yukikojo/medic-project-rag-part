"""
build_knowledge_base.py
一键构建 RAG 向量知识库
数据源: OpenKG 疾病知识图谱 (medical.json)
输出: ChromaDB 持久化目录 (medical_rag_db/)

Collection 设计:
  1. disease_knowledge  — 主知识库，8808 条疾病（症状+科室+描述）
  2. symptom_dept_direct — 症状→科室直接映射，约 2000 条高频症状
  3. department_info     — 科室信息库，54 个科室的诊疗范围
"""

import json
import time
import os
import sys
from collections import Counter, defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv as _load_dotenv

# Load .env from project root
_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

# ============================================================
# 配置区
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rag data", "openkg data", "medical.json")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "medical_rag_db")

# Embedding model — local path, configured via .env → EMBEDDING_MODEL_PATH
# BAAI/bge-m3: 1024-dim, multilingual, MTR hybrid retrieval, 8192 tokens, ~2.2GB
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_PATH",
    r"D:\floder-for-claude\medic\bge-m3"
)

# Batch size
BATCH_SIZE = 200

# ============================================================
# Step 1: 加载数据
# ============================================================
def load_diseases(data_path: str) -> list[dict]:
    """加载 medical.json (JSONL 格式)"""
    diseases = []
    print(f"[1/4] 加载数据: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                diseases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  警告: 跳过一行解析失败的数据: {e}")
    print(f"  共加载 {len(diseases)} 条疾病记录")
    return diseases


def analyze_data(diseases: list[dict]) -> dict:
    """分析数据质量并打印统计"""
    empty_symptom = sum(1 for d in diseases if not d.get("symptom"))
    empty_dept = sum(1 for d in diseases if not d.get("cure_department"))
    empty_desc = sum(1 for d in diseases if not d.get("desc"))

    all_symptoms = []
    all_depts = []
    for d in diseases:
        if d.get("symptom"):
            all_symptoms.extend(d["symptom"])
        if d.get("cure_department"):
            all_depts.extend(d["cure_department"])

    dept_counter = Counter(all_depts)
    symptom_counter = Counter(all_symptoms)

    stats = {
        "total": len(diseases),
        "empty_symptom": empty_symptom,
        "empty_dept": empty_dept,
        "empty_desc": empty_desc,
        "unique_depts": len(dept_counter),
        "unique_symptoms": len(symptom_counter),
        "dept_counter": dept_counter,
        "symptom_counter": symptom_counter,
    }

    print(f"  数据质量:")
    print(f"    无症状字段: {empty_symptom}")
    print(f"    无科室字段: {empty_dept}")
    print(f"    无描述字段: {empty_desc}")
    print(f"    独立科室数: {len(dept_counter)}")
    print(f"    独立症状数: {len(symptom_counter)}")
    return stats


# ============================================================
# Step 2: 准备 Collection 数据
# ============================================================
def prepare_disease_knowledge(diseases: list[dict]) -> tuple:
    """准备主知识库数据: 疾病名+症状+简介 → 向量, 元数据含科室"""
    documents = []
    metadatas = []
    ids = []

    for i, d in enumerate(diseases):
        name = d.get("name", "")
        symptoms = d.get("symptom", [])
        departments = d.get("cure_department", [])
        category = d.get("category", [])
        desc = d.get("desc", "")
        drugs = d.get("recommand_drug", []) or d.get("common_drug", [])

        # 构建用于向量检索的文本
        symptom_text = "、".join(symptoms) if symptoms else "暂无"
        dept_text = "、".join(departments) if departments else "暂无"
        cat_text = "、".join(category) if category else "暂无"
        desc_short = desc[:300] if desc else "暂无"

        search_text = (
            f"疾病：{name}。"
            f"症状：{symptom_text}。"
            f"所属科室：{dept_text}。"
            f"分类：{cat_text}。"
            f"简介：{desc_short}"
        )

        documents.append(search_text)
        metadatas.append({
            "disease": name,
            "symptoms": ", ".join(symptoms) if symptoms else "",
            "departments": ", ".join(departments) if departments else "",
            "category": ", ".join(category) if category else "",
            "drugs": ", ".join(drugs) if drugs else "",
            "desc": desc[:500] if desc else "",
        })
        ids.append(f"disease_{i:04d}")

    return documents, metadatas, ids


def prepare_symptom_dept_mapping(stats: dict) -> tuple:
    """
    准备症状→科室直接映射数据
    从疾病数据中聚合每个症状最常关联的科室
    """
    symptom_dept_map = defaultdict(lambda: defaultdict(int))

    # 已在 stats 中有 symptom_counter
    # 需要重建: 症状 → {科室: 出现次数}
    dept_counter = stats["dept_counter"]
    symptom_counter = stats["symptom_counter"]

    # 重新遍历疾病数据构建映射
    # 这里用一个新函数, 后面会在 build_all 里调用
    return None  # 占位, 实际在 build_all 里构建


def prepare_department_info(diseases: list[dict], stats: dict) -> tuple:
    """准备科室信息库数据"""
    dept_diseases = defaultdict(list)
    dept_symptoms = defaultdict(list)

    for d in diseases:
        departments = d.get("cure_department", [])
        symptoms = d.get("symptom", [])
        name = d.get("name", "")

        for dept in departments:
            dept_diseases[dept].append(name)
            dept_symptoms[dept].extend(symptoms)

    documents = []
    metadatas = []
    ids = []

    for i, (dept, disease_list) in enumerate(dept_diseases.items()):
        # 找到该科室最常见的症状
        symptom_freq = Counter(dept_symptoms[dept])
        top_symptoms = [s for s, _ in symptom_freq.most_common(10)]

        diseases_sample = disease_list[:5] if len(disease_list) > 5 else disease_list
        disease_count = len(disease_list)

        search_text = (
            f"科室：{dept}。"
            f"诊疗范围：涵盖{disease_count}种疾病，"
            f"包括{'、'.join(diseases_sample)}等。"
            f"常见症状：{'、'.join(top_symptoms)}。"
        )

        documents.append(search_text)
        metadatas.append({
            "department": dept,
            "disease_count": disease_count,
            "common_symptoms": ", ".join(top_symptoms),
            "sample_diseases": ", ".join(diseases_sample),
        })
        ids.append(f"dept_{i:03d}")

    return documents, metadatas, ids


# ============================================================
# Step 3: 构建所有 Collection
# ============================================================
def build_all():
    """主构建流程"""
    total_start = time.time()

    # 加载数据
    diseases = load_diseases(DATA_PATH)
    stats = analyze_data(diseases)

    # ---- 初始化模型 ----
    print(f"\n[2/4] 加载嵌入模型: {EMBEDDING_MODEL}")
    print(f"  (从本地加载, 无需联网)")
    model_start = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  模型加载完成, 耗时 {time.time() - model_start:.1f}s")

    # ---- 初始化 ChromaDB ----
    print(f"\n[3/4] 初始化 ChromaDB, 存储路径: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    # ---- Collection 1: 主知识库 ----
    print("\n  构建 Collection 1/3: disease_knowledge")
    docs1, metas1, ids1 = prepare_disease_knowledge(diseases)
    print(f"    共 {len(docs1)} 条记录, 开始向量化...")

    # 删除旧 collection 重建 (幂等)
    try:
        client.delete_collection("disease_knowledge")
    except Exception:
        pass

    coll1 = client.create_collection(
        name="disease_knowledge",
        metadata={"hnsw:space": "cosine", "description": "疾病知识库 - 症状+科室+描述"}
    )

    embed_start = time.time()
    for i in range(0, len(docs1), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(docs1))
        embeddings = model.encode(docs1[i:batch_end], show_progress_bar=False)
        coll1.add(
            embeddings=embeddings.tolist(),
            documents=docs1[i:batch_end],
            metadatas=metas1[i:batch_end],
            ids=ids1[i:batch_end],
        )
        pct = batch_end / len(docs1) * 100
        print(f"    入库进度: {batch_end}/{len(docs1)} ({pct:.0f}%)")
    print(f"  [OK] disease_knowledge 完成, 耗时 {time.time() - embed_start:.1f}s")

    # ---- Collection 2: 症状→科室直接映射 ----
    print("\n  构建 Collection 2/3: symptom_dept_direct")
    symptom_dept_map = defaultdict(lambda: defaultdict(int))
    for d in diseases:
        symptoms = d.get("symptom", [])
        departments = d.get("cure_department", [])
        for sym in symptoms:
            for dept in departments:
                symptom_dept_map[sym][dept] += 1

    # 只保留关联疾病数 >= 2 的症状 (过滤噪声)
    docs2, metas2, ids2 = [], [], []
    idx = 0
    for sym, dept_counts in symptom_dept_map.items():
        if stats["symptom_counter"].get(sym, 0) < 2:
            continue
        top_depts = sorted(dept_counts.items(), key=lambda x: -x[1])[:5]
        dept_names = [d for d, _ in top_depts]
        total_diseases = stats["symptom_counter"].get(sym, 0)

        search_text = f"症状：{sym}。常见关联科室：{'、'.join(dept_names)}。"
        docs2.append(search_text)
        metas2.append({
            "symptom": sym,
            "departments": ", ".join(dept_names),
            "disease_count": total_diseases,
        })
        ids2.append(f"sym_{idx:04d}")
        idx += 1

    print(f"    共 {len(docs2)} 条高频症状映射")

    try:
        client.delete_collection("symptom_dept_direct")
    except Exception:
        pass

    coll2 = client.create_collection(
        name="symptom_dept_direct",
        metadata={"hnsw:space": "cosine", "description": "症状→科室直接映射"}
    )

    embed_start = time.time()
    for i in range(0, len(docs2), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(docs2))
        embeddings = model.encode(docs2[i:batch_end], show_progress_bar=False)
        coll2.add(
            embeddings=embeddings.tolist(),
            documents=docs2[i:batch_end],
            metadatas=metas2[i:batch_end],
            ids=ids2[i:batch_end],
        )
    print(f"  [OK] symptom_dept_direct 完成, 耗时 {time.time() - embed_start:.1f}s")

    # ---- Collection 3: 科室信息库 ----
    print("\n  构建 Collection 3/3: department_info")
    docs3, metas3, ids3 = prepare_department_info(diseases, stats)
    print(f"    共 {len(docs3)} 个科室")

    try:
        client.delete_collection("department_info")
    except Exception:
        pass

    coll3 = client.create_collection(
        name="department_info",
        metadata={"hnsw:space": "cosine", "description": "科室信息库 - 诊疗范围+常见症状"}
    )

    embed_start = time.time()
    embeddings = model.encode(docs3, show_progress_bar=False)
    coll3.add(
        embeddings=embeddings.tolist(),
        documents=docs3,
        metadatas=metas3,
        ids=ids3,
    )
    print(f"  [OK] department_info 完成, 耗时 {time.time() - embed_start:.1f}s")

    # ---- 完成 ----
    total_elapsed = time.time() - total_start
    print(f"\n[4/4] [OK] 知识库构建完成!")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  数据库路径: {os.path.abspath(DB_PATH)}")
    print(f"  Collection 数量: 3")
    print(f"  disease_knowledge:   {coll1.count()} 条")
    print(f"  symptom_dept_direct: {coll2.count()} 条")
    print(f"  department_info:     {coll3.count()} 条")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据文件不存在: {DATA_PATH}")
        print(f"   请确保 OpenKG 数据已放置在正确路径")
        sys.exit(1)

    build_all()
