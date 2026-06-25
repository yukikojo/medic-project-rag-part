"""
diagnose_gpu.py
诊断脚本 — 检查 embedding / reranker 模型是否真的跑在 GPU 上

运行:
  cd "d:/medic project"
  python rag-db/tests/diagnose_gpu.py
"""

import os
import sys
import time

_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _PROJECT_DIR)

# ============================================================
# 1. 检查 PyTorch / CUDA 基础环境
# ============================================================
print("=" * 60)
print("  1. PyTorch / CUDA 环境检查")
print("=" * 60)

import torch
print(f"  PyTorch 版本:     {torch.__version__}")
print(f"  CUDA 可用:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本:        {torch.version.cuda}")
    print(f"  GPU 数量:         {torch.cuda.device_count()}")
    print(f"  GPU 名称:         {torch.cuda.get_device_name(0)}")
    print(f"  GPU 显存:         {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  [WARN] CUDA 不可用 — 所有模型将在 CPU 上运行!")

# ============================================================
# 2. 检查 Embedding 模型 (BGE-M3) 加载到哪个设备
# ============================================================
print("\n" + "=" * 60)
print("  2. Embedding 模型 (BGE-M3) 设备检查")
print("=" * 60)

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_PATH = r"D:\floder-for-claude\medic\bge-m3"
print(f"  模型路径: {EMBEDDING_MODEL_PATH}")
print(f"  路径存在: {os.path.exists(EMBEDDING_MODEL_PATH)}")

print("  加载模型中...")
t0 = time.time()
model_emb = SentenceTransformer(EMBEDDING_MODEL_PATH)
t_load = time.time() - t0

# 检查模型参数所在的设备
param_device = next(model_emb.parameters()).device
print(f"  模型参数设备:     {param_device}")
print(f"  在 GPU 上:        {param_device.type == 'cuda'}")
print(f"  加载耗时:         {t_load:.1f}s")

# 编码性能测试
test_texts = ["头痛发热咳嗽流鼻涕"] * 10
print(f"  编码测试 ({len(test_texts)} 条)...")
t0 = time.time()
embeddings = model_emb.encode(test_texts, show_progress_bar=False)
t_enc = time.time() - t0
print(f"  编码耗时:         {t_enc*1000:.0f}ms ({t_enc/len(test_texts)*1000:.0f}ms/条)")

# ============================================================
# 3. 检查 Reranker 模型 (BGE-Reranker) 加载到哪个设备
# ============================================================
print("\n" + "=" * 60)
print("  3. Reranker 模型 (BGE-Reranker) 设备检查")
print("=" * 60)

from sentence_transformers import CrossEncoder

RERANKER_MODEL_PATH = r"D:\floder-for-claude\medic\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
print(f"  模型路径: {RERANKER_MODEL_PATH}")
print(f"  路径存在: {os.path.exists(RERANKER_MODEL_PATH)}")

print("  加载模型中...")
t0 = time.time()
model_rerank = CrossEncoder(RERANKER_MODEL_PATH)
t_load = time.time() - t0

# 检查模型参数所在的设备
param_device_rr = next(model_rerank.model.parameters()).device
print(f"  模型参数设备:     {param_device_rr}")
print(f"  在 GPU 上:        {param_device_rr.type == 'cuda'}")
print(f"  加载耗时:         {t_load:.1f}s")

# Reranking 性能测试
query = "头痛发热咳嗽流鼻涕"
documents = [
    "疾病：感冒。症状：头痛、发热、咳嗽、流鼻涕。所属科室：呼吸内科。",
    "疾病：偏头痛。症状：头痛、恶心、畏光。所属科室：神经内科。",
    "疾病：过敏性鼻炎。症状：流鼻涕、打喷嚏、鼻塞。所属科室：耳鼻喉科。",
    "疾病：肺炎。症状：发热、咳嗽、咳痰、胸痛。所属科室：呼吸内科。",
    "疾病：高血压。症状：头痛、头晕、心悸。所属科室：心内科。",
]
pairs = [(query, doc) for doc in documents]

print(f"  Rerank 测试 ({len(pairs)} pairs)...")
t0 = time.time()
scores = model_rerank.predict(pairs, batch_size=32, show_progress_bar=False)
t_enc = time.time() - t0
print(f"  Rerank 耗时:      {t_enc*1000:.0f}ms ({t_enc/len(pairs)*1000:.0f}ms/pair)")

# ============================================================
# 4. 诊断结论
# ============================================================
print("\n" + "=" * 60)
print("  4. 诊断结论")
print("=" * 60)

issues = []

if param_device.type != "cuda":
    issues.append("Embedding 模型 (BGE-M3) 在 CPU 上 — 需要设置 device='cuda'")
if param_device_rr.type != "cuda":
    issues.append("Reranker 模型 (BGE-Reranker) 在 CPU 上 — 需要设置 device='cuda'")
if t_enc > 1.0:  # >1s for 5 pairs is very slow
    issues.append(f"Reranker 推理过慢 ({t_enc:.1f}s for 5 pairs)")

if not issues:
    print("  [OK] 所有模型都在 GPU 上运行，未发现配置问题。")
    print(f"  如果仍感觉慢，可能是:")
    print(f"    - 测试用例数量多 (A组100 + C组80*4 = 420次推理)")
    print(f"    - 每次 comprehensive_search 会调用 2 次 embedding + 1 次 rerank")
    print(f"    - 预计总耗时: ~{(420 * (t_enc/5 + 0.01)):.0f}s (取决于模型大小)")
else:
    for i, issue in enumerate(issues, 1):
        print(f"  [ISSUE {i}] {issue}")

print()
