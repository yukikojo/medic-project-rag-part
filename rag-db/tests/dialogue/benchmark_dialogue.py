"""
多轮对话延迟基准测试
测试 start + 连续追问 的每轮耗时，找出瓶颈
"""
import sys, os, time, json
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, _src)
from dialogue import DialogueManager

# Test case: headache (偏头痛), 5 turns
TURNS = [
    "头痛三天了，右边太阳穴跳着疼",
    "有时候会恶心，看到强光的时候疼得更厉害",
    "发作前有时候眼前会闪白光",
    "睡一觉会好一点，吃布洛芬能缓解",
    "对，最近工作压力确实很大，经常熬夜",
]

def benchmark():
    manager = DialogueManager(verbose=False)
    latencies = []
    total_start = time.time()

    # Turn 0: start session
    t0 = time.time()
    result = manager.start_session(patient_id=99999, initial_symptom=TURNS[0])
    t = time.time() - t0
    latencies.append(("start (首轮)", t, result.get("action", "?")))
    session_id = result.get("session_id", "")

    # Turns 1-4: continue
    for i, answer in enumerate(TURNS[1:], 1):
        t0 = time.time()
        result = manager.process(session_id=session_id, patient_input=answer)
        t = time.time() - t0
        action = result.get("action", "?")
        latencies.append((f"turn {i+1} ({'追问' if action=='ask' else action})", t, action))

        if action in ("recommend", "emergency"):
            break

    total = time.time() - total_start

    # Clean up
    manager.close_session(session_id)

    # Print results
    print(f"\n{'='*55}")
    print(f"  多轮对话延迟基准 — 模拟偏头痛问诊")
    print(f"{'='*55}")
    print(f"  {'阶段':<25} {'耗时':>8}  {'动作'}")
    print(f"  {'-'*45}")
    for name, t, action in latencies:
        print(f"  {name:<25} {t:>7.2f}s  {action}")

    avg = sum(t for _, t, _ in latencies) / len(latencies)
    print(f"  {'-'*45}")
    print(f"  {'总耗时':<25} {total:>7.2f}s")
    print(f"  {'平均每轮':<25} {avg:>7.2f}s ({len(latencies)} 轮)")
    print(f"{'='*55}")

    # Breakdown analysis
    print(f"\n  💡 每轮 LLM 调用: extract_symptoms + decision + followup = 3 次 API 调用")
    print(f"  如果每次 API 调用 ~1-2s，理论每轮 ~3-6s")
    print(f"  实际每轮 {avg:.1f}s → {'合理' if avg < 8 else '偏慢，检查 API 延迟'}")

    return avg

if __name__ == "__main__":
    benchmark()
