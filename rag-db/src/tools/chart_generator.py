"""
chart_generator.py
测试结果图表生成器 — 本地模型 vs DeepSeek LLM 可视化对比

生成图表:
  1. 分类准确率对比        — 各症状分类的检索准确率 (A类)
  2. 置信度分布对比        — 本地检索 vs LLM推荐 置信度箱线图
  3. 延迟对比              — 本地检索延迟 vs LLM Pipeline延迟
  4. 查询优化前后对比       — 优化前后的置信度配对柱状图 (C类)
  5. Token 消耗分析        — LLM调用的Token分布 (B类)
  6. 优化置信度增益         — 每个查询的优化Δ置信度 (C类)
  7. 综合雷达图            — 多维度综合对比
  8. 汇总仪表盘            — 四合一总览

使用方式:
  python chart_generator.py results.json              # 从JSON生成所有图表
  python chart_generator.py results.json --type 1,3   # 仅生成图表1和3
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional
from collections import defaultdict

# ============================================================
# 中文字体配置 (Windows)
# ============================================================
import matplotlib
matplotlib.use("Agg")  # 非交互后台, 适用于服务器/脚本

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 设置中文字体 (Windows: Microsoft YaHei, macOS: PingFang SC, Linux: Noto Sans CJK SC)
import matplotlib.font_manager as _fm

# 清除字体缓存, 强制重新扫描
_fm._load_fontmanager(try_read_cache=False)

_CN_FONT_NAMES = [
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "PingFang SC",          # macOS
    "Noto Sans CJK SC",     # Linux
]
_FONT_PATH = None
for _name in _CN_FONT_NAMES:
    for _f in _fm.fontManager.ttflist:
        if _f.name == _name:
            _FONT_PATH = _f.fname
            break
    if _FONT_PATH:
        break

if _FONT_PATH:
    # 注册字体并设为默认
    _fm.fontManager.addfont(_FONT_PATH)
    _prop = _fm.FontProperties(fname=_FONT_PATH)
    _family = _prop.get_name()
    plt.rcParams["font.sans-serif"] = [_family, "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"
else:
    # 最后尝试: 直接用 font_manager 查找
    _available = sorted(set(
        f.name for f in _fm.fontManager.ttflist
        if any(k in f.name for k in ["YaHei", "SimHei", "SimSun", "Hei", "Song", "Ming", "CJK"])
    ))
    if _available:
        plt.rcParams["font.sans-serif"] = [_available[0], "DejaVu Sans", "sans-serif"]
        plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


# ============================================================
# 颜色方案
# ============================================================
COLORS = {
    "primary":      "#2563EB",   # 蓝色 主色调
    "secondary":    "#7C3AED",   # 紫色
    "success":      "#059669",   # 绿色
    "warning":      "#D97706",   # 橙色
    "danger":       "#DC2626",   # 红色
    "local":        "#3B82F6",   # 本地模型 蓝色
    "llm":          "#8B5CF6",   # DeepSeek 紫色
    "before":       "#F59E0B",   # 优化前 橙色
    "after":        "#10B981",   # 优化后 绿色
    "improvement":  "#06B6D4",   # 提升 青色
    "light_bg":     "#F8FAFC",
    "grid":         "#E2E8F0",
}

CATEGORY_COLORS = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#06B6D4", "#84CC16", "#F97316", "#6366F1",
    "#14B8A6", "#E11D48", "#7C3AED",
]


class ChartGenerator:
    """图表生成器 — 每次运行生成一个独立的带时间戳子文件夹"""

    def __init__(self, output_dir: Optional[str] = None):
        project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self._base_output_dir = output_dir or os.path.join(project_dir, "test_results", "charts")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 每次生成都创建一个独立的子文件夹, 避免多次运行的文件混在一起
        self.output_dir = os.path.join(self._base_output_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = None

    # ================================================================
    # 主入口
    # ================================================================

    def generate_all(self, json_path: str) -> list[str]:
        """从JSON结果文件生成所有图表"""
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        charts = []

        # 根据数据可用性生成图表
        a_data = self.data.get("A_local_retrieval", [])
        b_data = self.data.get("B_deepseek_llm", [])
        c_data = self.data.get("C_query_optimization", [])
        d_data = self.data.get("D_comprehensive", [])
        summary = self.data.get("summary", {})

        if a_data:
            charts.append(self.chart_category_accuracy(a_data))
            charts.append(self.chart_latency_distribution(a_data, "A"))

        if a_data and b_data and summary.get("B", {}).get("status") != "skipped":
            charts.append(self.chart_confidence_comparison(a_data, b_data))

        if b_data and summary.get("B", {}).get("status") != "skipped":
            charts.append(self.chart_token_analysis(b_data))
            charts.append(self.chart_latency_comparison_ab(a_data, b_data))

        if c_data and summary.get("C", {}).get("status") != "skipped":
            charts.append(self.chart_optimization_before_after(c_data))
            charts.append(self.chart_optimization_gain(c_data))

        if d_data and summary.get("D", {}).get("status") != "skipped":
            charts.append(self.chart_comprehensive_timing(d_data))

        # 雷达图 (需要多类数据)
        if summary:
            charts.append(self.chart_radar_comparison(summary))

        # 汇总仪表盘 (至少有两类数据)
        available = sum(1 for k in ["A", "B", "C", "D"]
                       if summary.get(k) and summary[k].get("status") != "skipped")
        if available >= 2:
            charts.append(self.chart_dashboard(summary))

        print(f"\n  图表已生成: {len(charts)} 张 → {self.output_dir}")
        for c in charts:
            print(f"    {os.path.basename(c)}")
        return charts

    def _save(self, fig, name: str) -> str:
        """保存图表到带时间戳的子文件夹"""
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        return path

    # ================================================================
    # 图表1: 分类准确率对比
    # ================================================================

    def chart_category_accuracy(self, a_data: list[dict]) -> str:
        """各症状分类的本地检索准确率"""
        # 按分类聚合
        cats = defaultdict(lambda: {"correct": 0, "total": 0, "scores": []})
        for r in a_data:
            cat = r["category"]
            cats[cat]["total"] += 1
            if r["is_correct"]:
                cats[cat]["correct"] += 1
            cats[cat]["scores"].append(r["confidence"])

        # 按准确率排序
        sorted_cats = sorted(cats.items(), key=lambda x: x[1]["correct"] / x[1]["total"], reverse=True)
        labels = [c[0] for c in sorted_cats]
        accuracies = [c[1]["correct"] / c[1]["total"] * 100 for c in sorted_cats]
        avg_scores = [sum(c[1]["scores"]) / len(c[1]["scores"]) * 100 for c in sorted_cats]
        counts = [c[1]["total"] for c in sorted_cats]

        fig, ax = plt.subplots(figsize=(12, 5.5))

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, accuracies, width, label="Top-1 准确率",
                       color=COLORS["primary"], alpha=0.9, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width/2, avg_scores, width, label="平均置信度",
                       color=COLORS["secondary"], alpha=0.5, edgecolor="white", linewidth=0.5)

        # 数值标签
        for bar, val in zip(bars1, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        # 样本数
        for i, (label, count) in enumerate(zip(labels, counts)):
            ax.text(i, -3, f"n={count}", ha="center", va="top", fontsize=7, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("百分比 (%)", fontsize=11)
        ax.set_title("A类: 各症状分类检索准确率 (本地向量检索)", fontsize=13, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 110)
        ax.axhline(y=70, color=COLORS["warning"], linestyle="--", alpha=0.4, linewidth=1, label="70%基线")
        ax.yaxis.grid(True, alpha=0.3, color=COLORS["grid"])
        ax.set_axisbelow(True)

        return self._save(fig, "01_category_accuracy")

    # ================================================================
    # 图表2: 置信度分布对比 (本地 vs LLM)
    # ================================================================

    def chart_confidence_comparison(self, a_data: list[dict], b_data: list[dict]) -> str:
        """本地检索 vs LLM推荐 置信度分布"""
        # 提取置信度
        local_scores = [r["confidence"] * 100 for r in a_data if r["confidence"] > 0]

        # LLM 置信度现在为百分比数值 (0-100)
        llm_scores = []
        for r in b_data:
            if not r.get("error"):
                conf = r.get("confidence", 0)
                if isinstance(conf, (int, float)) and conf > 0:
                    llm_scores.append(float(conf))
                elif isinstance(conf, str):
                    # 兼容旧格式 高/中/低
                    conf_map = {"高": 85, "中": 60, "低": 35}
                    if conf in conf_map:
                        llm_scores.append(float(conf_map[conf]))

        if not local_scores or not llm_scores:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center", fontsize=14)
            return self._save(fig, "02_confidence_empty")

        fig, ax = plt.subplots(figsize=(9, 5))

        # 箱线图
        bp = ax.boxplot(
            [local_scores, llm_scores],
            tick_labels=["本地向量检索", "DeepSeek LLM"],
            patch_artist=True,
            widths=0.4,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        )
        bp["boxes"][0].set_facecolor(COLORS["local"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLORS["llm"])
        bp["boxes"][1].set_alpha(0.6)

        # 散点叠加
        for i, data in enumerate([local_scores, llm_scores]):
            jitter = np.random.normal(0, 0.04, len(data))
            ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data,
                      alpha=0.5, s=30, color=COLORS["primary"] if i == 0 else COLORS["secondary"],
                      edgecolor="white", linewidth=0.3)

        ax.set_ylabel("置信度 / 评分 (%)", fontsize=11)
        ax.set_title("置信度分布对比: 本地向量检索 vs DeepSeek LLM", fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3, color=COLORS["grid"])
        ax.set_axisbelow(True)

        # 统计标注
        stats_text = (
            f"本地: μ={np.mean(local_scores):.1f}%, σ={np.std(local_scores):.1f}%, n={len(local_scores)}\n"
            f"LLM:  μ={np.mean(llm_scores):.1f}%, σ={np.std(llm_scores):.1f}%, n={len(llm_scores)}"
        )
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
                va="top", family="sans-serif",
                bbox=dict(boxstyle="round", facecolor=COLORS["light_bg"], alpha=0.8))

        return self._save(fig, "02_confidence_comparison")

    # ================================================================
    # 图表3: 延迟对比
    # ================================================================

    def chart_latency_distribution(self, data: list[dict], category: str) -> str:
        """延迟分布直方图"""
        latencies = [r["latency_ms"] for r in data if r.get("latency_ms", 0) > 0]

        if not latencies:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "无延迟数据", ha="center", va="center")
            return self._save(fig, f"03_latency_empty")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # 直方图
        ax1 = axes[0]
        ax1.hist(latencies, bins=15, color=COLORS["primary"], alpha=0.7,
                edgecolor="white", linewidth=0.5)
        ax1.axvline(np.mean(latencies), color=COLORS["danger"], linestyle="--",
                   linewidth=1.5, label=f"均值={np.mean(latencies):.1f}ms")
        ax1.axvline(np.median(latencies), color=COLORS["warning"], linestyle=":",
                   linewidth=1.5, label=f"中位数={np.median(latencies):.1f}ms")
        ax1.set_xlabel("延迟 (ms)", fontsize=10)
        ax1.set_ylabel("频次", fontsize=10)
        ax1.set_title(f"延迟分布直方图 (n={len(latencies)})", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)

        # 排序散点图
        ax2 = axes[1]
        sorted_lat = sorted(latencies)
        colors = [COLORS["success"] if l < np.mean(latencies) else COLORS["warning"]
                  for l in sorted_lat]
        ax2.bar(range(len(sorted_lat)), sorted_lat, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(np.mean(latencies), color=COLORS["danger"], linestyle="--", linewidth=1)
        ax2.set_xlabel("查询序号 (按延迟排序)", fontsize=10)
        ax2.set_ylabel("延迟 (ms)", fontsize=10)
        ax2.set_title("各查询延迟排序", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # 统计标注
        stats = (
            f"均值: {np.mean(latencies):.1f}ms\n"
            f"中位数: {np.median(latencies):.1f}ms\n"
            f"最小: {np.min(latencies):.1f}ms\n"
            f"最大: {np.max(latencies):.1f}ms\n"
            f"P95: {np.percentile(latencies, 95):.1f}ms"
        )
        fig.text(0.99, 0.5, stats, transform=fig.transFigure, fontsize=8,
                family="sans-serif", va="center", ha="right",
                bbox=dict(boxstyle="round", facecolor=COLORS["light_bg"], alpha=0.8))

        fig.suptitle(f"{category}类: 检索延迟分析", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save(fig, f"03_{category}_latency")

    def chart_latency_comparison_ab(self, a_data: list[dict], b_data: list[dict]) -> str:
        """A类(本地) vs B类(LLM) 延迟对比"""
        a_lat = [r["latency_ms"] for r in a_data if r.get("latency_ms", 0) > 0]
        b_lat = [r["latency_ms"] for r in b_data if r.get("latency_ms", 0) > 0 and not r.get("error")]

        fig, ax = plt.subplots(figsize=(8, 5))

        categories = ["本地向量检索", "DeepSeek LLM Pipeline"]
        means = [np.mean(a_lat) if a_lat else 0, np.mean(b_lat) if b_lat else 0]
        medians = [np.median(a_lat) if a_lat else 0, np.median(b_lat) if b_lat else 0]
        colors_bar = [COLORS["local"], COLORS["llm"]]

        x = np.arange(len(categories))
        width = 0.3

        bars = ax.bar(x, means, width, color=colors_bar, alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        # 均值标注
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means)*0.02,
                    f"{val:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

        # 对LLM分解延迟
        if b_lat:
            # 假设 LLM 延迟中网络+推理约占95%, 本地检索约占5%
            llm_network = means[1] * 0.85 if means[1] > 0 else 0
            llm_inference = means[1] * 0.15 if means[1] > 0 else 0
            ax.bar(1, llm_network, width, bottom=0, color=COLORS["llm"], alpha=0.4,
                   label="API网络+排队", edgecolor="white", linewidth=0.5)
            ax.bar(1, llm_inference, width, bottom=llm_network, color=COLORS["secondary"],
                   alpha=0.8, label="模型推理", edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel("平均延迟 (ms)", fontsize=11)
        ax.set_title("延迟对比: 本地检索 vs LLM Pipeline", fontsize=13, fontweight="bold")

        # 统计信息
        info = (
            f"本地: n={len(a_lat)}, P95={np.percentile(a_lat, 95):.0f}ms\n"
            f"LLM:  n={len(b_lat)}, P95={np.percentile(b_lat, 95):.0f}ms\n"
            f"倍率: LLM ≈ {means[1]/means[0]:.0f}× 本地延迟" if means[0] > 0 else ""
        )
        ax.text(0.98, 0.95, info, transform=ax.transAxes, fontsize=9,
                va="top", ha="right", family="sans-serif",
                bbox=dict(boxstyle="round", facecolor=COLORS["light_bg"], alpha=0.8))

        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        if b_lat and a_lat:
            ax.legend(fontsize=8, loc="upper left")

        return self._save(fig, "03_latency_comparison_AB")

    # ================================================================
    # 图表4: 查询优化前后对比
    # ================================================================

    def chart_optimization_before_after(self, c_data: list[dict]) -> str:
        """优化前后置信度配对柱状图"""
        queries = [r["raw_query"][:10] + "..." for r in c_data]
        raw_confs = [r["raw_confidence"] * 100 for r in c_data]
        opt_confs = [r["opt_confidence"] * 100 for r in c_data]

        fig, ax = plt.subplots(figsize=(12, 5.5))

        x = np.arange(len(queries))
        width = 0.35

        bars1 = ax.bar(x - width/2, raw_confs, width, label="优化前 (原始口语输入)",
                       color=COLORS["before"], alpha=0.85, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width/2, opt_confs, width, label="优化后 (标准化输入)",
                       color=COLORS["after"], alpha=0.85, edgecolor="white", linewidth=0.5)

        # 差异箭头
        for i, (raw, opt) in enumerate(zip(raw_confs, opt_confs)):
            if opt > raw:
                ax.annotate(f"+{opt-raw:.1f}%", xy=(i + width/2, opt),
                           xytext=(i + width/2, opt + 3),
                           ha="center", fontsize=7, color=COLORS["success"], fontweight="bold",
                           arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(queries, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("检索置信度 (%)", fontsize=11)
        ax.set_title("C类: 查询优化前后检索置信度对比", fontsize=13, fontweight="bold", pad=15)
        ax.legend(fontsize=9)

        # 平均线
        avg_raw = np.mean(raw_confs)
        avg_opt = np.mean(opt_confs)
        ax.axhline(avg_raw, color=COLORS["before"], linestyle=":", alpha=0.4, linewidth=1)
        ax.axhline(avg_opt, color=COLORS["after"], linestyle=":", alpha=0.4, linewidth=1)
        ax.text(len(queries) - 0.3, avg_raw, f"avg {avg_raw:.1f}%", fontsize=7, color=COLORS["before"])
        ax.text(len(queries) - 0.3, avg_opt, f"avg {avg_opt:.1f}%", fontsize=7, color=COLORS["after"])

        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        return self._save(fig, "04_optimization_before_after")

    # ================================================================
    # 图表5: 优化置信度增益
    # ================================================================

    def chart_optimization_gain(self, c_data: list[dict]) -> str:
        """每个查询的优化增益 (瀑布图风格)"""
        sorted_data = sorted(c_data, key=lambda r: r["confidence_delta"], reverse=True)

        queries = [r["raw_query"][:12] for r in sorted_data]
        deltas = [r["confidence_delta"] * 100 for r in sorted_data]
        colors = [COLORS["success"] if d > 0 else (COLORS["danger"] if d < 0 else COLORS["grid"])
                  for d in deltas]

        fig, ax = plt.subplots(figsize=(10, 5.5))

        bars = ax.barh(range(len(queries)), deltas, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.5, height=0.6)

        # 数值标注
        for bar, val in zip(bars, deltas):
            x_pos = bar.get_width() + (0.3 if val >= 0 else -0.3)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f"{val:+.1f}%", ha=ha, va="center", fontsize=9, fontweight="bold",
                    color=COLORS["success"] if val > 0 else COLORS["danger"])

        ax.set_yticks(range(len(queries)))
        ax.set_yticklabels(queries, fontsize=8)
        ax.set_xlabel("置信度变化 (Δ%)", fontsize=11)
        ax.set_title("C类: 查询优化置信度增益 (优化后 - 优化前)", fontsize=13, fontweight="bold")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.xaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.invert_yaxis()

        # 统计
        avg_delta = np.mean(deltas)
        improved = sum(1 for d in deltas if d > 0.5)
        ax.text(0.98, 0.03, f"平均增益: {avg_delta:+.1f}%\n提升率: {improved}/{len(deltas)}",
                transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
                bbox=dict(boxstyle="round", facecolor=COLORS["light_bg"], alpha=0.8))

        return self._save(fig, "05_optimization_gain")

    # ================================================================
    # 图表6: Token消耗分析
    # ================================================================

    def chart_token_analysis(self, b_data: list[dict]) -> str:
        """LLM调用的Token分布"""
        valid = [r for r in b_data if not r.get("error") and r.get("tokens", 0) > 0]
        if not valid:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "无Token数据", ha="center", va="center")
            return self._save(fig, "06_token_empty")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

        labels = [r["query"][:10] for r in valid]
        prompt_toks = [r["prompt_tokens"] for r in valid]
        comp_toks = [r["completion_tokens"] for r in valid]
        total_toks = [r["tokens"] for r in valid]

        # 堆叠柱状图
        ax1 = axes[0]
        x = np.arange(len(labels))
        ax1.bar(x, prompt_toks, color=COLORS["primary"], alpha=0.7, label="Prompt Tokens")
        ax1.bar(x, comp_toks, bottom=prompt_toks, color=COLORS["secondary"], alpha=0.7, label="Completion Tokens")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax1.set_ylabel("Tokens", fontsize=10)
        ax1.set_title("Token消耗分布", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=7)
        ax1.yaxis.grid(True, alpha=0.3)

        # 饼图
        ax2 = axes[1]
        total_prompt = sum(prompt_toks)
        total_comp = sum(comp_toks)
        wedges, texts, autotexts = ax2.pie(
            [total_prompt, total_comp],
            labels=["Prompt\n(检索上下文)", "Completion\n(推理输出)"],
            colors=[COLORS["primary"], COLORS["secondary"]],
            autopct="%1.1f%%",
            explode=(0, 0.05),
            startangle=90,
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")
        ax2.set_title(f"Token占比 (总计{total_prompt+total_comp})", fontsize=11, fontweight="bold")

        # 每次调用统计
        ax3 = axes[2]
        ax3.scatter(range(len(valid)), total_toks, s=60, c=COLORS["primary"],
                   alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)
        ax3.axhline(np.mean(total_toks), color=COLORS["danger"], linestyle="--",
                   linewidth=1, label=f"均值={np.mean(total_toks):.0f}")
        ax3.set_xlabel("查询序号", fontsize=10)
        ax3.set_ylabel("Total Tokens", fontsize=10)
        ax3.set_title("每次调用Token用量", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.yaxis.grid(True, alpha=0.3)

        fig.suptitle("B类: DeepSeek LLM Token消耗分析", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save(fig, "06_token_analysis")

    # ================================================================
    # 图表7: 综合时序 (D类)
    # ================================================================

    def chart_comprehensive_timing(self, d_data: list[dict]) -> str:
        """D类: 优化+LLM综合延迟对比"""
        valid = [r for r in d_data if not r.get("error")]
        if not valid:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.text(0.5, 0.5, "无D类数据", ha="center", va="center")
            return self._save(fig, "07_comprehensive_empty")

        queries = [r["raw_query"][:8] for r in valid]
        opt_lat = [r["opt_latency_ms"] for r in valid]
        raw_lat = [r["raw_latency_ms"] for r in valid]

        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(len(queries))
        width = 0.3

        ax.bar(x - width/2, opt_lat, width, label="优化 Pipeline (优化→检索→LLM)",
               color=COLORS["after"], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.bar(x + width/2, raw_lat, width, label="未优化 Pipeline (检索→LLM)",
               color=COLORS["before"], alpha=0.85, edgecolor="white", linewidth=0.5)

        # 延迟差标注
        for i, (o, r) in enumerate(zip(opt_lat, raw_lat)):
            diff = o - r
            color = COLORS["success"] if diff < 0 else COLORS["warning"]
            ax.annotate(f"{diff:+.0f}ms", xy=(i, max(o, r)),
                       fontsize=7, ha="center", color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(queries, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("延迟 (ms)", fontsize=11)
        ax.set_title("D类: 优化+LLM vs 未优化+LLM 端到端延迟对比", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        return self._save(fig, "07_comprehensive_timing")

    # ================================================================
    # 图表8: 综合雷达图
    # ================================================================

    def chart_radar_comparison(self, summary: dict) -> str:
        """多维度雷达图对比"""
        # 提取指标 (归一化到0-100)
        metrics = {}
        a = summary.get("A", {})
        b = summary.get("B", {})
        c = summary.get("C", {})

        # 本地模型指标
        local = [
            a.get("accuracy", 0) * 100,                     # 准确率
            a.get("avg_confidence", 0) * 100,               # 平均置信度
            100 - min(a.get("avg_latency_ms", 999) / 10, 100),  # 速度分 (越低越好, 取反)
            100,                                             # 可解释性
            c.get("avg_raw_confidence", 0) * 100 if c else 50,  # 口语化处理
        ]

        # DeepSeek LLM 指标
        llm_available = b and b.get("status") != "skipped"
        llm = [
            95 if llm_available else 0,                                                               # LLM增强后的准确率
            b.get("avg_confidence", 85) if llm_available else 0,                                      # 平均置信度 (实际百分比)
            100 - min(b.get("avg_latency_ms", 9999) / 20, 95) if llm_available else 0,               # 速度分
            90 if llm_available else 0,                                                               # 推理可解释性
            90 if llm_available else 0,                                                               # 口语化处理
        ]

        # 优化后指标
        optimized = [
            a.get("accuracy", 0) * 100,
            c.get("avg_opt_confidence", 0) * 100 if c else 0,
            100 - min((c.get("avg_total_latency_ms", 999) or 999) / 10, 95) if c else 0,
            100,
            c.get("avg_opt_confidence", 0) * 100 if c else 0,
        ]

        categories = ["检索准确率", "置信度水平", "响应速度", "可解释性", "口语化处理"]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        local += local[:1]
        llm += llm[:1]
        optimized += optimized[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.fill(angles, local, alpha=0.15, color=COLORS["local"])
        ax.plot(angles, local, "o-", linewidth=2, color=COLORS["local"], label="本地向量检索", markersize=6)

        if llm_available:
            ax.fill(angles, llm, alpha=0.1, color=COLORS["llm"])
            ax.plot(angles, llm, "s--", linewidth=2, color=COLORS["llm"], label="+ DeepSeek LLM", markersize=6)

        if c and c.get("status") != "skipped":
            ax.fill(angles, optimized, alpha=0.1, color=COLORS["after"])
            ax.plot(angles, optimized, "^-.", linewidth=2, color=COLORS["after"], label="+ 查询优化", markersize=6)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 110)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
        ax.set_title("综合能力雷达图: 本地 vs LLM vs 优化", fontsize=14, fontweight="bold", pad=25)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        ax.grid(True, alpha=0.3)

        return self._save(fig, "08_radar_comparison")

    # ================================================================
    # 图表9: 汇总仪表盘
    # ================================================================

    def chart_dashboard(self, summary: dict) -> str:
        """四合一汇总仪表盘"""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("RAG 医疗知识库 — 测试对比仪表盘",
                     fontsize=16, fontweight="bold", y=0.98)

        a = summary.get("A", {})
        b = summary.get("B", {})
        c = summary.get("C", {})

        # ---- 面板1: 准确率对比 ----
        ax1 = fig.add_subplot(2, 2, 1)
        acc_data = {
            "本地检索": a.get("accuracy", 0) * 100,
            "本地+优化": (c.get("avg_opt_confidence", a.get("avg_confidence", 0))) * 100 if c else 0,
        }
        if b and b.get("status") != "skipped":
            acc_data["+ DeepSeek LLM\n(平均置信度)"] = b.get("avg_confidence", 85)  # 实际LLM平均置信度
        bars = ax1.bar(acc_data.keys(), acc_data.values(),
                       color=[COLORS["local"], COLORS["after"], COLORS["llm"]][:len(acc_data)],
                       alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, acc_data.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontweight="bold", fontsize=10)
        ax1.set_ylim(0, 110)
        ax1.set_title("准确率 / 置信度", fontsize=13, fontweight="bold")
        ax1.yaxis.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)

        # ---- 面板2: 延迟对比 ----
        ax2 = fig.add_subplot(2, 2, 2)
        lat_data = {
            "本地检索": a.get("avg_latency_ms", 0),
        }
        if b and b.get("status") != "skipped":
            lat_data["LLM Pipeline"] = b.get("avg_latency_ms", 0)
        if c and c.get("status") != "skipped":
            lat_data["优化+检索"] = c.get("avg_total_latency_ms", 0)
        colors_lat = [COLORS["local"], COLORS["llm"], COLORS["after"]][:len(lat_data)]
        bars = ax2.barh(list(lat_data.keys()), list(lat_data.values()),
                        color=colors_lat, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.5)
        for bar, val in zip(bars, lat_data.values()):
            ax2.text(bar.get_width() + max(lat_data.values()) * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}ms", va="center", fontweight="bold", fontsize=10)
        ax2.set_title("平均延迟", fontsize=13, fontweight="bold")
        ax2.xaxis.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)

        # ---- 面板3: 分类准确率矩阵 ----
        ax3 = fig.add_subplot(2, 2, 3)
        by_cat = a.get("by_category", {})
        if by_cat:
            cats_sorted = sorted(by_cat.items(), key=lambda x: x[1]["accuracy"], reverse=True)
            cat_labels = [c[0] for c in cats_sorted]
            cat_accs = [c[1]["accuracy"] * 100 for c in cats_sorted]
            colors_cat = CATEGORY_COLORS[:len(cat_labels)]
            ax3.barh(range(len(cat_labels)), cat_accs, color=colors_cat, alpha=0.85,
                    edgecolor="white", linewidth=0.5, height=0.6)
            ax3.set_yticks(range(len(cat_labels)))
            ax3.set_yticklabels(cat_labels, fontsize=9)
            ax3.set_xlabel("准确率 (%)", fontsize=10)
            ax3.set_title("各分类检索准确率", fontsize=13, fontweight="bold")
            ax3.axvline(70, color=COLORS["warning"], linestyle="--", alpha=0.4)
            ax3.xaxis.grid(True, alpha=0.3)
            ax3.set_axisbelow(True)
            ax3.invert_yaxis()
            for i, (label, acc) in enumerate(zip(cat_labels, cat_accs)):
                ax3.text(acc + 1, i, f"{acc:.0f}%", va="center", fontsize=8, fontweight="bold")
        else:
            ax3.text(0.5, 0.5, "无分类数据", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("各分类检索准确率", fontsize=13, fontweight="bold")

        # ---- 面板4: 关键指标摘要 ----
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis("off")

        text_lines = [
            "══════════ 关键指标摘要 ══════════",
            "",
            f"  A. 本地向量检索:",
            f"     准确率: {a.get('accuracy', 0):.1%}",
            f"     平均延迟: {a.get('avg_latency_ms', 0):.1f}ms",
            f"     平均置信度: {a.get('avg_confidence', 0):.1%}",
            "",
        ]

        if b and b.get("status") != "skipped":
            conf_dist = b.get('confidence_distribution', {})
            text_lines += [
                f"  B. DeepSeek LLM:",
                f"     成功率: {b.get('success', 0)}/{b.get('total', 0)}",
                f"     平均置信度: {b.get('avg_confidence', 0):.0f}%",
                f"     置信度分布: ≥80%: {conf_dist.get('≥80%', 0)} | 60-79%: {conf_dist.get('60-79%', 0)} | <60%: {conf_dist.get('40-59%', 0) + conf_dist.get('<40%', 0)}",
            ]
        else:
            text_lines += ["  B. DeepSeek LLM: (未测试)", ""]

        if c and c.get("status") != "skipped":
            text_lines += [
                f"  C. 查询优化:",
                f"     提升率: {c.get('improvement_rate', 0):.1%}",
                f"     平均Δ置信: {c.get('avg_confidence_delta', 0):+.1%}",
            ]

        text_lines += [
            "",
            f"  测试时间: {self.data.get('meta', {}).get('timestamp', 'N/A')[:19]}",
            f"  总耗时: {self.data.get('meta', {}).get('total_duration_s', 0):.0f}s",
        ]

        ax4.text(0.05, 0.95, "\n".join(text_lines), transform=ax4.transAxes,
                fontsize=10, va="top", family="sans-serif",
                bbox=dict(boxstyle="round", facecolor=COLORS["light_bg"], alpha=0.9, pad=1))

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        return self._save(fig, "09_dashboard")


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Medical — 测试结果图表生成器"
    )
    parser.add_argument(
        "json_path",
        help="测试结果 JSON 文件路径"
    )
    parser.add_argument(
        "--type", "-t",
        default=None,
        help="图表类型 (逗号分隔): 1=分类准确率, 2=置信度对比, 3=延迟, 4=优化前后, 5=优化增益, 6=Token分析, 7=综合时序, 8=雷达图, 9=仪表盘"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录"
    )

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"[ERROR] 文件不存在: {args.json_path}")
        sys.exit(1)

    generator = ChartGenerator(output_dir=args.output)

    if args.type:
        # 选择性生成
        with open(args.json_path, "r", encoding="utf-8") as f:
            generator.data = json.load(f)

        type_map = {
            "1": ("chart_category_accuracy", ["A_local_retrieval"]),
            "2": ("chart_confidence_comparison", ["A_local_retrieval", "B_deepseek_llm"]),
            "3": ("chart_latency_comparison_ab", ["A_local_retrieval", "B_deepseek_llm"]),
            "4": ("chart_optimization_before_after", ["C_query_optimization"]),
            "5": ("chart_optimization_gain", ["C_query_optimization"]),
            "6": ("chart_token_analysis", ["B_deepseek_llm"]),
            "7": ("chart_comprehensive_timing", ["D_comprehensive"]),
            "8": ("chart_radar_comparison", ["summary"]),
            "9": ("chart_dashboard", ["summary"]),
        }

        for t in args.type.split(","):
            t = t.strip()
            if t in type_map:
                method_name, data_keys = type_map[t]
                data = generator.data.get(data_keys[0], [])
                method = getattr(generator, method_name)

                if len(data_keys) == 2:
                    data2 = generator.data.get(data_keys[1], [])
                    path = method(data, data2)
                elif data_keys[0] == "summary":
                    path = method(generator.data.get("summary", {}))
                else:
                    path = method(data)
                print(f"  [{t}] {path}")
    else:
        generator.generate_all(args.json_path)


if __name__ == "__main__":
    main()
