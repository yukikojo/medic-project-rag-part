"""
health_suggestion — 个性化生活建议生成模块

面向患者的 5 类结构化生活建议:
  - diet (饮食)
  - exercise (运动)
  - sleep (睡眠)
  - medication (用药)
  - seasonal (季节性)

输入: health_record + consultation 两张表数据
输出: 按 category 分组的 JSON 建议列表
"""

from .suggestion_generator import HealthSuggestionGenerator, CATEGORIES, CATEGORY_LABELS

__all__ = ["HealthSuggestionGenerator", "CATEGORIES", "CATEGORY_LABELS"]
