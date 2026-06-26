"""
health_summary — 健康档案 AI 摘要生成

为 Java 后端传入的 health_record 表数据生成面向医生的专业摘要:
  - ai_summary: 自然语言段落, 概括患者健康状况、风险因素、注意事项
  - RAG 增强: 检索相关疾病知识作为 LLM 上下文, 提升专业性

使用:
  from health_summary import HealthSummaryGenerator
  gen = HealthSummaryGenerator()
  result = gen.generate(health_record_data)
  print(result["ai_summary"])
"""

from .summary_generator import HealthSummaryGenerator
