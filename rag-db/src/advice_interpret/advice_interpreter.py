"""
advice_interpreter.py
Doctor Advice Interpreter — 医嘱解读

将医生的专业诊疗建议翻译为面向患者的通俗语言说明。

场景: advice_interpret (ai_model_config)
输入: doctor_advice (文本) + patient_context (可选)
输出: 结构化解释 (通俗说明 + 要点 + 用药 + 注意事项)

使用:
    interpreter = AdviceInterpreter(verbose=True)
    result = interpreter.interpret(
        doctor_advice="建议低盐低脂饮食, 硝苯地平30mg qd, 监测血压",
        patient_context="患者张三, 男, 65岁, 高血压5年",
    )
"""
import os
import sys
import json
import time
import re
from typing import Optional

# Ensure parent src/ is importable
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(os.path.join(_src, "..", "..", ".env"))

# ============================================================
# Fallback prompt
# ============================================================

DEFAULT_SYSTEM_PROMPT = """你是一位资深临床医师，擅长将医生的专业诊疗建议解读为患者易懂的语言。

## 任务
根据医生的原始诊疗建议，生成一份面向患者的解读说明。

## 输出要求
请严格按照以下 JSON 格式输出:
{
  "plain_explanation": "用通俗语言解释医生的诊断和建议",
  "key_points": ["关键要点1", "要点2"],
  "medication_guide": "用药指导说明",
  "follow_up_advice": "复诊和生活注意事项"
}

## 约束
- 语言通俗易懂，避免过度使用专业术语
- 不确定的内容请标注「请以医生当面说明为准」
- 不要添加医生未提及的诊断或建议"""


# ============================================================
# AdviceInterpreter
# ============================================================

class AdviceInterpreter:
    """
    医生建议解读器 — 将专业医嘱翻译为患者语言。

    使用:
        interpreter = AdviceInterpreter(verbose=True)
        result = interpreter.interpret(
            doctor_advice="建议低盐饮食, 硝苯地平30mg qd",
            patient_context="患者张三, 65岁, 高血压",
        )
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
        verbose: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy-load DeepSeekClient."""
        if self._llm_client is None:
            import importlib.util as _iu
            _dc = os.path.join(_src, "generation", "deepseek_client.py")
            _spec = _iu.spec_from_file_location("generation.deepseek_client", _dc)
            _mod = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            self._llm_client = _mod.DeepSeekClient(model=self.model)
        return self._llm_client

    def interpret(
        self,
        doctor_advice: str,
        patient_context: Optional[str] = None,
    ) -> dict:
        """
        解读医生建议为患者友好语言。

        Args:
            doctor_advice: 医生原始诊断/建议文本
            patient_context: 患者背景信息 (可选, 如年龄/性别/病史)

        Returns:
            {
                "plain_explanation": "通俗解释",
                "key_points": ["要点1", "要点2"],
                "medication_guide": "用药指导",
                "follow_up_advice": "复诊注意事项",
                "metadata": {"model": str, "latency_ms": float, "tokens": dict},
            }
        """
        start = time.time()

        if not doctor_advice or not str(doctor_advice).strip():
            return {
                "plain_explanation": "暂无医嘱需要解读",
                "key_points": [],
                "medication_guide": "",
                "follow_up_advice": "",
                "error": "doctor_advice 为空",
                "metadata": {"latency_ms": 0},
            }

        doctor_advice = str(doctor_advice).strip()

        # Load config from ai_config_loader
        try:
            from ai_config_loader import get_prompt, get_params
            system_prompt = get_prompt("advice_interpret")
            cfg = get_params("advice_interpret")
            _temp = cfg["temperature"]
            _max_tok = cfg["max_tokens"]
            _model = cfg["model"]
        except Exception:
            system_prompt = DEFAULT_SYSTEM_PROMPT
            _temp = self.temperature
            _max_tok = self.max_tokens
            _model = self.model or "qwen-flash"

        # Build user message
        user_parts = [f"## 医生原始建议\n{doctor_advice}"]
        if patient_context and str(patient_context).strip():
            user_parts.append(f"\n## 患者背景\n{patient_context}")
        user_message = "\n".join(user_parts)

        if self.verbose:
            print(f"[AdviceInterpreter] 正在解读医嘱 ({len(doctor_advice)} 字)...")

        try:
            response = self.llm_client.client.chat.completions.create(
                model=_model or self.model or self.llm_client.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=_temp,
                max_tokens=_max_tok,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content.strip()
            latency_ms = round((time.time() - start) * 1000, 1)

            parsed = self._parse_llm_json(raw_text)

            result = {
                "plain_explanation": parsed.get("plain_explanation", ""),
                "key_points": parsed.get("key_points", []),
                "medication_guide": parsed.get("medication_guide", ""),
                "follow_up_advice": parsed.get("follow_up_advice", ""),
                "metadata": {
                    "model": response.model,
                    "latency_ms": latency_ms,
                    "tokens": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    } if response.usage else None,
                },
            }

            if self.verbose:
                print(f"[AdviceInterpreter] 解读完成, {latency_ms}ms, "
                      f"{len(parsed.get('key_points', []))} 个要点")

            return result

        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 1)
            return {
                "plain_explanation": f"医嘱解读服务暂时不可用，请咨询医生获取详细说明。",
                "key_points": ["请遵医嘱服药", "如有疑问请咨询医生"],
                "medication_guide": str(doctor_advice)[:200],
                "follow_up_advice": "请按医生要求定期复诊",
                "error": str(e)[:100],
                "metadata": {"latency_ms": latency_ms},
            }

    def _parse_llm_json(self, raw_text: str) -> dict:
        """4 层 JSON 解析容错。"""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return {}
