"""
query_optimizer.py
查询优化器 — LLM驱动的口语化/方言症状标准化

支持所有兼容 OpenAI SDK 接口的 LLM 提供商。
通过 .env 环境变量独立配置优化器使用的模型。

配置方式 (.env 文件):
    OPTIMIZER_API_KEY=sk-xxx  # 默认: LLM_API_KEY → DEEPSEEK_API_KEY
    OPTIMIZER_BASE_URL=...    # 默认: LLM_BASE_URL → https://api.deepseek.com
    OPTIMIZER_MODEL=...       # 默认: deepseek-chat

在 RAG 检索之前对用户输入进行预处理:
  1. 口语化表达 → 标准医学术语 ("肚子疼" → "腹痛")
  2. 方言表达 → 标准化描述 ("打摆子" → "寒战发热")
  3. 症状结构化提取 (主要症状、部位、持续时长、严重程度)
  4. 生成用于向量检索的优化查询文本

架构位置:
  用户原始输入 → QueryOptimizer.optimize() → 标准化query → VectorStore检索 → LLM推荐

使用示例:
    from retrieval.query_optimizer import QueryOptimizer

    optimizer = QueryOptimizer()
    result = optimizer.optimize("肚子疼拉稀想吐没胃口")
    print(result["optimized_query"])   # "腹痛 腹泻 恶心 食欲不振"
    print(result["symptoms"])          # ["腹痛", "腹泻", "恶心", "食欲不振"]
    print(result["body_parts"])        # ["腹部", "消化系统"]
"""

import os
import json
import time
import hashlib
from typing import Optional
from functools import lru_cache

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


# ============================================================
# 口语化/方言 → 标准术语 规则词典 (LLM不可用时的兜底方案)
# ============================================================
# 结构化设计: {标准术语: [口语化/方言表达列表]}
# 定期维护此词典可不断提升脱网场景下的标准化能力
COLLOQUIAL_MAP: dict[str, list[str]] = {
    # === 消化道 ===
    "腹痛":       ["肚子疼", "肚子痛", "肚疼", "肚痛", "小腹痛", "胃痛", "胃疼",
                   "腹疼", "绞肚", "肚脐眼疼", "心口疼"],
    "腹泻":       ["拉肚子", "拉稀", "拉水", "跑肚", "闹肚子", "拉痢", "水便",
                   "稀便", "大便稀", "拉得厉害", "噗噗"],
    "恶心":       ["想吐", "反胃", "要吐", "犯恶心", "干哕", "恶心想吐",
                   "想呕", "胃里翻", "心翻"],
    "呕吐":       ["吐了", "呕了", "吐出来", "哕了", "吐酸水", "吐饭"],
    "食欲不振":   ["不想吃", "没胃口", "吃不下", "不爱吃饭", "厌食",
                   "没食欲", "不想吃饭", "吃啥都不香", "没味口"],
    "消化不良":   ["不消化", "胃胀", "积食", "撑着", "吃了不消化",
                   "胃里堵", "打嗝", "胀气", "肚子胀"],
    "便秘":       ["拉不出", "大便干", "大便困难", "拉屎费劲", "好几天没拉",
                   "干结", "便干"],
    "便血":       ["大便带血", "拉血", "便中有血", "屎里有血", "拉黑便"],

    # === 呼吸道 ===
    "咳嗽":       ["咳", "咳嗦", "干咳", "咳不停", "咳得厉害", "咳痰",
                   "老咳", "咳喘"],
    "发热":       ["发烧", "烧", "发烧了", "烫", "体温高", "发烫",
                   "浑身烫", "高烧", "低烧", "发热了", "烧得厉害"],
    "咽痛":       ["嗓子疼", "喉咙痛", "嗓子痛", "咽喉痛", "嗓子干疼",
                   "喉咙疼", "嗓子发炎", "喉咙肿", "吞口水疼"],
    "鼻塞":       ["鼻子不通", "鼻子堵", "鼻塞不通", "鼻子不通气", "堵鼻子",
                   "鼻堵", "不通气"],
    "流涕":       ["流鼻涕", "鼻涕多", "淌鼻涕", "流鼻水", "清鼻涕",
                   "鼻涕流", "流清涕"],
    "呼吸困难":   ["喘不上气", "喘不过气", "气短", "上不来气", "气不够用",
                   "憋气", "吸气困难", "透不过气"],
    "咳痰":       ["咳出痰", "咳嗽带痰", "有痰咳不出", "黏痰"],
    "喷嚏":       ["打喷嚏", "阿嚏", "鼻子痒打喷嚏"],

    # === 心血管 ===
    "心悸":       ["心慌", "心跳快", "心里发慌", "心跳得厉害", "心突突",
                   "心乱跳", "心跳慢", "心跳不齐", "心里咯噔"],
    "胸闷":       ["胸口闷", "胸口堵", "胸发闷", "胸口不舒服", "胸口像压着",
                   "憋闷", "胸发紧", "胸口沉重"],
    "胸痛":       ["胸口疼", "胸疼", "胸口痛", "心口痛", "心口疼",
                   "前胸疼", "胸口刺痛", "胸口闷痛"],
    "头晕":       ["发晕", "晕", "头昏", "天旋地转", "站不住",
                   "头发蒙", "迷糊", "晕乎乎", "昏昏沉沉"],
    "头痛":       ["头疼", "脑壳疼", "偏头疼", "头胀", "头重",
                   "脑袋疼", "太阳穴疼", "后脑勺疼", "头不舒服"],

    # === 骨骼肌肉 ===
    "关节痛":     ["关节疼", "关节痛", "骨头节疼", "膝盖疼", "膝盖痛",
                   "膀子疼", "胳膊肘疼", "手腕疼", "脚脖子疼"],
    "腰背痛":     ["腰疼", "腰酸", "腰痛", "腰不舒服", "背疼", "背痛",
                   "后腰疼", "腰板疼", "腰杆痛"],
    "肌肉酸痛":   ["浑身酸", "肌肉疼", "身上疼", "全身疼", "肌肉痛",
                   "腿酸", "胳膊酸", "身上酸"],
    "腿麻":       ["腿发麻", "脚麻", "腿木", "腿发木", "脚发麻"],
    "颈椎痛":     ["脖子疼", "颈椎疼", "脖子僵", "脖子酸", "颈子痛",
                   "转不了头", "落枕"],
    "肩痛":       ["肩膀疼", "肩膀痛", "膀子疼", "肩关节疼", "抬不起胳膊"],

    # === 皮肤 ===
    "皮疹":       ["起疹子", "起红点", "长疹子", "皮肤起红", "出疹",
                   "起疙瘩", "小疙瘩", "红疙瘩", "红斑", "起斑"],
    "瘙痒":       ["痒", "发痒", "痒痒", "刺痒", "痒得厉害",
                   "皮肤痒", "起痒", "瘙痒"],
    "红肿":       ["又红又肿", "又红又肿", "皮肤肿", "发红", "肿了",
                   "皮肤发红", "红了一块"],
    "过敏":       ["过敏了", "起过敏", "皮肤过敏", "荨麻疹", "风团"],

    # === 精神/神经 ===
    "失眠":       ["睡不着", "睡不好", "失眠多梦", "老醒", "难入睡",
                   "熬夜睡不着", "半夜醒", "早醒", "入睡困难"],
    "乏力":       ["没劲", "没力气", "浑身没劲", "乏力", "全身无力",
                   "累得很", "没精神", "虚", "疲乏", "打不起精神"],
    "焦虑":       ["心里急", "着急", "烦躁", "心不安", "坐立不安",
                   "紧张", "心静不下来"],
    "多汗":       ["出汗多", "盗汗", "出虚汗", "爱出汗", "汗多",
                   "夜间盗汗", "手心出汗"],

    # === 泌尿 ===
    "尿频":       ["老想上厕所", "尿多", "总想尿", "一会一趟厕所",
                   "小便多", "尿的次数多", "爱跑厕所"],
    "尿痛":       ["尿尿疼", "小便痛", "尿道疼", "尿着疼", "撒尿疼"],
    "尿急":       ["尿憋不住", "想尿就得马上去", "尿急", "憋不住尿"],

    # === 口腔/牙科 ===
    "牙痛":       ["牙疼", "牙痛", "牙齿疼", "牙龈疼", "大牙疼",
                   "智齿疼", "牙洞疼"],
    "牙龈出血":   ["刷牙出血", "牙龈流血", "牙齿出血", "嘴里有血"],
    "口腔溃疡":   ["嘴里起泡", "烂嘴", "口腔起泡", "嘴里破", "口疮",
                   "舌头疼", "嘴起泡"],

    # === 耳鼻喉 ===
    "耳鸣":       ["耳朵响", "耳朵叫", "耳朵嗡嗡", "耳朵里有声音",
                   "耳有杂音", "耳朵嗡嗡响"],
    "听力下降":   ["耳朵背", "听不清", "耳朵不好使", "听力差", "耳聋"],
    "鼻出血":     ["流鼻血", "鼻子出血", "淌鼻血", "鼻血"],
    "声嘶":       ["嗓子哑", "声音哑了", "说话费劲", "失声", "声音嘶哑"],

    # === 眼科 ===
    "视力模糊":   ["看不清", "眼花", "眼睛模糊", "视力不好", "看东西模糊",
                   "眼睛花", "视物不清"],
    "眼痛":       ["眼睛疼", "眼疼", "眼睛痛", "眼球疼", "眼睛胀痛"],
    "眼干":       ["眼睛干", "眼干涩", "眼干眼涩", "眼睛涩"],

    # === 妇科 ===
    "月经不调":   ["月经不规律", "经期不准", "月经乱", "大姨妈不准",
                   "经期推迟", "月经提前", "闭经"],
    "痛经":       ["来月经肚子疼", "月经痛", "大姨妈痛", "经期肚子疼",
                   "痛经厉害", "月经期间腹痛"],

    # === 儿科 ===
    "小儿厌食":   ["小孩不吃饭", "宝宝不爱吃", "娃娃不吃饭", "挑食",
                   "小孩没胃口", "喂饭难"],
    "小儿夜啼":   ["宝宝晚上哭", "小孩夜里哭", "晚上不睡哭闹"],

    # === 方言/地区特有表达 ===
    "寒战":       ["打摆子", "发冷", "打冷战", "哆嗦", "寒战",
                   "发寒", "畏寒", "怕冷", "冷得发抖", "打寒颤"],
    "中暑":       ["热着了", "中暑了", "热伤风", "受热", "暑气",
                   "热晕了", "晒晕了"],
    "上火":       ["火气大", "上火", "内热", "热气", "有火",
                   "口干舌燥", "嘴巴起泡"],
}

# 构建反向索引: 口语词 → 标准术语 (用于快速查找)
_COLLOQUIAL_TO_STANDARD: dict[str, str] = {}
for _standard, _colloquial_list in COLLOQUIAL_MAP.items():
    for _c in _colloquial_list:
        _COLLOQUIAL_TO_STANDARD[_c] = _standard


# ============================================================
# 配置
# ============================================================
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
MAX_CACHE_SIZE = 512  # LRU 缓存最大条目数


class QueryOptimizer:
    """
    查询优化器

    在 RAG 检索前对用户输入进行标准化处理，提升向量检索准确率。

    工作模式:
      - LLM模式 (默认): 调用 DeepSeek API 进行智能标准化，效果好
      - 规则模式 (兜底): 基于词典匹配，不依赖网络，速度快
      - 混合模式 (推荐): LLM优先，失败时自动降级到规则模式
    """

    def __init__(
        self,
        mode: str = "hybrid",          # "llm" | "rule" | "hybrid"
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_enabled: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            mode: 工作模式
                - "llm": 仅使用LLM标准化
                - "rule": 仅使用规则词典
                - "hybrid": LLM优先，失败时自动降级到规则词典
            api_key: LLM API Key，默认从 OPTIMIZER_API_KEY → LLM_API_KEY → DEEPSEEK_API_KEY 读取
            model: 模型名称，默认从 OPTIMIZER_MODEL 读取，最终 fallback deepseek-chat
            base_url: LLM API 地址，默认从 OPTIMIZER_BASE_URL → LLM_BASE_URL → https://api.deepseek.com 读取
            cache_enabled: 是否启用查询缓存
            verbose: 是否打印详细日志
        """
        self.mode = mode
        self.cache_enabled = cache_enabled
        self.verbose = verbose

        # 初始化 LLM 客户端 (hybrid/llm 模式下需要)
        self._llm_client = None
        if mode in ("hybrid", "llm"):
            try:
                from openai import OpenAI

                # 多级 fallback: 参数 > OPTIMIZER_* 环境变量 > LLM_* 环境变量 > DeepSeek 默认值
                _key = (
                    api_key
                    or os.getenv("OPTIMIZER_API_KEY")
                    or os.getenv("LLM_API_KEY")
                    or os.getenv("DEEPSEEK_API_KEY")
                )
                _url = (
                    base_url
                    or os.getenv("OPTIMIZER_BASE_URL")
                    or os.getenv("LLM_BASE_URL")
                    or DEFAULT_BASE_URL
                )
                _model = (
                    model
                    or os.getenv("OPTIMIZER_MODEL")
                    or DEFAULT_MODEL
                )

                if _key:
                    self._llm_client = OpenAI(
                        api_key=_key,
                        base_url=_url,
                    )
                    self._llm_model = _model
                    if verbose:
                        print(f"[QueryOptimizer] LLM模式已就绪: {self._llm_model} @ {_url}")
                else:
                    if mode == "llm":
                        raise ValueError(
                            "LLM模式下必须设置 API Key。请在 .env 中设置 "
                            "OPTIMIZER_API_KEY、LLM_API_KEY 或 DEEPSEEK_API_KEY"
                        )
                    if verbose:
                        print("[QueryOptimizer] 未检测到API Key，回退到规则模式")
                    self.mode = "rule"
            except ImportError:
                if mode == "llm":
                    raise ImportError("LLM模式需要安装 openai 库: pip install openai")
                if verbose:
                    print("[QueryOptimizer] openai库未安装，回退到规则模式")
                self.mode = "rule"
            except Exception as e:
                if mode == "llm":
                    raise
                if verbose:
                    print(f"[QueryOptimizer] LLM初始化失败: {e}，回退到规则模式")
                self.mode = "rule"

        # 简单的内存缓存 (避免同一查询反复调用LLM)
        self._cache: dict[str, dict] = {}

    # ================================================================
    # 主入口
    # ================================================================

    def optimize(self, raw_query: str) -> dict:
        """
        优化用户查询，将口语化/方言表达标准化为医学症状描述。

        这是对外暴露的主方法。RAGPipeline 在调用 VectorStore 检索前先调用此方法。

        Args:
            raw_query: 用户原始输入，如 "肚子疼拉稀想吐没胃口"

        Returns:
            {
                "original_query": "肚子疼拉稀想吐没胃口",    # 保留原始输入
                "optimized_query": "腹痛 腹泻 恶心 食欲不振", # 标准化后 (用于向量检索)
                "symptoms": ["腹痛", "腹泻", "恶心", "食欲不振"],  # 结构化症状列表
                "body_parts": ["腹部", "消化系统"],           # 涉及的身体部位
                "duration": "未知",                          # 持续时长
                "severity": "中",                           # 严重程度: 轻/中/重/未知
                "keywords": ["腹痛", "腹泻", "恶心", "食欲不振"],
                "normalization_note": "LLM标准化",            # 标准化方式说明
                "latency_ms": 123.4,                        # 优化耗时
            }
        """
        # 空输入保护
        if not raw_query or not raw_query.strip():
            return self._empty_result(raw_query)

        raw_query = raw_query.strip()

        # 检查缓存
        if self.cache_enabled:
            cache_key = self._cache_key(raw_query)
            if cache_key in self._cache:
                if self.verbose:
                    print(f"[QueryOptimizer] 缓存命中: '{raw_query[:20]}...'")
                result = dict(self._cache[cache_key])
                result["from_cache"] = True
                return result

        start_time = time.time()

        # 按优先级尝试各模式
        if self.mode == "llm":
            result = self._optimize_with_llm(raw_query)
        elif self.mode == "rule":
            result = self._optimize_with_rules(raw_query)
        elif self.mode == "hybrid":
            result = self._optimize_with_llm(raw_query)
            if result.get("_llm_failed"):
                if self.verbose:
                    print(f"[QueryOptimizer] LLM失败，回退到规则模式")
                result = self._optimize_with_rules(raw_query)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")

        # 如果 LLM 也没有返回有效症状，用规则兜底
        if not result.get("symptoms") or not result.get("optimized_query"):
            if self.verbose:
                print("[QueryOptimizer] 标准化结果为空，使用规则兜底")
            fallback = self._optimize_with_rules(raw_query)
            result.update(fallback)

        result["latency_ms"] = round((time.time() - start_time) * 1000, 1)

        # 写入缓存
        if self.cache_enabled:
            self._cache_set(raw_query, result)

        return result

    # ================================================================
    # LLM 标准化
    # ================================================================

    def _optimize_with_llm(self, raw_query: str) -> dict:
        """使用 DeepSeek LLM 进行智能标准化"""
        if not self._llm_client:
            return self._empty_result(raw_query, _llm_failed=True)

        system_prompt = """你是一个中文医疗文本标准化助手。你的任务是将患者的口语化、方言化症状描述转化为标准化的医学症状术语。

## 规则
1. **口语→标准**: "肚子疼"→"腹痛"、"拉肚子"→"腹泻"、"想吐"→"恶心"、"发烧"→"发热"、"心慌"→"心悸"
2. **方言→标准**: "打摆子"→"寒战发热"、"心口疼"→"胃痛或心绞痛(根据上下文)"、"闹肚子"→"腹泻"
3. **保留关键修饰词**: 如果用户描述了程度/频率/位置 (如 "剧烈头痛"、"右下腹痛"、"持续咳嗽")，保留这些信息
4. **不要捏造**: 只标准化用户明确提到的症状，不要添加用户没说过的症状
5. **保留紧急信号**: 如 "剧烈胸痛"、"大出血"、"意识不清" 保留原样并标记 severity 为重
6. **多个症状用逗号分隔**

## 输出格式
请严格按照以下 JSON 格式输出 (不要输出其他内容):

{
  "standardized_text": "标准化后的症状描述, 用逗号分隔",
  "symptoms": ["标准化症状1", "标准化症状2", ...],
  "body_parts": ["头部", "胸部", ...],
  "duration": "持续时长 (未知则填'未知')",
  "severity": "轻/中/重/未知",
  "has_emergency_signals": false,
  "note": "标准化说明, 如'已将口语表达转换为标准术语'"
}"""

        user_message = f"请将以下患者的症状描述标准化为医学术语:\n\n{raw_query}"

        try:
            response = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content
            parsed = json.loads(raw_text)

            # 构建结果
            result = {
                "original_query": raw_query,
                "optimized_query": parsed.get("standardized_text", raw_query),
                "symptoms": parsed.get("symptoms", []),
                "body_parts": parsed.get("body_parts", []),
                "duration": parsed.get("duration", "未知"),
                "severity": parsed.get("severity", "未知"),
                "keywords": parsed.get("symptoms", []),
                "normalization_note": parsed.get("note", "LLM标准化"),
                "has_emergency_signals": parsed.get("has_emergency_signals", False),
                "_llm_failed": False,
            }

            # 如果 LLM 返回的 standardized_text 为空，用 symptoms 拼接
            if not result["optimized_query"] and result["symptoms"]:
                result["optimized_query"] = " ".join(result["symptoms"])

            return result

        except Exception as e:
            if self.verbose:
                print(f"[QueryOptimizer] LLM调用异常: {e}")
            return self._empty_result(raw_query, _llm_failed=True)

    # ================================================================
    # 规则词典标准化 (兜底方案)
    # ================================================================

    def _optimize_with_rules(self, raw_query: str) -> dict:
        """
        基于规则词典进行标准化。

        算法:
          1. 遍历口语→标准词典, 查找口语表达在 query 中的位置 → 映射为标准术语
          2. 遍历标准术语词典, 查找标准术语是否直接出现在 query 中 → 直接提取
          3. 去重 + 按位置排序
          4. 构建优化查询文本
        """
        matches = []  # [(start_pos, end_pos, colloquial, standard), ...]

        # Step 1: 按口语表达长度降序匹配 (优先长匹配, 避免"肚子疼"被拆成"肚子"+"疼")
        sorted_terms = sorted(
            _COLLOQUIAL_TO_STANDARD.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for colloquial, standard in sorted_terms:
            pos = 0
            while True:
                pos = raw_query.find(colloquial, pos)
                if pos == -1:
                    break
                matches.append((pos, pos + len(colloquial), colloquial, standard))
                pos += 1

        # Step 2: 也检测标准术语是否直接出现在查询中
        #   (避免漏掉用户已用标准术语表达的症状, 如 "咳嗽" → "咳嗽")
        standard_terms_by_len = sorted(
            COLLOQUIAL_MAP.keys(),
            key=len,
            reverse=True,
        )
        for standard in standard_terms_by_len:
            pos = 0
            while True:
                pos = raw_query.find(standard, pos)
                if pos == -1:
                    break
                # 只有当该位置没有被口语化匹配覆盖时才添加
                matches.append((pos, pos + len(standard), standard, standard))
                pos += 1

        # 按位置排序, 去除重叠匹配 (保留更长的匹配)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # 过滤重叠匹配
        filtered_matches = []
        last_end = -1
        for start, end, coll, std in matches:
            if start >= last_end:
                filtered_matches.append((start, end, coll, std))
                last_end = end

        if not filtered_matches:
            # 没有任何匹配，返回原始输入
            return self._empty_result(
                raw_query,
                note="规则词典未匹配到已知口语表达，保留原始输入",
                _llm_failed=False,
            )

        # 提取标准化症状列表
        symptoms = []
        for _, _, _, std in filtered_matches:
            if std not in symptoms:
                symptoms.append(std)

        # 构建优化后的查询文本
        optimized_query = " ".join(symptoms)

        # 推断身体部位
        body_parts = self._infer_body_parts(symptoms)

        return {
            "original_query": raw_query,
            "optimized_query": optimized_query,
            "symptoms": symptoms,
            "body_parts": body_parts,
            "duration": "未知",
            "severity": "未知",
            "keywords": symptoms,
            "normalization_note": f"规则词典标准化 (匹配 {len(filtered_matches)} 个口语表达)",
            "has_emergency_signals": self._check_emergency(raw_query),
            "_llm_failed": False,
        }

    # ================================================================
    # 辅助方法
    # ================================================================

    def _infer_body_parts(self, symptoms: list[str]) -> list[str]:
        """根据症状推断涉及的身体部位"""
        SYMPTOM_TO_PART = {
            "腹痛": "腹部", "腹泻": "腹部", "恶心": "消化系统", "呕吐": "消化系统",
            "食欲不振": "消化系统", "消化不良": "消化系统", "便秘": "消化系统",
            "咳嗽": "呼吸系统/胸部", "发热": "全身", "咽痛": "咽喉",
            "鼻塞": "鼻部", "流涕": "鼻部", "呼吸困难": "呼吸系统/胸部",
            "心悸": "心脏/胸部", "胸闷": "胸部", "胸痛": "胸部",
            "头晕": "头部", "头痛": "头部",
            "关节痛": "关节", "腰背痛": "腰背部", "肌肉酸痛": "肌肉/全身",
            "皮疹": "皮肤", "瘙痒": "皮肤", "红肿": "皮肤",
            "失眠": "精神/神经", "焦虑": "精神/神经", "乏力": "全身",
            "尿频": "泌尿系统", "尿痛": "泌尿系统",
            "牙痛": "口腔", "牙龈出血": "口腔",
            "耳鸣": "耳部", "听力下降": "耳部",
            "视力模糊": "眼部", "眼痛": "眼部", "眼干": "眼部",
            "月经不调": "妇科", "痛经": "妇科",
        }

        parts = set()
        for s in symptoms:
            # 精确匹配
            if s in SYMPTOM_TO_PART:
                parts.add(SYMPTOM_TO_PART[s])
            else:
                # 模糊匹配: 检查症状名是否包含部位关键词
                for sym, part in SYMPTOM_TO_PART.items():
                    if sym in s or s in sym:
                        parts.add(part)

        return list(parts) if parts else ["全身"]

    def _check_emergency(self, text: str) -> bool:
        """检查是否有危急症状信号"""
        EMERGENCY_KEYWORDS = [
            "剧烈胸痛", "剧烈头痛", "大出血", "意识不清", "突然晕倒",
            "呼吸困难", "窒息", "心脏骤停", "休克", "晕厥",
            "抽搐", "大咯血", "严重外伤", "中毒",
        ]
        return any(kw in text for kw in EMERGENCY_KEYWORDS)

    def _cache_key(self, raw_query: str) -> str:
        """生成缓存键"""
        return hashlib.md5(raw_query.encode("utf-8")).hexdigest()

    def _cache_set(self, raw_query: str, result: dict) -> None:
        """写入缓存 (LRU 淘汰策略)"""
        cache_key = self._cache_key(raw_query)
        if len(self._cache) >= MAX_CACHE_SIZE:
            # 淘汰最旧的一个条目
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        # 存储时不包含 from_cache 标记
        cached = dict(result)
        cached.pop("from_cache", None)
        self._cache[cache_key] = cached

    def _empty_result(
        self,
        raw_query: str = "",
        note: str = "空输入",
        _llm_failed: bool = False,
    ) -> dict:
        """生成空结果"""
        return {
            "original_query": raw_query or "",
            "optimized_query": raw_query or "",
            "symptoms": [],
            "body_parts": [],
            "duration": "未知",
            "severity": "未知",
            "keywords": [],
            "normalization_note": note,
            "has_emergency_signals": False,
            "_llm_failed": _llm_failed,
        }

    # ================================================================
    # 工具方法
    # ================================================================

    def clear_cache(self) -> int:
        """清空缓存, 返回清空的条目数"""
        count = len(self._cache)
        self._cache.clear()
        if self.verbose:
            print(f"[QueryOptimizer] 已清空 {count} 条缓存")
        return count

    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        return {
            "size": len(self._cache),
            "max_size": MAX_CACHE_SIZE,
            "enabled": self.cache_enabled,
            "mode": self.mode,
        }

    def get_dictionary_stats(self) -> dict:
        """获取规则词典统计"""
        return {
            "standard_terms": len(COLLOQUIAL_MAP),
            "colloquial_entries": len(_COLLOQUIAL_TO_STANDARD),
            "body_part_mappings": 25,
        }

    def add_colloquial_term(self, standard: str, colloquial: str) -> None:
        """
        动态添加口语化→标准术语映射 (热更新)

        Args:
            standard: 标准医学术语
            colloquial: 口语化/方言表达
        """
        if standard not in COLLOQUIAL_MAP:
            COLLOQUIAL_MAP[standard] = []
        if colloquial not in COLLOQUIAL_MAP[standard]:
            COLLOQUIAL_MAP[standard].append(colloquial)
        _COLLOQUIAL_TO_STANDARD[colloquial] = standard
        if self.verbose:
            print(f"[QueryOptimizer] 新增映射: '{colloquial}' → '{standard}'")

    def batch_add_terms(self, mappings: list[tuple[str, str]]) -> int:
        """
        批量添加口语化术语映射

        Args:
            mappings: [(标准术语, 口语表达), ...]

        Returns:
            成功添加的数量
        """
        count = 0
        for standard, colloquial in mappings:
            self.add_colloquial_term(standard, colloquial)
            count += 1
        return count


# ============================================================
# 便捷函数 (用于快速集成到现有 Pipeline)
# ============================================================

# 全局单例 (延迟初始化)
_optimizer: Optional[QueryOptimizer] = None


def get_optimizer(
    mode: str = "hybrid",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> QueryOptimizer:
    """
    获取全局 QueryOptimizer 单例。

    使用方式:
        from retrieval.query_optimizer import get_optimizer
        optimizer = get_optimizer()
        result = optimizer.optimize("肚子疼拉肚子")

    Args:
        mode: 工作模式 "hybrid" | "llm" | "rule"
        api_key: LLM API Key，默认从环境变量读取
        model: LLM 模型名，默认从环境变量读取
        base_url: LLM API 地址，默认从环境变量读取
        verbose: 是否打印详细日志
    """
    global _optimizer
    if _optimizer is None:
        try:
            _optimizer = QueryOptimizer(
                mode=mode,
                api_key=api_key,
                model=model,
                base_url=base_url,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[QueryOptimizer] 初始化失败: {e}，使用规则模式")
            _optimizer = QueryOptimizer(mode="rule", verbose=verbose)
    return _optimizer


# ============================================================
# 命令行测试
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  QueryOptimizer — 查询优化器测试")
    print("=" * 65)

    # 初始化 (自动检测 LLM 可用性)
    optimizer = QueryOptimizer(mode="hybrid", verbose=True)

    print(f"\n  当前模式: {optimizer.mode}")
    print(f"  缓存状态: {'启用' if optimizer.cache_enabled else '禁用'}")
    print(f"  词典规模: {optimizer.get_dictionary_stats()}")

    # 测试用例: 覆盖各种口语化表达
    test_cases = [
        # (输入, 预期标准化方向)
        ("肚子疼拉稀想吐没胃口", "消化道口语化"),
        ("发烧咳嗽流鼻涕嗓子疼", "呼吸道口语化"),
        ("心慌胸闷气短胸口疼", "心血管口语化"),
        ("腰疼腿麻关节疼走不动路", "骨骼肌肉口语化"),
        ("睡不着没精神心里发慌", "精神神经口语化"),
        ("皮肤起红疙瘩痒得厉害", "皮肤口语化"),
        ("老想上厕所尿尿疼", "泌尿口语化"),
        ("打摆子发冷全身烫", "方言+发热"),
        ("牙疼刷牙出血嘴里起泡", "口腔口语化"),
        ("最近一周肚子总是隐隐作痛拉肚子想吐吃不下饭", "复杂多症状"),
        ("", "空输入"),
        ("今天天气真好", "非医疗输入"),
        ("头痛发热咳嗽流鼻涕", "已是标准术语 (应基本保持不变)"),
    ]

    for query, description in test_cases:
        print(f"\n{'─' * 55}")
        print(f"  输入:   {query if query else '(空)'}")
        print(f"  类型:   {description}")

        if not query:
            result = optimizer.optimize(query)
            print(f"  症状:   {result['symptoms']}")
            continue

        result = optimizer.optimize(query)

        print(f"  优化后: {result['optimized_query']}")
        print(f"  症状:   {result['symptoms']}")
        print(f"  部位:   {result['body_parts']}")
        print(f"  严重度: {result['severity']}")
        print(f"  耗时:   {result.get('latency_ms', 0)}ms")
        print(f"  方式:   {result['normalization_note']}")

        if result.get("from_cache"):
            print(f"  (来自缓存)")

    # 缓存统计
    print(f"\n{'=' * 55}")
    stats = optimizer.get_cache_stats()
    print(f"  缓存条目: {stats['size']}")
