"""生成旧实验复现用的合成双语后训练数据（SFT、DPO 与 GRPO）。

推荐的新训练路线使用 ``scripts/data.py fetch-posttrain`` 下载固定版本的公开数据；
本脚本仅保留用于复现仓库已有的 synthetic GPU 实验，不是默认数据入口。

数据由本仓库自有的模板与词表确定性生成：不下载外部语料，不使用模型生成，
也不包含 ``configs/evaluation/lifecycle-v1.yaml`` 中的任何提示、分组或语段。
生成后的 JSONL 仍须交给 ``scripts/data.py prepare`` 做校验、分组切分与 Manifest。

模板按能力族组织（复写、算术、抽取、格式化、上下文记忆、多轮），
中英文使用同一实例参数的并行表述，因此同一 ``source_id`` 下的两种语言
天然互为翻译，会被切分逻辑放在同一个 split。

Generate legacy synthetic bilingual post-training data (SFT, DPO and GRPO).

Use ``scripts/data.py fetch-posttrain`` for the recommended pinned public-data
route. This script remains only to reproduce the repository's earlier
synthetic GPU experiments.

The corpus is produced deterministically from project-owned templates and word
lists: no external download, no model-generated text, and no prompt, group or
passage from ``configs/evaluation/lifecycle-v1.yaml``. The generated JSONL must
still be passed to ``scripts/data.py prepare`` for validation, group-aware
splitting and manifest creation.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.exceptions import LLMLabError

DATA_LICENSE = "Apache-2.0"
"""生成数据沿用仓库许可；词表与模板均为本仓库自撰内容。"""

DEFAULT_SEED = 20260101


# --------------------------------------------------------------------------
# 并行词表：同一索引的中英文互为翻译，保证 source_id 下两种语言同义。
# --------------------------------------------------------------------------

_WORDS: tuple[tuple[str, str], ...] = (
    ("apple", "苹果"),
    ("river", "河流"),
    ("mountain", "高山"),
    ("candle", "蜡烛"),
    ("garden", "花园"),
    ("window", "窗户"),
    ("bottle", "瓶子"),
    ("forest", "森林"),
    ("harbor", "港口"),
    ("meadow", "草地"),
    ("lantern", "灯笼"),
    ("bridge", "桥梁"),
    ("desert", "沙漠"),
    ("island", "岛屿"),
    ("temple", "寺庙"),
    ("market", "市场"),
    ("station", "车站"),
    ("library", "图书馆"),
    ("museum", "博物馆"),
    ("theater", "剧院"),
    ("factory", "工厂"),
    ("farm", "农场"),
    ("orchard", "果园"),
    ("bakery", "面包店"),
    ("school", "学校"),
    ("hospital", "医院"),
    ("airport", "机场"),
    ("tower", "塔楼"),
    ("castle", "城堡"),
    ("village", "村庄"),
    ("city", "城市"),
    ("street", "街道"),
    ("square", "广场"),
    ("palace", "宫殿"),
    ("tunnel", "隧道"),
    ("canal", "运河"),
    ("valley", "山谷"),
    ("cliff", "悬崖"),
    ("beach", "海滩"),
    ("cave", "洞穴"),
    ("waterfall", "瀑布"),
    ("lake", "湖泊"),
    ("pond", "池塘"),
    ("spring", "泉水"),
    ("field", "田野"),
    ("barn", "谷仓"),
    ("greenhouse", "温室"),
    ("windmill", "风车"),
    ("lighthouse", "灯塔"),
    ("cabin", "小屋"),
    ("cottage", "农舍"),
    ("courtyard", "庭院"),
    ("balcony", "阳台"),
    ("attic", "阁楼"),
    ("cellar", "地窖"),
    ("kitchen", "厨房"),
    ("bedroom", "卧室"),
    ("workshop", "作坊"),
    ("studio", "画室"),
    ("gallery", "画廊"),
    ("archive", "档案馆"),
    ("observatory", "天文台"),
    ("laboratory", "实验室"),
    ("clinic", "诊所"),
    ("pharmacy", "药房"),
    ("bookstore", "书店"),
    ("cafe", "咖啡馆"),
    ("restaurant", "餐厅"),
    ("hotel", "旅馆"),
    ("inn", "客栈"),
    ("hostel", "宿舍"),
    ("campus", "校园"),
    ("stadium", "体育场"),
    ("gymnasium", "体育馆"),
    ("pool", "泳池"),
    ("track", "跑道"),
    ("rink", "冰场"),
    ("court", "球场"),
    ("arena", "竞技场"),
    ("gateway", "大门"),
    ("fence", "栅栏"),
    ("hedge", "树篱"),
    ("path", "小径"),
    ("trail", "步道"),
    ("road", "公路"),
    ("railway", "铁路"),
    ("subway", "地铁"),
    ("ferry", "渡船"),
    ("sailboat", "帆船"),
    ("balloon", "热气球"),
    ("glider", "滑翔机"),
    ("rocket", "火箭"),
    ("satellite", "卫星"),
    ("telescope", "望远镜"),
    ("microscope", "显微镜"),
    ("compass", "指南针"),
    ("camera", "相机"),
    ("piano", "钢琴"),
    ("violin", "小提琴"),
    ("guitar", "吉他"),
    ("drum", "鼓"),
    ("flute", "长笛"),
)

_NAMES: tuple[tuple[str, str], ...] = (
    ("Ada", "阿达"),
    ("Bruno", "布鲁诺"),
    ("Clara", "克拉拉"),
    ("Dario", "达里奥"),
    ("Elena", "埃莱娜"),
    ("Felix", "费利克斯"),
    ("Greta", "格蕾塔"),
    ("Hugo", "雨果"),
    ("Iris", "伊里斯"),
    ("Jonas", "约纳斯"),
    ("Kira", "基拉"),
    ("Liam", "利亚姆"),
    ("Mona", "莫娜"),
    ("Nils", "尼尔斯"),
    ("Olga", "奥尔加"),
    ("Pablo", "帕布罗"),
    ("Quinn", "奎因"),
    ("Rosa", "罗莎"),
    ("Samir", "萨米尔"),
    ("Tessa", "泰莎"),
    ("Umar", "乌马尔"),
    ("Vera", "薇拉"),
    ("Walt", "沃尔特"),
    ("Xenia", "克谢尼娅"),
    ("Yara", "雅拉"),
    ("Zeno", "芝诺"),
    ("Anya", "安雅"),
    ("Bo", "博"),
    ("Cyrus", "赛勒斯"),
    ("Dila", "迪拉"),
    ("Emil", "埃米尔"),
    ("Farah", "法拉"),
    ("Gus", "古斯"),
    ("Hana", "哈娜"),
    ("Ivo", "伊沃"),
    ("Juno", "朱诺"),
    ("Kaito", "海斗"),
    ("Lena", "莱娜"),
    ("Milo", "米洛"),
    ("Noor", "努尔"),
)

_CITIES: tuple[tuple[str, str], ...] = (
    ("Oslo", "奥斯陆"),
    ("Lima", "利马"),
    ("Kyiv", "基辅"),
    ("Doha", "多哈"),
    ("Bern", "伯尔尼"),
    ("Nara", "奈良"),
    ("Porto", "波尔图"),
    ("Turin", "都灵"),
    ("Ghent", "根特"),
    ("Bilbao", "毕尔巴鄂"),
    ("Malmo", "马尔默"),
    ("Tallinn", "塔林"),
    ("Riga", "里加"),
    ("Ljubljana", "卢布尔雅那"),
    ("Zagreb", "萨格勒布"),
    ("Sofia", "索非亚"),
    ("Baku", "巴库"),
    ("Tbilisi", "第比利斯"),
    ("Almaty", "阿拉木图"),
    ("Tashkent", "塔什干"),
    ("Colombo", "科伦坡"),
    ("Dhaka", "达卡"),
    ("Kathmandu", "加德满都"),
    ("Yangon", "仰光"),
    ("Hanoi", "河内"),
    ("Da Nang", "岘港"),
    ("Busan", "釜山"),
    ("Fukuoka", "福冈"),
    ("Sapporo", "札幌"),
    ("Sendai", "仙台"),
)

_POSITIVE: tuple[tuple[str, str], ...] = (
    ("The soup was warm and delicious.", "这碗汤又热又好吃。"),
    ("I enjoyed every minute of the trip.", "这趟旅程我每一分钟都很享受。"),
    ("The staff were friendly and helpful.", "工作人员友好又乐于助人。"),
    ("This phone battery lasts all day.", "这部手机电池能用一整天。"),
    ("The room was clean and quiet.", "房间干净又安静。"),
    ("The movie made me laugh out loud.", "这部电影让我大笑不止。"),
    ("The teacher explained everything clearly.", "老师把一切都讲得很清楚。"),
    ("The bread came out of the oven perfect.", "面包出炉时完美极了。"),
    ("The train arrived exactly on time.", "列车准点到达。"),
    ("My new shoes are very comfortable.", "我的新鞋非常舒服。"),
    ("The garden looks lovely in spring.", "这座花园春天很美。"),
    ("The concert exceeded my expectations.", "音乐会超出了我的预期。"),
)

_NEGATIVE: tuple[tuple[str, str], ...] = (
    ("The soup was cold and bland.", "这碗汤又凉又没味道。"),
    ("I wasted the whole afternoon waiting.", "我等了整整一个下午，浪费了时间。"),
    ("The staff were rude and unhelpful.", "工作人员粗鲁又不帮忙。"),
    ("This phone battery dies in two hours.", "这部手机电池两小时就没电。"),
    ("The room was dusty and noisy.", "房间又脏又吵。"),
    ("The movie bored me from start to end.", "这部电影从头到尾让我无聊。"),
    ("The teacher rushed through every topic.", "老师把每个题目都草草带过。"),
    ("The bread was burnt on the outside.", "面包外面烤焦了。"),
    ("The train was delayed for an hour.", "列车晚点了一小时。"),
    ("My new shoes hurt my feet.", "我的新鞋磨脚。"),
    ("The garden is full of weeds.", "这座花园长满了杂草。"),
    ("The concert was far too loud.", "音乐会的声音实在太大了。"),
)

_OBJECTS: tuple[tuple[str, str], ...] = (
    ("apples", "苹果"),
    ("pears", "梨"),
    ("pencils", "铅笔"),
    ("marbles", "玻璃球"),
    ("cookies", "饼干"),
    ("stamps", "邮票"),
    ("buttons", "纽扣"),
    ("candles", "蜡烛"),
    ("spoons", "勺子"),
    ("notebooks", "笔记本"),
)

_UNITS: tuple[tuple[str, str, int], ...] = (
    ("meters", "米", 100),
    ("kilometers", "千米", 1000),
    ("kilograms", "千克", 1000),
)


# --------------------------------------------------------------------------
# 模板：sample 抽取实例参数，render 按语言渲染 (prompt, chosen, rejected)。
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Template:
    template_id: str
    family: str
    verifier: str
    sample: Callable[[random.Random], dict[str, Any]]
    render: Callable[[str, dict[str, Any], random.Random], tuple[str, str, str]]


def _pick(pairs: Sequence[tuple[str, ...]], index: int, language: str) -> str:
    return pairs[index][0] if language == "en" else pairs[index][1]


def _digit_copy_sample(rng: random.Random) -> dict[str, Any]:
    number = rng.randint(100, 999_999)
    digits = str(number)
    position = rng.randrange(len(digits))
    replacement = str((int(digits[position]) + rng.choice([1, 2, 3, 4])) % 10)
    mutated = digits[:position] + replacement + digits[position + 1 :]
    return {
        "digits": digits,
        "rejected": mutated,
        "verbose": rng.random() < 0.4,
    }


def _digit_copy_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    digits = str(params["digits"])
    prompt = (
        f"Repeat this number exactly, digits only: {digits}."
        if language == "en"
        else f"请原样重复这个数字，只输出数字：{digits}。"
    )
    rejected = (
        f"The number is {params['rejected']}."
        if params["verbose"] and language == "en"
        else str(params["rejected"])
    )
    if params["verbose"] and language == "zh":
        rejected = f"这个数字是{params['rejected']}。"
    return prompt, digits, rejected


def _word_echo_sample(rng: random.Random) -> dict[str, Any]:
    index = rng.randrange(len(_WORDS))
    other = rng.choice([i for i in range(len(_WORDS)) if i != index])
    return {
        "index": index,
        "other": other,
        "verbose": rng.random() < 0.4,
    }


def _word_echo_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    word = _pick(_WORDS, int(params["index"]), language)
    other = _pick(_WORDS, int(params["other"]), language)
    prompt = (
        f"Echo the following word and nothing else: {word}"
        if language == "en"
        else f"请复述下面这个词，不要输出其他内容：{word}"
    )
    rejected = (
        f"Sure, the word is {other}."
        if params["verbose"] and language == "en"
        else other
    )
    if params["verbose"] and language == "zh":
        rejected = f"好的，这个词是{other}。"
    return prompt, word, rejected


def _arith_sample(rng: random.Random, operation: str) -> dict[str, Any]:
    if operation == "add":
        left = rng.randint(2, 900)
        right = rng.randint(2, 900)
        value = left + right
    elif operation == "sub":
        left = rng.randint(50, 900)
        right = rng.randint(1, left)
        value = left - right
    else:
        left = rng.randint(2, 60)
        right = rng.randint(2, 20)
        value = left * right
    offset = rng.choice([1, 2, 3, 5, 10, -1, -2, -3])
    while value + offset == value:
        offset += 1
    return {
        "left": left,
        "right": right,
        "value": value,
        "wrong": max(0, value + offset),
        "verbose": rng.random() < 0.4,
    }


def _arith_render(
    language: str, params: dict[str, Any], operation: str
) -> tuple[str, str, str]:
    left, right = int(params["left"]), int(params["right"])
    value, wrong = int(params["value"]), int(params["wrong"])
    if operation == "add":
        prompt = (
            f"Compute {left} + {right}. Give the integer only."
            if language == "en"
            else f"计算 {left} 加 {right}。只给出整数。"
        )
    elif operation == "sub":
        prompt = (
            f"Compute {left} - {right}. Give the integer only."
            if language == "en"
            else f"计算 {left} 减 {right}。只给出整数。"
        )
    else:
        prompt = (
            f"Compute {left} * {right}. Give the integer only."
            if language == "en"
            else f"计算 {left} 乘 {right}。只给出整数。"
        )
    chosen = str(value)
    if params["verbose"]:
        rejected = (
            f"I think the answer is {wrong}."
            if language == "en"
            else f"我觉得答案是 {wrong}。"
        )
    else:
        rejected = str(wrong)
    return prompt, chosen, rejected


def _bracket_sample(rng: random.Random) -> dict[str, Any]:
    index = rng.randrange(len(_WORDS))
    other = rng.choice([i for i in range(len(_WORDS)) if i != index])
    return {"index": index, "other": other, "verbose": rng.random() < 0.4}


def _bracket_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    word = _pick(_WORDS, int(params["index"]), language)
    other = _pick(_WORDS, int(params["other"]), language)
    prompt = (
        f"Copy the word inside the square brackets and nothing else: [{word}]"
        if language == "en"
        else f"复制方括号里的词，不要输出其他内容：[{word}]"
    )
    if params["verbose"]:
        rejected = (
            f"The bracketed word is {other}."
            if language == "en"
            else f"括号里的词是{other}。"
        )
    else:
        rejected = other
    return prompt, word, rejected


def _json_wrap_sample(rng: random.Random) -> dict[str, Any]:
    key = rng.choice(["name", "city", "color", "fruit", "tool", "season", "animal"])
    index = rng.randrange(len(_WORDS))
    return {
        "key": key,
        "value": _pick(_WORDS, index, "en"),
        "value_zh": _pick(_WORDS, index, "zh"),
    }


def _json_wrap_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    key = str(params["key"])
    value = str(params["value"] if language == "en" else params["value_zh"])
    prompt = (
        f'Return JSON only: one key "{key}" whose value is the string "{value}".'
        if language == "en"
        else f'只输出 JSON：唯一的键是 "{key}"，值是字符串 "{value}"。'
    )
    chosen = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":"))
    rejected = f"The {key} is {value}." if language == "en" else f"{key} 是 {value}。"
    return prompt, chosen, rejected


def _sentiment_sample(rng: random.Random) -> dict[str, Any]:
    positive = rng.random() < 0.5
    index = rng.randrange(len(_POSITIVE))
    return {"positive": positive, "index": index, "verbose": rng.random() < 0.4}


def _sentiment_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    pool = _POSITIVE if params["positive"] else _NEGATIVE
    sentence = _pick(pool, int(params["index"]), language)
    prompt = (
        f"Read this review: {sentence} Reply with one word: positive or negative."
        if language == "en"
        else f"阅读这条评论：{sentence}用一个词回答：正面 或 负面。"
    )
    if language == "en":
        chosen = "positive" if params["positive"] else "negative"
        wrong = "negative" if params["positive"] else "positive"
        rejected = f"It feels {wrong} to me." if params["verbose"] else wrong
    else:
        chosen = "正面" if params["positive"] else "负面"
        wrong = "负面" if params["positive"] else "正面"
        rejected = f"我觉得是{wrong}。" if params["verbose"] else wrong
    return prompt, chosen, rejected


def _where_lives_sample(rng: random.Random) -> dict[str, Any]:
    name_index = rng.randrange(len(_NAMES))
    city_index = rng.randrange(len(_CITIES))
    other_index = rng.choice([i for i in range(len(_CITIES)) if i != city_index])
    return {
        "name": name_index,
        "city": city_index,
        "other": other_index,
        "verbose": rng.random() < 0.4,
    }


def _where_lives_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    name = _pick(_NAMES, int(params["name"]), language)
    city = _pick(_CITIES, int(params["city"]), language)
    other = _pick(_CITIES, int(params["other"]), language)
    if language == "en":
        prompt = (
            f"{name} lives in {city}. Which city does {name} live in? "
            "Answer with the city name only."
        )
        rejected = f"{name} lives in {other}." if params["verbose"] else other
    else:
        prompt = f"{name}住在{city}。{name}住在哪座城市？只回答城市名。"
        rejected = f"{name}住在{other}。" if params["verbose"] else other
    return prompt, city, rejected


def _count_objects_sample(rng: random.Random) -> dict[str, Any]:
    start = rng.randint(1, 40)
    added = rng.randint(1, 30)
    return {
        "start": start,
        "added": added,
        "object": rng.randrange(len(_OBJECTS)),
        "offset": rng.choice([1, 2, -1, -2]),
        "verbose": rng.random() < 0.4,
    }


def _count_objects_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    start, added = int(params["start"]), int(params["added"])
    thing = _pick(_OBJECTS, int(params["object"]), language)
    value = start + added
    wrong = max(0, value + int(params["offset"]))
    if language == "en":
        prompt = (
            f"There are {start} {thing} in the box. {added} more are put in. "
            "How many are there now? Answer with a digit."
        )
        rejected = f"There are {wrong} now." if params["verbose"] else str(wrong)
    else:
        prompt = (
            f"盒子里有 {start} 个{thing}，又放进 {added} 个。"
            "现在一共有多少个？用阿拉伯数字回答。"
        )
        rejected = f"现在有 {wrong} 个。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _unit_convert_sample(rng: random.Random) -> dict[str, Any]:
    unit_index = rng.randrange(len(_UNITS))
    amount = rng.randint(2, 40)
    offset = rng.choice([1, 2, 5, -1, -2])
    return {
        "unit": unit_index,
        "amount": amount,
        "offset": offset,
        "verbose": rng.random() < 0.4,
    }


def _unit_convert_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    unit_index = int(params["unit"])
    amount = int(params["amount"])
    factor = _UNITS[unit_index][2]
    value = amount * factor
    wrong = max(0, value + int(params["offset"]))
    if language == "en":
        source_unit, target = _UNITS[unit_index][0], "centimeters"
        if _UNITS[unit_index][0] == "kilometers":
            target = "meters"
        if _UNITS[unit_index][0] == "kilograms":
            target = "grams"
        prompt = f"Convert {amount} {source_unit} to {target}. Answer with an integer."
        rejected = f"That is {wrong} {target}." if params["verbose"] else str(wrong)
    else:
        source_unit, target = _UNITS[unit_index][1], "厘米"
        if _UNITS[unit_index][1] == "千米":
            target = "米"
        if _UNITS[unit_index][1] == "千克":
            target = "克"
        prompt = f"{amount} {source_unit}等于多少{target}？只回答整数。"
        rejected = f"一共是 {wrong} {target}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _sort_numbers_sample(rng: random.Random) -> dict[str, Any]:
    count = rng.randint(3, 5)
    values = [rng.randint(1, 99) for _ in range(count)]
    while len(set(values)) != len(values):
        values = [rng.randint(1, 99) for _ in range(count)]
    shifted = list(values[1:]) + list(values[:1])
    return {
        "values": values,
        "wrong": shifted,
        "verbose": rng.random() < 0.4,
    }


def _sort_numbers_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    values = [int(v) for v in params["values"]]
    wrong = [int(v) for v in params["wrong"]]
    joined = ", ".join(str(v) for v in values)
    prompt = (
        f"Sort these numbers from smallest to largest, separated by commas: {joined}"
        if language == "en"
        else f"把这些数字从小到大排序，用逗号分隔：{joined}"
    )
    chosen = ", ".join(str(v) for v in sorted(values))
    if params["verbose"]:
        rejected = (
            "Sorted: " + ", ".join(str(v) for v in wrong)
            if language == "en"
            else "排序结果：" + ", ".join(str(v) for v in wrong)
        )
    else:
        rejected = ", ".join(str(v) for v in wrong)
    return prompt, chosen, rejected


def _parity_sample(rng: random.Random) -> dict[str, Any]:
    return {"number": rng.randint(10, 999), "verbose": rng.random() < 0.4}


def _parity_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    number = int(params["number"])
    prompt = (
        f"Is {number} odd or even? Reply with one word."
        if language == "en"
        else f"{number} 是奇数还是偶数？只回答一个词。"
    )
    if language == "en":
        chosen = "even" if number % 2 == 0 else "odd"
        rejected = (
            ("It is an " + ("odd" if number % 2 == 0 else "even") + " number.")
            if params["verbose"]
            else ("odd" if number % 2 == 0 else "even")
        )
    else:
        chosen = "偶数" if number % 2 == 0 else "奇数"
        wrong = "奇数" if number % 2 == 0 else "偶数"
        rejected = f"这是{wrong}。" if params["verbose"] else wrong
    return prompt, chosen, rejected


def _make_arith(operation: str) -> _Template:
    template_id = {
        "add": "integer-sum-v1",
        "sub": "integer-difference-v1",
        "mul": "integer-product-v1",
    }[operation]
    return _Template(
        template_id=template_id,
        family="arithmetic",
        verifier="integer",
        sample=lambda rng: _arith_sample(rng, operation),
        render=lambda language, params, rng: _arith_render(language, params, operation),
    )


# --------------------------------------------------------------------------
# 第二批模板：列表、字符串、翻译、格式与更多可验证算术。
#
# A wider template set matters for measurement, not only for variety: the
# group-aware splitter assigns whole template groups by hash, so a corpus with
# only a handful of groups cannot populate train/dev/test reliably. Every template
# below stays programmatically verifiable and keeps parallel en/zh surface forms.
# --------------------------------------------------------------------------


def _list_items(params: dict[str, Any], language: str) -> list[str]:
    return [_pick(_WORDS, int(index), language) for index in params["items"]]


def _word_list_sample(
    rng: random.Random, low: int = 3, high: int = 6
) -> dict[str, Any]:
    count = rng.randint(low, high)
    items = rng.sample(range(len(_WORDS)), count)
    return {"items": items, "verbose": rng.random() < 0.4}


def _join_list(items: Sequence[str], language: str) -> str:
    return ", ".join(items) if language == "en" else "、".join(items)


def _list_position_render(
    language: str, params: dict[str, Any], position: str
) -> tuple[str, str, str]:
    items = _list_items(params, language)
    joined = _join_list(items, language)
    expected = items[0] if position == "first" else items[-1]
    wrong = items[-1] if position == "first" else items[0]
    if language == "en":
        prompt = (
            f"Here is a list: {joined}. Which item comes {position}? "
            "Answer with that item only."
        )
        rejected = f"The {position} item is {wrong}." if params["verbose"] else wrong
    else:
        label = "前" if position == "first" else "后"
        prompt = f"下面是一个列表：{joined}。哪一项排在最{label}？只回答那一项。"
        rejected = (
            f"最{'前' if position == 'first' else '后'}的是{wrong}。"
            if params["verbose"]
            else wrong
        )
    return prompt, expected, rejected


def _list_count_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    items = _list_items(params, language)
    joined = _join_list(items, language)
    value = len(items)
    wrong = value + rng.choice([1, 2, -1])
    if language == "en":
        prompt = f"How many items are in this list: {joined}? Answer with an integer."
        rejected = f"There are {wrong} items." if params["verbose"] else str(wrong)
    else:
        prompt = f"下面这个列表一共有几项：{joined}？只回答一个整数。"
        rejected = f"一共有 {wrong} 项。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _list_contains_sample(rng: random.Random) -> dict[str, Any]:
    params = _word_list_sample(rng)
    if rng.random() < 0.5:
        params["target"] = int(rng.choice(params["items"]))
    else:
        outside = [i for i in range(len(_WORDS)) if i not in params["items"]]
        params["target"] = int(rng.choice(outside))
    return params


def _list_contains_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    items = _list_items(params, language)
    joined = _join_list(items, language)
    target = _pick(_WORDS, int(params["target"]), language)
    present = int(params["target"]) in [int(i) for i in params["items"]]
    if language == "en":
        prompt = (
            f"Here is a list: {joined}. Does it contain {target}? "
            "Answer with one word: yes or no."
        )
        chosen = "yes" if present else "no"
        wrong = "no" if present else "yes"
        rejected = f"I think it is {wrong}." if params["verbose"] else wrong
    else:
        prompt = (
            f"下面是一个列表：{joined}。其中包含{target}吗？只回答一个词：是 或 否。"
        )
        chosen = "是" if present else "否"
        wrong = "否" if present else "是"
        rejected = f"我觉得是{wrong}。" if params["verbose"] else wrong
    return prompt, chosen, rejected


def _string_reverse_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    index = int(params["index"])
    word = _pick(_WORDS, index, language)
    other = _pick(_WORDS, int(params["other"]), language)
    if language == "en":
        prompt = f"Reverse the letters of this word and output nothing else: {word}"
        rejected = (
            f"Reversed, it is {other[::-1]}." if params["verbose"] else other[::-1]
        )
    else:
        prompt = f"把下面这个词的字符顺序颠倒，不要输出其他内容：{word}"
        rejected = f"颠倒后是{other[::-1]}。" if params["verbose"] else other[::-1]
    return prompt, word[::-1], rejected


def _char_count_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    word = _pick(_WORDS, int(params["index"]), language)
    value = len(word)
    wrong = value + rng.choice([1, 2, -1])
    if language == "en":
        prompt = f"How many letters are in the word {word}? Answer with an integer."
        rejected = f"There are {wrong} letters." if params["verbose"] else str(wrong)
    else:
        prompt = f"{word} 这个词有几个字符？只回答一个整数。"
        rejected = f"有 {wrong} 个字符。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _word_translate_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    index = int(params["index"])
    source = _pick(_WORDS, index, language)
    target_language = "zh" if language == "en" else "en"
    # Both renderings translate the same index, so the pair stays parallel.
    chosen = _pick(_WORDS, index, target_language)
    rejected = _pick(_WORDS, int(params["other"]), target_language)
    if language == "en":
        prompt = (
            "Translate this word into Chinese and output only the "
            f"translation: {source}"
        )
    else:
        prompt = f"把下面的词翻译成英文，只输出译文：{source}"
    return prompt, chosen, rejected


def _repeat_n_sample(rng: random.Random) -> dict[str, Any]:
    index = rng.randrange(len(_WORDS))
    return {"index": index, "count": rng.randint(2, 4), "verbose": rng.random() < 0.4}


def _repeat_n_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    word = _pick(_WORDS, int(params["index"]), language)
    count = int(params["count"])
    separator = ", " if language == "en" else "、"
    chosen = separator.join([word] * count)
    wrong = separator.join([word] * max(1, count - 1))
    if language == "en":
        prompt = (
            f"Repeat the word {word} exactly {count} times, separated by commas, "
            "and output nothing else."
        )
        rejected = f"Here it is: {wrong}" if params["verbose"] else wrong
    else:
        prompt = f"把{word}重复 {count} 遍，用顿号分隔，不要输出其他内容。"
        rejected = f"结果是：{wrong}" if params["verbose"] else wrong
    return prompt, chosen, rejected


def _numbers_sample(rng: random.Random, low: int = 3, high: int = 6) -> dict[str, Any]:
    count = rng.randint(low, high)
    values = [rng.randint(1, 99) for _ in range(count)]
    return {"values": values, "verbose": rng.random() < 0.4}


def _numbers_extreme_render(
    language: str, params: dict[str, Any], kind: str
) -> tuple[str, str, str]:
    values = [int(v) for v in params["values"]]
    joined = ", ".join(str(v) for v in values)
    value = max(values) if kind == "max" else min(values)
    wrong = value + 1 if kind == "max" else max(0, value - 1)
    if language == "en":
        label = "largest" if kind == "max" else "smallest"
        prompt = (
            f"Here are numbers: {joined}. Which is the {label}? Answer with an integer."
        )
        rejected = f"The {label} is {wrong}." if params["verbose"] else str(wrong)
    else:
        label = "最大" if kind == "max" else "最小"
        prompt = f"下面这些数字：{joined}。其中{label}的是哪个？只回答一个整数。"
        rejected = f"{label}的是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _sum_list_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    values = [int(v) for v in params["values"]]
    joined = ", ".join(str(v) for v in values)
    value = sum(values)
    wrong = value + rng.choice([1, 2, -1, 3])
    if language == "en":
        prompt = f"Add these numbers: {joined}. Answer with an integer."
        rejected = f"The total is {wrong}." if params["verbose"] else str(wrong)
    else:
        prompt = f"把这些数字相加：{joined}。只回答一个整数。"
        rejected = f"总和是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _odd_count_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    values = [int(v) for v in params["values"]]
    joined = ", ".join(str(v) for v in values)
    value = sum(1 for v in values if v % 2 == 1)
    wrong = max(0, value + rng.choice([1, -1]))
    if language == "en":
        prompt = (
            f"Here are numbers: {joined}. How many of them are odd? "
            "Answer with an integer."
        )
        rejected = (
            f"There are {wrong} odd numbers." if params["verbose"] else str(wrong)
        )
    else:
        prompt = f"下面这些数字：{joined}。其中有几个是奇数？只回答一个整数。"
        rejected = f"有 {wrong} 个奇数。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _simple_arith_render(
    language: str, params: dict[str, Any], kind: str
) -> tuple[str, str, str]:
    left, right = int(params["left"]), int(params["right"])
    value = int(params["value"])
    wrong = int(params["wrong"])
    if language == "en":
        prompts = {
            "div": f"Compute {left} / {right}. Give the integer only.",
            "mod": f"Compute {left} % {right}. Give the integer only.",
            "double": f"Double {left}. Give the integer only.",
            "half": f"Half of {left}. Give the integer only.",
            "square": f"Compute {left} squared. Give the integer only.",
        }
        prompt = prompts[kind]
        rejected = f"I think it is {wrong}." if params["verbose"] else str(wrong)
    else:
        prompts = {
            "div": f"计算 {left} 除以 {right}。只给出整数。",
            "mod": f"计算 {left} 除以 {right} 的余数。只给出整数。",
            "double": f"{left} 的两倍是多少？只给出整数。",
            "half": f"{left} 的一半是多少？只给出整数。",
            "square": f"计算 {left} 的平方。只给出整数。",
        }
        prompt = prompts[kind]
        rejected = f"我觉得是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _make_simple_arith(kind: str) -> _Template:
    def sample(rng: random.Random) -> dict[str, Any]:
        if kind == "div":
            right = rng.randint(2, 20)
            value = rng.randint(2, 40)
            left = right * value
        elif kind == "mod":
            left = rng.randint(20, 999)
            right = rng.randint(3, 25)
            value = left % right
        elif kind == "double":
            left = rng.randint(2, 500)
            right = 2
            value = left * 2
        elif kind == "half":
            value = rng.randint(1, 250)
            left = value * 2
            right = 2
        else:
            left = rng.randint(2, 40)
            right = left
            value = left * left
        wrong = max(0, value + rng.choice([1, 2, -1]))
        return {
            "left": left,
            "right": right,
            "value": value,
            "wrong": wrong,
            "verbose": rng.random() < 0.4,
        }

    names = {
        "div": "integer-quotient-v1",
        "mod": "integer-remainder-v1",
        "double": "integer-double-v1",
        "half": "integer-half-v1",
        "square": "integer-square-v1",
    }
    return _Template(
        template_id=names[kind],
        family="arithmetic",
        verifier="integer",
        sample=sample,
        render=lambda language, params, rng: _simple_arith_render(
            language, params, kind
        ),
    )


def _percent_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    percent, base = int(params["percent"]), int(params["base"])
    value = base * percent // 100
    wrong = max(0, value + int(params["offset"]))
    if language == "en":
        prompt = f"What is {percent}% of {base}? Answer with an integer."
        rejected = f"That is {wrong}." if params["verbose"] else str(wrong)
    else:
        prompt = f"{base} 的 {percent}% 是多少？只回答一个整数。"
        rejected = f"一共是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _time_add_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    hour, delta = int(params["hour"]), int(params["delta"])
    value = (hour + delta) % 24
    wrong = (value + 1) % 24
    if language == "en":
        prompt = (
            f"It is {hour}:00. What hour is it {delta} hours later? "
            "Answer with an integer from 0 to 23."
        )
        rejected = f"It will be {wrong}." if params["verbose"] else str(wrong)
    else:
        prompt = f"现在是 {hour} 点，{delta} 小时后是几点？只回答 0 到 23 的整数。"
        rejected = f"那时是 {wrong} 点。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _digit_stats_render(
    language: str, params: dict[str, Any], kind: str
) -> tuple[str, str, str]:
    number = int(params["number"])
    digits = str(number)
    value = sum(int(ch) for ch in digits) if kind == "sum" else len(digits)
    wrong = value + int(params["offset"])
    if language == "en":
        if kind == "sum":
            prompt = f"Sum the digits of {number}. Answer with an integer."
        else:
            prompt = f"How many digits does {number} have? Answer with an integer."
        rejected = f"It is {wrong}." if params["verbose"] else str(wrong)
    else:
        if kind == "sum":
            prompt = f"把 {number} 的各位数字相加，和是多少？只回答一个整数。"
        else:
            prompt = f"{number} 一共有几位数字？只回答一个整数。"
        rejected = f"是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _make_digit_stats(kind: str) -> _Template:
    def sample(rng: random.Random) -> dict[str, Any]:
        number = rng.randint(100, 999_999)
        return {
            "number": number,
            "offset": rng.choice([1, 2, -1]),
            "verbose": rng.random() < 0.4,
        }

    name = "digit-sum-v1" if kind == "sum" else "digit-count-v1"
    return _Template(
        template_id=name,
        family="arithmetic",
        verifier="integer",
        sample=sample,
        render=lambda language, params, rng: _digit_stats_render(
            language, params, kind
        ),
    )


def _sequence_next_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    start, step = int(params["start"]), int(params["step"])
    terms = [start, start + step, start + 2 * step]
    value = start + 3 * step
    wrong = value + rng.choice([1, step, -1])
    joined = ", ".join(str(v) for v in terms)
    if language == "en":
        prompt = f"What number comes next: {joined}? Answer with an integer."
        rejected = f"Next is {wrong}." if params["verbose"] else str(wrong)
    else:
        prompt = f"下面这组数字的下一个是什么：{joined}？只回答一个整数。"
        rejected = f"下一个是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _next_multiple_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    number = int(params["number"])
    value = ((number + 4) // 5) * 5
    wrong = value + 5 if rng.random() < 0.5 else max(0, value - 5)
    if language == "en":
        prompt = (
            f"What is the smallest multiple of 5 that is at least {number}? "
            "Answer with an integer."
        )
        rejected = f"That would be {wrong}." if params["verbose"] else str(wrong)
    else:
        prompt = f"不小于 {number} 的最小的 5 的倍数是多少？只回答一个整数。"
        rejected = f"应该是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _compare_two_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    left, right = int(params["left"]), int(params["right"])
    value = max(left, right)
    wrong = min(left, right)
    if language == "en":
        prompt = (
            f"Which number is larger, {left} or {right}? Answer with that number only."
        )
        rejected = f"The larger one is {wrong}." if params["verbose"] else str(wrong)
    else:
        prompt = f"{left} 和 {right} 哪个更大？只回答那个数字。"
        rejected = f"更大的是 {wrong}。" if params["verbose"] else str(wrong)
    return prompt, str(value), rejected


def _csv_line_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    values = [int(v) for v in params["values"]]
    chosen = ",".join(str(v) for v in values)
    wrong = ",".join(str(v) for v in reversed(values))
    joined = " ".join(str(v) for v in values)
    if language == "en":
        prompt = (
            f"Write these numbers as one comma-separated line, nothing else: {joined}"
        )
        rejected = f"The line is {wrong}" if params["verbose"] else wrong
    else:
        prompt = f"把这些数字写成一行，用逗号分隔，不要输出其他内容：{joined}"
        rejected = f"这一行是 {wrong}" if params["verbose"] else wrong
    return prompt, chosen, rejected


def _kv_extract_render(
    language: str, params: dict[str, Any], rng: random.Random
) -> tuple[str, str, str]:
    name = _pick(_NAMES, int(params["name"]), language)
    city = _pick(_CITIES, int(params["city"]), language)
    thing = _pick(_WORDS, int(params["item"]), language)
    field = str(params["field"])
    values = {"name": name, "city": city, "item": thing}
    chosen = values[field]
    labels = {
        "name": ("name", "姓名"),
        "city": ("city", "城市"),
        "item": ("item", "物品"),
    }
    if language == "en":
        record = f"name: {name}; city: {city}; item: {thing}"
        prompt = (
            f"Record: {record}. What is the value of {field}? "
            "Answer with that value only."
        )
        rejected = (
            f"The {field} is {values['item'] if field != 'item' else name}."
            if params["verbose"]
            else (values["item"] if field != "item" else name)
        )
    else:
        record = f"姓名：{name}；城市：{city}；物品：{thing}"
        prompt = f"记录：{record}。其中{labels[field][1]}的值是什么？只回答该值。"
        rejected = (
            f"{labels[field][1]}是{values['item'] if field != 'item' else name}。"
            if params["verbose"]
            else (values["item"] if field != "item" else name)
        )
    return prompt, chosen, rejected


def _edge_char_render(
    language: str, params: dict[str, Any], kind: str
) -> tuple[str, str, str]:
    word = _pick(_WORDS, int(params["index"]), language)
    other = _pick(_WORDS, int(params["other"]), language)
    chosen = word[0] if kind == "first" else word[-1]
    wrong = other[0] if kind == "first" else other[-1]
    if language == "en":
        label = "first" if kind == "first" else "last"
        prompt = f"What is the {label} letter of {word}? Answer with that letter only."
        rejected = f"The {label} letter is {wrong}." if params["verbose"] else wrong
    else:
        label = "第一个" if kind == "first" else "最后一个"
        prompt = f"{word} 的{label}字符是什么？只回答那个字符。"
        rejected = f"{label}字符是{wrong}。" if params["verbose"] else wrong
    return prompt, chosen, rejected


_EXTRA_TEMPLATES: tuple[_Template, ...] = (
    _Template(
        template_id="list-first-v1",
        family="qa",
        verifier="exact",
        sample=_word_list_sample,
        render=lambda language, params, rng: _list_position_render(
            language, params, "first"
        ),
    ),
    _Template(
        template_id="list-last-v1",
        family="qa",
        verifier="exact",
        sample=_word_list_sample,
        render=lambda language, params, rng: _list_position_render(
            language, params, "last"
        ),
    ),
    _Template(
        template_id="list-count-v1",
        family="qa",
        verifier="integer",
        sample=_word_list_sample,
        render=_list_count_render,
    ),
    _Template(
        template_id="list-contains-v1",
        family="qa",
        verifier="exact",
        sample=_list_contains_sample,
        render=_list_contains_render,
    ),
    _Template(
        template_id="string-reverse-v1",
        family="format",
        verifier="exact",
        sample=_word_echo_sample,
        render=_string_reverse_render,
    ),
    _Template(
        template_id="char-count-v1",
        family="qa",
        verifier="integer",
        sample=_word_echo_sample,
        render=_char_count_render,
    ),
    _Template(
        template_id="word-translate-v1",
        family="qa",
        verifier="exact",
        sample=_word_echo_sample,
        render=_word_translate_render,
    ),
    _Template(
        template_id="repeat-n-times-v1",
        family="format",
        verifier="exact",
        sample=_repeat_n_sample,
        render=_repeat_n_render,
    ),
    _Template(
        template_id="list-maximum-v1",
        family="qa",
        verifier="integer",
        sample=_numbers_sample,
        render=lambda language, params, rng: _numbers_extreme_render(
            language, params, "max"
        ),
    ),
    _Template(
        template_id="list-minimum-v1",
        family="qa",
        verifier="integer",
        sample=_numbers_sample,
        render=lambda language, params, rng: _numbers_extreme_render(
            language, params, "min"
        ),
    ),
    _Template(
        template_id="list-total-v1",
        family="arithmetic",
        verifier="integer",
        sample=_numbers_sample,
        render=_sum_list_render,
    ),
    _Template(
        template_id="odd-count-v1",
        family="qa",
        verifier="integer",
        sample=_numbers_sample,
        render=_odd_count_render,
    ),
    _make_simple_arith("div"),
    _make_simple_arith("mod"),
    _make_simple_arith("double"),
    _make_simple_arith("half"),
    _make_simple_arith("square"),
    _Template(
        template_id="percent-of-v1",
        family="arithmetic",
        verifier="integer",
        sample=lambda rng: {
            "percent": rng.choice([10, 20, 25, 50]),
            "base": rng.randrange(1, 10) * 100,
            "offset": rng.choice([1, 2, -1]),
            "verbose": rng.random() < 0.4,
        },
        render=_percent_render,
    ),
    _Template(
        template_id="clock-addition-v1",
        family="arithmetic",
        verifier="integer",
        sample=lambda rng: {
            "hour": rng.randint(0, 23),
            "delta": rng.randint(1, 11),
            "verbose": rng.random() < 0.4,
        },
        render=_time_add_render,
    ),
    _make_digit_stats("sum"),
    _make_digit_stats("count"),
    _Template(
        template_id="sequence-next-v1",
        family="arithmetic",
        verifier="integer",
        sample=lambda rng: {
            "start": rng.randint(1, 20),
            "step": rng.randint(2, 9),
            "verbose": rng.random() < 0.4,
        },
        render=_sequence_next_render,
    ),
    _Template(
        template_id="next-multiple-of-five-v1",
        family="arithmetic",
        verifier="integer",
        sample=lambda rng: {
            "number": rng.randint(1, 99),
            "verbose": rng.random() < 0.4,
        },
        render=_next_multiple_render,
    ),
    _Template(
        template_id="compare-two-numbers-v1",
        family="qa",
        verifier="integer",
        sample=lambda rng: {
            "left": rng.randint(1, 999),
            "right": rng.randint(1, 999),
            "verbose": rng.random() < 0.4,
        },
        render=_compare_two_render,
    ),
    _Template(
        template_id="csv-line-v1",
        family="format",
        verifier="exact",
        sample=_numbers_sample,
        render=_csv_line_render,
    ),
    _Template(
        template_id="record-field-v1",
        family="qa",
        verifier="exact",
        sample=lambda rng: {
            "name": rng.randrange(len(_NAMES)),
            "city": rng.randrange(len(_CITIES)),
            "item": rng.randrange(len(_WORDS)),
            "field": rng.choice(["name", "city", "item"]),
            "verbose": rng.random() < 0.4,
        },
        render=_kv_extract_render,
    ),
    _Template(
        template_id="first-character-v1",
        family="format",
        verifier="exact",
        sample=_word_echo_sample,
        render=lambda language, params, rng: _edge_char_render(
            language, params, "first"
        ),
    ),
    _Template(
        template_id="last-character-v1",
        family="format",
        verifier="exact",
        sample=_word_echo_sample,
        render=lambda language, params, rng: _edge_char_render(
            language, params, "last"
        ),
    ),
)


TEMPLATES: tuple[_Template, ...] = (
    _Template(
        template_id="digit-copy-v1",
        family="instruction",
        verifier="exact",
        sample=_digit_copy_sample,
        render=_digit_copy_render,
    ),
    _Template(
        template_id="word-echo-v1",
        family="instruction",
        verifier="exact",
        sample=_word_echo_sample,
        render=_word_echo_render,
    ),
    _make_arith("add"),
    _make_arith("sub"),
    _make_arith("mul"),
    _Template(
        template_id="bracket-copy-v1",
        family="instruction",
        verifier="exact",
        sample=_bracket_sample,
        render=_bracket_render,
    ),
    _Template(
        template_id="json-object-v1",
        family="format",
        verifier="json",
        sample=_json_wrap_sample,
        render=_json_wrap_render,
    ),
    _Template(
        template_id="review-polarity-v1",
        family="qa",
        verifier="exact",
        sample=_sentiment_sample,
        render=_sentiment_render,
    ),
    _Template(
        template_id="resident-city-v1",
        family="qa",
        verifier="exact",
        sample=_where_lives_sample,
        render=_where_lives_render,
    ),
    _Template(
        template_id="container-count-v1",
        family="qa",
        verifier="integer",
        sample=_count_objects_sample,
        render=_count_objects_render,
    ),
    _Template(
        template_id="unit-conversion-v1",
        family="arithmetic",
        verifier="integer",
        sample=_unit_convert_sample,
        render=_unit_convert_render,
    ),
    _Template(
        template_id="number-sort-v1",
        family="format",
        verifier="exact",
        sample=_sort_numbers_sample,
        render=_sort_numbers_render,
    ),
    _Template(
        template_id="parity-tag-v1",
        family="qa",
        verifier="exact",
        sample=_parity_sample,
        render=_parity_render,
    ),
) + _EXTRA_TEMPLATES

_MULTITURN_TEMPLATE_ID = "codeword-recall-v1"


def _multiturn_sample(rng: random.Random) -> dict[str, Any]:
    index = rng.randrange(len(_WORDS))
    filler = rng.choice([i for i in range(len(_WORDS)) if i != index])
    return {"index": index, "filler": filler}


def _multiturn_render(language: str, params: dict[str, Any]) -> list[dict[str, str]]:
    word = _pick(_WORDS, int(params["index"]), language)
    other = _pick(_WORDS, int(params["filler"]), language)
    if language == "en":
        return [
            {
                "role": "user",
                "content": f"Keep this word in mind: {word}. Reply only ok.",
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": "Which word did I ask you to keep? One word only.",
            },
            {"role": "assistant", "content": word},
            {"role": "user", "content": "Say that same word again, nothing else."},
            {"role": "assistant", "content": word},
        ]
    return [
        {"role": "user", "content": f"请记住这个词：{word}。只回答：好的"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "我让你记住的是哪个词？只输出一个词。"},
        {"role": "assistant", "content": other if False else word},
        {"role": "user", "content": "再说一次同一个词，不要其他内容。"},
        {"role": "assistant", "content": word},
    ]


# --------------------------------------------------------------------------
# 数据集构建
# --------------------------------------------------------------------------


def _instance_key(
    template: _Template, params: dict[str, Any], parts: tuple[int, ...]
) -> list[str]:
    """Keys on the rendered surface form, not on the raw sampling parameters.

    Two different parameter sets can render the same text whenever a sampled
    field only affects output that a dataset does not write, for example the
    rejected branch inside SFT or GRPO records. The Data Manifest loaders reject
    duplicate normalized records, so uniqueness has to be enforced on exactly the
    fields each kind actually persists: ``parts`` selects the rendered fields.

    One key is returned **per language**. Loaders dedupe individual records and
    each language becomes its own record, so a collision inside a single language
    is already a duplicate even when the other language still differs.
    """

    keys: list[str] = []
    for language in ("en", "zh"):
        # A throwaway generator keeps key computation independent of draws that
        # the renderer may consume while producing the recorded output.
        full = template.render(language, params, random.Random(0))
        keys.append(
            json.dumps(
                [full[index] for index in parts], ensure_ascii=False, sort_keys=True
            )
        )
    return keys


def _multiturn_key(params: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for language in ("en", "zh"):
        turns = _multiturn_render(language, params)
        keys.append(
            json.dumps(
                [[turn["role"], turn["content"]] for turn in turns],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return keys


def _instances_for(
    template: _Template,
    count: int,
    rng: random.Random,
    parts: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, Any]]:
    """Sample unique instances for one template, bounded by its real variety."""

    return _unique_instances(
        template.sample,
        count,
        rng,
        lambda params: _instance_key(template, params, parts),
    )


def _unique_instances(
    sample: Callable[[random.Random], dict[str, Any]],
    count: int,
    rng: random.Random,
    dedupe_key: Callable[[dict[str, Any]], list[str]],
) -> list[dict[str, Any]]:
    """Draw unique instances until the template's real variety is exhausted."""

    seen: set[str] = set()
    instances: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 40
    while len(instances) < count and attempts < max_attempts:
        attempts += 1
        params = sample(rng)
        keys = dedupe_key(params)
        if any(key in seen for key in keys):
            continue
        seen.update(keys)
        instances.append(params)
    return instances


def build_sft_records(
    seed: int, per_template: int, multiturn: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for template in TEMPLATES:
        # SFT only persists (prompt, answer); the rejected branch is not written.
        instances = _instances_for(template, per_template, rng, parts=(0, 1))
        for order, params in enumerate(instances):
            source_id = f"{template.template_id}-{order:05d}"
            for language in ("en", "zh"):
                prompt, answer, _ = template.render(language, params, rng)
                records.append(
                    {
                        "id": f"{source_id}-{language}",
                        "source_id": source_id,
                        "template_id": template.template_id,
                        "family": template.family,
                        "language": language,
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ],
                    }
                )
    unique_multiturn = _unique_instances(
        _multiturn_sample, multiturn, rng, _multiturn_key
    )
    for order, params in enumerate(unique_multiturn):
        source_id = f"{_MULTITURN_TEMPLATE_ID}-{order:05d}"
        for language in ("en", "zh"):
            records.append(
                {
                    "id": f"{source_id}-{language}",
                    "source_id": source_id,
                    "template_id": _MULTITURN_TEMPLATE_ID,
                    "family": "multiturn",
                    "language": language,
                    "messages": _multiturn_render(language, params),
                }
            )
    return records


def build_dpo_records(seed: int, per_template: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    records: list[dict[str, Any]] = []
    for template in TEMPLATES:
        # DPO persists the whole triple, so all three rendered fields matter.
        instances = _instances_for(template, per_template, rng, parts=(0, 1, 2))
        for order, params in enumerate(instances):
            source_id = f"{template.template_id}-dpo-{order:05d}"
            for language in ("en", "zh"):
                prompt, chosen, rejected = template.render(language, params, rng)
                if chosen.strip() == rejected.strip():
                    continue
                records.append(
                    {
                        "id": f"{source_id}-{language}",
                        "source_id": source_id,
                        "template_id": template.template_id,
                        "family": template.family,
                        "language": language,
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
    return records


def build_grpo_records(seed: int, per_template: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 2)
    records: list[dict[str, Any]] = []
    for template in TEMPLATES:
        # GRPO persists only (prompt, answer); the rejected branch is not written.
        instances = _instances_for(template, per_template, rng, parts=(0, 1))
        for order, params in enumerate(instances):
            source_id = f"{template.template_id}-grpo-{order:05d}"
            for language in ("en", "zh"):
                prompt, answer, _ = template.render(language, params, rng)
                records.append(
                    {
                        "id": f"{source_id}-{language}",
                        "source_id": source_id,
                        "template_id": template.template_id,
                        "family": template.family,
                        "language": language,
                        "prompt": prompt,
                        "answer": answer,
                        "metadata": {"verifier": template.verifier},
                    }
                )
    return records


def write_jsonl(path: Path, records: Sequence[Mapping]) -> None:  # type: ignore[valid-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_posttraining_data.py",
        description="Generate reproducible bilingual SFT, DPO and GRPO source data.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sft-per-template",
        type=int,
        default=300,
        help="unique instances per single-turn template for SFT",
    )
    parser.add_argument(
        "--sft-multiturn",
        type=int,
        default=200,
        help="unique multi-turn conversations for SFT",
    )
    parser.add_argument(
        "--dpo-per-template",
        type=int,
        default=140,
        help="unique preference pairs per template",
    )
    parser.add_argument(
        "--grpo-per-template",
        type=int,
        default=90,
        help="unique verifiable prompts per template",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir: Path = args.output_dir
        sft = build_sft_records(args.seed, args.sft_per_template, args.sft_multiturn)
        dpo = build_dpo_records(args.seed, args.dpo_per_template)
        grpo = build_grpo_records(args.seed, args.grpo_per_template)

        sft_path = output_dir / "sft-source.jsonl"
        dpo_path = output_dir / "dpo-source.jsonl"
        grpo_path = output_dir / "grpo-source.jsonl"
        write_jsonl(sft_path, sft)
        write_jsonl(dpo_path, dpo)
        write_jsonl(grpo_path, grpo)

        card = {
            "dataset_license": DATA_LICENSE,
            "seed": args.seed,
            "generated_by": "scripts/build_posttraining_data.py",
            "records": {
                "sft": len(sft),
                "dpo": len(dpo),
                "grpo": len(grpo),
            },
            "templates": [
                {
                    "template_id": t.template_id,
                    "family": t.family,
                    "verifier": t.verifier,
                }
                for t in TEMPLATES
            ]
            + [
                {
                    "template_id": _MULTITURN_TEMPLATE_ID,
                    "family": "multiturn",
                    "verifier": "exact",
                }
            ],
            "outputs": {
                "sft": str(sft_path),
                "dpo": str(dpo_path),
                "grpo": str(grpo_path),
            },
        }
        card_path = output_dir / "posttraining_data_card.json"
        card_path.write_text(
            json.dumps(card, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (LLMLabError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"sft records:  {len(sft)} -> {sft_path}")
    print(f"dpo records:  {len(dpo)} -> {dpo_path}")
    print(f"grpo records: {len(grpo)} -> {grpo_path}")
    print(f"data card:    {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
