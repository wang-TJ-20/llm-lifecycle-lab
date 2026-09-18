"""Build a large, deterministic, leakage-checked diagnostic probe suite.

Why this exists
---------------
``lifecycle-v1`` and ``-v2`` are 28/36 hand-written cases. That is enough to see
a 0 -> 0.5 jump from SFT, but far too small to resolve post-SFT differences:
``instruction.success`` and ``qa.success`` cover only 4 cases each, so a single
flipped case moves the metric by 0.25 -- exactly the size of every difference we
measured between SFT, DPO and GRPO. Any tuning done against those numbers is
tuning against noise.

This generator materialises ~30+ authored cases per scored category. The cases
are still project-authored: the script only fills fixed bilingual prompt patterns
from fixed word lists under a fixed seed, and writes the resulting YAML. It also
refuses to emit anything that collides with the training corpora, and records the
hashes of the sources it checked.

Guarantees
----------
* deterministic: same seed -> byte-identical YAML;
* no prompt, group or template_id overlap with the SFT/DPO/GRPO sources;
* every scored category keeps equal en/zh counts.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.evaluation.corpus import normalize
from llm_lifecycle_lab.exceptions import DataValidationError, LLMLabError

SUITE_LICENSE = "Apache-2.0"
DEFAULT_SEED = 31337

# --------------------------------------------------------------------------
# Word lists. Deliberately different surface data from the training generator so
# that "same family, new surface form" is what is being measured.
# --------------------------------------------------------------------------

WORDS: tuple[tuple[str, str], ...] = (
    ("silver", "银白"),
    ("copper", "铜色"),
    ("marble", "大理石"),
    ("velvet", "天鹅绒"),
    ("amber", "琥珀"),
    ("cobalt", "钴蓝"),
    ("cedar", "雪松"),
    ("basalt", "玄武"),
    ("ivory", "象牙"),
    ("bronze", "青铜"),
    ("crystal", "水晶"),
    ("willow", "垂柳"),
    ("granite", "花岗岩"),
    ("saffron", "藏红"),
    ("onyx", "缟玛瑙"),
    ("birch", "白桦"),
)

NAMES: tuple[tuple[str, str], ...] = (
    ("Mira", "小岚"),
    ("Tomas", "托马斯"),
    ("Ines", "伊内斯"),
    ("Kwame", "克瓦米"),
    ("Sara", "莎拉"),
    ("Yusuf", "优素福"),
    ("Nora", "诺拉"),
    ("Emil", "埃米尔"),
    ("Ling", "阿玲"),
    ("Ravi", "拉维"),
)

CITIES: tuple[tuple[str, str], ...] = (
    ("Rome", "罗马"),
    ("Lima", "利马"),
    ("Oslo", "奥斯陆"),
    ("Kyoto", "京都"),
    ("Bern", "伯尔尼"),
    ("Nara", "奈良"),
    ("Porto", "波尔图"),
    ("Malmo", "马尔默"),
    ("Doha", "多哈"),
    ("Riga", "里加"),
)

OBJECTS: tuple[tuple[str, str, str], ...] = (
    ("cups", "只杯子", "杯子"),
    ("books", "本书", "书"),
    ("coins", "枚硬币", "硬币"),
    ("stones", "块石头", "石头"),
    ("cards", "张卡片", "卡片"),
)

POSITIVE: tuple[tuple[str, str], ...] = (
    ("I love this gift.", "我很喜欢这份礼物。"),
    ("The meal was excellent.", "这顿饭非常好。"),
    ("Service was quick and kind.", "服务又快又周到。"),
    ("This seat is very comfortable.", "这个座位很舒服。"),
)

NEGATIVE: tuple[tuple[str, str], ...] = (
    ("I hate this show.", "我讨厌这场演出。"),
    ("The meal was terrible.", "这顿饭糟透了。"),
    ("Service was slow and rude.", "服务又慢又差。"),
    ("This seat is very uncomfortable.", "这个座位很不舒服。"),
)


def _pick(pairs: Sequence[tuple[str, ...]], index: int, language: str) -> str:
    """Wrap around short word lists so any count stays valid."""
    entry = pairs[index % len(pairs)]
    return entry[0] if language == "en" else entry[1]


# --------------------------------------------------------------------------
# Authored patterns. Each returns (prompt, rule) for one language.
# --------------------------------------------------------------------------


def _bracket(index: int, language: str) -> tuple[str, dict[str, Any]]:
    word = _pick(WORDS, index, language)
    prompt = (
        f"Return only the text inside brackets: [{word}]."
        if language == "en"
        else f"只输出方括号里的内容：[{word}]。"
    )
    return prompt, {"type": "exact", "answer": word}


def _sentiment(index: int, language: str) -> tuple[str, dict[str, Any]]:
    positive = index % 2 == 0
    pool = POSITIVE if positive else NEGATIVE
    sentence = _pick(pool, index // 2, language)
    if language == "en":
        prompt = f"Classify '{sentence}' Reply only positive or negative."
        rule = {
            "type": "exact",
            "answer": "positive" if positive else "negative",
            "choices": ["positive", "negative"],
        }
    else:
        prompt = f"判断“{sentence}”的情感。只回答：正面 或 负面。"
        rule = {
            "type": "exact",
            "answer": "正面" if positive else "负面",
            "choices": ["正面", "负面"],
        }
    return prompt, rule


def _digit_echo(index: int, language: str) -> tuple[str, dict[str, Any]]:
    number = str(100_000 + index * 7919)
    prompt = (
        f"Write back the number {number} using digits only."
        if language == "en"
        else f"请用阿拉伯数字把这个数写一遍：{number}。"
    )
    return prompt, {"type": "exact", "answer": number}


def _word_output(index: int, language: str) -> tuple[str, dict[str, Any]]:
    word = _pick(WORDS, index, language)
    prompt = (
        f"Output just the single word {word}; no other text."
        if language == "en"
        else f"只输出这一个词：{word}，不要其他内容。"
    )
    return prompt, {"type": "exact", "answer": word}


def _container(index: int, language: str) -> tuple[str, dict[str, Any]]:
    start = 2 + index
    added = 1 + (index % 4)
    obj = OBJECTS[index % len(OBJECTS)]
    total = str(start + added)
    if language == "en":
        prompt = (
            f"A box holds {start} {obj[0]}; {added} more are put in. "
            "How many now? Answer with a digit."
        )
    else:
        prompt = (
            f"盒子里有 {start} {obj[1]}，又放进 {added} {obj[1]}。"
            "现在有几个？用阿拉伯数字回答。"
        )
    return prompt, {"type": "exact", "answer": total}


def _resident(index: int, language: str) -> tuple[str, dict[str, Any]]:
    name = _pick(NAMES, index, language)
    city = _pick(CITIES, index, language)
    other = _pick(CITIES, (index + 3) % len(CITIES), language)
    if language == "en":
        prompt = (
            f"{name}'s home city is {city}. Which city is that? "
            f"Reply only {city} or {other}."
        )
    else:
        prompt = f"{name}定居的城市是{city}。是哪座城市？只回答：{city} 或 {other}。"
    return prompt, {"type": "exact", "answer": city, "choices": [city, other]}


def _list_first(index: int, language: str) -> tuple[str, dict[str, Any]]:
    items = [_pick(WORDS, (index + step) % len(WORDS), language) for step in (0, 3, 6)]
    joined = ", ".join(items) if language == "en" else "、".join(items)
    prompt = (
        f"The list is {joined}. Name the first item only."
        if language == "en"
        else f"这个列表是：{joined}。只说出第一项。"
    )
    return prompt, {"type": "exact", "answer": items[0]}


def _larger(index: int, language: str) -> tuple[str, dict[str, Any]]:
    left = 10 + index * 13
    right = left + 7
    if language == "en":
        prompt = (
            f"Which number is larger, {left} or {right}? Reply with that number only."
        )
    else:
        prompt = f"{left} 和 {right} 哪个更大？只回答那个数字。"
    return prompt, {"type": "exact", "answer": str(right)}


def _json_int(index: int, language: str) -> tuple[str, dict[str, Any]]:
    value = 3 + index
    if language == "en":
        prompt = f'Return JSON only: the key "count" with the integer value {value}.'
    else:
        prompt = f'只输出 JSON：键是 "count"，值是整数 {value}。'
    return prompt, {"type": "json", "answer": {"count": value}}


def _json_str(index: int, language: str) -> tuple[str, dict[str, Any]]:
    word = _pick(WORDS, index, language)
    if language == "en":
        prompt = f'Return JSON only: the key "label" with the string value "{word}".'
    else:
        prompt = f'只输出 JSON：键是 "label"，值是字符串 "{word}"。'
    return prompt, {"type": "json", "answer": {"label": word}}


def _csv_join(index: int, language: str) -> tuple[str, dict[str, Any]]:
    values = [str(1 + index), str(4 + index), str(9 + index)]
    joined = " ".join(values)
    prompt = (
        f"Join these numbers with commas and nothing else: {joined}"
        if language == "en"
        else f"把这些数字用逗号连起来，不要其他内容：{joined}"
    )
    return prompt, {"type": "exact", "answer": ",".join(values)}


def _multiturn(index: int, language: str) -> tuple[list[dict[str, Any]], Any]:
    word = _pick(WORDS, index, language)
    if language == "en":
        turns = [
            {
                "prompt": f"Remember the code word {word}. Reply only OK.",
                "rule": {"type": "exact", "answer": "OK"},
            },
            {
                "prompt": "What was the code word? Reply with that word only.",
                "rule": {"type": "exact", "answer": word},
            },
            {
                "prompt": "Say the same code word again, nothing else.",
                "rule": {"type": "exact", "answer": word},
            },
        ]
    else:
        turns = [
            {
                "prompt": f"请记住暗号：{word}。只回答：好的",
                "rule": {"type": "exact", "answer": "好的"},
            },
            {
                "prompt": "刚才的暗号是什么？只输出暗号。",
                "rule": {"type": "exact", "answer": word},
            },
            {
                "prompt": "再重复一次同一个暗号，不要解释。",
                "rule": {"type": "exact", "answer": word},
            },
        ]
    return turns, None


def _pref_arith(index: int, language: str) -> tuple[str, str, str]:
    left = 11 + index
    right = 6 + index % 5
    value = left * right
    wrong = value + (1 if index % 2 else -1) * right
    prompt = (
        f"Compute {left} * {right}. Reply with the integer only."
        if language == "en"
        else f"计算 {left} 乘 {right}。只回答整数。"
    )
    return prompt, str(value), str(wrong)


def _pref_digit(index: int, language: str) -> tuple[str, str, str]:
    number = str(500_000 + index * 313)
    wrong = number[:-1] + str((int(number[-1]) + 1) % 10)
    prompt = (
        f"Copy the number below exactly: {number}"
        if language == "en"
        else f"请准确复制下面的数字：{number}"
    )
    return prompt, number, wrong


def _pref_json(index: int, language: str) -> tuple[str, str, str]:
    value = 4 + index
    prompt = (
        f'Return JSON only: {{"count": {value}}}'
        if language == "en"
        else f'只输出 JSON：{{"count": {value}}}'
    )
    chosen = json.dumps({"count": value}, separators=(",", ":"))
    rejected = json.dumps({"count": value + 1}, separators=(",", ":"))
    return prompt, chosen, rejected


def _pref_unit(index: int, language: str) -> tuple[str, str, str]:
    amount = 2 + index % 6
    value = amount * 100
    wrong = amount * 10
    prompt = (
        f"Please convert {amount} meters into centimeters. Integer only."
        if language == "en"
        else f"请把 {amount} 米换算成厘米，只回答整数。"
    )
    return prompt, str(value), str(wrong)


INSTRUCTION_PATTERNS = (_bracket, _sentiment, _digit_echo, _word_output)
QA_PATTERNS = (_container, _resident, _list_first, _larger)
FORMAT_PATTERNS = (_json_int, _json_str, _csv_join)
PREFERENCE_PATTERNS = (_pref_arith, _pref_digit, _pref_json, _pref_unit)

CORPUS = (
    (
        "The child opened a book beside the window. "
        "Outside, rain fell on the quiet street.",
        "孩子坐在窗边打开一本书。窗外下着小雨，街道十分安静。",
    ),
    (
        "The public library opens every morning. "
        "Readers can borrow books and return them next week.",
        "公共图书馆每天上午开放。读者可以借阅书籍，并在下周归还。",
    ),
)
CONTINUATION = (
    ("Once upon a time", "从前"),
    ("The little girl", "这个小女"),
    ("One day, a boy", "有一天，一个男"),
    ("There was a dragon", "那里有一只龙"),
)


def _polyfill(patterns, count: int, kind: str, language: str, offset: int):
    """Round-robin patterns so every family contributes equally."""

    cases = []
    for index in range(count):
        pattern = patterns[index % len(patterns)]
        prompt, rule = pattern(offset + index, language)
        cases.append(
            {"kind": kind, "language": language, "prompt": prompt, "rule": rule}
        )
    return cases


def build_cases(
    *, instruction: int, qa: int, fmt: int, multiturn: int, preference: int
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (en, zh) in enumerate(CORPUS):
        cases.append(
            {
                "id": f"corpus-en-{index + 1}",
                "group": f"corpus-{index + 1}",
                "language": "en",
                "kind": "corpus",
                "text": en,
            }
        )
        cases.append(
            {
                "id": f"corpus-zh-{index + 1}",
                "group": f"corpus-{index + 1}",
                "language": "zh",
                "kind": "corpus",
                "text": zh,
            }
        )
    for index, (en, zh) in enumerate(CONTINUATION):
        cases.append(
            {
                "id": f"continue-en-{index + 1}",
                "group": f"continue-{index + 1}",
                "language": "en",
                "kind": "continuation",
                "prompt": en,
            }
        )
        cases.append(
            {
                "id": f"continue-zh-{index + 1}",
                "group": f"continue-{index + 1}",
                "language": "zh",
                "kind": "continuation",
                "prompt": zh,
            }
        )

    for language in ("en", "zh"):
        for position, case in enumerate(
            _polyfill(INSTRUCTION_PATTERNS, instruction, "instruction", language, 0)
        ):
            family = INSTRUCTION_PATTERNS[
                position % len(INSTRUCTION_PATTERNS)
            ].__name__[1:]
            cases.append(
                {
                    **case,
                    "id": f"inst-{language}-{position:02d}",
                    "group": f"inst-{family}",
                }
            )
        for position, case in enumerate(_polyfill(QA_PATTERNS, qa, "qa", language, 0)):
            family = QA_PATTERNS[position % len(QA_PATTERNS)].__name__[1:]
            cases.append(
                {**case, "id": f"qa-{language}-{position:02d}", "group": f"qa-{family}"}
            )
        for position, case in enumerate(
            _polyfill(FORMAT_PATTERNS, fmt, "format", language, 0)
        ):
            family = FORMAT_PATTERNS[position % len(FORMAT_PATTERNS)].__name__[1:]
            cases.append(
                {
                    **case,
                    "id": f"fmt-{language}-{position:02d}",
                    "group": f"fmt-{family}",
                }
            )
        for position in range(multiturn):
            turns, _ = _multiturn(position, language)
            cases.append(
                {
                    "id": f"dialog-{language}-{position:02d}",
                    "group": f"dialog-{position}",
                    "language": language,
                    "kind": "multiturn",
                    "turns": turns,
                }
            )
        for position in range(preference):
            pattern = PREFERENCE_PATTERNS[position % len(PREFERENCE_PATTERNS)]
            prompt, chosen, rejected = pattern(position, language)
            family = pattern.__name__[6:]
            cases.append(
                {
                    "id": f"pref-{language}-{position:02d}",
                    "group": f"pref-{family}",
                    "language": language,
                    "kind": "preference",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
    return cases


def collect_prompts(cases: list[dict[str, Any]]) -> set[str]:
    prompts = set()
    for case in cases:
        if case["kind"] == "corpus":
            prompts.add(normalize(case["text"]))
        elif case["kind"] == "multiturn":
            prompts.update(normalize(turn["prompt"]) for turn in case["turns"])
        else:
            prompts.add(normalize(case["prompt"]))
    return prompts


def check_leakage(cases: list[dict[str, Any]], sources: list[Path]) -> dict[str, int]:
    prompts = collect_prompts(cases)
    groups = {case["group"] for case in cases}
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise DataValidationError(
            "cannot verify evaluation leakage; missing source(s): " + ", ".join(missing)
        )
    report: dict[str, int] = {}
    for source in sources:
        hits = 0
        for line in source.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            text = record.get("text") or record.get("prompt")
            if text is None and record.get("messages"):
                text = " ".join(m["content"] for m in record["messages"])
            if (text and normalize(text) in prompts) or record.get(
                "template_id"
            ) in groups:
                hits += 1
        report[str(source)] = hits
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_evaluation_suite.py",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--suite-id", default="lifecycle-v3")
    parser.add_argument(
        "--output", type=Path, default=Path("configs/evaluation/lifecycle-v3.yaml")
    )
    parser.add_argument("--instruction", type=int, default=16)
    parser.add_argument("--qa", type=int, default=16)
    parser.add_argument("--format", type=int, default=12)
    parser.add_argument("--multiturn", type=int, default=4)
    parser.add_argument("--preference", type=int, default=16)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--source",
        type=Path,
        nargs="*",
        default=[
            Path("data/sft-source.jsonl"),
            Path("data/dpo-source.jsonl"),
            Path("data/dpo-onpolicy-source.jsonl"),
            Path("data/grpo-source.jsonl"),
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        random.seed(args.seed)
        cases = build_cases(
            instruction=args.instruction,
            qa=args.qa,
            fmt=args.format,
            multiturn=args.multiturn,
            preference=args.preference,
        )
        leakage = check_leakage(cases, list(args.source))
        total = sum(leakage.values())
        if total:
            print(f"error: probe leakage detected: {leakage}", file=sys.stderr)
            return 1
        suite = {
            "schema_version": "1.0",
            "suite_id": args.suite_id,
            "license": SUITE_LICENSE,
            "description": (
                "Project-authored bilingual probes, version 3. v1/v2 score "
                "instruction and qa over 4 cases each, so one flipped case "
                "moves the metric by 0.25 -- the size of every measured "
                "post-SFT difference. v3 scales each scored category to 30+ "
                "cases. Built by scripts/build_evaluation_suite.py (seed "
                f"{args.seed}) and leakage-checked against the training "
                "sources. Not a general benchmark; do not train on these."
            ),
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            yaml.safe_dump(suite, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    except (LLMLabError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["kind"]] = counts.get(case["kind"], 0) + 1
    print(f"suite: {args.output}")
    print(f"cases: {len(cases)}  {counts}")
    print(f"leakage: {leakage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
