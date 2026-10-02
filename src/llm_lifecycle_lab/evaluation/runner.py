"""One evaluation protocol shared by Base, SFT and aligned Native checkpoints."""

from __future__ import annotations

import importlib.metadata
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.contracts import utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.corpus import CORPUS_PROTOCOL, scan_corpus
from llm_lifecycle_lab.evaluation.native import NativeEvaluator
from llm_lifecycle_lab.evaluation.report import (
    compare_reports,
    render_report,
    validate_report,
    write_report,
)
from llm_lifecycle_lab.evaluation.suite import (
    SCORING_VERSION,
    digest,
    load_suite,
    metric,
    read_json,
    repeated_ngram_fraction,
    rule_baseline,
    score_response,
)
from llm_lifecycle_lab.exceptions import ContractError
from llm_lifecycle_lab.provenance import capture_runtime_provenance


def evaluate_capabilities(
    *,
    checkpoint: str | Path,
    suite_path: str | Path,
    output: str | Path,
    tokenizer_dir: str | Path | None = None,
    device: str = "cpu",
    prompt_protocol: str = "plain-v1",
    max_new_tokens: int = 32,
    pretrain_manifest: str | Path | None = None,
    baseline_path: str | Path | None = None,
    backend: str = "native",
    base_model: str | Path | None = None,
) -> dict[str, Any]:
    if Path(output).exists():
        raise ContractError("evaluation output exists; choose a new directory")
    if max_new_tokens <= 0 or prompt_protocol not in {"plain-v1", "native-chat-v1"}:
        raise ContractError("invalid generation limit or prompt protocol")
    suite = load_suite(suite_path)
    if backend == "hf":
        if tokenizer_dir is not None:
            raise ContractError(
                "HF evaluation uses the tokenizer bound to its snapshot"
            )
        from llm_lifecycle_lab.interop.evaluation import HFEvaluator

        evaluator = HFEvaluator(checkpoint, base_model=base_model, device=device)
    elif backend == "native":
        if base_model is not None:
            raise ContractError("--base-model is only used by the HF backend")
        evaluator = NativeEvaluator(
            checkpoint, tokenizer_dir=tokenizer_dir, device=device
        )
    else:
        raise ContractError("evaluation backend must be native or hf")
    tokenizer = evaluator.tokenizer
    vocabulary_size = evaluator.config.vocab_size
    runtime = capture_runtime_provenance(workdir=Path.cwd(), device=evaluator.device)
    source_root = Path(__file__).resolve().parents[1]
    sources = sorted(
        p
        for directory in (
            ("evaluation", "model", "tokenizer", "interop")
            if backend == "hf"
            else ("evaluation", "model", "tokenizer")
        )
        for p in (source_root / directory).rglob("*.py")
    )
    protocol = {
        "suite_sha256": digest(suite),
        "suite_id": suite["suite_id"],
        "scoring_version": SCORING_VERSION,
        "implementation_sha256": digest(
            {p.relative_to(source_root).as_posix(): sha256_file(p) for p in sources}
        ),
        "tokenizer_sha256": tokenizer.manifest.content_sha256,
        "prompt_protocol": prompt_protocol,
        "continuation_protocol": "bos-raw-v1",
        "preference_scoring": "mean-conditional-token-logprob-no-end-v1; tie=0.5",
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "seed": 42,
        },
        "dtype": "float32",
        "device": str(evaluator.device),
        "threads": torch.get_num_threads(),
        "packages": runtime["packages"],
        "pretrain_manifest_sha256": (
            sha256_file(pretrain_manifest) if pretrain_manifest else None
        ),
        "corpus_protocol": CORPUS_PROTOCOL,
    }
    if backend == "hf":
        protocol["backend"] = "hf"
        protocol["model_route"] = evaluator.identity["model_route"]
        protocol["hf_packages"] = {
            name: importlib.metadata.version(name)
            for name in ("transformers", "peft", "safetensors", "accelerate")
        }
        protocol["bos_token_id"] = tokenizer.bos_token_id
    baseline = read_json(baseline_path) if baseline_path else None
    if baseline:
        validate_report(baseline)
        if baseline["protocol_sha256"] != digest(protocol):
            raise ContractError(
                "stage-before baseline has a different protocol; rerun it with "
                "the same suite, implementation, tokenizer and generation settings"
            )
    samples = []
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    metrics: dict[str, Any] = {}

    def add(name: str, language: str, value: float, control: dict[str, Any]) -> None:
        for key in (name, f"{name}.{language}"):
            grouped[key].append((value, control))

    rule_zero = {
        "kind": "rule",
        "value": 0.0,
        "description": "Zero-overlap / no-degeneration diagnostic target.",
    }
    corpus_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0])
    for case in suite["cases"]:
        kind, language = case["kind"], case["language"]
        sample = {"id": case["id"], "kind": kind, "language": language}
        if kind == "corpus":
            ids = tokenizer.encode(case["text"])
            byte_count = len(tokenizer.decode(ids).encode("utf-8"))
            if not ids or byte_count == 0:
                raise ContractError("corpus probe has no content tokens or bytes")
            nll = evaluator.nll([tokenizer.bos_token_id], ids)
            for key in ("all", language):
                values = corpus_totals[key]
                values[0] += nll
                values[1] += len(ids)
                values[2] += byte_count
            sample.update(nll=nll, tokens=len(ids), normalized_bytes=byte_count)
        elif kind == "continuation":
            generated = evaluator.generate(
                tokenizer.encode(case["prompt"], add_bos=True),
                max_new_tokens=max_new_tokens,
            )
            add(
                "continuation.repeated_trigram",
                language,
                repeated_ngram_fraction(generated["token_ids"]),
                rule_zero,
            )
            add(
                "continuation.nonempty",
                language,
                float(bool(generated["text"].strip())),
                {
                    "kind": "rule",
                    "value": 0.0,
                    "description": "Always-empty generator.",
                },
            )
            sample.update(prompt=case["prompt"], output=generated)
        elif kind == "preference":
            prefix = evaluator.prompt_ids(
                [{"role": "user", "content": case["prompt"]}], prompt_protocol
            )
            scores = {}
            for label in ("chosen", "rejected"):
                target = tokenizer.encode(case[label])
                scores[label] = -evaluator.nll(prefix, target) / len(target)
            margin = scores["chosen"] - scores["rejected"]
            accuracy = 0.5 if abs(margin) <= 1e-8 else float(margin > 0)
            add(
                "preference.accuracy",
                language,
                accuracy,
                {
                    "kind": "analytic-random",
                    "value": 0.5,
                    "description": "Uniform pair ordering; ties receive half credit.",
                },
            )
            sample.update(scores=scores, margin=margin, accuracy=accuracy)
        else:
            turns = case["turns"] if kind == "multiturn" else [case]
            messages: list[dict[str, str]] = []
            results = []
            for turn in turns:
                messages.append({"role": "user", "content": turn["prompt"]})
                generated = evaluator.generate(
                    evaluator.prompt_ids(messages, prompt_protocol),
                    max_new_tokens=max_new_tokens,
                    protocol=prompt_protocol,
                )
                value = score_response(generated["text"], turn["rule"])
                control = rule_baseline(turn["rule"])
                add(f"{kind}.success", language, value, control)
                # This reward is the deterministic task verifier, not an RL reward
                # model and not a claim of human preference or safety alignment.
                add("verifiable.reward", language, value, control)
                results.append(
                    {
                        "prompt": turn["prompt"],
                        "output": generated,
                        "score": value,
                        "rule": turn["rule"],
                    }
                )
                messages.append({"role": "assistant", "content": generated["text"]})
            if kind == "multiturn":
                add(
                    "multiturn.all_correct",
                    language,
                    float(all(x["score"] == 1 for x in results)),
                    {
                        "kind": "rule",
                        "value": 0.0,
                        "description": "Always-empty responses across all turns.",
                    },
                )
            sample["turns"] = results
        samples.append(sample)
    for language, (nll, tokens, byte_count) in corpus_totals.items():
        for name, divisor in (("loss", tokens), ("bpb", byte_count * math.log(2))):
            metrics[f"corpus.{name}.{language}"] = metric(
                nll / divisor,
                count=int(tokens if name == "loss" else byte_count),
                direction="lower",
                baseline={
                    "kind": "analytic-random",
                    "value": tokens * math.log(vocabulary_size) / divisor,
                    "description": "Uniform probability over tokenizer vocabulary.",
                },
            )

    audit: dict[str, Any] = {"status": "not-checked", "reason": "No pretrain manifest."}
    if pretrain_manifest:
        audit = scan_corpus(pretrain_manifest, suite=suite, tokenizer=tokenizer)
        for name, numerator, denominator in (
            (
                "overlap.query_fraction",
                len(audit["matched_query_ids"]),
                audit["eligible_queries"],
            ),
            (
                "overlap.heldout_fraction",
                audit["heldout_exact_train_matches"],
                audit["heldout_records"],
            ),
        ):
            metrics[name] = metric(
                numerator / denominator if denominator else None,
                count=denominator,
                direction="lower",
                baseline=rule_zero,
                status="ok" if denominator else "no-eligible-samples",
            )
        for probe in audit.pop("probes"):
            if probe["split"] == "test" and probe["exact_train_overlap"]:
                continue
            generated = evaluator.generate(
                probe["prefix"], max_new_tokens=CORPUS_PROTOCOL["memory_suffix_tokens"]
            )
            correct = float(generated["token_ids"] == probe["target"])
            add(
                f"memory.{probe['split']}.suffix_exact",
                probe["language"],
                correct,
                {
                    "kind": "analytic-random",
                    "value": math.exp(
                        -len(probe["target"]) * math.log(vocabulary_size)
                    ),
                    "description": "Uniform independent tokens, fixed suffix length.",
                },
            )
            samples.append(
                {
                    "id": probe["id"],
                    "kind": "memory",
                    "language": probe["language"],
                    "split": probe["split"],
                    "text_sha256": probe["text_sha256"],
                    "source_id_sha256": probe["source_id_sha256"],
                    "target_sha256": digest({"tokens": probe["target"]}),
                    "score": correct,
                }
            )
    else:
        for name in ("overlap.query_fraction", "overlap.heldout_fraction"):
            metrics[name] = metric(
                None,
                count=0,
                direction="lower",
                baseline=rule_zero,
                status="not-checked",
            )
    for name, entries in sorted(grouped.items()):
        values, controls = zip(*entries, strict=True)
        control_kinds = {x["kind"] for x in controls}
        metrics[name] = metric(
            statistics.fmean(values),
            count=len(values),
            direction=(
                "diagnostic"
                if name.startswith("memory.")
                else "lower"
                if "repeated_trigram" in name
                else "higher"
            ),
            baseline={
                "kind": next(iter(control_kinds))
                if len(control_kinds) == 1
                else "mixed",
                "value": statistics.fmean(x["value"] for x in controls),
                "description": sorted({x["description"] for x in controls}),
            },
        )
    # Stable absent buckets never look like a score of zero.
    for split in ("train", "test"):
        for language in (None, "en", "zh"):
            key = f"memory.{split}.suffix_exact" + (f".{language}" if language else "")
            if key not in metrics:
                metrics[key] = metric(
                    None,
                    count=0,
                    direction="diagnostic",
                    status="not-checked",
                    baseline={
                        "kind": "analytic-random",
                        "value": vocabulary_size
                        ** -CORPUS_PROTOCOL["memory_suffix_tokens"],
                        "description": "Uniform independent tokens.",
                    },
                )
    report = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "model": evaluator.identity,
        "protocol": protocol,
        "protocol_sha256": digest(protocol),
        "suite": suite,
        "metrics": dict(sorted(metrics.items())),
        "samples": samples,
        "corpus_audit": audit,
        "runtime": runtime,
        "limitations": [
            "手写双语诊断集样本很少；不得作为通用语言能力或生产可用性的证明。",
            "BPB 使用当前 Tokenizer 解码后文本的 UTF-8 字节数，不计 EOS；"
            "不是旧 Reference 的 Packing BPB。",
            "续写重复率只度量退化，不度量事实性、流畅性或帮助程度。",
            "多轮输入使用模型自己的历史回答，不注入标准答案；空回复/控制串仅为续轮转义。",
            "偏好分数是回答 token 平均条件 log-prob，不含结束符；"
            "不是 reward model 准确率。",
            "verifiable.reward 是任务规则的 0/1 奖励，不代表安全性或人类偏好对齐。",
            "空输出规则基线是下界；随机基线是解析期望，不是随机初始化模型实测。",
            "记忆探针对固定语料前缀做精确后缀匹配；不命中不能证明模型没有记忆。",
            "重合检查只覆盖指定语料的规范化文本/子串，不能证明不存在泄漏。",
        ],
    }
    if baseline:
        report["comparison"] = compare_reports([baseline, report])
        report["baseline_report_sha256"] = sha256_file(baseline_path)
    write_report(output, report, render_report(report))
    return report
