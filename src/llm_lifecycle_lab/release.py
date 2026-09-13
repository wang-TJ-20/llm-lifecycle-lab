"""Build and verify self-contained Native model release artifacts."""

from __future__ import annotations

import json
import shutil
import statistics
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import torch

from llm_lifecycle_lab.contracts import CheckpointMetadata, utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError
from llm_lifecycle_lab.model.native import NativeTransformer, load_native_model_config
from llm_lifecycle_lab.model.protocol import GenerationConfig
from llm_lifecycle_lab.tokenizer import NativeTokenizer
from llm_lifecycle_lab.training.engine import evaluate_objective, resolve_device
from llm_lifecycle_lab.training.stages import PretrainObjective

RELEASE_SCHEMA_VERSION = "1.0"
EXPECTED_CHECKPOINT_FILES = frozenset(
    {
        "checkpoint_metadata.json",
        "trainer_state.json",
        "model/config.json",
        "model/model.pt",
    }
)
FIXED_PROMPTS: tuple[str, ...] = (
    "Once upon a time",
    "The little girl",
    "One day, a boy",
    "There was a dragon",
    "从前",
    "在中国",
    "这个故事",
    "北京是",
)
SMOKE_EVALUATION_RECORDS: tuple[dict[str, str], ...] = (
    {
        "language": "en",
        "text": "Once upon a time, a child found a small red stone.",
    },
    {
        "language": "en",
        "text": "The little dog waited beside the garden gate.",
    },
    {
        "language": "en",
        "text": "A teacher opened the book and began the lesson.",
    },
    {
        "language": "en",
        "text": "Rain fell quietly over the old village.",
    },
    {
        "language": "zh",
        "text": "从前有一个孩子，在河边发现了一块红色的石头。",
    },
    {
        "language": "zh",
        "text": "小狗安静地等在花园门口。",
    },
    {
        "language": "zh",
        "text": "老师打开课本，开始讲今天的内容。",
    },
    {
        "language": "zh",
        "text": "细雨轻轻落在古老的村庄里。",
    },
)


def extract_checkpoint_archive(
    archive_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Extract the exact checkpoint payload while rejecting unsafe members."""

    archive = Path(archive_path)
    target = Path(output_dir)
    if not archive.is_file():
        raise ArtifactError(f"checkpoint archive does not exist: {archive}")
    if target.exists():
        raise ArtifactError(
            f"checkpoint extraction target already exists; refusing to overwrite: "
            f"{target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
    )
    extracted: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            for member in handle.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ArtifactError(
                        f"checkpoint archive contains an unsafe path: {member.name}"
                    )
                normalized = str(path).rstrip("/")
                if member.isdir():
                    (temporary / normalized).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ArtifactError(
                        "checkpoint archive may contain only directories and files: "
                        f"{member.name}"
                    )
                if normalized not in EXPECTED_CHECKPOINT_FILES:
                    raise ArtifactError(
                        f"checkpoint archive contains an unexpected file: {member.name}"
                    )
                source = handle.extractfile(member)
                if source is None:
                    raise ArtifactError(
                        f"cannot read checkpoint archive member: {member.name}"
                    )
                destination = temporary / normalized
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.add(normalized)

        missing = sorted(EXPECTED_CHECKPOINT_FILES - extracted)
        if missing:
            raise ArtifactError(
                "checkpoint archive is incomplete: " + ", ".join(missing)
            )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def build_native_model_release(
    *,
    checkpoint_archive: str | Path,
    tokenizer_dir: str | Path,
    evidence_dir: str | Path,
    reference_spec: str | Path,
    license_path: str | Path,
    output_dir: str | Path,
    repository_id: str,
    model_id: str = "native-60m-base-v1",
    license_id: str = "Apache-2.0",
    device: str = "auto",
) -> dict[str, Any]:
    """Build an immutable ModelScope-ready release directory."""

    archive = Path(checkpoint_archive).resolve()
    tokenizer_source = Path(tokenizer_dir).resolve()
    evidence_source = Path(evidence_dir).resolve()
    reference_source = Path(reference_spec).resolve()
    license_source = Path(license_path).resolve()
    target = Path(output_dir).resolve()
    _require_repository_id(repository_id)
    for path, label in (
        (tokenizer_source, "tokenizer directory"),
        (evidence_source, "evidence directory"),
    ):
        if not path.is_dir():
            raise ArtifactError(f"{label} does not exist: {path}")
    for path, label in (
        (archive, "checkpoint archive"),
        (reference_source, "reference spec"),
        (license_source, "license"),
    ):
        if not path.is_file():
            raise ArtifactError(f"{label} does not exist: {path}")
    if target.exists():
        raise ArtifactError(
            f"release output already exists; refusing to overwrite: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
    )
    checkpoint = temporary / ".checkpoint"
    try:
        extract_checkpoint_archive(archive, checkpoint)
        metadata = _load_checkpoint_metadata(checkpoint)
        trainer_state = _read_json(checkpoint / "trainer_state.json")
        tokenizer = NativeTokenizer.from_directory(tokenizer_source)
        model_config = load_native_model_config(checkpoint / "model/config.json")
        if metadata.tokenizer_sha256 != tokenizer.manifest.content_sha256:
            raise ArtifactError("checkpoint and tokenizer SHA-256 values do not match")
        if int(trainer_state["global_step"]) != metadata.step:
            raise ArtifactError("checkpoint and trainer state steps do not match")

        model = NativeTransformer(model_config)
        model.load(checkpoint / "model")
        parameter_count = model.parameter_count
        del model

        _copy_file(checkpoint / "model/config.json", temporary / "model/config.json")
        _copy_file(checkpoint / "model/model.pt", temporary / "model/model.pt")
        _copy_file(
            checkpoint / "checkpoint_metadata.json",
            temporary / "metadata/checkpoint_metadata.json",
        )
        _copy_file(
            checkpoint / "trainer_state.json",
            temporary / "metadata/trainer_state.json",
        )
        _copy_file(
            tokenizer_source / "tokenizer.json",
            temporary / "tokenizer/tokenizer.json",
        )
        _copy_file(
            tokenizer_source / "tokenizer_manifest.json",
            temporary / "tokenizer/tokenizer_manifest.json",
        )
        _copy_file(reference_source, temporary / "evidence/reference.yaml")
        _copy_evidence(evidence_source, temporary / "evidence")
        _copy_file(license_source, temporary / "LICENSE")
        _write_smoke_records(temporary / "evaluation/smoke.jsonl")

        device_value = str(resolve_device(device))
        generated = generate_release_samples(
            temporary,
            max_new_tokens=48,
            device=device,
            verify_hashes=False,
        )
        smoke_metrics = evaluate_release_smoke(
            temporary,
            device=device,
            verify_hashes=False,
        )
        report = _build_report(
            model_id=model_id,
            repository_id=repository_id,
            license_id=license_id,
            archive=archive,
            metadata=metadata,
            trainer_state=trainer_state,
            parameter_count=parameter_count,
            model_config=asdict(model_config),
            tokenizer=tokenizer,
            evidence_dir=evidence_source,
            generated=generated,
            smoke_metrics=smoke_metrics,
            validation_device=device_value,
        )
        _write_json(temporary / "report.json", report)
        (temporary / "report.md").write_text(
            _render_report_markdown(report),
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            _render_model_card(report),
            encoding="utf-8",
        )
        _write_json(
            temporary / "configuration.json",
            {
                "framework": "pytorch",
                "task": "text-generation",
                "model": {
                    "type": "native-decoder-only",
                    "model_id": model_id,
                },
            },
        )
        shutil.rmtree(checkpoint)
        write_release_manifest(
            temporary,
            {
                "schema_version": RELEASE_SCHEMA_VERSION,
                "model_id": model_id,
                "repository_id": repository_id,
                "license": license_id,
                "created_at": utc_now(),
                "checkpoint_id": metadata.checkpoint_id,
                "checkpoint_step": metadata.step,
                "checkpoint_archive": {
                    "name": archive.name,
                    "sha256": sha256_file(archive),
                    "size_bytes": archive.stat().st_size,
                },
                "provenance_status": "accepted-with-provenance-waiver",
            },
        )
        verify_release_files(temporary)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return _read_json(target / "release_manifest.json")


def write_release_manifest(
    release_dir: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Write payload hashes and a standard SHA256SUMS file."""

    root = Path(release_dir)
    manifest_path = root / "release_manifest.json"
    checksum_path = root / "SHA256SUMS"
    if manifest_path.exists() or checksum_path.exists():
        raise ArtifactError("release manifests already exist; refusing to overwrite")

    files = _release_file_entries(
        root,
        excluded={"release_manifest.json", "SHA256SUMS"},
    )
    manifest = {**dict(metadata), "files": files}
    _write_json(manifest_path, manifest)
    checksum_entries = _release_file_entries(root, excluded={"SHA256SUMS"})
    checksum_path.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in checksum_entries),
        encoding="utf-8",
    )
    return manifest_path


def verify_release_files(release_dir: str | Path) -> dict[str, Any]:
    """Verify every file declared by release_manifest.json and SHA256SUMS."""

    root = Path(release_dir)
    manifest = _read_json(root / "release_manifest.json")
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ArtifactError("release manifest has an unsupported schema version")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ArtifactError("release manifest does not contain files")

    checked = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArtifactError("release manifest contains an invalid file entry")
        relative = _safe_relative_path(str(entry.get("path", "")))
        path = root / relative
        if not path.is_file():
            raise ArtifactError(f"release file is missing: {relative}")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ArtifactError(f"release file size mismatch: {relative}")
        if sha256_file(path) != entry.get("sha256"):
            raise ArtifactError(f"release file SHA-256 mismatch: {relative}")
        checked += 1

    checksum_entries = _parse_sha256sums(root / "SHA256SUMS")
    for relative, expected in checksum_entries.items():
        path = root / _safe_relative_path(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ArtifactError(f"SHA256SUMS verification failed: {relative}")
    return {
        "model_id": manifest["model_id"],
        "repository_id": manifest["repository_id"],
        "files_verified": checked,
        "sha256sums_verified": len(checksum_entries),
    }


def generate_release_samples(
    release_dir: str | Path,
    *,
    prompts: Sequence[str] = FIXED_PROMPTS,
    max_new_tokens: int = 48,
    seed: int = 42,
    device: str = "auto",
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Load a release and run deterministic fixed-prompt continuation."""

    root = Path(release_dir)
    if verify_hashes:
        verify_release_files(root)
    model, tokenizer, torch_device = _load_release_model(root, device=device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        seed=seed,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    results = []
    started_at = time.monotonic()
    generated_tokens = 0
    torch.manual_seed(seed)
    for prompt in prompts:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt, add_bos=True)],
            device=torch_device,
        )
        output = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            config=generation_config,
        )
        generated = output.token_ids[0, input_ids.shape[1] :].tolist()
        generated_tokens += int(output.generated_tokens)
        results.append(
            {
                "prompt": prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_tokens": int(output.generated_tokens),
                "stop_reason": output.stop_reason,
                "continuation": tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                ),
            }
        )
    elapsed = time.monotonic() - started_at
    return {
        "device": str(torch_device),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed if elapsed else 0.0,
        "results": results,
    }


def evaluate_release_smoke(
    release_dir: str | Path,
    *,
    device: str = "auto",
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Evaluate the release on its small bilingual integrity fixture."""

    root = Path(release_dir)
    if verify_hashes:
        verify_release_files(root)
    model, tokenizer, torch_device = _load_release_model(root, device=device)
    records = _read_jsonl(root / "evaluation/smoke.jsonl")
    batch = _collate_smoke_records(records, tokenizer=tokenizer)
    started_at = time.monotonic()
    metrics = evaluate_objective(
        model=model,
        objective=PretrainObjective(),
        batches=[batch],
        device=torch_device,
        dtype=torch.float32,
    )
    elapsed = time.monotonic() - started_at
    return {
        "suite": "release-smoke",
        "sample_count": len(records),
        "device": str(torch_device),
        "elapsed_seconds": elapsed,
        **metrics,
    }


def _load_release_model(
    root: Path,
    *,
    device: str,
) -> tuple[NativeTransformer, NativeTokenizer, torch.device]:
    model_config = load_native_model_config(root / "model/config.json")
    tokenizer = NativeTokenizer.from_directory(root / "tokenizer")
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ArtifactError("release model and tokenizer vocabulary sizes differ")
    model = NativeTransformer(model_config)
    model.load(root / "model")
    torch_device = resolve_device(device)
    model.to_device(torch_device)
    model.set_training(False)
    return model, tokenizer, torch_device


def _collate_smoke_records(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: NativeTokenizer,
) -> dict[str, torch.Tensor]:
    examples: list[dict[str, Any]] = []
    language_map = {"en": 0, "zh": 1}
    for record in records:
        text = str(record["text"])
        language = str(record["language"])
        if language not in language_map:
            raise ContractError(f"unsupported smoke evaluation language: {language}")
        content_tokens = tokenizer.encode(text)
        token_ids = [
            tokenizer.bos_token_id,
            *content_tokens,
            tokenizer.eos_token_id,
        ]
        if len(token_ids) > 128:
            raise ContractError("release smoke record exceeds 128 tokens")
        normalized_bytes = len(tokenizer.decode(content_tokens).encode("utf-8"))
        byte_weights = [0.0] * len(token_ids)
        if content_tokens:
            per_token = normalized_bytes / len(content_tokens)
            byte_weights[1:-1] = [per_token] * len(content_tokens)
        examples.append(
            {
                "token_ids": token_ids,
                "language_ids": [language_map[language]] * len(token_ids),
                "byte_weights": byte_weights,
                "source_bytes": float(normalized_bytes),
            }
        )

    length = max(len(example["token_ids"]) for example in examples)
    input_ids = []
    attention_masks = []
    labels = []
    language_ids = []
    byte_weights = []
    source_bytes = []
    for example in examples:
        valid = len(example["token_ids"])
        padding = length - valid
        input_ids.append(example["token_ids"] + [tokenizer.pad_token_id] * padding)
        attention_masks.append([True] * valid + [False] * padding)
        labels.append(example["token_ids"] + [-100] * padding)
        language_ids.append(example["language_ids"] + [-1] * padding)
        byte_weights.append(example["byte_weights"] + [0.0] * padding)
        source_bytes.append(example["source_bytes"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.bool),
        "labels": torch.tensor(labels, dtype=torch.long),
        "language_ids": torch.tensor(language_ids, dtype=torch.int8),
        "byte_weights": torch.tensor(byte_weights, dtype=torch.float32),
        "source_bytes": torch.tensor(source_bytes, dtype=torch.float32),
    }


def _build_report(
    *,
    model_id: str,
    repository_id: str,
    license_id: str,
    archive: Path,
    metadata: CheckpointMetadata,
    trainer_state: Mapping[str, Any],
    parameter_count: int,
    model_config: Mapping[str, Any],
    tokenizer: NativeTokenizer,
    evidence_dir: Path,
    generated: Mapping[str, Any],
    smoke_metrics: Mapping[str, Any],
    validation_device: str,
) -> dict[str, Any]:
    training = _read_json(evidence_dir / "training_result.json")
    runtime = _read_json(evidence_dir / "runtime_environment.json")
    acceptance = _read_json(evidence_dir / "acceptance.json")
    dev = _read_json(evidence_dir / "evaluations" / "pretrain-dev-step-00005649.json")
    test = _read_json(evidence_dir / "evaluations" / "pretrain-test-step-00005649.json")
    rows = _read_jsonl(evidence_dir / "metrics.jsonl")
    throughput = [
        float(row["tokens_per_second"]) for row in rows if "tokens_per_second" in row
    ]
    peak_memory = max(
        (
            int(row["cuda_max_memory_allocated_bytes"])
            for row in rows
            if "cuda_max_memory_allocated_bytes" in row
        ),
        default=0,
    )
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "model": {
            "model_id": model_id,
            "repository_id": repository_id,
            "license": license_id,
            "parameter_count": parameter_count,
            "architecture": "decoder-only Transformer",
            "config": dict(model_config),
        },
        "checkpoint": {
            **metadata.to_dict(),
            "archive_name": archive.name,
            "archive_sha256": sha256_file(archive),
            "archive_size_bytes": archive.stat().st_size,
            "best_eval_loss": float(trainer_state["best_eval_loss"]),
        },
        "tokenizer": {
            "tokenizer_id": tokenizer.manifest.tokenizer_id,
            "vocab_size": tokenizer.vocab_size,
            "content_sha256": tokenizer.manifest.content_sha256,
            "chat_template_version": tokenizer.manifest.chat_template_version,
        },
        "training": training,
        "evaluation": {
            "dev": dev,
            "test": test,
            "release_smoke": dict(smoke_metrics),
        },
        "performance": {
            "training_device": runtime["accelerator"]["devices"][0]["name"],
            "dtype": "bfloat16",
            "mean_logged_tokens_per_second": statistics.fmean(throughput),
            "min_logged_tokens_per_second": min(throughput),
            "max_logged_tokens_per_second": max(throughput),
            "cuda_peak_memory_bytes": peak_memory,
        },
        "generation": dict(generated),
        "provenance": {
            "status": acceptance["decision"],
            "waived_check": acceptance["waived_check"],
            "source_sha256": runtime["code"]["source_sha256"],
            "git_commit": runtime["code"]["commit"],
            "git_dirty": runtime["code"]["dirty"],
            "status_entries": runtime["code"]["status_entries"],
            "validation_device": validation_device,
            "statement": (
                "The run passed all frozen input, budget, metric, evaluation, "
                "checkpoint, hardware, package, and source-content checks. The "
                "recorded runtime-provenance check failed because training started "
                "from a Git worktree with five uncommitted changes."
            ),
        },
        "data": {
            "languages": ["English", "Chinese"],
            "sources": [
                {
                    "name": "SimpleStories",
                    "records": 100000,
                    "license": "MIT",
                },
                {
                    "name": "Wikimedia Wikipedia zh",
                    "records": 126000,
                    "license": "CC-BY-SA-3.0",
                },
            ],
            "training_supervised_tokens": int(training["tokens_seen"]),
        },
        "limitations": [
            "This is a base continuation model, not an instruction-following model.",
            "English generations are readable but strongly reflect the story domain.",
            "Chinese generation is unstable and often repetitive.",
            "The fixed dev/test reports each cover 64 packed windows, not full splits.",
            "The release inherits one explicitly documented dirty-Git "
            "provenance waiver.",
        ],
    }


def _render_report_markdown(report: Mapping[str, Any]) -> str:
    dev = report["evaluation"]["dev"]["metrics"]
    test = report["evaluation"]["test"]["metrics"]
    performance = report["performance"]
    training = report["training"]
    lines = [
        f"# {report['model']['model_id']} Report",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | ---: |",
        f"| Parameters | {report['model']['parameter_count']:,} |",
        f"| Optimizer steps | {training['global_step']:,} |",
        f"| Supervised tokens | {training['tokens_seen']:,} |",
        f"| Training seconds | {training['elapsed_seconds']:.1f} |",
        f"| Mean logged throughput | "
        f"{performance['mean_logged_tokens_per_second']:.0f} tokens/s |",
        f"| CUDA peak memory | "
        f"{performance['cuda_peak_memory_bytes'] / 1024**3:.3f} GiB |",
        "",
        "## Evaluation",
        "",
        "| Split | Loss | PPL | BPB | English loss | Chinese loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| dev | {dev['eval_loss']:.4f} | {dev['eval_perplexity']:.2f} | "
        f"{dev['eval_bits_per_byte']:.4f} | {dev['eval_en_loss']:.4f} | "
        f"{dev['eval_zh_loss']:.4f} |",
        f"| test | {test['eval_loss']:.4f} | {test['eval_perplexity']:.2f} | "
        f"{test['eval_bits_per_byte']:.4f} | {test['eval_en_loss']:.4f} | "
        f"{test['eval_zh_loss']:.4f} |",
        "",
        "Each published dev/test result covers 64 fixed packed windows.",
        "",
        "## Fixed Continuations",
        "",
    ]
    for item in report["generation"]["results"]:
        continuation = str(item["continuation"]).replace("\n", " ")
        lines.append(f"- `{item['prompt']}` -> {continuation}")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Status: `{report['provenance']['status']}`",
            f"- Source SHA-256: `{report['provenance']['source_sha256']}`",
            f"- Waived check: `{report['provenance']['waived_check']}`",
            "",
            report["provenance"]["statement"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _render_model_card(report: Mapping[str, Any]) -> str:
    dev = report["evaluation"]["dev"]["metrics"]
    repository_id = report["model"]["repository_id"]
    return f"""---
license: apache-2.0
frameworks:
  - Pytorch
tasks:
  - text-generation
language:
  - en
  - zh
---

# Native-60M-Base-v1

Native-60M-Base-v1 是 LLM Lifecycle Lab 从随机权重训练的 62.93M 参数双语
decoder-only Transformer。它用于教学、受控实验和后续训练，不是聊天模型。

## 快速使用

```bash
git clone https://github.com/wang-TJ-20/llm-lifecycle-lab.git
cd llm-lifecycle-lab
python -m pip install -r requirements.txt
python -m pip install -r requirements-modelscope.txt
python scripts/run_published_model.py --model-id {repository_id}
```

该命令下载模型、校验全部 SHA-256、运行固定中英文续写，并执行内置双语
Smoke 评测。完整 Reference 指标来自冻结的 64 个 dev/test 窗口。

## 模型结构

| 项目 | 取值 |
| --- | ---: |
| 参数量 | {report["model"]["parameter_count"]:,} |
| 层数 / Hidden | 8 / 768 |
| Attention heads / KV heads | 12 / 4 |
| 词表 | 16,384 |
| 最大长度 | 512 |
| 结构 | RMSNorm、RoPE、SwiGLU、GQA |

## 训练数据

- SimpleStories：100,000 条，MIT。
- Wikimedia Wikipedia 中文：126,000 条，CC-BY-SA-3.0。
- 共训练 {report["training"]["tokens_seen"]:,} 个监督 token，约 1 epoch。

发布包不包含原始训练数据。使用者应分别遵守数据来源的许可和署名要求。

## 固定评测

| 指标 | dev |
| --- | ---: |
| Loss | {dev["eval_loss"]:.4f} |
| Perplexity | {dev["eval_perplexity"]:.2f} |
| Bits per byte | {dev["eval_bits_per_byte"]:.4f} |
| English loss | {dev["eval_en_loss"]:.4f} |
| Chinese loss | {dev["eval_zh_loss"]:.4f} |

完整 dev/test、吞吐、显存和固定生成结果见 `report.md` 与 `report.json`。

## 能力边界

- 这是 Base 文本续写模型，不具备可靠的指令跟随或多轮对话能力。
- 英文能生成可读但模板化的故事片段。
- 中文生成明显弱于英文，容易退化和重复。
- 60M 参数与当前数据预算不足以提供可靠事实知识。
- 不应用于生产决策、高风险领域或生成事实性结论。

## Provenance

本权重来自已接受的 `native-60m-baseline-v1/step-00005649`。
运行通过固定输入、训练预算、指标、checkpoint、硬件和源码内容检查；
训练启动时 Git 工作区包含 5 个未提交改动，因此保留
`accepted-with-provenance-waiver` 状态。该事实没有被改写为 clean。

源码内容由 SHA-256
`{report["provenance"]["source_sha256"]}` 固定，完整证据位于 `evidence/`。

## License

模型权重与项目代码使用 Apache-2.0。训练数据保留各自原始许可。
"""


def _copy_evidence(source: Path, target: Path) -> None:
    required = (
        "acceptance.json",
        "metrics.jsonl",
        "resolved_config.yaml",
        "run_manifest.json",
        "runtime_environment.json",
        "training_budget.json",
        "training_result.json",
        "evaluations/pretrain-dev-step-00005649.json",
        "evaluations/pretrain-test-step-00005649.json",
    )
    for relative in required:
        _copy_file(source / relative, target / relative)


def _write_smoke_records(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in SMOKE_EVALUATION_RECORDS
        ),
        encoding="utf-8",
    )


def _load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    try:
        return CheckpointMetadata.from_dict(
            _read_json(path / "checkpoint_metadata.json")
        )
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        raise ArtifactError(f"invalid checkpoint metadata under {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSONL file {path}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ArtifactError(f"JSONL rows must be objects: {path}")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot serialize JSON file {path}: {exc}") from exc
    path.write_text(content + "\n", encoding="utf-8")


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ArtifactError(f"release input is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _release_file_entries(
    root: Path,
    *,
    excluded: set[str],
) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def _parse_sha256sums(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ArtifactError(f"cannot read SHA256SUMS: {exc}") from exc
    result: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ArtifactError("SHA256SUMS contains an invalid line")
        _safe_relative_path(relative)
        result[relative] = digest
    if not result:
        raise ArtifactError("SHA256SUMS is empty")
    return result


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ArtifactError(f"release path must be relative and contained: {value}")
    return path


def _require_repository_id(value: str) -> None:
    parts = value.split("/")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise ArtifactError("repository_id must have the form namespace/model-name")
