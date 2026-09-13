"""Reproducible CPU latency, throughput, RSS, and dynamic-INT8 diagnostics."""

from __future__ import annotations

import importlib
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.contracts import utc_now
from llm_lifecycle_lab.data.fingerprint import sha256_file
from llm_lifecycle_lab.evaluation.suite import digest
from llm_lifecycle_lab.exceptions import ArtifactError, ContractError
from llm_lifecycle_lab.interop.assets import require_hf, verify_hf_assets

PROMPTS = ("Once upon a time", "从前", "The little girl", "北京是")
QUANTIZED_ENGINE_PREFERENCE = ("x86", "fbgemm", "onednn", "qnnpack")


def require_psutil():
    try:
        return importlib.import_module("psutil")
    except ModuleNotFoundError as exc:
        raise ArtifactError(
            "Install optional HF dependencies: "
            "python -m pip install -r requirements-hf.txt"
        ) from exc


def generation_inputs(tokenizer, text: str) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt")
    if "input_ids" not in encoded:
        raise ContractError("tokenizer did not return input_ids")
    return {
        key: value
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }


def select_quantized_engine() -> str:
    supported = set(torch.backends.quantized.supported_engines)
    current = torch.backends.quantized.engine
    if current != "none" and current in supported:
        return current
    for engine in QUANTIZED_ENGINE_PREFERENCE:
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise ContractError("this PyTorch build has no supported quantized CPU engine")


def serialized_state_size(model) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(model.state_dict(), handle.name)
        return Path(handle.name).stat().st_size


@torch.inference_mode()
def measure_model(
    model,
    tokenizer,
    *,
    max_new_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    if repeats <= 0 or max_new_tokens <= 0:
        raise ContractError("benchmark repeats and max_new_tokens must be positive")
    psutil = require_psutil()
    process = psutil.Process()
    samples = []
    # Warm up kernels and caches; the warm-up is not included in measurements.
    warmup = generation_inputs(tokenizer, PROMPTS[0])
    model.generate(
        **warmup,
        do_sample=False,
        max_new_tokens=min(2, max_new_tokens),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    for repeat in range(repeats):
        for prompt in PROMPTS:
            inputs = generation_inputs(tokenizer, prompt)
            rss_before = process.memory_info().rss
            started = time.perf_counter()
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            elapsed = time.perf_counter() - started
            generated = output[0, inputs["input_ids"].shape[1] :].tolist()
            samples.append(
                {
                    "repeat": repeat,
                    "prompt": prompt,
                    "elapsed_seconds": elapsed,
                    "generated_tokens": len(generated),
                    "tokens_per_second": len(generated) / max(elapsed, 1e-9),
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": process.memory_info().rss,
                    "output_token_ids": generated,
                }
            )
    return {
        "state_dict_bytes": serialized_state_size(model),
        "median_seconds": statistics.median(x["elapsed_seconds"] for x in samples),
        "median_tokens_per_second": statistics.median(
            x["tokens_per_second"] for x in samples
        ),
        "peak_observed_rss_bytes": max(
            max(x["rss_before_bytes"], x["rss_after_bytes"]) for x in samples
        ),
        "samples": samples,
    }


def benchmark_hf_cpu(
    model_dir: str | Path,
    *,
    max_new_tokens: int = 16,
    repeats: int = 2,
    threads: int = 1,
    dynamic_int8: bool = True,
) -> dict[str, Any]:
    if threads <= 0:
        raise ContractError("benchmark threads must be positive")
    hf = require_hf()
    root = Path(model_dir)
    manifest = verify_hf_assets(root)
    if manifest["kind"] not in {"native-export", "qwen-snapshot"}:
        raise ArtifactError("benchmark requires a verified standalone HF model")
    torch.set_num_threads(threads)
    tokenizer = hf.AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    model = hf.AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    fp32 = measure_model(
        model, tokenizer, max_new_tokens=max_new_tokens, repeats=repeats
    )
    modes: dict[str, Any] = {"fp32": fp32}
    quantized_engine = None
    if dynamic_int8:
        quantized_engine = select_quantized_engine()
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
        ).eval()
        int8 = measure_model(
            quantized, tokenizer, max_new_tokens=max_new_tokens, repeats=repeats
        )
        fp_outputs = [sample["output_token_ids"] for sample in fp32["samples"]]
        int8_outputs = [sample["output_token_ids"] for sample in int8["samples"]]
        int8["greedy_sequence_match_fraction"] = sum(
            left == right for left, right in zip(fp_outputs, int8_outputs, strict=True)
        ) / len(fp_outputs)
        int8["state_size_ratio_to_fp32"] = (
            int8["state_dict_bytes"] / fp32["state_dict_bytes"]
        )
        int8["throughput_ratio_to_fp32"] = (
            int8["median_tokens_per_second"] / fp32["median_tokens_per_second"]
        )
        modes["dynamic-int8"] = int8
    psutil = require_psutil()
    protocol = {
        "device": "cpu",
        "threads": threads,
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "prompts": list(PROMPTS),
        "do_sample": False,
        "warmup_runs": 1,
        "memory_metric": "process RSS sampled immediately before/after generate",
        "quantization": (
            "torch.ao.quantization dynamic qint8 Linear; benchmark only"
            if dynamic_int8
            else "none"
        ),
        "quantized_engine": quantized_engine,
        "runtime": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": hf.__version__,
            "psutil_version": psutil.__version__,
            "logical_cpu_count": psutil.cpu_count(logical=True),
        },
    }
    model_identity = {
        "kind": manifest["kind"],
        "model_route": manifest["model_route"],
        "source": manifest.get("source"),
        "repo_id": manifest.get("repo_id"),
        "revision": manifest.get("revision"),
    }
    report = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "model_manifest_sha256": sha256_file(root / "hf_manifest.json"),
        "model_identity": model_identity,
        "protocol": protocol,
        "modes": modes,
        "limitations": [
            "Wall-clock and process RSS are machine-local diagnostics, "
            "not portable scores.",
            "RSS includes Python/runtime allocations and is not accelerator memory.",
            "RSS is sampled before/after generation and can miss transient peaks.",
            "Dynamic INT8 is an in-memory PyTorch benchmark, "
            "not a portable HF/GGUF export.",
            "The pinned PyTorch release deprecates torch.ao dynamic quantization; "
            "a future protocol version must migrate explicitly.",
            "Greedy sequence equality measures output stability, not task quality.",
        ],
    }
    report["report_sha256"] = digest(
        {
            "model_manifest_sha256": report["model_manifest_sha256"],
            "model_identity": model_identity,
            "protocol": protocol,
            "modes": modes,
            "limitations": report["limitations"],
        }
    )
    return report
