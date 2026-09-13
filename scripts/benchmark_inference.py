"""生成 HF 模型的 CPU FP32/动态 INT8 吞吐与内存报告。

Benchmark verified HF models on CPU without exporting nonportable quantized weights.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _project_path import add_project_src_to_path

add_project_src_to_path()

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.exceptions import LLMLabError
from llm_lifecycle_lab.interop.benchmark import benchmark_hf_cpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/benchmark_inference.py", description=__doc__
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-int8", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output already exists; refusing to overwrite")
        report = benchmark_hf_cpu(
            args.model,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
            threads=args.threads,
            dynamic_int8=not args.no_int8,
        )
        args.output.mkdir(parents=True)
        artifacts = RunArtifacts(args.output.name, args.output)
        artifacts.write_json("benchmark.json", report)
        lines = [
            "# CPU 推理基准",
            "",
            f"- 模型清单：`{report['model_manifest_sha256']}`",
            f"- 报告 digest：`{report['report_sha256']}`",
            f"- 线程：{args.threads}",
            f"- 每次最大新 token：{args.max_new_tokens}",
            f"- 机器：`{report['protocol']['runtime']['platform']}`",
            f"- PyTorch：`{report['protocol']['runtime']['torch_version']}`",
            f"- 量化引擎：`{report['protocol']['quantized_engine'] or 'none'}`",
            "",
            "| 模式 | 状态字节 | FP32 体积比 | 中位 tokens/s | "
            "FP32 吞吐比 | 观测峰值 RSS | greedy 匹配 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, value in report["modes"].items():
            match = value.get("greedy_sequence_match_fraction")
            lines.append(
                f"| {name} | {value['state_dict_bytes']} | "
                f"{value.get('state_size_ratio_to_fp32', 1.0):.3f} | "
                f"{value['median_tokens_per_second']:.3f} | "
                f"{value.get('throughput_ratio_to_fp32', 1.0):.3f} | "
                f"{value['peak_observed_rss_bytes']} | "
                f"{'-' if match is None else f'{match:.3f}'} |"
            )
        lines.extend(["", *(f"- {item}" for item in report["limitations"]), ""])
        artifacts.write_text("benchmark.md", "\n".join(lines))
        print(
            json.dumps(
                {"output": str(args.output), "modes": list(report["modes"])},
                ensure_ascii=False,
                indent=2,
            )
        )
    except (LLMLabError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
