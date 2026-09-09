"""Torch and accelerator checks for supported execution profiles."""

from __future__ import annotations

from typing import Any

from llm_lifecycle_lab.doctor.result import CheckResult, CheckStatus


def check_torch_and_device(
    *,
    require_torch: bool,
    require_cuda: bool,
    minimum_cuda_gib: float,
) -> tuple[CheckResult, CheckResult]:
    try:
        import torch
    except Exception as exc:
        torch_status = CheckStatus.FAIL if require_torch else CheckStatus.WARN
        torch_check = CheckResult(
            name="torch",
            status=torch_status,
            message=f"PyTorch cannot be imported: {type(exc).__name__}: {exc}",
            details={"required": require_torch},
        )
        device_check = CheckResult(
            name="device",
            status=CheckStatus.FAIL if require_cuda else CheckStatus.WARN,
            message="device detection skipped because PyTorch is unavailable",
            details={"requires_cuda": require_cuda},
        )
        return torch_check, device_check

    torch_check = CheckResult(
        name="torch",
        status=CheckStatus.PASS,
        message=f"PyTorch {torch.__version__} is importable",
        details={"version": torch.__version__},
    )

    if torch.cuda.is_available():
        return torch_check, _cuda_check(
            torch,
            require_cuda=require_cuda,
            minimum_cuda_gib=minimum_cuda_gib,
        )

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if require_cuda:
        return torch_check, CheckResult(
            name="device",
            status=CheckStatus.FAIL,
            message="this profile requires CUDA, but no CUDA device is available",
            details={"mps_available": mps_available},
        )

    device = "mps" if mps_available else "cpu"
    return torch_check, CheckResult(
        name="device",
        status=CheckStatus.PASS,
        message=f"{device} is available for the selected profile",
        details={"device": device},
    )


def _cuda_check(
    torch: Any,
    *,
    require_cuda: bool,
    minimum_cuda_gib: float,
) -> CheckResult:
    devices = []
    largest_bytes = 0
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        total_memory = int(properties.total_memory)
        largest_bytes = max(largest_bytes, total_memory)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": total_memory,
            }
        )

    largest_gib = largest_bytes / 1024**3
    enough_memory = largest_gib >= minimum_cuda_gib
    status = CheckStatus.PASS if enough_memory or not require_cuda else CheckStatus.FAIL
    return CheckResult(
        name="device",
        status=status,
        message=(
            f"CUDA is available; largest device has {largest_gib:.1f} GiB "
            f"(profile minimum {minimum_cuda_gib:.1f} GiB)"
        ),
        details={
            "device": "cuda",
            "devices": devices,
            "required_gib": minimum_cuda_gib,
        },
    )
