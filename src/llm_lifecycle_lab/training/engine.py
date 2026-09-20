"""Explicit PyTorch training loop shared by lifecycle stages."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.exceptions import ConfigError, ContractError
from llm_lifecycle_lab.model.bundle import ModelBundle
from llm_lifecycle_lab.model.protocol import ModelProtocol
from llm_lifecycle_lab.training.batching import DeterministicBatchStream
from llm_lifecycle_lab.training.checkpoint import (
    CheckpointManager,
    TrainerState,
)
from llm_lifecycle_lab.training.objective import TrainingObjective

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class ResolvedTrainingBudget:
    mode: str
    max_steps: int
    target_train_tokens: int
    estimated_train_tokens: int
    supervised_tokens_per_epoch: int
    examples_per_epoch: int
    micro_batches_per_epoch: int
    estimated_tokens_per_step: float
    estimated_epochs: float
    requested_max_steps: int | None = None
    requested_max_train_tokens: int | None = None
    requested_num_epochs: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    sequence_length: int
    max_steps: int | None = None
    max_train_tokens: int | None = None
    num_epochs: float | None = None
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    gradient_clipping: float = 1.0
    dtype: str = "float32"
    device: str = "auto"
    checkpoint_interval: int = 100
    eval_interval: int = 100
    eval_batches: int = 8
    log_interval: int = 1

    def __post_init__(self) -> None:
        positive_integers = {
            "sequence_length": self.sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "checkpoint_interval": self.checkpoint_interval,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "log_interval": self.log_interval,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ConfigError(f"training.{name} must be positive")
        budget_count = sum(
            value is not None
            for value in (
                self.max_steps,
                self.max_train_tokens,
                self.num_epochs,
            )
        )
        if budget_count != 1:
            raise ConfigError(
                "training must declare exactly one of max_steps, "
                "max_train_tokens, or num_epochs"
            )
        if self.max_steps is not None and self.max_steps <= 0:
            raise ConfigError("training.max_steps must be positive")
        if self.max_train_tokens is not None and self.max_train_tokens <= 0:
            raise ConfigError("training.max_train_tokens must be positive")
        if self.num_epochs is not None and (
            not math.isfinite(self.num_epochs) or self.num_epochs <= 0
        ):
            raise ConfigError("training.num_epochs must be finite and positive")
        if self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ConfigError("training.weight_decay must be non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ConfigError("training betas must be in [0, 1)")
        if self.warmup_steps < 0:
            raise ConfigError("training.warmup_steps must be non-negative")
        if self.max_steps is not None and self.warmup_steps >= self.max_steps:
            raise ConfigError("training.warmup_steps must be smaller than max_steps")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ConfigError("training.min_lr_ratio must be in [0, 1]")
        if self.gradient_clipping <= 0:
            raise ConfigError("training.gradient_clipping must be positive")
        if self.dtype not in _DTYPES:
            raise ConfigError(f"training.dtype must be one of {sorted(_DTYPES)}")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("training.device must be auto, cpu, cuda, or mps")

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineConfig:
        allowed = {
            "max_steps",
            "max_train_tokens",
            "num_epochs",
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "warmup_steps",
            "min_lr_ratio",
            "gradient_clipping",
            "dtype",
            "device",
            "checkpoint_interval",
            "eval_interval",
            "eval_batches",
            "log_interval",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ConfigError(f"unknown training config fields: {', '.join(unknown)}")
        try:
            return cls(
                sequence_length=int(data["sequence_length"]),
                max_steps=(int(data["max_steps"]) if "max_steps" in data else None),
                max_train_tokens=(
                    int(data["max_train_tokens"])
                    if "max_train_tokens" in data
                    else None
                ),
                num_epochs=(
                    float(data["num_epochs"]) if "num_epochs" in data else None
                ),
                micro_batch_size=int(data.get("micro_batch_size", 1)),
                gradient_accumulation_steps=int(
                    data.get("gradient_accumulation_steps", 1)
                ),
                learning_rate=float(data.get("learning_rate", 3e-4)),
                weight_decay=float(data.get("weight_decay", 0.1)),
                beta1=float(data.get("beta1", 0.9)),
                beta2=float(data.get("beta2", 0.95)),
                warmup_steps=int(data.get("warmup_steps", 0)),
                min_lr_ratio=float(data.get("min_lr_ratio", 0.1)),
                gradient_clipping=float(data.get("gradient_clipping", 1.0)),
                dtype=str(data.get("dtype", "float32")),
                device=str(data.get("device", "auto")),
                checkpoint_interval=int(data.get("checkpoint_interval", 100)),
                eval_interval=int(data.get("eval_interval", 100)),
                eval_batches=int(data.get("eval_batches", 8)),
                log_interval=int(data.get("log_interval", 1)),
            )
        except KeyError as exc:
            raise ConfigError(f"missing training config field: {exc.args[0]}") from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid training config value: {exc}") from exc

    def resolve_budget(
        self,
        *,
        examples_per_epoch: int,
        supervised_tokens_per_epoch: int,
    ) -> tuple[EngineConfig, ResolvedTrainingBudget]:
        if examples_per_epoch <= 0 or supervised_tokens_per_epoch <= 0:
            raise ConfigError("training budget requires a non-empty packed dataset")
        micro_batches_per_epoch = math.ceil(examples_per_epoch / self.micro_batch_size)
        estimated_tokens_per_step = (
            supervised_tokens_per_epoch
            / micro_batches_per_epoch
            * self.gradient_accumulation_steps
        )

        if self.max_steps is not None:
            mode = "max_steps"
            max_steps = self.max_steps
            target_train_tokens = math.ceil(max_steps * estimated_tokens_per_step)
        elif self.max_train_tokens is not None:
            mode = "max_train_tokens"
            target_train_tokens = self.max_train_tokens
            max_steps = math.ceil(target_train_tokens / estimated_tokens_per_step)
        else:
            if self.num_epochs is None:
                raise ConfigError("training budget is unresolved")
            mode = "num_epochs"
            target_train_tokens = math.ceil(
                self.num_epochs * supervised_tokens_per_epoch
            )
            max_steps = math.ceil(target_train_tokens / estimated_tokens_per_step)

        resolved = replace(
            self,
            max_steps=max_steps,
            max_train_tokens=None,
            num_epochs=None,
        )
        estimated_train_tokens = math.ceil(max_steps * estimated_tokens_per_step)
        return resolved, ResolvedTrainingBudget(
            mode=mode,
            max_steps=max_steps,
            target_train_tokens=target_train_tokens,
            estimated_train_tokens=estimated_train_tokens,
            supervised_tokens_per_epoch=supervised_tokens_per_epoch,
            examples_per_epoch=examples_per_epoch,
            micro_batches_per_epoch=micro_batches_per_epoch,
            estimated_tokens_per_step=estimated_tokens_per_step,
            estimated_epochs=(estimated_train_tokens / supervised_tokens_per_epoch),
            requested_max_steps=self.max_steps,
            requested_max_train_tokens=self.max_train_tokens,
            requested_num_epochs=self.num_epochs,
        )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    global_step: int
    tokens_seen: int
    target_train_tokens: int
    epochs_seen: float
    target_token_coverage: float
    final_loss: float
    best_eval_loss: float | None
    final_checkpoint: str
    elapsed_seconds: float


class TrainingEngine:
    def __init__(
        self,
        *,
        config: EngineConfig,
        budget: ResolvedTrainingBudget,
        artifacts: RunArtifacts,
        checkpoint_manager: CheckpointManager,
        seed: int,
    ) -> None:
        self.config = config
        self.budget = budget
        self.artifacts = artifacts
        self.checkpoint_manager = checkpoint_manager
        self.seed = seed
        self.device = resolve_device(config.device)
        self.compute_dtype = config.torch_dtype
        if config.max_steps != budget.max_steps:
            raise ConfigError("resolved training config and budget disagree")
        if self.compute_dtype is torch.float16 and self.device.type != "cuda":
            raise ConfigError("float16 training is supported only on CUDA")

    def train(
        self,
        *,
        bundle: ModelBundle,
        objective: TrainingObjective,
        train_stream: DeterministicBatchStream,
        evaluation_batches: Sequence[dict[str, Any]] = (),
        resume_from: str | Path | None = None,
        metric_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> TrainingResult:
        _seed_torch(self.seed)
        model = bundle.model
        model.to_device(self.device)
        model.set_training(True)
        named_parameters = tuple(model.trainable_parameters())
        if not named_parameters:
            raise ContractError("model has no trainable parameters")
        optimizer = _build_optimizer(named_parameters, self.config)
        scheduler = _build_scheduler(optimizer, self.config)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.compute_dtype is torch.float16,
        )
        state = TrainerState()
        if resume_from is not None:
            state = self.checkpoint_manager.load(
                resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stream=train_stream,
                scaler=scaler,
            )
        if state.global_step >= self.budget.max_steps:
            raise ConfigError("resume checkpoint already reached or exceeded max_steps")

        optimizer.zero_grad(set_to_none=True)
        self.artifacts.update_status("running")
        started_at = time.monotonic()
        final_loss = math.nan
        final_checkpoint: Path | None = None

        try:
            if evaluation_batches:
                baseline = evaluate_objective(
                    model=bundle.model,
                    objective=objective,
                    batches=evaluation_batches[: self.config.eval_batches],
                    device=self.device,
                    dtype=self.compute_dtype,
                )
                previous_best = state.best_eval_loss
                baseline_loss = baseline["eval_loss"]
                state = TrainerState(
                    global_step=state.global_step,
                    tokens_seen=state.tokens_seen,
                    best_eval_loss=(
                        baseline_loss
                        if previous_best is None
                        else min(previous_best, baseline_loss)
                    ),
                )
                baseline_metrics = {
                    "stage": objective.name,
                    "event": (
                        "baseline" if state.global_step == 0 else "resume-baseline"
                    ),
                    "step": state.global_step,
                    **baseline,
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                self.artifacts.append_metric(baseline_metrics)
                if metric_callback is not None:
                    metric_callback(baseline_metrics)

            while state.global_step < self.budget.max_steps:
                step_started_at = time.monotonic()
                loss_sum = 0.0
                tokens_this_step = 0
                bytes_this_step = 0.0
                for _ in range(self.config.gradient_accumulation_steps):
                    batch = _batch_to_device(
                        train_stream.next_batch(),
                        self.device,
                    )
                    with _autocast_context(self.device, self.compute_dtype):
                        output = objective(model, batch)
                    supervised_tokens = int(output.metrics.get("supervised_tokens", 0))
                    if supervised_tokens <= 0:
                        raise ContractError("training batch has zero supervised tokens")
                    token_loss = output.loss * supervised_tokens
                    if not bool(torch.isfinite(token_loss.detach())):
                        raise ContractError("training loss became non-finite")
                    scaler.scale(token_loss).backward()
                    loss_sum += float(output.loss.detach()) * supervised_tokens
                    tokens_this_step += supervised_tokens
                    bytes_this_step += float(output.metrics.get("source_bytes", 0))

                scaler.unscale_(optimizer)
                normalization = 1.0 / tokens_this_step
                for _, parameter in named_parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(normalization)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for _, parameter in named_parameters],
                    self.config.gradient_clipping,
                    error_if_nonfinite=True,
                )
                learning_rate = float(optimizer.param_groups[0]["lr"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step = state.global_step + 1
                tokens_seen = state.tokens_seen + tokens_this_step
                final_loss = loss_sum / max(tokens_this_step, 1)
                best_eval_loss = state.best_eval_loss
                step_seconds = time.monotonic() - step_started_at
                metrics: dict[str, Any] = {
                    "stage": objective.name,
                    "step": global_step,
                    "train_loss": final_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "tokens": tokens_this_step,
                    "tokens_seen": tokens_seen,
                    "epochs_seen": (
                        tokens_seen / self.budget.supervised_tokens_per_epoch
                    ),
                    "target_token_coverage": (
                        tokens_seen / self.budget.target_train_tokens
                    ),
                    "step_seconds": step_seconds,
                    "tokens_per_second": tokens_this_step / max(step_seconds, 1e-9),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                if bytes_this_step > 0:
                    metrics["train_bits_per_byte"] = (
                        loss_sum / math.log(2) / bytes_this_step
                    )
                if self.device.type == "cuda":
                    metrics["cuda_memory_allocated_bytes"] = (
                        torch.cuda.memory_allocated(self.device)
                    )
                    metrics["cuda_max_memory_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated(self.device)
                    )

                if evaluation_batches and (
                    global_step % self.config.eval_interval == 0
                    or global_step == self.budget.max_steps
                ):
                    evaluation = evaluate_objective(
                        model=bundle.model,
                        objective=objective,
                        batches=evaluation_batches[: self.config.eval_batches],
                        device=self.device,
                        dtype=self.compute_dtype,
                    )
                    metrics.update(evaluation)
                    eval_loss = evaluation["eval_loss"]
                    if best_eval_loss is None or eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss

                state = TrainerState(
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    best_eval_loss=best_eval_loss,
                )
                if (
                    global_step % self.config.log_interval == 0
                    or global_step == self.budget.max_steps
                ):
                    self.artifacts.append_metric(metrics)
                    if metric_callback is not None:
                        metric_callback(metrics)

                if (
                    global_step % self.config.checkpoint_interval == 0
                    or global_step == self.budget.max_steps
                ):
                    final_checkpoint = self.checkpoint_manager.save(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        stream=train_stream,
                        state=state,
                        scaler=scaler,
                    )
        except Exception as exc:
            self.artifacts.write_json(
                "failure.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            self.artifacts.update_status("failed")
            raise

        if final_checkpoint is None:
            raise ContractError("training completed without a checkpoint")
        result = TrainingResult(
            global_step=state.global_step,
            tokens_seen=state.tokens_seen,
            target_train_tokens=self.budget.target_train_tokens,
            epochs_seen=(state.tokens_seen / self.budget.supervised_tokens_per_epoch),
            target_token_coverage=(state.tokens_seen / self.budget.target_train_tokens),
            final_loss=final_loss,
            best_eval_loss=state.best_eval_loss,
            final_checkpoint=str(final_checkpoint),
            elapsed_seconds=time.monotonic() - started_at,
        )
        self.artifacts.write_json(
            "training_result.json",
            {
                "global_step": result.global_step,
                "tokens_seen": result.tokens_seen,
                "target_train_tokens": result.target_train_tokens,
                "epochs_seen": result.epochs_seen,
                "target_token_coverage": result.target_token_coverage,
                "final_loss": result.final_loss,
                "best_eval_loss": result.best_eval_loss,
                "final_checkpoint": result.final_checkpoint,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )
        self.artifacts.update_status("completed")
        return result


def evaluate_objective(
    *,
    model: ModelProtocol,
    objective: TrainingObjective,
    batches: Sequence[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    if not batches:
        raise ContractError("evaluation requires at least one batch")
    was_training = model.is_training()
    model.set_training(False)
    weighted_loss = 0.0
    supervised_tokens = 0
    source_bytes = 0.0
    language_totals = {
        language: {"nll": 0.0, "tokens": 0, "source_bytes": 0.0}
        for language in ("en", "zh")
    }
    try:
        with torch.inference_mode():
            for raw_batch in batches:
                batch = _batch_to_device(raw_batch, device)
                with _autocast_context(device, dtype):
                    output = objective(model, batch)
                tokens = int(output.metrics.get("supervised_tokens", 0))
                weighted_loss += float(output.loss) * tokens
                supervised_tokens += tokens
                source_bytes += float(output.metrics.get("source_bytes", 0))
                for language, totals in language_totals.items():
                    totals["nll"] += float(
                        output.metrics.get(
                            f"language_{language}_nll_sum",
                            0,
                        )
                    )
                    totals["tokens"] += int(
                        output.metrics.get(
                            f"language_{language}_tokens",
                            0,
                        )
                    )
                    totals["source_bytes"] += float(
                        output.metrics.get(
                            f"language_{language}_source_bytes",
                            0,
                        )
                    )
    finally:
        model.set_training(was_training)
    if supervised_tokens == 0:
        raise ContractError("evaluation has zero supervised tokens")
    loss = weighted_loss / supervised_tokens
    metrics = {
        "eval_loss": loss,
        "eval_perplexity": math.exp(min(loss, 20.0)),
        "eval_tokens": float(supervised_tokens),
    }
    if source_bytes > 0:
        metrics["eval_bits_per_byte"] = weighted_loss / math.log(2) / source_bytes
    for language, totals in language_totals.items():
        language_tokens = int(totals["tokens"])
        if language_tokens == 0:
            continue
        language_loss = totals["nll"] / language_tokens
        metrics[f"eval_{language}_loss"] = language_loss
        metrics[f"eval_{language}_perplexity"] = math.exp(min(language_loss, 20.0))
        metrics[f"eval_{language}_tokens"] = float(language_tokens)
        language_bytes = totals["source_bytes"]
        if language_bytes > 0:
            metrics[f"eval_{language}_bits_per_byte"] = (
                totals["nll"] / math.log(2) / language_bytes
            )
    return metrics


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError("training.device=cuda but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ConfigError("training.device=mps but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_optimizer(
    named_parameters: tuple[tuple[str, Any], ...],
    config: EngineConfig,
) -> torch.optim.AdamW:
    decay = [parameter for _, parameter in named_parameters if parameter.ndim >= 2]
    no_decay = [parameter for _, parameter in named_parameters if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: EngineConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.max_steps is None:
        raise ConfigError("training budget must be resolved before scheduler setup")
    max_steps = config.max_steps

    def multiplier(step: int) -> float:
        if config.warmup_steps and step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        decay_steps = max(max_steps - config.warmup_steps, 1)
        progress = min(
            max((step - config.warmup_steps) / decay_steps, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    if dtype is torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
