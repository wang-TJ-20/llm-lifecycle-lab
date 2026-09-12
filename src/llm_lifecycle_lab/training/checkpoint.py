"""Atomic checkpoints for model, optimizer, scheduler, RNG, and data position."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from llm_lifecycle_lab.artifacts import RunArtifacts
from llm_lifecycle_lab.contracts import (
    CheckpointMetadata,
    ModelRoute,
    Stage,
)
from llm_lifecycle_lab.exceptions import ArtifactError
from llm_lifecycle_lab.model.protocol import ModelProtocol
from llm_lifecycle_lab.training.batching import DeterministicBatchStream


@dataclass(frozen=True, slots=True)
class TrainerState:
    global_step: int = 0
    tokens_seen: int = 0
    best_eval_loss: float | None = None


class CheckpointManager:
    def __init__(
        self,
        artifacts: RunArtifacts,
        *,
        model_route: ModelRoute,
        stage: Stage,
        tokenizer_sha256: str,
        config_sha256: str,
    ) -> None:
        self.artifacts = artifacts
        self.model_route = model_route
        self.stage = stage
        self.tokenizer_sha256 = tokenizer_sha256
        self.config_sha256 = config_sha256

    def save(
        self,
        *,
        model: ModelProtocol,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        stream: DeterministicBatchStream,
        state: TrainerState,
        scaler: torch.amp.GradScaler | None = None,
    ) -> Path:
        checkpoint_id = f"step-{state.global_step:08d}"
        target = self.artifacts.artifact_path(f"checkpoints/{checkpoint_id}")
        if target.exists():
            raise ArtifactError(
                f"checkpoint already exists; refusing to overwrite: {target}"
            )

        temporary = Path(
            tempfile.mkdtemp(
                dir=target.parent,
                prefix=f".{checkpoint_id}.",
                suffix=".tmp",
            )
        )
        try:
            model.save(temporary / "model")
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                run_id=self.artifacts.run_id,
                model_route=self.model_route,
                stage=self.stage,
                step=state.global_step,
                tokenizer_sha256=self.tokenizer_sha256,
                config_sha256=self.config_sha256,
            )
            _write_json(temporary / "checkpoint_metadata.json", metadata.to_dict())
            _write_json(
                temporary / "trainer_state.json",
                {
                    **asdict(state),
                    "batch_stream": stream.state_dict(),
                },
            )
            _save_torch_state(
                temporary / "optimizer_state.pt",
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "mps_rng_state": _mps_rng_state(),
                },
            )
            os.replace(temporary, target)
            self.artifacts.write_json(
                "latest_checkpoint.json",
                {
                    "checkpoint_id": checkpoint_id,
                    "path": f"checkpoints/{checkpoint_id}",
                    "step": state.global_step,
                },
            )
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return target

    def load(
        self,
        checkpoint: str | Path,
        *,
        model: ModelProtocol,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        stream: DeterministicBatchStream,
        scaler: torch.amp.GradScaler | None = None,
    ) -> TrainerState:
        source = self._resolve_checkpoint(checkpoint)
        metadata = _load_checkpoint_metadata(source)
        self._validate_metadata(metadata)

        model.load(source / "model")
        try:
            trainer_value = json.loads(
                (source / "trainer_state.json").read_text(encoding="utf-8")
            )
            state = TrainerState(
                global_step=int(trainer_value["global_step"]),
                tokens_seen=int(trainer_value["tokens_seen"]),
                best_eval_loss=(
                    None
                    if trainer_value.get("best_eval_loss") is None
                    else float(trainer_value["best_eval_loss"])
                ),
            )
            stream.load_state_dict(trainer_value["batch_stream"])
            torch_state = torch.load(
                source / "optimizer_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            optimizer.load_state_dict(torch_state["optimizer"])
            scheduler.load_state_dict(torch_state["scheduler"])
            if scaler is not None and torch_state.get("scaler") is not None:
                scaler.load_state_dict(torch_state["scaler"])
            torch.set_rng_state(torch_state["torch_rng_state"])
            if (
                torch.cuda.is_available()
                and torch_state.get("cuda_rng_state") is not None
            ):
                torch.cuda.set_rng_state_all(torch_state["cuda_rng_state"])
            if torch_state.get("mps_rng_state") is not None:
                _set_mps_rng_state(torch_state["mps_rng_state"])
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ArtifactError(f"cannot restore checkpoint {source}: {exc}") from exc
        if state.global_step != metadata.step:
            raise ArtifactError("trainer state step does not match checkpoint metadata")
        return state

    def _resolve_checkpoint(self, checkpoint: str | Path) -> Path:
        value = Path(checkpoint)
        if value.is_absolute():
            try:
                value.resolve().relative_to(self.artifacts.path.resolve())
            except ValueError as exc:
                raise ArtifactError(
                    "resume checkpoint must belong to the current run"
                ) from exc
            return value
        if len(value.parts) == 1:
            return self.artifacts.artifact_path(f"checkpoints/{value}")
        return self.artifacts.artifact_path(value)

    def _validate_metadata(self, metadata: CheckpointMetadata) -> None:
        expected = {
            "run_id": (metadata.run_id, self.artifacts.run_id),
            "model_route": (metadata.model_route, self.model_route),
            "stage": (metadata.stage, self.stage),
            "tokenizer_sha256": (
                metadata.tokenizer_sha256,
                self.tokenizer_sha256,
            ),
            "config_sha256": (
                metadata.config_sha256,
                self.config_sha256,
            ),
        }
        mismatches = [
            field for field, (actual, wanted) in expected.items() if actual != wanted
        ]
        if mismatches:
            raise ArtifactError("checkpoint is incompatible: " + ", ".join(mismatches))


def _load_checkpoint_metadata(path: Path) -> CheckpointMetadata:
    try:
        value = json.loads(
            (path / "checkpoint_metadata.json").read_text(encoding="utf-8")
        )
        return CheckpointMetadata.from_dict(value)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid checkpoint metadata under {path}: {exc}") from exc


def _write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot write checkpoint metadata {path}: {exc}") from exc


def _save_torch_state(path: Path, value: dict[str, Any]) -> None:
    try:
        torch.save(value, path)
    except (OSError, RuntimeError) as exc:
        raise ArtifactError(f"cannot write optimizer state {path}: {exc}") from exc


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _mps_rng_state() -> torch.Tensor | None:
    if _mps_available():
        return torch.mps.get_rng_state()
    return None


def _set_mps_rng_state(state: torch.Tensor) -> None:
    if _mps_available():
        torch.mps.set_rng_state(state)
