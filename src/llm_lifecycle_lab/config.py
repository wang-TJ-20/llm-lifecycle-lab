"""Configuration loading and deterministic resolution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from llm_lifecycle_lab.contracts import RunConfig
from llm_lifecycle_lab.exceptions import ConfigError, ContractError


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON mapping without performing implicit interpolation."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            if config_path.suffix.lower() == ".json":
                value = json.load(handle)
            else:
                value = yaml.safe_load(handle)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot parse config {config_path}: {exc}") from exc

    if not isinstance(value, Mapping):
        raise ConfigError(f"config root must be a mapping: {config_path}")
    return dict(value)


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate an experiment configuration."""

    try:
        return RunConfig.from_dict(load_mapping(path))
    except ContractError as exc:
        raise ConfigError(f"invalid config {Path(path)}: {exc}") from exc


def canonical_json(data: Mapping[str, Any]) -> str:
    """Serialize a mapping deterministically for hashing and persistence."""

    try:
        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"configuration is not JSON serializable: {exc}") from exc


def config_sha256(config: RunConfig) -> str:
    """Hash the fully resolved configuration."""

    return hashlib.sha256(canonical_json(config.to_dict()).encode("utf-8")).hexdigest()


def dump_yaml(data: Mapping[str, Any]) -> str:
    """Render a stable, human-readable YAML representation."""

    try:
        return yaml.safe_dump(
            dict(data),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot serialize resolved config: {exc}") from exc
