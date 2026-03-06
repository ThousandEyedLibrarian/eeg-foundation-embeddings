"""YAML-based configuration with CLI override support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from shared.preprocessing import PreprocessingConfig


@dataclass
class ModelConfig:
    """Model-specific configuration."""

    model_name: str = ""            # "labram" or "reve"
    backend: str = "braindecode"    # "braindecode" or "original" (LaBraM only)
    size: str = "base"              # "base" or "large" (REVE only)
    checkpoint: str = ""            # path or URL, empty = use default
    device: str = "cuda"
    batch_size: int = 4
    embedding_dim: int = 0          # 0 = auto from model
    hf_token: str = ""              # HuggingFace token (REVE)


@dataclass
class OutputConfig:
    """Output configuration for Zarr files."""

    output_dir: str = "./output"
    zarr_compression: str = "zstd"
    zarr_compression_level: int = 5
    store_per_window: bool = True
    store_mean_pooled: bool = True
    zarr_format: int = 2            # v2 for cross-language compat


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    input_path: str = ""
    verbose: bool = False


def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_dataclass(cls, data: dict) -> Any:
    """Convert a dict to a dataclass, ignoring unknown keys."""
    known_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return cls(**filtered)


def _load_yaml(yaml_path: str | Path) -> dict:
    """Load a YAML config file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_hf_token(config: FullConfig) -> None:
    """Resolve HuggingFace token from config or environment."""
    if config.model.hf_token:
        return
    env_token = os.environ.get("HF_TOKEN", "")
    if env_token:
        config.model.hf_token = env_token


def load_config(
    yaml_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> FullConfig:
    """Load configuration from YAML file with CLI overrides.

    Resolution order (later wins):
    1. Dataclass defaults
    2. YAML file values
    3. CLI argument overrides

    Args:
        yaml_path: Path to YAML config file. None to skip.
        cli_overrides: Dict of CLI overrides. Keys use dot notation
            for nested values (e.g., 'model.batch_size').

    Returns:
        Fully resolved FullConfig instance.
    """
    config = FullConfig()

    # Load YAML if provided
    if yaml_path:
        raw = _load_yaml(yaml_path)

        if "model" in raw and isinstance(raw["model"], dict):
            model_dict = raw["model"]
            # Map 'name' to 'model_name' for convenience
            if "name" in model_dict:
                model_dict["model_name"] = model_dict.pop("name")
            config.model = _dict_to_dataclass(ModelConfig, model_dict)

        if "preprocessing" in raw and isinstance(raw["preprocessing"], dict):
            preproc_dict = dict(raw["preprocessing"])
            # Handle bandpass shorthand
            bandpass = preproc_dict.pop("bandpass", None)
            config.preprocessing = _dict_to_dataclass(
                PreprocessingConfig, preproc_dict,
            )
            if bandpass and isinstance(bandpass, (list, tuple)) and len(bandpass) == 2:
                config.preprocessing.bandpass_low = float(bandpass[0])
                config.preprocessing.bandpass_high = float(bandpass[1])

        if "output" in raw and isinstance(raw["output"], dict):
            config.output = _dict_to_dataclass(OutputConfig, raw["output"])

    # Apply CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            _set_nested_attr(config, key, value)

    # Resolve HF token from env if not set
    _resolve_hf_token(config)

    return config


def _set_nested_attr(obj: Any, dotted_key: str, value: Any) -> None:
    """Set a nested attribute using dot notation (e.g., 'model.batch_size')."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)

    attr_name = parts[-1]
    # Convert value to the correct type based on the field type
    current = getattr(obj, attr_name, None)
    if current is not None:
        target_type = type(current)
        if target_type is bool:
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
        elif target_type in (int, float, str):
            value = target_type(value)

    setattr(obj, attr_name, value)
