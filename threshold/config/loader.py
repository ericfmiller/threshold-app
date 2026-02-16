"""Configuration loading with YAML parsing and environment variable expansion."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from threshold.config.schema import ThresholdConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path("config.yaml"),
    Path("~/.threshold/config.yaml").expanduser(),
]


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} references in config values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        match = pattern.search(value)
        while match:
            env_var = match.group(1)
            env_value = os.environ.get(env_var, "")
            value = value[: match.start()] + env_value + value[match.end() :]
            match = pattern.search(value)
        return value
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _find_config_file(explicit_path: str | Path | None = None) -> Path | None:
    """Find the config file to load."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path
        logger.warning("Config file not found: %s", path)
        return None

    for candidate in DEFAULT_CONFIG_PATHS:
        resolved = candidate.expanduser()
        if resolved.exists():
            logger.info("Using config: %s", resolved)
            return resolved

    return None


def load_config(path: str | Path | None = None) -> ThresholdConfig:
    """Load and validate configuration.

    Resolution order:
    1. Explicit path argument
    2. config.yaml in current directory
    3. ~/.threshold/config.yaml
    4. All defaults (no file needed)

    Environment variables are expanded in string values: ${VAR_NAME}
    """
    config_path = _find_config_file(path)

    if config_path is not None:
        logger.info("Loading config from %s", config_path)
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _expand_env_vars(raw)
    else:
        logger.info("No config file found, using defaults")
        raw = {}

    config = ThresholdConfig.model_validate(raw)
    logger.debug("Config loaded: version=%d", config.version)
    return config


def resolve_path(path_str: str) -> Path:
    """Resolve a path from config, expanding ~ and making absolute."""
    return Path(path_str).expanduser().resolve()
