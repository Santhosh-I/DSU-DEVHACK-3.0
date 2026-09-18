"""
Plastic-Ledger — Cache Utilities
==================================
Stage output caching helpers and YAML config loading with env-var interpolation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

from pipeline.utils.logging_utils import get_logger

logger = get_logger(__name__)


def stage_output_exists(
    output_dir: Union[str, Path],
    expected_files: List[str],
) -> bool:
    """Check whether a stage's expected output files already exist.

    Args:
        output_dir: Directory that should contain the outputs.
        expected_files: List of filenames (relative to *output_dir*)
            that must all be present for the stage to be considered complete.

    Returns:
        ``True`` if **all** expected files exist; ``False`` otherwise.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return False

    for fname in expected_files:
        if not (output_dir / fname).exists():
            return False

    logger.info(
        "[bold green]Cache hit[/] — all %d outputs exist in %s",
        len(expected_files),
        output_dir,
    )
    return True


def _interpolate_env_vars(value: str) -> str:
    """Replace ``${VAR}`` placeholders with environment variable values."""
    pattern = re.compile(r"\$\{(\w+)\}")

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name, "")
        if not env_val:
            logger.warning("Environment variable %s is not set", var_name)
        return env_val

    return pattern.sub(_replace, value)


def _walk_and_interpolate(obj: Any) -> Any:
    """Recursively interpolate env vars in all string values of a config tree."""
    if isinstance(obj, str):
        return _interpolate_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_interpolate(item) for item in obj]
    return obj


def load_config(
    config_path: Union[str, Path] = "config/config.yaml",
    env_path: Union[str, Path, None] = ".env",
) -> Dict[str, Any]:
    """Load YAML config and interpolate ``${ENV_VAR}`` placeholders.

    Loads ``.env`` first (if it exists) so that environment variables are
    available for interpolation.

    Args:
        config_path: Path to the YAML configuration file.
        env_path: Path to a ``.env`` file.  Pass ``None`` to skip.

    Returns:
        Fully-resolved configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    config_path = Path(config_path)

    # Load .env if available
    if env_path is not None:
        env_file = Path(env_path)
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Loaded environment from %s", env_file)
    else:
        for candidate in [Path(".env"), Path(__file__).resolve().parents[3] / ".env", Path(__file__).resolve().parents[2] / ".env"]:
            if candidate.exists():
                load_dotenv(candidate)
                logger.info("Loaded environment from %s", candidate)
                break

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        raw = yaml.safe_load(fh)

    config = _walk_and_interpolate(raw)
    return config
