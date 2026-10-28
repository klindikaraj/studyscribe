"""Paths and settings shared by all modules."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

APP_DIR_NAME = "StudyScribe"
DATA_DIR_ENV = "STUDYSCRIBE_DATA_DIR"

# Internal audio format: PCM int16, mono, 16 kHz (the native input format of Whisper).
SAMPLE_RATE = 16_000


def default_data_dir(environ: Mapping[str, str] | None = None) -> Path:
    """%LOCALAPPDATA%\\StudyScribe, independent of the working directory."""
    env = os.environ if environ is None else environ
    base = env.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_DIR_NAME
    # LOCALAPPDATA is always set on Windows; this only matters on other systems.
    return Path.home() / "AppData" / "Local" / APP_DIR_NAME


def resolve_data_dir(
    cli_value: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Pick the data directory: --data-dir, then $STUDYSCRIBE_DATA_DIR, then the default."""
    env = os.environ if environ is None else environ
    if cli_value:
        path = Path(cli_value)
    elif env.get(DATA_DIR_ENV):
        path = Path(env[DATA_DIR_ENV])
    else:
        path = default_data_dir(env)
    return path.expanduser().resolve()
