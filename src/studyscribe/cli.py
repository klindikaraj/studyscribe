"""Command-line entry point. Subcommands (record, sessions, recover) arrive with step 1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from studyscribe import __version__
from studyscribe.config import DATA_DIR_ENV, resolve_data_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="studyscribe",
        description="Record online lectures, transcribe them locally and write study notes.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        help=f"data directory (default: ${DATA_DIR_ENV}, else %%LOCALAPPDATA%%\\StudyScribe)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    parser.print_help()
    return 0
