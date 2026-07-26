"""crewui.cli - the command-line surface.

``crewui`` is a library first: the point of the package is ``from crewui import
CrewAIPipelineTUI`` inside a host app. The CLI carries just enough to make an
install self-evidently working - a ``--version`` that both entry points share,
and a ``demo`` subcommand that runs the bundled offline pipeline so a fresh
``pipx install crewui`` shows something on screen without an API key.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from crewui import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crewui",
        description="A Textual TUI for sequential CrewAI pipelines.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"crewui {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    demo = sub.add_parser(
        "demo",
        help="Run the bundled offline demo pipeline in the TUI (no API key needed).",
    )
    demo.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the layout without walking the pipeline.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``crewui`` console script and ``python -m crewui``.

    Returns a process exit code. With no subcommand, prints help and exits 0 -
    the install is proven working by ``--version`` and ``demo``, so a bare
    invocation is informational, not an error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        # Imported lazily so ``crewui --version`` does not pull in textual and
        # construct crewai agents just to print a string.
        from crewui.demo import run_demo

        run_demo(dry_run=args.dry_run)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
