"""``python -m crewui`` entry point.

Kept to two lines and a guard so it mirrors the console script exactly: both
routes go through ``crewui.cli:main``, so a wheel cannot ship one working and
the other broken.
"""

from __future__ import annotations

import sys

from crewui.cli import main

if __name__ == "__main__":
    sys.exit(main())
