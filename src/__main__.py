"""Module entry point allowing ``python -m src`` invocation."""

from __future__ import annotations

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
