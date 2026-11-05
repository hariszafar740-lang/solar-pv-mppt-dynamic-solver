"""Module entry point allowing ``python -m pv_mppt`` invocation."""

from __future__ import annotations

import sys

from pv_mppt.cli import main

if __name__ == "__main__":
    sys.exit(main())
