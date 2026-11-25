"""Compatibility command: run a supplied ablation configuration with verified resume."""

import sys

from .run import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--resume"]
    main()
