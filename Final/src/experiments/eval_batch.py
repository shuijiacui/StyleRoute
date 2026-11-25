"""Compatibility command: run the evaluation matrix with verified resume."""

import sys

from .run import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--split", "evaluation", "--resume"]
    main()
