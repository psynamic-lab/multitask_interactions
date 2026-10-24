"""One logging setup for the training scripts.

Keeps the format the repo already uses in `multi_task_training/scripts/`
(`asctime - levelname - message`) but centralises it, so the sbatch .out files
look the same whichever script wrote them.

Logs go to **stdout**: `tqdm` writes its progress bars to stderr, so keeping the
two apart stops the bars from shredding the log lines. Each record is flushed as
it is emitted, which is what the scattered `print(..., flush=True)` calls were
for -- under sbatch an unflushed stream can sit in a buffer for minutes.

    from mtl.log import get_logger, setup
    logger = get_logger(__name__)
    ...
    setup(args.log_level)   # once, at the top of main()
"""

import logging
import sys

FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup(level="INFO"):
    """Configure root logging to stdout. Safe to call more than once."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()  # replace any handler a library installed on import
    root.addHandler(handler)
    root.setLevel(level)
    return root


def get_logger(name):
    """Module-level logger; call `setup()` once in main() to give it a handler."""
    return logging.getLogger(name)


def add_log_level_argument(parser):
    """Add the standard --log-level flag to an argparse parser."""
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    return parser


__all__ = ["FORMAT", "add_log_level_argument", "get_logger", "setup"]
