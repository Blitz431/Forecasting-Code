import logging
import sys
from pathlib import Path

from config.settings import ROOT_DIR

"""
Purpose: Centralized logger factory — creates named loggers with console and file handlers.

Connections:
  - config/settings.py: reads ROOT_DIR to resolve logs/ directory path
  - Used by: virtually every module calls setup_logger(__name__) at import time

In:  logger name (str), optional log level
Out: configured logging.Logger; appends to logs/autostockanalyzer.log
"""


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console and optional file output.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (logs/ directory)
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "autostockanalyzer.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
