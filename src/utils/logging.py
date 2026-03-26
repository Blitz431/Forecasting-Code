"""Structured logging setup for AutoStockAnalyzer."""

import logging
import sys
from pathlib import Path

from config.settings import ROOT_DIR


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
