"""
Logging configuration for the mock-bank service.

Creates rotating file-based loggers under mock-bank/logs/.
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

# logs directory (mock-bank/logs)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MOCK_BANK_LOG_FILE = LOG_DIR / "mock_bank.log"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def _setup_file_logger(name: str, log_file: Path, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers if setup_logging is called multiple times
    logger.handlers = []

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def setup_logging() -> None:
    """
    Configure root + service loggers for mock-bank.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Service logger
    _setup_file_logger("mock_bank", MOCK_BANK_LOG_FILE, level)

    # Keep SQLAlchemy logs informative but not too noisy
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper to get a named logger.
    """
    return logging.getLogger(name)

