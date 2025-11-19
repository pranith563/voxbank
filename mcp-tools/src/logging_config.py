"""
Logging configuration for the MCP tools server.

Writes rotating logs to mcp-tools/logs/mcp_server.log.
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MCP_SERVER_LOG_FILE = LOG_DIR / "mcp_server.log"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def _setup_file_logger(name: str, log_file: Path, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
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
    Configure logging for the MCP tools server.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    _setup_file_logger("mcp_server", MCP_SERVER_LOG_FILE, level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

