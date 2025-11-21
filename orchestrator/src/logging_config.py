"""
Logging Configuration
Sets up file-based logging with separate log files for different modules
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file paths
APP_LOG_FILE = LOG_DIR / "app.log"
AGENT_LOG_FILE = LOG_DIR / "agent.log"
MCP_CLIENT_LOG_FILE = LOG_DIR / "mcp_client.log"
GEMINI_CLIENT_LOG_FILE = LOG_DIR / "gemini_client.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Max log file size (10MB)
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


def setup_file_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a file-based logger with rotation
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    # Console handler (optional, for development)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors on console
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_logging():
    """
    Set up all loggers for the orchestrator
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)
    
    # Set up module-specific loggers
    setup_file_logger("voxbank.orchestrator", APP_LOG_FILE, level)
    setup_file_logger("agent", AGENT_LOG_FILE, level)
    setup_file_logger("voxbank.mcp_client", MCP_CLIENT_LOG_FILE, level)
    setup_file_logger("gemini_llm_client", GEMINI_CLIENT_LOG_FILE, level)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    logging.info("Logging configured. Log files in: %s", LOG_DIR)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance (creates if doesn't exist)
    """
    return logging.getLogger(name)

