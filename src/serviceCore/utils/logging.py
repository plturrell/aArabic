"""
Structured logging configuration
"""

import logging
import sys
from typing import Any, Dict
from backend.config.settings import settings


def setup_logging() -> None:
    """Configure application logging"""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    if settings.log_format == "json":
        # Use JSON formatter for structured logging
        import json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data: Dict[str, Any] = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                
                if hasattr(record, "extra"):
                    log_data.update(record.extra)
                
                return json.dumps(log_data)
        
        formatter = JSONFormatter()
    else:
        # Use standard text formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

