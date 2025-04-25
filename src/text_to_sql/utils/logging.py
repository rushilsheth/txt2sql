"""
Logging Configuration Module

This module provides enhanced logging configuration for the text-to-SQL system,
including structured logging support for integration with monitoring systems.
"""

import json
import logging
import logging.config
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Define custom log levels
TRACE = 5  # More detailed than DEBUG
logging.addLevelName(TRACE, "TRACE")

# Add trace method to logger
def trace(self, message, *args, **kwargs):
    """Log a message with level TRACE."""
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)

logging.Logger.trace = trace


class StructuredLogRecord(logging.LogRecord):
    """Custom LogRecord class that adds structured data."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tags = {}
        self.structured_data = {}


class StructuredLogger(logging.Logger):
    """Logger class that supports structured logging."""
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Create a LogRecord with structured data support."""
        record = StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if key in ["tags", "structured_data"]:
                    setattr(record, key, extra[key])
                elif key not in ["message", "asctime"] and not key.startswith("_"):
                    setattr(record, key, extra[key])
        return record
    
    def _log_with_tags(self, level, msg, tags=None, structured_data=None, *args, **kwargs):
        """Log with additional tags and structured data."""
        extra = kwargs.get("extra", {})
        if tags:
            extra["tags"] = tags
        if structured_data:
            extra["structured_data"] = structured_data
        kwargs["extra"] = extra
        return super().log(level, msg, *args, **kwargs)
    
    def debug(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with DEBUG level with tags."""
        return self._log_with_tags(logging.DEBUG, msg, tags, structured_data, *args, **kwargs)
    
    def info(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with INFO level with tags."""
        return self._log_with_tags(logging.INFO, msg, tags, structured_data, *args, **kwargs)
    
    def warning(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with WARNING level with tags."""
        return self._log_with_tags(logging.WARNING, msg, tags, structured_data, *args, **kwargs)
    
    def error(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with ERROR level with tags."""
        return self._log_with_tags(logging.ERROR, msg, tags, structured_data, *args, **kwargs)
    
    def critical(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with CRITICAL level with tags."""
        return self._log_with_tags(logging.CRITICAL, msg, tags, structured_data, *args, **kwargs)
    
    def trace(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with TRACE level with tags."""
        return self._log_with_tags(TRACE, msg, tags, structured_data, *args, **kwargs)
    
    def exception(self, msg, *args, tags=None, structured_data=None, **kwargs):
        """Log with ERROR level with exception info and tags."""
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        return self._log_with_tags(logging.ERROR, msg, tags, structured_data, *args, **kwargs)


# Register the StructuredLogger class
logging.setLoggerClass(StructuredLogger)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'CRITICAL': '\033[1;31m',  # Bold Red
        'ERROR': '\033[31m',       # Red
        'WARNING': '\033[33m',     # Yellow
        'INFO': '\033[32m',        # Green
        'DEBUG': '\033[36m',       # Cyan
        'TRACE': '\033[35m',       # Magenta
        'RESET': '\033[0m'         # Reset
    }
    
    def format(self, record):
        """Format the log record with colors."""
        # Skip coloring if not a terminal
        if not sys.stdout.isatty():
            return super().format(record)
        
        # Get the original format
        log_message = super().format(record)
        
        # Add color if the level has a color defined
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        
        return log_message


class JsonFormatter(logging.Formatter):
    """Formats logs as JSON objects for structured logging."""
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.default_keys = [
            'name', 'levelname', 'pathname', 'lineno', 'message', 
            'created', 'thread', 'threadName', 'process'
        ]
    
    def format(self, record):
        """Format log record as JSON."""
        # Start with basic record attributes
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'pathname': record.pathname,
            'lineno': record.lineno,
            'function': record.funcName,
            'thread': record.thread,
            'process': record.process
        }
        
        # Add structured data if available
        if hasattr(record, 'structured_data') and record.structured_data:
            log_data['data'] = record.structured_data
        
        # Add tags if available
        if hasattr(record, 'tags') and record.tags:
            log_data['tags'] = record.tags
        
        # Add exception info if available
        if record.exc_info:
            # Format the exception info
            exc_type, exc_value, exc_tb = record.exc_info
            log_data['exception'] = {
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in self.default_keys and not key.startswith('_') and key not in ['args', 'exc_info', 'exc_text', 'structured_data', 'tags']:
                log_data[key] = value
        
        return json.dumps(log_data)


def configure_logging(config: Dict = None) -> None:
    """
    Configure logging based on configuration.
    
    Args:
        config: Logging configuration dictionary
    """
    config = config or {}
    
    # Get configuration values with defaults
    log_level = config.get("level", "INFO")
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file")
    log_directory = config.get("directory", "logs")
    colored_output = config.get("colored", True)
    component_levels = config.get("component_levels", {})
    structured_output = config.get("structured", True)
    json_output = config.get("json", False)
    
    # Ensure log directory exists if logging to file
    if log_file:
        log_path = Path(log_directory)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to log filename if requested
        if "{timestamp}" in log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_file.replace("{timestamp}", timestamp)
        
        log_file = str(log_path / log_file)
    
    # Create handlers
    handlers = {}
    
    # Console handler
    console_handler = {
        "class": "logging.StreamHandler",
        "level": log_level,
        "stream": "ext://sys.stdout"
    }
    
    if json_output:
        console_handler["formatter"] = "json"
    elif colored_output and sys.stdout.isatty():
        console_handler["formatter"] = "colored"
    else:
        console_handler["formatter"] = "standard"
    
    handlers["console"] = console_handler
    
    # File handler (if specified)
    if log_file:
        file_handler = {
            "class": "logging.FileHandler",
            "level": log_level,
            "filename": log_file,
            "mode": "a"
        }
        
        if json_output or config.get("json_file", False):
            file_handler["formatter"] = "json"
        else:
            file_handler["formatter"] = "standard"
        
        handlers["file"] = file_handler
    
    # Create formatters
    formatters = {
        "standard": {
            "format": log_format
        },
        "colored": {
            "()": "text_to_sql.utils.logging.ColoredFormatter",
            "format": log_format
        },
        "json": {
            "()": "text_to_sql.utils.logging.JsonFormatter"
        }
    }
    
    # Create logger configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": list(handlers.keys())
            }
        }
    }
    
    # Configure component-specific log levels
    for component, level in component_levels.items():
        logging_config["loggers"][component] = {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False
        }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Set log level for TRACE if needed
    if log_level == "TRACE":
        logging.getLogger().setLevel(TRACE)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured at level {log_level}",
        tags={"component": "logging", "action": "configure"}
    )
    
    if log_file:
        logger.info(
            f"Log file: {log_file}", 
            tags={"component": "logging", "action": "configure", "file": log_file}
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level_for_component(component: str, level: Union[str, int]) -> None:
    """
    Set log level for a specific component.
    
    Args:
        component: Component name (logger name)
        level: Log level (string or integer)
    """
    logger = logging.getLogger(component)
    
    # Convert string level to integer if needed
    if isinstance(level, str):
        if level == "TRACE":
            level = TRACE
        else:
            level = getattr(logging, level)
    
    logger.setLevel(level)
    logging.getLogger(__name__).debug(
        f"Set log level for {component} to {logging.getLevelName(level)}",
        tags={"component": "logging", "action": "set_level", "target": component, "level": logging.getLevelName(level)}
    )


def get_all_loggers() -> Dict[str, logging.Logger]:
    """
    Get all configured loggers.
    
    Returns:
        Dictionary mapping logger names to logger instances
    """
    loggers = {}
    
    # Get the root logger
    root = logging.getLogger()
    loggers["root"] = root
    
    # Get all other loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        loggers[name] = logger
    
    return loggers