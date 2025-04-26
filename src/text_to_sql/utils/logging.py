"""
Logging Configuration Module

This module provides enhanced logging configuration for the text-to-SQL system,
including structured logging support with custom parameters such as tags and structured data.
"""

import json
import logging
import logging.config
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

# Define custom log levels
TRACE = 5  # More detailed than DEBUG
logging.addLevelName(TRACE, "TRACE")


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
    """Logger class that supports structured logging with custom tags and structured data."""
    
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        """Create a LogRecord with support for extra structured information."""
        record = StructuredLogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key, value in extra.items():
                if key in ["tags", "structured_data"]:
                    setattr(record, key, value)
                elif key not in ["message", "asctime"] and not key.startswith("_"):
                    setattr(record, key, value)
        return record

    def _log_with_tags(self, level, msg, tags=None, structured_data=None, *args, **kwargs):
        """
        Helper method to extract custom parameters, auto-inject caller info,
        and delegate to the base _log() method.
        """
        extra = kwargs.pop("extra", {})
        if tags:
            extra["tags"] = tags
        if structured_data:
            extra["structured_data"] = structured_data

        # Use inspect.stack() to reliably get caller frame (index 2: current->_log_with_tags->caller)
        stack = inspect.stack()
        caller_frame = stack[2]
        caller_globals = caller_frame.frame.f_globals
        caller_code = caller_frame.frame.f_code
        extra.setdefault("module", caller_globals.get("__name__", "unknown"))
        extra.setdefault("function", caller_code.co_name)
        kwargs["extra"] = extra

        self._log(level, msg, args, **kwargs)

    def debug(self, msg, *args, tags=None, structured_data=None, **kwargs):
        return self._log_with_tags(logging.DEBUG, msg, tags, structured_data, *args, **kwargs)
    
    def info(self, msg, *args, tags=None, structured_data=None, **kwargs):
        return self._log_with_tags(logging.INFO, msg, tags, structured_data, *args, **kwargs)
    
    def warning(self, msg, *args, tags=None, structured_data=None, **kwargs):
        return self._log_with_tags(logging.WARNING, msg, tags, structured_data, *args, **kwargs)
    
    def error(self, msg, *args, tags=None, structured_data=None, **kwargs):
        return self._log_with_tags(logging.ERROR, msg, tags, structured_data, *args, **kwargs)
    
    def critical(self, msg, *args, tags=None, structured_data=None, **kwargs):
        return self._log_with_tags(logging.CRITICAL, msg, tags, structured_data, *args, **kwargs)
    
    def trace(self, msg, *args, tags=None, structured_data=None, **kwargs):
        return self._log_with_tags(TRACE, msg, tags, structured_data, *args, **kwargs)
    
    def exception(self, msg, *args, tags=None, structured_data=None, **kwargs):
        kwargs["exc_info"] = kwargs.get("exc_info", True)
        return self._log_with_tags(logging.ERROR, msg, tags, structured_data, *args, **kwargs)


# Register our custom StructuredLogger before any logger is created
logging.setLoggerClass(StructuredLogger)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for terminal output."""
    
    COLORS = {
        'CRITICAL': '\033[1;31m',  # Bold Red
        'ERROR': '\033[31m',       # Red
        'WARNING': '\033[33m',     # Yellow
        'INFO': '\033[32m',        # Green
        'DEBUG': '\033[36m',       # Cyan
        'TRACE': '\033[35m',       # Magenta
        'RESET': '\033[0m'         # Reset color
    }
    
    def format(self, record):
        log_message = super().format(record)
        if sys.stdout.isatty() and record.levelname in self.COLORS:
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
        if hasattr(record, 'structured_data') and record.structured_data:
            log_data['data'] = record.structured_data
        if hasattr(record, 'tags') and record.tags:
            log_data['tags'] = record.tags
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_data['exception'] = {
                'type': exc_type.__name__,
                'message': str(exc_value),
                'traceback': self.formatException(record.exc_info)
            }
        for key, value in record.__dict__.items():
            if key not in self.default_keys and not key.startswith('_') \
               and key not in ['args', 'exc_info', 'exc_text', 'structured_data', 'tags']:
                log_data[key] = value
        return json.dumps(log_data)


def configure_logging(config: Dict = None) -> None:
    """
    Configure logging based on the provided configuration dictionary.
    
    Config keys:
      - level: Log level (default "INFO")
      - format: Log format (default "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
      - file: Log file name pattern (e.g., "app_{timestamp}.log")
      - directory: Directory to store log files (default "logs")
      - colored: Enable colored console output (default True)
      - component_levels: Dictionary of component-specific log levels
      - structured: Enable structured logging (default True)
      - json: Use JSON formatting for console output (default False)
    """
    config = config or {}
    log_level = config.get("level", "INFO")
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file")
    log_directory = config.get("directory", "logs")
    colored_output = config.get("colored", True)
    component_levels = config.get("component_levels", {})
    json_output = config.get("json", False)
    
    # If logging to file, ensure directory exists and format filename
    if log_file:
        log_path = Path(log_directory)
        log_path.mkdir(parents=True, exist_ok=True)
        if "{timestamp}" in log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_file.replace("{timestamp}", timestamp)
        log_file = str(log_path / log_file)
    
    # Build handlers configuration
    handlers = {}
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
    
    if log_file:
        file_handler = {
            "class": "logging.FileHandler",
            "level": log_level,
            "filename": log_file,
            "mode": "a"
        }
        # Use JSON formatter for file if requested; otherwise standard
        file_handler["formatter"] = "json" if json_output or config.get("json_file", False) else "standard"
        handlers["file"] = file_handler
    
    # Build formatters configuration
    formatters = {
        "standard": {"format": log_format},
        "colored": {"()": "text_to_sql.utils.logging.ColoredFormatter", "format": log_format},
        "json": {"()": "text_to_sql.utils.logging.JsonFormatter"}
    }
    
    # Build the root logger configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {  # Root logger
            "": {
                "level": log_level,
                "handlers": list(handlers.keys())
            }
        }
    }
    
    for component, level in component_levels.items():
        logging_config["loggers"][component] = {
            "level": level,
            "handlers": list(handlers.keys()),
            "propagate": False
        }
    
    logging.config.dictConfig(logging_config)
    
    # Set TRACE level if requested
    if log_level.upper() == "TRACE":
        logging.getLogger().setLevel(TRACE)
    
    # Log the configuration details
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level {log_level}",
                tags={"component": "logging", "action": "configure"})
    if log_file:
        logger.info(f"Log file: {log_file}",
                    tags={"component": "logging", "action": "configure", "file": log_file})


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name.
        
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def set_log_level_for_component(component: str, level: Union[str, int]) -> None:
    """
    Set log level for a specific component.
    
    Args:
        component: Component name (logger name).
        level: Log level (string or integer).
    """
    logger = logging.getLogger(component)
    if isinstance(level, str):
        level = TRACE if level.upper() == "TRACE" else getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logging.getLogger(__name__).debug(
        f"Set log level for {component} to {logging.getLevelName(level)}",
        tags={"component": "logging", "action": "set_level", "target": component,
              "level": logging.getLevelName(level)}
    )


def get_all_loggers() -> Dict[str, logging.Logger]:
    """
    Get all configured loggers.
    
    Returns:
        Dictionary mapping logger names to logger instances.
    """
    loggers = {"root": logging.getLogger()}
    for name in logging.root.manager.loggerDict:
        loggers[name] = logging.getLogger(name)
    return loggers