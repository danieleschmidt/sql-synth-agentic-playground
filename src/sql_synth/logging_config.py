"""Advanced logging configuration for SQL synthesis agent.

This module provides structured logging, performance monitoring,
and security event tracking for production deployments.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Any, Optional

import structlog


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "getMessage",
            ):
                log_data[key] = value

        return json.dumps(log_data, default=str)


class SecurityEventFilter(logging.Filter):
    """Filter for security-related log events."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter security events for special handling."""
        security_keywords = [
            "injection", "attack", "unauthorized", "breach",
            "violation", "malicious", "suspicious",
        ]

        message = record.getMessage().lower()
        return any(keyword in message for keyword in security_keywords)


class PerformanceEventFilter(logging.Filter):
    """Filter for performance-related log events."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter performance events for monitoring."""
        return hasattr(record, "execution_time") or hasattr(record, "response_time")


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_json: bool = True,
    enable_security_log: bool = True,
    enable_performance_log: bool = True,
) -> None:
    """Set up comprehensive logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for stdout only)
        enable_json: Whether to use JSON formatting
        enable_security_log: Whether to enable security event logging
        enable_performance_log: Whether to enable performance logging
    """
    # Configure structlog for structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if enable_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ),
        )

    root_logger.addHandler(console_handler)

    # File handlers (if log directory is specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

        # General application log
        app_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "application.log"),
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        ))
        root_logger.addHandler(app_handler)

        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "errors.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        ))
        root_logger.addHandler(error_handler)

        # Security log
        if enable_security_log:
            security_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, "security.log"),
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=20,  # Keep more security logs
            )
            security_handler.setLevel(logging.WARNING)
            security_handler.addFilter(SecurityEventFilter())
            security_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ))
            root_logger.addHandler(security_handler)

        # Performance log
        if enable_performance_log:
            performance_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, "performance.log"),
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=5,
            )
            performance_handler.setLevel(logging.INFO)
            performance_handler.addFilter(PerformanceEventFilter())
            performance_handler.setFormatter(JSONFormatter() if enable_json else logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ))
            root_logger.addHandler(performance_handler)

    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    # Note: Correlation ID processor is added inline above
    # to avoid configuration conflicts

    logger = structlog.get_logger(__name__)
    logger.info("Logging system initialized", level=level, json_enabled=enable_json)


def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to log events for request tracing."""
    # Ensure event_dict is a dictionary
    if not isinstance(event_dict, dict):
        event_dict = {"message": str(event_dict)}

    # Try to get correlation ID from various sources
    correlation_id = None

    # Check if running in Streamlit context
    try:
        import streamlit as st
        if hasattr(st, "session_state") and hasattr(st.session_state, "correlation_id"):
            correlation_id = st.session_state.correlation_id
    except (ImportError, Exception):
        pass

    # Check environment variable
    if not correlation_id:
        correlation_id = os.environ.get("CORRELATION_ID")

    # Generate if not found
    if not correlation_id:
        import uuid
        correlation_id = str(uuid.uuid4())[:8]

    event_dict["correlation_id"] = correlation_id
    return event_dict


class SQLSynthLogger:
    """Specialized logger for SQL synthesis operations."""

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)

    def log_query_generation(
        self,
        user_query: str,
        generated_sql: Optional[str],
        success: bool,
        generation_time: float,
        error: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log SQL query generation events."""
        self.logger.info(
            "SQL generation completed",
            user_query=user_query[:100] + "..." if len(user_query) > 100 else user_query,
            sql_generated=bool(generated_sql),
            success=success,
            generation_time=generation_time,
            error=error,
            **kwargs,
        )

    def log_query_execution(
        self,
        sql_query: str,
        success: bool,
        execution_time: float,
        rows_affected: Optional[int] = None,
        error: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log SQL query execution events."""
        self.logger.info(
            "SQL execution completed",
            sql_query=sql_query[:200] + "..." if len(sql_query) > 200 else sql_query,
            success=success,
            execution_time=execution_time,
            rows_affected=rows_affected,
            error=error,
            **kwargs,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_input: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log security-related events."""
        self.logger.warning(
            "Security event detected",
            event_type=event_type,
            severity=severity,
            description=description,
            user_input=user_input[:100] + "..." if user_input and len(user_input) > 100 else user_input,
            **kwargs,
        )

    def log_performance_metric(
        self,
        operation: str,
        duration: float,
        resource_usage: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration=duration,
            resource_usage=resource_usage,
            **kwargs,
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log error events with context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            exc_info=error,
            **kwargs,
        )


# Global logger instance
sql_synth_logger = SQLSynthLogger("sql_synth")


def get_logger(name: str) -> SQLSynthLogger:
    """Get a specialized logger instance."""
    return SQLSynthLogger(name)


# Auto-setup logging from environment
def auto_setup_logging() -> None:
    """Automatically set up logging based on environment variables."""
    level = os.environ.get("LOG_LEVEL", "INFO")
    log_dir = os.environ.get("LOG_DIR")
    enable_json = os.environ.get("LOG_JSON", "true").lower() == "true"

    setup_logging(
        level=level,
        log_dir=log_dir,
        enable_json=enable_json,
    )


# Initialize logging on import if not already configured
if not logging.getLogger().handlers:
    auto_setup_logging()
