"""Advanced logging configuration for SQL Synthesis Agent.

This module provides comprehensive logging setup with structured logging,
performance tracking, security audit trails, and error correlation.
"""

import logging
import logging.handlers
import json
import sys
import time
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                if key in ['args', 'msg', 'levelno', 'pathname', 'filename', 
                          'created', 'msecs', 'relativeCreated', 'thread', 
                          'threadName', 'processName', 'process']:
                    continue
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Logger for performance monitoring and metrics."""
    
    def __init__(self, logger_name: str = "sql_synth.performance"):
        self.logger = logging.getLogger(logger_name)
        
    def log_operation_start(self, operation: str, **context) -> str:
        """Log the start of an operation.
        
        Args:
            operation: Name of the operation
            **context: Additional context information
            
        Returns:
            Operation ID for correlation
        """
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        
        self.logger.info(
            "Operation started",
            extra={
                "operation_id": operation_id,
                "operation": operation,
                "phase": "start",
                "start_time": time.time(),
                **context
            }
        )
        
        return operation_id
    
    def log_operation_end(
        self, 
        operation_id: str, 
        operation: str, 
        success: bool = True,
        duration: Optional[float] = None,
        **context
    ) -> None:
        """Log the end of an operation.
        
        Args:
            operation_id: Operation ID from log_operation_start
            operation: Name of the operation
            success: Whether the operation succeeded
            duration: Operation duration in seconds
            **context: Additional context information
        """
        self.logger.info(
            "Operation completed",
            extra={
                "operation_id": operation_id,
                "operation": operation,
                "phase": "end",
                "success": success,
                "duration_seconds": duration,
                "end_time": time.time(),
                **context
            }
        )


class SecurityLogger:
    """Logger for security events and audit trails."""
    
    def __init__(self, logger_name: str = "sql_synth.security"):
        self.logger = logging.getLogger(logger_name)
        
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **context
    ) -> None:
        """Log a security event.
        
        Args:
            event_type: Type of security event
            severity: Event severity (low, medium, high, critical)
            description: Event description
            user_id: User identifier (if available)
            ip_address: IP address (if available)
            **context: Additional context information
        """
        self.logger.warning(
            "Security event",
            extra={
                "event_type": event_type,
                "severity": severity,
                "description": description,
                "user_id": user_id,
                "ip_address": ip_address,
                "timestamp": time.time(),
                **context
            }
        )
    
    def log_sql_security_violation(
        self,
        query: str,
        violations: list,
        user_id: Optional[str] = None,
        **context
    ) -> None:
        """Log SQL security violations.
        
        Args:
            query: The SQL query that violated security rules
            violations: List of security violations
            user_id: User identifier (if available)
            **context: Additional context information
        """
        self.log_security_event(
            event_type="sql_security_violation",
            severity="high",
            description="SQL query blocked due to security violations",
            user_id=user_id,
            sql_query=query[:200],  # Truncate for logging
            violations=violations,
            **context
        )


class ApplicationLogger:
    """Main application logger with structured logging."""
    
    def __init__(self, logger_name: str = "sql_synth"):
        self.logger = logging.getLogger(logger_name)
        self.performance = PerformanceLogger()
        self.security = SecurityLogger()
        
    def log_query_generation(
        self,
        natural_query: str,
        generated_sql: Optional[str] = None,
        sentiment_data: Optional[Dict] = None,
        success: bool = True,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        **context
    ) -> None:
        """Log SQL query generation events.
        
        Args:
            natural_query: Original natural language query
            generated_sql: Generated SQL query
            sentiment_data: Sentiment analysis results
            success: Whether generation succeeded
            duration: Generation time in seconds
            error: Error message if failed
            **context: Additional context information
        """
        log_data = {
            "event_type": "query_generation",
            "natural_query": natural_query[:500],  # Truncate
            "success": success,
            "duration_seconds": duration,
        }
        
        if generated_sql:
            log_data["generated_sql"] = generated_sql[:1000]  # Truncate
            
        if sentiment_data:
            log_data["sentiment"] = sentiment_data
            
        if error:
            log_data["error"] = error
            
        log_data.update(context)
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            "Query generation completed",
            extra=log_data
        )
    
    def log_query_execution(
        self,
        sql_query: str,
        row_count: Optional[int] = None,
        success: bool = True,
        duration: Optional[float] = None,
        error: Optional[str] = None,
        **context
    ) -> None:
        """Log SQL query execution events.
        
        Args:
            sql_query: SQL query executed
            row_count: Number of rows returned
            success: Whether execution succeeded
            duration: Execution time in seconds
            error: Error message if failed
            **context: Additional context information
        """
        log_data = {
            "event_type": "query_execution",
            "sql_query": sql_query[:1000],  # Truncate
            "success": success,
            "duration_seconds": duration,
            "row_count": row_count,
        }
        
        if error:
            log_data["error"] = error
            
        log_data.update(context)
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            "Query execution completed", 
            extra=log_data
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_structured_logging: bool = True,
    enable_file_logging: bool = True,
    enable_rotation: bool = True,
    max_file_size_mb: int = 100,
    backup_count: int = 5
) -> None:
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_structured_logging: Whether to use JSON structured logging
        enable_file_logging: Whether to log to files
        enable_rotation: Whether to use rotating file handlers
        max_file_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_structured_logging:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Main application log
        if enable_rotation:
            app_handler = logging.handlers.RotatingFileHandler(
                log_dir / "sql_synth.log",
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
        else:
            app_handler = logging.FileHandler(log_dir / "sql_synth.log")
            
        app_handler.setLevel(getattr(logging, log_level.upper()))
        app_handler.setFormatter(
            StructuredFormatter() if enable_structured_logging
            else logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        
        # Add to specific loggers
        logging.getLogger("sql_synth").addHandler(app_handler)
        
        # Performance log
        perf_handler = logging.FileHandler(log_dir / "performance.log")
        perf_handler.setFormatter(StructuredFormatter())
        logging.getLogger("sql_synth.performance").addHandler(perf_handler)
        
        # Security log
        security_handler = logging.FileHandler(log_dir / "security.log")
        security_handler.setFormatter(StructuredFormatter())
        logging.getLogger("sql_synth.security").addHandler(security_handler)
        
        # Error log (errors only)
        error_handler = logging.FileHandler(log_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
    
    logger = logging.getLogger("sql_synth.logging")
    logger.info("Logging configuration completed", extra={
        "log_level": log_level,
        "log_dir": str(log_dir) if log_dir else None,
        "structured_logging": enable_structured_logging,
        "file_logging": enable_file_logging,
        "rotation_enabled": enable_rotation
    })


# Global logger instances
app_logger = ApplicationLogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()