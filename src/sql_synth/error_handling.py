"""Advanced error handling and recovery system for SQL synthesis agent.

This module provides comprehensive error handling, retry mechanisms,
and graceful degradation for robust production operation.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for targeted handling strategies."""
    DATABASE_CONNECTION = "database_connection"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Detailed error context for analysis and recovery."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    timestamp: float = 0.0
    retry_count: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


class SQLSynthesisError(Exception):
    """Base exception for SQL synthesis operations."""

    def __init__(self, context: ErrorContext) -> None:
        self.context = context
        super().__init__(context.message)


class DatabaseConnectionError(SQLSynthesisError):
    """Database connection related errors."""


class SQLGenerationError(SQLSynthesisError):
    """SQL generation related errors."""


class SQLExecutionError(SQLSynthesisError):
    """SQL execution related errors."""


class ValidationError(SQLSynthesisError):
    """Input/output validation errors."""


class RateLimitError(SQLSynthesisError):
    """Rate limiting errors."""


class TimeoutError(SQLSynthesisError):
    """Operation timeout errors."""


class ErrorClassifier:
    """Intelligent error classification system."""

    ERROR_PATTERNS = {
        ErrorCategory.DATABASE_CONNECTION: [
            "connection refused", "connection timeout", "connection failed",
            "could not connect", "network unreachable", "authentication failed",
        ],
        ErrorCategory.SQL_GENERATION: [
            "generation failed", "model error", "api error", "rate limit",
            "invalid response", "parsing error",
        ],
        ErrorCategory.SQL_EXECUTION: [
            "syntax error", "table doesn't exist", "column doesn't exist",
            "permission denied", "execution timeout",
        ],
        ErrorCategory.VALIDATION: [
            "validation failed", "invalid input", "security violation",
            "dangerous operation", "malformed query",
        ],
        ErrorCategory.RATE_LIMITING: [
            "rate limit", "quota exceeded", "too many requests",
            "throttled", "limit exceeded",
        ],
        ErrorCategory.TIMEOUT: [
            "timeout", "timed out", "deadline exceeded", "operation took too long",
        ],
        ErrorCategory.RESOURCE_EXHAUSTION: [
            "memory error", "disk space", "resource limit", "out of memory",
        ],
    }

    @classmethod
    def classify_error(cls, error: Exception) -> ErrorCategory:
        """Classify error based on exception type and message.

        Args:
            error: Exception to classify

        Returns:
            Classified error category
        """
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check error message patterns
        for category, patterns in cls.ERROR_PATTERNS.items():
            if any(pattern in error_message or pattern in error_type for pattern in patterns):
                return category

        # Check exception types
        if "connection" in error_type or "database" in error_type:
            return ErrorCategory.DATABASE_CONNECTION
        if "sql" in error_type or "query" in error_type:
            return ErrorCategory.SQL_EXECUTION
        if "validation" in error_type or "security" in error_type:
            return ErrorCategory.VALIDATION
        if "timeout" in error_type:
            return ErrorCategory.TIMEOUT
        if "auth" in error_type:
            return ErrorCategory.AUTHENTICATION

        return ErrorCategory.UNKNOWN

    @classmethod
    def determine_severity(cls, category: ErrorCategory, error: Exception) -> ErrorSeverity:
        """Determine error severity based on category and context.

        Args:
            category: Error category
            error: Original exception

        Returns:
            Determined severity level
        """
        severity_mapping = {
            ErrorCategory.DATABASE_CONNECTION: ErrorSeverity.HIGH,
            ErrorCategory.SQL_GENERATION: ErrorSeverity.MEDIUM,
            ErrorCategory.SQL_EXECUTION: ErrorSeverity.MEDIUM,
            ErrorCategory.VALIDATION: ErrorSeverity.HIGH,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.CRITICAL,
            ErrorCategory.RATE_LIMITING: ErrorSeverity.LOW,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.RESOURCE_EXHAUSTION: ErrorSeverity.HIGH,
            ErrorCategory.EXTERNAL_SERVICE: ErrorSeverity.MEDIUM,
            ErrorCategory.CONFIGURATION: ErrorSeverity.HIGH,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
        }

        return severity_mapping.get(category, ErrorSeverity.MEDIUM)


class RetryStrategy:
    """Intelligent retry strategy with backoff and circuit breaking."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter

    def should_retry(self, error_context: ErrorContext) -> bool:
        """Determine if operation should be retried.

        Args:
            error_context: Error context containing retry information

        Returns:
            True if should retry, False otherwise
        """
        if error_context.retry_count >= self.max_attempts:
            return False

        # Don't retry critical errors or validation failures
        if error_context.severity == ErrorSeverity.CRITICAL:
            return False

        if error_context.category in [ErrorCategory.VALIDATION, ErrorCategory.AUTHENTICATION]:
            return False

        # Retry transient errors
        retryable_categories = [
            ErrorCategory.DATABASE_CONNECTION,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.RATE_LIMITING,
            ErrorCategory.TIMEOUT,
            ErrorCategory.SQL_GENERATION,
        ]

        return error_context.category in retryable_categories

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter

        return delay


class ErrorRecoveryManager:
    """Comprehensive error recovery and handling manager."""

    def __init__(self, retry_strategy: Optional[RetryStrategy] = None) -> None:
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.error_history: list[ErrorContext] = []
        self.recovery_handlers: dict[ErrorCategory, list[Callable]] = {}

    def register_recovery_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[ErrorContext], Optional[Any]],
    ) -> None:
        """Register a recovery handler for specific error category.

        Args:
            category: Error category to handle
            handler: Recovery handler function
        """
        if category not in self.recovery_handlers:
            self.recovery_handlers[category] = []
        self.recovery_handlers[category].append(handler)

    def handle_error(
        self,
        error: Exception,
        operation_name: str = "unknown",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ErrorContext:
        """Handle error with classification and recovery attempts.

        Args:
            error: Exception that occurred
            operation_name: Name of the operation that failed
            metadata: Additional context metadata

        Returns:
            Error context with handling information
        """
        # Classify error
        category = ErrorClassifier.classify_error(error)
        severity = ErrorClassifier.determine_severity(category, error)

        # Create error context
        error_context = ErrorContext(
            error_id=f"{operation_name}_{int(time.time())}",
            category=category,
            severity=severity,
            message=str(error),
            original_exception=error,
            metadata=metadata or {},
        )

        # Add to error history
        self.error_history.append(error_context)

        # Log error with appropriate level
        self._log_error(error_context)

        # Attempt recovery
        self._attempt_recovery(error_context)

        return error_context

    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity.

        Args:
            error_context: Error context to log
        """
        log_message = (
            f"Error [{error_context.error_id}] "
            f"{error_context.category.value} - {error_context.message}"
        )

        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=error_context.original_exception)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=error_context.original_exception)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _attempt_recovery(self, error_context: ErrorContext) -> Optional[Any]:
        """Attempt error recovery using registered handlers.

        Args:
            error_context: Error context for recovery

        Returns:
            Recovery result if successful, None otherwise
        """
        handlers = self.recovery_handlers.get(error_context.category, [])

        for handler in handlers:
            try:
                result = handler(error_context)
                if result is not None:
                    logger.info(f"Error recovery successful for {error_context.error_id}")
                    return result
            except Exception as recovery_error:
                logger.warning(
                    f"Recovery handler failed for {error_context.error_id}: {recovery_error}",
                )

        return None

    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics and patterns.

        Returns:
            Dictionary containing error statistics
        """
        if not self.error_history:
            return {"total_errors": 0}

        # Count by category
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        # Recent error rate (last hour)
        recent_errors = [
            error for error in self.error_history
            if time.time() - error.timestamp < 3600
        ]

        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "most_common_category": max(category_counts.keys(), key=lambda k: category_counts[k]) if category_counts else None,
        }


@contextmanager
def error_context(
    operation_name: str,
    recovery_manager: ErrorRecoveryManager,
    metadata: Optional[dict[str, Any]] = None,
):
    """Context manager for comprehensive error handling.

    Args:
        operation_name: Name of the operation being performed
        recovery_manager: Error recovery manager instance
        metadata: Additional context metadata

    Yields:
        None

    Raises:
        SQLSynthesisError: Re-raised with enhanced context
    """
    try:
        yield
    except Exception as error:
        error_context_obj = recovery_manager.handle_error(error, operation_name, metadata)

        # Raise appropriate SQL synthesis error
        if error_context_obj.category == ErrorCategory.DATABASE_CONNECTION:
            raise DatabaseConnectionError(error_context_obj) from error
        elif error_context_obj.category == ErrorCategory.SQL_GENERATION:
            raise SQLGenerationError(error_context_obj) from error
        elif error_context_obj.category == ErrorCategory.SQL_EXECUTION:
            raise SQLExecutionError(error_context_obj) from error
        elif error_context_obj.category == ErrorCategory.VALIDATION:
            raise ValidationError(error_context_obj) from error
        elif error_context_obj.category == ErrorCategory.RATE_LIMITING:
            raise RateLimitError(error_context_obj) from error
        elif error_context_obj.category == ErrorCategory.TIMEOUT:
            raise TimeoutError(error_context_obj) from error
        else:
            raise SQLSynthesisError(error_context_obj) from error


def retry_with_backoff(
    func: Callable[..., T],
    retry_strategy: Optional[RetryStrategy] = None,
    recovery_manager: Optional[ErrorRecoveryManager] = None,
) -> Callable[..., T]:
    """Decorator for automatic retry with intelligent backoff.

    Args:
        func: Function to wrap with retry logic
        retry_strategy: Retry strategy configuration
        recovery_manager: Error recovery manager

    Returns:
        Wrapped function with retry capability
    """
    strategy = retry_strategy or RetryStrategy()
    manager = recovery_manager or ErrorRecoveryManager()

    def wrapper(*args, **kwargs) -> T:
        last_error_context = None

        for attempt in range(strategy.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                error_context_obj = manager.handle_error(
                    error,
                    func.__name__,
                    {"attempt": attempt, "args": str(args)[:100]},
                )
                error_context_obj.retry_count = attempt
                last_error_context = error_context_obj

                if not strategy.should_retry(error_context_obj):
                    break

                if attempt < strategy.max_attempts - 1:  # Don't delay on last attempt
                    delay = strategy.calculate_delay(attempt)
                    logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 2})")
                    time.sleep(delay)

        # All retries exhausted
        if last_error_context:
            if last_error_context.category == ErrorCategory.DATABASE_CONNECTION:
                raise DatabaseConnectionError(last_error_context)
            if last_error_context.category == ErrorCategory.SQL_GENERATION:
                raise SQLGenerationError(last_error_context)
            if last_error_context.category == ErrorCategory.SQL_EXECUTION:
                raise SQLExecutionError(last_error_context)
            raise SQLSynthesisError(last_error_context)

        # Should never reach here
        msg = "Retry logic error: no exception context available"
        raise RuntimeError(msg)

    return wrapper


# Global error recovery manager instance
global_error_manager = ErrorRecoveryManager()


def register_default_recovery_handlers() -> None:
    """Register default recovery handlers for common error scenarios."""

    def connection_recovery_handler(error_context: ErrorContext) -> Optional[str]:
        """Handle database connection recovery."""
        logger.info("Attempting database connection recovery...")
        # Could implement connection pool refresh, fallback database, etc.
        return None

    def rate_limit_recovery_handler(error_context: ErrorContext) -> Optional[str]:
        """Handle rate limit recovery."""
        logger.info("Implementing rate limit backoff...")
        # Could implement exponential backoff, queue management, etc.
        return None

    def generation_recovery_handler(error_context: ErrorContext) -> Optional[str]:
        """Handle SQL generation recovery."""
        logger.info("Attempting SQL generation fallback...")
        # Could implement fallback models, simpler generation, etc.
        return None

    # Register handlers
    global_error_manager.register_recovery_handler(
        ErrorCategory.DATABASE_CONNECTION,
        connection_recovery_handler,
    )
    global_error_manager.register_recovery_handler(
        ErrorCategory.RATE_LIMITING,
        rate_limit_recovery_handler,
    )
    global_error_manager.register_recovery_handler(
        ErrorCategory.SQL_GENERATION,
        generation_recovery_handler,
    )


# Initialize default handlers
register_default_recovery_handlers()
