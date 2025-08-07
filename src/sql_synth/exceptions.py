"""Custom exception classes for SQL Synthesis Agent.

This module defines domain-specific exception classes to provide
better error handling and debugging information.
"""

from typing import Optional, List, Dict, Any


class SQLSynthesisError(Exception):
    """Base exception class for SQL synthesis operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message


class SentimentAnalysisError(SQLSynthesisError):
    """Exception raised during sentiment analysis operations."""
    
    def __init__(
        self, 
        message: str, 
        query: Optional[str] = None,
        model_failures: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.query = query
        self.model_failures = model_failures or []


class SQLGenerationError(SQLSynthesisError):
    """Exception raised during SQL generation operations."""
    
    def __init__(
        self,
        message: str,
        natural_query: Optional[str] = None, 
        partial_sql: Optional[str] = None,
        agent_output: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.natural_query = natural_query
        self.partial_sql = partial_sql
        self.agent_output = agent_output


class SQLSecurityError(SQLSynthesisError):
    """Exception raised when SQL security validation fails."""
    
    def __init__(
        self,
        message: str,
        sql_query: Optional[str] = None,
        violations: Optional[List[str]] = None,
        security_level: str = "high",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.sql_query = sql_query
        self.violations = violations or []
        self.security_level = security_level


class DatabaseConnectionError(SQLSynthesisError):
    """Exception raised for database connection issues."""
    
    def __init__(
        self,
        message: str,
        database_type: Optional[str] = None,
        connection_string: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.database_type = database_type
        # Never log the actual connection string for security
        self.connection_string_hash = hash(connection_string) if connection_string else None


class SQLExecutionError(SQLSynthesisError):
    """Exception raised during SQL query execution."""
    
    def __init__(
        self,
        message: str,
        sql_query: Optional[str] = None,
        execution_time: Optional[float] = None,
        database_error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.sql_query = sql_query
        self.execution_time = execution_time
        self.database_error = database_error


class CacheError(SQLSynthesisError):
    """Exception raised during caching operations."""
    
    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.cache_key = cache_key
        self.operation = operation


class ValidationError(SQLSynthesisError):
    """Exception raised during input validation."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        validation_rules: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.validation_rules = validation_rules or []


class ConfigurationError(SQLSynthesisError):
    """Exception raised for configuration issues."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[type] = None,
        actual_value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.config_key = config_key
        self.expected_type = expected_type
        self.actual_value = actual_value


class ModelLoadingError(SQLSynthesisError):
    """Exception raised when ML models fail to load."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        error_details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.model_name = model_name
        self.model_type = model_type
        self.error_details = error_details


class RateLimitError(SQLSynthesisError):
    """Exception raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        reset_time: Optional[float] = None,
        requests_remaining: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.service = service
        self.reset_time = reset_time
        self.requests_remaining = requests_remaining


def create_error_context(
    operation: str,
    timestamp: Optional[float] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create standardized error context dictionary.
    
    Args:
        operation: The operation being performed when error occurred
        timestamp: Unix timestamp of the error
        user_id: User identifier (if available)
        session_id: Session identifier (if available)
        **kwargs: Additional context parameters
        
    Returns:
        Dictionary containing error context information
    """
    import time
    
    context = {
        "operation": operation,
        "timestamp": timestamp or time.time(),
    }
    
    if user_id:
        context["user_id"] = user_id
    if session_id:
        context["session_id"] = session_id
        
    context.update(kwargs)
    return context


def handle_exception_with_context(
    exception: Exception,
    operation: str,
    logger,
    **context_kwargs
) -> SQLSynthesisError:
    """Convert generic exceptions to domain-specific ones with context.
    
    Args:
        exception: The original exception
        operation: Operation being performed
        logger: Logger instance for recording
        **context_kwargs: Additional context information
        
    Returns:
        Domain-specific SQLSynthesisError with context
    """
    context = create_error_context(operation, **context_kwargs)
    
    # Map common exceptions to domain-specific ones
    exception_mapping = {
        ValueError: ValidationError,
        ConnectionError: DatabaseConnectionError,
        TimeoutError: SQLExecutionError,
        PermissionError: SQLSecurityError,
    }
    
    error_class = exception_mapping.get(type(exception), SQLSynthesisError)
    
    # Create domain-specific error
    domain_error = error_class(
        message=f"{operation} failed: {str(exception)}",
        context=context
    )
    
    # Log with full context
    logger.exception(
        "Operation failed: %s",
        operation,
        extra={
            "error_type": type(exception).__name__,
            "domain_error_type": error_class.__name__,
            "context": context
        }
    )
    
    return domain_error