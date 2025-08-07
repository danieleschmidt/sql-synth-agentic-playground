"""Resilience patterns for SQL Synthesis Agent.

This module provides circuit breaker, retry, timeout, and rate limiting
patterns to make the system more resilient to failures and overload.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit is open, requests fail fast
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds to wait before trying again
    expected_exception: type = Exception # Exception type that counts as failure
    success_threshold: int = 3          # Successes needed to close circuit in half-open


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0 
    failed_requests: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    state_changes: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker logic."""
        with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit should remain open
            if self.state == CircuitState.OPEN:
                if time.time() - (self.stats.last_failure_time or 0) < self.config.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN - request failed fast")
                else:
                    # Try to recover
                    self._transition_to_half_open()
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        self.stats.successful_requests += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        
        # In half-open state, check if we should close the circuit
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.stats.failed_requests += 1
        self.stats.total_failures += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.stats.consecutive_failures >= self.config.failure_threshold):
            self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state reopens the circuit
            self._transition_to_open()
    
    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        self.state = CircuitState.OPEN
        self.stats.state_changes.append(f"OPEN at {datetime.now().isoformat()}")
        logger.warning("Circuit breaker opened due to failures", extra={
            "consecutive_failures": self.stats.consecutive_failures,
            "failure_threshold": self.config.failure_threshold
        })
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.stats.consecutive_successes = 0
        self.stats.state_changes.append(f"HALF_OPEN at {datetime.now().isoformat()}")
        logger.info("Circuit breaker transitioned to HALF_OPEN - testing recovery")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self.stats.state_changes.append(f"CLOSED at {datetime.now().isoformat()}")
        logger.info("Circuit breaker closed - service recovered", extra={
            "consecutive_successes": self.stats.consecutive_successes
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "stats": self.stats.__dict__,
            "config": self.config.__dict__
        }


class RetryConfig:
    """Configuration for retry mechanism."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


def with_retry(config: RetryConfig) -> Callable:
    """Decorator to add retry logic to functions.
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info("Function succeeded on retry", extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "total_attempts": config.max_attempts
                        })
                    return result
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error("Function failed after all retry attempts", extra={
                            "function": func.__name__,
                            "attempts": attempt,
                            "error": str(e)
                        })
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.delay * (config.backoff_factor ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning("Function failed, retrying", extra={
                        "function": func.__name__,
                        "attempt": attempt,
                        "delay": delay,
                        "error": str(e)
                    })
                    
                    time.sleep(delay)
            
            # All attempts exhausted
            raise last_exception
            
        return wrapper
    return decorator


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        """Initialize rate limiter.
        
        Args:
            rate: Tokens per second
            capacity: Maximum number of tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.RLock()
    
    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens
            
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self._lock:
                self._refill_tokens()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
            
            # Check timeout
            if timeout is not None and time.time() - start_time >= timeout:
                return False
            
            # Wait a bit before trying again
            time.sleep(0.01)
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now


class TimeoutError(Exception):
    """Exception raised when operation times out."""
    pass


def with_timeout(timeout_seconds: float) -> Callable:
    """Decorator to add timeout to functions.
    
    Args:
        timeout_seconds: Maximum execution time
        
    Returns:
        Decorated function with timeout
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
            
            # Set up timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
                
        return wrapper
    return decorator


class BulkheadIsolation:
    """Bulkhead isolation pattern to separate resources."""
    
    def __init__(self, max_concurrent: int):
        """Initialize bulkhead with concurrency limit.
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.active_count = 0
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply bulkhead isolation."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            acquired = self.semaphore.acquire(blocking=False)
            if not acquired:
                raise Exception(f"Bulkhead limit ({self.max_concurrent}) exceeded")
            
            try:
                with self._lock:
                    self.active_count += 1
                
                logger.debug("Bulkhead acquired", extra={
                    "function": func.__name__,
                    "active_count": self.active_count,
                    "max_concurrent": self.max_concurrent
                })
                
                return func(*args, **kwargs)
            finally:
                with self._lock:
                    self.active_count -= 1
                self.semaphore.release()
                
        return wrapper


# Predefined configurations for common scenarios
SENTIMENT_ANALYSIS_CIRCUIT_BREAKER = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception,
    success_threshold=2
))

SQL_GENERATION_CIRCUIT_BREAKER = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=Exception,
    success_threshold=3
))

DATABASE_CIRCUIT_BREAKER = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception,
    success_threshold=2
))

# Retry configurations
QUICK_RETRY = RetryConfig(
    max_attempts=3,
    delay=0.5,
    backoff_factor=1.5,
    max_delay=5.0
)

STANDARD_RETRY = RetryConfig(
    max_attempts=3,
    delay=1.0,
    backoff_factor=2.0,
    max_delay=30.0
)

PERSISTENT_RETRY = RetryConfig(
    max_attempts=5,
    delay=2.0,
    backoff_factor=2.0,
    max_delay=60.0
)

# Rate limiters
API_RATE_LIMITER = RateLimiter(rate=10.0, capacity=50)  # 10 requests/sec, burst of 50
DATABASE_RATE_LIMITER = RateLimiter(rate=5.0, capacity=20)  # 5 queries/sec, burst of 20

# Bulkhead isolation
SQL_GENERATION_BULKHEAD = BulkheadIsolation(max_concurrent=5)
DATABASE_BULKHEAD = BulkheadIsolation(max_concurrent=10)