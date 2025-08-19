"""Advanced resilience and fault tolerance system for SQL synthesis.

This module implements comprehensive resilience patterns including circuit breakers,
bulkheads, timeouts, retries, and adaptive failure recovery mechanisms.
"""

import logging
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class FailureType(Enum):
    """Types of failures that can occur."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


@dataclass
class FailureMetrics:
    """Metrics for tracking failures."""
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    response_times: list[float] = field(default_factory=list)


@dataclass
class ResilienceConfig:
    """Configuration for resilience mechanisms."""
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    max_retries: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    timeout_seconds: float = 30.0
    bulkhead_max_concurrent: int = 10
    adaptive_timeout_enabled: bool = True
    health_check_interval: int = 30


class CircuitBreaker:
    """Advanced circuit breaker with adaptive behavior."""

    def __init__(self, config: ResilienceConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitState.CLOSED
        self.metrics = FailureMetrics()
        self.last_state_change = datetime.now()
        self.adaptive_threshold = config.circuit_breaker_threshold

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name}: Attempting reset to HALF_OPEN")
            else:
                msg = f"Circuit breaker {self.name} is OPEN"
                raise CircuitOpenException(msg)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(execution_time, e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        time_since_open = datetime.now() - self.last_state_change
        return time_since_open.total_seconds() >= self.config.circuit_breaker_timeout

    def _record_success(self, execution_time: float) -> None:
        """Record successful execution."""
        self.metrics.success_count += 1
        self.metrics.last_success = datetime.now()
        self.metrics.response_times.append(execution_time)

        # Keep only last 100 response times
        if len(self.metrics.response_times) > 100:
            self.metrics.response_times.pop(0)

        self._update_metrics()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.last_state_change = datetime.now()
            logger.info(f"Circuit breaker {self.name}: Reset to CLOSED")

    def _record_failure(self, execution_time: float, exception: Exception) -> None:
        """Record failed execution."""
        self.metrics.failure_count += 1
        self.metrics.last_failure = datetime.now()
        self.metrics.response_times.append(execution_time)

        if len(self.metrics.response_times) > 100:
            self.metrics.response_times.pop(0)

        self._update_metrics()

        # Adaptive threshold adjustment
        if self.metrics.failure_rate > 0.8:  # Very high failure rate
            self.adaptive_threshold = max(1, self.adaptive_threshold - 1)

        if self._should_open_circuit():
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()
            logger.warning(f"Circuit breaker {self.name}: Opened due to failures")

    def _update_metrics(self) -> None:
        """Update failure rate and response time metrics."""
        total_calls = self.metrics.failure_count + self.metrics.success_count
        if total_calls > 0:
            self.metrics.failure_rate = self.metrics.failure_count / total_calls

        if self.metrics.response_times:
            self.metrics.avg_response_time = statistics.mean(self.metrics.response_times)

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open."""
        return (self.metrics.failure_count >= self.adaptive_threshold and
                self.metrics.failure_rate >= 0.5)


class AdaptiveRetry:
    """Adaptive retry mechanism with exponential backoff and jitter."""

    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.failure_history: dict[str, list[datetime]] = {}

    def execute_with_retry(self, func: Callable, operation_id: str, *args, **kwargs) -> Any:
        """Execute function with adaptive retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)

                if not self._should_retry(failure_type, attempt):
                    logger.exception(f"Not retrying {operation_id}, failure type: {failure_type.value}")
                    break

                if attempt < self.config.max_retries:
                    delay = self._calculate_retry_delay(attempt, operation_id, failure_type)
                    logger.info(f"Retrying {operation_id} in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)

        raise last_exception

    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure."""
        exception_type = type(exception).__name__.lower()
        exception_msg = str(exception).lower()

        if "timeout" in exception_type or "timeout" in exception_msg:
            return FailureType.TIMEOUT
        if "connection" in exception_msg or "network" in exception_msg:
            return FailureType.CONNECTION_ERROR
        if "auth" in exception_msg or "permission" in exception_msg:
            return FailureType.AUTHENTICATION_ERROR
        if "rate" in exception_msg or "limit" in exception_msg:
            return FailureType.RATE_LIMIT
        if "memory" in exception_msg or "resource" in exception_msg:
            return FailureType.RESOURCE_EXHAUSTION
        return FailureType.UNKNOWN

    def _should_retry(self, failure_type: FailureType, attempt: int) -> bool:
        """Determine if failure should be retried."""
        # Don't retry authentication errors
        if failure_type == FailureType.AUTHENTICATION_ERROR:
            return False

        # Limit retries for resource exhaustion
        if failure_type == FailureType.RESOURCE_EXHAUSTION and attempt >= 1:
            return False

        return attempt < self.config.max_retries

    def _calculate_retry_delay(self, attempt: int, operation_id: str, failure_type: FailureType) -> float:
        """Calculate adaptive retry delay with exponential backoff and jitter."""
        base_delay = self.config.base_retry_delay

        # Exponential backoff
        delay = base_delay * (2 ** attempt)

        # Failure type adjustments
        if failure_type == FailureType.RATE_LIMIT:
            delay *= 2  # Longer delays for rate limits
        elif failure_type == FailureType.TIMEOUT:
            delay *= 1.5  # Moderate increase for timeouts

        # Adaptive adjustment based on history
        if operation_id in self.failure_history:
            recent_failures = len([f for f in self.failure_history[operation_id]
                                 if f > datetime.now() - timedelta(minutes=5)])
            if recent_failures > 3:
                delay *= 1.5  # Increase delay for frequently failing operations

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        delay += jitter

        # Cap at maximum delay
        delay = min(delay, self.config.max_retry_delay)

        # Record failure for adaptive learning
        if operation_id not in self.failure_history:
            self.failure_history[operation_id] = []
        self.failure_history[operation_id].append(datetime.now())

        # Keep only recent failures
        cutoff = datetime.now() - timedelta(hours=1)
        self.failure_history[operation_id] = [f for f in self.failure_history[operation_id] if f > cutoff]

        return delay


class BulkheadIsolation:
    """Bulkhead pattern for resource isolation."""

    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.resource_pools: dict[str, ThreadPoolExecutor] = {}
        self.pool_metrics: dict[str, dict] = {}

    def execute_isolated(self, func: Callable, pool_name: str = "default", *args, **kwargs) -> Any:
        """Execute function in isolated resource pool."""
        if pool_name not in self.resource_pools:
            self.resource_pools[pool_name] = ThreadPoolExecutor(
                max_workers=self.config.bulkhead_max_concurrent,
                thread_name_prefix=f"bulkhead-{pool_name}",
            )
            self.pool_metrics[pool_name] = {
                "active_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
                "queue_size": 0,
            }

        pool = self.resource_pools[pool_name]
        metrics = self.pool_metrics[pool_name]

        # Check resource availability
        if metrics["active_tasks"] >= self.config.bulkhead_max_concurrent:
            msg = f"Pool {pool_name} at capacity"
            raise ResourceExhaustedException(msg)

        try:
            metrics["active_tasks"] += 1
            future = pool.submit(func, *args, **kwargs)
            result = future.result(timeout=self.config.timeout_seconds)
            metrics["completed_tasks"] += 1
            return result
        except Exception:
            metrics["failed_tasks"] += 1
            raise
        finally:
            metrics["active_tasks"] -= 1


class AdaptiveTimeout:
    """Adaptive timeout mechanism based on historical performance."""

    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.operation_history: dict[str, list[float]] = {}
        self.adaptive_timeouts: dict[str, float] = {}

    def get_adaptive_timeout(self, operation_id: str) -> float:
        """Get adaptive timeout for operation."""
        if not self.config.adaptive_timeout_enabled:
            return self.config.timeout_seconds

        if operation_id not in self.operation_history:
            return self.config.timeout_seconds

        history = self.operation_history[operation_id]
        if len(history) < 5:  # Need minimum history
            return self.config.timeout_seconds

        # Calculate adaptive timeout based on percentiles
        p95 = sorted(history)[int(len(history) * 0.95)]
        adaptive_timeout = p95 * 1.5  # 50% buffer above 95th percentile

        # Bounds checking
        min_timeout = self.config.timeout_seconds * 0.5
        max_timeout = self.config.timeout_seconds * 3.0
        adaptive_timeout = max(min_timeout, min(adaptive_timeout, max_timeout))

        self.adaptive_timeouts[operation_id] = adaptive_timeout
        return adaptive_timeout

    def record_execution_time(self, operation_id: str, execution_time: float) -> None:
        """Record execution time for adaptive learning."""
        if operation_id not in self.operation_history:
            self.operation_history[operation_id] = []

        self.operation_history[operation_id].append(execution_time)

        # Keep only recent history
        if len(self.operation_history[operation_id]) > 100:
            self.operation_history[operation_id].pop(0)


class ResilienceOrchestrator:
    """Orchestrates all resilience patterns."""

    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_handler = AdaptiveRetry(self.config)
        self.bulkhead = BulkheadIsolation(self.config)
        self.timeout_handler = AdaptiveTimeout(self.config)

    def execute_resilient(self,
                         func: Callable,
                         operation_id: str,
                         circuit_name: str = "default",
                         pool_name: str = "default",
                         *args, **kwargs) -> Any:
        """Execute function with full resilience protection."""

        # Get or create circuit breaker
        if circuit_name not in self.circuit_breakers:
            self.circuit_breakers[circuit_name] = CircuitBreaker(self.config, circuit_name)

        circuit = self.circuit_breakers[circuit_name]
        adaptive_timeout = self.timeout_handler.get_adaptive_timeout(operation_id)

        def protected_execution():
            start_time = time.time()
            try:
                # Execute with bulkhead isolation and timeout
                result = self.bulkhead.execute_isolated(
                    self._timeout_wrapper(func, adaptive_timeout),
                    pool_name, *args, **kwargs,
                )
                execution_time = time.time() - start_time
                self.timeout_handler.record_execution_time(operation_id, execution_time)
                return result
            except Exception:
                execution_time = time.time() - start_time
                self.timeout_handler.record_execution_time(operation_id, execution_time)
                raise

        # Execute with circuit breaker and retry
        return self.retry_handler.execute_with_retry(
            circuit.call, operation_id, protected_execution,
        )

    def _timeout_wrapper(self, func: Callable, timeout: float) -> Callable:
        """Wrap function with timeout."""
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError:
                    msg = f"Operation timed out after {timeout}s"
                    raise TimeoutException(msg)
        return wrapper

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status of all resilience components."""
        status = {
            "timestamp": datetime.now(),
            "circuit_breakers": {},
            "bulkhead_pools": {},
            "adaptive_timeouts": {},
            "overall_health": "healthy",
        }

        # Circuit breaker status
        for name, cb in self.circuit_breakers.items():
            status["circuit_breakers"][name] = {
                "state": cb.state.value,
                "failure_rate": cb.metrics.failure_rate,
                "avg_response_time": cb.metrics.avg_response_time,
                "success_count": cb.metrics.success_count,
                "failure_count": cb.metrics.failure_count,
            }

            if cb.state == CircuitState.OPEN:
                status["overall_health"] = "degraded"

        # Bulkhead status
        for pool_name, metrics in self.bulkhead.pool_metrics.items():
            status["bulkhead_pools"][pool_name] = metrics.copy()

            if metrics["active_tasks"] >= self.config.bulkhead_max_concurrent * 0.8:
                status["overall_health"] = "degraded"

        # Adaptive timeout status
        status["adaptive_timeouts"] = self.timeout_handler.adaptive_timeouts.copy()

        return status


# Custom exceptions
class CircuitOpenException(Exception):
    """Raised when circuit breaker is open."""


class ResourceExhaustedException(Exception):
    """Raised when resources are exhausted."""


class TimeoutException(Exception):
    """Raised when operation times out."""


# Global resilience orchestrator instance
_global_resilience = None


def get_global_resilience(config: Optional[ResilienceConfig] = None) -> ResilienceOrchestrator:
    """Get global resilience orchestrator instance."""
    global _global_resilience
    if _global_resilience is None:
        _global_resilience = ResilienceOrchestrator(config)
    return _global_resilience


def resilient_operation(operation_id: str,
                       circuit_name: str = "default",
                       pool_name: str = "default"):
    """Decorator for making operations resilient."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            resilience = get_global_resilience()
            return resilience.execute_resilient(
                func, operation_id, circuit_name, pool_name, *args, **kwargs,
            )
        return wrapper
    return decorator
