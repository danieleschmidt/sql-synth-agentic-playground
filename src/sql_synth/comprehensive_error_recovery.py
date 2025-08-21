"""Comprehensive Error Recovery and Resilience Framework.

This module implements advanced error handling and recovery mechanisms:
- Intelligent error classification and recovery strategies
- Circuit breakers and bulkhead patterns
- Adaptive retry mechanisms with backoff strategies
- Graceful degradation and fallback systems
- Self-healing capabilities
- Error prediction and prevention
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    BULKHEAD_ISOLATION = "bulkhead_isolation"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorContext:
    """Error context information."""
    error_id: str
    operation: str
    component: str
    error_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    recovery_time: float
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    recovery_timeout: float = 30.0
    max_failures: int = 10


class IntelligentErrorClassifier:
    """AI-powered error classification and analysis system."""

    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.classification_rules = self._load_classification_rules()
        self.error_history = deque(maxlen=10000)
        self.pattern_cache = {}

    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """Classify error and determine appropriate handling strategy.

        Args:
            error: The exception that occurred
            context: Optional context information

        Returns:
            ErrorContext with classification information
        """
        try:
            error_message = str(error)
            error_type = type(error).__name__
            
            # Generate unique error ID
            error_id = f"{error_type}_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Classify error category
            category = self._classify_category(error, error_message, context)
            
            # Determine severity
            severity = self._determine_severity(error, error_message, category, context)
            
            # Extract metadata
            metadata = self._extract_metadata(error, context)
            
            # Get operation and component from context
            operation = context.get('operation', 'unknown') if context else 'unknown'
            component = context.get('component', 'unknown') if context else 'unknown'
            
            error_context = ErrorContext(
                error_id=error_id,
                operation=operation,
                component=component,
                error_type=error_type,
                severity=severity,
                category=category,
                metadata=metadata,
                stack_trace=self._extract_stack_trace(error),
                user_context=context.get('user_context') if context else None,
            )
            
            # Store in history for learning
            self.error_history.append(error_context)
            
            return error_context
            
        except Exception as e:
            logger.exception(f"Error classification failed: {e}")
            # Fallback classification
            return ErrorContext(
                error_id=f"classify_error_{int(time.time())}",
                operation='unknown',
                component='error_classifier',
                error_type=type(error).__name__,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.BUSINESS_LOGIC,
                metadata={'classification_error': str(e)},
            )

    def _classify_category(self, error: Exception, error_message: str, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error into appropriate category."""
        try:
            error_message_lower = error_message.lower()
            error_type = type(error).__name__
            
            # Network-related errors
            network_indicators = ['connection', 'timeout', 'network', 'socket', 'dns', 'unreachable']
            if any(indicator in error_message_lower for indicator in network_indicators):
                return ErrorCategory.NETWORK
            
            # Database errors
            database_indicators = ['database', 'sql', 'table', 'column', 'constraint', 'duplicate', 'foreign key']
            if any(indicator in error_message_lower for indicator in database_indicators):
                return ErrorCategory.DATABASE
            
            # Authentication errors
            auth_indicators = ['authentication', 'login', 'password', 'token', 'unauthorized', 'invalid credentials']
            if any(indicator in error_message_lower for indicator in auth_indicators):
                return ErrorCategory.AUTHENTICATION
            
            # Authorization errors
            authz_indicators = ['permission', 'access denied', 'forbidden', 'authorization', 'privilege']
            if any(indicator in error_message_lower for indicator in authz_indicators):
                return ErrorCategory.AUTHORIZATION
            
            # Validation errors
            validation_indicators = ['validation', 'invalid', 'format', 'required', 'missing', 'malformed']
            if any(indicator in error_message_lower for indicator in validation_indicators):
                return ErrorCategory.VALIDATION
            
            # Timeout errors
            if 'timeout' in error_message_lower or error_type in ['TimeoutError', 'ConnectionTimeoutError']:
                return ErrorCategory.TIMEOUT
            
            # Resource errors
            resource_indicators = ['memory', 'disk', 'cpu', 'resource', 'limit', 'quota', 'capacity']
            if any(indicator in error_message_lower for indicator in resource_indicators):
                return ErrorCategory.RESOURCE
            
            # External service errors
            external_indicators = ['service unavailable', 'external', 'api', 'third party', 'downstream']
            if any(indicator in error_message_lower for indicator in external_indicators):
                return ErrorCategory.EXTERNAL_SERVICE
            
            # Configuration errors
            config_indicators = ['configuration', 'config', 'setting', 'parameter', 'property']
            if any(indicator in error_message_lower for indicator in config_indicators):
                return ErrorCategory.CONFIGURATION
            
            # Default to business logic
            return ErrorCategory.BUSINESS_LOGIC
            
        except Exception as e:
            logger.warning(f"Error category classification failed: {e}")
            return ErrorCategory.BUSINESS_LOGIC

    def _determine_severity(self, error: Exception, error_message: str, 
                           category: ErrorCategory, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity."""
        try:
            error_type = type(error).__name__
            error_message_lower = error_message.lower()
            
            # Critical severity indicators
            critical_indicators = [
                'critical', 'fatal', 'emergency', 'corruption', 'security breach',
                'data loss', 'system failure', 'catastrophic'
            ]
            if any(indicator in error_message_lower for indicator in critical_indicators):
                return ErrorSeverity.CRITICAL
            
            # High severity by category
            if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION]:
                return ErrorSeverity.HIGH
            
            # High severity indicators
            high_indicators = [
                'exception', 'failed', 'error', 'cannot', 'unable', 'denied',
                'refused', 'rejected', 'blocked'
            ]
            high_error_types = ['RuntimeError', 'SystemError', 'OSError', 'IOError']
            
            if (any(indicator in error_message_lower for indicator in high_indicators) or 
                error_type in high_error_types):
                return ErrorSeverity.HIGH
            
            # Medium severity by category
            if category in [ErrorCategory.DATABASE, ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_SERVICE]:
                return ErrorSeverity.MEDIUM
            
            # Medium severity indicators
            medium_indicators = ['warning', 'invalid', 'missing', 'timeout', 'retry']
            if any(indicator in error_message_lower for indicator in medium_indicators):
                return ErrorSeverity.MEDIUM
            
            # Default to low severity
            return ErrorSeverity.LOW
            
        except Exception as e:
            logger.warning(f"Severity determination failed: {e}")
            return ErrorSeverity.MEDIUM

    def _extract_metadata(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from error and context."""
        metadata = {}
        
        try:
            # Error-specific metadata
            metadata['error_message'] = str(error)
            metadata['error_type'] = type(error).__name__
            metadata['error_module'] = getattr(error, '__module__', 'unknown')
            
            # Context metadata
            if context:
                metadata['operation'] = context.get('operation')
                metadata['component'] = context.get('component')
                metadata['user_id'] = context.get('user_id')
                metadata['request_id'] = context.get('request_id')
                metadata['timestamp'] = time.time()
            
            # Additional error attributes
            if hasattr(error, 'errno'):
                metadata['errno'] = error.errno
            if hasattr(error, 'code'):
                metadata['error_code'] = error.code
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {'extraction_error': str(e)}

    def _extract_stack_trace(self, error: Exception) -> Optional[str]:
        """Extract stack trace from error."""
        try:
            import traceback
            return ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        except Exception as e:
            logger.warning(f"Stack trace extraction failed: {e}")
            return None

    def _load_error_patterns(self) -> Dict[str, Any]:
        """Load known error patterns."""
        return {
            'database_connection': {
                'patterns': ['connection refused', 'connection timeout', 'no connection'],
                'category': ErrorCategory.DATABASE,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.EXPONENTIAL_BACKOFF,
            },
            'authentication_failure': {
                'patterns': ['invalid credentials', 'authentication failed', 'login denied'],
                'category': ErrorCategory.AUTHENTICATION,
                'severity': ErrorSeverity.HIGH,
                'recovery_strategy': RecoveryStrategy.FAIL_FAST,
            },
            'validation_error': {
                'patterns': ['validation failed', 'invalid format', 'required field'],
                'category': ErrorCategory.VALIDATION,
                'severity': ErrorSeverity.MEDIUM,
                'recovery_strategy': RecoveryStrategy.FAIL_FAST,
            },
        }

    def _load_classification_rules(self) -> List[Dict[str, Any]]:
        """Load error classification rules."""
        return [
            {
                'condition': lambda error, message: 'timeout' in message.lower(),
                'category': ErrorCategory.TIMEOUT,
                'severity': ErrorSeverity.MEDIUM,
                'strategy': RecoveryStrategy.EXPONENTIAL_BACKOFF,
            },
            {
                'condition': lambda error, message: isinstance(error, PermissionError),
                'category': ErrorCategory.AUTHORIZATION,
                'severity': ErrorSeverity.HIGH,
                'strategy': RecoveryStrategy.FAIL_FAST,
            },
        ]

    def predict_error_likelihood(self, operation: str, context: Dict[str, Any]) -> float:
        """Predict likelihood of error occurrence based on historical data.

        Args:
            operation: Operation being performed
            context: Context information

        Returns:
            Probability of error occurrence (0.0 to 1.0)
        """
        try:
            # Analyze historical errors for similar operations
            similar_errors = [
                error for error in self.error_history
                if error.operation == operation
            ]
            
            if len(similar_errors) < 10:
                return 0.1  # Low confidence with little data
            
            # Calculate error rate for this operation
            total_operations = context.get('total_operations', len(similar_errors) * 10)  # Estimate
            error_rate = len(similar_errors) / total_operations
            
            # Adjust based on context factors
            risk_factors = []
            
            # Time-based risk
            current_hour = int((time.time() % 86400) // 3600)
            if current_hour < 6 or current_hour > 22:  # Night hours typically higher risk
                risk_factors.append(0.1)
            
            # Load-based risk
            system_load = context.get('system_load', 0.5)
            if system_load > 0.8:
                risk_factors.append(0.2)
            
            # Recent error frequency
            recent_errors = [
                error for error in similar_errors
                if time.time() - error.timestamp < 3600  # Last hour
            ]
            if len(recent_errors) > 3:
                risk_factors.append(0.15)
            
            # Calculate final probability
            base_probability = min(error_rate, 0.8)  # Cap base rate
            risk_adjustment = sum(risk_factors)
            
            final_probability = min(base_probability + risk_adjustment, 0.95)
            return final_probability
            
        except Exception as e:
            logger.warning(f"Error prediction failed: {e}")
            return 0.3  # Default moderate risk


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self._lock = threading.RLock()

    @contextmanager
    def call(self):
        """Context manager for circuit breaker calls."""
        if not self._can_execute():
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            yield
            self._on_success()
        except Exception as e:
            self._on_failure(e)
            raise

    def _can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            
            return False

    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            self.last_success_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery

    def _on_failure(self, error: Exception):
        """Handle failed execution."""
        with self._lock:
            self.last_failure_time = time.time()
            self.failure_count += 1
            
            if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker opened due to {self.failure_count} failures")

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'time_until_retry': max(0, self.config.recovery_timeout - (time.time() - self.last_failure_time)),
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class AdaptiveRetryManager:
    """Adaptive retry mechanism with intelligent backoff strategies."""

    def __init__(self):
        self.retry_strategies = {
            RecoveryStrategy.IMMEDIATE_RETRY: self._immediate_retry,
            RecoveryStrategy.EXPONENTIAL_BACKOFF: self._exponential_backoff,
            RecoveryStrategy.LINEAR_BACKOFF: self._linear_backoff,
        }
        self.retry_history = defaultdict(list)

    async def retry_operation(self, operation: Callable, error_context: ErrorContext,
                            max_retries: int = 3, strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF) -> Any:
        """Retry operation with adaptive strategy.

        Args:
            operation: Operation to retry
            error_context: Error context from initial failure
            max_retries: Maximum number of retry attempts
            strategy: Retry strategy to use

        Returns:
            Result of successful operation

        Raises:
            Exception: If all retries fail
        """
        retry_func = self.retry_strategies.get(strategy, self._exponential_backoff)
        
        last_error = None
        for attempt in range(max_retries):
            if attempt > 0:  # Don't wait before first retry
                wait_time = retry_func(attempt, error_context)
                logger.info(f"Retrying operation after {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            
            try:
                result = await self._execute_operation(operation)
                
                # Record successful retry
                self.retry_history[error_context.operation].append({
                    'timestamp': time.time(),
                    'attempt': attempt + 1,
                    'success': True,
                    'strategy': strategy.value,
                })
                
                return result
                
            except Exception as e:
                last_error = e
                error_context.retry_count = attempt + 1
                
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                
                # Record failed retry
                self.retry_history[error_context.operation].append({
                    'timestamp': time.time(),
                    'attempt': attempt + 1,
                    'success': False,
                    'error': str(e),
                    'strategy': strategy.value,
                })
        
        # All retries failed
        logger.error(f"All {max_retries} retry attempts failed for operation {error_context.operation}")
        raise last_error

    async def _execute_operation(self, operation: Callable) -> Any:
        """Execute operation with proper async handling."""
        if asyncio.iscoroutinefunction(operation):
            return await operation()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, operation)

    def _immediate_retry(self, attempt: int, error_context: ErrorContext) -> float:
        """Immediate retry with no delay."""
        return 0.0

    def _exponential_backoff(self, attempt: int, error_context: ErrorContext) -> float:
        """Exponential backoff with jitter."""
        base_delay = 1.0
        max_delay = 30.0
        
        # Exponential calculation
        delay = base_delay * (2 ** (attempt - 1))
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        final_delay = delay + jitter
        
        # Adjust based on error category
        if error_context.category == ErrorCategory.NETWORK:
            final_delay *= 1.5  # Longer delays for network issues
        elif error_context.category == ErrorCategory.RESOURCE:
            final_delay *= 2.0  # Much longer for resource exhaustion
        
        return final_delay

    def _linear_backoff(self, attempt: int, error_context: ErrorContext) -> float:
        """Linear backoff strategy."""
        base_delay = 2.0
        delay = base_delay * attempt
        
        # Add jitter
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter

    def get_retry_statistics(self, operation: str = None) -> Dict[str, Any]:
        """Get retry statistics for analysis.

        Args:
            operation: Specific operation to analyze (None for all)

        Returns:
            Retry statistics
        """
        try:
            if operation:
                history = self.retry_history.get(operation, [])
                operations = {operation: history}
            else:
                operations = dict(self.retry_history)
            
            total_retries = sum(len(hist) for hist in operations.values())
            successful_retries = sum(
                sum(1 for entry in hist if entry.get('success', False))
                for hist in operations.values()
            )
            
            # Calculate success rate by strategy
            strategy_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
            for hist in operations.values():
                for entry in hist:
                    strategy = entry.get('strategy', 'unknown')
                    strategy_stats[strategy]['total'] += 1
                    if entry.get('success', False):
                        strategy_stats[strategy]['successful'] += 1
            
            # Calculate average attempts per operation
            avg_attempts = {}
            for op, hist in operations.items():
                if hist:
                    attempts = [entry.get('attempt', 1) for entry in hist]
                    avg_attempts[op] = sum(attempts) / len(attempts)
            
            return {
                'total_retries': total_retries,
                'successful_retries': successful_retries,
                'overall_success_rate': successful_retries / max(total_retries, 1),
                'strategy_performance': {
                    strategy: {
                        'success_rate': stats['successful'] / max(stats['total'], 1),
                        'total_attempts': stats['total'],
                    }
                    for strategy, stats in strategy_stats.items()
                },
                'average_attempts_per_operation': avg_attempts,
                'operations_tracked': len(operations),
            }
            
        except Exception as e:
            logger.exception(f"Retry statistics calculation failed: {e}")
            return {'error': str(e)}


class GracefulDegradationManager:
    """Manager for graceful degradation and fallback mechanisms."""

    def __init__(self):
        self.fallback_strategies = {}
        self.degradation_modes = {}
        self.service_health = defaultdict(lambda: {'status': 'healthy', 'last_check': time.time()})

    def register_fallback(self, service: str, fallback_func: Callable, priority: int = 1):
        """Register fallback function for a service.

        Args:
            service: Service name
            fallback_func: Fallback function to call
            priority: Priority level (lower = higher priority)
        """
        if service not in self.fallback_strategies:
            self.fallback_strategies[service] = []
        
        self.fallback_strategies[service].append({
            'function': fallback_func,
            'priority': priority,
            'registered_at': time.time(),
        })
        
        # Sort by priority
        self.fallback_strategies[service].sort(key=lambda x: x['priority'])

    async def execute_with_fallback(self, service: str, primary_func: Callable,
                                   *args, **kwargs) -> Tuple[Any, bool]:
        """Execute function with fallback support.

        Args:
            service: Service name
            primary_func: Primary function to execute
            *args: Arguments for functions
            **kwargs: Keyword arguments for functions

        Returns:
            Tuple of (result, fallback_used)
        """
        try:
            # Try primary function first
            if asyncio.iscoroutinefunction(primary_func):
                result = await primary_func(*args, **kwargs)
            else:
                result = primary_func(*args, **kwargs)
            
            # Update service health
            self.service_health[service] = {
                'status': 'healthy',
                'last_check': time.time(),
                'consecutive_failures': 0,
            }
            
            return result, False
            
        except Exception as primary_error:
            logger.warning(f"Primary function failed for service {service}: {primary_error}")
            
            # Update service health
            health = self.service_health[service]
            health['status'] = 'degraded'
            health['last_check'] = time.time()
            health['consecutive_failures'] = health.get('consecutive_failures', 0) + 1
            health['last_error'] = str(primary_error)
            
            # Try fallback strategies in priority order
            fallbacks = self.fallback_strategies.get(service, [])
            
            for fallback_info in fallbacks:
                try:
                    fallback_func = fallback_info['function']
                    logger.info(f"Trying fallback for service {service}")
                    
                    if asyncio.iscoroutinefunction(fallback_func):
                        result = await fallback_func(*args, **kwargs)
                    else:
                        result = fallback_func(*args, **kwargs)
                    
                    logger.info(f"Fallback successful for service {service}")
                    return result, True
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback failed for service {service}: {fallback_error}")
                    continue
            
            # All fallbacks failed, mark service as unhealthy
            self.service_health[service]['status'] = 'unhealthy'
            raise primary_error

    def activate_degradation_mode(self, mode: str, config: Dict[str, Any]):
        """Activate degradation mode with specific configuration.

        Args:
            mode: Degradation mode name
            config: Configuration for the mode
        """
        self.degradation_modes[mode] = {
            'config': config,
            'activated_at': time.time(),
            'status': 'active',
        }
        
        logger.info(f"Activated degradation mode: {mode}")

    def deactivate_degradation_mode(self, mode: str):
        """Deactivate degradation mode.

        Args:
            mode: Degradation mode name
        """
        if mode in self.degradation_modes:
            self.degradation_modes[mode]['status'] = 'inactive'
            self.degradation_modes[mode]['deactivated_at'] = time.time()
            
            logger.info(f"Deactivated degradation mode: {mode}")

    def is_degradation_active(self, mode: str) -> bool:
        """Check if degradation mode is active.

        Args:
            mode: Degradation mode name

        Returns:
            True if mode is active
        """
        mode_info = self.degradation_modes.get(mode)
        return mode_info is not None and mode_info.get('status') == 'active'

    def get_service_health_status(self) -> Dict[str, Any]:
        """Get health status of all tracked services.

        Returns:
            Service health information
        """
        current_time = time.time()
        
        health_summary = {
            'services': dict(self.service_health),
            'overall_health': 'healthy',
            'degraded_services': [],
            'unhealthy_services': [],
            'active_degradation_modes': [],
            'timestamp': current_time,
        }
        
        # Analyze service health
        for service, health in self.service_health.items():
            if health['status'] == 'degraded':
                health_summary['degraded_services'].append(service)
            elif health['status'] == 'unhealthy':
                health_summary['unhealthy_services'].append(service)
        
        # Determine overall health
        if health_summary['unhealthy_services']:
            health_summary['overall_health'] = 'critical'
        elif health_summary['degraded_services']:
            health_summary['overall_health'] = 'degraded'
        
        # Add active degradation modes
        for mode, config in self.degradation_modes.items():
            if config.get('status') == 'active':
                health_summary['active_degradation_modes'].append({
                    'mode': mode,
                    'config': config['config'],
                    'activated_at': config['activated_at'],
                    'duration': current_time - config['activated_at'],
                })
        
        return health_summary


class ComprehensiveErrorRecoverySystem:
    """Main error recovery and resilience system."""

    def __init__(self):
        self.error_classifier = IntelligentErrorClassifier()
        self.retry_manager = AdaptiveRetryManager()
        self.degradation_manager = GracefulDegradationManager()
        self.circuit_breakers = {}
        self.bulkheads = {}
        self.recovery_history = deque(maxlen=10000)

    async def execute_resilient_operation(self, operation: Callable, operation_name: str,
                                        context: Dict[str, Any] = None,
                                        resilience_config: Dict[str, Any] = None) -> Any:
        """Execute operation with comprehensive resilience mechanisms.

        Args:
            operation: Operation to execute
            operation_name: Name of the operation
            context: Operation context
            resilience_config: Resilience configuration

        Returns:
            Operation result
        """
        start_time = time.time()
        context = context or {}
        config = resilience_config or {}
        
        recovery_result = RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.FAIL_FAST,
            attempts_made=1,
            recovery_time=0.0,
        )
        
        try:
            # Get or create circuit breaker
            circuit_breaker = self._get_circuit_breaker(operation_name, config)
            
            # Check if operation should be executed (circuit breaker)
            with circuit_breaker.call():
                # Execute with bulkhead isolation if configured
                if config.get('use_bulkhead', False):
                    result = await self._execute_with_bulkhead(operation, operation_name, context, config)
                else:
                    result = await self._execute_operation(operation)
            
            # Record successful execution
            recovery_result.success = True
            recovery_result.recovery_time = time.time() - start_time
            
            return result
            
        except CircuitBreakerOpenError as e:
            logger.warning(f"Circuit breaker open for {operation_name}")
            
            # Try fallback if available
            if config.get('enable_fallback', True):
                try:
                    result, fallback_used = await self.degradation_manager.execute_with_fallback(
                        operation_name, operation, **context
                    )
                    
                    recovery_result.success = True
                    recovery_result.fallback_used = fallback_used
                    recovery_result.strategy_used = RecoveryStrategy.FALLBACK
                    recovery_result.recovery_time = time.time() - start_time
                    
                    return result
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {operation_name}: {fallback_error}")
                    raise e
            else:
                raise e
                
        except Exception as error:
            # Classify error
            error_context = self.error_classifier.classify_error(error, {
                'operation': operation_name,
                'component': context.get('component', 'unknown'),
                **context
            })
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(error_context, config)
            
            # Apply recovery strategy
            try:
                if recovery_strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                    result = await self._apply_retry_strategy(
                        operation, error_context, RecoveryStrategy.IMMEDIATE_RETRY, config
                    )
                elif recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                    result = await self._apply_retry_strategy(
                        operation, error_context, RecoveryStrategy.EXPONENTIAL_BACKOFF, config
                    )
                elif recovery_strategy == RecoveryStrategy.LINEAR_BACKOFF:
                    result = await self._apply_retry_strategy(
                        operation, error_context, RecoveryStrategy.LINEAR_BACKOFF, config
                    )
                elif recovery_strategy == RecoveryStrategy.FALLBACK:
                    result, fallback_used = await self.degradation_manager.execute_with_fallback(
                        operation_name, operation, **context
                    )
                    recovery_result.fallback_used = fallback_used
                elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    result = await self._apply_graceful_degradation(
                        operation, operation_name, error_context, config
                    )
                else:  # FAIL_FAST
                    raise error
                
                recovery_result.success = True
                recovery_result.strategy_used = recovery_strategy
                recovery_result.attempts_made = getattr(error_context, 'retry_count', 1) + 1
                recovery_result.recovery_time = time.time() - start_time
                
                return result
                
            except Exception as recovery_error:
                # Recovery failed
                recovery_result.success = False
                recovery_result.strategy_used = recovery_strategy
                recovery_result.attempts_made = getattr(error_context, 'retry_count', 1) + 1
                recovery_result.recovery_time = time.time() - start_time
                recovery_result.metadata = {'original_error': str(error), 'recovery_error': str(recovery_error)}
                
                logger.error(f"Recovery failed for {operation_name}: {recovery_error}")
                raise recovery_error
                
        finally:
            # Record recovery attempt
            self.recovery_history.append({
                'operation_name': operation_name,
                'timestamp': time.time(),
                'recovery_result': recovery_result,
                'context': context,
            })

    async def _execute_operation(self, operation: Callable) -> Any:
        """Execute operation with proper async handling."""
        if asyncio.iscoroutinefunction(operation):
            return await operation()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, operation)

    async def _execute_with_bulkhead(self, operation: Callable, operation_name: str,
                                   context: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """Execute operation with bulkhead isolation."""
        bulkhead_name = config.get('bulkhead_name', operation_name)
        max_workers = config.get('bulkhead_max_workers', 5)
        
        if bulkhead_name not in self.bulkheads:
            self.bulkheads[bulkhead_name] = ThreadPoolExecutor(max_workers=max_workers)
        
        executor = self.bulkheads[bulkhead_name]
        
        try:
            # Execute with timeout in bulkhead
            timeout = config.get('bulkhead_timeout', 30)
            loop = asyncio.get_event_loop()
            
            if asyncio.iscoroutinefunction(operation):
                # For async operations, we need to handle them differently
                result = await asyncio.wait_for(operation(), timeout=timeout)
            else:
                # For sync operations, use the executor
                future = executor.submit(operation)
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future), timeout=timeout
                )
            
            return result
            
        except asyncio.TimeoutError as e:
            logger.warning(f"Bulkhead timeout for operation {operation_name}")
            raise e
        except FutureTimeoutError as e:
            logger.warning(f"Bulkhead execution timeout for operation {operation_name}")
            raise asyncio.TimeoutError("Bulkhead execution timeout") from e

    def _get_circuit_breaker(self, operation_name: str, config: Dict[str, Any]) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation_name not in self.circuit_breakers:
            circuit_config = CircuitBreakerConfig(
                failure_threshold=config.get('circuit_failure_threshold', 5),
                success_threshold=config.get('circuit_success_threshold', 3),
                timeout=config.get('circuit_timeout', 60.0),
                recovery_timeout=config.get('circuit_recovery_timeout', 30.0),
                max_failures=config.get('circuit_max_failures', 10),
            )
            self.circuit_breakers[operation_name] = CircuitBreaker(circuit_config)
        
        return self.circuit_breakers[operation_name]

    def _determine_recovery_strategy(self, error_context: ErrorContext,
                                   config: Dict[str, Any]) -> RecoveryStrategy:
        """Determine appropriate recovery strategy based on error context."""
        # Override strategy if specified in config
        if 'recovery_strategy' in config:
            return RecoveryStrategy(config['recovery_strategy'])
        
        # Strategy based on error category and severity
        category = error_context.category
        severity = error_context.severity
        
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAIL_FAST
        
        if category == ErrorCategory.NETWORK:
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        elif category == ErrorCategory.DATABASE:
            return RecoveryStrategy.LINEAR_BACKOFF
        elif category == ErrorCategory.EXTERNAL_SERVICE:
            return RecoveryStrategy.FALLBACK
        elif category == ErrorCategory.TIMEOUT:
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        elif category == ErrorCategory.RESOURCE:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        elif category in [ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORIZATION]:
            return RecoveryStrategy.FAIL_FAST
        elif category == ErrorCategory.VALIDATION:
            return RecoveryStrategy.FAIL_FAST
        else:
            return RecoveryStrategy.IMMEDIATE_RETRY

    async def _apply_retry_strategy(self, operation: Callable, error_context: ErrorContext,
                                  strategy: RecoveryStrategy, config: Dict[str, Any]) -> Any:
        """Apply retry strategy to operation."""
        max_retries = config.get('max_retries', 3)
        
        return await self.retry_manager.retry_operation(
            operation, error_context, max_retries, strategy
        )

    async def _apply_graceful_degradation(self, operation: Callable, operation_name: str,
                                        error_context: ErrorContext, config: Dict[str, Any]) -> Any:
        """Apply graceful degradation strategy."""
        # Activate degradation mode if not already active
        degradation_mode = config.get('degradation_mode', 'reduced_functionality')
        
        if not self.degradation_manager.is_degradation_active(degradation_mode):
            degradation_config = config.get('degradation_config', {
                'reduced_features': True,
                'simplified_responses': True,
                'increased_caching': True,
            })
            
            self.degradation_manager.activate_degradation_mode(degradation_mode, degradation_config)
        
        # Try to execute with degraded functionality
        # This would typically involve calling a simplified version of the operation
        # For now, we'll try the fallback mechanism
        result, fallback_used = await self.degradation_manager.execute_with_fallback(
            operation_name, operation
        )
        
        return result

    def get_resilience_analytics(self) -> Dict[str, Any]:
        """Get comprehensive resilience analytics."""
        try:
            current_time = time.time()
            
            # Recovery statistics
            total_operations = len(self.recovery_history)
            successful_recoveries = sum(
                1 for entry in self.recovery_history 
                if entry['recovery_result'].success
            )
            
            # Circuit breaker statistics
            circuit_stats = {}
            for name, circuit in self.circuit_breakers.items():
                circuit_stats[name] = circuit.get_state()
            
            # Retry statistics
            retry_stats = self.retry_manager.get_retry_statistics()
            
            # Service health
            service_health = self.degradation_manager.get_service_health_status()
            
            # Error distribution
            error_categories = defaultdict(int)
            error_severities = defaultdict(int)
            recovery_strategies = defaultdict(int)
            
            for entry in self.recovery_history:
                # Extract error info from context if available
                recovery_result = entry.get('recovery_result')
                if recovery_result:
                    recovery_strategies[recovery_result.strategy_used.value] += 1
            
            return {
                'summary': {
                    'total_operations': total_operations,
                    'successful_recoveries': successful_recoveries,
                    'recovery_success_rate': successful_recoveries / max(total_operations, 1),
                    'active_circuit_breakers': len([
                        cb for cb in circuit_stats.values() 
                        if cb['state'] != 'closed'
                    ]),
                    'degraded_services': len(service_health.get('degraded_services', [])),
                },
                'circuit_breakers': circuit_stats,
                'retry_performance': retry_stats,
                'service_health': service_health,
                'recovery_strategies_used': dict(recovery_strategies),
                'bulkheads_active': len(self.bulkheads),
                'system_resilience_score': self._calculate_resilience_score(),
                'timestamp': current_time,
            }
            
        except Exception as e:
            logger.exception(f"Resilience analytics calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_resilience_score(self) -> float:
        """Calculate overall system resilience score (0.0 to 1.0)."""
        try:
            if not self.recovery_history:
                return 0.8  # Default good score for new system
            
            # Recovery success rate (40% weight)
            total_operations = len(self.recovery_history)
            successful_recoveries = sum(
                1 for entry in self.recovery_history 
                if entry['recovery_result'].success
            )
            recovery_rate = successful_recoveries / total_operations
            
            # Circuit breaker health (20% weight)
            circuit_health = 1.0
            if self.circuit_breakers:
                open_circuits = sum(
                    1 for cb in self.circuit_breakers.values()
                    if cb.get_state()['state'] == 'open'
                )
                circuit_health = 1.0 - (open_circuits / len(self.circuit_breakers))
            
            # Service health (25% weight)
            service_health_status = self.degradation_manager.get_service_health_status()
            total_services = len(service_health_status.get('services', {}))
            if total_services > 0:
                unhealthy_services = len(service_health_status.get('unhealthy_services', []))
                service_health_score = 1.0 - (unhealthy_services / total_services)
            else:
                service_health_score = 1.0
            
            # Retry efficiency (15% weight)
            retry_stats = self.retry_manager.get_retry_statistics()
            retry_efficiency = retry_stats.get('overall_success_rate', 0.8)
            
            # Weighted score
            resilience_score = (
                recovery_rate * 0.4 +
                circuit_health * 0.2 +
                service_health_score * 0.25 +
                retry_efficiency * 0.15
            )
            
            return min(max(resilience_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Resilience score calculation failed: {e}")
            return 0.5


# Global error recovery system
global_error_recovery_system = ComprehensiveErrorRecoverySystem()


# Utility functions and decorators
def resilient_operation(operation_name: str, resilience_config: Dict[str, Any] = None):
    """Decorator for making operations resilient.

    Args:
        operation_name: Name of the operation
        resilience_config: Resilience configuration

    Returns:
        Decorated function with resilience capabilities
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            context = {'args': args, 'kwargs': kwargs}
            
            # Create operation callable
            if asyncio.iscoroutinefunction(func):
                async def operation():
                    return await func(*args, **kwargs)
            else:
                def operation():
                    return func(*args, **kwargs)
            
            return await global_error_recovery_system.execute_resilient_operation(
                operation, operation_name, context, resilience_config
            )
        
        return wrapper
    return decorator


async def execute_with_resilience(operation: Callable, operation_name: str,
                                 context: Dict[str, Any] = None,
                                 config: Dict[str, Any] = None) -> Any:
    """Execute operation with comprehensive resilience mechanisms.

    Args:
        operation: Operation to execute
        operation_name: Name of the operation
        context: Optional context
        config: Optional resilience configuration

    Returns:
        Operation result
    """
    return await global_error_recovery_system.execute_resilient_operation(
        operation, operation_name, context, config
    )


def register_fallback_function(service: str, fallback_func: Callable, priority: int = 1):
    """Register fallback function for a service.

    Args:
        service: Service name
        fallback_func: Fallback function
        priority: Priority level (lower = higher priority)
    """
    global_error_recovery_system.degradation_manager.register_fallback(
        service, fallback_func, priority
    )


def get_resilience_insights() -> Dict[str, Any]:
    """Get comprehensive resilience analytics and insights.

    Returns:
        Resilience analytics
    """
    return global_error_recovery_system.get_resilience_analytics()


# Context manager for error handling
@contextmanager
def error_recovery_context(operation_name: str, config: Dict[str, Any] = None):
    """Context manager for error recovery.

    Args:
        operation_name: Name of the operation
        config: Optional resilience configuration
    """
    start_time = time.time()
    error_context = None
    
    try:
        yield
    except Exception as e:
        # Classify and handle error
        context = {
            'operation': operation_name,
            'component': config.get('component', 'unknown') if config else 'unknown',
        }
        
        error_context = global_error_recovery_system.error_classifier.classify_error(e, context)
        
        logger.error(f"Error in {operation_name}: {e} (classified as {error_context.category.value})")
        raise
    finally:
        # Record metrics
        duration = time.time() - start_time
        logger.debug(f"Operation {operation_name} completed in {duration:.3f}s")