"""Advanced performance optimization for SQL Synthesis Agent.

This module provides comprehensive performance monitoring, query optimization,
connection pooling, and adaptive caching strategies.
"""

import asyncio
import logging
import time
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib
import json
import statistics

import psutil
from sqlalchemy.pool import QueuePool
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine


logger = logging.getLogger(__name__)


@dataclass 
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Real-time performance monitoring and metrics collection."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()
        
        # Performance counters
        self.operation_counters = defaultdict(int)
        self.operation_durations = defaultdict(list)
        self.error_counters = defaultdict(int)
        
        # System resource monitoring
        self.process = psutil.Process()
        
    def start_operation(self, operation_name: str, **context) -> PerformanceMetrics:
        """Start monitoring an operation.
        
        Args:
            operation_name: Name of the operation
            **context: Additional context information
            
        Returns:
            PerformanceMetrics object to track the operation
        """
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            memory_usage_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_usage_percent=self.process.cpu_percent(),
            context=context
        )
        
        with self._lock:
            self.operation_counters[operation_name] += 1
            
        return metrics
    
    def end_operation(self, metrics: PerformanceMetrics, error: Optional[Exception] = None) -> None:
        """End monitoring an operation.
        
        Args:
            metrics: PerformanceMetrics object from start_operation
            error: Exception if operation failed
        """
        end_time = time.time()
        duration = end_time - metrics.start_time
        
        # Update metrics
        metrics.end_time = end_time
        metrics.duration = duration
        metrics.success = error is None
        
        if error:
            metrics.error_type = type(error).__name__
            
        # Capture final resource usage
        try:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            metrics.memory_usage_mb = current_memory - (metrics.memory_usage_mb or 0)
        except Exception:
            pass
        
        with self._lock:
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update duration statistics
            self.operation_durations[metrics.operation_name].append(duration)
            if len(self.operation_durations[metrics.operation_name]) > 100:
                self.operation_durations[metrics.operation_name] = \
                    self.operation_durations[metrics.operation_name][-100:]
            
            # Update error counters
            if error:
                self.error_counters[f"{metrics.operation_name}_error"] += 1
                self.error_counters[f"{type(error).__name__}"] += 1
        
        # Log performance metrics
        logger.debug(
            "Operation completed",
            extra={
                "operation": metrics.operation_name,
                "duration": duration,
                "success": metrics.success,
                "memory_usage_mb": metrics.memory_usage_mb,
                "error_type": metrics.error_type
            }
        )
    
    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Args:
            operation_name: Filter by specific operation (optional)
            
        Returns:
            Dictionary containing performance statistics
        """
        with self._lock:
            if operation_name:
                filtered_metrics = [m for m in self.metrics_history 
                                  if m.operation_name == operation_name]
                durations = self.operation_durations.get(operation_name, [])
            else:
                filtered_metrics = list(self.metrics_history)
                durations = []
                for op_durations in self.operation_durations.values():
                    durations.extend(op_durations)
            
            if not filtered_metrics:
                return {"error": "No metrics available"}
            
            # Calculate statistics
            total_operations = len(filtered_metrics)
            successful_operations = sum(1 for m in filtered_metrics if m.success)
            success_rate = successful_operations / total_operations
            
            if durations:
                avg_duration = statistics.mean(durations)
                median_duration = statistics.median(durations)
                p95_duration = statistics.quantiles(durations, n=20)[18] if len(durations) > 20 else max(durations)
                p99_duration = statistics.quantiles(durations, n=100)[98] if len(durations) > 100 else max(durations)
            else:
                avg_duration = median_duration = p95_duration = p99_duration = 0
            
            return {
                "operation_name": operation_name,
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": success_rate,
                "average_duration_seconds": avg_duration,
                "median_duration_seconds": median_duration,
                "p95_duration_seconds": p95_duration,
                "p99_duration_seconds": p99_duration,
                "operation_counters": dict(self.operation_counters),
                "error_counters": dict(self.error_counters)
            }


class AdaptiveCacheStrategy:
    """Adaptive caching strategy based on usage patterns and performance."""
    
    def __init__(
        self,
        base_ttl: int = 3600,
        max_ttl: int = 86400,
        min_ttl: int = 300,
        cache_hit_threshold: float = 0.7
    ):
        """Initialize adaptive cache strategy.
        
        Args:
            base_ttl: Base TTL in seconds
            max_ttl: Maximum TTL in seconds
            min_ttl: Minimum TTL in seconds  
            cache_hit_threshold: Hit rate threshold for TTL adjustment
        """
        self.base_ttl = base_ttl
        self.max_ttl = max_ttl
        self.min_ttl = min_ttl
        self.cache_hit_threshold = cache_hit_threshold
        
        self.cache_stats = defaultdict(lambda: {
            'hits': 0, 
            'misses': 0, 
            'last_access': time.time(),
            'access_frequency': deque(maxlen=100)
        })
        self._lock = threading.RLock()
    
    def calculate_ttl(self, cache_key: str) -> int:
        """Calculate adaptive TTL for cache key.
        
        Args:
            cache_key: Cache key to calculate TTL for
            
        Returns:
            TTL in seconds
        """
        with self._lock:
            stats = self.cache_stats[cache_key]
            
            total_accesses = stats['hits'] + stats['misses']
            if total_accesses == 0:
                return self.base_ttl
            
            hit_rate = stats['hits'] / total_accesses
            
            # Calculate access frequency (accesses per minute)
            now = time.time()
            recent_accesses = [t for t in stats['access_frequency'] 
                             if now - t < 3600]  # Last hour
            access_frequency = len(recent_accesses) / 60  # per minute
            
            # Adaptive TTL calculation
            if hit_rate >= self.cache_hit_threshold:
                # High hit rate: increase TTL
                frequency_multiplier = min(2.0, 1.0 + (access_frequency / 10))
                ttl = min(self.max_ttl, int(self.base_ttl * frequency_multiplier))
            else:
                # Low hit rate: decrease TTL
                ttl = max(self.min_ttl, int(self.base_ttl * 0.5))
            
            return ttl
    
    def record_cache_hit(self, cache_key: str) -> None:
        """Record a cache hit for adaptive learning.
        
        Args:
            cache_key: Cache key that was hit
        """
        with self._lock:
            stats = self.cache_stats[cache_key]
            stats['hits'] += 1
            stats['last_access'] = time.time()
            stats['access_frequency'].append(time.time())
    
    def record_cache_miss(self, cache_key: str) -> None:
        """Record a cache miss for adaptive learning.
        
        Args:
            cache_key: Cache key that was missed
        """
        with self._lock:
            stats = self.cache_stats[cache_key]
            stats['misses'] += 1
            stats['last_access'] = time.time()
            stats['access_frequency'].append(time.time())


class ConnectionPoolOptimizer:
    """Connection pool optimization and monitoring."""
    
    def __init__(self, engine: Engine):
        """Initialize connection pool optimizer.
        
        Args:
            engine: SQLAlchemy engine to optimize
        """
        self.engine = engine
        self.pool_stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'pool_size': 0,
            'checked_out': 0,
            'overflow': 0,
            'invalid_connections': 0
        }
        
        # Set up event listeners
        self._setup_pool_events()
        
    def _setup_pool_events(self) -> None:
        """Setup SQLAlchemy pool event listeners."""
        
        @event.listens_for(self.engine, 'connect')
        def on_connect(dbapi_connection, connection_record):
            self.pool_stats['connections_created'] += 1
            logger.debug("Database connection created", extra={
                'pool_size': self.engine.pool.size(),
                'checked_out': self.engine.pool.checkedout()
            })
        
        @event.listens_for(self.engine, 'close')
        def on_close(dbapi_connection, connection_record):
            self.pool_stats['connections_closed'] += 1
            
        @event.listens_for(self.engine, 'checkout')
        def on_checkout(dbapi_connection, connection_record, connection_proxy):
            self.pool_stats['checked_out'] = self.engine.pool.checkedout()
            
        @event.listens_for(self.engine, 'checkin')
        def on_checkin(dbapi_connection, connection_record):
            self.pool_stats['checked_out'] = self.engine.pool.checkedout()
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status.
        
        Returns:
            Dictionary containing pool status information
        """
        pool = self.engine.pool
        
        return {
            'pool_size': pool.size(),
            'checked_out_connections': pool.checkedout(), 
            'overflow_connections': pool.overflow(),
            'invalid_connections': pool.invalid(),
            'total_created': self.pool_stats['connections_created'],
            'total_closed': self.pool_stats['connections_closed'],
            'pool_class': pool.__class__.__name__
        }
    
    def optimize_pool_settings(self) -> Dict[str, Any]:
        """Analyze pool usage and recommend optimization settings.
        
        Returns:
            Dictionary with optimization recommendations
        """
        status = self.get_pool_status()
        recommendations = []
        
        # Check for connection leaks
        if status['checked_out_connections'] > status['pool_size'] * 0.8:
            recommendations.append({
                'issue': 'high_checkout_ratio',
                'description': 'High percentage of connections checked out',
                'recommendation': 'Check for connection leaks or increase pool size'
            })
        
        # Check for overflow usage
        if status['overflow_connections'] > 0:
            recommendations.append({
                'issue': 'pool_overflow',
                'description': 'Pool is using overflow connections',
                'recommendation': 'Consider increasing base pool size'
            })
        
        # Check for invalid connections
        if status['invalid_connections'] > 0:
            recommendations.append({
                'issue': 'invalid_connections', 
                'description': 'Pool contains invalid connections',
                'recommendation': 'Check database connectivity and connection recycling settings'
            })
        
        return {
            'current_status': status,
            'recommendations': recommendations,
            'timestamp': time.time()
        }


class AsyncQueryExecutor:
    """Asynchronous query execution for improved performance."""
    
    def __init__(self, max_workers: int = 10):
        """Initialize async query executor.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_queries = set()
        self._lock = threading.RLock()
        
    def submit_query(
        self,
        query_func: Callable,
        query_id: Optional[str] = None,
        *args,
        **kwargs
    ) -> Tuple[str, asyncio.Future]:
        """Submit query for asynchronous execution.
        
        Args:
            query_func: Function to execute the query
            query_id: Optional query identifier
            *args: Arguments for query function
            **kwargs: Keyword arguments for query function
            
        Returns:
            Tuple of (query_id, future)
        """
        if query_id is None:
            query_id = f"query_{int(time.time() * 1000)}_{threading.get_ident()}"
        
        with self._lock:
            self.active_queries.add(query_id)
        
        future = self.executor.submit(self._execute_with_tracking, query_id, query_func, *args, **kwargs)
        
        return query_id, future
    
    def _execute_with_tracking(self, query_id: str, query_func: Callable, *args, **kwargs) -> Any:
        """Execute query with tracking and cleanup.
        
        Args:
            query_id: Query identifier
            query_func: Function to execute
            *args: Arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Query result
        """
        try:
            logger.debug("Starting async query execution", extra={'query_id': query_id})
            result = query_func(*args, **kwargs)
            logger.debug("Async query completed successfully", extra={'query_id': query_id})
            return result
        except Exception as e:
            logger.error("Async query failed", extra={'query_id': query_id, 'error': str(e)})
            raise
        finally:
            with self._lock:
                self.active_queries.discard(query_id)
    
    def get_active_queries(self) -> List[str]:
        """Get list of active query IDs.
        
        Returns:
            List of active query identifiers
        """
        with self._lock:
            return list(self.active_queries)
    
    def cancel_query(self, query_id: str) -> bool:
        """Cancel an active query.
        
        Args:
            query_id: Query identifier to cancel
            
        Returns:
            True if query was cancelled, False if not found
        """
        # Note: This is a simplified implementation
        # Real cancellation would require more sophisticated tracking
        with self._lock:
            if query_id in self.active_queries:
                self.active_queries.discard(query_id)
                return True
            return False
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the async executor.
        
        Args:
            wait: Whether to wait for active queries to complete
        """
        self.executor.shutdown(wait=wait)


class QueryOptimizer:
    """SQL query optimization and analysis."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.query_patterns = {}
        self.optimization_cache = {}
        self._lock = threading.RLock()
    
    def analyze_query_pattern(self, sql_query: str) -> Dict[str, Any]:
        """Analyze SQL query pattern for optimization opportunities.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        query_hash = hashlib.md5(sql_query.encode()).hexdigest()
        
        with self._lock:
            if query_hash in self.optimization_cache:
                return self.optimization_cache[query_hash]
        
        analysis = {
            'query_hash': query_hash,
            'query_length': len(sql_query),
            'has_joins': 'JOIN' in sql_query.upper(),
            'has_subqueries': '(' in sql_query and 'SELECT' in sql_query.upper(),
            'has_aggregation': any(agg in sql_query.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
            'has_groupby': 'GROUP BY' in sql_query.upper(),
            'has_orderby': 'ORDER BY' in sql_query.upper(),
            'has_limit': 'LIMIT' in sql_query.upper(),
            'estimated_complexity': self._estimate_complexity(sql_query),
            'optimization_suggestions': self._generate_optimization_suggestions(sql_query)
        }
        
        with self._lock:
            self.optimization_cache[query_hash] = analysis
            
        return analysis
    
    def _estimate_complexity(self, sql_query: str) -> str:
        """Estimate query complexity.
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Complexity level (low, medium, high)
        """
        query_upper = sql_query.upper()
        complexity_score = 0
        
        # Basic complexity factors
        if 'JOIN' in query_upper:
            complexity_score += query_upper.count('JOIN')
        if 'UNION' in query_upper:
            complexity_score += query_upper.count('UNION') * 2
        if query_upper.count('SELECT') > 1:  # Subqueries
            complexity_score += (query_upper.count('SELECT') - 1)
        if 'GROUP BY' in query_upper:
            complexity_score += 1
        if 'HAVING' in query_upper:
            complexity_score += 1
        if 'DISTINCT' in query_upper:
            complexity_score += 1
        
        if complexity_score <= 2:
            return 'low'
        elif complexity_score <= 5:
            return 'medium'
        else:
            return 'high'
    
    def _generate_optimization_suggestions(self, sql_query: str) -> List[str]:
        """Generate optimization suggestions for query.
        
        Args:
            sql_query: SQL query to optimize
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        query_upper = sql_query.upper()
        
        # Check for missing LIMIT
        if 'SELECT' in query_upper and 'LIMIT' not in query_upper:
            if not any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                suggestions.append("Consider adding LIMIT clause to prevent large result sets")
        
        # Check for SELECT *
        if 'SELECT *' in query_upper:
            suggestions.append("Avoid SELECT * - specify only required columns")
        
        # Check for complex WHERE clauses
        if 'WHERE' in query_upper and query_upper.count('AND') > 3:
            suggestions.append("Complex WHERE clause detected - consider indexing strategy")
        
        # Check for functions in WHERE clause
        if any(func in query_upper for func in ['UPPER(', 'LOWER(', 'SUBSTRING(']):
            suggestions.append("Functions in WHERE clause may prevent index usage")
        
        # Check for OR conditions
        if query_upper.count(' OR ') > 2:
            suggestions.append("Multiple OR conditions may benefit from UNION optimization")
        
        return suggestions


# Global instances
performance_monitor = PerformanceMonitor()
adaptive_cache_strategy = AdaptiveCacheStrategy()
async_query_executor = AsyncQueryExecutor()
query_optimizer = QueryOptimizer()


def monitor_performance(operation_name: str):
    """Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            metrics = performance_monitor.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_operation(metrics)
                return result
            except Exception as e:
                performance_monitor.end_operation(metrics, error=e)
                raise
        return wrapper
    return decorator