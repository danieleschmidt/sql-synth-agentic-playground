"""Advanced performance optimization system for SQL synthesis operations.

This module provides intelligent query optimization, connection pooling,
auto-scaling capabilities, and performance monitoring.
"""

import logging
import threading
import time
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_id: str
    operation_type: str
    start_time: float
    end_time: float
    duration: float
    cpu_usage_percent: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResourceUtilization:
    """System resource utilization snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    active_connections: int


class PerformanceProfiler:
    """Advanced performance profiler with resource monitoring."""

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self.resource_history: List[ResourceUtilization] = []
        self._lock = Lock()
        self._last_network_io = None
        self._last_disk_io = None
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start continuous resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")

    def profile_operation(
        self,
        func: Callable,
        operation_type: str,
        *args,
        **kwargs,
    ) -> Tuple[Any, PerformanceMetrics]:
        """Profile a function execution with detailed metrics.
        
        Args:
            func: Function to profile
            operation_type: Type of operation being profiled
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function result, performance metrics)
        """
        operation_id = f"{operation_type}_{int(time.time() * 1000)}"

        # Capture initial resource state
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        success = False
        result = None
        error_message = None

        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Capture final resource state
            final_cpu = process.cpu_percent()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Calculate averages
            avg_cpu = (initial_cpu + final_cpu) / 2
            avg_memory = (initial_memory + final_memory) / 2

            metrics = PerformanceMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                cpu_usage_percent=avg_cpu,
                memory_usage_mb=avg_memory,
                success=success,
                error_message=error_message,
                metadata={
                    "args_count": len(args),
                    "kwargs_count": len(kwargs),
                },
            )

            self._record_metrics(metrics)

        return result, metrics

    def _monitor_resources(self, interval: float) -> None:
        """Monitor system resources continuously."""
        while self._monitoring:
            try:
                # Get current resource utilization
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                # Network I/O
                network_io = psutil.net_io_counters()
                if self._last_network_io:
                    net_sent_mb = (network_io.bytes_sent - self._last_network_io.bytes_sent) / 1024 / 1024
                    net_recv_mb = (network_io.bytes_recv - self._last_network_io.bytes_recv) / 1024 / 1024
                else:
                    net_sent_mb = net_recv_mb = 0
                self._last_network_io = network_io

                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io and self._last_disk_io:
                    disk_read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / 1024 / 1024
                    disk_write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / 1024 / 1024
                else:
                    disk_read_mb = disk_write_mb = 0
                if disk_io:
                    self._last_disk_io = disk_io

                # Active connections (simplified)
                try:
                    active_connections = len(psutil.net_connections())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    active_connections = 0

                utilization = ResourceUtilization(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_available_mb=memory.available / 1024 / 1024,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_io_sent_mb=net_sent_mb,
                    network_io_recv_mb=net_recv_mb,
                    active_connections=active_connections,
                )

                self._record_resource_utilization(utilization)

            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")

            time.sleep(interval)

    def _record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

    def _record_resource_utilization(self, utilization: ResourceUtilization) -> None:
        """Record resource utilization."""
        with self._lock:
            self.resource_history.append(utilization)
            if len(self.resource_history) > self.max_history:
                self.resource_history.pop(0)

    def get_performance_summary(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary statistics.
        
        Args:
            operation_type: Filter by specific operation type
            
        Returns:
            Performance summary dictionary
        """
        with self._lock:
            metrics = self.metrics_history
            if operation_type:
                metrics = [m for m in metrics if m.operation_type == operation_type]

            if not metrics:
                return {"message": "No metrics available"}

            # Calculate statistics
            durations = [m.duration for m in metrics]
            cpu_usages = [m.cpu_usage_percent for m in metrics]
            memory_usages = [m.memory_usage_mb for m in metrics]
            success_count = sum(1 for m in metrics if m.success)

            return {
                "total_operations": len(metrics),
                "success_rate": success_count / len(metrics),
                "duration_stats": {
                    "min": min(durations),
                    "max": max(durations),
                    "avg": sum(durations) / len(durations),
                    "p95": sorted(durations)[int(len(durations) * 0.95)],
                },
                "cpu_usage_stats": {
                    "min": min(cpu_usages),
                    "max": max(cpu_usages),
                    "avg": sum(cpu_usages) / len(cpu_usages),
                },
                "memory_usage_stats": {
                    "min": min(memory_usages),
                    "max": max(memory_usages),
                    "avg": sum(memory_usages) / len(memory_usages),
                },
                "recent_errors": [
                    {"operation_id": m.operation_id, "error": m.error_message}
                    for m in metrics[-10:] if not m.success
                ],
            }

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary."""
        with self._lock:
            if not self.resource_history:
                return {"message": "No resource data available"}

            recent_data = self.resource_history[-60:]  # Last 60 readings

            cpu_values = [r.cpu_percent for r in recent_data]
            memory_values = [r.memory_percent for r in recent_data]

            return {
                "current_cpu_percent": recent_data[-1].cpu_percent,
                "current_memory_percent": recent_data[-1].memory_percent,
                "current_memory_available_mb": recent_data[-1].memory_available_mb,
                "active_connections": recent_data[-1].active_connections,
                "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                "avg_memory_percent": sum(memory_values) / len(memory_values),
                "peak_cpu_percent": max(cpu_values),
                "peak_memory_percent": max(memory_values),
                "total_disk_io_mb": sum(r.disk_io_read_mb + r.disk_io_write_mb for r in recent_data),
                "total_network_io_mb": sum(r.network_io_sent_mb + r.network_io_recv_mb for r in recent_data),
            }


class ConnectionPoolManager:
    """Advanced database connection pool manager with auto-scaling."""

    def __init__(
        self,
        min_connections: int = 2,
        max_connections: int = 10,
        connection_timeout: float = 30.0,
        idle_timeout: float = 300.0,
        scale_threshold: float = 0.8,
    ) -> None:
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        self.scale_threshold = scale_threshold

        self._pool: List[Any] = []
        self._in_use: Set[Any] = set()
        self._lock = RLock()
        self._last_cleanup = time.time()
        self._connection_factory: Optional[Callable] = None
        self._stats = {
            "total_created": 0,
            "total_destroyed": 0,
            "pool_hits": 0,
            "pool_misses": 0,
            "current_size": 0,
            "peak_usage": 0,
        }

    def set_connection_factory(self, factory: Callable) -> None:
        """Set the connection factory function."""
        self._connection_factory = factory

    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        with self._lock:
            # Try to get from pool
            if self._pool:
                connection = self._pool.pop()
                self._in_use.add(connection)
                self._stats["pool_hits"] += 1
                return connection

            # Create new connection if under limit
            if len(self._in_use) < self.max_connections:
                if not self._connection_factory:
                    raise RuntimeError("No connection factory configured")

                connection = self._connection_factory()
                self._in_use.add(connection)
                self._stats["total_created"] += 1
                self._stats["pool_misses"] += 1
                self._stats["current_size"] = len(self._pool) + len(self._in_use)
                self._stats["peak_usage"] = max(self._stats["peak_usage"], len(self._in_use))

                return connection

            # Pool exhausted
            raise RuntimeError(f"Connection pool exhausted (max: {self.max_connections})")

    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)

                # Check if connection is still valid
                if self._is_connection_valid(connection):
                    self._pool.append(connection)
                else:
                    self._destroy_connection(connection)

                self._stats["current_size"] = len(self._pool) + len(self._in_use)

                # Periodic cleanup
                if time.time() - self._last_cleanup > 60:  # Every minute
                    self._cleanup_idle_connections()

    def _is_connection_valid(self, connection: Any) -> bool:
        """Check if connection is still valid."""
        try:
            # Implement connection validation logic
            # This would depend on the connection type
            return hasattr(connection, "ping") and connection.ping()
        except Exception:
            return False

    def _destroy_connection(self, connection: Any) -> None:
        """Destroy a connection."""
        try:
            if hasattr(connection, "close"):
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            self._stats["total_destroyed"] += 1

    def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        with self._lock:
            current_time = time.time()
            pool_size = len(self._pool)

            # Remove excess connections beyond minimum
            while len(self._pool) > self.min_connections:
                connection = self._pool.pop(0)
                self._destroy_connection(connection)

            self._last_cleanup = current_time
            logger.debug(f"Pool cleanup: removed {pool_size - len(self._pool)} connections")

    def should_scale_up(self) -> bool:
        """Determine if pool should scale up."""
        with self._lock:
            if len(self._pool) + len(self._in_use) >= self.max_connections:
                return False

            utilization = len(self._in_use) / max(1, len(self._pool) + len(self._in_use))
            return utilization > self.scale_threshold

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                **self._stats,
                "pool_size": len(self._pool),
                "connections_in_use": len(self._in_use),
                "utilization_percent": (len(self._in_use) / self.max_connections) * 100,
                "hit_rate": self._stats["pool_hits"] / max(1, self._stats["pool_hits"] + self._stats["pool_misses"]),
            }


class QueryOptimizer:
    """Intelligent SQL query optimization system."""

    def __init__(self) -> None:
        self.optimization_rules = {
            "add_limit": self._add_limit_if_missing,
            "optimize_select": self._optimize_select_clause,
            "add_indexes_hint": self._suggest_indexes,
            "optimize_joins": self._optimize_joins,
            "cache_suggestions": self._suggest_caching,
        }
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = Lock()

    def optimize_query(
        self,
        sql_query: str,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Optimize SQL query with intelligent analysis.
        
        Args:
            sql_query: Original SQL query
            execution_context: Context information for optimization
            
        Returns:
            Optimization result with improved query and suggestions
        """
        query_hash = hash(sql_query)

        # Check cache first
        with self._cache_lock:
            if query_hash in self.query_cache:
                cached_result = self.query_cache[query_hash]
                cached_result["cache_hit"] = True
                return cached_result

        # Perform optimization
        optimized_query = sql_query
        optimizations_applied = []
        suggestions = []

        for rule_name, rule_func in self.optimization_rules.items():
            try:
                result = rule_func(optimized_query, execution_context or {})
                if result["modified"]:
                    optimized_query = result["query"]
                    optimizations_applied.append(rule_name)

                if result["suggestions"]:
                    suggestions.extend(result["suggestions"])

            except Exception as e:
                logger.warning(f"Optimization rule {rule_name} failed: {e}")

        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            sql_query, optimized_query, optimizations_applied,
        )

        result = {
            "original_query": sql_query,
            "optimized_query": optimized_query,
            "optimizations_applied": optimizations_applied,
            "suggestions": suggestions,
            "optimization_score": optimization_score,
            "estimated_performance_gain": len(optimizations_applied) * 0.15,  # Rough estimate
            "cache_hit": False,
        }

        # Cache result
        with self._cache_lock:
            self.query_cache[query_hash] = result
            # Limit cache size
            if len(self.query_cache) > 1000:
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]

        return result

    def _add_limit_if_missing(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add LIMIT clause if missing."""
        query_upper = query.upper()

        if "LIMIT" not in query_upper and "SELECT" in query_upper:
            # Don't add LIMIT to certain query types
            if any(keyword in query_upper for keyword in ["COUNT(", "SUM(", "MAX(", "MIN("]):
                return {"query": query, "modified": False, "suggestions": []}

            # Add reasonable LIMIT
            limit_value = context.get("default_limit", 1000)

            if query.rstrip().endswith(";"):
                optimized = query.rstrip()[:-1] + f" LIMIT {limit_value};"
            else:
                optimized = query.rstrip() + f" LIMIT {limit_value}"

            return {
                "query": optimized,
                "modified": True,
                "suggestions": [f"Added LIMIT {limit_value} to prevent large result sets"],
            }

        return {"query": query, "modified": False, "suggestions": []}

    def _optimize_select_clause(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize SELECT clause."""
        suggestions = []

        if "SELECT *" in query.upper():
            suggestions.append(
                "Consider specifying only needed columns instead of SELECT * "
                "for better performance and reduced network traffic",
            )

        return {"query": query, "modified": False, "suggestions": suggestions}

    def _suggest_indexes(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest database indexes for better performance."""
        suggestions = []
        query_upper = query.upper()

        # Look for WHERE clauses
        if "WHERE" in query_upper:
            suggestions.append(
                "Consider creating indexes on columns used in WHERE clauses "
                "for faster query execution",
            )

        # Look for JOIN conditions
        if "JOIN" in query_upper:
            suggestions.append(
                "Ensure proper indexes exist on JOIN columns for optimal performance",
            )

        # Look for ORDER BY
        if "ORDER BY" in query_upper:
            suggestions.append(
                "Consider composite indexes on ORDER BY columns to avoid sorting",
            )

        return {"query": query, "modified": False, "suggestions": suggestions}

    def _optimize_joins(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize JOIN operations."""
        suggestions = []
        query_upper = query.upper()

        join_count = query_upper.count("JOIN")

        if join_count > 3:
            suggestions.append(
                f"Query has {join_count} JOINs which may impact performance. "
                "Consider breaking into smaller queries or using temporary tables.",
            )

        # Check for potential Cartesian products
        if join_count > 0 and "ON" not in query_upper and "WHERE" not in query_upper:
            suggestions.append(
                "Potential Cartesian product detected. Ensure proper JOIN conditions.",
            )

        return {"query": query, "modified": False, "suggestions": suggestions}

    def _suggest_caching(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest caching opportunities."""
        suggestions = []
        query_upper = query.upper()

        # Suggest caching for expensive operations
        if any(keyword in query_upper for keyword in ["GROUP BY", "ORDER BY", "JOIN"]):
            suggestions.append(
                "This query performs expensive operations. Consider caching results "
                "if the data doesn't change frequently.",
            )

        # Suggest caching for aggregate functions
        if any(func in query_upper for func in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
            suggestions.append(
                "Aggregate functions detected. Consider materialized views or "
                "caching for frequently accessed aggregations.",
            )

        return {"query": query, "modified": False, "suggestions": suggestions}

    def _calculate_optimization_score(self, original: str, optimized: str, applied: List[str]) -> float:
        """Calculate optimization score."""
        base_score = 0.7  # Base score for any query

        # Bonus for optimizations applied
        optimization_bonus = len(applied) * 0.1

        # Bonus for query characteristics
        optimized_upper = optimized.upper()

        if "LIMIT" in optimized_upper:
            base_score += 0.1
        if "WHERE" in optimized_upper:
            base_score += 0.05
        if "SELECT *" not in optimized_upper:
            base_score += 0.05

        # Penalty for complexity
        complexity_penalty = min(0.2, optimized.count("(") * 0.02)

        final_score = min(1.0, base_score + optimization_bonus - complexity_penalty)
        return final_score

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query optimization cache statistics."""
        with self._cache_lock:
            return {
                "cache_size": len(self.query_cache),
                "cache_hit_rate": sum(1 for result in self.query_cache.values()
                                    if result.get("cache_hit", False)) / max(1, len(self.query_cache)),
            }


class AutoScalingManager:
    """Auto-scaling manager for dynamic resource allocation."""

    def __init__(
        self,
        profiler: PerformanceProfiler,
        pool_manager: ConnectionPoolManager,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scale_check_interval: float = 30.0,
    ) -> None:
        self.profiler = profiler
        self.pool_manager = pool_manager
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval

        self._scaling_thread = None
        self._scaling_active = False
        self._last_scale_action = 0
        self._scale_cooldown = 60.0  # Minimum time between scaling actions

    def start_auto_scaling(self) -> None:
        """Start auto-scaling monitoring."""
        if self._scaling_active:
            return

        self._scaling_active = True
        self._scaling_thread = threading.Thread(
            target=self._auto_scale_loop,
            daemon=True,
        )
        self._scaling_thread.start()
        logger.info("Auto-scaling started")

    def stop_auto_scaling(self) -> None:
        """Stop auto-scaling monitoring."""
        self._scaling_active = False
        if self._scaling_thread:
            self._scaling_thread.join(timeout=1.0)
        logger.info("Auto-scaling stopped")

    def _auto_scale_loop(self) -> None:
        """Main auto-scaling loop."""
        while self._scaling_active:
            try:
                self._evaluate_scaling_decision()
            except Exception as e:
                logger.error(f"Auto-scaling evaluation error: {e}")

            time.sleep(self.scale_check_interval)

    def _evaluate_scaling_decision(self) -> None:
        """Evaluate whether to scale up or down."""
        current_time = time.time()

        # Respect cooldown period
        if current_time - self._last_scale_action < self._scale_cooldown:
            return

        # Get current metrics
        pool_stats = self.pool_manager.get_pool_stats()
        resource_summary = self.profiler.get_resource_summary()

        utilization = pool_stats["utilization_percent"] / 100.0
        cpu_usage = resource_summary.get("current_cpu_percent", 0) / 100.0
        memory_usage = resource_summary.get("current_memory_percent", 0) / 100.0

        # Scale up decision
        if (
            utilization > self.scale_up_threshold or
            cpu_usage > self.scale_up_threshold or
            memory_usage > self.scale_up_threshold
        ):
            if self._can_scale_up():
                self._scale_up()
                self._last_scale_action = current_time

        # Scale down decision
        elif (
            utilization < self.scale_down_threshold and
            cpu_usage < self.scale_down_threshold and
            memory_usage < self.scale_down_threshold
        ):
            if self._can_scale_down():
                self._scale_down()
                self._last_scale_action = current_time

    def _can_scale_up(self) -> bool:
        """Check if scaling up is possible."""
        pool_stats = self.pool_manager.get_pool_stats()
        return pool_stats["pool_size"] + pool_stats["connections_in_use"] < self.pool_manager.max_connections

    def _can_scale_down(self) -> bool:
        """Check if scaling down is possible."""
        pool_stats = self.pool_manager.get_pool_stats()
        return pool_stats["pool_size"] > self.pool_manager.min_connections

    def _scale_up(self) -> None:
        """Scale up resources."""
        logger.info("Scaling up: Creating additional connections")
        # This would trigger connection pool expansion
        # The pool manager handles this automatically when connections are requested

    def _scale_down(self) -> None:
        """Scale down resources."""
        logger.info("Scaling down: Reducing connection pool size")
        # Trigger cleanup of idle connections
        self.pool_manager._cleanup_idle_connections()


# Global performance optimization instances
global_profiler = PerformanceProfiler()
global_pool_manager = ConnectionPoolManager()
global_query_optimizer = QueryOptimizer()
global_auto_scaler = AutoScalingManager(global_profiler, global_pool_manager)

# Start monitoring by default
global_profiler.start_monitoring()
global_auto_scaler.start_auto_scaling()


def optimize_operation(operation_type: str):
    """Decorator for automatic operation optimization and profiling."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            result, metrics = global_profiler.profile_operation(
                func, operation_type, *args, **kwargs,
            )

            # Log performance if operation was slow
            if metrics.duration > 5.0:  # 5 seconds threshold
                logger.warning(
                    f"Slow operation detected: {operation_type} took {metrics.duration:.2f}s",
                )

            return result
        return wrapper
    return decorator
