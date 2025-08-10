"""Performance and quality metrics tracking for SQL synthesis agent.

This module provides comprehensive metrics collection, analysis, and reporting
for monitoring agent performance, query quality, and system health.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class QueryMetric:
    """Individual query performance metric."""
    timestamp: datetime
    success: bool
    generation_time: float
    execution_time: Optional[float] = None
    query_length: Optional[int] = None
    rows_returned: Optional[int] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    dialect: Optional[str] = None


@dataclass
class PerformanceStats:
    """Performance statistics summary."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0
    avg_generation_time: float = 0.0
    avg_execution_time: float = 0.0
    avg_query_length: float = 0.0
    avg_rows_returned: float = 0.0
    p95_generation_time: float = 0.0
    p95_execution_time: float = 0.0
    min_generation_time: float = 0.0
    max_generation_time: float = 0.0
    last_24h_queries: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collection system."""

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)

        # Alert thresholds
        self.alert_thresholds = {
            "max_generation_time": 30.0,  # seconds
            "max_execution_time": 10.0,   # seconds
            "min_success_rate": 0.95,     # 95%
            "max_error_rate": 0.05,       # 5%
        }

    def record_query_metric(self, metric: QueryMetric) -> None:
        """Record a query performance metric."""
        with self.lock:
            self.metrics.append(metric)

            # Update counters
            self.counters["total_queries"] += 1
            if metric.success:
                self.counters["successful_queries"] += 1
            else:
                self.counters["failed_queries"] += 1
                if metric.error:
                    error_type = self._categorize_error(metric.error)
                    self.counters[f"error_{error_type}"] += 1

            # Update timers
            self.timers["generation_times"].append(metric.generation_time)
            if metric.execution_time is not None:
                self.timers["execution_times"].append(metric.execution_time)

            # Check for alerts
            self._check_alerts(metric)

    def _categorize_error(self, error: str) -> str:
        """Categorize error types for tracking."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        if "connection" in error_lower:
            return "connection"
        if "syntax" in error_lower or "parsing" in error_lower:
            return "syntax"
        if "permission" in error_lower or "unauthorized" in error_lower:
            return "permission"
        if "security" in error_lower or "injection" in error_lower:
            return "security"
        return "other"

    def _check_alerts(self, metric: QueryMetric) -> None:
        """Check if metric triggers any alerts."""
        alerts = []

        # Check generation time
        if metric.generation_time > self.alert_thresholds["max_generation_time"]:
            alerts.append(f"High generation time: {metric.generation_time:.2f}s")

        # Check execution time
        if (metric.execution_time is not None and
            metric.execution_time > self.alert_thresholds["max_execution_time"]):
            alerts.append(f"High execution time: {metric.execution_time:.2f}s")

        # Check recent success rate
        recent_metrics = self._get_recent_metrics(timedelta(minutes=5))
        if len(recent_metrics) >= 10:  # Only check if we have enough data
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            if success_rate < self.alert_thresholds["min_success_rate"]:
                alerts.append(f"Low success rate: {success_rate:.2%}")

        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Performance alert: {alert}")

    def _get_recent_metrics(self, time_window: timedelta) -> List[QueryMetric]:
        """Get metrics within a time window."""
        cutoff_time = datetime.now() - time_window
        return [m for m in self.metrics if m.timestamp >= cutoff_time]

    def get_performance_stats(self, time_window: Optional[timedelta] = None) -> PerformanceStats:
        """Get comprehensive performance statistics."""
        with self.lock:
            if time_window:
                metrics = self._get_recent_metrics(time_window)
            else:
                metrics = list(self.metrics)

            if not metrics:
                return PerformanceStats()

            successful_metrics = [m for m in metrics if m.success]
            failed_metrics = [m for m in metrics if not m.success]

            # Basic counts
            total_queries = len(metrics)
            successful_queries = len(successful_metrics)
            failed_queries = len(failed_metrics)
            success_rate = successful_queries / total_queries if total_queries > 0 else 0.0

            # Generation time stats
            generation_times = [m.generation_time for m in metrics]
            avg_generation_time = statistics.mean(generation_times) if generation_times else 0.0
            p95_generation_time = self._percentile(generation_times, 95) if generation_times else 0.0
            min_generation_time = min(generation_times) if generation_times else 0.0
            max_generation_time = max(generation_times) if generation_times else 0.0

            # Execution time stats
            execution_times = [m.execution_time for m in metrics if m.execution_time is not None]
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0.0
            p95_execution_time = self._percentile(execution_times, 95) if execution_times else 0.0

            # Query characteristics
            query_lengths = [m.query_length for m in metrics if m.query_length is not None]
            avg_query_length = statistics.mean(query_lengths) if query_lengths else 0.0

            rows_returned = [m.rows_returned for m in metrics if m.rows_returned is not None]
            avg_rows_returned = statistics.mean(rows_returned) if rows_returned else 0.0

            # Last 24h queries
            last_24h_metrics = self._get_recent_metrics(timedelta(hours=24))
            last_24h_queries = len(last_24h_metrics)

            # Error analysis
            error_types = defaultdict(int)
            for metric in failed_metrics:
                if metric.error:
                    error_category = self._categorize_error(metric.error)
                    error_types[error_category] += 1

            return PerformanceStats(
                total_queries=total_queries,
                successful_queries=successful_queries,
                failed_queries=failed_queries,
                success_rate=success_rate,
                avg_generation_time=avg_generation_time,
                avg_execution_time=avg_execution_time,
                avg_query_length=avg_query_length,
                avg_rows_returned=avg_rows_returned,
                p95_generation_time=p95_generation_time,
                p95_execution_time=p95_execution_time,
                min_generation_time=min_generation_time,
                max_generation_time=max_generation_time,
                last_24h_queries=last_24h_queries,
                error_types=dict(error_types),
            )

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        recent_stats = self.get_performance_stats(timedelta(minutes=15))

        # Determine health status
        health_score = 100
        issues = []

        # Check success rate
        if recent_stats.success_rate < 0.9:
            health_score -= 30
            issues.append(f"Low success rate: {recent_stats.success_rate:.1%}")

        # Check response times
        if recent_stats.avg_generation_time > 15:
            health_score -= 20
            issues.append(f"High generation time: {recent_stats.avg_generation_time:.1f}s")

        if recent_stats.avg_execution_time > 5:
            health_score -= 15
            issues.append(f"High execution time: {recent_stats.avg_execution_time:.1f}s")

        # Check error rate
        error_rate = recent_stats.failed_queries / max(recent_stats.total_queries, 1)
        if error_rate > 0.1:
            health_score -= 25
            issues.append(f"High error rate: {error_rate:.1%}")

        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "health_score": max(health_score, 0),
            "issues": issues,
            "stats": recent_stats,
            "timestamp": datetime.now().isoformat(),
        }


class QueryMetrics:
    """Simplified metrics interface for agent usage."""

    def __init__(self):
        self.collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)

    def record_generation(
        self,
        success: bool,
        generation_time: float,
        query_length: Optional[int] = None,
        error: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> None:
        """Record SQL generation metrics."""
        metric = QueryMetric(
            timestamp=datetime.now(),
            success=success,
            generation_time=generation_time,
            query_length=query_length,
            error=error,
            model_used=model_used,
        )
        self.collector.record_query_metric(metric)

    def record_execution(
        self,
        success: bool,
        execution_time: float,
        rows_returned: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record SQL execution metrics."""
        # Find the most recent generation metric to update
        with self.collector.lock:
            if self.collector.metrics:
                last_metric = self.collector.metrics[-1]
                # Update the last metric with execution data
                last_metric.execution_time = execution_time
                if not success:
                    last_metric.success = False
                    last_metric.error = error
                if rows_returned is not None:
                    last_metric.rows_returned = rows_returned

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        stats = self.collector.get_performance_stats()
        health = self.collector.get_health_status()

        return {
            "performance": {
                "total_queries": stats.total_queries,
                "success_rate": stats.success_rate,
                "avg_generation_time": stats.avg_generation_time,
                "avg_execution_time": stats.avg_execution_time,
                "last_24h_queries": stats.last_24h_queries,
            },
            "health": health,
            "errors": stats.error_types,
            "thresholds": self.collector.alert_thresholds,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics (for testing/development)."""
        with self.collector.lock:
            self.collector.metrics.clear()
            self.collector.counters.clear()
            self.collector.timers.clear()
        self.logger.info("Metrics reset")


# Performance monitoring decorator
def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                execution_time = time.time() - start_time

                # Log performance metric
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Performance: {operation_name} - "
                    f"Time: {execution_time:.3f}s, Success: {success}",
                    extra={
                        "operation": operation_name,
                        "execution_time": execution_time,
                        "success": success,
                        "error": error,
                    },
                )

        return wrapper
    return decorator


# Global metrics instance
global_metrics = QueryMetrics()
