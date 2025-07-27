"""
Monitoring and observability utilities for the SQL Synthesis system.
"""

import time
import logging
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
from functools import wraps
import json


@dataclass
class MetricData:
    """Data structure for metrics."""
    name: str
    value: float
    unit: str
    timestamp: float
    labels: Dict[str, str]


class HealthCheck:
    """Health check utilities."""
    
    @staticmethod
    def check_database(db_manager) -> Dict[str, Any]:
        """Check database connectivity and health."""
        try:
            start_time = time.time()
            with db_manager.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time * 1000, 2),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    @staticmethod
    def check_memory() -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy" if memory.percent < 90 else "warning",
            "usage_percent": memory.percent,
            "available_mb": round(memory.available / 1024 / 1024, 2),
            "total_mb": round(memory.total / 1024 / 1024, 2),
            "timestamp": time.time()
        }
    
    @staticmethod
    def check_disk() -> Dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        return {
            "status": "healthy" if usage_percent < 85 else "warning",
            "usage_percent": round(usage_percent, 2),
            "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
            "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
            "timestamp": time.time()
        }


class MetricsCollector:
    """Collect and manage application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, MetricData] = {}
        self.query_times = []
        self.error_counts = {}
    
    def record_query_time(self, query_time: float, query_type: str = "general"):
        """Record SQL query generation time."""
        self.query_times.append(query_time)
        
        # Keep only last 1000 measurements
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]
        
        metric = MetricData(
            name="query_generation_time",
            value=query_time,
            unit="seconds",
            timestamp=time.time(),
            labels={"type": query_type}
        )
        self.metrics[f"query_time_{query_type}"] = metric
    
    def record_error(self, error_type: str):
        """Record application errors."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        metric = MetricData(
            name="error_count",
            value=self.error_counts[error_type],
            unit="count",
            timestamp=time.time(),
            labels={"error_type": error_type}
        )
        self.metrics[f"error_{error_type}"] = metric
    
    def get_average_query_time(self) -> float:
        """Get average query generation time."""
        if not self.query_times:
            return 0.0
        return sum(self.query_times) / len(self.query_times)
    
    def get_p95_query_time(self) -> float:
        """Get 95th percentile query time."""
        if not self.query_times:
            return 0.0
        sorted_times = sorted(self.query_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "query_metrics": {
                "total_queries": len(self.query_times),
                "average_time": self.get_average_query_time(),
                "p95_time": self.get_p95_query_time(),
                "recent_queries": len([t for t in self.query_times if time.time() - t < 3600])
            },
            "error_metrics": self.error_counts,
            "system_metrics": {
                "memory": HealthCheck.check_memory(),
                "disk": HealthCheck.check_disk()
            },
            "timestamp": time.time()
        }


class PerformanceMonitor:
    """Monitor performance of operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    @contextmanager
    def measure_time(self, operation_name: str):
        """Context manager to measure operation time."""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self.metrics_collector.record_error(f"{operation_name}_error")
            raise
        finally:
            duration = time.time() - start_time
            self.metrics_collector.record_query_time(duration, operation_name)
    
    def monitor_function(self, operation_name: Optional[str] = None):
        """Decorator to monitor function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                with self.measure_time(op_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class StructuredLogger:
    """Structured logging for better observability."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging format."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_query_event(self, event_type: str, query: str, 
                       duration: Optional[float] = None, 
                       error: Optional[str] = None):
        """Log query-related events with structured data."""
        event_data = {
            "event_type": event_type,
            "query_length": len(query),
            "timestamp": time.time()
        }
        
        if duration is not None:
            event_data["duration_ms"] = round(duration * 1000, 2)
        
        if error:
            event_data["error"] = error
            self.logger.error(f"Query event: {json.dumps(event_data)}")
        else:
            self.logger.info(f"Query event: {json.dumps(event_data)}")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events."""
        event_data = {
            "event_type": f"security_{event_type}",
            "timestamp": time.time(),
            **details
        }
        
        self.logger.warning(f"Security event: {json.dumps(event_data)}")
    
    def log_performance_alert(self, metric_name: str, value: float, threshold: float):
        """Log performance alerts."""
        event_data = {
            "event_type": "performance_alert",
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "timestamp": time.time()
        }
        
        self.logger.warning(f"Performance alert: {json.dumps(event_data)}")


class ApplicationMonitor:
    """Main monitoring coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.logger = StructuredLogger("sql_synth")
        self.alert_thresholds = {
            "query_time": 5.0,  # seconds
            "memory_usage": 90.0,  # percent
            "disk_usage": 85.0,  # percent
            "error_rate": 0.1  # 10%
        }
    
    def get_health_status(self, db_manager=None) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "memory": HealthCheck.check_memory(),
                "disk": HealthCheck.check_disk()
            },
            "metrics": self.metrics_collector.get_metrics_summary()
        }
        
        if db_manager:
            health_data["checks"]["database"] = HealthCheck.check_database(db_manager)
        
        # Determine overall status
        for check_name, check_result in health_data["checks"].items():
            if check_result.get("status") == "unhealthy":
                health_data["status"] = "unhealthy"
                break
            elif check_result.get("status") == "warning":
                health_data["status"] = "warning"
        
        return health_data
    
    def check_alerts(self):
        """Check for alert conditions."""
        metrics = self.metrics_collector.get_metrics_summary()
        
        # Check query time alerts
        avg_query_time = metrics["query_metrics"]["average_time"]
        if avg_query_time > self.alert_thresholds["query_time"]:
            self.logger.log_performance_alert(
                "average_query_time", 
                avg_query_time, 
                self.alert_thresholds["query_time"]
            )
        
        # Check memory alerts
        memory_usage = metrics["system_metrics"]["memory"]["usage_percent"]
        if memory_usage > self.alert_thresholds["memory_usage"]:
            self.logger.log_performance_alert(
                "memory_usage", 
                memory_usage, 
                self.alert_thresholds["memory_usage"]
            )
        
        # Check disk alerts
        disk_usage = metrics["system_metrics"]["disk"]["usage_percent"]
        if disk_usage > self.alert_thresholds["disk_usage"]:
            self.logger.log_performance_alert(
                "disk_usage", 
                disk_usage, 
                self.alert_thresholds["disk_usage"]
            )
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.metrics_collector.get_metrics_summary()
        prometheus_output = []
        
        # Query metrics
        prometheus_output.append(f"# HELP sql_synth_query_total Total number of queries processed")
        prometheus_output.append(f"# TYPE sql_synth_query_total counter")
        prometheus_output.append(f"sql_synth_query_total {metrics['query_metrics']['total_queries']}")
        
        prometheus_output.append(f"# HELP sql_synth_query_duration_seconds Average query duration")
        prometheus_output.append(f"# TYPE sql_synth_query_duration_seconds gauge")
        prometheus_output.append(f"sql_synth_query_duration_seconds {metrics['query_metrics']['average_time']}")
        
        prometheus_output.append(f"# HELP sql_synth_query_duration_p95_seconds 95th percentile query duration")
        prometheus_output.append(f"# TYPE sql_synth_query_duration_p95_seconds gauge")
        prometheus_output.append(f"sql_synth_query_duration_p95_seconds {metrics['query_metrics']['p95_time']}")
        
        # Error metrics
        for error_type, count in metrics['error_metrics'].items():
            prometheus_output.append(f"# HELP sql_synth_errors_total Total number of errors by type")
            prometheus_output.append(f"# TYPE sql_synth_errors_total counter")
            prometheus_output.append(f"sql_synth_errors_total{{error_type=\"{error_type}\"}} {count}")
        
        # System metrics
        prometheus_output.append(f"# HELP sql_synth_memory_usage_percent Memory usage percentage")
        prometheus_output.append(f"# TYPE sql_synth_memory_usage_percent gauge")
        prometheus_output.append(f"sql_synth_memory_usage_percent {metrics['system_metrics']['memory']['usage_percent']}")
        
        prometheus_output.append(f"# HELP sql_synth_disk_usage_percent Disk usage percentage")
        prometheus_output.append(f"# TYPE sql_synth_disk_usage_percent gauge")
        prometheus_output.append(f"sql_synth_disk_usage_percent {metrics['system_metrics']['disk']['usage_percent']}")
        
        return "\n".join(prometheus_output)


# Global monitoring instance
monitor = ApplicationMonitor()