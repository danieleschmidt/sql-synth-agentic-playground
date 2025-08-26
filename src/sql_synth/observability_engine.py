"""
Observability Engine - Generation 2 Implementation
Comprehensive monitoring, metrics, tracing, and alerting system for SQL synthesis platform.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import uuid
import statistics
from contextlib import contextmanager
import functools

import numpy as np
import psutil

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from .advanced_error_handling import ErrorSeverity, ErrorCategory
except ImportError:
    class ErrorSeverity:
        DEBUG = "debug"
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
        FATAL = "fatal"
    
    class ErrorCategory:
        SYSTEM = "system"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "started"


@dataclass
class Alert:
    """System alert definition."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    conditions: Dict[str, Any]
    triggered_at: float
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    status: str
    last_check: float
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """High-performance metrics collection system."""
    
    def __init__(self, max_points: int = 100000):
        self.metrics = defaultdict(deque)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.rates = defaultdict(list)
        self.max_points = max_points
        self._lock = threading.RLock()
        
        # Aggregation windows
        self.aggregation_windows = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '24h': 86400
        }
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.counters[key] += value
            
            self._add_metric_point(
                name, self.counters[key], MetricType.COUNTER, labels or {}
            )
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge metric value."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.gauges[key] = value
            
            self._add_metric_point(
                name, value, MetricType.GAUGE, labels or {}
            )
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.histograms[key].append((time.time(), value))
            
            # Keep only recent values
            cutoff_time = time.time() - self.aggregation_windows['1h']
            self.histograms[key] = [
                (ts, val) for ts, val in self.histograms[key]
                if ts > cutoff_time
            ]
            
            self._add_metric_point(
                name, value, MetricType.HISTOGRAM, labels or {}
            )
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Create a timer context manager."""
        return TimerContext(self, name, labels or {})
    
    def rate(
        self,
        name: str,
        count: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a rate metric."""
        with self._lock:
            key = self._make_key(name, labels or {})
            self.rates[key].append((time.time(), count))
            
            # Keep only recent values for rate calculation
            cutoff_time = time.time() - self.aggregation_windows['5m']
            self.rates[key] = [
                (ts, count) for ts, count in self.rates[key]
                if ts > cutoff_time
            ]
            
            # Calculate current rate (events per second)
            if len(self.rates[key]) >= 2:
                recent_events = self.rates[key][-10:]  # Last 10 events
                time_span = recent_events[-1][0] - recent_events[0][0]
                if time_span > 0:
                    rate_value = len(recent_events) / time_span
                    self._add_metric_point(
                        name, rate_value, MetricType.RATE, labels or {}
                    )
    
    def get_metrics(
        self,
        name_pattern: Optional[str] = None,
        time_window: str = '1m'
    ) -> Dict[str, Any]:
        """Get metrics data."""
        with self._lock:
            window_seconds = self.aggregation_windows.get(time_window, 60)
            cutoff_time = time.time() - window_seconds
            
            result = {
                'counters': {},
                'gauges': {},
                'histograms': {},
                'rates': {},
                'summary': {}
            }
            
            # Counters
            for key, value in self.counters.items():
                if name_pattern is None or name_pattern in key:
                    result['counters'][key] = value
            
            # Gauges
            for key, value in self.gauges.items():
                if name_pattern is None or name_pattern in key:
                    result['gauges'][key] = value
            
            # Histograms
            for key, values in self.histograms.items():
                if name_pattern is None or name_pattern in key:
                    recent_values = [
                        val for ts, val in values if ts > cutoff_time
                    ]
                    if recent_values:
                        result['histograms'][key] = {
                            'count': len(recent_values),
                            'min': min(recent_values),
                            'max': max(recent_values),
                            'mean': statistics.mean(recent_values),
                            'median': statistics.median(recent_values),
                            'p95': np.percentile(recent_values, 95),
                            'p99': np.percentile(recent_values, 99)
                        }
            
            # Rates
            for key, events in self.rates.items():
                if name_pattern is None or name_pattern in key:
                    recent_events = [
                        (ts, count) for ts, count in events if ts > cutoff_time
                    ]
                    if len(recent_events) >= 2:
                        time_span = recent_events[-1][0] - recent_events[0][0]
                        if time_span > 0:
                            total_events = sum(count for _, count in recent_events)
                            rate = total_events / time_span
                            result['rates'][key] = rate
            
            # Summary statistics
            result['summary'] = {
                'total_counters': len(self.counters),
                'total_gauges': len(self.gauges),
                'total_histograms': len(self.histograms),
                'total_rates': len(self.rates),
                'collection_time': time.time(),
                'time_window': time_window
            }
            
            return result
    
    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}{{{','.join(label_parts)}}}"
    
    def _add_metric_point(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        labels: Dict[str, str]
    ) -> None:
        """Add metric point to collection."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time(),
            labels=labels,
            metric_type=metric_type
        )
        
        key = self._make_key(name, labels)
        self.metrics[key].append(point)
        
        # Keep only recent points
        while len(self.metrics[key]) > self.max_points:
            self.metrics[key].popleft()


class TimerContext:
    """Timer context manager for measuring execution time."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.histogram(f"{self.name}_duration", duration, self.labels)


class DistributedTracer:
    """Distributed tracing system for request tracking."""
    
    def __init__(self, service_name: str = "sql-synth"):
        self.service_name = service_name
        self.active_spans = {}
        self.completed_spans = deque(maxlen=10000)
        self._lock = threading.RLock()
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new trace span."""
        with self._lock:
            span_id = str(uuid.uuid4())
            if trace_id is None:
                trace_id = str(uuid.uuid4()) if parent_span_id is None else self._get_trace_id(parent_span_id)
            
            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=time.time(),
                tags=tags or {}
            )
            
            self.active_spans[span_id] = span
            return span
    
    def finish_span(
        self,
        span: TraceSpan,
        tags: Optional[Dict[str, Any]] = None,
        status: str = "completed"
    ) -> None:
        """Finish a trace span."""
        with self._lock:
            span.end_time = time.time()
            span.duration = span.end_time - span.start_time
            span.status = status
            
            if tags:
                span.tags.update(tags)
            
            # Move to completed spans
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            
            self.completed_spans.append(span)
    
    def add_log(self, span: TraceSpan, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Add log entry to span."""
        log_entry = {
            'timestamp': time.time(),
            'event': event,
            'payload': payload or {}
        }
        span.logs.append(log_entry)
    
    def trace_context(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for automatic span management."""
        return TraceContext(self, operation_name, parent_span_id, tags)
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary for a specific trace."""
        spans = [
            span for span in self.completed_spans
            if span.trace_id == trace_id
        ]
        
        if not spans:
            return {}
        
        total_duration = sum(span.duration or 0 for span in spans)
        operations = [span.operation_name for span in spans]
        
        return {
            'trace_id': trace_id,
            'total_spans': len(spans),
            'total_duration': total_duration,
            'operations': operations,
            'start_time': min(span.start_time for span in spans),
            'end_time': max(span.end_time or span.start_time for span in spans),
            'service': self.service_name,
            'spans': [asdict(span) for span in spans]
        }
    
    def _get_trace_id(self, span_id: str) -> str:
        """Get trace ID from span ID."""
        for span in self.active_spans.values():
            if span.span_id == span_id:
                return span.trace_id
        
        for span in self.completed_spans:
            if span.span_id == span_id:
                return span.trace_id
        
        return str(uuid.uuid4())  # Fallback


class TraceContext:
    """Context manager for distributed tracing."""
    
    def __init__(
        self,
        tracer: DistributedTracer,
        operation_name: str,
        parent_span_id: Optional[str],
        tags: Optional[Dict[str, Any]]
    ):
        self.tracer = tracer
        self.operation_name = operation_name
        self.parent_span_id = parent_span_id
        self.tags = tags
        self.span = None
    
    def __enter__(self) -> TraceSpan:
        self.span = self.tracer.start_span(
            self.operation_name,
            self.parent_span_id,
            tags=self.tags
        )
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            status = "error" if exc_type else "completed"
            error_tags = {}
            if exc_type:
                error_tags = {
                    'error': True,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val) if exc_val else ""
                }
            
            self.tracer.finish_span(self.span, error_tags, status)


class AlertManager:
    """Intelligent alerting system with adaptive thresholds."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = {}
        self.notification_channels = []
        self.alert_history = deque(maxlen=1000)
        self.suppression_rules = {}
        self._lock = threading.RLock()
    
    def add_alert_rule(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: str = "",
        cooldown_seconds: int = 300
    ) -> None:
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[name] = {
                'condition': condition,
                'severity': severity,
                'description': description,
                'cooldown_seconds': cooldown_seconds,
                'last_triggered': None,
                'trigger_count': 0
            }
    
    def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        triggered_alerts = []
        
        with self._lock:
            current_time = time.time()
            
            for rule_name, rule in self.alert_rules.items():
                try:
                    # Check cooldown
                    last_triggered = rule.get('last_triggered')
                    if (last_triggered and 
                        current_time - last_triggered < rule['cooldown_seconds']):
                        continue
                    
                    # Evaluate condition
                    if rule['condition'](metrics):
                        alert_id = f"{rule_name}_{int(current_time)}"
                        
                        alert = Alert(
                            alert_id=alert_id,
                            name=rule_name,
                            description=rule['description'],
                            severity=rule['severity'],
                            conditions={'metrics': metrics},
                            triggered_at=current_time
                        )
                        
                        triggered_alerts.append(alert)
                        self.alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        
                        # Update rule state
                        rule['last_triggered'] = current_time
                        rule['trigger_count'] += 1
                        
                        logger.warning(f"Alert triggered: {rule_name} - {rule['description']}")
                
                except Exception as e:
                    logger.error(f"Error evaluating alert rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved_at = time.time()
                del self.alerts[alert_id]
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self.alerts.values())


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = {}
        self._monitoring_active = False
        self._monitor_thread = None
    
    def start_monitoring(self, interval_seconds: float = 30.0) -> None:
        """Start system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def add_health_check(
        self,
        name: str,
        check_function: Callable[[], Dict[str, Any]]
    ) -> None:
        """Add a health check."""
        self.health_checks[name] = check_function
    
    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                self._run_health_checks()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Brief pause on error
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.gauge("system_cpu_percent", cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.metrics_collector.gauge("system_cpu_count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.gauge("system_memory_total", memory.total)
            self.metrics_collector.gauge("system_memory_used", memory.used)
            self.metrics_collector.gauge("system_memory_percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.gauge("system_disk_total", disk.total)
            self.metrics_collector.gauge("system_disk_used", disk.used)
            self.metrics_collector.gauge("system_disk_percent", disk.percent)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics_collector.gauge("system_network_bytes_sent", network.bytes_sent)
            self.metrics_collector.gauge("system_network_bytes_recv", network.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            self.metrics_collector.gauge("process_memory_rss", process.memory_info().rss)
            self.metrics_collector.gauge("process_cpu_percent", process.cpu_percent())
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    def _run_health_checks(self) -> None:
        """Run all health checks."""
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_function()
                response_time = time.time() - start_time
                
                status = result.get('status', 'unknown')
                self.metrics_collector.gauge(
                    "health_check_status",
                    1.0 if status == 'healthy' else 0.0,
                    {'check_name': name}
                )
                self.metrics_collector.histogram(
                    "health_check_response_time",
                    response_time,
                    {'check_name': name}
                )
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                self.metrics_collector.gauge(
                    "health_check_status",
                    0.0,
                    {'check_name': name}
                )


class ObservabilityEngine:
    """Main observability engine coordinating all monitoring components."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.tracer = DistributedTracer()
        self.alert_manager = AlertManager()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        
        # Initialize default alert rules
        self._setup_default_alerts()
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        
        logger.info("Observability Engine initialized")
    
    def _setup_default_alerts(self) -> None:
        """Setup default system alerts."""
        # High CPU usage alert
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            lambda metrics: metrics.get('gauges', {}).get('system_cpu_percent', 0) > 80,
            AlertSeverity.WARNING,
            "CPU usage is above 80%",
            cooldown_seconds=300
        )
        
        # High memory usage alert
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            lambda metrics: metrics.get('gauges', {}).get('system_memory_percent', 0) > 90,
            AlertSeverity.CRITICAL,
            "Memory usage is above 90%",
            cooldown_seconds=180
        )
        
        # High error rate alert
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            lambda metrics: metrics.get('rates', {}).get('sql_generation_errors', 0) > 0.1,
            AlertSeverity.WARNING,
            "SQL generation error rate is high",
            cooldown_seconds=120
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        metrics = self.metrics_collector.get_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'metrics': metrics,
            'alerts': {
                'active_count': len(active_alerts),
                'alerts': [asdict(alert) for alert in active_alerts]
            },
            'system_health': self._get_system_health(),
            'traces': {
                'active_spans': len(self.tracer.active_spans),
                'completed_spans': len(self.tracer.completed_spans)
            },
            'timestamp': time.time()
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        metrics = self.metrics_collector.get_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall health
        if any(alert.severity == AlertSeverity.CRITICAL for alert in active_alerts):
            health_status = "critical"
        elif any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            'status': health_status,
            'cpu_usage': metrics.get('gauges', {}).get('system_cpu_percent', 0),
            'memory_usage': metrics.get('gauges', {}).get('system_memory_percent', 0),
            'disk_usage': metrics.get('gauges', {}).get('system_disk_percent', 0),
            'active_alerts': len(active_alerts),
            'uptime': time.time() - (getattr(self, '_start_time', time.time()))
        }
    
    def instrument_function(
        self,
        name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        trace: bool = True
    ):
        """Decorator for automatic function instrumentation."""
        def decorator(func):
            func_name = name or f"{func.__module__}.{func.__name__}"
            func_labels = labels or {}
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Metrics
                self.metrics_collector.increment(f"{func_name}_calls", labels=func_labels)
                
                # Tracing
                with self.tracer.trace_context(func_name) as span:
                    span.tags.update(func_labels)
                    
                    # Timer
                    with self.metrics_collector.timer(f"{func_name}_duration", func_labels):
                        try:
                            result = await func(*args, **kwargs)
                            self.metrics_collector.increment(f"{func_name}_success", labels=func_labels)
                            return result
                        except Exception as e:
                            self.metrics_collector.increment(f"{func_name}_errors", labels=func_labels)
                            span.tags['error'] = True
                            span.tags['error_type'] = type(e).__name__
                            raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Metrics
                self.metrics_collector.increment(f"{func_name}_calls", labels=func_labels)
                
                # Timer
                with self.metrics_collector.timer(f"{func_name}_duration", func_labels):
                    try:
                        result = func(*args, **kwargs)
                        self.metrics_collector.increment(f"{func_name}_success", labels=func_labels)
                        return result
                    except Exception as e:
                        self.metrics_collector.increment(f"{func_name}_errors", labels=func_labels)
                        raise
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


# Global observability engine instance
global_observability = ObservabilityEngine()


# Convenience functions
def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Record a metric value."""
    global_observability.metrics_collector.gauge(name, value, labels)


def increment_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter metric."""
    global_observability.metrics_collector.increment(name, value, labels)


def start_trace(operation_name: str, **kwargs) -> TraceSpan:
    """Start a distributed trace span."""
    return global_observability.tracer.start_span(operation_name, **kwargs)


def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data."""
    return global_observability.get_dashboard_data()


def instrument(name: Optional[str] = None, **kwargs):
    """Instrument a function with observability."""
    return global_observability.instrument_function(name, **kwargs)