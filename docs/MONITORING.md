# Monitoring and Observability

## Overview

This document outlines the monitoring and observability strategy for the SQL Synthesis Agentic Playground, including metrics collection, alerting, and operational dashboards.

## Monitoring Stack

### Core Components

1. **Prometheus** - Metrics collection and storage
2. **Grafana** - Visualization and dashboards
3. **AlertManager** - Alert routing and management
4. **OpenTelemetry** - Distributed tracing
5. **Structured Logging** - Application logs

### Configuration Files

Monitoring configuration is located in:
- `config/prometheus.yml` - Prometheus configuration
- `config/alert_rules.yml` - Alerting rules
- `docker-compose.yml` - Monitoring stack deployment

## Key Metrics

### Application Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `sql_synthesis_requests_total` | Counter | Total SQL generation requests |
| `sql_synthesis_request_duration_seconds` | Histogram | Request processing time |
| `sql_synthesis_accuracy_ratio` | Gauge | Query accuracy against benchmarks |
| `sql_synthesis_errors_total` | Counter | Total errors by type |
| `database_connections_active` | Gauge | Active database connections |
| `database_query_duration_seconds` | Histogram | Database query execution time |

### Infrastructure Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `system_cpu_usage_percent` | Gauge | CPU utilization |
| `system_memory_usage_bytes` | Gauge | Memory consumption |
| `system_disk_usage_bytes` | Gauge | Disk space utilization |
| `network_bytes_total` | Counter | Network traffic |

### Business Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `active_users_total` | Gauge | Current active users |
| `query_types_distribution` | Counter | Distribution of query types |
| `benchmark_scores` | Gauge | Spider/WikiSQL benchmark scores |
| `user_satisfaction_score` | Gauge | User feedback ratings |

## Instrumentation

### Application Instrumentation

```python
# Example metrics instrumentation
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter('sql_synthesis_requests_total', 
                       'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('sql_synthesis_request_duration_seconds',
                           'Request latency', ['method', 'endpoint'])
ACCURACY_SCORE = Gauge('sql_synthesis_accuracy_ratio',
                      'Query accuracy score', ['benchmark'])

# Usage in application code
@REQUEST_LATENCY.time()
def generate_sql_query(natural_language_query):
    REQUEST_COUNT.labels(method='POST', endpoint='/generate').inc()
    
    # Process query
    result = process_query(natural_language_query)
    
    # Update accuracy metric
    accuracy = calculate_accuracy(result)
    ACCURACY_SCORE.labels(benchmark='spider').set(accuracy)
    
    return result
```

### Custom Metrics Collection

```python
# src/sql_synth/monitoring.py - Enhanced monitoring
import logging
import time
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, Info

logger = logging.getLogger(__name__)

class ApplicationMetrics:
    """Centralized metrics collection for the application."""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'sql_synthesis_requests_total',
            'Total SQL synthesis requests',
            ['endpoint', 'method', 'status']
        )
        
        self.request_duration = Histogram(
            'sql_synthesis_request_duration_seconds',
            'Time spent processing requests',
            ['endpoint', 'method']
        )
        
        # Business metrics
        self.query_accuracy = Gauge(
            'sql_synthesis_accuracy_ratio',
            'Query accuracy against benchmarks',
            ['benchmark', 'dialect']
        )
        
        self.active_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database_type']
        )
        
        # Error tracking
        self.error_count = Counter(
            'sql_synthesis_errors_total',
            'Total errors by type',
            ['error_type', 'component']
        )
        
        # System info
        self.app_info = Info(
            'sql_synthesis_app_info',
            'Application information'
        )
    
    def record_request(self, endpoint: str, method: str, status: str, duration: float):
        """Record request metrics."""
        self.request_count.labels(
            endpoint=endpoint, 
            method=method, 
            status=status
        ).inc()
        
        self.request_duration.labels(
            endpoint=endpoint, 
            method=method
        ).observe(duration)
    
    def record_accuracy(self, benchmark: str, dialect: str, score: float):
        """Record query accuracy."""
        self.query_accuracy.labels(
            benchmark=benchmark, 
            dialect=dialect
        ).set(score)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.error_count.labels(
            error_type=error_type, 
            component=component
        ).inc()

# Global metrics instance
metrics = ApplicationMetrics()

def monitor_endpoint(endpoint: str):
    """Decorator to monitor endpoint performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics.record_error(
                    error_type=type(e).__name__,
                    component=func.__module__
                )
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_request(
                    endpoint=endpoint,
                    method="POST",  # Adjust based on actual method
                    status=status,
                    duration=duration
                )
        
        return wrapper
    return decorator
```

## Alerting Rules

### Critical Alerts

```yaml
# High-priority alerts requiring immediate attention
groups:
  - name: critical
    rules:
      - alert: ApplicationDown
        expr: up{job="sql-synthesis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "SQL Synthesis application is down"
          description: "The application has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(sql_synthesis_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec for 2 minutes"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(sql_synthesis_request_duration_seconds_bucket[5m])) > 5
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "High response time"
          description: "95th percentile response time is {{ $value }}s"
```

### Warning Alerts

```yaml
  - name: warnings
    rules:
      - alert: AccuracyDegradation
        expr: sql_synthesis_accuracy_ratio < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Query accuracy degradation"
          description: "Accuracy has dropped to {{ $value }} for {{ $labels.benchmark }}"

      - alert: HighMemoryUsage
        expr: (system_memory_usage_bytes / system_memory_total_bytes) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
```

## Dashboards

### Application Dashboard

Key panels to include:

1. **Request Volume**: Requests per second over time
2. **Response Times**: P50, P95, P99 latencies
3. **Error Rates**: Error percentage and types
4. **Accuracy Metrics**: Benchmark performance trends
5. **Database Metrics**: Connection pool status, query times

### Infrastructure Dashboard

1. **System Resources**: CPU, memory, disk usage
2. **Network Traffic**: Inbound/outbound bandwidth
3. **Container Metrics**: Container resource usage
4. **Database Health**: Connection status, slow queries

### Business Dashboard

1. **User Activity**: Active users, session duration
2. **Query Analysis**: Query types, complexity distribution
3. **Performance Trends**: Historical accuracy and speed
4. **Usage Patterns**: Peak hours, feature adoption

## Log Management

### Structured Logging

```python
import logging
import json
from typing import Dict, Any

class StructuredLogger:
    """Structured logging for better observability."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _get_formatter(self):
        """JSON formatter for structured logs."""
        return logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        log_data = {"event": message, **kwargs}
        self.logger.info(json.dumps(log_data))
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error with structured data."""
        log_data = {
            "event": message,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            **kwargs
        }
        self.logger.error(json.dumps(log_data))

# Usage
logger = StructuredLogger(__name__)
logger.info("SQL query generated", 
           query_type="SELECT", 
           execution_time=1.23, 
           user_id="user123")
```

### Log Aggregation

Configure log shipping to centralized systems:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Cloud Logging**: AWS CloudWatch, GCP Cloud Logging
- **Modern Solutions**: Datadog, New Relic, Splunk

## Health Checks

### Application Health

```python
# src/sql_synth/health.py - Enhanced health checks
from typing import Dict, Any
import time
from sqlalchemy import text

class HealthChecker:
    """Comprehensive health checking."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        checks = {
            "timestamp": time.time(),
            "status": "healthy",
            "checks": {}
        }
        
        # Database connectivity
        checks["checks"]["database"] = self._check_database()
        
        # Memory usage
        checks["checks"]["memory"] = self._check_memory()
        
        # Disk space
        checks["checks"]["disk"] = self._check_disk()
        
        # Overall status
        if any(check["status"] != "healthy" for check in checks["checks"].values()):
            checks["status"] = "unhealthy"
        
        return checks
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "healthy", "response_time": 0.1}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        import psutil
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        if usage_percent > 90:
            status = "unhealthy"
        elif usage_percent > 80:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "usage_percent": usage_percent,
            "available_mb": memory.available // 1024 // 1024
        }
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk space."""
        import psutil
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        
        if usage_percent > 90:
            status = "unhealthy"
        elif usage_percent > 80:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "usage_percent": usage_percent,
            "free_gb": disk.free // 1024 // 1024 // 1024
        }
```

## Troubleshooting Guide

### Common Issues

1. **High Response Times**
   - Check database connection pool
   - Review slow query logs
   - Analyze application bottlenecks

2. **Memory Leaks**
   - Monitor memory usage trends
   - Profile application memory
   - Check for unclosed connections

3. **Accuracy Degradation**
   - Verify benchmark data integrity
   - Check model performance
   - Review recent code changes

### Monitoring Best Practices

1. **Metric Naming**: Use consistent naming conventions
2. **Label Consistency**: Maintain consistent label schemes
3. **Alert Fatigue**: Avoid over-alerting
4. **Documentation**: Keep runbooks updated
5. **Testing**: Test monitoring during deployments

## References

- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Grafana Dashboard Design](https://grafana.com/docs/grafana/latest/best-practices/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Structured Logging Best Practices](https://betterstack.com/community/guides/logging/structured-logging/)