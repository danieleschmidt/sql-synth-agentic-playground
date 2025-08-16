"""Advanced monitoring and alerting system for SQL synthesis agent."""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

from .logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    comparison: str
    duration: int
    is_active: bool = False
    triggered_at: Optional[datetime] = None


@dataclass 
class MetricValue:
    """Individual metric value."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]


class MonitoringSystem:
    """Comprehensive monitoring and alerting system."""

    def __init__(self):
        self.metrics: Dict[str, List[MetricValue]] = {}
        self.alerts: Dict[str, Alert] = {}
        self.max_metric_history = 10000
        self.system_monitor_interval = 30
        self.last_system_check = 0
        
        self._setup_default_alerts()
        logger.logger.info("Monitoring system initialized")

    def _setup_default_alerts(self) -> None:
        """Set up default system alerts."""
        default_alerts = [
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage exceeds 80%",
                severity=AlertSeverity.WARNING,
                metric_name="system.cpu_percent",
                threshold=80.0,
                comparison="gt",
                duration=60,
            ),
            Alert(
                id="high_memory_usage",
                name="High Memory Usage", 
                description="Memory usage exceeds 85%",
                severity=AlertSeverity.WARNING,
                metric_name="system.memory_percent",
                threshold=85.0,
                comparison="gt",
                duration=60,
            ),
        ]
        
        for alert in default_alerts:
            self.alerts[alert.id] = alert

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if labels is None:
            labels = {}
            
        metric_value = MetricValue(
            timestamp=datetime.now(),
            value=value,
            labels=labels,
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric_value)
        
        # Limit history size
        if len(self.metrics[name]) > self.max_metric_history:
            self.metrics[name] = self.metrics[name][-self.max_metric_history:]
        
        self._check_alerts_for_metric(name, value)

    def _check_alerts_for_metric(self, metric_name: str, value: float) -> None:
        """Check if any alerts should be triggered."""
        for alert in self.alerts.values():
            if alert.metric_name == metric_name:
                self._evaluate_alert(alert, value)

    def _evaluate_alert(self, alert: Alert, current_value: float) -> None:
        """Evaluate whether an alert should be triggered."""
        condition_met = False
        if alert.comparison == "gt":
            condition_met = current_value > alert.threshold
        elif alert.comparison == "lt":
            condition_met = current_value < alert.threshold
        
        if condition_met and not alert.is_active:
            self._trigger_alert(alert, current_value)
        elif not condition_met and alert.is_active:
            self._resolve_alert(alert, current_value)

    def _trigger_alert(self, alert: Alert, value: float) -> None:
        """Trigger an alert."""
        alert.is_active = True
        alert.triggered_at = datetime.now()
        
        logger.logger.error(
            "Alert triggered",
            alert_id=alert.id,
            alert_name=alert.name,
            severity=alert.severity.value,
            current_value=value,
        )

    def _resolve_alert(self, alert: Alert, value: float) -> None:
        """Resolve an active alert."""
        alert.is_active = False
        
        logger.logger.info(
            "Alert resolved",
            alert_id=alert.id,
            alert_name=alert.name,
            current_value=value,
        )

    def collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        current_time = time.time()
        
        if current_time - self.last_system_check < self.system_monitor_interval:
            return
        
        self.last_system_check = current_time
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_metric("system.cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system.disk_percent", disk_percent)
            
        except Exception as e:
            logger.log_error(error=e, context={"operation": "collect_system_metrics"})

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        self.collect_system_metrics()
        
        active_alerts = [
            {
                "id": alert.id,
                "name": alert.name,
                "severity": alert.severity.value,
                "description": alert.description,
            }
            for alert in self.alerts.values() if alert.is_active
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "active_alerts": active_alerts,
            "alert_count": len(active_alerts),
            "system_status": "healthy" if not active_alerts else "warning",
        }


# Global monitoring system instance
global_monitor = MonitoringSystem()


def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Quick function to record a metric."""
    global_monitor.record_metric(name, value, labels)


def get_monitoring_dashboard() -> Dict[str, Any]:
    """Get monitoring dashboard data."""
    return global_monitor.get_dashboard_data()