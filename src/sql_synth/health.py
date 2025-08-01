"""Health check and monitoring endpoints for the SQL synthesis application."""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import psutil
import streamlit as st
from src.sql_synth.database import DatabaseManager
from src.sql_synth.monitoring import MetricsCollector


class HealthChecker:
    """Provides health check functionality for the application."""

    def __init__(self):
        self.start_time = time.time()
        self.metrics_collector = MetricsCollector()
        self.database_manager = DatabaseManager()

    def get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status."""
        current_time = time.time()
        uptime_seconds = current_time - self.start_time

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "version": "0.1.0",
            "service": "sql-synth-agentic-playground"
        }

    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health status including dependencies."""
        basic_health = self.get_basic_health()
        
        # Check database connectivity
        db_status = self._check_database_health()
        
        # Check system resources
        system_status = self._check_system_health()
        
        # Check application metrics
        app_status = self._check_application_health()
        
        # Determine overall status
        overall_status = "healthy"
        if (not db_status["healthy"] or 
            not system_status["healthy"] or 
            not app_status["healthy"]):
            overall_status = "unhealthy"
        
        return {
            **basic_health,
            "status": overall_status,
            "checks": {
                "database": db_status,
                "system": system_status,
                "application": app_status
            }
        }

    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Try to connect and run a simple query
            with self.database_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            response_time = time.time() - start_time
            
            return {
                "healthy": True,
                "response_time_ms": round(response_time * 1000, 2),
                "message": "Database connection successful"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "message": "Database connection failed"
            }

    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine if system is healthy based on thresholds
            cpu_healthy = cpu_percent < 80.0
            memory_healthy = memory_percent < 80.0
            disk_healthy = disk_percent < 90.0
            
            overall_healthy = cpu_healthy and memory_healthy and disk_healthy
            
            return {
                "healthy": overall_healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "thresholds": {
                    "cpu_max": 80.0,
                    "memory_max": 80.0,
                    "disk_max": 90.0
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "message": "System health check failed"
            }

    def _check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health metrics."""
        try:
            # Get application metrics
            metrics = self.metrics_collector.get_current_metrics()
            
            # Check for any critical errors in recent time window
            error_rate = metrics.get("error_rate", 0.0)
            response_time_avg = metrics.get("avg_response_time", 0.0)
            
            # Define health thresholds
            error_rate_healthy = error_rate < 0.05  # Less than 5% error rate
            response_time_healthy = response_time_avg < 5.0  # Less than 5 seconds avg
            
            overall_healthy = error_rate_healthy and response_time_healthy
            
            return {
                "healthy": overall_healthy,
                "error_rate": error_rate,
                "avg_response_time_ms": round(response_time_avg * 1000, 2),
                "total_requests": metrics.get("total_requests", 0),
                "active_sessions": metrics.get("active_sessions", 0),
                "thresholds": {
                    "max_error_rate": 0.05,
                    "max_avg_response_time_ms": 5000
                }
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "message": "Application health check failed"
            }

    def get_readiness(self) -> Dict[str, Any]:
        """Check if application is ready to serve requests."""
        try:
            # Check if all critical dependencies are available
            db_ready = self._check_database_health()["healthy"]
            
            # Check if application has completed initialization
            app_initialized = hasattr(st.session_state, 'app_initialized')
            
            ready = db_ready and app_initialized
            
            return {
                "ready": ready,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checks": {
                    "database": db_ready,
                    "application_initialized": app_initialized
                }
            }
            
        except Exception as e:
            return {
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_liveness(self) -> Dict[str, Any]:
        """Check if application is alive (basic liveness probe)."""
        return {
            "alive": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get application metrics in Prometheus format."""
        try:
            metrics = self.metrics_collector.get_prometheus_metrics()
            return {
                "status": "success",
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global health checker instance
health_checker = HealthChecker()


def create_health_endpoints():
    """Create health check endpoints for Streamlit app."""
    
    # Add query parameters for different health checks
    query_params = st.query_params
    
    if "health" in query_params:
        health_type = query_params.get("health", "basic")
        
        if health_type == "detailed":
            health_data = health_checker.get_detailed_health()
        elif health_type == "ready":
            health_data = health_checker.get_readiness()
        elif health_type == "live":
            health_data = health_checker.get_liveness()
        elif health_type == "metrics":
            health_data = health_checker.get_metrics()
        else:
            health_data = health_checker.get_basic_health()
        
        # Return JSON response
        st.json(health_data)
        return True
    
    return False


def add_health_sidebar():
    """Add health status information to Streamlit sidebar."""
    with st.sidebar:
        st.subheader("🏥 System Health")
        
        # Get basic health info
        health_data = health_checker.get_basic_health()
        
        # Display uptime
        uptime_hours = health_data["uptime_seconds"] / 3600
        st.metric("Uptime", f"{uptime_hours:.1f} hours")
        
        # Get system metrics
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
            st.metric("Memory Usage", f"{memory_percent:.1f}%")
            
            # Color-code based on usage
            if cpu_percent > 80:
                st.error("High CPU usage!")
            elif cpu_percent > 60:
                st.warning("Moderate CPU usage")
            
            if memory_percent > 80:
                st.error("High memory usage!")
            elif memory_percent > 60:
                st.warning("Moderate memory usage")
                
        except Exception as e:
            st.error(f"Failed to get system metrics: {e}")
        
        # Health check buttons
        if st.button("🔍 Detailed Health Check"):
            detailed_health = health_checker.get_detailed_health()
            st.json(detailed_health)
        
        if st.button("📊 Application Metrics"):
            metrics = health_checker.get_metrics()
            st.json(metrics)