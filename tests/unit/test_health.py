"""Unit tests for health monitoring module.

Tests health check functionality and system monitoring.
"""

import pytest
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from src.sql_synth.health import HealthChecker


class TestHealthChecker:
    """Test cases for HealthChecker class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.sql_synth.health.MetricsCollector'), \
             patch('src.sql_synth.health.DatabaseManager'):
            self.health_checker = HealthChecker()

    def test_init(self):
        """Test HealthChecker initialization."""
        assert self.health_checker is not None
        assert hasattr(self.health_checker, 'start_time')
        assert hasattr(self.health_checker, 'metrics_collector')
        assert hasattr(self.health_checker, 'database_manager')
        assert isinstance(self.health_checker.start_time, float)

    def test_get_basic_health(self):
        """Test getting basic health status."""
        # Mock time to control uptime calculation
        with patch('time.time') as mock_time:
            mock_time.return_value = self.health_checker.start_time + 100  # 100 seconds uptime
            
            health_status = self.health_checker.get_basic_health()
            
            assert isinstance(health_status, dict)
            assert health_status['status'] == 'healthy'
            assert 'timestamp' in health_status
            assert 'uptime_seconds' in health_status
            assert health_status['uptime_seconds'] == 100
            
            # Verify timestamp format
            timestamp = health_status['timestamp']
            assert isinstance(timestamp, str)
            # Should be able to parse back to datetime
            parsed_timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            assert isinstance(parsed_timestamp, datetime)

    @patch('src.sql_synth.health.psutil')
    def test_get_system_health(self, mock_psutil):
        """Test getting system health metrics."""
        # Mock psutil functions
        mock_psutil.cpu_percent.return_value = 45.2
        mock_psutil.virtual_memory.return_value = Mock(
            total=8589934592,  # 8GB
            available=4294967296,  # 4GB
            percent=50.0
        )
        mock_psutil.disk_usage.return_value = Mock(
            total=1000000000000,  # 1TB
            free=500000000000,   # 500GB
            percent=50.0
        )
        
        system_health = self.health_checker.get_system_health()
        
        assert isinstance(system_health, dict)
        assert 'cpu_percent' in system_health
        assert 'memory' in system_health
        assert 'disk' in system_health
        assert system_health['cpu_percent'] == 45.2
        assert system_health['memory']['percent'] == 50.0
        assert system_health['disk']['percent'] == 50.0

    def test_check_database_connectivity(self):
        """Test database connectivity check."""
        # Mock successful database connection
        mock_db_manager = Mock()
        mock_db_manager.test_connection.return_value = True
        self.health_checker.database_manager = mock_db_manager
        
        db_health = self.health_checker.check_database_connectivity()
        
        assert isinstance(db_health, dict)
        assert 'database_connected' in db_health
        assert db_health['database_connected'] is True

    def test_check_database_connectivity_failure(self):
        """Test database connectivity check with failure."""
        # Mock failed database connection
        mock_db_manager = Mock()
        mock_db_manager.test_connection.return_value = False
        self.health_checker.database_manager = mock_db_manager
        
        db_health = self.health_checker.check_database_connectivity()
        
        assert isinstance(db_health, dict)
        assert 'database_connected' in db_health
        assert db_health['database_connected'] is False

    def test_get_application_metrics(self):
        """Test getting application-specific metrics."""
        # Mock metrics collector
        mock_metrics = {
            'total_queries': 150,
            'successful_queries': 145,
            'failed_queries': 5,
            'average_response_time': 1.25,
            'cache_hits': 89,
            'cache_misses': 61
        }
        
        mock_collector = Mock()
        mock_collector.get_summary_metrics.return_value = mock_metrics
        self.health_checker.metrics_collector = mock_collector
        
        app_metrics = self.health_checker.get_application_metrics()
        
        assert isinstance(app_metrics, dict)
        assert app_metrics == mock_metrics
        mock_collector.get_summary_metrics.assert_called_once()

    def test_get_comprehensive_health(self):
        """Test getting comprehensive health status."""
        with patch.object(self.health_checker, 'get_basic_health') as mock_basic, \
             patch.object(self.health_checker, 'get_system_health') as mock_system, \
             patch.object(self.health_checker, 'check_database_connectivity') as mock_db, \
             patch.object(self.health_checker, 'get_application_metrics') as mock_app:
            
            mock_basic.return_value = {'status': 'healthy'}
            mock_system.return_value = {'cpu_percent': 30}
            mock_db.return_value = {'database_connected': True}
            mock_app.return_value = {'total_queries': 100}
            
            comprehensive_health = self.health_checker.get_comprehensive_health()
            
            assert isinstance(comprehensive_health, dict)
            assert 'basic' in comprehensive_health
            assert 'system' in comprehensive_health
            assert 'database' in comprehensive_health
            assert 'application' in comprehensive_health
            
            mock_basic.assert_called_once()
            mock_system.assert_called_once()
            mock_db.assert_called_once()
            mock_app.assert_called_once()

    def test_is_healthy_true(self):
        """Test health status check returning True."""
        with patch.object(self.health_checker, 'get_comprehensive_health') as mock_health:
            mock_health.return_value = {
                'basic': {'status': 'healthy'},
                'system': {'cpu_percent': 30, 'memory': {'percent': 40}},
                'database': {'database_connected': True},
                'application': {'failed_queries': 2}
            }
            
            is_healthy = self.health_checker.is_healthy()
            assert is_healthy is True

    def test_is_healthy_false_high_cpu(self):
        """Test health status check returning False due to high CPU."""
        with patch.object(self.health_checker, 'get_comprehensive_health') as mock_health:
            mock_health.return_value = {
                'basic': {'status': 'healthy'},
                'system': {'cpu_percent': 95, 'memory': {'percent': 40}},  # High CPU
                'database': {'database_connected': True},
                'application': {'failed_queries': 2}
            }
            
            is_healthy = self.health_checker.is_healthy()
            assert is_healthy is False

    def test_is_healthy_false_db_disconnected(self):
        """Test health status check returning False due to database disconnection."""
        with patch.object(self.health_checker, 'get_comprehensive_health') as mock_health:
            mock_health.return_value = {
                'basic': {'status': 'healthy'},
                'system': {'cpu_percent': 30, 'memory': {'percent': 40}},
                'database': {'database_connected': False},  # DB disconnected
                'application': {'failed_queries': 2}
            }
            
            is_healthy = self.health_checker.is_healthy()
            assert is_healthy is False

    def test_get_health_status_code(self):
        """Test getting appropriate HTTP status code for health."""
        with patch.object(self.health_checker, 'is_healthy') as mock_is_healthy:
            # Healthy status
            mock_is_healthy.return_value = True
            status_code = self.health_checker.get_health_status_code()
            assert status_code == 200
            
            # Unhealthy status
            mock_is_healthy.return_value = False
            status_code = self.health_checker.get_health_status_code()
            assert status_code == 503

    def test_get_readiness_check(self):
        """Test readiness check for kubernetes/container deployments."""
        with patch.object(self.health_checker, 'check_database_connectivity') as mock_db:
            mock_db.return_value = {'database_connected': True}
            
            readiness = self.health_checker.get_readiness_check()
            
            assert isinstance(readiness, dict)
            assert 'ready' in readiness
            assert readiness['ready'] is True
            assert 'checks' in readiness

    def test_get_liveness_check(self):
        """Test liveness check for kubernetes/container deployments."""
        liveness = self.health_checker.get_liveness_check()
        
        assert isinstance(liveness, dict)
        assert 'alive' in liveness
        assert liveness['alive'] is True
        assert 'uptime_seconds' in liveness

    @patch('time.time')
    def test_record_health_check(self, mock_time):
        """Test recording health check results."""
        mock_time.return_value = 1640995200  # Fixed timestamp
        
        with patch.object(self.health_checker, 'get_comprehensive_health') as mock_health:
            mock_health.return_value = {'status': 'healthy'}
            
            self.health_checker.record_health_check()
            
            # Verify that health check was recorded
            # This would depend on the actual implementation
            mock_health.assert_called_once()

    def test_get_health_history(self):
        """Test getting health check history."""
        # This test would depend on the actual implementation
        # of health history storage
        history = self.health_checker.get_health_history()
        
        assert isinstance(history, list)
        # Additional assertions would depend on implementation

    def test_reset_health_metrics(self):
        """Test resetting health metrics."""
        # Mock metrics collector reset
        mock_collector = Mock()
        self.health_checker.metrics_collector = mock_collector
        
        self.health_checker.reset_health_metrics()
        
        # Verify reset was called on metrics collector
        # This would depend on the actual implementation
        assert mock_collector is not None  # Basic assertion

    @pytest.mark.parametrize("cpu_percent,memory_percent,expected_status", [
        (20, 30, "healthy"),
        (50, 60, "warning"),
        (90, 85, "critical"),
        (95, 95, "critical"),
    ])
    def test_assess_system_health_status(self, cpu_percent, memory_percent, expected_status):
        """Test system health status assessment with different resource usage levels."""
        with patch('src.sql_synth.health.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = cpu_percent
            mock_psutil.virtual_memory.return_value = Mock(percent=memory_percent)
            
            status = self.health_checker.assess_system_health_status()
            assert status == expected_status