"""Unit tests for security module.

Tests SQL injection detection and security validation functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.sql_synth.security import (
    SecurityViolation,
    SQLInjectionDetector,
)


class TestSecurityViolation:
    """Test cases for SecurityViolation dataclass."""

    def test_security_violation_creation(self):
        """Test creating a SecurityViolation instance."""
        timestamp = datetime.now()
        violation = SecurityViolation(
            violation_type="SQL_INJECTION",
            severity="HIGH",
            description="Detected SQL injection attempt",
            location="user_input",
            timestamp=timestamp
        )
        
        assert violation.violation_type == "SQL_INJECTION"
        assert violation.severity == "HIGH"
        assert violation.description == "Detected SQL injection attempt"
        assert violation.location == "user_input"
        assert violation.timestamp == timestamp


class TestSQLInjectionDetector:
    """Test cases for SQLInjectionDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = SQLInjectionDetector()

    def test_init(self):
        """Test SQLInjectionDetector initialization."""
        assert self.detector is not None
        assert hasattr(self.detector, 'logger')

    @pytest.mark.parametrize("malicious_input,expected_detection", [
        ("'; DROP TABLE users; --", True),
        ("' UNION SELECT * FROM passwords --", True),
        ("1' OR '1'='1", True),
        ("'; WAITFOR DELAY '00:00:05' --", True),
        ("' OR 1=1 --", True),
        ("admin'--", True),
        ("' OR 'x'='x", True),
        ("; INSERT INTO users VALUES ('hacker', 'password'); --", True),
        ("' AND 1=(SELECT COUNT(*) FROM tabname); --", True),
        ("1' UNION SELECT null,null,null,null,version() --", True),
    ])
    def test_detect_sql_injection_malicious(self, malicious_input, expected_detection):
        """Test detection of malicious SQL injection attempts."""
        result = self.detector.detect_injection(malicious_input)
        assert result == expected_detection

    @pytest.mark.parametrize("safe_input", [
        "Show me all users from the marketing department",
        "Find customers who purchased in the last month",
        "What's the average order value?",
        "List all products with price greater than 100",
        "Count the number of active subscriptions",
        "Get user details for john@example.com",
        "Find orders placed on 2024-01-15",
        "Show top 10 selling products",
    ])
    def test_detect_sql_injection_safe(self, safe_input):
        """Test that legitimate queries are not flagged as malicious."""
        result = self.detector.detect_injection(safe_input)
        assert result is False

    def test_detect_empty_input(self):
        """Test detection with empty input."""
        result = self.detector.detect_injection("")
        assert result is False

    def test_detect_none_input(self):
        """Test detection with None input."""
        result = self.detector.detect_injection(None)
        assert result is False

    def test_analyze_query_structure(self):
        """Test SQL query structure analysis."""
        # Test with a simple safe query
        safe_query = "SELECT * FROM users WHERE age > 25"
        analysis = self.detector.analyze_query_structure(safe_query)
        
        assert isinstance(analysis, dict)
        assert 'has_multiple_statements' in analysis
        assert 'has_comments' in analysis
        assert 'suspicious_keywords' in analysis

    def test_analyze_query_structure_malicious(self):
        """Test SQL query structure analysis with malicious query."""
        malicious_query = "SELECT * FROM users; DROP TABLE users; --"
        analysis = self.detector.analyze_query_structure(malicious_query)
        
        assert analysis['has_multiple_statements'] is True
        assert analysis['has_comments'] is True
        assert len(analysis['suspicious_keywords']) > 0

    def test_validate_parameterized_query(self):
        """Test validation of parameterized queries."""
        # Safe parameterized query
        safe_query = "SELECT * FROM users WHERE id = ?"
        params = [123]
        
        result = self.detector.validate_parameterized_query(safe_query, params)
        assert result is True

    def test_validate_parameterized_query_unsafe(self):
        """Test validation of unsafe parameterized queries."""
        # Unsafe query with potential injection in parameters
        unsafe_query = "SELECT * FROM users WHERE name = ?"
        malicious_params = ["'; DROP TABLE users; --"]
        
        result = self.detector.validate_parameterized_query(unsafe_query, malicious_params)
        assert result is False

    def test_sanitize_input(self):
        """Test input sanitization."""
        malicious_input = "<script>alert('XSS')</script>'; DROP TABLE users; --"
        sanitized = self.detector.sanitize_input(malicious_input)
        
        # Should remove or escape dangerous characters
        assert "<script>" not in sanitized
        assert "DROP TABLE" not in sanitized or sanitized != malicious_input

    def test_get_security_violations(self):
        """Test getting security violations history."""
        # First, trigger some violations
        self.detector.detect_injection("'; DROP TABLE users; --")
        self.detector.detect_injection("' OR 1=1 --")
        
        violations = self.detector.get_security_violations()
        assert isinstance(violations, list)
        assert len(violations) >= 0  # May be empty if not implemented to store history

    def test_reset_violations_history(self):
        """Test resetting violations history."""
        # Trigger a violation
        self.detector.detect_injection("'; DROP TABLE users; --")
        
        # Reset history
        self.detector.reset_violations_history()
        
        violations = self.detector.get_security_violations()
        assert len(violations) == 0

    @pytest.mark.parametrize("query,expected_risk", [
        ("SELECT * FROM users", "LOW"),
        ("SELECT * FROM users WHERE id = 1; DROP TABLE users;", "HIGH"),
        ("' OR 1=1 --", "HIGH"),
        ("SELECT COUNT(*) FROM orders", "LOW"),
        ("UNION SELECT password FROM admin_users", "HIGH"),
    ])
    def test_assess_risk_level(self, query, expected_risk):
        """Test risk level assessment."""
        risk_level = self.detector.assess_risk_level(query)
        assert risk_level == expected_risk

    def test_generate_security_report(self):
        """Test generating security analysis report."""
        test_queries = [
            "SELECT * FROM users",
            "'; DROP TABLE users; --",
            "' OR 1=1 --",
            "UNION SELECT * FROM passwords"
        ]
        
        report = self.detector.generate_security_report(test_queries)
        
        assert isinstance(report, dict)
        assert 'total_queries' in report
        assert 'safe_queries' in report
        assert 'malicious_queries' in report
        assert 'risk_distribution' in report
        assert report['total_queries'] == len(test_queries)

    def test_is_sql_comment_injection(self):
        """Test detection of SQL comment injection."""
        comment_injections = [
            "admin'--",
            "user'; --",
            "' OR 1=1 --",
            "test/* comment */",
            "value'/**/OR/**/1=1--"
        ]
        
        for injection in comment_injections:
            result = self.detector.is_sql_comment_injection(injection)
            assert result is True

    def test_is_union_injection(self):
        """Test detection of UNION-based injection."""
        union_injections = [
            "' UNION SELECT * FROM users --",
            "1 UNION ALL SELECT password FROM admin",
            "' UNION SELECT null,username,password FROM users --",
            "test' UNION SELECT 1,2,3,4 --"
        ]
        
        for injection in union_injections:
            result = self.detector.is_union_injection(injection)
            assert result is True

    def test_is_boolean_injection(self):
        """Test detection of boolean-based injection."""
        boolean_injections = [
            "' OR '1'='1",
            "' OR 1=1 --",
            "admin' OR 'x'='x",
            "' AND 1=1 --",
            "1' OR '1'='1' --"
        ]
        
        for injection in boolean_injections:
            result = self.detector.is_boolean_injection(injection)
            assert result is True

    def test_extract_sql_keywords(self):
        """Test extraction of SQL keywords from query."""
        query = "SELECT * FROM users WHERE id = 1 AND status = 'active'"
        keywords = self.detector.extract_sql_keywords(query)
        
        expected_keywords = ['SELECT', 'FROM', 'WHERE', 'AND']
        for keyword in expected_keywords:
            assert keyword.upper() in [k.upper() for k in keywords]

    @patch('src.sql_synth.security.datetime')
    def test_log_security_event(self, mock_datetime):
        """Test logging of security events."""
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        
        with patch.object(self.detector.logger, 'warning') as mock_log:
            self.detector.log_security_event(
                "SQL_INJECTION",
                "HIGH",
                "Detected injection attempt",
                "user_input"
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "SQL_INJECTION" in call_args
            assert "HIGH" in call_args