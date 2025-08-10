"""
Security utilities and validation for SQL synthesis.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import sqlparse
from sqlparse import tokens


@dataclass
class SecurityViolation:
    """Represents a security violation."""
    violation_type: str
    severity: str
    description: str
    location: str
    timestamp: datetime


class SQLInjectionDetector:
    """Detect potential SQL injection attempts."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common SQL injection patterns
        self.injection_patterns = [
            # Classic injection patterns
            r"';.*drop\s+table",
            r"';.*delete\s+from",
            r"';.*insert\s+into",
            r"';.*update\s+.*set",
            r"'.*union\s+select",
            r"'.*or\s+1\s*=\s*1",
            r"'.*or\s+'1'\s*=\s*'1'",
            r"'.*and\s+1\s*=\s*1",

            # Comment-based injections
            r"--.*",
            r"/\*.*\*/",
            r"#.*",

            # Time-based blind injection
            r"waitfor\s+delay",
            r"sleep\s*\(",
            r"benchmark\s*\(",

            # Stacked queries
            r";\s*drop\s+",
            r";\s*delete\s+",
            r";\s*insert\s+",
            r";\s*update\s+",

            # Function-based injection
            r"concat\s*\(",
            r"char\s*\(",
            r"ascii\s*\(",
            r"substring\s*\(",

            # Boolean-based injection
            r"'\s*or\s*'.*'\s*=\s*'",
            r"'\s*and\s*'.*'\s*=\s*'",
            r"'\s*or\s*'.*'\s*like\s*'",
        ]

        # Compile patterns for better performance
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.injection_patterns
        ]

        # Dangerous keywords that should trigger alerts
        self.dangerous_keywords = {
            "drop", "delete", "truncate", "alter", "create", "grant", "revoke",
            "exec", "execute", "sp_", "xp_", "cmdshell", "openrowset", "opendatasource",
        }

    def analyze_query(self, query: str) -> List[SecurityViolation]:
        """Analyze a query for potential injection attempts."""
        violations = []

        # Check for injection patterns
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(query):
                violations.append(SecurityViolation(
                    violation_type="sql_injection_pattern",
                    severity="high",
                    description=f"Detected SQL injection pattern: {self.injection_patterns[i]}",
                    location="user_input",
                    timestamp=datetime.now(),
                ))

        # Check for dangerous keywords
        query_lower = query.lower()
        for keyword in self.dangerous_keywords:
            if keyword in query_lower:
                violations.append(SecurityViolation(
                    violation_type="dangerous_keyword",
                    severity="medium",
                    description=f"Dangerous SQL keyword detected: {keyword}",
                    location="user_input",
                    timestamp=datetime.now(),
                ))

        # Analyze SQL structure
        try:
            parsed = sqlparse.parse(query)
            if parsed:
                structure_violations = self._analyze_sql_structure(parsed[0])
                violations.extend(structure_violations)
        except Exception as e:
            self.logger.warning(f"Failed to parse SQL for security analysis: {e}")

        return violations

    def _analyze_sql_structure(self, parsed_sql) -> List[SecurityViolation]:
        """Analyze parsed SQL structure for security issues."""
        violations = []

        # Check for multiple statements (potential stacked queries)
        statements = [token for token in parsed_sql.flatten()
                     if token.ttype is tokens.Keyword and token.value.upper() in
                     ("SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE")]

        if len(statements) > 1:
            violations.append(SecurityViolation(
                violation_type="stacked_queries",
                severity="high",
                description="Multiple SQL statements detected (potential stacked query injection)",
                location="sql_structure",
                timestamp=datetime.now(),
            ))

        # Check for suspicious functions
        for token in parsed_sql.flatten():
            if token.ttype is tokens.Name:
                token_value = token.value.lower()
                if token_value in ["load_file", "into_outfile", "into_dumpfile"]:
                    violations.append(SecurityViolation(
                        violation_type="file_operation",
                        severity="high",
                        description=f"File operation function detected: {token_value}",
                        location="sql_function",
                        timestamp=datetime.now(),
                    ))

        return violations


class InputValidator:
    """Validate and sanitize user inputs."""

    def __init__(self):
        self.max_query_length = 10000
        self.max_word_length = 100

        # Allowed characters for natural language queries
        self.allowed_chars = re.compile(r'^[a-zA-Z0-9\s\.,\?\!\-\(\)\'\"]+$')

        # Common encoding attack patterns
        self.encoding_attacks = [
            r"%[0-9a-fA-F]{2}",  # URL encoding
            r"&#[0-9]+;",        # HTML entity encoding
            r"\\x[0-9a-fA-F]{2}", # Hex encoding
            r"\\u[0-9a-fA-F]{4}", # Unicode encoding
        ]

        self.compiled_encoding_patterns = [
            re.compile(pattern) for pattern in self.encoding_attacks
        ]

    def validate_natural_language_query(self, query: str) -> Tuple[bool, List[str]]:
        """Validate a natural language query input."""
        errors = []

        # Check length
        if len(query) > self.max_query_length:
            errors.append(f"Query too long: {len(query)} characters (max: {self.max_query_length})")

        if len(query.strip()) == 0:
            errors.append("Query cannot be empty")

        # Check for suspicious characters
        if not self.allowed_chars.match(query):
            errors.append("Query contains invalid characters")

        # Check for encoding attacks
        for pattern in self.compiled_encoding_patterns:
            if pattern.search(query):
                errors.append("Potential encoding attack detected")

        # Check for excessively long words (potential buffer overflow)
        words = query.split()
        long_words = [word for word in words if len(word) > self.max_word_length]
        if long_words:
            errors.append(f"Words too long: {long_words[:3]}..." if len(long_words) > 3 else f"Words too long: {long_words}")

        # Check for repeated characters (potential DoS)
        for char in query:
            if query.count(char) > len(query) * 0.5:  # More than 50% of the same character
                errors.append(f"Excessive repetition of character: {char}")
                break

        return len(errors) == 0, errors

    def sanitize_input(self, query: str) -> str:
        """Sanitize user input while preserving meaning."""
        # Remove null bytes
        query = query.replace("\x00", "")

        # Normalize whitespace
        query = " ".join(query.split())

        # Limit length
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]

        return query.strip()


class QueryAnalyzer:
    """Analyze generated SQL queries for security compliance."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Required security features for generated queries
        self.security_requirements = {
            "must_use_parameters": True,
            "no_dynamic_sql": True,
            "limit_result_sets": True,
            "no_admin_operations": True,
        }

    def validate_generated_query(self, query: str, parameters: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """Validate that a generated query meets security requirements."""
        errors = []

        # Check for parameterized queries
        if self.security_requirements["must_use_parameters"]:
            if not self._is_parameterized(query, parameters):
                errors.append("Query must use parameterized statements")

        # Check for dynamic SQL construction
        if self.security_requirements["no_dynamic_sql"]:
            if self._contains_dynamic_sql(query):
                errors.append("Query must not contain dynamic SQL construction")

        # Check for administrative operations
        if self.security_requirements["no_admin_operations"]:
            admin_operations = self._check_admin_operations(query)
            if admin_operations:
                errors.append(f"Query contains prohibited administrative operations: {admin_operations}")

        # Check for result set limits
        if self.security_requirements["limit_result_sets"]:
            if not self._has_result_limit(query):
                errors.append("Query should include result set limits (LIMIT clause)")

        return len(errors) == 0, errors

    def _is_parameterized(self, query: str, parameters: Optional[Dict]) -> bool:
        """Check if query uses parameterized statements."""
        # Look for parameter placeholders
        placeholder_patterns = [
            r"\?",           # ? placeholder
            r"%s",           # %s placeholder
            r":\w+",         # :named placeholder
            r"\$\d+",        # $1, $2, etc. placeholder
        ]

        has_placeholders = any(
            re.search(pattern, query) for pattern in placeholder_patterns
        )

        # If query has placeholders, parameters should be provided
        if has_placeholders:
            return parameters is not None and len(parameters) > 0

        # If no placeholders, check if query contains literal values that should be parameterized
        literal_patterns = [
            r"'[^']*'",      # String literals
            r'"[^"]*"',      # Quoted literals
        ]

        has_literals = any(
            re.search(pattern, query) for pattern in literal_patterns
        )

        # Queries with literals should use parameters instead
        return not has_literals

    def _contains_dynamic_sql(self, query: str) -> bool:
        """Check for dynamic SQL construction patterns."""
        dynamic_patterns = [
            r"concat\s*\(",
            r"\|\|",             # String concatenation
            r'\+\s*[\'"]',       # String concatenation with +
            r"exec\s*\(",
            r"execute\s*\(",
        ]

        return any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in dynamic_patterns
        )

    def _check_admin_operations(self, query: str) -> List[str]:
        """Check for administrative operations that should be prohibited."""
        admin_keywords = [
            "drop", "create", "alter", "truncate", "grant", "revoke",
            "backup", "restore", "dbcc", "bulk", "openrowset",
        ]

        query_lower = query.lower()
        found_operations = [
            keyword for keyword in admin_keywords
            if re.search(r"\b" + keyword + r"\b", query_lower)
        ]

        return found_operations

    def _has_result_limit(self, query: str) -> bool:
        """Check if query has appropriate result limiting."""
        # Check for LIMIT clause
        limit_patterns = [
            r"\blimit\s+\d+",
            r"\btop\s+\d+",
            r"\bfirst\s+\d+",
            r"\brownum\s*<=?\s*\d+",
        ]

        return any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in limit_patterns
        )


class SecurityAuditor:
    """Central security auditing system."""

    def __init__(self):
        self.injection_detector = SQLInjectionDetector()
        self.input_validator = InputValidator()
        self.query_analyzer = QueryAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Track security events
        self.security_events = []
        self.max_events = 1000

    def audit_user_input(self, user_query: str) -> Tuple[bool, List[SecurityViolation]]:
        """Comprehensive security audit of user input."""
        violations = []

        # Input validation
        is_valid, validation_errors = self.input_validator.validate_natural_language_query(user_query)
        if not is_valid:
            for error in validation_errors:
                violations.append(SecurityViolation(
                    violation_type="input_validation",
                    severity="medium",
                    description=error,
                    location="user_input",
                    timestamp=datetime.now(),
                ))

        # SQL injection detection
        injection_violations = self.injection_detector.analyze_query(user_query)
        violations.extend(injection_violations)

        # Log security events
        if violations:
            self._log_security_event("user_input_violation", {
                "query": user_query[:100] + "..." if len(user_query) > 100 else user_query,
                "violations": len(violations),
                "severity": max(v.severity for v in violations),
            })

        return len(violations) == 0, violations

    def audit_generated_query(self, sql_query: str, parameters: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """Audit generated SQL query for security compliance."""
        is_valid, errors = self.query_analyzer.validate_generated_query(sql_query, parameters)

        if not is_valid:
            self._log_security_event("generated_query_violation", {
                "query": sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
                "errors": errors,
                "parameters_provided": parameters is not None,
            })

        return is_valid, errors

    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "session_id": self._get_session_id(),
        }

        # Add to local cache
        self.security_events.append(event)
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]

        # Log for external monitoring
        self.logger.warning(f"Security event: {event_type}", extra=event)

    def _get_session_id(self) -> str:
        """Generate a session identifier for tracking."""
        # In a real implementation, this would come from the session management system
        return hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]

    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of recent security events."""
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]

        event_types = {}
        for event in recent_events:
            event_type = event["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_events_24h": len(recent_events),
            "event_types": event_types,
            "last_event": self.security_events[-1] if self.security_events else None,
            "summary_timestamp": datetime.now().isoformat(),
        }


# Global security auditor instance
security_auditor = SecurityAuditor()
