"""Advanced input validation and sanitization for SQL synthesis agent.

This module provides comprehensive input validation, sanitization,
and security checks for user inputs and database queries.
"""

import contextlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .logging_config import get_logger
from .security import SQLInjectionDetector

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    violations: list[str]
    risk_level: str
    confidence: float


class InputValidator:
    """Comprehensive input validation and sanitization."""

    def __init__(self):
        self.sql_detector = SQLInjectionDetector()

        # Input constraints
        self.max_query_length = 10000
        self.max_word_length = 200
        self.max_words = 100

        # Allowed characters (expanded for international support)
        self.allowed_chars = re.compile(r'^[a-zA-Z0-9\s\.,\?\!\-\(\)\'\"@_\/:;%&=\+\[\]]+$')

        # Language detection patterns
        self.english_chars = re.compile(r"[a-zA-Z]")
        self.number_chars = re.compile(r"[0-9]")
        self.special_chars = re.compile(r"[^a-zA-Z0-9\s]")

        # Common business terms that should be allowed
        self.business_terms = {
            "sales", "revenue", "customers", "orders", "products", "users",
            "employees", "inventory", "departments", "marketing", "finance",
            "reports", "analytics", "dashboard", "metrics", "kpi", "roi",
            "quarter", "monthly", "yearly", "average", "total", "count",
            "sum", "maximum", "minimum", "growth", "trend", "forecast",
        }

        # Suspicious patterns (beyond SQL injection)
        self.suspicious_patterns = [
            r"\b(password|credit_card|ssn|social_security)\b",
            r"\b(admin|root|superuser|sysadmin)\b",
            r"\b(hack|exploit|vulnerability|backdoor)\b",
            r"<script[^>]*>.*?</script>",  # XSS patterns
            r"javascript:",
            r"data:text/html",
            r"\b(eval|exec|system|shell)\b",
            r"(\.\.\/){3,}",  # Path traversal
            r"\b(localhost|127\.0\.0\.1|0\.0\.0\.0)\b",
        ]

        self.compiled_suspicious = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.suspicious_patterns
        ]

    def validate_user_query(self, user_input: str) -> ValidationResult:
        """Comprehensive validation of user natural language query."""
        violations = []
        risk_level = "low"
        confidence = 1.0

        # Basic input validation
        if not user_input:
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                violations=["Empty input"],
                risk_level="low",
                confidence=1.0,
            )

        # Length validation
        if len(user_input) > self.max_query_length:
            violations.append(f"Input too long (max {self.max_query_length} characters)")
            risk_level = "medium"

        # Word count validation
        words = user_input.split()
        if len(words) > self.max_words:
            violations.append(f"Too many words (max {self.max_words})")
            risk_level = "medium"

        # Individual word length validation
        long_words = [w for w in words if len(w) > self.max_word_length]
        if long_words:
            violations.append(f"Words too long: {long_words[:3]}")
            risk_level = "medium"

        # Character validation
        if not self.allowed_chars.match(user_input):
            violations.append("Contains disallowed characters")
            risk_level = "medium"
            confidence *= 0.8

        # SQL injection detection
        sql_violations = self.sql_detector.analyze_query(user_input)
        if sql_violations:
            violations.extend([v.description for v in sql_violations])
            risk_level = "high"
            confidence *= 0.5

        # Suspicious pattern detection
        for pattern in self.compiled_suspicious:
            if pattern.search(user_input):
                violations.append(f"Suspicious pattern detected: {pattern.pattern}")
                risk_level = "high" if risk_level != "critical" else risk_level
                confidence *= 0.6

        # Language and content analysis
        content_analysis = self._analyze_content(user_input)
        if content_analysis["suspicious"]:
            violations.extend(content_analysis["issues"])
            risk_level = content_analysis["risk_level"]
            confidence *= content_analysis["confidence_factor"]

        # Sanitize input
        sanitized_input = self._sanitize_input(user_input)

        # Determine overall validity
        is_valid = (
            len(violations) == 0 or
            (risk_level in ["low", "medium"] and confidence > 0.7)
        )

        # Log validation result
        logger.log_security_event(
            event_type="input_validation",
            severity=risk_level,
            description=f"Input validation: {'passed' if is_valid else 'failed'}",
            user_input=user_input,
            violations=violations,
            confidence=confidence,
        )

        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_input,
            violations=violations,
            risk_level=risk_level,
            confidence=confidence,
        )

    def _analyze_content(self, text: str) -> dict[str, Any]:
        """Analyze content for suspicious patterns and context."""
        issues = []
        risk_level = "low"
        confidence_factor = 1.0

        # Normalize text for analysis
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        lower_text = normalized.lower()

        # Check for encoding attacks
        encoding_patterns = [
            r"%[0-9a-fA-F]{2}",  # URL encoding
            r"&#[0-9]+;",        # HTML entity encoding
            r"\\x[0-9a-fA-F]{2}", # Hex encoding
            r"\\u[0-9a-fA-F]{4}", # Unicode encoding
        ]

        for pattern in encoding_patterns:
            if re.search(pattern, text):
                issues.append(f"Encoded characters detected: {pattern}")
                risk_level = "medium"
                confidence_factor *= 0.8

        # Check character distribution
        char_analysis = self._analyze_character_distribution(text)
        if char_analysis["suspicious"]:
            issues.extend(char_analysis["issues"])
            risk_level = "medium"
            confidence_factor *= 0.9

        # Check for business context
        business_score = self._calculate_business_context_score(lower_text)
        if business_score < 0.3:
            issues.append("Low business context relevance")
            confidence_factor *= 0.9

        # Check for excessive repetition
        repetition_analysis = self._analyze_repetition(text)
        if repetition_analysis["excessive"]:
            issues.append("Excessive character/word repetition")
            risk_level = "medium"
            confidence_factor *= 0.8

        return {
            "suspicious": len(issues) > 0,
            "issues": issues,
            "risk_level": risk_level,
            "confidence_factor": confidence_factor,
            "business_score": business_score,
        }

    def _analyze_character_distribution(self, text: str) -> dict[str, Any]:
        """Analyze character distribution for anomalies."""
        issues = []

        total_chars = len(text)
        if total_chars == 0:
            return {"suspicious": False, "issues": []}

        english_count = len(self.english_chars.findall(text))
        number_count = len(self.number_chars.findall(text))
        special_count = len(self.special_chars.findall(text))

        # Calculate ratios
        english_ratio = english_count / total_chars
        number_ratio = number_count / total_chars
        special_ratio = special_count / total_chars

        # Check for suspicious distributions
        if english_ratio < 0.3:
            issues.append(f"Low English character ratio: {english_ratio:.2f}")

        if special_ratio > 0.5:
            issues.append(f"High special character ratio: {special_ratio:.2f}")

        if number_ratio > 0.7:
            issues.append(f"High number ratio: {number_ratio:.2f}")

        return {
            "suspicious": len(issues) > 0,
            "issues": issues,
            "ratios": {
                "english": english_ratio,
                "numbers": number_ratio,
                "special": special_ratio,
            },
        }

    def _calculate_business_context_score(self, text: str) -> float:
        """Calculate how relevant the text is to business/database queries."""
        words = set(re.findall(r"\b\w+\b", text.lower()))

        # Count business-relevant terms
        business_matches = len(words.intersection(self.business_terms))
        total_significant_words = len([w for w in words if len(w) > 3])

        if total_significant_words == 0:
            return 0.0

        return business_matches / total_significant_words

    def _analyze_repetition(self, text: str) -> dict[str, Any]:
        """Check for excessive repetition patterns."""
        # Check character repetition
        char_repetition = max(
            (len(list(group)) for char, group in
             __import__("itertools").groupby(text) if char.isalnum()),
            default=0,
        )

        # Check word repetition
        words = text.split()
        if len(words) > 1:
            word_counts = {}
            for word in words:
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1

            max_word_repetition = max(word_counts.values())
            repeated_words = [word for word, count in word_counts.items() if count > 3]
        else:
            max_word_repetition = 1
            repeated_words = []

        return {
            "excessive": char_repetition > 10 or max_word_repetition > 5,
            "char_repetition": char_repetition,
            "max_word_repetition": max_word_repetition,
            "repeated_words": repeated_words,
        }

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input while preserving legitimate content."""
        # Normalize Unicode
        sanitized = unicodedata.normalize("NFKD", text)

        # Remove null bytes and control characters
        sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char in "\t\n\r")

        # Decode common encoding attacks
        sanitized = self._decode_encoded_chars(sanitized)

        # Remove excessive whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        # Escape potentially dangerous characters while preserving legitimate ones
        return sanitized.replace("\\", "\\\\")


    def _decode_encoded_chars(self, text: str) -> str:
        """Safely decode encoded characters."""
        import html
        import urllib.parse

        # HTML entity decoding
        with contextlib.suppress(Exception):
            text = html.unescape(text)

        # URL decoding (only if it looks like URL encoding)
        if "%" in text and re.search(r"%[0-9a-fA-F]{2}", text):
            with contextlib.suppress(Exception):
                text = urllib.parse.unquote(text)

        return text

    def validate_sql_output(self, sql_query: str) -> ValidationResult:
        """Validate generated SQL query for safety."""
        violations = []
        risk_level = "low"

        if not sql_query:
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                violations=["Empty SQL query"],
                risk_level="low",
                confidence=1.0,
            )

        # Check for dangerous SQL operations
        dangerous_operations = [
            "DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE",
            "GRANT", "REVOKE", "INSERT", "UPDATE",
        ]

        sql_upper = sql_query.upper()
        for op in dangerous_operations:
            if f" {op} " in sql_upper or sql_upper.startswith(f"{op} "):
                violations.append(f"Dangerous SQL operation: {op}")
                risk_level = "high"

        # Check for file operations
        file_operations = ["INTO OUTFILE", "LOAD_FILE", "LOAD DATA"]
        for op in file_operations:
            if op in sql_upper:
                violations.append(f"File operation detected: {op}")
                risk_level = "critical"

        # Ensure it's a SELECT query (read-only)
        if not sql_upper.strip().startswith("SELECT") and not sql_upper.strip().startswith("WITH"):
            violations.append("Non-SELECT query detected")
            risk_level = "high"

        return ValidationResult(
            is_valid=len(violations) == 0 or risk_level in ["low", "medium"],
            sanitized_input=sql_query,
            violations=violations,
            risk_level=risk_level,
            confidence=1.0 if len(violations) == 0 else 0.5,
        )

    def validate_database_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate database connection parameters."""
        violations = []
        risk_level = "low"

        required_fields = ["db_type", "database_url"]
        for field in required_fields:
            if field not in params or not params[field]:
                violations.append(f"Missing required field: {field}")

        # Validate database URL format
        if "database_url" in params:
            url = params["database_url"]
            if not isinstance(url, str):
                violations.append("Database URL must be a string")
            elif len(url) > 500:
                violations.append("Database URL too long")
            elif any(dangerous in url.lower() for dangerous in ["file://", "javascript:", "data:"]):
                violations.append("Dangerous protocol in database URL")
                risk_level = "high"

        # Validate database type
        if "db_type" in params:
            allowed_types = {"postgresql", "mysql", "sqlite", "snowflake"}
            if params["db_type"] not in allowed_types:
                violations.append(f"Unsupported database type: {params['db_type']}")

        return ValidationResult(
            is_valid=len(violations) == 0,
            sanitized_input=str(params),
            violations=violations,
            risk_level=risk_level,
            confidence=1.0,
        )


# Global validator instance
global_validator = InputValidator()


def validate_user_input(user_input: str) -> ValidationResult:
    """Quick validation function for user input."""
    return global_validator.validate_user_query(user_input)


def validate_sql_query(sql_query: str) -> ValidationResult:
    """Quick validation function for SQL queries."""
    return global_validator.validate_sql_output(sql_query)
