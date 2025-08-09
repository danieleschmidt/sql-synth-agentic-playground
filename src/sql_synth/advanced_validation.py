"""Advanced validation system for SQL synthesis operations.

This module provides comprehensive input/output validation, semantic analysis,
and quality assurance for SQL generation and execution.
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlparse
from sqlparse import sql, tokens

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic" 
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS_LOGIC = "business_logic"
    DATA_QUALITY = "data_quality"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    issue_id: str
    severity: ValidationSeverity
    validation_type: ValidationType
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    is_valid: bool
    issues: List[ValidationIssue]
    validation_score: float  # 0.0 to 1.0
    execution_time: float
    metadata: Dict[str, Any]
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues that prevent execution."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
    
    @property
    def error_issues(self) -> List[ValidationIssue]:
        """Get error issues that should block execution."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    @property
    def warning_issues(self) -> List[ValidationIssue]:
        """Get warning issues that should be noted."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]


class NaturalLanguageValidator:
    """Advanced validation for natural language queries."""
    
    SUSPICIOUS_PATTERNS = [
        # SQL injection attempts in natural language
        r";\s*(drop|delete|insert|update|create|alter)",
        r"union\s+select",
        r"--.*$",
        r"/\*.*\*/",
        r"xp_\w+",
        r"sp_\w+",
        
        # Potentially malicious requests
        r"show\s+all\s+(passwords|secrets|keys)",
        r"bypass\s+(security|authentication)",
        r"ignore\s+(permissions|access\s+control)",
    ]
    
    COMPLEXITY_INDICATORS = [
        r"\b(join|inner\s+join|left\s+join|right\s+join|outer\s+join)\b",
        r"\b(group\s+by|having|order\s+by)\b", 
        r"\b(subquery|nested)\b",
        r"\b(window\s+function|over\s+partition)\b",
        r"\b(case\s+when|if\s+then\s+else)\b",
        r"\b(with\s+recursive|cte)\b",
    ]
    
    def __init__(self) -> None:
        self.compiled_suspicious = [re.compile(pattern, re.IGNORECASE) for pattern in self.SUSPICIOUS_PATTERNS]
        self.compiled_complexity = [re.compile(pattern, re.IGNORECASE) for pattern in self.COMPLEXITY_INDICATORS]
    
    def validate(self, natural_query: str) -> ValidationResult:
        """Validate natural language query input.
        
        Args:
            natural_query: User's natural language query
            
        Returns:
            Validation result with issues and score
        """
        start_time = time.time()
        issues = []
        
        # Basic input validation
        issues.extend(self._validate_basic_input(natural_query))
        
        # Security validation
        issues.extend(self._validate_security(natural_query))
        
        # Complexity analysis
        issues.extend(self._analyze_complexity(natural_query))
        
        # Business logic validation
        issues.extend(self._validate_business_logic(natural_query))
        
        execution_time = time.time() - start_time
        
        # Calculate validation score
        score = self._calculate_validation_score(issues)
        
        # Determine if valid (no critical or error issues)
        is_valid = not any(
            issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
            for issue in issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            validation_score=score,
            execution_time=execution_time,
            metadata={
                "query_length": len(natural_query),
                "word_count": len(natural_query.split()),
                "estimated_complexity": self._estimate_complexity(natural_query)
            }
        )
    
    def _validate_basic_input(self, query: str) -> List[ValidationIssue]:
        """Validate basic input requirements."""
        issues = []
        
        # Empty or whitespace-only query
        if not query or not query.strip():
            issues.append(ValidationIssue(
                issue_id="empty_query",
                severity=ValidationSeverity.CRITICAL,
                validation_type=ValidationType.SYNTAX,
                message="Query cannot be empty",
                suggestion="Provide a descriptive natural language query"
            ))
        
        # Query too short
        elif len(query.strip()) < 5:
            issues.append(ValidationIssue(
                issue_id="query_too_short",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.SYNTAX,
                message="Query is too short to be meaningful",
                suggestion="Provide more specific details about what data you want"
            ))
        
        # Query too long
        elif len(query) > 1000:
            issues.append(ValidationIssue(
                issue_id="query_too_long",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.SYNTAX,
                message=f"Query is very long ({len(query)} characters)",
                suggestion="Consider breaking down complex requests into simpler parts"
            ))
        
        # Non-ASCII characters that might indicate encoding issues
        if not query.isascii():
            non_ascii_chars = [char for char in query if not char.isascii()]
            issues.append(ValidationIssue(
                issue_id="non_ascii_characters",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.SYNTAX,
                message=f"Query contains non-ASCII characters: {set(non_ascii_chars)}",
                suggestion="Use standard ASCII characters for better compatibility"
            ))
        
        return issues
    
    def _validate_security(self, query: str) -> List[ValidationIssue]:
        """Validate query for security concerns."""
        issues = []
        
        # Check for suspicious patterns
        for pattern in self.compiled_suspicious:
            if pattern.search(query):
                issues.append(ValidationIssue(
                    issue_id="suspicious_pattern",
                    severity=ValidationSeverity.CRITICAL,
                    validation_type=ValidationType.SECURITY,
                    message=f"Potentially malicious pattern detected: {pattern.pattern}",
                    suggestion="Rephrase your query using natural language without SQL syntax"
                ))
        
        # Check for potential social engineering
        social_engineering_terms = [
            "show passwords", "reveal secrets", "bypass security",
            "admin access", "root privileges", "ignore permissions"
        ]
        
        query_lower = query.lower()
        for term in social_engineering_terms:
            if term in query_lower:
                issues.append(ValidationIssue(
                    issue_id="social_engineering",
                    severity=ValidationSeverity.CRITICAL,
                    validation_type=ValidationType.SECURITY,
                    message=f"Potential security violation: '{term}'",
                    suggestion="This type of request is not allowed for security reasons"
                ))
        
        return issues
    
    def _analyze_complexity(self, query: str) -> List[ValidationIssue]:
        """Analyze query complexity and provide guidance."""
        issues = []
        
        complexity_score = 0
        detected_features = []
        
        for pattern in self.compiled_complexity:
            if pattern.search(query):
                complexity_score += 1
                detected_features.append(pattern.pattern)
        
        if complexity_score >= 4:
            issues.append(ValidationIssue(
                issue_id="high_complexity",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.PERFORMANCE,
                message=f"High complexity query detected (score: {complexity_score})",
                suggestion="Complex queries may take longer to generate and execute",
                metadata={"complexity_score": complexity_score, "features": detected_features}
            ))
        
        # Check for potentially expensive operations
        expensive_patterns = [
            r"\b(all|every)\s+\w+\s+in\s+\w+",  # Cartesian product indicators
            r"\b(maximum|minimum|average|sum)\s+of\s+all\b",  # Full table scans
            r"\b(compare|match)\s+\w+\s+with\s+\w+",  # Potential joins
        ]
        
        for pattern in expensive_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(ValidationIssue(
                    issue_id="potentially_expensive",
                    severity=ValidationSeverity.WARNING,
                    validation_type=ValidationType.PERFORMANCE,
                    message=f"Query may result in expensive database operations",
                    suggestion="Consider adding filters or limits to reduce data processing"
                ))
                break
        
        return issues
    
    def _validate_business_logic(self, query: str) -> List[ValidationIssue]:
        """Validate business logic aspects of the query."""
        issues = []
        
        # Check for unrealistic date ranges
        if re.search(r"\b(last\s+\d+\s+years?)\b", query, re.IGNORECASE):
            years_match = re.search(r"last\s+(\d+)\s+years?", query, re.IGNORECASE)
            if years_match:
                years = int(years_match.group(1))
                if years > 10:
                    issues.append(ValidationIssue(
                        issue_id="unrealistic_date_range",
                        severity=ValidationSeverity.WARNING,
                        validation_type=ValidationType.BUSINESS_LOGIC,
                        message=f"Very large date range: {years} years",
                        suggestion="Consider if such a large date range is necessary"
                    ))
        
        # Check for vague requirements
        vague_terms = ["all", "everything", "anything", "lots", "many", "some"]
        query_words = query.lower().split()
        
        vague_count = sum(1 for term in vague_terms if term in query_words)
        if vague_count >= 2:
            issues.append(ValidationIssue(
                issue_id="vague_requirements",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.BUSINESS_LOGIC,
                message="Query contains vague terms that may lead to imprecise results",
                suggestion="Be more specific about what data you need"
            ))
        
        return issues
    
    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity level."""
        complexity_score = sum(1 for pattern in self.compiled_complexity if pattern.search(query))
        
        if complexity_score == 0:
            return "simple"
        elif complexity_score <= 2:
            return "moderate"
        elif complexity_score <= 4:
            return "complex"
        else:
            return "very_complex"
    
    def _calculate_validation_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall validation score."""
        if not issues:
            return 1.0
        
        # Weight penalties by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 0.5,
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.INFO: 0.05
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 0.1) for issue in issues)
        score = max(0.0, 1.0 - total_penalty)
        
        return score


class SQLValidator:
    """Advanced validation for generated SQL queries."""
    
    DANGEROUS_KEYWORDS = [
        "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "TRUNCATE",
        "EXEC", "EXECUTE", "CALL", "MERGE", "REPLACE", "LOAD", "IMPORT"
    ]
    
    ALLOWED_FUNCTIONS = [
        "COUNT", "SUM", "AVG", "MIN", "MAX", "SUBSTRING", "CONCAT",
        "UPPER", "LOWER", "TRIM", "COALESCE", "CASE", "CAST", "CONVERT",
        "DATE", "DATETIME", "YEAR", "MONTH", "DAY", "NOW", "CURRENT_DATE"
    ]
    
    def __init__(self) -> None:
        pass
    
    def validate(self, sql_query: str, schema_info: Optional[Dict] = None) -> ValidationResult:
        """Comprehensive SQL validation.
        
        Args:
            sql_query: Generated SQL query to validate
            schema_info: Optional database schema information
            
        Returns:
            Validation result with detailed analysis
        """
        start_time = time.time()
        issues = []
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql_query)[0]
        except Exception as e:
            issues.append(ValidationIssue(
                issue_id="parse_error",
                severity=ValidationSeverity.CRITICAL,
                validation_type=ValidationType.SYNTAX,
                message=f"SQL parsing failed: {e}",
                suggestion="Generated SQL has syntax errors"
            ))
            
            return ValidationResult(
                is_valid=False,
                issues=issues,
                validation_score=0.0,
                execution_time=time.time() - start_time,
                metadata={"parse_error": True}
            )
        
        # Perform validation checks
        issues.extend(self._validate_syntax(parsed, sql_query))
        issues.extend(self._validate_security(parsed, sql_query))
        issues.extend(self._validate_performance(parsed, sql_query))
        
        if schema_info:
            issues.extend(self._validate_schema_compliance(parsed, sql_query, schema_info))
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        score = self._calculate_sql_score(issues, sql_query)
        is_valid = not any(
            issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
            for issue in issues
        )
        
        # Analyze query characteristics
        characteristics = self._analyze_query_characteristics(parsed, sql_query)
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            validation_score=score,
            execution_time=execution_time,
            metadata={
                "query_length": len(sql_query),
                "characteristics": characteristics,
                "estimated_rows": self._estimate_result_size(parsed, sql_query)
            }
        )
    
    def _validate_syntax(self, parsed: sql.Statement, sql_query: str) -> List[ValidationIssue]:
        """Validate SQL syntax and structure."""
        issues = []
        
        # Check for basic SQL structure
        if not sql_query.strip().upper().startswith(('SELECT', 'WITH', 'EXPLAIN', 'DESCRIBE', 'SHOW')):
            issues.append(ValidationIssue(
                issue_id="invalid_statement_type",
                severity=ValidationSeverity.CRITICAL,
                validation_type=ValidationType.SYNTAX,
                message="Only SELECT, WITH, EXPLAIN, DESCRIBE, and SHOW statements are allowed",
                suggestion="Modify the query to use only read operations"
            ))
        
        # Check for dangerous keywords
        sql_upper = sql_query.upper()
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                issues.append(ValidationIssue(
                    issue_id="dangerous_keyword",
                    severity=ValidationSeverity.CRITICAL,
                    validation_type=ValidationType.SECURITY,
                    message=f"Dangerous keyword detected: {keyword}",
                    suggestion="Only read-only operations are permitted"
                ))
        
        # Check for missing semicolon
        if not sql_query.strip().endswith(';'):
            issues.append(ValidationIssue(
                issue_id="missing_semicolon",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.SYNTAX,
                message="SQL query should end with semicolon",
                suggestion="Add ';' at the end of the query"
            ))
        
        return issues
    
    def _validate_security(self, parsed: sql.Statement, sql_query: str) -> List[ValidationIssue]:
        """Validate SQL security aspects."""
        issues = []
        
        # Check for potential SQL injection patterns
        injection_patterns = [
            r"'[^']*'[^']*'",  # String concatenation
            r"--[^\r\n]*",     # SQL comments
            r"/\*.*?\*/",      # Block comments
            r";\s*\w+",        # Multiple statements
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql_query, re.DOTALL):
                issues.append(ValidationIssue(
                    issue_id="injection_pattern",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.SECURITY,
                    message=f"Potential SQL injection pattern: {pattern}",
                    suggestion="Use parameterized queries instead"
                ))
        
        # Check for dynamic SQL construction
        if re.search(r"CONCAT\s*\(.*SELECT", sql_query, re.IGNORECASE):
            issues.append(ValidationIssue(
                issue_id="dynamic_sql",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.SECURITY,
                message="Dynamic SQL construction detected",
                suggestion="Avoid building SQL strings dynamically"
            ))
        
        return issues
    
    def _validate_performance(self, parsed: sql.Statement, sql_query: str) -> List[ValidationIssue]:
        """Validate SQL performance implications."""
        issues = []
        
        sql_upper = sql_query.upper()
        
        # Check for SELECT *
        if "SELECT *" in sql_upper:
            issues.append(ValidationIssue(
                issue_id="select_star",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.PERFORMANCE,
                message="SELECT * may return unnecessary columns",
                suggestion="Specify only the columns you need"
            ))
        
        # Check for missing LIMIT clause
        if "LIMIT" not in sql_upper and "TOP" not in sql_upper:
            issues.append(ValidationIssue(
                issue_id="missing_limit",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.PERFORMANCE,
                message="Query lacks LIMIT clause",
                suggestion="Consider adding LIMIT to control result size"
            ))
        
        # Check for Cartesian products
        if sql_upper.count("FROM") > 1 and "JOIN" not in sql_upper and "WHERE" not in sql_upper:
            issues.append(ValidationIssue(
                issue_id="potential_cartesian_product",
                severity=ValidationSeverity.ERROR,
                validation_type=ValidationType.PERFORMANCE,
                message="Potential Cartesian product detected",
                suggestion="Add appropriate JOIN conditions or WHERE clauses"
            ))
        
        # Check for complex nested queries
        nesting_level = sql_query.count('(') - sql_query.count(')')
        if abs(nesting_level) > 3:
            issues.append(ValidationIssue(
                issue_id="deep_nesting",
                severity=ValidationSeverity.WARNING,
                validation_type=ValidationType.PERFORMANCE,
                message=f"Deeply nested query (level: {abs(nesting_level)})",
                suggestion="Consider simplifying the query structure"
            ))
        
        return issues
    
    def _validate_schema_compliance(
        self,
        parsed: sql.Statement,
        sql_query: str,
        schema_info: Dict
    ) -> List[ValidationIssue]:
        """Validate query against database schema."""
        issues = []
        
        # Extract table and column references
        table_refs = self._extract_table_references(sql_query)
        column_refs = self._extract_column_references(sql_query)
        
        # Validate table existence
        available_tables = schema_info.get("tables", [])
        for table in table_refs:
            if table not in available_tables:
                issues.append(ValidationIssue(
                    issue_id="unknown_table",
                    severity=ValidationSeverity.ERROR,
                    validation_type=ValidationType.SEMANTIC,
                    message=f"Table '{table}' does not exist",
                    suggestion=f"Available tables: {', '.join(available_tables)}"
                ))
        
        return issues
    
    def _extract_table_references(self, sql_query: str) -> Set[str]:
        """Extract table names from SQL query."""
        # Simplified table extraction - in production, use proper SQL parsing
        table_pattern = r"\bFROM\s+(\w+)"
        join_pattern = r"\bJOIN\s+(\w+)"
        
        tables = set()
        
        for match in re.finditer(table_pattern, sql_query, re.IGNORECASE):
            tables.add(match.group(1).lower())
        
        for match in re.finditer(join_pattern, sql_query, re.IGNORECASE):
            tables.add(match.group(1).lower())
        
        return tables
    
    def _extract_column_references(self, sql_query: str) -> Set[str]:
        """Extract column names from SQL query."""
        # Simplified column extraction
        # In production, would use proper SQL AST parsing
        columns = set()
        
        # Extract from SELECT clause
        select_match = re.search(r"SELECT\s+(.*?)\s+FROM", sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Basic column extraction (excludes functions, aliases, etc.)
            column_pattern = r"\b([a-zA-Z_]\w*)\b"
            for match in re.finditer(column_pattern, select_clause):
                column = match.group(1).lower()
                if column not in ['select', 'from', 'where', 'and', 'or', 'as']:
                    columns.add(column)
        
        return columns
    
    def _analyze_query_characteristics(self, parsed: sql.Statement, sql_query: str) -> Dict[str, Any]:
        """Analyze query characteristics for insights."""
        sql_upper = sql_query.upper()
        
        characteristics = {
            "has_joins": "JOIN" in sql_upper,
            "has_subqueries": sql_query.count('(') > 0,
            "has_aggregations": any(func in sql_upper for func in ["COUNT", "SUM", "AVG", "MIN", "MAX"]),
            "has_grouping": "GROUP BY" in sql_upper,
            "has_ordering": "ORDER BY" in sql_upper,
            "has_filtering": "WHERE" in sql_upper,
            "has_having": "HAVING" in sql_upper,
            "join_count": sql_upper.count("JOIN"),
            "subquery_count": max(0, sql_query.count('(') - sql_query.count(')')),
            "estimated_complexity": "high" if sql_upper.count("JOIN") > 2 or sql_query.count('(') > 3 else "medium" if "JOIN" in sql_upper or sql_query.count('(') > 0 else "low"
        }
        
        return characteristics
    
    def _estimate_result_size(self, parsed: sql.Statement, sql_query: str) -> str:
        """Estimate result set size category."""
        sql_upper = sql_query.upper()
        
        # Look for LIMIT clause
        limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
        if limit_match:
            limit = int(limit_match.group(1))
            if limit <= 10:
                return "small"
            elif limit <= 100:
                return "medium"
            elif limit <= 1000:
                return "large"
            else:
                return "very_large"
        
        # Estimate based on query characteristics
        if "GROUP BY" in sql_upper or "DISTINCT" in sql_upper:
            return "medium"
        elif "JOIN" in sql_upper:
            return "large"
        else:
            return "unknown"
    
    def _calculate_sql_score(self, issues: List[ValidationIssue], sql_query: str) -> float:
        """Calculate SQL quality score."""
        base_score = 1.0
        
        # Penalty weights
        severity_weights = {
            ValidationSeverity.CRITICAL: 0.5,
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.INFO: 0.05
        }
        
        # Calculate penalties
        total_penalty = sum(severity_weights.get(issue.severity, 0.1) for issue in issues)
        
        # Bonus for good practices
        bonus = 0.0
        sql_upper = sql_query.upper()
        
        if "LIMIT" in sql_upper:
            bonus += 0.1
        if not "SELECT *" in sql_upper and "SELECT" in sql_upper:
            bonus += 0.05
        if sql_query.strip().endswith(';'):
            bonus += 0.05
        
        final_score = max(0.0, min(1.0, base_score - total_penalty + bonus))
        return final_score


class ComprehensiveValidator:
    """Unified validation system combining all validation types."""
    
    def __init__(self) -> None:
        self.nl_validator = NaturalLanguageValidator()
        self.sql_validator = SQLValidator()
    
    def validate_natural_language(self, query: str) -> ValidationResult:
        """Validate natural language input."""
        return self.nl_validator.validate(query)
    
    def validate_sql(self, sql_query: str, schema_info: Optional[Dict] = None) -> ValidationResult:
        """Validate generated SQL."""
        return self.sql_validator.validate(sql_query, schema_info)
    
    def validate_end_to_end(
        self,
        natural_query: str,
        generated_sql: str,
        schema_info: Optional[Dict] = None
    ) -> Tuple[ValidationResult, ValidationResult]:
        """Perform end-to-end validation of the complete pipeline.
        
        Args:
            natural_query: Original natural language query
            generated_sql: Generated SQL query
            schema_info: Optional database schema information
            
        Returns:
            Tuple of (natural language result, SQL result)
        """
        nl_result = self.validate_natural_language(natural_query)
        sql_result = self.validate_sql(generated_sql, schema_info)
        
        return nl_result, sql_result


# Global validator instance
global_validator = ComprehensiveValidator()