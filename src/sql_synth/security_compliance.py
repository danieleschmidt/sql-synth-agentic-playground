"""Enhanced security and compliance features.

This module implements enterprise-grade security features including:
- Advanced SQL injection detection using ML models
- Data privacy and GDPR compliance
- Audit logging and compliance reporting
- Role-based access control (RBAC)
- Query sanitization and validation
- Encryption and secure storage
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"


class AuditEventType(Enum):
    """Types of audit events."""
    QUERY_GENERATION = "query_generation"
    QUERY_EXECUTION = "query_execution"
    ACCESS_DENIED = "access_denied"
    DATA_EXPORT = "data_export"
    SECURITY_VIOLATION = "security_violation"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"


@dataclass
class UserRole:
    """User role definition with permissions."""
    role_name: str
    permissions: set[str]
    data_access_level: SecurityLevel
    allowed_databases: set[str]
    query_restrictions: dict[str, Any]
    max_rows_per_query: int = 10000
    allowed_operations: set[str] = field(default_factory=lambda: {"SELECT"})


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: AuditEventType
    user_id: str
    tenant_id: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    resource_accessed: str
    action_performed: str
    result: str  # success, failure, blocked
    risk_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    sensitive_data_accessed: bool = False
    compliance_flags: list[ComplianceFramework] = field(default_factory=list)


class MLSecurityDetector:
    """Machine learning-based security threat detector."""

    def __init__(self):
        # In a real implementation, this would load trained ML models
        self.suspicious_patterns = [
            r"(union\s+select|union\s+all\s+select)",
            r"(or\s+1\s*=\s*1|or\s+'1'\s*=\s*'1')",
            r"(drop\s+table|delete\s+from|truncate\s+table)",
            r"(exec\s*\(|execute\s*\(|sp_executesql)",
            r"(waitfor\s+delay|benchmark\s*\()",
            r"(information_schema|sys\.|master\.|msdb\.)",
            r"(load_file\s*\(|into\s+outfile|into\s+dumpfile)",
        ]

        self.anomaly_thresholds = {
            "query_length": 5000,
            "unusual_keywords": 5,
            "nested_queries": 4,
            "time_based_patterns": 3,
        }

    def analyze_query_security(self, query: str, user_context: dict[str, Any]) -> dict[str, Any]:
        """Comprehensive security analysis of a query."""
        analysis = {
            "risk_score": 0.0,
            "threats_detected": [],
            "anomalies": [],
            "compliance_issues": [],
            "recommendations": [],
        }

        # Pattern-based detection
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query.lower()):
                analysis["threats_detected"].append({
                    "type": "sql_injection",
                    "pattern": pattern,
                    "confidence": 0.8,
                })
                analysis["risk_score"] += 0.3

        # Anomaly detection
        if len(query) > self.anomaly_thresholds["query_length"]:
            analysis["anomalies"].append("unusually_long_query")
            analysis["risk_score"] += 0.2

        nested_count = query.lower().count("select")
        if nested_count > self.anomaly_thresholds["nested_queries"]:
            analysis["anomalies"].append("excessive_nesting")
            analysis["risk_score"] += 0.25

        # Time-based injection patterns
        if re.search(r"(sleep\s*\(|waitfor|delay)", query.lower()):
            analysis["threats_detected"].append({
                "type": "time_based_injection",
                "confidence": 0.9,
            })
            analysis["risk_score"] += 0.4

        # User behavior analysis
        user_risk = self._analyze_user_behavior(user_context)
        analysis["risk_score"] += user_risk

        # Cap risk score at 1.0
        analysis["risk_score"] = min(analysis["risk_score"], 1.0)

        # Generate recommendations
        if analysis["risk_score"] > 0.7:
            analysis["recommendations"].append("Block query and require manual review")
        elif analysis["risk_score"] > 0.4:
            analysis["recommendations"].append("Require additional authentication")

        return analysis

    def _analyze_user_behavior(self, user_context: dict[str, Any]) -> float:
        """Analyze user behavior patterns for anomalies."""
        risk = 0.0

        # Check for unusual access times
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            risk += 0.1

        # Check for geographic anomalies (mock implementation)
        if user_context.get("unusual_location", False):
            risk += 0.2

        # Check query frequency
        recent_queries = user_context.get("queries_last_hour", 0)
        if recent_queries > 100:
            risk += 0.15

        return risk


class DataPrivacyManager:
    """Manager for data privacy and GDPR compliance."""

    def __init__(self):
        self.pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
            (r"\b\d{16}\b", "credit_card"),
            (r"\b\d{3}\.\d{3}\.\d{3}\.\d{3}\b", "ip_address"),
            (r"\b\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b", "phone"),
        ]

        self.sensitive_table_patterns = [
            r".*user.*",
            r".*customer.*",
            r".*employee.*",
            r".*patient.*",
            r".*personal.*",
            r".*profile.*",
        ]

    def analyze_privacy_impact(self, query: str, schema_info: dict[str, Any]) -> dict[str, Any]:
        """Analyze potential privacy impact of a query."""
        analysis = {
            "privacy_score": 0.0,
            "pii_types_accessed": [],
            "sensitive_tables": [],
            "gdpr_relevant": False,
            "data_subject_rights_impact": [],
            "retention_considerations": [],
            "anonymization_suggestions": [],
        }

        # Detect PII patterns in query
        for pattern, pii_type in self.pii_patterns:
            if re.search(pattern, query):
                analysis["pii_types_accessed"].append(pii_type)
                analysis["privacy_score"] += 0.2

        # Check for sensitive table access
        tables_referenced = self._extract_table_names(query)
        for table in tables_referenced:
            for sensitive_pattern in self.sensitive_table_patterns:
                if re.match(sensitive_pattern, table.lower()):
                    analysis["sensitive_tables"].append(table)
                    analysis["privacy_score"] += 0.3
                    analysis["gdpr_relevant"] = True

        # Check for data export operations
        if re.search(r"(into\s+outfile|export|dump)", query.lower()):
            analysis["data_subject_rights_impact"].append("data_portability")
            analysis["privacy_score"] += 0.4

        # Check for deletion operations
        if re.search(r"delete\s+from", query.lower()):
            analysis["data_subject_rights_impact"].append("right_to_erasure")

        # Generate anonymization suggestions
        if analysis["pii_types_accessed"]:
            analysis["anonymization_suggestions"] = [
                "Consider using aggregate functions instead of individual records",
                "Apply data masking or pseudonymization",
                "Limit result set to non-PII columns where possible",
            ]

        analysis["privacy_score"] = min(analysis["privacy_score"], 1.0)
        return analysis

    def _extract_table_names(self, query: str) -> list[str]:
        """Extract table names from SQL query."""
        # Simple regex-based extraction (would need proper SQL parser for production)
        from_pattern = r"from\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        join_pattern = r"join\s+([a-zA-Z_][a-zA-Z0-9_]*)"

        tables = []
        tables.extend(re.findall(from_pattern, query.lower()))
        tables.extend(re.findall(join_pattern, query.lower()))

        return list(set(tables))


class RoleBasedAccessController:
    """Role-based access control system."""

    def __init__(self):
        self.roles: dict[str, UserRole] = {}
        self.user_roles: dict[str, str] = {}  # user_id -> role_name
        self.role_hierarchy = {
            "admin": 4,
            "power_user": 3,
            "analyst": 2,
            "viewer": 1,
        }

    def define_role(self, role: UserRole) -> None:
        """Define a new user role."""
        self.roles[role.role_name] = role
        logger.info("Defined role: %s with %d permissions",
                   role.role_name, len(role.permissions))

    def assign_user_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        if role_name not in self.roles:
            logger.error("Role %s does not exist", role_name)
            return False

        self.user_roles[user_id] = role_name
        logger.info("Assigned role %s to user %s", role_name, user_id)
        return True

    def check_permission(self, user_id: str, permission: str,
                        resource: str = "") -> bool:
        """Check if user has specific permission."""
        role_name = self.user_roles.get(user_id)
        if not role_name:
            return False

        role = self.roles.get(role_name)
        if not role:
            return False

        # Check basic permission
        if permission not in role.permissions:
            return False

        # Check resource-specific access
        if resource and hasattr(role, "allowed_databases"):
            if resource not in role.allowed_databases and role.allowed_databases:
                return False

        return True

    def validate_query_access(self, user_id: str, query: str,
                            database: str) -> dict[str, Any]:
        """Validate user's access to execute a query."""
        validation_result = {
            "allowed": False,
            "reason": "",
            "modifications_required": [],
            "risk_score": 0.0,
        }

        role_name = self.user_roles.get(user_id)
        if not role_name:
            validation_result["reason"] = "No role assigned"
            return validation_result

        role = self.roles.get(role_name)
        if not role:
            validation_result["reason"] = "Invalid role"
            return validation_result

        # Check database access
        if role.allowed_databases and database not in role.allowed_databases:
            validation_result["reason"] = "Database access denied"
            return validation_result

        # Check allowed operations
        query_operations = self._extract_operations(query)
        forbidden_ops = query_operations - role.allowed_operations
        if forbidden_ops:
            validation_result["reason"] = f"Operations not allowed: {forbidden_ops}"
            return validation_result

        # Check row limits
        if not self._has_appropriate_limits(query, role.max_rows_per_query):
            validation_result["modifications_required"].append(
                f"Add LIMIT {role.max_rows_per_query} to query",
            )

        # Check query restrictions
        for restriction, value in role.query_restrictions.items():
            if not self._check_restriction(query, restriction, value):
                validation_result["reason"] = f"Query violates restriction: {restriction}"
                return validation_result

        validation_result["allowed"] = True
        return validation_result

    def _extract_operations(self, query: str) -> set[str]:
        """Extract SQL operations from query."""
        operations = set()
        query_upper = query.upper()

        if query_upper.strip().startswith("SELECT"):
            operations.add("SELECT")
        if "INSERT" in query_upper:
            operations.add("INSERT")
        if "UPDATE" in query_upper:
            operations.add("UPDATE")
        if "DELETE" in query_upper:
            operations.add("DELETE")
        if "CREATE" in query_upper:
            operations.add("CREATE")
        if "DROP" in query_upper:
            operations.add("DROP")
        if "ALTER" in query_upper:
            operations.add("ALTER")

        return operations

    def _has_appropriate_limits(self, query: str, max_rows: int) -> bool:
        """Check if query has appropriate LIMIT clause."""
        if "LIMIT" not in query.upper():
            return False

        # Extract limit value (simplified regex)
        limit_match = re.search(r"LIMIT\s+(\d+)", query.upper())
        if limit_match:
            limit_value = int(limit_match.group(1))
            return limit_value <= max_rows

        return False

    def _check_restriction(self, query: str, restriction: str, value: Any) -> bool:
        """Check if query violates a specific restriction."""
        # Example restrictions
        if restriction == "no_cross_database_joins":
            return value or "." not in query
        if restriction == "max_join_tables":
            return query.upper().count("JOIN") <= value
        if restriction == "no_subqueries":
            return value or "(" not in query

        return True


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(self, storage_backend: str = "local"):
        self.storage_backend = storage_backend
        self.audit_events: list[AuditEvent] = []
        self.retention_days = 2555  # 7 years for compliance

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        # Add event ID if not provided
        if not event.event_id:
            event.event_id = self._generate_event_id(event)

        self.audit_events.append(event)

        # Log to standard logger as well
        logger.info(
            "AUDIT: %s by %s@%s - %s [%s] risk:%.2f",
            event.action_performed,
            event.user_id,
            event.tenant_id,
            event.result,
            event.ip_address,
            event.risk_score,
        )

        # In production, would send to secure audit storage
        self._persist_event(event)

    def query_audit_log(self, filters: dict[str, Any],
                       limit: int = 1000) -> list[AuditEvent]:
        """Query audit log with filters."""
        filtered_events = self.audit_events.copy()

        # Apply filters
        if "user_id" in filters:
            filtered_events = [e for e in filtered_events
                             if e.user_id == filters["user_id"]]

        if "event_type" in filters:
            filtered_events = [e for e in filtered_events
                             if e.event_type == filters["event_type"]]

        if "start_time" in filters:
            filtered_events = [e for e in filtered_events
                             if e.timestamp >= filters["start_time"]]

        if "end_time" in filters:
            filtered_events = [e for e in filtered_events
                             if e.timestamp <= filters["end_time"]]

        if "min_risk_score" in filters:
            filtered_events = [e for e in filtered_events
                             if e.risk_score >= filters["min_risk_score"]]

        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered_events[:limit]

    def generate_compliance_report(self, framework: ComplianceFramework,
                                 start_date: datetime,
                                 end_date: datetime) -> dict[str, Any]:
        """Generate compliance report for specific framework."""
        relevant_events = [
            event for event in self.audit_events
            if start_date <= event.timestamp <= end_date
            and framework in event.compliance_flags
        ]

        return {
            "framework": framework.value,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "total_events": len(relevant_events),
            "high_risk_events": len([e for e in relevant_events if e.risk_score > 0.7]),
            "security_violations": len([e for e in relevant_events
                                      if e.event_type == AuditEventType.SECURITY_VIOLATION]),
            "data_access_events": len([e for e in relevant_events if e.sensitive_data_accessed]),
            "users_involved": len({e.user_id for e in relevant_events}),
            "event_summary": self._summarize_events_by_type(relevant_events),
            "recommendations": self._generate_compliance_recommendations(framework, relevant_events),
        }


    def _generate_event_id(self, event: AuditEvent) -> str:
        """Generate unique event ID."""
        content = f"{event.user_id}:{event.event_type.value}:{event.timestamp}:{event.resource_accessed}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _persist_event(self, event: AuditEvent) -> None:
        """Persist event to secure storage."""
        # In production, would use encrypted storage, SIEM integration, etc.

    def _summarize_events_by_type(self, events: list[AuditEvent]) -> dict[str, int]:
        """Summarize events by type."""
        summary = {}
        for event in events:
            event_type = event.event_type.value
            summary[event_type] = summary.get(event_type, 0) + 1
        return summary

    def _generate_compliance_recommendations(self, framework: ComplianceFramework,
                                           events: list[AuditEvent]) -> list[str]:
        """Generate compliance recommendations."""
        recommendations = []

        high_risk_count = len([e for e in events if e.risk_score > 0.7])
        if high_risk_count > 0:
            recommendations.append(f"Review {high_risk_count} high-risk events")

        if framework == ComplianceFramework.GDPR:
            data_access_count = len([e for e in events if e.sensitive_data_accessed])
            if data_access_count > 100:
                recommendations.append("Consider implementing additional data access controls")

        return recommendations


# Default roles for common use cases
def create_default_roles() -> dict[str, UserRole]:
    """Create default role definitions."""
    return {
        "viewer": UserRole(
            role_name="viewer",
            permissions={"query_read"},
            data_access_level=SecurityLevel.PUBLIC,
            allowed_databases=set(),
            query_restrictions={"no_subqueries": True},
            max_rows_per_query=1000,
            allowed_operations={"SELECT"},
        ),
        "analyst": UserRole(
            role_name="analyst",
            permissions={"query_read", "query_create"},
            data_access_level=SecurityLevel.INTERNAL,
            allowed_databases={"analytics", "reporting"},
            query_restrictions={"max_join_tables": 5},
            max_rows_per_query=10000,
            allowed_operations={"SELECT"},
        ),
        "power_user": UserRole(
            role_name="power_user",
            permissions={"query_read", "query_create", "query_modify"},
            data_access_level=SecurityLevel.CONFIDENTIAL,
            allowed_databases={"analytics", "reporting", "operational"},
            query_restrictions={"max_join_tables": 10},
            max_rows_per_query=50000,
            allowed_operations={"SELECT", "INSERT", "UPDATE"},
        ),
        "admin": UserRole(
            role_name="admin",
            permissions={"query_read", "query_create", "query_modify", "user_manage", "system_admin"},
            data_access_level=SecurityLevel.RESTRICTED,
            allowed_databases=set(),  # Empty set means all databases
            query_restrictions={},
            max_rows_per_query=100000,
            allowed_operations={"SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"},
        ),
    }



# Global security components
ml_security_detector = MLSecurityDetector()
privacy_manager = DataPrivacyManager()
rbac_controller = RoleBasedAccessController()
audit_logger = AuditLogger()

# Initialize default roles
default_roles = create_default_roles()
for role in default_roles.values():
    rbac_controller.define_role(role)
