"""Advanced Security Framework with AI-Powered Threat Detection and Zero-Trust Architecture.

This module implements cutting-edge security capabilities:
- AI-powered threat detection and prevention
- Zero-trust security architecture
- Advanced behavioral analysis
- Real-time security monitoring
- Automated incident response
- Compliance and audit automation
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jwt
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    SQL_INJECTION_ATTEMPT = "sql_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    BRUTE_FORCE_ATTACK = "brute_force"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_QUERY = "suspicious_query"


@dataclass
class SecurityThreat:
    """Security threat representation."""
    threat_id: str
    threat_type: SecurityEventType
    threat_level: ThreatLevel
    confidence: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    indicators: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: str
    permissions: Set[str]
    ip_address: str
    user_agent: str
    authentication_method: str
    risk_score: float = 0.0
    last_activity: float = field(default_factory=time.time)
    behavioral_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessAttempt:
    """Access attempt record."""
    timestamp: float
    user_id: str
    resource: str
    action: str
    success: bool
    ip_address: str
    user_agent: str
    risk_factors: List[str] = field(default_factory=list)


class BehavioralAnalyzer:
    """AI-powered behavioral analysis for anomaly detection."""

    def __init__(self):
        self.user_profiles = {}
        self.baseline_behaviors = {}
        self.anomaly_threshold = 0.7
        self.learning_window = 86400  # 24 hours
        self.behavior_history = defaultdict(lambda: deque(maxlen=1000))

    def analyze_behavior(self, context: SecurityContext, action: str, 
                        query_metadata: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Analyze user behavior for anomalies.

        Args:
            context: Security context
            action: Action being performed
            query_metadata: Metadata about the query/action

        Returns:
            Tuple of (anomaly_score, risk_factors)
        """
        try:
            user_id = context.user_id
            risk_factors = []
            anomaly_scores = []

            # Initialize user profile if new
            if user_id not in self.user_profiles:
                self._initialize_user_profile(user_id, context)
                return 0.0, []  # No baseline for new users

            profile = self.user_profiles[user_id]
            
            # Temporal analysis
            temporal_score = self._analyze_temporal_patterns(context, profile)
            if temporal_score > self.anomaly_threshold:
                risk_factors.append(f"unusual_access_time")
            anomaly_scores.append(temporal_score)

            # Query complexity analysis
            complexity_score = self._analyze_query_complexity(query_metadata, profile)
            if complexity_score > self.anomaly_threshold:
                risk_factors.append("unusual_query_complexity")
            anomaly_scores.append(complexity_score)

            # Access pattern analysis
            pattern_score = self._analyze_access_patterns(context, action, profile)
            if pattern_score > self.anomaly_threshold:
                risk_factors.append("unusual_access_pattern")
            anomaly_scores.append(pattern_score)

            # Geographic analysis
            geo_score = self._analyze_geographic_patterns(context, profile)
            if geo_score > self.anomaly_threshold:
                risk_factors.append("unusual_geographic_location")
            anomaly_scores.append(geo_score)

            # Device/agent analysis
            device_score = self._analyze_device_patterns(context, profile)
            if device_score > self.anomaly_threshold:
                risk_factors.append("unusual_device_or_agent")
            anomaly_scores.append(device_score)

            # Update behavioral profile
            self._update_behavioral_profile(user_id, context, action, query_metadata)

            # Calculate overall anomaly score
            overall_score = np.mean(anomaly_scores) if anomaly_scores else 0.0

            return overall_score, risk_factors

        except Exception as e:
            logger.exception(f"Behavioral analysis failed: {e}")
            return 0.5, ["analysis_error"]

    def _initialize_user_profile(self, user_id: str, context: SecurityContext):
        """Initialize behavioral profile for new user."""
        self.user_profiles[user_id] = {
            'first_seen': time.time(),
            'typical_hours': set(),
            'typical_days': set(),
            'typical_ips': set(),
            'typical_agents': set(),
            'query_complexity_stats': {'mean': 0.0, 'std': 1.0, 'samples': []},
            'access_frequency': {'mean': 0.0, 'std': 1.0, 'samples': []},
            'session_duration_stats': {'mean': 0.0, 'std': 1.0, 'samples': []},
            'last_update': time.time(),
        }

    def _analyze_temporal_patterns(self, context: SecurityContext, profile: Dict[str, Any]) -> float:
        """Analyze temporal access patterns."""
        try:
            current_time = time.time()
            current_hour = int((current_time % 86400) // 3600)  # Hour of day
            current_day = int(current_time // 86400) % 7  # Day of week

            typical_hours = profile.get('typical_hours', set())
            typical_days = profile.get('typical_days', set())

            # Calculate anomaly based on typical patterns
            hour_anomaly = 0.0 if current_hour in typical_hours else 1.0
            day_anomaly = 0.0 if current_day in typical_days else 0.5

            # If we don't have enough data, be less strict
            if len(typical_hours) < 5:
                hour_anomaly *= 0.5
            if len(typical_days) < 3:
                day_anomaly *= 0.5

            return (hour_anomaly + day_anomaly) / 2.0

        except Exception as e:
            logger.warning(f"Temporal pattern analysis failed: {e}")
            return 0.0

    def _analyze_query_complexity(self, query_metadata: Dict[str, Any], profile: Dict[str, Any]) -> float:
        """Analyze query complexity patterns."""
        try:
            # Extract complexity metrics
            query_length = query_metadata.get('query_length', 0)
            table_count = query_metadata.get('table_count', 0)
            join_count = query_metadata.get('join_count', 0)
            
            # Simple complexity score
            complexity = query_length / 100.0 + table_count * 0.5 + join_count * 0.3

            complexity_stats = profile.get('query_complexity_stats', {'mean': 0.0, 'std': 1.0})
            
            if complexity_stats['std'] > 0:
                # Z-score based anomaly detection
                z_score = abs(complexity - complexity_stats['mean']) / complexity_stats['std']
                anomaly_score = min(z_score / 3.0, 1.0)  # Normalize to [0,1]
            else:
                anomaly_score = 0.0

            return anomaly_score

        except Exception as e:
            logger.warning(f"Query complexity analysis failed: {e}")
            return 0.0

    def _analyze_access_patterns(self, context: SecurityContext, action: str, profile: Dict[str, Any]) -> float:
        """Analyze access frequency patterns."""
        try:
            current_time = time.time()
            user_id = context.user_id
            
            # Get recent access history
            recent_accesses = [
                access for access in self.behavior_history[user_id]
                if current_time - access < 3600  # Last hour
            ]
            
            current_frequency = len(recent_accesses)
            
            frequency_stats = profile.get('access_frequency', {'mean': 0.0, 'std': 1.0})
            
            if frequency_stats['std'] > 0:
                z_score = abs(current_frequency - frequency_stats['mean']) / frequency_stats['std']
                anomaly_score = min(z_score / 2.0, 1.0)
            else:
                anomaly_score = 0.0

            return anomaly_score

        except Exception as e:
            logger.warning(f"Access pattern analysis failed: {e}")
            return 0.0

    def _analyze_geographic_patterns(self, context: SecurityContext, profile: Dict[str, Any]) -> float:
        """Analyze geographic access patterns (simplified IP-based)."""
        try:
            current_ip = context.ip_address
            typical_ips = profile.get('typical_ips', set())

            # Simple IP-based geographic analysis
            # In practice, would use GeoIP services
            ip_parts = current_ip.split('.')[:2]  # First two octets
            current_network = '.'.join(ip_parts)

            typical_networks = set()
            for ip in typical_ips:
                ip_parts = ip.split('.')[:2]
                typical_networks.add('.'.join(ip_parts))

            # Anomaly if from completely new network
            if current_network in typical_networks:
                return 0.0
            elif len(typical_networks) < 3:  # Not enough data
                return 0.3
            else:
                return 0.8

        except Exception as e:
            logger.warning(f"Geographic pattern analysis failed: {e}")
            return 0.0

    def _analyze_device_patterns(self, context: SecurityContext, profile: Dict[str, Any]) -> float:
        """Analyze device/user agent patterns."""
        try:
            current_agent = context.user_agent
            typical_agents = profile.get('typical_agents', set())

            # Simple user agent similarity
            if current_agent in typical_agents:
                return 0.0
            
            # Check for similar agents (simplified)
            for agent in typical_agents:
                # Simple similarity check
                if len(set(current_agent.split()) & set(agent.split())) > len(current_agent.split()) * 0.7:
                    return 0.2  # Similar but not exact

            # Completely new agent
            return 0.6 if len(typical_agents) > 2 else 0.3

        except Exception as e:
            logger.warning(f"Device pattern analysis failed: {e}")
            return 0.0

    def _update_behavioral_profile(self, user_id: str, context: SecurityContext, 
                                  action: str, query_metadata: Dict[str, Any]):
        """Update user's behavioral profile."""
        try:
            profile = self.user_profiles[user_id]
            current_time = time.time()

            # Update temporal patterns
            current_hour = int((current_time % 86400) // 3600)
            current_day = int(current_time // 86400) % 7
            
            profile['typical_hours'].add(current_hour)
            profile['typical_days'].add(current_day)

            # Limit set sizes
            if len(profile['typical_hours']) > 12:  # Max half the day
                profile['typical_hours'] = set(list(profile['typical_hours'])[-12:])

            # Update IP patterns
            profile['typical_ips'].add(context.ip_address)
            if len(profile['typical_ips']) > 10:
                profile['typical_ips'] = set(list(profile['typical_ips'])[-10:])

            # Update agent patterns
            profile['typical_agents'].add(context.user_agent)
            if len(profile['typical_agents']) > 5:
                profile['typical_agents'] = set(list(profile['typical_agents'])[-5:])

            # Update complexity statistics
            query_length = query_metadata.get('query_length', 0)
            table_count = query_metadata.get('table_count', 0)
            join_count = query_metadata.get('join_count', 0)
            complexity = query_length / 100.0 + table_count * 0.5 + join_count * 0.3

            complexity_samples = profile['query_complexity_stats']['samples']
            complexity_samples.append(complexity)
            if len(complexity_samples) > 100:
                complexity_samples = complexity_samples[-100:]
            
            if len(complexity_samples) > 1:
                profile['query_complexity_stats']['mean'] = np.mean(complexity_samples)
                profile['query_complexity_stats']['std'] = max(np.std(complexity_samples), 0.1)
            
            profile['query_complexity_stats']['samples'] = complexity_samples

            # Update access frequency
            self.behavior_history[user_id].append(current_time)

            # Update frequency statistics
            if len(self.behavior_history[user_id]) > 10:
                recent_intervals = []
                history = list(self.behavior_history[user_id])
                for i in range(1, len(history)):
                    interval = history[i] - history[i-1]
                    recent_intervals.append(interval)

                if recent_intervals:
                    frequency_samples = profile['access_frequency']['samples']
                    frequency_samples.extend(recent_intervals)
                    if len(frequency_samples) > 100:
                        frequency_samples = frequency_samples[-100:]
                    
                    profile['access_frequency']['mean'] = np.mean(frequency_samples)
                    profile['access_frequency']['std'] = max(np.std(frequency_samples), 60.0)  # Min 1 minute
                    profile['access_frequency']['samples'] = frequency_samples

            profile['last_update'] = current_time

        except Exception as e:
            logger.warning(f"Behavioral profile update failed: {e}")


class ThreatDetectionEngine:
    """AI-powered threat detection and analysis engine."""

    def __init__(self):
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.threat_patterns = self._load_threat_patterns()
        self.active_threats = {}
        self.threat_history = deque(maxlen=10000)
        self.ml_models = self._initialize_ml_models()

    def detect_threats(self, context: SecurityContext, query: str, 
                      query_metadata: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect security threats in user request.

        Args:
            context: Security context
            query: User query string
            query_metadata: Query metadata

        Returns:
            List of detected threats
        """
        threats = []

        try:
            # SQL Injection detection
            sql_threats = self._detect_sql_injection(query, context)
            threats.extend(sql_threats)

            # Behavioral analysis
            behavioral_threats = self._detect_behavioral_anomalies(context, query, query_metadata)
            threats.extend(behavioral_threats)

            # Pattern-based detection
            pattern_threats = self._detect_threat_patterns(query, context, query_metadata)
            threats.extend(pattern_threats)

            # Privilege escalation detection
            privilege_threats = self._detect_privilege_escalation(context, query_metadata)
            threats.extend(privilege_threats)

            # Data exfiltration detection
            exfiltration_threats = self._detect_data_exfiltration(query, context, query_metadata)
            threats.extend(exfiltration_threats)

            # Store detected threats
            for threat in threats:
                self.active_threats[threat.threat_id] = threat
                self.threat_history.append(threat)

            return threats

        except Exception as e:
            logger.exception(f"Threat detection failed: {e}")
            return []

    def _detect_sql_injection(self, query: str, context: SecurityContext) -> List[SecurityThreat]:
        """Detect SQL injection attempts."""
        threats = []
        
        try:
            suspicious_patterns = [
                r"(\bUNION\b|\bSELECT\b).*(\bFROM\b|\bWHERE\b).*(\b1\s*=\s*1\b|\b'\s*OR\s*')",
                r"(\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b|\bEXEC\b|\bEXECUTE\b)",
                r"(\b--\b|\/\*|\*\/)",
                r"(\bxp_\w+\b|\bsp_\w+\b)",
                r"(\bCAST\b|\bCONVERT\b).*(\bCHAR\b|\bVARCHAR\b).*(\bSELECT\b|\bUNION\b)",
            ]

            query_lower = query.lower()
            detected_patterns = []
            
            import re
            for pattern in suspicious_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected_patterns.append(pattern)

            if detected_patterns:
                confidence = min(len(detected_patterns) / 3.0, 1.0)  # Up to 3 patterns for full confidence
                threat_level = ThreatLevel.HIGH if confidence > 0.7 else ThreatLevel.MEDIUM

                threat = SecurityThreat(
                    threat_id=str(uuid.uuid4()),
                    threat_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
                    threat_level=threat_level,
                    confidence=confidence,
                    source_ip=context.ip_address,
                    user_id=context.user_id,
                    description=f"Potential SQL injection detected in query",
                    indicators=[f"Pattern: {pattern}" for pattern in detected_patterns],
                    mitigation_actions=["block_query", "alert_security_team", "increase_monitoring"],
                    metadata={
                        'query_preview': query[:100],
                        'patterns_matched': len(detected_patterns),
                        'session_id': context.session_id,
                    }
                )
                threats.append(threat)

        except Exception as e:
            logger.warning(f"SQL injection detection failed: {e}")

        return threats

    def _detect_behavioral_anomalies(self, context: SecurityContext, query: str, 
                                   query_metadata: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect behavioral anomalies."""
        threats = []

        try:
            anomaly_score, risk_factors = self.behavioral_analyzer.analyze_behavior(
                context, "query_execution", query_metadata
            )

            if anomaly_score > 0.7:  # High anomaly threshold
                threat_level = ThreatLevel.HIGH if anomaly_score > 0.9 else ThreatLevel.MEDIUM

                threat = SecurityThreat(
                    threat_id=str(uuid.uuid4()),
                    threat_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                    threat_level=threat_level,
                    confidence=anomaly_score,
                    source_ip=context.ip_address,
                    user_id=context.user_id,
                    description=f"Anomalous user behavior detected (score: {anomaly_score:.2f})",
                    indicators=risk_factors,
                    mitigation_actions=["increase_monitoring", "require_additional_auth", "alert_security_team"],
                    metadata={
                        'anomaly_score': anomaly_score,
                        'risk_factors': risk_factors,
                        'session_id': context.session_id,
                        'behavioral_context': query_metadata,
                    }
                )
                threats.append(threat)

        except Exception as e:
            logger.warning(f"Behavioral anomaly detection failed: {e}")

        return threats

    def _detect_threat_patterns(self, query: str, context: SecurityContext, 
                              query_metadata: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect known threat patterns."""
        threats = []

        try:
            # Check for suspicious query characteristics
            suspicious_indicators = []

            # Large result set requests
            if 'LIMIT' not in query.upper() and query_metadata.get('estimated_rows', 0) > 100000:
                suspicious_indicators.append("large_result_set_without_limit")

            # Unusual table access
            table_count = query_metadata.get('table_count', 0)
            if table_count > 10:
                suspicious_indicators.append("accessing_many_tables")

            # Time-based queries at unusual hours
            current_hour = int((time.time() % 86400) // 3600)
            if current_hour < 6 or current_hour > 22:  # Night hours
                if 'system' in query.lower() or 'admin' in query.lower():
                    suspicious_indicators.append("system_access_unusual_hours")

            # Rapid-fire queries
            user_id = context.user_id
            current_time = time.time()
            recent_queries = [
                t for t in self.threat_history 
                if t.user_id == user_id and current_time - t.timestamp < 60  # Last minute
            ]
            
            if len(recent_queries) > 20:  # More than 20 queries per minute
                suspicious_indicators.append("rapid_fire_queries")

            if suspicious_indicators:
                confidence = min(len(suspicious_indicators) / 4.0, 1.0)
                threat_level = ThreatLevel.MEDIUM if confidence > 0.5 else ThreatLevel.LOW

                threat = SecurityThreat(
                    threat_id=str(uuid.uuid4()),
                    threat_type=SecurityEventType.SUSPICIOUS_QUERY,
                    threat_level=threat_level,
                    confidence=confidence,
                    source_ip=context.ip_address,
                    user_id=context.user_id,
                    description=f"Suspicious query patterns detected",
                    indicators=suspicious_indicators,
                    mitigation_actions=["monitor_closely", "apply_rate_limiting"],
                    metadata={
                        'query_characteristics': query_metadata,
                        'session_id': context.session_id,
                    }
                )
                threats.append(threat)

        except Exception as e:
            logger.warning(f"Threat pattern detection failed: {e}")

        return threats

    def _detect_privilege_escalation(self, context: SecurityContext, 
                                   query_metadata: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect privilege escalation attempts."""
        threats = []

        try:
            # Check if user is trying to access resources beyond their permissions
            required_permissions = query_metadata.get('required_permissions', set())
            user_permissions = context.permissions

            unauthorized_permissions = required_permissions - user_permissions

            if unauthorized_permissions:
                confidence = min(len(unauthorized_permissions) / 5.0, 1.0)
                threat_level = ThreatLevel.HIGH if confidence > 0.6 else ThreatLevel.MEDIUM

                threat = SecurityThreat(
                    threat_id=str(uuid.uuid4()),
                    threat_type=SecurityEventType.PRIVILEGE_ESCALATION,
                    threat_level=threat_level,
                    confidence=confidence,
                    source_ip=context.ip_address,
                    user_id=context.user_id,
                    description=f"Privilege escalation attempt detected",
                    indicators=[f"Missing permission: {perm}" for perm in unauthorized_permissions],
                    mitigation_actions=["deny_access", "alert_security_team", "audit_user_permissions"],
                    metadata={
                        'unauthorized_permissions': list(unauthorized_permissions),
                        'user_permissions': list(user_permissions),
                        'session_id': context.session_id,
                    }
                )
                threats.append(threat)

        except Exception as e:
            logger.warning(f"Privilege escalation detection failed: {e}")

        return threats

    def _detect_data_exfiltration(self, query: str, context: SecurityContext, 
                                query_metadata: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect potential data exfiltration attempts."""
        threats = []

        try:
            exfiltration_indicators = []

            # Large data requests
            estimated_rows = query_metadata.get('estimated_rows', 0)
            if estimated_rows > 50000:
                exfiltration_indicators.append("large_data_request")

            # Sensitive table access
            tables_accessed = query_metadata.get('tables_accessed', [])
            sensitive_tables = {'users', 'accounts', 'payments', 'personal_info', 'credentials'}
            
            for table in tables_accessed:
                if any(sensitive in table.lower() for sensitive in sensitive_tables):
                    exfiltration_indicators.append(f"sensitive_table_access:{table}")

            # Unusual export patterns
            if any(keyword in query.lower() for keyword in ['export', 'dump', 'backup', 'copy']):
                exfiltration_indicators.append("export_keywords_detected")

            # Bulk data selection
            if '*' in query and 'WHERE' not in query.upper():
                exfiltration_indicators.append("bulk_select_without_filter")

            if exfiltration_indicators:
                confidence = min(len(exfiltration_indicators) / 3.0, 1.0)
                threat_level = ThreatLevel.CRITICAL if confidence > 0.8 else ThreatLevel.HIGH

                threat = SecurityThreat(
                    threat_id=str(uuid.uuid4()),
                    threat_type=SecurityEventType.DATA_EXFILTRATION,
                    threat_level=threat_level,
                    confidence=confidence,
                    source_ip=context.ip_address,
                    user_id=context.user_id,
                    description=f"Potential data exfiltration detected",
                    indicators=exfiltration_indicators,
                    mitigation_actions=["block_query", "alert_dpo", "forensic_analysis"],
                    metadata={
                        'estimated_data_size': estimated_rows,
                        'tables_accessed': tables_accessed,
                        'session_id': context.session_id,
                    }
                )
                threats.append(threat)

        except Exception as e:
            logger.warning(f"Data exfiltration detection failed: {e}")

        return threats

    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load known threat patterns."""
        # In practice, would load from database or configuration
        return {
            'sql_injection': {
                'patterns': [
                    'union select',
                    'drop table',
                    '1=1',
                    'exec xp_',
                    'sp_password',
                ],
                'severity': 'high',
            },
            'data_exfiltration': {
                'patterns': [
                    'select * from',
                    'limit 999999',
                    'order by rand()',
                ],
                'severity': 'high',
            },
        }

    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize ML models for threat detection."""
        # Placeholder for ML model initialization
        # In practice, would load trained models
        return {
            'anomaly_detector': None,
            'classifier': None,
            'clustering': None,
        }

    def get_threat_analytics(self) -> Dict[str, Any]:
        """Get comprehensive threat analytics."""
        try:
            current_time = time.time()
            
            # Recent threats (last 24 hours)
            recent_threats = [
                t for t in self.threat_history
                if current_time - t.timestamp < 86400
            ]

            # Threat statistics
            threat_by_type = defaultdict(int)
            threat_by_level = defaultdict(int)
            threat_by_user = defaultdict(int)

            for threat in recent_threats:
                threat_by_type[threat.threat_type.value] += 1
                threat_by_level[threat.threat_level.value] += 1
                if threat.user_id:
                    threat_by_user[threat.user_id] += 1

            # Active critical threats
            active_critical = [
                t for t in self.active_threats.values()
                if t.threat_level == ThreatLevel.CRITICAL and current_time - t.timestamp < 3600
            ]

            # Top risk users
            top_risk_users = sorted(threat_by_user.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                'summary': {
                    'total_threats_24h': len(recent_threats),
                    'active_critical_threats': len(active_critical),
                    'threat_detection_rate': len(recent_threats) / max(len(self.threat_history), 1),
                    'most_common_threat': max(threat_by_type.items(), key=lambda x: x[1])[0] if threat_by_type else 'none',
                },
                'threat_breakdown': {
                    'by_type': dict(threat_by_type),
                    'by_level': dict(threat_by_level),
                    'by_user': dict(threat_by_user),
                },
                'active_threats': len(self.active_threats),
                'top_risk_users': top_risk_users,
                'system_status': {
                    'detection_engine_health': 'healthy',
                    'behavioral_profiles': len(self.behavioral_analyzer.user_profiles),
                    'threat_patterns_loaded': len(self.threat_patterns),
                },
                'timestamp': current_time,
            }

        except Exception as e:
            logger.exception(f"Threat analytics failed: {e}")
            return {'error': str(e)}


class ZeroTrustSecurityController:
    """Zero-trust security architecture implementation."""

    def __init__(self):
        self.threat_detector = ThreatDetectionEngine()
        self.access_policies = {}
        self.session_manager = {}
        self.encryption_keys = {}
        self.audit_trail = deque(maxlen=100000)

    def authorize_request(self, context: SecurityContext, resource: str, 
                         action: str, query_metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Authorize request using zero-trust principles.

        Args:
            context: Security context
            resource: Resource being accessed
            action: Action being performed
            query_metadata: Query metadata

        Returns:
            Tuple of (authorized, reasons)
        """
        try:
            authorization_reasons = []
            
            # 1. Verify identity and session
            if not self._verify_identity(context):
                return False, ["invalid_identity"]

            # 2. Check session validity
            if not self._verify_session(context):
                return False, ["invalid_session"]

            # 3. Evaluate risk score
            risk_score = self._calculate_risk_score(context, query_metadata)
            if risk_score > 0.8:
                return False, ["high_risk_score"]
            elif risk_score > 0.6:
                authorization_reasons.append("elevated_risk_monitoring")

            # 4. Check explicit permissions
            if not self._check_permissions(context, resource, action):
                return False, ["insufficient_permissions"]

            # 5. Detect threats
            threats = self.threat_detector.detect_threats(context, query_metadata.get('query', ''), query_metadata)
            
            critical_threats = [t for t in threats if t.threat_level == ThreatLevel.CRITICAL]
            high_threats = [t for t in threats if t.threat_level == ThreatLevel.HIGH]

            if critical_threats:
                return False, ["critical_threat_detected"]
            elif high_threats:
                return False, ["high_threat_detected"]
            elif threats:
                authorization_reasons.append("threats_detected_monitoring")

            # 6. Apply dynamic policies
            policy_result = self._apply_dynamic_policies(context, resource, action, risk_score)
            if not policy_result['authorized']:
                return False, policy_result['reasons']
            
            authorization_reasons.extend(policy_result['conditions'])

            # 7. Record access attempt
            self._record_access_attempt(context, resource, action, True, authorization_reasons)

            return True, authorization_reasons

        except Exception as e:
            logger.exception(f"Authorization failed: {e}")
            self._record_access_attempt(context, resource, action, False, ["authorization_error"])
            return False, ["authorization_error"]

    def _verify_identity(self, context: SecurityContext) -> bool:
        """Verify user identity."""
        try:
            # Check if user_id is valid and not expired
            if not context.user_id or context.user_id == 'anonymous':
                return False

            # Check authentication method strength
            strong_auth_methods = {'mfa', 'certificate', 'biometric'}
            if context.authentication_method not in strong_auth_methods:
                # Allow but increase risk score
                context.risk_score += 0.2

            return True

        except Exception as e:
            logger.warning(f"Identity verification failed: {e}")
            return False

    def _verify_session(self, context: SecurityContext) -> bool:
        """Verify session validity."""
        try:
            session_id = context.session_id
            current_time = time.time()

            # Check if session exists
            if session_id not in self.session_manager:
                return False

            session_info = self.session_manager[session_id]
            
            # Check session expiration
            if current_time > session_info.get('expires_at', 0):
                del self.session_manager[session_id]
                return False

            # Check session activity
            if current_time - session_info.get('last_activity', 0) > 3600:  # 1 hour inactivity
                del self.session_manager[session_id]
                return False

            # Update last activity
            session_info['last_activity'] = current_time
            
            return True

        except Exception as e:
            logger.warning(f"Session verification failed: {e}")
            return False

    def _calculate_risk_score(self, context: SecurityContext, query_metadata: Dict[str, Any]) -> float:
        """Calculate dynamic risk score."""
        try:
            risk_factors = []
            base_risk = context.risk_score

            # Time-based risk
            current_hour = int((time.time() % 86400) // 3600)
            if current_hour < 6 or current_hour > 22:  # Night hours
                risk_factors.append(0.1)

            # IP-based risk
            if self._is_suspicious_ip(context.ip_address):
                risk_factors.append(0.3)

            # Query complexity risk
            complexity = query_metadata.get('complexity_score', 0.0)
            if complexity > 0.8:
                risk_factors.append(0.2)

            # User behavior risk
            behavioral_risk = context.behavioral_profile.get('anomaly_score', 0.0)
            risk_factors.append(behavioral_risk * 0.3)

            # Calculate total risk
            total_risk = base_risk + sum(risk_factors)
            return min(total_risk, 1.0)

        except Exception as e:
            logger.warning(f"Risk score calculation failed: {e}")
            return 0.5

    def _check_permissions(self, context: SecurityContext, resource: str, action: str) -> bool:
        """Check explicit permissions."""
        try:
            required_permission = f"{resource}:{action}"
            
            # Check direct permissions
            if required_permission in context.permissions:
                return True

            # Check wildcard permissions
            wildcard_permission = f"{resource}:*"
            if wildcard_permission in context.permissions:
                return True

            # Check admin permissions
            if "admin:*" in context.permissions:
                return True

            return False

        except Exception as e:
            logger.warning(f"Permission check failed: {e}")
            return False

    def _apply_dynamic_policies(self, context: SecurityContext, resource: str, 
                              action: str, risk_score: float) -> Dict[str, Any]:
        """Apply dynamic security policies."""
        try:
            conditions = []
            
            # High-risk policies
            if risk_score > 0.7:
                conditions.append("require_additional_verification")
                conditions.append("enhanced_logging")

            # Sensitive resource policies
            sensitive_resources = {'user_data', 'financial_data', 'pii'}
            if any(sensitive in resource.lower() for sensitive in sensitive_resources):
                conditions.append("sensitive_data_access_logged")
                
                # Require stronger authentication for sensitive data
                if context.authentication_method not in {'mfa', 'certificate'}:
                    return {
                        'authorized': False,
                        'reasons': ['insufficient_auth_for_sensitive_data'],
                        'conditions': []
                    }

            # Time-based policies
            current_hour = int((time.time() % 86400) // 3600)
            if current_hour < 6 or current_hour > 22:
                conditions.append("after_hours_access_monitoring")

            # Rate limiting policies
            if self._check_rate_limits(context.user_id, resource, action):
                return {
                    'authorized': False,
                    'reasons': ['rate_limit_exceeded'],
                    'conditions': []
                }

            return {
                'authorized': True,
                'reasons': [],
                'conditions': conditions
            }

        except Exception as e:
            logger.warning(f"Dynamic policy application failed: {e}")
            return {
                'authorized': False,
                'reasons': ['policy_evaluation_error'],
                'conditions': []
            }

    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious."""
        # Simplified implementation
        # In practice, would check against threat intelligence feeds
        suspicious_patterns = [
            '10.0.0.',  # Internal networks from external
            '192.168.',  # Private networks
            '127.',      # Localhost
        ]
        
        return any(ip_address.startswith(pattern) for pattern in suspicious_patterns)

    def _check_rate_limits(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has exceeded rate limits."""
        try:
            current_time = time.time()
            key = f"{user_id}:{resource}:{action}"
            
            # Get recent access attempts
            recent_attempts = [
                entry for entry in self.audit_trail
                if (entry.get('user_id') == user_id and 
                    entry.get('resource') == resource and 
                    entry.get('action') == action and
                    current_time - entry.get('timestamp', 0) < 3600)  # Last hour
            ]
            
            # Default rate limits
            rate_limits = {
                'query:execute': 1000,  # 1000 queries per hour
                'data:export': 10,      # 10 exports per hour
                'admin:manage': 100,    # 100 admin actions per hour
            }
            
            action_key = f"{resource}:{action}"
            limit = rate_limits.get(action_key, 500)  # Default limit
            
            return len(recent_attempts) >= limit

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            return False

    def _record_access_attempt(self, context: SecurityContext, resource: str, 
                             action: str, success: bool, reasons: List[str]):
        """Record access attempt in audit trail."""
        try:
            audit_entry = {
                'timestamp': time.time(),
                'user_id': context.user_id,
                'session_id': context.session_id,
                'ip_address': context.ip_address,
                'user_agent': context.user_agent,
                'resource': resource,
                'action': action,
                'success': success,
                'reasons': reasons,
                'risk_score': getattr(context, 'risk_score', 0.0),
            }
            
            self.audit_trail.append(audit_entry)

        except Exception as e:
            logger.warning(f"Audit trail recording failed: {e}")

    def create_secure_session(self, user_id: str, authentication_method: str, 
                            permissions: Set[str]) -> str:
        """Create secure session with encryption."""
        try:
            session_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Create session
            session_info = {
                'user_id': user_id,
                'created_at': current_time,
                'last_activity': current_time,
                'expires_at': current_time + 28800,  # 8 hours
                'authentication_method': authentication_method,
                'permissions': permissions,
                'ip_address': None,  # To be set on first request
                'security_level': self._calculate_security_level(authentication_method, permissions),
            }
            
            # Generate encryption key for session
            session_key = Fernet.generate_key()
            self.encryption_keys[session_id] = session_key
            
            self.session_manager[session_id] = session_info
            
            return session_id

        except Exception as e:
            logger.exception(f"Session creation failed: {e}")
            raise

    def _calculate_security_level(self, auth_method: str, permissions: Set[str]) -> str:
        """Calculate security level based on authentication and permissions."""
        # High security for admin permissions or strong auth
        if 'admin:*' in permissions or auth_method in {'mfa', 'certificate', 'biometric'}:
            return 'high'
        elif auth_method in {'password', 'token'}:
            return 'medium'
        else:
            return 'low'

    def get_security_analytics(self) -> Dict[str, Any]:
        """Get comprehensive security analytics."""
        try:
            current_time = time.time()
            
            # Get threat analytics
            threat_analytics = self.threat_detector.get_threat_analytics()
            
            # Session analytics
            active_sessions = len(self.session_manager)
            recent_logins = len([
                entry for entry in self.audit_trail
                if entry.get('action') == 'login' and current_time - entry.get('timestamp', 0) < 3600
            ])
            
            # Access analytics
            recent_accesses = [
                entry for entry in self.audit_trail
                if current_time - entry.get('timestamp', 0) < 86400
            ]
            
            success_rate = sum(1 for entry in recent_accesses if entry.get('success', False)) / max(len(recent_accesses), 1)
            
            # Risk analytics
            high_risk_sessions = sum(
                1 for session in self.session_manager.values()
                if session.get('security_level') == 'high'
            )
            
            return {
                'threat_detection': threat_analytics,
                'access_control': {
                    'active_sessions': active_sessions,
                    'recent_logins_1h': recent_logins,
                    'access_success_rate_24h': success_rate,
                    'total_access_attempts_24h': len(recent_accesses),
                },
                'risk_management': {
                    'high_security_sessions': high_risk_sessions,
                    'audit_trail_size': len(self.audit_trail),
                    'encryption_keys_managed': len(self.encryption_keys),
                },
                'system_health': {
                    'zero_trust_status': 'operational',
                    'policy_engine_status': 'healthy',
                    'threat_detector_status': 'active',
                },
                'timestamp': current_time,
            }

        except Exception as e:
            logger.exception(f"Security analytics failed: {e}")
            return {'error': str(e)}


# Global security controller
global_security_controller = ZeroTrustSecurityController()


# Utility functions
def create_security_context(user_id: str, session_id: str, permissions: Set[str],
                          ip_address: str, user_agent: str, auth_method: str) -> SecurityContext:
    """Create security context for request."""
    return SecurityContext(
        user_id=user_id,
        session_id=session_id,
        permissions=permissions,
        ip_address=ip_address,
        user_agent=user_agent,
        authentication_method=auth_method,
    )


def authorize_sql_request(context: SecurityContext, sql_query: str,
                         query_metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Authorize SQL request using zero-trust security.

    Args:
        context: Security context
        sql_query: SQL query string
        query_metadata: Query metadata

    Returns:
        Tuple of (authorized, reasons/conditions)
    """
    query_metadata['query'] = sql_query
    return global_security_controller.authorize_request(
        context, 'database', 'query', query_metadata
    )


def get_security_insights() -> Dict[str, Any]:
    """Get comprehensive security insights and analytics.

    Returns:
        Security analytics and insights
    """
    return global_security_controller.get_security_analytics()