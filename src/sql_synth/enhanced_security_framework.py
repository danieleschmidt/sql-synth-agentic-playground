"""
Enhanced Security Framework - Generation 1 Implementation
Advanced multi-layer security validation for SQL synthesis with AI-powered threat detection.
"""

import re
import logging
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of SQL injection attacks."""
    UNION_BASED = "union_based"
    BOOLEAN_BASED = "boolean_based"
    TIME_BASED = "time_based"
    ERROR_BASED = "error_based"
    STACKED_QUERIES = "stacked_queries"
    SECOND_ORDER = "second_order"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_type: AttackType
    threat_level: ThreatLevel
    confidence: float
    description: str
    payload: str
    mitigation: str
    evidence: List[str]


@dataclass 
class SecurityAssessment:
    """Complete security assessment result."""
    is_safe: bool
    overall_risk: ThreatLevel
    confidence_score: float
    threats: List[SecurityThreat]
    security_score: float
    recommendations: List[str]
    analysis_time: float
    metadata: Dict[str, Any]


class EnhancedSQLSecurityValidator:
    """
    Advanced SQL security validator with AI-powered threat detection,
    behavioral analysis, and adaptive learning capabilities.
    """
    
    def __init__(self):
        self.threat_patterns = self._initialize_threat_patterns()
        self.ml_detector = self._initialize_ml_detector()
        self.threat_history = []
        self.adaptive_thresholds = self._initialize_thresholds()
        
        # Performance tracking
        self.validation_count = 0
        self.threat_detection_count = 0
        self.false_positive_rate = 0.0
        
        logger.info("Enhanced SQL Security Validator initialized")
    
    def _initialize_threat_patterns(self) -> Dict[AttackType, List[Dict[str, Any]]]:
        """Initialize comprehensive threat detection patterns."""
        return {
            AttackType.UNION_BASED: [
                {
                    'pattern': re.compile(r'union\s+(?:all\s+)?select', re.IGNORECASE),
                    'severity': 'high',
                    'description': 'UNION-based SQL injection attempt'
                },
                {
                    'pattern': re.compile(r'union\s+.*\s+select\s+.*from', re.IGNORECASE),
                    'severity': 'critical',
                    'description': 'Advanced UNION-based injection with data extraction'
                }
            ],
            AttackType.BOOLEAN_BASED: [
                {
                    'pattern': re.compile(r"(?:\d+\s*=\s*\d+|\'\w*\'\s*=\s*\'\w*\')", re.IGNORECASE),
                    'severity': 'medium',
                    'description': 'Boolean-based injection pattern'
                },
                {
                    'pattern': re.compile(r"(?:and|or)\s+\d+\s*[=<>]\s*\d+", re.IGNORECASE),
                    'severity': 'high',
                    'description': 'Boolean logic manipulation'
                }
            ],
            AttackType.TIME_BASED: [
                {
                    'pattern': re.compile(r'waitfor\s+delay\s+\'\d+:\d+:\d+\'', re.IGNORECASE),
                    'severity': 'high',
                    'description': 'Time-based blind SQL injection (SQL Server)'
                },
                {
                    'pattern': re.compile(r'sleep\s*\(\s*\d+\s*\)', re.IGNORECASE),
                    'severity': 'high',
                    'description': 'Time-based blind SQL injection (MySQL)'
                },
                {
                    'pattern': re.compile(r'pg_sleep\s*\(\s*\d+\s*\)', re.IGNORECASE),
                    'severity': 'high',
                    'description': 'Time-based blind SQL injection (PostgreSQL)'
                }
            ],
            AttackType.ERROR_BASED: [
                {
                    'pattern': re.compile(r'convert\s*\(\s*int\s*,\s*\w+\s*\)', re.IGNORECASE),
                    'severity': 'medium',
                    'description': 'Error-based injection using type conversion'
                },
                {
                    'pattern': re.compile(r'cast\s*\(\s*\w+\s+as\s+int\s*\)', re.IGNORECASE),
                    'severity': 'medium',
                    'description': 'Error-based injection using CAST'
                }
            ],
            AttackType.STACKED_QUERIES: [
                {
                    'pattern': re.compile(r';\s*(?:drop|create|alter|insert|update|delete)', re.IGNORECASE),
                    'severity': 'critical',
                    'description': 'Stacked query injection with destructive commands'
                },
                {
                    'pattern': re.compile(r';\s*exec\s*\(', re.IGNORECASE),
                    'severity': 'critical',
                    'description': 'Stacked query with dynamic SQL execution'
                }
            ],
            AttackType.PRIVILEGE_ESCALATION: [
                {
                    'pattern': re.compile(r'(?:xp_cmdshell|sp_oacreate|sp_oamethod)', re.IGNORECASE),
                    'severity': 'critical',
                    'description': 'System command execution attempt'
                },
                {
                    'pattern': re.compile(r'(?:grant|revoke)\s+.*\s+(?:to|from)', re.IGNORECASE),
                    'severity': 'high',
                    'description': 'Privilege escalation attempt'
                }
            ]
        }
    
    def _initialize_ml_detector(self) -> IsolationForest:
        """Initialize machine learning anomaly detector."""
        return IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
    
    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize adaptive security thresholds."""
        return {
            'anomaly_score_threshold': -0.1,
            'pattern_confidence_threshold': 0.7,
            'behavioral_anomaly_threshold': 0.8,
            'entropy_threshold': 4.0,
            'suspicious_keyword_ratio': 0.3
        }
    
    async def validate_sql_security(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityAssessment:
        """
        Perform comprehensive security validation of SQL query.
        
        Args:
            sql_query: SQL query to validate
            context: Additional context for validation
            
        Returns:
            Comprehensive security assessment
        """
        start_time = time.time()
        
        try:
            # Phase 1: Pattern-based threat detection
            pattern_threats = self._detect_pattern_threats(sql_query)
            
            # Phase 2: ML-based anomaly detection
            anomaly_score = self._detect_ml_anomalies(sql_query)
            
            # Phase 3: Behavioral analysis
            behavioral_threats = self._analyze_behavioral_patterns(sql_query, context)
            
            # Phase 4: Entropy and complexity analysis
            entropy_analysis = self._analyze_query_entropy(sql_query)
            
            # Phase 5: Combine assessments
            all_threats = pattern_threats + behavioral_threats
            
            # Calculate overall risk and confidence
            overall_risk = self._calculate_overall_risk(all_threats, anomaly_score, entropy_analysis)
            confidence_score = self._calculate_confidence_score(all_threats, anomaly_score)
            security_score = self._calculate_security_score(overall_risk, confidence_score)
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(all_threats, entropy_analysis)
            
            # Create assessment
            assessment = SecurityAssessment(
                is_safe=overall_risk in [ThreatLevel.MINIMAL, ThreatLevel.LOW],
                overall_risk=overall_risk,
                confidence_score=confidence_score,
                threats=all_threats,
                security_score=security_score,
                recommendations=recommendations,
                analysis_time=time.time() - start_time,
                metadata={
                    'anomaly_score': anomaly_score,
                    'entropy_analysis': entropy_analysis,
                    'pattern_matches': len(pattern_threats),
                    'behavioral_anomalies': len(behavioral_threats),
                    'query_length': len(sql_query),
                    'validation_timestamp': time.time()
                }
            )
            
            # Update metrics and learning
            self._update_validation_metrics(assessment)
            await self._adaptive_learning_update(sql_query, assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return SecurityAssessment(
                is_safe=False,
                overall_risk=ThreatLevel.CRITICAL,
                confidence_score=0.0,
                threats=[],
                security_score=0.0,
                recommendations=["Validation failed - manual review required"],
                analysis_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _detect_pattern_threats(self, sql_query: str) -> List[SecurityThreat]:
        """Detect threats using pattern matching."""
        threats = []
        
        for attack_type, patterns in self.threat_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                matches = pattern.finditer(sql_query)
                
                for match in matches:
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(
                        match, pattern_info, sql_query
                    )
                    
                    if confidence >= self.adaptive_thresholds['pattern_confidence_threshold']:
                        threat_level = self._severity_to_threat_level(pattern_info['severity'])
                        
                        threat = SecurityThreat(
                            threat_type=attack_type,
                            threat_level=threat_level,
                            confidence=confidence,
                            description=pattern_info['description'],
                            payload=match.group(),
                            mitigation=self._get_mitigation_advice(attack_type),
                            evidence=[f"Matched pattern at position {match.start()}-{match.end()}"]
                        )
                        threats.append(threat)
        
        return threats
    
    def _detect_ml_anomalies(self, sql_query: str) -> float:
        """Detect anomalies using machine learning."""
        try:
            # Create feature vector from SQL query
            features = self._extract_ml_features(sql_query)
            
            # Reshape for sklearn
            features_array = np.array(features).reshape(1, -1)
            
            # Get anomaly score
            anomaly_score = self.ml_detector.decision_function(features_array)[0]
            
            return anomaly_score
            
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
            return 0.0
    
    def _analyze_behavioral_patterns(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[SecurityThreat]:
        """Analyze behavioral patterns for threats."""
        threats = []
        
        # Check for suspicious behavioral patterns
        behavioral_checks = [
            self._check_query_complexity_anomaly,
            self._check_unusual_keyword_combinations,
            self._check_encoding_anomalies,
            self._check_temporal_patterns,
            self._check_context_inconsistencies
        ]
        
        for check in behavioral_checks:
            try:
                threat = check(sql_query, context)
                if threat:
                    threats.append(threat)
            except Exception as e:
                logger.warning(f"Behavioral check failed: {e}")
        
        return threats
    
    def _analyze_query_entropy(self, sql_query: str) -> Dict[str, float]:
        """Analyze query entropy and complexity metrics."""
        try:
            # Calculate Shannon entropy
            shannon_entropy = self._calculate_shannon_entropy(sql_query)
            
            # Calculate keyword density
            keyword_density = self._calculate_keyword_density(sql_query)
            
            # Calculate character distribution entropy
            char_entropy = self._calculate_character_entropy(sql_query)
            
            # Calculate structural complexity
            structural_complexity = self._calculate_structural_complexity(sql_query)
            
            return {
                'shannon_entropy': shannon_entropy,
                'keyword_density': keyword_density,
                'character_entropy': char_entropy,
                'structural_complexity': structural_complexity,
                'length_normalized_entropy': shannon_entropy / max(1, len(sql_query))
            }
            
        except Exception as e:
            logger.warning(f"Entropy analysis failed: {e}")
            return {}
    
    def _extract_ml_features(self, sql_query: str) -> List[float]:
        """Extract features for ML anomaly detection."""
        features = []
        
        # Length features
        features.append(len(sql_query))
        features.append(len(sql_query.split()))
        
        # Character frequency features
        for char in ['\'', '"', ';', '-', '(', ')', '=', '<', '>']:
            features.append(sql_query.count(char))
        
        # Keyword features
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN',
            'UNION', 'ORDER', 'GROUP', 'HAVING', 'DROP', 'CREATE', 'ALTER'
        ]
        for keyword in sql_keywords:
            features.append(sql_query.upper().count(keyword))
        
        # Structural features
        features.append(sql_query.count('('))  # Nested queries
        features.append(sql_query.count(','))  # Column count
        features.append(len(re.findall(r'\b\w+\b', sql_query)))  # Word count
        
        # Entropy feature
        features.append(self._calculate_shannon_entropy(sql_query))
        
        return features
    
    def _calculate_pattern_confidence(
        self,
        match: re.Match,
        pattern_info: Dict[str, Any],
        full_query: str
    ) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = 0.7
        
        # Adjust for pattern specificity
        if len(match.group()) > 10:
            base_confidence += 0.1
        
        # Adjust for context
        context_words = full_query[max(0, match.start()-20):match.end()+20]
        if any(word in context_words.lower() for word in ['admin', 'password', 'user']):
            base_confidence += 0.15
        
        # Adjust for severity
        severity_boost = {
            'low': 0.0,
            'medium': 0.05,
            'high': 0.1,
            'critical': 0.2
        }
        base_confidence += severity_boost.get(pattern_info['severity'], 0.0)
        
        return min(1.0, base_confidence)
    
    def _severity_to_threat_level(self, severity: str) -> ThreatLevel:
        """Convert severity string to ThreatLevel enum."""
        mapping = {
            'low': ThreatLevel.LOW,
            'medium': ThreatLevel.MEDIUM,
            'high': ThreatLevel.HIGH,
            'critical': ThreatLevel.CRITICAL
        }
        return mapping.get(severity, ThreatLevel.MEDIUM)
    
    def _get_mitigation_advice(self, attack_type: AttackType) -> str:
        """Get mitigation advice for specific attack type."""
        advice = {
            AttackType.UNION_BASED: "Use parameterized queries and validate input",
            AttackType.BOOLEAN_BASED: "Implement proper input validation and use prepared statements",
            AttackType.TIME_BASED: "Disable error messages and use query timeouts",
            AttackType.ERROR_BASED: "Suppress database errors from user interface",
            AttackType.STACKED_QUERIES: "Use parameterized queries and restrict database permissions",
            AttackType.PRIVILEGE_ESCALATION: "Apply principle of least privilege and disable dangerous functions"
        }
        return advice.get(attack_type, "Use secure coding practices")
    
    def _check_query_complexity_anomaly(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[SecurityThreat]:
        """Check for query complexity anomalies."""
        complexity_score = (
            len(sql_query) * 0.01 +
            sql_query.count('(') * 0.1 +
            sql_query.count('UNION') * 0.5 +
            sql_query.count('SELECT') * 0.2
        )
        
        if complexity_score > 10.0:  # Threshold for anomaly
            return SecurityThreat(
                threat_type=AttackType.UNION_BASED,  # Most common in complex queries
                threat_level=ThreatLevel.MEDIUM,
                confidence=min(0.9, complexity_score / 15.0),
                description="Unusually complex query structure detected",
                payload=sql_query[:100] + "..." if len(sql_query) > 100 else sql_query,
                mitigation="Review query complexity and validate business logic",
                evidence=[f"Complexity score: {complexity_score:.2f}"]
            )
        
        return None
    
    def _check_unusual_keyword_combinations(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[SecurityThreat]:
        """Check for unusual keyword combinations."""
        suspicious_combinations = [
            (['UNION', 'SELECT'], 'UNION-based injection pattern'),
            (['DROP', 'TABLE'], 'Destructive command combination'),
            (['EXEC', 'MASTER'], 'Privilege escalation pattern'),
            (['WAITFOR', 'DELAY'], 'Time-based attack pattern')
        ]
        
        query_upper = sql_query.upper()
        
        for keywords, description in suspicious_combinations:
            if all(keyword in query_upper for keyword in keywords):
                return SecurityThreat(
                    threat_type=AttackType.STACKED_QUERIES,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.85,
                    description=f"Suspicious keyword combination: {description}",
                    payload=" ".join(keywords),
                    mitigation="Validate query intent and use parameterized statements",
                    evidence=[f"Found keywords: {', '.join(keywords)}"]
                )
        
        return None
    
    def _check_encoding_anomalies(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[SecurityThreat]:
        """Check for encoding-based obfuscation."""
        # Check for hex encoding
        hex_pattern = re.compile(r'0x[0-9a-fA-F]+', re.IGNORECASE)
        hex_matches = hex_pattern.findall(sql_query)
        
        if len(hex_matches) > 2:  # Threshold for suspicious hex usage
            return SecurityThreat(
                threat_type=AttackType.ERROR_BASED,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.75,
                description="Multiple hex-encoded values detected",
                payload=", ".join(hex_matches[:3]),
                mitigation="Validate hex values and check for obfuscation",
                evidence=[f"Found {len(hex_matches)} hex values"]
            )
        
        return None
    
    def _check_temporal_patterns(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[SecurityThreat]:
        """Check for time-based attack patterns."""
        time_functions = ['WAITFOR', 'SLEEP', 'PG_SLEEP', 'BENCHMARK']
        query_upper = sql_query.upper()
        
        found_functions = [func for func in time_functions if func in query_upper]
        
        if found_functions:
            return SecurityThreat(
                threat_type=AttackType.TIME_BASED,
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                description="Time-based attack functions detected",
                payload=", ".join(found_functions),
                mitigation="Remove time functions and implement query timeouts",
                evidence=[f"Time functions: {', '.join(found_functions)}"]
            )
        
        return None
    
    def _check_context_inconsistencies(
        self,
        sql_query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[SecurityThreat]:
        """Check for context inconsistencies that might indicate attacks."""
        if not context:
            return None
        
        # Check if query complexity matches expected user level
        expected_complexity = context.get('expected_complexity', 'low')
        actual_complexity = self._assess_query_complexity(sql_query)
        
        if expected_complexity == 'low' and actual_complexity == 'high':
            return SecurityThreat(
                threat_type=AttackType.SECOND_ORDER,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.6,
                description="Query complexity inconsistent with user context",
                payload=f"Expected: {expected_complexity}, Actual: {actual_complexity}",
                mitigation="Verify user authorization for complex queries",
                evidence=["Context-complexity mismatch detected"]
            )
        
        return None
    
    def _calculate_shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Get character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_keyword_density(self, sql_query: str) -> float:
        """Calculate SQL keyword density."""
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN',
            'UNION', 'ORDER', 'GROUP', 'HAVING', 'DROP', 'CREATE', 'ALTER',
            'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE', 'BETWEEN'
        ]
        
        words = sql_query.upper().split()
        if not words:
            return 0.0
        
        keyword_count = sum(1 for word in words if word in sql_keywords)
        return keyword_count / len(words)
    
    def _calculate_character_entropy(self, sql_query: str) -> float:
        """Calculate character-level entropy."""
        if not sql_query:
            return 0.0
        
        # Focus on special characters that are often used in attacks
        special_chars = "';-()=<>*%"
        special_count = sum(sql_query.count(char) for char in special_chars)
        
        if len(sql_query) == 0:
            return 0.0
        
        return special_count / len(sql_query)
    
    def _calculate_structural_complexity(self, sql_query: str) -> float:
        """Calculate structural complexity score."""
        complexity_factors = {
            'nested_queries': sql_query.count('('),
            'joins': len(re.findall(r'\bJOIN\b', sql_query, re.IGNORECASE)),
            'unions': len(re.findall(r'\bUNION\b', sql_query, re.IGNORECASE)),
            'subqueries': len(re.findall(r'\bSELECT\b', sql_query, re.IGNORECASE)) - 1,
            'conditions': sql_query.count('WHERE') + sql_query.count('HAVING'),
        }
        
        # Weighted sum
        weights = {'nested_queries': 0.2, 'joins': 0.3, 'unions': 0.4, 
                  'subqueries': 0.3, 'conditions': 0.1}
        
        complexity = sum(
            complexity_factors[factor] * weights[factor] 
            for factor in complexity_factors
        )
        
        return complexity
    
    def _assess_query_complexity(self, sql_query: str) -> str:
        """Assess query complexity level."""
        complexity_score = self._calculate_structural_complexity(sql_query)
        
        if complexity_score < 1.0:
            return 'low'
        elif complexity_score < 3.0:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_overall_risk(
        self,
        threats: List[SecurityThreat],
        anomaly_score: float,
        entropy_analysis: Dict[str, float]
    ) -> ThreatLevel:
        """Calculate overall risk level."""
        if not threats and anomaly_score > self.adaptive_thresholds['anomaly_score_threshold']:
            return ThreatLevel.MINIMAL
        
        # Find highest threat level
        max_threat_level = ThreatLevel.MINIMAL
        for threat in threats:
            if threat.threat_level.value == 'critical':
                return ThreatLevel.CRITICAL
            elif threat.threat_level.value == 'high':
                max_threat_level = ThreatLevel.HIGH
            elif threat.threat_level.value == 'medium' and max_threat_level == ThreatLevel.MINIMAL:
                max_threat_level = ThreatLevel.MEDIUM
            elif threat.threat_level.value == 'low' and max_threat_level == ThreatLevel.MINIMAL:
                max_threat_level = ThreatLevel.LOW
        
        # Adjust based on anomaly score
        if anomaly_score < -0.5:  # Very anomalous
            if max_threat_level == ThreatLevel.MINIMAL:
                max_threat_level = ThreatLevel.MEDIUM
            elif max_threat_level == ThreatLevel.LOW:
                max_threat_level = ThreatLevel.HIGH
        
        return max_threat_level
    
    def _calculate_confidence_score(
        self,
        threats: List[SecurityThreat],
        anomaly_score: float
    ) -> float:
        """Calculate overall confidence score."""
        if not threats:
            return 0.95 if anomaly_score > -0.1 else 0.7
        
        # Average confidence of detected threats
        threat_confidences = [threat.confidence for threat in threats]
        avg_confidence = np.mean(threat_confidences)
        
        # Adjust based on number of threats
        confidence_boost = min(0.2, len(threats) * 0.05)
        
        return min(1.0, avg_confidence + confidence_boost)
    
    def _calculate_security_score(self, risk_level: ThreatLevel, confidence: float) -> float:
        """Calculate overall security score (0-1, higher is more secure)."""
        risk_penalties = {
            ThreatLevel.MINIMAL: 0.0,
            ThreatLevel.LOW: 0.1,
            ThreatLevel.MEDIUM: 0.3,
            ThreatLevel.HIGH: 0.7,
            ThreatLevel.CRITICAL: 1.0
        }
        
        penalty = risk_penalties.get(risk_level, 0.5)
        base_score = 1.0 - penalty
        
        # Adjust for confidence
        adjusted_score = base_score * confidence
        
        return max(0.0, min(1.0, adjusted_score))
    
    def _generate_security_recommendations(
        self,
        threats: List[SecurityThreat],
        entropy_analysis: Dict[str, float]
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if not threats:
            recommendations.append("Query appears safe - continue with normal processing")
        else:
            # Add threat-specific recommendations
            for threat in threats:
                recommendations.append(f"{threat.threat_type.value}: {threat.mitigation}")
        
        # Add entropy-based recommendations
        if entropy_analysis.get('shannon_entropy', 0) > 6.0:
            recommendations.append("High entropy detected - verify query legitimacy")
        
        if entropy_analysis.get('keyword_density', 0) > 0.5:
            recommendations.append("High keyword density - review for injection patterns")
        
        # General recommendations
        recommendations.extend([
            "Use parameterized queries for all user input",
            "Implement proper input validation and sanitization",
            "Apply principle of least privilege for database access",
            "Monitor and log all database queries"
        ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _update_validation_metrics(self, assessment: SecurityAssessment) -> None:
        """Update internal validation metrics."""
        self.validation_count += 1
        
        if assessment.threats:
            self.threat_detection_count += 1
        
        # Update false positive rate (simplified calculation)
        if self.validation_count > 0:
            self.false_positive_rate = 1.0 - (self.threat_detection_count / self.validation_count)
    
    async def _adaptive_learning_update(
        self,
        sql_query: str,
        assessment: SecurityAssessment
    ) -> None:
        """Update adaptive learning based on assessment results."""
        try:
            # Store assessment for learning
            self.threat_history.append({
                'query_hash': hashlib.sha256(sql_query.encode()).hexdigest(),
                'assessment': assessment,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(self.threat_history) > 1000:
                self.threat_history = self.threat_history[-1000:]
            
            # Adapt thresholds based on recent patterns
            if len(self.threat_history) >= 10:
                recent_assessments = self.threat_history[-10:]
                
                # Adjust anomaly threshold
                anomaly_scores = [
                    h['assessment'].metadata.get('anomaly_score', 0.0)
                    for h in recent_assessments
                ]
                avg_anomaly = np.mean(anomaly_scores)
                
                if avg_anomaly < -0.3:  # Many anomalies detected
                    self.adaptive_thresholds['anomaly_score_threshold'] = min(
                        0.0, self.adaptive_thresholds['anomaly_score_threshold'] + 0.05
                    )
                elif avg_anomaly > 0.1:  # Few anomalies
                    self.adaptive_thresholds['anomaly_score_threshold'] = max(
                        -0.5, self.adaptive_thresholds['anomaly_score_threshold'] - 0.05
                    )
            
            logger.debug(f"Adaptive thresholds updated: {self.adaptive_thresholds}")
            
        except Exception as e:
            logger.warning(f"Adaptive learning update failed: {e}")
    
    def get_validator_metrics(self) -> Dict[str, Any]:
        """Get validator performance metrics."""
        return {
            'validation_count': self.validation_count,
            'threat_detection_count': self.threat_detection_count,
            'false_positive_rate': self.false_positive_rate,
            'adaptive_thresholds': self.adaptive_thresholds.copy(),
            'threat_history_size': len(self.threat_history),
            'supported_attack_types': [attack.value for attack in AttackType],
            'threat_levels': [level.value for level in ThreatLevel]
        }


# Global validator instance
enhanced_security_validator = EnhancedSQLSecurityValidator()


async def validate_sql_security(
    sql_query: str,
    context: Optional[Dict[str, Any]] = None
) -> SecurityAssessment:
    """
    Validate SQL query security using enhanced framework.
    
    Args:
        sql_query: SQL query to validate
        context: Additional context for validation
        
    Returns:
        Security assessment result
    """
    return await enhanced_security_validator.validate_sql_security(sql_query, context)