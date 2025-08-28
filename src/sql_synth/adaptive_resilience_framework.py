"""Adaptive Resilience Framework - Self-Healing and Self-Adapting System Architecture.

This module implements a comprehensive resilience framework that automatically detects,
responds to, and learns from failures to continuously improve system reliability.
"""

import asyncio
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from pydantic import BaseModel

from .error_handling import global_error_manager, ErrorCategory, ErrorSeverity
from .metrics import record_metric
from .monitoring import get_monitoring_dashboard


class FailureType(Enum):
    """Types of failures the system can encounter."""
    NETWORK_TIMEOUT = "network_timeout"
    DATABASE_CONNECTION = "database_connection"
    MEMORY_EXHAUSTION = "memory_exhaustion" 
    CPU_OVERLOAD = "cpu_overload"
    DISK_FULL = "disk_full"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION_ERROR = "configuration_error"


class ResiliencePattern(Enum):
    """Resilience patterns that can be applied."""
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    CACHE_ASIDE = "cache_aside"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    LOAD_SHEDDING = "load_shedding"
    HEALTH_CHECK = "health_check"
    REDUNDANCY = "redundancy"


@dataclass
class FailureIncident:
    """Represents a failure incident for learning and adaptation."""
    id: str
    failure_type: FailureType
    timestamp: datetime
    context: Dict[str, Any]
    impact_severity: float  # 0.0 to 1.0
    resolution_time: Optional[float] = None
    resolution_method: Optional[ResiliencePattern] = None
    lessons_learned: List[str] = field(default_factory=list)


@dataclass 
class ResilienceRule:
    """Represents an adaptive resilience rule."""
    id: str
    pattern: ResiliencePattern
    trigger_conditions: Dict[str, Any]
    parameters: Dict[str, Any]
    effectiveness_score: float = 0.5
    usage_count: int = 0
    success_rate: float = 0.0
    last_updated: Optional[datetime] = None


class CircuitBreaker:
    """Adaptive circuit breaker implementation."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, success_threshold: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.adaptive_threshold = failure_threshold
        
        # Learning parameters
        self.failure_history = deque(maxlen=100)
        self.adaptation_rate = 0.1
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.success_count = 0
            else:
                raise Exception(f"Circuit breaker {self.name} is open")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    async def _on_success(self):
        """Handle successful execution."""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                await self._adapt_parameters()
        elif self.state == "closed":
            # Reset failure count on successful execution
            self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self, error: Exception):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.failure_history.append({
            "timestamp": time.time(),
            "error": str(error),
            "state": self.state
        })
        
        if self.failure_count >= self.adaptive_threshold:
            self.state = "open"
            await self._adapt_parameters()
        
        record_metric(f"circuit_breaker.{self.name}.failures", 1)
    
    async def _adapt_parameters(self):
        """Adapt circuit breaker parameters based on historical data."""
        if len(self.failure_history) < 10:
            return
        
        # Analyze failure patterns
        recent_failures = [f for f in self.failure_history if time.time() - f["timestamp"] < 3600]  # Last hour
        failure_rate = len(recent_failures) / min(len(self.failure_history), 60)  # Failures per minute
        
        # Adapt threshold based on failure patterns
        if failure_rate > 0.5:  # High failure rate
            self.adaptive_threshold = max(1, int(self.failure_threshold * 0.8))  # Lower threshold
        elif failure_rate < 0.1:  # Low failure rate
            self.adaptive_threshold = min(10, int(self.failure_threshold * 1.2))  # Higher threshold
        
        # Adapt recovery timeout
        avg_recovery_time = self._calculate_average_recovery_time()
        if avg_recovery_time > 0:
            self.recovery_timeout = int(avg_recovery_time * 1.5)  # 50% buffer
    
    def _calculate_average_recovery_time(self) -> float:
        """Calculate average recovery time from historical data."""
        recovery_times = []
        last_open_time = None
        
        for failure in self.failure_history:
            if failure["state"] == "open" and last_open_time is None:
                last_open_time = failure["timestamp"]
            elif failure["state"] == "closed" and last_open_time is not None:
                recovery_times.append(failure["timestamp"] - last_open_time)
                last_open_time = None
        
        return np.mean(recovery_times) if recovery_times else 0.0


class AdaptiveResilienceFramework:
    """Core adaptive resilience framework that learns from failures."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.incidents: Dict[str, FailureIncident] = {}
        self.rules: Dict[str, ResilienceRule] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_patterns: Dict[FailureType, List[Dict]] = defaultdict(list)
        
        # Learning and adaptation parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.8
        self.pattern_recognition_window = timedelta(hours=24)
        
        # Resilience metrics
        self.system_health_score = 1.0
        self.adaptation_history: List[Dict] = []
        
        self._initialize_default_rules()
        self._start_resilience_monitoring()
    
    def _initialize_default_rules(self):
        """Initialize default resilience rules."""
        
        # Network timeout resilience
        self.rules["network_timeout_retry"] = ResilienceRule(
            id="network_timeout_retry",
            pattern=ResiliencePattern.RETRY_WITH_BACKOFF,
            trigger_conditions={"failure_type": FailureType.NETWORK_TIMEOUT},
            parameters={
                "max_retries": 3,
                "initial_delay": 1.0,
                "backoff_multiplier": 2.0,
                "max_delay": 30.0
            }
        )
        
        # Database connection circuit breaker
        self.rules["db_circuit_breaker"] = ResilienceRule(
            id="db_circuit_breaker", 
            pattern=ResiliencePattern.CIRCUIT_BREAKER,
            trigger_conditions={"failure_type": FailureType.DATABASE_CONNECTION},
            parameters={
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "success_threshold": 3
            }
        )
        
        # Memory exhaustion load shedding
        self.rules["memory_load_shedding"] = ResilienceRule(
            id="memory_load_shedding",
            pattern=ResiliencePattern.LOAD_SHEDDING,
            trigger_conditions={"failure_type": FailureType.MEMORY_EXHAUSTION},
            parameters={
                "shed_percentage": 0.3,
                "priority_threshold": 0.7
            }
        )
        
        # Rate limiting backoff
        self.rules["rate_limit_backoff"] = ResilienceRule(
            id="rate_limit_backoff",
            pattern=ResiliencePattern.RETRY_WITH_BACKOFF,
            trigger_conditions={"failure_type": FailureType.RATE_LIMIT_EXCEEDED},
            parameters={
                "max_retries": 5,
                "initial_delay": 5.0,
                "backoff_multiplier": 1.5,
                "jitter": True
            }
        )
        
        self.logger.info(f"Initialized {len(self.rules)} default resilience rules")
    
    def _start_resilience_monitoring(self):
        """Start continuous resilience monitoring and adaptation."""
        asyncio.create_task(self._resilience_monitoring_loop())
    
    async def _resilience_monitoring_loop(self):
        """Continuous monitoring and adaptation loop."""
        while True:
            try:
                await self._analyze_failure_patterns()
                await self._adapt_rules()
                await self._update_system_health()
                await self._cleanup_old_data()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Resilience monitoring error: {e}")
                await asyncio.sleep(300)  # Back off on error
    
    async def handle_failure(self, failure_type: FailureType, context: Dict[str, Any], 
                           error: Exception) -> Dict[str, Any]:
        """Handle a failure incident with adaptive resilience."""
        incident_id = f"{failure_type.value}_{int(time.time())}"
        
        # Create failure incident
        incident = FailureIncident(
            id=incident_id,
            failure_type=failure_type,
            timestamp=datetime.now(),
            context=context,
            impact_severity=self._calculate_impact_severity(failure_type, context),
        )
        
        self.incidents[incident_id] = incident
        
        # Find applicable resilience rules
        applicable_rules = self._find_applicable_rules(failure_type, context)
        
        # Apply resilience patterns
        recovery_result = await self._apply_resilience_patterns(incident, applicable_rules)
        
        # Update rule effectiveness
        await self._update_rule_effectiveness(applicable_rules, recovery_result)
        
        # Record failure pattern for learning
        self._record_failure_pattern(incident)
        
        return recovery_result
    
    def _find_applicable_rules(self, failure_type: FailureType, 
                             context: Dict[str, Any]) -> List[ResilienceRule]:
        """Find rules applicable to the current failure."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if self._rule_matches(rule, failure_type, context):
                applicable_rules.append(rule)
        
        # Sort by effectiveness score
        applicable_rules.sort(key=lambda r: r.effectiveness_score, reverse=True)
        return applicable_rules
    
    def _rule_matches(self, rule: ResilienceRule, failure_type: FailureType, 
                     context: Dict[str, Any]) -> bool:
        """Check if a rule matches the current failure conditions."""
        trigger_conditions = rule.trigger_conditions
        
        # Check failure type match
        if "failure_type" in trigger_conditions:
            if trigger_conditions["failure_type"] != failure_type:
                return False
        
        # Check context conditions
        for key, expected_value in trigger_conditions.items():
            if key == "failure_type":
                continue
            if key not in context or context[key] != expected_value:
                return False
        
        return True
    
    async def _apply_resilience_patterns(self, incident: FailureIncident, 
                                       rules: List[ResilienceRule]) -> Dict[str, Any]:
        """Apply resilience patterns to handle the failure."""
        results = []
        
        for rule in rules[:3]:  # Apply top 3 rules
            try:
                result = await self._apply_single_pattern(incident, rule)
                results.append(result)
                
                if result.get("success", False):
                    incident.resolution_method = rule.pattern
                    incident.resolution_time = time.time() - incident.timestamp.timestamp()
                    break  # Stop on first successful recovery
                    
            except Exception as e:
                self.logger.error(f"Failed to apply resilience rule {rule.id}: {e}")
        
        return {
            "incident_id": incident.id,
            "patterns_applied": len(results),
            "success": any(r.get("success", False) for r in results),
            "results": results
        }
    
    async def _apply_single_pattern(self, incident: FailureIncident, 
                                  rule: ResilienceRule) -> Dict[str, Any]:
        """Apply a single resilience pattern."""
        pattern = rule.pattern
        parameters = rule.parameters
        
        rule.usage_count += 1
        
        if pattern == ResiliencePattern.CIRCUIT_BREAKER:
            return await self._apply_circuit_breaker(incident, parameters)
        elif pattern == ResiliencePattern.RETRY_WITH_BACKOFF:
            return await self._apply_retry_with_backoff(incident, parameters)
        elif pattern == ResiliencePattern.LOAD_SHEDDING:
            return await self._apply_load_shedding(incident, parameters)
        elif pattern == ResiliencePattern.GRACEFUL_DEGRADATION:
            return await self._apply_graceful_degradation(incident, parameters)
        elif pattern == ResiliencePattern.FAIL_FAST:
            return await self._apply_fail_fast(incident, parameters)
        else:
            return {"success": False, "error": f"Unknown pattern: {pattern}"}
    
    async def _apply_circuit_breaker(self, incident: FailureIncident, 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply circuit breaker pattern."""
        service_name = incident.context.get("service", "unknown")
        
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                name=service_name,
                failure_threshold=parameters.get("failure_threshold", 5),
                recovery_timeout=parameters.get("recovery_timeout", 60),
                success_threshold=parameters.get("success_threshold", 3)
            )
        
        circuit_breaker = self.circuit_breakers[service_name]
        
        # Circuit breaker automatically handles the failure
        return {
            "success": True,
            "action": "circuit_breaker_activated",
            "state": circuit_breaker.state,
            "failure_count": circuit_breaker.failure_count
        }
    
    async def _apply_retry_with_backoff(self, incident: FailureIncident,
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply retry with exponential backoff pattern."""
        max_retries = parameters.get("max_retries", 3)
        initial_delay = parameters.get("initial_delay", 1.0)
        backoff_multiplier = parameters.get("backoff_multiplier", 2.0)
        max_delay = parameters.get("max_delay", 60.0)
        use_jitter = parameters.get("jitter", False)
        
        for attempt in range(max_retries):
            delay = min(initial_delay * (backoff_multiplier ** attempt), max_delay)
            
            if use_jitter:
                delay *= (0.5 + random.random())  # Add jitter
            
            await asyncio.sleep(delay)
            
            # Simulate retry (in real implementation, would retry the actual operation)
            success_probability = 0.7 + (0.2 * attempt)  # Increasing success probability
            if random.random() < success_probability:
                return {
                    "success": True,
                    "action": "retry_succeeded",
                    "attempts": attempt + 1,
                    "total_delay": delay
                }
        
        return {
            "success": False,
            "action": "retry_exhausted",
            "attempts": max_retries
        }
    
    async def _apply_load_shedding(self, incident: FailureIncident,
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply load shedding pattern."""
        shed_percentage = parameters.get("shed_percentage", 0.2)
        priority_threshold = parameters.get("priority_threshold", 0.5)
        
        # Simulate load shedding
        current_load = incident.context.get("system_load", 0.8)
        target_load = current_load * (1 - shed_percentage)
        
        return {
            "success": True,
            "action": "load_shedding_applied",
            "shed_percentage": shed_percentage,
            "new_load": target_load,
            "priority_threshold": priority_threshold
        }
    
    async def _apply_graceful_degradation(self, incident: FailureIncident,
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply graceful degradation pattern."""
        degradation_level = parameters.get("degradation_level", 0.5)
        
        return {
            "success": True,
            "action": "graceful_degradation",
            "degradation_level": degradation_level,
            "available_features": 1 - degradation_level
        }
    
    async def _apply_fail_fast(self, incident: FailureIncident,
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fail fast pattern."""
        timeout = parameters.get("timeout", 5.0)
        
        return {
            "success": True,
            "action": "fail_fast_activated",
            "timeout": timeout
        }
    
    def _calculate_impact_severity(self, failure_type: FailureType, 
                                 context: Dict[str, Any]) -> float:
        """Calculate the impact severity of a failure."""
        base_severity = {
            FailureType.NETWORK_TIMEOUT: 0.3,
            FailureType.DATABASE_CONNECTION: 0.8,
            FailureType.MEMORY_EXHAUSTION: 0.9,
            FailureType.CPU_OVERLOAD: 0.7,
            FailureType.DISK_FULL: 0.8,
            FailureType.SERVICE_UNAVAILABLE: 0.9,
            FailureType.AUTHENTICATION_FAILURE: 0.6,
            FailureType.RATE_LIMIT_EXCEEDED: 0.4,
            FailureType.DATA_CORRUPTION: 1.0,
            FailureType.CONFIGURATION_ERROR: 0.7
        }.get(failure_type, 0.5)
        
        # Adjust based on context
        user_impact = context.get("affected_users", 0) / 1000.0  # Normalize
        system_load = context.get("system_load", 0.5)
        
        adjusted_severity = base_severity + (user_impact * 0.3) + (system_load * 0.2)
        return min(1.0, adjusted_severity)
    
    async def _update_rule_effectiveness(self, rules: List[ResilienceRule],
                                       recovery_result: Dict[str, Any]):
        """Update rule effectiveness based on recovery results."""
        success = recovery_result.get("success", False)
        
        for rule in rules:
            # Update success rate
            if rule.usage_count == 1:
                rule.success_rate = 1.0 if success else 0.0
            else:
                # Exponential moving average
                alpha = self.learning_rate
                rule.success_rate = (1 - alpha) * rule.success_rate + alpha * (1.0 if success else 0.0)
            
            # Update effectiveness score (combines success rate and usage)
            usage_factor = min(1.0, rule.usage_count / 10.0)  # Max factor at 10 uses
            rule.effectiveness_score = rule.success_rate * 0.8 + usage_factor * 0.2
            
            rule.last_updated = datetime.now()
    
    def _record_failure_pattern(self, incident: FailureIncident):
        """Record failure pattern for learning."""
        pattern_data = {
            "timestamp": incident.timestamp.timestamp(),
            "context": incident.context,
            "impact_severity": incident.impact_severity,
            "resolution_time": incident.resolution_time,
            "resolution_method": incident.resolution_method.value if incident.resolution_method else None
        }
        
        self.failure_patterns[incident.failure_type].append(pattern_data)
        
        # Keep only recent patterns
        cutoff_time = (datetime.now() - self.pattern_recognition_window).timestamp()
        self.failure_patterns[incident.failure_type] = [
            p for p in self.failure_patterns[incident.failure_type]
            if p["timestamp"] > cutoff_time
        ]
    
    async def _analyze_failure_patterns(self):
        """Analyze failure patterns to identify trends and create new rules."""
        for failure_type, patterns in self.failure_patterns.items():
            if len(patterns) < 5:  # Need sufficient data
                continue
            
            # Analyze pattern trends
            recent_patterns = [p for p in patterns 
                             if p["timestamp"] > (time.time() - 3600)]  # Last hour
            
            if len(recent_patterns) >= 3:  # Spike in failures
                await self._create_adaptive_rule(failure_type, recent_patterns)
    
    async def _create_adaptive_rule(self, failure_type: FailureType, patterns: List[Dict]):
        """Create adaptive rule based on failure patterns."""
        avg_severity = np.mean([p["impact_severity"] for p in patterns])
        avg_resolution_time = np.mean([p.get("resolution_time", 60) for p in patterns if p.get("resolution_time")])
        
        # Create rule based on pattern analysis
        rule_id = f"adaptive_{failure_type.value}_{int(time.time())}"
        
        if avg_severity > 0.7:  # High severity failures
            # Create aggressive circuit breaker
            new_rule = ResilienceRule(
                id=rule_id,
                pattern=ResiliencePattern.CIRCUIT_BREAKER,
                trigger_conditions={"failure_type": failure_type},
                parameters={
                    "failure_threshold": 3,  # More aggressive
                    "recovery_timeout": max(30, int(avg_resolution_time * 0.8)),
                    "success_threshold": 2
                }
            )
        else:
            # Create retry with backoff
            new_rule = ResilienceRule(
                id=rule_id,
                pattern=ResiliencePattern.RETRY_WITH_BACKOFF,
                trigger_conditions={"failure_type": failure_type},
                parameters={
                    "max_retries": 2,
                    "initial_delay": min(5.0, avg_resolution_time / 10),
                    "backoff_multiplier": 1.5
                }
            )
        
        self.rules[rule_id] = new_rule
        
        # Record adaptation event
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": "rule_created",
            "rule_id": rule_id,
            "failure_type": failure_type.value,
            "trigger_patterns": len(patterns),
            "avg_severity": avg_severity
        })
        
        self.logger.info(f"Created adaptive rule {rule_id} for {failure_type.value}")
    
    async def _adapt_rules(self):
        """Adapt existing rules based on their effectiveness."""
        for rule in self.rules.values():
            if rule.usage_count < 5 or rule.last_updated is None:
                continue
            
            # Adapt based on effectiveness
            if rule.effectiveness_score < 0.3:  # Poor performance
                await self._deprecate_rule(rule)
            elif rule.effectiveness_score > 0.8:  # Good performance
                await self._enhance_rule(rule)
    
    async def _deprecate_rule(self, rule: ResilienceRule):
        """Deprecate a poorly performing rule."""
        self.logger.info(f"Deprecating low-effectiveness rule: {rule.id}")
        # Mark for removal (in production, might soft-delete)
        rule.parameters["deprecated"] = True
    
    async def _enhance_rule(self, rule: ResilienceRule):
        """Enhance a well-performing rule."""
        # Make the rule more efficient
        if rule.pattern == ResiliencePattern.RETRY_WITH_BACKOFF:
            # Reduce delays for successful patterns
            rule.parameters["initial_delay"] *= 0.9
            rule.parameters["max_delay"] = min(rule.parameters.get("max_delay", 60), 30)
        elif rule.pattern == ResiliencePattern.CIRCUIT_BREAKER:
            # Adjust thresholds for successful patterns
            rule.parameters["failure_threshold"] = min(rule.parameters.get("failure_threshold", 5) + 1, 8)
    
    async def _update_system_health(self):
        """Update overall system health score."""
        recent_incidents = [
            incident for incident in self.incidents.values()
            if (datetime.now() - incident.timestamp) < timedelta(hours=1)
        ]
        
        if not recent_incidents:
            self.system_health_score = min(1.0, self.system_health_score + 0.1)
        else:
            avg_severity = np.mean([i.impact_severity for i in recent_incidents])
            incident_rate = len(recent_incidents) / 60.0  # Incidents per minute
            
            health_impact = avg_severity * incident_rate * 0.1
            self.system_health_score = max(0.0, self.system_health_score - health_impact)
        
        record_metric("resilience.system_health_score", self.system_health_score)
    
    async def _cleanup_old_data(self):
        """Clean up old incident and pattern data."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        # Remove old incidents
        old_incident_ids = [
            incident_id for incident_id, incident in self.incidents.items()
            if incident.timestamp < cutoff_time
        ]
        
        for incident_id in old_incident_ids:
            del self.incidents[incident_id]
        
        # Clean up old patterns
        cutoff_timestamp = cutoff_time.timestamp()
        for failure_type in self.failure_patterns:
            self.failure_patterns[failure_type] = [
                p for p in self.failure_patterns[failure_type]
                if p["timestamp"] > cutoff_timestamp
            ]
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get current resilience framework status."""
        return {
            "system_health_score": self.system_health_score,
            "total_incidents": len(self.incidents),
            "active_rules": len([r for r in self.rules.values() if not r.parameters.get("deprecated", False)]),
            "circuit_breakers": {name: cb.state for name, cb in self.circuit_breakers.items()},
            "recent_adaptations": len(self.adaptation_history[-10:]),
            "failure_patterns": {ft.value: len(patterns) for ft, patterns in self.failure_patterns.items()}
        }


# Global adaptive resilience framework instance
adaptive_resilience = AdaptiveResilienceFramework()


async def handle_system_failure(failure_type: FailureType, context: Dict[str, Any], 
                              error: Exception) -> Dict[str, Any]:
    """Handle system failure with adaptive resilience."""
    return await adaptive_resilience.handle_failure(failure_type, context, error)


def get_resilience_status() -> Dict[str, Any]:
    """Get current resilience framework status."""
    return adaptive_resilience.get_resilience_status()


async def simulate_chaos_testing():
    """Simulate chaos testing to validate resilience."""
    chaos_scenarios = [
        (FailureType.NETWORK_TIMEOUT, {"service": "database", "timeout": 5.0}),
        (FailureType.DATABASE_CONNECTION, {"connection_pool": "exhausted"}),
        (FailureType.MEMORY_EXHAUSTION, {"used_memory": 0.95}),
        (FailureType.RATE_LIMIT_EXCEEDED, {"rate": 100, "limit": 50}),
    ]
    
    results = []
    for failure_type, context in chaos_scenarios:
        try:
            result = await handle_system_failure(
                failure_type, context, Exception(f"Simulated {failure_type.value}")
            )
            results.append(result)
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    
    return results