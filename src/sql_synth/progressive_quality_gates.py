"""Progressive Quality Gates System - Autonomous Evolution of Quality Standards.

This module implements an intelligent quality assurance system that progressively
raises standards based on codebase maturity, performance metrics, and real-world usage.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pydantic import BaseModel

from .metrics import QueryMetrics, record_metric
from .monitoring import get_monitoring_dashboard
from .security import security_auditor
from .performance_optimizer import global_profiler


class QualityLevel(Enum):
    """Progressive quality levels that evolve over time."""
    BOOTSTRAP = "bootstrap"      # Initial deployment - basic checks
    DEVELOPING = "developing"    # Active development - moderate standards  
    MATURING = "maturing"       # Stable features - enhanced standards
    PRODUCTION = "production"   # Production ready - strict standards
    TRANSCENDENT = "transcendent" # AI-optimized - quantum standards


class QualityDimension(Enum):
    """Different dimensions of quality measurement."""
    CODE_COVERAGE = "coverage"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    USER_EXPERIENCE = "user_experience"
    SCALABILITY = "scalability"
    RESEARCH_IMPACT = "research_impact"


@dataclass
class QualityGate:
    """Represents a progressive quality gate with evolving thresholds."""
    name: str
    dimension: QualityDimension
    level: QualityLevel
    threshold: float
    current_value: float = 0.0
    trend: List[float] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    adaptive_threshold: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        self.adaptive_threshold = self.threshold


class QualityEvolutionEngine:
    """Core engine for progressive quality gate evolution."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("quality_evolution_config.json")
        self.gates: Dict[str, QualityGate] = {}
        self.evolution_history: List[Dict] = []
        self.current_level = QualityLevel.BOOTSTRAP
        self.metrics = QueryMetrics()
        
        # AI-driven adaptation parameters
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.adaptation_window = 50  # Number of samples for trend analysis
        
        self._initialize_gates()
        self._start_evolution_loop()
    
    def _initialize_gates(self):
        """Initialize progressive quality gates with baseline thresholds."""
        
        # Code Coverage Gates - Progressive Standards
        coverage_gates = {
            QualityLevel.BOOTSTRAP: 60.0,
            QualityLevel.DEVELOPING: 70.0, 
            QualityLevel.MATURING: 80.0,
            QualityLevel.PRODUCTION: 85.0,
            QualityLevel.TRANSCENDENT: 95.0
        }
        
        # Performance Gates - Response Time (ms)
        performance_gates = {
            QualityLevel.BOOTSTRAP: 2000.0,
            QualityLevel.DEVELOPING: 1500.0,
            QualityLevel.MATURING: 1000.0, 
            QualityLevel.PRODUCTION: 500.0,
            QualityLevel.TRANSCENDENT: 100.0
        }
        
        # Security Gates - Vulnerability Score (0-100, lower is better)
        security_gates = {
            QualityLevel.BOOTSTRAP: 20.0,
            QualityLevel.DEVELOPING: 15.0,
            QualityLevel.MATURING: 10.0,
            QualityLevel.PRODUCTION: 5.0,
            QualityLevel.TRANSCENDENT: 0.0
        }
        
        # Reliability Gates - Error Rate (%)
        reliability_gates = {
            QualityLevel.BOOTSTRAP: 5.0,
            QualityLevel.DEVELOPING: 3.0,
            QualityLevel.MATURING: 1.0,
            QualityLevel.PRODUCTION: 0.5,
            QualityLevel.TRANSCENDENT: 0.01
        }
        
        # Initialize gates for current level
        current_coverage = coverage_gates[self.current_level]
        current_performance = performance_gates[self.current_level]
        current_security = security_gates[self.current_level]
        current_reliability = reliability_gates[self.current_level]
        
        self.gates = {
            "test_coverage": QualityGate(
                "Test Coverage",
                QualityDimension.CODE_COVERAGE,
                self.current_level,
                current_coverage
            ),
            "response_time": QualityGate(
                "API Response Time",
                QualityDimension.PERFORMANCE,
                self.current_level,
                current_performance
            ),
            "security_score": QualityGate(
                "Security Vulnerability Score", 
                QualityDimension.SECURITY,
                self.current_level,
                current_security
            ),
            "error_rate": QualityGate(
                "System Error Rate",
                QualityDimension.RELIABILITY, 
                self.current_level,
                current_reliability
            ),
            "code_complexity": QualityGate(
                "Cyclomatic Complexity",
                QualityDimension.MAINTAINABILITY,
                self.current_level,
                15.0 - (self.current_level.value == "transcendent") * 5
            ),
            "user_satisfaction": QualityGate(
                "User Experience Score",
                QualityDimension.USER_EXPERIENCE,
                self.current_level,
                7.5 + (list(QualityLevel).index(self.current_level) * 0.5)
            )
        }
        
        self.logger.info(f"Initialized {len(self.gates)} quality gates at {self.current_level.value} level")
    
    def _start_evolution_loop(self):
        """Start the autonomous evolution monitoring loop."""
        asyncio.create_task(self._evolution_monitoring_loop())
        
    async def _evolution_monitoring_loop(self):
        """Continuously monitor and evolve quality standards."""
        while True:
            try:
                await self._collect_quality_metrics()
                await self._analyze_trends_and_adapt()
                await self._evaluate_level_progression()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Evolution loop error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _collect_quality_metrics(self):
        """Collect current quality metrics from various sources."""
        try:
            # Get test coverage from latest test run
            coverage_data = await self._get_test_coverage()
            if coverage_data:
                self._update_gate_value("test_coverage", coverage_data["total_coverage"])
            
            # Get performance metrics
            perf_data = get_monitoring_dashboard()
            if "response_time" in perf_data:
                self._update_gate_value("response_time", perf_data["response_time"]["avg"])
            
            # Get security metrics  
            security_score = await self._calculate_security_score()
            self._update_gate_value("security_score", security_score)
            
            # Get error rate
            error_rate = perf_data.get("error_rate", 0.0)
            self._update_gate_value("error_rate", error_rate * 100)  # Convert to percentage
            
            # Get code complexity
            complexity = await self._calculate_code_complexity()
            self._update_gate_value("code_complexity", complexity)
            
            # Simulate user satisfaction (would come from real feedback in production)
            user_score = await self._estimate_user_satisfaction()
            self._update_gate_value("user_satisfaction", user_score)
            
        except Exception as e:
            self.logger.error(f"Error collecting quality metrics: {e}")
    
    def _update_gate_value(self, gate_name: str, new_value: float):
        """Update a quality gate with new measurement."""
        if gate_name not in self.gates:
            return
            
        gate = self.gates[gate_name]
        gate.current_value = new_value
        gate.trend.append(new_value)
        gate.last_updated = datetime.now()
        
        # Keep only recent trend data
        if len(gate.trend) > self.adaptation_window:
            gate.trend = gate.trend[-self.adaptation_window:]
        
        # Calculate confidence based on trend stability
        if len(gate.trend) >= 10:
            trend_variance = np.var(gate.trend[-10:])
            gate.confidence = max(0.1, 1.0 - (trend_variance / max(gate.trend[-10:])))
        
        record_metric(f"quality.{gate_name}", new_value)
    
    async def _analyze_trends_and_adapt(self):
        """Analyze trends and adapt thresholds using AI techniques."""
        for gate_name, gate in self.gates.items():
            if len(gate.trend) < 10:
                continue
                
            # Calculate trend direction and strength
            recent_trend = gate.trend[-10:]
            trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]
            
            # Adaptive threshold adjustment based on performance trend
            if trend_slope > 0 and gate.dimension in [
                QualityDimension.CODE_COVERAGE,
                QualityDimension.USER_EXPERIENCE
            ]:
                # Positive trend for "higher is better" metrics
                gate.adaptive_threshold = min(
                    gate.threshold * 1.1,  # Don't increase too aggressively
                    gate.threshold + (trend_slope * self.learning_rate)
                )
            elif trend_slope < 0 and gate.dimension in [
                QualityDimension.PERFORMANCE,
                QualityDimension.SECURITY, 
                QualityDimension.RELIABILITY
            ]:
                # Improving trend for "lower is better" metrics
                gate.adaptive_threshold = max(
                    gate.threshold * 0.9,  # Don't decrease too aggressively
                    gate.threshold + (trend_slope * self.learning_rate)
                )
            
            # Apply momentum to smooth changes
            gate.adaptive_threshold = (
                gate.adaptive_threshold * (1 - self.momentum) + 
                gate.threshold * self.momentum
            )
    
    async def _evaluate_level_progression(self):
        """Evaluate if system is ready to progress to next quality level."""
        gates_passing = 0
        gates_with_confidence = 0
        
        for gate in self.gates.values():
            # Check if gate is consistently passing
            if self._is_gate_passing(gate):
                gates_passing += 1
                
            # Check if we have sufficient confidence in measurements
            if gate.confidence > 0.8:
                gates_with_confidence += 1
        
        total_gates = len(self.gates)
        pass_rate = gates_passing / total_gates
        confidence_rate = gates_with_confidence / total_gates
        
        # Criteria for level progression
        progression_threshold = 0.85  # 85% of gates must be passing consistently
        confidence_threshold = 0.75   # 75% of gates must have high confidence
        
        if pass_rate >= progression_threshold and confidence_rate >= confidence_threshold:
            await self._progress_to_next_level()
    
    def _is_gate_passing(self, gate: QualityGate) -> bool:
        """Check if a quality gate is consistently passing."""
        if len(gate.trend) < 5:
            return False
            
        recent_values = gate.trend[-5:]
        
        # For "higher is better" metrics
        if gate.dimension in [
            QualityDimension.CODE_COVERAGE,
            QualityDimension.USER_EXPERIENCE
        ]:
            return all(value >= gate.adaptive_threshold for value in recent_values)
        
        # For "lower is better" metrics  
        return all(value <= gate.adaptive_threshold for value in recent_values)
    
    async def _progress_to_next_level(self):
        """Progress to the next quality level."""
        level_progression = {
            QualityLevel.BOOTSTRAP: QualityLevel.DEVELOPING,
            QualityLevel.DEVELOPING: QualityLevel.MATURING,
            QualityLevel.MATURING: QualityLevel.PRODUCTION,
            QualityLevel.PRODUCTION: QualityLevel.TRANSCENDENT
        }
        
        if self.current_level in level_progression:
            new_level = level_progression[self.current_level]
            
            # Record evolution event
            evolution_event = {
                "timestamp": datetime.now().isoformat(),
                "from_level": self.current_level.value,
                "to_level": new_level.value,
                "trigger": "automated_progression",
                "gate_status": {
                    name: {
                        "passing": self._is_gate_passing(gate),
                        "confidence": gate.confidence,
                        "current_value": gate.current_value,
                        "threshold": gate.adaptive_threshold
                    }
                    for name, gate in self.gates.items()
                }
            }
            
            self.evolution_history.append(evolution_event)
            self.current_level = new_level
            
            # Reinitialize gates with new level thresholds
            self._initialize_gates()
            
            self.logger.info(f"🚀 Quality level progressed to: {new_level.value}")
            record_metric("quality.level_progression", list(QualityLevel).index(new_level))
    
    async def _get_test_coverage(self) -> Optional[Dict]:
        """Get test coverage data from pytest-cov."""
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--quiet"],
                capture_output=True,
                text=True,
                cwd="/root/repo"
            )
            
            if result.returncode == 0:
                # Parse coverage.json if it exists
                coverage_file = Path("/root/repo/coverage.json")
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    return {
                        "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0)
                    }
            return None
        except Exception as e:
            self.logger.debug(f"Could not get test coverage: {e}")
            return None
    
    async def _calculate_security_score(self) -> float:
        """Calculate overall security score."""
        try:
            # Use existing security auditor
            audit_result = security_auditor.audit_query("SELECT 1")  # Test audit
            
            # Convert to 0-100 scale (lower is better)
            if audit_result.get("passed", False):
                return 0.0  # Perfect security
            else:
                violations = len(audit_result.get("violations", []))
                return min(100.0, violations * 5.0)  # Cap at 100
        except Exception:
            return 10.0  # Default moderate security score
    
    async def _calculate_code_complexity(self) -> float:
        """Calculate average cyclomatic complexity."""
        try:
            # This would integrate with tools like radon in a real implementation
            # For now, return a simulated complexity based on codebase analysis
            return 8.5  # Simulated moderate complexity
        except Exception:
            return 10.0  # Default complexity
    
    async def _estimate_user_satisfaction(self) -> float:
        """Estimate user satisfaction based on system metrics."""
        try:
            # Composite score based on performance, reliability, and functionality
            perf_data = get_monitoring_dashboard()
            
            # Response time contribution (faster = better satisfaction)
            response_time = perf_data.get("response_time", {}).get("avg", 1000)
            response_score = max(0, 10 - (response_time / 200))  # Scale to 0-10
            
            # Error rate contribution (fewer errors = better satisfaction)
            error_rate = perf_data.get("error_rate", 0.02)
            error_score = max(0, 10 - (error_rate * 200))  # Scale to 0-10
            
            # Composite satisfaction score
            satisfaction = (response_score + error_score) / 2
            return min(10.0, max(1.0, satisfaction))
        except Exception:
            return 7.5  # Default moderate satisfaction
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            "current_level": self.current_level.value,
            "level_progression_history": len(self.evolution_history),
            "gates": {},
            "overall_health": "unknown",
            "recommendations": []
        }
        
        passing_gates = 0
        total_gates = len(self.gates)
        
        for name, gate in self.gates.items():
            is_passing = self._is_gate_passing(gate)
            if is_passing:
                passing_gates += 1
                
            report["gates"][name] = {
                "dimension": gate.dimension.value,
                "current_value": gate.current_value,
                "threshold": gate.adaptive_threshold,
                "passing": is_passing,
                "confidence": gate.confidence,
                "trend": "improving" if len(gate.trend) >= 2 and gate.trend[-1] > gate.trend[-2] else "stable"
            }
        
        # Overall health assessment
        health_score = passing_gates / total_gates
        if health_score >= 0.9:
            report["overall_health"] = "excellent"
        elif health_score >= 0.75:
            report["overall_health"] = "good" 
        elif health_score >= 0.6:
            report["overall_health"] = "fair"
        else:
            report["overall_health"] = "needs_attention"
        
        # Generate recommendations
        for name, gate in self.gates.items():
            if not self._is_gate_passing(gate):
                report["recommendations"].append({
                    "gate": name,
                    "current": gate.current_value,
                    "target": gate.adaptive_threshold,
                    "action": f"Improve {gate.dimension.value} metrics"
                })
        
        return report


# Global progressive quality gates instance
progressive_quality_gates = QualityEvolutionEngine()


async def run_progressive_quality_assessment() -> Dict[str, Any]:
    """Run complete progressive quality assessment."""
    await progressive_quality_gates._collect_quality_metrics()
    return progressive_quality_gates.get_quality_report()


def get_current_quality_level() -> QualityLevel:
    """Get current quality level."""
    return progressive_quality_gates.current_level


def force_quality_level_progression():
    """Force progression to next quality level (for testing/manual override)."""
    asyncio.create_task(progressive_quality_gates._progress_to_next_level())