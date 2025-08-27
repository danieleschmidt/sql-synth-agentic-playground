"""Intelligent Deployment Orchestrator - AI-Driven Multi-Environment Deployment Management.

This module implements an intelligent deployment orchestrator that manages deployments
across multiple environments with AI-driven risk assessment, automated rollbacks,
and progressive delivery strategies.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import yaml
from pydantic import BaseModel

from .metrics import record_metric
from .monitoring import get_monitoring_dashboard
from .adaptive_resilience_framework import adaptive_resilience, FailureType


class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE = "blue"
    GREEN = "green"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "a_b_testing"
    SHADOW = "shadow"
    RECREATE = "recreate"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    id: str
    version: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    image_tag: str
    replicas: int = 1
    health_check_timeout: int = 300
    rollback_threshold: float = 0.1  # Error rate threshold for rollback
    traffic_percentage: float = 1.0  # For canary/A-B testing
    success_criteria: Dict[str, float] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentRisk:
    """Deployment risk assessment."""
    overall_score: float  # 0.0 to 1.0 (higher is riskier)
    code_risk: float
    infrastructure_risk: float
    timing_risk: float
    dependency_risk: float
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass 
class DeploymentExecution:
    """Represents a deployment execution."""
    id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    risk_assessment: Optional[DeploymentRisk] = None
    health_metrics: Dict[str, float] = field(default_factory=dict)
    rollback_plan: Optional[Dict[str, Any]] = None
    logs: List[str] = field(default_factory=list)


class IntelligentDeploymentOrchestrator:
    """AI-driven deployment orchestrator with risk assessment and automated management."""
    
    def __init__(self, workspace_path: Path = Path("/root/repo")):
        self.logger = logging.getLogger(__name__)
        self.workspace_path = workspace_path
        self.active_deployments: Dict[str, DeploymentExecution] = {}
        self.deployment_history: List[DeploymentExecution] = []
        
        # AI-driven parameters
        self.risk_tolerance = 0.3  # Maximum acceptable risk score
        self.learning_rate = 0.1
        self.deployment_success_memory: Dict[str, float] = {}
        
        # Health monitoring
        self.monitoring_interval = 30  # seconds
        self.health_check_retries = 3
        
        # Environment management
        self.environment_configs = self._load_environment_configs()
        
        self._start_deployment_monitoring()
    
    def _load_environment_configs(self) -> Dict[DeploymentEnvironment, Dict[str, Any]]:
        """Load environment-specific configurations."""
        configs = {
            DeploymentEnvironment.DEVELOPMENT: {
                "replicas": 1,
                "health_check_timeout": 60,
                "resource_limits": {"cpu": "500m", "memory": "512Mi"},
                "auto_rollback": False
            },
            DeploymentEnvironment.STAGING: {
                "replicas": 2,
                "health_check_timeout": 120,
                "resource_limits": {"cpu": "1000m", "memory": "1Gi"},
                "auto_rollback": True
            },
            DeploymentEnvironment.PRODUCTION: {
                "replicas": 3,
                "health_check_timeout": 300,
                "resource_limits": {"cpu": "2000m", "memory": "2Gi"},
                "auto_rollback": True,
                "blue_green_enabled": True
            },
            DeploymentEnvironment.CANARY: {
                "replicas": 1,
                "health_check_timeout": 180,
                "resource_limits": {"cpu": "1000m", "memory": "1Gi"},
                "traffic_percentage": 0.05  # 5% traffic
            }
        }
        
        self.logger.info(f"Loaded configurations for {len(configs)} environments")
        return configs
    
    def _start_deployment_monitoring(self):
        """Start continuous deployment monitoring."""
        asyncio.create_task(self._deployment_monitoring_loop())
    
    async def _deployment_monitoring_loop(self):
        """Monitor active deployments and manage their lifecycle."""
        while True:
            try:
                await self._monitor_active_deployments()
                await self._cleanup_completed_deployments()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Deployment monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def deploy(self, config: DeploymentConfig) -> str:
        """Initiate intelligent deployment with risk assessment."""
        deployment_id = f"deploy_{config.environment.value}_{int(time.time())}"
        
        # Create deployment execution
        execution = DeploymentExecution(
            id=deployment_id,
            config=config,
            status=DeploymentStatus.PENDING
        )
        
        self.active_deployments[deployment_id] = execution
        
        # Perform risk assessment
        execution.risk_assessment = await self._assess_deployment_risk(config)
        
        self.logger.info(f"Starting deployment {deployment_id} with risk score: {execution.risk_assessment.overall_score:.2f}")
        
        # Check if risk is acceptable
        if execution.risk_assessment.overall_score > self.risk_tolerance:
            execution.status = DeploymentStatus.FAILED
            execution.logs.append(f"Deployment blocked due to high risk score: {execution.risk_assessment.overall_score:.2f}")
            
            # Suggest mitigation strategies
            self.logger.warning(f"Deployment {deployment_id} blocked. Mitigation strategies: {execution.risk_assessment.mitigation_strategies}")
            return deployment_id
        
        # Execute deployment asynchronously
        asyncio.create_task(self._execute_deployment(execution))
        
        return deployment_id
    
    async def _assess_deployment_risk(self, config: DeploymentConfig) -> DeploymentRisk:
        """Assess deployment risk using multiple factors."""
        
        # Code risk assessment
        code_risk = await self._assess_code_risk(config)
        
        # Infrastructure risk assessment  
        infrastructure_risk = await self._assess_infrastructure_risk(config)
        
        # Timing risk assessment
        timing_risk = await self._assess_timing_risk(config)
        
        # Dependency risk assessment
        dependency_risk = await self._assess_dependency_risk(config)
        
        # Calculate overall risk score
        weights = {"code": 0.3, "infrastructure": 0.25, "timing": 0.2, "dependency": 0.25}
        overall_score = (
            code_risk * weights["code"] +
            infrastructure_risk * weights["infrastructure"] +
            timing_risk * weights["timing"] +
            dependency_risk * weights["dependency"]
        )
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            overall_score, code_risk, infrastructure_risk, timing_risk, dependency_risk
        )
        
        return DeploymentRisk(
            overall_score=overall_score,
            code_risk=code_risk,
            infrastructure_risk=infrastructure_risk,
            timing_risk=timing_risk,
            dependency_risk=dependency_risk,
            mitigation_strategies=mitigation_strategies
        )
    
    async def _assess_code_risk(self, config: DeploymentConfig) -> float:
        """Assess code-related deployment risk."""
        risk_factors = []
        
        # Check for recent commits
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=24 hours ago"],
                capture_output=True, text=True, cwd=self.workspace_path
            )
            recent_commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            
            # More recent commits = higher risk
            commit_risk = min(1.0, recent_commits / 10.0)
            risk_factors.append(commit_risk)
            
        except Exception as e:
            self.logger.warning(f"Could not assess commit risk: {e}")
            risk_factors.append(0.3)  # Default moderate risk
        
        # Check test coverage
        try:
            # Simulate test coverage check
            test_coverage = 0.85  # Would be retrieved from actual test results
            coverage_risk = max(0.0, 1.0 - (test_coverage / 0.9))  # Risk increases below 90%
            risk_factors.append(coverage_risk)
        except Exception:
            risk_factors.append(0.4)  # Default risk for unknown coverage
        
        # Check code complexity
        complexity_risk = 0.2  # Simulated complexity risk
        risk_factors.append(complexity_risk)
        
        return np.mean(risk_factors)
    
    async def _assess_infrastructure_risk(self, config: DeploymentConfig) -> float:
        """Assess infrastructure-related deployment risk."""
        risk_factors = []
        
        # Check resource availability
        monitoring_data = get_monitoring_dashboard()
        current_cpu = monitoring_data.get("cpu_usage", 0.5)
        current_memory = monitoring_data.get("memory_usage", 0.5)
        
        # Higher resource usage = higher risk
        resource_risk = max(current_cpu, current_memory)
        risk_factors.append(resource_risk)
        
        # Environment-specific risk
        env_risk_map = {
            DeploymentEnvironment.DEVELOPMENT: 0.1,
            DeploymentEnvironment.STAGING: 0.2,
            DeploymentEnvironment.PRODUCTION: 0.4,
            DeploymentEnvironment.CANARY: 0.3
        }
        environment_risk = env_risk_map.get(config.environment, 0.3)
        risk_factors.append(environment_risk)
        
        # Network and storage risk (simulated)
        network_risk = 0.15
        storage_risk = 0.1
        risk_factors.extend([network_risk, storage_risk])
        
        return np.mean(risk_factors)
    
    async def _assess_timing_risk(self, config: DeploymentConfig) -> float:
        """Assess timing-related deployment risk."""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        # Higher risk during business hours and weekdays
        business_hour_risk = 0.7 if 9 <= current_hour <= 17 else 0.2
        weekday_risk = 0.6 if current_day < 5 else 0.3  # Monday=0, Sunday=6
        
        # Check for concurrent deployments
        concurrent_deployments = len(self.active_deployments)
        concurrency_risk = min(1.0, concurrent_deployments / 3.0)
        
        return np.mean([business_hour_risk, weekday_risk, concurrency_risk])
    
    async def _assess_dependency_risk(self, config: DeploymentConfig) -> float:
        """Assess dependency-related deployment risk."""
        risk_factors = []
        
        # Check for external service dependencies
        external_service_risk = 0.25  # Simulated risk from external services
        risk_factors.append(external_service_risk)
        
        # Check database migration risk
        db_migration_risk = 0.1  # Simulated - would check for pending migrations
        risk_factors.append(db_migration_risk)
        
        # Check for breaking changes
        breaking_change_risk = 0.15  # Simulated API compatibility check
        risk_factors.append(breaking_change_risk)
        
        return np.mean(risk_factors)
    
    def _generate_mitigation_strategies(self, overall_score: float, code_risk: float,
                                      infrastructure_risk: float, timing_risk: float,
                                      dependency_risk: float) -> List[str]:
        """Generate mitigation strategies based on risk assessment."""
        strategies = []
        
        if overall_score > 0.7:
            strategies.append("Consider deploying during off-peak hours")
            strategies.append("Use canary deployment strategy")
            strategies.append("Increase monitoring frequency")
        
        if code_risk > 0.5:
            strategies.append("Run additional integration tests")
            strategies.append("Perform manual code review")
            strategies.append("Consider feature flags for new functionality")
        
        if infrastructure_risk > 0.6:
            strategies.append("Scale up infrastructure before deployment")
            strategies.append("Use blue-green deployment for zero downtime")
            strategies.append("Prepare quick rollback procedures")
        
        if timing_risk > 0.5:
            strategies.append("Schedule deployment for low-traffic period")
            strategies.append("Coordinate with other deployment activities")
        
        if dependency_risk > 0.4:
            strategies.append("Verify external service availability")
            strategies.append("Test database migrations in staging")
            strategies.append("Implement graceful degradation")
        
        return strategies
    
    async def _execute_deployment(self, execution: DeploymentExecution):
        """Execute the actual deployment process."""
        execution.status = DeploymentStatus.IN_PROGRESS
        execution.start_time = datetime.now()
        execution.logs.append(f"Starting deployment at {execution.start_time}")
        
        try:
            config = execution.config
            
            # Create rollback plan before deployment
            execution.rollback_plan = await self._create_rollback_plan(config)
            
            # Execute deployment based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(execution)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(execution)
            elif config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                await self._execute_rolling_update(execution)
            else:
                await self._execute_standard_deployment(execution)
            
            # Validate deployment
            execution.status = DeploymentStatus.VALIDATING
            validation_success = await self._validate_deployment(execution)
            
            if validation_success:
                execution.status = DeploymentStatus.COMPLETED
                execution.end_time = datetime.now()
                execution.logs.append(f"Deployment completed successfully at {execution.end_time}")
                
                # Update success memory for learning
                env_key = f"{config.environment.value}_{config.strategy.value}"
                self.deployment_success_memory[env_key] = self.deployment_success_memory.get(env_key, 0.5) * 0.9 + 0.1
                
            else:
                await self._initiate_rollback(execution, "Validation failed")
                
        except Exception as e:
            execution.logs.append(f"Deployment failed: {str(e)}")
            await self._initiate_rollback(execution, str(e))
        
        # Record metrics
        record_metric("deployment.execution", 1)
        record_metric(f"deployment.{execution.config.environment.value}.status.{execution.status.value}", 1)
        if execution.risk_assessment:
            record_metric("deployment.risk_score", execution.risk_assessment.overall_score)
    
    async def _execute_blue_green_deployment(self, execution: DeploymentExecution):
        """Execute blue-green deployment strategy."""
        config = execution.config
        execution.logs.append("Executing blue-green deployment")
        
        # Deploy to green environment
        await self._deploy_to_environment("green", config)
        execution.logs.append("Deployed to green environment")
        
        # Health check green environment
        green_healthy = await self._health_check_environment("green", config)
        if not green_healthy:
            raise Exception("Green environment health check failed")
        
        # Switch traffic to green
        await self._switch_traffic("blue", "green")
        execution.logs.append("Traffic switched to green environment")
        
        # Monitor for issues
        await asyncio.sleep(60)  # Monitor for 1 minute
        
        # Clean up blue environment
        await self._cleanup_environment("blue")
        execution.logs.append("Blue environment cleaned up")
    
    async def _execute_canary_deployment(self, execution: DeploymentExecution):
        """Execute canary deployment strategy."""
        config = execution.config
        traffic_percentage = config.traffic_percentage
        execution.logs.append(f"Executing canary deployment with {traffic_percentage*100}% traffic")
        
        # Deploy canary version
        await self._deploy_to_environment("canary", config)
        execution.logs.append("Deployed to canary environment")
        
        # Gradually increase traffic
        for percentage in [0.05, 0.10, 0.25, 0.50, traffic_percentage]:
            await self._set_traffic_percentage("canary", percentage)
            execution.logs.append(f"Increased canary traffic to {percentage*100}%")
            
            # Monitor metrics
            await asyncio.sleep(120)  # Monitor for 2 minutes
            metrics_healthy = await self._check_canary_metrics(execution)
            
            if not metrics_healthy:
                raise Exception(f"Canary metrics unhealthy at {percentage*100}% traffic")
        
        # Promote canary to production
        await self._promote_canary_to_production(execution)
        execution.logs.append("Canary promoted to production")
    
    async def _execute_rolling_update(self, execution: DeploymentExecution):
        """Execute rolling update deployment strategy."""
        config = execution.config
        execution.logs.append("Executing rolling update deployment")
        
        replicas = config.replicas
        
        # Update replicas one by one
        for i in range(replicas):
            await self._update_replica(i, config)
            execution.logs.append(f"Updated replica {i+1}/{replicas}")
            
            # Health check updated replica
            replica_healthy = await self._health_check_replica(i, config)
            if not replica_healthy:
                raise Exception(f"Replica {i} health check failed")
            
            await asyncio.sleep(30)  # Wait between replica updates
    
    async def _execute_standard_deployment(self, execution: DeploymentExecution):
        """Execute standard deployment strategy."""
        config = execution.config
        execution.logs.append("Executing standard deployment")
        
        # Update all replicas at once
        await self._deploy_to_environment(config.environment.value, config)
        execution.logs.append("Deployed to environment")
        
        # Wait for rollout
        await asyncio.sleep(60)
    
    async def _validate_deployment(self, execution: DeploymentExecution) -> bool:
        """Validate deployment success."""
        config = execution.config
        
        # Health check validation
        health_check_passed = await self._comprehensive_health_check(config)
        if not health_check_passed:
            execution.logs.append("Health check validation failed")
            return False
        
        # Performance validation
        performance_acceptable = await self._validate_performance_metrics(execution)
        if not performance_acceptable:
            execution.logs.append("Performance validation failed")
            return False
        
        # Success criteria validation
        success_criteria_met = await self._validate_success_criteria(execution)
        if not success_criteria_met:
            execution.logs.append("Success criteria validation failed")
            return False
        
        execution.logs.append("All validations passed")
        return True
    
    async def _initiate_rollback(self, execution: DeploymentExecution, reason: str):
        """Initiate deployment rollback."""
        execution.status = DeploymentStatus.ROLLING_BACK
        execution.logs.append(f"Initiating rollback due to: {reason}")
        
        try:
            if execution.rollback_plan:
                await self._execute_rollback_plan(execution.rollback_plan)
                execution.status = DeploymentStatus.ROLLED_BACK
                execution.logs.append("Rollback completed successfully")
            else:
                execution.status = DeploymentStatus.FAILED
                execution.logs.append("No rollback plan available")
                
        except Exception as e:
            execution.status = DeploymentStatus.FAILED
            execution.logs.append(f"Rollback failed: {str(e)}")
        
        # Report failure to resilience framework
        await adaptive_resilience.handle_failure(
            FailureType.SERVICE_UNAVAILABLE,
            {"deployment_id": execution.id, "environment": execution.config.environment.value},
            Exception(reason)
        )
        
        execution.end_time = datetime.now()
    
    async def _monitor_active_deployments(self):
        """Monitor all active deployments for health and progress."""
        for deployment_id, execution in self.active_deployments.items():
            if execution.status in [DeploymentStatus.IN_PROGRESS, DeploymentStatus.VALIDATING]:
                # Check for timeout
                if execution.start_time:
                    elapsed = (datetime.now() - execution.start_time).total_seconds()
                    if elapsed > execution.config.health_check_timeout:
                        await self._initiate_rollback(execution, "Deployment timeout")
                        continue
                
                # Monitor health metrics
                await self._update_health_metrics(execution)
                
                # Check for automatic rollback conditions
                if await self._should_auto_rollback(execution):
                    await self._initiate_rollback(execution, "Automatic rollback triggered")
    
    async def _should_auto_rollback(self, execution: DeploymentExecution) -> bool:
        """Determine if deployment should be automatically rolled back."""
        if execution.config.environment == DeploymentEnvironment.DEVELOPMENT:
            return False  # No auto-rollback in development
        
        # Check error rate threshold
        current_error_rate = execution.health_metrics.get("error_rate", 0.0)
        if current_error_rate > execution.config.rollback_threshold:
            return True
        
        # Check response time degradation
        current_response_time = execution.health_metrics.get("response_time", 0.0)
        baseline_response_time = 1000.0  # Would be retrieved from historical data
        if current_response_time > baseline_response_time * 1.5:  # 50% increase
            return True
        
        # Check availability
        current_availability = execution.health_metrics.get("availability", 1.0)
        if current_availability < 0.99:  # Below 99% availability
            return True
        
        return False
    
    # Placeholder implementations for deployment operations
    # These would be implemented with actual infrastructure APIs (K8s, Docker, etc.)
    
    async def _deploy_to_environment(self, environment: str, config: DeploymentConfig):
        """Deploy to specific environment."""
        await asyncio.sleep(2)  # Simulate deployment time
        self.logger.info(f"Deployed {config.image_tag} to {environment}")
    
    async def _health_check_environment(self, environment: str, config: DeploymentConfig) -> bool:
        """Health check specific environment."""
        await asyncio.sleep(1)  # Simulate health check time
        return True  # Simulate success
    
    async def _switch_traffic(self, from_env: str, to_env: str):
        """Switch traffic between environments."""
        await asyncio.sleep(1)  # Simulate traffic switch
        self.logger.info(f"Switched traffic from {from_env} to {to_env}")
    
    async def _cleanup_environment(self, environment: str):
        """Clean up environment."""
        await asyncio.sleep(1)  # Simulate cleanup
        self.logger.info(f"Cleaned up {environment} environment")
    
    async def _set_traffic_percentage(self, environment: str, percentage: float):
        """Set traffic percentage for environment."""
        await asyncio.sleep(0.5)  # Simulate traffic adjustment
        self.logger.info(f"Set {environment} traffic to {percentage*100}%")
    
    async def _check_canary_metrics(self, execution: DeploymentExecution) -> bool:
        """Check canary deployment metrics."""
        # Simulate metric checking
        error_rate = 0.02  # 2% error rate
        return error_rate < execution.config.rollback_threshold
    
    async def _promote_canary_to_production(self, execution: DeploymentExecution):
        """Promote canary deployment to production."""
        await asyncio.sleep(2)  # Simulate promotion
        self.logger.info("Promoted canary to production")
    
    async def _update_replica(self, replica_id: int, config: DeploymentConfig):
        """Update specific replica."""
        await asyncio.sleep(1)  # Simulate replica update
        self.logger.info(f"Updated replica {replica_id}")
    
    async def _health_check_replica(self, replica_id: int, config: DeploymentConfig) -> bool:
        """Health check specific replica."""
        await asyncio.sleep(0.5)  # Simulate health check
        return True  # Simulate success
    
    async def _comprehensive_health_check(self, config: DeploymentConfig) -> bool:
        """Perform comprehensive health check."""
        await asyncio.sleep(2)  # Simulate comprehensive check
        return True  # Simulate success
    
    async def _validate_performance_metrics(self, execution: DeploymentExecution) -> bool:
        """Validate performance metrics."""
        # Simulate performance validation
        response_time = 150.0  # ms
        return response_time < 500.0  # Acceptable threshold
    
    async def _validate_success_criteria(self, execution: DeploymentExecution) -> bool:
        """Validate custom success criteria."""
        # Check custom success criteria from config
        for metric, threshold in execution.config.success_criteria.items():
            current_value = execution.health_metrics.get(metric, 0.0)
            if current_value < threshold:
                return False
        return True
    
    async def _create_rollback_plan(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create rollback plan for deployment."""
        return {
            "strategy": "revert_to_previous_version",
            "previous_version": "v1.0.0",  # Would be retrieved from deployment history
            "rollback_timeout": 300,
            "cleanup_required": True
        }
    
    async def _execute_rollback_plan(self, rollback_plan: Dict[str, Any]):
        """Execute rollback plan."""
        await asyncio.sleep(3)  # Simulate rollback execution
        self.logger.info("Executed rollback plan")
    
    async def _update_health_metrics(self, execution: DeploymentExecution):
        """Update health metrics for deployment."""
        # Simulate metric collection
        execution.health_metrics.update({
            "error_rate": 0.01,  # 1% error rate
            "response_time": 200.0,  # 200ms average response time
            "availability": 0.999,  # 99.9% availability
            "cpu_usage": 0.4,  # 40% CPU usage
            "memory_usage": 0.6  # 60% memory usage
        })
    
    async def _cleanup_completed_deployments(self):
        """Clean up completed deployments from active tracking."""
        completed_deployments = [
            deployment_id for deployment_id, execution in self.active_deployments.items()
            if execution.status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]
        ]
        
        for deployment_id in completed_deployments:
            execution = self.active_deployments.pop(deployment_id)
            self.deployment_history.append(execution)
            
            # Keep only recent history
            if len(self.deployment_history) > 100:
                self.deployment_history = self.deployment_history[-100:]
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific deployment."""
        execution = self.active_deployments.get(deployment_id)
        if not execution:
            # Check history
            execution = next((e for e in self.deployment_history if e.id == deployment_id), None)
        
        if not execution:
            return None
        
        return {
            "id": execution.id,
            "status": execution.status.value,
            "environment": execution.config.environment.value,
            "strategy": execution.config.strategy.value,
            "start_time": execution.start_time.isoformat() if execution.start_time else None,
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "risk_score": execution.risk_assessment.overall_score if execution.risk_assessment else None,
            "health_metrics": execution.health_metrics,
            "logs": execution.logs[-10:]  # Last 10 log entries
        }
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get overall orchestrator status."""
        return {
            "active_deployments": len(self.active_deployments),
            "deployment_history": len(self.deployment_history),
            "risk_tolerance": self.risk_tolerance,
            "success_rates": {
                env_strategy: rate for env_strategy, rate in self.deployment_success_memory.items()
            },
            "recent_deployments": [
                {
                    "id": execution.id,
                    "status": execution.status.value,
                    "environment": execution.config.environment.value
                }
                for execution in list(self.active_deployments.values())[-5:]
            ]
        }


# Global intelligent deployment orchestrator instance
deployment_orchestrator = IntelligentDeploymentOrchestrator()


async def deploy_with_intelligence(config: DeploymentConfig) -> str:
    """Deploy with intelligent orchestration."""
    return await deployment_orchestrator.deploy(config)


def get_deployment_status(deployment_id: str) -> Optional[Dict[str, Any]]:
    """Get deployment status."""
    return deployment_orchestrator.get_deployment_status(deployment_id)


def get_deployment_orchestrator_status() -> Dict[str, Any]:
    """Get deployment orchestrator status."""
    return deployment_orchestrator.get_orchestrator_status()