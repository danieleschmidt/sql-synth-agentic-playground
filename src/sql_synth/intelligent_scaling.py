"""Intelligent auto-scaling and resource management system.

This module implements advanced auto-scaling algorithms, predictive resource allocation,
and intelligent load balancing for SQL synthesis operations.
"""

import logging
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Auto-scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_CHANGE = "no_change"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU_THREADS = "cpu_threads"
    MEMORY_POOL = "memory_pool"
    CONNECTION_POOL = "connection_pool"
    CACHE_SIZE = "cache_size"
    WORKER_PROCESSES = "worker_processes"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: datetime
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    response_time_p95: float
    throughput_qps: float
    error_rate: float
    active_connections: int
    cache_hit_rate: float
    pending_requests: int


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    resource_type: ResourceType
    min_instances: int
    max_instances: int
    target_utilization: float
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_cooldown_seconds: int
    scale_down_cooldown_seconds: int
    scaling_factor: float = 1.5
    prediction_window_minutes: int = 5
    enable_predictive_scaling: bool = True


@dataclass
class ResourcePool:
    """Dynamic resource pool."""
    pool_type: ResourceType
    current_size: int
    min_size: int
    max_size: int
    utilization: float
    last_scaled: datetime
    pending_operations: int = 0
    active_operations: int = 0


class PredictiveScaler:
    """Predictive scaling using time series analysis."""

    def __init__(self, prediction_horizon_minutes: int = 10):
        self.prediction_horizon = prediction_horizon_minutes
        self.historical_metrics: list[ScalingMetrics] = []
        self.seasonal_patterns: dict[str, list[float]] = {}
        self.trend_coefficients: dict[str, float] = {}

    def predict_resource_demand(self, current_metrics: ScalingMetrics) -> dict[ResourceType, float]:
        """Predict future resource demand using time series analysis."""

        if len(self.historical_metrics) < 10:
            # Not enough data for prediction, return current demand
            return self._current_demand_estimate(current_metrics)

        predictions = {}

        # Predict for each resource type
        for resource_type in ResourceType:
            prediction = self._predict_resource_type_demand(resource_type, current_metrics)
            predictions[resource_type] = prediction

        return predictions

    def _predict_resource_type_demand(self, resource_type: ResourceType,
                                    current_metrics: ScalingMetrics) -> float:
        """Predict demand for specific resource type."""

        # Extract relevant metric based on resource type
        metric_values = self._extract_metric_values(resource_type)

        if len(metric_values) < 5:
            return metric_values[-1] if metric_values else 0.5

        # Time series decomposition
        trend = self._calculate_trend(metric_values)
        seasonal = self._calculate_seasonal_pattern(metric_values)

        # Linear extrapolation with seasonal adjustment
        prediction_steps = self.prediction_horizon
        trend_prediction = metric_values[-1] + trend * prediction_steps

        # Add seasonal component
        current_hour = datetime.now().hour
        seasonal_factor = seasonal.get(str(current_hour % 24), 1.0)

        prediction = trend_prediction * seasonal_factor

        # Apply bounds and smoothing
        prediction = max(0.0, min(1.0, prediction))

        # Exponential smoothing with recent observations
        alpha = 0.3
        return alpha * prediction + (1 - alpha) * metric_values[-1]


    def _extract_metric_values(self, resource_type: ResourceType) -> list[float]:
        """Extract relevant metric values for resource type."""
        values = []

        for metrics in self.historical_metrics[-100:]:  # Last 100 data points
            if resource_type == ResourceType.CPU_THREADS:
                values.append(metrics.cpu_utilization)
            elif resource_type == ResourceType.MEMORY_POOL:
                values.append(metrics.memory_utilization)
            elif resource_type == ResourceType.CONNECTION_POOL:
                values.append(metrics.active_connections / 100.0)  # Normalize
            elif resource_type == ResourceType.CACHE_SIZE:
                values.append(1.0 - metrics.cache_hit_rate)  # Cache miss rate
            elif resource_type == ResourceType.WORKER_PROCESSES:
                values.append(metrics.queue_length / 50.0)  # Normalize queue length

        return values

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate linear trend in time series."""
        if len(values) < 2:
            return 0.0

        x = list(range(len(values)))
        y = values

        n = len(values)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0

        return (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

    def _calculate_seasonal_pattern(self, values: list[float]) -> dict[str, float]:
        """Calculate seasonal pattern (hourly)."""
        if len(values) < 24:
            return {str(i): 1.0 for i in range(24)}

        # Group by hour of day
        hourly_values: dict[int, list[float]] = {i: [] for i in range(24)}

        current_time = datetime.now()
        for i, value in enumerate(reversed(values[-24*7:])):  # Last week
            hour = (current_time - timedelta(hours=i)).hour
            hourly_values[hour].append(value)

        # Calculate average for each hour
        seasonal_pattern = {}
        overall_avg = statistics.mean(values[-24*7:]) if len(values) >= 24*7 else statistics.mean(values)

        for hour in range(24):
            if hourly_values[hour]:
                hour_avg = statistics.mean(hourly_values[hour])
                seasonal_pattern[str(hour)] = hour_avg / overall_avg if overall_avg > 0 else 1.0
            else:
                seasonal_pattern[str(hour)] = 1.0

        return seasonal_pattern

    def _current_demand_estimate(self, metrics: ScalingMetrics) -> dict[ResourceType, float]:
        """Estimate current demand when insufficient historical data."""
        return {
            ResourceType.CPU_THREADS: metrics.cpu_utilization,
            ResourceType.MEMORY_POOL: metrics.memory_utilization,
            ResourceType.CONNECTION_POOL: min(1.0, metrics.active_connections / 100.0),
            ResourceType.CACHE_SIZE: 1.0 - metrics.cache_hit_rate,
            ResourceType.WORKER_PROCESSES: min(1.0, metrics.queue_length / 50.0),
        }

    def add_metrics(self, metrics: ScalingMetrics) -> None:
        """Add metrics to historical data."""
        self.historical_metrics.append(metrics)

        # Keep only recent data
        cutoff_time = datetime.now() - timedelta(days=7)
        self.historical_metrics = [m for m in self.historical_metrics if m.timestamp > cutoff_time]


class IntelligentLoadBalancer:
    """Intelligent load balancing with adaptive algorithms."""

    def __init__(self):
        self.worker_pools: dict[str, ResourcePool] = {}
        self.load_metrics: dict[str, list[float]] = {}
        self.routing_weights: dict[str, float] = {}

    def add_worker_pool(self, pool_id: str, pool: ResourcePool) -> None:
        """Add worker pool to load balancer."""
        self.worker_pools[pool_id] = pool
        self.load_metrics[pool_id] = []
        self.routing_weights[pool_id] = 1.0

    def route_request(self, request_complexity: float = 0.5) -> Optional[str]:
        """Route request to optimal worker pool."""

        if not self.worker_pools:
            return None

        # Calculate routing scores for each pool
        scores = {}
        for pool_id, pool in self.worker_pools.items():
            score = self._calculate_routing_score(pool, request_complexity)
            scores[pool_id] = score

        # Select pool with highest score (least loaded, most suitable)
        best_pool_id = max(scores.keys(), key=lambda pid: scores[pid])

        # Update pool utilization
        self.worker_pools[best_pool_id].pending_operations += 1

        return best_pool_id

    def _calculate_routing_score(self, pool: ResourcePool, request_complexity: float) -> float:
        """Calculate routing score for worker pool."""

        # Base score inversely related to utilization
        utilization_score = 1.0 - pool.utilization

        # Penalize pools with many pending operations
        pending_penalty = pool.pending_operations / max(pool.current_size, 1)
        pending_score = max(0.0, 1.0 - pending_penalty)

        # Capacity score based on available resources
        capacity_ratio = (pool.current_size - pool.active_operations) / max(pool.current_size, 1)
        capacity_score = max(0.0, capacity_ratio)

        # Historical performance score
        pool_id = id(pool)  # Use object id as key
        recent_performance = self._get_recent_performance(str(pool_id))
        performance_score = recent_performance

        # Complexity matching score (favor pools that handle similar complexity well)
        complexity_score = self._calculate_complexity_score(pool, request_complexity)

        # Weighted combination
        return (
            0.4 * utilization_score +
            0.2 * pending_score +
            0.2 * capacity_score +
            0.1 * performance_score +
            0.1 * complexity_score
        )


    def _get_recent_performance(self, pool_id: str) -> float:
        """Get recent performance score for pool."""
        if pool_id not in self.load_metrics or not self.load_metrics[pool_id]:
            return 0.5  # Neutral score for new pools

        recent_metrics = self.load_metrics[pool_id][-10:]  # Last 10 measurements
        return 1.0 - statistics.mean(recent_metrics)  # Convert latency to score

    def _calculate_complexity_score(self, pool: ResourcePool, request_complexity: float) -> float:
        """Calculate how well pool handles given complexity level."""
        # Simplified: assume all pools handle all complexities equally well
        # In practice, you might track complexity-specific performance
        return 1.0

    def update_pool_metrics(self, pool_id: str, response_time: float, success: bool) -> None:
        """Update performance metrics for pool."""
        if pool_id in self.load_metrics:
            # Record response time (normalized)
            normalized_time = min(1.0, response_time / 10.0)  # 10s max
            self.load_metrics[pool_id].append(normalized_time)

            # Keep only recent metrics
            if len(self.load_metrics[pool_id]) > 100:
                self.load_metrics[pool_id].pop(0)

        # Update pool state
        if pool_id in self.worker_pools:
            pool = self.worker_pools[pool_id]
            pool.pending_operations = max(0, pool.pending_operations - 1)

            if success:
                pool.active_operations = max(0, pool.active_operations - 1)


class AutoScaler:
    """Intelligent auto-scaling system."""

    def __init__(self):
        self.scaling_policies: dict[ResourceType, ScalingPolicy] = {}
        self.resource_pools: dict[ResourceType, ResourcePool] = {}
        self.predictive_scaler = PredictiveScaler()
        self.load_balancer = IntelligentLoadBalancer()
        self.scaling_history: list[dict] = []

    def add_scaling_policy(self, policy: ScalingPolicy) -> None:
        """Add auto-scaling policy."""
        self.scaling_policies[policy.resource_type] = policy

        # Initialize resource pool if not exists
        if policy.resource_type not in self.resource_pools:
            self.resource_pools[policy.resource_type] = ResourcePool(
                pool_type=policy.resource_type,
                current_size=policy.min_instances,
                min_size=policy.min_instances,
                max_size=policy.max_instances,
                utilization=0.0,
                last_scaled=datetime.now(),
            )

    def evaluate_scaling(self, metrics: ScalingMetrics) -> dict[ResourceType, ScalingDirection]:
        """Evaluate scaling decisions for all resource types."""

        # Add metrics to predictive scaler
        self.predictive_scaler.add_metrics(metrics)

        scaling_decisions = {}

        for resource_type, policy in self.scaling_policies.items():
            if resource_type not in self.resource_pools:
                continue

            pool = self.resource_pools[resource_type]

            # Check cooldown period
            time_since_last_scale = (datetime.now() - pool.last_scaled).total_seconds()
            if time_since_last_scale < policy.scale_up_cooldown_seconds:
                scaling_decisions[resource_type] = ScalingDirection.NO_CHANGE
                continue

            # Get current utilization for this resource type
            current_utilization = self._get_resource_utilization(resource_type, metrics)

            # Get predictive demand if enabled
            if policy.enable_predictive_scaling:
                predicted_demand = self.predictive_scaler.predict_resource_demand(metrics)
                predicted_utilization = predicted_demand.get(resource_type, current_utilization)

                # Use higher of current and predicted
                effective_utilization = max(current_utilization, predicted_utilization)
            else:
                effective_utilization = current_utilization

            # Make scaling decision
            decision = self._make_scaling_decision(
                effective_utilization, policy, pool, metrics,
            )

            scaling_decisions[resource_type] = decision

            # Record decision
            if decision != ScalingDirection.NO_CHANGE:
                self.scaling_history.append({
                    "timestamp": datetime.now(),
                    "resource_type": resource_type.value,
                    "decision": decision.value,
                    "current_utilization": current_utilization,
                    "predicted_utilization": predicted_demand.get(resource_type, current_utilization) if policy.enable_predictive_scaling else None,
                    "pool_size_before": pool.current_size,
                })

        return scaling_decisions

    def _get_resource_utilization(self, resource_type: ResourceType,
                                metrics: ScalingMetrics) -> float:
        """Get current utilization for specific resource type."""

        if resource_type == ResourceType.CPU_THREADS:
            return metrics.cpu_utilization
        if resource_type == ResourceType.MEMORY_POOL:
            return metrics.memory_utilization
        if resource_type == ResourceType.CONNECTION_POOL:
            return min(1.0, metrics.active_connections / 100.0)
        if resource_type == ResourceType.CACHE_SIZE:
            return 1.0 - metrics.cache_hit_rate  # High cache miss = high utilization
        if resource_type == ResourceType.WORKER_PROCESSES:
            return min(1.0, metrics.queue_length / 50.0)

        return 0.0

    def _make_scaling_decision(self, utilization: float, policy: ScalingPolicy,
                             pool: ResourcePool, metrics: ScalingMetrics) -> ScalingDirection:
        """Make scaling decision based on utilization and policy."""

        # Scale up conditions
        if utilization > policy.scale_up_threshold and pool.current_size < pool.max_size:
            # Additional checks for scale up
            if self._should_scale_up(pool, metrics, policy):
                return ScalingDirection.SCALE_UP

        # Scale down conditions
        elif utilization < policy.scale_down_threshold and pool.current_size > pool.min_size:
            # Additional checks for scale down
            if self._should_scale_down(pool, metrics, policy):
                return ScalingDirection.SCALE_DOWN

        return ScalingDirection.NO_CHANGE

    def _should_scale_up(self, pool: ResourcePool, metrics: ScalingMetrics,
                        policy: ScalingPolicy) -> bool:
        """Additional checks for scaling up."""

        # Check if queue is growing
        queue_pressure = metrics.queue_length > pool.current_size * 2

        # Check if response time is degrading
        response_time_pressure = metrics.response_time_p95 > 5.0  # 5 second threshold

        # Check error rate
        error_rate_pressure = metrics.error_rate > 0.05  # 5% error threshold

        return queue_pressure or response_time_pressure or error_rate_pressure

    def _should_scale_down(self, pool: ResourcePool, metrics: ScalingMetrics,
                          policy: ScalingPolicy) -> bool:
        """Additional checks for scaling down."""

        # Only scale down if consistently low utilization
        time_since_last_scale = (datetime.now() - pool.last_scaled).total_seconds()
        sufficient_cooldown = time_since_last_scale > policy.scale_down_cooldown_seconds

        # Check if system is stable
        low_queue = metrics.queue_length < pool.current_size
        good_response_time = metrics.response_time_p95 < 2.0
        low_error_rate = metrics.error_rate < 0.01

        return sufficient_cooldown and low_queue and good_response_time and low_error_rate

    def apply_scaling_decision(self, resource_type: ResourceType,
                             decision: ScalingDirection) -> bool:
        """Apply scaling decision to resource pool."""

        if resource_type not in self.resource_pools or resource_type not in self.scaling_policies:
            return False

        pool = self.resource_pools[resource_type]
        policy = self.scaling_policies[resource_type]

        if decision == ScalingDirection.SCALE_UP:
            new_size = min(pool.max_size,
                          math.ceil(pool.current_size * policy.scaling_factor))
            pool.current_size = new_size
            pool.last_scaled = datetime.now()

            logger.info(f"Scaled up {resource_type.value} to {new_size} instances")
            return True

        if decision == ScalingDirection.SCALE_DOWN:
            new_size = max(pool.min_size,
                          math.floor(pool.current_size / policy.scaling_factor))
            pool.current_size = new_size
            pool.last_scaled = datetime.now()

            logger.info(f"Scaled down {resource_type.value} to {new_size} instances")
            return True

        return False

    def get_scaling_recommendations(self, metrics: ScalingMetrics) -> dict[str, Any]:
        """Get scaling recommendations and system status."""

        decisions = self.evaluate_scaling(metrics)

        recommendations = {
            "timestamp": datetime.now(),
            "scaling_decisions": {rt.value: decision.value for rt, decision in decisions.items()},
            "resource_status": {},
            "predicted_demand": {},
            "system_health": self._assess_system_health(metrics),
        }

        # Add resource status
        for resource_type, pool in self.resource_pools.items():
            recommendations["resource_status"][resource_type.value] = {
                "current_size": pool.current_size,
                "utilization": pool.utilization,
                "min_size": pool.min_size,
                "max_size": pool.max_size,
                "pending_operations": pool.pending_operations,
                "active_operations": pool.active_operations,
            }

        # Add predicted demand
        predicted_demand = self.predictive_scaler.predict_resource_demand(metrics)
        recommendations["predicted_demand"] = {
            rt.value: demand for rt, demand in predicted_demand.items()
        }

        return recommendations

    def _assess_system_health(self, metrics: ScalingMetrics) -> dict[str, Any]:
        """Assess overall system health."""

        health_score = 1.0
        issues = []

        # CPU health
        if metrics.cpu_utilization > 0.8:
            health_score -= 0.2
            issues.append("High CPU utilization")

        # Memory health
        if metrics.memory_utilization > 0.8:
            health_score -= 0.2
            issues.append("High memory utilization")

        # Response time health
        if metrics.response_time_p95 > 5.0:
            health_score -= 0.3
            issues.append("High response time")

        # Error rate health
        if metrics.error_rate > 0.05:
            health_score -= 0.2
            issues.append("High error rate")

        # Queue health
        if metrics.queue_length > 20:
            health_score -= 0.1
            issues.append("High queue length")

        health_status = "healthy"
        if health_score < 0.7:
            health_status = "degraded"
        if health_score < 0.4:
            health_status = "critical"

        return {
            "health_score": max(0.0, health_score),
            "status": health_status,
            "issues": issues,
        }


# Global auto-scaler instance
_global_autoscaler = None


def get_global_autoscaler() -> AutoScaler:
    """Get global auto-scaler instance."""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = AutoScaler()

        # Initialize with default policies
        default_policies = [
            ScalingPolicy(
                resource_type=ResourceType.CPU_THREADS,
                min_instances=2,
                max_instances=16,
                target_utilization=0.7,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                scale_up_cooldown_seconds=60,
                scale_down_cooldown_seconds=300,
            ),
            ScalingPolicy(
                resource_type=ResourceType.MEMORY_POOL,
                min_instances=1,
                max_instances=8,
                target_utilization=0.6,
                scale_up_threshold=0.8,
                scale_down_threshold=0.2,
                scale_up_cooldown_seconds=120,
                scale_down_cooldown_seconds=600,
            ),
            ScalingPolicy(
                resource_type=ResourceType.CONNECTION_POOL,
                min_instances=5,
                max_instances=50,
                target_utilization=0.7,
                scale_up_threshold=0.8,
                scale_down_threshold=0.4,
                scale_up_cooldown_seconds=30,
                scale_down_cooldown_seconds=180,
            ),
        ]

        for policy in default_policies:
            _global_autoscaler.add_scaling_policy(policy)

    return _global_autoscaler
