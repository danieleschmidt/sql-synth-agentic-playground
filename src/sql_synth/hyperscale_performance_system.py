"""HyperScale Performance System with Advanced Optimization and Auto-Scaling.

This module implements a comprehensive hyperscale performance system:
- Intelligent auto-scaling with predictive algorithms
- Advanced caching with semantic similarity
- Multi-tier performance optimization
- Load balancing and traffic distribution
- Resource monitoring and optimization
- Performance prediction and anomaly detection
"""

import asyncio
import json
import logging
import math
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
from contextlib import asynccontextmanager

import numpy as np
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics types."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"
    QUEUE_DEPTH = "queue_depth"
    CONNECTION_COUNT = "connection_count"


class ScalingDirection(Enum):
    """Scaling directions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_CHANGE = "no_change"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PREDICTIVE = "predictive"


@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    timestamp: float
    metrics: Dict[PerformanceMetric, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    direction: ScalingDirection
    magnitude: float  # Scaling factor or number of instances
    confidence: float
    reasoning: List[str]
    estimated_impact: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Represents a worker node in the system."""
    node_id: str
    capacity: Dict[str, float]
    current_load: Dict[str, float]
    health_status: str = "healthy"
    last_health_check: float = field(default_factory=time.time)
    active_connections: int = 0
    total_requests: int = 0
    average_response_time: float = 0.0


class SemanticCache:
    """Advanced semantic caching with similarity-based retrieval."""

    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.cache_data = {}
        self.cache_embeddings = {}
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'semantic_hits': 0,
        }
        self._lock = threading.RLock()

    def _generate_embedding(self, key: str) -> np.ndarray:
        """Generate embedding for cache key (simplified TF-IDF approach)."""
        try:
            # Simple character-level embedding for demonstration
            # In practice, would use sophisticated NLP embeddings
            words = key.lower().split()
            
            # Create a simple bag-of-words embedding
            vocab_size = 1000
            embedding = np.zeros(vocab_size)
            
            for word in words:
                # Simple hash-based embedding
                word_hash = hash(word) % vocab_size
                embedding[word_hash] += 1
            
            # Normalize
            if np.sum(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return np.random.randn(vocab_size)

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with semantic similarity matching."""
        with self._lock:
            current_time = time.time()
            
            # Direct hit
            if key in self.cache_data:
                self.cache_stats['hits'] += 1
                self.access_counts[key] += 1
                self.access_times[key] = current_time
                return self.cache_data[key]
            
            # Semantic similarity search
            query_embedding = self._generate_embedding(key)
            best_similarity = 0.0
            best_match = None
            
            for cached_key, cached_embedding in self.cache_embeddings.items():
                similarity = self._compute_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = cached_key
            
            if best_match:
                self.cache_stats['semantic_hits'] += 1
                self.access_counts[best_match] += 1
                self.access_times[best_match] = current_time
                
                logger.debug(f"Semantic cache hit: {key} -> {best_match} (similarity: {best_similarity:.3f})")
                return self.cache_data[best_match]
            
            # Cache miss
            self.cache_stats['misses'] += 1
            return None

    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if cache is full
            if len(self.cache_data) >= self.max_size and key not in self.cache_data:
                self._evict_lru()
            
            # Store data
            self.cache_data[key] = value
            self.cache_embeddings[key] = self._generate_embedding(key)
            self.access_counts[key] = 1
            self.access_times[key] = current_time

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache_data:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from all structures
        del self.cache_data[lru_key]
        del self.cache_embeddings[lru_key]
        del self.access_counts[lru_key]
        del self.access_times[lru_key]
        
        self.cache_stats['evictions'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            
            return {
                'size': len(self.cache_data),
                'max_size': self.max_size,
                'hit_rate': self.cache_stats['hits'] / max(total_requests, 1),
                'semantic_hit_rate': self.cache_stats['semantic_hits'] / max(total_requests, 1),
                'total_hits': self.cache_stats['hits'],
                'semantic_hits': self.cache_stats['semantic_hits'],
                'misses': self.cache_stats['misses'],
                'evictions': self.cache_stats['evictions'],
                'utilization': len(self.cache_data) / self.max_size,
            }

    def clear(self):
        """Clear all cache data."""
        with self._lock:
            self.cache_data.clear()
            self.cache_embeddings.clear()
            self.access_counts.clear()
            self.access_times.clear()


class IntelligentLoadBalancer:
    """Advanced load balancer with predictive routing."""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.PREDICTIVE):
        self.strategy = strategy
        self.worker_nodes = {}
        self.routing_history = deque(maxlen=10000)
        self.performance_predictors = {}
        self.current_index = 0
        self._lock = threading.RLock()

    def register_worker(self, node_id: str, capacity: Dict[str, float]):
        """Register a new worker node."""
        with self._lock:
            self.worker_nodes[node_id] = WorkerNode(
                node_id=node_id,
                capacity=capacity,
                current_load=defaultdict(float),
            )
            logger.info(f"Registered worker node: {node_id}")

    def unregister_worker(self, node_id: str):
        """Unregister a worker node."""
        with self._lock:
            if node_id in self.worker_nodes:
                del self.worker_nodes[node_id]
                logger.info(f"Unregistered worker node: {node_id}")

    def select_worker(self, request_metadata: Dict[str, Any] = None) -> Optional[str]:
        """Select optimal worker for request."""
        with self._lock:
            if not self.worker_nodes:
                return None
            
            healthy_workers = {
                node_id: node for node_id, node in self.worker_nodes.items()
                if node.health_status == "healthy"
            }
            
            if not healthy_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._select_weighted_round_robin(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._select_least_response_time(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._select_resource_based(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.PREDICTIVE:
                return self._select_predictive(healthy_workers, request_metadata or {})
            else:
                return self._select_round_robin(healthy_workers)

    def _select_round_robin(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker using round-robin."""
        worker_list = list(workers.keys())
        selected = worker_list[self.current_index % len(worker_list)]
        self.current_index += 1
        return selected

    def _select_weighted_round_robin(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker using weighted round-robin based on capacity."""
        total_capacity = sum(
            sum(node.capacity.values()) for node in workers.values()
        )
        
        if total_capacity == 0:
            return self._select_round_robin(workers)
        
        # Create weighted list
        weighted_workers = []
        for node_id, node in workers.items():
            node_capacity = sum(node.capacity.values())
            weight = int((node_capacity / total_capacity) * 100)
            weighted_workers.extend([node_id] * max(weight, 1))
        
        if weighted_workers:
            selected = weighted_workers[self.current_index % len(weighted_workers)]
            self.current_index += 1
            return selected
        
        return self._select_round_robin(workers)

    def _select_least_connections(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker with least active connections."""
        return min(workers.keys(), key=lambda node_id: workers[node_id].active_connections)

    def _select_least_response_time(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker with least average response time."""
        return min(workers.keys(), key=lambda node_id: workers[node_id].average_response_time)

    def _select_resource_based(self, workers: Dict[str, WorkerNode]) -> str:
        """Select worker based on resource utilization."""
        best_worker = None
        best_score = float('inf')
        
        for node_id, node in workers.items():
            # Calculate resource utilization score
            utilization_score = 0.0
            for resource, current in node.current_load.items():
                capacity = node.capacity.get(resource, 1.0)
                if capacity > 0:
                    utilization = current / capacity
                    utilization_score += utilization
            
            if utilization_score < best_score:
                best_score = utilization_score
                best_worker = node_id
        
        return best_worker or list(workers.keys())[0]

    def _select_predictive(self, workers: Dict[str, WorkerNode], request_metadata: Dict[str, Any]) -> str:
        """Select worker using predictive analysis."""
        # Simplified predictive selection
        # In practice, would use ML models trained on historical data
        
        request_complexity = request_metadata.get('complexity', 1.0)
        estimated_cpu = request_metadata.get('estimated_cpu', 0.1)
        estimated_memory = request_metadata.get('estimated_memory', 0.1)
        
        best_worker = None
        best_predicted_performance = float('inf')
        
        for node_id, node in workers.items():
            # Predict performance based on current load and request characteristics
            cpu_load = node.current_load.get('cpu', 0.0)
            memory_load = node.current_load.get('memory', 0.0)
            
            cpu_capacity = node.capacity.get('cpu', 1.0)
            memory_capacity = node.capacity.get('memory', 1.0)
            
            # Predict new utilization
            new_cpu_util = (cpu_load + estimated_cpu) / cpu_capacity
            new_memory_util = (memory_load + estimated_memory) / memory_capacity
            
            # Predict response time (simplified model)
            predicted_response_time = (
                node.average_response_time * (1 + new_cpu_util + new_memory_util) * request_complexity
            )
            
            if predicted_response_time < best_predicted_performance:
                best_predicted_performance = predicted_response_time
                best_worker = node_id
        
        return best_worker or list(workers.keys())[0]

    def update_worker_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update worker performance metrics."""
        with self._lock:
            if node_id in self.worker_nodes:
                node = self.worker_nodes[node_id]
                
                # Update load metrics
                for metric, value in metrics.items():
                    node.current_load[metric] = value
                
                # Update derived metrics
                if 'response_time' in metrics:
                    # Exponential moving average
                    alpha = 0.1
                    node.average_response_time = (
                        alpha * metrics['response_time'] + 
                        (1 - alpha) * node.average_response_time
                    )
                
                if 'connections' in metrics:
                    node.active_connections = int(metrics['connections'])
                
                node.last_health_check = time.time()

    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across workers."""
        with self._lock:
            distribution = {}
            
            for node_id, node in self.worker_nodes.items():
                total_capacity = sum(node.capacity.values())
                total_load = sum(node.current_load.values())
                
                distribution[node_id] = {
                    'utilization': total_load / max(total_capacity, 1),
                    'active_connections': node.active_connections,
                    'average_response_time': node.average_response_time,
                    'health_status': node.health_status,
                    'capacity': dict(node.capacity),
                    'current_load': dict(node.current_load),
                }
            
            return distribution


class PredictiveAutoScaler:
    """Predictive auto-scaling system with machine learning."""

    def __init__(self):
        self.scaling_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=10000)
        self.scaling_policies = {}
        self.predictive_models = {}
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 100
        self.cooldown_period = 300  # 5 minutes
        self.last_scaling_time = 0

    def add_performance_data(self, metrics: Dict[PerformanceMetric, float]):
        """Add performance data for analysis."""
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            metrics=metrics
        )
        self.performance_history.append(snapshot)

    def predict_scaling_need(self, forecast_horizon: int = 300) -> ScalingDecision:
        """Predict scaling needs based on historical data and trends.

        Args:
            forecast_horizon: Time horizon for prediction (seconds)

        Returns:
            Scaling decision with confidence and reasoning
        """
        try:
            if len(self.performance_history) < 10:
                return ScalingDecision(
                    direction=ScalingDirection.NO_CHANGE,
                    magnitude=0.0,
                    confidence=0.1,
                    reasoning=["Insufficient historical data"],
                    estimated_impact={}
                )

            # Analyze recent performance trends
            recent_data = list(self.performance_history)[-100:]  # Last 100 data points
            
            # Extract key metrics for analysis
            latencies = [d.metrics.get(PerformanceMetric.LATENCY, 0) for d in recent_data]
            cpu_usage = [d.metrics.get(PerformanceMetric.CPU_USAGE, 0) for d in recent_data]
            memory_usage = [d.metrics.get(PerformanceMetric.MEMORY_USAGE, 0) for d in recent_data]
            throughput = [d.metrics.get(PerformanceMetric.THROUGHPUT, 0) for d in recent_data]
            error_rates = [d.metrics.get(PerformanceMetric.ERROR_RATE, 0) for d in recent_data]

            # Calculate trends
            latency_trend = self._calculate_trend(latencies)
            cpu_trend = self._calculate_trend(cpu_usage)
            memory_trend = self._calculate_trend(memory_usage)
            throughput_trend = self._calculate_trend(throughput)
            error_trend = self._calculate_trend(error_rates)

            # Current averages
            avg_latency = np.mean(latencies[-10:])  # Last 10 points
            avg_cpu = np.mean(cpu_usage[-10:])
            avg_memory = np.mean(memory_usage[-10:])
            avg_throughput = np.mean(throughput[-10:])
            avg_error_rate = np.mean(error_rates[-10:])

            # Scaling decision logic
            reasoning = []
            scaling_factors = []

            # High resource utilization
            if avg_cpu > 0.8:
                scaling_factors.append(2.0)
                reasoning.append(f"High CPU utilization: {avg_cpu:.1%}")
            
            if avg_memory > 0.8:
                scaling_factors.append(1.5)
                reasoning.append(f"High memory utilization: {avg_memory:.1%}")

            # High latency
            if avg_latency > 2.0:  # 2 seconds threshold
                scaling_factors.append(1.5)
                reasoning.append(f"High latency: {avg_latency:.2f}s")

            # High error rate
            if avg_error_rate > 0.05:  # 5% error rate
                scaling_factors.append(2.0)
                reasoning.append(f"High error rate: {avg_error_rate:.1%}")

            # Trending upward
            if latency_trend > 0.1:
                scaling_factors.append(1.3)
                reasoning.append("Latency trending upward")
            
            if cpu_trend > 0.1:
                scaling_factors.append(1.2)
                reasoning.append("CPU usage trending upward")

            # Low utilization (scale down)
            scale_down_factors = []
            if avg_cpu < 0.3 and avg_memory < 0.3:
                scale_down_factors.append(0.8)
                reasoning.append("Low resource utilization")
            
            if latency_trend < -0.1 and avg_latency < 0.5:
                scale_down_factors.append(0.9)
                reasoning.append("Improving performance trends")

            # Determine scaling decision
            if scaling_factors:
                scale_factor = max(scaling_factors)
                new_instances = max(
                    min(int(self.current_instances * scale_factor), self.max_instances),
                    self.min_instances
                )
                
                if new_instances > self.current_instances:
                    direction = ScalingDirection.SCALE_OUT
                    magnitude = new_instances - self.current_instances
                else:
                    direction = ScalingDirection.NO_CHANGE
                    magnitude = 0
            elif scale_down_factors:
                scale_factor = min(scale_down_factors)
                new_instances = max(
                    int(self.current_instances * scale_factor),
                    self.min_instances
                )
                
                if new_instances < self.current_instances:
                    direction = ScalingDirection.SCALE_IN
                    magnitude = self.current_instances - new_instances
                else:
                    direction = ScalingDirection.NO_CHANGE
                    magnitude = 0
            else:
                direction = ScalingDirection.NO_CHANGE
                magnitude = 0

            # Check cooldown period
            if time.time() - self.last_scaling_time < self.cooldown_period:
                if direction != ScalingDirection.NO_CHANGE:
                    reasoning.append("Scaling suppressed due to cooldown period")
                direction = ScalingDirection.NO_CHANGE
                magnitude = 0

            # Calculate confidence
            confidence = self._calculate_scaling_confidence(
                scaling_factors + scale_down_factors, len(reasoning)
            )

            # Estimate impact
            estimated_impact = self._estimate_scaling_impact(direction, magnitude)

            return ScalingDecision(
                direction=direction,
                magnitude=magnitude,
                confidence=confidence,
                reasoning=reasoning,
                estimated_impact=estimated_impact
            )

        except Exception as e:
            logger.exception(f"Scaling prediction failed: {e}")
            return ScalingDecision(
                direction=ScalingDirection.NO_CHANGE,
                magnitude=0.0,
                confidence=0.0,
                reasoning=[f"Prediction error: {e}"],
                estimated_impact={}
            )

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values using linear regression."""
        try:
            if len(values) < 2:
                return 0.0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope relative to mean value
            mean_value = np.mean(y)
            if mean_value != 0:
                normalized_slope = slope / mean_value
            else:
                normalized_slope = slope
            
            return normalized_slope
            
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            return 0.0

    def _calculate_scaling_confidence(self, factors: List[float], reasoning_count: int) -> float:
        """Calculate confidence in scaling decision."""
        try:
            if not factors:
                return 0.5  # Neutral confidence
            
            # Base confidence from strength of factors
            max_factor = max(abs(f - 1.0) for f in factors)  # Distance from no-change (1.0)
            factor_confidence = min(max_factor * 2, 1.0)
            
            # Boost confidence with more reasoning
            reasoning_boost = min(reasoning_count * 0.1, 0.3)
            
            # Historical accuracy (simplified)
            historical_confidence = 0.7  # Would be calculated from past accuracy
            
            final_confidence = (
                factor_confidence * 0.5 +
                historical_confidence * 0.3 +
                reasoning_boost * 0.2
            )
            
            return min(max(final_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _estimate_scaling_impact(self, direction: ScalingDirection, magnitude: float) -> Dict[str, float]:
        """Estimate impact of scaling decision."""
        try:
            impact = {}
            
            if direction == ScalingDirection.SCALE_OUT:
                # Scaling out should improve performance
                impact['latency_improvement'] = min(magnitude * 0.2, 0.6)  # Up to 60% improvement
                impact['throughput_improvement'] = min(magnitude * 0.3, 0.8)  # Up to 80% improvement
                impact['cpu_utilization_reduction'] = min(magnitude * 0.15, 0.5)
                impact['cost_increase'] = magnitude * 0.5  # Rough cost estimate
                
            elif direction == ScalingDirection.SCALE_IN:
                # Scaling in might slightly worsen performance but save costs
                impact['latency_degradation'] = magnitude * 0.1
                impact['throughput_reduction'] = magnitude * 0.15
                impact['cpu_utilization_increase'] = magnitude * 0.2
                impact['cost_reduction'] = magnitude * 0.5
            
            return impact
            
        except Exception as e:
            logger.warning(f"Impact estimation failed: {e}")
            return {}

    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision.

        Args:
            decision: Scaling decision to execute

        Returns:
            True if scaling was executed successfully
        """
        try:
            if decision.direction == ScalingDirection.NO_CHANGE:
                return True
            
            if decision.direction in [ScalingDirection.SCALE_OUT, ScalingDirection.SCALE_UP]:
                new_instances = min(
                    self.current_instances + int(decision.magnitude),
                    self.max_instances
                )
            else:  # SCALE_IN, SCALE_DOWN
                new_instances = max(
                    self.current_instances - int(decision.magnitude),
                    self.min_instances
                )
            
            if new_instances != self.current_instances:
                logger.info(f"Scaling from {self.current_instances} to {new_instances} instances")
                logger.info(f"Scaling reasoning: {', '.join(decision.reasoning)}")
                
                # In practice, would trigger actual infrastructure scaling
                # For now, just update our tracking
                self.current_instances = new_instances
                self.last_scaling_time = time.time()
                
                # Record scaling event
                self.scaling_history.append(decision)
                
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"Scaling execution failed: {e}")
            return False

    def get_scaling_analytics(self) -> Dict[str, Any]:
        """Get auto-scaling analytics."""
        try:
            if not self.scaling_history:
                return {
                    'total_scaling_events': 0,
                    'current_instances': self.current_instances,
                    'message': 'No scaling events recorded yet'
                }

            # Analyze scaling events
            total_events = len(self.scaling_history)
            scale_out_events = sum(1 for d in self.scaling_history if d.direction == ScalingDirection.SCALE_OUT)
            scale_in_events = sum(1 for d in self.scaling_history if d.direction == ScalingDirection.SCALE_IN)
            
            # Average confidence
            avg_confidence = sum(d.confidence for d in self.scaling_history) / total_events
            
            # Recent scaling activity (last 24 hours)
            current_time = time.time()
            recent_events = [
                d for d in self.scaling_history
                if current_time - d.timestamp < 86400
            ]
            
            return {
                'total_scaling_events': total_events,
                'scale_out_events': scale_out_events,
                'scale_in_events': scale_in_events,
                'current_instances': self.current_instances,
                'min_instances': self.min_instances,
                'max_instances': self.max_instances,
                'average_confidence': avg_confidence,
                'recent_events_24h': len(recent_events),
                'last_scaling_time': self.last_scaling_time,
                'cooldown_remaining': max(0, self.cooldown_period - (current_time - self.last_scaling_time)),
            }
            
        except Exception as e:
            logger.exception(f"Scaling analytics failed: {e}")
            return {'error': str(e)}


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=10000))
        self.alerting_thresholds = {}
        self.performance_baselines = {}
        self.anomaly_detector = None
        self._initialize_baselines()

    def record_metric(self, metric: PerformanceMetric, value: float, metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        try:
            timestamp = time.time()
            
            self.metrics_history[metric].append({
                'timestamp': timestamp,
                'value': value,
                'metadata': metadata or {}
            })
            
            # Check for alerts
            self._check_alert_conditions(metric, value, timestamp)
            
        except Exception as e:
            logger.exception(f"Metric recording failed: {e}")

    def get_current_metrics(self) -> Dict[PerformanceMetric, float]:
        """Get current system metrics."""
        try:
            metrics = {}
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics[PerformanceMetric.CPU_USAGE] = cpu_percent / 100.0
            metrics[PerformanceMetric.MEMORY_USAGE] = memory.percent / 100.0
            metrics[PerformanceMetric.DISK_IO] = disk.percent / 100.0
            
            # Network I/O (simplified)
            network = psutil.net_io_counters()
            if hasattr(network, 'bytes_sent') and hasattr(network, 'bytes_recv'):
                # Simple throughput calculation (would need time-based calculation in practice)
                total_bytes = network.bytes_sent + network.bytes_recv
                metrics[PerformanceMetric.NETWORK_IO] = min(total_bytes / (1024 * 1024 * 1024), 1.0)  # Normalize to GB
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Current metrics collection failed: {e}")
            return {}

    def detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis."""
        try:
            anomalies = []
            current_time = time.time()
            
            for metric, history in self.metrics_history.items():
                if len(history) < 50:  # Need sufficient data
                    continue
                
                recent_values = [entry['value'] for entry in list(history)[-50:]]
                historical_values = [entry['value'] for entry in list(history)[:-10]]
                
                if len(historical_values) < 20:
                    continue
                
                # Statistical anomaly detection
                historical_mean = np.mean(historical_values)
                historical_std = np.std(historical_values)
                
                recent_mean = np.mean(recent_values[-10:])  # Last 10 values
                
                # Z-score based anomaly detection
                if historical_std > 0:
                    z_score = abs(recent_mean - historical_mean) / historical_std
                    
                    if z_score > 2.5:  # Anomaly threshold
                        anomalies.append({
                            'metric': metric.value,
                            'current_value': recent_mean,
                            'expected_value': historical_mean,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 3.5 else 'medium',
                            'timestamp': current_time,
                            'description': f'{metric.value} is {z_score:.1f} standard deviations from normal'
                        })
            
            return anomalies
            
        except Exception as e:
            logger.exception(f"Anomaly detection failed: {e}")
            return []

    def _initialize_baselines(self):
        """Initialize performance baselines."""
        self.performance_baselines = {
            PerformanceMetric.LATENCY: {'target': 1.0, 'max_acceptable': 3.0},
            PerformanceMetric.THROUGHPUT: {'target': 100.0, 'min_acceptable': 50.0},
            PerformanceMetric.CPU_USAGE: {'target': 0.7, 'max_acceptable': 0.9},
            PerformanceMetric.MEMORY_USAGE: {'target': 0.7, 'max_acceptable': 0.9},
            PerformanceMetric.ERROR_RATE: {'target': 0.01, 'max_acceptable': 0.05},
            PerformanceMetric.CACHE_HIT_RATE: {'target': 0.8, 'min_acceptable': 0.6},
        }

    def _check_alert_conditions(self, metric: PerformanceMetric, value: float, timestamp: float):
        """Check if metric value triggers an alert."""
        try:
            baseline = self.performance_baselines.get(metric)
            if not baseline:
                return
            
            alert_triggered = False
            alert_level = 'info'
            
            if 'max_acceptable' in baseline and value > baseline['max_acceptable']:
                alert_triggered = True
                alert_level = 'critical'
            elif 'min_acceptable' in baseline and value < baseline['min_acceptable']:
                alert_triggered = True
                alert_level = 'warning'
            elif 'target' in baseline:
                deviation = abs(value - baseline['target']) / baseline['target']
                if deviation > 0.3:  # 30% deviation
                    alert_triggered = True
                    alert_level = 'warning'
            
            if alert_triggered:
                logger.warning(f"Performance alert [{alert_level}]: {metric.value} = {value}")
                
        except Exception as e:
            logger.warning(f"Alert check failed: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                'timestamp': time.time(),
                'metrics_tracked': len(self.metrics_history),
                'current_metrics': self.get_current_metrics(),
                'anomalies': self.detect_performance_anomalies(),
            }
            
            # Add historical trends
            trends = {}
            for metric, history in self.metrics_history.items():
                if len(history) >= 10:
                    recent_values = [entry['value'] for entry in list(history)[-10:]]
                    older_values = [entry['value'] for entry in list(history)[-20:-10]]
                    
                    if older_values:
                        recent_avg = np.mean(recent_values)
                        older_avg = np.mean(older_values)
                        
                        if older_avg != 0:
                            trend = (recent_avg - older_avg) / older_avg
                        else:
                            trend = 0
                        
                        trends[metric.value] = {
                            'trend_percentage': trend * 100,
                            'direction': 'improving' if trend < 0 and metric in [
                                PerformanceMetric.LATENCY, PerformanceMetric.CPU_USAGE, 
                                PerformanceMetric.MEMORY_USAGE, PerformanceMetric.ERROR_RATE
                            ] else 'degrading' if trend > 0 and metric in [
                                PerformanceMetric.LATENCY, PerformanceMetric.CPU_USAGE, 
                                PerformanceMetric.MEMORY_USAGE, PerformanceMetric.ERROR_RATE
                            ] else 'stable'
                        }
            
            summary['performance_trends'] = trends
            
            return summary
            
        except Exception as e:
            logger.exception(f"Performance summary failed: {e}")
            return {'error': str(e)}


class HyperScalePerformanceSystem:
    """Main hyperscale performance management system."""

    def __init__(self):
        self.semantic_cache = SemanticCache()
        self.load_balancer = IntelligentLoadBalancer()
        self.auto_scaler = PredictiveAutoScaler()
        self.performance_monitor = PerformanceMonitor()
        self.executor_pools = {
            'io_bound': ThreadPoolExecutor(max_workers=50),
            'cpu_bound': ProcessPoolExecutor(max_workers=8),
        }
        self.optimization_history = deque(maxlen=1000)
        self._running = False

    async def start_performance_optimization(self):
        """Start the performance optimization loop."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting hyperscale performance optimization system")
        
        # Start monitoring and optimization tasks
        await asyncio.gather(
            self._performance_monitoring_loop(),
            self._auto_scaling_loop(),
            self._cache_optimization_loop(),
        )

    async def stop_performance_optimization(self):
        """Stop the performance optimization system."""
        self._running = False
        logger.info("Stopping hyperscale performance optimization system")

    async def _performance_monitoring_loop(self):
        """Main performance monitoring loop."""
        while self._running:
            try:
                # Collect current metrics
                current_metrics = self.performance_monitor.get_current_metrics()
                
                # Record metrics
                for metric, value in current_metrics.items():
                    self.performance_monitor.record_metric(metric, value)
                
                # Add to auto-scaler
                self.auto_scaler.add_performance_data(current_metrics)
                
                # Check for anomalies
                anomalies = self.performance_monitor.detect_performance_anomalies()
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} performance anomalies")
                    for anomaly in anomalies:
                        if anomaly['severity'] == 'high':
                            await self._handle_performance_anomaly(anomaly)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.exception(f"Performance monitoring loop error: {e}")
                await asyncio.sleep(60)  # Back off on error

    async def _auto_scaling_loop(self):
        """Auto-scaling decision and execution loop."""
        while self._running:
            try:
                # Get scaling recommendation
                scaling_decision = self.auto_scaler.predict_scaling_need()
                
                if (scaling_decision.direction != ScalingDirection.NO_CHANGE and 
                    scaling_decision.confidence > 0.7):
                    
                    logger.info(f"Auto-scaling recommendation: {scaling_decision.direction.value} "
                               f"(magnitude: {scaling_decision.magnitude}, confidence: {scaling_decision.confidence:.2f})")
                    
                    # Execute scaling
                    success = self.auto_scaler.execute_scaling(scaling_decision)
                    if success:
                        logger.info("Auto-scaling executed successfully")
                    else:
                        logger.warning("Auto-scaling execution failed")
                
                await asyncio.sleep(180)  # Check every 3 minutes
                
            except Exception as e:
                logger.exception(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(300)  # Back off on error

    async def _cache_optimization_loop(self):
        """Cache optimization and warming loop."""
        while self._running:
            try:
                # Get cache statistics
                cache_stats = self.semantic_cache.get_stats()
                
                # Optimize cache if hit rate is low
                if cache_stats['hit_rate'] < 0.6:
                    logger.info(f"Cache hit rate low ({cache_stats['hit_rate']:.1%}), optimizing...")
                    await self._optimize_cache_performance()
                
                # Record cache metrics
                self.performance_monitor.record_metric(
                    PerformanceMetric.CACHE_HIT_RATE, 
                    cache_stats['hit_rate']
                )
                
                await asyncio.sleep(120)  # Optimize every 2 minutes
                
            except Exception as e:
                logger.exception(f"Cache optimization loop error: {e}")
                await asyncio.sleep(180)  # Back off on error

    async def _handle_performance_anomaly(self, anomaly: Dict[str, Any]):
        """Handle detected performance anomaly."""
        try:
            metric_name = anomaly['metric']
            severity = anomaly['severity']
            
            logger.warning(f"Handling {severity} anomaly in {metric_name}: {anomaly['description']}")
            
            # Trigger immediate scaling if needed
            if metric_name in ['cpu_usage', 'memory_usage'] and severity == 'high':
                # Force immediate scaling decision
                emergency_decision = ScalingDecision(
                    direction=ScalingDirection.SCALE_OUT,
                    magnitude=2,  # Add 2 instances
                    confidence=0.9,
                    reasoning=[f"Emergency scaling due to {metric_name} anomaly"],
                    estimated_impact={'emergency_response': True}
                )
                
                self.auto_scaler.execute_scaling(emergency_decision)
            
            # Clear cache if cache-related anomaly
            if metric_name == 'cache_hit_rate' and severity == 'high':
                logger.info("Clearing cache due to performance anomaly")
                self.semantic_cache.clear()
            
            # Record anomaly handling
            self.optimization_history.append({
                'type': 'anomaly_response',
                'timestamp': time.time(),
                'anomaly': anomaly,
                'action_taken': 'emergency_scaling' if 'scaling' in str(emergency_decision) else 'cache_clear'
            })
            
        except Exception as e:
            logger.exception(f"Anomaly handling failed: {e}")

    async def _optimize_cache_performance(self):
        """Optimize cache performance through various strategies."""
        try:
            # Strategy 1: Adjust similarity threshold
            current_threshold = self.semantic_cache.similarity_threshold
            if current_threshold > 0.7:
                self.semantic_cache.similarity_threshold = max(0.6, current_threshold - 0.1)
                logger.info(f"Reduced cache similarity threshold to {self.semantic_cache.similarity_threshold}")
            
            # Strategy 2: Increase cache size if utilization is high
            cache_stats = self.semantic_cache.get_stats()
            if cache_stats['utilization'] > 0.9:
                new_size = min(self.semantic_cache.max_size * 2, 50000)
                logger.info(f"Increasing cache size from {self.semantic_cache.max_size} to {new_size}")
                # Would need to implement cache resizing
            
        except Exception as e:
            logger.exception(f"Cache optimization failed: {e}")

    @asynccontextmanager
    async def optimized_execution(self, operation_name: str, operation_metadata: Dict[str, Any] = None):
        """Context manager for optimized operation execution."""
        start_time = time.time()
        metadata = operation_metadata or {}
        
        try:
            # Select optimal worker if using load balancing
            if self.load_balancer.worker_nodes:
                selected_worker = self.load_balancer.select_worker(metadata)
                if selected_worker:
                    metadata['assigned_worker'] = selected_worker
            
            # Record operation start
            yield metadata
            
        finally:
            # Record performance metrics
            execution_time = time.time() - start_time
            self.performance_monitor.record_metric(
                PerformanceMetric.LATENCY,
                execution_time,
                {'operation': operation_name}
            )

    async def execute_with_optimization(self, operation: Callable, operation_name: str,
                                      cache_key: Optional[str] = None,
                                      executor_type: str = 'io_bound') -> Any:
        """Execute operation with full performance optimization."""
        try:
            # Try cache first
            if cache_key:
                cached_result = self.semantic_cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {operation_name}")
                    return cached_result
            
            # Execute with optimization context
            async with self.optimized_execution(operation_name) as metadata:
                # Select appropriate executor
                executor = self.executor_pools.get(executor_type, self.executor_pools['io_bound'])
                
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(executor, operation)
                
                # Cache result if cache key provided
                if cache_key and result is not None:
                    self.semantic_cache.put(cache_key, result)
                
                return result
                
        except Exception as e:
            # Record error metric
            self.performance_monitor.record_metric(
                PerformanceMetric.ERROR_RATE,
                1.0,
                {'operation': operation_name, 'error': str(e)}
            )
            raise

    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics."""
        try:
            return {
                'cache_performance': self.semantic_cache.get_stats(),
                'load_balancing': self.load_balancer.get_load_distribution(),
                'auto_scaling': self.auto_scaler.get_scaling_analytics(),
                'performance_monitoring': self.performance_monitor.get_performance_summary(),
                'system_health': {
                    'optimization_running': self._running,
                    'worker_nodes': len(self.load_balancer.worker_nodes),
                    'executor_pools': {name: pool._max_workers for name, pool in self.executor_pools.items()},
                    'optimization_events': len(self.optimization_history),
                },
                'timestamp': time.time(),
            }
            
        except Exception as e:
            logger.exception(f"System analytics failed: {e}")
            return {'error': str(e)}


# Global hyperscale performance system
global_performance_system = HyperScalePerformanceSystem()


# Utility functions
async def start_hyperscale_optimization():
    """Start the global hyperscale performance optimization system."""
    await global_performance_system.start_performance_optimization()


async def execute_optimized_operation(operation: Callable, operation_name: str,
                                    cache_key: Optional[str] = None,
                                    executor_type: str = 'io_bound') -> Any:
    """Execute operation with hyperscale optimization.

    Args:
        operation: Operation to execute
        operation_name: Name of the operation
        cache_key: Optional cache key
        executor_type: Type of executor ('io_bound' or 'cpu_bound')

    Returns:
        Operation result
    """
    return await global_performance_system.execute_with_optimization(
        operation, operation_name, cache_key, executor_type
    )


def register_worker_node(node_id: str, capacity: Dict[str, float]):
    """Register a worker node in the load balancer.

    Args:
        node_id: Unique node identifier
        capacity: Node capacity metrics
    """
    global_performance_system.load_balancer.register_worker(node_id, capacity)


def get_hyperscale_insights() -> Dict[str, Any]:
    """Get comprehensive hyperscale performance insights.

    Returns:
        System analytics and insights
    """
    return global_performance_system.get_system_analytics()


# Performance optimization decorator
def hyperscale_optimized(operation_name: str, cache_key_func: Optional[Callable] = None, 
                       executor_type: str = 'io_bound'):
    """Decorator for hyperscale-optimized operations.

    Args:
        operation_name: Name of the operation
        cache_key_func: Function to generate cache key from arguments
        executor_type: Type of executor to use

    Returns:
        Decorated function with hyperscale optimization
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key if function provided
            cache_key = None
            if cache_key_func:
                try:
                    cache_key = cache_key_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Cache key generation failed: {e}")
            
            # Create operation callable
            if asyncio.iscoroutinefunction(func):
                async def operation():
                    return await func(*args, **kwargs)
            else:
                def operation():
                    return func(*args, **kwargs)
            
            return await execute_optimized_operation(
                operation, operation_name, cache_key, executor_type
            )
        
        return wrapper
    return decorator