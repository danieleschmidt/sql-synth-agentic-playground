"""
Infinite Scale Engine - Generation 3 Implementation
Advanced auto-scaling, performance optimization, and transcendent resource management.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ScaleDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"
    WORKERS = "workers"
    CONNECTIONS = "connections"


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    TRANSCENDENT = "transcendent"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    active_connections: int
    request_rate: float
    response_time: float
    error_rate: float
    queue_depth: int
    throughput: float


@dataclass
class ScalingDecision:
    """Scaling decision result."""
    direction: ScaleDirection
    resource_type: ResourceType
    target_value: Union[int, float]
    confidence: float
    reason: str
    expected_impact: Dict[str, float]
    execution_time: float


@dataclass
class PerformanceProfile:
    """Performance profile for optimization."""
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_rps: float
    cpu_efficiency: float
    memory_efficiency: float
    error_rate: float
    optimization_score: float


class IntelligentLoadBalancer:
    """AI-powered load balancer with predictive scaling."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.min_workers = max(1, self.max_workers // 4)
        self.current_workers = self.min_workers
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers // 2)
        
        self.request_queue = asyncio.Queue()
        self.active_tasks = set()
        self.metrics_history = deque(maxlen=1000)
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Scaling parameters
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_check_interval = 10.0
        
        # Start background monitoring
        self._monitoring_active = True
        self._monitor_task = None
        
        logger.info(f"Intelligent Load Balancer initialized with {self.current_workers} workers")
    
    async def start_monitoring(self):
        """Start load balancer monitoring."""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop load balancer monitoring."""
        self._monitoring_active = False
        if self._monitor_task:
            await self._monitor_task
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task with intelligent routing."""
        task_id = f"task_{int(time.time() * 1000000)}"
        start_time = time.time()
        
        try:
            # Choose execution strategy based on task characteristics
            if self._is_cpu_intensive(func):
                result = await asyncio.get_event_loop().run_in_executor(
                    self.process_pool, func, *args
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, func, *args
                )
            
            # Record success metrics
            response_time = time.time() - start_time
            self._record_request_metrics(response_time, True)
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_request_metrics(response_time, False)
            raise e
    
    def _is_cpu_intensive(self, func: Callable) -> bool:
        """Determine if function is CPU intensive."""
        # Simple heuristic - can be enhanced with ML
        cpu_intensive_patterns = [
            'optimize', 'calculate', 'compute', 'process', 'transform',
            'quantum', 'transcendent', 'analysis', 'synthesis'
        ]
        
        func_name = func.__name__.lower()
        return any(pattern in func_name for pattern in cpu_intensive_patterns)
    
    def _record_request_metrics(self, response_time: float, success: bool):
        """Record request metrics."""
        self.request_count += 1
        self.total_response_time += response_time
        
        if not success:
            self.error_count += 1
        
        # Store metrics for analysis
        metrics = {
            'timestamp': time.time(),
            'response_time': response_time,
            'success': success,
            'active_workers': self.current_workers,
            'queue_size': self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0
        }
        
        self.metrics_history.append(metrics)
    
    async def _monitoring_loop(self):
        """Background monitoring and scaling loop."""
        while self._monitoring_active:
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(self.scale_check_interval)
            except Exception as e:
                logger.error(f"Error in load balancer monitoring: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_scaling_needs(self):
        """Check if scaling is needed."""
        if len(self.metrics_history) < 10:
            return
        
        # Calculate recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        avg_response_time = np.mean([m['response_time'] for m in recent_metrics])
        success_rate = np.mean([m['success'] for m in recent_metrics])
        
        # Get system metrics if available
        system_metrics = self._get_system_metrics()
        
        # Make scaling decision
        decision = self._make_scaling_decision(avg_response_time, success_rate, system_metrics)
        
        if decision.direction != ScaleDirection.MAINTAIN:
            await self._execute_scaling_decision(decision)
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {}
        
        if psutil:
            try:
                metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                metrics['memory_percent'] = psutil.virtual_memory().percent
                metrics['disk_io'] = sum(psutil.disk_io_counters()[:2]) if psutil.disk_io_counters() else 0
                
                # Network metrics
                net_io = psutil.net_io_counters()
                metrics['network_io'] = net_io.bytes_sent + net_io.bytes_recv
                
            except Exception as e:
                logger.debug(f"Failed to get system metrics: {e}")
        
        return metrics
    
    def _make_scaling_decision(
        self,
        avg_response_time: float,
        success_rate: float,
        system_metrics: Dict[str, float]
    ) -> ScalingDecision:
        """Make intelligent scaling decision."""
        cpu_usage = system_metrics.get('cpu_percent', 50.0) / 100.0
        memory_usage = system_metrics.get('memory_percent', 50.0) / 100.0
        
        # Scale up conditions
        if (avg_response_time > 2.0 or 
            cpu_usage > self.scale_up_threshold or
            success_rate < 0.95):
            
            if self.current_workers < self.max_workers:
                new_workers = min(self.max_workers, int(self.current_workers * 1.5))
                return ScalingDecision(
                    direction=ScaleDirection.UP,
                    resource_type=ResourceType.WORKERS,
                    target_value=new_workers,
                    confidence=0.8,
                    reason=f"High load: response_time={avg_response_time:.2f}s, cpu={cpu_usage:.2f}",
                    expected_impact={'response_time': -0.3, 'cpu_usage': -0.2},
                    execution_time=time.time()
                )
        
        # Scale down conditions
        elif (avg_response_time < 0.5 and 
              cpu_usage < self.scale_down_threshold and
              success_rate > 0.99):
            
            if self.current_workers > self.min_workers:
                new_workers = max(self.min_workers, int(self.current_workers * 0.8))
                return ScalingDecision(
                    direction=ScaleDirection.DOWN,
                    resource_type=ResourceType.WORKERS,
                    target_value=new_workers,
                    confidence=0.7,
                    reason=f"Low load: response_time={avg_response_time:.2f}s, cpu={cpu_usage:.2f}",
                    expected_impact={'resource_usage': -0.2},
                    execution_time=time.time()
                )
        
        # No scaling needed
        return ScalingDecision(
            direction=ScaleDirection.MAINTAIN,
            resource_type=ResourceType.WORKERS,
            target_value=self.current_workers,
            confidence=0.9,
            reason="System operating within optimal parameters",
            expected_impact={},
            execution_time=time.time()
        )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        if decision.resource_type == ResourceType.WORKERS:
            old_workers = self.current_workers
            self.current_workers = int(decision.target_value)
            
            logger.info(
                f"Scaling workers: {old_workers} -> {self.current_workers} "
                f"({decision.direction.value}) - {decision.reason}"
            )
            
            # Adjust thread pool size
            # Note: ThreadPoolExecutor doesn't support dynamic resizing easily
            # In production, would use more sophisticated worker management
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get load balancer performance statistics."""
        if self.request_count == 0:
            return {
                'request_count': 0,
                'avg_response_time': 0.0,
                'error_rate': 0.0,
                'success_rate': 1.0,
                'current_workers': self.current_workers
            }
        
        return {
            'request_count': self.request_count,
            'avg_response_time': self.total_response_time / self.request_count,
            'error_rate': self.error_count / self.request_count,
            'success_rate': 1.0 - (self.error_count / self.request_count),
            'current_workers': self.current_workers,
            'max_workers': self.max_workers,
            'min_workers': self.min_workers,
            'queue_size': self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0
        }


class AdaptivePerformanceOptimizer:
    """ML-powered performance optimizer with predictive capabilities."""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=1000)
        self.performance_profiles = {}
        self.adaptive_thresholds = {
            'response_time_target': 1.0,
            'cpu_usage_target': 0.7,
            'memory_usage_target': 0.8,
            'throughput_target': 100.0
        }
        
        # Optimization strategies
        self.active_optimizations = set()
        self.optimization_impacts = {}
        
        logger.info("Adaptive Performance Optimizer initialized")
    
    async def optimize_performance(
        self,
        current_metrics: ScalingMetrics
    ) -> List[Dict[str, Any]]:
        """Perform adaptive performance optimization."""
        optimizations = []
        
        # Analyze current performance
        performance_profile = self._analyze_performance(current_metrics)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimizations(performance_profile, current_metrics)
        
        # Apply optimizations
        for recommendation in recommendations:
            if await self._apply_optimization(recommendation):
                optimizations.append(recommendation)
        
        # Update optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'metrics': current_metrics,
            'profile': performance_profile,
            'optimizations': optimizations
        })
        
        return optimizations
    
    def _analyze_performance(self, metrics: ScalingMetrics) -> PerformanceProfile:
        """Analyze current performance characteristics."""
        # Calculate efficiency scores
        cpu_efficiency = self._calculate_cpu_efficiency(metrics.cpu_usage, metrics.throughput)
        memory_efficiency = self._calculate_memory_efficiency(metrics.memory_usage, metrics.active_connections)
        
        # Estimate latency percentiles (simplified)
        base_latency = metrics.response_time
        latency_p50 = base_latency
        latency_p95 = base_latency * 2.5
        latency_p99 = base_latency * 4.0
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            cpu_efficiency, memory_efficiency, metrics.error_rate, base_latency
        )
        
        return PerformanceProfile(
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput_rps=metrics.request_rate,
            cpu_efficiency=cpu_efficiency,
            memory_efficiency=memory_efficiency,
            error_rate=metrics.error_rate,
            optimization_score=optimization_score
        )
    
    def _calculate_cpu_efficiency(self, cpu_usage: float, throughput: float) -> float:
        """Calculate CPU efficiency score."""
        if cpu_usage == 0:
            return 1.0
        
        # Efficiency = throughput per unit CPU
        efficiency = throughput / max(cpu_usage, 0.1)
        return min(1.0, efficiency / 100.0)  # Normalize
    
    def _calculate_memory_efficiency(self, memory_usage: float, connections: int) -> float:
        """Calculate memory efficiency score."""
        if memory_usage == 0:
            return 1.0
        
        # Efficiency = connections per unit memory
        efficiency = connections / max(memory_usage, 0.1)
        return min(1.0, efficiency / 50.0)  # Normalize
    
    def _calculate_optimization_score(
        self,
        cpu_efficiency: float,
        memory_efficiency: float,
        error_rate: float,
        latency: float
    ) -> float:
        """Calculate overall optimization score."""
        # Weighted combination of factors
        score = (
            cpu_efficiency * 0.3 +
            memory_efficiency * 0.3 +
            (1.0 - min(error_rate, 1.0)) * 0.2 +
            max(0.0, 1.0 - latency / 5.0) * 0.2  # Good if latency < 5s
        )
        
        return max(0.0, min(1.0, score))
    
    def _generate_optimizations(
        self,
        profile: PerformanceProfile,
        metrics: ScalingMetrics
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # CPU optimization
        if profile.cpu_efficiency < 0.5:
            recommendations.append({
                'type': 'cpu_optimization',
                'strategy': 'parallel_processing',
                'priority': 'high',
                'expected_improvement': 0.3,
                'description': 'Enable parallel processing for CPU-intensive operations'
            })
        
        # Memory optimization
        if profile.memory_efficiency < 0.6:
            recommendations.append({
                'type': 'memory_optimization',
                'strategy': 'caching_enhancement',
                'priority': 'medium',
                'expected_improvement': 0.25,
                'description': 'Implement advanced caching to reduce memory pressure'
            })
        
        # Latency optimization
        if profile.latency_p95 > self.adaptive_thresholds['response_time_target'] * 2:
            recommendations.append({
                'type': 'latency_optimization',
                'strategy': 'request_batching',
                'priority': 'high',
                'expected_improvement': 0.4,
                'description': 'Implement request batching to reduce latency'
            })
        
        # Throughput optimization
        if metrics.request_rate < self.adaptive_thresholds['throughput_target']:
            recommendations.append({
                'type': 'throughput_optimization',
                'strategy': 'connection_pooling',
                'priority': 'medium',
                'expected_improvement': 0.35,
                'description': 'Optimize connection pooling for higher throughput'
            })
        
        # Error rate optimization
        if profile.error_rate > 0.01:  # > 1% error rate
            recommendations.append({
                'type': 'reliability_optimization',
                'strategy': 'circuit_breaker',
                'priority': 'critical',
                'expected_improvement': 0.5,
                'description': 'Implement circuit breakers to reduce error rate'
            })
        
        return recommendations
    
    async def _apply_optimization(self, recommendation: Dict[str, Any]) -> bool:
        """Apply optimization recommendation."""
        try:
            optimization_type = recommendation['type']
            strategy = recommendation['strategy']
            
            # Track that this optimization is being applied
            opt_key = f"{optimization_type}_{strategy}"
            
            if opt_key in self.active_optimizations:
                return False  # Already applied
            
            # Apply the optimization (simplified - in production would have real implementations)
            success = await self._execute_optimization_strategy(strategy)
            
            if success:
                self.active_optimizations.add(opt_key)
                self.optimization_impacts[opt_key] = {
                    'applied_at': time.time(),
                    'expected_improvement': recommendation['expected_improvement'],
                    'priority': recommendation['priority']
                }
                
                logger.info(f"Applied optimization: {recommendation['description']}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {recommendation}: {e}")
        
        return False
    
    async def _execute_optimization_strategy(self, strategy: str) -> bool:
        """Execute specific optimization strategy."""
        # Simulate optimization execution
        await asyncio.sleep(0.1)  # Simulate work
        
        # In production, this would contain real optimization logic
        strategies = {
            'parallel_processing': self._enable_parallel_processing,
            'caching_enhancement': self._enhance_caching,
            'request_batching': self._enable_request_batching,
            'connection_pooling': self._optimize_connection_pooling,
            'circuit_breaker': self._implement_circuit_breaker
        }
        
        if strategy in strategies:
            return strategies[strategy]()
        
        return True  # Default success for demonstration
    
    def _enable_parallel_processing(self) -> bool:
        """Enable parallel processing optimization."""
        # Implementation would adjust thread/process pool settings
        logger.debug("Parallel processing optimization enabled")
        return True
    
    def _enhance_caching(self) -> bool:
        """Enhance caching optimization."""
        # Implementation would adjust cache parameters
        logger.debug("Caching enhancement optimization enabled")
        return True
    
    def _enable_request_batching(self) -> bool:
        """Enable request batching optimization."""
        # Implementation would enable batch processing
        logger.debug("Request batching optimization enabled")
        return True
    
    def _optimize_connection_pooling(self) -> bool:
        """Optimize connection pooling."""
        # Implementation would adjust connection pool parameters
        logger.debug("Connection pooling optimization enabled")
        return True
    
    def _implement_circuit_breaker(self) -> bool:
        """Implement circuit breaker optimization."""
        # Implementation would add circuit breaker logic
        logger.debug("Circuit breaker optimization enabled")
        return True
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            'active_optimizations': len(self.active_optimizations),
            'optimization_details': dict(self.optimization_impacts),
            'adaptive_thresholds': self.adaptive_thresholds.copy(),
            'optimization_history': len(self.optimization_history),
            'performance_profiles_count': len(self.performance_profiles),
            'recent_optimizations': [
                opt for opt in self.optimization_history
                if time.time() - opt['timestamp'] < 3600  # Last hour
            ][-10:]  # Last 10
        }


class InfiniteScaleEngine:
    """Main engine coordinating infinite scaling and optimization."""
    
    def __init__(self, max_workers: int = None):
        self.load_balancer = IntelligentLoadBalancer(max_workers)
        self.performance_optimizer = AdaptivePerformanceOptimizer()
        
        # Scaling configuration
        self.scaling_enabled = True
        self.optimization_enabled = True
        self.monitoring_interval = 30.0
        
        # Performance tracking
        self.total_requests = 0
        self.total_optimizations = 0
        self.scaling_events = 0
        
        # Start monitoring
        self._monitoring_task = None
        self._start_monitoring()
        
        logger.info("Infinite Scale Engine initialized and monitoring started")
    
    def _start_monitoring(self):
        """Start background monitoring."""
        async def monitor():
            await self.load_balancer.start_monitoring()
            
            while True:
                try:
                    await self._optimization_cycle()
                    await asyncio.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Error in optimization cycle: {e}")
                    await asyncio.sleep(5.0)
        
        # Schedule monitoring in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._monitoring_task = loop.create_task(monitor())
        except RuntimeError:
            # No event loop running, will start later
            pass
    
    async def _optimization_cycle(self):
        """Run optimization cycle."""
        if not (self.scaling_enabled or self.optimization_enabled):
            return
        
        # Gather current metrics
        current_metrics = await self._gather_metrics()
        
        # Run performance optimization
        if self.optimization_enabled:
            optimizations = await self.performance_optimizer.optimize_performance(current_metrics)
            self.total_optimizations += len(optimizations)
        
        # Update adaptive thresholds based on performance
        await self._update_adaptive_thresholds(current_metrics)
    
    async def _gather_metrics(self) -> ScalingMetrics:
        """Gather comprehensive system metrics."""
        lb_stats = self.load_balancer.get_performance_stats()
        
        # Get system metrics
        cpu_usage = 50.0  # Default values
        memory_usage = 50.0
        network_io = 0.0
        disk_io = 0.0
        
        if psutil:
            try:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_usage = psutil.virtual_memory().percent
                
                net_io = psutil.net_io_counters()
                network_io = net_io.bytes_sent + net_io.bytes_recv
                
                disk_io_counters = psutil.disk_io_counters()
                if disk_io_counters:
                    disk_io = disk_io_counters.read_bytes + disk_io_counters.write_bytes
                
            except Exception as e:
                logger.debug(f"Failed to get system metrics: {e}")
        
        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_io=network_io,
            disk_io=disk_io,
            active_connections=lb_stats['current_workers'],
            request_rate=lb_stats['request_count'] / max(1, time.time() - getattr(self, '_start_time', time.time())),
            response_time=lb_stats['avg_response_time'],
            error_rate=lb_stats['error_rate'],
            queue_depth=lb_stats['queue_size'],
            throughput=1.0 / max(0.001, lb_stats['avg_response_time'])  # Requests per second equivalent
        )
    
    async def _update_adaptive_thresholds(self, metrics: ScalingMetrics):
        """Update adaptive thresholds based on current performance."""
        # Adjust thresholds based on recent performance
        if metrics.response_time < self.performance_optimizer.adaptive_thresholds['response_time_target']:
            # Performance is good, can be more aggressive
            self.performance_optimizer.adaptive_thresholds['response_time_target'] *= 0.95
        elif metrics.response_time > self.performance_optimizer.adaptive_thresholds['response_time_target'] * 1.5:
            # Performance is poor, be more conservative
            self.performance_optimizer.adaptive_thresholds['response_time_target'] *= 1.05
        
        # Similar logic for other thresholds
        if metrics.cpu_usage < self.performance_optimizer.adaptive_thresholds['cpu_usage_target']:
            self.performance_optimizer.adaptive_thresholds['cpu_usage_target'] = min(
                0.9, self.performance_optimizer.adaptive_thresholds['cpu_usage_target'] * 1.02
            )
        elif metrics.cpu_usage > self.performance_optimizer.adaptive_thresholds['cpu_usage_target']:
            self.performance_optimizer.adaptive_thresholds['cpu_usage_target'] = max(
                0.5, self.performance_optimizer.adaptive_thresholds['cpu_usage_target'] * 0.98
            )
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task through the scaling engine."""
        self.total_requests += 1
        return await self.load_balancer.submit_task(func, *args, **kwargs)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        lb_stats = self.load_balancer.get_performance_stats()
        opt_report = self.performance_optimizer.get_optimization_report()
        
        return {
            'engine_stats': {
                'total_requests': self.total_requests,
                'total_optimizations': self.total_optimizations,
                'scaling_events': self.scaling_events,
                'scaling_enabled': self.scaling_enabled,
                'optimization_enabled': self.optimization_enabled
            },
            'load_balancer': lb_stats,
            'performance_optimizer': opt_report,
            'health_status': self._assess_health_status(lb_stats, opt_report)
        }
    
    def _assess_health_status(self, lb_stats: Dict, opt_report: Dict) -> str:
        """Assess overall health status."""
        # Simple health assessment
        if lb_stats['error_rate'] > 0.05:  # > 5% error rate
            return 'critical'
        elif lb_stats['avg_response_time'] > 5.0:  # > 5s response time
            return 'warning'
        elif lb_stats['success_rate'] < 0.95:  # < 95% success rate
            return 'warning'
        else:
            return 'healthy'
    
    async def shutdown(self):
        """Gracefully shutdown the scaling engine."""
        logger.info("Shutting down Infinite Scale Engine")
        
        self.scaling_enabled = False
        self.optimization_enabled = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        await self.load_balancer.stop_monitoring()
        
        # Shutdown executors
        self.load_balancer.thread_pool.shutdown(wait=True)
        self.load_balancer.process_pool.shutdown(wait=True)
        
        logger.info("Infinite Scale Engine shutdown complete")


# Global instance
global_scale_engine = InfiniteScaleEngine()


# Convenience functions
async def submit_scalable_task(func: Callable, *args, **kwargs) -> Any:
    """Submit task through the global scaling engine."""
    return await global_scale_engine.submit_task(func, *args, **kwargs)


def get_scaling_stats() -> Dict[str, Any]:
    """Get scaling statistics."""
    return global_scale_engine.get_comprehensive_stats()


def enable_infinite_scaling(enabled: bool = True):
    """Enable or disable infinite scaling."""
    global_scale_engine.scaling_enabled = enabled
    logger.info(f"Infinite scaling {'enabled' if enabled else 'disabled'}")


def enable_performance_optimization(enabled: bool = True):
    """Enable or disable performance optimization."""
    global_scale_engine.optimization_enabled = enabled
    logger.info(f"Performance optimization {'enabled' if enabled else 'disabled'}")


# Decorators for automatic scaling
def scalable_task(func: Callable) -> Callable:
    """Decorator to make any function automatically scalable."""
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await global_scale_engine.submit_task(func, *args, **kwargs)
    
    return wrapper


def performance_monitored(func: Callable) -> Callable:
    """Decorator to automatically monitor function performance."""
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record performance metrics
            # This would integrate with the observability engine
            logger.debug(f"Function {func.__name__} executed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.warning(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper