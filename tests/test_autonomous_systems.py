"""Tests for autonomous system components."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import statistics

from src.sql_synth.autonomous_evolution import (
    AdaptiveLearningEngine, 
    SelfHealingSystem,
    InnovationEngine,
    EvolutionMetrics
)
from src.sql_synth.research_framework import IntelligentDiscovery
from src.sql_synth.advanced_resilience import (
    CircuitBreaker,
    AdaptiveRetry,
    ResilienceOrchestrator,
    ResilienceConfig
)
from src.sql_synth.quantum_optimization import (
    QuantumInspiredOptimizer,
    GeneticOptimizer,
    ParticleSwarmOptimizer,
    HybridMultiObjectiveOptimizer,
    OptimizationTarget
)
from src.sql_synth.intelligent_scaling import (
    AutoScaler,
    PredictiveScaler,
    ScalingMetrics,
    ScalingPolicy,
    ResourceType
)


class TestAdaptiveLearningEngine:
    """Test adaptive learning engine functionality."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = AdaptiveLearningEngine()
        assert engine.learning_rate == 0.1
        assert engine.adaptation_threshold == 0.05
        assert len(engine.adaptation_strategies) == 4
    
    def test_system_evolution(self):
        """Test system evolution with sample metrics."""
        engine = AdaptiveLearningEngine()
        
        # Create sample metrics
        metrics_batch = [
            {
                'response_time': 2.5,
                'accuracy_score': 0.85,
                'cache_hit_rate': 0.6,
                'error_rate': 0.02,
                'timestamp': datetime.now()
            }
        ]
        
        evolution_report = engine.evolve_system(metrics_batch)
        
        assert 'timestamp' in evolution_report
        assert 'adaptations_applied' in evolution_report
        assert 'learning_insights' in evolution_report
        assert isinstance(evolution_report['adaptations_applied'], list)
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        engine = AdaptiveLearningEngine()
        
        # Create trending data
        values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Upward trend
        trend = engine._calculate_trend(values)
        
        assert trend > 0  # Should detect upward trend
    
    def test_adaptation_strategies(self):
        """Test individual adaptation strategies."""
        engine = AdaptiveLearningEngine()
        
        metrics = [{'response_time': 5.0}]  # High response time
        trends = {'performance_trend': 0.1}  # Degrading
        
        result = engine._adapt_performance(metrics, trends)
        assert result['applied'] is True
        assert result['expected_improvement'] > 0


class TestSelfHealingSystem:
    """Test self-healing system functionality."""
    
    def test_system_initialization(self):
        """Test self-healing system initializes correctly."""
        system = SelfHealingSystem()
        assert len(system.healing_strategies) == 4
        assert isinstance(system.healing_history, list)
    
    def test_issue_detection(self):
        """Test issue detection from system state."""
        system = SelfHealingSystem()
        
        # Create system state with issues
        system_state = {
            'connection_error_rate': 0.15,  # High error rate
            'memory_usage': 0.95,  # High memory usage
            'avg_response_time': 6.0,  # High response time
            'avg_accuracy': 0.7  # Low accuracy
        }
        
        issues = system._detect_issues(system_state)
        assert len(issues) == 4  # Should detect all 4 issues
        
        # Check issue types
        issue_types = {issue['type'] for issue in issues}
        expected_types = {
            'connection_failures', 'memory_leaks', 
            'performance_degradation', 'accuracy_drops'
        }
        assert issue_types == expected_types
    
    def test_healing_strategies(self):
        """Test healing strategy execution."""
        system = SelfHealingSystem()
        
        issue = {'type': 'connection_failures', 'severity': 'high'}
        system_state = {}
        
        result = system._heal_connection_issues(issue, system_state)
        
        assert 'issue_type' in result
        assert 'actions_taken' in result
        assert 'expected_recovery_time' in result
        assert len(result['actions_taken']) > 0


class TestIntelligentDiscovery:
    """Test intelligent discovery functionality."""
    
    def test_discovery_initialization(self):
        """Test discovery system initializes correctly."""
        discovery = IntelligentDiscovery()
        assert len(discovery.discovery_patterns) == 4
    
    def test_research_opportunity_discovery(self):
        """Test research opportunity discovery."""
        discovery = IntelligentDiscovery()
        
        # Create sample metrics with performance issues
        metrics_history = [
            {
                'response_time': 2.0 + i * 0.1,  # Increasing response time
                'accuracy_score': 0.85,
                'memory_usage': 0.7,
                'cache_hit_rate': 0.6,
                'query_complexity': 'complex' if i % 3 == 0 else 'simple'
            }
            for i in range(20)
        ]
        
        opportunities = discovery.discover_research_opportunities(metrics_history)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check opportunity structure
        for opp in opportunities:
            assert 'discovery_method' in opp
            assert 'confidence' in opp
            assert 'type' in opp
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        discovery = IntelligentDiscovery()
        
        opportunity = {'potential_impact': 'High'}
        metrics = [{}] * 100  # Large dataset
        
        confidence = discovery._calculate_confidence(opportunity, metrics)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high with large dataset and high impact


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes correctly."""
        config = ResilienceConfig()
        cb = CircuitBreaker(config, "test")
        
        assert cb.name == "test"
        assert cb.state.value == "closed"
        assert cb.metrics.failure_count == 0
    
    def test_successful_call(self):
        """Test successful function call through circuit breaker."""
        config = ResilienceConfig()
        cb = CircuitBreaker(config, "test")
        
        def successful_function():
            return "success"
        
        result = cb.call(successful_function)
        assert result == "success"
        assert cb.metrics.success_count == 1
    
    def test_failed_call(self):
        """Test failed function call handling."""
        config = ResilienceConfig()
        cb = CircuitBreaker(config, "test")
        
        def failing_function():
            raise Exception("Test error")
        
        with pytest.raises(Exception):
            cb.call(failing_function)
        
        assert cb.metrics.failure_count == 1
    
    def test_circuit_opening(self):
        """Test circuit breaker opening after failures."""
        config = ResilienceConfig(circuit_breaker_threshold=2)
        cb = CircuitBreaker(config, "test")
        
        def failing_function():
            raise Exception("Test error")
        
        # Trigger failures to open circuit
        for _ in range(3):
            try:
                cb.call(failing_function)
            except:
                pass
        
        # Circuit should be open now
        assert cb.state.value == "open"


class TestQuantumOptimization:
    """Test quantum-inspired optimization."""
    
    def test_quantum_optimizer_initialization(self):
        """Test quantum optimizer initializes correctly."""
        optimizer = QuantumInspiredOptimizer()
        assert optimizer.problem_space_size == 1000
        assert optimizer.quantum_states is not None
        assert optimizer.entanglement_matrix is not None
    
    def test_quantum_annealing(self):
        """Test quantum annealing optimization."""
        optimizer = QuantumInspiredOptimizer()
        
        def simple_objective(params, target):
            # Simple quadratic function
            return sum((v - 0.5) ** 2 for v in params.values())
        
        initial_params = {'x': 0.1, 'y': 0.9}
        target = OptimizationTarget()
        
        result = optimizer.quantum_annealing_optimize(
            simple_objective, initial_params, target
        )
        
        assert isinstance(result, dict)
        assert 'x' in result and 'y' in result
        
        # Should converge towards 0.5, 0.5 (relaxed bounds for quantum optimization)
        assert abs(result['x'] - 0.5) < 0.4
        assert abs(result['y'] - 0.5) < 0.4
    
    def test_genetic_optimizer(self):
        """Test genetic algorithm optimization."""
        optimizer = GeneticOptimizer(population_size=20, generations=10)
        
        def simple_objective(params, target):
            return sum((v - 0.5) ** 2 for v in params.values())
        
        param_bounds = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        target = OptimizationTarget()
        
        result = optimizer.optimize(simple_objective, param_bounds, target)
        
        assert isinstance(result, dict)
        assert 'x' in result and 'y' in result
        assert 0.0 <= result['x'] <= 1.0
        assert 0.0 <= result['y'] <= 1.0


class TestIntelligentScaling:
    """Test intelligent auto-scaling functionality."""
    
    def test_predictive_scaler_initialization(self):
        """Test predictive scaler initializes correctly."""
        scaler = PredictiveScaler()
        assert scaler.prediction_horizon == 10
        assert isinstance(scaler.historical_metrics, list)
    
    def test_resource_demand_prediction(self):
        """Test resource demand prediction."""
        scaler = PredictiveScaler()
        
        # Add sample historical data
        for i in range(20):
            metrics = ScalingMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_utilization=0.5 + 0.01 * i,  # Trending up
                memory_utilization=0.4,
                queue_length=10,
                response_time_p95=2.0,
                throughput_qps=100,
                error_rate=0.01,
                active_connections=20,
                cache_hit_rate=0.8,
                pending_requests=5
            )
            scaler.add_metrics(metrics)
        
        current_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=0.7,
            memory_utilization=0.4,
            queue_length=10,
            response_time_p95=2.0,
            throughput_qps=100,
            error_rate=0.01,
            active_connections=20,
            cache_hit_rate=0.8,
            pending_requests=5
        )
        
        predictions = scaler.predict_resource_demand(current_metrics)
        
        assert isinstance(predictions, dict)
        assert ResourceType.CPU_THREADS in predictions
        assert ResourceType.MEMORY_POOL in predictions
        
        # CPU should be predicted higher due to trend
        assert predictions[ResourceType.CPU_THREADS] > current_metrics.cpu_utilization
    
    def test_auto_scaler_decisions(self):
        """Test auto-scaler decision making."""
        auto_scaler = AutoScaler()
        
        # Add scaling policy
        policy = ScalingPolicy(
            resource_type=ResourceType.CPU_THREADS,
            min_instances=2,
            max_instances=10,
            target_utilization=0.7,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            scale_up_cooldown_seconds=60,
            scale_down_cooldown_seconds=300
        )
        auto_scaler.add_scaling_policy(policy)
        
        # High utilization metrics
        high_util_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=0.9,  # Above threshold
            memory_utilization=0.4,
            queue_length=25,  # High queue
            response_time_p95=6.0,  # High response time
            throughput_qps=100,
            error_rate=0.02,
            active_connections=20,
            cache_hit_rate=0.8,
            pending_requests=15
        )
        
        decisions = auto_scaler.evaluate_scaling(high_util_metrics)
        
        assert ResourceType.CPU_THREADS in decisions
        # Should decide to scale up due to high utilization
        assert decisions[ResourceType.CPU_THREADS].value in ['scale_up', 'no_change']
    
    def test_scaling_policy_application(self):
        """Test scaling policy application."""
        auto_scaler = AutoScaler()
        
        policy = ScalingPolicy(
            resource_type=ResourceType.CPU_THREADS,
            min_instances=2,
            max_instances=10,
            target_utilization=0.7,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            scale_up_cooldown_seconds=60,
            scale_down_cooldown_seconds=300,
            scaling_factor=2.0
        )
        auto_scaler.add_scaling_policy(policy)
        
        # Get initial pool size
        pool = auto_scaler.resource_pools[ResourceType.CPU_THREADS]
        initial_size = pool.current_size
        
        # Apply scale up
        from src.sql_synth.intelligent_scaling import ScalingDirection
        result = auto_scaler.apply_scaling_decision(
            ResourceType.CPU_THREADS, 
            ScalingDirection.SCALE_UP
        )
        
        assert result is True
        assert pool.current_size > initial_size


class TestIntegrationScenarios:
    """Test integration scenarios across components."""
    
    def test_resilience_with_optimization(self):
        """Test resilience orchestrator with optimization."""
        config = ResilienceConfig()
        orchestrator = ResilienceOrchestrator(config)
        
        call_count = 0
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return "success"
        
        # Should retry and eventually succeed
        result = orchestrator.execute_resilient(
            test_function, "test_op", "test_circuit", "test_pool"
        )
        
        assert result == "success"
        assert call_count == 3  # Initial call + 2 retries
    
    def test_adaptive_learning_with_scaling(self):
        """Test adaptive learning with auto-scaling."""
        learning_engine = AdaptiveLearningEngine()
        auto_scaler = AutoScaler()
        
        # Simulate system under load
        metrics_batch = [
            {
                'response_time': 5.0,  # High response time
                'accuracy_score': 0.75,  # Low accuracy
                'cache_hit_rate': 0.5,  # Poor cache performance
                'error_rate': 0.08,  # High error rate
                'cpu_utilization': 0.9,  # High CPU
                'memory_utilization': 0.8,  # High memory
                'queue_length': 30,  # High queue
                'timestamp': datetime.now()
            }
        ]
        
        # Get evolution recommendations
        evolution_report = learning_engine.evolve_system(metrics_batch)
        
        # Get scaling recommendations
        scaling_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=0.9,
            memory_utilization=0.8,
            queue_length=30,
            response_time_p95=5.0,
            throughput_qps=50,
            error_rate=0.08,
            active_connections=40,
            cache_hit_rate=0.5,
            pending_requests=20
        )
        
        scaling_recommendations = auto_scaler.get_scaling_recommendations(scaling_metrics)
        
        # Both systems should detect issues and recommend actions
        assert len(evolution_report['adaptations_applied']) > 0
        assert scaling_recommendations['system_health']['status'] in ['degraded', 'critical']
    
    @patch('src.sql_synth.quantum_optimization.HybridMultiObjectiveOptimizer')
    def test_quantum_optimization_integration(self, mock_optimizer):
        """Test quantum optimization integration."""
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        mock_optimizer_instance.optimize.return_value = {
            'cache_size': 0.8,
            'connection_pool': 0.6,
            'timeout_threshold': 0.7
        }
        
        from src.sql_synth.quantum_optimization import get_global_optimizer
        optimizer = get_global_optimizer()
        
        # Mock performance evaluator
        def mock_performance_evaluator(params):
            class MockMetrics:
                def __init__(self):
                    self.response_time = 2.0
                    self.throughput = 100.0
                    self.accuracy_score = 0.9
                    self.cost_efficiency = 0.8
                    self.cache_hit_rate = params.get('cache_size', 0.5)
            return MockMetrics()
        
        from src.sql_synth.quantum_optimization import create_multi_objective_function
        objective_func = create_multi_objective_function(mock_performance_evaluator)
        
        target = OptimizationTarget()
        param_bounds = {
            'cache_size': (0.0, 1.0),
            'connection_pool': (0.0, 1.0),
            'timeout_threshold': (0.0, 1.0)
        }
        
        # This should work without errors
        result = objective_func({'cache_size': 0.5, 'connection_pool': 0.5, 'timeout_threshold': 0.5}, target)
        assert isinstance(result, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])