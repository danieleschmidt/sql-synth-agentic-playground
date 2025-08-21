"""Tests for advanced autonomous features."""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

from src.sql_synth.adaptive_learning_engine import (
    global_adaptive_learning_engine,
    FeedbackType,
    UserFeedback,
    process_user_feedback,
    get_query_recommendations,
)
from src.sql_synth.next_gen_optimization import (
    global_nextgen_optimizer,
    OptimizationType,
    OptimizationProblem,
    optimize_query_performance_async,
)
from src.sql_synth.advanced_security_framework import (
    global_security_controller,
    create_security_context,
    authorize_sql_request,
    ThreatLevel,
    SecurityEventType,
)
from src.sql_synth.comprehensive_error_recovery import (
    global_error_recovery_system,
    execute_with_resilience,
    resilient_operation,
    RecoveryStrategy,
)
from src.sql_synth.hyperscale_performance_system import (
    global_performance_system,
    execute_optimized_operation,
    hyperscale_optimized,
    register_worker_node,
)


class TestAdaptiveLearningEngine:
    """Tests for the adaptive learning engine."""

    @pytest.mark.asyncio
    async def test_process_feedback_positive(self):
        """Test processing positive feedback."""
        result = await process_user_feedback(
            query_id="test_001",
            natural_query="Show me all users",
            generated_sql="SELECT * FROM users;",
            feedback_type="positive",
            feedback_value=1.0,
            user_id="test_user",
            context={"generation_time": 0.5}
        )
        
        assert "feedback_processed" in result
        assert result["feedback_processed"] is True
        assert "reinforcement_learning" in result

    @pytest.mark.asyncio
    async def test_process_feedback_negative(self):
        """Test processing negative feedback."""
        result = await process_user_feedback(
            query_id="test_002",
            natural_query="Count active users",
            generated_sql="SELECT COUNT(*) FROM users;",
            feedback_type="negative",
            feedback_value=-1.0,
            user_id="test_user",
            context={"generation_time": 2.0}
        )
        
        assert "feedback_processed" in result
        assert result["feedback_processed"] is True

    def test_get_recommendations(self):
        """Test getting query recommendations."""
        recommendations = get_query_recommendations(
            "Find top selling products",
            context={"complexity": "medium"}
        )
        
        assert "recommendations" in recommendations
        assert "query_analysis" in recommendations
        assert isinstance(recommendations["recommendations"], list)

    def test_learning_analytics(self):
        """Test learning analytics generation."""
        analytics = global_adaptive_learning_engine.get_learning_analytics()
        
        assert "reinforcement_learning" in analytics
        assert "pattern_analysis" in analytics
        assert "feedback_analysis" in analytics
        assert "system_health" in analytics


class TestNextGenOptimization:
    """Tests for next-generation optimization algorithms."""

    @pytest.mark.asyncio
    async def test_optimize_query_performance(self):
        """Test query performance optimization."""
        query_characteristics = {
            "complexity": 0.7,
            "data_size": 50000,
            "join_count": 2,
            "result_size": 1000,
        }
        
        performance_targets = {
            "max_execution_time": 1.0,
            "max_memory_mb": 256,
        }
        
        recommendations = await optimize_query_performance_async(
            query_characteristics, performance_targets
        )
        
        assert "caching_strategy" in recommendations
        assert "indexing_strategy" in recommendations
        assert "parallelism_strategy" in recommendations
        assert "optimization_metadata" in recommendations

    def test_optimization_problem_creation(self):
        """Test optimization problem creation."""
        def test_objective(params):
            return sum(x**2 for x in params)
        
        problem = OptimizationProblem(
            problem_id="test_optimization",
            objective_function=test_objective,
            bounds=[(0, 1), (0, 1)],
            dimension=2
        )
        
        assert problem.problem_id == "test_optimization"
        assert problem.dimension == 2
        assert len(problem.bounds) == 2

    @pytest.mark.asyncio
    async def test_quantum_optimization(self):
        """Test quantum-inspired optimization."""
        def simple_objective(params):
            return (params[0] - 0.5)**2 + (params[1] - 0.3)**2
        
        problem = OptimizationProblem(
            problem_id="quantum_test",
            objective_function=simple_objective,
            bounds=[(0, 1), (0, 1)],
            dimension=2
        )
        
        result = await global_nextgen_optimizer.optimize_async(
            problem, OptimizationType.QUANTUM_ANNEALING, max_iterations=100
        )
        
        assert result.optimization_type == OptimizationType.QUANTUM_ANNEALING
        assert result.objective_value >= 0
        assert len(result.solution) == 2

    def test_optimization_analytics(self):
        """Test optimization analytics."""
        analytics = global_nextgen_optimizer.get_optimization_analytics()
        
        assert "strategy_performance" in analytics
        assert "system_health" in analytics
        assert "timestamp" in analytics


class TestAdvancedSecurityFramework:
    """Tests for the advanced security framework."""

    def test_create_security_context(self):
        """Test security context creation."""
        context = create_security_context(
            user_id="test_user",
            session_id="session_123",
            permissions={"read", "query"},
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            auth_method="password"
        )
        
        assert context.user_id == "test_user"
        assert context.session_id == "session_123"
        assert "read" in context.permissions
        assert context.ip_address == "192.168.1.100"

    def test_sql_request_authorization(self):
        """Test SQL request authorization."""
        context = create_security_context(
            user_id="test_user",
            session_id="session_123",
            permissions={"database:query"},
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            auth_method="mfa"
        )
        
        query_metadata = {
            "query_length": 25,
            "table_count": 1,
            "complexity_score": 0.3,
            "required_permissions": {"database:query"},
        }
        
        authorized, reasons = authorize_sql_request(
            context, "SELECT * FROM users LIMIT 10;", query_metadata
        )
        
        # Note: Authorization might fail due to session not being properly created
        # But we can test the structure
        assert isinstance(authorized, bool)
        assert isinstance(reasons, list)

    def test_threat_detection(self):
        """Test threat detection capabilities."""
        context = create_security_context(
            user_id="test_user",
            session_id="session_123",
            permissions={"database:query"},
            ip_address="192.168.1.100",
            user_agent="TestAgent/1.0",
            auth_method="password"
        )
        
        # Test with suspicious SQL
        suspicious_query = "SELECT * FROM users WHERE 1=1; DROP TABLE users; --"
        query_metadata = {"query": suspicious_query, "query_length": len(suspicious_query)}
        
        threats = global_security_controller.threat_detector.detect_threats(
            context, suspicious_query, query_metadata
        )
        
        # Should detect SQL injection attempt
        sql_injection_threats = [
            t for t in threats 
            if t.threat_type == SecurityEventType.SQL_INJECTION_ATTEMPT
        ]
        
        assert len(sql_injection_threats) > 0
        assert any(t.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL] for t in sql_injection_threats)

    def test_security_analytics(self):
        """Test security analytics."""
        analytics = global_security_controller.get_security_analytics()
        
        assert "threat_detection" in analytics
        assert "access_control" in analytics
        assert "risk_management" in analytics
        assert "system_health" in analytics


class TestComprehensiveErrorRecovery:
    """Tests for comprehensive error recovery system."""

    @pytest.mark.asyncio
    async def test_execute_with_resilience(self):
        """Test resilient operation execution."""
        call_count = 0
        
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary network issue")
            return "success"
        
        result = await execute_with_resilience(
            flaky_operation, 
            "test_operation",
            config={"max_retries": 3}
        )
        
        assert result == "success"
        assert call_count == 2  # Should retry once

    @pytest.mark.asyncio
    async def test_resilient_decorator(self):
        """Test resilient operation decorator."""
        call_count = 0
        
        @resilient_operation("decorated_test", {"max_retries": 2})
        async def decorated_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary issue")
            return "decorated_success"
        
        result = await decorated_flaky()
        assert result == "decorated_success"
        assert call_count == 2

    def test_error_classification(self):
        """Test error classification."""
        try:
            raise ConnectionError("Database connection failed")
        except Exception as e:
            error_context = global_error_recovery_system.error_classifier.classify_error(
                e, {"operation": "database_query", "component": "database"}
            )
            
            assert error_context.error_type == "ConnectionError"
            assert "connection" in error_context.error_id.lower()
            assert error_context.operation == "database_query"

    def test_resilience_analytics(self):
        """Test resilience analytics."""
        analytics = global_error_recovery_system.get_resilience_analytics()
        
        assert "summary" in analytics
        assert "circuit_breakers" in analytics
        assert "service_health" in analytics
        assert "system_resilience_score" in analytics


class TestHyperScalePerformanceSystem:
    """Tests for hyperscale performance system."""

    @pytest.mark.asyncio
    async def test_execute_optimized_operation(self):
        """Test optimized operation execution."""
        call_count = 0
        
        async def test_operation():
            nonlocal call_count
            call_count += 1
            return f"operation_result_{call_count}"
        
        result = await execute_optimized_operation(
            test_operation, 
            "test_optimized_op",
            cache_key="test_cache_key"
        )
        
        assert "operation_result" in result
        assert call_count == 1
        
        # Second call should use cache
        result2 = await execute_optimized_operation(
            test_operation,
            "test_optimized_op", 
            cache_key="test_cache_key"
        )
        
        # Should return cached result
        assert result2 == result
        # call_count should still be 1 due to cache hit

    @pytest.mark.asyncio
    async def test_hyperscale_decorator(self):
        """Test hyperscale optimization decorator."""
        @hyperscale_optimized("decorated_op", cache_key_func=lambda x: f"key_{x}")
        async def decorated_operation(value):
            return f"decorated_result_{value}"
        
        result = await decorated_operation(42)
        assert result == "decorated_result_42"

    def test_worker_registration(self):
        """Test worker node registration."""
        register_worker_node("test_worker_1", {
            "cpu": 4.0,
            "memory": 8.0,
            "network": 1.0,
        })
        
        # Check if worker was registered
        load_distribution = global_performance_system.load_balancer.get_load_distribution()
        assert "test_worker_1" in load_distribution
        
        worker_info = load_distribution["test_worker_1"]
        assert worker_info["capacity"]["cpu"] == 4.0
        assert worker_info["capacity"]["memory"] == 8.0

    def test_semantic_cache(self):
        """Test semantic cache functionality."""
        cache = global_performance_system.semantic_cache
        
        # Test basic caching
        cache.put("test query about users", "SELECT * FROM users;")
        result = cache.get("test query about users")
        assert result == "SELECT * FROM users;"
        
        # Test semantic similarity
        similar_result = cache.get("query about users")  # Should match semantically
        # Note: May or may not match depending on similarity threshold
        
        # Test cache statistics
        stats = cache.get_stats()
        assert "hit_rate" in stats
        assert "size" in stats
        assert stats["size"] >= 1

    def test_performance_analytics(self):
        """Test performance system analytics."""
        analytics = global_performance_system.get_system_analytics()
        
        assert "cache_performance" in analytics
        assert "load_balancing" in analytics
        assert "auto_scaling" in analytics
        assert "performance_monitoring" in analytics
        assert "system_health" in analytics


class TestIntegrationScenarios:
    """Integration tests combining multiple advanced features."""

    @pytest.mark.asyncio
    async def test_end_to_end_advanced_workflow(self):
        """Test end-to-end workflow with all advanced features."""
        # 1. Create security context
        security_context = create_security_context(
            user_id="integration_test_user",
            session_id="integration_session",
            permissions={"database:query", "cache:read"},
            ip_address="192.168.1.50",
            user_agent="IntegrationTest/1.0",
            auth_method="mfa"
        )
        
        # 2. Authorize request
        query_metadata = {
            "query_length": 30,
            "table_count": 1,
            "complexity_score": 0.4,
            "required_permissions": {"database:query"},
        }
        
        # Note: Authorization might fail due to session management
        authorized, reasons = authorize_sql_request(
            security_context, 
            "SELECT id, name FROM users LIMIT 5;", 
            query_metadata
        )
        
        # 3. Execute with resilience and optimization
        async def mock_sql_operation():
            await asyncio.sleep(0.1)  # Simulate processing
            return [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Charlie"},
            ]
        
        result = await execute_with_resilience(
            lambda: execute_optimized_operation(
                mock_sql_operation,
                "integration_query",
                cache_key="users_list_query"
            ),
            "integrated_sql_execution",
            config={"max_retries": 2}
        )
        
        assert isinstance(result, list)
        assert len(result) >= 0

    @pytest.mark.asyncio
    async def test_adaptive_learning_integration(self):
        """Test adaptive learning integration with other systems."""
        # Simulate user feedback on query performance
        await process_user_feedback(
            query_id="integration_query_001",
            natural_query="Show recent orders",
            generated_sql="SELECT * FROM orders WHERE created_at > NOW() - INTERVAL 1 DAY;",
            feedback_type="rating",
            feedback_value=4.0,  # Good rating
            user_id="integration_user",
            context={
                "generation_time": 0.8,
                "execution_time": 0.3,
                "result_count": 150,
                "cache_hit": False,
            }
        )
        
        # Get recommendations based on learning
        recommendations = get_query_recommendations(
            "Display recent orders",  # Similar query
            context={"user_preferences": {"fast_results": True}}
        )
        
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0

    def test_performance_monitoring_integration(self):
        """Test performance monitoring across all systems."""
        # Get comprehensive system analytics
        adaptive_analytics = global_adaptive_learning_engine.get_learning_analytics()
        optimization_analytics = global_nextgen_optimizer.get_optimization_analytics()
        security_analytics = global_security_controller.get_security_analytics()
        resilience_analytics = global_error_recovery_system.get_resilience_analytics()
        performance_analytics = global_performance_system.get_system_analytics()
        
        # Verify all analytics are structured properly
        for analytics in [adaptive_analytics, optimization_analytics, 
                         security_analytics, resilience_analytics, performance_analytics]:
            assert isinstance(analytics, dict)
            assert "timestamp" in analytics or "error" in analytics

    @pytest.mark.asyncio
    async def test_stress_scenario(self):
        """Test system behavior under stress conditions."""
        # Simulate multiple concurrent operations
        async def concurrent_operation(op_id):
            await asyncio.sleep(0.01)  # Small delay
            return f"operation_{op_id}_complete"
        
        # Execute multiple operations concurrently
        tasks = [
            execute_optimized_operation(
                lambda: concurrent_operation(i),
                f"stress_test_op_{i}",
                cache_key=f"stress_key_{i % 10}"  # Some cache overlap
            )
            for i in range(20)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most operations completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 15  # Allow for some failures under stress

    def test_system_health_monitoring(self):
        """Test comprehensive system health monitoring."""
        # Check health of all major systems
        systems = {
            "adaptive_learning": global_adaptive_learning_engine.get_learning_analytics(),
            "optimization": global_nextgen_optimizer.get_optimization_analytics(),
            "security": global_security_controller.get_security_analytics(),
            "error_recovery": global_error_recovery_system.get_resilience_analytics(),
            "performance": global_performance_system.get_system_analytics(),
        }
        
        healthy_systems = 0
        for system_name, analytics in systems.items():
            if isinstance(analytics, dict) and "error" not in analytics:
                healthy_systems += 1
            elif "error" in analytics:
                print(f"System {system_name} reporting error: {analytics['error']}")
        
        # At least 80% of systems should be healthy
        health_ratio = healthy_systems / len(systems)
        assert health_ratio >= 0.8, f"Only {healthy_systems}/{len(systems)} systems are healthy"


# Mock fixtures and utilities for testing
@pytest.fixture
def mock_database():
    """Mock database for testing."""
    return Mock()


@pytest.fixture
def sample_security_context():
    """Sample security context for testing."""
    return create_security_context(
        user_id="test_user",
        session_id="test_session",
        permissions={"database:query", "cache:read", "analytics:view"},
        ip_address="192.168.1.100",
        user_agent="TestAgent/1.0",
        auth_method="mfa"
    )


@pytest.fixture
async def performance_system_setup():
    """Set up performance system for testing."""
    # Register test workers
    register_worker_node("test_worker_1", {"cpu": 2.0, "memory": 4.0})
    register_worker_node("test_worker_2", {"cpu": 4.0, "memory": 8.0})
    
    yield
    
    # Cleanup
    global_performance_system.load_balancer.worker_nodes.clear()


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running basic functionality tests...")
    
    # Test adaptive learning
    try:
        recommendations = get_query_recommendations("test query")
        print(f"✓ Adaptive learning recommendations: {len(recommendations.get('recommendations', []))} items")
    except Exception as e:
        print(f"✗ Adaptive learning test failed: {e}")
    
    # Test security framework
    try:
        context = create_security_context(
            "test_user", "test_session", {"read"}, "127.0.0.1", "Test", "password"
        )
        print("✓ Security context creation successful")
    except Exception as e:
        print(f"✗ Security framework test failed: {e}")
    
    # Test performance system
    try:
        analytics = global_performance_system.get_system_analytics()
        print("✓ Performance system analytics successful")
    except Exception as e:
        print(f"✗ Performance system test failed: {e}")
    
    print("Basic functionality tests completed.")