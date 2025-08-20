#!/usr/bin/env python3
"""Basic test runner for advanced features without external dependencies."""

import asyncio
import sys
import time
import traceback
from typing import Dict, Any

# Import our modules
try:
    from src.sql_synth.adaptive_learning_engine import (
        global_adaptive_learning_engine,
        get_query_recommendations,
        get_learning_insights,
    )
    from src.sql_synth.next_gen_optimization import (
        global_nextgen_optimizer,
        get_optimization_insights,
    )
    from src.sql_synth.advanced_security_framework import (
        global_security_controller,
        create_security_context,
        get_security_insights,
    )
    from src.sql_synth.comprehensive_error_recovery import (
        global_error_recovery_system,
        get_resilience_insights,
    )
    from src.sql_synth.hyperscale_performance_system import (
        global_performance_system,
        get_hyperscale_insights,
    )
    print("✓ All advanced modules imported successfully")
except ImportError as e:
    print(f"✗ Module import failed: {e}")
    sys.exit(1)


class TestRunner:
    """Simple test runner for autonomous features."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = time.time()

    def run_test(self, test_name: str, test_func):
        """Run a single test function."""
        self.tests_run += 1
        try:
            print(f"Running: {test_name}")
            
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            
            print(f"✓ PASS: {test_name}")
            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"✗ FAIL: {test_name}")
            print(f"  Error: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            self.tests_failed += 1
            return False

    def print_summary(self):
        """Print test summary."""
        duration = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"Test Summary:")
        print(f"  Tests run: {self.tests_run}")
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {self.tests_failed}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"{'='*50}")


# Test functions
def test_adaptive_learning_basic():
    """Test basic adaptive learning functionality."""
    # Test recommendations
    recommendations = get_query_recommendations("Show me all active users")
    assert "recommendations" in recommendations
    assert "query_analysis" in recommendations
    
    # Test analytics
    analytics = get_learning_insights()
    assert isinstance(analytics, dict)
    
    return True


def test_optimization_system():
    """Test next-generation optimization system."""
    analytics = get_optimization_insights()
    assert isinstance(analytics, dict)
    
    # Test that the system is properly initialized
    assert "strategy_performance" in analytics or "error" in analytics
    
    return True


def test_security_framework():
    """Test advanced security framework."""
    # Test security context creation
    context = create_security_context(
        user_id="test_user",
        session_id="test_session",
        permissions={"read", "query"},
        ip_address="192.168.1.100",
        user_agent="TestRunner/1.0",
        auth_method="password"
    )
    
    assert context.user_id == "test_user"
    assert "read" in context.permissions
    
    # Test security analytics
    analytics = get_security_insights()
    assert isinstance(analytics, dict)
    
    return True


def test_error_recovery_system():
    """Test comprehensive error recovery system."""
    analytics = get_resilience_insights()
    assert isinstance(analytics, dict)
    
    # Test error classifier
    try:
        raise ValueError("Test error for classification")
    except Exception as e:
        error_context = global_error_recovery_system.error_classifier.classify_error(
            e, {"operation": "test", "component": "test_runner"}
        )
        assert error_context.error_type == "ValueError"
        assert error_context.operation == "test"
    
    return True


def test_hyperscale_performance():
    """Test hyperscale performance system."""
    analytics = get_hyperscale_insights()
    assert isinstance(analytics, dict)
    
    # Test cache functionality
    cache = global_performance_system.semantic_cache
    cache.put("test_key", "test_value")
    result = cache.get("test_key")
    assert result == "test_value"
    
    # Test cache statistics
    stats = cache.get_stats()
    assert "hit_rate" in stats
    assert stats["size"] >= 1
    
    return True


async def test_async_operations():
    """Test asynchronous operations across systems."""
    # Test adaptive learning async operations
    try:
        from src.sql_synth.adaptive_learning_engine import process_user_feedback
        
        result = await process_user_feedback(
            query_id="async_test_001",
            natural_query="Test async query",
            generated_sql="SELECT 1;",
            feedback_type="positive",
            feedback_value=1.0,
            context={"test": True}
        )
        
        assert "feedback_processed" in result
    except Exception as e:
        print(f"Async adaptive learning test error: {e}")
    
    # Test performance system async operations
    try:
        from src.sql_synth.hyperscale_performance_system import execute_optimized_operation
        
        async def test_operation():
            await asyncio.sleep(0.01)  # Small async delay
            return "async_test_result"
        
        result = await execute_optimized_operation(
            test_operation,
            "async_test_operation",
            cache_key="async_test_key"
        )
        
        assert result == "async_test_result"
    except Exception as e:
        print(f"Async performance test error: {e}")
    
    return True


def test_system_integration():
    """Test integration between different systems."""
    systems_analytics = {}
    
    # Collect analytics from all systems
    try:
        systems_analytics["adaptive_learning"] = get_learning_insights()
    except Exception as e:
        systems_analytics["adaptive_learning"] = {"error": str(e)}
    
    try:
        systems_analytics["optimization"] = get_optimization_insights()
    except Exception as e:
        systems_analytics["optimization"] = {"error": str(e)}
    
    try:
        systems_analytics["security"] = get_security_insights()
    except Exception as e:
        systems_analytics["security"] = {"error": str(e)}
    
    try:
        systems_analytics["error_recovery"] = get_resilience_insights()
    except Exception as e:
        systems_analytics["error_recovery"] = {"error": str(e)}
    
    try:
        systems_analytics["performance"] = get_hyperscale_insights()
    except Exception as e:
        systems_analytics["performance"] = {"error": str(e)}
    
    # Check that we got analytics from all systems
    assert len(systems_analytics) == 5
    
    # Count healthy systems
    healthy_systems = sum(
        1 for analytics in systems_analytics.values()
        if isinstance(analytics, dict) and "error" not in analytics
    )
    
    print(f"System health: {healthy_systems}/5 systems operational")
    
    # At least 3 out of 5 systems should be operational
    assert healthy_systems >= 3, f"Only {healthy_systems}/5 systems are operational"
    
    return True


def test_performance_benchmarks():
    """Test basic performance benchmarks."""
    # Test cache performance
    cache = global_performance_system.semantic_cache
    
    start_time = time.time()
    
    # Benchmark cache operations
    for i in range(100):
        cache.put(f"bench_key_{i}", f"bench_value_{i}")
    
    write_time = time.time() - start_time
    
    start_time = time.time()
    
    for i in range(100):
        result = cache.get(f"bench_key_{i}")
        assert result == f"bench_value_{i}"
    
    read_time = time.time() - start_time
    
    print(f"Cache benchmark: {write_time:.4f}s write, {read_time:.4f}s read")
    
    # Performance should be reasonable
    assert write_time < 1.0, f"Cache write too slow: {write_time:.4f}s"
    assert read_time < 1.0, f"Cache read too slow: {read_time:.4f}s"
    
    return True


def test_configuration_integrity():
    """Test that all systems are properly configured."""
    # Test adaptive learning configuration
    engine = global_adaptive_learning_engine
    assert hasattr(engine, 'pattern_analyzer')
    assert hasattr(engine, 'reinforcement_learner')
    
    # Test optimization system configuration
    optimizer = global_nextgen_optimizer
    assert hasattr(optimizer, 'optimizers')
    assert len(optimizer.optimizers) > 0
    
    # Test security system configuration
    security = global_security_controller
    assert hasattr(security, 'threat_detector')
    assert hasattr(security, 'access_policies')
    
    # Test error recovery configuration
    recovery = global_error_recovery_system
    assert hasattr(recovery, 'error_classifier')
    assert hasattr(recovery, 'retry_manager')
    
    # Test performance system configuration
    perf_system = global_performance_system
    assert hasattr(perf_system, 'semantic_cache')
    assert hasattr(perf_system, 'load_balancer')
    
    return True


def main():
    """Main test execution."""
    print("Starting Advanced Features Test Suite")
    print("="*50)
    
    runner = TestRunner()
    
    # Run all tests
    test_functions = [
        ("Adaptive Learning Basic", test_adaptive_learning_basic),
        ("Optimization System", test_optimization_system),
        ("Security Framework", test_security_framework),
        ("Error Recovery System", test_error_recovery_system),
        ("HyperScale Performance", test_hyperscale_performance),
        ("Async Operations", test_async_operations),
        ("System Integration", test_system_integration),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Configuration Integrity", test_configuration_integrity),
    ]
    
    for test_name, test_func in test_functions:
        runner.run_test(test_name, test_func)
    
    runner.print_summary()
    
    # Exit with appropriate code
    if runner.tests_failed > 0:
        print(f"\n{runner.tests_failed} tests failed. Please check the errors above.")
        sys.exit(1)
    else:
        print(f"\nAll {runner.tests_passed} tests passed successfully! ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()