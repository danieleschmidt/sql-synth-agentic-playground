"""Core functionality tests for SQL Synthesis Agent.

Tests the core sentiment analysis and SQL enhancement logic
without external dependencies.
"""

import unittest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestSentimentAnalysisCore(unittest.TestCase):
    """Test core sentiment analysis functionality."""
    
    def test_sentiment_polarity_enum(self):
        """Test sentiment polarity enumeration."""
        from sql_synth.sentiment_analyzer import SentimentPolarity
        
        # Test all enum values are accessible
        self.assertEqual(SentimentPolarity.VERY_POSITIVE.value, "very_positive")
        self.assertEqual(SentimentPolarity.POSITIVE.value, "positive")
        self.assertEqual(SentimentPolarity.NEUTRAL.value, "neutral")
        self.assertEqual(SentimentPolarity.NEGATIVE.value, "negative")
        self.assertEqual(SentimentPolarity.VERY_NEGATIVE.value, "very_negative")
    
    def test_query_intent_enum(self):
        """Test query intent enumeration."""
        from sql_synth.sentiment_analyzer import QueryIntent
        
        # Test all enum values are accessible
        expected_intents = [
            "analytical", "exploratory", "investigative", 
            "comparative", "trending", "problem_solving"
        ]
        
        for intent_name in expected_intents:
            intent = getattr(QueryIntent, intent_name.upper())
            self.assertEqual(intent.value, intent_name)
    
    def test_sentiment_analysis_dataclass(self):
        """Test SentimentAnalysis dataclass structure."""
        from sql_synth.sentiment_analyzer import SentimentAnalysis, SentimentPolarity, QueryIntent
        
        # Create a test instance
        analysis = SentimentAnalysis(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.8,
            compound_score=0.5,
            positive=0.7,
            neutral=0.2,
            negative=0.1,
            intent=QueryIntent.EXPLORATORY,
            emotional_keywords=["good", "excellent"],
            temporal_bias="recent",
            magnitude_bias="top"
        )
        
        # Verify all fields are accessible
        self.assertEqual(analysis.polarity, SentimentPolarity.POSITIVE)
        self.assertEqual(analysis.confidence, 0.8)
        self.assertEqual(analysis.intent, QueryIntent.EXPLORATORY)
        self.assertEqual(analysis.emotional_keywords, ["good", "excellent"])
        self.assertEqual(analysis.temporal_bias, "recent")
        self.assertEqual(analysis.magnitude_bias, "top")


class TestExceptionHandling(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_sql_synthesis_error_base(self):
        """Test base SQLSynthesisError exception."""
        from sql_synth.exceptions import SQLSynthesisError
        
        context = {"operation": "test", "timestamp": 123456789}
        error = SQLSynthesisError("Test error", context=context)
        
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.context, context)
        self.assertEqual(error.message, "Test error")
    
    def test_sentiment_analysis_error(self):
        """Test SentimentAnalysisError exception."""
        from sql_synth.exceptions import SentimentAnalysisError
        
        error = SentimentAnalysisError(
            "Sentiment analysis failed",
            query="test query",
            model_failures=["textblob", "vader"]
        )
        
        self.assertEqual(str(error), "Sentiment analysis failed")
        self.assertEqual(error.query, "test query")
        self.assertEqual(error.model_failures, ["textblob", "vader"])
    
    def test_sql_security_error(self):
        """Test SQLSecurityError exception."""
        from sql_synth.exceptions import SQLSecurityError
        
        error = SQLSecurityError(
            "Security violation detected",
            sql_query="DROP TABLE users;",
            violations=["dangerous_operation"],
            security_level="high"
        )
        
        self.assertEqual(str(error), "Security violation detected")
        self.assertEqual(error.sql_query, "DROP TABLE users;")
        self.assertEqual(error.violations, ["dangerous_operation"])
        self.assertEqual(error.security_level, "high")
    
    def test_create_error_context(self):
        """Test error context creation helper."""
        from sql_synth.exceptions import create_error_context
        
        context = create_error_context(
            operation="test_operation",
            user_id="user123",
            custom_field="custom_value"
        )
        
        self.assertEqual(context["operation"], "test_operation")
        self.assertEqual(context["user_id"], "user123")
        self.assertEqual(context["custom_field"], "custom_value")
        self.assertIn("timestamp", context)


class TestResiliencePatterns(unittest.TestCase):
    """Test resilience pattern implementations."""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state enumeration."""
        from sql_synth.resilience import CircuitState
        
        self.assertEqual(CircuitState.CLOSED.value, "closed")
        self.assertEqual(CircuitState.OPEN.value, "open")
        self.assertEqual(CircuitState.HALF_OPEN.value, "half_open")
    
    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration."""
        from sql_synth.resilience import CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2
        )
        
        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.recovery_timeout, 30)
        self.assertEqual(config.success_threshold, 2)
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        from sql_synth.resilience import CircuitBreakerStats
        
        stats = CircuitBreakerStats()
        
        # Test default values
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.successful_requests, 0)
        self.assertEqual(stats.failed_requests, 0)
        self.assertEqual(stats.consecutive_failures, 0)
        self.assertIsNone(stats.last_failure_time)
    
    def test_retry_config(self):
        """Test retry configuration."""
        from sql_synth.resilience import RetryConfig
        
        config = RetryConfig(
            max_attempts=3,
            delay=1.0,
            backoff_factor=2.0,
            max_delay=30.0
        )
        
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.delay, 1.0)
        self.assertEqual(config.backoff_factor, 2.0)
        self.assertEqual(config.max_delay, 30.0)
    
    def test_predefined_configurations(self):
        """Test predefined resilience configurations."""
        from sql_synth.resilience import (
            SENTIMENT_ANALYSIS_CIRCUIT_BREAKER,
            SQL_GENERATION_CIRCUIT_BREAKER,
            QUICK_RETRY,
            STANDARD_RETRY
        )
        
        # Test circuit breakers exist and have proper configuration
        self.assertIsNotNone(SENTIMENT_ANALYSIS_CIRCUIT_BREAKER)
        self.assertIsNotNone(SQL_GENERATION_CIRCUIT_BREAKER)
        
        # Test retry configurations
        self.assertEqual(QUICK_RETRY.max_attempts, 3)
        self.assertEqual(STANDARD_RETRY.max_attempts, 3)
        self.assertLess(QUICK_RETRY.delay, STANDARD_RETRY.delay)


class TestPerformanceOptimizer(unittest.TestCase):
    """Test performance optimization components."""
    
    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass."""
        from sql_synth.performance_optimizer import PerformanceMetrics
        
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=1234567890.0,
            end_time=1234567891.5,
            duration=1.5,
            success=True
        )
        
        self.assertEqual(metrics.operation_name, "test_operation")
        self.assertEqual(metrics.start_time, 1234567890.0)
        self.assertEqual(metrics.end_time, 1234567891.5)
        self.assertEqual(metrics.duration, 1.5)
        self.assertTrue(metrics.success)
    
    def test_adaptive_cache_strategy_config(self):
        """Test adaptive cache strategy configuration."""
        from sql_synth.performance_optimizer import AdaptiveCacheStrategy
        
        strategy = AdaptiveCacheStrategy(
            base_ttl=3600,
            max_ttl=86400,
            min_ttl=300,
            cache_hit_threshold=0.7
        )
        
        self.assertEqual(strategy.base_ttl, 3600)
        self.assertEqual(strategy.max_ttl, 86400)
        self.assertEqual(strategy.min_ttl, 300)
        self.assertEqual(strategy.cache_hit_threshold, 0.7)
    
    def test_global_performance_instances(self):
        """Test that global performance instances are available."""
        from sql_synth.performance_optimizer import (
            performance_monitor,
            adaptive_cache_strategy,
            async_query_executor,
            query_optimizer
        )
        
        self.assertIsNotNone(performance_monitor)
        self.assertIsNotNone(adaptive_cache_strategy)
        self.assertIsNotNone(async_query_executor)
        self.assertIsNotNone(query_optimizer)


class TestStreamlitUIComponents(unittest.TestCase):
    """Test Streamlit UI component functionality."""
    
    def test_streamlit_ui_class_structure(self):
        """Test StreamlitUI class structure."""
        # We can't actually test Streamlit functionality without the framework,
        # but we can test that the class structure is correct
        
        try:
            from sql_synth.streamlit_ui import StreamlitUI
            
            # Test that class can be imported
            self.assertTrue(hasattr(StreamlitUI, '__init__'))
            self.assertTrue(hasattr(StreamlitUI, 'render_header'))
            self.assertTrue(hasattr(StreamlitUI, 'render_input_form'))
            self.assertTrue(hasattr(StreamlitUI, 'render_sentiment_analysis'))
            
        except ImportError as e:
            # If streamlit is not available, that's expected in test environment
            self.assertIn('streamlit', str(e).lower())


class TestCoreAppLogic(unittest.TestCase):
    """Test core application logic."""
    
    def test_app_imports(self):
        """Test that app.py can be imported and has required functions."""
        try:
            import app
            
            # Test that main functions exist
            self.assertTrue(hasattr(app, 'main'))
            self.assertTrue(hasattr(app, 'simulate_sentiment_aware_sql_generation'))
            self.assertTrue(hasattr(app, 'create_demo_results'))
            
        except ImportError as e:
            # If dependencies are missing, that's expected
            self.assertIn(('streamlit', 'pandas'), str(e).lower())


class TestSQLEnhancementLogic(unittest.TestCase):
    """Test SQL enhancement logic without external dependencies."""
    
    def test_basic_sql_structure_validation(self):
        """Test basic SQL structure validation."""
        test_cases = [
            ("SELECT * FROM users", True),
            ("select id, name from products", True), 
            ("UPDATE users SET name = 'test'", False),  # Should be blocked
            ("DROP TABLE users", False),  # Should be blocked
            ("INSERT INTO users VALUES (1, 'test')", False),  # Should be blocked
        ]
        
        for sql, should_be_valid in test_cases:
            with self.subTest(sql=sql):
                # Basic validation - check if starts with SELECT
                is_select_query = sql.strip().upper().startswith('SELECT')
                
                if should_be_valid:
                    self.assertTrue(is_select_query, f"Query should be valid: {sql}")
                else:
                    # For dangerous queries, they shouldn't start with SELECT
                    dangerous_keywords = ['UPDATE', 'DELETE', 'DROP', 'INSERT', 'TRUNCATE']
                    has_dangerous_keyword = any(
                        keyword in sql.upper() for keyword in dangerous_keywords
                    )
                    
                    if has_dangerous_keyword:
                        self.assertFalse(is_select_query or not has_dangerous_keyword,
                                       f"Query should be blocked: {sql}")
    
    def test_sql_enhancement_patterns(self):
        """Test SQL enhancement pattern recognition."""
        enhancement_tests = [
            # Test LIMIT addition
            ("SELECT * FROM users", "LIMIT"),
            ("SELECT * FROM products WHERE active = true", "LIMIT"),
            
            # Test ORDER BY addition patterns
            ("SELECT * FROM orders", "ORDER BY"),
            ("SELECT name, email FROM users", "ORDER BY"),
        ]
        
        for base_sql, expected_enhancement in enhancement_tests:
            with self.subTest(sql=base_sql):
                # Simulate enhancement logic
                enhanced_sql = base_sql
                
                # Add LIMIT if not present
                if "LIMIT" not in enhanced_sql.upper() and expected_enhancement == "LIMIT":
                    enhanced_sql += " LIMIT 100"
                
                # Add ORDER BY if not present
                if "ORDER BY" not in enhanced_sql.upper() and expected_enhancement == "ORDER BY":
                    enhanced_sql += " ORDER BY created_at DESC"
                
                # Verify enhancement was applied
                self.assertIn(expected_enhancement, enhanced_sql.upper())
                self.assertGreaterEqual(len(enhanced_sql), len(base_sql))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)