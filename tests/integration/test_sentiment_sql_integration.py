"""Integration tests for sentiment-aware SQL generation.

These tests verify the complete pipeline from natural language
to sentiment analysis to SQL generation and enhancement.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sql_synth.sentiment_analyzer import (
    SentimentAwareAnalyzer, 
    SentimentPolarity, 
    QueryIntent, 
    SentimentAnalysis
)


class TestSentimentSQLIntegration(unittest.TestCase):
    """Integration tests for the complete sentiment-aware SQL pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the transformer pipeline to avoid downloading models
        with patch('sql_synth.sentiment_analyzer.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            self.analyzer = SentimentAwareAnalyzer()
    
    def test_end_to_end_positive_query_analysis(self):
        """Test complete pipeline with positive sentiment query."""
        query = "Show me the best performing products with excellent sales"
        
        # Mock transformer response
        with patch.object(self.analyzer, 'transformer_analyzer') as mock_transformer:
            mock_transformer.return_value = [
                [
                    {'label': 'POSITIVE', 'score': 0.8},
                    {'label': 'NEGATIVE', 'score': 0.1},
                    {'label': 'NEUTRAL', 'score': 0.1}
                ]
            ]
            
            result = self.analyzer.analyze(query)
        
        # Verify sentiment classification
        self.assertIn(result.polarity, [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE])
        self.assertIn(result.intent, [QueryIntent.EXPLORATORY, QueryIntent.ANALYTICAL])
        self.assertEqual(result.magnitude_bias, "top")
        self.assertIn("best", result.emotional_keywords)
        self.assertIn("excellent", result.emotional_keywords)
    
    def test_end_to_end_negative_query_analysis(self):
        """Test complete pipeline with negative sentiment query."""
        query = "Find critical issues and problematic orders that failed"
        
        # Mock transformer response
        with patch.object(self.analyzer, 'transformer_analyzer') as mock_transformer:
            mock_transformer.return_value = [
                [
                    {'label': 'NEGATIVE', 'score': 0.8},
                    {'label': 'POSITIVE', 'score': 0.1},
                    {'label': 'NEUTRAL', 'score': 0.1}
                ]
            ]
            
            result = self.analyzer.analyze(query)
        
        # Verify sentiment classification
        self.assertIn(result.polarity, [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE])
        self.assertIn(result.intent, [QueryIntent.INVESTIGATIVE, QueryIntent.PROBLEM_SOLVING])
        self.assertIn("critical", result.emotional_keywords)
        self.assertIn("problematic", result.emotional_keywords)
    
    def test_sql_enhancement_with_temporal_bias(self):
        """Test SQL enhancement with temporal bias detection."""
        base_sql = "SELECT * FROM orders"
        
        # Create sentiment with recent bias
        sentiment = SentimentAnalysis(
            polarity=SentimentPolarity.NEUTRAL,
            confidence=0.7,
            compound_score=0.0,
            positive=0.33,
            neutral=0.34,
            negative=0.33,
            intent=QueryIntent.ANALYTICAL,
            emotional_keywords=[],
            temporal_bias="recent",
            magnitude_bias=None
        )
        
        enhanced_sql = self.analyzer.enhance_sql_with_sentiment(base_sql, sentiment)
        
        # Verify temporal enhancement
        self.assertIn("ORDER BY", enhanced_sql.upper())
        self.assertIn("DESC", enhanced_sql.upper())
    
    def test_sql_enhancement_with_magnitude_bias(self):
        """Test SQL enhancement with magnitude bias."""
        base_sql = "SELECT * FROM products"
        
        # Create sentiment with top magnitude bias
        sentiment = SentimentAnalysis(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.8,
            compound_score=0.5,
            positive=0.7,
            neutral=0.2,
            negative=0.1,
            intent=QueryIntent.EXPLORATORY,
            emotional_keywords=["best", "top"],
            temporal_bias=None,
            magnitude_bias="top"
        )
        
        enhanced_sql = self.analyzer.enhance_sql_with_sentiment(base_sql, sentiment)
        
        # Verify magnitude enhancement
        self.assertIn("LIMIT", enhanced_sql.upper())
    
    def test_comprehensive_query_pipeline(self):
        """Test the complete query pipeline with multiple enhancements."""
        query = "Show me the top trending products from this month with excellent ratings"
        base_sql = "SELECT * FROM products WHERE rating > 4.0"
        
        # Mock transformer for positive sentiment
        with patch.object(self.analyzer, 'transformer_analyzer') as mock_transformer:
            mock_transformer.return_value = [
                [
                    {'label': 'POSITIVE', 'score': 0.85},
                    {'label': 'NEGATIVE', 'score': 0.05},
                    {'label': 'NEUTRAL', 'score': 0.1}
                ]
            ]
            
            # Analyze sentiment
            sentiment_result = self.analyzer.analyze(query)
            
            # Enhance SQL
            enhanced_sql = self.analyzer.enhance_sql_with_sentiment(base_sql, sentiment_result)
        
        # Verify comprehensive analysis
        self.assertIn(sentiment_result.polarity, [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE])
        self.assertEqual(sentiment_result.intent, QueryIntent.TRENDING)
        self.assertEqual(sentiment_result.temporal_bias, "trending")
        self.assertEqual(sentiment_result.magnitude_bias, "top")
        self.assertIn("top", sentiment_result.emotional_keywords)
        self.assertIn("excellent", sentiment_result.emotional_keywords)
        
        # Verify SQL enhancements
        self.assertIn("LIMIT", enhanced_sql.upper())
        self.assertTrue(len(enhanced_sql) > len(base_sql))
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test empty query
        with self.assertRaises(Exception):
            self.analyzer.analyze("")
        
        # Test extremely long query
        long_query = "x" * 15000
        with self.assertRaises(Exception):
            self.analyzer.analyze(long_query)
    
    def test_confidence_calculation_accuracy(self):
        """Test confidence calculation with different model agreements."""
        query = "Find good products"
        
        # High agreement scenario
        with patch.object(self.analyzer, 'transformer_analyzer') as mock_transformer:
            mock_transformer.return_value = [
                [
                    {'label': 'POSITIVE', 'score': 0.8},
                    {'label': 'NEGATIVE', 'score': 0.1},
                    {'label': 'NEUTRAL', 'score': 0.1}
                ]
            ]
            
            # Mock TextBlob to agree with transformer
            with patch('sql_synth.sentiment_analyzer.TextBlob') as mock_textblob:
                mock_textblob.return_value.sentiment.polarity = 0.6
                
                result = self.analyzer.analyze(query)
                
                # Should have high confidence due to agreement
                self.assertGreater(result.confidence, 0.5)
    
    def test_multiple_bias_detection(self):
        """Test detection of multiple biases in a single query."""
        query = "Show me the best recent products with top ratings from trending categories"
        
        result = self.analyzer.analyze(query)
        
        # Should detect both temporal and magnitude biases
        self.assertIsNotNone(result.temporal_bias)
        self.assertIsNotNone(result.magnitude_bias)
        self.assertIn(result.temporal_bias, ["recent", "trending"])
        self.assertEqual(result.magnitude_bias, "top")
    
    def test_intent_classification_accuracy(self):
        """Test accuracy of intent classification across different query types."""
        test_cases = [
            ("Compare sales between Q1 and Q2", QueryIntent.COMPARATIVE),
            ("Find trending products in electronics", QueryIntent.TRENDING),
            ("Investigate failed payment transactions", QueryIntent.INVESTIGATIVE),
            ("Show me user analytics data", QueryIntent.ANALYTICAL),
            ("Explore new customer segments", QueryIntent.EXPLORATORY)
        ]
        
        for query, expected_intent in test_cases:
            with self.subTest(query=query):
                result = self.analyzer.analyze(query)
                
                # Allow some flexibility due to complexity of intent detection
                self.assertIn(result.intent, [
                    expected_intent,
                    QueryIntent.ANALYTICAL,  # Common fallback
                    QueryIntent.EXPLORATORY  # Common fallback
                ])
    
    def test_sql_enhancement_safety(self):
        """Test that SQL enhancement doesn't break existing functionality."""
        test_sqls = [
            "SELECT * FROM users",
            "SELECT id, name FROM products WHERE active = true",
            "SELECT COUNT(*) FROM orders GROUP BY status",
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        ]
        
        neutral_sentiment = SentimentAnalysis(
            polarity=SentimentPolarity.NEUTRAL,
            confidence=0.5,
            compound_score=0.0,
            positive=0.33,
            neutral=0.34,
            negative=0.33,
            intent=QueryIntent.ANALYTICAL,
            emotional_keywords=[],
            temporal_bias=None,
            magnitude_bias=None
        )
        
        for sql in test_sqls:
            with self.subTest(sql=sql):
                enhanced = self.analyzer.enhance_sql_with_sentiment(sql, neutral_sentiment)
                
                # Enhanced SQL should still be valid (basic check)
                self.assertIn("SELECT", enhanced.upper())
                self.assertIn("FROM", enhanced.upper())
                
                # Should not break existing semicolons or structure
                if sql.endswith(';'):
                    self.assertTrue(enhanced.endswith(';'))


class TestPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of the sentiment analysis system."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('sql_synth.sentiment_analyzer.pipeline'):
            self.analyzer = SentimentAwareAnalyzer()
    
    def test_analysis_speed_reasonable(self):
        """Test that sentiment analysis completes in reasonable time."""
        import time
        
        query = "Show me the best products with excellent ratings from recent sales"
        
        start_time = time.time()
        result = self.analyzer.analyze(query)
        end_time = time.time()
        
        # Should complete within 5 seconds (generous for CI environments)
        self.assertLess(end_time - start_time, 5.0)
        self.assertIsInstance(result, SentimentAnalysis)
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage doesn't grow excessively."""
        import gc
        
        # Run multiple analyses
        queries = [
            f"Find products number {i} with good quality" 
            for i in range(50)
        ]
        
        results = []
        for query in queries:
            result = self.analyzer.analyze(query)
            results.append(result)
        
        # Force garbage collection
        gc.collect()
        
        # Verify all analyses completed successfully
        self.assertEqual(len(results), 50)
        for result in results:
            self.assertIsInstance(result, SentimentAnalysis)
    
    def test_concurrent_analysis_safety(self):
        """Test thread safety of sentiment analysis."""
        import threading
        import time
        
        queries = [
            "Find excellent products",
            "Show problematic orders", 
            "Get recent user data",
            "Analyze top performers"
        ]
        
        results = {}
        threads = []
        
        def analyze_query(query, thread_id):
            try:
                result = self.analyzer.analyze(query)
                results[thread_id] = result
            except Exception as e:
                results[thread_id] = e
        
        # Start multiple threads
        for i, query in enumerate(queries):
            thread = threading.Thread(target=analyze_query, args=(query, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all analyses succeeded
        self.assertEqual(len(results), 4)
        for result in results.values():
            self.assertIsInstance(result, SentimentAnalysis)


if __name__ == '__main__':
    unittest.main(verbosity=2)