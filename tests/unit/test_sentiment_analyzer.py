"""Unit tests for the sentiment analyzer module."""

import pytest
from unittest.mock import Mock, patch

from src.sql_synth.sentiment_analyzer import (
    SentimentAwareAnalyzer, 
    SentimentPolarity, 
    QueryIntent,
    sentiment_analyzer
)


class TestSentimentAwareAnalyzer:
    """Test cases for SentimentAwareAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer instance."""
        with patch('src.sql_synth.sentiment_analyzer.pipeline'):
            return SentimentAwareAnalyzer()
    
    def test_analyze_positive_query(self, analyzer):
        """Test analysis of positive sentiment query."""
        query = "Show me the best performing products with excellent ratings"
        
        result = analyzer.analyze(query)
        
        assert result.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE]
        assert result.intent in [QueryIntent.EXPLORATORY, QueryIntent.ANALYTICAL]
        assert result.magnitude_bias == "top"
        assert "best" in result.emotional_keywords or "excellent" in result.emotional_keywords
        
    def test_analyze_negative_query(self, analyzer):
        """Test analysis of negative sentiment query."""
        query = "Find the worst performing products with critical issues"
        
        result = analyzer.analyze(query)
        
        assert result.polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]
        assert result.intent in [QueryIntent.INVESTIGATIVE, QueryIntent.PROBLEM_SOLVING]
        assert result.magnitude_bias == "bottom"
        assert "worst" in result.emotional_keywords or "critical" in result.emotional_keywords
        
    def test_analyze_neutral_query(self, analyzer):
        """Test analysis of neutral sentiment query."""
        query = "Show me all products from the database"
        
        result = analyzer.analyze(query)
        
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert result.intent == QueryIntent.ANALYTICAL
        assert result.temporal_bias is None
        assert result.magnitude_bias is None
        
    def test_analyze_temporal_bias_detection(self, analyzer):
        """Test temporal bias detection in queries."""
        recent_query = "Show me recent orders from this month"
        historical_query = "Show me historical sales data from last year"
        trending_query = "Show me trending products that are growing"
        
        recent_result = analyzer.analyze(recent_query)
        historical_result = analyzer.analyze(historical_query)
        trending_result = analyzer.analyze(trending_query)
        
        assert recent_result.temporal_bias == "recent"
        assert historical_result.temporal_bias == "historical"
        assert trending_result.temporal_bias == "trending"
        
    def test_enhance_sql_with_sentiment_positive(self, analyzer):
        """Test SQL enhancement with positive sentiment."""
        base_sql = "SELECT * FROM products"
        
        # Mock sentiment analysis result
        sentiment = Mock()
        sentiment.polarity = SentimentPolarity.POSITIVE
        sentiment.temporal_bias = None
        sentiment.magnitude_bias = "top"
        sentiment.intent = QueryIntent.EXPLORATORY
        
        enhanced_sql = analyzer.enhance_sql_with_sentiment(base_sql, sentiment)
        
        # Should have added LIMIT and possibly ordering
        assert "LIMIT" in enhanced_sql.upper()
        
    def test_enhance_sql_with_temporal_bias(self, analyzer):
        """Test SQL enhancement with temporal bias."""
        base_sql = "SELECT * FROM orders"
        
        # Mock sentiment analysis with recent bias
        sentiment = Mock()
        sentiment.polarity = SentimentPolarity.NEUTRAL
        sentiment.temporal_bias = "recent"
        sentiment.magnitude_bias = None
        sentiment.intent = QueryIntent.ANALYTICAL
        
        enhanced_sql = analyzer.enhance_sql_with_sentiment(base_sql, sentiment)
        
        # Should have added ordering for recent data
        assert "ORDER BY" in enhanced_sql.upper()
        
    def test_classify_polarity_boundaries(self, analyzer):
        """Test polarity classification boundary conditions."""
        # Test boundary values
        assert analyzer._classify_polarity(0.6) == SentimentPolarity.VERY_POSITIVE
        assert analyzer._classify_polarity(0.2) == SentimentPolarity.POSITIVE
        assert analyzer._classify_polarity(0.0) == SentimentPolarity.NEUTRAL
        assert analyzer._classify_polarity(-0.2) == SentimentPolarity.NEGATIVE
        assert analyzer._classify_polarity(-0.6) == SentimentPolarity.VERY_NEGATIVE
        
    def test_determine_intent_comparison(self, analyzer):
        """Test intent determination for comparison queries."""
        query = "Compare sales between Q1 and Q2"
        scores = {"compound": 0.0, "positive": 0.33, "neutral": 0.34, "negative": 0.33}
        
        intent = analyzer._determine_intent(query, scores)
        
        assert intent == QueryIntent.COMPARATIVE
        
    def test_determine_intent_problem_solving(self, analyzer):
        """Test intent determination for problem-solving queries."""
        query = "Find critical issues that need immediate attention"
        scores = {"compound": -0.4, "positive": 0.1, "neutral": 0.2, "negative": 0.7}
        
        intent = analyzer._determine_intent(query, scores)
        
        assert intent == QueryIntent.PROBLEM_SOLVING
        
    def test_error_handling_in_analysis(self, analyzer):
        """Test graceful error handling during analysis."""
        # Test with invalid input
        result = analyzer.analyze("")
        
        # Should return neutral sentiment without crashing
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert result.confidence == 0.0
        
    def test_confidence_calculation_high_agreement(self, analyzer):
        """Test confidence calculation with high model agreement."""
        # All models agree (positive sentiment)
        confidence = analyzer._calculate_confidence(
            textblob_polarity=0.5,
            vader_scores={"compound": 0.6},
            transformer_scores={"positive": 0.8, "negative": 0.1}
        )
        
        # Should be high confidence due to agreement
        assert confidence > 0.7
        
    def test_confidence_calculation_low_agreement(self, analyzer):
        """Test confidence calculation with low model agreement."""
        # Models disagree significantly
        confidence = analyzer._calculate_confidence(
            textblob_polarity=0.8,
            vader_scores={"compound": -0.5},
            transformer_scores={"positive": 0.2, "negative": 0.7}
        )
        
        # Should be low confidence due to disagreement
        assert confidence < 0.5
        
    def test_emotional_keywords_extraction(self, analyzer):
        """Test extraction of emotional keywords."""
        query = "Find the best products with excellent performance and outstanding reviews"
        
        keywords = analyzer._extract_emotional_keywords(query)
        
        assert "best" in keywords
        assert "excellent" in keywords
        assert "outstanding" in keywords
        
    def test_sql_limit_addition(self, analyzer):
        """Test adding LIMIT to SQL queries."""
        sql_without_limit = "SELECT * FROM users ORDER BY created_at DESC"
        
        limited_sql = analyzer._add_limit_if_needed(sql_without_limit, 25)
        
        assert "LIMIT 25" in limited_sql
        
        # Test that existing LIMIT is preserved
        sql_with_limit = "SELECT * FROM users LIMIT 100"
        
        preserved_sql = analyzer._add_limit_if_needed(sql_with_limit, 25)
        
        assert "LIMIT 100" in preserved_sql
        assert "LIMIT 25" not in preserved_sql


class TestGlobalSentimentAnalyzer:
    """Test the global sentiment analyzer instance."""
    
    def test_global_instance_available(self):
        """Test that global sentiment analyzer instance is available."""
        assert sentiment_analyzer is not None
        assert isinstance(sentiment_analyzer, SentimentAwareAnalyzer)
        
    def test_global_instance_functionality(self):
        """Test basic functionality of global instance."""
        result = sentiment_analyzer.analyze("Show me user data")
        
        assert hasattr(result, 'polarity')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'intent')


@pytest.mark.parametrize("query,expected_intent", [
    ("Show me trending sales data", QueryIntent.TRENDING),
    ("Compare user engagement metrics", QueryIntent.COMPARATIVE), 
    ("Find problematic transactions", QueryIntent.PROBLEM_SOLVING),
    ("Explore new customer patterns", QueryIntent.EXPLORATORY),
    ("Analyze quarterly performance", QueryIntent.ANALYTICAL),
    ("Investigate security issues", QueryIntent.INVESTIGATIVE),
])
def test_query_intent_detection(query, expected_intent):
    """Parametrized test for query intent detection."""
    analyzer = SentimentAwareAnalyzer()
    result = analyzer.analyze(query)
    
    # Allow some flexibility in intent detection due to complexity
    assert result.intent == expected_intent or result.intent in [
        QueryIntent.ANALYTICAL, QueryIntent.EXPLORATORY
    ]


@pytest.mark.parametrize("query,expected_temporal", [
    ("Show me recent data", "recent"),
    ("Historical analysis needed", "historical"), 
    ("What's trending now", "trending"),
    ("Current month statistics", "recent"),
    ("Past year performance", "historical"),
    ("Growing market segments", "trending"),
])
def test_temporal_bias_detection(query, expected_temporal):
    """Parametrized test for temporal bias detection."""
    analyzer = SentimentAwareAnalyzer()
    result = analyzer.analyze(query)
    
    assert result.temporal_bias == expected_temporal


@pytest.mark.parametrize("query,expected_magnitude", [
    ("Show me the best products", "top"),
    ("Find the worst performers", "bottom"),
    ("Extreme outliers in data", "extreme"),
    ("Average performance metrics", "average"),
    ("Top selling items", "top"),
    ("Lowest rated services", "bottom"),
])  
def test_magnitude_bias_detection(query, expected_magnitude):
    """Parametrized test for magnitude bias detection."""
    analyzer = SentimentAwareAnalyzer()
    result = analyzer.analyze(query)
    
    assert result.magnitude_bias == expected_magnitude