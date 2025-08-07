"""Advanced sentiment analysis for SQL query enhancement.

This module provides sentiment-aware SQL generation by analyzing the emotional
context and intent of natural language queries to generate more contextually
appropriate SQL queries with enhanced filtering and ordering.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .exceptions import (
    SentimentAnalysisError, 
    ValidationError, 
    ModelLoadingError,
    create_error_context,
    handle_exception_with_context
)

logger = logging.getLogger(__name__)


class SentimentPolarity(Enum):
    """Sentiment polarity classification."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class QueryIntent(Enum):
    """Query intent classification based on sentiment context."""
    ANALYTICAL = "analytical"           # Neutral, data-focused
    EXPLORATORY = "exploratory"         # Positive, discovery-focused  
    INVESTIGATIVE = "investigative"     # Negative, problem-focused
    COMPARATIVE = "comparative"         # Mixed, comparison-focused
    TRENDING = "trending"               # Positive, growth-focused
    PROBLEM_SOLVING = "problem_solving" # Negative, issue-focused


@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis result."""
    polarity: SentimentPolarity
    confidence: float
    compound_score: float
    positive: float
    neutral: float
    negative: float
    intent: QueryIntent
    emotional_keywords: List[str]
    temporal_bias: Optional[str]  # "recent", "historical", "trending"
    magnitude_bias: Optional[str]  # "top", "bottom", "extreme", "average"


class SentimentAwareAnalyzer:
    """Advanced sentiment analyzer for SQL query enhancement."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """Initialize the sentiment analyzer with multiple models.
        
        Args:
            model_name: HuggingFace model for transformer-based sentiment analysis
        """
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize transformer model for advanced sentiment analysis
        try:
            self.transformer_analyzer = pipeline(
                "sentiment-analysis", 
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            logger.info("Transformer model loaded: %s", model_name)
        except Exception as e:
            logger.warning("Failed to load transformer model: %s", e)
            self.transformer_analyzer = None
            # Store the loading error for potential retry
            self._model_loading_error = ModelLoadingError(
                message="Failed to load transformer model",
                model_name=model_name,
                model_type="transformer",
                error_details=str(e)
            )
        
        # Emotional keywords for context enhancement
        self.emotional_keywords = {
            "positive": ["good", "great", "excellent", "best", "top", "successful", 
                        "growth", "increase", "improvement", "outstanding", "peak"],
            "negative": ["bad", "worst", "poor", "decline", "decrease", "problem", 
                        "issue", "failure", "drop", "low", "concerning", "critical"],
            "temporal": ["recent", "latest", "new", "current", "today", "yesterday",
                        "last", "historical", "old", "previous", "trending", "ongoing"],
            "magnitude": ["highest", "lowest", "maximum", "minimum", "extreme", 
                         "average", "typical", "unusual", "significant", "major"]
        }
        
        # TF-IDF for semantic similarity
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
    def analyze(self, query: str) -> SentimentAnalysis:
        """Perform comprehensive sentiment analysis on query.
        
        Args:
            query: Natural language query to analyze
            
        Returns:
            SentimentAnalysis object with detailed sentiment information
        """
        start_time = time.time()
        
        # Input validation
        if not query or not query.strip():
            raise ValidationError(
                "Query cannot be empty",
                field_name="query",
                invalid_value=query,
                validation_rules=["non_empty", "string_type"]
            )
            
        if len(query) > 10000:  # Reasonable limit for sentiment analysis
            raise ValidationError(
                "Query too long for sentiment analysis", 
                field_name="query",
                invalid_value=len(query),
                validation_rules=["max_length_10000"]
            )
        
        try:
            # TextBlob analysis
            blob = TextBlob(query)
            polarity_score = blob.sentiment.polarity
            
            # VADER analysis
            vader_scores = self.vader.polarity_scores(query)
            
            # Transformer analysis (if available)
            transformer_scores = None
            if self.transformer_analyzer:
                try:
                    transformer_result = self.transformer_analyzer(query)[0]
                    transformer_scores = {
                        item['label'].lower(): item['score'] 
                        for item in transformer_result
                    }
                except Exception as e:
                    logger.warning("Transformer analysis failed: %s", e)
            
            # Combine scores for robust sentiment classification
            final_scores = self._combine_sentiment_scores(
                polarity_score, vader_scores, transformer_scores
            )
            
            # Classify polarity
            polarity = self._classify_polarity(final_scores['compound'])
            
            # Determine query intent
            intent = self._determine_intent(query, final_scores)
            
            # Extract emotional keywords
            emotional_keywords = self._extract_emotional_keywords(query)
            
            # Detect temporal and magnitude bias
            temporal_bias = self._detect_temporal_bias(query)
            magnitude_bias = self._detect_magnitude_bias(query)
            
            analysis_time = time.time() - start_time
            logger.debug("Sentiment analysis completed in %.2fs", analysis_time)
            
            return SentimentAnalysis(
                polarity=polarity,
                confidence=final_scores['confidence'],
                compound_score=final_scores['compound'],
                positive=final_scores['positive'],
                neutral=final_scores['neutral'],
                negative=final_scores['negative'],
                intent=intent,
                emotional_keywords=emotional_keywords,
                temporal_bias=temporal_bias,
                magnitude_bias=magnitude_bias
            )
            
        except Exception as e:
            analysis_time = time.time() - start_time
            context = create_error_context(
                operation="sentiment_analysis",
                query_length=len(query) if query else 0,
                analysis_time=analysis_time
            )
            
            # Determine specific error type
            if isinstance(e, (ValueError, TypeError)):
                raise ValidationError(
                    f"Invalid input for sentiment analysis: {str(e)}",
                    field_name="query",
                    invalid_value=query,
                    context=context
                )
            else:
                raise SentimentAnalysisError(
                    f"Sentiment analysis failed: {str(e)}",
                    query=query,
                    model_failures=["textblob", "vader", "transformer"],
                    context=context
                )
    
    def enhance_sql_with_sentiment(self, base_sql: str, sentiment: SentimentAnalysis) -> str:
        """Enhance SQL query based on sentiment analysis.
        
        Args:
            base_sql: Original SQL query
            sentiment: Sentiment analysis result
            
        Returns:
            Enhanced SQL query with sentiment-aware modifications
        """
        enhanced_sql = base_sql
        
        try:
            # Apply temporal bias
            if sentiment.temporal_bias:
                enhanced_sql = self._apply_temporal_bias(enhanced_sql, sentiment.temporal_bias)
            
            # Apply magnitude bias
            if sentiment.magnitude_bias:
                enhanced_sql = self._apply_magnitude_bias(enhanced_sql, sentiment.magnitude_bias)
            
            # Apply sentiment-based ordering
            enhanced_sql = self._apply_sentiment_ordering(enhanced_sql, sentiment)
            
            # Apply intent-based filtering
            enhanced_sql = self._apply_intent_filtering(enhanced_sql, sentiment.intent)
            
            logger.info("SQL enhanced with sentiment context: %s -> %s", 
                       sentiment.polarity.value, sentiment.intent.value)
            
            return enhanced_sql
            
        except Exception as e:
            logger.warning("SQL enhancement failed: %s", e)
            return base_sql
    
    def _combine_sentiment_scores(
        self, 
        textblob_polarity: float, 
        vader_scores: Dict[str, float], 
        transformer_scores: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine multiple sentiment analysis scores."""
        
        # Base scores from VADER
        combined = {
            'positive': vader_scores['pos'],
            'neutral': vader_scores['neu'],
            'negative': vader_scores['neg'],
            'compound': vader_scores['compound']
        }
        
        # Weight and combine with TextBlob
        textblob_weight = 0.3
        vader_weight = 0.4
        transformer_weight = 0.3
        
        if transformer_scores:
            # Combine all three models
            combined['compound'] = (
                vader_weight * vader_scores['compound'] +
                textblob_weight * textblob_polarity +
                transformer_weight * (
                    transformer_scores.get('positive', 0) - 
                    transformer_scores.get('negative', 0)
                )
            )
        else:
            # Combine only VADER and TextBlob
            combined['compound'] = (
                0.6 * vader_scores['compound'] + 0.4 * textblob_polarity
            )
        
        # Calculate confidence based on agreement between models
        confidence = self._calculate_confidence(
            textblob_polarity, vader_scores, transformer_scores
        )
        combined['confidence'] = confidence
        
        return combined
    
    def _calculate_confidence(
        self,
        textblob_polarity: float,
        vader_scores: Dict[str, float],
        transformer_scores: Optional[Dict[str, float]]
    ) -> float:
        """Calculate confidence based on model agreement."""
        
        scores = [textblob_polarity, vader_scores['compound']]
        if transformer_scores:
            transformer_compound = (
                transformer_scores.get('positive', 0) - 
                transformer_scores.get('negative', 0)
            )
            scores.append(transformer_compound)
        
        # Calculate standard deviation (lower = higher agreement = higher confidence)
        std_dev = np.std(scores)
        confidence = max(0.0, 1.0 - (std_dev / 2.0))  # Normalize to 0-1 range
        
        return confidence
    
    def _classify_polarity(self, compound_score: float) -> SentimentPolarity:
        """Classify sentiment polarity based on compound score."""
        if compound_score >= 0.5:
            return SentimentPolarity.VERY_POSITIVE
        elif compound_score >= 0.1:
            return SentimentPolarity.POSITIVE
        elif compound_score > -0.1:
            return SentimentPolarity.NEUTRAL
        elif compound_score > -0.5:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.VERY_NEGATIVE
    
    def _determine_intent(self, query: str, scores: Dict[str, float]) -> QueryIntent:
        """Determine query intent based on sentiment and keywords."""
        query_lower = query.lower()
        
        # Look for comparison indicators
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return QueryIntent.COMPARATIVE
        
        # Look for trending/growth indicators
        if any(word in query_lower for word in ['trend', 'growing', 'increasing', 'rising']):
            return QueryIntent.TRENDING
        
        # Based on sentiment polarity
        if scores['compound'] < -0.3:
            if any(word in query_lower for word in ['problem', 'issue', 'error', 'bug']):
                return QueryIntent.PROBLEM_SOLVING
            else:
                return QueryIntent.INVESTIGATIVE
        elif scores['compound'] > 0.3:
            return QueryIntent.EXPLORATORY
        else:
            return QueryIntent.ANALYTICAL
    
    def _extract_emotional_keywords(self, query: str) -> List[str]:
        """Extract emotional keywords from query."""
        query_lower = query.lower()
        found_keywords = []
        
        for category, keywords in self.emotional_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    found_keywords.append(keyword)
        
        return found_keywords
    
    def _detect_temporal_bias(self, query: str) -> Optional[str]:
        """Detect temporal bias in query."""
        query_lower = query.lower()
        
        recent_keywords = ['recent', 'latest', 'new', 'current', 'today', 'this']
        historical_keywords = ['historical', 'old', 'previous', 'past', 'before']
        trending_keywords = ['trend', 'trending', 'growing', 'increasing']
        
        if any(keyword in query_lower for keyword in recent_keywords):
            return "recent"
        elif any(keyword in query_lower for keyword in historical_keywords):
            return "historical"
        elif any(keyword in query_lower for keyword in trending_keywords):
            return "trending"
        
        return None
    
    def _detect_magnitude_bias(self, query: str) -> Optional[str]:
        """Detect magnitude bias in query."""
        query_lower = query.lower()
        
        top_keywords = ['best', 'top', 'highest', 'maximum', 'peak', 'most']
        bottom_keywords = ['worst', 'bottom', 'lowest', 'minimum', 'least']
        extreme_keywords = ['extreme', 'unusual', 'significant', 'major']
        average_keywords = ['average', 'typical', 'normal', 'standard']
        
        if any(keyword in query_lower for keyword in top_keywords):
            return "top"
        elif any(keyword in query_lower for keyword in bottom_keywords):
            return "bottom"
        elif any(keyword in query_lower for keyword in extreme_keywords):
            return "extreme"
        elif any(keyword in query_lower for keyword in average_keywords):
            return "average"
        
        return None
    
    def _apply_temporal_bias(self, sql: str, temporal_bias: str) -> str:
        """Apply temporal bias to SQL query."""
        sql_upper = sql.upper()
        
        if temporal_bias == "recent" and "ORDER BY" not in sql_upper:
            # Add recency ordering
            if ";" in sql:
                sql = sql.replace(";", " ORDER BY created_at DESC, updated_at DESC;")
        elif temporal_bias == "historical" and "ORDER BY" not in sql_upper:
            # Add historical ordering
            if ";" in sql:
                sql = sql.replace(";", " ORDER BY created_at ASC, updated_at ASC;")
        elif temporal_bias == "trending":
            # Add trending/growth filters and ordering
            if "WHERE" in sql_upper:
                sql = sql.replace(" WHERE ", " WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) AND ")
            elif "SELECT" in sql_upper:
                sql = sql.replace(" FROM ", " FROM ", 1)  # Placeholder for more complex logic
        
        return sql
    
    def _apply_magnitude_bias(self, sql: str, magnitude_bias: str) -> str:
        """Apply magnitude bias to SQL query."""
        if magnitude_bias in ["top", "bottom"]:
            if "LIMIT" not in sql.upper():
                # Add LIMIT for top/bottom queries
                if ";" in sql:
                    sql = sql.replace(";", " LIMIT 10;")
            
            if magnitude_bias == "top" and "ORDER BY" not in sql.upper():
                # Add descending order for "top" queries
                if ";" in sql:
                    sql = sql.replace(";", " ORDER BY value DESC, score DESC LIMIT 10;")
            elif magnitude_bias == "bottom" and "ORDER BY" not in sql.upper():
                # Add ascending order for "bottom" queries  
                if ";" in sql:
                    sql = sql.replace(";", " ORDER BY value ASC, score ASC LIMIT 10;")
        
        return sql
    
    def _apply_sentiment_ordering(self, sql: str, sentiment: SentimentAnalysis) -> str:
        """Apply sentiment-based ordering to SQL query."""
        if sentiment.polarity in [SentimentPolarity.POSITIVE, SentimentPolarity.VERY_POSITIVE]:
            # Positive sentiment: prefer successful/high-value results
            if "ORDER BY" not in sql.upper() and any(col in sql.lower() for col in ['rating', 'score', 'value']):
                sql = sql.replace(";", " ORDER BY rating DESC, score DESC;") if ";" in sql else sql
        elif sentiment.polarity in [SentimentPolarity.NEGATIVE, SentimentPolarity.VERY_NEGATIVE]:
            # Negative sentiment: prefer problematic/low-value results for investigation
            if "ORDER BY" not in sql.upper() and any(col in sql.lower() for col in ['error', 'issue', 'problem']):
                sql = sql.replace(";", " ORDER BY error_count DESC, issue_severity DESC;") if ";" in sql else sql
        
        return sql
    
    def _apply_intent_filtering(self, sql: str, intent: QueryIntent) -> str:
        """Apply intent-based filtering to SQL query."""
        if intent == QueryIntent.PROBLEM_SOLVING:
            # Add filters for problem-focused queries
            if "WHERE" in sql.upper():
                sql = sql.replace(" WHERE ", " WHERE status != 'resolved' AND ")
        elif intent == QueryIntent.EXPLORATORY:
            # Add diversity for exploratory queries
            if "LIMIT" not in sql.upper():
                sql = sql.replace(";", " LIMIT 50;") if ";" in sql else sql
        elif intent == QueryIntent.TRENDING:
            # Add time-based filters for trending queries
            if "WHERE" in sql.upper():
                sql = sql.replace(" WHERE ", " WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) AND ")
        
        return sql


# Global instance for easy access
sentiment_analyzer = SentimentAwareAnalyzer()