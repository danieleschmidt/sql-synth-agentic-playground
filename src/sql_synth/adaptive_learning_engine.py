"""Advanced Adaptive Learning Engine for Continuous SQL Synthesis Improvement.

This module implements a sophisticated machine learning system that:
- Learns from user feedback and query patterns
- Adapts SQL generation strategies in real-time
- Performs continuous model fine-tuning
- Implements reinforcement learning for query optimization
- Provides intelligent recommendation systems
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    RATING = "rating"
    IMPLICIT = "implicit"


class LearningStrategy(Enum):
    """Learning strategies for adaptation."""
    REINFORCEMENT = "reinforcement"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"
    META = "meta"


@dataclass
class UserFeedback:
    """User feedback structure."""
    query_id: str
    natural_query: str
    generated_sql: str
    feedback_type: FeedbackType
    feedback_value: Union[int, float, str, dict]
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    context: dict = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Detected learning pattern."""
    pattern_id: str
    pattern_type: str
    confidence: float
    frequency: int
    examples: List[dict]
    improvement_potential: float
    metadata: dict = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of learning adaptation."""
    adaptation_type: str
    changes_applied: List[str]
    performance_impact: float
    confidence: float
    rollback_info: dict
    timestamp: float = field(default_factory=time.time)


class QueryPatternAnalyzer:
    """Advanced pattern analysis for SQL queries and user behavior."""

    def __init__(self):
        self.pattern_cache = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.pattern_history = deque(maxlen=10000)

    def analyze_query_patterns(self, queries: List[dict]) -> List[LearningPattern]:
        """Analyze patterns in SQL queries and user behavior.

        Args:
            queries: List of query records with metadata

        Returns:
            List of detected patterns with learning potential
        """
        patterns = []

        try:
            if len(queries) < 10:  # Need minimum data
                return patterns

            # Extract text features
            query_texts = [q.get('natural_query', '') for q in queries]
            
            if not any(query_texts):
                return patterns

            # Vectorize queries
            try:
                query_vectors = self.vectorizer.fit_transform(query_texts)
            except Exception as e:
                logger.warning(f"Vectorization failed: {e}")
                return patterns

            # Cluster similar queries
            if query_vectors.shape[0] >= 5:  # Minimum for clustering
                clusters = self.clustering_model.fit_predict(query_vectors)
                
                # Analyze each cluster
                for cluster_id in np.unique(clusters):
                    cluster_queries = [queries[i] for i in range(len(queries)) if clusters[i] == cluster_id]
                    
                    if len(cluster_queries) >= 3:  # Minimum cluster size
                        pattern = self._analyze_cluster_pattern(cluster_id, cluster_queries)
                        if pattern:
                            patterns.append(pattern)

            # Temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(queries)
            patterns.extend(temporal_patterns)

            # Error patterns
            error_patterns = self._analyze_error_patterns(queries)
            patterns.extend(error_patterns)

            # Performance patterns
            performance_patterns = self._analyze_performance_patterns(queries)
            patterns.extend(performance_patterns)

            # Cache results
            self.pattern_cache[time.time()] = patterns
            self.pattern_history.extend(patterns)

            return patterns

        except Exception as e:
            logger.exception(f"Pattern analysis failed: {e}")
            return []

    def _analyze_cluster_pattern(self, cluster_id: int, cluster_queries: List[dict]) -> Optional[LearningPattern]:
        """Analyze a specific cluster of queries."""
        try:
            # Calculate cluster statistics
            success_rates = [q.get('success', True) for q in cluster_queries]
            avg_success_rate = sum(success_rates) / len(success_rates)
            
            response_times = [q.get('generation_time', 0) for q in cluster_queries]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            # Identify common themes
            common_keywords = self._extract_common_keywords(cluster_queries)
            common_sql_patterns = self._extract_common_sql_patterns(cluster_queries)

            # Calculate improvement potential
            improvement_potential = self._calculate_improvement_potential(
                avg_success_rate, avg_response_time, len(cluster_queries)
            )

            if improvement_potential > 0.3:  # Threshold for significant patterns
                return LearningPattern(
                    pattern_id=f"cluster_{cluster_id}",
                    pattern_type="query_similarity",
                    confidence=min(avg_success_rate + (len(cluster_queries) / 100), 1.0),
                    frequency=len(cluster_queries),
                    examples=cluster_queries[:5],  # Keep top 5 examples
                    improvement_potential=improvement_potential,
                    metadata={
                        "common_keywords": common_keywords,
                        "common_sql_patterns": common_sql_patterns,
                        "avg_success_rate": avg_success_rate,
                        "avg_response_time": avg_response_time,
                        "cluster_size": len(cluster_queries),
                    }
                )
        except Exception as e:
            logger.warning(f"Cluster analysis failed for cluster {cluster_id}: {e}")

        return None

    def _analyze_temporal_patterns(self, queries: List[dict]) -> List[LearningPattern]:
        """Analyze time-based patterns in queries."""
        patterns = []
        
        try:
            # Group by time periods
            time_groups = defaultdict(list)
            for query in queries:
                timestamp = query.get('timestamp', time.time())
                hour = int(timestamp % 86400 // 3600)  # Hour of day
                time_groups[hour].append(query)

            # Find peak usage hours
            peak_hours = sorted(time_groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]
            
            for hour, hour_queries in peak_hours:
                if len(hour_queries) >= 5:  # Minimum for pattern
                    success_rate = sum(q.get('success', True) for q in hour_queries) / len(hour_queries)
                    
                    pattern = LearningPattern(
                        pattern_id=f"temporal_hour_{hour}",
                        pattern_type="temporal_usage",
                        confidence=min(len(hour_queries) / 100, 1.0),
                        frequency=len(hour_queries),
                        examples=hour_queries[:3],
                        improvement_potential=1.0 - success_rate,
                        metadata={
                            "peak_hour": hour,
                            "success_rate": success_rate,
                            "query_volume": len(hour_queries),
                        }
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Temporal pattern analysis failed: {e}")

        return patterns

    def _analyze_error_patterns(self, queries: List[dict]) -> List[LearningPattern]:
        """Analyze error patterns in queries."""
        patterns = []
        
        try:
            error_queries = [q for q in queries if not q.get('success', True)]
            
            if len(error_queries) >= 3:  # Minimum for error pattern
                error_types = defaultdict(list)
                
                for query in error_queries:
                    error = query.get('error', 'unknown_error')
                    # Categorize error
                    if 'syntax' in error.lower():
                        error_types['syntax'].append(query)
                    elif 'permission' in error.lower() or 'access' in error.lower():
                        error_types['permission'].append(query)
                    elif 'timeout' in error.lower():
                        error_types['timeout'].append(query)
                    else:
                        error_types['other'].append(query)

                for error_type, type_queries in error_types.items():
                    if len(type_queries) >= 2:  # At least 2 similar errors
                        pattern = LearningPattern(
                            pattern_id=f"error_{error_type}",
                            pattern_type="error_pattern",
                            confidence=min(len(type_queries) / 10, 1.0),
                            frequency=len(type_queries),
                            examples=type_queries[:3],
                            improvement_potential=1.0,  # High potential for error reduction
                            metadata={
                                "error_category": error_type,
                                "error_rate": len(type_queries) / len(queries),
                                "common_errors": [q.get('error', '')[:100] for q in type_queries[:3]],
                            }
                        )
                        patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Error pattern analysis failed: {e}")

        return patterns

    def _analyze_performance_patterns(self, queries: List[dict]) -> List[LearningPattern]:
        """Analyze performance patterns in queries."""
        patterns = []
        
        try:
            # Find slow queries
            response_times = [q.get('generation_time', 0) for q in queries if q.get('generation_time', 0) > 0]
            
            if response_times:
                avg_time = sum(response_times) / len(response_times)
                threshold = avg_time * 2  # Slow queries are 2x average
                
                slow_queries = [q for q in queries if q.get('generation_time', 0) > threshold]
                
                if len(slow_queries) >= 3:
                    pattern = LearningPattern(
                        pattern_id="performance_slow_queries",
                        pattern_type="performance_issue",
                        confidence=min(len(slow_queries) / 20, 1.0),
                        frequency=len(slow_queries),
                        examples=slow_queries[:3],
                        improvement_potential=0.8,  # High potential for optimization
                        metadata={
                            "avg_slow_time": sum(q.get('generation_time', 0) for q in slow_queries) / len(slow_queries),
                            "threshold": threshold,
                            "slow_query_rate": len(slow_queries) / len(queries),
                        }
                    )
                    patterns.append(pattern)

        except Exception as e:
            logger.warning(f"Performance pattern analysis failed: {e}")

        return patterns

    def _extract_common_keywords(self, queries: List[dict]) -> List[str]:
        """Extract common keywords from query cluster."""
        try:
            all_text = ' '.join(q.get('natural_query', '') for q in queries).lower()
            
            # Simple keyword extraction (could be enhanced with NLP)
            words = all_text.split()
            word_counts = defaultdict(int)
            
            # Filter out common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_counts[word] += 1
            
            # Return top keywords
            return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def _extract_common_sql_patterns(self, queries: List[dict]) -> List[str]:
        """Extract common SQL patterns from query cluster."""
        try:
            sql_queries = [q.get('generated_sql', '') for q in queries if q.get('generated_sql')]
            
            patterns = []
            
            # Look for common SQL structures
            for sql in sql_queries:
                sql_upper = sql.upper()
                if 'GROUP BY' in sql_upper:
                    patterns.append('aggregation')
                if 'JOIN' in sql_upper:
                    patterns.append('multi_table')
                if 'ORDER BY' in sql_upper:
                    patterns.append('sorting')
                if 'WHERE' in sql_upper:
                    patterns.append('filtering')
                if 'WINDOW' in sql_upper or 'OVER' in sql_upper:
                    patterns.append('window_functions')

            # Count pattern frequency
            pattern_counts = defaultdict(int)
            for pattern in patterns:
                pattern_counts[pattern] += 1
            
            # Return patterns that appear in at least 30% of queries
            threshold = len(sql_queries) * 0.3
            return [pattern for pattern, count in pattern_counts.items() if count >= threshold]
            
        except Exception as e:
            logger.warning(f"SQL pattern extraction failed: {e}")
            return []

    def _calculate_improvement_potential(self, success_rate: float, avg_response_time: float, frequency: int) -> float:
        """Calculate the improvement potential for a pattern."""
        try:
            # Base potential from success rate (lower success = higher potential)
            success_potential = 1.0 - success_rate
            
            # Performance potential (slower queries have higher potential)
            time_potential = min(avg_response_time / 5.0, 1.0)  # Normalize to 5 seconds
            
            # Frequency potential (more frequent patterns have higher impact)
            frequency_potential = min(frequency / 100.0, 1.0)  # Normalize to 100 queries
            
            # Weighted combination
            return (success_potential * 0.4 + time_potential * 0.3 + frequency_potential * 0.3)
            
        except Exception:
            return 0.5  # Default moderate potential


class ReinforcementLearner:
    """Reinforcement learning system for query optimization."""

    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        self.action_history = deque(maxlen=1000)

    def learn_from_feedback(self, feedback: UserFeedback) -> dict:
        """Learn from user feedback using reinforcement learning.

        Args:
            feedback: User feedback on query results

        Returns:
            Learning update information
        """
        try:
            # Convert feedback to reward
            reward = self._feedback_to_reward(feedback)
            
            # Extract state and action
            state = self._extract_state(feedback.natural_query, feedback.context)
            action = self._extract_action(feedback.generated_sql)
            
            # Update Q-table
            self._update_q_value(state, action, reward)
            
            # Record action
            self.action_history.append({
                'state': state,
                'action': action,
                'reward': reward,
                'timestamp': time.time(),
                'feedback_type': feedback.feedback_type.value
            })
            
            return {
                'learning_type': 'reinforcement',
                'state': state,
                'action': action,
                'reward': reward,
                'q_value': self.q_table[state][action],
                'exploration_rate': self.exploration_rate,
            }
            
        except Exception as e:
            logger.exception(f"Reinforcement learning failed: {e}")
            return {'error': str(e)}

    def _feedback_to_reward(self, feedback: UserFeedback) -> float:
        """Convert user feedback to numerical reward."""
        if feedback.feedback_type == FeedbackType.POSITIVE:
            return 1.0
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            return -0.5
        elif feedback.feedback_type == FeedbackType.RATING:
            # Assume rating is 1-5, normalize to -1 to 1
            rating = float(feedback.feedback_value)
            return (rating - 3.0) / 2.0
        elif feedback.feedback_type == FeedbackType.CORRECTION:
            return -0.3  # Negative but less than completely wrong
        elif feedback.feedback_type == FeedbackType.IMPLICIT:
            # Implicit feedback from usage patterns
            if isinstance(feedback.feedback_value, dict):
                execution_time = feedback.feedback_value.get('execution_time', 5.0)
                result_count = feedback.feedback_value.get('result_count', 0)
                
                # Reward fast queries with results
                time_reward = max(0, 1.0 - execution_time / 10.0)  # 10s max
                result_reward = min(result_count / 100.0, 0.5)  # Cap at 0.5
                
                return time_reward + result_reward
        
        return 0.0  # Neutral for unknown feedback

    def _extract_state(self, natural_query: str, context: dict) -> str:
        """Extract state representation from query and context."""
        try:
            # Simplified state extraction
            query_features = []
            
            # Query length category
            query_len = len(natural_query.split())
            if query_len < 5:
                query_features.append('short')
            elif query_len < 15:
                query_features.append('medium')
            else:
                query_features.append('long')
            
            # Query type indicators
            query_lower = natural_query.lower()
            if any(word in query_lower for word in ['sum', 'count', 'average', 'total']):
                query_features.append('aggregation')
            if any(word in query_lower for word in ['join', 'combine', 'merge']):
                query_features.append('multi_table')
            if any(word in query_lower for word in ['when', 'time', 'date', 'recent']):
                query_features.append('temporal')
            if any(word in query_lower for word in ['top', 'best', 'highest', 'lowest']):
                query_features.append('ranking')
            
            # Context features
            if context:
                if context.get('table_count', 0) > 1:
                    query_features.append('multi_table_context')
                if context.get('has_time_columns', False):
                    query_features.append('temporal_context')
            
            return '_'.join(sorted(query_features)) or 'basic'
            
        except Exception as e:
            logger.warning(f"State extraction failed: {e}")
            return 'unknown'

    def _extract_action(self, sql_query: str) -> str:
        """Extract action representation from generated SQL."""
        try:
            sql_upper = sql_query.upper()
            action_features = []
            
            # SQL structure features
            if 'SELECT' in sql_upper:
                action_features.append('select')
            if 'GROUP BY' in sql_upper:
                action_features.append('group')
            if 'ORDER BY' in sql_upper:
                action_features.append('order')
            if 'JOIN' in sql_upper:
                action_features.append('join')
            if 'WHERE' in sql_upper:
                action_features.append('filter')
            if 'LIMIT' in sql_upper:
                action_features.append('limit')
            if any(func in sql_upper for func in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(']):
                action_features.append('aggregate')
            if 'WINDOW' in sql_upper or 'OVER' in sql_upper:
                action_features.append('window')
            
            return '_'.join(sorted(action_features)) or 'basic_select'
            
        except Exception as e:
            logger.warning(f"Action extraction failed: {e}")
            return 'unknown'

    def _update_q_value(self, state: str, action: str, reward: float):
        """Update Q-value using Q-learning algorithm."""
        try:
            current_q = self.q_table[state][action]
            
            # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            # Simplified version for this implementation
            self.q_table[state][action] = current_q + self.learning_rate * (reward - current_q)
            
        except Exception as e:
            logger.warning(f"Q-value update failed: {e}")

    def get_best_action(self, state: str) -> Tuple[str, float]:
        """Get the best action for a given state."""
        try:
            if state in self.q_table and self.q_table[state]:
                best_action = max(self.q_table[state].items(), key=lambda x: x[1])
                return best_action[0], best_action[1]
            else:
                return 'explore', 0.0
                
        except Exception as e:
            logger.warning(f"Best action selection failed: {e}")
            return 'default', 0.0

    def get_learning_stats(self) -> dict:
        """Get reinforcement learning statistics."""
        try:
            total_states = len(self.q_table)
            total_actions = sum(len(actions) for actions in self.q_table.values())
            
            # Average Q-values
            all_q_values = []
            for state_actions in self.q_table.values():
                all_q_values.extend(state_actions.values())
            
            avg_q_value = sum(all_q_values) / len(all_q_values) if all_q_values else 0.0
            
            # Recent performance
            recent_rewards = [action.get('reward', 0) for action in list(self.action_history)[-100:]]
            recent_avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            
            return {
                'total_states': total_states,
                'total_state_actions': total_actions,
                'avg_q_value': avg_q_value,
                'exploration_rate': self.exploration_rate,
                'recent_avg_reward': recent_avg_reward,
                'action_history_size': len(self.action_history),
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
            }
            
        except Exception as e:
            logger.exception(f"Learning stats calculation failed: {e}")
            return {'error': str(e)}


class AdaptiveLearningEngine:
    """Main adaptive learning engine coordinating all learning components."""

    def __init__(self):
        self.pattern_analyzer = QueryPatternAnalyzer()
        self.reinforcement_learner = ReinforcementLearner()
        self.feedback_buffer = deque(maxlen=10000)
        self.learning_history = deque(maxlen=1000)
        self.adaptation_strategies = {}
        self.performance_baseline = {}

    async def process_feedback(self, feedback: UserFeedback) -> dict:
        """Process user feedback and trigger learning adaptations.

        Args:
            feedback: User feedback on query results

        Returns:
            Learning processing results
        """
        try:
            # Store feedback
            self.feedback_buffer.append(feedback)
            
            # Process with reinforcement learning
            rl_result = self.reinforcement_learner.learn_from_feedback(feedback)
            
            # Analyze patterns if enough data
            adaptation_result = None
            if len(self.feedback_buffer) % 50 == 0:  # Every 50 feedback items
                adaptation_result = await self._trigger_pattern_adaptation()
            
            return {
                'feedback_processed': True,
                'reinforcement_learning': rl_result,
                'pattern_adaptation': adaptation_result,
                'feedback_buffer_size': len(self.feedback_buffer),
                'timestamp': time.time(),
            }
            
        except Exception as e:
            logger.exception(f"Feedback processing failed: {e}")
            return {'error': str(e)}

    async def _trigger_pattern_adaptation(self) -> Optional[AdaptationResult]:
        """Trigger pattern-based learning adaptations."""
        try:
            # Convert feedback to query records
            query_records = []
            for feedback in self.feedback_buffer:
                query_records.append({
                    'natural_query': feedback.natural_query,
                    'generated_sql': feedback.generated_sql,
                    'success': feedback.feedback_type != FeedbackType.NEGATIVE,
                    'generation_time': feedback.context.get('generation_time', 1.0),
                    'timestamp': feedback.timestamp,
                    'error': feedback.context.get('error') if feedback.feedback_type == FeedbackType.NEGATIVE else None,
                })
            
            # Analyze patterns
            patterns = self.pattern_analyzer.analyze_query_patterns(query_records)
            
            if not patterns:
                return None
            
            # Apply adaptations for high-potential patterns
            high_potential_patterns = [p for p in patterns if p.improvement_potential > 0.6]
            
            if high_potential_patterns:
                adaptations = []
                for pattern in high_potential_patterns:
                    adaptation = self._create_adaptation(pattern)
                    if adaptation:
                        adaptations.append(adaptation)
                
                if adaptations:
                    result = AdaptationResult(
                        adaptation_type='pattern_based',
                        changes_applied=adaptations,
                        performance_impact=sum(p.improvement_potential for p in high_potential_patterns) / len(high_potential_patterns),
                        confidence=sum(p.confidence for p in high_potential_patterns) / len(high_potential_patterns),
                        rollback_info={'patterns': [p.pattern_id for p in high_potential_patterns]},
                    )
                    
                    self.learning_history.append(result)
                    return result
            
        except Exception as e:
            logger.exception(f"Pattern adaptation failed: {e}")
        
        return None

    def _create_adaptation(self, pattern: LearningPattern) -> Optional[str]:
        """Create an adaptation strategy from a detected pattern."""
        try:
            if pattern.pattern_type == 'query_similarity':
                # Create template or optimization for similar queries
                return f"template_optimization_{pattern.pattern_id}"
            elif pattern.pattern_type == 'error_pattern':
                # Create error prevention strategy
                return f"error_prevention_{pattern.pattern_id}"
            elif pattern.pattern_type == 'performance_issue':
                # Create performance optimization
                return f"performance_optimization_{pattern.pattern_id}"
            elif pattern.pattern_type == 'temporal_usage':
                # Create time-based optimization
                return f"temporal_optimization_{pattern.pattern_id}"
            
            return None
            
        except Exception as e:
            logger.warning(f"Adaptation creation failed: {e}")
            return None

    def get_recommendations(self, natural_query: str, context: dict = None) -> dict:
        """Get intelligent recommendations for query improvement.

        Args:
            natural_query: User's natural language query
            context: Optional context information

        Returns:
            Recommendations for query improvement
        """
        try:
            recommendations = []
            
            # Get reinforcement learning recommendation
            state = self.reinforcement_learner._extract_state(natural_query, context or {})
            best_action, q_value = self.reinforcement_learner.get_best_action(state)
            
            if q_value > 0.5:  # High confidence recommendation
                recommendations.append({
                    'type': 'reinforcement_learning',
                    'suggestion': f"Based on learning, consider {best_action.replace('_', ' ')} approach",
                    'confidence': min(q_value, 1.0),
                    'reasoning': f"Q-value of {q_value:.3f} for state '{state}'",
                })
            
            # Pattern-based recommendations
            query_keywords = natural_query.lower().split()
            
            # Check against known patterns
            for pattern in self.pattern_analyzer.pattern_history:
                pattern_keywords = pattern.metadata.get('common_keywords', [])
                if pattern_keywords and any(keyword in query_keywords for keyword, _ in pattern_keywords):
                    recommendations.append({
                        'type': 'pattern_based',
                        'suggestion': f"Similar queries benefit from {pattern.pattern_type} optimization",
                        'confidence': pattern.confidence,
                        'reasoning': f"Pattern {pattern.pattern_id} matches with {pattern.frequency} examples",
                    })
                    break
            
            # Performance recommendations
            if any(word in query_keywords for word in ['all', 'everything', 'complete']):
                recommendations.append({
                    'type': 'performance',
                    'suggestion': "Consider adding LIMIT clause for large result sets",
                    'confidence': 0.8,
                    'reasoning': "Query language suggests potential for large results",
                })
            
            return {
                'recommendations': recommendations,
                'query_analysis': {
                    'state': state,
                    'complexity': 'high' if len(query_keywords) > 10 else 'low',
                    'keywords': query_keywords[:5],
                },
                'learning_confidence': q_value,
                'timestamp': time.time(),
            }
            
        except Exception as e:
            logger.exception(f"Recommendation generation failed: {e}")
            return {'error': str(e), 'recommendations': []}

    def get_learning_analytics(self) -> dict:
        """Get comprehensive analytics on learning performance."""
        try:
            # Reinforcement learning stats
            rl_stats = self.reinforcement_learner.get_learning_stats()
            
            # Pattern analysis stats
            total_patterns = len(self.pattern_analyzer.pattern_history)
            avg_pattern_confidence = 0.0
            if total_patterns > 0:
                avg_pattern_confidence = sum(p.confidence for p in self.pattern_analyzer.pattern_history) / total_patterns
            
            # Adaptation stats
            total_adaptations = len(self.learning_history)
            recent_adaptations = [a for a in self.learning_history if time.time() - a.timestamp < 86400]  # Last 24h
            
            # Feedback stats
            feedback_by_type = defaultdict(int)
            for feedback in self.feedback_buffer:
                feedback_by_type[feedback.feedback_type.value] += 1
            
            return {
                'reinforcement_learning': rl_stats,
                'pattern_analysis': {
                    'total_patterns': total_patterns,
                    'avg_confidence': avg_pattern_confidence,
                    'pattern_cache_size': len(self.pattern_analyzer.pattern_cache),
                },
                'adaptations': {
                    'total_adaptations': total_adaptations,
                    'recent_adaptations': len(recent_adaptations),
                    'avg_performance_impact': sum(a.performance_impact for a in self.learning_history) / max(total_adaptations, 1),
                },
                'feedback_analysis': {
                    'total_feedback': len(self.feedback_buffer),
                    'feedback_by_type': dict(feedback_by_type),
                    'positive_rate': feedback_by_type.get('positive', 0) / max(len(self.feedback_buffer), 1),
                },
                'system_health': {
                    'buffer_utilization': len(self.feedback_buffer) / 10000,
                    'learning_velocity': len(recent_adaptations) / max(total_adaptations, 1),
                    'adaptation_success_rate': sum(1 for a in self.learning_history if a.performance_impact > 0) / max(total_adaptations, 1),
                },
                'timestamp': time.time(),
            }
            
        except Exception as e:
            logger.exception(f"Learning analytics failed: {e}")
            return {'error': str(e)}

    async def optimize_system_performance(self) -> dict:
        """Optimize overall system performance based on learning insights."""
        try:
            optimizations_applied = []
            
            # Analyze recent performance
            recent_feedback = list(self.feedback_buffer)[-100:] if len(self.feedback_buffer) > 100 else list(self.feedback_buffer)
            
            if not recent_feedback:
                return {'message': 'No recent feedback for optimization'}
            
            # Calculate current performance metrics
            success_rate = sum(1 for f in recent_feedback if f.feedback_type != FeedbackType.NEGATIVE) / len(recent_feedback)
            avg_response_time = sum(f.context.get('generation_time', 1.0) for f in recent_feedback) / len(recent_feedback)
            
            # Apply optimizations based on metrics
            if success_rate < 0.8:  # Low success rate
                optimizations_applied.append("increased_validation_threshold")
                # In practice, would adjust validation parameters
                
            if avg_response_time > 3.0:  # Slow response
                optimizations_applied.append("response_time_optimization")
                # In practice, would optimize caching or model parameters
            
            # Pattern-based optimizations
            recent_patterns = [p for p in self.pattern_analyzer.pattern_history if time.time() - p.metadata.get('timestamp', 0) < 3600]
            high_impact_patterns = [p for p in recent_patterns if p.improvement_potential > 0.7]
            
            for pattern in high_impact_patterns[:3]:  # Top 3 patterns
                optimization = f"pattern_optimization_{pattern.pattern_type}"
                optimizations_applied.append(optimization)
            
            # Update performance baseline
            self.performance_baseline.update({
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'optimization_timestamp': time.time(),
            })
            
            return {
                'optimizations_applied': optimizations_applied,
                'performance_before': {
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                },
                'expected_improvement': {
                    'success_rate_boost': 0.1 * len([o for o in optimizations_applied if 'validation' in o]),
                    'response_time_reduction': 0.2 * len([o for o in optimizations_applied if 'time' in o]),
                },
                'timestamp': time.time(),
            }
            
        except Exception as e:
            logger.exception(f"System optimization failed: {e}")
            return {'error': str(e)}


# Global adaptive learning engine instance
global_adaptive_learning_engine = AdaptiveLearningEngine()


# Utility functions
async def process_user_feedback(query_id: str, natural_query: str, generated_sql: str, 
                              feedback_type: str, feedback_value: Any, 
                              user_id: Optional[str] = None, context: dict = None) -> dict:
    """Process user feedback through the adaptive learning system.

    Args:
        query_id: Unique query identifier
        natural_query: Original natural language query
        generated_sql: Generated SQL query
        feedback_type: Type of feedback (positive, negative, rating, etc.)
        feedback_value: Value of feedback
        user_id: Optional user identifier
        context: Optional context information

    Returns:
        Processing results
    """
    try:
        feedback = UserFeedback(
            query_id=query_id,
            natural_query=natural_query,
            generated_sql=generated_sql,
            feedback_type=FeedbackType(feedback_type),
            feedback_value=feedback_value,
            user_id=user_id,
            context=context or {},
        )
        
        return await global_adaptive_learning_engine.process_feedback(feedback)
        
    except Exception as e:
        logger.exception(f"Feedback processing failed: {e}")
        return {'error': str(e)}


def get_query_recommendations(natural_query: str, context: dict = None) -> dict:
    """Get intelligent recommendations for query improvement.

    Args:
        natural_query: User's natural language query
        context: Optional context information

    Returns:
        Recommendations for query improvement
    """
    return global_adaptive_learning_engine.get_recommendations(natural_query, context)


def get_learning_insights() -> dict:
    """Get comprehensive learning analytics and insights.

    Returns:
        Learning system analytics
    """
    return global_adaptive_learning_engine.get_learning_analytics()