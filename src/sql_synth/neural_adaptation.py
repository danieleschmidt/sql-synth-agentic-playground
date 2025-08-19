"""Neural adaptation and autonomous learning system for SQL synthesis.

This module implements advanced neural adaptation capabilities including:
- Continuous learning from query patterns
- Adaptive model parameter tuning
- Neural network-based query optimization
- Autonomous pattern recognition and adaptation
- Self-improving SQL generation strategies
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of neural adaptations."""
    PARAMETER_TUNING = "parameter_tuning"
    PATTERN_RECOGNITION = "pattern_recognition"
    STRATEGY_EVOLUTION = "strategy_evolution"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CONTEXT_LEARNING = "context_learning"
    ERROR_CORRECTION = "error_correction"


@dataclass
class LearningPattern:
    """Represents a learned pattern from query data."""
    pattern_id: str
    pattern_type: str
    features: dict[str, Any]
    success_rate: float
    usage_count: int
    confidence: float
    last_updated: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptationResult:
    """Result of neural adaptation process."""
    adaptation_type: AdaptationType
    success: bool
    improvement_score: float
    parameters_changed: dict[str, Any]
    patterns_learned: list[str]
    confidence: float
    adaptation_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryPatternAnalyzer:
    """Advanced query pattern analysis and recognition system."""

    def __init__(self):
        self.patterns = {}
        self.pattern_cache = {}
        self.analysis_history = []

    def analyze_query_patterns(self, query_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze patterns in query history for learning opportunities.

        Args:
            query_history: List of historical query data

        Returns:
            Comprehensive pattern analysis
        """
        if not query_history:
            return {"error": "No query history provided"}

        try:
            # Extract various pattern types
            linguistic_patterns = self._analyze_linguistic_patterns(query_history)
            structural_patterns = self._analyze_structural_patterns(query_history)
            performance_patterns = self._analyze_performance_patterns(query_history)
            temporal_patterns = self._analyze_temporal_patterns(query_history)
            context_patterns = self._analyze_context_patterns(query_history)

            # Identify emerging patterns
            emerging_patterns = self._identify_emerging_patterns(query_history)

            # Calculate pattern significance
            pattern_significance = self._calculate_pattern_significance([
                linguistic_patterns, structural_patterns, performance_patterns,
                temporal_patterns, context_patterns,
            ])

            analysis_result = {
                "linguistic_patterns": linguistic_patterns,
                "structural_patterns": structural_patterns,
                "performance_patterns": performance_patterns,
                "temporal_patterns": temporal_patterns,
                "context_patterns": context_patterns,
                "emerging_patterns": emerging_patterns,
                "pattern_significance": pattern_significance,
                "total_queries_analyzed": len(query_history),
                "analysis_timestamp": time.time(),
                "learning_opportunities": self._identify_learning_opportunities(
                    linguistic_patterns, structural_patterns, performance_patterns,
                ),
            }

            self.analysis_history.append(analysis_result)
            return analysis_result

        except Exception as e:
            logger.exception(f"Pattern analysis failed: {e}")
            return {"error": str(e), "analysis_timestamp": time.time()}

    def _analyze_linguistic_patterns(self, query_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze linguistic patterns in natural language queries."""
        linguistic_features = {
            "common_phrases": {},
            "query_complexity": [],
            "question_types": {},
            "domain_vocabulary": {},
            "syntax_patterns": {},
        }

        for query_data in query_history:
            original_query = query_data.get("metadata", {}).get("original_query", "")
            if not original_query:
                continue

            # Extract common phrases (simplified)
            words = original_query.lower().split()
            for i in range(len(words) - 1):
                bigram = " ".join(words[i:i+2])
                linguistic_features["common_phrases"][bigram] = \
                    linguistic_features["common_phrases"].get(bigram, 0) + 1

            # Query complexity (word count, special keywords)
            complexity_score = len(words)
            complexity_keywords = ["join", "group", "order", "having", "window", "recursive"]
            complexity_score += sum(1 for word in words if word in complexity_keywords) * 2
            linguistic_features["query_complexity"].append(complexity_score)

            # Question types
            question_starters = ["what", "how", "when", "where", "which", "show", "find", "get"]
            for starter in question_starters:
                if original_query.lower().startswith(starter):
                    linguistic_features["question_types"][starter] = \
                        linguistic_features["question_types"].get(starter, 0) + 1
                    break

            # Domain vocabulary
            domain_terms = ["customer", "order", "product", "user", "transaction", "revenue", "sales"]
            for term in domain_terms:
                if term in original_query.lower():
                    linguistic_features["domain_vocabulary"][term] = \
                        linguistic_features["domain_vocabulary"].get(term, 0) + 1

        # Calculate statistics
        return {
            "top_phrases": sorted(linguistic_features["common_phrases"].items(),
                                key=lambda x: x[1], reverse=True)[:10],
            "avg_complexity": np.mean(linguistic_features["query_complexity"]) if linguistic_features["query_complexity"] else 0,
            "complexity_distribution": {
                "simple": sum(1 for c in linguistic_features["query_complexity"] if c <= 5),
                "medium": sum(1 for c in linguistic_features["query_complexity"] if 5 < c <= 15),
                "complex": sum(1 for c in linguistic_features["query_complexity"] if c > 15),
            },
            "question_type_distribution": linguistic_features["question_types"],
            "domain_focus": max(linguistic_features["domain_vocabulary"].items(),
                              key=lambda x: x[1])[0] if linguistic_features["domain_vocabulary"] else "general",
            "vocabulary_diversity": len(linguistic_features["domain_vocabulary"]),
        }

    def _analyze_structural_patterns(self, query_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze structural patterns in generated SQL queries."""
        structural_features = {
            "sql_keywords": {},
            "join_patterns": {},
            "aggregation_patterns": {},
            "complexity_metrics": [],
            "clause_combinations": {},
        }

        for query_data in query_history:
            sql_query = query_data.get("sql_query", "")
            if not sql_query:
                continue

            sql_upper = sql_query.upper()

            # SQL keywords frequency
            keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY",
                       "HAVING", "LIMIT", "WITH", "CASE", "UNION"]
            for keyword in keywords:
                if keyword in sql_upper:
                    structural_features["sql_keywords"][keyword] = \
                        structural_features["sql_keywords"].get(keyword, 0) + 1

            # Join patterns
            join_types = ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN", "CROSS JOIN"]
            for join_type in join_types:
                if join_type in sql_upper:
                    structural_features["join_patterns"][join_type] = \
                        structural_features["join_patterns"].get(join_type, 0) + 1

            # Aggregation patterns
            agg_functions = ["COUNT", "SUM", "AVG", "MAX", "MIN", "GROUP_CONCAT"]
            for func in agg_functions:
                if func in sql_upper:
                    structural_features["aggregation_patterns"][func] = \
                        structural_features["aggregation_patterns"].get(func, 0) + 1

            # Query complexity
            complexity = len(sql_query.split()) + sql_upper.count("(") + sql_upper.count("SELECT")
            structural_features["complexity_metrics"].append(complexity)

            # Clause combinations
            clauses = []
            if "WHERE" in sql_upper: clauses.append("WHERE")
            if "GROUP BY" in sql_upper: clauses.append("GROUP_BY")
            if "ORDER BY" in sql_upper: clauses.append("ORDER_BY")
            if "HAVING" in sql_upper: clauses.append("HAVING")

            clause_combo = "_".join(sorted(clauses))
            if clause_combo:
                structural_features["clause_combinations"][clause_combo] = \
                    structural_features["clause_combinations"].get(clause_combo, 0) + 1

        return {
            "keyword_frequency": structural_features["sql_keywords"],
            "join_pattern_distribution": structural_features["join_patterns"],
            "aggregation_usage": structural_features["aggregation_patterns"],
            "avg_sql_complexity": np.mean(structural_features["complexity_metrics"]) if structural_features["complexity_metrics"] else 0,
            "complexity_range": {
                "min": min(structural_features["complexity_metrics"]) if structural_features["complexity_metrics"] else 0,
                "max": max(structural_features["complexity_metrics"]) if structural_features["complexity_metrics"] else 0,
                "std": np.std(structural_features["complexity_metrics"]) if len(structural_features["complexity_metrics"]) > 1 else 0,
            },
            "popular_clause_combinations": sorted(structural_features["clause_combinations"].items(),
                                                key=lambda x: x[1], reverse=True)[:5],
        }

    def _analyze_performance_patterns(self, query_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze performance patterns and optimization opportunities."""
        performance_data = {
            "generation_times": [],
            "success_rates": {},
            "error_patterns": {},
            "optimization_opportunities": [],
        }

        for query_data in query_history:
            # Generation time analysis
            gen_time = query_data.get("generation_time", 0)
            performance_data["generation_times"].append(gen_time)

            # Success/failure patterns
            success = query_data.get("success", False)
            query_type = self._classify_query_type(query_data.get("sql_query", ""))

            if query_type not in performance_data["success_rates"]:
                performance_data["success_rates"][query_type] = {"total": 0, "successful": 0}

            performance_data["success_rates"][query_type]["total"] += 1
            if success:
                performance_data["success_rates"][query_type]["successful"] += 1

            # Error pattern analysis
            if not success and "error" in query_data:
                error_type = self._classify_error_type(query_data["error"])
                performance_data["error_patterns"][error_type] = \
                    performance_data["error_patterns"].get(error_type, 0) + 1

        # Calculate success rates
        success_rate_summary = {}
        for query_type, stats in performance_data["success_rates"].items():
            success_rate_summary[query_type] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0

        # Identify optimization opportunities
        if performance_data["generation_times"]:
            avg_time = np.mean(performance_data["generation_times"])
            slow_queries = [t for t in performance_data["generation_times"] if t > avg_time * 1.5]
            if slow_queries:
                performance_data["optimization_opportunities"].append(
                    f"Optimize slow queries: {len(slow_queries)} queries above 1.5x average time",
                )

        for query_type, rate in success_rate_summary.items():
            if rate < 0.8:  # Less than 80% success rate
                performance_data["optimization_opportunities"].append(
                    f"Improve {query_type} queries: {rate:.1%} success rate",
                )

        return {
            "avg_generation_time": np.mean(performance_data["generation_times"]) if performance_data["generation_times"] else 0,
            "generation_time_distribution": {
                "p50": np.percentile(performance_data["generation_times"], 50) if performance_data["generation_times"] else 0,
                "p90": np.percentile(performance_data["generation_times"], 90) if performance_data["generation_times"] else 0,
                "p95": np.percentile(performance_data["generation_times"], 95) if performance_data["generation_times"] else 0,
            },
            "success_rates_by_type": success_rate_summary,
            "error_pattern_distribution": performance_data["error_patterns"],
            "optimization_opportunities": performance_data["optimization_opportunities"],
        }

    def _analyze_temporal_patterns(self, query_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze temporal patterns in query usage."""
        temporal_data = {
            "hourly_distribution": [0] * 24,
            "daily_patterns": {},
            "query_frequency_trends": [],
            "seasonal_patterns": {},
        }

        for query_data in query_history:
            timestamp = query_data.get("metadata", {}).get("timestamp", time.time())

            # Hour of day analysis
            hour = time.localtime(timestamp).tm_hour
            temporal_data["hourly_distribution"][hour] += 1

            # Day of week patterns
            day = time.strftime("%A", time.localtime(timestamp))
            temporal_data["daily_patterns"][day] = temporal_data["daily_patterns"].get(day, 0) + 1

            # Monthly patterns (simplified seasonality)
            month = time.strftime("%B", time.localtime(timestamp))
            temporal_data["seasonal_patterns"][month] = temporal_data["seasonal_patterns"].get(month, 0) + 1

        return {
            "peak_hours": [i for i, count in enumerate(temporal_data["hourly_distribution"])
                          if count == max(temporal_data["hourly_distribution"])],
            "hourly_distribution": temporal_data["hourly_distribution"],
            "busiest_day": max(temporal_data["daily_patterns"].items(), key=lambda x: x[1])[0]
                          if temporal_data["daily_patterns"] else "Unknown",
            "daily_distribution": temporal_data["daily_patterns"],
            "seasonal_trends": temporal_data["seasonal_patterns"],
        }

    def _analyze_context_patterns(self, query_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze contextual patterns in queries."""
        context_data = {
            "domain_patterns": {},
            "user_patterns": {},
            "session_patterns": {},
            "complexity_contexts": {},
        }

        for query_data in query_history:
            metadata = query_data.get("metadata", {})

            # Domain context
            domain = metadata.get("domain", "general")
            context_data["domain_patterns"][domain] = context_data["domain_patterns"].get(domain, 0) + 1

            # User context (if available)
            user_type = metadata.get("user_type", "unknown")
            context_data["user_patterns"][user_type] = context_data["user_patterns"].get(user_type, 0) + 1

            # Session context
            session_id = metadata.get("session_id", "unknown")
            if session_id != "unknown":
                context_data["session_patterns"][session_id] = \
                    context_data["session_patterns"].get(session_id, 0) + 1

            # Complexity in context
            complexity = metadata.get("estimated_complexity", "unknown")
            context_data["complexity_contexts"][complexity] = \
                context_data["complexity_contexts"].get(complexity, 0) + 1

        return {
            "dominant_domain": max(context_data["domain_patterns"].items(), key=lambda x: x[1])[0]
                              if context_data["domain_patterns"] else "general",
            "domain_distribution": context_data["domain_patterns"],
            "user_type_distribution": context_data["user_patterns"],
            "avg_queries_per_session": np.mean(list(context_data["session_patterns"].values()))
                                      if context_data["session_patterns"] else 0,
            "complexity_distribution": context_data["complexity_contexts"],
        }

    def _identify_emerging_patterns(self, query_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify emerging patterns in recent queries."""
        if len(query_history) < 10:
            return []

        # Split into recent and historical
        split_point = len(query_history) // 2
        historical = query_history[:split_point]
        recent = query_history[split_point:]

        emerging_patterns = []

        # Compare keyword usage
        historical_keywords = self._extract_keywords(historical)
        recent_keywords = self._extract_keywords(recent)

        for keyword, recent_count in recent_keywords.items():
            historical_count = historical_keywords.get(keyword, 0)
            recent_rate = recent_count / len(recent)
            historical_rate = historical_count / len(historical) if len(historical) > 0 else 0

            if recent_rate > historical_rate * 1.5:  # 50% increase
                emerging_patterns.append({
                    "type": "keyword_emergence",
                    "pattern": keyword,
                    "recent_frequency": recent_rate,
                    "historical_frequency": historical_rate,
                    "growth_factor": recent_rate / max(historical_rate, 0.001),
                })

        return sorted(emerging_patterns, key=lambda x: x.get("growth_factor", 0), reverse=True)[:5]

    def _extract_keywords(self, queries: list[dict[str, Any]]) -> dict[str, int]:
        """Extract and count keywords from queries."""
        keywords = {}
        for query_data in queries:
            sql_query = query_data.get("sql_query", "")
            if sql_query:
                words = sql_query.upper().split()
                for word in words:
                    if word.isalpha() and len(word) > 2:
                        keywords[word] = keywords.get(word, 0) + 1
        return keywords

    def _classify_query_type(self, sql_query: str) -> str:
        """Classify SQL query type."""
        if not sql_query:
            return "unknown"

        sql_upper = sql_query.upper()

        if "JOIN" in sql_upper:
            return "join_query"
        if "GROUP BY" in sql_upper:
            return "aggregation_query"
        if "ORDER BY" in sql_upper:
            return "sorted_query"
        if "WHERE" in sql_upper:
            return "filtered_query"
        return "simple_query"

    def _classify_error_type(self, error_message: str) -> str:
        """Classify error type from error message."""
        error_lower = error_message.lower()

        if "syntax" in error_lower or "parse" in error_lower:
            return "syntax_error"
        if "table" in error_lower and "not found" in error_lower:
            return "table_not_found"
        if "column" in error_lower and "not found" in error_lower:
            return "column_not_found"
        if "timeout" in error_lower:
            return "timeout_error"
        if "permission" in error_lower or "access" in error_lower:
            return "permission_error"
        return "unknown_error"

    def _calculate_pattern_significance(self, pattern_analyses: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate significance scores for different pattern types."""
        significance_scores = {}

        for i, analysis in enumerate(pattern_analyses):
            pattern_type = ["linguistic", "structural", "performance", "temporal", "context"][i]

            # Calculate entropy-based significance
            if isinstance(analysis, dict):
                values = []
                for _key, value in analysis.items():
                    if isinstance(value, (int, float)):
                        values.append(value)
                    elif isinstance(value, dict):
                        values.extend([v for v in value.values() if isinstance(v, (int, float))])

                if values:
                    # Normalize values
                    total = sum(values)
                    if total > 0:
                        probabilities = [v / total for v in values]
                        # Calculate entropy (lower entropy = more significant patterns)
                        pattern_entropy = entropy(probabilities)
                        significance_scores[pattern_type] = max(0, 1 - pattern_entropy / np.log(len(probabilities)))
                    else:
                        significance_scores[pattern_type] = 0.0
                else:
                    significance_scores[pattern_type] = 0.0
            else:
                significance_scores[pattern_type] = 0.0

        return significance_scores

    def _identify_learning_opportunities(
        self,
        linguistic_patterns: dict[str, Any],
        structural_patterns: dict[str, Any],
        performance_patterns: dict[str, Any],
    ) -> list[str]:
        """Identify specific learning opportunities from pattern analysis."""
        opportunities = []

        # Linguistic learning opportunities
        if linguistic_patterns.get("vocabulary_diversity", 0) > 5:
            opportunities.append("High vocabulary diversity - implement domain-specific optimizations")

        complexity_dist = linguistic_patterns.get("complexity_distribution", {})
        if complexity_dist.get("complex", 0) > complexity_dist.get("simple", 0):
            opportunities.append("Complex query dominance - enhance advanced pattern recognition")

        # Structural learning opportunities
        join_usage = sum(structural_patterns.get("join_pattern_distribution", {}).values())
        if join_usage > len(structural_patterns.get("keyword_frequency", {})) * 0.3:
            opportunities.append("High join usage - optimize multi-table query generation")

        # Performance learning opportunities
        avg_time = performance_patterns.get("avg_generation_time", 0)
        if avg_time > 2.0:  # More than 2 seconds average
            opportunities.append("High generation times - implement performance optimization learning")

        success_rates = performance_patterns.get("success_rates_by_type", {})
        low_success_types = [qtype for qtype, rate in success_rates.items() if rate < 0.8]
        if low_success_types:
            opportunities.append(f"Low success rates for {', '.join(low_success_types)} - targeted improvement needed")

        return opportunities


class NeuralParameterOptimizer:
    """Neural parameter optimization system for SQL synthesis."""

    def __init__(self):
        self.optimization_history = []
        self.current_parameters = {
            "temperature": 0.0,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        self.performance_baseline = None

    def optimize_parameters(
        self,
        performance_data: dict[str, Any],
        optimization_target: str = "accuracy",
    ) -> dict[str, Any]:
        """Optimize neural parameters based on performance data.

        Args:
            performance_data: Historical performance metrics
            optimization_target: Target metric to optimize ('accuracy', 'speed', 'balance')

        Returns:
            Optimization results with new parameters
        """
        try:
            start_time = time.time()

            # Establish baseline if not exists
            if self.performance_baseline is None:
                self.performance_baseline = self._calculate_baseline_performance(performance_data)

            # Define optimization objective
            objective_function = self._create_objective_function(performance_data, optimization_target)

            # Set parameter bounds
            parameter_bounds = self._get_parameter_bounds()

            # Perform optimization
            optimization_result = self._run_optimization(objective_function, parameter_bounds)

            # Validate optimized parameters
            validation_result = self._validate_parameters(optimization_result.x, performance_data)

            optimization_time = time.time() - start_time

            # Update current parameters if improvement found
            if validation_result["improvement_score"] > 0:
                self.current_parameters = self._array_to_parameters(optimization_result.x)

            result = {
                "success": optimization_result.success,
                "optimization_target": optimization_target,
                "original_parameters": self._parameters_to_array(self.current_parameters),
                "optimized_parameters": optimization_result.x,
                "parameter_dict": self._array_to_parameters(optimization_result.x),
                "improvement_score": validation_result["improvement_score"],
                "baseline_performance": self.performance_baseline,
                "optimized_performance": validation_result["projected_performance"],
                "optimization_time": optimization_time,
                "optimization_iterations": optimization_result.nit,
                "convergence_info": {
                    "converged": optimization_result.success,
                    "final_objective_value": optimization_result.fun,
                    "message": optimization_result.message,
                },
            }

            self.optimization_history.append(result)
            return result

        except Exception as e:
            logger.exception(f"Parameter optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "optimization_target": optimization_target,
                "optimization_time": time.time() - start_time if "start_time" in locals() else 0,
            }

    def _calculate_baseline_performance(self, performance_data: dict[str, Any]) -> dict[str, float]:
        """Calculate baseline performance metrics."""
        metrics = performance_data.get("metrics", {})

        return {
            "accuracy": metrics.get("generation_success_rate", 0.8),
            "speed": 1.0 / max(metrics.get("avg_generation_time", 1.0), 0.1),  # Inverse of time
            "error_rate": metrics.get("error_rate", 0.2),
            "confidence": metrics.get("avg_confidence", 0.7),
        }


    def _create_objective_function(self, performance_data: dict[str, Any], target: str):
        """Create optimization objective function."""
        def objective(params):
            # Convert parameter array to dict
            param_dict = self._array_to_parameters(params)

            # Simulate performance with these parameters
            projected_performance = self._project_performance(param_dict, performance_data)

            # Calculate objective based on target
            if target == "accuracy":
                # Maximize accuracy, minimize error rate
                score = projected_performance["accuracy"] - projected_performance["error_rate"] * 0.5
            elif target == "speed":
                # Maximize speed (inverse time), maintain reasonable accuracy
                score = projected_performance["speed"] * 0.7 + projected_performance["accuracy"] * 0.3
            elif target == "balance":
                # Balanced optimization
                score = (projected_performance["accuracy"] * 0.4 +
                        projected_performance["speed"] * 0.3 +
                        projected_performance["confidence"] * 0.3 -
                        projected_performance["error_rate"] * 0.2)
            else:
                score = projected_performance["accuracy"]

            # Return negative for minimization
            return -score

        return objective

    def _get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization."""
        return [
            (0.0, 1.0),      # temperature
            (100, 2000),     # max_tokens
            (0.1, 1.0),      # top_p
            (0.0, 2.0),      # frequency_penalty
            (0.0, 2.0),       # presence_penalty
        ]

    def _run_optimization(self, objective_function, bounds) -> Any:
        """Run the optimization process."""
        # Initial guess (current parameters)
        x0 = self._parameters_to_array(self.current_parameters)

        # Use scipy.optimize.minimize
        return minimize(
            objective_function,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 100,
                "ftol": 1e-6,
                "gtol": 1e-6,
            },
        )


    def _validate_parameters(self, optimized_params: np.ndarray, performance_data: dict[str, Any]) -> dict[str, Any]:
        """Validate optimized parameters."""
        param_dict = self._array_to_parameters(optimized_params)

        # Project performance with optimized parameters
        projected_performance = self._project_performance(param_dict, performance_data)

        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(
            self.performance_baseline, projected_performance,
        )

        return {
            "projected_performance": projected_performance,
            "improvement_score": improvement_score,
            "parameter_validity": self._check_parameter_validity(param_dict),
        }

    def _project_performance(self, parameters: dict[str, Any], performance_data: dict[str, Any]) -> dict[str, float]:
        """Project performance with given parameters."""
        # Simplified performance projection model
        baseline = self.performance_baseline

        # Temperature effect on accuracy and speed
        temp_effect = parameters["temperature"]
        accuracy_modifier = 1.0 - (temp_effect * 0.1)  # Lower temp = higher accuracy
        speed_modifier = 1.0 + (temp_effect * 0.2)      # Higher temp = faster generation

        # Max tokens effect
        token_ratio = parameters["max_tokens"] / 1000.0
        token_speed_modifier = 1.0 / max(token_ratio, 0.1)

        # Penalty effects
        penalty_accuracy_modifier = 1.0 + (parameters["frequency_penalty"] * 0.05)

        return {
            "accuracy": min(1.0, baseline["accuracy"] * accuracy_modifier * penalty_accuracy_modifier),
            "speed": baseline["speed"] * speed_modifier * token_speed_modifier,
            "error_rate": max(0.0, baseline["error_rate"] * (2.0 - accuracy_modifier)),
            "confidence": min(1.0, baseline["confidence"] * (1.0 + temp_effect * 0.1)),
        }


    def _calculate_improvement_score(self, baseline: dict[str, float], projected: dict[str, float]) -> float:
        """Calculate overall improvement score."""
        improvements = []

        # Accuracy improvement
        acc_improvement = (projected["accuracy"] - baseline["accuracy"]) / baseline["accuracy"]
        improvements.append(acc_improvement * 0.4)

        # Speed improvement
        speed_improvement = (projected["speed"] - baseline["speed"]) / baseline["speed"]
        improvements.append(speed_improvement * 0.3)

        # Error rate improvement (negative is good)
        error_improvement = (baseline["error_rate"] - projected["error_rate"]) / baseline["error_rate"]
        improvements.append(error_improvement * 0.2)

        # Confidence improvement
        conf_improvement = (projected["confidence"] - baseline["confidence"]) / baseline["confidence"]
        improvements.append(conf_improvement * 0.1)

        return sum(improvements)

    def _check_parameter_validity(self, parameters: dict[str, Any]) -> dict[str, bool]:
        """Check if parameters are valid."""
        validity = {
            "temperature_valid": 0.0 <= parameters["temperature"] <= 1.0,
            "max_tokens_valid": 100 <= parameters["max_tokens"] <= 2000,
            "top_p_valid": 0.1 <= parameters["top_p"] <= 1.0,
            "frequency_penalty_valid": 0.0 <= parameters["frequency_penalty"] <= 2.0,
            "presence_penalty_valid": 0.0 <= parameters["presence_penalty"] <= 2.0,
        }

        validity["all_valid"] = all(validity.values())
        return validity

    def _parameters_to_array(self, param_dict: dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        return np.array([
            param_dict["temperature"],
            param_dict["max_tokens"],
            param_dict["top_p"],
            param_dict["frequency_penalty"],
            param_dict["presence_penalty"],
        ])

    def _array_to_parameters(self, param_array: np.ndarray) -> dict[str, Any]:
        """Convert parameter array to dictionary."""
        return {
            "temperature": float(param_array[0]),
            "max_tokens": int(param_array[1]),
            "top_p": float(param_array[2]),
            "frequency_penalty": float(param_array[3]),
            "presence_penalty": float(param_array[4]),
        }

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get history of optimization attempts."""
        return self.optimization_history

    def get_current_parameters(self) -> dict[str, Any]:
        """Get current optimized parameters."""
        return self.current_parameters.copy()


class AutonomousLearningEngine:
    """Autonomous learning engine that coordinates pattern analysis and parameter optimization."""

    def __init__(self, persistence_path: str = "neural_adaptation_data"):
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(exist_ok=True)

        self.pattern_analyzer = QueryPatternAnalyzer()
        self.parameter_optimizer = NeuralParameterOptimizer()
        self.learning_patterns = {}
        self.adaptation_history = []

        # Load persisted data
        self._load_persistent_data()

    def continuous_learning_cycle(
        self,
        query_history: list[dict[str, Any]],
        performance_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a complete autonomous learning cycle.

        Args:
            query_history: Recent query execution history
            performance_metrics: Current performance metrics

        Returns:
            Learning cycle results with adaptations made
        """
        cycle_start_time = time.time()

        try:
            # Step 1: Analyze patterns
            logger.info("Starting pattern analysis phase...")
            pattern_analysis = self.pattern_analyzer.analyze_query_patterns(query_history)

            if "error" in pattern_analysis:
                return {
                    "success": False,
                    "error": f"Pattern analysis failed: {pattern_analysis['error']}",
                    "cycle_time": time.time() - cycle_start_time,
                }

            # Step 2: Identify learning opportunities
            learning_opportunities = pattern_analysis.get("learning_opportunities", [])
            logger.info(f"Identified {len(learning_opportunities)} learning opportunities")

            # Step 3: Optimize parameters
            logger.info("Starting parameter optimization phase...")
            optimization_target = self._determine_optimization_target(pattern_analysis, performance_metrics)

            optimization_result = self.parameter_optimizer.optimize_parameters(
                performance_metrics, optimization_target,
            )

            # Step 4: Learn new patterns
            logger.info("Learning new patterns...")
            new_patterns = self._learn_patterns_from_analysis(pattern_analysis)

            # Step 5: Evaluate adaptations
            adaptation_results = self._evaluate_adaptations(
                pattern_analysis, optimization_result, new_patterns,
            )

            # Step 6: Update persistent learning state
            self._update_learning_state(pattern_analysis, optimization_result, new_patterns)

            cycle_time = time.time() - cycle_start_time

            learning_cycle_result = {
                "success": True,
                "cycle_time": cycle_time,
                "pattern_analysis": pattern_analysis,
                "optimization_result": optimization_result,
                "new_patterns_learned": len(new_patterns),
                "learning_opportunities": learning_opportunities,
                "adaptation_results": adaptation_results,
                "optimization_target": optimization_target,
                "performance_improvement": adaptation_results.get("overall_improvement", 0.0),
                "adaptations_applied": adaptation_results.get("adaptations_applied", []),
                "cycle_timestamp": time.time(),
            }

            self.adaptation_history.append(learning_cycle_result)

            # Persist results
            self._persist_learning_data()

            logger.info(f"Learning cycle completed in {cycle_time:.2f}s with {adaptation_results.get('overall_improvement', 0.0):.1%} improvement")

            return learning_cycle_result

        except Exception as e:
            logger.exception(f"Learning cycle failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "cycle_time": time.time() - cycle_start_time,
                "cycle_timestamp": time.time(),
            }

    def _determine_optimization_target(
        self,
        pattern_analysis: dict[str, Any],
        performance_metrics: dict[str, Any],
    ) -> str:
        """Determine the optimization target based on current state."""
        # Check current performance
        success_rate = performance_metrics.get("generation_success_rate", 0.8)
        avg_time = performance_metrics.get("avg_generation_time", 1.0)

        # Analyze patterns to determine focus
        performance_patterns = pattern_analysis.get("performance_patterns", {})
        optimization_opportunities = performance_patterns.get("optimization_opportunities", [])

        # Decision logic
        if success_rate < 0.75:  # Low accuracy
            return "accuracy"
        if avg_time > 3.0 or any("slow queries" in opp.lower() for opp in optimization_opportunities):     # Slow performance
            return "speed"
        return "balance"     # Balanced optimization

    def _learn_patterns_from_analysis(self, pattern_analysis: dict[str, Any]) -> list[LearningPattern]:
        """Learn new patterns from analysis results."""
        new_patterns = []

        # Learn from linguistic patterns
        linguistic_patterns = pattern_analysis.get("linguistic_patterns", {})
        top_phrases = linguistic_patterns.get("top_phrases", [])

        for phrase, frequency in top_phrases[:5]:  # Top 5 phrases
            pattern_id = f"phrase_{phrase.replace(' ', '_')}"
            if pattern_id not in self.learning_patterns:
                new_pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type="linguistic_phrase",
                    features={"phrase": phrase, "frequency": frequency},
                    success_rate=0.8,  # Initial estimate
                    usage_count=frequency,
                    confidence=min(frequency / 10.0, 1.0),  # Confidence based on frequency
                )
                new_patterns.append(new_pattern)
                self.learning_patterns[pattern_id] = new_pattern

        # Learn from structural patterns
        structural_patterns = pattern_analysis.get("structural_patterns", {})
        clause_combinations = structural_patterns.get("popular_clause_combinations", [])

        for combo, frequency in clause_combinations[:3]:  # Top 3 combinations
            pattern_id = f"structure_{combo}"
            if pattern_id not in self.learning_patterns:
                new_pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type="sql_structure",
                    features={"clause_combination": combo, "frequency": frequency},
                    success_rate=0.85,  # SQL structures tend to be reliable
                    usage_count=frequency,
                    confidence=min(frequency / 5.0, 1.0),
                )
                new_patterns.append(new_pattern)
                self.learning_patterns[pattern_id] = new_pattern

        # Learn from emerging patterns
        emerging_patterns = pattern_analysis.get("emerging_patterns", [])
        for emerging in emerging_patterns[:2]:  # Top 2 emerging patterns
            pattern_id = f"emerging_{emerging['pattern']}"
            if pattern_id not in self.learning_patterns:
                new_pattern = LearningPattern(
                    pattern_id=pattern_id,
                    pattern_type="emerging_trend",
                    features=emerging,
                    success_rate=0.7,  # Lower initial confidence for emerging patterns
                    usage_count=1,
                    confidence=min(emerging.get("growth_factor", 1.0) / 5.0, 1.0),
                )
                new_patterns.append(new_pattern)
                self.learning_patterns[pattern_id] = new_pattern

        return new_patterns

    def _evaluate_adaptations(
        self,
        pattern_analysis: dict[str, Any],
        optimization_result: dict[str, Any],
        new_patterns: list[LearningPattern],
    ) -> dict[str, Any]:
        """Evaluate the effectiveness of adaptations."""
        adaptations_applied = []
        overall_improvement = 0.0

        # Evaluate parameter optimization
        if optimization_result.get("success", False):
            improvement = optimization_result.get("improvement_score", 0.0)
            if improvement > 0.05:  # 5% improvement threshold
                adaptations_applied.append({
                    "type": "parameter_optimization",
                    "improvement": improvement,
                    "details": optimization_result.get("optimization_target", "unknown"),
                })
                overall_improvement += improvement * 0.6  # 60% weight for parameter optimization

        # Evaluate pattern learning
        pattern_improvement = 0.0
        if new_patterns:
            # Estimate improvement from new patterns
            avg_confidence = sum(p.confidence for p in new_patterns) / len(new_patterns)
            pattern_improvement = avg_confidence * 0.1  # Conservative estimate

            adaptations_applied.append({
                "type": "pattern_learning",
                "improvement": pattern_improvement,
                "details": f"Learned {len(new_patterns)} new patterns",
            })
            overall_improvement += pattern_improvement * 0.4  # 40% weight for pattern learning

        # Evaluate pattern significance
        pattern_significance = pattern_analysis.get("pattern_significance", {})
        significance_score = sum(pattern_significance.values()) / len(pattern_significance) if pattern_significance else 0.0

        if significance_score > 0.7:  # High significance
            adaptations_applied.append({
                "type": "high_significance_patterns",
                "improvement": significance_score * 0.05,
                "details": f"High pattern significance: {significance_score:.2f}",
            })
            overall_improvement += significance_score * 0.05

        return {
            "overall_improvement": overall_improvement,
            "adaptations_applied": adaptations_applied,
            "adaptation_count": len(adaptations_applied),
            "pattern_significance_score": significance_score,
            "learning_velocity": len(new_patterns) / max(len(self.learning_patterns), 1),
        }

    def _update_learning_state(
        self,
        pattern_analysis: dict[str, Any],
        optimization_result: dict[str, Any],
        new_patterns: list[LearningPattern],
    ) -> None:
        """Update persistent learning state."""
        # Update pattern usage counts and success rates
        for pattern in self.learning_patterns.values():
            pattern.last_updated = time.time()

        # Record learning insights for future cycles
        learning_insight = {
            "timestamp": time.time(),
            "patterns_learned": len(new_patterns),
            "optimization_improvement": optimization_result.get("improvement_score", 0.0),
            "dominant_domain": pattern_analysis.get("context_patterns", {}).get("dominant_domain", "general"),
            "complexity_trend": pattern_analysis.get("linguistic_patterns", {}).get("avg_complexity", 0.0),
        }

        # Store in learning history (keep last 100 insights)
        if not hasattr(self, "learning_insights"):
            self.learning_insights = []

        self.learning_insights.append(learning_insight)
        if len(self.learning_insights) > 100:
            self.learning_insights = self.learning_insights[-100:]

    def _load_persistent_data(self) -> None:
        """Load persistent learning data."""
        try:
            # Load learning patterns (using JSON instead of pickle for security)
            patterns_file = self.persistence_path / "learning_patterns.json"
            if patterns_file.exists():
                with open(patterns_file) as f:
                    patterns_data = json.load(f)
                    # Reconstruct LearningPattern objects from JSON data
                    self.learning_patterns = {}
                    for pattern_id, pattern_dict in patterns_data.items():
                        pattern = LearningPattern(
                            pattern_id=pattern_dict["pattern_id"],
                            pattern_type=pattern_dict["pattern_type"],
                            features=pattern_dict["features"],
                            success_rate=pattern_dict["success_rate"],
                            usage_count=pattern_dict["usage_count"],
                            confidence=pattern_dict["confidence"],
                            last_updated=pattern_dict.get("last_updated", time.time()),
                            metadata=pattern_dict.get("metadata", {}),
                        )
                        self.learning_patterns[pattern_id] = pattern
                logger.info(f"Loaded {len(self.learning_patterns)} learning patterns")

            # Load parameter optimization history
            param_file = self.persistence_path / "parameter_history.json"
            if param_file.exists():
                with open(param_file) as f:
                    param_data = json.load(f)
                    self.parameter_optimizer.optimization_history = param_data.get("history", [])
                    self.parameter_optimizer.current_parameters = param_data.get("current_params",
                                                                                self.parameter_optimizer.current_parameters)
                logger.info(f"Loaded parameter optimization history with {len(self.parameter_optimizer.optimization_history)} entries")

            # Load learning insights
            insights_file = self.persistence_path / "learning_insights.json"
            if insights_file.exists():
                with open(insights_file) as f:
                    self.learning_insights = json.load(f)
                logger.info(f"Loaded {len(self.learning_insights)} learning insights")

        except Exception as e:
            logger.warning(f"Failed to load persistent data: {e}")

    def _persist_learning_data(self) -> None:
        """Persist learning data to disk."""
        try:
            # Save learning patterns (using JSON instead of pickle for security)
            patterns_file = self.persistence_path / "learning_patterns.json"
            patterns_data = {}
            for pattern_id, pattern in self.learning_patterns.items():
                patterns_data[pattern_id] = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "features": pattern.features,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "confidence": pattern.confidence,
                    "last_updated": pattern.last_updated,
                    "metadata": pattern.metadata,
                }
            with open(patterns_file, "w") as f:
                json.dump(patterns_data, f, indent=2, default=str)

            # Save parameter optimization data
            param_file = self.persistence_path / "parameter_history.json"
            param_data = {
                "history": self.parameter_optimizer.optimization_history,
                "current_params": self.parameter_optimizer.current_parameters,
            }
            with open(param_file, "w") as f:
                json.dump(param_data, f, indent=2, default=str)

            # Save learning insights
            if hasattr(self, "learning_insights"):
                insights_file = self.persistence_path / "learning_insights.json"
                with open(insights_file, "w") as f:
                    json.dump(self.learning_insights, f, indent=2, default=str)

            logger.debug("Successfully persisted learning data")

        except Exception as e:
            logger.exception(f"Failed to persist learning data: {e}")

    def get_learning_analytics(self) -> dict[str, Any]:
        """Get comprehensive learning analytics."""
        if not self.adaptation_history:
            return {"message": "No learning history available"}

        total_cycles = len(self.adaptation_history)
        successful_cycles = sum(1 for cycle in self.adaptation_history if cycle.get("success", False))

        # Performance improvement trends
        improvements = [cycle.get("performance_improvement", 0.0) for cycle in self.adaptation_history if cycle.get("success", False)]

        # Pattern learning statistics
        total_patterns_learned = sum(cycle.get("new_patterns_learned", 0) for cycle in self.adaptation_history)

        # Optimization target distribution
        target_distribution = {}
        for cycle in self.adaptation_history:
            target = cycle.get("optimization_target", "unknown")
            target_distribution[target] = target_distribution.get(target, 0) + 1

        return {
            "total_learning_cycles": total_cycles,
            "successful_cycles": successful_cycles,
            "success_rate": successful_cycles / total_cycles if total_cycles > 0 else 0.0,
            "performance_metrics": {
                "avg_improvement_per_cycle": sum(improvements) / len(improvements) if improvements else 0.0,
                "cumulative_improvement": sum(improvements),
                "best_cycle_improvement": max(improvements) if improvements else 0.0,
                "improvement_trend": improvements[-10:] if len(improvements) >= 10 else improvements,
            },
            "learning_statistics": {
                "total_patterns_learned": total_patterns_learned,
                "current_pattern_count": len(self.learning_patterns),
                "avg_patterns_per_cycle": total_patterns_learned / max(successful_cycles, 1),
                "pattern_types": list({p.pattern_type for p in self.learning_patterns.values()}),
            },
            "optimization_focus": target_distribution,
            "recent_cycles": self.adaptation_history[-5:] if len(self.adaptation_history) >= 5 else self.adaptation_history,
        }


# Global autonomous learning engine
global_learning_engine = AutonomousLearningEngine()

# Example usage functions
def run_autonomous_learning_cycle(query_history: list[dict[str, Any]], performance_metrics: dict[str, Any]) -> dict[str, Any]:
    """Run an autonomous learning cycle with the global learning engine."""
    return global_learning_engine.continuous_learning_cycle(query_history, performance_metrics)

def get_neural_adaptation_analytics() -> dict[str, Any]:
    """Get analytics from the global learning engine."""
    return global_learning_engine.get_learning_analytics()
