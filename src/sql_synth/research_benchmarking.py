"""Research framework for benchmarking SQL generation approaches.

This module implements a comprehensive research and benchmarking system
for comparing different SQL generation strategies, models, and approaches.
It includes statistical analysis, reproducible experiments, and performance
evaluation against standard datasets like Spider and WikiSQL.
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    EXECUTION_ACCURACY = "execution_accuracy"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    SYNTACTIC_CORRECTNESS = "syntactic_correctness"
    PERFORMANCE = "performance"
    COMPLEXITY = "complexity"


@dataclass
class ExperimentConfig:
    """Configuration for a research experiment."""
    experiment_id: str
    name: str
    description: str
    dataset_name: str
    models_to_compare: List[str]
    evaluation_metrics: List[MetricType]
    sample_size: Optional[int] = None
    random_seed: int = 42
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result of SQL generation and execution."""
    query_id: str
    natural_query: str
    generated_sql: Optional[str]
    reference_sql: Optional[str]
    execution_result: Optional[Any]
    reference_result: Optional[Any]
    generation_time: float
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Performance metrics for a model in an experiment."""
    model_name: str
    total_queries: int
    successful_queries: int
    accuracy_score: float
    execution_accuracy_score: float
    avg_generation_time: float
    avg_execution_time: float
    complexity_scores: List[float]
    error_types: Dict[str, int]
    confidence_intervals: Dict[str, Tuple[float, float]]


class DatasetLoader:
    """Loader for standard SQL generation datasets."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.supported_datasets = ["spider", "wikisql", "synthetic_benchmark"]

    def load_dataset(self, dataset_name: str, split: str = "test",
                    sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load a dataset split."""
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        dataset_path = self.data_dir / dataset_name / f"{split}.json"

        if not dataset_path.exists():
            # Create synthetic data if dataset doesn't exist
            logger.warning("Dataset %s not found, creating synthetic data", dataset_name)
            return self._create_synthetic_dataset(sample_size or 100)

        with open(dataset_path) as f:
            data = json.load(f)

        if sample_size:
            data = data[:sample_size]

        logger.info("Loaded %d examples from %s/%s", len(data), dataset_name, split)
        return data

    def _create_synthetic_dataset(self, size: int) -> List[Dict[str, Any]]:
        """Create synthetic dataset for testing."""
        synthetic_data = []

        templates = [
            {
                "natural_query": "Show all users from the marketing department",
                "sql": "SELECT * FROM users WHERE department = 'marketing';",
                "database_id": "company_db",
            },
            {
                "natural_query": "Find the average salary by department",
                "sql": "SELECT department, AVG(salary) FROM employees GROUP BY department;",
                "database_id": "hr_db",
            },
            {
                "natural_query": "List top 5 customers by total orders",
                "sql": "SELECT customer_id, COUNT(*) as order_count FROM orders GROUP BY customer_id ORDER BY order_count DESC LIMIT 5;",
                "database_id": "ecommerce_db",
            },
            {
                "natural_query": "Get products with price greater than average",
                "sql": "SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products);",
                "database_id": "catalog_db",
            },
        ]

        for i in range(size):
            template = templates[i % len(templates)]
            synthetic_data.append({
                "query_id": f"synthetic_{i}",
                "natural_query": template["natural_query"],
                "sql": template["sql"],
                "database_id": template["database_id"],
                "complexity": "medium",
            })

        return synthetic_data


class SQLAccuracyEvaluator:
    """Evaluator for SQL generation accuracy."""

    def __init__(self):
        self.string_similarity_threshold = 0.8

    def evaluate_exact_match(self, generated_sql: str, reference_sql: str) -> bool:
        """Evaluate exact string match (after normalization)."""
        gen_normalized = self._normalize_sql(generated_sql)
        ref_normalized = self._normalize_sql(reference_sql)
        return gen_normalized == ref_normalized

    def evaluate_execution_accuracy(self, generated_result: Any,
                                  reference_result: Any) -> bool:
        """Evaluate if execution results match."""
        if generated_result is None or reference_result is None:
            return False

        # Simple result comparison (would need more sophisticated logic for real datasets)
        try:
            if isinstance(generated_result, (list, tuple)) and isinstance(reference_result, (list, tuple)):
                return len(generated_result) == len(reference_result) and \
                       all(g == r for g, r in zip(generated_result, reference_result))
            return generated_result == reference_result
        except Exception:
            return False

    def evaluate_semantic_similarity(self, generated_sql: str, reference_sql: str) -> float:
        """Evaluate semantic similarity using AST comparison."""
        try:
            # Simplified semantic similarity - in real implementation,
            # you'd use SQL parsing and AST comparison
            gen_tokens = set(self._tokenize_sql(generated_sql))
            ref_tokens = set(self._tokenize_sql(reference_sql))

            if not ref_tokens:
                return 0.0

            intersection = len(gen_tokens & ref_tokens)
            union = len(gen_tokens | ref_tokens)

            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.warning("Semantic similarity evaluation failed: %s", e)
            return 0.0

    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        # Remove extra whitespace, convert to lowercase, etc.
        normalized = " ".join(sql.strip().lower().split())
        # Remove trailing semicolon
        if normalized.endswith(";"):
            normalized = normalized[:-1]
        return normalized

    def _tokenize_sql(self, sql: str) -> List[str]:
        """Simple SQL tokenization."""
        import re
        # Very basic tokenization - would need proper SQL parser for production
        tokens = re.findall(r"\b\w+\b", sql.upper())
        return tokens


class StatisticalAnalyzer:
    """Statistical analysis of experiment results."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compare_models(self, model_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Compare multiple models using statistical tests."""
        results = {
            "model_comparisons": {},
            "anova_test": None,
            "best_model": None,
            "statistical_significance": {},
        }

        # ANOVA test for multiple model comparison
        if len(model_results) > 2:
            try:
                model_values = list(model_results.values())
                f_stat, p_value = stats.f_oneway(*model_values)
                results["anova_test"] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < self.alpha,
                }
            except Exception as e:
                logger.warning("ANOVA test failed: %s", e)

        # Pairwise t-tests
        model_names = list(model_results.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                try:
                    t_stat, p_value = stats.ttest_ind(
                        model_results[model1],
                        model_results[model2],
                    )

                    comparison_key = f"{model1}_vs_{model2}"
                    results["model_comparisons"][comparison_key] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < self.alpha,
                        "effect_size": self._calculate_effect_size(
                            model_results[model1],
                            model_results[model2],
                        ),
                    }
                except Exception as e:
                    logger.warning("T-test failed for %s vs %s: %s", model1, model2, e)

        # Determine best model
        model_means = {name: np.mean(values) for name, values in model_results.items()}
        results["best_model"] = max(model_means, key=model_means.get)

        return results

    def calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values."""
        if not values:
            return (0.0, 0.0)

        n = len(values)
        mean = statistics.mean(values)

        if n == 1:
            return (mean, mean)

        std_error = statistics.stdev(values) / (n ** 0.5)
        t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)

        margin_of_error = t_critical * std_error
        return (mean - margin_of_error, mean + margin_of_error)

    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group1 or not group2:
            return 0.0

        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)

        if len(group1) == 1 and len(group2) == 1:
            return abs(mean1 - mean2)

        std1 = statistics.stdev(group1) if len(group1) > 1 else 0
        std2 = statistics.stdev(group2) if len(group2) > 1 else 0

        pooled_std = ((std1**2 + std2**2) / 2) ** 0.5

        if pooled_std == 0:
            return 0.0

        return abs(mean1 - mean2) / pooled_std


class ExperimentRunner:
    """Main experiment runner for SQL generation benchmarking."""

    def __init__(self, data_dir: Path = Path("data"), results_dir: Path = Path("results")):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True)

        self.dataset_loader = DatasetLoader(data_dir)
        self.accuracy_evaluator = SQLAccuracyEvaluator()
        self.statistical_analyzer = StatisticalAnalyzer()

        # Mock models for demonstration
        self.available_models = {
            "gpt-3.5-turbo": self._mock_gpt35_model,
            "gpt-4": self._mock_gpt4_model,
            "claude-2": self._mock_claude_model,
            "code-llama": self._mock_codellama_model,
        }

    async def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run a complete benchmarking experiment."""
        logger.info("Starting experiment: %s", config.name)
        start_time = time.time()

        # Load dataset
        dataset = self.dataset_loader.load_dataset(
            config.dataset_name,
            sample_size=config.sample_size,
        )

        # Run each model
        model_results = {}
        for model_name in config.models_to_compare:
            if model_name not in self.available_models:
                logger.warning("Model %s not available, skipping", model_name)
                continue

            logger.info("Evaluating model: %s", model_name)
            model_performance = await self._evaluate_model(
                model_name, dataset, config,
            )
            model_results[model_name] = model_performance

        # Statistical analysis
        accuracy_scores = {
            model: perf.accuracy_score
            for model, perf in model_results.items()
        }

        statistical_analysis = self.statistical_analyzer.compare_models({
            model: [perf.accuracy_score] * perf.total_queries
            for model, perf in model_results.items()
        })

        # Compile final results
        experiment_results = {
            "experiment_config": config.__dict__,
            "dataset_size": len(dataset),
            "model_performances": {
                model: perf.__dict__ for model, perf in model_results.items()
            },
            "statistical_analysis": statistical_analysis,
            "experiment_duration": time.time() - start_time,
            "timestamp": time.time(),
        }

        # Save results
        results_file = self.results_dir / f"{config.experiment_id}_results.json"
        with open(results_file, "w") as f:
            json.dump(experiment_results, f, indent=2, default=str)

        logger.info("Experiment completed in %.2f seconds", experiment_results["experiment_duration"])
        return experiment_results

    async def _evaluate_model(self, model_name: str, dataset: List[Dict[str, Any]],
                            config: ExperimentConfig) -> ModelPerformance:
        """Evaluate a single model on the dataset."""
        model_func = self.available_models[model_name]

        results = []
        successful_queries = 0
        accuracy_scores = []
        execution_accuracy_scores = []
        generation_times = []
        execution_times = []
        complexity_scores = []
        error_types = {}

        for i, example in enumerate(dataset):
            query_id = example.get("query_id", f"query_{i}")
            natural_query = example["natural_query"]
            reference_sql = example.get("sql", "")

            try:
                # Generate SQL
                gen_start = time.time()
                generated_sql = await model_func(natural_query, example.get("database_id", ""))
                generation_time = time.time() - gen_start

                # Mock execution (in real implementation, execute against database)
                exec_start = time.time()
                execution_result = await self._mock_execute_sql(generated_sql)
                reference_result = await self._mock_execute_sql(reference_sql)
                execution_time = time.time() - exec_start

                # Evaluate accuracy
                exact_match = self.accuracy_evaluator.evaluate_exact_match(
                    generated_sql, reference_sql,
                )
                execution_match = self.accuracy_evaluator.evaluate_execution_accuracy(
                    execution_result, reference_result,
                )
                semantic_sim = self.accuracy_evaluator.evaluate_semantic_similarity(
                    generated_sql, reference_sql,
                )

                accuracy_scores.append(1.0 if exact_match else semantic_sim)
                execution_accuracy_scores.append(1.0 if execution_match else 0.0)
                generation_times.append(generation_time)
                execution_times.append(execution_time)

                # Calculate complexity score
                complexity_score = self._calculate_query_complexity(generated_sql)
                complexity_scores.append(complexity_score)

                if generated_sql and execution_result is not None:
                    successful_queries += 1

            except Exception as e:
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1

                # Add zero scores for failed queries
                accuracy_scores.append(0.0)
                execution_accuracy_scores.append(0.0)
                generation_times.append(config.timeout_seconds)
                execution_times.append(0.0)
                complexity_scores.append(0.0)

        # Calculate confidence intervals
        accuracy_ci = self.statistical_analyzer.calculate_confidence_interval(accuracy_scores)
        exec_acc_ci = self.statistical_analyzer.calculate_confidence_interval(execution_accuracy_scores)
        gen_time_ci = self.statistical_analyzer.calculate_confidence_interval(generation_times)

        return ModelPerformance(
            model_name=model_name,
            total_queries=len(dataset),
            successful_queries=successful_queries,
            accuracy_score=statistics.mean(accuracy_scores) if accuracy_scores else 0.0,
            execution_accuracy_score=statistics.mean(execution_accuracy_scores) if execution_accuracy_scores else 0.0,
            avg_generation_time=statistics.mean(generation_times) if generation_times else 0.0,
            avg_execution_time=statistics.mean(execution_times) if execution_times else 0.0,
            complexity_scores=complexity_scores,
            error_types=error_types,
            confidence_intervals={
                "accuracy": accuracy_ci,
                "execution_accuracy": exec_acc_ci,
                "generation_time": gen_time_ci,
            },
        )

    def _calculate_query_complexity(self, sql: str) -> float:
        """Calculate query complexity score."""
        if not sql:
            return 0.0

        complexity = 0.0
        sql_upper = sql.upper()

        # Basic complexity factors
        complexity += len(sql) / 1000  # Length factor
        complexity += sql_upper.count("JOIN") * 0.3
        complexity += sql_upper.count("SUBQUERY") * 0.4
        complexity += sql_upper.count("UNION") * 0.5
        complexity += sql_upper.count("GROUP BY") * 0.2
        complexity += sql_upper.count("ORDER BY") * 0.1

        return min(complexity, 5.0)  # Cap at 5.0

    async def _mock_execute_sql(self, sql: str) -> Optional[List[Dict[str, Any]]]:
        """Mock SQL execution for testing."""
        if not sql or "ERROR" in sql.upper():
            return None

        # Return mock results based on query type
        if "COUNT" in sql.upper():
            return [{"count": 42}]
        if "AVG" in sql.upper():
            return [{"avg": 125.5}]
        return [
            {"id": 1, "name": "Alice", "dept": "Engineering"},
            {"id": 2, "name": "Bob", "dept": "Marketing"},
        ]

    async def _mock_gpt35_model(self, natural_query: str, database_id: str) -> str:
        """Mock GPT-3.5 model."""
        await asyncio.sleep(0.1)  # Simulate API call
        if "users" in natural_query.lower():
            return "SELECT * FROM users WHERE condition = 'example' LIMIT 50;"
        if "average" in natural_query.lower():
            return "SELECT AVG(column) FROM table GROUP BY category;"
        return "SELECT * FROM table WHERE id > 0;"

    async def _mock_gpt4_model(self, natural_query: str, database_id: str) -> str:
        """Mock GPT-4 model (more accurate)."""
        await asyncio.sleep(0.15)  # Slightly slower
        if "users" in natural_query.lower():
            return "SELECT u.* FROM users u WHERE u.status = 'active' ORDER BY u.created_at DESC LIMIT 100;"
        if "average" in natural_query.lower():
            return "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department HAVING COUNT(*) > 1;"
        return "SELECT * FROM table t WHERE t.active = true ORDER BY t.id;"

    async def _mock_claude_model(self, natural_query: str, database_id: str) -> str:
        """Mock Claude model."""
        await asyncio.sleep(0.12)
        if "users" in natural_query.lower():
            return "SELECT * FROM users WHERE active = 1 ORDER BY last_login DESC;"
        if "average" in natural_query.lower():
            return "SELECT dept, AVG(salary) FROM emp_table GROUP BY dept;"
        return "SELECT col1, col2 FROM main_table WHERE status = 'valid';"

    async def _mock_codellama_model(self, natural_query: str, database_id: str) -> str:
        """Mock Code Llama model."""
        await asyncio.sleep(0.08)  # Faster local model
        if "users" in natural_query.lower():
            return "SELECT * FROM users;"  # Simpler queries
        if "average" in natural_query.lower():
            return "SELECT AVG(value) FROM data;"
        return "SELECT * FROM table1;"

    def generate_report(self, experiment_results: Dict[str, Any]) -> str:
        """Generate a human-readable experiment report."""
        report = []
        report.append("# SQL Generation Benchmarking Report")
        report.append(f"**Experiment:** {experiment_results['experiment_config']['name']}")
        report.append(f"**Dataset:** {experiment_results['experiment_config']['dataset_name']}")
        report.append(f"**Dataset Size:** {experiment_results['dataset_size']}")
        report.append(f"**Duration:** {experiment_results['experiment_duration']:.2f} seconds")
        report.append("")

        report.append("## Model Performance Summary")
        report.append("| Model | Accuracy | Exec. Accuracy | Avg Gen Time (s) | Success Rate |")
        report.append("|-------|----------|----------------|------------------|--------------|")

        for model_name, perf in experiment_results["model_performances"].items():
            success_rate = (perf["successful_queries"] / perf["total_queries"]) * 100
            report.append(
                f"| {model_name} | {perf['accuracy_score']:.3f} | "
                f"{perf['execution_accuracy_score']:.3f} | "
                f"{perf['avg_generation_time']:.3f} | {success_rate:.1f}% |",
            )

        report.append("")

        # Statistical analysis
        if experiment_results["statistical_analysis"]["anova_test"]:
            anova = experiment_results["statistical_analysis"]["anova_test"]
            report.append("## Statistical Analysis")
            report.append(f"**ANOVA Test:** F={anova['f_statistic']:.3f}, p={anova['p_value']:.6f}")
            report.append(f"**Significant Difference:** {'Yes' if anova['significant'] else 'No'}")
            report.append("")

        best_model = experiment_results["statistical_analysis"]["best_model"]
        report.append(f"**Best Performing Model:** {best_model}")
        report.append("")

        return "\n".join(report)


# Example usage and experiment configurations
def create_benchmark_experiments() -> List[ExperimentConfig]:
    """Create standard benchmarking experiments."""
    experiments = [
        ExperimentConfig(
            experiment_id="accuracy_comparison_2024",
            name="SQL Generation Accuracy Comparison",
            description="Compare accuracy of different models on standard datasets",
            dataset_name="synthetic_benchmark",
            models_to_compare=["gpt-3.5-turbo", "gpt-4", "claude-2"],
            evaluation_metrics=[MetricType.ACCURACY, MetricType.EXECUTION_ACCURACY],
            sample_size=100,
        ),
        ExperimentConfig(
            experiment_id="performance_benchmark_2024",
            name="Generation Performance Benchmark",
            description="Evaluate generation speed vs accuracy tradeoffs",
            dataset_name="synthetic_benchmark",
            models_to_compare=["gpt-3.5-turbo", "code-llama"],
            evaluation_metrics=[MetricType.PERFORMANCE, MetricType.ACCURACY],
            sample_size=200,
        ),
    ]

    return experiments


# Global experiment runner instance
experiment_runner = ExperimentRunner()
