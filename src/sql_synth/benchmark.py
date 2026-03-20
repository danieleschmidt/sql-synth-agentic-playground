"""Benchmark evaluation framework against Spider/WikiSQL-style examples."""
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sql_synth.nl2sql import translate


@dataclass
class BenchmarkExample:
    question: str
    gold_sql: str
    db_name: str = "default"
    context: str = ""


@dataclass
class EvalResult:
    example: BenchmarkExample
    predicted_sql: str
    confidence: float
    exact_match: bool
    structural_match: bool  # matches ignoring whitespace/case
    method: str


@dataclass
class BenchmarkReport:
    total: int
    exact_matches: int
    structural_matches: int
    exact_match_rate: float
    structural_match_rate: float
    avg_confidence: float
    results: list = field(default_factory=list)


# Built-in subset of Spider-style examples for quick evaluation
SPIDER_SUBSET = [
    BenchmarkExample(
        question="How many students are there",
        gold_sql="SELECT COUNT(*) FROM student",
        db_name="school",
    ),
    BenchmarkExample(
        question="List all employees",
        gold_sql="SELECT * FROM employee",
        db_name="company",
    ),
    BenchmarkExample(
        question="Show all products",
        gold_sql="SELECT * FROM product",
        db_name="store",
    ),
    BenchmarkExample(
        question="Find all users where age > 18",
        gold_sql="SELECT * FROM user WHERE age > 18",
        db_name="users",
    ),
    BenchmarkExample(
        question="Show the top 10 orders",
        gold_sql="SELECT * FROM order LIMIT 10",
        db_name="ecommerce",
    ),
    BenchmarkExample(
        question="Get all customers where country is USA",
        gold_sql="SELECT * FROM customer WHERE country = 'USA'",
        db_name="sales",
    ),
    BenchmarkExample(
        question="Show employees ordered by salary desc",
        gold_sql="SELECT * FROM employee ORDER BY salary DESC",
        db_name="company",
    ),
    BenchmarkExample(
        question="What is the average salary of employees",
        gold_sql="SELECT AVG(salary) FROM employee",
        db_name="company",
    ),
    BenchmarkExample(
        question="What is the maximum price of products",
        gold_sql="SELECT MAX(price) FROM product",
        db_name="store",
    ),
    BenchmarkExample(
        question="Count students per grade",
        gold_sql="SELECT grade, COUNT(*) FROM student GROUP BY grade",
        db_name="school",
    ),
]


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, collapse whitespace."""
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    # Remove quotes around values for structural comparison
    sql = sql.replace("'", "").replace('"', "")
    return sql


def evaluate_example(example: BenchmarkExample) -> EvalResult:
    """Evaluate a single benchmark example."""
    result = translate(example.question)
    predicted = result.sql

    exact_match = predicted.strip() == example.gold_sql.strip()
    structural_match = normalize_sql(predicted) == normalize_sql(example.gold_sql)

    return EvalResult(
        example=example,
        predicted_sql=predicted,
        confidence=result.confidence,
        exact_match=exact_match,
        structural_match=structural_match,
        method=result.method,
    )


def run_benchmark(examples: Optional[list] = None) -> BenchmarkReport:
    """Run benchmark evaluation and return report."""
    if examples is None:
        examples = SPIDER_SUBSET

    results = [evaluate_example(ex) for ex in examples]

    exact = sum(1 for r in results if r.exact_match)
    structural = sum(1 for r in results if r.structural_match)
    avg_conf = sum(r.confidence for r in results) / len(results) if results else 0

    return BenchmarkReport(
        total=len(results),
        exact_matches=exact,
        structural_matches=structural,
        exact_match_rate=exact / len(results) if results else 0,
        structural_match_rate=structural / len(results) if results else 0,
        avg_confidence=avg_conf,
        results=results,
    )


def load_custom_benchmark(path: str) -> list:
    """Load a custom benchmark from a JSON file.
    
    Expected format:
    [{"question": "...", "gold_sql": "...", "db_name": "..."}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    return [
        BenchmarkExample(
            question=item["question"],
            gold_sql=item["gold_sql"],
            db_name=item.get("db_name", "default"),
        )
        for item in data
    ]
