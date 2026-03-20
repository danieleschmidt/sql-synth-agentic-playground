"""Tests for SQL Synth Agentic Playground."""
import pytest
import sqlite3


class TestNL2SQL:
    def test_select_all(self):
        from sql_synth.nl2sql import translate
        result = translate("List all employees")
        assert "SELECT" in result.sql.upper()
        assert "employee" in result.sql.lower()
        assert result.confidence > 0.5

    def test_count_query(self):
        from sql_synth.nl2sql import translate
        result = translate("How many customers are there")
        assert "COUNT" in result.sql.upper()
        assert "customer" in result.sql.lower()

    def test_select_where_equals(self):
        from sql_synth.nl2sql import translate
        result = translate("Get all customers where country is USA")
        assert "WHERE" in result.sql.upper()
        assert "USA" in result.sql
        assert result.confidence >= 0.5

    def test_select_where_greater_than(self):
        from sql_synth.nl2sql import translate
        result = translate("Find all employees where salary > 50000")
        assert "WHERE" in result.sql.upper()
        assert ">" in result.sql
        assert "50000" in result.sql

    def test_order_by(self):
        from sql_synth.nl2sql import translate
        result = translate("Show employees ordered by salary desc")
        assert "ORDER BY" in result.sql.upper()
        assert "DESC" in result.sql.upper()

    def test_limit(self):
        from sql_synth.nl2sql import translate
        result = translate("Show the top 10 orders")
        assert "LIMIT" in result.sql.upper()
        assert "10" in result.sql

    def test_aggregate_avg(self):
        from sql_synth.nl2sql import translate
        result = translate("What is the average salary of employees")
        assert "AVG" in result.sql.upper()

    def test_aggregate_max(self):
        from sql_synth.nl2sql import translate
        result = translate("What is the maximum price of products")
        assert "MAX" in result.sql.upper()

    def test_group_by(self):
        from sql_synth.nl2sql import translate
        result = translate("Count students per grade")
        assert "GROUP BY" in result.sql.upper()

    def test_empty_query_returns_result(self):
        from sql_synth.nl2sql import translate
        result = translate("something")
        # Should not raise, should return a result object
        assert result is not None
        assert result.natural_query == "something"

    def test_confidence_range(self):
        from sql_synth.nl2sql import translate
        result = translate("List all employees")
        assert 0.0 <= result.confidence <= 1.0

    def test_result_has_method(self):
        from sql_synth.nl2sql import translate
        result = translate("List all employees")
        assert result.method in ("exact", "pattern", "select_all", "count_simple",
                                  "count_where", "count_group_by", "select_where_eq",
                                  "select_where_gt", "select_where_lt", "select_ordered",
                                  "select_limit", "select_distinct", "aggregate_sum",
                                  "aggregate_avg", "aggregate_max", "aggregate_min",
                                  "delete_where", "fallback", "failed")


class TestBenchmark:
    def test_run_benchmark_returns_report(self):
        from sql_synth.benchmark import run_benchmark
        report = run_benchmark()
        assert report.total > 0
        assert 0.0 <= report.exact_match_rate <= 1.0
        assert 0.0 <= report.structural_match_rate <= 1.0

    def test_benchmark_has_results(self):
        from sql_synth.benchmark import run_benchmark
        report = run_benchmark()
        assert len(report.results) == report.total

    def test_normalize_sql(self):
        from sql_synth.benchmark import normalize_sql
        assert normalize_sql("SELECT  *  FROM  employees") == "select * from employees"
        assert normalize_sql("SELECT * FROM employees") == normalize_sql("select * from employees")

    def test_structural_match_ignores_case(self):
        from sql_synth.benchmark import normalize_sql
        s1 = normalize_sql("SELECT * FROM Employee")
        s2 = normalize_sql("select * from employee")
        assert s1 == s2

    def test_spider_subset_not_empty(self):
        from sql_synth.benchmark import SPIDER_SUBSET
        assert len(SPIDER_SUBSET) >= 5

    def test_custom_examples(self):
        from sql_synth.benchmark import run_benchmark, BenchmarkExample
        custom = [
            BenchmarkExample("List all employees", "SELECT * FROM employee"),
            BenchmarkExample("Show all products", "SELECT * FROM product"),
        ]
        report = run_benchmark(examples=custom)
        assert report.total == 2


class TestDatabase:
    def test_demo_db_created(self):
        from sql_synth.db import create_demo_db
        conn = create_demo_db()
        cursor = conn.execute("SELECT COUNT(*) FROM employee")
        count = cursor.fetchone()[0]
        assert count > 0

    def test_execute_sql_success(self):
        from sql_synth.db import execute_sql
        result = execute_sql("SELECT 1 + 1 AS result")
        assert result["success"] is True
        assert result["rows"][0]["result"] == 2

    def test_execute_sql_error(self):
        from sql_synth.db import execute_sql
        result = execute_sql("SELECT * FROM nonexistent_table_xyz")
        assert result["success"] is False
        assert "error" in result

    def test_demo_db_tables(self):
        from sql_synth.db import create_demo_db
        conn = create_demo_db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "employee" in table_names
        assert "product" in table_names
        assert "customer" in table_names
