"""Sample test data fixtures for SQL synthesis testing."""

import pytest
from typing import Dict, List, Any

# Sample database schemas for testing
SAMPLE_SCHEMAS = {
    "employees": {
        "tables": [
            {
                "name": "employees",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "first_name", "type": "VARCHAR(50)"},
                    {"name": "last_name", "type": "VARCHAR(50)"},
                    {"name": "email", "type": "VARCHAR(100)", "unique": True},
                    {"name": "hire_date", "type": "DATE"},
                    {"name": "salary", "type": "DECIMAL(10,2)"},
                    {"name": "department_id", "type": "INTEGER", "foreign_key": "departments.id"},
                ]
            },
            {
                "name": "departments",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "name", "type": "VARCHAR(50)"},
                    {"name": "manager_id", "type": "INTEGER", "foreign_key": "employees.id"},
                ]
            }
        ]
    },
    "sales": {
        "tables": [
            {
                "name": "products",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "name", "type": "VARCHAR(100)"},
                    {"name": "price", "type": "DECIMAL(8,2)"},
                    {"name": "category_id", "type": "INTEGER"},
                ]
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "customer_id", "type": "INTEGER"},
                    {"name": "order_date", "type": "DATE"},
                    {"name": "total_amount", "type": "DECIMAL(10,2)"},
                ]
            }
        ]
    }
}

# Sample natural language queries and expected SQL
SAMPLE_QUERIES = [
    {
        "natural_language": "Show me all employees with their department names",
        "expected_sql": """
            SELECT e.first_name, e.last_name, d.name as department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.id
        """,
        "schema": "employees",
        "difficulty": "basic"
    },
    {
        "natural_language": "Find employees who earn more than the average salary",
        "expected_sql": """
            SELECT first_name, last_name, salary
            FROM employees
            WHERE salary > (SELECT AVG(salary) FROM employees)
        """,
        "schema": "employees",
        "difficulty": "intermediate"
    },
    {
        "natural_language": "Get the top 5 highest paid employees in each department",
        "expected_sql": """
            WITH ranked_employees AS (
                SELECT 
                    first_name, 
                    last_name, 
                    salary,
                    department_id,
                    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rank
                FROM employees
            )
            SELECT first_name, last_name, salary, department_id
            FROM ranked_employees
            WHERE rank <= 5
        """,
        "schema": "employees",
        "difficulty": "advanced"
    }
]

# Performance test data
PERFORMANCE_TEST_QUERIES = [
    {
        "query": "SELECT COUNT(*) FROM employees",
        "expected_max_time_ms": 100,
        "dataset_size": "small"
    },
    {
        "query": "SELECT e.*, d.name FROM employees e JOIN departments d ON e.department_id = d.id",
        "expected_max_time_ms": 500,
        "dataset_size": "medium"
    }
]

@pytest.fixture
def sample_employee_schema():
    """Fixture providing sample employee database schema."""
    return SAMPLE_SCHEMAS["employees"]

@pytest.fixture
def sample_sales_schema():
    """Fixture providing sample sales database schema."""
    return SAMPLE_SCHEMAS["sales"]

@pytest.fixture
def basic_queries():
    """Fixture providing basic difficulty sample queries."""
    return [q for q in SAMPLE_QUERIES if q["difficulty"] == "basic"]

@pytest.fixture
def intermediate_queries():
    """Fixture providing intermediate difficulty sample queries."""
    return [q for q in SAMPLE_QUERIES if q["difficulty"] == "intermediate"]

@pytest.fixture
def advanced_queries():
    """Fixture providing advanced difficulty sample queries."""
    return [q for q in SAMPLE_QUERIES if q["difficulty"] == "advanced"]

@pytest.fixture
def all_sample_queries():
    """Fixture providing all sample queries."""
    return SAMPLE_QUERIES

@pytest.fixture
def performance_queries():
    """Fixture providing performance test queries."""
    return PERFORMANCE_TEST_QUERIES