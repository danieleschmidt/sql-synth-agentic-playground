import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def temp_db() -> Generator[str, None, None]:
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        yield f"sqlite:///{db_path}"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def memory_db() -> str:
    """Create an in-memory SQLite database for testing."""
    return "sqlite:///:memory:"


@pytest.fixture
def test_engine(memory_db: str):
    """Create a test database engine."""
    engine = create_engine(
        memory_db,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False}
    )
    return engine


@pytest.fixture
def sample_queries() -> list[dict]:
    """Sample test queries for NL-to-SQL testing."""
    return [
        {
            "natural_language": "Show me all users",
            "expected_sql": "SELECT * FROM users",
            "description": "Simple SELECT all query"
        },
        {
            "natural_language": "Count the number of orders",
            "expected_sql": "SELECT COUNT(*) FROM orders",
            "description": "COUNT aggregation query"
        },
        {
            "natural_language": "Find users with age greater than 25",
            "expected_sql": "SELECT * FROM users WHERE age > 25",
            "description": "WHERE clause with comparison"
        },
        {
            "natural_language": "Get the top 10 most expensive products",
            "expected_sql": "SELECT * FROM products ORDER BY price DESC LIMIT 10",
            "description": "ORDER BY with LIMIT"
        }
    ]


@pytest.fixture
def sample_schema() -> dict:
    """Sample database schema for testing."""
    return {
        "users": {
            "columns": ["id", "name", "email", "age", "created_at"],
            "types": ["INTEGER", "TEXT", "TEXT", "INTEGER", "DATETIME"]
        },
        "orders": {
            "columns": ["id", "user_id", "total", "status", "created_at"],
            "types": ["INTEGER", "INTEGER", "DECIMAL", "TEXT", "DATETIME"]
        },
        "products": {
            "columns": ["id", "name", "price", "category", "in_stock"],
            "types": ["INTEGER", "TEXT", "DECIMAL", "TEXT", "BOOLEAN"]
        }
    }


@pytest.fixture
def security_test_cases() -> list[dict]:
    """SQL injection test cases for security testing."""
    return [
        {
            "input": "'; DROP TABLE users; --",
            "should_block": True,
            "description": "Classic SQL injection attempt"
        },
        {
            "input": "' UNION SELECT * FROM sensitive_table --",
            "should_block": True,
            "description": "Union-based injection"
        },
        {
            "input": "1' OR '1'='1",
            "should_block": True,
            "description": "Boolean-based injection"
        },
        {
            "input": "'; WAITFOR DELAY '00:00:05' --",
            "should_block": True,
            "description": "Time-based injection"
        },
        {
            "input": "Show me all users from the marketing department",
            "should_block": False,
            "description": "Legitimate query"
        }
    ]


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"
    os.environ["ENABLE_METRICS"] = "false"
    
    yield
    
    # Cleanup
    test_vars = ["TESTING", "LOG_LEVEL", "ENABLE_METRICS"]
    for var in test_vars:
        os.environ.pop(var, None)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    def _mock_response(query: str) -> str:
        """Return a mock SQL response based on the input query."""
        query_lower = query.lower()
        
        if "users" in query_lower and "all" in query_lower:
            return "SELECT * FROM users"
        elif "count" in query_lower and "orders" in query_lower:
            return "SELECT COUNT(*) FROM orders"
        elif "age" in query_lower and "25" in query_lower:
            return "SELECT * FROM users WHERE age > 25"
        elif "expensive" in query_lower and "products" in query_lower:
            return "SELECT * FROM products ORDER BY price DESC LIMIT 10"
        else:
            return "SELECT 1"  # Default response
    
    return _mock_response


@pytest.fixture
def benchmark_data_sample():
    """Sample benchmark data for testing evaluation."""
    return {
        "spider": [
            {
                "question": "How many singers do we have?",
                "sql": "SELECT count(*) FROM singer",
                "db_id": "concert_singer",
                "difficulty": "easy"
            },
            {
                "question": "What is the average age of all singers?",
                "sql": "SELECT avg(age) FROM singer",
                "db_id": "concert_singer", 
                "difficulty": "easy"
            }
        ],
        "wikisql": [
            {
                "question": "How many entries are there?",
                "sql": "SELECT count(*) FROM table",
                "table_id": "1-1000181-1",
                "difficulty": "easy"
            }
        ]
    }


class MockStreamlitApp:
    """Mock Streamlit app for UI testing."""
    
    def __init__(self):
        self.session_state = {}
        self.sidebar_elements = []
        self.main_elements = []
    
    def write(self, content):
        self.main_elements.append(("write", content))
    
    def text_input(self, label, value="", key=None):
        if key and key in self.session_state:
            return self.session_state[key]
        return value
    
    def button(self, label, key=None):
        return False  # Default to not pressed
    
    def sidebar(self):
        return self


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for UI testing."""
    return MockStreamlitApp()


@pytest.fixture(scope="session")
def performance_thresholds():
    """Performance testing thresholds."""
    return {
        "query_generation_time": 2.0,  # seconds
        "database_connection_time": 1.0,  # seconds
        "ui_response_time": 0.5,  # seconds
        "memory_usage_mb": 100,  # MB
    }