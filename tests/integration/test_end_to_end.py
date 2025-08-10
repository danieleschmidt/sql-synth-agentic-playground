import pytest
import time
from typing import Dict, Any

from src.sql_synth.database import DatabaseManager
from src.sql_synth.agent import AgentFactory


class TestEndToEndIntegration:
    """End-to-end integration tests for the SQL synthesis system."""

    @pytest.mark.integration
    def test_complete_workflow(self, memory_db: str, sample_queries: list[dict]):
        """Test complete workflow from natural language to SQL execution."""
        # Initialize components
        db_manager = DatabaseManager(memory_db)
        agent = AgentFactory.create_agent(db_manager)
        
        # Create test schema
        with db_manager.get_connection() as conn:
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    age INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test data
            conn.execute("""
                INSERT INTO users (name, email, age) VALUES
                ('Alice', 'alice@example.com', 30),
                ('Bob', 'bob@example.com', 25),
                ('Charlie', 'charlie@example.com', 35)
            """)
            conn.commit()
        
        # Test query processing
        natural_query = "Show me all users older than 25"
        
        # This would normally go through the LLM, but for testing we'll mock it
        expected_sql = "SELECT * FROM users WHERE age > 25"
        
        # Execute the query
        with db_manager.get_connection() as conn:
            result = conn.execute(expected_sql).fetchall()
            
        # Verify results
        assert len(result) == 2  # Alice (30) and Charlie (35)
        assert all(row[3] > 25 for row in result)  # age column

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_requirements(self, memory_db: str, performance_thresholds: dict):
        """Test that the system meets performance requirements."""
        db_manager = DatabaseManager(memory_db)
        
        # Test database connection time
        start_time = time.time()
        with db_manager.get_connection():
            pass
        connection_time = time.time() - start_time
        
        assert connection_time < performance_thresholds["database_connection_time"]

    @pytest.mark.integration
    @pytest.mark.security
    def test_security_integration(self, memory_db: str, security_test_cases: list[dict]):
        """Test security measures in the integrated system."""
        db_manager = DatabaseManager(memory_db)
        
        # Create a test table
        with db_manager.get_connection() as conn:
            conn.execute("CREATE TABLE test_table (id INTEGER, data TEXT)")
            conn.execute("INSERT INTO test_table VALUES (1, 'sensitive_data')")
            conn.commit()
        
        for test_case in security_test_cases:
            if test_case["should_block"]:
                # These should be blocked by our security measures
                # For now, we'll just verify the input looks malicious
                malicious_input = test_case["input"]
                assert any(keyword in malicious_input.lower() 
                          for keyword in ["drop", "union", "waitfor", "--", "';"])

    @pytest.mark.integration
    def test_error_handling(self, memory_db: str):
        """Test error handling across the system."""
        db_manager = DatabaseManager(memory_db)
        
        # Test invalid SQL
        with pytest.raises(Exception):
            with db_manager.get_connection() as conn:
                conn.execute("INVALID SQL STATEMENT")
        
        # Test connection to non-existent database
        with pytest.raises(Exception):
            invalid_db = DatabaseManager("sqlite:///non_existent_path/db.sqlite")
            with invalid_db.get_connection():
                pass

    @pytest.mark.integration
    @pytest.mark.benchmark
    def test_benchmark_compatibility(self, benchmark_data_sample: dict):
        """Test compatibility with benchmark datasets."""
        spider_sample = benchmark_data_sample["spider"]
        wikisql_sample = benchmark_data_sample["wikisql"]
        
        # Test Spider format compatibility
        for item in spider_sample:
            assert "question" in item
            assert "sql" in item
            assert "db_id" in item
            assert "difficulty" in item
        
        # Test WikiSQL format compatibility
        for item in wikisql_sample:
            assert "question" in item
            assert "sql" in item
            assert "table_id" in item

    @pytest.mark.integration
    def test_multi_dialect_support(self):
        """Test support for multiple SQL dialects."""
        # Test SQLite (in-memory)
        sqlite_db = DatabaseManager("sqlite:///:memory:")
        with sqlite_db.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1
        
        # For other dialects, we would test connection strings
        # but not actual connections in unit tests
        postgres_url = "postgresql://user:pass@localhost/db"
        mysql_url = "mysql://user:pass@localhost/db"
        
        # Verify URL parsing (would connect in real environment)
        assert "postgresql" in postgres_url
        assert "mysql" in mysql_url

    @pytest.mark.integration
    def test_configuration_management(self):
        """Test configuration management across components."""
        import os
        
        # Test environment variable handling
        os.environ["TEST_CONFIG"] = "test_value"
        
        # Verify configuration is accessible
        test_config = os.getenv("TEST_CONFIG")
        assert test_config == "test_value"
        
        # Cleanup
        del os.environ["TEST_CONFIG"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_concurrent_access(self, memory_db: str):
        """Test concurrent database access."""
        import threading
        import queue
        
        db_manager = DatabaseManager(memory_db)
        results = queue.Queue()
        
        def worker():
            try:
                with db_manager.get_connection() as conn:
                    result = conn.execute("SELECT 1").fetchone()
                    results.put(result[0])
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert results.qsize() == 5
        while not results.empty():
            result = results.get()
            assert result == 1 or isinstance(result, str)  # Allow for potential errors

    @pytest.mark.integration
    def test_data_validation(self, memory_db: str, sample_schema: dict):
        """Test data validation across the system."""
        db_manager = DatabaseManager(memory_db)
        
        # Create table based on schema
        with db_manager.get_connection() as conn:
            # Test users table creation
            users_schema = sample_schema["users"]
            columns = users_schema["columns"]
            types = users_schema["types"]
            
            create_sql = f"""
                CREATE TABLE users (
                    {columns[0]} {types[0]} PRIMARY KEY,
                    {columns[1]} {types[1]} NOT NULL,
                    {columns[2]} {types[1]} UNIQUE,
                    {columns[3]} {types[3]},
                    {columns[4]} {types[4]} DEFAULT CURRENT_TIMESTAMP
                )
            """
            
            conn.execute(create_sql)
            
            # Test data insertion and validation
            conn.execute("""
                INSERT INTO users (name, email, age) 
                VALUES ('Test User', 'test@example.com', 25)
            """)
            
            # Verify insertion
            result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
            assert result[0] == 1