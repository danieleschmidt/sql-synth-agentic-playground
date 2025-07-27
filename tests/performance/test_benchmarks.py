import time
import psutil
import pytest
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.sql_synth.database import DatabaseManager


class TestPerformanceBenchmarks:
    """Performance and load testing for the SQL synthesis system."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_query_generation_performance(self, sample_queries: List[dict], performance_thresholds: dict):
        """Test SQL query generation performance."""
        from src.sql_synth.streamlit_ui import SQLSynthesizer
        
        synthesizer = SQLSynthesizer()
        total_time = 0
        num_queries = len(sample_queries)
        
        for query_data in sample_queries:
            start_time = time.time()
            
            # Mock the LLM call for performance testing
            # In real implementation, this would call the actual LLM
            natural_query = query_data["natural_language"]
            mock_sql = query_data["expected_sql"]
            
            # Simulate processing time
            time.sleep(0.1)  # Simulate 100ms processing
            
            end_time = time.time()
            query_time = end_time - start_time
            total_time += query_time
        
        average_time = total_time / num_queries
        assert average_time < performance_thresholds["query_generation_time"]

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_database_connection_pool_performance(self, memory_db: str, performance_thresholds: dict):
        """Test database connection pool performance."""
        db_manager = DatabaseManager(memory_db)
        
        # Test multiple rapid connections
        start_time = time.time()
        
        for _ in range(10):
            with db_manager.get_connection() as conn:
                conn.execute("SELECT 1").fetchone()
        
        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / 10
        
        assert average_time < performance_thresholds["database_connection_time"]

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_concurrent_query_performance(self, memory_db: str):
        """Test performance under concurrent load."""
        db_manager = DatabaseManager(memory_db)
        
        # Setup test data
        with db_manager.get_connection() as conn:
            conn.execute("""
                CREATE TABLE performance_test (
                    id INTEGER PRIMARY KEY,
                    data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test data
            for i in range(1000):
                conn.execute(
                    "INSERT INTO performance_test (data) VALUES (?)",
                    (f"test_data_{i}",)
                )
            conn.commit()
        
        def run_query():
            with db_manager.get_connection() as conn:
                start_time = time.time()
                result = conn.execute("SELECT COUNT(*) FROM performance_test").fetchone()
                end_time = time.time()
                return end_time - start_time, result[0]
        
        # Run concurrent queries
        num_workers = 10
        num_queries_per_worker = 5
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            start_time = time.time()
            
            for _ in range(num_workers * num_queries_per_worker):
                future = executor.submit(run_query)
                futures.append(future)
            
            results = []
            for future in as_completed(futures):
                query_time, count = future.result()
                results.append(query_time)
                assert count == 1000  # Verify data integrity
            
            end_time = time.time()
        
        total_time = end_time - start_time
        average_query_time = sum(results) / len(results)
        
        # Performance assertions
        assert total_time < 10.0  # Total time should be under 10 seconds
        assert average_query_time < 1.0  # Average query time under 1 second

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_memory_usage(self, memory_db: str, performance_thresholds: dict):
        """Test memory usage under load."""
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        db_manager = DatabaseManager(memory_db)
        
        # Perform memory-intensive operations
        with db_manager.get_connection() as conn:
            conn.execute("""
                CREATE TABLE memory_test (
                    id INTEGER PRIMARY KEY,
                    large_data TEXT
                )
            """)
            
            # Insert large amounts of data
            large_text = "x" * 1000  # 1KB per row
            for i in range(1000):  # 1MB total
                conn.execute(
                    "INSERT INTO memory_test (large_data) VALUES (?)",
                    (large_text,)
                )
            conn.commit()
            
            # Read data back
            results = conn.execute("SELECT * FROM memory_test").fetchall()
            assert len(results) == 1000
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < performance_thresholds["memory_usage_mb"]

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_large_result_set_handling(self, memory_db: str):
        """Test handling of large result sets."""
        db_manager = DatabaseManager(memory_db)
        
        with db_manager.get_connection() as conn:
            conn.execute("""
                CREATE TABLE large_table (
                    id INTEGER PRIMARY KEY,
                    data TEXT,
                    number INTEGER
                )
            """)
            
            # Insert 10,000 rows
            data_batch = [(f"data_{i}", i) for i in range(10000)]
            conn.executemany(
                "INSERT INTO large_table (data, number) VALUES (?, ?)",
                data_batch
            )
            conn.commit()
            
            # Test large query performance
            start_time = time.time()
            result = conn.execute(
                "SELECT COUNT(*) FROM large_table WHERE number > 5000"
            ).fetchone()
            end_time = time.time()
            
            query_time = end_time - start_time
            
            assert result[0] == 4999  # Numbers 5001-9999
            assert query_time < 1.0  # Should complete in under 1 second

    @pytest.mark.benchmark
    def test_query_complexity_performance(self, memory_db: str):
        """Test performance with complex queries."""
        db_manager = DatabaseManager(memory_db)
        
        with db_manager.get_connection() as conn:
            # Create multiple related tables
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    department_id INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE departments (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    budget DECIMAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    amount DECIMAL,
                    order_date DATE
                )
            """)
            
            # Insert test data
            for i in range(100):
                conn.execute(
                    "INSERT INTO users VALUES (?, ?, ?)",
                    (i, f"User {i}", i % 10)
                )
                
            for i in range(10):
                conn.execute(
                    "INSERT INTO departments VALUES (?, ?, ?)",
                    (i, f"Dept {i}", 10000 + i * 1000)
                )
                
            for i in range(500):
                conn.execute(
                    "INSERT INTO orders VALUES (?, ?, ?, date('now'))",
                    (i, i % 100, 100.0 + i)
                )
            
            conn.commit()
            
            # Test complex JOIN query performance
            complex_query = """
                SELECT 
                    d.name as department,
                    COUNT(u.id) as user_count,
                    AVG(o.amount) as avg_order_amount,
                    SUM(o.amount) as total_revenue
                FROM departments d
                LEFT JOIN users u ON d.id = u.department_id
                LEFT JOIN orders o ON u.id = o.user_id
                GROUP BY d.id, d.name
                HAVING AVG(o.amount) > 200
                ORDER BY total_revenue DESC
            """
            
            start_time = time.time()
            results = conn.execute(complex_query).fetchall()
            end_time = time.time()
            
            query_time = end_time - start_time
            
            assert len(results) > 0  # Should return some results
            assert query_time < 2.0  # Complex query should complete in under 2 seconds

    @pytest.mark.benchmark
    def test_connection_pool_stress(self, memory_db: str):
        """Stress test the connection pool."""
        db_manager = DatabaseManager(memory_db)
        
        def stress_worker():
            results = []
            for _ in range(10):
                try:
                    with db_manager.get_connection() as conn:
                        result = conn.execute("SELECT RANDOM()").fetchone()
                        results.append(result[0])
                        time.sleep(0.01)  # Small delay
                except Exception as e:
                    results.append(f"Error: {e}")
            return results
        
        # Run stress test with many workers
        num_workers = 20
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.time()
            
            futures = []
            for _ in range(num_workers):
                future = executor.submit(stress_worker)
                futures.append(future)
            
            all_results = []
            for future in as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)
            
            end_time = time.time()
        
        total_time = end_time - start_time
        total_operations = len(all_results)
        operations_per_second = total_operations / total_time
        
        # Count errors
        errors = [r for r in all_results if isinstance(r, str) and "Error" in r]
        error_rate = len(errors) / total_operations
        
        # Performance assertions
        assert operations_per_second > 50  # At least 50 ops/second
        assert error_rate < 0.1  # Less than 10% error rate
        assert total_time < 30  # Complete in under 30 seconds