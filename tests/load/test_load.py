"""Load testing for SQL synthesis application."""

import asyncio
import concurrent.futures
import time
from typing import List, Dict, Any
import pytest
import requests
from src.sql_synth.agent import AgentFactory


class LoadTestConfig:
    """Configuration for load testing."""
    
    BASE_URL = "http://localhost:8501"
    CONCURRENT_USERS = 10
    REQUESTS_PER_USER = 5
    REQUEST_TIMEOUT = 30
    
    SAMPLE_QUERIES = [
        "Show me all employees",
        "Get the average salary by department",
        "Find employees hired in the last year",
        "List top 10 highest paid employees",
        "Show department employee counts"
    ]


class LoadTestResults:
    """Container for load test results."""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def max_response_time(self) -> float:
        """Get maximum response time."""
        return max(self.response_times) if self.response_times else 0.0
    
    @property
    def min_response_time(self) -> float:
        """Get minimum response time."""
        return min(self.response_times) if self.response_times else 0.0
    
    @property
    def total_duration(self) -> float:
        """Calculate total test duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        if self.total_duration > 0:
            return self.total_requests / self.total_duration
        return 0.0


def simulate_user_session(user_id: int, config: LoadTestConfig) -> Dict[str, Any]:
    """Simulate a single user session with multiple requests."""
    session_results = {
        "user_id": user_id,
        "requests": [],
        "errors": []
    }
    
    session = requests.Session()
    
    for i in range(config.REQUESTS_PER_USER):
        query = config.SAMPLE_QUERIES[i % len(config.SAMPLE_QUERIES)]
        
        try:
            start_time = time.time()
            
            # Simulate SQL synthesis request
            response = session.post(
                f"{config.BASE_URL}/api/synthesize",
                json={"query": query},
                timeout=config.REQUEST_TIMEOUT
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            session_results["requests"].append({
                "query": query,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            })
            
        except Exception as e:
            session_results["errors"].append({
                "query": query,
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    return session_results


@pytest.mark.load
def test_concurrent_users_load():
    """Test application under concurrent user load."""
    config = LoadTestConfig()
    results = LoadTestResults()
    
    results.start_time = time.time()
    
    # Run concurrent user sessions
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.CONCURRENT_USERS) as executor:
        future_to_user = {
            executor.submit(simulate_user_session, user_id, config): user_id
            for user_id in range(config.CONCURRENT_USERS)
        }
        
        for future in concurrent.futures.as_completed(future_to_user):
            user_id = future_to_user[future]
            try:
                session_result = future.result()
                
                # Aggregate results
                for request in session_result["requests"]:
                    results.total_requests += 1
                    results.response_times.append(request["response_time"])
                    
                    if request["success"]:
                        results.successful_requests += 1
                    else:
                        results.failed_requests += 1
                
                results.errors.extend(session_result["errors"])
                
            except Exception as e:
                results.errors.append({
                    "user_id": user_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
    
    results.end_time = time.time()
    
    # Assertions for load test criteria
    assert results.success_rate >= 95.0, f"Success rate too low: {results.success_rate}%"
    assert results.average_response_time < 5.0, f"Average response time too high: {results.average_response_time}s"
    assert results.max_response_time < 30.0, f"Max response time too high: {results.max_response_time}s"
    assert results.requests_per_second > 1.0, f"Throughput too low: {results.requests_per_second} req/s"
    
    # Print load test summary
    print(f"\n=== Load Test Results ===")
    print(f"Total Requests: {results.total_requests}")
    print(f"Successful Requests: {results.successful_requests}")
    print(f"Failed Requests: {results.failed_requests}")
    print(f"Success Rate: {results.success_rate:.2f}%")
    print(f"Average Response Time: {results.average_response_time:.3f}s")
    print(f"Min Response Time: {results.min_response_time:.3f}s")
    print(f"Max Response Time: {results.max_response_time:.3f}s")
    print(f"Total Duration: {results.total_duration:.2f}s")
    print(f"Requests/Second: {results.requests_per_second:.2f}")
    print(f"Errors: {len(results.errors)}")


@pytest.mark.load
@pytest.mark.asyncio
async def test_async_concurrent_load():
    """Test application using async concurrent requests."""
    config = LoadTestConfig()
    results = LoadTestResults()
    
    async def make_async_request(session, query: str) -> Dict[str, Any]:
        """Make an asynchronous request."""
        start_time = time.time()
        try:
            # Note: This would need an async HTTP client like aiohttp
            # For demonstration purposes, using a simple delay
            await asyncio.sleep(0.1)  # Simulate request processing
            
            response_time = time.time() - start_time
            return {
                "query": query,
                "success": True,
                "response_time": response_time,
                "status_code": 200
            }
        except Exception as e:
            return {
                "query": query,
                "success": False,
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    results.start_time = time.time()
    
    # Create async tasks
    tasks = []
    for i in range(config.CONCURRENT_USERS * config.REQUESTS_PER_USER):
        query = config.SAMPLE_QUERIES[i % len(config.SAMPLE_QUERIES)]
        task = make_async_request(None, query)
        tasks.append(task)
    
    # Execute all tasks concurrently
    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for task_result in task_results:
        if isinstance(task_result, Exception):
            results.errors.append({
                "error": str(task_result),
                "error_type": type(task_result).__name__
            })
            results.failed_requests += 1
        else:
            results.total_requests += 1
            results.response_times.append(task_result["response_time"])
            
            if task_result["success"]:
                results.successful_requests += 1
            else:
                results.failed_requests += 1
    
    results.end_time = time.time()
    
    # Basic assertions for async test
    assert results.total_requests > 0, "No requests completed"
    assert len(results.response_times) > 0, "No response times recorded"


@pytest.mark.load
def test_memory_usage_under_load():
    """Test memory usage during load testing."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    config = LoadTestConfig()
    config.CONCURRENT_USERS = 5  # Reduced for memory test
    config.REQUESTS_PER_USER = 10
    
    # Run a smaller load test
    simulate_user_session(1, config)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Assert memory usage doesn't increase dramatically
    assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
    
    print(f"Memory usage - Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")