"""Concurrent processing capabilities for SQL synthesis agent.

This module provides thread-safe concurrent execution, load balancing,
and auto-scaling capabilities for handling multiple requests efficiently.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from queue import Queue, Empty
import statistics


logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a concurrent task execution."""
    task_id: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    start_time: datetime
    end_time: datetime


@dataclass
class WorkerStats:
    """Statistics for a worker thread."""
    worker_id: str
    tasks_completed: int
    tasks_failed: int
    total_execution_time: float
    avg_execution_time: float
    last_task_time: Optional[datetime]
    is_active: bool


class ThreadSafeCounter:
    """Thread-safe counter for tracking metrics."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= 1
            return self._value
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> None:
        """Set counter value."""
        with self._lock:
            self._value = value


class LoadBalancer:
    """Simple round-robin load balancer for worker selection."""
    
    def __init__(self, workers: List[str]):
        self.workers = workers
        self.current_index = 0
        self._lock = threading.Lock()
        self.worker_loads = {worker: ThreadSafeCounter() for worker in workers}
    
    def get_next_worker(self) -> str:
        """Get next worker using round-robin algorithm."""
        with self._lock:
            worker = self.workers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.workers)
            return worker
    
    def get_least_loaded_worker(self) -> str:
        """Get worker with least current load."""
        min_load = float('inf')
        selected_worker = self.workers[0]
        
        for worker in self.workers:
            current_load = self.worker_loads[worker].get()
            if current_load < min_load:
                min_load = current_load
                selected_worker = worker
        
        return selected_worker
    
    def increment_worker_load(self, worker: str) -> None:
        """Increment load for a worker."""
        if worker in self.worker_loads:
            self.worker_loads[worker].increment()
    
    def decrement_worker_load(self, worker: str) -> None:
        """Decrement load for a worker."""
        if worker in self.worker_loads:
            self.worker_loads[worker].decrement()
    
    def get_load_stats(self) -> Dict[str, int]:
        """Get current load for all workers."""
        return {worker: counter.get() for worker, counter in self.worker_loads.items()}


class ConcurrentExecutor:
    """Manages concurrent execution of SQL synthesis tasks."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.queue_size = queue_size
        
        # Thread pool and task management
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SQLSynth")
        self.task_queue = Queue(maxsize=queue_size)
        self.active_tasks = ThreadSafeCounter()
        self.completed_tasks = ThreadSafeCounter()
        self.failed_tasks = ThreadSafeCounter()
        
        # Load balancing
        worker_ids = [f"worker_{i}" for i in range(max_workers)]
        self.load_balancer = LoadBalancer(worker_ids)
        
        # Performance tracking
        self.task_results: List[TaskResult] = []
        self.worker_stats: Dict[str, WorkerStats] = {
            worker_id: WorkerStats(
                worker_id=worker_id,
                tasks_completed=0,
                tasks_failed=0,
                total_execution_time=0.0,
                avg_execution_time=0.0,
                last_task_time=None,
                is_active=False
            )
            for worker_id in worker_ids
        }
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = True
        self.scale_up_threshold = 0.8  # Scale up when 80% of workers are busy
        self.scale_down_threshold = 0.3  # Scale down when less than 30% are busy
        self.min_workers = 2
        self.max_workers_limit = 10
        
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        logger.info("ConcurrentExecutor initialized with %d workers", max_workers)
    
    def submit_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit a task for concurrent execution.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task ID for tracking
            
        Raises:
            RuntimeError: If queue is full
        """
        task_id = f"task_{int(time.time() * 1000)}_{self.completed_tasks.get()}"
        
        try:
            # Select worker using load balancing
            worker_id = self.load_balancer.get_least_loaded_worker()
            
            # Submit task to thread pool
            future = self.executor.submit(self._execute_task, task_id, worker_id, func, *args, **kwargs)
            
            # Track active task
            self.active_tasks.increment()
            self.load_balancer.increment_worker_load(worker_id)
            
            logger.debug("Task submitted: %s to worker %s", task_id, worker_id)
            return task_id
            
        except Exception as e:
            logger.error("Failed to submit task: %s", str(e))
            raise RuntimeError(f"Failed to submit task: {e}") from e
    
    def _execute_task(self, task_id: str, worker_id: str, func: Callable, *args, **kwargs) -> TaskResult:
        """Execute a task and track performance metrics."""
        start_time = datetime.now()
        
        # Update worker status
        self.worker_stats[worker_id].is_active = True
        self.worker_stats[worker_id].last_task_time = start_time
        
        try:
            # Execute the actual function
            result = func(*args, **kwargs)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                error=None,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
            # Update statistics
            self._update_worker_stats(worker_id, True, execution_time)
            self.completed_tasks.increment()
            
            logger.debug("Task completed successfully: %s (%.3fs)", task_id, execution_time)
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Create error result
            task_result = TaskResult(
                task_id=task_id,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
            # Update statistics
            self._update_worker_stats(worker_id, False, execution_time)
            self.failed_tasks.increment()
            
            logger.error("Task failed: %s - %s (%.3fs)", task_id, str(e), execution_time)
        
        finally:
            # Clean up
            self.active_tasks.decrement()
            self.load_balancer.decrement_worker_load(worker_id)
            self.worker_stats[worker_id].is_active = False
            
            # Store result for monitoring
            self.task_results.append(task_result)
            
            # Keep only recent results (last 1000)
            if len(self.task_results) > 1000:
                self.task_results = self.task_results[-1000:]
        
        return task_result
    
    def _update_worker_stats(self, worker_id: str, success: bool, execution_time: float) -> None:
        """Update worker performance statistics."""
        worker = self.worker_stats[worker_id]
        
        if success:
            worker.tasks_completed += 1
        else:
            worker.tasks_failed += 1
        
        worker.total_execution_time += execution_time
        total_tasks = worker.tasks_completed + worker.tasks_failed
        worker.avg_execution_time = worker.total_execution_time / total_tasks if total_tasks > 0 else 0.0
    
    def execute_batch(self, tasks: List[Tuple[Callable, tuple, dict]], timeout: Optional[float] = None) -> List[TaskResult]:
        """Execute multiple tasks concurrently and wait for all to complete.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            timeout: Maximum time to wait for all tasks
            
        Returns:
            List of TaskResult objects
        """
        if not tasks:
            return []
        
        # Submit all tasks
        futures = []
        task_ids = []
        
        for func, args, kwargs in tasks:
            task_id = self.submit_task(func, *args, **kwargs)
            task_ids.append(task_id)
        
        # Wait for completion
        results = []
        start_time = time.time()
        
        # Get futures from executor
        active_futures = list(self.executor._threads)
        
        try:
            # Wait for all tasks with timeout
            for future in as_completed(active_futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error("Batch task failed: %s", str(e))
                    
                    # Check timeout
                    if timeout and (time.time() - start_time) > timeout:
                        logger.warning("Batch execution timeout after %.2fs", timeout)
                        break
        
        except TimeoutError:
            logger.warning("Batch execution timed out after %.2fs", timeout or 0)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_tasks = self.completed_tasks.get() + self.failed_tasks.get()
        success_rate = self.completed_tasks.get() / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate recent performance (last 100 tasks)
        recent_results = self.task_results[-100:] if len(self.task_results) >= 100 else self.task_results
        
        if recent_results:
            recent_times = [r.execution_time for r in recent_results]
            avg_execution_time = statistics.mean(recent_times)
            p95_execution_time = statistics.quantiles(recent_times, n=20)[18] if len(recent_times) > 20 else max(recent_times)
        else:
            avg_execution_time = 0.0
            p95_execution_time = 0.0
        
        return {
            "active_tasks": self.active_tasks.get(),
            "completed_tasks": self.completed_tasks.get(),
            "failed_tasks": self.failed_tasks.get(),
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "p95_execution_time": p95_execution_time,
            "worker_count": self.max_workers,
            "worker_utilization": self.active_tasks.get() / self.max_workers,
            "load_balancer_stats": self.load_balancer.get_load_stats(),
            "worker_stats": {wid: {
                "tasks_completed": ws.tasks_completed,
                "tasks_failed": ws.tasks_failed,
                "avg_execution_time": ws.avg_execution_time,
                "is_active": ws.is_active,
                "last_task_time": ws.last_task_time.isoformat() if ws.last_task_time else None
            } for wid, ws in self.worker_stats.items()}
        }
    
    def start_auto_scaling_monitor(self) -> None:
        """Start background thread for auto-scaling monitoring."""
        if self.auto_scaling_enabled and (self._monitoring_thread is None or not self._monitoring_thread.is_alive()):
            self._shutdown_event.clear()
            self._monitoring_thread = threading.Thread(target=self._auto_scaling_worker, daemon=True)
            self._monitoring_thread.start()
            logger.info("Auto-scaling monitor started")
    
    def stop_auto_scaling_monitor(self) -> None:
        """Stop auto-scaling monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._shutdown_event.set()
            self._monitoring_thread.join(timeout=5)
            logger.info("Auto-scaling monitor stopped")
    
    def _auto_scaling_worker(self) -> None:
        """Background worker for auto-scaling decisions."""
        while not self._shutdown_event.is_set():
            try:
                utilization = self.active_tasks.get() / self.max_workers
                
                # Scale up decision
                if utilization > self.scale_up_threshold and self.max_workers < self.max_workers_limit:
                    new_worker_count = min(self.max_workers + 1, self.max_workers_limit)
                    self._scale_workers(new_worker_count)
                    logger.info("Scaled up to %d workers (utilization: %.2f)", new_worker_count, utilization)
                
                # Scale down decision
                elif utilization < self.scale_down_threshold and self.max_workers > self.min_workers:
                    new_worker_count = max(self.max_workers - 1, self.min_workers)
                    self._scale_workers(new_worker_count)
                    logger.info("Scaled down to %d workers (utilization: %.2f)", new_worker_count, utilization)
                
                # Wait before next check
                self._shutdown_event.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Error in auto-scaling worker: %s", str(e))
                self._shutdown_event.wait(60)  # Wait longer on error
    
    def _scale_workers(self, new_count: int) -> None:
        """Scale the number of worker threads."""
        if new_count == self.max_workers:
            return
        
        # Update worker count
        old_count = self.max_workers
        self.max_workers = new_count
        
        # Recreate thread pool with new size
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=new_count, thread_name_prefix="SQLSynth")
        
        # Update load balancer
        worker_ids = [f"worker_{i}" for i in range(new_count)]
        self.load_balancer = LoadBalancer(worker_ids)
        
        # Update worker stats
        if new_count > old_count:
            # Add new workers
            for i in range(old_count, new_count):
                worker_id = f"worker_{i}"
                self.worker_stats[worker_id] = WorkerStats(
                    worker_id=worker_id,
                    tasks_completed=0,
                    tasks_failed=0,
                    total_execution_time=0.0,
                    avg_execution_time=0.0,
                    last_task_time=None,
                    is_active=False
                )
        else:
            # Remove excess workers
            for i in range(new_count, old_count):
                worker_id = f"worker_{i}"
                if worker_id in self.worker_stats:
                    del self.worker_stats[worker_id]
        
        # Shutdown old executor gracefully
        try:
            old_executor.shutdown(wait=True, timeout=30)
        except Exception as e:
            logger.warning("Error shutting down old executor: %s", str(e))
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Shutdown the concurrent executor."""
        logger.info("Shutting down concurrent executor...")
        
        # Stop auto-scaling
        self.stop_auto_scaling_monitor()
        
        # Shutdown thread pool
        try:
            self.executor.shutdown(wait=wait, timeout=timeout)
            logger.info("Concurrent executor shutdown complete")
        except Exception as e:
            logger.error("Error during executor shutdown: %s", str(e))


# Global concurrent executor instance
concurrent_executor = ConcurrentExecutor()


def get_concurrent_executor() -> ConcurrentExecutor:
    """Get global concurrent executor instance."""
    return concurrent_executor


# Decorator for concurrent execution
def concurrent_task(timeout: Optional[float] = None):
    """Decorator to execute function concurrently."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Submit task to concurrent executor
            task_id = concurrent_executor.submit_task(func, *args, **kwargs)
            
            # For now, we'll execute synchronously and return the result
            # In a real async environment, this would return a Future
            return func(*args, **kwargs)
        return wrapper
    return decorator