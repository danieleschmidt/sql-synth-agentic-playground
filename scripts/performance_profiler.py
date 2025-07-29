#!/usr/bin/env python3
"""Advanced Performance Profiling and Optimization Tools.

This script provides comprehensive performance analysis capabilities for the 
SQL Synthesis Agentic Playground, including memory profiling, execution timing,
query performance analysis, and optimization recommendations.
"""

import argparse
import asyncio
import cProfile
import io
import json
import logging
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
import psutil
import pandas as pd

# Performance monitoring imports
try:
    import memory_profiler
    import line_profiler
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    print("Warning: Advanced profiling tools not available. Install with: pip install memory-profiler line-profiler")

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    function_name: str
    execution_time: float
    memory_usage: float
    peak_memory: float
    cpu_usage: float
    query_time: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    error_rate: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class PerformanceProfiler:
    """Advanced performance profiler for SQL synthesis operations."""
    
    def __init__(self, output_dir: Path = Path("performance_reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[PerformanceMetrics] = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up performance logging."""
        logger = logging.getLogger("performance_profiler")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "performance.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    @contextmanager
    def profile_execution(self, function_name: str) -> Iterator[PerformanceMetrics]:
        """Context manager for profiling function execution."""
        # Start monitoring
        tracemalloc.start()
        process = psutil.Process()
        start_time = time.perf_counter()
        start_cpu = process.cpu_percent()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # Collect metrics
            end_time = time.perf_counter()
            end_cpu = process.cpu_percent()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            metrics = PerformanceMetrics(
                function_name=function_name,
                execution_time=end_time - start_time,
                memory_usage=current / 1024 / 1024,  # MB
                peak_memory=peak / 1024 / 1024,  # MB
                cpu_usage=(start_cpu + end_cpu) / 2
            )
            
            self.metrics.append(metrics)
            self.logger.info(f"Profiled {function_name}: {metrics.execution_time:.3f}s, {metrics.peak_memory:.2f}MB peak")

    def profile_sql_query(self, query: str, execution_func, *args, **kwargs) -> PerformanceMetrics:
        """Profile SQL query execution with database-specific metrics."""
        with self.profile_execution(f"sql_query_{hash(query) % 10000}") as _:
            start_time = time.perf_counter()
            
            try:
                result = execution_func(*args, **kwargs)
                query_time = time.perf_counter() - start_time
                error_rate = 0.0
            except Exception as e:
                query_time = time.perf_counter() - start_time
                error_rate = 1.0
                self.logger.error(f"Query failed: {e}")
                raise
            
            # Update the last metric with query-specific data
            if self.metrics:
                self.metrics[-1].query_time = query_time
                self.metrics[-1].error_rate = error_rate
                
            return self.metrics[-1]

    def run_memory_profile(self, target_function, *args, **kwargs) -> str:
        """Run detailed memory profiling using memory_profiler."""
        if not PROFILING_AVAILABLE:
            return "Memory profiling not available - install memory-profiler"
            
        try:
            from memory_profiler import profile
            
            # Create a wrapper function for profiling
            @profile
            def wrapper():
                return target_function(*args, **kwargs)
            
            # Capture output
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()
            
            try:
                wrapper()
                memory_report = mystdout.getvalue()
            finally:
                sys.stdout = old_stdout
                
            # Save report
            report_file = self.output_dir / f"memory_profile_{int(time.time())}.txt"
            with open(report_file, 'w') as f:
                f.write(memory_report)
                
            return memory_report
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            return f"Memory profiling error: {e}"

    def run_line_profile(self, target_function, *args, **kwargs) -> str:
        """Run line-by-line performance profiling."""
        if not PROFILING_AVAILABLE:
            return "Line profiling not available - install line-profiler"
            
        try:
            profiler = cProfile.Profile()
            profiler.enable()
            
            result = target_function(*args, **kwargs)
            
            profiler.disable()
            
            # Generate report
            output = io.StringIO()
            stats = pstats.Stats(profiler, stream=output)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_report = output.getvalue()
            
            # Save report
            report_file = self.output_dir / f"line_profile_{int(time.time())}.txt"
            with open(report_file, 'w') as f:
                f.write(profile_report)
                
            return profile_report
            
        except Exception as e:
            self.logger.error(f"Line profiling failed: {e}")
            return f"Line profiling error: {e}"

    def benchmark_sql_generation(self, queries: List[str], generation_function) -> Dict[str, Any]:
        """Benchmark SQL generation performance across multiple queries."""
        results = {
            "total_queries": len(queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "average_time": 0.0,
            "median_time": 0.0,
            "p95_time": 0.0,
            "max_memory": 0.0,
            "query_results": []
        }
        
        execution_times = []
        memory_peaks = []
        
        for i, query in enumerate(queries):
            try:
                with self.profile_execution(f"benchmark_query_{i}") as _:
                    _ = generation_function(query)
                    
                # Get the latest metric
                metric = self.metrics[-1]
                execution_times.append(metric.execution_time)
                memory_peaks.append(metric.peak_memory)
                
                results["successful_queries"] += 1
                results["query_results"].append({
                    "query_id": i,
                    "success": True,
                    "execution_time": metric.execution_time,
                    "memory_usage": metric.peak_memory
                })
                
            except Exception as e:
                results["failed_queries"] += 1
                results["query_results"].append({
                    "query_id": i,
                    "success": False,
                    "error": str(e)
                })
                self.logger.error(f"Benchmark query {i} failed: {e}")
        
        if execution_times:
            results["average_time"] = sum(execution_times) / len(execution_times)
            results["median_time"] = sorted(execution_times)[len(execution_times) // 2]
            results["p95_time"] = sorted(execution_times)[int(len(execution_times) * 0.95)]
            results["max_memory"] = max(memory_peaks)
        
        return results

    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on collected metrics."""
        recommendations = []
        
        if not self.metrics:
            return [{"type": "info", "message": "No performance data available for analysis"}]
        
        # Analyze execution times
        execution_times = [m.execution_time for m in self.metrics]
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        
        if max_time > 5.0:
            recommendations.append({
                "type": "critical",
                "category": "performance",
                "message": f"Detected slow operations (max: {max_time:.2f}s). Consider caching or optimization.",
                "suggested_actions": ["Implement query result caching", "Optimize database queries", "Use async processing"]
            })
        
        # Analyze memory usage
        memory_usage = [m.peak_memory for m in self.metrics]
        avg_memory = sum(memory_usage) / len(memory_usage)
        max_memory = max(memory_usage)
        
        if max_memory > 500:  # 500MB
            recommendations.append({
                "type": "warning",
                "category": "memory",
                "message": f"High memory usage detected (max: {max_memory:.2f}MB).",
                "suggested_actions": ["Implement memory pooling", "Optimize data structures", "Add memory limits"]
            })
        
        # Analyze error rates
        error_rates = [m.error_rate for m in self.metrics if m.error_rate is not None]
        if error_rates:
            avg_error_rate = sum(error_rates) / len(error_rates)
            if avg_error_rate > 0.05:  # 5% error rate
                recommendations.append({
                    "type": "critical",
                    "category": "reliability",
                    "message": f"High error rate detected ({avg_error_rate:.1%}).",
                    "suggested_actions": ["Improve error handling", "Add input validation", "Implement retry logic"]
                })
        
        # Performance trends
        if len(self.metrics) > 10:
            recent_times = execution_times[-5:] if len(execution_times) > 5 else execution_times
            older_times = execution_times[:-5] if len(execution_times) > 5 else []
            
            if older_times and sum(recent_times) / len(recent_times) > sum(older_times) / len(older_times) * 1.2:
                recommendations.append({
                    "type": "warning",
                    "category": "trend",
                    "message": "Performance degradation trend detected.",
                    "suggested_actions": ["Review recent changes", "Check for resource leaks", "Monitor system resources"]
                })
        
        return recommendations

    def export_metrics(self, format: str = "json") -> str:
        """Export collected metrics to various formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = self.output_dir / f"performance_metrics_{timestamp}.json"
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_metrics": len(self.metrics),
                "metrics": [asdict(m) for m in self.metrics],
                "recommendations": self.generate_optimization_recommendations()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format.lower() == "csv":
            filename = self.output_dir / f"performance_metrics_{timestamp}.csv"
            df = pd.DataFrame([asdict(m) for m in self.metrics])
            df.to_csv(filename, index=False)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return str(filename)

    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.metrics:
            return "No performance data available"
        
        report_lines = [
            "# Performance Analysis Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Operations Profiled: {len(self.metrics)}",
            "",
            "## Summary Statistics",
        ]
        
        # Calculate summary statistics
        execution_times = [m.execution_time for m in self.metrics]
        memory_usage = [m.peak_memory for m in self.metrics]
        
        report_lines.extend([
            f"- Average Execution Time: {sum(execution_times) / len(execution_times):.3f}s",
            f"- Maximum Execution Time: {max(execution_times):.3f}s",
            f"- Average Memory Usage: {sum(memory_usage) / len(memory_usage):.2f}MB",
            f"- Peak Memory Usage: {max(memory_usage):.2f}MB",
            "",
            "## Top 10 Slowest Operations",
        ])
        
        # Sort by execution time and show top 10
        sorted_metrics = sorted(self.metrics, key=lambda x: x.execution_time, reverse=True)[:10]
        for i, metric in enumerate(sorted_metrics, 1):
            report_lines.append(
                f"{i}. {metric.function_name}: {metric.execution_time:.3f}s "
                f"({metric.peak_memory:.2f}MB)"
            )
        
        report_lines.extend(["", "## Optimization Recommendations"])
        recommendations = self.generate_optimization_recommendations()
        
        for rec in recommendations:
            report_lines.extend([
                f"### {rec['type'].upper()}: {rec.get('category', 'General')}",
                f"- {rec['message']}",
                "- Suggested Actions:",
            ])
            for action in rec.get('suggested_actions', []):
                report_lines.append(f"  - {action}")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return report_content

def main():
    """Main CLI interface for the performance profiler."""
    parser = argparse.ArgumentParser(description="Advanced Performance Profiler for SQL Synthesis")
    parser.add_argument("--output-dir", type=Path, default=Path("performance_reports"),
                       help="Output directory for reports")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark suite")
    parser.add_argument("--export-format", choices=["json", "csv"], default="json",
                       help="Export format for metrics")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate comprehensive performance report")
    
    args = parser.parse_args()
    
    profiler = PerformanceProfiler(args.output_dir)
    
    if args.benchmark:
        print("Running performance benchmark suite...")
        # Example benchmark queries
        test_queries = [
            "Show me all users who registered last month",
            "What are the top 10 selling products?",
            "Find orders with total amount greater than $100",
            "List active users with their recent orders",
            "Show revenue by month for the last year"
        ]
        
        def mock_sql_generator(query: str) -> str:
            """Mock SQL generation function for benchmarking."""
            time.sleep(0.1)  # Simulate processing time
            return f"SELECT * FROM table WHERE condition = '{query[:10]}';"
        
        results = profiler.benchmark_sql_generation(test_queries, mock_sql_generator)
        print(f"Benchmark Results: {results['successful_queries']}/{results['total_queries']} successful")
        print(f"Average time: {results['average_time']:.3f}s")
    
    if args.generate_report:
        print("Generating performance report...")
        report = profiler.generate_performance_report()
        print("Performance report generated")
    
    if profiler.metrics:
        export_file = profiler.export_metrics(args.export_format)
        print(f"Metrics exported to: {export_file}")

if __name__ == "__main__":
    main()