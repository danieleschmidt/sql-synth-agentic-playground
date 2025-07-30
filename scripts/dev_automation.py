#!/usr/bin/env python3
"""
Advanced development automation script for SQL Synthesis Agentic Playground.

This script provides advanced development workflow automation including:
- Intelligent test execution based on changed files
- Performance regression detection
- Automated code quality checks
- Development environment health monitoring
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Set
import json
import time
from datetime import datetime


class DevAutomation:
    """Advanced development automation utilities."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        
    def get_changed_files(self, base_branch: str = "main") -> Set[str]:
        """Get list of changed files compared to base branch."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            return set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
        except subprocess.CalledProcessError:
            return set()
    
    def get_related_tests(self, changed_files: Set[str]) -> Set[str]:
        """Identify test files related to changed source files."""
        related_tests = set()
        
        for file_path in changed_files:
            if file_path.startswith("src/"):
                # Direct test mapping
                relative_path = file_path[4:]  # Remove 'src/'
                test_path = f"tests/test_{relative_path}"
                if Path(self.project_root / test_path).exists():
                    related_tests.add(test_path)
                
                # Module-level test mapping
                if "/" in relative_path:
                    module_path = relative_path.split("/")[0]
                    test_module_path = f"tests/test_{module_path}.py"
                    if Path(self.project_root / test_module_path).exists():
                        related_tests.add(test_module_path)
        
        return related_tests
    
    def run_smart_tests(self, full_suite: bool = False) -> bool:
        """Run tests intelligently based on changed files."""
        print("🧪 Running intelligent test execution...")
        
        if full_suite:
            print("Running full test suite...")
            cmd = ["pytest", "-v", "--cov=src", "--cov-report=term-missing"]
        else:
            changed_files = self.get_changed_files()
            if not changed_files:
                print("No changes detected, running smoke tests...")
                cmd = ["pytest", "-v", "-m", "smoke", "--tb=short"]
            else:
                related_tests = self.get_related_tests(changed_files)
                if related_tests:
                    print(f"Running tests for changed files: {', '.join(changed_files)}")
                    cmd = ["pytest", "-v"] + list(related_tests)
                else:
                    print("No specific tests found, running full suite...")
                    cmd = ["pytest", "-v", "--cov=src", "--cov-report=term-missing"]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
    
    def check_performance_regression(self) -> bool:
        """Check for performance regressions."""
        print("⚡ Checking for performance regressions...")
        
        # Run performance benchmarks
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/performance/", "-v", "--benchmark-json=benchmark.json"],
            cwd=self.project_root,
            capture_output=True
        )
        
        if result.returncode != 0:
            print("❌ Performance tests failed")
            return False
        
        # Analyze benchmark results
        benchmark_file = self.project_root / "benchmark.json"
        if benchmark_file.exists():
            with open(benchmark_file) as f:
                data = json.load(f)
            
            # Simple regression check (can be enhanced with historical data)
            for benchmark in data.get("benchmarks", []):
                if benchmark["stats"]["mean"] > 2.0:  # 2 second threshold
                    print(f"⚠️  Performance regression detected in {benchmark['name']}")
                    return False
        
        print("✅ No performance regressions detected")
        return True
    
    def run_quality_checks(self) -> bool:
        """Run comprehensive code quality checks."""
        print("🔍 Running code quality checks...")
        
        checks = [
            (["ruff", "check", "."], "Ruff linting"),
            (["black", "--check", "."], "Black formatting"),
            (["mypy", "src/"], "Type checking"),
            (["python", "scripts/security_scan.py"], "Security scan")
        ]
        
        all_passed = True
        for cmd, description in checks:
            print(f"  Running {description}...")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
            if result.returncode != 0:
                print(f"  ❌ {description} failed")
                print(result.stdout.decode() if result.stdout else "")
                print(result.stderr.decode() if result.stderr else "")
                all_passed = False
            else:
                print(f"  ✅ {description} passed")
        
        return all_passed
    
    def check_dev_environment(self) -> bool:
        """Check development environment health."""
        print("🏥 Checking development environment health...")
        
        checks = [
            ("Python version", self._check_python_version),
            ("Dependencies", self._check_dependencies),
            ("Git hooks", self._check_git_hooks),
            ("Docker", self._check_docker)
        ]
        
        all_passed = True
        for name, check_func in checks:
            if check_func():
                print(f"  ✅ {name} - OK")
            else:
                print(f"  ❌ {name} - Issue detected")
                all_passed = False
        
        return all_passed
    
    def _check_python_version(self) -> bool:
        """Check Python version compatibility."""
        return sys.version_info >= (3, 9)
    
    def _check_dependencies(self) -> bool:
        """Check if all dependencies are installed."""
        try:
            result = subprocess.run(
                ["pip", "check"], 
                capture_output=True, 
                cwd=self.project_root
            )
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
    
    def _check_git_hooks(self) -> bool:
        """Check if pre-commit hooks are installed."""
        hooks_dir = self.project_root / ".git" / "hooks"
        return (hooks_dir / "pre-commit").exists()
    
    def _check_docker(self) -> bool:
        """Check Docker availability."""
        try:
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def generate_dev_report(self) -> dict:
        """Generate comprehensive development environment report."""
        print("📊 Generating development report...")
        
        start_time = time.time()
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment_health": self.check_dev_environment(),
            "code_quality": self.run_quality_checks(),
            "test_results": self.run_smart_tests(),
            "performance_check": self.check_performance_regression(),
            "execution_time": 0
        }
        
        report["execution_time"] = round(time.time() - start_time, 2)
        report["overall_status"] = all([
            report["environment_health"],
            report["code_quality"], 
            report["test_results"]
        ])
        
        # Save report
        report_file = self.project_root / f"dev_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        return report


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Advanced development automation")
    parser.add_argument(
        "command",
        choices=["test", "quality", "health", "perf", "report", "all"],
        help="Command to execute"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite (for test command)"
    )
    
    args = parser.parse_args()
    dev_auto = DevAutomation()
    
    success = True
    
    if args.command == "test":
        success = dev_auto.run_smart_tests(full_suite=args.full)
    elif args.command == "quality":
        success = dev_auto.run_quality_checks()
    elif args.command == "health":
        success = dev_auto.check_dev_environment()
    elif args.command == "perf":
        success = dev_auto.check_performance_regression()
    elif args.command == "report":
        report = dev_auto.generate_dev_report()
        success = report["overall_status"]
    elif args.command == "all":
        print("🚀 Running comprehensive development automation...")
        report = dev_auto.generate_dev_report()
        success = report["overall_status"]
    
    if success:
        print("\n✅ All checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()