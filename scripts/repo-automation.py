#!/usr/bin/env python3
"""
Repository Automation System

Comprehensive automation for repository maintenance, monitoring, and optimization.
Handles automated tasks like dependency updates, security monitoring, performance
tracking, and repository health checks.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RepositoryAutomation:
    """Main automation system for repository management."""
    
    def __init__(self, config_path: str = "config/automation.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.metrics_dir = self.repo_root / "metrics"
        self.reports_dir = self.repo_root / "reports"
        
        # Ensure directories exist
        self.metrics_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        (self.repo_root / "logs").mkdir(exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load automation configuration."""
        default_config = {
            "schedules": {
                "metrics_collection": "daily",
                "security_scan": "daily",
                "dependency_check": "weekly",
                "performance_test": "weekly",
                "cleanup": "monthly"
            },
            "thresholds": {
                "test_coverage": 80,
                "security_score": 90,
                "performance_regression": 10,
                "technical_debt_hours": 40
            },
            "notifications": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
                "email_recipients": [],
                "alert_channels": ["#alerts", "#development"]
            },
            "integrations": {
                "github_token": os.getenv("GITHUB_TOKEN"),
                "sonarqube_url": os.getenv("SONARQUBE_URL"),
                "grafana_url": os.getenv("GRAFANA_URL")
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                return default_config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
        
        return default_config
    
    def run_command(self, cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd or self.repo_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {cmd}")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running command '{cmd}': {e}")
            return 1, "", str(e)
    
    def collect_metrics(self) -> Dict:
        """Collect comprehensive repository metrics."""
        logger.info("Starting metrics collection...")
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "repository": self._get_repository_info(),
            "codebase": self._analyze_codebase(),
            "tests": self._analyze_tests(),
            "security": self._analyze_security(),
            "performance": self._analyze_performance(),
            "dependencies": self._analyze_dependencies(),
            "quality": self._analyze_code_quality()
        }
        
        # Save metrics
        metrics_file = self.metrics_dir / f"metrics-{datetime.now().strftime('%Y%m%d')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
        return metrics
    
    def _get_repository_info(self) -> Dict:
        """Get basic repository information."""
        try:
            # Get Git information
            exit_code, stdout, stderr = self.run_command("git rev-parse --show-toplevel")
            repo_root = stdout.strip() if exit_code == 0 else str(self.repo_root)
            
            exit_code, stdout, stderr = self.run_command("git rev-parse HEAD")
            current_commit = stdout.strip() if exit_code == 0 else "unknown"
            
            exit_code, stdout, stderr = self.run_command("git rev-parse --abbrev-ref HEAD")
            current_branch = stdout.strip() if exit_code == 0 else "unknown"
            
            exit_code, stdout, stderr = self.run_command("git config --get remote.origin.url")
            remote_url = stdout.strip() if exit_code == 0 else "unknown"
            
            return {
                "path": repo_root,
                "commit": current_commit,
                "branch": current_branch,
                "remote_url": remote_url,
                "last_analyzed": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting repository info: {e}")
            return {}
    
    def _analyze_codebase(self) -> Dict:
        """Analyze codebase structure and statistics."""
        try:
            # Count lines of code by file type
            file_counts = {}
            line_counts = {}
            
            for pattern, file_type in [
                ("*.py", "python"),
                ("*.yml", "yaml"),
                ("*.yaml", "yaml"),
                ("*.json", "json"),
                ("*.md", "markdown"),
                ("*.sh", "shell"),
                ("Dockerfile*", "dockerfile")
            ]:
                exit_code, stdout, stderr = self.run_command(
                    f"find . -name '{pattern}' -not -path './.*' -not -path './venv/*' -not -path './env/*' | wc -l"
                )
                if exit_code == 0:
                    file_counts[file_type] = int(stdout.strip())
                
                exit_code, stdout, stderr = self.run_command(
                    f"find . -name '{pattern}' -not -path './.*' -not -path './venv/*' -not -path './env/*' -exec wc -l {{}} + 2>/dev/null | tail -1 | awk '{{print $1}}'"
                )
                if exit_code == 0 and stdout.strip().isdigit():
                    line_counts[file_type] = int(stdout.strip())
            
            return {
                "files": file_counts,
                "lines": line_counts,
                "total_files": sum(file_counts.values()),
                "total_lines": sum(line_counts.values())
            }
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}")
            return {}
    
    def _analyze_tests(self) -> Dict:
        """Analyze test coverage and execution."""
        try:
            test_metrics = {}
            
            # Run tests with coverage
            exit_code, stdout, stderr = self.run_command(
                "python -m pytest --cov=src --cov-report=json:coverage.json --tb=short tests/"
            )
            
            if exit_code == 0 or exit_code == 1:  # 1 means some tests failed but coverage was generated
                # Parse coverage report
                coverage_file = self.repo_root / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    
                    test_metrics = {
                        "coverage_percentage": coverage_data["totals"]["percent_covered"],
                        "covered_lines": coverage_data["totals"]["covered_lines"],
                        "total_lines": coverage_data["totals"]["num_statements"],
                        "missing_lines": coverage_data["totals"]["missing_lines"]
                    }
            
            # Count test files
            exit_code, stdout, stderr = self.run_command(
                "find tests/ -name 'test_*.py' | wc -l"
            )
            if exit_code == 0:
                test_metrics["test_files"] = int(stdout.strip())
            
            # Count total tests
            exit_code, stdout, stderr = self.run_command(
                "grep -r 'def test_' tests/ | wc -l"
            )
            if exit_code == 0:
                test_metrics["total_tests"] = int(stdout.strip())
            
            return test_metrics
        except Exception as e:
            logger.error(f"Error analyzing tests: {e}")
            return {}
    
    def _analyze_security(self) -> Dict:
        """Analyze security posture."""
        try:
            security_metrics = {}
            
            # Run Bandit security scan
            exit_code, stdout, stderr = self.run_command(
                "bandit -r src/ -f json -o bandit-report.json"
            )
            
            bandit_file = self.repo_root / "bandit-report.json"
            if bandit_file.exists():
                with open(bandit_file, 'r') as f:
                    bandit_data = json.load(f)
                
                security_metrics["bandit_issues"] = len(bandit_data.get("results", []))
                security_metrics["bandit_score"] = bandit_data.get("metrics", {}).get("_totals", {}).get("CONFIDENCE.HIGH", 0)
            
            # Run Safety check
            exit_code, stdout, stderr = self.run_command(
                "safety check --json --output safety-report.json"
            )
            
            safety_file = self.repo_root / "safety-report.json"
            if safety_file.exists():
                with open(safety_file, 'r') as f:
                    safety_data = json.load(f)
                
                security_metrics["safety_vulnerabilities"] = len(safety_data)
            
            # Check for secrets
            exit_code, stdout, stderr = self.run_command(
                "git log --all --full-history -- '*.py' '*.yml' '*.yaml' '*.json' | grep -i -E '(password|secret|key|token)' | wc -l"
            )
            if exit_code == 0:
                security_metrics["potential_secrets"] = int(stdout.strip())
            
            return security_metrics
        except Exception as e:
            logger.error(f"Error analyzing security: {e}")
            return {}
    
    def _analyze_performance(self) -> Dict:
        """Analyze performance metrics."""
        try:
            performance_metrics = {}
            
            # Run performance tests if available
            if (self.repo_root / "tests" / "performance").exists():
                exit_code, stdout, stderr = self.run_command(
                    "python -m pytest tests/performance/ --benchmark-json=benchmark.json"
                )
                
                benchmark_file = self.repo_root / "benchmark.json"
                if benchmark_file.exists():
                    with open(benchmark_file, 'r') as f:
                        benchmark_data = json.load(f)
                    
                    benchmarks = benchmark_data.get("benchmarks", [])
                    if benchmarks:
                        performance_metrics["benchmark_count"] = len(benchmarks)
                        performance_metrics["average_time"] = sum(b["stats"]["mean"] for b in benchmarks) / len(benchmarks)
            
            # Check Docker image size if available
            exit_code, stdout, stderr = self.run_command(
                "docker images sql-synth-agentic-playground:latest --format 'table {{.Size}}' | tail -1"
            )
            if exit_code == 0 and stdout.strip():
                performance_metrics["docker_image_size"] = stdout.strip()
            
            return performance_metrics
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}
    
    def _analyze_dependencies(self) -> Dict:
        """Analyze dependency status."""
        try:
            dependency_metrics = {}
            
            # Check for outdated packages
            exit_code, stdout, stderr = self.run_command(
                "pip list --outdated --format=json"
            )
            if exit_code == 0:
                outdated_packages = json.loads(stdout)
                dependency_metrics["outdated_count"] = len(outdated_packages)
                dependency_metrics["outdated_packages"] = [pkg["name"] for pkg in outdated_packages]
            
            # Count total dependencies
            exit_code, stdout, stderr = self.run_command(
                "pip list --format=json"
            )
            if exit_code == 0:
                all_packages = json.loads(stdout)
                dependency_metrics["total_dependencies"] = len(all_packages)
            
            return dependency_metrics
        except Exception as e:
            logger.error(f"Error analyzing dependencies: {e}")
            return {}
    
    def _analyze_code_quality(self) -> Dict:
        """Analyze code quality metrics."""
        try:
            quality_metrics = {}
            
            # Run Ruff linting
            exit_code, stdout, stderr = self.run_command(
                "ruff check . --format=json --output-file=ruff-report.json"
            )
            
            ruff_file = self.repo_root / "ruff-report.json"
            if ruff_file.exists():
                with open(ruff_file, 'r') as f:
                    ruff_data = json.load(f)
                
                quality_metrics["ruff_issues"] = len(ruff_data)
            
            # Run MyPy type checking
            exit_code, stdout, stderr = self.run_command(
                "mypy src/ --json-report mypy-report"
            )
            
            mypy_file = self.repo_root / "mypy-report" / "index.txt"
            if mypy_file.exists():
                with open(mypy_file, 'r') as f:
                    mypy_content = f.read()
                quality_metrics["mypy_errors"] = mypy_content.count("error")
            
            return quality_metrics
        except Exception as e:
            logger.error(f"Error analyzing code quality: {e}")
            return {}
    
    def check_thresholds(self, metrics: Dict) -> List[str]:
        """Check if metrics meet defined thresholds."""
        violations = []
        thresholds = self.config.get("thresholds", {})
        
        # Check test coverage
        coverage = metrics.get("tests", {}).get("coverage_percentage", 0)
        if coverage < thresholds.get("test_coverage", 80):
            violations.append(f"Test coverage ({coverage}%) is below threshold ({thresholds['test_coverage']}%)")
        
        # Check security issues
        bandit_issues = metrics.get("security", {}).get("bandit_issues", 0)
        safety_vulns = metrics.get("security", {}).get("safety_vulnerabilities", 0)
        total_security_issues = bandit_issues + safety_vulns
        
        if total_security_issues > 0:
            violations.append(f"Found {total_security_issues} security issues (Bandit: {bandit_issues}, Safety: {safety_vulns})")
        
        # Check code quality
        ruff_issues = metrics.get("quality", {}).get("ruff_issues", 0)
        if ruff_issues > 10:  # Arbitrary threshold
            violations.append(f"Too many linting issues: {ruff_issues}")
        
        return violations
    
    def send_notification(self, message: str, channel: str = "#alerts") -> bool:
        """Send notification via Slack webhook."""
        webhook_url = self.config.get("notifications", {}).get("slack_webhook")
        if not webhook_url:
            logger.warning("No Slack webhook configured, skipping notification")
            return False
        
        try:
            payload = {
                "channel": channel,
                "username": "Repository Automation",
                "text": message,
                "icon_emoji": ":robot_face:"
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Notification sent to {channel}: {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    def generate_report(self, metrics: Dict) -> str:
        """Generate a comprehensive metrics report."""
        report_lines = [
            "# Repository Metrics Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "## Overview",
            f"- Repository: {metrics.get('repository', {}).get('remote_url', 'Unknown')}",
            f"- Branch: {metrics.get('repository', {}).get('branch', 'Unknown')}",
            f"- Commit: {metrics.get('repository', {}).get('commit', 'Unknown')[:8]}",
            "",
            "## Codebase Statistics",
            f"- Total Files: {metrics.get('codebase', {}).get('total_files', 'N/A')}",
            f"- Total Lines: {metrics.get('codebase', {}).get('total_lines', 'N/A')}",
            "",
            "## Test Coverage",
            f"- Coverage: {metrics.get('tests', {}).get('coverage_percentage', 'N/A')}%",
            f"- Total Tests: {metrics.get('tests', {}).get('total_tests', 'N/A')}",
            "",
            "## Security Status",
            f"- Bandit Issues: {metrics.get('security', {}).get('bandit_issues', 'N/A')}",
            f"- Safety Vulnerabilities: {metrics.get('security', {}).get('safety_vulnerabilities', 'N/A')}",
            "",
            "## Code Quality",
            f"- Ruff Issues: {metrics.get('quality', {}).get('ruff_issues', 'N/A')}",
            f"- MyPy Errors: {metrics.get('quality', {}).get('mypy_errors', 'N/A')}",
            "",
            "## Dependencies",
            f"- Total Dependencies: {metrics.get('dependencies', {}).get('total_dependencies', 'N/A')}",
            f"- Outdated Packages: {metrics.get('dependencies', {}).get('outdated_count', 'N/A')}",
        ]
        
        return "\n".join(report_lines)
    
    def run_automation(self, tasks: Optional[List[str]] = None) -> bool:
        """Run automation tasks."""
        logger.info("Starting repository automation...")
        
        available_tasks = [
            "metrics",
            "security",
            "dependencies",
            "performance",
            "cleanup"
        ]
        
        if tasks is None:
            tasks = available_tasks
        
        success = True
        
        try:
            if "metrics" in tasks:
                logger.info("Collecting metrics...")
                metrics = self.collect_metrics()
                
                # Check thresholds
                violations = self.check_thresholds(metrics)
                if violations:
                    violation_message = "Repository metrics violations detected:\n" + "\n".join(f"• {v}" for v in violations)
                    logger.warning(violation_message)
                    self.send_notification(violation_message, "#alerts")
                    success = False
                
                # Generate report
                report = self.generate_report(metrics)
                report_file = self.reports_dir / f"metrics-report-{datetime.now().strftime('%Y%m%d')}.md"
                with open(report_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report generated: {report_file}")
            
            if "cleanup" in tasks:
                logger.info("Running cleanup tasks...")
                self._cleanup_old_files()
            
            logger.info("Repository automation completed successfully")
            return success
            
        except Exception as e:
            logger.error(f"Automation failed: {e}")
            self.send_notification(f"Repository automation failed: {e}", "#alerts")
            return False
    
    def _cleanup_old_files(self):
        """Clean up old temporary files and reports."""
        # Clean up old metrics files (keep 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for file_path in self.metrics_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                logger.info(f"Deleted old metrics file: {file_path}")
        
        # Clean up old reports (keep 30 days)
        for file_path in self.reports_dir.glob("*.md"):
            if file_path.stat().st_mtime < cutoff_date.timestamp():
                file_path.unlink()
                logger.info(f"Deleted old report: {file_path}")
        
        # Clean up temporary files
        for pattern in ["*.json", "*.xml", "*.tmp"]:
            for file_path in self.repo_root.glob(pattern):
                if file_path.name in ["coverage.json", "bandit-report.json", "safety-report.json", "ruff-report.json", "benchmark.json"]:
                    file_path.unlink()
                    logger.info(f"Deleted temporary file: {file_path}")


def main():
    """Main entry point for the automation system."""
    parser = argparse.ArgumentParser(description="Repository Automation System")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["metrics", "security", "dependencies", "performance", "cleanup"],
        help="Specific tasks to run (default: all)"
    )
    parser.add_argument(
        "--config",
        default="config/automation.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        automation = RepositoryAutomation(config_path=args.config)
        success = automation.run_automation(tasks=args.tasks)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Automation system failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()