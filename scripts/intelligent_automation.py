#!/usr/bin/env python3
"""
Intelligent automation system for SQL Synth Agentic Playground.
Provides smart automation for development workflows, CI/CD optimization, and adaptive task management.
"""

import json
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
import yaml
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentAutomation:
    """Intelligent automation system for optimized development workflows."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / ".automation-config.yml"
        self.state_path = project_root / ".automation-state.json"
        self.config = self._load_config()
        self.state = self._load_state()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load automation configuration."""
        default_config = {
            "automation": {
                "smart_testing": {
                    "enabled": True,
                    "change_detection": True,
                    "parallel_execution": True,
                    "coverage_threshold": 80
                },
                "dependency_management": {
                    "auto_update": True,
                    "security_only": False,
                    "compatibility_check": True
                },
                "code_quality": {
                    "auto_format": True,
                    "pre_commit_optimization": True,
                    "complexity_monitoring": True
                },
                "performance_monitoring": {
                    "continuous_profiling": True,
                    "regression_detection": True,
                    "optimization_suggestions": True
                }
            },
            "notifications": {
                "slack_webhook": None,
                "email_alerts": [],
                "github_integration": True
            },
            "schedules": {
                "dependency_audit": "weekly",
                "performance_analysis": "daily",
                "security_scan": "daily"
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)
                # Merge with defaults
                return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        # Create default config file
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_state(self) -> Dict[str, Any]:
        """Load automation state."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        
        return {
            "last_runs": {},
            "performance_history": [],
            "dependency_updates": [],
            "optimization_applied": []
        }
    
    def _save_state(self):
        """Save automation state."""
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def detect_changed_files(self) -> List[str]:
        """Detect changed files since last commit."""
        try:
            result = subprocess.run([
                "git", "diff", "--name-only", "HEAD~1", "HEAD"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Fallback to staged files
            result = subprocess.run([
                "git", "diff", "--name-only", "--staged"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
            
        except Exception as e:
            logger.warning(f"Failed to detect changed files: {e}")
            return []
    
    def smart_test_selection(self) -> Dict[str, Any]:
        """Intelligently select tests to run based on changes."""
        logger.info("Running smart test selection...")
        
        changed_files = self.detect_changed_files()
        if not changed_files:
            logger.info("No changes detected, running full test suite")
            return {"strategy": "full", "files": [], "reason": "no_changes"}
        
        # Analyze changed files
        python_files = [f for f in changed_files if f.endswith('.py')]
        test_files = [f for f in python_files if 'test_' in f or f.startswith('tests/')]
        source_files = [f for f in python_files if not ('test_' in f or f.startswith('tests/'))]
        
        test_strategy = {
            "strategy": "smart",
            "changed_files": changed_files,
            "python_files": python_files,
            "test_files": test_files,
            "source_files": source_files
        }
        
        # Determine test strategy
        if any(f in changed_files for f in ['pyproject.toml', 'requirements.txt', 'Dockerfile']):
            test_strategy["strategy"] = "full"
            test_strategy["reason"] = "dependency_changes"
        elif len(source_files) > 10:
            test_strategy["strategy"] = "full" 
            test_strategy["reason"] = "extensive_changes"
        elif test_files:
            test_strategy["strategy"] = "targeted"
            test_strategy["target_tests"] = test_files
            test_strategy["reason"] = "test_changes"
        elif source_files:
            # Map source files to potential test files
            related_tests = []
            for src_file in source_files:
                # Simple mapping: src/module.py -> tests/test_module.py
                test_file = src_file.replace('src/', 'tests/test_').replace('.py', '.py')
                if Path(self.project_root / test_file).exists():
                    related_tests.append(test_file)
            
            if related_tests:
                test_strategy["strategy"] = "targeted"
                test_strategy["target_tests"] = related_tests
                test_strategy["reason"] = "related_tests"
            else:
                test_strategy["strategy"] = "unit"
                test_strategy["reason"] = "no_specific_tests"
        
        return test_strategy
    
    def optimize_test_execution(self, test_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize test execution based on strategy."""
        logger.info(f"Optimizing test execution with strategy: {test_strategy['strategy']}")
        
        base_cmd = ["pytest"]
        
        if test_strategy["strategy"] == "full":
            cmd = base_cmd + ["tests/", "-v", "--tb=short"]
        elif test_strategy["strategy"] == "targeted":
            target_tests = test_strategy.get("target_tests", [])
            cmd = base_cmd + target_tests + ["-v", "--tb=short"]
        elif test_strategy["strategy"] == "unit":
            cmd = base_cmd + ["tests/unit/", "-v", "--tb=short"]
        else:
            cmd = base_cmd + ["tests/", "-v", "--tb=short"]
        
        # Add performance optimizations
        if self.config["automation"]["smart_testing"]["parallel_execution"]:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            if cpu_count > 1:
                cmd.extend(["-n", str(min(cpu_count, 4))])  # Limit to 4 processes
        
        # Add coverage if configured
        coverage_threshold = self.config["automation"]["smart_testing"]["coverage_threshold"]
        if coverage_threshold > 0:
            cmd.extend([
                "--cov=src",
                f"--cov-fail-under={coverage_threshold}",
                "--cov-report=term-missing"
            ])
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            execution_time = time.time() - start_time
            
            return {
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                "command": " ".join(cmd),
                "error": str(e),
                "success": False
            }
    
    def automated_dependency_update(self) -> Dict[str, Any]:
        """Automated dependency analysis and updates."""
        logger.info("Running automated dependency analysis...")
        
        updates = {
            "security_updates": [],
            "compatibility_updates": [], 
            "optimization_suggestions": [],
            "status": "completed"
        }
        
        try:
            # Check for security vulnerabilities
            safety_result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if safety_result.returncode != 0 and safety_result.stdout:
                try:
                    safety_data = json.loads(safety_result.stdout)
                    for vuln in safety_data:
                        updates["security_updates"].append({
                            "package": vuln.get("package_name", "unknown"),
                            "current_version": vuln.get("installed_version", "unknown"),
                            "safe_version": vuln.get("safe_version", "unknown"),
                            "vulnerability": vuln.get("vulnerability", "unknown")
                        })
                except json.JSONDecodeError:
                    logger.warning("Failed to parse safety output")
            
            # Check for outdated packages
            outdated_result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if outdated_result.returncode == 0 and outdated_result.stdout:
                try:
                    outdated_data = json.loads(outdated_result.stdout)
                    for package in outdated_data:
                        updates["compatibility_updates"].append({
                            "package": package["name"],
                            "current_version": package["version"],
                            "latest_version": package["latest_version"],
                            "type": package.get("latest_filetype", "wheel")
                        })
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip list output")
            
            # Generate optimization suggestions
            if len(updates["security_updates"]) > 0:
                updates["optimization_suggestions"].append({
                    "type": "security",
                    "priority": "high",
                    "message": f"Found {len(updates['security_updates'])} security vulnerabilities",
                    "action": "Update vulnerable packages immediately"
                })
            
            if len(updates["compatibility_updates"]) > 10:
                updates["optimization_suggestions"].append({
                    "type": "maintenance",
                    "priority": "medium", 
                    "message": f"Found {len(updates['compatibility_updates'])} outdated packages",
                    "action": "Consider batch updating non-breaking changes"
                })
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            updates["status"] = "error"
            updates["error"] = str(e)
        
        return updates
    
    def intelligent_pre_commit_optimization(self) -> Dict[str, Any]:
        """Optimize pre-commit hooks based on changes."""
        logger.info("Optimizing pre-commit configuration...")
        
        changed_files = self.detect_changed_files()
        optimization = {
            "original_hooks": [],
            "optimized_hooks": [],
            "skipped_hooks": [],
            "reasoning": []
        }
        
        pre_commit_config = self.project_root / ".pre-commit-config.yaml"
        if not pre_commit_config.exists():
            return {"status": "skipped", "reason": "no_pre_commit_config"}
        
        try:
            with open(pre_commit_config) as f:
                config = yaml.safe_load(f)
            
            # Analyze which hooks are needed based on changes
            has_python_changes = any(f.endswith('.py') for f in changed_files)
            has_yaml_changes = any(f.endswith(('.yml', '.yaml')) for f in changed_files)
            has_json_changes = any(f.endswith('.json') for f in changed_files)
            has_markdown_changes = any(f.endswith('.md') for f in changed_files)
            
            for repo in config.get('repos', []):
                for hook in repo.get('hooks', []):
                    hook_id = hook['id']
                    optimization["original_hooks"].append(hook_id)
                    
                    # Determine if hook should run based on changes
                    should_run = True
                    
                    if hook_id in ['black', 'ruff', 'mypy', 'bandit'] and not has_python_changes:
                        should_run = False
                        optimization["reasoning"].append(f"Skipping {hook_id}: no Python changes")
                    elif hook_id == 'yamllint' and not has_yaml_changes:
                        should_run = False
                        optimization["reasoning"].append(f"Skipping {hook_id}: no YAML changes")
                    elif hook_id == 'check-json' and not has_json_changes:
                        should_run = False
                        optimization["reasoning"].append(f"Skipping {hook_id}: no JSON changes")
                    elif hook_id == 'prettier' and not (has_markdown_changes or has_yaml_changes or has_json_changes):
                        should_run = False
                        optimization["reasoning"].append(f"Skipping {hook_id}: no relevant changes")
                    
                    if should_run:
                        optimization["optimized_hooks"].append(hook_id)
                    else:
                        optimization["skipped_hooks"].append(hook_id)
            
            optimization["optimization_percentage"] = (
                len(optimization["skipped_hooks"]) / len(optimization["original_hooks"]) * 100
                if optimization["original_hooks"] else 0
            )
            
        except Exception as e:
            logger.error(f"Pre-commit optimization failed: {e}")
            optimization["status"] = "error"
            optimization["error"] = str(e)
        
        return optimization
    
    def generate_automation_report(self) -> Dict[str, Any]:
        """Generate comprehensive automation report."""
        logger.info("Generating automation report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": "SQL Synth Agentic Playground",
            "automation_analysis": {}
        }
        
        # Run automation analyses
        test_strategy = self.smart_test_selection()
        report["automation_analysis"]["smart_testing"] = test_strategy
        
        if test_strategy["strategy"] != "skip":
            test_results = self.optimize_test_execution(test_strategy)
            report["automation_analysis"]["test_execution"] = test_results
        
        report["automation_analysis"]["dependency_management"] = self.automated_dependency_update()
        report["automation_analysis"]["pre_commit_optimization"] = self.intelligent_pre_commit_optimization()
        
        # Calculate automation efficiency score
        efficiency_score = 100
        
        # Deduct points for issues
        dep_updates = report["automation_analysis"]["dependency_management"]
        if dep_updates.get("security_updates"):
            efficiency_score -= len(dep_updates["security_updates"]) * 10
        
        test_results = report["automation_analysis"].get("test_execution", {})
        if not test_results.get("success", True):
            efficiency_score -= 20
        
        report["efficiency_score"] = max(0, efficiency_score)
        
        # Update state
        self.state["last_runs"]["automation_report"] = report["timestamp"]
        self._save_state()
        
        # Save report
        report_path = self.project_root / "automation-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Automation report saved to {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Intelligent automation system")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--action", choices=[
        "smart-test", "dependency-update", "pre-commit-optimize", "full-report"
    ], default="full-report", help="Automation action to perform")
    parser.add_argument("--output", type=Path,
                       help="Output file for reports")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    automation = IntelligentAutomation(args.project_root)
    
    if args.action == "smart-test":
        strategy = automation.smart_test_selection()
        result = automation.optimize_test_execution(strategy)
        print(json.dumps(result, indent=2))
    elif args.action == "dependency-update":
        result = automation.automated_dependency_update()
        print(json.dumps(result, indent=2))
    elif args.action == "pre-commit-optimize":
        result = automation.intelligent_pre_commit_optimization()
        print(json.dumps(result, indent=2))
    else:
        # Full report
        report = automation.generate_automation_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n🤖 Automation Analysis Summary:")
        print(f"  Efficiency score: {report['efficiency_score']}/100")
        
        test_analysis = report["automation_analysis"]["smart_testing"]
        print(f"  Test strategy: {test_analysis['strategy']}")
        
        dep_analysis = report["automation_analysis"]["dependency_management"]
        security_issues = len(dep_analysis.get("security_updates", []))
        if security_issues > 0:
            print(f"  ⚠️  {security_issues} security vulnerabilities found")
            sys.exit(1)

if __name__ == "__main__":
    main()