#!/usr/bin/env python3
"""Compliance and Audit Automation for SQL Synthesis Agentic Playground.

This script provides automated compliance checking and audit reporting capabilities
for various security and regulatory standards including SLSA, OWASP, and custom
organizational requirements.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

# External imports for compliance checking
try:
    import requests
    import boto3
    CLOUD_AUDIT_AVAILABLE = True
except ImportError:
    CLOUD_AUDIT_AVAILABLE = False
    print("Warning: Cloud audit features not available. Install with: pip install requests boto3")

@dataclass
class ComplianceResult:
    """Result of a compliance check."""
    check_name: str
    status: str  # 'pass', 'fail', 'warning', 'skip'
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    message: str
    details: Dict[str, Any]
    timestamp: str
    remediation: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class ComplianceAuditor:
    """Automated compliance and audit tool."""
    
    def __init__(self, output_dir: Path = Path("compliance_reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[ComplianceResult] = []
        self.logger = self._setup_logging()
        self.project_root = Path.cwd()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for compliance auditing."""
        logger = logging.getLogger("compliance_auditor")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "compliance_audit.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def add_result(self, result: ComplianceResult):
        """Add a compliance result."""
        self.results.append(result)
        self.logger.info(f"{result.check_name}: {result.status} - {result.message}")

    def run_file_security_checks(self) -> List[ComplianceResult]:
        """Run file-based security compliance checks."""
        checks = []
        
        # Check for sensitive files
        sensitive_patterns = [
            "*.key", "*.pem", "*.p12", "*.pfx", "*.cer",
            ".env", "*.env", "config.json", "secrets.json",
            "id_rsa", "id_dsa", "*.ovpn"
        ]
        
        sensitive_files = []
        for pattern in sensitive_patterns:
            files = list(self.project_root.rglob(pattern))
            sensitive_files.extend(files)
        
        if sensitive_files:
            checks.append(ComplianceResult(
                check_name="sensitive_files_check",
                status="warning",
                severity="high",
                message=f"Found {len(sensitive_files)} potentially sensitive files",
                details={"files": [str(f) for f in sensitive_files]},
                timestamp="",
                remediation="Review and remove or properly secure sensitive files"
            ))
        else:
            checks.append(ComplianceResult(
                check_name="sensitive_files_check",
                status="pass",
                severity="info",
                message="No sensitive files found in repository",
                details={},
                timestamp=""
            ))
        
        # Check .gitignore completeness
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            required_patterns = [
                "*.pyc", "__pycache__/", ".env", "*.log",
                ".DS_Store", "*.tmp", "*.swp", ".coverage"
            ]
            
            missing_patterns = [p for p in required_patterns if p not in gitignore_content]
            
            if missing_patterns:
                checks.append(ComplianceResult(
                    check_name="gitignore_completeness",
                    status="warning",
                    severity="medium",
                    message=f"Missing {len(missing_patterns)} recommended .gitignore patterns",
                    details={"missing_patterns": missing_patterns},
                    timestamp="",
                    remediation="Add missing patterns to .gitignore"
                ))
            else:
                checks.append(ComplianceResult(
                    check_name="gitignore_completeness",
                    status="pass",
                    severity="info",
                    message=".gitignore contains recommended patterns",
                    details={},
                    timestamp=""
                ))
        
        return checks

    def run_dependency_security_checks(self) -> List[ComplianceResult]:
        """Run dependency security compliance checks."""
        checks = []
        
        # Check for security scanning tools
        pyproject_path = self.project_root / "pyproject.toml"
        requirements_path = self.project_root / "requirements.txt"
        
        if pyproject_path.exists():
            try:
                import tomli
                with open(pyproject_path, "rb") as f:
                    pyproject = tomli.load(f)
                
                dev_deps = pyproject.get("project", {}).get("optional-dependencies", {}).get("dev", [])
                security_tools = ["bandit", "safety", "pip-audit"]
                found_tools = [tool for tool in security_tools if any(tool in dep for dep in dev_deps)]
                
                if len(found_tools) >= 2:
                    checks.append(ComplianceResult(
                        check_name="security_tools_check",
                        status="pass",
                        severity="info",
                        message=f"Found security tools: {', '.join(found_tools)}",
                        details={"tools": found_tools},
                        timestamp=""
                    ))
                else:
                    checks.append(ComplianceResult(
                        check_name="security_tools_check",
                        status="warning",
                        severity="medium",
                        message="Insufficient security scanning tools in dependencies",
                        details={"found_tools": found_tools, "recommended": security_tools},
                        timestamp="",
                        remediation="Add bandit, safety, and pip-audit to dev dependencies"
                    ))
                    
            except ImportError:
                checks.append(ComplianceResult(
                    check_name="security_tools_check",
                    status="skip",
                    severity="info",
                    message="Cannot parse pyproject.toml - tomli not available",
                    details={},
                    timestamp=""
                ))
        
        # Run pip-audit if available
        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout) if result.stdout else []
                vulnerabilities = audit_data if isinstance(audit_data, list) else []
                
                critical_vulns = [v for v in vulnerabilities if v.get("severity") == "high"]
                
                if critical_vulns:
                    checks.append(ComplianceResult(
                        check_name="dependency_vulnerabilities",
                        status="fail",
                        severity="critical",
                        message=f"Found {len(critical_vulns)} critical vulnerabilities",
                        details={"vulnerabilities": critical_vulns},
                        timestamp="",
                        remediation="Update vulnerable packages immediately"
                    ))
                elif vulnerabilities:
                    checks.append(ComplianceResult(
                        check_name="dependency_vulnerabilities",
                        status="warning",
                        severity="medium",
                        message=f"Found {len(vulnerabilities)} vulnerabilities",
                        details={"vulnerabilities": vulnerabilities},
                        timestamp="",
                        remediation="Review and update vulnerable packages"
                    ))
                else:
                    checks.append(ComplianceResult(
                        check_name="dependency_vulnerabilities",
                        status="pass",
                        severity="info",
                        message="No known vulnerabilities in dependencies",
                        details={},
                        timestamp=""
                    ))
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            checks.append(ComplianceResult(
                check_name="dependency_vulnerabilities",
                status="skip",
                severity="info",
                message="Could not run pip-audit - tool not available or failed",
                details={},
                timestamp=""
            ))
        
        return checks

    def run_code_quality_checks(self) -> List[ComplianceResult]:
        """Run code quality compliance checks."""
        checks = []
        
        # Check for pre-commit configuration
        precommit_path = self.project_root / ".pre-commit-config.yaml"
        if precommit_path.exists():
            try:
                with open(precommit_path) as f:
                    precommit_config = yaml.safe_load(f)
                
                security_hooks = []
                quality_hooks = []
                
                for repo in precommit_config.get("repos", []):
                    for hook in repo.get("hooks", []):
                        hook_id = hook.get("id", "")
                        if hook_id in ["bandit", "gitleaks", "safety"]:
                            security_hooks.append(hook_id)
                        elif hook_id in ["black", "ruff", "mypy", "isort"]:
                            quality_hooks.append(hook_id)
                
                if len(security_hooks) >= 2 and len(quality_hooks) >= 3:
                    checks.append(ComplianceResult(
                        check_name="precommit_hooks_check",
                        status="pass",
                        severity="info",
                        message="Comprehensive pre-commit hooks configured",
                        details={"security_hooks": security_hooks, "quality_hooks": quality_hooks},
                        timestamp=""
                    ))
                else:
                    checks.append(ComplianceResult(
                        check_name="precommit_hooks_check",
                        status="warning",
                        severity="medium",
                        message="Insufficient pre-commit hooks for security/quality",
                        details={"security_hooks": security_hooks, "quality_hooks": quality_hooks},
                        timestamp="",
                        remediation="Add more security and quality pre-commit hooks"
                    ))
                    
            except Exception as e:
                checks.append(ComplianceResult(
                    check_name="precommit_hooks_check",
                    status="fail",
                    severity="medium",
                    message=f"Failed to parse pre-commit config: {e}",
                    details={},
                    timestamp="",
                    remediation="Fix pre-commit configuration file"
                ))
        else:
            checks.append(ComplianceResult(
                check_name="precommit_hooks_check",
                status="fail",
                severity="high",
                message="No pre-commit configuration found",
                details={},
                timestamp="",
                remediation="Add .pre-commit-config.yaml with security and quality hooks"
            ))
        
        # Check test coverage requirements
        pytest_ini = self.project_root / "pytest.ini"
        pyproject_toml = self.project_root / "pyproject.toml"
        
        coverage_threshold = None
        
        if pytest_ini.exists():
            with open(pytest_ini) as f:
                content = f.read()
                if "--cov-fail-under=" in content:
                    # Extract coverage threshold
                    for line in content.split("\n"):
                        if "--cov-fail-under=" in line:
                            try:
                                threshold = int(line.split("--cov-fail-under=")[1].split()[0])
                                coverage_threshold = threshold
                                break
                            except (IndexError, ValueError):
                                pass
        
        if coverage_threshold is not None:
            if coverage_threshold >= 80:
                checks.append(ComplianceResult(
                    check_name="test_coverage_threshold",
                    status="pass",
                    severity="info",
                    message=f"Good test coverage threshold: {coverage_threshold}%",
                    details={"threshold": coverage_threshold},
                    timestamp=""
                ))
            else:
                checks.append(ComplianceResult(
                    check_name="test_coverage_threshold",
                    status="warning",
                    severity="medium",
                    message=f"Low test coverage threshold: {coverage_threshold}%",
                    details={"threshold": coverage_threshold},
                    timestamp="",
                    remediation="Increase test coverage threshold to at least 80%"
                ))
        else:
            checks.append(ComplianceResult(
                check_name="test_coverage_threshold",
                status="warning",
                severity="medium",
                message="No test coverage threshold configured",
                details={},
                timestamp="",
                remediation="Configure minimum test coverage threshold"
            ))
        
        return checks

    def run_documentation_checks(self) -> List[ComplianceResult]:
        """Run documentation compliance checks."""
        checks = []
        
        # Check for required documentation files
        required_docs = {
            "README.md": "Project overview and setup instructions",
            "CONTRIBUTING.md": "Contribution guidelines",
            "CODE_OF_CONDUCT.md": "Code of conduct",
            "SECURITY.md": "Security policy and vulnerability reporting",
            "LICENSE": "License information"
        }
        
        missing_docs = []
        existing_docs = []
        
        for doc_file, description in required_docs.items():
            if not (self.project_root / doc_file).exists():
                missing_docs.append({"file": doc_file, "purpose": description})
            else:
                existing_docs.append(doc_file)
        
        if not missing_docs:
            checks.append(ComplianceResult(
                check_name="required_documentation",
                status="pass",
                severity="info",
                message="All required documentation files present",
                details={"existing_docs": existing_docs},
                timestamp=""
            ))
        else:
            checks.append(ComplianceResult(
                check_name="required_documentation",
                status="warning",
                severity="medium",
                message=f"Missing {len(missing_docs)} required documentation files",
                details={"missing_docs": missing_docs, "existing_docs": existing_docs},
                timestamp="",
                remediation="Create missing documentation files"
            ))
        
        # Check for API documentation
        api_docs = [
            self.project_root / "docs" / "API.md",
            self.project_root / "api.md",
            self.project_root / "docs" / "api.md"
        ]
        
        has_api_docs = any(doc.exists() for doc in api_docs)
        
        if has_api_docs:
            checks.append(ComplianceResult(
                check_name="api_documentation",
                status="pass",
                severity="info",
                message="API documentation found",
                details={},
                timestamp=""
            ))
        else:
            checks.append(ComplianceResult(
                check_name="api_documentation",
                status="warning",
                severity="low",
                message="No API documentation found",
                details={},
                timestamp="",
                remediation="Create API documentation"
            ))
        
        return checks

    def run_infrastructure_checks(self) -> List[ComplianceResult]:
        """Run infrastructure compliance checks."""
        checks = []
        
        # Check for containerization
        dockerfile_paths = [
            self.project_root / "Dockerfile",
            self.project_root / "docker" / "Dockerfile"
        ]
        
        has_dockerfile = any(path.exists() for path in dockerfile_paths)
        
        if has_dockerfile:
            # Check Dockerfile security best practices
            dockerfile = next(path for path in dockerfile_paths if path.exists())
            content = dockerfile.read_text()
            
            security_issues = []
            
            if "USER root" in content or "USER 0" in content:
                security_issues.append("Running as root user")
            
            if "FROM" in content and "latest" in content:
                security_issues.append("Using 'latest' tag instead of specific version")
            
            if "RUN apt-get update" in content and "rm -rf /var/lib/apt/lists/*" not in content:
                security_issues.append("Not cleaning apt cache")
            
            if security_issues:
                checks.append(ComplianceResult(
                    check_name="dockerfile_security",
                    status="warning",
                    severity="medium",
                    message=f"Found {len(security_issues)} Dockerfile security issues",
                    details={"issues": security_issues},
                    timestamp="",
                    remediation="Fix Dockerfile security issues"
                ))
            else:
                checks.append(ComplianceResult(
                    check_name="dockerfile_security",
                    status="pass",
                    severity="info",
                    message="Dockerfile follows security best practices",
                    details={},
                    timestamp=""
                ))
        else:
            checks.append(ComplianceResult(
                check_name="dockerfile_security",
                status="skip",
                severity="info",
                message="No Dockerfile found",
                details={},
                timestamp=""
            ))
        
        # Check for Infrastructure as Code
        iac_paths = [
            self.project_root / "terraform",
            self.project_root / "infrastructure",
            self.project_root / ".github" / "workflows",
            self.project_root / "k8s",
            self.project_root / "kubernetes"
        ]
        
        iac_found = []
        for path in iac_paths:
            if path.exists() and any(path.iterdir()):
                iac_found.append(path.name)
        
        if iac_found:
            checks.append(ComplianceResult(
                check_name="infrastructure_as_code",
                status="pass",
                severity="info",
                message=f"Infrastructure as Code found: {', '.join(iac_found)}",
                details={"iac_types": iac_found},
                timestamp=""
            ))
        else:
            checks.append(ComplianceResult(
                check_name="infrastructure_as_code",
                status="warning",
                severity="medium",
                message="No Infrastructure as Code configuration found",
                details={},
                timestamp="",
                remediation="Add Terraform, Kubernetes, or CI/CD configurations"
            ))
        
        return checks

    def run_slsa_compliance_checks(self) -> List[ComplianceResult]:
        """Run SLSA (Supply-chain Levels for Software Artifacts) compliance checks."""
        checks = []
        
        # SLSA Level 1: Build - Scripted build process
        build_files = [
            self.project_root / "Dockerfile",
            self.project_root / "pyproject.toml",
            self.project_root / "setup.py",
            self.project_root / "Makefile",
            self.project_root / "build.sh"
        ]
        
        has_build_script = any(path.exists() for path in build_files)
        
        if has_build_script:
            checks.append(ComplianceResult(
                check_name="slsa_build_scripted",
                status="pass",
                severity="info",
                message="SLSA Level 1: Automated build process present",
                details={"build_files": [str(f) for f in build_files if f.exists()]},
                timestamp=""
            ))
        else:
            checks.append(ComplianceResult(
                check_name="slsa_build_scripted",
                status="fail",
                severity="high",
                message="SLSA Level 1: No automated build process found",
                details={},
                timestamp="",
                remediation="Add automated build configuration (Dockerfile, pyproject.toml, etc.)"
            ))
        
        # SLSA Level 2: Source - Version controlled
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            checks.append(ComplianceResult(
                check_name="slsa_source_version_controlled",
                status="pass",
                severity="info",
                message="SLSA Level 2: Source code is version controlled",
                details={},
                timestamp=""
            ))
        else:
            checks.append(ComplianceResult(
                check_name="slsa_source_version_controlled",
                status="fail",
                severity="critical",
                message="SLSA Level 2: Source code not version controlled",
                details={},
                timestamp="",
                remediation="Initialize Git repository"
            ))
        
        # SLSA Level 3: Provenance - Build service generates provenance
        github_workflows = self.project_root / ".github" / "workflows"
        has_ci_cd = github_workflows.exists() and any(github_workflows.glob("*.yml"))
        
        if has_ci_cd:
            checks.append(ComplianceResult(
                check_name="slsa_provenance_generation",
                status="pass",
                severity="info",
                message="SLSA Level 3: CI/CD workflows present for provenance generation",
                details={},
                timestamp=""
            ))
        else:
            checks.append(ComplianceResult(
                check_name="slsa_provenance_generation",
                status="warning",
                severity="medium",
                message="SLSA Level 3: No CI/CD workflows for provenance generation",
                details={},
                timestamp="",
                remediation="Add GitHub Actions workflows for automated builds"
            ))
        
        return checks

    def run_full_audit(self) -> Dict[str, Any]:
        """Run complete compliance audit."""
        self.logger.info("Starting comprehensive compliance audit")
        start_time = time.time()
        
        # Clear previous results
        self.results = []
        
        # Run all compliance checks
        check_suites = [
            ("File Security", self.run_file_security_checks),
            ("Dependency Security", self.run_dependency_security_checks),
            ("Code Quality", self.run_code_quality_checks),
            ("Documentation", self.run_documentation_checks),
            ("Infrastructure", self.run_infrastructure_checks),
            ("SLSA Compliance", self.run_slsa_compliance_checks),
        ]
        
        for suite_name, check_function in check_suites:
            self.logger.info(f"Running {suite_name} checks")
            try:
                suite_results = check_function()
                self.results.extend(suite_results)
            except Exception as e:
                self.logger.error(f"Error in {suite_name} checks: {e}")
                self.results.append(ComplianceResult(
                    check_name=f"{suite_name.lower().replace(' ', '_')}_error",
                    status="fail",
                    severity="high",
                    message=f"Error running {suite_name} checks: {e}",
                    details={"error": str(e)},
                    timestamp=""
                ))
        
        # Calculate summary statistics
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == "pass"])
        failed = len([r for r in self.results if r.status == "fail"])
        warnings = len([r for r in self.results if r.status == "warning"])
        skipped = len([r for r in self.results if r.status == "skip"])
        
        critical_issues = len([r for r in self.results if r.severity == "critical"])
        high_issues = len([r for r in self.results if r.severity == "high"])
        
        # Calculate compliance score
        compliance_score = (passed / total_checks) * 100 if total_checks > 0 else 0
        
        execution_time = time.time() - start_time
        
        summary = {
            "audit_timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "skipped": skipped,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "compliance_score": compliance_score,
            "compliance_grade": self._get_compliance_grade(compliance_score, critical_issues, high_issues),
            "results": [asdict(r) for r in self.results]
        }
        
        self.logger.info(f"Audit completed: {compliance_score:.1f}% compliance score")
        return summary

    def _get_compliance_grade(self, score: float, critical: int, high: int) -> str:
        """Calculate compliance grade based on score and issues."""
        if critical > 0:
            return "F"
        elif high > 2:
            return "D"
        elif score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def generate_compliance_report(self, audit_results: Dict[str, Any], format: str = "markdown") -> str:
        """Generate comprehensive compliance report."""
        if format.lower() == "markdown":
            return self._generate_markdown_report(audit_results)
        elif format.lower() == "json":
            return json.dumps(audit_results, indent=2)
        elif format.lower() == "html":
            return self._generate_html_report(audit_results)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_markdown_report(self, audit_results: Dict[str, Any]) -> str:
        """Generate Markdown compliance report."""
        lines = [
            "# Compliance Audit Report",
            f"**Generated:** {audit_results['audit_timestamp']}",
            f"**Execution Time:** {audit_results['execution_time_seconds']:.2f}s",
            "",
            "## Executive Summary",
            f"- **Compliance Score:** {audit_results['compliance_score']:.1f}% (Grade: {audit_results['compliance_grade']})",
            f"- **Total Checks:** {audit_results['total_checks']}",
            f"- **Passed:** {audit_results['passed']} ✅",
            f"- **Failed:** {audit_results['failed']} ❌",
            f"- **Warnings:** {audit_results['warnings']} ⚠️",
            f"- **Skipped:** {audit_results['skipped']} ⏭️",
            "",
            "## Risk Assessment",
            f"- **Critical Issues:** {audit_results['critical_issues']} 🔴",
            f"- **High Priority Issues:** {audit_results['high_issues']} 🟠",
            "",
        ]
        
        # Group results by status and severity
        failed_results = [r for r in audit_results['results'] if r['status'] == 'fail']
        warning_results = [r for r in audit_results['results'] if r['status'] == 'warning']
        
        if failed_results:
            lines.extend([
                "## Critical Issues (Action Required)",
                ""
            ])
            
            for result in sorted(failed_results, key=lambda x: x['severity'], reverse=True):
                lines.extend([
                    f"### ❌ {result['check_name']}",
                    f"**Severity:** {result['severity'].upper()}",
                    f"**Message:** {result['message']}",
                    ""
                ])
                
                if result.get('remediation'):
                    lines.extend([
                        f"**Remediation:** {result['remediation']}",
                        ""
                    ])
        
        if warning_results:
            lines.extend([
                "## Warnings (Recommended Actions)",
                ""
            ])
            
            for result in warning_results:
                lines.extend([
                    f"### ⚠️ {result['check_name']}",
                    f"**Message:** {result['message']}",
                ])
                
                if result.get('remediation'):
                    lines.extend([
                        f"**Remediation:** {result['remediation']}",
                    ])
                
                lines.append("")
        
        # Add detailed results
        lines.extend([
            "## Detailed Results",
            ""
        ])
        
        for result in audit_results['results']:
            status_emoji = {
                'pass': '✅',
                'fail': '❌',
                'warning': '⚠️',
                'skip': '⏭️'
            }.get(result['status'], '❓')
            
            lines.extend([
                f"### {status_emoji} {result['check_name']}",
                f"- **Status:** {result['status']}",
                f"- **Severity:** {result['severity']}",
                f"- **Message:** {result['message']}",
                ""
            ])
        
        lines.extend([
            "## Recommendations",
            "",
            "### Immediate Actions",
            "1. Address all critical and high-severity failed checks",
            "2. Review and implement recommended security measures",
            "3. Update documentation and compliance procedures",
            "",
            "### Continuous Improvement",
            "1. Schedule regular compliance audits",
            "2. Implement automated compliance checking in CI/CD",
            "3. Monitor compliance metrics and trends",
            "",
            "---",
            f"*Report generated by SQL Synthesis Agentic Playground Compliance Auditor*"
        ])
        
        return "\n".join(lines)

    def export_report(self, audit_results: Dict[str, Any], format: str = "markdown") -> str:
        """Export compliance report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "markdown":
            filename = self.output_dir / f"compliance_report_{timestamp}.md"
            content = self.generate_compliance_report(audit_results, "markdown")
        elif format.lower() == "json":
            filename = self.output_dir / f"compliance_report_{timestamp}.json"
            content = self.generate_compliance_report(audit_results, "json")
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        with open(filename, 'w') as f:
            f.write(content)
        
        self.logger.info(f"Compliance report exported to {filename}")
        return str(filename)

def main():
    """CLI interface for compliance auditor."""
    parser = argparse.ArgumentParser(description="Compliance and Audit Automation")
    parser.add_argument("--output-dir", type=Path, default=Path("compliance_reports"),
                       help="Output directory for reports")
    parser.add_argument("--format", choices=["markdown", "json", "html"], default="markdown",
                       help="Report format")
    parser.add_argument("--export", action="store_true",
                       help="Export report to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Run compliance audit
    auditor = ComplianceAuditor(args.output_dir)
    
    print("🔍 Running comprehensive compliance audit...")
    audit_results = auditor.run_full_audit()
    
    # Display summary
    print(f"\n📊 Compliance Audit Results:")
    print(f"   Score: {audit_results['compliance_score']:.1f}% (Grade: {audit_results['compliance_grade']})")
    print(f"   Checks: {audit_results['passed']}/{audit_results['total_checks']} passed")
    print(f"   Issues: {audit_results['critical_issues']} critical, {audit_results['high_issues']} high priority")
    
    # Export report if requested
    if args.export:
        report_file = auditor.export_report(audit_results, args.format)
        print(f"📄 Report exported to: {report_file}")
    else:
        # Print report to stdout
        report = auditor.generate_compliance_report(audit_results, args.format)
        print("\n" + report)

if __name__ == "__main__":
    main()