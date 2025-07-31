#!/usr/bin/env python3
"""
Advanced security scanning automation for SQL Synth Agentic Playground.
Performs comprehensive security analysis beyond basic static analysis.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SecurityScanner:
    """Advanced security scanner for comprehensive analysis."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        
    def run_gitleaks_scan(self) -> Dict[str, Any]:
        """Run gitleaks for secrets detection."""
        logger.info("Running gitleaks secrets scan...")
        try:
            result = subprocess.run([
                "gitleaks", "detect", 
                "--source", str(self.project_root),
                "--config", str(self.project_root / ".gitleaks.toml"),
                "--report-format", "json",
                "--report-path", str(self.project_root / "gitleaks-report.json"),
                "--verbose"
            ], capture_output=True, text=True, check=True)
            
            # Read results
            report_path = self.project_root / "gitleaks-report.json"
            if report_path.exists():
                with open(report_path) as f:
                    return json.load(f)
            return {"findings": [], "status": "clean"}
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Gitleaks found potential secrets: {e.stderr}")
            # Still try to read the report
            report_path = self.project_root / "gitleaks-report.json"
            if report_path.exists():
                with open(report_path) as f:
                    return json.load(f)
            return {"findings": [], "status": "error", "error": str(e)}
    
    def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run semgrep for advanced SAST analysis."""
        logger.info("Running semgrep security analysis...")
        try:
            result = subprocess.run([
                "semgrep", "--config=auto",
                "--json", "--output", str(self.project_root / "semgrep-report.json"),
                str(self.project_root / "src"),
                str(self.project_root / "app.py")
            ], capture_output=True, text=True, check=True)
            
            with open(self.project_root / "semgrep-report.json") as f:
                return json.load(f)
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Semgrep not available or failed: {e}")
            return {"results": [], "status": "skipped"}
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies for vulnerabilities."""
        logger.info("Analyzing dependencies for vulnerabilities...")
        try:
            # Run safety check
            result = subprocess.run([
                "safety", "check", 
                "--json", "--output", str(self.project_root / "safety-report.json"),
                "--policy-file", str(self.project_root / ".safety-policy.json")
            ], capture_output=True, text=True, check=True)
            
            with open(self.project_root / "safety-report.json") as f:
                safety_data = json.load(f)
            
            # Run pip-audit
            audit_result = subprocess.run([
                "pip-audit", "--format=json", 
                "--output", str(self.project_root / "pip-audit-report.json")
            ], capture_output=True, text=True, check=True)
            
            with open(self.project_root / "pip-audit-report.json") as f:
                audit_data = json.load(f)
                
            return {
                "safety": safety_data,
                "pip_audit": audit_data,
                "status": "completed"
            }
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Dependency analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def check_docker_security(self) -> Dict[str, Any]:
        """Check Docker configuration security."""
        logger.info("Analyzing Docker security configuration...")
        findings = []
        
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path) as f:
                dockerfile_content = f.read()
                
            # Check for security best practices
            if "USER root" in dockerfile_content:
                findings.append({
                    "rule": "docker-root-user",
                    "severity": "HIGH",
                    "message": "Container running as root user",
                    "file": "Dockerfile"
                })
                
            if "--no-cache-dir" not in dockerfile_content:
                findings.append({
                    "rule": "docker-pip-cache",
                    "severity": "LOW", 
                    "message": "pip install without --no-cache-dir",
                    "file": "Dockerfile"
                })
        
        return {"findings": findings, "status": "completed"}
    
    def analyze_sql_injection_patterns(self) -> Dict[str, Any]:
        """Analyze code for SQL injection vulnerabilities."""
        logger.info("Analyzing SQL injection patterns...")
        findings = []
        
        # Scan Python files for SQL injection patterns
        for py_file in self.project_root.glob("**/*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file) as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    # Check for string formatting in SQL
                    if any(pattern in line for pattern in [
                        ".format(", "f\"", "f'", "%s" % ", "% (", "+ "
                    ]) and any(sql_keyword in line.upper() for sql_keyword in [
                        "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE"
                    ]):
                        findings.append({
                            "rule": "potential-sql-injection",
                            "severity": "HIGH",
                            "message": "Potential SQL injection via string formatting",
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": i,
                            "code": line.strip()
                        })
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
                
        return {"findings": findings, "status": "completed"}
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        logger.info("Generating comprehensive security report...")
        
        report = {
            "timestamp": subprocess.run(["date", "-Iseconds"], 
                                      capture_output=True, text=True).stdout.strip(),
            "project": "SQL Synth Agentic Playground",
            "scans": {}
        }
        
        # Run all security scans
        report["scans"]["secrets"] = self.run_gitleaks_scan()
        report["scans"]["sast"] = self.run_semgrep_scan()
        report["scans"]["dependencies"] = self.analyze_dependencies()
        report["scans"]["docker"] = self.check_docker_security()
        report["scans"]["sql_injection"] = self.analyze_sql_injection_patterns()
        
        # Calculate summary
        total_findings = 0
        high_severity = 0
        
        for scan_name, scan_results in report["scans"].items():
            findings = scan_results.get("findings", [])
            if isinstance(findings, list):
                total_findings += len(findings)
                high_severity += len([f for f in findings 
                                    if f.get("severity") == "HIGH"])
        
        report["summary"] = {
            "total_findings": total_findings,
            "high_severity_findings": high_severity,
            "security_score": max(0, 100 - (high_severity * 10) - (total_findings * 2))
        }
        
        # Save comprehensive report
        report_path = self.project_root / "comprehensive-security-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Security report saved to {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Advanced security scanner")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--output", type=Path, 
                       help="Output file for security report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    scanner = SecurityScanner(args.project_root)
    report = scanner.generate_security_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n🔒 Security Scan Summary:")
    print(f"  Total findings: {report['summary']['total_findings']}")
    print(f"  High severity: {report['summary']['high_severity_findings']}")
    print(f"  Security score: {report['summary']['security_score']}/100")
    
    if report['summary']['high_severity_findings'] > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()