#!/usr/bin/env python3
"""
Security scanning script for the SQL Synthesis Agentic Playground.

This script performs comprehensive security scans including:
- Dependency vulnerability scanning
- SAST (Static Application Security Testing)
- Container security scanning
- Configuration security review
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil


@dataclass
class SecurityFinding:
    """Represents a security finding."""
    severity: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    confidence: str = "medium"
    remediation: Optional[str] = None


class SecurityScanner:
    """Comprehensive security scanner."""
    
    SEVERITY_LEVELS = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
        "INFO": 0
    }
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.findings: List[SecurityFinding] = []
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"
    
    def scan_dependencies(self) -> List[SecurityFinding]:
        """Scan dependencies for known vulnerabilities using safety."""
        findings = []
        
        print("🔍 Scanning dependencies for vulnerabilities...")
        
        # Install safety if not available
        exit_code, _, _ = self.run_command(["python", "-m", "pip", "show", "safety"])
        if exit_code != 0:
            print("  Installing safety...")
            self.run_command(["python", "-m", "pip", "install", "safety"])
        
        # Run safety check
        exit_code, stdout, stderr = self.run_command([
            "python", "-m", "safety", "check", "--json", "--short-report"
        ])
        
        if exit_code != 0 and "No known security vulnerabilities found" not in stderr:
            try:
                vulnerabilities = json.loads(stdout) if stdout else []
                
                for vuln in vulnerabilities:
                    finding = SecurityFinding(
                        severity="HIGH" if vuln.get("id") else "MEDIUM",
                        title=f"Vulnerable dependency: {vuln.get('package', 'Unknown')}",
                        description=f"Package {vuln.get('package')} version {vuln.get('installed_version')} "
                                  f"has a known vulnerability: {vuln.get('vulnerability', 'No description')}",
                        cve_id=vuln.get("id"),
                        remediation=f"Update to version {vuln.get('safe_version', 'latest')}"
                    )
                    findings.append(finding)
                    
            except json.JSONDecodeError:
                # Fallback: parse text output
                if "found" in stderr.lower() and "vulnerabilities" in stderr.lower():
                    finding = SecurityFinding(
                        severity="MEDIUM",
                        title="Dependency vulnerabilities detected",
                        description="Safety found potential vulnerabilities in dependencies",
                        remediation="Run 'safety check' for detailed information"
                    )
                    findings.append(finding)
        
        print(f"  Found {len(findings)} dependency vulnerabilities")
        return findings
    
    def scan_with_bandit(self) -> List[SecurityFinding]:
        """Scan Python code for security issues using bandit."""
        findings = []
        
        print("🔍 Running SAST scan with bandit...")
        
        # Install bandit if not available
        exit_code, _, _ = self.run_command(["python", "-m", "pip", "show", "bandit"])
        if exit_code != 0:
            print("  Installing bandit...")
            self.run_command(["python", "-m", "pip", "install", "bandit"])
        
        # Run bandit scan
        exit_code, stdout, stderr = self.run_command([
            "python", "-m", "bandit", "-r", "src/", "-f", "json", "-ll"
        ])
        
        if stdout:
            try:
                results = json.loads(stdout)
                
                for result in results.get("results", []):
                    severity_map = {
                        "HIGH": "HIGH",
                        "MEDIUM": "MEDIUM", 
                        "LOW": "LOW"
                    }
                    
                    finding = SecurityFinding(
                        severity=severity_map.get(result.get("issue_severity", "MEDIUM"), "MEDIUM"),
                        title=f"SAST: {result.get('test_name', 'Security Issue')}",
                        description=result.get("issue_text", "No description available"),
                        file_path=result.get("filename"),
                        line_number=result.get("line_number"),
                        cwe_id=result.get("test_id"),
                        confidence=result.get("issue_confidence", "medium").lower(),
                        remediation="Review code and apply secure coding practices"
                    )
                    findings.append(finding)
                    
            except json.JSONDecodeError:
                print(f"  Error parsing bandit output: {stderr}")
        
        print(f"  Found {len(findings)} SAST issues")
        return findings
    
    def scan_secrets(self) -> List[SecurityFinding]:
        """Scan for exposed secrets and credentials."""
        findings = []
        
        print("🔍 Scanning for exposed secrets...")
        
        # Common secret patterns
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']{3,}["\']', "Hardcoded password"),
            (r'api[_-]?key\s*[=:]\s*["\'][^"\']{10,}["\']', "API key"),
            (r'secret[_-]?key\s*[=:]\s*["\'][^"\']{10,}["\']', "Secret key"),
            (r'aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*["\'][^"\']{16,}["\']', "AWS Access Key"),
            (r'aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\'][^"\']{30,}["\']', "AWS Secret Key"),
            (r'postgres://[^:]+:[^@]+@[^/]+/\w+', "Database connection string"),
            (r'mysql://[^:]+:[^@]+@[^/]+/\w+', "Database connection string"),
            (r'mongodb://[^:]+:[^@]+@[^/]+/\w+', "Database connection string"),
        ]
        
        import re
        
        # Scan Python files
        for py_file in self.project_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            finding = SecurityFinding(
                                severity="CRITICAL",
                                title=f"Exposed {secret_type}",
                                description=f"Potential {secret_type.lower()} found in source code",
                                file_path=str(py_file.relative_to(self.project_path)),
                                line_number=line_num,
                                remediation="Move secrets to environment variables or secure vault"
                            )
                            findings.append(finding)
                            
            except (UnicodeDecodeError, IOError):
                continue
        
        # Check for .env files in git
        exit_code, stdout, stderr = self.run_command(["git", "ls-files", "*.env*"])
        if exit_code == 0 and stdout.strip():
            for env_file in stdout.strip().split('\n'):
                finding = SecurityFinding(
                    severity="HIGH",
                    title="Environment file in version control",
                    description=f"Environment file {env_file} is tracked by git",
                    file_path=env_file,
                    remediation="Add environment files to .gitignore and remove from git history"
                )
                findings.append(finding)
        
        print(f"  Found {len(findings)} secret exposures")
        return findings
    
    def scan_docker_security(self) -> List[SecurityFinding]:
        """Scan Docker configuration for security issues."""
        findings = []
        
        print("🔍 Scanning Docker configuration...")
        
        dockerfile_path = self.project_path / "Dockerfile"
        if not dockerfile_path.exists():
            return findings
            
        try:
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
                lines = dockerfile_content.split('\n')
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Check for running as root
                if line.startswith('USER root') or (line.startswith('RUN') and 'sudo' in line):
                    finding = SecurityFinding(
                        severity="MEDIUM",
                        title="Container running as root",
                        description="Container may be running with elevated privileges",
                        file_path="Dockerfile",
                        line_number=line_num,
                        remediation="Create and use a non-root user"
                    )
                    findings.append(finding)
                
                # Check for exposed secrets
                if any(keyword in line.lower() for keyword in ['password', 'secret', 'key']):
                    if '=' in line and any(quote in line for quote in ['"', "'"]):
                        finding = SecurityFinding(
                            severity="HIGH",
                            title="Potential secret in Dockerfile",
                            description="Dockerfile may contain hardcoded secrets",
                            file_path="Dockerfile",
                            line_number=line_num,
                            remediation="Use build arguments or multi-stage builds"
                        )
                        findings.append(finding)
                
                # Check for latest tag usage
                if 'FROM' in line and ':latest' in line:
                    finding = SecurityFinding(
                        severity="LOW",
                        title="Using latest tag",
                        description="Using 'latest' tag can lead to unpredictable builds",
                        file_path="Dockerfile",
                        line_number=line_num,
                        remediation="Use specific version tags"
                    )
                    findings.append(finding)
                    
        except IOError:
            pass
            
        print(f"  Found {len(findings)} Docker security issues")
        return findings
    
    def scan_configuration(self) -> List[SecurityFinding]:
        """Scan configuration files for security issues."""
        findings = []
        
        print("🔍 Scanning configuration files...")
        
        # Check for debug mode in production configs
        config_files = list(self.project_path.rglob("*.yml")) + \
                      list(self.project_path.rglob("*.yaml")) + \
                      list(self.project_path.rglob("*.json")) + \
                      list(self.project_path.rglob("*.toml"))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                # Check for debug mode
                if 'debug: true' in content or 'debug=true' in content:
                    finding = SecurityFinding(
                        severity="MEDIUM",
                        title="Debug mode enabled",
                        description="Debug mode may expose sensitive information",
                        file_path=str(config_file.relative_to(self.project_path)),
                        remediation="Disable debug mode in production"
                    )
                    findings.append(finding)
                
                # Check for insecure protocols
                if any(protocol in content for protocol in ['http://', 'ftp://', 'telnet://']):
                    finding = SecurityFinding(
                        severity="MEDIUM",
                        title="Insecure protocol usage",
                        description="Configuration uses insecure protocols",
                        file_path=str(config_file.relative_to(self.project_path)),
                        remediation="Use secure protocols (HTTPS, SFTP, SSH)"
                    )
                    findings.append(finding)
                    
            except (UnicodeDecodeError, IOError):
                continue
        
        print(f"  Found {len(findings)} configuration issues")
        return findings
    
    def generate_report(self, output_format: str = "json") -> str:
        """Generate security report in specified format."""
        # Sort findings by severity
        sorted_findings = sorted(
            self.findings,
            key=lambda x: self.SEVERITY_LEVELS.get(x.severity, 0),
            reverse=True
        )
        
        if output_format == "json":
            return json.dumps([asdict(f) for f in sorted_findings], indent=2)
        
        elif output_format == "markdown":
            report = "# Security Scan Report\n\n"
            report += f"**Scan Date**: {subprocess.check_output(['date'], text=True).strip()}\n"
            report += f"**Total Findings**: {len(sorted_findings)}\n\n"
            
            # Summary by severity
            severity_counts = {}
            for finding in sorted_findings:
                severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
            report += "## Summary\n\n"
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    report += f"- **{severity}**: {count}\n"
            
            report += "\n## Findings\n\n"
            
            for i, finding in enumerate(sorted_findings, 1):
                report += f"### {i}. {finding.title}\n\n"
                report += f"**Severity**: {finding.severity}\n\n"
                report += f"**Description**: {finding.description}\n\n"
                
                if finding.file_path:
                    location = finding.file_path
                    if finding.line_number:
                        location += f":{finding.line_number}"
                    report += f"**Location**: `{location}`\n\n"
                
                if finding.cve_id:
                    report += f"**CVE**: {finding.cve_id}\n\n"
                
                if finding.cwe_id:
                    report += f"**CWE**: {finding.cwe_id}\n\n"
                
                if finding.remediation:
                    report += f"**Remediation**: {finding.remediation}\n\n"
                
                report += "---\n\n"
            
            return report
        
        else:  # text
            report = "SECURITY SCAN REPORT\n"
            report += "=" * 50 + "\n\n"
            report += f"Total Findings: {len(sorted_findings)}\n\n"
            
            for i, finding in enumerate(sorted_findings, 1):
                report += f"{i}. [{finding.severity}] {finding.title}\n"
                report += f"   {finding.description}\n"
                if finding.file_path:
                    location = finding.file_path
                    if finding.line_number:
                        location += f":{finding.line_number}"
                    report += f"   Location: {location}\n"
                report += "\n"
            
            return report
    
    def run_full_scan(self) -> List[SecurityFinding]:
        """Run all security scans."""
        print("🚀 Starting comprehensive security scan...\n")
        
        # Run all scans
        self.findings.extend(self.scan_dependencies())
        self.findings.extend(self.scan_with_bandit())
        self.findings.extend(self.scan_secrets())
        self.findings.extend(self.scan_docker_security())
        self.findings.extend(self.scan_configuration())
        
        print(f"\n✅ Security scan completed. Found {len(self.findings)} total findings.")
        
        # Print summary
        severity_counts = {}
        for finding in self.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        if severity_counts:
            print("\nSummary by severity:")
            for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    print(f"  {severity}: {count}")
        
        return self.findings


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive security scanner")
    parser.add_argument("--path", default=".", help="Project path to scan")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "markdown", "text"], 
                       default="text", help="Output format")
    parser.add_argument("--fail-on", choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                       help="Fail (exit 1) if findings of this severity or higher are found")
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.path)
    findings = scanner.run_full_scan()
    
    # Generate report
    report = scanner.generate_report(args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    else:
        print("\n" + "=" * 50)
        print(report)
    
    # Check if we should fail based on severity
    if args.fail_on:
        fail_threshold = scanner.SEVERITY_LEVELS[args.fail_on]
        for finding in findings:
            if scanner.SEVERITY_LEVELS.get(finding.severity, 0) >= fail_threshold:
                print(f"\n❌ Failing due to {finding.severity} finding: {finding.title}")
                sys.exit(1)
    
    print("\n🔒 Security scan completed successfully!")


if __name__ == "__main__":
    main()