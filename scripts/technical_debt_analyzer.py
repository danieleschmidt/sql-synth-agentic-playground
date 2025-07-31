#!/usr/bin/env python3
"""
Technical debt analyzer for SQL Synth Agentic Playground.
Identifies, quantifies, and prioritizes technical debt for modernization efforts.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import re
from collections import defaultdict
from datetime import datetime
import radon.complexity as radon_cc
import radon.metrics as radon_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalDebtAnalyzer:
    """Comprehensive technical debt analysis and modernization recommendations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.debt_categories = {
            "code_complexity": {"weight": 0.3, "issues": []},
            "test_coverage": {"weight": 0.2, "issues": []},
            "documentation": {"weight": 0.15, "issues": []},
            "dependencies": {"weight": 0.15, "issues": []},
            "security": {"weight": 0.1, "issues": []},
            "maintainability": {"weight": 0.1, "issues": []}
        }
        
    def analyze_code_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity and identify complex functions/classes."""
        logger.info("Analyzing code complexity...")
        
        complexity_analysis = {
            "high_complexity_functions": [],
            "complex_files": [],
            "total_complexity": 0,
            "average_complexity": 0,
            "recommendations": []
        }
        
        python_files = list(self.project_root.glob("**/*.py"))
        python_files = [f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)]
        
        total_complexity = 0
        function_count = 0
        
        for py_file in python_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                
                # Calculate cyclomatic complexity
                try:
                    complexity_results = radon_cc.cc_visit(content)
                    
                    for result in complexity_results:
                        function_count += 1
                        total_complexity += result.complexity
                        
                        if result.complexity > 10:  # High complexity threshold
                            complexity_analysis["high_complexity_functions"].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "function": result.name,
                                "complexity": result.complexity,
                                "line": result.lineno,
                                "severity": "high" if result.complexity > 15 else "medium"
                            })
                    
                    # File-level complexity
                    file_complexity = sum(r.complexity for r in complexity_results)
                    if file_complexity > 50:  # High file complexity
                        complexity_analysis["complex_files"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "complexity": file_complexity,
                            "functions": len(complexity_results)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze complexity for {py_file}: {e}")
                    
            except Exception as e:
                logger.warning(f"Could not read {py_file}: {e}")
        
        complexity_analysis["total_complexity"] = total_complexity
        complexity_analysis["average_complexity"] = total_complexity / function_count if function_count > 0 else 0
        
        # Generate recommendations
        high_complexity_count = len(complexity_analysis["high_complexity_functions"])
        if high_complexity_count > 0:
            complexity_analysis["recommendations"].append({
                "type": "refactoring",
                "priority": "high" if high_complexity_count > 10 else "medium",
                "message": f"Found {high_complexity_count} high-complexity functions",
                "action": "Refactor complex functions into smaller, more manageable pieces"
            })
        
        if complexity_analysis["average_complexity"] > 8:
            complexity_analysis["recommendations"].append({
                "type": "architecture",
                "priority": "medium",
                "message": f"Average complexity is {complexity_analysis['average_complexity']:.1f}",
                "action": "Consider architectural improvements to reduce overall complexity"
            })
        
        return complexity_analysis
    
    def analyze_test_coverage_debt(self) -> Dict[str, Any]:
        """Analyze test coverage and identify testing gaps."""
        logger.info("Analyzing test coverage debt...")
        
        coverage_analysis = {
            "coverage_percentage": 0,
            "uncovered_files": [],
            "missing_test_files": [],
            "recommendations": []
        }
        
        try:
            # Run coverage analysis
            result = subprocess.run([
                "pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse coverage report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                coverage_analysis["coverage_percentage"] = coverage_data.get("totals", {}).get("percent_covered", 0)
                
                # Find uncovered files
                for filename, file_data in coverage_data.get("files", {}).items():
                    file_coverage = file_data.get("summary", {}).get("percent_covered", 100)
                    if file_coverage < 80:  # Low coverage threshold
                        coverage_analysis["uncovered_files"].append({
                            "file": filename,
                            "coverage": file_coverage,
                            "missing_lines": file_data.get("missing_lines", [])
                        })
            
            # Find source files without corresponding test files
            src_files = list((self.project_root / "src").glob("**/*.py"))
            for src_file in src_files:
                if src_file.name == "__init__.py":
                    continue
                
                # Check for corresponding test file
                relative_path = src_file.relative_to(self.project_root / "src")
                test_file = self.project_root / "tests" / f"test_{relative_path}"
                
                if not test_file.exists():
                    coverage_analysis["missing_test_files"].append(str(relative_path))
            
            # Generate recommendations
            if coverage_analysis["coverage_percentage"] < 80:
                coverage_analysis["recommendations"].append({
                    "type": "testing",
                    "priority": "high",
                    "message": f"Test coverage is {coverage_analysis['coverage_percentage']:.1f}%",
                    "action": "Increase test coverage to at least 80%"
                })
            
            if len(coverage_analysis["missing_test_files"]) > 0:
                coverage_analysis["recommendations"].append({
                    "type": "testing",
                    "priority": "medium",
                    "message": f"Found {len(coverage_analysis['missing_test_files'])} files without tests",
                    "action": "Create test files for uncovered modules"
                })
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            coverage_analysis["error"] = str(e)
        
        return coverage_analysis
    
    def analyze_documentation_debt(self) -> Dict[str, Any]:
        """Analyze documentation completeness and quality."""
        logger.info("Analyzing documentation debt...")
        
        doc_analysis = {
            "missing_docstrings": [],
            "outdated_docs": [],
            "documentation_score": 0,
            "recommendations": []
        }
        
        python_files = list(self.project_root.glob("**/*.py"))
        python_files = [f for f in python_files if "venv" not in str(f) and "__pycache__" not in str(f)]
        
        total_functions = 0
        documented_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                
                # Parse AST to find functions and classes
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        
                        # Check for docstring
                        if (node.body and isinstance(node.body[0], ast.Expr) and 
                            isinstance(node.body[0].value, ast.Constant) and 
                            isinstance(node.body[0].value.value, str)):
                            documented_functions += 1
                        else:
                            doc_analysis["missing_docstrings"].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "name": node.name,
                                "type": type(node).__name__,
                                "line": node.lineno
                            })
                            
            except Exception as e:
                logger.warning(f"Could not analyze {py_file}: {e}")
        
        # Calculate documentation score
        doc_analysis["documentation_score"] = (
            documented_functions / total_functions * 100 if total_functions > 0 else 0
        )
        
        # Check for README and other documentation
        required_docs = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md"]
        missing_docs = []
        
        for doc in required_docs:
            if not (self.project_root / doc).exists():
                missing_docs.append(doc)
        
        # Generate recommendations
        if doc_analysis["documentation_score"] < 70:
            doc_analysis["recommendations"].append({
                "type": "documentation",
                "priority": "medium",
                "message": f"Documentation coverage is {doc_analysis['documentation_score']:.1f}%",
                "action": "Add docstrings to functions and classes"
            })
        
        if missing_docs:
            doc_analysis["recommendations"].append({
                "type": "documentation",
                "priority": "low", 
                "message": f"Missing documentation files: {', '.join(missing_docs)}",
                "action": "Create missing documentation files"
            })
        
        return doc_analysis
    
    def analyze_dependency_debt(self) -> Dict[str, Any]:
        """Analyze dependency-related technical debt."""
        logger.info("Analyzing dependency debt...")
        
        dependency_analysis = {
            "outdated_dependencies": [],
            "security_vulnerabilities": [],
            "dependency_conflicts": [],
            "recommendations": []
        }
        
        try:
            # Check for outdated dependencies
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                dependency_analysis["outdated_dependencies"] = outdated
            
            # Check for security vulnerabilities
            safety_result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if safety_result.stdout:
                try:
                    vulnerabilities = json.loads(safety_result.stdout)
                    dependency_analysis["security_vulnerabilities"] = vulnerabilities
                except json.JSONDecodeError:
                    pass
            
            # Generate recommendations
            if len(dependency_analysis["outdated_dependencies"]) > 5:
                dependency_analysis["recommendations"].append({
                    "type": "maintenance",
                    "priority": "medium",
                    "message": f"Found {len(dependency_analysis['outdated_dependencies'])} outdated dependencies",
                    "action": "Update dependencies to latest compatible versions"
                })
            
            if len(dependency_analysis["security_vulnerabilities"]) > 0:
                dependency_analysis["recommendations"].append({
                    "type": "security",
                    "priority": "high",
                    "message": f"Found {len(dependency_analysis['security_vulnerabilities'])} security vulnerabilities",
                    "action": "Update vulnerable dependencies immediately"
                })
                
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            dependency_analysis["error"] = str(e)
        
        return dependency_analysis
    
    def calculate_debt_score(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall technical debt score and prioritize issues."""
        debt_score = 100  # Start with perfect score
        priority_issues = []
        
        for category, data in self.debt_categories.items():
            category_score = 100
            category_issues = 0
            
            if category == "code_complexity":
                analysis = analyses.get("complexity", {})
                high_complexity = len(analysis.get("high_complexity_functions", []))
                if high_complexity > 0:
                    category_score -= min(high_complexity * 5, 50)
                    category_issues = high_complexity
                    
            elif category == "test_coverage":
                analysis = analyses.get("coverage", {})
                coverage = analysis.get("coverage_percentage", 100)
                if coverage < 80:
                    category_score = coverage
                    category_issues = 100 - coverage
                    
            elif category == "documentation":
                analysis = analyses.get("documentation", {})
                doc_score = analysis.get("documentation_score", 100)
                category_score = doc_score
                category_issues = 100 - doc_score
                
            elif category == "dependencies":
                analysis = analyses.get("dependencies", {})
                vulnerabilities = len(analysis.get("security_vulnerabilities", []))
                outdated = len(analysis.get("outdated_dependencies", []))
                category_score -= (vulnerabilities * 10 + outdated * 2)
                category_issues = vulnerabilities + outdated
            
            # Apply category weight to overall score
            weighted_impact = (100 - category_score) * data["weight"]
            debt_score -= weighted_impact
            
            # Collect high-priority issues
            for analysis_type, analysis_data in analyses.items():
                for rec in analysis_data.get("recommendations", []):
                    if rec.get("priority") == "high":
                        priority_issues.append({
                            "category": category,
                            "type": rec["type"],
                            "message": rec["message"],
                            "action": rec["action"],
                            "priority": rec["priority"]
                        })
        
        return {
            "overall_score": max(0, debt_score),
            "priority_issues": priority_issues,
            "debt_level": (
                "low" if debt_score > 80 else
                "medium" if debt_score > 60 else
                "high"
            )
        }
    
    def generate_modernization_roadmap(self, analyses: Dict[str, Any], debt_score: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized modernization roadmap."""
        roadmap = []
        
        # High priority items (security, critical complexity)
        high_priority = [item for item in debt_score["priority_issues"] if item["priority"] == "high"]
        if high_priority:
            roadmap.append({
                "phase": "Immediate (1-2 weeks)",
                "priority": "critical",
                "items": high_priority,
                "effort": "high",
                "impact": "high"
            })
        
        # Medium priority items
        medium_priority = []
        for analysis in analyses.values():
            for rec in analysis.get("recommendations", []):
                if rec.get("priority") == "medium":
                    medium_priority.append(rec)
        
        if medium_priority:
            roadmap.append({
                "phase": "Short-term (1-2 months)",
                "priority": "important",
                "items": medium_priority,
                "effort": "medium",
                "impact": "medium"
            })
        
        # Low priority items
        low_priority = []
        for analysis in analyses.values():
            for rec in analysis.get("recommendations", []):
                if rec.get("priority") == "low":
                    low_priority.append(rec)
        
        if low_priority:
            roadmap.append({
                "phase": "Long-term (3-6 months)",
                "priority": "nice-to-have",
                "items": low_priority,
                "effort": "low",
                "impact": "low"
            })
        
        return roadmap
    
    def generate_debt_report(self) -> Dict[str, Any]:
        """Generate comprehensive technical debt report."""
        logger.info("Generating technical debt analysis report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": "SQL Synth Agentic Playground",
            "analyses": {}
        }
        
        # Run all debt analyses
        report["analyses"]["complexity"] = self.analyze_code_complexity()
        report["analyses"]["coverage"] = self.analyze_test_coverage_debt()
        report["analyses"]["documentation"] = self.analyze_documentation_debt()
        report["analyses"]["dependencies"] = self.analyze_dependency_debt()
        
        # Calculate overall debt score
        debt_score = self.calculate_debt_score(report["analyses"])
        report["debt_score"] = debt_score
        
        # Generate modernization roadmap
        roadmap = self.generate_modernization_roadmap(report["analyses"], debt_score)
        report["modernization_roadmap"] = roadmap
        
        # Save report
        report_path = self.project_root / "technical-debt-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Technical debt report saved to {report_path}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Technical debt analyzer")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--output", type=Path,
                       help="Output file for debt report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    analyzer = TechnicalDebtAnalyzer(args.project_root)
    report = analyzer.generate_debt_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Print summary
    debt_score = report["debt_score"]
    print(f"\n📊 Technical Debt Analysis Summary:")
    print(f"  Overall debt score: {debt_score['overall_score']:.1f}/100")
    print(f"  Debt level: {debt_score['debt_level']}")
    print(f"  Priority issues: {len(debt_score['priority_issues'])}")
    
    if debt_score["debt_level"] == "high":
        print("  ⚠️  High technical debt detected - immediate action recommended")
        sys.exit(1)
    elif len(debt_score["priority_issues"]) > 0:
        print(f"  🔧 {len(debt_score['priority_issues'])} high-priority issues need attention")

if __name__ == "__main__":
    main()