#!/usr/bin/env python3
"""Lightweight Quality Gates - Dependency-Free System Validation.

This module implements quality validation without external dependencies,
focusing on code quality, security, and architectural soundness.
"""

import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

class LightweightQualityGates:
    """Lightweight quality validation system."""

    def __init__(self):
        self.results = {
            "code_quality": {},
            "security_analysis": {},
            "architecture_analysis": {},
            "documentation_quality": {},
            "deployment_readiness": {}
        }

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation suitable for production systems."""
        print("🤖 TERRAGON LIGHTWEIGHT QUALITY GATES v4.0")
        print("=" * 55)
        
        # Validation 1: Code Quality Analysis
        print("\n📝 Code Quality Analysis")
        self._analyze_code_quality()
        
        # Validation 2: Security Analysis  
        print("\n🔒 Security Analysis")
        self._analyze_security()
        
        # Validation 3: Architecture Analysis
        print("\n🏗️  Architecture Analysis")
        self._analyze_architecture()
        
        # Validation 4: Documentation Quality
        print("\n📚 Documentation Quality")
        self._analyze_documentation()
        
        # Validation 5: Deployment Readiness
        print("\n🚀 Deployment Readiness")
        self._analyze_deployment_readiness()
        
        # Calculate overall score
        overall_score = self._calculate_overall_score()
        
        print(f"\n🎯 Overall Quality Score: {overall_score:.1%}")
        return self.results

    def _analyze_code_quality(self):
        """Analyze code quality metrics."""
        src_files = list(Path("src").rglob("*.py"))
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        complex_functions = 0
        documented_functions = 0
        syntax_errors = 0
        
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                lines = content.split('\n')
                total_lines += len(lines)
                
                # Analyze AST nodes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        
                        # Check complexity (simple heuristic: number of if/for/while/try)
                        complexity = sum(1 for child in ast.walk(node) 
                                       if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)))
                        if complexity > 5:  # Arbitrary threshold
                            complex_functions += 1
                        
                        # Check documentation
                        if (ast.get_docstring(node) or 
                            (len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and 
                             isinstance(node.body[0].value, ast.Constant))):
                            documented_functions += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        
            except SyntaxError:
                syntax_errors += 1
            except Exception:
                pass  # Skip problematic files
        
        # Calculate metrics
        documentation_rate = documented_functions / total_functions if total_functions > 0 else 0
        complexity_rate = complex_functions / total_functions if total_functions > 0 else 0
        
        self.results["code_quality"] = {
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "syntax_errors": syntax_errors,
            "documentation_rate": documentation_rate,
            "complexity_rate": complexity_rate,
            "quality_score": (
                (0 if syntax_errors > 0 else 0.3) +  # No syntax errors
                (documentation_rate * 0.4) +          # Documentation coverage
                ((1 - complexity_rate) * 0.3)         # Lower complexity is better
            )
        }
        
        print(f"  📊 Lines of code: {total_lines:,}")
        print(f"  🔧 Functions: {total_functions}")
        print(f"  🏗️  Classes: {total_classes}")  
        print(f"  📖 Documentation rate: {documentation_rate:.1%}")
        print(f"  🔀 Complex functions: {complex_functions} ({complexity_rate:.1%})")
        print(f"  ✅ Quality score: {self.results['code_quality']['quality_score']:.1%}")

    def _analyze_security(self):
        """Analyze security aspects."""
        src_files = list(Path("src").rglob("*.py"))
        
        # Security patterns to check
        security_patterns = {
            "sql_injection": {
                "patterns": [r'f".*SELECT.*{', r"f'.*SELECT.*{", r'\.format\(.*SELECT', r'%.*SELECT'],
                "severity": "HIGH",
                "description": "Potential SQL injection vulnerability"
            },
            "command_injection": {
                "patterns": [r'os\.system\(', r'subprocess\..*shell=True', r'eval\(', r'exec\('],
                "severity": "CRITICAL",
                "description": "Command injection risk"
            },
            "hardcoded_secrets": {
                "patterns": [r'password\s*=\s*["\'][^"\']+["\']', r'api_key\s*=\s*["\'][^"\']+["\']',
                           r'secret\s*=\s*["\'][^"\']+["\']', r'token\s*=\s*["\'][^"\']+["\']'],
                "severity": "HIGH", 
                "description": "Hardcoded secrets detected"
            },
            "insecure_random": {
                "patterns": [r'random\.random\(', r'random\.randint\('],
                "severity": "MEDIUM",
                "description": "Insecure randomness - use secrets module for crypto"
            }
        }
        
        security_issues = []
        
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for category, config in security_patterns.items():
                    for pattern in config["patterns"]:
                        for i, line in enumerate(lines):
                            if re.search(pattern, line, re.IGNORECASE):
                                security_issues.append({
                                    "file": str(file_path),
                                    "line": i + 1,
                                    "category": category,
                                    "severity": config["severity"],
                                    "description": config["description"],
                                    "code_snippet": line.strip()
                                })
                                
            except Exception:
                continue
        
        # Calculate security score
        critical_issues = sum(1 for issue in security_issues if issue["severity"] == "CRITICAL")
        high_issues = sum(1 for issue in security_issues if issue["severity"] == "HIGH") 
        medium_issues = sum(1 for issue in security_issues if issue["severity"] == "MEDIUM")
        
        # Security score penalizes issues by severity
        security_score = max(0, 1.0 - (critical_issues * 0.5) - (high_issues * 0.2) - (medium_issues * 0.1))
        
        self.results["security_analysis"] = {
            "total_issues": len(security_issues),
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "medium_issues": medium_issues,
            "security_score": security_score,
            "issues": security_issues[:10]  # Top 10 issues
        }
        
        print(f"  🚨 Security issues: {len(security_issues)} total")
        print(f"  🔴 Critical: {critical_issues}, High: {high_issues}, Medium: {medium_issues}")
        print(f"  🛡️  Security score: {security_score:.1%}")

    def _analyze_architecture(self):
        """Analyze system architecture."""
        src_files = list(Path("src").rglob("*.py"))
        
        # Analyze imports and dependencies
        imports = set()
        local_imports = set()
        circular_imports = []
        
        module_dependencies = {}
        
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                file_imports = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                            if alias.name.startswith('src.'):
                                local_imports.add(alias.name)
                                file_imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                            if node.module.startswith('src.'):
                                local_imports.add(node.module)
                                file_imports.add(node.module)
                
                module_name = str(file_path).replace('/', '.').replace('.py', '')
                module_dependencies[module_name] = file_imports
                
            except Exception:
                continue
        
        # Check for potential circular dependencies (simplified)
        for module, deps in module_dependencies.items():
            for dep in deps:
                if dep in module_dependencies and module in module_dependencies[dep]:
                    circular_imports.append((module, dep))
        
        # Analyze architectural patterns
        has_factory_pattern = any('Factory' in str(f) for f in src_files)
        has_singleton_pattern = any('singleton' in open(f).read().lower() for f in src_files if f.stat().st_size < 100000)
        has_observer_pattern = any('observer' in open(f).read().lower() for f in src_files if f.stat().st_size < 100000)
        
        # Calculate architecture score
        architecture_score = 0.0
        architecture_score += 0.3 if len(circular_imports) == 0 else 0.0  # No circular deps
        architecture_score += 0.2 if has_factory_pattern else 0.0          # Good patterns
        architecture_score += 0.2 if len(local_imports) > 5 else 0.0       # Modular design
        architecture_score += 0.3  # Base score for working system
        
        self.results["architecture_analysis"] = {
            "total_imports": len(imports),
            "local_imports": len(local_imports), 
            "circular_imports": len(circular_imports),
            "has_design_patterns": has_factory_pattern or has_singleton_pattern or has_observer_pattern,
            "architecture_score": architecture_score,
            "modularity": len(local_imports) / len(src_files) if src_files else 0
        }
        
        print(f"  📦 Total imports: {len(imports)}")
        print(f"  🔄 Local modules: {len(local_imports)}")
        print(f"  ⚠️  Circular imports: {len(circular_imports)}")
        print(f"  🎯 Architecture score: {architecture_score:.1%}")

    def _analyze_documentation(self):
        """Analyze documentation quality."""
        
        doc_files = {
            "README.md": Path("README.md"),
            "ARCHITECTURE.md": Path("ARCHITECTURE.md"), 
            "IMPLEMENTATION_STATUS.md": Path("IMPLEMENTATION_STATUS.md"),
            "CHANGELOG.md": Path("CHANGELOG.md"),
            "SECURITY.md": Path("SECURITY.md")
        }
        
        docs_found = 0
        total_doc_size = 0
        quality_indicators = 0
        
        for doc_name, doc_path in doc_files.items():
            if doc_path.exists():
                docs_found += 1
                size = doc_path.stat().st_size
                total_doc_size += size
                
                try:
                    content = doc_path.read_text(encoding='utf-8')
                    
                    # Quality indicators
                    if len(content) > 1000:  # Substantial content
                        quality_indicators += 1
                    if '##' in content or '###' in content:  # Structure
                        quality_indicators += 1
                    if 'installation' in content.lower() or 'setup' in content.lower():  # Setup info
                        quality_indicators += 1
                    if 'example' in content.lower() or '```' in content:  # Examples
                        quality_indicators += 1
                        
                except Exception:
                    pass
        
        # Check code documentation
        src_files = list(Path("src").rglob("*.py"))
        documented_modules = 0
        
        for file_path in src_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:  # Has docstrings
                        documented_modules += 1
            except Exception:
                continue
        
        code_doc_rate = documented_modules / len(src_files) if src_files else 0
        
        documentation_score = (
            (docs_found / len(doc_files)) * 0.4 +     # File coverage
            (quality_indicators / 20) * 0.3 +         # Quality indicators
            code_doc_rate * 0.3                       # Code documentation
        )
        
        self.results["documentation_quality"] = {
            "docs_found": docs_found,
            "total_doc_files": len(doc_files),
            "total_doc_size": total_doc_size,
            "quality_indicators": quality_indicators,
            "code_doc_rate": code_doc_rate,
            "documentation_score": min(1.0, documentation_score)
        }
        
        print(f"  📄 Documentation files: {docs_found}/{len(doc_files)}")
        print(f"  📏 Total doc size: {total_doc_size:,} bytes")
        print(f"  ✨ Quality indicators: {quality_indicators}")
        print(f"  📝 Code documentation: {code_doc_rate:.1%}")
        print(f"  📚 Documentation score: {documentation_score:.1%}")

    def _analyze_deployment_readiness(self):
        """Analyze deployment readiness."""
        
        deployment_files = {
            "Dockerfile": Path("Dockerfile"),
            "requirements.txt": Path("requirements.txt"),
            "pyproject.toml": Path("pyproject.toml"),
            "docker-compose.yml": Path("docker-compose.yml"),
            ".env.example": Path(".env.example"),
            "Makefile": Path("Makefile")
        }
        
        deployment_files_found = 0
        config_quality = 0
        
        for file_name, file_path in deployment_files.items():
            if file_path.exists():
                deployment_files_found += 1
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Quality checks specific to file types
                    if file_name == "Dockerfile":
                        if "FROM" in content and "CMD" in content:
                            config_quality += 1
                        if "COPY" in content or "ADD" in content:
                            config_quality += 1
                    elif file_name == "requirements.txt":
                        if len(content.strip().split('\n')) > 5:  # Has dependencies
                            config_quality += 1
                    elif file_name == "pyproject.toml":
                        if "[build-system]" in content and "[project]" in content:
                            config_quality += 1
                    
                except Exception:
                    pass
        
        # Check for CI/CD
        cicd_files = list(Path(".github/workflows").glob("*.yml")) if Path(".github/workflows").exists() else []
        has_cicd = len(cicd_files) > 0
        
        # Check for scripts
        scripts_dir = Path("scripts")
        has_scripts = scripts_dir.exists() and len(list(scripts_dir.glob("*.py"))) > 0
        
        # Check for tests
        tests_dir = Path("tests")
        has_tests = tests_dir.exists() and len(list(tests_dir.glob("*.py"))) > 0
        
        deployment_score = (
            (deployment_files_found / len(deployment_files)) * 0.4 +  # File coverage
            (config_quality / 10) * 0.3 +                            # Config quality  
            (0.1 if has_cicd else 0.0) +                             # CI/CD
            (0.1 if has_scripts else 0.0) +                          # Scripts
            (0.1 if has_tests else 0.0)                              # Tests
        )
        
        self.results["deployment_readiness"] = {
            "deployment_files_found": deployment_files_found,
            "total_deployment_files": len(deployment_files),
            "config_quality": config_quality,
            "has_cicd": has_cicd,
            "has_scripts": has_scripts,
            "has_tests": has_tests,
            "deployment_score": min(1.0, deployment_score)
        }
        
        print(f"  📦 Deployment files: {deployment_files_found}/{len(deployment_files)}")
        print(f"  ⚙️  Config quality: {config_quality}/10")
        print(f"  🔄 CI/CD: {'✅' if has_cicd else '❌'}")
        print(f"  📜 Scripts: {'✅' if has_scripts else '❌'}")
        print(f"  🧪 Tests: {'✅' if has_tests else '❌'}")
        print(f"  🚀 Deployment score: {deployment_score:.1%}")

    def _calculate_overall_score(self) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            "code_quality": 0.25,
            "security_analysis": 0.25,
            "architecture_analysis": 0.2,
            "documentation_quality": 0.15,
            "deployment_readiness": 0.15
        }
        
        total_score = 0.0
        
        print(f"\n📊 Quality Breakdown:")
        for category, weight in weights.items():
            score_key = f"{category.split('_')[0]}_score"
            score = self.results[category].get(score_key, 0)
            total_score += score * weight
            print(f"  {category.replace('_', ' ').title()}: {score:.1%} (weight: {weight:.0%})")
        
        self.results["overall_score"] = total_score
        
        # Quality classification
        if total_score >= 0.9:
            quality_level = "EXCELLENT"
            status = "🏆"
        elif total_score >= 0.8:
            quality_level = "GOOD" 
            status = "✅"
        elif total_score >= 0.7:
            quality_level = "ACCEPTABLE"
            status = "⚠️"
        else:
            quality_level = "NEEDS_IMPROVEMENT"
            status = "❌"
        
        self.results["quality_level"] = quality_level
        print(f"\n{status} Quality Level: {quality_level}")
        
        return total_score

    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report."""
        
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        overall_score = self.results.get("overall_score", 0)
        quality_level = self.results.get("quality_level", "Unknown")
        
        report = f"""# System Quality Assessment Report

**Generated**: {timestamp}  
**Overall Score**: {overall_score:.1%}  
**Quality Level**: {quality_level}

## Executive Summary

This autonomous quality assessment evaluates the SQL Synthesis Agentic Playground across five critical dimensions: code quality, security, architecture, documentation, and deployment readiness.

## Detailed Analysis

### 1. Code Quality Analysis
- **Lines of Code**: {self.results['code_quality'].get('total_lines', 0):,}
- **Functions**: {self.results['code_quality'].get('total_functions', 0)}
- **Classes**: {self.results['code_quality'].get('total_classes', 0)}
- **Documentation Rate**: {self.results['code_quality'].get('documentation_rate', 0):.1%}
- **Complex Functions**: {self.results['code_quality'].get('complexity_rate', 0):.1%}
- **Score**: {self.results['code_quality'].get('quality_score', 0):.1%}

### 2. Security Analysis
- **Total Issues**: {self.results['security_analysis'].get('total_issues', 0)}
- **Critical Issues**: {self.results['security_analysis'].get('critical_issues', 0)}
- **High Issues**: {self.results['security_analysis'].get('high_issues', 0)}
- **Medium Issues**: {self.results['security_analysis'].get('medium_issues', 0)}
- **Security Score**: {self.results['security_analysis'].get('security_score', 0):.1%}

### 3. Architecture Analysis
- **Total Imports**: {self.results['architecture_analysis'].get('total_imports', 0)}
- **Local Modules**: {self.results['architecture_analysis'].get('local_imports', 0)}
- **Circular Dependencies**: {self.results['architecture_analysis'].get('circular_imports', 0)}
- **Design Patterns**: {'Yes' if self.results['architecture_analysis'].get('has_design_patterns', False) else 'No'}
- **Architecture Score**: {self.results['architecture_analysis'].get('architecture_score', 0):.1%}

### 4. Documentation Quality
- **Documentation Files**: {self.results['documentation_quality'].get('docs_found', 0)}/{self.results['documentation_quality'].get('total_doc_files', 0)}
- **Total Size**: {self.results['documentation_quality'].get('total_doc_size', 0):,} bytes
- **Quality Indicators**: {self.results['documentation_quality'].get('quality_indicators', 0)}
- **Code Documentation**: {self.results['documentation_quality'].get('code_doc_rate', 0):.1%}
- **Documentation Score**: {self.results['documentation_quality'].get('documentation_score', 0):.1%}

### 5. Deployment Readiness
- **Deployment Files**: {self.results['deployment_readiness'].get('deployment_files_found', 0)}/{self.results['deployment_readiness'].get('total_deployment_files', 0)}
- **Config Quality**: {self.results['deployment_readiness'].get('config_quality', 0)}/10
- **CI/CD**: {'Yes' if self.results['deployment_readiness'].get('has_cicd', False) else 'No'}
- **Scripts**: {'Yes' if self.results['deployment_readiness'].get('has_scripts', False) else 'No'}
- **Tests**: {'Yes' if self.results['deployment_readiness'].get('has_tests', False) else 'No'}
- **Deployment Score**: {self.results['deployment_readiness'].get('deployment_score', 0):.1%}

## Recommendations

"""
        
        # Add recommendations based on scores
        if overall_score >= 0.9:
            report += "✅ **Excellent Quality**: System exceeds all quality standards and is ready for production deployment.\n\n"
        elif overall_score >= 0.8:
            report += "✅ **Good Quality**: System meets production standards with minor areas for improvement.\n\n"
        elif overall_score >= 0.7:
            report += "⚠️ **Acceptable Quality**: System is functional but requires attention to reach production standards.\n\n"
        else:
            report += "❌ **Needs Improvement**: System requires significant quality improvements before production deployment.\n\n"
        
        # Specific recommendations
        if self.results['security_analysis'].get('security_score', 1) < 0.8:
            report += "- **Security**: Address identified security issues, especially critical and high severity items.\n"
        
        if self.results['code_quality'].get('documentation_rate', 1) < 0.6:
            report += "- **Documentation**: Improve code documentation coverage for better maintainability.\n"
        
        if self.results['architecture_analysis'].get('circular_imports', 0) > 0:
            report += "- **Architecture**: Resolve circular import dependencies to improve modularity.\n"
        
        if self.results['deployment_readiness'].get('deployment_score', 1) < 0.7:
            report += "- **Deployment**: Enhance deployment configuration and automation.\n"
        
        report += f"""
## Quality Gates Status

{'🟢 PASSED' if overall_score >= 0.8 else '🔴 FAILED'} - Overall quality score: {overall_score:.1%}

---
*Generated by Terragon Autonomous Quality Gates v4.0*
"""
        
        return report


def main():
    """Run lightweight quality gates."""
    quality_gates = LightweightQualityGates()
    
    try:
        results = quality_gates.run_comprehensive_validation()
        
        print("\n" + "=" * 55)
        print("📋 GENERATING COMPREHENSIVE QUALITY REPORT...")
        
        report = quality_gates.generate_quality_report()
        
        # Save report
        with open("comprehensive_quality_report.md", "w") as f:
            f.write(report)
        
        print(f"📄 Quality report saved to: comprehensive_quality_report.md")
        
        overall_score = results.get("overall_score", 0)
        
        if overall_score >= 0.8:
            print("\n🎉 QUALITY GATES PASSED - System ready for production!")
            return 0
        else:
            print(f"\n⚠️  QUALITY GATES: {overall_score:.1%} - Review recommendations")
            return 1
            
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())