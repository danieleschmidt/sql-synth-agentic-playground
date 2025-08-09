#!/usr/bin/env python3
"""Autonomous Quality Gates System

Comprehensive quality validation including security scanning,
code analysis, performance benchmarks, and deployment readiness.
"""

import ast
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root)
        self.results: Dict[str, Any] = {}
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        print("🚀 Starting Autonomous Quality Gates Validation...")
        start_time = time.time()
        
        # Security Analysis
        print("\n🛡️  Security Analysis...")
        self.results["security"] = self._security_scan()
        
        # Code Quality Analysis
        print("\n📊 Code Quality Analysis...")
        self.results["code_quality"] = self._code_quality_scan()
        
        # Architecture Validation
        print("\n🏗️  Architecture Validation...")
        self.results["architecture"] = self._architecture_validation()
        
        # Performance Analysis
        print("\n⚡ Performance Analysis...")
        self.results["performance"] = self._performance_analysis()
        
        # Documentation Completeness
        print("\n📚 Documentation Analysis...")
        self.results["documentation"] = self._documentation_analysis()
        
        # Dependency Security
        print("\n🔐 Dependency Security...")
        self.results["dependencies"] = self._dependency_security_scan()
        
        # Production Readiness
        print("\n🚀 Production Readiness...")
        self.results["production"] = self._production_readiness_check()
        
        # Calculate overall quality score
        self.results["overall"] = self._calculate_overall_score()
        self.results["execution_time"] = time.time() - start_time
        
        print(f"\n✅ Quality Gates completed in {self.results['execution_time']:.2f}s")
        return self.results
    
    def _security_scan(self) -> Dict[str, Any]:
        """Comprehensive security vulnerability scan."""
        issues = []
        security_score = 100
        
        # Scan for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded secret key detected"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token detected"),
            (r'(aws|gcp|azure)[_-]?(access|secret)[_-]?key', "Cloud credentials detected"),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, message in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append({
                            "type": "security",
                            "severity": "high",
                            "file": str(py_file.relative_to(self.project_root)),
                            "message": message,
                            "line_count": len(matches)
                        })
                        security_score -= 20
            except Exception as e:
                print(f"Warning: Could not scan {py_file}: {e}")
        
        # Check for SQL injection vulnerabilities
        sql_injection_patterns = [
            (r'f["\'].*SELECT.*{.*}.*["\']', "Potential SQL injection via f-string"),
            (r'["\'].*SELECT.*["\'].*\+.*', "Potential SQL injection via string concatenation"),
            (r'execute\(["\'].*%.*["\'].*%', "Potential SQL injection via % formatting"),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern, message in sql_injection_patterns:
                    if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                        issues.append({
                            "type": "security",
                            "severity": "critical",
                            "file": str(py_file.relative_to(self.project_root)),
                            "message": message
                        })
                        security_score -= 30
            except Exception:
                continue
        
        # Check for insecure random usage
        insecure_random_pattern = r'import random(?!\nimport secrets)|from random import'
        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                if re.search(insecure_random_pattern, content):
                    # Check if it's used for cryptographic purposes
                    crypto_usage = re.search(r'random\.(choice|randint|random)', content)
                    if crypto_usage and ('token' in content.lower() or 'password' in content.lower()):
                        issues.append({
                            "type": "security",
                            "severity": "medium",
                            "file": str(py_file.relative_to(self.project_root)),
                            "message": "Insecure random module used for security-sensitive operations"
                        })
                        security_score -= 15
            except Exception:
                continue
        
        return {
            "score": max(0, security_score),
            "issues": issues,
            "total_issues": len(issues),
            "critical_issues": len([i for i in issues if i["severity"] == "critical"]),
            "high_issues": len([i for i in issues if i["severity"] == "high"]),
            "medium_issues": len([i for i in issues if i["severity"] == "medium"]),
            "recommendations": self._get_security_recommendations(issues)
        }
    
    def _code_quality_scan(self) -> Dict[str, Any]:
        """Code quality analysis including complexity and best practices."""
        quality_score = 100
        issues = []
        stats = {
            "total_lines": 0,
            "total_files": 0,
            "avg_complexity": 0,
            "functions_analyzed": 0
        }
        
        complexity_scores = []
        
        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                stats["total_files"] += 1
                stats["total_lines"] += len(content.splitlines())
                
                # Parse AST for complexity analysis
                try:
                    tree = ast.parse(content)
                    complexity_analyzer = ComplexityAnalyzer()
                    complexity_analyzer.visit(tree)
                    
                    for func_name, complexity in complexity_analyzer.complexities.items():
                        complexity_scores.append(complexity)
                        stats["functions_analyzed"] += 1
                        
                        if complexity > 10:
                            issues.append({
                                "type": "complexity",
                                "severity": "high" if complexity > 15 else "medium",
                                "file": str(py_file.relative_to(self.project_root)),
                                "function": func_name,
                                "complexity": complexity,
                                "message": f"High cyclomatic complexity: {complexity}"
                            })
                            quality_score -= min(20, complexity - 10)
                            
                except SyntaxError as e:
                    issues.append({
                        "type": "syntax",
                        "severity": "critical",
                        "file": str(py_file.relative_to(self.project_root)),
                        "message": f"Syntax error: {e}"
                    })
                    quality_score -= 30
                
                # Check for code smells
                code_smells = self._detect_code_smells(content, py_file)
                issues.extend(code_smells)
                quality_score -= len(code_smells) * 5
                
            except Exception as e:
                print(f"Warning: Could not analyze {py_file}: {e}")
        
        if complexity_scores:
            stats["avg_complexity"] = sum(complexity_scores) / len(complexity_scores)
        
        return {
            "score": max(0, quality_score),
            "issues": issues,
            "statistics": stats,
            "recommendations": self._get_quality_recommendations(issues, stats)
        }
    
    def _detect_code_smells(self, content: str, file_path: Path) -> List[Dict]:
        """Detect common code smells."""
        smells = []
        lines = content.splitlines()
        
        # Long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                smells.append({
                    "type": "style",
                    "severity": "low",
                    "file": str(file_path.relative_to(self.project_root)),
                    "line": i,
                    "message": f"Line too long ({len(line)} characters)"
                })
        
        # TODO/FIXME/HACK comments
        for i, line in enumerate(lines, 1):
            if re.search(r'#.*\b(TODO|FIXME|HACK|BUG)\b', line, re.IGNORECASE):
                smells.append({
                    "type": "maintenance",
                    "severity": "low",
                    "file": str(file_path.relative_to(self.project_root)),
                    "line": i,
                    "message": "Maintenance comment found"
                })
        
        # Excessive nesting
        for i, line in enumerate(lines, 1):
            indent_level = (len(line) - len(line.lstrip())) // 4
            if indent_level > 4:
                smells.append({
                    "type": "complexity",
                    "severity": "medium",
                    "file": str(file_path.relative_to(self.project_root)),
                    "line": i,
                    "message": f"Excessive nesting (level {indent_level})"
                })
        
        # Long functions (heuristic based on line count)
        in_function = False
        function_start = 0
        function_name = ""
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("def ") and ":" in stripped:
                if in_function and i - function_start > 50:
                    smells.append({
                        "type": "complexity",
                        "severity": "medium",
                        "file": str(file_path.relative_to(self.project_root)),
                        "function": function_name,
                        "message": f"Long function ({i - function_start} lines)"
                    })
                
                in_function = True
                function_start = i
                function_name = stripped.split("(")[0].replace("def ", "")
            
            elif stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                if in_function and i - function_start > 50:
                    smells.append({
                        "type": "complexity",
                        "severity": "medium",
                        "file": str(file_path.relative_to(self.project_root)),
                        "function": function_name,
                        "message": f"Long function ({i - function_start} lines)"
                    })
                in_function = False
        
        return smells
    
    def _architecture_validation(self) -> Dict[str, Any]:
        """Validate software architecture and design patterns."""
        architecture_score = 100
        issues = []
        
        # Check for proper module structure
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            issues.append({
                "type": "structure",
                "severity": "medium",
                "message": "Missing src/ directory structure"
            })
            architecture_score -= 15
        
        # Check for __init__.py files
        python_dirs = []
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            python_dirs.append(py_file.parent)
        
        python_dirs = list(set(python_dirs))
        missing_init = []
        
        for py_dir in python_dirs:
            if py_dir.name in ["scripts", "tests"]:
                continue
            init_file = py_dir / "__init__.py"
            if not init_file.exists() and any(py_dir.glob("*.py")):
                missing_init.append(str(py_dir.relative_to(self.project_root)))
        
        if missing_init:
            issues.append({
                "type": "structure",
                "severity": "low",
                "message": f"Missing __init__.py files in: {', '.join(missing_init)}"
            })
            architecture_score -= len(missing_init) * 3
        
        # Check for circular dependencies (simplified)
        import_graph = self._build_import_graph()
        circular_deps = self._detect_circular_dependencies(import_graph)
        
        for cycle in circular_deps:
            issues.append({
                "type": "dependency",
                "severity": "high",
                "message": f"Circular dependency detected: {' -> '.join(cycle)}"
            })
            architecture_score -= 25
        
        # Check for proper separation of concerns
        business_logic_in_ui = self._check_business_logic_separation()
        issues.extend(business_logic_in_ui)
        architecture_score -= len(business_logic_in_ui) * 10
        
        return {
            "score": max(0, architecture_score),
            "issues": issues,
            "import_graph_nodes": len(import_graph),
            "circular_dependencies": len(circular_deps),
            "recommendations": self._get_architecture_recommendations(issues)
        }
    
    def _build_import_graph(self) -> Dict[str, List[str]]:
        """Build import dependency graph."""
        import_graph = {}
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            
            module_name = str(py_file.relative_to(self.project_root)).replace("/", ".").replace(".py", "")
            imports = []
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Find relative imports within the project
                import_patterns = [
                    r"from\s+(\.[.\w]*)\s+import",  # from .module import
                    r"from\s+(src\.[.\w]*)\s+import",  # from src.module import
                    r"import\s+(src\.[.\w]*)",  # import src.module
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    imports.extend(matches)
                
                import_graph[module_name] = imports
                
            except Exception:
                continue
        
        return import_graph
    
    def _detect_circular_dependencies(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies using DFS."""
        def dfs(node: str, visited: set, rec_stack: set, path: List[str]) -> List[str]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    cycle = dfs(neighbor, visited, rec_stack, path)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            path.pop()
            return []
        
        cycles = []
        visited = set()
        
        for node in graph.keys():
            if node not in visited:
                cycle = dfs(node, visited, set(), [])
                if cycle:
                    cycles.append(cycle)
        
        return cycles
    
    def _check_business_logic_separation(self) -> List[Dict]:
        """Check for business logic mixed with UI code."""
        issues = []
        
        # Look for UI files that might contain business logic
        ui_files = list(self.project_root.glob("**/streamlit_ui.py")) + list(self.project_root.glob("**/app.py"))
        
        for ui_file in ui_files:
            try:
                content = ui_file.read_text(encoding='utf-8')
                
                # Look for business logic patterns in UI files
                business_patterns = [
                    r"SELECT\s+.*FROM",  # SQL queries
                    r"def\s+calculate_\w+",  # Calculation functions
                    r"def\s+process_\w+",  # Processing functions
                    r"class\s+\w*Manager\w*",  # Manager classes
                ]
                
                for pattern in business_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append({
                            "type": "separation",
                            "severity": "medium",
                            "file": str(ui_file.relative_to(self.project_root)),
                            "message": "Business logic detected in UI layer"
                        })
                        break
                        
            except Exception:
                continue
        
        return issues
    
    def _performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance characteristics and bottlenecks."""
        performance_score = 100
        issues = []
        
        # Check for performance anti-patterns
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Look for performance issues
                perf_issues = [
                    (r"for\s+\w+\s+in\s+range\(len\([^)]+\)\)", "Use enumerate() instead of range(len())"),
                    (r"\+=.*\[.*\]", "Consider using list comprehension or extend()"),
                    (r"time\.sleep\(\d+\)", "Long sleep detected - consider async alternatives"),
                    (r"\.join\(.*\)", "String concatenation in loop - consider using list and join()"),
                ]
                
                for pattern, message in perf_issues:
                    matches = re.findall(pattern, content)
                    if matches:
                        issues.append({
                            "type": "performance",
                            "severity": "medium",
                            "file": str(py_file.relative_to(self.project_root)),
                            "message": message,
                            "count": len(matches)
                        })
                        performance_score -= min(15, len(matches) * 5)
                
                # Check for database N+1 query patterns
                if "for" in content and ("query" in content or "execute" in content):
                    lines = content.splitlines()
                    in_loop = False
                    for i, line in enumerate(lines):
                        if re.search(r"for\s+\w+\s+in", line):
                            in_loop = True
                            loop_end = self._find_loop_end(lines, i)
                        elif in_loop and i < loop_end:
                            if re.search(r"\.(query|execute|fetch)", line):
                                issues.append({
                                    "type": "performance",
                                    "severity": "high",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": i + 1,
                                    "message": "Potential N+1 query problem"
                                })
                                performance_score -= 20
                                break
                
            except Exception:
                continue
        
        # Check for caching implementation
        caching_implemented = False
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if "@cache" in content or "cache" in content.lower():
                    caching_implemented = True
                    break
            except Exception:
                continue
        
        if not caching_implemented:
            issues.append({
                "type": "performance",
                "severity": "medium",
                "message": "No caching implementation detected"
            })
            performance_score -= 15
        
        return {
            "score": max(0, performance_score),
            "issues": issues,
            "caching_implemented": caching_implemented,
            "recommendations": self._get_performance_recommendations(issues)
        }
    
    def _find_loop_end(self, lines: List[str], loop_start: int) -> int:
        """Find the end of a loop block."""
        indent_level = len(lines[loop_start]) - len(lines[loop_start].lstrip())
        
        for i in range(loop_start + 1, len(lines)):
            line = lines[i]
            if line.strip() == "":
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level:
                return i
        
        return len(lines)
    
    def _documentation_analysis(self) -> Dict[str, Any]:
        """Analyze documentation completeness and quality."""
        doc_score = 100
        issues = []
        stats = {
            "total_functions": 0,
            "documented_functions": 0,
            "total_classes": 0,
            "documented_classes": 0,
            "total_modules": 0,
            "documented_modules": 0
        }
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                stats["total_modules"] += 1
                if ast.get_docstring(tree):
                    stats["documented_modules"] += 1
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stats["total_functions"] += 1
                        if ast.get_docstring(node):
                            stats["documented_functions"] += 1
                        else:
                            # Skip private functions and simple getters/setters
                            if not node.name.startswith("_") and len(node.body) > 2:
                                issues.append({
                                    "type": "documentation",
                                    "severity": "low",
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "function": node.name,
                                    "message": "Missing function docstring"
                                })
                    
                    elif isinstance(node, ast.ClassDef):
                        stats["total_classes"] += 1
                        if ast.get_docstring(node):
                            stats["documented_classes"] += 1
                        else:
                            issues.append({
                                "type": "documentation",
                                "severity": "medium",
                                "file": str(py_file.relative_to(self.project_root)),
                                "class": node.name,
                                "message": "Missing class docstring"
                            })
                            doc_score -= 10
                            
            except Exception:
                continue
        
        # Calculate documentation coverage
        func_coverage = (stats["documented_functions"] / max(1, stats["total_functions"])) * 100
        class_coverage = (stats["documented_classes"] / max(1, stats["total_classes"])) * 100
        module_coverage = (stats["documented_modules"] / max(1, stats["total_modules"])) * 100
        
        overall_coverage = (func_coverage + class_coverage + module_coverage) / 3
        
        # Adjust score based on coverage
        if overall_coverage < 50:
            doc_score -= 30
        elif overall_coverage < 70:
            doc_score -= 20
        elif overall_coverage < 85:
            doc_score -= 10
        
        # Check for README and other important documentation
        readme_files = list(self.project_root.glob("README*"))
        if not readme_files:
            issues.append({
                "type": "documentation",
                "severity": "medium",
                "message": "Missing README file"
            })
            doc_score -= 15
        
        return {
            "score": max(0, doc_score),
            "issues": issues,
            "statistics": stats,
            "coverage": {
                "functions": func_coverage,
                "classes": class_coverage,
                "modules": module_coverage,
                "overall": overall_coverage
            },
            "recommendations": self._get_documentation_recommendations(overall_coverage, issues)
        }
    
    def _dependency_security_scan(self) -> Dict[str, Any]:
        """Scan dependencies for known security vulnerabilities."""
        dep_score = 100
        issues = []
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        vulnerable_packages = [
            "django<3.2.0",  # Example vulnerable versions
            "requests<2.20.0",
            "sqlalchemy<1.3.0",
            "flask<1.0.0",
        ]
        
        if req_file.exists():
            try:
                requirements = req_file.read_text(encoding='utf-8')
                for vulnerable in vulnerable_packages:
                    pkg_name = vulnerable.split("<")[0]
                    if pkg_name in requirements.lower():
                        # Check if version constraint exists
                        if "<" not in requirements or pkg_name.lower() in requirements.lower():
                            issues.append({
                                "type": "dependency",
                                "severity": "high",
                                "package": pkg_name,
                                "message": f"Potentially vulnerable package version: {vulnerable}"
                            })
                            dep_score -= 25
            except Exception:
                pass
        
        # Check for unpinned dependencies
        if req_file.exists():
            try:
                requirements = req_file.read_text(encoding='utf-8')
                lines = requirements.splitlines()
                unpinned = []
                
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if ">=" not in line and "==" not in line and "~=" not in line:
                            unpinned.append(line)
                
                if unpinned:
                    issues.append({
                        "type": "dependency",
                        "severity": "medium",
                        "message": f"Unpinned dependencies detected: {', '.join(unpinned)}"
                    })
                    dep_score -= len(unpinned) * 5
            except Exception:
                pass
        
        return {
            "score": max(0, dep_score),
            "issues": issues,
            "recommendations": self._get_dependency_recommendations(issues)
        }
    
    def _production_readiness_check(self) -> Dict[str, Any]:
        """Check production readiness criteria."""
        readiness_score = 100
        issues = []
        
        # Check for environment configuration
        env_files = list(self.project_root.glob(".env*"))
        if not env_files:
            issues.append({
                "type": "configuration",
                "severity": "medium",
                "message": "No environment configuration files found"
            })
            readiness_score -= 15
        
        # Check for proper logging configuration
        logging_configured = False
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if "logging.basicConfig" in content or "getLogger" in content:
                    logging_configured = True
                    break
            except Exception:
                continue
        
        if not logging_configured:
            issues.append({
                "type": "observability",
                "severity": "medium",
                "message": "No logging configuration detected"
            })
            readiness_score -= 15
        
        # Check for health check endpoints
        health_check_found = False
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if "health" in content.lower() and ("endpoint" in content.lower() or "route" in content.lower()):
                    health_check_found = True
                    break
            except Exception:
                continue
        
        if not health_check_found:
            issues.append({
                "type": "monitoring",
                "severity": "low",
                "message": "No health check endpoint detected"
            })
            readiness_score -= 10
        
        # Check for Docker configuration
        docker_files = list(self.project_root.glob("Dockerfile*")) + list(self.project_root.glob("docker-compose*"))
        if not docker_files:
            issues.append({
                "type": "deployment",
                "severity": "low",
                "message": "No Docker configuration found"
            })
            readiness_score -= 10
        
        # Check for CI/CD configuration
        ci_files = (
            list(self.project_root.glob(".github/workflows/*")) +
            list(self.project_root.glob(".gitlab-ci.yml")) +
            list(self.project_root.glob("azure-pipelines.yml"))
        )
        
        if not ci_files:
            issues.append({
                "type": "automation",
                "severity": "medium",
                "message": "No CI/CD configuration found"
            })
            readiness_score -= 15
        
        return {
            "score": max(0, readiness_score),
            "issues": issues,
            "checks": {
                "environment_config": len(env_files) > 0,
                "logging_configured": logging_configured,
                "health_check": health_check_found,
                "docker_config": len(docker_files) > 0,
                "ci_cd_config": len(ci_files) > 0
            },
            "recommendations": self._get_production_recommendations(issues)
        }
    
    def _calculate_overall_score(self) -> Dict[str, Any]:
        """Calculate overall quality score and determine pass/fail."""
        weights = {
            "security": 0.25,
            "code_quality": 0.20,
            "architecture": 0.15,
            "performance": 0.15,
            "documentation": 0.10,
            "dependencies": 0.10,
            "production": 0.05
        }
        
        weighted_score = 0
        total_issues = 0
        critical_issues = 0
        
        for category, weight in weights.items():
            if category in self.results:
                score = self.results[category].get("score", 0)
                weighted_score += score * weight
                
                issues = self.results[category].get("issues", [])
                total_issues += len(issues)
                critical_issues += len([i for i in issues if i.get("severity") == "critical"])
        
        # Quality gates criteria
        gates = {
            "minimum_score": weighted_score >= 75,
            "no_critical_issues": critical_issues == 0,
            "security_threshold": self.results.get("security", {}).get("score", 0) >= 80,
            "code_quality_threshold": self.results.get("code_quality", {}).get("score", 0) >= 70
        }
        
        passed_gates = sum(gates.values())
        total_gates = len(gates)
        
        overall_status = "PASS" if passed_gates == total_gates else "FAIL"
        
        return {
            "weighted_score": weighted_score,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "quality_gates": gates,
            "gates_passed": f"{passed_gates}/{total_gates}",
            "overall_status": overall_status,
            "grade": self._calculate_grade(weighted_score),
            "deployment_ready": overall_status == "PASS" and critical_issues == 0
        }
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade based on score."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        else:
            return "F"
    
    def _get_security_recommendations(self, issues: List[Dict]) -> List[str]:
        """Get security improvement recommendations."""
        recommendations = []
        
        if any(i["severity"] == "critical" for i in issues):
            recommendations.append("🚨 Address critical security vulnerabilities immediately")
        
        if any("hardcoded" in i["message"].lower() for i in issues):
            recommendations.append("🔐 Use environment variables for sensitive configuration")
            recommendations.append("🔐 Implement secrets management system")
        
        if any("sql injection" in i["message"].lower() for i in issues):
            recommendations.append("🛡️  Use parameterized queries exclusively")
            recommendations.append("🛡️  Implement input validation and sanitization")
        
        recommendations.append("🔍 Consider implementing automated security scanning in CI/CD")
        recommendations.append("📋 Regular security audit and penetration testing")
        
        return recommendations
    
    def _get_quality_recommendations(self, issues: List[Dict], stats: Dict) -> List[str]:
        """Get code quality improvement recommendations."""
        recommendations = []
        
        if stats["avg_complexity"] > 8:
            recommendations.append("🔧 Refactor high-complexity functions")
            recommendations.append("🔧 Consider breaking down large functions")
        
        if any(i["type"] == "style" for i in issues):
            recommendations.append("💅 Set up automated code formatting (Black, Prettier)")
            recommendations.append("💅 Configure pre-commit hooks for consistent style")
        
        recommendations.append("📏 Implement code review process")
        recommendations.append("🔄 Set up continuous integration with quality gates")
        
        return recommendations
    
    def _get_architecture_recommendations(self, issues: List[Dict]) -> List[str]:
        """Get architecture improvement recommendations."""
        recommendations = []
        
        if any("circular" in i["message"].lower() for i in issues):
            recommendations.append("🏗️  Resolve circular dependencies")
            recommendations.append("🏗️  Consider dependency injection patterns")
        
        if any("separation" in i["type"] for i in issues):
            recommendations.append("🎯 Implement proper separation of concerns")
            recommendations.append("🎯 Move business logic to dedicated service layers")
        
        recommendations.append("📐 Consider implementing design patterns (Factory, Observer, etc.)")
        recommendations.append("🔍 Regular architecture review sessions")
        
        return recommendations
    
    def _get_performance_recommendations(self, issues: List[Dict]) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        if any("n+1" in i["message"].lower() for i in issues):
            recommendations.append("⚡ Implement query optimization and batching")
            recommendations.append("⚡ Consider using ORM query optimization")
        
        if any("caching" in i["message"].lower() for i in issues):
            recommendations.append("🏪 Implement caching strategy (Redis, Memcached)")
            recommendations.append("🏪 Consider application-level caching")
        
        recommendations.append("📊 Set up performance monitoring and alerting")
        recommendations.append("🎯 Implement database indexing strategy")
        
        return recommendations
    
    def _get_documentation_recommendations(self, coverage: float, issues: List[Dict]) -> List[str]:
        """Get documentation improvement recommendations."""
        recommendations = []
        
        if coverage < 70:
            recommendations.append("📚 Increase docstring coverage for functions and classes")
            recommendations.append("📚 Document complex algorithms and business logic")
        
        if any("README" in i["message"] for i in issues):
            recommendations.append("📖 Create comprehensive README with setup instructions")
            recommendations.append("📖 Add API documentation and examples")
        
        recommendations.append("🔄 Set up automated documentation generation")
        recommendations.append("👥 Create contributor guidelines and code of conduct")
        
        return recommendations
    
    def _get_dependency_recommendations(self, issues: List[Dict]) -> List[str]:
        """Get dependency management recommendations."""
        recommendations = []
        
        if any("vulnerable" in i["message"].lower() for i in issues):
            recommendations.append("🔒 Update vulnerable packages immediately")
            recommendations.append("🔒 Set up automated vulnerability scanning")
        
        if any("unpinned" in i["message"].lower() for i in issues):
            recommendations.append("📌 Pin dependency versions for reproducible builds")
            recommendations.append("📌 Use lock files (requirements.lock, poetry.lock)")
        
        recommendations.append("🔄 Regular dependency updates and security patches")
        recommendations.append("🔍 Implement dependency audit in CI/CD pipeline")
        
        return recommendations
    
    def _get_production_recommendations(self, issues: List[Dict]) -> List[str]:
        """Get production readiness recommendations."""
        recommendations = []
        
        if any("configuration" in i["type"] for i in issues):
            recommendations.append("⚙️  Set up environment-specific configurations")
            recommendations.append("⚙️  Implement configuration validation")
        
        if any("monitoring" in i["type"] for i in issues):
            recommendations.append("📊 Implement health checks and monitoring")
            recommendations.append("📊 Set up logging and alerting")
        
        if any("deployment" in i["type"] for i in issues):
            recommendations.append("🚀 Create Docker containerization")
            recommendations.append("🚀 Set up CI/CD pipeline")
        
        recommendations.append("🔄 Implement rollback and disaster recovery procedures")
        recommendations.append("🔒 Set up production security hardening")
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate comprehensive quality report."""
        report = []
        report.append("=" * 80)
        report.append("🚀 AUTONOMOUS QUALITY GATES REPORT")
        report.append("=" * 80)
        report.append("")
        
        overall = self.results.get("overall", {})
        report.append(f"📊 OVERALL STATUS: {overall.get('overall_status', 'UNKNOWN')}")
        report.append(f"🎯 QUALITY SCORE: {overall.get('weighted_score', 0):.1f}/100 (Grade: {overall.get('grade', 'N/A')})")
        report.append(f"⚠️  TOTAL ISSUES: {overall.get('total_issues', 0)}")
        report.append(f"🚨 CRITICAL ISSUES: {overall.get('critical_issues', 0)}")
        report.append(f"✅ QUALITY GATES: {overall.get('gates_passed', '0/0')}")
        report.append(f"🚀 DEPLOYMENT READY: {'Yes' if overall.get('deployment_ready', False) else 'No'}")
        report.append("")
        
        # Individual category scores
        report.append("📈 CATEGORY BREAKDOWN:")
        report.append("-" * 50)
        
        categories = [
            ("🛡️  Security", "security"),
            ("📊 Code Quality", "code_quality"),
            ("🏗️  Architecture", "architecture"),
            ("⚡ Performance", "performance"),
            ("📚 Documentation", "documentation"),
            ("🔐 Dependencies", "dependencies"),
            ("🚀 Production", "production")
        ]
        
        for name, key in categories:
            if key in self.results:
                score = self.results[key].get("score", 0)
                issues_count = len(self.results[key].get("issues", []))
                report.append(f"{name:20} | Score: {score:5.1f} | Issues: {issues_count:3d}")
        
        report.append("")
        
        # Critical issues that must be addressed
        critical_issues = []
        for category_result in self.results.values():
            if isinstance(category_result, dict) and "issues" in category_result:
                critical_issues.extend([
                    issue for issue in category_result["issues"]
                    if issue.get("severity") == "critical"
                ])
        
        if critical_issues:
            report.append("🚨 CRITICAL ISSUES (MUST FIX):")
            report.append("-" * 50)
            for issue in critical_issues:
                report.append(f"❌ {issue.get('file', 'Unknown')}: {issue.get('message', 'No message')}")
            report.append("")
        
        # Top recommendations
        all_recommendations = []
        for category_result in self.results.values():
            if isinstance(category_result, dict) and "recommendations" in category_result:
                all_recommendations.extend(category_result["recommendations"][:3])  # Top 3 per category
        
        if all_recommendations:
            report.append("💡 TOP RECOMMENDATIONS:")
            report.append("-" * 50)
            for i, rec in enumerate(all_recommendations[:10], 1):  # Top 10 overall
                report.append(f"{i:2d}. {rec}")
            report.append("")
        
        # Quality gates details
        if "overall" in self.results and "quality_gates" in self.results["overall"]:
            gates = self.results["overall"]["quality_gates"]
            report.append("🚪 QUALITY GATES STATUS:")
            report.append("-" * 50)
            for gate_name, passed in gates.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                report.append(f"{gate_name.replace('_', ' ').title():25} | {status}")
            report.append("")
        
        report.append("⏱️  ANALYSIS COMPLETED IN: {:.2f} seconds".format(self.results.get("execution_time", 0)))
        report.append("=" * 80)
        
        return "\n".join(report)


class ComplexityAnalyzer(ast.NodeVisitor):
    """Cyclomatic complexity analyzer."""
    
    def __init__(self) -> None:
        self.complexities: Dict[str, int] = {}
        self.current_function: Optional[str] = None
        self.current_complexity: int = 0
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        old_function = self.current_function
        old_complexity = self.current_complexity
        
        self.current_function = node.name
        self.current_complexity = 1  # Base complexity
        
        self.generic_visit(node)
        
        self.complexities[node.name] = self.current_complexity
        
        self.current_function = old_function
        self.current_complexity = old_complexity
    
    def visit_If(self, node: ast.If) -> None:
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> None:
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For) -> None:
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_With(self, node: ast.With) -> None:
        self.current_complexity += 1
        self.generic_visit(node)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Quality Gates System")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    # Run quality gates
    validator = QualityGateValidator(args.project_root)
    results = validator.run_all_quality_gates()
    
    # Generate and display report
    if args.json:
        import json
        output = json.dumps(results, indent=2, default=str)
    else:
        output = validator.generate_report()
    
    print(output)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"\n📝 Detailed report saved to: {args.output}")
    
    # Exit with appropriate code
    overall_status = results.get("overall", {}).get("overall_status", "FAIL")
    exit(0 if overall_status == "PASS" else 1)


if __name__ == "__main__":
    main()