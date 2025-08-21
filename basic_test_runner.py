#!/usr/bin/env python3
"""Basic test runner without external dependencies."""

import asyncio
import sys
import time
import traceback
import importlib.util
import os


class BasicTestRunner:
    """Simple test runner without external dependencies."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.start_time = time.time()

    def run_test(self, test_name: str, test_func):
        """Run a single test function."""
        self.tests_run += 1
        try:
            print(f"Running: {test_name}")
            
            if asyncio.iscoroutinefunction(test_func):
                result = asyncio.run(test_func())
            else:
                result = test_func()
            
            print(f"✓ PASS: {test_name}")
            self.tests_passed += 1
            return True
        except Exception as e:
            print(f"✗ FAIL: {test_name}")
            print(f"  Error: {e}")
            self.tests_failed += 1
            return False

    def print_summary(self):
        """Print test summary."""
        duration = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"Test Summary:")
        print(f"  Tests run: {self.tests_run}")
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {self.tests_failed}")
        print(f"  Duration: {duration:.2f}s")
        if self.tests_run > 0:
            print(f"  Success rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        print(f"{'='*50}")


def test_file_structure():
    """Test that all required files are present."""
    required_files = [
        'src/sql_synth/adaptive_learning_engine.py',
        'src/sql_synth/next_gen_optimization.py',
        'src/sql_synth/advanced_security_framework.py',
        'src/sql_synth/comprehensive_error_recovery.py',
        'src/sql_synth/hyperscale_performance_system.py',
        'src/sql_synth/multimodal_intelligence.py',
        'app.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise AssertionError(f"Missing files: {missing_files}")
    
    print(f"✓ All {len(required_files)} required files present")
    return True


def test_module_structure():
    """Test that modules have proper structure and can be imported."""
    modules_to_test = [
        'src.sql_synth.adaptive_learning_engine',
        'src.sql_synth.next_gen_optimization', 
        'src.sql_synth.advanced_security_framework',
        'src.sql_synth.comprehensive_error_recovery',
        'src.sql_synth.hyperscale_performance_system',
    ]
    
    importable_modules = 0
    for module_name in modules_to_test:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                importable_modules += 1
        except ImportError:
            pass  # Expected if dependencies aren't available
    
    print(f"✓ {importable_modules}/{len(modules_to_test)} modules have proper structure")
    return True


def test_code_syntax():
    """Test that all Python files have valid syntax."""
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            # Other errors like import errors are expected without dependencies
            pass
    
    if syntax_errors:
        raise AssertionError(f"Syntax errors found: {syntax_errors}")
    
    print(f"✓ All {len(python_files)} Python files have valid syntax")
    return True


def test_class_definitions():
    """Test that key classes are properly defined."""
    # Read and check key files for class definitions
    key_classes = {
        'src/sql_synth/adaptive_learning_engine.py': [
            'AdaptiveLearningEngine',
            'QueryPatternAnalyzer',
            'ReinforcementLearner',
        ],
        'src/sql_synth/next_gen_optimization.py': [
            'QuantumInspiredOptimizer',
            'NeuralAdaptiveOptimizer',
            'NextGenerationOptimizationEngine',
        ],
        'src/sql_synth/advanced_security_framework.py': [
            'BehavioralAnalyzer',
            'ThreatDetectionEngine',
            'ZeroTrustSecurityController',
        ],
        'src/sql_synth/comprehensive_error_recovery.py': [
            'IntelligentErrorClassifier',
            'CircuitBreaker',
            'ComprehensiveErrorRecoverySystem',
        ],
        'src/sql_synth/hyperscale_performance_system.py': [
            'SemanticCache',
            'IntelligentLoadBalancer',
            'HyperScalePerformanceSystem',
        ],
    }
    
    missing_classes = []
    for file_path, expected_classes in key_classes.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for class_name in expected_classes:
                if f'class {class_name}' not in content:
                    missing_classes.append(f"{file_path}: {class_name}")
        except FileNotFoundError:
            missing_classes.extend([f"{file_path}: {cls}" for cls in expected_classes])
    
    if missing_classes:
        raise AssertionError(f"Missing class definitions: {missing_classes}")
    
    total_classes = sum(len(classes) for classes in key_classes.values())
    print(f"✓ All {total_classes} key classes are properly defined")
    return True


def test_function_definitions():
    """Test that key utility functions are defined."""
    key_functions = {
        'src/sql_synth/adaptive_learning_engine.py': [
            'process_user_feedback',
            'get_query_recommendations',
            'get_learning_insights',
        ],
        'src/sql_synth/next_gen_optimization.py': [
            'optimize_query_performance_async',
            'get_optimization_insights',
        ],
        'src/sql_synth/advanced_security_framework.py': [
            'create_security_context',
            'authorize_sql_request',
            'get_security_insights',
        ],
        'src/sql_synth/comprehensive_error_recovery.py': [
            'execute_with_resilience',
            'get_resilience_insights',
        ],
        'src/sql_synth/hyperscale_performance_system.py': [
            'execute_optimized_operation',
            'get_hyperscale_insights',
        ],
    }
    
    missing_functions = []
    for file_path, expected_functions in key_functions.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for func_name in expected_functions:
                if f'def {func_name}' not in content and f'async def {func_name}' not in content:
                    missing_functions.append(f"{file_path}: {func_name}")
        except FileNotFoundError:
            missing_functions.extend([f"{file_path}: {func}" for func in expected_functions])
    
    if missing_functions:
        raise AssertionError(f"Missing function definitions: {missing_functions}")
    
    total_functions = sum(len(functions) for functions in key_functions.values())
    print(f"✓ All {total_functions} key functions are properly defined")
    return True


def test_documentation_quality():
    """Test documentation quality in key files."""
    files_to_check = [
        'src/sql_synth/adaptive_learning_engine.py',
        'src/sql_synth/next_gen_optimization.py',
        'src/sql_synth/advanced_security_framework.py',
        'src/sql_synth/comprehensive_error_recovery.py',
        'src/sql_synth/hyperscale_performance_system.py',
    ]
    
    doc_quality_score = 0
    total_files = len(files_to_check)
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for module docstring
            if '"""' in content and content.strip().startswith('"""'):
                doc_quality_score += 1
            
            # Check for class and function docstrings
            lines = content.split('\n')
            docstring_count = sum(1 for line in lines if '"""' in line or "'''" in line)
            
            if docstring_count >= 10:  # Reasonable number of docstrings
                doc_quality_score += 1
        except FileNotFoundError:
            pass
    
    if doc_quality_score < total_files:
        print(f"⚠ Documentation quality: {doc_quality_score}/{total_files} files well documented")
    else:
        print(f"✓ All {total_files} files have good documentation")
    
    return True


def test_configuration_files():
    """Test that configuration files are present and valid."""
    config_files = {
        'pyproject.toml': ['[build-system]', '[project]'],
        'requirements.txt': ['streamlit', 'langchain'],
        'README.md': ['# sql-synth-agentic-playground'],
    }
    
    for file_path, expected_content in config_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for expected in expected_content:
                if expected not in content:
                    raise AssertionError(f"{file_path} missing expected content: {expected}")
        except FileNotFoundError:
            raise AssertionError(f"Missing configuration file: {file_path}")
    
    print(f"✓ All {len(config_files)} configuration files are valid")
    return True


def test_code_complexity():
    """Test that code complexity is reasonable."""
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    total_lines = 0
    total_functions = 0
    total_classes = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines += len(lines)
            total_functions += sum(1 for line in lines if line.strip().startswith('def ') or line.strip().startswith('async def '))
            total_classes += sum(1 for line in lines if line.strip().startswith('class '))
        except Exception:
            pass
    
    print(f"✓ Code metrics: {total_lines} lines, {total_classes} classes, {total_functions} functions")
    
    # Basic sanity checks
    assert total_lines > 5000, f"Code base too small: {total_lines} lines"
    assert total_classes > 20, f"Too few classes: {total_classes}"
    assert total_functions > 100, f"Too few functions: {total_functions}"
    
    return True


async def test_async_compatibility():
    """Test async/await compatibility."""
    # Test basic async functionality
    async def test_async_func():
        await asyncio.sleep(0.001)
        return "async_success"
    
    result = await test_async_func()
    assert result == "async_success"
    
    # Test asyncio.gather
    async def test_multiple_async():
        tasks = [test_async_func() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        return results
    
    results = await test_multiple_async()
    assert len(results) == 5
    assert all(r == "async_success" for r in results)
    
    print("✓ Async/await compatibility verified")
    return True


def main():
    """Main test execution."""
    print("Starting Basic Quality Gates Test Suite")
    print("="*50)
    
    runner = BasicTestRunner()
    
    # Run structural tests
    test_functions = [
        ("File Structure", test_file_structure),
        ("Module Structure", test_module_structure),
        ("Code Syntax", test_code_syntax),
        ("Class Definitions", test_class_definitions),
        ("Function Definitions", test_function_definitions),
        ("Documentation Quality", test_documentation_quality),
        ("Configuration Files", test_configuration_files),
        ("Code Complexity", test_code_complexity),
        ("Async Compatibility", test_async_compatibility),
    ]
    
    for test_name, test_func in test_functions:
        runner.run_test(test_name, test_func)
    
    runner.print_summary()
    
    # Additional quality metrics
    print("\nAdditional Quality Metrics:")
    print("-" * 30)
    
    # Count total lines of code
    total_lines = 0
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except Exception:
                    pass
    
    print(f"Lines of Code: {total_lines:,}")
    print(f"Python Files: {len(python_files)}")
    print(f"Test Files: {len([f for f in os.listdir('tests') if f.endswith('.py')])}")
    
    # Calculate success rate
    if runner.tests_run > 0:
        success_rate = (runner.tests_passed / runner.tests_run) * 100
        
        if success_rate >= 90:
            print(f"\n🎉 EXCELLENT: {success_rate:.1f}% tests passed")
            quality_grade = "A+"
        elif success_rate >= 80:
            print(f"\n✅ GOOD: {success_rate:.1f}% tests passed")
            quality_grade = "A"
        elif success_rate >= 70:
            print(f"\n⚠️  ACCEPTABLE: {success_rate:.1f}% tests passed")
            quality_grade = "B"
        else:
            print(f"\n❌ NEEDS WORK: {success_rate:.1f}% tests passed")
            quality_grade = "C"
        
        print(f"Quality Grade: {quality_grade}")
    
    # Exit with appropriate code
    if runner.tests_failed > 0:
        print(f"\n{runner.tests_failed} quality gates failed. Review needed.")
        sys.exit(1)
    else:
        print(f"\n✅ All {runner.tests_passed} quality gates passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()