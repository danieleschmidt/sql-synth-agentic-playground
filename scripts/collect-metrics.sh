#!/bin/bash
# Comprehensive Metrics Collection Script
# Collects and aggregates project metrics from various sources

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
METRICS_DIR="metrics"
REPORTS_DIR="reports"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATE_SUFFIX=$(date -u +"%Y%m%d")

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to create output directories
setup_directories() {
    print_status "Setting up output directories..."
    mkdir -p "$METRICS_DIR/raw" "$METRICS_DIR/processed" "$REPORTS_DIR"
}

# Function to collect Git metrics
collect_git_metrics() {
    print_status "Collecting Git metrics..."
    
    local output_file="$METRICS_DIR/raw/git-metrics-$DATE_SUFFIX.json"
    
    # Collect commit statistics
    local commits_last_week=$(git rev-list --count --since="1 week ago" HEAD)
    local commits_last_month=$(git rev-list --count --since="1 month ago" HEAD)
    local total_commits=$(git rev-list --count HEAD)
    
    # Collect contributor statistics
    local contributors_last_month=$(git shortlog --since="1 month ago" -sn | wc -l)
    local total_contributors=$(git shortlog -sn | wc -l)
    
    # Collect lines of code statistics
    local total_lines=$(find . -name "*.py" -not -path "./.*" -not -path "./venv/*" -not -path "./env/*" | xargs wc -l | tail -1 | awk '{print $1}')
    
    # Collect file statistics
    local python_files=$(find . -name "*.py" -not -path "./.*" | wc -l)
    local yaml_files=$(find . -name "*.yml" -o -name "*.yaml" -not -path "./.*" | wc -l)
    local md_files=$(find . -name "*.md" -not -path "./.*" | wc -l)
    
    # Generate JSON output
    cat <<EOF > "$output_file"
{
  "timestamp": "$TIMESTAMP",
  "repository": {
    "name": "$(basename $(git rev-parse --show-toplevel))",
    "url": "$(git config --get remote.origin.url)",
    "branch": "$(git rev-parse --abbrev-ref HEAD)",
    "commit": "$(git rev-parse HEAD)"
  },
  "commits": {
    "total": $total_commits,
    "lastWeek": $commits_last_week,
    "lastMonth": $commits_last_month
  },
  "contributors": {
    "total": $total_contributors,
    "lastMonth": $contributors_last_month
  },
  "codebase": {
    "totalLines": $total_lines,
    "files": {
      "python": $python_files,
      "yaml": $yaml_files,
      "markdown": $md_files
    }
  }
}
EOF
    
    print_success "Git metrics collected: $output_file"
}

# Function to collect test metrics
collect_test_metrics() {
    print_status "Collecting test metrics..."
    
    local output_file="$METRICS_DIR/raw/test-metrics-$DATE_SUFFIX.json"
    
    if [[ -f "pytest.ini" || -f "pyproject.toml" ]]; then
        print_status "Running test suite with coverage..."
        
        # Run tests with coverage and JSON output
        python -m pytest \
            --cov=src \
            --cov-report=json:"$METRICS_DIR/raw/coverage-$DATE_SUFFIX.json" \
            --junitxml="$METRICS_DIR/raw/junit-$DATE_SUFFIX.xml" \
            --json-report --json-report-file="$METRICS_DIR/raw/pytest-$DATE_SUFFIX.json" \
            tests/ || true
        
        # Extract metrics from coverage report
        if [[ -f "$METRICS_DIR/raw/coverage-$DATE_SUFFIX.json" ]]; then
            local coverage_percent=$(jq -r '.totals.percent_covered' "$METRICS_DIR/raw/coverage-$DATE_SUFFIX.json")
            local covered_lines=$(jq -r '.totals.covered_lines' "$METRICS_DIR/raw/coverage-$DATE_SUFFIX.json")
            local total_lines=$(jq -r '.totals.num_statements' "$METRICS_DIR/raw/coverage-$DATE_SUFFIX.json")
        else
            local coverage_percent=0
            local covered_lines=0
            local total_lines=0
        fi
        
        # Count test files and tests
        local test_files=$(find tests/ -name "test_*.py" 2>/dev/null | wc -l || echo 0)
        local total_tests=$(grep -r "def test_" tests/ 2>/dev/null | wc -l || echo 0)
        
        # Generate test metrics JSON
        cat <<EOF > "$output_file"
{
  "timestamp": "$TIMESTAMP",
  "coverage": {
    "percentage": $coverage_percent,
    "coveredLines": $covered_lines,
    "totalLines": $total_lines
  },
  "tests": {
    "totalFiles": $test_files,
    "totalTests": $total_tests
  }
}
EOF
    else
        print_warning "No test configuration found, skipping test metrics"
        echo '{}' > "$output_file"
    fi
    
    print_success "Test metrics collected: $output_file"
}

# Function to collect security metrics
collect_security_metrics() {
    print_status "Collecting security metrics..."
    
    local output_file="$METRICS_DIR/raw/security-metrics-$DATE_SUFFIX.json"
    
    # Initialize security metrics
    local bandit_issues=0
    local safety_vulnerabilities=0
    local outdated_packages=0
    
    # Run Bandit security scan
    if command -v bandit >/dev/null 2>&1; then
        print_status "Running Bandit security scan..."
        bandit -r src/ -f json -o "$METRICS_DIR/raw/bandit-$DATE_SUFFIX.json" || true
        
        if [[ -f "$METRICS_DIR/raw/bandit-$DATE_SUFFIX.json" ]]; then
            bandit_issues=$(jq '.results | length' "$METRICS_DIR/raw/bandit-$DATE_SUFFIX.json" 2>/dev/null || echo 0)
        fi
    fi
    
    # Run Safety vulnerability check
    if command -v safety >/dev/null 2>&1; then
        print_status "Running Safety vulnerability check..."
        safety check --json --output "$METRICS_DIR/raw/safety-$DATE_SUFFIX.json" || true
        
        if [[ -f "$METRICS_DIR/raw/safety-$DATE_SUFFIX.json" ]]; then
            safety_vulnerabilities=$(jq '. | length' "$METRICS_DIR/raw/safety-$DATE_SUFFIX.json" 2>/dev/null || echo 0)
        fi
    fi
    
    # Check for outdated packages
    if command -v pip >/dev/null 2>&1; then
        print_status "Checking for outdated packages..."
        pip list --outdated --format=json > "$METRICS_DIR/raw/outdated-packages-$DATE_SUFFIX.json" 2>/dev/null || echo '[]' > "$METRICS_DIR/raw/outdated-packages-$DATE_SUFFIX.json"
        outdated_packages=$(jq '. | length' "$METRICS_DIR/raw/outdated-packages-$DATE_SUFFIX.json")
    fi
    
    # Generate security metrics JSON
    cat <<EOF > "$output_file"
{
  "timestamp": "$TIMESTAMP",
  "vulnerabilities": {
    "banditIssues": $bandit_issues,
    "safetyVulnerabilities": $safety_vulnerabilities,
    "outdatedPackages": $outdated_packages
  }
}
EOF
    
    print_success "Security metrics collected: $output_file"
}

# Function to collect performance metrics
collect_performance_metrics() {
    print_status "Collecting performance metrics..."
    
    local output_file="$METRICS_DIR/raw/performance-metrics-$DATE_SUFFIX.json"
    
    # Run performance tests if available
    if [[ -d "tests/performance" ]]; then
        print_status "Running performance tests..."
        python -m pytest tests/performance/ \
            --benchmark-json="$METRICS_DIR/raw/benchmark-$DATE_SUFFIX.json" \
            --tb=short || true
    fi
    
    # Collect Docker image metrics if available
    local image_size=0
    if command -v docker >/dev/null 2>&1; then
        local image_name="sql-synth-agentic-playground:latest"
        if docker image inspect "$image_name" >/dev/null 2>&1; then
            image_size=$(docker image inspect "$image_name" --format='{{.Size}}' 2>/dev/null || echo 0)
        fi
    fi
    
    # Generate performance metrics JSON
    cat <<EOF > "$output_file"
{
  "timestamp": "$TIMESTAMP",
  "docker": {
    "imageSize": $image_size
  }
}
EOF
    
    print_success "Performance metrics collected: $output_file"
}

# Function to collect code quality metrics
collect_code_quality_metrics() {
    print_status "Collecting code quality metrics..."
    
    local output_file="$METRICS_DIR/raw/quality-metrics-$DATE_SUFFIX.json"
    
    # Run linting checks
    local ruff_issues=0
    local mypy_errors=0
    
    if command -v ruff >/dev/null 2>&1; then
        print_status "Running Ruff linting..."
        ruff check . --format=json --output-file="$METRICS_DIR/raw/ruff-$DATE_SUFFIX.json" || true
        
        if [[ -f "$METRICS_DIR/raw/ruff-$DATE_SUFFIX.json" ]]; then
            ruff_issues=$(jq '. | length' "$METRICS_DIR/raw/ruff-$DATE_SUFFIX.json" 2>/dev/null || echo 0)
        fi
    fi
    
    if command -v mypy >/dev/null 2>&1; then
        print_status "Running MyPy type checking..."
        mypy src/ --json-report "$METRICS_DIR/raw/mypy-$DATE_SUFFIX" || true
        
        if [[ -f "$METRICS_DIR/raw/mypy-$DATE_SUFFIX/index.txt" ]]; then
            mypy_errors=$(grep -c "error" "$METRICS_DIR/raw/mypy-$DATE_SUFFIX/index.txt" 2>/dev/null || echo 0)
        fi
    fi
    
    # Calculate cyclomatic complexity
    local avg_complexity=0
    if command -v radon >/dev/null 2>&1; then
        radon cc src/ -j > "$METRICS_DIR/raw/complexity-$DATE_SUFFIX.json" || true
        # This would require more complex parsing to get average
    fi
    
    # Generate code quality metrics JSON
    cat <<EOF > "$output_file"
{
  "timestamp": "$TIMESTAMP",
  "linting": {
    "ruffIssues": $ruff_issues,
    "mypyErrors": $mypy_errors
  },
  "complexity": {
    "averageCyclomaticComplexity": $avg_complexity
  }
}
EOF
    
    print_success "Code quality metrics collected: $output_file"
}

# Function to aggregate all metrics
aggregate_metrics() {
    print_status "Aggregating metrics..."
    
    local output_file="$METRICS_DIR/processed/aggregated-metrics-$DATE_SUFFIX.json"
    
    # Combine all raw metrics
    python3 <<EOF
import json
import glob
from datetime import datetime

# Read all raw metrics files
metrics = {
    "timestamp": "$TIMESTAMP",
    "date": "$DATE_SUFFIX",
    "git": {},
    "tests": {},
    "security": {},
    "performance": {},
    "quality": {}
}

# Load individual metric files
for file_pattern, key in [
    ("$METRICS_DIR/raw/git-metrics-*.json", "git"),
    ("$METRICS_DIR/raw/test-metrics-*.json", "tests"),
    ("$METRICS_DIR/raw/security-metrics-*.json", "security"),
    ("$METRICS_DIR/raw/performance-metrics-*.json", "performance"),
    ("$METRICS_DIR/raw/quality-metrics-*.json", "quality")
]:
    files = glob.glob(file_pattern)
    if files:
        try:
            with open(files[0], 'r') as f:
                metrics[key] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {files[0]}: {e}")
            metrics[key] = {}

# Calculate derived metrics
if 'git' in metrics and 'codebase' in metrics['git']:
    total_lines = metrics['git']['codebase'].get('totalLines', 0)
    if 'tests' in metrics and 'coverage' in metrics['tests']:
        coverage_percent = metrics['tests']['coverage'].get('percentage', 0)
        metrics['derived'] = {
            'linesPerCommit': total_lines / max(metrics['git']['commits'].get('total', 1), 1),
            'testCoverageGrade': 'A' if coverage_percent >= 90 else 'B' if coverage_percent >= 80 else 'C' if coverage_percent >= 70 else 'D',
            'codeQualityScore': 100 - (metrics.get('security', {}).get('vulnerabilities', {}).get('banditIssues', 0) * 5) - (metrics.get('quality', {}).get('linting', {}).get('ruffIssues', 0) * 2)
        }

# Save aggregated metrics
with open("$output_file", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Aggregated metrics saved to $output_file")
EOF
    
    print_success "Metrics aggregated: $output_file"
}

# Function to generate reports
generate_reports() {
    print_status "Generating reports..."
    
    local metrics_file="$METRICS_DIR/processed/aggregated-metrics-$DATE_SUFFIX.json"
    
    if [[ ! -f "$metrics_file" ]]; then
        print_error "Aggregated metrics file not found: $metrics_file"
        return 1
    fi
    
    # Generate HTML report
    python3 <<EOF
import json
from datetime import datetime

with open("$metrics_file", 'r') as f:
    metrics = json.load(f)

html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Project Metrics Report - {metrics.get('date', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric-box {{ background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 5px; }}
        .good {{ background: #d4edda; }}
        .warning {{ background: #fff3cd; }}
        .danger {{ background: #f8d7da; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Project Metrics Report</h1>
    <p><strong>Generated:</strong> {metrics.get('timestamp', 'Unknown')}</p>
    
    <div class="metric-box good">
        <h2>📊 Overview</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Commits</td><td>{metrics.get('git', {}).get('commits', {}).get('total', 'N/A')}</td></tr>
            <tr><td>Total Lines of Code</td><td>{metrics.get('git', {}).get('codebase', {}).get('totalLines', 'N/A')}</td></tr>
            <tr><td>Test Coverage</td><td>{metrics.get('tests', {}).get('coverage', {}).get('percentage', 'N/A')}%</td></tr>
            <tr><td>Security Issues</td><td>{metrics.get('security', {}).get('vulnerabilities', {}).get('banditIssues', 'N/A')}</td></tr>
        </table>
    </div>
    
    <div class="metric-box">
        <h2>🔒 Security Status</h2>
        <p>Bandit Issues: {metrics.get('security', {}).get('vulnerabilities', {}).get('banditIssues', 'N/A')}</p>
        <p>Safety Vulnerabilities: {metrics.get('security', {}).get('vulnerabilities', {}).get('safetyVulnerabilities', 'N/A')}</p>
        <p>Outdated Packages: {metrics.get('security', {}).get('vulnerabilities', {}).get('outdatedPackages', 'N/A')}</p>
    </div>
    
    <div class="metric-box">
        <h2>✅ Code Quality</h2>
        <p>Ruff Issues: {metrics.get('quality', {}).get('linting', {}).get('ruffIssues', 'N/A')}</p>
        <p>MyPy Errors: {metrics.get('quality', {}).get('linting', {}).get('mypyErrors', 'N/A')}</p>
    </div>
    
    <div class="metric-box">
        <h2>📈 Derived Metrics</h2>
        <p>Code Quality Score: {metrics.get('derived', {}).get('codeQualityScore', 'N/A')}</p>
        <p>Test Coverage Grade: {metrics.get('derived', {}).get('testCoverageGrade', 'N/A')}</p>
    </div>
</body>
</html>
'''

with open("$REPORTS_DIR/metrics-report-$DATE_SUFFIX.html", 'w') as f:
    f.write(html_content)

print(f"HTML report generated: $REPORTS_DIR/metrics-report-$DATE_SUFFIX.html")
EOF
    
    # Generate summary report
    local summary_file="$REPORTS_DIR/metrics-summary-$DATE_SUFFIX.txt"
    
    echo "Project Metrics Summary - $DATE_SUFFIX" > "$summary_file"
    echo "========================================" >> "$summary_file"
    echo "Generated: $TIMESTAMP" >> "$summary_file"
    echo "" >> "$summary_file"
    
    if command -v jq >/dev/null 2>&1; then
        echo "Git Metrics:" >> "$summary_file"
        jq -r '"  Total Commits: " + (.git.commits.total // "N/A" | tostring)' "$metrics_file" >> "$summary_file" 2>/dev/null || echo "  Total Commits: N/A" >> "$summary_file"
        jq -r '"  Total Lines: " + (.git.codebase.totalLines // "N/A" | tostring)' "$metrics_file" >> "$summary_file" 2>/dev/null || echo "  Total Lines: N/A" >> "$summary_file"
        echo "" >> "$summary_file"
        
        echo "Test Metrics:" >> "$summary_file"
        jq -r '"  Coverage: " + (.tests.coverage.percentage // "N/A" | tostring) + "%"' "$metrics_file" >> "$summary_file" 2>/dev/null || echo "  Coverage: N/A" >> "$summary_file"
        echo "" >> "$summary_file"
        
        echo "Security Metrics:" >> "$summary_file"
        jq -r '"  Bandit Issues: " + (.security.vulnerabilities.banditIssues // "N/A" | tostring)' "$metrics_file" >> "$summary_file" 2>/dev/null || echo "  Bandit Issues: N/A" >> "$summary_file"
        jq -r '"  Safety Vulnerabilities: " + (.security.vulnerabilities.safetyVulnerabilities // "N/A" | tostring)' "$metrics_file" >> "$summary_file" 2>/dev/null || echo "  Safety Vulnerabilities: N/A" >> "$summary_file"
    fi
    
    print_success "Reports generated in $REPORTS_DIR/"
}

# Function to update project metrics JSON
update_project_metrics() {
    print_status "Updating project metrics JSON..."
    
    local project_metrics_file=".github/project-metrics.json"
    local aggregated_file="$METRICS_DIR/processed/aggregated-metrics-$DATE_SUFFIX.json"
    
    if [[ ! -f "$aggregated_file" ]]; then
        print_warning "Aggregated metrics file not found, skipping project metrics update"
        return
    fi
    
    # Update project metrics with latest data
    python3 <<EOF
import json
from datetime import datetime

# Load current project metrics
try:
    with open("$project_metrics_file", 'r') as f:
        project_metrics = json.load(f)
except FileNotFoundError:
    project_metrics = {}

# Load aggregated metrics
with open("$aggregated_file", 'r') as f:
    aggregated = json.load(f)

# Update project metrics with latest data
project_metrics['lastUpdated'] = "$TIMESTAMP"

if 'git' in aggregated and 'codebase' in aggregated['git']:
    if 'codebase' not in project_metrics:
        project_metrics['codebase'] = {'metrics': {}}
    
    project_metrics['codebase']['metrics']['totalLines'] = {
        'value': aggregated['git']['codebase'].get('totalLines', 0),
        'lastMeasured': "$TIMESTAMP"
    }

if 'tests' in aggregated and 'coverage' in aggregated['tests']:
    if 'testing' not in project_metrics:
        project_metrics['testing'] = {'coverage': {}}
    
    project_metrics['testing']['coverage']['overall'] = {
        'percentage': aggregated['tests']['coverage'].get('percentage', 0),
        'lastMeasured': "$TIMESTAMP"
    }

if 'security' in aggregated:
    if 'security' not in project_metrics:
        project_metrics['security'] = {'vulnerabilities': {}}
    
    vulns = aggregated['security'].get('vulnerabilities', {})
    project_metrics['security']['vulnerabilities'] = {
        'total': vulns.get('banditIssues', 0) + vulns.get('safetyVulnerabilities', 0),
        'lastScanned': "$TIMESTAMP"
    }

# Save updated project metrics
with open("$project_metrics_file", 'w') as f:
    json.dump(project_metrics, f, indent=2)

print(f"Project metrics updated: $project_metrics_file")
EOF
    
    print_success "Project metrics JSON updated"
}

# Function to clean up old metrics
cleanup_old_metrics() {
    print_status "Cleaning up old metrics..."
    
    # Keep only last 30 days of raw metrics
    find "$METRICS_DIR/raw" -name "*.json" -mtime +30 -delete 2>/dev/null || true
    find "$METRICS_DIR/raw" -name "*.xml" -mtime +30 -delete 2>/dev/null || true
    
    # Keep only last 7 days of processed metrics
    find "$METRICS_DIR/processed" -name "*.json" -mtime +7 -delete 2>/dev/null || true
    
    # Keep only last 30 days of reports
    find "$REPORTS_DIR" -name "*.html" -mtime +30 -delete 2>/dev/null || true
    find "$REPORTS_DIR" -name "*.txt" -mtime +30 -delete 2>/dev/null || true
    
    print_success "Old metrics cleaned up"
}

# Main execution function
main() {
    print_status "Starting comprehensive metrics collection..."
    
    # Parse command line arguments
    local skip_tests=false
    local skip_security=false
    local skip_reports=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-security)
                skip_security=true
                shift
                ;;
            --skip-reports)
                skip_reports=true
                shift
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-tests     Skip test metrics collection"
                echo "  --skip-security  Skip security metrics collection"
                echo "  --skip-reports   Skip report generation"
                echo "  --help           Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute collection steps
    setup_directories
    collect_git_metrics
    
    if [[ "$skip_tests" != "true" ]]; then
        collect_test_metrics
    fi
    
    if [[ "$skip_security" != "true" ]]; then
        collect_security_metrics
    fi
    
    collect_performance_metrics
    collect_code_quality_metrics
    aggregate_metrics
    
    if [[ "$skip_reports" != "true" ]]; then
        generate_reports
    fi
    
    update_project_metrics
    cleanup_old_metrics
    
    print_success "Metrics collection completed successfully!"
    print_status "Results available in:"
    print_status "  - Raw metrics: $METRICS_DIR/raw/"
    print_status "  - Processed metrics: $METRICS_DIR/processed/"
    print_status "  - Reports: $REPORTS_DIR/"
    print_status "  - Project metrics: .github/project-metrics.json"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi