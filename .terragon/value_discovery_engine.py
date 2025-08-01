#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Optimized for Advanced Maturity Repositories (75%+)

This engine continuously discovers, scores, and prioritizes work items
using WSJF, ICE, and Technical Debt scoring methodologies.
"""

import json
import logging
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """Represents a discovered work item with comprehensive scoring."""
    id: str
    title: str
    description: str
    category: str  # security, performance, debt, feature, maintenance
    source: str    # gitHistory, staticAnalysis, etc.
    files_affected: List[str]
    effort_estimate: float  # story points or hours
    
    # WSJF Components
    user_business_value: float
    time_criticality: float
    risk_reduction: float
    opportunity_enablement: float
    
    # ICE Components
    impact: float
    confidence: float
    ease: float
    
    # Technical Debt Components
    debt_cost: float
    debt_interest: float
    hotspot_multiplier: float
    
    # Computed Scores
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    composite_score: float = 0.0
    
    # Metadata
    discovered_at: str = ""
    priority: str = "medium"
    tags: List[str] = None
    dependencies: List[str] = None
    risk_level: str = "low"


class ValueDiscoveryEngine:
    """Main engine for autonomous value discovery and prioritization."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.discovered_items: List[WorkItem] = []
        self.execution_history: List[Dict] = []
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for advanced repositories."""
        return {
            "repository_info": {"maturity_level": "advanced"},
            "scoring": {
                "weights": {
                    "advanced": {
                        "wsjf": 0.5,
                        "ice": 0.1,
                        "technicalDebt": 0.3,
                        "security": 0.1
                    }
                },
                "thresholds": {
                    "minScore": 15,
                    "maxRisk": 0.7,
                    "securityBoost": 2.0,
                    "complianceBoost": 1.8
                }
            }
        }
    
    def discover_value_items(self) -> List[WorkItem]:
        """Main discovery orchestration method."""
        logger.info("🔍 Starting comprehensive value discovery...")
        
        items = []
        
        # 1. Git History Analysis
        items.extend(self._discover_from_git_history())
        
        # 2. Static Analysis
        items.extend(self._discover_from_static_analysis())
        
        # 3. Security Analysis
        items.extend(self._discover_security_opportunities())
        
        # 4. Performance Analysis
        items.extend(self._discover_performance_opportunities())
        
        # 5. Dependency Analysis
        items.extend(self._discover_dependency_updates())
        
        # 6. Architecture Analysis
        items.extend(self._discover_architectural_improvements())
        
        # 7. Test Coverage Analysis
        items.extend(self._discover_test_improvements())
        
        self.discovered_items = items
        logger.info(f"📊 Discovered {len(items)} value opportunities")
        
        return items
    
    def _discover_from_git_history(self) -> List[WorkItem]:
        """Analyze git history for technical debt indicators."""
        items = []
        
        try:
            # Get recent commits with debt indicators
            result = subprocess.run([
                "git", "log", "--oneline", "--grep=TODO\\|FIXME\\|HACK\\|temp\\|quick",
                "-i", "--since=3 months ago"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            commits = result.stdout.strip().split('\n')
            debt_patterns = defaultdict(int)
            
            for commit in commits:
                if commit.strip():
                    # Extract patterns indicating technical debt
                    if re.search(r'(quick|temp|hack|fix)', commit.lower()):
                        debt_patterns['quick_fixes'] += 1
                    if re.search(r'(todo|fixme)', commit.lower()):
                        debt_patterns['deferred_work'] += 1
            
            # Create work items for high-frequency debt patterns
            if debt_patterns['quick_fixes'] > 3:
                items.append(WorkItem(
                    id=f"debt-quickfixes-{datetime.now().strftime('%Y%m%d')}",
                    title="Refactor accumulated quick fixes",
                    description=f"Found {debt_patterns['quick_fixes']} commits with quick fix patterns",
                    category="debt",
                    source="gitHistory",
                    files_affected=[],
                    effort_estimate=8.0,
                    user_business_value=6.0,
                    time_criticality=4.0,
                    risk_reduction=7.0,
                    opportunity_enablement=5.0,
                    impact=7.0,
                    confidence=8.0,
                    ease=6.0,
                    debt_cost=40.0,
                    debt_interest=10.0,
                    hotspot_multiplier=2.0,
                    tags=["refactoring", "technical-debt"],
                    risk_level="medium"
                ))
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git history analysis failed: {e}")
        
        return items
    
    def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Run static analysis tools to discover improvement opportunities."""
        items = []
        
        # Run ruff for code quality issues
        try:
            result = subprocess.run([
                "ruff", "check", ".", "--output-format=json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                
                # Group issues by severity and type
                issue_groups = defaultdict(list)
                for issue in issues:
                    severity = issue.get('fix', {}).get('applicability', 'medium')
                    issue_groups[severity].append(issue)
                
                # Create work items for high-impact issue groups
                if len(issue_groups.get('automatic', [])) > 5:
                    items.append(WorkItem(
                        id=f"lint-autofix-{datetime.now().strftime('%Y%m%d')}",
                        title="Apply automatic lint fixes",
                        description=f"Fix {len(issue_groups['automatic'])} automatically fixable issues",
                        category="maintenance",
                        source="staticAnalysis",
                        files_affected=[],
                        effort_estimate=2.0,
                        user_business_value=3.0,
                        time_criticality=2.0,
                        risk_reduction=4.0,
                        opportunity_enablement=3.0,
                        impact=4.0,
                        confidence=9.0,
                        ease=9.0,
                        debt_cost=10.0,
                        debt_interest=5.0,
                        hotspot_multiplier=1.0,
                        tags=["code-quality", "automation"],
                        risk_level="low"
                    ))
        
        except subprocess.CalledProcessError:
            logger.warning("Ruff analysis failed")
        except json.JSONDecodeError:
            logger.warning("Failed to parse ruff output")
        
        return items
    
    def _discover_security_opportunities(self) -> List[WorkItem]:
        """Discover security improvement opportunities."""
        items = []
        
        # Check if advanced security scan exists and run it
        security_script = self.repo_root / "scripts" / "advanced_security_scan.py"
        if security_script.exists():
            try:
                result = subprocess.run([
                    "python", str(security_script), "--json"
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                if result.returncode == 0 and result.stdout:
                    security_data = json.loads(result.stdout)
                    
                    for vuln in security_data.get('vulnerabilities', []):
                        items.append(WorkItem(
                            id=f"sec-{vuln.get('id', 'unknown')}-{datetime.now().strftime('%Y%m%d')}",
                            title=f"Fix {vuln.get('title', 'security issue')}",
                            description=vuln.get('description', 'Security vulnerability'),
                            category="security",
                            source="securityScan",
                            files_affected=vuln.get('files', []),
                            effort_estimate=float(vuln.get('effort', 4)),
                            user_business_value=8.0,
                            time_criticality=9.0,
                            risk_reduction=10.0,
                            opportunity_enablement=6.0,
                            impact=9.0,
                            confidence=8.0,
                            ease=float(vuln.get('ease', 6)),
                            debt_cost=50.0,
                            debt_interest=20.0,
                            hotspot_multiplier=3.0,
                            tags=["security", "vulnerability"],
                            risk_level=vuln.get('severity', 'high')
                        ))
            
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                logger.warning(f"Security scan failed: {e}")
        
        return items
    
    def _discover_performance_opportunities(self) -> List[WorkItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Check for performance analyzer
        perf_script = self.repo_root / "scripts" / "advanced_performance_analyzer.py"
        if perf_script.exists():
            try:
                result = subprocess.run([
                    "python", str(perf_script), "--analyze", "--json"
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                if result.returncode == 0 and result.stdout:
                    perf_data = json.loads(result.stdout)
                    
                    for bottleneck in perf_data.get('bottlenecks', []):
                        items.append(WorkItem(
                            id=f"perf-{bottleneck.get('id', 'opt')}-{datetime.now().strftime('%Y%m%d')}",
                            title=f"Optimize {bottleneck.get('component', 'performance issue')}",
                            description=bottleneck.get('description', 'Performance optimization opportunity'),
                            category="performance",
                            source="performanceAnalysis",
                            files_affected=bottleneck.get('files', []),
                            effort_estimate=float(bottleneck.get('effort', 6)),
                            user_business_value=7.0,
                            time_criticality=5.0,
                            risk_reduction=4.0,
                            opportunity_enablement=8.0,
                            impact=8.0,
                            confidence=7.0,
                            ease=float(bottleneck.get('ease', 5)),
                            debt_cost=30.0,
                            debt_interest=15.0,
                            hotspot_multiplier=2.0,
                            tags=["performance", "optimization"],
                            risk_level="medium"
                        ))
            
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                logger.warning(f"Performance analysis failed: {e}")
        
        return items
    
    def _discover_dependency_updates(self) -> List[WorkItem]:
        """Discover dependency update opportunities."""
        items = []
        
        try:
            # Check for outdated dependencies
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                
                # Group by criticality
                critical_updates = []
                regular_updates = []
                
                for pkg in outdated:
                    if self._is_security_critical_package(pkg['name']):
                        critical_updates.append(pkg)
                    else:
                        regular_updates.append(pkg)
                
                # Create work items for critical updates
                if critical_updates:
                    items.append(WorkItem(
                        id=f"deps-critical-{datetime.now().strftime('%Y%m%d')}",
                        title=f"Update {len(critical_updates)} critical dependencies",
                        description=f"Security-critical updates: {', '.join([p['name'] for p in critical_updates[:3]])}",
                        category="security",
                        source="dependencyAnalysis",
                        files_affected=["requirements.txt", "pyproject.toml"],
                        effort_estimate=4.0,
                        user_business_value=9.0,
                        time_criticality=9.0,
                        risk_reduction=10.0,
                        opportunity_enablement=5.0,
                        impact=9.0,
                        confidence=8.0,
                        ease=7.0,
                        debt_cost=45.0,
                        debt_interest=25.0,
                        hotspot_multiplier=2.5,
                        tags=["dependencies", "security", "updates"],
                        risk_level="high"
                    ))
                
                # Create work item for regular updates if many exist
                if len(regular_updates) > 10:
                    items.append(WorkItem(
                        id=f"deps-regular-{datetime.now().strftime('%Y%m%d')}",
                        title=f"Update {len(regular_updates)} regular dependencies",
                        description=f"Regular dependency updates for better features and performance",
                        category="maintenance",
                        source="dependencyAnalysis",
                        files_affected=["requirements.txt", "pyproject.toml"],
                        effort_estimate=6.0,
                        user_business_value=4.0,
                        time_criticality=3.0,
                        risk_reduction=5.0,
                        opportunity_enablement=6.0,
                        impact=5.0,
                        confidence=7.0,
                        ease=6.0,
                        debt_cost=20.0,
                        debt_interest=8.0,
                        hotspot_multiplier=1.2,
                        tags=["dependencies", "maintenance"],
                        risk_level="low"
                    ))
        
        except subprocess.CalledProcessError as e:
            logger.warning(f"Dependency analysis failed: {e}")
        
        return items
    
    def _discover_architectural_improvements(self) -> List[WorkItem]:
        """Discover architectural improvement opportunities."""
        items = []
        
        # Check for architectural debt analyzer
        arch_script = self.repo_root / "scripts" / "technical_debt_analyzer.py"
        if arch_script.exists():
            try:
                result = subprocess.run([
                    "python", str(arch_script), "--json"
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                if result.returncode == 0 and result.stdout:
                    debt_data = json.loads(result.stdout)
                    
                    for debt_item in debt_data.get('technical_debt', []):
                        items.append(WorkItem(
                            id=f"arch-{debt_item.get('id', 'improve')}-{datetime.now().strftime('%Y%m%d')}",
                            title=f"Refactor {debt_item.get('component', 'architectural component')}",
                            description=debt_item.get('description', 'Architectural improvement opportunity'),
                            category="debt",
                            source="architecturalAnalysis",
                            files_affected=debt_item.get('files', []),
                            effort_estimate=float(debt_item.get('effort', 10)),
                            user_business_value=6.0,
                            time_criticality=4.0,
                            risk_reduction=8.0,
                            opportunity_enablement=9.0,
                            impact=8.0,
                            confidence=6.0,
                            ease=float(debt_item.get('ease', 4)),
                            debt_cost=float(debt_item.get('cost', 60)),
                            debt_interest=float(debt_item.get('interest', 20)),
                            hotspot_multiplier=float(debt_item.get('hotspot', 2.5)),
                            tags=["architecture", "refactoring", "technical-debt"],
                            risk_level=debt_item.get('risk', 'medium')
                        ))
            
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                logger.warning(f"Architectural analysis failed: {e}")
        
        return items
    
    def _discover_test_improvements(self) -> List[WorkItem]:
        """Discover test coverage and quality improvements."""
        items = []
        
        try:
            # Run coverage analysis
            result = subprocess.run([
                "pytest", "--cov=src", "--cov-report=json", "--collect-only", "-q"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            # Check for coverage.json if it exists
            coverage_file = self.repo_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 100)
                
                if total_coverage < 85:  # Target higher than minimum 80%
                    items.append(WorkItem(
                        id=f"test-coverage-{datetime.now().strftime('%Y%m%d')}",
                        title=f"Improve test coverage from {total_coverage:.1f}% to 85%+",
                        description="Add missing test cases to improve code quality and reliability",
                        category="testing",
                        source="testAnalysis",
                        files_affected=["tests/"],
                        effort_estimate=8.0,
                        user_business_value=7.0,
                        time_criticality=4.0,
                        risk_reduction=9.0,
                        opportunity_enablement=6.0,
                        impact=8.0,
                        confidence=8.0,
                        ease=7.0,
                        debt_cost=35.0,
                        debt_interest=12.0,
                        hotspot_multiplier=1.5,
                        tags=["testing", "coverage", "quality"],
                        risk_level="low"
                    ))
        
        except subprocess.CalledProcessError:
            logger.warning("Test coverage analysis failed")
        
        return items
    
    def _is_security_critical_package(self, package_name: str) -> bool:
        """Check if a package is security-critical."""
        critical_packages = {
            'requests', 'urllib3', 'cryptography', 'pycryptodome',
            'sqlalchemy', 'psycopg2', 'pymysql', 'django', 'flask',
            'streamlit', 'pandas', 'numpy'  # Framework-specific
        }
        return package_name.lower() in critical_packages
    
    def score_work_items(self, items: List[WorkItem]) -> List[WorkItem]:
        """Apply comprehensive scoring to work items."""
        logger.info("📊 Scoring discovered work items...")
        
        weights = self.config['scoring']['weights']['advanced']
        thresholds = self.config['scoring']['thresholds']
        
        for item in items:
            # Calculate WSJF Score
            cost_of_delay = (
                item.user_business_value +
                item.time_criticality +
                item.risk_reduction +
                item.opportunity_enablement
            )
            item.wsjf_score = cost_of_delay / max(item.effort_estimate, 0.1)
            
            # Calculate ICE Score
            item.ice_score = item.impact * item.confidence * item.ease
            
            # Calculate Technical Debt Score
            item.technical_debt_score = (
                (item.debt_cost + item.debt_interest) * item.hotspot_multiplier
            )
            
            # Calculate Composite Score
            normalized_wsjf = min(item.wsjf_score / 10, 10)
            normalized_ice = min(item.ice_score / 100, 10)
            normalized_debt = min(item.technical_debt_score / 50, 10)
            
            item.composite_score = (
                weights['wsjf'] * normalized_wsjf +
                weights['ice'] * normalized_ice +
                weights['technicalDebt'] * normalized_debt
            )
            
            # Apply category-specific boosts
            if item.category == 'security':
                item.composite_score *= thresholds['securityBoost']
            elif item.category == 'performance':
                item.composite_score *= thresholds.get('performanceBoost', 1.5)
            elif item.risk_level == 'high':
                item.composite_score *= 1.3
            
            # Set priority based on composite score
            if item.composite_score >= 8:
                item.priority = 'high'
            elif item.composite_score >= 5:
                item.priority = 'medium'
            else:
                item.priority = 'low'
        
        # Sort by composite score descending
        items.sort(key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"✅ Scored {len(items)} items, top score: {items[0].composite_score:.2f}")
        return items
    
    def select_next_best_value(self, items: List[WorkItem]) -> Optional[WorkItem]:
        """Select the next highest-value item for execution."""
        thresholds = self.config['scoring']['thresholds']
        
        for item in items:
            # Skip if below minimum score threshold
            if item.composite_score < thresholds['minScore']:
                continue
            
            # Skip if risk too high
            risk_score = self._calculate_risk_score(item)
            if risk_score > thresholds['maxRisk']:
                continue
            
            # Found our next best value item
            logger.info(f"🎯 Selected next value item: {item.title} (score: {item.composite_score:.2f})")
            return item
        
        logger.info("🔄 No qualifying high-value items, generating housekeeping task")
        return self._generate_housekeeping_task()
    
    def _calculate_risk_score(self, item: WorkItem) -> float:
        """Calculate risk score for a work item."""
        risk_factors = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
        
        base_risk = risk_factors.get(item.risk_level, 0.5)
        
        # Adjust based on effort (higher effort = higher risk)
        effort_risk = min(item.effort_estimate / 20, 0.3)
        
        # Adjust based on files affected (more files = higher risk)
        files_risk = min(len(item.files_affected) / 10, 0.2)
        
        return min(base_risk + effort_risk + files_risk, 1.0)
    
    def _generate_housekeeping_task(self) -> WorkItem:
        """Generate a housekeeping task when no high-value items qualify."""
        return WorkItem(
            id=f"housekeeping-{datetime.now().strftime('%Y%m%d-%H%M')}",
            title="Dependency security scan",
            description="Run security scan on all dependencies for latest vulnerabilities",
            category="maintenance",
            source="housekeeping",
            files_affected=["requirements.txt"],
            effort_estimate=1.0,
            user_business_value=4.0,
            time_criticality=3.0,
            risk_reduction=6.0,
            opportunity_enablement=3.0,
            impact=5.0,
            confidence=9.0,
            ease=9.0,
            debt_cost=5.0,
            debt_interest=2.0,
            hotspot_multiplier=1.0,
            priority="medium",
            tags=["housekeeping", "security"],
            risk_level="low"
        )
    
    def save_metrics(self, items: List[WorkItem]) -> None:
        """Save value metrics and execution history."""
        metrics_file = self.repo_root / ".terragon" / "value-metrics.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "repository_maturity": self.config['repository_info']['maturity_score'],
            "discovered_items": len(items),
            "high_priority_items": len([i for i in items if i.priority == 'high']),
            "categories": {
                cat: len([i for i in items if i.category == cat])
                for cat in set(item.category for item in items)
            },
            "average_composite_score": sum(item.composite_score for item in items) / len(items) if items else 0,
            "top_items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "category": item.category,
                    "composite_score": round(item.composite_score, 2),
                    "priority": item.priority
                }
                for item in items[:10]
            ]
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"💾 Saved metrics to {metrics_file}")


def main():
    """Main execution function for value discovery."""
    engine = ValueDiscoveryEngine()
    
    # Discover value opportunities
    items = engine.discover_value_items()
    
    # Score and prioritize items
    scored_items = engine.score_work_items(items)
    
    # Select next best value item
    next_item = engine.select_next_best_value(scored_items)
    
    # Save metrics
    engine.save_metrics(scored_items)
    
    # Output results
    if next_item:
        print(f"\n🎯 NEXT BEST VALUE ITEM:")
        print(f"   Title: {next_item.title}")
        print(f"   Score: {next_item.composite_score:.2f}")
        print(f"   Category: {next_item.category}")
        print(f"   Effort: {next_item.effort_estimate} hours")
        print(f"   Priority: {next_item.priority}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total items discovered: {len(scored_items)}")
    print(f"   High priority items: {len([i for i in scored_items if i.priority == 'high'])}")
    print(f"   Average score: {sum(i.composite_score for i in scored_items) / len(scored_items):.2f}")


if __name__ == "__main__":
    main()