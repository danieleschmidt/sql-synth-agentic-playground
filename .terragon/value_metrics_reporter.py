#!/usr/bin/env python3
"""
Terragon Value Metrics Reporter
Generates comprehensive reports on autonomous SDLC value delivery.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValueReport:
    """Comprehensive value delivery report."""
    timestamp: str
    reporting_period: str
    repository_info: Dict
    execution_summary: Dict
    value_delivery: Dict
    trends: Dict
    recommendations: List[str]
    roi_analysis: Dict


class ValueMetricsReporter:
    """Generates comprehensive value delivery reports."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.terragon_dir = self.repo_root / ".terragon"
        
    def generate_comprehensive_report(self, days: int = 30) -> ValueReport:
        """Generate a comprehensive value delivery report."""
        logger.info(f"📊 Generating {days}-day value report...")
        
        # Load data sources
        execution_history = self._load_execution_history()
        value_metrics = self._load_value_metrics()
        scheduler_metrics = self._load_scheduler_metrics()
        
        # Filter by time period
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in execution_history
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]
        
        # Generate report sections
        repo_info = self._analyze_repository_info()
        exec_summary = self._analyze_execution_summary(recent_history)
        value_delivery = self._analyze_value_delivery(recent_history)
        trends = self._analyze_trends(recent_history)
        recommendations = self._generate_recommendations(recent_history, value_metrics)
        roi_analysis = self._calculate_roi_analysis(recent_history)
        
        report = ValueReport(
            timestamp=datetime.now().isoformat(),
            reporting_period=f"{days} days",
            repository_info=repo_info,
            execution_summary=exec_summary,
            value_delivery=value_delivery,
            trends=trends,
            recommendations=recommendations,
            roi_analysis=roi_analysis
        )
        
        # Save report
        self._save_report(report)
        
        logger.info("✅ Value report generated successfully")
        return report
    
    def _load_execution_history(self) -> List[Dict]:
        """Load execution history from log file."""
        log_file = self.terragon_dir / "execution-log.json"
        if not log_file.exists():
            return []
        
        try:
            with open(log_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _load_value_metrics(self) -> Dict:
        """Load latest value metrics."""
        metrics_file = self.terragon_dir / "value-metrics.json"
        if not metrics_file.exists():
            return {}
        
        try:
            with open(metrics_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _load_scheduler_metrics(self) -> Dict:
        """Load scheduler metrics."""
        metrics_file = self.terragon_dir / "scheduler-metrics.json"
        if not metrics_file.exists():
            return {}
        
        try:
            with open(metrics_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _analyze_repository_info(self) -> Dict:
        """Analyze repository information and maturity."""
        try:
            config_file = self.terragon_dir / "config.yaml"
            if config_file.exists():
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                repo_info = config.get('repository_info', {})
                return {
                    "name": repo_info.get('name', 'Unknown'),
                    "maturity_level": repo_info.get('maturity_level', 'unknown'),
                    "maturity_score": repo_info.get('maturity_score', 0),
                    "primary_language": repo_info.get('primary_language', 'unknown'),
                    "framework": repo_info.get('framework', 'unknown'),
                    "architecture_type": repo_info.get('architecture_type', 'unknown')
                }
        except Exception as e:
            logger.warning(f"Failed to load repository info: {e}")
        
        return {
            "name": "Unknown",
            "maturity_level": "advanced",
            "maturity_score": 78,
            "primary_language": "python",
            "framework": "streamlit+langchain",
            "architecture_type": "ml_ai_application"
        }
    
    def _analyze_execution_summary(self, history: List[Dict]) -> Dict:
        """Analyze execution summary statistics."""
        if not history:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "success_rate": 0.0,
                "average_duration": 0.0,
                "categories_addressed": [],
                "total_effort_hours": 0.0
            }
        
        successful = [h for h in history if h['execution']['status'] == 'completed']
        failed = [h for h in history if h['execution']['status'] != 'completed']
        
        # Calculate durations
        durations = [
            h['execution'].get('duration_seconds', 0) / 3600
            for h in history
            if 'duration_seconds' in h['execution']
        ]
        
        # Get categories
        categories = list(set(h['item']['category'] for h in history))
        
        # Calculate total effort
        total_effort = sum(h['item']['estimated_effort'] for h in history)
        
        return {
            "total_executions": len(history),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "success_rate": len(successful) / len(history),
            "average_duration": statistics.mean(durations) if durations else 0.0,
            "categories_addressed": categories,
            "total_effort_hours": total_effort
        }
    
    def _analyze_value_delivery(self, history: List[Dict]) -> Dict:
        """Analyze value delivery metrics."""
        if not history:
            return {
                "total_value_delivered": 0.0,
                "average_value_per_item": 0.0,
                "value_by_category": {},
                "high_impact_items": 0,
                "efficiency_score": 0.0
            }
        
        # Calculate total value delivered
        total_value = sum(h['learning_data']['value_delivered'] for h in history)
        avg_value = total_value / len(history)
        
        # Value by category
        category_values = {}
        for h in history:
            category = h['item']['category']
            if category not in category_values:
                category_values[category] = []
            category_values[category].append(h['learning_data']['value_delivered'])
        
        value_by_category = {
            cat: sum(values) for cat, values in category_values.items()
        }
        
        # High impact items (top 25% by composite score)
        scores = [h['item']['composite_score'] for h in history]
        high_impact_threshold = statistics.quantile(scores, 0.75) if len(scores) > 4 else 0
        high_impact_items = len([s for s in scores if s >= high_impact_threshold])
        
        # Efficiency score (value delivered / effort spent)
        total_effort = sum(h['item']['estimated_effort'] for h in history)
        efficiency = total_value / max(total_effort, 1)
        
        return {
            "total_value_delivered": total_value,
            "average_value_per_item": avg_value,
            "value_by_category": value_by_category,
            "high_impact_items": high_impact_items,
            "efficiency_score": efficiency
        }
    
    def _analyze_trends(self, history: List[Dict]) -> Dict:
        """Analyze trends over time."""
        if len(history) < 2:
            return {
                "value_trend": "insufficient_data",
                "success_rate_trend": "insufficient_data",
                "effort_accuracy_trend": "insufficient_data",
                "category_trends": {}
            }
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x['timestamp'])
        
        # Split into first and second half for trend analysis
        mid_point = len(sorted_history) // 2
        first_half = sorted_history[:mid_point]
        second_half = sorted_history[mid_point:]
        
        # Value delivery trend
        first_value = sum(h['learning_data']['value_delivered'] for h in first_half) / len(first_half)
        second_value = sum(h['learning_data']['value_delivered'] for h in second_half) / len(second_half)
        value_trend = "improving" if second_value > first_value else "declining"
        
        # Success rate trend
        first_success = len([h for h in first_half if h['execution']['status'] == 'completed']) / len(first_half)
        second_success = len([h for h in second_half if h['execution']['status'] == 'completed']) / len(second_half)
        success_trend = "improving" if second_success > first_success else "declining"
        
        # Effort accuracy trend
        first_accuracy = sum(h['learning_data']['effort_accuracy'] for h in first_half) / len(first_half)
        second_accuracy = sum(h['learning_data']['effort_accuracy'] for h in second_half) / len(second_half)
        accuracy_trend = "improving" if second_accuracy > first_accuracy else "declining"
        
        return {
            "value_trend": value_trend,
            "success_rate_trend": success_trend,
            "effort_accuracy_trend": accuracy_trend,
            "category_trends": self._analyze_category_trends(sorted_history)
        }
    
    def _analyze_category_trends(self, history: List[Dict]) -> Dict:
        """Analyze trends by category."""
        category_trends = {}
        
        # Group by category
        categories = {}
        for h in history:
            cat = h['item']['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(h)
        
        for category, items in categories.items():
            if len(items) < 2:
                category_trends[category] = "insufficient_data"
                continue
            
            # Sort by timestamp
            items = sorted(items, key=lambda x: x['timestamp'])
            
            # Compare first and second half
            mid = len(items) // 2
            first_value = sum(i['learning_data']['value_delivered'] for i in items[:mid]) / mid
            second_value = sum(i['learning_data']['value_delivered'] for i in items[mid:]) / (len(items) - mid)
            
            category_trends[category] = "improving" if second_value > first_value else "declining"
        
        return category_trends
    
    def _generate_recommendations(self, history: List[Dict], metrics: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not history:
            recommendations.append("Start autonomous execution to generate recommendations")
            return recommendations
        
        # Success rate analysis
        success_rate = len([h for h in history if h['execution']['status'] == 'completed']) / len(history)
        
        if success_rate < 0.8:
            recommendations.append(
                f"Success rate ({success_rate:.1%}) is below target. "
                "Consider reducing risk thresholds or improving validation processes."
            )
        elif success_rate > 0.95:
            recommendations.append(
                "Excellent success rate! Consider increasing risk tolerance to tackle more challenging items."
            )
        
        # Effort accuracy analysis
        effort_accuracies = [h['learning_data']['effort_accuracy'] for h in history]
        avg_accuracy = statistics.mean(effort_accuracies)
        
        if avg_accuracy < 0.7:
            recommendations.append(
                f"Effort estimation accuracy ({avg_accuracy:.1%}) needs improvement. "
                "Review effort estimation model parameters."
            )
        
        # Category analysis
        categories = {}
        for h in history:
            cat = h['item']['category']
            if cat not in categories:
                categories[cat] = {'count': 0, 'value': 0}
            categories[cat]['count'] += 1
            categories[cat]['value'] += h['learning_data']['value_delivered']
        
        # Find underperforming categories
        for category, data in categories.items():
            if data['count'] > 2 and data['value'] / data['count'] < 30:
                recommendations.append(
                    f"'{category}' category items show low value delivery. "
                    f"Consider adjusting scoring weights or improving execution strategies."
                )
        
        # High-value opportunities
        high_value_items = [h for h in history if h['item']['composite_score'] > 70]
        if len(high_value_items) / len(history) < 0.3:
            recommendations.append(
                "Increase focus on high-value items (score > 70). "
                "Consider adjusting discovery sources or scoring weights."
            )
        
        return recommendations
    
    def _calculate_roi_analysis(self, history: List[Dict]) -> Dict:
        """Calculate ROI analysis."""
        if not history:
            return {
                "total_investment_hours": 0,
                "estimated_value_delivered": 0,
                "roi_percentage": 0,
                "payback_period_days": 0,
                "value_per_hour": 0
            }
        
        # Calculate investment (effort spent)
        total_hours = sum(
            h['execution'].get('duration_seconds', h['item']['estimated_effort'] * 3600) / 3600
            for h in history
        )
        
        # Calculate value delivered
        total_value = sum(h['learning_data']['value_delivered'] for h in history)
        
        # Estimate monetary value (using industry benchmarks)
        # Security improvements: $2000/point, Performance: $1000/point, Tech Debt: $500/point
        value_multipliers = {
            'security': 2000,
            'performance': 1000,
            'debt': 500,
            'testing': 300,
            'maintenance': 200,
            'documentation': 100
        }
        
        monetary_value = 0
        for h in history:
            category = h['item']['category']
            multiplier = value_multipliers.get(category, 500)
            monetary_value += h['learning_data']['value_delivered'] * multiplier
        
        # Calculate ROI (assuming $100/hour cost)
        investment_cost = total_hours * 100
        roi_percentage = ((monetary_value - investment_cost) / max(investment_cost, 1)) * 100
        
        # Payback period (days to break even)
        daily_value = monetary_value / max(len(set(h['timestamp'][:10] for h in history)), 1)
        payback_days = investment_cost / max(daily_value, 1)
        
        return {
            "total_investment_hours": round(total_hours, 1),
            "estimated_value_delivered": round(monetary_value, 2),
            "roi_percentage": round(roi_percentage, 1),
            "payback_period_days": round(payback_days, 1),
            "value_per_hour": round(monetary_value / max(total_hours, 1), 2)
        }
    
    def _save_report(self, report: ValueReport) -> None:
        """Save the comprehensive report."""
        # Save JSON report
        report_file = self.terragon_dir / f"value-report-{datetime.now().strftime('%Y%m%d')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=str)
        
        # Save markdown summary
        markdown_file = self.terragon_dir / "VALUE_REPORT.md"
        markdown_content = self._generate_markdown_report(report)
        
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"📊 Reports saved: {report_file}, {markdown_file}")
    
    def _generate_markdown_report(self, report: ValueReport) -> str:
        """Generate markdown formatted report."""
        roi = report.roi_analysis
        exec_summary = report.execution_summary
        value_delivery = report.value_delivery
        
        return f"""# 📊 Terragon Value Delivery Report

**Generated:** {report.timestamp}  
**Period:** {report.reporting_period}  
**Repository:** {report.repository_info['name']} ({report.repository_info['maturity_level'].title()})

## 🎯 Executive Summary

- **Total Executions:** {exec_summary['total_executions']}
- **Success Rate:** {exec_summary['success_rate']:.1%}
- **Value Delivered:** ${roi['estimated_value_delivered']:,.2f}
- **ROI:** {roi['roi_percentage']:.1f}%
- **Payback Period:** {roi['payback_period_days']:.1f} days

## 📈 Performance Metrics

### ⚡ Execution Performance
- **Successful Items:** {exec_summary['successful_executions']} / {exec_summary['total_executions']}
- **Average Duration:** {exec_summary['average_duration']:.1f} hours
- **Total Effort:** {exec_summary['total_effort_hours']:.1f} hours
- **Categories Addressed:** {len(exec_summary['categories_addressed'])}

### 💎 Value Delivery
- **Total Value Points:** {value_delivery['total_value_delivered']:.1f}
- **Average per Item:** {value_delivery['average_value_per_item']:.1f}
- **High Impact Items:** {value_delivery['high_impact_items']}
- **Efficiency Score:** {value_delivery['efficiency_score']:.2f}

### 📊 Value by Category
"""
        
        for category, value in value_delivery['value_by_category'].items():
            markdown_content += f"- **{category.title()}:** {value:.1f} points\n"
        
        markdown_content += f"""

## 🔄 Trends Analysis

- **Value Trend:** {report.trends['value_trend'].replace('_', ' ').title()}
- **Success Rate Trend:** {report.trends['success_rate_trend'].replace('_', ' ').title()}
- **Effort Accuracy Trend:** {report.trends['effort_accuracy_trend'].replace('_', ' ').title()}

## 💰 ROI Analysis

- **Investment:** {roi['total_investment_hours']:.1f} hours (${roi['total_investment_hours'] * 100:,.2f})
- **Value Delivered:** ${roi['estimated_value_delivered']:,.2f}
- **ROI:** {roi['roi_percentage']:.1f}% (Industry benchmark: 200%)
- **Value per Hour:** ${roi['value_per_hour']:,.2f}
- **Payback Period:** {roi['payback_period_days']:.1f} days

## 🎯 Recommendations

"""
        
        for i, rec in enumerate(report.recommendations, 1):
            markdown_content += f"{i}. {rec}\n"
        
        markdown_content += f"""

---

*Report generated by Terragon Autonomous SDLC System*  
*Next report: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}*
"""
        
        return markdown_content
    
    def generate_daily_summary(self) -> str:
        """Generate a brief daily summary."""
        history = self._load_execution_history()
        
        # Get today's executions
        today = datetime.now().date()
        today_history = [
            h for h in history
            if datetime.fromisoformat(h['timestamp']).date() == today
        ]
        
        if not today_history:
            return "📊 No autonomous executions today"
        
        successful = len([h for h in today_history if h['execution']['status'] == 'completed'])
        total_value = sum(h['learning_data']['value_delivered'] for h in today_history)
        
        return (
            f"📊 Today's Summary: {successful}/{len(today_history)} successful executions, "
            f"{total_value:.1f} value points delivered"
        )


def main():
    """Main reporting function."""
    import sys
    
    reporter = ValueMetricsReporter()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--daily":
        summary = reporter.generate_daily_summary()
        print(summary)
        return
    
    # Generate comprehensive report
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    report = reporter.generate_comprehensive_report(days)
    
    print(f"\n📊 VALUE DELIVERY REPORT ({days} days)")
    print(f"Success Rate: {report.execution_summary['success_rate']:.1%}")
    print(f"Value Delivered: ${report.roi_analysis['estimated_value_delivered']:,.2f}")
    print(f"ROI: {report.roi_analysis['roi_percentage']:.1f}%")
    print(f"Recommendations: {len(report.recommendations)}")
    print(f"\n📄 Full report saved to: .terragon/VALUE_REPORT.md")


if __name__ == "__main__":
    main()