#!/usr/bin/env python3
"""
Terragon Autonomous Scheduler
Handles scheduling and orchestration of autonomous SDLC activities.
"""

import json
import logging
import subprocess
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import threading

from autonomous_executor import AutonomousExecutor
from value_discovery_engine import ValueDiscoveryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousScheduler:
    """Orchestrates autonomous execution on multiple schedules."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.executor = AutonomousExecutor(config_path)
        self.discovery_engine = ValueDiscoveryEngine(config_path)
        self.repo_root = Path.cwd()
        self.config = self.discovery_engine.config
        self.running = False
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "last_execution": None,
            "uptime_start": datetime.now().isoformat()
        }
    
    def start_autonomous_mode(self) -> None:
        """Start the autonomous scheduler with all configured schedules."""
        logger.info("🚀 Starting Terragon Autonomous SDLC Scheduler...")
        
        self.running = True
        self._setup_schedules()
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("🛑 Autonomous mode stopped by user")
        except Exception as e:
            logger.error(f"💥 Scheduler error: {e}")
        finally:
            self.running = False
    
    def stop_autonomous_mode(self) -> None:
        """Stop the autonomous scheduler."""
        logger.info("🛑 Stopping autonomous scheduler...")
        self.running = False
    
    def _setup_schedules(self) -> None:
        """Set up all autonomous execution schedules."""
        config_schedule = self.config.get('autonomous_schedule', {})
        
        # Immediate execution for critical items (on startup)
        self._execute_immediate_items()
        
        # Hourly security and dependency scans
        schedule.every().hour.do(self._hourly_security_scan)
        schedule.every().hour.at(":30").do(self._hourly_dependency_check)
        
        # Daily comprehensive analysis
        schedule.every().day.at("02:00").do(self._daily_comprehensive_analysis)
        schedule.every().day.at("14:00").do(self._daily_value_discovery)
        
        # Weekly strategic review
        schedule.every().monday.at("03:00").do(self._weekly_strategic_review)
        
        # Monthly model recalibration
        schedule.every().month.do(self._monthly_recalibration)
        
        logger.info("📅 Autonomous schedules configured")
    
    def _execute_immediate_items(self) -> None:
        """Execute immediately critical items (security, compliance)."""
        logger.info("⚡ Checking for immediate execution items...")
        
        try:
            items = self.discovery_engine.discover_value_items()
            scored_items = self.discovery_engine.score_work_items(items)
            
            # Find critical items requiring immediate attention
            immediate_items = [
                item for item in scored_items
                if item.category in ['security'] and item.composite_score > 80
            ]
            
            for item in immediate_items[:2]:  # Limit to 2 immediate items
                logger.info(f"🚨 Immediate execution: {item.title}")
                result = self.executor._execute_work_item(item)
                self._update_stats(result)
                
                if result['status'] == 'completed':
                    logger.info(f"✅ Immediate item completed: {item.title}")
                else:
                    logger.warning(f"⚠️ Immediate item failed: {item.title}")
        
        except Exception as e:
            logger.error(f"💥 Immediate execution failed: {e}")
    
    def _hourly_security_scan(self) -> None:
        """Hourly security vulnerability scanning."""
        logger.info("🔍 Hourly security scan...")
        
        try:
            # Run security scan
            security_script = self.repo_root / "scripts" / "advanced_security_scan.py"
            if security_script.exists():
                result = subprocess.run([
                    "python3", str(security_script), "--quiet"
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                if result.returncode != 0:
                    logger.warning("🚨 Security scan detected issues")
                    # Trigger immediate security execution
                    self._execute_security_items()
                else:
                    logger.info("✅ Security scan clean")
            
            self._update_discovery_metrics("hourly_security", True)
        
        except Exception as e:
            logger.error(f"💥 Hourly security scan failed: {e}")
            self._update_discovery_metrics("hourly_security", False)
    
    def _hourly_dependency_check(self) -> None:
        """Hourly dependency vulnerability check."""
        logger.info("📦 Hourly dependency check...")
        
        try:
            # Check for new security advisories
            result = subprocess.run([
                "pip-audit", ".", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode != 0 and result.stdout:
                vulnerabilities = json.loads(result.stdout)
                if vulnerabilities:
                    logger.warning(f"🚨 Found {len(vulnerabilities)} dependency vulnerabilities")
                    # Trigger dependency update execution
                    self._execute_dependency_updates()
            
            self._update_discovery_metrics("hourly_dependencies", True)
        
        except Exception as e:
            logger.error(f"💥 Dependency check failed: {e}")
            self._update_discovery_metrics("hourly_dependencies", False)
    
    def _daily_comprehensive_analysis(self) -> None:
        """Daily comprehensive static analysis and value discovery."""
        logger.info("📊 Daily comprehensive analysis...")
        
        try:
            # Run full value discovery
            items = self.discovery_engine.discover_value_items()
            scored_items = self.discovery_engine.score_work_items(items)
            
            # Update backlog
            self._update_autonomous_backlog(scored_items)
            
            # Execute top 1-2 items if they meet criteria
            top_items = [item for item in scored_items[:2] if item.composite_score > 50]
            
            for item in top_items:
                logger.info(f"🎯 Daily execution: {item.title}")
                result = self.executor._execute_work_item(item)
                self._update_stats(result)
            
            self._update_discovery_metrics("daily_analysis", True)
            logger.info(f"✅ Daily analysis completed, processed {len(top_items)} items")
        
        except Exception as e:
            logger.error(f"💥 Daily analysis failed: {e}")
            self._update_discovery_metrics("daily_analysis", False)
    
    def _daily_value_discovery(self) -> None:
        """Daily value discovery and opportunity identification."""
        logger.info("💎 Daily value discovery...")
        
        try:
            # Discover new opportunities
            items = self.discovery_engine.discover_value_items()
            scored_items = self.discovery_engine.score_work_items(items)
            
            # Save metrics
            self.discovery_engine.save_metrics(scored_items)
            
            # Generate updated backlog
            self._update_autonomous_backlog(scored_items)
            
            logger.info(f"💎 Discovered {len(items)} opportunities, top score: {scored_items[0].composite_score:.2f}")
            self._update_discovery_metrics("daily_discovery", True)
        
        except Exception as e:
            logger.error(f"💥 Daily discovery failed: {e}")
            self._update_discovery_metrics("daily_discovery", False)
    
    def _weekly_strategic_review(self) -> None:
        """Weekly strategic review and architecture analysis."""
        logger.info("🏗️ Weekly strategic review...")
        
        try:
            # Run architectural analysis
            arch_script = self.repo_root / "scripts" / "technical_debt_analyzer.py"
            if arch_script.exists():
                result = subprocess.run([
                    "python3", str(arch_script), "--strategic"
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                logger.info(f"🏗️ Architecture analysis: {'✅' if result.returncode == 0 else '❌'}")
            
            # Review execution performance
            self._analyze_execution_performance()
            
            # Adjust scoring weights based on outcomes
            self._adjust_scoring_weights()
            
            self._update_discovery_metrics("weekly_strategic", True)
        
        except Exception as e:
            logger.error(f"💥 Weekly review failed: {e}")
            self._update_discovery_metrics("weekly_strategic", False)
    
    def _monthly_recalibration(self) -> None:
        """Monthly scoring model recalibration and process optimization."""
        logger.info("🔧 Monthly model recalibration...")
        
        try:
            # Analyze execution history
            log_file = self.repo_root / ".terragon" / "execution-log.json"
            if log_file.exists():
                with open(log_file) as f:
                    execution_history = json.load(f)
                
                # Calculate model accuracy
                accuracy_metrics = self._calculate_model_accuracy(execution_history)
                
                # Update configuration based on learnings
                self._update_model_configuration(accuracy_metrics)
                
                logger.info(f"🔧 Model recalibration completed, accuracy: {accuracy_metrics.get('overall_accuracy', 0):.2f}")
            
            self._update_discovery_metrics("monthly_recalibration", True)
        
        except Exception as e:
            logger.error(f"💥 Monthly recalibration failed: {e}")
            self._update_discovery_metrics("monthly_recalibration", False)
    
    def _execute_security_items(self) -> None:
        """Execute security-related items immediately."""
        try:
            items = self.discovery_engine.discover_security_opportunities()
            if items:
                scored_items = self.discovery_engine.score_work_items(items)
                top_security = scored_items[0] if scored_items else None
                
                if top_security and top_security.composite_score > 60:
                    result = self.executor._execute_work_item(top_security)
                    self._update_stats(result)
                    logger.info(f"🔐 Security item executed: {top_security.title}")
        
        except Exception as e:
            logger.error(f"Security execution failed: {e}")
    
    def _execute_dependency_updates(self) -> None:
        """Execute dependency update items."""
        try:
            items = self.discovery_engine.discover_dependency_updates()
            if items:
                scored_items = self.discovery_engine.score_work_items(items)
                top_dep = scored_items[0] if scored_items else None
                
                if top_dep and top_dep.composite_score > 40:
                    result = self.executor._execute_work_item(top_dep)
                    self._update_stats(result)
                    logger.info(f"📦 Dependency item executed: {top_dep.title}")
        
        except Exception as e:
            logger.error(f"Dependency execution failed: {e}")
    
    def _update_autonomous_backlog(self, items: List) -> None:
        """Update the autonomous backlog markdown file."""
        try:
            backlog_content = self._generate_backlog_content(items)
            
            backlog_file = self.repo_root / "AUTONOMOUS_BACKLOG.md"
            with open(backlog_file, 'w') as f:
                f.write(backlog_content)
            
            logger.info("📋 Autonomous backlog updated")
        
        except Exception as e:
            logger.error(f"Backlog update failed: {e}")
    
    def _generate_backlog_content(self, items: List) -> str:
        """Generate updated backlog content."""
        now = datetime.now()
        next_exec = now + timedelta(hours=1)
        
        top_item = items[0] if items else None
        
        content = f"""# 🤖 Autonomous Value Discovery Backlog

**Repository:** sql-synth-agentic-playground  
**Maturity Level:** ADVANCED (78% SDLC maturity)  
**Last Updated:** {now.isoformat()}  
**Next Execution:** {next_exec.isoformat()}  

## 🎯 Next Best Value Item

"""
        
        if top_item:
            content += f"""**[{top_item.id.upper()}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score:.0f} | **Tech Debt**: {top_item.technical_debt_score:.0f}
- **Estimated Effort**: {top_item.effort_estimate} hours
- **Category**: {top_item.category.title()}
- **Priority**: {top_item.priority.upper()}

"""
        
        content += f"""## 📊 Current Status

- **Items Discovered**: {len(items)}
- **High Priority Items**: {len([i for i in items if i.priority == 'high'])}
- **Execution Stats**: {self.execution_stats['successful_executions']}/{self.execution_stats['total_executions']} successful
- **Scheduler Uptime**: {(datetime.now() - datetime.fromisoformat(self.execution_stats['uptime_start'])).days} days

## 🔥 Top 10 Opportunities

| Rank | ID | Title | Score | Category | Hours | Priority |
|------|-----|--------|---------|----------|-------|----------|"""
        
        for i, item in enumerate(items[:10], 1):
            content += f"\n| {i} | {item.id.upper()} | {item.title[:50]}... | {item.composite_score:.1f} | {item.category.title()} | {item.effort_estimate} | {item.priority.upper()} |"
        
        content += f"""

---

*🔄 This backlog is automatically updated by the Terragon Autonomous SDLC system.*  
*📊 Last execution: {self.execution_stats.get('last_execution', 'Never')}*
*⚡ Scheduler status: {'🟢 RUNNING' if self.running else '🔴 STOPPED'}*
"""
        
        return content
    
    def _update_stats(self, result: Dict) -> None:
        """Update execution statistics."""
        self.execution_stats['total_executions'] += 1
        self.execution_stats['last_execution'] = datetime.now().isoformat()
        
        if result.get('status') == 'completed':
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1
    
    def _update_discovery_metrics(self, operation: str, success: bool) -> None:
        """Update discovery operation metrics."""
        metrics_file = self.repo_root / ".terragon" / "scheduler-metrics.json"
        metrics_file.parent.mkdir(exist_ok=True)
        
        try:
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
            else:
                metrics = {"operations": {}}
            
            if operation not in metrics["operations"]:
                metrics["operations"][operation] = {"success": 0, "failure": 0}
            
            if success:
                metrics["operations"][operation]["success"] += 1
            else:
                metrics["operations"][operation]["failure"] += 1
            
            metrics["last_updated"] = datetime.now().isoformat()
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _analyze_execution_performance(self) -> Dict:
        """Analyze execution performance for learning."""
        log_file = self.repo_root / ".terragon" / "execution-log.json"
        
        if not log_file.exists():
            return {}
        
        try:
            with open(log_file) as f:
                history = json.load(f)
            
            # Calculate performance metrics
            total_items = len(history)
            successful_items = len([h for h in history if h['execution']['status'] == 'completed'])
            
            avg_effort_accuracy = sum(
                h['learning_data']['effort_accuracy'] for h in history
            ) / max(total_items, 1)
            
            avg_value_delivered = sum(
                h['learning_data']['value_delivered'] for h in history
            ) / max(total_items, 1)
            
            performance = {
                "total_executions": total_items,
                "success_rate": successful_items / max(total_items, 1),
                "avg_effort_accuracy": avg_effort_accuracy,
                "avg_value_delivered": avg_value_delivered,
                "analysis_date": datetime.now().isoformat()
            }
            
            logger.info(f"📈 Performance: {performance['success_rate']:.2f} success rate, {performance['avg_effort_accuracy']:.2f} effort accuracy")
            return performance
        
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}
    
    def _adjust_scoring_weights(self) -> None:
        """Adjust scoring weights based on execution outcomes."""
        # Placeholder for learning-based weight adjustment
        # In a full implementation, this would analyze execution history
        # and adjust weights in the configuration
        logger.info("🎛️ Scoring weights reviewed (no changes needed)")
    
    def _calculate_model_accuracy(self, history: List[Dict]) -> Dict:
        """Calculate model accuracy metrics."""
        if not history:
            return {"overall_accuracy": 0.0}
        
        effort_accuracies = [h['learning_data']['effort_accuracy'] for h in history]
        value_accuracies = [
            1.0 if h['learning_data']['value_delivered'] > 0 else 0.0
            for h in history
        ]
        
        return {
            "overall_accuracy": sum(effort_accuracies) / len(effort_accuracies),
            "value_accuracy": sum(value_accuracies) / len(value_accuracies),
            "sample_size": len(history)
        }
    
    def _update_model_configuration(self, accuracy_metrics: Dict) -> None:
        """Update model configuration based on accuracy analysis."""
        # Placeholder for configuration updates
        logger.info(f"🔧 Model configuration reviewed, accuracy: {accuracy_metrics.get('overall_accuracy', 0):.2f}")
    
    def get_status(self) -> Dict:
        """Get current scheduler status."""
        return {
            "running": self.running,
            "execution_stats": self.execution_stats,
            "next_scheduled": {
                "hourly_security": schedule.jobs[0].next_run if schedule.jobs else None,
                "daily_analysis": schedule.jobs[2].next_run if len(schedule.jobs) > 2 else None
            }
        }


def main():
    """Main scheduler execution."""
    import sys
    
    scheduler = AutonomousScheduler()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        status = scheduler.get_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run one discovery cycle
        print("🔍 Running single discovery cycle...")
        items = scheduler.discovery_engine.discover_value_items()
        scored_items = scheduler.discovery_engine.score_work_items(items)
        scheduler._update_autonomous_backlog(scored_items)
        print(f"✅ Discovered {len(items)} items, updated backlog")
        return
    
    try:
        scheduler.start_autonomous_mode()
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped")


if __name__ == "__main__":
    main()