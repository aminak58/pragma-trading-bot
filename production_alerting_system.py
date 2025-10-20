#!/usr/bin/env python3
"""
Production Alerting System
==========================

This module implements a comprehensive alerting system for live trading including:
- Email notifications
- Webhook alerts
- SMS notifications (optional)
- Custom alert rules
- Alert escalation

Author: Pragma Trading Bot Team
Date: 2025-10-20
"""

import smtplib
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
from collections import deque

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # trading, risk, system
    metric: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    level: str  # info, warning, critical
    enabled: bool = True
    cooldown: int = 300  # seconds
    escalation: bool = False

@dataclass
class AlertNotification:
    """Alert notification data"""
    timestamp: str
    rule_name: str
    level: str
    category: str
    message: str
    metric_value: float
    threshold: float
    data: Dict[str, Any]

@dataclass
class AlertConfig:
    """Alert configuration"""
    email: Dict[str, Any]
    webhook: Dict[str, Any]
    sms: Dict[str, Any]
    rules: List[AlertRule]
    escalation: Dict[str, Any]

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.get('enabled', False)
        
        if self.enabled:
            self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
            self.smtp_port = config.get('smtp_port', 587)
            self.username = config.get('username', '')
            self.password = config.get('password', '')
            self.from_email = config.get('from_email', self.username)
            self.to_emails = config.get('to_emails', [])
    
    def send_alert(self, notification: AlertNotification) -> bool:
        """Send email alert"""
        if not self.enabled:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{notification.level.upper()}] {notification.category} Alert"
            
            # Create body
            body = f"""
Alert Details:
- Time: {notification.timestamp}
- Rule: {notification.rule_name}
- Level: {notification.level.upper()}
- Category: {notification.category}
- Message: {notification.message}
- Metric Value: {notification.metric_value}
- Threshold: {notification.threshold}

Additional Data:
{json.dumps(notification.data, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            self.logger.info(f"Email alert sent: {notification.rule_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            return False
    
    def send_summary(self, alerts: List[AlertNotification]) -> bool:
        """Send alert summary email"""
        if not self.enabled or not alerts:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"Trading Bot Alert Summary - {len(alerts)} alerts"
            
            # Create summary body
            body = f"""
Alert Summary ({len(alerts)} alerts):

"""
            
            for alert in alerts[-10:]:  # Last 10 alerts
                body += f"- [{alert.level.upper()}] {alert.category}: {alert.message}\n"
            
            body += f"\nTotal alerts in this period: {len(alerts)}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            self.logger.info(f"Alert summary email sent: {len(alerts)} alerts")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending summary email: {e}")
            return False

class WebhookNotifier:
    """Webhook notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = config.get('enabled', False)
        
        if self.enabled:
            self.webhook_url = config.get('url', '')
            self.timeout = config.get('timeout', 10)
    
    def send_alert(self, notification: AlertNotification) -> bool:
        """Send webhook alert"""
        if not self.enabled:
            return False
        
        try:
            # Prepare payload
            payload = {
                "timestamp": notification.timestamp,
                "level": notification.level,
                "category": notification.category,
                "rule_name": notification.rule_name,
                "message": notification.message,
                "metric_value": notification.metric_value,
                "threshold": notification.threshold,
                "data": notification.data
            }
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook alert sent: {notification.rule_name}")
                return True
            else:
                self.logger.error(f"Webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
            return False
    
    def send_summary(self, alerts: List[AlertNotification]) -> bool:
        """Send alert summary webhook"""
        if not self.enabled or not alerts:
            return False
        
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "type": "summary",
                "alert_count": len(alerts),
                "alerts": [asdict(alert) for alert in alerts[-10:]]  # Last 10 alerts
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook summary sent: {len(alerts)} alerts")
                return True
            else:
                self.logger.error(f"Webhook summary failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending webhook summary: {e}")
            return False

class AlertRuleEngine:
    """Alert rule evaluation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = []
        self.last_triggered = {}  # Track last trigger time for cooldown
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.rules.append(rule)
        self.logger.info(f"Added alert rule: {rule.name}")
    
    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[AlertNotification]:
        """Evaluate all rules against current metrics"""
        notifications = []
        current_time = time.time()
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.name in self.last_triggered:
                time_since_last = current_time - self.last_triggered[rule.name]
                if time_since_last < rule.cooldown:
                    continue
            
            # Get metric value
            metric_value = self._get_metric_value(metrics, rule.condition, rule.metric)
            if metric_value is None:
                continue
            
            # Evaluate condition
            if self._evaluate_condition(metric_value, rule.operator, rule.threshold):
                notification = AlertNotification(
                    timestamp=datetime.now().isoformat(),
                    rule_name=rule.name,
                    level=rule.level,
                    category=rule.condition,
                    message=f"{rule.metric} {rule.operator} {rule.threshold} (value: {metric_value})",
                    metric_value=metric_value,
                    threshold=rule.threshold,
                    data=metrics.get(rule.condition, {})
                )
                
                notifications.append(notification)
                self.last_triggered[rule.name] = current_time
                
                self.logger.info(f"Alert triggered: {rule.name}")
        
        return notifications
    
    def _get_metric_value(self, metrics: Dict[str, Any], condition: str, metric: str) -> Optional[float]:
        """Get metric value from metrics dictionary"""
        try:
            condition_data = metrics.get(condition, {})
            if isinstance(condition_data, dict):
                return condition_data.get(metric)
            elif hasattr(condition_data, metric):
                return getattr(condition_data, metric)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error getting metric value: {e}")
            return None
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate condition"""
        try:
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            elif operator == '!=':
                return value != threshold
            else:
                self.logger.error(f"Unknown operator: {operator}")
                return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False

class AlertEscalationManager:
    """Alert escalation manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.escalation_history = deque(maxlen=1000)
    
    def should_escalate(self, notification: AlertNotification) -> bool:
        """Check if alert should be escalated"""
        try:
            # Check if escalation is enabled
            if not self.config.get('enabled', False):
                return False
            
            # Check escalation rules
            escalation_rules = self.config.get('rules', [])
            for rule in escalation_rules:
                if self._matches_escalation_rule(notification, rule):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking escalation: {e}")
            return False
    
    def _matches_escalation_rule(self, notification: AlertNotification, rule: Dict[str, Any]) -> bool:
        """Check if notification matches escalation rule"""
        try:
            # Check level
            if rule.get('level') and notification.level != rule['level']:
                return False
            
            # Check category
            if rule.get('category') and notification.category != rule['category']:
                return False
            
            # Check frequency (number of alerts in time window)
            if rule.get('frequency'):
                frequency_config = rule['frequency']
                time_window = frequency_config.get('time_window', 3600)  # 1 hour
                max_alerts = frequency_config.get('max_alerts', 5)
                
                cutoff_time = datetime.now() - timedelta(seconds=time_window)
                recent_alerts = [
                    alert for alert in self.escalation_history
                    if datetime.fromisoformat(alert['timestamp']) > cutoff_time
                    and alert['rule_name'] == notification.rule_name
                ]
                
                if len(recent_alerts) >= max_alerts:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error matching escalation rule: {e}")
            return False
    
    def escalate_alert(self, notification: AlertNotification) -> bool:
        """Escalate alert"""
        try:
            escalation_data = {
                'timestamp': datetime.now().isoformat(),
                'original_alert': asdict(notification),
                'escalation_reason': 'Frequency threshold exceeded'
            }
            
            self.escalation_history.append(escalation_data)
            
            self.logger.warning(f"Alert escalated: {notification.rule_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error escalating alert: {e}")
            return False

class ProductionAlertingSystem:
    """Main production alerting system"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.email_notifier = EmailNotifier(config.email)
        self.webhook_notifier = WebhookNotifier(config.webhook)
        self.rule_engine = AlertRuleEngine()
        self.escalation_manager = AlertEscalationManager(config.escalation)
        
        # Add rules
        for rule in config.rules:
            self.rule_engine.add_rule(rule)
        
        # Alert history
        self.alert_history = deque(maxlen=10000)
        self.summary_thread = None
        self.is_running = False
    
    def process_metrics(self, metrics: Dict[str, Any]) -> List[AlertNotification]:
        """Process metrics and generate alerts"""
        try:
            # Evaluate rules
            notifications = self.rule_engine.evaluate_rules(metrics)
            
            # Process each notification
            for notification in notifications:
                # Store in history
                self.alert_history.append(asdict(notification))
                
                # Send notifications
                self.email_notifier.send_alert(notification)
                self.webhook_notifier.send_alert(notification)
                
                # Check for escalation
                if self.escalation_manager.should_escalate(notification):
                    self.escalation_manager.escalate_alert(notification)
            
            return notifications
            
        except Exception as e:
            self.logger.error(f"Error processing metrics: {e}")
            return []
    
    def start_summary_scheduler(self, interval: int = 3600):  # 1 hour
        """Start summary email scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self.summary_thread = threading.Thread(
            target=self._summary_loop,
            args=(interval,),
            daemon=True
        )
        self.summary_thread.start()
        
        self.logger.info(f"Alert summary scheduler started with {interval}s interval")
    
    def stop_summary_scheduler(self):
        """Stop summary scheduler"""
        self.is_running = False
        if self.summary_thread:
            self.summary_thread.join(timeout=5)
        
        self.logger.info("Alert summary scheduler stopped")
    
    def _summary_loop(self, interval: int):
        """Summary loop"""
        while self.is_running:
            try:
                time.sleep(interval)
                
                # Get recent alerts
                recent_alerts = list(self.alert_history)
                if recent_alerts:
                    # Send summary
                    self.email_notifier.send_summary(recent_alerts)
                    self.webhook_notifier.send_summary(recent_alerts)
                
            except Exception as e:
                self.logger.error(f"Error in summary loop: {e}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        try:
            alerts = list(self.alert_history)
            
            # Count by level
            level_counts = {}
            for alert in alerts:
                level = alert.get('level', 'unknown')
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Count by category
            category_counts = {}
            for alert in alerts:
                category = alert.get('category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count by rule
            rule_counts = {}
            for alert in alerts:
                rule = alert.get('rule_name', 'unknown')
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            
            return {
                "total_alerts": len(alerts),
                "level_counts": level_counts,
                "category_counts": category_counts,
                "rule_counts": rule_counts,
                "recent_alerts": alerts[-10:] if alerts else [],
                "escalation_count": len(self.escalation_manager.escalation_history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {e}")
            return {"error": str(e)}

def create_default_alert_config() -> AlertConfig:
    """Create default alert configuration"""
    rules = [
        AlertRule(
            name="high_drawdown",
            condition="trading",
            metric="current_drawdown",
            operator="<",
            threshold=-0.10,
            level="warning",
            cooldown=1800  # 30 minutes
        ),
        AlertRule(
            name="critical_drawdown",
            condition="trading",
            metric="current_drawdown",
            operator="<",
            threshold=-0.15,
            level="critical",
            cooldown=600  # 10 minutes
        ),
        AlertRule(
            name="high_exposure",
            condition="risk",
            metric="portfolio_heat",
            operator=">",
            threshold=0.8,
            level="warning",
            cooldown=1800
        ),
        AlertRule(
            name="critical_exposure",
            condition="risk",
            metric="portfolio_heat",
            operator=">",
            threshold=0.9,
            level="critical",
            cooldown=600
        ),
        AlertRule(
            name="high_cpu",
            condition="system",
            metric="cpu_percent",
            operator=">",
            threshold=80,
            level="warning",
            cooldown=1800
        ),
        AlertRule(
            name="critical_cpu",
            condition="system",
            metric="cpu_percent",
            operator=">",
            threshold=95,
            level="critical",
            cooldown=600
        ),
        AlertRule(
            name="high_memory",
            condition="system",
            metric="memory_percent",
            operator=">",
            threshold=85,
            level="warning",
            cooldown=1800
        ),
        AlertRule(
            name="critical_memory",
            condition="system",
            metric="memory_percent",
            operator=">",
            threshold=95,
            level="critical",
            cooldown=600
        )
    ]
    
    config = AlertConfig(
        email={
            "enabled": True,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your_email@gmail.com",
            "password": "your_app_password",
            "from_email": "your_email@gmail.com",
            "to_emails": ["admin@yourcompany.com"]
        },
        webhook={
            "enabled": False,
            "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "timeout": 10
        },
        sms={
            "enabled": False,
            "provider": "twilio",
            "account_sid": "",
            "auth_token": "",
            "from_number": "",
            "to_numbers": []
        },
        rules=rules,
        escalation={
            "enabled": True,
            "rules": [
                {
                    "level": "critical",
                    "frequency": {
                        "time_window": 3600,  # 1 hour
                        "max_alerts": 3
                    }
                }
            ]
        }
    )
    
    return config

def main():
    """Test alerting system"""
    logging.basicConfig(level=logging.INFO)
    
    print("Production Alerting System Test")
    print("=" * 50)
    
    # Create alert configuration
    print("\n1. Creating alert configuration...")
    config = create_default_alert_config()
    print(f"   Alert rules created: {len(config.rules)}")
    print(f"   Email notifications: {'enabled' if config.email['enabled'] else 'disabled'}")
    print(f"   Webhook notifications: {'enabled' if config.webhook['enabled'] else 'disabled'}")
    
    # Create alerting system
    print("\n2. Creating alerting system...")
    alerting_system = ProductionAlertingSystem(config)
    print("   Alerting system created successfully")
    
    # Test with sample metrics
    print("\n3. Testing with sample metrics...")
    
    # Test trading metrics
    trading_metrics = {
        "trading": {
            "current_drawdown": -0.12,  # Should trigger high_drawdown
            "portfolio_heat": 0.85,     # Should trigger high_exposure
            "win_rate": 0.55
        },
        "risk": {
            "portfolio_heat": 0.85,
            "concentration_risk": 0.3
        },
        "system": {
            "cpu_percent": 45.0,
            "memory_percent": 70.0,
            "disk_percent": 80.0
        }
    }
    
    notifications = alerting_system.process_metrics(trading_metrics)
    print(f"   Notifications generated: {len(notifications)}")
    
    for notification in notifications:
        print(f"   - [{notification.level.upper()}] {notification.category}: {notification.message}")
    
    # Test system metrics
    print("\n4. Testing with system metrics...")
    system_metrics = {
        "trading": {
            "current_drawdown": -0.05,
            "portfolio_heat": 0.5
        },
        "risk": {
            "portfolio_heat": 0.5,
            "concentration_risk": 0.2
        },
        "system": {
            "cpu_percent": 85.0,  # Should trigger high_cpu
            "memory_percent": 90.0,  # Should trigger high_memory
            "disk_percent": 85.0
        }
    }
    
    notifications = alerting_system.process_metrics(system_metrics)
    print(f"   Notifications generated: {len(notifications)}")
    
    for notification in notifications:
        print(f"   - [{notification.level.upper()}] {notification.category}: {notification.message}")
    
    # Get alert statistics
    print("\n5. Getting alert statistics...")
    stats = alerting_system.get_alert_statistics()
    
    print(f"   Total alerts: {stats['total_alerts']}")
    print(f"   Level counts: {stats['level_counts']}")
    print(f"   Category counts: {stats['category_counts']}")
    print(f"   Rule counts: {stats['rule_counts']}")
    print(f"   Escalation count: {stats['escalation_count']}")
    
    # Test summary scheduler
    print("\n6. Testing summary scheduler...")
    alerting_system.start_summary_scheduler(interval=10)  # 10 seconds for testing
    print("   Summary scheduler started")
    
    time.sleep(15)  # Let it run for a bit
    
    alerting_system.stop_summary_scheduler()
    print("   Summary scheduler stopped")
    
    print("\nProduction Alerting System test completed!")
    print("\nAlerting system features:")
    print("  - Configurable alert rules")
    print("  - Email notifications")
    print("  - Webhook notifications")
    print("  - Alert escalation")
    print("  - Cooldown periods")
    print("  - Summary reports")
    print("  - Alert statistics")

if __name__ == "__main__":
    main()
