#!/usr/bin/env python3
"""
Monitoring Script - Pragma Trading Bot

This script provides comprehensive monitoring of the Pragma Trading Bot.
Monitors system health, trading performance, and alerts on issues.

Usage:
    python scripts/monitor.py --mode production
    python scripts/monitor.py --mode dry-run --dashboard
    python scripts/monitor.py --health-check
"""

import argparse
import sys
import os
import json
import time
import psutil
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitors system resources and health"""
    
    def __init__(self):
        """Initialize system monitor"""
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'disk_usage': 90.0,
            'load_average': 4.0
        }
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percentage': memory.percent,
            'free': memory.free
        }
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percentage': (disk.used / disk.total) * 100
        }
    
    def get_load_average(self) -> List[float]:
        """Get system load average"""
        return list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
    
    def get_network_io(self) -> Dict[str, Any]:
        """Get network I/O statistics"""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': self.get_cpu_usage(),
            'memory': self.get_memory_usage(),
            'disk': self.get_disk_usage(),
            'load_average': self.get_load_average(),
            'network': self.get_network_io(),
            'alerts': []
        }
        
        # Check thresholds
        if health['cpu_usage'] > self.thresholds['cpu_usage']:
            health['alerts'].append({
                'level': 'WARNING',
                'message': f"High CPU usage: {health['cpu_usage']:.1f}%"
            })
        
        if health['memory']['percentage'] > self.thresholds['memory_usage']:
            health['alerts'].append({
                'level': 'WARNING',
                'message': f"High memory usage: {health['memory']['percentage']:.1f}%"
            })
        
        if health['disk']['percentage'] > self.thresholds['disk_usage']:
            health['alerts'].append({
                'level': 'CRITICAL',
                'message': f"High disk usage: {health['disk']['percentage']:.1f}%"
            })
        
        load_avg = health['load_average'][0] if health['load_average'] else 0
        if load_avg > self.thresholds['load_average']:
            health['alerts'].append({
                'level': 'WARNING',
                'message': f"High load average: {load_avg:.2f}"
            })
        
        return health

class TradingMonitor:
    """Monitors trading performance and status"""
    
    def __init__(self, config_path: str):
        """
        Initialize trading monitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def check_freqtrade_status(self) -> Dict[str, Any]:
        """Check Freqtrade status"""
        try:
            # Check if Freqtrade process is running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'freqtrade' in ' '.join(proc.info['cmdline'] or []):
                    return {
                        'running': True,
                        'pid': proc.info['pid'],
                        'status': 'active'
                    }
            
            return {'running': False, 'status': 'stopped'}
            
        except Exception as e:
            logger.error(f"Error checking Freqtrade status: {e}")
            return {'running': False, 'status': 'error', 'error': str(e)}
    
    def check_api_server(self) -> Dict[str, Any]:
        """Check API server status"""
        try:
            api_config = self.config.get('api_server', {})
            if not api_config.get('enabled', False):
                return {'enabled': False, 'status': 'disabled'}
            
            port = api_config.get('listen_port', 8080)
            url = f"http://localhost:{port}/api/v1/ping"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return {
                    'enabled': True,
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'enabled': True,
                    'status': 'unhealthy',
                    'status_code': response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'enabled': True,
                'status': 'unreachable',
                'error': str(e)
            }
        except Exception as e:
            return {
                'enabled': True,
                'status': 'error',
                'error': str(e)
            }
    
    def check_trading_performance(self) -> Dict[str, Any]:
        """Check trading performance metrics"""
        try:
            # This would integrate with actual trading data
            # For now, return mock data
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'daily_pnl': 0.0,
                'open_positions': 0
            }
            
        except Exception as e:
            logger.error(f"Error checking trading performance: {e}")
            return {'error': str(e)}
    
    def check_risk_metrics(self) -> Dict[str, Any]:
        """Check risk management metrics"""
        try:
            # This would integrate with risk management data
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'position_sizes': [],
                'circuit_breakers': {
                    'enabled': True,
                    'status': 'normal'
                },
                'kelly_utilization': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error checking risk metrics: {e}")
            return {'error': str(e)}

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.alert_config = config.get('alerts', {})
        self.telegram_config = config.get('telegram', {})
    
    def send_alert(self, level: str, message: str, data: Optional[Dict] = None):
        """
        Send alert notification.
        
        Args:
            level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            message: Alert message
            data: Additional data
        """
        try:
            # Log alert
            logger.log(
                getattr(logging, level),
                f"ALERT {level}: {message}"
            )
            
            # Send Telegram alert if enabled
            if self.alert_config.get('telegram', False) and self.telegram_config.get('enabled', False):
                self._send_telegram_alert(level, message, data)
            
            # Send email alert if enabled
            if self.alert_config.get('email', False):
                self._send_email_alert(level, message, data)
            
            # Send webhook alert if enabled
            if self.alert_config.get('webhook', False):
                self._send_webhook_alert(level, message, data)
                
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _send_telegram_alert(self, level: str, message: str, data: Optional[Dict] = None):
        """Send Telegram alert"""
        try:
            token = self.telegram_config.get('token')
            chat_id = self.telegram_config.get('chat_id')
            
            if not token or not chat_id:
                return
            
            # Format message
            emoji = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }.get(level, '📢')
            
            text = f"{emoji} *{level}*\n\n{message}"
            
            if data:
                text += f"\n\n```json\n{json.dumps(data, indent=2)}\n```"
            
            # Send message
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def _send_email_alert(self, level: str, message: str, data: Optional[Dict] = None):
        """Send email alert"""
        # Email implementation would go here
        logger.info(f"Email alert: {level} - {message}")
    
    def _send_webhook_alert(self, level: str, message: str, data: Optional[Dict] = None):
        """Send webhook alert"""
        # Webhook implementation would go here
        logger.info(f"Webhook alert: {level} - {message}")

class MonitoringDashboard:
    """Provides monitoring dashboard functionality"""
    
    def __init__(self, config_path: str):
        """
        Initialize monitoring dashboard.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.system_monitor = SystemMonitor()
        self.trading_monitor = TradingMonitor(config_path)
        self.alert_manager = AlertManager(self.config)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'system': self.system_monitor.check_system_health(),
            'freqtrade': self.trading_monitor.check_freqtrade_status(),
            'api_server': self.trading_monitor.check_api_server(),
            'trading': self.trading_monitor.check_trading_performance(),
            'risk': self.trading_monitor.check_risk_metrics()
        }
    
    def print_dashboard(self):
        """Print monitoring dashboard to console"""
        data = self.get_dashboard_data()
        
        print("\n" + "="*80)
        print("PRAGMA TRADING BOT - MONITORING DASHBOARD")
        print("="*80)
        print(f"Timestamp: {data['timestamp']}")
        print()
        
        # System Health
        system = data['system']
        print("SYSTEM HEALTH:")
        print(f"  CPU Usage: {system['cpu_usage']:.1f}%")
        print(f"  Memory Usage: {system['memory']['percentage']:.1f}%")
        print(f"  Disk Usage: {system['disk']['percentage']:.1f}%")
        print(f"  Load Average: {system['load_average'][0]:.2f}")
        
        if system['alerts']:
            print("  ALERTS:")
            for alert in system['alerts']:
                print(f"    {alert['level']}: {alert['message']}")
        print()
        
        # Freqtrade Status
        freqtrade = data['freqtrade']
        print("FREQTRADE STATUS:")
        print(f"  Running: {freqtrade['running']}")
        print(f"  Status: {freqtrade['status']}")
        if 'pid' in freqtrade:
            print(f"  PID: {freqtrade['pid']}")
        print()
        
        # API Server
        api = data['api_server']
        print("API SERVER:")
        print(f"  Enabled: {api['enabled']}")
        print(f"  Status: {api['status']}")
        if 'response_time' in api:
            print(f"  Response Time: {api['response_time']:.3f}s")
        print()
        
        # Trading Performance
        trading = data['trading']
        print("TRADING PERFORMANCE:")
        print(f"  Total Trades: {trading['total_trades']}")
        print(f"  Win Rate: {trading['win_rate']:.1%}")
        print(f"  Total Profit: {trading['total_profit']:.2f}")
        print(f"  Max Drawdown: {trading['max_drawdown']:.2%}")
        print(f"  Open Positions: {trading['open_positions']}")
        print()
        
        # Risk Metrics
        risk = data['risk']
        print("RISK METRICS:")
        print(f"  Current Drawdown: {risk['current_drawdown']:.2%}")
        print(f"  Max Drawdown: {risk['max_drawdown']:.2%}")
        print(f"  Circuit Breakers: {risk['circuit_breakers']['status']}")
        print(f"  Kelly Utilization: {risk['kelly_utilization']:.1%}")
        print()
        
        print("="*80)
    
    def run_continuous_monitoring(self, interval: int = 60):
        """Run continuous monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while True:
                # Get dashboard data
                data = self.get_dashboard_data()
                
                # Print dashboard
                self.print_dashboard()
                
                # Check for alerts
                self._check_alerts(data)
                
                # Wait for next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def _check_alerts(self, data: Dict[str, Any]):
        """Check for alert conditions"""
        # System alerts
        system = data['system']
        for alert in system['alerts']:
            self.alert_manager.send_alert(
                alert['level'],
                alert['message'],
                {'component': 'system'}
            )
        
        # Freqtrade alerts
        freqtrade = data['freqtrade']
        if not freqtrade['running']:
            self.alert_manager.send_alert(
                'CRITICAL',
                'Freqtrade is not running',
                {'component': 'freqtrade'}
            )
        
        # API server alerts
        api = data['api_server']
        if api['enabled'] and api['status'] not in ['healthy', 'disabled']:
            self.alert_manager.send_alert(
                'WARNING',
                f'API server status: {api["status"]}',
                {'component': 'api_server'}
            )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Monitor Pragma Trading Bot"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config-private.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--mode",
        choices=["production", "staging", "development"],
        default="development",
        help="Monitoring mode"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show monitoring dashboard"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check only"
    )
    
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous monitoring"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize dashboard
        dashboard = MonitoringDashboard(args.config)
        
        if args.health_check:
            # Run health check
            data = dashboard.get_dashboard_data()
            
            # Check for critical issues
            critical_issues = []
            
            if not data['freqtrade']['running']:
                critical_issues.append("Freqtrade not running")
            
            if data['system']['alerts']:
                critical_alerts = [a for a in data['system']['alerts'] if a['level'] == 'CRITICAL']
                if critical_alerts:
                    critical_issues.extend([a['message'] for a in critical_alerts])
            
            if critical_issues:
                print("❌ Health check failed:")
                for issue in critical_issues:
                    print(f"  - {issue}")
                return 1
            else:
                print("✅ Health check passed")
                return 0
        
        elif args.dashboard:
            # Show dashboard
            dashboard.print_dashboard()
        
        elif args.continuous:
            # Run continuous monitoring
            dashboard.run_continuous_monitoring(args.interval)
        
        else:
            # Default: show dashboard once
            dashboard.print_dashboard()
        
        return 0
        
    except Exception as e:
        logger.error(f"Monitoring error: {e}")
        print(f"❌ Monitoring error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
