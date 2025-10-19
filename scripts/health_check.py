#!/usr/bin/env python3
"""
Health Check Script - Pragma Trading Bot

This script performs comprehensive health checks on the Pragma Trading Bot.
Used by Docker health checks and monitoring systems.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --config configs/config-private.json
    python scripts/health_check.py --verbose
"""

import argparse
import sys
import os
import json
import time
import psutil
import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Performs health checks on the Pragma Trading Bot"""
    
    def __init__(self, config_path: str = "configs/config-private.json"):
        """
        Initialize health checker.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.checks_passed = 0
        self.checks_failed = 0
        self.issues = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def check_system_resources(self) -> Tuple[bool, List[str]]:
        """Check system resource usage"""
        issues = []
        
        try:
            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 90:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            # Check load average
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0]
                cpu_count = psutil.cpu_count()
                if load_avg > cpu_count * 2:
                    issues.append(f"High load average: {load_avg:.2f}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return False, [f"System resource check failed: {e}"]
    
    def check_freqtrade_process(self) -> Tuple[bool, List[str]]:
        """Check if Freqtrade process is running"""
        issues = []
        
        try:
            freqtrade_running = False
            freqtrade_pid = None
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'freqtrade' in cmdline and 'trade' in cmdline:
                        freqtrade_running = True
                        freqtrade_pid = proc.info['pid']
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not freqtrade_running:
                issues.append("Freqtrade process not running")
            else:
                logger.info(f"Freqtrade running with PID: {freqtrade_pid}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking Freqtrade process: {e}")
            return False, [f"Freqtrade process check failed: {e}"]
    
    def check_api_server(self) -> Tuple[bool, List[str]]:
        """Check API server health"""
        issues = []
        
        try:
            api_config = self.config.get('api_server', {})
            if not api_config.get('enabled', False):
                logger.info("API server disabled")
                return True, []
            
            port = api_config.get('listen_port', 8080)
            url = f"http://localhost:{port}/api/v1/ping"
            
            try:
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    logger.info("API server responding")
                else:
                    issues.append(f"API server returned status {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                issues.append("API server not reachable")
            except requests.exceptions.Timeout:
                issues.append("API server timeout")
            except Exception as e:
                issues.append(f"API server error: {e}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking API server: {e}")
            return False, [f"API server check failed: {e}"]
    
    def check_log_files(self) -> Tuple[bool, List[str]]:
        """Check log files for errors"""
        issues = []
        
        try:
            log_dirs = ['logs', 'user_data/logs']
            
            for log_dir in log_dirs:
                if not os.path.exists(log_dir):
                    continue
                
                # Check for recent error logs
                for log_file in Path(log_dir).glob("*.log"):
                    try:
                        # Check file size (not too large)
                        if log_file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                            issues.append(f"Large log file: {log_file}")
                        
                        # Check for recent errors (last 100 lines)
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            recent_lines = lines[-100:] if len(lines) > 100 else lines
                            
                            error_count = sum(1 for line in recent_lines if 'ERROR' in line or 'CRITICAL' in line)
                            if error_count > 10:
                                issues.append(f"High error count in {log_file}: {error_count}")
                    
                    except Exception as e:
                        logger.warning(f"Error checking log file {log_file}: {e}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking log files: {e}")
            return False, [f"Log file check failed: {e}"]
    
    def check_data_files(self) -> Tuple[bool, List[str]]:
        """Check data files and directories"""
        issues = []
        
        try:
            # Check data directory
            data_dir = "user_data/data"
            if not os.path.exists(data_dir):
                issues.append("Data directory not found")
            else:
                # Check for recent data files
                data_files = list(Path(data_dir).glob("**/*.csv"))
                if not data_files:
                    issues.append("No data files found")
                else:
                    # Check if data files are recent (within last 24 hours)
                    recent_files = []
                    cutoff_time = time.time() - (24 * 3600)
                    
                    for file in data_files:
                        if file.stat().st_mtime > cutoff_time:
                            recent_files.append(file)
                    
                    if not recent_files:
                        issues.append("No recent data files (last 24 hours)")
            
            # Check models directory
            models_dir = "user_data/models"
            if not os.path.exists(models_dir):
                logger.warning("Models directory not found")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking data files: {e}")
            return False, [f"Data file check failed: {e}"]
    
    def check_configuration(self) -> Tuple[bool, List[str]]:
        """Check configuration validity"""
        issues = []
        
        try:
            if not self.config:
                issues.append("No configuration loaded")
                return False, issues
            
            # Check required fields
            required_fields = ['strategy', 'timeframe', 'stake_currency']
            for field in required_fields:
                if field not in self.config:
                    issues.append(f"Missing required field: {field}")
            
            # Check exchange configuration
            exchange = self.config.get('exchange', {})
            if not exchange.get('name'):
                issues.append("Exchange name not specified")
            
            # Check for dry_run in live mode
            if not self.config.get('dry_run', True):
                if not exchange.get('key') or not exchange.get('secret'):
                    issues.append("Live trading requires API credentials")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking configuration: {e}")
            return False, [f"Configuration check failed: {e}"]
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Check Python dependencies"""
        issues = []
        
        try:
            required_packages = [
                'freqtrade',
                'pandas',
                'numpy',
                'scikit-learn',
                'hmmlearn'
            ]
            
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    issues.append(f"Missing package: {package}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False, [f"Dependency check failed: {e}"]
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        logger.info("Starting health checks...")
        
        checks = [
            ("System Resources", self.check_system_resources),
            ("Freqtrade Process", self.check_freqtrade_process),
            ("API Server", self.check_api_server),
            ("Log Files", self.check_log_files),
            ("Data Files", self.check_data_files),
            ("Configuration", self.check_configuration),
            ("Dependencies", self.check_dependencies)
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'issues': [],
            'summary': {
                'total_checks': len(checks),
                'passed': 0,
                'failed': 0
            }
        }
        
        for check_name, check_func in checks:
            try:
                passed, issues = check_func()
                
                results['checks'][check_name] = {
                    'status': 'passed' if passed else 'failed',
                    'issues': issues
                }
                
                if passed:
                    self.checks_passed += 1
                    results['summary']['passed'] += 1
                    logger.info(f"✅ {check_name}: PASSED")
                else:
                    self.checks_failed += 1
                    results['summary']['failed'] += 1
                    results['issues'].extend(issues)
                    logger.error(f"❌ {check_name}: FAILED - {', '.join(issues)}")
                
            except Exception as e:
                self.checks_failed += 1
                results['summary']['failed'] += 1
                error_msg = f"{check_name} check error: {e}"
                results['issues'].append(error_msg)
                logger.error(f"❌ {check_name}: ERROR - {e}")
        
        # Determine overall status
        if self.checks_failed == 0:
            results['overall_status'] = 'healthy'
        elif self.checks_failed <= 2:
            results['overall_status'] = 'degraded'
        else:
            results['overall_status'] = 'unhealthy'
        
        logger.info(f"Health checks completed: {self.checks_passed} passed, {self.checks_failed} failed")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print health check summary"""
        print("\n" + "="*60)
        print("PRAGMA TRADING BOT - HEALTH CHECK SUMMARY")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print()
        
        print("CHECK RESULTS:")
        for check_name, check_result in results['checks'].items():
            status = check_result['status'].upper()
            emoji = "✅" if status == "PASSED" else "❌"
            print(f"  {emoji} {check_name}: {status}")
            
            if check_result['issues']:
                for issue in check_result['issues']:
                    print(f"    - {issue}")
        
        print()
        print(f"SUMMARY: {results['summary']['passed']}/{results['summary']['total_checks']} checks passed")
        
        if results['issues']:
            print("\nISSUES FOUND:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Health check for Pragma Trading Bot"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config-private.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if unhealthy"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize health checker
        checker = HealthChecker(args.config)
        
        # Run health checks
        results = checker.run_all_checks()
        
        # Output results
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            checker.print_summary(results)
        
        # Exit with appropriate code
        if args.exit_code:
            if results['overall_status'] == 'healthy':
                return 0
            else:
                return 1
        else:
            return 0
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        if args.json:
            print(json.dumps({
                'error': str(e),
                'overall_status': 'error'
            }, indent=2))
        else:
            print(f"❌ Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
