#!/usr/bin/env python3
"""
Production Infrastructure Setup
===============================

This module sets up the production infrastructure for live trading including:
- Production configuration management
- Infrastructure monitoring
- System health checks
- Data pipeline setup

Author: Pragma Trading Bot Team
Date: 2025-10-20
"""

import os
import json
import yaml
import logging
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import requests
from dataclasses import dataclass, asdict

@dataclass
class SystemMetrics:
    """System metrics data class"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime: float

@dataclass
class InfrastructureStatus:
    """Infrastructure status data class"""
    timestamp: str
    status: str  # healthy, warning, critical
    components: Dict[str, str]
    metrics: SystemMetrics
    alerts: List[str]

class ProductionConfigManager:
    """Production configuration manager"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def create_production_config(self) -> Dict[str, Any]:
        """Create production configuration"""
        config = {
            "environment": "production",
            "trading": {
                "enabled": True,
                "dry_run": False,
                "max_positions": 5,
                "max_portfolio_heat": 0.8,
                "max_drawdown": 0.15,
                "daily_loss_limit": 0.05
            },
            "risk_management": {
                "kelly_safety_factor": 0.25,
                "max_position_size": 0.2,
                "volatility_threshold": 0.05,
                "correlation_threshold": 0.7,
                "circuit_breakers": {
                    "daily_loss": 0.05,
                    "max_drawdown": 0.15,
                    "max_positions": 5,
                    "volatility": 0.1
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 60,  # seconds
                "health_check_interval": 300,  # seconds
                "alert_thresholds": {
                    "cpu_percent": 80,
                    "memory_percent": 85,
                    "disk_percent": 90,
                    "network_latency": 1000  # ms
                }
            },
            "alerts": {
                "enabled": True,
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "your_email@gmail.com",
                    "password": "your_app_password"
                },
                "webhook": {
                    "enabled": False,
                    "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
                }
            },
            "data": {
                "sources": ["binance", "coinbase"],
                "timeframes": ["5m", "15m", "1h"],
                "pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "retention_days": 30
            },
            "logging": {
                "level": "INFO",
                "file": "logs/production.log",
                "max_size": "10MB",
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        return config
    
    def save_config(self, config: Dict[str, Any], filename: str = "production_config.json"):
        """Save configuration to file"""
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def load_config(self, filename: str = "production_config.json") -> Optional[Dict[str, Any]]:
        """Load configuration from file"""
        config_path = self.config_dir / filename
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file {config_path} not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return None
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # Check required sections
        required_sections = ["trading", "risk_management", "monitoring", "alerts", "data", "logging"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate trading settings
        if "trading" in config:
            trading = config["trading"]
            if trading.get("max_positions", 0) <= 0:
                errors.append("max_positions must be positive")
            if not 0 < trading.get("max_portfolio_heat", 0) <= 1:
                errors.append("max_portfolio_heat must be between 0 and 1")
            if not 0 < trading.get("max_drawdown", 0) <= 1:
                errors.append("max_drawdown must be between 0 and 1")
        
        # Validate risk management settings
        if "risk_management" in config:
            risk = config["risk_management"]
            if not 0 < risk.get("kelly_safety_factor", 0) <= 1:
                errors.append("kelly_safety_factor must be between 0 and 1")
            if not 0 < risk.get("max_position_size", 0) <= 1:
                errors.append("max_position_size must be between 0 and 1")
        
        is_valid = len(errors) == 0
        
        if errors:
            self.logger.error(f"Configuration validation errors: {errors}")
        else:
            self.logger.info("Configuration validation passed")
        
        return is_valid, errors

class SystemMonitor:
    """System monitoring and health checks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Uptime
            uptime = time.time() - self.start_time
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count,
                uptime=uptime
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0,
                uptime=0.0
            )
    
    def check_system_health(self, metrics: SystemMetrics, 
                          thresholds: Dict[str, float]) -> Tuple[str, List[str]]:
        """Check system health against thresholds"""
        alerts = []
        status = "healthy"
        
        # CPU check
        if metrics.cpu_percent > thresholds.get("cpu_percent", 80):
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            status = "warning"
        
        # Memory check
        if metrics.memory_percent > thresholds.get("memory_percent", 85):
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
            status = "warning"
        
        # Disk check
        if metrics.disk_percent > thresholds.get("disk_percent", 90):
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
            status = "critical"
        
        # Critical thresholds
        if metrics.cpu_percent > 95:
            status = "critical"
        if metrics.memory_percent > 95:
            status = "critical"
        
        return status, alerts
    
    def check_infrastructure_components(self) -> Dict[str, str]:
        """Check infrastructure components"""
        components = {}
        
        try:
            # Check if required processes are running
            processes = ["python", "freqtrade"]
            for process in processes:
                try:
                    result = subprocess.run(
                        ["pgrep", process], 
                        capture_output=True, 
                        text=True
                    )
                    if result.returncode == 0:
                        components[f"{process}_process"] = "running"
                    else:
                        components[f"{process}_process"] = "stopped"
                except:
                    components[f"{process}_process"] = "unknown"
            
            # Check network connectivity
            try:
                response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
                if response.status_code == 200:
                    components["binance_api"] = "connected"
                else:
                    components["binance_api"] = "error"
            except:
                components["binance_api"] = "disconnected"
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free > 1024**3:  # 1GB free
                components["disk_space"] = "sufficient"
            else:
                components["disk_space"] = "low"
            
            # Check log files
            log_dir = Path("logs")
            if log_dir.exists():
                components["log_directory"] = "exists"
            else:
                components["log_directory"] = "missing"
            
        except Exception as e:
            self.logger.error(f"Error checking infrastructure components: {e}")
            components["error"] = str(e)
        
        return components
    
    def get_infrastructure_status(self, config: Dict[str, Any]) -> InfrastructureStatus:
        """Get complete infrastructure status"""
        metrics = self.get_system_metrics()
        components = self.check_infrastructure_components()
        
        # Get health status
        monitoring_config = config.get("monitoring", {})
        thresholds = monitoring_config.get("alert_thresholds", {})
        status, alerts = self.check_system_health(metrics, thresholds)
        
        # Add component alerts
        for component, state in components.items():
            if state in ["stopped", "disconnected", "low", "missing"]:
                alerts.append(f"Component {component}: {state}")
                if status == "healthy":
                    status = "warning"
        
        return InfrastructureStatus(
            timestamp=datetime.now().isoformat(),
            status=status,
            components=components,
            metrics=metrics,
            alerts=alerts
        )

class DataPipelineManager:
    """Data pipeline management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def setup_data_directories(self) -> bool:
        """Setup data directories"""
        try:
            data_config = self.config.get("data", {})
            
            # Create data directories
            directories = [
                "user_data/data",
                "user_data/data/binance",
                "user_data/data/coinbase",
                "logs",
                "backups",
                "reports"
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up data directories: {e}")
            return False
    
    def validate_data_sources(self) -> Dict[str, bool]:
        """Validate data sources"""
        data_config = self.config.get("data", {})
        sources = data_config.get("sources", [])
        
        validation_results = {}
        
        for source in sources:
            try:
                if source == "binance":
                    response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
                    validation_results[source] = response.status_code == 200
                elif source == "coinbase":
                    response = requests.get("https://api.exchange.coinbase.com/products", timeout=5)
                    validation_results[source] = response.status_code == 200
                else:
                    validation_results[source] = False
                    
            except Exception as e:
                self.logger.error(f"Error validating {source}: {e}")
                validation_results[source] = False
        
        return validation_results
    
    def setup_logging(self) -> bool:
        """Setup logging configuration"""
        try:
            logging_config = self.config.get("logging", {})
            
            # Create logs directory
            log_file = logging_config.get("file", "logs/production.log")
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Configure logging
            log_level = getattr(logging, logging_config.get("level", "INFO").upper())
            
            logging.basicConfig(
                level=log_level,
                format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            self.logger.info("Logging configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up logging: {e}")
            return False

class ProductionInfrastructureSetup:
    """Main production infrastructure setup class"""
    
    def __init__(self):
        self.config_manager = ProductionConfigManager()
        self.system_monitor = SystemMonitor()
        self.logger = logging.getLogger(__name__)
        self.config = None
    
    def setup_production_infrastructure(self) -> bool:
        """Setup complete production infrastructure"""
        try:
            self.logger.info("Starting production infrastructure setup...")
            
            # Step 1: Create and validate configuration
            self.logger.info("Step 1: Creating production configuration...")
            self.config = self.config_manager.create_production_config()
            
            is_valid, errors = self.config_manager.validate_config(self.config)
            if not is_valid:
                self.logger.error(f"Configuration validation failed: {errors}")
                return False
            
            # Step 2: Save configuration
            self.logger.info("Step 2: Saving configuration...")
            if not self.config_manager.save_config(self.config):
                return False
            
            # Step 3: Setup data pipeline
            self.logger.info("Step 3: Setting up data pipeline...")
            data_manager = DataPipelineManager(self.config)
            if not data_manager.setup_data_directories():
                return False
            
            # Step 4: Setup logging
            self.logger.info("Step 4: Setting up logging...")
            if not data_manager.setup_logging():
                return False
            
            # Step 5: Validate data sources
            self.logger.info("Step 5: Validating data sources...")
            validation_results = data_manager.validate_data_sources()
            for source, is_valid in validation_results.items():
                if is_valid:
                    self.logger.info(f"Data source {source}: Valid")
                else:
                    self.logger.warning(f"Data source {source}: Invalid")
            
            # Step 6: Initial system health check
            self.logger.info("Step 6: Performing initial system health check...")
            infrastructure_status = self.system_monitor.get_infrastructure_status(self.config)
            
            self.logger.info(f"Infrastructure status: {infrastructure_status.status}")
            if infrastructure_status.alerts:
                self.logger.warning(f"Alerts: {infrastructure_status.alerts}")
            
            self.logger.info("Production infrastructure setup completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up production infrastructure: {e}")
            return False
    
    def get_setup_report(self) -> Dict[str, Any]:
        """Get setup report"""
        if not self.config:
            return {"error": "Infrastructure not setup"}
        
        infrastructure_status = self.system_monitor.get_infrastructure_status(self.config)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "setup_status": "completed",
            "infrastructure_status": asdict(infrastructure_status),
            "configuration": self.config,
            "data_sources": DataPipelineManager(self.config).validate_data_sources(),
            "directories_created": [
                "user_data/data",
                "user_data/data/binance", 
                "user_data/data/coinbase",
                "logs",
                "backups",
                "reports"
            ]
        }
        
        return report

def main():
    """Test production infrastructure setup"""
    logging.basicConfig(level=logging.INFO)
    
    print("Production Infrastructure Setup Test")
    print("=" * 50)
    
    # Create setup instance
    setup = ProductionInfrastructureSetup()
    
    # Setup infrastructure
    print("\n1. Setting up production infrastructure...")
    success = setup.setup_production_infrastructure()
    
    if success:
        print("   Production infrastructure setup completed successfully!")
        
        # Get setup report
        print("\n2. Generating setup report...")
        report = setup.get_setup_report()
        
        print(f"   Setup Status: {report['setup_status']}")
        print(f"   Infrastructure Status: {report['infrastructure_status']['status']}")
        print(f"   Components Checked: {len(report['infrastructure_status']['components'])}")
        print(f"   Alerts: {len(report['infrastructure_status']['alerts'])}")
        
        # System metrics
        metrics = report['infrastructure_status']['metrics']
        print(f"\n3. System Metrics:")
        print(f"   CPU Usage: {metrics['cpu_percent']:.1f}%")
        print(f"   Memory Usage: {metrics['memory_percent']:.1f}%")
        print(f"   Disk Usage: {metrics['disk_percent']:.1f}%")
        print(f"   Process Count: {metrics['process_count']}")
        print(f"   Uptime: {metrics['uptime']:.1f} seconds")
        
        # Data sources
        print(f"\n4. Data Sources:")
        for source, is_valid in report['data_sources'].items():
            status = "Valid" if is_valid else "Invalid"
            print(f"   {source}: {status}")
        
        print(f"\n5. Directories Created:")
        for directory in report['directories_created']:
            print(f"   - {directory}")
        
        print(f"\nProduction infrastructure is ready!")
        print(f"Configuration saved to: configs/production_config.json")
        print(f"Logs will be written to: logs/production.log")
        
    else:
        print("   Production infrastructure setup failed!")
        print("   Check logs for details.")
    
    print("\nProduction Infrastructure Setup test completed!")

if __name__ == "__main__":
    main()
