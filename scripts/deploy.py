#!/usr/bin/env python3
"""
Deployment Script - Pragma Trading Bot

This script handles deployment of the Pragma Trading Bot to production.
Supports different deployment modes and environments.

Usage:
    python scripts/deploy.py --mode production
    python scripts/deploy.py --mode staging --config configs/staging_config.json
    python scripts/deploy.py --mode development --docker
"""

import argparse
import sys
import os
import json
import subprocess
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployment of Pragma Trading Bot"""
    
    def __init__(self, mode: str, config_path: str, docker: bool = False):
        """
        Initialize deployment manager.
        
        Args:
            mode: Deployment mode (production, staging, development)
            config_path: Path to configuration file
            docker: Whether to use Docker deployment
        """
        self.mode = mode
        self.config_path = config_path
        self.docker = docker
        self.project_root = Path(__file__).parent.parent
        self.deployment_dir = self.project_root / "deployments" / mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup deployment directory
        self._setup_deployment_dir()
    
    def _load_config(self):
        """Load deployment configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _setup_deployment_dir(self):
        """Setup deployment directory"""
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Deployment directory: {self.deployment_dir}")
    
    def validate_environment(self):
        """Validate deployment environment"""
        logger.info("Validating deployment environment...")
        
        # Check Python version
        if sys.version_info < (3, 11):
            raise RuntimeError("Python 3.11+ required")
        
        # Check required tools
        required_tools = ['git', 'pip']
        if self.docker:
            required_tools.append('docker')
        
        for tool in required_tools:
            if not shutil.which(tool):
                raise RuntimeError(f"Required tool not found: {tool}")
        
        # Check configuration
        self._validate_config()
        
        logger.info("Environment validation passed")
        return True
    
    def _validate_config(self):
        """Validate deployment configuration"""
        required_fields = ['strategy', 'timeframe', 'stake_currency']
        missing = [field for field in required_fields if field not in self.config]
        
        if missing:
            raise ValueError(f"Missing required configuration fields: {missing}")
        
        # Mode-specific validation
        if self.mode == "production":
            if self.config.get('dry_run', True):
                raise ValueError("Production mode cannot use dry_run=True")
            
            if not self.config.get('exchange', {}).get('key'):
                raise ValueError("Production mode requires exchange API key")
    
    def prepare_deployment(self):
        """Prepare deployment package"""
        logger.info("Preparing deployment package...")
        
        # Create deployment package
        package_dir = self.deployment_dir / f"pragma-bot-{self.timestamp}"
        package_dir.mkdir(exist_ok=True)
        
        # Copy source code
        self._copy_source_code(package_dir)
        
        # Copy configuration
        self._copy_configuration(package_dir)
        
        # Copy scripts
        self._copy_scripts(package_dir)
        
        # Copy documentation
        self._copy_documentation(package_dir)
        
        # Create requirements file
        self._create_requirements(package_dir)
        
        # Create deployment script
        self._create_deployment_script(package_dir)
        
        logger.info(f"Deployment package created: {package_dir}")
        return package_dir
    
    def _copy_source_code(self, package_dir):
        """Copy source code to package"""
        src_dir = package_dir / "src"
        shutil.copytree(self.project_root / "src", src_dir)
        logger.info("Source code copied")
    
    def _copy_configuration(self, package_dir):
        """Copy configuration files"""
        config_dir = package_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        # Copy main config
        shutil.copy2(self.config_path, config_dir / "config.json")
        
        # Copy environment template
        env_template = self.project_root / "configs" / "env.example"
        if env_template.exists():
            shutil.copy2(env_template, config_dir / ".env.example")
        
        logger.info("Configuration files copied")
    
    def _copy_scripts(self, package_dir):
        """Copy deployment scripts"""
        scripts_dir = package_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Copy all scripts
        for script in (self.project_root / "scripts").glob("*.py"):
            shutil.copy2(script, scripts_dir)
        
        # Make scripts executable
        for script in scripts_dir.glob("*.py"):
            script.chmod(0o755)
        
        logger.info("Scripts copied")
    
    def _copy_documentation(self, package_dir):
        """Copy documentation"""
        docs_dir = package_dir / "docs"
        shutil.copytree(self.project_root / "docs", docs_dir)
        logger.info("Documentation copied")
    
    def _create_requirements(self, package_dir):
        """Create requirements.txt for deployment"""
        requirements = [
            "freqtrade>=2025.9.1",
            "ccxt>=4.0.0",
            "datasieve>=0.0.1",
            "hmmlearn>=0.3.0",
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
            "catboost>=1.2.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "TA-Lib>=0.4.25",
            "python-dotenv>=1.0.0",
            "requests>=2.31.0",
            "colorama>=0.4.6"
        ]
        
        requirements_file = package_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        
        logger.info("Requirements file created")
    
    def _create_deployment_script(self, package_dir):
        """Create deployment script for target environment"""
        if self.docker:
            self._create_docker_deployment(package_dir)
        else:
            self._create_native_deployment(package_dir)
    
    def _create_docker_deployment(self, package_dir):
        """Create Docker deployment files"""
        
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs user_data/data user_data/models

# Set permissions
RUN chmod +x scripts/*.py

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python scripts/health_check.py

# Default command
CMD ["python", "scripts/start_simulation.py", "--mode", "dry-run"]
"""
        
        dockerfile_path = package_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        compose_content = f"""
version: '3.8'

services:
  pragma-bot:
    build: .
    container_name: pragma-bot-{self.mode}
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./configs:/app/configs
      - ./logs:/app/logs
      - ./user_data:/app/user_data
    environment:
      - MODE={self.mode}
      - CONFIG_PATH=configs/config.json
    networks:
      - pragma-network

networks:
  pragma-network:
    driver: bridge
"""
        
        compose_path = package_dir / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# Pragma Trading Bot - Docker Deployment Script

set -e

echo "Starting Pragma Trading Bot deployment..."

# Build Docker image
echo "Building Docker image..."
docker build -t pragma-bot-{self.mode}:latest .

# Stop existing container if running
echo "Stopping existing container..."
docker-compose down || true

# Start new container
echo "Starting new container..."
docker-compose up -d

# Wait for container to be ready
echo "Waiting for container to be ready..."
sleep 10

# Check container status
echo "Checking container status..."
docker-compose ps

echo "Deployment completed successfully!"
echo "Container logs: docker-compose logs -f"
echo "Stop container: docker-compose down"
"""
        
        deploy_script_path = package_dir / "deploy.sh"
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
        deploy_script_path.chmod(0o755)
        
        logger.info("Docker deployment files created")
    
    def _create_native_deployment(self, package_dir):
        """Create native deployment files"""
        
        # Create systemd service file
        service_content = f"""[Unit]
Description=Pragma Trading Bot ({self.mode})
After=network.target

[Service]
Type=simple
User=freqtrade
Group=freqtrade
WorkingDirectory={package_dir}
Environment=PATH={package_dir}/venv/bin
ExecStart={package_dir}/venv/bin/python scripts/start_simulation.py --mode {self.mode} --config configs/config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_path = package_dir / "pragma-bot.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# Pragma Trading Bot - Native Deployment Script

set -e

echo "Starting Pragma Trading Bot deployment..."

# Create virtual environment
echo "Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs user_data/data user_data/models

# Set permissions
echo "Setting permissions..."
chmod +x scripts/*.py

# Install systemd service
echo "Installing systemd service..."
sudo cp pragma-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pragma-bot

# Start service
echo "Starting service..."
sudo systemctl start pragma-bot

# Check status
echo "Checking service status..."
sudo systemctl status pragma-bot

echo "Deployment completed successfully!"
echo "Service logs: sudo journalctl -u pragma-bot -f"
echo "Stop service: sudo systemctl stop pragma-bot"
"""
        
        deploy_script_path = package_dir / "deploy.sh"
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
        deploy_script_path.chmod(0o755)
        
        logger.info("Native deployment files created")
    
    def deploy(self):
        """Deploy the application"""
        logger.info(f"Starting deployment for {self.mode} mode...")
        
        try:
            # Validate environment
            self.validate_environment()
            
            # Prepare deployment
            package_dir = self.prepare_deployment()
            
            # Deploy based on mode
            if self.docker:
                self._deploy_docker(package_dir)
            else:
                self._deploy_native(package_dir)
            
            logger.info("Deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _deploy_docker(self, package_dir):
        """Deploy using Docker"""
        logger.info("Deploying with Docker...")
        
        # Change to package directory
        os.chdir(package_dir)
        
        # Run deployment script
        result = subprocess.run(["./deploy.sh"], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Docker deployment failed: {result.stderr}")
        
        logger.info("Docker deployment completed")
    
    def _deploy_native(self, package_dir):
        """Deploy natively"""
        logger.info("Deploying natively...")
        
        # Change to package directory
        os.chdir(package_dir)
        
        # Run deployment script
        result = subprocess.run(["./deploy.sh"], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Native deployment failed: {result.stderr}")
        
        logger.info("Native deployment completed")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Deploy Pragma Trading Bot"
    )
    
    parser.add_argument(
        "--mode",
        choices=["production", "staging", "development"],
        required=True,
        help="Deployment mode"
    )
    
    parser.add_argument(
        "--config",
        default="configs/production_config.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker deployment"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate environment, don't deploy"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize deployment manager
        manager = DeploymentManager(
            mode=args.mode,
            config_path=args.config,
            docker=args.docker
        )
        
        if args.validate_only:
            # Only validate
            manager.validate_environment()
            print("✅ Environment validation passed")
            return 0
        
        # Deploy
        success = manager.deploy()
        
        if success:
            print("✅ Deployment completed successfully!")
            return 0
        else:
            print("❌ Deployment failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Deployment error: {e}")
        print(f"❌ Deployment error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
