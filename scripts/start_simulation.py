#!/usr/bin/env python3
"""
Start Simulation Script - Pragma Trading Bot

This script starts the trading bot in simulation mode (backtest or dry-run).
Safe for testing and development.

Usage:
    python scripts/start_simulation.py --mode backtest
    python scripts/start_simulation.py --mode dry-run
    python scripts/start_simulation.py --mode backtest --config configs/backtest_config.json
"""

import argparse
import sys
import os
import logging
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from execution.simulated import SimulatedExecutor
from execution.base import ExecutionMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from file"""
    import json
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def validate_config(config, mode):
    """Validate configuration for simulation mode"""
    
    # Required fields
    required_fields = ['stake_currency', 'timeframe']
    missing = [field for field in required_fields if field not in config]
    
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    # Ensure dry_run is True for simulation
    if not config.get('dry_run', True):
        logger.warning("Forcing dry_run=True for simulation mode")
        config['dry_run'] = True
    
    # Validate mode-specific requirements
    if mode == "backtest":
        if 'timerange' not in config:
            logger.warning("No timerange specified, using default")
            config['timerange'] = "20240101-20241001"
    
    elif mode == "dry-run":
        if 'exchange' not in config:
            raise ValueError("Dry-run mode requires exchange configuration")
        
        if not config['exchange'].get('key') or not config['exchange'].get('secret'):
            raise ValueError("Dry-run mode requires valid API credentials")
    
    return config

def start_freqtrade_simulation(config, mode):
    """Start Freqtrade in simulation mode"""
    
    # Build Freqtrade command
    cmd = [
        "freqtrade", "trade",
        "--strategy", "RegimeAdaptiveStrategy",
        "--config", "configs/config-private.json"
    ]
    
    if mode == "backtest":
        cmd.extend([
            "--timerange", config.get('timerange', '20240101-20241001'),
            "--export", "trades"
        ])
    else:  # dry-run
        cmd.append("--dry-run")
    
    # Add additional options
    if config.get('timeframe'):
        cmd.extend(["--timeframe", config['timeframe']])
    
    if config.get('pairs'):
        cmd.extend(["--pairs"] + config['pairs'])
    
    logger.info(f"Starting Freqtrade simulation: {' '.join(cmd)}")
    
    try:
        # Start Freqtrade process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitor process
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            # Check if process is still running
            if process.poll() is not None:
                break
        
        # Get return code
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("Simulation completed successfully")
            return True
        else:
            logger.error(f"Simulation failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return False

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Start Pragma Trading Bot in simulation mode"
    )
    
    parser.add_argument(
        "--mode",
        choices=["backtest", "dry-run"],
        required=True,
        help="Simulation mode: backtest or dry-run"
    )
    
    parser.add_argument(
        "--config",
        default="configs/config-private.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--strategy-path",
        default="src/strategies",
        help="Path to strategy files"
    )
    
    parser.add_argument(
        "--data-path",
        default="user_data/data",
        help="Path to market data"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Validate configuration
        logger.info(f"Validating configuration for {args.mode} mode")
        config = validate_config(config, args.mode)
        
        # Initialize executor
        logger.info(f"Initializing {args.mode} executor")
        executor = SimulatedExecutor(config, mode=args.mode)
        
        # Validate execution
        if not executor.validate_execution_allowed():
            logger.error("Execution not allowed")
            return 1
        
        # Log execution start
        executor.log_execution_start()
        
        # Start simulation
        logger.info(f"Starting {args.mode} simulation")
        success = start_freqtrade_simulation(config, args.mode)
        
        if success:
            logger.info("Simulation completed successfully")
            return 0
        else:
            logger.error("Simulation failed")
            return 1
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
