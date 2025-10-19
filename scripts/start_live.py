#!/usr/bin/env python3
"""
Start Live Trading Script - Pragma Trading Bot

⚠️  WARNING: This script starts LIVE TRADING with REAL MONEY.
Use with extreme caution and only after thorough testing.

Usage:
    python scripts/start_live.py --confirm-live
    python scripts/start_live.py --confirm-live --safety-phrase "I_UNDERSTAND_THE_RISKS"
"""

import argparse
import sys
import os
import logging
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from execution.live import LiveExecutor
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

def validate_live_config(config):
    """Validate configuration for live trading"""
    
    # Required fields for live trading
    required_fields = [
        'stake_currency',
        'timeframe',
        'exchange.key',
        'exchange.secret',
        'max_open_trades'
    ]
    
    missing = []
    for field in required_fields:
        if '.' in field:
            # Nested field
            parts = field.split('.')
            value = config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    missing.append(field)
                    break
        else:
            if field not in config:
                missing.append(field)
    
    if missing:
        raise ValueError(f"Missing required fields for live trading: {missing}")
    
    # Must not be dry_run
    if config.get('dry_run', True):
        raise ValueError("Live trading cannot use dry_run config")
    
    # Validate exchange credentials
    exchange = config.get('exchange', {})
    if not exchange.get('key') or not exchange.get('secret'):
        raise ValueError("Live trading requires valid exchange credentials")
    
    # Validate risk parameters
    if config.get('max_open_trades', 0) <= 0:
        raise ValueError("max_open_trades must be greater than 0")
    
    if config.get('stake_amount', 0) <= 0:
        raise ValueError("stake_amount must be greater than 0")
    
    return config

def confirm_live_trading():
    """Get confirmation for live trading"""
    
    print("=" * 70)
    print("⚠️  LIVE TRADING CONFIRMATION ⚠️")
    print("=" * 70)
    print()
    print("This will start LIVE TRADING with REAL MONEY.")
    print("Incorrect usage can result in significant financial losses.")
    print()
    print("Before proceeding, confirm that you have:")
    print("✅ Tested thoroughly in backtest mode")
    print("✅ Validated in dry-run mode for at least 1 week")
    print("✅ Reviewed all safety procedures")
    print("✅ Set appropriate risk limits")
    print("✅ Have emergency procedures ready")
    print("✅ Monitored the system closely")
    print()
    
    # Get confirmation
    confirm = input("Do you want to proceed with live trading? (yes/no): ").lower().strip()
    
    if confirm != "yes":
        print("Live trading cancelled.")
        return False
    
    # Get safety phrase
    safety_phrase = input("Enter safety phrase 'I_UNDERSTAND_THE_RISKS': ").strip()
    
    if safety_phrase != "I_UNDERSTAND_THE_RISKS":
        print("Invalid safety phrase. Live trading cancelled.")
        return False
    
    print()
    print("⚠️  FINAL WARNING: LIVE TRADING WILL START ⚠️")
    print("You can stop trading at any time by pressing Ctrl+C")
    print()
    
    final_confirm = input("Type 'START LIVE TRADING' to proceed: ").strip()
    
    if final_confirm != "START LIVE TRADING":
        print("Live trading cancelled.")
        return False
    
    return True

def start_freqtrade_live(config):
    """Start Freqtrade in live mode"""
    
    # Build Freqtrade command
    cmd = [
        "freqtrade", "trade",
        "--strategy", "RegimeAdaptiveStrategy",
        "--config", "configs/config-private.json"
    ]
    
    # Add additional options
    if config.get('timeframe'):
        cmd.extend(["--timeframe", config['timeframe']])
    
    if config.get('pairs'):
        cmd.extend(["--pairs"] + config['pairs'])

    logger.critical("=" * 70)
    logger.critical("🚨 STARTING LIVE TRADING 🚨")
    logger.critical("=" * 70)
    logger.critical(f"Exchange: {config.get('exchange', {}).get('name')}")
    logger.critical(f"Stake Currency: {config.get('stake_currency')}")
    logger.critical(f"Max Open Trades: {config.get('max_open_trades')}")
    logger.critical(f"Stake Amount: {config.get('stake_amount')}")
    logger.critical("=" * 70)
    logger.critical("⚠️  REAL TRADES WILL BE EXECUTED ⚠️")
    logger.critical("=" * 70)
    
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
            logger.info("Live trading completed successfully")
            return True
        else:
            logger.error(f"Live trading failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        logger.critical("Live trading interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        logger.error(f"Error starting live trading: {e}")
        return False

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Start Pragma Trading Bot in live mode"
    )
    
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Confirm live trading (required)"
    )
    
    parser.add_argument(
        "--safety-phrase",
        default="I_UNDERSTAND_THE_RISKS",
        help="Safety phrase for confirmation"
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
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip interactive confirmation (DANGEROUS)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Check if live trading is confirmed
        if not args.confirm_live:
            logger.error("Live trading requires --confirm-live flag")
            return 1
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Validate configuration
        logger.info("Validating configuration for live trading")
        config = validate_live_config(config)
        
        # Get confirmation (unless skipped)
        if not args.skip_confirmation:
            if not confirm_live_trading():
                return 1
        
        # Enable live trading globally
        logger.info("Enabling live trading globally")
        LiveExecutor.enable_live_trading(True)
        
        # Initialize executor
        logger.info("Initializing live executor")
        executor = LiveExecutor(
            config=config,
            confirm_live=True,
            safety_phrase=args.safety_phrase
        )
        
        # Validate execution
        if not executor.validate_execution_allowed():
            logger.error("Execution not allowed")
            return 1
        
        # Log execution start
        executor.log_execution_start()
        
        # Start live trading
        logger.critical("Starting live trading")
        success = start_freqtrade_live(config)
        
        if success:
            logger.info("Live trading completed successfully")
            return 0
        else:
            logger.error("Live trading failed")
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
    finally:
        # Disable live trading
        LiveExecutor.enable_live_trading(False)
        logger.info("Live trading disabled")

if __name__ == "__main__":
    sys.exit(main())
