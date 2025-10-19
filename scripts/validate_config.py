#!/usr/bin/env python3
"""
Configuration Validation Script - Pragma Trading Bot

This script validates configuration files for different execution modes.
Ensures all required fields are present and values are valid.

Usage:
    python scripts/validate_config.py --config configs/config-private.json
    python scripts/validate_config.py --config configs/backtest_config.json --mode backtest
    python scripts/validate_config.py --config configs/live_config.json --mode live
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from execution.base import ExecutionMode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from file"""
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def validate_common_fields(config):
    """Validate common required fields"""
    
    errors = []
    warnings = []
    
    # Required fields
    required_fields = {
        'stake_currency': str,
        'timeframe': str,
        'max_open_trades': int
    }
    
    for field, field_type in required_fields.items():
        if field not in config:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(config[field], field_type):
            errors.append(f"Invalid type for {field}: expected {field_type.__name__}")
    
    # Validate stake_currency
    if 'stake_currency' in config:
        valid_currencies = ['USDT', 'USDC', 'BTC', 'ETH']
        if config['stake_currency'] not in valid_currencies:
            warnings.append(f"Unusual stake currency: {config['stake_currency']}")
    
    # Validate timeframe
    if 'timeframe' in config:
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
        if config['timeframe'] not in valid_timeframes:
            errors.append(f"Invalid timeframe: {config['timeframe']}")
    
    # Validate max_open_trades
    if 'max_open_trades' in config:
        if config['max_open_trades'] <= 0:
            errors.append("max_open_trades must be greater than 0")
        elif config['max_open_trades'] > 10:
            warnings.append("High max_open_trades value")
    
    return errors, warnings

def validate_exchange_config(config):
    """Validate exchange configuration"""
    
    errors = []
    warnings = []
    
    if 'exchange' not in config:
        errors.append("Missing exchange configuration")
        return errors, warnings
    
    exchange = config['exchange']
    
    # Required exchange fields
    required_fields = ['name', 'key', 'secret']
    for field in required_fields:
        if field not in exchange:
            errors.append(f"Missing exchange field: {field}")
    
    # Validate exchange name
    if 'name' in exchange:
        valid_exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex']
        if exchange['name'] not in valid_exchanges:
            warnings.append(f"Unusual exchange: {exchange['name']}")
    
    # Validate API credentials
    if 'key' in exchange and 'secret' in exchange:
        if not exchange['key'] or not exchange['secret']:
            errors.append("Empty API credentials")
        elif exchange['key'] == 'YOUR_API_KEY' or exchange['secret'] == 'YOUR_SECRET':
            errors.append("Using placeholder API credentials")
    
    return errors, warnings

def validate_backtest_config(config):
    """Validate backtest configuration"""
    
    errors = []
    warnings = []
    
    # Must be dry_run
    if not config.get('dry_run', True):
        errors.append("Backtest mode must use dry_run=True")
    
    # Validate timerange
    if 'timerange' not in config:
        warnings.append("No timerange specified, using default")
    else:
        timerange = config['timerange']
        if not isinstance(timerange, str):
            errors.append("timerange must be a string")
        elif '-' not in timerange:
            errors.append("timerange must be in format YYYYMMDD-YYYYMMDD")
    
    # Validate data path
    if 'data_path' in config:
        data_path = config['data_path']
        if not os.path.exists(data_path):
            warnings.append(f"Data path does not exist: {data_path}")
    
    return errors, warnings

def validate_dryrun_config(config):
    """Validate dry-run configuration"""
    
    errors = []
    warnings = []
    
    # Must be dry_run
    if not config.get('dry_run', True):
        errors.append("Dry-run mode must use dry_run=True")
    
    # Must have exchange config
    if 'exchange' not in config:
        errors.append("Dry-run mode requires exchange configuration")
    
    # Validate stake_amount
    if 'stake_amount' in config:
        stake_amount = config['stake_amount']
        if isinstance(stake_amount, (int, float)):
            if stake_amount <= 0:
                errors.append("stake_amount must be greater than 0")
        elif isinstance(stake_amount, str):
            if stake_amount != "unlimited":
                try:
                    amount = float(stake_amount)
                    if amount <= 0:
                        errors.append("stake_amount must be greater than 0")
                except ValueError:
                    errors.append("Invalid stake_amount format")
    
    return errors, warnings

def validate_live_config(config):
    """Validate live trading configuration"""
    
    errors = []
    warnings = []
    
    # Must not be dry_run
    if config.get('dry_run', True):
        errors.append("Live trading cannot use dry_run=True")
    
    # Must have exchange config
    if 'exchange' not in config:
        errors.append("Live trading requires exchange configuration")
    
    # Validate stake_amount
    if 'stake_amount' not in config:
        errors.append("Live trading requires stake_amount")
    else:
        stake_amount = config['stake_amount']
        if isinstance(stake_amount, (int, float)):
            if stake_amount <= 0:
                errors.append("stake_amount must be greater than 0")
        elif isinstance(stake_amount, str):
            if stake_amount != "unlimited":
                try:
                    amount = float(stake_amount)
                    if amount <= 0:
                        errors.append("stake_amount must be greater than 0")
                except ValueError:
                    errors.append("Invalid stake_amount format")
    
    # Validate risk parameters
    if 'max_open_trades' in config:
        if config['max_open_trades'] > 5:
            warnings.append("High max_open_trades for live trading")
    
    # Check for test credentials
    if 'exchange' in config:
        exchange = config['exchange']
        if exchange.get('key') == 'test_key' or exchange.get('secret') == 'test_secret':
            errors.append("Live trading cannot use test credentials")
    
    return errors, warnings

def validate_strategy_config(config):
    """Validate strategy configuration"""
    
    errors = []
    warnings = []
    
    # Validate strategy name
    if 'strategy' not in config:
        errors.append("Missing strategy name")
    else:
        strategy = config['strategy']
        if strategy != 'RegimeAdaptiveStrategy':
            warnings.append(f"Using custom strategy: {strategy}")
    
    # Validate strategy path
    if 'strategy_path' in config:
        strategy_path = config['strategy_path']
        if not os.path.exists(strategy_path):
            warnings.append(f"Strategy path does not exist: {strategy_path}")
    
    return errors, warnings

def validate_risk_config(config):
    """Validate risk management configuration"""
    
    errors = []
    warnings = []
    
    # Validate risk parameters
    risk_params = [
        'max_open_trades',
        'stake_amount',
        'tradable_balance_ratio'
    ]
    
    for param in risk_params:
        if param in config:
            value = config[param]
            if isinstance(value, (int, float)):
                if value <= 0:
                    errors.append(f"{param} must be greater than 0")
                elif value > 1 and param == 'tradable_balance_ratio':
                    errors.append(f"{param} must be between 0 and 1")
    
    return errors, warnings

def validate_execution_config(config):
    """Validate execution configuration"""
    
    errors = []
    warnings = []
    
    if 'execution' in config:
        execution = config['execution']
        
        # Validate mode
        if 'mode' in execution:
            valid_modes = ['backtest', 'dry-run', 'live']
            if execution['mode'] not in valid_modes:
                errors.append(f"Invalid execution mode: {execution['mode']}")
        
        # Validate safety checks
        if 'safety_checks' in execution:
            if not isinstance(execution['safety_checks'], bool):
                errors.append("safety_checks must be boolean")
        
        # Validate monitoring
        if 'monitoring' in execution:
            if not isinstance(execution['monitoring'], bool):
                errors.append("monitoring must be boolean")
    
    return errors, warnings

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="Validate Pragma Trading Bot configuration"
    )
    
    parser.add_argument(
        "--config",
        required=True,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--mode",
        choices=["backtest", "dry-run", "live"],
        help="Execution mode (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix common issues"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Auto-detect mode if not specified
        if not args.mode:
            if config.get('dry_run', True):
                if 'timerange' in config:
                    args.mode = "backtest"
                else:
                    args.mode = "dry-run"
            else:
                args.mode = "live"
        
        logger.info(f"Validating configuration for {args.mode} mode")
        
        # Run validations
        all_errors = []
        all_warnings = []
        
        # Common validations
        errors, warnings = validate_common_fields(config)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Exchange validations
        errors, warnings = validate_exchange_config(config)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Strategy validations
        errors, warnings = validate_strategy_config(config)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Risk validations
        errors, warnings = validate_risk_config(config)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Execution validations
        errors, warnings = validate_execution_config(config)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Mode-specific validations
        if args.mode == "backtest":
            errors, warnings = validate_backtest_config(config)
        elif args.mode == "dry-run":
            errors, warnings = validate_dryrun_config(config)
        elif args.mode == "live":
            errors, warnings = validate_live_config(config)
        
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        
        # Report results
        print(f"\nConfiguration Validation Results for {args.mode} mode:")
        print("=" * 50)
        
        if all_errors:
            print(f"\n❌ ERRORS ({len(all_errors)}):")
            for error in all_errors:
                print(f"  - {error}")
        
        if all_warnings:
            print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
            for warning in all_warnings:
                print(f"  - {warning}")
        
        if not all_errors and not all_warnings:
            print("\n✅ Configuration is valid!")
            return 0
        
        if not all_errors and not args.strict:
            print("\n✅ Configuration is valid (warnings only)")
            return 0
        
        if all_errors:
            print(f"\n❌ Configuration has {len(all_errors)} errors")
            return 1
        
        if args.strict and all_warnings:
            print(f"\n❌ Configuration has {len(all_warnings)} warnings (strict mode)")
            return 1
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
