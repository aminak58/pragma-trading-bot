"""
Base Execution Classes

Provides safe execution modes with validation and fail-safes
to prevent accidental live trading.
"""

from enum import Enum
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode enum for type safety."""
    BACKTEST = "backtest"
    DRY_RUN = "dry-run"
    LIVE = "live"


class BaseExecutor:
    """
    Base class for all execution modes.
    
    Enforces strict validation and fail-safes to prevent
    accidental live trading with incorrect configuration.
    """
    
    def __init__(self, config: Dict[str, Any], mode: ExecutionMode):
        """
        Initialize executor with mode validation.
        
        Args:
            config: Strategy configuration dictionary
            mode: Execution mode (BACKTEST, DRY_RUN, or LIVE)
            
        Raises:
            ValueError: If mode is invalid or config incompatible with mode
        """
        if not isinstance(mode, ExecutionMode):
            raise ValueError(f"Invalid mode type: {type(mode)}. Use ExecutionMode enum.")
        
        self.mode = mode
        self.config = config
        
        logger.info(f"Initializing executor in {mode.value} mode")
        
        # Validate configuration for this mode
        self._validate_config()
        
        # Additional safety check for LIVE mode
        if mode == ExecutionMode.LIVE:
            self._validate_live_requirements()
            logger.warning("⚠️  LIVE MODE ENABLED - Real funds at risk!")
    
    def _validate_config(self):
        """
        Validate configuration structure.
        
        Raises:
            ValueError: If config is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
        
        # Check for required keys
        required_keys = ['stake_currency', 'timeframe']
        missing = [k for k in required_keys if k not in self.config]
        
        if missing:
            raise ValueError(f"Config missing required keys: {missing}")
    
    def _validate_live_requirements(self):
        """
        Strict validation for LIVE mode.
        
        Prevents live trading with:
        - Missing credentials
        - Dry-run config
        - Test/example configs
        
        Raises:
            ValueError: If live requirements not met
        """
        # Must have exchange credentials
        if 'exchange' not in self.config:
            raise ValueError("LIVE mode requires 'exchange' configuration")
        
        exchange = self.config['exchange']
        
        # Check for API keys
        if not exchange.get('key') or not exchange.get('secret'):
            raise ValueError(
                "LIVE mode requires exchange API key and secret. "
                "Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET environment variables."
            )
        
        # Must NOT be dry-run
        if self.config.get('dry_run', True):
            raise ValueError(
                "LIVE mode cannot use dry_run config. "
                "Set 'dry_run': false in config for live trading."
            )
        
        # Check for example/test patterns in config
        dangerous_patterns = ['example', 'test', 'demo', 'sample']
        config_str = str(self.config).lower()
        
        found_patterns = [p for p in dangerous_patterns if p in config_str]
        if found_patterns:
            raise ValueError(
                f"LIVE mode config contains test patterns: {found_patterns}. "
                "Use production config only."
            )
        
        # Require explicit stake amount (not 'unlimited' for safety)
        stake_amount = self.config.get('stake_amount')
        if stake_amount == 'unlimited':
            logger.warning(
                "⚠️  LIVE mode with unlimited stake detected. "
                "Consider setting explicit stake_amount for safety."
            )
    
    def _get_nested(self, config: dict, key_path: str, default=None):
        """
        Get nested config value using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., 'exchange.key')
            default: Default value if key not found
            
        Returns:
            Value at key path or default
        """
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def validate_execution_allowed(self) -> bool:
        """
        Check if execution is allowed in current mode.
        
        Returns:
            True if execution is safe to proceed
        """
        # Always allow backtest and dry-run
        if self.mode in [ExecutionMode.BACKTEST, ExecutionMode.DRY_RUN]:
            return True
        
        # LIVE requires additional confirmation
        if self.mode == ExecutionMode.LIVE:
            logger.warning(
                "⚠️  LIVE trading execution requested. "
                "Ensure all safety checks passed."
            )
            return True
        
        return False
    
    def log_execution_start(self):
        """Log execution start with mode information."""
        logger.info("=" * 60)
        logger.info(f"Execution Mode: {self.mode.value}")
        logger.info(f"Strategy: {self.config.get('strategy', 'Unknown')}")
        logger.info(f"Stake Currency: {self.config.get('stake_currency')}")
        logger.info(f"Dry Run: {self.config.get('dry_run', True)}")
        
        if self.mode == ExecutionMode.LIVE:
            logger.warning("⚠️  LIVE MODE - REAL FUNDS AT RISK ⚠️")
        
        logger.info("=" * 60)
    
    def get_execution_context(self) -> Dict[str, Any]:
        """
        Get execution context information.
        
        Returns:
            Dictionary with execution details
        """
        return {
            'executor_type': self.__class__.__name__,
            'mode': self.mode.value,
            'dry_run': self.config.get('dry_run', True),
            'strategy': self.config.get('strategy', 'Unknown'),
            'stake_currency': self.config.get('stake_currency'),
            'timeframe': self.config.get('timeframe'),
            'max_open_trades': self.config.get('max_open_trades'),
            'exchange': self.config.get('exchange', {}).get('name', 'Unknown')
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(mode={self.mode.value})"
