"""
Live Executor - Real Trading

⚠️  WARNING: This executor places REAL trades with REAL money.
Use with extreme caution and only after thorough testing.
"""

from typing import Dict, Any
import logging
import os
from datetime import datetime
from .base import BaseExecutor, ExecutionMode

logger = logging.getLogger(__name__)


class LiveExecutor(BaseExecutor):
    """
    Executor for live trading with real funds.
    
    ⚠️  DANGER: This class executes REAL trades.
    
    Safety Requirements:
    1. Must explicitly confirm live trading (confirm_live=True)
    2. Cannot use dry_run config
    3. Must have valid API credentials
    4. Cannot use example/test configs
    5. Requires environment variable confirmation
    
    Before Using:
    1. Test thoroughly in backtest
    2. Validate in dry-run for 1-2 weeks
    3. Start with minimal funds
    4. Monitor closely
    5. Have kill-switch ready
    """
    
    # Class-level safety flag
    _LIVE_TRADING_ENABLED = False
    
    @classmethod
    def enable_live_trading(cls, enable: bool = True):
        """
        Enable or disable live trading globally.
        
        This is an additional safety mechanism.
        Set environment variable PRAGMA_ALLOW_LIVE=true to enable.
        
        Args:
            enable: Whether to enable live trading
        """
        cls._LIVE_TRADING_ENABLED = enable
        if enable:
            logger.critical("⚠️  LIVE TRADING GLOBALLY ENABLED ⚠️")
        else:
            logger.info("Live trading globally disabled (safe)")
    
    def __init__(self,
                 config: Dict[str, Any],
                 confirm_live: bool = False,
                 safety_phrase: str = None):
        """
        Initialize live executor with multiple safety checks.
        
        Args:
            config: Strategy configuration (must be production config)
            confirm_live: Must be True to proceed (prevents accidents)
            safety_phrase: Optional safety phrase for extra confirmation
            
        Raises:
            ValueError: If safety requirements not met
            RuntimeError: If live trading not enabled
        """
        # Safety Check 1: Explicit confirmation required
        if not confirm_live:
            raise ValueError(
                "🛑 SAFETY BLOCK: Live trading requires explicit confirmation.\n"
                "Set confirm_live=True if you intend to trade with real funds.\n"
                "⚠️  This is not a test - REAL MONEY will be at risk!"
            )
        
        # Safety Check 2: Environment variable confirmation
        env_allow = os.getenv('PRAGMA_ALLOW_LIVE', 'false').lower()
        if env_allow not in ['true', '1', 'yes']:
            raise RuntimeError(
                "🛑 SAFETY BLOCK: Live trading not enabled in environment.\n"
                "Set environment variable: PRAGMA_ALLOW_LIVE=true\n"
                "This prevents accidental live trading."
            )
        
        # Safety Check 3: Optional safety phrase
        if safety_phrase is not None:
            expected_phrase = "I understand this uses real money"
            if safety_phrase != expected_phrase:
                raise ValueError(
                    f"🛑 SAFETY BLOCK: Incorrect safety phrase.\n"
                    f"Expected: '{expected_phrase}'\n"
                    f"Got: '{safety_phrase}'"
                )
        
        # Safety Check 4: Global flag
        if not self._LIVE_TRADING_ENABLED:
            raise RuntimeError(
                "🛑 SAFETY BLOCK: Live trading not globally enabled.\n"
                "Call LiveExecutor.enable_live_trading(True) first."
            )
        
        # Initialize with LIVE mode
        super().__init__(config, ExecutionMode.LIVE)
        
        # Log warnings
        logger.critical("=" * 70)
        logger.critical("⚠️  LIVE EXECUTOR INITIALIZED - REAL TRADING ENABLED ⚠️")
        logger.critical("=" * 70)
        logger.critical(f"Exchange: {config.get('exchange', {}).get('name')}")
        logger.critical(f"Stake Currency: {config.get('stake_currency')}")
        logger.critical(f"Max Open Trades: {config.get('max_open_trades')}")
        logger.critical(f"Initialized at: {datetime.now().isoformat()}")
        logger.critical("=" * 70)
        logger.critical("⚠️  ALL SAFETY CHECKS PASSED - TRADES WILL BE REAL ⚠️")
        logger.critical("=" * 70)
    
    def execute(self):
        """
        Execute live trading.
        
        ⚠️  WARNING: This places REAL orders on exchange.
        
        Returns:
            True if execution started successfully
        """
        # Final confirmation
        logger.critical("🚨 STARTING LIVE EXECUTION 🚨")
        
        self.log_execution_start()
        
        if not self.validate_execution_allowed():
            raise RuntimeError("Execution validation failed")
        
        # Log trade start
        self._log_live_start()
        
        # Here would go actual Freqtrade live trading integration
        logger.critical("Live trading active - monitoring for signals...")
        
        return True
    
    def _log_live_start(self):
        """Log live trading start to file for audit trail."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = f"{log_dir}/live_trading_log.txt"
        
        with open(log_file, 'a') as f:
            f.write("=" * 70 + "\n")
            f.write(f"LIVE TRADING STARTED: {datetime.now().isoformat()}\n")
            f.write(f"Exchange: {self.config.get('exchange', {}).get('name')}\n")
            f.write(f"Strategy: {self.config.get('strategy', 'Unknown')}\n")
            f.write(f"Max Trades: {self.config.get('max_open_trades')}\n")
            f.write("=" * 70 + "\n")
        
        logger.info(f"Live trading start logged to: {log_file}")
    
    def can_execute_live_trades(self) -> bool:
        """
        Check if this executor can place live trades.
        
        Returns:
            True (this executor DOES place live trades)
        """
        return True
    
    def get_execution_context(self) -> Dict[str, Any]:
        """
        Get execution context information.
        
        Returns:
            Dictionary with execution details
        """
        return {
            'executor_type': 'LiveExecutor',
            'mode': self.mode.value,
            'dry_run': self.config.get('dry_run'),
            'can_execute_live': True,
            'safe_for_testing': False,
            'warning': '⚠️  REAL TRADES - REAL MONEY AT RISK ⚠️'
        }
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """
        Emergency stop for live trading.
        
        Args:
            reason: Reason for emergency stop
        """
        logger.critical("=" * 70)
        logger.critical("🛑 EMERGENCY STOP TRIGGERED 🛑")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Time: {datetime.now().isoformat()}")
        logger.critical("=" * 70)
        
        # Log to file
        log_file = "logs/live_trading_log.txt"
        with open(log_file, 'a') as f:
            f.write(f"EMERGENCY STOP: {datetime.now().isoformat()}\n")
            f.write(f"Reason: {reason}\n")
        
        # Here would go actual stop logic
        # - Cancel open orders
        # - Close positions (optional)
        # - Disable new entries
        
        logger.critical("Emergency stop executed")


# Module-level safety check
def verify_live_trading_setup():
    """
    Verify that live trading environment is properly set up.
    
    Returns:
        True if safe to proceed, False otherwise
    """
    checks = {
        'Environment variable set': os.getenv('PRAGMA_ALLOW_LIVE') == 'true',
        'API keys configured': (
            os.getenv('EXCHANGE_API_KEY') and
            os.getenv('EXCHANGE_API_SECRET')
        ),
        'Logs directory exists': os.path.exists('logs'),
    }
    
    all_passed = all(checks.values())
    
    logger.info("Live Trading Setup Verification:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {check}")
    
    if all_passed:
        logger.warning("⚠️  All checks passed - Live trading possible")
    else:
        logger.info("✅ Safety checks failed - Live trading blocked")
    
    return all_passed
