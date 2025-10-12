"""
Simulated Executor - Backtest and Dry-Run

This executor is for backtesting and paper trading only.
It cannot execute live trades.
"""

from typing import Dict, Any
import logging
from .base import BaseExecutor, ExecutionMode

logger = logging.getLogger(__name__)


class SimulatedExecutor(BaseExecutor):
    """
    Executor for simulated trading (backtest and dry-run).
    
    This class is explicitly for testing and simulation.
    It enforces dry_run=True and prevents live trading.
    
    Use Cases:
    - Backtesting historical data
    - Paper trading (dry-run)
    - Strategy validation
    - Parameter optimization
    
    Safety: Cannot execute real trades
    """
    
    def __init__(self, config: Dict[str, Any], mode: str = "backtest"):
        """
        Initialize simulated executor.
        
        Args:
            config: Strategy configuration
            mode: Either "backtest" or "dry-run"
            
        Raises:
            ValueError: If mode is "live" or invalid
        """
        # Convert string mode to enum
        if mode == "backtest":
            execution_mode = ExecutionMode.BACKTEST
        elif mode == "dry-run":
            execution_mode = ExecutionMode.DRY_RUN
        else:
            raise ValueError(
                f"SimulatedExecutor only supports 'backtest' or 'dry-run'. "
                f"Got: {mode}. For live trading, use LiveExecutor."
            )
        
        # Ensure dry_run is True
        if 'dry_run' not in config:
            config['dry_run'] = True
            logger.info("Automatically set dry_run=True for simulation")
        elif not config['dry_run']:
            logger.warning("Forcing dry_run=True for SimulatedExecutor")
            config['dry_run'] = True
        
        super().__init__(config, execution_mode)
        
        logger.info(
            f"SimulatedExecutor initialized in {mode} mode. "
            "No real trades will be executed."
        )
    
    def execute(self):
        """
        Execute simulated trading.
        
        This is a placeholder for actual Freqtrade execution logic.
        In production, this would interface with Freqtrade's backtesting
        or dry-run functionality.
        """
        self.log_execution_start()
        
        logger.info("Simulated execution starting...")
        logger.info("Safe mode: No real funds at risk")
        
        # Validation
        if not self.validate_execution_allowed():
            raise RuntimeError("Execution not allowed")
        
        # Here would go actual Freqtrade integration
        # For now, just log
        logger.info(f"Mode: {self.mode.value}")
        logger.info("Strategy would execute here (simulated)")
        
        return True
    
    def can_execute_live_trades(self) -> bool:
        """
        Check if this executor can place live trades.
        
        Returns:
            False (always - this is simulated only)
        """
        return False
    
    def get_execution_context(self) -> Dict[str, Any]:
        """
        Get execution context information.
        
        Returns:
            Dictionary with execution details
        """
        return {
            'executor_type': 'SimulatedExecutor',
            'mode': self.mode.value,
            'dry_run': self.config.get('dry_run'),
            'can_execute_live': False,
            'safe_for_testing': True
        }
