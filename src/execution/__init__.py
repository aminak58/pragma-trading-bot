"""
Execution Module - Backtest vs Live Separation

This module provides clear separation between backtest/simulation
and live trading execution to prevent accidental live trading.
"""

from .base import ExecutionMode, BaseExecutor
from .simulated import SimulatedExecutor
from .live import LiveExecutor
from .freqtrade_integration import FreqtradeIntegration, FreqtradeManager

__all__ = [
    "ExecutionMode",
    "BaseExecutor",
    "SimulatedExecutor",
    "LiveExecutor",
    "FreqtradeIntegration",
    "FreqtradeManager"
]
