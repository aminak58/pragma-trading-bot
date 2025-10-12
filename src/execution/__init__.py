"""
Execution Module - Backtest vs Live Separation

This module provides clear separation between backtest/simulation
and live trading execution to prevent accidental live trading.
"""

from .base import ExecutionMode
from .simulated import SimulatedExecutor
from .live import LiveExecutor

__all__ = [
    "ExecutionMode",
    "SimulatedExecutor",
    "LiveExecutor"
]
