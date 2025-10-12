"""
Trading Strategies Module
Freqtrade strategies with HMM regime detection
"""

from .regime_adaptive_strategy import RegimeAdaptiveStrategy

__all__ = ["RegimeAdaptiveStrategy"]
