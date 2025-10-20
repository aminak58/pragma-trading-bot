"""
Scientific Trading Strategy Framework

This package provides a comprehensive framework for developing and validating
trading strategies using scientific methodology to avoid over-optimization
and curve fitting.

Main Components:
- StrategyTester: Main testing framework
- DataManager: Data preparation and quality control
- PerformanceAnalyzer: Performance metrics and analysis
- RiskManager: Risk management and monitoring
- ValidationEngine: Statistical validation and testing
"""

from .framework import StrategyTester
from .data_manager import DataManager
from .performance_analyzer import PerformanceAnalyzer
from .risk_manager import RiskManager
from .validation_engine import ValidationEngine
from .example_strategy import ExampleScientificStrategy
from .utils import (
    generate_sample_trades,
    generate_sample_returns,
    generate_sample_positions,
    calculate_rolling_metrics,
    detect_regime_changes,
    calculate_regime_metrics,
    calculate_correlation_matrix,
    calculate_portfolio_metrics,
    calculate_max_drawdown,
    calculate_rolling_sharpe,
    calculate_regime_transition_matrix,
    calculate_regime_persistence,
    validate_data_quality,
    format_performance_report,
    create_summary_statistics
)

__all__ = [
    'StrategyTester',
    'DataManager',
    'PerformanceAnalyzer',
    'RiskManager',
    'ValidationEngine',
    'ExampleScientificStrategy',
    'generate_sample_trades',
    'generate_sample_returns',
    'generate_sample_positions',
    'calculate_rolling_metrics',
    'detect_regime_changes',
    'calculate_regime_metrics',
    'calculate_correlation_matrix',
    'calculate_portfolio_metrics',
    'calculate_max_drawdown',
    'calculate_rolling_sharpe',
    'calculate_regime_transition_matrix',
    'calculate_regime_persistence',
    'validate_data_quality',
    'format_performance_report',
    'create_summary_statistics'
]

__version__ = '1.0.0'
__author__ = 'Pragma Trading Bot Team'
__description__ = 'Scientific Trading Strategy Framework'