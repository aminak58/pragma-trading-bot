"""
Scientific Trading Strategy Framework - Utility Functions

This module provides utility functions for the scientific trading strategy framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats

def generate_sample_trades(n_trades: int = 1000, 
                          win_rate: float = 0.55,
                          avg_win: float = 0.02,
                          avg_loss: float = -0.015) -> pd.DataFrame:
    """
    Generate sample trade data for testing
    
    Args:
        n_trades: Number of trades to generate
        win_rate: Win rate (0-1)
        avg_win: Average winning trade return
        avg_loss: Average losing trade return
        
    Returns:
        DataFrame with sample trade data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate trade outcomes
    wins = np.random.random(n_trades) < win_rate
    
    # Generate PnL
    pnl = np.zeros(n_trades)
    pnl[wins] = np.random.normal(avg_win, avg_win * 0.3, np.sum(wins))
    pnl[~wins] = np.random.normal(avg_loss, abs(avg_loss) * 0.3, np.sum(~wins))
    
    # Generate trade data
    trades = pd.DataFrame({
        'pnl': pnl,
        'size': np.random.uniform(0.01, 0.05, n_trades),
        'entry_time': pd.date_range(start='2020-01-01', periods=n_trades, freq='H'),
        'exit_time': pd.date_range(start='2020-01-01', periods=n_trades, freq='H') + timedelta(hours=1),
        'duration': np.random.uniform(0.5, 24, n_trades)
    })
    
    return trades

def generate_sample_returns(n_periods: int = 1000, 
                           mean_return: float = 0.001,
                           volatility: float = 0.02) -> pd.Series:
    """
    Generate sample return series for testing
    
    Args:
        n_periods: Number of periods
        mean_return: Mean daily return
        volatility: Daily volatility
        
    Returns:
        Series of returns
    """
    np.random.seed(42)  # For reproducible results
    
    returns = np.random.normal(mean_return, volatility, n_periods)
    return pd.Series(returns, index=pd.date_range(start='2020-01-01', periods=n_periods))

def generate_sample_positions(n_positions: int = 100) -> pd.DataFrame:
    """
    Generate sample position data for testing
    
    Args:
        n_positions: Number of positions
        
    Returns:
        DataFrame with sample position data
    """
    np.random.seed(42)  # For reproducible results
    
    positions = pd.DataFrame({
        'size': np.random.uniform(0.01, 0.05, n_positions),
        'pnl': np.random.normal(0, 0.01, n_positions),
        'entry_time': pd.date_range(start='2020-01-01', periods=n_positions, freq='H')
    })
    
    return positions

def calculate_rolling_metrics(data: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate rolling performance metrics
    
    Args:
        data: Time series data
        window: Rolling window size
        
    Returns:
        Dictionary of rolling metrics
    """
    return {
        'mean': data.rolling(window=window).mean(),
        'std': data.rolling(window=window).std(),
        'min': data.rolling(window=window).min(),
        'max': data.rolling(window=window).max(),
        'skew': data.rolling(window=window).skew(),
        'kurt': data.rolling(window=window).kurt()
    }

def detect_regime_changes(data: pd.Series, threshold: float = 0.02) -> List[int]:
    """
    Detect regime changes in time series
    
    Args:
        data: Time series data
        threshold: Threshold for regime change detection
        
    Returns:
        List of indices where regime changes occur
    """
    # Calculate rolling statistics
    rolling_mean = data.rolling(window=20).mean()
    rolling_std = data.rolling(window=20).std()
    
    # Detect changes in mean and volatility
    mean_changes = np.abs(rolling_mean.diff()) > threshold
    vol_changes = np.abs(rolling_std.diff()) > threshold * 0.5
    
    # Combine changes
    regime_changes = mean_changes | vol_changes
    
    return regime_changes[regime_changes].index.tolist()

def calculate_regime_metrics(data: pd.Series, regimes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different regimes
    
    Args:
        data: Time series data
        regimes: List of regime labels
        
    Returns:
        Dictionary of regime metrics
    """
    regime_metrics = {}
    
    for regime in set(regimes):
        regime_data = data[regimes == regime]
        
        if len(regime_data) > 0:
            regime_metrics[regime] = {
                'mean': regime_data.mean(),
                'std': regime_data.std(),
                'min': regime_data.min(),
                'max': regime_data.max(),
                'skew': regime_data.skew(),
                'kurt': regime_data.kurtosis(),
                'count': len(regime_data)
            }
    
    return regime_metrics

def calculate_correlation_matrix(data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Calculate correlation matrix with significance testing
    
    Args:
        data: DataFrame with numeric columns
        
    Returns:
        Tuple of (correlation_matrix, p_values)
    """
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    # Calculate p-values for correlations
    n = len(data)
    p_values = np.zeros_like(corr_matrix)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                corr = corr_matrix.iloc[i, j]
                if not np.isnan(corr):
                    # Calculate t-statistic
                    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                    # Calculate p-value
                    p_values[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_values[i, j] = np.nan
    
    return corr_matrix, p_values

def calculate_portfolio_metrics(returns: pd.Series, weights: Optional[pd.Series] = None) -> Dict[str, float]:
    """
    Calculate portfolio-level metrics
    
    Args:
        returns: Series of returns
        weights: Optional series of weights
        
    Returns:
        Dictionary of portfolio metrics
    """
    if weights is None:
        weights = pd.Series(1.0, index=returns.index)
    
    # Calculate weighted returns
    weighted_returns = returns * weights
    
    # Calculate metrics
    metrics = {
        'total_return': weighted_returns.sum(),
        'mean_return': weighted_returns.mean(),
        'volatility': weighted_returns.std(),
        'sharpe_ratio': weighted_returns.mean() / weighted_returns.std() if weighted_returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(weighted_returns),
        'var_95': np.percentile(weighted_returns, 5),
        'cvar_95': weighted_returns[weighted_returns <= np.percentile(weighted_returns, 5)].mean()
    }
    
    return metrics

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown as negative percentage
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = cumulative.expanding().max()
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()

def calculate_rolling_sharpe(returns: pd.Series, window: int = 252, 
                           risk_free_rate: float = 0.02) -> pd.Series:
    """
    Calculate rolling Sharpe ratio
    
    Args:
        returns: Series of returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Series of rolling Sharpe ratios
    """
    # Calculate rolling mean and std
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize returns and risk-free rate
    annual_return = rolling_mean * 252
    annual_vol = rolling_std * np.sqrt(252)
    annual_rf = risk_free_rate
    
    # Calculate Sharpe ratio
    sharpe = (annual_return - annual_rf) / annual_vol
    
    return sharpe

def calculate_regime_transition_matrix(regimes: List[str]) -> pd.DataFrame:
    """
    Calculate regime transition matrix
    
    Args:
        regimes: List of regime labels
        
    Returns:
        Transition matrix DataFrame
    """
    # Create transition matrix
    unique_regimes = list(set(regimes))
    n_regimes = len(unique_regimes)
    
    transition_matrix = np.zeros((n_regimes, n_regimes))
    
    # Count transitions
    for i in range(len(regimes) - 1):
        current_regime = regimes[i]
        next_regime = regimes[i + 1]
        
        current_idx = unique_regimes.index(current_regime)
        next_idx = unique_regimes.index(next_regime)
        
        transition_matrix[current_idx, next_idx] += 1
    
    # Normalize to probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]
    
    # Create DataFrame
    transition_df = pd.DataFrame(
        transition_matrix,
        index=unique_regimes,
        columns=unique_regimes
    )
    
    return transition_df

def calculate_regime_persistence(regimes: List[str]) -> Dict[str, float]:
    """
    Calculate regime persistence metrics
    
    Args:
        regimes: List of regime labels
        
    Returns:
        Dictionary of persistence metrics
    """
    unique_regimes = list(set(regimes))
    persistence_metrics = {}
    
    for regime in unique_regimes:
        # Find regime periods
        regime_periods = []
        current_period_start = None
        
        for i, r in enumerate(regimes):
            if r == regime:
                if current_period_start is None:
                    current_period_start = i
            else:
                if current_period_start is not None:
                    regime_periods.append(i - current_period_start)
                    current_period_start = None
        
        # Handle case where regime continues to end
        if current_period_start is not None:
            regime_periods.append(len(regimes) - current_period_start)
        
        # Calculate persistence metrics
        if regime_periods:
            persistence_metrics[regime] = {
                'avg_duration': np.mean(regime_periods),
                'max_duration': np.max(regime_periods),
                'min_duration': np.min(regime_periods),
                'std_duration': np.std(regime_periods),
                'num_periods': len(regime_periods)
            }
        else:
            persistence_metrics[regime] = {
                'avg_duration': 0,
                'max_duration': 0,
                'min_duration': 0,
                'std_duration': 0,
                'num_periods': 0
            }
    
    return persistence_metrics

def validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Dictionary of quality metrics
    """
    quality_metrics = {
        'total_records': len(data),
        'missing_values': data.isnull().sum().sum(),
        'missing_ratio': data.isnull().sum().sum() / (len(data) * len(data.columns)),
        'duplicate_records': data.duplicated().sum(),
        'data_types': data.dtypes.to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'quality_score': 0.0
    }
    
    # Calculate quality score
    quality_score = 1.0
    
    # Penalize missing values
    quality_score -= quality_metrics['missing_ratio'] * 0.3
    
    # Penalize duplicates
    quality_score -= (quality_metrics['duplicate_records'] / quality_metrics['total_records']) * 0.2
    
    # Penalize mixed data types
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(data.columns) * 0.8:
        quality_score -= 0.1
    
    quality_metrics['quality_score'] = max(0.0, quality_score)
    
    return quality_metrics

def format_performance_report(metrics: Dict[str, Any]) -> str:
    """
    Format performance metrics as a readable report
    
    Args:
        metrics: Dictionary of performance metrics
        
    Returns:
        Formatted report string
    """
    report = "=== PERFORMANCE REPORT ===\n\n"
    
    # Basic metrics
    report += "Basic Metrics:\n"
    report += f"  Total Trades: {metrics.get('num_trades', 0):,}\n"
    report += f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n"
    report += f"  Total Return: {metrics.get('total_return', 0):.2%}\n"
    report += f"  CAGR: {metrics.get('cagr', 0):.2%}\n\n"
    
    # Risk metrics
    report += "Risk Metrics:\n"
    report += f"  Volatility: {metrics.get('volatility', 0):.2%}\n"
    report += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
    report += f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
    report += f"  VaR (95%): {metrics.get('var_95', 0):.2%}\n"
    report += f"  CVaR (95%): {metrics.get('cvar_95', 0):.2%}\n\n"
    
    # Additional metrics
    report += "Additional Metrics:\n"
    report += f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
    report += f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}\n"
    report += f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}\n"
    report += f"  Omega Ratio: {metrics.get('omega_ratio', 0):.2f}\n"
    
    return report

def create_summary_statistics(data: Union[pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics
    
    Args:
        data: Series or DataFrame
        
    Returns:
        Dictionary of summary statistics
    """
    if isinstance(data, pd.Series):
        stats = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'median': data.median(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'skew': data.skew(),
            'kurt': data.kurtosis()
        }
    else:
        stats = {}
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    'count': len(data[col]),
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75),
                    'skew': data[col].skew(),
                    'kurt': data[col].kurtosis()
                }
    
    return stats