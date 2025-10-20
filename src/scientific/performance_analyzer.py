"""
Scientific Trading Strategy Framework - Performance Analyzer

This module implements the PerformanceAnalyzer class for calculating
comprehensive performance metrics, statistical analysis, and confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import norm, t
import warnings

class PerformanceAnalyzer:
    """
    Performance metrics and analysis for scientific strategy validation
    
    This class handles:
    - Calculation of comprehensive performance metrics
    - Statistical significance testing
    - Confidence interval calculation
    - Risk-adjusted performance analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PerformanceAnalyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.metrics = {}
        self.confidence_intervals = {}
        self.statistical_tests = {}
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def calculate_metrics(self, trades: pd.DataFrame, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: DataFrame containing trade information
            returns: Series of returns
            
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Calculating performance metrics...")
        
        # Basic trade metrics
        self.metrics['num_trades'] = len(trades)
        self.metrics['winning_trades'] = len(trades[trades['pnl'] > 0])
        self.metrics['losing_trades'] = len(trades[trades['pnl'] <= 0])
        self.metrics['win_rate'] = self.metrics['winning_trades'] / max(1, self.metrics['num_trades'])
        
        # PnL metrics
        if len(trades) > 0:
            self.metrics['total_pnl'] = trades['pnl'].sum()
            self.metrics['avg_win'] = trades[trades['pnl'] > 0]['pnl'].mean() if self.metrics['winning_trades'] > 0 else 0
            self.metrics['avg_loss'] = trades[trades['pnl'] <= 0]['pnl'].mean() if self.metrics['losing_trades'] > 0 else 0
            self.metrics['profit_factor'] = abs(self.metrics['avg_win'] * self.metrics['winning_trades']) / max(1, abs(self.metrics['avg_loss'] * self.metrics['losing_trades']))
        else:
            self.metrics.update({
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            })
        
        # Return metrics
        if len(returns) > 0:
            self.metrics['total_return'] = returns.sum()
            self.metrics['avg_return'] = returns.mean()
            self.metrics['volatility'] = returns.std()
            self.metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
            self.metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
            self.metrics['cagr'] = self.calculate_cagr(returns)
        else:
            self.metrics.update({
                'total_return': 0,
                'avg_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'cagr': 0
            })
        
        # Risk metrics
        if len(returns) > 0:
            self.metrics['var_95'] = self.calculate_var(returns, 0.95)
            self.metrics['cvar_95'] = self.calculate_cvar(returns, 0.95)
            self.metrics['skewness'] = self.calculate_skewness(returns)
            self.metrics['kurtosis'] = self.calculate_kurtosis(returns)
        else:
            self.metrics.update({
                'var_95': 0,
                'cvar_95': 0,
                'skewness': 0,
                'kurtosis': 0
            })
        
        # Additional metrics
        self.metrics['calmar_ratio'] = self.calculate_calmar_ratio()
        self.metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        self.metrics['omega_ratio'] = self.calculate_omega_ratio(returns)
        
        self.logger.info(f"Performance metrics calculated. Win Rate: {self.metrics['win_rate']:.2%}")
        return self.metrics
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualize returns and risk-free rate
        annual_return = returns.mean() * 252  # Assuming daily returns
        annual_volatility = returns.std() * np.sqrt(252)
        excess_return = annual_return - self.risk_free_rate
        
        return excess_return / annual_volatility if annual_volatility > 0 else 0.0
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
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
    
    def calculate_cagr(self, returns: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            returns: Series of returns
            
        Returns:
            CAGR as percentage
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate total return
        total_return = (1 + returns).prod() - 1
        
        # Calculate years
        years = len(returns) / 252  # Assuming daily returns
        
        # Calculate CAGR
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        
        return cagr
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR as negative percentage
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)
            
        Returns:
            CVaR as negative percentage
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """
        Calculate skewness of returns
        
        Args:
            returns: Series of returns
            
        Returns:
            Skewness value
        """
        if len(returns) < 3:
            return 0.0
        
        return stats.skew(returns)
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """
        Calculate kurtosis of returns
        
        Args:
            returns: Series of returns
            
        Returns:
            Kurtosis value
        """
        if len(returns) < 4:
            return 0.0
        
        return stats.kurtosis(returns)
    
    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (CAGR / Max Drawdown)
        
        Returns:
            Calmar ratio
        """
        cagr = self.metrics.get('cagr', 0)
        max_dd = abs(self.metrics.get('max_drawdown', 0))
        
        return cagr / max_dd if max_dd > 0 else 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio
        
        Args:
            returns: Series of returns
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Calculate excess return
        annual_return = returns.mean() * 252
        excess_return = annual_return - self.risk_free_rate
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0.0
    
    def calculate_omega_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Omega ratio
        
        Args:
            returns: Series of returns
            
        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate gains and losses
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        return gains / losses if losses > 0 else float('inf')
    
    def calculate_confidence_intervals(self, metric: str, confidence: float = 0.95, 
                                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate confidence intervals for metrics using bootstrap
        
        Args:
            metric: Metric name
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if metric not in self.metrics:
            return 0.0, 0.0
        
        # For now, return simple confidence intervals based on normal distribution
        # In practice, this would use bootstrap sampling
        value = self.metrics[metric]
        
        if metric in ['win_rate', 'total_return', 'avg_return']:
            # For proportions and means, use normal approximation
            if metric == 'win_rate':
                n = self.metrics.get('num_trades', 1)
                se = np.sqrt(value * (1 - value) / n) if n > 0 else 0
            else:
                se = self.metrics.get('volatility', 0) / np.sqrt(self.metrics.get('num_trades', 1))
            
            z_score = norm.ppf((1 + confidence) / 2)
            margin = z_score * se
            
            ci_lower = value - margin
            ci_upper = value + margin
        else:
            # For other metrics, use simple approximation
            margin = abs(value) * 0.1  # 10% margin
            ci_lower = value - margin
            ci_upper = value + margin
        
        self.confidence_intervals[metric] = (ci_lower, ci_upper)
        return ci_lower, ci_upper
    
    def perform_statistical_tests(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Perform statistical significance tests
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of statistical test results
        """
        if len(returns) < 30:  # Minimum sample size for meaningful tests
            return {'insufficient_data': True}
        
        test_results = {}
        
        # Test for normality
        try:
            shapiro_stat, shapiro_p = stats.shapiro(returns)
            test_results['normality'] = {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        except:
            test_results['normality'] = {'error': 'Shapiro test failed'}
        
        # Test for zero mean
        try:
            t_stat, t_p = stats.ttest_1samp(returns, 0)
            test_results['zero_mean'] = {
                't_stat': t_stat,
                't_p': t_p,
                'is_zero_mean': t_p > 0.05
            }
        except:
            test_results['zero_mean'] = {'error': 'T-test failed'}
        
        # Test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljungbox_result = acorr_ljungbox(returns, lags=10, return_df=True)
            test_results['autocorrelation'] = {
                'ljungbox_stat': ljungbox_result['lb_stat'].iloc[-1],
                'ljungbox_p': ljungbox_result['lb_pvalue'].iloc[-1],
                'has_autocorrelation': ljungbox_result['lb_pvalue'].iloc[-1] < 0.05
            }
        except:
            test_results['autocorrelation'] = {'error': 'Ljung-Box test failed'}
        
        self.statistical_tests = test_results
        return test_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        report = {
            'summary': {
                'total_trades': self.metrics.get('num_trades', 0),
                'win_rate': self.metrics.get('win_rate', 0),
                'total_return': self.metrics.get('total_return', 0),
                'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
                'max_drawdown': self.metrics.get('max_drawdown', 0),
                'cagr': self.metrics.get('cagr', 0)
            },
            'risk_metrics': {
                'volatility': self.metrics.get('volatility', 0),
                'var_95': self.metrics.get('var_95', 0),
                'cvar_95': self.metrics.get('cvar_95', 0),
                'skewness': self.metrics.get('skewness', 0),
                'kurtosis': self.metrics.get('kurtosis', 0)
            },
            'risk_adjusted_metrics': {
                'sharpe_ratio': self.metrics.get('sharpe_ratio', 0),
                'sortino_ratio': self.metrics.get('sortino_ratio', 0),
                'calmar_ratio': self.metrics.get('calmar_ratio', 0),
                'omega_ratio': self.metrics.get('omega_ratio', 0)
            },
            'confidence_intervals': self.confidence_intervals,
            'statistical_tests': self.statistical_tests
        }
        
        return report
    
    def check_performance_quality(self) -> Dict[str, Any]:
        """
        Check performance quality and identify potential issues
        
        Returns:
            Quality assessment dictionary
        """
        quality_issues = []
        quality_score = 1.0
        
        # Check for unrealistic performance
        if self.metrics.get('win_rate', 0) > 0.8:
            quality_issues.append("Win rate suspiciously high (>80%)")
            quality_score -= 0.3
        
        if self.metrics.get('max_drawdown', 0) > -0.02:
            quality_issues.append("Max drawdown suspiciously low (<2%)")
            quality_score -= 0.3
        
        if self.metrics.get('sharpe_ratio', 0) > 3.0:
            quality_issues.append("Sharpe ratio suspiciously high (>3.0)")
            quality_score -= 0.2
        
        # Check for insufficient data
        if self.metrics.get('num_trades', 0) < 100:
            quality_issues.append("Insufficient number of trades (<100)")
            quality_score -= 0.2
        
        # Check for statistical significance
        if 'zero_mean' in self.statistical_tests:
            if not self.statistical_tests['zero_mean'].get('is_zero_mean', True):
                quality_issues.append("Returns not statistically different from zero")
                quality_score -= 0.1
        
        return {
            'quality_score': max(0.0, quality_score),
            'issues': quality_issues,
            'status': 'good' if quality_score > 0.7 else 'poor'
        }