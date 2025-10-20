"""
Scientific Trading Strategy Framework - Validation Engine

This module implements the ValidationEngine class for statistical validation,
significance testing, and performance consistency verification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
from scipy import stats
from scipy.stats import ttest_1samp, chi2_contingency
import warnings

class ValidationEngine:
    """
    Statistical validation and testing for scientific strategy validation
    
    This class handles:
    - Performance consistency validation
    - Statistical significance testing
    - Sample size adequacy verification
    - Performance degradation analysis
    """
    
    def __init__(self, confidence_level: float = 0.95, 
                 min_sample_size: int = 1000,
                 max_degradation: float = 0.20):
        """
        Initialize ValidationEngine
        
        Args:
            confidence_level: Confidence level for statistical tests
            min_sample_size: Minimum sample size for statistical significance
            max_degradation: Maximum acceptable performance degradation
        """
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.max_degradation = max_degradation
        self.validation_results = {}
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def validate_performance(self, in_sample_metrics: Dict[str, Any], 
                           out_of_sample_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate performance consistency between in-sample and out-of-sample
        
        Args:
            in_sample_metrics: In-sample performance metrics
            out_of_sample_metrics: Out-of-sample performance metrics
            
        Returns:
            Validation results dictionary
        """
        self.logger.info("Validating performance consistency...")
        
        validation_results = {}
        
        # Check sample size adequacy
        validation_results['sample_size_valid'] = self.check_sample_size(out_of_sample_metrics)
        
        # Check performance consistency
        validation_results['performance_consistent'] = self.check_performance_consistency(
            in_sample_metrics, out_of_sample_metrics
        )
        
        # Check statistical significance
        validation_results['statistically_significant'] = self.check_statistical_significance(
            out_of_sample_metrics
        )
        
        # Check performance degradation
        validation_results['degradation_acceptable'] = self.check_degradation(
            in_sample_metrics, out_of_sample_metrics
        )
        
        # Check stability across metrics
        validation_results['stability_check'] = self.check_stability(
            in_sample_metrics, out_of_sample_metrics
        )
        
        # Overall validation status
        validation_results['overall_valid'] = all([
            validation_results['sample_size_valid'],
            validation_results['performance_consistent'],
            validation_results['statistically_significant'],
            validation_results['degradation_acceptable']
        ])
        
        self.validation_results = validation_results
        self.logger.info(f"Performance validation completed. Overall valid: {validation_results['overall_valid']}")
        
        return validation_results
    
    def check_sample_size(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if sample size is sufficient for statistical significance
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            True if sample size is adequate
        """
        num_trades = metrics.get('num_trades', 0)
        is_adequate = num_trades >= self.min_sample_size
        
        if not is_adequate:
            self.logger.warning(f"Insufficient sample size: {num_trades} < {self.min_sample_size}")
        
        return is_adequate
    
    def check_performance_consistency(self, in_sample: Dict[str, Any], 
                                    out_of_sample: Dict[str, Any]) -> bool:
        """
        Check if performance is consistent between in-sample and out-of-sample
        
        Args:
            in_sample: In-sample performance metrics
            out_of_sample: Out-of-sample performance metrics
            
        Returns:
            True if performance is consistent
        """
        # Key metrics to check for consistency
        key_metrics = ['win_rate', 'sharpe_ratio', 'profit_factor', 'cagr']
        
        for metric in key_metrics:
            if metric in in_sample and metric in out_of_sample:
                in_value = in_sample[metric]
                out_value = out_of_sample[metric]
                
                # Avoid division by zero
                if abs(in_value) < 1e-10:
                    continue
                
                # Calculate relative degradation
                degradation = abs(out_value - in_value) / abs(in_value)
                
                if degradation > self.max_degradation:
                    self.logger.warning(f"Performance degradation in {metric}: {degradation:.2%} > {self.max_degradation:.2%}")
                    return False
        
        return True
    
    def check_statistical_significance(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if performance metrics are statistically significant
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            True if metrics are statistically significant
        """
        # Check if win rate is significantly different from random (50%)
        win_rate = metrics.get('win_rate', 0.5)
        num_trades = metrics.get('num_trades', 0)
        
        if num_trades < 30:  # Minimum for meaningful statistical test
            return False
        
        # Binomial test for win rate
        try:
            wins = int(win_rate * num_trades)
            p_value = stats.binom_test(wins, num_trades, p=0.5, alternative='two-sided')
            is_significant = p_value < (1 - self.confidence_level)
            
            if not is_significant:
                self.logger.warning(f"Win rate not statistically significant: p-value = {p_value:.4f}")
            
            return is_significant
            
        except Exception as e:
            self.logger.error(f"Error in statistical significance test: {e}")
            return False
    
    def check_degradation(self, in_sample: Dict[str, Any], 
                         out_of_sample: Dict[str, Any]) -> bool:
        """
        Check if performance degradation is acceptable
        
        Args:
            in_sample: In-sample performance metrics
            out_of_sample: Out-of-sample performance metrics
            
        Returns:
            True if degradation is acceptable
        """
        # Check for excessive degradation in key metrics
        key_metrics = ['win_rate', 'sharpe_ratio', 'profit_factor']
        
        for metric in key_metrics:
            if metric in in_sample and metric in out_of_sample:
                in_value = in_sample[metric]
                out_value = out_of_sample[metric]
                
                # Calculate degradation
                if abs(in_value) > 1e-10:
                    degradation = (in_value - out_value) / abs(in_value)
                    
                    # Check if degradation exceeds acceptable threshold
                    if degradation > self.max_degradation:
                        self.logger.warning(f"Excessive degradation in {metric}: {degradation:.2%} > {self.max_degradation:.2%}")
                        return False
        
        return True
    
    def check_stability(self, in_sample: Dict[str, Any], 
                       out_of_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check stability of performance metrics
        
        Args:
            in_sample: In-sample performance metrics
            out_of_sample: Out-of-sample performance metrics
            
        Returns:
            Stability check results
        """
        stability_results = {}
        
        # Check for stability in key metrics
        key_metrics = ['win_rate', 'sharpe_ratio', 'profit_factor', 'max_drawdown']
        
        for metric in key_metrics:
            if metric in in_sample and metric in out_of_sample:
                in_value = in_sample[metric]
                out_value = out_of_sample[metric]
                
                # Calculate stability score (closer to 1 is more stable)
                if abs(in_value) > 1e-10:
                    stability = 1 - abs(out_value - in_value) / abs(in_value)
                else:
                    stability = 1.0 if abs(out_value) < 1e-10 else 0.0
                
                stability_results[metric] = {
                    'in_sample': in_value,
                    'out_of_sample': out_value,
                    'stability_score': stability,
                    'stable': stability > (1 - self.max_degradation)
                }
        
        # Overall stability
        stability_scores = [result['stability_score'] for result in stability_results.values()]
        overall_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        stability_results['overall_stability'] = {
            'stability_score': overall_stability,
            'stable': overall_stability > (1 - self.max_degradation)
        }
        
        return stability_results
    
    def perform_walk_forward_validation(self, data: pd.DataFrame, 
                                      strategy, n_periods: int = 5) -> Dict[str, Any]:
        """
        Perform walk-forward validation
        
        Args:
            data: Full dataset
            strategy: Strategy instance
            n_periods: Number of walk-forward periods
            
        Returns:
            Walk-forward validation results
        """
        self.logger.info(f"Performing walk-forward validation with {n_periods} periods...")
        
        # Split data into periods
        period_length = len(data) // n_periods
        validation_results = []
        
        for i in range(n_periods - 1):  # Leave last period for final test
            # Training data
            train_start = i * period_length
            train_end = (i + 1) * period_length
            train_data = data.iloc[train_start:train_end]
            
            # Testing data
            test_start = train_end
            test_end = (i + 2) * period_length if i < n_periods - 2 else len(data)
            test_data = data.iloc[test_start:test_end]
            
            # Train strategy
            strategy.train(train_data)
            
            # Test strategy
            signals = strategy.generate_signals(test_data)
            trades = strategy.generate_trades(signals, test_data)
            returns = strategy.calculate_returns(trades, test_data)
            
            # Calculate performance
            from .performance_analyzer import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()
            performance = analyzer.calculate_metrics(trades, returns)
            
            validation_results.append({
                'period': i + 1,
                'train_period': (train_data.index.min(), train_data.index.max()),
                'test_period': (test_data.index.min(), test_data.index.max()),
                'performance': performance
            })
        
        # Analyze walk-forward results
        wf_analysis = self._analyze_walk_forward_results(validation_results)
        
        return {
            'validation_results': validation_results,
            'analysis': wf_analysis,
            'n_periods': n_periods
        }
    
    def _analyze_walk_forward_results(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze walk-forward validation results
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Analysis of walk-forward results
        """
        if not validation_results:
            return {'error': 'No validation results to analyze'}
        
        # Extract key metrics
        win_rates = [r['performance'].get('win_rate', 0) for r in validation_results]
        sharpe_ratios = [r['performance'].get('sharpe_ratio', 0) for r in validation_results]
        max_drawdowns = [r['performance'].get('max_drawdown', 0) for r in validation_results]
        num_trades = [r['performance'].get('num_trades', 0) for r in validation_results]
        
        # Calculate statistics
        analysis = {
            'win_rate': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates),
                'min': np.min(win_rates),
                'max': np.max(win_rates),
                'trend': self._calculate_trend(win_rates)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios),
                'trend': self._calculate_trend(sharpe_ratios)
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
                'trend': self._calculate_trend(max_drawdowns)
            },
            'num_trades': {
                'mean': np.mean(num_trades),
                'std': np.std(num_trades),
                'min': np.min(num_trades),
                'max': np.max(num_trades),
                'trend': self._calculate_trend(num_trades)
            },
            'stability': {
                'win_rate_stable': np.std(win_rates) < 0.1,
                'sharpe_stable': np.std(sharpe_ratios) < 0.5,
                'overall_stable': np.std(win_rates) < 0.1 and np.std(sharpe_ratios) < 0.5
            }
        }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction
        
        Args:
            values: List of values
            
        Returns:
            Trend direction ('increasing', 'decreasing', 'stable')
        """
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def validate_regime_robustness(self, regime_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate robustness across different market regimes
        
        Args:
            regime_results: Dictionary of results by regime
            
        Returns:
            Regime robustness validation results
        """
        self.logger.info("Validating regime robustness...")
        
        robustness_results = {}
        
        # Check if strategy works across all regimes
        regimes = list(regime_results.keys())
        regime_performance = {}
        
        for regime, results in regime_results.items():
            regime_performance[regime] = {
                'win_rate': results.get('win_rate', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'num_trades': results.get('num_trades', 0)
            }
        
        # Check for regime-specific failures
        failed_regimes = []
        for regime, perf in regime_performance.items():
            if (perf['win_rate'] < 0.4 or 
                perf['sharpe_ratio'] < 0.5 or 
                perf['max_drawdown'] < -0.2 or
                perf['num_trades'] < 50):
                failed_regimes.append(regime)
        
        # Calculate regime consistency
        win_rates = [perf['win_rate'] for perf in regime_performance.values()]
        sharpe_ratios = [perf['sharpe_ratio'] for perf in regime_performance.values()]
        
        robustness_results = {
            'regime_performance': regime_performance,
            'failed_regimes': failed_regimes,
            'regime_consistency': {
                'win_rate_std': np.std(win_rates),
                'sharpe_std': np.std(sharpe_ratios),
                'consistent': np.std(win_rates) < 0.15 and np.std(sharpe_ratios) < 0.5
            },
            'robust': len(failed_regimes) == 0 and np.std(win_rates) < 0.15
        }
        
        return robustness_results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Returns:
            Validation report dictionary
        """
        return {
            'validation_results': self.validation_results,
            'validation_summary': {
                'overall_valid': self.validation_results.get('overall_valid', False),
                'sample_size_valid': self.validation_results.get('sample_size_valid', False),
                'performance_consistent': self.validation_results.get('performance_consistent', False),
                'statistically_significant': self.validation_results.get('statistically_significant', False),
                'degradation_acceptable': self.validation_results.get('degradation_acceptable', False)
            },
            'recommendations': self.get_validation_recommendations(),
            'timestamp': pd.Timestamp.now()
        }
    
    def get_validation_recommendations(self) -> List[str]:
        """
        Get validation recommendations
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not self.validation_results.get('sample_size_valid', False):
            recommendations.append("Increase sample size for statistical significance")
        
        if not self.validation_results.get('performance_consistent', False):
            recommendations.append("Investigate performance inconsistency between in-sample and out-of-sample")
        
        if not self.validation_results.get('statistically_significant', False):
            recommendations.append("Strategy performance may not be statistically significant")
        
        if not self.validation_results.get('degradation_acceptable', False):
            recommendations.append("Performance degradation exceeds acceptable limits")
        
        if not self.validation_results.get('overall_valid', False):
            recommendations.append("Overall validation failed - strategy needs improvement")
        
        return recommendations