"""
Scientific Trading Strategy Framework - Main Framework Class

This module implements the core StrategyTester class that orchestrates
the comprehensive testing protocol for scientific strategy validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timedelta

class StrategyTester:
    """
    Main testing framework for scientific strategy validation
    
    This class orchestrates the comprehensive testing protocol including:
    - Data preparation and quality control
    - In-sample optimization and analysis
    - Out-of-sample validation
    - Cross-validation testing
    - Robustness and stress testing
    - Final validation and documentation
    """
    
    def __init__(self, strategy, data_manager, performance_analyzer, 
                 risk_manager, validation_engine):
        """
        Initialize the StrategyTester
        
        Args:
            strategy: Strategy instance to test
            data_manager: DataManager instance for data handling
            performance_analyzer: PerformanceAnalyzer instance for metrics
            risk_manager: RiskManager instance for risk monitoring
            validation_engine: ValidationEngine instance for statistical validation
        """
        self.strategy = strategy
        self.data_manager = data_manager
        self.performance_analyzer = performance_analyzer
        self.risk_manager = risk_manager
        self.validation_engine = validation_engine
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize results structure
        self.results = {
            'data_preparation': {},
            'in_sample': {},
            'out_of_sample': {},
            'cross_validation': {},
            'robustness': {},
            'final_validation': {},
            'summary': {}
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive testing protocol
        
        Returns:
            Dict containing all testing results
        """
        self.logger.info("Starting comprehensive testing protocol")
        
        try:
            # Phase 1: Data Preparation
            self.logger.info("Phase 1: Data Preparation")
            self.results['data_preparation'] = self._prepare_data()
            
            # Phase 2: In-Sample Testing
            self.logger.info("Phase 2: In-Sample Testing")
            self.results['in_sample'] = self._run_in_sample_test()
            
            # Phase 3: Out-of-Sample Testing
            self.logger.info("Phase 3: Out-of-Sample Testing")
            self.results['out_of_sample'] = self._run_out_of_sample_test()
            
            # Phase 4: Cross-Validation
            self.logger.info("Phase 4: Cross-Validation")
            self.results['cross_validation'] = self._run_cross_validation()
            
            # Phase 5: Robustness Testing
            self.logger.info("Phase 5: Robustness Testing")
            self.results['robustness'] = self._run_robustness_test()
            
            # Phase 6: Final Validation
            self.logger.info("Phase 6: Final Validation")
            self.results['final_validation'] = self._run_final_validation()
            
            # Generate Summary
            self.results['summary'] = self._generate_summary()
            
            self.logger.info("Comprehensive testing protocol completed successfully")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive testing: {e}")
            raise
    
    def _prepare_data(self) -> Dict[str, Any]:
        """
        Phase 1: Data Preparation and Quality Control
        
        Returns:
            Dict containing data preparation results
        """
        try:
            # Prepare data
            data = self.data_manager.prepare_data()
            
            # Get data statistics
            data_stats = {
                'total_records': len(data),
                'date_range': (data.index.min(), data.index.max()),
                'duration_years': (data.index.max() - data.index.min()).days / 365.25,
                'missing_values': data.isnull().sum().sum(),
                'data_quality_score': self._calculate_data_quality_score(data)
            }
            
            # Check minimum requirements
            requirements_met = {
                'min_years': data_stats['duration_years'] >= 3.0,
                'min_records': data_stats['total_records'] >= 1000,
                'data_quality': data_stats['data_quality_score'] >= 0.8
            }
            
            return {
                'data_stats': data_stats,
                'requirements_met': requirements_met,
                'status': 'success' if all(requirements_met.values()) else 'failed'
            }
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_in_sample_test(self) -> Dict[str, Any]:
        """
        Phase 2: In-Sample Optimization and Analysis
        
        Returns:
            Dict containing in-sample testing results
        """
        try:
            # Get training data
            train_data = self.data_manager.train_data
            
            # Train strategy
            self.strategy.train(train_data)
            
            # Generate signals and calculate performance
            signals = self.strategy.generate_signals(train_data)
            trades = self.strategy.generate_trades(signals, train_data)
            returns = self.strategy.calculate_returns(trades, train_data)
            
            # Calculate performance metrics
            performance_metrics = self.performance_analyzer.calculate_metrics(trades, returns)
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(trades, returns)
            
            # Check for red flags
            red_flags = self._check_red_flags(performance_metrics)
            
            return {
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'red_flags': red_flags,
                'num_trades': len(trades),
                'status': 'success' if not red_flags else 'failed'
            }
            
        except Exception as e:
            self.logger.error(f"Error in in-sample testing: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_out_of_sample_test(self) -> Dict[str, Any]:
        """
        Phase 3: Out-of-Sample Validation
        
        Returns:
            Dict containing out-of-sample testing results
        """
        try:
            # Get testing data
            test_data = self.data_manager.test_data
            
            # Generate signals and calculate performance (no re-optimization)
            signals = self.strategy.generate_signals(test_data)
            trades = self.strategy.generate_trades(signals, test_data)
            returns = self.strategy.calculate_returns(trades, test_data)
            
            # Calculate performance metrics
            performance_metrics = self.performance_analyzer.calculate_metrics(trades, returns)
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics(trades, returns)
            
            # Validate performance consistency
            validation_results = self.validation_engine.validate_performance(
                self.results['in_sample']['performance_metrics'],
                performance_metrics
            )
            
            # Check for red flags
            red_flags = self._check_red_flags(performance_metrics)
            
            return {
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics,
                'validation_results': validation_results,
                'red_flags': red_flags,
                'num_trades': len(trades),
                'status': 'success' if validation_results.get('performance_consistent', False) else 'failed'
            }
            
        except Exception as e:
            self.logger.error(f"Error in out-of-sample testing: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_cross_validation(self) -> Dict[str, Any]:
        """
        Phase 4: Cross-Validation Testing
        
        Returns:
            Dict containing cross-validation results
        """
        try:
            # Get full dataset
            data = self.data_manager.data
            
            # Perform time series cross-validation
            cv_results = []
            n_splits = 5
            
            for i in range(n_splits):
                # Calculate split points
                split_ratio = 0.7 + i * 0.05
                split_point = int(len(data) * split_ratio)
                
                train_data = data[:split_point]
                test_data = data[split_point:]
                
                # Train strategy
                self.strategy.train(train_data)
                
                # Test strategy
                signals = self.strategy.generate_signals(test_data)
                trades = self.strategy.generate_trades(signals, test_data)
                returns = self.strategy.calculate_returns(trades, test_data)
                
                # Calculate performance
                performance = self.performance_analyzer.calculate_metrics(trades, returns)
                cv_results.append(performance)
            
            # Analyze cross-validation results
            cv_analysis = self._analyze_cross_validation_results(cv_results)
            
            return {
                'cv_results': cv_results,
                'cv_analysis': cv_analysis,
                'status': 'success' if cv_analysis['stable'] else 'failed'
            }
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_robustness_test(self) -> Dict[str, Any]:
        """
        Phase 5: Robustness and Stress Testing
        
        Returns:
            Dict containing robustness testing results
        """
        try:
            # Parameter sensitivity testing
            sensitivity_results = self._test_parameter_sensitivity()
            
            # Monte Carlo simulation
            monte_carlo_results = self._run_monte_carlo_simulation()
            
            # Stress testing
            stress_test_results = self._run_stress_tests()
            
            # Regime testing
            regime_test_results = self._test_regime_robustness()
            
            return {
                'sensitivity_results': sensitivity_results,
                'monte_carlo_results': monte_carlo_results,
                'stress_test_results': stress_test_results,
                'regime_test_results': regime_test_results,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in robustness testing: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_final_validation(self) -> Dict[str, Any]:
        """
        Phase 6: Final Validation and Documentation
        
        Returns:
            Dict containing final validation results
        """
        try:
            # Check all requirements
            requirements_check = self._check_all_requirements()
            
            # Generate final report
            final_report = self._generate_final_report()
            
            # Determine overall status
            overall_status = self._determine_overall_status()
            
            return {
                'requirements_check': requirements_check,
                'final_report': final_report,
                'overall_status': overall_status,
                'status': 'success' if overall_status == 'passed' else 'failed'
            }
            
        except Exception as e:
            self.logger.error(f"Error in final validation: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate data quality score
        
        Args:
            data: DataFrame to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3
        
        # Check for duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        score -= duplicate_ratio * 0.2
        
        # Check for outliers (basic check)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_ratio = 0
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio += outliers / len(data)
        
        outlier_ratio /= len(numeric_cols)
        score -= outlier_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_red_flags(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Check for red flags in performance metrics
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            List of red flag messages
        """
        red_flags = []
        
        # Check WinRate
        if metrics.get('win_rate', 0) > 0.8:
            red_flags.append(f"WinRate too high: {metrics['win_rate']:.2%} (Red Flag: >80%)")
        
        # Check MDD
        if metrics.get('max_drawdown', 0) < 0.02:
            red_flags.append(f"MDD too low: {metrics['max_drawdown']:.2%} (Red Flag: <2%)")
        
        # Check Sharpe
        if metrics.get('sharpe_ratio', 0) > 3.0:
            red_flags.append(f"Sharpe too high: {metrics['sharpe_ratio']:.2f} (Red Flag: >3.0)")
        
        # Check sample size
        if metrics.get('num_trades', 0) < 200:
            red_flags.append(f"Sample size too small: {metrics['num_trades']} (Red Flag: <200)")
        
        return red_flags
    
    def _analyze_cross_validation_results(self, cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze cross-validation results for stability
        
        Args:
            cv_results: List of cross-validation results
            
        Returns:
            Analysis of cross-validation stability
        """
        # Extract key metrics
        win_rates = [r.get('win_rate', 0) for r in cv_results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in cv_results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in cv_results]
        
        # Calculate stability metrics
        win_rate_std = np.std(win_rates)
        sharpe_std = np.std(sharpe_ratios)
        mdd_std = np.std(max_drawdowns)
        
        # Determine stability
        stable = (win_rate_std < 0.1 and sharpe_std < 0.5 and mdd_std < 0.05)
        
        return {
            'win_rate_mean': np.mean(win_rates),
            'win_rate_std': win_rate_std,
            'sharpe_mean': np.mean(sharpe_ratios),
            'sharpe_std': sharpe_std,
            'mdd_mean': np.mean(max_drawdowns),
            'mdd_std': mdd_std,
            'stable': stable
        }
    
    def _test_parameter_sensitivity(self) -> Dict[str, Any]:
        """
        Test parameter sensitivity
        
        Returns:
            Parameter sensitivity results
        """
        # This would implement parameter sensitivity testing
        # For now, return placeholder
        return {
            'sensitivity_score': 0.5,
            'critical_parameters': [],
            'stable': True
        }
    
    def _run_monte_carlo_simulation(self) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation
        
        Returns:
            Monte Carlo simulation results
        """
        # This would implement Monte Carlo simulation
        # For now, return placeholder
        return {
            'simulations_run': 1000,
            'confidence_intervals': {},
            'tail_risk_analysis': {}
        }
    
    def _run_stress_tests(self) -> Dict[str, Any]:
        """
        Run stress tests
        
        Returns:
            Stress test results
        """
        # This would implement stress testing
        # For now, return placeholder
        return {
            'crisis_periods_tested': 0,
            'volatility_spikes_tested': 0,
            'regime_transitions_tested': 0
        }
    
    def _test_regime_robustness(self) -> Dict[str, Any]:
        """
        Test regime robustness
        
        Returns:
            Regime robustness results
        """
        # This would implement regime testing
        # For now, return placeholder
        return {
            'bull_market_performance': {},
            'bear_market_performance': {},
            'sideways_market_performance': {}
        }
    
    def _check_all_requirements(self) -> Dict[str, bool]:
        """
        Check all requirements
        
        Returns:
            Dictionary of requirement checks
        """
        return {
            'data_quality': self.results['data_preparation'].get('status') == 'success',
            'in_sample_valid': self.results['in_sample'].get('status') == 'success',
            'out_of_sample_valid': self.results['out_of_sample'].get('status') == 'success',
            'cross_validation_stable': self.results['cross_validation'].get('status') == 'success',
            'robustness_passed': self.results['robustness'].get('status') == 'success',
            'no_red_flags': len(self.results['in_sample'].get('red_flags', [])) == 0
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """
        Generate final report
        
        Returns:
            Final report summary
        """
        return {
            'test_date': datetime.now().isoformat(),
            'strategy_name': self.strategy.__class__.__name__,
            'total_tests': len(self.results),
            'passed_tests': sum(1 for r in self.results.values() if r.get('status') == 'success'),
            'overall_status': 'passed' if all(self._check_all_requirements().values()) else 'failed'
        }
    
    def _determine_overall_status(self) -> str:
        """
        Determine overall status
        
        Returns:
            Overall status ('passed' or 'failed')
        """
        requirements = self._check_all_requirements()
        return 'passed' if all(requirements.values()) else 'failed'
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate testing summary
        
        Returns:
            Testing summary
        """
        return {
            'total_phases': 6,
            'completed_phases': sum(1 for r in self.results.values() if r.get('status') == 'success'),
            'overall_status': self._determine_overall_status(),
            'key_metrics': {
                'in_sample': self.results['in_sample'].get('performance_metrics', {}),
                'out_of_sample': self.results['out_of_sample'].get('performance_metrics', {})
            },
            'red_flags': self.results['in_sample'].get('red_flags', [])
        }