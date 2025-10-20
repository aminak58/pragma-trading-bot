#!/usr/bin/env python3
"""
Final Validation and Fine-Tuning System
=======================================

This module implements the final validation and fine-tuning system including:
- Comprehensive performance validation
- Parameter optimization
- Risk management validation
- System stability testing
- Live trading readiness assessment

Author: Pragma Trading Bot Team
Date: 2025-10-20
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque
import threading

@dataclass
class ValidationCriteria:
    """Validation criteria data class"""
    metric: str
    threshold: float
    operator: str  # >, <, >=, <=, ==
    weight: float  # Importance weight
    description: str

@dataclass
class ValidationResult:
    """Validation result data class"""
    criteria: ValidationCriteria
    actual_value: float
    passed: bool
    score: float
    message: str

@dataclass
class OptimizationResult:
    """Optimization result data class"""
    parameter: str
    original_value: Any
    optimized_value: Any
    improvement: float
    confidence: float

@dataclass
class SystemHealthCheck:
    """System health check data class"""
    component: str
    status: str  # healthy, warning, critical
    metrics: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]

class PerformanceValidator:
    """Performance validation system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_criteria = self._create_default_criteria()
        self.validation_results = []
    
    def _create_default_criteria(self) -> List[ValidationCriteria]:
        """Create default validation criteria"""
        criteria = [
            ValidationCriteria(
                metric="win_rate",
                threshold=0.55,
                operator=">=",
                weight=0.3,
                description="Win rate should be at least 55%"
            ),
            ValidationCriteria(
                metric="sharpe_ratio",
                threshold=1.0,
                operator=">=",
                weight=0.25,
                description="Sharpe ratio should be at least 1.0"
            ),
            ValidationCriteria(
                metric="max_drawdown",
                threshold=0.15,
                operator="<=",
                weight=0.2,
                description="Max drawdown should be at most 15%"
            ),
            ValidationCriteria(
                metric="total_return",
                threshold=0.0,
                operator=">",
                weight=0.15,
                description="Total return should be positive"
            ),
            ValidationCriteria(
                metric="volatility",
                threshold=0.1,
                operator="<=",
                weight=0.1,
                description="Volatility should be at most 10%"
            )
        ]
        return criteria
    
    def validate_performance(self, performance_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate performance against criteria"""
        results = []
        
        try:
            for criteria in self.validation_criteria:
                actual_value = performance_data.get(criteria.metric, 0.0)
                
                # Evaluate condition
                passed = self._evaluate_condition(actual_value, criteria.operator, criteria.threshold)
                
                # Calculate score
                score = self._calculate_score(actual_value, criteria)
                
                # Create message
                message = self._create_message(criteria, actual_value, passed)
                
                result = ValidationResult(
                    criteria=criteria,
                    actual_value=actual_value,
                    passed=passed,
                    score=score,
                    message=message
                )
                
                results.append(result)
                self.logger.info(f"Validation: {criteria.metric} = {actual_value:.3f} ({'PASS' if passed else 'FAIL'})")
            
            self.validation_results.extend(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating performance: {e}")
            return []
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate condition"""
        try:
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _calculate_score(self, value: float, criteria: ValidationCriteria) -> float:
        """Calculate validation score"""
        try:
            if criteria.operator == '>=':
                if value >= criteria.threshold:
                    return criteria.weight
                else:
                    return criteria.weight * (value / criteria.threshold)
            elif criteria.operator == '<=':
                if value <= criteria.threshold:
                    return criteria.weight
                else:
                    return criteria.weight * (criteria.threshold / value)
            elif criteria.operator == '>':
                if value > criteria.threshold:
                    return criteria.weight
                else:
                    return criteria.weight * (value / criteria.threshold)
            elif criteria.operator == '<':
                if value < criteria.threshold:
                    return criteria.weight
                else:
                    return criteria.weight * (criteria.threshold / value)
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating score: {e}")
            return 0.0
    
    def _create_message(self, criteria: ValidationCriteria, value: float, passed: bool) -> str:
        """Create validation message"""
        status = "PASSED" if passed else "FAILED"
        return f"{criteria.description}: {value:.3f} {criteria.operator} {criteria.threshold} ({status})"
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        try:
            if not self.validation_results:
                return {"error": "No validation results available"}
            
            total_score = sum(result.score for result in self.validation_results)
            max_score = sum(criteria.weight for criteria in self.validation_criteria)
            overall_score = (total_score / max_score) * 100 if max_score > 0 else 0
            
            passed_count = sum(1 for result in self.validation_results if result.passed)
            total_count = len(self.validation_results)
            pass_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
            
            return {
                "overall_score": overall_score,
                "pass_rate": pass_rate,
                "passed_criteria": passed_count,
                "total_criteria": total_count,
                "total_score": total_score,
                "max_score": max_score,
                "results": [asdict(result) for result in self.validation_results]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {e}")
            return {"error": str(e)}

class ParameterOptimizer:
    """Parameter optimization system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_results = []
    
    def optimize_parameters(self, current_params: Dict[str, Any], 
                          performance_data: Dict[str, Any]) -> List[OptimizationResult]:
        """Optimize parameters based on performance"""
        results = []
        
        try:
            # Optimize Kelly safety factor
            kelly_result = self._optimize_kelly_safety_factor(current_params, performance_data)
            if kelly_result:
                results.append(kelly_result)
            
            # Optimize stop loss
            stoploss_result = self._optimize_stop_loss(current_params, performance_data)
            if stoploss_result:
                results.append(stoploss_result)
            
            # Optimize position sizing
            position_result = self._optimize_position_sizing(current_params, performance_data)
            if position_result:
                results.append(position_result)
            
            # Optimize risk thresholds
            risk_result = self._optimize_risk_thresholds(current_params, performance_data)
            if risk_result:
                results.append(risk_result)
            
            self.optimization_results.extend(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            return []
    
    def _optimize_kelly_safety_factor(self, params: Dict[str, Any], 
                                    performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize Kelly safety factor"""
        try:
            current_factor = params.get('kelly_safety_factor', 0.25)
            win_rate = performance.get('win_rate', 0.6)
            max_drawdown = abs(performance.get('max_drawdown', 0.1))
            
            # Adjust based on performance
            if win_rate > 0.7 and max_drawdown < 0.05:
                # High win rate, low drawdown - can increase factor
                optimized_factor = min(current_factor * 1.2, 0.5)
                improvement = (optimized_factor - current_factor) / current_factor
            elif win_rate < 0.5 or max_drawdown > 0.15:
                # Low win rate or high drawdown - decrease factor
                optimized_factor = max(current_factor * 0.8, 0.1)
                improvement = (optimized_factor - current_factor) / current_factor
            else:
                return None
            
            return OptimizationResult(
                parameter="kelly_safety_factor",
                original_value=current_factor,
                optimized_value=optimized_factor,
                improvement=improvement,
                confidence=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing Kelly factor: {e}")
            return None
    
    def _optimize_stop_loss(self, params: Dict[str, Any], 
                          performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize stop loss"""
        try:
            current_stoploss = params.get('base_stop_loss', 0.02)
            max_drawdown = abs(performance.get('max_drawdown', 0.1))
            volatility = performance.get('volatility', 0.05)
            
            # Adjust based on drawdown and volatility
            if max_drawdown > 0.15:
                # High drawdown - tighten stop loss
                optimized_stoploss = max(current_stoploss * 0.8, 0.01)
                improvement = (optimized_stoploss - current_stoploss) / current_stoploss
            elif max_drawdown < 0.05 and volatility > 0.08:
                # Low drawdown, high volatility - widen stop loss
                optimized_stoploss = min(current_stoploss * 1.3, 0.05)
                improvement = (optimized_stoploss - current_stoploss) / current_stoploss
            else:
                return None
            
            return OptimizationResult(
                parameter="base_stop_loss",
                original_value=current_stoploss,
                optimized_value=optimized_stoploss,
                improvement=improvement,
                confidence=0.7
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing stop loss: {e}")
            return None
    
    def _optimize_position_sizing(self, params: Dict[str, Any], 
                                performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize position sizing"""
        try:
            current_max_size = params.get('max_position_size', 0.2)
            total_return = performance.get('total_return', 0.0)
            max_drawdown = abs(performance.get('max_drawdown', 0.1))
            
            # Adjust based on return and drawdown
            if total_return > 0.1 and max_drawdown < 0.08:
                # Good returns, low drawdown - can increase position size
                optimized_size = min(current_max_size * 1.2, 0.3)
                improvement = (optimized_size - current_max_size) / current_max_size
            elif total_return < 0 or max_drawdown > 0.15:
                # Poor returns or high drawdown - decrease position size
                optimized_size = max(current_max_size * 0.8, 0.1)
                improvement = (optimized_size - current_max_size) / current_max_size
            else:
                return None
            
            return OptimizationResult(
                parameter="max_position_size",
                original_value=current_max_size,
                optimized_value=optimized_size,
                improvement=improvement,
                confidence=0.6
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing position sizing: {e}")
            return None
    
    def _optimize_risk_thresholds(self, params: Dict[str, Any], 
                                performance: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Optimize risk thresholds"""
        try:
            current_threshold = params.get('volatility_threshold', 0.05)
            volatility = performance.get('volatility', 0.05)
            max_drawdown = abs(performance.get('max_drawdown', 0.1))
            
            # Adjust based on actual volatility and drawdown
            if volatility > current_threshold * 1.5 and max_drawdown > 0.1:
                # High volatility causing high drawdown - lower threshold
                optimized_threshold = max(current_threshold * 0.8, 0.03)
                improvement = (optimized_threshold - current_threshold) / current_threshold
            elif volatility < current_threshold * 0.5 and max_drawdown < 0.05:
                # Low volatility, low drawdown - can increase threshold
                optimized_threshold = min(current_threshold * 1.2, 0.08)
                improvement = (optimized_threshold - current_threshold) / current_threshold
            else:
                return None
            
            return OptimizationResult(
                parameter="volatility_threshold",
                original_value=current_threshold,
                optimized_value=optimized_threshold,
                improvement=improvement,
                confidence=0.5
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing risk thresholds: {e}")
            return None

class SystemHealthChecker:
    """System health checking system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checks = []
    
    def check_system_health(self, system_data: Dict[str, Any]) -> List[SystemHealthCheck]:
        """Check system health"""
        checks = []
        
        try:
            # Check monitoring system
            monitoring_check = self._check_monitoring_system(system_data)
            checks.append(monitoring_check)
            
            # Check alerting system
            alerting_check = self._check_alerting_system(system_data)
            checks.append(alerting_check)
            
            # Check risk management
            risk_check = self._check_risk_management(system_data)
            checks.append(risk_check)
            
            # Check data pipeline
            data_check = self._check_data_pipeline(system_data)
            checks.append(data_check)
            
            # Check trading engine
            trading_check = self._check_trading_engine(system_data)
            checks.append(trading_check)
            
            self.health_checks.extend(checks)
            return checks
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return []
    
    def _check_monitoring_system(self, data: Dict[str, Any]) -> SystemHealthCheck:
        """Check monitoring system health"""
        try:
            metrics_collected = data.get('metrics_collected', 0)
            monitoring_active = data.get('monitoring_active', False)
            
            issues = []
            recommendations = []
            status = "healthy"
            
            if not monitoring_active:
                issues.append("Monitoring system not active")
                status = "critical"
                recommendations.append("Start monitoring system")
            
            if metrics_collected < 10:
                issues.append("Low metrics collection count")
                status = "warning" if status == "healthy" else status
                recommendations.append("Check metrics collection frequency")
            
            return SystemHealthCheck(
                component="monitoring_system",
                status=status,
                metrics={"metrics_collected": metrics_collected, "active": monitoring_active},
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking monitoring system: {e}")
            return SystemHealthCheck(
                component="monitoring_system",
                status="critical",
                metrics={},
                issues=["Error checking monitoring system"],
                recommendations=["Fix monitoring system check"]
            )
    
    def _check_alerting_system(self, data: Dict[str, Any]) -> SystemHealthCheck:
        """Check alerting system health"""
        try:
            alerts_generated = data.get('alerts_generated', 0)
            alerting_active = data.get('alerting_active', False)
            
            issues = []
            recommendations = []
            status = "healthy"
            
            if not alerting_active:
                issues.append("Alerting system not active")
                status = "critical"
                recommendations.append("Start alerting system")
            
            if alerts_generated > 50:
                issues.append("High alert frequency")
                status = "warning" if status == "healthy" else status
                recommendations.append("Review alert thresholds")
            
            return SystemHealthCheck(
                component="alerting_system",
                status=status,
                metrics={"alerts_generated": alerts_generated, "active": alerting_active},
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking alerting system: {e}")
            return SystemHealthCheck(
                component="alerting_system",
                status="critical",
                metrics={},
                issues=["Error checking alerting system"],
                recommendations=["Fix alerting system check"]
            )
    
    def _check_risk_management(self, data: Dict[str, Any]) -> SystemHealthCheck:
        """Check risk management health"""
        try:
            risk_level = data.get('risk_level', 'unknown')
            circuit_breakers = data.get('circuit_breakers', {})
            
            issues = []
            recommendations = []
            status = "healthy"
            
            if risk_level == "critical":
                issues.append("Critical risk level detected")
                status = "critical"
                recommendations.append("Immediate risk management action required")
            elif risk_level == "high":
                issues.append("High risk level detected")
                status = "warning"
                recommendations.append("Review risk management parameters")
            
            failed_breakers = [k for k, v in circuit_breakers.items() if not v]
            if failed_breakers:
                issues.append(f"Circuit breakers triggered: {failed_breakers}")
                status = "critical"
                recommendations.append("Investigate circuit breaker triggers")
            
            return SystemHealthCheck(
                component="risk_management",
                status=status,
                metrics={"risk_level": risk_level, "circuit_breakers": circuit_breakers},
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking risk management: {e}")
            return SystemHealthCheck(
                component="risk_management",
                status="critical",
                metrics={},
                issues=["Error checking risk management"],
                recommendations=["Fix risk management check"]
            )
    
    def _check_data_pipeline(self, data: Dict[str, Any]) -> SystemHealthCheck:
        """Check data pipeline health"""
        try:
            data_quality = data.get('data_quality', 0.0)
            data_sources = data.get('data_sources', {})
            
            issues = []
            recommendations = []
            status = "healthy"
            
            if data_quality < 0.8:
                issues.append("Low data quality")
                status = "warning"
                recommendations.append("Check data sources and processing")
            
            invalid_sources = [k for k, v in data_sources.items() if not v]
            if invalid_sources:
                issues.append(f"Invalid data sources: {invalid_sources}")
                status = "critical"
                recommendations.append("Fix data source connections")
            
            return SystemHealthCheck(
                component="data_pipeline",
                status=status,
                metrics={"data_quality": data_quality, "data_sources": data_sources},
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking data pipeline: {e}")
            return SystemHealthCheck(
                component="data_pipeline",
                status="critical",
                metrics={},
                issues=["Error checking data pipeline"],
                recommendations=["Fix data pipeline check"]
            )
    
    def _check_trading_engine(self, data: Dict[str, Any]) -> SystemHealthCheck:
        """Check trading engine health"""
        try:
            trades_executed = data.get('trades_executed', 0)
            engine_active = data.get('trading_engine_active', False)
            
            issues = []
            recommendations = []
            status = "healthy"
            
            if not engine_active:
                issues.append("Trading engine not active")
                status = "critical"
                recommendations.append("Start trading engine")
            
            if trades_executed == 0:
                issues.append("No trades executed")
                status = "warning" if status == "healthy" else status
                recommendations.append("Check trading signals and conditions")
            
            return SystemHealthCheck(
                component="trading_engine",
                status=status,
                metrics={"trades_executed": trades_executed, "active": engine_active},
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error checking trading engine: {e}")
            return SystemHealthCheck(
                component="trading_engine",
                status="critical",
                metrics={},
                issues=["Error checking trading engine"],
                recommendations=["Fix trading engine check"]
            )

class FinalValidationSystem:
    """Main final validation system"""
    
    def __init__(self):
        self.performance_validator = PerformanceValidator()
        self.parameter_optimizer = ParameterOptimizer()
        self.system_health_checker = SystemHealthChecker()
        self.logger = logging.getLogger(__name__)
    
    def run_final_validation(self, performance_data: Dict[str, Any], 
                            system_data: Dict[str, Any],
                            current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive final validation"""
        try:
            self.logger.info("Starting final validation...")
            
            # Performance validation
            self.logger.info("1. Running performance validation...")
            performance_results = self.performance_validator.validate_performance(performance_data)
            performance_summary = self.performance_validator.get_validation_summary()
            
            # Parameter optimization
            self.logger.info("2. Running parameter optimization...")
            optimization_results = self.parameter_optimizer.optimize_parameters(current_params, performance_data)
            
            # System health check
            self.logger.info("3. Running system health check...")
            health_checks = self.system_health_checker.check_system_health(system_data)
            
            # Generate final report
            self.logger.info("4. Generating final validation report...")
            final_report = self._generate_final_report(
                performance_summary, optimization_results, health_checks
            )
            
            self.logger.info("Final validation completed successfully!")
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error in final validation: {e}")
            return {"error": str(e)}
    
    def _generate_final_report(self, performance_summary: Dict[str, Any],
                             optimization_results: List[OptimizationResult],
                             health_checks: List[SystemHealthCheck]) -> Dict[str, Any]:
        """Generate final validation report"""
        try:
            # Calculate overall readiness score
            performance_score = performance_summary.get('overall_score', 0)
            
            # Calculate optimization score
            optimization_score = 0
            if optimization_results:
                avg_improvement = np.mean([r.improvement for r in optimization_results])
                optimization_score = min(avg_improvement * 100, 100)
            
            # Calculate health score
            health_score = 0
            if health_checks:
                healthy_components = sum(1 for check in health_checks if check.status == "healthy")
                health_score = (healthy_components / len(health_checks)) * 100
            
            # Overall readiness score
            overall_score = (performance_score * 0.5 + optimization_score * 0.2 + health_score * 0.3)
            
            # Determine readiness level
            if overall_score >= 80:
                readiness_level = "READY"
                readiness_message = "System is ready for live trading"
            elif overall_score >= 60:
                readiness_level = "CONDITIONAL"
                readiness_message = "System is conditionally ready with recommendations"
            else:
                readiness_level = "NOT_READY"
                readiness_message = "System is not ready for live trading"
            
            report = {
                "validation_timestamp": datetime.now().isoformat(),
                "overall_readiness": {
                    "score": overall_score,
                    "level": readiness_level,
                    "message": readiness_message
                },
                "performance_validation": performance_summary,
                "parameter_optimization": {
                    "results": [asdict(result) for result in optimization_results],
                    "optimization_score": optimization_score,
                    "recommendations": self._generate_optimization_recommendations(optimization_results)
                },
                "system_health": {
                    "checks": [asdict(check) for check in health_checks],
                    "health_score": health_score,
                    "critical_issues": self._get_critical_issues(health_checks),
                    "recommendations": self._generate_health_recommendations(health_checks)
                },
                "final_recommendations": self._generate_final_recommendations(
                    performance_summary, optimization_results, health_checks
                )
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            return {"error": str(e)}
    
    def _generate_optimization_recommendations(self, results: List[OptimizationResult]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for result in results:
            if result.improvement > 0.1:  # Significant improvement
                recommendations.append(f"Consider increasing {result.parameter} to {result.optimized_value}")
            elif result.improvement < -0.1:  # Significant decrease
                recommendations.append(f"Consider decreasing {result.parameter} to {result.optimized_value}")
        
        return recommendations
    
    def _get_critical_issues(self, health_checks: List[SystemHealthCheck]) -> List[str]:
        """Get critical issues from health checks"""
        critical_issues = []
        
        for check in health_checks:
            if check.status == "critical":
                critical_issues.extend(check.issues)
        
        return critical_issues
    
    def _generate_health_recommendations(self, health_checks: List[SystemHealthCheck]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        for check in health_checks:
            recommendations.extend(check.recommendations)
        
        return recommendations
    
    def _generate_final_recommendations(self, performance_summary: Dict[str, Any],
                                      optimization_results: List[OptimizationResult],
                                      health_checks: List[SystemHealthCheck]) -> List[str]:
        """Generate final recommendations"""
        recommendations = []
        
        # Performance recommendations
        if performance_summary.get('overall_score', 0) < 70:
            recommendations.append("Improve strategy performance before live trading")
        
        # Optimization recommendations
        if optimization_results:
            recommendations.append("Apply parameter optimizations for better performance")
        
        # Health recommendations
        critical_issues = self._get_critical_issues(health_checks)
        if critical_issues:
            recommendations.append("Resolve critical system issues before live trading")
        
        # General recommendations
        recommendations.append("Start with small position sizes in live trading")
        recommendations.append("Monitor system closely during initial live trading period")
        recommendations.append("Have emergency stop procedures ready")
        
        return recommendations

def main():
    """Test final validation system"""
    logging.basicConfig(level=logging.INFO)
    
    print("Final Validation and Fine-Tuning System Test")
    print("=" * 60)
    
    # Create validation system
    validator = FinalValidationSystem()
    
    # Sample performance data
    performance_data = {
        "win_rate": 0.63,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.12,
        "total_return": 0.08,
        "volatility": 0.06
    }
    
    # Sample system data
    system_data = {
        "metrics_collected": 150,
        "monitoring_active": True,
        "alerts_generated": 5,
        "alerting_active": True,
        "risk_level": "medium",
        "circuit_breakers": {"daily_loss": True, "max_drawdown": True},
        "data_quality": 0.95,
        "data_sources": {"binance": True, "coinbase": True},
        "trades_executed": 25,
        "trading_engine_active": True
    }
    
    # Sample current parameters
    current_params = {
        "kelly_safety_factor": 0.25,
        "base_stop_loss": 0.02,
        "max_position_size": 0.2,
        "volatility_threshold": 0.05
    }
    
    print(f"\n1. Sample Performance Data:")
    for key, value in performance_data.items():
        print(f"   {key}: {value}")
    
    print(f"\n2. Sample System Data:")
    for key, value in system_data.items():
        print(f"   {key}: {value}")
    
    print(f"\n3. Current Parameters:")
    for key, value in current_params.items():
        print(f"   {key}: {value}")
    
    # Run final validation
    print(f"\n4. Running final validation...")
    final_report = validator.run_final_validation(performance_data, system_data, current_params)
    
    if "error" in final_report:
        print(f"   Error: {final_report['error']}")
    else:
        print(f"   Final validation completed successfully!")
        
        # Overall readiness
        readiness = final_report['overall_readiness']
        print(f"\n5. Overall Readiness:")
        print(f"   Score: {readiness['score']:.1f}")
        print(f"   Level: {readiness['level']}")
        print(f"   Message: {readiness['message']}")
        
        # Performance validation
        performance = final_report['performance_validation']
        print(f"\n6. Performance Validation:")
        print(f"   Overall Score: {performance['overall_score']:.1f}")
        print(f"   Pass Rate: {performance['pass_rate']:.1f}%")
        print(f"   Passed Criteria: {performance['passed_criteria']}/{performance['total_criteria']}")
        
        # Parameter optimization
        optimization = final_report['parameter_optimization']
        print(f"\n7. Parameter Optimization:")
        print(f"   Optimization Score: {optimization['optimization_score']:.1f}")
        print(f"   Results: {len(optimization['results'])}")
        for result in optimization['results']:
            print(f"     - {result['parameter']}: {result['original_value']} -> {result['optimized_value']} ({result['improvement']:.1%})")
        
        # System health
        health = final_report['system_health']
        print(f"\n8. System Health:")
        print(f"   Health Score: {health['health_score']:.1f}")
        print(f"   Critical Issues: {len(health['critical_issues'])}")
        for issue in health['critical_issues']:
            print(f"     - {issue}")
        
        # Final recommendations
        recommendations = final_report['final_recommendations']
        print(f"\n9. Final Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    print("\nFinal Validation and Fine-Tuning System test completed!")
    print("\nFinal validation features:")
    print("  - Comprehensive performance validation")
    print("  - Parameter optimization")
    print("  - System health checking")
    print("  - Live trading readiness assessment")
    print("  - Detailed recommendations")

if __name__ == "__main__":
    main()
