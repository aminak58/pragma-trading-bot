#!/usr/bin/env python3
"""
Enhanced Risk Management System
===============================

This module implements an enhanced risk management system with:
- Kelly Criterion based position sizing
- Dynamic stop-loss management
- Circuit breakers and safety mechanisms
- Real-time risk monitoring

Author: Pragma Trading Bot Team
Date: 2025-10-20
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk metrics data class"""
    portfolio_value: float
    total_exposure: float
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    var_99: float
    correlation_risk: float
    concentration_risk: float
    leverage_risk: float
    liquidity_risk: float

@dataclass
class PositionInfo:
    """Position information data class"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float
    margin_used: float
    risk_score: float

class KellyCriterionCalculator:
    """Kelly Criterion calculator for optimal position sizing"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, 
                               avg_loss: float, n_trades: int = 100) -> float:
        """
        Calculate Kelly Criterion fraction
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            n_trades: Number of trades for confidence adjustment
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Basic Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply confidence adjustment based on sample size
        confidence_factor = min(1.0, np.sqrt(n_trades / 100))
        kelly_fraction *= confidence_factor
        
        # Apply safety factor (use fractional Kelly)
        safety_factor = 0.25  # Use 25% of Kelly
        kelly_fraction *= safety_factor
        
        # Ensure bounds
        kelly_fraction = max(0.0, min(kelly_fraction, 0.2))  # Max 20% per trade
        
        return kelly_fraction
    
    def calculate_position_size(self, account_balance: float, win_rate: float,
                              avg_win: float, avg_loss: float, 
                              current_price: float, n_trades: int = 100) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            account_balance: Current account balance
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
            current_price: Current asset price
            n_trades: Number of historical trades
            
        Returns:
            Optimal position size in units
        """
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss, n_trades)
        
        if kelly_fraction <= 0:
            return 0.0
        
        # Calculate position value
        position_value = account_balance * kelly_fraction
        
        # Convert to units
        position_size = position_value / current_price
        
        return position_size

class DynamicStopLossManager:
    """Dynamic stop-loss manager based on volatility and market conditions"""
    
    def __init__(self, base_stop_loss: float = 0.02, max_stop_loss: float = 0.10):
        self.base_stop_loss = base_stop_loss
        self.max_stop_loss = max_stop_loss
    
    def calculate_dynamic_stop_loss(self, entry_price: float, current_price: float,
                                  volatility: float, atr: float, 
                                  market_regime: str = "normal") -> float:
        """
        Calculate dynamic stop-loss based on market conditions
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            volatility: Current market volatility
            atr: Average True Range
            market_regime: Current market regime
            
        Returns:
            Dynamic stop-loss percentage
        """
        # Base stop-loss
        stop_loss = self.base_stop_loss
        
        # Adjust for volatility
        if volatility > 0.05:  # High volatility
            stop_loss *= 1.5
        elif volatility < 0.02:  # Low volatility
            stop_loss *= 0.8
        
        # Adjust for ATR
        atr_percent = atr / current_price
        if atr_percent > 0.03:  # High ATR
            stop_loss *= 1.3
        elif atr_percent < 0.01:  # Low ATR
            stop_loss *= 0.9
        
        # Adjust for market regime
        regime_multipliers = {
            "bull": 1.2,      # Wider stops in bull market
            "bear": 0.8,      # Tighter stops in bear market
            "sideways": 1.0,  # Normal stops in sideways market
            "high_vol": 1.5,  # Much wider stops in high volatility
            "normal": 1.0
        }
        
        stop_loss *= regime_multipliers.get(market_regime, 1.0)
        
        # Ensure bounds
        stop_loss = max(self.base_stop_loss * 0.5, min(stop_loss, self.max_stop_loss))
        
        return stop_loss
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                               highest_price: float, volatility: float) -> float:
        """
        Calculate trailing stop-loss
        
        Args:
            entry_price: Entry price
            current_price: Current price
            highest_price: Highest price since entry
            volatility: Current volatility
            
        Returns:
            Trailing stop-loss percentage
        """
        # Calculate profit percentage
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct <= 0:
            return self.base_stop_loss
        
        # Dynamic trailing stop based on profit and volatility
        trailing_stop = self.base_stop_loss
        
        # Tighten stop as profit increases
        if profit_pct > 0.05:  # 5% profit
            trailing_stop *= 0.8
        if profit_pct > 0.10:  # 10% profit
            trailing_stop *= 0.7
        if profit_pct > 0.20:  # 20% profit
            trailing_stop *= 0.6
        
        # Adjust for volatility
        trailing_stop *= (1 + volatility * 2)
        
        return max(trailing_stop, self.base_stop_loss * 0.5)

class CircuitBreakerManager:
    """Circuit breaker manager for risk control"""
    
    def __init__(self, max_daily_loss: float = 0.05, max_drawdown: float = 0.15,
                 max_positions: int = 5, max_correlation: float = 0.7):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions
        self.max_correlation = max_correlation
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.active_positions = []
        self.correlation_matrix = {}
    
    def check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Check if daily loss limit is exceeded"""
        # Reset daily PnL if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
        
        self.daily_pnl += current_pnl
        
        return self.daily_pnl < -self.max_daily_loss
    
    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """Check if drawdown limit is exceeded"""
        return current_drawdown < -self.max_drawdown
    
    def check_position_limit(self, current_positions: int) -> bool:
        """Check if position limit is exceeded"""
        return current_positions < self.max_positions
    
    def check_correlation_limit(self, new_position: str, 
                              existing_positions: List[str]) -> bool:
        """Check if correlation limit is exceeded"""
        if not existing_positions:
            return True
        
        # Calculate correlation with existing positions
        max_corr = 0.0
        for existing_pos in existing_positions:
            if (new_position, existing_pos) in self.correlation_matrix:
                corr = abs(self.correlation_matrix[(new_position, existing_pos)])
                max_corr = max(max_corr, corr)
        
        return max_corr < self.max_correlation
    
    def should_halt_trading(self, risk_metrics: RiskMetrics) -> Tuple[bool, str]:
        """
        Determine if trading should be halted
        
        Args:
            risk_metrics: Current risk metrics
            
        Returns:
            Tuple of (should_halt, reason)
        """
        # Check daily loss limit
        if not self.check_daily_loss_limit(risk_metrics.current_drawdown):
            return True, "Daily loss limit exceeded"
        
        # Check drawdown limit
        if not self.check_drawdown_limit(risk_metrics.max_drawdown):
            return True, "Maximum drawdown exceeded"
        
        # Check position limit
        if not self.check_position_limit(len(self.active_positions)):
            return True, "Maximum positions exceeded"
        
        # Check volatility limit
        if risk_metrics.volatility > 0.1:  # 10% volatility
            return True, "Excessive volatility detected"
        
        # Check VaR limit
        if risk_metrics.var_95 < -0.1:  # 10% VaR
            return True, "VaR limit exceeded"
        
        return False, ""

class EnhancedRiskManager:
    """Enhanced risk management system"""
    
    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = account_balance
        self.kelly_calculator = KellyCriterionCalculator()
        self.stop_loss_manager = DynamicStopLossManager()
        self.circuit_breaker = CircuitBreakerManager()
        self.risk_history = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, symbol: str, current_price: float,
                              win_rate: float, avg_win: float, avg_loss: float,
                              n_trades: int = 100) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            symbol: Trading symbol
            current_price: Current asset price
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
            n_trades: Number of historical trades
            
        Returns:
            Optimal position size
        """
        try:
            position_size = self.kelly_calculator.calculate_position_size(
                self.account_balance, win_rate, avg_win, avg_loss, 
                current_price, n_trades
            )
            
            self.logger.info(f"Calculated position size for {symbol}: {position_size:.4f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_dynamic_stop_loss(self, entry_price: float, current_price: float,
                                  volatility: float, atr: float,
                                  market_regime: str = "normal") -> float:
        """
        Calculate dynamic stop-loss
        
        Args:
            entry_price: Entry price
            current_price: Current price
            volatility: Current volatility
            atr: Average True Range
            market_regime: Market regime
            
        Returns:
            Dynamic stop-loss percentage
        """
        try:
            stop_loss = self.stop_loss_manager.calculate_dynamic_stop_loss(
                entry_price, current_price, volatility, atr, market_regime
            )
            
            self.logger.info(f"Calculated dynamic stop-loss: {stop_loss:.2%}")
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop-loss: {e}")
            return self.stop_loss_manager.base_stop_loss
    
    def check_risk_limits(self, risk_metrics: RiskMetrics) -> Tuple[bool, List[str]]:
        """
        Check all risk limits
        
        Args:
            risk_metrics: Current risk metrics
            
        Returns:
            Tuple of (within_limits, violations)
        """
        violations = []
        
        # Check circuit breakers
        should_halt, reason = self.circuit_breaker.should_halt_trading(risk_metrics)
        if should_halt:
            violations.append(reason)
        
        # Check portfolio heat
        if risk_metrics.total_exposure > 0.8:  # 80% exposure
            violations.append("Portfolio heat too high")
        
        # Check concentration risk
        if risk_metrics.concentration_risk > 0.3:  # 30% concentration
            violations.append("Position concentration too high")
        
        # Check leverage risk
        if risk_metrics.leverage_risk > 3.0:  # 3x leverage
            violations.append("Leverage too high")
        
        # Check liquidity risk
        if risk_metrics.liquidity_risk > 0.1:  # 10% liquidity risk
            violations.append("Liquidity risk too high")
        
        within_limits = len(violations) == 0
        
        if violations:
            self.logger.warning(f"Risk limit violations: {violations}")
        
        return within_limits, violations
    
    def update_risk_metrics(self, positions: List[PositionInfo], 
                           market_data: Dict[str, float]) -> RiskMetrics:
        """
        Update risk metrics based on current positions and market data
        
        Args:
            positions: List of current positions
            market_data: Current market data
            
        Returns:
            Updated risk metrics
        """
        try:
            # Calculate portfolio value
            portfolio_value = self.account_balance
            total_exposure = 0.0
            total_pnl = 0.0
            
            for position in positions:
                portfolio_value += position.unrealized_pnl
                total_exposure += abs(position.size * position.current_price)
                total_pnl += position.unrealized_pnl + position.realized_pnl
            
            # Calculate drawdown
            current_drawdown = total_pnl / self.account_balance
            max_drawdown = min(0, current_drawdown)  # Simplified
            
            # Calculate volatility (simplified)
            volatility = market_data.get('volatility', 0.02)
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = market_data.get('sharpe_ratio', 1.0)
            
            # Calculate VaR (simplified)
            var_95 = -volatility * 1.645  # 95% VaR
            var_99 = -volatility * 2.326  # 99% VaR
            
            # Calculate correlation risk
            correlation_risk = 0.0  # Simplified
            
            # Calculate concentration risk
            concentration_risk = 0.0
            if positions:
                max_position_value = max(abs(pos.size * pos.current_price) for pos in positions)
                concentration_risk = max_position_value / portfolio_value
            
            # Calculate leverage risk
            leverage_risk = 1.0  # Simplified
            
            # Calculate liquidity risk
            liquidity_risk = 0.0  # Simplified
            
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                var_99=var_99,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                leverage_risk=leverage_risk,
                liquidity_risk=liquidity_risk
            )
            
            # Store in history
            self.risk_history.append(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
            # Return default metrics
            return RiskMetrics(
                portfolio_value=self.account_balance,
                total_exposure=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                volatility=0.02,
                sharpe_ratio=1.0,
                var_95=-0.05,
                var_99=-0.10,
                correlation_risk=0.0,
                concentration_risk=0.0,
                leverage_risk=1.0,
                liquidity_risk=0.0
            )
    
    def generate_risk_report(self) -> Dict[str, any]:
        """
        Generate comprehensive risk report
        
        Returns:
            Risk report dictionary
        """
        if not self.risk_history:
            return {"error": "No risk history available"}
        
        latest_metrics = self.risk_history[-1]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "account_balance": self.account_balance,
            "portfolio_value": latest_metrics.portfolio_value,
            "total_exposure": latest_metrics.total_exposure,
            "exposure_percentage": latest_metrics.total_exposure / self.account_balance,
            "current_drawdown": latest_metrics.current_drawdown,
            "max_drawdown": latest_metrics.max_drawdown,
            "volatility": latest_metrics.volatility,
            "sharpe_ratio": latest_metrics.sharpe_ratio,
            "var_95": latest_metrics.var_95,
            "var_99": latest_metrics.var_99,
            "correlation_risk": latest_metrics.correlation_risk,
            "concentration_risk": latest_metrics.concentration_risk,
            "leverage_risk": latest_metrics.leverage_risk,
            "liquidity_risk": latest_metrics.liquidity_risk,
            "risk_level": self._determine_risk_level(latest_metrics),
            "recommendations": self._generate_recommendations(latest_metrics)
        }
        
        return report
    
    def _determine_risk_level(self, metrics: RiskMetrics) -> RiskLevel:
        """Determine current risk level"""
        risk_score = 0
        
        # Drawdown risk
        if metrics.current_drawdown < -0.15:
            risk_score += 3
        elif metrics.current_drawdown < -0.10:
            risk_score += 2
        elif metrics.current_drawdown < -0.05:
            risk_score += 1
        
        # Volatility risk
        if metrics.volatility > 0.08:
            risk_score += 2
        elif metrics.volatility > 0.05:
            risk_score += 1
        
        # Concentration risk
        if metrics.concentration_risk > 0.5:
            risk_score += 2
        elif metrics.concentration_risk > 0.3:
            risk_score += 1
        
        # Leverage risk
        if metrics.leverage_risk > 3:
            risk_score += 2
        elif metrics.leverage_risk > 2:
            risk_score += 1
        
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(self, metrics: RiskMetrics) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if metrics.current_drawdown < -0.10:
            recommendations.append("Consider reducing position sizes due to high drawdown")
        
        if metrics.volatility > 0.05:
            recommendations.append("Increase stop-loss distances due to high volatility")
        
        if metrics.concentration_risk > 0.3:
            recommendations.append("Diversify positions to reduce concentration risk")
        
        if metrics.leverage_risk > 2:
            recommendations.append("Reduce leverage to manage risk")
        
        if metrics.total_exposure > 0.8:
            recommendations.append("Reduce total exposure to manage portfolio heat")
        
        if not recommendations:
            recommendations.append("Risk levels are within acceptable ranges")
        
        return recommendations

def main():
    """Test the enhanced risk management system"""
    logging.basicConfig(level=logging.INFO)
    
    print("Enhanced Risk Management System Test")
    print("=" * 50)
    
    # Create risk manager
    risk_manager = EnhancedRiskManager(account_balance=10000.0)
    
    # Test Kelly Criterion
    print("\n1. Testing Kelly Criterion:")
    position_size = risk_manager.calculate_position_size(
        symbol="BTC/USDT",
        current_price=50000.0,
        win_rate=0.63,
        avg_win=0.03,
        avg_loss=0.02,
        n_trades=100
    )
    print(f"   Optimal position size: {position_size:.4f} BTC")
    
    # Test dynamic stop-loss
    print("\n2. Testing Dynamic Stop-Loss:")
    stop_loss = risk_manager.calculate_dynamic_stop_loss(
        entry_price=50000.0,
        current_price=51000.0,
        volatility=0.03,
        atr=1500.0,
        market_regime="bull"
    )
    print(f"   Dynamic stop-loss: {stop_loss:.2%}")
    
    # Test risk metrics
    print("\n3. Testing Risk Metrics:")
    positions = [
        PositionInfo(
            symbol="BTC/USDT",
            size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            leverage=1.0,
            margin_used=5000.0,
            risk_score=0.5
        )
    ]
    
    market_data = {
        'volatility': 0.03,
        'sharpe_ratio': 1.5
    }
    
    risk_metrics = risk_manager.update_risk_metrics(positions, market_data)
    print(f"   Portfolio value: ${risk_metrics.portfolio_value:.2f}")
    print(f"   Total exposure: ${risk_metrics.total_exposure:.2f}")
    print(f"   Current drawdown: {risk_metrics.current_drawdown:.2%}")
    print(f"   Volatility: {risk_metrics.volatility:.2%}")
    
    # Test risk limits
    print("\n4. Testing Risk Limits:")
    within_limits, violations = risk_manager.check_risk_limits(risk_metrics)
    print(f"   Within limits: {within_limits}")
    if violations:
        print(f"   Violations: {violations}")
    
    # Generate risk report
    print("\n5. Risk Report:")
    report = risk_manager.generate_risk_report()
    print(f"   Risk level: {report['risk_level'].value}")
    print(f"   Recommendations: {len(report['recommendations'])}")
    
    print("\nEnhanced Risk Management System test completed!")

if __name__ == "__main__":
    main()
