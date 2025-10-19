"""
Kelly Criterion Position Sizing

Implements the Kelly Criterion for optimal position sizing based on:
- Win rate (probability of winning)
- Average win/loss ratio
- Confidence in the trade
- Risk tolerance

The Kelly Criterion calculates the optimal fraction of capital to risk:
f* = (bp - q) / b

Where:
- f* = optimal fraction to bet
- b = odds received on the wager (win/loss ratio)
- p = probability of winning
- q = probability of losing (1-p)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator.
    
    Calculates optimal position size based on historical performance
    and current trade confidence.
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,
        min_kelly_fraction: float = 0.01,
        confidence_adjustment: bool = True,
        regime_adjustment: bool = True
    ):
        """
        Initialize Kelly Criterion calculator.
        
        Args:
            max_kelly_fraction: Maximum fraction of capital to risk (default: 25%)
            min_kelly_fraction: Minimum fraction of capital to risk (default: 1%)
            confidence_adjustment: Whether to adjust based on trade confidence
            regime_adjustment: Whether to adjust based on market regime
        """
        self.max_kelly_fraction = max_kelly_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.confidence_adjustment = confidence_adjustment
        self.regime_adjustment = regime_adjustment
        
        # Performance tracking
        self.trade_history = []
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_wins': 0.0,
            'total_losses': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0
        }
        
        logger.info("KellyCriterion initialized")
    
    def add_trade_result(self, profit_loss: float, confidence: float = 1.0) -> None:
        """
        Add a completed trade result to performance tracking.
        
        Args:
            profit_loss: Profit/loss of the trade (positive for win, negative for loss)
            confidence: Confidence level of the trade (0-1)
        """
        self.trade_history.append({
            'profit_loss': profit_loss,
            'confidence': confidence,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update performance stats
        self._update_performance_stats()
        
        logger.debug(f"Added trade result: {profit_loss:.4f}, confidence: {confidence:.2f}")
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics from trade history."""
        if not self.trade_history:
            return
        
        # Filter recent trades (last 100 or all if less)
        recent_trades = self.trade_history[-100:]
        
        wins = [t for t in recent_trades if t['profit_loss'] > 0]
        losses = [t for t in recent_trades if t['profit_loss'] < 0]
        
        self.performance_stats.update({
            'total_trades': len(recent_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'total_wins': sum(t['profit_loss'] for t in wins),
            'total_losses': abs(sum(t['profit_loss'] for t in losses)),
            'win_rate': len(wins) / len(recent_trades) if recent_trades else 0.0,
            'avg_win': np.mean([t['profit_loss'] for t in wins]) if wins else 0.0,
            'avg_loss': np.mean([abs(t['profit_loss']) for t in losses]) if losses else 0.0
        })
        
        # Calculate win/loss ratio
        if self.performance_stats['avg_loss'] > 0:
            self.performance_stats['win_loss_ratio'] = (
                self.performance_stats['avg_win'] / self.performance_stats['avg_loss']
            )
        else:
            self.performance_stats['win_loss_ratio'] = 0.0
    
    def calculate_kelly_fraction(
        self,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
        confidence: float = 1.0,
        regime: str = 'unknown'
    ) -> float:
        """
        Calculate optimal Kelly fraction for position sizing.
        
        Args:
            win_rate: Probability of winning (0-1). If None, uses historical average.
            win_loss_ratio: Average win / average loss ratio. If None, uses historical.
            confidence: Current trade confidence (0-1)
            regime: Current market regime ('trending', 'low_volatility', 'high_volatility')
            
        Returns:
            Optimal fraction of capital to risk (0-1)
        """
        # Use historical stats if not provided
        if win_rate is None:
            win_rate = self.performance_stats['win_rate']
        if win_loss_ratio is None:
            win_loss_ratio = self.performance_stats['win_loss_ratio']
        
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win_rate: {win_rate}, using 0.5")
            win_rate = 0.5
        
        if win_loss_ratio <= 0:
            logger.warning(f"Invalid win_loss_ratio: {win_loss_ratio}, using 1.0")
            win_loss_ratio = 1.0
        
        # Basic Kelly formula: f* = (bp - q) / b
        # Where b = win_loss_ratio, p = win_rate, q = 1 - win_rate
        b = win_loss_ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply confidence adjustment
        if self.confidence_adjustment:
            kelly_fraction *= confidence
        
        # Apply regime adjustment
        if self.regime_adjustment:
            kelly_fraction *= self._get_regime_multiplier(regime)
        
        # Apply bounds
        kelly_fraction = max(self.min_kelly_fraction, kelly_fraction)
        kelly_fraction = min(self.max_kelly_fraction, kelly_fraction)
        
        # Ensure non-negative
        kelly_fraction = max(0.0, kelly_fraction)
        
        logger.debug(
            f"Kelly calculation: win_rate={win_rate:.3f}, "
            f"win_loss_ratio={win_loss_ratio:.3f}, "
            f"confidence={confidence:.3f}, regime={regime}, "
            f"kelly_fraction={kelly_fraction:.3f}"
        )
        
        return kelly_fraction
    
    def _get_regime_multiplier(self, regime: str) -> float:
        """
        Get position size multiplier based on market regime.
        
        Args:
            regime: Market regime name
            
        Returns:
            Multiplier for position size (0-1)
        """
        regime_multipliers = {
            'trending': 1.0,           # Full position in trending markets
            'low_volatility': 0.8,     # Reduce position in low volatility
            'high_volatility': 0.5,    # Significantly reduce in high volatility
            'unknown': 0.3             # Conservative for unknown regimes
        }
        
        return regime_multipliers.get(regime, 0.5)
    
    def calculate_position_size(
        self,
        account_balance: float,
        current_price: float,
        stop_loss_price: float,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None,
        confidence: float = 1.0,
        regime: str = 'unknown'
    ) -> Dict[str, float]:
        """
        Calculate optimal position size for a trade.
        
        Args:
            account_balance: Current account balance
            current_price: Current asset price
            stop_loss_price: Stop loss price
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss ratio
            confidence: Trade confidence (0-1)
            regime: Market regime
            
        Returns:
            Dictionary with position sizing information
        """
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        
        if risk_per_share <= 0:
            logger.warning("Invalid stop loss price, using 1% risk")
            risk_per_share = current_price * 0.01
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            confidence=confidence,
            regime=regime
        )
        
        # Calculate position size
        risk_amount = account_balance * kelly_fraction
        position_size = risk_amount / risk_per_share
        
        # Calculate position value
        position_value = position_size * current_price
        
        # Calculate leverage (if applicable)
        leverage = position_value / account_balance if account_balance > 0 else 0.0
        
        result = {
            'kelly_fraction': kelly_fraction,
            'risk_amount': risk_amount,
            'position_size': position_size,
            'position_value': position_value,
            'leverage': leverage,
            'risk_per_share': risk_per_share,
            'risk_reward_ratio': risk_per_share / (current_price * 0.02)  # Assuming 2% target
        }
        
        logger.info(
            f"Position sizing: balance={account_balance:.2f}, "
            f"kelly_fraction={kelly_fraction:.3f}, "
            f"position_value={position_value:.2f}, "
            f"leverage={leverage:.2f}x"
        )
        
        return result
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get current performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_stats.copy()
    
    def reset_performance(self) -> None:
        """Reset performance tracking."""
        self.trade_history = []
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_wins': 0.0,
            'total_losses': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0
        }
        
        logger.info("Performance tracking reset")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KellyCriterion(max_fraction={self.max_kelly_fraction}, "
            f"trades={self.performance_stats['total_trades']}, "
            f"win_rate={self.performance_stats['win_rate']:.3f})"
        )
