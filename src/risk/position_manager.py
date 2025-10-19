"""
Position Management System

Manages position sizing, risk allocation, and position adjustments based on:
- Kelly Criterion calculations
- Market regime and confidence
- Portfolio risk limits
- Dynamic position scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from .kelly_criterion import KellyCriterion
from .dynamic_stops import DynamicStopLoss

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Comprehensive position management system.
    
    Integrates Kelly Criterion, dynamic stops, and risk management
    to provide optimal position sizing and management.
    """
    
    def __init__(
        self,
        max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
        max_position_risk: float = 0.01,   # 1% max per position
        max_correlation: float = 0.7,      # Max correlation between positions
        rebalance_threshold: float = 0.1   # 10% threshold for rebalancing
    ):
        """
        Initialize position manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk (as fraction of capital)
            max_position_risk: Maximum risk per individual position
            max_correlation: Maximum correlation between positions
            rebalance_threshold: Threshold for triggering rebalancing
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.rebalance_threshold = rebalance_threshold
        
        # Initialize components
        self.kelly_criterion = KellyCriterion()
        self.dynamic_stops = DynamicStopLoss()
        
        # Position tracking
        self.positions = {}
        self.portfolio_value = 0.0
        self.available_capital = 0.0
        
        # Performance tracking
        self.performance_history = []
        
        logger.info("PositionManager initialized")
    
    def calculate_position_size(
        self,
        pair: str,
        current_price: float,
        stop_loss_price: float,
        confidence: float,
        regime: str,
        win_rate: Optional[float] = None,
        win_loss_ratio: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal position size for a trade.
        
        Args:
            pair: Trading pair
            current_price: Current asset price
            stop_loss_price: Stop loss price
            confidence: Trade confidence (0-1)
            regime: Market regime
            win_rate: Historical win rate (optional)
            win_loss_ratio: Historical win/loss ratio (optional)
            
        Returns:
            Position sizing information
        """
        # Calculate Kelly-based position size
        kelly_result = self.kelly_criterion.calculate_position_size(
            account_balance=self.available_capital,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            confidence=confidence,
            regime=regime
        )
        
        # Apply portfolio risk limits
        position_size = self._apply_risk_limits(
            pair=pair,
            position_size=kelly_result['position_size'],
            position_value=kelly_result['position_value'],
            current_price=current_price,
            stop_loss_price=stop_loss_price
        )
        
        # Calculate final position metrics
        position_value = position_size * current_price
        risk_amount = abs(current_price - stop_loss_price) * position_size
        risk_percent = risk_amount / self.portfolio_value if self.portfolio_value > 0 else 0
        
        result = {
            'pair': pair,
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'kelly_fraction': kelly_result['kelly_fraction'],
            'confidence': confidence,
            'regime': regime,
            'leverage': position_value / self.available_capital if self.available_capital > 0 else 0
        }
        
        logger.info(
            f"Position sizing for {pair}: size={position_size:.4f}, "
            f"value={position_value:.2f}, risk={risk_percent:.2%}"
        )
        
        return result
    
    def _apply_risk_limits(
        self,
        pair: str,
        position_size: float,
        position_value: float,
        current_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Apply portfolio and position risk limits.
        
        Args:
            pair: Trading pair
            position_size: Calculated position size
            position_value: Calculated position value
            current_price: Current asset price
            stop_loss_price: Stop loss price
            
        Returns:
            Adjusted position size
        """
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        
        if risk_per_share <= 0:
            logger.warning(f"Invalid stop loss for {pair}, using 1% risk")
            risk_per_share = current_price * 0.01
        
        # Calculate maximum position size based on risk limits
        max_risk_amount = self.portfolio_value * self.max_position_risk
        max_position_by_risk = max_risk_amount / risk_per_share
        
        # Calculate maximum position size based on available capital
        max_position_by_capital = self.available_capital / current_price
        
        # Use the more restrictive limit
        max_position_size = min(max_position_by_risk, max_position_by_capital)
        
        # Apply correlation limits
        max_position_size = self._apply_correlation_limits(
            pair, max_position_size, current_price
        )
        
        # Final position size
        final_position_size = min(position_size, max_position_size)
        
        # Ensure minimum viable position
        min_position_size = 0.001  # Minimum 0.1% of portfolio
        if final_position_size < min_position_size:
            final_position_size = 0.0
        
        logger.debug(
            f"Risk limits for {pair}: "
            f"kelly_size={position_size:.4f}, "
            f"max_by_risk={max_position_by_risk:.4f}, "
            f"max_by_capital={max_position_by_capital:.4f}, "
            f"final={final_position_size:.4f}"
        )
        
        return final_position_size
    
    def _apply_correlation_limits(
        self,
        pair: str,
        position_size: float,
        current_price: float
    ) -> float:
        """
        Apply correlation-based position limits.
        
        Reduces position size if highly correlated positions exist.
        """
        if not self.positions:
            return position_size
        
        # Calculate correlation with existing positions
        # This is a simplified version - in practice, you'd use actual correlation data
        base_currency = pair.split('/')[0]
        correlated_exposure = 0.0
        
        for existing_pair, pos in self.positions.items():
            existing_base = existing_pair.split('/')[0]
            
            # Simple correlation based on base currency
            if existing_base == base_currency:
                correlation = 0.9  # High correlation for same base currency
            elif existing_base in ['BTC', 'ETH'] and base_currency in ['BTC', 'ETH']:
                correlation = 0.7  # Medium correlation for major cryptos
            else:
                correlation = 0.3  # Low correlation for different assets
            
            if correlation > self.max_correlation:
                correlated_exposure += pos['position_value'] * correlation
        
        # Reduce position size based on correlated exposure
        max_correlated_value = self.portfolio_value * 0.3  # Max 30% in correlated positions
        
        if correlated_exposure > max_correlated_value:
            reduction_factor = max_correlated_value / (correlated_exposure + 1e-8)
            position_size *= reduction_factor
            
            logger.debug(
                f"Correlation limit applied to {pair}: "
                f"correlated_exposure={correlated_exposure:.2f}, "
                f"reduction_factor={reduction_factor:.3f}"
            )
        
        return position_size
    
    def open_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        confidence: float,
        regime: str,
        position_data: Dict[str, float]
    ) -> str:
        """
        Open a new position.
        
        Args:
            pair: Trading pair
            side: Trade side ('long' or 'short')
            entry_price: Entry price
            stop_loss_price: Stop loss price
            confidence: Trade confidence
            regime: Market regime
            position_data: Position sizing data
            
        Returns:
            Position ID
        """
        position_id = f"{pair}_{side}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create position record
        position = {
            'id': position_id,
            'pair': pair,
            'side': side,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'position_size': position_data['position_size'],
            'position_value': position_data['position_value'],
            'confidence': confidence,
            'regime': regime,
            'opened_at': pd.Timestamp.now(),
            'status': 'open',
            'unrealized_pnl': 0.0,
            'highest_price': entry_price if side == 'long' else None,
            'lowest_price': entry_price if side == 'short' else None
        }
        
        # Add to tracking
        self.positions[position_id] = position
        
        # Add to dynamic stops tracking
        self.dynamic_stops.add_trade(
            trade_id=position_id,
            side=side,
            entry_price=entry_price,
            stop_price=stop_loss_price,
            regime=regime
        )
        
        # Update available capital
        self.available_capital -= position_data['position_value']
        
        logger.info(
            f"Opened position {position_id}: {side} {pair} "
            f"size={position_data['position_size']:.4f} "
            f"value={position_data['position_value']:.2f}"
        )
        
        return position_id
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        dataframe: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """
        Update position with current market data.
        
        Args:
            position_id: Position identifier
            current_price: Current asset price
            dataframe: Recent OHLCV data
            
        Returns:
            Updated position data or None if position not found
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        # Calculate unrealized P&L
        if position['side'] == 'long':
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['position_size']
            position['highest_price'] = max(position['highest_price'], current_price)
        else:  # short
            position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['position_size']
            position['lowest_price'] = min(position['lowest_price'], current_price)
        
        # Update trailing stop
        new_stop = self.dynamic_stops.update_trailing_stop(
            position_id, current_price, dataframe
        )
        
        if new_stop:
            position['stop_loss_price'] = new_stop
            logger.debug(f"Updated trailing stop for {position_id}: {new_stop:.4f}")
        
        return {
            'unrealized_pnl': position['unrealized_pnl'],
            'stop_loss_price': position['stop_loss_price'],
            'current_price': current_price
        }
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = 'manual'
    ) -> Optional[Dict[str, float]]:
        """
        Close a position.
        
        Args:
            position_id: Position identifier
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Position summary or None if not found
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        # Calculate final P&L
        if position['side'] == 'long':
            realized_pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # short
            realized_pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Calculate return percentage
        return_percent = realized_pnl / position['position_value']
        
        # Update position
        position['exit_price'] = exit_price
        position['realized_pnl'] = realized_pnl
        position['return_percent'] = return_percent
        position['closed_at'] = pd.Timestamp.now()
        position['status'] = 'closed'
        position['close_reason'] = reason
        
        # Update available capital
        self.available_capital += position['position_value'] + realized_pnl
        
        # Add to performance history
        self.performance_history.append({
            'position_id': position_id,
            'pair': position['pair'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'realized_pnl': realized_pnl,
            'return_percent': return_percent,
            'confidence': position['confidence'],
            'regime': position['regime'],
            'duration_minutes': (position['closed_at'] - position['opened_at']).total_seconds() / 60,
            'close_reason': reason
        })
        
        # Update Kelly Criterion with trade result
        self.kelly_criterion.add_trade_result(
            profit_loss=realized_pnl,
            confidence=position['confidence']
        )
        
        # Remove from tracking
        del self.positions[position_id]
        self.dynamic_stops.remove_trade(position_id)
        
        logger.info(
            f"Closed position {position_id}: P&L={realized_pnl:.2f} "
            f"({return_percent:.2%}) reason={reason}"
        )
        
        return {
            'realized_pnl': realized_pnl,
            'return_percent': return_percent,
            'duration_minutes': (position['closed_at'] - position['opened_at']).total_seconds() / 60
        }
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check all positions for stop loss triggers.
        
        Args:
            current_prices: Dictionary of pair -> current price
            
        Returns:
            List of position IDs that hit stop loss
        """
        triggered_positions = []
        
        for position_id, position in self.positions.items():
            pair = position['pair']
            if pair not in current_prices:
                continue
            
            current_price = current_prices[pair]
            stop_price = position['stop_loss_price']
            
            # Check stop loss trigger
            if position['side'] == 'long' and current_price <= stop_price:
                triggered_positions.append(position_id)
            elif position['side'] == 'short' and current_price >= stop_price:
                triggered_positions.append(position_id)
        
        return triggered_positions
    
    def get_portfolio_summary(self) -> Dict[str, float]:
        """
        Get current portfolio summary.
        
        Returns:
            Portfolio metrics
        """
        total_value = sum(pos['position_value'] for pos in self.positions.values())
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        # Calculate total risk
        total_risk = 0.0
        for position in self.positions.values():
            risk_per_share = abs(position['entry_price'] - position['stop_loss_price'])
            position_risk = risk_per_share * position['position_size']
            total_risk += position_risk
        
        portfolio_risk_percent = total_risk / self.portfolio_value if self.portfolio_value > 0 else 0
        
        return {
            'total_positions': len(self.positions),
            'total_value': total_value,
            'available_capital': self.available_capital,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_risk': total_risk,
            'portfolio_risk_percent': portfolio_risk_percent,
            'portfolio_value': self.portfolio_value
        }
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update total portfolio value."""
        self.portfolio_value = new_value
        self.available_capital = new_value - sum(pos['position_value'] for pos in self.positions.values())
        
        logger.info(f"Updated portfolio value: {new_value:.2f}, available: {self.available_capital:.2f}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_pnl': 0.0
            }
        
        total_trades = len(self.performance_history)
        winning_trades = len([t for t in self.performance_history if t['realized_pnl'] > 0])
        losing_trades = len([t for t in self.performance_history if t['realized_pnl'] < 0])
        
        total_pnl = sum(t['realized_pnl'] for t in self.performance_history)
        avg_return = np.mean([t['return_percent'] for t in self.performance_history])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0.0,
            'avg_return': avg_return,
            'total_pnl': total_pnl
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"PositionManager(positions={len(self.positions)}, portfolio_value={self.portfolio_value:.2f})"
