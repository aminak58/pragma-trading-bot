"""
Dynamic Stop Loss Management

Implements adaptive stop loss strategies based on:
- ATR (Average True Range) for volatility-based stops
- Market regime (trending vs ranging)
- Trade confidence and time in trade
- Trailing stop mechanisms
- Support/resistance levels
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class DynamicStopLoss:
    """
    Dynamic stop loss manager with multiple strategies.
    
    Provides adaptive stop loss calculation based on market conditions,
    volatility, and trade characteristics.
    """
    
    def __init__(
        self,
        base_stop_percent: float = 0.02,
        atr_multiplier: float = 2.0,
        trailing_activation: float = 0.01,
        trailing_distance: float = 0.005,
        regime_adjustments: bool = True
    ):
        """
        Initialize dynamic stop loss manager.
        
        Args:
            base_stop_percent: Base stop loss percentage (default: 2%)
            atr_multiplier: ATR multiplier for volatility-based stops (default: 2.0)
            trailing_activation: Profit threshold to activate trailing stop (default: 1%)
            trailing_distance: Trailing stop distance (default: 0.5%)
            regime_adjustments: Whether to adjust stops based on regime
        """
        self.base_stop_percent = base_stop_percent
        self.atr_multiplier = atr_multiplier
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance
        self.regime_adjustments = regime_adjustments
        
        # Active trades tracking
        self.active_trades = {}
        
        logger.info("DynamicStopLoss initialized")
    
    def calculate_atr_stop(
        self,
        dataframe: pd.DataFrame,
        current_price: float,
        side: str = 'long',
        atr_period: int = 14
    ) -> float:
        """
        Calculate ATR-based stop loss.
        
        Args:
            dataframe: OHLCV dataframe with ATR data
            current_price: Current asset price
            side: Trade side ('long' or 'short')
            atr_period: ATR calculation period
            
        Returns:
            Stop loss price
        """
        if 'atr' not in dataframe.columns:
            # Calculate ATR if not present
            atr = self._calculate_atr(dataframe, atr_period)
        else:
            atr = dataframe['atr'].iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            logger.warning("Invalid ATR, using base stop percentage")
            return self._calculate_percentage_stop(current_price, side, self.base_stop_percent)
        
        # Calculate stop distance based on ATR
        stop_distance = atr * self.atr_multiplier
        
        if side == 'long':
            stop_price = current_price - stop_distance
        else:  # short
            stop_price = current_price + stop_distance
        
        logger.debug(f"ATR stop: price={current_price:.4f}, atr={atr:.4f}, stop={stop_price:.4f}")
        
        return stop_price
    
    def calculate_regime_stop(
        self,
        current_price: float,
        side: str,
        regime: str,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate regime-adaptive stop loss.
        
        Args:
            current_price: Current asset price
            side: Trade side ('long' or 'short')
            regime: Market regime
            confidence: Trade confidence (0-1)
            
        Returns:
            Stop loss price
        """
        # Base stop percentage
        base_stop = self.base_stop_percent
        
        # Regime adjustments
        if self.regime_adjustments:
            regime_multipliers = {
                'trending': 1.2,           # Wider stops for trends
                'low_volatility': 0.8,     # Tighter stops in low vol
                'high_volatility': 1.5,    # Much wider stops in high vol
                'unknown': 1.0             # Default for unknown
            }
            
            regime_multiplier = regime_multipliers.get(regime, 1.0)
            base_stop *= regime_multiplier
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        base_stop *= confidence_multiplier
        
        return self._calculate_percentage_stop(current_price, side, base_stop)
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        highest_price: Optional[float] = None,
        lowest_price: Optional[float] = None
    ) -> float:
        """
        Calculate trailing stop loss.
        
        Args:
            entry_price: Original entry price
            current_price: Current asset price
            side: Trade side ('long' or 'short')
            highest_price: Highest price since entry (for long trades)
            lowest_price: Lowest price since entry (for short trades)
            
        Returns:
            Trailing stop price
        """
        if side == 'long':
            # For long trades, trail from highest price
            if highest_price is None:
                highest_price = current_price
            
            # Check if we should activate trailing
            profit_percent = (highest_price - entry_price) / entry_price
            
            if profit_percent < self.trailing_activation:
                # Not enough profit, use entry stop
                return entry_price * (1 - self.base_stop_percent)
            
            # Calculate trailing stop from highest price
            trailing_stop = highest_price * (1 - self.trailing_distance)
            
            # Don't move stop against us
            entry_stop = entry_price * (1 - self.base_stop_percent)
            return max(trailing_stop, entry_stop)
        
        else:  # short
            # For short trades, trail from lowest price
            if lowest_price is None:
                lowest_price = current_price
            
            # Check if we should activate trailing
            profit_percent = (entry_price - lowest_price) / entry_price
            
            if profit_percent < self.trailing_activation:
                # Not enough profit, use entry stop
                return entry_price * (1 + self.base_stop_percent)
            
            # Calculate trailing stop from lowest price
            trailing_stop = lowest_price * (1 + self.trailing_distance)
            
            # Don't move stop against us
            entry_stop = entry_price * (1 + self.base_stop_percent)
            return min(trailing_stop, entry_stop)
    
    def calculate_adaptive_stop(
        self,
        dataframe: pd.DataFrame,
        current_price: float,
        side: str,
        regime: str = 'unknown',
        confidence: float = 1.0,
        time_in_trade: int = 0,
        entry_price: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate adaptive stop loss using multiple methods.
        
        Args:
            dataframe: OHLCV dataframe
            current_price: Current asset price
            side: Trade side ('long' or 'short')
            regime: Market regime
            confidence: Trade confidence (0-1)
            time_in_trade: Minutes in trade
            entry_price: Original entry price (for trailing)
            
        Returns:
            Dictionary with different stop loss calculations
        """
        stops = {}
        
        # ATR-based stop
        stops['atr'] = self.calculate_atr_stop(dataframe, current_price, side)
        
        # Regime-based stop
        stops['regime'] = self.calculate_regime_stop(current_price, side, regime, confidence)
        
        # Time-based adjustment
        time_multiplier = self._get_time_multiplier(time_in_trade)
        stops['time_adjusted'] = stops['regime'] * time_multiplier
        
        # Trailing stop (if entry price provided)
        if entry_price is not None:
            if side == 'long':
                highest_price = dataframe['high'].iloc[-min(50, len(dataframe)):].max()
                stops['trailing'] = self.calculate_trailing_stop(
                    entry_price, current_price, side, highest_price
                )
            else:
                lowest_price = dataframe['low'].iloc[-min(50, len(dataframe)):].min()
                stops['trailing'] = self.calculate_trailing_stop(
                    entry_price, current_price, side, lowest_price=lowest_price
                )
        
        # Select best stop based on regime and confidence
        recommended_stop = self._select_best_stop(stops, side, regime, confidence)
        stops['recommended'] = recommended_stop
        
        # Calculate stop distance and risk
        stop_distance = abs(current_price - recommended_stop)
        stop_percent = stop_distance / current_price
        
        stops['distance'] = stop_distance
        stops['percent'] = stop_percent
        
        logger.debug(
            f"Adaptive stop: price={current_price:.4f}, "
            f"recommended={recommended_stop:.4f}, "
            f"distance={stop_distance:.4f}, "
            f"percent={stop_percent:.3%}"
        )
        
        return stops
    
    def _calculate_atr(self, dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = dataframe['high']
        low = dataframe['low']
        close = dataframe['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR as rolling mean of TR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_percentage_stop(
        self,
        price: float,
        side: str,
        stop_percent: float
    ) -> float:
        """Calculate percentage-based stop loss."""
        if side == 'long':
            return price * (1 - stop_percent)
        else:  # short
            return price * (1 + stop_percent)
    
    def _get_time_multiplier(self, time_in_trade: int) -> float:
        """
        Get time-based stop adjustment multiplier.
        
        Tighter stops for longer trades to prevent giving back profits.
        """
        if time_in_trade < 60:  # Less than 1 hour
            return 1.0
        elif time_in_trade < 240:  # Less than 4 hours
            return 0.9
        elif time_in_trade < 480:  # Less than 8 hours
            return 0.8
        else:  # More than 8 hours
            return 0.7
    
    def _select_best_stop(
        self,
        stops: Dict[str, float],
        side: str,
        regime: str,
        confidence: float
    ) -> float:
        """
        Select the best stop loss from available options.
        
        Strategy:
        - High confidence: Use tighter stops
        - Low confidence: Use wider stops
        - Trending regime: Prefer ATR stops
        - Ranging regime: Prefer percentage stops
        """
        available_stops = [v for v in stops.values() if v is not None]
        
        if not available_stops:
            logger.warning("No valid stops calculated, using base stop")
            return stops.get('regime', 0.0)
        
        if side == 'long':
            if confidence > 0.8:
                # High confidence: use tighter stop
                return min(available_stops)
            elif regime == 'trending' and 'atr' in stops:
                # Trending: prefer ATR
                return stops['atr']
            else:
                # Default: use regime-based
                return stops['regime']
        else:  # short
            if confidence > 0.8:
                # High confidence: use tighter stop
                return max(available_stops)
            elif regime == 'trending' and 'atr' in stops:
                # Trending: prefer ATR
                return stops['atr']
            else:
                # Default: use regime-based
                return stops['regime']
    
    def update_trailing_stop(
        self,
        trade_id: str,
        current_price: float,
        dataframe: pd.DataFrame
    ) -> Optional[float]:
        """
        Update trailing stop for an active trade.
        
        Args:
            trade_id: Unique trade identifier
            current_price: Current asset price
            dataframe: Recent OHLCV data
            
        Returns:
            Updated stop price or None if no update needed
        """
        if trade_id not in self.active_trades:
            return None
        
        trade = self.active_trades[trade_id]
        side = trade['side']
        entry_price = trade['entry_price']
        
        # Calculate new trailing stop
        if side == 'long':
            highest_price = max(trade.get('highest_price', entry_price), current_price)
            new_stop = self.calculate_trailing_stop(
                entry_price, current_price, side, highest_price
            )
            
            # Only update if stop moved in our favor
            if new_stop > trade['stop_price']:
                trade['stop_price'] = new_stop
                trade['highest_price'] = highest_price
                return new_stop
        else:  # short
            lowest_price = min(trade.get('lowest_price', entry_price), current_price)
            new_stop = self.calculate_trailing_stop(
                entry_price, current_price, side, lowest_price=lowest_price
            )
            
            # Only update if stop moved in our favor
            if new_stop < trade['stop_price']:
                trade['stop_price'] = new_stop
                trade['lowest_price'] = lowest_price
                return new_stop
        
        return None
    
    def add_trade(
        self,
        trade_id: str,
        side: str,
        entry_price: float,
        stop_price: float,
        regime: str = 'unknown'
    ) -> None:
        """
        Add a new trade to tracking.
        
        Args:
            trade_id: Unique trade identifier
            side: Trade side ('long' or 'short')
            entry_price: Entry price
            stop_price: Initial stop price
            regime: Market regime
        """
        self.active_trades[trade_id] = {
            'side': side,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'regime': regime,
            'highest_price': entry_price if side == 'long' else None,
            'lowest_price': entry_price if side == 'short' else None
        }
        
        logger.info(f"Added trade {trade_id}: {side} at {entry_price:.4f}, stop at {stop_price:.4f}")
    
    def remove_trade(self, trade_id: str) -> None:
        """Remove trade from tracking."""
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]
            logger.info(f"Removed trade {trade_id}")
    
    def get_trade_stop(self, trade_id: str) -> Optional[float]:
        """Get current stop price for a trade."""
        if trade_id in self.active_trades:
            return self.active_trades[trade_id]['stop_price']
        return None
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DynamicStopLoss(active_trades={len(self.active_trades)})"
