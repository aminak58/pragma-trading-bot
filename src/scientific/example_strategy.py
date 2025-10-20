"""
Scientific Trading Strategy Framework - Example Strategy

This module provides an example implementation of a scientific trading strategy
that demonstrates the framework usage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

class ExampleScientificStrategy:
    """
    Example implementation of scientific trading strategy
    
    This strategy implements a simple mean reversion approach based on RSI
    and Bollinger Bands with proper hypothesis and validation.
    """
    
    def __init__(self):
        """
        Initialize the example strategy
        """
        self.hypothesis = """
        Hypothesis: Mean reversion strategy will be profitable in sideways markets
        when RSI indicates oversold conditions and price is near lower Bollinger Band.
        
        Rationale: 
        - Oversold RSI (< 30) indicates potential reversal
        - Price near lower Bollinger Band suggests mean reversion opportunity
        - Strategy targets quick profits from temporary price dislocations
        """
        
        self.parameters = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_threshold': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.015
        }
        
        self.logger = logging.getLogger(__name__)
        self.trained = False
    
    def train(self, training_data: pd.DataFrame) -> bool:
        """
        Train strategy on training data
        
        Args:
            training_data: Training data DataFrame
            
        Returns:
            True if training successful
        """
        try:
            self.logger.info("Training example strategy...")
            
            # Calculate indicators
            training_data = self.calculate_indicators(training_data)
            
            # Optimize parameters (simplified)
            self.optimize_parameters(training_data)
            
            self.trained = True
            self.logger.info("Strategy training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Strategy training failed: {e}")
            return False
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data: Price data DataFrame
            
        Returns:
            DataFrame with indicators
        """
        # Calculate RSI
        data['rsi'] = self.calculate_rsi(data['close'], self.parameters['rsi_period'])
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            data['close'], self.parameters['bb_period'], self.parameters['bb_std']
        )
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        data['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Calculate price position relative to Bollinger Bands
        data['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Calculate volume ratio
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        return data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI indicator
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> tuple:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (upper, middle, lower) bands
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def optimize_parameters(self, data: pd.DataFrame) -> None:
        """
        Optimize strategy parameters (simplified)
        
        Args:
            data: Training data with indicators
        """
        # Simple parameter optimization based on historical performance
        # In practice, this would use more sophisticated optimization
        
        # Find optimal RSI oversold threshold
        rsi_values = np.arange(20, 40, 2)
        best_rsi = self.parameters['rsi_oversold']
        best_performance = 0
        
        for rsi_threshold in rsi_values:
            # Calculate performance for this threshold
            performance = self.calculate_performance_for_threshold(data, rsi_threshold)
            if performance > best_performance:
                best_performance = performance
                best_rsi = rsi_threshold
        
        self.parameters['rsi_oversold'] = best_rsi
        self.logger.info(f"Optimized RSI oversold threshold: {best_rsi}")
    
    def calculate_performance_for_threshold(self, data: pd.DataFrame, 
                                          rsi_threshold: float) -> float:
        """
        Calculate performance for a given RSI threshold
        
        Args:
            data: Data with indicators
            rsi_threshold: RSI threshold to test
            
        Returns:
            Performance score
        """
        # Generate signals
        signals = self.generate_signals(data, rsi_threshold)
        
        # Calculate simple performance metric
        if len(signals) > 0:
            winning_signals = signals[signals['pnl'] > 0]
            win_rate = len(winning_signals) / len(signals)
            avg_return = signals['pnl'].mean()
            return win_rate * avg_return
        else:
            return 0.0
    
    def generate_signals(self, data: pd.DataFrame, 
                        rsi_threshold: float = None) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            data: Data with indicators
            rsi_threshold: RSI threshold (uses parameter if None)
            
        Returns:
            DataFrame with signals
        """
        if rsi_threshold is None:
            rsi_threshold = self.parameters['rsi_oversold']
        
        # Entry conditions
        entry_condition = (
            (data['rsi'] < rsi_threshold) &
            (data['bb_position'] < self.parameters['bb_threshold']) &
            (data['volume_ratio'] > 1.2) &
            (data['bb_width'] > 0.02)  # Ensure sufficient volatility
        )
        
        # Exit conditions
        exit_condition = (
            (data['rsi'] > self.parameters['rsi_overbought']) |
            (data['bb_position'] > 0.8) |
            (data['volume_ratio'] < 0.8)
        )
        
        # Generate signals
        signals = []
        in_position = False
        entry_price = 0
        
        for i, (idx, row) in enumerate(data.iterrows()):
            if not in_position and entry_condition.iloc[i]:
                # Enter position
                in_position = True
                entry_price = row['close']
                entry_time = idx
                
            elif in_position and exit_condition.iloc[i]:
                # Exit position
                exit_price = row['close']
                pnl = (exit_price - entry_price) / entry_price
                
                signals.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'duration': (idx - entry_time).total_seconds() / 3600  # hours
                })
                
                in_position = False
        
        return pd.DataFrame(signals)
    
    def generate_trades(self, signals: pd.DataFrame, 
                       data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trade data from signals
        
        Args:
            signals: Signal DataFrame
            data: Price data
            
        Returns:
            Trade DataFrame
        """
        if len(signals) == 0:
            return pd.DataFrame()
        
        trades = signals.copy()
        
        # Add position sizing
        trades['size'] = 0.02  # 2% position size
        
        # Add stop loss and take profit
        trades['stop_loss'] = trades['entry_price'] * (1 - self.parameters['stop_loss'])
        trades['take_profit'] = trades['entry_price'] * (1 + self.parameters['take_profit'])
        
        return trades
    
    def calculate_returns(self, trades: pd.DataFrame, 
                         data: pd.DataFrame) -> pd.Series:
        """
        Calculate returns from trades
        
        Args:
            trades: Trade DataFrame
            data: Price data
            
        Returns:
            Series of returns
        """
        if len(trades) == 0:
            return pd.Series(dtype=float)
        
        # Calculate returns for each trade
        returns = trades['pnl'] * trades['size']
        
        # Create time series of returns
        return_series = pd.Series(returns.values, index=trades['exit_time'])
        
        return return_series
    
    def get_hypothesis(self) -> str:
        """
        Get strategy hypothesis
        
        Returns:
            Strategy hypothesis
        """
        return self.hypothesis
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters
        
        Returns:
            Strategy parameters
        """
        return self.parameters
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary
        
        Returns:
            Performance summary
        """
        return {
            'strategy_name': self.__class__.__name__,
            'hypothesis': self.hypothesis,
            'parameters': self.parameters,
            'trained': self.trained
        }
