"""
Simple Reliable Strategy - Phase 3 Implementation
- Uses simple technical indicators: RSI + BB + Volume + Trend
- No HMM complexity, proven indicators only
- Target: WinRate > 50%, Sharpe > 1.5, MDD < 0.5%, NumTrades: 10-20/week
"""
from pathlib import Path
import sys

# Add project root and src to path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / 'src'))

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.strategy import IStrategy
import talib.abstract as ta

class SimpleReliableStrategy(IStrategy):
    """
    Simple Reliable Strategy - Phase 3
    - Uses simple technical indicators: RSI + BB + Volume + Trend
    - No HMM complexity, proven indicators only
    - Target: WinRate > 50%, Sharpe > 1.5, MDD < 0.5%, NumTrades: 10-20/week
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 200
    can_short = False

    # Conservative ROI
    minimal_roi = {"0": 0.015, "20": 0.01, "40": 0.005, "80": 0.002}
    
    # Dynamic stoploss
    stoploss = -0.02
    
    # Position sizing
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    def __init__(self, config: dict) -> None:
        super().__init__(config)

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Basic technical indicators
        df['ema20'] = ta.EMA(df, timeperiod=20)
        df['ema50'] = ta.EMA(df, timeperiod=50)
        df['ema100'] = ta.EMA(df, timeperiod=100)
        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['adx'] = ta.ADX(df, timeperiod=14)
        df['atr'] = ta.ATR(df, timeperiod=14)
        
        # Bollinger Bands
        bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df['bb_upper'] = bb['upperband']
        df['bb_middle'] = bb['middleband']
        df['bb_lower'] = bb['lowerband']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        macd = ta.MACD(df)
        df['macd'] = macd['macd']
        df['macd_signal'] = macd['macdsignal']
        df['macd_hist'] = macd['macdhist']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Price momentum
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # Trend strength
        df['trend_strength'] = (df['ema20'] - df['ema50']) / df['ema50']
        df['trend_direction'] = np.where(df['ema20'] > df['ema50'], 1, -1)
        
        # Volatility
        df['volatility'] = df['price_change_5'].rolling(20).std()
        df['volatility_percentile'] = df['volatility'].rolling(100).rank(pct=True)
        
        # Support/Resistance levels
        df['support'] = df['low'].rolling(20).min()
        df['resistance'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support']) / df['close']
        df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = False
        
        # Simple but effective entry conditions
        entry_mask = (
            # Trend confirmation
            (df['ema20'] > df['ema50']) &  # Short EMA above long EMA
            (df['ema50'] > df['ema100']) &  # Medium EMA above long EMA
            (df['trend_strength'] > 0.01) &  # Strong trend
            
            # RSI conditions
            (df['rsi'] > 45) &  # Not oversold
            (df['rsi'] < 75) &  # Not overbought
            
            # Bollinger Bands
            (df['bb_position'] > 0.2) &  # Above lower BB
            (df['bb_position'] < 0.8) &  # Below upper BB
            (df['bb_width'] > 0.02) &  # Sufficient volatility
            (df['bb_width'] < 0.08) &  # Not extreme volatility
            
            # MACD confirmation
            (df['macd'] > df['macd_signal']) &  # MACD bullish
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD improving
            
            # Volume confirmation
            (df['volume_ratio'] > 1.1) &  # Above average volume
            (df['volume_trend'] > 0.9) &  # Volume trend positive
            
            # Price momentum
            (df['price_change_5'] > 0) &  # Positive short-term momentum
            (df['price_change_10'] > -0.01) &  # Not severely declining
            
            # Support/Resistance
            (df['support_distance'] > 0.005) &  # Above support
            (df['resistance_distance'] > 0.01) &  # Room to resistance
            
            # Market structure
            (df['higher_high'] == True) &  # Higher high pattern
            
            # Volatility
            (df['volatility_percentile'] > 0.3) &  # Not too low volatility
            (df['volatility_percentile'] < 0.8)  # Not too high volatility
        )
        
        df['enter_long'] = entry_mask
        
        # Enter tags for analysis
        df['enter_tag'] = 'Simple_Entry'
        
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = False
        
        # Simple but effective exit conditions
        exit_mask = (
            # Trend weakening
            (df['ema20'] < df['ema50']) |  # Short EMA below medium EMA
            (df['trend_strength'] < 0.005) |  # Weak trend
            
            # RSI conditions
            (df['rsi'] > 80) |  # Severely overbought
            (df['rsi'] < 35) |  # Severely oversold
            
            # Bollinger Bands
            (df['bb_position'] > 0.9) |  # Near upper BB
            (df['bb_position'] < 0.1) |  # Near lower BB
            (df['bb_width'] > 0.10) |  # Extreme volatility
            
            # MACD bearish
            (df['macd'] < df['macd_signal']) |  # MACD bearish
            (df['macd_hist'] < df['macd_hist'].shift(1)) |  # MACD deteriorating
            
            # Volume issues
            (df['volume_ratio'] < 0.7) |  # Very low volume
            (df['volume_trend'] < 0.8) |  # Volume trend negative
            
            # Price momentum
            (df['price_change_5'] < -0.005) |  # Negative momentum
            (df['price_change_10'] < -0.01) |  # Declining trend
            
            # Support/Resistance
            (df['support_distance'] < 0.002) |  # Near support
            (df['resistance_distance'] < 0.005) |  # Near resistance
            
            # Market structure
            (df['lower_low'] == True) |  # Lower low pattern
            
            # Volatility
            (df['volatility_percentile'] > 0.9)  # Extreme volatility
        )
        
        df['exit_long'] = exit_mask
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Simple dynamic stoploss
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss
            
        last_candle = dataframe.iloc[-1]
        atr_percent = last_candle.get('atr', 0) / current_rate
        volatility_percentile = last_candle.get('volatility_percentile', 0.5)
        
        # Base stoploss
        base_stoploss = self.stoploss
        
        # Adjust based on ATR
        if atr_percent > 0.01:  # High volatility
            base_stoploss *= 1.2
        elif atr_percent < 0.005:  # Low volatility
            base_stoploss *= 0.8
            
        # Adjust based on volatility percentile
        if volatility_percentile > 0.7:  # High volatility
            base_stoploss *= 1.1
        elif volatility_percentile < 0.3:  # Low volatility
            base_stoploss *= 0.9
            
        # Adjust based on profit
        if current_profit > 0.01:  # In profit
            base_stoploss = max(base_stoploss, -0.015)  # Trail stop
        elif current_profit < -0.005:  # In loss
            base_stoploss *= 1.05  # Slightly wider
            
        return max(base_stoploss, -0.08)  # Cap at -8%

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Simple dynamic position sizing
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
            
        last_candle = dataframe.iloc[-1]
        volatility_percentile = last_candle.get('volatility_percentile', 0.5)
        trend_strength = last_candle.get('trend_strength', 0.0)
        
        # Base position size
        base_multiplier = 1.0
        
        # Adjust based on volatility
        if volatility_percentile > 0.7:  # High volatility
            base_multiplier *= 0.8
        elif volatility_percentile < 0.3:  # Low volatility
            base_multiplier *= 1.1
            
        # Adjust based on trend strength
        if trend_strength > 0.02:  # Strong trend
            base_multiplier *= 1.1
        elif trend_strength < 0.005:  # Weak trend
            base_multiplier *= 0.9
            
        adjusted_stake = proposed_stake * base_multiplier
        
        # Ensure within bounds
        return max(min_stake, min(adjusted_stake, max_stake))

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Conservative leverage
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 1.0
            
        last_candle = dataframe.iloc[-1]
        volatility_percentile = last_candle.get('volatility_percentile', 0.5)
        
        # Conservative leverage based on volatility
        if volatility_percentile > 0.7:  # High volatility
            return 1.0  # No leverage
        elif volatility_percentile < 0.3:  # Low volatility
            return min(1.5, max_leverage)
        else:  # Medium volatility
            return min(1.2, max_leverage)
