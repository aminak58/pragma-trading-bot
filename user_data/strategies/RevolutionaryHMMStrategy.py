"""
Revolutionary HMM v2.0 Strategy - Phase 2 Implementation
- Addresses HMM over-confidence issue
- Uses regime-specific logic instead of Trend Phase Score
- Implements dynamic risk management
- Target: WinRate > 40%, Sharpe > 1.0, MDD < 1%
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

from src.regime.hmm_detector import RegimeDetector


class RevolutionaryHMMStrategy(IStrategy):
    """
    Revolutionary HMM v2.0 Strategy - Phase 2
    - Addresses HMM over-confidence issue
    - Uses regime-specific logic instead of Trend Phase Score
    - Implements dynamic risk management
    - Target: WinRate > 40%, Sharpe > 1.0, MDD < 1%
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 1000
    can_short = False

    # Conservative ROI
    minimal_roi = {"0": 0.02, "30": 0.015, "60": 0.01, "120": 0.005}
    
    # Dynamic stoploss
    stoploss = -0.03
    
    # Position sizing
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.detector = RegimeDetector()
        self.trained = False

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
        
        # MACD
        macd = ta.MACD(df)
        df['macd'] = macd['macd']
        df['macd_signal'] = macd['macdsignal']
        df['macd_hist'] = macd['macdhist']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['price_change_20'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility_5'] = df['price_change_5'].rolling(20).std()
        df['volatility_20'] = df['price_change_20'].rolling(20).std()

        # Train HMM once sufficient candles
        if not self.trained and len(df) >= 1000:
            try:
                self.detector.train(df.tail(1000))
                self.trained = True
            except Exception as e:
                self.trained = False

        # Predict regimes/conf if trained
        if self.trained:
            regimes, confidences = self.detector.predict_regime_sequence(df, smooth_window=3)
            # align
            if len(regimes) != len(df):
                n = min(len(regimes), len(df))
                regimes = regimes[-n:]
                confidences = confidences[-n:]
                df = df.tail(n).copy()
            df['regime'] = regimes
            df['regime_conf'] = confidences
        else:
            df['regime'] = 'unknown'
            df['regime_conf'] = 0.0

        # Regime-specific indicators
        df['regime_strength'] = self._calculate_regime_strength(df)
        df['market_momentum'] = self._calculate_market_momentum(df)
        df['volatility_regime'] = self._calculate_volatility_regime(df)

        return df

    def _calculate_regime_strength(self, df: DataFrame) -> pd.Series:
        """Calculate regime strength based on multiple factors"""
        strength = pd.Series(0.0, index=df.index)
        
        # EMA alignment strength
        ema_strength = 0.0
        if 'ema20' in df.columns and 'ema50' in df.columns:
            ema_strength = (df['ema20'] - df['ema50']) / df['ema50']
        
        # MACD strength
        macd_strength = 0.0
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_strength = (df['macd'] - df['macd_signal']) / abs(df['macd_signal'] + 1e-8)
        
        # RSI momentum
        rsi_momentum = 0.0
        if 'rsi' in df.columns:
            rsi_momentum = (df['rsi'] - 50) / 50
        
        # Combine factors
        strength = 0.4 * ema_strength + 0.3 * macd_strength + 0.3 * rsi_momentum
        return strength

    def _calculate_market_momentum(self, df: DataFrame) -> pd.Series:
        """Calculate market momentum"""
        momentum = pd.Series(0.0, index=df.index)
        
        if 'price_change_5' in df.columns and 'price_change_20' in df.columns:
            # Short-term momentum
            short_momentum = df['price_change_5']
            # Long-term momentum
            long_momentum = df['price_change_20']
            # Combined momentum
            momentum = 0.6 * short_momentum + 0.4 * long_momentum
        
        return momentum

    def _calculate_volatility_regime(self, df: DataFrame) -> pd.Series:
        """Calculate volatility regime"""
        vol_regime = pd.Series(0.0, index=df.index)
        
        if 'volatility_20' in df.columns:
            vol_percentile = df['volatility_20'].rolling(100).rank(pct=True)
            vol_regime = vol_percentile - 0.5  # Center around 0
        
        return vol_regime

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = False
        
        # Regime-specific entry logic
        entry_mask = pd.Series(False, index=df.index)
        
        # HIGH_VOLATILITY regime entries (best performer from analysis)
        high_vol_mask = (
            (df['regime'] == 'high_volatility') &
            (df['regime_conf'] > 0.6) &  # Lower confidence threshold
            (df['regime_strength'] > 0.1) &  # Positive regime strength
            (df['market_momentum'] > 0.001) &  # Positive momentum
            (df['rsi'] > 35) &  # Not oversold
            (df['rsi'] < 70) &  # Not overbought
            (df['close'] > df['ema20']) &  # Above short EMA
            (df['macd'] > df['macd_signal']) &  # MACD bullish
            (df['volume_ratio'] > 1.1) &  # Above average volume
            (df['bb_width'] > 0.02) &  # Sufficient volatility
            (df['bb_width'] < 0.08)  # Not extreme volatility
        )
        
        # LOW_VOLATILITY regime entries (mean reversion)
        low_vol_mask = (
            (df['regime'] == 'low_volatility') &
            (df['regime_conf'] > 0.6) &
            (df['regime_strength'] < -0.1) &  # Negative regime strength (oversold)
            (df['market_momentum'] < -0.001) &  # Negative momentum (oversold)
            (df['rsi'] < 40) &  # Oversold
            (df['rsi'] > 20) &  # Not extremely oversold
            (df['close'] < df['bb_lower'] * 1.02) &  # Near lower BB
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD improving
            (df['bb_width'] < 0.04) &  # Low volatility
            (df['volume_ratio'] > 0.9)  # Decent volume
        )
        
        # TRENDING regime entries (avoided based on analysis)
        # Skip trending regime as it showed negative returns
        
        # Combine masks
        entry_mask = high_vol_mask | low_vol_mask
        
        df['enter_long'] = entry_mask
        
        # Enter tags for analysis
        df['enter_tag'] = None
        df.loc[high_vol_mask, 'enter_tag'] = 'HighVol_Entry'
        df.loc[low_vol_mask, 'enter_tag'] = 'LowVol_Entry'
        
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = False
        
        # Regime-specific exit logic
        exit_mask = pd.Series(False, index=df.index)
        
        # HIGH_VOLATILITY exits
        high_vol_exit = (
            (df['regime'] == 'high_volatility') &
            (
                (df['rsi'] > 75) |  # Overbought
                (df['close'] < df['ema20']) |  # Below short EMA
                (df['macd'] < df['macd_signal']) |  # MACD bearish
                (df['bb_width'] > 0.10) |  # Extreme volatility
                (df['regime_strength'] < 0.05) |  # Weak regime strength
                (df['market_momentum'] < 0.0)  # Negative momentum
            )
        )
        
        # LOW_VOLATILITY exits
        low_vol_exit = (
            (df['regime'] == 'low_volatility') &
            (
                (df['rsi'] > 60) |  # Approaching overbought
                (df['close'] > df['bb_upper'] * 0.98) |  # Near upper BB
                (df['macd'] < df['macd_signal']) |  # MACD bearish
                (df['regime_strength'] > 0.05) |  # Strong regime strength
                (df['market_momentum'] > 0.001)  # Positive momentum
            )
        )
        
        # Universal exits
        universal_exit = (
            (df['rsi'] > 85) |  # Severely overbought
            (df['close'] < df['ema50']) |  # Below long EMA
            (df['volume_ratio'] < 0.5) |  # Very low volume
            (df['bb_width'] > 0.12)  # Extreme volatility
        )
        
        # Combine masks
        exit_mask = high_vol_exit | low_vol_exit | universal_exit
        
        df['exit_long'] = exit_mask
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced dynamic stoploss based on regime
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        regime_conf = last_candle.get('regime_conf', 0.0)
        atr_percent = last_candle.get('atr', 0) / current_rate
        
        # Base stoploss by regime
        if regime == 'high_volatility':
            base_stoploss = -0.04  # Tighter for high vol (best performer)
        elif regime == 'low_volatility':
            base_stoploss = -0.02  # Much tighter for low vol (mean reversion)
        elif regime == 'trending':
            base_stoploss = -0.06  # Wider for trending (avoided)
        else:
            base_stoploss = self.stoploss
        
        # Adjust based on confidence (inverse relationship)
        if regime_conf > 0.9:
            base_stoploss *= 1.1  # Wider for over-confidence
        elif regime_conf < 0.7:
            base_stoploss *= 0.9  # Tighter for lower confidence
            
        # Adjust based on ATR
        if atr_percent > 0.015:  # High volatility
            base_stoploss *= 1.2
        elif atr_percent < 0.005:  # Low volatility
            base_stoploss *= 0.8
            
        # Adjust based on profit
        if current_profit > 0.01:  # In profit
            base_stoploss = max(base_stoploss, -0.015)  # Trail stop
        elif current_profit < -0.01:  # In loss
            base_stoploss *= 1.05  # Slightly wider
            
        return max(base_stoploss, -0.10)  # Cap at -10%

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Enhanced dynamic position sizing based on regime
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        regime_conf = last_candle.get('regime_conf', 0.0)
        
        # Base position size by regime
        if regime == 'high_volatility':
            base_multiplier = 0.8  # Moderate size for high vol
        elif regime == 'low_volatility':
            base_multiplier = 0.6  # Smaller for low vol (mean reversion)
        elif regime == 'trending':
            base_multiplier = 0.3  # Very small for trending (avoided)
        else:
            base_multiplier = 0.5
            
        # Adjust based on confidence (inverse relationship)
        if regime_conf > 0.9:
            base_multiplier *= 0.8  # Reduce for over-confidence
        elif regime_conf < 0.7:
            base_multiplier *= 1.1  # Increase for lower confidence
            
        # Adjust based on entry tag
        if entry_tag == 'HighVol_Entry':
            base_multiplier *= 1.0  # No extra boost
        elif entry_tag == 'LowVol_Entry':
            base_multiplier *= 0.8  # Smaller for mean reversion
            
        adjusted_stake = proposed_stake * base_multiplier
        
        # Ensure within bounds
        return max(min_stake, min(adjusted_stake, max_stake))

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Conservative leverage based on regime
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 1.0
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        
        # Conservative leverage by regime
        if regime == 'high_volatility':
            return min(1.3, max_leverage)
        elif regime == 'low_volatility':
            return min(1.1, max_leverage)
        elif regime == 'trending':
            return 1.0  # No leverage for trending
        else:
            return 1.0
