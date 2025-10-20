"""
Improved Trend Phase Strategy - Phase 1 Optimization
- Enhanced entry logic with better filtering
- Improved risk management
- Target: WinRate > 30%, Sharpe > 0.5, MDD < 2%
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


class ImprovedTrendPhaseStrategy(IStrategy):
    """
    Improved Trend Phase Strategy - Phase 1 Optimization
    - Enhanced entry logic with better filtering
    - Improved risk management
    - Target: WinRate > 30%, Sharpe > 0.5, MDD < 2%
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 1000
    can_short = False

    # More conservative ROI
    minimal_roi = {"0": 0.03, "45": 0.02, "90": 0.015, "180": 0.01}
    
    # Tighter stoploss
    stoploss = -0.05
    
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
        
        # Bollinger Bands for volatility
        bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df['bb_upper'] = bb['upperband']
        df['bb_middle'] = bb['middleband']
        df['bb_lower'] = bb['lowerband']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # MACD for momentum
        macd = ta.MACD(df)
        df['macd'] = macd['macd']
        df['macd_signal'] = macd['macdsignal']
        df['macd_hist'] = macd['macdhist']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change(5)
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1

        # Train HMM once sufficient candles
        if not self.trained and len(df) >= 1000:
            try:
                self.detector.train(df.tail(1000))
                self.trained = True
            except Exception as e:
                self.trained = False

        # Predict regimes/conf if trained
        if self.trained:
            regimes, conf = self.detector.predict_regime_sequence(df, smooth_window=5)
            # align
            if len(regimes) != len(df):
                n = min(len(regimes), len(df))
                regimes = regimes[-n:]
                conf = conf[-n:]
                df = df.tail(n).copy()
            df['regime'] = regimes
            df['regime_conf'] = conf
        else:
            df['regime'] = 'unknown'
            df['regime_conf'] = 0.0

        # Compute Trend Phase Score
        try:
            score = self.detector.compute_trend_phase_score(df)
            df['trend_phase_score'] = score
        except Exception as e:
            df['trend_phase_score'] = 0.0

        # Enhanced percentile-based labels with confidence gating
        score_series = df['trend_phase_score'].dropna()
        if not score_series.empty:
            p30 = score_series.quantile(0.30)
            p50 = score_series.quantile(0.50)
            p70 = score_series.quantile(0.70)
            p80 = score_series.quantile(0.80)
            
            df['phase_label'] = 'Neutral'
            # More selective thresholds
            df.loc[df['trend_phase_score'] >= p80, 'phase_label'] = 'Uptrend_Early'
            df.loc[(df['trend_phase_score'] >= p70) & (df['trend_phase_score'] < p80), 'phase_label'] = 'Uptrend_Late'
            df.loc[df['trend_phase_score'] <= p30, 'phase_label'] = 'Downtrend_Early'
            df.loc[(df['trend_phase_score'] > p30) & (df['trend_phase_score'] < p50), 'phase_label'] = 'Downtrend_Late'
        else:
            df['phase_label'] = 'Neutral'

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = False
        
        # Enhanced entry logic with multiple confirmations
        entry_mask = (
            # Core trend phase condition
            (df['phase_label'] == 'Uptrend_Early') &
            
            # Regime filtering with confidence
            (df['regime'].isin(['trending', 'high_volatility'])) &
            (df['regime_conf'] > 0.6) &
            
            # Technical confirmations
            (df['rsi'] > 40) &  # Not oversold
            (df['rsi'] < 75) &  # Not overbought
            (df['close'] > df['ema20']) &  # Above short EMA
            (df['ema20'] > df['ema50']) &  # EMA alignment
            (df['macd'] > df['macd_signal']) &  # MACD bullish
            (df['macd_hist'] > df['macd_hist'].shift(1)) &  # MACD improving
            
            # Volume confirmation
            (df['volume_ratio'] > 1.2) &  # Above average volume
            
            # Price momentum
            (df['price_change'] > 0) &  # Positive short-term momentum
            (df['price_momentum'] > -0.02) &  # Not in severe decline
            
            # Volatility check
            (df['bb_width'] > 0.02) &  # Sufficient volatility
            (df['bb_width'] < 0.08)  # Not extreme volatility
        )
        
        df['enter_long'] = entry_mask
        
        # Enter tags for analysis
        df['enter_tag'] = None
        df.loc[entry_mask & (df['regime'] == 'trending'), 'enter_tag'] = 'UpEarly_trending'
        df.loc[entry_mask & (df['regime'] == 'high_volatility'), 'enter_tag'] = 'UpEarly_highvol'
        
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = False
        
        # Enhanced exit logic
        exit_mask = (
            # Phase change exits
            (df['phase_label'].isin(['Uptrend_Late', 'Downtrend_Early', 'Downtrend_Late'])) |
            
            # Technical exits
            (df['rsi'] > 80) |  # Severely overbought
            (df['close'] < df['ema20']) |  # Below short EMA
            (df['ema20'] < df['ema50']) |  # EMA bearish cross
            (df['macd'] < df['macd_signal']) |  # MACD bearish
            (df['macd_hist'] < df['macd_hist'].shift(1)) |  # MACD deteriorating
            
            # Volume exits
            (df['volume_ratio'] < 0.8) |  # Low volume
            
            # Momentum exits
            (df['price_change'] < -0.01) |  # Negative momentum
            (df['price_momentum'] < -0.03) |  # Severe decline
            
            # Volatility exits
            (df['bb_width'] > 0.10) |  # Extreme volatility
            (df['regime'] == 'high_volatility') & (df['bb_width'] > 0.06)  # High vol + wide BB
        )
        
        df['exit_long'] = exit_mask
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Enhanced dynamic stoploss
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        regime_conf = last_candle.get('regime_conf', 0.0)
        atr_percent = last_candle.get('atr', 0) / current_rate
        
        # Base stoploss by regime
        if regime == 'trending':
            base_stoploss = -0.04  # Tighter for trending
        elif regime == 'low_volatility':
            base_stoploss = -0.06  # Wider for low vol
        elif regime == 'high_volatility':
            base_stoploss = -0.08  # Much wider for high vol
        else:
            base_stoploss = self.stoploss
        
        # Adjust based on confidence
        if regime_conf > 0.8:
            base_stoploss *= 0.9  # Tighter for high confidence
        elif regime_conf < 0.6:
            base_stoploss *= 1.1  # Wider for low confidence
            
        # Adjust based on ATR
        if atr_percent > 0.015:  # High volatility
            base_stoploss *= 1.2
        elif atr_percent < 0.005:  # Low volatility
            base_stoploss *= 0.95
            
        # Adjust based on profit
        if current_profit > 0.015:  # In profit
            base_stoploss = max(base_stoploss, -0.025)  # Trail stop
        elif current_profit < -0.015:  # In loss
            base_stoploss *= 1.05  # Slightly wider
            
        return max(base_stoploss, -0.15)  # Cap at -15%

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Enhanced dynamic position sizing
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        regime_conf = last_candle.get('regime_conf', 0.0)
        
        # Base position size by regime
        if regime == 'trending':
            base_multiplier = 0.8  # Reduced from 1.0
        elif regime == 'low_volatility':
            base_multiplier = 0.6  # Reduced from 0.8
        elif regime == 'high_volatility':
            base_multiplier = 0.4  # Much smaller for high vol
        else:
            base_multiplier = 0.5
            
        # Adjust based on confidence
        if regime_conf > 0.8:
            base_multiplier *= 1.1  # Increase for high confidence
        elif regime_conf < 0.6:
            base_multiplier *= 0.8  # Reduce for low confidence
            
        # Adjust based on entry tag
        if entry_tag == 'UpEarly_trending':
            base_multiplier *= 1.0  # No extra boost
        elif entry_tag == 'UpEarly_highvol':
            base_multiplier *= 0.8  # Smaller for high vol regime
            
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
        regime = last_candle.get('regime', 'unknown')
        
        # Conservative leverage by regime
        if regime == 'trending':
            return min(1.5, max_leverage)
        elif regime == 'low_volatility':
            return min(1.2, max_leverage)
        elif regime == 'high_volatility':
            return 1.0  # No leverage for high vol
        else:
            return 1.0
