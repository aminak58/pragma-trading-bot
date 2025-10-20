"""
Optimized Trend Phase Strategy - Phase 1 Implementation
- Removes extra filters from Scenario B
- Improves Risk Management
- Uses HMM v2.0 + Trend Phase Score
- Target: WinRate > 30%, NumTrades > 15/week, Sharpe > 0.5
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


class OptimizedTrendPhaseStrategy(IStrategy):
    """
    Optimized Trend Phase Strategy - Phase 1
    - Simplified entry logic with minimal filters
    - Enhanced risk management
    - Uses HMM v2.0 + Trend Phase Score
    """
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 1000
    can_short = False

    # Optimized ROI - more aggressive for better exits
    minimal_roi = {"0": 0.05, "30": 0.025, "60": 0.015, "120": 0.01}
    
    # Dynamic stoploss based on regime
    stoploss = -0.08  # Wider initial stoploss
    
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

        # Train HMM once sufficient candles
        if not self.trained and len(df) >= 1000:
            try:
                self.detector.train(df.tail(1000))
                self.trained = True
                # self.logger.info("HMM trained successfully")
            except Exception as e:
                # self.logger.warning(f'HMM train skipped: {e}')
                self.trained = False

        # Predict regimes/conf if trained
        if self.trained:
            regimes, conf = self.detector.predict_regime_sequence(df, smooth_window=3)
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
            # self.logger.warning(f'Trend Phase Score computation failed: {e}')
            df['trend_phase_score'] = 0.0

        # Simple percentile-based labels (no hysteresis/gating)
        score_series = df['trend_phase_score'].dropna()
        if not score_series.empty:
            p25 = score_series.quantile(0.25)
            p40 = score_series.quantile(0.40)
            p60 = score_series.quantile(0.60)
            p75 = score_series.quantile(0.75)
            
            df['phase_label'] = 'Neutral'
            df.loc[df['trend_phase_score'] >= p75, 'phase_label'] = 'Uptrend_Early'
            df.loc[(df['trend_phase_score'] >= p60) & (df['trend_phase_score'] < p75), 'phase_label'] = 'Uptrend_Late'
            df.loc[df['trend_phase_score'] <= p25, 'phase_label'] = 'Downtrend_Early'
            df.loc[(df['trend_phase_score'] > p25) & (df['trend_phase_score'] < p40), 'phase_label'] = 'Downtrend_Late'
        else:
            df['phase_label'] = 'Neutral'

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = False
        
        # Simplified entry logic - minimal filters
        entry_mask = (
            (df['phase_label'] == 'Uptrend_Early') &
            (df['regime'].isin(['trending', 'high_volatility'])) &
            (df['rsi'] > 30) &  # Not oversold
            (df['rsi'] < 80) &  # Not overbought
            (df['close'] > df['ema20']) &  # Above short EMA
            (df['macd'] > df['macd_signal'])  # MACD bullish
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
            (df['phase_label'].isin(['Uptrend_Late', 'Downtrend_Early', 'Downtrend_Late'])) |
            (df['rsi'] > 85) |  # Severely overbought
            (df['close'] < df['ema20']) |  # Below short EMA
            (df['macd'] < df['macd_signal']) |  # MACD bearish
            (df['regime'] == 'high_volatility') & (df['bb_width'] > 0.08)  # Extreme volatility
        )
        
        df['exit_long'] = exit_mask
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stoploss based on regime and market conditions
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
            base_stoploss = -0.06  # Tighter for trending
        elif regime == 'low_volatility':
            base_stoploss = -0.08  # Wider for low vol
        elif regime == 'high_volatility':
            base_stoploss = -0.12  # Much wider for high vol
        else:
            base_stoploss = self.stoploss
        
        # Adjust based on confidence
        if regime_conf > 0.8:
            base_stoploss *= 0.8  # Tighter for high confidence
        elif regime_conf < 0.5:
            base_stoploss *= 1.2  # Wider for low confidence
            
        # Adjust based on ATR
        if atr_percent > 0.02:  # High volatility
            base_stoploss *= 1.3
        elif atr_percent < 0.005:  # Low volatility
            base_stoploss *= 0.9
            
        # Adjust based on profit
        if current_profit > 0.02:  # In profit
            base_stoploss = max(base_stoploss, -0.03)  # Trail stop
        elif current_profit < -0.02:  # In loss
            base_stoploss *= 1.1  # Slightly wider
            
        return max(base_stoploss, -0.20)  # Cap at -20%

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Dynamic position sizing based on regime and confidence
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        regime_conf = last_candle.get('regime_conf', 0.0)
        
        # Base position size by regime
        if regime == 'trending':
            base_multiplier = 1.0  # Full size for trending
        elif regime == 'low_volatility':
            base_multiplier = 0.8  # Reduced for low vol
        elif regime == 'high_volatility':
            base_multiplier = 0.6  # Much smaller for high vol
        else:
            base_multiplier = 0.7
            
        # Adjust based on confidence
        if regime_conf > 0.8:
            base_multiplier *= 1.2  # Increase for high confidence
        elif regime_conf < 0.5:
            base_multiplier *= 0.8  # Reduce for low confidence
            
        # Adjust based on entry tag
        if entry_tag == 'UpEarly_trending':
            base_multiplier *= 1.1  # Slightly larger for trending regime
        elif entry_tag == 'UpEarly_highvol':
            base_multiplier *= 0.9  # Slightly smaller for high vol regime
            
        adjusted_stake = proposed_stake * base_multiplier
        
        # Ensure within bounds
        return max(min_stake, min(adjusted_stake, max_stake))

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Dynamic leverage based on regime
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return 1.0
            
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        
        # Leverage by regime
        if regime == 'trending':
            return min(2.0, max_leverage)
        elif regime == 'low_volatility':
            return min(1.5, max_leverage)
        elif regime == 'high_volatility':
            return 1.0  # No leverage for high vol
        else:
            return 1.0
