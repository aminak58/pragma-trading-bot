"""
Regime-Adaptive Trading Strategy

This strategy uses HMM-based regime detection to adapt trading logic
to current market conditions. It combines:
- HMM regime detection (low_volatility, high_volatility, trending)
- Multi-timeframe technical analysis
- Regime-specific entry/exit rules
- Dynamic risk management

Strategy Logic:
- Trending regime: Follow momentum with trend indicators
- Low volatility: Mean reversion with Bollinger Bands
- High volatility: Reduce position size, tighter stops
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import logging
from typing import Optional
import pandas as pd
import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import talib.abstract as ta

from regime.hmm_detector import EnhancedRegimeDetector

logger = logging.getLogger(__name__)


class RegimeAdaptiveStrategy(IStrategy):
    """
    Regime-adaptive strategy using HMM for market classification.
    
    Adapts entry/exit logic and risk management based on detected regime:
    - Trending: Momentum following
    - Low Volatility: Mean reversion
    - High Volatility: Conservative approach
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # ROI table (regime-adaptive in practice)
    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.03,
        "120": 0.01
    }
    
    # Stoploss
    stoploss = -0.05
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'
    
    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True
    
    # Experimental settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Hyperopt parameters
    # Trending regime
    buy_adx_trending = IntParameter(20, 40, default=25, space='buy')
    buy_ema_short_trending = IntParameter(8, 20, default=12, space='buy')
    buy_ema_long_trending = IntParameter(20, 50, default=26, space='buy')
    
    # Low volatility regime
    buy_bb_lower_offset = DecimalParameter(
        0.95, 0.99, default=0.98, decimals=3, space='buy'
    )
    buy_rsi_low_vol = IntParameter(20, 40, default=30, space='buy')
    
    # High volatility regime
    buy_vol_threshold = DecimalParameter(
        0.01, 0.05, default=0.03, decimals=3, space='buy'
    )
    
    # Exit parameters
    sell_adx = IntParameter(20, 50, default=30, space='sell')
    sell_rsi = IntParameter(60, 85, default=70, space='sell')
    
    # Regime detection parameters (Best practice: 3000-10000 for 5m)
    regime_training_lookback = IntParameter(
        3000, 10000, default=5000, space='buy'
    )
    regime_confidence_threshold = DecimalParameter(
        0.5, 0.9, default=0.7, decimals=2, space='buy'
    )
    
    def __init__(self, config: dict) -> None:
        """Initialize strategy with HMM regime detector."""
        super().__init__(config)
        
        # Initialize Enhanced HMM regime detector
        self.regime_detector = EnhancedRegimeDetector(n_states=3, random_state=42)
        self.regime_trained = False
        
        # Performance tracking
        self.regime_stats = {
            'trending': {'count': 0, 'trades': 0},
            'low_volatility': {'count': 0, 'trades': 0},
            'high_volatility': {'count': 0, 'trades': 0}
        }
        
        logger.info("RegimeAdaptiveStrategy initialized with HMM detector")
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add technical indicators and regime detection to dataframe.
        
        Args:
            dataframe: OHLCV dataframe
            metadata: Pair metadata
            
        Returns:
            Dataframe with indicators and regime columns
        """
        # Train HMM if enough data
        if len(dataframe) >= 100 and not self.regime_trained:
            try:
                lookback = min(len(dataframe), self.regime_training_lookback.value)
                self.regime_detector.train(dataframe, lookback=lookback)
                self.regime_trained = True
                logger.info(f"HMM trained for {metadata['pair']} with {lookback} candles")
            except Exception as e:
                logger.warning(f"HMM training failed for {metadata['pair']}: {e}")
        
        # Add regime prediction if trained
        if self.regime_trained:
            try:
                regime, confidence = self.regime_detector.predict_regime(dataframe)
                dataframe['regime'] = regime
                dataframe['regime_confidence'] = confidence
                
                # Get full probability distribution
                probs = self.regime_detector.get_regime_probabilities(dataframe)
                for regime_name, prob in probs.items():
                    dataframe[f'regime_prob_{regime_name}'] = prob
                
                # Track regime occurrences
                if regime in self.regime_stats:
                    self.regime_stats[regime]['count'] += 1
                    
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}")
                dataframe['regime'] = 'unknown'
                dataframe['regime_confidence'] = 0.0
        else:
            dataframe['regime'] = 'unknown'
            dataframe['regime_confidence'] = 0.0
        
        # === Technical Indicators ===
        
        # EMA
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=26)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(
            dataframe, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_width'] = (
            (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        )
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry signals based on regime.
        
        Args:
            dataframe: Dataframe with indicators
            metadata: Pair metadata
            
        Returns:
            Dataframe with entry_long column
        """
        conditions_trending = []
        conditions_low_vol = []
        conditions_high_vol = []
        
        # === Trending Regime: Momentum following ===
        conditions_trending.append(
            (dataframe['regime'] == 'trending') &
            (dataframe['regime_confidence'] >= self.regime_confidence_threshold.value)
        )
        conditions_trending.append(dataframe['volume'] > dataframe['volume_mean'])
        conditions_trending.append(
            dataframe['adx'] > self.buy_adx_trending.value
        )
        conditions_trending.append(
            dataframe['ema_short'] > dataframe['ema_long']
        )
        conditions_trending.append(dataframe['close'] > dataframe['ema_short'])
        conditions_trending.append(dataframe['rsi'] < 70)
        
        # === Low Volatility Regime: Mean reversion ===
        conditions_low_vol.append(
            (dataframe['regime'] == 'low_volatility') &
            (dataframe['regime_confidence'] >= self.regime_confidence_threshold.value)
        )
        conditions_low_vol.append(
            dataframe['close'] < (
                dataframe['bb_lower'] * (1 + self.buy_bb_lower_offset.value - 1)
            )
        )
        conditions_low_vol.append(
            dataframe['rsi'] < self.buy_rsi_low_vol.value
        )
        conditions_low_vol.append(dataframe['volume'] > 0)
        
        # === High Volatility Regime: Conservative ===
        conditions_high_vol.append(
            (dataframe['regime'] == 'high_volatility') &
            (dataframe['regime_confidence'] >= self.regime_confidence_threshold.value)
        )
        conditions_high_vol.append(
            dataframe['atr_percent'] < self.buy_vol_threshold.value
        )
        conditions_high_vol.append(dataframe['adx'] > 25)
        conditions_high_vol.append(dataframe['rsi'] > 40)
        conditions_high_vol.append(dataframe['rsi'] < 60)
        conditions_high_vol.append(dataframe['volume'] > dataframe['volume_mean'] * 1.5)
        
        # Combine conditions
        if conditions_trending:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_trending),
                'enter_long'
            ] = 1
        
        if conditions_low_vol:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_low_vol),
                'enter_long'
            ] = 1
        
        if conditions_high_vol:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_high_vol),
                'enter_long'
            ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define exit signals based on regime.
        
        Args:
            dataframe: Dataframe with indicators
            metadata: Pair metadata
            
        Returns:
            Dataframe with exit_long column
        """
        conditions = []
        
        # Exit on regime change to high volatility
        conditions.append(
            (dataframe['regime'] == 'high_volatility') &
            (dataframe['regime_confidence'] > 0.8)
        )
        
        # Exit on overbought RSI
        conditions.append(dataframe['rsi'] > self.sell_rsi.value)
        
        # Exit on ADX decline (trend weakening)
        conditions.append(
            (dataframe['adx'] < self.sell_adx.value) &
            (dataframe['regime'] == 'trending')
        )
        
        # Exit on price above upper BB (mean reversion)
        conditions.append(
            (dataframe['close'] > dataframe['bb_upper']) &
            (dataframe['regime'] == 'low_volatility')
        )
        
        # MACD bearish crossover
        conditions.append(
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macdhist'] < 0)
        )
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        
        return dataframe
    
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        Custom stoploss adapted to regime.
        
        - Trending: Wider stops (let profits run)
        - Low volatility: Standard stops
        - High volatility: Tighter stops
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        
        # Regime-adaptive stoploss
        if regime == 'trending':
            # Wider stop for trends
            return -0.08
        elif regime == 'low_volatility':
            # Standard stop
            return -0.05
        elif regime == 'high_volatility':
            # Tighter stop for volatile markets
            return -0.03
        else:
            return self.stoploss
    
    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> bool:
        """
        Confirm trade entry with final regime check.
        
        Prevents entries in unfavorable regimes.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return False
        
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        confidence = last_candle.get('regime_confidence', 0.0)
        
        # Block trades in unknown regime
        if regime == 'unknown':
            logger.info(f"Blocking {pair} entry: unknown regime")
            return False
        
        # Require minimum confidence
        if confidence < 0.6:
            logger.info(
                f"Blocking {pair} entry: low confidence ({confidence:.2f})"
            )
            return False
        
        # Log regime for tracking
        logger.info(
            f"Confirming {pair} entry in {regime} regime "
            f"(confidence: {confidence:.2%})"
        )
        
        if regime in self.regime_stats:
            self.regime_stats[regime]['trades'] += 1
        
        return True
    
    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Regime-adaptive leverage.
        
        Reduces leverage in high volatility regimes.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return 1.0
        
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        
        # Regime-adaptive leverage
        if regime == 'high_volatility':
            return min(proposed_leverage, 1.0)  # No leverage in volatile markets
        elif regime == 'low_volatility':
            return min(proposed_leverage, 2.0)  # Moderate leverage
        elif regime == 'trending':
            return min(proposed_leverage, 3.0)  # Higher leverage for trends
        else:
            return 1.0
    
    def __del__(self):
        """Log regime statistics on strategy destruction."""
        if hasattr(self, 'regime_stats'):
            logger.info("=" * 60)
            logger.info("Regime Statistics Summary")
            logger.info("=" * 60)
            for regime, stats in self.regime_stats.items():
                logger.info(
                    f"{regime:20s}: {stats['count']:5d} occurrences, "
                    f"{stats['trades']:4d} trades"
                )
            logger.info("=" * 60)


def reduce(function, iterable, initializer=None):
    """Python 3 reduce implementation."""
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
