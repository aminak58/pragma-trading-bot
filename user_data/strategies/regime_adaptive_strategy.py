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
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import logging
from typing import Optional
import pandas as pd
import numpy as np
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
import talib.abstract as ta

from src.regime.hmm_detector import RegimeDetector

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
    
    # Regime detection parameters
    regime_training_lookback = IntParameter(
        300, 700, default=500, space='buy'
    )
    regime_confidence_threshold = DecimalParameter(
        0.5, 0.9, default=0.3, decimals=2, space='buy'
    )
    
    # Dynamic parameters based on regime
    def get_dynamic_stoploss(self, regime: str, confidence: float) -> float:
        """Get dynamic stoploss based on regime and confidence"""
        base_stoploss = -0.05
        
        if regime == 'trending':
            # Tighter stoploss for trending (momentum following)
            return base_stoploss * 0.8  # -0.04
        elif regime == 'low_volatility':
            # Wider stoploss for mean reversion
            return base_stoploss * 1.2  # -0.06
        elif regime == 'high_volatility':
            # Much tighter stoploss for high volatility
            return base_stoploss * 0.5  # -0.025
        
        return base_stoploss
    
    def get_dynamic_position_size(self, regime: str, confidence: float, base_amount: float) -> float:
        """Get dynamic position size based on regime and confidence (relaxed)"""
        if regime == 'trending' and confidence > 0.6:  # Lowered from 0.8
            return base_amount * 1.1  # Slightly increased size
        elif regime == 'low_volatility' and confidence > 0.5:  # Lowered from 0.7
            return base_amount * 0.9  # Less reduction
        elif regime == 'high_volatility' and confidence > 0.6:  # Added confidence check
            return base_amount * 0.7  # Less reduction
        
        return base_amount
    
    def get_dynamic_roi(self, regime: str, confidence: float) -> dict:
        """Get dynamic ROI based on regime"""
        if regime == 'trending':
            return {
                "0": 0.12,  # Higher target for trending
                "30": 0.06,
                "60": 0.04,
                "120": 0.02
            }
        elif regime == 'low_volatility':
            return {
                "0": 0.08,  # Lower target for mean reversion
                "30": 0.04,
                "60": 0.02,
                "120": 0.01
            }
        elif regime == 'high_volatility':
            return {
                "0": 0.15,  # Higher target for high volatility
                "30": 0.08,
                "60": 0.05,
                "120": 0.02
            }
        
        return self.minimal_roi
    
    def __init__(self, config: dict) -> None:
        """Initialize strategy with HMM regime detector."""
        super().__init__(config)
        
        # Initialize HMM regime detector
        self.regime_detector = RegimeDetector(n_states=3, random_state=42)
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
        # Train HMM if enough data (minimum 1000 candles required)
        if len(dataframe) >= 1000 and not self.regime_trained:
            try:
                lookback = min(len(dataframe), self.regime_training_lookback.value)
                self.regime_detector.train(dataframe, lookback=lookback)
                self.regime_trained = True
                logger.info(f"HMM trained for {metadata['pair']} with {lookback} candles")
            except Exception as e:
                logger.warning(f"HMM training failed for {metadata['pair']}: {e}")
                # Set default regime if training fails
                dataframe['regime'] = 'unknown'
                dataframe['regime_confidence'] = 0.0
                return dataframe
        
        # Add regime prediction if trained
        if self.regime_trained:
            try:
                    # Use sequence prediction with smoothing for better stability
                    regime_sequence, confidence_sequence = self.regime_detector.predict_regime_sequence(
                        dataframe, smooth_window=1  # Minimal smoothing to avoid over-confidence
                    )
                    
                    # Ensure length matches dataframe
                    if len(regime_sequence) != len(dataframe):
                        logger.warning(f"Length mismatch: regime_sequence={len(regime_sequence)}, dataframe={len(dataframe)}")
                        # Pad or truncate to match dataframe length
                        if len(regime_sequence) < len(dataframe):
                            # Pad with last value
                            regime_sequence = np.concatenate([
                                regime_sequence, 
                                np.full(len(dataframe) - len(regime_sequence), regime_sequence[-1])
                            ])
                            confidence_sequence = np.concatenate([
                                confidence_sequence,
                                np.full(len(dataframe) - len(confidence_sequence), confidence_sequence[-1])
                            ])
                        else:
                            # Truncate to match
                            regime_sequence = regime_sequence[:len(dataframe)]
                            confidence_sequence = confidence_sequence[:len(dataframe)]
                    
                    dataframe['regime'] = regime_sequence
                    dataframe['regime_confidence'] = confidence_sequence
                    
                    # Get full probability distribution for last candle
                    probs = self.regime_detector.get_regime_probabilities(dataframe)
                    for regime_name, prob in probs.items():
                        dataframe[f'regime_prob_{regime_name}'] = prob
                    
                    # Track regime occurrences (use last regime)
                    last_regime = regime_sequence[-1] if len(regime_sequence) > 0 else 'unknown'
                    if last_regime in self.regime_stats:
                        self.regime_stats[last_regime]['count'] += 1
                    
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
            dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0
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
        Define enhanced entry signals based on regime with better signal quality.
        
        Args:
            dataframe: Dataframe with indicators
            metadata: Pair metadata
            
        Returns:
            Dataframe with entry_long column
        """
        # Initialize entry conditions
        dataframe['entry_long'] = False
        
        # Regime-specific conditions with enhanced logic
        conditions_trending = []
        conditions_low_vol = []
        conditions_high_vol = []
        
        # Ultra-relaxed trending regime conditions (Tier 1: Essential only)
        conditions_trending.extend([
            # Essential trend confirmation (2 conditions only)
            (dataframe['adx'] > 15),  # Very relaxed trend strength
            (dataframe['ema_short'] > dataframe['ema_long']),  # Uptrend
        ])
        
        # Ultra-relaxed low volatility regime conditions (Tier 1: Essential only)
        conditions_low_vol.extend([
            # Essential mean reversion setup (2 conditions only)
            (dataframe['rsi'] < 45),  # Very relaxed oversold
            (dataframe['close'] < dataframe['bb_lower'] * 1.05),  # Near lower BB
        ])
        
        # Ultra-relaxed high volatility regime conditions (Tier 1: Essential only)
        conditions_high_vol.extend([
            # Essential breakout setup (2 conditions only)
            (dataframe['close'] > dataframe['bb_upper'] * 0.95),  # Breaking upper BB
            (dataframe['bb_width'] > 0.04),  # Very relaxed high volatility
        ])
        
        # === Enhanced Regime-Specific Entry Logic ===
        
        # Trending Regime: Momentum following with confirmation
        trending_entry = (
            (dataframe['regime'] == 'trending') &
            (dataframe['regime_confidence'] >= self.regime_confidence_threshold.value) &
            # All technical conditions must be met
            (dataframe['adx'] > 25) & (dataframe['adx'] < 50) &  # Strong but not extreme trend
            (dataframe['ema_short'] > dataframe['ema_long']) &  # Uptrend
            (dataframe['close'] > dataframe['ema_short']) &  # Price above short EMA
            (dataframe['macd'] > dataframe['macdsignal']) &  # MACD bullish
            (dataframe['macdhist'] > 0) &  # MACD histogram positive
            (dataframe['rsi'] > 45) & (dataframe['rsi'] < 75) &  # RSI in healthy range
            (dataframe['bb_width'] > 0.02) &  # Sufficient volatility
            (dataframe['volume'] > dataframe['volume_mean'] * 1.2)  # Above average volume
        )
        
        # Low Volatility Regime: Mean reversion with confirmation
        low_vol_entry = (
            (dataframe['regime'] == 'low_volatility') &
            (dataframe['regime_confidence'] >= self.regime_confidence_threshold.value) &
            # Mean reversion setup
            (dataframe['rsi'] < 35) &  # Oversold
            (dataframe['close'] < dataframe['bb_lower'] * 1.01) &  # Near lower BB
            (dataframe['bb_width'] < 0.05) &  # Low volatility
            (dataframe['adx'] < 20) &  # Weak trend
            (dataframe['macdhist'] > dataframe['macdhist'].shift(1)) &  # MACD improving
            (dataframe['volume'] > dataframe['volume_mean'] * 0.8)  # Decent volume
        )
        
        # High Volatility Regime: Breakout with confirmation
        high_vol_entry = (
            (dataframe['regime'] == 'high_volatility') &
            (dataframe['regime_confidence'] >= self.regime_confidence_threshold.value) &
            # Breakout setup
            (dataframe['close'] > dataframe['bb_upper'] * 0.99) &  # Breaking upper BB
            (dataframe['bb_width'] > 0.08) &  # High volatility
            (dataframe['volume'] > dataframe['volume_mean'] * 1.5) &  # High volume
            (dataframe['rsi'] > 55) & (dataframe['rsi'] < 80) &  # Strong but not overbought
            (dataframe['adx'] > 30) &  # Strong trend
            (dataframe['macd'] > dataframe['macdsignal'])  # MACD bullish
        )
        
        # Apply entry signals with regime-specific logic
        dataframe.loc[trending_entry, 'enter_long'] = 1
        dataframe.loc[low_vol_entry, 'enter_long'] = 1
        dataframe.loc[high_vol_entry, 'enter_long'] = 1
        
        # Minimal signal confirmation (only essential checks)
        # 1. Basic volume check (very relaxed)
        dataframe['volume_confirmation'] = (
            dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * 0.8
        ).astype(bool)
        
        # Final entry signal with minimal confirmations
        dataframe['entry_long'] = (
            (dataframe['enter_long'] == 1) &
            dataframe['volume_confirmation']
            # All other confirmations removed for maximum trade frequency
        )
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define enhanced exit signals based on regime with better timing.
        
        Args:
            dataframe: Dataframe with indicators
            metadata: Pair metadata
            
        Returns:
            Dataframe with exit_long column
        """
        # Initialize exit conditions
        dataframe['exit_long'] = False
        
        # === Regime-Specific Exit Logic ===
        
        # 1. Trending Regime Exits (momentum weakening)
        trending_exit = (
            (dataframe['regime'] == 'trending') &
            (
                # Trend weakening signals
                (dataframe['adx'] < 20) |  # Trend strength declining
                (dataframe['macd'] < dataframe['macdsignal']) |  # MACD bearish
                (dataframe['macdhist'] < dataframe['macdhist'].shift(1)) |  # MACD histogram declining
                (dataframe['close'] < dataframe['ema_short']) |  # Price below short EMA
                (dataframe['rsi'] > 75)  # Overbought
            )
        )
        
        # 2. Low Volatility Regime Exits (mean reversion target reached)
        low_vol_exit = (
            (dataframe['regime'] == 'low_volatility') &
            (
                # Mean reversion target reached
                (dataframe['close'] > dataframe['bb_upper'] * 0.99) |  # Near upper BB
                (dataframe['rsi'] > 65) |  # RSI approaching overbought
                (dataframe['macdhist'] < 0)  # MACD turning negative
            )
        )
        
        # 3. High Volatility Regime Exits (risk management)
        high_vol_exit = (
            (dataframe['regime'] == 'high_volatility') &
            (
                # Risk management exits
                (dataframe['atr_percent'] > 0.1) |  # Extreme volatility
                (dataframe['rsi'] > 80) |  # Severely overbought
                (dataframe['bb_width'] > 0.15)  # Extreme volatility
            )
        )
        
        # 4. Universal Exit Conditions (regime-independent)
        universal_exit = (
            # Regime change to unfavorable conditions
            (dataframe['regime'] == 'high_volatility') & (dataframe['regime_confidence'] > 0.8) |
            # Technical overbought
            (dataframe['rsi'] > 80) |
            # Volume drying up
            (dataframe['volume'] < dataframe['volume_mean'] * 0.5) |
            # Price action deterioration
            (dataframe['close'] < dataframe['close'].rolling(window=5).min()) |
            # MACD divergence
            (dataframe['macd'] < dataframe['macdsignal']) & (dataframe['macdhist'] < 0)
        )
        
        # Apply exit signals
        dataframe.loc[trending_exit, 'exit_long'] = 1
        dataframe.loc[low_vol_exit, 'exit_long'] = 1
        dataframe.loc[high_vol_exit, 'exit_long'] = 1
        dataframe.loc[universal_exit, 'exit_long'] = 1
        
        # Exit confirmation mechanisms
        # 1. Consecutive exit signals
        dataframe['exit_confirmation'] = (
            dataframe['exit_long'].rolling(window=2).sum() >= 1
        ).astype(bool)
        
        # 2. Profit target reached
        dataframe['profit_target'] = (
            dataframe['close'] > dataframe['close'].shift(1) * 1.02  # 2% profit
        ).astype(bool)
        
        # 3. Time-based exit (avoid holding too long) - simplified
        dataframe['time_exit'] = False  # Disable for now to avoid complexity
        
        # Final exit signal
        dataframe['exit_long'] = (
            (dataframe['exit_long'] == 1) &
            (dataframe['exit_confirmation'] | dataframe['profit_target'] | dataframe['time_exit'])
        )
        
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
        Enhanced custom stoploss adapted to regime and confidence.
        
        - Trending: Dynamic stops based on confidence
        - Low volatility: Wider stops for mean reversion
        - High volatility: Tighter stops for risk management
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        confidence = last_candle.get('regime_confidence', 0.5)
        
        # Get dynamic stoploss based on regime and confidence
        dynamic_stoploss = self.get_dynamic_stoploss(regime, confidence)
        
        # Additional adjustments based on trade performance
        if current_profit > 0.02:  # If already 2% profit
            # Tighten stoploss to lock in profits
            dynamic_stoploss = max(dynamic_stoploss, -0.02)
        elif current_profit < -0.01:  # If losing 1%
            # Tighten stoploss to limit losses
            dynamic_stoploss = max(dynamic_stoploss, -0.03)
        
        # ATR-based adjustment
        atr_percent = last_candle.get('atr_percent', 2.0)
        if atr_percent > 3.0:  # High volatility
            dynamic_stoploss = max(dynamic_stoploss, -0.02)  # Tighter stop
        elif atr_percent < 1.0:  # Low volatility
            dynamic_stoploss = min(dynamic_stoploss, -0.08)  # Wider stop
        
        return dynamic_stoploss
    
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
        Enhanced regime-adaptive leverage with confidence-based adjustments.
        
        Adjusts leverage based on regime and confidence level.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return 1.0
        
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        confidence = last_candle.get('regime_confidence', 0.5)
        
        # Base leverage by regime
        if regime == 'trending':
            base_leverage = 2.0  # Higher leverage for trends
        elif regime == 'low_volatility':
            base_leverage = 1.5  # Moderate leverage
        elif regime == 'high_volatility':
            base_leverage = 1.0  # Conservative leverage
        else:
            return 1.0
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        adjusted_leverage = base_leverage * confidence_multiplier
        
        # Additional risk management
        atr_percent = last_candle.get('atr_percent', 2.0)
        if atr_percent > 4.0:  # Extreme volatility
            adjusted_leverage *= 0.5  # Reduce leverage by half
        elif atr_percent < 1.0:  # Very low volatility
            adjusted_leverage *= 1.2  # Slightly increase leverage
        
        return min(adjusted_leverage, max_leverage)
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        Dynamic position sizing based on regime and confidence.
        
        Adjusts position size based on regime characteristics and confidence level.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return proposed_stake
        
        last_candle = dataframe.iloc[-1]
        regime = last_candle.get('regime', 'unknown')
        confidence = last_candle.get('regime_confidence', 0.5)
        
        # Get dynamic position size
        dynamic_stake = self.get_dynamic_position_size(regime, confidence, proposed_stake)
        
        # Additional adjustments
        atr_percent = last_candle.get('atr_percent', 2.0)
        if atr_percent > 3.0:  # High volatility
            dynamic_stake *= 0.7  # Reduce position size
        elif atr_percent < 1.0:  # Low volatility
            dynamic_stake *= 1.1  # Slightly increase position size
        
        # Ensure within bounds
        return max(min_stake, min(dynamic_stake, max_stake))
    
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
