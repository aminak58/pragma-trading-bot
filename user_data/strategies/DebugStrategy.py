"""
Debug Strategy for Deep Analysis
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
from freqtrade.strategy import IStrategy
import talib.abstract as ta

from src.regime.hmm_detector import RegimeDetector

logger = logging.getLogger(__name__)


class DebugStrategy(IStrategy):
    """
    Debug strategy to analyze HMM behavior and entry conditions
    """
    
    INTERFACE_VERSION = 3
    can_short: bool = False
    timeframe = '5m'
    
    minimal_roi = {"0": 0.1}
    stoploss = -0.05
    trailing_stop = False
    process_only_new_candles = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 50
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.regime_detector = RegimeDetector(n_states=3, random_state=42)
        self.regime_trained = False
        self.debug_data = []
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add indicators and debug HMM behavior"""
        
        # Train HMM if enough data (minimum 1000 candles required)
        if len(dataframe) >= 1000 and not self.regime_trained:
            try:
                lookback = min(len(dataframe), 1000)
                self.regime_detector.train(dataframe, lookback=lookback)
                self.regime_trained = True
                logger.info(f"HMM trained with {lookback} candles")
            except Exception as e:
                logger.warning(f"HMM training failed: {e}")
                # Set default regime if training fails
                dataframe['regime'] = 'unknown'
                dataframe['regime_confidence'] = 0.0
        
        # Add regime prediction
        if self.regime_trained:
            try:
                regime, confidence = self.regime_detector.predict_regime(dataframe)
                dataframe['regime'] = regime
                dataframe['regime_confidence'] = confidence
                
                # Get probabilities
                probs = self.regime_detector.get_regime_probabilities(dataframe)
                for regime_name, prob in probs.items():
                    dataframe[f'regime_prob_{regime_name}'] = prob
                    
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}")
                dataframe['regime'] = 'unknown'
                dataframe['regime_confidence'] = 0.0
        else:
            dataframe['regime'] = 'unknown'
            dataframe['regime_confidence'] = 0.0
        
        # Technical indicators
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']
        
        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        
        # Debug: Log last few rows
        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]
            debug_info = {
                'timestamp': last_row.get('date', 'unknown'),
                'close': last_row.get('close', 0),
                'regime': last_row.get('regime', 'unknown'),
                'confidence': last_row.get('regime_confidence', 0),
                'rsi': last_row.get('rsi', 0),
                'adx': last_row.get('adx', 0),
                'atr_percent': last_row.get('atr_percent', 0),
                'volume_ratio': last_row.get('volume', 0) / last_row.get('volume_mean', 1) if last_row.get('volume_mean', 0) > 0 else 0
            }
            self.debug_data.append(debug_info)
            
            # Log every 50 candles
            if len(self.debug_data) % 50 == 0:
                logger.info(f"Debug data point {len(self.debug_data)}: {debug_info}")
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Debug entry conditions"""
        
        # Initialize enter_long column
        dataframe['enter_long'] = 0
        
        # Simple entry condition for testing
        entry_condition = (
            (dataframe['regime'] != 'unknown') &
            (dataframe['regime_confidence'] > 0.5) &
            (dataframe['rsi'] < 70) &
            (dataframe['rsi'] > 30) &
            (dataframe['close'] > dataframe['ema_short'])
        )
        dataframe.loc[entry_condition, 'enter_long'] = 1
        
        # Log entry conditions
        entry_signals = dataframe[dataframe['enter_long'] == 1]
        if len(entry_signals) > 0:
            logger.info(f"Found {len(entry_signals)} entry signals")
            for idx, row in entry_signals.iterrows():
                logger.info(f"Entry signal at {row.get('date', 'unknown')}: "
                          f"regime={row.get('regime')}, confidence={row.get('regime_confidence', 0):.2f}, "
                          f"rsi={row.get('rsi', 0):.2f}")
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Simple exit condition"""
        # Initialize exit_long column
        dataframe['exit_long'] = 0
        
        exit_condition = (
            (dataframe['rsi'] > 80) |
            (dataframe['close'] < dataframe['ema_short'])
        )
        dataframe.loc[exit_condition, 'exit_long'] = 1
        
        return dataframe
    
    def __del__(self):
        """Log final debug summary"""
        if hasattr(self, 'debug_data') and self.debug_data:
            logger.info("=" * 60)
            logger.info("DEBUG SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total data points: {len(self.debug_data)}")
            
            # Regime distribution
            regimes = [d['regime'] for d in self.debug_data]
            regime_counts = pd.Series(regimes).value_counts()
            logger.info(f"Regime distribution: {dict(regime_counts)}")
            
            # Confidence stats
            confidences = [d['confidence'] for d in self.debug_data if d['confidence'] > 0]
            if confidences:
                logger.info(f"Confidence stats: min={min(confidences):.2f}, max={max(confidences):.2f}, avg={np.mean(confidences):.2f}")
            
            # RSI stats
            rsi_values = [d['rsi'] for d in self.debug_data if d['rsi'] > 0]
            if rsi_values:
                logger.info(f"RSI stats: min={min(rsi_values):.2f}, max={max(rsi_values):.2f}, avg={np.mean(rsi_values):.2f}")
            
            logger.info("=" * 60)
