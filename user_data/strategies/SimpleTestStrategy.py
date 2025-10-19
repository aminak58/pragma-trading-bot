"""
Simple Test Strategy for Docker Testing
"""

from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np


class SimpleTestStrategy(IStrategy):
    """
    Simple test strategy that just buys and holds
    """
    
    INTERFACE_VERSION = 3
    
    # Strategy interface version - allow new iterations of the strategy IStrategy
    # to be compatible. (1 = No interface, 2 = IStrategy, 3 = IStrategy + (minimal) FreqaiInterface)
    # Check the documentation or the IStrategy class to see the latest tests (tests = IStrategy + FreqaiInterface)
    can_short: bool = False
    timeframe = '5m'
    
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 0.1
    }
    
    # Optimal stoploss designed for the strategy
    stoploss = -0.1
    
    # Trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = False
    
    # Optimal timeframe for the strategy
    timeframe = '5m'
    
    # Run "populate_indicators" only for new candle
    process_only_new_candles = False
    
    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30
    
    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # Optional order time in force
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will automatically be available for use in the `populate_indicators` method.
        For more information, please see the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("BTC/USDT", "5m"),
                            ("ETH/USDT", "5m"),
                            ]
        """
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        
        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategy
        """
        
        # RSI
        dataframe['rsi'] = 50  # Simple RSI placeholder
        
        # Simple moving averages
        dataframe['sma_20'] = dataframe['close'].rolling(window=20).mean()
        dataframe['sma_50'] = dataframe['close'].rolling(window=50).mean()
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['sma_20']) &
                (dataframe['sma_20'] > dataframe['sma_50'])
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['sma_20']) |
                (dataframe['sma_20'] < dataframe['sma_50'])
            ),
            'exit_long'] = 1
        
        return dataframe
