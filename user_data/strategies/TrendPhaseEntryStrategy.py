"""
Trend Phase Entry Strategies (A/B)
- A: Uses 5-state labeling with hysteresis, gating (>=0.50), median=3, dwell>=2, entry only in Early states, 0-candle delay after change, volume confirmation 1.00x.
- B: Uses simple percentile labels without hysteresis/gating, otherwise same constraints.
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


class _BaseTrendPhase(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    process_only_new_candles = True
    startup_candle_count = 1000
    can_short = False

    minimal_roi = {"0": 0.03, "60": 0.015, "120": 0.01}
    stoploss = -0.06

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.detector = RegimeDetector()
        self.trained = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Basic indicators for info
        df['ema20'] = ta.EMA(df, timeperiod=20)
        df['ema50'] = ta.EMA(df, timeperiod=50)
        df['adx'] = ta.ADX(df, timeperiod=14)
        # Volume confirmation base (1.00x)
        df['vol_sma20'] = df['volume'].rolling(20).mean()
        df['vol_ok'] = df['volume'] > (df['vol_sma20'] * 1.00)

        # Train HMM once sufficient candles
        if not self.trained and len(df) >= 1000:
            try:
                self.detector.train(df.tail(1000))
                self.trained = True
            except Exception as e:
                self.logger.warning(f'HMM train skipped: {e}')
                self.trained = False

        # Predict regimes/conf if trained
        if self.trained:
            regimes, conf = self.detector.predict_regime_sequence(df)
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

        # Labels by variant
        labels = self._label(df)
        # Align if needed
        if len(labels) != len(df):
            m = min(len(labels), len(df))
            df = df.tail(m).copy()
            labels = labels.tail(m)
        df['phase_label'] = labels.values

        # 0-candle delay (no enforced stability window)
        df['stable0'] = True

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['enter_long'] = False
        mask = (
            (df['phase_label'] == 'Uptrend_Early') &
            (df['stable0']) &
            (df['vol_ok']) &
            (df['regime'].isin(['trending', 'high_volatility']))
        )
        df['enter_long'] = mask
        # enter_tag for regime breakdown
        df['enter_tag'] = None
        df.loc[mask & (df['regime'] == 'trending'), 'enter_tag'] = 'UpEarly_trending'
        df.loc[mask & (df['regime'] == 'high_volatility'), 'enter_tag'] = 'UpEarly_highvol'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['exit_long'] = (
            (df['phase_label'].isin(['Uptrend_Late', 'Downtrend_Early', 'Downtrend_Late', 'Sideways'])) |
            (df['regime'] == 'high_volatility')
        )
        return df

    def _label(self, df: DataFrame) -> pd.Series:
        raise NotImplementedError


class TrendPhaseEntryStrategyA(_BaseTrendPhase):
    """A''': hysteresis + gating (>=0.40), median=3, dwell>=2, Early-only entries"""
    def _label(self, df: DataFrame) -> pd.Series:
        conf_s = pd.Series(df['regime_conf'].values, index=df.index)
        labels = self.detector.label_trend_phase_5state(
            dataframe=df,
            conf_series=conf_s,
            conf_min=0.40,
            median_window=3,
            dwell_min=2,
            early_up_in=0.60,
            early_up_out=0.50,
            early_dn_in=0.40,
            early_dn_out=0.50,
            late_up_in=0.90,
            late_up_out=0.85,
            late_dn_in=0.10,
            late_dn_out=0.15,
        )
        labels = labels.mask(labels == 'Sideways', 'Neutral')
        return labels


class TrendPhaseEntryStrategyB(_BaseTrendPhase):
    """B: simple percentile labels (no hysteresis/gating)"""
    def _label(self, df: DataFrame) -> pd.Series:
        score = self.detector.compute_trend_phase_score(df)
        s = score.dropna()
        if s.empty:
            return pd.Series(index=df.index, data='Neutral')
        p25 = s.quantile(0.25)
        p40 = s.quantile(0.40)
        p60 = s.quantile(0.60)
        p75 = s.quantile(0.75)
        labels = pd.Series(index=df.index, data='Neutral', dtype='object')
        labels[score >= p75] = 'Uptrend_Early'
        labels[(score >= p60) & (score < p75)] = 'Uptrend_Late'
        labels[score <= p25] = 'Downtrend_Early'
        labels[(score > p25) & (score < p40)] = 'Downtrend_Late'
        labels = labels.fillna('Neutral')
        return labels
