"""
HMM-based Market Regime Detector

This module implements a Hidden Markov Model for detecting market regimes
using Gaussian HMM with multiple features including returns, volatility,
volume, and trend strength.

Regime Types:
- high_volatility: Volatile/choppy markets
- low_volatility: Calm/ranging markets  
- trending: Strong directional movements
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Market regime detector using Hidden Markov Models.
    
    Uses a 3-state Gaussian HMM to classify market conditions into:
    - High Volatility (choppy, risky)
    - Low Volatility (calm, ranging)
    - Trending (directional, momentum)
    
    Attributes:
        n_states (int): Number of hidden states (default: 3)
        model (GaussianHMM): The trained HMM model
        scaler (RobustScaler): Feature scaler for robust normalization
        is_trained (bool): Whether the model has been trained
        regime_names (dict): Mapping from state index to regime name
    """
    
    def __init__(self, n_states: int = 3, random_state: int = 42):
        """
        Initialize the RegimeDetector.
        
        Args:
            n_states: Number of hidden states (default: 3 for stability)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        
        # Initialize HMM with diagonal covariance for stability
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",  # Diagonal covariance for stability
            n_iter=50,   # Reduced iterations to prevent overfitting
            tol=1e-3,    # Relaxed tolerance for better convergence
            random_state=random_state,
            verbose=False
        )
        
        # Feature scaler for normalization (using RobustScaler for better outlier handling)
        self.scaler = RobustScaler()
        
        # Training status
        self.is_trained = False
        
        # Regime names (will be determined after training based on characteristics)
        self.regime_names = {
            i: f"regime_{i}" for i in range(n_states)
        }
        
    def prepare_features(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Prepare enhanced, stable features for HMM training.
        
        Features calculated:
        1. Returns: Log returns over key periods (better for crypto data)
        2. Volatility: Rolling standard deviation (smoothed)
        3. Momentum: Price momentum indicators
        4. Volume: Volume trend (simplified)
        5. Price Range: High-low normalized range
        6. Slope Analysis: Trend strength indicators
        7. Skewness: Distribution asymmetry analysis
        
        Args:
            dataframe: DataFrame with OHLCV data
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in dataframe.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = dataframe.copy()
        features = []
        
        # Feature 1: Log Returns (1, 5, 20 periods) - Better for crypto data
        for period in [1, 5, 20]:
            # Use log returns instead of pct_change for better crypto data handling
            df[f'returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
            features.append(f'returns_{period}')
        
        # Feature 2: Volatility (rolling standard deviation - smoothed)
        df['volatility_5'] = df['returns_1'].rolling(window=5).std()
        df['volatility_20'] = df['returns_1'].rolling(window=20).std()
        # Smooth volatility to reduce noise
        df['volatility_5'] = df['volatility_5'].rolling(window=3).mean()
        df['volatility_20'] = df['volatility_20'].rolling(window=5).mean()
        features.extend(['volatility_5', 'volatility_20'])
        
        # Feature 3: Price momentum (simple moving averages)
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['momentum_5'] = (df['close'] - df['sma_5']) / df['sma_5']
        df['momentum_20'] = (df['close'] - df['sma_20']) / df['sma_20']
        features.extend(['momentum_5', 'momentum_20'])
        
        # Feature 4: Volume trend (simplified and smoothed)
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
        # Smooth volume ratio to reduce noise
        df['volume_ratio'] = df['volume_ratio'].rolling(window=3).mean()
        features.append('volume_ratio')
        
        # Feature 5: Price range (high-low normalized)
        df['price_range'] = (df['high'] - df['low']) / df['close']
        # Smooth price range
        df['price_range'] = df['price_range'].rolling(window=3).mean()
        features.append('price_range')
        
        # Feature 6: Slope analysis (trend strength)
        df['sma_slope_5'] = df['sma_5'].diff(5)  # 5-period slope of SMA
        df['sma_slope_20'] = df['sma_20'].diff(5)  # 5-period slope of SMA
        df['trend_strength'] = df['sma_slope_20'] * df['momentum_20']  # Combined trend strength
        features.extend(['sma_slope_5', 'sma_slope_20', 'trend_strength'])
        
        # Feature 7: Skewness analysis (distribution asymmetry)
        df['returns_skew_5'] = df['returns_1'].rolling(window=5).skew()
        df['returns_skew_20'] = df['returns_1'].rolling(window=20).skew()
        # Smooth skewness to reduce noise
        df['returns_skew_5'] = df['returns_skew_5'].rolling(window=3).mean()
        df['returns_skew_20'] = df['returns_skew_20'].rolling(window=5).mean()
        features.extend(['returns_skew_5', 'returns_skew_20'])

        # Trend Phase Score scaffold (v2): lightweight, no behavior change for now
        # This computes components but does not add to features to avoid impacting the trained model yet.
        try:
            ema20 = df['close'].ewm(span=20, adjust=False).mean()
            ema50 = df['close'].ewm(span=50, adjust=False).mean()
            ema100 = df['close'].ewm(span=100, adjust=False).mean()
            # Slopes
            slope20 = ema20.diff()
            slope50 = ema50.diff()
            slope100 = ema100.diff()
            # MACD slope (approx via diff of MACD line)
            macd_fast = df['close'].ewm(span=12, adjust=False).mean()
            macd_slow = df['close'].ewm(span=26, adjust=False).mean()
            macd = macd_fast - macd_slow
            macd_slope = macd.diff()
            # Z-score vs EMA50
            price_minus_ema50 = df['close'] - ema50
            zscore_ema50 = (price_minus_ema50 - price_minus_ema50.rolling(50).mean()) / (price_minus_ema50.rolling(50).std())
            # Distance to BB mid (20)
            bb_mid = df['close'].rolling(20).mean()
            dist_bb_mid = (df['close'] - bb_mid) / bb_mid
            # Phase location: distance to rolling extrema
            roll_high = df['close'].rolling(100).max()
            roll_low = df['close'].rolling(100).min()
            dist_to_high = (roll_high - df['close']) / roll_high
            dist_to_low = (df['close'] - roll_low) / roll_low
            recovery_from_low = (df['close'] - roll_low) / (roll_high - roll_low + 1e-9)
            # Trend quality / volatility
            bb_width = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
            atr_pct = ((df['high'] - df['low']).rolling(14).mean() / df['close'])
            # Exhaustion proxy
            rsi = 100 - (100 / (1 + (df['close'].diff().clip(lower=0).rolling(14).mean() / (df['close'].diff().abs().rolling(14).mean() - df['close'].diff().clip(lower=0).rolling(14).mean() + 1e-9))))
            # Composite (not returned): weighted sum
            trend_phase_score = (
                0.25 * slope20.fillna(0) +
                0.15 * slope50.fillna(0) +
                0.10 * slope100.fillna(0) +
                0.15 * macd_slope.fillna(0) +
                0.15 * zscore_ema50.fillna(0) +
                0.10 * dist_bb_mid.fillna(0) +
                0.05 * recovery_from_low.fillna(0) +
                (-0.05) * bb_width.fillna(0)
            )
            # Store in df for debugging (not part of features to keep current behavior stable)
            df['_trend_phase_score'] = trend_phase_score
        except Exception:
            # Silent: keep current behavior intact if any component fails
            df['_trend_phase_score'] = np.nan
        
        # Create feature dataframe
        feature_df = df[features].dropna()
        
        if len(feature_df) == 0:
            raise ValueError("No valid features after preprocessing")
        
        # Remove extreme outliers (beyond 3 standard deviations)
        for col in feature_df.columns:
            mean_val = feature_df[col].mean()
            std_val = feature_df[col].std()
            if std_val > 0:  # Avoid division by zero
                feature_df[col] = feature_df[col].clip(
                    lower=mean_val - 3*std_val,
                    upper=mean_val + 3*std_val
                )
        
        return feature_df.values
    
    
    def compute_trend_phase_score(self, dataframe: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        Compute a composite Trend Phase Score with conservative weighting and rolling
        standardization to be robust across pairs/timeframes. This does not affect
        the HMM model; it is a separate diagnostic signal.

        Components (all smoothed, standardized with rolling z-score):
        - EMA20/50/100 slopes + MACD slope (higher weight)
        - Z-score of price vs EMA50, distance to BB mid
        - Phase location: distance to rolling high/low, recovery-from-low
        - Volatility context: BB width, ATR%

        Args:
            dataframe: OHLCV dataframe with columns ['open','high','low','close','volume']
            window: Rolling window for z-score standardization (default: 50)

        Returns:
            pd.Series of the composite score aligned to dataframe.index
        """
        df = dataframe.copy()
        if any(col not in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Dataframe must contain ['open','high','low','close','volume']")

        # EMAs and slopes
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        ema100 = df['close'].ewm(span=100, adjust=False).mean()
        slope20 = ema20.diff()
        slope50 = ema50.diff()
        slope100 = ema100.diff()

        # MACD slope
        macd_fast = df['close'].ewm(span=12, adjust=False).mean()
        macd_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd = macd_fast - macd_slow
        macd_slope = macd.diff()

        # Price vs EMA50 z-score
        price_minus_ema50 = df['close'] - ema50
        zscore_ema50 = (price_minus_ema50 - price_minus_ema50.rolling(window).mean()) / (
            price_minus_ema50.rolling(window).std()
        )

        # Distance to BB mid (20 SMA)
        bb_mid = df['close'].rolling(20).mean()
        dist_bb_mid = (df['close'] - bb_mid) / bb_mid

        # Phase location relative to rolling extrema
        roll_high = df['close'].rolling(100).max()
        roll_low = df['close'].rolling(100).min()
        dist_to_high = (roll_high - df['close']) / roll_high
        recovery_from_low = (df['close'] - roll_low) / (roll_high - roll_low + 1e-9)

        # Volatility context
        bb_width = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
        atr_pct = (df['high'] - df['low']).rolling(14).mean() / df['close']

        # Rolling z-score standardization helper
        def rz(series: pd.Series, w: int = window) -> pd.Series:
            m = series.rolling(w).mean()
            s = series.rolling(w).std()
            return (series - m) / (s + 1e-9)

        # Standardize components
        s_slope20 = rz(slope20)
        s_slope50 = rz(slope50)
        s_slope100 = rz(slope100)
        s_macd_slope = rz(macd_slope)
        s_z_ema50 = rz(zscore_ema50)
        s_dist_bb_mid = rz(dist_bb_mid)
        s_dist_to_high = rz(dist_to_high)
        s_recovery_from_low = rz(recovery_from_low)
        s_bb_width = rz(bb_width)
        s_atr_pct = rz(atr_pct)

        # Conservative weighting: slopes/acceleration get higher weights
        score = (
            0.22 * s_slope20.fillna(0) +
            0.16 * s_slope50.fillna(0) +
            0.10 * s_slope100.fillna(0) +
            0.16 * s_macd_slope.fillna(0) +
            0.12 * s_z_ema50.fillna(0) +
            0.08 * s_dist_bb_mid.fillna(0) +
            0.08 * s_recovery_from_low.fillna(0) +
            (-0.04) * s_dist_to_high.fillna(0) +
            (-0.08) * s_bb_width.fillna(0) +
            (-0.04) * s_atr_pct.fillna(0)
        )

        return score.rename('trend_phase_score')


    def summarize_trend_phase(self, dataframe: pd.DataFrame) -> dict:
        """
        Quick validation helper: returns distribution stats and correlations
        with forward log-returns (5 and 20 candles).

        Returns a dict with percentiles and correlations for rapid inspection.
        """
        score = self.compute_trend_phase_score(dataframe)
        df = dataframe.copy()
        df['score'] = score
        # Forward returns (no leakage for current bar)
        df['fret_5'] = np.log(df['close'].shift(-5) / df['close'])
        df['fret_20'] = np.log(df['close'].shift(-20) / df['close'])

        desc = df['score'].dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        corr5 = float(df[['score', 'fret_5']].dropna().corr().iloc[0, 1]) if df[['score', 'fret_5']].dropna().shape[0] > 5 else np.nan
        corr20 = float(df[['score', 'fret_20']].dropna().corr().iloc[0, 1]) if df[['score', 'fret_20']].dropna().shape[0] > 20 else np.nan

        return {
            'percentiles': desc,
            'corr_forward_5': corr5,
            'corr_forward_20': corr20,
            'count': int(df['score'].dropna().shape[0])
        }

    # ========================
    # v2: 5-State Phase Labeling
    # ========================
    @staticmethod
    def _winsorize(series: pd.Series, p_low: float = 0.01, p_high: float = 0.99) -> pd.Series:
        s = series.dropna()
        if s.empty:
            return series
        lo = s.quantile(p_low)
        hi = s.quantile(p_high)
        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def _rolling_z(series: pd.Series, window: int = 300) -> pd.Series:
        m = series.rolling(window).mean()
        sd = series.rolling(window).std()
        return (series - m) / (sd + 1e-9)

    def get_default_percentile_thresholds(self, dataframe: pd.DataFrame) -> dict:
        """
        Compute default percentile thresholds on rolling-z normalized Trend Phase Score.
        Percentiles are computed on the available sample to remain pair/timeframe-robust.
        Returns a dict with keys: p10,p15,p20,p30,p40,p60,p70,p80,p85,p90
        """
        raw = self.compute_trend_phase_score(dataframe)
        score_w = self._winsorize(raw)
        score = self._rolling_z(score_w)
        qs = score.dropna().quantile([0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.85, 0.90]).to_dict()
        # Map to named keys
        keys = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.85, 0.90]
        names = ['p10','p15','p20','p30','p40','p60','p70','p80','p85','p90']
        return {name: float(qs.get(k, np.nan)) for name, k in zip(names, keys)}

    def label_trend_phase_5state(
        self,
        dataframe: pd.DataFrame,
        conf_series: Optional[pd.Series] = None,
        conf_min: float = 0.6,
        median_window: int = 5,
        dwell_min: int = 3,
        # Percentile overrides
        early_up_in: float = 0.80,
        early_up_out: float = 0.70,
        early_dn_in: float = 0.20,
        early_dn_out: float = 0.30,
        late_up_in: float = 0.90,
        late_up_out: float = 0.85,
        late_dn_in: float = 0.10,
        late_dn_out: float = 0.15
    ) -> pd.Series:
        """
        Label each bar with one of 5 phases using Trend Phase Score + simple slope/volatility proxies
        and hysteresis banding. Confidence gating is optional via conf_series.

        States: ['Uptrend_Early','Uptrend_Late','Downtrend_Early','Downtrend_Late','Sideways']

        Rules (percentile-based thresholds after rolling z-score):
        - Uptrend_Early: score >= p80 AND acceleration > 0
        - Uptrend_Late: p85..p90 AND acceleration <= 0 (exhaustion)
        - Downtrend_Early: score <= p20 AND acceleration < 0
        - Downtrend_Late: p10..p15 AND acceleration >= 0
        - Sideways override: low volatility (bb_width below p30) AND |slope20| small

        Hysteresis (entry/exit bands):
        - Early in/out = 80/70 (up) and 20/30 (down)
        - Late in/out = 90/85 (up) and 10/15 (down)
        Median filter window=5 and min dwell of 3 bars applied.
        """
        df = dataframe.copy()

        # Compute normalized score
        raw = self.compute_trend_phase_score(df)
        score = self._rolling_z(self._winsorize(raw)).rename('score')

        # EMA-based slope/acceleration proxies
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        slope20 = ema20.diff()
        accel20 = slope20.diff()

        # Volatility proxy (BB width approx via range) and small-slope override
        bb_width = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']

        # Percentiles computed on current sample
        thresholds = self.get_default_percentile_thresholds(df)
        p10 = thresholds['p10']; p15 = thresholds['p15']; p20 = thresholds['p20']
        p30 = thresholds['p30']; p70 = thresholds['p70']
        p80 = thresholds['p80']; p85 = thresholds['p85']; p90 = thresholds['p90']

        # Apply overrides by re-mapping desired percentiles
        # compute corresponding score cutoffs from sample percentiles
        def q(p: float) -> float:
            return float(score.dropna().quantile(p)) if score.notna().any() else np.nan

        up_in_cut = q(early_up_in)
        up_out_cut = q(early_up_out)
        dn_in_cut = q(early_dn_in)
        dn_out_cut = q(early_dn_out)
        up_late_in_cut = q(late_up_in)
        up_late_out_cut = q(late_up_out)
        dn_late_in_cut = q(late_dn_in)
        dn_late_out_cut = q(late_dn_out)

        # Hysteresis labeling
        labels = pd.Series('Neutral', index=score.index, dtype='object')
        state = 'Neutral'
        for t, v in score.items():
            a = accel20.loc[t] if t in accel20.index else np.nan
            bw = bb_width.loc[t] if t in bb_width.index else np.nan
            # Sideways override (low bb width and small slope)
            is_sideways = False
            if not pd.isna(bw) and not pd.isna(slope20.loc[t]):
                is_sideways = (bw <= bb_width.quantile(0.30)) and (abs(slope20.loc[t]) <= abs(slope20).quantile(0.30))

            if pd.isna(v):
                labels.at[t] = state
                continue

            # Confidence gating (optional)
            if conf_series is not None:
                c = conf_series.loc[t] if t in conf_series.index else np.nan
                if pd.isna(c) or c < conf_min:
                    # keep prior state but mark as Neutral if required
                    labels.at[t] = state
                    continue

            if state == 'Neutral':
                if is_sideways:
                    state = 'Sideways'
                elif v >= up_late_in_cut and (not pd.isna(a) and a <= 0):
                    state = 'Uptrend_Late'
                elif v >= up_in_cut and (not pd.isna(a) and a > 0):
                    state = 'Uptrend_Early'
                elif v <= dn_late_in_cut and (not pd.isna(a) and a >= 0):
                    state = 'Downtrend_Late'
                elif v <= dn_in_cut and (not pd.isna(a) and a < 0):
                    state = 'Downtrend_Early'
            elif state == 'Uptrend_Early':
                # Exit band to Neutral
                if v < up_out_cut:
                    state = 'Neutral'
            elif state == 'Downtrend_Early':
                if v > dn_out_cut:
                    state = 'Neutral'
            elif state == 'Uptrend_Late':
                if v < up_late_out_cut:
                    state = 'Neutral'
            elif state == 'Downtrend_Late':
                if v > dn_late_out_cut:
                    state = 'Neutral'

            labels.at[t] = state if not is_sideways else 'Sideways'

        # Median filter (stability)
        if median_window and median_window > 1:
            as_int = labels.map({
                'Downtrend_Late': -2,
                'Downtrend_Early': -1,
                'Neutral': 0,
                'Sideways': 0,
                'Uptrend_Early': 1,
                'Uptrend_Late': 2
            }).fillna(0)
            filt = as_int.rolling(median_window, center=True, min_periods=1).median().round().astype(int)
            labels = filt.map({-2: 'Downtrend_Late', -1: 'Downtrend_Early', 0: 'Neutral', 1: 'Uptrend_Early', 2: 'Uptrend_Late'})

        # Enforce minimum dwell by collapsing short runs to previous state
        if dwell_min and dwell_min > 1:
            runs = (labels != labels.shift(1)).cumsum()
            sizes = labels.groupby(runs).size()
            idx = 0
            for rid, size in sizes.items():
                if size < dwell_min:
                    start = idx
                    end = idx + size
                    prev_state = labels.iloc[start - 1] if start > 0 else 'Neutral'
                    labels.iloc[start:end] = prev_state
                idx += size

        return labels.rename('trend_phase_5state')

    
    def train(self, dataframe: pd.DataFrame, lookback: int = 1000) -> 'RegimeDetector':
        """
        Train the HMM on historical data.
        
        Args:
            dataframe: Historical OHLCV dataframe
            lookback: Number of most recent candles to use for training (minimum 1000)
            
        Returns:
            Self (for method chaining)
            
        Raises:
            ValueError: If insufficient data for training
        """
        # Ensure minimum data requirements for HMM convergence
        min_required = 1000
        if len(dataframe) < min_required:
            raise ValueError(f"Insufficient data: need at least {min_required} candles, got {len(dataframe)}")
        
        # Use most recent lookback candles
        df_train = dataframe.tail(lookback).copy()
        
        # Prepare features
        X = self.prepare_features(df_train)
        
        if len(X) < 200:
            raise ValueError(f"Insufficient valid samples after feature preparation: {len(X)} < 200")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train HMM with better parameters for convergence
        self.model.n_iter = 200  # Increase iterations
        self.model.tol = 1e-6    # Tighter tolerance
        
        try:
            self.model.fit(X_scaled)
            self.is_trained = True
            
            # Determine regime names based on characteristics
            self._assign_regime_names(X_scaled)
            
            logger.info(f"HMM trained successfully with {X_scaled.shape[0]} samples")
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            raise
        
        return self
    
    def _assign_regime_names(self, X_scaled: np.ndarray) -> None:
        """
        Assign meaningful names to regimes based on their characteristics.
        
        Analyzes the mean feature values for each state to determine:
        - High volatility regime (high volatility features)
        - Low volatility regime (low volatility features)
        - Trending regime (high momentum features)
        
        Args:
            X_scaled: Scaled feature matrix used for training
        """
        # Predict states for training data
        states = self.model.predict(X_scaled)
        
        # Calculate mean features for each state
        state_characteristics = {}
        
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() > 0:
                state_mean = X_scaled[mask].mean(axis=0)
                
                # Features: [returns_1, returns_5, returns_20, volatility_5, volatility_20, momentum_5, momentum_20, volume_ratio, price_range, sma_slope_5, sma_slope_20, trend_strength, returns_skew_5, returns_skew_20]
                volatility = (state_mean[3] + state_mean[4]) / 2  # avg of volatility features
                momentum = (state_mean[5] + state_mean[6]) / 2    # avg of momentum features
                volume = state_mean[7]  # volume_ratio
                slope = (state_mean[9] + state_mean[10]) / 2     # avg of slope features
                skewness = (state_mean[12] + state_mean[13]) / 2  # avg of skewness features
                trend_strength = state_mean[11]  # trend_strength
                
                state_characteristics[state] = {
                    'volatility': volatility,
                    'momentum': momentum,
                    'volume': volume,
                    'slope': slope,
                    'skewness': skewness,
                    'trend_strength': trend_strength,
                    'count': mask.sum()
                }
        
        # Sort states by characteristics
        states_by_vol = sorted(state_characteristics.items(), 
                              key=lambda x: x[1]['volatility'])
        states_by_momentum = sorted(state_characteristics.items(),
                                   key=lambda x: x[1]['momentum'])
        states_by_trend = sorted(state_characteristics.items(),
                                key=lambda x: x[1]['trend_strength'])
        states_by_slope = sorted(state_characteristics.items(),
                                key=lambda x: x[1]['slope'])
        
        # Enhanced regime assignment using multiple criteria
        # 1. High volatility + high skewness = high_volatility
        # 2. High trend strength + positive slope = trending  
        # 3. Low volatility + low momentum = low_volatility
        
        high_vol_state = states_by_vol[-1][0]
        high_trend_state = states_by_trend[-1][0]
        high_slope_state = states_by_slope[-1][0]
        
        # Assign high_volatility: highest volatility with consideration of skewness
        self.regime_names[high_vol_state] = "high_volatility"
        
        # Assign trending: best combination of trend strength and slope
        if high_trend_state == high_vol_state:
            # If same state, choose next best for trending
            high_trend_state = states_by_trend[-2][0] if len(states_by_trend) > 1 else high_slope_state
        
        self.regime_names[high_trend_state] = "trending"
        
        # Assign remaining states to low_volatility
        for state in range(self.n_states):
            if state not in [high_vol_state, high_trend_state]:
                self.regime_names[state] = "low_volatility"
        
        # Log regime characteristics for debugging
        logger.info("Enhanced regime characteristics:")
        for state, chars in state_characteristics.items():
            logger.info(f"  {self.regime_names[state]}: vol={chars['volatility']:.3f}, "
                       f"momentum={chars['momentum']:.3f}, slope={chars['slope']:.3f}, "
                       f"skewness={chars['skewness']:.3f}, trend_strength={chars['trend_strength']:.3f}, "
                       f"count={chars['count']}")
        
        # If high_vol and trending are the same state, assign differently
        if high_vol_state == high_trend_state:
            remaining_states = [s for s in range(self.n_states) if s != high_vol_state]
            if len(remaining_states) >= 2:
                self.regime_names[remaining_states[0]] = "low_volatility"
                self.regime_names[remaining_states[1]] = "medium_volatility"
    
    def predict_regime(self, dataframe: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict current market regime with confidence score.
        
        Args:
            dataframe: Recent OHLCV dataframe (at least last 50 candles recommended)
            
        Returns:
            Tuple of (regime_name, confidence_score)
            - regime_name: One of ["high_volatility", "low_volatility", "trending"]
            - confidence_score: Probability of predicted regime (0-1)
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        # Prepare features
        X = self.prepare_features(dataframe)
        
        # Use only the most recent sample for prediction
        X_recent = X[-1:].reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X_recent)
        
        # Predict state
        state = self.model.predict(X_scaled)[0]
        
        # Get probability distribution
        probs = self.model.predict_proba(X_scaled)[0]
        confidence = probs[state]
        
        # Get regime name
        regime_name = self.regime_names[state]
        
        return regime_name, float(confidence)
    
    def predict_regime_sequence(self, dataframe: pd.DataFrame, smooth_window: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regimes for entire dataframe with dynamic smoothing.
        
        Args:
            dataframe: OHLCV dataframe
            smooth_window: Window size for smoothing regime transitions.
                          If None, uses dynamic window based on dataset size.
            
        Returns:
            Tuple of (regime_sequence, confidence_sequence)
            - regime_sequence: Array of regime names for each candle
            - confidence_sequence: Array of confidence scores for each candle
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction. Call train() first.")
        
        # Prepare features
        X = self.prepare_features(dataframe)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict states for entire sequence
        states = self.model.predict(X_scaled)
        
        # Get probability distributions
        probs = self.model.predict_proba(X_scaled)
        confidences = np.max(probs, axis=1)
        
        # Calculate dynamic smoothing window if not provided
        if smooth_window is None:
            smooth_window = self._calculate_dynamic_smooth_window(len(states))
        
        # Apply smoothing to reduce noise
        if smooth_window > 1 and len(states) > smooth_window:
            states = self._smooth_states(states, smooth_window)
        
        # Convert states to regime names
        regime_sequence = np.array([self.regime_names[state] for state in states])
        
        return regime_sequence, confidences
    
    def _smooth_states(self, states: np.ndarray, window: int) -> np.ndarray:
        """
        Apply smoothing to state sequence to reduce noise.
        
        Args:
            states: Array of state indices
            window: Smoothing window size
            
        Returns:
            Smoothed state array
        """
        smoothed = states.copy()
        
        for i in range(window, len(states) - window):
            # Get window of states
            window_states = states[i-window//2:i+window//2+1]
            
            # Find most common state in window
            unique, counts = np.unique(window_states, return_counts=True)
            most_common = unique[np.argmax(counts)]
            
            # Only change if confidence is high (most common state appears > 50% of time)
            if np.max(counts) > window // 2:
                smoothed[i] = most_common
        
        return smoothed
    
    def _calculate_dynamic_smooth_window(self, sequence_length: int) -> int:
        """
        Calculate dynamic smoothing window based on sequence length and characteristics.
        
        Args:
            sequence_length: Length of the sequence to be smoothed
            
        Returns:
            Optimal smoothing window size
        """
        # Base window size based on sequence length
        # Use 0.2% of sequence length, but with reasonable bounds
        base_window = max(3, int(sequence_length * 0.002))
        
        # Adjust based on sequence characteristics
        if sequence_length < 100:
            # Small datasets: minimal smoothing
            window = min(3, base_window)
        elif sequence_length < 500:
            # Medium datasets: moderate smoothing
            window = min(5, base_window)
        elif sequence_length < 2000:
            # Large datasets: more smoothing
            window = min(10, base_window)
        else:
            # Very large datasets: adaptive smoothing
            window = min(20, base_window)
        
        # Ensure window is odd for better smoothing
        if window % 2 == 0:
            window += 1
        
        # Log the dynamic window calculation
        logger.debug(f"Dynamic smooth window: {window} (sequence_length: {sequence_length})")
        
        return window
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the state transition matrix.
        
        Returns:
            Transition probability matrix of shape (n_states, n_states)
            where element [i,j] is P(state_j | state_i)
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before accessing transition matrix.")
        
        return self.model.transmat_
    
    def get_regime_probabilities(self, dataframe: pd.DataFrame) -> dict:
        """
        Get probability distribution over all regimes.
        
        Args:
            dataframe: Recent OHLCV dataframe
            
        Returns:
            Dictionary mapping regime names to probabilities
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        
        # Prepare and scale features
        X = self.prepare_features(dataframe)
        X_recent = X[-1:].reshape(1, -1)
        X_scaled = self.scaler.transform(X_recent)
        
        # Get probabilities
        probs = self.model.predict_proba(X_scaled)[0]
        
        # Map to regime names
        regime_probs = {
            self.regime_names[state]: float(probs[state])
            for state in range(self.n_states)
        }
        
        return regime_probs
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        status = "trained" if self.is_trained else "untrained"
        return f"RegimeDetector(n_states={self.n_states}, status={status})"
