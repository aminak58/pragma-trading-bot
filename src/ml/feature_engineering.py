"""
Feature Engineering for ML Models

Provides comprehensive feature engineering capabilities:
- Technical indicator features
- Regime-specific features
- Time-based features
- Statistical features
- Cross-asset features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import talib.abstract as ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for ML models.
    
    Creates a wide variety of features from OHLCV data and technical indicators
    to improve ML model performance.
    """
    
    def __init__(
        self,
        include_technical_indicators: bool = True,
        include_regime_features: bool = True,
        include_time_features: bool = True,
        include_statistical_features: bool = True,
        max_lag_periods: int = 20
    ):
        """
        Initialize feature engineer.
        
        Args:
            include_technical_indicators: Whether to include technical indicators
            include_regime_features: Whether to include regime-specific features
            include_time_features: Whether to include time-based features
            include_statistical_features: Whether to include statistical features
            max_lag_periods: Maximum number of lag periods for features
        """
        self.include_technical_indicators = include_technical_indicators
        self.include_regime_features = include_regime_features
        self.include_time_features = include_time_features
        self.include_statistical_features = include_statistical_features
        self.max_lag_periods = max_lag_periods
        
        # Feature groups
        self.feature_groups = {
            'price': [],
            'volume': [],
            'technical': [],
            'regime': [],
            'time': [],
            'statistical': [],
            'cross_asset': []
        }
        
        logger.info("FeatureEngineer initialized")
    
    def create_all_features(
        self,
        dataframe: pd.DataFrame,
        regime: str = 'unknown',
        additional_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Create all available features.
        
        Args:
            dataframe: OHLCV dataframe
            regime: Market regime
            additional_data: Additional data for cross-asset features
            
        Returns:
            Dataframe with all features
        """
        df = dataframe.copy()
        
        # Basic price features
        df = self._create_price_features(df)
        
        # Volume features
        df = self._create_volume_features(df)
        
        # Technical indicators
        if self.include_technical_indicators:
            df = self._create_technical_features(df)
        
        # Regime features
        if self.include_regime_features:
            df = self._create_regime_features(df, regime)
        
        # Time features
        if self.include_time_features:
            df = self._create_time_features(df)
        
        # Statistical features
        if self.include_statistical_features:
            df = self._create_statistical_features(df)
        
        # Cross-asset features
        if additional_data:
            df = self._create_cross_asset_features(df, additional_data)
        
        # Lagged features
        df = self._create_lagged_features(df)
        
        # Interaction features
        df = self._create_interaction_features(df)
        
        # Clean up
        df = self._clean_features(df)
        
        logger.info(f"Created {len(df.columns)} features for {len(df)} samples")
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        # Returns
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_returns_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_returns_5'] = np.log(df['close'] / df['close'].shift(5))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_close_ratio'] = df['high'] / df['close']
        df['low_close_ratio'] = df['low'] / df['close']
        
        # Price position within range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Price acceleration
        df['acceleration_5'] = df['momentum_5'] - df['momentum_5'].shift(1)
        df['acceleration_10'] = df['momentum_10'] - df['momentum_10'].shift(1)
        
        # Volatility (rolling standard deviation)
        df['volatility_5'] = df['returns_1'].rolling(5).std()
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Price channels
        df['price_channel_high'] = df['high'].rolling(20).max()
        df['price_channel_low'] = df['low'].rolling(20).min()
        df['price_channel_position'] = (df['close'] - df['price_channel_low']) / (df['price_channel_high'] - df['price_channel_low'])
        
        # Add to feature groups
        price_features = [
            'returns_1', 'returns_5', 'returns_10', 'returns_20',
            'log_returns_1', 'log_returns_5',
            'high_low_ratio', 'close_open_ratio', 'high_close_ratio', 'low_close_ratio',
            'price_position', 'momentum_5', 'momentum_10', 'momentum_20',
            'acceleration_5', 'acceleration_10',
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_ratio',
            'price_channel_position'
        ]
        self.feature_groups['price'].extend(price_features)
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        # Volume ratios
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_sma_10'] = df['volume'].rolling(10).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # Volume momentum
        df['volume_momentum_5'] = df['volume'] / df['volume'].shift(5) - 1
        df['volume_momentum_10'] = df['volume'] / df['volume'].shift(10) - 1
        
        # Volume-price relationship (create returns_1 if not exists)
        if 'returns_1' not in df.columns:
            df['returns_1'] = df['close'].pct_change(1)
        df['volume_price_trend'] = df['volume_ratio_20'] * df['returns_1']
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
        
        # Volume volatility
        df['volume_volatility'] = df['volume'].rolling(10).std()
        df['volume_volatility_ratio'] = df['volume_volatility'] / df['volume_sma_20']
        
        # On-balance volume approximation
        df['obv'] = (df['volume'] * np.sign(df['returns_1'])).cumsum()
        df['obv_ratio'] = df['obv'] / df['obv'].rolling(20).mean()
        
        # Add to feature groups
        volume_features = [
            'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
            'volume_momentum_5', 'volume_momentum_10',
            'volume_price_trend', 'volume_price_correlation',
            'volume_volatility', 'volume_volatility_ratio',
            'obv_ratio'
        ]
        self.feature_groups['volume'].extend(volume_features)
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        try:
            # Moving averages
            df['sma_5'] = ta.SMA(df, timeperiod=5)
            df['sma_10'] = ta.SMA(df, timeperiod=10)
            df['sma_20'] = ta.SMA(df, timeperiod=20)
            df['sma_50'] = ta.SMA(df, timeperiod=50)
            
            df['ema_5'] = ta.EMA(df, timeperiod=5)
            df['ema_10'] = ta.EMA(df, timeperiod=10)
            df['ema_20'] = ta.EMA(df, timeperiod=20)
            df['ema_50'] = ta.EMA(df, timeperiod=50)
            
            # Moving average crossovers
            df['sma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(int)
            
            # Price relative to moving averages
            df['price_sma_20_ratio'] = df['close'] / df['sma_20']
            df['price_ema_20_ratio'] = df['close'] / df['ema_20']
            
            # RSI
            df['rsi'] = ta.RSI(df, timeperiod=14)
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
            
            # MACD
            macd = ta.MACD(df)
            df['macd'] = macd['macd']
            df['macd_signal'] = macd['macdsignal']
            df['macd_histogram'] = macd['macdhist']
            df['macd_histogram_normalized'] = df['macd_histogram'] / df['close']
            
            # Bollinger Bands
            bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
            df['bb_upper'] = bb['upperband']
            df['bb_middle'] = bb['middleband']
            df['bb_lower'] = bb['lowerband']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ADX
            df['adx'] = ta.ADX(df, timeperiod=14)
            df['adx_normalized'] = df['adx'] / 100
            
            # ATR
            df['atr'] = ta.ATR(df, timeperiod=14)
            df['atr_percent'] = df['atr'] / df['close']
            
            # Stochastic
            stoch = ta.STOCH(df)
            df['stoch_k'] = stoch['slowk']
            df['stoch_d'] = stoch['slowd']
            df['stoch_normalized'] = (df['stoch_k'] - 50) / 50
            
            # Williams %R
            df['williams_r'] = ta.WILLR(df, timeperiod=14)
            df['williams_r_normalized'] = df['williams_r'] / 100
            
            # CCI
            df['cci'] = ta.CCI(df, timeperiod=14)
            df['cci_normalized'] = df['cci'] / 100
            
            # Add to feature groups
            technical_features = [
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'sma_cross_5_20', 'ema_cross_10_20',
                'price_sma_20_ratio', 'price_ema_20_ratio',
                'rsi_normalized', 'macd', 'macd_signal', 'macd_histogram_normalized',
                'bb_width', 'bb_position', 'adx_normalized',
                'atr_percent', 'stoch_normalized', 'williams_r_normalized', 'cci_normalized'
            ]
            self.feature_groups['technical'].extend(technical_features)
            
        except Exception as e:
            logger.warning(f"Error creating technical features: {e}")
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Create regime-specific features."""
        # Regime indicators
        df['regime_trending'] = int(regime == 'trending')
        df['regime_low_volatility'] = int(regime == 'low_volatility')
        df['regime_high_volatility'] = int(regime == 'high_volatility')
        df['regime_unknown'] = int(regime == 'unknown')
        
        # Regime-specific volatility adjustments
        if 'returns_1' not in df.columns:
            df['returns_1'] = df['close'].pct_change(1)
            
        if regime == 'high_volatility':
            df['volatility_adjusted_returns'] = df['returns_1'] * 0.5
        elif regime == 'low_volatility':
            df['volatility_adjusted_returns'] = df['returns_1'] * 1.5
        else:
            df['volatility_adjusted_returns'] = df['returns_1']
        
        # Regime-specific volume adjustments
        if 'volume_ratio_20' not in df.columns:
            # Create volume_ratio_20 if not exists
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
            
        if regime == 'trending':
            df['volume_trend_adjusted'] = df['volume_ratio_20'] * 1.2
        else:
            df['volume_trend_adjusted'] = df['volume_ratio_20']
        
        # Add to feature groups
        regime_features = [
            'regime_trending', 'regime_low_volatility', 'regime_high_volatility', 'regime_unknown',
            'volatility_adjusted_returns', 'volume_trend_adjusted'
        ]
        self.feature_groups['regime'].extend(regime_features)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        # Extract time components
        if 'date' in df.columns:
            df['hour'] = pd.to_datetime(df['date']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['date']).dt.day
            df['month'] = pd.to_datetime(df['date']).dt.month
        else:
            # Use index if no date column
            df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
            df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
            df['day_of_month'] = df.index.day if hasattr(df.index, 'day') else 0
            df['month'] = df.index.month if hasattr(df.index, 'month') else 0
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market session indicators
        df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_americas_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add to feature groups
        time_features = [
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_asia_session', 'is_europe_session', 'is_americas_session',
            'is_weekend'
        ]
        self.feature_groups['time'].extend(time_features)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        # Create returns_1 if not exists
        if 'returns_1' not in df.columns:
            df['returns_1'] = df['close'].pct_change(1)
            
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'mean_{window}'] = df['close'].rolling(window).mean()
            df[f'std_{window}'] = df['close'].rolling(window).std()
            df[f'skewness_{window}'] = df['returns_1'].rolling(window).skew()
            df[f'kurtosis_{window}'] = df['returns_1'].rolling(window).kurt()
            
            # Z-scores
            df[f'zscore_{window}'] = (df['close'] - df[f'mean_{window}']) / df[f'std_{window}']
        
        # Percentile ranks
        for window in [10, 20]:
            df[f'percentile_rank_{window}'] = df['close'].rolling(window).rank(pct=True)
        
        # Autocorrelation
        for lag in [1, 2, 3, 5]:
            df[f'autocorr_{lag}'] = df['returns_1'].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
        
        # Rolling correlations
        df['returns_volume_corr'] = df['returns_1'].rolling(20).corr(df['volume'])
        
        # Add to feature groups
        statistical_features = [
            'mean_5', 'mean_10', 'mean_20',
            'std_5', 'std_10', 'std_20',
            'skewness_5', 'skewness_10', 'skewness_20',
            'kurtosis_5', 'kurtosis_10', 'kurtosis_20',
            'zscore_5', 'zscore_10', 'zscore_20',
            'percentile_rank_10', 'percentile_rank_20',
            'autocorr_1', 'autocorr_2', 'autocorr_3', 'autocorr_5',
            'returns_volume_corr'
        ]
        self.feature_groups['statistical'].extend(statistical_features)
        
        return df
    
    def _create_cross_asset_features(
        self,
        df: pd.DataFrame,
        additional_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Create cross-asset features."""
        for asset_name, asset_df in additional_data.items():
            if 'close' not in asset_df.columns:
                continue
            
            # Calculate returns for other assets
            asset_returns = asset_df['close'].pct_change()
            
            # Align data
            common_index = df.index.intersection(asset_df.index)
            if len(common_index) == 0:
                continue
            
            # Cross-asset correlations
            df[f'{asset_name}_returns'] = asset_returns.reindex(df.index)
            df[f'{asset_name}_corr'] = df['returns_1'].rolling(20).corr(df[f'{asset_name}_returns'])
            
            # Relative strength
            df[f'{asset_name}_relative_strength'] = df['close'] / asset_df['close'].reindex(df.index)
        
        # Add to feature groups
        cross_asset_features = [col for col in df.columns if any(asset in col for asset in additional_data.keys())]
        self.feature_groups['cross_asset'].extend(cross_asset_features)
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features."""
        # Key features to lag
        lag_features = [
            'returns_1', 'volume_ratio_20', 'rsi_normalized',
            'bb_position', 'macd_histogram_normalized'
        ]
        
        # Create returns_1 if not exists
        if 'returns_1' not in df.columns:
            df['returns_1'] = df['close'].pct_change(1)
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in range(1, min(self.max_lag_periods + 1, 6)):  # Limit to 5 lags
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # Volume-price interactions
        if 'volume_ratio_20' in df.columns and 'returns_1' in df.columns:
            df['volume_price_interaction'] = df['volume_ratio_20'] * df['returns_1']
        
        # Volatility-momentum interactions
        if 'volatility_20' in df.columns and 'momentum_20' in df.columns:
            df['volatility_momentum_interaction'] = df['volatility_20'] * df['momentum_20']
        
        # RSI-momentum interactions
        if 'rsi_normalized' in df.columns and 'momentum_5' in df.columns:
            df['rsi_momentum_interaction'] = df['rsi_normalized'] * df['momentum_5']
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove constant features
        constant_features = df.columns[df.nunique() <= 1]
        if len(constant_features) > 0:
            logger.warning(f"Removing constant features: {constant_features.tolist()}")
            df = df.drop(columns=constant_features)
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups."""
        return self.feature_groups.copy()
    
    def get_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest'
    ) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to use
            
        Returns:
            DataFrame with feature importance
        """
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=42, eval_metric='logloss')
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return pd.DataFrame()
    
    def __repr__(self) -> str:
        """String representation."""
        total_features = sum(len(features) for features in self.feature_groups.values())
        return f"FeatureEngineer(total_features={total_features}, groups={list(self.feature_groups.keys())})"
