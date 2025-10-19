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
from sklearn.preprocessing import StandardScaler
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
        scaler (StandardScaler): Feature scaler for normalization
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
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        
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
    
    def predict_regime_sequence(self, dataframe: pd.DataFrame, smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict market regimes for entire dataframe with smoothing.
        
        Args:
            dataframe: OHLCV dataframe
            smooth_window: Window size for smoothing regime transitions
            
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
