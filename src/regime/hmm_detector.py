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
            n_states: Number of hidden states (default: 3)
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        
        # Initialize HMM with full covariance
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_state,
            verbose=False
        )
        
        # Feature scaler for normalization
        self.scaler = StandardScaler()
        
        # Training status
        self.is_trained = False
        
        # Regime names (will be determined after training based on characteristics)
        self.regime_names = {
            0: "regime_0",
            1: "regime_1", 
            2: "regime_2"
        }
        
    def prepare_features(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix from OHLCV dataframe.
        
        Features calculated:
        1. Returns: Log returns over different periods
        2. Volatility: Rolling standard deviation of returns
        3. Volume Ratio: Current volume / average volume
        4. Trend Strength: ADX or similar momentum indicator
        
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
        
        # Feature 1: Returns (multiple timeframes)
        df['returns_1'] = np.log(df['close'] / df['close'].shift(1))
        df['returns_5'] = np.log(df['close'] / df['close'].shift(5))
        df['returns_20'] = np.log(df['close'] / df['close'].shift(20))
        features.extend(['returns_1', 'returns_5', 'returns_20'])
        
        # Feature 2: Volatility (rolling std of returns)
        df['volatility_10'] = df['returns_1'].rolling(window=10).std()
        df['volatility_30'] = df['returns_1'].rolling(window=30).std()
        features.extend(['volatility_10', 'volatility_30'])
        
        # Feature 3: Volume ratio (current vs average)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        features.append('volume_ratio')
        
        # Feature 4: Trend strength (simple ADX approximation)
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'], 0
        )
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'], 0
        )
        
        # Smoothed indicators (14-period)
        period = 14
        df['tr_smooth'] = df['tr'].rolling(window=period).mean()
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr_smooth'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr_smooth'])
        
        # ADX approximation
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        features.append('adx')
        
        # Extract feature matrix and handle NaN values
        feature_matrix = df[features].values
        
        # Fill NaN with forward fill then backward fill
        feature_df = pd.DataFrame(feature_matrix, columns=features)
        feature_df = feature_df.ffill().bfill()
        
        # If still NaN (empty dataframe), fill with zeros
        feature_df = feature_df.fillna(0)
        
        return feature_df.values
    
    def train(self, dataframe: pd.DataFrame, lookback: int = 500) -> 'RegimeDetector':
        """
        Train the HMM on historical data.
        
        Args:
            dataframe: Historical OHLCV dataframe
            lookback: Number of most recent candles to use for training
            
        Returns:
            Self (for method chaining)
            
        Raises:
            ValueError: If insufficient data for training
        """
        if len(dataframe) < 100:
            raise ValueError(f"Insufficient data: need at least 100 candles, got {len(dataframe)}")
        
        # Use most recent lookback candles
        df_train = dataframe.tail(lookback).copy()
        
        # Prepare features
        X = self.prepare_features(df_train)
        
        if len(X) < 50:
            raise ValueError(f"Insufficient valid samples after feature preparation: {len(X)}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train HMM
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Determine regime names based on characteristics
        self._assign_regime_names(X_scaled)
        
        return self
    
    def _assign_regime_names(self, X_scaled: np.ndarray) -> None:
        """
        Assign meaningful names to regimes based on their characteristics.
        
        Analyzes the mean feature values for each state to determine:
        - High volatility regime (high volatility features)
        - Low volatility regime (low volatility features)
        - Trending regime (high ADX values)
        
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
                
                # Features: [returns_1, returns_5, returns_20, vol_10, vol_30, vol_ratio, adx]
                volatility = (state_mean[3] + state_mean[4]) / 2  # avg of vol features
                trend_strength = state_mean[6]  # ADX
                
                state_characteristics[state] = {
                    'volatility': volatility,
                    'trend': trend_strength
                }
        
        # Sort states by characteristics
        states_by_vol = sorted(state_characteristics.items(), 
                              key=lambda x: x[1]['volatility'])
        states_by_trend = sorted(state_characteristics.items(),
                                key=lambda x: x[1]['trend'])
        
        # Assign names based on characteristics
        # Highest volatility = high_volatility
        # Highest trend = trending
        # Remaining = low_volatility
        
        high_vol_state = states_by_vol[-1][0]
        high_trend_state = states_by_trend[-1][0]
        
        self.regime_names[high_vol_state] = "high_volatility"
        self.regime_names[high_trend_state] = "trending"
        
        # Find remaining state for low_volatility
        for state in range(self.n_states):
            if state not in [high_vol_state, high_trend_state]:
                self.regime_names[state] = "low_volatility"
                break
        
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
