"""
Enhanced HMM-based Market Regime Detector (v2)

Implements best practices from academic research and community consensus:
- Dynamic state labeling based on learned characteristics
- Proper training window sizing (3000-10000 candles)
- Enhanced feature set with volatility, skewness, kurtosis
- Model validation and convergence checks
- Transition matrix analysis

References:
- Hamilton (1989) - Economic regime switching
- Kritzman et al. (2012) - Regime shifts
- Nystrup et al. (2015) - Regime-based allocation
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class EnhancedRegimeDetector:
    """
    Enhanced market regime detector using HMM with best practices.
    
    Key Improvements:
    - Dynamic state labeling (not pre-assigned)
    - Larger training windows (3000-10000 candles)
    - Model validation checks
    - Enhanced feature set
    - Transition matrix analysis
    """
    
    def __init__(self,
                 n_states: int = 3,
                 covariance_type: str = 'full',
                 n_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize Enhanced Regime Detector.
        
        Args:
            n_states: Number of hidden states (2-4 recommended)
            covariance_type: 'full', 'diag', 'tied', or 'spherical'
            n_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.random_state = random_state
        
        # Initialize HMM
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Training status
        self.is_trained = False
        
        # Regime mapping (determined after training)
        self.regime_mapping = {}
        self.state_profiles = {}
        
    def prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced feature set based on research.
        
        Features (10-12 total):
        1-3. Returns (1, 5, 20 periods)
        4-5. Volatility (20, 60 periods)
        6. High-Low range
        7-8. Volume features
        9. Trend strength (ADX)
        10-11. Higher moments (skew, kurt)
        
        Args:
            dataframe: OHLCV data
            
        Returns:
            DataFrame with feature columns
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        df = dataframe.copy()
        features = pd.DataFrame(index=df.index)
        
        # 1-3. Returns (multiple horizons)
        features['returns_1'] = np.log(df['close'] / df['close'].shift(1))
        features['returns_5'] = np.log(df['close'] / df['close'].shift(5))
        features['returns_20'] = np.log(df['close'] / df['close'].shift(20))
        
        # 4-5. Volatility (multiple horizons)
        features['volatility_20'] = features['returns_1'].rolling(20).std()
        features['volatility_60'] = features['returns_1'].rolling(60).std()
        
        # 6. High-Low range (normalized)
        features['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # 7. Volume ratio
        volume_ma = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / volume_ma
        
        # 8. Volume volatility
        features['volume_volatility'] = (
            df['volume'].rolling(20).std() / volume_ma
        )
        
        # 9. Trend strength (simplified ADX calculation)
        # Using ATR-based trend measure
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features['trend_strength'] = atr / df['close']
        
        # 10-11. Higher moments (skewness, kurtosis)
        features['return_skew'] = features['returns_1'].rolling(20).skew()
        features['return_kurt'] = features['returns_1'].rolling(20).kurt()
        
        # Drop NaN rows
        features_clean = features.dropna()
        
        logger.info(f"Prepared {len(features_clean)} samples "
                   f"with {len(features_clean.columns)} features")
        
        return features_clean
    
    def train(self,
              dataframe: pd.DataFrame,
              lookback: int = 5000) -> bool:
        """
        Train HMM on historical data.
        
        Args:
            dataframe: OHLCV data
            lookback: Number of candles for training (3000-10000 recommended)
            
        Returns:
            True if training successful
        """
        # Use most recent lookback candles
        df_train = dataframe.iloc[-lookback:].copy() if len(dataframe) > lookback else dataframe.copy()
        
        logger.info(f"Training HMM on {len(df_train)} candles...")
        
        # Prepare features
        features_df = self.prepare_features(df_train)
        
        if len(features_df) < 100:
            logger.warning(f"Insufficient training data: {len(features_df)} samples")
            return False
        
        # Convert to numpy array
        features_array = features_df.values
        
        # Fit scaler on training data
        features_scaled = self.scaler.fit_transform(features_array)
        
        # Train HMM
        try:
            self.model.fit(features_scaled)
            self.is_trained = True
            
            # Validate model
            self._validate_model(features_scaled, features_df)
            
            # Assign regime labels based on characteristics
            self._assign_regime_labels(features_scaled, features_df)
            
            logger.info("HMM training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            return False
    
    def _validate_model(self,
                       features_scaled: np.ndarray,
                       features_df: pd.DataFrame):
        """
        Validate HMM training quality.
        
        Checks:
        1. Convergence status
        2. State persistence (diagonal of transition matrix)
        3. State distribution balance
        4. Log-likelihood score
        """
        # 1. Convergence
        converged = self.model.monitor_.converged
        n_iter = self.model.monitor_.iter
        
        if not converged:
            logger.warning(f"HMM did not converge! Iterations: {n_iter}/{self.model.n_iter}")
        else:
            logger.info(f"HMM converged in {n_iter} iterations")
        
        # 2. State persistence
        trans_matrix = self.model.transitionprob_
        diag = np.diag(trans_matrix)
        persistence = diag.mean()
        
        if persistence < 0.5:
            logger.warning(f"Low state persistence (avg={persistence:.2f}). "
                          f"States may be unstable.")
        else:
            logger.info(f"State persistence: {diag}")
        
        # 3. State distribution
        states = self.model.predict(features_scaled)
        state_counts = np.bincount(states, minlength=self.n_states)
        state_dist = state_counts / len(states)
        
        if state_dist.max() > 0.85:
            logger.warning(f"Highly skewed state distribution: {state_dist}. "
                          f"One state dominates!")
        else:
            logger.info(f"State distribution: {state_dist}")
        
        # 4. Log-likelihood
        score = self.model.score(features_scaled)
        logger.info(f"Model log-likelihood: {score:.2f}")
        
        # Store validation metrics
        self.validation_metrics = {
            'converged': converged,
            'iterations': n_iter,
            'persistence': persistence,
            'state_distribution': state_dist,
            'log_likelihood': score
        }
    
    def _assign_regime_labels(self,
                             features_scaled: np.ndarray,
                             features_df: pd.DataFrame):
        """
        Assign meaningful labels to states based on learned characteristics.
        
        This is the CORRECT way to label HMM states:
        1. Predict states for training data
        2. Analyze characteristics of each state
        3. Assign labels based on volatility/return patterns
        """
        # Predict states
        states = self.model.predict(features_scaled)
        
        # Calculate characteristics for each state
        state_profiles = {}
        
        for state in range(self.n_states):
            mask = (states == state)
            state_data = features_df[mask]
            
            if len(state_data) == 0:
                logger.warning(f"State {state} has no samples!")
                continue
            
            profile = {
                'count': mask.sum(),
                'percentage': mask.sum() / len(states) * 100,
                'avg_return_1': state_data['returns_1'].mean(),
                'avg_return_20': state_data['returns_20'].mean(),
                'avg_volatility': state_data['volatility_20'].mean(),
                'avg_volume_ratio': state_data['volume_ratio'].mean(),
                'avg_skew': state_data['return_skew'].mean(),
            }
            
            state_profiles[state] = profile
            
            logger.info(f"State {state}: {mask.sum()} samples "
                       f"({profile['percentage']:.1f}%), "
                       f"vol={profile['avg_volatility']:.4f}, "
                       f"ret={profile['avg_return_1']:.4f}")
        
        # Sort states by volatility
        sorted_by_vol = sorted(
            state_profiles.items(),
            key=lambda x: x[1]['avg_volatility']
        )
        
        # Assign labels based on volatility levels
        if len(sorted_by_vol) >= 3:
            # 3-state model
            low_vol_state = sorted_by_vol[0][0]
            mid_vol_state = sorted_by_vol[1][0]
            high_vol_state = sorted_by_vol[2][0]
            
            self.regime_mapping[low_vol_state] = 'low_volatility'
            self.regime_mapping[high_vol_state] = 'high_volatility'
            
            # Mid volatility - check trend direction
            mid_return = state_profiles[mid_vol_state]['avg_return_1']
            if abs(mid_return) < 0.0001:
                self.regime_mapping[mid_vol_state] = 'ranging'
            elif mid_return > 0:
                self.regime_mapping[mid_vol_state] = 'trending_up'
            else:
                self.regime_mapping[mid_vol_state] = 'trending_down'
        
        elif len(sorted_by_vol) == 2:
            # 2-state model
            low_vol_state = sorted_by_vol[0][0]
            high_vol_state = sorted_by_vol[1][0]
            
            self.regime_mapping[low_vol_state] = 'low_volatility'
            self.regime_mapping[high_vol_state] = 'high_volatility'
        
        # Store profiles
        self.state_profiles = state_profiles
        
        logger.info(f"Regime mapping: {self.regime_mapping}")
    
    def predict_regime(self,
                      dataframe: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict current market regime.
        
        Args:
            dataframe: Recent OHLCV data (at least 100 candles)
            
        Returns:
            (regime_name, confidence)
        """
        if not self.is_trained:
            return 'unknown', 0.0
        
        # Prepare features
        features_df = self.prepare_features(dataframe)
        
        if len(features_df) == 0:
            return 'unknown', 0.0
        
        # Use most recent sample
        features_array = features_df.iloc[-1:].values
        features_scaled = self.scaler.transform(features_array)
        
        # Predict state
        state = self.model.predict(features_scaled)[0]
        
        # Get confidence (probability of predicted state)
        probs = self.model.predict_proba(features_scaled)[0]
        confidence = probs[state]
        
        # Map to regime name
        regime_name = self.regime_mapping.get(state, f'regime_{state}')
        
        return regime_name, float(confidence)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get transition probability matrix."""
        if not self.is_trained:
            return None
        return self.model.transitionprob_
    
    def get_state_profiles(self) -> Dict:
        """Get detailed state characteristic profiles."""
        return self.state_profiles
