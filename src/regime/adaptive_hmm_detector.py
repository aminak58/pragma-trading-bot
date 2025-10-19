#!/usr/bin/env python3
"""
Adaptive HMM Detector with online retraining capabilities
"""
import logging
from typing import Tuple, Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from .hmm_detector import RegimeDetector

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor HMM performance for adaptive retraining decisions"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.confidence_history = []
        self.regime_changes = []
        self.performance_metrics = []
        
    def add_prediction(self, confidence: float, regime: str, previous_regime: Optional[str] = None):
        """Add a new prediction to monitoring"""
        self.confidence_history.append(confidence)
        
        if previous_regime and previous_regime != regime:
            self.regime_changes.append(1)
        else:
            self.regime_changes.append(0)
    
    def should_retrain(self, min_confidence: float = 0.8, max_change_rate: float = 0.1) -> bool:
        """Determine if retraining is needed based on performance metrics"""
        if len(self.confidence_history) < self.window_size:
            return False
        
        # Check recent confidence
        recent_confidences = self.confidence_history[-self.window_size:]
        avg_confidence = np.mean(recent_confidences)
        
        # Check regime change rate
        recent_changes = self.regime_changes[-self.window_size:]
        change_rate = np.mean(recent_changes)
        
        # Retrain if confidence is low or change rate is high
        should_retrain = (avg_confidence < min_confidence) or (change_rate > max_change_rate)
        
        if should_retrain:
            logger.info(f"Retraining triggered: confidence={avg_confidence:.3f}, change_rate={change_rate:.3f}")
        
        return should_retrain
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance summary"""
        if not self.confidence_history:
            return {}
        
        recent_confidences = self.confidence_history[-self.window_size:]
        recent_changes = self.regime_changes[-self.window_size:]
        
        return {
            'avg_confidence': np.mean(recent_confidences),
            'min_confidence': np.min(recent_confidences),
            'change_rate': np.mean(recent_changes),
            'total_predictions': len(self.confidence_history)
        }

class AdaptiveHMMDetector(RegimeDetector):
    """
    Enhanced HMM detector with adaptive retraining capabilities.
    
    Features:
    - Online retraining based on performance metrics
    - Performance monitoring and degradation detection
    - Configurable retraining intervals and triggers
    - Model versioning and rollback capabilities
    """
    
    def __init__(self, 
                 n_states: int = 3, 
                 random_state: int = 42,
                 retrain_interval_hours: int = 6,
                 min_confidence_threshold: float = 0.8,
                 max_change_rate_threshold: float = 0.1,
                 performance_window: int = 100):
        """
        Initialize Adaptive HMM Detector.
        
        Args:
            n_states: Number of hidden states
            random_state: Random seed for reproducibility
            retrain_interval_hours: Hours between retraining attempts
            min_confidence_threshold: Minimum confidence for good performance
            max_change_rate_threshold: Maximum regime change rate for good performance
            performance_window: Window size for performance monitoring
        """
        super().__init__(n_states, random_state)
        
        # Adaptive retraining parameters
        self.retrain_interval_hours = retrain_interval_hours
        self.min_confidence_threshold = min_confidence_threshold
        self.max_change_rate_threshold = max_change_rate_threshold
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(performance_window)
        
        # Retraining state
        self.last_retrain = None
        self.retrain_count = 0
        self.model_versions = []
        
        # Current prediction state
        self.last_regime = None
        self.last_confidence = None
        
    def should_retrain_now(self) -> bool:
        """Check if retraining should be performed now"""
        # Check time-based retraining
        if self.last_retrain is None:
            return True
        
        time_since_retrain = datetime.now() - self.last_retrain
        time_based = time_since_retrain >= timedelta(hours=self.retrain_interval_hours)
        
        # Check performance-based retraining
        performance_based = self.performance_monitor.should_retrain(
            self.min_confidence_threshold,
            self.max_change_rate_threshold
        )
        
        return time_based or performance_based
    
    def adaptive_retrain(self, dataframe: pd.DataFrame, lookback: int = 1000) -> bool:
        """
        Perform adaptive retraining if conditions are met.
        
        Args:
            dataframe: OHLCV dataframe for retraining
            lookback: Number of recent candles to use for training
            
        Returns:
            True if retraining was performed, False otherwise
        """
        if not self.should_retrain_now():
            return False
        
        logger.info(f"Starting adaptive retraining (attempt #{self.retrain_count + 1})")
        
        try:
            # Store current model as backup
            if self.is_trained:
                self.model_versions.append({
                    'model': self.model,
                    'scaler': self.scaler,
                    'regime_names': self.regime_names.copy(),
                    'timestamp': datetime.now(),
                    'performance': self.performance_monitor.get_performance_summary()
                })
                
                # Keep only last 3 versions
                if len(self.model_versions) > 3:
                    self.model_versions.pop(0)
            
            # Perform retraining
            self.train(dataframe, lookback=lookback)
            
            # Update retraining state
            self.last_retrain = datetime.now()
            self.retrain_count += 1
            
            # Reset performance monitoring
            self.performance_monitor = PerformanceMonitor()
            
            logger.info(f"Adaptive retraining completed successfully (version {self.retrain_count})")
            return True
            
        except Exception as e:
            logger.error(f"Adaptive retraining failed: {e}")
            
            # Rollback to previous model if available
            if self.model_versions:
                logger.info("Rolling back to previous model version")
                latest_version = self.model_versions[-1]
                self.model = latest_version['model']
                self.scaler = latest_version['scaler']
                self.regime_names = latest_version['regime_names']
                self.is_trained = True
            
            return False
    
    def predict_regime_adaptive(self, dataframe: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict regime with adaptive retraining.
        
        Args:
            dataframe: OHLCV dataframe
            
        Returns:
            Tuple of (regime_name, confidence_score)
        """
        # Check if retraining is needed
        if self.is_trained and len(dataframe) >= 1000:
            self.adaptive_retrain(dataframe, lookback=1000)
        
        # Perform prediction
        regime, confidence = self.predict_regime(dataframe)
        
        # Update performance monitoring
        self.performance_monitor.add_prediction(
            confidence, 
            regime, 
            self.last_regime
        )
        
        # Update state
        self.last_regime = regime
        self.last_confidence = confidence
        
        return regime, confidence
    
    def predict_regime_sequence_adaptive(self, dataframe: pd.DataFrame, smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime sequence with adaptive retraining.
        
        Args:
            dataframe: OHLCV dataframe
            smooth_window: Window size for smoothing regime transitions
            
        Returns:
            Tuple of (regime_sequence, confidence_sequence)
        """
        # Check if retraining is needed
        if self.is_trained and len(dataframe) >= 1000:
            self.adaptive_retrain(dataframe, lookback=1000)
        
        # Perform sequence prediction
        regime_sequence, confidence_sequence = self.predict_regime_sequence(dataframe, smooth_window)
        
        # Update performance monitoring for each prediction
        for i, (regime, confidence) in enumerate(zip(regime_sequence, confidence_sequence)):
            previous_regime = self.last_regime if i == 0 else regime_sequence[i-1]
            self.performance_monitor.add_prediction(confidence, regime, previous_regime)
        
        # Update state
        if len(regime_sequence) > 0:
            self.last_regime = regime_sequence[-1]
            self.last_confidence = confidence_sequence[-1]
        
        return regime_sequence, confidence_sequence
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status and performance metrics"""
        performance = self.performance_monitor.get_performance_summary()
        
        return {
            'is_trained': self.is_trained,
            'retrain_count': self.retrain_count,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'next_retrain_due': (self.last_retrain + timedelta(hours=self.retrain_interval_hours)).isoformat() if self.last_retrain else None,
            'should_retrain': self.should_retrain_now(),
            'performance': performance,
            'model_versions': len(self.model_versions),
            'current_regime': self.last_regime,
            'current_confidence': self.last_confidence
        }
    
    def force_retrain(self, dataframe: pd.DataFrame, lookback: int = 1000) -> bool:
        """Force immediate retraining regardless of conditions"""
        logger.info("Forcing immediate retraining")
        self.last_retrain = None  # Reset to force retraining
        return self.adaptive_retrain(dataframe, lookback)
    
    def rollback_model(self, version_index: int = -1) -> bool:
        """Rollback to a previous model version"""
        if not self.model_versions or abs(version_index) > len(self.model_versions):
            logger.error(f"Cannot rollback: version {version_index} not available")
            return False
        
        try:
            version = self.model_versions[version_index]
            self.model = version['model']
            self.scaler = version['scaler']
            self.regime_names = version['regime_names'].copy()
            self.is_trained = True
            
            logger.info(f"Rolled back to model version from {version['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
