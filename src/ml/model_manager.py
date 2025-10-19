"""
Model Manager for ML Models

Manages multiple ML models across different regimes and timeframes:
- Model lifecycle management
- Performance monitoring
- Auto-retraining coordination
- Model selection and ensemble methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

from .freqai_helper import FreqAIHelper

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages multiple ML models across different regimes and timeframes.
    
    Provides centralized management of model training, evaluation,
    and selection for different market conditions.
    """
    
    def __init__(
        self,
        model_path: str = "user_data/models",
        retrain_frequency_days: int = 15,
        performance_window_days: int = 30,
        min_performance_threshold: float = 0.55
    ):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to store models
            retrain_frequency_days: Days between retraining
            performance_window_days: Days to look back for performance evaluation
            min_performance_threshold: Minimum accuracy threshold for model selection
        """
        self.model_path = Path(model_path)
        self.retrain_frequency_days = retrain_frequency_days
        self.performance_window_days = performance_window_days
        self.min_performance_threshold = min_performance_threshold
        
        # Initialize FreqAI helper
        self.freqai_helper = FreqAIHelper(
            model_path=model_path,
            retrain_frequency_days=retrain_frequency_days
        )
        
        # Model tracking
        self.active_models = {}
        self.model_history = []
        self.performance_tracking = {}
        
        # Ensemble settings
        self.ensemble_weights = {}
        self.ensemble_enabled = True
        
        logger.info("ModelManager initialized")
    
    def train_regime_model(
        self,
        dataframe: pd.DataFrame,
        regime: str,
        target_column: str = 'target_direction',
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Train or retrain model for specific regime.
        
        Args:
            dataframe: Training data
            regime: Market regime
            target_column: Target column name
            force_retrain: Force retraining even if not needed
            
        Returns:
            Training results
        """
        # Check if retraining is needed
        if not force_retrain and not self.freqai_helper.should_retrain(regime):
            logger.info(f"Model for {regime} is up to date, skipping retraining")
            return self.freqai_helper.get_model_info(regime)
        
        # Prepare features
        features_df = self.freqai_helper.prepare_features(dataframe, regime)
        
        if len(features_df) < self.freqai_helper.min_training_samples:
            logger.warning(
                f"Insufficient data for {regime}: {len(features_df)} samples "
                f"(minimum: {self.freqai_helper.min_training_samples})"
            )
            return None
        
        # Create training data
        X, y = self.freqai_helper.create_training_data(features_df, regime, target_column)
        
        # Train model
        training_result = self.freqai_helper.train_model(X, y, regime)
        
        if training_result:
            # Update active models
            self.active_models[regime] = {
                'model_name': training_result['model_name'],
                'trained_at': datetime.now(),
                'accuracy': training_result['accuracy'],
                'cv_score': training_result['cv_score']
            }
            
            # Add to history
            self.model_history.append({
                'regime': regime,
                'model_name': training_result['model_name'],
                'trained_at': datetime.now(),
                'accuracy': training_result['accuracy'],
                'cv_score': training_result['cv_score'],
                'training_samples': training_result['training_samples']
            })
            
            # Cleanup old models
            self.freqai_helper.cleanup_old_models(regime)
            
            logger.info(f"Successfully trained model for {regime}")
        
        return training_result
    
    def predict_regime(
        self,
        dataframe: pd.DataFrame,
        regime: str,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions for specific regime.
        
        Args:
            dataframe: Input data
            regime: Market regime
            return_confidence: Whether to return confidence scores
            
        Returns:
            Prediction results
        """
        if regime not in self.active_models:
            logger.warning(f"No active model for regime: {regime}")
            return {
                'predictions': np.zeros(len(dataframe)),
                'confidence': np.zeros(len(dataframe)),
                'model_name': None
            }
        
        # Prepare features
        features_df = self.freqai_helper.prepare_features(dataframe, regime)
        
        # Get feature columns for prediction
        model_info = self.freqai_helper.get_model_info(regime)
        if not model_info:
            logger.error(f"Model info not found for regime: {regime}")
            return {
                'predictions': np.zeros(len(dataframe)),
                'confidence': np.zeros(len(dataframe)),
                'model_name': None
            }
        
        # Select features
        feature_columns = model_info['features']
        X = features_df[feature_columns].copy()
        
        # Make predictions
        predictions = self.freqai_helper.predict(X, regime, return_probability=True)
        
        # Calculate confidence (distance from 0.5)
        if return_confidence:
            confidence = np.abs(predictions - 0.5) * 2  # Scale to 0-1
        else:
            confidence = np.ones_like(predictions)
        
        result = {
            'predictions': predictions,
            'confidence': confidence,
            'model_name': self.active_models[regime]['model_name'],
            'regime': regime,
            'model_accuracy': self.active_models[regime]['accuracy']
        }
        
        return result
    
    def predict_ensemble(
        self,
        dataframe: pd.DataFrame,
        regime: str,
        use_all_regimes: bool = False
    ) -> Dict[str, Any]:
        """
        Make ensemble predictions using multiple models.
        
        Args:
            dataframe: Input data
            regime: Primary regime
            use_all_regimes: Whether to use all available regimes
            
        Returns:
            Ensemble prediction results
        """
        if not self.ensemble_enabled:
            return self.predict_regime(dataframe, regime)
        
        # Get available regimes
        if use_all_regimes:
            regimes = list(self.active_models.keys())
        else:
            regimes = [regime] if regime in self.active_models else []
        
        if not regimes:
            logger.warning("No models available for ensemble prediction")
            return {
                'predictions': np.zeros(len(dataframe)),
                'confidence': np.zeros(len(dataframe)),
                'ensemble_weights': {},
                'individual_predictions': {}
            }
        
        # Get predictions from each regime
        individual_predictions = {}
        individual_confidences = {}
        
        for reg in regimes:
            pred_result = self.predict_regime(dataframe, reg)
            individual_predictions[reg] = pred_result['predictions']
            individual_confidences[reg] = pred_result['confidence']
        
        # Calculate ensemble weights
        weights = self._calculate_ensemble_weights(regimes, individual_confidences)
        
        # Find the minimum length among all predictions
        min_length = min(len(pred) for pred in individual_predictions.values())
        
        # Weighted average of predictions (truncate to minimum length)
        ensemble_predictions = np.zeros(min_length)
        ensemble_confidence = np.zeros(min_length)
        
        for reg in regimes:
            weight = weights.get(reg, 0.0)
            pred = individual_predictions[reg][:min_length]  # Truncate to min length
            conf = individual_confidences[reg][:min_length]  # Truncate to min length
            ensemble_predictions += weight * pred
            ensemble_confidence += weight * conf
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            ensemble_predictions /= total_weight
            ensemble_confidence /= total_weight
        
        result = {
            'predictions': ensemble_predictions,
            'confidence': ensemble_confidence,
            'ensemble_weights': weights,
            'individual_predictions': individual_predictions,
            'regimes_used': regimes
        }
        
        return result
    
    def _calculate_ensemble_weights(
        self,
        regimes: List[str],
        confidences: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate ensemble weights based on model performance and confidence.
        
        Args:
            regimes: List of regimes
            confidences: Confidence scores for each regime
            
        Returns:
            Weight dictionary
        """
        weights = {}
        
        for regime in regimes:
            if regime not in self.active_models:
                weights[regime] = 0.0
                continue
            
            # Base weight from model accuracy
            accuracy = self.active_models[regime]['accuracy']
            base_weight = max(0.0, accuracy - self.min_performance_threshold)
            
            # Confidence adjustment
            avg_confidence = np.mean(confidences[regime])
            confidence_weight = avg_confidence
            
            # Combined weight
            weights[regime] = base_weight * confidence_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def update_performance(
        self,
        regime: str,
        predictions: np.ndarray,
        actual: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> None:
        """
        Update model performance tracking.
        
        Args:
            regime: Market regime
            predictions: Model predictions
            actual: Actual values
            timestamps: Prediction timestamps
        """
        if timestamps is None:
            timestamps = [datetime.now()] * len(predictions)
        
        # Calculate performance metrics
        accuracy = np.mean(predictions == actual)
        mse = np.mean((predictions - actual) ** 2)
        
        # Update performance tracking
        if regime not in self.performance_tracking:
            self.performance_tracking[regime] = []
        
        self.performance_tracking[regime].append({
            'timestamp': timestamps[0] if timestamps else datetime.now(),
            'accuracy': accuracy,
            'mse': mse,
            'sample_count': len(predictions)
        })
        
        # Keep only recent performance data
        cutoff_date = datetime.now() - timedelta(days=self.performance_window_days)
        self.performance_tracking[regime] = [
            perf for perf in self.performance_tracking[regime]
            if perf['timestamp'] >= cutoff_date
        ]
        
        logger.debug(f"Updated performance for {regime}: accuracy={accuracy:.3f}")
    
    def get_model_performance(self, regime: str) -> Dict[str, Any]:
        """
        Get performance metrics for a regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Performance metrics
        """
        if regime not in self.performance_tracking:
            return {'regime': regime, 'samples': 0, 'avg_accuracy': 0.0}
        
        perf_data = self.performance_tracking[regime]
        
        if not perf_data:
            return {'regime': regime, 'samples': 0, 'avg_accuracy': 0.0}
        
        # Calculate aggregate metrics
        total_samples = sum(perf['sample_count'] for perf in perf_data)
        avg_accuracy = np.mean([perf['accuracy'] for perf in perf_data])
        avg_mse = np.mean([perf['mse'] for perf in perf_data])
        
        return {
            'regime': regime,
            'samples': total_samples,
            'avg_accuracy': avg_accuracy,
            'avg_mse': avg_mse,
            'recent_performance': perf_data[-5:] if len(perf_data) >= 5 else perf_data
        }
    
    def get_all_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all regimes."""
        return {
            regime: self.get_model_performance(regime)
            for regime in self.active_models.keys()
        }
    
    def should_retrain_any(self) -> List[str]:
        """
        Check which regimes need retraining.
        
        Returns:
            List of regimes that need retraining
        """
        regimes_to_retrain = []
        
        for regime in self.active_models.keys():
            if self.freqai_helper.should_retrain(regime):
                regimes_to_retrain.append(regime)
        
        return regimes_to_retrain
    
    def retrain_all_needed(
        self,
        data_by_regime: Dict[str, pd.DataFrame],
        target_column: str = 'target_direction'
    ) -> Dict[str, Any]:
        """
        Retrain all models that need retraining.
        
        Args:
            data_by_regime: Data dictionary by regime
            target_column: Target column name
            
        Returns:
            Retraining results
        """
        regimes_to_retrain = self.should_retrain_any()
        results = {}
        
        for regime in regimes_to_retrain:
            if regime in data_by_regime:
                logger.info(f"Retraining model for regime: {regime}")
                result = self.train_regime_model(
                    data_by_regime[regime], regime, target_column
                )
                results[regime] = result
            else:
                logger.warning(f"No data available for regime: {regime}")
                results[regime] = None
        
        return results
    
    def save_state(self, filepath: Optional[str] = None) -> None:
        """
        Save model manager state to file.
        
        Args:
            filepath: Optional custom filepath
        """
        if filepath is None:
            filepath = self.model_path / "model_manager_state.json"
        else:
            filepath = Path(filepath)
        
        state = {
            'active_models': {
                regime: {
                    'model_name': info['model_name'],
                    'trained_at': info['trained_at'].isoformat(),
                    'accuracy': info['accuracy'],
                    'cv_score': info['cv_score']
                }
                for regime, info in self.active_models.items()
            },
            'model_history': [
                {
                    'regime': hist['regime'],
                    'model_name': hist['model_name'],
                    'trained_at': hist['trained_at'].isoformat(),
                    'accuracy': hist['accuracy'],
                    'cv_score': hist['cv_score'],
                    'training_samples': hist['training_samples']
                }
                for hist in self.model_history
            ],
            'performance_tracking': {
                regime: [
                    {
                        'timestamp': perf['timestamp'].isoformat(),
                        'accuracy': perf['accuracy'],
                        'mse': perf['mse'],
                        'sample_count': perf['sample_count']
                    }
                    for perf in perf_list
                ]
                for regime, perf_list in self.performance_tracking.items()
            },
            'ensemble_weights': self.ensemble_weights,
            'ensemble_enabled': self.ensemble_enabled
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved model manager state to: {filepath}")
    
    def load_state(self, filepath: Optional[str] = None) -> bool:
        """
        Load model manager state from file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            True if loaded successfully
        """
        if filepath is None:
            filepath = self.model_path / "model_manager_state.json"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"State file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Load active models
            self.active_models = {}
            for regime, info in state.get('active_models', {}).items():
                self.active_models[regime] = {
                    'model_name': info['model_name'],
                    'trained_at': datetime.fromisoformat(info['trained_at']),
                    'accuracy': info['accuracy'],
                    'cv_score': info['cv_score']
                }
            
            # Load model history
            self.model_history = []
            for hist in state.get('model_history', []):
                self.model_history.append({
                    'regime': hist['regime'],
                    'model_name': hist['model_name'],
                    'trained_at': datetime.fromisoformat(hist['trained_at']),
                    'accuracy': hist['accuracy'],
                    'cv_score': hist['cv_score'],
                    'training_samples': hist['training_samples']
                })
            
            # Load performance tracking
            self.performance_tracking = {}
            for regime, perf_list in state.get('performance_tracking', {}).items():
                self.performance_tracking[regime] = [
                    {
                        'timestamp': datetime.fromisoformat(perf['timestamp']),
                        'accuracy': perf['accuracy'],
                        'mse': perf['mse'],
                        'sample_count': perf['sample_count']
                    }
                    for perf in perf_list
                ]
            
            # Load ensemble settings
            self.ensemble_weights = state.get('ensemble_weights', {})
            self.ensemble_enabled = state.get('ensemble_enabled', True)
            
            logger.info(f"Loaded model manager state from: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state from {filepath}: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ModelManager(active_models={len(self.active_models)}, ensemble_enabled={self.ensemble_enabled})"
