"""
Unit tests for HMM Regime Detector

Tests cover:
- Initialization
- Feature preparation
- Model training
- Regime prediction
- Error handling
- Edge cases
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
import pandas as pd
from regime.hmm_detector import RegimeDetector


# Fixtures
@pytest.fixture
def sample_dataframe():
    """Generate sample OHLCV dataframe for testing."""
    np.random.seed(42)
    n = 1000
    
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(15, 0.5, n)
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def small_dataframe():
    """Generate small dataframe for edge case testing."""
    np.random.seed(42)
    n = 50
    
    close = 100 + np.cumsum(np.random.randn(n))
    
    return pd.DataFrame({
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'volume': np.random.rand(n) * 1000
    })


@pytest.fixture
def detector():
    """Create fresh RegimeDetector instance."""
    return RegimeDetector(n_states=3, random_state=42)


@pytest.fixture
def trained_detector(detector, sample_dataframe):
    """Create trained detector."""
    detector.train(sample_dataframe, lookback=500)
    return detector


# Test Class
class TestRegimeDetectorInitialization:
    """Test detector initialization."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        detector = RegimeDetector()
        
        assert detector.n_states == 3
        assert detector.random_state == 42
        assert detector.is_trained is False
        assert detector.model is not None
        assert detector.scaler is not None
    
    def test_custom_states(self):
        """Test initialization with custom state count."""
        detector = RegimeDetector(n_states=4)
        assert detector.n_states == 4
        assert len(detector.regime_names) == 4
    
    def test_custom_random_state(self):
        """Test initialization with custom random state."""
        detector = RegimeDetector(random_state=123)
        assert detector.random_state == 123
    
    def test_repr(self):
        """Test string representation."""
        detector = RegimeDetector()
        assert "untrained" in repr(detector)
        assert "n_states=3" in repr(detector)


class TestFeaturePreparation:
    """Test feature preparation methods."""
    
    def test_prepare_features_shape(self, detector, sample_dataframe):
        """Test feature matrix has correct shape."""
        features = detector.prepare_features(sample_dataframe)
        
        assert features.shape[0] == len(sample_dataframe)
        assert features.shape[1] == 7  # 7 features
    
    def test_prepare_features_no_nan(self, detector, sample_dataframe):
        """Test that features have no NaN values."""
        features = detector.prepare_features(sample_dataframe)
        assert not np.isnan(features).any()
    
    def test_prepare_features_missing_columns(self, detector):
        """Test error handling for missing columns."""
        df = pd.DataFrame({'close': [100, 101, 102]})
        
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.prepare_features(df)
    
    def test_prepare_features_deterministic(self, detector, sample_dataframe):
        """Test feature preparation is deterministic."""
        features1 = detector.prepare_features(sample_dataframe)
        features2 = detector.prepare_features(sample_dataframe)
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_feature_types(self, detector, sample_dataframe):
        """Test that features are numeric."""
        features = detector.prepare_features(sample_dataframe)
        assert features.dtype in [np.float64, np.float32]


class TestModelTraining:
    """Test model training functionality."""
    
    def test_train_successful(self, detector, sample_dataframe):
        """Test successful training."""
        result = detector.train(sample_dataframe, lookback=500)
        
        assert detector.is_trained is True
        assert result is detector  # Check method chaining
    
    def test_train_insufficient_data(self, detector):
        """Test error handling for insufficient data."""
        df = pd.DataFrame({
            'open': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.train(df)
    
    def test_train_regime_names_assigned(self, detector, sample_dataframe):
        """Test that regime names are assigned after training."""
        detector.train(sample_dataframe)
        
        regime_names = set(detector.regime_names.values())
        
        # Should have meaningful names (not default regime_0, etc)
        assert len(regime_names) == 3
        # At least one should be a known regime type
        known_regimes = {
            'low_volatility', 'high_volatility', 'trending',
            'medium_volatility'
        }
        assert len(regime_names & known_regimes) >= 2
    
    def test_train_lookback(self, detector, sample_dataframe):
        """Test training with different lookback windows."""
        detector.train(sample_dataframe, lookback=300)
        assert detector.is_trained is True
        
        detector2 = RegimeDetector(random_state=42)
        detector2.train(sample_dataframe, lookback=700)
        assert detector2.is_trained is True
    
    def test_train_scaler_fitted(self, detector, sample_dataframe):
        """Test that scaler is fitted during training."""
        detector.train(sample_dataframe)
        
        # Scaler should have learned mean and std
        assert hasattr(detector.scaler, 'mean_')
        assert hasattr(detector.scaler, 'scale_')
        assert len(detector.scaler.mean_) == 7


class TestRegimePrediction:
    """Test regime prediction functionality."""
    
    def test_predict_before_training(self, detector, sample_dataframe):
        """Test error when predicting before training."""
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.predict_regime(sample_dataframe)
    
    def test_predict_returns_tuple(self, trained_detector, sample_dataframe):
        """Test prediction returns (regime, confidence) tuple."""
        result = trained_detector.predict_regime(sample_dataframe)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_predict_regime_name(self, trained_detector, sample_dataframe):
        """Test predicted regime is a valid name."""
        regime, confidence = trained_detector.predict_regime(sample_dataframe)
        
        assert isinstance(regime, str)
        assert regime in trained_detector.regime_names.values()
    
    def test_predict_confidence_range(self, trained_detector, sample_dataframe):
        """Test confidence score is in valid range [0, 1]."""
        regime, confidence = trained_detector.predict_regime(sample_dataframe)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_deterministic(self, trained_detector, sample_dataframe):
        """Test predictions are deterministic."""
        regime1, conf1 = trained_detector.predict_regime(sample_dataframe)
        regime2, conf2 = trained_detector.predict_regime(sample_dataframe)
        
        assert regime1 == regime2
        assert conf1 == conf2
    
    def test_predict_different_data(self, trained_detector):
        """Test prediction on different market conditions."""
        # Create trending data
        trending_data = pd.DataFrame({
            'open': np.arange(100, 200, 1),
            'high': np.arange(101, 201, 1),
            'low': np.arange(99, 199, 1),
            'close': np.arange(100, 200, 1),
            'volume': [1000] * 100
        })
        
        regime, conf = trained_detector.predict_regime(trending_data)
        assert isinstance(regime, str)
        assert 0.0 <= conf <= 1.0


class TestRegimeProbabilities:
    """Test probability distribution methods."""
    
    def test_get_probabilities_before_training(self, detector, sample_dataframe):
        """Test error when getting probabilities before training."""
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.get_regime_probabilities(sample_dataframe)
    
    def test_probabilities_sum_to_one(self, trained_detector, sample_dataframe):
        """Test that probabilities sum to 1.0."""
        probs = trained_detector.get_regime_probabilities(sample_dataframe)
        
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6
    
    def test_probabilities_all_regimes(self, trained_detector, sample_dataframe):
        """Test probabilities returned for all regimes."""
        probs = trained_detector.get_regime_probabilities(sample_dataframe)
        
        assert len(probs) == trained_detector.n_states
        
        # Check all regime names are present
        regime_names = set(trained_detector.regime_names.values())
        assert set(probs.keys()) == regime_names
    
    def test_probabilities_values_valid(self, trained_detector, sample_dataframe):
        """Test all probability values are in [0, 1]."""
        probs = trained_detector.get_regime_probabilities(sample_dataframe)
        
        for regime, prob in probs.items():
            assert 0.0 <= prob <= 1.0
    
    def test_highest_probability_matches_prediction(
        self, trained_detector, sample_dataframe
    ):
        """Test that highest probability regime matches prediction."""
        regime, _ = trained_detector.predict_regime(sample_dataframe)
        probs = trained_detector.get_regime_probabilities(sample_dataframe)
        
        highest_prob_regime = max(probs.items(), key=lambda x: x[1])[0]
        assert regime == highest_prob_regime


class TestTransitionMatrix:
    """Test state transition matrix."""
    
    def test_get_transition_before_training(self, detector):
        """Test error when accessing transition matrix before training."""
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.get_transition_matrix()
    
    def test_transition_matrix_shape(self, trained_detector):
        """Test transition matrix has correct shape."""
        trans = trained_detector.get_transition_matrix()
        
        n = trained_detector.n_states
        assert trans.shape == (n, n)
    
    def test_transition_matrix_probabilities(self, trained_detector):
        """Test transition matrix rows sum to 1."""
        trans = trained_detector.get_transition_matrix()
        
        for row in trans:
            assert abs(row.sum() - 1.0) < 1e-6
    
    def test_transition_matrix_values(self, trained_detector):
        """Test all transition probabilities are in [0, 1]."""
        trans = trained_detector.get_transition_matrix()
        
        assert np.all(trans >= 0.0)
        assert np.all(trans <= 1.0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_dataframe(self, detector):
        """Test handling of very small dataframe."""
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })
        
        with pytest.raises(ValueError):
            detector.train(df)
    
    def test_constant_price_data(self, detector):
        """Test handling of constant prices (no volatility)."""
        df = pd.DataFrame({
            'open': [100] * 200,
            'high': [100] * 200,
            'low': [100] * 200,
            'close': [100] * 200,
            'volume': [1000] * 200
        })
        
        # Should handle gracefully (features will be zero)
        detector.train(df)
        assert detector.is_trained
    
    def test_extreme_volatility(self, detector):
        """Test handling of extreme volatility."""
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.5, 200)))
        
        df = pd.DataFrame({
            'open': close * 0.9,
            'high': close * 1.1,
            'low': close * 0.8,
            'close': close,
            'volume': np.random.rand(200) * 10000
        })
        
        detector.train(df)
        regime, conf = detector.predict_regime(df)
        
        assert isinstance(regime, str)
        assert 0.0 <= conf <= 1.0
    
    def test_zero_volume(self, detector):
        """Test handling of zero volume."""
        df = pd.DataFrame({
            'open': np.arange(100, 300),
            'high': np.arange(101, 301),
            'low': np.arange(99, 299),
            'close': np.arange(100, 300),
            'volume': [0] * 200
        })
        
        # Should handle gracefully
        detector.train(df)
        assert detector.is_trained


class TestReproducibility:
    """Test reproducibility with random state."""
    
    def test_same_random_state_same_results(self, sample_dataframe):
        """Test same random state produces identical results."""
        detector1 = RegimeDetector(n_states=3, random_state=42)
        detector2 = RegimeDetector(n_states=3, random_state=42)
        
        detector1.train(sample_dataframe, lookback=500)
        detector2.train(sample_dataframe, lookback=500)
        
        regime1, conf1 = detector1.predict_regime(sample_dataframe)
        regime2, conf2 = detector2.predict_regime(sample_dataframe)
        
        assert regime1 == regime2
        assert abs(conf1 - conf2) < 1e-10
    
    def test_different_random_state_may_differ(self, sample_dataframe):
        """Test different random states may produce different results."""
        detector1 = RegimeDetector(n_states=3, random_state=42)
        detector2 = RegimeDetector(n_states=3, random_state=123)
        
        detector1.train(sample_dataframe, lookback=500)
        detector2.train(sample_dataframe, lookback=500)
        
        # Results may differ, but should still be valid
        regime1, conf1 = detector1.predict_regime(sample_dataframe)
        regime2, conf2 = detector2.predict_regime(sample_dataframe)
        
        assert isinstance(regime1, str)
        assert isinstance(regime2, str)
        assert 0.0 <= conf1 <= 1.0
        assert 0.0 <= conf2 <= 1.0


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self, sample_dataframe):
        """Test complete train-predict workflow."""
        # Initialize
        detector = RegimeDetector(n_states=3, random_state=42)
        assert not detector.is_trained
        
        # Train
        detector.train(sample_dataframe, lookback=500)
        assert detector.is_trained
        
        # Predict
        regime, confidence = detector.predict_regime(sample_dataframe)
        assert regime in detector.regime_names.values()
        assert 0.0 <= confidence <= 1.0
        
        # Get probabilities
        probs = detector.get_regime_probabilities(sample_dataframe)
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        
        # Get transitions
        trans = detector.get_transition_matrix()
        assert trans.shape == (3, 3)
    
    def test_multiple_predictions(self, trained_detector, sample_dataframe):
        """Test multiple predictions maintain consistency."""
        regimes = []
        confidences = []
        
        for _ in range(5):
            regime, conf = trained_detector.predict_regime(sample_dataframe)
            regimes.append(regime)
            confidences.append(conf)
        
        # All should be identical (deterministic)
        assert len(set(regimes)) == 1
        assert len(set(confidences)) == 1
    
    def test_retrain_updates_model(self, detector, sample_dataframe):
        """Test retraining updates the model."""
        # First training
        detector.train(sample_dataframe, lookback=400)
        regime1, _ = detector.predict_regime(sample_dataframe)
        
        # Retrain with different data
        new_data = sample_dataframe.iloc[100:].copy()
        detector.train(new_data, lookback=300)
        regime2, _ = detector.predict_regime(new_data)
        
        # Model should still work (regimes may or may not differ)
        assert isinstance(regime1, str)
        assert isinstance(regime2, str)


# Performance test (optional, can be slow)
@pytest.mark.slow
class TestPerformance:
    """Test performance characteristics."""
    
    def test_training_speed(self, sample_dataframe):
        """Test training completes in reasonable time."""
        import time
        
        detector = RegimeDetector(n_states=3)
        
        start = time.time()
        detector.train(sample_dataframe, lookback=500)
        duration = time.time() - start
        
        # Should complete in less than 5 seconds
        assert duration < 5.0
    
    def test_prediction_speed(self, trained_detector, sample_dataframe):
        """Test prediction is fast."""
        import time
        
        start = time.time()
        for _ in range(100):
            trained_detector.predict_regime(sample_dataframe)
        duration = time.time() - start
        
        # 100 predictions should complete in less than 2 seconds
        assert duration < 2.0
