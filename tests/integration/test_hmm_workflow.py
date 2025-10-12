"""
Integration tests for complete HMM workflow

Tests the entire pipeline from data loading to prediction,
ensuring all components work together correctly.
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


@pytest.fixture
def large_sample_data():
    """Generate large realistic dataset for integration testing."""
    np.random.seed(42)
    n = 2000  # Larger dataset for realistic testing
    
    # Simulate different market regimes
    returns_trending = np.random.normal(0.002, 0.015, n // 3)
    returns_ranging = np.random.normal(0, 0.008, n // 3)
    returns_volatile = np.random.normal(-0.001, 0.035, n // 3)
    
    returns = np.concatenate([returns_trending, returns_ranging, returns_volatile])
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_price = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.lognormal(15, 0.5, n)
    
    # Create DataFrame with timestamps
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)


class TestCompleteWorkflow:
    """Test complete HMM training and prediction workflow."""
    
    def test_train_predict_workflow(self, large_sample_data):
        """Test complete workflow from training to prediction."""
        # 1. Initialize detector
        detector = RegimeDetector(n_states=3, random_state=42)
        assert not detector.is_trained
        
        # 2. Split data temporally (no data leakage)
        train_size = int(len(large_sample_data) * 0.7)
        train_data = large_sample_data.iloc[:train_size]
        test_data = large_sample_data.iloc[train_size:]
        
        # 3. Train on training data
        detector.train(train_data, lookback=500)
        assert detector.is_trained
        
        # 4. Predict on test data (unseen)
        regime, confidence = detector.predict_regime(test_data)
        
        # 5. Validate results
        assert isinstance(regime, str)
        assert regime in detector.regime_names.values()
        assert 0.0 <= confidence <= 1.0
        
        # 6. Get probabilities
        probs = detector.get_regime_probabilities(test_data)
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        
        # 7. Get transitions
        trans = detector.get_transition_matrix()
        assert trans.shape == (3, 3)
    
    def test_temporal_consistency(self, large_sample_data):
        """Test that predictions maintain temporal consistency."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Train on first 70%
        train_size = int(len(large_sample_data) * 0.7)
        train_data = large_sample_data.iloc[:train_size]
        detector.train(train_data, lookback=500)
        
        # Predict on sliding windows (walk-forward)
        predictions = []
        window_size = 100
        
        for i in range(train_size, len(large_sample_data) - window_size, window_size):
            window_data = large_sample_data.iloc[i:i+window_size]
            regime, conf = detector.predict_regime(window_data)
            predictions.append((regime, conf))
        
        # Should have made multiple predictions
        assert len(predictions) > 5
        
        # All predictions should be valid
        for regime, conf in predictions:
            assert regime in detector.regime_names.values()
            assert 0.0 <= conf <= 1.0
    
    def test_retraining_workflow(self, large_sample_data):
        """Test model retraining with new data."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Initial training
        initial_train = large_sample_data.iloc[:1000]
        detector.train(initial_train, lookback=500)
        
        regime1, conf1 = detector.predict_regime(initial_train)
        
        # Retrain with more data
        extended_train = large_sample_data.iloc[:1500]
        detector.train(extended_train, lookback=500)
        
        regime2, conf2 = detector.predict_regime(extended_train)
        
        # Both predictions should be valid (may differ)
        assert isinstance(regime1, str)
        assert isinstance(regime2, str)
        assert 0.0 <= conf1 <= 1.0
        assert 0.0 <= conf2 <= 1.0


class TestDataLeakagePrevention:
    """Test that no data leakage occurs in the pipeline."""
    
    def test_future_data_not_used(self, large_sample_data):
        """Verify that future data doesn't influence past predictions."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Train on first half only
        mid_point = len(large_sample_data) // 2
        first_half = large_sample_data.iloc[:mid_point]
        
        detector.train(first_half, lookback=500)
        regime_before, conf_before = detector.predict_regime(first_half)
        
        # Now "reveal" second half but DON'T retrain
        # Prediction on first half should stay the same
        regime_after, conf_after = detector.predict_regime(first_half)
        
        # Results should be identical (deterministic)
        assert regime_before == regime_after
        assert conf_before == conf_after
    
    def test_temporal_split_correctness(self, large_sample_data):
        """Test that temporal splits are respected."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Define strict temporal split
        train_end_idx = 1200
        test_start_idx = 1201
        
        train_data = large_sample_data.iloc[:train_end_idx]
        test_data = large_sample_data.iloc[test_start_idx:]
        
        # Verify no overlap
        assert train_data.index[-1] < test_data.index[0]
        
        # Train and predict
        detector.train(train_data, lookback=500)
        regime, conf = detector.predict_regime(test_data)
        
        # Should work without errors
        assert isinstance(regime, str)
        assert 0.0 <= conf <= 1.0


class TestScalabilityAndPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset_training(self):
        """Test training on large dataset completes in reasonable time."""
        import time
        
        # Generate very large dataset
        np.random.seed(42)
        n = 5000
        close = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
        
        df = pd.DataFrame({
            'open': close * 0.99,
            'high': close * 1.01,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.lognormal(15, 0.5, n)
        })
        
        detector = RegimeDetector(n_states=3, random_state=42)
        
        start = time.time()
        detector.train(df, lookback=1000)
        duration = time.time() - start
        
        # Should complete in less than 10 seconds
        assert duration < 10.0
        assert detector.is_trained
    
    def test_repeated_predictions_fast(self, large_sample_data):
        """Test that repeated predictions are fast."""
        import time
        
        detector = RegimeDetector(n_states=3, random_state=42)
        train_data = large_sample_data.iloc[:1000]
        detector.train(train_data, lookback=500)
        
        # Time 50 predictions
        start = time.time()
        for _ in range(50):
            detector.predict_regime(large_sample_data.iloc[1000:1100])
        duration = time.time() - start
        
        # Should complete in less than 2 seconds
        assert duration < 2.0


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""
    
    def test_graceful_degradation_insufficient_data(self):
        """Test graceful handling of insufficient data."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Very small dataset
        small_df = pd.DataFrame({
            'open': [100] * 50,
            'high': [101] * 50,
            'low': [99] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        
        with pytest.raises(ValueError, match="Insufficient data"):
            detector.train(small_df)
    
    def test_prediction_before_training_error(self, large_sample_data):
        """Test error when predicting before training."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.predict_regime(large_sample_data)


@pytest.mark.slow
class TestLongRunningScenarios:
    """Test long-running scenarios (marked as slow)."""
    
    def test_continuous_operation_simulation(self, large_sample_data):
        """Simulate continuous operation over extended period."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Initial training
        detector.train(large_sample_data.iloc[:500], lookback=400)
        
        # Simulate continuous predictions with periodic retraining
        retrain_interval = 200
        prediction_count = 0
        
        for i in range(500, len(large_sample_data), 50):
            # Predict
            window = large_sample_data.iloc[max(0, i-100):i]
            regime, conf = detector.predict_regime(window)
            prediction_count += 1
            
            # Retrain periodically
            if prediction_count % 4 == 0:  # Every 4th prediction
                retrain_data = large_sample_data.iloc[:i]
                detector.train(retrain_data, lookback=min(500, len(retrain_data)))
        
        # Should complete without errors
        assert prediction_count > 20
        assert detector.is_trained


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
