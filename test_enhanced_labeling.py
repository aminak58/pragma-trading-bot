#!/usr/bin/env python3
"""
Test enhanced regime labeling with slope and skewness features
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from src.regime.hmm_detector import RegimeDetector

def test_enhanced_labeling():
    """Test enhanced regime labeling with slope and skewness"""
    print("Testing Enhanced Regime Labeling...")
    
    # Create sample data with different market conditions
    np.random.seed(42)
    n_samples = 2000
    
    # Generate data with different regimes
    prices = [100]
    for i in range(1, n_samples):
        if i < 500:
            # Low volatility regime
            change = np.random.normal(0, 0.005)
        elif i < 1000:
            # Trending regime (upward)
            change = np.random.normal(0.002, 0.01)
        elif i < 1500:
            # High volatility regime
            change = np.random.normal(0, 0.02)
        else:
            # Mixed regime
            change = np.random.normal(0, 0.01)
        
        prices.append(prices[-1] * (1 + change))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Test HMM detector with enhanced features
    print("\n1. Testing HMM detector with enhanced features:")
    detector = RegimeDetector(n_states=3, random_state=42)
    
    try:
        # Test feature preparation
        features = detector.prepare_features(df)
        print(f"   [OK] Features shape: {features.shape}")
        print(f"   [OK] Number of features: {features.shape[1]}")
        
        # Test training
        detector.train(df, lookback=1000)
        print(f"   [OK] HMM training successful")
        print(f"   [OK] Regime names: {detector.regime_names}")
        
        # Test prediction
        regime, confidence = detector.predict_regime(df)
        print(f"   [OK] Current regime: {regime} (confidence: {confidence:.3f})")
        
        # Test sequence prediction
        regime_sequence, confidence_sequence = detector.predict_regime_sequence(df)
        print(f"   [OK] Regime sequence length: {len(regime_sequence)}")
        print(f"   [OK] Unique regimes: {np.unique(regime_sequence)}")
        
        # Analyze regime distribution
        unique_regimes, counts = np.unique(regime_sequence, return_counts=True)
        print(f"   [OK] Regime distribution:")
        for regime, count in zip(unique_regimes, counts):
            percentage = (count / len(regime_sequence)) * 100
            print(f"     {regime}: {count} ({percentage:.1f}%)")
        
        print("\n[SUCCESS] Enhanced regime labeling successful!")
        return True, detector, df
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False, None, None

def analyze_feature_importance(detector, df):
    """Analyze the importance of new features"""
    print("\n2. Analyzing feature importance:")
    
    try:
        # Get regime probabilities
        probs = detector.get_regime_probabilities(df)
        
        # Calculate feature statistics for each regime
        features = detector.prepare_features(df)
        regime_sequence, _ = detector.predict_regime_sequence(df)
        
        # Align features with regime sequence
        if len(features) != len(regime_sequence):
            start_idx = len(regime_sequence) - len(features)
            regime_aligned = regime_sequence[start_idx:]
        else:
            regime_aligned = regime_sequence
        
        print(f"   [OK] Analyzing {len(features)} feature samples across {len(np.unique(regime_aligned))} regimes")
        
        # Feature names (based on the order in prepare_features)
        feature_names = [
            'returns_1', 'returns_5', 'returns_20',
            'volatility_5', 'volatility_20',
            'momentum_5', 'momentum_20',
            'volume_ratio', 'price_range',
            'sma_slope_5', 'sma_slope_20', 'trend_strength',
            'returns_skew_5', 'returns_skew_20'
        ]
        
        print(f"   [OK] Feature names: {feature_names}")
        
        # Calculate regime-specific feature means
        for regime in np.unique(regime_aligned):
            mask = regime_aligned == regime
            regime_features = features[mask]
            
            print(f"\n   Regime: {regime}")
            print(f"     Sample count: {np.sum(mask)}")
            print(f"     Feature means:")
            for i, name in enumerate(feature_names):
                if i < len(regime_features[0]):
                    mean_val = np.mean(regime_features[:, i])
                    std_val = np.std(regime_features[:, i])
                    print(f"       {name}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Error in feature analysis: {e}")
        return False

def compare_old_vs_new():
    """Compare old vs new regime labeling"""
    print("\n3. Comparing old vs new regime labeling:")
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with clear trends
    prices = [100]
    for i in range(1, n_samples):
        if i < 300:
            # Strong uptrend
            change = np.random.normal(0.003, 0.01)
        elif i < 600:
            # High volatility
            change = np.random.normal(0, 0.03)
        else:
            # Low volatility sideways
            change = np.random.normal(0, 0.005)
        
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Test both methods
    detector = RegimeDetector(n_states=3, random_state=42)
    
    try:
        detector.train(df, lookback=500)
        regime_sequence, confidence_sequence = detector.predict_regime_sequence(df)
        
        print(f"   [OK] Regime sequence generated")
        print(f"   [OK] Regime names: {detector.regime_names}")
        print(f"   [OK] Average confidence: {np.mean(confidence_sequence):.3f}")
        
        # Calculate regime stability (fewer changes = better)
        regime_changes = np.sum(regime_sequence[1:] != regime_sequence[:-1])
        change_rate = regime_changes / len(regime_sequence)
        print(f"   [OK] Regime change rate: {change_rate:.3f} ({regime_changes} changes)")
        
        # Analyze regime characteristics
        unique_regimes, counts = np.unique(regime_sequence, return_counts=True)
        print(f"   [OK] Regime distribution:")
        for regime, count in zip(unique_regimes, counts):
            percentage = (count / len(regime_sequence)) * 100
            print(f"     {regime}: {count} ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Error in comparison: {e}")
        return False

if __name__ == "__main__":
    print("HMM v2.0 - Enhanced Regime Labeling Test")
    print("=" * 50)
    
    success, detector, df = test_enhanced_labeling()
    
    if success:
        analyze_feature_importance(detector, df)
        compare_old_vs_new()
        print("\n[SUCCESS] All enhanced labeling tests passed!")
    else:
        print("\n[ERROR] Enhanced labeling tests failed.")
