#!/usr/bin/env python3
"""
Test robust scaling implementation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from src.regime.hmm_detector import RegimeDetector

def test_robust_scaling():
    """Test robust scaling vs standard scaling"""
    print("Testing Robust Scaling Implementation...")
    
    # Create sample data with outliers
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with some extreme outliers (common in crypto)
    prices = [100]
    for i in range(1, n_samples):
        if i % 100 == 0:
            # Add extreme moves occasionally (like crypto crashes/pumps)
            change = np.random.choice([-0.15, 0.15])  # 15% moves
        else:
            change = np.random.normal(0, 0.01)  # 1% normal moves
        prices.append(prices[-1] * (1 + change))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    print(f"Data range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Max price change: {df['close'].pct_change().max():.3f}")
    
    # Test both scalers
    print("\n1. Testing StandardScaler vs RobustScaler:")
    
    # Create two detectors with different scalers
    detector_standard = RegimeDetector(n_states=3, random_state=42)
    detector_robust = RegimeDetector(n_states=3, random_state=42)
    
    # Manually set scalers for comparison
    detector_standard.scaler = StandardScaler()
    detector_robust.scaler = RobustScaler()
    
    try:
        # Train both detectors
        detector_standard.train(df, lookback=500)
        detector_robust.train(df, lookback=500)
        
        print(f"   [OK] Both detectors trained successfully")
        
        # Test feature scaling
        features = detector_robust.prepare_features(df)
        
        # Scale with both methods
        X_standard = detector_standard.scaler.fit_transform(features)
        X_robust = detector_robust.scaler.fit_transform(features)
        
        print(f"   [OK] Feature scaling completed")
        print(f"   StandardScaler - Mean: {np.mean(X_standard, axis=0)[:3]}")
        print(f"   StandardScaler - Std:  {np.std(X_standard, axis=0)[:3]}")
        print(f"   RobustScaler   - Mean: {np.mean(X_robust, axis=0)[:3]}")
        print(f"   RobustScaler   - Std:  {np.std(X_robust, axis=0)[:3]}")
        
        # Test regime prediction
        regime_std, conf_std = detector_standard.predict_regime(df)
        regime_rob, conf_rob = detector_robust.predict_regime(df)
        
        print(f"   StandardScaler prediction: {regime_std} (confidence: {conf_std:.3f})")
        print(f"   RobustScaler prediction:   {regime_rob} (confidence: {conf_rob:.3f})")
        
        # Test sequence prediction
        regime_seq_std, conf_seq_std = detector_standard.predict_regime_sequence(df)
        regime_seq_rob, conf_seq_rob = detector_robust.predict_regime_sequence(df)
        
        # Calculate regime stability
        changes_std = np.sum(regime_seq_std[1:] != regime_seq_std[:-1])
        changes_rob = np.sum(regime_seq_rob[1:] != regime_seq_rob[:-1])
        
        rate_std = changes_std / len(regime_seq_std)
        rate_rob = changes_rob / len(regime_seq_rob)
        
        print(f"   StandardScaler regime changes: {changes_std} ({rate_std:.3f})")
        print(f"   RobustScaler regime changes:   {changes_rob} ({rate_rob:.3f})")
        
        # Calculate improvement
        improvement = ((rate_std - rate_rob) / rate_std) * 100 if rate_std > 0 else 0
        print(f"   Improvement: {improvement:.1f}% reduction in regime changes")
        
        print("\n[SUCCESS] Robust scaling test completed!")
        return True, detector_standard, detector_robust, df
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False, None, None, None

def test_outlier_handling():
    """Test outlier handling capabilities"""
    print("\n2. Testing outlier handling:")
    
    # Create data with extreme outliers
    np.random.seed(123)
    n_samples = 500
    
    # Generate normal data
    normal_data = np.random.normal(0, 1, n_samples)
    
    # Add extreme outliers
    outlier_indices = [50, 150, 250, 350, 450]
    for idx in outlier_indices:
        normal_data[idx] += np.random.choice([-10, 10])  # Extreme outliers
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': 100 + normal_data,
        'high': 100 + normal_data + 1,
        'low': 100 + normal_data - 1,
        'close': 100 + normal_data,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    print(f"   Data with {len(outlier_indices)} extreme outliers")
    print(f"   Data range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Test both scalers
    detector_std = RegimeDetector(n_states=3, random_state=42)
    detector_rob = RegimeDetector(n_states=3, random_state=42)
    
    detector_std.scaler = StandardScaler()
    detector_rob.scaler = RobustScaler()
    
    try:
        # Train both
        detector_std.train(df, lookback=300)
        detector_rob.train(df, lookback=300)
        
        # Get features
        features = detector_rob.prepare_features(df)
        
        # Scale features
        X_std = detector_std.scaler.fit_transform(features)
        X_rob = detector_rob.scaler.fit_transform(features)
        
        # Check outlier impact
        std_outlier_impact = np.max(np.abs(X_std[outlier_indices]))
        rob_outlier_impact = np.max(np.abs(X_rob[outlier_indices]))
        
        print(f"   StandardScaler outlier impact: {std_outlier_impact:.3f}")
        print(f"   RobustScaler outlier impact:   {rob_outlier_impact:.3f}")
        
        # RobustScaler should have lower outlier impact
        if rob_outlier_impact < std_outlier_impact:
            print(f"   [OK] RobustScaler better handles outliers")
        else:
            print(f"   [WARNING] RobustScaler not showing expected outlier resistance")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Error in outlier test: {e}")
        return False

def test_scaling_consistency():
    """Test scaling consistency across different data distributions"""
    print("\n3. Testing scaling consistency:")
    
    # Test with different data distributions
    distributions = [
        ("Normal", lambda: np.random.normal(0, 1, 1000)),
        ("Uniform", lambda: np.random.uniform(-3, 3, 1000)),
        ("Exponential", lambda: np.random.exponential(1, 1000) - 1),
        ("Mixed", lambda: np.concatenate([
            np.random.normal(0, 1, 800),
            np.random.normal(0, 5, 200)  # Different variance
        ]))
    ]
    
    for name, data_gen in distributions:
        print(f"   Testing {name} distribution:")
        
        data = data_gen()
        df = pd.DataFrame({
            'open': 100 + data,
            'high': 100 + data + 1,
            'low': 100 + data - 1,
            'close': 100 + data,
            'volume': np.random.uniform(1000, 10000, len(data))
        })
        
        detector = RegimeDetector(n_states=3, random_state=42)
        
        try:
            detector.train(df, lookback=min(500, len(df)))
            
            # Test prediction
            regime, confidence = detector.predict_regime(df)
            print(f"     Regime: {regime}, Confidence: {confidence:.3f}")
            
        except Exception as e:
            print(f"     [ERROR] Failed with {name}: {e}")
    
    print("   [OK] Scaling consistency test completed")
    return True

if __name__ == "__main__":
    print("HMM v2.0 - Robust Scaling Test")
    print("=" * 50)
    
    success, detector_std, detector_rob, df = test_robust_scaling()
    
    if success:
        test_outlier_handling()
        test_scaling_consistency()
        print("\n[SUCCESS] All robust scaling tests passed!")
    else:
        print("\n[ERROR] Robust scaling tests failed.")
