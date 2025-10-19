#!/usr/bin/env python3
"""
Create GitHub issues for HMM v2.0 improvements
"""
import json
from datetime import datetime

def create_issues():
    """Create GitHub issues for HMM v2 improvements"""
    
    issues = [
        {
            "title": "HMM v2.0: Implement Log Returns for Better Crypto Data Handling",
            "body": """## 🎯 Objective
Replace `pct_change()` with log returns for better handling of crypto market volatility and more stable feature distributions.

## 📋 Current Implementation
```python
df[f'returns_{period}'] = df['close'].pct_change(period)
```

## 🔧 Proposed Solution
```python
df[f'returns_{period}'] = np.log(df['close'] / df['close'].shift(period))
```

## ✅ Benefits
- Better handling of high volatility crypto data
- More stable feature distributions
- Reduced bias in feature engineering
- Improved regime detection accuracy

## 🧪 Testing
- [ ] Unit tests for log returns calculation
- [ ] Integration tests with HMM training
- [ ] Performance comparison with pct_change
- [ ] Backtest validation

## 📊 Success Metrics
- Feature stability score > 95%
- Regime detection accuracy improvement
- Reduced overfitting in high volatility periods

**Priority**: High  
**Estimated Time**: 4-6 hours  
**Labels**: enhancement, hmm-v2, feature-engineering""",
            "labels": ["enhancement", "hmm-v2", "feature-engineering"]
        },
        {
            "title": "HMM v2.0: Dynamic Smoothing Window for Adaptive Performance",
            "body": """## 🎯 Objective
Implement dynamic smoothing window sizing based on dataset length for better adaptation to different timeframes and data sizes.

## 📋 Current Implementation
```python
smooth_window = 5  # Fixed window
```

## 🔧 Proposed Solution
```python
smooth_window = max(3, int(len(dataframe) * 0.002))
```

## ✅ Benefits
- Adaptive to different dataset sizes
- Better performance on various timeframes
- Reduced over-smoothing on small datasets
- Improved smoothing on large datasets

## 🧪 Testing
- [ ] Test with different dataset sizes (100, 1000, 10000 candles)
- [ ] Validate on different timeframes (1m, 5m, 1h, 1d)
- [ ] Performance comparison with fixed window
- [ ] Regime stability analysis

## 📊 Success Metrics
- Regime change rate < 5%
- Consistent performance across timeframes
- Improved stability on large datasets

**Priority**: Medium  
**Estimated Time**: 3-4 hours  
**Labels**: enhancement, hmm-v2, performance""",
            "labels": ["enhancement", "hmm-v2", "performance"]
        },
        {
            "title": "HMM v2.0: Enhanced Regime Labeling with Slope and Skewness",
            "body": """## 🎯 Objective
Improve regime labeling by adding slope analysis and skewness features to better distinguish between weak trends and ranges.

## 📋 Current Implementation
```python
# Only uses volatility and momentum
volatility = (state_mean[3] + state_mean[4]) / 2
momentum = (state_mean[5] + state_mean[6]) / 2
```

## 🔧 Proposed Solution
```python
# Add slope analysis
df['sma_slope'] = df['sma_20'].diff(5)
df['trend_strength'] = df['sma_slope'] * df['momentum_20']

# Add skewness
df['returns_skew'] = df['returns_1'].rolling(20).skew()

# Enhanced labeling logic
slope = state_mean[slope_index]
skewness = state_mean[skew_index]
```

## ✅ Benefits
- Better distinction between ranges and weak trends
- More accurate regime classification
- Reduced false trending signals
- Improved strategy performance

## 🧪 Testing
- [ ] Unit tests for new features
- [ ] Regime classification accuracy tests
- [ ] Backtest performance comparison
- [ ] Visual validation of regime boundaries

## 📊 Success Metrics
- Improved regime classification accuracy
- Reduced false trending signals
- Better strategy performance
- More stable regime transitions

**Priority**: High  
**Estimated Time**: 6-8 hours  
**Labels**: enhancement, hmm-v2, feature-engineering""",
            "labels": ["enhancement", "hmm-v2", "feature-engineering"]
        },
        {
            "title": "HMM v2.0: Implement Robust Scaling for Better Data Adaptation",
            "body": """## 🎯 Objective
Replace StandardScaler with RobustScaler for better handling of outliers and more stable normalization across different market conditions.

## 📋 Current Implementation
```python
self.scaler = StandardScaler()
```

## 🔧 Proposed Solution
```python
self.scaler = RobustScaler()
```

## ✅ Benefits
- Better handling of outliers
- More stable normalization
- Improved adaptation to market regime changes
- Reduced impact of extreme values

## 🧪 Testing
- [ ] Unit tests for RobustScaler integration
- [ ] Performance comparison with StandardScaler
- [ ] Outlier handling validation
- [ ] Cross-market testing

## 📊 Success Metrics
- Improved feature stability
- Better outlier handling
- More consistent normalization
- Enhanced regime detection accuracy

**Priority**: Medium  
**Estimated Time**: 3-4 hours  
**Labels**: enhancement, hmm-v2, data-processing""",
            "labels": ["enhancement", "hmm-v2", "data-processing"]
        },
        {
            "title": "HMM v2.0: Adaptive Retraining for Real-time Market Adaptation",
            "body": """## 🎯 Objective
Implement online retraining mechanism that automatically retrains the HMM model every 4-6 hours using the most recent data for better adaptation to changing market conditions.

## 📋 Current Implementation
```python
# One-time training during strategy initialization
detector.train(dataframe, lookback=1000)
```

## 🔧 Proposed Solution
```python
class AdaptiveHMMDetector(RegimeDetector):
    def __init__(self, retrain_interval_hours=6):
        self.retrain_interval = retrain_interval_hours
        self.last_retrain = None
        self.performance_monitor = PerformanceMonitor()
    
    def should_retrain(self):
        # Check if retrain interval has passed
        # Monitor performance degradation
        # Validate data quality
        pass
    
    def adaptive_retrain(self, new_data):
        # Retrain with recent data
        # Validate performance
        # Update model if improved
        pass
```

## ✅ Benefits
- Real-time adaptation to market changes
- Improved performance over time
- Automatic model maintenance
- Better handling of regime shifts

## 🧪 Testing
- [ ] Unit tests for retraining logic
- [ ] Performance monitoring tests
- [ ] Integration tests with strategy
- [ ] Long-term stability tests

## 📊 Success Metrics
- Automatic retraining every 4-6 hours
- Performance improvement over time
- Stable operation in live trading
- Reduced manual intervention

**Priority**: High  
**Estimated Time**: 8-10 hours  
**Labels**: enhancement, hmm-v2, real-time, performance""",
            "labels": ["enhancement", "hmm-v2", "real-time", "performance"]
        },
        {
            "title": "HMM v2.0: On-chain and Macro Features Integration",
            "body": """## 🎯 Objective
Integrate on-chain data and macro features to enhance regime detection beyond price/volume data for more comprehensive market state analysis.

## 📋 Current Implementation
```python
# Only OHLCV data
features = ['returns', 'volatility', 'momentum', 'volume', 'price_range']
```

## 🔧 Proposed Solution
```python
# Add on-chain features
onchain_features = [
    'funding_rate',
    'open_interest',
    'netflow_large_wallets',
    'long_short_ratio',
    'exchange_inflows',
    'exchange_outflows'
]

# Add macro features
macro_features = [
    'vix_correlation',
    'dxy_correlation',
    'gold_correlation',
    'market_sentiment',
    'fear_greed_index'
]
```

## ✅ Benefits
- More comprehensive market analysis
- Better regime detection accuracy
- Reduced false signals
- Enhanced strategy performance

## 🧪 Testing
- [ ] Data source integration tests
- [ ] Feature engineering validation
- [ ] Performance impact analysis
- [ ] Cross-validation with different markets

## 📊 Success Metrics
- Improved regime detection accuracy
- Better strategy performance
- Reduced false signals
- Enhanced market understanding

**Priority**: Low  
**Estimated Time**: 2-3 weeks  
**Labels**: enhancement, hmm-v2, external-data, advanced""",
            "labels": ["enhancement", "hmm-v2", "external-data", "advanced"]
        }
    ]
    
    # Save issues to file
    with open('hmm_v2_issues.json', 'w', encoding='utf-8') as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    
    print(f"Created {len(issues)} GitHub issues for HMM v2.0 improvements")
    print("Issues saved to: hmm_v2_issues.json")
    
    return issues

if __name__ == "__main__":
    issues = create_issues()
    
    print("\nHMM v2.0 Improvement Issues:")
    print("=" * 50)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['title']}")
        print(f"   Priority: {issue['labels'][0]}")
        print(f"   Labels: {', '.join(issue['labels'])}")
        print()
