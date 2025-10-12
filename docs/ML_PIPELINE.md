# Machine Learning Pipeline Documentation

**Critical Document: Data Leakage Prevention & ML Best Practices**

---

## Table of Contents

1. [Overview](#overview)
2. [Data Leakage Prevention](#data-leakage-prevention)
3. [HMM Training Pipeline](#hmm-training-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Time-Series Split Strategy](#time-series-split-strategy)
6. [Model Training & Validation](#model-training--validation)
7. [Retraining Strategy](#retraining-strategy)
8. [Model Versioning](#model-versioning)
9. [Production Deployment](#production-deployment)
10. [Monitoring & Validation](#monitoring--validation)

---

## Overview

This document outlines the complete ML pipeline for the Pragma Trading Bot, with **special emphasis on preventing data leakage** — the #1 cause of false backtesting results in algorithmic trading.

### Key Principles

1. **Strict Temporal Ordering**: Never use future data in past predictions
2. **Time-Series Aware Splits**: No random shuffling
3. **Feature Calculation Discipline**: Only use data available at prediction time
4. **Walk-Forward Validation**: Realistic simulation of production behavior
5. **Independent Test Sets**: Never seen during training or hyperparameter tuning

---

## Data Leakage Prevention

### What is Data Leakage?

**Data leakage** occurs when information from outside the training dataset is used to create the model. In time-series trading, this typically happens when:

- Future data influences past predictions
- Training/validation sets overlap temporally
- Features use future information (lookahead bias)
- Scaling is done on entire dataset before splitting

### Common Leakage Sources in Trading Bots

#### ❌ **WRONG: Scaling Before Splitting**

```python
# DANGEROUS - Information from test set leaks into training
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
all_data_scaled = scaler.fit_transform(all_data)  # ❌ Fit on ALL data
train, test = split(all_data_scaled)  # Test statistics leaked!
```

#### ✅ **CORRECT: Fit Scaler on Training Only**

```python
# SAFE - Scaler only learns from training data
train, test = split(all_data)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)  # ✅ Fit on train only
test_scaled = scaler.transform(test)  # ✅ Apply learned parameters
```

#### ❌ **WRONG: Using Future Data in Features**

```python
# DANGEROUS - Uses future returns
df['future_return'] = df['close'].pct_change().shift(-5)  # ❌ Lookahead!
df['signal'] = df['future_return'] > 0  # Using future to predict past
```

#### ✅ **CORRECT: Only Historical Features**

```python
# SAFE - Only past data
df['past_return'] = df['close'].pct_change(5)  # ✅ Historical only
df['signal'] = df['past_return'] > threshold  # Using past to predict future
```

### Our Prevention Strategy

1. **Temporal Data Splits**: Use `TimeSeriesSplit` or manual date-based splits
2. **Fit-Transform Pattern**: Always fit on train, transform on train/test separately
3. **Feature Timestamp Verification**: Automated checks for lookahead bias
4. **Walk-Forward Testing**: Continuous retraining simulation
5. **Code Reviews**: All feature engineering reviewed for temporal correctness

---

## HMM Training Pipeline

### Current Implementation: RegimeDetector

The `RegimeDetector` class implements a 3-state Gaussian HMM for market regime detection.

#### Safe Training Process

```python
from regime.hmm_detector import RegimeDetector

# 1. Initialize detector
detector = RegimeDetector(n_states=3, random_state=42)

# 2. Define training period (PAST data only)
train_end_date = '2024-09-01'
train_data = dataframe[dataframe.index <= train_end_date].copy()

# 3. Train on historical data only
detector.train(train_data, lookback=500)

# 4. Predict on NEW data (never seen during training)
test_data = dataframe[dataframe.index > train_end_date].copy()
regime, confidence = detector.predict_regime(test_data)
```

#### Feature Calculation (No Leakage)

The `prepare_features()` method ensures temporal correctness:

```python
def prepare_features(self, dataframe: pd.DataFrame) -> np.ndarray:
    """
    ALL features use ONLY historical data:
    - returns_1: log(close_t / close_t-1) ✅
    - returns_5: log(close_t / close_t-5) ✅
    - volatility_10: rolling_std(returns, 10) ✅ (window=10 past candles)
    - volume_ratio: volume_t / sma(volume, 20) ✅
    - ADX: calculated from past 14 periods ✅
    
    NO future data is used!
    """
    # Implementation uses .shift() and .rolling() correctly
    # All indicators look backward, never forward
```

---

## Feature Engineering

### Safe Feature Design

#### ✅ **Allowed Operations**

- **Lag features**: `df['price_lag1'] = df['close'].shift(1)`
- **Rolling statistics**: `df['ma20'] = df['close'].rolling(20).mean()`
- **Cumulative operations**: `df['cum_return'] = df['returns'].cumsum()`
- **Historical ratios**: `df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()`

#### ❌ **Forbidden Operations**

- **Negative shifts**: `df['future'] = df['close'].shift(-5)`  # ❌
- **Entire-series operations without time awareness**: `df['zscore'] = (df['close'] - df['close'].mean()) / df['close'].std()`  # ❌
- **Forward-filling from future**: Be careful with `.fillna(method='ffill')` across splits

### Feature Template

```python
def calculate_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Template for leak-free feature engineering.
    """
    df = df.copy()
    
    # ✅ Safe: Historical returns
    df['returns_1d'] = df['close'].pct_change(1)
    df['returns_5d'] = df['close'].pct_change(5)
    
    # ✅ Safe: Rolling statistics
    df['volatility'] = df['returns_1d'].rolling(window=20).std()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    
    # ✅ Safe: Historical comparisons
    df['above_ma'] = (df['close'] > df['ma_50']).astype(int)
    
    # ⚠️ Careful: NaN handling
    # Use forward fill ONLY within a single row's history
    df = df.ffill().bfill()  # ✅ OK if done after temporal split
    
    return df
```

---

## Time-Series Split Strategy

### Never Use `train_test_split` for Time Series!

```python
# ❌ WRONG for time series
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, shuffle=True)  # ❌ Shuffle!
```

### ✅ Use Time-Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

# Option 1: Scikit-learn TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    # Train and evaluate

# Option 2: Manual date-based split (BEST for trading)
train_end = '2024-09-01'
val_start = '2024-09-02'
val_end = '2024-10-01'
test_start = '2024-10-02'

train_data = df[df.index <= train_end]
val_data = df[(df.index >= val_start) & (df.index <= val_end)]
test_data = df[df.index >= test_start]

# Option 3: Walk-Forward (BEST for realism)
# Train on months 1-6, test on month 7
# Train on months 2-7, test on month 8
# Train on months 3-8, test on month 9
# ... (continuously rolling window)
```

### Our Recommended Split

```
|<------- Training ------>|<-- Validation -->|<----- Test ----->|
  Jan      ...      Aug      Sep      Oct       Nov      Dec
                     ↑                 ↑                   ↑
                  Train End         Val End            Test End
                  
- Training: Jan 1 - Aug 31 (8 months)
- Validation: Sep 1 - Oct 31 (2 months) - for hyperparameter tuning
- Test: Nov 1 - Dec 31 (2 months) - NEVER touched during development

CRITICAL: Test set is used ONLY ONCE at the very end!
```

---

## Model Training & Validation

### Complete Safe Training Workflow

```python
def train_hmm_safe(dataframe: pd.DataFrame, 
                   train_end: str,
                   val_end: str) -> Tuple[RegimeDetector, dict]:
    """
    Complete leak-free HMM training workflow.
    
    Args:
        dataframe: Full OHLCV dataframe
        train_end: Last date for training (e.g., '2024-08-31')
        val_end: Last date for validation (e.g., '2024-10-31')
    
    Returns:
        Trained detector and validation metrics
    """
    # 1. Temporal split
    train_df = dataframe[dataframe.index <= train_end].copy()
    val_df = dataframe[(dataframe.index > train_end) & 
                       (dataframe.index <= val_end)].copy()
    
    # 2. Initialize detector
    detector = RegimeDetector(n_states=3, random_state=42)
    
    # 3. Train on training set ONLY
    detector.train(train_df, lookback=500)
    
    # 4. Validate on UNSEEN validation data
    val_regime, val_conf = detector.predict_regime(val_df)
    
    # 5. Calculate validation metrics
    metrics = {
        'regime': val_regime,
        'confidence': val_conf,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'train_end': train_end,
        'val_end': val_end
    }
    
    return detector, metrics
```

### Validation Metrics (No Leakage)

```python
def validate_regime_detector(detector: RegimeDetector,
                             val_dataframe: pd.DataFrame) -> dict:
    """
    Validate detector on unseen data.
    """
    # Predict regimes
    regime, confidence = detector.predict_regime(val_dataframe)
    probs = detector.get_regime_probabilities(val_dataframe)
    trans_matrix = detector.get_transition_matrix()
    
    # Validation metrics
    metrics = {
        'predicted_regime': regime,
        'confidence': confidence,
        'regime_probabilities': probs,
        'transition_matrix': trans_matrix,
        'validation_samples': len(val_dataframe),
        
        # Sanity checks
        'confidence_valid': 0.0 <= confidence <= 1.0,
        'probs_sum_to_one': abs(sum(probs.values()) - 1.0) < 1e-6,
        'trans_matrix_valid': all((row.sum() - 1.0) < 1e-6 
                                  for row in trans_matrix)
    }
    
    return metrics
```

---

## Retraining Strategy

### Automatic Retraining (Production)

The strategy implements automatic retraining every N candles:

```python
class RegimeAdaptiveStrategy(IStrategy):
    def populate_indicators(self, dataframe, metadata):
        # Check if retraining is needed
        if len(dataframe) >= 100 and not self.regime_trained:
            try:
                # Use recent data for training (lookback window)
                lookback = min(len(dataframe), self.regime_training_lookback.value)
                train_data = dataframe.iloc[-lookback:].copy()  # Recent history only
                
                # Train HMM
                self.regime_detector.train(train_data, lookback=lookback)
                self.regime_trained = True
                
                logger.info(f"HMM trained with {lookback} candles")
            except Exception as e:
                logger.warning(f"HMM training failed: {e}")
        
        # Predict on current data
        if self.regime_trained:
            regime, confidence = self.regime_detector.predict_regime(dataframe)
            dataframe['regime'] = regime
            dataframe['regime_confidence'] = confidence
        
        return dataframe
```

### Retrain Frequency

**Recommendation based on market dynamics:**

- **High volatility markets**: Retrain every 3-7 days
- **Stable markets**: Retrain every 14-30 days
- **Current setting**: Every 500 candles (~2 days on 5m timeframe)

### Retrain Validation

```python
def retrain_with_validation(detector: RegimeDetector,
                            new_data: pd.DataFrame,
                            validation_window: int = 100) -> bool:
    """
    Retrain and validate before deployment.
    
    Returns:
        True if retrained model passes validation, False otherwise
    """
    # Split new data
    train_size = len(new_data) - validation_window
    train_df = new_data.iloc[:train_size]
    val_df = new_data.iloc[train_size:]
    
    # Create new detector instance
    new_detector = RegimeDetector(n_states=3, random_state=42)
    
    # Train
    new_detector.train(train_df, lookback=min(500, len(train_df)))
    
    # Validate
    regime, conf = new_detector.predict_regime(val_df)
    
    # Validation checks
    if conf < 0.5:
        logger.warning(f"Low confidence after retrain: {conf}")
        return False
    
    # If passes, deploy
    detector.model = new_detector.model
    detector.scaler = new_detector.scaler
    detector.regime_names = new_detector.regime_names
    detector.is_trained = True
    
    return True
```

---

## Model Versioning

### Save Model with Metadata

```python
import json
import joblib
from datetime import datetime

def save_model_with_metadata(detector: RegimeDetector,
                             train_data: pd.DataFrame,
                             metrics: dict,
                             model_dir: str = 'models/'):
    """
    Save model with complete metadata for reproducibility.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"hmm_regime_{timestamp}"
    
    # Save model
    model_path = f"{model_dir}/{model_id}.pkl"
    joblib.dump(detector, model_path)
    
    # Save metadata
    metadata = {
        'model_id': model_id,
        'timestamp': timestamp,
        'model_type': 'GaussianHMM',
        'n_states': detector.n_states,
        'random_state': detector.random_state,
        
        # Training data info
        'train_start': str(train_data.index[0]),
        'train_end': str(train_data.index[-1]),
        'train_samples': len(train_data),
        'train_data_hash': hash(str(train_data.values.tobytes())),
        
        # Feature info
        'features': [
            'returns_1', 'returns_5', 'returns_20',
            'volatility_10', 'volatility_30',
            'volume_ratio', 'adx'
        ],
        
        # Model info
        'regime_names': detector.regime_names,
        'scaler_mean': detector.scaler.mean_.tolist(),
        'scaler_scale': detector.scaler.scale_.tolist(),
        
        # Performance metrics
        'validation_metrics': metrics,
        
        # Reproducibility
        'requirements_hash': get_requirements_hash(),
        'python_version': sys.version
    }
    
    metadata_path = f"{model_dir}/{model_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved: {model_id}")
    return model_id
```

### Load Model

```python
def load_model_with_validation(model_id: str,
                               model_dir: str = 'models/') -> RegimeDetector:
    """
    Load model and validate metadata.
    """
    # Load metadata
    metadata_path = f"{model_dir}/{model_id}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Validate compatibility
    if metadata['python_version'] != sys.version:
        logger.warning("Python version mismatch")
    
    # Load model
    model_path = f"{model_dir}/{model_id}.pkl"
    detector = joblib.load(model_path)
    
    logger.info(f"Model loaded: {model_id} (trained on {metadata['train_end']})")
    return detector, metadata
```

---

## Production Deployment

### Deployment Checklist

```
□ Model trained on correct temporal split
□ Validation metrics acceptable (confidence > 0.7)
□ No data leakage confirmed (code review)
□ Model and metadata saved
□ Scaling parameters saved
□ Feature definitions documented
□ Retraining schedule configured
□ Monitoring alerts configured
□ Rollback plan prepared
□ Test in dry-run mode first
```

### Deployment Script

```python
def deploy_model_to_production(model_id: str,
                               config_path: str = 'configs/config-private.json'):
    """
    Deploy trained model to live trading.
    """
    # 1. Load and validate model
    detector, metadata = load_model_with_validation(model_id)
    
    # 2. Run final validation
    # ... validation checks ...
    
    # 3. Update production config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['model_id'] = model_id
    config['model_deployment_time'] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 4. Log deployment
    logger.info(f"Model {model_id} deployed to production")
    
    # 5. Send alert
    send_deployment_notification(model_id, metadata)
```

---

## Monitoring & Validation

### Production Monitoring

```python
class ModelMonitor:
    """Monitor model performance in production."""
    
    def __init__(self):
        self.regime_history = []
        self.confidence_history = []
        self.performance_by_regime = {}
    
    def log_prediction(self, regime: str, confidence: float, timestamp):
        """Log each prediction."""
        self.regime_history.append((timestamp, regime))
        self.confidence_history.append((timestamp, confidence))
    
    def check_model_drift(self) -> dict:
        """Detect if model behavior has changed."""
        recent_conf = [c for _, c in self.confidence_history[-100:]]
        avg_conf = np.mean(recent_conf)
        
        alerts = {}
        
        # Alert if confidence drops
        if avg_conf < 0.6:
            alerts['low_confidence'] = f"Avg confidence: {avg_conf:.2f}"
        
        # Alert if stuck in one regime
        recent_regimes = [r for _, r in self.regime_history[-100:]]
        regime_diversity = len(set(recent_regimes)) / len(recent_regimes)
        
        if regime_diversity < 0.1:
            alerts['regime_stuck'] = f"Diversity: {regime_diversity:.2f}"
        
        return alerts
```

### Automated Alerts

```python
def setup_monitoring_alerts():
    """Configure automated monitoring."""
    monitor = ModelMonitor()
    
    # Alert conditions
    ALERT_CONDITIONS = {
        'low_confidence_threshold': 0.6,
        'max_drawdown_pct': 5.0,
        'min_trades_per_day': 1,
        'max_trades_per_day': 20,
        'regime_change_alert': True
    }
    
    return monitor, ALERT_CONDITIONS
```

---

## Summary: Data Leakage Prevention Checklist

### ✅ Before Every Training Session

- [ ] Temporal split verified (train < validation < test)
- [ ] No shuffle in split
- [ ] Scaler fit on training data only
- [ ] Features use only historical data
- [ ] No lookahead bias in any indicator
- [ ] Rolling windows look backward only
- [ ] Test set never touched until final evaluation

### ✅ Code Review Checklist

- [ ] No `.shift(-N)` with negative N
- [ ] No global statistics on entire dataset
- [ ] All `.rolling()` use past windows
- [ ] Feature timestamps verified
- [ ] Backtesting uses `--cache none` for freshness

### ✅ Production Deployment

- [ ] Model versioned and metadata saved
- [ ] Retraining schedule configured
- [ ] Monitoring alerts active
- [ ] Dry-run tested successfully
- [ ] Rollback plan ready

---

## References

- **Time Series Cross-Validation**: [Scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- **Freqtrade Strategy Development**: [Official Docs](https://www.freqtrade.io/en/stable/strategy-customization/)
- **HMM Implementation**: [hmmlearn Documentation](https://hmmlearn.readthedocs.io/)
- **Avoiding Data Leakage**: ["Common Pitfalls in ML"](https://machinelearningmastery.com/data-leakage-machine-learning/)

---

**Last Updated**: 2025-10-12  
**Version**: 1.0  
**Maintainer**: Pragma Trading Team
