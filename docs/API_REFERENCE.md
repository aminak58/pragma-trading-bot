# API Reference: Pragma Trading Bot

**نسخه:** 1.0  
**تاریخ:** 2025-10-12  
**وضعیت:** 📋 Complete

---

## 📚 Overview

This document provides comprehensive API reference for all Pragma Trading Bot modules.

## 🧠 HMM Regime Detection

### RegimeDetector Class

```python
from src.regime.hmm_detector import RegimeDetector

# Initialize detector
detector = RegimeDetector(n_states=3, random_state=42)

# Train on historical data
detector.train(dataframe, lookback=500)

# Predict current regime
regime, confidence = detector.predict_regime(dataframe)

# Get regime probabilities
probabilities = detector.get_regime_probabilities(dataframe)
```

**Methods:**
- `train(dataframe, lookback=500)` - Train HMM model
- `predict_regime(dataframe)` - Predict current regime
- `get_regime_probabilities(dataframe)` - Get probability distribution
- `get_transition_matrix()` - Get state transition matrix

## 🛡️ Risk Management

### KellyCriterion Class

```python
from src.risk.kelly_criterion import KellyCriterion

# Initialize Kelly Criterion
kelly = KellyCriterion(max_kelly_fraction=0.25)

# Add trade results
kelly.add_trade_result(profit_loss=0.05, confidence=0.8)

# Calculate position size
position_data = kelly.calculate_position_size(
    account_balance=10000,
    current_price=50000,
    stop_loss_price=48000,
    confidence=0.8,
    regime='trending'
)
```

### DynamicStopLoss Class

```python
from src.risk.dynamic_stops import DynamicStopLoss

# Initialize dynamic stops
stops = DynamicStopLoss(base_stop_percent=0.02)

# Calculate ATR-based stop
stop_price = stops.calculate_atr_stop(dataframe, current_price, 'long')

# Calculate adaptive stop
stops_dict = stops.calculate_adaptive_stop(
    dataframe=dataframe,
    current_price=current_price,
    side='long',
    regime='trending',
    confidence=0.8
)
```

### PositionManager Class

```python
from src.risk.position_manager import PositionManager

# Initialize position manager
pm = PositionManager(max_portfolio_risk=0.02)

# Update portfolio value
pm.update_portfolio_value(10000)

# Calculate position size
position_data = pm.calculate_position_size(
    pair='BTC/USDT',
    current_price=50000,
    stop_loss_price=48000,
    confidence=0.8,
    regime='trending'
)

# Open position
position_id = pm.open_position(
    pair='BTC/USDT',
    side='long',
    entry_price=50000,
    stop_loss_price=48000,
    confidence=0.8,
    regime='trending',
    position_data=position_data
)
```

### CircuitBreaker Class

```python
from src.risk.circuit_breakers import CircuitBreaker

# Initialize circuit breaker
breaker = CircuitBreaker(max_drawdown=0.05)

# Update balance
breaker.update_balance(10000)

# Add trade result
breaker.add_trade_result(pnl=0.01, pair='BTC/USDT', confidence=0.8)

# Check if trading allowed
allowed, reason = breaker.check_trading_allowed()
```

## 🤖 Machine Learning

### FreqAIHelper Class

```python
from src.ml.freqai_helper import FreqAIHelper

# Initialize FreqAI helper
freqai = FreqAIHelper(model_path="user_data/models")

# Prepare features
features_df = freqai.prepare_features(dataframe, regime='trending')

# Create training data
X, y = freqai.create_training_data(features_df, 'trending')

# Train model
result = freqai.train_model(X, y, 'trending')

# Make predictions
predictions = freqai.predict(X, 'trending')
```

### ModelManager Class

```python
from src.ml.model_manager import ModelManager

# Initialize model manager
mm = ModelManager()

# Train regime model
result = mm.train_regime_model(dataframe, 'trending')

# Predict regime
prediction = mm.predict_regime(dataframe, 'trending')

# Ensemble prediction
ensemble_pred = mm.predict_ensemble(dataframe, 'trending')
```

### FeatureEngineer Class

```python
from src.ml.feature_engineering import FeatureEngineer

# Initialize feature engineer
fe = FeatureEngineer()

# Create all features
features_df = fe.create_all_features(dataframe, regime='trending')

# Get feature importance
importance_df = fe.get_feature_importance(X, y)
```

## 📊 Strategy Integration

### RegimeAdaptiveStrategy Class

```python
# Freqtrade strategy integration
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy

# Strategy automatically integrates:
# - HMM regime detection
# - Risk management
# - ML predictions
# - Dynamic position sizing
```

## 🔧 Configuration

### Example Configuration

```json
{
  "max_open_trades": 5,
  "stake_currency": "USDT",
  "stake_amount": "unlimited",
  "timeframe": "5m",
  "dry_run": true,
  "exchange": {
    "name": "binance",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_SECRET",
    "pair_whitelist": ["BTC/USDT", "ETH/USDT"]
  }
}
```

## 📈 Performance Monitoring

### Key Metrics

- **Sharpe Ratio:** Target > 1.5
- **Max Drawdown:** Target < 3%
- **Win Rate:** Target > 70%
- **Daily Trades:** 10-20
- **Daily Return:** 1-2% target

### Monitoring Methods

```python
# Get performance stats
stats = position_manager.get_performance_stats()

# Get model performance
model_perf = model_manager.get_model_performance('trending')

# Get circuit breaker status
breaker_status = circuit_breaker.get_status()
```

## 🚨 Error Handling

### Common Exceptions

- `ValueError`: Invalid input parameters
- `RuntimeError`: Model not trained
- `KeyError`: Missing required data
- `AttributeError`: Invalid method calls

### Best Practices

1. Always check if models are trained before prediction
2. Validate input data before processing
3. Handle missing features gracefully
4. Use try-catch blocks for external API calls
5. Log errors for debugging

## 📝 Examples

### Complete Workflow Example

```python
import pandas as pd
from src.regime.hmm_detector import RegimeDetector
from src.risk.position_manager import PositionManager
from src.ml.model_manager import ModelManager

# 1. Initialize components
detector = RegimeDetector()
pm = PositionManager()
mm = ModelManager()

# 2. Load data
data = pd.read_csv('market_data.csv')

# 3. Detect regime
regime, confidence = detector.predict_regime(data)

# 4. Train ML model
mm.train_regime_model(data, regime)

# 5. Make predictions
ml_prediction = mm.predict_regime(data, regime)

# 6. Calculate position size
position_data = pm.calculate_position_size(
    pair='BTC/USDT',
    current_price=data['close'].iloc[-1],
    stop_loss_price=data['close'].iloc[-1] * 0.95,
    confidence=confidence,
    regime=regime
)

# 7. Open position
position_id = pm.open_position(
    pair='BTC/USDT',
    side='long',
    entry_price=data['close'].iloc[-1],
    stop_loss_price=data['close'].iloc[-1] * 0.95,
    confidence=confidence,
    regime=regime,
    position_data=position_data
)
```

---

**Last Updated:** 2025-10-12  
**Next Review:** After major feature additions
