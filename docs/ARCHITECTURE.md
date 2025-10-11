# Architecture Document: Pragma Trading Bot

**نسخه:** 1.0  
**تاریخ:** 2025-10-11  
**وضعیت:** 📋 Draft

---

## 🎯 Design Principles

### 1. Community-Tested > Custom
استفاده از patterns و libraries که توسط community تست شده‌اند.

### 2. Incremental > Big Bang
هر feature به صورت جداگانه develop و test می‌شود.

### 3. Data-Driven > Assumption
هر تصمیم معماری با backtest validate می‌شود.

### 4. Simple > Complex
وقتی هر دو کار می‌کنند، ساده‌تر را انتخاب کن.

### 5. Testable > Elegant
کدی که قابل test است بهتر از کد elegant ولی غیرقابل test است.

---

## 🏗️ System Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────┐
│                   User Interface                      │
│          (Telegram Bot, WebUI - Future)               │
└──────────────────────────────────────────────────────┘
                         ↓↑
┌──────────────────────────────────────────────────────┐
│              Freqtrade Core Engine                    │
│  (Strategy Execution, Order Management, Persistence)  │
└──────────────────────────────────────────────────────┘
                         ↓↑
┌──────────────────────────────────────────────────────┐
│             Pragma Trading Strategy                   │
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │    1. Market Data Pipeline                  │    │
│  │       - Data fetching                       │    │
│  │       - Feature engineering                 │    │
│  │       - Validation & cleaning               │    │
│  └─────────────────────────────────────────────┘    │
│                      ↓                               │
│  ┌─────────────────────────────────────────────┐    │
│  │    2. HMM Regime Detection                  │    │
│  │       - 3-state model (Bull/Bear/Sideways)  │    │
│  │       - Confidence scoring                  │    │
│  │       - Transition tracking                 │    │
│  └─────────────────────────────────────────────┘    │
│                      ↓                               │
│  ┌─────────────────────────────────────────────┐    │
│  │    3. FreqAI ML Layer                       │    │
│  │       - XGBoost/CatBoost models             │    │
│  │       - Auto-retraining (15 days)           │    │
│  │       - Regime-aware features               │    │
│  └─────────────────────────────────────────────┘    │
│                      ↓                               │
│  ┌─────────────────────────────────────────────┐    │
│  │    4. Strategy Logic                        │    │
│  │       - Regime-specific entry/exit          │    │
│  │       - ML confidence filtering             │    │
│  │       - Multi-strategy ensemble             │    │
│  └─────────────────────────────────────────────┘    │
│                      ↓                               │
│  ┌─────────────────────────────────────────────┐    │
│  │    5. Risk Management                       │    │
│  │       - Kelly Criterion sizing              │    │
│  │       - Dynamic ATR stops                   │    │
│  │       - Confidence-based DCA                │    │
│  │       - Circuit breakers                    │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
                         ↓↑
┌──────────────────────────────────────────────────────┐
│              Exchange API (Binance)                   │
└──────────────────────────────────────────────────────┘
```

---

## 📦 Component Details

### 1. Market Data Pipeline

**Responsibility:** داده‌های خام بازار را دریافت و آماده‌سازی می‌کند.

**Interface:**
```python
class DataPipeline:
    def fetch_data(pair: str, timeframe: str) -> DataFrame
    def engineer_features(df: DataFrame) -> DataFrame
    def validate_data(df: DataFrame) -> bool
    def remove_outliers(df: DataFrame) -> DataFrame
```

**Dependencies:**
- Freqtrade data downloaders
- TA-Lib for indicators
- Pandas for data manipulation

**Configuration:**
```json
{
  "data_pipeline": {
    "outlier_removal": true,
    "outlier_method": "IQR",
    "feature_count": 85,
    "validation_strict": true
  }
}
```

---

### 2. HMM Regime Detection

**Responsibility:** شناسایی وضعیت فعلی بازار (Bull/Bear/Sideways).

**Tech Stack:**
- **Library:** hmmlearn
- **Model:** Gaussian HMM
- **States:** 3 (coded as 0, 1, 2)
- **Features:** Returns, Volatility, Volume Ratio, ADX

**Interface:**
```python
class RegimeDetector:
    def __init__(n_states: int = 3)
    def train(dataframe: DataFrame, lookback: int = 500) -> Self
    def predict_regime(dataframe: DataFrame) -> Tuple[str, float]
    def get_transition_matrix() -> np.ndarray
```

**State Mapping:**
```python
{
    0: 'high_volatility',  # Bear or crash
    1: 'low_volatility',   # Sideways
    2: 'trending'          # Bull
}
```

**Training:**
- Initial training: 500 candles minimum
- Retraining: Every 100 candles (optional)
- Features: Standardized (zero mean, unit variance)

**Decision Record:** [ADR-001](./decisions/ADR-001-hmm-implementation.md)

---

### 3. FreqAI ML Layer

**Responsibility:** پیش‌بینی حرکات قیمت با ML.

**Tech Stack:**
- **Framework:** FreqAI (built into Freqtrade)
- **Models:** XGBoost, CatBoost
- **Training Window:** 30 days
- **Test Window:** 7 days
- **Retraining:** Every 15 days

**Configuration:**
```json
{
  "freqai": {
    "enabled": true,
    "purge_old_models": 2,
    "train_period_days": 30,
    "backtest_period_days": 7,
    "live_retrain_hours": 360,
    
    "feature_parameters": {
      "include_timeframes": ["5m", "15m", "1h"],
      "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
      "indicator_periods_candles": [10, 20, 50],
      "DI_threshold": 0.5,
      "use_SVM_to_remove_outliers": true
    },
    
    "model_training_parameters": {
      "n_estimators": 1000,
      "learning_rate": 0.01,
      "max_depth": 8,
      "min_child_weight": 1
    }
  }
}
```

**Features Engineering:**
```python
def feature_engineering_expand_all(dataframe, period, **kwargs):
    # Technical indicators
    dataframe['rsi'] = ta.RSI(dataframe, period=14)
    dataframe['atr'] = ta.ATR(dataframe, period=14)
    dataframe['adx'] = ta.ADX(dataframe, period=14)
    
    # Regime features (from HMM)
    regime, confidence = regime_detector.predict_regime(dataframe)
    dataframe['regime'] = regime
    dataframe['regime_confidence'] = confidence
    
    # Regime-specific indicators
    if regime == 'trending':
        # Momentum indicators
    elif regime == 'high_volatility':
        # Volatility indicators
    else:  # low_volatility
        # Mean reversion indicators
    
    return dataframe
```

**Decision Record:** [ADR-002](./decisions/ADR-002-freqai-setup.md)

---

### 4. Strategy Logic

**Responsibility:** تصمیم‌گیری برای ورود/خروج بر اساس regime و ML.

**Pattern:** Strategy Pattern با Regime Switching

**Interface:**
```python
class PragmaAdaptiveScalper(IStrategy):
    def populate_indicators(df: DataFrame) -> DataFrame
    def populate_entry_trend(df: DataFrame) -> DataFrame
    def populate_exit_trend(df: DataFrame) -> DataFrame
    
    # Regime-specific sub-strategies
    def entry_logic_bull(df: DataFrame) -> Series
    def entry_logic_bear(df: DataFrame) -> Series
    def entry_logic_sideways(df: DataFrame) -> Series
```

**Entry Logic:**
```python
# Conditions:
1. FreqAI prediction > threshold
2. Regime confidence > 0.6
3. Regime-specific technical confirmations
4. Volume > average
5. Not in cooldown period
```

**Exit Logic:**
```python
# Conditions:
1. Profit target hit (regime-dependent)
2. Stop loss hit (dynamic)
3. ML prediction reversal
4. Regime change detected
5. Time-based exit (12 candles = 1 hour)
```

**Decision Record:** [ADR-003](./decisions/ADR-003-strategy-logic.md)

---

### 5. Risk Management

**Responsibility:** مدیریت position size, stops, و drawdown protection.

**Components:**

#### 5.1 Kelly Criterion Position Sizing
```python
def custom_stake_amount(...) -> float:
    # Get historical stats
    trades = Trade.get_recent(pair, limit=50)
    win_rate = calculate_win_rate(trades)
    avg_win = calculate_avg_win(trades)
    avg_loss = calculate_avg_loss(trades)
    
    # Kelly formula
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    
    # Safety: Fractional Kelly (25%)
    safe_kelly = kelly_fraction * 0.25
    
    # Regime adjustment
    regime_multiplier = {'trending': 1.2, 'high_volatility': 0.6, 'low_volatility': 0.8}
    adjusted = safe_kelly * regime_multiplier[regime]
    
    return proposed_stake * adjusted
```

#### 5.2 Dynamic ATR Stop Loss
```python
def custom_stoploss(...) -> float:
    atr = dataframe.iloc[-1]['atr']
    
    # Breakeven protection
    if current_profit > 0.02:  # 2%
        return stoploss_from_absolute(trade.open_rate * 1.005, ...)
    
    # ATR trailing
    if current_profit > 0.01:  # 1%
        trailing_stop = current_rate - (atr * 2)
        return stoploss_from_absolute(trailing_stop, ...)
    
    return self.stoploss  # -3%
```

#### 5.3 Confidence-Based DCA
```python
def adjust_trade_position(...) -> Optional[float]:
    # Only if drawdown AND high ML confidence
    if current_profit < -0.02 and ml_confidence > 0.7:
        # DCA with half of original stake
        return trade.stake_amount * 0.5
    
    return None
```

**Decision Record:** [ADR-004](./decisions/ADR-004-risk-management.md)

---

## 🔄 Data Flow

### Training Flow
```
1. Download Data (180 days)
     ↓
2. Feature Engineering (85+ features)
     ↓
3. Train HMM (500 candles)
     ↓
4. Add Regime Features
     ↓
5. Train FreqAI Model (30 days train, 7 days test)
     ↓
6. Validate Model (walk-forward)
     ↓
7. Save Model with metadata
```

### Live Trading Flow
```
1. New Candle Arrives
     ↓
2. Update Indicators
     ↓
3. Predict Regime (HMM)
     ↓
4. Add Regime Features
     ↓
5. FreqAI Prediction
     ↓
6. Strategy Logic (entry/exit)
     ↓
7. Risk Management (position sizing, stops)
     ↓
8. Execute Order (if all conditions met)
     ↓
9. Log & Monitor
```

### Auto-Retraining Flow
```
Every 15 days (or 360 hours):
1. Download latest data
     ↓
2. Retrain HMM (if needed)
     ↓
3. Retrain FreqAI model
     ↓
4. Validate new model vs old model
     ↓
5. If better: Deploy new model
   If worse: Keep old model, alert
     ↓
6. Log results
```

---

## 📊 Database Schema

### Trade History (Freqtrade built-in)
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    pair VARCHAR,
    open_date DATETIME,
    close_date DATETIME,
    profit_percent FLOAT,
    regime VARCHAR,  -- Custom field
    ml_confidence FLOAT,  -- Custom field
    ...
)
```

### Regime History (Custom)
```sql
CREATE TABLE regime_history (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    pair VARCHAR,
    regime VARCHAR,
    confidence FLOAT,
    transition_from VARCHAR
)
```

### Model Performance (Custom)
```sql
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY,
    model_version VARCHAR,
    train_date DATETIME,
    test_sharpe FLOAT,
    test_accuracy FLOAT,
    deployed BOOLEAN
)
```

---

## 🔧 Technology Stack

### Core
- **Python:** 3.11+
- **Freqtrade:** 2024.x
- **OS:** Windows/Linux

### ML/AI
- **FreqAI:** Built-in Freqtrade
- **XGBoost:** 2.0+
- **CatBoost:** 1.2+
- **hmmlearn:** 0.3+
- **scikit-learn:** 1.3+

### Data & Analysis
- **Pandas:** 2.0+
- **NumPy:** 1.24+
- **TA-Lib:** 0.4+

### Testing
- **pytest:** 7.4+
- **hypothesis:** 6.82+

### DevOps (Future)
- **Docker:** 24.0+
- **GitHub Actions:** CI/CD

---

## 🧪 Testing Strategy

### 1. Unit Tests
```
tests/unit/
├── test_regime_detector.py
├── test_feature_engineering.py
├── test_kelly_criterion.py
└── test_dynamic_stoploss.py
```

### 2. Integration Tests
```
tests/integration/
├── test_strategy_pipeline.py
├── test_freqai_integration.py
└── test_risk_management.py
```

### 3. Backtest Validation
```
- Walk-forward testing
- Multiple market regimes
- Stress testing
- Monte Carlo simulation
```

**Target Coverage:** > 80%

---

## 📈 Performance Monitoring

### Metrics Tracked
1. **Trading Performance**
   - Sharpe Ratio
   - Max Drawdown
   - Win Rate
   - Profit Factor

2. **ML Performance**
   - Prediction accuracy
   - Model confidence
   - Feature importance

3. **Regime Detection**
   - Regime stability
   - Transition frequency
   - Detection confidence

4. **System Health**
   - Retraining frequency
   - Model age
   - Error rate

---

## 🔐 Security Considerations

1. **API Keys:** Environment variables only
2. **Sensitive Data:** Never commit to git
3. **Database:** Encrypted at rest
4. **Backups:** Automated daily

---

## 🚀 Deployment

### Local Development
```bash
python -m venv venv
pip install -r requirements.txt
freqtrade backtesting --strategy PragmaAdaptiveScalper
```

### Production (Future)
```bash
docker-compose up -d
```

---

## 📝 Decision Records

All architectural decisions are documented in:
- [ADR-001: HMM Implementation](./decisions/ADR-001-hmm-implementation.md)
- [ADR-002: FreqAI Setup](./decisions/ADR-002-freqai-setup.md)
- [ADR-003: Strategy Logic](./decisions/ADR-003-strategy-logic.md)
- [ADR-004: Risk Management](./decisions/ADR-004-risk-management.md)

---

**Last Updated:** 2025-10-11  
**Next Review:** After Phase 1 completion
