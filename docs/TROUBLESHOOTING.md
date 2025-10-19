# Troubleshooting Guide: Pragma Trading Bot

**نسخه:** 1.0  
**تاریخ:** 2025-10-12  
**وضعیت:** 📋 Complete

---

## 🚨 Common Issues & Solutions

### 1. Import Errors

#### Problem: `ModuleNotFoundError: No module named 'regime'`

**Symptoms:**
```
ModuleNotFoundError: No module named 'regime'
ModuleNotFoundError: No module named 'risk'
ModuleNotFoundError: No module named 'ml'
```

**Solutions:**
```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH="/path/to/pragma-trading-bot/src:$PYTHONPATH"

# Option 2: Install in development mode
cd /path/to/pragma-trading-bot
pip install -e .

# Option 3: Copy modules to Freqtrade
cp -r src/regime /path/to/freqtrade/user_data/strategies/
cp -r src/risk /path/to/freqtrade/user_data/strategies/
cp -r src/ml /path/to/freqtrade/user_data/strategies/
```

#### Problem: `ImportError: No module named 'hmmlearn'`

**Solutions:**
```bash
# Install missing dependencies
pip install hmmlearn scikit-learn

# Or install all requirements
pip install -r requirements.txt
```

### 2. Data Issues

#### Problem: `KeyError: 'close'` or missing columns

**Symptoms:**
```
KeyError: 'close'
KeyError: 'high'
KeyError: 'low'
```

**Solutions:**
```bash
# Check data format
freqtrade show-trades --config config.json

# Download fresh data
freqtrade download-data --exchange binance --pairs BTC/USDT --timeframes 5m --days 180

# Verify data structure
python -c "
import pandas as pd
df = pd.read_csv('user_data/data/binance/BTC_USDT-5m.csv')
print(df.columns.tolist())
print(df.head())
"
```

#### Problem: `ValueError: Insufficient training data`

**Solutions:**
```python
# Increase lookback period
detector.train(dataframe, lookback=1000)  # Instead of 500

# Check data availability
print(f"Data length: {len(dataframe)}")
print(f"Required: {1000}")  # min_training_samples
```

### 3. Model Training Issues

#### Problem: `ValueError: Input contains NaN, infinity or a value too large`

**Solutions:**
```python
# Clean data before training
dataframe = dataframe.dropna()
dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
dataframe = dataframe.dropna()

# Check for infinite values
print(f"NaN values: {dataframe.isnull().sum().sum()}")
print(f"Inf values: {np.isinf(dataframe.select_dtypes(include=[np.number])).sum().sum()}")
```

#### Problem: `ConvergenceWarning: Maximum number of iterations reached`

**Solutions:**
```python
# Increase iterations
model = GaussianHMM(
    n_components=3,
    n_iter=200,  # Instead of 100
    random_state=42
)

# Or reduce complexity
model = GaussianHMM(
    n_components=2,  # Instead of 3
    n_iter=100,
    random_state=42
)
```

### 4. Risk Management Issues

#### Problem: `ZeroDivisionError: float division by zero`

**Symptoms:**
```
ZeroDivisionError: float division by zero
  File "src/risk/kelly_criterion.py", line 45, in calculate_position_size
    leverage = position_value / account_balance
```

**Solutions:**
```python
# Check account balance
if account_balance > 0:
    leverage = position_value / account_balance
else:
    leverage = 0.0

# Or set minimum balance
account_balance = max(account_balance, 1.0)
```

#### Problem: `ValueError: Kelly fraction must be between 0 and 1`

**Solutions:**
```python
# Clamp Kelly fraction
kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Max 25%

# Check win rate and loss ratio
if win_rate <= 0 or win_rate >= 1:
    kelly_fraction = 0.0
```

### 5. Freqtrade Integration Issues

#### Problem: `ValueError: Strategy not found`

**Solutions:**
```bash
# Check strategy file location
ls -la /path/to/freqtrade/user_data/strategies/RegimeAdaptiveStrategy.py

# Copy strategy file
cp src/strategies/regime_adaptive_strategy.py /path/to/freqtrade/user_data/strategies/

# Check strategy name in config
grep -i strategy config.json
```

#### Problem: `AttributeError: 'DataFrame' object has no attribute 'rolling'`

**Solutions:**
```python
# Ensure pandas DataFrame
import pandas as pd
if not isinstance(dataframe, pd.DataFrame):
    dataframe = pd.DataFrame(dataframe)

# Check data types
print(dataframe.dtypes)
```

### 6. Performance Issues

#### Problem: High memory usage

**Solutions:**
```python
# Reduce data size
dataframe = dataframe.tail(1000)  # Keep last 1000 rows

# Use chunked processing
for chunk in pd.read_csv('data.csv', chunksize=1000):
    process_chunk(chunk)

# Clear unused variables
del large_dataframe
import gc
gc.collect()
```

#### Problem: Slow model training

**Solutions:**
```python
# Reduce features
features = features[['close', 'volume', 'rsi', 'atr']]

# Use smaller lookback
detector.train(dataframe, lookback=500)

# Parallel processing
from joblib import Parallel, delayed
```

### 7. Configuration Issues

#### Problem: `KeyError: 'exchange'` in config

**Solutions:**
```json
{
  "exchange": {
    "name": "binance",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_SECRET_KEY"
  }
}
```

#### Problem: `ValueError: Invalid timeframe`

**Solutions:**
```json
{
  "timeframe": "5m",  // Valid: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
  "stake_currency": "USDT",
  "stake_amount": "unlimited"
}
```

---

## 🔍 Debugging Techniques

### 1. Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in Freqtrade
freqtrade trade --strategy RegimeAdaptiveStrategy --config config.json --verbosity 3
```

### 2. Check Data Quality

```python
def check_data_quality(df):
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types: {df.dtypes}")
    print(f"NaN values: {df.isnull().sum()}")
    print(f"Inf values: {np.isinf(df.select_dtypes(include=[np.number])).sum()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df
```

### 3. Validate Model Performance

```python
def validate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Check regime distribution
    unique, counts = np.unique(predictions, return_counts=True)
    for regime, count in zip(unique, counts):
        print(f"Regime {regime}: {count} samples ({count/len(predictions):.1%})")
```

### 4. Monitor Memory Usage

```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"Memory percent: {process.memory_percent():.1f}%")
```

---

## 📞 Getting Help

### 1. Check Logs

```bash
# Freqtrade logs
tail -f /path/to/freqtrade/logs/freqtrade.log

# Bot logs
tail -f /path/to/logs/pragma-bot.log

# System logs
journalctl -u pragma-bot -f
```

### 2. Collect Debug Information

```bash
# Create debug package
tar -czf debug-info-$(date +%Y%m%d).tar.gz \
  logs/ \
  config.json \
  user_data/ \
  /var/log/syslog
```

### 3. Common Commands

```bash
# Check bot status
freqtrade status --config config.json

# Show trades
freqtrade show-trades --config config.json

# Show profit
freqtrade profit --config config.json

# Test pairlist
freqtrade test-pairlist --config config.json
```

---

## 🚨 Emergency Procedures

### 1. Stop Trading Immediately

```bash
# Stop Freqtrade
pkill -f freqtrade

# Or if using systemd
sudo systemctl stop pragma-bot

# Check for open positions
freqtrade show-trades --config config.json
```

### 2. Emergency Configuration

```json
{
  "max_open_trades": 0,
  "dry_run": true,
  "stake_amount": 0
}
```

### 3. Data Recovery

```bash
# Backup current data
cp -r user_data/data user_data/data_backup_$(date +%Y%m%d)

# Restore from backup
cp -r user_data/data_backup_20241012 user_data/data
```

---

## 📚 Additional Resources

- **Freqtrade Documentation:** https://www.freqtrade.io/
- **HMM Documentation:** https://hmmlearn.readthedocs.io/
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/

---

**Last Updated:** 2025-10-12  
**Next Review:** After major updates
