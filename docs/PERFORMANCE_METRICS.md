# Performance Metrics: Pragma Trading Bot

**نسخه:** 1.0  
**تاریخ:** 2025-10-12  
**وضعیت:** 📋 Complete

---

## 📊 Key Performance Indicators (KPIs)

### 1. Trading Performance Metrics

#### **Sharpe Ratio**
- **Target:** > 1.5
- **Current:** TBD (Live trading)
- **Calculation:** `(Portfolio Return - Risk-Free Rate) / Portfolio Volatility`

#### **Maximum Drawdown**
- **Target:** < 3%
- **Current:** TBD (Live trading)
- **Calculation:** `Max(Peak - Trough) / Peak`

#### **Win Rate**
- **Target:** > 70%
- **Current:** TBD (Live trading)
- **Calculation:** `Winning Trades / Total Trades`

#### **Average Daily Return**
- **Target:** 1-2%
- **Current:** TBD (Live trading)
- **Calculation:** `Daily P&L / Portfolio Value`

#### **Profit Factor**
- **Target:** > 1.5
- **Current:** TBD (Live trading)
- **Calculation:** `Gross Profit / Gross Loss`

### 2. Risk Management Metrics

#### **Kelly Criterion Utilization**
- **Target:** 15-25% of optimal Kelly
- **Current:** Implemented
- **Monitoring:** Position sizing accuracy

#### **Stop Loss Effectiveness**
- **Target:** 80% of stops prevent larger losses
- **Current:** Implemented
- **Monitoring:** Stop loss hit rate vs. profit protection

#### **Circuit Breaker Triggers**
- **Target:** < 1 trigger per month
- **Current:** Implemented
- **Monitoring:** Drawdown protection activation

### 3. Machine Learning Metrics

#### **HMM Regime Detection Accuracy**
- **Target:** > 75%
- **Current:** TBD (Backtesting)
- **Calculation:** `Correct Regime Predictions / Total Predictions`

#### **ML Model Performance**
- **Target:** R² > 0.6
- **Current:** TBD (Backtesting)
- **Monitoring:** Prediction accuracy vs. actual returns

#### **Feature Importance Stability**
- **Target:** Top 5 features consistent across retraining
- **Current:** Implemented
- **Monitoring:** Feature ranking changes

### 4. System Performance Metrics

#### **Response Time**
- **Target:** < 100ms average
- **Current:** TBD (Live trading)
- **Monitoring:** Strategy execution time

#### **Memory Usage**
- **Target:** < 2GB RAM
- **Current:** TBD (Live trading)
- **Monitoring:** System resource consumption

#### **CPU Usage**
- **Target:** < 5% average
- **Current:** TBD (Live trading)
- **Monitoring:** Processing overhead

#### **Uptime**
- **Target:** > 99.9%
- **Current:** TBD (Live trading)
- **Monitoring:** System availability

---

## 📈 Performance Monitoring Dashboard

### Real-Time Metrics

```python
# Performance monitoring script
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_time = datetime.now()
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def calculate_win_rate(self, trades):
        """Calculate win rate"""
        winning_trades = trades[trades['pnl'] > 0]
        return len(winning_trades) / len(trades)
    
    def get_daily_metrics(self):
        """Get daily performance metrics"""
        today = datetime.now().date()
        
        # Get today's trades
        today_trades = self.get_trades_for_date(today)
        
        # Calculate metrics
        daily_return = today_trades['pnl'].sum() / self.portfolio_value
        win_rate = self.calculate_win_rate(today_trades)
        
        return {
            'date': today,
            'daily_return': daily_return,
            'win_rate': win_rate,
            'total_trades': len(today_trades),
            'total_pnl': today_trades['pnl'].sum()
        }
```

### Historical Performance Analysis

```python
def analyze_historical_performance(data, lookback_days=30):
    """Analyze historical performance"""
    
    # Calculate rolling metrics
    data['rolling_sharpe'] = data['returns'].rolling(30).apply(
        lambda x: calculate_sharpe_ratio(x)
    )
    
    data['rolling_drawdown'] = data['cumulative_returns'].rolling(30).apply(
        lambda x: calculate_max_drawdown(x)
    )
    
    # Regime-specific performance
    regime_performance = data.groupby('regime').agg({
        'returns': ['mean', 'std', 'count'],
        'pnl': ['sum', 'mean']
    })
    
    return {
        'overall_metrics': {
            'sharpe_ratio': calculate_sharpe_ratio(data['returns']),
            'max_drawdown': calculate_max_drawdown(data['cumulative_returns']),
            'win_rate': calculate_win_rate(data),
            'total_return': data['cumulative_returns'].iloc[-1] - 1
        },
        'regime_performance': regime_performance,
        'rolling_metrics': data[['rolling_sharpe', 'rolling_drawdown']]
    }
```

---

## 🎯 Performance Targets by Regime

### Trending Market
- **Sharpe Ratio:** > 2.0
- **Win Rate:** > 80%
- **Max Drawdown:** < 2%
- **Daily Trades:** 15-25

### Low Volatility Market
- **Sharpe Ratio:** > 1.0
- **Win Rate:** > 60%
- **Max Drawdown:** < 1%
- **Daily Trades:** 5-10

### High Volatility Market
- **Sharpe Ratio:** > 1.5
- **Win Rate:** > 70%
- **Max Drawdown:** < 4%
- **Daily Trades:** 10-20

---

## 📊 Performance Reporting

### Daily Report Template

```
=== PRAGMA TRADING BOT - DAILY REPORT ===
Date: 2025-10-12
Status: ACTIVE

TRADING PERFORMANCE:
- Daily Return: +1.2%
- Total Trades: 18
- Win Rate: 77.8%
- Sharpe Ratio: 1.8
- Max Drawdown: -0.8%

RISK MANAGEMENT:
- Kelly Utilization: 22%
- Stop Loss Hits: 3
- Circuit Breaker: OK
- Position Sizes: Normal

MACHINE LEARNING:
- HMM Accuracy: 78%
- ML Model R²: 0.65
- Feature Stability: Good
- Last Retrain: 2025-10-10

SYSTEM PERFORMANCE:
- Uptime: 99.9%
- Response Time: 45ms
- Memory Usage: 1.2GB
- CPU Usage: 3.2%

REGIME ANALYSIS:
- Current Regime: Trending
- Confidence: 85%
- Regime Duration: 3 days
- Expected Performance: Good

NEXT ACTIONS:
- Monitor regime transition
- Check ML model performance
- Review risk parameters
```

### Weekly Report Template

```
=== PRAGMA TRADING BOT - WEEKLY REPORT ===
Week: 2025-10-06 to 2025-10-12
Status: EXCELLENT

WEEKLY PERFORMANCE:
- Weekly Return: +8.5%
- Total Trades: 125
- Win Rate: 74.4%
- Sharpe Ratio: 1.9
- Max Drawdown: -1.2%

REGIME BREAKDOWN:
- Trending: 4 days (57%) - +6.2% return
- Low Vol: 2 days (29%) - +1.8% return
- High Vol: 1 day (14%) - +0.5% return

RISK MANAGEMENT:
- Kelly Utilization: 20% average
- Stop Loss Effectiveness: 85%
- Circuit Breaker: 0 triggers
- Position Sizing: Optimal

MACHINE LEARNING:
- HMM Accuracy: 76% average
- ML Model Performance: R² = 0.68
- Feature Importance: Stable
- Retraining: Completed 1x

SYSTEM PERFORMANCE:
- Uptime: 99.95%
- Average Response: 42ms
- Memory Usage: 1.1GB average
- CPU Usage: 2.8% average

LESSONS LEARNED:
- Trending regime performed exceptionally well
- Low volatility regime needs parameter tuning
- ML model showing good stability

RECOMMENDATIONS:
- Increase position size in trending regime
- Adjust parameters for low volatility
- Continue monitoring ML performance
```

---

## 🚨 Performance Alerts

### Critical Alerts
- **Drawdown > 5%:** Immediate stop trading
- **Win Rate < 50%:** Review strategy parameters
- **Sharpe Ratio < 0.5:** Check model performance
- **System Uptime < 95%:** Investigate technical issues

### Warning Alerts
- **Drawdown > 3%:** Reduce position sizes
- **Win Rate < 60%:** Monitor closely
- **Sharpe Ratio < 1.0:** Consider retraining
- **Response Time > 200ms:** Check system load

### Information Alerts
- **Daily Return > 3%:** Excellent performance
- **Win Rate > 80%:** Outstanding performance
- **Sharpe Ratio > 2.0:** Exceptional performance
- **Zero trades in 24h:** Check market conditions

---

## 📈 Performance Optimization

### 1. Strategy Optimization

```python
def optimize_strategy_parameters(data, parameter_ranges):
    """Optimize strategy parameters using grid search"""
    
    best_params = None
    best_sharpe = -np.inf
    
    for params in parameter_ranges:
        # Test parameters
        results = backtest_strategy(data, params)
        sharpe = calculate_sharpe_ratio(results['returns'])
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
    
    return best_params, best_sharpe
```

### 2. Risk Management Optimization

```python
def optimize_risk_parameters(trades, risk_ranges):
    """Optimize risk management parameters"""
    
    best_risk_params = None
    best_risk_metric = -np.inf
    
    for risk_params in risk_ranges:
        # Simulate with different risk parameters
        simulated_trades = simulate_risk_management(trades, risk_params)
        risk_metric = calculate_risk_adjusted_return(simulated_trades)
        
        if risk_metric > best_risk_metric:
            best_risk_metric = risk_metric
            best_risk_params = risk_params
    
    return best_risk_params, best_risk_metric
```

### 3. ML Model Optimization

```python
def optimize_ml_parameters(X, y, model_ranges):
    """Optimize ML model parameters"""
    
    best_model_params = None
    best_model_score = -np.inf
    
    for model_params in model_ranges:
        # Train model with parameters
        model = train_model(X, y, model_params)
        score = evaluate_model(model, X, y)
        
        if score > best_model_score:
            best_model_score = score
            best_model_params = model_params
    
    return best_model_params, best_model_score
```

---

## 📊 Performance Visualization

### 1. Performance Charts

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_performance_dashboard(data):
    """Create comprehensive performance dashboard"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Cumulative returns
    axes[0, 0].plot(data['cumulative_returns'])
    axes[0, 0].set_title('Cumulative Returns')
    
    # Drawdown
    axes[0, 1].fill_between(data.index, data['drawdown'], 0, alpha=0.3)
    axes[0, 1].set_title('Drawdown')
    
    # Rolling Sharpe
    axes[0, 2].plot(data['rolling_sharpe'])
    axes[0, 2].set_title('Rolling Sharpe Ratio')
    
    # Regime distribution
    regime_counts = data['regime'].value_counts()
    axes[1, 0].pie(regime_counts.values, labels=regime_counts.index)
    axes[1, 0].set_title('Regime Distribution')
    
    # Trade distribution
    axes[1, 1].hist(data['trade_pnl'], bins=20)
    axes[1, 1].set_title('Trade P&L Distribution')
    
    # Monthly returns
    monthly_returns = data['returns'].resample('M').sum()
    axes[1, 2].bar(monthly_returns.index, monthly_returns.values)
    axes[1, 2].set_title('Monthly Returns')
    
    plt.tight_layout()
    plt.show()
```

### 2. Real-Time Monitoring

```python
def create_realtime_dashboard():
    """Create real-time performance dashboard"""
    
    # This would integrate with a web dashboard
    # For now, return key metrics
    return {
        'current_return': get_current_return(),
        'today_trades': get_today_trades(),
        'current_regime': get_current_regime(),
        'system_status': get_system_status()
    }
```

---

## 📝 Performance Logging

### 1. Performance Logger

```python
import logging
from datetime import datetime

class PerformanceLogger:
    def __init__(self, log_file='performance.log'):
        self.logger = logging.getLogger('performance')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_trade(self, trade_data):
        """Log individual trade"""
        self.logger.info(f"TRADE: {trade_data}")
    
    def log_daily_metrics(self, metrics):
        """Log daily performance metrics"""
        self.logger.info(f"DAILY: {metrics}")
    
    def log_alert(self, alert_type, message):
        """Log performance alerts"""
        self.logger.warning(f"ALERT {alert_type}: {message}")
```

### 2. Performance Database

```python
import sqlite3

class PerformanceDatabase:
    def __init__(self, db_file='performance.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()
    
    def create_tables(self):
        """Create performance tables"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                daily_return REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER
            )
        ''')
    
    def insert_daily_metrics(self, metrics):
        """Insert daily metrics"""
        self.conn.execute('''
            INSERT OR REPLACE INTO daily_metrics VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metrics['date'],
            metrics['daily_return'],
            metrics['win_rate'],
            metrics['sharpe_ratio'],
            metrics['max_drawdown'],
            metrics['total_trades']
        ))
        self.conn.commit()
```

---

**Last Updated:** 2025-10-12  
**Next Review:** After live trading data available
