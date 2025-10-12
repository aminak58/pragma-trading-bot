# 🤖 Pragma Trading Bot

**Production-Ready Adaptive Trading System**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Freqtrade](https://img.shields.io/badge/freqtrade-2024.x-green.svg)](https://www.freqtrade.io/)
[![License](https://img.shields.io/badge/license-Private-red.svg)]()

## 🎯 Overview

Pragma is an intelligent, self-adaptive trading bot that combines:
- **HMM Regime Detection** - Market state awareness
- **FreqAI Machine Learning** - Predictive modeling with auto-retraining
- **Reinforcement Learning** - Adaptive execution (optional)
- **Intelligent Risk Management** - Kelly Criterion, Dynamic stops, Smart DCA

## ✨ Key Features

### 🧠 Intelligence
- ✅ **Anti-Overfitting** - Auto-retraining every 15 days
- ✅ **Regime-Aware** - 3-state HMM (Bull/Bear/Sideways)
- ✅ **ML-Powered** - FreqAI with XGBoost/CatBoost
- ✅ **Adaptive** - Learns from market conditions

### 🛡️ Risk Management
- ✅ **Kelly Criterion** - Optimal position sizing
- ✅ **Dynamic Stops** - ATR-based trailing
- ✅ **Smart DCA** - Confidence-based position adjustment
- ✅ **Circuit Breakers** - Drawdown protection

### 📊 Performance Targets
- **Sharpe Ratio:** > 1.5
- **Max Drawdown:** < 3%
- **Win Rate:** > 70%
- **Daily Trades:** 10-20 (balanced)
- **Daily Return:** 1-2% target

## 🏗️ Architecture

```
Market Data → HMM Regime Detection → FreqAI ML → Strategy Logic → Risk Management → Execution
```

### Components
- **src/regime/** - HMM-based regime detection
- **src/strategies/** - Trading strategies
- **src/risk/** - Risk management modules
- **tests/** - Unit & integration tests
- **docs/** - Documentation & ADRs

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+
# Freqtrade 2024.x
# hmmlearn, stable-baselines3
```

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/pragma-trading-bot.git
cd pragma-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Freqtrade link
# (Will be documented in setup guide)
```

### Run Backtest
```bash
freqtrade backtesting \
  --strategy PragmaAdaptiveScalper \
  --config configs/pragma_config.json \
  --timerange 20240101-20250101
```

## 📋 Project Status

**Current Phase:** Foundation Complete ✅  
**Sprint:** 1 (Week 1-2)  
**Progress:** 🟢 75% Complete (3/4 issues done)

### Sprint 1 Progress
- [x] **Issue #2:** Environment Setup & Validation ✅
- [x] **Issue #3:** HMM Regime Detector Implementation ✅
- [x] **Issue #4:** HMM Unit Tests (99% coverage) ✅
- [x] **Issue #5:** Freqtrade Integration ✅

### Milestones
- [x] **Milestone 1:** Foundation (Week 1-2) ✅
  - [x] HMM Regime Detector
  - [x] Freqtrade Integration
  - [x] Regime-Adaptive Strategy
- [ ] **Milestone 2:** Risk Management (Week 3)
  - [ ] Kelly Criterion
  - [ ] Dynamic Stops
  - [ ] Position Adjustment
- [ ] **Milestone 3:** Production (Week 4-5)
  - [ ] Testing & Validation
  - [ ] Deployment
  - [ ] Monitoring

## 📚 Documentation

- [Project Charter](docs/PROJECT_CHARTER.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Integration Guide](docs/INTEGRATION_GUIDE.md) ⭐ **New!**
- [HMM Regime Detection](src/regime/README.md)
- [Contributing Guidelines](docs/CONTRIBUTING.md)

## 🔄 Development Workflow

### Branch Strategy
```
main (production)
  └── develop (development)
       └── feature/* (features)
```

### Git Workflow
```bash
# Start new feature
git checkout develop
git pull
git checkout -b feature/your-feature

# Work...
git add .
git commit -m "feat: description"
git push origin feature/your-feature

# Create PR: feature/your-feature → develop
```

## 📊 Performance History

### Baseline (MtfScalperOptimized)
- **Period:** Sep 2025 (1 month)
- **Profit:** +1.03%
- **Trades:** 95
- **Win Rate:** 88.4%
- **Max DD:** 1.38%

**Issue:** Manual hyperopt required, overfitting on recent data

### Target (Pragma v1.0)
- **Auto-retraining:** Every 15 days
- **Regime-aware:** 3-state HMM
- **Expected:** Better risk-adjusted returns with lower maintenance

## 🤝 Contributing

This is a private project. For collaboration inquiries, please contact the maintainer.

## 📄 License

Private - All Rights Reserved

## 🙏 Acknowledgments

- **Freqtrade Community** - Framework and strategies
- **QuantStart** - HMM regime detection methodology
- **FreqAI Examples** - ML integration patterns

---

**🚀 Pragma Trading Bot** - Built with precision, powered by intelligence
