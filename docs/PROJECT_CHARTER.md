# Project Charter: Pragma Trading Bot

**تاریخ شروع:** 2025-10-11  
**وضعیت:** 🟢 Active  
**مالک:** Your Name  

---

## 🎯 Vision

ایجاد یک سیستم معاملاتی هوشمند و خودکار که:
1. **Self-Adaptive** - خودش با بازار تطبیق می‌کند
2. **Anti-Overfitting** - دچار overfitting نمی‌شود
3. **Production-Ready** - آماده استفاده واقعی
4. **Low-Maintenance** - نیاز به دخالت دستی کمتری دارد

---

## 🎬 Problem Statement

### مشکل فعلی
استراتژی موجود (MtfScalperOptimized) مشکلات زیر را دارد:

1. **Overfitting شدید**
   - Sep 2025: +1.03% profit ✅
   - Jul-Oct 2025: -2.13% profit ❌
   - تفاوت: -305% performance

2. **نیاز به Hyperopt مداوم**
   - هر چند هفته باید دوباره optimize شود
   - زمان‌بر و خطرناک

3. **عدم Regime Awareness**
   - یک strategy برای همه شرایط
   - Performance ضعیف در بازارهای مختلف

4. **Risk Management ضعیف**
   - Fixed position sizing
   - Static stop loss
   - عدم protection در برابر drawdown

---

## ✅ Success Criteria

### 1. Technical KPIs

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| **Sharpe Ratio** | 1.85 → -2.62 | > 1.5 | > 2.0 |
| **Max Drawdown** | 5.24% | < 3% | < 2% |
| **Win Rate** | 86.3% | > 70% | > 75% |
| **Trades/Day** | 3.8 | 10-20 | 15-25 |
| **Profit Factor** | 0.90 | > 1.3 | > 1.5 |
| **Daily Return** | Variable | 1-2% | 2-3% |

### 2. System KPIs

| Metric | Target |
|--------|--------|
| **Auto-Retraining** | Every 15 days |
| **Model Expiration** | Max 30 days |
| **Regime Detection Accuracy** | > 75% |
| **Uptime** | > 99% |
| **Alert Response Time** | < 5 minutes |

### 3. Maintenance KPIs

| Metric | Current | Target |
|--------|---------|--------|
| **Manual Hyperopt** | Every 2-4 weeks | Never |
| **Parameter Tuning** | Weekly | Monthly (validation only) |
| **Intervention Required** | High | Low |

---

## 🚫 Non-Goals

چیزهایی که **نمی**‌خواهیم انجام دهیم:

1. ❌ **Custom ML از صفر** - استفاده از FreqAI
2. ❌ **Over-engineering** - ساده نگه داریم
3. ❌ **Perfect prediction** - Focus روی risk management
4. ❌ **High-frequency trading** - Scalping متعادل
5. ❌ **Multiple exchange support** - فقط Binance (فعلاً)

---

## 🏗️ Solution Architecture

### Core Components

```
┌─────────────────────────────────────────────────┐
│         Market Data Pipeline                    │
├─────────────────────────────────────────────────┤
│  • Data fetching & validation                   │
│  • Feature engineering (85+ features)           │
│  • Outlier detection & cleaning                 │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│      HMM Regime Detection                       │
├─────────────────────────────────────────────────┤
│  • 3-State Model (Bull/Bear/Sideways)           │
│  • Features: Returns, Volatility, Volume, Trend │
│  • Confidence scoring                           │
│  • Transition probability tracking              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│       FreqAI Machine Learning                   │
├─────────────────────────────────────────────────┤
│  • XGBoost/CatBoost models                      │
│  • Auto-retraining every 15 days                │
│  • Regime-specific targets                      │
│  • Confidence-based filtering                   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│      Strategy Logic                             │
├─────────────────────────────────────────────────┤
│  • Regime-aware entry/exit                      │
│  • Multi-strategy ensemble                      │
│  • ML prediction integration                    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│    Risk Management Layer                        │
├─────────────────────────────────────────────────┤
│  • Kelly Criterion position sizing              │
│  • Dynamic ATR-based stops                      │
│  • Confidence-based DCA                         │
│  • Circuit breakers                             │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Execution & Monitoring                  │
├─────────────────────────────────────────────────┤
│  • Order management                             │
│  • Performance tracking                         │
│  • Telegram alerts                              │
│  • Logging & debugging                          │
└─────────────────────────────────────────────────┘
```

---

## 📅 Timeline

### Phase 1: Foundation (Week 1-2)
**Goal:** HMM + FreqAI infrastructure

- Week 1, Days 1-3: HMM Regime Detector
- Week 1, Days 4-5: Integration testing
- Week 2, Days 1-3: FreqAI setup
- Week 2, Days 4-5: Baseline testing

**Deliverable:** Working HMM + FreqAI prototype

### Phase 2: Strategy Development (Week 3)
**Goal:** Regime-aware adaptive strategy

- Days 1-2: Regime-specific strategies
- Days 3-4: Risk management callbacks
- Day 5: Integration & testing

**Deliverable:** Complete adaptive strategy

### Phase 3: Production Readiness (Week 4-5)
**Goal:** Deployment & monitoring

- Week 4: Walk-forward validation
- Week 5: Production deployment & monitoring

**Deliverable:** Production-ready system

---

## 💰 Expected ROI

### Time Investment
- Development: 4-5 weeks
- Learning curve: Included in development
- Maintenance: ~2 hours/month (vs ~8 hours/month)

### Expected Benefits

| Benefit | Impact |
|---------|--------|
| **Reduced Maintenance** | -75% time |
| **Better Risk Management** | -40% drawdown |
| **Consistent Performance** | +50% Sharpe |
| **Scalability** | Multi-pair ready |
| **Peace of Mind** | Auto-adaptive |

---

## 🎓 Learning Objectives

این پروژه فرصتی برای یادگیری:

1. **HMM Applications** - Regime detection
2. **FreqAI Advanced Usage** - Auto-retraining, custom features
3. **Production ML Systems** - Deployment, monitoring
4. **Risk Management** - Kelly Criterion, dynamic strategies
5. **Software Engineering** - Git workflow, testing, CI/CD

---

## 📊 Validation Strategy

### 1. Walk-Forward Testing
```
Train: Jan-Mar 2024 → Test: Apr 2024
Train: Feb-Apr 2024 → Test: May 2024
...
Train: Aug-Oct 2025 → Test: Sep-Oct 2025
```

### 2. Regime Stability Testing
- Bull market performance
- Bear market performance
- Sideways market performance
- Regime transition handling

### 3. Stress Testing
- Flash crash simulation
- Extended drawdown scenarios
- High volatility periods
- Low liquidity conditions

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **HMM overfitting** | High | Medium | Validate on multiple regimes |
| **ML model degradation** | High | Medium | Auto-retraining every 15 days |
| **Market regime change** | Medium | High | Multi-regime testing |
| **Technical bugs** | Medium | Low | Comprehensive testing |
| **Over-complexity** | Low | Medium | Keep it pragmatic |

---

## 🤝 Stakeholders

- **Developer:** You
- **User:** You
- **Reviewers:** Community (optional)
- **Advisors:** Freqtrade community, QuantStart resources

---

## 📝 Approval

**Approved by:** Your Name  
**Date:** 2025-10-11  
**Next Review:** After Phase 1 completion

---

## 🔄 Change Log

| Date | Change | Reason |
|------|--------|--------|
| 2025-10-11 | Initial charter | Project kickoff |

---

**Signature:** _________________________  
**Date:** 2025-10-11
