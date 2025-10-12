# Validation Scripts

**Automated scripts for backtesting and validation workflow**

---

## Quick Start

```powershell
# 1. Download data (5-15 minutes)
.\scripts\1_download_data.ps1

# 2. Run backtest (5-30 minutes)
.\scripts\2_run_backtest.ps1

# 3. Analyze results (1-2 minutes)
.\scripts\3_analyze_results.ps1

# 4. Optimize (2-4 hours)
.\scripts\4_run_hyperopt.ps1
```

---

## Scripts Overview

### 1_download_data.ps1

**Purpose**: Download historical market data for backtesting

**Usage**:
```powershell
.\scripts\1_download_data.ps1
```

**What it does**:
- Downloads 180 days of 5m data
- From Binance exchange
- For 10 major pairs (BTC, ETH, BNB, SOL, etc.)
- Saves to `user_data/data/`

**Time**: 5-15 minutes

---

### 2_run_backtest.ps1

**Purpose**: Run Freqtrade backtest with RegimeAdaptiveStrategy

**Usage**:
```powershell
# Basic
.\scripts\2_run_backtest.ps1

# Custom timerange
.\scripts\2_run_backtest.ps1 -Timerange "20240801-20241010"

# With detailed breakdown
.\scripts\2_run_backtest.ps1 -Detailed

# Custom config
.\scripts\2_run_backtest.ps1 -Config "configs/my_config.json"
```

**Parameters**:
- `-Timerange`: Date range (default: 20240701-20241010)
- `-Config`: Config file path (default: configs/backtest_config.example.json)
- `-Detailed`: Show day/week/month breakdown

**What it does**:
- Copies strategy to `user_data/strategies/`
- Runs Freqtrade backtest
- Exports trade results
- Shows performance metrics

**Time**: 5-30 minutes

---

### 3_analyze_results.ps1

**Purpose**: Analyze backtest results and create report

**Usage**:
```powershell
.\scripts\3_analyze_results.ps1
```

**What it does**:
- Runs Freqtrade analysis
- Shows key metrics
- Creates report template in `docs/BACKTEST_RESULTS.md`
- Provides recommendations

**Time**: 1-2 minutes

---

### 4_run_hyperopt.ps1

**Purpose**: Optimize strategy parameters using hyperopt

**Usage**:
```powershell
# Standard (100 epochs, ~2-4 hours)
.\scripts\4_run_hyperopt.ps1

# Quick test (50 epochs)
.\scripts\4_run_hyperopt.ps1 -Epochs 50

# Specific spaces
.\scripts\4_run_hyperopt.ps1 -Spaces "buy"
.\scripts\4_run_hyperopt.ps1 -Spaces "sell"
.\scripts\4_run_hyperopt.ps1 -Spaces "buy sell"

# Different loss function
.\scripts\4_run_hyperopt.ps1 -Loss "SortinoHyperOptLoss"
.\scripts\4_run_hyperopt.ps1 -Loss "MaxDrawDownHyperOptLoss"

# Custom timerange
.\scripts\4_run_hyperopt.ps1 -Timerange "20240701-20240930"
```

**Parameters**:
- `-Epochs`: Number of iterations (default: 100)
- `-Loss`: Loss function (default: SharpeHyperOptLoss)
- `-Spaces`: Parameter spaces to optimize (default: "buy sell")
- `-Timerange`: Date range (default: 20240701-20241010)

**Loss Functions**:
- `SharpeHyperOptLoss` - Maximize Sharpe ratio (recommended)
- `SortinoHyperOptLoss` - Minimize downside risk
- `MaxDrawDownHyperOptLoss` - Minimize drawdown
- `CalmarHyperOptLoss` - Calmar ratio
- `OnlyProfitHyperOptLoss` - Simple profit

**What it does**:
- Tests different parameter combinations
- Finds optimal parameters
- Saves results to `user_data/hyperopt_results/`
- Shows best parameters

**Time**: 2-6 hours for 100 epochs

---

## Complete Workflow

### Step-by-Step Process

```powershell
# Step 1: Download data (once)
.\scripts\1_download_data.ps1

# Step 2: Initial backtest
.\scripts\2_run_backtest.ps1

# Step 3: Analyze results
.\scripts\3_analyze_results.ps1

# Step 4: If results are good, optimize
.\scripts\4_run_hyperopt.ps1

# Step 5: Apply optimal parameters to strategy
# (Edit src/strategies/regime_adaptive_strategy.py)

# Step 6: Re-test with optimal parameters
.\scripts\2_run_backtest.ps1

# Step 7: Validate on different period
.\scripts\2_run_backtest.ps1 -Timerange "20240401-20240630"
```

---

## Requirements

- Windows PowerShell 5.1+
- Freqtrade installed (`pip install freqtrade`)
- Virtual environment activated
- Sufficient disk space (5-10GB for data)
- Internet connection (for data download)

---

## Troubleshooting

### "Freqtrade not found"

```powershell
# Activate virtual environment
venv\Scripts\Activate.ps1

# Install Freqtrade
pip install freqtrade
```

### "Strategy not found"

```powershell
# Manually copy strategy
Copy-Item src/strategies/regime_adaptive_strategy.py user_data/strategies/
Copy-Item -Recurse src/regime user_data/strategies/
```

### "No data available"

```powershell
# Re-download data
.\scripts\1_download_data.ps1
```

### "Memory error during hyperopt"

```powershell
# Use fewer epochs
.\scripts\4_run_hyperopt.ps1 -Epochs 50

# Or close other applications
```

---

## Output Locations

```
user_data/
├── data/                    # Historical market data
│   └── binance/
│       ├── BTC_USDT-5m.json
│       ├── ETH_USDT-5m.json
│       └── ...
│
├── strategies/              # Strategy files (copied)
│   ├── regime_adaptive_strategy.py
│   └── regime/             # Dependencies
│
├── backtest_results/        # Backtest output
│   ├── backtest-result-{date}.json
│   └── ...
│
└── hyperopt_results/        # Hyperopt output
    ├── hyperopt_results.pkl
    └── ...

docs/
└── BACKTEST_RESULTS.md      # Analysis report
```

---

## Tips

### Faster Backtesting

- Use shorter timeranges for testing
- Reduce number of pairs in config
- Use cached data (`--cache default`)

### Better Hyperopt Results

- Use more epochs (200-500) for thorough search
- Run overnight for long optimizations
- Test multiple loss functions
- Validate results out-of-sample

### Avoid Overfitting

- Don't optimize on entire dataset
- Always validate on different period
- Keep some data for final validation
- Don't over-optimize (fewer epochs is sometimes better)

---

## Best Practices

✅ **DO**:
- Start with initial backtest before hyperopt
- Document all results
- Test on multiple periods
- Keep realistic expectations
- Validate out-of-sample

❌ **DON'T**:
- Skip initial backtest
- Optimize on test data
- Cherry-pick best results
- Rush to production
- Ignore warnings

---

## Next Steps

After successful validation:

1. **Document Results**: Fill in `docs/BACKTEST_RESULTS.md`
2. **Update Strategy**: Apply optimal parameters
3. **Dry-Run Testing**: Paper trading for 1-2 weeks
4. **Production**: Deploy with real funds (start small)

See: `docs/VALIDATION_GUIDE.md` for complete workflow

---

**For detailed validation workflow, see**: [VALIDATION_GUIDE.md](../docs/VALIDATION_GUIDE.md)
